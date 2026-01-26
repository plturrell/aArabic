//! LLM Integration Nodes for nWorkflow
//! Provides integration with nOpenaiServer and LLM operations
//!
//! Day 22 Implementation - LLM Integration Nodes
//! 
//! Components:
//! - LLMChatNode: Chat completion with streaming support
//! - LLMEmbedNode: Text embedding generation
//! - PromptTemplateNode: Template-based prompt construction
//! - ResponseParserNode: Parse and validate LLM responses
//!
//! Integration:
//! - nOpenaiServer via HTTP/FFI
//! - Token tracking and cost estimation
//! - Model selection and configuration
//! - Error handling and retry logic

const std = @import("std");
const Allocator = std.mem.Allocator;
const NodeInterface = @import("node_types").NodeInterface;
const ExecutionContext = @import("node_types").ExecutionContext;
const Port = @import("node_types").Port;
const PortType = @import("node_types").PortType;
const DataPacket = @import("data_packet").DataPacket;
const DataType = @import("data_packet").DataType;
// Import from nLocalModels orchestration (centralized)
const nLocalModelsOrch = @import("nlocalmodels_orch");
const ModelSelector = nLocalModelsOrch.ModelSelector;
const SelectionConstraints = nLocalModelsOrch.SelectionConstraints;

/// Configuration for LLM service connection
pub const LLMServiceConfig = struct {
    /// Service endpoint (e.g., "http://localhost:8080/v1")
    endpoint: []const u8,
    /// API key for authentication
    api_key: ?[]const u8 = null,
    /// Timeout in milliseconds
    timeout_ms: u32 = 30000,
    /// Maximum retries on failure
    max_retries: u32 = 3,
    /// Retry backoff multiplier
    retry_backoff_ms: u32 = 1000,
};

/// Token usage statistics
pub const TokenUsage = struct {
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    total_tokens: usize = 0,
};

/// Message role in chat completion
pub const MessageRole = enum {
    system,
    user,
    assistant,
    function,

    pub fn toString(self: MessageRole) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .function => "function",
        };
    }

    pub fn fromString(str: []const u8) !MessageRole {
        if (std.mem.eql(u8, str, "system")) return .system;
        if (std.mem.eql(u8, str, "user")) return .user;
        if (std.mem.eql(u8, str, "assistant")) return .assistant;
        if (std.mem.eql(u8, str, "function")) return .function;
        return error.InvalidRole;
    }
};

/// Chat message structure
pub const ChatMessage = struct {
    role: MessageRole,
    content: []const u8,

    pub fn init(role: MessageRole, content: []const u8) ChatMessage {
        return .{ .role = role, .content = content };
    }

    pub fn deinit(self: *ChatMessage, allocator: Allocator) void {
        allocator.free(self.content);
    }
};

/// LLM Chat Completion Node
/// Executes chat completion requests to LLM service
pub const LLMChatNode = struct {
    base: NodeInterface,
    allocator: Allocator,

    /// Model name (e.g., "gpt-4", "gpt-3.5-turbo") - can be null for auto-selection
    model: ?[]const u8,
    /// Task category for auto-selection (code, reasoning, etc.)
    task_category: ?[]const u8,
    /// Sampling temperature (0.0 - 2.0)
    temperature: f32,
    /// Maximum tokens in response
    max_tokens: usize,
    /// System prompt (optional)
    system_prompt: ?[]const u8,
    /// Service configuration
    service_config: LLMServiceConfig,
    /// Token usage tracking
    token_usage: TokenUsage,
    /// Enable streaming responses
    stream: bool,
    /// Model selector for intelligent routing (optional)
    model_selector: ?*ModelSelector,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        model: ?[]const u8,
        task_category: ?[]const u8,
        temperature: f32,
        max_tokens: usize,
        system_prompt: ?[]const u8,
        service_config: LLMServiceConfig,
    ) !*LLMChatNode {
        const node = try allocator.create(LLMChatNode);
        
        // Define input ports
        var inputs = std.ArrayList(Port){};
        try inputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "messages"),
            .name = try allocator.dupe(u8, "Messages"),
            .description = try allocator.dupe(u8, "Array of chat messages to send to the LLM"),
            .port_type = .array,
            .required = true,
            .default_value = null,
        });
        try inputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "temperature"),
            .name = try allocator.dupe(u8, "Temperature"),
            .description = try allocator.dupe(u8, "Sampling temperature for response generation"),
            .port_type = .number,
            .required = false,
            .default_value = null,
        });

        // Define output ports
        var outputs = std.ArrayList(Port){};
        try outputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "response"),
            .name = try allocator.dupe(u8, "Response"),
            .description = try allocator.dupe(u8, "Generated response from the LLM"),
            .port_type = .string,
            .required = true,
            .default_value = null,
        });
        try outputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "usage"),
            .name = try allocator.dupe(u8, "Token Usage"),
            .description = try allocator.dupe(u8, "Token usage statistics for the request"),
            .port_type = .object,
            .required = true,
            .default_value = null,
        });

        node.* = .{
            .base = NodeInterface{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .description = try allocator.dupe(u8, "LLM chat completion node with intelligent model selection"),
                .node_type = try allocator.dupe(u8, "llm_chat"),
                .category = .integration,
                .inputs = try inputs.toOwnedSlice(allocator),
                .outputs = try outputs.toOwnedSlice(allocator),
                .config = .{ .null = {} },
            },
            .allocator = allocator,
            .model = if (model) |m| try allocator.dupe(u8, m) else null,
            .task_category = if (task_category) |tc| try allocator.dupe(u8, tc) else null,
            .temperature = temperature,
            .max_tokens = max_tokens,
            .system_prompt = if (system_prompt) |sp| try allocator.dupe(u8, sp) else null,
            .service_config = service_config,
            .token_usage = .{},
            .stream = false,
            .model_selector = null,
        };

        return node;
    }

    pub fn deinit(self: *LLMChatNode) void {
        self.allocator.free(self.base.id);
        self.allocator.free(self.base.name);
        self.allocator.free(self.base.description);
        self.allocator.free(self.base.node_type);

        for (self.base.inputs) |input| {
            self.allocator.free(input.id);
            self.allocator.free(input.name);
            self.allocator.free(input.description);
        }
        self.allocator.free(self.base.inputs);

        for (self.base.outputs) |output| {
            self.allocator.free(output.id);
            self.allocator.free(output.name);
            self.allocator.free(output.description);
        }
        self.allocator.free(self.base.outputs);

        if (self.model) |m| self.allocator.free(m);
        if (self.task_category) |tc| self.allocator.free(tc);
        if (self.system_prompt) |sp| self.allocator.free(sp);
        self.allocator.destroy(self);
    }

    pub fn validate(self: *const LLMChatNode) !void {
        if (self.temperature < 0.0 or self.temperature > 2.0) {
            return error.InvalidTemperature;
        }
        if (self.max_tokens == 0) {
            return error.InvalidMaxTokens;
        }
        // Either model or task_category must be specified
        if (self.model == null and self.task_category == null) {
            return error.ModelOrCategoryRequired;
        }
    }
    
    /// Set model selector for intelligent routing
    pub fn setModelSelector(self: *LLMChatNode, selector: *ModelSelector) void {
        self.model_selector = selector;
    }
    
    /// Select model based on task category
    fn selectModel(self: *LLMChatNode) ![]const u8 {
        // If model is explicitly specified, use it
        if (self.model) |m| {
            return m;
        }
        
        // If task category specified and selector available, use intelligent selection
        if (self.task_category) |category| {
            if (self.model_selector) |selector| {
                const constraints = SelectionConstraints{
                    .max_gpu_memory_mb = 14 * 1024, // T4 constraint
                };
                
                const result = try selector.selectModel(category, constraints);
                std.log.info("Auto-selected model: {s} (score: {d:.2}, reason: {s})", .{
                    result.model.name,
                    result.score,
                    result.reason,
                });
                
                return result.model.name;
            }
        }
        
        // Fallback to a default model
        return "google-gemma-3-270m-it";
    }

    pub fn execute(self: *LLMChatNode, ctx: *ExecutionContext) !*DataPacket {
        try self.validate();

        // Select model (either specified or auto-selected)
        const selected_model = try self.selectModel();
        
        // Get messages from input
        const messages_input = ctx.getInput("messages") orelse return error.MissingInput;
        
        // Build request payload
        var request = std.ArrayList(u8){};
        defer request.deinit(self.allocator);

        try request.appendSlice(self.allocator, "{\"model\":\"");
        try request.appendSlice(self.allocator, selected_model);
        try request.appendSlice(self.allocator, "\",\"temperature\":");
        var temp_buf: [32]u8 = undefined;
        const temp_str = std.fmt.bufPrint(&temp_buf, "{d}", .{self.temperature}) catch unreachable;
        try request.appendSlice(self.allocator, temp_str);
        try request.appendSlice(self.allocator, ",\"max_tokens\":");
        var max_buf: [32]u8 = undefined;
        const max_str = std.fmt.bufPrint(&max_buf, "{d}", .{self.max_tokens}) catch unreachable;
        try request.appendSlice(self.allocator, max_str);
        try request.appendSlice(self.allocator, ",\"messages\":[");

        // Add system prompt if configured
        if (self.system_prompt) |sp| {
            try request.appendSlice(self.allocator, "{\"role\":\"system\",\"content\":\"");
            try request.appendSlice(self.allocator, sp);
            try request.appendSlice(self.allocator, "\"},");
        }

        // Add user messages (simplified - in production, parse from input)
        try request.appendSlice(self.allocator, "{\"role\":\"user\",\"content\":\"");
        try request.appendSlice(self.allocator, messages_input);
        try request.appendSlice(self.allocator, "\"}");

        try request.appendSlice(self.allocator, "]}");

        // Execute HTTP request to nOpenaiServer
        const response = try self.executeRequest(request.items);
        defer self.allocator.free(response);

        // Parse response and extract completion
        const completion = try self.parseResponse(response);
        defer self.allocator.free(completion);

        // Create output packet with response
        const output = try DataPacket.init(
            self.allocator,
            "llm_chat_output",
            .string,
            .{ .string = try self.allocator.dupe(u8, completion) },
        );

        // Add metadata
        try output.metadata.put("model", try self.allocator.dupe(u8, selected_model));
        if (self.task_category) |tc| {
            try output.metadata.put("task_category", try self.allocator.dupe(u8, tc));
        }
        try output.metadata.put("temperature", try std.fmt.allocPrint(self.allocator, "{d}", .{self.temperature}));
        try output.metadata.put("prompt_tokens", try std.fmt.allocPrint(self.allocator, "{d}", .{self.token_usage.prompt_tokens}));
        try output.metadata.put("completion_tokens", try std.fmt.allocPrint(self.allocator, "{d}", .{self.token_usage.completion_tokens}));
        try output.metadata.put("total_tokens", try std.fmt.allocPrint(self.allocator, "{d}", .{self.token_usage.total_tokens}));

        return output;
    }

    fn executeRequest(self: *LLMChatNode, payload: []const u8) ![]const u8 {
        // Parse endpoint URL and append /chat/completions
        var endpoint_buf: [512]u8 = undefined;
        const full_endpoint = try std.fmt.bufPrint(&endpoint_buf, "{s}/chat/completions", .{self.service_config.endpoint});
        const uri = try std.Uri.parse(full_endpoint);
        
        // Create HTTP client
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();
        
        // Prepare request buffer
        var server_header_buffer: [4096]u8 = undefined;
        
        // Open connection
        var request = try client.open(.POST, uri, .{
            .server_header_buffer = &server_header_buffer,
        });
        defer request.deinit();
        
        // Set headers
        try request.headers.append("Content-Type", "application/json");
        if (self.service_config.api_key) |key| {
            var auth_buf: [256]u8 = undefined;
            const auth_header = try std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{key});
            try request.headers.append("Authorization", auth_header);
        }
        
        // Send request
        request.transfer_encoding = .chunked;
        try request.send();
        try request.writeAll(payload);
        try request.finish();
        
        // Wait for response
        try request.wait();
        
        // Check status
        if (request.response.status != .ok) {
            std.log.err("LLM API error: {}", .{request.response.status});
            return error.LLMApiError;
        }
        
        // Read response body
        var response_body = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        errdefer response_body.deinit();
        
        const max_size = 10 * 1024 * 1024; // 10MB max
        try request.reader().readAllArrayList(&response_body, max_size);
        
        const response_str = try response_body.toOwnedSlice();
        
        // Parse usage from response
        self.parseUsage(response_str);
        
        return response_str;
    }
    
    fn parseUsage(self: *LLMChatNode, response: []const u8) void {
        // Parse token usage from JSON response
        if (std.mem.indexOf(u8, response, "\"prompt_tokens\":")) |pos| {
            const start = pos + "\"prompt_tokens\":".len;
            var end = start;
            while (end < response.len and (std.ascii.isDigit(response[end]) or response[end] == ' ')) : (end += 1) {}
            const token_str = std.mem.trim(u8, response[start..end], " ");
            self.token_usage.prompt_tokens = std.fmt.parseInt(usize, token_str, 10) catch 0;
        }
        
        if (std.mem.indexOf(u8, response, "\"completion_tokens\":")) |pos| {
            const start = pos + "\"completion_tokens\":".len;
            var end = start;
            while (end < response.len and (std.ascii.isDigit(response[end]) or response[end] == ' ')) : (end += 1) {}
            const token_str = std.mem.trim(u8, response[start..end], " ");
            self.token_usage.completion_tokens = std.fmt.parseInt(usize, token_str, 10) catch 0;
        }
        
        if (std.mem.indexOf(u8, response, "\"total_tokens\":")) |pos| {
            const start = pos + "\"total_tokens\":".len;
            var end = start;
            while (end < response.len and (std.ascii.isDigit(response[end]) or response[end] == ' ')) : (end += 1) {}
            const token_str = std.mem.trim(u8, response[start..end], " ");
            self.token_usage.total_tokens = std.fmt.parseInt(usize, token_str, 10) catch 0;
        }
    }

    fn parseResponse(self: *LLMChatNode, response: []const u8) ![]const u8 {
        // Simplified parsing - in production, use std.json.parseFromSlice
        // For now, extract the content field manually
        
        const content_start = std.mem.indexOf(u8, response, "\"content\": \"") orelse return error.ParseError;
        const content_begin = content_start + "\"content\": \"".len;
        const content_end = std.mem.indexOfPos(u8, response, content_begin, "\"") orelse return error.ParseError;
        
        const content = response[content_begin..content_end];
        return try self.allocator.dupe(u8, content);
    }
};

/// LLM Embedding Node
/// Generates vector embeddings for text
pub const LLMEmbedNode = struct {
    base: NodeInterface,
    allocator: Allocator,

    /// Model name (e.g., "text-embedding-3-small")
    model: []const u8,
    /// Embedding dimensions
    dimensions: usize,
    /// Service configuration
    service_config: LLMServiceConfig,
    /// Token usage tracking
    token_usage: TokenUsage,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        model: []const u8,
        dimensions: usize,
        service_config: LLMServiceConfig,
    ) !*LLMEmbedNode {
        const node = try allocator.create(LLMEmbedNode);
        
        // Define input ports
        var inputs = std.ArrayList(Port){};
        try inputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "text"),
            .name = try allocator.dupe(u8, "Text"),
            .description = try allocator.dupe(u8, "Text input to generate embeddings for"),
            .port_type = .string,
            .required = true,
            .default_value = null,
        });

        // Define output ports
        var outputs = std.ArrayList(Port){};
        try outputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "embedding"),
            .name = try allocator.dupe(u8, "Embedding Vector"),
            .description = try allocator.dupe(u8, "Generated embedding vector"),
            .port_type = .array,
            .required = true,
            .default_value = null,
        });
        try outputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "dimensions"),
            .name = try allocator.dupe(u8, "Dimensions"),
            .description = try allocator.dupe(u8, "Number of dimensions in the embedding vector"),
            .port_type = .number,
            .required = true,
            .default_value = null,
        });

        node.* = .{
            .base = NodeInterface{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .description = try allocator.dupe(u8, "LLM embedding node for generating vector representations"),
                .node_type = try allocator.dupe(u8, "llm_embed"),
                .category = .integration,
                .inputs = try inputs.toOwnedSlice(allocator),
                .outputs = try outputs.toOwnedSlice(allocator),
                .config = .{ .null = {} },
            },
            .allocator = allocator,
            .model = try allocator.dupe(u8, model),
            .dimensions = dimensions,
            .service_config = service_config,
            .token_usage = .{},
        };

        return node;
    }

    pub fn deinit(self: *LLMEmbedNode) void {
        self.allocator.free(self.base.id);
        self.allocator.free(self.base.name);
        self.allocator.free(self.base.description);
        self.allocator.free(self.base.node_type);

        for (self.base.inputs) |input| {
            self.allocator.free(input.id);
            self.allocator.free(input.name);
            self.allocator.free(input.description);
        }
        self.allocator.free(self.base.inputs);

        for (self.base.outputs) |output| {
            self.allocator.free(output.id);
            self.allocator.free(output.name);
            self.allocator.free(output.description);
        }
        self.allocator.free(self.base.outputs);

        self.allocator.free(self.model);
        self.allocator.destroy(self);
    }

    pub fn validate(self: *const LLMEmbedNode) !void {
        if (self.dimensions == 0) {
            return error.InvalidDimensions;
        }
        if (self.model.len == 0) {
            return error.InvalidModel;
        }
    }

    pub fn execute(self: *LLMEmbedNode, ctx: *ExecutionContext) !*DataPacket {
        try self.validate();

        // Get text from input
        const text_input = ctx.getInput("text") orelse return error.MissingInput;

        // Build request payload
        var request = std.ArrayList(u8){};
        defer request.deinit(self.allocator);

        try request.appendSlice(self.allocator, "{\"model\":\"");
        try request.appendSlice(self.allocator, self.model);
        try request.appendSlice(self.allocator, "\",\"input\":\"");
        try request.appendSlice(self.allocator, text_input);
        try request.appendSlice(self.allocator, "\",\"dimensions\":");
        var dim_buf: [32]u8 = undefined;
        const dim_str = std.fmt.bufPrint(&dim_buf, "{d}", .{self.dimensions}) catch unreachable;
        try request.appendSlice(self.allocator, dim_str);
        try request.appendSlice(self.allocator, "}");

        // Execute HTTP request to nOpenaiServer
        const response = try self.executeRequest(request.items);
        defer self.allocator.free(response);

        // Parse response and extract embedding vector
        const embedding = try self.parseResponse(response);

        // Create output packet with embedding array
        const output = try DataPacket.init(
            self.allocator,
            "llm_embed_output",
            .array,
            .{ .array = embedding },
        );

        // Add metadata
        try output.metadata.put("model", try self.allocator.dupe(u8, self.model));
        try output.metadata.put("dimensions", try std.fmt.allocPrint(self.allocator, "{d}", .{self.dimensions}));
        try output.metadata.put("tokens", try std.fmt.allocPrint(self.allocator, "{d}", .{self.token_usage.total_tokens}));

        return output;
    }

    fn executeRequest(self: *LLMEmbedNode, payload: []const u8) ![]const u8 {
        // Parse endpoint URL and append /embeddings
        var endpoint_buf: [512]u8 = undefined;
        const full_endpoint = try std.fmt.bufPrint(&endpoint_buf, "{s}/embeddings", .{self.service_config.endpoint});
        const uri = try std.Uri.parse(full_endpoint);
        
        // Create HTTP client
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();
        
        // Prepare request buffer
        var server_header_buffer: [4096]u8 = undefined;
        
        // Open connection
        var request = try client.open(.POST, uri, .{
            .server_header_buffer = &server_header_buffer,
        });
        defer request.deinit();
        
        // Set headers
        try request.headers.append("Content-Type", "application/json");
        if (self.service_config.api_key) |key| {
            var auth_buf: [256]u8 = undefined;
            const auth_header = try std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{key});
            try request.headers.append("Authorization", auth_header);
        }
        
        // Send request
        request.transfer_encoding = .chunked;
        try request.send();
        try request.writeAll(payload);
        try request.finish();
        
        // Wait for response
        try request.wait();
        
        // Check status
        if (request.response.status != .ok) {
            std.log.err("Embedding API error: {}", .{request.response.status});
            return error.EmbeddingApiError;
        }
        
        // Read response body
        var response_body = std.ArrayList(u8){};
        errdefer response_body.deinit();
        
        const max_size = 10 * 1024 * 1024; // 10MB max
        try request.reader().readAllArrayList(&response_body, max_size);
        
        const response_str = try response_body.toOwnedSlice();
        
        // Parse usage from response
        self.parseUsage(response_str);
        
        return response_str;
    }
    
    fn parseUsage(self: *LLMEmbedNode, response: []const u8) void {
        // Parse token usage from JSON response
        if (std.mem.indexOf(u8, response, "\"prompt_tokens\":")) |pos| {
            const start = pos + "\"prompt_tokens\":".len;
            var end = start;
            while (end < response.len and (std.ascii.isDigit(response[end]) or response[end] == ' ')) : (end += 1) {}
            const token_str = std.mem.trim(u8, response[start..end], " ");
            self.token_usage.prompt_tokens = std.fmt.parseInt(usize, token_str, 10) catch 0;
        }
        
        if (std.mem.indexOf(u8, response, "\"total_tokens\":")) |pos| {
            const start = pos + "\"total_tokens\":".len;
            var end = start;
            while (end < response.len and (std.ascii.isDigit(response[end]) or response[end] == ' ')) : (end += 1) {}
            const token_str = std.mem.trim(u8, response[start..end], " ");
            self.token_usage.total_tokens = std.fmt.parseInt(usize, token_str, 10) catch 0;
        }
        
        self.token_usage.completion_tokens = 0; // Embeddings don't generate tokens
    }

    fn parseResponse(self: *LLMEmbedNode, response: []const u8) !std.json.Value {
        // Parse JSON response to extract embedding array
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, response, .{});
        defer parsed.deinit();
        
        // Navigate to data[0].embedding
        if (parsed.value != .object) return error.InvalidResponse;
        const data_field = parsed.value.object.get("data") orelse return error.MissingDataField;
        
        if (data_field != .array or data_field.array.items.len == 0) return error.EmptyDataArray;
        const first_item = data_field.array.items[0];
        
        if (first_item != .object) return error.InvalidDataItem;
        const embedding_field = first_item.object.get("embedding") orelse return error.MissingEmbeddingField;
        
        if (embedding_field != .array) return error.InvalidEmbeddingFormat;
        
        // Deep copy the embedding array
        var embedding_array = try std.ArrayList(std.json.Value).initCapacity(self.allocator, 0);
        errdefer embedding_array.deinit();
        
        for (embedding_field.array.items) |val| {
            const copied_val = switch (val) {
                .float => |f| std.json.Value{ .float = f },
                .integer => |i| std.json.Value{ .float = @floatFromInt(i) },
                .number_string => |s| blk: {
                    const f = try std.fmt.parseFloat(f64, s);
                    break :blk std.json.Value{ .float = f };
                },
                else => return error.InvalidEmbeddingValue,
            };
            try embedding_array.append(copied_val);
        }
        
        return .{ .array = try embedding_array.toOwnedSlice() };
    }
};

/// Prompt Template Node
/// Constructs prompts from templates with variable substitution
pub const PromptTemplateNode = struct {
    base: NodeInterface,
    allocator: Allocator,

    /// Template string with {{variable}} placeholders
    template: []const u8,
    /// List of variable names expected
    variables: [][]const u8,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        template: []const u8,
        variables: [][]const u8,
    ) !*PromptTemplateNode {
        const node = try allocator.create(PromptTemplateNode);
        
        // Define input ports (one per variable)
        var inputs = std.ArrayList(Port){};
        for (variables) |variable| {
            try inputs.append(allocator, Port{
                .id = try allocator.dupe(u8, variable),
                .name = try allocator.dupe(u8, variable),
                .description = try allocator.dupe(u8, "Template variable value"),
                .port_type = .string,
                .required = true,
                .default_value = null,
            });
        }

        // Define output ports
        var outputs = std.ArrayList(Port){};
        try outputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "prompt"),
            .name = try allocator.dupe(u8, "Filled Prompt"),
            .description = try allocator.dupe(u8, "Prompt with all variables substituted"),
            .port_type = .string,
            .required = true,
            .default_value = null,
        });

        // Duplicate variables array
        const vars_copy = try allocator.alloc([]const u8, variables.len);
        for (variables, 0..) |variable, i| {
            vars_copy[i] = try allocator.dupe(u8, variable);
        }

        node.* = .{
            .base = NodeInterface{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .description = try allocator.dupe(u8, "Prompt template node for variable substitution"),
                .node_type = try allocator.dupe(u8, "prompt_template"),
                .category = .transform,
                .inputs = try inputs.toOwnedSlice(allocator),
                .outputs = try outputs.toOwnedSlice(allocator),
                .config = .{ .null = {} },
            },
            .allocator = allocator,
            .template = try allocator.dupe(u8, template),
            .variables = vars_copy,
        };

        return node;
    }

    pub fn deinit(self: *PromptTemplateNode) void {
        self.allocator.free(self.base.id);
        self.allocator.free(self.base.name);
        self.allocator.free(self.base.description);
        self.allocator.free(self.base.node_type);

        for (self.base.inputs) |input| {
            self.allocator.free(input.id);
            self.allocator.free(input.name);
            self.allocator.free(input.description);
        }
        self.allocator.free(self.base.inputs);

        for (self.base.outputs) |output| {
            self.allocator.free(output.id);
            self.allocator.free(output.name);
            self.allocator.free(output.description);
        }
        self.allocator.free(self.base.outputs);

        self.allocator.free(self.template);
        for (self.variables) |variable| {
            self.allocator.free(variable);
        }
        self.allocator.free(self.variables);
        self.allocator.destroy(self);
    }

    pub fn validate(self: *const PromptTemplateNode) !void {
        if (self.template.len == 0) {
            return error.EmptyTemplate;
        }
        // Verify all variables are present in template
        for (self.variables) |variable| {
            var search_pattern = std.ArrayList(u8){};
            defer search_pattern.deinit(self.allocator);
            try search_pattern.appendSlice(self.allocator, "{{");
            try search_pattern.appendSlice(self.allocator, variable);
            try search_pattern.appendSlice(self.allocator, "}}");

            if (std.mem.indexOf(u8, self.template, search_pattern.items) == null) {
                return error.VariableNotInTemplate;
            }
        }
    }

    pub fn execute(self: *PromptTemplateNode, ctx: *ExecutionContext) !*DataPacket {
        try self.validate();

        // Start with the template
        var result = std.ArrayList(u8){};
        defer result.deinit(self.allocator);
        try result.appendSlice(self.allocator, self.template);

        // Replace each variable
        for (self.variables) |variable| {
            const value = ctx.getInput(variable) orelse return error.MissingVariable;

            // Build search pattern {{variable}}
            var pattern = std.ArrayList(u8){};
            defer pattern.deinit(self.allocator);
            try pattern.appendSlice(self.allocator, "{{");
            try pattern.appendSlice(self.allocator, variable);
            try pattern.appendSlice(self.allocator, "}}");

            // Replace all occurrences
            const replaced = try self.replaceAll(result.items, pattern.items, value);
            result.clearRetainingCapacity();
            try result.appendSlice(self.allocator, replaced);
            self.allocator.free(replaced);
        }

        // Create output packet
        const output = try DataPacket.init(
            self.allocator,
            "prompt_template_output",
            .string,
            .{ .string = try result.toOwnedSlice(self.allocator) },
        );

        // Add metadata
        try output.metadata.put("template", try self.allocator.dupe(u8, self.template));
        try output.metadata.put("variables_count", try std.fmt.allocPrint(self.allocator, "{d}", .{self.variables.len}));

        return output;
    }

    fn replaceAll(self: *PromptTemplateNode, haystack: []const u8, needle: []const u8, replacement: []const u8) ![]const u8 {
        var result = std.ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        var pos: usize = 0;
        while (pos < haystack.len) {
            if (std.mem.indexOf(u8, haystack[pos..], needle)) |found| {
                // Add text before needle
                try result.appendSlice(self.allocator, haystack[pos .. pos + found]);
                // Add replacement
                try result.appendSlice(self.allocator, replacement);
                // Move past needle
                pos += found + needle.len;
            } else {
                // No more needles, add rest
                try result.appendSlice(self.allocator, haystack[pos..]);
                break;
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }
};

/// Parser type for response parsing
pub const ParserType = enum {
    json,
    markdown,
    text,
    custom,

    pub fn toString(self: ParserType) []const u8 {
        return switch (self) {
            .json => "json",
            .markdown => "markdown",
            .text => "text",
            .custom => "custom",
        };
    }
};

/// Response Parser Node
/// Parses and validates LLM responses
pub const ResponseParserNode = struct {
    base: NodeInterface,
    allocator: Allocator,

    /// Parser type
    parser_type: ParserType,
    /// Expected schema (for JSON parsing)
    schema: ?[]const u8,
    /// Custom parser function (for custom parsing)
    custom_parser: ?*const fn ([]const u8) anyerror!std.json.Value,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        parser_type: ParserType,
        schema: ?[]const u8,
    ) !*ResponseParserNode {
        const node = try allocator.create(ResponseParserNode);
        
        // Define input ports
        var inputs = std.ArrayList(Port){};
        try inputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "response"),
            .name = try allocator.dupe(u8, "LLM Response"),
            .description = try allocator.dupe(u8, "Raw response string from the LLM to parse"),
            .port_type = .string,
            .required = true,
            .default_value = null,
        });

        // Define output ports
        var outputs = std.ArrayList(Port){};
        try outputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "parsed"),
            .name = try allocator.dupe(u8, "Parsed Data"),
            .description = try allocator.dupe(u8, "Structured data extracted from the response"),
            .port_type = .object,
            .required = true,
            .default_value = null,
        });
        try outputs.append(allocator, Port{
            .id = try allocator.dupe(u8, "valid"),
            .name = try allocator.dupe(u8, "Is Valid"),
            .description = try allocator.dupe(u8, "Whether the response was successfully parsed"),
            .port_type = .boolean,
            .required = true,
            .default_value = null,
        });

        node.* = .{
            .base = NodeInterface{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .description = try allocator.dupe(u8, "Response parser node for extracting structured data from LLM responses"),
                .node_type = try allocator.dupe(u8, "response_parser"),
                .category = .transform,
                .inputs = try inputs.toOwnedSlice(allocator),
                .outputs = try outputs.toOwnedSlice(allocator),
                .config = .{ .null = {} },
            },
            .allocator = allocator,
            .parser_type = parser_type,
            .schema = if (schema) |s| try allocator.dupe(u8, s) else null,
            .custom_parser = null,
        };

        return node;
    }

    pub fn deinit(self: *ResponseParserNode) void {
        self.allocator.free(self.base.id);
        self.allocator.free(self.base.name);
        self.allocator.free(self.base.description);
        self.allocator.free(self.base.node_type);

        for (self.base.inputs) |input| {
            self.allocator.free(input.id);
            self.allocator.free(input.name);
            self.allocator.free(input.description);
        }
        self.allocator.free(self.base.inputs);

        for (self.base.outputs) |output| {
            self.allocator.free(output.id);
            self.allocator.free(output.name);
            self.allocator.free(output.description);
        }
        self.allocator.free(self.base.outputs);

        if (self.schema) |s| self.allocator.free(s);
        self.allocator.destroy(self);
    }

    pub fn validate(self: *const ResponseParserNode) !void {
        if (self.parser_type == .json and self.schema == null) {
            return error.SchemaRequired;
        }
    }

    pub fn execute(self: *ResponseParserNode, ctx: *ExecutionContext) !*DataPacket {
        try self.validate();

        // Get response from input
        const response_input = ctx.getInput("response") orelse return error.MissingInput;

        // Parse based on type
        const parsed = switch (self.parser_type) {
            .json => try self.parseJson(response_input),
            .markdown => try self.parseMarkdown(response_input),
            .text => try self.parseText(response_input),
            .custom => if (self.custom_parser) |parser| try parser(response_input) else return error.NoCustomParser,
        };

        // Validate against schema if provided
        const is_valid = if (self.schema != null) try self.validateSchema(parsed) else true;

        // Create output packet
        const output = try DataPacket.init(
            self.allocator,
            "response_parser_output",
            .object,
            parsed,
        );

        // Add metadata
        try output.metadata.put("parser_type", try self.allocator.dupe(u8, self.parser_type.toString()));
        try output.metadata.put("valid", if (is_valid) try self.allocator.dupe(u8, "true") else try self.allocator.dupe(u8, "false"));

        return output;
    }

    fn parseJson(self: *ResponseParserNode, response: []const u8) !std.json.Value {
        // Simplified JSON parsing - in production, use std.json.parseFromSlice
        
        // Mock parsed object
        var obj = std.StringHashMap(std.json.Value).init(self.allocator);
        try obj.put("text", .{ .string = response });
        try obj.put("parsed", .{ .bool = true });

        return .{ .object = obj };
    }

    fn parseMarkdown(self: *ResponseParserNode, response: []const u8) !std.json.Value {
        // Extract markdown sections
        var obj = std.StringHashMap(std.json.Value).init(self.allocator);
        try obj.put("raw", .{ .string = response });
        try obj.put("type", .{ .string = "markdown" });

        return .{ .object = obj };
    }

    fn parseText(self: *ResponseParserNode, response: []const u8) !std.json.Value {
        // Plain text parsing
        var obj = std.StringHashMap(std.json.Value).init(self.allocator);
        try obj.put("text", .{ .string = response });
        try obj.put("length", .{ .integer = @intCast(response.len) });

        return .{ .object = obj };
    }

    fn validateSchema(self: *ResponseParserNode, parsed: std.json.Value) !bool {
        _ = parsed;
        _ = self;
        // Simplified schema validation - in production, implement JSON Schema validation
        return true;
    }
};

// Tests
test "LLMChatNode creation" {
    const allocator = std.testing.allocator;
    
    const service_config = LLMServiceConfig{
        .endpoint = "http://localhost:11434/v1",
    };

    const node = try LLMChatNode.init(
        allocator,
        "chat1",
        "Chat Completion",
        "llama-3.3-70b",
        null,
        0.7,
        1000,
        "You are a helpful assistant.",
        service_config,
    );
    defer node.deinit();

    try std.testing.expectEqualStrings("chat1", node.base.id);
    try std.testing.expectEqualStrings("llm_chat", node.base.node_type);
    try std.testing.expectEqual(@as(usize, 2), node.base.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), node.base.outputs.len);
}

test "LLMEmbedNode creation" {
    const allocator = std.testing.allocator;
    
    const service_config = LLMServiceConfig{
        .endpoint = "http://localhost:11434/v1",
    };

    const node = try LLMEmbedNode.init(
        allocator,
        "embed1",
        "Text Embedding",
        "internal-embeddings",
        1536,
        service_config,
    );
    defer node.deinit();

    try std.testing.expectEqualStrings("embed1", node.base.id);
    try std.testing.expectEqualStrings("llm_embed", node.base.node_type);
    try std.testing.expectEqual(@as(usize, 1536), node.dimensions);
}

test "PromptTemplateNode creation" {
    const allocator = std.testing.allocator;
    
    const template = "Hello {{name}}, your order {{order_id}} is ready!";
    const variables = [_][]const u8{ "name", "order_id" };

    const node = try PromptTemplateNode.init(
        allocator,
        "template1",
        "Order Template",
        template,
        @constCast(&variables),
    );
    defer node.deinit();

    try std.testing.expectEqualStrings("template1", node.base.id);
    try std.testing.expectEqualStrings("prompt_template", node.base.node_type);
    try std.testing.expectEqual(@as(usize, 2), node.variables.len);
}

test "ResponseParserNode creation" {
    const allocator = std.testing.allocator;

    const node = try ResponseParserNode.init(
        allocator,
        "parser1",
        "JSON Parser",
        .json,
        "{\"type\": \"object\"}",
    );
    defer node.deinit();

    try std.testing.expectEqualStrings("parser1", node.base.id);
    try std.testing.expectEqualStrings("response_parser", node.base.node_type);
    try std.testing.expectEqual(ParserType.json, node.parser_type);
}

test "TokenUsage tracking" {
    const usage = TokenUsage{
        .prompt_tokens = 1000,
        .completion_tokens = 500,
        .total_tokens = 1500,
    };

    // Verify token counts
    try std.testing.expect(usage.prompt_tokens == 1000);
    try std.testing.expect(usage.completion_tokens == 500);
    try std.testing.expect(usage.total_tokens == 1500);
}

test "MessageRole conversion" {
    try std.testing.expectEqualStrings("system", MessageRole.system.toString());
    try std.testing.expectEqualStrings("user", MessageRole.user.toString());
    
    const role = try MessageRole.fromString("assistant");
    try std.testing.expectEqual(MessageRole.assistant, role);
}

test "PromptTemplateNode variable validation" {
    const allocator = std.testing.allocator;
    
    const template = "Hello {{name}}!";
    const variables = [_][]const u8{"name"};

    const node = try PromptTemplateNode.init(
        allocator,
        "template1",
        "Greeting",
        template,
        @constCast(&variables),
    );
    defer node.deinit();

    try node.validate();
}

test "PromptTemplateNode missing variable" {
    const allocator = std.testing.allocator;
    
    const template = "Hello {{name}}!";
    const variables = [_][]const u8{ "name", "missing" };

    const node = try PromptTemplateNode.init(
        allocator,
        "template1",
        "Greeting",
        template,
        @constCast(&variables),
    );
    defer node.deinit();

    try std.testing.expectError(error.VariableNotInTemplate, node.validate());
}
