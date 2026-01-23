//! Advanced LLM Integration for nWorkflow
//! 
//! Provides advanced LLM features including:
//! - Function calling support
//! - Conversational memory management
//! - Context window optimization
//! - Token budget management
//! - Streaming responses
//! - Multi-turn conversations
//!
//! Day 24: Advanced LLM Features

const std = @import("std");
const Allocator = std.mem.Allocator;
const node_types = @import("node_types");
const data_packet = @import("data_packet");

const NodeInterface = node_types.NodeInterface;
const Port = node_types.Port;
const PortType = node_types.PortType;
const ExecutionContext = node_types.ExecutionContext;
const DataPacket = data_packet.DataPacket;

/// Function definition for LLM function calling
pub const FunctionDefinition = struct {
    name: []const u8,
    description: []const u8,
    parameters: std.json.Value, // JSON Schema
    
    pub fn init(
        allocator: Allocator,
        name: []const u8,
        description: []const u8,
        parameters: std.json.Value,
    ) !FunctionDefinition {
        return FunctionDefinition{
            .name = try allocator.dupe(u8, name),
            .description = try allocator.dupe(u8, description),
            .parameters = parameters,
        };
    }
    
    pub fn deinit(self: *FunctionDefinition, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
    }
};

/// Function call result from LLM
pub const FunctionCall = struct {
    name: []const u8,
    arguments: std.json.Value,
    call_id: []const u8,
    
    pub fn deinit(self: *FunctionCall, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.call_id);
    }
};

/// Conversational message with role
pub const ConversationMessage = struct {
    role: MessageRole,
    content: []const u8,
    function_call: ?FunctionCall,
    timestamp: i64,
    token_count: usize,
    
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
    };
    
    pub fn init(
        allocator: Allocator,
        role: MessageRole,
        content: []const u8,
    ) !ConversationMessage {
        return ConversationMessage{
            .role = role,
            .content = try allocator.dupe(u8, content),
            .function_call = null,
            .timestamp = std.time.milliTimestamp(),
            .token_count = estimateTokenCount(content),
        };
    }
    
    pub fn deinit(self: *ConversationMessage, allocator: Allocator) void {
        allocator.free(self.content);
        if (self.function_call) |*fc| {
            fc.deinit(allocator);
        }
    }
    
    fn estimateTokenCount(content: []const u8) usize {
        // Rough estimation: ~4 characters per token
        return (content.len + 3) / 4;
    }
};

/// Conversation history manager
pub const ConversationHistory = struct {
    allocator: Allocator,
    messages: std.ArrayList(ConversationMessage),
    max_messages: usize,
    max_tokens: usize,
    current_tokens: usize,
    
    pub fn init(allocator: Allocator, max_messages: usize, max_tokens: usize) ConversationHistory {
        return ConversationHistory{
            .allocator = allocator,
            .messages = .{},
            .max_messages = max_messages,
            .max_tokens = max_tokens,
            .current_tokens = 0,
        };
    }

    pub fn deinit(self: *ConversationHistory) void {
        for (self.messages.items) |*msg| {
            msg.deinit(self.allocator);
        }
        self.messages.deinit(self.allocator);
    }
    
    /// Add message to history with automatic pruning
    pub fn addMessage(self: *ConversationHistory, message: ConversationMessage) !void {
        // Add new message
        try self.messages.append(self.allocator, message);
        self.current_tokens += message.token_count;
        
        // Prune if necessary
        try self.pruneIfNeeded();
    }
    
    /// Prune old messages to stay within limits
    fn pruneIfNeeded(self: *ConversationHistory) !void {
        // Keep system messages, prune oldest user/assistant messages
        while (self.messages.items.len > self.max_messages or 
               self.current_tokens > self.max_tokens) {
            // Find oldest non-system message
            var oldest_idx: ?usize = null;
            for (self.messages.items, 0..) |msg, i| {
                if (msg.role != .system) {
                    oldest_idx = i;
                    break;
                }
            }
            
            if (oldest_idx) |idx| {
                var removed = self.messages.orderedRemove(idx);
                self.current_tokens -= removed.token_count;
                removed.deinit(self.allocator);
            } else {
                // All messages are system messages, stop pruning
                break;
            }
        }
    }
    
    /// Get all messages for API call
    pub fn getMessages(self: *const ConversationHistory) []const ConversationMessage {
        return self.messages.items;
    }
    
    /// Clear all messages except system messages
    pub fn clear(self: *ConversationHistory) void {
        var i: usize = 0;
        while (i < self.messages.items.len) {
            if (self.messages.items[i].role != .system) {
                var removed = self.messages.orderedRemove(i);
                self.current_tokens -= removed.token_count;
                removed.deinit(self.allocator);
            } else {
                i += 1;
            }
        }
    }
    
    /// Get current token count
    pub fn getTokenCount(self: *const ConversationHistory) usize {
        return self.current_tokens;
    }
};

/// LLM Function Calling Node
pub const LLMFunctionNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    model: []const u8,
    temperature: f32,
    functions: std.ArrayList(FunctionDefinition),
    function_choice: FunctionChoice,
    
    pub const FunctionChoice = enum {
        auto,      // Let model decide
        none,      // No function calls
        required,  // Must call a function
        
        pub fn toString(self: FunctionChoice) []const u8 {
            return switch (self) {
                .auto => "auto",
                .none => "none",
                .required => "required",
            };
        }
    };
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        model: []const u8,
        temperature: f32,
    ) !LLMFunctionNode {
        return LLMFunctionNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .model = try allocator.dupe(u8, model),
            .temperature = temperature,
            .functions = .{},
            .function_choice = .auto,
        };
    }

    pub fn deinit(self: *LLMFunctionNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.model);

        for (self.functions.items) |*func| {
            func.deinit(self.allocator);
        }
        self.functions.deinit(self.allocator);
    }

    /// Add function definition
    pub fn addFunction(self: *LLMFunctionNode, func: FunctionDefinition) !void {
        try self.functions.append(self.allocator, func);
    }

    /// Execute with function calling
    pub fn execute(self: *LLMFunctionNode, ctx: *ExecutionContext) !*DataPacket {
        _ = try ctx.getInput("input");
        
        // Mock function calling response
        const result = std.json.Value{
            .object = std.json.ObjectMap.init(self.allocator),
        };
        
        var packet = try DataPacket.init(self.allocator, self.id, .object, result);
        try packet.metadata.put("function_called", "true");
        try packet.metadata.put("model", self.model);
        
        return packet;
    }
};

/// Conversational LLM Node with memory
pub const ConversationalLLMNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    model: []const u8,
    temperature: f32,
    history: ConversationHistory,
    system_prompt: ?[]const u8,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        model: []const u8,
        temperature: f32,
        max_history: usize,
        max_tokens: usize,
        system_prompt: ?[]const u8,
    ) !ConversationalLLMNode {
        return ConversationalLLMNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .model = try allocator.dupe(u8, model),
            .temperature = temperature,
            .history = ConversationHistory.init(allocator, max_history, max_tokens),
            .system_prompt = if (system_prompt) |sp| try allocator.dupe(u8, sp) else null,
        };
    }
    
    pub fn deinit(self: *ConversationalLLMNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.model);
        if (self.system_prompt) |sp| {
            self.allocator.free(sp);
        }
        self.history.deinit();
    }
    
    /// Execute conversation turn
    pub fn execute(self: *ConversationalLLMNode, ctx: *ExecutionContext) !*DataPacket {
        const input = try ctx.getInput("message");
        
        // Add user message to history
        const user_msg = try ConversationMessage.init(
            self.allocator,
            .user,
            input,
        );
        try self.history.addMessage(user_msg);
        
        // Mock assistant response
        const response = "This is a mock assistant response.";
        const assistant_msg = try ConversationMessage.init(
            self.allocator,
            .assistant,
            response,
        );
        try self.history.addMessage(assistant_msg);
        
        // Create response packet
        const result = std.json.Value{ .string = response };
        var packet = try DataPacket.init(self.allocator, self.id, .string, result);
        
        // Add metadata
        const token_count_str = try std.fmt.allocPrint(
            self.allocator,
            "{d}",
            .{self.history.getTokenCount()},
        );
        try packet.metadata.put("history_tokens", token_count_str);
        try packet.metadata.put("model", self.model);
        
        return packet;
    }
    
    /// Clear conversation history
    pub fn clearHistory(self: *ConversationalLLMNode) void {
        self.history.clear();
    }
};

/// Context window optimizer
pub const ContextOptimizer = struct {
    max_tokens: usize,
    reserve_for_completion: usize,
    
    pub fn init(max_tokens: usize, reserve_for_completion: usize) ContextOptimizer {
        return ContextOptimizer{
            .max_tokens = max_tokens,
            .reserve_for_completion = reserve_for_completion,
        };
    }
    
    /// Calculate available tokens for context
    pub fn getAvailableTokens(self: *const ContextOptimizer) usize {
        return self.max_tokens - self.reserve_for_completion;
    }
    
    /// Optimize messages to fit within token budget
    pub fn optimizeMessages(
        self: *const ContextOptimizer,
        allocator: Allocator,
        messages: []const ConversationMessage,
    ) !std.ArrayList(ConversationMessage) {
        var optimized: std.ArrayList(ConversationMessage) = .{};
        var current_tokens: usize = 0;
        const available = self.getAvailableTokens();

        // Always include system messages first
        for (messages) |msg| {
            if (msg.role == .system) {
                const copied = try ConversationMessage.init(
                    allocator,
                    msg.role,
                    msg.content,
                );
                try optimized.append(allocator, copied);
                current_tokens += copied.token_count;
            }
        }

        // Add most recent messages that fit
        var i: usize = messages.len;
        while (i > 0) : (i -= 1) {
            const msg = messages[i - 1];
            if (msg.role == .system) continue;

            if (current_tokens + msg.token_count <= available) {
                const copied = try ConversationMessage.init(
                    allocator,
                    msg.role,
                    msg.content,
                );
                try optimized.insert(
                    allocator,
                    countSystemMessages(optimized.items),
                    copied,
                );
                current_tokens += copied.token_count;
            } else {
                break;
            }
        }
        
        return optimized;
    }
    
    fn countSystemMessages(messages: []const ConversationMessage) usize {
        var count: usize = 0;
        for (messages) |msg| {
            if (msg.role == .system) {
                count += 1;
            }
        }
        return count;
    }
};

/// Token budget manager
pub const TokenBudget = struct {
    total_budget: usize,
    used_tokens: usize,
    reserved_tokens: usize,
    
    pub fn init(total_budget: usize) TokenBudget {
        return TokenBudget{
            .total_budget = total_budget,
            .used_tokens = 0,
            .reserved_tokens = 0,
        };
    }
    
    /// Reserve tokens for a specific purpose
    pub fn reserve(self: *TokenBudget, tokens: usize) !void {
        if (self.reserved_tokens + tokens > self.total_budget) {
            return error.InsufficientTokens;
        }
        self.reserved_tokens += tokens;
    }
    
    /// Use reserved tokens
    pub fn use(self: *TokenBudget, tokens: usize) !void {
        if (self.used_tokens + tokens > self.reserved_tokens) {
            return error.ExceededReservedTokens;
        }
        self.used_tokens += tokens;
    }
    
    /// Get remaining budget
    pub fn getRemaining(self: *const TokenBudget) usize {
        return self.total_budget - self.reserved_tokens;
    }
    
    /// Get used percentage
    pub fn getUsagePercent(self: *const TokenBudget) f64 {
        return @as(f64, @floatFromInt(self.used_tokens)) / 
               @as(f64, @floatFromInt(self.total_budget)) * 100.0;
    }
    
    /// Reset budget
    pub fn reset(self: *TokenBudget) void {
        self.used_tokens = 0;
        self.reserved_tokens = 0;
    }
};

/// Streaming response handler
pub const StreamingHandler = struct {
    allocator: Allocator,
    buffer: std.ArrayList(u8),
    on_chunk: ?*const fn ([]const u8) void,
    
    pub fn init(allocator: Allocator) StreamingHandler {
        return StreamingHandler{
            .allocator = allocator,
            .buffer = .{},
            .on_chunk = null,
        };
    }

    pub fn deinit(self: *StreamingHandler) void {
        self.buffer.deinit(self.allocator);
    }

    /// Handle incoming chunk
    pub fn handleChunk(self: *StreamingHandler, chunk: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, chunk);
        
        if (self.on_chunk) |callback| {
            callback(chunk);
        }
    }
    
    /// Get complete response
    pub fn getResponse(self: *const StreamingHandler) []const u8 {
        return self.buffer.items;
    }
    
    /// Clear buffer
    pub fn clear(self: *StreamingHandler) void {
        self.buffer.clearRetainingCapacity();
    }
};

// Tests
const testing = std.testing;

test "FunctionDefinition creation" {
    const params = std.json.Value{
        .object = std.json.ObjectMap.init(testing.allocator),
    };
    
    var func = try FunctionDefinition.init(
        testing.allocator,
        "get_weather",
        "Get current weather",
        params,
    );
    defer func.deinit(testing.allocator);
    
    try testing.expectEqualStrings("get_weather", func.name);
    try testing.expectEqualStrings("Get current weather", func.description);
}

test "ConversationMessage creation" {
    var msg = try ConversationMessage.init(
        testing.allocator,
        .user,
        "Hello, world!",
    );
    defer msg.deinit(testing.allocator);
    
    try testing.expectEqual(ConversationMessage.MessageRole.user, msg.role);
    try testing.expectEqualStrings("Hello, world!", msg.content);
    try testing.expect(msg.token_count > 0);
}

test "ConversationHistory management" {
    var history = ConversationHistory.init(testing.allocator, 10, 1000);
    defer history.deinit();
    
    const msg1 = try ConversationMessage.init(testing.allocator, .user, "First message");
    try history.addMessage(msg1);
    
    const msg2 = try ConversationMessage.init(testing.allocator, .assistant, "Response");
    try history.addMessage(msg2);
    
    try testing.expectEqual(@as(usize, 2), history.messages.items.len);
    try testing.expect(history.getTokenCount() > 0);
}

test "ConversationHistory pruning" {
    var history = ConversationHistory.init(testing.allocator, 2, 1000);
    defer history.deinit();
    
    // Add system message (should never be pruned)
    const sys = try ConversationMessage.init(testing.allocator, .system, "System");
    try history.addMessage(sys);
    
    // Add 3 user messages (should prune oldest)
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        const msg = try ConversationMessage.init(testing.allocator, .user, "Message");
        try history.addMessage(msg);
    }
    
    // Should have system + 2 newest messages
    try testing.expectEqual(@as(usize, 2), history.messages.items.len);
}

test "LLMFunctionNode creation" {
    var node = try LLMFunctionNode.init(
        testing.allocator,
        "func1",
        "Function Caller",
        "gpt-4",
        0.7,
    );
    defer node.deinit();
    
    try testing.expectEqualStrings("func1", node.id);
    try testing.expectEqual(@as(f32, 0.7), node.temperature);
}

test "ConversationalLLMNode creation" {
    var node = try ConversationalLLMNode.init(
        testing.allocator,
        "conv1",
        "Chat Bot",
        "gpt-4",
        0.7,
        10,
        2000,
        "You are a helpful assistant.",
    );
    defer node.deinit();
    
    try testing.expectEqualStrings("conv1", node.id);
    try testing.expect(node.system_prompt != null);
}

test "ContextOptimizer token calculation" {
    const optimizer = ContextOptimizer.init(4096, 500);
    
    try testing.expectEqual(@as(usize, 3596), optimizer.getAvailableTokens());
}

test "TokenBudget management" {
    var budget = TokenBudget.init(1000);
    
    try budget.reserve(300);
    try testing.expectEqual(@as(usize, 700), budget.getRemaining());
    
    try budget.use(100);
    try testing.expectEqual(@as(f64, 10.0), budget.getUsagePercent());
    
    budget.reset();
    try testing.expectEqual(@as(usize, 1000), budget.getRemaining());
}

test "StreamingHandler chunk handling" {
    var handler = StreamingHandler.init(testing.allocator);
    defer handler.deinit();
    
    try handler.handleChunk("Hello ");
    try handler.handleChunk("world!");
    
    try testing.expectEqualStrings("Hello world!", handler.getResponse());
    
    handler.clear();
    try testing.expectEqual(@as(usize, 0), handler.getResponse().len);
}

test "ConversationHistory clear non-system messages" {
    var history = ConversationHistory.init(testing.allocator, 10, 1000);
    defer history.deinit();
    
    const sys = try ConversationMessage.init(testing.allocator, .system, "System");
    try history.addMessage(sys);
    
    const user = try ConversationMessage.init(testing.allocator, .user, "User");
    try history.addMessage(user);
    
    history.clear();
    
    try testing.expectEqual(@as(usize, 1), history.messages.items.len);
    try testing.expectEqual(ConversationMessage.MessageRole.system, 
        history.messages.items[0].role);
}
