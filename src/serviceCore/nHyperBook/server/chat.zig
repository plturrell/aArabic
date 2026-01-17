// ============================================================================
// HyperShimmy Chat Handler (Zig)
// ============================================================================
//
// Day 26 Implementation: Chat endpoint with RAG support
//
// Features:
// - POST /chat endpoint
// - Integration with Mojo LLM module
// - RAG context from semantic search
// - JSON request/response
// - Error handling
//
// Integration:
// - Calls llm_chat.mojo via FFI
// - Uses semantic_search.zig for context
// - Prepares for streaming (Day 30)
// ============================================================================

const std = @import("std");
const http = std.http;
const json = std.json;
const mem = std.mem;

// ============================================================================
// Chat Request/Response Structures
// ============================================================================

pub const ChatRequest = struct {
    message: []const u8,
    source_ids: ?[]const []const u8 = null,
    session_id: ?[]const u8 = null,
    stream: bool = false,
    temperature: ?f32 = null,
    max_tokens: ?usize = null,
};

pub const ChatResponse = struct {
    response: []const u8,
    sources_used: []const []const u8,
    tokens_used: usize,
    processing_time_ms: u64,
    session_id: []const u8,
};

pub const ChatError = struct {
    error: []const u8,
    code: []const u8,
    message: []const u8,
};

// ============================================================================
// Chat Handler
// ============================================================================

pub const ChatHandler = struct {
    allocator: mem.Allocator,
    shimmy_endpoint: []const u8,
    max_context_chunks: usize,
    
    pub fn init(allocator: mem.Allocator, shimmy_endpoint: []const u8) ChatHandler {
        return .{
            .allocator = allocator,
            .shimmy_endpoint = shimmy_endpoint,
            .max_context_chunks = 5,
        };
    }
    
    pub fn handleChat(
        self: *ChatHandler,
        request_body: []const u8,
    ) ![]const u8 {
        // Parse request
        const parsed = try json.parseFromSlice(
            ChatRequest,
            self.allocator,
            request_body,
            .{},
        );
        defer parsed.deinit();
        
        const req = parsed.value;
        
        std.debug.print("\n" ++ "=" ** 60 ++ "\n", .{});
        std.debug.print("🤖 Chat Request\n", .{});
        std.debug.print("=" ** 60 ++ "\n", .{});
        std.debug.print("Message: {s}\n", .{req.message});
        std.debug.print("Sources: {?}\n", .{req.source_ids});
        std.debug.print("Session: {?s}\n", .{req.session_id});
        std.debug.print("Stream: {}\n", .{req.stream});
        
        const start_time = std.time.milliTimestamp();
        
        // Get context from semantic search if sources provided
        var context_chunks = std.ArrayList([]const u8).init(self.allocator);
        defer context_chunks.deinit();
        
        var sources_used = std.ArrayList([]const u8).init(self.allocator);
        defer sources_used.deinit();
        
        if (req.source_ids) |source_ids| {
            std.debug.print("\n📚 Retrieving context...\n", .{});
            
            // In production, would call semantic search
            // For now, use mock context
            for (source_ids) |source_id| {
                const chunk = try std.fmt.allocPrint(
                    self.allocator,
                    "Context from source {s}: Machine learning concepts and AI fundamentals.",
                    .{source_id},
                );
                try context_chunks.append(chunk);
                try sources_used.append(try self.allocator.dupe(u8, source_id));
            }
            
            std.debug.print("Retrieved {} context chunks\n", .{context_chunks.items.len});
        }
        
        // Generate response using Shimmy LLM
        std.debug.print("\n🤖 Generating response...\n", .{});
        
        // In production, would call Mojo FFI hs_chat_complete()
        // For now, generate mock response
        const response_text = try self.generateMockResponse(
            req.message,
            context_chunks.items,
        );
        
        const end_time = std.time.milliTimestamp();
        const processing_time = @as(u64, @intCast(end_time - start_time));
        
        // Estimate tokens
        const tokens_used = req.message.len / 4 + response_text.len / 4;
        
        std.debug.print("✅ Response generated in {}ms\n", .{processing_time});
        std.debug.print("Tokens used: ~{}\n", .{tokens_used});
        
        // Build response
        const session_id = req.session_id orelse "default";
        
        const response = ChatResponse{
            .response = response_text,
            .sources_used = sources_used.items,
            .tokens_used = tokens_used,
            .processing_time_ms = processing_time,
            .session_id = session_id,
        };
        
        // Serialize to JSON
        var response_json = std.ArrayList(u8).init(self.allocator);
        defer response_json.deinit();
        
        try json.stringify(response, .{}, response_json.writer());
        
        return try self.allocator.dupe(u8, response_json.items);
    }
    
    fn generateMockResponse(
        self: *ChatHandler,
        message: []const u8,
        context: []const []const u8,
    ) ![]const u8 {
        // Generate contextual mock response
        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();
        
        if (context.len > 0) {
            try response.appendSlice("Based on your documents, ");
            
            // Check message intent
            if (mem.indexOf(u8, message, "summarize") != null or
                mem.indexOf(u8, message, "summary") != null)
            {
                try response.appendSlice(
                    \\here's a summary of the key points:
                    \\
                    \\• The documents discuss machine learning and AI concepts
                    \\• Deep learning is mentioned as a subset of ML
                    \\• Neural networks are key technologies
                    \\
                    \\This information comes from your uploaded sources.
                );
            } else if (mem.indexOf(u8, message, "explain") != null) {
                try response.appendSlice(
                    \\I can explain based on the context:
                    \\
                    \\The documents cover fundamental AI concepts including machine learning, 
                    \\neural networks, and pattern recognition. These technologies enable 
                    \\computers to learn from data without explicit programming.
                    \\
                    \\Would you like me to elaborate on any specific aspect?
                );
            } else if (mem.indexOf(u8, message, "compare") != null) {
                try response.appendSlice(
                    \\comparing the information in your documents:
                    \\
                    \\Both documents discuss AI and machine learning, but from different 
                    \\perspectives. They complement each other in covering both theory 
                    \\and practical applications.
                );
            } else {
                try response.appendSlice(
                    \\I found relevant information in your documents. The content discusses 
                    \\machine learning and AI concepts that may be relevant to your question. 
                    \\Would you like me to provide more specific details?
                );
            }
        } else {
            try response.appendSlice(
                \\I don't have any relevant context to answer this question. 
                \\Please add some documents first or try rephrasing your question.
            );
        }
        
        return try self.allocator.dupe(u8, response.items);
    }
    
    pub fn handleChatStream(
        self: *ChatHandler,
        request_body: []const u8,
        writer: anytype,
    ) !void {
        // For streaming support (Day 30)
        // Would implement Server-Sent Events (SSE) here
        _ = self;
        _ = request_body;
        _ = writer;
        
        return error.NotImplemented;
    }
    
    pub fn deinit(self: *ChatHandler) void {
        _ = self;
        // Cleanup if needed
    }
};

// ============================================================================
// HTTP Handler Integration
// ============================================================================

pub fn handleChatRequest(
    allocator: mem.Allocator,
    method: http.Method,
    body: []const u8,
) ![]const u8 {
    if (method != .POST) {
        const error_response = ChatError{
            .error = "method_not_allowed",
            .code = "405",
            .message = "Only POST method is allowed for /chat",
        };
        
        var json_buffer = std.ArrayList(u8).init(allocator);
        defer json_buffer.deinit();
        
        try json.stringify(error_response, .{}, json_buffer.writer());
        return try allocator.dupe(u8, json_buffer.items);
    }
    
    // Create chat handler
    var handler = ChatHandler.init(allocator, "http://localhost:8001");
    defer handler.deinit();
    
    // Handle chat
    return try handler.handleChat(body);
}

// ============================================================================
// Testing
// ============================================================================

test "chat handler basic" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "message": "What is machine learning?",
        \\  "source_ids": ["doc_001"],
        \\  "session_id": "test_123"
        \\}
    ;
    
    const response = try handleChatRequest(
        allocator,
        .POST,
        request_json,
    );
    defer allocator.free(response);
    
    // Should return valid JSON
    try testing.expect(response.len > 0);
    try testing.expect(mem.indexOf(u8, response, "response") != null);
}

test "chat handler without context" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "message": "Hello",
        \\  "session_id": "test_456"
        \\}
    ;
    
    const response = try handleChatRequest(
        allocator,
        .POST,
        request_json,
    );
    defer allocator.free(response);
    
    // Should return response without context
    try testing.expect(response.len > 0);
}

test "chat handler method not allowed" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json = "{}";
    
    const response = try handleChatRequest(
        allocator,
        .GET,
        request_json,
    );
    defer allocator.free(response);
    
    // Should return error
    try testing.expect(mem.indexOf(u8, response, "method_not_allowed") != null);
}
