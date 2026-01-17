// ============================================================================
// HyperShimmy OData Chat Action Handler (Zig)
// ============================================================================
//
// Day 28 Implementation: Chat OData V4 action
//
// Features:
// - OData V4 Chat action endpoint
// - Request/response mapping to ChatRequest/ChatResponse complex types
// - Integration with orchestrator handler
// - Proper OData error handling
//
// Endpoint:
// - POST /odata/v4/research/Chat
//
// Integration:
// - Uses orchestrator.zig for RAG pipeline
// - Maps OData complex types to orchestrator structs
// - Returns OData-compliant responses
// ============================================================================

const std = @import("std");
const json = std.json;
const mem = std.mem;
const orchestrator = @import("orchestrator.zig");

// ============================================================================
// OData Complex Types (matching metadata.xml)
// ============================================================================

/// ChatRequest complex type from OData metadata
pub const ChatRequest = struct {
    SessionId: []const u8,
    Message: []const u8,
    IncludeSources: bool,
    MaxTokens: ?i32 = null,
    Temperature: ?f64 = null,
};

/// ChatResponse complex type from OData metadata
pub const ChatResponse = struct {
    MessageId: []const u8,
    Content: []const u8,
    SourceIds: []const []const u8,
    Metadata: []const u8,
};

/// OData error response structure
pub const ODataError = struct {
    @"error": ErrorDetails,
    
    pub const ErrorDetails = struct {
        code: []const u8,
        message: []const u8,
        target: ?[]const u8 = null,
        details: ?[]ErrorDetail = null,
    };
    
    pub const ErrorDetail = struct {
        code: []const u8,
        message: []const u8,
        target: ?[]const u8 = null,
    };
};

// ============================================================================
// OData Chat Action Handler
// ============================================================================

pub const ODataChatHandler = struct {
    allocator: mem.Allocator,
    orchestrator_handler: *orchestrator.OrchestratorHandler,
    
    pub fn init(
        allocator: mem.Allocator,
        orch_handler: *orchestrator.OrchestratorHandler,
    ) ODataChatHandler {
        return .{
            .allocator = allocator,
            .orchestrator_handler = orch_handler,
        };
    }
    
    /// Handle OData Chat action
    pub fn handleChatAction(
        self: *ODataChatHandler,
        request_body: []const u8,
    ) ![]const u8 {
        std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
        std.debug.print("📡 OData Chat Action Request\n", .{});
        std.debug.print("=" ** 70 ++ "\n", .{});
        
        // Parse OData ChatRequest
        const parsed = json.parseFromSlice(
            ChatRequest,
            self.allocator,
            request_body,
            .{},
        ) catch |err| {
            std.debug.print("❌ Failed to parse ChatRequest: {any}\n", .{err});
            return try self.formatODataError(
                "BadRequest",
                "Invalid ChatRequest format",
                null,
            );
        };
        defer parsed.deinit();
        
        const chat_req = parsed.value;
        
        std.debug.print("SessionId: {s}\n", .{chat_req.SessionId});
        std.debug.print("Message: {s}\n", .{chat_req.Message});
        std.debug.print("IncludeSources: {}\n", .{chat_req.IncludeSources});
        if (chat_req.MaxTokens) |max_tokens| {
            std.debug.print("MaxTokens: {d}\n", .{max_tokens});
        }
        if (chat_req.Temperature) |temp| {
            std.debug.print("Temperature: {d}\n", .{temp});
        }
        
        // Convert OData ChatRequest to OrchestrateRequest
        const orch_request = try self.chatRequestToOrchestrateRequest(chat_req);
        defer if (orch_request.source_ids) |ids| self.allocator.free(ids);
        
        // Serialize orchestrate request
        var orch_json = std.ArrayList(u8).init(self.allocator);
        defer orch_json.deinit();
        
        try json.stringify(orch_request, .{}, orch_json.writer());
        
        // Call orchestrator
        const orch_response_json = self.orchestrator_handler.handleOrchestrate(
            orch_json.items,
        ) catch |err| {
            std.debug.print("❌ Orchestrator failed: {any}\n", .{err});
            return try self.formatODataError(
                "InternalError",
                "Chat orchestration failed",
                null,
            );
        };
        defer self.allocator.free(orch_response_json);
        
        // Parse orchestrator response
        const orch_parsed = try json.parseFromSlice(
            orchestrator.OrchestrateResponse,
            self.allocator,
            orch_response_json,
            .{},
        );
        defer orch_parsed.deinit();
        
        const orch_resp = orch_parsed.value;
        
        // Convert to OData ChatResponse
        const chat_response = try self.orchestrateResponseToChatResponse(
            chat_req.SessionId,
            orch_resp,
        );
        
        // Serialize OData response
        var response_json = std.ArrayList(u8).init(self.allocator);
        defer response_json.deinit();
        
        try json.stringify(chat_response, .{}, response_json.writer());
        
        std.debug.print("\n✅ Chat action completed successfully\n", .{});
        std.debug.print("=" ** 70 ++ "\n\n", .{});
        
        return try self.allocator.dupe(u8, response_json.items);
    }
    
    /// Convert OData ChatRequest to OrchestrateRequest
    fn chatRequestToOrchestrateRequest(
        self: *ODataChatHandler,
        chat_req: ChatRequest,
    ) !orchestrator.OrchestrateRequest {
        // For now, no source filtering (use all available sources)
        // In production, would query Sources entity set by SessionId
        
        return orchestrator.OrchestrateRequest{
            .query = chat_req.Message,
            .source_ids = null, // TODO: Fetch from SessionId
            .collection_name = null,
            .enable_reformulation = true,
            .enable_reranking = true,
            .max_chunks = if (chat_req.MaxTokens) |mt| @as(usize, @intCast(@divFloor(mt, 100))) else null,
            .min_score = null,
            .add_citations = chat_req.IncludeSources,
            .use_cache = false,
        };
    }
    
    /// Convert OrchestrateResponse to OData ChatResponse
    fn orchestrateResponseToChatResponse(
        self: *ODataChatHandler,
        session_id: []const u8,
        orch_resp: orchestrator.OrchestrateResponse,
    ) !ChatResponse {
        // Generate message ID (in production, would be from database)
        const message_id = try self.generateMessageId(session_id);
        
        // Build metadata JSON
        const metadata = try self.buildMetadata(orch_resp);
        
        // Duplicate citations for response
        var citations = std.ArrayList([]const u8).init(self.allocator);
        for (orch_resp.citations) |citation| {
            try citations.append(try self.allocator.dupe(u8, citation));
        }
        
        return ChatResponse{
            .MessageId = message_id,
            .Content = try self.allocator.dupe(u8, orch_resp.response),
            .SourceIds = try citations.toOwnedSlice(),
            .Metadata = metadata,
        };
    }
    
    /// Generate message ID (mock implementation)
    fn generateMessageId(self: *ODataChatHandler, session_id: []const u8) ![]const u8 {
        const timestamp = std.time.timestamp();
        return try std.fmt.allocPrint(
            self.allocator,
            "{s}-msg-{d}",
            .{ session_id, timestamp },
        );
    }
    
    /// Build metadata JSON from orchestrate response
    fn buildMetadata(
        self: *ODataChatHandler,
        orch_resp: orchestrator.OrchestrateResponse,
    ) ![]const u8 {
        const metadata = .{
            .confidence = orch_resp.confidence,
            .query_intent = orch_resp.query_intent,
            .reformulated_query = orch_resp.reformulated_query,
            .chunks_retrieved = orch_resp.chunks_retrieved,
            .chunks_used = orch_resp.chunks_used,
            .tokens_used = orch_resp.tokens_used,
            .retrieval_time_ms = orch_resp.retrieval_time_ms,
            .generation_time_ms = orch_resp.generation_time_ms,
            .total_time_ms = orch_resp.total_time_ms,
            .from_cache = orch_resp.from_cache,
        };
        
        var metadata_json = std.ArrayList(u8).init(self.allocator);
        defer metadata_json.deinit();
        
        try json.stringify(metadata, .{}, metadata_json.writer());
        
        return try self.allocator.dupe(u8, metadata_json.items);
    }
    
    /// Format OData error response
    fn formatODataError(
        self: *ODataChatHandler,
        code: []const u8,
        message: []const u8,
        target: ?[]const u8,
    ) ![]const u8 {
        const error_response = ODataError{
            .@"error" = .{
                .code = code,
                .message = message,
                .target = target,
                .details = null,
            },
        };
        
        var error_json = std.ArrayList(u8).init(self.allocator);
        defer error_json.deinit();
        
        try json.stringify(error_response, .{}, error_json.writer());
        
        return try self.allocator.dupe(u8, error_json.items);
    }
    
    pub fn deinit(self: *ODataChatHandler) void {
        _ = self;
        // Cleanup if needed
    }
};

// ============================================================================
// HTTP Handler Integration
// ============================================================================

/// Handle OData Chat action endpoint
pub fn handleODataChatRequest(
    allocator: mem.Allocator,
    body: []const u8,
) ![]const u8 {
    // Create orchestrator handler
    var orch_handler = orchestrator.OrchestratorHandler.init(allocator);
    defer orch_handler.deinit();
    
    // Create OData chat handler
    var chat_handler = ODataChatHandler.init(allocator, &orch_handler);
    defer chat_handler.deinit();
    
    // Handle chat action
    return try chat_handler.handleChatAction(body);
}

// ============================================================================
// Testing
// ============================================================================

test "odata chat handler basic" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "SessionId": "session-123",
        \\  "Message": "What is machine learning?",
        \\  "IncludeSources": true,
        \\  "MaxTokens": 500,
        \\  "Temperature": 0.7
        \\}
    ;
    
    const response = try handleODataChatRequest(allocator, request_json);
    defer allocator.free(response);
    
    // Should return valid ChatResponse JSON
    try testing.expect(response.len > 0);
    try testing.expect(mem.indexOf(u8, response, "MessageId") != null);
    try testing.expect(mem.indexOf(u8, response, "Content") != null);
    try testing.expect(mem.indexOf(u8, response, "SourceIds") != null);
    try testing.expect(mem.indexOf(u8, response, "Metadata") != null);
}

test "odata chat handler without sources" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "SessionId": "session-456",
        \\  "Message": "Explain neural networks",
        \\  "IncludeSources": false
        \\}
    ;
    
    const response = try handleODataChatRequest(allocator, request_json);
    defer allocator.free(response);
    
    try testing.expect(response.len > 0);
}

test "odata chat handler invalid json" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const invalid_json = "{ invalid json }";
    
    const response = try handleODataChatRequest(allocator, invalid_json);
    defer allocator.free(response);
    
    // Should return OData error
    try testing.expect(mem.indexOf(u8, response, "error") != null);
    try testing.expect(mem.indexOf(u8, response, "BadRequest") != null);
}
