// ============================================================================
// HyperShimmy OData Summary Action Handler (Zig)
// ============================================================================
//
// Day 32 Implementation: Summary OData V4 action
//
// Features:
// - OData V4 Summary action endpoint
// - Request/response mapping to SummaryRequest/SummaryResponse complex types
// - Integration with Mojo summary generator via FFI
// - Support for multiple summary types
// - Proper OData error handling
//
// Endpoint:
// - POST /odata/v4/research/GenerateSummary
//
// Integration:
// - Uses Mojo summary_generator.mojo via FFI
// - Maps OData complex types to Mojo structs
// - Returns OData-compliant responses
// ============================================================================

const std = @import("std");
const json = std.json;
const mem = std.mem;

// ============================================================================
// OData Complex Types (matching metadata.xml)
// ============================================================================

/// SummaryRequest complex type from OData metadata
pub const SummaryRequest = struct {
    SourceIds: []const []const u8,
    SummaryType: []const u8, // "brief", "detailed", "executive", "bullet_points", "comparative"
    MaxLength: ?i32 = null, // Max words in summary
    IncludeCitations: bool = true,
    IncludeKeyPoints: bool = true,
    Tone: ?[]const u8 = null, // "professional", "academic", "casual"
    FocusAreas: ?[]const []const u8 = null, // Specific topics to emphasize
};

/// KeyPoint structure for response
pub const KeyPoint = struct {
    Content: []const u8,
    Importance: f32, // 0.0-1.0
    SourceIds: []const []const u8,
    Category: []const u8,
};

/// SummaryResponse complex type from OData metadata
pub const SummaryResponse = struct {
    SummaryId: []const u8,
    SummaryText: []const u8,
    KeyPoints: []const KeyPoint,
    SourceIds: []const []const u8,
    SummaryType: []const u8,
    WordCount: i32,
    Confidence: f32,
    ProcessingTimeMs: i32,
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
// Mojo FFI Structures (mirroring summary_generator.mojo)
// ============================================================================

/// FFI structure for Mojo SummaryRequest
const MojoSummaryRequest = extern struct {
    source_ids_ptr: [*]const [*:0]const u8,
    source_ids_len: usize,
    summary_type: [*:0]const u8,
    max_length: i32,
    include_citations: bool,
    include_key_points: bool,
    tone: [*:0]const u8,
    focus_areas_ptr: [*]const [*:0]const u8,
    focus_areas_len: usize,
};

/// FFI structure for Mojo KeyPoint
const MojoKeyPoint = extern struct {
    content: [*:0]const u8,
    importance: f32,
    source_ids_ptr: [*]const [*:0]const u8,
    source_ids_len: usize,
    category: [*:0]const u8,
};

/// FFI structure for Mojo SummaryResponse
const MojoSummaryResponse = extern struct {
    summary_text: [*:0]const u8,
    key_points_ptr: [*]const MojoKeyPoint,
    key_points_len: usize,
    source_ids_ptr: [*]const [*:0]const u8,
    source_ids_len: usize,
    summary_type: [*:0]const u8,
    word_count: i32,
    confidence: f32,
    processing_time_ms: i32,
    metadata: [*:0]const u8,
};

// ============================================================================
// Mojo FFI Declarations
// ============================================================================

extern "C" fn mojo_generate_summary(request: *const MojoSummaryRequest) callconv(.C) *MojoSummaryResponse;
extern "C" fn mojo_free_summary_response(response: *MojoSummaryResponse) callconv(.C) void;

// ============================================================================
// OData Summary Action Handler
// ============================================================================

pub const ODataSummaryHandler = struct {
    allocator: mem.Allocator,
    
    pub fn init(allocator: mem.Allocator) ODataSummaryHandler {
        return .{
            .allocator = allocator,
        };
    }
    
    /// Handle OData GenerateSummary action
    pub fn handleSummaryAction(
        self: *ODataSummaryHandler,
        request_body: []const u8,
    ) ![]const u8 {
        std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
        std.debug.print("📊 OData Summary Action Request\n", .{});
        std.debug.print("=" ** 70 ++ "\n", .{});
        
        // Parse OData SummaryRequest
        const parsed = json.parseFromSlice(
            SummaryRequest,
            self.allocator,
            request_body,
            .{},
        ) catch |err| {
            std.debug.print("❌ Failed to parse SummaryRequest: {any}\n", .{err});
            return try self.formatODataError(
                "BadRequest",
                "Invalid SummaryRequest format",
                null,
            );
        };
        defer parsed.deinit();
        
        const summary_req = parsed.value;
        
        std.debug.print("SourceIds: {d} sources\n", .{summary_req.SourceIds.len});
        std.debug.print("SummaryType: {s}\n", .{summary_req.SummaryType});
        std.debug.print("IncludeCitations: {}\n", .{summary_req.IncludeCitations});
        std.debug.print("IncludeKeyPoints: {}\n", .{summary_req.IncludeKeyPoints});
        if (summary_req.MaxLength) |max_len| {
            std.debug.print("MaxLength: {d}\n", .{max_len});
        }
        if (summary_req.Tone) |tone| {
            std.debug.print("Tone: {s}\n", .{tone});
        }
        
        // Validate summary type
        if (!self.isValidSummaryType(summary_req.SummaryType)) {
            std.debug.print("❌ Invalid summary type: {s}\n", .{summary_req.SummaryType});
            return try self.formatODataError(
                "BadRequest",
                "Invalid SummaryType. Must be one of: brief, detailed, executive, bullet_points, comparative",
                null,
            );
        }
        
        // Convert OData SummaryRequest to Mojo FFI structure
        const mojo_request = try self.summaryRequestToMojoFFI(summary_req);
        defer self.freeMojoRequest(mojo_request);
        
        // Call Mojo summary generator via FFI
        std.debug.print("🔧 Calling Mojo summary generator...\n", .{});
        const mojo_response = mojo_generate_summary(&mojo_request);
        defer mojo_free_summary_response(mojo_response);
        
        // Convert Mojo FFI response to OData SummaryResponse
        const summary_response = try self.mojoFFIToSummaryResponse(mojo_response);
        
        // Serialize OData response
        var response_json = std.ArrayList(u8).init(self.allocator);
        defer response_json.deinit();
        
        try json.stringify(summary_response, .{}, response_json.writer());
        
        std.debug.print("\n✅ Summary action completed successfully\n", .{});
        std.debug.print("Summary length: {d} words\n", .{summary_response.WordCount});
        std.debug.print("Key points: {d}\n", .{summary_response.KeyPoints.len});
        std.debug.print("Confidence: {d:.2}\n", .{summary_response.Confidence});
        std.debug.print("Processing time: {d}ms\n", .{summary_response.ProcessingTimeMs});
        std.debug.print("=" ** 70 ++ "\n\n", .{});
        
        return try self.allocator.dupe(u8, response_json.items);
    }
    
    /// Validate summary type
    fn isValidSummaryType(self: *ODataSummaryHandler, summary_type: []const u8) bool {
        _ = self;
        const valid_types = [_][]const u8{
            "brief",
            "detailed",
            "executive",
            "bullet_points",
            "comparative",
        };
        
        for (valid_types) |valid_type| {
            if (mem.eql(u8, summary_type, valid_type)) {
                return true;
            }
        }
        return false;
    }
    
    /// Convert OData SummaryRequest to Mojo FFI structure
    fn summaryRequestToMojoFFI(
        self: *ODataSummaryHandler,
        summary_req: SummaryRequest,
    ) !MojoSummaryRequest {
        // Convert source IDs to C strings
        const source_ids_c = try self.allocator.alloc([*:0]const u8, summary_req.SourceIds.len);
        for (summary_req.SourceIds, 0..) |source_id, i| {
            source_ids_c[i] = try self.allocator.dupeZ(u8, source_id);
        }
        
        // Convert summary type to C string
        const summary_type_c = try self.allocator.dupeZ(u8, summary_req.SummaryType);
        
        // Convert tone to C string (default to "professional")
        const tone = summary_req.Tone orelse "professional";
        const tone_c = try self.allocator.dupeZ(u8, tone);
        
        // Convert focus areas to C strings
        var focus_areas_c: [*]const [*:0]const u8 = undefined;
        var focus_areas_len: usize = 0;
        if (summary_req.FocusAreas) |focus_areas| {
            const areas_c = try self.allocator.alloc([*:0]const u8, focus_areas.len);
            for (focus_areas, 0..) |area, i| {
                areas_c[i] = try self.allocator.dupeZ(u8, area);
            }
            focus_areas_c = areas_c.ptr;
            focus_areas_len = focus_areas.len;
        } else {
            focus_areas_c = undefined;
        }
        
        return MojoSummaryRequest{
            .source_ids_ptr = source_ids_c.ptr,
            .source_ids_len = summary_req.SourceIds.len,
            .summary_type = summary_type_c,
            .max_length = summary_req.MaxLength orelse 500,
            .include_citations = summary_req.IncludeCitations,
            .include_key_points = summary_req.IncludeKeyPoints,
            .tone = tone_c,
            .focus_areas_ptr = focus_areas_c,
            .focus_areas_len = focus_areas_len,
        };
    }
    
    /// Free Mojo FFI request memory
    fn freeMojoRequest(self: *ODataSummaryHandler, request: MojoSummaryRequest) void {
        // Free source IDs
        const source_ids = request.source_ids_ptr[0..request.source_ids_len];
        for (source_ids) |source_id| {
            const slice = mem.span(source_id);
            self.allocator.free(slice);
        }
        self.allocator.free(source_ids);
        
        // Free summary type
        const summary_type_slice = mem.span(request.summary_type);
        self.allocator.free(summary_type_slice);
        
        // Free tone
        const tone_slice = mem.span(request.tone);
        self.allocator.free(tone_slice);
        
        // Free focus areas if present
        if (request.focus_areas_len > 0) {
            const focus_areas = request.focus_areas_ptr[0..request.focus_areas_len];
            for (focus_areas) |area| {
                const slice = mem.span(area);
                self.allocator.free(slice);
            }
            self.allocator.free(focus_areas);
        }
    }
    
    /// Convert Mojo FFI response to OData SummaryResponse
    fn mojoFFIToSummaryResponse(
        self: *ODataSummaryHandler,
        mojo_resp: *const MojoSummaryResponse,
    ) !SummaryResponse {
        // Generate summary ID
        const summary_id = try self.generateSummaryId();
        
        // Convert summary text
        const summary_text = try self.allocator.dupe(u8, mem.span(mojo_resp.summary_text));
        
        // Convert key points
        const key_points = try self.convertKeyPoints(
            mojo_resp.key_points_ptr[0..mojo_resp.key_points_len],
        );
        
        // Convert source IDs
        const source_ids_slice = mojo_resp.source_ids_ptr[0..mojo_resp.source_ids_len];
        var source_ids = std.ArrayList([]const u8).init(self.allocator);
        for (source_ids_slice) |source_id| {
            try source_ids.append(try self.allocator.dupe(u8, mem.span(source_id)));
        }
        
        // Convert summary type
        const summary_type = try self.allocator.dupe(u8, mem.span(mojo_resp.summary_type));
        
        // Convert metadata
        const metadata = try self.allocator.dupe(u8, mem.span(mojo_resp.metadata));
        
        return SummaryResponse{
            .SummaryId = summary_id,
            .SummaryText = summary_text,
            .KeyPoints = key_points,
            .SourceIds = try source_ids.toOwnedSlice(),
            .SummaryType = summary_type,
            .WordCount = mojo_resp.word_count,
            .Confidence = mojo_resp.confidence,
            .ProcessingTimeMs = mojo_resp.processing_time_ms,
            .Metadata = metadata,
        };
    }
    
    /// Convert Mojo key points to OData key points
    fn convertKeyPoints(
        self: *ODataSummaryHandler,
        mojo_key_points: []const MojoKeyPoint,
    ) ![]const KeyPoint {
        var key_points = try self.allocator.alloc(KeyPoint, mojo_key_points.len);
        
        for (mojo_key_points, 0..) |mojo_kp, i| {
            // Convert content
            const content = try self.allocator.dupe(u8, mem.span(mojo_kp.content));
            
            // Convert source IDs
            const source_ids_slice = mojo_kp.source_ids_ptr[0..mojo_kp.source_ids_len];
            var source_ids = std.ArrayList([]const u8).init(self.allocator);
            for (source_ids_slice) |source_id| {
                try source_ids.append(try self.allocator.dupe(u8, mem.span(source_id)));
            }
            
            // Convert category
            const category = try self.allocator.dupe(u8, mem.span(mojo_kp.category));
            
            key_points[i] = KeyPoint{
                .Content = content,
                .Importance = mojo_kp.importance,
                .SourceIds = try source_ids.toOwnedSlice(),
                .Category = category,
            };
        }
        
        return key_points;
    }
    
    /// Generate summary ID (mock implementation)
    fn generateSummaryId(self: *ODataSummaryHandler) ![]const u8 {
        const timestamp = std.time.timestamp();
        return try std.fmt.allocPrint(
            self.allocator,
            "summary-{d}",
            .{timestamp},
        );
    }
    
    /// Format OData error response
    fn formatODataError(
        self: *ODataSummaryHandler,
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
    
    pub fn deinit(self: *ODataSummaryHandler) void {
        _ = self;
        // Cleanup if needed
    }
};

// ============================================================================
// HTTP Handler Integration
// ============================================================================

/// Handle OData GenerateSummary action endpoint
pub fn handleODataSummaryRequest(
    allocator: mem.Allocator,
    body: []const u8,
) ![]const u8 {
    // Create OData summary handler
    var summary_handler = ODataSummaryHandler.init(allocator);
    defer summary_handler.deinit();
    
    // Handle summary action
    return try summary_handler.handleSummaryAction(body);
}

// ============================================================================
// Testing
// ============================================================================

test "odata summary handler brief" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "SourceIds": ["doc_001", "doc_002", "doc_003"],
        \\  "SummaryType": "brief",
        \\  "MaxLength": 150,
        \\  "IncludeCitations": true,
        \\  "IncludeKeyPoints": true,
        \\  "Tone": "professional"
        \\}
    ;
    
    const response = try handleODataSummaryRequest(allocator, request_json);
    defer allocator.free(response);
    
    // Should return valid SummaryResponse JSON
    try testing.expect(response.len > 0);
    try testing.expect(mem.indexOf(u8, response, "SummaryId") != null);
    try testing.expect(mem.indexOf(u8, response, "SummaryText") != null);
    try testing.expect(mem.indexOf(u8, response, "KeyPoints") != null);
    try testing.expect(mem.indexOf(u8, response, "WordCount") != null);
    try testing.expect(mem.indexOf(u8, response, "Confidence") != null);
}

test "odata summary handler executive" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "SourceIds": ["doc_001", "doc_002"],
        \\  "SummaryType": "executive",
        \\  "IncludeCitations": true,
        \\  "IncludeKeyPoints": true
        \\}
    ;
    
    const response = try handleODataSummaryRequest(allocator, request_json);
    defer allocator.free(response);
    
    try testing.expect(response.len > 0);
}

test "odata summary handler bullet points" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "SourceIds": ["doc_001", "doc_002", "doc_003"],
        \\  "SummaryType": "bullet_points",
        \\  "IncludeCitations": true,
        \\  "IncludeKeyPoints": false
        \\}
    ;
    
    const response = try handleODataSummaryRequest(allocator, request_json);
    defer allocator.free(response);
    
    try testing.expect(response.len > 0);
}

test "odata summary handler invalid type" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "SourceIds": ["doc_001"],
        \\  "SummaryType": "invalid_type",
        \\  "IncludeCitations": true,
        \\  "IncludeKeyPoints": true
        \\}
    ;
    
    const response = try handleODataSummaryRequest(allocator, request_json);
    defer allocator.free(response);
    
    // Should return OData error
    try testing.expect(mem.indexOf(u8, response, "error") != null);
    try testing.expect(mem.indexOf(u8, response, "BadRequest") != null);
}

test "odata summary handler invalid json" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const invalid_json = "{ invalid json }";
    
    const response = try handleODataSummaryRequest(allocator, invalid_json);
    defer allocator.free(response);
    
    // Should return OData error
    try testing.expect(mem.indexOf(u8, response, "error") != null);
    try testing.expect(mem.indexOf(u8, response, "BadRequest") != null);
}

test "odata summary handler with focus areas" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const request_json =
        \\{
        \\  "SourceIds": ["doc_001", "doc_002", "doc_003"],
        \\  "SummaryType": "detailed",
        \\  "MaxLength": 500,
        \\  "IncludeCitations": true,
        \\  "IncludeKeyPoints": true,
        \\  "Tone": "academic",
        \\  "FocusAreas": ["machine learning", "neural networks"]
        \\}
    ;
    
    const response = try handleODataSummaryRequest(allocator, request_json);
    defer allocator.free(response);
    
    try testing.expect(response.len > 0);
}
