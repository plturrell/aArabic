// ============================================================================
// HyperShimmy OData Mindmap Action Handler (Zig)
// ============================================================================
//
// Day 38 Implementation: Mindmap OData V4 action
//
// Features:
// - OData V4 Mindmap action endpoint
// - Request/response mapping to MindmapRequest/MindmapResponse complex types
// - Integration with Mojo mindmap generator
// - Support for tree and radial layouts
// - Configuration options (depth, children, layout algorithm)
// - Proper OData error handling
//
// Endpoint:
// - POST /odata/v4/research/GenerateMindmap
//
// Integration:
// - Uses mindmap_generator.mojo for mindmap generation
// - Maps OData complex types to Mojo FFI structs
// - Returns OData-compliant responses with nodes and edges
// ============================================================================

const std = @import("std");
const json = std.json;
const mem = std.mem;

// ============================================================================
// OData Complex Types (matching metadata.xml)
// ============================================================================

/// MindmapRequest complex type from OData metadata
pub const MindmapRequest = struct {
    SourceIds: []const []const u8,
    LayoutAlgorithm: []const u8, // "tree", "radial"
    MaxDepth: ?i32 = null,
    MaxChildrenPerNode: ?i32 = null,
    CanvasWidth: ?f32 = null,
    CanvasHeight: ?f32 = null,
    AutoSelectRoot: ?bool = null,
    RootEntityId: ?[]const u8 = null,
};

/// MindmapNode in response
pub const MindmapNode = struct {
    Id: []const u8,
    Label: []const u8,
    NodeType: []const u8, // "root", "branch", "leaf"
    EntityType: []const u8,
    Level: i32,
    X: f32,
    Y: f32,
    Confidence: f32,
    ChildCount: i32,
    ParentId: []const u8,
};

/// MindmapEdge in response
pub const MindmapEdge = struct {
    FromNodeId: []const u8,
    ToNodeId: []const u8,
    RelationshipType: []const u8,
    Label: []const u8,
    Style: []const u8, // "solid", "dashed", "dotted"
};

/// MindmapResponse complex type from OData metadata
pub const MindmapResponse = struct {
    MindmapId: []const u8,
    Title: []const u8,
    Nodes: []const MindmapNode,
    Edges: []const MindmapEdge,
    RootNodeId: []const u8,
    LayoutAlgorithm: []const u8,
    MaxDepth: i32,
    NodeCount: i32,
    EdgeCount: i32,
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
// FFI Structures for Mojo Integration
// ============================================================================

/// FFI request structure for Mojo
const MojoMindmapRequest = extern struct {
    source_ids_ptr: [*]const [*:0]const u8,
    source_ids_len: usize,
    layout_algorithm: [*:0]const u8,
    max_depth: i32,
    max_children_per_node: i32,
    canvas_width: f32,
    canvas_height: f32,
    auto_select_root: bool,
    root_entity_id: [*:0]const u8,
};

/// FFI node structure from Mojo
const MojoMindmapNode = extern struct {
    id: [*:0]const u8,
    label: [*:0]const u8,
    node_type: [*:0]const u8,
    entity_type: [*:0]const u8,
    level: i32,
    x: f32,
    y: f32,
    confidence: f32,
    child_count: i32,
    parent_id: [*:0]const u8,
};

/// FFI edge structure from Mojo
const MojoMindmapEdge = extern struct {
    from_node_id: [*:0]const u8,
    to_node_id: [*:0]const u8,
    relationship_type: [*:0]const u8,
    label: [*:0]const u8,
    style: [*:0]const u8,
};

/// FFI response structure from Mojo
const MojoMindmapResponse = extern struct {
    mindmap_id: [*:0]const u8,
    title: [*:0]const u8,
    nodes_ptr: [*]const MojoMindmapNode,
    nodes_len: usize,
    edges_ptr: [*]const MojoMindmapEdge,
    edges_len: usize,
    root_node_id: [*:0]const u8,
    layout_algorithm: [*:0]const u8,
    max_depth: i32,
    node_count: i32,
    edge_count: i32,
    processing_time_ms: i32,
    metadata: [*:0]const u8,
};

/// FFI function declarations
extern "C" fn mojo_generate_mindmap(request: *const MojoMindmapRequest) callconv(.C) *MojoMindmapResponse;
extern "C" fn mojo_free_mindmap_response(response: *MojoMindmapResponse) callconv(.C) void;

// ============================================================================
// OData Mindmap Action Handler
// ============================================================================

pub const ODataMindmapHandler = struct {
    allocator: mem.Allocator,

    pub fn init(allocator: mem.Allocator) ODataMindmapHandler {
        return .{
            .allocator = allocator,
        };
    }

    /// Handle OData Mindmap action
    pub fn handleMindmapAction(
        self: *ODataMindmapHandler,
        request_body: []const u8,
    ) ![]const u8 {
        std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
        std.debug.print("🗺️  OData Mindmap Action Request\n", .{});
        std.debug.print("=" ** 70 ++ "\n", .{});

        // Parse OData MindmapRequest
        const parsed = json.parseFromSlice(
            MindmapRequest,
            self.allocator,
            request_body,
            .{},
        ) catch |err| {
            std.debug.print("❌ Failed to parse MindmapRequest: {any}\n", .{err});
            return try self.formatODataError(
                "BadRequest",
                "Invalid MindmapRequest format",
                null,
            );
        };
        defer parsed.deinit();

        const mindmap_req = parsed.value;

        std.debug.print("SourceIds: {d} sources\n", .{mindmap_req.SourceIds.len});
        std.debug.print("LayoutAlgorithm: {s}\n", .{mindmap_req.LayoutAlgorithm});
        if (mindmap_req.MaxDepth) |depth| {
            std.debug.print("MaxDepth: {d}\n", .{depth});
        }
        if (mindmap_req.MaxChildrenPerNode) |children| {
            std.debug.print("MaxChildrenPerNode: {d}\n", .{children});
        }

        // Validate layout algorithm
        if (!self.isValidLayoutAlgorithm(mindmap_req.LayoutAlgorithm)) {
            return try self.formatODataError(
                "BadRequest",
                "Invalid LayoutAlgorithm. Must be one of: tree, radial",
                null,
            );
        }

        // Convert to Mojo FFI structure
        const start_time = std.time.milliTimestamp();
        const mojo_request = try self.mindmapRequestToMojoFFI(mindmap_req);
        defer self.freeMojoRequest(mojo_request);

        // Call Mojo mindmap generator
        std.debug.print("\n🔄 Calling Mojo mindmap generator...\n", .{});
        const mojo_response = mojo_generate_mindmap(&mojo_request);
        defer mojo_free_mindmap_response(mojo_response);

        const end_time = std.time.milliTimestamp();
        const processing_time = @as(i32, @intCast(end_time - start_time));

        std.debug.print("✅ Mindmap generated in {d}ms\n", .{processing_time});
        std.debug.print("Nodes: {d}\n", .{mojo_response.node_count});
        std.debug.print("Edges: {d}\n", .{mojo_response.edge_count});

        // Convert to OData response
        const mindmap_response = try self.mojoResponseToMindmapResponse(
            mojo_response,
            processing_time,
        );

        // Serialize OData response
        var response_json = std.ArrayList(u8).init(self.allocator);
        defer response_json.deinit();

        try json.stringify(mindmap_response, .{}, response_json.writer());

        std.debug.print("\n✅ Mindmap action completed successfully\n", .{});
        std.debug.print("=" ** 70 ++ "\n\n", .{});

        return try self.allocator.dupe(u8, response_json.items);
    }

    /// Validate layout algorithm
    fn isValidLayoutAlgorithm(self: *ODataMindmapHandler, algorithm: []const u8) bool {
        _ = self;
        const valid_algorithms = [_][]const u8{
            "tree",
            "radial",
        };

        for (valid_algorithms) |valid_alg| {
            if (mem.eql(u8, algorithm, valid_alg)) {
                return true;
            }
        }
        return false;
    }

    /// Convert OData MindmapRequest to Mojo FFI structure
    fn mindmapRequestToMojoFFI(
        self: *ODataMindmapHandler,
        mindmap_req: MindmapRequest,
    ) !MojoMindmapRequest {
        // Convert source IDs to C strings
        const source_ids = try self.allocator.alloc([*:0]const u8, mindmap_req.SourceIds.len);
        for (mindmap_req.SourceIds, 0..) |source_id, i| {
            const c_str = try self.allocator.dupeZ(u8, source_id);
            source_ids[i] = c_str.ptr;
        }

        // Convert layout algorithm to C string
        const layout_algorithm = try self.allocator.dupeZ(u8, mindmap_req.LayoutAlgorithm);

        // Convert optional root entity ID
        const root_entity_id = if (mindmap_req.RootEntityId) |root_id|
            try self.allocator.dupeZ(u8, root_id)
        else
            try self.allocator.dupeZ(u8, "");

        return MojoMindmapRequest{
            .source_ids_ptr = source_ids.ptr,
            .source_ids_len = source_ids.len,
            .layout_algorithm = layout_algorithm.ptr,
            .max_depth = mindmap_req.MaxDepth orelse 5,
            .max_children_per_node = mindmap_req.MaxChildrenPerNode orelse 10,
            .canvas_width = mindmap_req.CanvasWidth orelse 1200.0,
            .canvas_height = mindmap_req.CanvasHeight orelse 800.0,
            .auto_select_root = mindmap_req.AutoSelectRoot orelse true,
            .root_entity_id = root_entity_id.ptr,
        };
    }

    /// Free Mojo FFI request structure
    fn freeMojoRequest(self: *ODataMindmapHandler, request: MojoMindmapRequest) void {
        // Free source IDs
        const source_ids = request.source_ids_ptr[0..request.source_ids_len];
        for (source_ids) |source_id| {
            const slice = mem.span(source_id);
            self.allocator.free(slice);
        }
        self.allocator.free(source_ids);

        // Free layout algorithm
        const layout_alg = mem.span(request.layout_algorithm);
        self.allocator.free(layout_alg);

        // Free root entity ID
        const root_id = mem.span(request.root_entity_id);
        self.allocator.free(root_id);
    }

    /// Convert Mojo FFI response to OData MindmapResponse
    fn mojoResponseToMindmapResponse(
        self: *ODataMindmapHandler,
        mojo_resp: *MojoMindmapResponse,
        processing_time: i32,
    ) !MindmapResponse {
        // Convert nodes
        const mojo_nodes = mojo_resp.nodes_ptr[0..mojo_resp.nodes_len];
        var nodes = std.ArrayList(MindmapNode).init(self.allocator);

        for (mojo_nodes) |mojo_node| {
            const node = MindmapNode{
                .Id = try self.allocator.dupe(u8, mem.span(mojo_node.id)),
                .Label = try self.allocator.dupe(u8, mem.span(mojo_node.label)),
                .NodeType = try self.allocator.dupe(u8, mem.span(mojo_node.node_type)),
                .EntityType = try self.allocator.dupe(u8, mem.span(mojo_node.entity_type)),
                .Level = mojo_node.level,
                .X = mojo_node.x,
                .Y = mojo_node.y,
                .Confidence = mojo_node.confidence,
                .ChildCount = mojo_node.child_count,
                .ParentId = try self.allocator.dupe(u8, mem.span(mojo_node.parent_id)),
            };
            try nodes.append(node);
        }

        // Convert edges
        const mojo_edges = mojo_resp.edges_ptr[0..mojo_resp.edges_len];
        var edges = std.ArrayList(MindmapEdge).init(self.allocator);

        for (mojo_edges) |mojo_edge| {
            const edge = MindmapEdge{
                .FromNodeId = try self.allocator.dupe(u8, mem.span(mojo_edge.from_node_id)),
                .ToNodeId = try self.allocator.dupe(u8, mem.span(mojo_edge.to_node_id)),
                .RelationshipType = try self.allocator.dupe(u8, mem.span(mojo_edge.relationship_type)),
                .Label = try self.allocator.dupe(u8, mem.span(mojo_edge.label)),
                .Style = try self.allocator.dupe(u8, mem.span(mojo_edge.style)),
            };
            try edges.append(edge);
        }

        return MindmapResponse{
            .MindmapId = try self.allocator.dupe(u8, mem.span(mojo_resp.mindmap_id)),
            .Title = try self.allocator.dupe(u8, mem.span(mojo_resp.title)),
            .Nodes = try nodes.toOwnedSlice(),
            .Edges = try edges.toOwnedSlice(),
            .RootNodeId = try self.allocator.dupe(u8, mem.span(mojo_resp.root_node_id)),
            .LayoutAlgorithm = try self.allocator.dupe(u8, mem.span(mojo_resp.layout_algorithm)),
            .MaxDepth = mojo_resp.max_depth,
            .NodeCount = mojo_resp.node_count,
            .EdgeCount = mojo_resp.edge_count,
            .ProcessingTimeMs = processing_time,
            .Metadata = try self.allocator.dupe(u8, mem.span(mojo_resp.metadata)),
        };
    }

    /// Format OData error response
    fn formatODataError(
        self: *ODataMindmapHandler,
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

    pub fn deinit(self: *ODataMindmapHandler) void {
        _ = self;
        // Cleanup if needed
    }
};

// ============================================================================
// HTTP Handler Integration
// ============================================================================

/// Handle OData Mindmap action endpoint
pub fn handleODataMindmapRequest(
    allocator: mem.Allocator,
    body: []const u8,
) ![]const u8 {
    // Create OData mindmap handler
    var mindmap_handler = ODataMindmapHandler.init(allocator);
    defer mindmap_handler.deinit();

    // Handle mindmap action
    return try mindmap_handler.handleMindmapAction(body);
}

// ============================================================================
// Testing
// ============================================================================

test "odata mindmap handler basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const request_json =
        \\{
        \\  "SourceIds": ["doc_001", "doc_002", "doc_003"],
        \\  "LayoutAlgorithm": "tree",
        \\  "MaxDepth": 5,
        \\  "MaxChildrenPerNode": 10,
        \\  "AutoSelectRoot": true
        \\}
    ;

    const response = try handleODataMindmapRequest(allocator, request_json);
    defer allocator.free(response);

    // Should return valid MindmapResponse JSON
    try testing.expect(response.len > 0);
    try testing.expect(mem.indexOf(u8, response, "MindmapId") != null);
    try testing.expect(mem.indexOf(u8, response, "Nodes") != null);
    try testing.expect(mem.indexOf(u8, response, "Edges") != null);
}

test "odata mindmap handler radial layout" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const request_json =
        \\{
        \\  "SourceIds": ["doc_001", "doc_002"],
        \\  "LayoutAlgorithm": "radial",
        \\  "MaxDepth": 3,
        \\  "CanvasWidth": 1000.0,
        \\  "CanvasHeight": 800.0
        \\}
    ;

    const response = try handleODataMindmapRequest(allocator, request_json);
    defer allocator.free(response);

    try testing.expect(response.len > 0);
}

test "odata mindmap handler invalid algorithm" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const invalid_json =
        \\{
        \\  "SourceIds": ["doc_001"],
        \\  "LayoutAlgorithm": "invalid_layout"
        \\}
    ;

    const response = try handleODataMindmapRequest(allocator, invalid_json);
    defer allocator.free(response);

    // Should return OData error
    try testing.expect(mem.indexOf(u8, response, "error") != null);
    try testing.expect(mem.indexOf(u8, response, "BadRequest") != null);
}

test "odata mindmap handler invalid json" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const invalid_json = "{ invalid json }";

    const response = try handleODataMindmapRequest(allocator, invalid_json);
    defer allocator.free(response);

    // Should return OData error
    try testing.expect(mem.indexOf(u8, response, "error") != null);
    try testing.expect(mem.indexOf(u8, response, "BadRequest") != null);
}
