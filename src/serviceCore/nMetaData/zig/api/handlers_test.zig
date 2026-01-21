//! API Handlers Tests - Day 30
//!
//! Comprehensive tests for API endpoint handlers.
//! Tests request validation, response formats, error handling, and business logic.

const std = @import("std");
const testing = std.testing;

const Request = @import("../http/types.zig").Request;
const Response = @import("../http/types.zig").Response;
const handlers = @import("handlers.zig");

// ============================================================================
// Dataset Handler Tests
// ============================================================================

test "listDatasets: default pagination" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/datasets");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.listDatasets(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
    try testing.expect(resp.body != null);
}

test "listDatasets: custom pagination" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/datasets");
    try req.parseQuery("page=2&limit=20");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.listDatasets(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "listDatasets: limit validation" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/datasets");
    try req.parseQuery("limit=150");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.listDatasets(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

test "createDataset: valid request" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/v1/datasets");
    req.body = 
        \\{"name":"test_table","type":"table","schema":"public"}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.createDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 201), resp.status);
}

test "createDataset: missing name" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/v1/datasets");
    req.body = 
        \\{"name":"","type":"table"}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.createDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

test "createDataset: invalid type" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/v1/datasets");
    req.body = 
        \\{"name":"test","type":"invalid_type"}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.createDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

test "createDataset: invalid JSON" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/v1/datasets");
    req.body = "not json";
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.createDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

test "getDataset: existing dataset" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/datasets/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.getDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "getDataset: not found" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/datasets/ds-999");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-999"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.getDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 404), resp.status);
}

test "updateDataset: valid request" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .PUT;
    req.path = try allocator.dupe(u8, "/api/v1/datasets/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    req.body = 
        \\{"name":"updated_name","description":"Updated description"}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.updateDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "updateDataset: no fields provided" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .PUT;
    req.path = try allocator.dupe(u8, "/api/v1/datasets/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    req.body = "{}";
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.updateDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

test "deleteDataset: without dependencies" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .DELETE;
    req.path = try allocator.dupe(u8, "/api/v1/datasets/ds-002");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-002"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.deleteDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "deleteDataset: with dependencies no force" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .DELETE;
    req.path = try allocator.dupe(u8, "/api/v1/datasets/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.deleteDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 409), resp.status);
}

test "deleteDataset: with dependencies force delete" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .DELETE;
    req.path = try allocator.dupe(u8, "/api/v1/datasets/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    try req.parseQuery("force=true");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.deleteDataset(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

// ============================================================================
// Lineage Handler Tests
// ============================================================================

test "getUpstreamLineage: default depth" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/lineage/upstream/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.getUpstreamLineage(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "getUpstreamLineage: custom depth" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/lineage/upstream/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    try req.parseQuery("depth=3");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.getUpstreamLineage(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "getUpstreamLineage: depth too high" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/lineage/upstream/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    try req.parseQuery("depth=15");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.getUpstreamLineage(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

test "getDownstreamLineage: default depth" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/lineage/downstream/ds-001");
    try req.params.put(
        try allocator.dupe(u8, "id"),
        try allocator.dupe(u8, "ds-001"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.getDownstreamLineage(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "createLineageEdge: valid request" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/v1/lineage/edges");
    req.body = 
        \\{"source_id":"ds-001","target_id":"ds-002","edge_type":"direct"}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.createLineageEdge(&req, &resp);
    
    try testing.expectEqual(@as(u16, 201), resp.status);
}

test "createLineageEdge: missing source" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/v1/lineage/edges");
    req.body = 
        \\{"source_id":"","target_id":"ds-002"}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.createLineageEdge(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

test "createLineageEdge: self loop" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/v1/lineage/edges");
    req.body = 
        \\{"source_id":"ds-001","target_id":"ds-001"}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.createLineageEdge(&req, &resp);
    
    try testing.expectEqual(@as(u16, 400), resp.status);
}

// ============================================================================
// System Handler Tests
// ============================================================================

test "healthCheck: returns healthy" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/health");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.healthCheck(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
    try testing.expect(resp.body != null);
}

test "systemStatus: returns status" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/status");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.systemStatus(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
    try testing.expect(resp.body != null);
}

test "apiInfo: returns info" {
    const allocator = testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/v1/info");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try handlers.apiInfo(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
    try testing.expect(resp.body != null);
}
