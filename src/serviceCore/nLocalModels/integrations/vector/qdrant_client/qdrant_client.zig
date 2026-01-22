// Qdrant Vector Database Client in Zig
// High-performance HTTP client for Qdrant REST API
// Target: 5-10x faster than Python client
//
// Features:
// - REST API implementation
// - JSON serialization/deserialization
// - Vector search operations
// - Point CRUD operations
// - Collection management
// - C ABI for Mojo integration

const std = @import("std");
const http = std.http;
const json = std.json;
const mem = std.mem;
const Allocator = mem.Allocator;

/// Vector point representation
pub const VectorPoint = struct {
    id: []const u8,
    vector: []f32,
    payload: ?json.Value = null,
};

/// Search result
pub const SearchResult = struct {
    id: []const u8,
    score: f32,
    payload_json: ?[]const u8 = null,
    vector: ?[]f32 = null,
};

/// Search parameters
pub const SearchParams = struct {
    vector: []const f32,
    collection_name: []const u8,
    limit: u32 = 10,
    score_threshold: ?f32 = null,
    with_payload: bool = true,
    with_vector: bool = false,
};

/// Qdrant client
pub const QdrantClient = struct {
    allocator: Allocator,
    base_url: []const u8,
    client: http.Client,
    
    pub fn init(allocator: Allocator, host: []const u8, port: u16) !*QdrantClient {
        const qdrant = try allocator.create(QdrantClient);
        
        // Build base URL
        const base_url = try std.fmt.allocPrint(
            allocator,
            "http://{s}:{d}",
            .{ host, port }
        );
        
        qdrant.* = QdrantClient{
            .allocator = allocator,
            .base_url = base_url,
            .client = http.Client{ .allocator = allocator },
        };
        
        return qdrant;
    }
    
    pub fn deinit(self: *QdrantClient) void {
        self.allocator.free(self.base_url);
        self.client.deinit();
        self.allocator.destroy(self);
    }

    pub fn freeSearchResults(self: *QdrantClient, results: []SearchResult) void {
        for (results) |result| {
            self.allocator.free(result.id);
            if (result.payload_json) |payload| {
                self.allocator.free(payload);
            }
            if (result.vector) |vector| {
                self.allocator.free(vector);
            }
        }
        self.allocator.free(results);
    }
    
    /// Search for similar vectors
    pub fn search(
        self: *QdrantClient,
        params: SearchParams,
    ) ![]SearchResult {
        // Build request URL
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/collections/{s}/points/search",
            .{ self.base_url, params.collection_name }
        );
        defer self.allocator.free(url);
        
        // Build JSON request body
        var request_body: std.io.Writer.Allocating = .init(self.allocator);
        defer request_body.deinit();
        const writer = &request_body.writer;

        try writer.writeAll("{\"vector\":[");
        
        for (params.vector, 0..) |value, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{d}", .{value});
        }
        
        try writer.print(
            "],\"limit\":{d},\"with_payload\":{},\"with_vector\":{}",
            .{ params.limit, params.with_payload, params.with_vector }
        );
        
        if (params.score_threshold) |threshold| {
            try writer.print(",\"score_threshold\":{d}", .{threshold});
        }
        
        try writer.writeAll("}");
        
        // Make HTTP request with fetch
        const uri = try std.Uri.parse(url);

        var response_body: std.io.Writer.Allocating = .init(self.allocator);
        defer response_body.deinit();

        const req = try self.client.fetch(.{
            .location = .{ .uri = uri },
            .method = .POST,
            .payload = request_body.written(),
            .response_writer = &response_body.writer,
            .headers = .{ .content_type = .{ .override = "application/json" } },
        });
        
        if (req.status != .ok) {
            return error.HttpRequestFailed;
        }
        
        const body = response_body.written();
        if (body.len == 0) {
            return self.allocator.alloc(SearchResult, 0);
        }

        // Parse JSON response
        const parsed = try json.parseFromSlice(
            json.Value,
            self.allocator,
            body,
            .{}
        );
        defer parsed.deinit();
        
        const result_value = parsed.value.object.get("result") orelse return error.InvalidResponse;
        const result_array = switch (result_value) {
            .array => |array| array,
            else => return error.InvalidResponse,
        };
        
        // Convert to SearchResult array
        const results = try self.allocator.alloc(SearchResult, result_array.items.len);
        var filled: usize = 0;
        errdefer {
            for (results[0..filled]) |result| {
                self.allocator.free(result.id);
                if (result.payload_json) |payload| {
                    self.allocator.free(payload);
                }
                if (result.vector) |vector| {
                    self.allocator.free(vector);
                }
            }
            self.allocator.free(results);
        }

        for (result_array.items, 0..) |item, i| {
            const obj = switch (item) {
                .object => |object| object,
                else => return error.InvalidResponse,
            };

            const id_value = obj.get("id") orelse return error.InvalidResponse;
            const score_value = obj.get("score") orelse return error.InvalidResponse;
            const payload_value = obj.get("payload");
            const vector_value = obj.get("vector");

            const id_str = switch (id_value) {
                .string => |s| try self.allocator.dupe(u8, s),
                .integer => |n| try std.fmt.allocPrint(self.allocator, "{d}", .{n}),
                else => try self.allocator.dupe(u8, "unknown"),
            };
            errdefer self.allocator.free(id_str);

            const score = switch (score_value) {
                .float => |f| @as(f32, @floatCast(f)),
                .integer => |n| @as(f32, @floatFromInt(n)),
                else => 0.0,
            };

            var payload_json: ?[]const u8 = null;
            errdefer if (payload_json) |payload| self.allocator.free(payload);
            if (payload_value) |payload| {
                payload_json = try json.Stringify.valueAlloc(self.allocator, payload, .{});
            }

            var vector_out: ?[]f32 = null;
            errdefer if (vector_out) |vector| self.allocator.free(vector);
            if (params.with_vector) {
                if (vector_value) |vector_json| {
                    const vector_array = switch (vector_json) {
                        .array => |array| array,
                        else => return error.InvalidResponse,
                    };
                    const vector_slice = try self.allocator.alloc(f32, vector_array.items.len);
                    for (vector_array.items, 0..) |vector_item, idx| {
                        vector_slice[idx] = switch (vector_item) {
                            .float => |f| @as(f32, @floatCast(f)),
                            .integer => |n| @as(f32, @floatFromInt(n)),
                            else => return error.InvalidResponse,
                        };
                    }
                    vector_out = vector_slice;
                }
            }

            results[i] = SearchResult{
                .id = id_str,
                .score = score,
                .payload_json = payload_json,
                .vector = vector_out,
            };
            filled += 1;
        }

        return results;
    }
    
    /// Upsert (insert or update) points
    pub fn upsert(
        self: *QdrantClient,
        collection_name: []const u8,
        points: []const VectorPoint,
    ) !void {
        // Build request URL
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/collections/{s}/points",
            .{ self.base_url, collection_name }
        );
        defer self.allocator.free(url);
        
        // Build JSON request body
        var request_body: std.io.Writer.Allocating = .init(self.allocator);
        defer request_body.deinit();
        const writer = &request_body.writer;

        try writer.writeAll("{\"points\":[");
        
        for (points, 0..) |point, i| {
            if (i > 0) try writer.writeAll(",");
            
            try writer.writeAll("{\"id\":\"");
            try writer.writeAll(point.id);
            try writer.writeAll("\",\"vector\":[");
            
            for (point.vector, 0..) |value, j| {
                if (j > 0) try writer.writeAll(",");
                try writer.print("{d}", .{value});
            }
            
            try writer.writeAll("]");
            
            if (point.payload) |payload| {
                try writer.writeAll(",\"payload\":");
                const payload_json = try json.Stringify.valueAlloc(self.allocator, payload, .{});
                defer self.allocator.free(payload_json);
                try writer.writeAll(payload_json);
            }
            
            try writer.writeAll("}");
        }
        
        try writer.writeAll("]}");
        
        // Make HTTP request with fetch
        const uri = try std.Uri.parse(url);
        
        const req = try self.client.fetch(.{
            .location = .{ .uri = uri },
            .method = .PUT,
            .payload = request_body.written(),
            .headers = .{ .content_type = .{ .override = "application/json" } },
        });
        
        if (req.status != .ok and req.status != .accepted) {
            return error.UpsertFailed;
        }
    }
    
    /// Delete points by IDs
    pub fn delete_points(
        self: *QdrantClient,
        collection_name: []const u8,
        point_ids: []const []const u8,
    ) !void {
        // Build request URL
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/collections/{s}/points/delete",
            .{ self.base_url, collection_name }
        );
        defer self.allocator.free(url);
        
        // Build JSON request body
        var request_body: std.io.Writer.Allocating = .init(self.allocator);
        defer request_body.deinit();
        const writer = &request_body.writer;

        try writer.writeAll("{\"points\":[");
        
        for (point_ids, 0..) |id, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("\"");
            try writer.writeAll(id);
            try writer.writeAll("\"");
        }
        
        try writer.writeAll("]}");
        
        // Make HTTP request with fetch
        const uri = try std.Uri.parse(url);
        
        const req = try self.client.fetch(.{
            .location = .{ .uri = uri },
            .method = .POST,
            .payload = request_body.written(),
            .headers = .{ .content_type = .{ .override = "application/json" } },
        });
        
        if (req.status != .ok and req.status != .accepted) {
            return error.DeleteFailed;
        }
    }
};

// ============================================================================
// C ABI for Mojo Integration
// ============================================================================

const CClient = opaque {};
const CSearchResult = extern struct {
    id: [*:0]const u8,
    score: f32,
    payload_json: [*:0]const u8, // JSON string or empty string
    vector_len: usize,
    vector: ?[*]const f32,
};

/// Create Qdrant client
export fn qdrant_client_create(
    host: [*:0]const u8,
    port: u16,
) callconv(.c) ?*CClient {
    const allocator = std.heap.c_allocator;
    const host_slice = mem.span(host);
    
    const client = QdrantClient.init(allocator, host_slice, port) catch return null;
    return @ptrCast(client);
}

/// Destroy Qdrant client
export fn qdrant_client_destroy(client: *CClient) callconv(.c) void {
    const real_client: *QdrantClient = @ptrCast(@alignCast(client));
    real_client.deinit();
}

/// Search vectors
export fn qdrant_search(
    client: *CClient,
    collection_name: [*:0]const u8,
    vector: [*]const f32,
    vector_len: usize,
    limit: u32,
    results_out: *[*]CSearchResult,
    count_out: *usize,
) callconv(.c) i32 {
    const real_client: *QdrantClient = @ptrCast(@alignCast(client));
    const collection_slice = mem.span(collection_name);
    const vector_slice = vector[0..vector_len];
    
    const params = SearchParams{
        .vector = vector_slice,
        .collection_name = collection_slice,
        .limit = limit,
        .with_payload = true,
        .with_vector = true,
    };
    
    const results = real_client.search(params) catch return -1;
    defer real_client.freeSearchResults(results);
    
    // Convert to C results
    const c_results = real_client.allocator.alloc(CSearchResult, results.len) catch return -1;
    var filled: usize = 0;
    errdefer {
        for (c_results[0..filled]) |result| {
            real_client.allocator.free(mem.span(result.id));
            real_client.allocator.free(mem.span(result.payload_json));
            if (result.vector) |vector_ptr| {
                real_client.allocator.free(vector_ptr[0..result.vector_len]);
            }
        }
        real_client.allocator.free(c_results);
    }

    for (results, 0..) |result, i| {
        const id_cstr = real_client.allocator.dupeZ(u8, result.id) catch return -1;
        const payload_cstr = if (result.payload_json) |payload| payload_blk: {
            const payload_copy = real_client.allocator.dupeZ(u8, payload) catch {
                real_client.allocator.free(id_cstr);
                return -1;
            };
            break :payload_blk payload_copy;
        } else real_client.allocator.dupeZ(u8, "") catch {
            real_client.allocator.free(id_cstr);
            return -1;
        };

        var vector_ptr: ?[*]const f32 = null;
        var vector_len_out: usize = 0;
        if (result.vector) |result_vector| {
            const vector_copy = real_client.allocator.alloc(f32, result_vector.len) catch {
                real_client.allocator.free(id_cstr);
                real_client.allocator.free(payload_cstr);
                return -1;
            };
            @memcpy(vector_copy, result_vector);
            vector_ptr = @as([*]const f32, @ptrCast(vector_copy.ptr));
            vector_len_out = result_vector.len;
        }

        c_results[i] = CSearchResult{
            .id = id_cstr.ptr,
            .score = result.score,
            .payload_json = payload_cstr.ptr,
            .vector_len = vector_len_out,
            .vector = vector_ptr,
        };
        filled += 1;
    }
    
    results_out.* = c_results.ptr;
    count_out.* = results.len;
    
    return 0;
}

/// Free search results
export fn qdrant_free_results(results: [*]CSearchResult, count: usize) callconv(.c) void {
    const allocator = std.heap.c_allocator;
    const slice = results[0..count];
    
    for (slice) |result| {
        const id_slice = mem.span(result.id);
        allocator.free(id_slice);
        
        const payload_slice = mem.span(result.payload_json);
        allocator.free(payload_slice);

        if (result.vector) |vector_ptr| {
            allocator.free(vector_ptr[0..result.vector_len]);
        }
    }
    
    allocator.free(slice);
}
