const std = @import("std");
const mem = std.mem;

// ============================================================================
// HyperShimmy Document Indexing Handler
// ============================================================================
//
// Day 24 Implementation: Document indexing API endpoints
//
// Features:
// - Index document endpoint
// - Re-index document endpoint
// - Delete document index endpoint
// - Get indexing status endpoint
// - Batch indexing support
// - Integration with upload handler
//
// Endpoints:
// - POST /api/index - Index a document
// - POST /api/reindex - Re-index a document
// - DELETE /api/index/:fileId - Delete document index
// - GET /api/index/status/:fileId - Get indexing status
// ============================================================================

/// Index request
pub const IndexRequest = struct {
    file_id: []const u8,
    text: []const u8,
    filename: []const u8,
    file_type: []const u8,
    chunk_size: u32 = 512,
    overlap_size: u32 = 50,
    batch_size: u32 = 10,
    
    pub fn validate(self: IndexRequest) !void {
        if (self.file_id.len == 0) return error.EmptyFileId;
        if (self.text.len == 0) return error.EmptyText;
        if (self.chunk_size < 100 or self.chunk_size > 2000) return error.InvalidChunkSize;
        if (self.overlap_size >= self.chunk_size) return error.InvalidOverlapSize;
        if (self.batch_size < 1 or self.batch_size > 100) return error.InvalidBatchSize;
    }
};

/// Index status response
pub const IndexStatus = struct {
    success: bool,
    status: []const u8, // "pending", "processing", "completed", "failed"
    file_id: []const u8,
    total_chunks: u32,
    processed_chunks: u32,
    indexed_points: u32,
    progress_percent: u32,
    error_message: ?[]const u8 = null,
    
    pub fn deinit(self: *IndexStatus, allocator: mem.Allocator) void {
        allocator.free(self.status);
        allocator.free(self.file_id);
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
    }
};

/// Re-index request
pub const ReindexRequest = struct {
    file_id: []const u8,
    
    pub fn validate(self: ReindexRequest) !void {
        if (self.file_id.len == 0) return error.EmptyFileId;
    }
};

/// Batch index request
pub const BatchIndexRequest = struct {
    file_ids: [][]const u8,
    
    pub fn validate(self: BatchIndexRequest) !void {
        if (self.file_ids.len == 0) return error.EmptyFileIdList;
        if (self.file_ids.len > 100) return error.TooManyFiles;
    }
};

/// Document indexer handler
pub const IndexerHandler = struct {
    allocator: mem.Allocator,
    
    pub fn init(allocator: mem.Allocator) IndexerHandler {
        return IndexerHandler{
            .allocator = allocator,
        };
    }
    
    /// Handle index document request
    pub fn handleIndex(self: *IndexerHandler, request_body: []const u8) ![]u8 {
        // Parse JSON request
        const request = try self.parseIndexRequest(request_body);
        defer {
            self.allocator.free(request.file_id);
            self.allocator.free(request.text);
            self.allocator.free(request.filename);
            self.allocator.free(request.file_type);
        }
        
        // Validate request
        try request.validate();
        
        // Call Mojo indexing function via FFI
        // For now, mock the response
        const status = try self.indexDocument(request);
        
        // Convert to JSON
        const json = try self.statusToJson(status);
        
        return json;
    }
    
    /// Handle re-index request
    pub fn handleReindex(self: *IndexerHandler, request_body: []const u8) ![]u8 {
        // Parse JSON request
        const request = try self.parseReindexRequest(request_body);
        defer self.allocator.free(request.file_id);
        
        // Validate request
        try request.validate();
        
        // Call Mojo re-indexing function via FFI
        const status = try self.reindexDocument(request.file_id);
        
        // Convert to JSON
        const json = try self.statusToJson(status);
        
        return json;
    }
    
    /// Handle delete index request
    pub fn handleDeleteIndex(self: *IndexerHandler, file_id: []const u8) ![]u8 {
        if (file_id.len == 0) return error.EmptyFileId;
        
        // Call Mojo delete function via FFI
        const success = try self.deleteDocumentIndex(file_id);
        
        if (success) {
            return try std.fmt.allocPrint(
                self.allocator,
                \\{{"success":true,"message":"Index deleted successfully","fileId":"{s}"}}
                ,
                .{file_id},
            );
        } else {
            return try std.fmt.allocPrint(
                self.allocator,
                \\{{"success":false,"error":"Failed to delete index","fileId":"{s}"}}
                ,
                .{file_id},
            );
        }
    }
    
    /// Handle get status request
    pub fn handleGetStatus(self: *IndexerHandler, file_id: []const u8) ![]u8 {
        if (file_id.len == 0) return error.EmptyFileId;
        
        // Call Mojo status function via FFI
        const status = try self.getIndexStatus(file_id);
        
        // Convert to JSON
        const json = try self.statusToJson(status);
        
        return json;
    }
    
    /// Handle batch index request
    pub fn handleBatchIndex(self: *IndexerHandler, request_body: []const u8) ![]u8 {
        // Parse JSON request
        const request = try self.parseBatchIndexRequest(request_body);
        defer {
            for (request.file_ids) |file_id| {
                self.allocator.free(file_id);
            }
            self.allocator.free(request.file_ids);
        }
        
        // Validate request
        try request.validate();
        
        // Process each file
        var results = std.ArrayList(IndexStatus).init(self.allocator);
        defer {
            for (results.items) |*status| {
                status.deinit(self.allocator);
            }
            results.deinit();
        }
        
        for (request.file_ids) |file_id| {
            const status = try self.getIndexStatus(file_id);
            try results.append(status);
        }
        
        // Convert to JSON array
        const json = try self.batchStatusToJson(results.items);
        
        return json;
    }
    
    /// Parse index request from JSON
    fn parseIndexRequest(self: *IndexerHandler, json: []const u8) !IndexRequest {
        // Simplified JSON parsing - in real implementation, use a JSON parser
        // For now, create a mock request
        return IndexRequest{
            .file_id = try self.allocator.dupe(u8, "file_123"),
            .text = try self.allocator.dupe(u8, "Sample document text"),
            .filename = try self.allocator.dupe(u8, "document.txt"),
            .file_type = try self.allocator.dupe(u8, "text/plain"),
            .chunk_size = 512,
            .overlap_size = 50,
            .batch_size = 10,
        };
    }
    
    /// Parse re-index request from JSON
    fn parseReindexRequest(self: *IndexerHandler, json: []const u8) !ReindexRequest {
        return ReindexRequest{
            .file_id = try self.allocator.dupe(u8, "file_123"),
        };
    }
    
    /// Parse batch index request from JSON
    fn parseBatchIndexRequest(self: *IndexerHandler, json: []const u8) !BatchIndexRequest {
        var file_ids = try self.allocator.alloc([]const u8, 2);
        file_ids[0] = try self.allocator.dupe(u8, "file_123");
        file_ids[1] = try self.allocator.dupe(u8, "file_456");
        
        return BatchIndexRequest{
            .file_ids = file_ids,
        };
    }
    
    /// Call Mojo indexing function
    fn indexDocument(self: *IndexerHandler, request: IndexRequest) !IndexStatus {
        // Mock implementation - would call Mojo FFI function
        // extern fn index_document(...) -> *u8;
        
        // For now, return mock status
        return IndexStatus{
            .success = true,
            .status = try self.allocator.dupe(u8, "completed"),
            .file_id = try self.allocator.dupe(u8, request.file_id),
            .total_chunks = 10,
            .processed_chunks = 10,
            .indexed_points = 10,
            .progress_percent = 100,
            .error_message = null,
        };
    }
    
    /// Call Mojo re-indexing function
    fn reindexDocument(self: *IndexerHandler, file_id: []const u8) !IndexStatus {
        // Mock implementation
        return IndexStatus{
            .success = true,
            .status = try self.allocator.dupe(u8, "completed"),
            .file_id = try self.allocator.dupe(u8, file_id),
            .total_chunks = 10,
            .processed_chunks = 10,
            .indexed_points = 10,
            .progress_percent = 100,
            .error_message = null,
        };
    }
    
    /// Call Mojo delete function
    fn deleteDocumentIndex(self: *IndexerHandler, file_id: []const u8) !bool {
        // Mock implementation
        _ = file_id;
        return true;
    }
    
    /// Call Mojo status function
    fn getIndexStatus(self: *IndexerHandler, file_id: []const u8) !IndexStatus {
        // Mock implementation
        return IndexStatus{
            .success = true,
            .status = try self.allocator.dupe(u8, "completed"),
            .file_id = try self.allocator.dupe(u8, file_id),
            .total_chunks = 10,
            .processed_chunks = 10,
            .indexed_points = 10,
            .progress_percent = 100,
            .error_message = null,
        };
    }
    
    /// Convert status to JSON
    fn statusToJson(self: *IndexerHandler, status: IndexStatus) ![]u8 {
        if (!status.success) {
            return try std.fmt.allocPrint(
                self.allocator,
                \\{{"success":false,"status":"{s}","fileId":"{s}","error":"{s}"}}
                ,
                .{
                    status.status,
                    status.file_id,
                    status.error_message orelse "Unknown error",
                },
            );
        }
        
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{"success":true,"status":"{s}","fileId":"{s}","totalChunks":{d},"processedChunks":{d},"indexedPoints":{d},"progressPercent":{d}}}
            ,
            .{
                status.status,
                status.file_id,
                status.total_chunks,
                status.processed_chunks,
                status.indexed_points,
                status.progress_percent,
            },
        );
    }
    
    /// Convert batch status to JSON array
    fn batchStatusToJson(self: *IndexerHandler, statuses: []IndexStatus) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        defer buffer.deinit();
        
        try buffer.appendSlice("{\"results\":[");
        
        for (statuses, 0..) |status, i| {
            if (i > 0) try buffer.append(',');
            
            const status_json = try self.statusToJson(status);
            defer self.allocator.free(status_json);
            
            try buffer.appendSlice(status_json);
        }
        
        try buffer.appendSlice("]}");
        
        return try buffer.toOwnedSlice();
    }
    
    /// Handle OData index action
    pub fn handleODataIndexAction(allocator: mem.Allocator, params: anytype) ![]u8 {
        var handler = IndexerHandler.init(allocator);
        
        // Extract parameters
        const file_id = params.get("fileId") orelse return error.MissingFileId;
        const text = params.get("text") orelse return error.MissingText;
        
        // Create request
        const request = IndexRequest{
            .file_id = file_id,
            .text = text,
            .filename = params.get("filename") orelse "unknown",
            .file_type = params.get("fileType") orelse "text/plain",
            .chunk_size = 512,
            .overlap_size = 50,
            .batch_size = 10,
        };
        
        // Validate and index
        try request.validate();
        const status = try handler.indexDocument(request);
        
        // Convert to OData format
        return try handler.statusToJson(status);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "index request validation" {
    const request = IndexRequest{
        .file_id = "test_123",
        .text = "Sample text",
        .filename = "test.txt",
        .file_type = "text/plain",
        .chunk_size = 512,
        .overlap_size = 50,
        .batch_size = 10,
    };
    
    try request.validate();
}

test "invalid chunk size" {
    const request = IndexRequest{
        .file_id = "test_123",
        .text = "Sample text",
        .filename = "test.txt",
        .file_type = "text/plain",
        .chunk_size = 50, // Too small
        .overlap_size = 25,
        .batch_size = 10,
    };
    
    try std.testing.expectError(error.InvalidChunkSize, request.validate());
}

test "invalid overlap size" {
    const request = IndexRequest{
        .file_id = "test_123",
        .text = "Sample text",
        .filename = "test.txt",
        .file_type = "text/plain",
        .chunk_size = 512,
        .overlap_size = 600, // Larger than chunk size
        .batch_size = 10,
    };
    
    try std.testing.expectError(error.InvalidOverlapSize, request.validate());
}

test "indexer handler init" {
    const handler = IndexerHandler.init(std.testing.allocator);
    _ = handler;
}

test "index request parsing" {
    var handler = IndexerHandler.init(std.testing.allocator);
    
    const json = "{}"; // Simplified
    const request = try handler.parseIndexRequest(json);
    defer {
        handler.allocator.free(request.file_id);
        handler.allocator.free(request.text);
        handler.allocator.free(request.filename);
        handler.allocator.free(request.file_type);
    }
    
    try std.testing.expect(request.file_id.len > 0);
}

test "status to json" {
    var handler = IndexerHandler.init(std.testing.allocator);
    
    const status = IndexStatus{
        .success = true,
        .status = try handler.allocator.dupe(u8, "completed"),
        .file_id = try handler.allocator.dupe(u8, "test_123"),
        .total_chunks = 10,
        .processed_chunks = 10,
        .indexed_points = 10,
        .progress_percent = 100,
        .error_message = null,
    };
    defer {
        handler.allocator.free(status.status);
        handler.allocator.free(status.file_id);
    }
    
    const json = try handler.statusToJson(status);
    defer handler.allocator.free(json);
    
    try std.testing.expect(json.len > 0);
    try std.testing.expect(mem.indexOf(u8, json, "success") != null);
}
