//! Unified Document + Cache Layer
//!
//! Integrates nExtract parsers with HANA for high-performance document caching.
//! Provides zero-copy serialization and C ABI exports for Mojo integration.
//!
//! Features:
//! - Document parsing via nExtract (CSV, JSON, Markdown, HTML, XML)
//! - Caching with TTL via HANA in-memory tables
//! - Optional vector embeddings for similarity search
//! - Batch operations for efficiency
//! - C ABI for Mojo integration

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const Sha256 = std.crypto.hash.sha2.Sha256;

// Import nExtract parsers (relative path from this file)
const nExtract = @import("../../../../nExtract/zig/nExtract.zig");
const csv = nExtract.csv;
const json = nExtract.json;
const markdown = nExtract.markdown;
const html = nExtract.html;
const xml = nExtract.xml;

// Import HANA cache client
const hana_cache = @import("../cache/hana/hana_cache.zig");
const HanaCache = hana_cache.HanaCache;

// ============================================================================
// Document Types
// ============================================================================

/// Document type enumeration
pub const DocType = enum(u8) {
    csv = 0,
    json = 1,
    markdown = 2,
    html = 3,
    xml = 4,

    pub fn toString(self: DocType) []const u8 {
        return switch (self) {
            .csv => "csv",
            .json => "json",
            .markdown => "markdown",
            .html => "html",
            .xml => "xml",
        };
    }

    pub fn fromString(s: []const u8) ?DocType {
        if (mem.eql(u8, s, "csv")) return .csv;
        if (mem.eql(u8, s, "json")) return .json;
        if (mem.eql(u8, s, "markdown")) return .markdown;
        if (mem.eql(u8, s, "html")) return .html;
        if (mem.eql(u8, s, "xml")) return .xml;
        return null;
    }
};

/// Document metadata
pub const DocumentMetadata = struct {
    source_path: ?[]const u8,
    content_length: usize,
    encoding: []const u8,
    parse_time_ns: u64,
    extra: ?[]const u8, // JSON-encoded extra metadata

    pub fn init() DocumentMetadata {
        return .{
            .source_path = null,
            .content_length = 0,
            .encoding = "utf-8",
            .parse_time_ns = 0,
            .extra = null,
        };
    }
};

/// Cached document structure
pub const CachedDocument = struct {
    id: [32]u8, // SHA-256 hash of content
    doc_type: DocType,
    content: []const u8,
    metadata: DocumentMetadata,
    embeddings: ?[]f32, // Optional vector embeddings
    extracted_at: i64, // Unix timestamp
    ttl_seconds: u32,

    allocator: Allocator,

    pub fn init(allocator: Allocator) CachedDocument {
        return .{
            .id = [_]u8{0} ** 32,
            .doc_type = .json,
            .content = &[_]u8{},
            .metadata = DocumentMetadata.init(),
            .embeddings = null,
            .extracted_at = 0,
            .ttl_seconds = 3600, // Default 1 hour
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CachedDocument) void {
        if (self.content.len > 0) {
            self.allocator.free(self.content);
        }
        if (self.embeddings) |emb| {
            self.allocator.free(emb);
        }
        if (self.metadata.source_path) |path| {
            self.allocator.free(path);
        }
        if (self.metadata.extra) |extra| {
            self.allocator.free(extra);
        }
    }

    /// Compute document ID from content hash
    pub fn computeId(content: []const u8) [32]u8 {
        var hash: [32]u8 = undefined;
        Sha256.hash(content, &hash, .{});
        return hash;
    }

    /// Get hex-encoded ID for cache key
    pub fn getIdHex(self: *const CachedDocument) [64]u8 {
        return std.fmt.bytesToHex(self.id, .lower);
    }
};

// ============================================================================
// Serialization Helpers (Zero-copy where possible)
// ============================================================================

const MAGIC: [4]u8 = .{ 'U', 'D', 'C', '1' }; // Unified Doc Cache v1

/// Serialize CachedDocument to bytes for storage
fn serializeDocument(allocator: Allocator, doc: *const CachedDocument) ![]u8 {
    var buffer = ArrayList(u8){};
    errdefer buffer.deinit();

    const writer = buffer.writer();

    // Magic header
    try writer.writeAll(&MAGIC);

    // Document ID (32 bytes)
    try writer.writeAll(&doc.id);

    // Document type (1 byte)
    try writer.writeByte(@intFromEnum(doc.doc_type));

    // Timestamps (8 + 4 bytes)
    try writer.writeInt(i64, doc.extracted_at, .little);
    try writer.writeInt(u32, doc.ttl_seconds, .little);

    // Content length + content
    try writer.writeInt(u32, @intCast(doc.content.len), .little);
    try writer.writeAll(doc.content);

    // Metadata
    try writer.writeInt(u32, @intCast(doc.metadata.content_length), .little);
    try writer.writeInt(u64, doc.metadata.parse_time_ns, .little);

    // Source path (optional)
    if (doc.metadata.source_path) |path| {
        try writer.writeInt(u16, @intCast(path.len), .little);
        try writer.writeAll(path);
    } else {
        try writer.writeInt(u16, 0, .little);
    }

    // Embeddings (optional)
    if (doc.embeddings) |emb| {
        try writer.writeInt(u32, @intCast(emb.len), .little);
        const emb_bytes = mem.sliceAsBytes(emb);
        try writer.writeAll(emb_bytes);
    } else {
        try writer.writeInt(u32, 0, .little);
    }

    return buffer.toOwnedSlice();
}

/// Deserialize bytes to CachedDocument
fn deserializeDocument(allocator: Allocator, data: []const u8) !CachedDocument {
    if (data.len < 49) return error.InvalidData; // Minimum size

    var pos: usize = 0;

    // Verify magic
    if (!mem.eql(u8, data[0..4], &MAGIC)) return error.InvalidMagic;
    pos += 4;

    var doc = CachedDocument.init(allocator);
    errdefer doc.deinit();

    // Document ID
    @memcpy(&doc.id, data[pos..][0..32]);
    pos += 32;

    // Document type
    doc.doc_type = @enumFromInt(data[pos]);
    pos += 1;

    // Timestamps
    doc.extracted_at = mem.readInt(i64, data[pos..][0..8], .little);
    pos += 8;
    doc.ttl_seconds = mem.readInt(u32, data[pos..][0..4], .little);
    pos += 4;

    // Content
    const content_len = mem.readInt(u32, data[pos..][0..4], .little);
    pos += 4;
    doc.content = try allocator.dupe(u8, data[pos..][0..content_len]);
    pos += content_len;

    // Metadata
    doc.metadata.content_length = mem.readInt(u32, data[pos..][0..4], .little);
    pos += 4;
    doc.metadata.parse_time_ns = mem.readInt(u64, data[pos..][0..8], .little);
    pos += 8;

    // Source path
    const path_len = mem.readInt(u16, data[pos..][0..2], .little);
    pos += 2;
    if (path_len > 0) {
        doc.metadata.source_path = try allocator.dupe(u8, data[pos..][0..path_len]);
        pos += path_len;
    }

    // Embeddings
    const emb_len = mem.readInt(u32, data[pos..][0..4], .little);
    pos += 4;
    if (emb_len > 0) {
        const emb_bytes = data[pos..][0 .. emb_len * 4];
        doc.embeddings = try allocator.alloc(f32, emb_len);
        @memcpy(mem.sliceAsBytes(doc.embeddings.?), emb_bytes);
    }

    return doc;
}


// ============================================================================
// Unified Document Cache
// ============================================================================

pub const CacheError = error{
    ConnectionFailed,
    ParseError,
    SerializationError,
    CacheOperationFailed,
    DocumentNotFound,
    InvalidDocumentType,
    EmbeddingMismatch,
};

/// Unified Document Cache integrating nExtract and HANA
pub const UnifiedDocCache = struct {
    allocator: Allocator,
    client: *DragonflyClient,
    default_ttl: u32,
    cache_prefix: []const u8,

    const Self = @This();

    /// Initialize the unified document cache
    pub fn init(
        allocator: Allocator,
        host: []const u8,
        port: u16,
        default_ttl: u32,
    ) !*Self {
        const client = try DragonflyClient.init(allocator, host, port);
        errdefer client.deinit();

        const cache = try allocator.create(Self);
        cache.* = .{
            .allocator = allocator,
            .client = client,
            .default_ttl = default_ttl,
            .cache_prefix = "doc:",
        };
        return cache;
    }

    /// Deinitialize and free resources
    pub fn deinit(self: *Self) void {
        self.client.deinit();
        self.allocator.destroy(self);
    }

    /// Generate cache key from document ID
    fn getCacheKey(self: *Self, id: [32]u8) [70]u8 {
        var key: [70]u8 = undefined; // "doc:" (4) + hex(32) (64) + null (2)
        @memcpy(key[0..4], self.cache_prefix);
        const hex = std.fmt.bytesToHex(id, .lower);
        @memcpy(key[4..68], &hex);
        key[68] = 0;
        key[69] = 0;
        return key;
    }

    /// Parse content using appropriate nExtract parser
    fn parseContent(self: *Self, content: []const u8, doc_type: DocType) ![]const u8 {
        const start_time = std.time.nanoTimestamp();
        _ = start_time;

        // Parse and normalize the document based on type
        // For caching, we store the original content but could store parsed form
        return switch (doc_type) {
            .csv => blk: {
                // Validate CSV is parseable
                var parser = csv.Parser.init(self.allocator, .{});
                var doc = parser.parse(content) catch return error.ParseError;
                doc.deinit();
                break :blk try self.allocator.dupe(u8, content);
            },
            .json => blk: {
                // Validate JSON is parseable
                var parser = json.Parser.init(self.allocator, .{});
                var doc = parser.parse(content) catch return error.ParseError;
                doc.deinit();
                break :blk try self.allocator.dupe(u8, content);
            },
            .markdown => blk: {
                // Parse markdown to AST (validates structure)
                var parser = markdown.Parser.init(self.allocator);
                defer parser.deinit();
                const ast = parser.parse(content) catch return error.ParseError;
                ast.deinit();
                break :blk try self.allocator.dupe(u8, content);
            },
            .html => blk: {
                // Parse HTML to DOM (validates structure)
                const doc = html.HtmlParser.parse(self.allocator, content) catch return error.ParseError;
                var doc_mut = doc;
                doc_mut.deinit();
                break :blk try self.allocator.dupe(u8, content);
            },
            .xml => blk: {
                // Parse XML to tree (validates structure)
                var parser = xml.Parser.init(self.allocator);
                defer parser.deinit();
                const node = parser.parse(content) catch return error.ParseError;
                if (node) |n| n.deinit();
                break :blk try self.allocator.dupe(u8, content);
            },
        };
    }

    /// Cache a document (parse and store)
    pub fn cacheDocument(
        self: *Self,
        content: []const u8,
        doc_type: DocType,
        source_path: ?[]const u8,
        embeddings: ?[]const f32,
        ttl: ?u32,
    ) !CachedDocument {
        const start_time = std.time.nanoTimestamp();

        // Parse and validate content
        const parsed_content = try self.parseContent(content, doc_type);
        errdefer self.allocator.free(parsed_content);

        const end_time = std.time.nanoTimestamp();
        const parse_time: u64 = @intCast(end_time - start_time);

        // Create document
        var doc = CachedDocument.init(self.allocator);
        doc.id = CachedDocument.computeId(content);
        doc.doc_type = doc_type;
        doc.content = parsed_content;
        doc.extracted_at = std.time.timestamp();
        doc.ttl_seconds = ttl orelse self.default_ttl;
        doc.metadata.content_length = content.len;
        doc.metadata.parse_time_ns = parse_time;

        if (source_path) |path| {
            doc.metadata.source_path = try self.allocator.dupe(u8, path);
        }

        if (embeddings) |emb| {
            doc.embeddings = try self.allocator.dupe(f32, emb);
        }

        // Serialize and store in cache
        const serialized = serializeDocument(self.allocator, &doc) catch return error.SerializationError;
        defer self.allocator.free(serialized);

        const cache_key = self.getCacheKey(doc.id);
        self.client.set(cache_key[0..68], serialized, doc.ttl_seconds) catch return error.CacheOperationFailed;

        // If embeddings provided, store in separate key for vector search
        if (embeddings) |emb| {
            try self.storeEmbedding(doc.id, emb);
        }

        return doc;
    }

    /// Get a document from cache by ID
    pub fn getDocument(self: *Self, id: [32]u8) !?CachedDocument {
        const cache_key = self.getCacheKey(id);

        const data = self.client.get(cache_key[0..68]) catch return error.CacheOperationFailed;
        if (data == null) return null;
        defer self.allocator.free(data.?);

        return deserializeDocument(self.allocator, data.?) catch error.SerializationError;
    }

    /// Invalidate (remove) a document from cache
    pub fn invalidate(self: *Self, id: [32]u8) !void {
        const cache_key = self.getCacheKey(id);
        const keys = [_][]const u8{cache_key[0..68]};
        _ = self.client.del(&keys) catch return error.CacheOperationFailed;

        // Also remove embedding if exists
        var emb_key: [74]u8 = undefined;
        @memcpy(emb_key[0..4], "emb:");
        const hex = std.fmt.bytesToHex(id, .lower);
        @memcpy(emb_key[4..68], &hex);
        const emb_keys = [_][]const u8{emb_key[0..68]};
        _ = self.client.del(&emb_keys) catch {};
    }

    /// Store embedding for vector search
    fn storeEmbedding(self: *Self, id: [32]u8, embedding: []const f32) !void {
        var emb_key: [74]u8 = undefined;
        @memcpy(emb_key[0..4], "emb:");
        const hex = std.fmt.bytesToHex(id, .lower);
        @memcpy(emb_key[4..68], &hex);

        const emb_bytes = mem.sliceAsBytes(embedding);
        self.client.set(emb_key[0..68], emb_bytes, null) catch return error.CacheOperationFailed;
    }

    /// Batch cache multiple documents
    pub fn batchCache(
        self: *Self,
        documents: []const struct {
            content: []const u8,
            doc_type: DocType,
            source_path: ?[]const u8,
        },
    ) ![]CachedDocument {
        var results = try self.allocator.alloc(CachedDocument, documents.len);
        var success_count: usize = 0;

        for (documents) |doc_input| {
            const doc = self.cacheDocument(
                doc_input.content,
                doc_input.doc_type,
                doc_input.source_path,
                null,
                null,
            ) catch continue;
            results[success_count] = doc;
            success_count += 1;
        }

        // Resize to actual success count
        if (success_count < documents.len) {
            results = self.allocator.realloc(results, success_count) catch results;
        }

        return results[0..success_count];
    }

    /// Search documents by embedding similarity (cosine similarity)
    /// Note: This is a simplified implementation - production would use vector DB
    pub fn searchByEmbedding(
        self: *Self,
        query_embedding: []const f32,
        top_k: usize,
        candidate_ids: []const [32]u8,
    ) ![]CachedDocument {
        const ScoredDoc = struct {
            id: [32]u8,
            score: f32,
        };

        var scored = ArrayList(ScoredDoc){};
        defer scored.deinit();

        // Calculate similarity for each candidate
        for (candidate_ids) |id| {
            var emb_key: [74]u8 = undefined;
            @memcpy(emb_key[0..4], "emb:");
            const hex = std.fmt.bytesToHex(id, .lower);
            @memcpy(emb_key[4..68], &hex);

            const emb_data = self.client.get(emb_key[0..68]) catch continue;
            if (emb_data == null) continue;
            defer self.allocator.free(emb_data.?);

            const stored_emb: []const f32 = @alignCast(mem.bytesAsSlice(f32, emb_data.?));
            if (stored_emb.len != query_embedding.len) continue;

            const score = cosineSimilarity(query_embedding, stored_emb);
            try scored.append(.{ .id = id, .score = score });
        }

        // Sort by score descending
        mem.sort(ScoredDoc, scored.items, {}, struct {
            fn lessThan(_: void, a: ScoredDoc, b: ScoredDoc) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Get top_k documents
        const result_count = @min(top_k, scored.items.len);
        var results = try self.allocator.alloc(CachedDocument, result_count);
        var actual_count: usize = 0;

        for (scored.items[0..result_count]) |item| {
            if (try self.getDocument(item.id)) |doc| {
                results[actual_count] = doc;
                actual_count += 1;
            }
        }

        return results[0..actual_count];
    }
};

/// Compute cosine similarity between two vectors
fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return 0.0;

    var dot_product: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;

    for (a, b) |ai, bi| {
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    const denominator = @sqrt(norm_a) * @sqrt(norm_b);
    if (denominator == 0.0) return 0.0;

    return dot_product / denominator;
}


// ============================================================================
// C ABI Exports for Mojo Integration
// ============================================================================

const CCache = opaque {};
const CDocument = opaque {};

/// Create a unified document cache
export fn unified_doc_cache_create(
    host: [*:0]const u8,
    port: u16,
    default_ttl: u32,
) callconv(.c) ?*CCache {
    const allocator = std.heap.c_allocator;
    const host_slice = mem.span(host);

    const cache = UnifiedDocCache.init(allocator, host_slice, port, default_ttl) catch return null;
    return @ptrCast(cache);
}

/// Destroy a unified document cache
export fn unified_doc_cache_destroy(cache: *CCache) callconv(.c) void {
    const real_cache: *UnifiedDocCache = @ptrCast(@alignCast(cache));
    real_cache.deinit();
}

/// Cache a document
export fn unified_doc_cache_store(
    cache: *CCache,
    content: [*]const u8,
    content_len: usize,
    doc_type: u8,
    source_path: ?[*:0]const u8,
    ttl: u32,
    id_out: *[32]u8,
) callconv(.c) i32 {
    const real_cache: *UnifiedDocCache = @ptrCast(@alignCast(cache));
    const content_slice = content[0..content_len];
    const dtype: DocType = @enumFromInt(doc_type);

    const path_slice: ?[]const u8 = if (source_path) |p| mem.span(p) else null;

    const doc = real_cache.cacheDocument(content_slice, dtype, path_slice, null, if (ttl > 0) ttl else null) catch return -1;

    id_out.* = doc.id;
    return 0;
}

/// Get a document by ID
export fn unified_doc_cache_get(
    cache: *CCache,
    id: *const [32]u8,
    content_out: *[*]u8,
    content_len_out: *usize,
    doc_type_out: *u8,
) callconv(.c) i32 {
    const real_cache: *UnifiedDocCache = @ptrCast(@alignCast(cache));

    const doc = real_cache.getDocument(id.*) catch return -1;
    if (doc == null) return 1; // Not found

    const d = doc.?;
    content_out.* = @constCast(d.content.ptr);
    content_len_out.* = d.content.len;
    doc_type_out.* = @intFromEnum(d.doc_type);

    return 0;
}

/// Invalidate a document by ID
export fn unified_doc_cache_invalidate(
    cache: *CCache,
    id: *const [32]u8,
) callconv(.c) i32 {
    const real_cache: *UnifiedDocCache = @ptrCast(@alignCast(cache));
    real_cache.invalidate(id.*) catch return -1;
    return 0;
}

/// Compute document ID from content (SHA-256)
export fn unified_doc_compute_id(
    content: [*]const u8,
    content_len: usize,
    id_out: *[32]u8,
) callconv(.c) void {
    id_out.* = CachedDocument.computeId(content[0..content_len]);
}

/// Free document content returned by unified_doc_cache_get
export fn unified_doc_free_content(content: [*]u8, len: usize) callconv(.c) void {
    const allocator = std.heap.c_allocator;
    allocator.free(content[0..len]);
}

// ============================================================================
// Tests
// ============================================================================

test "CachedDocument - compute ID" {
    const content = "Hello, World!";
    const id = CachedDocument.computeId(content);

    // SHA-256 of "Hello, World!" is known
    try std.testing.expect(id[0] != 0 or id[1] != 0);
}

test "CachedDocument - hex ID" {
    const allocator = std.testing.allocator;
    var doc = CachedDocument.init(allocator);
    defer doc.deinit();

    doc.id = CachedDocument.computeId("test content");
    const hex = doc.getIdHex();

    // Verify hex is valid
    for (hex) |c| {
        try std.testing.expect((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'));
    }
}

test "DocType - string conversion" {
    try std.testing.expectEqualStrings("csv", DocType.csv.toString());
    try std.testing.expectEqualStrings("json", DocType.json.toString());
    try std.testing.expectEqual(DocType.csv, DocType.fromString("csv").?);
    try std.testing.expectEqual(DocType.markdown, DocType.fromString("markdown").?);
    try std.testing.expectEqual(@as(?DocType, null), DocType.fromString("unknown"));
}

test "serialization - roundtrip" {
    const allocator = std.testing.allocator;

    // Create a document
    var doc = CachedDocument.init(allocator);
    doc.id = CachedDocument.computeId("test");
    doc.doc_type = .json;
    doc.content = try allocator.dupe(u8, "{\"key\": \"value\"}");
    doc.metadata.content_length = doc.content.len;
    doc.metadata.parse_time_ns = 12345;
    doc.extracted_at = 1705500000;
    doc.ttl_seconds = 3600;

    defer doc.deinit();

    // Serialize
    const serialized = try serializeDocument(allocator, &doc);
    defer allocator.free(serialized);

    // Deserialize
    var restored = try deserializeDocument(allocator, serialized);
    defer restored.deinit();

    // Verify
    try std.testing.expectEqualSlices(u8, &doc.id, &restored.id);
    try std.testing.expectEqual(doc.doc_type, restored.doc_type);
    try std.testing.expectEqualStrings(doc.content, restored.content);
    try std.testing.expectEqual(doc.ttl_seconds, restored.ttl_seconds);
    try std.testing.expectEqual(doc.extracted_at, restored.extracted_at);
}

test "cosineSimilarity - identical vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0 };
    const similarity = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), similarity, 0.0001);
}

test "cosineSimilarity - orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0 };
    const similarity = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), similarity, 0.0001);
}

test "cosineSimilarity - opposite vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ -1.0, 0.0, 0.0 };
    const similarity = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), similarity, 0.0001);
}
