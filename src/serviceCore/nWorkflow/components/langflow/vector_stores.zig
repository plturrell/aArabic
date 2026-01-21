//! Vector Store Components - Day 30
//! 
//! Integration with Qdrant vector database for semantic search and RAG.
//! Provides nodes for embedding generation, vector storage, and similarity search.
//!
//! Components:
//! - QdrantUpsertNode: Store vectors with metadata
//! - QdrantSearchNode: Semantic similarity search
//! - QdrantCollectionNode: Collection management
//! - EmbeddingNode: Generate embeddings from text
//! - SemanticCacheNode: Semantic caching with vector similarity

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// QDRANT UPSERT NODE
// ============================================================================

/// Upsert vectors into Qdrant collection
pub const QdrantUpsertNode = struct {
    allocator: Allocator,
    collection_name: []const u8,
    connection_url: []const u8,
    batch_size: usize,
    points: std.ArrayList(QdrantPoint),

    pub const QdrantPoint = struct {
        id: []const u8,
        vector: []f32,
        metadata: std.StringHashMap([]const u8),

        pub fn deinit(self: *QdrantPoint, allocator: Allocator) void {
            allocator.free(self.id);
            allocator.free(self.vector);
            var iter = self.metadata.iterator();
            while (iter.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            self.metadata.deinit();
        }
    };

    pub fn init(
        allocator: Allocator,
        collection_name: []const u8,
        connection_url: []const u8,
        batch_size: usize,
    ) !*QdrantUpsertNode {
        const node = try allocator.create(QdrantUpsertNode);
        node.* = QdrantUpsertNode{
            .allocator = allocator,
            .collection_name = try allocator.dupe(u8, collection_name),
            .connection_url = try allocator.dupe(u8, connection_url),
            .batch_size = batch_size,
            .points = std.ArrayList(QdrantPoint){},
        };
        return node;
    }

    pub fn deinit(self: *QdrantUpsertNode) void {
        self.allocator.free(self.collection_name);
        self.allocator.free(self.connection_url);
        for (self.points.items) |*point| {
            point.deinit(self.allocator);
        }
        self.points.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn addPoint(self: *QdrantUpsertNode, id: []const u8, vector: []const f32, metadata: std.StringHashMap([]const u8)) !void {
        const point = QdrantPoint{
            .id = try self.allocator.dupe(u8, id),
            .vector = try self.allocator.dupe(f32, vector),
            .metadata = metadata,
        };
        try self.points.append(self.allocator, point);

        if (self.points.items.len >= self.batch_size) {
            try self.flush();
        }
    }

    pub fn flush(self: *QdrantUpsertNode) !void {
        if (self.points.items.len == 0) return;

        // TODO: Actual Qdrant HTTP API call
        // For now, simulate successful upsert
        for (self.points.items) |*point| {
            point.deinit(self.allocator);
        }
        self.points.clearRetainingCapacity();
    }

    pub fn getPointCount(self: *const QdrantUpsertNode) usize {
        return self.points.items.len;
    }
};

// ============================================================================
// QDRANT SEARCH NODE
// ============================================================================

/// Search for similar vectors in Qdrant
pub const QdrantSearchNode = struct {
    allocator: Allocator,
    collection_name: []const u8,
    connection_url: []const u8,
    top_k: usize,
    score_threshold: f32,

    pub const SearchResult = struct {
        id: []const u8,
        score: f32,
        metadata: std.StringHashMap([]const u8),

        pub fn deinit(self: *SearchResult, allocator: Allocator) void {
            allocator.free(self.id);
            var iter = self.metadata.iterator();
            while (iter.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            self.metadata.deinit();
        }
    };

    pub fn init(
        allocator: Allocator,
        collection_name: []const u8,
        connection_url: []const u8,
        top_k: usize,
        score_threshold: f32,
    ) !*QdrantSearchNode {
        const node = try allocator.create(QdrantSearchNode);
        node.* = QdrantSearchNode{
            .allocator = allocator,
            .collection_name = try allocator.dupe(u8, collection_name),
            .connection_url = try allocator.dupe(u8, connection_url),
            .top_k = top_k,
            .score_threshold = score_threshold,
        };
        return node;
    }

    pub fn deinit(self: *QdrantSearchNode) void {
        self.allocator.free(self.collection_name);
        self.allocator.free(self.connection_url);
        self.allocator.destroy(self);
    }

    pub fn search(self: *const QdrantSearchNode, query_vector: []const f32) !std.ArrayList(SearchResult) {
        var results = std.ArrayList(SearchResult){};

        // TODO: Actual Qdrant HTTP API call
        // For now, return mock results
        var i: usize = 0;
        while (i < @min(self.top_k, 3)) : (i += 1) {
            const score = 0.95 - (@as(f32, @floatFromInt(i)) * 0.1);
            if (score < self.score_threshold) break;

            var metadata = std.StringHashMap([]const u8).init(self.allocator);
            try metadata.put(try self.allocator.dupe(u8, "text"), try self.allocator.dupe(u8, "Mock result"));

            const result = SearchResult{
                .id = try std.fmt.allocPrint(self.allocator, "doc_{d}", .{i}),
                .score = score,
                .metadata = metadata,
            };
            try results.append(self.allocator, result);
        }

        _ = query_vector; // Suppress unused warning
        return results;
    }
};

// ============================================================================
// QDRANT COLLECTION NODE
// ============================================================================

/// Manage Qdrant collections
pub const QdrantCollectionNode = struct {
    allocator: Allocator,
    connection_url: []const u8,

    pub const CollectionConfig = struct {
        name: []const u8,
        vector_size: usize,
        distance: DistanceMetric,

        pub const DistanceMetric = enum {
            cosine,
            euclidean,
            dot_product,

            pub fn toString(self: DistanceMetric) []const u8 {
                return switch (self) {
                    .cosine => "Cosine",
                    .euclidean => "Euclid",
                    .dot_product => "Dot",
                };
            }
        };
    };

    pub fn init(allocator: Allocator, connection_url: []const u8) !*QdrantCollectionNode {
        const node = try allocator.create(QdrantCollectionNode);
        node.* = QdrantCollectionNode{
            .allocator = allocator,
            .connection_url = try allocator.dupe(u8, connection_url),
        };
        return node;
    }

    pub fn deinit(self: *QdrantCollectionNode) void {
        self.allocator.free(self.connection_url);
        self.allocator.destroy(self);
    }

    pub fn createCollection(self: *const QdrantCollectionNode, config: CollectionConfig) !void {
        // TODO: Actual Qdrant HTTP API call
        // For now, simulate creation
        _ = self;
        _ = config;
    }

    pub fn deleteCollection(self: *const QdrantCollectionNode, name: []const u8) !void {
        // TODO: Actual Qdrant HTTP API call
        _ = self;
        _ = name;
    }

    pub fn collectionExists(self: *const QdrantCollectionNode, name: []const u8) !bool {
        // TODO: Actual Qdrant HTTP API call
        _ = self;
        _ = name;
        return true; // Mock response
    }
};

// ============================================================================
// EMBEDDING NODE
// ============================================================================

/// Generate text embeddings
pub const EmbeddingNode = struct {
    allocator: Allocator,
    model: []const u8,
    dimensions: usize,
    normalize: bool,

    pub fn init(
        allocator: Allocator,
        model: []const u8,
        dimensions: usize,
        normalize: bool,
    ) !*EmbeddingNode {
        const node = try allocator.create(EmbeddingNode);
        node.* = EmbeddingNode{
            .allocator = allocator,
            .model = try allocator.dupe(u8, model),
            .dimensions = dimensions,
            .normalize = normalize,
        };
        return node;
    }

    pub fn deinit(self: *EmbeddingNode) void {
        self.allocator.free(self.model);
        self.allocator.destroy(self);
    }

    pub fn embed(self: *const EmbeddingNode, text: []const u8) ![]f32 {
        // TODO: Call nOpenaiServer embedding API
        // For now, generate mock embeddings
        const vector = try self.allocator.alloc(f32, self.dimensions);
        
        // Simple mock: hash text to seed, generate pseudo-random vector
        var hash: u64 = 0;
        for (text) |c| {
            hash = hash *% 31 +% c;
        }

        var i: usize = 0;
        while (i < self.dimensions) : (i += 1) {
            hash = hash *% 1103515245 +% 12345;
            vector[i] = @as(f32, @floatFromInt(hash % 1000)) / 1000.0 - 0.5;
        }

        if (self.normalize) {
            const norm = self.vectorNorm(vector);
            for (vector) |*v| {
                v.* /= norm;
            }
        }

        return vector;
    }

    fn vectorNorm(self: *const EmbeddingNode, vector: []const f32) f32 {
        _ = self;
        var sum: f32 = 0.0;
        for (vector) |v| {
            sum += v * v;
        }
        return @sqrt(sum);
    }
};

// ============================================================================
// SEMANTIC CACHE NODE
// ============================================================================

/// Cache with semantic similarity lookup
pub const SemanticCacheNode = struct {
    allocator: Allocator,
    cache: std.ArrayList(CacheEntry),
    similarity_threshold: f32,
    max_entries: usize,

    pub const CacheEntry = struct {
        query_vector: []f32,
        response: []const u8,
        timestamp: i64,

        pub fn deinit(self: *CacheEntry, allocator: Allocator) void {
            allocator.free(self.query_vector);
            allocator.free(self.response);
        }
    };

    pub fn init(
        allocator: Allocator,
        similarity_threshold: f32,
        max_entries: usize,
    ) !*SemanticCacheNode {
        const node = try allocator.create(SemanticCacheNode);
        node.* = SemanticCacheNode{
            .allocator = allocator,
            .cache = std.ArrayList(CacheEntry){},
            .similarity_threshold = similarity_threshold,
            .max_entries = max_entries,
        };
        return node;
    }

    pub fn deinit(self: *SemanticCacheNode) void {
        for (self.cache.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.cache.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn get(self: *const SemanticCacheNode, query_vector: []const f32) !?[]const u8 {
        var best_similarity: f32 = 0.0;
        var best_response: ?[]const u8 = null;

        for (self.cache.items) |*entry| {
            const similarity = try self.cosineSimilarity(query_vector, entry.query_vector);
            if (similarity > best_similarity and similarity >= self.similarity_threshold) {
                best_similarity = similarity;
                best_response = entry.response;
            }
        }

        return best_response;
    }

    pub fn put(self: *SemanticCacheNode, query_vector: []const f32, response: []const u8) !void {
        // Evict oldest if at capacity
        if (self.cache.items.len >= self.max_entries) {
            var oldest_entry = &self.cache.items[0];
            oldest_entry.deinit(self.allocator);
            _ = self.cache.orderedRemove(0);
        }

        const entry = CacheEntry{
            .query_vector = try self.allocator.dupe(f32, query_vector),
            .response = try self.allocator.dupe(u8, response),
            .timestamp = std.time.timestamp(),
        };
        try self.cache.append(self.allocator, entry);
    }

    fn cosineSimilarity(self: *const SemanticCacheNode, a: []const f32, b: []const f32) !f32 {
        _ = self;
        if (a.len != b.len) return error.VectorSizeMismatch;

        var dot_product: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;

        for (a, b) |a_val, b_val| {
            dot_product += a_val * b_val;
            norm_a += a_val * a_val;
            norm_b += b_val * b_val;
        }

        const denominator = @sqrt(norm_a) * @sqrt(norm_b);
        if (denominator == 0.0) return 0.0;
        return dot_product / denominator;
    }

    pub fn clear(self: *SemanticCacheNode) void {
        for (self.cache.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.cache.clearRetainingCapacity();
    }

    pub fn size(self: *const SemanticCacheNode) usize {
        return self.cache.items.len;
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "QdrantUpsertNode - basic operations" {
    const allocator = std.testing.allocator;

    var node = try QdrantUpsertNode.init(allocator, "test_collection", "http://localhost:6333", 10);
    defer node.deinit();

    try std.testing.expectEqual(@as(usize, 0), node.getPointCount());

    const vector = [_]f32{ 0.1, 0.2, 0.3 };
    var metadata = std.StringHashMap([]const u8).init(allocator);
    try metadata.put(try allocator.dupe(u8, "key"), try allocator.dupe(u8, "value"));

    // Note: addPoint takes ownership of metadata, so we don't defer deinit
    try node.addPoint("id1", &vector, metadata);
    try std.testing.expectEqual(@as(usize, 1), node.getPointCount());
}

test "QdrantSearchNode - search results" {
    const allocator = std.testing.allocator;

    var node = try QdrantSearchNode.init(allocator, "test_collection", "http://localhost:6333", 5, 0.7);
    defer node.deinit();

    const query_vector = [_]f32{ 0.1, 0.2, 0.3 };
    var results = try node.search(&query_vector);
    defer {
        for (results.items) |*result| {
            result.deinit(allocator);
        }
        results.deinit(allocator);
    }

    try std.testing.expect(results.items.len > 0);
    try std.testing.expect(results.items[0].score >= 0.7);
}

test "QdrantCollectionNode - collection management" {
    const allocator = std.testing.allocator;

    var node = try QdrantCollectionNode.init(allocator, "http://localhost:6333");
    defer node.deinit();

    const config = QdrantCollectionNode.CollectionConfig{
        .name = "test",
        .vector_size = 384,
        .distance = .cosine,
    };

    try node.createCollection(config);
    try std.testing.expect(try node.collectionExists("test"));
}

test "EmbeddingNode - generate embeddings" {
    const allocator = std.testing.allocator;

    var node = try EmbeddingNode.init(allocator, "text-embedding-3-small", 384, true);
    defer node.deinit();

    const vector = try node.embed("Hello, world!");
    defer allocator.free(vector);

    try std.testing.expectEqual(@as(usize, 384), vector.len);
    
    // Check normalization (L2 norm should be ~1.0)
    var sum: f32 = 0.0;
    for (vector) |v| {
        sum += v * v;
    }
    const norm = @sqrt(sum);
    try std.testing.expect(@abs(norm - 1.0) < 0.01);
}

test "SemanticCacheNode - cache operations" {
    const allocator = std.testing.allocator;

    var cache = try SemanticCacheNode.init(allocator, 0.9, 100);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 0), cache.size());

    const vector = [_]f32{ 0.1, 0.2, 0.3 };
    try cache.put(&vector, "cached response");
    try std.testing.expectEqual(@as(usize, 1), cache.size());

    const result = try cache.get(&vector);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("cached response", result.?);
}

test "SemanticCacheNode - similarity threshold" {
    const allocator = std.testing.allocator;

    var cache = try SemanticCacheNode.init(allocator, 0.95, 100);
    defer cache.deinit();

    const vector1 = [_]f32{ 1.0, 0.0, 0.0 };
    const vector2 = [_]f32{ 0.9, 0.1, 0.0 }; // Different but similar
    const vector3 = [_]f32{ 0.0, 1.0, 0.0 }; // Very different

    try cache.put(&vector1, "response1");

    // Similar vector should hit cache
    const result1 = try cache.get(&vector2);
    try std.testing.expect(result1 != null);

    // Dissimilar vector should miss cache
    const result2 = try cache.get(&vector3);
    try std.testing.expect(result2 == null);
}

test "SemanticCacheNode - eviction" {
    const allocator = std.testing.allocator;

    var cache = try SemanticCacheNode.init(allocator, 0.9, 2); // Max 2 entries
    defer cache.deinit();

    const vector1 = [_]f32{ 1.0, 0.0, 0.0 };
    const vector2 = [_]f32{ 0.0, 1.0, 0.0 };
    const vector3 = [_]f32{ 0.0, 0.0, 1.0 };

    try cache.put(&vector1, "response1");
    try cache.put(&vector2, "response2");
    try std.testing.expectEqual(@as(usize, 2), cache.size());

    // Adding third should evict first
    try cache.put(&vector3, "response3");
    try std.testing.expectEqual(@as(usize, 2), cache.size());
}

test "EmbeddingNode - consistent embeddings" {
    const allocator = std.testing.allocator;

    var node = try EmbeddingNode.init(allocator, "test-model", 128, false);
    defer node.deinit();

    const text = "Hello, world!";
    const vector1 = try node.embed(text);
    defer allocator.free(vector1);
    
    const vector2 = try node.embed(text);
    defer allocator.free(vector2);

    // Same text should produce same embedding
    for (vector1, vector2) |v1, v2| {
        try std.testing.expectEqual(v1, v2);
    }
}
