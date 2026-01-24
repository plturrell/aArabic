//! Semantic Document Index
//!
//! Combines full-text search (BM25) with vector similarity for hybrid search.
//! Provides efficient document indexing and retrieval with configurable weighting.
//!
//! Features:
//! - BM25-style full-text search with inverted index
//! - SIMD-optimized vector similarity search
//! - Hybrid search with configurable text/vector weighting
//! - C ABI exports for Mojo integration

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const AutoHashMap = std.AutoHashMap;

const unified_doc = @import("../document_cache/unified_doc_cache.zig");
const DocType = unified_doc.DocType;

const hana_cache = @import("../cache/hana/hana_cache.zig");

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
const simdMagnitude = vector_cache.simdMagnitude;
const precomputedCosineSimilarity = vector_cache.precomputedCosineSimilarity;

// ============================================================================
// Types
// ============================================================================

/// Index entry for a document
pub const IndexEntry = struct {
    id: [32]u8, // Document ID
    doc_type: DocType,
    title: []const u8,
    content: []const u8, // Full text content
    tokens: [][]const u8, // Tokenized for full-text search
    embedding: []f32, // Vector embedding
    metadata: []const u8, // JSON metadata
    indexed_at: i64,
    // Precomputed for faster similarity search
    embedding_magnitude: f32,

    pub fn deinit(self: *IndexEntry, allocator: Allocator) void {
        if (self.title.len > 0) allocator.free(self.title);
        if (self.content.len > 0) allocator.free(self.content);
        if (self.metadata.len > 0) allocator.free(self.metadata);
        if (self.embedding.len > 0) allocator.free(self.embedding);
        for (self.tokens) |token| {
            allocator.free(token);
        }
        if (self.tokens.len > 0) allocator.free(self.tokens);
    }
};

/// Search result with scores
pub const SearchResult = struct {
    id: [32]u8,
    score: f32, // Combined/final score
    text_score: f32, // BM25 text score
    vector_score: f32, // Cosine similarity score
    snippet: []const u8, // Relevant text snippet

    pub fn deinit(self: *SearchResult, allocator: Allocator) void {
        if (self.snippet.len > 0) allocator.free(self.snippet);
    }
};

/// Index statistics
pub const IndexStats = struct {
    document_count: u64,
    total_tokens: u64,
    unique_terms: u64,
    avg_doc_length: f32,
    index_size_bytes: u64,
};

/// Configuration for the semantic index
pub const IndexConfig = struct {
    max_documents: u32 = 100_000,
    max_snippet_length: u32 = 256,
    bm25_k1: f32 = 1.2, // Term frequency saturation
    bm25_b: f32 = 0.75, // Document length normalization
    default_text_weight: f32 = 0.5,
    embedding_dims: u32 = 768,
};

/// Posting in the inverted index: (document index, positions)
const Posting = struct {
    doc_idx: u32,
    positions: []u32,
    term_frequency: u32,
};

/// Posting list for a term
const PostingList = struct {
    postings: ArrayList(Posting),
    document_frequency: u32,

    pub fn init(allocator: Allocator) PostingList {
        return .{
            .postings = ArrayList(Posting).init(allocator),
            .document_frequency = 0,
        };
    }

    pub fn deinit(self: *PostingList, allocator: Allocator) void {
        for (self.postings.items) |*posting| {
            allocator.free(posting.positions);
        }
        self.postings.deinit();
    }
};

/// Semantic index error types
pub const SemanticIndexError = error{
    IndexFull,
    DocumentNotFound,
    InvalidEmbedding,
    DimensionMismatch,
    TokenizationFailed,
    OutOfMemory,
    AllocationFailed,
};

// ============================================================================
// Tokenizer
// ============================================================================

/// Simple tokenizer: splits on whitespace, lowercases, removes punctuation
pub fn tokenize(allocator: Allocator, text: []const u8) ![][]const u8 {
    var tokens = ArrayList([]const u8).init(allocator);
    errdefer {
        for (tokens.items) |token| allocator.free(token);
        tokens.deinit();
    }

    var start: usize = 0;
    var in_word = false;

    for (text, 0..) |c, i| {
        const is_word_char = isWordChar(c);

        if (is_word_char and !in_word) {
            start = i;
            in_word = true;
        } else if (!is_word_char and in_word) {
            const token = try normalizeToken(allocator, text[start..i]);
            if (token.len > 0) {
                try tokens.append(token);
            } else {
                allocator.free(token);
            }
            in_word = false;
        }
    }

    // Handle last token
    if (in_word) {
        const token = try normalizeToken(allocator, text[start..]);
        if (token.len > 0) {
            try tokens.append(token);
        } else {
            allocator.free(token);
        }
    }

    return tokens.toOwnedSlice();
}

fn isWordChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or
        (c >= 'A' and c <= 'Z') or
        (c >= '0' and c <= '9') or
        c == '_' or c == '-' or c == '\'';
}

fn normalizeToken(allocator: Allocator, token: []const u8) ![]u8 {
    // Lowercase and remove leading/trailing punctuation
    var start: usize = 0;
    var end: usize = token.len;

    // Strip leading non-alphanumeric
    while (start < end and !isAlphanumeric(token[start])) : (start += 1) {}
    // Strip trailing non-alphanumeric
    while (end > start and !isAlphanumeric(token[end - 1])) : (end -= 1) {}

    if (start >= end) {
        return try allocator.alloc(u8, 0);
    }

    const result = try allocator.alloc(u8, end - start);
    for (token[start..end], 0..) |c, i| {
        result[i] = std.ascii.toLower(c);
    }
    return result;
}

fn isAlphanumeric(c: u8) bool {
    return (c >= 'a' and c <= 'z') or
        (c >= 'A' and c <= 'Z') or
        (c >= '0' and c <= '9');
}

// ============================================================================
// Semantic Index
// ============================================================================

pub const SemanticIndex = struct {
    allocator: Allocator,
    config: IndexConfig,

    // Document storage
    documents: ArrayList(IndexEntry),
    id_to_idx: AutoHashMap([32]u8, u32),

    // Inverted index for full-text search
    inverted_index: StringHashMap(PostingList),
    total_tokens: u64,

    pub fn init(allocator: Allocator, config: IndexConfig) SemanticIndex {
        return .{
            .allocator = allocator,
            .config = config,
            .documents = ArrayList(IndexEntry).init(allocator),
            .id_to_idx = AutoHashMap([32]u8, u32).init(allocator),
            .inverted_index = StringHashMap(PostingList).init(allocator),
            .total_tokens = 0,
        };
    }

    pub fn deinit(self: *SemanticIndex) void {
        // Free documents
        for (self.documents.items) |*doc| {
            doc.deinit(self.allocator);
        }
        self.documents.deinit();
        self.id_to_idx.deinit();

        // Free inverted index
        var it = self.inverted_index.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.inverted_index.deinit();
    }

    /// Add a document to the index
    pub fn add(self: *SemanticIndex, entry: IndexEntry) !void {
        if (self.documents.items.len >= self.config.max_documents) {
            return SemanticIndexError.IndexFull;
        }

        // Check if document already exists
        if (self.id_to_idx.get(entry.id)) |_| {
            // Remove old entry first
            try self.remove(entry.id);
        }

        const doc_idx: u32 = @intCast(self.documents.items.len);

        // Clone entry data
        var new_entry = IndexEntry{
            .id = entry.id,
            .doc_type = entry.doc_type,
            .title = try self.allocator.dupe(u8, entry.title),
            .content = try self.allocator.dupe(u8, entry.content),
            .tokens = undefined,
            .embedding = try self.allocator.dupe(f32, entry.embedding),
            .metadata = try self.allocator.dupe(u8, entry.metadata),
            .indexed_at = entry.indexed_at,
            .embedding_magnitude = simdMagnitude(entry.embedding),
        };

        // Tokenize content
        const tokens = try tokenize(self.allocator, entry.content);
        new_entry.tokens = tokens;
        self.total_tokens += tokens.len;

        // Update inverted index
        try self.indexTokens(doc_idx, tokens);

        try self.documents.append(new_entry);
        try self.id_to_idx.put(entry.id, doc_idx);
    }

    fn indexTokens(self: *SemanticIndex, doc_idx: u32, tokens: [][]const u8) !void {
        // Track positions for each term
        var term_positions = StringHashMap(ArrayList(u32)).init(self.allocator);
        defer {
            var it = term_positions.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            term_positions.deinit();
        }

        for (tokens, 0..) |token, pos| {
            const gop = try term_positions.getOrPut(token);
            if (!gop.found_existing) {
                gop.value_ptr.* = ArrayList(u32).init(self.allocator);
            }
            try gop.value_ptr.append(@intCast(pos));
        }

        // Add to inverted index
        var it = term_positions.iterator();
        while (it.next()) |entry| {
            const term = entry.key_ptr.*;
            const positions = entry.value_ptr.*;

            const gop = try self.inverted_index.getOrPut(term);
            if (!gop.found_existing) {
                const owned_term = try self.allocator.dupe(u8, term);
                gop.key_ptr.* = owned_term;
                gop.value_ptr.* = PostingList.init(self.allocator);
            }

            const posting = Posting{
                .doc_idx = doc_idx,
                .positions = try positions.toOwnedSlice(),
                .term_frequency = @intCast(positions.items.len),
            };
            try gop.value_ptr.postings.append(posting);
            gop.value_ptr.document_frequency += 1;
        }
    }

    /// Remove a document from the index
    pub fn remove(self: *SemanticIndex, id: [32]u8) !void {
        const idx = self.id_to_idx.get(id) orelse return SemanticIndexError.DocumentNotFound;

        // Remove from inverted index
        const doc = &self.documents.items[idx];
        for (doc.tokens) |token| {
            if (self.inverted_index.getPtr(token)) |posting_list| {
                // Find and remove posting for this document
                var i: usize = 0;
                while (i < posting_list.postings.items.len) {
                    if (posting_list.postings.items[i].doc_idx == idx) {
                        self.allocator.free(posting_list.postings.items[i].positions);
                        _ = posting_list.postings.swapRemove(i);
                        posting_list.document_frequency -|= 1;
                    } else {
                        i += 1;
                    }
                }
            }
        }

        self.total_tokens -|= doc.tokens.len;
        doc.deinit(self.allocator);

        // Swap remove from documents array
        const last_idx = self.documents.items.len - 1;
        if (idx != last_idx) {
            const last_doc = &self.documents.items[last_idx];
            self.documents.items[idx] = last_doc.*;
            try self.id_to_idx.put(last_doc.id, idx);

            // Update inverted index for moved document
            for (self.documents.items[idx].tokens) |token| {
                if (self.inverted_index.getPtr(token)) |posting_list| {
                    for (posting_list.postings.items) |*posting| {
                        if (posting.doc_idx == @as(u32, @intCast(last_idx))) {
                            posting.doc_idx = idx;
                        }
                    }
                }
            }
        }

        _ = self.documents.pop();
        _ = self.id_to_idx.remove(id);
    }

    /// Full-text search using BM25 scoring
    pub fn searchFullText(self: *SemanticIndex, query: []const u8, top_k: u32) ![]SearchResult {
        const query_tokens = try tokenize(self.allocator, query);
        defer {
            for (query_tokens) |token| self.allocator.free(token);
            self.allocator.free(query_tokens);
        }

        if (query_tokens.len == 0 or self.documents.items.len == 0) {
            return try self.allocator.alloc(SearchResult, 0);
        }

        // Calculate BM25 scores for all documents
        const n_docs = self.documents.items.len;
        const avg_doc_len = if (n_docs > 0)
            @as(f32, @floatFromInt(self.total_tokens)) / @as(f32, @floatFromInt(n_docs))
        else
            0.0;

        var scores = try self.allocator.alloc(f32, n_docs);
        defer self.allocator.free(scores);
        @memset(scores, 0.0);

        for (query_tokens) |term| {
            if (self.inverted_index.get(term)) |posting_list| {
                // IDF = log((N - df + 0.5) / (df + 0.5) + 1)
                const df = @as(f32, @floatFromInt(posting_list.document_frequency));
                const n = @as(f32, @floatFromInt(n_docs));
                const idf = @log((n - df + 0.5) / (df + 0.5) + 1.0);

                for (posting_list.postings.items) |posting| {
                    const doc = &self.documents.items[posting.doc_idx];
                    const doc_len = @as(f32, @floatFromInt(doc.tokens.len));
                    const tf = @as(f32, @floatFromInt(posting.term_frequency));

                    // BM25 score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
                    const k1 = self.config.bm25_k1;
                    const b = self.config.bm25_b;
                    const numerator = tf * (k1 + 1.0);
                    const denominator = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len);
                    const term_score = idf * numerator / denominator;

                    scores[posting.doc_idx] += term_score;
                }
            }
        }

        return try self.collectTopK(scores, null, top_k, query);
    }

    /// Vector similarity search using cosine similarity
    pub fn searchVector(self: *SemanticIndex, embedding: []const f32, top_k: u32) ![]SearchResult {
        if (self.documents.items.len == 0) {
            return try self.allocator.alloc(SearchResult, 0);
        }

        const query_magnitude = simdMagnitude(embedding);
        var scores = try self.allocator.alloc(f32, self.documents.items.len);
        defer self.allocator.free(scores);

        for (self.documents.items, 0..) |*doc, i| {
            if (doc.embedding.len == embedding.len) {
                scores[i] = precomputedCosineSimilarity(
                    embedding,
                    query_magnitude,
                    doc.embedding,
                    doc.embedding_magnitude,
                );
            } else {
                scores[i] = 0.0;
            }
        }

        return try self.collectTopK(null, scores, top_k, "");
    }

    /// Hybrid search combining text and vector scores
    pub fn searchHybrid(
        self: *SemanticIndex,
        query: []const u8,
        embedding: []const f32,
        top_k: u32,
        text_weight: f32,
    ) ![]SearchResult {
        if (self.documents.items.len == 0) {
            return try self.allocator.alloc(SearchResult, 0);
        }

        const query_tokens = try tokenize(self.allocator, query);
        defer {
            for (query_tokens) |token| self.allocator.free(token);
            self.allocator.free(query_tokens);
        }

        const n_docs = self.documents.items.len;
        const avg_doc_len = if (n_docs > 0)
            @as(f32, @floatFromInt(self.total_tokens)) / @as(f32, @floatFromInt(n_docs))
        else
            0.0;

        var text_scores = try self.allocator.alloc(f32, n_docs);
        defer self.allocator.free(text_scores);
        @memset(text_scores, 0.0);

        var vector_scores = try self.allocator.alloc(f32, n_docs);
        defer self.allocator.free(vector_scores);

        // Calculate BM25 text scores
        for (query_tokens) |term| {
            if (self.inverted_index.get(term)) |posting_list| {
                const df = @as(f32, @floatFromInt(posting_list.document_frequency));
                const n = @as(f32, @floatFromInt(n_docs));
                const idf = @log((n - df + 0.5) / (df + 0.5) + 1.0);

                for (posting_list.postings.items) |posting| {
                    const doc = &self.documents.items[posting.doc_idx];
                    const doc_len = @as(f32, @floatFromInt(doc.tokens.len));
                    const tf = @as(f32, @floatFromInt(posting.term_frequency));

                    const k1 = self.config.bm25_k1;
                    const b = self.config.bm25_b;
                    const numerator = tf * (k1 + 1.0);
                    const denominator = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len);
                    const term_score = idf * numerator / denominator;

                    text_scores[posting.doc_idx] += term_score;
                }
            }
        }

        // Calculate vector scores
        const query_magnitude = simdMagnitude(embedding);
        for (self.documents.items, 0..) |*doc, i| {
            if (doc.embedding.len == embedding.len) {
                vector_scores[i] = precomputedCosineSimilarity(
                    embedding,
                    query_magnitude,
                    doc.embedding,
                    doc.embedding_magnitude,
                );
            } else {
                vector_scores[i] = 0.0;
            }
        }

        // Normalize scores to [0, 1] range
        const max_text = maxScore(text_scores);
        const max_vector = maxScore(vector_scores);

        if (max_text > 0.0) {
            for (text_scores) |*s| s.* /= max_text;
        }
        if (max_vector > 0.0) {
            for (vector_scores) |*s| s.* /= max_vector;
        }

        // Combine scores
        var hybrid_scores = try self.allocator.alloc(f32, n_docs);
        defer self.allocator.free(hybrid_scores);

        const vector_weight = 1.0 - text_weight;
        for (0..n_docs) |i| {
            hybrid_scores[i] = text_weight * text_scores[i] + vector_weight * vector_scores[i];
        }

        return try self.collectTopKHybrid(text_scores, vector_scores, hybrid_scores, top_k, query);
    }

    /// Get index statistics
    pub fn getStats(self: *const SemanticIndex) IndexStats {
        const doc_count = self.documents.items.len;
        const avg_len = if (doc_count > 0)
            @as(f32, @floatFromInt(self.total_tokens)) / @as(f32, @floatFromInt(doc_count))
        else
            0.0;

        // Estimate index size
        var index_size: u64 = 0;
        index_size += @sizeOf(SemanticIndex);
        index_size += doc_count * @sizeOf(IndexEntry);

        for (self.documents.items) |doc| {
            index_size += doc.content.len;
            index_size += doc.title.len;
            index_size += doc.metadata.len;
            index_size += doc.embedding.len * @sizeOf(f32);
            for (doc.tokens) |token| {
                index_size += token.len;
            }
        }

        var it = self.inverted_index.iterator();
        while (it.next()) |entry| {
            index_size += entry.key_ptr.len;
            index_size += entry.value_ptr.postings.items.len * @sizeOf(Posting);
        }

        return IndexStats{
            .document_count = @intCast(doc_count),
            .total_tokens = self.total_tokens,
            .unique_terms = @intCast(self.inverted_index.count()),
            .avg_doc_length = avg_len,
            .index_size_bytes = index_size,
        };
    }

    // Helper functions

    fn maxScore(scores: []const f32) f32 {
        var max: f32 = 0.0;
        for (scores) |s| {
            if (s > max) max = s;
        }
        return max;
    }

    fn collectTopK(
        self: *SemanticIndex,
        text_scores: ?[]const f32,
        vector_scores: ?[]const f32,
        top_k: u32,
        query: []const u8,
    ) ![]SearchResult {
        const n = self.documents.items.len;
        const k = @min(top_k, @as(u32, @intCast(n)));

        // Simple top-k using partial sort
        var indices = try self.allocator.alloc(u32, n);
        defer self.allocator.free(indices);
        for (0..n) |i| indices[i] = @intCast(i);

        const scores = text_scores orelse vector_scores.?;

        // Partial sort to get top k
        for (0..k) |i| {
            var max_idx = i;
            for (i + 1..n) |j| {
                if (scores[indices[j]] > scores[indices[max_idx]]) {
                    max_idx = j;
                }
            }
            const tmp = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = tmp;
        }

        var results = try self.allocator.alloc(SearchResult, k);
        errdefer self.allocator.free(results);

        for (0..k) |i| {
            const idx = indices[i];
            const doc = &self.documents.items[idx];

            results[i] = SearchResult{
                .id = doc.id,
                .score = scores[idx],
                .text_score = if (text_scores) |ts| ts[idx] else 0.0,
                .vector_score = if (vector_scores) |vs| vs[idx] else 0.0,
                .snippet = try self.generateSnippet(doc, query),
            };
        }

        return results;
    }

    fn collectTopKHybrid(
        self: *SemanticIndex,
        text_scores: []const f32,
        vector_scores: []const f32,
        hybrid_scores: []const f32,
        top_k: u32,
        query: []const u8,
    ) ![]SearchResult {
        const n = self.documents.items.len;
        const k = @min(top_k, @as(u32, @intCast(n)));

        var indices = try self.allocator.alloc(u32, n);
        defer self.allocator.free(indices);
        for (0..n) |i| indices[i] = @intCast(i);

        for (0..k) |i| {
            var max_idx = i;
            for (i + 1..n) |j| {
                if (hybrid_scores[indices[j]] > hybrid_scores[indices[max_idx]]) {
                    max_idx = j;
                }
            }
            const tmp = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = tmp;
        }

        var results = try self.allocator.alloc(SearchResult, k);
        errdefer self.allocator.free(results);

        for (0..k) |i| {
            const idx = indices[i];
            const doc = &self.documents.items[idx];

            results[i] = SearchResult{
                .id = doc.id,
                .score = hybrid_scores[idx],
                .text_score = text_scores[idx],
                .vector_score = vector_scores[idx],
                .snippet = try self.generateSnippet(doc, query),
            };
        }

        return results;
    }

    fn generateSnippet(self: *SemanticIndex, doc: *const IndexEntry, query: []const u8) ![]const u8 {
        const max_len = self.config.max_snippet_length;
        if (doc.content.len <= max_len) {
            return try self.allocator.dupe(u8, doc.content);
        }

        // Find best matching position based on query terms
        if (query.len > 0) {
            const query_tokens = tokenize(self.allocator, query) catch
                return try self.allocator.dupe(u8, doc.content[0..max_len]);
            defer {
                for (query_tokens) |token| self.allocator.free(token);
                self.allocator.free(query_tokens);
            }

            if (query_tokens.len > 0) {
                // Find first occurrence of any query term
                for (doc.content, 0..) |c, i| {
                    _ = c;
                    for (query_tokens) |term| {
                        if (i + term.len <= doc.content.len) {
                            var match = true;
                            for (term, 0..) |tc, j| {
                                if (std.ascii.toLower(doc.content[i + j]) != tc) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match) {
                                const start = if (i > max_len / 4) i - max_len / 4 else 0;
                                const end = @min(start + max_len, doc.content.len);
                                return try self.allocator.dupe(u8, doc.content[start..end]);
                            }
                        }
                    }
                }
            }
        }

        // Default: return start of document
        return try self.allocator.dupe(u8, doc.content[0..max_len]);
    }
};

// ============================================================================
// C ABI Exports
// ============================================================================

/// Opaque type for C ABI
pub const CSemanticIndex = opaque {};
pub const CSearchResult = extern struct {
    id: [32]u8,
    score: f32,
    text_score: f32,
    vector_score: f32,
    snippet: [*]const u8,
    snippet_len: u32,
};

var global_allocator: ?Allocator = null;

fn getGlobalAllocator() Allocator {
    return global_allocator orelse std.heap.page_allocator;
}

/// Create a new semantic index
export fn semantic_index_create() callconv(.c) ?*CSemanticIndex {
    const allocator = getGlobalAllocator();
    const index = allocator.create(SemanticIndex) catch return null;
    index.* = SemanticIndex.init(allocator, IndexConfig{});
    return @ptrCast(index);
}

/// Create semantic index with custom config
export fn semantic_index_create_with_config(
    max_documents: u32,
    embedding_dims: u32,
    bm25_k1: f32,
    bm25_b: f32,
    text_weight: f32,
) callconv(.c) ?*CSemanticIndex {
    const allocator = getGlobalAllocator();
    const index = allocator.create(SemanticIndex) catch return null;
    index.* = SemanticIndex.init(allocator, IndexConfig{
        .max_documents = max_documents,
        .embedding_dims = embedding_dims,
        .bm25_k1 = bm25_k1,
        .bm25_b = bm25_b,
        .default_text_weight = text_weight,
    });
    return @ptrCast(index);
}

/// Add document to index
export fn semantic_index_add(
    index_ptr: *CSemanticIndex,
    id: *const [32]u8,
    doc_type: u8,
    title: [*]const u8,
    title_len: u32,
    content: [*]const u8,
    content_len: u32,
    embedding: [*]const f32,
    dims: u32,
    metadata: [*]const u8,
    metadata_len: u32,
) callconv(.c) i32 {
    const index: *SemanticIndex = @ptrCast(@alignCast(index_ptr));

    const entry = IndexEntry{
        .id = id.*,
        .doc_type = @enumFromInt(doc_type),
        .title = title[0..title_len],
        .content = content[0..content_len],
        .tokens = &[_][]const u8{},
        .embedding = @constCast(embedding[0..dims]),
        .metadata = metadata[0..metadata_len],
        .indexed_at = std.time.timestamp(),
        .embedding_magnitude = 0.0,
    };

    index.add(entry) catch return -1;
    return 0;
}

/// Remove document from index
export fn semantic_index_remove(
    index_ptr: *CSemanticIndex,
    id: *const [32]u8,
) callconv(.c) i32 {
    const index: *SemanticIndex = @ptrCast(@alignCast(index_ptr));
    index.remove(id.*) catch return -1;
    return 0;
}

/// Full-text search
export fn semantic_index_search_text(
    index_ptr: *CSemanticIndex,
    query: [*]const u8,
    query_len: u32,
    top_k: u32,
    results: [*]CSearchResult,
    results_count: *u32,
) callconv(.c) i32 {
    const index: *SemanticIndex = @ptrCast(@alignCast(index_ptr));
    const search_results = index.searchFullText(query[0..query_len], top_k) catch return -1;
    defer {
        for (search_results) |*r| {
            var result = r.*;
            result.deinit(index.allocator);
        }
        index.allocator.free(search_results);
    }

    for (search_results, 0..) |r, i| {
        results[i] = CSearchResult{
            .id = r.id,
            .score = r.score,
            .text_score = r.text_score,
            .vector_score = r.vector_score,
            .snippet = r.snippet.ptr,
            .snippet_len = @intCast(r.snippet.len),
        };
    }
    results_count.* = @intCast(search_results.len);
    return 0;
}

/// Vector similarity search
export fn semantic_index_search_vector(
    index_ptr: *CSemanticIndex,
    embedding: [*]const f32,
    dims: u32,
    top_k: u32,
    results: [*]CSearchResult,
    results_count: *u32,
) callconv(.c) i32 {
    const index: *SemanticIndex = @ptrCast(@alignCast(index_ptr));
    const search_results = index.searchVector(embedding[0..dims], top_k) catch return -1;
    defer {
        for (search_results) |*r| {
            var result = r.*;
            result.deinit(index.allocator);
        }
        index.allocator.free(search_results);
    }

    for (search_results, 0..) |r, i| {
        results[i] = CSearchResult{
            .id = r.id,
            .score = r.score,
            .text_score = r.text_score,
            .vector_score = r.vector_score,
            .snippet = r.snippet.ptr,
            .snippet_len = @intCast(r.snippet.len),
        };
    }
    results_count.* = @intCast(search_results.len);
    return 0;
}

/// Hybrid search
export fn semantic_index_search_hybrid(
    index_ptr: *CSemanticIndex,
    query: [*]const u8,
    query_len: u32,
    embedding: [*]const f32,
    dims: u32,
    top_k: u32,
    text_weight: f32,
    results: [*]CSearchResult,
    results_count: *u32,
) callconv(.c) i32 {
    const index: *SemanticIndex = @ptrCast(@alignCast(index_ptr));
    const search_results = index.searchHybrid(
        query[0..query_len],
        embedding[0..dims],
        top_k,
        text_weight,
    ) catch return -1;
    defer {
        for (search_results) |*r| {
            var result = r.*;
            result.deinit(index.allocator);
        }
        index.allocator.free(search_results);
    }

    for (search_results, 0..) |r, i| {
        results[i] = CSearchResult{
            .id = r.id,
            .score = r.score,
            .text_score = r.text_score,
            .vector_score = r.vector_score,
            .snippet = r.snippet.ptr,
            .snippet_len = @intCast(r.snippet.len),
        };
    }
    results_count.* = @intCast(search_results.len);
    return 0;
}

/// Get index statistics
export fn semantic_index_get_stats(
    index_ptr: *CSemanticIndex,
    document_count: *u64,
    total_tokens: *u64,
    unique_terms: *u64,
    avg_doc_length: *f32,
    index_size_bytes: *u64,
) callconv(.c) i32 {
    const index: *SemanticIndex = @ptrCast(@alignCast(index_ptr));
    const stats = index.getStats();
    document_count.* = stats.document_count;
    total_tokens.* = stats.total_tokens;
    unique_terms.* = stats.unique_terms;
    avg_doc_length.* = stats.avg_doc_length;
    index_size_bytes.* = stats.index_size_bytes;
    return 0;
}

/// Destroy semantic index
export fn semantic_index_destroy(index_ptr: *CSemanticIndex) callconv(.c) void {
    const index: *SemanticIndex = @ptrCast(@alignCast(index_ptr));
    index.deinit();
    getGlobalAllocator().destroy(index);
}

/// Set global allocator (call before creating index)
export fn semantic_index_set_allocator(allocator_ptr: ?*anyopaque) callconv(.c) void {
    _ = allocator_ptr;
    global_allocator = std.heap.page_allocator;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "tokenize - basic" {
    const allocator = std.testing.allocator;

    const tokens = try tokenize(allocator, "Hello, World! This is a TEST.");
    defer {
        for (tokens) |token| allocator.free(token);
        allocator.free(tokens);
    }

    try std.testing.expectEqual(@as(usize, 6), tokens.len);
    try std.testing.expectEqualStrings("hello", tokens[0]);
    try std.testing.expectEqualStrings("world", tokens[1]);
    try std.testing.expectEqualStrings("this", tokens[2]);
    try std.testing.expectEqualStrings("is", tokens[3]);
    try std.testing.expectEqualStrings("a", tokens[4]);
    try std.testing.expectEqualStrings("test", tokens[5]);
}

test "tokenize - punctuation handling" {
    const allocator = std.testing.allocator;

    const tokens = try tokenize(allocator, "don't worry, it's fine!");
    defer {
        for (tokens) |token| allocator.free(token);
        allocator.free(tokens);
    }

    try std.testing.expectEqual(@as(usize, 4), tokens.len);
    try std.testing.expectEqualStrings("don't", tokens[0]);
    try std.testing.expectEqualStrings("worry", tokens[1]);
    try std.testing.expectEqualStrings("it's", tokens[2]);
    try std.testing.expectEqualStrings("fine", tokens[3]);
}

test "tokenize - empty string" {
    const allocator = std.testing.allocator;

    const tokens = try tokenize(allocator, "");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "SemanticIndex - add and search" {
    const allocator = std.testing.allocator;

    var index = SemanticIndex.init(allocator, IndexConfig{});
    defer index.deinit();

    // Create test embeddings
    var embedding1 = [_]f32{1.0} ** 8;
    var embedding2 = [_]f32{0.5} ** 8;

    const entry1 = IndexEntry{
        .id = [_]u8{1} ** 32,
        .doc_type = .markdown,
        .title = "Test Document One",
        .content = "The quick brown fox jumps over the lazy dog",
        .tokens = &[_][]const u8{},
        .embedding = &embedding1,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    };

    const entry2 = IndexEntry{
        .id = [_]u8{2} ** 32,
        .doc_type = .markdown,
        .title = "Test Document Two",
        .content = "A lazy cat sleeps on the warm sunny windowsill",
        .tokens = &[_][]const u8{},
        .embedding = &embedding2,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    };

    try index.add(entry1);
    try index.add(entry2);

    // Test full-text search
    const text_results = try index.searchFullText("lazy", 2);
    defer {
        for (text_results) |*r| {
            var result = r.*;
            result.deinit(allocator);
        }
        allocator.free(text_results);
    }

    try std.testing.expect(text_results.len == 2);
    try std.testing.expect(text_results[0].text_score > 0);
}

test "BM25 scoring - basic" {
    const allocator = std.testing.allocator;

    var index = SemanticIndex.init(allocator, IndexConfig{});
    defer index.deinit();

    var embedding1 = [_]f32{1.0} ** 4;
    var embedding2 = [_]f32{1.0} ** 4;
    var embedding3 = [_]f32{1.0} ** 4;

    // Document with multiple occurrences of search term
    try index.add(IndexEntry{
        .id = [_]u8{1} ** 32,
        .doc_type = .markdown,
        .title = "Doc1",
        .content = "machine learning is about machine intelligence",
        .tokens = &[_][]const u8{},
        .embedding = &embedding1,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    });

    // Document with single occurrence
    try index.add(IndexEntry{
        .id = [_]u8{2} ** 32,
        .doc_type = .markdown,
        .title = "Doc2",
        .content = "artificial intelligence and deep learning",
        .tokens = &[_][]const u8{},
        .embedding = &embedding2,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    });

    // Document without search term
    try index.add(IndexEntry{
        .id = [_]u8{3} ** 32,
        .doc_type = .markdown,
        .title = "Doc3",
        .content = "cats and dogs are pets",
        .tokens = &[_][]const u8{},
        .embedding = &embedding3,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    });

    const results = try index.searchFullText("machine", 3);
    defer {
        for (results) |*r| {
            var result = r.*;
            result.deinit(allocator);
        }
        allocator.free(results);
    }

    // Doc with "machine" twice should rank highest
    try std.testing.expect(results.len == 3);
    try std.testing.expectEqual([_]u8{1} ** 32, results[0].id);
    try std.testing.expect(results[0].text_score > results[1].text_score);
}

test "hybrid search - combined scoring" {
    const allocator = std.testing.allocator;

    var index = SemanticIndex.init(allocator, IndexConfig{});
    defer index.deinit();

    // Similar text, different embeddings
    var embedding1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var embedding2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    var query_embedding = [_]f32{ 0.9, 0.1, 0.0, 0.0 };

    try index.add(IndexEntry{
        .id = [_]u8{1} ** 32,
        .doc_type = .json,
        .title = "Doc1",
        .content = "neural network deep learning",
        .tokens = &[_][]const u8{},
        .embedding = &embedding1,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    });

    try index.add(IndexEntry{
        .id = [_]u8{2} ** 32,
        .doc_type = .json,
        .title = "Doc2",
        .content = "neural network machine learning",
        .tokens = &[_][]const u8{},
        .embedding = &embedding2,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    });

    // Hybrid search should combine both signals
    const results = try index.searchHybrid("neural", &query_embedding, 2, 0.5);
    defer {
        for (results) |*r| {
            var result = r.*;
            result.deinit(allocator);
        }
        allocator.free(results);
    }

    try std.testing.expect(results.len == 2);
    // Both docs have same text score for "neural", but doc1 has higher vector similarity
    try std.testing.expect(results[0].vector_score >= results[1].vector_score);
}

test "index stats" {
    const allocator = std.testing.allocator;

    var index = SemanticIndex.init(allocator, IndexConfig{});
    defer index.deinit();

    var embedding = [_]f32{1.0} ** 4;

    try index.add(IndexEntry{
        .id = [_]u8{1} ** 32,
        .doc_type = .csv,
        .title = "Test",
        .content = "one two three four five",
        .tokens = &[_][]const u8{},
        .embedding = &embedding,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    });

    const stats = index.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.document_count);
    try std.testing.expectEqual(@as(u64, 5), stats.total_tokens);
    try std.testing.expectEqual(@as(u64, 5), stats.unique_terms);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), stats.avg_doc_length, 0.01);
}

test "remove document" {
    const allocator = std.testing.allocator;

    var index = SemanticIndex.init(allocator, IndexConfig{});
    defer index.deinit();

    var embedding = [_]f32{1.0} ** 4;

    try index.add(IndexEntry{
        .id = [_]u8{1} ** 32,
        .doc_type = .html,
        .title = "Test",
        .content = "hello world",
        .tokens = &[_][]const u8{},
        .embedding = &embedding,
        .metadata = "{}",
        .indexed_at = 0,
        .embedding_magnitude = 0,
    });

    try std.testing.expectEqual(@as(u64, 1), index.getStats().document_count);

    try index.remove([_]u8{1} ** 32);
    try std.testing.expectEqual(@as(u64, 0), index.getStats().document_count);
}

