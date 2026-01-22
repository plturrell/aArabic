// Tokenizer Profiler - Specialized Profiling for Tokenization Performance
// Tracks vocabulary lookup efficiency, BPE merge operations, and tokenization bottlenecks

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const TokenizerProfileConfig = struct {
    track_lookups: bool = true,
    track_merges: bool = true,
    track_cache_hits: bool = true,
    sample_rate: u32 = 1, // Track all operations by default
};

pub const LookupStats = struct {
    operation: []const u8, // "vocab_lookup", "trie_lookup", "linear_search"
    token: []const u8,
    duration_ns: i64,
    cache_hit: bool,
    vocab_size: usize,
};

pub const MergeStats = struct {
    pattern: []const u8,
    duration_ns: i64,
    merge_count: u32,
    cached: bool,
};

pub const TokenizationSession = struct {
    text: []const u8,
    total_duration_ns: i64,
    token_count: u32,
    lookup_count: u32,
    merge_count: u32,
    cache_hit_rate: f32,
    timestamp_ns: i64,
};

pub const TokenizerProfile = struct {
    lookups: std.ArrayList(LookupStats),
    merges: std.ArrayList(MergeStats),
    sessions: std.ArrayList(TokenizationSession),
    
    // Aggregate statistics
    total_lookups: u64,
    total_merges: u64,
    total_cache_hits: u64,
    total_duration_ns: i64,
    
    // Vocabulary efficiency metrics
    vocab_size: usize,
    trie_depth: u32,
    avg_lookup_time_ns: f64,
    avg_merge_time_ns: f64,
    
    allocator: Allocator,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: Allocator) TokenizerProfile {
        return .{
            .lookups = std.ArrayList(LookupStats).init(allocator),
            .merges = std.ArrayList(MergeStats).init(allocator),
            .sessions = std.ArrayList(TokenizationSession).init(allocator),
            .total_lookups = 0,
            .total_merges = 0,
            .total_cache_hits = 0,
            .total_duration_ns = 0,
            .vocab_size = 0,
            .trie_depth = 0,
            .avg_lookup_time_ns = 0.0,
            .avg_merge_time_ns = 0.0,
            .allocator = allocator,
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *TokenizerProfile) void {
        for (self.lookups.items) |*lookup| {
            self.allocator.free(lookup.operation);
            self.allocator.free(lookup.token);
        }
        for (self.merges.items) |*merge| {
            self.allocator.free(merge.pattern);
        }
        for (self.sessions.items) |*session| {
            self.allocator.free(session.text);
        }
        self.lookups.deinit();
        self.merges.deinit();
        self.sessions.deinit();
    }

    pub fn addLookup(self: *TokenizerProfile, lookup: LookupStats) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.lookups.append(lookup);
        self.total_lookups += 1;
        if (lookup.cache_hit) {
            self.total_cache_hits += 1;
        }

        // Update average
        const n = @as(f64, @floatFromInt(self.total_lookups));
        const old_avg = self.avg_lookup_time_ns;
        const new_time = @as(f64, @floatFromInt(lookup.duration_ns));
        self.avg_lookup_time_ns = (old_avg * (n - 1.0) + new_time) / n;

        // Limit history size
        if (self.lookups.items.len > 10000) {
            var old = self.lookups.orderedRemove(0);
            self.allocator.free(old.operation);
            self.allocator.free(old.token);
        }
    }

    pub fn addMerge(self: *TokenizerProfile, merge: MergeStats) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.merges.append(merge);
        self.total_merges += 1;

        // Update average
        const n = @as(f64, @floatFromInt(self.total_merges));
        const old_avg = self.avg_merge_time_ns;
        const new_time = @as(f64, @floatFromInt(merge.duration_ns));
        self.avg_merge_time_ns = (old_avg * (n - 1.0) + new_time) / n;

        // Limit history
        if (self.merges.items.len > 10000) {
            var old = self.merges.orderedRemove(0);
            self.allocator.free(old.pattern);
        }
    }

    pub fn addSession(self: *TokenizerProfile, session: TokenizationSession) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.sessions.append(session);
        self.total_duration_ns += session.total_duration_ns;

        // Limit history
        if (self.sessions.items.len > 1000) {
            var old = self.sessions.orderedRemove(0);
            self.allocator.free(old.text);
        }
    }

    pub fn getCacheHitRate(self: *const TokenizerProfile) f32 {
        if (self.total_lookups == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_cache_hits)) / @as(f32, @floatFromInt(self.total_lookups)) * 100.0;
    }

    pub fn getTokensPerSecond(self: *const TokenizerProfile) f64 {
        if (self.total_duration_ns == 0) return 0.0;
        var total_tokens: u64 = 0;
        for (self.sessions.items) |session| {
            total_tokens += session.token_count;
        }
        const seconds = @as(f64, @floatFromInt(self.total_duration_ns)) / 1_000_000_000.0;
        return @as(f64, @floatFromInt(total_tokens)) / seconds;
    }

    pub fn getSlowestLookups(self: *TokenizerProfile, limit: usize) ![]LookupStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Sort by duration
        var sorted = try self.allocator.dupe(LookupStats, self.lookups.items);
        std.sort.pdq(LookupStats, sorted, {}, struct {
            fn lessThan(_: void, a: LookupStats, b: LookupStats) bool {
                return a.duration_ns > b.duration_ns;
            }
        }.lessThan);

        const result_len = @min(limit, sorted.len);
        const result = try self.allocator.alloc(LookupStats, result_len);
        @memcpy(result, sorted[0..result_len]);
        self.allocator.free(sorted);

        return result;
    }

    pub fn toJson(self: *TokenizerProfile, writer: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try writer.writeAll("{");
        try writer.print("\"total_lookups\":{d},", .{self.total_lookups});
        try writer.print("\"total_merges\":{d},", .{self.total_merges});
        try writer.print("\"cache_hit_rate\":{d:.2},", .{self.getCacheHitRate()});
        try writer.print("\"avg_lookup_time_ns\":{d:.2},", .{self.avg_lookup_time_ns});
        try writer.print("\"avg_merge_time_ns\":{d:.2},", .{self.avg_merge_time_ns});
        try writer.print("\"tokens_per_second\":{d:.2},", .{self.getTokensPerSecond()});
        try writer.print("\"vocab_size\":{d},", .{self.vocab_size});
        try writer.print("\"trie_depth\":{d},", .{self.trie_depth});

        // Recent sessions
        try writer.writeAll("\"recent_sessions\":[");
        const recent_count = @min(10, self.sessions.items.len);
        if (recent_count > 0) {
            const start_idx = self.sessions.items.len - recent_count;
            for (self.sessions.items[start_idx..], 0..) |session, i| {
                if (i > 0) try writer.writeAll(",");
                try writer.writeAll("{");
                try writer.print("\"text_length\":{d},", .{session.text.len});
                try writer.print("\"token_count\":{d},", .{session.token_count});
                try writer.print("\"duration_ms\":{d:.2},", .{@as(f64, @floatFromInt(session.total_duration_ns)) / 1_000_000.0});
                try writer.print("\"cache_hit_rate\":{d:.2}", .{session.cache_hit_rate});
                try writer.writeAll("}");
            }
        }
        try writer.writeAll("]");

        try writer.writeAll("}");
    }
};

pub const TokenizerProfiler = struct {
    config: TokenizerProfileConfig,
    profile: TokenizerProfile,
    is_running: std.atomic.Value(bool),
    sample_counter: std.atomic.Value(u32),
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: TokenizerProfileConfig) !*TokenizerProfiler {
        const profiler = try allocator.create(TokenizerProfiler);
        profiler.* = .{
            .config = config,
            .profile = TokenizerProfile.init(allocator),
            .is_running = std.atomic.Value(bool).init(false),
            .sample_counter = std.atomic.Value(u32).init(0),
            .allocator = allocator,
        };
        return profiler;
    }

    pub fn deinit(self: *TokenizerProfiler) void {
        self.stop();
        self.profile.deinit();
        self.allocator.destroy(self);
    }

    pub fn start(self: *TokenizerProfiler) void {
        self.is_running.store(true, .release);
    }

    pub fn stop(self: *TokenizerProfiler) void {
        self.is_running.store(false, .release);
    }

    pub fn getProfile(self: *TokenizerProfiler) *TokenizerProfile {
        return &self.profile;
    }

    pub fn trackLookup(
        self: *TokenizerProfiler,
        operation: []const u8,
        token: []const u8,
        duration_ns: i64,
        cache_hit: bool,
        vocab_size: usize,
    ) !void {
        if (!self.is_running.load(.acquire)) return;
        if (!self.config.track_lookups) return;

        const counter = self.sample_counter.fetchAdd(1, .monotonic);
        if (counter % self.config.sample_rate != 0) return;

        const lookup = LookupStats{
            .operation = try self.allocator.dupe(u8, operation),
            .token = try self.allocator.dupe(u8, token),
            .duration_ns = duration_ns,
            .cache_hit = cache_hit,
            .vocab_size = vocab_size,
        };

        try self.profile.addLookup(lookup);
    }

    pub fn trackMerge(
        self: *TokenizerProfiler,
        pattern: []const u8,
        duration_ns: i64,
        merge_count: u32,
        cached: bool,
    ) !void {
        if (!self.is_running.load(.acquire)) return;
        if (!self.config.track_merges) return;

        const merge = MergeStats{
            .pattern = try self.allocator.dupe(u8, pattern),
            .duration_ns = duration_ns,
            .merge_count = merge_count,
            .cached = cached,
        };

        try self.profile.addMerge(merge);
    }

    pub fn trackSession(
        self: *TokenizerProfiler,
        text: []const u8,
        total_duration_ns: i64,
        token_count: u32,
        lookup_count: u32,
        merge_count: u32,
        cache_hit_rate: f32,
    ) !void {
        if (!self.is_running.load(.acquire)) return;

        const session = TokenizationSession{
            .text = try self.allocator.dupe(u8, text),
            .total_duration_ns = total_duration_ns,
            .token_count = token_count,
            .lookup_count = lookup_count,
            .merge_count = merge_count,
            .cache_hit_rate = cache_hit_rate,
            .timestamp_ns = std.time.nanoTimestamp(),
        };

        try self.profile.addSession(session);
    }

    pub fn analyzeVocabularyEfficiency(self: *TokenizerProfiler) !VocabularyAnalysis {
        const profile = self.getProfile();
        
        return VocabularyAnalysis{
            .vocab_size = profile.vocab_size,
            .avg_lookup_time_ns = profile.avg_lookup_time_ns,
            .cache_hit_rate = profile.getCacheHitRate(),
            .recommendation = try self.generateVocabRecommendation(profile),
            .allocator = self.allocator,
        };
    }

    fn generateVocabRecommendation(self: *TokenizerProfiler, profile: *TokenizerProfile) ![]const u8 {
        const cache_hit_rate = profile.getCacheHitRate();
        const avg_lookup_ns = profile.avg_lookup_time_ns;

        if (avg_lookup_ns > 1000.0) { // > 1 microsecond
            if (cache_hit_rate < 80.0) {
                return try self.allocator.dupe(u8, "Slow lookups with low cache hit rate. Consider: 1) Implementing trie-based lookup, 2) Increase cache size, 3) Use hash-based vocabulary lookup");
            } else {
                return try self.allocator.dupe(u8, "Slow lookups despite good cache hit rate. Consider: 1) Optimizing trie traversal, 2) Using SIMD for string comparison, 3) Reducing vocabulary size");
            }
        }

        if (cache_hit_rate < 50.0) {
            return try self.allocator.dupe(u8, "Low cache hit rate. Consider: 1) Increase cache size, 2) Implement LRU eviction, 3) Pre-warm cache with common tokens");
        }

        if (profile.avg_merge_time_ns > 500.0) {
            return try self.allocator.dupe(u8, "Slow BPE merges. Consider: 1) Cache merge results, 2) Optimize merge algorithm, 3) Use faster data structures");
        }

        return try self.allocator.dupe(u8, "Tokenization performance is good. Minor optimizations: 1) Monitor for regressions, 2) Profile edge cases, 3) Consider incremental improvements");
    }
};

pub const VocabularyAnalysis = struct {
    vocab_size: usize,
    avg_lookup_time_ns: f64,
    cache_hit_rate: f32,
    recommendation: []const u8,
    allocator: Allocator,

    pub fn deinit(self: *VocabularyAnalysis) void {
        self.allocator.free(self.recommendation);
    }
};

// Testing
test "TokenizerProfiler basic" {
    const allocator = std.testing.allocator;

    const config = TokenizerProfileConfig{
        .track_lookups = true,
        .sample_rate = 1,
    };

    var profiler = try TokenizerProfiler.init(allocator, config);
    defer profiler.deinit();

    profiler.start();

    // Track some lookups
    try profiler.trackLookup("vocab_lookup", "hello", 500, false, 50000);
    try profiler.trackLookup("trie_lookup", "world", 300, true, 50000);

    const profile = profiler.getProfile();
    try std.testing.expect(profile.total_lookups == 2);
    try std.testing.expect(profile.total_cache_hits == 1);
}
