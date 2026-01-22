// Tokenizer Optimization Example
// Demonstrates vocabulary optimization techniques and profiling

const std = @import("std");
const tokenizer_profiler = @import("../tokenizer_profiler.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Tokenizer Optimization Example ===\n\n", .{});

    // Example 1: Profile vocabulary lookup performance
    try example1_vocabularyLookup(allocator);

    // Example 2: Profile BPE merge operations
    try example2_bpeMerges(allocator);

    // Example 3: Compare different lookup strategies
    try example3_compareLookupStrategies(allocator);

    // Example 4: Analyze and optimize
    try example4_analyzeAndOptimize(allocator);
}

// Example 1: Profile vocabulary lookup performance
fn example1_vocabularyLookup(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 1: Vocabulary Lookup Profiling\n", .{});
    std.debug.print("---------------------------------------\n", .{});

    const config = tokenizer_profiler.TokenizerProfileConfig{
        .track_lookups = true,
        .track_cache_hits = true,
        .sample_rate = 1,
    };

    var profiler = try tokenizer_profiler.TokenizerProfiler.init(allocator, config);
    defer profiler.deinit();

    profiler.start();

    // Simulate vocabulary lookups
    const vocab_size: usize = 50000;
    const test_tokens = [_][]const u8{ "hello", "world", "the", "a", "is", "hello", "world" };

    for (test_tokens) |token| {
        const start = std.time.nanoTimestamp();
        
        // Simulate lookup (replace with actual vocabulary lookup)
        simulateVocabLookup(token, vocab_size);
        
        const duration = std.time.nanoTimestamp() - start;
        const cache_hit = std.mem.eql(u8, token, "hello") or std.mem.eql(u8, token, "world");

        try profiler.trackLookup("vocab_lookup", token, duration, cache_hit, vocab_size);
    }

    profiler.stop();

    // Get results
    const profile = profiler.getProfile();
    std.debug.print("Total lookups: {d}\n", .{profile.total_lookups});
    std.debug.print("Cache hits: {d}\n", .{profile.total_cache_hits});
    std.debug.print("Cache hit rate: {d:.1}%\n", .{profile.getCacheHitRate()});
    std.debug.print("Average lookup time: {d:.0} ns\n\n", .{profile.avg_lookup_time_ns});
}

// Example 2: Profile BPE merge operations
fn example2_bpeMerges(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 2: BPE Merge Profiling\n", .{});
    std.debug.print("-------------------------------\n", .{});

    const config = tokenizer_profiler.TokenizerProfileConfig{
        .track_merges = true,
        .sample_rate = 1,
    };

    var profiler = try tokenizer_profiler.TokenizerProfiler.init(allocator, config);
    defer profiler.deinit();

    profiler.start();

    // Simulate BPE merges
    const merge_patterns = [_][]const u8{ "th", "he", "in", "er", "an" };
    var cached_count: u32 = 0;

    for (merge_patterns, 0..) |pattern, i| {
        const start = std.time.nanoTimestamp();
        
        // Simulate merge operation
        simulateBPEMerge(pattern);
        
        const duration = std.time.nanoTimestamp() - start;
        const is_cached = i >= 3; // Last 2 are cached
        if (is_cached) cached_count += 1;

        try profiler.trackMerge(pattern, duration, 1, is_cached);
    }

    profiler.stop();

    const profile = profiler.getProfile();
    std.debug.print("Total merges: {d}\n", .{profile.total_merges});
    std.debug.print("Cached merges: {d}\n", .{cached_count});
    std.debug.print("Average merge time: {d:.0} ns\n\n", .{profile.avg_merge_time_ns});
}

// Example 3: Compare different lookup strategies
fn example3_compareLookupStrategies(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 3: Compare Lookup Strategies\n", .{});
    std.debug.print("-------------------------------------\n", .{});

    const strategies = [_][]const u8{ "linear_search", "hash_lookup", "trie_lookup" };
    const vocab_size: usize = 50000;
    const test_token = "example";

    for (strategies) |strategy| {
        var total_time: i64 = 0;
        const iterations = 1000;

        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            const start = std.time.nanoTimestamp();
            
            if (std.mem.eql(u8, strategy, "linear_search")) {
                simulateLinearSearch(test_token, vocab_size);
            } else if (std.mem.eql(u8, strategy, "hash_lookup")) {
                simulateHashLookup(test_token, vocab_size);
            } else {
                simulateTrieLookup(test_token, vocab_size);
            }
            
            total_time += std.time.nanoTimestamp() - start;
        }

        const avg_time = @divFloor(total_time, iterations);
        std.debug.print("{s}: {d} ns (avg over {d} iterations)\n", .{ strategy, avg_time, iterations });
    }
    std.debug.print("\n", .{});
}

// Example 4: Analyze and provide optimization recommendations
fn example4_analyzeAndOptimize(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 4: Analyze & Optimize\n", .{});
    std.debug.print("------------------------------\n", .{});

    const config = tokenizer_profiler.TokenizerProfileConfig{
        .track_lookups = true,
        .track_merges = true,
        .track_cache_hits = true,
        .sample_rate = 1,
    };

    var profiler = try tokenizer_profiler.TokenizerProfiler.init(allocator, config);
    defer profiler.deinit();

    profiler.start();

    // Simulate full tokenization session
    const text = "Hello world! This is a test of tokenization performance.";
    const start_time = std.time.nanoTimestamp();

    // Simulate tokenization
    var token_count: u32 = 0;
    var lookup_count: u32 = 0;
    var merge_count: u32 = 0;
    var cache_hits: u32 = 0;

    // Mock tokenization process
    const words = [_][]const u8{ "Hello", "world", "This", "is", "a", "test", "of", "tokenization", "performance" };
    for (words) |word| {
        // Track lookup
        const lookup_start = std.time.nanoTimestamp();
        simulateVocabLookup(word, 50000);
        const lookup_duration = std.time.nanoTimestamp() - lookup_start;
        
        const is_cached = token_count > 3;
        if (is_cached) cache_hits += 1;
        
        try profiler.trackLookup("vocab_lookup", word, lookup_duration, is_cached, 50000);
        
        lookup_count += 1;
        token_count += 1;

        // Simulate some BPE merges
        if (word.len > 4) {
            const merge_start = std.time.nanoTimestamp();
            simulateBPEMerge(word[0..2]);
            const merge_duration = std.time.nanoTimestamp() - merge_start;
            
            try profiler.trackMerge(word[0..2], merge_duration, 1, false);
            merge_count += 1;
        }
    }

    const total_duration = std.time.nanoTimestamp() - start_time;
    const cache_hit_rate = @as(f32, @floatFromInt(cache_hits)) / @as(f32, @floatFromInt(lookup_count)) * 100.0;

    // Track session
    try profiler.trackSession(text, total_duration, token_count, lookup_count, merge_count, cache_hit_rate);

    profiler.stop();

    // Analyze and get recommendations
    var analysis = try profiler.analyzeVocabularyEfficiency();
    defer analysis.deinit();

    const profile = profiler.getProfile();
    std.debug.print("Tokenization Results:\n", .{});
    std.debug.print("  Text: \"{s}\"\n", .{text});
    std.debug.print("  Tokens: {d}\n", .{token_count});
    std.debug.print("  Lookups: {d}\n", .{lookup_count});
    std.debug.print("  Merges: {d}\n", .{merge_count});
    std.debug.print("  Duration: {d:.2} ms\n", .{@as(f64, @floatFromInt(total_duration)) / 1_000_000.0});
    std.debug.print("  Tokens/sec: {d:.0}\n", .{profile.getTokensPerSecond()});
    std.debug.print("  Cache hit rate: {d:.1}%\n", .{cache_hit_rate});
    std.debug.print("\n", .{});

    std.debug.print("Performance Analysis:\n", .{});
    std.debug.print("  Vocab size: {d}\n", .{analysis.vocab_size});
    std.debug.print("  Avg lookup time: {d:.0} ns\n", .{analysis.avg_lookup_time_ns});
    std.debug.print("  Cache hit rate: {d:.1}%\n", .{analysis.cache_hit_rate});
    std.debug.print("\n", .{});

    std.debug.print("Recommendations:\n", .{});
    std.debug.print("  {s}\n", .{analysis.recommendation});
    std.debug.print("\n", .{});

    // Export to JSON
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try profile.toJson(buffer.writer());
    std.debug.print("JSON Profile ({d} bytes):\n{s}\n\n", .{ buffer.items.len, buffer.items });
}

// Simulation functions (replace with actual implementations)
fn simulateVocabLookup(token: []const u8, vocab_size: usize) void {
    _ = token;
    _ = vocab_size;
    // Simulate work
    var sum: u32 = 0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        sum +%= i;
    }
}

fn simulateBPEMerge(pattern: []const u8) void {
    _ = pattern;
    // Simulate work
    var sum: u32 = 0;
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        sum +%= i;
    }
}

fn simulateLinearSearch(token: []const u8, vocab_size: usize) void {
    _ = token;
    // Simulate O(n) search
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < vocab_size / 100) : (i += 1) {
        sum +%= i;
    }
}

fn simulateHashLookup(token: []const u8, vocab_size: usize) void {
    _ = vocab_size;
    // Simulate O(1) hash lookup
    var hash: u64 = 0;
    for (token) |c| {
        hash = hash *% 31 +% c;
    }
}

fn simulateTrieLookup(token: []const u8, vocab_size: usize) void {
    _ = vocab_size;
    // Simulate O(k) trie lookup where k = token length
    var sum: u32 = 0;
    for (token) |c| {
        sum +%= c;
    }
}
