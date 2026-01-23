//! Benchmark-Based Scoring Module
//! Provides dynamic model scoring based on actual benchmark performance
//!
//! Features:
//! - Weighted benchmark scoring per category
//! - Configurable benchmark weights
//! - Relevant benchmark filtering
//! - Normalized scoring (0-50 points)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Benchmark scoring configuration and calculator
pub const BenchmarkScoring = struct {
    allocator: Allocator,
    benchmark_weights: std.StringHashMap(f32),
    
    pub fn init(allocator: Allocator) !*BenchmarkScoring {
        const self = try allocator.create(BenchmarkScoring);
        self.* = .{
            .allocator = allocator,
            .benchmark_weights = std.StringHashMap(f32).init(allocator),
        };
        
        // Initialize default benchmark weights
        try self.initializeDefaultWeights();
        
        return self;
    }
    
    pub fn deinit(self: *BenchmarkScoring) void {
        self.benchmark_weights.deinit();
        self.allocator.destroy(self);
    }
    
    fn initializeDefaultWeights(self: *BenchmarkScoring) !void {
        // Code benchmarks
        try self.benchmark_weights.put("HumanEval", 1.0);
        try self.benchmark_weights.put("MBPP", 0.8);
        
        // Math benchmarks
        try self.benchmark_weights.put("GSM8K", 1.2);
        try self.benchmark_weights.put("MATH", 1.5);
        
        // Reasoning benchmarks
        try self.benchmark_weights.put("MMLU", 1.5);
        try self.benchmark_weights.put("ARC-Challenge", 1.3);
        try self.benchmark_weights.put("HellaSwag", 1.0);
        try self.benchmark_weights.put("Winogrande", 1.1);
        try self.benchmark_weights.put("TruthfulQA", 1.2);
        
        // Summarization benchmarks
        try self.benchmark_weights.put("SummScreen", 1.0);
        try self.benchmark_weights.put("GovReport", 1.1);
        
        // SQL/Relational benchmarks
        try self.benchmark_weights.put("Spider", 1.3);
        try self.benchmark_weights.put("WikiSQL", 1.0);
        
        // Retrieval benchmarks
        try self.benchmark_weights.put("BEIR", 1.2);
        try self.benchmark_weights.put("MSMARCO", 1.0);
        try self.benchmark_weights.put("MTEB", 1.4);
    }
    
    /// Set custom weight for a benchmark
    pub fn setWeight(self: *BenchmarkScoring, benchmark: []const u8, weight: f32) !void {
        try self.benchmark_weights.put(benchmark, weight);
    }
    
    /// Model benchmark representation
    pub const ModelBenchmark = struct {
        name: []const u8,
        score: f32,
        metric: []const u8,
    };
    
    /// Score a model based on its benchmark performance for a category
    pub fn scoreModel(
        self: *BenchmarkScoring,
        benchmarks: []const ModelBenchmark,
        category: []const u8,
    ) f32 {
        const relevant_benchmarks = self.getRelevantBenchmarks(category);
        
        var total_score: f32 = 0.0;
        var weight_sum: f32 = 0.0;
        
        for (benchmarks) |benchmark| {
            if (self.isRelevant(benchmark.name, relevant_benchmarks)) {
                const weight = self.benchmark_weights.get(benchmark.name) orelse 1.0;
                
                // Normalize score to 0-100 range if needed
                const normalized_score = if (benchmark.score > 1.0) benchmark.score else benchmark.score * 100.0;
                
                total_score += normalized_score * weight;
                weight_sum += weight;
            }
        }
        
        if (weight_sum > 0.0) {
            // Scale to 0-50 points (max bonus from benchmarks)
            const avg_score = total_score / weight_sum;
            return (avg_score / 100.0) * 50.0;
        }
        
        return 0.0; // No relevant benchmarks
    }
    
    /// Get relevant benchmarks for a category
    fn getRelevantBenchmarks(self: *BenchmarkScoring, category: []const u8) []const []const u8 {
        _ = self;
        
        // Code category
        if (std.mem.eql(u8, category, "code")) {
            const benchmarks = [_][]const u8{ "HumanEval", "MBPP" };
            return &benchmarks;
        }
        
        // Math category
        if (std.mem.eql(u8, category, "math")) {
            const benchmarks = [_][]const u8{ "GSM8K", "MATH" };
            return &benchmarks;
        }
        
        // Reasoning category
        if (std.mem.eql(u8, category, "reasoning")) {
            const benchmarks = [_][]const u8{
                "MMLU",
                "ARC-Challenge",
                "HellaSwag",
                "Winogrande",
                "TruthfulQA",
            };
            return &benchmarks;
        }
        
        // Summarization category
        if (std.mem.eql(u8, category, "summarization")) {
            const benchmarks = [_][]const u8{ "SummScreen", "GovReport" };
            return &benchmarks;
        }
        
        // Relational/SQL category
        if (std.mem.eql(u8, category, "relational")) {
            const benchmarks = [_][]const u8{ "Spider", "WikiSQL" };
            return &benchmarks;
        }
        
        // Vector search category
        if (std.mem.eql(u8, category, "vector_search")) {
            const benchmarks = [_][]const u8{ "BEIR", "MSMARCO", "MTEB" };
            return &benchmarks;
        }
        
        // Default: no specific benchmarks
        return &[_][]const u8{};
    }
    
    /// Check if a benchmark is relevant
    fn isRelevant(
        self: *BenchmarkScoring,
        benchmark_name: []const u8,
        relevant: []const []const u8,
    ) bool {
        _ = self;
        
        for (relevant) |rel| {
            if (std.mem.eql(u8, benchmark_name, rel)) {
                return true;
            }
        }
        return false;
    }
    
    /// Get all benchmarks with non-zero weights
    pub fn getConfiguredBenchmarks(self: *BenchmarkScoring, allocator: Allocator) ![][]const u8 {
        var benchmarks = std.ArrayList([]const u8).init(allocator);
        errdefer benchmarks.deinit();
        
        var it = self.benchmark_weights.iterator();
        while (it.next()) |entry| {
            try benchmarks.append(try allocator.dupe(u8, entry.key_ptr.*));
        }
        
        return try benchmarks.toOwnedSlice();
    }
};

// Tests
test "BenchmarkScoring initialization" {
    const allocator = std.testing.allocator;
    
    const scorer = try BenchmarkScoring.init(allocator);
    defer scorer.deinit();
    
    // Verify default weights loaded
    try std.testing.expect(scorer.benchmark_weights.count() > 0);
}

test "BenchmarkScoring - code category" {
    const allocator = std.testing.allocator;
    
    const scorer = try BenchmarkScoring.init(allocator);
    defer scorer.deinit();
    
    const benchmarks = [_]BenchmarkScoring.ModelBenchmark{
        .{ .name = "HumanEval", .score = 85.0, .metric = "pass@1" },
        .{ .name = "MBPP", .score = 75.0, .metric = "pass@1" },
        .{ .name = "MMLU", .score = 70.0, .metric = "accuracy" }, // Not relevant
    };
    
    const score = scorer.scoreModel(&benchmarks, "code");
    
    // Should score based on HumanEval (weight 1.0) and MBPP (weight 0.8)
    // Expected: ((85 * 1.0) + (75 * 0.8)) / (1.0 + 0.8) = 145 / 1.8 = 80.56
    // Scaled to 0-50: 80.56 / 100 * 50 = 40.28
    try std.testing.expect(score > 35.0 and score < 45.0);
}

test "BenchmarkScoring - math category" {
    const allocator = std.testing.allocator;
    
    const scorer = try BenchmarkScoring.init(allocator);
    defer scorer.deinit();
    
    const benchmarks = [_]BenchmarkScoring.ModelBenchmark{
        .{ .name = "GSM8K", .score = 90.0, .metric = "accuracy" },
        .{ .name = "MATH", .score = 60.0, .metric = "accuracy" },
    };
    
    const score = scorer.scoreModel(&benchmarks, "math");
    
    // Should score based on GSM8K (weight 1.2) and MATH (weight 1.5)
    try std.testing.expect(score > 30.0 and score < 50.0);
}

test "BenchmarkScoring - no relevant benchmarks" {
    const allocator = std.testing.allocator;
    
    const scorer = try BenchmarkScoring.init(allocator);
    defer scorer.deinit();
    
    const benchmarks = [_]BenchmarkScoring.ModelBenchmark{
        .{ .name = "UnknownBenchmark", .score = 90.0, .metric = "accuracy" },
    };
    
    const score = scorer.scoreModel(&benchmarks, "code");
    
    // No relevant benchmarks, should return 0
    try std.testing.expectEqual(@as(f32, 0.0), score);
}

test "BenchmarkScoring - custom weights" {
    const allocator = std.testing.allocator;
    
    const scorer = try BenchmarkScoring.init(allocator);
    defer scorer.deinit();
    
    // Set custom weight
    try scorer.setWeight("HumanEval", 2.0);
    
    const benchmarks = [_]BenchmarkScoring.ModelBenchmark{
        .{ .name = "HumanEval", .score = 80.0, .metric = "pass@1" },
    };
    
    const score = scorer.scoreModel(&benchmarks, "code");
    
    // With weight 2.0, score should be: 80 / 100 * 50 = 40.0
    try std.testing.expect(score > 35.0 and score < 45.0);
}

test "BenchmarkScoring - get configured benchmarks" {
    const allocator = std.testing.allocator;
    
    const scorer = try BenchmarkScoring.init(allocator);
    defer scorer.deinit();
    
    const benchmarks = try scorer.getConfiguredBenchmarks(allocator);
    defer {
        for (benchmarks) |b| allocator.free(b);
        allocator.free(benchmarks);
    }
    
    // Should have multiple benchmarks configured
    try std.testing.expect(benchmarks.len > 10);
}
