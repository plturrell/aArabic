//! Benchmark Validator
//! Validates and tracks benchmark scores for models in MODEL_REGISTRY.json
//!
//! Features:
//! - Validate benchmark scores against known ranges
//! - Compare models across benchmarks
//! - Generate benchmark reports
//! - Track benchmark changes over time

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Known benchmark ranges and validation rules
const BenchmarkRange = struct {
    min: f64,
    max: f64,
    unit: []const u8,
};

const BENCHMARK_RANGES = std.ComptimeStringMap(BenchmarkRange, .{
    .{ "gsm8k", .{ .min = 0, .max = 100, .unit = "%" } },
    .{ "humaneval", .{ .min = 0, .max = 100, .unit = "%" } },
    .{ "mbpp", .{ .min = 0, .max = 100, .unit = "%" } },
    .{ "mmlu", .{ .min = 0, .max = 100, .unit = "%" } },
    .{ "hellaswag", .{ .min = 0, .max = 100, .unit = "%" } },
    .{ "arc_challenge", .{ .min = 0, .max = 100, .unit = "%" } },
    .{ "truthfulqa", .{ .min = 0, .max = 100, .unit = "%" } },
    .{ "winogrande", .{ .min = 0, .max = 100, .unit = "%" } },
});

/// Benchmark to task category mapping
const BENCHMARK_CATEGORIES = std.ComptimeStringMap([]const u8, .{
    .{ "gsm8k", "math" },
    .{ "humaneval", "code" },
    .{ "mbpp", "code" },
    .{ "mmlu", "reasoning" },
    .{ "hellaswag", "reasoning" },
    .{ "arc_challenge", "reasoning" },
    .{ "truthfulqa", "reasoning" },
    .{ "winogrande", "reasoning" },
});

pub const BenchmarkValidator = struct {
    allocator: Allocator,
    registry_path: []const u8,
    registry: std.json.Value,
    validation_errors: std.ArrayList([]const u8),
    warnings: std.ArrayList([]const u8),
    
    pub fn init(allocator: Allocator, registry_path: []const u8) !*BenchmarkValidator {
        const self = try allocator.create(BenchmarkValidator);
        self.* = .{
            .allocator = allocator,
            .registry_path = try allocator.dupe(u8, registry_path),
            .registry = undefined,
            .validation_errors = std.ArrayList([]const u8).init(allocator),
            .warnings = std.ArrayList([]const u8).init(allocator),
        };
        
        try self.loadRegistry();
        return self;
    }
    
    pub fn deinit(self: *BenchmarkValidator) void {
        for (self.validation_errors.items) |err| {
            self.allocator.free(err);
        }
        self.validation_errors.deinit();
        
        for (self.warnings.items) |warn| {
            self.allocator.free(warn);
        }
        self.warnings.deinit();
        
        self.allocator.free(self.registry_path);
        self.registry.deinit(self.allocator);
        self.allocator.destroy(self);
    }
    
    fn loadRegistry(self: *BenchmarkValidator) !void {
        const file = try std.fs.cwd().openFile(self.registry_path, .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(self.allocator, 100 * 1024 * 1024);
        defer self.allocator.free(content);
        
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            content,
            .{}
        );
        self.registry = parsed.value;
    }
    
    /// Validate all benchmarks in registry
    pub fn validateAll(self: *BenchmarkValidator) !bool {
        const stdout = std.io.getStdOut().writer();
        
        try stdout.print("\n{'=':**60}\n", .{});
        try stdout.print("Benchmark Validation Report\n", .{});
        try stdout.print("{'=':**60}\n\n", .{});
        
        var all_valid = true;
        var models_with_benchmarks: usize = 0;
        var total_benchmarks: usize = 0;
        
        const models = self.registry.object.get("models") orelse return error.NoModelsField;
        const models_array = models.array;
        
        for (models_array.items) |model| {
            const model_obj = model.object;
            const model_name = if (model_obj.get("name")) |n| n.string else "unknown";
            const benchmarks = model_obj.get("benchmarks");
            
            if (benchmarks == null or benchmarks.?.object.count() == 0) {
                const warn_msg = try std.fmt.allocPrint(
                    self.allocator,
                    "{s}: No benchmarks defined",
                    .{model_name}
                );
                try self.warnings.append(warn_msg);
                continue;
            }
            
            models_with_benchmarks += 1;
            try stdout.print("\nModel: {s}\n", .{model_name});
            try stdout.print("{'-':**40}\n", .{});
            
            var bench_iter = benchmarks.?.object.iterator();
            while (bench_iter.next()) |entry| {
                total_benchmarks += 1;
                const benchmark_name = entry.key_ptr.*;
                const benchmark_data = entry.value_ptr.*;
                
                if (benchmark_data == .object and benchmark_data.object.get("score")) |score_value| {
                    const score = switch (score_value.*) {
                        .float => |f| f,
                        .integer => |i| @as(f64, @floatFromInt(i)),
                        else => continue,
                    };
                    
                    const is_valid = try self.validateBenchmark(model_name, benchmark_name, score);
                    
                    if (!is_valid) {
                        all_valid = false;
                    }
                    
                    const status = if (is_valid) "✓" else "✗";
                    try stdout.print("  {s} {s}: {d:.2}\n", .{ status, benchmark_name, score });
                } else {
                    try stdout.print("  ⚠ {s}: Invalid format\n", .{benchmark_name});
                    const warn_msg = try std.fmt.allocPrint(
                        self.allocator,
                        "{s}/{s}: Invalid benchmark format",
                        .{ model_name, benchmark_name }
                    );
                    try self.warnings.append(warn_msg);
                }
            }
        }
        
        // Print summary
        try stdout.print("\n{'=':**60}\n", .{});
        try stdout.print("Summary\n", .{});
        try stdout.print("{'=':**60}\n", .{});
        try stdout.print("Models with benchmarks: {d}/{d}\n", 
            .{ models_with_benchmarks, models_array.items.len });
        try stdout.print("Total benchmarks: {d}\n", .{total_benchmarks});
        try stdout.print("Validation errors: {d}\n", .{self.validation_errors.items.len});
        try stdout.print("Warnings: {d}\n", .{self.warnings.items.len});
        
        if (self.validation_errors.items.len > 0) {
            try stdout.print("\n{'=':**60}\n", .{});
            try stdout.print("Validation Errors\n", .{});
            try stdout.print("{'=':**60}\n", .{});
            for (self.validation_errors.items) |err| {
                try stdout.print("  ✗ {s}\n", .{err});
            }
        }
        
        if (self.warnings.items.len > 0) {
            try stdout.print("\n{'=':**60}\n", .{});
            try stdout.print("Warnings\n", .{});
            try stdout.print("{'=':**60}\n", .{});
            for (self.warnings.items) |warn| {
                try stdout.print("  ⚠ {s}\n", .{warn});
            }
        }
        
        try stdout.print("\n", .{});
        return all_valid;
    }
    
    fn validateBenchmark(
        self: *BenchmarkValidator,
        model_name: []const u8,
        benchmark_name: []const u8,
        score: f64
    ) !bool {
        // Check if benchmark is known
        const valid_range = BENCHMARK_RANGES.get(benchmark_name);
        if (valid_range == null) {
            const warn_msg = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}: Unknown benchmark",
                .{ model_name, benchmark_name }
            );
            try self.warnings.append(warn_msg);
            return true; // Don't fail on unknown benchmarks
        }
        
        // Validate score range
        const range = valid_range.?;
        if (score < range.min or score > range.max) {
            const err_msg = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}: Score {d:.2} outside valid range [{d:.0}, {d:.0}]",
                .{ model_name, benchmark_name, score, range.min, range.max }
            );
            try self.validation_errors.append(err_msg);
            return false;
        }
        
        return true;
    }
    
    /// Compare all models on a specific benchmark
    pub fn compareModels(self: *BenchmarkValidator, benchmark_name: []const u8) !void {
        const stdout = std.io.getStdOut().writer();
        
        try stdout.print("\n{'=':**60}\n", .{});
        try stdout.print("Benchmark Comparison: {s}\n", .{benchmark_name});
        try stdout.print("{'=':**60}\n\n", .{});
        
        const ModelScore = struct {
            model: []const u8,
            score: f64,
            date: []const u8,
        };
        
        var scores = std.ArrayList(ModelScore).init(self.allocator);
        defer scores.deinit();
        
        const models = self.registry.object.get("models").?.array;
        
        for (models.items) |model| {
            const model_obj = model.object;
            const model_name = if (model_obj.get("name")) |n| n.string else "unknown";
            const benchmarks = model_obj.get("benchmarks") orelse continue;
            
            if (benchmarks.object.get(benchmark_name)) |bench_data| {
                if (bench_data.object.get("score")) |score_value| {
                    const score = switch (score_value) {
                        .float => |f| f,
                        .integer => |i| @as(f64, @floatFromInt(i)),
                        else => continue,
                    };
                    
                    const date = if (bench_data.object.get("date")) |d| d.string else "unknown";
                    
                    try scores.append(.{
                        .model = model_name,
                        .score = score,
                        .date = date,
                    });
                }
            }
        }
        
        if (scores.items.len == 0) {
            try stdout.print("No models found with {s} benchmark\n", .{benchmark_name});
            return;
        }
        
        // Sort by score (descending)
        std.mem.sort(ModelScore, scores.items, {}, struct {
            fn lessThan(_: void, a: ModelScore, b: ModelScore) bool {
                return a.score > b.score;
            }
        }.lessThan);
        
        try stdout.print("{s:<6} {s:<40} {s:<10} {s:<12}\n", 
            .{ "Rank", "Model", "Score", "Date" });
        try stdout.print("{'-':**70}\n", .{});
        
        for (scores.items, 1..) |entry, i| {
            try stdout.print("{d:<6} {s:<40} {d:<10.2} {s:<12}\n",
                .{ i, entry.model, entry.score, entry.date });
        }
        
        try stdout.print("\n", .{});
    }
    
    /// Generate comprehensive benchmark report
    pub fn generateReport(self: *BenchmarkValidator) !void {
        const stdout = std.io.getStdOut().writer();
        
        try stdout.print("\n{'=':**60}\n", .{});
        try stdout.print("Comprehensive Benchmark Report\n", .{});
        try stdout.print("{'=':**60}\n\n", .{});
        
        // Organize data by category
        const CategoryData = std.StringHashMap(std.ArrayList(struct {
            model: []const u8,
            score: f64,
        }));
        
        var category_data = std.StringHashMap(CategoryData).init(self.allocator);
        defer {
            var cat_iter = category_data.iterator();
            while (cat_iter.next()) |entry| {
                var bench_iter = entry.value_ptr.iterator();
                while (bench_iter.next()) |b_entry| {
                    b_entry.value_ptr.deinit();
                }
                entry.value_ptr.deinit();
            }
            category_data.deinit();
        }
        
        const models = self.registry.object.get("models").?.array;
        
        for (models.items) |model| {
            const model_obj = model.object;
            const model_name = if (model_obj.get("name")) |n| n.string else "unknown";
            const benchmarks = model_obj.get("benchmarks") orelse continue;
            
            var bench_iter = benchmarks.object.iterator();
            while (bench_iter.next()) |entry| {
                const benchmark_name = entry.key_ptr.*;
                const benchmark_data = entry.value_ptr.*;
                
                if (benchmark_data.object.get("score")) |score_value| {
                    const score = switch (score_value) {
                        .float => |f| f,
                        .integer => |i| @as(f64, @floatFromInt(i)),
                        else => continue,
                    };
                    
                    const category = BENCHMARK_CATEGORIES.get(benchmark_name) orelse "other";
                    
                    // Get or create category map
                    const cat_result = try category_data.getOrPut(category);
                    if (!cat_result.found_existing) {
                        cat_result.value_ptr.* = CategoryData.init(self.allocator);
                    }
                    
                    // Get or create benchmark list
                    const bench_result = try cat_result.value_ptr.getOrPut(benchmark_name);
                    if (!bench_result.found_existing) {
                        bench_result.value_ptr.* = std.ArrayList(@TypeOf(.{ 
                            .model = model_name, 
                            .score = score 
                        })).init(self.allocator);
                    }
                    
                    try bench_result.value_ptr.append(.{
                        .model = model_name,
                        .score = score,
                    });
                }
            }
        }
        
        // Print report by category
        var cat_iter = category_data.iterator();
        while (cat_iter.next()) |cat_entry| {
            try stdout.print("\n{s} Category\n", .{cat_entry.key_ptr.*});
            try stdout.print("{'-':**60}\n", .{});
            
            var bench_iter = cat_entry.value_ptr.iterator();
            while (bench_iter.next()) |bench_entry| {
                const entries = bench_entry.value_ptr.items;
                
                if (entries.len > 0) {
                    // Find best
                    var best_score: f64 = 0;
                    var best_model: []const u8 = "";
                    var sum: f64 = 0;
                    
                    for (entries) |e| {
                        if (e.score > best_score) {
                            best_score = e.score;
                            best_model = e.model;
                        }
                        sum += e.score;
                    }
                    
                    const avg = sum / @as(f64, @floatFromInt(entries.len));
                    
                    try stdout.print("\n  {s}:\n", .{bench_entry.key_ptr.*});
                    try stdout.print("    Models: {d}\n", .{entries.len});
                    try stdout.print("    Best: {s} ({d:.2})\n", .{ best_model, best_score });
                    try stdout.print("    Average: {d:.2}\n", .{avg});
                }
            }
        }
        
        try stdout.print("\n", .{});
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        try std.io.getStdErr().writeAll("Usage: ");
        try std.io.getStdErr().writeAll(args[0]);
        try std.io.getStdErr().writeAll(" <registry_path> [--compare BENCHMARK | --report]\n");
        std.process.exit(1);
    }
    
    const registry_path = args[1];
    
    const validator = try BenchmarkValidator.init(allocator, registry_path);
    defer validator.deinit();
    
    // Parse options
    if (args.len > 2) {
        if (std.mem.eql(u8, args[2], "--compare") and args.len > 3) {
            try validator.compareModels(args[3]);
        } else if (std.mem.eql(u8, args[2], "--report")) {
            try validator.generateReport();
        } else {
            const is_valid = try validator.validateAll();
            std.process.exit(if (is_valid) 0 else 1);
        }
    } else {
        const is_valid = try validator.validateAll();
        std.process.exit(if (is_valid) 0 else 1);
    }
}
