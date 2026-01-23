//! Routing Performance Benchmark CLI
//! Measures model selection performance and decision quality
//!
//! Features:
//! - Benchmark routing decision time
//! - Test various constraint combinations
//! - Validate selection consistency
//! - Generate performance reports

const std = @import("std");
const ModelSelector = @import("model_selector.zig").ModelSelector;
const SelectionConstraints = @import("model_selector.zig").SelectionConstraints;

const BenchmarkResults = struct {
    category: []const u8,
    mean_ms: f64,
    median_ms: f64,
    stdev_ms: f64,
    min_ms: f64,
    max_ms: f64,
    
    pub fn deinit(self: *BenchmarkResults, allocator: std.mem.Allocator) void {
        allocator.free(self.category);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    // Parse arguments
    var iterations: usize = 1000;
    if (args.len > 1) {
        iterations = try std.fmt.parseInt(usize, args[1], 10);
    }
    
    const stdout = std.fs.File.stdout().deprecatedWriter();
    
    try stdout.print("\n{'=':**60}\n", .{});
    try stdout.print("Routing Performance Benchmark ({d} iterations)\n", .{iterations});
    try stdout.print("{'=':**60}\n\n", .{});
    
    // Initialize model selector
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    try stdout.print("Loaded {d} models and {d} categories\n\n", 
        .{selector.models.items.len, selector.categories.count()});
    
    // Run benchmarks
    try benchmarkSelectionTime(allocator, selector, iterations);
    try testConstraintCombinations(allocator, selector);
    try testSelectionConsistency(allocator, selector, iterations);
    try validateCategoryCoverage(allocator, selector);
    
    try stdout.print("\n{'=':**60}\n", .{});
    try stdout.print("Benchmark Complete\n", .{});
    try stdout.print("{'=':**60}\n", .{});
}

fn benchmarkSelectionTime(
    allocator: std.mem.Allocator,
    selector: *ModelSelector,
    iterations: usize,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    
    try stdout.print("Selection Time Benchmark:\n\n", .{});
    
    var cat_iter = selector.categories.iterator();
    while (cat_iter.next()) |entry| {
        const category = entry.key_ptr.*;
        
        var times = std.ArrayList(f64).init(allocator);
        defer times.deinit();
        
        // Run iterations
        for (0..iterations) |_| {
            const start = std.time.nanoTimestamp();
            
            const result = selector.selectModel(category, .{
                .max_gpu_memory_mb = 14 * 1024,
            }) catch continue;
            defer result.deinit(allocator);
            
            const end = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
            try times.append(duration_ms);
        }
        
        if (times.items.len == 0) continue;
        
        // Calculate statistics
        const mean = calculateMean(times.items);
        const median = try calculateMedian(allocator, times.items);
        const stdev = calculateStdev(times.items, mean);
        const min_time = std.mem.min(f64, times.items);
        const max_time = std.mem.max(f64, times.items);
        
        try stdout.print("{s}:\n", .{category});
        try stdout.print("  Mean:   {d:.4} ms\n", .{mean});
        try stdout.print("  Median: {d:.4} ms\n", .{median});
        try stdout.print("  StdDev: {d:.4} ms\n", .{stdev});
        try stdout.print("  Min:    {d:.4} ms\n", .{min_time});
        try stdout.print("  Max:    {d:.4} ms\n", .{max_time});
        try stdout.print("\n", .{});
    }
}

fn testConstraintCombinations(
    allocator: std.mem.Allocator,
    selector: *ModelSelector,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    
    try stdout.print("\nConstraint Combination Tests:\n\n", .{});
    
    const test_cases = [_]struct {
        name: []const u8,
        category: []const u8,
        max_gpu_mb: ?usize,
        agent: ?[]const u8,
    }{
        .{ .name = "T4 GPU (14GB)", .category = "code", .max_gpu_mb = 14 * 1024, .agent = "inference" },
        .{ .name = "A100 GPU (40GB)", .category = "code", .max_gpu_mb = 40 * 1024, .agent = "inference" },
        .{ .name = "Translation (tool)", .category = "relational", .max_gpu_mb = 16 * 1024, .agent = "tool" },
        .{ .name = "Small model only", .category = "code", .max_gpu_mb = 4 * 1024, .agent = null },
    };
    
    for (test_cases) |test_case| {
        const result = selector.selectModel(test_case.category, .{
            .max_gpu_memory_mb = test_case.max_gpu_mb,
            .required_agent_type = test_case.agent,
        }) catch |err| {
            try stdout.print("{s}:\n  ERROR: {any}\n\n", .{test_case.name, err});
            continue;
        };
        defer result.deinit(allocator);
        
        try stdout.print("{s}:\n", .{test_case.name});
        try stdout.print("  Selected: {s}\n", .{result.model.name});
        try stdout.print("  Score: {d:.2}\n", .{result.score});
        try stdout.print("  Category: {s}\n", .{test_case.category});
        try stdout.print("\n", .{});
    }
}

fn testSelectionConsistency(
    allocator: std.mem.Allocator,
    selector: *ModelSelector,
    iterations: usize,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    
    try stdout.print("\nSelection Consistency Test ({d} iterations):\n\n", .{iterations});
    
    var cat_iter = selector.categories.iterator();
    while (cat_iter.next()) |entry| {
        const category = entry.key_ptr.*;
        
        var selections = std.StringHashMap(usize).init(allocator);
        defer {
            var it = selections.iterator();
            while (it.next()) |s_entry| {
                allocator.free(s_entry.key_ptr.*);
            }
            selections.deinit();
        }
        
        // Run iterations
        for (0..iterations) |_| {
            const result = selector.selectModel(category, .{
                .max_gpu_memory_mb = 14 * 1024,
            }) catch continue;
            defer result.deinit(allocator);
            
            const model_result = try selections.getOrPut(result.model.name);
            if (!model_result.found_existing) {
                model_result.key_ptr.* = try allocator.dupe(u8, result.model.name);
                model_result.value_ptr.* = 0;
            }
            model_result.value_ptr.* += 1;
        }
        
        if (selections.count() == 0) continue;
        
        // Calculate consistency
        var most_common: usize = 0;
        var it = selections.iterator();
        while (it.next()) |s_entry| {
            if (s_entry.value_ptr.* > most_common) {
                most_common = s_entry.value_ptr.*;
            }
        }
        
        const consistency = (@as(f64, @floatFromInt(most_common)) / 
                           @as(f64, @floatFromInt(iterations))) * 100.0;
        
        try stdout.print("{s}:\n", .{category});
        try stdout.print("  Consistency: {d:.1}%\n", .{consistency});
        try stdout.print("  Selections:\n", .{});
        
        it = selections.iterator();
        while (it.next()) |s_entry| {
            const pct = (@as(f64, @floatFromInt(s_entry.value_ptr.*)) / 
                        @as(f64, @floatFromInt(iterations))) * 100.0;
            try stdout.print("    - {s}: {d} ({d:.1}%)\n", 
                .{s_entry.key_ptr.*, s_entry.value_ptr.*, pct});
        }
        try stdout.print("\n", .{});
    }
}

fn validateCategoryCoverage(
    allocator: std.mem.Allocator,
    selector: *ModelSelector,
) !void {
    _ = allocator;
    const stdout = std.fs.File.stdout().deprecatedWriter();
    
    try stdout.print("\nCategory Coverage Validation:\n\n", .{});
    
    var cat_iter = selector.categories.iterator();
    while (cat_iter.next()) |entry| {
        const category = entry.key_ptr.*;
        const cat_data = entry.value_ptr;
        
        // Count valid models
        var valid_count: usize = 0;
        for (cat_data.models) |model_name| {
            for (selector.models.items) |model| {
                if (std.mem.eql(u8, model.name, model_name)) {
                    valid_count += 1;
                    break;
                }
            }
        }
        
        const coverage = if (cat_data.models.len > 0)
            (@as(f64, @floatFromInt(valid_count)) / 
             @as(f64, @floatFromInt(cat_data.models.len))) * 100.0
        else
            0.0;
        
        const status = if (coverage == 100.0) "✓" else "✗";
        
        try stdout.print("{s} {s}:\n", .{status, category});
        try stdout.print("  Models: {d}/{d} valid\n", .{valid_count, cat_data.models.len});
        try stdout.print("  Coverage: {d:.1}%\n", .{coverage});
        try stdout.print("\n", .{});
    }
}

// Helper functions
fn calculateMean(values: []const f64) f64 {
    if (values.len == 0) return 0.0;
    var sum: f64 = 0.0;
    for (values) |v| sum += v;
    return sum / @as(f64, @floatFromInt(values.len));
}

fn calculateMedian(allocator: std.mem.Allocator, values: []const f64) !f64 {
    if (values.len == 0) return 0.0;
    
    const sorted = try allocator.dupe(f64, values);
    defer allocator.free(sorted);
    
    std.mem.sort(f64, sorted, {}, comptime std.sort.asc(f64));
    
    const mid = sorted.len / 2;
    if (sorted.len % 2 == 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    } else {
        return sorted[mid];
    }
}

fn calculateStdev(values: []const f64, mean: f64) f64 {
    if (values.len <= 1) return 0.0;
    
    var sum_sq_diff: f64 = 0.0;
    for (values) |v| {
        const diff = v - mean;
        sum_sq_diff += diff * diff;
    }
    
    const variance = sum_sq_diff / @as(f64, @floatFromInt(values.len - 1));
    return @sqrt(variance);
}
