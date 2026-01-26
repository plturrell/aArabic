// Compression Benchmark Example
// Compares DEFLATE, LZ4, and Zstandard performance for different use cases

const std = @import("std");
const compression_profiler = @import("../compression_profiler.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Compression Algorithm Benchmark ===\n\n", .{});

    // Example 1: Compare all algorithms
    try example1_compareAlgorithms(allocator);

    // Example 2: Find optimal compression level
    try example2_optimalLevel(allocator);

    // Example 3: Profile by use case
    try example3_useCaseOptimization(allocator);

    // Example 4: Throughput analysis
    try example4_throughputAnalysis(allocator);
}

// Example 1: Compare compression algorithms
fn example1_compareAlgorithms(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 1: Algorithm Comparison\n", .{});
    std.debug.print("--------------------------------\n", .{});

    var profiler = try compression_profiler.CompressionProfiler.init(allocator);
    defer profiler.deinit();

    profiler.start();

    // Generate test data (simulating KV cache data)
    const test_data = try generateTestData(allocator, 1024 * 1024); // 1 MB
    defer allocator.free(test_data);

    const algorithms = [_]compression_profiler.CompressionAlgorithm{
        .deflate,
        .lz4,
        .zstd,
    };

    std.debug.print("Testing with 1 MB of data:\n\n", .{});

    for (algorithms) |algo| {
        const stats = try profiler.profileCompression(algo, .default, test_data);
        
        std.debug.print("{s}:\n", .{algo.toString()});
        std.debug.print("  Compressed size: {d:.2} KB\n", .{
            @as(f64, @floatFromInt(stats.compressed_size)) / 1024.0
        });
        std.debug.print("  Compression ratio: {d:.2}x\n", .{stats.compression_ratio});
        std.debug.print("  Compression time: {d:.2} ms\n", .{
            @as(f64, @floatFromInt(stats.compression_time_ns)) / 1_000_000.0
        });
        std.debug.print("  Decompression time: {d:.2} ms\n", .{
            @as(f64, @floatFromInt(stats.decompression_time_ns)) / 1_000_000.0
        });
        std.debug.print("  Throughput: {d:.2} MB/s\n\n", .{stats.throughput_mb_s});
    }

    // Overall statistics
    const profile = profiler.getProfile();
    std.debug.print("Overall Results:\n", .{});
    std.debug.print("  Total processed: {d:.2} MB\n", .{
        @as(f64, @floatFromInt(profile.total_original_bytes)) / 1_048_576.0
    });
    std.debug.print("  Total compressed: {d:.2} MB\n", .{
        @as(f64, @floatFromInt(profile.total_compressed_bytes)) / 1_048_576.0
    });
    std.debug.print("  Space saved: {d:.2} MB\n\n", .{
        @as(f64, @floatFromInt(profile.total_original_bytes - profile.total_compressed_bytes)) / 1_048_576.0
    });
}

// Example 2: Find optimal compression level
fn example2_optimalLevel(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 2: Optimal Compression Level\n", .{});
    std.debug.print("-------------------------------------\n", .{});

    var profiler = try compression_profiler.CompressionProfiler.init(allocator);
    defer profiler.deinit();

    profiler.start();

    const test_data = try generateTestData(allocator, 512 * 1024); // 512 KB
    defer allocator.free(test_data);

    const goals = [_]compression_profiler.OptimizationGoal{ .ratio, .speed, .balanced };

    for (goals) |goal| {
        std.debug.print("Optimizing for: {s}\n", .{@tagName(goal)});
        
        const optimal = try profiler.findOptimalLevel(.zstd, test_data, goal);
        std.debug.print("  Optimal level for Zstandard: {s} (level {d})\n\n", .{
            @tagName(optimal),
            optimal.toInt(),
        });
    }
}

// Example 3: Optimize by use case
fn example3_useCaseOptimization(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 3: Use Case Optimization\n", .{});
    std.debug.print("---------------------------------\n", .{});

    var profiler = try compression_profiler.CompressionProfiler.init(allocator);
    defer profiler.deinit();

    profiler.start();

    const use_cases = [_]compression_profiler.UseCase{
        .kv_cache,
        .model_weights,
        .activations,
        .embeddings,
    };

    for (use_cases) |use_case| {
        std.debug.print("\n{s}:\n", .{@tagName(use_case)});
        
        const recommendation = try profiler.generateRecommendation(use_case);
        defer allocator.free(recommendation);
        
        std.debug.print("  {s}\n", .{recommendation});
    }
    std.debug.print("\n", .{});
}

// Example 4: Throughput analysis
fn example4_throughputAnalysis(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 4: Throughput Analysis\n", .{});
    std.debug.print("-------------------------------\n", .{});

    var profiler = try compression_profiler.CompressionProfiler.init(allocator);
    defer profiler.deinit();

    profiler.start();

    const data_sizes = [_]usize{
        64 * 1024,      // 64 KB
        256 * 1024,     // 256 KB
        1024 * 1024,    // 1 MB
        4 * 1024 * 1024, // 4 MB
    };

    const algorithms = [_]compression_profiler.CompressionAlgorithm{ .lz4, .zstd, .deflate };

    std.debug.print("Throughput by Data Size:\n\n", .{});

    for (data_sizes) |size| {
        const test_data = try generateTestData(allocator, size);
        defer allocator.free(test_data);

        std.debug.print("Data size: {d} KB\n", .{size / 1024});

        for (algorithms) |algo| {
            const stats = try profiler.profileCompression(algo, .default, test_data);
            std.debug.print("  {s}: {d:.2} MB/s (ratio: {d:.2}x)\n", .{
                algo.toString(),
                stats.throughput_mb_s,
                stats.compression_ratio,
            });
        }
        std.debug.print("\n", .{});
    }

    // Find best algorithm
    const profile = profiler.getProfile();
    const best_for_ratio = profile.getBestAlgorithm(.ratio);
    const best_for_speed = profile.getBestAlgorithm(.speed);
    const best_balanced = profile.getBestAlgorithm(.balanced);

    std.debug.print("Recommendations:\n", .{});
    if (best_for_ratio) |algo| {
        std.debug.print("  Best ratio: {s}\n", .{algo.toString()});
    }
    if (best_for_speed) |algo| {
        std.debug.print("  Fastest: {s}\n", .{algo.toString()});
    }
    if (best_balanced) |algo| {
        std.debug.print("  Best balanced: {s}\n", .{algo.toString()});
    }
    std.debug.print("\n", .{});

    // Export profile
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit();

    try profile.toJson(buffer.writer());
    std.debug.print("JSON Profile:\n{s}\n", .{buffer.items});
}

// Generate test data with realistic patterns
fn generateTestData(allocator: std.mem.Allocator, size: usize) ![]u8 {
    var data = try allocator.alloc(u8, size);
    
    // Fill with semi-compressible data (realistic for KV cache/tensors)
    var prng = std.rand.DefaultPrng.init(12345);
    const random = prng.random();
    
    var i: usize = 0;
    while (i < size) : (i += 1) {
        // 70% repeated patterns, 30% random (realistic compression scenario)
        if (random.int(u8) < 179) { // ~70%
            data[i] = @as(u8, @intCast(i % 256));
        } else {
            data[i] = random.int(u8);
        }
    }
    
    return data;
}
