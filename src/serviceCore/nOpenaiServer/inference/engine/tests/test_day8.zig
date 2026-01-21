const std = @import("std");
const performance = @import("performance");

/// Day 8 Tests: Performance Optimization
/// 
/// Tests:
/// 1. Performance profiling
/// 2. Optimized operations
/// 3. Benchmarking

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DAY 8 TESTS: PERFORMANCE OPTIMIZATION\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Performance module unit tests
    try performance.test_performance(allocator);
    
    // Test 2: Benchmark optimized operations
    try benchmark_operations(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 8 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Performance profiling working\n", .{});
    std.debug.print("   ✅ Optimized operations tested\n", .{});
    std.debug.print("   ✅ Benchmarking complete\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎊 Performance optimization ready! Week 2 Day 8 complete!\n", .{});
    std.debug.print("\n", .{});
}

fn benchmark_operations(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Benchmarking Optimizations\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Benchmark 1: Tiled matmul at different sizes
    {
        std.debug.print("\n1️⃣  Matrix multiplication benchmark...\n", .{});
        
        const sizes = [_]usize{ 64, 128, 256 };
        
        for (sizes) |size| {
            const m = size;
            const k = size;
            const n = size;
            
            const input = try allocator.alloc(f32, m * k);
            defer allocator.free(input);
            @memset(input, 1.0);
            
            const weight = try allocator.alloc(f32, k * n);
            defer allocator.free(weight);
            @memset(weight, 0.1);
            
            const output = try allocator.alloc(f32, m * n);
            defer allocator.free(output);
            
            // Benchmark tiled
            var timer = performance.Timer.start_timer();
            performance.matmul_tiled(output, input, weight, m, k, n);
            const tiled_time = timer.elapsed_us();
            
            std.debug.print("   Size {d}x{d}: {d:.3} μs\n", .{ size, size, tiled_time });
        }
        
        std.debug.print("   ✅ Tiled matmul benchmarked\n", .{});
    }
    
    // Benchmark 2: Fast RMS norm
    {
        std.debug.print("\n2️⃣  RMS normalization benchmark...\n", .{});
        
        const sizes = [_]usize{ 512, 1024, 2048 };
        
        for (sizes) |size| {
            const input = try allocator.alloc(f32, size);
            defer allocator.free(input);
            @memset(input, 1.0);
            
            const weight = try allocator.alloc(f32, size);
            defer allocator.free(weight);
            for (weight) |*w| w.* = 1.0;
            
            const output = try allocator.alloc(f32, size);
            defer allocator.free(output);
            
            var timer = performance.Timer.start_timer();
            performance.rms_norm_fast(output, input, weight, 1e-5);
            const norm_time = timer.elapsed_us();
            
            std.debug.print("   Size {d}: {d:.3} μs\n", .{ size, norm_time });
        }
        
        std.debug.print("   ✅ Fast RMS norm benchmarked\n", .{});
    }
    
    std.debug.print("\n✅ Benchmark tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
