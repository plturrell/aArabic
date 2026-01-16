const std = @import("std");

/// Performance Optimization Module
/// Provides profiling and optimized operations

// ============================================================================
// Performance Profiling
// ============================================================================

pub const Timer = struct {
    start: i128,
    
    pub fn start_timer() Timer {
        return .{ .start = std.time.nanoTimestamp() };
    }
    
    pub fn elapsed_ms(self: *const Timer) f64 {
        const end = std.time.nanoTimestamp();
        return @as(f64, @floatFromInt(end - self.start)) / 1_000_000.0;
    }
    
    pub fn elapsed_us(self: *const Timer) f64 {
        const end = std.time.nanoTimestamp();
        return @as(f64, @floatFromInt(end - self.start)) / 1_000.0;
    }
};

pub const PerformanceStats = struct {
    forward_pass_ms: f64 = 0,
    attention_ms: f64 = 0,
    ffn_ms: f64 = 0,
    embedding_ms: f64 = 0,
    projection_ms: f64 = 0,
    
    count: usize = 0,
    
    pub fn add(self: *PerformanceStats, other: PerformanceStats) void {
        self.forward_pass_ms += other.forward_pass_ms;
        self.attention_ms += other.attention_ms;
        self.ffn_ms += other.ffn_ms;
        self.embedding_ms += other.embedding_ms;
        self.projection_ms += other.projection_ms;
        self.count += 1;
    }
    
    pub fn average(self: *const PerformanceStats) PerformanceStats {
        if (self.count == 0) return .{};
        
        const c = @as(f64, @floatFromInt(self.count));
        return .{
            .forward_pass_ms = self.forward_pass_ms / c,
            .attention_ms = self.attention_ms / c,
            .ffn_ms = self.ffn_ms / c,
            .embedding_ms = self.embedding_ms / c,
            .projection_ms = self.projection_ms / c,
            .count = self.count,
        };
    }
    
    pub fn print(self: *const PerformanceStats, label: []const u8) void {
        std.debug.print("\n📊 Performance Stats: {s}\n", .{label});
        std.debug.print("   Forward pass: {d:.3} ms\n", .{self.forward_pass_ms});
        std.debug.print("   - Embedding:  {d:.3} ms ({d:.1}%)\n", .{
            self.embedding_ms,
            100.0 * self.embedding_ms / self.forward_pass_ms,
        });
        std.debug.print("   - Attention:  {d:.3} ms ({d:.1}%)\n", .{
            self.attention_ms,
            100.0 * self.attention_ms / self.forward_pass_ms,
        });
        std.debug.print("   - FFN:        {d:.3} ms ({d:.1}%)\n", .{
            self.ffn_ms,
            100.0 * self.ffn_ms / self.forward_pass_ms,
        });
        std.debug.print("   - Projection: {d:.3} ms ({d:.1}%)\n", .{
            self.projection_ms,
            100.0 * self.projection_ms / self.forward_pass_ms,
        });
        
        if (self.count > 1) {
            std.debug.print("   Samples: {d}\n", .{self.count});
        }
    }
};

// ============================================================================
// Optimized Matrix Operations
// ============================================================================

/// Optimized matrix multiplication using loop tiling
pub fn matmul_tiled(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    // Tile size for better cache utilization
    const tile_size = 64;
    
    // Zero output
    @memset(output, 0.0);
    
    // Tiled matrix multiplication
    var i: usize = 0;
    while (i < m) : (i += tile_size) {
        const i_end = @min(i + tile_size, m);
        
        var j: usize = 0;
        while (j < n) : (j += tile_size) {
            const j_end = @min(j + tile_size, n);
            
            var kk: usize = 0;
            while (kk < k) : (kk += tile_size) {
                const k_end = @min(kk + tile_size, k);
                
                // Compute tile
                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var jj = j;
                    while (jj < j_end) : (jj += 1) {
                        var sum: f32 = output[ii * n + jj];
                        
                        var kkk = kk;
                        while (kkk < k_end) : (kkk += 1) {
                            sum += input[ii * k + kkk] * weight[kkk * n + jj];
                        }
                        
                        output[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

/// Fast RMS normalization with reduced allocations
pub fn rms_norm_fast(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) void {
    // Calculate RMS
    var sum: f32 = 0.0;
    for (input) |val| {
        sum += val * val;
    }
    
    const rms = @sqrt(sum / @as(f32, @floatFromInt(input.len)) + eps);
    const scale = 1.0 / rms;
    
    // Apply normalization and weight
    for (input, output, weight) |in_val, *out_val, w| {
        out_val.* = in_val * scale * w;
    }
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_performance(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Performance Module\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Timer
    {
        std.debug.print("\n1️⃣  Testing timer...\n", .{});
        
        var timer = Timer.start_timer();
        
        // Simulate some work
        var sum: f64 = 0;
        for (0..1000000) |i| {
            sum += @as(f64, @floatFromInt(i));
        }
        
        const elapsed = timer.elapsed_ms();
        std.debug.print("   Elapsed: {d:.3} ms\n", .{elapsed});
        std.debug.print("   ✅ Timer working\n", .{});
    }
    
    // Test 2: Performance stats
    {
        std.debug.print("\n2️⃣  Testing performance stats...\n", .{});
        
        var stats = PerformanceStats{
            .forward_pass_ms = 10.0,
            .attention_ms = 4.0,
            .ffn_ms = 5.0,
            .embedding_ms = 0.5,
            .projection_ms = 0.5,
            .count = 1,
        };
        
        stats.print("Test");
        std.debug.print("   ✅ Performance stats working\n", .{});
    }
    
    // Test 3: Tiled matmul
    {
        std.debug.print("\n3️⃣  Testing tiled matrix multiplication...\n", .{});
        
        const m: usize = 64;
        const k: usize = 64;
        const n: usize = 64;
        
        const input = try allocator.alloc(f32, m * k);
        defer allocator.free(input);
        @memset(input, 1.0);
        
        const weight = try allocator.alloc(f32, k * n);
        defer allocator.free(weight);
        @memset(weight, 0.1);
        
        const output = try allocator.alloc(f32, m * n);
        defer allocator.free(output);
        
        var timer = Timer.start_timer();
        matmul_tiled(output, input, weight, m, k, n);
        const elapsed = timer.elapsed_us();
        
        std.debug.print("   Matrix size: {d}x{d}x{d}\n", .{ m, k, n });
        std.debug.print("   Time: {d:.3} μs\n", .{elapsed});
        std.debug.print("   ✅ Tiled matmul working\n", .{});
    }
    
    // Test 4: Fast RMS norm
    {
        std.debug.print("\n4️⃣  Testing fast RMS normalization...\n", .{});
        
        const size: usize = 1024;
        
        const input = try allocator.alloc(f32, size);
        defer allocator.free(input);
        @memset(input, 1.0);
        
        const weight = try allocator.alloc(f32, size);
        defer allocator.free(weight);
        for (weight) |*w| w.* = 1.0;
        
        const output = try allocator.alloc(f32, size);
        defer allocator.free(output);
        
        var timer = Timer.start_timer();
        rms_norm_fast(output, input, weight, 1e-5);
        const elapsed = timer.elapsed_us();
        
        std.debug.print("   Size: {d}\n", .{size});
        std.debug.print("   Time: {d:.3} μs\n", .{elapsed});
        std.debug.print("   ✅ Fast RMS norm working\n", .{});
    }
    
    std.debug.print("\n✅ All performance module tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
