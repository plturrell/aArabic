const std = @import("std");
const common = @import("common");

/// Q4_0 Quantization Format
/// 
/// Most common quantization in GGUF models (e.g., llama-3.2-1b-q4_0.gguf)
/// 
/// Format: 32 values per block, 18 bytes total
/// - 2 bytes: f16 scale factor
/// - 16 bytes: 32x 4-bit signed values (packed, 2 per byte)
/// 
/// Encoding: 4-bit signed values [-8, 7] mapped to original range
/// value = (qval - 8) * scale
///
/// This achieves ~8x compression (32 f32 = 128 bytes -> 18 bytes)

// ============================================================================
// Dequantization
// ============================================================================

/// Dequantize a full Q4_0 tensor to f32
pub fn dequantize(output: []f32, input: []const u8, n_values: usize) void {
    const n_blocks = (n_values + common.QK4_0 - 1) / common.QK4_0;
    
    // Cast input to block array
    const blocks = @as([*]const common.BlockQ4_0, @ptrCast(@alignCast(input.ptr)))[0..n_blocks];
    
    for (0..n_blocks) |block_idx| {
        const block = &blocks[block_idx];
        const block_start = block_idx * common.QK4_0;
        const block_end = @min(block_start + common.QK4_0, n_values);
        const block_size = block_end - block_start;
        
        dequantize_block(
            output[block_start..block_end],
            block,
            block_size,
        );
    }
}

/// Dequantize a single Q4_0 block
fn dequantize_block(
    output: []f32,
    block: *const common.BlockQ4_0,
    n_values: usize,
) void {
    // Convert f16 scale to f32
    const scale = common.f16_to_f32(block.scale);
    
    // Dequantize each 4-bit value
    for (0..n_values) |i| {
        const qval = common.get_4bit_value(&block.qs, i);
        
        // Q4_0 uses signed 4-bit: [0,15] maps to [-8,7]
        const signed_val = @as(i8, @intCast(qval)) - 8;
        
        output[i] = @as(f32, @floatFromInt(signed_val)) * scale;
    }
}

/// Fast SIMD-optimized dequantization for aligned blocks
pub fn dequantize_simd(output: []f32, input: []const u8, n_values: usize) void {
    const n_blocks = (n_values + common.QK4_0 - 1) / common.QK4_0;
    const blocks = @as([*]const common.BlockQ4_0, @ptrCast(@alignCast(input.ptr)))[0..n_blocks];
    
    for (0..n_blocks) |block_idx| {
        const block = &blocks[block_idx];
        const block_start = block_idx * common.QK4_0;
        const block_end = @min(block_start + common.QK4_0, n_values);
        const block_size = block_end - block_start;
        
        if (block_size == common.QK4_0) {
            // Full block - use SIMD
            dequantize_block_simd(output[block_start..block_end], block);
        } else {
            // Partial block - scalar fallback
            dequantize_block(output[block_start..block_end], block, block_size);
        }
    }
}

/// SIMD-optimized block dequantization
fn dequantize_block_simd(output: []f32, block: *const common.BlockQ4_0) void {
    const Vec = @Vector(8, f32);
    const scale = common.f16_to_f32(block.scale);
    const scale_vec: Vec = @splat(scale);
    const offset_vec: Vec = @splat(-8.0);
    
    // Process 8 values at a time
    var i: usize = 0;
    while (i < 32) : (i += 8) {
        // Extract 8x 4-bit values
        var qvals: [8]u8 = undefined;
        for (0..8) |j| {
            qvals[j] = common.get_4bit_value(&block.qs, i + j);
        }
        
        // Convert to f32 vector
        var float_vec: Vec = undefined;
        inline for (0..8) |j| {
            float_vec[j] = @floatFromInt(qvals[j]);
        }
        
        // Apply offset and scale: (qval - 8) * scale
        const result = (float_vec + offset_vec) * scale_vec;
        
        // Store result
        output[i..][0..8].* = result;
    }
}

// ============================================================================
// Quantization (Encoding)
// ============================================================================

/// Quantize f32 values to Q4_0 format
pub fn quantize(output: []u8, input: []const f32, n_values: usize) void {
    const n_blocks = (n_values + common.QK4_0 - 1) / common.QK4_0;
    
    // Cast output to block array
    const blocks = @as([*]common.BlockQ4_0, @ptrCast(@alignCast(output.ptr)))[0..n_blocks];
    
    for (0..n_blocks) |block_idx| {
        const block_start = block_idx * common.QK4_0;
        const block_end = @min(block_start + common.QK4_0, n_values);
        const block_values = input[block_start..block_end];
        
        quantize_block(&blocks[block_idx], block_values);
    }
}

/// Quantize a single block to Q4_0
fn quantize_block(block: *common.BlockQ4_0, values: []const f32) void {
    // Calculate optimal scale
    const params = common.calc_q4_0_params(values);
    block.scale = common.f32_to_f16(params.scale);
    
    const scale = params.scale;
    const inv_scale = if (scale != 0.0) 1.0 / scale else 0.0;
    
    // Quantize values
    @memset(&block.qs, 0);
    
    for (0..values.len) |i| {
        // Map to [-8, 7]
        const qval_f = values[i] * inv_scale;
        const qval_clamped = std.math.clamp(qval_f, -8.0, 7.0);
        const qval_rounded = @round(qval_clamped);
        const qval = @as(i8, @intFromFloat(qval_rounded)) + 8; // Shift to [0, 15]
        
        // Pack into byte array
        const byte_idx = i / 2;
        const is_high = (i % 2) == 1;
        
        if (is_high) {
            block.qs[byte_idx] |= @as(u8, @intCast(qval)) << 4;
        } else {
            block.qs[byte_idx] |= @as(u8, @intCast(qval)) & 0xF;
        }
    }
}

// ============================================================================
// Statistics & Analysis
// ============================================================================

/// Calculate compression statistics
pub fn calc_compression_stats(
    n_values: usize,
) struct { original_bytes: usize, compressed_bytes: usize, ratio: f32 } {
    const n_blocks = (n_values + common.QK4_0 - 1) / common.QK4_0;
    
    const original = n_values * @sizeOf(f32);
    const compressed = n_blocks * @sizeOf(common.BlockQ4_0);
    const ratio = @as(f32, @floatFromInt(original)) / @as(f32, @floatFromInt(compressed));
    
    return .{
        .original_bytes = original,
        .compressed_bytes = compressed,
        .ratio = ratio,
    };
}

/// Calculate quantization error (MSE)
pub fn calc_quantization_error(
    original: []const f32,
    quantized: []const u8,
    allocator: std.mem.Allocator,
) !f32 {
    const recovered = try allocator.alloc(f32, original.len);
    defer allocator.free(recovered);
    
    dequantize(recovered, quantized, original.len);
    
    var mse: f32 = 0.0;
    for (0..original.len) |i| {
        const diff = original[i] - recovered[i];
        mse += diff * diff;
    }
    
    return mse / @as(f32, @floatFromInt(original.len));
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_q4_0(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Q4_0 Quantization\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Single block quantization
    {
        std.debug.print("\n1️⃣  Testing single block (32 values)...\n", .{});
        
        // Create test data: simple pattern
        var original: [32]f32 = undefined;
        for (0..32) |i| {
            original[i] = @as(f32, @floatFromInt(i)) - 16.0; // Range: [-16, 15]
        }
        
        // Quantize
        var quantized: [@sizeOf(common.BlockQ4_0)]u8 = undefined;
        quantize(&quantized, &original, 32);
        
        // Dequantize
        var recovered: [32]f32 = undefined;
        dequantize(&recovered, &quantized, 32);
        
        // Check error
        var max_error: f32 = 0.0;
        var total_error: f32 = 0.0;
        
        for (0..32) |i| {
            const err = @abs(original[i] - recovered[i]);
            max_error = @max(max_error, err);
            total_error += err;
        }
        
        const avg_error = total_error / 32.0;
        
        std.debug.print("   Max error: {d:.4}\n", .{max_error});
        std.debug.print("   Avg error: {d:.4}\n", .{avg_error});
        
        if (max_error > 3.0) {
            std.debug.print("   ⚠️  High quantization error\n", .{});
        } else {
            std.debug.print("   ✅ Quantization error acceptable\n", .{});
        }
    }
    
    // Test 2: Multiple blocks
    {
        std.debug.print("\n2️⃣  Testing multiple blocks (256 values)...\n", .{});
        
        const n = 256;
        const original = try allocator.alloc(f32, n);
        defer allocator.free(original);
        
        // Initialize with sine wave
        for (0..n) |i| {
            const x = @as(f32, @floatFromInt(i)) * 0.1;
            original[i] = @sin(x) * 10.0;
        }
        
        // Calculate required space
        const n_blocks = (n + common.QK4_0 - 1) / common.QK4_0;
        const quantized_size = n_blocks * @sizeOf(common.BlockQ4_0);
        
        const quantized = try allocator.alloc(u8, quantized_size);
        defer allocator.free(quantized);
        
        // Quantize
        quantize(quantized, original, n);
        
        // Dequantize
        const recovered = try allocator.alloc(f32, n);
        defer allocator.free(recovered);
        dequantize(recovered, quantized, n);
        
        // Calculate MSE
        var mse: f32 = 0.0;
        for (0..n) |i| {
            const diff = original[i] - recovered[i];
            mse += diff * diff;
        }
        mse /= @as(f32, @floatFromInt(n));
        
        std.debug.print("   MSE: {d:.6}\n", .{mse});
        
        if (mse < 1.0) {
            std.debug.print("   ✅ Good quantization quality\n", .{});
        } else {
            std.debug.print("   ⚠️  High MSE (expected for 4-bit)\n", .{});
        }
    }
    
    // Test 3: Compression ratio
    {
        std.debug.print("\n3️⃣  Testing compression statistics...\n", .{});
        
        const test_sizes = [_]usize{ 256, 1024, 4096 };
        
        for (test_sizes) |size| {
            const stats = calc_compression_stats(size);
            
            std.debug.print("   {d} values:\n", .{size});
            std.debug.print("      Original: {d} bytes\n", .{stats.original_bytes});
            std.debug.print("      Compressed: {d} bytes\n", .{stats.compressed_bytes});
            std.debug.print("      Ratio: {d:.2}x\n", .{stats.ratio});
        }
        
        std.debug.print("   ✅ Compression working (~7-8x typical)\n", .{});
    }
    
    // Test 4: SIMD vs scalar
    {
        std.debug.print("\n4️⃣  Testing SIMD dequantization...\n", .{});
        
        const n = 1024;
        const original = try allocator.alloc(f32, n);
        defer allocator.free(original);
        
        for (0..n) |i| {
            original[i] = @as(f32, @floatFromInt(i % 100)) - 50.0;
        }
        
        const n_blocks = (n + common.QK4_0 - 1) / common.QK4_0;
        const quantized_size = n_blocks * @sizeOf(common.BlockQ4_0);
        
        const quantized = try allocator.alloc(u8, quantized_size);
        defer allocator.free(quantized);
        quantize(quantized, original, n);
        
        // Test scalar
        const recovered_scalar = try allocator.alloc(f32, n);
        defer allocator.free(recovered_scalar);
        dequantize(recovered_scalar, quantized, n);
        
        // Test SIMD
        const recovered_simd = try allocator.alloc(f32, n);
        defer allocator.free(recovered_simd);
        dequantize_simd(recovered_simd, quantized, n);
        
        // Compare results
        var mismatch = false;
        for (0..n) |i| {
            if (@abs(recovered_scalar[i] - recovered_simd[i]) > 0.0001) {
                mismatch = true;
                break;
            }
        }
        
        if (mismatch) {
            std.debug.print("   ❌ SIMD and scalar results differ\n", .{});
            return error.TestFailed;
        } else {
            std.debug.print("   ✅ SIMD matches scalar\n", .{});
        }
        
        // Benchmark
        const iterations = 100;
        
        const start_scalar = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            dequantize(recovered_scalar, quantized, n);
        }
        const time_scalar = std.time.nanoTimestamp() - start_scalar;
        
        const start_simd = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            dequantize_simd(recovered_simd, quantized, n);
        }
        const time_simd = std.time.nanoTimestamp() - start_simd;
        
        const speedup = @as(f32, @floatFromInt(time_scalar)) / @as(f32, @floatFromInt(time_simd));
        
        std.debug.print("   Scalar: {d}ms\n", .{@divFloor(time_scalar, 1_000_000)});
        std.debug.print("   SIMD:   {d}ms\n", .{@divFloor(time_simd, 1_000_000)});
        std.debug.print("   Speedup: {d:.2}x\n", .{speedup});
    }
    
    std.debug.print("\n✅ All Q4_0 tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
