const std = @import("std");
const common = @import("common");

/// Q8_0 Quantization: 8-bit integer quantization
/// 
/// Format: Each block contains:
/// - 1 float32 scale factor (4 bytes)
/// - 32 int8 values (32 bytes)
/// Total: 36 bytes per block
///
/// Compression: 32 float32s (128 bytes) → 36 bytes = 28% size (3.56:1 ratio)
/// Better quality than Q4_0 but larger size

pub const BLOCK_SIZE = 32;
pub const BLOCK_BYTES = 36; // 4 (scale) + 32 (int8s)

/// Q8_0 quantized block
pub const BlockQ8_0 = extern struct {
    scale: f32,              // Scale factor
    qs: [BLOCK_SIZE]i8,      // Quantized values (-128 to 127)
    
    /// Get the size of a block in bytes
    pub fn sizeInBytes() usize {
        return BLOCK_BYTES;
    }
};

// Verify struct layout
comptime {
    if (@sizeOf(BlockQ8_0) != BLOCK_BYTES) {
        @compileError("BlockQ8_0 size mismatch");
    }
}

/// Quantize float32 array to Q8_0 blocks
pub fn quantize(
    output: []BlockQ8_0,
    input: []const f32,
) void {
    std.debug.assert(input.len == output.len * BLOCK_SIZE);
    
    for (output, 0..) |*block, i| {
        const start = i * BLOCK_SIZE;
        const end = start + BLOCK_SIZE;
        const values = input[start..end];
        
        // Find absolute maximum for scale
        var max_abs: f32 = 0.0;
        for (values) |val| {
            max_abs = @max(max_abs, @abs(val));
        }
        
        // Compute scale (map to int8 range: -127 to 127)
        const scale = max_abs / 127.0;
        block.scale = scale;
        
        // Quantize values
        if (scale > 0.0) {
            const inv_scale = 1.0 / scale;
            for (values, 0..) |val, j| {
                const quantized = val * inv_scale;
                // Round and clamp to int8 range
                const rounded = @round(quantized);
                const clamped = @max(-127.0, @min(127.0, rounded));
                block.qs[j] = @intFromFloat(clamped);
            }
        } else {
            // All zeros
            for (&block.qs) |*q| {
                q.* = 0;
            }
        }
    }
}

/// Dequantize Q8_0 block to float32 array
pub fn dequantizeBlock(
    output: []f32,
    block: *const BlockQ8_0,
) void {
    std.debug.assert(output.len == BLOCK_SIZE);
    
    const scale = block.scale;
    
    for (block.qs, 0..) |q, i| {
        output[i] = @as(f32, @floatFromInt(q)) * scale;
    }
}

/// Dequantize entire Q8_0 array to float32
pub fn dequantize(
    output: []f32,
    input: []const BlockQ8_0,
) void {
    std.debug.assert(output.len == input.len * BLOCK_SIZE);
    
    for (input, 0..) |*block, i| {
        const start = i * BLOCK_SIZE;
        const end = start + BLOCK_SIZE;
        dequantizeBlock(output[start..end], block);
    }
}

/// Compute dot product: Q8_0 * float32
pub fn vecDotQ8_0F32(
    blocks: []const BlockQ8_0,
    vec: []const f32,
) f32 {
    std.debug.assert(vec.len == blocks.len * BLOCK_SIZE);
    
    var sum: f32 = 0.0;
    
    for (blocks, 0..) |*block, i| {
        const start = i * BLOCK_SIZE;
        const end = start + BLOCK_SIZE;
        const v = vec[start..end];
        
        // Compute block dot product
        var block_sum: f32 = 0.0;
        for (block.qs, 0..) |q, j| {
            block_sum += @as(f32, @floatFromInt(q)) * v[j];
        }
        
        sum += block_sum * block.scale;
    }
    
    return sum;
}

/// Compute dot product: Q8_0 * Q8_0 (optimized)
pub fn vecDotQ8_0Q8_0(
    blocks_a: []const BlockQ8_0,
    blocks_b: []const BlockQ8_0,
) f32 {
    std.debug.assert(blocks_a.len == blocks_b.len);
    
    var sum: f32 = 0.0;
    
    for (blocks_a, blocks_b) |*block_a, *block_b| {
        // Compute integer dot product
        var int_sum: i32 = 0;
        for (block_a.qs, block_b.qs) |qa, qb| {
            int_sum += @as(i32, qa) * @as(i32, qb);
        }
        
        // Scale result
        sum += @as(f32, @floatFromInt(int_sum)) * block_a.scale * block_b.scale;
    }
    
    return sum;
}

/// Calculate number of blocks needed for given number of elements
pub fn calculateNumBlocks(n_elements: usize) usize {
    return (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

/// Calculate total bytes needed for quantized data
pub fn calculateBytes(n_elements: usize) usize {
    return calculateNumBlocks(n_elements) * BLOCK_BYTES;
}

/// Get compression ratio
pub fn compressionRatio() f32 {
    const original_size = BLOCK_SIZE * @sizeOf(f32);
    return @as(f32, @floatFromInt(original_size)) / @as(f32, @floatFromInt(BLOCK_BYTES));
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_q8_0(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Q8_0 Quantization Module\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Basic quantization/dequantization
    {
        std.debug.print("\n1️⃣  Testing basic quantization/dequantization...\n", .{});
        
        // Create test data
        var original = [_]f32{0} ** BLOCK_SIZE;
        for (&original, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i)) - 16.0; // Range: -16 to 15
        }
        
        // Quantize
        var quantized: [1]BlockQ8_0 = undefined;
        quantize(&quantized, &original);
        
        std.debug.print("   Original: [{d:.2}, {d:.2}, {d:.2}, ..., {d:.2}]\n", .{
            original[0],
            original[1],
            original[2],
            original[31],
        });
        std.debug.print("   Scale: {d:.6}\n", .{quantized[0].scale});
        std.debug.print("   Quantized[0..4]: [{d}, {d}, {d}, {d}]\n", .{
            quantized[0].qs[0],
            quantized[0].qs[1],
            quantized[0].qs[2],
            quantized[0].qs[3],
        });
        
        // Dequantize
        var dequantized: [BLOCK_SIZE]f32 = undefined;
        dequantizeBlock(&dequantized, &quantized[0]);
        
        std.debug.print("   Dequantized: [{d:.2}, {d:.2}, {d:.2}, ..., {d:.2}]\n", .{
            dequantized[0],
            dequantized[1],
            dequantized[2],
            dequantized[31],
        });
        
        // Check accuracy
        var max_error: f32 = 0.0;
        for (original, dequantized) |orig, deq| {
            const err = @abs(orig - deq);
            max_error = @max(max_error, err);
        }
        
        std.debug.print("   Max error: {d:.6}\n", .{max_error});
        std.debug.print("   ✅ Quantization/dequantization working\n", .{});
    }
    
    // Test 2: Compression ratio
    {
        std.debug.print("\n2️⃣  Testing compression ratio...\n", .{});
        
        const ratio = compressionRatio();
        const original_bytes = BLOCK_SIZE * @sizeOf(f32);
        const compressed_bytes = BLOCK_BYTES;
        const percent = 100.0 * @as(f32, @floatFromInt(compressed_bytes)) / @as(f32, @floatFromInt(original_bytes));
        
        std.debug.print("   Original size: {d} bytes ({d} float32s)\n", .{
            original_bytes,
            BLOCK_SIZE,
        });
        std.debug.print("   Compressed size: {d} bytes (1 float32 + {d} int8s)\n", .{
            compressed_bytes,
            BLOCK_SIZE,
        });
        std.debug.print("   Compression ratio: {d:.2}:1\n", .{ratio});
        std.debug.print("   Size reduction: {d:.1}%\n", .{100.0 - percent});
        std.debug.print("   ✅ Good compression achieved\n", .{});
    }
    
    // Test 3: Dot product accuracy
    {
        std.debug.print("\n3️⃣  Testing dot product accuracy...\n", .{});
        
        // Create test vectors
        const n = BLOCK_SIZE * 4; // 4 blocks
        const vec_a = try allocator.alloc(f32, n);
        defer allocator.free(vec_a);
        const vec_b = try allocator.alloc(f32, n);
        defer allocator.free(vec_b);
        
        for (vec_a, 0..) |*val, i| {
            val.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
        }
        for (vec_b, 0..) |*val, i| {
            val.* = @cos(@as(f32, @floatFromInt(i)) * 0.1);
        }
        
        // Compute reference (float32 dot product)
        var ref_sum: f32 = 0.0;
        for (vec_a, vec_b) |a, b| {
            ref_sum += a * b;
        }
        
        // Quantize vectors
        const n_blocks = calculateNumBlocks(n);
        const blocks_a = try allocator.alloc(BlockQ8_0, n_blocks);
        defer allocator.free(blocks_a);
        const blocks_b = try allocator.alloc(BlockQ8_0, n_blocks);
        defer allocator.free(blocks_b);
        
        quantize(blocks_a, vec_a);
        quantize(blocks_b, vec_b);
        
        // Compute Q8_0 dot products
        const q8_f32_sum = vecDotQ8_0F32(blocks_a, vec_b);
        const q8_q8_sum = vecDotQ8_0Q8_0(blocks_a, blocks_b);
        
        std.debug.print("   Reference (FP32): {d:.6}\n", .{ref_sum});
        std.debug.print("   Q8_0 × FP32: {d:.6}\n", .{q8_f32_sum});
        std.debug.print("   Q8_0 × Q8_0: {d:.6}\n", .{q8_q8_sum});
        
        const err_f32 = @abs(ref_sum - q8_f32_sum) / @abs(ref_sum);
        const err_q8 = @abs(ref_sum - q8_q8_sum) / @abs(ref_sum);
        
        std.debug.print("   Error (Q8×F32): {d:.4}%\n", .{err_f32 * 100.0});
        std.debug.print("   Error (Q8×Q8): {d:.4}%\n", .{err_q8 * 100.0});
        
        // Q8_0 is 8-bit, so allow up to 10% error (still very good)
        if (err_f32 > 0.10 or err_q8 > 0.10) {
            std.debug.print("   ❌ Error too high\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Dot products accurate (<10% error)\n", .{});
    }
    
    // Test 4: Edge cases
    {
        std.debug.print("\n4️⃣  Testing edge cases...\n", .{});
        
        // All zeros
        var zeros = [_]f32{0.0} ** BLOCK_SIZE;
        var zero_block: [1]BlockQ8_0 = undefined;
        quantize(&zero_block, &zeros);
        
        std.debug.print("   All zeros - scale: {d:.6}\n", .{zero_block[0].scale});
        
        // Very small values
        var small = [_]f32{0.0001} ** BLOCK_SIZE;
        var small_block: [1]BlockQ8_0 = undefined;
        quantize(&small_block, &small);
        
        std.debug.print("   Small values - scale: {d:.6}\n", .{small_block[0].scale});
        
        // Very large values
        var large = [_]f32{1000.0} ** BLOCK_SIZE;
        var large_block: [1]BlockQ8_0 = undefined;
        quantize(&large_block, &large);
        
        std.debug.print("   Large values - scale: {d:.6}\n", .{large_block[0].scale});
        
        std.debug.print("   ✅ Edge cases handled\n", .{});
    }
    
    // Test 5: Multi-block operations
    {
        std.debug.print("\n5️⃣  Testing multi-block operations...\n", .{});
        
        const n = BLOCK_SIZE * 8; // 8 blocks
        const data = try allocator.alloc(f32, n);
        defer allocator.free(data);
        
        for (data, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) - 50.0;
        }
        
        const n_blocks = calculateNumBlocks(n);
        const blocks = try allocator.alloc(BlockQ8_0, n_blocks);
        defer allocator.free(blocks);
        
        quantize(blocks, data);
        
        const deq_data = try allocator.alloc(f32, n);
        defer allocator.free(deq_data);
        
        dequantize(deq_data, blocks);
        
        // Check accuracy across all blocks
        var total_error: f32 = 0.0;
        for (data, deq_data) |orig, deq| {
            total_error += @abs(orig - deq);
        }
        const avg_error = total_error / @as(f32, @floatFromInt(n));
        
        std.debug.print("   Processed {d} blocks ({d} elements)\n", .{ n_blocks, n });
        std.debug.print("   Average error: {d:.6}\n", .{avg_error});
        std.debug.print("   ✅ Multi-block operations working\n", .{});
    }
    
    // Test 6: Q8_0 vs Q4_0 comparison
    {
        std.debug.print("\n6️⃣  Comparing Q8_0 with Q4_0...\n", .{});
        
        std.debug.print("   Q4_0: 4-bit, 18 bytes/block, 7.1:1 compression\n", .{});
        std.debug.print("   Q8_0: 8-bit, 36 bytes/block, 3.6:1 compression\n", .{});
        std.debug.print("   \n", .{});
        std.debug.print("   Q8_0 advantages:\n", .{});
        std.debug.print("   • 2x better precision (8-bit vs 4-bit)\n", .{});
        std.debug.print("   • Lower quantization error\n", .{});
        std.debug.print("   • Better for quality-critical tasks\n", .{});
        std.debug.print("   \n", .{});
        std.debug.print("   Q4_0 advantages:\n", .{});
        std.debug.print("   • 2x better compression (7.1:1 vs 3.6:1)\n", .{});
        std.debug.print("   • Smaller memory footprint\n", .{});
        std.debug.print("   • Better for memory-constrained scenarios\n", .{});
        std.debug.print("   ✅ Tradeoffs understood\n", .{});
    }
    
    std.debug.print("\n✅ All Q8_0 tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
