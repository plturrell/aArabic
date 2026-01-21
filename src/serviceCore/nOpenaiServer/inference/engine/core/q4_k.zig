const std = @import("std");

/// Q4_K quantization format support
/// Block size: 256 floats
/// Each block: 2 scales (f16) + mins (12 bytes) + 128 bytes packed 4-bit values
/// Total: 144 bytes per 256 values (~4.5 bits per value)

pub const BLOCK_SIZE: usize = 256;
pub const BLOCK_BYTES: usize = 144;
pub const K_SCALE_SIZE: usize = 12;

/// Q4_K block structure (144 bytes for 256 values)
pub const BlockQ4_K = extern struct {
    d: u16,                      // super-block scale (f16)
    dmin: u16,                   // super-block min (f16)
    scales: [K_SCALE_SIZE]u8,    // scales and mins for sub-blocks
    qs: [128]u8,                 // packed 4-bit quantized values
};

/// Convert f16 bits to f32
inline fn f16ToF32(bits: u16) f32 {
    const sign: u32 = @as(u32, bits >> 15) << 31;
    const exp_bits = (bits >> 10) & 0x1F;
    const mant_bits = bits & 0x3FF;
    if (exp_bits == 0) {
        if (mant_bits == 0) return @bitCast(sign);
        const mant_f: f32 = @floatFromInt(mant_bits);
        const val = mant_f * (1.0 / 16777216.0);
        return if (sign != 0) -val else val;
    }
    if (exp_bits == 31) {
        return if (mant_bits != 0) std.math.nan(f32) else @bitCast(sign | 0x7F800000);
    }
    const exp: u32 = @as(u32, exp_bits) + (127 - 15);
    const mant: u32 = @as(u32, mant_bits) << 13;
    return @bitCast(sign | (exp << 23) | mant);
}

/// Dequantize a single Q4_K block to f32 output
pub fn dequantizeBlock(output: []f32, block: *const BlockQ4_K) void {
    const d = f16ToF32(block.d);
    const dmin = f16ToF32(block.dmin);
    
    var out_idx: usize = 0;
    
    // Process 8 sub-blocks of 32 values each
    for (0..8) |sb| {
        // Extract scale and min for this sub-block from packed scales
        const scale_idx = sb;
        var sc: u8 = undefined;
        var m: u8 = undefined;
        
        if (sb < 4) {
            sc = block.scales[sb] & 0x3F;
            m = block.scales[sb + 4] & 0x3F;
        } else {
            const idx = sb - 4;
            sc = (block.scales[idx] >> 6) | ((block.scales[idx + 4] >> 4) & 0x0C);
            m = (block.scales[idx + 4] >> 6) | ((block.scales[idx + 8] >> 4) & 0x0C);
        }
        
        const scale = d * @as(f32, @floatFromInt(sc));
        const min_val = dmin * @as(f32, @floatFromInt(m));
        
        // Process 32 values (16 bytes, 2 values per byte)
        const qs_offset = scale_idx * 16;
        for (0..16) |i| {
            const byte = block.qs[qs_offset + i];
            const lo: i8 = @intCast(byte & 0x0F);
            const hi: i8 = @intCast(byte >> 4);
            
            output[out_idx] = scale * @as(f32, @floatFromInt(lo)) - min_val;
            output[out_idx + 1] = scale * @as(f32, @floatFromInt(hi)) - min_val;
            out_idx += 2;
        }
    }
}

/// Dequantize raw Q4_K data to f32
pub fn dequantize(input: []const u8, output: []f32) void {
    const num_blocks = output.len / BLOCK_SIZE;
    var in_offset: usize = 0;
    var out_offset: usize = 0;
    
    for (0..num_blocks) |_| {
        const block = @as(*const BlockQ4_K, @ptrCast(@alignCast(&input[in_offset])));
        dequantizeBlock(output[out_offset..][0..BLOCK_SIZE], block);
        in_offset += BLOCK_BYTES;
        out_offset += BLOCK_SIZE;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "BlockQ4_K size" {
    try std.testing.expectEqual(@as(usize, BLOCK_BYTES), @sizeOf(BlockQ4_K));
}

test "dequantize zero block" {
    var block = std.mem.zeroes(BlockQ4_K);
    block.d = 0x3C00; // 1.0 in f16
    
    var output: [BLOCK_SIZE]f32 = undefined;
    dequantizeBlock(&output, &block);
    
    // With zero scales/mins, all outputs should be 0 or near-zero
    for (output) |val| {
        try std.testing.expect(@abs(val) < 1e-6);
    }
}

test "dequantize basic values" {
    var block = std.mem.zeroes(BlockQ4_K);
    block.d = 0x3C00;    // 1.0 in f16
    block.dmin = 0x0000; // 0.0 in f16
    block.scales[0] = 1; // scale = 1 for first sub-block
    block.qs[0] = 0x21;  // values: 1 and 2
    
    var output: [BLOCK_SIZE]f32 = undefined;
    dequantizeBlock(&output, &block);
    
    // First two values: scale(1) * value - 0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 0.01);
}

test "dequantize multi-block" {
    var input: [BLOCK_BYTES * 2]u8 = undefined;
    @memset(&input, 0);
    
    var output: [BLOCK_SIZE * 2]f32 = undefined;
    dequantize(&input, &output);
    
    // All zeros input should produce all zeros output
    for (output) |val| {
        try std.testing.expect(@abs(val) < 1e-6);
    }
}

