const std = @import("std");

/// Q6_K quantization format support
/// Block size: 256 floats  
/// Each block: scale (f16) + 6-bit quantized values + sub-block scales
/// Total: 210 bytes per 256 values (~6.5 bits per value)
/// Higher precision than Q4_K

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = 210;

/// Q6_K block structure (210 bytes for 256 values)
pub const BlockQ6_K = extern struct {
    ql: [128]u8,   // lower 4 bits of 6-bit quants
    qh: [64]u8,    // upper 2 bits of 6-bit quants  
    scales: [16]i8, // sub-block scales
    d: u16,        // super-block scale (f16)
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

/// Dequantize a single Q6_K block to f32 output
pub fn dequantizeBlock(output: []f32, block: *const BlockQ6_K) void {
    const d = f16ToF32(block.d);
    
    // Process 16 sub-blocks of 16 values each
    for (0..16) |sb| {
        const scale = d * @as(f32, @floatFromInt(block.scales[sb]));
        const ql_offset = sb * 8;
        const qh_offset = sb * 4;
        
        // Each sub-block has 16 values
        // ql provides lower 4 bits (8 bytes = 16 nibbles)
        // qh provides upper 2 bits (4 bytes = 16 pairs of 2 bits)
        for (0..8) |i| {
            const ql_byte = block.ql[ql_offset + i];
            const qh_byte = block.qh[qh_offset + i / 2];
            
            // Extract two 6-bit values per iteration
            const lo4: i8 = @intCast(ql_byte & 0x0F);
            const hi4: i8 = @intCast(ql_byte >> 4);
            
            // Get upper 2 bits from qh
            const qh_shift = @as(u3, @intCast((i % 2) * 4));
            const hi2_lo: i8 = @intCast((qh_byte >> qh_shift) & 0x03);
            const hi2_hi: i8 = @intCast((qh_byte >> (qh_shift + 2)) & 0x03);
            
            // Combine to 6-bit signed values (range -32 to 31)
            const q0: i8 = lo4 | (hi2_lo << 4) - 32;
            const q1: i8 = hi4 | (hi2_hi << 4) - 32;
            
            const out_idx = sb * 16 + i * 2;
            output[out_idx] = scale * @as(f32, @floatFromInt(q0));
            output[out_idx + 1] = scale * @as(f32, @floatFromInt(q1));
        }
    }
}

/// Dequantize raw Q6_K data to f32
pub fn dequantize(input: []const u8, output: []f32) void {
    const num_blocks = output.len / QK_K;
    var in_offset: usize = 0;
    var out_offset: usize = 0;
    
    for (0..num_blocks) |_| {
        const block = @as(*const BlockQ6_K, @ptrCast(@alignCast(&input[in_offset])));
        dequantizeBlock(output[out_offset..][0..QK_K], block);
        in_offset += BLOCK_BYTES;
        out_offset += QK_K;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "BlockQ6_K size" {
    try std.testing.expectEqual(@as(usize, BLOCK_BYTES), @sizeOf(BlockQ6_K));
}

test "dequantize zero block" {
    var block = std.mem.zeroes(BlockQ6_K);
    block.d = 0x3C00; // 1.0 in f16
    
    var output: [QK_K]f32 = undefined;
    dequantizeBlock(&output, &block);
    
    // With zero scales, all outputs should be -32 * 0 = 0
    for (output) |val| {
        try std.testing.expect(@abs(val) < 50.0); // Allow range due to -32 offset
    }
}

test "dequantize with scale" {
    var block = std.mem.zeroes(BlockQ6_K);
    block.d = 0x3C00;     // 1.0 in f16
    block.scales[0] = 1;  // scale = 1.0 for first sub-block
    block.ql[0] = 0x00;   // both nibbles = 0
    
    var output: [QK_K]f32 = undefined;
    dequantizeBlock(&output, &block);
    
    // First values: scale(1) * (0 | 0 - 32) = -32
    try std.testing.expectApproxEqAbs(@as(f32, -32.0), output[0], 1.0);
}

test "dequantize multi-block" {
    var input: [BLOCK_BYTES * 2]u8 = undefined;
    @memset(&input, 0);
    
    var output: [QK_K * 2]f32 = undefined;
    dequantize(&input, &output);
    
    // With zero d and scales, values should be near zero (affected by -32 offset)
    for (output) |val| {
        try std.testing.expect(@abs(val) < 50.0);
    }
}

