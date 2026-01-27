// Numeric quantization module for zig-libc
// Implements Float16 conversion and 4-bit/8-bit quantization
// Pure Zig implementations with C-compatible exports

const std = @import("std");
const math = std.math;

// ============================================================================
// IEEE 754 binary16 (Float16) constants
// ============================================================================

const F16_EXP_BITS: u32 = 5;
const F16_MANT_BITS: u32 = 10;
const F16_EXP_BIAS: i32 = 15;
const F16_EXP_MAX: u32 = 31;

const F32_EXP_BITS: u32 = 8;
const F32_MANT_BITS: u32 = 23;
const F32_EXP_BIAS: i32 = 127;

// ============================================================================
// Float16 Conversion (IEEE 754 binary16)
// ============================================================================

/// Convert f32 to f16 bits (IEEE 754 binary16)
pub fn f32_to_f16(val: f32) u16 {
    const bits: u32 = @bitCast(val);
    const sign: u32 = (bits >> 31) & 1;
    const exp: i32 = @as(i32, @intCast((bits >> 23) & 0xFF)) - F32_EXP_BIAS;
    const mant: u32 = bits & 0x7FFFFF;

    // Handle special cases
    if (exp == 128) {
        // Inf or NaN
        if (mant == 0) {
            // Infinity
            return @intCast((sign << 15) | (F16_EXP_MAX << 10));
        } else {
            // NaN - preserve some mantissa bits
            return @intCast((sign << 15) | (F16_EXP_MAX << 10) | (mant >> 13) | 1);
        }
    }

    // Calculate f16 exponent
    const f16_exp = exp + F16_EXP_BIAS;

    if (f16_exp >= @as(i32, F16_EXP_MAX)) {
        // Overflow to infinity
        return @intCast((sign << 15) | (F16_EXP_MAX << 10));
    }

    if (f16_exp <= 0) {
        // Subnormal or zero
        if (f16_exp < -10) {
            // Too small, round to zero
            return @intCast(sign << 15);
        }
        // Subnormal: shift mantissa and add implicit 1
        const shift: u5 = @intCast(1 - f16_exp);
        const subnorm_mant = (0x800000 | mant) >> (shift + 13);
        return @intCast((sign << 15) | subnorm_mant);
    }

    // Normal case: round mantissa
    const f16_mant = (mant + 0x1000) >> 13; // Round to nearest
    if (f16_mant >= 0x400) {
        // Mantissa overflow, increment exponent
        return @intCast((sign << 15) | (@as(u32, @intCast(f16_exp + 1)) << 10));
    }
    return @intCast((sign << 15) | (@as(u32, @intCast(f16_exp)) << 10) | f16_mant);
}

/// Convert f16 bits to f32 (IEEE 754 binary16)
pub fn f16_to_f32(val: u16) f32 {
    const sign: u32 = (@as(u32, val) >> 15) & 1;
    const exp: u32 = (@as(u32, val) >> 10) & 0x1F;
    const mant: u32 = @as(u32, val) & 0x3FF;

    var result: u32 = undefined;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            result = sign << 31;
        } else {
            // Subnormal - normalize it
            var m = mant;
            var e: i32 = -14;
            while ((m & 0x400) == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // Remove implicit 1
            const f32_exp: u32 = @intCast(e + F32_EXP_BIAS);
            result = (sign << 31) | (f32_exp << 23) | (m << 13);
        }
    } else if (exp == F16_EXP_MAX) {
        // Inf or NaN
        result = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        // Normal
        const f32_exp: u32 = @intCast(@as(i32, @intCast(exp)) - F16_EXP_BIAS + F32_EXP_BIAS);
        result = (sign << 31) | (f32_exp << 23) | (mant << 13);
    }

    return @bitCast(result);
}

/// Batch convert f32 array to f16 bits
pub fn f32_array_to_f16(src: [*]const f32, dst: [*]u16, n: usize) void {
    for (0..n) |i| {
        dst[i] = f32_to_f16(src[i]);
    }
}

/// Batch convert f16 bits to f32 array
pub fn f16_array_to_f32(src: [*]const u16, dst: [*]f32, n: usize) void {
    for (0..n) |i| {
        dst[i] = f16_to_f32(src[i]);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Pack two 4-bit values (0-15) into one byte
pub fn pack_4bit(low: u4, high: u4) u8 {
    return @as(u8, low) | (@as(u8, high) << 4);
}

/// Unpack byte into two 4-bit values
pub fn unpack_4bit(byte: u8) struct { low: u4, high: u4 } {
    return .{
        .low = @truncate(byte & 0x0F),
        .high = @truncate(byte >> 4),
    };
}

/// Calculate symmetric quantization scale from array values
/// Returns max absolute value / max_quant_level
pub fn calc_scale_symmetric(values: [*]const f32, n: usize) f32 {
    if (n == 0) return 1.0;
    var max_abs: f32 = 0.0;
    for (0..n) |i| {
        const abs_val = @abs(values[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    return if (max_abs == 0.0) 1.0 else max_abs;
}

/// Calculate asymmetric quantization parameters
/// Returns scale and zero_point for mapping [min, max] to quantized range
pub fn calc_scale_asymmetric(values: [*]const f32, n: usize) struct { scale: f32, zero_point: f32 } {
    if (n == 0) return .{ .scale = 1.0, .zero_point = 0.0 };
    var min_val: f32 = values[0];
    var max_val: f32 = values[0];
    for (1..n) |i| {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }
    const range = max_val - min_val;
    const scale = if (range == 0.0) 1.0 else range;
    return .{ .scale = scale, .zero_point = min_val };
}

// ============================================================================
// 4-bit Quantization
// ============================================================================

/// Q4 block: scale (f16) + 16 bytes of packed 4-bit values (32 values total)
pub const BlockQ4 = extern struct {
    scale: u16, // f16 stored as bits
    data: [16]u8, // 32 values packed as 4-bit pairs
};

/// Quantize single value to 4-bit (symmetric, range -8 to 7)
pub fn quantize_4bit_value(val: f32, scale: f32) i4 {
    if (scale == 0.0) return 0;
    const q = val / scale * 7.0; // Scale to [-7, 7] range
    const clamped = @max(-8.0, @min(7.0, q));
    return @intFromFloat(math.round(clamped));
}

/// Dequantize single 4-bit value
pub fn dequantize_4bit_value(qval: i4, scale: f32) f32 {
    return @as(f32, @floatFromInt(qval)) * scale / 7.0;
}

/// Quantize f32 array to Q4 blocks
/// src: input f32 array
/// n: number of f32 values (should be multiple of 32)
/// blocks: output array of BlockQ4 (size = n / 32)
pub fn quantize_q4(src: [*]const f32, n: usize, blocks: [*]BlockQ4) void {
    const block_size: usize = 32;
    const n_blocks = n / block_size;

    for (0..n_blocks) |b| {
        const block_start = b * block_size;
        // Calculate block scale
        var max_abs: f32 = 0.0;
        for (0..block_size) |i| {
            const abs_val = @abs(src[block_start + i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        const scale = if (max_abs == 0.0) 1.0 else max_abs;
        blocks[b].scale = f32_to_f16(scale);

        // Quantize and pack values
        for (0..16) |i| {
            const qv0 = quantize_4bit_value(src[block_start + i * 2], scale);
            const qv1 = quantize_4bit_value(src[block_start + i * 2 + 1], scale);
            // Store as unsigned with offset (i4 range -8..7 -> u4 range 0..15)
            const uv0: u4 = @bitCast(qv0);
            const uv1: u4 = @bitCast(qv1);
            blocks[b].data[i] = pack_4bit(uv0, uv1);
        }
    }
}

/// Dequantize Q4 blocks to f32 array
pub fn dequantize_q4(blocks: [*]const BlockQ4, n_blocks: usize, dst: [*]f32) void {
    const block_size: usize = 32;

    for (0..n_blocks) |b| {
        const block_start = b * block_size;
        const scale = f16_to_f32(blocks[b].scale);

        for (0..16) |i| {
            const unpacked = unpack_4bit(blocks[b].data[i]);
            const v0: i4 = @bitCast(unpacked.low);
            const v1: i4 = @bitCast(unpacked.high);
            dst[block_start + i * 2] = dequantize_4bit_value(v0, scale);
            dst[block_start + i * 2 + 1] = dequantize_4bit_value(v1, scale);
        }
    }
}


// ============================================================================
// 8-bit Quantization
// ============================================================================

/// Q8 block: scale (f16) + 32 bytes of i8 values
pub const BlockQ8 = extern struct {
    scale: u16, // f16 stored as bits
    data: [32]i8, // 32 quantized values
};

/// Quantize single value to 8-bit (symmetric, range -127 to 127)
pub fn quantize_8bit_value(val: f32, scale: f32) i8 {
    if (scale == 0.0) return 0;
    const q = val / scale * 127.0;
    const clamped = @max(-127.0, @min(127.0, q));
    return @intFromFloat(math.round(clamped));
}

/// Dequantize single 8-bit value
pub fn dequantize_8bit_value(qval: i8, scale: f32) f32 {
    return @as(f32, @floatFromInt(qval)) * scale / 127.0;
}

/// Quantize f32 array to Q8 blocks
/// src: input f32 array
/// n: number of f32 values (should be multiple of 32)
/// blocks: output array of BlockQ8 (size = n / 32)
pub fn quantize_q8(src: [*]const f32, n: usize, blocks: [*]BlockQ8) void {
    const block_size: usize = 32;
    const n_blocks = n / block_size;

    for (0..n_blocks) |b| {
        const block_start = b * block_size;
        // Calculate block scale
        var max_abs: f32 = 0.0;
        for (0..block_size) |i| {
            const abs_val = @abs(src[block_start + i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        const scale = if (max_abs == 0.0) 1.0 else max_abs;
        blocks[b].scale = f32_to_f16(scale);

        // Quantize values
        for (0..block_size) |i| {
            blocks[b].data[i] = quantize_8bit_value(src[block_start + i], scale);
        }
    }
}

/// Dequantize Q8 blocks to f32 array
pub fn dequantize_q8(blocks: [*]const BlockQ8, n_blocks: usize, dst: [*]f32) void {
    const block_size: usize = 32;

    for (0..n_blocks) |b| {
        const block_start = b * block_size;
        const scale = f16_to_f32(blocks[b].scale);

        for (0..block_size) |i| {
            dst[block_start + i] = dequantize_8bit_value(blocks[b].data[i], scale);
        }
    }
}

// ============================================================================
// C-compatible exports with n_ prefix
// ============================================================================

pub export fn n_f32_to_f16(val: f32) u16 {
    return f32_to_f16(val);
}

pub export fn n_f16_to_f32(val: u16) f32 {
    return f16_to_f32(val);
}

pub export fn n_f32_array_to_f16(src: [*]const f32, dst: [*]u16, n: usize) void {
    f32_array_to_f16(src, dst, n);
}

pub export fn n_f16_array_to_f32(src: [*]const u16, dst: [*]f32, n: usize) void {
    f16_array_to_f32(src, dst, n);
}

pub export fn n_pack_4bit(low: u8, high: u8) u8 {
    return pack_4bit(@truncate(low), @truncate(high));
}

pub export fn n_unpack_4bit_low(byte: u8) u8 {
    return unpack_4bit(byte).low;
}

pub export fn n_unpack_4bit_high(byte: u8) u8 {
    return unpack_4bit(byte).high;
}

pub export fn n_calc_scale_symmetric(values: [*]const f32, n: usize) f32 {
    return calc_scale_symmetric(values, n);
}

pub export fn n_quantize_q4(src: [*]const f32, n: usize, blocks: [*]BlockQ4) void {
    quantize_q4(src, n, blocks);
}

pub export fn n_dequantize_q4(blocks: [*]const BlockQ4, n_blocks: usize, dst: [*]f32) void {
    dequantize_q4(blocks, n_blocks, dst);
}

pub export fn n_quantize_q8(src: [*]const f32, n: usize, blocks: [*]BlockQ8) void {
    quantize_q8(src, n, blocks);
}

pub export fn n_dequantize_q8(blocks: [*]const BlockQ8, n_blocks: usize, dst: [*]f32) void {
    dequantize_q8(blocks, n_blocks, dst);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "f16 conversion roundtrip" {
    const values = [_]f32{ 0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.00006103515625 };
    for (values) |v| {
        const h = f32_to_f16(v);
        const back = f16_to_f32(h);
        try testing.expectApproxEqRel(v, back, 0.001);
    }
}

test "f16 infinity and nan" {
    // Infinity
    const inf_h = f32_to_f16(math.inf(f32));
    try testing.expect(f16_to_f32(inf_h) == math.inf(f32));

    // Negative infinity
    const ninf_h = f32_to_f16(-math.inf(f32));
    try testing.expect(f16_to_f32(ninf_h) == -math.inf(f32));

    // NaN
    const nan_h = f32_to_f16(math.nan(f32));
    try testing.expect(math.isNan(f16_to_f32(nan_h)));
}

test "pack_4bit and unpack_4bit" {
    const packed_byte = pack_4bit(5, 10);
    const unpacked = unpack_4bit(packed_byte);
    try testing.expectEqual(@as(u4, 5), unpacked.low);
    try testing.expectEqual(@as(u4, 10), unpacked.high);
}

test "q4 quantization roundtrip" {
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16)) * 0.1;
    }

    var blocks: [1]BlockQ4 = undefined;
    quantize_q4(&src, 32, &blocks);

    var dst: [32]f32 = undefined;
    dequantize_q4(&blocks, 1, &dst);

    // Check approximate match (quantization has error)
    for (0..32) |i| {
        try testing.expectApproxEqAbs(src[i], dst[i], 0.3);
    }
}

test "q8 quantization roundtrip" {
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16)) * 0.1;
    }

    var blocks: [1]BlockQ8 = undefined;
    quantize_q8(&src, 32, &blocks);

    var dst: [32]f32 = undefined;
    dequantize_q8(&blocks, 1, &dst);

    // Check approximate match (Q8 has better precision than Q4)
    for (0..32) |i| {
        try testing.expectApproxEqAbs(src[i], dst[i], 0.02);
    }
}

test "calc_scale_symmetric" {
    const values = [_]f32{ -3.0, 1.0, 2.5, -1.5 };
    const scale = calc_scale_symmetric(&values, 4);
    try testing.expectEqual(@as(f32, 3.0), scale);
}

test "calc_scale_asymmetric" {
    const values = [_]f32{ 1.0, 5.0, 3.0, 2.0 };
    const result = calc_scale_asymmetric(&values, 4);
    try testing.expectEqual(@as(f32, 4.0), result.scale);
    try testing.expectEqual(@as(f32, 1.0), result.zero_point);
}
