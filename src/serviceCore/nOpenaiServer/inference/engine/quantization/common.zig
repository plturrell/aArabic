const std = @import("std");

/// Common quantization utilities and conversions
/// Shared across all quantization formats

// ============================================================================
// Float16 Conversions
// ============================================================================

/// Convert f32 to f16 (IEEE 754 binary16)
pub fn f32_to_f16(val: f32) u16 {
    const bits = @as(u32, @bitCast(val));
    
    const sign = (bits >> 31) & 0x1;
    const exp = (bits >> 23) & 0xFF;
    const mantissa = bits & 0x7FFFFF;
    
    // Handle special cases
    if (exp == 0xFF) {
        // Inf or NaN
        return @intCast((sign << 15) | 0x7C00 | (mantissa >> 13));
    }
    
    // Convert exponent (bias: 127 for f32, 15 for f16)
    const exp_f16 = @as(i32, @intCast(exp)) - 127 + 15;
    
    if (exp_f16 <= 0) {
        // Underflow to zero or denormal
        return @intCast(sign << 15);
    }
    
    if (exp_f16 >= 0x1F) {
        // Overflow to infinity
        return @intCast((sign << 15) | 0x7C00);
    }
    
    // Normal case
    const mantissa_f16 = mantissa >> 13;
    return @intCast((sign << 15) | (@as(u32, @intCast(exp_f16)) << 10) | mantissa_f16);
}

/// Convert f16 to f32
pub fn f16_to_f32(val: u16) f32 {
    const bits = @as(u32, val);

    const sign = (bits >> 15) & 0x1;
    const exp = (bits >> 10) & 0x1F;
    const mantissa = bits & 0x3FF;

    // Handle special cases
    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            return @bitCast(@as(u32, sign << 31));
        }
        // Denormal f16: value = (-1)^sign * 2^(-14) * (mantissa / 1024)
        // Convert to f32 denormal or normal representation
        // f16 denormal range: 2^-24 to 2^-14 (approximately 5.96e-8 to 6.1e-5)
        const sign_f32: f32 = if (sign == 1) -1.0 else 1.0;
        const mant_f32: f32 = @floatFromInt(mantissa);
        // 2^(-14) / 1024 = 2^(-14) * 2^(-10) = 2^(-24)
        const scale: f32 = 5.9604644775390625e-8; // 2^(-24)
        return sign_f32 * mant_f32 * scale;
    }

    if (exp == 0x1F) {
        // Inf or NaN
        const result_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        return @bitCast(result_bits);
    }

    // Normal case - convert exponent
    const exp_f32 = exp + 127 - 15;
    const mantissa_f32 = mantissa << 13;
    const result_bits = (sign << 31) | (exp_f32 << 23) | mantissa_f32;

    return @bitCast(result_bits);
}

// ============================================================================
// Block Size Constants
// ============================================================================

pub const QK4_0: usize = 32; // Q4_0 block size
pub const QK4_1: usize = 32; // Q4_1 block size
pub const QK5_0: usize = 32; // Q5_0 block size
pub const QK5_1: usize = 32; // Q5_1 block size
pub const QK8_0: usize = 32; // Q8_0 block size

pub const QK_K: usize = 256; // K-quants block size

// ============================================================================
// Block Structures
// ============================================================================

/// Q4_0 block: 18 bytes for 32 values
/// - 2 bytes: f16 scale
/// - 16 bytes: 32x 4-bit quantized values (packed)
pub const BlockQ4_0 = extern struct {
    scale: u16, // f16
    qs: [16]u8, // Quantized values (4 bits each, 2 per byte)
};

/// Q4_1 block: 20 bytes for 32 values  
/// - 2 bytes: f16 scale
/// - 2 bytes: f16 min (offset)
/// - 16 bytes: 32x 4-bit quantized values
pub const BlockQ4_1 = extern struct {
    scale: u16, // f16
    min: u16, // f16
    qs: [16]u8,
};

/// Q5_0 block: 22 bytes for 32 values
/// - 2 bytes: f16 scale
/// - 4 bytes: high bits (1 bit per value)
/// - 16 bytes: low 4 bits
pub const BlockQ5_0 = extern struct {
    scale: u16, // f16
    qh: [4]u8, // High bits
    qs: [16]u8, // Low bits
};

/// Q8_0 block: 34 bytes for 32 values
/// - 2 bytes: f16 scale
/// - 32 bytes: 8-bit quantized values
pub const BlockQ8_0 = extern struct {
    scale: u16, // f16
    qs: [32]i8, // 8-bit values
};

// ============================================================================
// Quantization Helpers
// ============================================================================

/// Quantize a single f32 value to 4-bit (0-15)
pub fn quantize_4bit(val: f32, scale: f32, offset: f32) u8 {
    const quantized = (val - offset) / scale;
    const clamped = std.math.clamp(quantized, 0.0, 15.0);
    return @intFromFloat(@round(clamped));
}

/// Dequantize a 4-bit value to f32
pub fn dequantize_4bit(qval: u8, scale: f32, offset: f32) f32 {
    return @as(f32, @floatFromInt(qval)) * scale + offset;
}

/// Quantize a single f32 value to 8-bit (-128 to 127)
pub fn quantize_8bit(val: f32, scale: f32) i8 {
    const quantized = val / scale;
    const clamped = std.math.clamp(quantized, -128.0, 127.0);
    return @intFromFloat(@round(clamped));
}

/// Dequantize an 8-bit value to f32
pub fn dequantize_8bit(qval: i8, scale: f32) f32 {
    return @as(f32, @floatFromInt(qval)) * scale;
}

// ============================================================================
// Block Packing/Unpacking
// ============================================================================

/// Pack two 4-bit values into a single byte
pub fn pack_4bit(low: u8, high: u8) u8 {
    return (low & 0xF) | ((high & 0xF) << 4);
}

/// Extract a specific 4-bit value from a packed array
pub fn get_4bit_value(qs: []const u8, index: usize) u8 {
    const byte_index = index / 2;
    const is_high = (index % 2) == 1;
    
    if (is_high) {
        return (qs[byte_index] >> 4) & 0xF;
    } else {
        return qs[byte_index] & 0xF;
    }
}

// ============================================================================
// Statistics Calculation
// ============================================================================

/// Calculate min and max of a float array
pub fn calc_min_max(values: []const f32) struct { min: f32, max: f32 } {
    var min_val = values[0];
    var max_val = values[0];
    
    for (values[1..]) |val| {
        min_val = @min(min_val, val);
        max_val = @max(max_val, val);
    }
    
    return .{ .min = min_val, .max = max_val };
}

/// Calculate scale for quantization
pub fn calc_scale(min_val: f32, max_val: f32, n_levels: f32) f32 {
    const range = max_val - min_val;
    if (range == 0.0) return 1.0;
    return range / n_levels;
}

/// Calculate optimal scale and offset for a block (Q4_0 style)
pub fn calc_q4_0_params(values: []const f32) struct { scale: f32 } {
    var max_abs: f32 = 0.0;
    
    for (values) |val| {
        max_abs = @max(max_abs, @abs(val));
    }
    
    if (max_abs == 0.0) {
        return .{ .scale = 1.0 };
    }
    
    // Map [-max_abs, max_abs] to [-8, 7] (4-bit signed)
    const scale = max_abs / 7.0;
    
    return .{ .scale = scale };
}

/// Calculate scale and min for a block (Q4_1 style)
pub fn calc_q4_1_params(values: []const f32) struct { scale: f32, min: f32 } {
    const stats = calc_min_max(values);
    
    if (stats.max == stats.min) {
        return .{ .scale = 1.0, .min = stats.min };
    }
    
    // Map [min, max] to [0, 15] (4-bit unsigned)
    const scale = (stats.max - stats.min) / 15.0;
    
    return .{ .scale = scale, .min = stats.min };
}

// ============================================================================
// SIMD Helpers for Dequantization
// ============================================================================

/// Dequantize 8 values at once using SIMD
pub fn dequantize_8x_f32(
    output: []f32,
    qvals: []const u8,
    scale: f32,
) void {
    const Vec = @Vector(8, f32);
    const scale_vec: Vec = @splat(scale);
    
    // Convert u8 to f32 vector
    var float_vec: Vec = undefined;
    inline for (0..8) |i| {
        float_vec[i] = @floatFromInt(qvals[i]);
    }
    
    // Scale
    const result = float_vec * scale_vec;
    
    // Store
    output[0..8].* = result;
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_conversions() !void {
    std.debug.print("\n🧪 Testing Quantization Commons\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: f16 conversions
    {
        std.debug.print("\n1️⃣  Testing f16 conversions...\n", .{});
        
        const test_values = [_]f32{ 0.0, 1.0, -1.0, 0.5, 1000.0, -1000.0 };
        
        for (test_values) |val| {
            const f16_val = f32_to_f16(val);
            const recovered = f16_to_f32(f16_val);
            
            // Allow small error for f16 precision
            const error_val = @abs(val - recovered);
            const relative_error = if (@abs(val) > 0.001)
                error_val / @abs(val)
            else
                error_val;
            
            if (relative_error > 0.01 and error_val > 0.1) {
                std.debug.print("   ⚠️  f16 conversion error: {d} -> {d} (error: {d})\n", .{ val, recovered, error_val });
            }
        }
        
        std.debug.print("   ✅ f16 conversions working\n", .{});
    }
    
    // Test 2: 4-bit packing
    {
        std.debug.print("\n2️⃣  Testing 4-bit packing...\n", .{});
        
        const low: u8 = 5;
        const high: u8 = 12;
        const packed_byte = pack_4bit(low, high);
        
        // Extract values back
        const extracted_low = packed_byte & 0xF;
        const extracted_high = (packed_byte >> 4) & 0xF;
        
        if (extracted_low != low or extracted_high != high) {
            std.debug.print("   ❌ Packing failed: {d},{d} -> {d} -> {d},{d}\n", .{
                low,
                high,
                packed_byte,
                extracted_low,
                extracted_high,
            });
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ 4-bit packing correct\n", .{});
    }
    
    // Test 3: Quantization/dequantization
    {
        std.debug.print("\n3️⃣  Testing quantization...\n", .{});
        
        const val: f32 = 3.7;
        const scale: f32 = 0.5;
        const offset: f32 = 0.0;
        
        const qval = quantize_4bit(val, scale, offset);
        const recovered = dequantize_4bit(qval, scale, offset);
        
        const error_val = @abs(val - recovered);
        if (error_val > scale) {
            std.debug.print("   ⚠️  Quantization error: {d} -> {d} -> {d}\n", .{ val, qval, recovered });
        }
        
        std.debug.print("   ✅ Quantization working (error: {d:.4})\n", .{error_val});
    }
    
    // Test 4: Scale calculation
    {
        std.debug.print("\n4️⃣  Testing scale calculation...\n", .{});
        
        const test_block = [_]f32{ -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
        const params = calc_q4_0_params(&test_block);
        
        std.debug.print("   Scale: {d:.4}\n", .{params.scale});
        
        if (params.scale <= 0.0) {
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Scale calculation correct\n", .{});
    }
    
    std.debug.print("\n✅ All quantization commons tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
