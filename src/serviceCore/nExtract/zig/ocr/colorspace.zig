// nExtract - Day 26: Color Space Conversions
// Pure Zig implementation for image color space transformations
// Part of Phase 2, Week 6: Image Processing Primitives

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

// ============================================================================
// Color Space Structures
// ============================================================================

/// RGB color (0-255 per channel)
pub const RGB = struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn init(r: u8, g: u8, b: u8) RGB {
        return .{ .r = r, .g = g, .b = b };
    }

    pub fn toGrayscale(self: RGB) u8 {
        return rgbToGrayscale(self.r, self.g, self.b);
    }

    pub fn toHSV(self: RGB) HSV {
        return rgbToHSV(self.r, self.g, self.b);
    }

    pub fn toHSL(self: RGB) HSL {
        return rgbToHSL(self.r, self.g, self.b);
    }

    pub fn toYCbCr(self: RGB) YCbCr {
        return rgbToYCbCr(self.r, self.g, self.b);
    }
};

/// RGBA color (with alpha channel)
pub const RGBA = struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,

    pub fn init(r: u8, g: u8, b: u8, a: u8) RGBA {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    pub fn toRGB(self: RGBA) RGB {
        return RGB.init(self.r, self.g, self.b);
    }
};

/// HSV color (Hue, Saturation, Value)
pub const HSV = struct {
    h: f32, // 0.0 - 360.0
    s: f32, // 0.0 - 1.0
    v: f32, // 0.0 - 1.0

    pub fn init(h: f32, s: f32, v: f32) HSV {
        return .{ .h = h, .s = s, .v = v };
    }

    pub fn toRGB(self: HSV) RGB {
        return hsvToRGB(self.h, self.s, self.v);
    }
};

/// HSL color (Hue, Saturation, Lightness)
pub const HSL = struct {
    h: f32, // 0.0 - 360.0
    s: f32, // 0.0 - 1.0
    l: f32, // 0.0 - 1.0

    pub fn init(h: f32, s: f32, l: f32) HSL {
        return .{ .h = h, .s = s, .l = l };
    }

    pub fn toRGB(self: HSL) RGB {
        return hslToRGB(self.h, self.s, self.l);
    }
};

/// YCbCr color (JPEG color space)
pub const YCbCr = struct {
    y: u8,  // Luminance
    cb: u8, // Blue chrominance
    cr: u8, // Red chrominance

    pub fn init(y: u8, cb: u8, cr: u8) YCbCr {
        return .{ .y = y, .cb = cb, .cr = cr };
    }

    pub fn toRGB(self: YCbCr) RGB {
        return ycbcrToRGB(self.y, self.cb, self.cr);
    }
};

/// CMYK color (Print color space)
pub const CMYK = struct {
    c: u8, // Cyan
    m: u8, // Magenta
    y: u8, // Yellow
    k: u8, // Black

    pub fn init(c: u8, m: u8, y: u8, k: u8) CMYK {
        return .{ .c = c, .m = m, .y = y, .k = k };
    }

    pub fn toRGB(self: CMYK) RGB {
        return cmykToRGB(self.c, self.m, self.y, self.k);
    }
};

// ============================================================================
// RGB to Grayscale
// ============================================================================

/// Convert RGB to grayscale using weighted average
/// Formula: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
pub fn rgbToGrayscale(r: u8, g: u8, b: u8) u8 {
    const rf = @as(f32, @floatFromInt(r));
    const gf = @as(f32, @floatFromInt(g));
    const bf = @as(f32, @floatFromInt(b));
    
    const gray = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    return @intFromFloat(@min(255.0, @max(0.0, gray)));
}

/// Convert grayscale image buffer to RGB (duplicate value to all channels)
pub fn grayscaleToRGB(gray: u8) RGB {
    return RGB.init(gray, gray, gray);
}

/// Batch convert RGB buffer to grayscale
pub fn batchRgbToGrayscale(rgb_data: []const u8, gray_data: []u8) !void {
    if (rgb_data.len % 3 != 0) return error.InvalidRGBBufferSize;
    if (gray_data.len != rgb_data.len / 3) return error.InvalidGrayscaleBufferSize;
    
    var i: usize = 0;
    var j: usize = 0;
    while (i < rgb_data.len) : (i += 3) {
        gray_data[j] = rgbToGrayscale(rgb_data[i], rgb_data[i + 1], rgb_data[i + 2]);
        j += 1;
    }
}

// ============================================================================
// RGB <-> HSV
// ============================================================================

/// Convert RGB to HSV
pub fn rgbToHSV(r: u8, g: u8, b: u8) HSV {
    const rf = @as(f32, @floatFromInt(r)) / 255.0;
    const gf = @as(f32, @floatFromInt(g)) / 255.0;
    const bf = @as(f32, @floatFromInt(b)) / 255.0;
    
    const max_val = @max(@max(rf, gf), bf);
    const min_val = @min(@min(rf, gf), bf);
    const delta = max_val - min_val;
    
    // Calculate Hue
    var h: f32 = 0.0;
    if (delta > 0.0) {
        if (max_val == rf) {
            h = 60.0 * (@mod((gf - bf) / delta, 6.0));
        } else if (max_val == gf) {
            h = 60.0 * (((bf - rf) / delta) + 2.0);
        } else {
            h = 60.0 * (((rf - gf) / delta) + 4.0);
        }
    }
    if (h < 0.0) h += 360.0;
    
    // Calculate Saturation
    const s: f32 = if (max_val > 0.0) delta / max_val else 0.0;
    
    // Value is just the max
    const v: f32 = max_val;
    
    return HSV.init(h, s, v);
}

/// Convert HSV to RGB
pub fn hsvToRGB(h: f32, s: f32, v: f32) RGB {
    const c = v * s;
    const x = c * (1.0 - @abs(@mod(h / 60.0, 2.0) - 1.0));
    const m = v - c;
    
    var rf: f32 = 0.0;
    var gf: f32 = 0.0;
    var bf: f32 = 0.0;
    
    const h_sector = @as(i32, @intFromFloat(h / 60.0));
    switch (h_sector) {
        0 => { rf = c; gf = x; bf = 0.0; },
        1 => { rf = x; gf = c; bf = 0.0; },
        2 => { rf = 0.0; gf = c; bf = x; },
        3 => { rf = 0.0; gf = x; bf = c; },
        4 => { rf = x; gf = 0.0; bf = c; },
        else => { rf = c; gf = 0.0; bf = x; },
    }
    
    const r = @as(u8, @intFromFloat((rf + m) * 255.0));
    const g = @as(u8, @intFromFloat((gf + m) * 255.0));
    const b = @as(u8, @intFromFloat((bf + m) * 255.0));
    
    return RGB.init(r, g, b);
}

// ============================================================================
// RGB <-> HSL
// ============================================================================

/// Convert RGB to HSL
pub fn rgbToHSL(r: u8, g: u8, b: u8) HSL {
    const rf = @as(f32, @floatFromInt(r)) / 255.0;
    const gf = @as(f32, @floatFromInt(g)) / 255.0;
    const bf = @as(f32, @floatFromInt(b)) / 255.0;
    
    const max_val = @max(@max(rf, gf), bf);
    const min_val = @min(@min(rf, gf), bf);
    const delta = max_val - min_val;
    
    // Calculate Lightness
    const l = (max_val + min_val) / 2.0;
    
    // Calculate Saturation
    var s: f32 = 0.0;
    if (delta > 0.0) {
        s = if (l < 0.5)
            delta / (max_val + min_val)
        else
            delta / (2.0 - max_val - min_val);
    }
    
    // Calculate Hue (same as HSV)
    var h: f32 = 0.0;
    if (delta > 0.0) {
        if (max_val == rf) {
            h = 60.0 * (@mod((gf - bf) / delta, 6.0));
        } else if (max_val == gf) {
            h = 60.0 * (((bf - rf) / delta) + 2.0);
        } else {
            h = 60.0 * (((rf - gf) / delta) + 4.0);
        }
    }
    if (h < 0.0) h += 360.0;
    
    return HSL.init(h, s, l);
}

/// Helper function for HSL to RGB conversion
fn hueToRGB(p: f32, q: f32, t_input: f32) f32 {
    var t = t_input;
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    
    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0 / 2.0) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    return p;
}

/// Convert HSL to RGB
pub fn hslToRGB(h: f32, s: f32, l: f32) RGB {
    var rf: f32 = undefined;
    var gf: f32 = undefined;
    var bf: f32 = undefined;
    
    if (s == 0.0) {
        // Achromatic (gray)
        rf = l;
        gf = l;
        bf = l;
    } else {
        const q = if (l < 0.5) l * (1.0 + s) else l + s - l * s;
        const p = 2.0 * l - q;
        const h_normalized = h / 360.0;
        
        rf = hueToRGB(p, q, h_normalized + 1.0 / 3.0);
        gf = hueToRGB(p, q, h_normalized);
        bf = hueToRGB(p, q, h_normalized - 1.0 / 3.0);
    }
    
    const r = @as(u8, @intFromFloat(rf * 255.0));
    const g = @as(u8, @intFromFloat(gf * 255.0));
    const b = @as(u8, @intFromFloat(bf * 255.0));
    
    return RGB.init(r, g, b);
}

// ============================================================================
// RGB <-> YCbCr (JPEG Color Space)
// ============================================================================

/// Convert RGB to YCbCr (ITU-R BT.601)
pub fn rgbToYCbCr(r: u8, g: u8, b: u8) YCbCr {
    const rf = @as(f32, @floatFromInt(r));
    const gf = @as(f32, @floatFromInt(g));
    const bf = @as(f32, @floatFromInt(b));
    
    const y_val = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    const cb_val = 128.0 + (-0.168736 * rf - 0.331264 * gf + 0.5 * bf);
    const cr_val = 128.0 + (0.5 * rf - 0.418688 * gf - 0.081312 * bf);
    
    const y = @as(u8, @intFromFloat(@min(255.0, @max(0.0, y_val))));
    const cb = @as(u8, @intFromFloat(@min(255.0, @max(0.0, cb_val))));
    const cr = @as(u8, @intFromFloat(@min(255.0, @max(0.0, cr_val))));
    
    return YCbCr.init(y, cb, cr);
}

/// Convert YCbCr to RGB (ITU-R BT.601)
pub fn ycbcrToRGB(y: u8, cb: u8, cr: u8) RGB {
    const yf = @as(f32, @floatFromInt(y));
    const cbf = @as(f32, @floatFromInt(cb)) - 128.0;
    const crf = @as(f32, @floatFromInt(cr)) - 128.0;
    
    const r_val = yf + 1.402 * crf;
    const g_val = yf - 0.344136 * cbf - 0.714136 * crf;
    const b_val = yf + 1.772 * cbf;
    
    const r = @as(u8, @intFromFloat(@min(255.0, @max(0.0, r_val))));
    const g = @as(u8, @intFromFloat(@min(255.0, @max(0.0, g_val))));
    const b = @as(u8, @intFromFloat(@min(255.0, @max(0.0, b_val))));
    
    return RGB.init(r, g, b);
}

// ============================================================================
// CMYK <-> RGB
// ============================================================================

/// Convert RGB to CMYK
pub fn rgbToCMYK(r: u8, g: u8, b: u8) CMYK {
    const rf = @as(f32, @floatFromInt(r)) / 255.0;
    const gf = @as(f32, @floatFromInt(g)) / 255.0;
    const bf = @as(f32, @floatFromInt(b)) / 255.0;
    
    const k = 1.0 - @max(@max(rf, gf), bf);
    
    if (k >= 1.0) {
        // Pure black
        return CMYK.init(0, 0, 0, 255);
    }
    
    const c = (1.0 - rf - k) / (1.0 - k);
    const m = (1.0 - gf - k) / (1.0 - k);
    const y = (1.0 - bf - k) / (1.0 - k);
    
    return CMYK.init(
        @intFromFloat(c * 255.0),
        @intFromFloat(m * 255.0),
        @intFromFloat(y * 255.0),
        @intFromFloat(k * 255.0),
    );
}

/// Convert CMYK to RGB
pub fn cmykToRGB(c: u8, m: u8, y: u8, k: u8) RGB {
    const cf = @as(f32, @floatFromInt(c)) / 255.0;
    const mf = @as(f32, @floatFromInt(m)) / 255.0;
    const yf = @as(f32, @floatFromInt(y)) / 255.0;
    const kf = @as(f32, @floatFromInt(k)) / 255.0;
    
    const r_val = 255.0 * (1.0 - cf) * (1.0 - kf);
    const g_val = 255.0 * (1.0 - mf) * (1.0 - kf);
    const b_val = 255.0 * (1.0 - yf) * (1.0 - kf);
    
    const r = @as(u8, @intFromFloat(@min(255.0, @max(0.0, r_val))));
    const g = @as(u8, @intFromFloat(@min(255.0, @max(0.0, g_val))));
    const b = @as(u8, @intFromFloat(@min(255.0, @max(0.0, b_val))));
    
    return RGB.init(r, g, b);
}

// ============================================================================
// Gamma Correction
// ============================================================================

/// Apply gamma correction (encode)
pub fn gammaEncode(linear: f32, gamma: f32) f32 {
    return math.pow(f32, linear, 1.0 / gamma);
}

/// Remove gamma correction (decode)
pub fn gammaDecode(encoded: f32, gamma: f32) f32 {
    return math.pow(f32, encoded, gamma);
}

/// Apply sRGB gamma correction to RGB values
pub fn srgbGammaEncode(r: u8, g: u8, b: u8) RGB {
    const rf = @as(f32, @floatFromInt(r)) / 255.0;
    const gf = @as(f32, @floatFromInt(g)) / 255.0;
    const bf = @as(f32, @floatFromInt(b)) / 255.0;
    
    const r_encoded = if (rf <= 0.0031308)
        rf * 12.92
    else
        1.055 * math.pow(f32, rf, 1.0 / 2.4) - 0.055;
    
    const g_encoded = if (gf <= 0.0031308)
        gf * 12.92
    else
        1.055 * math.pow(f32, gf, 1.0 / 2.4) - 0.055;
    
    const b_encoded = if (bf <= 0.0031308)
        bf * 12.92
    else
        1.055 * math.pow(f32, bf, 1.0 / 2.4) - 0.055;
    
    return RGB.init(
        @intFromFloat(r_encoded * 255.0),
        @intFromFloat(g_encoded * 255.0),
        @intFromFloat(b_encoded * 255.0),
    );
}

/// Remove sRGB gamma correction from RGB values
pub fn srgbGammaDecode(r: u8, g: u8, b: u8) RGB {
    const rf = @as(f32, @floatFromInt(r)) / 255.0;
    const gf = @as(f32, @floatFromInt(g)) / 255.0;
    const bf = @as(f32, @floatFromInt(b)) / 255.0;
    
    const r_linear = if (rf <= 0.04045)
        rf / 12.92
    else
        math.pow(f32, (rf + 0.055) / 1.055, 2.4);
    
    const g_linear = if (gf <= 0.04045)
        gf / 12.92
    else
        math.pow(f32, (gf + 0.055) / 1.055, 2.4);
    
    const b_linear = if (bf <= 0.04045)
        bf / 12.92
    else
        math.pow(f32, (bf + 0.055) / 1.055, 2.4);
    
    return RGB.init(
        @intFromFloat(r_linear * 255.0),
        @intFromFloat(g_linear * 255.0),
        @intFromFloat(b_linear * 255.0),
    );
}

// ============================================================================
// Color Temperature Adjustment
// ============================================================================

/// Adjust color temperature (warm/cool)
/// temperature: -1.0 (cool) to 1.0 (warm)
pub fn adjustColorTemperature(r: u8, g: u8, b: u8, temperature: f32) RGB {
    const temp = @max(-1.0, @min(1.0, temperature));
    
    var rf = @as(f32, @floatFromInt(r));
    var gf = @as(f32, @floatFromInt(g));
    var bf = @as(f32, @floatFromInt(b));
    
    if (temp > 0.0) {
        // Warm (add red, reduce blue)
        rf += temp * (255.0 - rf) * 0.5;
        bf -= temp * bf * 0.3;
    } else if (temp < 0.0) {
        // Cool (add blue, reduce red)
        bf += (-temp) * (255.0 - bf) * 0.5;
        rf -= (-temp) * rf * 0.3;
    }
    
    return RGB.init(
        @intFromFloat(@min(255.0, @max(0.0, rf))),
        @intFromFloat(@min(255.0, @max(0.0, gf))),
        @intFromFloat(@min(255.0, @max(0.0, bf))),
    );
}

// ============================================================================
// Batch Conversion Functions
// ============================================================================

/// Batch convert RGB to HSV
pub fn batchRgbToHSV(rgb_data: []const u8, hsv_data: []HSV) !void {
    if (rgb_data.len % 3 != 0) return error.InvalidRGBBufferSize;
    if (hsv_data.len != rgb_data.len / 3) return error.InvalidHSVBufferSize;
    
    var i: usize = 0;
    var j: usize = 0;
    while (i < rgb_data.len) : (i += 3) {
        hsv_data[j] = rgbToHSV(rgb_data[i], rgb_data[i + 1], rgb_data[i + 2]);
        j += 1;
    }
}

/// Batch convert YCbCr to RGB (common in JPEG decoding)
pub fn batchYCbCrToRGB(ycbcr_data: []const u8, rgb_data: []u8) !void {
    if (ycbcr_data.len % 3 != 0) return error.InvalidYCbCrBufferSize;
    if (rgb_data.len != ycbcr_data.len) return error.InvalidRGBBufferSize;
    
    var i: usize = 0;
    while (i < ycbcr_data.len) : (i += 3) {
        const rgb = ycbcrToRGB(ycbcr_data[i], ycbcr_data[i + 1], ycbcr_data[i + 2]);
        rgb_data[i] = rgb.r;
        rgb_data[i + 1] = rgb.g;
        rgb_data[i + 2] = rgb.b;
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "RGB to Grayscale" {
    // White
    try testing.expectEqual(@as(u8, 255), rgbToGrayscale(255, 255, 255));
    
    // Black
    try testing.expectEqual(@as(u8, 0), rgbToGrayscale(0, 0, 0));
    
    // Red (should be ~76 based on 0.299 coefficient)
    const red_gray = rgbToGrayscale(255, 0, 0);
    try testing.expect(red_gray >= 75 and red_gray <= 77);
    
    // Green (should be ~150 based on 0.587 coefficient)
    const green_gray = rgbToGrayscale(0, 255, 0);
    try testing.expect(green_gray >= 149 and green_gray <= 151);
    
    // Blue (should be ~29 based on 0.114 coefficient)
    const blue_gray = rgbToGrayscale(0, 0, 255);
    try testing.expect(blue_gray >= 28 and blue_gray <= 30);
}

test "RGB <-> HSV round-trip" {
    const colors = [_]RGB{
        RGB.init(255, 0, 0),     // Red
        RGB.init(0, 255, 0),     // Green
        RGB.init(0, 0, 255),     // Blue
        RGB.init(255, 255, 0),   // Yellow
        RGB.init(128, 128, 128), // Gray
    };
    
    for (colors) |rgb| {
        const hsv = rgb.toHSV();
        const rgb_back = hsv.toRGB();
        
        // Allow small error due to floating point
        try testing.expect(@abs(@as(i16, rgb.r) - @as(i16, rgb_back.r)) <= 1);
        try testing.expect(@abs(@as(i16, rgb.g) - @as(i16, rgb_back.g)) <= 1);
        try testing.expect(@abs(@as(i16, rgb.b) - @as(i16, rgb_back.b)) <= 1);
    }
}

test "RGB <-> YCbCr round-trip" {
    const colors = [_]RGB{
        RGB.init(255, 255, 255), // White
        RGB.init(0, 0, 0),       // Black
        RGB.init(255, 0, 0),     // Red
        RGB.init(0, 255, 0),     // Green
        RGB.init(0, 0, 255),     // Blue
    };
    
    for (colors) |rgb| {
        const ycbcr = rgb.toYCbCr();
        const rgb_back = ycbcr.toRGB();
        
        // Allow small error due to color space conversion
        try testing.expect(@abs(@as(i16, rgb.r) - @as(i16, rgb_back.r)) <= 2);
        try testing.expect(@abs(@as(i16, rgb.g) - @as(i16, rgb_back.g)) <= 2);
        try testing.expect(@abs(@as(i16, rgb.b) - @as(i16, rgb_back.b)) <= 2);
    }
}

test "RGB <-> CMYK round-trip" {
    const colors = [_]RGB{
        RGB.init(255, 0, 0),   // Red
        RGB.init(0, 255, 0),   // Green
        RGB.init(0, 0, 255),   // Blue
        RGB.init(255, 255, 0), // Yellow
    };
    
    for (colors) |rgb| {
        const cmyk = rgbToCMYK(rgb.r, rgb.g, rgb.b);
        const rgb_back = cmyk.toRGB();
        
        // Allow small error
        try testing.expect(@abs(@as(i16, rgb.r) - @as(i16, rgb_back.r)) <= 2);
        try testing.expect(@abs(@as(i16, rgb.g) - @as(i16, rgb_back.g)) <= 2);
        try testing.expect(@abs(@as(i16, rgb.b) - @as(i16, rgb_back.b)) <= 2);
    }
}
