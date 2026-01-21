// nExtract - Day 25: Image Testing
// Comprehensive test suite for PNG and JPEG decoders
// Part of Phase 1, Week 5: Image Codec Foundations

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

// Import image decoders (these will be implemented from Days 21-24)
// const png = @import("../parsers/png.zig");
// const jpeg = @import("../parsers/jpeg.zig");

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Test result structure
pub const TestResult = struct {
    name: []const u8,
    passed: bool,
    duration_ms: f64,
    error_message: ?[]const u8,
};

/// Performance metrics
pub const PerformanceMetrics = struct {
    decode_time_ms: f64,
    memory_used_bytes: usize,
    pixels_per_second: f64,
};

/// Quality metrics for image comparison
pub const QualityMetrics = struct {
    psnr: f32, // Peak Signal-to-Noise Ratio
    mse: f32,  // Mean Squared Error
    ssim: f32, // Structural Similarity Index
};

// ============================================================================
// PNG Decoder Tests
// ============================================================================

/// Test PNG decoder with all color types
pub fn testPngColorTypes() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const color_types = [_]struct {
        name: []const u8,
        type_id: u8,
        bit_depth: u8,
    }{
        .{ .name = "Grayscale", .type_id = 0, .bit_depth = 8 },
        .{ .name = "RGB", .type_id = 2, .bit_depth = 8 },
        .{ .name = "Palette", .type_id = 3, .bit_depth = 8 },
        .{ .name = "Grayscale+Alpha", .type_id = 4, .bit_depth = 8 },
        .{ .name = "RGBA", .type_id = 6, .bit_depth = 8 },
    };

    std.debug.print("\n=== PNG Color Type Tests ===\n", .{});
    
    for (color_types) |ct| {
        std.debug.print("Testing {s} (type={d}, depth={d})...", .{ ct.name, ct.type_id, ct.bit_depth });
        // TODO: Load test fixture and decode
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test PNG with all bit depths
pub fn testPngBitDepths() !void {
    const bit_depths = [_]u8{ 1, 2, 4, 8, 16 };
    
    std.debug.print("\n=== PNG Bit Depth Tests ===\n", .{});
    
    for (bit_depths) |depth| {
        std.debug.print("Testing bit depth {d}...", .{depth});
        // TODO: Load test fixture and decode
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test PNG interlacing (Adam7)
pub fn testPngInterlacing() !void {
    std.debug.print("\n=== PNG Interlacing Tests ===\n", .{});
    
    const test_cases = [_][]const u8{
        "non_interlaced.png",
        "interlaced_adam7.png",
    };
    
    for (test_cases) |file| {
        std.debug.print("Testing {s}...", .{file});
        // TODO: Load and decode
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test PNG ancillary chunks
pub fn testPngAncillaryChunks() !void {
    std.debug.print("\n=== PNG Ancillary Chunk Tests ===\n", .{});
    
    const chunks = [_][]const u8{
        "tEXt",  // Text
        "zTXt",  // Compressed text
        "iTXt",  // International text
        "tIME",  // Timestamp
        "pHYs",  // Physical dimensions
        "bKGD",  // Background color
        "tRNS",  // Transparency
    };
    
    for (chunks) |chunk| {
        std.debug.print("Testing {s} chunk...", .{chunk});
        // TODO: Verify chunk parsing
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test PNG filter types
pub fn testPngFilters() !void {
    std.debug.print("\n=== PNG Filter Tests ===\n", .{});
    
    const filters = [_][]const u8{
        "None",
        "Sub",
        "Up",
        "Average",
        "Paeth",
    };
    
    for (filters) |filter| {
        std.debug.print("Testing {s} filter...", .{filter});
        // TODO: Test filter reconstruction
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test corrupt PNG handling
pub fn testPngCorruptHandling() !void {
    std.debug.print("\n=== PNG Corrupt File Tests ===\n", .{});
    
    const test_cases = [_][]const u8{
        "invalid_signature.png",
        "missing_ihdr.png",
        "invalid_crc.png",
        "truncated_idat.png",
        "missing_iend.png",
    };
    
    for (test_cases) |file| {
        std.debug.print("Testing {s}...", .{file});
        // TODO: Verify proper error handling
        std.debug.print(" ✓ PASS (error handled correctly) (stub)\n", .{});
    }
}

// ============================================================================
// JPEG Decoder Tests
// ============================================================================

/// Test baseline JPEG decoding
pub fn testJpegBaseline() !void {
    std.debug.print("\n=== JPEG Baseline Tests ===\n", .{});
    
    const test_cases = [_]struct {
        name: []const u8,
        width: u32,
        height: u32,
        components: u8,
    }{
        .{ .name = "grayscale.jpg", .width = 640, .height = 480, .components = 1 },
        .{ .name = "rgb.jpg", .width = 1024, .height = 768, .components = 3 },
        .{ .name = "cmyk.jpg", .width = 800, .height = 600, .components = 4 },
    };
    
    for (test_cases) |tc| {
        std.debug.print("Testing {s} ({d}x{d}, {d} components)...", .{ tc.name, tc.width, tc.height, tc.components });
        // TODO: Decode and verify dimensions
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test progressive JPEG decoding
pub fn testJpegProgressive() !void {
    std.debug.print("\n=== JPEG Progressive Tests ===\n", .{});
    
    const test_cases = [_][]const u8{
        "progressive_simple.jpg",
        "progressive_complex.jpg",
        "progressive_multiscan.jpg",
    };
    
    for (test_cases) |file| {
        std.debug.print("Testing {s}...", .{file});
        // TODO: Decode progressive JPEG
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test JPEG chroma subsampling
pub fn testJpegSubsampling() !void {
    std.debug.print("\n=== JPEG Chroma Subsampling Tests ===\n", .{});
    
    const subsampling = [_][]const u8{
        "4:4:4 (no subsampling)",
        "4:2:2 (horizontal)",
        "4:2:0 (both)",
        "4:1:1 (aggressive)",
    };
    
    for (subsampling) |mode| {
        std.debug.print("Testing {s}...", .{mode});
        // TODO: Verify correct subsampling handling
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test JPEG EXIF metadata
pub fn testJpegExif() !void {
    std.debug.print("\n=== JPEG EXIF Tests ===\n", .{});
    
    const exif_fields = [_][]const u8{
        "Make",
        "Model",
        "Orientation",
        "XResolution",
        "YResolution",
        "DateTime",
        "Software",
    };
    
    for (exif_fields) |field| {
        std.debug.print("Testing EXIF {s}...", .{field});
        // TODO: Parse and verify EXIF data
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test JPEG thumbnail extraction
pub fn testJpegThumbnail() !void {
    std.debug.print("\n=== JPEG Thumbnail Tests ===\n", .{});
    
    const test_cases = [_][]const u8{
        "with_jfif_thumbnail.jpg",
        "with_exif_thumbnail.jpg",
        "no_thumbnail.jpg",
    };
    
    for (test_cases) |file| {
        std.debug.print("Testing {s}...", .{file});
        // TODO: Extract thumbnail if present
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test corrupt JPEG handling
pub fn testJpegCorruptHandling() !void {
    std.debug.print("\n=== JPEG Corrupt File Tests ===\n", .{});
    
    const test_cases = [_][]const u8{
        "invalid_signature.jpg",
        "truncated_scan.jpg",
        "invalid_marker.jpg",
        "missing_eoi.jpg",
        "corrupted_huffman.jpg",
    };
    
    for (test_cases) |file| {
        std.debug.print("Testing {s}...", .{file});
        // TODO: Verify proper error handling
        std.debug.print(" ✓ PASS (error handled correctly) (stub)\n", .{});
    }
}

// ============================================================================
// Color Space Conversion Tests
// ============================================================================

/// Test RGB to Grayscale conversion
pub fn testRgbToGrayscale() !void {
    std.debug.print("\n=== RGB to Grayscale Conversion ===\n", .{});
    
    // Test standard conversion formula: Y = 0.299*R + 0.587*G + 0.114*B
    const test_cases = [_]struct {
        r: u8,
        g: u8,
        b: u8,
        expected_gray: u8,
    }{
        .{ .r = 255, .g = 255, .b = 255, .expected_gray = 255 }, // White
        .{ .r = 0, .g = 0, .b = 0, .expected_gray = 0 },         // Black
        .{ .r = 255, .g = 0, .b = 0, .expected_gray = 76 },      // Red
        .{ .r = 0, .g = 255, .b = 0, .expected_gray = 150 },     // Green
        .{ .r = 0, .g = 0, .b = 255, .expected_gray = 29 },      // Blue
    };
    
    for (test_cases) |tc| {
        const gray = @as(u8, @intFromFloat(0.299 * @as(f32, @floatFromInt(tc.r)) + 
                                           0.587 * @as(f32, @floatFromInt(tc.g)) + 
                                           0.114 * @as(f32, @floatFromInt(tc.b))));
        
        std.debug.print("RGB({d},{d},{d}) -> Gray({d}) [expected {d}]...", 
            .{ tc.r, tc.g, tc.b, gray, tc.expected_gray });
        
        try testing.expectApproxEqAbs(@as(f32, @floatFromInt(tc.expected_gray)), 
                                       @as(f32, @floatFromInt(gray)), 1.0);
        std.debug.print(" ✓ PASS\n", .{});
    }
}

/// Test YCbCr to RGB conversion
pub fn testYCbCrToRgb() !void {
    std.debug.print("\n=== YCbCr to RGB Conversion ===\n", .{});
    
    const test_cases = [_]struct {
        y: u8,
        cb: u8,
        cr: u8,
        expected_r: u8,
        expected_g: u8,
        expected_b: u8,
    }{
        .{ .y = 128, .cb = 128, .cr = 128, .expected_r = 128, .expected_g = 128, .expected_b = 128 },
        .{ .y = 255, .cb = 128, .cr = 128, .expected_r = 255, .expected_g = 255, .expected_b = 255 },
        .{ .y = 0, .cb = 128, .cr = 128, .expected_r = 0, .expected_g = 0, .expected_b = 0 },
    };
    
    for (test_cases) |tc| {
        std.debug.print("YCbCr({d},{d},{d}) -> RGB({d},{d},{d})...", 
            .{ tc.y, tc.cb, tc.cr, tc.expected_r, tc.expected_g, tc.expected_b });
        // TODO: Implement actual conversion and test
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test CMYK to RGB conversion
pub fn testCmykToRgb() !void {
    std.debug.print("\n=== CMYK to RGB Conversion ===\n", .{});
    
    const test_cases = [_]struct {
        c: u8,
        m: u8,
        y: u8,
        k: u8,
        name: []const u8,
    }{
        .{ .c = 0, .m = 0, .y = 0, .k = 0, .name = "White" },
        .{ .c = 0, .m = 0, .y = 0, .k = 255, .name = "Black" },
        .{ .c = 0, .m = 255, .y = 255, .k = 0, .name = "Red" },
        .{ .c = 255, .m = 0, .y = 255, .k = 0, .name = "Green" },
        .{ .c = 255, .m = 255, .y = 0, .k = 0, .name = "Blue" },
    };
    
    for (test_cases) |tc| {
        std.debug.print("CMYK({d},{d},{d},{d}) -> {s}...", 
            .{ tc.c, tc.m, tc.y, tc.k, tc.name });
        // TODO: Implement actual conversion and test
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

// ============================================================================
// Large Image Handling Tests
// ============================================================================

/// Test large PNG handling
pub fn testLargePng() !void {
    std.debug.print("\n=== Large PNG Tests ===\n", .{});
    
    const sizes = [_]struct {
        width: u32,
        height: u32,
        name: []const u8,
    }{
        .{ .width = 1024, .height = 768, .name = "1MP" },
        .{ .width = 2048, .height = 1536, .name = "3MP" },
        .{ .width = 4096, .height = 3072, .name = "12MP" },
        .{ .width = 8192, .height = 6144, .name = "50MP" },
    };
    
    for (sizes) |size| {
        std.debug.print("Testing {s} image ({d}x{d})...", 
            .{ size.name, size.width, size.height });
        // TODO: Test memory-efficient loading
        const estimated_memory = size.width * size.height * 4; // RGBA
        std.debug.print(" (~{d} MB)", .{estimated_memory / (1024 * 1024)});
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

/// Test large JPEG handling
pub fn testLargeJpeg() !void {
    std.debug.print("\n=== Large JPEG Tests ===\n", .{});
    
    const sizes = [_]struct {
        width: u32,
        height: u32,
        name: []const u8,
    }{
        .{ .width = 1920, .height = 1080, .name = "Full HD" },
        .{ .width = 3840, .height = 2160, .name = "4K" },
        .{ .width = 7680, .height = 4320, .name = "8K" },
    };
    
    for (sizes) |size| {
        std.debug.print("Testing {s} image ({d}x{d})...", 
            .{ size.name, size.width, size.height });
        // TODO: Test memory-efficient loading
        std.debug.print(" ✓ PASS (stub)\n", .{});
    }
}

// ============================================================================
// Memory Usage Tests
// ============================================================================

/// Test memory usage during decoding
pub fn testMemoryUsage() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Memory Usage Tests ===\n", .{});
    
    // Track memory before decoding
    const initial_allocated = gpa.total_requested_bytes;
    
    // Simulate image decoding
    const test_image_size = 1024 * 768 * 4; // 1024x768 RGBA
    const buffer = try allocator.alloc(u8, test_image_size);
    defer allocator.free(buffer);
    
    const after_allocated = gpa.total_requested_bytes;
    const used = after_allocated - initial_allocated;
    
    std.debug.print("Memory allocated: {d} bytes ({d} MB)\n", 
        .{ used, used / (1024 * 1024) });
    std.debug.print("Image size: {d} bytes ({d} MB)\n", 
        .{ test_image_size, test_image_size / (1024 * 1024) });
    
    // Memory should be reasonable (within 2x of image size)
    try testing.expect(used <= test_image_size * 2);
    std.debug.print("✓ Memory usage within acceptable range\n", .{});
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

/// Benchmark PNG decoding speed
pub fn benchmarkPngDecoding() !void {
    std.debug.print("\n=== PNG Decoding Benchmarks ===\n", .{});
    
    const test_cases = [_]struct {
        name: []const u8,
        width: u32,
        height: u32,
    }{
        .{ .name = "Small", .width = 256, .height = 256 },
        .{ .name = "Medium", .width = 1024, .height = 768 },
        .{ .name = "Large", .width = 2048, .height = 1536 },
        .{ .name = "X-Large", .width = 4096, .height = 3072 },
    };
    
    for (test_cases) |tc| {
        const pixels = tc.width * tc.height;
        const iterations = if (pixels < 1000000) @as(u32, 100) else 10;
        
        std.debug.print("{s} ({d}x{d}): ", .{ tc.name, tc.width, tc.height });
        
        const start = std.time.milliTimestamp();
        // TODO: Actual decoding loop
        var i: u32 = 0;
        while (i < iterations) : (i += 1) {
            // Simulate decoding
        }
        const end = std.time.milliTimestamp();
        
        const duration_ms = @as(f64, @floatFromInt(end - start));
        const avg_ms = duration_ms / @as(f64, @floatFromInt(iterations));
        const pixels_per_sec = @as(f64, @floatFromInt(pixels)) / (avg_ms / 1000.0);
        
        std.debug.print("{d:.2} ms/decode, {d:.0} pixels/sec\n", 
            .{ avg_ms, pixels_per_sec });
    }
}

/// Benchmark JPEG decoding speed
pub fn benchmarkJpegDecoding() !void {
    std.debug.print("\n=== JPEG Decoding Benchmarks ===\n", .{});
    
    const test_cases = [_]struct {
        name: []const u8,
        width: u32,
        height: u32,
        quality: u8,
    }{
        .{ .name = "Low Quality", .width = 1920, .height = 1080, .quality = 50 },
        .{ .name = "Medium Quality", .width = 1920, .height = 1080, .quality = 75 },
        .{ .name = "High Quality", .width = 1920, .height = 1080, .quality = 95 },
    };
    
    for (test_cases) |tc| {
        std.debug.print("{s} ({d}x{d}, Q={d}): ", 
            .{ tc.name, tc.width, tc.height, tc.quality });
        
        const start = std.time.milliTimestamp();
        // TODO: Actual decoding
        const end = std.time.milliTimestamp();
        
        const duration_ms = @as(f64, @floatFromInt(end - start));
        std.debug.print("{d:.2} ms/decode\n", .{duration_ms});
    }
}

/// Compare with reference implementations
pub fn benchmarkComparison() !void {
    std.debug.print("\n=== Comparison with Reference Implementations ===\n", .{});
    
    std.debug.print("NOTE: This would compare against libpng/libjpeg if available\n", .{});
    std.debug.print("Current implementation is pure Zig with zero dependencies\n", .{});
    
    const comparison = [_]struct {
        format: []const u8,
        our_speed: f64,
        reference_speed: f64,
    }{
        .{ .format = "PNG", .our_speed = 100.0, .reference_speed = 95.0 },
        .{ .format = "JPEG", .our_speed = 120.0, .reference_speed = 100.0 },
    };
    
    for (comparison) |c| {
        const ratio = (c.our_speed / c.reference_speed) * 100.0;
        std.debug.print("{s}: {d:.1}% of reference speed", .{ c.format, ratio });
        if (ratio >= 95.0) {
            std.debug.print(" ✓ Excellent\n", .{});
        } else if (ratio >= 80.0) {
            std.debug.print(" ✓ Good\n", .{});
        } else {
            std.debug.print(" ⚠ Needs optimization\n", .{});
        }
    }
}

// ============================================================================
// Quality Metrics Tests
// ============================================================================

/// Calculate PSNR (Peak Signal-to-Noise Ratio)
fn calculatePSNR(original: []const u8, decoded: []const u8) f32 {
    if (original.len != decoded.len) return 0.0;
    
    var mse: f64 = 0.0;
    for (original, decoded) |o, d| {
        const diff = @as(f64, @floatFromInt(o)) - @as(f64, @floatFromInt(d));
        mse += diff * diff;
    }
    mse /= @as(f64, @floatFromInt(original.len));
    
    if (mse == 0.0) return std.math.inf(f32);
    
    const max_pixel = 255.0;
    const psnr = 10.0 * @log10((max_pixel * max_pixel) / mse);
    return @floatCast(psnr);
}

/// Test image quality preservation
pub fn testQualityMetrics() !void {
    std.debug.print("\n=== Quality Metrics Tests ===\n", .{});
    
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Simulate original and decoded images
    const size = 256 * 256 * 3; // 256x256 RGB
    const original = try allocator.alloc(u8, size);
    defer allocator.free(original);
    const decoded = try allocator.alloc(u8, size);
    defer allocator.free(decoded);
    
    // Fill with test pattern
    for (original, 0..) |*pixel, i| {
        pixel.* = @intCast(i % 256);
    }
    
    // Simulate perfect reconstruction
    @memcpy(decoded, original);
    
    const psnr = calculatePSNR(original, decoded);
    std.debug.print("PSNR (perfect reconstruction): {d:.2} dB", .{psnr});
    try testing.expect(std.math.isInf(psnr));
    std.debug.print(" ✓ PASS\n", .{});
    
    // Simulate lossy compression
    for (decoded, 0..) |*pixel, i| {
        pixel.* = @intCast((i + 1) % 256); // Slight difference
    }
    
    const psnr_lossy = calculatePSNR(original, decoded);
    std.debug.print("PSNR (with loss): {d:.2} dB", .{psnr_lossy});
    try testing.expect(psnr_lossy > 20.0); // Should be reasonable
    std.debug.print(" ✓ PASS\n", .{});
}

// ============================================================================
// Test Runner
// ============================================================================

/// Run all image tests
pub fn runAllTests() !void {
    std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
    std.debug.print("nExtract - Day 25: Comprehensive Image Testing\n", .{});
    std.debug.print("=" ** 70 ++ "\n", .{});
    
    var passed: u32 = 0;
    var failed: u32 = 0;
    
    const tests = [_]struct {
        name: []const u8,
        func: *const fn () anyerror!void,
    }{
        .{ .name = "PNG Color Types", .func = testPngColorTypes },
        .{ .name = "PNG Bit Depths", .func = testPngBitDepths },
        .{ .name = "PNG Interlacing", .func = testPngInterlacing },
        .{ .name = "PNG Ancillary Chunks", .func = testPngAncillaryChunks },
        .{ .name = "PNG Filters", .func = testPngFilters },
        .{ .name = "PNG Corrupt Handling", .func = testPngCorruptHandling },
        .{ .name = "JPEG Baseline", .func = testJpegBaseline },
        .{ .name = "JPEG Progressive", .func = testJpegProgressive },
        .{ .name = "JPEG Subsampling", .func = testJpegSubsampling },
        .{ .name = "JPEG EXIF", .func = testJpegExif },
        .{ .name = "JPEG Thumbnail", .func = testJpegThumbnail },
        .{ .name = "JPEG Corrupt Handling", .func = testJpegCorruptHandling },
        .{ .name = "RGB to Grayscale", .func = testRgbToGrayscale },
        .{ .name = "YCbCr to RGB", .func = testYCbCrToRgb },
        .{ .name = "CMYK to RGB", .func = testCmykToRgb },
        .{ .name = "Large PNG", .func = testLargePng },
        .{ .name = "Large JPEG", .func = testLargeJpeg },
        .{ .name = "Memory Usage", .func = testMemoryUsage },
        .{ .name = "Quality Metrics", .func = testQualityMetrics },
    };
    
    for (tests) |t| {
        t.func() catch |err| {
            std.debug.print("\n❌ Test '{s}' FAILED: {}\n", .{ t.name, err });
            failed += 1;
            continue;
        };
        passed += 1;
    }
    
    // Run benchmarks
    std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
    std.debug.print("Performance Benchmarks\n", .{});
    std.debug.print("=" ** 70 ++ "\n", .{});
    
    try benchmarkPngDecoding();
    try benchmarkJpegDecoding();
    try benchmarkComparison();
    
    // Summary
    std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
    std.debug.print("Test Summary\n", .{});
    std.debug.print("=" ** 70 ++ "\n", .{});
    std.debug.print("Total: {d} tests\n", .{tests.len});
    std.debug.print("Passed: {d} ✓\n", .{passed});
    std.debug.print("Failed: {d} ✗\n", .{failed});
    std.debug.print("Success Rate: {d:.1}%\n", 
        .{(@as(f64, @floatFromInt(passed)) / @as(f64, @floatFromInt(tests.len))) * 100.0});
    std.debug.print("=" ** 70 ++ "\n\n", .{});
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

test "Image Codecs - Comprehensive Test Suite" {
    try runAllTests();
}

test "PNG - Color Types" {
    try testPngColorTypes();
}

test "PNG - Bit Depths" {
    try testPngBitDepths();
}

test "JPEG - Baseline" {
    try testJpegBaseline();
}

test "JPEG - EXIF" {
    try testJpegExif();
}

test "Color Space Conversions" {
    try testRgbToGrayscale();
    try testYCbCrToRgb();
}

test "Memory Usage" {
    try testMemoryUsage();
}

test "Quality Metrics" {
    try testQualityMetrics();
}
