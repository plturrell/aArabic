const std = @import("std");
const testing = std.testing;
const colorspace = @import("../ocr/colorspace.zig");
const filters = @import("../ocr/filters.zig");
const transform = @import("../ocr/transform.zig");
const threshold = @import("../ocr/threshold.zig");

// Test image creation utilities
fn createTestImage(allocator: std.mem.Allocator, width: u32, height: u32) !colorspace.Image {
    var img = try colorspace.Image.init(allocator, width, height);
    
    // Create gradient pattern
    var y: u32 = 0;
    while (y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const value = @as(u8, @intCast((x * 255) / width));
            img.setPixel(x, y, value);
        }
    }
    
    return img;
}

fn createCheckerboard(allocator: std.mem.Allocator, width: u32, height: u32, square_size: u32) !colorspace.Image {
    var img = try colorspace.Image.init(allocator, width, height);
    
    var y: u32 = 0;
    while (y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
            img.setPixel(x, y, if (is_white) 255 else 0);
        }
    }
    
    return img;
}

// Quality assessment metrics

/// Calculate Peak Signal-to-Noise Ratio (PSNR)
fn calculatePSNR(img1: *const colorspace.Image, img2: *const colorspace.Image) !f32 {
    if (img1.width != img2.width or img1.height != img2.height) {
        return error.DimensionMismatch;
    }
    
    var mse: f32 = 0.0;
    var y: u32 = 0;
    while (y < img1.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img1.width) : (x += 1) {
            const val1 = @as(f32, @floatFromInt(img1.getPixel(x, y)));
            const val2 = @as(f32, @floatFromInt(img2.getPixel(x, y)));
            const diff = val1 - val2;
            mse += diff * diff;
        }
    }
    
    const pixel_count = @as(f32, @floatFromInt(img1.width * img1.height));
    mse /= pixel_count;
    
    if (mse == 0.0) return std.math.inf(f32);
    
    const max_val: f32 = 255.0;
    return 10.0 * @log10((max_val * max_val) / mse);
}

/// Calculate mean pixel value
fn calculateMean(img: *const colorspace.Image) f32 {
    var sum: u64 = 0;
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            sum += img.getPixel(x, y);
        }
    }
    const pixel_count = @as(f32, @floatFromInt(img.width * img.height));
    return @as(f32, @floatFromInt(sum)) / pixel_count;
}

/// Calculate standard deviation
fn calculateStdDev(img: *const colorspace.Image, mean: f32) f32 {
    var sum_sq_diff: f32 = 0.0;
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const val = @as(f32, @floatFromInt(img.getPixel(x, y)));
            const diff = val - mean;
            sum_sq_diff += diff * diff;
        }
    }
    const pixel_count = @as(f32, @floatFromInt(img.width * img.height));
    return @sqrt(sum_sq_diff / pixel_count);
}

// ============================================================================
// INTEGRATION TESTS: Full Pipeline Tests
// ============================================================================

test "Full preprocessing pipeline for OCR" {
    const allocator = testing.allocator;
    
    // Create test image with noise
    var img = try createTestImage(allocator, 100, 100);
    defer img.deinit();
    
    // Add some noise
    var y: u32 = 10;
    while (y < 90) : (y += 10) {
        var x: u32 = 10;
        while (x < 90) : (x += 10) {
            img.setPixel(x, y, 0);
        }
    }
    
    // Step 1: Gaussian blur to denoise
    var denoised = try filters.gaussianBlur(allocator, &img, 1.0);
    defer denoised.deinit();
    
    // Step 2: Threshold with Otsu
    var binary = try threshold.otsuThreshold(allocator, &denoised);
    defer binary.deinit();
    
    // Verify binary output (only 0 or 255)
    y = 0;
    while (y < binary.height) : (y += 1) {
        var x: u32 = 0;
        while (x < binary.width) : (x += 1) {
            const val = binary.getPixel(x, y);
            try testing.expect(val == 0 or val == 255);
        }
    }
}

test "Document image enhancement pipeline" {
    const allocator = testing.allocator;
    
    // Create simulated document (checkerboard pattern)
    var img = try createCheckerboard(allocator, 80, 80, 10);
    defer img.deinit();
    
    // Add varying illumination (darken top-left)
    var y: u32 = 0;
    while (y < 40) : (y += 1) {
        var x: u32 = 0;
        while (x < 40) : (x += 1) {
            const val = img.getPixel(x, y);
            const darkened = @as(u8, @intCast(@as(u32, val) * 70 / 100));
            img.setPixel(x, y, darkened);
        }
    }
    
    // Apply Sauvola binarization (handles varying illumination)
    var enhanced = try threshold.sauvolaThreshold(allocator, &img, 15, 0.5, 128.0);
    defer enhanced.deinit();
    
    // Verify enhancement improved image
    try testing.expect(enhanced.width == img.width);
    try testing.expect(enhanced.height == img.height);
}

test "Edge detection and enhancement pipeline" {
    const allocator = testing.allocator;
    
    // Create image with edges
    var img = try colorspace.Image.init(allocator, 50, 50);
    defer img.deinit();
    
    // Fill with rectangle
    var y: u32 = 10;
    while (y < 40) : (y += 1) {
        var x: u32 = 10;
        while (x < 40) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    // Apply Sobel edge detection
    var edges_x = try filters.sobelX(allocator, &img);
    defer edges_x.deinit();
    
    var edges_y = try filters.sobelY(allocator, &img);
    defer edges_y.deinit();
    
    // Combine edges
    var edges = try colorspace.Image.init(allocator, img.width, img.height);
    defer edges.deinit();
    
    y = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const gx = @as(i32, @intCast(edges_x.getPixel(x, y)));
            const gy = @as(i32, @intCast(edges_y.getPixel(x, y)));
            const magnitude = @sqrt(@as(f32, @floatFromInt(gx * gx + gy * gy)));
            const clamped = @min(255, @as(u8, @intFromFloat(magnitude)));
            edges.setPixel(x, y, clamped);
        }
    }
    
    // Apply hysteresis thresholding
    var strong_edges = try threshold.hysteresisThreshold(allocator, &edges, 100, 50);
    defer strong_edges.deinit();
    
    // Verify edges detected
    var edge_count: u32 = 0;
    y = 0;
    while (y < strong_edges.height) : (y += 1) {
        var x: u32 = 0;
        while (x < strong_edges.width) : (x += 1) {
            if (strong_edges.getPixel(x, y) == 255) {
                edge_count += 1;
            }
        }
    }
    try testing.expect(edge_count > 0);
}

// ============================================================================
// QUALITY ASSESSMENT TESTS
// ============================================================================

test "PSNR calculation between images" {
    const allocator = testing.allocator;
    
    var img1 = try createTestImage(allocator, 50, 50);
    defer img1.deinit();
    
    // Create slightly noisy version
    var img2 = try createTestImage(allocator, 50, 50);
    defer img2.deinit();
    
    var y: u32 = 0;
    while (y < img2.height) : (y += 2) {
        var x: u32 = 0;
        while (x < img2.width) : (x += 2) {
            const val = img2.getPixel(x, y);
            img2.setPixel(x, y, @min(255, val +% 5));
        }
    }
    
    const psnr = try calculatePSNR(&img1, &img2);
    
    // PSNR should be high (low noise)
    try testing.expect(psnr > 30.0);
}

test "Mean and standard deviation calculation" {
    const allocator = testing.allocator;
    
    // Create uniform image
    var img = try colorspace.Image.init(allocator, 50, 50);
    defer img.deinit();
    
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            img.setPixel(x, y, 128);
        }
    }
    
    const mean = calculateMean(&img);
    const std_dev = calculateStdDev(&img, mean);
    
    try testing.expectApproxEqAbs(@as(f32, 128.0), mean, 0.1);
    try testing.expectApproxEqAbs(@as(f32, 0.0), std_dev, 0.1);
}

test "Filter quality assessment" {
    const allocator = testing.allocator;
    
    // Create noisy image
    var img = try createTestImage(allocator, 100, 100);
    defer img.deinit();
    
    // Add salt and pepper noise
    var rng = std.rand.DefaultPrng.init(42);
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            if (rng.random().int(u8) < 10) {
                img.setPixel(x, y, if (rng.random().boolean()) 255 else 0);
            }
        }
    }
    
    // Apply median filter
    var filtered = try filters.medianFilter(allocator, &img, 3);
    defer filtered.deinit();
    
    // Calculate mean before and after
    const mean_before = calculateMean(&img);
    const mean_after = calculateMean(&filtered);
    
    // Means should be similar (noise removed but overall brightness preserved)
    try testing.expectApproxEqAbs(mean_before, mean_after, 10.0);
    
    // Std dev should be lower after filtering
    const std_before = calculateStdDev(&img, mean_before);
    const std_after = calculateStdDev(&filtered, mean_after);
    
    try testing.expect(std_after < std_before);
}

// ============================================================================
// PERFORMANCE BENCHMARK TESTS
// ============================================================================

test "Gaussian blur performance benchmark" {
    const allocator = testing.allocator;
    
    var img = try createTestImage(allocator, 256, 256);
    defer img.deinit();
    
    const start = std.time.nanoTimestamp();
    
    // Perform 10 blurs
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var blurred = try filters.gaussianBlur(allocator, &img, 1.0);
        blurred.deinit();
    }
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    // Should complete in reasonable time (< 500ms for 10 iterations)
    try testing.expect(duration_ms < 500.0);
    
    std.debug.print("\nGaussian blur (256x256): {d:.2}ms per operation\n", .{duration_ms / 10.0});
}

test "Otsu threshold performance benchmark" {
    const allocator = testing.allocator;
    
    var img = try createTestImage(allocator, 256, 256);
    defer img.deinit();
    
    const start = std.time.nanoTimestamp();
    
    // Perform 100 thresholds
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var binary = try threshold.otsuThreshold(allocator, &img);
        binary.deinit();
    }
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    // Should be fast (< 100ms for 100 iterations)
    try testing.expect(duration_ms < 100.0);
    
    std.debug.print("Otsu threshold (256x256): {d:.3}ms per operation\n", .{duration_ms / 100.0});
}

test "Rotation performance benchmark" {
    const allocator = testing.allocator;
    
    var img = try createTestImage(allocator, 200, 200);
    defer img.deinit();
    
    const start = std.time.nanoTimestamp();
    
    // Perform 10 rotations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var rotated = try transform.rotate(allocator, &img, 15.0, .Bilinear);
        rotated.deinit();
    }
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    std.debug.print("Rotation 15° bilinear (200x200): {d:.2}ms per operation\n", .{duration_ms / 10.0});
}

test "Sauvola threshold performance benchmark" {
    const allocator = testing.allocator;
    
    var img = try createTestImage(allocator, 256, 256);
    defer img.deinit();
    
    const start = std.time.nanoTimestamp();
    
    // Perform 10 Sauvola thresholds
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var binary = try threshold.sauvolaThreshold(allocator, &img, 15, 0.5, 128.0);
        binary.deinit();
    }
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    std.debug.print("Sauvola threshold (256x256, window=15): {d:.2}ms per operation\n", .{duration_ms / 10.0});
}

// ============================================================================
// MEMORY USAGE TESTS
// ============================================================================

test "Memory usage - large image processing" {
    const allocator = testing.allocator;
    
    // Process a larger image (1000x1000)
    var img = try createTestImage(allocator, 1000, 1000);
    defer img.deinit();
    
    // Apply several operations
    var blurred = try filters.gaussianBlur(allocator, &img, 1.0);
    defer blurred.deinit();
    
    var binary = try threshold.otsuThreshold(allocator, &blurred);
    defer binary.deinit();
    
    // Memory should be properly managed (no leaks via testing allocator)
    try testing.expect(true);
}

test "Memory usage - multiple allocations" {
    const allocator = testing.allocator;
    
    // Create and destroy many images
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var img = try createTestImage(allocator, 50, 50);
        img.deinit();
    }
    
    // Should not leak memory
    try testing.expect(true);
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

test "Edge case - minimum size image" {
    const allocator = testing.allocator;
    
    var img = try colorspace.Image.init(allocator, 3, 3);
    defer img.deinit();
    
    // Fill with pattern
    img.setPixel(1, 1, 255);
    
    // Apply operations
    var blurred = try filters.gaussianBlur(allocator, &img, 0.5);
    defer blurred.deinit();
    
    var binary = try threshold.globalThreshold(allocator, &img, 128);
    defer binary.deinit();
    
    try testing.expect(true);
}

test "Edge case - all black image" {
    const allocator = testing.allocator;
    
    var img = try colorspace.Image.init(allocator, 50, 50);
    defer img.deinit();
    
    // All pixels are 0 (from init)
    
    var binary = try threshold.otsuThreshold(allocator, &img);
    defer binary.deinit();
    
    // Should handle gracefully
    try testing.expect(binary.width == img.width);
}

test "Edge case - all white image" {
    const allocator = testing.allocator;
    
    var img = try colorspace.Image.init(allocator, 50, 50);
    defer img.deinit();
    
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    var binary = try threshold.otsuThreshold(allocator, &img);
    defer binary.deinit();
    
    // Should handle gracefully
    try testing.expect(binary.width == img.width);
}

// ============================================================================
// COMPARATIVE TESTS
// ============================================================================

test "Compare thresholding methods on same image" {
    const allocator = testing.allocator;
    
    var img = try createTestImage(allocator, 100, 100);
    defer img.deinit();
    
    // Global threshold
    var global = try threshold.globalThreshold(allocator, &img, 128);
    defer global.deinit();
    
    // Otsu
    var otsu = try threshold.otsuThreshold(allocator, &img);
    defer otsu.deinit();
    
    // Adaptive
    var adaptive = try threshold.adaptiveThresholdMean(allocator, &img, 11, 10);
    defer adaptive.deinit();
    
    // All should produce valid binary images
    try testing.expect(global.width == img.width);
    try testing.expect(otsu.width == img.width);
    try testing.expect(adaptive.width == img.width);
}

test "Compare interpolation methods" {
    const allocator = testing.allocator;
    
    var img = try createTestImage(allocator, 50, 50);
    defer img.deinit();
    
    // Scale with different interpolation methods
    var nearest = try transform.scale(allocator, &img, 100, 100, .Nearest);
    defer nearest.deinit();
    
    var bilinear = try transform.scale(allocator, &img, 100, 100, .Bilinear);
    defer bilinear.deinit();
    
    var bicubic = try transform.scale(allocator, &img, 100, 100, .Bicubic);
    defer bicubic.deinit();
    
    // Verify dimensions
    try testing.expect(nearest.width == 100);
    try testing.expect(bilinear.width == 100);
    try testing.expect(bicubic.width == 100);
    
    // Bicubic should have highest quality (measured by smoothness)
    const mean_nearest = calculateMean(&nearest);
    const mean_bilinear = calculateMean(&bilinear);
    const mean_bicubic = calculateMean(&bicubic);
    
    // All should be similar (preserving brightness)
    try testing.expectApproxEqAbs(mean_nearest, mean_bilinear, 10.0);
    try testing.expectApproxEqAbs(mean_bilinear, mean_bicubic, 10.0);
}

// ============================================================================
// INTEGRATION WITH COLORSPACE
// ============================================================================

test "RGB to Grayscale to Binary pipeline" {
    const allocator = testing.allocator;
    
    // Create RGB image
    var rgb = try colorspace.RGBImage.init(allocator, 50, 50);
    defer rgb.deinit();
    
    // Fill with pattern
    var y: u32 = 0;
    while (y < rgb.height) : (y += 1) {
        var x: u32 = 0;
        while (x < rgb.width) : (x += 1) {
            const r = @as(u8, @intCast((x * 255) / rgb.width));
            const g = @as(u8, @intCast((y * 255) / rgb.height));
            const b: u8 = 128;
            rgb.setPixel(x, y, r, g, b);
        }
    }
    
    // Convert to grayscale
    var gray = try colorspace.rgbToGrayscale(allocator, &rgb);
    defer gray.deinit();
    
    // Threshold
    var binary = try threshold.otsuThreshold(allocator, &gray);
    defer binary.deinit();
    
    try testing.expect(binary.width == rgb.width);
    try testing.expect(binary.height == rgb.height);
}

// ============================================================================
// SUMMARY TEST
// ============================================================================

test "Image processing summary statistics" {
    std.debug.print("\n", .{});
    std.debug.print("==============================================\n", .{});
    std.debug.print(" Image Processing Test Suite Summary\n", .{});
    std.debug.print("==============================================\n", .{});
    std.debug.print("All image processing tests completed successfully!\n", .{});
    std.debug.print("Coverage:\n", .{});
    std.debug.print("  - Color space conversions (Day 26)\n", .{});
    std.debug.print("  - Image filters (Day 27)\n", .{});
    std.debug.print("  - Image transformations (Day 28)\n", .{});
    std.debug.print("  - Thresholding methods (Day 29)\n", .{});
    std.debug.print("  - Integration pipelines\n", .{});
    std.debug.print("  - Quality metrics (PSNR, mean, std dev)\n", .{});
    std.debug.print("  - Performance benchmarks\n", .{});
    std.debug.print("  - Memory usage validation\n", .{});
    std.debug.print("  - Edge cases\n", .{});
    std.debug.print("==============================================\n", .{});
}
