const std = @import("std");
const threshold = @import("../ocr/threshold.zig");
const Image = threshold.Image;

test "image creation and cleanup" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 100, 100);
    defer img.deinit();
    
    try std.testing.expectEqual(@as(u32, 100), img.width);
    try std.testing.expectEqual(@as(u32, 100), img.height);
    try std.testing.expectEqual(@as(usize, 10000), img.data.len);
}

test "global thresholding" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10);
    defer img.deinit();
    
    // Fill with gradient
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        img.data[i] = @intCast(i * 2);
    }
    
    var thresholded = try threshold.globalThreshold(allocator, &img, 128);
    defer thresholded.deinit();
    
    // Check that pixels below threshold are 0, above are 255
    try std.testing.expectEqual(@as(u8, 0), thresholded.data[0]); // 0 < 128
    try std.testing.expectEqual(@as(u8, 255), thresholded.data[99]); // 198 > 128
}

test "Otsu's method" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20);
    defer img.deinit();
    
    // Create bimodal distribution (two peaks)
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        if (i < 100) {
            img.data[i] = 50; // Dark pixels
        } else {
            img.data[i] = 200; // Bright pixels
        }
    }
    
    var thresholded = try threshold.otsuThreshold(allocator, &img);
    defer thresholded.deinit();
    
    // Verify binarization
    try std.testing.expectEqual(@as(u8, 0), thresholded.data[0]); // Dark -> 0
    try std.testing.expectEqual(@as(u8, 255), thresholded.data[150]); // Bright -> 255
}

test "adaptive threshold - mean" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20);
    defer img.deinit();
    
    // Fill with varying intensity
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            const val = @min((x + y) * 6, 255);
            img.setPixel(x, y, @intCast(val));
        }
    }
    
    var thresholded = try threshold.adaptiveThresholdMean(allocator, &img, 5, 10);
    defer thresholded.deinit();
    
    // Verify adaptive thresholding applied
    try std.testing.expect(thresholded.width == 20);
    try std.testing.expect(thresholded.height == 20);
}

test "adaptive threshold - Gaussian" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20);
    defer img.deinit();
    
    // Fill with checkerboard pattern
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            const val: u8 = if ((x / 5 + y / 5) % 2 == 0) 100 else 200;
            img.setPixel(x, y, val);
        }
    }
    
    var thresholded = try threshold.adaptiveThresholdGaussian(allocator, &img, 7, 5);
    defer thresholded.deinit();
    
    // Verify adaptive thresholding applied
    try std.testing.expect(thresholded.width == 20);
    try std.testing.expect(thresholded.height == 20);
}

test "Sauvola thresholding" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20);
    defer img.deinit();
    
    // Create document-like image with text and background
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            // Simulate text (dark) on light background
            const val: u8 = if (x > 8 and x < 12) 50 else 200;
            img.setPixel(x, y, val);
        }
    }
    
    var thresholded = try threshold.sauvolaThreshold(allocator, &img, 5, 0.5, 128.0);
    defer thresholded.deinit();
    
    // Verify binarization
    try std.testing.expectEqual(@as(u8, 0), thresholded.getPixel(10, 10)); // Text
    try std.testing.expectEqual(@as(u8, 255), thresholded.getPixel(5, 10)); // Background
}

test "Niblack thresholding" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20);
    defer img.deinit();
    
    // Fill with gradient
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            const val = @min(x * 12, 255);
            img.setPixel(x, y, @intCast(val));
        }
    }
    
    var thresholded = try threshold.niblackThreshold(allocator, &img, 5, -0.2);
    defer thresholded.deinit();
    
    // Verify thresholding applied
    try std.testing.expect(thresholded.width == 20);
    try std.testing.expect(thresholded.height == 20);
}

test "Bradley adaptive thresholding" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20);
    defer img.deinit();
    
    // Create document-like image
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            // Simulate varying illumination
            const base: u8 = @intCast(100 + (x + y) * 2);
            const val: u8 = if ((x / 3) % 3 == 0) base - 50 else base;
            img.setPixel(x, y, val);
        }
    }
    
    var thresholded = try threshold.bradleyThreshold(allocator, &img, 8, 0.15);
    defer thresholded.deinit();
    
    // Verify thresholding applied
    try std.testing.expect(thresholded.width == 20);
    try std.testing.expect(thresholded.height == 20);
}

test "hysteresis thresholding" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20);
    defer img.deinit();
    
    // Create edge-like pattern
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            if (x == 10) {
                img.setPixel(x, y, 200); // Strong edge
            } else if (x == 9 or x == 11) {
                img.setPixel(x, y, 100); // Weak edge
            } else {
                img.setPixel(x, y, 20); // Background
            }
        }
    }
    
    var thresholded = try threshold.hysteresisThreshold(allocator, &img, 80, 150);
    defer thresholded.deinit();
    
    // Verify edge detection
    try std.testing.expectEqual(@as(u8, 255), thresholded.getPixel(10, 10)); // Strong edge
    try std.testing.expectEqual(@as(u8, 255), thresholded.getPixel(9, 10)); // Connected weak edge
    try std.testing.expectEqual(@as(u8, 0), thresholded.getPixel(5, 10)); // Background
}

test "histogram calculation" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10);
    defer img.deinit();
    
    // Fill with known values
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        img.data[i] = 100;
    }
    while (i < 100) : (i += 1) {
        img.data[i] = 200;
    }
    
    // Otsu will calculate histogram internally
    var thresholded = try threshold.otsuThreshold(allocator, &img);
    defer thresholded.deinit();
    
    // Should threshold between 100 and 200
    try std.testing.expectEqual(@as(u8, 0), thresholded.data[0]); // 100 -> 0
    try std.testing.expectEqual(@as(u8, 255), thresholded.data[99]); // 200 -> 255
}

test "varying illumination" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 30, 30);
    defer img.deinit();
    
    // Simulate document with varying illumination
    var y: u32 = 0;
    while (y < 30) : (y += 1) {
        var x: u32 = 0;
        while (x < 30) : (x += 1) {
            // Illumination gradient from left to right
            const illumination = @min(100 + x * 3, 255);
            // Text in middle
            const is_text = (x > 12 and x < 18);
            const val: u8 = if (is_text) 
                @intCast(@max(0, @as(i32, illumination) - 80))
            else 
                @intCast(illumination);
            img.setPixel(x, y, val);
        }
    }
    
    // Adaptive methods should handle this better than global
    var adaptive = try threshold.adaptiveThresholdGaussian(allocator, &img, 11, 10);
    defer adaptive.deinit();
    
    // Text should be detected even with illumination variation
    try std.testing.expectEqual(@as(u8, 0), adaptive.getPixel(15, 15)); // Text
    try std.testing.expectEqual(@as(u8, 255), adaptive.getPixel(5, 15)); // Background
}

test "document binarization comparison" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 50, 50);
    defer img.deinit();
    
    // Create document-like image
    var y: u32 = 0;
    while (y < 50) : (y += 1) {
        var x: u32 = 0;
        while (x < 50) : (x += 1) {
            // Background with slight noise
            var val: u8 = 220;
            // Add text regions
            if ((x > 10 and x < 20) or (x > 30 and x < 40)) {
                if (y % 5 < 3) {
                    val = 40; // Text
                }
            }
            img.setPixel(x, y, val);
        }
    }
    
    // Try different methods
    var otsu_result = try threshold.otsuThreshold(allocator, &img);
    defer otsu_result.deinit();
    
    var sauvola_result = try threshold.sauvolaThreshold(allocator, &img, 15, 0.5, 128.0);
    defer sauvola_result.deinit();
    
    var adaptive_result = try threshold.adaptiveThresholdGaussian(allocator, &img, 11, 10);
    defer adaptive_result.deinit();
    
    // All should detect text regions
    try std.testing.expectEqual(@as(u8, 0), otsu_result.getPixel(15, 10));
    try std.testing.expectEqual(@as(u8, 0), sauvola_result.getPixel(15, 10));
    try std.testing.expectEqual(@as(u8, 0), adaptive_result.getPixel(15, 10));
}

test "edge cases - uniform image" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10);
    defer img.deinit();
    
    // Fill with uniform value
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        img.data[i] = 128;
    }
    
    var thresholded = try threshold.otsuThreshold(allocator, &img);
    defer thresholded.deinit();
    
    // Should handle uniform image gracefully
    try std.testing.expect(thresholded.width == 10);
    try std.testing.expect(thresholded.height == 10);
}

test "edge cases - all black" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10);
    defer img.deinit();
    
    // Fill with zeros
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        img.data[i] = 0;
    }
    
    var thresholded = try threshold.globalThreshold(allocator, &img, 128);
    defer thresholded.deinit();
    
    // All should remain zero
    try std.testing.expectEqual(@as(u8, 0), thresholded.data[0]);
    try std.testing.expectEqual(@as(u8, 0), thresholded.data[99]);
}

test "edge cases - all white" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10);
    defer img.deinit();
    
    // Fill with 255
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        img.data[i] = 255;
    }
    
    var thresholded = try threshold.globalThreshold(allocator, &img, 128);
    defer thresholded.deinit();
    
    // All should become 255
    try std.testing.expectEqual(@as(u8, 255), thresholded.data[0]);
    try std.testing.expectEqual(@as(u8, 255), thresholded.data[99]);
}

test "window size effects" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 30, 30);
    defer img.deinit();
    
    // Create pattern
    var y: u32 = 0;
    while (y < 30) : (y += 1) {
        var x: u32 = 0;
        while (x < 30) : (x += 1) {
            const val: u8 = if ((x / 5 + y / 5) % 2 == 0) 80 else 180;
            img.setPixel(x, y, val);
        }
    }
    
    // Test different window sizes
    var small_window = try threshold.adaptiveThresholdMean(allocator, &img, 5, 5);
    defer small_window.deinit();
    
    var large_window = try threshold.adaptiveThresholdMean(allocator, &img, 15, 5);
    defer large_window.deinit();
    
    // Both should produce valid results
    try std.testing.expect(small_window.width == 30);
    try std.testing.expect(large_window.width == 30);
}
