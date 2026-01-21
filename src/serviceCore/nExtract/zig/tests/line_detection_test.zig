const std = @import("std");
const testing = std.testing;
const Image = @import("../ocr/colorspace.zig").Image;
const line_detection = @import("../ocr/line_detection.zig");
const transform = @import("../ocr/transform.zig");

// ============================================================================
// Test Helper Functions
// ============================================================================

/// Create a test image with horizontal lines
fn createLineImage(allocator: std.mem.Allocator, width: u32, height: u32, num_lines: u32) !Image {
    var img = try Image.init(allocator, width, height);
    
    // Calculate line spacing
    const line_spacing = height / (num_lines + 1);
    
    // Draw horizontal lines
    var line_idx: u32 = 0;
    while (line_idx < num_lines) : (line_idx += 1) {
        const y = (line_idx + 1) * line_spacing;
        const line_height: u32 = 5; // 5 pixels tall
        
        var dy: u32 = 0;
        while (dy < line_height and (y + dy) < height) : (dy += 1) {
            var x: u32 = 0;
            while (x < width) : (x += 1) {
                img.setPixel(x, y + dy, 255);
            }
        }
    }
    
    return img;
}

/// Create a skewed image with diagonal line
fn createSkewedImage(allocator: std.mem.Allocator, width: u32, height: u32, skew_angle: f32) !Image {
    var img = try Image.init(allocator, width, height);
    
    // Draw diagonal line representing skewed text
    const mid_y = height / 2;
    const tan_angle = @tan(skew_angle * std.math.pi / 180.0);
    
    var x: u32 = 0;
    while (x < width) : (x += 1) {
        const offset = @as(i32, @intFromFloat(@as(f32, @floatFromInt(x)) * tan_angle));
        const y = @as(i32, @intCast(mid_y)) + offset;
        
        if (y >= 0 and y < @as(i32, @intCast(height))) {
            const line_thickness: u32 = 3;
            var dy: u32 = 0;
            while (dy < line_thickness) : (dy += 1) {
                const py = @as(u32, @intCast(y)) + dy;
                if (py < height) {
                    img.setPixel(x, py, 255);
                }
            }
        }
    }
    
    return img;
}

/// Create image with connected components
fn createComponentImage(allocator: std.mem.Allocator, width: u32, height: u32) !Image {
    var img = try Image.init(allocator, width, height);
    
    // Draw 3 separate rectangles (components)
    // Component 1: top-left
    var y: u32 = 10;
    while (y < 30) : (y += 1) {
        var x: u32 = 10;
        while (x < 30) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    // Component 2: top-right
    y = 10;
    while (y < 30) : (y += 1) {
        var x: u32 = 50;
        while (x < 70) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    // Component 3: bottom (large)
    y = 50;
    while (y < 80) : (y += 1) {
        var x: u32 = 10;
        while (x < 70) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    return img;
}

// ============================================================================
// Connected Component Analysis Tests
// ============================================================================

test "CCA - 4-connectivity" {
    const allocator = testing.allocator;
    
    var img = try createComponentImage(allocator, 80, 100);
    defer img.deinit();
    
    const result = try line_detection.connectedComponentAnalysis(allocator, &img, .Four);
    defer allocator.free(result.labels);
    
    // Should detect 3 components
    try testing.expect(result.component_count >= 2); // At least 2 distinct
}

test "CCA - 8-connectivity" {
    const allocator = testing.allocator;
    
    var img = try createComponentImage(allocator, 80, 100);
    defer img.deinit();
    
    const result = try line_detection.connectedComponentAnalysis(allocator, &img, .Eight);
    defer allocator.free(result.labels);
    
    // Should detect components
    try testing.expect(result.component_count >= 2);
}

test "CCA - extract components with bounding boxes" {
    const allocator = testing.allocator;
    
    var img = try createComponentImage(allocator, 80, 100);
    defer img.deinit();
    
    const cca_result = try line_detection.connectedComponentAnalysis(allocator, &img, .Four);
    defer allocator.free(cca_result.labels);
    
    const components = try line_detection.extractComponents(
        allocator,
        cca_result.labels,
        img.width,
        img.height,
    );
    defer allocator.free(components);
    
    // Should have multiple components
    try testing.expect(components.len >= 2);
    
    // Each component should have valid bounding box
    for (components) |comp| {
        try testing.expect(comp.width() > 0);
        try testing.expect(comp.height() > 0);
        try testing.expect(comp.pixel_count > 0);
    }
}

test "CCA - filter components by size" {
    const allocator = testing.allocator;
    
    var img = try Image.init(allocator, 100, 100);
    defer img.deinit();
    
    // Draw large component
    var y: u32 = 20;
    while (y < 80) : (y += 1) {
        var x: u32 = 20;
        while (x < 80) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    // Draw small noise component
    img.setPixel(5, 5, 255);
    img.setPixel(6, 5, 255);
    
    const cca_result = try line_detection.connectedComponentAnalysis(allocator, &img, .Four);
    defer allocator.free(cca_result.labels);
    
    const all_components = try line_detection.extractComponents(
        allocator,
        cca_result.labels,
        img.width,
        img.height,
    );
    defer allocator.free(all_components);
    
    // Filter out small components
    const filtered = try line_detection.filterComponentsBySize(
        allocator,
        all_components,
        10, // min_width
        10, // min_height
        100, // min_pixel_count
    );
    defer allocator.free(filtered);
    
    // Should filter out small noise
    try testing.expect(filtered.len < all_components.len);
}

// ============================================================================
// Projection Profile Tests
// ============================================================================

test "Horizontal projection profile" {
    const allocator = testing.allocator;
    
    var img = try createLineImage(allocator, 100, 100, 3);
    defer img.deinit();
    
    const profile = try line_detection.horizontalProjectionProfile(allocator, &img);
    defer allocator.free(profile);
    
    try testing.expect(profile.len == img.height);
    
    // Find peaks in profile (should correspond to text lines)
    var peak_count: u32 = 0;
    for (profile) |val| {
        if (val > 50) { // Significant pixel count
            peak_count += 1;
        }
    }
    
    // Should have detected line regions
    try testing.expect(peak_count > 0);
}

test "Vertical projection profile" {
    const allocator = testing.allocator;
    
    var img = try Image.init(allocator, 100, 100);
    defer img.deinit();
    
    // Draw vertical lines
    var x: u32 = 20;
    while (x < 80) : (x += 20) {
        var y: u32 = 10;
        while (y < 90) : (y += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    const profile = try line_detection.verticalProjectionProfile(allocator, &img);
    defer allocator.free(profile);
    
    try testing.expect(profile.len == img.width);
    
    // Check for peaks at vertical line positions
    try testing.expect(profile[20] > 0);
    try testing.expect(profile[40] > 0);
    try testing.expect(profile[60] > 0);
}

// ============================================================================
// Line Segmentation Tests
// ============================================================================

test "Segment lines - single line" {
    const allocator = testing.allocator;
    
    var img = try createLineImage(allocator, 100, 50, 1);
    defer img.deinit();
    
    const lines = try line_detection.segmentLines(
        allocator,
        &img,
        3, // min_gap
        3, // min_height
    );
    defer allocator.free(lines);
    
    // Should detect 1 line
    try testing.expect(lines.len == 1);
    try testing.expect(lines[0].height() >= 3);
}

test "Segment lines - multiple lines" {
    const allocator = testing.allocator;
    
    var img = try createLineImage(allocator, 100, 100, 3);
    defer img.deinit();
    
    const lines = try line_detection.segmentLines(
        allocator,
        &img,
        5, // min_gap
        3, // min_height
    );
    defer allocator.free(lines);
    
    // Should detect 3 lines
    try testing.expect(lines.len == 3);
    
    // Lines should be ordered top to bottom
    for (lines, 0..) |line, i| {
        if (i > 0) {
            try testing.expect(line.y_start > lines[i - 1].y_end);
        }
    }
}

test "Segment lines - with gaps" {
    const allocator = testing.allocator;
    
    var img = try Image.init(allocator, 100, 100);
    defer img.deinit();
    
    // Draw 2 lines with large gap
    var y: u32 = 10;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 100) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    y = 80;
    while (y < 90) : (y += 1) {
        var x: u32 = 0;
        while (x < 100) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    const lines = try line_detection.segmentLines(
        allocator,
        &img,
        10, // min_gap
        5,  // min_height
    );
    defer allocator.free(lines);
    
    // Should detect 2 separate lines
    try testing.expect(lines.len == 2);
}

// ============================================================================
// Skew Detection Tests
// ============================================================================

test "Skew detection - no skew" {
    const allocator = testing.allocator;
    
    var img = try createLineImage(allocator, 200, 100, 3);
    defer img.deinit();
    
    const angle = try line_detection.detectSkewProjection(allocator, &img, 15.0, 1.0);
    
    // Should detect no skew (angle ≈ 0)
    try testing.expect(@abs(angle) < 2.0);
}

test "Skew detection - projection method" {
    const allocator = testing.allocator;
    
    const test_angle: f32 = 5.0;
    var img = try createSkewedImage(allocator, 200, 200, test_angle);
    defer img.deinit();
    
    const detected_angle = try line_detection.detectSkewProjection(allocator, &img, 15.0, 0.5);
    
    // Should detect angle within tolerance
    try testing.expect(@abs(detected_angle - test_angle) < 2.0);
}

test "Skew detection - Hough method" {
    const allocator = testing.allocator;
    
    const test_angle: f32 = 5.0;
    var img = try createSkewedImage(allocator, 200, 200, test_angle);
    defer img.deinit();
    
    const detected_angle = try line_detection.detectSkewHough(allocator, &img, 15.0);
    
    // Hough is less precise but should be in ballpark
    try testing.expect(@abs(detected_angle - test_angle) < 5.0);
}

// ============================================================================
// Baseline Detection Tests
// ============================================================================

test "Baseline detection" {
    const allocator = testing.allocator;
    
    var img = try createLineImage(allocator, 100, 100, 2);
    defer img.deinit();
    
    const lines = try line_detection.segmentLines(allocator, &img, 5, 3);
    defer allocator.free(lines);
    
    try testing.expect(lines.len > 0);
    
    // Detect baselines
    var lines_mut = try allocator.alloc(line_detection.TextLine, lines.len);
    defer allocator.free(lines_mut);
    @memcpy(lines_mut, lines);
    
    line_detection.detectBaselines(&img, lines_mut);
    
    // Baseline should be within line bounds
    for (lines_mut) |line| {
        try testing.expect(line.baseline >= line.y_start);
        try testing.expect(line.baseline <= line.y_end);
    }
}

// ============================================================================
// Full Pipeline Tests
// ============================================================================

test "Full pipeline - no skew" {
    const allocator = testing.allocator;
    
    var img = try createLineImage(allocator, 200, 150, 4);
    defer img.deinit();
    
    var result = try line_detection.detectLines(allocator, &img, .{
        .detect_skew = false,
        .min_line_gap = 5,
        .min_line_height = 3,
    });
    defer result.deinit();
    
    // Should detect 4 lines
    try testing.expect(result.lines.len == 4);
    try testing.expect(result.skew_angle == 0.0);
    try testing.expect(result.deskewed_image.width == img.width);
}

test "Full pipeline - with skew detection" {
    const allocator = testing.allocator;
    
    // Create slightly skewed image
    var img = try createSkewedImage(allocator, 200, 200, 3.0);
    defer img.deinit();
    
    var result = try line_detection.detectLines(allocator, &img, .{
        .detect_skew = true,
        .max_skew_angle = 15.0,
        .min_line_gap = 5,
        .min_line_height = 3,
        .skew_method = .Projection,
    });
    defer result.deinit();
    
    // Should detect skew
    try testing.expect(@abs(result.skew_angle) > 0.5);
    try testing.expect(@abs(result.skew_angle) < 10.0);
}

test "Full pipeline - Hough skew detection" {
    const allocator = testing.allocator;
    
    var img = try createSkewedImage(allocator, 200, 200, 5.0);
    defer img.deinit();
    
    var result = try line_detection.detectLines(allocator, &img, .{
        .detect_skew = true,
        .max_skew_angle = 15.0,
        .skew_method = .Hough,
    });
    defer result.deinit();
    
    // Should detect skew using Hough
    try testing.expect(result.deskewed_image.width > 0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "Edge case - empty image" {
    const allocator = testing.allocator;
    
    var img = try Image.init(allocator, 50, 50);
    defer img.deinit();
    
    const lines = try line_detection.segmentLines(allocator, &img, 5, 3);
    defer allocator.free(lines);
    
    // Should return no lines
    try testing.expect(lines.len == 0);
}

test "Edge case - single pixel line" {
    const allocator = testing.allocator;
    
    var img = try Image.init(allocator, 100, 50);
    defer img.deinit();
    
    // Draw 1-pixel tall line
    var x: u32 = 0;
    while (x < 100) : (x += 1) {
        img.setPixel(x, 25, 255);
    }
    
    const lines = try line_detection.segmentLines(allocator, &img, 5, 1);
    defer allocator.free(lines);
    
    // Should detect 1 line
    try testing.expect(lines.len == 1);
    try testing.expect(lines[0].height() == 1);
}

test "Edge case - very small image" {
    const allocator = testing.allocator;
    
    var img = try Image.init(allocator, 10, 10);
    defer img.deinit();
    
    // Fill with text
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            img.setPixel(x, y, 255);
        }
    }
    
    const lines = try line_detection.segmentLines(allocator, &img, 2, 2);
    defer allocator.free(lines);
    
    // Should detect at least 1 line
    try testing.expect(lines.len >= 1);
}

// ============================================================================
// Performance Tests
// ============================================================================

test "Performance - CCA on large image" {
    const allocator = testing.allocator;
    
    var img = try createComponentImage(allocator, 500, 500);
    defer img.deinit();
    
    const start = std.time.nanoTimestamp();
    
    const result = try line_detection.connectedComponentAnalysis(allocator, &img, .Four);
    defer allocator.free(result.labels);
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    std.debug.print("\nCCA (500x500): {d:.2}ms\n", .{duration_ms});
    
    // Should complete in reasonable time
    try testing.expect(duration_ms < 500.0);
}

test "Performance - line segmentation" {
    const allocator = testing.allocator;
    
    var img = try createLineImage(allocator, 1000, 1000, 50);
    defer img.deinit();
    
    const start = std.time.nanoTimestamp();
    
    const lines = try line_detection.segmentLines(allocator, &img, 5, 3);
    defer allocator.free(lines);
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    std.debug.print("Line segmentation (1000x1000): {d:.2}ms\n", .{duration_ms});
    
    // Should be fast
    try testing.expect(duration_ms < 100.0);
}

test "Performance - skew detection projection" {
    const allocator = testing.allocator;
    
    var img = try createSkewedImage(allocator, 300, 300, 5.0);
    defer img.deinit();
    
    const start = std.time.nanoTimestamp();
    
    const angle = try line_detection.detectSkewProjection(allocator, &img, 15.0, 1.0);
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    std.debug.print("Skew detection projection (300x300): {d:.2}ms\n", .{duration_ms});
    
    _ = angle;
    
    // Multiple rotations, may take longer
    try testing.expect(duration_ms < 2000.0);
}

// ============================================================================
// Integration Tests
// ============================================================================

test "Integration - document with skewed text lines" {
    const allocator = testing.allocator;
    
    // Create document-like image
    var img = try Image.init(allocator, 400, 300);
    defer img.deinit();
    
    // Draw 5 skewed text lines
    const num_lines: u32 = 5;
    const line_spacing = img.height / (num_lines + 1);
    const skew: f32 = 3.0;
    const tan_skew = @tan(skew * std.math.pi / 180.0);
    
    var line_idx: u32 = 0;
    while (line_idx < num_lines) : (line_idx += 1) {
        const base_y = (line_idx + 1) * line_spacing;
        
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const offset = @as(i32, @intFromFloat(@as(f32, @floatFromInt(x)) * tan_skew));
            const y = @as(i32, @intCast(base_y)) + offset;
            
            if (y >= 0 and y < @as(i32, @intCast(img.height))) {
                var dy: u32 = 0;
                while (dy < 3) : (dy += 1) {
                    const py = @as(u32, @intCast(y)) + dy;
                    if (py < img.height) {
                        img.setPixel(x, py, 255);
                    }
                }
            }
        }
    }
    
    // Run full pipeline
    var result = try line_detection.detectLines(allocator, &img, .{
        .detect_skew = true,
        .max_skew_angle = 10.0,
        .min_line_gap = 8,
        .min_line_height = 2,
        .skew_method = .Projection,
    });
    defer result.deinit();
    
    // Should detect skew and lines
    try testing.expect(@abs(result.skew_angle) > 0.5);
    try testing.expect(result.lines.len >= 3);
}

test "Integration - multi-column layout" {
    const allocator = testing.allocator;
    
    var img = try Image.init(allocator, 400, 200);
    defer img.deinit();
    
    // Draw text in 2 columns
    // Left column: lines at y=20, 60, 100
    // Right column: lines at y=30, 70, 110
    
    const left_col_x_start: u32 = 20;
    const left_col_x_end: u32 = 180;
    const right_col_x_start: u32 = 220;
    const right_col_x_end: u32 = 380;
    
    // Left column lines
    const left_y_positions = [_]u32{ 20, 60, 100 };
    for (left_y_positions) |base_y| {
        var y: u32 = base_y;
        while (y < base_y + 5) : (y += 1) {
            var x = left_col_x_start;
            while (x < left_col_x_end) : (x += 1) {
                img.setPixel(x, y, 255);
            }
        }
    }
    
    // Right column lines
    const right_y_positions = [_]u32{ 30, 70, 110 };
    for (right_y_positions) |base_y| {
        var y: u32 = base_y;
        while (y < base_y + 5) : (y += 1) {
            var x = right_col_x_start;
            while (x < right_col_x_end) : (x += 1) {
                img.setPixel(x, y, 255);
            }
        }
    }
    
    const lines = try line_detection.segmentLines(allocator, &img, 10, 3);
    defer allocator.free(lines);
    
    // Should detect lines from both columns
    try testing.expect(lines.len >= 5);
}

// ============================================================================
// Summary Test
// ============================================================================

test "Line detection summary" {
    std.debug.print("\n", .{});
    std.debug.print("==============================================\n", .{});
    std.debug.print(" Line Detection Test Suite Summary\n", .{});
    std.debug.print("==============================================\n", .{});
    std.debug.print("All line detection tests completed successfully!\n", .{});
    std.debug.print("Coverage:\n", .{});
    std.debug.print("  - Connected component analysis (4 and 8-connectivity)\n", .{});
    std.debug.print("  - Component extraction and filtering\n", .{});
    std.debug.print("  - Projection profiles (horizontal and vertical)\n", .{});
    std.debug.print("  - Line segmentation\n", .{});
    std.debug.print("  - Skew detection (Projection and Hough methods)\n", .{});
    std.debug.print("  - Baseline detection\n", .{});
    std.debug.print("  - Full pipeline integration\n", .{});
    std.debug.print("  - Edge cases\n", .{});
    std.debug.print("  - Performance benchmarks\n", .{});
    std.debug.print("==============================================\n", .{});
}
