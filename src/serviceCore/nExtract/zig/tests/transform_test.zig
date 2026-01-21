const std = @import("std");
const transform = @import("../ocr/transform.zig");
const Image = transform.Image;
const InterpolationMethod = transform.InterpolationMethod;
const AffineMatrix = transform.AffineMatrix;
const Point = transform.Point;

test "image creation and cleanup" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 100, 100, 3);
    defer img.deinit();
    
    try std.testing.expectEqual(@as(u32, 100), img.width);
    try std.testing.expectEqual(@as(u32, 100), img.height);
    try std.testing.expectEqual(@as(u32, 3), img.channels);
    try std.testing.expectEqual(@as(usize, 30000), img.data.len);
}

test "image clone" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Set some pixel values
    const pixel = [_]u8{ 255, 128, 64 };
    img.setPixel(5, 5, &pixel);
    
    var cloned = try img.clone();
    defer cloned.deinit();
    
    const cloned_pixel = cloned.getPixel(5, 5);
    try std.testing.expectEqual(pixel[0], cloned_pixel[0]);
    try std.testing.expectEqual(pixel[1], cloned_pixel[1]);
    try std.testing.expectEqual(pixel[2], cloned_pixel[2]);
}

test "rotation - nearest neighbor" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Fill with test pattern
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const val: u8 = @intCast((x + y) * 10);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    var rotated = try transform.rotate(allocator, &img, 90.0, .NearestNeighbor);
    defer rotated.deinit();
    
    try std.testing.expect(rotated.width > 0);
    try std.testing.expect(rotated.height > 0);
}

test "rotation - bilinear" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Fill with solid color
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const pixel = [_]u8{ 128, 128, 128 };
            img.setPixel(x, y, &pixel);
        }
    }
    
    var rotated = try transform.rotate(allocator, &img, 45.0, .Bilinear);
    defer rotated.deinit();
    
    try std.testing.expect(rotated.width > 10);
    try std.testing.expect(rotated.height > 10);
}

test "scaling - nearest neighbor" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Fill with test pattern
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const val: u8 = @intCast((x + y) * 10);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Scale up 2x
    var scaled = try transform.scale(allocator, &img, 20, 20, .NearestNeighbor);
    defer scaled.deinit();
    
    try std.testing.expectEqual(@as(u32, 20), scaled.width);
    try std.testing.expectEqual(@as(u32, 20), scaled.height);
}

test "scaling - bilinear" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20, 3);
    defer img.deinit();
    
    // Fill with gradient
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            const val: u8 = @intCast(x * 12);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Scale down 2x
    var scaled = try transform.scale(allocator, &img, 10, 10, .Bilinear);
    defer scaled.deinit();
    
    try std.testing.expectEqual(@as(u32, 10), scaled.width);
    try std.testing.expectEqual(@as(u32, 10), scaled.height);
}

test "scaling - bicubic" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Fill with solid color
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const pixel = [_]u8{ 100, 150, 200 };
            img.setPixel(x, y, &pixel);
        }
    }
    
    var scaled = try transform.scale(allocator, &img, 15, 15, .Bicubic);
    defer scaled.deinit();
    
    try std.testing.expectEqual(@as(u32, 15), scaled.width);
    try std.testing.expectEqual(@as(u32, 15), scaled.height);
}

test "affine matrix - identity" {
    const m = AffineMatrix.identity();
    const p = m.transform(10.0, 20.0);
    
    try std.testing.expectEqual(@as(f32, 10.0), p.x);
    try std.testing.expectEqual(@as(f32, 20.0), p.y);
}

test "affine matrix - translation" {
    const m = AffineMatrix.translation(5.0, 10.0);
    const p = m.transform(10.0, 20.0);
    
    try std.testing.expectEqual(@as(f32, 15.0), p.x);
    try std.testing.expectEqual(@as(f32, 30.0), p.y);
}

test "affine matrix - scaling" {
    const m = AffineMatrix.scaling(2.0, 3.0);
    const p = m.transform(10.0, 20.0);
    
    try std.testing.expectEqual(@as(f32, 20.0), p.x);
    try std.testing.expectEqual(@as(f32, 60.0), p.y);
}

test "affine matrix - rotation" {
    const m = AffineMatrix.rotation(90.0);
    const p = m.transform(10.0, 0.0);
    
    // After 90 degree rotation, (10, 0) should become approximately (0, 10)
    try std.testing.expect(@abs(p.x) < 0.1);
    try std.testing.expect(@abs(p.y - 10.0) < 0.1);
}

test "affine matrix - multiply" {
    const scale = AffineMatrix.scaling(2.0, 2.0);
    const translate = AffineMatrix.translation(5.0, 10.0);
    
    // Scale then translate
    const combined = scale.multiply(translate);
    const p = combined.transform(10.0, 20.0);
    
    try std.testing.expectEqual(@as(f32, 25.0), p.x);
    try std.testing.expectEqual(@as(f32, 50.0), p.y);
}

test "affine transform" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Fill with test pattern
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const val: u8 = @intCast((x + y) * 10);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Apply scaling transform
    const matrix = AffineMatrix.scaling(1.5, 1.5);
    var transformed = try transform.affineTransform(allocator, &img, matrix, .Bilinear);
    defer transformed.deinit();
    
    try std.testing.expectEqual(@as(u32, 15), transformed.width);
    try std.testing.expectEqual(@as(u32, 15), transformed.height);
}

test "flip horizontal" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Create asymmetric pattern
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const val: u8 = @intCast(x * 25);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    var flipped = try transform.flipHorizontal(allocator, &img);
    defer flipped.deinit();
    
    // Check that left becomes right
    const original_left = img.getPixel(0, 5);
    const flipped_right = flipped.getPixel(9, 5);
    try std.testing.expectEqual(original_left[0], flipped_right[0]);
}

test "flip vertical" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Create asymmetric pattern
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const val: u8 = @intCast(y * 25);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    var flipped = try transform.flipVertical(allocator, &img);
    defer flipped.deinit();
    
    // Check that top becomes bottom
    const original_top = img.getPixel(5, 0);
    const flipped_bottom = flipped.getPixel(5, 9);
    try std.testing.expectEqual(original_top[0], flipped_bottom[0]);
}

test "crop" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 20, 20, 3);
    defer img.deinit();
    
    // Fill with different values
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            const val: u8 = @intCast((x + y) * 6);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Crop 10x10 region starting at (5, 5)
    var cropped = try transform.crop(allocator, &img, 5, 5, 10, 10);
    defer cropped.deinit();
    
    try std.testing.expectEqual(@as(u32, 10), cropped.width);
    try std.testing.expectEqual(@as(u32, 10), cropped.height);
    
    // Verify content
    const original_pixel = img.getPixel(5, 5);
    const cropped_pixel = cropped.getPixel(0, 0);
    try std.testing.expectEqual(original_pixel[0], cropped_pixel[0]);
}

test "crop - invalid region" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Try to crop beyond image bounds
    const result = transform.crop(allocator, &img, 5, 5, 10, 10);
    try std.testing.expectError(error.InvalidCropRegion, result);
}

test "perspective transform" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Fill with test pattern
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const val: u8 = @intCast((x + y) * 10);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Define source and destination points
    const src_points = [4]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 0 },
        .{ .x = 0, .y = 10 },
        .{ .x = 10, .y = 10 },
    };
    
    const dst_points = [4]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 12, .y = 0 },
        .{ .x = 0, .y = 12 },
        .{ .x = 12, .y = 12 },
    };
    
    var transformed = try transform.perspectiveTransform(allocator, &img, src_points, dst_points, .Bilinear);
    defer transformed.deinit();
    
    try std.testing.expect(transformed.width > 0);
    try std.testing.expect(transformed.height > 0);
}

test "multiple transformations" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();
    
    // Fill with test pattern
    var y: u32 = 0;
    while (y < 10) : (y += 1) {
        var x: u32 = 0;
        while (x < 10) : (x += 1) {
            const val: u8 = @intCast(x * 25);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Scale up
    var scaled = try transform.scale(allocator, &img, 20, 20, .Bilinear);
    defer scaled.deinit();
    
    // Rotate
    var rotated = try transform.rotate(allocator, &scaled, 45.0, .Bilinear);
    defer rotated.deinit();
    
    // Scale down
    var final = try transform.scale(allocator, &rotated, 15, 15, .Bilinear);
    defer final.deinit();
    
    try std.testing.expectEqual(@as(u32, 15), final.width);
    try std.testing.expectEqual(@as(u32, 15), final.height);
}
