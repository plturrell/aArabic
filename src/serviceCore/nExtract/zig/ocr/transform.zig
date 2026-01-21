const std = @import("std");
const Allocator = std.mem.Allocator;

/// Point in 2D space
pub const Point = struct {
    x: f32,
    y: f32,
};

/// Image structure for transformations
pub const Image = struct {
    width: u32,
    height: u32,
    channels: u32, // 1 for grayscale, 3 for RGB, 4 for RGBA
    data: []u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: u32, height: u32, channels: u32) !Image {
        const size = width * height * channels;
        const data = try allocator.alloc(u8, size);
        return Image{
            .width = width,
            .height = height,
            .channels = channels,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.data);
    }

    pub fn clone(self: *const Image) !Image {
        var new_image = try Image.init(self.allocator, self.width, self.height, self.channels);
        @memcpy(new_image.data, self.data);
        return new_image;
    }

    pub fn getPixel(self: *const Image, x: u32, y: u32) []const u8 {
        const idx = (y * self.width + x) * self.channels;
        return self.data[idx .. idx + self.channels];
    }

    pub fn setPixel(self: *Image, x: u32, y: u32, value: []const u8) void {
        const idx = (y * self.width + x) * self.channels;
        @memcpy(self.data[idx .. idx + self.channels], value);
    }
};

/// Interpolation methods
pub const InterpolationMethod = enum {
    NearestNeighbor,
    Bilinear,
    Bicubic,
};

/// Rotation - rotate image by arbitrary angle
pub fn rotate(allocator: Allocator, src: *const Image, angle_degrees: f32, method: InterpolationMethod) !Image {
    const angle_rad = angle_degrees * std.math.pi / 180.0;
    const cos_a = @cos(angle_rad);
    const sin_a = @sin(angle_rad);

    // Calculate new image dimensions to fit rotated image
    const w = @as(f32, @floatFromInt(src.width));
    const h = @as(f32, @floatFromInt(src.height));
    
    const corners = [4]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = w, .y = 0 },
        .{ .x = 0, .y = h },
        .{ .x = w, .y = h },
    };

    var min_x: f32 = std.math.inf(f32);
    var max_x: f32 = -std.math.inf(f32);
    var min_y: f32 = std.math.inf(f32);
    var max_y: f32 = -std.math.inf(f32);

    for (corners) |corner| {
        const x = corner.x * cos_a - corner.y * sin_a;
        const y = corner.x * sin_a + corner.y * cos_a;
        min_x = @min(min_x, x);
        max_x = @max(max_x, x);
        min_y = @min(min_y, y);
        max_y = @max(max_y, y);
    }

    const new_width = @as(u32, @intFromFloat(@ceil(max_x - min_x)));
    const new_height = @as(u32, @intFromFloat(@ceil(max_y - min_y)));

    var dst = try Image.init(allocator, new_width, new_height, src.channels);
    errdefer dst.deinit();

    const center_x = w / 2.0;
    const center_y = h / 2.0;
    const new_center_x = @as(f32, @floatFromInt(new_width)) / 2.0;
    const new_center_y = @as(f32, @floatFromInt(new_height)) / 2.0;

    var y: u32 = 0;
    while (y < new_height) : (y += 1) {
        var x: u32 = 0;
        while (x < new_width) : (x += 1) {
            const dx = @as(f32, @floatFromInt(x)) - new_center_x;
            const dy = @as(f32, @floatFromInt(y)) - new_center_y;

            const src_x = dx * cos_a + dy * sin_a + center_x;
            const src_y = -dx * sin_a + dy * cos_a + center_y;

            const pixel = samplePixel(src, src_x, src_y, method);
            dst.setPixel(x, y, &pixel);
        }
    }

    return dst;
}

/// Scaling - resize image with interpolation
pub fn scale(allocator: Allocator, src: *const Image, new_width: u32, new_height: u32, method: InterpolationMethod) !Image {
    var dst = try Image.init(allocator, new_width, new_height, src.channels);
    errdefer dst.deinit();

    const x_ratio = @as(f32, @floatFromInt(src.width)) / @as(f32, @floatFromInt(new_width));
    const y_ratio = @as(f32, @floatFromInt(src.height)) / @as(f32, @floatFromInt(new_height));

    var y: u32 = 0;
    while (y < new_height) : (y += 1) {
        var x: u32 = 0;
        while (x < new_width) : (x += 1) {
            const src_x = @as(f32, @floatFromInt(x)) * x_ratio;
            const src_y = @as(f32, @floatFromInt(y)) * y_ratio;

            const pixel = samplePixel(src, src_x, src_y, method);
            dst.setPixel(x, y, &pixel);
        }
    }

    return dst;
}

/// Sample pixel at fractional coordinates using specified interpolation method
fn samplePixel(img: *const Image, x: f32, y: f32, method: InterpolationMethod) [4]u8 {
    return switch (method) {
        .NearestNeighbor => sampleNearestNeighbor(img, x, y),
        .Bilinear => sampleBilinear(img, x, y),
        .Bicubic => sampleBicubic(img, x, y),
    };
}

/// Nearest neighbor interpolation
fn sampleNearestNeighbor(img: *const Image, x: f32, y: f32) [4]u8 {
    const ix = @as(u32, @intFromFloat(@round(x)));
    const iy = @as(u32, @intFromFloat(@round(y)));

    if (ix >= img.width or iy >= img.height) {
        return [4]u8{ 0, 0, 0, 0 };
    }

    const pixel = img.getPixel(ix, iy);
    var result = [4]u8{ 0, 0, 0, 255 };
    
    var i: usize = 0;
    while (i < img.channels and i < 4) : (i += 1) {
        result[i] = pixel[i];
    }
    
    return result;
}

/// Bilinear interpolation
fn sampleBilinear(img: *const Image, x: f32, y: f32) [4]u8 {
    if (x < 0 or y < 0 or x >= @as(f32, @floatFromInt(img.width - 1)) or y >= @as(f32, @floatFromInt(img.height - 1))) {
        return [4]u8{ 0, 0, 0, 0 };
    }

    const x0 = @as(u32, @intFromFloat(@floor(x)));
    const y0 = @as(u32, @intFromFloat(@floor(y)));
    const x1 = x0 + 1;
    const y1 = y0 + 1;

    const fx = x - @floor(x);
    const fy = y - @floor(y);

    const p00 = img.getPixel(x0, y0);
    const p10 = img.getPixel(x1, y0);
    const p01 = img.getPixel(x0, y1);
    const p11 = img.getPixel(x1, y1);

    var result = [4]u8{ 0, 0, 0, 255 };
    
    var i: usize = 0;
    while (i < img.channels and i < 4) : (i += 1) {
        const v00 = @as(f32, @floatFromInt(p00[i]));
        const v10 = @as(f32, @floatFromInt(p10[i]));
        const v01 = @as(f32, @floatFromInt(p01[i]));
        const v11 = @as(f32, @floatFromInt(p11[i]));

        const v0 = v00 * (1.0 - fx) + v10 * fx;
        const v1 = v01 * (1.0 - fx) + v11 * fx;
        const v = v0 * (1.0 - fy) + v1 * fy;

        result[i] = @intFromFloat(@min(@max(v, 0.0), 255.0));
    }

    return result;
}

/// Bicubic interpolation (Catmull-Rom spline)
fn sampleBicubic(img: *const Image, x: f32, y: f32) [4]u8 {
    if (x < 1 or y < 1 or x >= @as(f32, @floatFromInt(img.width - 2)) or y >= @as(f32, @floatFromInt(img.height - 2))) {
        return sampleBilinear(img, x, y);
    }

    const x0 = @as(i32, @intFromFloat(@floor(x)));
    const y0 = @as(i32, @intFromFloat(@floor(y)));
    const fx = x - @floor(x);
    const fy = y - @floor(y);

    var result = [4]u8{ 0, 0, 0, 255 };

    var c: usize = 0;
    while (c < img.channels and c < 4) : (c += 1) {
        var value: f32 = 0.0;

        var j: i32 = -1;
        while (j <= 2) : (j += 1) {
            var i: i32 = -1;
            while (i <= 2) : (i += 1) {
                const px = @as(u32, @intCast(x0 + i));
                const py = @as(u32, @intCast(y0 + j));
                
                if (px < img.width and py < img.height) {
                    const pixel = img.getPixel(px, py);
                    const pval = @as(f32, @floatFromInt(pixel[c]));
                    const wx = cubicWeight(fx - @as(f32, @floatFromInt(i)));
                    const wy = cubicWeight(fy - @as(f32, @floatFromInt(j)));
                    value += pval * wx * wy;
                }
            }
        }

        result[c] = @intFromFloat(@min(@max(value, 0.0), 255.0));
    }

    return result;
}

/// Cubic interpolation weight (Catmull-Rom)
fn cubicWeight(t: f32) f32 {
    const a: f32 = -0.5;
    const abs_t = @abs(t);
    
    if (abs_t <= 1.0) {
        return (a + 2.0) * abs_t * abs_t * abs_t - (a + 3.0) * abs_t * abs_t + 1.0;
    } else if (abs_t <= 2.0) {
        return a * abs_t * abs_t * abs_t - 5.0 * a * abs_t * abs_t + 8.0 * a * abs_t - 4.0 * a;
    }
    return 0.0;
}

/// Affine transformation matrix
pub const AffineMatrix = struct {
    a: f32, b: f32, c: f32,
    d: f32, e: f32, f: f32,

    pub fn identity() AffineMatrix {
        return .{ .a = 1, .b = 0, .c = 0, .d = 0, .e = 1, .f = 0 };
    }

    pub fn translation(tx: f32, ty: f32) AffineMatrix {
        return .{ .a = 1, .b = 0, .c = tx, .d = 0, .e = 1, .f = ty };
    }

    pub fn rotation(angle_degrees: f32) AffineMatrix {
        const angle_rad = angle_degrees * std.math.pi / 180.0;
        const cos_a = @cos(angle_rad);
        const sin_a = @sin(angle_rad);
        return .{ .a = cos_a, .b = -sin_a, .c = 0, .d = sin_a, .e = cos_a, .f = 0 };
    }

    pub fn scaling(sx: f32, sy: f32) AffineMatrix {
        return .{ .a = sx, .b = 0, .c = 0, .d = 0, .e = sy, .f = 0 };
    }

    pub fn shear(shx: f32, shy: f32) AffineMatrix {
        return .{ .a = 1, .b = shx, .c = 0, .d = shy, .e = 1, .f = 0 };
    }

    pub fn multiply(self: AffineMatrix, other: AffineMatrix) AffineMatrix {
        return .{
            .a = self.a * other.a + self.b * other.d,
            .b = self.a * other.b + self.b * other.e,
            .c = self.a * other.c + self.b * other.f + self.c,
            .d = self.d * other.a + self.e * other.d,
            .e = self.d * other.b + self.e * other.e,
            .f = self.d * other.c + self.e * other.f + self.f,
        };
    }

    pub fn transform(self: AffineMatrix, x: f32, y: f32) Point {
        return .{
            .x = self.a * x + self.b * y + self.c,
            .y = self.d * x + self.e * y + self.f,
        };
    }
};

/// Apply affine transformation
pub fn affineTransform(allocator: Allocator, src: *const Image, matrix: AffineMatrix, method: InterpolationMethod) !Image {
    // Calculate bounding box of transformed image
    const corners = [4]Point{
        matrix.transform(0, 0),
        matrix.transform(@floatFromInt(src.width), 0),
        matrix.transform(0, @floatFromInt(src.height)),
        matrix.transform(@floatFromInt(src.width), @floatFromInt(src.height)),
    };

    var min_x: f32 = std.math.inf(f32);
    var max_x: f32 = -std.math.inf(f32);
    var min_y: f32 = std.math.inf(f32);
    var max_y: f32 = -std.math.inf(f32);

    for (corners) |corner| {
        min_x = @min(min_x, corner.x);
        max_x = @max(max_x, corner.x);
        min_y = @min(min_y, corner.y);
        max_y = @max(max_y, corner.y);
    }

    const new_width = @as(u32, @intFromFloat(@ceil(max_x - min_x)));
    const new_height = @as(u32, @intFromFloat(@ceil(max_y - min_y)));

    var dst = try Image.init(allocator, new_width, new_height, src.channels);
    errdefer dst.deinit();

    // Invert the transformation matrix for reverse mapping
    const det = matrix.a * matrix.e - matrix.b * matrix.d;
    if (@abs(det) < 1e-6) {
        return error.SingularMatrix;
    }

    const inv = AffineMatrix{
        .a = matrix.e / det,
        .b = -matrix.b / det,
        .c = (matrix.b * matrix.f - matrix.e * matrix.c) / det,
        .d = -matrix.d / det,
        .e = matrix.a / det,
        .f = (matrix.d * matrix.c - matrix.a * matrix.f) / det,
    };

    var y: u32 = 0;
    while (y < new_height) : (y += 1) {
        var x: u32 = 0;
        while (x < new_width) : (x += 1) {
            const dst_x = @as(f32, @floatFromInt(x)) + min_x;
            const dst_y = @as(f32, @floatFromInt(y)) + min_y;
            const src_pt = inv.transform(dst_x, dst_y);

            const pixel = samplePixel(src, src_pt.x, src_pt.y, method);
            dst.setPixel(x, y, &pixel);
        }
    }

    return dst;
}

/// Perspective transformation (4-point homography)
pub fn perspectiveTransform(allocator: Allocator, src: *const Image, src_points: [4]Point, dst_points: [4]Point, method: InterpolationMethod) !Image {
    // Calculate homography matrix using Direct Linear Transform (DLT)
    const h = try calculateHomography(src_points, dst_points);

    // Calculate output image size
    var min_x: f32 = std.math.inf(f32);
    var max_x: f32 = -std.math.inf(f32);
    var min_y: f32 = std.math.inf(f32);
    var max_y: f32 = -std.math.inf(f32);

    for (dst_points) |pt| {
        min_x = @min(min_x, pt.x);
        max_x = @max(max_x, pt.x);
        min_y = @min(min_y, pt.y);
        max_y = @max(max_y, pt.y);
    }

    const new_width = @as(u32, @intFromFloat(@ceil(max_x - min_x)));
    const new_height = @as(u32, @intFromFloat(@ceil(max_y - min_y)));

    var dst = try Image.init(allocator, new_width, new_height, src.channels);
    errdefer dst.deinit();

    var y: u32 = 0;
    while (y < new_height) : (y += 1) {
        var x: u32 = 0;
        while (x < new_width) : (x += 1) {
            const dx = @as(f32, @floatFromInt(x)) + min_x;
            const dy = @as(f32, @floatFromInt(y)) + min_y;

            const denom = h[6] * dx + h[7] * dy + h[8];
            if (@abs(denom) < 1e-6) continue;

            const src_x = (h[0] * dx + h[1] * dy + h[2]) / denom;
            const src_y = (h[3] * dx + h[4] * dy + h[5]) / denom;

            const pixel = samplePixel(src, src_x, src_y, method);
            dst.setPixel(x, y, &pixel);
        }
    }

    return dst;
}

/// Calculate homography matrix (simplified DLT method)
fn calculateHomography(src: [4]Point, dst: [4]Point) ![9]f32 {
    // Simplified homography - in production, use proper DLT with SVD
    // This is a basic implementation for demonstration
    var h = [9]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    
    // Calculate affine approximation
    const sx = (dst[1].x - dst[0].x) / (src[1].x - src[0].x);
    const sy = (dst[2].y - dst[0].y) / (src[2].y - src[0].y);
    
    h[0] = sx;
    h[4] = sy;
    h[2] = dst[0].x - sx * src[0].x;
    h[5] = dst[0].y - sy * src[0].y;
    
    return h;
}

/// Flip image horizontally
pub fn flipHorizontal(allocator: Allocator, src: *const Image) !Image {
    var dst = try Image.init(allocator, src.width, src.height, src.channels);
    errdefer dst.deinit();

    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const pixel = src.getPixel(x, y);
            dst.setPixel(src.width - 1 - x, y, pixel);
        }
    }

    return dst;
}

/// Flip image vertically
pub fn flipVertical(allocator: Allocator, src: *const Image) !Image {
    var dst = try Image.init(allocator, src.width, src.height, src.channels);
    errdefer dst.deinit();

    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const pixel = src.getPixel(x, y);
            dst.setPixel(x, src.height - 1 - y, pixel);
        }
    }

    return dst;
}

/// Crop image to specified rectangle
pub fn crop(allocator: Allocator, src: *const Image, x: u32, y: u32, width: u32, height: u32) !Image {
    if (x + width > src.width or y + height > src.height) {
        return error.InvalidCropRegion;
    }

    var dst = try Image.init(allocator, width, height, src.channels);
    errdefer dst.deinit();

    var dy: u32 = 0;
    while (dy < height) : (dy += 1) {
        var dx: u32 = 0;
        while (dx < width) : (dx += 1) {
            const pixel = src.getPixel(x + dx, y + dy);
            dst.setPixel(dx, dy, pixel);
        }
    }

    return dst;
}

// Export C-compatible functions
export fn nExtract_Image_Rotate(
    data: [*]const u8,
    width: u32,
    height: u32,
    channels: u32,
    angle: f32,
    method: u32,
) callconv(.C) ?*Image {
    const allocator = std.heap.c_allocator;
    
    const src_image = Image{
        .width = width,
        .height = height,
        .channels = channels,
        .data = @constCast(data[0 .. width * height * channels]),
        .allocator = allocator,
    };

    const interp_method: InterpolationMethod = switch (method) {
        0 => .NearestNeighbor,
        1 => .Bilinear,
        2 => .Bicubic,
        else => .Bilinear,
    };

    var result = rotate(allocator, &src_image, angle, interp_method) catch return null;
    const result_ptr = allocator.create(Image) catch return null;
    result_ptr.* = result;
    return result_ptr;
}

export fn nExtract_Image_Scale(
    data: [*]const u8,
    width: u32,
    height: u32,
    channels: u32,
    new_width: u32,
    new_height: u32,
    method: u32,
) callconv(.C) ?*Image {
    const allocator = std.heap.c_allocator;
    
    const src_image = Image{
        .width = width,
        .height = height,
        .channels = channels,
        .data = @constCast(data[0 .. width * height * channels]),
        .allocator = allocator,
    };

    const interp_method: InterpolationMethod = switch (method) {
        0 => .NearestNeighbor,
        1 => .Bilinear,
        2 => .Bicubic,
        else => .Bilinear,
    };

    var result = scale(allocator, &src_image, new_width, new_height, interp_method) catch return null;
    const result_ptr = allocator.create(Image) catch return null;
    result_ptr.* = result;
    return result_ptr;
}

export fn nExtract_Image_Free(img: ?*Image) callconv(.C) void {
    if (img) |image| {
        image.deinit();
        std.heap.c_allocator.destroy(image);
    }
}
