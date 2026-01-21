const std = @import("std");
const Allocator = std.mem.Allocator;

/// Image structure for thresholding operations
pub const Image = struct {
    width: u32,
    height: u32,
    data: []u8, // Grayscale image (1 channel)
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: u32, height: u32) !Image {
        const data = try allocator.alloc(u8, width * height);
        return Image{
            .width = width,
            .height = height,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.data);
    }

    pub fn getPixel(self: *const Image, x: u32, y: u32) u8 {
        return self.data[y * self.width + x];
    }

    pub fn setPixel(self: *Image, x: u32, y: u32, value: u8) void {
        self.data[y * self.width + x] = value;
    }
};

/// Global thresholding with fixed threshold value
pub fn globalThreshold(allocator: Allocator, src: *const Image, threshold: u8) !Image {
    var dst = try Image.init(allocator, src.width, src.height);
    errdefer dst.deinit();

    var i: usize = 0;
    while (i < src.data.len) : (i += 1) {
        dst.data[i] = if (src.data[i] > threshold) 255 else 0;
    }

    return dst;
}

/// Calculate histogram of grayscale image
fn calculateHistogram(img: *const Image) [256]u32 {
    var histogram = [_]u32{0} ** 256;
    
    for (img.data) |pixel| {
        histogram[pixel] += 1;
    }
    
    return histogram;
}

/// Otsu's method for automatic threshold selection
pub fn otsuThreshold(allocator: Allocator, src: *const Image) !Image {
    const histogram = calculateHistogram(src);
    const total_pixels = src.width * src.height;
    
    // Calculate total sum
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < 256) : (i += 1) {
        sum += @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(histogram[i]));
    }
    
    var sum_background: f32 = 0.0;
    var weight_background: u32 = 0;
    var max_variance: f32 = 0.0;
    var threshold: u8 = 0;
    
    // Find threshold that maximizes between-class variance
    var t: usize = 0;
    while (t < 256) : (t += 1) {
        weight_background += histogram[t];
        if (weight_background == 0) continue;
        
        const weight_foreground = total_pixels - weight_background;
        if (weight_foreground == 0) break;
        
        sum_background += @as(f32, @floatFromInt(t)) * @as(f32, @floatFromInt(histogram[t]));
        
        const mean_background = sum_background / @as(f32, @floatFromInt(weight_background));
        const mean_foreground = (sum - sum_background) / @as(f32, @floatFromInt(weight_foreground));
        
        // Calculate between-class variance
        const variance = @as(f32, @floatFromInt(weight_background)) * 
                        @as(f32, @floatFromInt(weight_foreground)) * 
                        (mean_background - mean_foreground) * 
                        (mean_background - mean_foreground);
        
        if (variance > max_variance) {
            max_variance = variance;
            threshold = @intCast(t);
        }
    }
    
    return globalThreshold(allocator, src, threshold);
}

/// Adaptive thresholding method
pub const AdaptiveMethod = enum {
    Mean,
    Gaussian,
};

/// Adaptive thresholding - mean method
pub fn adaptiveThresholdMean(
    allocator: Allocator,
    src: *const Image,
    window_size: u32,
    constant: i32,
) !Image {
    var dst = try Image.init(allocator, src.width, src.height);
    errdefer dst.deinit();
    
    const half_window = window_size / 2;
    
    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            // Calculate mean in window
            var sum: u32 = 0;
            var count: u32 = 0;
            
            const y_start = if (y >= half_window) y - half_window else 0;
            const y_end = @min(y + half_window + 1, src.height);
            const x_start = if (x >= half_window) x - half_window else 0;
            const x_end = @min(x + half_window + 1, src.width);
            
            var wy = y_start;
            while (wy < y_end) : (wy += 1) {
                var wx = x_start;
                while (wx < x_end) : (wx += 1) {
                    sum += src.getPixel(wx, wy);
                    count += 1;
                }
            }
            
            const mean = sum / count;
            const threshold = @as(i32, @intCast(mean)) - constant;
            const pixel = src.getPixel(x, y);
            
            dst.setPixel(x, y, if (@as(i32, pixel) > threshold) 255 else 0);
        }
    }
    
    return dst;
}

/// Gaussian weight for adaptive thresholding
fn gaussianWeight(x: i32, y: i32, sigma: f32) f32 {
    const x_f = @as(f32, @floatFromInt(x));
    const y_f = @as(f32, @floatFromInt(y));
    const exp_arg = -(x_f * x_f + y_f * y_f) / (2.0 * sigma * sigma);
    return @exp(exp_arg);
}

/// Adaptive thresholding - Gaussian method
pub fn adaptiveThresholdGaussian(
    allocator: Allocator,
    src: *const Image,
    window_size: u32,
    constant: i32,
) !Image {
    var dst = try Image.init(allocator, src.width, src.height);
    errdefer dst.deinit();
    
    const half_window = window_size / 2;
    const sigma = @as(f32, @floatFromInt(window_size)) / 6.0;
    
    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            // Calculate Gaussian-weighted mean in window
            var weighted_sum: f32 = 0.0;
            var weight_sum: f32 = 0.0;
            
            const y_start = if (y >= half_window) y - half_window else 0;
            const y_end = @min(y + half_window + 1, src.height);
            const x_start = if (x >= half_window) x - half_window else 0;
            const x_end = @min(x + half_window + 1, src.width);
            
            var wy = y_start;
            while (wy < y_end) : (wy += 1) {
                var wx = x_start;
                while (wx < x_end) : (wx += 1) {
                    const dx = @as(i32, @intCast(wx)) - @as(i32, @intCast(x));
                    const dy = @as(i32, @intCast(wy)) - @as(i32, @intCast(y));
                    const weight = gaussianWeight(dx, dy, sigma);
                    
                    weighted_sum += @as(f32, @floatFromInt(src.getPixel(wx, wy))) * weight;
                    weight_sum += weight;
                }
            }
            
            const mean = weighted_sum / weight_sum;
            const threshold = mean - @as(f32, @floatFromInt(constant));
            const pixel = src.getPixel(x, y);
            
            dst.setPixel(x, y, if (@as(f32, @floatFromInt(pixel)) > threshold) 255 else 0);
        }
    }
    
    return dst;
}

/// Sauvola binarization (good for document images)
pub fn sauvolaThreshold(
    allocator: Allocator,
    src: *const Image,
    window_size: u32,
    k: f32,
    r: f32,
) !Image {
    var dst = try Image.init(allocator, src.width, src.height);
    errdefer dst.deinit();
    
    const half_window = window_size / 2;
    
    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            // Calculate mean and standard deviation in window
            var sum: f32 = 0.0;
            var sum_sq: f32 = 0.0;
            var count: f32 = 0.0;
            
            const y_start = if (y >= half_window) y - half_window else 0;
            const y_end = @min(y + half_window + 1, src.height);
            const x_start = if (x >= half_window) x - half_window else 0;
            const x_end = @min(x + half_window + 1, src.width);
            
            var wy = y_start;
            while (wy < y_end) : (wy += 1) {
                var wx = x_start;
                while (wx < x_end) : (wx += 1) {
                    const val = @as(f32, @floatFromInt(src.getPixel(wx, wy)));
                    sum += val;
                    sum_sq += val * val;
                    count += 1.0;
                }
            }
            
            const mean = sum / count;
            const variance = (sum_sq / count) - (mean * mean);
            const std_dev = @sqrt(@max(variance, 0.0));
            
            // Sauvola formula: T = mean * (1 + k * ((std_dev / r) - 1))
            const threshold = mean * (1.0 + k * ((std_dev / r) - 1.0));
            const pixel = @as(f32, @floatFromInt(src.getPixel(x, y)));
            
            dst.setPixel(x, y, if (pixel > threshold) 255 else 0);
        }
    }
    
    return dst;
}

/// Niblack thresholding method
pub fn niblackThreshold(
    allocator: Allocator,
    src: *const Image,
    window_size: u32,
    k: f32,
) !Image {
    var dst = try Image.init(allocator, src.width, src.height);
    errdefer dst.deinit();
    
    const half_window = window_size / 2;
    
    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            // Calculate mean and standard deviation in window
            var sum: f32 = 0.0;
            var sum_sq: f32 = 0.0;
            var count: f32 = 0.0;
            
            const y_start = if (y >= half_window) y - half_window else 0;
            const y_end = @min(y + half_window + 1, src.height);
            const x_start = if (x >= half_window) x - half_window else 0;
            const x_end = @min(x + half_window + 1, src.width);
            
            var wy = y_start;
            while (wy < y_end) : (wy += 1) {
                var wx = x_start;
                while (wx < x_end) : (wx += 1) {
                    const val = @as(f32, @floatFromInt(src.getPixel(wx, wy)));
                    sum += val;
                    sum_sq += val * val;
                    count += 1.0;
                }
            }
            
            const mean = sum / count;
            const variance = (sum_sq / count) - (mean * mean);
            const std_dev = @sqrt(@max(variance, 0.0));
            
            // Niblack formula: T = mean + k * std_dev
            const threshold = mean + k * std_dev;
            const pixel = @as(f32, @floatFromInt(src.getPixel(x, y)));
            
            dst.setPixel(x, y, if (pixel > threshold) 255 else 0);
        }
    }
    
    return dst;
}

/// Bradley adaptive thresholding (fast integral image method)
pub fn bradleyThreshold(
    allocator: Allocator,
    src: *const Image,
    window_size: u32,
    t: f32,
) !Image {
    // Build integral image
    const integral = try buildIntegralImage(allocator, src);
    defer allocator.free(integral);
    
    var dst = try Image.init(allocator, src.width, src.height);
    errdefer dst.deinit();
    
    const s = window_size / 2;
    
    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const x1 = if (x >= s) x - s else 0;
            const y1 = if (y >= s) y - s else 0;
            const x2 = @min(x + s, src.width - 1);
            const y2 = @min(y + s, src.height - 1);
            
            const count = (x2 - x1) * (y2 - y1);
            const sum = getIntegralSum(integral, src.width, x1, y1, x2, y2);
            const mean = sum / @as(f32, @floatFromInt(count));
            
            const pixel = @as(f32, @floatFromInt(src.getPixel(x, y)));
            const threshold = mean * (1.0 - t);
            
            dst.setPixel(x, y, if (pixel > threshold) 255 else 0);
        }
    }
    
    return dst;
}

/// Build integral image for fast sum calculation
fn buildIntegralImage(allocator: Allocator, img: *const Image) ![]u32 {
    const integral = try allocator.alloc(u32, img.width * img.height);
    
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const idx = y * img.width + x;
            var sum: u32 = img.getPixel(x, y);
            
            if (x > 0) sum += integral[idx - 1];
            if (y > 0) sum += integral[idx - img.width];
            if (x > 0 and y > 0) sum -= integral[idx - img.width - 1];
            
            integral[idx] = sum;
        }
    }
    
    return integral;
}

/// Get sum of rectangle using integral image
fn getIntegralSum(integral: []const u32, width: u32, x1: u32, y1: u32, x2: u32, y2: u32) f32 {
    const idx_d = y2 * width + x2;
    var sum = @as(f32, @floatFromInt(integral[idx_d]));
    
    if (x1 > 0) {
        const idx_c = y2 * width + (x1 - 1);
        sum -= @as(f32, @floatFromInt(integral[idx_c]));
    }
    
    if (y1 > 0) {
        const idx_b = (y1 - 1) * width + x2;
        sum -= @as(f32, @floatFromInt(integral[idx_b]));
    }
    
    if (x1 > 0 and y1 > 0) {
        const idx_a = (y1 - 1) * width + (x1 - 1);
        sum += @as(f32, @floatFromInt(integral[idx_a]));
    }
    
    return sum;
}

/// Hysteresis thresholding (for edge detection, used in Canny)
pub fn hysteresisThreshold(
    allocator: Allocator,
    src: *const Image,
    low_threshold: u8,
    high_threshold: u8,
) !Image {
    var dst = try Image.init(allocator, src.width, src.height);
    errdefer dst.deinit();
    
    // First pass: mark strong edges and candidates
    var i: usize = 0;
    while (i < src.data.len) : (i += 1) {
        if (src.data[i] >= high_threshold) {
            dst.data[i] = 255; // Strong edge
        } else if (src.data[i] >= low_threshold) {
            dst.data[i] = 128; // Weak edge (candidate)
        } else {
            dst.data[i] = 0; // Not an edge
        }
    }
    
    // Second pass: connect weak edges to strong edges
    var changed = true;
    while (changed) {
        changed = false;
        
        var y: u32 = 1;
        while (y < src.height - 1) : (y += 1) {
            var x: u32 = 1;
            while (x < src.width - 1) : (x += 1) {
                if (dst.getPixel(x, y) == 128) {
                    // Check if connected to strong edge
                    var has_strong_neighbor = false;
                    
                    var dy: i32 = -1;
                    while (dy <= 1) : (dy += 1) {
                        var dx: i32 = -1;
                        while (dx <= 1) : (dx += 1) {
                            if (dx == 0 and dy == 0) continue;
                            
                            const nx = @as(u32, @intCast(@as(i32, @intCast(x)) + dx));
                            const ny = @as(u32, @intCast(@as(i32, @intCast(y)) + dy));
                            
                            if (dst.getPixel(nx, ny) == 255) {
                                has_strong_neighbor = true;
                                break;
                            }
                        }
                        if (has_strong_neighbor) break;
                    }
                    
                    if (has_strong_neighbor) {
                        dst.setPixel(x, y, 255);
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Final pass: remove remaining weak edges
    i = 0;
    while (i < dst.data.len) : (i += 1) {
        if (dst.data[i] == 128) {
            dst.data[i] = 0;
        }
    }
    
    return dst;
}

// Export C-compatible functions
export fn nExtract_Threshold_Global(
    data: [*]const u8,
    width: u32,
    height: u32,
    threshold: u8,
) callconv(.C) ?*Image {
    const allocator = std.heap.c_allocator;
    
    const src_image = Image{
        .width = width,
        .height = height,
        .data = @constCast(data[0 .. width * height]),
        .allocator = allocator,
    };
    
    var result = globalThreshold(allocator, &src_image, threshold) catch return null;
    const result_ptr = allocator.create(Image) catch return null;
    result_ptr.* = result;
    return result_ptr;
}

export fn nExtract_Threshold_Otsu(
    data: [*]const u8,
    width: u32,
    height: u32,
) callconv(.C) ?*Image {
    const allocator = std.heap.c_allocator;
    
    const src_image = Image{
        .width = width,
        .height = height,
        .data = @constCast(data[0 .. width * height]),
        .allocator = allocator,
    };
    
    var result = otsuThreshold(allocator, &src_image) catch return null;
    const result_ptr = allocator.create(Image) catch return null;
    result_ptr.* = result;
    return result_ptr;
}

export fn nExtract_Threshold_Adaptive(
    data: [*]const u8,
    width: u32,
    height: u32,
    window_size: u32,
    constant: i32,
    method: u32,
) callconv(.C) ?*Image {
    const allocator = std.heap.c_allocator;
    
    const src_image = Image{
        .width = width,
        .height = height,
        .data = @constCast(data[0 .. width * height]),
        .allocator = allocator,
    };
    
    var result = if (method == 0)
        adaptiveThresholdMean(allocator, &src_image, window_size, constant) catch return null
    else
        adaptiveThresholdGaussian(allocator, &src_image, window_size, constant) catch return null;
        
    const result_ptr = allocator.create(Image) catch return null;
    result_ptr.* = result;
    return result_ptr;
}

export fn nExtract_Threshold_Sauvola(
    data: [*]const u8,
    width: u32,
    height: u32,
    window_size: u32,
    k: f32,
    r: f32,
) callconv(.C) ?*Image {
    const allocator = std.heap.c_allocator;
    
    const src_image = Image{
        .width = width,
        .height = height,
        .data = @constCast(data[0 .. width * height]),
        .allocator = allocator,
    };
    
    var result = sauvolaThreshold(allocator, &src_image, window_size, k, r) catch return null;
    const result_ptr = allocator.create(Image) catch return null;
    result_ptr.* = result;
    return result_ptr;
}

export fn nExtract_Image_Free_Threshold(img: ?*Image) callconv(.C) void {
    if (img) |image| {
        image.deinit();
        std.heap.c_allocator.destroy(image);
    }
}
