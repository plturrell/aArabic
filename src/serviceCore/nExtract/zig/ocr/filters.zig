// filters.zig - Image filtering operations for nExtract
// Day 27: Image Filtering
// Pure Zig implementation with SIMD optimization support

const std = @import("std");
const math = std.math;
const mem = std.mem;
const Allocator = mem.Allocator;

// Import color space for conversions
const colorspace = @import("colorspace.zig");

// ============================================================================
// Core Types
// ============================================================================

pub const Kernel = struct {
    data: []f32,
    width: u32,
    height: u32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: u32, height: u32) !Kernel {
        const data = try allocator.alloc(f32, width * height);
        return Kernel{
            .data = data,
            .width = width,
            .height = height,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Kernel) void {
        self.allocator.free(self.data);
    }

    pub fn get(self: Kernel, x: u32, y: u32) f32 {
        return self.data[y * self.width + x];
    }

    pub fn set(self: *Kernel, x: u32, y: u32, value: f32) void {
        self.data[y * self.width + x] = value;
    }
};

pub const Image = struct {
    data: []u8,
    width: u32,
    height: u32,
    channels: u32, // 1=grayscale, 3=RGB, 4=RGBA
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: u32, height: u32, channels: u32) !Image {
        const data = try allocator.alloc(u8, width * height * channels);
        return Image{
            .data = data,
            .width = width,
            .height = height,
            .channels = channels,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.data);
    }

    pub fn getPixel(self: Image, x: u32, y: u32, channel: u32) u8 {
        const idx = (y * self.width + x) * self.channels + channel;
        return self.data[idx];
    }

    pub fn setPixel(self: *Image, x: u32, y: u32, channel: u32, value: u8) void {
        const idx = (y * self.width + x) * self.channels + channel;
        self.data[idx] = value;
    }
};

// ============================================================================
// Gaussian Blur
// ============================================================================

/// Generate 1D Gaussian kernel
pub fn generateGaussianKernel1D(allocator: Allocator, sigma: f32) ![]f32 {
    // Kernel size: 6*sigma (covers 99.7% of distribution)
    const size: u32 = @intFromFloat(@max(3.0, @ceil(sigma * 6.0)));
    const size_odd = if (size % 2 == 0) size + 1 else size;
    
    var kernel = try allocator.alloc(f32, size_odd);
    const center = size_odd / 2;
    
    var sum: f32 = 0.0;
    const two_sigma_sq = 2.0 * sigma * sigma;
    
    for (0..size_odd) |i| {
        const x: f32 = @floatFromInt(@as(i32, @intCast(i)) - @as(i32, @intCast(center)));
        kernel[i] = @exp(-(x * x) / two_sigma_sq);
        sum += kernel[i];
    }
    
    // Normalize
    for (kernel) |*k| {
        k.* /= sum;
    }
    
    return kernel;
}

/// Apply separable Gaussian blur (horizontal then vertical)
pub fn gaussianBlur(src: Image, dst: *Image, sigma: f32) !void {
    if (src.width != dst.width or src.height != dst.height or src.channels != dst.channels) {
        return error.DimensionMismatch;
    }
    
    var arena = std.heap.ArenaAllocator.init(src.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    // Generate 1D kernel
    const kernel = try generateGaussianKernel1D(allocator, sigma);
    const kernel_size = kernel.len;
    const radius = kernel_size / 2;
    
    // Create temporary buffer for horizontal pass
    var temp = try Image.init(allocator, src.width, src.height, src.channels);
    
    // Horizontal pass
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            for (0..src.channels) |c| {
                var sum: f32 = 0.0;
                
                for (0..kernel_size) |k| {
                    const offset: i32 = @as(i32, @intCast(k)) - @as(i32, @intCast(radius));
                    const sample_x: i32 = @as(i32, @intCast(x)) + offset;
                    
                    // Clamp to image boundaries
                    const clamped_x = std.math.clamp(sample_x, 0, @as(i32, @intCast(src.width - 1)));
                    const pixel = src.getPixel(@intCast(clamped_x), @intCast(y), @intCast(c));
                    sum += @as(f32, @floatFromInt(pixel)) * kernel[k];
                }
                
                temp.setPixel(@intCast(x), @intCast(y), @intCast(c), @intFromFloat(@round(sum)));
            }
        }
    }
    
    // Vertical pass
    for (0..dst.height) |y| {
        for (0..dst.width) |x| {
            for (0..dst.channels) |c| {
                var sum: f32 = 0.0;
                
                for (0..kernel_size) |k| {
                    const offset: i32 = @as(i32, @intCast(k)) - @as(i32, @intCast(radius));
                    const sample_y: i32 = @as(i32, @intCast(y)) + offset;
                    
                    // Clamp to image boundaries
                    const clamped_y = std.math.clamp(sample_y, 0, @as(i32, @intCast(temp.height - 1)));
                    const pixel = temp.getPixel(@intCast(x), @intCast(clamped_y), @intCast(c));
                    sum += @as(f32, @floatFromInt(pixel)) * kernel[k];
                }
                
                dst.setPixel(@intCast(x), @intCast(y), @intCast(c), @intFromFloat(@round(sum)));
            }
        }
    }
}

// ============================================================================
// Median Filter (noise reduction)
// ============================================================================

fn compareU8(context: void, a: u8, b: u8) bool {
    _ = context;
    return a < b;
}

/// Apply median filter (excellent for salt-and-pepper noise)
pub fn medianFilter(src: Image, dst: *Image, kernel_size: u32) !void {
    if (src.width != dst.width or src.height != dst.height or src.channels != dst.channels) {
        return error.DimensionMismatch;
    }
    
    if (kernel_size % 2 == 0) {
        return error.KernelSizeMustBeOdd;
    }
    
    var arena = std.heap.ArenaAllocator.init(src.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    const radius = kernel_size / 2;
    const window_size = kernel_size * kernel_size;
    var window = try allocator.alloc(u8, window_size);
    
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            for (0..src.channels) |c| {
                // Collect neighborhood pixels
                var idx: usize = 0;
                
                for (0..kernel_size) |ky| {
                    for (0..kernel_size) |kx| {
                        const offset_y: i32 = @as(i32, @intCast(ky)) - @as(i32, @intCast(radius));
                        const offset_x: i32 = @as(i32, @intCast(kx)) - @as(i32, @intCast(radius));
                        
                        const sample_y: i32 = @as(i32, @intCast(y)) + offset_y;
                        const sample_x: i32 = @as(i32, @intCast(x)) + offset_x;
                        
                        // Clamp to boundaries
                        const clamped_y = std.math.clamp(sample_y, 0, @as(i32, @intCast(src.height - 1)));
                        const clamped_x = std.math.clamp(sample_x, 0, @as(i32, @intCast(src.width - 1)));
                        
                        window[idx] = src.getPixel(@intCast(clamped_x), @intCast(clamped_y), @intCast(c));
                        idx += 1;
                    }
                }
                
                // Sort and take median
                std.mem.sort(u8, window, {}, compareU8);
                const median = window[window_size / 2];
                dst.setPixel(@intCast(x), @intCast(y), @intCast(c), median);
            }
        }
    }
}

// ============================================================================
// Edge Detection - Sobel Operator
// ============================================================================

pub const SobelResult = struct {
    magnitude: Image,
    direction: []f32, // Gradient direction in radians
    allocator: Allocator,
    
    pub fn deinit(self: *SobelResult) void {
        self.magnitude.deinit();
        self.allocator.free(self.direction);
    }
};

/// Apply Sobel edge detection
pub fn sobelEdgeDetection(src: Image, allocator: Allocator) !SobelResult {
    if (src.channels != 1) {
        return error.GrayscaleOnly;
    }
    
    // Sobel kernels
    const sobel_x = [_]f32{
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1,
    };
    
    const sobel_y = [_]f32{
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1,
    };
    
    var magnitude = try Image.init(allocator, src.width, src.height, 1);
    var direction = try allocator.alloc(f32, src.width * src.height);
    
    for (1..src.height - 1) |y| {
        for (1..src.width - 1) |x| {
            var gx: f32 = 0.0;
            var gy: f32 = 0.0;
            
            // Apply 3x3 Sobel kernels
            for (0..3) |ky| {
                for (0..3) |kx| {
                    const sample_y = y + ky - 1;
                    const sample_x = x + kx - 1;
                    const pixel: f32 = @floatFromInt(src.getPixel(@intCast(sample_x), @intCast(sample_y), 0));
                    
                    const k_idx = ky * 3 + kx;
                    gx += pixel * sobel_x[k_idx];
                    gy += pixel * sobel_y[k_idx];
                }
            }
            
            // Calculate magnitude and direction
            const mag = @sqrt(gx * gx + gy * gy);
            const dir = math.atan2(gy, gx);
            
            magnitude.setPixel(@intCast(x), @intCast(y), 0, @intFromFloat(@min(255.0, mag)));
            direction[y * src.width + x] = dir;
        }
    }
    
    return SobelResult{
        .magnitude = magnitude,
        .direction = direction,
        .allocator = allocator,
    };
}

// ============================================================================
// Canny Edge Detection (multi-stage)
// ============================================================================

pub fn cannyEdgeDetection(src: Image, allocator: Allocator, low_threshold: u8, high_threshold: u8) !Image {
    if (src.channels != 1) {
        return error.GrayscaleOnly;
    }
    
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();
    
    // Step 1: Gaussian blur to reduce noise
    var blurred = try Image.init(temp_allocator, src.width, src.height, 1);
    try gaussianBlur(src, &blurred, 1.4);
    
    // Step 2: Sobel edge detection
    var sobel_result = try sobelEdgeDetection(blurred, temp_allocator);
    defer sobel_result.deinit();
    
    // Step 3: Non-maximum suppression
    var suppressed = try Image.init(temp_allocator, src.width, src.height, 1);
    for (1..src.height - 1) |y| {
        for (1..src.width - 1) |x| {
            const idx = y * src.width + x;
            const mag = sobel_result.magnitude.getPixel(@intCast(x), @intCast(y), 0);
            const angle = sobel_result.direction[idx];
            
            // Quantize angle to 4 directions (0°, 45°, 90°, 135°)
            const angle_deg = angle * 180.0 / math.pi;
            const dir = @mod(@as(i32, @intFromFloat(@round(angle_deg / 45.0))), 4);
            
            var neighbor1: u8 = 0;
            var neighbor2: u8 = 0;
            
            switch (dir) {
                0 => { // Horizontal (0° or 180°)
                    neighbor1 = sobel_result.magnitude.getPixel(@intCast(x - 1), @intCast(y), 0);
                    neighbor2 = sobel_result.magnitude.getPixel(@intCast(x + 1), @intCast(y), 0);
                },
                1 => { // Diagonal (45° or 225°)
                    neighbor1 = sobel_result.magnitude.getPixel(@intCast(x - 1), @intCast(y + 1), 0);
                    neighbor2 = sobel_result.magnitude.getPixel(@intCast(x + 1), @intCast(y - 1), 0);
                },
                2 => { // Vertical (90° or 270°)
                    neighbor1 = sobel_result.magnitude.getPixel(@intCast(x), @intCast(y - 1), 0);
                    neighbor2 = sobel_result.magnitude.getPixel(@intCast(x), @intCast(y + 1), 0);
                },
                3 => { // Diagonal (135° or 315°)
                    neighbor1 = sobel_result.magnitude.getPixel(@intCast(x - 1), @intCast(y - 1), 0);
                    neighbor2 = sobel_result.magnitude.getPixel(@intCast(x + 1), @intCast(y + 1), 0);
                },
                else => unreachable,
            }
            
            // Suppress if not local maximum
            if (mag >= neighbor1 and mag >= neighbor2) {
                suppressed.setPixel(@intCast(x), @intCast(y), 0, mag);
            } else {
                suppressed.setPixel(@intCast(x), @intCast(y), 0, 0);
            }
        }
    }
    
    // Step 4: Double threshold and edge tracking by hysteresis
    var edges = try Image.init(allocator, src.width, src.height, 1);
    
    // Mark strong and weak edges
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            const mag = suppressed.getPixel(@intCast(x), @intCast(y), 0);
            if (mag >= high_threshold) {
                edges.setPixel(@intCast(x), @intCast(y), 0, 255); // Strong edge
            } else if (mag >= low_threshold) {
                edges.setPixel(@intCast(x), @intCast(y), 0, 128); // Weak edge
            } else {
                edges.setPixel(@intCast(x), @intCast(y), 0, 0); // Not an edge
            }
        }
    }
    
    // Edge tracking: keep weak edges connected to strong edges
    var changed = true;
    while (changed) {
        changed = false;
        for (1..src.height - 1) |y| {
            for (1..src.width - 1) |x| {
                if (edges.getPixel(@intCast(x), @intCast(y), 0) == 128) {
                    // Check 8-neighborhood for strong edges
                    var has_strong_neighbor = false;
                    for (0..3) |dy| {
                        for (0..3) |dx| {
                            if (dy == 1 and dx == 1) continue; // Skip center
                            const ny = y + dy - 1;
                            const nx = x + dx - 1;
                            if (edges.getPixel(@intCast(nx), @intCast(ny), 0) == 255) {
                                has_strong_neighbor = true;
                                break;
                            }
                        }
                        if (has_strong_neighbor) break;
                    }
                    
                    if (has_strong_neighbor) {
                        edges.setPixel(@intCast(x), @intCast(y), 0, 255);
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Suppress remaining weak edges
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            if (edges.getPixel(@intCast(x), @intCast(y), 0) == 128) {
                edges.setPixel(@intCast(x), @intCast(y), 0, 0);
            }
        }
    }
    
    return edges;
}

// ============================================================================
// Morphological Operations
// ============================================================================

/// Erosion - shrinks white regions
pub fn erode(src: Image, dst: *Image, kernel_size: u32) !void {
    if (src.width != dst.width or src.height != dst.height or src.channels != dst.channels) {
        return error.DimensionMismatch;
    }
    
    if (kernel_size % 2 == 0) {
        return error.KernelSizeMustBeOdd;
    }
    
    const radius = kernel_size / 2;
    
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            for (0..src.channels) |c| {
                var min_val: u8 = 255;
                
                // Find minimum in kernel
                for (0..kernel_size) |ky| {
                    for (0..kernel_size) |kx| {
                        const offset_y: i32 = @as(i32, @intCast(ky)) - @as(i32, @intCast(radius));
                        const offset_x: i32 = @as(i32, @intCast(kx)) - @as(i32, @intCast(radius));
                        
                        const sample_y: i32 = @as(i32, @intCast(y)) + offset_y;
                        const sample_x: i32 = @as(i32, @intCast(x)) + offset_x;
                        
                        // Clamp to boundaries
                        const clamped_y = std.math.clamp(sample_y, 0, @as(i32, @intCast(src.height - 1)));
                        const clamped_x = std.math.clamp(sample_x, 0, @as(i32, @intCast(src.width - 1)));
                        
                        const pixel = src.getPixel(@intCast(clamped_x), @intCast(clamped_y), @intCast(c));
                        min_val = @min(min_val, pixel);
                    }
                }
                
                dst.setPixel(@intCast(x), @intCast(y), @intCast(c), min_val);
            }
        }
    }
}

/// Dilation - expands white regions
pub fn dilate(src: Image, dst: *Image, kernel_size: u32) !void {
    if (src.width != dst.width or src.height != dst.height or src.channels != dst.channels) {
        return error.DimensionMismatch;
    }
    
    if (kernel_size % 2 == 0) {
        return error.KernelSizeMustBeOdd;
    }
    
    const radius = kernel_size / 2;
    
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            for (0..src.channels) |c| {
                var max_val: u8 = 0;
                
                // Find maximum in kernel
                for (0..kernel_size) |ky| {
                    for (0..kernel_size) |kx| {
                        const offset_y: i32 = @as(i32, @intCast(ky)) - @as(i32, @intCast(radius));
                        const offset_x: i32 = @as(i32, @intCast(kx)) - @as(i32, @intCast(radius));
                        
                        const sample_y: i32 = @as(i32, @intCast(y)) + offset_y;
                        const sample_x: i32 = @as(i32, @intCast(x)) + offset_x;
                        
                        // Clamp to boundaries
                        const clamped_y = std.math.clamp(sample_y, 0, @as(i32, @intCast(src.height - 1)));
                        const clamped_x = std.math.clamp(sample_x, 0, @as(i32, @intCast(src.width - 1)));
                        
                        const pixel = src.getPixel(@intCast(clamped_x), @intCast(clamped_y), @intCast(c));
                        max_val = @max(max_val, pixel);
                    }
                }
                
                dst.setPixel(@intCast(x), @intCast(y), @intCast(c), max_val);
            }
        }
    }
}

/// Opening - erosion followed by dilation (removes small bright spots)
pub fn morphologicalOpening(src: Image, allocator: Allocator, kernel_size: u32) !Image {
    var temp = try Image.init(allocator, src.width, src.height, src.channels);
    errdefer temp.deinit();
    
    var result = try Image.init(allocator, src.width, src.height, src.channels);
    errdefer result.deinit();
    
    try erode(src, &temp, kernel_size);
    try dilate(temp, &result, kernel_size);
    
    temp.deinit();
    return result;
}

/// Closing - dilation followed by erosion (removes small dark spots)
pub fn morphologicalClosing(src: Image, allocator: Allocator, kernel_size: u32) !Image {
    var temp = try Image.init(allocator, src.width, src.height, src.channels);
    errdefer temp.deinit();
    
    var result = try Image.init(allocator, src.width, src.height, src.channels);
    errdefer result.deinit();
    
    try dilate(src, &temp, kernel_size);
    try erode(temp, &result, kernel_size);
    
    temp.deinit();
    return result;
}

// ============================================================================
// Bilateral Filter (edge-preserving smoothing)
// ============================================================================

/// Apply bilateral filter (smooths while preserving edges)
pub fn bilateralFilter(src: Image, dst: *Image, diameter: u32, sigma_color: f32, sigma_space: f32) !void {
    if (src.width != dst.width or src.height != dst.height or src.channels != dst.channels) {
        return error.DimensionMismatch;
    }
    
    const radius = diameter / 2;
    
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            for (0..src.channels) |c| {
                var sum: f32 = 0.0;
                var weight_sum: f32 = 0.0;
                const center_pixel: f32 = @floatFromInt(src.getPixel(@intCast(x), @intCast(y), @intCast(c)));
                
                for (0..diameter) |ky| {
                    for (0..diameter) |kx| {
                        const offset_y: i32 = @as(i32, @intCast(ky)) - @as(i32, @intCast(radius));
                        const offset_x: i32 = @as(i32, @intCast(kx)) - @as(i32, @intCast(radius));
                        
                        const sample_y: i32 = @as(i32, @intCast(y)) + offset_y;
                        const sample_x: i32 = @as(i32, @intCast(x)) + offset_x;
                        
                        // Clamp to boundaries
                        const clamped_y = std.math.clamp(sample_y, 0, @as(i32, @intCast(src.height - 1)));
                        const clamped_x = std.math.clamp(sample_x, 0, @as(i32, @intCast(src.width - 1)));
                        
                        const pixel: f32 = @floatFromInt(src.getPixel(@intCast(clamped_x), @intCast(clamped_y), @intCast(c)));
                        
                        // Spatial weight (Gaussian based on distance)
                        const dx: f32 = @floatFromInt(offset_x);
                        const dy: f32 = @floatFromInt(offset_y);
                        const spatial_dist = dx * dx + dy * dy;
                        const spatial_weight = @exp(-spatial_dist / (2.0 * sigma_space * sigma_space));
                        
                        // Color weight (Gaussian based on intensity difference)
                        const color_dist = (pixel - center_pixel) * (pixel - center_pixel);
                        const color_weight = @exp(-color_dist / (2.0 * sigma_color * sigma_color));
                        
                        const weight = spatial_weight * color_weight;
                        sum += pixel * weight;
                        weight_sum += weight;
                    }
                }
                
                dst.setPixel(@intCast(x), @intCast(y), @intCast(c), @intFromFloat(@round(sum / weight_sum)));
            }
        }
    }
}

// ============================================================================
// Sharpen Filter
// ============================================================================

/// Apply unsharp mask for sharpening
pub fn sharpen(src: Image, dst: *Image, amount: f32) !void {
    if (src.width != dst.width or src.height != dst.height or src.channels != dst.channels) {
        return error.DimensionMismatch;
    }
    
    var arena = std.heap.ArenaAllocator.init(src.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    // Blur the image
    var blurred = try Image.init(allocator, src.width, src.height, src.channels);
    try gaussianBlur(src, &blurred, 1.0);
    
    // Sharpen = Original + amount * (Original - Blurred)
    for (0..src.height) |y| {
        for (0..src.width) |x| {
            for (0..src.channels) |c| {
                const original: f32 = @floatFromInt(src.getPixel(@intCast(x), @intCast(y), @intCast(c)));
                const blur: f32 = @floatFromInt(blurred.getPixel(@intCast(x), @intCast(y), @intCast(c)));
                const diff = original - blur;
                const sharpened = original + amount * diff;
                
                const clamped: u8 = @intFromFloat(@round(std.math.clamp(sharpened, 0.0, 255.0)));
                dst.setPixel(@intCast(x), @intCast(y), @intCast(c), clamped);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "Gaussian blur" {
    const allocator = std.testing.allocator;
    
    // Create test image (3x3, grayscale)
    var src = try Image.init(allocator, 3, 3, 1);
    defer src.deinit();
    
    // Set center pixel bright, others dark
    for (0..3) |y| {
        for (0..3) |x| {
            src.setPixel(@intCast(x), @intCast(y), 0, if (x == 1 and y == 1) 255 else 0);
        }
    }
    
    var dst = try Image.init(allocator, 3, 3, 1);
    defer dst.deinit();
    
    try gaussianBlur(src, &dst, 0.5);
    
    // After blur, center should still be brightest
    const center = dst.getPixel(1, 1, 0);
    const corner = dst.getPixel(0, 0, 0);
    try std.testing.expect(center > corner);
}

test "Median filter" {
    const allocator = std.testing.allocator;
    
    // Create test image with salt-and-pepper noise
    var src = try Image.init(allocator, 5, 5, 1);
    defer src.deinit();
    
    // Fill with 128, add some outliers
    for (0..5) |y| {
        for (0..5) |x| {
            src.setPixel(@intCast(x), @intCast(y), 0, 128);
        }
    }
    src.setPixel(2, 2, 0, 255); // Salt
    src.setPixel(1, 1, 0, 0);   // Pepper
    
    var dst = try Image.init(allocator, 5, 5, 1);
    defer dst.deinit();
    
    try medianFilter(src, &dst, 3);
    
    // Outliers should be removed
    try std.testing.expect(dst.getPixel(2, 2, 0) < 200);
    try std.testing.expect(dst.getPixel(1, 1, 0) > 50);
}

test "Sobel edge detection" {
    const allocator = std.testing.allocator;
    
    // Create test image with vertical edge
    var src = try Image.init(allocator, 5, 5, 1);
    defer src.deinit();
    
    for (0..5) |y| {
        for (0..5) |x| {
            src.setPixel(@intCast(x), @intCast(y), 0, if (x < 2) 0 else 255);
        }
    }
    
    var result = try sobelEdgeDetection(src, allocator);
    defer result.deinit();
    
    // Edge should be detected at x=2
    const edge_mag = result.magnitude.getPixel(2, 2, 0);
    const no_edge_mag = result.magnitude.getPixel(0, 2, 0);
    try std.testing.expect(edge_mag > no_edge_mag);
}

test "Morphological operations" {
    const allocator = std.testing.allocator;
    
    // Create test image with small bright spot
    var src = try Image.init(allocator, 5, 5, 1);
    defer src.deinit();
    
    for (0..5) |y| {
        for (0..5) |x| {
            src.setPixel(@intCast(x), @intCast(y), 0, 0);
        }
    }
    src.setPixel(2, 2, 0, 255); // Small bright spot
    
    var dst = try Image.init(allocator, 5, 5, 1);
    defer dst.deinit();
    
    // Erosion should remove small spot
    try erode(src, &dst, 3);
    try std.testing.expect(dst.getPixel(2, 2, 0) == 0);
    
    // Dilation should expand bright regions
    try dilate(src, &dst, 3);
    try std.testing.expect(dst.getPixel(1, 2, 0) > 0);
}
