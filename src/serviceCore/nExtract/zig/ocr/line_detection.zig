const std = @import("std");
const Image = @import("colorspace.zig").Image;
const transform = @import("transform.zig");

// ============================================================================
// Connected Component Analysis (CCA)
// ============================================================================

pub const ConnectivityType = enum {
    Four,  // 4-connectivity (N, S, E, W)
    Eight, // 8-connectivity (includes diagonals)
};

pub const Component = struct {
    label: u32,
    bbox: BoundingBox,
    pixel_count: u32,
    
    pub fn init(label: u32) Component {
        return Component{
            .label = label,
            .bbox = BoundingBox{
                .min_x = std.math.maxInt(u32),
                .min_y = std.math.maxInt(u32),
                .max_x = 0,
                .max_y = 0,
            },
            .pixel_count = 0,
        };
    }
    
    pub fn updateBounds(self: *Component, x: u32, y: u32) void {
        self.bbox.min_x = @min(self.bbox.min_x, x);
        self.bbox.min_y = @min(self.bbox.min_y, y);
        self.bbox.max_x = @max(self.bbox.max_x, x);
        self.bbox.max_y = @max(self.bbox.max_y, y);
        self.pixel_count += 1;
    }
    
    pub fn width(self: *const Component) u32 {
        if (self.bbox.max_x >= self.bbox.min_x) {
            return self.bbox.max_x - self.bbox.min_x + 1;
        }
        return 0;
    }
    
    pub fn height(self: *const Component) u32 {
        if (self.bbox.max_y >= self.bbox.min_y) {
            return self.bbox.max_y - self.bbox.min_y + 1;
        }
        return 0;
    }
};

pub const BoundingBox = struct {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    
    pub fn center_x(self: *const BoundingBox) f32 {
        return @as(f32, @floatFromInt(self.min_x + self.max_x)) / 2.0;
    }
    
    pub fn center_y(self: *const BoundingBox) f32 {
        return @as(f32, @floatFromInt(self.min_y + self.max_y)) / 2.0;
    }
    
    pub fn width(self: *const BoundingBox) u32 {
        if (self.max_x >= self.min_x) {
            return self.max_x - self.min_x + 1;
        }
        return 0;
    }
    
    pub fn height(self: *const BoundingBox) u32 {
        if (self.max_y >= self.min_y) {
            return self.max_y - self.min_y + 1;
        }
        return 0;
    }
};

/// Perform connected component analysis on binary image
/// Returns label image where each pixel's value is its component label
pub fn connectedComponentAnalysis(
    allocator: std.mem.Allocator,
    binary_image: *const Image,
    connectivity: ConnectivityType,
) !struct { labels: []u32, component_count: u32 } {
    const width = binary_image.width;
    const height = binary_image.height;
    
    // Allocate label array
    const labels = try allocator.alloc(u32, width * height);
    errdefer allocator.free(labels);
    @memset(labels, 0);
    
    var next_label: u32 = 1;
    var equivalences = std.ArrayList([2]u32).init(allocator);
    defer equivalences.deinit();
    
    // First pass: assign labels
    var y: u32 = 0;
    while (y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const pixel = binary_image.getPixel(x, y);
            
            // Only process foreground pixels (255)
            if (pixel != 255) continue;
            
            const idx = y * width + x;
            
            // Check neighbors
            var neighbor_labels = std.ArrayList(u32).init(allocator);
            defer neighbor_labels.deinit();
            
            // North
            if (y > 0) {
                const north_idx = (y - 1) * width + x;
                if (labels[north_idx] != 0) {
                    try neighbor_labels.append(labels[north_idx]);
                }
            }
            
            // West
            if (x > 0) {
                const west_idx = y * width + (x - 1);
                if (labels[west_idx] != 0) {
                    try neighbor_labels.append(labels[west_idx]);
                }
            }
            
            // 8-connectivity: NW and NE
            if (connectivity == .Eight) {
                // Northwest
                if (x > 0 and y > 0) {
                    const nw_idx = (y - 1) * width + (x - 1);
                    if (labels[nw_idx] != 0) {
                        try neighbor_labels.append(labels[nw_idx]);
                    }
                }
                
                // Northeast
                if (x < width - 1 and y > 0) {
                    const ne_idx = (y - 1) * width + (x + 1);
                    if (labels[ne_idx] != 0) {
                        try neighbor_labels.append(labels[ne_idx]);
                    }
                }
            }
            
            if (neighbor_labels.items.len == 0) {
                // No neighbors: assign new label
                labels[idx] = next_label;
                next_label += 1;
            } else {
                // Find minimum neighbor label
                var min_label: u32 = std.math.maxInt(u32);
                for (neighbor_labels.items) |label| {
                    min_label = @min(min_label, label);
                }
                labels[idx] = min_label;
                
                // Record equivalences
                for (neighbor_labels.items) |label| {
                    if (label != min_label) {
                        try equivalences.append(.{ min_label, label });
                    }
                }
            }
        }
    }
    
    // Build equivalence map
    var equiv_map = std.AutoHashMap(u32, u32).init(allocator);
    defer equiv_map.deinit();
    
    for (equivalences.items) |pair| {
        const min_val = pair[0];
        const max_val = pair[1];
        try equiv_map.put(max_val, min_val);
    }
    
    // Resolve equivalences transitively
    var changed = true;
    while (changed) {
        changed = false;
        var it = equiv_map.iterator();
        while (it.next()) |entry| {
            if (equiv_map.get(entry.value_ptr.*)) |new_val| {
                if (new_val != entry.value_ptr.*) {
                    entry.value_ptr.* = new_val;
                    changed = true;
                }
            }
        }
    }
    
    // Second pass: resolve labels
    for (labels, 0..) |*label, i| {
        if (label.* != 0) {
            if (equiv_map.get(label.*)) |resolved| {
                label.* = resolved;
            }
        }
    }
    
    // Count unique components
    var unique_labels = std.AutoHashMap(u32, void).init(allocator);
    defer unique_labels.deinit();
    
    for (labels) |label| {
        if (label != 0) {
            try unique_labels.put(label, {});
        }
    }
    
    return .{
        .labels = labels,
        .component_count = @as(u32, @intCast(unique_labels.count())),
    };
}

/// Extract components with bounding boxes
pub fn extractComponents(
    allocator: std.mem.Allocator,
    labels: []const u32,
    width: u32,
    height: u32,
) ![]Component {
    // Find all unique labels
    var label_map = std.AutoHashMap(u32, usize).init(allocator);
    defer label_map.deinit();
    
    var components = std.ArrayList(Component).init(allocator);
    
    for (labels, 0..) |label, i| {
        if (label == 0) continue;
        
        const x = @as(u32, @intCast(i % width));
        const y = @as(u32, @intCast(i / width));
        
        if (label_map.get(label)) |comp_idx| {
            components.items[comp_idx].updateBounds(x, y);
        } else {
            var comp = Component.init(label);
            comp.updateBounds(x, y);
            try components.append(comp);
            try label_map.put(label, components.items.len - 1);
        }
    }
    
    return components.toOwnedSlice();
}

/// Filter components by size (remove noise)
pub fn filterComponentsBySize(
    allocator: std.mem.Allocator,
    components: []const Component,
    min_width: u32,
    min_height: u32,
    min_pixel_count: u32,
) ![]Component {
    var filtered = std.ArrayList(Component).init(allocator);
    
    for (components) |comp| {
        const w = comp.width();
        const h = comp.height();
        if (w >= min_width and h >= min_height and comp.pixel_count >= min_pixel_count) {
            try filtered.append(comp);
        }
    }
    
    return filtered.toOwnedSlice();
}

// ============================================================================
// Projection Profiles
// ============================================================================

/// Calculate horizontal projection profile (sum of pixels per row)
pub fn horizontalProjectionProfile(
    allocator: std.mem.Allocator,
    binary_image: *const Image,
) ![]u32 {
    const profile = try allocator.alloc(u32, binary_image.height);
    @memset(profile, 0);
    
    var y: u32 = 0;
    while (y < binary_image.height) : (y += 1) {
        var count: u32 = 0;
        var x: u32 = 0;
        while (x < binary_image.width) : (x += 1) {
            if (binary_image.getPixel(x, y) == 255) {
                count += 1;
            }
        }
        profile[y] = count;
    }
    
    return profile;
}

/// Calculate vertical projection profile (sum of pixels per column)
pub fn verticalProjectionProfile(
    allocator: std.mem.Allocator,
    binary_image: *const Image,
) ![]u32 {
    const profile = try allocator.alloc(u32, binary_image.width);
    @memset(profile, 0);
    
    var x: u32 = 0;
    while (x < binary_image.width) : (x += 1) {
        var count: u32 = 0;
        var y: u32 = 0;
        while (y < binary_image.height) : (y += 1) {
            if (binary_image.getPixel(x, y) == 255) {
                count += 1;
            }
        }
        profile[x] = count;
    }
    
    return profile;
}

// ============================================================================
// Line Segmentation
// ============================================================================

pub const TextLine = struct {
    y_start: u32,
    y_end: u32,
    baseline: u32,
    
    pub fn height(self: *const TextLine) u32 {
        return self.y_end - self.y_start + 1;
    }
    
    pub fn center_y(self: *const TextLine) f32 {
        return @as(f32, @floatFromInt(self.y_start + self.y_end)) / 2.0;
    }
};

/// Segment image into text lines using horizontal projection profile
pub fn segmentLines(
    allocator: std.mem.Allocator,
    binary_image: *const Image,
    min_gap: u32,
    min_height: u32,
) ![]TextLine {
    const profile = try horizontalProjectionProfile(allocator, binary_image);
    defer allocator.free(profile);
    
    var lines = std.ArrayList(TextLine).init(allocator);
    
    var in_line = false;
    var line_start: u32 = 0;
    
    var y: u32 = 0;
    while (y < binary_image.height) : (y += 1) {
        const has_text = profile[y] > 0;
        
        if (has_text and !in_line) {
            // Start of new line
            line_start = y;
            in_line = true;
        } else if (!has_text and in_line) {
            // Potential end of line (check gap)
            var gap_size: u32 = 0;
            var gap_y = y;
            while (gap_y < binary_image.height and profile[gap_y] == 0) : (gap_y += 1) {
                gap_size += 1;
            }
            
            if (gap_size >= min_gap or gap_y >= binary_image.height) {
                // End of line
                const line_end = y - 1;
                const line_height = line_end - line_start + 1;
                
                if (line_height >= min_height) {
                    const baseline = line_end; // Approximate baseline at bottom
                    try lines.append(TextLine{
                        .y_start = line_start,
                        .y_end = line_end,
                        .baseline = baseline,
                    });
                }
                
                in_line = false;
            }
        }
    }
    
    // Handle case where line extends to bottom of image
    if (in_line) {
        const line_end = binary_image.height - 1;
        const line_height = line_end - line_start + 1;
        if (line_height >= min_height) {
            try lines.append(TextLine{
                .y_start = line_start,
                .y_end = line_end,
                .baseline = line_end,
            });
        }
    }
    
    return lines.toOwnedSlice();
}

// ============================================================================
// Skew Detection
// ============================================================================

/// Detect skew angle using projection profile method
pub fn detectSkewProjection(
    allocator: std.mem.Allocator,
    binary_image: *const Image,
    max_angle: f32,
    angle_step: f32,
) !f32 {
    var best_angle: f32 = 0.0;
    var max_variance: f32 = 0.0;
    
    var angle = -max_angle;
    while (angle <= max_angle) : (angle += angle_step) {
        // Rotate image
        var rotated = try transform.rotate(allocator, binary_image, angle, .Nearest);
        defer rotated.deinit();
        
        // Calculate horizontal projection profile
        const profile = try horizontalProjectionProfile(allocator, &rotated);
        defer allocator.free(profile);
        
        // Calculate variance (higher variance = better aligned)
        var sum: f64 = 0.0;
        var sum_sq: f64 = 0.0;
        for (profile) |val| {
            const f_val = @as(f64, @floatFromInt(val));
            sum += f_val;
            sum_sq += f_val * f_val;
        }
        
        const n = @as(f64, @floatFromInt(profile.len));
        const mean = sum / n;
        const variance = (sum_sq / n) - (mean * mean);
        
        if (variance > max_variance) {
            max_variance = variance;
            best_angle = angle;
        }
    }
    
    return best_angle;
}

/// Detect skew using Hough transform (simplified)
pub fn detectSkewHough(
    allocator: std.mem.Allocator,
    binary_image: *const Image,
    max_angle: f32,
) !f32 {
    // Collect foreground pixel coordinates
    var points = std.ArrayList([2]u32).init(allocator);
    defer points.deinit();
    
    var y: u32 = 0;
    while (y < binary_image.height) : (y += 1) {
        var x: u32 = 0;
        while (x < binary_image.width) : (x += 1) {
            if (binary_image.getPixel(x, y) == 255) {
                try points.append(.{ x, y });
            }
        }
    }
    
    // Sample subset for performance (max 10000 points)
    const sample_size = @min(points.items.len, 10000);
    
    // Simplified Hough: test angles and count votes
    const angle_bins = 180; // -90 to +90 degrees, 1 degree resolution
    const angle_votes = try allocator.alloc(u32, angle_bins);
    defer allocator.free(angle_votes);
    @memset(angle_votes, 0);
    
    // For each pair of points, calculate angle
    var i: usize = 0;
    while (i < sample_size) : (i += 10) { // Sample every 10th point
        var j: usize = i + 1;
        while (j < sample_size) : (j += 10) {
            const p1 = points.items[i];
            const p2 = points.items[j];
            
            const dx = @as(f32, @floatFromInt(@as(i32, @intCast(p2[0])) - @as(i32, @intCast(p1[0]))));
            const dy = @as(f32, @floatFromInt(@as(i32, @intCast(p2[1])) - @as(i32, @intCast(p1[1]))));
            
            if (dx != 0) {
                const angle_rad = std.math.atan(dy / dx);
                const angle_deg = angle_rad * 180.0 / std.math.pi;
                
                if (@abs(angle_deg) <= max_angle) {
                    const bin = @as(usize, @intFromFloat(angle_deg + 90.0));
                    if (bin < angle_bins) {
                        angle_votes[bin] += 1;
                    }
                }
            }
        }
    }
    
    // Find angle with most votes
    var max_votes: u32 = 0;
    var best_bin: usize = 90; // Default to 0 degrees
    
    for (angle_votes, 0..) |votes, bin| {
        if (votes > max_votes) {
            max_votes = votes;
            best_bin = bin;
        }
    }
    
    return @as(f32, @floatFromInt(best_bin)) - 90.0;
}

// ============================================================================
// Baseline Detection
// ============================================================================

/// Detect baseline for a text line (bottom of characters)
pub fn detectBaseline(
    binary_image: *const Image,
    line: *const TextLine,
) u32 {
    // Simple heuristic: baseline is at the bottom of the line
    // More sophisticated: find the row with most black pixels in lower half
    
    const mid_y = (line.y_start + line.y_end) / 2;
    var max_pixels: u32 = 0;
    var baseline_y = line.y_end;
    
    var y = mid_y;
    while (y <= line.y_end) : (y += 1) {
        var pixel_count: u32 = 0;
        var x: u32 = 0;
        while (x < binary_image.width) : (x += 1) {
            if (binary_image.getPixel(x, y) == 255) {
                pixel_count += 1;
            }
        }
        
        if (pixel_count > max_pixels) {
            max_pixels = pixel_count;
            baseline_y = y;
        }
    }
    
    return baseline_y;
}

/// Update baselines for all lines
pub fn detectBaselines(
    binary_image: *const Image,
    lines: []TextLine,
) void {
    for (lines) |*line| {
        line.baseline = detectBaseline(binary_image, line);
    }
}

// ============================================================================
// Full Pipeline
// ============================================================================

pub const LineDetectionResult = struct {
    lines: []TextLine,
    skew_angle: f32,
    deskewed_image: Image,
    
    pub fn deinit(self: *LineDetectionResult) void {
        self.deskewed_image.allocator.free(self.lines);
        self.deskewed_image.deinit();
    }
};

/// Full line detection pipeline
pub fn detectLines(
    allocator: std.mem.Allocator,
    binary_image: *const Image,
    options: struct {
        detect_skew: bool = true,
        max_skew_angle: f32 = 15.0,
        min_line_gap: u32 = 5,
        min_line_height: u32 = 5,
        skew_method: enum { Projection, Hough } = .Projection,
    },
) !LineDetectionResult {
    var skew_angle: f32 = 0.0;
    var working_image: Image = undefined;
    var needs_deinit = false;
    
    // Detect and correct skew if enabled
    if (options.detect_skew) {
        skew_angle = switch (options.skew_method) {
            .Projection => try detectSkewProjection(
                allocator,
                binary_image,
                options.max_skew_angle,
                0.5, // 0.5 degree steps
            ),
            .Hough => try detectSkewHough(
                allocator,
                binary_image,
                options.max_skew_angle,
            ),
        };
        
        if (@abs(skew_angle) > 0.1) {
            // Rotate to correct skew
            working_image = try transform.rotate(
                allocator,
                binary_image,
                -skew_angle, // Negative to correct
                .Nearest,
            );
            needs_deinit = true;
        } else {
            // No significant skew
            working_image = binary_image.*;
            skew_angle = 0.0;
        }
    } else {
        working_image = binary_image.*;
    }
    
    // Segment into lines
    var lines = try segmentLines(
        allocator,
        &working_image,
        options.min_line_gap,
        options.min_line_height,
    );
    
    // Detect baselines
    detectBaselines(&working_image, lines);
    
    // If we rotated, keep the deskewed image; otherwise copy
    var result_image: Image = undefined;
    if (needs_deinit) {
        result_image = working_image;
    } else {
        result_image = try Image.init(allocator, binary_image.width, binary_image.height);
        @memcpy(result_image.data, binary_image.data);
    }
    
    return LineDetectionResult{
        .lines = lines,
        .skew_angle = skew_angle,
        .deskewed_image = result_image,
    };
}

// ============================================================================
// C FFI Exports
// ============================================================================

const CTextLine = extern struct {
    y_start: u32,
    y_end: u32,
    baseline: u32,
};

const CLineDetectionResult = extern struct {
    lines: [*]CTextLine,
    line_count: u32,
    skew_angle: f32,
    deskewed_width: u32,
    deskewed_height: u32,
    deskewed_data: [*]u8,
};

export fn nExtract_CCA_analyze(
    image_data: [*]const u8,
    width: u32,
    height: u32,
    connectivity: u8, // 4 or 8
) callconv(.C) ?*CLineDetectionResult {
    _ = image_data;
    _ = width;
    _ = height;
    _ = connectivity;
    // TODO: Implement C wrapper
    return null;
}

export fn nExtract_Lines_detect(
    image_data: [*]const u8,
    width: u32,
    height: u32,
    detect_skew: bool,
) callconv(.C) ?*CLineDetectionResult {
    _ = image_data;
    _ = width;
    _ = height;
    _ = detect_skew;
    // TODO: Implement C wrapper
    return null;
}

export fn nExtract_Lines_free(result: *CLineDetectionResult) callconv(.C) void {
    _ = result;
    // TODO: Implement cleanup
}
