// Flame Graph Generator - Interactive Performance Visualization
// Generates SVG flame graphs from CPU profiling data

const std = @import("std");
const Allocator = std.mem.Allocator;
const cpu_profiler = @import("cpu_profiler.zig");

pub const FlameGraphConfig = struct {
    min_width_percent: f32 = 0.1,
    color_scheme: ColorScheme = .hot,
    interactive: bool = true,
    title: []const u8 = "CPU Profile Flame Graph",
    width: u32 = 1200,
    height: u32 = 800,
};

pub const ColorScheme = enum {
    hot,
    cold,
    rainbow,
    monochrome,
};

const StackNode = struct {
    name: []const u8,
    file: []const u8,
    line: u32,
    samples: u64,
    children: std.StringHashMap(*StackNode),
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8, file: []const u8, line: u32) !*StackNode {
        const node = try allocator.create(StackNode);
        node.* = .{
            .name = try allocator.dupe(u8, name),
            .file = try allocator.dupe(u8, file),
            .line = line,
            .samples = 0,
            .children = std.StringHashMap(*StackNode).init(allocator),
            .allocator = allocator,
        };
        return node;
    }

    pub fn deinit(self: *StackNode) void {
        var iter = self.children.valueIterator();
        while (iter.next()) |child| {
            child.*.deinit();
        }
        self.children.deinit();
        self.allocator.free(self.name);
        self.allocator.free(self.file);
        self.allocator.destroy(self);
    }

    pub fn addSample(self: *StackNode, stack: []const cpu_profiler.StackFrame) !void {
        self.samples += 1;

        if (stack.len == 0) return;

        const frame = stack[0];
        const key = frame.function_name;

        const child = if (self.children.get(key)) |existing|
            existing
        else blk: {
            const new_child = try StackNode.init(self.allocator, frame.function_name, frame.file_path, frame.line_number);
            try self.children.put(try self.allocator.dupe(u8, key), new_child);
            break :blk new_child;
        };

        try child.addSample(stack[1..]);
    }
};

pub const FlameGraph = struct {
    config: FlameGraphConfig,
    root: *StackNode,
    total_samples: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: FlameGraphConfig) !*FlameGraph {
        const graph = try allocator.create(FlameGraph);
        graph.* = .{
            .config = config,
            .root = try StackNode.init(allocator, "root", "", 0),
            .total_samples = 0,
            .allocator = allocator,
        };
        return graph;
    }

    pub fn deinit(self: *FlameGraph) void {
        self.root.deinit();
        self.allocator.destroy(self);
    }

    pub fn addProfile(self: *FlameGraph, profile: *const cpu_profiler.CpuProfile) !void {
        for (profile.samples.items) |sample| {
            if (sample.stack.len > 0) {
                // Reverse stack (root at index 0)
                var reversed = try self.allocator.alloc(cpu_profiler.StackFrame, sample.stack.len);
                defer self.allocator.free(reversed);

                var i: usize = 0;
                while (i < sample.stack.len) : (i += 1) {
                    reversed[i] = sample.stack[sample.stack.len - 1 - i];
                }

                try self.root.addSample(reversed);
                self.total_samples += 1;
            }
        }
    }

    pub fn generateSvg(self: *const FlameGraph, writer: anytype) !void {
        // SVG header
        try writer.print(
            \\<?xml version="1.0" standalone="no"?>
            \\<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
            \\<svg version="1.1" width="{d}" height="{d}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
            \\
        , .{ self.config.width, self.config.height });

        // Title
        try writer.print(
            \\<text x="50%" y="24" text-anchor="middle" font-size="17" font-family="Verdana" fill="black">{s}</text>
            \\
        , .{self.config.title});

        // Details text (for mouseover)
        try writer.writeAll(
            \\<text id="details" x="10" y="754" font-size="12" font-family="Verdana" fill="black"></text>
            \\
        );

        // JavaScript for interactivity
        if (self.config.interactive) {
            try self.writeJavaScript(writer);
        }

        // Render flame graph
        const frame_height = 16;
        const y_offset = 40;
        try self.renderNode(writer, self.root, 0, @floatFromInt(self.config.width), y_offset, frame_height);

        try writer.writeAll("</svg>\n");
    }

    fn renderNode(
        self: *const FlameGraph,
        writer: anytype,
        node: *const StackNode,
        x: f32,
        width: f32,
        y: u32,
        frame_height: u32,
    ) !void {
        if (node.samples == 0) return;

        const percent = (@as(f32, @floatFromInt(node.samples)) / @as(f32, @floatFromInt(self.total_samples))) * 100.0;

        // Skip if too narrow
        const pixel_width = (width * percent) / 100.0;
        if (pixel_width < (self.config.config.width * self.config.min_width_percent / 100.0)) {
            return;
        }

        // Choose color
        const color = self.getColor(percent);

        // Draw rectangle
        try writer.print(
            \\<g class="func_g" onmouseover="s('{s}')" onmouseout="c()" onclick="zoom(this)">
            \\<title>{s} ({d} samples, {d:.2}%)</title>
            \\<rect x="{d:.2}" y="{d}" width="{d:.2}" height="{d}" fill="{s}" stroke="white"/>
            \\<text x="{d:.2}" y="{d}" font-size="12" font-family="Verdana" fill="black">{s}</text>
            \\</g>
            \\
        , .{
            node.name,
            node.name,
            node.samples,
            percent,
            x,
            y,
            pixel_width,
            frame_height - 1,
            color,
            x + 3,
            y + frame_height - 4,
            self.truncateText(node.name, pixel_width),
        });

        // Render children
        var child_x = x;
        var iter = node.children.valueIterator();
        while (iter.next()) |child| {
            const child_percent = (@as(f32, @floatFromInt(child.*.samples)) / @as(f32, @floatFromInt(node.samples))) * 100.0;
            const child_width = pixel_width * child_percent / 100.0;

            try self.renderNode(writer, child.*, child_x, width, y + frame_height, frame_height);

            child_x += child_width;
        }
    }

    fn getColor(self: *const FlameGraph, percent: f32) []const u8 {
        return switch (self.config.color_scheme) {
            .hot => self.getHotColor(percent),
            .cold => self.getColdColor(percent),
            .rainbow => self.getRainbowColor(percent),
            .monochrome => "#cccccc",
        };
    }

    fn getHotColor(_: *const FlameGraph, percent: f32) []const u8 {
        // Red-orange-yellow gradient based on percentage
        const intensity = @as(u8, @intFromFloat(percent * 2.55));
        _ = intensity;

        if (percent > 10) return "#ff3030";
        if (percent > 5) return "#ff8000";
        if (percent > 2) return "#ffaa00";
        if (percent > 1) return "#ffcc00";
        return "#ffff00";
    }

    fn getColdColor(_: *const FlameGraph, percent: f32) []const u8 {
        if (percent > 10) return "#0000ff";
        if (percent > 5) return "#3366ff";
        if (percent > 2) return "#6699ff";
        if (percent > 1) return "#99ccff";
        return "#ccffff";
    }

    fn getRainbowColor(_: *const FlameGraph, percent: f32) []const u8 {
        const hue = @mod(percent * 3.6, 360.0);
        _ = hue;

        if (percent > 15) return "#ff0000";
        if (percent > 12) return "#ff8000";
        if (percent > 9) return "#ffff00";
        if (percent > 6) return "#00ff00";
        if (percent > 3) return "#0000ff";
        return "#8000ff";
    }

    fn truncateText(_: *const FlameGraph, text: []const u8, width: f32) []const u8 {
        const chars_per_pixel = 0.08;
        const max_chars = @as(usize, @intFromFloat(width * chars_per_pixel));

        if (text.len <= max_chars) return text;
        if (max_chars < 3) return "";

        return text[0 .. max_chars - 2];
    }

    fn writeJavaScript(self: *const FlameGraph, writer: anytype) !void {
        _ = self;
        try writer.writeAll(
            \\<script type="text/ecmascript">
            \\<![CDATA[
            \\var details = document.getElementById("details");
            \\function s(txt) { details.textContent = txt; }
            \\function c() { details.textContent = ""; }
            \\function zoom(element) {
            \\    var title = element.getElementsByTagName("title")[0];
            \\    if (title) { console.log("Zoom to: " + title.textContent); }
            \\}
            \\]]>
            \\</script>
            \\
        );
    }

    pub fn generateHtml(self: *const FlameGraph, writer: anytype) !void {
        try writer.writeAll(
            \\<!DOCTYPE html>
            \\<html>
            \\<head>
            \\<meta charset="utf-8">
            \\<title>Flame Graph</title>
            \\<style>
            \\body { font-family: Verdana, sans-serif; margin: 20px; }
            \\h1 { margin-bottom: 10px; }
            \\#flamegraph { border: 1px solid #ccc; }
            \\.controls { margin: 10px 0; }
            \\.controls button { margin-right: 10px; padding: 5px 15px; }
            \\</style>
            \\</head>
            \\<body>
            \\<h1>Performance Flame Graph</h1>
            \\<div class="controls">
            \\<button onclick="resetZoom()">Reset Zoom</button>
            \\<button onclick="exportSvg()">Export SVG</button>
            \\</div>
            \\<div id="flamegraph">
            \\
        );

        // Embed SVG
        try self.generateSvg(writer);

        try writer.writeAll(
            \\</div>
            \\<script>
            \\function resetZoom() { location.reload(); }
            \\function exportSvg() {
            \\  var svg = document.querySelector('svg');
            \\  var serializer = new XMLSerializer();
            \\  var source = serializer.serializeToString(svg);
            \\  var blob = new Blob([source], {type: 'image/svg+xml'});
            \\  var url = URL.createObjectURL(blob);
            \\  var a = document.createElement('a');
            \\  a.href = url;
            \\  a.download = 'flamegraph.svg';
            \\  a.click();
            \\}
            \\</script>
            \\</body>
            \\</html>
            \\
        );
    }
};

// Generate flame graph from CPU profile
pub fn generateFlameGraph(allocator: Allocator, profile: *const cpu_profiler.CpuProfile, config: FlameGraphConfig) !*FlameGraph {
    var graph = try FlameGraph.init(allocator, config);
    errdefer graph.deinit();

    try graph.addProfile(profile);

    return graph;
}

// Testing
test "FlameGraph basic" {
    const allocator = std.testing.allocator;

    const config = FlameGraphConfig{
        .min_width_percent = 0.1,
        .color_scheme = .hot,
        .interactive = true,
    };

    var graph = try FlameGraph.init(allocator, config);
    defer graph.deinit();

    // Test empty graph
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try graph.generateSvg(buffer.writer());
    try std.testing.expect(buffer.items.len > 0);
}
