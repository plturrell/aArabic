const std = @import("std");
const graph = @import("graph.zig");

const GraphResult = graph.GraphResult;

/// Graph visualization format
pub const VisualizationFormat = enum {
    dot,      // Graphviz DOT
    json,     // JSON graph format
    cytoscape, // Cytoscape.js format
    d3,       // D3.js force-directed
    
    pub fn toString(self: VisualizationFormat) []const u8 {
        return switch (self) {
            .dot => "dot",
            .json => "json",
            .cytoscape => "cytoscape",
            .d3 => "d3",
        };
    }
};

/// Graph visualizer
pub const GraphVisualizer = struct {
    allocator: std.mem.Allocator,
    format: VisualizationFormat,
    
    pub fn init(allocator: std.mem.Allocator, format: VisualizationFormat) GraphVisualizer {
        return GraphVisualizer{
            .allocator = allocator,
            .format = format,
        };
    }
    
    /// Convert graph results to visualization format
    pub fn visualize(self: GraphVisualizer, results: []GraphResult) ![]const u8 {
        return switch (self.format) {
            .dot => try self.toDot(results),
            .json => try self.toJSON(results),
            .cytoscape => try self.toCytoscape(results),
            .d3 => try self.toD3(results),
        };
    }
    
    /// Convert to Graphviz DOT format
    fn toDot(self: GraphVisualizer, results: []GraphResult) ![]const u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit();
        
        const writer = output.writer();
        
        try writer.writeAll("digraph LineageGraph {\n");
        try writer.writeAll("  rankdir=LR;\n");
        try writer.writeAll("  node [shape=box, style=rounded];\n\n");
        
        // Add nodes and edges
        for (results) |result| {
            if (result.edge_id) |edge_id| {
                try writer.print(
                    "  \"{s}\" -> \"{s}\" [label=\"{s}\"];\n",
                    .{ result.vertex_id, edge_id, result.vertex_id },
                );
            } else {
                try writer.print("  \"{s}\";\n", .{result.vertex_id});
            }
        }
        
        try writer.writeAll("}\n");
        
        return output.toOwnedSlice();
    }
    
    /// Convert to JSON format
    fn toJSON(self: GraphVisualizer, results: []GraphResult) ![]const u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit();
        
        const writer = output.writer();
        
        try writer.writeAll("{\n");
        try writer.writeAll("  \"nodes\": [\n");
        
        for (results, 0..) |result, i| {
            try writer.print(
                "    {{\"id\": \"{s}\", \"depth\": {d}}}",
                .{ result.vertex_id, result.hop_distance },
            );
            if (i < results.len - 1) {
                try writer.writeAll(",");
            }
            try writer.writeAll("\n");
        }
        
        try writer.writeAll("  ],\n");
        try writer.writeAll("  \"edges\": [\n");
        
        var edge_count: usize = 0;
        for (results) |result| {
            if (result.edge_id) |edge_id| {
                if (edge_count > 0) {
                    try writer.writeAll(",\n");
                }
                try writer.print(
                    "    {{\"id\": \"{s}\", \"source\": \"{s}\", \"target\": \"{s}\"}}",
                    .{ edge_id, result.vertex_id, edge_id },
                );
                edge_count += 1;
            }
        }
        
        try writer.writeAll("\n  ]\n");
        try writer.writeAll("}\n");
        
        return output.toOwnedSlice();
    }
    
    /// Convert to Cytoscape.js format
    fn toCytoscape(self: GraphVisualizer, results: []GraphResult) ![]const u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit();
        
        const writer = output.writer();
        
        try writer.writeAll("{\n");
        try writer.writeAll("  \"elements\": {\n");
        try writer.writeAll("    \"nodes\": [\n");
        
        for (results, 0..) |result, i| {
            try writer.print(
                "      {{\"data\": {{\"id\": \"{s}\", \"depth\": {d}}}}}",
                .{ result.vertex_id, result.hop_distance },
            );
            if (i < results.len - 1) {
                try writer.writeAll(",");
            }
            try writer.writeAll("\n");
        }
        
        try writer.writeAll("    ],\n");
        try writer.writeAll("    \"edges\": [\n");
        
        var edge_count: usize = 0;
        for (results) |result| {
            if (result.edge_id) |edge_id| {
                if (edge_count > 0) {
                    try writer.writeAll(",\n");
                }
                try writer.print(
                    "      {{\"data\": {{\"id\": \"{s}\", \"source\": \"{s}\", \"target\": \"{s}\"}}}}",
                    .{ edge_id, result.vertex_id, edge_id },
                );
                edge_count += 1;
            }
        }
        
        try writer.writeAll("\n    ]\n");
        try writer.writeAll("  }\n");
        try writer.writeAll("}\n");
        
        return output.toOwnedSlice();
    }
    
    /// Convert to D3.js format
    fn toD3(self: GraphVisualizer, results: []GraphResult) ![]const u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit();
        
        const writer = output.writer();
        
        try writer.writeAll("{\n");
        try writer.writeAll("  \"nodes\": [\n");
        
        for (results, 0..) |result, i| {
            try writer.print(
                "    {{\"id\": \"{s}\", \"group\": {d}}}",
                .{ result.vertex_id, result.hop_distance },
            );
            if (i < results.len - 1) {
                try writer.writeAll(",");
            }
            try writer.writeAll("\n");
        }
        
        try writer.writeAll("  ],\n");
        try writer.writeAll("  \"links\": [\n");
        
        var edge_count: usize = 0;
        for (results) |result| {
            if (result.edge_id) |_| {
                if (edge_count > 0) {
                    try writer.writeAll(",\n");
                }
                try writer.print(
                    "    {{\"source\": \"{s}\", \"target\": \"{s}\", \"value\": 1}}",
                    .{ result.vertex_id, result.vertex_id },
                );
                edge_count += 1;
            }
        }
        
        try writer.writeAll("\n  ]\n");
        try writer.writeAll("}\n");
        
        return output.toOwnedSlice();
    }
};

/// Lineage path formatter
pub const LineagePathFormatter = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) LineagePathFormatter {
        return LineagePathFormatter{
            .allocator = allocator,
        };
    }
    
    /// Format path as arrow chain
    pub fn formatAsArrowChain(self: LineagePathFormatter, path: []const []const u8) ![]const u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit();
        
        const writer = output.writer();
        
        for (path, 0..) |node, i| {
            try writer.writeAll(node);
            if (i < path.len - 1) {
                try writer.writeAll(" -> ");
            }
        }
        
        return output.toOwnedSlice();
    }
    
    /// Format path as hierarchical tree
    pub fn formatAsTree(self: LineagePathFormatter, paths: []const []const []const u8) ![]const u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit();
        
        const writer = output.writer();
        
        for (paths, 0..) |path, path_idx| {
            try writer.print("Path {d}:\n", .{path_idx + 1});
            for (path, 0..) |node, depth| {
                const indent = depth * 2;
                var i: usize = 0;
                while (i < indent) : (i += 1) {
                    try writer.writeAll(" ");
                }
                try writer.print("└─ {s}\n", .{node});
            }
            try writer.writeAll("\n");
        }
        
        return output.toOwnedSlice();
    }
    
    /// Format path with metadata
    pub fn formatWithMetadata(
        self: LineagePathFormatter,
        results: []GraphResult,
    ) ![]const u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit();
        
        const writer = output.writer();
        
        try writer.writeAll("Lineage Path:\n");
        try writer.writeAll("═══════════════════════════════════════\n\n");
        
        for (results, 0..) |result, i| {
            try writer.print("[{d}] {s}\n", .{ i, result.vertex_id });
            try writer.print("    Hop Distance: {d}\n", .{result.hop_distance});
            if (result.edge_id) |edge_id| {
                try writer.print("    Edge: {s}\n", .{edge_id});
            }
            if (result.path.len > 0) {
                try writer.writeAll("    Full Path: ");
                for (result.path, 0..) |node, j| {
                    try writer.writeAll(node);
                    if (j < result.path.len - 1) {
                        try writer.writeAll(" → ");
                    }
                }
                try writer.writeAll("\n");
            }
            try writer.writeAll("\n");
        }
        
        return output.toOwnedSlice();
    }
};

/// Graph metrics calculator
pub const GraphMetrics = struct {
    total_vertices: usize,
    total_edges: usize,
    max_depth: u32,
    avg_degree: f64,
    diameter: u32,
    
    pub fn calculate(results: []GraphResult) GraphMetrics {
        var unique_vertices = std.StringHashMap(void).init(std.heap.page_allocator);
        defer unique_vertices.deinit();
        
        var max_depth: u32 = 0;
        var edge_count: usize = 0;
        
        for (results) |result| {
            unique_vertices.put(result.vertex_id, {}) catch {};
            if (result.hop_distance > max_depth) {
                max_depth = result.hop_distance;
            }
            if (result.edge_id) |_| {
                edge_count += 1;
            }
        }
        
        const vertex_count = unique_vertices.count();
        const avg_degree = if (vertex_count > 0)
            @as(f64, @floatFromInt(edge_count * 2)) / @as(f64, @floatFromInt(vertex_count))
        else
            0.0;
        
        return GraphMetrics{
            .total_vertices = vertex_count,
            .total_edges = edge_count,
            .max_depth = max_depth,
            .avg_degree = avg_degree,
            .diameter = max_depth, // Approximation
        };
    }
    
    pub fn format(
        self: GraphMetrics,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            \\
            \\Graph Metrics:
            \\  Total Vertices: {d}
            \\  Total Edges: {d}
            \\  Max Depth: {d}
            \\  Average Degree: {d:.2}
            \\  Diameter: {d}
            \\
        ,
            .{
                self.total_vertices,
                self.total_edges,
                self.max_depth,
                self.avg_degree,
                self.diameter,
            },
        );
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "VisualizationFormat - toString" {
    try std.testing.expectEqualStrings("dot", VisualizationFormat.dot.toString());
    try std.testing.expectEqualStrings("json", VisualizationFormat.json.toString());
    try std.testing.expectEqualStrings("cytoscape", VisualizationFormat.cytoscape.toString());
    try std.testing.expectEqualStrings("d3", VisualizationFormat.d3.toString());
}

test "GraphVisualizer - init" {
    const allocator = std.testing.allocator;
    const visualizer = GraphVisualizer.init(allocator, .json);
    
    try std.testing.expectEqual(VisualizationFormat.json, visualizer.format);
}

test "LineagePathFormatter - formatAsArrowChain" {
    const allocator = std.testing.allocator;
    const formatter = LineagePathFormatter.init(allocator);
    
    const path = [_][]const u8{ "A", "B", "C" };
    const result = try formatter.formatAsArrowChain(&path);
    defer allocator.free(result);
    
    try std.testing.expectEqualStrings("A -> B -> C", result);
}

test "GraphMetrics - calculate" {
    const allocator = std.testing.allocator;
    
    var results = [_]GraphResult{
        GraphResult{
            .vertex_id = try allocator.dupe(u8, "v1"),
            .edge_id = try allocator.dupe(u8, "e1"),
            .hop_distance = 1,
            .path = &[_][]const u8{},
        },
        GraphResult{
            .vertex_id = try allocator.dupe(u8, "v2"),
            .edge_id = null,
            .hop_distance = 2,
            .path = &[_][]const u8{},
        },
    };
    defer {
        for (results) |*r| {
            allocator.free(r.vertex_id);
            if (r.edge_id) |e| allocator.free(e);
        }
    }
    
    const metrics = GraphMetrics.calculate(&results);
    
    try std.testing.expectEqual(@as(usize, 2), metrics.total_vertices);
    try std.testing.expectEqual(@as(usize, 1), metrics.total_edges);
    try std.testing.expectEqual(@as(u32, 2), metrics.max_depth);
}
