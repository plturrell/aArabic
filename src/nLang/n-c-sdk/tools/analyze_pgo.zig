//! PGO Profile Analysis Tool
//! Analyzes profile data and provides optimization recommendations
//!
//! Usage:
//!   zig run tools/analyze_pgo.zig -- profile.pgo
//!   zig run tools/analyze_pgo.zig -- profile.pgo --detailed

const std = @import("std");
const mem = std.mem;
const fs = std.fs;

const stdout = std.fs.File.stdout().deprecatedWriter();
const stderr = std.fs.File.stderr().deprecatedWriter();

const PGOData = struct {
    function_frequencies: std.StringHashMap(u64),
    edge_frequencies: std.AutoHashMap(Edge, u64),
    
    const Edge = struct {
        from: u32,
        to: u32,
    };
    
    pub fn init(allocator: std.mem.Allocator) PGOData {
        return .{
            .function_frequencies = std.StringHashMap(u64).init(allocator),
            .edge_frequencies = std.AutoHashMap(Edge, u64).init(allocator),
        };
    }
    
    pub fn deinit(self: *PGOData) void {
        var it = self.function_frequencies.keyIterator();
        while (it.next()) |key| {
            self.function_frequencies.allocator.free(key.*);
        }
        self.function_frequencies.deinit();
        self.edge_frequencies.deinit();
    }
    
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !PGOData {
        var result = init(allocator);
        errdefer result.deinit();
        
        const file = try fs.cwd().openFile(path, .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
        defer allocator.free(content);
        
        var line_it = mem.tokenize(u8, content, "\n");
        while (line_it.next()) |line| {
            if (mem.indexOf(u8, line, "function:")) |_| {
                if (try parseFunctionFrequency(allocator, line)) |entry| {
                    try result.function_frequencies.put(entry.name, entry.count);
                }
            } else if (mem.indexOf(u8, line, "edge:")) |_| {
                if (parseEdgeFrequency(line)) |entry| {
                    try result.edge_frequencies.put(entry.edge, entry.count);
                }
            }
        }
        
        return result;
    }
    
    fn parseFunctionFrequency(allocator: std.mem.Allocator, line: []const u8) !?struct {
        name: []const u8,
        count: u64,
    } {
        var parts = mem.tokenize(u8, line, " ");
        _ = parts.next() orelse return null; // "function:"
        const name_slice = parts.next() orelse return null;
        _ = parts.next() orelse return null; // "count:"
        const count_str = parts.next() orelse return null;
        const count = std.fmt.parseInt(u64, count_str, 10) catch return null;
        
        const name = try allocator.dupe(u8, name_slice);
        return .{ .name = name, .count = count };
    }
    
    fn parseEdgeFrequency(line: []const u8) ?struct {
        edge: Edge,
        count: u64,
    } {
        var parts = mem.tokenize(u8, line, " ->");
        _ = parts.next() orelse return null; // "edge:"
        const from_str = parts.next() orelse return null;
        const to_str = parts.next() orelse return null;
        _ = parts.next() orelse return null; // "count:"
        const count_str = parts.next() orelse return null;
        
        const from = std.fmt.parseInt(u32, from_str, 10) catch return null;
        const to = std.fmt.parseInt(u32, to_str, 10) catch return null;
        const count = std.fmt.parseInt(u64, count_str, 10) catch return null;
        
        return .{
            .edge = .{ .from = from, .to = to },
            .count = count,
        };
    }
};

const FunctionStats = struct {
    name: []const u8,
    count: u64,
    percentage: f64,
    
    fn lessThan(_: void, lhs: FunctionStats, rhs: FunctionStats) bool {
        return lhs.count > rhs.count; // Sort descending
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        try stderr.writeAll("Usage: analyze_pgo <profile.pgo> [--detailed]\n");
        std.process.exit(1);
    }
    
    const profile_path = args[1];
    const detailed = args.len > 2 and mem.eql(u8, args[2], "--detailed");
    
    var pgo_data = try PGOData.loadFromFile(allocator, profile_path);
    defer pgo_data.deinit();
    
    try analyzeProfile(allocator, &pgo_data, detailed);
}

fn analyzeProfile(allocator: std.mem.Allocator, pgo_data: *PGOData, detailed: bool) !void {
    // Calculate total calls
    var total_calls: u64 = 0;
    var func_it = pgo_data.function_frequencies.valueIterator();
    while (func_it.next()) |count| {
        total_calls += count.*;
    }
    
    if (total_calls == 0) {
        try stderr.writeAll("⚠️  No profiling data found in file\n");
        return;
    }
    
    // Convert to sorted list
    var stats = std.ArrayList(FunctionStats).init(allocator);
    defer stats.deinit();
    
    var entry_it = pgo_data.function_frequencies.iterator();
    while (entry_it.next()) |entry| {
        const percentage = @as(f64, @floatFromInt(entry.value_ptr.*)) / 
                          @as(f64, @floatFromInt(total_calls)) * 100.0;
        try stats.append(.{
            .name = entry.key_ptr.*,
            .count = entry.value_ptr.*,
            .percentage = percentage,
        });
    }
    
    mem.sort(FunctionStats, stats.items, {}, FunctionStats.lessThan);
    
    // Print header
    try stdout.writeAll("\n🔥 Hot Path Analysis Report\n");
    try stdout.writeAll("═══════════════════════════════════════════════════════════════\n\n");
    
    // Print statistics
    try stdout.writeAll("📊 FUNCTION STATISTICS\n");
    try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
    try stdout.print("Total Functions: {d}\n", .{stats.items.len});
    try stdout.print("Total Calls:     {d}\n", .{total_calls});
    try stdout.writeAll("\n");
    
    // Categorize functions
    var hot_count: usize = 0;
    var warm_count: usize = 0;
    var cold_count: usize = 0;
    var hot_percentage: f64 = 0;
    
    for (stats.items) |stat| {
        if (stat.percentage >= 5.0) {
            hot_count += 1;
            hot_percentage += stat.percentage;
        } else if (stat.percentage >= 1.0) {
            warm_count += 1;
        } else {
            cold_count += 1;
        }
    }
    
    try stdout.writeAll("Function Coverage:\n");
    try stdout.print("  Total functions:     {d}\n", .{stats.items.len});
    try stdout.print("  Hot (≥5% CPU):       {d} ({d:.1}%)\n", .{
        hot_count,
        @as(f64, @floatFromInt(hot_count)) / @as(f64, @floatFromInt(stats.items.len)) * 100.0,
    });
    try stdout.print("  Warm (1-5% CPU):     {d} ({d:.1}%)\n", .{
        warm_count,
        @as(f64, @floatFromInt(warm_count)) / @as(f64, @floatFromInt(stats.items.len)) * 100.0,
    });
    try stdout.print("  Cold (<1% CPU):      {d} ({d:.1}%)\n", .{
        cold_count,
        @as(f64, @floatFromInt(cold_count)) / @as(f64, @floatFromInt(stats.items.len)) * 100.0,
    });
    try stdout.writeAll("\n");
    
    // Print hot functions
    if (hot_count > 0) {
        try stdout.writeAll("🔥 HOT FUNCTIONS (≥5% CPU)\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        try stdout.writeAll("Function                              CPU%       Calls\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        
        for (stats.items, 0..) |stat, i| {
            if (stat.percentage < 5.0) break;
            
            const name = if (stat.name.len > 35)
                stat.name[0..35]
            else
                stat.name;
            
            try stdout.print("{d:2}. {s:<35} {d:5.1}%  {d:>12}\n", .{
                i + 1,
                name,
                stat.percentage,
                stat.count,
            });
            
            if (detailed and i < 10) {
                try stdout.print("    Status: ", .{});
                
                // Simple heuristic: check if name looks like it has validation
                if (mem.indexOf(u8, stat.name, "Unsafe") != null or
                    mem.indexOf(u8, stat.name, "Fast") != null)
                {
                    try stdout.writeAll("⚠️  Likely already optimized\n");
                } else {
                    try stdout.writeAll("✅ Optimization candidate\n");
                }
            }
        }
        try stdout.writeAll("\n");
    }
    
    // Print warm functions if detailed
    if (detailed and warm_count > 0) {
        try stdout.writeAll("🌡️  WARM FUNCTIONS (1-5% CPU)\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        
        var shown: usize = 0;
        for (stats.items) |stat| {
            if (stat.percentage >= 5.0) continue;
            if (stat.percentage < 1.0) break;
            if (shown >= 10) break;
            
            const name = if (stat.name.len > 35)
                stat.name[0..35]
            else
                stat.name;
            
            try stdout.print("  {s:<35} {d:5.1}%  {d:>12}\n", .{
                name,
                stat.percentage,
                stat.count,
            });
            shown += 1;
        }
        
        if (warm_count > shown) {
            try stdout.print("  ... and {d} more\n", .{warm_count - shown});
        }
        try stdout.writeAll("\n");
    }
    
    // Print optimization recommendations
    try stdout.writeAll("🎯 OPTIMIZATION RECOMMENDATIONS\n");
    try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
    
    if (hot_count == 0) {
        try stdout.writeAll("  ✨ No hot functions detected (all <5% CPU time)\n");
        try stdout.writeAll("  → Profile may need more representative workload\n");
        try stdout.writeAll("  → Or application is well-balanced already\n");
    } else {
        try stdout.writeAll("HIGH PRIORITY:\n");
        
        var recommendation_count: usize = 0;
        for (stats.items, 0..) |stat, i| {
            if (stat.percentage < 5.0) break;
            if (recommendation_count >= 3) break;
            
            recommendation_count += 1;
            const potential_gain = stat.percentage * 0.5; // Assume 50% speedup
            
            try stdout.print("  {d}. {s}\n", .{ recommendation_count, stat.name });
            try stdout.print("     CPU time: {d:.1}%\n", .{stat.percentage});
            try stdout.print("     Calls: {d:>12}\n", .{stat.count});
            try stdout.print("     Potential gain: {d:.1}% of total runtime\n", .{potential_gain});
            
            if (i == 0) {
                try stdout.writeAll("     Priority: CRITICAL (highest CPU consumer)\n");
            } else {
                try stdout.writeAll("     Priority: HIGH\n");
            }
            try stdout.writeAll("\n");
        }
        
        if (hot_count > 3) {
            try stdout.print("  ... and {d} more hot functions\n\n", .{hot_count - 3});
        }
        
        try stdout.print("Expected Overall Speedup: {d:.1}x - {d:.1}x\n", .{
            1.0 + (hot_percentage * 0.3 / 100.0), // Conservative estimate
            1.0 + (hot_percentage * 0.5 / 100.0), // Optimistic estimate
        });
    }
    
    try stdout.writeAll("\n");
    
    // Edge statistics
    if (detailed and pgo_data.edge_frequencies.count() > 0) {
        try stdout.writeAll("📈 BRANCH PREDICTION ANALYSIS\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        
        var edge_it = pgo_data.edge_frequencies.iterator();
        var shown_edges: usize = 0;
        while (edge_it.next()) |entry| : (shown_edges += 1) {
            if (shown_edges >= 10) break;
            
            try stdout.print("  Edge {d} → {d}: {d:>12} executions\n", .{
                entry.key_ptr.from,
                entry.key_ptr.to,
                entry.value_ptr.*,
            });
        }
        
        if (pgo_data.edge_frequencies.count() > shown_edges) {
            try stdout.print("  ... and {d} more edges\n", .{
                pgo_data.edge_frequencies.count() - shown_edges,
            });
        }
        try stdout.writeAll("\n");
    }
    
    // Summary
    try stdout.writeAll("💡 SUMMARY\n");
    try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
    try stdout.print("• {d} hot function{s} identified ({d:.1}% of runtime)\n", .{
        hot_count,
        if (hot_count == 1) "" else "s",
        hot_percentage,
    });
    
    if (hot_count > 0) {
        try stdout.print("• Top {d} optimization{s} could provide significant gains\n", .{
            @min(hot_count, 3),
            if (hot_count == 1) "" else "s",
        });
        try stdout.writeAll("• Ensure all hot functions have:\n");
        try stdout.writeAll("  - Documented safety contracts\n");
        try stdout.writeAll("  - Validated inputs before optimization\n");
        try stdout.writeAll("  - Comprehensive test coverage\n");
    }
    
    if (hot_count < 3) {
        try stdout.writeAll("\n💭 CONSIDERATIONS\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        try stdout.writeAll("• Few hot functions detected - workload may be:\n");
        try stdout.writeAll("  - I/O bound (network, disk)\n");
        try stdout.writeAll("  - Well-balanced already\n");
        try stdout.writeAll("  - Need more representative profiling\n");
    }
    
    try stdout.writeAll("\n");
}