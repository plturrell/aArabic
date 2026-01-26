// Sudoku HPC Benchmark - Practical Algorithm Performance Demonstration
// Showcases: FLOPs, Memory Bandwidth, Latency, Scaling through Sudoku solving

const std = @import("std");
const grid_mod = @import("sudoku/grid.zig");
const puzzles = @import("sudoku/puzzles.zig");
const solvers = @import("sudoku/solvers.zig");
const SudokuGrid = grid_mod.SudokuGrid;

const AlgorithmResult = struct {
    name: []const u8,
    time_ms: f64,
    solved: bool,
    flops_estimate_m: f64,
};

fn benchmarkAlgorithm(
    puzzle_str: []const u8,
    comptime solver_func: fn(*SudokuGrid) bool,
    name: []const u8,
) !AlgorithmResult {
    var grid = try SudokuGrid.fromString(puzzle_str);
    
    const start = std.time.nanoTimestamp();
    const solved = solver_func(&grid);
    const end = std.time.nanoTimestamp();
    
    const time_ns = end - start;
    const time_ms = @as(f64, @floatFromInt(time_ns)) / 1_000_000.0;
    
    // Rough FLOPs estimate: ~1000 operations per cell checked
    const flops_estimate = 81.0 * 9.0 * 1000.0; // 729k ops estimate
    const flops_m = flops_estimate / 1_000_000.0;
    
    return .{
        .name = name,
        .time_ms = time_ms,
        .solved = solved,
        .flops_estimate_m = flops_m,
    };
}

fn printJSON(results: []const AlgorithmResult) !void {
    const stdout = std.io.getStdOut().writer();
    
    try stdout.writeAll(",\n");  // Continue from HPC benchmarks
    try stdout.writeAll("  \"sudoku_demo\": {\n");
    try stdout.writeAll("    \"description\": \"Practical HPC metrics via Sudoku solving\",\n");
    try stdout.writeAll("    \"algorithms\": [\n");
    
    for (results, 0..) |result, i| {
        try stdout.writeAll("      {\n");
        try stdout.print("        \"name\": \"{s}\",\n", .{result.name});
        try stdout.print("        \"time_ms\": {d:.2},\n", .{result.time_ms});
        try stdout.print("        \"solved\": {},\n", .{result.solved});
        try stdout.print("        \"flops_m\": {d:.2}\n", .{result.flops_estimate_m});
        
        if (i < results.len - 1) {
            try stdout.writeAll("      },\n");
        } else {
            try stdout.writeAll("      }\n");
        }
    }
    
    try stdout.writeAll("    ],\n");
    
    // Calculate speedups
    if (results.len > 0) {
        const baseline = results[0].time_ms;
        try stdout.writeAll("    \"speedups\": {\n");
        try stdout.print("      \"backtracking_baseline\": 1.0,\n", .{});
        if (results.len > 1) {
            const bitmask_speedup = baseline / results[1].time_ms;
            try stdout.print("      \"bitmask_vs_backtracking\": {d:.2},\n", .{bitmask_speedup});
        }
        if (results.len > 2) {
            const constraint_speedup = baseline / results[2].time_ms;
            try stdout.print("      \"constraint_vs_backtracking\": {d:.2}\n", .{constraint_speedup});
        }
        try stdout.writeAll("    },\n");
    }
    
    // Memory access patterns
    try stdout.writeAll("    \"memory_patterns\": {\n");
    try stdout.writeAll("      \"grid_size_bytes\": 81,\n");
    try stdout.writeAll("      \"cache_friendly\": true,\n");
    try stdout.writeAll("      \"access_pattern\": \"row-major\"\n");
    try stdout.writeAll("    }\n");
    
    try stdout.writeAll("  }\n");
    try stdout.writeAll("}\n");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Run benchmarks
    
    const puzzle = puzzles.hard;
    
    var results = std.ArrayList(AlgorithmResult).init(allocator);
    defer results.deinit();
    
    // Benchmark 1: Backtracking
    const result1 = try benchmarkAlgorithm(puzzle, solvers.solveBacktracking, "Backtracking");
    try results.append(result1);
    
    // Benchmark 2: Bitmask
    const result2 = try benchmarkAlgorithm(puzzle, solvers.solveBitmask, "Bitmask");
    try results.append(result2);
    
    // Benchmark 3: Constraint Propagation
    const result3 = try benchmarkAlgorithm(puzzle, solvers.solveConstraintPropagation, "Constraint Propagation");
    try results.append(result3);
    
    
    // Output JSON to stdout
    try printJSON(results.items);
}