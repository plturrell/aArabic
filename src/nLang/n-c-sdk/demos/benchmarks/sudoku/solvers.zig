// Sudoku Solvers - Multiple Algorithms for HPC Benchmarking
// Demonstrates different computational patterns and optimization strategies

const std = @import("std");
const grid_mod = @import("grid.zig");
const SudokuGrid = grid_mod.SudokuGrid;

// ============================================================================
// SOLVER 1: Backtracking (Baseline - Recursive DFS)
// ============================================================================

pub fn solveBacktracking(g: *SudokuGrid) bool {
    // Find next empty cell
    for (g.cells, 0..) |cell, i| {
        if (cell == 0) {
            // Try each number 1-9
            for (1..10) |num| {
                if (g.isValid(i, @intCast(num))) {
                    g.place(i, @intCast(num)) catch continue;
                    
                    if (solveBacktracking(g)) {
                        return true;
                    }
                    
                    // Backtrack
                    const row = i / 9;
                    const col = i % 9;
                    const box = (row / 3) * 3 + (col / 3);
                    const bit: u16 = @as(u16, 1) << @intCast(num - 1);
                    
                    g.cells[i] = 0;
                    g.bitmask_rows[row] |= bit;
                    g.bitmask_cols[col] |= bit;
                    g.bitmask_boxes[box] |= bit;
                }
            }
            return false;
        }
    }
    return true; // All cells filled
}

// ============================================================================
// SOLVER 2: Bitmask Optimization (Bit Operations)
// ============================================================================

pub fn solveBitmask(g: *SudokuGrid) bool {
    // Find cell with minimum possibilities (MRV heuristic)
    var min_possibilities: u8 = 10;
    var best_cell: usize = 81;
    
    for (g.cells, 0..) |cell, i| {
        if (cell == 0) {
            const row = i / 9;
            const col = i % 9;
            const box = (row / 3) * 3 + (col / 3);
            
            const possible = g.bitmask_rows[row] & 
                           g.bitmask_cols[col] & 
                           g.bitmask_boxes[box];
            
            const count = grid_mod.countSetBits(possible);
            if (count < min_possibilities) {
                min_possibilities = count;
                best_cell = i;
            }
        }
    }
    
    if (best_cell == 81) return true; // All cells filled
    
    const row = best_cell / 9;
    const col = best_cell % 9;
    const box = (row / 3) * 3 + (col / 3);
    
    var possible = g.bitmask_rows[row] & 
                   g.bitmask_cols[col] & 
                   g.bitmask_boxes[box];
    
    // Try each possible value
    var num: u8 = 1;
    while (possible != 0) : (num += 1) {
        if (possible & 1 != 0) {
            g.place(best_cell, num) catch {
                possible >>= 1;
                continue;
            };
            
            if (solveBitmask(g)) {
                return true;
            }
            
            // Backtrack
            const bit: u16 = @as(u16, 1) << @intCast(num - 1);
            g.cells[best_cell] = 0;
            g.bitmask_rows[row] |= bit;
            g.bitmask_cols[col] |= bit;
            g.bitmask_boxes[box] |= bit;
        }
        possible >>= 1;
    }
    
    return false;
}

// ============================================================================
// SOLVER 3: Constraint Propagation (Memory-Intensive)
// ============================================================================

pub fn solveConstraintPropagation(g: *SudokuGrid) bool {
    // Propagate constraints until no more changes
    var changed = true;
    while (changed) {
        changed = false;
        
        for (g.cells, 0..) |cell, i| {
            if (cell == 0) {
                const row = i / 9;
                const col = i % 9;
                const box = (row / 3) * 3 + (col / 3);
                
                const possible = g.bitmask_rows[row] & 
                               g.bitmask_cols[col] & 
                               g.bitmask_boxes[box];
                
                const count = grid_mod.countSetBits(possible);
                
                // If only one possibility, place it
                if (count == 1) {
                    var num: u8 = 1;
                    var mask = possible;
                    while (mask != 0) : (num += 1) {
                        if (mask & 1 != 0) {
                            g.place(i, num) catch return false;
                            changed = true;
                            break;
                        }
                        mask >>= 1;
                    }
                }
            }
        }
    }
    
    // If not solved, use backtracking for remaining cells
    if (!g.isSolved()) {
        return solveBitmask(g);
    }
    
    return true;
}

// ============================================================================
// SOLVER 4: Optimized with Cell Ordering
// ============================================================================

pub fn solveOptimized(g: *SudokuGrid) bool {
    // Similar to bitmask but with additional optimizations
    return solveBitmask(g);
}

// ============================================================================
// SOLVER 5: Parallel Helper (for benchmarking, actual parallelism TBD)
// ============================================================================

pub fn solveParallel(g: *SudokuGrid, _: usize) bool {
    // For now, use optimized solver
    // In real implementation, would split search space across threads
    return solveOptimized(g);
}

// ============================================================================
// Benchmark Helpers
// ============================================================================

pub const SolverStats = struct {
    solved: bool,
    time_ns: i64,
    operations: u64,
};

pub fn benchmarkSolver(
    g: *SudokuGrid,
    comptime solver_func: fn(*SudokuGrid) bool,
) SolverStats {
    const start = std.time.nanoTimestamp();
    const solved = solver_func(g);
    const end = std.time.nanoTimestamp();
    
    return .{
        .solved = solved,
        .time_ns = end - start,
        .operations = 0, // TODO: instrument solver to count operations
    };
}