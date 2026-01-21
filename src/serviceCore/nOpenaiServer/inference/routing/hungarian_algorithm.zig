// ============================================================================
// Hungarian Algorithm - Day 32 Implementation
// ============================================================================
// Purpose: Optimal assignment algorithm (Kuhn-Munkres) for agent-model matching
// Week: Week 7 (Days 31-35) - Advanced Routing Strategies
// Phase: Month 3 - Advanced Features & Optimization
// ============================================================================

const std = @import("std");

// ============================================================================
// HUNGARIAN ALGORITHM SOLVER
// ============================================================================

pub const HungarianSolver = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) HungarianSolver {
        return .{ .allocator = allocator };
    }
    
    /// Solve assignment problem for NxM cost matrix (maximize scores)
    /// Returns array of assignments: result[agent_idx] = model_idx (or null)
    pub fn solve(self: *HungarianSolver, cost_matrix: [][]f32) ![]?usize {
        const n = cost_matrix.len;
        if (n == 0) return try self.allocator.alloc(?usize, 0);
        
        const m = cost_matrix[0].len;
        if (m == 0) {
            var result = try self.allocator.alloc(?usize, n);
            for (result) |*r| r.* = null;
            return result;
        }
        
        // Create working matrix (convert max to min problem)
        var matrix = try self.createWorkingMatrix(cost_matrix);
        defer self.freeMatrix(matrix);
        
        // Step 1: Row reduction
        self.subtractRowMins(matrix);
        
        // Step 2: Column reduction
        self.subtractColMins(matrix);
        
        // Step 3-5: Iterative assignment
        var assignments = try self.findOptimalAssignment(matrix);
        
        return assignments;
    }
    
    /// Create working copy and convert maximization to minimization
    fn createWorkingMatrix(self: *HungarianSolver, original: [][]f32) ![][]f32 {
        const n = original.len;
        const m = original[0].len;
        
        // Find maximum value
        var max_val: f32 = 0.0;
        for (original) |row| {
            for (row) |val| {
                max_val = @max(max_val, val);
            }
        }
        
        // Create copy and negate (max → min problem)
        var matrix = try self.allocator.alloc([]f32, n);
        for (original, 0..) |row, i| {
            matrix[i] = try self.allocator.alloc(f32, m);
            for (row, 0..) |val, j| {
                matrix[i][j] = max_val - val; // Convert to minimization
            }
        }
        
        return matrix;
    }
    
    /// Step 1: Subtract minimum from each row
    fn subtractRowMins(self: *HungarianSolver, matrix: [][]f32) void {
        _ = self;
        for (matrix) |row| {
            var min: f32 = std.math.floatMax(f32);
            for (row) |val| {
                min = @min(min, val);
            }
            for (row) |*val| {
                val.* -= min;
            }
        }
    }
    
    /// Step 2: Subtract minimum from each column
    fn subtractColMins(self: *HungarianSolver, matrix: [][]f32) void {
        _ = self;
        const n = matrix.len;
        const m = matrix[0].len;
        
        var j: usize = 0;
        while (j < m) : (j += 1) {
            var min: f32 = std.math.floatMax(f32);
            for (matrix) |row| {
                min = @min(min, row[j]);
            }
            for (matrix) |row| {
                row[j] -= min;
            }
        }
    }
    
    /// Steps 3-5: Find optimal assignment
    fn findOptimalAssignment(self: *HungarianSolver, matrix: [][]f32) ![]?usize {
        const n = matrix.len;
        const m = matrix[0].len;
        
        var assignments = try self.allocator.alloc(?usize, n);
        for (assignments) |*a| a.* = null;
        
        var row_covered = try self.allocator.alloc(bool, n);
        defer self.allocator.free(row_covered);
        @memset(row_covered, false);
        
        var col_covered = try self.allocator.alloc(bool, m);
        defer self.allocator.free(col_covered);
        @memset(col_covered, false);
        
        // Greedy initial assignment (find zeros)
        for (matrix, 0..) |row, i| {
            for (row, 0..) |val, j| {
                if (val < 0.0001 and !col_covered[j]) { // epsilon comparison
                    assignments[i] = j;
                    col_covered[j] = true;
                    break;
                }
            }
        }
        
        // For now, return greedy assignment
        // Full Hungarian algorithm with augmenting paths to be added
        return assignments;
    }
    
    fn freeMatrix(self: *HungarianSolver, matrix: [][]f32) void {
        for (matrix) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(matrix);
    }
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "HungarianSolver: basic 2x2 assignment" {
    const allocator = std.testing.allocator;
    
    var solver = HungarianSolver.init(allocator);
    
    // Simple 2x2 matrix
    var matrix = try allocator.alloc([]f32, 2);
    matrix[0] = try allocator.alloc(f32, 2);
    matrix[1] = try allocator.alloc(f32, 2);
    
    matrix[0][0] = 90.0; // Agent 0, Model 0
    matrix[0][1] = 75.0; // Agent 0, Model 1
    matrix[1][0] = 70.0; // Agent 1, Model 0
    matrix[1][1] = 95.0; // Agent 1, Model 1
    
    const assignments = try solver.solve(matrix);
    defer allocator.free(assignments);
    
    // Optimal: Agent 0→Model 0 (90), Agent 1→Model 1 (95) = 185
    // vs Agent 0→Model 1 (75), Agent 1→Model 0 (70) = 145
    try std.testing.expectEqual(@as(?usize, 0), assignments[0]);
    try std.testing.expectEqual(@as(?usize, 1), assignments[1]);
    
    // Cleanup
    allocator.free(matrix[0]);
    allocator.free(matrix[1]);
    allocator.free(matrix);
}

test "HungarianSolver: row reduction" {
    const allocator = std.testing.allocator;
    
    var solver = HungarianSolver.init(allocator);
    
    var matrix = try allocator.alloc([]f32, 2);
    matrix[0] = try allocator.alloc(f32, 2);
    matrix[1] = try allocator.alloc(f32, 2);
    
    matrix[0][0] = 10.0;
    matrix[0][1] = 15.0;
    matrix[1][0] = 20.0;
    matrix[1][1] = 25.0;
    
    solver.subtractRowMins(matrix);
    
    // Row 0 min was 10, Row 1 min was 20
    try std.testing.expectEqual(@as(f32, 0.0), matrix[0][0]);
    try std.testing.expectEqual(@as(f32, 5.0), matrix[0][1]);
    try std.testing.expectEqual(@as(f32, 0.0), matrix[1][0]);
    try std.testing.expectEqual(@as(f32, 5.0), matrix[1][1]);
    
    allocator.free(matrix[0]);
    allocator.free(matrix[1]);
    allocator.free(matrix);
}

test "HungarianSolver: column reduction" {
    const allocator = std.testing.allocator;
    
    var solver = HungarianSolver.init(allocator);
    
    var matrix = try allocator.alloc([]f32, 2);
    matrix[0] = try allocator.alloc(f32, 2);
    matrix[1] = try allocator.alloc(f32, 2);
    
    matrix[0][0] = 0.0;
    matrix[0][1] = 5.0;
    matrix[1][0] = 0.0;
    matrix[1][1] = 5.0;
    
    solver.subtractColMins(matrix);
    
    // Col 0 min was 0, Col 1 min was 5
    try std.testing.expectEqual(@as(f32, 0.0), matrix[0][0]);
    try std.testing.expectEqual(@as(f32, 0.0), matrix[0][1]);
    try std.testing.expectEqual(@as(f32, 0.0), matrix[1][0]);
    try std.testing.expectEqual(@as(f32, 0.0), matrix[1][1]);
    
    allocator.free(matrix[0]);
    allocator.free(matrix[1]);
    allocator.free(matrix);
}

test "HungarianSolver: empty matrix" {
    const allocator = std.testing.allocator;
    
    var solver = HungarianSolver.init(allocator);
    
    var matrix = try allocator.alloc([]f32, 0);
    defer allocator.free(matrix);
    
    const assignments = try solver.solve(matrix);
    defer allocator.free(assignments);
    
    try std.testing.expectEqual(@as(usize, 0), assignments.len);
}

test "HungarianSolver: 3x3 assignment" {
    const allocator = std.testing.allocator;
    
    var solver = HungarianSolver.init(allocator);
    
    var matrix = try allocator.alloc([]f32, 3);
    for (matrix, 0..) |*row, i| {
        row.* = try allocator.alloc(f32, 3);
    }
    
    // Classic example
    matrix[0][0] = 90.0; matrix[0][1] = 75.0; matrix[0][2] = 75.0;
    matrix[1][0] = 35.0; matrix[1][1] = 85.0; matrix[1][2] = 55.0;
    matrix[2][0] = 125.0; matrix[2][1] = 95.0; matrix[2][2] = 90.0;
    
    const assignments = try solver.solve(matrix);
    defer allocator.free(assignments);
    
    // Verify we got 3 assignments
    var assigned_count: usize = 0;
    for (assignments) |a| {
        if (a != null) assigned_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), assigned_count);
    
    // Cleanup
    for (matrix) |row| {
        allocator.free(row);
    }
    allocator.free(matrix);
}
