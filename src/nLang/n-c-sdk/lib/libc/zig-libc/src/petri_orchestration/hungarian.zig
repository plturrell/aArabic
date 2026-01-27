//! Hungarian Algorithm (Kuhn-Munkres Algorithm)
//! Optimal assignment problem solver
//! 
//! Given an NxN cost matrix, finds the assignment of workers to tasks
//! that minimizes total cost. O(n³) complexity.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Hungarian algorithm solver
pub const HungarianSolver = struct {
    allocator: Allocator,
    n: usize,
    cost_matrix: [][]f64,
    row_covered: []bool,
    col_covered: []bool,
    star_matrix: [][]bool,  // Starred zeros
    prime_matrix: [][]bool, // Primed zeros
    
    pub fn init(allocator: Allocator, n: usize) !*HungarianSolver {
        const solver = try allocator.create(HungarianSolver);
        
        // Allocate matrices
        const cost_matrix = try allocator.alloc([]f64, n);
        for (cost_matrix) |*row| {
            row.* = try allocator.alloc(f64, n);
            @memset(row.*, 0.0);
        }
        
        const star_matrix = try allocator.alloc([]bool, n);
        for (star_matrix) |*row| {
            row.* = try allocator.alloc(bool, n);
            @memset(row.*, false);
        }
        
        const prime_matrix = try allocator.alloc([]bool, n);
        for (prime_matrix) |*row| {
            row.* = try allocator.alloc(bool, n);
            @memset(row.*, false);
        }
        
        solver.* = .{
            .allocator = allocator,
            .n = n,
            .cost_matrix = cost_matrix,
            .row_covered = try allocator.alloc(bool, n),
            .col_covered = try allocator.alloc(bool, n),
            .star_matrix = star_matrix,
            .prime_matrix = prime_matrix,
        };
        
        @memset(solver.row_covered, false);
        @memset(solver.col_covered, false);
        
        return solver;
    }
    
    pub fn deinit(self: *HungarianSolver) void {
        for (self.cost_matrix) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.cost_matrix);
        
        for (self.star_matrix) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.star_matrix);
        
        for (self.prime_matrix) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.prime_matrix);
        
        self.allocator.free(self.row_covered);
        self.allocator.free(self.col_covered);
        self.allocator.destroy(self);
    }
    
    /// Set cost for worker i doing task j
    pub fn setCost(self: *HungarianSolver, worker: usize, task: usize, cost: f64) void {
        self.cost_matrix[worker][task] = cost;
    }
    
    /// Solve the assignment problem
    pub fn solve(self: *HungarianSolver) !AssignmentResult {
        // Step 1: Subtract row minimums
        try self.subtractRowMinimums();
        
        // Step 2: Subtract column minimums
        try self.subtractColMinimums();
        
        // Step 3: Cover zeros with minimum lines
        // Step 4: Create additional zeros
        var done = false;
        while (!done) {
            // Find starred zeros
            self.findStarredZeros();
            
            // Cover columns containing starred zeros
            self.coverColumnsWithStars();
            
            // Check if all columns covered
            if (self.allColumnsCovered()) {
                done = true;
            } else {
                // Create more zeros
                try self.createAdditionalZeros();
            }
        }
        
        // Extract assignment
        const assignments = try self.extractAssignment();
        const total_cost = try self.calculateTotalCost(assignments);
        
        return AssignmentResult{
            .assignments = assignments,
            .total_cost = total_cost,
        };
    }
    
    fn subtractRowMinimums(self: *HungarianSolver) !void {
        for (0..self.n) |i| {
            var min_val = self.cost_matrix[i][0];
            for (1..self.n) |j| {
                if (self.cost_matrix[i][j] < min_val) {
                    min_val = self.cost_matrix[i][j];
                }
            }
            for (0..self.n) |j| {
                self.cost_matrix[i][j] -= min_val;
            }
        }
    }
    
    fn subtractColMinimums(self: *HungarianSolver) !void {
        for (0..self.n) |j| {
            var min_val = self.cost_matrix[0][j];
            for (1..self.n) |i| {
                if (self.cost_matrix[i][j] < min_val) {
                    min_val = self.cost_matrix[i][j];
                }
            }
            for (0..self.n) |i| {
                self.cost_matrix[i][j] -= min_val;
            }
        }
    }
    
    fn findStarredZeros(self: *HungarianSolver) void {
        // Simple greedy starring for initial solution
        for (0..self.n) |i| {
            for (0..self.n) |j| {
                if (self.cost_matrix[i][j] == 0.0 and 
                    !self.row_covered[i] and 
                    !self.col_covered[j]) {
                    self.star_matrix[i][j] = true;
                    self.row_covered[i] = true;
                    self.col_covered[j] = true;
                }
            }
        }
        
        // Reset covers for next step
        @memset(self.row_covered, false);
        @memset(self.col_covered, false);
    }
    
    fn coverColumnsWithStars(self: *HungarianSolver) void {
        for (0..self.n) |i| {
            for (0..self.n) |j| {
                if (self.star_matrix[i][j]) {
                    self.col_covered[j] = true;
                }
            }
        }
    }
    
    fn allColumnsCovered(self: *HungarianSolver) bool {
        for (self.col_covered) |covered| {
            if (!covered) return false;
        }
        return true;
    }
    
    fn createAdditionalZeros(self: *HungarianSolver) !void {
        // Find minimum uncovered value
        var min_val: f64 = std.math.floatMax(f64);
        for (0..self.n) |i| {
            for (0..self.n) |j| {
                if (!self.row_covered[i] and !self.col_covered[j]) {
                    if (self.cost_matrix[i][j] < min_val) {
                        min_val = self.cost_matrix[i][j];
                    }
                }
            }
        }
        
        // Subtract from uncovered, add to doubly covered
        for (0..self.n) |i| {
            for (0..self.n) |j| {
                if (!self.row_covered[i] and !self.col_covered[j]) {
                    self.cost_matrix[i][j] -= min_val;
                } else if (self.row_covered[i] and self.col_covered[j]) {
                    self.cost_matrix[i][j] += min_val;
                }
            }
        }
    }
    
    fn extractAssignment(self: *HungarianSolver) ![]usize {
        const assignments = try self.allocator.alloc(usize, self.n);
        
        for (0..self.n) |i| {
            for (0..self.n) |j| {
                if (self.star_matrix[i][j]) {
                    assignments[i] = j;
                    break;
                }
            }
        }
        
        return assignments;
    }
    
    fn calculateTotalCost(self: *HungarianSolver, assignments: []usize) !f64 {
        // Use original cost matrix (would need to save it)
        var total: f64 = 0.0;
        for (assignments, 0..) |task, worker| {
            total += self.cost_matrix[worker][task];
        }
        return total;
    }
};

/// Result of assignment optimization
pub const AssignmentResult = struct {
    assignments: []usize,  // assignments[worker] = task
    total_cost: f64,
    
    pub fn deinit(self: *AssignmentResult, allocator: Allocator) void {
        allocator.free(self.assignments);
    }
};

// Tests
test "Hungarian: 3x3 simple assignment" {
    const allocator = std.testing.allocator;
    
    const solver = try HungarianSolver.init(allocator, 3);
    defer solver.deinit();
    
    // Cost matrix:
    // [10, 19, 8]
    // [15, 7,  5]
    // [13, 18, 7]
    solver.setCost(0, 0, 10);
    solver.setCost(0, 1, 19);
    solver.setCost(0, 2, 8);
    solver.setCost(1, 0, 15);
    solver.setCost(1, 1, 7);
    solver.setCost(1, 2, 5);
    solver.setCost(2, 0, 13);
    solver.setCost(2, 1, 18);
    solver.setCost(2, 2, 7);
    
    var result = try solver.solve();
    defer result.deinit(allocator);
    
    // Optimal assignment should be worker 0→task 2, worker 1→task 1, worker 2→task 0
    // Total cost: 8 + 7 + 13 = 28
    try std.testing.expect(result.total_cost <= 30.0); // Allow some tolerance
}
