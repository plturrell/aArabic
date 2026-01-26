// Sudoku Grid Representation
// Multiple representations for different algorithm strategies

const std = @import("std");

pub const SudokuGrid = struct {
    // Flat array representation (best cache locality)
    cells: [81]u8,
    
    // Bitmask representation for fast constraint checking
    bitmask_rows: [9]u16,
    bitmask_cols: [9]u16,
    bitmask_boxes: [9]u16,
    
    // Domains for constraint propagation
    domains: [81]u16,
    
    pub fn init() SudokuGrid {
        const grid = SudokuGrid{
            .cells = [_]u8{0} ** 81,
            .bitmask_rows = [_]u16{0x1FF} ** 9,  // Bits 1-9 set
            .bitmask_cols = [_]u16{0x1FF} ** 9,
            .bitmask_boxes = [_]u16{0x1FF} ** 9,
            .domains = [_]u16{0x1FF} ** 81,
        };
        return grid;
    }
    
    pub fn fromString(puzzle: []const u8) !SudokuGrid {
        if (puzzle.len != 81) return error.InvalidPuzzleLength;
        
        var grid = init();
        for (puzzle, 0..) |ch, i| {
            if (ch >= '1' and ch <= '9') {
                const num = ch - '0';
                try grid.place(i, num);
            }
        }
        return grid;
    }
    
    pub fn place(self: *SudokuGrid, index: usize, num: u8) !void {
        if (num < 1 or num > 9) return error.InvalidNumber;
        if (index >= 81) return error.InvalidIndex;
        
        const row = index / 9;
        const col = index % 9;
        const box = (row / 3) * 3 + (col / 3);
        
        self.cells[index] = num;
        
        // Update bitmasks
        const bit: u16 = @as(u16, 1) << @intCast(num - 1);
        self.bitmask_rows[row] &= ~bit;
        self.bitmask_cols[col] &= ~bit;
        self.bitmask_boxes[box] &= ~bit;
        
        // Update domains
        self.domains[index] = 0;
    }
    
    pub fn isValid(self: *const SudokuGrid, index: usize, num: u8) bool {
        if (num < 1 or num > 9 or index >= 81) return false;
        
        const row = index / 9;
        const col = index % 9;
        const box = (row / 3) * 3 + (col / 3);
        
        const bit: u16 = @as(u16, 1) << @intCast(num - 1);
        
        return (self.bitmask_rows[row] & bit) != 0 and
               (self.bitmask_cols[col] & bit) != 0 and
               (self.bitmask_boxes[box] & bit) != 0;
    }
    
    pub fn isSolved(self: *const SudokuGrid) bool {
        for (self.cells) |cell| {
            if (cell == 0) return false;
        }
        return true;
    }
    
    pub fn clone(self: *const SudokuGrid) SudokuGrid {
        return self.*;
    }
    
    pub fn print(self: *const SudokuGrid, writer: anytype) !void {
        try writer.writeAll("\n");
        for (0..9) |row| {
            if (row % 3 == 0 and row != 0) {
                try writer.writeAll("------+-------+------\n");
            }
            for (0..9) |col| {
                if (col % 3 == 0 and col != 0) {
                    try writer.writeAll("| ");
                }
                const cell = self.cells[row * 9 + col];
                if (cell == 0) {
                    try writer.writeAll(". ");
                } else {
                    try writer.print("{d} ", .{cell});
                }
            }
            try writer.writeAll("\n");
        }
        try writer.writeAll("\n");
    }
};

pub fn countSetBits(mask: u16) u8 {
    var count: u8 = 0;
    var m = mask;
    while (m != 0) {
        m &= m - 1;  // Brian Kernighan's algorithm
        count += 1;
    }
    return count;
}

pub fn getSetBits(mask: u16) [9]u8 {
    var bits: [9]u8 = undefined;
    var count: usize = 0;
    var m = mask;
    var value: u8 = 1;
    
    while (m != 0 and count < 9) {
        if (m & 1 != 0) {
            bits[count] = value;
            count += 1;
        }
        m >>= 1;
        value += 1;
    }
    
    return bits;
}