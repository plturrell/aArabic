// Control Flow Analysis for Borrow Checker
// Day 61: Branch tracking, loops, conditional borrows, early returns

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const borrow_checker = @import("borrow_checker.zig");
const borrow_advanced = @import("borrow_advanced.zig");
const lifetimes = @import("lifetimes.zig");
const BorrowKind = borrow_checker.BorrowKind;
const Borrow = borrow_checker.Borrow;
const BorrowPath = borrow_checker.BorrowPath;
const BorrowScope = borrow_checker.BorrowScope;
const Lifetime = lifetimes.Lifetime;

// ============================================================================
// Control Flow Graph
// ============================================================================

/// Basic block in control flow graph
pub const BasicBlock = struct {
    id: u32,
    instructions: ArrayList(Instruction),
    successors: ArrayList(u32), // Block IDs
    predecessors: ArrayList(u32),
    borrow_state_in: ?BorrowStateSnapshot,
    borrow_state_out: ?BorrowStateSnapshot,
    
    pub const Instruction = union(enum) {
        Borrow: BorrowOp,
        EndBorrow: BorrowPath,
        Move: MoveOp,
        Assign: AssignOp,
        Call: CallOp,
    };
    
    pub const BorrowOp = struct {
        path: BorrowPath,
        kind: BorrowKind,
        lifetime: Lifetime,
    };
    
    pub const MoveOp = struct {
        source: BorrowPath,
        destination: BorrowPath,
    };
    
    pub const AssignOp = struct {
        target: BorrowPath,
    };
    
    pub const CallOp = struct {
        func_name: []const u8,
        args: []const BorrowPath,
    };
    
    pub fn init(allocator: Allocator, id: u32) BasicBlock {
        _ = allocator;
        return BasicBlock{
            .id = id,
            .instructions = ArrayList(Instruction){},
            .successors = ArrayList(u32){},
            .predecessors = ArrayList(u32){},
            .borrow_state_in = null,
            .borrow_state_out = null,
        };
    }
    
    pub fn deinit(self: *BasicBlock, allocator: Allocator) void {
        self.instructions.deinit(allocator);
        self.successors.deinit(allocator);
        self.predecessors.deinit(allocator);
    }
};

/// Snapshot of borrow state at a program point
pub const BorrowStateSnapshot = struct {
    active_borrows: ArrayList(Borrow),
    moved_vars: ArrayList([]const u8),
    
    pub fn init(allocator: Allocator) BorrowStateSnapshot {
        _ = allocator;
        return BorrowStateSnapshot{
            .active_borrows = ArrayList(Borrow){},
            .moved_vars = ArrayList([]const u8){},
        };
    }
    
    pub fn deinit(self: *BorrowStateSnapshot, allocator: Allocator) void {
        self.active_borrows.deinit(allocator);
        self.moved_vars.deinit(allocator);
    }
    
    pub fn clone(self: BorrowStateSnapshot, allocator: Allocator) !BorrowStateSnapshot {
        var new = BorrowStateSnapshot.init(allocator);
        try new.active_borrows.appendSlice(allocator, self.active_borrows.items);
        try new.moved_vars.appendSlice(allocator, self.moved_vars.items);
        return new;
    }
};

/// Control flow graph
pub const ControlFlowGraph = struct {
    allocator: Allocator,
    blocks: ArrayList(BasicBlock),
    entry_block: u32,
    exit_blocks: ArrayList(u32),
    
    pub fn init(allocator: Allocator) ControlFlowGraph {
        return ControlFlowGraph{
            .allocator = allocator,
            .blocks = ArrayList(BasicBlock){},
            .entry_block = 0,
            .exit_blocks = ArrayList(u32){},
        };
    }
    
    pub fn deinit(self: *ControlFlowGraph) void {
        for (self.blocks.items) |*block| {
            block.deinit(self.allocator);
        }
        self.blocks.deinit(self.allocator);
        self.exit_blocks.deinit(self.allocator);
    }
    
    pub fn addBlock(self: *ControlFlowGraph) !u32 {
        const id = @as(u32, @intCast(self.blocks.items.len));
        const block = BasicBlock.init(self.allocator, id);
        try self.blocks.append(self.allocator, block);
        return id;
    }
    
    pub fn addEdge(self: *ControlFlowGraph, from: u32, to: u32) !void {
        try self.blocks.items[from].successors.append(self.allocator, to);
        try self.blocks.items[to].predecessors.append(self.allocator, from);
    }
};

// ============================================================================
// Branch Analysis
// ============================================================================

/// Analyzes branches (if/else) in control flow
pub const BranchAnalyzer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) BranchAnalyzer {
        return BranchAnalyzer{ .allocator = allocator };
    }
    
    /// Merge borrow states from multiple branches
    pub fn mergeBranchStates(
        self: *BranchAnalyzer,
        states: []const BorrowStateSnapshot,
    ) !BorrowStateSnapshot {
        if (states.len == 0) {
            return BorrowStateSnapshot.init(self.allocator);
        }
        
        if (states.len == 1) {
            return try states[0].clone(self.allocator);
        }
        
        var merged = BorrowStateSnapshot.init(self.allocator);
        
        // Only include borrows that exist in ALL branches
        for (states[0].active_borrows.items) |borrow| {
            var in_all = true;
            for (states[1..]) |state| {
                var found = false;
                for (state.active_borrows.items) |b| {
                    if (std.mem.eql(u8, b.path.root, borrow.path.root)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    in_all = false;
                    break;
                }
            }
            if (in_all) {
                try merged.active_borrows.append(self.allocator, borrow);
            }
        }
        
        // Include moves that happen in ALL branches
        for (states[0].moved_vars.items) |var_name| {
            var in_all = true;
            for (states[1..]) |state| {
                var found = false;
                for (state.moved_vars.items) |v| {
                    if (std.mem.eql(u8, v, var_name)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    in_all = false;
                    break;
                }
            }
            if (in_all) {
                try merged.moved_vars.append(self.allocator, var_name);
            }
        }
        
        return merged;
    }
    
    /// Check if borrow is valid in conditional context
    pub fn checkConditionalBorrow(
        self: *BranchAnalyzer,
        borrow: Borrow,
        condition_state: BorrowStateSnapshot,
    ) bool {
        _ = self;
        
        // Check if value is moved in condition
        for (condition_state.moved_vars.items) |var_name| {
            if (std.mem.eql(u8, var_name, borrow.path.root)) {
                return false; // Cannot borrow moved value
            }
        }
        
        // Check for conflicting borrows
        for (condition_state.active_borrows.items) |active| {
            if (borrow.path.overlaps(active.path)) {
                if (borrow.kind == .Mutable or active.kind == .Mutable) {
                    return false;
                }
            }
        }
        
        return true;
    }
};

// ============================================================================
// Loop Analysis
// ============================================================================

/// Analyzes loops in control flow
pub const LoopAnalyzer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) LoopAnalyzer {
        return LoopAnalyzer{ .allocator = allocator };
    }
    
    /// Detect loop back-edges in CFG
    pub fn detectLoops(self: *LoopAnalyzer, cfg: *ControlFlowGraph) !ArrayList(Loop) {
        var loops = ArrayList(Loop){};
        
        // Simple loop detection: look for back-edges
        for (cfg.blocks.items) |block| {
            for (block.successors.items) |succ_id| {
                if (succ_id <= block.id) {
                    // Back-edge detected (successor is earlier in CFG)
                    const loop = Loop{
                        .header = succ_id,
                        .body = ArrayList(u32){},
                        .exit_blocks = ArrayList(u32){},
                    };
                    try loops.append(self.allocator, loop);
                }
            }
        }
        
        return loops;
    }
    
    /// Check loop invariants for borrows
    pub fn checkLoopInvariants(
        self: *LoopAnalyzer,
        loop: Loop,
        cfg: *ControlFlowGraph,
    ) !bool {
        _ = self;
        
        const header_state = cfg.blocks.items[loop.header].borrow_state_in orelse return true;
        
        // Verify borrows are consistent across all iterations
        for (loop.body.items) |block_id| {
            const block = &cfg.blocks.items[block_id];
            if (block.borrow_state_out) |out_state| {
                // Check that borrows entering loop match borrows exiting
                if (out_state.active_borrows.items.len != header_state.active_borrows.items.len) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    pub const Loop = struct {
        header: u32, // Loop header block
        body: ArrayList(u32), // Body blocks
        exit_blocks: ArrayList(u32), // Exit points
    };
};

// ============================================================================
// Early Return Analysis
// ============================================================================

/// Analyzes early returns and their impact on borrows
pub const EarlyReturnAnalyzer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) EarlyReturnAnalyzer {
        return EarlyReturnAnalyzer{ .allocator = allocator };
    }
    
    /// Check if early return is safe with active borrows
    pub fn checkEarlyReturn(
        self: *EarlyReturnAnalyzer,
        return_point: u32,
        cfg: *ControlFlowGraph,
    ) !bool {
        _ = self;
        
        const block = &cfg.blocks.items[return_point];
        if (block.borrow_state_out) |state| {
            // All borrows must be ended before return
            if (state.active_borrows.items.len > 0) {
                return false; // Cannot return with active borrows
            }
        }
        
        return true;
    }
    
    /// Find all return points in CFG
    pub fn findReturnPoints(self: *EarlyReturnAnalyzer, cfg: *ControlFlowGraph) !ArrayList(u32) {
        var returns = ArrayList(u32){};
        
        for (cfg.blocks.items, 0..) |block, i| {
            if (block.successors.items.len == 0) {
                // Block with no successors is a return
                try returns.append(self.allocator, @intCast(i));
            }
        }
        
        return returns;
    }
};

// ============================================================================
// Dataflow Analysis
// ============================================================================

/// Performs dataflow analysis for borrows
pub const DataflowAnalyzer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DataflowAnalyzer {
        return DataflowAnalyzer{ .allocator = allocator };
    }
    
    /// Compute borrow states for all blocks using worklist algorithm
    pub fn analyze(self: *DataflowAnalyzer, cfg: *ControlFlowGraph) !void {
        var worklist = ArrayList(u32){};
        defer worklist.deinit(self.allocator);
        
        // Initialize worklist with all blocks
        for (cfg.blocks.items, 0..) |_, i| {
            try worklist.append(self.allocator, @intCast(i));
        }
        
        // Initialize entry block state
        cfg.blocks.items[cfg.entry_block].borrow_state_in = BorrowStateSnapshot.init(self.allocator);
        
        // Iterate until fixpoint
        while (worklist.items.len > 0) {
            const block_id = worklist.pop() orelse break;
            const changed = try self.analyzeBlock(cfg, block_id);
            
            if (changed) {
                // Add successors to worklist
                for (cfg.blocks.items[block_id].successors.items) |succ| {
                    try worklist.append(self.allocator, succ);
                }
            }
        }
    }
    
    fn analyzeBlock(self: *DataflowAnalyzer, cfg: *ControlFlowGraph, block_id: u32) !bool {
        _ = self;
        var block = &cfg.blocks.items[block_id];
        
        // Merge predecessor states
        if (block.predecessors.items.len > 0) {
            var pred_states = ArrayList(BorrowStateSnapshot){};
            defer pred_states.deinit(cfg.allocator);
            
            for (block.predecessors.items) |pred_id| {
                if (cfg.blocks.items[pred_id].borrow_state_out) |state| {
                    try pred_states.append(cfg.allocator, state);
                }
            }
            
            if (pred_states.items.len > 0) {
                var analyzer = BranchAnalyzer.init(cfg.allocator);
                const merged = try analyzer.mergeBranchStates(pred_states.items);
                block.borrow_state_in = merged;
            }
        }
        
        // For now, just copy input to output
        if (block.borrow_state_in) |in_state| {
            block.borrow_state_out = try in_state.clone(cfg.allocator);
            return true;
        }
        
        return false;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "basic block creation" {
    const allocator = std.testing.allocator;
    
    var block = BasicBlock.init(allocator, 0);
    defer block.deinit(allocator);
    
    try std.testing.expectEqual(@as(u32, 0), block.id);
    try std.testing.expectEqual(@as(usize, 0), block.instructions.items.len);
}

test "control flow graph" {
    const allocator = std.testing.allocator;
    
    var cfg = ControlFlowGraph.init(allocator);
    defer cfg.deinit();
    
    const b0 = try cfg.addBlock();
    const b1 = try cfg.addBlock();
    
    try cfg.addEdge(b0, b1);
    
    try std.testing.expectEqual(@as(usize, 1), cfg.blocks.items[b0].successors.items.len);
    try std.testing.expectEqual(@as(usize, 1), cfg.blocks.items[b1].predecessors.items.len);
}

test "branch merge empty states" {
    const allocator = std.testing.allocator;
    
    var analyzer = BranchAnalyzer.init(allocator);
    
    var state1 = BorrowStateSnapshot.init(allocator);
    defer state1.deinit(allocator);
    
    var state2 = BorrowStateSnapshot.init(allocator);
    defer state2.deinit(allocator);
    
    const states = [_]BorrowStateSnapshot{ state1, state2 };
    var merged = try analyzer.mergeBranchStates(&states);
    defer merged.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 0), merged.active_borrows.items.len);
}

test "branch conditional borrow check" {
    const allocator = std.testing.allocator;
    
    var analyzer = BranchAnalyzer.init(allocator);
    
    var condition_state = BorrowStateSnapshot.init(allocator);
    defer condition_state.deinit(allocator);
    
    const lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt.name);
    
    var path = try BorrowPath.init(allocator, "x");
    defer path.deinit(allocator);
    
    const borrow = Borrow{
        .kind = .Shared,
        .lifetime = lt,
        .path = path,
        .location = .{ .line = 1, .column = 1 },
    };
    
    try std.testing.expect(analyzer.checkConditionalBorrow(borrow, condition_state));
}

test "loop detection" {
    const allocator = std.testing.allocator;
    
    var cfg = ControlFlowGraph.init(allocator);
    defer cfg.deinit();
    
    const b0 = try cfg.addBlock();
    const b1 = try cfg.addBlock();
    
    // Create loop: b1 -> b0
    try cfg.addEdge(b0, b1);
    try cfg.addEdge(b1, b0); // Back-edge
    
    var analyzer = LoopAnalyzer.init(allocator);
    var loops = try analyzer.detectLoops(&cfg);
    defer loops.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 1), loops.items.len);
}

test "early return finder" {
    const allocator = std.testing.allocator;
    
    var cfg = ControlFlowGraph.init(allocator);
    defer cfg.deinit();
    
    const b0 = try cfg.addBlock();
    const b1 = try cfg.addBlock();
    
    // Connect b0 to b1, so b1 is the only return
    try cfg.addEdge(b0, b1);
    
    var analyzer = EarlyReturnAnalyzer.init(allocator);
    var returns = try analyzer.findReturnPoints(&cfg);
    defer returns.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 1), returns.items.len);
    try std.testing.expectEqual(b1, returns.items[0]);
}

test "dataflow analysis initialization" {
    const allocator = std.testing.allocator;
    
    var cfg = ControlFlowGraph.init(allocator);
    defer cfg.deinit();
    
    _ = try cfg.addBlock();
    
    var analyzer = DataflowAnalyzer.init(allocator);
    try analyzer.analyze(&cfg);
    
    try std.testing.expect(cfg.blocks.items[0].borrow_state_in != null);
}

test "borrow state snapshot clone" {
    const allocator = std.testing.allocator;
    
    var original = BorrowStateSnapshot.init(allocator);
    defer original.deinit(allocator);
    
    try original.moved_vars.append(allocator, "x");
    
    var cloned = try original.clone(allocator);
    defer cloned.deinit(allocator);
    
    try std.testing.expectEqual(original.moved_vars.items.len, cloned.moved_vars.items.len);
}

test "loop invariant checking" {
    const allocator = std.testing.allocator;
    
    var cfg = ControlFlowGraph.init(allocator);
    defer cfg.deinit();
    
    const b0 = try cfg.addBlock();
    cfg.blocks.items[b0].borrow_state_in = BorrowStateSnapshot.init(allocator);
    cfg.blocks.items[b0].borrow_state_out = BorrowStateSnapshot.init(allocator);
    
    var analyzer = LoopAnalyzer.init(allocator);
    const loop = LoopAnalyzer.Loop{
        .header = b0,
        .body = ArrayList(u32){},
        .exit_blocks = ArrayList(u32){},
    };
    
    const valid = try analyzer.checkLoopInvariants(loop, &cfg);
    try std.testing.expect(valid);
}

test "early return with no active borrows" {
    const allocator = std.testing.allocator;
    
    var cfg = ControlFlowGraph.init(allocator);
    defer cfg.deinit();
    
    const b0 = try cfg.addBlock();
    cfg.blocks.items[b0].borrow_state_out = BorrowStateSnapshot.init(allocator);
    
    var analyzer = EarlyReturnAnalyzer.init(allocator);
    const safe = try analyzer.checkEarlyReturn(b0, &cfg);
    
    try std.testing.expect(safe);
}
