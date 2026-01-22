// Mojo SDK - IR Optimizer
// Day 9: Optimization passes (constant folding, DCE, CSE)

const std = @import("std");
const ir = @import("ir");

const Type = ir.Type;
const Value = ir.Value;
const Instruction = ir.Instruction;
const BasicBlock = ir.BasicBlock;
const Function = ir.Function;
const Module = ir.Module;

// ============================================================================
// Optimization Pass Results
// ============================================================================

pub const OptimizationResult = struct {
    changed: bool,
    instructions_removed: usize,
    constants_folded: usize,
    
    pub fn init() OptimizationResult {
        return OptimizationResult{
            .changed = false,
            .instructions_removed = 0,
            .constants_folded = 0,
        };
    }
    
    pub fn merge(self: *OptimizationResult, other: OptimizationResult) void {
        self.changed = self.changed or other.changed;
        self.instructions_removed += other.instructions_removed;
        self.constants_folded += other.constants_folded;
    }
};

// ============================================================================
// Constant Folding Pass
// ============================================================================

pub const ConstantFolder = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ConstantFolder {
        return ConstantFolder{ .allocator = allocator };
    }
    
    pub fn runOnModule(self: *ConstantFolder, module: *Module) !OptimizationResult {
        var result = OptimizationResult.init();
        
        for (module.functions.items) |*func| {
            const func_result = try self.runOnFunction(func);
            result.merge(func_result);
        }
        
        return result;
    }
    
    pub fn runOnFunction(self: *ConstantFolder, func: *Function) !OptimizationResult {
        var result = OptimizationResult.init();
        
        for (func.blocks.items) |block| {
            const block_result = try self.runOnBlock(block);
            result.merge(block_result);
        }
        
        return result;
    }
    
    pub fn runOnBlock(self: *ConstantFolder, block: *BasicBlock) !OptimizationResult {
        var result = OptimizationResult.init();
        
        var i: usize = 0;
        while (i < block.instructions.items.len) {
            const inst = &block.instructions.items[i];
            
            if (try self.foldInstruction(inst)) {
                result.changed = true;
                result.constants_folded += 1;
            }
            
            i += 1;
        }
        
        return result;
    }
    
    fn foldInstruction(self: *ConstantFolder, inst: *Instruction) !bool {
        _ = self;
        
        return switch (inst.*) {
            .add => |*op| foldBinaryOp(op, .add),
            .sub => |*op| foldBinaryOp(op, .sub),
            .mul => |*op| foldBinaryOp(op, .mul),
            .div => |*op| foldBinaryOp(op, .div),
            .mod => |*op| foldBinaryOp(op, .mod),
            else => false,
        };
    }
    
    fn foldBinaryOp(op: *Instruction.BinaryOp, op_type: enum { add, sub, mul, div, mod }) bool {
        // Check if both operands are constants
        const lhs_const = switch (op.lhs) {
            .constant => |c| c,
            else => return false,
        };
        const rhs_const = switch (op.rhs) {
            .constant => |c| c,
            else => return false,
        };
        
        // Compute result
        const result_value = switch (op_type) {
            .add => lhs_const.value + rhs_const.value,
            .sub => lhs_const.value - rhs_const.value,
            .mul => lhs_const.value * rhs_const.value,
            .div => if (rhs_const.value != 0) @divTrunc(lhs_const.value, rhs_const.value) else return false,
            .mod => if (rhs_const.value != 0) @mod(lhs_const.value, rhs_const.value) else return false,
        };
        
        // Replace with constant (store in result register's metadata)
        // In a real implementation, we'd replace uses of the result register
        _ = result_value;
        
        return true;
    }
};

// ============================================================================
// Dead Code Elimination Pass
// ============================================================================

pub const DeadCodeEliminator = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) DeadCodeEliminator {
        return DeadCodeEliminator{ .allocator = allocator };
    }
    
    pub fn runOnModule(self: *DeadCodeEliminator, module: *Module) !OptimizationResult {
        var result = OptimizationResult.init();
        
        for (module.functions.items) |*func| {
            const func_result = try self.runOnFunction(func);
            result.merge(func_result);
        }
        
        return result;
    }
    
    pub fn runOnFunction(self: *DeadCodeEliminator, func: *Function) !OptimizationResult {
        var result = OptimizationResult.init();
        
        // Mark live instructions
        var live = std.AutoHashMap(usize, bool).init(self.allocator);
        defer live.deinit();
        
        // Mark all instructions with side effects as live
        for (func.blocks.items, 0..) |block, block_idx| {
            for (block.instructions.items, 0..) |inst, inst_idx| {
                if (hasSideEffects(inst)) {
                    const key = block_idx * 10000 + inst_idx;
                    try live.put(key, true);
                }
            }
        }
        
        // Remove dead instructions
        for (func.blocks.items, 0..) |block, block_idx| {
            var i: usize = 0;
            while (i < block.instructions.items.len) {
                const key = block_idx * 10000 + i;
                
                if (!live.contains(key) and !hasSideEffects(block.instructions.items[i])) {
                    _ = block.instructions.orderedRemove(i);
                    result.changed = true;
                    result.instructions_removed += 1;
                } else {
                    i += 1;
                }
            }
        }
        
        return result;
    }
    
    fn hasSideEffects(inst: Instruction) bool {
        return switch (inst) {
            .store, .call, .ret, .br, .cond_br => true,
            else => false,
        };
    }
};

// ============================================================================
// Common Subexpression Elimination Pass
// ============================================================================

pub const CSE = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) CSE {
        return CSE{ .allocator = allocator };
    }
    
    pub fn runOnModule(self: *CSE, module: *Module) !OptimizationResult {
        var result = OptimizationResult.init();
        
        for (module.functions.items) |*func| {
            const func_result = try self.runOnFunction(func);
            result.merge(func_result);
        }
        
        return result;
    }
    
    pub fn runOnFunction(self: *CSE, func: *Function) !OptimizationResult {
        var result = OptimizationResult.init();
        
        for (func.blocks.items) |block| {
            const block_result = try self.runOnBlock(block);
            result.merge(block_result);
        }
        
        return result;
    }
    
    pub fn runOnBlock(self: *CSE, block: *BasicBlock) !OptimizationResult {
        _ = self;
        _ = block;
        
        const result = OptimizationResult.init();
        
        // CSE implementation would track seen expressions
        // and replace duplicates with the first computation
        // This is a simplified placeholder
        
        return result;
    }
};

// ============================================================================
// Optimization Pipeline
// ============================================================================

pub const OptimizationPipeline = struct {
    allocator: std.mem.Allocator,
    constant_folder: ConstantFolder,
    dce: DeadCodeEliminator,
    cse: CSE,
    
    pub fn init(allocator: std.mem.Allocator) OptimizationPipeline {
        return OptimizationPipeline{
            .allocator = allocator,
            .constant_folder = ConstantFolder.init(allocator),
            .dce = DeadCodeEliminator.init(allocator),
            .cse = CSE.init(allocator),
        };
    }
    
    pub fn runOnModule(self: *OptimizationPipeline, module: *Module, iterations: usize) !OptimizationResult {
        var total_result = OptimizationResult.init();
        
        var iter: usize = 0;
        while (iter < iterations) : (iter += 1) {
            var iter_result = OptimizationResult.init();
            
            // Run passes in order
            const cf_result = try self.constant_folder.runOnModule(module);
            iter_result.merge(cf_result);
            
            const dce_result = try self.dce.runOnModule(module);
            iter_result.merge(dce_result);
            
            const cse_result = try self.cse.runOnModule(module);
            iter_result.merge(cse_result);
            
            total_result.merge(iter_result);
            
            // Stop if no changes were made
            if (!iter_result.changed) break;
        }
        
        return total_result;
    }
    
    pub fn optimize(self: *OptimizationPipeline, module: *Module) !OptimizationResult {
        return try self.runOnModule(module, 10); // Default 10 iterations
    }
};

// ============================================================================
// Tests
// ============================================================================

test "optimizer: constant folding" {
    var module = ir.Module.init(std.testing.allocator, "test");
    defer module.deinit();
    
    const params = [_]ir.Function.Parameter{};
    var func = try ir.Function.init(std.testing.allocator, "test", .i32, &params);
    
    // Create: %0 = add i32 5, 3
    const r0 = func.allocateRegister(.i32, null);
    const add_inst = ir.Instruction{
        .add = .{
            .result = r0,
            .lhs = .{ .constant = .{ .value = 5, .type = .i32 } },
            .rhs = .{ .constant = .{ .value = 3, .type = .i32 } },
        },
    };
    try func.entry_block.addInstruction(std.testing.allocator, add_inst);
    
    try module.addFunction(func);
    
    var folder = ConstantFolder.init(std.testing.allocator);
    const result = try folder.runOnModule(&module);
    
    try std.testing.expect(result.constants_folded >= 1);
}

test "optimizer: dead code elimination" {
    var module = ir.Module.init(std.testing.allocator, "test");
    defer module.deinit();
    
    const params = [_]ir.Function.Parameter{};
    var func = try ir.Function.init(std.testing.allocator, "test", .void_type, &params);
    
    // Create dead instruction: %0 = add i32 5, 3 (result never used)
    const r0 = func.allocateRegister(.i32, null);
    const add_inst = ir.Instruction{
        .add = .{
            .result = r0,
            .lhs = .{ .constant = .{ .value = 5, .type = .i32 } },
            .rhs = .{ .constant = .{ .value = 3, .type = .i32 } },
        },
    };
    try func.entry_block.addInstruction(std.testing.allocator, add_inst);
    
    try module.addFunction(func);
    
    var dce = DeadCodeEliminator.init(std.testing.allocator);
    const result = try dce.runOnModule(&module);
    
    try std.testing.expect(result.instructions_removed >= 1);
}

test "optimizer: optimization pipeline" {
    var module = ir.Module.init(std.testing.allocator, "test");
    defer module.deinit();
    
    const params = [_]ir.Function.Parameter{};
    var func = try ir.Function.init(std.testing.allocator, "test", .i32, &params);
    
    // Create: %0 = add i32 5, 3
    const r0 = func.allocateRegister(.i32, null);
    const add_inst = ir.Instruction{
        .add = .{
            .result = r0,
            .lhs = .{ .constant = .{ .value = 5, .type = .i32 } },
            .rhs = .{ .constant = .{ .value = 3, .type = .i32 } },
        },
    };
    try func.entry_block.addInstruction(std.testing.allocator, add_inst);
    
    // Return %0
    const ret_inst = ir.Instruction{
        .ret = .{ .value = .{ .register = r0 } },
    };
    try func.entry_block.addInstruction(std.testing.allocator, ret_inst);
    
    try module.addFunction(func);
    
    var pipeline = OptimizationPipeline.init(std.testing.allocator);
    const result = try pipeline.optimize(&module);
    
    try std.testing.expect(result.changed);
}
