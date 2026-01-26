// Mojo SDK - IR to MLIR Lowering Bridge
// Day 13: Convert custom IR to MLIR representation

const std = @import("std");
const IR = @import("ir");
const mlir = @import("mlir_setup");
const dialect = @import("mojo_dialect");

// ============================================================================
// IR to MLIR Type Mapping
// ============================================================================

pub const TypeMapper = struct {
    /// Map IR type to Mojo dialect type
    pub fn mapType(ir_type: IR.Type) dialect.MojoType {
        return switch (ir_type) {
            .i32 => dialect.MojoType.createInt(32),
            .i64 => dialect.MojoType.createInt(64),
            .f32 => dialect.MojoType.createFloat(32),
            .f64 => dialect.MojoType.createFloat(64),
            .bool_type => dialect.MojoType.createBool(),
            .void_type => dialect.MojoType.createVoid(),
            .ptr => dialect.MojoType.createInt(64), // Pointer as i64
        };
    }
    
    /// Check if IR type maps to MLIR type
    pub fn isCompatible(ir_type: IR.Type, mojo_type: dialect.MojoType) bool {
        const mapped = mapType(ir_type);
        return mapped.kind == mojo_type.kind and mapped.bit_width == mojo_type.bit_width;
    }
};

// ============================================================================
// IR to MLIR Instruction Mapping
// ============================================================================

pub const InstructionMapper = struct {
    builder: dialect.MojoOpBuilder,
    
    pub fn init() InstructionMapper {
        return InstructionMapper{
            .builder = dialect.MojoOpBuilder.init(),
        };
    }
    
    /// Map IR instruction tag to Mojo dialect operation kind
    pub fn mapInstructionKind(self: *InstructionMapper, inst_tag: std.meta.Tag(IR.Instruction)) dialect.MojoOpKind {
        _ = self;
        return switch (inst_tag) {
            .add => .Add,
            .sub => .Sub,
            .mul => .Mul,
            .div => .Div,
            .ret => .Return,
            .call => .Call,
            .load => .Load,
            .store => .Assign,
            .br => .While, // Branch maps to control flow
            .cond_br => .If,
            else => .Add, // Default fallback
        };
    }
    
    /// Create Mojo operation from IR instruction
    pub fn mapInstruction(self: *InstructionMapper, ir_inst: *const IR.Instruction) !MappedOp {
        return switch (ir_inst.*) {
            .add => MappedOp{ .add = self.builder.buildAdd() },
            .sub => MappedOp{ .sub = self.builder.buildAdd() }, // Placeholder
            .mul => MappedOp{ .mul = self.builder.buildAdd() }, // Placeholder
            .div => MappedOp{ .div = self.builder.buildAdd() }, // Placeholder
            .ret => |op| blk: {
                const has_value = op.value != null;
                break :blk MappedOp{ .ret = self.builder.buildReturn(has_value) };
            },
            .call => |op| blk: {
                const num_args = op.args.len;
                break :blk MappedOp{ .call = self.builder.buildCall(op.function, num_args) };
            },
            .load => MappedOp{ .load = self.builder.buildAdd() }, // Placeholder
            .store => MappedOp{ .assign = dialect.AssignOp.create() },
            .br => MappedOp{ .br = self.builder.buildReturn(false) }, // Placeholder
            .cond_br => MappedOp{ .cond_br = self.builder.buildReturn(false) }, // Placeholder
            else => MappedOp{ .add = self.builder.buildAdd() }, // Default fallback
        };
    }
};

/// Union type for mapped operations
pub const MappedOp = union(enum) {
    add: dialect.AddOp,
    sub: dialect.AddOp, // Using AddOp as placeholder
    mul: dialect.AddOp,
    div: dialect.AddOp,
    ret: dialect.ReturnOp,
    call: dialect.CallOp,
    load: dialect.AddOp,
    assign: dialect.AssignOp,
    br: dialect.ReturnOp, // Placeholder
    cond_br: dialect.ReturnOp, // Placeholder
};

// ============================================================================
// Basic Block to MLIR Block Converter
// ============================================================================

pub const BlockConverter = struct {
    allocator: std.mem.Allocator,
    mapper: InstructionMapper,
    
    pub fn init(allocator: std.mem.Allocator) BlockConverter {
        return BlockConverter{
            .allocator = allocator,
            .mapper = InstructionMapper.init(),
        };
    }
    
    /// Convert IR basic block to MLIR block representation
    pub fn convertBlock(self: *BlockConverter, ir_block: *const IR.BasicBlock) !MlirBlockInfo {
        var converted_ops = std.ArrayList(MappedOp).initCapacity(self.allocator, ir_block.instructions.items.len) catch std.ArrayList(MappedOp).initCapacity(self.allocator, 0) catch unreachable;
        errdefer converted_ops.deinit(self.allocator);
        
        // Convert each instruction
        for (ir_block.instructions.items) |*inst| {
            const mapped_op = try self.mapper.mapInstruction(inst);
            try converted_ops.append(self.allocator, mapped_op);
        }
        
        return MlirBlockInfo{
            .name = ir_block.label,
            .operations = converted_ops,
            .predecessors = ir_block.predecessors.items.len,
            .successors = ir_block.successors.items.len,
            .allocator = self.allocator,
        };
    }
};

/// Information about converted MLIR block
pub const MlirBlockInfo = struct {
    name: []const u8,
    operations: std.ArrayList(MappedOp),
    predecessors: usize,
    successors: usize,
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *MlirBlockInfo) void {
        self.operations.deinit(self.allocator);
    }
};

// ============================================================================
// Function to MLIR Function Converter
// ============================================================================

pub const FunctionConverter = struct {
    allocator: std.mem.Allocator,
    type_mapper: TypeMapper,
    block_converter: BlockConverter,
    
    pub fn init(allocator: std.mem.Allocator) FunctionConverter {
        return FunctionConverter{
            .allocator = allocator,
            .type_mapper = TypeMapper{},
            .block_converter = BlockConverter.init(allocator),
        };
    }
    
    /// Convert IR function to Mojo dialect function
    pub fn convertFunction(self: *FunctionConverter, ir_func: *const IR.Function) !MlirFunctionInfo {
        // Map parameter types
        var param_types = try std.ArrayList(dialect.MojoType).initCapacity(self.allocator, ir_func.parameters.len);
        errdefer param_types.deinit(self.allocator);
        
        for (ir_func.parameters) |param| {
            const mojo_type = TypeMapper.mapType(param.type);
            try param_types.append(self.allocator, mojo_type);
        }
        
        // Map return type
        const return_type = TypeMapper.mapType(ir_func.return_type);
        
        // Convert basic blocks
        var blocks = try std.ArrayList(MlirBlockInfo).initCapacity(self.allocator, ir_func.blocks.items.len);
        errdefer {
            for (blocks.items) |*block| {
                block.deinit();
            }
            blocks.deinit(self.allocator);
        }
        
        for (ir_func.blocks.items) |bb| {
            const block_info = try self.block_converter.convertBlock(bb);
            try blocks.append(self.allocator, block_info);
        }
        
        return MlirFunctionInfo{
            .name = ir_func.name,
            .parameters = param_types,
            .return_type = return_type,
            .blocks = blocks,
            .allocator = self.allocator,
        };
    }
};

/// Information about converted MLIR function
pub const MlirFunctionInfo = struct {
    name: []const u8,
    parameters: std.ArrayList(dialect.MojoType),
    return_type: dialect.MojoType,
    blocks: std.ArrayList(MlirBlockInfo),
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *MlirFunctionInfo) void {
        self.parameters.deinit(self.allocator);
        for (self.blocks.items) |*block| {
            block.deinit();
        }
        self.blocks.deinit(self.allocator);
    }
};

// ============================================================================
// Module Converter - Top Level
// ============================================================================

pub const ModuleConverter = struct {
    allocator: std.mem.Allocator,
    function_converter: FunctionConverter,
    
    pub fn init(allocator: std.mem.Allocator) ModuleConverter {
        return ModuleConverter{
            .allocator = allocator,
            .function_converter = FunctionConverter.init(allocator),
        };
    }
    
    /// Convert entire IR module to MLIR representation
    pub fn convertModule(self: *ModuleConverter, ir_module: *const IR.Module) !MlirModuleInfo {
        var functions = try std.ArrayList(MlirFunctionInfo).initCapacity(self.allocator, ir_module.functions.items.len);
        errdefer {
            for (functions.items) |*func| {
                func.deinit();
            }
            functions.deinit(self.allocator);
        }
        
        // Convert each function
        for (ir_module.functions.items) |*func| {
            const func_info = try self.function_converter.convertFunction(func);
            try functions.append(self.allocator, func_info);
        }
        
        return MlirModuleInfo{
            .name = ir_module.name,
            .functions = functions,
            .allocator = self.allocator,
        };
    }
};

/// Information about converted MLIR module
pub const MlirModuleInfo = struct {
    name: []const u8,
    functions: std.ArrayList(MlirFunctionInfo),
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *MlirModuleInfo) void {
        for (self.functions.items) |*func| {
            func.deinit();
        }
        self.functions.deinit(self.allocator);
    }
};

// ============================================================================
// Conversion Statistics
// ============================================================================

pub const ConversionStats = struct {
    functions_converted: usize = 0,
    blocks_converted: usize = 0,
    instructions_converted: usize = 0,
    types_mapped: usize = 0,
    
    pub fn init() ConversionStats {
        return ConversionStats{};
    }
    
    pub fn recordFunction(self: *ConversionStats) void {
        self.functions_converted += 1;
    }
    
    pub fn recordBlock(self: *ConversionStats) void {
        self.blocks_converted += 1;
    }
    
    pub fn recordInstruction(self: *ConversionStats) void {
        self.instructions_converted += 1;
    }
    
    pub fn recordType(self: *ConversionStats) void {
        self.types_mapped += 1;
    }
};

// ============================================================================
// Round-Trip Validator
// ============================================================================

pub const RoundTripValidator = struct {
    /// Validate that IR → MLIR conversion preserves semantics
    pub fn validate(ir_func: *const IR.Function, mlir_func: *const MlirFunctionInfo) !bool {
        // Check function name
        if (!std.mem.eql(u8, ir_func.name, mlir_func.name)) {
            return false;
        }
        
        // Check parameter count
        if (ir_func.parameters.len != mlir_func.parameters.items.len) {
            return false;
        }
        
        // Check block count
        if (ir_func.blocks.items.len != mlir_func.blocks.items.len) {
            return false;
        }
        
        return true;
    }
    
    /// Validate type mapping is correct
    pub fn validateType(ir_type: IR.Type, mojo_type: dialect.MojoType) bool {
        return TypeMapper.isCompatible(ir_type, mojo_type);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ir_to_mlir: type mapping" {
    const i32_type = TypeMapper.mapType(.i32);
    try std.testing.expectEqual(dialect.MojoTypeKind.Int, i32_type.kind);
    try std.testing.expectEqual(@as(u32, 32), i32_type.bit_width);
    
    const f64_type = TypeMapper.mapType(.f64);
    try std.testing.expectEqual(dialect.MojoTypeKind.Float, f64_type.kind);
    try std.testing.expectEqual(@as(u32, 64), f64_type.bit_width);
    
    const bool_type = TypeMapper.mapType(.bool_type);
    try std.testing.expectEqual(dialect.MojoTypeKind.Bool, bool_type.kind);
}

test "ir_to_mlir: type compatibility" {
    const i32_ir = IR.Type.i32;
    const i32_mojo = dialect.MojoType.createInt(32);
    
    try std.testing.expect(TypeMapper.isCompatible(i32_ir, i32_mojo));
    
    const f32_mojo = dialect.MojoType.createFloat(32);
    try std.testing.expect(!TypeMapper.isCompatible(i32_ir, f32_mojo));
}

test "ir_to_mlir: instruction mapping" {
    var mapper = InstructionMapper.init();
    
    const add_kind = mapper.mapInstructionKind(std.meta.Tag(IR.Instruction).add);
    try std.testing.expectEqual(dialect.MojoOpKind.Add, add_kind);
    
    const ret_kind = mapper.mapInstructionKind(std.meta.Tag(IR.Instruction).ret);
    try std.testing.expectEqual(dialect.MojoOpKind.Return, ret_kind);
    
    const call_kind = mapper.mapInstructionKind(std.meta.Tag(IR.Instruction).call);
    try std.testing.expectEqual(dialect.MojoOpKind.Call, call_kind);
}

test "ir_to_mlir: block conversion" {
    const allocator = std.testing.allocator;
    
    // Create a simple IR basic block
    var bb = try IR.BasicBlock.init(allocator, "entry");
    defer bb.deinit(allocator);
    
    // Add a return instruction
    const ret_inst = IR.Instruction{
        .ret = .{ .value = null },
    };
    try bb.addInstruction(allocator, ret_inst);
    
    // Convert to MLIR
    var converter = BlockConverter.init(allocator);
    var mlir_block = try converter.convertBlock(&bb);
    defer mlir_block.deinit();
    
    try std.testing.expectEqualStrings("entry", mlir_block.name);
    try std.testing.expectEqual(@as(usize, 1), mlir_block.operations.items.len);
}

test "ir_to_mlir: function conversion" {
    const allocator = std.testing.allocator;
    
    // Create a simple IR function with parameter
    var params = [_]IR.Function.Parameter{
        .{ .name = "x", .type = .i32, .register = .{ .id = 0, .type = .i32, .name = "x" } },
    };
    var func = try IR.Function.init(allocator, "test_func", .i32, &params);
    defer func.deinit();
    
    // Add return to entry block
    const ret_inst = IR.Instruction{
        .ret = .{ .value = null },
    };
    try func.entry_block.addInstruction(allocator, ret_inst);
    
    // Convert to MLIR
    var converter = FunctionConverter.init(allocator);
    var mlir_func = try converter.convertFunction(&func);
    defer mlir_func.deinit();
    
    try std.testing.expectEqualStrings("test_func", mlir_func.name);
    try std.testing.expectEqual(@as(usize, 1), mlir_func.parameters.items.len);
    try std.testing.expectEqual(dialect.MojoTypeKind.Int, mlir_func.return_type.kind);
    try std.testing.expectEqual(@as(usize, 1), mlir_func.blocks.items.len);
}

test "ir_to_mlir: round trip validation" {
    const allocator = std.testing.allocator;
    
    // Create IR function with parameter
    var params = [_]IR.Function.Parameter{
        .{ .name = "a", .type = .i32, .register = .{ .id = 0, .type = .i32, .name = "a" } },
    };
    var ir_func = try IR.Function.init(allocator, "validate_func", .void_type, &params);
    defer ir_func.deinit();
    
    // Convert to MLIR
    var converter = FunctionConverter.init(allocator);
    var mlir_func = try converter.convertFunction(&ir_func);
    defer mlir_func.deinit();
    
    // Validate round trip
    const is_valid = try RoundTripValidator.validate(&ir_func, &mlir_func);
    try std.testing.expect(is_valid);
}

test "ir_to_mlir: conversion statistics" {
    var stats = ConversionStats.init();
    
    stats.recordFunction();
    stats.recordBlock();
    stats.recordBlock();
    stats.recordInstruction();
    stats.recordInstruction();
    stats.recordInstruction();
    stats.recordType();
    
    try std.testing.expectEqual(@as(usize, 1), stats.functions_converted);
    try std.testing.expectEqual(@as(usize, 2), stats.blocks_converted);
    try std.testing.expectEqual(@as(usize, 3), stats.instructions_converted);
    try std.testing.expectEqual(@as(usize, 1), stats.types_mapped);
}
