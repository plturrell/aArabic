// Mojo SDK - LLVM IR Lowering
// Day 15: Lower MLIR to LLVM IR for native code generation

const std = @import("std");
const mlir = @import("mlir_setup");
const dialect = @import("mojo_dialect");
const ir_to_mlir = @import("ir_to_mlir");

// ============================================================================
// LLVM Type System
// ============================================================================

pub const LLVMTypeKind = enum {
    Void,
    Integer,
    Float,
    Double,
    Pointer,
    Function,
    Struct,
    Array,
    
    pub fn toString(self: LLVMTypeKind) []const u8 {
        return switch (self) {
            .Void => "void",
            .Integer => "integer",
            .Float => "float",
            .Double => "double",
            .Pointer => "pointer",
            .Function => "function",
            .Struct => "struct",
            .Array => "array",
        };
    }
};

pub const LLVMType = struct {
    kind: LLVMTypeKind,
    bit_width: u32 = 0,
    name: ?[]const u8 = null,
    
    pub fn createVoid() LLVMType {
        return LLVMType{ .kind = .Void };
    }
    
    pub fn createInt(bit_width: u32) LLVMType {
        return LLVMType{
            .kind = .Integer,
            .bit_width = bit_width,
        };
    }
    
    pub fn createFloat() LLVMType {
        return LLVMType{
            .kind = .Float,
            .bit_width = 32,
        };
    }
    
    pub fn createDouble() LLVMType {
        return LLVMType{
            .kind = .Double,
            .bit_width = 64,
        };
    }
    
    pub fn createPointer() LLVMType {
        return LLVMType{
            .kind = .Pointer,
            .bit_width = 64,
        };
    }
    
    pub fn isInteger(self: *const LLVMType) bool {
        return self.kind == .Integer;
    }
    
    pub fn isFloatingPoint(self: *const LLVMType) bool {
        return self.kind == .Float or self.kind == .Double;
    }
};

// ============================================================================
// MLIR to LLVM Type Lowering
// ============================================================================

pub const TypeLowering = struct {
    /// Lower Mojo dialect type to LLVM type
    pub fn lowerType(mojo_type: dialect.MojoType) LLVMType {
        return switch (mojo_type.kind) {
            .Int => LLVMType.createInt(mojo_type.bit_width),
            .Float => if (mojo_type.bit_width == 32)
                LLVMType.createFloat()
            else
                LLVMType.createDouble(),
            .Bool => LLVMType.createInt(1),
            .Void => LLVMType.createVoid(),
            .String => LLVMType.createPointer(), // String as pointer
            .Struct => LLVMType.createPointer(), // Struct as pointer
            .Array => LLVMType.createPointer(), // Array as pointer
            .Function => LLVMType.createPointer(), // Function as pointer
            .Tuple => LLVMType.createPointer(), // Tuple as pointer
            .Unknown => LLVMType.createVoid(),
        };
    }
    
    /// Check if lowering is valid
    pub fn isValidLowering(mojo_type: dialect.MojoType, llvm_type: LLVMType) bool {
        const lowered = lowerType(mojo_type);
        return lowered.kind == llvm_type.kind and lowered.bit_width == llvm_type.bit_width;
    }
};

// ============================================================================
// LLVM Instructions
// ============================================================================

pub const LLVMInstKind = enum {
    // Arithmetic
    Add,
    Sub,
    Mul,
    SDiv,  // Signed division
    UDiv,  // Unsigned division
    
    // Comparison
    ICmpEQ,  // Integer compare equal
    ICmpNE,  // Integer compare not equal
    ICmpSLT, // Signed less than
    ICmpSLE, // Signed less than or equal
    ICmpSGT, // Signed greater than
    ICmpSGE, // Signed greater than or equal
    
    // Memory
    Alloca,
    Load,
    Store,
    
    // Control flow
    Br,      // Unconditional branch
    CondBr,  // Conditional branch
    Ret,     // Return
    Call,    // Function call
    
    pub fn getName(self: LLVMInstKind) []const u8 {
        return switch (self) {
            .Add => "add",
            .Sub => "sub",
            .Mul => "mul",
            .SDiv => "sdiv",
            .UDiv => "udiv",
            .ICmpEQ => "icmp eq",
            .ICmpNE => "icmp ne",
            .ICmpSLT => "icmp slt",
            .ICmpSLE => "icmp sle",
            .ICmpSGT => "icmp sgt",
            .ICmpSGE => "icmp sge",
            .Alloca => "alloca",
            .Load => "load",
            .Store => "store",
            .Br => "br",
            .CondBr => "br",
            .Ret => "ret",
            .Call => "call",
        };
    }
};

pub const LLVMInstruction = struct {
    kind: LLVMInstKind,
    result_type: ?LLVMType = null,
    operands: usize = 0,
    
    pub fn create(kind: LLVMInstKind) LLVMInstruction {
        return LLVMInstruction{ .kind = kind };
    }
};

// ============================================================================
// Operation Lowering
// ============================================================================

pub const OperationLowering = struct {
    /// Lower Mojo operation to LLVM instruction
    pub fn lowerOperation(mojo_op: dialect.MojoOpKind) LLVMInstKind {
        return switch (mojo_op) {
            .Add => .Add,
            .Sub => .Sub,
            .Mul => .Mul,
            .Div => .SDiv,
            .Eq => .ICmpEQ,
            .Ne => .ICmpNE,
            .Lt => .ICmpSLT,
            .Le => .ICmpSLE,
            .Gt => .ICmpSGT,
            .Ge => .ICmpSGE,
            .Load => .Load,
            .Assign => .Store,
            .Return => .Ret,
            .Call => .Call,
            .Var => .Alloca,
            else => .Ret, // Default fallback
        };
    }
};

// ============================================================================
// LLVM Basic Block
// ============================================================================

pub const LLVMBasicBlock = struct {
    name: []const u8,
    instructions: std.ArrayList(LLVMInstruction),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) !LLVMBasicBlock {
        return LLVMBasicBlock{
            .name = name,
            .instructions = try std.ArrayList(LLVMInstruction).initCapacity(allocator, 8),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *LLVMBasicBlock) void {
        self.instructions.deinit(self.allocator);
    }
    
    pub fn addInstruction(self: *LLVMBasicBlock, inst: LLVMInstruction) !void {
        try self.instructions.append(self.allocator, inst);
    }
};

// ============================================================================
// LLVM Function
// ============================================================================

pub const LLVMFunction = struct {
    name: []const u8,
    return_type: LLVMType,
    parameters: std.ArrayList(LLVMType),
    basic_blocks: std.ArrayList(LLVMBasicBlock),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, ret_type: LLVMType) !LLVMFunction {
        return LLVMFunction{
            .name = name,
            .return_type = ret_type,
            .parameters = try std.ArrayList(LLVMType).initCapacity(allocator, 4),
            .basic_blocks = try std.ArrayList(LLVMBasicBlock).initCapacity(allocator, 4),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *LLVMFunction) void {
        self.parameters.deinit(self.allocator);
        for (self.basic_blocks.items) |*bb| {
            bb.deinit();
        }
        self.basic_blocks.deinit(self.allocator);
    }
    
    pub fn addParameter(self: *LLVMFunction, param_type: LLVMType) !void {
        try self.parameters.append(self.allocator, param_type);
    }
    
    pub fn addBasicBlock(self: *LLVMFunction, block: LLVMBasicBlock) !void {
        try self.basic_blocks.append(self.allocator, block);
    }
};

// ============================================================================
// LLVM Module
// ============================================================================

pub const LLVMModule = struct {
    name: []const u8,
    functions: std.ArrayList(LLVMFunction),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) !LLVMModule {
        return LLVMModule{
            .name = name,
            .functions = try std.ArrayList(LLVMFunction).initCapacity(allocator, 4),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *LLVMModule) void {
        for (self.functions.items) |*func| {
            func.deinit();
        }
        self.functions.deinit(self.allocator);
    }
    
    pub fn addFunction(self: *LLVMModule, func: LLVMFunction) !void {
        try self.functions.append(self.allocator, func);
    }
};

// ============================================================================
// Block Lowering
// ============================================================================

pub const BlockLowering = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) BlockLowering {
        return BlockLowering{ .allocator = allocator };
    }
    
    /// Lower MLIR block to LLVM basic block
    pub fn lowerBlock(self: *BlockLowering, mlir_block: *const ir_to_mlir.MlirBlockInfo) !LLVMBasicBlock {
        var llvm_block = try LLVMBasicBlock.init(self.allocator, mlir_block.name);
        errdefer llvm_block.deinit();
        
        // Lower each operation to LLVM instruction
        for (mlir_block.operations.items) |op| {
            const llvm_inst = try self.lowerMappedOp(op);
            try llvm_block.addInstruction(llvm_inst);
        }
        
        return llvm_block;
    }
    
    fn lowerMappedOp(self: *BlockLowering, op: ir_to_mlir.MappedOp) !LLVMInstruction {
        _ = self;
        return switch (op) {
            .add => LLVMInstruction.create(.Add),
            .sub => LLVMInstruction.create(.Sub),
            .mul => LLVMInstruction.create(.Mul),
            .div => LLVMInstruction.create(.SDiv),
            .ret => LLVMInstruction.create(.Ret),
            .call => LLVMInstruction.create(.Call),
            .load => LLVMInstruction.create(.Load),
            .assign => LLVMInstruction.create(.Store),
            .br => LLVMInstruction.create(.Br),
            .cond_br => LLVMInstruction.create(.CondBr),
        };
    }
};

// ============================================================================
// Function Lowering
// ============================================================================

pub const FunctionLowering = struct {
    allocator: std.mem.Allocator,
    block_lowering: BlockLowering,
    
    pub fn init(allocator: std.mem.Allocator) FunctionLowering {
        return FunctionLowering{
            .allocator = allocator,
            .block_lowering = BlockLowering.init(allocator),
        };
    }
    
    /// Lower MLIR function to LLVM function
    pub fn lowerFunction(self: *FunctionLowering, mlir_func: *const ir_to_mlir.MlirFunctionInfo) !LLVMFunction {
        // Lower return type
        const llvm_ret_type = TypeLowering.lowerType(mlir_func.return_type);
        
        var llvm_func = try LLVMFunction.init(self.allocator, mlir_func.name, llvm_ret_type);
        errdefer llvm_func.deinit();
        
        // Lower parameters
        for (mlir_func.parameters.items) |param_type| {
            const llvm_param_type = TypeLowering.lowerType(param_type);
            try llvm_func.addParameter(llvm_param_type);
        }
        
        // Lower basic blocks
        for (mlir_func.blocks.items) |*block| {
            const llvm_block = try self.block_lowering.lowerBlock(block);
            try llvm_func.addBasicBlock(llvm_block);
        }
        
        return llvm_func;
    }
};

// ============================================================================
// Module Lowering
// ============================================================================

pub const ModuleLowering = struct {
    allocator: std.mem.Allocator,
    function_lowering: FunctionLowering,
    
    pub fn init(allocator: std.mem.Allocator) ModuleLowering {
        return ModuleLowering{
            .allocator = allocator,
            .function_lowering = FunctionLowering.init(allocator),
        };
    }
    
    /// Lower MLIR module to LLVM module
    pub fn lowerModule(self: *ModuleLowering, mlir_module: *const ir_to_mlir.MlirModuleInfo) !LLVMModule {
        var llvm_module = try LLVMModule.init(self.allocator, mlir_module.name);
        errdefer llvm_module.deinit();
        
        // Lower each function
        for (mlir_module.functions.items) |*func| {
            const llvm_func = try self.function_lowering.lowerFunction(func);
            try llvm_module.addFunction(llvm_func);
        }
        
        return llvm_module;
    }
};

// ============================================================================
// Lowering Statistics
// ============================================================================

pub const LoweringStats = struct {
    functions_lowered: usize = 0,
    blocks_lowered: usize = 0,
    instructions_lowered: usize = 0,
    types_lowered: usize = 0,
    
    pub fn init() LoweringStats {
        return LoweringStats{};
    }
    
    pub fn recordFunction(self: *LoweringStats) void {
        self.functions_lowered += 1;
    }
    
    pub fn recordBlock(self: *LoweringStats) void {
        self.blocks_lowered += 1;
    }
    
    pub fn recordInstruction(self: *LoweringStats) void {
        self.instructions_lowered += 1;
    }
    
    pub fn recordType(self: *LoweringStats) void {
        self.types_lowered += 1;
    }
    
    pub fn print(self: *const LoweringStats, writer: anytype) !void {
        try writer.print("LLVM Lowering Statistics:\n", .{});
        try writer.print("  Functions lowered: {}\n", .{self.functions_lowered});
        try writer.print("  Blocks lowered: {}\n", .{self.blocks_lowered});
        try writer.print("  Instructions lowered: {}\n", .{self.instructions_lowered});
        try writer.print("  Types lowered: {}\n", .{self.types_lowered});
    }
};

// ============================================================================
// LLVM Backend Configuration
// ============================================================================

pub const BackendConfig = struct {
    target_triple: []const u8 = "x86_64-apple-darwin", // Default to macOS
    cpu: []const u8 = "generic",
    features: []const u8 = "",
    optimization_level: u8 = 2, // 0-3
    
    pub fn forMacOS() BackendConfig {
        return BackendConfig{
            .target_triple = "x86_64-apple-darwin",
            .cpu = "generic",
        };
    }
    
    pub fn forLinux() BackendConfig {
        return BackendConfig{
            .target_triple = "x86_64-unknown-linux-gnu",
            .cpu = "generic",
        };
    }
    
    pub fn forWindows() BackendConfig {
        return BackendConfig{
            .target_triple = "x86_64-pc-windows-msvc",
            .cpu = "generic",
        };
    }
};

// ============================================================================
// LLVM Lowering Engine
// ============================================================================

pub const LLVMLoweringEngine = struct {
    allocator: std.mem.Allocator,
    config: BackendConfig,
    module_lowering: ModuleLowering,
    stats: LoweringStats,
    
    pub fn init(allocator: std.mem.Allocator, config: BackendConfig) LLVMLoweringEngine {
        return LLVMLoweringEngine{
            .allocator = allocator,
            .config = config,
            .module_lowering = ModuleLowering.init(allocator),
            .stats = LoweringStats.init(),
        };
    }
    
    /// Lower MLIR module to LLVM module
    pub fn lower(self: *LLVMLoweringEngine, mlir_module: *const ir_to_mlir.MlirModuleInfo) !LLVMModule {
        const llvm_module = try self.module_lowering.lowerModule(mlir_module);
        
        // Update statistics
        self.stats.functions_lowered = llvm_module.functions.items.len;
        for (llvm_module.functions.items) |*func| {
            self.stats.blocks_lowered += func.basic_blocks.items.len;
            for (func.basic_blocks.items) |*block| {
                self.stats.instructions_lowered += block.instructions.items.len;
            }
        }
        
        return llvm_module;
    }
    
    pub fn getStats(self: *const LLVMLoweringEngine) LoweringStats {
        return self.stats;
    }
    
    pub fn printStats(self: *const LLVMLoweringEngine, writer: anytype) !void {
        try writer.print("Target: {s}\n", .{self.config.target_triple});
        try writer.print("CPU: {s}\n", .{self.config.cpu});
        try self.stats.print(writer);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "llvm_lowering: type lowering" {
    const mojo_i32 = dialect.MojoType.createInt(32);
    const llvm_i32 = TypeLowering.lowerType(mojo_i32);
    
    try std.testing.expectEqual(LLVMTypeKind.Integer, llvm_i32.kind);
    try std.testing.expectEqual(@as(u32, 32), llvm_i32.bit_width);
    
    const mojo_f64 = dialect.MojoType.createFloat(64);
    const llvm_f64 = TypeLowering.lowerType(mojo_f64);
    
    try std.testing.expectEqual(LLVMTypeKind.Double, llvm_f64.kind);
}

test "llvm_lowering: type validation" {
    const mojo_i32 = dialect.MojoType.createInt(32);
    const llvm_i32 = LLVMType.createInt(32);
    
    try std.testing.expect(TypeLowering.isValidLowering(mojo_i32, llvm_i32));
}

test "llvm_lowering: operation lowering" {
    const add_inst = OperationLowering.lowerOperation(.Add);
    try std.testing.expectEqual(LLVMInstKind.Add, add_inst);
    
    const ret_inst = OperationLowering.lowerOperation(.Return);
    try std.testing.expectEqual(LLVMInstKind.Ret, ret_inst);
    
    const call_inst = OperationLowering.lowerOperation(.Call);
    try std.testing.expectEqual(LLVMInstKind.Call, call_inst);
}

test "llvm_lowering: instruction names" {
    try std.testing.expectEqualStrings("add", LLVMInstKind.Add.getName());
    try std.testing.expectEqualStrings("ret", LLVMInstKind.Ret.getName());
    try std.testing.expectEqualStrings("icmp eq", LLVMInstKind.ICmpEQ.getName());
}

test "llvm_lowering: basic block creation" {
    const allocator = std.testing.allocator;
    
    var bb = try LLVMBasicBlock.init(allocator, "entry");
    defer bb.deinit();
    
    const add_inst = LLVMInstruction.create(.Add);
    try bb.addInstruction(add_inst);
    
    try std.testing.expectEqualStrings("entry", bb.name);
    try std.testing.expectEqual(@as(usize, 1), bb.instructions.items.len);
}

test "llvm_lowering: function creation" {
    const allocator = std.testing.allocator;
    
    var func = try LLVMFunction.init(allocator, "test", LLVMType.createVoid());
    defer func.deinit();
    
    try func.addParameter(LLVMType.createInt(32));
    
    var bb = try LLVMBasicBlock.init(allocator, "entry");
    const ret_inst = LLVMInstruction.create(.Ret);
    try bb.addInstruction(ret_inst);
    try func.addBasicBlock(bb);
    
    try std.testing.expectEqualStrings("test", func.name);
    try std.testing.expectEqual(@as(usize, 1), func.parameters.items.len);
    try std.testing.expectEqual(@as(usize, 1), func.basic_blocks.items.len);
}

test "llvm_lowering: module creation" {
    const allocator = std.testing.allocator;
    
    var module = try LLVMModule.init(allocator, "test_module");
    defer module.deinit();
    
    const func = try LLVMFunction.init(allocator, "main", LLVMType.createInt(32));
    try module.addFunction(func);
    
    try std.testing.expectEqualStrings("test_module", module.name);
    try std.testing.expectEqual(@as(usize, 1), module.functions.items.len);
}

test "llvm_lowering: backend configuration" {
    const macos_config = BackendConfig.forMacOS();
    try std.testing.expectEqualStrings("x86_64-apple-darwin", macos_config.target_triple);
    
    const linux_config = BackendConfig.forLinux();
    try std.testing.expectEqualStrings("x86_64-unknown-linux-gnu", linux_config.target_triple);
}

test "llvm_lowering: lowering statistics" {
    var stats = LoweringStats.init();
    
    stats.recordFunction();
    stats.recordBlock();
    stats.recordBlock();
    stats.recordInstruction();
    stats.recordInstruction();
    stats.recordInstruction();
    stats.recordType();
    
    try std.testing.expectEqual(@as(usize, 1), stats.functions_lowered);
    try std.testing.expectEqual(@as(usize, 2), stats.blocks_lowered);
    try std.testing.expectEqual(@as(usize, 3), stats.instructions_lowered);
    try std.testing.expectEqual(@as(usize, 1), stats.types_lowered);
}

test "llvm_lowering: create engine" {
    const allocator = std.testing.allocator;
    
    const config = BackendConfig.forMacOS();
    const engine = LLVMLoweringEngine.init(allocator, config);
    
    try std.testing.expectEqualStrings("x86_64-apple-darwin", engine.config.target_triple);
    try std.testing.expectEqual(@as(usize, 0), engine.stats.functions_lowered);
}
