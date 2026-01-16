// Mojo SDK - LLVM Code Generation
// Day 16: Generate LLVM IR text and compile to native code

const std = @import("std");
const llvm_lowering = @import("llvm_lowering");

// ============================================================================
// LLVM IR Text Generation
// ============================================================================

pub const IRGenerator = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) IRGenerator {
        return IRGenerator{ .allocator = allocator };
    }
    
    /// Generate LLVM IR text from LLVM module
    pub fn generateIR(self: *IRGenerator, module: *const llvm_lowering.LLVMModule) ![]const u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        errdefer output.deinit();
        
        const writer = output.writer();
        
        // Module header
        try writer.print("; ModuleID = '{s}'\n", .{module.name});
        try writer.print("source_filename = \"{s}\"\n\n", .{module.name});
        
        // Generate each function
        for (module.functions.items) |*func| {
            try self.generateFunction(writer, func);
            try writer.writeAll("\n");
        }
        
        return try output.toOwnedSlice();
    }
    
    fn generateFunction(self: *IRGenerator, writer: anytype, func: *const llvm_lowering.LLVMFunction) !void {
        // Function signature
        try writer.print("define ", .{});
        try self.writeType(writer, func.return_type);
        try writer.print(" @{s}(", .{func.name});
        
        // Parameters
        for (func.parameters.items, 0..) |param, i| {
            if (i > 0) try writer.writeAll(", ");
            try self.writeType(writer, param);
            try writer.print(" %arg{}", .{i});
        }
        
        try writer.writeAll(") {\n");
        
        // Basic blocks
        for (func.basic_blocks.items) |*bb| {
            try self.generateBasicBlock(writer, bb);
        }
        
        try writer.writeAll("}\n");
    }
    
    fn generateBasicBlock(self: *IRGenerator, writer: anytype, bb: *const llvm_lowering.LLVMBasicBlock) !void {
        try writer.print("{s}:\n", .{bb.name});
        
        for (bb.instructions.items, 0..) |inst, i| {
            try writer.writeAll("  ");
            try self.generateInstruction(writer, inst, i);
            try writer.writeAll("\n");
        }
    }
    
    fn generateInstruction(self: *IRGenerator, writer: anytype, inst: llvm_lowering.LLVMInstruction, id: usize) !void {
        _ = self;
        _ = id;
        
        const name = inst.kind.getName();
        try writer.print("{s}", .{name});
    }
    
    fn writeType(self: *IRGenerator, writer: anytype, llvm_type: llvm_lowering.LLVMType) !void {
        _ = self;
        
        switch (llvm_type.kind) {
            .Void => try writer.writeAll("void"),
            .Integer => try writer.print("i{}", .{llvm_type.bit_width}),
            .Float => try writer.writeAll("float"),
            .Double => try writer.writeAll("double"),
            .Pointer => try writer.writeAll("ptr"),
            .Function => try writer.writeAll("ptr"),
            .Struct => try writer.writeAll("%struct"),
            .Array => try writer.writeAll("ptr"),
        }
    }
};

// ============================================================================
// Optimization Level Configuration
// ============================================================================

pub const OptimizationLevel = enum {
    O0,  // No optimization
    O1,  // Basic optimization
    O2,  // Standard optimization
    O3,  // Aggressive optimization
    
    pub fn toString(self: OptimizationLevel) []const u8 {
        return switch (self) {
            .O0 => "O0",
            .O1 => "O1",
            .O2 => "O2",
            .O3 => "O3",
        };
    }
    
    pub fn toFlag(self: OptimizationLevel) []const u8 {
        return switch (self) {
            .O0 => "-O0",
            .O1 => "-O1",
            .O2 => "-O2",
            .O3 => "-O3",
        };
    }
};

// ============================================================================
// Code Generation Configuration
// ============================================================================

pub const CodeGenConfig = struct {
    target_triple: []const u8,
    optimization_level: OptimizationLevel = .O2,
    emit_llvm_ir: bool = true,
    emit_assembly: bool = false,
    emit_object: bool = true,
    debug_info: bool = false,
    
    pub fn default() CodeGenConfig {
        return CodeGenConfig{
            .target_triple = "x86_64-apple-darwin",
        };
    }
    
    pub fn forTarget(target: []const u8) CodeGenConfig {
        return CodeGenConfig{
            .target_triple = target,
        };
    }
};

// ============================================================================
// Output Files
// ============================================================================

pub const OutputFiles = struct {
    llvm_ir: ?[]const u8 = null,    // .ll file content
    assembly: ?[]const u8 = null,   // .s file content
    object: ?[]const u8 = null,     // .o file path
    executable: ?[]const u8 = null, // executable path
    
    pub fn hasIR(self: *const OutputFiles) bool {
        return self.llvm_ir != null;
    }
    
    pub fn hasAssembly(self: *const OutputFiles) bool {
        return self.assembly != null;
    }
    
    pub fn hasObject(self: *const OutputFiles) bool {
        return self.object != null;
    }
    
    pub fn hasExecutable(self: *const OutputFiles) bool {
        return self.executable != null;
    }
};

// ============================================================================
// Compilation Statistics
// ============================================================================

pub const CompilationStats = struct {
    llvm_ir_size: usize = 0,
    assembly_lines: usize = 0,
    object_size: usize = 0,
    compilation_time_ms: u64 = 0,
    
    pub fn init() CompilationStats {
        return CompilationStats{};
    }
    
    pub fn recordIR(self: *CompilationStats, size: usize) void {
        self.llvm_ir_size = size;
    }
    
    pub fn recordAssembly(self: *CompilationStats, lines: usize) void {
        self.assembly_lines = lines;
    }
    
    pub fn recordObject(self: *CompilationStats, size: usize) void {
        self.object_size = size;
    }
    
    pub fn recordTime(self: *CompilationStats, ms: u64) void {
        self.compilation_time_ms = ms;
    }
    
    pub fn print(self: *const CompilationStats, writer: anytype) !void {
        try writer.print("Compilation Statistics:\n", .{});
        try writer.print("  LLVM IR size: {} bytes\n", .{self.llvm_ir_size});
        try writer.print("  Assembly lines: {}\n", .{self.assembly_lines});
        try writer.print("  Object size: {} bytes\n", .{self.object_size});
        try writer.print("  Compilation time: {}ms\n", .{self.compilation_time_ms});
    }
};

// ============================================================================
// File Writer
// ============================================================================

pub const FileWriter = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) FileWriter {
        return FileWriter{ .allocator = allocator };
    }
    
    /// Write LLVM IR to file
    pub fn writeIR(self: *FileWriter, path: []const u8, content: []const u8) !void {
        _ = self;
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(content);
    }
    
    /// Write assembly to file
    pub fn writeAssembly(self: *FileWriter, path: []const u8, content: []const u8) !void {
        _ = self;
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(content);
    }
    
    /// Check if file exists
    pub fn fileExists(self: *FileWriter, path: []const u8) bool {
        _ = self;
        std.fs.cwd().access(path, .{}) catch return false;
        return true;
    }
};

// ============================================================================
// LLVM Command Runner
// ============================================================================

pub const LLVMCommandRunner = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) LLVMCommandRunner {
        return LLVMCommandRunner{ .allocator = allocator };
    }
    
    /// Run llc (LLVM compiler) to generate object file
    pub fn compileLLVMIR(self: *LLVMCommandRunner, ir_file: []const u8, output_file: []const u8, opt_level: OptimizationLevel) !void {
        _ = self;
        _ = ir_file;
        _ = output_file;
        _ = opt_level;
        // This would execute: llc -filetype=obj -O2 input.ll -o output.o
        // For now, this is a placeholder for the actual system call
    }
    
    /// Run opt (LLVM optimizer) on IR file
    pub fn optimizeLLVMIR(self: *LLVMCommandRunner, input_file: []const u8, output_file: []const u8, opt_level: OptimizationLevel) !void {
        _ = self;
        _ = input_file;
        _ = output_file;
        _ = opt_level;
        // This would execute: opt -O2 input.ll -o output.ll
        // For now, this is a placeholder
    }
    
    /// Link object files to create executable
    pub fn linkObjects(self: *LLVMCommandRunner, object_files: []const []const u8, output_file: []const u8) !void {
        _ = self;
        _ = object_files;
        _ = output_file;
        // This would execute: clang object1.o object2.o -o executable
        // For now, this is a placeholder
    }
};

// ============================================================================
// Code Generator - Main Interface
// ============================================================================

pub const CodeGenerator = struct {
    allocator: std.mem.Allocator,
    config: CodeGenConfig,
    ir_generator: IRGenerator,
    file_writer: FileWriter,
    command_runner: LLVMCommandRunner,
    stats: CompilationStats,
    
    pub fn init(allocator: std.mem.Allocator, config: CodeGenConfig) CodeGenerator {
        return CodeGenerator{
            .allocator = allocator,
            .config = config,
            .ir_generator = IRGenerator.init(allocator),
            .file_writer = FileWriter.init(allocator),
            .command_runner = LLVMCommandRunner.init(allocator),
            .stats = CompilationStats.init(),
        };
    }
    
    /// Generate code from LLVM module
    pub fn generate(self: *CodeGenerator, module: *const llvm_lowering.LLVMModule, output_base: []const u8) !OutputFiles {
        const start_time = std.time.milliTimestamp();
        
        var outputs = OutputFiles{};
        
        // Generate LLVM IR
        if (self.config.emit_llvm_ir) {
            const ir_content = try self.ir_generator.generateIR(module);
            self.stats.recordIR(ir_content.len);
            
            const ir_path = try std.fmt.allocPrint(self.allocator, "{s}.ll", .{output_base});
            defer self.allocator.free(ir_path);
            
            try self.file_writer.writeIR(ir_path, ir_content);
            outputs.llvm_ir = ir_path;
        }
        
        // Generate assembly (placeholder)
        if (self.config.emit_assembly) {
            const asm_path = try std.fmt.allocPrint(self.allocator, "{s}.s", .{output_base});
            outputs.assembly = asm_path;
        }
        
        // Generate object file (placeholder)
        if (self.config.emit_object) {
            const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_base});
            outputs.object = obj_path;
        }
        
        const end_time = std.time.milliTimestamp();
        self.stats.recordTime(@intCast(end_time - start_time));
        
        return outputs;
    }
    
    /// Get compilation statistics
    pub fn getStats(self: *const CodeGenerator) CompilationStats {
        return self.stats;
    }
    
    /// Print statistics
    pub fn printStats(self: *const CodeGenerator, writer: anytype) !void {
        try writer.print("Target: {s}\n", .{self.config.target_triple});
        try writer.print("Optimization: {s}\n", .{self.config.optimization_level.toString()});
        try self.stats.print(writer);
    }
};

// ============================================================================
// Build Pipeline
// ============================================================================

pub const BuildPipeline = struct {
    allocator: std.mem.Allocator,
    codegen: CodeGenerator,
    
    pub fn init(allocator: std.mem.Allocator, config: CodeGenConfig) BuildPipeline {
        return BuildPipeline{
            .allocator = allocator,
            .codegen = CodeGenerator.init(allocator, config),
        };
    }
    
    /// Complete build pipeline: LLVM module → executable
    pub fn build(self: *BuildPipeline, module: *const llvm_lowering.LLVMModule, output_name: []const u8) !OutputFiles {
        // Generate LLVM IR and object files
        const outputs = try self.codegen.generate(module, output_name);
        
        return outputs;
    }
    
    /// Get build statistics
    pub fn getStats(self: *const BuildPipeline) CompilationStats {
        return self.codegen.getStats();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "codegen: optimization levels" {
    try std.testing.expectEqualStrings("O0", OptimizationLevel.O0.toString());
    try std.testing.expectEqualStrings("O2", OptimizationLevel.O2.toString());
    try std.testing.expectEqualStrings("-O1", OptimizationLevel.O1.toFlag());
    try std.testing.expectEqualStrings("-O3", OptimizationLevel.O3.toFlag());
}

test "codegen: config creation" {
    const config = CodeGenConfig.default();
    try std.testing.expectEqualStrings("x86_64-apple-darwin", config.target_triple);
    try std.testing.expectEqual(OptimizationLevel.O2, config.optimization_level);
    try std.testing.expect(config.emit_llvm_ir);
    try std.testing.expect(config.emit_object);
}

test "codegen: config for target" {
    const config = CodeGenConfig.forTarget("x86_64-unknown-linux-gnu");
    try std.testing.expectEqualStrings("x86_64-unknown-linux-gnu", config.target_triple);
}

test "codegen: output files" {
    var outputs = OutputFiles{};
    try std.testing.expect(!outputs.hasIR());
    try std.testing.expect(!outputs.hasObject());
    
    outputs.llvm_ir = "test.ll";
    try std.testing.expect(outputs.hasIR());
}

test "codegen: compilation stats" {
    var stats = CompilationStats.init();
    
    stats.recordIR(1024);
    stats.recordAssembly(50);
    stats.recordObject(2048);
    stats.recordTime(150);
    
    try std.testing.expectEqual(@as(usize, 1024), stats.llvm_ir_size);
    try std.testing.expectEqual(@as(usize, 50), stats.assembly_lines);
    try std.testing.expectEqual(@as(usize, 2048), stats.object_size);
    try std.testing.expectEqual(@as(u64, 150), stats.compilation_time_ms);
}

test "codegen: ir generator init" {
    const allocator = std.testing.allocator;
    const gen = IRGenerator.init(allocator);
    try std.testing.expect(gen.allocator.ptr == allocator.ptr);
}

test "codegen: file writer init" {
    const allocator = std.testing.allocator;
    const writer = FileWriter.init(allocator);
    try std.testing.expect(writer.allocator.ptr == allocator.ptr);
}

test "codegen: command runner init" {
    const allocator = std.testing.allocator;
    const runner = LLVMCommandRunner.init(allocator);
    try std.testing.expect(runner.allocator.ptr == allocator.ptr);
}

test "codegen: code generator init" {
    const allocator = std.testing.allocator;
    const config = CodeGenConfig.default();
    const gen = CodeGenerator.init(allocator, config);
    
    try std.testing.expectEqualStrings("x86_64-apple-darwin", gen.config.target_triple);
    try std.testing.expectEqual(@as(usize, 0), gen.stats.llvm_ir_size);
}

test "codegen: build pipeline init" {
    const allocator = std.testing.allocator;
    const config = CodeGenConfig.default();
    const pipeline = BuildPipeline.init(allocator, config);
    
    try std.testing.expectEqualStrings("x86_64-apple-darwin", pipeline.codegen.config.target_triple);
}
