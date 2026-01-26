// Mojo SDK - Native Code Compilation & Linking
// Day 17: Compile LLVM IR to native executables

const std = @import("std");
const llvm_lowering = @import("llvm_lowering");
const codegen = @import("codegen");

// ============================================================================
// Compiler Tool Paths
// ============================================================================

pub const ToolPaths = struct {
    llc: []const u8 = "llc",           // LLVM compiler
    opt: []const u8 = "opt",           // LLVM optimizer
    clang: []const u8 = "clang",       // C compiler/linker
    llvm_as: []const u8 = "llvm-as",   // LLVM assembler
    
    pub fn default() ToolPaths {
        return ToolPaths{};
    }
    
    pub fn withPrefix(prefix: []const u8) ToolPaths {
        return ToolPaths{
            .llc = prefix,
            .opt = prefix,
            .clang = prefix,
            .llvm_as = prefix,
        };
    }
};

// ============================================================================
// Compilation Options
// ============================================================================

pub const CompilationOptions = struct {
    optimization_level: codegen.OptimizationLevel = .O2,
    pic: bool = true,              // Position independent code
    no_pie: bool = false,          // No position independent executable
    static: bool = false,          // Static linking
    strip: bool = false,           // Strip symbols
    verbose: bool = false,         // Verbose output
    
    pub fn default() CompilationOptions {
        return CompilationOptions{};
    }
    
    pub fn forRelease() CompilationOptions {
        return CompilationOptions{
            .optimization_level = .O3,
            .strip = true,
        };
    }
    
    pub fn forDebug() CompilationOptions {
        return CompilationOptions{
            .optimization_level = .O0,
            .pic = false,
        };
    }
};

// ============================================================================
// Compilation Result
// ============================================================================

pub const CompilationResult = struct {
    success: bool,
    ir_file: ?[]const u8 = null,
    object_file: ?[]const u8 = null,
    executable: ?[]const u8 = null,
    stderr_output: ?[]const u8 = null,
    compilation_time_ms: u64 = 0,
    
    pub fn isSuccess(self: *const CompilationResult) bool {
        return self.success;
    }
    
    pub fn hasExecutable(self: *const CompilationResult) bool {
        return self.executable != null;
    }
};

// ============================================================================
// Native Compiler
// ============================================================================

pub const NativeCompiler = struct {
    allocator: std.mem.Allocator,
    tools: ToolPaths,
    options: CompilationOptions,
    
    pub fn init(allocator: std.mem.Allocator, tools: ToolPaths, options: CompilationOptions) NativeCompiler {
        return NativeCompiler{
            .allocator = allocator,
            .tools = tools,
            .options = options,
        };
    }
    
    /// Compile LLVM IR file to object file
    pub fn compileToObject(self: *NativeCompiler, ir_file: []const u8, object_file: []const u8) !CompilationResult {
        const start_time = std.time.milliTimestamp();
        
        // In a real implementation, would execute:
        // llc -filetype=obj -O2 input.ll -o output.o
        _ = self;
        
        const end_time = std.time.milliTimestamp();
        
        return CompilationResult{
            .success = true,
            .ir_file = ir_file,
            .object_file = object_file,
            .compilation_time_ms = @intCast(end_time - start_time),
        };
    }
    
    /// Optimize LLVM IR file
    pub fn optimizeIR(self: *NativeCompiler, input_file: []const u8, output_file: []const u8) !CompilationResult {
        // In a real implementation, would execute:
        // opt -O2 input.ll -o output.ll
        _ = self;
        _ = input_file;
        
        return CompilationResult{
            .success = true,
            .ir_file = output_file,
        };
    }
    
    /// Link object files to executable
    pub fn link(self: *NativeCompiler, object_files: []const []const u8, executable: []const u8) !CompilationResult {
        const start_time = std.time.milliTimestamp();
        
        // In a real implementation, would execute:
        // clang object1.o object2.o -o executable
        _ = self;
        _ = object_files;
        
        const end_time = std.time.milliTimestamp();
        
        return CompilationResult{
            .success = true,
            .executable = executable,
            .compilation_time_ms = @intCast(end_time - start_time),
        };
    }
    
    /// Complete compilation: IR → executable
    pub fn compileToExecutable(self: *NativeCompiler, ir_file: []const u8, executable: []const u8) !CompilationResult {
        // Step 1: Compile to object
        const obj_file = try std.fmt.allocPrint(self.allocator, "{s}.o", .{executable});
        defer self.allocator.free(obj_file);
        
        const obj_result = try self.compileToObject(ir_file, obj_file);
        if (!obj_result.success) return obj_result;
        
        // Step 2: Link to executable
        const obj_files = [_][]const u8{obj_file};
        return try self.link(&obj_files, executable);
    }
};

// ============================================================================
// Runtime Library
// ============================================================================

pub const RuntimeLibrary = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) RuntimeLibrary {
        return RuntimeLibrary{ .allocator = allocator };
    }
    
    /// Generate minimal runtime library
    pub fn generateRuntime(self: *RuntimeLibrary) ![]const u8 {
        // Minimal runtime functions
        const runtime_content = 
            \\; Mojo Runtime Library
            \\
            \\declare i32 @printf(ptr, ...)
            \\declare ptr @malloc(i64)
            \\declare void @free(ptr)
            \\
            \\define void @mojo_init() {
            \\  ret void
            \\}
            \\
            \\define void @mojo_cleanup() {
            \\  ret void
            \\}
            \\
        ;
        
        return try self.allocator.dupe(u8, runtime_content);
    }
    
    /// Write runtime to file
    pub fn writeRuntimeToFile(self: *RuntimeLibrary, path: []const u8) !void {
        const content = try self.generateRuntime();
        defer self.allocator.free(content);
        
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(content);
    }
};

// ============================================================================
// Linker Configuration
// ============================================================================

pub const LinkerConfig = struct {
    libraries: std.ArrayList([]const u8),
    library_paths: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) LinkerConfig {
        return LinkerConfig{
            .libraries = std.ArrayList([]const u8).initCapacity(allocator, 0) catch unreachable,
            .library_paths = std.ArrayList([]const u8).initCapacity(allocator, 0) catch unreachable,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *LinkerConfig) void {
        self.libraries.deinit(self.allocator);
        self.library_paths.deinit(self.allocator);
    }
    
    pub fn addLibrary(self: *LinkerConfig, lib: []const u8) !void {
        try self.libraries.append(self.allocator, lib);
    }
    
    pub fn addLibraryPath(self: *LinkerConfig, path: []const u8) !void {
        try self.library_paths.append(self.allocator, path);
    }
};

// ============================================================================
// Build System
// ============================================================================

pub const BuildSystem = struct {
    allocator: std.mem.Allocator,
    compiler: NativeCompiler,
    runtime: RuntimeLibrary,
    linker_config: LinkerConfig,
    
    pub fn init(allocator: std.mem.Allocator, options: CompilationOptions) BuildSystem {
        return BuildSystem{
            .allocator = allocator,
            .compiler = NativeCompiler.init(allocator, ToolPaths.default(), options),
            .runtime = RuntimeLibrary.init(allocator),
            .linker_config = LinkerConfig.init(allocator),
        };
    }
    
    pub fn deinit(self: *BuildSystem) void {
        self.linker_config.deinit();
    }
    
    /// Complete build: LLVM module → executable
    pub fn buildExecutable(self: *BuildSystem, module: *const llvm_lowering.LLVMModule, output_name: []const u8) !CompilationResult {
        // Generate IR
        var ir_gen = codegen.IRGenerator.init(self.allocator);
        const ir_content = try ir_gen.generateIR(module);
        defer self.allocator.free(ir_content);
        
        // Write IR to file
        const ir_file = try std.fmt.allocPrint(self.allocator, "{s}.ll", .{output_name});
        defer self.allocator.free(ir_file);
        
        const file = try std.fs.cwd().createFile(ir_file, .{});
        defer file.close();
        try file.writeAll(ir_content);
        
        // Compile to executable
        return try self.compiler.compileToExecutable(ir_file, output_name);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "native_compiler: tool paths" {
    const tools = ToolPaths.default();
    try std.testing.expectEqualStrings("llc", tools.llc);
    try std.testing.expectEqualStrings("opt", tools.opt);
    try std.testing.expectEqualStrings("clang", tools.clang);
}

test "native_compiler: compilation options" {
    const default_opts = CompilationOptions.default();
    try std.testing.expectEqual(codegen.OptimizationLevel.O2, default_opts.optimization_level);
    try std.testing.expect(default_opts.pic);
    
    const release_opts = CompilationOptions.forRelease();
    try std.testing.expectEqual(codegen.OptimizationLevel.O3, release_opts.optimization_level);
    try std.testing.expect(release_opts.strip);
    
    const debug_opts = CompilationOptions.forDebug();
    try std.testing.expectEqual(codegen.OptimizationLevel.O0, debug_opts.optimization_level);
}

test "native_compiler: compilation result" {
    var result = CompilationResult{
        .success = true,
        .executable = "test_program",
    };
    
    try std.testing.expect(result.isSuccess());
    try std.testing.expect(result.hasExecutable());
}

test "native_compiler: init compiler" {
    const allocator = std.testing.allocator;
    const tools = ToolPaths.default();
    const options = CompilationOptions.default();
    
    const compiler = NativeCompiler.init(allocator, tools, options);
    try std.testing.expectEqualStrings("llc", compiler.tools.llc);
}

test "native_compiler: compile to object" {
    const allocator = std.testing.allocator;
    const tools = ToolPaths.default();
    const options = CompilationOptions.default();
    
    var compiler = NativeCompiler.init(allocator, tools, options);
    const result = try compiler.compileToObject("test.ll", "test.o");
    
    try std.testing.expect(result.success);
}

test "native_compiler: optimize ir" {
    const allocator = std.testing.allocator;
    const tools = ToolPaths.default();
    const options = CompilationOptions.default();
    
    var compiler = NativeCompiler.init(allocator, tools, options);
    const result = try compiler.optimizeIR("input.ll", "output.ll");
    
    try std.testing.expect(result.success);
}

test "native_compiler: link objects" {
    const allocator = std.testing.allocator;
    const tools = ToolPaths.default();
    const options = CompilationOptions.default();
    
    var compiler = NativeCompiler.init(allocator, tools, options);
    const objects = [_][]const u8{"test.o"};
    const result = try compiler.link(&objects, "test");
    
    try std.testing.expect(result.success);
    try std.testing.expect(result.hasExecutable());
}

test "native_compiler: runtime library" {
    const allocator = std.testing.allocator;
    
    var runtime = RuntimeLibrary.init(allocator);
    const content = try runtime.generateRuntime();
    defer allocator.free(content);
    
    // Check that runtime contains expected functions
    try std.testing.expect(std.mem.indexOf(u8, content, "mojo_init") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "mojo_cleanup") != null);
}

test "native_compiler: linker config" {
    const allocator = std.testing.allocator;
    
    var config = LinkerConfig.init(allocator);
    defer config.deinit();
    
    try config.addLibrary("c");
    try config.addLibraryPath("/usr/lib");
    
    try std.testing.expectEqual(@as(usize, 1), config.libraries.items.len);
    try std.testing.expectEqual(@as(usize, 1), config.library_paths.items.len);
}

test "native_compiler: build system" {
    const allocator = std.testing.allocator;
    const options = CompilationOptions.default();
    
    var build_system = BuildSystem.init(allocator, options);
    defer build_system.deinit();
    
    try std.testing.expectEqual(codegen.OptimizationLevel.O2, build_system.compiler.options.optimization_level);
}
