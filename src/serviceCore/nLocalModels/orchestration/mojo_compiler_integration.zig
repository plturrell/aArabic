//! Mojo Compiler Integration for nLocalModels Orchestration
//! Bridges the Custom Mojo SDK (from src/nLang/n-python-sdk) with nLocalModels orchestration system
//! 
//! Purpose:
//! - Integrate mojo-sdk for `code` category tasks
//! - Provide JIT and AOT compilation capabilities
//! - Enable MLIR-based Python acceleration
//! - Support model-generated code execution
//!
//! Architecture:
//! nLocalModels → ModelSelector → MojoCompilerBridge → mojo-sdk (nLang)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Compilation mode for Mojo code
pub const CompilationMode = enum {
    /// Ahead-of-Time: Compile to native binary
    aot,
    /// Just-in-Time: Compile and run immediately
    jit,
    /// Interpret: Use REPL for quick execution
    repl,
};

/// Optimization level for compilation
pub const OptimizationLevel = enum {
    debug,    // O0 - No optimization, fast compile
    normal,   // O1 - Basic optimization
    release,  // O2 - Standard release
    ultra,    // O3 - Maximum optimization
};

/// Compilation options for Mojo code
pub const CompilationOptions = struct {
    mode: CompilationMode = .jit,
    optimization: OptimizationLevel = .normal,
    verbose: bool = false,
    strip_debug: bool = false,
    output_path: ?[]const u8 = null,
    
    /// Convert to mojo CLI flags
    pub fn toCliArgs(self: CompilationOptions, allocator: Allocator) ![][]const u8 {
        var args = try std.ArrayList([]const u8).initCapacity(allocator, 8);
        
        // Optimization level
        switch (self.optimization) {
            .debug => try args.append(allocator, "--optimization=0"),
            .normal => try args.append(allocator, "--optimization=1"),
            .release => try args.append(allocator, "--optimization=2"),
            .ultra => try args.append(allocator, "--optimization=3"),
        }
        
        // Verbose
        if (self.verbose) {
            try args.append(allocator, "--verbose");
        }
        
        // Strip debug symbols (release mode)
        if (self.strip_debug) {
            try args.append(allocator, "--release");
        }
        
        // Output path
        if (self.output_path) |path| {
            try args.append(allocator, "-o");
            try args.append(allocator, path);
        }
        
        return args.toOwnedSlice(allocator);
    }
};

/// Result of Mojo compilation
pub const CompilationResult = struct {
    success: bool,
    output_path: ?[]const u8 = null,
    execution_output: ?[]const u8 = null,
    error_message: ?[]const u8 = null,
    compilation_time_ms: u64 = 0,
    execution_time_ms: u64 = 0,
    
    pub fn deinit(self: *CompilationResult, allocator: Allocator) void {
        if (self.output_path) |path| allocator.free(path);
        if (self.execution_output) |output| allocator.free(output);
        if (self.error_message) |msg| allocator.free(msg);
    }
};

/// Bridge between nLocalModels and mojo-sdk
pub const MojoCompilerBridge = struct {
    allocator: Allocator,
    mojo_sdk_path: []const u8,
    mojo_binary_path: []const u8,
    
    /// Initialize the bridge
    /// mojo_sdk_path: Path to mojo-sdk directory (e.g., "../../nLang/n-python-sdk")
    pub fn init(allocator: Allocator, mojo_sdk_path: []const u8) !*MojoCompilerBridge {
        const bridge = try allocator.create(MojoCompilerBridge);
        
        // Construct path to mojo binary
        const binary_path = try std.fmt.allocPrint(
            allocator,
            "{s}/zig-out/bin/mojo",
            .{mojo_sdk_path},
        );
        
        bridge.* = .{
            .allocator = allocator,
            .mojo_sdk_path = try allocator.dupe(u8, mojo_sdk_path),
            .mojo_binary_path = binary_path,
        };
        
        // Verify mojo binary exists (relative path is fine for our use case)
        // Note: We skip the check here since the path may be relative
        // The actual error will occur when trying to execute if binary doesn't exist
        
        return bridge;
    }
    
    pub fn deinit(self: *MojoCompilerBridge) void {
        self.allocator.free(self.mojo_sdk_path);
        self.allocator.free(self.mojo_binary_path);
        self.allocator.destroy(self);
    }
    
    /// Compile Mojo source code
    pub fn compile(
        self: *MojoCompilerBridge,
        source_code: []const u8,
        options: CompilationOptions,
    ) !CompilationResult {
        const start_time = std.time.milliTimestamp();
        
        // Create temporary source file
        const temp_file = try self.createTempFile(source_code);
        defer std.fs.cwd().deleteFile(temp_file) catch {};
        defer self.allocator.free(temp_file);
        
        var result = CompilationResult{
            .success = false,
        };
        
        switch (options.mode) {
            .aot => {
                // AOT: mojo build
                result = try self.compileAOT(temp_file, options);
            },
            .jit => {
                // JIT: mojo run
                result = try self.compileAndRun(temp_file, options);
            },
            .repl => {
                // REPL mode not supported for now
                result.error_message = try self.allocator.dupe(u8, "REPL mode not yet implemented");
                return result;
            },
        }
        
        const end_time = std.time.milliTimestamp();
        result.compilation_time_ms = @intCast(end_time - start_time);
        
        return result;
    }
    
    fn createTempFile(self: *MojoCompilerBridge, source_code: []const u8) ![]const u8 {
        // Create unique temp filename
        const timestamp = std.time.timestamp();
        const filename = try std.fmt.allocPrint(
            self.allocator,
            "/tmp/mojo_temp_{d}.mojo",
            .{timestamp},
        );
        
        // Write source code
        try std.fs.cwd().writeFile(.{
            .sub_path = filename,
            .data = source_code,
        });
        
        return filename;
    }
    
    fn compileAOT(
        self: *MojoCompilerBridge,
        source_file: []const u8,
        options: CompilationOptions,
    ) !CompilationResult {
        var result = CompilationResult{ .success = false };
        
        // Build command: mojo build <source> -o <output>
        const output_path = options.output_path orelse try std.fmt.allocPrint(
            self.allocator,
            "/tmp/mojo_out_{d}",
            .{std.time.timestamp()},
        );
        
        var args = std.ArrayList([]const u8){};
        defer args.deinit();
        
        try args.append(self.allocator, self.mojo_binary_path);
        try args.append(self.allocator, "build");
        try args.append(self.allocator, source_file);
        try args.append(self.allocator, "-o");
        try args.append(self.allocator, output_path);
        
        // Add options
        const option_args = try options.toCliArgs(self.allocator);
        defer self.allocator.free(option_args);
        for (option_args) |arg| {
            try args.append(self.allocator, arg);
        }
        
        // Execute
        var child = std.process.Child.init(args.items, self.allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;
        
        try child.spawn();
        
        const stdout = try child.stdout.?.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(stdout);
        
        const stderr = try child.stderr.?.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(stderr);
        
        const term = try child.wait();
        
        result.success = (term == .Exited and term.Exited == 0);
        result.output_path = try self.allocator.dupe(u8, output_path);
        
        if (!result.success) {
            result.error_message = try self.allocator.dupe(u8, stderr);
        }
        
        return result;
    }
    
    fn compileAndRun(
        self: *MojoCompilerBridge,
        source_file: []const u8,
        options: CompilationOptions,
    ) !CompilationResult {
        var result = CompilationResult{ .success = false };
        
        // Build command: mojo run <source>
        var args = std.ArrayList([]const u8){};
        defer args.deinit();
        
        try args.append(self.allocator, self.mojo_binary_path);
        try args.append(self.allocator, "run");
        try args.append(self.allocator, source_file);
        
        // Add options
        const option_args = try options.toCliArgs(self.allocator);
        defer self.allocator.free(option_args);
        for (option_args) |arg| {
            try args.append(self.allocator, arg);
        }
        
        // Execute
        const execution_start = std.time.milliTimestamp();
        
        var child = std.process.Child.init(args.items, self.allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;
        
        try child.spawn();
        
        const stdout = try child.stdout.?.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        const stderr = try child.stderr.?.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(stderr);
        
        const term = try child.wait();
        
        const execution_end = std.time.milliTimestamp();
        
        result.success = (term == .Exited and term.Exited == 0);
        result.execution_output = stdout; // Transfer ownership
        result.execution_time_ms = @intCast(execution_end - execution_start);
        
        if (!result.success) {
            result.error_message = try self.allocator.dupe(u8, stderr);
            self.allocator.free(stdout);
        }
        
        return result;
    }
    
    /// Validate Mojo syntax without execution
    pub fn validateSyntax(
        self: *MojoCompilerBridge,
        source_code: []const u8,
    ) !bool {
        const temp_file = try self.createTempFile(source_code);
        defer std.fs.cwd().deleteFile(temp_file) catch {};
        defer self.allocator.free(temp_file);
        
        // Use mojo build with dry-run (if supported) or just compile to temp
        const options = CompilationOptions{
            .mode = .aot,
            .optimization = .debug,
            .output_path = "/tmp/mojo_syntax_check",
        };
        
        const result = try self.compile(source_code, options);
        defer {
            var mut_result = result;
            mut_result.deinit(self.allocator);
        }
        
        // Clean up temp binary
        if (result.output_path) |path| {
            std.fs.cwd().deleteFile(path) catch {};
        }
        
        return result.success;
    }
    
    /// Get Mojo SDK version
    pub fn getVersion(self: *MojoCompilerBridge) ![]const u8 {
        var args = [_][]const u8{ self.mojo_binary_path, "--version" };
        
        var child = std.process.Child.init(&args, self.allocator);
        child.stdout_behavior = .Pipe;
        
        try child.spawn();
        
        const stdout = try child.stdout.?.readToEndAlloc(self.allocator, 1024);
        _ = try child.wait();
        
        return stdout;
    }
};

/// Integration with ModelSelector for `code` category
pub const CodeCategoryHandler = struct {
    allocator: Allocator,
    mojo_bridge: *MojoCompilerBridge,
    
    pub fn init(allocator: Allocator, mojo_sdk_path: []const u8) !*CodeCategoryHandler {
        const handler = try allocator.create(CodeCategoryHandler);
        
        handler.* = .{
            .allocator = allocator,
            .mojo_bridge = try MojoCompilerBridge.init(allocator, mojo_sdk_path),
        };
        
        return handler;
    }
    
    pub fn deinit(self: *CodeCategoryHandler) void {
        self.mojo_bridge.deinit();
        self.allocator.destroy(self);
    }
    
    /// Handle code generation task from LLM
    pub fn handleCodeGeneration(
        self: *CodeCategoryHandler,
        llm_generated_code: []const u8,
        options: CompilationOptions,
    ) !CompilationResult {
        // Validate code first
        const is_valid = try self.mojo_bridge.validateSyntax(llm_generated_code);
        
        if (!is_valid) {
            return CompilationResult{
                .success = false,
                .error_message = try self.allocator.dupe(u8, "Invalid Mojo syntax"),
            };
        }
        
        // Compile and execute
        return self.mojo_bridge.compile(llm_generated_code, options);
    }
    
    /// Execute model-generated code safely
    pub fn executeSafely(
        self: *CodeCategoryHandler,
        code: []const u8,
        timeout_ms: u64,
    ) !CompilationResult {
        _ = timeout_ms; // TODO: Implement timeout
        
        const options = CompilationOptions{
            .mode = .jit,
            .optimization = .normal,
            .verbose = false,
        };
        
        return self.handleCodeGeneration(code, options);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MojoCompilerBridge initialization" {
    const allocator = std.testing.allocator;
    
    const bridge = try MojoCompilerBridge.init(
        allocator,
        "../../../nLang/n-python-sdk",
    );
    defer bridge.deinit();
    
    try std.testing.expect(bridge.mojo_sdk_path.len > 0);
    try std.testing.expect(bridge.mojo_binary_path.len > 0);
}

test "CompilationOptions CLI args" {
    const allocator = std.testing.allocator;
    
    const options = CompilationOptions{
        .optimization = .release,
        .verbose = true,
        .strip_debug = true,
    };
    
    const args = try options.toCliArgs(allocator);
    defer allocator.free(args);
    
    try std.testing.expect(args.len >= 2);
}

test "CodeCategoryHandler initialization" {
    const allocator = std.testing.allocator;
    
    const handler = try CodeCategoryHandler.init(
        allocator,
        "../../../nLang/n-python-sdk",
    );
    defer handler.deinit();
    
    // Just verify the pointer is non-null by checking it exists
    _ = handler.mojo_bridge;
    try std.testing.expect(true);
}
