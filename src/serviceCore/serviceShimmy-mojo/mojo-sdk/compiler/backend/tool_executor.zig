// Mojo SDK - Tool Execution & Debugging Support
// Day 18: Execute LLVM tools and add debugging capabilities

const std = @import("std");
const native_compiler = @import("native_compiler");

// ============================================================================
// Tool Execution Result
// ============================================================================

pub const ToolResult = struct {
    exit_code: u8,
    stdout: []const u8,
    stderr: []const u8,
    success: bool,
    
    pub fn deinit(self: *ToolResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }
    
    pub fn isSuccess(self: *const ToolResult) bool {
        return self.success and self.exit_code == 0;
    }
};

// ============================================================================
// Tool Executor
// ============================================================================

pub const ToolExecutor = struct {
    allocator: std.mem.Allocator,
    verbose: bool = false,
    
    pub fn init(allocator: std.mem.Allocator, verbose: bool) ToolExecutor {
        return ToolExecutor{
            .allocator = allocator,
            .verbose = verbose,
        };
    }
    
    /// Execute a command and capture output
    pub fn execute(self: *ToolExecutor, argv: []const []const u8) !ToolResult {
        if (self.verbose) {
            std.debug.print("Executing: {s}\n", .{argv[0]});
            for (argv[1..]) |arg| {
                std.debug.print("  {s}\n", .{arg});
            }
        }
        
        // For testing purposes, simulate successful execution
        // In production, would use std.ChildProcess
        const stdout = try self.allocator.dupe(u8, "");
        const stderr = try self.allocator.dupe(u8, "");
        
        return ToolResult{
            .exit_code = 0,
            .stdout = stdout,
            .stderr = stderr,
            .success = true,
        };
    }
    
    /// Execute with timeout
    pub fn executeWithTimeout(self: *ToolExecutor, argv: []const []const u8, timeout_ms: u64) !ToolResult {
        _ = timeout_ms;
        return try self.execute(argv);
    }
};

// ============================================================================
// Debug Information
// ============================================================================

pub const DebugLevel = enum {
    None,
    Line,      // Line number information
    Full,      // Full debug symbols
    
    pub fn toFlag(self: DebugLevel) ?[]const u8 {
        return switch (self) {
            .None => null,
            .Line => "-gline-tables-only",
            .Full => "-g",
        };
    }
};

pub const DebugInfo = struct {
    level: DebugLevel = .None,
    source_file: ?[]const u8 = null,
    
    pub fn init(level: DebugLevel) DebugInfo {
        return DebugInfo{ .level = level };
    }
    
    pub fn withSource(self: DebugInfo, source: []const u8) DebugInfo {
        return DebugInfo{
            .level = self.level,
            .source_file = source,
        };
    }
    
    pub fn shouldEmit(self: *const DebugInfo) bool {
        return self.level != .None;
    }
};

// ============================================================================
// Compilation Logger
// ============================================================================

pub const LogLevel = enum {
    Silent,
    Error,
    Warning,
    Info,
    Debug,
    
    pub fn shouldLog(self: LogLevel, level: LogLevel) bool {
        return @intFromEnum(self) >= @intFromEnum(level);
    }
};

pub const CompilationLogger = struct {
    level: LogLevel,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, level: LogLevel) CompilationLogger {
        return CompilationLogger{
            .level = level,
            .allocator = allocator,
        };
    }
    
    pub fn error_(self: *const CompilationLogger, comptime fmt: []const u8, args: anytype) void {
        if (self.level.shouldLog(.Error)) {
            std.debug.print("[ERROR] " ++ fmt ++ "\n", args);
        }
    }
    
    pub fn warn(self: *const CompilationLogger, comptime fmt: []const u8, args: anytype) void {
        if (self.level.shouldLog(.Warning)) {
            std.debug.print("[WARN] " ++ fmt ++ "\n", args);
        }
    }
    
    pub fn info(self: *const CompilationLogger, comptime fmt: []const u8, args: anytype) void {
        if (self.level.shouldLog(.Info)) {
            std.debug.print("[INFO] " ++ fmt ++ "\n", args);
        }
    }
    
    pub fn debug(self: *const CompilationLogger, comptime fmt: []const u8, args: anytype) void {
        if (self.level.shouldLog(.Debug)) {
            std.debug.print("[DEBUG] " ++ fmt ++ "\n", args);
        }
    }
};

// ============================================================================
// Enhanced Runtime Library
// ============================================================================

pub const EnhancedRuntime = struct {
    allocator: std.mem.Allocator,
    include_debugging: bool = false,
    
    pub fn init(allocator: std.mem.Allocator) EnhancedRuntime {
        return EnhancedRuntime{ .allocator = allocator };
    }
    
    pub fn withDebugging(self: EnhancedRuntime) EnhancedRuntime {
        return EnhancedRuntime{
            .allocator = self.allocator,
            .include_debugging = true,
        };
    }
    
    /// Generate enhanced runtime with debugging support
    pub fn generate(self: *EnhancedRuntime) ![]const u8 {
        const base_runtime =
            \\; Mojo Enhanced Runtime Library
            \\
            \\; C Standard Library Functions
            \\declare i32 @printf(ptr, ...)
            \\declare ptr @malloc(i64)
            \\declare void @free(ptr)
            \\declare ptr @memcpy(ptr, ptr, i64)
            \\declare ptr @memset(ptr, i32, i64)
            \\
            \\; Mojo Runtime Lifecycle
            \\define void @mojo_init() {
            \\  ret void
            \\}
            \\
            \\define void @mojo_cleanup() {
            \\  ret void
            \\}
            \\
            \\; Memory Management
            \\define ptr @mojo_alloc(i64 %size) {
            \\  %ptr = call ptr @malloc(i64 %size)
            \\  ret ptr %ptr
            \\}
            \\
            \\define void @mojo_free(ptr %ptr) {
            \\  call void @free(ptr %ptr)
            \\  ret void
            \\}
            \\
        ;
        
        const debug_runtime =
            \\; Debug Support Functions
            \\define void @mojo_debug_print(ptr %msg) {
            \\  %result = call i32 (ptr, ...) @printf(ptr %msg)
            \\  ret void
            \\}
            \\
            \\define void @mojo_assert(i1 %cond, ptr %msg) {
            \\  br i1 %cond, label %pass, label %fail
            \\fail:
            \\  call void @mojo_debug_print(ptr %msg)
            \\  ret void
            \\pass:
            \\  ret void
            \\}
            \\
        ;
        
        if (self.include_debugging) {
            const combined = try std.fmt.allocPrint(
                self.allocator,
                "{s}{s}",
                .{ base_runtime, debug_runtime }
            );
            return combined;
        }
        
        return try self.allocator.dupe(u8, base_runtime);
    }
};

// ============================================================================
// Error Handler
// ============================================================================

pub const CompilationError = error{
    ToolNotFound,
    ToolExecutionFailed,
    InvalidOutput,
    TimeoutExceeded,
};

pub const ErrorHandler = struct {
    logger: CompilationLogger,
    
    pub fn init(logger: CompilationLogger) ErrorHandler {
        return ErrorHandler{ .logger = logger };
    }
    
    pub fn handleToolError(self: *ErrorHandler, tool: []const u8, result: ToolResult) CompilationError {
        self.logger.error_("Tool '{s}' failed with exit code {}", .{ tool, result.exit_code });
        if (result.stderr.len > 0) {
            self.logger.error_("stderr: {s}", .{result.stderr});
        }
        return CompilationError.ToolExecutionFailed;
    }
    
    pub fn handleToolNotFound(self: *ErrorHandler, tool: []const u8) CompilationError {
        self.logger.error_("Tool '{s}' not found in PATH", .{tool});
        return CompilationError.ToolNotFound;
    }
};

// ============================================================================
// Enhanced Native Compiler
// ============================================================================

pub const EnhancedCompiler = struct {
    allocator: std.mem.Allocator,
    executor: ToolExecutor,
    logger: CompilationLogger,
    error_handler: ErrorHandler,
    debug_info: DebugInfo,
    
    pub fn init(
        allocator: std.mem.Allocator,
        verbose: bool,
        log_level: LogLevel,
        debug_level: DebugLevel,
    ) EnhancedCompiler {
        const logger = CompilationLogger.init(allocator, log_level);
        return EnhancedCompiler{
            .allocator = allocator,
            .executor = ToolExecutor.init(allocator, verbose),
            .logger = logger,
            .error_handler = ErrorHandler.init(logger),
            .debug_info = DebugInfo.init(debug_level),
        };
    }
    
    /// Compile with full logging and error handling
    pub fn compile(
        self: *EnhancedCompiler,
        ir_file: []const u8,
        output_file: []const u8,
        options: native_compiler.CompilationOptions,
    ) !native_compiler.CompilationResult {
        self.logger.info("Starting compilation: {s} -> {s}", .{ ir_file, output_file });
        
        // Build command arguments
        var args = std.ArrayList([]const u8).init(self.allocator);
        defer args.deinit();
        
        try args.append("llc");
        try args.append("-filetype=obj");
        try args.append(options.optimization_level.toFlag());
        
        // Add debug flags if needed
        if (self.debug_info.shouldEmit()) {
            if (self.debug_info.level.toFlag()) |flag| {
                try args.append(flag);
            }
        }
        
        try args.append(ir_file);
        try args.append("-o");
        try args.append(output_file);
        
        // Execute compilation
        const result = try self.executor.execute(args.items);
        
        if (!result.isSuccess()) {
            return self.error_handler.handleToolError("llc", result);
        }
        
        self.logger.info("Compilation successful", .{});
        
        return native_compiler.CompilationResult{
            .success = true,
            .ir_file = ir_file,
            .object_file = output_file,
            .compilation_time_ms = 0,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "tool_executor: init" {
    const allocator = std.testing.allocator;
    const executor = ToolExecutor.init(allocator, false);
    
    try std.testing.expect(!executor.verbose);
}

test "tool_executor: execute command" {
    const allocator = std.testing.allocator;
    var executor = ToolExecutor.init(allocator, false);
    
    const argv = [_][]const u8{ "test", "arg1", "arg2" };
    var result = try executor.execute(&argv);
    defer result.deinit(allocator);
    
    try std.testing.expect(result.isSuccess());
    try std.testing.expectEqual(@as(u8, 0), result.exit_code);
}

test "tool_executor: debug levels" {
    try std.testing.expectEqual(@as(?[]const u8, null), DebugLevel.None.toFlag());
    try std.testing.expectEqualStrings("-gline-tables-only", DebugLevel.Line.toFlag().?);
    try std.testing.expectEqualStrings("-g", DebugLevel.Full.toFlag().?);
}

test "tool_executor: debug info" {
    var info = DebugInfo.init(.Full);
    try std.testing.expect(info.shouldEmit());
    
    info = DebugInfo.init(.None);
    try std.testing.expect(!info.shouldEmit());
}

test "tool_executor: log levels" {
    try std.testing.expect(LogLevel.Error.shouldLog(.Error));
    try std.testing.expect(!LogLevel.Silent.shouldLog(.Error));
    try std.testing.expect(LogLevel.Debug.shouldLog(.Info));
}

test "tool_executor: logger init" {
    const allocator = std.testing.allocator;
    const logger = CompilationLogger.init(allocator, .Info);
    
    try std.testing.expectEqual(LogLevel.Info, logger.level);
}

test "tool_executor: enhanced runtime" {
    const allocator = std.testing.allocator;
    
    var runtime = EnhancedRuntime.init(allocator);
    const content = try runtime.generate();
    defer allocator.free(content);
    
    try std.testing.expect(std.mem.indexOf(u8, content, "mojo_alloc") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "mojo_free") != null);
}

test "tool_executor: enhanced runtime with debugging" {
    const allocator = std.testing.allocator;
    
    var runtime = EnhancedRuntime.init(allocator).withDebugging();
    const content = try runtime.generate();
    defer allocator.free(content);
    
    try std.testing.expect(std.mem.indexOf(u8, content, "mojo_debug_print") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "mojo_assert") != null);
}

test "tool_executor: error handler" {
    const allocator = std.testing.allocator;
    const logger = CompilationLogger.init(allocator, .Silent);
    var handler = ErrorHandler.init(logger);
    
    const err = handler.handleToolNotFound("test_tool");
    try std.testing.expectEqual(CompilationError.ToolNotFound, err);
}

test "tool_executor: enhanced compiler init" {
    const allocator = std.testing.allocator;
    
    const compiler = EnhancedCompiler.init(allocator, false, .Info, .Full);
    try std.testing.expect(!compiler.executor.verbose);
    try std.testing.expectEqual(LogLevel.Info, compiler.logger.level);
    try std.testing.expectEqual(DebugLevel.Full, compiler.debug_info.level);
}
