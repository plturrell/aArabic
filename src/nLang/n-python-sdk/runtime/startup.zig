// Mojo Runtime Startup
// Entry point and initialization for compiled Mojo programs
//
// This module handles:
// - Program entry point (_start or main)
// - Runtime initialization before user code
// - Command line argument handling
// - Exit handling and cleanup

const std = @import("std");
const core = @import("core.zig");
const memory = @import("memory.zig");
const ffi = @import("ffi.zig");
const builtin = @import("builtin");

// ============================================================================
// Command Line Arguments
// ============================================================================

pub const Args = struct {
    args: []const [:0]const u8,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Get argument count
    pub fn count(self: *const Self) usize {
        return self.args.len;
    }

    /// Get argument at index
    pub fn get(self: *const Self, index: usize) ?[:0]const u8 {
        if (index >= self.args.len) return null;
        return self.args[index];
    }

    /// Get argument as MojoString
    pub fn getString(self: *const Self, index: usize) !?memory.MojoString {
        if (self.get(index)) |arg| {
            return try memory.MojoString.fromSlice(arg);
        }
        return null;
    }

    /// Get all arguments as a MojoList of MojoStrings
    pub fn toList(self: *const Self) !memory.MojoList(memory.MojoString) {
        var list = try memory.MojoList(memory.MojoString).withCapacity(self.args.len);

        for (self.args) |arg| {
            const str = try memory.MojoString.fromSlice(arg);
            try list.append(str);
        }

        return list;
    }
};

var global_args: ?Args = null;

/// Get command line arguments
pub fn getArgs() *const Args {
    if (global_args == null) {
        @panic("Arguments not initialized. Runtime startup failed.");
    }
    return &global_args.?;
}

// ============================================================================
// Environment Variables
// ============================================================================

pub const Env = struct {
    /// Get environment variable
    pub fn get(key: []const u8) ?[]const u8 {
        return std.posix.getenv(key);
    }

    /// Get environment variable as MojoString
    pub fn getString(key: []const u8) !?memory.MojoString {
        if (get(key)) |value| {
            return try memory.MojoString.fromSlice(value);
        }
        return null;
    }

    /// Check if environment variable exists
    pub fn has(key: []const u8) bool {
        return get(key) != null;
    }
};

// ============================================================================
// Exit Handling
// ============================================================================

/// Exit codes
pub const ExitCode = enum(u8) {
    Success = 0,
    GeneralError = 1,
    MisuseOfCommand = 2,
    RuntimeError = 3,
    OutOfMemory = 4,
    Panic = 5,

    pub fn toInt(self: ExitCode) u8 {
        return @intFromEnum(self);
    }
};

/// Exit the program with a code
pub fn exit(code: ExitCode) noreturn {
    exitWithCode(code.toInt());
}

/// Exit with an arbitrary code
pub fn exitWithCode(code: u8) noreturn {
    // Cleanup runtime
    cleanup();

    // Exit
    std.process.exit(code);
}

/// Exit handlers
var exit_handlers: std.ArrayListUnmanaged(*const fn () void) = .{};
var exit_handlers_initialized: bool = false;

fn initExitHandlers() void {
    if (!exit_handlers_initialized) {
        exit_handlers = .{};
        exit_handlers_initialized = true;
    }
}

/// Register an exit handler
pub fn atExit(handler: *const fn () void) !void {
    initExitHandlers();
    try exit_handlers.append(std.heap.page_allocator, handler);
}

/// Run all exit handlers
fn runExitHandlers() void {
    if (!exit_handlers_initialized) return;

    // Run in reverse order (LIFO)
    var i = exit_handlers.items.len;
    while (i > 0) {
        i -= 1;
        exit_handlers.items[i]();
    }
}

// ============================================================================
// Panic Handling
// ============================================================================

/// Custom panic handler for Mojo programs
pub fn panic(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = error_return_trace;
    _ = ret_addr;

    std.debug.print("\n=== MOJO PANIC ===\n", .{});
    std.debug.print("{s}\n", .{msg});

    // Print stack trace if available
    if (builtin.mode == .Debug) {
        std.debug.print("\nStack trace:\n", .{});
        std.debug.dumpCurrentStackTrace(null);
    }

    std.debug.print("==================\n\n", .{});

    // Run exit handlers
    runExitHandlers();

    // Cleanup runtime
    cleanup();

    std.process.exit(ExitCode.Panic.toInt());
}

// ============================================================================
// Startup and Cleanup
// ============================================================================

var startup_complete: bool = false;

/// Initialize the Mojo runtime (called before main)
pub fn startup(args: []const [:0]const u8) !void {
    if (startup_complete) return;

    // Initialize runtime core
    try core.init(.{
        .enable_stats = builtin.mode == .Debug,
        .enable_poisoning = builtin.mode == .Debug,
    });

    // Store arguments
    global_args = .{
        .args = args,
        .allocator = std.heap.page_allocator,
    };

    // Initialize exit handlers
    initExitHandlers();

    startup_complete = true;
}

/// Cleanup the Mojo runtime (called after main)
pub fn cleanup() void {
    if (!startup_complete) return;

    // Run exit handlers
    runExitHandlers();

    // Deinitialize exit handlers
    if (exit_handlers_initialized) {
        exit_handlers.deinit(std.heap.page_allocator);
        exit_handlers_initialized = false;
    }

    // Deinitialize runtime core
    core.deinit();

    global_args = null;
    startup_complete = false;
}

// ============================================================================
// Main Entry Point Generator
// ============================================================================

/// Generate a main function that wraps user's main
pub fn generateMain(comptime userMain: fn () anyerror!void) fn () void {
    return struct {
        fn main() void {
            // Get process arguments
            const args = std.process.argsAlloc(std.heap.page_allocator) catch {
                std.debug.print("Failed to get command line arguments\n", .{});
                std.process.exit(ExitCode.RuntimeError.toInt());
            };
            defer std.process.argsFree(std.heap.page_allocator, args);

            // Initialize runtime
            startup(args) catch |err| {
                std.debug.print("Failed to initialize Mojo runtime: {}\n", .{err});
                std.process.exit(ExitCode.RuntimeError.toInt());
            };

            // Call user's main
            userMain() catch |err| {
                std.debug.print("Unhandled error in main: {}\n", .{err});
                cleanup();
                std.process.exit(ExitCode.GeneralError.toInt());
            };

            // Cleanup
            cleanup();
        }
    }.main;
}

/// Generate main with int return type
pub fn generateMainWithReturn(comptime userMain: fn () anyerror!i32) fn () u8 {
    return struct {
        fn main() u8 {
            // Get process arguments (already sentinel-terminated from std.process)
            const args = std.process.argsAlloc(std.heap.page_allocator) catch {
                std.debug.print("Failed to get command line arguments\n", .{});
                return ExitCode.RuntimeError.toInt();
            };
            defer std.process.argsFree(std.heap.page_allocator, args);

            // Initialize runtime
            startup(args) catch |err| {
                std.debug.print("Failed to initialize Mojo runtime: {}\n", .{err});
                return ExitCode.RuntimeError.toInt();
            };

            // Call user's main
            const result = userMain() catch |err| {
                std.debug.print("Unhandled error in main: {}\n", .{err});
                cleanup();
                return ExitCode.GeneralError.toInt();
            };

            // Cleanup
            cleanup();

            return @intCast(@as(u32, @bitCast(result)) & 0xFF);
        }
    }.main;
}

// ============================================================================
// C ABI Entry Points
// ============================================================================

/// Type for user's main function
pub const MojoMainFn = *const fn () callconv(.c) i32;

var user_main_fn: ?MojoMainFn = null;

/// Set the user's main function (called by generated code)
export fn mojo_set_main(main_fn: MojoMainFn) callconv(.c) void {
    user_main_fn = main_fn;
}

/// C entry point - initializes runtime and calls user's main
export fn mojo_main(argc: c_int, argv: [*][*:0]const u8) callconv(.c) c_int {
    // Convert C args to Zig slices
    const args_slice = argv[0..@intCast(argc)];

    var sentinel_args: std.ArrayListUnmanaged([:0]const u8) = .{};
    defer sentinel_args.deinit(std.heap.page_allocator);

    for (args_slice) |arg| {
        sentinel_args.append(std.heap.page_allocator, std.mem.span(arg)) catch return -1;
    }

    // Initialize runtime
    startup(sentinel_args.items) catch return -1;

    // Call user's main if set
    var result: i32 = 0;
    if (user_main_fn) |main_fn| {
        result = main_fn();
    }

    // Cleanup
    cleanup();

    return result;
}

/// Get argument count
export fn mojo_argc() callconv(.c) c_int {
    if (global_args) |args| {
        return @intCast(args.count());
    }
    return 0;
}

/// Get argument at index
export fn mojo_argv(index: c_int) callconv(.c) ?[*:0]const u8 {
    if (global_args) |args| {
        if (args.get(@intCast(index))) |arg| {
            return arg.ptr;
        }
    }
    return null;
}

/// Get environment variable
export fn mojo_getenv(key: [*:0]const u8) callconv(.c) ?[*:0]const u8 {
    const key_slice = std.mem.span(key);
    if (Env.get(key_slice)) |value| {
        // Note: This returns a pointer to the environment, which is safe
        // as long as setenv isn't called
        return @ptrCast(value.ptr);
    }
    return null;
}

/// Exit with code
export fn mojo_exit(code: c_int) callconv(.c) noreturn {
    exitWithCode(@intCast(@as(u32, @bitCast(code)) & 0xFF));
}

/// Register exit handler
export fn mojo_atexit(handler: *const fn () callconv(.c) void) callconv(.c) c_int {
    // Wrap C callback
    const wrapper = struct {
        var c_handler: *const fn () callconv(.c) void = undefined;

        fn call() void {
            c_handler();
        }
    };

    wrapper.c_handler = handler;
    atExit(&wrapper.call) catch return -1;
    return 0;
}

// ============================================================================
// Tests
// ============================================================================

test "startup and cleanup" {
    const test_args = [_][:0]const u8{ "test_program", "--flag", "value" };
    try startup(&test_args);
    defer cleanup();

    const args = getArgs();
    try std.testing.expect(args.count() == 3);
    try std.testing.expectEqualStrings("test_program", args.get(0).?);
    try std.testing.expectEqualStrings("--flag", args.get(1).?);
}

test "environment" {
    try core.initDefault();
    defer core.deinit();

    // PATH should exist on most systems
    const path = Env.get("PATH");
    try std.testing.expect(path != null);
}

test "exit handlers" {
    const test_args = [_][:0]const u8{"test"};
    try startup(&test_args);
    defer cleanup();

    var handler_called = false;
    const handler = struct {
        var called: *bool = undefined;

        fn handle() void {
            called.* = true;
        }
    };

    handler.called = &handler_called;
    try atExit(&handler.handle);

    runExitHandlers();
    try std.testing.expect(handler_called);
}

test "args to list" {
    const test_args = [_][:0]const u8{ "prog", "arg1", "arg2" };
    try startup(&test_args);
    defer cleanup();

    const args = getArgs();
    var list = try args.toList();
    defer {
        for (list.asMutSlice()) |*str| {
            str.deinit();
        }
        list.deinit();
    }

    try std.testing.expect(list.length() == 3);
}
