// Mojo Runtime Library
// Unified module that re-exports all runtime components
//
// This module provides the public interface to the Mojo runtime.

pub const core = @import("core.zig");
pub const memory = @import("memory.zig");
pub const ffi = @import("ffi.zig");
pub const startup = @import("startup.zig");
pub const async_runtime = @import("async_runtime.zig");
pub const timer = @import("timer.zig");
pub const blocking = @import("blocking_pool.zig");

// Re-export commonly used items at top level
pub const RuntimeConfig = core.RuntimeConfig;
pub const RuntimeAllocator = core.RuntimeAllocator;
pub const MojoString = memory.MojoString;
pub const MojoList = memory.MojoList;
pub const MojoDict = memory.MojoDict;
pub const MojoSet = memory.MojoSet;
pub const TypeConverter = ffi.TypeConverter;
pub const DynLib = ffi.DynLib;
pub const Args = startup.Args;
pub const Env = startup.Env;
pub const ExitCode = startup.ExitCode;

// Async exports
pub const Task = async_runtime.Task;
pub const Executor = async_runtime.Executor;
pub const Future = async_runtime.Future;
pub const TaskHandle = async_runtime.TaskHandle;
pub const TimerDriver = timer.TimerDriver;
pub const BlockingPool = blocking.BlockingPool;
pub const BlockingFn = blocking.BlockingFn;

/// Initialize the runtime with default settings
pub fn init() !void {
    try core.initDefault();
}

/// Initialize the runtime with custom configuration
pub fn initWithConfig(config: RuntimeConfig) !void {
    try core.init(config);
}

/// Deinitialize the runtime
pub fn deinit() void {
    core.deinit();
}

/// Get the global allocator
pub fn getAllocator() *RuntimeAllocator {
    return core.getAllocator();
}

/// Print runtime statistics
pub fn printStats() void {
    core.stats.print();
}

test "runtime module exports" {
    try init();
    defer deinit();

    // Test core exports
    const allocator = getAllocator();
    const data = try allocator.alloc(u8, 32);
    defer allocator.free(u8, data);

    // Test memory exports
    var str = try MojoString.fromSlice("test");
    defer str.deinit();

    // Test ffi exports
    const c_val = TypeConverter.boolToC(true);
    _ = c_val;
}
