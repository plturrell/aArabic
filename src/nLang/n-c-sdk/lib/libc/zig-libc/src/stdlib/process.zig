// Process control functions for stdlib
// Phase 1.2 - Week 28
// Implements exit, atexit, abort, _Exit

const std = @import("std");

// Exit handler function type
pub const ExitHandlerFn = *const fn () callconv(.C) void;

// Exit handler storage (max 32 handlers)
var exit_handlers: [32]?ExitHandlerFn = [_]?ExitHandlerFn{null} ** 32;
var exit_handler_count: usize = 0;

/// Normal program termination
/// C signature: void exit(int status);
pub export fn exit(status: c_int) noreturn {
    // Call exit handlers in reverse order
    var i = exit_handler_count;
    while (i > 0) {
        i -= 1;
        if (exit_handlers[i]) |handler| {
            handler();
        }
    }
    
    // Call Zig's exit
    std.process.exit(@intCast(status));
}

/// Register exit handler
/// C signature: int atexit(void (*function)(void));
pub export fn atexit(function: ExitHandlerFn) c_int {
    if (exit_handler_count >= exit_handlers.len) {
        return -1; // Too many handlers
    }
    
    exit_handlers[exit_handler_count] = function;
    exit_handler_count += 1;
    return 0;
}

/// Abnormal program termination
/// C signature: void abort(void);
pub export fn abort() noreturn {
    // Don't call exit handlers for abort
    std.process.exit(134); // 128 + SIGABRT(6)
}

/// Exit without cleanup
/// C signature: void _Exit(int status);
pub export fn _Exit(status: c_int) noreturn {
    // Don't call exit handlers
    std.process.exit(@intCast(status));
}
