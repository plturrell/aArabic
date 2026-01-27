// dlfcn module - Phase 1.22
// Dynamic linking functions using Zig's DynLib
const std = @import("std");
const builtin = @import("builtin");

// POSIX dlfcn flags
pub const RTLD_LAZY: c_int = 1;
pub const RTLD_NOW: c_int = 2;
pub const RTLD_GLOBAL: c_int = 0x100;
pub const RTLD_LOCAL: c_int = 0;
pub const RTLD_NODELETE: c_int = 0x1000;
pub const RTLD_NOLOAD: c_int = 0x4;
pub const RTLD_DEEPBIND: c_int = 0x8;

// Handle storage - maps opaque handles to DynLib instances
const MAX_HANDLES = 256;
var handles: [MAX_HANDLES]?std.DynLib = [_]?std.DynLib{null} ** MAX_HANDLES;
var handle_count: usize = 0;

// Thread-local error message
threadlocal var last_error: ?[]const u8 = null;
threadlocal var error_buf: [256]u8 = undefined;

fn setError(msg: []const u8) void {
    const len = @min(msg.len, error_buf.len - 1);
    @memcpy(error_buf[0..len], msg[0..len]);
    error_buf[len] = 0;
    last_error = error_buf[0..len];
}

fn clearError() void {
    last_error = null;
}

/// Open a dynamic library
pub export fn dlopen(filename: ?[*:0]const u8, flag: c_int) ?*anyopaque {
    _ = flag; // Zig's DynLib doesn't support lazy binding flags
    clearError();

    // Find a free handle slot
    var slot: usize = 0;
    while (slot < MAX_HANDLES) : (slot += 1) {
        if (handles[slot] == null) break;
    }

    if (slot >= MAX_HANDLES) {
        setError("Too many open libraries");
        return null;
    }

    // Convert filename
    const path: ?[:0]const u8 = if (filename) |f| std.mem.span(f) else null;

    if (path == null) {
        // NULL filename means return handle to main program
        // Return a special sentinel value
        return @ptrFromInt(0xFFFFFFFF);
    }

    // Open the library
    const lib = std.DynLib.open(path.?) catch |err| {
        const msg = switch (err) {
            error.FileNotFound => "Library not found",
            error.PermissionDenied => "Permission denied",
            else => "Failed to open library",
        };
        setError(msg);
        return null;
    };

    handles[slot] = lib;
    handle_count += 1;

    // Return handle as pointer (slot + 1 to avoid null)
    return @ptrFromInt(slot + 1);
}

/// Close a dynamic library
pub export fn dlclose(handle: ?*anyopaque) c_int {
    clearError();

    if (handle == null) {
        setError("Invalid handle");
        return -1;
    }

    const handle_val = @intFromPtr(handle);

    // Special case: main program handle
    if (handle_val == 0xFFFFFFFF) {
        return 0;
    }

    if (handle_val == 0 or handle_val > MAX_HANDLES) {
        setError("Invalid handle");
        return -1;
    }

    const slot = handle_val - 1;
    if (handles[slot]) |*lib| {
        lib.close();
        handles[slot] = null;
        handle_count -= 1;
        return 0;
    }

    setError("Handle not open");
    return -1;
}

/// Look up a symbol in a dynamic library
pub export fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque {
    clearError();

    if (handle == null) {
        setError("Invalid handle");
        return null;
    }

    const handle_val = @intFromPtr(handle);
    const sym_name = std.mem.span(symbol);

    // Special case: main program handle (not fully supported)
    if (handle_val == 0xFFFFFFFF) {
        setError("Symbol lookup in main program not supported");
        return null;
    }

    if (handle_val == 0 or handle_val > MAX_HANDLES) {
        setError("Invalid handle");
        return null;
    }

    const slot = handle_val - 1;
    if (handles[slot]) |lib| {
        const ptr = lib.lookup(*anyopaque, sym_name);
        if (ptr == null) {
            setError("Symbol not found");
        }
        return ptr;
    }

    setError("Handle not open");
    return null;
}

/// Return error message from last dl* call
pub export fn dlerror() ?[*:0]u8 {
    if (last_error) |_| {
        const result: [*:0]u8 = @ptrCast(&error_buf);
        last_error = null; // Clear after returning
        return result;
    }
    return null;
}

/// Dl_info structure for dladdr
pub const Dl_info = extern struct {
    dli_fname: ?[*:0]const u8,
    dli_fbase: ?*anyopaque,
    dli_sname: ?[*:0]const u8,
    dli_saddr: ?*anyopaque,
};

/// Get information about an address (limited implementation)
pub export fn dladdr(addr: ?*const anyopaque, info: ?*Dl_info) c_int {
    _ = addr;

    if (info) |i| {
        i.dli_fname = null;
        i.dli_fbase = null;
        i.dli_sname = null;
        i.dli_saddr = null;
    }

    // Not fully implemented - would need debug info parsing
    return 0;
}
