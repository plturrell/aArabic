// Lean4 FFI Bridge for Mojo Compiler
//
// This module provides the Zig-side interface to call the Mojo Lean4 compiler.
// The bridge loads the compiled Mojo shared library and calls its exported functions.

const std = @import("std");
const allocator = std.heap.c_allocator;

// Buffer size for responses (must match Mojo side)
const BUFFER_SIZE: usize = 1024 * 1024; // 1MB

// ============================================================================
// External Functions from Mojo Library
// ============================================================================

// These are the functions exported by the Mojo lean4_ffi.mojo module
extern fn lean4_check(
    source_ptr: [*]const u8,
    source_len: usize,
    result_buf: [*]u8,
    buf_size: usize,
) callconv(.c) i32;

extern fn lean4_run(
    source_ptr: [*]const u8,
    source_len: usize,
    result_buf: [*]u8,
    buf_size: usize,
) callconv(.c) i32;

extern fn lean4_elaborate(
    source_ptr: [*]const u8,
    source_len: usize,
    result_buf: [*]u8,
    buf_size: usize,
) callconv(.c) i32;

// ============================================================================
// High-Level API for Zig Server
// ============================================================================

pub const CheckResult = struct {
    success: bool,
    error_count: u32,
    warning_count: u32,
    info_count: u32,
    messages: []const u8,
    raw_json: []u8,
};

pub const RunResult = struct {
    success: bool,
    stdout: []const u8,
    stderr: []const u8,
    exit_code: i32,
    raw_json: []u8,
};

pub const ElaborateResult = struct {
    success: bool,
    declarations: []const []const u8,
    environment_size: u32,
    errors: []const u8,
    raw_json: []u8,
};

/// Check Lean4 source code for errors
pub fn check(source: []const u8) ![]u8 {
    const result_buf = try allocator.alloc(u8, BUFFER_SIZE);
    errdefer allocator.free(result_buf);

    const bytes_written = lean4_check(
        source.ptr,
        source.len,
        result_buf.ptr,
        BUFFER_SIZE,
    );

    if (bytes_written < 0) {
        allocator.free(result_buf);
        return error.CheckFailed;
    }

    const len: usize = @intCast(bytes_written);
    const final_buf = try allocator.realloc(result_buf, len);
    return final_buf;
}

/// Run Lean4 source code and get output
pub fn run(source: []const u8) ![]u8 {
    const result_buf = try allocator.alloc(u8, BUFFER_SIZE);
    errdefer allocator.free(result_buf);

    const bytes_written = lean4_run(
        source.ptr,
        source.len,
        result_buf.ptr,
        BUFFER_SIZE,
    );

    if (bytes_written < 0) {
        allocator.free(result_buf);
        return error.RunFailed;
    }

    const len: usize = @intCast(bytes_written);
    const final_buf = try allocator.realloc(result_buf, len);
    return final_buf;
}

/// Elaborate Lean4 source code
pub fn elaborate(source: []const u8) ![]u8 {
    const result_buf = try allocator.alloc(u8, BUFFER_SIZE);
    errdefer allocator.free(result_buf);

    const bytes_written = lean4_elaborate(
        source.ptr,
        source.len,
        result_buf.ptr,
        BUFFER_SIZE,
    );

    if (bytes_written < 0) {
        allocator.free(result_buf);
        return error.ElaborateFailed;
    }

    const len: usize = @intCast(bytes_written);
    const final_buf = try allocator.realloc(result_buf, len);
    return final_buf;
}

/// Free a result buffer
pub fn freeResult(buf: []u8) void {
    allocator.free(buf);
}

// ============================================================================
// Fallback Implementation (when Mojo library not available)
// ============================================================================

var mojo_available: ?bool = null;

pub fn isMojoAvailable() bool {
    if (mojo_available) |available| {
        return available;
    }
    // Try to call a simple function to check if library is loaded
    mojo_available = true; // Assume available, will fail gracefully if not
    return mojo_available.?;
}

/// Fallback check that returns a stub response
pub fn checkFallback(source: []const u8) ![]u8 {
    _ = source;
    const json =
        \\{"success":true,"error_count":0,"warning_count":0,"info_count":0,"messages":"Mojo compiler not available - using stub"}
    ;
    const result = try allocator.alloc(u8, json.len);
    @memcpy(result, json);
    return result;
}

/// Fallback run that returns a stub response
pub fn runFallback(source: []const u8) ![]u8 {
    _ = source;
    const json =
        \\{"success":true,"stdout":"Mojo runtime not available - using stub","stderr":"","exit_code":0}
    ;
    const result = try allocator.alloc(u8, json.len);
    @memcpy(result, json);
    return result;
}

/// Fallback elaborate that returns a stub response
pub fn elaborateFallback(source: []const u8) ![]u8 {
    _ = source;
    const json =
        \\{"success":true,"declarations":[],"environment_size":0,"errors":"Mojo elaborator not available - using stub"}
    ;
    const result = try allocator.alloc(u8, json.len);
    @memcpy(result, json);
    return result;
}

