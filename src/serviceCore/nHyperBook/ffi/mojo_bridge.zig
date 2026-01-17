///! Zig wrapper for Mojo FFI bridge
///! Provides type-safe Zig interface to Mojo runtime

const std = @import("std");
const c = @cImport({
    @cInclude("hypershimmy_ffi.h");
});

/// Result type for FFI operations
pub const Result = enum(c_int) {
    success = c.HS_SUCCESS,
    invalid_argument = c.HS_ERROR_INVALID_ARGUMENT,
    out_of_memory = c.HS_ERROR_OUT_OF_MEMORY,
    not_initialized = c.HS_ERROR_NOT_INITIALIZED,
    already_initialized = c.HS_ERROR_ALREADY_INITIALIZED,
    internal_error = c.HS_ERROR_INTERNAL,
    not_implemented = c.HS_ERROR_NOT_IMPLEMENTED,

    pub fn isSuccess(self: Result) bool {
        return self == .success;
    }

    pub fn toError(self: Result) !void {
        return switch (self) {
            .success => {},
            .invalid_argument => error.InvalidArgument,
            .out_of_memory => error.OutOfMemory,
            .not_initialized => error.NotInitialized,
            .already_initialized => error.AlreadyInitialized,
            .internal_error => error.InternalError,
            .not_implemented => error.NotImplemented,
        };
    }
};

/// Source type enumeration
pub const SourceType = enum(c_int) {
    url = c.HS_SOURCE_TYPE_URL,
    pdf = c.HS_SOURCE_TYPE_PDF,
    text = c.HS_SOURCE_TYPE_TEXT,
    file = c.HS_SOURCE_TYPE_FILE,
};

/// Source status enumeration
pub const SourceStatus = enum(c_int) {
    pending = c.HS_STATUS_PENDING,
    processing = c.HS_STATUS_PROCESSING,
    ready = c.HS_STATUS_READY,
    failed = c.HS_STATUS_FAILED,
};

/// FFI String wrapper
pub const FFIString = struct {
    inner: c.HSString,

    pub fn init(data: []const u8) FFIString {
        return .{
            .inner = .{
                .data = data.ptr,
                .length = data.len,
            },
        };
    }

    pub fn toSlice(self: FFIString) []const u8 {
        return self.inner.data[0..self.inner.length];
    }
};

/// FFI Buffer wrapper
pub const FFIBuffer = struct {
    inner: c.HSBuffer,

    pub fn init(data: []const u8) FFIBuffer {
        return .{
            .inner = .{
                .data = data.ptr,
                .length = data.len,
            },
        };
    }

    pub fn toSlice(self: FFIBuffer) []const u8 {
        return self.inner.data[0..self.inner.length];
    }
};

/// Mojo runtime context handle
pub const Context = struct {
    handle: *c.HSContext,

    /// Initialize the Mojo runtime
    pub fn init() !Context {
        var ctx: ?*c.HSContext = null;
        const result = @as(Result, @enumFromInt(c.hs_init(&ctx)));
        try result.toError();
        
        return Context{
            .handle = ctx orelse return error.InitializationFailed,
        };
    }

    /// Cleanup the Mojo runtime
    pub fn deinit(self: Context) void {
        _ = c.hs_cleanup(self.handle);
    }

    /// Check if the runtime is initialized
    pub fn isInitialized(self: Context) bool {
        return c.hs_is_initialized(self.handle);
    }

    /// Get the Mojo runtime version
    pub fn getVersion(self: Context, allocator: std.mem.Allocator) ![]const u8 {
        var version: c.HSString = undefined;
        const result = @as(Result, @enumFromInt(c.hs_get_version(self.handle, &version)));
        try result.toError();
        
        const version_slice = version.data[0..version.length];
        const owned = try allocator.dupe(u8, version_slice);
        
        // Free the string on the Mojo side
        _ = c.hs_string_free(self.handle, &version);
        
        return owned;
    }

    /// Get the last error message
    pub fn getLastError(self: Context, allocator: std.mem.Allocator) ![]const u8 {
        var error_msg: c.HSString = undefined;
        const result = @as(Result, @enumFromInt(c.hs_get_last_error(self.handle, &error_msg)));
        try result.toError();
        
        const error_slice = error_msg.data[0..error_msg.length];
        const owned = try allocator.dupe(u8, error_slice);
        
        // Free the string on the Mojo side
        _ = c.hs_string_free(self.handle, &error_msg);
        
        return owned;
    }

    /// Clear the last error
    pub fn clearError(self: Context) !void {
        const result = @as(Result, @enumFromInt(c.hs_clear_error(self.handle)));
        try result.toError();
    }

    // ========================================================================
    // Source Management (Day 8)
    // ========================================================================

    /// Create a new source
    pub fn createSource(
        self: Context,
        allocator: std.mem.Allocator,
        title: []const u8,
        source_type: SourceType,
        url: []const u8,
        content: []const u8,
    ) ![]const u8 {
        const title_str = FFIString.init(title);
        const url_str = FFIString.init(url);
        const content_str = FFIString.init(content);
        var source_id: c.HSString = undefined;

        const result = @as(Result, @enumFromInt(c.hs_source_create(
            self.handle,
            title_str.inner,
            @intFromEnum(source_type),
            url_str.inner,
            content_str.inner,
            &source_id,
        )));
        try result.toError();

        const id_slice = source_id.data[0..source_id.length];
        const owned = try allocator.dupe(u8, id_slice);

        // Free the string on the Mojo side
        _ = c.hs_string_free(self.handle, &source_id);

        return owned;
    }

    /// Delete a source
    pub fn deleteSource(self: Context, source_id: []const u8) !void {
        const id_str = FFIString.init(source_id);
        const result = @as(Result, @enumFromInt(c.hs_source_delete(
            self.handle,
            id_str.inner,
        )));
        try result.toError();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Context init and deinit" {
    const ctx = try Context.init();
    defer ctx.deinit();
    
    try std.testing.expect(ctx.isInitialized());
}

test "FFIString conversion" {
    const data = "Hello, Mojo!";
    const ffi_str = FFIString.init(data);
    const slice = ffi_str.toSlice();
    
    try std.testing.expectEqualStrings(data, slice);
}

test "FFIBuffer conversion" {
    const data = "Binary data";
    const ffi_buf = FFIBuffer.init(data);
    const slice = ffi_buf.toSlice();
    
    try std.testing.expectEqualStrings(data, slice);
}
