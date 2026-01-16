// Mojo Runtime FFI (Foreign Function Interface)
// Bridge between Mojo and C code
//
// Provides mechanisms for:
// - Calling C functions from Mojo
// - Exposing Mojo functions to C
// - Type marshalling between Mojo and C
// - Dynamic library loading

const std = @import("std");
const core = @import("core.zig");
const memory = @import("memory.zig");
const builtin = @import("builtin");

// ============================================================================
// Type Conversions
// ============================================================================

/// Convert Mojo types to C-compatible representations
pub const TypeConverter = struct {
    /// Convert MojoString to C string (borrowed, do not free)
    pub fn stringToC(str: *const memory.MojoString) [*:0]const u8 {
        if (str.isSSO()) {
            return @ptrCast(&str.data.small);
        } else {
            return @ptrCast(str.data.heap.ptr);
        }
    }

    /// Convert C string to MojoString (makes a copy)
    pub fn stringFromC(cstr: [*:0]const u8) !memory.MojoString {
        const len = std.mem.len(cstr);
        return memory.MojoString.fromSlice(cstr[0..len]);
    }

    /// Convert Mojo int to C int
    pub fn intToC(value: i64) c_long {
        return @intCast(value);
    }

    /// Convert C int to Mojo int
    pub fn intFromC(value: c_long) i64 {
        return @intCast(value);
    }

    /// Convert Mojo float to C double
    pub fn floatToC(value: f64) f64 {
        return value;
    }

    /// Convert C double to Mojo float
    pub fn floatFromC(value: f64) f64 {
        return value;
    }

    /// Convert Mojo bool to C int (0 or 1)
    pub fn boolToC(value: bool) c_int {
        return if (value) 1 else 0;
    }

    /// Convert C int to Mojo bool
    pub fn boolFromC(value: c_int) bool {
        return value != 0;
    }
};

// ============================================================================
// Dynamic Library Loading
// ============================================================================

pub const DynLib = struct {
    handle: std.DynLib,
    path: []const u8,

    const Self = @This();

    /// Load a dynamic library
    pub fn open(path: []const u8) !Self {
        const lib = try std.DynLib.open(path);
        return .{
            .handle = lib,
            .path = path,
        };
    }

    /// Load with platform-specific naming
    pub fn openLibrary(name: []const u8) !Self {
        // Try different platform-specific names
        const extensions = switch (builtin.os.tag) {
            .macos => &[_][]const u8{ ".dylib", "" },
            .linux => &[_][]const u8{ ".so", "" },
            .windows => &[_][]const u8{ ".dll", "" },
            else => &[_][]const u8{""},
        };

        const prefixes = switch (builtin.os.tag) {
            .windows => &[_][]const u8{""},
            else => &[_][]const u8{ "lib", "" },
        };

        var buf: [512]u8 = undefined;

        for (prefixes) |prefix| {
            for (extensions) |ext| {
                const full_name = std.fmt.bufPrint(&buf, "{s}{s}{s}", .{ prefix, name, ext }) catch continue;
                if (std.DynLib.open(full_name)) |lib| {
                    return .{
                        .handle = lib,
                        .path = name,
                    };
                } else |_| continue;
            }
        }

        return error.LibraryNotFound;
    }

    /// Get a function pointer
    pub fn getSymbol(self: *Self, comptime T: type, name: [:0]const u8) ?T {
        return self.handle.lookup(T, name);
    }

    /// Get a variable pointer
    pub fn getVariable(self: *Self, comptime T: type, name: [:0]const u8) ?*T {
        const ptr = self.handle.lookup(*T, name);
        return ptr;
    }

    /// Close the library
    pub fn close(self: *Self) void {
        self.handle.close();
    }
};

// ============================================================================
// C Function Wrapper
// ============================================================================

/// Wrapper for calling C functions with Mojo types
pub fn CFunction(comptime ReturnType: type, comptime ArgTypes: []const type) type {
    return struct {
        const Self = @This();

        ptr: *const fn () callconv(.c) ReturnType,

        pub fn init(ptr: anytype) Self {
            return .{ .ptr = @ptrCast(ptr) };
        }

        pub fn call(self: Self, args: anytype) ReturnType {
            const ArgsType = @TypeOf(args);
            const fields = @typeInfo(ArgsType).Struct.fields;

            if (fields.len != ArgTypes.len) {
                @compileError("Argument count mismatch");
            }

            // This is a simplified implementation
            // A full implementation would handle arbitrary argument counts
            return @call(.auto, self.ptr, args);
        }
    };
}

// ============================================================================
// Callback Registry
// ============================================================================

/// Registry for Mojo callbacks that can be called from C
pub const CallbackRegistry = struct {
    callbacks: std.StringHashMap(CallbackEntry),
    allocator: std.mem.Allocator,

    const CallbackEntry = struct {
        ptr: *const anyopaque,
        signature: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator) CallbackRegistry {
        return .{
            .callbacks = std.StringHashMap(CallbackEntry).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CallbackRegistry) void {
        self.callbacks.deinit();
    }

    /// Register a callback
    pub fn register(self: *CallbackRegistry, name: []const u8, ptr: *const anyopaque, signature: []const u8) !void {
        try self.callbacks.put(name, .{
            .ptr = ptr,
            .signature = signature,
        });
    }

    /// Get a callback
    pub fn get(self: *const CallbackRegistry, name: []const u8) ?CallbackEntry {
        return self.callbacks.get(name);
    }

    /// Unregister a callback
    pub fn unregister(self: *CallbackRegistry, name: []const u8) bool {
        return self.callbacks.remove(name);
    }
};

var global_callback_registry: ?CallbackRegistry = null;

pub fn getCallbackRegistry() *CallbackRegistry {
    if (global_callback_registry == null) {
        global_callback_registry = CallbackRegistry.init(std.heap.page_allocator);
    }
    return &global_callback_registry.?;
}

// ============================================================================
// Struct Marshalling
// ============================================================================

/// Helper for marshalling structs between Mojo and C
pub fn StructMarshaller(comptime MojoType: type, comptime CType: type) type {
    return struct {
        const Self = @This();

        /// Convert Mojo struct to C struct
        pub fn toC(mojo: MojoType) CType {
            var c_val: CType = undefined;

            inline for (@typeInfo(MojoType).Struct.fields) |field| {
                if (@hasField(CType, field.name)) {
                    @field(c_val, field.name) = @field(mojo, field.name);
                }
            }

            return c_val;
        }

        /// Convert C struct to Mojo struct
        pub fn fromC(c_val: CType) MojoType {
            var mojo: MojoType = undefined;

            inline for (@typeInfo(CType).Struct.fields) |field| {
                if (@hasField(MojoType, field.name)) {
                    @field(mojo, field.name) = @field(c_val, field.name);
                }
            }

            return mojo;
        }
    };
}

// ============================================================================
// Error Handling
// ============================================================================

/// Last FFI error
var last_ffi_error: ?[]const u8 = null;

/// Set FFI error
pub fn setError(msg: []const u8) void {
    last_ffi_error = msg;
}

/// Get FFI error
pub fn getError() ?[]const u8 {
    return last_ffi_error;
}

/// Clear FFI error
pub fn clearError() void {
    last_ffi_error = null;
}

// ============================================================================
// C ABI Exports
// ============================================================================

/// Register a callback from C
export fn mojo_ffi_register_callback(
    name: [*:0]const u8,
    ptr: *const anyopaque,
    signature: [*:0]const u8,
) callconv(.c) i32 {
    const registry = getCallbackRegistry();
    const name_slice = std.mem.span(name);
    const sig_slice = std.mem.span(signature);

    registry.register(name_slice, ptr, sig_slice) catch return -1;
    return 0;
}

/// Get a callback by name
export fn mojo_ffi_get_callback(name: [*:0]const u8) callconv(.c) ?*const anyopaque {
    const registry = getCallbackRegistry();
    const name_slice = std.mem.span(name);

    if (registry.get(name_slice)) |entry| {
        return entry.ptr;
    }
    return null;
}

/// Load a dynamic library
export fn mojo_ffi_load_library(path: [*:0]const u8) callconv(.c) ?*DynLib {
    const path_slice = std.mem.span(path);
    const allocator = core.getAllocator();

    const lib = allocator.backing.create(DynLib) catch return null;
    lib.* = DynLib.open(path_slice) catch {
        allocator.backing.destroy(lib);
        return null;
    };

    return lib;
}

/// Get symbol from library
export fn mojo_ffi_get_symbol(lib: *DynLib, name: [*:0]const u8) callconv(.c) ?*anyopaque {
    const name_span = std.mem.span(name);
    const symbol = lib.handle.lookup(*anyopaque, name_span);
    return symbol;
}

/// Close a dynamic library
export fn mojo_ffi_close_library(lib: *DynLib) callconv(.c) void {
    lib.close();
    const allocator = core.getAllocator();
    allocator.backing.destroy(lib);
}

/// Get last FFI error
export fn mojo_ffi_get_error() callconv(.c) ?[*:0]const u8 {
    if (last_ffi_error) |err| {
        return @ptrCast(err.ptr);
    }
    return null;
}

/// Clear last FFI error
export fn mojo_ffi_clear_error() callconv(.c) void {
    clearError();
}

// ============================================================================
// Common C Library Functions (libc wrappers)
// ============================================================================

/// Printf wrapper
export fn mojo_print(format: [*:0]const u8) callconv(.c) void {
    const slice = std.mem.span(format);
    std.debug.print("{s}", .{slice});
}

/// Println wrapper
export fn mojo_println(format: [*:0]const u8) callconv(.c) void {
    const slice = std.mem.span(format);
    std.debug.print("{s}\n", .{slice});
}

/// Print integer
export fn mojo_print_int(value: i64) callconv(.c) void {
    std.debug.print("{d}", .{value});
}

/// Print float
export fn mojo_print_float(value: f64) callconv(.c) void {
    std.debug.print("{d}", .{value});
}

/// Read line from stdin
export fn mojo_read_line(buffer: [*]u8, max_len: usize) callconv(.c) isize {
    const stdin = std.posix.STDIN_FILENO;
    const result = std.posix.read(stdin, buffer[0..max_len]) catch return -1;
    return @intCast(result);
}

// ============================================================================
// Tests
// ============================================================================

test "type converter string" {
    try core.initDefault();
    defer core.deinit();

    var str = try memory.MojoString.fromSlice("Hello FFI");
    defer str.deinit();

    const cstr = TypeConverter.stringToC(&str);
    try std.testing.expectEqualStrings("Hello FFI", std.mem.span(cstr));
}

test "type converter int" {
    const mojo_val: i64 = 42;
    const c_val = TypeConverter.intToC(mojo_val);
    const back = TypeConverter.intFromC(c_val);
    try std.testing.expect(back == mojo_val);
}

test "type converter bool" {
    try std.testing.expect(TypeConverter.boolToC(true) == 1);
    try std.testing.expect(TypeConverter.boolToC(false) == 0);
    try std.testing.expect(TypeConverter.boolFromC(1) == true);
    try std.testing.expect(TypeConverter.boolFromC(0) == false);
}

test "callback registry" {
    try core.initDefault();
    defer core.deinit();

    var registry = CallbackRegistry.init(std.heap.page_allocator);
    defer registry.deinit();

    const dummy_fn = struct {
        fn call() void {}
    }.call;

    try registry.register("test_callback", @ptrCast(&dummy_fn), "void()");

    const entry = registry.get("test_callback");
    try std.testing.expect(entry != null);
    try std.testing.expectEqualStrings("void()", entry.?.signature);
}
