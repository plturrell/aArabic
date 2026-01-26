const std = @import("std");
const builtin = @import("builtin");

// Tracy C++ integration has been removed from this codebase.
// All Tracy functions are now no-ops. The API is preserved for compatibility.
pub const enable = false;
pub const enable_allocation = false;
pub const enable_callstack = false;
pub const callstack_depth: u32 = 10;

// No-op context - Tracy C++ integration has been removed
pub const Ctx = struct {
    pub inline fn end(self: @This()) void {
        _ = self;
    }

    pub inline fn addText(self: @This(), text: []const u8) void {
        _ = self;
        _ = text;
    }

    pub inline fn setName(self: @This(), name: []const u8) void {
        _ = self;
        _ = name;
    }

    pub inline fn setColor(self: @This(), color: u32) void {
        _ = self;
        _ = color;
    }

    pub inline fn setValue(self: @This(), value: u64) void {
        _ = self;
        _ = value;
    }
};

// Tracy C++ integration removed - these are now no-ops
pub inline fn trace(comptime src: std.builtin.SourceLocation) Ctx {
    _ = src;
    return .{};
}

pub inline fn traceNamed(comptime src: std.builtin.SourceLocation, comptime name: [:0]const u8) Ctx {
    _ = src;
    _ = name;
    return .{};
}

pub fn tracyAllocator(allocator: std.mem.Allocator) TracyAllocator(null) {
    return TracyAllocator(null).init(allocator);
}

pub fn TracyAllocator(comptime name: ?[:0]const u8) type {
    return struct {
        parent_allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(parent_allocator: std.mem.Allocator) Self {
            return .{
                .parent_allocator = parent_allocator,
            };
        }

        pub fn allocator(self: *Self) std.mem.Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = allocFn,
                    .resize = resizeFn,
                    .remap = remapFn,
                    .free = freeFn,
                },
            };
        }

        fn allocFn(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const result = self.parent_allocator.rawAlloc(len, alignment, ret_addr);
            if (result) |memory| {
                if (len != 0) {
                    if (name) |n| {
                        allocNamed(memory, len, n);
                    } else {
                        alloc(memory, len);
                    }
                }
            } else {
                messageColor("allocation failed", 0xFF0000);
            }
            return result;
        }

        fn resizeFn(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
            const self: *Self = @ptrCast(@alignCast(ptr));
            if (self.parent_allocator.rawResize(memory, alignment, new_len, ret_addr)) {
                if (name) |n| {
                    freeNamed(memory.ptr, n);
                    allocNamed(memory.ptr, new_len, n);
                } else {
                    free(memory.ptr);
                    alloc(memory.ptr, new_len);
                }

                return true;
            }

            // during normal operation the compiler hits this case thousands of times due to this
            // emitting messages for it is both slow and causes clutter
            return false;
        }

        fn remapFn(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ptr));
            if (self.parent_allocator.rawRemap(memory, alignment, new_len, ret_addr)) |new_memory| {
                if (name) |n| {
                    freeNamed(memory.ptr, n);
                    allocNamed(new_memory, new_len, n);
                } else {
                    free(memory.ptr);
                    alloc(new_memory, new_len);
                }
                return new_memory;
            } else {
                messageColor("reallocation failed", 0xFF0000);
                return null;
            }
        }

        fn freeFn(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.parent_allocator.rawFree(memory, alignment, ret_addr);
            // this condition is to handle free being called on an empty slice that was never even allocated
            // example case: `std.process.getSelfExeSharedLibPaths` can return `&[_][:0]u8{}`
            if (memory.len != 0) {
                if (name) |n| {
                    freeNamed(memory.ptr, n);
                } else {
                    free(memory.ptr);
                }
            }
        }
    };
}

// Tracy C++ integration removed - all functions are now no-ops
pub inline fn message(comptime msg: [:0]const u8) void {
    _ = msg;
}

pub inline fn messageColor(comptime msg: [:0]const u8, color: u32) void {
    _ = msg;
    _ = color;
}

pub inline fn messageCopy(msg: []const u8) void {
    _ = msg;
}

pub inline fn messageColorCopy(msg: [:0]const u8, color: u32) void {
    _ = msg;
    _ = color;
}

pub inline fn frameMark() void {}

pub inline fn frameMarkNamed(comptime name: [:0]const u8) void {
    _ = name;
}

pub inline fn namedFrame(comptime name: [:0]const u8) Frame(name) {
    return .{};
}

pub fn Frame(comptime name: [:0]const u8) type {
    _ = name;
    return struct {
        pub fn end(_: @This()) void {}
    };
}

inline fn frameMarkStart(comptime name: [:0]const u8) void {
    _ = name;
}

inline fn frameMarkEnd(comptime name: [:0]const u8) void {
    _ = name;
}

// Tracy C++ extern functions have been removed
// These internal functions are now no-ops
inline fn alloc(ptr: [*]u8, len: usize) void {
    _ = ptr;
    _ = len;
}

inline fn allocNamed(ptr: [*]u8, len: usize, comptime name: [:0]const u8) void {
    _ = ptr;
    _ = len;
    _ = name;
}

inline fn free(ptr: [*]u8) void {
    _ = ptr;
}

inline fn freeNamed(ptr: [*]u8, comptime name: [:0]const u8) void {
    _ = ptr;
    _ = name;
}
