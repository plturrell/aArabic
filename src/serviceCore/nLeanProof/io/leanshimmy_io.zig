const std = @import("std");

const allocator = std.heap.c_allocator;
const max_read_size: usize = 64 * 1024 * 1024;

export fn leanshimmy_read_file(path: [*:0]const u8, out_len: *usize) callconv(.c) ?[*]u8 {
    const path_slice = std.mem.span(path);
    var file = std.fs.cwd().openFile(path_slice, .{}) catch {
        out_len.* = 0;
        return null;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, max_read_size) catch {
        out_len.* = 0;
        return null;
    };

    out_len.* = data.len;
    return data.ptr;
}

export fn leanshimmy_free(ptr: ?[*]u8, len: usize) callconv(.c) void {
    if (ptr == null) return;
    allocator.free(ptr.?[0..len]);
}
