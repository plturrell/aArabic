// POSIX Shared Memory - Production Implementation
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Flags for shm_open
pub const O_RDONLY: c_int = 0x0000;
pub const O_WRONLY: c_int = 0x0001;
pub const O_RDWR: c_int = 0x0002;
pub const O_CREAT: c_int = 0x0040;
pub const O_EXCL: c_int = 0x0080;
pub const O_TRUNC: c_int = 0x0200;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

const ShmObject = struct {
    name: []const u8,
    fd: c_int,
    size: usize,
    refcount: std.atomic.Value(u32),
};

var shm_objects = std.StringHashMap(*ShmObject).init(std.heap.page_allocator);
var shm_mutex = std.Thread.Mutex{};

/// Open POSIX shared memory object
pub export fn shm_open(name: [*:0]const u8, oflag: c_int, mode: c_uint) c_int {
    const name_slice = std.mem.span(name);
    
    // Validate name (must start with /)
    if (name_slice.len == 0 or name_slice[0] != '/') {
        setErrno(.INVAL);
        return -1;
    }
    
    shm_mutex.lock();
    defer shm_mutex.unlock();
    
    // Check for existing object
    if (shm_objects.get(name_slice)) |obj| {
        if (oflag & O_CREAT != 0 and oflag & O_EXCL != 0) {
            setErrno(.EXIST);
            return -1;
        }
        
        _ = obj.refcount.fetchAdd(1, .seq_cst);
        return obj.fd;
    }
    
    // Create new object
    if (oflag & O_CREAT == 0) {
        setErrno(.NOENT);
        return -1;
    }
    
    // On Linux, POSIX shm is in /dev/shm
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/dev/shm{s}", .{name_slice}) catch {
        setErrno(.NAMETOOLONG);
        return -1;
    };
    
    // Create/open file
    const fd = std.posix.open(path, @intCast(oflag), mode) catch |err| {
        setErrno(switch (err) {
            error.FileNotFound => .NOENT,
            error.AccessDenied => .ACCES,
            error.PathAlreadyExists => .EXIST,
            else => .INVAL,
        });
        return -1;
    };
    
    const obj = std.heap.page_allocator.create(ShmObject) catch {
        std.posix.close(@intCast(fd));
        setErrno(.NOMEM);
        return -1;
    };
    
    obj.* = ShmObject{
        .name = std.heap.page_allocator.dupe(u8, name_slice) catch {
            std.posix.close(@intCast(fd));
            std.heap.page_allocator.destroy(obj);
            setErrno(.NOMEM);
            return -1;
        },
        .fd = @intCast(fd),
        .size = 0,
        .refcount = std.atomic.Value(u32).init(1),
    };
    
    shm_objects.put(obj.name, obj) catch {
        std.heap.page_allocator.free(obj.name);
        std.posix.close(@intCast(fd));
        std.heap.page_allocator.destroy(obj);
        setErrno(.NOMEM);
        return -1;
    };
    
    return obj.fd;
}

/// Unlink POSIX shared memory object
pub export fn shm_unlink(name: [*:0]const u8) c_int {
    const name_slice = std.mem.span(name);
    
    if (name_slice.len == 0 or name_slice[0] != '/') {
        setErrno(.INVAL);
        return -1;
    }
    
    shm_mutex.lock();
    defer shm_mutex.unlock();
    
    // Remove from registry
    if (shm_objects.fetchRemove(name_slice)) |entry| {
        const obj = entry.value;
        
        // Close if no references
        if (obj.refcount.load(.seq_cst) == 0) {
            std.posix.close(@intCast(obj.fd));
            std.heap.page_allocator.free(obj.name);
            std.heap.page_allocator.destroy(obj);
        }
    }
    
    // Unlink file
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/dev/shm{s}", .{name_slice}) catch {
        setErrno(.NAMETOOLONG);
        return -1;
    };
    
    std.posix.unlink(path) catch |err| {
        setErrno(switch (err) {
            error.FileNotFound => .NOENT,
            error.AccessDenied => .ACCES,
            else => .INVAL,
        });
        return -1;
    };
    
    return 0;
}
