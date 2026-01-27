// dirent module - Phase 1.3 - Directory operations with real implementations
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Directory entry types
pub const DT_UNKNOWN: u8 = 0;
pub const DT_FIFO: u8 = 1;
pub const DT_CHR: u8 = 2;
pub const DT_DIR: u8 = 4;
pub const DT_BLK: u8 = 6;
pub const DT_REG: u8 = 8;
pub const DT_LNK: u8 = 10;
pub const DT_SOCK: u8 = 12;
pub const DT_WHT: u8 = 14;

// Directory entry structure
pub const dirent = extern struct {
    d_ino: c_ulong,
    d_off: c_long,
    d_reclen: c_ushort,
    d_type: u8,
    d_name: [256]u8,
};

// Internal DIR structure
const DirStream = struct {
    fd: c_int,
    allocator: std.mem.Allocator,
    buf: []u8,
    buf_pos: usize,
    buf_end: usize,
    tell_pos: c_long,
    entry: dirent,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// DIR as opaque pointer to DirStream
pub const DIR = opaque {};

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

inline fn failIfErrno(rc: anytype) bool {
    const err = std.posix.errno(rc);
    if (err != .SUCCESS) {
        setErrno(err);
        return true;
    }
    return false;
}

fn dirFromPtr(ptr: *DIR) *DirStream {
    return @ptrCast(@alignCast(ptr));
}

fn ptrFromDir(dir: *DirStream) *DIR {
    return @ptrCast(@alignCast(dir));
}

// Directory operations
pub export fn opendir(name: [*:0]const u8) ?*DIR {
    const fd = std.posix.system.open(name, std.posix.O.RDONLY | std.posix.O.DIRECTORY, 0);
    if (failIfErrno(fd)) return null;
    
    return fdopendir(fd);
}

pub export fn fdopendir(fd: c_int) ?*DIR {
    const dir = allocator.create(DirStream) catch {
        setErrno(.NOMEM);
        return null;
    };
    
    const buf = allocator.alloc(u8, 4096) catch {
        allocator.destroy(dir);
        setErrno(.NOMEM);
        return null;
    };
    
    dir.* = DirStream{
        .fd = fd,
        .allocator = allocator,
        .buf = buf,
        .buf_pos = 0,
        .buf_end = 0,
        .tell_pos = 0,
        .entry = std.mem.zeroes(dirent),
    };
    
    return ptrFromDir(dir);
}

pub export fn closedir(dirp: ?*DIR) c_int {
    const dir = dirFromPtr(dirp orelse return -1);
    
    _ = std.posix.system.close(dir.fd);
    dir.allocator.free(dir.buf);
    dir.allocator.destroy(dir);
    
    return 0;
}

pub export fn readdir(dirp: ?*DIR) ?*dirent {
    const dir = dirFromPtr(dirp orelse {
        setErrno(.BADF);
        return null;
    });
    
    // If buffer exhausted, read more entries
    if (dir.buf_pos >= dir.buf_end) {
        const rc = std.posix.system.getdents64(dir.fd, dir.buf.ptr, dir.buf.len);
        if (failIfErrno(rc)) return null;
        
        if (rc == 0) return null; // End of directory
        
        dir.buf_end = @intCast(rc);
        dir.buf_pos = 0;
    }
    
    // Parse next entry from buffer
    const linux_dirent = @as(*align(1) extern struct {
        d_ino: u64,
        d_off: i64,
        d_reclen: u16,
        d_type: u8,
        d_name: [256]u8,
    }, @ptrCast(&dir.buf[dir.buf_pos]));
    
    // Copy to our dirent structure
    dir.entry.d_ino = linux_dirent.d_ino;
    dir.entry.d_off = linux_dirent.d_off;
    dir.entry.d_reclen = linux_dirent.d_reclen;
    dir.entry.d_type = linux_dirent.d_type;
    
    // Copy name (find null terminator)
    var i: usize = 0;
    while (i < 256 and linux_dirent.d_name[i] != 0) : (i += 1) {
        dir.entry.d_name[i] = linux_dirent.d_name[i];
    }
    if (i < 256) dir.entry.d_name[i] = 0;
    
    dir.buf_pos += linux_dirent.d_reclen;
    dir.tell_pos += 1;
    
    return &dir.entry;
}

pub export fn readdir_r(dirp: ?*DIR, entry: *dirent, result: **dirent) c_int {
    const ent = readdir(dirp);
    if (ent) |e| {
        entry.* = e.*;
        result.* = entry;
        return 0;
    }
    
    result.* = @ptrFromInt(0);
    
    // Check if error or end of directory
    const err_val = errno_mod.__errno_location().*;
    if (err_val != 0) return err_val;
    return 0; // End of directory
}

pub export fn rewinddir(dirp: ?*DIR) void {
    const dir = dirFromPtr(dirp orelse return);

    _ = std.posix.system.lseek(dir.fd, 0, std.posix.SEEK.SET);
    dir.buf_pos = 0;
    dir.buf_end = 0;
    dir.tell_pos = 0;
}

pub export fn seekdir(dirp: ?*DIR, loc: c_long) void {
    // Simplified: just rewind and skip entries
    rewinddir(dirp);
    
    var i: c_long = 0;
    while (i < loc) : (i += 1) {
        if (readdir(dirp) == null) break;
    }
}

pub export fn telldir(dirp: ?*DIR) c_long {
    const dir = dirFromPtr(dirp orelse return -1);
    return dir.tell_pos;
}

pub export fn dirfd(dirp: ?*DIR) c_int {
    const dir = dirFromPtr(dirp orelse {
        setErrno(.BADF);
        return -1;
    });
    return dir.fd;
}

pub export fn scandir(
    dirpath: [*:0]const u8,
    namelist: *?[*]*dirent,
    filter: ?*const fn (*const dirent) callconv(.C) c_int,
    compar: ?*const fn (*const *const dirent, *const *const dirent) callconv(.C) c_int,
) c_int {
    const dir = opendir(dirpath) orelse return -1;
    defer _ = closedir(dir);
    
    var entries = std.ArrayList(*dirent).init(allocator);
    defer {
        for (entries.items) |e| {
            allocator.destroy(e);
        }
        entries.deinit();
    }
    
    // Read all entries
    while (readdir(dir)) |entry| {
        // Skip "." and ".."
        if (entry.d_name[0] == '.' and 
            (entry.d_name[1] == 0 or 
             (entry.d_name[1] == '.' and entry.d_name[2] == 0))) {
            continue;
        }
        
        // Apply filter if provided
        if (filter) |f| {
            if (f(entry) == 0) continue;
        }
        
        // Allocate and copy entry
        const new_entry = allocator.create(dirent) catch return -1;
        new_entry.* = entry.*;
        entries.append(new_entry) catch {
            allocator.destroy(new_entry);
            return -1;
        };
    }
    
    // Sort if comparator provided
    if (compar) |cmp| {
        std.sort.pdq(*dirent, entries.items, cmp, struct {
            fn lessThan(ctx: @TypeOf(cmp), a: *dirent, b: *dirent) bool {
                return ctx(&a, &b) < 0;
            }
        }.lessThan);
    }
    
    // Allocate result array
    const count = entries.items.len;
    if (count == 0) {
        namelist.* = null;
        return 0;
    }
    
    const result = allocator.alloc(*dirent, count) catch return -1;
    @memcpy(result, entries.items);
    
    namelist.* = result.ptr;
    
    // Clear entries list to prevent cleanup
    entries.clearRetainingCapacity();
    
    return @intCast(count);
}

pub export fn alphasort(a: *const *const dirent, b: *const *const dirent) c_int {
    const a_name = @as([*:0]const u8, @ptrCast(&a.*.d_name));
    const b_name = @as([*:0]const u8, @ptrCast(&b.*.d_name));
    
    return std.mem.orderZ(u8, a_name, b_name).compare(.eq);
}

pub export fn versionsort(a: *const *const dirent, b: *const *const dirent) c_int {
    // Simplified: just use alphasort
    return alphasort(a, b);
}

// Additional directory utilities

pub export fn scandirat(
    dirfd_param: c_int,
    dirpath: [*:0]const u8,
    namelist: *?[*]*dirent,
    filter: ?*const fn (*const dirent) callconv(.C) c_int,
    compar: ?*const fn (*const *const dirent, *const *const dirent) callconv(.C) c_int,
) c_int {
    _ = dirfd_param;
    // Simplified: just use scandir
    return scandir(dirpath, namelist, filter, compar);
}

pub export fn getdirentries(
    fd: c_int,
    buf: [*]u8,
    nbytes: usize,
    basep: *c_long,
) isize {
    const rc = std.posix.system.getdents64(fd, buf, nbytes);
    if (failIfErrno(rc)) return -1;
    
    if (rc > 0) {
        const new_pos = std.posix.system.lseek(fd, 0, std.posix.SEEK.CUR);
        if (!failIfErrno(new_pos)) {
            basep.* = new_pos;
        }
    }
    
    return rc;
}
