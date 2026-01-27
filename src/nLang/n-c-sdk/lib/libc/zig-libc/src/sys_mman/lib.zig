// sys/mman module - Phase 1.3 - Memory mapping with real implementations
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Protection flags
pub const PROT_NONE: c_int = 0x0;
pub const PROT_READ: c_int = 0x1;
pub const PROT_WRITE: c_int = 0x2;
pub const PROT_EXEC: c_int = 0x4;

// Mapping flags
pub const MAP_SHARED: c_int = 0x01;
pub const MAP_PRIVATE: c_int = 0x02;
pub const MAP_FIXED: c_int = 0x10;
pub const MAP_ANONYMOUS: c_int = 0x20;
pub const MAP_ANON: c_int = MAP_ANONYMOUS;
pub const MAP_GROWSDOWN: c_int = 0x100;
pub const MAP_DENYWRITE: c_int = 0x800;
pub const MAP_EXECUTABLE: c_int = 0x1000;
pub const MAP_LOCKED: c_int = 0x2000;
pub const MAP_NORESERVE: c_int = 0x4000;
pub const MAP_POPULATE: c_int = 0x8000;
pub const MAP_NONBLOCK: c_int = 0x10000;
pub const MAP_STACK: c_int = 0x20000;
pub const MAP_HUGETLB: c_int = 0x40000;

// Special values
pub const MAP_FAILED: ?*anyopaque = @ptrFromInt(std.math.maxInt(usize));

// Sync flags
pub const MS_ASYNC: c_int = 1;
pub const MS_SYNC: c_int = 4;
pub const MS_INVALIDATE: c_int = 2;

// Advice values
pub const MADV_NORMAL: c_int = 0;
pub const MADV_RANDOM: c_int = 1;
pub const MADV_SEQUENTIAL: c_int = 2;
pub const MADV_WILLNEED: c_int = 3;
pub const MADV_DONTNEED: c_int = 4;
pub const MADV_FREE: c_int = 8;
pub const MADV_REMOVE: c_int = 9;
pub const MADV_DONTFORK: c_int = 10;
pub const MADV_DOFORK: c_int = 11;
pub const MADV_MERGEABLE: c_int = 12;
pub const MADV_UNMERGEABLE: c_int = 13;
pub const MADV_HUGEPAGE: c_int = 14;
pub const MADV_NOHUGEPAGE: c_int = 15;

// Memory locking flags
pub const MCL_CURRENT: c_int = 1;
pub const MCL_FUTURE: c_int = 2;
pub const MCL_ONFAULT: c_int = 4;

// mremap flags
pub const MREMAP_MAYMOVE: c_int = 1;
pub const MREMAP_FIXED: c_int = 2;

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

// Memory mapping functions

pub export fn mmap(addr: ?*anyopaque, length: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque {
    const prot_flags: u32 = @intCast(prot);
    const map_flags: u32 = @intCast(flags);
    const off: i64 = offset;
    
    const rc = std.posix.system.mmap(addr, length, prot_flags, map_flags, fd, off);
    
    // Check for MAP_FAILED
    if (@intFromPtr(rc) == std.math.maxInt(usize)) {
        setErrno(std.posix.errno(@as(isize, -1)));
        return MAP_FAILED;
    }
    
    return rc;
}

pub export fn mmap64(addr: ?*anyopaque, length: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque {
    return mmap(addr, length, prot, flags, fd, offset);
}

pub export fn munmap(addr: ?*anyopaque, length: usize) c_int {
    const rc = std.posix.system.munmap(addr, length);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn mprotect(addr: ?*anyopaque, length: usize, prot: c_int) c_int {
    const prot_flags: u32 = @intCast(prot);
    const rc = std.posix.system.mprotect(addr, length, prot_flags);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn msync(addr: ?*anyopaque, length: usize, flags: c_int) c_int {
    const sync_flags: c_int = flags;
    const rc = std.posix.system.msync(addr, length, sync_flags);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn mlock(addr: ?*const anyopaque, length: usize) c_int {
    const rc = std.posix.system.mlock(addr, length);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn munlock(addr: ?*const anyopaque, length: usize) c_int {
    const rc = std.posix.system.munlock(addr, length);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn mlockall(flags: c_int) c_int {
    const rc = std.posix.system.mlockall(flags);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn munlockall() c_int {
    const rc = std.posix.system.munlockall();
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn madvise(addr: ?*anyopaque, length: usize, advice: c_int) c_int {
    if (@hasDecl(std.posix.system, "madvise")) {
        const rc = std.posix.system.madvise(addr, length, advice);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    // Not all systems support madvise
    return 0;
}

pub export fn mincore(addr: ?*anyopaque, length: usize, vec: [*]u8) c_int {
    if (@hasDecl(std.posix.system, "mincore")) {
        const rc = std.posix.system.mincore(addr, length, vec);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn mremap(old_address: ?*anyopaque, old_size: usize, new_size: usize, flags: c_int, ...) ?*anyopaque {
    if (@hasDecl(std.posix.system, "mremap")) {
        var new_address: ?*anyopaque = null;
        
        if ((flags & MREMAP_FIXED) != 0) {
            var args = @cVaStart();
            new_address = @cVaArg(&args, ?*anyopaque);
            @cVaEnd(&args);
        }
        
        const mremap_flags: u32 = @intCast(flags);
        const rc = std.posix.system.mremap(old_address, old_size, new_size, mremap_flags, @intFromPtr(new_address orelse @as(*anyopaque, @ptrFromInt(0))));
        
        if (@intFromPtr(rc) == std.math.maxInt(usize)) {
            setErrno(std.posix.errno(@as(isize, -1)));
            return MAP_FAILED;
        }
        
        return rc;
    }
    
    setErrno(.NOSYS);
    return MAP_FAILED;
}

pub export fn remap_file_pages(addr: ?*anyopaque, size: usize, prot: c_int, pgoff: usize, flags: c_int) c_int {
    if (@hasDecl(std.posix.system, "remap_file_pages")) {
        const rc = std.posix.system.remap_file_pages(addr, size, prot, pgoff, flags);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

// POSIX shared memory functions

pub export fn shm_open(name: [*:0]const u8, oflag: c_int, mode: c_uint) c_int {
    if (@hasDecl(std.posix.system, "shm_open")) {
        const rc = std.posix.system.shm_open(name, oflag, mode);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn shm_unlink(name: [*:0]const u8) c_int {
    if (@hasDecl(std.posix.system, "shm_unlink")) {
        const rc = std.posix.system.shm_unlink(name);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

// Memory info functions

pub export fn memfd_create(name: [*:0]const u8, flags: c_uint) c_int {
    if (@hasDecl(std.posix.system, "memfd_create")) {
        const rc = std.posix.system.memfd_create(name, flags);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn mlock2(addr: ?*const anyopaque, length: usize, flags: c_int) c_int {
    if (@hasDecl(std.posix.system, "mlock2")) {
        const rc = std.posix.system.mlock2(addr, length, flags);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    // Fallback to regular mlock
    return mlock(addr, length);
}

pub export fn pkey_alloc(flags: c_uint, access_rights: c_uint) c_int {
    if (@hasDecl(std.posix.system, "pkey_alloc")) {
        const rc = std.posix.system.pkey_alloc(flags, access_rights);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn pkey_free(pkey: c_int) c_int {
    if (@hasDecl(std.posix.system, "pkey_free")) {
        const rc = std.posix.system.pkey_free(pkey);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn pkey_mprotect(addr: ?*anyopaque, length: usize, prot: c_int, pkey: c_int) c_int {
    if (@hasDecl(std.posix.system, "pkey_mprotect")) {
        const prot_flags: u32 = @intCast(prot);
        const rc = std.posix.system.pkey_mprotect(addr, length, prot_flags, pkey);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    // Fallback to regular mprotect
    return mprotect(addr, length, prot);
}
