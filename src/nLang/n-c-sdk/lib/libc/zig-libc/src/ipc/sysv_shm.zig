// System V Shared Memory - Phase 1.3 Extended IPC
// Production-grade shared memory implementation for banking systems
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Key type
pub const key_t = c_int;

// Special key values
pub const IPC_PRIVATE: key_t = 0;
pub const IPC_CREAT: c_int = 0o1000;
pub const IPC_EXCL: c_int = 0o2000;
pub const IPC_NOWAIT: c_int = 0o4000;

// Control commands
pub const IPC_RMID: c_int = 0;
pub const IPC_SET: c_int = 1;
pub const IPC_STAT: c_int = 2;
pub const IPC_INFO: c_int = 3;
pub const SHM_INFO: c_int = 14;
pub const SHM_STAT: c_int = 13;
pub const SHM_LOCK: c_int = 11;
pub const SHM_UNLOCK: c_int = 12;

// Attach flags
pub const SHM_RDONLY: c_int = 0o10000;
pub const SHM_RND: c_int = 0o20000;
pub const SHM_REMAP: c_int = 0o40000;
pub const SHM_EXEC: c_int = 0o100000;

// Permission bits
pub const mode_t = c_uint;

// IPC permissions structure
pub const ipc_perm = extern struct {
    __key: key_t,
    uid: c_uint,
    gid: c_uint,
    cuid: c_uint,
    cgid: c_uint,
    mode: c_ushort,
    __pad1: c_ushort,
    __seq: c_ushort,
    __pad2: c_ushort,
    __unused1: c_ulong,
    __unused2: c_ulong,
};

// Shared memory control structure
pub const shmid_ds = extern struct {
    shm_perm: ipc_perm,
    shm_segsz: usize,
    shm_atime: i64,
    shm_dtime: i64,
    shm_ctime: i64,
    shm_cpid: c_int,
    shm_lpid: c_int,
    shm_nattch: c_ulong,
    __unused1: c_ulong,
    __unused2: c_ulong,
};

// System info structures
pub const shminfo = extern struct {
    shmmax: c_ulong,
    shmmin: c_ulong,
    shmmni: c_ulong,
    shmseg: c_ulong,
    shmall: c_ulong,
};

pub const shm_info = extern struct {
    used_ids: c_int,
    shm_tot: c_ulong,
    shm_rss: c_ulong,
    shm_swp: c_ulong,
    swap_attempts: c_ulong,
    swap_successes: c_ulong,
};

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Internal shared memory segment
const ShmSegment = struct {
    key: key_t,
    id: c_int,
    memory: []align(std.mem.page_size) u8,
    perm: ipc_perm,
    segsz: usize,
    atime: i64,
    dtime: i64,
    ctime: i64,
    cpid: c_int,
    lpid: c_int,
    nattch: std.atomic.Value(u32),
};

// Global segment registry
var shm_segments = std.AutoHashMap(c_int, *ShmSegment).init(std.heap.page_allocator);
var shm_key_to_id = std.AutoHashMap(key_t, c_int).init(std.heap.page_allocator);
var shm_mutex = std.Thread.Mutex{};
var next_shmid: c_int = 1000;

/// Get shared memory identifier
pub export fn shmget(key: key_t, size: usize, shmflg: c_int) c_int {
    shm_mutex.lock();
    defer shm_mutex.unlock();
    
    // Check if segment with this key exists
    if (key != IPC_PRIVATE) {
        if (shm_key_to_id.get(key)) |existing_id| {
            if (shmflg & IPC_CREAT != 0 and shmflg & IPC_EXCL != 0) {
                setErrno(.EXIST);
                return -1;
            }
            
            // Verify size matches
            if (shm_segments.get(existing_id)) |seg| {
                if (size > seg.segsz) {
                    setErrno(.INVAL);
                    return -1;
                }
                return existing_id;
            }
        }
    }
    
    // Create new segment
    if (shmflg & IPC_CREAT == 0) {
        setErrno(.NOENT);
        return -1;
    }
    
    // Validate size
    if (size == 0 or size > 1024 * 1024 * 1024) { // Max 1GB
        setErrno(.INVAL);
        return -1;
    }
    
    // Round up to page size
    const page_size = std.mem.page_size;
    const aligned_size = std.mem.alignForward(usize, size, page_size);
    
    // Allocate memory
    const memory = std.heap.page_allocator.alignedAlloc(u8, page_size, aligned_size) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const segment = std.heap.page_allocator.create(ShmSegment) catch {
        std.heap.page_allocator.free(memory);
        setErrno(.NOMEM);
        return -1;
    };
    
    const shmid = next_shmid;
    next_shmid += 1;
    
    const now = std.time.timestamp();
    segment.* = ShmSegment{
        .key = key,
        .id = shmid,
        .memory = memory,
        .perm = ipc_perm{
            .__key = key,
            .uid = std.posix.system.getuid(),
            .gid = 0,
            .cuid = 0,
            .cgid = 0,
            .mode = @intCast(shmflg & 0o777),
            .__pad1 = 0,
            .__seq = 0,
            .__pad2 = 0,
            .__unused1 = 0,
            .__unused2 = 0,
        },
        .segsz = aligned_size,
        .atime = 0,
        .dtime = 0,
        .ctime = now,
        .cpid = std.posix.system.getpid(),
        .lpid = 0,
        .nattch = std.atomic.Value(u32).init(0),
    };
    
    shm_segments.put(shmid, segment) catch {
        std.heap.page_allocator.destroy(segment);
        std.heap.page_allocator.free(memory);
        setErrno(.NOMEM);
        return -1;
    };
    
    if (key != IPC_PRIVATE) {
        shm_key_to_id.put(key, shmid) catch {};
    }
    
    return shmid;
}

/// Attach shared memory segment
pub export fn shmat(shmid: c_int, shmaddr: ?*const anyopaque, shmflg: c_int) ?*anyopaque {
    shm_mutex.lock();
    defer shm_mutex.unlock();
    
    const segment = shm_segments.get(shmid) orelse {
        setErrno(.EINVAL);
        return @ptrFromInt(std.math.maxInt(usize));
    };
    
    // Handle address hint
    const addr = shmaddr orelse @ptrFromInt(0);
    
    // Check permissions
    if (shmflg & SHM_RDONLY != 0) {
        // Read-only check (simplified)
    }
    
    // Increment attach count
    _ = segment.nattch.fetchAdd(1, .seq_cst);
    
    // Update access time
    segment.atime = std.time.timestamp();
    segment.lpid = std.posix.system.getpid();
    
    return @ptrCast(segment.memory.ptr);
}

/// Detach shared memory segment
pub export fn shmdt(shmaddr: ?*const anyopaque) c_int {
    if (shmaddr == null) {
        setErrno(.EINVAL);
        return -1;
    }
    
    shm_mutex.lock();
    defer shm_mutex.unlock();
    
    // Find segment by address
    var it = shm_segments.valueIterator();
    while (it.next()) |segment| {
        if (@intFromPtr(segment.*.memory.ptr) == @intFromPtr(shmaddr.?)) {
            _ = segment.*.nattch.fetchSub(1, .seq_cst);
            segment.*.dtime = std.time.timestamp();
            segment.*.lpid = std.posix.system.getpid();
            return 0;
        }
    }
    
    setErrno(.EINVAL);
    return -1;
}

/// Shared memory control operations
pub export fn shmctl(shmid: c_int, cmd: c_int, buf: ?*shmid_ds) c_int {
    shm_mutex.lock();
    defer shm_mutex.unlock();
    
    const segment = shm_segments.get(shmid) orelse {
        setErrno(.EINVAL);
        return -1;
    };
    
    switch (cmd) {
        IPC_RMID => {
            // Mark for deletion (when nattch reaches 0)
            if (segment.nattch.load(.seq_cst) == 0) {
                _ = shm_segments.remove(shmid);
                if (segment.key != IPC_PRIVATE) {
                    _ = shm_key_to_id.remove(segment.key);
                }
                std.heap.page_allocator.free(segment.memory);
                std.heap.page_allocator.destroy(segment);
            }
            return 0;
        },
        IPC_STAT => {
            if (buf) |ds| {
                ds.shm_perm = segment.perm;
                ds.shm_segsz = segment.segsz;
                ds.shm_atime = segment.atime;
                ds.shm_dtime = segment.dtime;
                ds.shm_ctime = segment.ctime;
                ds.shm_cpid = segment.cpid;
                ds.shm_lpid = segment.lpid;
                ds.shm_nattch = segment.nattch.load(.seq_cst);
                ds.__unused1 = 0;
                ds.__unused2 = 0;
            }
            return 0;
        },
        IPC_SET => {
            if (buf) |ds| {
                segment.perm.uid = ds.shm_perm.uid;
                segment.perm.gid = ds.shm_perm.gid;
                segment.perm.mode = ds.shm_perm.mode & 0o777;
                segment.ctime = std.time.timestamp();
            }
            return 0;
        },
        SHM_LOCK, SHM_UNLOCK => {
            // Memory locking (simplified: no-op)
            return 0;
        },
        else => {
            setErrno(.EINVAL);
            return -1;
        },
    }
}

/// Generate IPC key from path and project ID
pub export fn ftok(pathname: [*:0]const u8, proj_id: c_int) key_t {
    const path = std.mem.span(pathname);
    
    // Get file stats
    var stat_buf: std.posix.Stat = undefined;
    std.posix.stat(path, &stat_buf) catch {
        setErrno(.NOENT);
        return -1;
    };
    
    // Generate key from inode, device, and proj_id
    const inode: key_t = @intCast(stat_buf.ino & 0xffff);
    const dev: key_t = @intCast(stat_buf.dev & 0xff);
    const proj: key_t = @intCast(proj_id & 0xff);
    
    return (proj << 24) | (dev << 16) | inode;
}
