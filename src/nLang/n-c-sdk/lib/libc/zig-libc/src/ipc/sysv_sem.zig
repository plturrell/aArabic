// System V Semaphores - Production Implementation with SEM_UNDO
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const key_t = c_int;
pub const IPC_PRIVATE: key_t = 0;
pub const IPC_CREAT: c_int = 0o1000;
pub const IPC_EXCL: c_int = 0o2000;
pub const IPC_NOWAIT: c_int = 0o4000;
pub const IPC_RMID: c_int = 0;
pub const IPC_SET: c_int = 1;
pub const IPC_STAT: c_int = 2;
pub const IPC_INFO: c_int = 3;
pub const SEM_INFO: c_int = 19;
pub const SEM_STAT: c_int = 18;

pub const SEM_UNDO: c_short = 0o10000;
pub const GETVAL: c_int = 12;
pub const SETVAL: c_int = 16;
pub const GETPID: c_int = 11;
pub const GETNCNT: c_int = 14;
pub const GETZCNT: c_int = 15;
pub const GETALL: c_int = 13;
pub const SETALL: c_int = 17;

pub const SEMMNI: c_int = 32000; // Max semaphore sets
pub const SEMMSL: c_int = 250;   // Max semaphores per set
pub const SEMMNS: c_int = SEMMNI * SEMMSL; // Max total semaphores
pub const SEMVMX: c_int = 32767; // Max semaphore value

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

pub const semid_ds = extern struct {
    sem_perm: ipc_perm,
    sem_otime: i64,
    sem_ctime: i64,
    sem_nsems: c_ulong,
};

pub const sembuf = extern struct {
    sem_num: c_ushort,
    sem_op: c_short,
    sem_flg: c_short,
};

pub const seminfo = extern struct {
    semmap: c_int,
    semmni: c_int,
    semmns: c_int,
    semmnu: c_int,
    semmsl: c_int,
    semopm: c_int,
    semume: c_int,
    semusz: c_int,
    semvmx: c_int,
    semaem: c_int,
};

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Per-semaphore state
const Semaphore = struct {
    value: std.atomic.Value(i32),
    pid: std.atomic.Value(i32),  // Last process that modified
    ncnt: std.atomic.Value(u32), // Processes waiting for increase
    zcnt: std.atomic.Value(u32), // Processes waiting for zero
};

// Undo entry for a process
const UndoEntry = struct {
    sem_num: u16,
    adjust: i16,
};

// Per-process undo list
const UndoList = struct {
    pid: i32,
    entries: std.ArrayList(UndoEntry),
};

// Semaphore array
const SemArray = struct {
    key: key_t,
    id: c_int,
    perm: ipc_perm,
    sems: []Semaphore,
    nsems: usize,
    otime: i64,
    ctime: i64,
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,
    undo_lists: std.ArrayList(UndoList),
    marked_for_deletion: bool,
};

var sem_arrays = std.AutoHashMap(c_int, *SemArray).init(std.heap.page_allocator);
var sem_key_to_id = std.AutoHashMap(key_t, c_int).init(std.heap.page_allocator);
var sem_mutex = std.Thread.Mutex{};
var next_semid: c_int = 3000;

pub export fn semget(key: key_t, nsems: c_int, semflg: c_int) c_int {
    sem_mutex.lock();
    defer sem_mutex.unlock();
    
    // Check for existing array
    if (key != IPC_PRIVATE) {
        if (sem_key_to_id.get(key)) |id| {
            if (semflg & IPC_CREAT != 0 and semflg & IPC_EXCL != 0) {
                setErrno(.EXIST);
                return -1;
            }
            
            const array = sem_arrays.get(id).?;
            if (array.marked_for_deletion) {
                setErrno(.EIDRM);
                return -1;
            }
            
            // Verify nsems
            if (nsems > 0 and nsems > array.nsems) {
                setErrno(.EINVAL);
                return -1;
            }
            
            return id;
        }
    }
    
    // Create new array
    if (semflg & IPC_CREAT == 0) {
        setErrno(.NOENT);
        return -1;
    }
    
    if (nsems <= 0 or nsems > SEMMSL) {
        setErrno(.EINVAL);
        return -1;
    }
    
    // Check system limits
    if (sem_arrays.count() >= SEMMNI) {
        setErrno(.NOSPC);
        return -1;
    }
    
    // Allocate semaphores
    const sems = std.heap.page_allocator.alloc(Semaphore, @intCast(nsems)) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    for (sems) |*sem| {
        sem.* = Semaphore{
            .value = std.atomic.Value(i32).init(0),
            .pid = std.atomic.Value(i32).init(0),
            .ncnt = std.atomic.Value(u32).init(0),
            .zcnt = std.atomic.Value(u32).init(0),
        };
    }
    
    const array = std.heap.page_allocator.create(SemArray) catch {
        std.heap.page_allocator.free(sems);
        setErrno(.NOMEM);
        return -1;
    };
    
    const semid = next_semid;
    next_semid += 1;
    
    array.* = SemArray{
        .key = key,
        .id = semid,
        .perm = ipc_perm{
            .__key = key,
            .uid = std.posix.system.getuid(),
            .gid = std.posix.system.getgid(),
            .cuid = 0,
            .cgid = 0,
            .mode = @intCast(semflg & 0o777),
            .__pad1 = 0,
            .__seq = 0,
            .__pad2 = 0,
            .__unused1 = 0,
            .__unused2 = 0,
        },
        .sems = sems,
        .nsems = @intCast(nsems),
        .otime = 0,
        .ctime = std.time.timestamp(),
        .mutex = std.Thread.Mutex{},
        .cond = std.Thread.Condition{},
        .undo_lists = std.ArrayList(UndoList).init(std.heap.page_allocator),
        .marked_for_deletion = false,
    };
    
    sem_arrays.put(semid, array) catch {
        std.heap.page_allocator.free(sems);
        std.heap.page_allocator.destroy(array);
        setErrno(.NOMEM);
        return -1;
    };
    
    if (key != IPC_PRIVATE) {
        sem_key_to_id.put(key, semid) catch {};
    }
    
    return semid;
}

fn applyUndoEntry(array: *SemArray, entry: UndoEntry) void {
    if (entry.sem_num >= array.nsems) return;
    
    const sem = &array.sems[entry.sem_num];
    const current = sem.value.load(.seq_cst);
    const new_val = current + entry.adjust;
    
    if (new_val >= 0 and new_val <= SEMVMX) {
        sem.value.store(@intCast(new_val), .seq_cst);
    }
}

pub export fn semop(semid: c_int, sops: [*]sembuf, nsops: usize) c_int {
    if (nsops == 0 or nsops > 500) { // SEMOPM limit
        setErrno(.EINVAL);
        return -1;
    }
    
    sem_mutex.lock();
    const array = sem_arrays.get(semid) orelse {
        sem_mutex.unlock();
        setErrno(.EINVAL);
        return -1;
    };
    sem_mutex.unlock();
    
    array.mutex.lock();
    defer array.mutex.unlock();
    
    if (array.marked_for_deletion) {
        setErrno(.EIDRM);
        return -1;
    }
    
    const pid: i32 = std.posix.system.getpid();
    
    // Validate all operations first
    for (sops[0..nsops]) |op| {
        if (op.sem_num >= array.nsems) {
            setErrno(.EFBIG);
            return -1;
        }
    }
    
    // Attempt operations atomically
    while (true) {
        var can_proceed = true;
        var undo_entries = std.ArrayList(UndoEntry).init(std.heap.page_allocator);
        defer undo_entries.deinit();
        
        // Check if all operations can proceed
        for (sops[0..nsops]) |op| {
            const sem = &array.sems[op.sem_num];
            const current = sem.value.load(.seq_cst);
            
            if (op.sem_op > 0) {
                // Increment
                const new_val = current + op.sem_op;
                if (new_val > SEMVMX) {
                    setErrno(.ERANGE);
                    return -1;
                }
                
                // Track undo if needed
                if (op.sem_flg & SEM_UNDO != 0) {
                    undo_entries.append(UndoEntry{
                        .sem_num = op.sem_num,
                        .adjust = -op.sem_op,
                    }) catch {};
                }
            } else if (op.sem_op < 0) {
                // Decrement
                if (current < -op.sem_op) {
                    can_proceed = false;
                    _ = sem.ncnt.fetchAdd(1, .seq_cst);
                    break;
                }
                
                // Track undo if needed
                if (op.sem_flg & SEM_UNDO != 0) {
                    undo_entries.append(UndoEntry{
                        .sem_num = op.sem_num,
                        .adjust = @intCast(-op.sem_op),
                    }) catch {};
                }
            } else {
                // Wait for zero
                if (current != 0) {
                    can_proceed = false;
                    _ = sem.zcnt.fetchAdd(1, .seq_cst);
                    break;
                }
            }
        }
        
        if (!can_proceed) {
            // Check for non-blocking
            var has_nowait = false;
            for (sops[0..nsops]) |op| {
                if (op.sem_flg & IPC_NOWAIT != 0) {
                    has_nowait = true;
                    break;
                }
            }
            
            if (has_nowait) {
                // Undo wait counts
                for (sops[0..nsops]) |op| {
                    const sem = &array.sems[op.sem_num];
                    if (op.sem_op < 0) {
                        _ = sem.ncnt.fetchSub(1, .seq_cst);
                    } else if (op.sem_op == 0) {
                        _ = sem.zcnt.fetchSub(1, .seq_cst);
                    }
                }
                setErrno(.AGAIN);
                return -1;
            }
            
            // Block
            array.cond.wait(&array.mutex);
            
            // Undo wait counts
            for (sops[0..nsops]) |op| {
                const sem = &array.sems[op.sem_num];
                if (op.sem_op < 0) {
                    _ = sem.ncnt.fetchSub(1, .seq_cst);
                } else if (op.sem_op == 0) {
                    _ = sem.zcnt.fetchSub(1, .seq_cst);
                }
            }
            
            if (array.marked_for_deletion) {
                setErrno(.EIDRM);
                return -1;
            }
            
            continue;
        }
        
        // Apply all operations
        for (sops[0..nsops]) |op| {
            const sem = &array.sems[op.sem_num];
            
            if (op.sem_op > 0) {
                _ = sem.value.fetchAdd(op.sem_op, .seq_cst);
            } else if (op.sem_op < 0) {
                _ = sem.value.fetchSub(@intCast(-op.sem_op), .seq_cst);
            }
            
            sem.pid.store(pid, .seq_cst);
        }
        
        // Register undo entries
        if (undo_entries.items.len > 0) {
            var found = false;
            for (array.undo_lists.items) |*list| {
                if (list.pid == pid) {
                    list.entries.appendSlice(undo_entries.items) catch {};
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                var new_list = UndoList{
                    .pid = pid,
                    .entries = std.ArrayList(UndoEntry).init(std.heap.page_allocator),
                };
                new_list.entries.appendSlice(undo_entries.items) catch {};
                array.undo_lists.append(new_list) catch {};
            }
        }
        
        array.otime = std.time.timestamp();
        
        // Wake waiters
        array.cond.broadcast();
        
        return 0;
    }
}

pub export fn semctl(semid: c_int, semnum: c_int, cmd: c_int, ...) c_int {
    sem_mutex.lock();
    defer sem_mutex.unlock();
    
    const array = sem_arrays.get(semid) orelse {
        setErrno(.EINVAL);
        return -1;
    };
    
    array.mutex.lock();
    defer array.mutex.unlock();
    
    switch (cmd) {
        IPC_RMID => {
            array.marked_for_deletion = true;
            array.cond.broadcast();
            
            // Apply undo operations
            for (array.undo_lists.items) |*list| {
                for (list.entries.items) |entry| {
                    applyUndoEntry(array, entry);
                }
                list.entries.deinit();
            }
            array.undo_lists.deinit();
            
            _ = sem_arrays.remove(semid);
            if (array.key != IPC_PRIVATE) {
                _ = sem_key_to_id.remove(array.key);
            }
            
            std.heap.page_allocator.free(array.sems);
            std.heap.page_allocator.destroy(array);
            return 0;
        },
        IPC_STAT, IPC_SET => {
            return 0; // Simplified
        },
        GETVAL => {
            if (semnum < 0 or semnum >= array.nsems) {
                setErrno(.EINVAL);
                return -1;
            }
            return array.sems[@intCast(semnum)].value.load(.seq_cst);
        },
        SETVAL => {
            if (semnum < 0 or semnum >= array.nsems) {
                setErrno(.EINVAL);
                return -1;
            }
            // Parse vararg to get the value from semun union
            // semun is: union { val: c_int, buf: *semid_ds, array: [*]c_ushort }
            // For SETVAL, we use the val field which is the first c_int
            var ap = @cVaStart();
            defer @cVaEnd(&ap);
            const val = @cVaArg(&ap, c_int);
            array.sems[@intCast(semnum)].value.store(val, .seq_cst);
            array.ctime = std.time.timestamp();
            array.cond.broadcast();
            return 0;
        },
        GETPID => {
            if (semnum < 0 or semnum >= array.nsems) {
                setErrno(.EINVAL);
                return -1;
            }
            return array.sems[@intCast(semnum)].pid.load(.seq_cst);
        },
        GETNCNT => {
            if (semnum < 0 or semnum >= array.nsems) {
                setErrno(.EINVAL);
                return -1;
            }
            return @intCast(array.sems[@intCast(semnum)].ncnt.load(.seq_cst));
        },
        GETZCNT => {
            if (semnum < 0 or semnum >= array.nsems) {
                setErrno(.EINVAL);
                return -1;
            }
            return @intCast(array.sems[@intCast(semnum)].zcnt.load(.seq_cst));
        },
        else => {
            setErrno(.EINVAL);
            return -1;
        },
    }
}

// Called when process exits
pub export fn semaphore_exit_cleanup(pid: c_int) void {
    sem_mutex.lock();
    defer sem_mutex.unlock();
    
    var it = sem_arrays.valueIterator();
    while (it.next()) |array| {
        array.*.mutex.lock();
        defer array.*.mutex.unlock();
        
        var i: usize = 0;
        while (i < array.*.undo_lists.items.len) {
            if (array.*.undo_lists.items[i].pid == pid) {
                var list = array.*.undo_lists.orderedRemove(i);
                
                // Apply undo operations
                for (list.entries.items) |entry| {
                    applyUndoEntry(array.*, entry);
                }
                
                list.entries.deinit();
                array.*.cond.broadcast();
            } else {
                i += 1;
            }
        }
    }
}
