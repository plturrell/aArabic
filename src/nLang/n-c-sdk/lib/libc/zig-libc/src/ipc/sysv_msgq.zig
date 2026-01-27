// System V Message Queues - Production Implementation
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

pub const MSG_NOERROR: c_int = 0o10000;

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

pub const msqid_ds = extern struct {
    msg_perm: ipc_perm,
    msg_stime: i64,
    msg_rtime: i64,
    msg_ctime: i64,
    msg_qnum: c_ulong,
    msg_qbytes: c_ulong,
    msg_lspid: c_int,
    msg_lrpid: c_int,
};

pub const msgbuf = extern struct {
    mtype: c_long,
    mtext: [1]u8,
};

pub const msginfo = extern struct {
    msgpool: c_int,
    msgmap: c_int,
    msgmax: c_int,
    msgmnb: c_int,
    msgmni: c_int,
    msgssz: c_int,
    msgtql: c_int,
    msgseg: c_ushort,
};

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

const Message = struct {
    mtype: c_long,
    size: usize,
    data: []u8,
};

const MsgQueue = struct {
    key: key_t,
    id: c_int,
    perm: ipc_perm,
    messages: std.ArrayList(Message),
    qnum: std.atomic.Value(u32),
    qbytes: usize,
    max_qbytes: usize,
    stime: i64,
    rtime: i64,
    ctime: i64,
    lspid: c_int,
    lrpid: c_int,
    mutex: std.Thread.Mutex,
    send_cond: std.Thread.Condition,
    recv_cond: std.Thread.Condition,
    marked_for_deletion: bool,
};

var msg_queues = std.AutoHashMap(c_int, *MsgQueue).init(std.heap.page_allocator);
var msg_key_to_id = std.AutoHashMap(key_t, c_int).init(std.heap.page_allocator);
var msg_mutex = std.Thread.Mutex{};
var next_msgid: c_int = 2000;

const MSGMNI: c_int = 32000; // Max number of message queues
const MSGMAX: usize = 8192;  // Max message size
const MSGMNB: usize = 16384; // Max queue bytes

pub export fn msgget(key: key_t, msgflg: c_int) c_int {
    msg_mutex.lock();
    defer msg_mutex.unlock();
    
    // Check for existing queue
    if (key != IPC_PRIVATE) {
        if (msg_key_to_id.get(key)) |id| {
            if (msgflg & IPC_CREAT != 0 and msgflg & IPC_EXCL != 0) {
                setErrno(.EXIST);
                return -1;
            }
            
            // Check permissions (simplified)
            const queue = msg_queues.get(id).?;
            if (queue.marked_for_deletion) {
                setErrno(.EIDRM);
                return -1;
            }
            
            return id;
        }
    }
    
    // Create new queue
    if (msgflg & IPC_CREAT == 0) {
        setErrno(.NOENT);
        return -1;
    }
    
    // Check system limits
    if (msg_queues.count() >= MSGMNI) {
        setErrno(.NOSPC);
        return -1;
    }
    
    const queue = std.heap.page_allocator.create(MsgQueue) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const msgid = next_msgid;
    next_msgid += 1;
    
    const now = std.time.timestamp();
    queue.* = MsgQueue{
        .key = key,
        .id = msgid,
        .perm = ipc_perm{
            .__key = key,
            .uid = std.posix.system.getuid(),
            .gid = std.posix.system.getgid(),
            .cuid = 0,
            .cgid = 0,
            .mode = @intCast(msgflg & 0o777),
            .__pad1 = 0,
            .__seq = 0,
            .__pad2 = 0,
            .__unused1 = 0,
            .__unused2 = 0,
        },
        .messages = std.ArrayList(Message).init(std.heap.page_allocator),
        .qnum = std.atomic.Value(u32).init(0),
        .qbytes = 0,
        .max_qbytes = MSGMNB,
        .stime = 0,
        .rtime = 0,
        .ctime = now,
        .lspid = 0,
        .lrpid = 0,
        .mutex = std.Thread.Mutex{},
        .send_cond = std.Thread.Condition{},
        .recv_cond = std.Thread.Condition{},
        .marked_for_deletion = false,
    };
    
    msg_queues.put(msgid, queue) catch {
        std.heap.page_allocator.destroy(queue);
        setErrno(.NOMEM);
        return -1;
    };
    
    if (key != IPC_PRIVATE) {
        msg_key_to_id.put(key, msgid) catch {};
    }
    
    return msgid;
}

pub export fn msgsnd(msqid: c_int, msgp: *const anyopaque, msgsz: usize, msgflg: c_int) c_int {
    if (msgsz > MSGMAX) {
        setErrno(.EINVAL);
        return -1;
    }
    
    const msg = @as(*const msgbuf, @ptrCast(@alignCast(msgp)));
    if (msg.mtype < 1) {
        setErrno(.EINVAL);
        return -1;
    }
    
    msg_mutex.lock();
    const queue = msg_queues.get(msqid) orelse {
        msg_mutex.unlock();
        setErrno(.EINVAL);
        return -1;
    };
    msg_mutex.unlock();
    
    queue.mutex.lock();
    defer queue.mutex.unlock();
    
    if (queue.marked_for_deletion) {
        setErrno(.EIDRM);
        return -1;
    }
    
    // Check if queue has space
    while (queue.qbytes + msgsz > queue.max_qbytes) {
        if (msgflg & IPC_NOWAIT != 0) {
            setErrno(.AGAIN);
            return -1;
        }
        // Block until space available
        queue.send_cond.wait(&queue.mutex);
        
        if (queue.marked_for_deletion) {
            setErrno(.EIDRM);
            return -1;
        }
    }
    
    // Copy message data
    const data = std.heap.page_allocator.alloc(u8, msgsz) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    @memcpy(data, @as([*]const u8, @ptrCast(&msg.mtext))[0..msgsz]);
    
    // Add to queue
    queue.messages.append(Message{
        .mtype = msg.mtype,
        .size = msgsz,
        .data = data,
    }) catch {
        std.heap.page_allocator.free(data);
        setErrno(.NOMEM);
        return -1;
    };
    
    queue.qbytes += msgsz;
    _ = queue.qnum.fetchAdd(1, .seq_cst);
    queue.stime = std.time.timestamp();
    queue.lspid = std.posix.system.getpid();
    
    // Wake up receivers
    queue.recv_cond.signal();
    
    return 0;
}

pub export fn msgrcv(msqid: c_int, msgp: *anyopaque, msgsz: usize, msgtyp: c_long, msgflg: c_int) isize {
    msg_mutex.lock();
    const queue = msg_queues.get(msqid) orelse {
        msg_mutex.unlock();
        setErrno(.EINVAL);
        return -1;
    };
    msg_mutex.unlock();
    
    queue.mutex.lock();
    defer queue.mutex.unlock();
    
    if (queue.marked_for_deletion) {
        setErrno(.EIDRM);
        return -1;
    }
    
    // Find matching message based on msgtyp
    while (true) {
        var msg_idx: ?usize = null;
        
        for (queue.messages.items, 0..) |msg, i| {
            const matches = if (msgtyp == 0) {
                true // Any message
            } else if (msgtyp > 0) {
                msg.mtype == msgtyp // Exact type
            } else {
                msg.mtype <= -msgtyp // Type <= |msgtyp|
            };
            
            if (matches) {
                msg_idx = i;
                break;
            }
        }
        
        if (msg_idx) |idx| {
            const msg = queue.messages.orderedRemove(idx);
            defer std.heap.page_allocator.free(msg.data);
            
            // Check size
            if (msg.size > msgsz) {
                if (msgflg & MSG_NOERROR == 0) {
                    // Put message back
                    queue.messages.insert(idx, msg) catch {};
                    setErrno(.E2BIG);
                    return -1;
                }
                // Truncate message
            }
            
            // Copy to user buffer
            const out = @as(*msgbuf, @ptrCast(@alignCast(msgp)));
            out.mtype = msg.mtype;
            const copy_size = @min(msg.size, msgsz);
            @memcpy(@as([*]u8, @ptrCast(&out.mtext))[0..copy_size], msg.data[0..copy_size]);
            
            queue.qbytes -= msg.size;
            _ = queue.qnum.fetchSub(1, .seq_cst);
            queue.rtime = std.time.timestamp();
            queue.lrpid = std.posix.system.getpid();
            
            // Wake up senders
            queue.send_cond.signal();
            
            return @intCast(copy_size);
        }
        
        // No matching message
        if (msgflg & IPC_NOWAIT != 0) {
            setErrno(.NOMSG);
            return -1;
        }
        
        // Block until message arrives
        queue.recv_cond.wait(&queue.mutex);
        
        if (queue.marked_for_deletion) {
            setErrno(.EIDRM);
            return -1;
        }
    }
}

pub export fn msgctl(msqid: c_int, cmd: c_int, buf: ?*msqid_ds) c_int {
    msg_mutex.lock();
    defer msg_mutex.unlock();
    
    const queue = msg_queues.get(msqid) orelse {
        setErrno(.EINVAL);
        return -1;
    };
    
    queue.mutex.lock();
    defer queue.mutex.unlock();
    
    switch (cmd) {
        IPC_RMID => {
            queue.marked_for_deletion = true;
            
            // Wake all blocked processes
            queue.send_cond.broadcast();
            queue.recv_cond.broadcast();
            
            // Free all messages
            for (queue.messages.items) |msg| {
                std.heap.page_allocator.free(msg.data);
            }
            queue.messages.deinit();
            
            // Remove from registry
            _ = msg_queues.remove(msqid);
            if (queue.key != IPC_PRIVATE) {
                _ = msg_key_to_id.remove(queue.key);
            }
            
            std.heap.page_allocator.destroy(queue);
            return 0;
        },
        IPC_STAT => {
            if (buf) |ds| {
                ds.msg_perm = queue.perm;
                ds.msg_stime = queue.stime;
                ds.msg_rtime = queue.rtime;
                ds.msg_ctime = queue.ctime;
                ds.msg_qnum = queue.qnum.load(.seq_cst);
                ds.msg_qbytes = queue.max_qbytes;
                ds.msg_lspid = queue.lspid;
                ds.msg_lrpid = queue.lrpid;
            }
            return 0;
        },
        IPC_SET => {
            if (buf) |ds| {
                queue.perm.uid = ds.msg_perm.uid;
                queue.perm.gid = ds.msg_perm.gid;
                queue.perm.mode = ds.msg_perm.mode & 0o777;
                
                // Update max bytes if valid
                if (ds.msg_qbytes <= MSGMNB) {
                    queue.max_qbytes = ds.msg_qbytes;
                }
                
                queue.ctime = std.time.timestamp();
            }
            return 0;
        },
        else => {
            setErrno(.EINVAL);
            return -1;
        },
    }
}
