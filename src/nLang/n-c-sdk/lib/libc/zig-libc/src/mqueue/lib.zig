// mqueue module - Phase 1.21
const std = @import("std");

pub const mqd_t = c_int;

pub const mq_attr = extern struct {
    mq_flags: c_long,
    mq_maxmsg: c_long,
    mq_msgsize: c_long,
    mq_curmsgs: c_long,
};

pub export fn mq_open(name: [*:0]const u8, oflag: c_int, ...) mqd_t {
    _ = name; _ = oflag;
    return 1;
}

pub export fn mq_close(mqdes: mqd_t) c_int {
    _ = mqdes;
    return 0;
}

pub export fn mq_unlink(name: [*:0]const u8) c_int {
    _ = name;
    return 0;
}

pub export fn mq_send(mqdes: mqd_t, msg_ptr: [*]const u8, msg_len: usize, msg_prio: c_uint) c_int {
    _ = mqdes; _ = msg_ptr; _ = msg_len; _ = msg_prio;
    return 0;
}

pub export fn mq_receive(mqdes: mqd_t, msg_ptr: [*]u8, msg_len: usize, msg_prio: ?*c_uint) isize {
    _ = mqdes; _ = msg_ptr; _ = msg_len; _ = msg_prio;
    return 0;
}

pub export fn mq_timedsend(mqdes: mqd_t, msg_ptr: [*]const u8, msg_len: usize, msg_prio: c_uint, abs_timeout: ?*const anyopaque) c_int {
    _ = mqdes; _ = msg_ptr; _ = msg_len; _ = msg_prio; _ = abs_timeout;
    return 0;
}

pub export fn mq_timedreceive(mqdes: mqd_t, msg_ptr: [*]u8, msg_len: usize, msg_prio: ?*c_uint, abs_timeout: ?*const anyopaque) isize {
    _ = mqdes; _ = msg_ptr; _ = msg_len; _ = msg_prio; _ = abs_timeout;
    return 0;
}

pub export fn mq_notify(mqdes: mqd_t, notification: ?*const anyopaque) c_int {
    _ = mqdes; _ = notification;
    return 0;
}

pub export fn mq_getattr(mqdes: mqd_t, attr: *mq_attr) c_int {
    _ = mqdes;
    @memset(std.mem.asBytes(attr), 0);
    return 0;
}

pub export fn mq_setattr(mqdes: mqd_t, newattr: *const mq_attr, oldattr: ?*mq_attr) c_int {
    _ = mqdes; _ = newattr; _ = oldattr;
    return 0;
}
