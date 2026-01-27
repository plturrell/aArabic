// semaphore module - Phase 1.15
const std = @import("std");

pub const sem_t = extern struct {
    __size: [32]u8,
};

pub const SEM_FAILED: ?*sem_t = null;

pub export fn sem_init(sem: *sem_t, pshared: c_int, value: c_uint) c_int {
    _ = pshared; _ = value;
    @memset(std.mem.asBytes(sem), 0);
    return 0;
}

pub export fn sem_destroy(sem: *sem_t) c_int {
    _ = sem;
    return 0;
}

pub export fn sem_wait(sem: *sem_t) c_int {
    _ = sem;
    return 0;
}

pub export fn sem_trywait(sem: *sem_t) c_int {
    _ = sem;
    return 0;
}

pub export fn sem_timedwait(sem: *sem_t, abs_timeout: ?*const anyopaque) c_int {
    _ = sem; _ = abs_timeout;
    return 0;
}

pub export fn sem_post(sem: *sem_t) c_int {
    _ = sem;
    return 0;
}

pub export fn sem_getvalue(sem: *sem_t, sval: *c_int) c_int {
    _ = sem;
    sval.* = 0;
    return 0;
}

pub export fn sem_open(name: [*:0]const u8, oflag: c_int, ...) ?*sem_t {
    _ = name; _ = oflag;
    return @ptrFromInt(1);
}

pub export fn sem_close(sem: *sem_t) c_int {
    _ = sem;
    return 0;
}

pub export fn sem_unlink(name: [*:0]const u8) c_int {
    _ = name;
    return 0;
}
