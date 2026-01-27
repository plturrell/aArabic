// errno module - Phase 1.5
const std = @import("std");

// Thread-local errno
threadlocal var _errno: c_int = 0;

pub export fn __errno_location() *c_int {
    return &_errno;
}

// Standard errno values
pub const E2BIG: c_int = 7;
pub const EACCES: c_int = 13;
pub const EADDRINUSE: c_int = 48;
pub const EADDRNOTAVAIL: c_int = 49;
pub const EAFNOSUPPORT: c_int = 47;
pub const EAGAIN: c_int = 35;
pub const EALREADY: c_int = 37;
pub const EBADF: c_int = 9;
pub const EBADMSG: c_int = 94;
pub const EBUSY: c_int = 16;
pub const ECANCELED: c_int = 89;
pub const ECHILD: c_int = 10;
pub const ECONNABORTED: c_int = 53;
pub const ECONNREFUSED: c_int = 61;
pub const ECONNRESET: c_int = 54;
pub const EDEADLK: c_int = 11;
pub const EDESTADDRREQ: c_int = 39;
pub const EDOM: c_int = 33;
pub const EDQUOT: c_int = 69;
pub const EEXIST: c_int = 17;
pub const EFAULT: c_int = 14;
pub const EFBIG: c_int = 27;
pub const EHOSTUNREACH: c_int = 65;
pub const EIDRM: c_int = 90;
pub const EILSEQ: c_int = 92;
pub const EINPROGRESS: c_int = 36;
pub const EINTR: c_int = 4;
pub const EINVAL: c_int = 22;
pub const EIO: c_int = 5;
pub const EISCONN: c_int = 56;
pub const EISDIR: c_int = 21;
pub const ELOOP: c_int = 62;
pub const EMFILE: c_int = 24;
pub const EMLINK: c_int = 31;
pub const EMSGSIZE: c_int = 40;
pub const ENAMETOOLONG: c_int = 63;
pub const ENETDOWN: c_int = 50;
pub const ENETRESET: c_int = 52;
pub const ENETUNREACH: c_int = 51;
pub const ENFILE: c_int = 23;
pub const ENOBUFS: c_int = 55;
pub const ENODEV: c_int = 19;
pub const ENOENT: c_int = 2;
pub const ENOEXEC: c_int = 8;
pub const ENOLCK: c_int = 77;
pub const ENOMEM: c_int = 12;
pub const ENOMSG: c_int = 91;
pub const ENOPROTOOPT: c_int = 42;
pub const ENOSPC: c_int = 28;
pub const ENOSYS: c_int = 78;
pub const ENOTCONN: c_int = 57;
pub const ENOTDIR: c_int = 20;
pub const ENOTEMPTY: c_int = 66;
pub const ENOTSOCK: c_int = 38;
pub const ENOTSUP: c_int = 45;
pub const ENOTTY: c_int = 25;
pub const ENXIO: c_int = 6;
pub const EOPNOTSUPP: c_int = 102;
pub const EOVERFLOW: c_int = 84;
pub const EPERM: c_int = 1;
pub const EPIPE: c_int = 32;
pub const ERANGE: c_int = 34;
pub const EROFS: c_int = 30;
pub const ESPIPE: c_int = 29;
pub const ESRCH: c_int = 3;
pub const ETIMEDOUT: c_int = 60;
pub const ETXTBSY: c_int = 26;
pub const EWOULDBLOCK: c_int = 35;
pub const EXDEV: c_int = 18;
