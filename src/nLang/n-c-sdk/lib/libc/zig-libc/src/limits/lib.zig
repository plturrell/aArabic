// limits module - Phase 1.5
// System limits and constants

// Char limits
pub const CHAR_BIT: c_int = 8;
pub const SCHAR_MIN: c_int = -128;
pub const SCHAR_MAX: c_int = 127;
pub const UCHAR_MAX: c_int = 255;
pub const CHAR_MIN: c_int = SCHAR_MIN;
pub const CHAR_MAX: c_int = SCHAR_MAX;

// Short limits
pub const SHRT_MIN: c_int = -32768;
pub const SHRT_MAX: c_int = 32767;
pub const USHRT_MAX: c_uint = 65535;

// Int limits
pub const INT_MIN: c_int = -2147483648;
pub const INT_MAX: c_int = 2147483647;
pub const UINT_MAX: c_uint = 4294967295;

// Long limits (64-bit)
pub const LONG_MIN: c_long = -9223372036854775808;
pub const LONG_MAX: c_long = 9223372036854775807;
pub const ULONG_MAX: c_ulong = 18446744073709551615;

// Long long limits
pub const LLONG_MIN: c_longlong = -9223372036854775808;
pub const LLONG_MAX: c_longlong = 9223372036854775807;
pub const ULLONG_MAX: c_ulonglong = 18446744073709551615;

// Multibyte limits
pub const MB_LEN_MAX: c_int = 16;

// POSIX limits
pub const PATH_MAX: c_int = 4096;
pub const NAME_MAX: c_int = 255;
pub const PIPE_BUF: c_int = 4096;
pub const OPEN_MAX: c_int = 1024;
pub const ARG_MAX: c_int = 131072;
pub const CHILD_MAX: c_int = 999;
pub const LINK_MAX: c_int = 127;
pub const MAX_CANON: c_int = 255;
pub const MAX_INPUT: c_int = 255;
pub const NGROUPS_MAX: c_int = 65536;
pub const SYMLOOP_MAX: c_int = 40;
