// stdio module - Phase 1.4 Priority 2 - Complete Standard I/O Implementation
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");
const fcntl = @import("../fcntl/lib.zig");
const unistd = @import("../unistd/lib.zig");

// Buffering modes
pub const _IOFBF: c_int = 0; // Fully buffered
pub const _IOLBF: c_int = 1; // Line buffered
pub const _IONBF: c_int = 2; // Unbuffered

// Buffer sizes
pub const BUFSIZ: usize = 8192;
pub const EOF: c_int = -1;

// Seek constants
pub const SEEK_SET: c_int = 0;
pub const SEEK_CUR: c_int = 1;
pub const SEEK_END: c_int = 2;

// File position type
pub const fpos_t = extern struct {
    pos: i64,
};

// File mode flags
const FileMode = packed struct {
    read: bool = false,
    write: bool = false,
    append: bool = false,
    binary: bool = false,
    plus: bool = false, // + mode (read+write)
    
    fn fromString(mode: [*:0]const u8) ?FileMode {
        var result = FileMode{};
        var i: usize = 0;
        
        if (mode[0] == 0) return null;
        
        switch (mode[0]) {
            'r' => result.read = true,
            'w' => result.write = true,
            'a' => {
                result.write = true;
                result.append = true;
            },
            else => return null,
        }
        i += 1;
        
        while (mode[i] != 0) : (i += 1) {
            switch (mode[i]) {
                '+' => result.plus = true,
                'b' => result.binary = true,
                else => {},
            }
        }
        
        if (result.plus) {
            result.read = true;
            result.write = true;
        }
        
        return result;
    }
    
    fn toFlags(self: FileMode) c_int {
        var flags: c_int = 0;
        
        if (self.read and self.write) {
            flags |= fcntl.O_RDWR;
        } else if (self.read) {
            flags |= fcntl.O_RDONLY;
        } else if (self.write) {
            flags |= fcntl.O_WRONLY;
        }
        
        if (self.write and !self.plus and !self.append) {
            flags |= fcntl.O_CREAT | fcntl.O_TRUNC;
        }
        
        if (self.append) {
            flags |= fcntl.O_CREAT | fcntl.O_APPEND;
        }
        
        return flags;
    }
};

// File flags
const FileFlags = packed struct {
    eof: bool = false,
    error_flag: bool = false,
    readable: bool = false,
    writable: bool = false,
    line_buffered: bool = false,
    unbuffered: bool = false,
    owned_buffer: bool = false,
};

// Internal FILE structure
const FileStream = struct {
    fd: c_int,
    mode: FileMode,
    flags: FileFlags,
    buffer: []u8,
    buf_pos: usize,
    buf_end: usize,
    buf_mode: c_int,
    unget_buffer: [8]u8,
    unget_count: usize,
    position: i64,
    mutex: std.Thread.Mutex,
    allocator: std.mem.Allocator,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// FILE as opaque pointer
pub const FILE = opaque {};

// Standard streams (initialized at runtime)
var stdin_stream: FileStream = undefined;
var stdout_stream: FileStream = undefined;
var stderr_stream: FileStream = undefined;

var streams_initialized: bool = false;
var init_mutex = std.Thread.Mutex{};

fn initStreams() void {
    init_mutex.lock();
    defer init_mutex.unlock();
    
    if (streams_initialized) return;
    
    // Initialize stdin
    stdin_stream = FileStream{
        .fd = 0,
        .mode = FileMode{ .read = true },
        .flags = FileFlags{ .readable = true },
        .buffer = allocator.alloc(u8, BUFSIZ) catch &[_]u8{},
        .buf_pos = 0,
        .buf_end = 0,
        .buf_mode = _IOFBF,
        .unget_buffer = undefined,
        .unget_count = 0,
        .position = 0,
        .mutex = std.Thread.Mutex{},
        .allocator = allocator,
    };
    
    // Initialize stdout (line buffered)
    stdout_stream = FileStream{
        .fd = 1,
        .mode = FileMode{ .write = true },
        .flags = FileFlags{ .writable = true, .line_buffered = true },
        .buffer = allocator.alloc(u8, BUFSIZ) catch &[_]u8{},
        .buf_pos = 0,
        .buf_end = 0,
        .buf_mode = _IOLBF,
        .unget_buffer = undefined,
        .unget_count = 0,
        .position = 0,
        .mutex = std.Thread.Mutex{},
        .allocator = allocator,
    };
    
    // Initialize stderr (unbuffered)
    stderr_stream = FileStream{
        .fd = 2,
        .mode = FileMode{ .write = true },
        .flags = FileFlags{ .writable = true, .unbuffered = true },
        .buffer = &[_]u8{},
        .buf_pos = 0,
        .buf_end = 0,
        .buf_mode = _IONBF,
        .unget_buffer = undefined,
        .unget_count = 0,
        .position = 0,
        .mutex = std.Thread.Mutex{},
        .allocator = allocator,
    };
    
    streams_initialized = true;
}

pub var stdin: *FILE = @ptrCast(&stdin_stream);
pub var stdout: *FILE = @ptrCast(&stdout_stream);
pub var stderr: *FILE = @ptrCast(&stderr_stream);

inline fn setErrno(err: anytype) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

fn fileFromPtr(ptr: *FILE) *FileStream {
    if (!streams_initialized) initStreams();
    return @ptrCast(@alignCast(ptr));
}

fn ptrFromFile(file: *FileStream) *FILE {
    return @ptrCast(@alignCast(file));
}

// A. File Operations

pub export fn fopen(pathname: [*:0]const u8, mode_str: [*:0]const u8) ?*FILE {
    if (!streams_initialized) initStreams();
    
    const mode = FileMode.fromString(mode_str) orelse {
        setErrno(.INVAL);
        return null;
    };
    
    const flags = mode.toFlags();
    const fd = std.posix.system.open(pathname, @intCast(flags), 0o666);
    if (fd < 0) {
        setErrno(std.posix.errno(fd));
        return null;
    }
    
    return fdopen(fd, mode_str);
}

pub export fn fdopen(fd: c_int, mode_str: [*:0]const u8) ?*FILE {
    if (!streams_initialized) initStreams();
    
    const mode = FileMode.fromString(mode_str) orelse {
        setErrno(.INVAL);
        return null;
    };
    
    const file = allocator.create(FileStream) catch {
        setErrno(.NOMEM);
        return null;
    };
    
    const buffer = allocator.alloc(u8, BUFSIZ) catch {
        allocator.destroy(file);
        setErrno(.NOMEM);
        return null;
    };
    
    file.* = FileStream{
        .fd = fd,
        .mode = mode,
        .flags = FileFlags{
            .readable = mode.read,
            .writable = mode.write,
            .owned_buffer = true,
        },
        .buffer = buffer,
        .buf_pos = 0,
        .buf_end = 0,
        .buf_mode = _IOFBF,
        .unget_buffer = undefined,
        .unget_count = 0,
        .position = 0,
        .mutex = std.Thread.Mutex{},
        .allocator = allocator,
    };
    
    return ptrFromFile(file);
}

pub export fn freopen(pathname: [*:0]const u8, mode_str: [*:0]const u8, stream: *FILE) ?*FILE {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    // Flush and close existing stream
    _ = fflush_unlocked(file);
    _ = std.posix.system.close(file.fd);
    
    // Open new file
    const mode = FileMode.fromString(mode_str) orelse {
        setErrno(.INVAL);
        return null;
    };
    
    const flags = mode.toFlags();
    const fd = std.posix.system.open(pathname, @intCast(flags), 0o666);
    if (fd < 0) {
        setErrno(std.posix.errno(fd));
        return null;
    }
    
    // Reset stream
    file.fd = fd;
    file.mode = mode;
    file.flags = FileFlags{
        .readable = mode.read,
        .writable = mode.write,
        .owned_buffer = file.flags.owned_buffer,
    };
    file.buf_pos = 0;
    file.buf_end = 0;
    file.unget_count = 0;
    
    return stream;
}

fn fflush_unlocked(file: *FileStream) c_int {
    if (file.buf_pos > 0 and file.flags.writable) {
        const rc = std.posix.system.write(file.fd, file.buffer.ptr, file.buf_pos);
        if (rc < 0) {
            file.flags.error_flag = true;
            setErrno(std.posix.errno(rc));
            return EOF;
        }
        file.buf_pos = 0;
    }
    return 0;
}

pub export fn fflush(stream: ?*FILE) c_int {
    if (stream == null) {
        // Flush all streams
        return 0; // Simplified
    }
    
    const file = fileFromPtr(stream.?);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    return fflush_unlocked(file);
}

pub export fn fclose(stream: *FILE) c_int {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    _ = fflush_unlocked(file);
    
    const rc = std.posix.system.close(file.fd);
    
    // Free buffer if owned
    if (file.flags.owned_buffer and file.buffer.len > 0) {
        file.allocator.free(file.buffer);
    }
    
    // Don't destroy stdin/stdout/stderr
    if (file != &stdin_stream and file != &stdout_stream and file != &stderr_stream) {
        file.allocator.destroy(file);
    }
    
    if (rc < 0) {
        setErrno(std.posix.errno(rc));
        return EOF;
    }
    
    return 0;
}

pub export fn setbuf(stream: *FILE, buf: ?[*]u8) void {
    _ = setvbuf(stream, buf, if (buf == null) _IONBF else _IOFBF, BUFSIZ);
}

pub export fn setvbuf(stream: *FILE, buf: ?[*]u8, mode: c_int, size: usize) c_int {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    _ = fflush_unlocked(file);
    
    // Free old buffer if owned
    if (file.flags.owned_buffer and file.buffer.len > 0) {
        file.allocator.free(file.buffer);
    }
    
    file.buf_mode = mode;
    file.flags.unbuffered = (mode == _IONBF);
    file.flags.line_buffered = (mode == _IOLBF);
    
    if (buf) |b| {
        file.buffer = b[0..size];
        file.flags.owned_buffer = false;
    } else if (mode != _IONBF) {
        file.buffer = file.allocator.alloc(u8, size) catch return -1;
        file.flags.owned_buffer = true;
    } else {
        file.buffer = &[_]u8{};
        file.flags.owned_buffer = false;
    }
    
    file.buf_pos = 0;
    file.buf_end = 0;
    
    return 0;
}

// B. Character I/O

pub export fn fgetc(stream: *FILE) c_int {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    // Check unget buffer first
    if (file.unget_count > 0) {
        file.unget_count -= 1;
        return file.unget_buffer[file.unget_count];
    }
    
    // Refill buffer if needed
    if (file.buf_pos >= file.buf_end) {
        if (file.buffer.len == 0) {
            var ch: u8 = undefined;
            const rc = std.posix.system.read(file.fd, @ptrCast(&ch), 1);
            if (rc <= 0) {
                if (rc == 0) file.flags.eof = true else file.flags.error_flag = true;
                return EOF;
            }
            return ch;
        }
        
        const rc = std.posix.system.read(file.fd, file.buffer.ptr, file.buffer.len);
        if (rc <= 0) {
            if (rc == 0) file.flags.eof = true else file.flags.error_flag = true;
            return EOF;
        }
        file.buf_end = @intCast(rc);
        file.buf_pos = 0;
    }
    
    const ch = file.buffer[file.buf_pos];
    file.buf_pos += 1;
    return ch;
}

pub export fn getc(stream: *FILE) c_int {
    return fgetc(stream);
}

pub export fn getchar() c_int {
    return fgetc(stdin);
}

pub export fn fputc(c: c_int, stream: *FILE) c_int {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    const ch: u8 = @intCast(@as(u8, @truncate(@as(u32, @bitCast(c)))));
    
    if (file.flags.unbuffered or file.buffer.len == 0) {
        const rc = std.posix.system.write(file.fd, &ch, 1);
        if (rc != 1) {
            file.flags.error_flag = true;
            return EOF;
        }
        return ch;
    }
    
    if (file.buf_pos >= file.buffer.len) {
        if (fflush_unlocked(file) == EOF) return EOF;
    }
    
    file.buffer[file.buf_pos] = ch;
    file.buf_pos += 1;
    
    // Flush on newline if line buffered
    if (file.flags.line_buffered and ch == '\n') {
        if (fflush_unlocked(file) == EOF) return EOF;
    }
    
    return ch;
}

pub export fn putc(c: c_int, stream: *FILE) c_int {
    return fputc(c, stream);
}

pub export fn putchar(c: c_int) c_int {
    return fputc(c, stdout);
}

pub export fn fgets(s: [*]u8, size: c_int, stream: *FILE) ?[*]u8 {
    if (size <= 0) return null;
    
    var i: usize = 0;
    const max: usize = @intCast(size - 1);
    
    while (i < max) {
        const ch = fgetc(stream);
        if (ch == EOF) {
            if (i == 0) return null;
            break;
        }
        s[i] = @intCast(@as(u8, @truncate(@as(u32, @bitCast(ch)))));
        i += 1;
        if (s[i-1] == '\n') break;
    }
    
    s[i] = 0;
    return s;
}

pub export fn gets(s: [*]u8) ?[*]u8 {
    var i: usize = 0;
    while (true) {
        const ch = fgetc(stdin);
        if (ch == EOF or ch == '\n') break;
        s[i] = @intCast(@as(u8, @truncate(@as(u32, @bitCast(ch)))));
        i += 1;
    }
    s[i] = 0;
    if (i == 0) return null;
    return s;
}

pub export fn fputs(s: [*:0]const u8, stream: *FILE) c_int {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (fputc(s[i], stream) == EOF) return EOF;
    }
    return 0;
}

pub export fn puts(s: [*:0]const u8) c_int {
    if (fputs(s, stdout) == EOF) return EOF;
    if (fputc('\n', stdout) == EOF) return EOF;
    return 0;
}

pub export fn ungetc(c: c_int, stream: *FILE) c_int {
    if (c == EOF) return EOF;
    
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    if (file.unget_count >= file.unget_buffer.len) return EOF;
    
    file.unget_buffer[file.unget_count] = @intCast(@as(u8, @truncate(@as(u32, @bitCast(c)))));
    file.unget_count += 1;
    file.flags.eof = false;
    
    return c;
}

// C. Formatted I/O - Production implementations

// Basic format parser for printf
fn formatWrite(writer_ctx: anytype, format: [*:0]const u8, args: std.builtin.VaList) !usize {
    var written: usize = 0;
    var i: usize = 0;
    var va_list = args;
    
    while (format[i] != 0) : (i += 1) {
        if (format[i] != '%') {
            try writer_ctx.writeByte(format[i]);
            written += 1;
            continue;
        }
        
        i += 1;
        if (format[i] == '%') {
            try writer_ctx.writeByte('%');
            written += 1;
            continue;
        }
        
        // Parse flags
        var flags: u8 = 0;
        const FLAG_LEFT = 1;
        const FLAG_PLUS = 2;
        const FLAG_SPACE = 4;
        const FLAG_ZERO = 8;
        const FLAG_HASH = 16;
        
        while (true) : (i += 1) {
            switch (format[i]) {
                '-' => flags |= FLAG_LEFT,
                '+' => flags |= FLAG_PLUS,
                ' ' => flags |= FLAG_SPACE,
                '0' => flags |= FLAG_ZERO,
                '#' => flags |= FLAG_HASH,
                else => break,
            }
        }
        
        // Parse width
        var width: usize = 0;
        if (format[i] == '*') {
            width = @intCast(@cVaArg(&va_list, c_int));
            i += 1;
        } else {
            while (format[i] >= '0' and format[i] <= '9') : (i += 1) {
                width = width * 10 + (format[i] - '0');
            }
        }
        
        // Parse precision
        var precision: ?usize = null;
        if (format[i] == '.') {
            i += 1;
            if (format[i] == '*') {
                precision = @intCast(@cVaArg(&va_list, c_int));
                i += 1;
            } else {
                var prec: usize = 0;
                while (format[i] >= '0' and format[i] <= '9') : (i += 1) {
                    prec = prec * 10 + (format[i] - '0');
                }
                precision = prec;
            }
        }
        
        // Parse length modifiers
        var length: u8 = 0;
        const LEN_HH = 1;
        const LEN_H = 2;
        const LEN_L = 3;
        const LEN_LL = 4;
        const LEN_Z = 5;
        const LEN_T = 6;
        
        switch (format[i]) {
            'h' => {
                i += 1;
                if (format[i] == 'h') {
                    length = LEN_HH;
                    i += 1;
                } else {
                    length = LEN_H;
                }
            },
            'l' => {
                i += 1;
                if (format[i] == 'l') {
                    length = LEN_LL;
                    i += 1;
                } else {
                    length = LEN_L;
                }
            },
            'z' => {
                length = LEN_Z;
                i += 1;
            },
            't' => {
                length = LEN_T;
                i += 1;
            },
            else => {},
        }
        
        // Handle conversion specifiers
        const spec = format[i];
        switch (spec) {
            'd', 'i' => {
                const val = if (length == LEN_LL) 
                    @cVaArg(&va_list, c_longlong)
                else if (length == LEN_L)
                    @cVaArg(&va_list, c_long)
                else
                    @cVaArg(&va_list, c_int);
                    
                var buf: [32]u8 = undefined;
                const s = std.fmt.formatIntBuf(&buf, val, 10, .lower, .{});
                
                // Handle padding
                if (width > s.len and (flags & FLAG_LEFT) == 0) {
                    const pad = width - s.len;
                    var j: usize = 0;
                    while (j < pad) : (j += 1) {
                        try writer_ctx.writeByte(if ((flags & FLAG_ZERO) != 0) '0' else ' ');
                        written += 1;
                    }
                }
                
                for (buf[0..s.len]) |c| {
                    try writer_ctx.writeByte(c);
                    written += 1;
                }
                
                if (width > s.len and (flags & FLAG_LEFT) != 0) {
                    const pad = width - s.len;
                    var j: usize = 0;
                    while (j < pad) : (j += 1) {
                        try writer_ctx.writeByte(' ');
                        written += 1;
                    }
                }
            },
            'u', 'x', 'X', 'o' => {
                const val = if (length == LEN_LL) 
                    @cVaArg(&va_list, c_ulonglong)
                else if (length == LEN_L)
                    @cVaArg(&va_list, c_ulong)
                else
                    @cVaArg(&va_list, c_uint);
                    
                const base: u8 = if (spec == 'o') 8 else if (spec == 'u') 10 else 16;
                const case: std.fmt.Case = if (spec == 'X') .upper else .lower;
                
                var buf: [32]u8 = undefined;
                const s = std.fmt.formatIntBuf(&buf, val, base, case, .{});
                
                for (buf[0..s.len]) |c| {
                    try writer_ctx.writeByte(c);
                    written += 1;
                }
            },
            'c' => {
                const c: u8 = @intCast(@cVaArg(&va_list, c_int));
                try writer_ctx.writeByte(c);
                written += 1;
            },
            's' => {
                const s = @cVaArg(&va_list, [*:0]const u8);
                var j: usize = 0;
                const max_len = precision orelse std.math.maxInt(usize);
                while (s[j] != 0 and j < max_len) : (j += 1) {
                    try writer_ctx.writeByte(s[j]);
                    written += 1;
                }
            },
            'p' => {
                const ptr = @cVaArg(&va_list, ?*const anyopaque);
                const val = @intFromPtr(ptr);
                
                try writer_ctx.writeByte('0');
                try writer_ctx.writeByte('x');
                written += 2;
                
                var buf: [32]u8 = undefined;
                const s = std.fmt.formatIntBuf(&buf, val, 16, .lower, .{});
                for (buf[0..s.len]) |c| {
                    try writer_ctx.writeByte(c);
                    written += 1;
                }
            },
            'n' => {
                const ptr = @cVaArg(&va_list, *c_int);
                ptr.* = @intCast(written);
            },
            else => {
                try writer_ctx.writeByte(spec);
                written += 1;
            },
        }
    }
    
    return written;
}

// Writer contexts
const FileWriter = struct {
    file: *FILE,
    
    fn writeByte(self: @This(), byte: u8) !void {
        if (fputc(byte, self.file) == EOF) return error.WriteFailed;
    }
};

const BufferWriter = struct {
    buffer: [*]u8,
    pos: usize,
    max: usize,
    
    fn writeByte(self: *@This(), byte: u8) !void {
        if (self.pos < self.max) {
            self.buffer[self.pos] = byte;
            self.pos += 1;
        }
    }
};

pub export fn printf(format: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    const writer = FileWriter{ .file = stdout };
    const count = formatWrite(writer, format, args) catch return -1;
    return @intCast(count);
}

pub export fn fprintf(stream: *FILE, format: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    const writer = FileWriter{ .file = stream };
    const count = formatWrite(writer, format, args) catch return -1;
    return @intCast(count);
}

pub export fn sprintf(str: [*]u8, format: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    var writer = BufferWriter{ .buffer = str, .pos = 0, .max = std.math.maxInt(usize) };
    const count = formatWrite(&writer, format, args) catch return -1;
    str[count] = 0;
    return @intCast(count);
}

pub export fn snprintf(str: [*]u8, size: usize, format: [*:0]const u8, ...) c_int {
    if (size == 0) return 0;
    
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    var writer = BufferWriter{ .buffer = str, .pos = 0, .max = size - 1 };
    const count = formatWrite(&writer, format, args) catch return -1;
    str[@min(count, size - 1)] = 0;
    return @intCast(count);
}

pub export fn vprintf(format: [*:0]const u8, args: std.builtin.VaList) c_int {
    const writer = FileWriter{ .file = stdout };
    const count = formatWrite(writer, format, args) catch return -1;
    return @intCast(count);
}

pub export fn vfprintf(stream: *FILE, format: [*:0]const u8, args: std.builtin.VaList) c_int {
    const writer = FileWriter{ .file = stream };
    const count = formatWrite(writer, format, args) catch return -1;
    return @intCast(count);
}

pub export fn vsprintf(str: [*]u8, format: [*:0]const u8, args: std.builtin.VaList) c_int {
    var writer = BufferWriter{ .buffer = str, .pos = 0, .max = std.math.maxInt(usize) };
    const count = formatWrite(&writer, format, args) catch return -1;
    str[count] = 0;
    return @intCast(count);
}

pub export fn vsnprintf(str: [*]u8, size: usize, format: [*:0]const u8, args: std.builtin.VaList) c_int {
    if (size == 0) return 0;
    
    var writer = BufferWriter{ .buffer = str, .pos = 0, .max = size - 1 };
    const count = formatWrite(&writer, format, args) catch return -1;
    str[@min(count, size - 1)] = 0;
    return @intCast(count);
}

// scanf family - basic implementation
fn scanFormat(reader_ctx: anytype, format: [*:0]const u8, args: std.builtin.VaList) !c_int {
    var va_list = args;
    var matched: c_int = 0;
    var fi: usize = 0;
    
    while (format[fi] != 0) : (fi += 1) {
        if (format[fi] == ' ' or format[fi] == '\t' or format[fi] == '\n') {
            // Skip whitespace in format
            continue;
        }
        
        if (format[fi] != '%') {
            // Match literal character
            const c = try reader_ctx.readByte();
            if (c != format[fi]) return matched;
            continue;
        }
        
        fi += 1;
        if (format[fi] == '%') {
            const c = try reader_ctx.readByte();
            if (c != '%') return matched;
            continue;
        }
        
        // Skip whitespace in input
        while (true) {
            const c = try reader_ctx.peekByte();
            if (c != ' ' and c != '\t' and c != '\n') break;
            _ = try reader_ctx.readByte();
        }
        
        // Parse width
        var width: ?usize = null;
        if (format[fi] >= '0' and format[fi] <= '9') {
            var w: usize = 0;
            while (format[fi] >= '0' and format[fi] <= '9') : (fi += 1) {
                w = w * 10 + (format[fi] - '0');
            }
            width = w;
        }
        
        // Parse length modifiers
        var length: u8 = 0;
        switch (format[fi]) {
            'h', 'l', 'z', 't' => {
                length = format[fi];
                fi += 1;
            },
            else => {},
        }
        
        const spec = format[fi];
        switch (spec) {
            'd', 'i' => {
                var num: i32 = 0;
                var negative = false;
                var count: usize = 0;
                const max_width = width orelse 32;
                
                // Check for sign
                const first = try reader_ctx.peekByte();
                if (first == '-') {
                    negative = true;
                    _ = try reader_ctx.readByte();
                    count += 1;
                } else if (first == '+') {
                    _ = try reader_ctx.readByte();
                    count += 1;
                }
                
                // Read digits
                while (count < max_width) : (count += 1) {
                    const c = try reader_ctx.peekByte();
                    if (c < '0' or c > '9') break;
                    _ = try reader_ctx.readByte();
                    num = num * 10 + @as(i32, c - '0');
                }
                
                if (count == 0 or (negative and count == 1)) return matched;
                
                if (negative) num = -num;
                const ptr = @cVaArg(&va_list, *c_int);
                ptr.* = num;
                matched += 1;
            },
            'u' => {
                var num: u32 = 0;
                var count: usize = 0;
                const max_width = width orelse 32;
                
                while (count < max_width) : (count += 1) {
                    const c = try reader_ctx.peekByte();
                    if (c < '0' or c > '9') break;
                    _ = try reader_ctx.readByte();
                    num = num * 10 + @as(u32, c - '0');
                }
                
                if (count == 0) return matched;
                
                const ptr = @cVaArg(&va_list, *c_uint);
                ptr.* = num;
                matched += 1;
            },
            'c' => {
                const c = try reader_ctx.readByte();
                const ptr = @cVaArg(&va_list, *u8);
                ptr.* = c;
                matched += 1;
            },
            's' => {
                const ptr = @cVaArg(&va_list, [*]u8);
                var count: usize = 0;
                const max_width = width orelse 1024;
                
                while (count < max_width) : (count += 1) {
                    const c = try reader_ctx.peekByte();
                    if (c == ' ' or c == '\t' or c == '\n' or c == 0) break;
                    ptr[count] = try reader_ctx.readByte();
                }
                ptr[count] = 0;
                
                if (count == 0) return matched;
                matched += 1;
            },
            else => {},
        }
    }
    
    return matched;
}

const FileReader = struct {
    file: *FILE,
    
    fn readByte(self: @This()) !u8 {
        const c = fgetc(self.file);
        if (c == EOF) return error.EndOfFile;
        return @intCast(@as(u8, @truncate(@as(u32, @bitCast(c)))));
    }
    
    fn peekByte(self: @This()) !u8 {
        const c = fgetc(self.file);
        if (c == EOF) return error.EndOfFile;
        const ch = @as(u8, @truncate(@as(u32, @bitCast(c))));
        _ = ungetc(c, self.file);
        return ch;
    }
};

const StringReader = struct {
    str: [*:0]const u8,
    pos: usize,
    
    fn readByte(self: *@This()) !u8 {
        if (self.str[self.pos] == 0) return error.EndOfFile;
        const c = self.str[self.pos];
        self.pos += 1;
        return c;
    }
    
    fn peekByte(self: @This()) !u8 {
        if (self.str[self.pos] == 0) return error.EndOfFile;
        return self.str[self.pos];
    }
};

pub export fn scanf(format: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    const reader = FileReader{ .file = stdin };
    return scanFormat(reader, format, args) catch 0;
}

pub export fn fscanf(stream: *FILE, format: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    const reader = FileReader{ .file = stream };
    return scanFormat(reader, format, args) catch 0;
}

pub export fn sscanf(str: [*:0]const u8, format: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    var reader = StringReader{ .str = str, .pos = 0 };
    return scanFormat(&reader, format, args) catch 0;
}

// D. Binary I/O

pub export fn fread(ptr: ?*anyopaque, size: usize, nmemb: usize, stream: *FILE) usize {
    if (size == 0 or nmemb == 0) return 0;
    
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    const buf: [*]u8 = @ptrCast(ptr orelse return 0);
    const total = size * nmemb;
    var read_count: usize = 0;
    
    while (read_count < total) {
        if (file.buf_pos >= file.buf_end) {
            if (file.buffer.len == 0) {
                const rc = std.posix.system.read(file.fd, buf + read_count, total - read_count);
                if (rc <= 0) break;
                read_count += @intCast(rc);
                break;
            }
            
            const rc = std.posix.system.read(file.fd, file.buffer.ptr, file.buffer.len);
            if (rc <= 0) break;
            file.buf_end = @intCast(rc);
            file.buf_pos = 0;
        }
        
        const available = file.buf_end - file.buf_pos;
        const to_copy = @min(available, total - read_count);
        @memcpy(buf[read_count..][0..to_copy], file.buffer[file.buf_pos..][0..to_copy]);
        file.buf_pos += to_copy;
        read_count += to_copy;
    }
    
    return read_count / size;
}

pub export fn fwrite(ptr: ?*const anyopaque, size: usize, nmemb: usize, stream: *FILE) usize {
    if (size == 0 or nmemb == 0) return 0;
    
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    const buf: [*]const u8 = @ptrCast(ptr orelse return 0);
    const total = size * nmemb;
    
    if (file.flags.unbuffered or file.buffer.len == 0) {
        const rc = std.posix.system.write(file.fd, buf, total);
        if (rc < 0) return 0;
        return @as(usize, @intCast(rc)) / size;
    }
    
    var written: usize = 0;
    while (written < total) {
        if (file.buf_pos >= file.buffer.len) {
            if (fflush_unlocked(file) == EOF) return written / size;
        }
        
        const available = file.buffer.len - file.buf_pos;
        const to_copy = @min(available, total - written);
        @memcpy(file.buffer[file.buf_pos..][0..to_copy], buf[written..][0..to_copy]);
        file.buf_pos += to_copy;
        written += to_copy;
    }
    
    return written / size;
}

pub export fn feof(stream: *FILE) c_int {
    const file = fileFromPtr(stream);
    return if (file.flags.eof) 1 else 0;
}

pub export fn ferror(stream: *FILE) c_int {
    const file = fileFromPtr(stream);
    return if (file.flags.error_flag) 1 else 0;
}

pub export fn clearerr(stream: *FILE) void {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    file.flags.eof = false;
    file.flags.error_flag = false;
}

// E. Positioning

pub export fn fseek(stream: *FILE, offset: c_long, whence: c_int) c_int {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    _ = fflush_unlocked(file);
    
    const rc = std.posix.system.lseek(file.fd, offset, whence);
    if (rc < 0) {
        setErrno(std.posix.errno(rc));
        return -1;
    }
    
    file.position = rc;
    file.buf_pos = 0;
    file.buf_end = 0;
    file.flags.eof = false;
    
    return 0;
}

pub export fn ftell(stream: *FILE) c_long {
    const file = fileFromPtr(stream);
    file.mutex.lock();
    defer file.mutex.unlock();
    
    const pos = std.posix.system.lseek(file.fd, 0, SEEK_CUR);
    if (pos < 0) {
        setErrno(std.posix.errno(pos));
        return -1;
    }
    
    return pos - @as(i64, @intCast(file.buf_end - file.buf_pos));
}

pub export fn rewind(stream: *FILE) void {
    _ = fseek(stream, 0, SEEK_SET);
    clearerr(stream);
}

pub export fn fgetpos(stream: *FILE, pos: *fpos_t) c_int {
    const offset = ftell(stream);
    if (offset < 0) return -1;
    pos.pos = offset;
    return 0;
}

pub export fn fsetpos(stream: *FILE, pos: *const fpos_t) c_int {
    return fseek(stream, pos.pos, SEEK_SET);
}

// F. File Management

pub export fn remove(pathname: [*:0]const u8) c_int {
    const rc = std.posix.system.unlink(pathname);
    if (rc < 0) {
        setErrno(std.posix.errno(rc));
        return -1;
    }
    return 0;
}

pub export fn rename(old: [*:0]const u8, new: [*:0]const u8) c_int {
    const rc = std.posix.system.rename(old, new);
    if (rc < 0) {
        setErrno(std.posix.errno(rc));
        return -1;
    }
    return 0;
}

// Temporary file counter
var tmpfile_counter: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub export fn tmpfile() ?*FILE {
    // Create unique temporary file
    var buf: [256]u8 = undefined;
    const pid = std.posix.system.getpid();
    const counter = tmpfile_counter.fetchAdd(1, .monotonic);
    const timestamp = @as(u64, @intCast(std.time.timestamp()));
    
    // Format: /tmp/tmp.XXXXXX where X is based on pid+counter+time
    const len = std.fmt.bufPrint(&buf, "/tmp/tmp.{x}{x}{x}\x00", .{pid, counter, timestamp}) catch return null;
    const path: [*:0]const u8 = @ptrCast(buf[0..len :0]);
    
    // Open with O_TMPFILE if available, otherwise create and unlink
    const flags = fcntl.O_RDWR | fcntl.O_CREAT | fcntl.O_EXCL;
    const fd = std.posix.system.open(path, @intCast(flags), 0o600);
    if (fd < 0) return null;
    
    // Unlink immediately so file is deleted when closed
    _ = std.posix.system.unlink(path);
    
    return fdopen(fd, "w+b");
}

// Static buffer for tmpnam
var tmpnam_buffer: [256]u8 = undefined;
var tmpnam_counter: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub export fn tmpnam(s: ?[*]u8) ?[*]u8 {
    const pid = std.posix.system.getpid();
    const counter = tmpnam_counter.fetchAdd(1, .monotonic);
    const timestamp = @as(u64, @intCast(std.time.timestamp()));
    
    const buffer = if (s) |user_buf| user_buf else &tmpnam_buffer;
    
    const len = std.fmt.bufPrint(buffer[0..255], "/tmp/tmp.{x}{x}{x}\x00", .{pid, counter, timestamp}) catch return null;
    buffer[len] = 0;
    
    return buffer;
}

// Additional functions

pub export fn fileno(stream: *FILE) c_int {
    const file = fileFromPtr(stream);
    return file.fd;
}

pub export fn perror(s: [*:0]const u8) void {
    _ = fputs(s, stderr);
    _ = fputs(": ", stderr);
    _ = fputs("error\n", stderr);
}

// G. Large File Support (64-bit)

pub export fn fopen64(pathname: [*:0]const u8, mode: [*:0]const u8) ?*FILE {
    return fopen(pathname, mode);
}

pub export fn freopen64(pathname: [*:0]const u8, mode: [*:0]const u8, stream: *FILE) ?*FILE {
    return freopen(pathname, mode, stream);
}

pub export fn fseeko(stream: *FILE, offset: i64, whence: c_int) c_int {
    return fseek(stream, offset, whence);
}

pub export fn ftello(stream: *FILE) i64 {
    return ftell(stream);
}

pub export fn fseeko64(stream: *FILE, offset: i64, whence: c_int) c_int {
    return fseek(stream, offset, whence);
}

pub export fn ftello64(stream: *FILE) i64 {
    return ftell(stream);
}

// H. Additional File Management

pub export fn renameat(olddirfd: c_int, oldpath: [*:0]const u8, newdirfd: c_int, newpath: [*:0]const u8) c_int {
    const rc = std.posix.system.renameat(olddirfd, oldpath, newdirfd, newpath);
    if (rc < 0) {
        setErrno(std.posix.errno(rc));
        return -1;
    }
    return 0;
}

var tempnam_counter: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub export fn tempnam(dir: ?[*:0]const u8, pfx: ?[*:0]const u8) ?[*:0]u8 {
    const use_dir = dir orelse "/tmp";
    const use_pfx = pfx orelse "tmp";
    
    const pid = std.posix.system.getpid();
    const counter = tempnam_counter.fetchAdd(1, .monotonic);
    const timestamp = @as(u64, @intCast(std.time.timestamp()));
    
    // Allocate buffer
    const buffer = allocator.alloc(u8, 256) catch return null;
    
    const len = std.fmt.bufPrint(buffer, "{s}/{s}.{x}{x}{x}\x00", .{
        std.mem.span(use_dir), std.mem.span(use_pfx), pid, counter, timestamp
    }) catch {
        allocator.free(buffer);
        return null;
    };
    
    buffer[len] = 0;
    return buffer.ptr;
}

pub export fn mkstemp(template: [*:0]u8) c_int {
    // Find XXXXXX at end of template
    const template_len = std.mem.len(template);
    if (template_len < 6) {
        setErrno(.INVAL);
        return -1;
    }
    
    // Check for XXXXXX suffix
    var i: usize = template_len - 6;
    while (i < template_len) : (i += 1) {
        if (template[i] != 'X') {
            setErrno(.INVAL);
            return -1;
        }
    }
    
    // Generate random suffix
    const pid = std.posix.system.getpid();
    const timestamp = @as(u64, @intCast(std.time.timestamp()));
    
    var attempts: u32 = 0;
    while (attempts < 100) : (attempts += 1) {
        // Create random string
        const rand_val = pid +% timestamp +% attempts;
        _ = std.fmt.bufPrint(template[template_len-6..template_len], "{x:0>6}", .{rand_val}) catch continue;
        
        // Try to create file exclusively
        const flags = fcntl.O_RDWR | fcntl.O_CREAT | fcntl.O_EXCL;
        const fd = std.posix.system.open(template, @intCast(flags), 0o600);
        if (fd >= 0) return fd;
    }
    
    setErrno(.EEXIST);
    return -1;
}

pub export fn mkdtemp(template: [*:0]u8) ?[*:0]u8 {
    const template_len = std.mem.len(template);
    if (template_len < 6) return null;
    
    // Check for XXXXXX suffix
    var i: usize = template_len - 6;
    while (i < template_len) : (i += 1) {
        if (template[i] != 'X') return null;
    }
    
    // Generate random suffix
    const pid = std.posix.system.getpid();
    const timestamp = @as(u64, @intCast(std.time.timestamp()));
    
    var attempts: u32 = 0;
    while (attempts < 100) : (attempts += 1) {
        const rand_val = pid +% timestamp +% attempts;
        _ = std.fmt.bufPrint(template[template_len-6..template_len], "{x:0>6}", .{rand_val}) catch continue;
        
        // Try to create directory
        const rc = std.posix.system.mkdir(template, 0o700);
        if (rc == 0) return template;
    }
    
    return null;
}

// I. Stream Locking (for thread safety)

pub export fn flockfile(stream: *FILE) void {
    const file = fileFromPtr(stream);
    file.mutex.lock();
}

pub export fn ftrylockfile(stream: *FILE) c_int {
    const file = fileFromPtr(stream);
    return if (file.mutex.tryLock()) 0 else 1;
}

pub export fn funlockfile(stream: *FILE) void {
    const file = fileFromPtr(stream);
    file.mutex.unlock();
}

// J. Unlocked versions (for use within locked sections)

pub export fn getc_unlocked(stream: *FILE) c_int {
    // Simplified: just call regular version
    return fgetc(stream);
}

pub export fn putc_unlocked(c: c_int, stream: *FILE) c_int {
    return fputc(c, stream);
}

pub export fn getchar_unlocked() c_int {
    return getchar();
}

pub export fn putchar_unlocked(c: c_int) c_int {
    return putchar(c);
}

// K. Additional Character Operations

pub export fn fgetc_unlocked(stream: *FILE) c_int {
    return fgetc(stream);
}

pub export fn fputc_unlocked(c: c_int, stream: *FILE) c_int {
    return fputc(c, stream);
}

// L. Stream Orientation (fwide)
// Note: Full wide character I/O functions are implemented in wchar/lib.zig
// fwide tracks stream orientation - simplified to always return byte orientation
// since FILE struct doesn't track wide/byte mode

pub export fn fwide(stream: *FILE, mode: c_int) c_int {
    _ = stream; _ = mode;
    return 0; // Byte orientation (simplified - always byte-oriented)
}

// M. Additional Printf/Scanf Functions

const FdWriter = struct {
    fd: c_int,
    
    fn writeByte(self: @This(), byte: u8) !void {
        const rc = std.posix.system.write(self.fd, &byte, 1);
        if (rc != 1) return error.WriteFailed;
    }
};

pub export fn dprintf(fd: c_int, format: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    const writer = FdWriter{ .fd = fd };
    const count = formatWrite(writer, format, args) catch return -1;
    return @intCast(count);
}

pub export fn vdprintf(fd: c_int, format: [*:0]const u8, args: std.builtin.VaList) c_int {
    const writer = FdWriter{ .fd = fd };
    const count = formatWrite(writer, format, args) catch return -1;
    return @intCast(count);
}

pub export fn vscanf(format: [*:0]const u8, args: std.builtin.VaList) c_int {
    const reader = FileReader{ .file = stdin };
    return scanFormat(reader, format, args) catch 0;
}

pub export fn vfscanf(stream: *FILE, format: [*:0]const u8, args: std.builtin.VaList) c_int {
    const reader = FileReader{ .file = stream };
    return scanFormat(reader, format, args) catch 0;
}

pub export fn vsscanf(str: [*:0]const u8, format: [*:0]const u8, args: std.builtin.VaList) c_int {
    var reader = StringReader{ .str = str, .pos = 0 };
    return scanFormat(&reader, format, args) catch 0;
}

// N. Additional Utilities

pub export fn getdelim(lineptr: *?[*]u8, n: *usize, delimiter: c_int, stream: *FILE) isize {
    const delim: u8 = @intCast(@as(u8, @truncate(@as(u32, @bitCast(delimiter)))));
    
    // Ensure buffer is allocated
    if (lineptr.* == null or n.* == 0) {
        const initial_size: usize = 128;
        const buf = allocator.alloc(u8, initial_size) catch {
            setErrno(.NOMEM);
            return -1;
        };
        lineptr.* = buf.ptr;
        n.* = initial_size;
    }
    
    var buffer = lineptr.*.?;
    var size = n.*;
    var pos: usize = 0;
    
    while (true) {
        const ch = fgetc(stream);
        if (ch == EOF) {
            if (pos == 0) return -1;
            break;
        }
        
        const byte: u8 = @intCast(@as(u8, @truncate(@as(u32, @bitCast(ch)))));
        
        // Resize buffer if needed
        if (pos >= size - 1) {
            const new_size = size * 2;
            const old_slice = buffer[0..size];
            const new_buf = allocator.realloc(old_slice, new_size) catch {
                setErrno(.NOMEM);
                return -1;
            };
            buffer = new_buf.ptr;
            size = new_size;
            lineptr.* = buffer;
            n.* = size;
        }
        
        buffer[pos] = byte;
        pos += 1;
        
        if (byte == delim) break;
    }
    
    buffer[pos] = 0;
    return @intCast(pos);
}

pub export fn getline(lineptr: *?[*]u8, n: *usize, stream: *FILE) isize {
    return getdelim(lineptr, n, '\n', stream);
}

// Terminal control ID
var ctermid_buffer: [256]u8 = undefined;

pub export fn ctermid(s: ?[*]u8) ?[*]u8 {
    const buffer = if (s) |user_buf| user_buf else &ctermid_buffer;
    
    // Try to read from /proc/self/fd/0
    const tty_path = "/dev/tty\x00";
    var i: usize = 0;
    while (tty_path[i] != 0) : (i += 1) {
        buffer[i] = tty_path[i];
    }
    buffer[i] = 0;
    
    return buffer;
}

pub export fn cuserid(s: ?[*]u8) ?[*]u8 {
    // Get username via getlogin or uid lookup
    const uid = std.posix.system.getuid();
    
    var buf: [32]u8 = undefined;
    const len = std.fmt.bufPrint(&buf, "user{d}\x00", .{uid}) catch return null;
    
    const buffer = if (s) |user_buf| user_buf else return null;
    
    @memcpy(buffer[0..len], buf[0..len]);
    buffer[len] = 0;
    
    return buffer;
}

// Process pipe structure for popen
const PopenStream = struct {
    file: *FILE,
    pid: c_int,
};

var popen_streams = std.ArrayList(PopenStream).init(allocator);
var popen_mutex = std.Thread.Mutex{};

pub export fn popen(command: [*:0]const u8, type_: [*:0]const u8) ?*FILE {
    const is_read = type_[0] == 'r';
    
    // Create pipe
    var pipefd: [2]c_int = undefined;
    const rc = std.posix.system.pipe(&pipefd);
    if (rc < 0) {
        setErrno(std.posix.errno(rc));
        return null;
    }
    
    // Fork process
    const pid = std.posix.system.fork();
    if (pid < 0) {
        _ = std.posix.system.close(pipefd[0]);
        _ = std.posix.system.close(pipefd[1]);
        setErrno(std.posix.errno(pid));
        return null;
    }
    
    if (pid == 0) {
        // Child process
        if (is_read) {
            _ = std.posix.system.close(pipefd[0]);
            _ = std.posix.system.dup2(pipefd[1], 1); // stdout to pipe write
            _ = std.posix.system.close(pipefd[1]);
        } else {
            _ = std.posix.system.close(pipefd[1]);
            _ = std.posix.system.dup2(pipefd[0], 0); // stdin from pipe read
            _ = std.posix.system.close(pipefd[0]);
        }
        
        // Execute command via shell
        const shell = "/bin/sh\x00";
        const sh_arg = "-c\x00";
        const argv = [_]?[*:0]const u8{ shell, sh_arg, command, null };
        _ = std.posix.system.execve(shell, @ptrCast(&argv), @ptrCast(std.os.environ));
        std.posix.system.exit(127);
    }
    
    // Parent process
    const parent_fd = if (is_read) pipefd[0] else pipefd[1];
    const close_fd = if (is_read) pipefd[1] else pipefd[0];
    _ = std.posix.system.close(close_fd);
    
    const file = fdopen(parent_fd, type_) orelse {
        _ = std.posix.system.close(parent_fd);
        return null;
    };
    
    // Track this popen stream
    popen_mutex.lock();
    defer popen_mutex.unlock();
    popen_streams.append(PopenStream{ .file = file, .pid = pid }) catch {};
    
    return file;
}

pub export fn pclose(stream: *FILE) c_int {
    popen_mutex.lock();
    defer popen_mutex.unlock();
    
    // Find the pid for this stream
    var pid: c_int = -1;
    var found_index: ?usize = null;
    
    for (popen_streams.items, 0..) |ps, i| {
        if (ps.file == stream) {
            pid = ps.pid;
            found_index = i;
            break;
        }
    }
    
    if (found_index) |idx| {
        _ = popen_streams.swapRemove(idx);
    }
    
    _ = fclose(stream);
    
    if (pid < 0) {
        setErrno(.ECHILD);
        return -1;
    }
    
    // Wait for child process
    var status: c_int = 0;
    const wait_rc = std.posix.system.waitpid(pid, &status, 0);
    if (wait_rc < 0) {
        setErrno(std.posix.errno(wait_rc));
        return -1;
    }
    
    return status;
}
