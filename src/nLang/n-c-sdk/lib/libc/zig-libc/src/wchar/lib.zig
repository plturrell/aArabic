// wchar module - Phase 1.10 Priority 8 - Wide Character Support
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");
const stdio = @import("../stdio/lib.zig");
const time_mod = @import("../time/lib.zig");

pub const wchar_t = c_int;
pub const wint_t = c_uint;
pub const wctype_t = c_ulong;
pub const mbstate_t = extern struct { __opaque: [128]u8 };

pub const WEOF: wint_t = 0xffffffff;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Wide character I/O functions

/// Get a wide character from a stream
pub export fn fgetwc(stream: *stdio.FILE) wint_t {
    // Read a multibyte character and convert to wide character
    // Simplified: assume 1 byte = 1 wide char (ASCII/Latin1)
    const c = stdio.fgetc(stream);
    if (c == stdio.EOF) return WEOF;
    return @intCast(@as(u32, @bitCast(c)));
}

/// Get a wide character from a stream (same as fgetwc)
pub export fn getwc(stream: *stdio.FILE) wint_t {
    return fgetwc(stream);
}

/// Get a wide character from stdin
pub export fn getwchar() wint_t {
    return fgetwc(stdio.stdin);
}

/// Put a wide character to a stream
pub export fn fputwc(wc: wchar_t, stream: *stdio.FILE) wint_t {
    // Convert wide character to multibyte and write
    // Simplified: assume 1 wide char = 1 byte (ASCII/Latin1)
    if (wc < 0 or wc > 255) {
        // For non-ASCII wide chars, write as UTF-8 would be better
        // but for now, just truncate to byte
        const c = stdio.fputc(@intCast(wc & 0x7f), stream);
        if (c == stdio.EOF) return WEOF;
        return @intCast(@as(u32, @bitCast(wc)));
    }
    const c = stdio.fputc(@intCast(wc & 0xff), stream);
    if (c == stdio.EOF) return WEOF;
    return @intCast(@as(u32, @bitCast(wc)));
}

/// Put a wide character to a stream (same as fputwc)
pub export fn putwc(wc: wchar_t, stream: *stdio.FILE) wint_t {
    return fputwc(wc, stream);
}

/// Put a wide character to stdout
pub export fn putwchar(wc: wchar_t) wint_t {
    return fputwc(wc, stdio.stdout);
}

/// Push a wide character back onto the stream
pub export fn ungetwc(wc: wint_t, stream: *stdio.FILE) wint_t {
    if (wc == WEOF) return WEOF;
    // Simplified: only handle single-byte pushback
    const c = stdio.ungetc(@intCast(wc & 0xff), stream);
    if (c == stdio.EOF) return WEOF;
    return wc;
}

/// Read a wide character string from a stream
pub export fn fgetws(ws: [*]wchar_t, n: c_int, stream: *stdio.FILE) ?[*]wchar_t {
    if (n <= 0) return null;

    var i: usize = 0;
    const max: usize = @intCast(n - 1);

    while (i < max) {
        const wc = fgetwc(stream);
        if (wc == WEOF) {
            if (i == 0) return null;
            break;
        }
        ws[i] = @intCast(wc);
        i += 1;
        if (wc == '\n') break;
    }
    ws[i] = 0;
    return ws;
}

/// Write a wide character string to a stream
pub export fn fputws(ws: [*:0]const wchar_t, stream: *stdio.FILE) c_int {
    var i: usize = 0;
    while (ws[i] != 0) : (i += 1) {
        if (fputwc(ws[i], stream) == WEOF) return -1;
    }
    return 0;
}

// Wide character format string parsing and I/O

// Length modifier constants
const LEN_NONE: u8 = 0;
const LEN_H: u8 = 1;
const LEN_HH: u8 = 2;
const LEN_L: u8 = 3;
const LEN_LL: u8 = 4;
const LEN_Z: u8 = 5;
const LEN_T: u8 = 6;

// Flag constants
const FLAG_LEFT: u8 = 1;
const FLAG_PLUS: u8 = 2;
const FLAG_SPACE: u8 = 4;
const FLAG_ZERO: u8 = 8;
const FLAG_HASH: u8 = 16;

/// Helper: Write integer to wide char buffer, returns number of chars written
fn intToWide(buf: [*]wchar_t, max_len: usize, value: anytype, base: u8, uppercase: bool) usize {
    if (max_len == 0) return 0;

    const T = @TypeOf(value);
    const is_signed = @typeInfo(T) == .int and @typeInfo(T).int.signedness == .signed;

    var val: u64 = undefined;
    var negative = false;

    if (is_signed) {
        if (value < 0) {
            negative = true;
            val = @intCast(-@as(i64, value));
        } else {
            val = @intCast(value);
        }
    } else {
        val = @intCast(value);
    }

    // Build digits in reverse
    var tmp: [32]wchar_t = undefined;
    var len: usize = 0;
    const digits = if (uppercase) "0123456789ABCDEF" else "0123456789abcdef";

    if (val == 0) {
        tmp[len] = '0';
        len = 1;
    } else {
        while (val > 0 and len < 32) {
            tmp[len] = digits[@intCast(val % base)];
            val /= base;
            len += 1;
        }
    }

    // Write sign
    var written: usize = 0;
    if (negative and written < max_len) {
        buf[written] = '-';
        written += 1;
    }

    // Write digits in correct order
    while (len > 0 and written < max_len) {
        len -= 1;
        buf[written] = tmp[len];
        written += 1;
    }

    return written;
}

/// Wide char writer for FILE streams
const WideFileWriter = struct {
    file: *stdio.FILE,
    count: usize,
    failed: bool,

    fn writeWchar(self: *@This(), wc: wchar_t) void {
        if (self.failed) return;
        if (fputwc(wc, self.file) == WEOF) {
            self.failed = true;
        } else {
            self.count += 1;
        }
    }

    fn writeWideStr(self: *@This(), ws: [*:0]const wchar_t) void {
        var i: usize = 0;
        while (ws[i] != 0) : (i += 1) {
            self.writeWchar(ws[i]);
            if (self.failed) return;
        }
    }

    fn writeNarrowStr(self: *@This(), s: [*:0]const u8) void {
        var i: usize = 0;
        while (s[i] != 0) : (i += 1) {
            self.writeWchar(@intCast(s[i]));
            if (self.failed) return;
        }
    }

    fn writePadding(self: *@This(), count: usize, ch: wchar_t) void {
        for (0..count) |_| {
            self.writeWchar(ch);
            if (self.failed) return;
        }
    }
};

/// Wide char writer for buffer
const WideBufferWriter = struct {
    buf: [*]wchar_t,
    max: usize,
    pos: usize,
    count: usize, // Total chars that would be written

    fn writeWchar(self: *@This(), wc: wchar_t) void {
        self.count += 1;
        if (self.pos < self.max) {
            self.buf[self.pos] = wc;
            self.pos += 1;
        }
    }

    fn writeWideStr(self: *@This(), ws: [*:0]const wchar_t) void {
        var i: usize = 0;
        while (ws[i] != 0) : (i += 1) {
            self.writeWchar(ws[i]);
        }
    }

    fn writeNarrowStr(self: *@This(), s: [*:0]const u8) void {
        var i: usize = 0;
        while (s[i] != 0) : (i += 1) {
            self.writeWchar(@intCast(s[i]));
        }
    }

    fn writePadding(self: *@This(), count: usize, ch: wchar_t) void {
        for (0..count) |_| {
            self.writeWchar(ch);
        }
    }
};

/// Format wide string with varargs
fn wideFormatWrite(writer: anytype, format: [*:0]const wchar_t, args: std.builtin.VaList) void {
    var va_list = args;
    var i: usize = 0;

    while (format[i] != 0) : (i += 1) {
        if (format[i] != '%') {
            writer.writeWchar(format[i]);
            continue;
        }

        i += 1;
        if (format[i] == 0) break;
        if (format[i] == '%') {
            writer.writeWchar('%');
            continue;
        }

        // Parse flags
        var flags: u8 = 0;
        while (true) {
            switch (format[i]) {
                '-' => flags |= FLAG_LEFT,
                '+' => flags |= FLAG_PLUS,
                ' ' => flags |= FLAG_SPACE,
                '0' => flags |= FLAG_ZERO,
                '#' => flags |= FLAG_HASH,
                else => break,
            }
            i += 1;
        }

        // Parse width
        var width: usize = 0;
        if (format[i] == '*') {
            const w = @cVaArg(&va_list, c_int);
            width = if (w > 0) @intCast(w) else 0;
            i += 1;
        } else {
            while (format[i] >= '0' and format[i] <= '9') {
                width = width * 10 + @as(usize, @intCast(format[i] - '0'));
                i += 1;
            }
        }

        // Parse precision
        var precision: ?usize = null;
        if (format[i] == '.') {
            i += 1;
            if (format[i] == '*') {
                const p = @cVaArg(&va_list, c_int);
                precision = if (p >= 0) @intCast(p) else 0;
                i += 1;
            } else {
                var prec: usize = 0;
                while (format[i] >= '0' and format[i] <= '9') {
                    prec = prec * 10 + @as(usize, @intCast(format[i] - '0'));
                    i += 1;
                }
                precision = prec;
            }
        }

        // Parse length modifier
        var length: u8 = LEN_NONE;
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

        // Handle conversion specifier
        const spec = format[i];
        switch (spec) {
            'd', 'i' => {
                const val = if (length == LEN_LL)
                    @cVaArg(&va_list, c_longlong)
                else if (length == LEN_L)
                    @cVaArg(&va_list, c_long)
                else
                    @cVaArg(&va_list, c_int);

                var buf: [32]wchar_t = undefined;
                const len = intToWide(&buf, 32, val, 10, false);

                // Padding before
                if (width > len and (flags & FLAG_LEFT) == 0) {
                    const pad_char: wchar_t = if ((flags & FLAG_ZERO) != 0) '0' else ' ';
                    writer.writePadding(width - len, pad_char);
                }

                for (0..len) |j| {
                    writer.writeWchar(buf[j]);
                }

                // Padding after
                if (width > len and (flags & FLAG_LEFT) != 0) {
                    writer.writePadding(width - len, ' ');
                }
            },
            'u' => {
                const val = if (length == LEN_LL)
                    @cVaArg(&va_list, c_ulonglong)
                else if (length == LEN_L)
                    @cVaArg(&va_list, c_ulong)
                else
                    @cVaArg(&va_list, c_uint);

                var buf: [32]wchar_t = undefined;
                const len = intToWide(&buf, 32, val, 10, false);

                if (width > len and (flags & FLAG_LEFT) == 0) {
                    const pad_char: wchar_t = if ((flags & FLAG_ZERO) != 0) '0' else ' ';
                    writer.writePadding(width - len, pad_char);
                }

                for (0..len) |j| {
                    writer.writeWchar(buf[j]);
                }

                if (width > len and (flags & FLAG_LEFT) != 0) {
                    writer.writePadding(width - len, ' ');
                }
            },
            'x', 'X' => {
                const val = if (length == LEN_LL)
                    @cVaArg(&va_list, c_ulonglong)
                else if (length == LEN_L)
                    @cVaArg(&va_list, c_ulong)
                else
                    @cVaArg(&va_list, c_uint);

                var buf: [32]wchar_t = undefined;
                const len = intToWide(&buf, 32, val, 16, spec == 'X');

                if (width > len and (flags & FLAG_LEFT) == 0) {
                    const pad_char: wchar_t = if ((flags & FLAG_ZERO) != 0) '0' else ' ';
                    writer.writePadding(width - len, pad_char);
                }

                for (0..len) |j| {
                    writer.writeWchar(buf[j]);
                }

                if (width > len and (flags & FLAG_LEFT) != 0) {
                    writer.writePadding(width - len, ' ');
                }
            },
            'o' => {
                const val = if (length == LEN_LL)
                    @cVaArg(&va_list, c_ulonglong)
                else if (length == LEN_L)
                    @cVaArg(&va_list, c_ulong)
                else
                    @cVaArg(&va_list, c_uint);

                var buf: [32]wchar_t = undefined;
                const len = intToWide(&buf, 32, val, 8, false);

                if (width > len and (flags & FLAG_LEFT) == 0) {
                    const pad_char: wchar_t = if ((flags & FLAG_ZERO) != 0) '0' else ' ';
                    writer.writePadding(width - len, pad_char);
                }

                for (0..len) |j| {
                    writer.writeWchar(buf[j]);
                }

                if (width > len and (flags & FLAG_LEFT) != 0) {
                    writer.writePadding(width - len, ' ');
                }
            },
            'c' => {
                if (length == LEN_L) {
                    // %lc - wide char
                    const wc = @cVaArg(&va_list, wint_t);
                    if (width > 1 and (flags & FLAG_LEFT) == 0) {
                        writer.writePadding(width - 1, ' ');
                    }
                    writer.writeWchar(@intCast(wc));
                    if (width > 1 and (flags & FLAG_LEFT) != 0) {
                        writer.writePadding(width - 1, ' ');
                    }
                } else {
                    // %c - narrow char
                    const c: u8 = @intCast(@cVaArg(&va_list, c_int) & 0xFF);
                    if (width > 1 and (flags & FLAG_LEFT) == 0) {
                        writer.writePadding(width - 1, ' ');
                    }
                    writer.writeWchar(@intCast(c));
                    if (width > 1 and (flags & FLAG_LEFT) != 0) {
                        writer.writePadding(width - 1, ' ');
                    }
                }
            },
            's' => {
                if (length == LEN_L) {
                    // %ls - wide string
                    const ws = @cVaArg(&va_list, [*:0]const wchar_t);
                    var slen = wcslen(ws);
                    if (precision) |p| {
                        slen = @min(slen, p);
                    }

                    if (width > slen and (flags & FLAG_LEFT) == 0) {
                        writer.writePadding(width - slen, ' ');
                    }

                    for (0..slen) |j| {
                        writer.writeWchar(ws[j]);
                    }

                    if (width > slen and (flags & FLAG_LEFT) != 0) {
                        writer.writePadding(width - slen, ' ');
                    }
                } else {
                    // %s - narrow string (convert to wide)
                    const s = @cVaArg(&va_list, [*:0]const u8);
                    var slen: usize = 0;
                    while (s[slen] != 0) : (slen += 1) {}
                    if (precision) |p| {
                        slen = @min(slen, p);
                    }

                    if (width > slen and (flags & FLAG_LEFT) == 0) {
                        writer.writePadding(width - slen, ' ');
                    }

                    for (0..slen) |j| {
                        writer.writeWchar(@intCast(s[j]));
                    }

                    if (width > slen and (flags & FLAG_LEFT) != 0) {
                        writer.writePadding(width - slen, ' ');
                    }
                }
            },
            'p' => {
                const ptr = @cVaArg(&va_list, ?*const anyopaque);
                const val = @intFromPtr(ptr);

                writer.writeWchar('0');
                writer.writeWchar('x');

                var buf: [32]wchar_t = undefined;
                const len = intToWide(&buf, 32, val, 16, false);

                for (0..len) |j| {
                    writer.writeWchar(buf[j]);
                }
            },
            'n' => {
                const ptr = @cVaArg(&va_list, *c_int);
                ptr.* = @intCast(writer.count);
            },
            else => {
                writer.writeWchar(spec);
            },
        }
    }
}

/// Wide character printf
pub export fn wprintf(format: [*:0]const wchar_t, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    return vwprintf(format, args);
}

/// Wide character fprintf
pub export fn fwprintf(stream: *stdio.FILE, format: [*:0]const wchar_t, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    return vfwprintf(stream, format, args);
}

/// Wide character swprintf
pub export fn swprintf(ws: [*]wchar_t, n: usize, format: [*:0]const wchar_t, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    return vswprintf(ws, n, format, args);
}

/// Wide character vwprintf
pub export fn vwprintf(format: [*:0]const wchar_t, args: std.builtin.VaList) c_int {
    var writer = WideFileWriter{ .file = stdio.stdout, .count = 0, .failed = false };
    wideFormatWrite(&writer, format, args);
    return if (writer.failed) -1 else @intCast(writer.count);
}

/// Wide character vfwprintf
pub export fn vfwprintf(stream: *stdio.FILE, format: [*:0]const wchar_t, args: std.builtin.VaList) c_int {
    var writer = WideFileWriter{ .file = stream, .count = 0, .failed = false };
    wideFormatWrite(&writer, format, args);
    return if (writer.failed) -1 else @intCast(writer.count);
}

/// Wide character vswprintf
pub export fn vswprintf(ws: [*]wchar_t, n: usize, format: [*:0]const wchar_t, args: std.builtin.VaList) c_int {
    if (n == 0) return -1;
    var writer = WideBufferWriter{ .buf = ws, .max = n - 1, .pos = 0, .count = 0 };
    wideFormatWrite(&writer, format, args);
    // Null terminate
    ws[writer.pos] = 0;
    // Return -1 if output was truncated (per C standard)
    return if (writer.count >= n) -1 else @intCast(writer.count);
}

// Wide scanf implementation

/// Wide char reader for FILE streams
const WideFileReader = struct {
    file: *stdio.FILE,

    fn readWchar(self: @This()) !wint_t {
        const wc = fgetwc(self.file);
        if (wc == WEOF) return error.EndOfFile;
        return wc;
    }

    fn peekWchar(self: @This()) !wint_t {
        const wc = fgetwc(self.file);
        if (wc == WEOF) return error.EndOfFile;
        _ = ungetwc(wc, self.file);
        return wc;
    }

    fn unreadWchar(self: @This(), wc: wint_t) void {
        _ = ungetwc(wc, self.file);
    }
};

/// Wide char reader for string
const WideStringReader = struct {
    str: [*:0]const wchar_t,
    pos: usize,

    fn readWchar(self: *@This()) !wint_t {
        if (self.str[self.pos] == 0) return error.EndOfFile;
        const wc: wint_t = @intCast(self.str[self.pos]);
        self.pos += 1;
        return wc;
    }

    fn peekWchar(self: @This()) !wint_t {
        if (self.str[self.pos] == 0) return error.EndOfFile;
        return @intCast(self.str[self.pos]);
    }

    fn unreadWchar(self: *@This(), wc: wint_t) void {
        _ = wc;
        if (self.pos > 0) self.pos -= 1;
    }
};

/// Scan wide format string with varargs
fn wideScanFormat(reader: anytype, format: [*:0]const wchar_t, args: std.builtin.VaList) c_int {
    var va_list = args;
    var matched: c_int = 0;
    var fi: usize = 0;

    while (format[fi] != 0) : (fi += 1) {
        // Skip whitespace in format (matches any whitespace in input)
        if (iswspace(@intCast(format[fi])) != 0) {
            while (true) {
                const wc = reader.peekWchar() catch break;
                if (iswspace(wc) == 0) break;
                _ = reader.readWchar() catch break;
            }
            continue;
        }

        if (format[fi] != '%') {
            // Match literal character
            const wc = reader.readWchar() catch return matched;
            if (wc != @as(wint_t, @intCast(format[fi]))) return matched;
            continue;
        }

        fi += 1;
        if (format[fi] == 0) break;
        if (format[fi] == '%') {
            const wc = reader.readWchar() catch return matched;
            if (wc != '%') return matched;
            continue;
        }

        // Check for assignment suppression
        var suppress = false;
        if (format[fi] == '*') {
            suppress = true;
            fi += 1;
        }

        // Parse width
        var width: ?usize = null;
        if (format[fi] >= '0' and format[fi] <= '9') {
            var w: usize = 0;
            while (format[fi] >= '0' and format[fi] <= '9') : (fi += 1) {
                w = w * 10 + @as(usize, @intCast(format[fi] - '0'));
            }
            width = w;
        }

        // Parse length modifier
        var length: u8 = LEN_NONE;
        switch (format[fi]) {
            'h' => {
                fi += 1;
                if (format[fi] == 'h') {
                    length = LEN_HH;
                    fi += 1;
                } else {
                    length = LEN_H;
                }
            },
            'l' => {
                fi += 1;
                if (format[fi] == 'l') {
                    length = LEN_LL;
                    fi += 1;
                } else {
                    length = LEN_L;
                }
            },
            else => {},
        }

        const spec = format[fi];
        switch (spec) {
            'd', 'i' => {
                // Skip leading whitespace
                while (true) {
                    const wc = reader.peekWchar() catch break;
                    if (iswspace(wc) == 0) break;
                    _ = reader.readWchar() catch break;
                }

                var num: i64 = 0;
                var negative = false;
                var count: usize = 0;
                const max_width = width orelse 64;

                // Check for sign
                const first = reader.peekWchar() catch return matched;
                if (first == '-') {
                    negative = true;
                    _ = reader.readWchar() catch {};
                    count += 1;
                } else if (first == '+') {
                    _ = reader.readWchar() catch {};
                    count += 1;
                }

                // Read digits
                var has_digits = false;
                while (count < max_width) {
                    const wc = reader.peekWchar() catch break;
                    if (wc < '0' or wc > '9') break;
                    _ = reader.readWchar() catch break;
                    num = num * 10 + @as(i64, @intCast(wc - '0'));
                    count += 1;
                    has_digits = true;
                }

                if (!has_digits) return matched;

                if (negative) num = -num;
                if (!suppress) {
                    const ptr = @cVaArg(&va_list, *c_int);
                    ptr.* = @intCast(num);
                    matched += 1;
                }
            },
            'u' => {
                while (true) {
                    const wc = reader.peekWchar() catch break;
                    if (iswspace(wc) == 0) break;
                    _ = reader.readWchar() catch break;
                }

                var num: u64 = 0;
                var count: usize = 0;
                const max_width = width orelse 64;
                var has_digits = false;

                while (count < max_width) {
                    const wc = reader.peekWchar() catch break;
                    if (wc < '0' or wc > '9') break;
                    _ = reader.readWchar() catch break;
                    num = num * 10 + @as(u64, @intCast(wc - '0'));
                    count += 1;
                    has_digits = true;
                }

                if (!has_digits) return matched;

                if (!suppress) {
                    const ptr = @cVaArg(&va_list, *c_uint);
                    ptr.* = @intCast(num);
                    matched += 1;
                }
            },
            'x', 'X' => {
                while (true) {
                    const wc = reader.peekWchar() catch break;
                    if (iswspace(wc) == 0) break;
                    _ = reader.readWchar() catch break;
                }

                var num: u64 = 0;
                var count: usize = 0;
                const max_width = width orelse 64;
                var has_digits = false;

                // Skip optional 0x prefix
                const first = reader.peekWchar() catch return matched;
                if (first == '0') {
                    _ = reader.readWchar() catch {};
                    count += 1;
                    const second = reader.peekWchar() catch {
                        if (!suppress) {
                            const ptr = @cVaArg(&va_list, *c_uint);
                            ptr.* = 0;
                            matched += 1;
                        }
                        continue;
                    };
                    if (second == 'x' or second == 'X') {
                        _ = reader.readWchar() catch {};
                        count += 1;
                    } else if (iswxdigit(second) != 0) {
                        // It was a digit, include it
                        has_digits = true;
                    }
                }

                while (count < max_width) {
                    const wc = reader.peekWchar() catch break;
                    const digit: u64 = if (wc >= '0' and wc <= '9')
                        wc - '0'
                    else if (wc >= 'A' and wc <= 'F')
                        wc - 'A' + 10
                    else if (wc >= 'a' and wc <= 'f')
                        wc - 'a' + 10
                    else
                        break;
                    _ = reader.readWchar() catch break;
                    num = num * 16 + digit;
                    count += 1;
                    has_digits = true;
                }

                if (!has_digits) return matched;

                if (!suppress) {
                    const ptr = @cVaArg(&va_list, *c_uint);
                    ptr.* = @intCast(num);
                    matched += 1;
                }
            },
            'c' => {
                if (length == LEN_L) {
                    // %lc - read wide char
                    const wc = reader.readWchar() catch return matched;
                    if (!suppress) {
                        const ptr = @cVaArg(&va_list, *wchar_t);
                        ptr.* = @intCast(wc);
                        matched += 1;
                    }
                } else {
                    // %c - read narrow char
                    const wc = reader.readWchar() catch return matched;
                    if (!suppress) {
                        const ptr = @cVaArg(&va_list, *u8);
                        ptr.* = @intCast(wc & 0xFF);
                        matched += 1;
                    }
                }
            },
            's' => {
                // Skip leading whitespace
                while (true) {
                    const wc = reader.peekWchar() catch break;
                    if (iswspace(wc) == 0) break;
                    _ = reader.readWchar() catch break;
                }

                if (length == LEN_L) {
                    // %ls - read wide string
                    const ptr = if (!suppress) @cVaArg(&va_list, [*]wchar_t) else undefined;
                    var count: usize = 0;
                    const max_width = width orelse 1024;

                    while (count < max_width) {
                        const wc = reader.peekWchar() catch break;
                        if (iswspace(wc) != 0) break;
                        _ = reader.readWchar() catch break;
                        if (!suppress) {
                            ptr[count] = @intCast(wc);
                        }
                        count += 1;
                    }

                    if (!suppress) {
                        ptr[count] = 0;
                    }
                    if (count == 0) return matched;
                    if (!suppress) matched += 1;
                } else {
                    // %s - read narrow string
                    const ptr = if (!suppress) @cVaArg(&va_list, [*]u8) else undefined;
                    var count: usize = 0;
                    const max_width = width orelse 1024;

                    while (count < max_width) {
                        const wc = reader.peekWchar() catch break;
                        if (iswspace(wc) != 0) break;
                        _ = reader.readWchar() catch break;
                        if (!suppress) {
                            ptr[count] = @intCast(wc & 0xFF);
                        }
                        count += 1;
                    }

                    if (!suppress) {
                        ptr[count] = 0;
                    }
                    if (count == 0) return matched;
                    if (!suppress) matched += 1;
                }
            },
            else => {},
        }
    }

    return matched;
}

/// Wide character scanf
pub export fn wscanf(format: [*:0]const wchar_t, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    return vwscanf(format, args);
}

/// Wide character fscanf
pub export fn fwscanf(stream: *stdio.FILE, format: [*:0]const wchar_t, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    return vfwscanf(stream, format, args);
}

/// Wide character swscanf
pub export fn swscanf(ws: [*:0]const wchar_t, format: [*:0]const wchar_t, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    return vswscanf(ws, format, args);
}

/// Wide character vwscanf
pub export fn vwscanf(format: [*:0]const wchar_t, args: std.builtin.VaList) c_int {
    var reader = WideFileReader{ .file = stdio.stdin };
    return wideScanFormat(&reader, format, args);
}

/// Wide character vfwscanf
pub export fn vfwscanf(stream: *stdio.FILE, format: [*:0]const wchar_t, args: std.builtin.VaList) c_int {
    var reader = WideFileReader{ .file = stream };
    return wideScanFormat(&reader, format, args);
}

/// Wide character vswscanf
pub export fn vswscanf(ws: [*:0]const wchar_t, format: [*:0]const wchar_t, args: std.builtin.VaList) c_int {
    var reader = WideStringReader{ .str = ws, .pos = 0 };
    return wideScanFormat(&reader, format, args);
}

// A. Wide String Functions (20 functions)
pub export fn wcslen(s: [*:0]const wchar_t) usize {
    var len: usize = 0;
    while (s[len] != 0) : (len += 1) {}
    return len;
}

pub export fn wcscpy(dest: [*:0]wchar_t, src: [*:0]const wchar_t) [*:0]wchar_t {
    var i: usize = 0;
    while (src[i] != 0) : (i += 1) {
        dest[i] = src[i];
    }
    dest[i] = 0;
    return dest;
}

pub export fn wcsncpy(dest: [*:0]wchar_t, src: [*:0]const wchar_t, n: usize) [*:0]wchar_t {
    var i: usize = 0;
    while (i < n and src[i] != 0) : (i += 1) {
        dest[i] = src[i];
    }
    while (i < n) : (i += 1) {
        dest[i] = 0;
    }
    return dest;
}

pub export fn wcscat(dest: [*:0]wchar_t, src: [*:0]const wchar_t) [*:0]wchar_t {
    const dest_len = wcslen(dest);
    _ = wcscpy(dest + dest_len, src);
    return dest;
}

pub export fn wcsncat(dest: [*:0]wchar_t, src: [*:0]const wchar_t, n: usize) [*:0]wchar_t {
    const dest_len = wcslen(dest);
    var i: usize = 0;
    while (i < n and src[i] != 0) : (i += 1) {
        dest[dest_len + i] = src[i];
    }
    dest[dest_len + i] = 0;
    return dest;
}

pub export fn wcscmp(s1: [*:0]const wchar_t, s2: [*:0]const wchar_t) c_int {
    var i: usize = 0;
    while (s1[i] != 0 and s1[i] == s2[i]) : (i += 1) {}
    return if (s1[i] < s2[i]) -1 else if (s1[i] > s2[i]) 1 else 0;
}

pub export fn wcsncmp(s1: [*:0]const wchar_t, s2: [*:0]const wchar_t, n: usize) c_int {
    var i: usize = 0;
    while (i < n and s1[i] != 0 and s1[i] == s2[i]) : (i += 1) {}
    if (i >= n) return 0;
    return if (s1[i] < s2[i]) -1 else if (s1[i] > s2[i]) 1 else 0;
}

pub export fn wcschr(s: [*:0]const wchar_t, c: wchar_t) ?[*:0]const wchar_t {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (s[i] == c) return s + i;
    }
    return if (c == 0) s + i else null;
}

pub export fn wcsrchr(s: [*:0]const wchar_t, c: wchar_t) ?[*:0]const wchar_t {
    const len = wcslen(s);
    var i: usize = len;
    while (i > 0) {
        i -= 1;
        if (s[i] == c) return s + i;
    }
    return if (c == 0) s + len else null;
}

pub export fn wcsstr(haystack: [*:0]const wchar_t, needle: [*:0]const wchar_t) ?[*:0]const wchar_t {
    const needle_len = wcslen(needle);
    if (needle_len == 0) return haystack;
    
    var i: usize = 0;
    while (haystack[i] != 0) : (i += 1) {
        if (wcsncmp(haystack + i, needle, needle_len) == 0) {
            return haystack + i;
        }
    }
    return null;
}

pub export fn wcsspn(s: [*:0]const wchar_t, accept: [*:0]const wchar_t) usize {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (wcschr(accept, s[i]) == null) break;
    }
    return i;
}

pub export fn wcscspn(s: [*:0]const wchar_t, reject: [*:0]const wchar_t) usize {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (wcschr(reject, s[i]) != null) break;
    }
    return i;
}

pub export fn wcspbrk(s: [*:0]const wchar_t, accept: [*:0]const wchar_t) ?[*:0]const wchar_t {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (wcschr(accept, s[i]) != null) return s + i;
    }
    return null;
}

pub export fn wcstok(s: ?[*:0]wchar_t, delim: [*:0]const wchar_t, ptr: *?[*:0]wchar_t) ?[*:0]wchar_t {
    var str = s orelse ptr.* orelse return null;
    
    // Skip leading delimiters
    while (str[0] != 0 and wcschr(delim, str[0]) != null) {
        str += 1;
    }
    
    if (str[0] == 0) {
        ptr.* = null;
        return null;
    }
    
    const start = str;
    while (str[0] != 0 and wcschr(delim, str[0]) == null) {
        str += 1;
    }
    
    if (str[0] != 0) {
        str[0] = 0;
        ptr.* = str + 1;
    } else {
        ptr.* = null;
    }
    
    return start;
}

// B. Wide Character Classification (14 functions)
pub export fn iswalnum(wc: wint_t) c_int {
    return if ((wc >= 'A' and wc <= 'Z') or (wc >= 'a' and wc <= 'z') or (wc >= '0' and wc <= '9')) 1 else 0;
}

pub export fn iswalpha(wc: wint_t) c_int {
    return if ((wc >= 'A' and wc <= 'Z') or (wc >= 'a' and wc <= 'z')) 1 else 0;
}

pub export fn iswblank(wc: wint_t) c_int {
    return if (wc == ' ' or wc == '\t') 1 else 0;
}

pub export fn iswcntrl(wc: wint_t) c_int {
    return if (wc < 32 or wc == 127) 1 else 0;
}

pub export fn iswdigit(wc: wint_t) c_int {
    return if (wc >= '0' and wc <= '9') 1 else 0;
}

pub export fn iswgraph(wc: wint_t) c_int {
    return if (wc > 32 and wc < 127) 1 else 0;
}

pub export fn iswlower(wc: wint_t) c_int {
    return if (wc >= 'a' and wc <= 'z') 1 else 0;
}

pub export fn iswprint(wc: wint_t) c_int {
    return if (wc >= 32 and wc < 127) 1 else 0;
}

pub export fn iswpunct(wc: wint_t) c_int {
    const is_alnum = iswalnum(wc);
    const is_space = (wc == ' ');
    return if (iswgraph(wc) != 0 and is_alnum == 0 and !is_space) 1 else 0;
}

pub export fn iswspace(wc: wint_t) c_int {
    return if (wc == ' ' or wc == '\t' or wc == '\n' or wc == '\r' or wc == '\x0b' or wc == '\x0c') 1 else 0;
}

pub export fn iswupper(wc: wint_t) c_int {
    return if (wc >= 'A' and wc <= 'Z') 1 else 0;
}

pub export fn iswxdigit(wc: wint_t) c_int {
    return if ((wc >= '0' and wc <= '9') or (wc >= 'A' and wc <= 'F') or (wc >= 'a' and wc <= 'f')) 1 else 0;
}

pub export fn towlower(wc: wint_t) wint_t {
    return if (wc >= 'A' and wc <= 'Z') wc + 32 else wc;
}

pub export fn towupper(wc: wint_t) wint_t {
    return if (wc >= 'a' and wc <= 'z') wc - 32 else wc;
}

// C. Multibyte Conversions (16 functions)
pub export fn mblen(s: ?[*:0]const u8, n: usize) c_int {
    _ = n;
    if (s == null) return 0;
    const str = s.?;
    if (str[0] == 0) return 0;
    return 1; // Simplified: 1 byte per char
}

pub export fn mbtowc(pwc: ?*wchar_t, s: ?[*:0]const u8, n: usize) c_int {
    if (s == null) return 0;
    const str = s.?;
    if (str[0] == 0) return 0;
    if (n == 0) return -1;
    
    if (pwc) |wc| {
        wc.* = str[0];
    }
    return 1;
}

pub export fn wctomb(s: ?[*]u8, wc: wchar_t) c_int {
    if (s == null) return 0;
    const str = s.?;
    str[0] = @intCast(wc & 0xff);
    return 1;
}

pub export fn mbstowcs(dest: [*]wchar_t, src: [*:0]const u8, n: usize) usize {
    var i: usize = 0;
    while (i < n and src[i] != 0) : (i += 1) {
        dest[i] = src[i];
    }
    return i;
}

pub export fn wcstombs(dest: [*]u8, src: [*:0]const wchar_t, n: usize) usize {
    var i: usize = 0;
    while (i < n and src[i] != 0) : (i += 1) {
        dest[i] = @intCast(src[i] & 0xff);
    }
    return i;
}

pub export fn mbrtowc(pwc: ?*wchar_t, s: ?[*]const u8, n: usize, ps: ?*mbstate_t) usize {
    _ = ps;
    if (s == null) return 0;
    const str = s.?;
    if (n == 0) return @bitCast(@as(isize, -2));
    if (str[0] == 0) return 0;
    
    if (pwc) |wc| {
        wc.* = str[0];
    }
    return 1;
}

pub export fn wcrtomb(s: ?[*]u8, wc: wchar_t, ps: ?*mbstate_t) usize {
    _ = ps;
    if (s == null) return 1;
    const str = s.?;
    str[0] = @intCast(wc & 0xff);
    return 1;
}

pub export fn mbsrtowcs(dest: ?[*]wchar_t, src: *?[*]const u8, n: usize, ps: ?*mbstate_t) usize {
    _ = ps;
    const source = src.* orelse return 0;
    var i: usize = 0;
    
    while ((dest == null or i < n) and source[i] != 0) : (i += 1) {
        if (dest) |d| {
            d[i] = source[i];
        }
    }
    
    if (source[i] == 0) {
        src.* = null;
    } else {
        src.* = source + i;
    }
    
    return i;
}

pub export fn wcsrtombs(dest: ?[*]u8, src: *?[*]const wchar_t, n: usize, ps: ?*mbstate_t) usize {
    _ = ps;
    const source = src.* orelse return 0;
    var i: usize = 0;
    
    while ((dest == null or i < n) and source[i] != 0) : (i += 1) {
        if (dest) |d| {
            d[i] = @intCast(source[i] & 0xff);
        }
    }
    
    if (source[i] == 0) {
        src.* = null;
    } else {
        src.* = source + i;
    }
    
    return i;
}

pub export fn mbsinit(ps: ?*const mbstate_t) c_int {
    _ = ps;
    return 1; // Always in initial state (simplified)
}

// D. Wide String Numeric Conversions (10 functions)
pub export fn wcstol(nptr: [*:0]const wchar_t, endptr: ?*?[*:0]const wchar_t, base: c_int) c_long {
    _ = base; // Simplified: assume base 10
    var result: c_long = 0;
    var i: usize = 0;
    var sign: c_long = 1;
    
    while (iswspace(@intCast(nptr[i])) != 0) : (i += 1) {}
    
    if (nptr[i] == '-') {
        sign = -1;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    while (iswdigit(@intCast(nptr[i])) != 0) : (i += 1) {
        result = result * 10 + (nptr[i] - '0');
    }
    
    if (endptr) |ptr| {
        ptr.* = nptr + i;
    }
    
    return result * sign;
}

pub export fn wcstoll(nptr: [*:0]const wchar_t, endptr: ?*?[*:0]const wchar_t, base: c_int) c_longlong {
    return wcstol(nptr, endptr, base);
}

pub export fn wcstoul(nptr: [*:0]const wchar_t, endptr: ?*?[*:0]const wchar_t, base: c_int) c_ulong {
    return @intCast(wcstol(nptr, endptr, base));
}

pub export fn wcstoull(nptr: [*:0]const wchar_t, endptr: ?*?[*:0]const wchar_t, base: c_int) c_ulonglong {
    return @intCast(wcstol(nptr, endptr, base));
}

pub export fn wcstod(nptr: [*:0]const wchar_t, endptr: ?*?[*:0]const wchar_t) f64 {
    _ = endptr;
    const result: f64 = @floatFromInt(wcstol(nptr, null, 10));
    return result;
}

pub export fn wcstof(nptr: [*:0]const wchar_t, endptr: ?*?[*:0]const wchar_t) f32 {
    return @floatCast(wcstod(nptr, endptr));
}

pub export fn wcstold(nptr: [*:0]const wchar_t, endptr: ?*?[*:0]const wchar_t) c_longdouble {
    return wcstod(nptr, endptr);
}

// Additional helpers (7 functions)
pub export fn wcsdup(s: [*:0]const wchar_t) ?[*:0]wchar_t {
    const len = wcslen(s);
    const new_str = @as(?[*]wchar_t, @ptrFromInt(@intFromPtr(std.c.malloc((len + 1) * @sizeOf(wchar_t)))));
    if (new_str == null) return null;
    return wcscpy(@ptrCast(new_str), s);
}

pub export fn wcsnlen(s: [*:0]const wchar_t, maxlen: usize) usize {
    var len: usize = 0;
    while (len < maxlen and s[len] != 0) : (len += 1) {}
    return len;
}

pub export fn wmemcpy(dest: [*]wchar_t, src: [*]const wchar_t, n: usize) [*]wchar_t {
    for (0..n) |i| {
        dest[i] = src[i];
    }
    return dest;
}

pub export fn wmemmove(dest: [*]wchar_t, src: [*]const wchar_t, n: usize) [*]wchar_t {
    if (@intFromPtr(dest) < @intFromPtr(src)) {
        for (0..n) |i| {
            dest[i] = src[i];
        }
    } else {
        var i = n;
        while (i > 0) {
            i -= 1;
            dest[i] = src[i];
        }
    }
    return dest;
}

pub export fn wmemset(s: [*]wchar_t, c: wchar_t, n: usize) [*]wchar_t {
    for (0..n) |i| {
        s[i] = c;
    }
    return s;
}

pub export fn wmemcmp(s1: [*]const wchar_t, s2: [*]const wchar_t, n: usize) c_int {
    for (0..n) |i| {
        if (s1[i] != s2[i]) {
            return if (s1[i] < s2[i]) -1 else 1;
        }
    }
    return 0;
}

pub export fn wmemchr(s: [*]const wchar_t, c: wchar_t, n: usize) ?[*]const wchar_t {
    for (0..n) |i| {
        if (s[i] == c) return s + i;
    }
    return null;
}

// E. Single-byte/Wide character conversions

/// Convert single byte to wide character
pub export fn btowc(c: c_int) wint_t {
    if (c == stdio.EOF) return WEOF;
    if (c < 0 or c > 127) return WEOF; // Only ASCII is single-byte safe
    return @intCast(@as(u32, @bitCast(c)));
}

/// Convert wide character to single byte
pub export fn wctob(wc: wint_t) c_int {
    if (wc == WEOF) return stdio.EOF;
    if (wc > 127) return stdio.EOF; // Not a single-byte character
    return @intCast(wc);
}

// F. Wide character type functions (POSIX)
pub const wctrans_t = usize;

/// Get character transformation descriptor
pub export fn wctrans(property: [*:0]const u8) wctrans_t {
    // Check for known transformations
    if (property[0] == 't' and property[1] == 'o') {
        if (property[2] == 'l' and property[3] == 'o' and property[4] == 'w' and property[5] == 'e' and property[6] == 'r' and property[7] == 0) {
            return 1; // tolower
        }
        if (property[2] == 'u' and property[3] == 'p' and property[4] == 'p' and property[5] == 'e' and property[6] == 'r' and property[7] == 0) {
            return 2; // toupper
        }
    }
    return 0;
}

/// Transform wide character according to descriptor
pub export fn towctrans(wc: wint_t, desc: wctrans_t) wint_t {
    return switch (desc) {
        1 => towlower(wc), // tolower
        2 => towupper(wc), // toupper
        else => wc,
    };
}

/// Get wide character type descriptor
pub export fn wctype(property: [*:0]const u8) wctype_t {
    // Map property names to type codes
    const props = [_]struct { name: []const u8, code: wctype_t }{
        .{ .name = "alnum", .code = 1 },
        .{ .name = "alpha", .code = 2 },
        .{ .name = "blank", .code = 3 },
        .{ .name = "cntrl", .code = 4 },
        .{ .name = "digit", .code = 5 },
        .{ .name = "graph", .code = 6 },
        .{ .name = "lower", .code = 7 },
        .{ .name = "print", .code = 8 },
        .{ .name = "punct", .code = 9 },
        .{ .name = "space", .code = 10 },
        .{ .name = "upper", .code = 11 },
        .{ .name = "xdigit", .code = 12 },
    };

    for (props) |p| {
        var i: usize = 0;
        var match = true;
        while (i < p.name.len) : (i += 1) {
            if (property[i] != p.name[i]) {
                match = false;
                break;
            }
        }
        if (match and property[p.name.len] == 0) return p.code;
    }
    return 0;
}

/// Check if wide character belongs to the given type
pub export fn iswctype(wc: wint_t, desc: wctype_t) c_int {
    return switch (desc) {
        1 => iswalnum(wc),
        2 => iswalpha(wc),
        3 => iswblank(wc),
        4 => iswcntrl(wc),
        5 => iswdigit(wc),
        6 => iswgraph(wc),
        7 => iswlower(wc),
        8 => iswprint(wc),
        9 => iswpunct(wc),
        10 => iswspace(wc),
        11 => iswupper(wc),
        12 => iswxdigit(wc),
        else => 0,
    };
}

// G. Wide string collation

/// Compare wide strings using current locale collation
pub export fn wcscoll(s1: [*:0]const wchar_t, s2: [*:0]const wchar_t) c_int {
    // Simplified: just use wcscmp (no locale-specific collation)
    return wcscmp(s1, s2);
}

/// Transform wide string for collation comparison
pub export fn wcsxfrm(dest: ?[*]wchar_t, src: [*:0]const wchar_t, n: usize) usize {
    const len = wcslen(src);
    if (dest) |d| {
        var i: usize = 0;
        while (i < n and i < len) : (i += 1) {
            d[i] = src[i];
        }
        if (i < n) d[i] = 0;
    }
    return len;
}

// H. Wide string time formatting

// Day and month name arrays for wcsftime
const wday_names = [_][:0]const u8{
    "Sunday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday",
};

const wday_abbr = [_][:0]const u8{
    "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
};

const month_names = [_][:0]const u8{
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
};

const month_abbr = [_][:0]const u8{
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
};

/// Helper to write a narrow string to wide buffer
fn writeNarrowToWide(wcs: [*]wchar_t, pos: *usize, maxsize: usize, str: []const u8) void {
    for (str) |c| {
        if (pos.* >= maxsize - 1) return;
        wcs[pos.*] = @intCast(c);
        pos.* += 1;
    }
}

/// Helper to write a 2-digit number to wide buffer
fn write2DigitWide(wcs: [*]wchar_t, pos: *usize, maxsize: usize, val: c_int) void {
    if (pos.* >= maxsize - 1) return;
    const v: u32 = if (val >= 0) @intCast(val) else 0;
    wcs[pos.*] = @intCast('0' + @divTrunc(v, 10) % 10);
    pos.* += 1;
    if (pos.* >= maxsize - 1) return;
    wcs[pos.*] = @intCast('0' + @mod(v, 10));
    pos.* += 1;
}

/// Helper to write a 4-digit year to wide buffer
fn write4DigitWide(wcs: [*]wchar_t, pos: *usize, maxsize: usize, year: c_int) void {
    const y: u32 = if (year >= 0) @intCast(year + 1900) else @intCast(1900 - @as(u32, @intCast(-year)));
    if (pos.* >= maxsize - 1) return;
    wcs[pos.*] = @intCast('0' + @divTrunc(y, 1000) % 10);
    pos.* += 1;
    if (pos.* >= maxsize - 1) return;
    wcs[pos.*] = @intCast('0' + @divTrunc(y, 100) % 10);
    pos.* += 1;
    if (pos.* >= maxsize - 1) return;
    wcs[pos.*] = @intCast('0' + @divTrunc(y, 10) % 10);
    pos.* += 1;
    if (pos.* >= maxsize - 1) return;
    wcs[pos.*] = @intCast('0' + @mod(y, 10));
    pos.* += 1;
}

/// Format time as wide string
pub export fn wcsftime(wcs: [*]wchar_t, maxsize: usize, format: [*:0]const wchar_t, timeptr: ?*const time_mod.tm) usize {
    if (maxsize == 0) return 0;
    if (timeptr == null) return 0;

    const tm = timeptr.?;
    var written: usize = 0;
    var i: usize = 0;

    while (format[i] != 0 and written < maxsize - 1) : (i += 1) {
        if (format[i] != '%') {
            wcs[written] = format[i];
            written += 1;
            continue;
        }

        i += 1;
        if (format[i] == 0) break;

        switch (format[i]) {
            'Y' => {
                // 4-digit year
                write4DigitWide(wcs, &written, maxsize, tm.tm_year);
            },
            'y' => {
                // 2-digit year
                const y = @mod(tm.tm_year, 100);
                write2DigitWide(wcs, &written, maxsize, @intCast(y));
            },
            'm' => {
                // Month 01-12
                write2DigitWide(wcs, &written, maxsize, tm.tm_mon + 1);
            },
            'd' => {
                // Day of month 01-31
                write2DigitWide(wcs, &written, maxsize, tm.tm_mday);
            },
            'H' => {
                // Hour 00-23
                write2DigitWide(wcs, &written, maxsize, tm.tm_hour);
            },
            'M' => {
                // Minute 00-59
                write2DigitWide(wcs, &written, maxsize, tm.tm_min);
            },
            'S' => {
                // Second 00-60
                write2DigitWide(wcs, &written, maxsize, tm.tm_sec);
            },
            'A' => {
                // Full weekday name
                const wday: usize = @intCast(@mod(tm.tm_wday, 7));
                writeNarrowToWide(wcs, &written, maxsize, wday_names[wday]);
            },
            'a' => {
                // Abbreviated weekday name
                const wday: usize = @intCast(@mod(tm.tm_wday, 7));
                writeNarrowToWide(wcs, &written, maxsize, wday_abbr[wday]);
            },
            'B' => {
                // Full month name
                const mon: usize = @intCast(@mod(tm.tm_mon, 12));
                writeNarrowToWide(wcs, &written, maxsize, month_names[mon]);
            },
            'b', 'h' => {
                // Abbreviated month name
                const mon: usize = @intCast(@mod(tm.tm_mon, 12));
                writeNarrowToWide(wcs, &written, maxsize, month_abbr[mon]);
            },
            'j' => {
                // Day of year 001-366
                const yday = tm.tm_yday + 1;
                if (written < maxsize - 1) {
                    wcs[written] = @intCast('0' + @divTrunc(@as(u32, @intCast(yday)), 100) % 10);
                    written += 1;
                }
                write2DigitWide(wcs, &written, maxsize, @intCast(@mod(@as(u32, @intCast(yday)), 100)));
            },
            'w' => {
                // Weekday as number 0-6 (Sunday = 0)
                if (written < maxsize - 1) {
                    wcs[written] = @intCast('0' + @mod(@as(u32, @intCast(tm.tm_wday)), 7));
                    written += 1;
                }
            },
            'I' => {
                // Hour 01-12 (12-hour format)
                var hour = @mod(tm.tm_hour, 12);
                if (hour == 0) hour = 12;
                write2DigitWide(wcs, &written, maxsize, hour);
            },
            'p' => {
                // AM/PM
                if (tm.tm_hour < 12) {
                    writeNarrowToWide(wcs, &written, maxsize, "AM");
                } else {
                    writeNarrowToWide(wcs, &written, maxsize, "PM");
                }
            },
            'n' => {
                // Newline
                if (written < maxsize - 1) {
                    wcs[written] = '\n';
                    written += 1;
                }
            },
            't' => {
                // Tab
                if (written < maxsize - 1) {
                    wcs[written] = '\t';
                    written += 1;
                }
            },
            '%' => {
                // Literal %
                if (written < maxsize - 1) {
                    wcs[written] = '%';
                    written += 1;
                }
            },
            else => {
                // Unknown specifier, just output it
                if (written < maxsize - 1) {
                    wcs[written] = format[i];
                    written += 1;
                }
            },
        }
    }

    wcs[written] = 0;
    return written;
}
