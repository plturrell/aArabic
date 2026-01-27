// iconv module - Phase 1.24
// Character set conversion
const std = @import("std");

pub const iconv_t = ?*anyopaque;

// Encoding types
const Encoding = enum {
    utf8,
    utf16le,
    utf16be,
    utf32le,
    utf32be,
    ascii,
    latin1, // ISO-8859-1
    unknown,
};

// Conversion descriptor
const ConvDesc = struct {
    from: Encoding,
    to: Encoding,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

fn parseEncoding(name: []const u8) Encoding {
    // Normalize to uppercase for comparison
    var upper_buf: [32]u8 = undefined;
    const len = @min(name.len, 31);
    for (name[0..len], 0..) |c, i| {
        upper_buf[i] = std.ascii.toUpper(c);
    }
    const upper = upper_buf[0..len];

    if (std.mem.eql(u8, upper, "UTF-8") or std.mem.eql(u8, upper, "UTF8")) return .utf8;
    if (std.mem.eql(u8, upper, "UTF-16LE") or std.mem.eql(u8, upper, "UTF16LE")) return .utf16le;
    if (std.mem.eql(u8, upper, "UTF-16BE") or std.mem.eql(u8, upper, "UTF16BE")) return .utf16be;
    if (std.mem.eql(u8, upper, "UTF-16") or std.mem.eql(u8, upper, "UTF16")) return .utf16le; // Default to LE
    if (std.mem.eql(u8, upper, "UTF-32LE") or std.mem.eql(u8, upper, "UTF32LE")) return .utf32le;
    if (std.mem.eql(u8, upper, "UTF-32BE") or std.mem.eql(u8, upper, "UTF32BE")) return .utf32be;
    if (std.mem.eql(u8, upper, "UTF-32") or std.mem.eql(u8, upper, "UTF32")) return .utf32le;
    if (std.mem.eql(u8, upper, "ASCII") or std.mem.eql(u8, upper, "US-ASCII")) return .ascii;
    if (std.mem.eql(u8, upper, "ISO-8859-1") or std.mem.eql(u8, upper, "LATIN1") or std.mem.eql(u8, upper, "LATIN-1")) return .latin1;

    return .unknown;
}

/// Open a conversion descriptor
pub export fn iconv_open(tocode: [*:0]const u8, fromcode: [*:0]const u8) iconv_t {
    const allocator = gpa.allocator();

    const from = parseEncoding(std.mem.span(fromcode));
    const to = parseEncoding(std.mem.span(tocode));

    if (from == .unknown or to == .unknown) {
        return null; // EINVAL
    }

    const desc = allocator.create(ConvDesc) catch return null;
    desc.from = from;
    desc.to = to;

    return @ptrCast(desc);
}

/// Perform character set conversion
pub export fn iconv(cd: iconv_t, inbuf: ?*[*]u8, inbytesleft: ?*usize, outbuf: ?*[*]u8, outbytesleft: ?*usize) usize {
    if (cd == null) return @as(usize, @bitCast(@as(isize, -1)));

    const desc: *ConvDesc = @ptrCast(@alignCast(cd));

    // Reset state if inbuf is null
    if (inbuf == null) {
        return 0;
    }

    const in_ptr = inbuf.?;
    const out_ptr = outbuf orelse return @as(usize, @bitCast(@as(isize, -1)));
    const in_left = inbytesleft orelse return @as(usize, @bitCast(@as(isize, -1)));
    const out_left = outbytesleft orelse return @as(usize, @bitCast(@as(isize, -1)));

    var converted: usize = 0;

    // Convert based on from/to encodings
    if (desc.from == desc.to) {
        // Same encoding - just copy
        const copy_len = @min(in_left.*, out_left.*);
        @memcpy(out_ptr.*[0..copy_len], in_ptr.*[0..copy_len]);
        in_ptr.* += copy_len;
        out_ptr.* += copy_len;
        in_left.* -= copy_len;
        out_left.* -= copy_len;
        return 0;
    }

    // UTF-8 to UTF-16LE
    if (desc.from == .utf8 and desc.to == .utf16le) {
        converted = convertUtf8ToUtf16Le(in_ptr, in_left, out_ptr, out_left);
    }
    // UTF-16LE to UTF-8
    else if (desc.from == .utf16le and desc.to == .utf8) {
        converted = convertUtf16LeToUtf8(in_ptr, in_left, out_ptr, out_left);
    }
    // UTF-8 to Latin-1
    else if (desc.from == .utf8 and desc.to == .latin1) {
        converted = convertUtf8ToLatin1(in_ptr, in_left, out_ptr, out_left);
    }
    // Latin-1 to UTF-8
    else if (desc.from == .latin1 and desc.to == .utf8) {
        converted = convertLatin1ToUtf8(in_ptr, in_left, out_ptr, out_left);
    }
    // ASCII to UTF-8 (passthrough for 7-bit)
    else if (desc.from == .ascii and desc.to == .utf8) {
        const copy_len = @min(in_left.*, out_left.*);
        @memcpy(out_ptr.*[0..copy_len], in_ptr.*[0..copy_len]);
        in_ptr.* += copy_len;
        out_ptr.* += copy_len;
        in_left.* -= copy_len;
        out_left.* -= copy_len;
    }
    // Unsupported conversion
    else {
        return @as(usize, @bitCast(@as(isize, -1)));
    }

    return converted;
}

fn convertUtf8ToUtf16Le(in_ptr: *[*]u8, in_left: *usize, out_ptr: *[*]u8, out_left: *usize) usize {
    var conversions: usize = 0;

    while (in_left.* > 0 and out_left.* >= 2) {
        const in_slice = in_ptr.*[0..in_left.*];
        const len = std.unicode.utf8ByteSequenceLength(in_slice[0]) catch break;

        if (len > in_left.*) break; // Incomplete sequence

        const codepoint = std.unicode.utf8Decode(in_slice[0..len]) catch break;

        // Encode as UTF-16LE
        if (codepoint <= 0xFFFF) {
            if (out_left.* < 2) break;
            out_ptr.*[0] = @intCast(codepoint & 0xFF);
            out_ptr.*[1] = @intCast((codepoint >> 8) & 0xFF);
            out_ptr.* += 2;
            out_left.* -= 2;
        } else {
            // Surrogate pair
            if (out_left.* < 4) break;
            const cp = codepoint - 0x10000;
            const high: u16 = @intCast(0xD800 + (cp >> 10));
            const low: u16 = @intCast(0xDC00 + (cp & 0x3FF));
            out_ptr.*[0] = @intCast(high & 0xFF);
            out_ptr.*[1] = @intCast(high >> 8);
            out_ptr.*[2] = @intCast(low & 0xFF);
            out_ptr.*[3] = @intCast(low >> 8);
            out_ptr.* += 4;
            out_left.* -= 4;
        }

        in_ptr.* += len;
        in_left.* -= len;
        conversions += 1;
    }

    return conversions;
}

fn convertUtf16LeToUtf8(in_ptr: *[*]u8, in_left: *usize, out_ptr: *[*]u8, out_left: *usize) usize {
    var conversions: usize = 0;

    while (in_left.* >= 2) {
        const low = in_ptr.*[0];
        const high = in_ptr.*[1];
        var codepoint: u21 = @as(u16, low) | (@as(u16, high) << 8);

        var consumed: usize = 2;

        // Check for surrogate pair
        if (codepoint >= 0xD800 and codepoint <= 0xDBFF) {
            if (in_left.* < 4) break;
            const low2 = in_ptr.*[2];
            const high2 = in_ptr.*[3];
            const low_surrogate: u16 = @as(u16, low2) | (@as(u16, high2) << 8);
            codepoint = 0x10000 + ((@as(u21, codepoint) - 0xD800) << 10) + (low_surrogate - 0xDC00);
            consumed = 4;
        }

        // Encode as UTF-8
        const len = std.unicode.utf8CodepointSequenceLength(codepoint) catch break;
        if (out_left.* < len) break;

        var utf8_buf: [4]u8 = undefined;
        _ = std.unicode.utf8Encode(codepoint, &utf8_buf) catch break;
        @memcpy(out_ptr.*[0..len], utf8_buf[0..len]);

        out_ptr.* += len;
        out_left.* -= len;
        in_ptr.* += consumed;
        in_left.* -= consumed;
        conversions += 1;
    }

    return conversions;
}

fn convertUtf8ToLatin1(in_ptr: *[*]u8, in_left: *usize, out_ptr: *[*]u8, out_left: *usize) usize {
    var conversions: usize = 0;

    while (in_left.* > 0 and out_left.* > 0) {
        const in_slice = in_ptr.*[0..in_left.*];
        const len = std.unicode.utf8ByteSequenceLength(in_slice[0]) catch break;

        if (len > in_left.*) break;

        const codepoint = std.unicode.utf8Decode(in_slice[0..len]) catch break;

        if (codepoint > 0xFF) {
            // Cannot represent in Latin-1 - use replacement char
            out_ptr.*[0] = '?';
        } else {
            out_ptr.*[0] = @intCast(codepoint);
        }

        out_ptr.* += 1;
        out_left.* -= 1;
        in_ptr.* += len;
        in_left.* -= len;
        conversions += 1;
    }

    return conversions;
}

fn convertLatin1ToUtf8(in_ptr: *[*]u8, in_left: *usize, out_ptr: *[*]u8, out_left: *usize) usize {
    var conversions: usize = 0;

    while (in_left.* > 0) {
        const byte = in_ptr.*[0];
        const codepoint: u21 = byte;

        const len = std.unicode.utf8CodepointSequenceLength(codepoint) catch break;
        if (out_left.* < len) break;

        var utf8_buf: [4]u8 = undefined;
        _ = std.unicode.utf8Encode(codepoint, &utf8_buf) catch break;
        @memcpy(out_ptr.*[0..len], utf8_buf[0..len]);

        out_ptr.* += len;
        out_left.* -= len;
        in_ptr.* += 1;
        in_left.* -= 1;
        conversions += 1;
    }

    return conversions;
}

/// Close conversion descriptor
pub export fn iconv_close(cd: iconv_t) c_int {
    if (cd == null) return -1;

    const allocator = gpa.allocator();
    const desc: *ConvDesc = @ptrCast(@alignCast(cd));
    allocator.destroy(desc);

    return 0;
}
