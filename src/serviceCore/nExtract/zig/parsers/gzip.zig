// GZIP Parser - RFC 1952 Implementation
// Day 14: GZIP Format Support

const std = @import("std");
const deflate = @import("deflate.zig");
const Allocator = std.mem.Allocator;

/// GZIP header magic bytes
pub const GZIP_MAGIC1: u8 = 0x1f;
pub const GZIP_MAGIC2: u8 = 0x8b;

/// Compression methods
pub const CompressionMethod = enum(u8) {
    deflate = 8,
    _,
};

/// GZIP header flags
pub const Flags = packed struct {
    ftext: bool,      // Bit 0: Text file hint
    fhcrc: bool,      // Bit 1: Header CRC16 present
    fextra: bool,     // Bit 2: Extra fields present
    fname: bool,      // Bit 3: Original filename present
    fcomment: bool,   // Bit 4: File comment present
    reserved: u3,     // Bits 5-7: Reserved (must be zero)
};

/// GZIP header structure
pub const Header = struct {
    magic1: u8,
    magic2: u8,
    compression_method: u8,
    flags: Flags,
    mtime: u32,           // Modification time (Unix timestamp)
    extra_flags: u8,      // XFL - extra flags
    os: u8,               // Operating system
    extra_field: ?[]const u8,
    filename: ?[]const u8,
    comment: ?[]const u8,
    header_crc16: ?u16,
};

/// GZIP footer structure
pub const Footer = struct {
    crc32: u32,          // CRC32 of uncompressed data
    isize: u32,          // Size of uncompressed data modulo 2^32
};

/// GZIP decompression result
pub const DecompressResult = struct {
    data: []u8,
    header: Header,
    allocator: Allocator,

    pub fn deinit(self: *DecompressResult) void {
        self.allocator.free(self.data);
        if (self.header.extra_field) |extra| {
            self.allocator.free(extra);
        }
        if (self.header.filename) |name| {
            self.allocator.free(name);
        }
        if (self.header.comment) |comment| {
            self.allocator.free(comment);
        }
    }
};

/// GZIP parser errors
pub const GzipError = error{
    InvalidMagic,
    UnsupportedCompressionMethod,
    InvalidHeader,
    InvalidFooter,
    CrcMismatch,
    SizeMismatch,
    DecompressionFailed,
};

/// Parse GZIP header from data
pub fn parseHeader(data: []const u8, allocator: Allocator) !struct { header: Header, offset: usize } {
    if (data.len < 10) return error.InvalidHeader;

    // Check magic bytes
    if (data[0] != GZIP_MAGIC1 or data[1] != GZIP_MAGIC2) {
        return error.InvalidMagic;
    }

    // Parse basic header fields
    const compression_method = data[2];
    if (compression_method != @intFromEnum(CompressionMethod.deflate)) {
        return error.UnsupportedCompressionMethod;
    }

    const flags_byte = data[3];
    const flags = @as(*const Flags, @ptrCast(&flags_byte)).*;
    
    const mtime = std.mem.readInt(u32, data[4..8], .little);
    const extra_flags = data[8];
    const os = data[9];

    var offset: usize = 10;

    // Parse extra field if present
    var extra_field: ?[]const u8 = null;
    if (flags.fextra) {
        if (data.len < offset + 2) return error.InvalidHeader;
        const xlen = std.mem.readInt(u16, data[offset..][0..2], .little);
        offset += 2;
        
        if (data.len < offset + xlen) return error.InvalidHeader;
        extra_field = try allocator.dupe(u8, data[offset..][0..xlen]);
        offset += xlen;
    }

    // Parse filename if present
    var filename: ?[]const u8 = null;
    if (flags.fname) {
        const start = offset;
        while (offset < data.len and data[offset] != 0) : (offset += 1) {}
        if (offset >= data.len) return error.InvalidHeader;
        filename = try allocator.dupe(u8, data[start..offset]);
        offset += 1; // Skip null terminator
    }

    // Parse comment if present
    var comment: ?[]const u8 = null;
    if (flags.fcomment) {
        const start = offset;
        while (offset < data.len and data[offset] != 0) : (offset += 1) {}
        if (offset >= data.len) return error.InvalidHeader;
        comment = try allocator.dupe(u8, data[start..offset]);
        offset += 1; // Skip null terminator
    }

    // Parse header CRC16 if present
    var header_crc16: ?u16 = null;
    if (flags.fhcrc) {
        if (data.len < offset + 2) return error.InvalidHeader;
        header_crc16 = std.mem.readInt(u16, data[offset..][0..2], .little);
        offset += 2;

        // Verify header CRC16
        const calculated_crc = crc16(data[0..offset - 2]);
        if (calculated_crc != header_crc16.?) {
            return error.CrcMismatch;
        }
    }

    const header = Header{
        .magic1 = GZIP_MAGIC1,
        .magic2 = GZIP_MAGIC2,
        .compression_method = compression_method,
        .flags = flags,
        .mtime = mtime,
        .extra_flags = extra_flags,
        .os = os,
        .extra_field = extra_field,
        .filename = filename,
        .comment = comment,
        .header_crc16 = header_crc16,
    };

    return .{ .header = header, .offset = offset };
}

/// Parse GZIP footer from data
pub fn parseFooter(data: []const u8) !Footer {
    if (data.len < 8) return error.InvalidFooter;

    const crc32_value = std.mem.readInt(u32, data[0..4], .little);
    const isize = std.mem.readInt(u32, data[4..8], .little);

    return Footer{
        .crc32 = crc32_value,
        .isize = isize,
    };
}

/// Decompress GZIP data
pub fn decompress(data: []const u8, allocator: Allocator) !DecompressResult {
    // Parse header
    const header_result = try parseHeader(data, allocator);
    const header = header_result.header;
    var offset = header_result.offset;

    // Get compressed data (everything between header and footer)
    if (data.len < offset + 8) return error.InvalidFooter;
    const compressed_data = data[offset .. data.len - 8];

    // Decompress using DEFLATE
    const decompressed = deflate.decompress(compressed_data, allocator) catch {
        return error.DecompressionFailed;
    };
    errdefer allocator.free(decompressed);

    // Parse footer
    const footer = try parseFooter(data[data.len - 8 ..]);

    // Verify CRC32
    const calculated_crc32 = crc32(decompressed);
    if (calculated_crc32 != footer.crc32) {
        allocator.free(decompressed);
        return error.CrcMismatch;
    }

    // Verify size (modulo 2^32)
    const size_mod = @as(u32, @truncate(decompressed.len));
    if (size_mod != footer.isize) {
        allocator.free(decompressed);
        return error.SizeMismatch;
    }

    return DecompressResult{
        .data = decompressed,
        .header = header,
        .allocator = allocator,
    };
}

/// Calculate CRC16 checksum (used for header CRC)
fn crc16(data: []const u8) u16 {
    var crc: u16 = 0;
    for (data) |byte| {
        crc ^= byte;
        var i: u8 = 0;
        while (i < 8) : (i += 1) {
            if (crc & 1 != 0) {
                crc = (crc >> 1) ^ 0xa001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

/// Calculate CRC32 checksum
fn crc32(data: []const u8) u32 {
    const crc32_table = comptime blk: {
        var table: [256]u32 = undefined;
        for (&table, 0..) |*entry, i| {
            var crc: u32 = @intCast(i);
            var j: u8 = 0;
            while (j < 8) : (j += 1) {
                if (crc & 1 != 0) {
                    crc = (crc >> 1) ^ 0xedb88320;
                } else {
                    crc >>= 1;
                }
            }
            entry.* = crc;
        }
        break :blk table;
    };

    var crc: u32 = 0xffffffff;
    for (data) |byte| {
        const index = @as(u8, @truncate((crc ^ byte) & 0xff));
        crc = (crc >> 8) ^ crc32_table[index];
    }
    return crc ^ 0xffffffff;
}

/// Check if data is GZIP format
pub fn isGzip(data: []const u8) bool {
    return data.len >= 2 and data[0] == GZIP_MAGIC1 and data[1] == GZIP_MAGIC2;
}

// Export for FFI
export fn nExtract_GZIP_decompress(
    data: [*]const u8,
    len: usize,
    out_len: *usize,
) ?[*]u8 {
    const allocator = std.heap.c_allocator;
    const input = data[0..len];

    const result = decompress(input, allocator) catch return null;
    out_len.* = result.data.len;
    
    // Transfer ownership to caller
    const output = result.data.ptr;
    return output;
}

export fn nExtract_GZIP_is_gzip(data: [*]const u8, len: usize) bool {
    return isGzip(data[0..len]);
}

export fn nExtract_GZIP_free(data: [*]u8, len: usize) void {
    const allocator = std.heap.c_allocator;
    allocator.free(data[0..len]);
}
