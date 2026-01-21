// ZLIB Parser - RFC 1950 Implementation
// Day 14: ZLIB Format Support

const std = @import("std");
const deflate = @import("deflate.zig");
const Allocator = std.mem.Allocator;

/// ZLIB compression methods
pub const CompressionMethod = enum(u4) {
    deflate = 8,
    _,
};

/// ZLIB compression levels
pub const CompressionLevel = enum(u2) {
    fastest = 0,
    fast = 1,
    default = 2,
    maximum = 3,
};

/// ZLIB header structure
pub const Header = struct {
    cmf: u8,              // Compression Method and Flags
    flg: u8,              // Flags
    
    // Extracted from CMF
    compression_method: CompressionMethod,
    compression_info: u4,  // CINFO: base-2 log of window size - 8
    
    // Extracted from FLG
    fcheck: u5,           // Check bits for CMF and FLG
    fdict: bool,          // Preset dictionary flag
    flevel: CompressionLevel,
    
    // Optional dictionary ID (if FDICT = 1)
    dict_id: ?u32,
};

/// ZLIB decompression result
pub const DecompressResult = struct {
    data: []u8,
    header: Header,
    allocator: Allocator,

    pub fn deinit(self: *DecompressResult) void {
        self.allocator.free(self.data);
    }
};

/// ZLIB parser errors
pub const ZlibError = error{
    InvalidHeader,
    InvalidChecksum,
    UnsupportedCompressionMethod,
    DictionaryRequired,
    InvalidWindowSize,
    DecompressionFailed,
    Adler32Mismatch,
};

/// Parse ZLIB header from data
pub fn parseHeader(data: []const u8) !struct { header: Header, offset: usize } {
    if (data.len < 2) return error.InvalidHeader;

    const cmf = data[0];
    const flg = data[1];

    // Extract fields from CMF
    const compression_method_byte = @as(u4, @truncate(cmf & 0x0f));
    const compression_method = @as(CompressionMethod, @enumFromInt(compression_method_byte));
    const compression_info = @as(u4, @truncate((cmf >> 4) & 0x0f));

    // Validate compression method
    if (compression_method != .deflate) {
        return error.UnsupportedCompressionMethod;
    }

    // Validate window size (CINFO)
    if (compression_info > 7) {
        return error.InvalidWindowSize;
    }

    // Extract fields from FLG
    const fcheck = @as(u5, @truncate(flg & 0x1f));
    const fdict = (flg & 0x20) != 0;
    const flevel_byte = @as(u2, @truncate((flg >> 6) & 0x03));
    const flevel = @as(CompressionLevel, @enumFromInt(flevel_byte));

    // Validate FCHECK
    const check_value = (@as(u16, cmf) * 256 + @as(u16, flg)) % 31;
    if (check_value != 0) {
        return error.InvalidChecksum;
    }

    var offset: usize = 2;
    var dict_id: ?u32 = null;

    // Parse dictionary ID if present
    if (fdict) {
        if (data.len < 6) return error.InvalidHeader;
        dict_id = std.mem.readInt(u32, data[2..6], .big);
        offset = 6;
    }

    const header = Header{
        .cmf = cmf,
        .flg = flg,
        .compression_method = compression_method,
        .compression_info = compression_info,
        .fcheck = fcheck,
        .fdict = fdict,
        .flevel = flevel,
        .dict_id = dict_id,
    };

    return .{ .header = header, .offset = offset };
}

/// Decompress ZLIB data
pub fn decompress(data: []const u8, allocator: Allocator) !DecompressResult {
    // Parse header
    const header_result = try parseHeader(data);
    const header = header_result.header;
    const offset = header_result.offset;

    // Check if dictionary is required (we don't support it for now)
    if (header.fdict) {
        return error.DictionaryRequired;
    }

    // Get compressed data (everything between header and checksum)
    if (data.len < offset + 4) return error.InvalidHeader;
    const compressed_data = data[offset .. data.len - 4];

    // Decompress using DEFLATE
    const decompressed = deflate.decompress(compressed_data, allocator) catch {
        return error.DecompressionFailed;
    };
    errdefer allocator.free(decompressed);

    // Parse Adler32 checksum (last 4 bytes)
    const adler32_value = std.mem.readInt(u32, data[data.len - 4 ..][0..4], .big);

    // Verify Adler32
    const calculated_adler32 = adler32(decompressed);
    if (calculated_adler32 != adler32_value) {
        allocator.free(decompressed);
        return error.Adler32Mismatch;
    }

    return DecompressResult{
        .data = decompressed,
        .header = header,
        .allocator = allocator,
    };
}

/// Calculate Adler32 checksum
fn adler32(data: []const u8) u32 {
    const MOD_ADLER: u32 = 65521;
    
    var a: u32 = 1;
    var b: u32 = 0;
    
    for (data) |byte| {
        a = (a + byte) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    
    return (b << 16) | a;
}

/// Get window size from header
pub fn getWindowSize(header: Header) u32 {
    return @as(u32, 1) << (@as(u5, header.compression_info) + 8);
}

/// Check if data is ZLIB format
pub fn isZlib(data: []const u8) bool {
    if (data.len < 2) return false;
    
    const cmf = data[0];
    const flg = data[1];
    
    // Check compression method (should be 8 for DEFLATE)
    const cm = cmf & 0x0f;
    if (cm != 8) return false;
    
    // Check FCHECK
    const check_value = (@as(u16, cmf) * 256 + @as(u16, flg)) % 31;
    return check_value == 0;
}

// Export for FFI
export fn nExtract_ZLIB_decompress(
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

export fn nExtract_ZLIB_is_zlib(data: [*]const u8, len: usize) bool {
    return isZlib(data[0..len]);
}

export fn nExtract_ZLIB_free(data: [*]u8, len: usize) void {
    const allocator = std.heap.c_allocator;
    allocator.free(data[0..len]);
}

export fn nExtract_ZLIB_get_window_size(cmf: u8) u32 {
    const compression_info = @as(u4, @truncate((cmf >> 4) & 0x0f));
    return @as(u32, 1) << (@as(u5, compression_info) + 8);
}
