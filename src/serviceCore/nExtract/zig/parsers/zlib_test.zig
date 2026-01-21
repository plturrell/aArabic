// ZLIB Parser Tests
// Day 14: Comprehensive test suite for ZLIB functionality

const std = @import("std");
const zlib = @import("zlib.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Helper function to create test ZLIB data
fn createZlibData(allocator: Allocator, uncompressed: []const u8, options: struct {
    compression_info: u4 = 7, // Default: 32KB window (7 + 8 = 15, 2^15 = 32KB)
    compression_level: zlib.CompressionLevel = .default,
    add_dict: bool = false,
}) ![]u8 {
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // CMF byte
    const compression_method: u4 = 8; // DEFLATE
    const cmf = (options.compression_info << 4) | compression_method;
    try data.append(cmf);

    // FLG byte
    var flg: u8 = 0;
    
    // Set compression level
    flg |= (@as(u8, @intFromEnum(options.compression_level)) << 6);
    
    // Set FDICT flag
    if (options.add_dict) {
        flg |= 0x20;
    }
    
    // Calculate FCHECK to make (CMF * 256 + FLG) % 31 == 0
    const base = (@as(u16, cmf) * 256 + @as(u16, flg));
    const remainder = base % 31;
    if (remainder != 0) {
        flg += @as(u8, @truncate(31 - remainder));
    }
    
    try data.append(flg);

    // Add dictionary ID if needed
    if (options.add_dict) {
        // Dictionary Adler32 (just use a test value)
        try data.appendSlice(&[_]u8{ 0x12, 0x34, 0x56, 0x78 });
    }

    // For simplicity, we'll use uncompressed blocks in DEFLATE
    // BFINAL=1, BTYPE=00 (no compression)
    try data.append(0x01); // BFINAL=1, BTYPE=00

    // Length of uncompressed data
    const len = @as(u16, @truncate(uncompressed.len));
    try data.appendSlice(&[_]u8{
        @as(u8, @truncate(len)),
        @as(u8, @truncate(len >> 8)),
    });

    // One's complement of length
    const nlen = ~len;
    try data.appendSlice(&[_]u8{
        @as(u8, @truncate(nlen)),
        @as(u8, @truncate(nlen >> 8)),
    });

    // Uncompressed data
    try data.appendSlice(uncompressed);

    // Adler32 checksum of uncompressed data
    const adler32_value = calculateAdler32(uncompressed);
    try data.appendSlice(&[_]u8{
        @as(u8, @truncate(adler32_value >> 24)),
        @as(u8, @truncate(adler32_value >> 16)),
        @as(u8, @truncate(adler32_value >> 8)),
        @as(u8, @truncate(adler32_value)),
    });

    return data.toOwnedSlice();
}

fn calculateAdler32(data: []const u8) u32 {
    const MOD_ADLER: u32 = 65521;
    
    var a: u32 = 1;
    var b: u32 = 0;
    
    for (data) |byte| {
        a = (a + byte) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    
    return (b << 16) | a;
}

test "ZLIB: Basic decompression" {
    const allocator = testing.allocator;
    const original_data = "Hello, ZLIB World!";

    const zlib_data = try createZlibData(allocator, original_data, .{});
    defer allocator.free(zlib_data);

    var result = try zlib.decompress(zlib_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
    try testing.expectEqual(zlib.CompressionMethod.deflate, result.header.compression_method);
}

test "ZLIB: Different compression levels" {
    const allocator = testing.allocator;
    const original_data = "Test data for compression levels";

    // Test each compression level
    const levels = [_]zlib.CompressionLevel{ .fastest, .fast, .default, .maximum };
    
    for (levels) |level| {
        const zlib_data = try createZlibData(allocator, original_data, .{ .compression_level = level });
        defer allocator.free(zlib_data);

        var result = try zlib.decompress(zlib_data, allocator);
        defer result.deinit();

        try testing.expectEqualSlices(u8, original_data, result.data);
        try testing.expectEqual(level, result.header.flevel);
    }
}

test "ZLIB: Different window sizes" {
    const allocator = testing.allocator;
    const original_data = "Test data for window sizes";

    // Test various window sizes (CINFO from 0 to 7)
    var cinfo: u4 = 0;
    while (cinfo <= 7) : (cinfo += 1) {
        const zlib_data = try createZlibData(allocator, original_data, .{ .compression_info = cinfo });
        defer allocator.free(zlib_data);

        var result = try zlib.decompress(zlib_data, allocator);
        defer result.deinit();

        try testing.expectEqualSlices(u8, original_data, result.data);
        try testing.expectEqual(cinfo, result.header.compression_info);
        
        // Verify window size calculation
        const expected_window_size = @as(u32, 1) << (@as(u5, cinfo) + 8);
        const actual_window_size = zlib.getWindowSize(result.header);
        try testing.expectEqual(expected_window_size, actual_window_size);
    }
}

test "ZLIB: Empty data" {
    const allocator = testing.allocator;
    const original_data = "";

    const zlib_data = try createZlibData(allocator, original_data, .{});
    defer allocator.free(zlib_data);

    var result = try zlib.decompress(zlib_data, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.data.len);
}

test "ZLIB: Large data" {
    const allocator = testing.allocator;

    // Create 10KB of test data
    var original_data = try allocator.alloc(u8, 10240);
    defer allocator.free(original_data);

    // Fill with pattern
    for (original_data, 0..) |*byte, i| {
        byte.* = @as(u8, @truncate(i % 256));
    }

    const zlib_data = try createZlibData(allocator, original_data, .{});
    defer allocator.free(zlib_data);

    var result = try zlib.decompress(zlib_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
}

test "ZLIB: Invalid compression method" {
    const allocator = testing.allocator;
    
    // CMF with compression method = 9 (invalid)
    const invalid_data = [_]u8{ 0x79, 0x9c, 0x01, 0x00, 0x00, 0xff, 0xff, 0x00, 0x01, 0x00, 0x00 };
    
    const result = zlib.decompress(&invalid_data, allocator);
    try testing.expectError(error.UnsupportedCompressionMethod, result);
}

test "ZLIB: Invalid FCHECK" {
    const allocator = testing.allocator;
    
    // CMF=0x78, FLG=0x00 (wrong FCHECK, should be 0x9c)
    const invalid_data = [_]u8{ 0x78, 0x00, 0x01, 0x00, 0x00, 0xff, 0xff, 0x00, 0x01, 0x00, 0x00 };
    
    const result = zlib.decompress(&invalid_data, allocator);
    try testing.expectError(error.InvalidChecksum, result);
}

test "ZLIB: Invalid window size" {
    const allocator = testing.allocator;
    
    // CMF with CINFO=8 (invalid, must be <= 7)
    const invalid_data = [_]u8{ 0x88, 0x5e, 0x01, 0x00, 0x00, 0xff, 0xff, 0x00, 0x01, 0x00, 0x00 };
    
    const result = zlib.decompress(&invalid_data, allocator);
    try testing.expectError(error.InvalidWindowSize, result);
}

test "ZLIB: Truncated header" {
    const allocator = testing.allocator;
    
    const invalid_data = [_]u8{0x78};
    
    const result = zlib.decompress(&invalid_data, allocator);
    try testing.expectError(error.InvalidHeader, result);
}

test "ZLIB: Dictionary required (unsupported)" {
    const allocator = testing.allocator;
    const original_data = "Data with dictionary";

    const zlib_data = try createZlibData(allocator, original_data, .{ .add_dict = true });
    defer allocator.free(zlib_data);

    const result = zlib.decompress(zlib_data, allocator);
    try testing.expectError(error.DictionaryRequired, result);
}

test "ZLIB: isZlib detection" {
    // Valid ZLIB (CMF=0x78, FLG=0x9c, default compression)
    const valid = [_]u8{ 0x78, 0x9c, 0x01, 0x00 };
    try testing.expect(zlib.isZlib(&valid));

    // Valid ZLIB (CMF=0x78, FLG=0xda, maximum compression)
    const valid2 = [_]u8{ 0x78, 0xda, 0x01, 0x00 };
    try testing.expect(zlib.isZlib(&valid2));

    // Invalid compression method
    const invalid1 = [_]u8{ 0x79, 0x9c, 0x01, 0x00 };
    try testing.expect(!zlib.isZlib(&invalid1));

    // Invalid FCHECK
    const invalid2 = [_]u8{ 0x78, 0x00, 0x01, 0x00 };
    try testing.expect(!zlib.isZlib(&invalid2));

    // Too short
    const too_short = [_]u8{0x78};
    try testing.expect(!zlib.isZlib(&too_short));
}

test "ZLIB: Header parsing" {
    const allocator = testing.allocator;
    const original_data = "Header test";

    const zlib_data = try createZlibData(allocator, original_data, .{
        .compression_info = 5,
        .compression_level = .fast,
    });
    defer allocator.free(zlib_data);

    const header_result = try zlib.parseHeader(zlib_data);
    
    try testing.expectEqual(zlib.CompressionMethod.deflate, header_result.header.compression_method);
    try testing.expectEqual(@as(u4, 5), header_result.header.compression_info);
    try testing.expectEqual(zlib.CompressionLevel.fast, header_result.header.flevel);
    try testing.expectEqual(@as(bool, false), header_result.header.fdict);
    try testing.expectEqual(@as(?u32, null), header_result.header.dict_id);
}

test "ZLIB: Unicode content" {
    const allocator = testing.allocator;
    const original_data = "Hello 世界 🌍 مرحبا";

    const zlib_data = try createZlibData(allocator, original_data, .{});
    defer allocator.free(zlib_data);

    var result = try zlib.decompress(zlib_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
}

test "ZLIB: Binary data" {
    const allocator = testing.allocator;
    
    var original_data: [256]u8 = undefined;
    for (&original_data, 0..) |*byte, i| {
        byte.* = @as(u8, @truncate(i));
    }

    const zlib_data = try createZlibData(allocator, &original_data, .{});
    defer allocator.free(zlib_data);

    var result = try zlib.decompress(zlib_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, &original_data, result.data);
}

test "ZLIB: Adler32 calculation" {
    const test_data = "Wikipedia";
    const expected_adler32: u32 = 0x11E60398;
    
    const actual_adler32 = calculateAdler32(test_data);
    try testing.expectEqual(expected_adler32, actual_adler32);
}

test "ZLIB: Window size calculation" {
    // Test window size calculation for different CINFO values
    const test_cases = [_]struct { cinfo: u4, expected_size: u32 }{
        .{ .cinfo = 0, .expected_size = 256 },     // 2^8
        .{ .cinfo = 1, .expected_size = 512 },     // 2^9
        .{ .cinfo = 2, .expected_size = 1024 },    // 2^10
        .{ .cinfo = 3, .expected_size = 2048 },    // 2^11
        .{ .cinfo = 4, .expected_size = 4096 },    // 2^12
        .{ .cinfo = 5, .expected_size = 8192 },    // 2^13
        .{ .cinfo = 6, .expected_size = 16384 },   // 2^14
        .{ .cinfo = 7, .expected_size = 32768 },   // 2^15
    };
    
    for (test_cases) |test_case| {
        const header = zlib.Header{
            .cmf = (@as(u8, test_case.cinfo) << 4) | 8,
            .flg = 0,
            .compression_method = .deflate,
            .compression_info = test_case.cinfo,
            .fcheck = 0,
            .fdict = false,
            .flevel = .default,
            .dict_id = null,
        };
        
        const window_size = zlib.getWindowSize(header);
        try testing.expectEqual(test_case.expected_size, window_size);
    }
}

test "ZLIB: Repeating pattern data" {
    const allocator = testing.allocator;
    
    // Create data with repeating pattern (good for compression)
    var original_data = try allocator.alloc(u8, 1000);
    defer allocator.free(original_data);
    
    for (original_data, 0..) |*byte, i| {
        byte.* = @as(u8, @truncate(i % 10)); // 0-9 repeating
    }

    const zlib_data = try createZlibData(allocator, original_data, .{});
    defer allocator.free(zlib_data);

    var result = try zlib.decompress(zlib_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
}
