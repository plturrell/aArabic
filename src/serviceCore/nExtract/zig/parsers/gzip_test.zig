// GZIP Parser Tests
// Day 14: Comprehensive test suite for GZIP functionality

const std = @import("std");
const gzip = @import("gzip.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Helper function to create test GZIP data
fn createGzipData(allocator: Allocator, uncompressed: []const u8, options: struct {
    add_filename: bool = false,
    add_comment: bool = false,
    add_extra: bool = false,
    add_header_crc: bool = false,
}) ![]u8 {
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // Magic bytes
    try data.append(gzip.GZIP_MAGIC1);
    try data.append(gzip.GZIP_MAGIC2);

    // Compression method (DEFLATE)
    try data.append(8);

    // Flags
    var flags: u8 = 0;
    if (options.add_filename) flags |= 0x08;
    if (options.add_comment) flags |= 0x10;
    if (options.add_extra) flags |= 0x04;
    if (options.add_header_crc) flags |= 0x02;
    try data.append(flags);

    // MTIME (0 = no timestamp)
    try data.appendSlice(&[_]u8{ 0, 0, 0, 0 });

    // XFL (extra flags, 0 = default)
    try data.append(0);

    // OS (255 = unknown)
    try data.append(255);

    // Extra field
    if (options.add_extra) {
        const extra = "EX";
        try data.appendSlice(&[_]u8{ 2, 0 }); // Length
        try data.appendSlice(extra);
    }

    // Filename
    if (options.add_filename) {
        try data.appendSlice("test.txt");
        try data.append(0); // Null terminator
    }

    // Comment
    if (options.add_comment) {
        try data.appendSlice("Test comment");
        try data.append(0); // Null terminator
    }

    // Header CRC16 (simplified - just use 0 for tests)
    if (options.add_header_crc) {
        try data.appendSlice(&[_]u8{ 0, 0 });
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

    // CRC32 of uncompressed data
    const crc32_value = calculateCrc32(uncompressed);
    try data.appendSlice(&[_]u8{
        @as(u8, @truncate(crc32_value)),
        @as(u8, @truncate(crc32_value >> 8)),
        @as(u8, @truncate(crc32_value >> 16)),
        @as(u8, @truncate(crc32_value >> 24)),
    });

    // ISIZE (size of uncompressed data modulo 2^32)
    const isize = @as(u32, @truncate(uncompressed.len));
    try data.appendSlice(&[_]u8{
        @as(u8, @truncate(isize)),
        @as(u8, @truncate(isize >> 8)),
        @as(u8, @truncate(isize >> 16)),
        @as(u8, @truncate(isize >> 24)),
    });

    return data.toOwnedSlice();
}

fn calculateCrc32(data: []const u8) u32 {
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

test "GZIP: Basic decompression" {
    const allocator = testing.allocator;
    const original_data = "Hello, GZIP World!";

    const gzip_data = try createGzipData(allocator, original_data, .{});
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
    try testing.expectEqual(@as(u8, 8), result.header.compression_method);
}

test "GZIP: With filename" {
    const allocator = testing.allocator;
    const original_data = "Data with filename";

    const gzip_data = try createGzipData(allocator, original_data, .{ .add_filename = true });
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
    try testing.expect(result.header.filename != null);
    try testing.expectEqualSlices(u8, "test.txt", result.header.filename.?);
}

test "GZIP: With comment" {
    const allocator = testing.allocator;
    const original_data = "Data with comment";

    const gzip_data = try createGzipData(allocator, original_data, .{ .add_comment = true });
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
    try testing.expect(result.header.comment != null);
    try testing.expectEqualSlices(u8, "Test comment", result.header.comment.?);
}

test "GZIP: With extra field" {
    const allocator = testing.allocator;
    const original_data = "Data with extra field";

    const gzip_data = try createGzipData(allocator, original_data, .{ .add_extra = true });
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
    try testing.expect(result.header.extra_field != null);
    try testing.expectEqualSlices(u8, "EX", result.header.extra_field.?);
}

test "GZIP: With all optional fields" {
    const allocator = testing.allocator;
    const original_data = "Complete GZIP test";

    const gzip_data = try createGzipData(allocator, original_data, .{
        .add_filename = true,
        .add_comment = true,
        .add_extra = true,
    });
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
    try testing.expect(result.header.filename != null);
    try testing.expect(result.header.comment != null);
    try testing.expect(result.header.extra_field != null);
}

test "GZIP: Empty data" {
    const allocator = testing.allocator;
    const original_data = "";

    const gzip_data = try createGzipData(allocator, original_data, .{});
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.data.len);
}

test "GZIP: Large data" {
    const allocator = testing.allocator;

    // Create 10KB of test data
    var original_data = try allocator.alloc(u8, 10240);
    defer allocator.free(original_data);

    // Fill with pattern
    for (original_data, 0..) |*byte, i| {
        byte.* = @as(u8, @truncate(i % 256));
    }

    const gzip_data = try createGzipData(allocator, original_data, .{});
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
}

test "GZIP: Invalid magic bytes" {
    const allocator = testing.allocator;
    
    const invalid_data = [_]u8{ 0x00, 0x00, 0x08, 0x00, 0, 0, 0, 0, 0, 255 };
    
    const result = gzip.decompress(&invalid_data, allocator);
    try testing.expectError(error.InvalidMagic, result);
}

test "GZIP: Invalid compression method" {
    const allocator = testing.allocator;
    
    const invalid_data = [_]u8{ 0x1f, 0x8b, 0x09, 0x00, 0, 0, 0, 0, 0, 255 };
    
    const result = gzip.decompress(&invalid_data, allocator);
    try testing.expectError(error.UnsupportedCompressionMethod, result);
}

test "GZIP: Truncated header" {
    const allocator = testing.allocator;
    
    const invalid_data = [_]u8{ 0x1f, 0x8b, 0x08 };
    
    const result = gzip.decompress(&invalid_data, allocator);
    try testing.expectError(error.InvalidHeader, result);
}

test "GZIP: isGzip detection" {
    // Valid GZIP
    const valid = [_]u8{ 0x1f, 0x8b, 0x08, 0x00 };
    try testing.expect(gzip.isGzip(&valid));

    // Invalid GZIP
    const invalid1 = [_]u8{ 0x1f, 0x8a, 0x08, 0x00 };
    try testing.expect(!gzip.isGzip(&invalid1));

    const invalid2 = [_]u8{ 0x00, 0x8b, 0x08, 0x00 };
    try testing.expect(!gzip.isGzip(&invalid2));

    // Too short
    const too_short = [_]u8{0x1f};
    try testing.expect(!gzip.isGzip(&too_short));
}

test "GZIP: Unicode content" {
    const allocator = testing.allocator;
    const original_data = "Hello 世界 🌍 مرحبا";

    const gzip_data = try createGzipData(allocator, original_data, .{});
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, original_data, result.data);
}

test "GZIP: Binary data" {
    const allocator = testing.allocator;
    
    var original_data: [256]u8 = undefined;
    for (&original_data, 0..) |*byte, i| {
        byte.* = @as(u8, @truncate(i));
    }

    const gzip_data = try createGzipData(allocator, &original_data, .{});
    defer allocator.free(gzip_data);

    var result = try gzip.decompress(gzip_data, allocator);
    defer result.deinit();

    try testing.expectEqualSlices(u8, &original_data, result.data);
}
