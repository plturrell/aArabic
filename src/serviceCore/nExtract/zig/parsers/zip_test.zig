const std = @import("std");
const zip = @import("zip.zig");
const testing = std.testing;

test "CRC32 calculation" {
    const data = "Hello, World!";
    const crc = zip.crc32(data);
    
    // Known CRC32 for "Hello, World!"
    try testing.expectEqual(@as(u32, 0xec4ac3d0), crc);
}

test "CRC32 empty data" {
    const data = "";
    const crc = zip.crc32(data);
    try testing.expectEqual(@as(u32, 0), crc);
}

test "CRC32 incremental" {
    const part1 = "Hello, ";
    const full = "Hello, World!";
    
    const crc_full = zip.crc32(full);
    const crc_part = zip.crc32(part1);
    
    // Note: This test shows CRC32 is not simply additive
    // The full CRC should differ from just the first part
    try testing.expect(crc_full != crc_part);
}

// Helper to create a minimal valid ZIP archive
fn createMinimalZip(allocator: std.mem.Allocator, file_name: []const u8, file_content: []const u8) ![]u8 {
    var data = std.ArrayList(u8){};
    errdefer data.deinit(allocator);
    
    const crc = zip.crc32(file_content);
    
    // Local file header
    const local_header_offset: u32 = 0;
    const writer = data.writer(allocator);
    try writer.writeInt(u32, 0x04034b50, .little); // Signature
    try writer.writeInt(u16, 20, .little); // Version needed
    try writer.writeInt(u16, 0, .little); // Flags
    try writer.writeInt(u16, 0, .little); // Compression method (store)
    try writer.writeInt(u16, 0, .little); // Mod time
    try writer.writeInt(u16, 0, .little); // Mod date
    try writer.writeInt(u32, crc, .little); // CRC32
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little); // Compressed size
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little); // Uncompressed size
    try writer.writeInt(u16, @as(u16, @intCast(file_name.len)), .little); // File name length
    try writer.writeInt(u16, 0, .little); // Extra field length
    try writer.writeAll(file_name);
    try writer.writeAll(file_content);
    
    // Central directory
    const cd_offset = data.items.len;
    try writer.writeInt(u32, 0x02014b50, .little); // Signature
    try writer.writeInt(u16, 20, .little); // Version made by
    try writer.writeInt(u16, 20, .little); // Version needed
    try writer.writeInt(u16, 0, .little); // Flags
    try writer.writeInt(u16, 0, .little); // Compression method
    try writer.writeInt(u16, 0, .little); // Mod time
    try writer.writeInt(u16, 0, .little); // Mod date
    try writer.writeInt(u32, crc, .little); // CRC32
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little); // Compressed size
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little); // Uncompressed size
    try writer.writeInt(u16, @as(u16, @intCast(file_name.len)), .little); // File name length
    try writer.writeInt(u16, 0, .little); // Extra field length
    try writer.writeInt(u16, 0, .little); // Comment length
    try writer.writeInt(u16, 0, .little); // Disk number
    try writer.writeInt(u16, 0, .little); // Internal attributes
    try writer.writeInt(u32, 0, .little); // External attributes
    try writer.writeInt(u32, local_header_offset, .little); // Local header offset
    try writer.writeAll(file_name);
    
    // End of central directory
    const cd_size = data.items.len - cd_offset;
    try writer.writeInt(u32, 0x06054b50, .little); // Signature
    try writer.writeInt(u16, 0, .little); // Disk number
    try writer.writeInt(u16, 0, .little); // Disk with CD
    try writer.writeInt(u16, 1, .little); // Entries on this disk
    try writer.writeInt(u16, 1, .little); // Total entries
    try writer.writeInt(u32, @as(u32, @intCast(cd_size)), .little); // CD size
    try writer.writeInt(u32, @as(u32, @intCast(cd_offset)), .little); // CD offset
    try writer.writeInt(u16, 0, .little); // Comment length
    
    return data.toOwnedSlice(allocator);
}

test "Create and parse minimal ZIP" {
    const allocator = testing.allocator;
    
    const file_content = "test content";
    const file_name = "test.txt";
    
    const zip_data = try createMinimalZip(allocator, file_name, file_content);
    defer allocator.free(zip_data);
    
    var archive = try zip.ZipArchive.open(allocator, zip_data);
    defer archive.deinit();
    
    try testing.expectEqual(@as(usize, 1), archive.entries.items.len);
    
    const entry = &archive.entries.items[0];
    try testing.expectEqualStrings(file_name, entry.file_name);
    try testing.expectEqual(@as(u64, file_content.len), entry.uncompressed_size);
    try testing.expectEqual(zip.CompressionMethod.Store, entry.compression_method);
}

test "Extract file from ZIP" {
    const allocator = testing.allocator;
    
    const file_content = "Hello from ZIP!";
    const file_name = "hello.txt";
    
    const zip_data = try createMinimalZip(allocator, file_name, file_content);
    defer allocator.free(zip_data);
    
    var archive = try zip.ZipArchive.open(allocator, zip_data);
    defer archive.deinit();
    
    const entry = &archive.entries.items[0];
    
    const output = try allocator.alloc(u8, entry.uncompressed_size);
    defer allocator.free(output);
    
    const size = try archive.extractFile(entry, output);
    try testing.expectEqual(file_content.len, size);
    try testing.expectEqualStrings(file_content, output);
}

test "Extract file with extractFileAlloc" {
    const allocator = testing.allocator;
    
    const file_content = "Allocated extraction test";
    const file_name = "alloc.txt";
    
    const zip_data = try createMinimalZip(allocator, file_name, file_content);
    defer allocator.free(zip_data);
    
    var archive = try zip.ZipArchive.open(allocator, zip_data);
    defer archive.deinit();
    
    const entry = &archive.entries.items[0];
    const extracted = try archive.extractFileAlloc(entry);
    defer allocator.free(extracted);
    
    try testing.expectEqualStrings(file_content, extracted);
}

test "Find entry by name" {
    const allocator = testing.allocator;
    
    const file_content = "findme";
    const file_name = "target.txt";
    
    const zip_data = try createMinimalZip(allocator, file_name, file_content);
    defer allocator.free(zip_data);
    
    var archive = try zip.ZipArchive.open(allocator, zip_data);
    defer archive.deinit();
    
    // Find existing entry
    const found = archive.findEntry(file_name);
    try testing.expect(found != null);
    try testing.expectEqualStrings(file_name, found.?.file_name);
    
    // Find non-existing entry
    const not_found = archive.findEntry("nonexistent.txt");
    try testing.expect(not_found == null);
}

test "CRC32 mismatch detection" {
    const allocator = testing.allocator;
    
    const file_content = "original content";
    const file_name = "corrupted.txt";
    
    var zip_data = try createMinimalZip(allocator, file_name, file_content);
    defer allocator.free(zip_data);
    
    // Corrupt the data by changing a byte in the content
    // Find the content offset (after headers and filename)
    const content_start = 30 + file_name.len;
    if (content_start < zip_data.len) {
        zip_data[content_start] ^= 0xFF; // Flip bits
    }
    
    var archive = try zip.ZipArchive.open(allocator, zip_data);
    defer archive.deinit();
    
    const entry = &archive.entries.items[0];
    const output = try allocator.alloc(u8, entry.uncompressed_size);
    defer allocator.free(output);
    
    // Should fail with CRC mismatch
    const result = archive.extractFile(entry, output);
    try testing.expectError(error.CRC32Mismatch, result);
}

test "Multiple files in ZIP" {
    const allocator = testing.allocator;
    
    // Create a ZIP with multiple files manually
    var data = std.ArrayList(u8){};
    defer data.deinit(allocator);
    
    const files = [_]struct {
        name: []const u8,
        content: []const u8,
    }{
        .{ .name = "file1.txt", .content = "Content 1" },
        .{ .name = "file2.txt", .content = "Content 2" },
        .{ .name = "file3.txt", .content = "Content 3" },
    };
    
    var local_offsets: [3]u32 = undefined;
    
    const writer = data.writer(allocator);
    
    // Write local file headers and data
    for (files, 0..) |file, i| {
        local_offsets[i] = @intCast(data.items.len);
        const crc = zip.crc32(file.content);
        
        try writer.writeInt(u32, 0x04034b50, .little);
        try writer.writeInt(u16, 20, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u32, crc, .little);
        try writer.writeInt(u32, @as(u32, @intCast(file.content.len)), .little);
        try writer.writeInt(u32, @as(u32, @intCast(file.content.len)), .little);
        try writer.writeInt(u16, @as(u16, @intCast(file.name.len)), .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeAll(file.name);
        try writer.writeAll(file.content);
    }
    
    // Write central directory
    const cd_offset = data.items.len;
    for (files, 0..) |file, i| {
        const crc = zip.crc32(file.content);
        
        try writer.writeInt(u32, 0x02014b50, .little);
        try writer.writeInt(u16, 20, .little);
        try writer.writeInt(u16, 20, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u32, crc, .little);
        try writer.writeInt(u32, @as(u32, @intCast(file.content.len)), .little);
        try writer.writeInt(u32, @as(u32, @intCast(file.content.len)), .little);
        try writer.writeInt(u16, @as(u16, @intCast(file.name.len)), .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u16, 0, .little);
        try writer.writeInt(u32, 0, .little);
        try writer.writeInt(u32, local_offsets[i], .little);
        try writer.writeAll(file.name);
    }
    
    // Write EOCD
    const cd_size = data.items.len - cd_offset;
    try writer.writeInt(u32, 0x06054b50, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, @as(u16, @intCast(files.len)), .little);
    try writer.writeInt(u16, @as(u16, @intCast(files.len)), .little);
    try writer.writeInt(u32, @as(u32, @intCast(cd_size)), .little);
    try writer.writeInt(u32, @as(u32, @intCast(cd_offset)), .little);
    try writer.writeInt(u16, 0, .little);
    
    // Parse the ZIP
    var archive = try zip.ZipArchive.open(allocator, data.items);
    defer archive.deinit();
    
    try testing.expectEqual(files.len, archive.entries.items.len);
    
    // Verify each file
    for (files, 0..) |file, i| {
        const entry = &archive.entries.items[i];
        try testing.expectEqualStrings(file.name, entry.file_name);
        
        const extracted = try archive.extractFileAlloc(entry);
        defer allocator.free(extracted);
        try testing.expectEqualStrings(file.content, extracted);
    }
}

test "ZIP with comment" {
    const allocator = testing.allocator;
    
    const file_content = "test";
    const file_name = "test.txt";
    const comment = "This is a ZIP comment";
    
    var data = std.ArrayList(u8){};
    defer data.deinit(allocator);
    
    const crc = zip.crc32(file_content);
    const writer = data.writer(allocator);
    
    // Local file header
    try writer.writeInt(u32, 0x04034b50, .little);
    try writer.writeInt(u16, 20, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u32, crc, .little);
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little);
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little);
    try writer.writeInt(u16, @as(u16, @intCast(file_name.len)), .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeAll(file_name);
    try writer.writeAll(file_content);
    
    // Central directory
    const cd_offset = data.items.len;
    try writer.writeInt(u32, 0x02014b50, .little);
    try writer.writeInt(u16, 20, .little);
    try writer.writeInt(u16, 20, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u32, crc, .little);
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little);
    try writer.writeInt(u32, @as(u32, @intCast(file_content.len)), .little);
    try writer.writeInt(u16, @as(u16, @intCast(file_name.len)), .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeAll(file_name);
    
    // EOCD with comment
    const cd_size = data.items.len - cd_offset;
    try writer.writeInt(u32, 0x06054b50, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 1, .little);
    try writer.writeInt(u16, 1, .little);
    try writer.writeInt(u32, @as(u32, @intCast(cd_size)), .little);
    try writer.writeInt(u32, @as(u32, @intCast(cd_offset)), .little);
    try writer.writeInt(u16, @as(u16, @intCast(comment.len)), .little);
    try writer.writeAll(comment);
    
    // Parse ZIP
    var archive = try zip.ZipArchive.open(allocator, data.items);
    defer archive.deinit();
    
    try testing.expectEqualStrings(comment, archive.eocd.comment);
}

test "Invalid ZIP detection" {
    const allocator = testing.allocator;
    
    // Empty data
    {
        const result = zip.ZipArchive.open(allocator, &[_]u8{});
        try testing.expectError(error.InvalidZipFile, result);
    }
    
    // Too short
    {
        const short_data = [_]u8{0} ** 10;
        const result = zip.ZipArchive.open(allocator, &short_data);
        try testing.expectError(error.InvalidZipFile, result);
    }
    
    // Invalid signature
    {
        var invalid = [_]u8{0} ** 30;
        const result = zip.ZipArchive.open(allocator, &invalid);
        try testing.expectError(error.EndOfCentralDirNotFound, result);
    }
}

test "Buffer too small error" {
    const allocator = testing.allocator;
    
    const file_content = "This is a long content that won't fit";
    const file_name = "large.txt";
    
    const zip_data = try createMinimalZip(allocator, file_name, file_content);
    defer allocator.free(zip_data);
    
    var archive = try zip.ZipArchive.open(allocator, zip_data);
    defer archive.deinit();
    
    const entry = &archive.entries.items[0];
    
    // Buffer too small
    var small_buffer: [5]u8 = undefined;
    const result = archive.extractFile(entry, &small_buffer);
    try testing.expectError(error.BufferTooSmall, result);
}
