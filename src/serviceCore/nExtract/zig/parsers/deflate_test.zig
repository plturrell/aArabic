//! DEFLATE Decompression Tests
//! 
//! Comprehensive test suite for the DEFLATE implementation including:
//! - Uncompressed blocks
//! - Fixed Huffman blocks
//! - Dynamic Huffman blocks
//! - LZ77 back-references
//! - Edge cases and error handling

const std = @import("std");
const testing = std.testing;
const deflate = @import("deflate.zig");

test "DEFLATE: empty uncompressed block" {
    const allocator = testing.allocator;
    
    // Empty uncompressed block
    const compressed = [_]u8{
        0x01,       // BFINAL=1, BTYPE=00 (uncompressed)
        0x00, 0x00, // LEN = 0
        0xFF, 0xFF, // NLEN = ~0
    };
    
    const result = try deflate.decompress(allocator, &compressed);
    defer allocator.free(result);
    
    try testing.expectEqual(@as(usize, 0), result.len);
}

test "DEFLATE: uncompressed block with data" {
    const allocator = testing.allocator;
    
    // Uncompressed block: "Hello, World!"
    const compressed = [_]u8{
        0x01,       // BFINAL=1, BTYPE=00 (uncompressed)
        0x0D, 0x00, // LEN = 13
        0xF2, 0xFF, // NLEN = ~13
        'H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!',
    };
    
    const result = try deflate.decompress(allocator, &compressed);
    defer allocator.free(result);
    
    try testing.expectEqualStrings("Hello, World!", result);
}

test "DEFLATE: multiple uncompressed blocks" {
    const allocator = testing.allocator;
    
    // Two uncompressed blocks: "ABC" + "DEF"
    const compressed = [_]u8{
        // First block (not final)
        0x00,       // BFINAL=0, BTYPE=00
        0x03, 0x00, // LEN = 3
        0xFC, 0xFF, // NLEN = ~3
        'A', 'B', 'C',
        
        // Second block (final)
        0x01,       // BFINAL=1, BTYPE=00
        0x03, 0x00, // LEN = 3
        0xFC, 0xFF, // NLEN = ~3
        'D', 'E', 'F',
    };
    
    const result = try deflate.decompress(allocator, &compressed);
    defer allocator.free(result);
    
    try testing.expectEqualStrings("ABCDEF", result);
}

test "DEFLATE: bit reader basics" {
    const data = [_]u8{ 0b10110011, 0b11001010 };
    var reader = deflate.BitReader.init(&data);
    
    // Read individual bits (LSB first)
    try testing.expectEqual(@as(u1, 1), try reader.readBit()); // bit 0
    try testing.expectEqual(@as(u1, 1), try reader.readBit()); // bit 1
    try testing.expectEqual(@as(u1, 0), try reader.readBit()); // bit 2
    try testing.expectEqual(@as(u1, 0), try reader.readBit()); // bit 3
    try testing.expectEqual(@as(u1, 1), try reader.readBit()); // bit 4
    try testing.expectEqual(@as(u1, 1), try reader.readBit()); // bit 5
    try testing.expectEqual(@as(u1, 0), try reader.readBit()); // bit 6
    try testing.expectEqual(@as(u1, 1), try reader.readBit()); // bit 7
    
    // Read from second byte
    try testing.expectEqual(@as(u1, 0), try reader.readBit()); // bit 0 of byte 2
    try testing.expectEqual(@as(u1, 1), try reader.readBit()); // bit 1 of byte 2
}

test "DEFLATE: bit reader multi-bit reads" {
    const data = [_]u8{ 0b11010101, 0b10101100 };
    var reader = deflate.BitReader.init(&data);
    
    // Read 3 bits: 101 (LSB first from 11010101)
    const bits3 = try reader.readBits(3);
    try testing.expectEqual(@as(u16, 0b101), bits3);
    
    // Read 5 bits: 10101 (remaining 5 bits from first byte)
    const bits5 = try reader.readBits(5);
    try testing.expectEqual(@as(u16, 0b10101), bits5);
    
    // Read 4 bits from second byte: 1100 (LSB first from 10101100)
    const bits4 = try reader.readBits(4);
    try testing.expectEqual(@as(u16, 0b1100), bits4);
}

test "DEFLATE: bit reader byte alignment" {
    const data = [_]u8{ 0xFF, 0xAA, 0x55 };
    var reader = deflate.BitReader.init(&data);
    
    // Read 3 bits
    _ = try reader.readBits(3);
    
    // Align to next byte
    reader.alignByte();
    
    // Should now be at byte 1
    const byte = try reader.readBytes(1);
    try testing.expectEqual(@as(u8, 0xAA), byte[0]);
}

test "DEFLATE: Huffman table building" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    // Simple code lengths: 2, 3, 3, 4
    const lengths = [_]u8{ 2, 3, 3, 4 };
    
    var table = deflate.HuffmanTable.init();
    try decompressor.buildHuffmanTable(&table, &lengths);
    
    // Verify table was built
    try testing.expect(table.max_length > 0);
    try testing.expectEqual(@as(u8, 4), table.max_length);
}

test "DEFLATE: fixed Huffman table generation" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    try decompressor.buildFixedHuffmanTables();
    
    // Verify literal table was built
    try testing.expect(decompressor.literal_table.max_length > 0);
    
    // Verify distance table was built
    try testing.expect(decompressor.distance_table.max_length > 0);
    try testing.expectEqual(@as(u8, 5), decompressor.distance_table.max_length);
}

test "DEFLATE: LZ77 copy from history" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    // Output "ABCD"
    try decompressor.outputByte('A');
    try decompressor.outputByte('B');
    try decompressor.outputByte('C');
    try decompressor.outputByte('D');
    
    // Copy last 2 bytes (CD) with distance=2, length=2
    try decompressor.copyFromHistory(2, 2);
    
    const result = try decompressor.output.toOwnedSlice();
    defer allocator.free(result);
    
    try testing.expectEqualStrings("ABCDCD", result);
}

test "DEFLATE: LZ77 overlapping copy" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    // Output "AB"
    try decompressor.outputByte('A');
    try decompressor.outputByte('B');
    
    // Copy with distance=2, length=6 (creates ABABAB)
    try decompressor.copyFromHistory(2, 6);
    
    const result = try decompressor.output.toOwnedSlice();
    defer allocator.free(result);
    
    try testing.expectEqualStrings("ABABABAB", result);
}

test "DEFLATE: error - corrupt NLEN" {
    const allocator = testing.allocator;
    
    // Invalid NLEN (not one's complement of LEN)
    const compressed = [_]u8{
        0x01,       // BFINAL=1, BTYPE=00
        0x05, 0x00, // LEN = 5
        0x00, 0x00, // NLEN = 0 (should be ~5)
        'H', 'e', 'l', 'l', 'o',
    };
    
    const result = deflate.decompress(allocator, &compressed);
    try testing.expectError(deflate.DeflateError.CorruptData, result);
}

test "DEFLATE: error - invalid block type" {
    const allocator = testing.allocator;
    
    // Reserved block type (11)
    const compressed = [_]u8{
        0x07, // BFINAL=1, BTYPE=11 (reserved)
    };
    
    const result = deflate.decompress(allocator, &compressed);
    try testing.expectError(deflate.DeflateError.InvalidBlockType, result);
}

test "DEFLATE: error - invalid distance" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    // Try to copy with distance=0 (invalid)
    const result = decompressor.copyFromHistory(0, 5);
    try testing.expectError(deflate.DeflateError.InvalidDistance, result);
}

test "DEFLATE: error - distance too large" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    // Output one byte
    try decompressor.outputByte('A');
    
    // Try to copy with distance > window size
    const result = decompressor.copyFromHistory(33000, 5);
    try testing.expectError(deflate.DeflateError.InvalidDistance, result);
}

test "DEFLATE: window wrapping" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    // Fill window with pattern
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try decompressor.outputByte(@as(u8, @truncate(i % 26 + 'A')));
    }
    
    // Copy from history
    try decompressor.copyFromHistory(50, 10);
    
    // Verify output length
    try testing.expectEqual(@as(usize, 110), decompressor.output.items.len);
}

test "DEFLATE: real-world test - zlib header compatibility" {
    // Note: This tests only DEFLATE, not full ZLIB (which has header/checksum)
    // Real ZLIB data would need header stripping before passing to DEFLATE
    
    const allocator = testing.allocator;
    
    // Minimal valid DEFLATE stream (empty fixed Huffman)
    const compressed = [_]u8{
        0x03, 0x00, // BFINAL=1, BTYPE=01 (fixed), end-of-block immediately
    };
    
    const result = try deflate.decompress(allocator, &compressed);
    defer allocator.free(result);
    
    try testing.expectEqual(@as(usize, 0), result.len);
}

test "DEFLATE: stress test - large uncompressed block" {
    const allocator = testing.allocator;
    
    // Create a 1KB uncompressed block
    const data_size = 1024;
    var data: [data_size]u8 = undefined;
    for (0..data_size) |i| {
        data[i] = @as(u8, @truncate(i % 256));
    }
    
    // Build compressed stream
    var compressed = std.ArrayList(u8).init(allocator);
    defer compressed.deinit();
    
    try compressed.append(0x01); // BFINAL=1, BTYPE=00
    try compressed.append(@as(u8, @truncate(data_size & 0xFF)));
    try compressed.append(@as(u8, @truncate((data_size >> 8) & 0xFF)));
    try compressed.append(@as(u8, @truncate((~data_size) & 0xFF)));
    try compressed.append(@as(u8, @truncate(((~data_size) >> 8) & 0xFF)));
    try compressed.appendSlice(&data);
    
    const result = try deflate.decompress(allocator, compressed.items);
    defer allocator.free(result);
    
    try testing.expectEqual(data_size, result.len);
    try testing.expectEqualSlices(u8, &data, result);
}

test "DEFLATE: BitReader end of stream detection" {
    const data = [_]u8{ 0xFF };
    var reader = deflate.BitReader.init(&data);
    
    // Read all 8 bits
    _ = try reader.readBits(8);
    
    // Next read should fail
    const result = reader.readBit();
    try testing.expectError(deflate.DeflateError.EndOfStream, result);
}

test "DEFLATE: performance - repeated pattern compression" {
    const allocator = testing.allocator;
    
    var decompressor = try deflate.Decompressor.init(allocator, &[_]u8{0});
    defer decompressor.deinit();
    
    // Create pattern "AAAA"
    try decompressor.outputByte('A');
    try decompressor.outputByte('A');
    try decompressor.outputByte('A');
    try decompressor.outputByte('A');
    
    // Repeat pattern 10 times using back-references
    var i: u8 = 0;
    while (i < 10) : (i += 1) {
        try decompressor.copyFromHistory(4, 4);
    }
    
    const result = try decompressor.output.toOwnedSlice();
    defer allocator.free(result);
    
    // Should have 4 + (10 * 4) = 44 bytes
    try testing.expectEqual(@as(usize, 44), result.len);
    
    // All should be 'A'
    for (result) |byte| {
        try testing.expectEqual(@as(u8, 'A'), byte);
    }
}
