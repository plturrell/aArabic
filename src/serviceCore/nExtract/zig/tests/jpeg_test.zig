const std = @import("std");
const testing = std.testing;
const jpeg = @import("../parsers/jpeg.zig");

// Helper to create a minimal valid JPEG structure
fn createMinimalJPEG(allocator: std.mem.Allocator, width: u16, height: u16) ![]u8 {
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // SOI marker
    try data.append(0xFF);
    try data.append(0xD8);

    // APP0 (JFIF) marker
    try data.append(0xFF);
    try data.append(0xE0);
    try data.append(0x00);
    try data.append(0x10); // Length = 16
    try data.appendSlice("JFIF\x00"); // Identifier
    try data.append(0x01); // Major version
    try data.append(0x01); // Minor version
    try data.append(0x00); // Density units
    try data.append(0x00); // X density
    try data.append(0x01);
    try data.append(0x00); // Y density
    try data.append(0x01);
    try data.append(0x00); // Thumbnail width
    try data.append(0x00); // Thumbnail height

    // SOF0 marker (Baseline DCT)
    try data.append(0xFF);
    try data.append(0xC0);
    const sof_length: u16 = 8 + (3 * 1); // 8 bytes + 3 bytes per component
    try data.append(@intCast((sof_length >> 8) & 0xFF));
    try data.append(@intCast(sof_length & 0xFF));
    try data.append(0x08); // Precision = 8 bits
    try data.append(@intCast((height >> 8) & 0xFF));
    try data.append(@intCast(height & 0xFF));
    try data.append(@intCast((width >> 8) & 0xFF));
    try data.append(@intCast(width & 0xFF));
    try data.append(0x01); // Component count = 1 (grayscale)
    // Component 1
    try data.append(0x01); // ID
    try data.append(0x11); // Sampling factors (1x1)
    try data.append(0x00); // Quantization table ID

    // DQT marker (Define Quantization Table)
    try data.append(0xFF);
    try data.append(0xDB);
    try data.append(0x00);
    try data.append(0x43); // Length = 67 (2 + 1 + 64)
    try data.append(0x00); // Precision = 8-bit, Table ID = 0
    // Quantization table (64 values)
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        try data.append(@intCast((i % 16) + 1));
    }

    // DHT marker (Define Huffman Table) - DC table
    try data.append(0xFF);
    try data.append(0xC4);
    try data.append(0x00);
    try data.append(0x1F); // Length = 31
    try data.append(0x00); // Table class = DC, Table ID = 0
    // Bits
    try data.appendSlice(&[_]u8{ 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 });
    // Values
    try data.appendSlice(&[_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });

    // SOS marker (Start of Scan)
    try data.append(0xFF);
    try data.append(0xDA);
    try data.append(0x00);
    try data.append(0x08); // Length = 8
    try data.append(0x01); // Component count = 1
    try data.append(0x01); // Component ID = 1
    try data.append(0x00); // DC/AC table IDs
    try data.append(0x00); // Start of spectral selection
    try data.append(0x3F); // End of spectral selection
    try data.append(0x00); // Successive approximation

    // Compressed data (minimal)
    try data.appendSlice(&[_]u8{ 0xFF, 0x00 });

    // EOI marker
    try data.append(0xFF);
    try data.append(0xD9);

    return data.toOwnedSlice();
}

test "JPEG marker constants" {
    try testing.expectEqual(@as(u16, 0xFFD8), jpeg.MARKER_SOI);
    try testing.expectEqual(@as(u16, 0xFFD9), jpeg.MARKER_EOI);
    try testing.expectEqual(@as(u16, 0xFFDA), jpeg.MARKER_SOS);
    try testing.expectEqual(@as(u16, 0xFFC0), jpeg.MARKER_SOF0);
    try testing.expectEqual(@as(u16, 0xFFDB), jpeg.MARKER_DQT);
    try testing.expectEqual(@as(u16, 0xFFC4), jpeg.MARKER_DHT);
}

test "QuantTable initialization" {
    const qt = jpeg.QuantTable.init();
    try testing.expectEqual(@as(u8, 0), qt.precision);
    try testing.expectEqual(@as(u16, 0), qt.table[0]);
    try testing.expectEqual(@as(u16, 0), qt.table[63]);
}

test "HuffmanTable initialization" {
    const ht = jpeg.HuffmanTable.init();
    try testing.expectEqual(@as(u16, 0), ht.codes[0]);
    try testing.expectEqual(@as(u8, 0), ht.values[0]);
    try testing.expectEqual(@as(i32, -1), ht.min_code[0]);
    try testing.expectEqual(@as(i32, -1), ht.max_code[0]);
}

test "ExifMetadata initialization and cleanup" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var exif = jpeg.ExifMetadata.init(allocator);
    defer exif.deinit();

    try testing.expect(exif.make == null);
    try testing.expect(exif.model == null);
    try testing.expectEqual(@as(u16, 1), exif.orientation);
    try testing.expectEqual(@as(f32, 72.0), exif.x_resolution);
    try testing.expectEqual(@as(f32, 72.0), exif.y_resolution);
}

test "JpegImage creation and destruction" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var image = try jpeg.JpegImage.create(allocator, 100, 100, 3);
    defer image.deinit();

    try testing.expectEqual(@as(u16, 100), image.width);
    try testing.expectEqual(@as(u16, 100), image.height);
    try testing.expectEqual(@as(u8, 3), image.component_count);
    try testing.expectEqual(@as(usize, 30000), image.pixels.len); // 100*100*3
}

test "JpegImage pixel get/set" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var image = try jpeg.JpegImage.create(allocator, 10, 10, 3);
    defer image.deinit();

    // Set pixel at (5, 5)
    const test_pixel = [3]u8{ 255, 128, 64 };
    image.setPixel(5, 5, test_pixel);

    // Get pixel back
    const retrieved = image.getPixel(5, 5);
    try testing.expect(retrieved != null);
    try testing.expectEqual(test_pixel[0], retrieved.?[0]);
    try testing.expectEqual(test_pixel[1], retrieved.?[1]);
    try testing.expectEqual(test_pixel[2], retrieved.?[2]);

    // Test out of bounds
    const oob = image.getPixel(20, 20);
    try testing.expect(oob == null);
}

test "JpegDecoder creation" {
    const allocator = testing.allocator;
    var decoder = jpeg.JpegDecoder.create(allocator);

    try testing.expectEqual(@as(u16, 0), decoder.restart_interval);
    try testing.expectEqual(@as(u8, 0), decoder.quant_tables[0].precision);
}

test "JpegDecoder invalid SOI marker" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var decoder = jpeg.JpegDecoder.create(allocator);

    // Invalid SOI marker
    const invalid_data = [_]u8{ 0x00, 0x00, 0x01, 0x02, 0x03 };
    const result = decoder.decode(&invalid_data);
    try testing.expectError(error.InvalidJpegMarker, result);
}

test "JpegDecoder minimal valid JPEG" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const jpeg_data = try createMinimalJPEG(allocator, 8, 8);
    defer allocator.free(jpeg_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    var image = try decoder.decode(jpeg_data);
    defer image.deinit();

    try testing.expectEqual(@as(u16, 8), image.width);
    try testing.expectEqual(@as(u16, 8), image.height);
    try testing.expectEqual(@as(u8, 1), image.component_count);
}

test "JpegDecoder SOF parsing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // SOF0 data
    try data.append(0x00);
    try data.append(0x11); // Length = 17
    try data.append(0x08); // Precision = 8
    try data.append(0x00);
    try data.append(0x40); // Height = 64
    try data.append(0x00);
    try data.append(0x40); // Width = 64
    try data.append(0x03); // Component count = 3
    // Component 1 (Y)
    try data.append(0x01);
    try data.append(0x22); // 2x2 sampling
    try data.append(0x00); // Quant table 0
    // Component 2 (Cb)
    try data.append(0x02);
    try data.append(0x11); // 1x1 sampling
    try data.append(0x01); // Quant table 1
    // Component 3 (Cr)
    try data.append(0x03);
    try data.append(0x11); // 1x1 sampling
    try data.append(0x01); // Quant table 1

    const sof_data = try data.toOwnedSlice();
    defer allocator.free(sof_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    const header = try decoder.parseSOF(sof_data, jpeg.MARKER_SOF0);

    try testing.expectEqual(@as(u8, 8), header.precision);
    try testing.expectEqual(@as(u16, 64), header.height);
    try testing.expectEqual(@as(u16, 64), header.width);
    try testing.expectEqual(@as(u8, 3), header.component_count);
    try testing.expectEqual(@as(u8, 1), header.components[0].id);
    try testing.expectEqual(@as(u8, 2), header.components[0].h_sampling);
    try testing.expectEqual(@as(u8, 2), header.components[0].v_sampling);
}

test "JpegDecoder DQT parsing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // DQT data
    try data.append(0x00);
    try data.append(0x43); // Length = 67
    try data.append(0x00); // Precision = 8-bit, Table ID = 0

    // 64 quantization values
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        try data.append(@intCast(i + 1));
    }

    const dqt_data = try data.toOwnedSlice();
    defer allocator.free(dqt_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    try decoder.parseDQT(dqt_data);

    try testing.expectEqual(@as(u8, 0), decoder.quant_tables[0].precision);
    try testing.expectEqual(@as(u16, 1), decoder.quant_tables[0].table[0]);
    try testing.expectEqual(@as(u16, 64), decoder.quant_tables[0].table[63]);
}

test "JpegDecoder DHT parsing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // DHT data
    try data.append(0x00);
    try data.append(0x1F); // Length = 31
    try data.append(0x00); // Table class = DC, Table ID = 0

    // Bits (16 bytes)
    try data.appendSlice(&[_]u8{ 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 });

    // Values (12 bytes, total from bits array)
    try data.appendSlice(&[_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });

    const dht_data = try data.toOwnedSlice();
    defer allocator.free(dht_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    try decoder.parseDHT(dht_data);

    try testing.expectEqual(@as(u8, 0), decoder.dc_tables[0].bits[0]);
    try testing.expectEqual(@as(u8, 1), decoder.dc_tables[0].bits[1]);
    try testing.expectEqual(@as(u8, 5), decoder.dc_tables[0].bits[2]);
    try testing.expectEqual(@as(u8, 0), decoder.dc_tables[0].values[0]);
    try testing.expectEqual(@as(u8, 11), decoder.dc_tables[0].values[11]);
}

test "JpegDecoder DRI parsing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // DRI data
    try data.append(0x00);
    try data.append(0x04); // Length = 4
    try data.append(0x00);
    try data.append(0x40); // Restart interval = 64

    const dri_data = try data.toOwnedSlice();
    defer allocator.free(dri_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    try decoder.parseDRI(dri_data);

    try testing.expectEqual(@as(u16, 64), decoder.restart_interval);
}

test "JpegDecoder COM parsing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // COM data
    try data.append(0x00);
    try data.append(0x0E); // Length = 14
    try data.appendSlice("Test comment");

    const com_data = try data.toOwnedSlice();
    defer allocator.free(com_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    const comment = try decoder.parseCOM(com_data);

    try testing.expect(comment != null);
    try testing.expectEqualStrings("Test comment", comment.?);
    allocator.free(comment.?);
}

test "JpegDecoder isSOF detection" {
    const decoder = jpeg.JpegDecoder.create(testing.allocator);
    _ = decoder;

    try testing.expect(jpeg.JpegDecoder.isSOF(jpeg.MARKER_SOF0));
    try testing.expect(jpeg.JpegDecoder.isSOF(jpeg.MARKER_SOF1));
    try testing.expect(jpeg.JpegDecoder.isSOF(jpeg.MARKER_SOF2));
    try testing.expect(!jpeg.JpegDecoder.isSOF(jpeg.MARKER_SOI));
    try testing.expect(!jpeg.JpegDecoder.isSOF(jpeg.MARKER_EOI));
    try testing.expect(!jpeg.JpegDecoder.isSOF(jpeg.MARKER_DQT));
}

test "FrameHeader isProgressive" {
    const header = jpeg.FrameHeader{
        .precision = 8,
        .height = 100,
        .width = 100,
        .component_count = 3,
        .components = [_]jpeg.Component{.{
            .id = 0,
            .h_sampling = 1,
            .v_sampling = 1,
            .quant_table_id = 0,
            .dc_table_id = 0,
            .ac_table_id = 0,
        }} ** 4,
    };

    try testing.expect(header.isProgressive(jpeg.MARKER_SOF2));
    try testing.expect(!header.isProgressive(jpeg.MARKER_SOF0));
}

test "Component structure" {
    const comp = jpeg.Component{
        .id = 1,
        .h_sampling = 2,
        .v_sampling = 2,
        .quant_table_id = 0,
        .dc_table_id = 0,
        .ac_table_id = 1,
    };

    try testing.expectEqual(@as(u8, 1), comp.id);
    try testing.expectEqual(@as(u8, 2), comp.h_sampling);
    try testing.expectEqual(@as(u8, 2), comp.v_sampling);
}

test "FFI exports" {
    const allocator = testing.allocator;
    const jpeg_data = try createMinimalJPEG(allocator, 16, 16);
    defer allocator.free(jpeg_data);

    // Test decode
    const image = jpeg.nExtract_JPEG_decode(jpeg_data.ptr, jpeg_data.len);
    try testing.expect(image != null);
    defer jpeg.nExtract_JPEG_destroy(image.?);

    // Test getters
    try testing.expectEqual(@as(u16, 16), jpeg.nExtract_JPEG_getWidth(image.?));
    try testing.expectEqual(@as(u16, 16), jpeg.nExtract_JPEG_getHeight(image.?));

    const pixels = jpeg.nExtract_JPEG_getPixels(image.?);
    try testing.expect(pixels != undefined);
}

test "JpegDecoder with multiple quantization tables" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // DQT with two tables
    try data.append(0x00);
    try data.append(0x84); // Length = 132 (2 + 65 + 65)
    
    // Table 0
    try data.append(0x00); // Precision = 8-bit, Table ID = 0
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        try data.append(@intCast(i + 1));
    }
    
    // Table 1
    try data.append(0x01); // Precision = 8-bit, Table ID = 1
    i = 0;
    while (i < 64) : (i += 1) {
        try data.append(@intCast(64 - i));
    }

    const dqt_data = try data.toOwnedSlice();
    defer allocator.free(dqt_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    try decoder.parseDQT(dqt_data);

    try testing.expectEqual(@as(u16, 1), decoder.quant_tables[0].table[0]);
    try testing.expectEqual(@as(u16, 64), decoder.quant_tables[1].table[0]);
}

test "JpegDecoder missing frame header" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // SOI
    try data.append(0xFF);
    try data.append(0xD8);
    
    // SOS without SOF
    try data.append(0xFF);
    try data.append(0xDA);
    try data.append(0x00);
    try data.append(0x08);
    try data.append(0x01);
    try data.append(0x01);
    try data.append(0x00);
    try data.append(0x00);
    try data.append(0x3F);
    try data.append(0x00);

    const jpeg_data = try data.toOwnedSlice();
    defer allocator.free(jpeg_data);

    var decoder = jpeg.JpegDecoder.create(allocator);
    const result = decoder.decode(jpeg_data);
    try testing.expectError(error.MissingFrameHeader, result);
}

// Day 23 specific tests - IDCT, YCbCr, Huffman decoding

test "YCbCr to RGB conversion" {
    // Test pure white
    {
        const rgb = jpeg.ycbcrToRGB(255, 128, 128);
        try testing.expectEqual(@as(u8, 255), rgb[0]);
        try testing.expectEqual(@as(u8, 255), rgb[1]);
        try testing.expectEqual(@as(u8, 255), rgb[2]);
    }
    
    // Test pure black
    {
        const rgb = jpeg.ycbcrToRGB(0, 128, 128);
        try testing.expectEqual(@as(u8, 0), rgb[0]);
        try testing.expectEqual(@as(u8, 0), rgb[1]);
        try testing.expectEqual(@as(u8, 0), rgb[2]);
    }
    
    // Test red (approximate)
    {
        const rgb = jpeg.ycbcrToRGB(76, 85, 255);
        // Should be close to red
        try testing.expect(rgb[0] > 200); // High red
        try testing.expect(rgb[1] < 100); // Low green
        try testing.expect(rgb[2] < 100); // Low blue
    }
    
    // Test gray
    {
        const rgb = jpeg.ycbcrToRGB(128, 128, 128);
        try testing.expectEqual(@as(u8, 128), rgb[0]);
        try testing.expectEqual(@as(u8, 128), rgb[1]);
        try testing.expectEqual(@as(u8, 128), rgb[2]);
    }
}

test "Clamp to byte function" {
    try testing.expectEqual(@as(u8, 0), jpeg.clampToByte(-100));
    try testing.expectEqual(@as(u8, 0), jpeg.clampToByte(-1));
    try testing.expectEqual(@as(u8, 0), jpeg.clampToByte(0));
    try testing.expectEqual(@as(u8, 128), jpeg.clampToByte(128));
    try testing.expectEqual(@as(u8, 255), jpeg.clampToByte(255));
    try testing.expectEqual(@as(u8, 255), jpeg.clampToByte(256));
    try testing.expectEqual(@as(u8, 255), jpeg.clampToByte(1000));
}

test "Zigzag scan order" {
    // Verify zigzag pattern starts correctly
    try testing.expectEqual(@as(u8, 0), jpeg.ZIGZAG[0]);
    try testing.expectEqual(@as(u8, 1), jpeg.ZIGZAG[1]);
    try testing.expectEqual(@as(u8, 8), jpeg.ZIGZAG[2]);
    try testing.expectEqual(@as(u8, 16), jpeg.ZIGZAG[3]);
    try testing.expectEqual(@as(u8, 9), jpeg.ZIGZAG[4]);
    
    // Verify zigzag ends at bottom-right
    try testing.expectEqual(@as(u8, 63), jpeg.ZIGZAG[63]);
    
    // Verify all positions 0-63 appear exactly once
    var seen = [_]bool{false} ** 64;
    for (jpeg.ZIGZAG) |pos| {
        try testing.expect(pos < 64);
        try testing.expect(!seen[pos]); // Not seen before
        seen[pos] = true;
    }
    
    // All positions should be seen
    for (seen) |s| {
        try testing.expect(s);
    }
}

test "Dequantize DCT coefficients" {
    var block: jpeg.Block = [_]i16{0} ** 64;
    var quant_table = jpeg.QuantTable.init();
    
    // Set some test values
    block[0] = 10;
    block[1] = 5;
    block[2] = -3;
    
    quant_table.table[0] = 2;
    quant_table.table[1] = 4;
    quant_table.table[2] = 3;
    
    jpeg.dequantize(&block, &quant_table);
    
    try testing.expectEqual(@as(i16, 20), block[0]); // 10 * 2
    try testing.expectEqual(@as(i16, 20), block[1]); // 5 * 4
    try testing.expectEqual(@as(i16, -9), block[2]); // -3 * 3
}

test "IDCT with zero input" {
    var input: jpeg.Block = [_]i16{0} ** 64;
    var output: jpeg.Block = undefined;
    
    jpeg.idct(&input, &output);
    
    // All zeros in should give all zeros out
    for (output) |val| {
        try testing.expectEqual(@as(i16, 0), val);
    }
}

test "IDCT with DC-only input" {
    var input: jpeg.Block = [_]i16{0} ** 64;
    input[0] = 128; // DC coefficient only
    
    var output: jpeg.Block = undefined;
    jpeg.idct(&input, &output);
    
    // DC-only should produce flat block (all same value)
    const expected = output[0];
    for (output) |val| {
        // Values should be close (within floating point error)
        const diff = @abs(val - expected);
        try testing.expect(diff <= 1);
    }
}

test "BitReader initialization" {
    const data = [_]u8{ 0xAB, 0xCD, 0xEF };
    const reader = jpeg.BitReader.init(&data);
    
    try testing.expectEqual(@as(usize, 0), reader.byte_offset);
    try testing.expectEqual(@as(u3, 0), reader.bit_offset);
}

test "BitReader read single bits" {
    const data = [_]u8{ 0b10110100 }; // Binary: 10110100
    var reader = jpeg.BitReader.init(&data);
    
    try testing.expectEqual(@as(u16, 1), try reader.readBits(1));
    try testing.expectEqual(@as(u16, 0), try reader.readBits(1));
    try testing.expectEqual(@as(u16, 1), try reader.readBits(1));
    try testing.expectEqual(@as(u16, 1), try reader.readBits(1));
    try testing.expectEqual(@as(u16, 0), try reader.readBits(1));
    try testing.expectEqual(@as(u16, 1), try reader.readBits(1));
    try testing.expectEqual(@as(u16, 0), try reader.readBits(1));
    try testing.expectEqual(@as(u16, 0), try reader.readBits(1));
}

test "BitReader read multiple bits" {
    const data = [_]u8{ 0b10110100, 0b11001010 };
    var reader = jpeg.BitReader.init(&data);
    
    try testing.expectEqual(@as(u16, 0b1011), try reader.readBits(4));
    try testing.expectEqual(@as(u16, 0b0100), try reader.readBits(4));
    try testing.expectEqual(@as(u16, 0b1100), try reader.readBits(4));
    try testing.expectEqual(@as(u16, 0b1010), try reader.readBits(4));
}

test "BitReader byte stuffing" {
    // 0xFF should be followed by 0x00 (byte stuffing)
    const data = [_]u8{ 0xFF, 0x00, 0xAB };
    var reader = jpeg.BitReader.init(&data);
    
    // Read 8 bits from 0xFF
    try testing.expectEqual(@as(u16, 0xFF), try reader.readBits(8));
    
    // Next should be 0xAB (0x00 is skipped)
    try testing.expectEqual(@as(u16, 0xAB), try reader.readBits(8));
}

test "BitReader align to byte" {
    const data = [_]u8{ 0b10110100, 0xAB };
    var reader = jpeg.BitReader.init(&data);
    
    // Read 3 bits
    _ = try reader.readBits(3);
    try testing.expectEqual(@as(u3, 3), reader.bit_offset);
    
    // Align to byte
    reader.alignToByte();
    try testing.expectEqual(@as(u3, 0), reader.bit_offset);
    try testing.expectEqual(@as(usize, 1), reader.byte_offset);
    
    // Next read should be from 0xAB
    try testing.expectEqual(@as(u16, 0xAB), try reader.readBits(8));
}

test "Huffman table building" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var decoder = jpeg.JpegDecoder.create(allocator);
    var table = jpeg.HuffmanTable.init();
    
    // Simple Huffman table: 2 symbols with 2-bit codes
    table.bits[1] = 0;
    table.bits[2] = 2; // 2 codes of length 2
    table.values[0] = 5;
    table.values[1] = 7;
    
    try decoder.buildHuffmanTable(&table);
    
    try testing.expectEqual(@as(i32, 0), table.min_code[1]); // Min code for length 2
    try testing.expectEqual(@as(i32, 1), table.max_code[1]); // Max code for length 2
    try testing.expectEqual(@as(u16, 0), table.codes[0]);
    try testing.expectEqual(@as(u16, 1), table.codes[1]);
}

test "Decode Huffman symbol" {
    var table = jpeg.HuffmanTable.init();
    
    // Build simple table: code 0 (1 bit) = symbol 5
    table.bits[1] = 1;
    table.values[0] = 5;
    table.min_code[0] = 0;
    table.max_code[0] = 0;
    table.val_offset[0] = 0;
    
    // Test data: single bit 0
    const data = [_]u8{ 0x00 };
    var reader = jpeg.BitReader.init(&data);
    
    const symbol = try jpeg.decodeHuffmanSymbol(&reader, &table);
    try testing.expectEqual(@as(u8, 5), symbol);
}

test "Chroma subsampling factors" {
    const comp_4_4_4 = jpeg.Component{
        .id = 1,
        .h_sampling = 1,
        .v_sampling = 1,
        .quant_table_id = 0,
        .dc_table_id = 0,
        .ac_table_id = 0,
    };
    try testing.expectEqual(@as(u8, 1), comp_4_4_4.h_sampling);
    try testing.expectEqual(@as(u8, 1), comp_4_4_4.v_sampling);
    
    const comp_4_2_2 = jpeg.Component{
        .id = 1,
        .h_sampling = 2,
        .v_sampling = 1,
        .quant_table_id = 0,
        .dc_table_id = 0,
        .ac_table_id = 0,
    };
    try testing.expectEqual(@as(u8, 2), comp_4_2_2.h_sampling);
    try testing.expectEqual(@as(u8, 1), comp_4_2_2.v_sampling);
    
    const comp_4_2_0 = jpeg.Component{
        .id = 1,
        .h_sampling = 2,
        .v_sampling = 2,
        .quant_table_id = 0,
        .dc_table_id = 0,
        .ac_table_id = 0,
    };
    try testing.expectEqual(@as(u8, 2), comp_4_2_0.h_sampling);
    try testing.expectEqual(@as(u8, 2), comp_4_2_0.v_sampling);
}
