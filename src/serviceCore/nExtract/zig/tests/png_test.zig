const std = @import("std");
const testing = std.testing;
const png = @import("../parsers/png.zig");

// Helper to create a minimal valid PNG file structure
fn createMinimalPNG(allocator: std.mem.Allocator, width: u32, height: u32, color_type: png.ColorType, bit_depth: u8) ![]u8 {
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // PNG Signature
    try data.appendSlice(&png.PNG_SIGNATURE);

    // IHDR chunk
    const ihdr_data = [_]u8{
        @intCast((width >> 24) & 0xFF),
        @intCast((width >> 16) & 0xFF),
        @intCast((width >> 8) & 0xFF),
        @intCast(width & 0xFF),
        @intCast((height >> 24) & 0xFF),
        @intCast((height >> 16) & 0xFF),
        @intCast((height >> 8) & 0xFF),
        @intCast(height & 0xFF),
        bit_depth,
        @intFromEnum(color_type),
        0, // compression
        0, // filter
        0, // interlace
    };

    // Write IHDR chunk
    try writeChunk(&data, "IHDR", &ihdr_data);

    // Minimal IDAT chunk (compressed data)
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);

    // IEND chunk
    try writeChunk(&data, "IEND", &[_]u8{});

    return data.toOwnedSlice();
}

fn writeChunk(data: *std.ArrayList(u8), chunk_type: *const [4]u8, chunk_data: []const u8) !void {
    // Length
    const len: u32 = @intCast(chunk_data.len);
    try data.append(@intCast((len >> 24) & 0xFF));
    try data.append(@intCast((len >> 16) & 0xFF));
    try data.append(@intCast((len >> 8) & 0xFF));
    try data.append(@intCast(len & 0xFF));

    // Type
    try data.appendSlice(chunk_type);

    // Data
    try data.appendSlice(chunk_data);

    // CRC
    var crc = std.hash.Crc32.init();
    crc.update(chunk_type);
    crc.update(chunk_data);
    const crc_val = crc.final();
    try data.append(@intCast((crc_val >> 24) & 0xFF));
    try data.append(@intCast((crc_val >> 16) & 0xFF));
    try data.append(@intCast((crc_val >> 8) & 0xFF));
    try data.append(@intCast(crc_val & 0xFF));
}

test "PNG signature validation" {
    const valid_sig = png.PNG_SIGNATURE;
    try testing.expectEqual(@as(usize, 8), valid_sig.len);
    try testing.expectEqual(@as(u8, 0x89), valid_sig[0]);
    try testing.expectEqual(@as(u8, 0x50), valid_sig[1]); // 'P'
    try testing.expectEqual(@as(u8, 0x4E), valid_sig[2]); // 'N'
    try testing.expectEqual(@as(u8, 0x47), valid_sig[3]); // 'G'
}

test "ColorType channel count" {
    try testing.expectEqual(@as(u8, 1), png.ColorType.grayscale.channelCount());
    try testing.expectEqual(@as(u8, 3), png.ColorType.rgb.channelCount());
    try testing.expectEqual(@as(u8, 1), png.ColorType.palette.channelCount());
    try testing.expectEqual(@as(u8, 2), png.ColorType.grayscale_alpha.channelCount());
    try testing.expectEqual(@as(u8, 4), png.ColorType.rgba.channelCount());
}

test "ColorType has alpha" {
    try testing.expect(!png.ColorType.grayscale.hasAlpha());
    try testing.expect(!png.ColorType.rgb.hasAlpha());
    try testing.expect(!png.ColorType.palette.hasAlpha());
    try testing.expect(png.ColorType.grayscale_alpha.hasAlpha());
    try testing.expect(png.ColorType.rgba.hasAlpha());
}

test "ImageHeader bytes per pixel calculation" {
    const header_grayscale8 = png.ImageHeader{
        .width = 100,
        .height = 100,
        .bit_depth = 8,
        .color_type = .grayscale,
        .compression_method = .deflate,
        .filter_method = .adaptive,
        .interlace_method = .none,
    };
    try testing.expectEqual(@as(u32, 1), header_grayscale8.bytesPerPixel());

    const header_rgb8 = png.ImageHeader{
        .width = 100,
        .height = 100,
        .bit_depth = 8,
        .color_type = .rgb,
        .compression_method = .deflate,
        .filter_method = .adaptive,
        .interlace_method = .none,
    };
    try testing.expectEqual(@as(u32, 3), header_rgb8.bytesPerPixel());

    const header_rgba8 = png.ImageHeader{
        .width = 100,
        .height = 100,
        .bit_depth = 8,
        .color_type = .rgba,
        .compression_method = .deflate,
        .filter_method = .adaptive,
        .interlace_method = .none,
    };
    try testing.expectEqual(@as(u32, 4), header_rgba8.bytesPerPixel());
}

test "ImageHeader scanline bytes calculation" {
    const header = png.ImageHeader{
        .width = 100,
        .height = 50,
        .bit_depth = 8,
        .color_type = .rgb,
        .compression_method = .deflate,
        .filter_method = .adaptive,
        .interlace_method = .none,
    };
    try testing.expectEqual(@as(u32, 300), header.scanlineBytes()); // 100 * 3
}

test "PngImage creation and destruction" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const header = png.ImageHeader{
        .width = 10,
        .height = 10,
        .bit_depth = 8,
        .color_type = .rgba,
        .compression_method = .deflate,
        .filter_method = .adaptive,
        .interlace_method = .none,
    };

    var image = try png.PngImage.create(allocator, header);
    defer image.deinit();

    try testing.expectEqual(@as(u32, 10), image.header.width);
    try testing.expectEqual(@as(u32, 10), image.header.height);
    try testing.expectEqual(@as(usize, 400), image.pixels.len); // 10*10*4
}

test "PngImage pixel get/set" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const header = png.ImageHeader{
        .width = 5,
        .height = 5,
        .bit_depth = 8,
        .color_type = .rgba,
        .compression_method = .deflate,
        .filter_method = .adaptive,
        .interlace_method = .none,
    };

    var image = try png.PngImage.create(allocator, header);
    defer image.deinit();

    // Set pixel at (2, 3)
    const test_pixel = [4]u8{ 255, 128, 64, 200 };
    image.setPixel(2, 3, test_pixel);

    // Get pixel back
    const retrieved = image.getPixel(2, 3);
    try testing.expect(retrieved != null);
    try testing.expectEqual(test_pixel[0], retrieved.?[0]);
    try testing.expectEqual(test_pixel[1], retrieved.?[1]);
    try testing.expectEqual(test_pixel[2], retrieved.?[2]);
    try testing.expectEqual(test_pixel[3], retrieved.?[3]);

    // Test out of bounds
    const oob = image.getPixel(10, 10);
    try testing.expect(oob == null);
}

test "PngDecoder invalid signature" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var decoder = png.PngDecoder.create(allocator);

    // Invalid signature
    const invalid_data = [_]u8{ 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
    const result = decoder.decode(&invalid_data);
    try testing.expectError(error.InvalidPngSignature, result);
}

test "PngDecoder minimal valid PNG" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const png_data = try createMinimalPNG(allocator, 4, 4, .rgb, 8);
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();

    try testing.expectEqual(@as(u32, 4), image.header.width);
    try testing.expectEqual(@as(u32, 4), image.header.height);
    try testing.expectEqual(@as(u8, 8), image.header.bit_depth);
    try testing.expectEqual(png.ColorType.rgb, image.header.color_type);
}

test "PngDecoder with palette" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // PNG Signature
    try data.appendSlice(&png.PNG_SIGNATURE);

    // IHDR chunk
    const ihdr_data = [_]u8{
        0, 0, 0, 4, // width = 4
        0, 0, 0, 4, // height = 4
        8, // bit depth
        3, // color type = palette
        0, // compression
        0, // filter
        0, // interlace
    };
    try writeChunk(&data, "IHDR", &ihdr_data);

    // PLTE chunk (3 colors)
    const plte_data = [_]u8{
        255, 0,   0,   // Red
        0,   255, 0,   // Green
        0,   0,   255, // Blue
    };
    try writeChunk(&data, "PLTE", &plte_data);

    // Minimal IDAT
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);

    // IEND
    try writeChunk(&data, "IEND", &[_]u8{});

    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();

    try testing.expect(image.palette != null);
    try testing.expectEqual(@as(usize, 3), image.palette.?.len);
    try testing.expectEqual(@as(u8, 255), image.palette.?[0].r);
    try testing.expectEqual(@as(u8, 0), image.palette.?[0].g);
    try testing.expectEqual(@as(u8, 0), image.palette.?[0].b);
}

test "PngDecoder with text chunk" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    // PNG Signature
    try data.appendSlice(&png.PNG_SIGNATURE);

    // IHDR chunk
    const ihdr_data = [_]u8{
        0, 0, 0, 2, // width = 2
        0, 0, 0, 2, // height = 2
        8, 0, 0, 0, 0,
    };
    try writeChunk(&data, "IHDR", &ihdr_data);

    // tEXt chunk
    const text_data = "Title\x00PNG Test Image";
    try writeChunk(&data, "tEXt", text_data);

    // IDAT
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);

    // IEND
    try writeChunk(&data, "IEND", &[_]u8{});

    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 1), image.text_chunks.items.len);
    try testing.expectEqualStrings("Title", image.text_chunks.items[0].keyword);
    try testing.expectEqualStrings("PNG Test Image", image.text_chunks.items[0].text);
}

test "PngDecoder with pHYs chunk" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    try data.appendSlice(&png.PNG_SIGNATURE);

    const ihdr_data = [_]u8{ 0, 0, 0, 2, 0, 0, 0, 2, 8, 0, 0, 0, 0 };
    try writeChunk(&data, "IHDR", &ihdr_data);

    // pHYs chunk: 2835 pixels per meter (72 DPI)
    const phys_data = [_]u8{
        0, 0, 0x0B, 0x13, // 2835 pixels/meter X
        0, 0, 0x0B, 0x13, // 2835 pixels/meter Y
        1, // meters
    };
    try writeChunk(&data, "pHYs", &phys_data);

    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);
    try writeChunk(&data, "IEND", &[_]u8{});

    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();

    try testing.expect(image.phys != null);
    try testing.expectEqual(@as(u32, 2835), image.phys.?.pixels_per_unit_x);
    try testing.expectEqual(@as(u32, 2835), image.phys.?.pixels_per_unit_y);
    try testing.expectEqual(@as(u8, 1), image.phys.?.unit_specifier);
}

test "PngDecoder with tIME chunk" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    try data.appendSlice(&png.PNG_SIGNATURE);

    const ihdr_data = [_]u8{ 0, 0, 0, 2, 0, 0, 0, 2, 8, 0, 0, 0, 0 };
    try writeChunk(&data, "IHDR", &ihdr_data);

    // tIME chunk: 2026-01-17 18:30:00
    const time_data = [_]u8{
        0x07, 0xEA, // 2026
        1,  // January
        17, // 17th
        18, // 18:00
        30, // 30 minutes
        0,  // 0 seconds
    };
    try writeChunk(&data, "tIME", &time_data);

    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);
    try writeChunk(&data, "IEND", &[_]u8{});

    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();

    try testing.expect(image.time != null);
    try testing.expectEqual(@as(u16, 2026), image.time.?.year);
    try testing.expectEqual(@as(u8, 1), image.time.?.month);
    try testing.expectEqual(@as(u8, 17), image.time.?.day);
    try testing.expectEqual(@as(u8, 18), image.time.?.hour);
    try testing.expectEqual(@as(u8, 30), image.time.?.minute);
    try testing.expectEqual(@as(u8, 0), image.time.?.second);
}

test "PngDecoder CRC validation" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    try data.appendSlice(&png.PNG_SIGNATURE);

    // IHDR with correct CRC
    const ihdr_data = [_]u8{ 0, 0, 0, 2, 0, 0, 0, 2, 8, 0, 0, 0, 0 };
    try writeChunk(&data, "IHDR", &ihdr_data);

    // IDAT with WRONG CRC (manually corrupted)
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    const len: u32 = @intCast(idat_data.len);
    try data.append(@intCast((len >> 24) & 0xFF));
    try data.append(@intCast((len >> 16) & 0xFF));
    try data.append(@intCast((len >> 8) & 0xFF));
    try data.append(@intCast(len & 0xFF));
    try data.appendSlice("IDAT");
    try data.appendSlice(&idat_data);
    // Wrong CRC
    try data.appendSlice(&[_]u8{ 0xFF, 0xFF, 0xFF, 0xFF });

    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    const result = decoder.decode(png_data);
    try testing.expectError(error.InvalidCRC, result);
}

test "PngDecoder missing IHDR" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    try data.appendSlice(&png.PNG_SIGNATURE);

    // Skip IHDR, go straight to IDAT
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);
    try writeChunk(&data, "IEND", &[_]u8{});

    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    const result = decoder.decode(png_data);
    try testing.expectError(error.MissingIHDR, result);
}

test "PngDecoder missing IDAT" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    try data.appendSlice(&png.PNG_SIGNATURE);

    const ihdr_data = [_]u8{ 0, 0, 0, 2, 0, 0, 0, 2, 8, 0, 0, 0, 0 };
    try writeChunk(&data, "IHDR", &ihdr_data);
    
    // Skip IDAT, go straight to IEND
    try writeChunk(&data, "IEND", &[_]u8{});

    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);

    var decoder = png.PngDecoder.create(allocator);
    const result = decoder.decode(png_data);
    try testing.expectError(error.MissingIDAT, result);
}

test "Chunk criticality detection" {
    const allocator = testing.allocator;
    
    // Critical chunk (IHDR - uppercase first letter)
    var critical_chunk = png.Chunk{
        .chunk_type = [4]u8{ 'I', 'H', 'D', 'R' },
        .data = &[_]u8{},
        .crc = 0,
        .allocator = allocator,
    };
    try testing.expect(critical_chunk.isCritical());

    // Ancillary chunk (tEXt - lowercase first letter)
    var ancillary_chunk = png.Chunk{
        .chunk_type = [4]u8{ 't', 'E', 'X', 't' },
        .data = &[_]u8{},
        .crc = 0,
        .allocator = allocator,
    };
    try testing.expect(!ancillary_chunk.isCritical());
}

test "FFI exports" {
    const allocator = testing.allocator;
    const png_data = try createMinimalPNG(allocator, 8, 8, .rgba, 8);
    defer allocator.free(png_data);

    // Test decode
    const image = png.nExtract_PNG_decode(png_data.ptr, png_data.len);
    try testing.expect(image != null);
    defer png.nExtract_PNG_destroy(image.?);

    // Test getters
    try testing.expectEqual(@as(u32, 8), png.nExtract_PNG_getWidth(image.?));
    try testing.expectEqual(@as(u32, 8), png.nExtract_PNG_getHeight(image.?));
    
    const pixels = png.nExtract_PNG_getPixels(image.?);
    try testing.expect(pixels != undefined);
}

// Day 22 specific tests - Filtering, Interlacing, Advanced Features

test "Paeth predictor function" {
    const paethPredictor = @import("../parsers/png.zig").paethPredictor;
    
    // Test basic cases
    try testing.expectEqual(@as(u8, 10), paethPredictor(10, 20, 15));
    try testing.expectEqual(@as(u8, 100), paethPredictor(100, 100, 100));
    try testing.expectEqual(@as(u8, 0), paethPredictor(0, 0, 0));
    
    // Edge cases
    try testing.expectEqual(@as(u8, 255), paethPredictor(255, 128, 64));
}

test "Scale to U8 conversion" {
    const scaleToU8 = @import("../parsers/png.zig").scaleToU8;
    
    // 1-bit: 0 or 255
    try testing.expectEqual(@as(u8, 0), scaleToU8(0, 1));
    try testing.expectEqual(@as(u8, 255), scaleToU8(1, 1));
    
    // 2-bit: 0, 85, 170, 255
    try testing.expectEqual(@as(u8, 0), scaleToU8(0, 2));
    try testing.expectEqual(@as(u8, 85), scaleToU8(1, 2));
    try testing.expectEqual(@as(u8, 170), scaleToU8(2, 2));
    try testing.expectEqual(@as(u8, 255), scaleToU8(3, 2));
    
    // 4-bit: 0 to 255 in steps of 17
    try testing.expectEqual(@as(u8, 0), scaleToU8(0, 4));
    try testing.expectEqual(@as(u8, 17), scaleToU8(1, 4));
    try testing.expectEqual(@as(u8, 255), scaleToU8(15, 4));
    
    // 8-bit: direct
    try testing.expectEqual(@as(u8, 128), scaleToU8(128, 8));
    try testing.expectEqual(@as(u8, 255), scaleToU8(255, 8));
    
    // 16-bit: high byte
    try testing.expectEqual(@as(u8, 128), scaleToU8(32768, 16));
    try testing.expectEqual(@as(u8, 255), scaleToU8(65535, 16));
}

test "Color types with different bit depths" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    // Test grayscale with 1-bit depth
    {
        const header = png.ImageHeader{
            .width = 8,
            .height = 8,
            .bit_depth = 1,
            .color_type = .grayscale,
            .compression_method = .deflate,
            .filter_method = .adaptive,
            .interlace_method = .none,
        };
        
        var image = try png.PngImage.create(allocator, header);
        defer image.deinit();
        
        try testing.expectEqual(@as(u32, 8), image.header.width);
        try testing.expectEqual(@as(u8, 1), image.header.bit_depth);
    }
    
    // Test RGB with 16-bit depth
    {
        const header = png.ImageHeader{
            .width = 4,
            .height = 4,
            .bit_depth = 16,
            .color_type = .rgb,
            .compression_method = .deflate,
            .filter_method = .adaptive,
            .interlace_method = .none,
        };
        
        var image = try png.PngImage.create(allocator, header);
        defer image.deinit();
        
        try testing.expectEqual(@as(u8, 16), image.header.bit_depth);
        try testing.expectEqual(@as(u32, 6), header.bytesPerPixel()); // 3 channels * 2 bytes
    }
}

test "Adam7 interlace method detection" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();
    
    try data.appendSlice(&png.PNG_SIGNATURE);
    
    // IHDR with Adam7 interlacing
    const ihdr_data = [_]u8{
        0, 0, 0, 8, // width = 8
        0, 0, 0, 8, // height = 8
        8, // bit depth
        0, // color type = grayscale
        0, // compression
        0, // filter
        1, // interlace = Adam7
    };
    try writeChunk(&data, "IHDR", &ihdr_data);
    
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);
    try writeChunk(&data, "IEND", &[_]u8{});
    
    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);
    
    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();
    
    try testing.expectEqual(png.InterlaceMethod.adam7, image.header.interlace_method);
}

test "Transparency chunk for grayscale" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();
    
    try data.appendSlice(&png.PNG_SIGNATURE);
    
    const ihdr_data = [_]u8{
        0, 0, 0, 4, 0, 0, 0, 4, // 4x4
        8, 0, 0, 0, 0, // 8-bit grayscale
    };
    try writeChunk(&data, "IHDR", &ihdr_data);
    
    // tRNS chunk for grayscale: gray value that should be transparent
    const trns_data = [_]u8{ 0x00, 0x80 }; // gray level 128
    try writeChunk(&data, "tRNS", &trns_data);
    
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);
    try writeChunk(&data, "IEND", &[_]u8{});
    
    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);
    
    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();
    
    try testing.expect(image.transparency != null);
    try testing.expectEqual(@as(u16, 128), image.transparency.?.grayscale);
}

test "Background color chunk for RGB" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();
    
    try data.appendSlice(&png.PNG_SIGNATURE);
    
    const ihdr_data = [_]u8{
        0, 0, 0, 4, 0, 0, 0, 4, // 4x4
        8, 2, 0, 0, 0, // 8-bit RGB
    };
    try writeChunk(&data, "IHDR", &ihdr_data);
    
    // bKGD chunk for RGB
    const bkgd_data = [_]u8{
        0xFF, 0x00, // R = 65280
        0x80, 0x00, // G = 32768
        0x40, 0x00, // B = 16384
    };
    try writeChunk(&data, "bKGD", &bkgd_data);
    
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);
    try writeChunk(&data, "IEND", &[_]u8{});
    
    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);
    
    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();
    
    try testing.expect(image.background != null);
    try testing.expectEqual(@as(u16, 65280), image.background.?.rgb.r);
    try testing.expectEqual(@as(u16, 32768), image.background.?.rgb.g);
    try testing.expectEqual(@as(u16, 16384), image.background.?.rgb.b);
}

test "Multiple text chunks" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();
    
    try data.appendSlice(&png.PNG_SIGNATURE);
    
    const ihdr_data = [_]u8{ 0, 0, 0, 2, 0, 0, 0, 2, 8, 0, 0, 0, 0 };
    try writeChunk(&data, "IHDR", &ihdr_data);
    
    // Multiple text chunks
    try writeChunk(&data, "tEXt", "Title\x00Test Image");
    try writeChunk(&data, "tEXt", "Author\x00John Doe");
    try writeChunk(&data, "tEXt", "Description\x00A test PNG image");
    
    const idat_data = [_]u8{ 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
    try writeChunk(&data, "IDAT", &idat_data);
    try writeChunk(&data, "IEND", &[_]u8{});
    
    const png_data = try data.toOwnedSlice();
    defer allocator.free(png_data);
    
    var decoder = png.PngDecoder.create(allocator);
    var image = try decoder.decode(png_data);
    defer image.deinit();
    
    try testing.expectEqual(@as(usize, 3), image.text_chunks.items.len);
    try testing.expectEqualStrings("Title", image.text_chunks.items[0].keyword);
    try testing.expectEqualStrings("Author", image.text_chunks.items[1].keyword);
    try testing.expectEqualStrings("Description", image.text_chunks.items[2].keyword);
}

test "Filter type enum" {
    try testing.expectEqual(@as(u8, 0), @intFromEnum(png.FilterType.none));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(png.FilterType.sub));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(png.FilterType.up));
    try testing.expectEqual(@as(u8, 3), @intFromEnum(png.FilterType.average));
    try testing.expectEqual(@as(u8, 4), @intFromEnum(png.FilterType.paeth));
}

test "Scanline byte calculation for various formats" {
    // 1-bit grayscale, 8 pixels wide
    {
        const header = png.ImageHeader{
            .width = 8,
            .height = 1,
            .bit_depth = 1,
            .color_type = .grayscale,
            .compression_method = .deflate,
            .filter_method = .adaptive,
            .interlace_method = .none,
        };
        try testing.expectEqual(@as(u32, 1), header.scanlineBytes()); // 8 pixels * 1 bit = 1 byte
    }
    
    // 4-bit grayscale, 10 pixels wide
    {
        const header = png.ImageHeader{
            .width = 10,
            .height = 1,
            .bit_depth = 4,
            .color_type = .grayscale,
            .compression_method = .deflate,
            .filter_method = .adaptive,
            .interlace_method = .none,
        };
        try testing.expectEqual(@as(u32, 5), header.scanlineBytes()); // 10 pixels * 4 bits = 40 bits = 5 bytes
    }
    
    // 16-bit RGB, 3 pixels wide
    {
        const header = png.ImageHeader{
            .width = 3,
            .height = 1,
            .bit_depth = 16,
            .color_type = .rgb,
            .compression_method = .deflate,
            .filter_method = .adaptive,
            .interlace_method = .none,
        };
        try testing.expectEqual(@as(u32, 18), header.scanlineBytes()); // 3 pixels * 3 channels * 2 bytes = 18 bytes
    }
}
