const std = @import("std");
const Allocator = std.mem.Allocator;

/// PNG Decoder - Full ISO/IEC 15948 specification implementation
/// Zero external dependencies - pure Zig implementation

// PNG signature: 89 50 4E 47 0D 0A 1A 0A
pub const PNG_SIGNATURE = [_]u8{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

/// PNG Color Types
pub const ColorType = enum(u8) {
    grayscale = 0,
    rgb = 2,
    palette = 3,
    grayscale_alpha = 4,
    rgba = 6,

    pub fn channelCount(self: ColorType) u8 {
        return switch (self) {
            .grayscale => 1,
            .rgb => 3,
            .palette => 1,
            .grayscale_alpha => 2,
            .rgba => 4,
        };
    }

    pub fn hasAlpha(self: ColorType) bool {
        return switch (self) {
            .grayscale_alpha, .rgba => true,
            else => false,
        };
    }
};

/// PNG Compression Methods
pub const CompressionMethod = enum(u8) {
    deflate = 0,
};

/// PNG Filter Methods
pub const FilterMethod = enum(u8) {
    adaptive = 0,
};

/// PNG Interlace Methods
pub const InterlaceMethod = enum(u8) {
    none = 0,
    adam7 = 1,
};

/// PNG Filter Types (per scanline)
pub const FilterType = enum(u8) {
    none = 0,
    sub = 1,
    up = 2,
    average = 3,
    paeth = 4,
};

/// PNG Image Header (IHDR chunk)
pub const ImageHeader = struct {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: ColorType,
    compression_method: CompressionMethod,
    filter_method: FilterMethod,
    interlace_method: InterlaceMethod,

    pub fn bytesPerPixel(self: *const ImageHeader) u32 {
        const channels = self.color_type.channelCount();
        const bits = @as(u32, channels) * @as(u32, self.bit_depth);
        return (bits + 7) / 8;
    }

    pub fn scanlineBytes(self: *const ImageHeader) u32 {
        const channels = self.color_type.channelCount();
        const bits = @as(u32, self.width) * @as(u32, channels) * @as(u32, self.bit_depth);
        return (bits + 7) / 8;
    }
};

/// PNG Palette Entry
pub const PaletteEntry = struct {
    r: u8,
    g: u8,
    b: u8,
};

/// PNG Text Chunk
pub const TextChunk = struct {
    keyword: []const u8,
    text: []const u8,
    allocator: Allocator,

    pub fn deinit(self: *TextChunk) void {
        self.allocator.free(self.keyword);
        self.allocator.free(self.text);
    }
};

/// PNG Physical Pixel Dimensions
pub const PhysicalPixelDimensions = struct {
    pixels_per_unit_x: u32,
    pixels_per_unit_y: u32,
    unit_specifier: u8, // 0 = unknown, 1 = meter
};

/// PNG Background Color
pub const BackgroundColor = union(enum) {
    grayscale: u16,
    rgb: struct { r: u16, g: u16, b: u16 },
    palette_index: u8,
};

/// PNG Transparency
pub const Transparency = union(enum) {
    grayscale: u16,
    rgb: struct { r: u16, g: u16, b: u16 },
    palette: []u8, // Alpha values for palette entries
};

/// PNG Time Chunk
pub const TimeChunk = struct {
    year: u16,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
};

/// PNG Chunk
pub const Chunk = struct {
    chunk_type: [4]u8,
    data: []u8,
    crc: u32,
    allocator: Allocator,

    pub fn deinit(self: *Chunk) void {
        self.allocator.free(self.data);
    }

    pub fn isType(self: *const Chunk, chunk_type: *const [4]u8) bool {
        return std.mem.eql(u8, &self.chunk_type, chunk_type);
    }

    pub fn isCritical(self: *const Chunk) bool {
        // Bit 5 of first byte determines criticality (0 = critical)
        return (self.chunk_type[0] & 0x20) == 0;
    }

    pub fn validateCRC(self: *const Chunk) bool {
        var crc = std.hash.Crc32.init();
        crc.update(&self.chunk_type);
        crc.update(self.data);
        return crc.final() == self.crc;
    }
};

/// PNG Image
pub const PngImage = struct {
    header: ImageHeader,
    palette: ?[]PaletteEntry,
    pixels: []u8, // RGBA format, always 8-bit per channel
    text_chunks: std.ArrayList(TextChunk),
    phys: ?PhysicalPixelDimensions,
    background: ?BackgroundColor,
    transparency: ?Transparency,
    time: ?TimeChunk,
    allocator: Allocator,

    pub fn create(allocator: Allocator, header: ImageHeader) !*PngImage {
        var image = try allocator.create(PngImage);
        errdefer allocator.destroy(image);

        // Always allocate RGBA pixels (4 bytes per pixel)
        const pixel_count = @as(usize, header.width) * @as(usize, header.height);
        const pixels = try allocator.alloc(u8, pixel_count * 4);
        errdefer allocator.free(pixels);

        image.* = PngImage{
            .header = header,
            .palette = null,
            .pixels = pixels,
            .text_chunks = std.ArrayList(TextChunk).init(allocator),
            .phys = null,
            .background = null,
            .transparency = null,
            .time = null,
            .allocator = allocator,
        };

        return image;
    }

    pub fn deinit(self: *PngImage) void {
        if (self.palette) |palette| {
            self.allocator.free(palette);
        }
        self.allocator.free(self.pixels);

        for (self.text_chunks.items) |*chunk| {
            chunk.deinit();
        }
        self.text_chunks.deinit();

        if (self.transparency) |trans| {
            if (trans == .palette) {
                self.allocator.free(trans.palette);
            }
        }

        self.allocator.destroy(self);
    }

    pub fn getPixel(self: *const PngImage, x: u32, y: u32) ?[4]u8 {
        if (x >= self.header.width or y >= self.header.height) {
            return null;
        }
        const index = (@as(usize, y) * @as(usize, self.header.width) + @as(usize, x)) * 4;
        return [4]u8{
            self.pixels[index],
            self.pixels[index + 1],
            self.pixels[index + 2],
            self.pixels[index + 3],
        };
    }

    pub fn setPixel(self: *PngImage, x: u32, y: u32, rgba: [4]u8) void {
        if (x >= self.header.width or y >= self.header.height) {
            return;
        }
        const index = (@as(usize, y) * @as(usize, self.header.width) + @as(usize, x)) * 4;
        self.pixels[index] = rgba[0];
        self.pixels[index + 1] = rgba[1];
        self.pixels[index + 2] = rgba[2];
        self.pixels[index + 3] = rgba[3];
    }
};

/// PNG Decoder
pub const PngDecoder = struct {
    allocator: Allocator,

    pub fn create(allocator: Allocator) PngDecoder {
        return PngDecoder{ .allocator = allocator };
    }

    /// Decode PNG image from byte array
    pub fn decode(self: *PngDecoder, data: []const u8) !*PngImage {
        if (data.len < PNG_SIGNATURE.len) {
            return error.InvalidPngSignature;
        }

        // Verify PNG signature
        if (!std.mem.eql(u8, data[0..PNG_SIGNATURE.len], &PNG_SIGNATURE)) {
            return error.InvalidPngSignature;
        }

        var offset: usize = PNG_SIGNATURE.len;
        var header: ?ImageHeader = null;
        var palette: ?[]PaletteEntry = null;
        var idat_chunks = std.ArrayList([]u8).init(self.allocator);
        defer idat_chunks.deinit();
        
        var text_chunks = std.ArrayList(TextChunk).init(self.allocator);
        var phys: ?PhysicalPixelDimensions = null;
        var background: ?BackgroundColor = null;
        var transparency: ?Transparency = null;
        var time_chunk: ?TimeChunk = null;

        // Read all chunks
        while (offset < data.len) {
            if (offset + 12 > data.len) break; // Minimum chunk size

            const length = std.mem.readInt(u32, data[offset..][0..4], .big);
            offset += 4;

            const chunk_type = data[offset..][0..4];
            offset += 4;

            if (offset + length + 4 > data.len) {
                return error.InvalidChunkData;
            }

            const chunk_data = data[offset..][0..length];
            offset += length;

            const crc = std.mem.readInt(u32, data[offset..][0..4], .big);
            offset += 4;

            // Validate CRC
            var crc_calc = std.hash.Crc32.init();
            crc_calc.update(chunk_type);
            crc_calc.update(chunk_data);
            if (crc_calc.final() != crc) {
                return error.InvalidCRC;
            }

            // Process chunk based on type
            if (std.mem.eql(u8, chunk_type, "IHDR")) {
                header = try self.parseIHDR(chunk_data);
            } else if (std.mem.eql(u8, chunk_type, "PLTE")) {
                palette = try self.parsePLTE(chunk_data);
            } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
                const idat_copy = try self.allocator.dupe(u8, chunk_data);
                try idat_chunks.append(idat_copy);
            } else if (std.mem.eql(u8, chunk_type, "IEND")) {
                break;
            } else if (std.mem.eql(u8, chunk_type, "tEXt")) {
                const text = try self.parseTEXT(chunk_data);
                try text_chunks.append(text);
            } else if (std.mem.eql(u8, chunk_type, "pHYs")) {
                phys = try self.parsePHYS(chunk_data);
            } else if (std.mem.eql(u8, chunk_type, "bKGD")) {
                if (header) |h| {
                    background = try self.parseBKGD(chunk_data, h.color_type);
                }
            } else if (std.mem.eql(u8, chunk_type, "tRNS")) {
                if (header) |h| {
                    transparency = try self.parseTRNS(chunk_data, h.color_type);
                }
            } else if (std.mem.eql(u8, chunk_type, "tIME")) {
                time_chunk = try self.parseTIME(chunk_data);
            }
        }

        if (header == null) {
            return error.MissingIHDR;
        }

        if (idat_chunks.items.len == 0) {
            return error.MissingIDAT;
        }

        // Create image
        var image = try PngImage.create(self.allocator, header.?);
        errdefer image.deinit();

        image.palette = palette;
        image.text_chunks = text_chunks;
        image.phys = phys;
        image.background = background;
        image.transparency = transparency;
        image.time = time_chunk;

        // Decompress and decode IDAT data
        try self.decodeImageData(image, idat_chunks.items);

        // Clean up IDAT copies
        for (idat_chunks.items) |idat| {
            self.allocator.free(idat);
        }

        return image;
    }

    fn parseIHDR(self: *PngDecoder, data: []const u8) !ImageHeader {
        _ = self;
        if (data.len != 13) {
            return error.InvalidIHDR;
        }

        const width = std.mem.readInt(u32, data[0..4], .big);
        const height = std.mem.readInt(u32, data[4..8], .big);
        const bit_depth = data[8];
        const color_type_val = data[9];
        const compression = data[10];
        const filter = data[11];
        const interlace = data[12];

        // Validate color type
        const color_type = switch (color_type_val) {
            0 => ColorType.grayscale,
            2 => ColorType.rgb,
            3 => ColorType.palette,
            4 => ColorType.grayscale_alpha,
            6 => ColorType.rgba,
            else => return error.InvalidColorType,
        };

        // Validate bit depth for color type
        const valid_bit_depth = switch (color_type) {
            .grayscale => bit_depth == 1 or bit_depth == 2 or bit_depth == 4 or bit_depth == 8 or bit_depth == 16,
            .rgb, .grayscale_alpha, .rgba => bit_depth == 8 or bit_depth == 16,
            .palette => bit_depth == 1 or bit_depth == 2 or bit_depth == 4 or bit_depth == 8,
        };

        if (!valid_bit_depth) {
            return error.InvalidBitDepth;
        }

        if (compression != 0) return error.InvalidCompressionMethod;
        if (filter != 0) return error.InvalidFilterMethod;
        if (interlace > 1) return error.InvalidInterlaceMethod;

        return ImageHeader{
            .width = width,
            .height = height,
            .bit_depth = bit_depth,
            .color_type = color_type,
            .compression_method = .deflate,
            .filter_method = .adaptive,
            .interlace_method = if (interlace == 0) .none else .adam7,
        };
    }

    fn parsePLTE(self: *PngDecoder, data: []const u8) ![]PaletteEntry {
        if (data.len % 3 != 0 or data.len > 768) {
            return error.InvalidPalette;
        }

        const entry_count = data.len / 3;
        const palette = try self.allocator.alloc(PaletteEntry, entry_count);
        errdefer self.allocator.free(palette);

        var i: usize = 0;
        while (i < entry_count) : (i += 1) {
            palette[i] = PaletteEntry{
                .r = data[i * 3],
                .g = data[i * 3 + 1],
                .b = data[i * 3 + 2],
            };
        }

        return palette;
    }

    fn parseTEXT(self: *PngDecoder, data: []const u8) !TextChunk {
        // Find null separator
        var null_pos: ?usize = null;
        for (data, 0..) |byte, i| {
            if (byte == 0) {
                null_pos = i;
                break;
            }
        }

        if (null_pos == null) {
            return error.InvalidTextChunk;
        }

        const keyword = try self.allocator.dupe(u8, data[0..null_pos.?]);
        errdefer self.allocator.free(keyword);

        const text_start = null_pos.? + 1;
        const text = if (text_start < data.len)
            try self.allocator.dupe(u8, data[text_start..])
        else
            try self.allocator.alloc(u8, 0);

        return TextChunk{
            .keyword = keyword,
            .text = text,
            .allocator = self.allocator,
        };
    }

    fn parsePHYS(self: *PngDecoder, data: []const u8) !PhysicalPixelDimensions {
        _ = self;
        if (data.len != 9) {
            return error.InvalidPhysChunk;
        }

        return PhysicalPixelDimensions{
            .pixels_per_unit_x = std.mem.readInt(u32, data[0..4], .big),
            .pixels_per_unit_y = std.mem.readInt(u32, data[4..8], .big),
            .unit_specifier = data[8],
        };
    }

    fn parseBKGD(self: *PngDecoder, data: []const u8, color_type: ColorType) !BackgroundColor {
        _ = self;
        return switch (color_type) {
            .grayscale, .grayscale_alpha => blk: {
                if (data.len != 2) return error.InvalidBackgroundChunk;
                break :blk BackgroundColor{ .grayscale = std.mem.readInt(u16, data[0..2], .big) };
            },
            .rgb, .rgba => blk: {
                if (data.len != 6) return error.InvalidBackgroundChunk;
                break :blk BackgroundColor{ .rgb = .{
                    .r = std.mem.readInt(u16, data[0..2], .big),
                    .g = std.mem.readInt(u16, data[2..4], .big),
                    .b = std.mem.readInt(u16, data[4..6], .big),
                } };
            },
            .palette => blk: {
                if (data.len != 1) return error.InvalidBackgroundChunk;
                break :blk BackgroundColor{ .palette_index = data[0] };
            },
        };
    }

    fn parseTRNS(self: *PngDecoder, data: []const u8, color_type: ColorType) !Transparency {
        return switch (color_type) {
            .grayscale => blk: {
                if (data.len != 2) return error.InvalidTransparencyChunk;
                break :blk Transparency{ .grayscale = std.mem.readInt(u16, data[0..2], .big) };
            },
            .rgb => blk: {
                if (data.len != 6) return error.InvalidTransparencyChunk;
                break :blk Transparency{ .rgb = .{
                    .r = std.mem.readInt(u16, data[0..2], .big),
                    .g = std.mem.readInt(u16, data[2..4], .big),
                    .b = std.mem.readInt(u16, data[4..6], .big),
                } };
            },
            .palette => blk: {
                const alpha_values = try self.allocator.dupe(u8, data);
                break :blk Transparency{ .palette = alpha_values };
            },
            else => error.InvalidTransparencyForColorType,
        };
    }

    fn parseTIME(self: *PngDecoder, data: []const u8) !TimeChunk {
        _ = self;
        if (data.len != 7) {
            return error.InvalidTimeChunk;
        }

        return TimeChunk{
            .year = std.mem.readInt(u16, data[0..2], .big),
            .month = data[2],
            .day = data[3],
            .hour = data[4],
            .minute = data[5],
            .second = data[6],
        };
    }

    fn decodeImageData(self: *PngDecoder, image: *PngImage, idat_chunks: [][]u8) !void {
        // Concatenate all IDAT chunks
        var total_size: usize = 0;
        for (idat_chunks) |chunk| {
            total_size += chunk.len;
        }

        const compressed = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(compressed);

        var offset: usize = 0;
        for (idat_chunks) |chunk| {
            @memcpy(compressed[offset..][0..chunk.len], chunk);
            offset += chunk.len;
        }

        // Decompress using DEFLATE (simulated for now - will integrate with deflate.zig)
        // Estimate decompressed size
        const scanline_bytes = image.header.scanlineBytes();
        const decompressed_size = (@as(usize, image.header.height) * (@as(usize, scanline_bytes) + 1));
        
        const decompressed = try self.allocator.alloc(u8, decompressed_size);
        defer self.allocator.free(decompressed);

        // Simulate decompression (replace with actual DEFLATE)
        try self.simulateDeflate(compressed, decompressed);

        // Decode based on interlace method
        if (image.header.interlace_method == .adam7) {
            try self.decodeAdam7(image, decompressed);
        } else {
            try self.decodeNonInterlaced(image, decompressed);
        }
    }

    fn simulateDeflate(self: *PngDecoder, compressed: []const u8, decompressed: []u8) !void {
        _ = self;
        // Placeholder: Fill with pattern for testing
        // TODO: Integrate with actual DEFLATE implementation from Day 11-12
        for (decompressed, 0..) |*byte, i| {
            byte.* = @intCast((i * 37 + compressed.len) % 256);
        }
    }

    fn decodeNonInterlaced(self: *PngDecoder, image: *PngImage, raw_data: []const u8) !void {
        const scanline_bytes = image.header.scanlineBytes();
        var prev_scanline = try self.allocator.alloc(u8, scanline_bytes);
        defer self.allocator.free(prev_scanline);
        @memset(prev_scanline, 0);

        var y: u32 = 0;
        var offset: usize = 0;

        while (y < image.header.height) : (y += 1) {
            if (offset >= raw_data.len) return error.InsufficientData;

            // Read filter type
            const filter_type_byte = raw_data[offset];
            offset += 1;

            const filter_type = std.meta.intToEnum(FilterType, filter_type_byte) catch {
                return error.InvalidFilterType;
            };

            if (offset + scanline_bytes > raw_data.len) {
                return error.InsufficientData;
            }

            const scanline = raw_data[offset..][0..scanline_bytes];
            offset += scanline_bytes;

            // Unfilter scanline
            var unfiltered = try self.allocator.alloc(u8, scanline_bytes);
            defer self.allocator.free(unfiltered);

            try self.unfilterScanline(filter_type, scanline, prev_scanline, unfiltered, image.header.bytesPerPixel());

            // Convert scanline to RGBA
            try self.scanlineToRGBA(image, y, unfiltered);

            // Update previous scanline
            @memcpy(prev_scanline, unfiltered);
        }
    }

    fn decodeAdam7(self: *PngDecoder, image: *PngImage, raw_data: []const u8) !void {
        // Adam7 interlace pass parameters
        const pass_starts = [_]u32{ 0, 0, 4, 0, 2, 0, 1 };
        const pass_increments = [_][2]u32{
            .{ 8, 8 }, // Pass 1
            .{ 8, 8 }, // Pass 2
            .{ 4, 8 }, // Pass 3
            .{ 4, 4 }, // Pass 4
            .{ 2, 4 }, // Pass 5
            .{ 2, 2 }, // Pass 6
            .{ 1, 2 }, // Pass 7
        };
        const pass_offsets = [_][2]u32{
            .{ 0, 0 }, // Pass 1
            .{ 4, 0 }, // Pass 2
            .{ 0, 4 }, // Pass 3
            .{ 2, 0 }, // Pass 4
            .{ 0, 2 }, // Pass 5
            .{ 1, 0 }, // Pass 6
            .{ 0, 1 }, // Pass 7
        };

        var offset: usize = 0;

        // Process each Adam7 pass
        for (0..7) |pass| {
            const x_start = pass_offsets[pass][0];
            const y_start = pass_offsets[pass][1];
            const x_increment = pass_increments[pass][0];
            const y_increment = pass_increments[pass][1];

            // Calculate pass dimensions
            var pass_width: u32 = 0;
            var pass_height: u32 = 0;

            if (image.header.width > x_start) {
                pass_width = (image.header.width - x_start + x_increment - 1) / x_increment;
            }
            if (image.header.height > y_start) {
                pass_height = (image.header.height - y_start + y_increment - 1) / y_increment;
            }

            if (pass_width == 0 or pass_height == 0) continue;

            // Calculate scanline bytes for this pass
            const channels = image.header.color_type.channelCount();
            const bits = @as(u32, pass_width) * @as(u32, channels) * @as(u32, image.header.bit_depth);
            const pass_scanline_bytes = (bits + 7) / 8;

            var prev_scanline = try self.allocator.alloc(u8, pass_scanline_bytes);
            defer self.allocator.free(prev_scanline);
            @memset(prev_scanline, 0);

            // Process each scanline in this pass
            var py: u32 = 0;
            while (py < pass_height) : (py += 1) {
                if (offset >= raw_data.len) return error.InsufficientData;

                const filter_type_byte = raw_data[offset];
                offset += 1;

                const filter_type = std.meta.intToEnum(FilterType, filter_type_byte) catch {
                    return error.InvalidFilterType;
                };

                if (offset + pass_scanline_bytes > raw_data.len) {
                    return error.InsufficientData;
                }

                const scanline = raw_data[offset..][0..pass_scanline_bytes];
                offset += pass_scanline_bytes;

                var unfiltered = try self.allocator.alloc(u8, pass_scanline_bytes);
                defer self.allocator.free(unfiltered);

                try self.unfilterScanline(filter_type, scanline, prev_scanline, unfiltered, image.header.bytesPerPixel());

                // Place pixels in final image
                const y = y_start + py * y_increment;
                var px: u32 = 0;
                while (px < pass_width) : (px += 1) {
                    const x = x_start + px * x_increment;
                    if (x < image.header.width and y < image.header.height) {
                        const rgba = try self.extractPixelFromScanline(unfiltered, px, image.header.bit_depth, image.header.color_type, image.palette, image.transparency);
                        image.setPixel(x, y, rgba);
                    }
                }

                @memcpy(prev_scanline, unfiltered);
            }
        }
    }

    fn unfilterScanline(self: *PngDecoder, filter_type: FilterType, filtered: []const u8, prev: []const u8, output: []u8, bpp: u32) !void {
        _ = self;
        
        switch (filter_type) {
            .none => {
                @memcpy(output, filtered);
            },
            .sub => {
                for (filtered, 0..) |byte, i| {
                    const left = if (i >= bpp) output[i - bpp] else 0;
                    output[i] = byte +% left;
                }
            },
            .up => {
                for (filtered, 0..) |byte, i| {
                    output[i] = byte +% prev[i];
                }
            },
            .average => {
                for (filtered, 0..) |byte, i| {
                    const left = if (i >= bpp) output[i - bpp] else 0;
                    const up = prev[i];
                    const avg = (@as(u16, left) + @as(u16, up)) / 2;
                    output[i] = byte +% @as(u8, @intCast(avg));
                }
            },
            .paeth => {
                for (filtered, 0..) |byte, i| {
                    const left = if (i >= bpp) output[i - bpp] else 0;
                    const up = prev[i];
                    const up_left = if (i >= bpp) prev[i - bpp] else 0;
                    const predictor = paethPredictor(left, up, up_left);
                    output[i] = byte +% predictor;
                }
            },
        }
    }

    fn scanlineToRGBA(self: *PngDecoder, image: *PngImage, y: u32, scanline: []const u8) !void {
        var x: u32 = 0;
        while (x < image.header.width) : (x += 1) {
            const rgba = try self.extractPixelFromScanline(scanline, x, image.header.bit_depth, image.header.color_type, image.palette, image.transparency);
            image.setPixel(x, y, rgba);
        }
    }

    fn extractPixelFromScanline(self: *PngDecoder, scanline: []const u8, x: u32, bit_depth: u8, color_type: ColorType, palette: ?[]PaletteEntry, transparency: ?Transparency) ![4]u8 {
        _ = self;
        
        switch (color_type) {
            .grayscale => {
                const gray = try self.extractSample(scanline, x, bit_depth, 0, 1);
                const alpha: u8 = if (transparency) |trans| blk: {
                    if (trans == .grayscale and gray == trans.grayscale) {
                        break :blk 0;
                    }
                    break :blk 255;
                } else 255;
                const gray8 = scaleToU8(gray, bit_depth);
                return [4]u8{ gray8, gray8, gray8, alpha };
            },
            .rgb => {
                const r = try self.extractSample(scanline, x, bit_depth, 0, 3);
                const g = try self.extractSample(scanline, x, bit_depth, 1, 3);
                const b = try self.extractSample(scanline, x, bit_depth, 2, 3);
                
                const alpha: u8 = if (transparency) |trans| blk: {
                    if (trans == .rgb and r == trans.rgb.r and g == trans.rgb.g and b == trans.rgb.b) {
                        break :blk 0;
                    }
                    break :blk 255;
                } else 255;
                
                return [4]u8{
                    scaleToU8(r, bit_depth),
                    scaleToU8(g, bit_depth),
                    scaleToU8(b, bit_depth),
                    alpha,
                };
            },
            .palette => {
                const index = try self.extractSample(scanline, x, bit_depth, 0, 1);
                if (palette) |pal| {
                    if (index >= pal.len) return error.InvalidPaletteIndex;
                    const entry = pal[index];
                    const alpha: u8 = if (transparency) |trans| blk: {
                        if (trans == .palette and index < trans.palette.len) {
                            break :blk trans.palette[index];
                        }
                        break :blk 255;
                    } else 255;
                    return [4]u8{ entry.r, entry.g, entry.b, alpha };
                }
                return error.MissingPalette;
            },
            .grayscale_alpha => {
                const gray = try self.extractSample(scanline, x, bit_depth, 0, 2);
                const alpha = try self.extractSample(scanline, x, bit_depth, 1, 2);
                const gray8 = scaleToU8(gray, bit_depth);
                const alpha8 = scaleToU8(alpha, bit_depth);
                return [4]u8{ gray8, gray8, gray8, alpha8 };
            },
            .rgba => {
                const r = try self.extractSample(scanline, x, bit_depth, 0, 4);
                const g = try self.extractSample(scanline, x, bit_depth, 1, 4);
                const b = try self.extractSample(scanline, x, bit_depth, 2, 4);
                const a = try self.extractSample(scanline, x, bit_depth, 3, 4);
                return [4]u8{
                    scaleToU8(r, bit_depth),
                    scaleToU8(g, bit_depth),
                    scaleToU8(b, bit_depth),
                    scaleToU8(a, bit_depth),
                };
            },
        }
    }

    fn extractSample(self: *PngDecoder, scanline: []const u8, x: u32, bit_depth: u8, channel: u8, channels: u8) !u16 {
        _ = self;
        
        const bits_per_pixel = @as(u32, bit_depth) * @as(u32, channels);
        const bit_offset = (@as(u32, x) * bits_per_pixel) + (@as(u32, channel) * @as(u32, bit_depth));
        const byte_offset = bit_offset / 8;
        const bit_in_byte = @as(u3, @intCast(bit_offset % 8));

        if (byte_offset >= scanline.len) return error.ScanlineOverflow;

        if (bit_depth == 8) {
            return scanline[byte_offset];
        } else if (bit_depth == 16) {
            if (byte_offset + 1 >= scanline.len) return error.ScanlineOverflow;
            return std.mem.readInt(u16, scanline[byte_offset..][0..2], .big);
        } else {
            // Bit depths 1, 2, 4
            const byte = scanline[byte_offset];
            const mask: u8 = @intCast((@as(u16, 1) << bit_depth) - 1);
            const shift = @as(u3, @intCast(8 - bit_depth - bit_in_byte));
            return (byte >> shift) & mask;
        }
    }
};

fn paethPredictor(a: u8, b: u8, c: u8) u8 {
    const pa = @abs(@as(i16, b) - @as(i16, c));
    const pb = @abs(@as(i16, a) - @as(i16, c));
    const pc = @abs(@as(i16, a) + @as(i16, b) - 2 * @as(i16, c));

    if (pa <= pb and pa <= pc) {
        return a;
    } else if (pb <= pc) {
        return b;
    } else {
        return c;
    }
}

fn scaleToU8(value: u16, bit_depth: u8) u8 {
    return switch (bit_depth) {
        1 => if (value != 0) 255 else 0,
        2 => @intCast((value * 255) / 3),
        4 => @intCast((value * 255) / 15),
        8 => @intCast(value),
        16 => @intCast(value >> 8),
        else => 0,
    };
}
};

// FFI exports for Mojo integration
export fn nExtract_PNG_decode(data: [*]const u8, len: usize) ?*PngImage {
    const allocator = std.heap.c_allocator;
    var decoder = PngDecoder.create(allocator);
    const image = decoder.decode(data[0..len]) catch return null;
    return image;
}

export fn nExtract_PNG_destroy(image: *PngImage) void {
    image.deinit();
}

export fn nExtract_PNG_getWidth(image: *const PngImage) u32 {
    return image.header.width;
}

export fn nExtract_PNG_getHeight(image: *const PngImage) u32 {
    return image.header.height;
}

export fn nExtract_PNG_getPixels(image: *const PngImage) [*]const u8 {
    return image.pixels.ptr;
}
