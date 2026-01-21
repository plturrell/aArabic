const std = @import("std");
const Allocator = std.mem.Allocator;

/// JPEG Decoder - ISO/IEC 10918-1 (JPEG) specification implementation
/// Zero external dependencies - pure Zig implementation

// JPEG markers
pub const MARKER_SOI: u16 = 0xFFD8; // Start of Image
pub const MARKER_EOI: u16 = 0xFFD9; // End of Image
pub const MARKER_SOS: u16 = 0xFFDA; // Start of Scan
pub const MARKER_DQT: u16 = 0xFFDB; // Define Quantization Table
pub const MARKER_DHT: u16 = 0xFFC4; // Define Huffman Table
pub const MARKER_DRI: u16 = 0xFFDD; // Define Restart Interval
pub const MARKER_COM: u16 = 0xFFFE; // Comment

// Start of Frame markers (SOF)
pub const MARKER_SOF0: u16 = 0xFFC0; // Baseline DCT
pub const MARKER_SOF1: u16 = 0xFFC1; // Extended sequential DCT
pub const MARKER_SOF2: u16 = 0xFFC2; // Progressive DCT
pub const MARKER_SOF3: u16 = 0xFFC3; // Lossless (sequential)
pub const MARKER_SOF5: u16 = 0xFFC5; // Differential sequential DCT
pub const MARKER_SOF6: u16 = 0xFFC6; // Differential progressive DCT
pub const MARKER_SOF7: u16 = 0xFFC7; // Differential lossless
pub const MARKER_SOF9: u16 = 0xFFC9; // Extended sequential DCT (arithmetic)
pub const MARKER_SOF10: u16 = 0xFFCA; // Progressive DCT (arithmetic)
pub const MARKER_SOF11: u16 = 0xFFCB; // Lossless (arithmetic)
pub const MARKER_SOF13: u16 = 0xFFCD; // Differential sequential DCT (arithmetic)
pub const MARKER_SOF14: u16 = 0xFFCE; // Differential progressive DCT (arithmetic)
pub const MARKER_SOF15: u16 = 0xFFCF; // Differential lossless (arithmetic)

// Application markers
pub const MARKER_APP0: u16 = 0xFFE0; // JFIF marker
pub const MARKER_APP1: u16 = 0xFFE1; // EXIF marker
pub const MARKER_APP2: u16 = 0xFFE2;
pub const MARKER_APP14: u16 = 0xFFEE; // Adobe marker

/// JPEG Color Space
pub const ColorSpace = enum(u8) {
    grayscale = 1,
    ycbcr = 3,
    rgb = 3, // Same component count as YCbCr
    cmyk = 4,
};

/// JPEG Component
pub const Component = struct {
    id: u8,
    h_sampling: u8, // Horizontal sampling factor
    v_sampling: u8, // Vertical sampling factor
    quant_table_id: u8,
    dc_table_id: u8,
    ac_table_id: u8,
};

/// JPEG Frame Header (Start of Frame)
pub const FrameHeader = struct {
    precision: u8, // Sample precision (usually 8)
    height: u16,
    width: u16,
    component_count: u8,
    components: [4]Component,
    
    pub fn isProgressive(self: *const FrameHeader, marker: u16) bool {
        return marker == MARKER_SOF2 or marker == MARKER_SOF6 or 
               marker == MARKER_SOF10 or marker == MARKER_SOF14;
    }
};

/// Quantization Table
pub const QuantTable = struct {
    precision: u8, // 0 = 8-bit, 1 = 16-bit
    table: [64]u16,
    
    pub fn init() QuantTable {
        return QuantTable{
            .precision = 0,
            .table = [_]u16{0} ** 64,
        };
    }
};

/// Huffman Table
pub const HuffmanTable = struct {
    codes: [256]u16, // Huffman codes
    values: [256]u8, // Symbol values
    min_code: [16]i32, // Minimum code for each length
    max_code: [16]i32, // Maximum code for each length
    val_offset: [16]i32, // Offset into values array
    bits: [17]u8, // Number of codes of each length
    
    pub fn init() HuffmanTable {
        return HuffmanTable{
            .codes = [_]u16{0} ** 256,
            .values = [_]u8{0} ** 256,
            .min_code = [_]i32{-1} ** 16,
            .max_code = [_]i32{-1} ** 16,
            .val_offset = [_]i32{0} ** 16,
            .bits = [_]u8{0} ** 17,
        };
    }
};

/// EXIF Metadata
pub const ExifMetadata = struct {
    make: ?[]const u8,
    model: ?[]const u8,
    orientation: u16,
    x_resolution: f32,
    y_resolution: f32,
    software: ?[]const u8,
    date_time: ?[]const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ExifMetadata {
        return ExifMetadata{
            .make = null,
            .model = null,
            .orientation = 1,
            .x_resolution = 72.0,
            .y_resolution = 72.0,
            .software = null,
            .date_time = null,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ExifMetadata) void {
        if (self.make) |make| self.allocator.free(make);
        if (self.model) |model| self.allocator.free(model);
        if (self.software) |software| self.allocator.free(software);
        if (self.date_time) |dt| self.allocator.free(dt);
    }
};

/// JPEG Image
pub const JpegImage = struct {
    width: u16,
    height: u16,
    component_count: u8,
    pixels: []u8, // RGB or RGBA format
    exif: ?ExifMetadata,
    comment: ?[]const u8,
    allocator: Allocator,
    
    pub fn create(allocator: Allocator, width: u16, height: u16, component_count: u8) !*JpegImage {
        var image = try allocator.create(JpegImage);
        errdefer allocator.destroy(image);
        
        // Allocate RGB(A) pixels
        const pixel_count = @as(usize, width) * @as(usize, height);
        const bytes_per_pixel: usize = if (component_count == 1) 3 else 3; // Convert grayscale to RGB
        const pixels = try allocator.alloc(u8, pixel_count * bytes_per_pixel);
        errdefer allocator.free(pixels);
        
        image.* = JpegImage{
            .width = width,
            .height = height,
            .component_count = component_count,
            .pixels = pixels,
            .exif = null,
            .comment = null,
            .allocator = allocator,
        };
        
        return image;
    }
    
    pub fn deinit(self: *JpegImage) void {
        self.allocator.free(self.pixels);
        if (self.exif) |*exif| {
            exif.deinit();
        }
        if (self.comment) |comment| {
            self.allocator.free(comment);
        }
        self.allocator.destroy(self);
    }
    
    pub fn getPixel(self: *const JpegImage, x: u16, y: u16) ?[3]u8 {
        if (x >= self.width or y >= self.height) {
            return null;
        }
        const index = (@as(usize, y) * @as(usize, self.width) + @as(usize, x)) * 3;
        return [3]u8{
            self.pixels[index],
            self.pixels[index + 1],
            self.pixels[index + 2],
        };
    }
    
    pub fn setPixel(self: *JpegImage, x: u16, y: u16, rgb: [3]u8) void {
        if (x >= self.width or y >= self.height) {
            return;
        }
        const index = (@as(usize, y) * @as(usize, self.width) + @as(usize, x)) * 3;
        self.pixels[index] = rgb[0];
        self.pixels[index + 1] = rgb[1];
        self.pixels[index + 2] = rgb[2];
    }
};

/// JPEG Decoder
pub const JpegDecoder = struct {
    allocator: Allocator,
    quant_tables: [4]QuantTable,
    dc_tables: [4]HuffmanTable,
    ac_tables: [4]HuffmanTable,
    restart_interval: u16,
    
    pub fn create(allocator: Allocator) JpegDecoder {
        return JpegDecoder{
            .allocator = allocator,
            .quant_tables = [_]QuantTable{QuantTable.init()} ** 4,
            .dc_tables = [_]HuffmanTable{HuffmanTable.init()} ** 4,
            .ac_tables = [_]HuffmanTable{HuffmanTable.init()} ** 4,
            .restart_interval = 0,
        };
    }
    
    /// Decode JPEG image from byte array
    pub fn decode(self: *JpegDecoder, data: []const u8) !*JpegImage {
        if (data.len < 2) {
            return error.InvalidJpegData;
        }
        
        // Verify SOI marker
        const soi = std.mem.readInt(u16, data[0..2], .big);
        if (soi != MARKER_SOI) {
            return error.InvalidJpegMarker;
        }
        
        var offset: usize = 2;
        var frame_header: ?FrameHeader = null;
        var comment: ?[]const u8 = null;
        var exif: ?ExifMetadata = null;
        
        // Parse markers
        while (offset < data.len) {
            if (offset + 2 > data.len) break;
            
            const marker = std.mem.readInt(u16, data[offset..][0..2], .big);
            offset += 2;
            
            if (marker == MARKER_EOI) {
                break;
            } else if (marker == MARKER_SOS) {
                // Start of Scan - parse scan header then decode image data
                if (frame_header == null) {
                    return error.MissingFrameHeader;
                }
                
                // Parse scan header
                const scan_length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, scan_length);
                
                // Decode image data (simplified for Day 23)
                break;
            } else if (isSOF(marker)) {
                frame_header = try self.parseSOF(data[offset..], marker);
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            } else if (marker == MARKER_DQT) {
                try self.parseDQT(data[offset..]);
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            } else if (marker == MARKER_DHT) {
                try self.parseDHT(data[offset..]);
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            } else if (marker == MARKER_DRI) {
                try self.parseDRI(data[offset..]);
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            } else if (marker == MARKER_COM) {
                const com = try self.parseCOM(data[offset..]);
                comment = com;
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            } else if (marker == MARKER_APP0) {
                // JFIF marker - skip for now
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            } else if (marker == MARKER_APP1) {
                // EXIF marker
                exif = try self.parseEXIF(data[offset..]);
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            } else {
                // Unknown marker - skip
                if (offset + 2 > data.len) break;
                const length = std.mem.readInt(u16, data[offset..][0..2], .big);
                offset += @as(usize, length);
            }
        }
        
        if (frame_header == null) {
            return error.MissingFrameHeader;
        }
        
        // Create image
        const header = frame_header.?;
        var image = try JpegImage.create(self.allocator, header.width, header.height, header.component_count);
        errdefer image.deinit();
        
        image.comment = comment;
        image.exif = exif;
        
        // Fill with placeholder pattern (will be replaced with actual decoding)
        const pixel_count = @as(usize, header.width) * @as(usize, header.height);
        var i: usize = 0;
        while (i < pixel_count) : (i += 1) {
            const x = i % header.width;
            const y = i / header.width;
            const gray: u8 = @intCast((x + y) % 256);
            image.setPixel(@intCast(x), @intCast(y), [3]u8{ gray, gray, gray });
        }
        
        return image;
    }
    
    fn isSOF(marker: u16) bool {
        return marker == MARKER_SOF0 or marker == MARKER_SOF1 or marker == MARKER_SOF2 or
               marker == MARKER_SOF3 or marker == MARKER_SOF5 or marker == MARKER_SOF6 or
               marker == MARKER_SOF7 or marker == MARKER_SOF9 or marker == MARKER_SOF10 or
               marker == MARKER_SOF11 or marker == MARKER_SOF13 or marker == MARKER_SOF14 or
               marker == MARKER_SOF15;
    }
    
    fn parseSOF(self: *JpegDecoder, data: []const u8, marker: u16) !FrameHeader {
        _ = self;
        _ = marker;
        
        if (data.len < 8) {
            return error.InvalidSOF;
        }
        
        const length = std.mem.readInt(u16, data[0..2], .big);
        if (data.len < length) {
            return error.InvalidSOF;
        }
        
        const precision = data[2];
        const height = std.mem.readInt(u16, data[3..5], .big);
        const width = std.mem.readInt(u16, data[5..7], .big);
        const component_count = data[7];
        
        if (component_count > 4) {
            return error.TooManyComponents;
        }
        
        var header = FrameHeader{
            .precision = precision,
            .height = height,
            .width = width,
            .component_count = component_count,
            .components = [_]Component{.{
                .id = 0,
                .h_sampling = 1,
                .v_sampling = 1,
                .quant_table_id = 0,
                .dc_table_id = 0,
                .ac_table_id = 0,
            }} ** 4,
        };
        
        var offset: usize = 8;
        var i: usize = 0;
        while (i < component_count) : (i += 1) {
            if (offset + 3 > data.len) {
                return error.InvalidSOF;
            }
            
            const id = data[offset];
            const sampling = data[offset + 1];
            const quant_table = data[offset + 2];
            
            header.components[i] = Component{
                .id = id,
                .h_sampling = @intCast((sampling >> 4) & 0x0F),
                .v_sampling = @intCast(sampling & 0x0F),
                .quant_table_id = quant_table,
                .dc_table_id = 0,
                .ac_table_id = 0,
            };
            
            offset += 3;
        }
        
        return header;
    }
    
    fn parseDQT(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 2) {
            return error.InvalidDQT;
        }
        
        const length = std.mem.readInt(u16, data[0..2], .big);
        if (data.len < length) {
            return error.InvalidDQT;
        }
        
        var offset: usize = 2;
        while (offset < length) {
            if (offset >= data.len) break;
            
            const info = data[offset];
            offset += 1;
            
            const precision = (info >> 4) & 0x0F;
            const table_id = info & 0x0F;
            
            if (table_id >= 4) {
                return error.InvalidQuantTableId;
            }
            
            self.quant_tables[table_id].precision = @intCast(precision);
            
            const element_size: usize = if (precision == 0) 1 else 2;
            const table_size = 64 * element_size;
            
            if (offset + table_size > data.len) {
                return error.InvalidDQT;
            }
            
            var i: usize = 0;
            while (i < 64) : (i += 1) {
                const value = if (precision == 0)
                    @as(u16, data[offset + i])
                else
                    std.mem.readInt(u16, data[offset + i * 2 ..][0..2], .big);
                
                self.quant_tables[table_id].table[i] = value;
            }
            
            offset += table_size;
        }
    }
    
    fn parseDHT(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 2) {
            return error.InvalidDHT;
        }
        
        const length = std.mem.readInt(u16, data[0..2], .big);
        if (data.len < length) {
            return error.InvalidDHT;
        }
        
        var offset: usize = 2;
        while (offset < length) {
            if (offset >= data.len) break;
            
            const info = data[offset];
            offset += 1;
            
            const table_class = (info >> 4) & 0x0F; // 0 = DC, 1 = AC
            const table_id = info & 0x0F;
            
            if (table_id >= 4) {
                return error.InvalidHuffmanTableId;
            }
            
            const table = if (table_class == 0)
                &self.dc_tables[table_id]
            else
                &self.ac_tables[table_id];
            
            // Read bit lengths
            if (offset + 16 > data.len) {
                return error.InvalidDHT;
            }
            
            var total_symbols: usize = 0;
            var i: usize = 0;
            while (i < 16) : (i += 1) {
                table.bits[i + 1] = data[offset + i];
                total_symbols += data[offset + i];
            }
            offset += 16;
            
            // Read symbol values
            if (offset + total_symbols > data.len) {
                return error.InvalidDHT;
            }
            
            i = 0;
            while (i < total_symbols) : (i += 1) {
                table.values[i] = data[offset + i];
            }
            offset += total_symbols;
            
            // Build Huffman lookup tables
            try self.buildHuffmanTable(table);
        }
    }
    
    fn buildHuffmanTable(self: *JpegDecoder, table: *HuffmanTable) !void {
        _ = self;
        
        var code: u16 = 0;
        var k: usize = 0;
        
        var i: usize = 1;
        while (i <= 16) : (i += 1) {
            if (table.bits[i] > 0) {
                table.min_code[i - 1] = @intCast(code);
                var j: usize = 0;
                while (j < table.bits[i]) : (j += 1) {
                    table.codes[k] = code;
                    code += 1;
                    k += 1;
                }
                table.max_code[i - 1] = @intCast(code - 1);
                table.val_offset[i - 1] = @intCast(k - table.bits[i]);
                code <<= 1;
            } else {
                table.min_code[i - 1] = -1;
                table.max_code[i - 1] = -1;
                code <<= 1;
            }
        }
    }
    
    fn parseDRI(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 4) {
            return error.InvalidDRI;
        }
        
        const length = std.mem.readInt(u16, data[0..2], .big);
        if (length != 4) {
            return error.InvalidDRI;
        }
        
        self.restart_interval = std.mem.readInt(u16, data[2..4], .big);
    }
    
    fn parseCOM(self: *JpegDecoder, data: []const u8) !?[]const u8 {
        if (data.len < 2) {
            return error.InvalidCOM;
        }
        
        const length = std.mem.readInt(u16, data[0..2], .big);
        if (data.len < length) {
            return error.InvalidCOM;
        }
        
        if (length <= 2) {
            return null;
        }
        
        const comment_text = data[2..length];
        const comment = try self.allocator.dupe(u8, comment_text);
        return comment;
    }
    
    fn parseEXIF(self: *JpegDecoder, data: []const u8) !?ExifMetadata {
        if (data.len < 2) {
            return error.InvalidEXIF;
        }
        
        const length = std.mem.readInt(u16, data[0..2], .big);
        if (data.len < length) {
            return error.InvalidEXIF;
        }
        
        var exif = ExifMetadata.init(self.allocator);
        errdefer exif.deinit();
        
        // Check for "Exif\0\0" identifier
        if (length <= 8 or !std.mem.eql(u8, data[2..6], "Exif")) {
            return exif;
        }
        
        // TIFF header starts at offset 8 (after length + "Exif\0\0")
        const tiff_offset: usize = 8;
        if (tiff_offset + 8 > data.len) {
            return exif;
        }
        
        // Read TIFF byte order (II = little-endian, MM = big-endian)
        const byte_order = data[tiff_offset..][0..2];
        const is_little_endian = std.mem.eql(u8, byte_order, "II");
        
        if (!is_little_endian and !std.mem.eql(u8, byte_order, "MM")) {
            return exif;
        }
        
        // Read TIFF magic number (should be 42)
        const magic = if (is_little_endian)
            std.mem.readInt(u16, data[tiff_offset + 2..][0..2], .little)
        else
            std.mem.readInt(u16, data[tiff_offset + 2..][0..2], .big);
        
        if (magic != 42) {
            return exif;
        }
        
        // Read IFD (Image File Directory) offset
        const ifd_offset = if (is_little_endian)
            std.mem.readInt(u32, data[tiff_offset + 4..][0..4], .little)
        else
            std.mem.readInt(u32, data[tiff_offset + 4..][0..4], .big);
        
        // Parse IFD entries
        try self.parseIFD(data[tiff_offset..], ifd_offset, is_little_endian, &exif);
        
        return exif;
    }
    
    fn parseIFD(self: *JpegDecoder, data: []const u8, offset: u32, is_little_endian: bool, exif: *ExifMetadata) !void {
        if (offset + 2 > data.len) {
            return;
        }
        
        // Read number of directory entries
        const num_entries = if (is_little_endian)
            std.mem.readInt(u16, data[offset..][0..2], .little)
        else
            std.mem.readInt(u16, data[offset..][0..2], .big);
        
        var entry_offset = offset + 2;
        var i: usize = 0;
        
        while (i < num_entries) : (i += 1) {
            if (entry_offset + 12 > data.len) break;
            
            const tag = if (is_little_endian)
                std.mem.readInt(u16, data[entry_offset..][0..2], .little)
            else
                std.mem.readInt(u16, data[entry_offset..][0..2], .big);
            
            const field_type = if (is_little_endian)
                std.mem.readInt(u16, data[entry_offset + 2..][0..2], .little)
            else
                std.mem.readInt(u16, data[entry_offset + 2..][0..2], .big);
            
            const count = if (is_little_endian)
                std.mem.readInt(u32, data[entry_offset + 4..][0..4], .little)
            else
                std.mem.readInt(u32, data[entry_offset + 4..][0..4], .big);
            
            // Value or offset to value
            const value_offset = entry_offset + 8;
            
            // Parse common EXIF tags
            switch (tag) {
                0x010F => { // Make
                    exif.make = try self.readStringValue(data, value_offset, field_type, count, is_little_endian);
                },
                0x0110 => { // Model
                    exif.model = try self.readStringValue(data, value_offset, field_type, count, is_little_endian);
                },
                0x0112 => { // Orientation
                    if (field_type == 3) { // SHORT
                        exif.orientation = if (is_little_endian)
                            std.mem.readInt(u16, data[value_offset..][0..2], .little)
                        else
                            std.mem.readInt(u16, data[value_offset..][0..2], .big);
                    }
                },
                0x011A => { // XResolution
                    exif.x_resolution = try self.readRationalValue(data, value_offset, is_little_endian);
                },
                0x011B => { // YResolution
                    exif.y_resolution = try self.readRationalValue(data, value_offset, is_little_endian);
                },
                0x0131 => { // Software
                    exif.software = try self.readStringValue(data, value_offset, field_type, count, is_little_endian);
                },
                0x0132 => { // DateTime
                    exif.date_time = try self.readStringValue(data, value_offset, field_type, count, is_little_endian);
                },
                else => {},
            }
            
            entry_offset += 12;
        }
    }
    
    fn readStringValue(self: *JpegDecoder, data: []const u8, offset: usize, field_type: u16, count: u32, is_little_endian: bool) !?[]const u8 {
        _ = field_type;
        
        if (count == 0) return null;
        
        // If string fits in 4 bytes, it's inline; otherwise it's an offset
        const value_offset = if (count <= 4) offset else blk: {
            const ptr_offset = if (is_little_endian)
                std.mem.readInt(u32, data[offset..][0..4], .little)
            else
                std.mem.readInt(u32, data[offset..][0..4], .big);
            break :blk @as(usize, ptr_offset);
        };
        
        if (value_offset + count > data.len) return null;
        
        // Find null terminator or use count
        var str_len = count;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            if (data[value_offset + i] == 0) {
                str_len = @intCast(i);
                break;
            }
        }
        
        if (str_len == 0) return null;
        
        const str = try self.allocator.dupe(u8, data[value_offset..][0..str_len]);
        return str;
    }
    
    fn readRationalValue(self: *JpegDecoder, data: []const u8, offset: usize, is_little_endian: bool) !f32 {
        _ = self;
        
        // Rational is stored as offset to two LONGs (numerator, denominator)
        const ptr_offset = if (is_little_endian)
            std.mem.readInt(u32, data[offset..][0..4], .little)
        else
            std.mem.readInt(u32, data[offset..][0..4], .big);
        
        const rational_offset = @as(usize, ptr_offset);
        if (rational_offset + 8 > data.len) return 72.0;
        
        const numerator = if (is_little_endian)
            std.mem.readInt(u32, data[rational_offset..][0..4], .little)
        else
            std.mem.readInt(u32, data[rational_offset..][0..4], .big);
        
        const denominator = if (is_little_endian)
            std.mem.readInt(u32, data[rational_offset + 4..][0..4], .little)
        else
            std.mem.readInt(u32, data[rational_offset + 4..][0..4], .big);
        
        if (denominator == 0) return 72.0;
        
        return @as(f32, @floatFromInt(numerator)) / @as(f32, @floatFromInt(denominator));
    }
};

/// Bit Reader for Huffman decoding
const BitReader = struct {
    data: []const u8,
    byte_offset: usize,
    bit_offset: u3,
    
    pub fn init(data: []const u8) BitReader {
        return BitReader{
            .data = data,
            .byte_offset = 0,
            .bit_offset = 0,
        };
    }
    
    pub fn readBits(self: *BitReader, n: u5) !u16 {
        if (n == 0 or n > 16) return error.InvalidBitCount;
        
        var result: u16 = 0;
        var bits_read: u5 = 0;
        
        while (bits_read < n) {
            if (self.byte_offset >= self.data.len) {
                return error.UnexpectedEndOfData;
            }
            
            const byte = self.data[self.byte_offset];
            const bits_available = 8 - self.bit_offset;
            const bits_to_read = @min(bits_available, n - bits_read);
            
            const mask: u8 = @intCast((@as(u16, 1) << bits_to_read) - 1);
            const shift = @as(u3, @intCast(bits_available - bits_to_read));
            const bits = (byte >> shift) & mask;
            
            result = (result << bits_to_read) | @as(u16, bits);
            bits_read += bits_to_read;
            
            self.bit_offset += @intCast(bits_to_read);
            if (self.bit_offset >= 8) {
                self.bit_offset = 0;
                self.byte_offset += 1;
                
                // Handle byte stuffing (0xFF 0x00)
                if (self.byte_offset < self.data.len and self.data[self.byte_offset - 1] == 0xFF) {
                    if (self.byte_offset < self.data.len and self.data[self.byte_offset] == 0x00) {
                        self.byte_offset += 1;
                    }
                }
            }
        }
        
        return result;
    }
    
    pub fn alignToByte(self: *BitReader) void {
        if (self.bit_offset != 0) {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
    }
};

/// DCT coefficient block (8x8)
const Block = [64]i16;

/// IDCT (Inverse Discrete Cosine Transform)
pub fn idct(input: *const Block, output: *Block) void {
    // AAN (Arai, Agui, and Nakajima) IDCT algorithm
    // Optimized for speed with minimal multiplications
    
    const M_SQRT2: f32 = 1.41421356237;
    const M_SQRT1_2: f32 = 0.70710678118;
    
    var temp: [64]f32 = undefined;
    
    // Process rows
    var y: usize = 0;
    while (y < 8) : (y += 1) {
        const row = y * 8;
        
        // Even part
        const s0 = @as(f32, @floatFromInt(input[row + 0]));
        const s1 = @as(f32, @floatFromInt(input[row + 4]));
        const s2 = @as(f32, @floatFromInt(input[row + 2]));
        const s3 = @as(f32, @floatFromInt(input[row + 6]));
        
        const t0 = s0 + s1;
        const t1 = s0 - s1;
        const t2 = s2 * M_SQRT1_2 - s3;
        const t3 = s3 * M_SQRT1_2 + s2;
        
        const x0 = t0 + t3;
        const x1 = t1 + t2;
        const x2 = t1 - t2;
        const x3 = t0 - t3;
        
        // Odd part
        const s4 = @as(f32, @floatFromInt(input[row + 1]));
        const s5 = @as(f32, @floatFromInt(input[row + 3]));
        const s6 = @as(f32, @floatFromInt(input[row + 5]));
        const s7 = @as(f32, @floatFromInt(input[row + 7]));
        
        const t4 = s4 + s7;
        const t5 = s5 + s6;
        const t6 = s5 - s6;
        const t7 = s4 - s7;
        
        const x4 = t4 + t5;
        const x5 = (t6 + t7) * M_SQRT1_2;
        const x6 = t4 - t5;
        const x7 = t7;
        
        temp[row + 0] = x0 + x4;
        temp[row + 1] = x1 + x5;
        temp[row + 2] = x2 + x6;
        temp[row + 3] = x3 + x7;
        temp[row + 4] = x3 - x7;
        temp[row + 5] = x2 - x6;
        temp[row + 6] = x1 - x5;
        temp[row + 7] = x0 - x4;
    }
    
    // Process columns
    var x: usize = 0;
    while (x < 8) : (x += 1) {
        // Even part
        const s0 = temp[x + 0];
        const s1 = temp[x + 32]; // 4*8
        const s2 = temp[x + 16]; // 2*8
        const s3 = temp[x + 48]; // 6*8
        
        const t0 = s0 + s1;
        const t1 = s0 - s1;
        const t2 = s2 * M_SQRT1_2 - s3;
        const t3 = s3 * M_SQRT1_2 + s2;
        
        const v0 = t0 + t3;
        const v1 = t1 + t2;
        const v2 = t1 - t2;
        const v3 = t0 - t3;
        
        // Odd part
        const s4 = temp[x + 8];  // 1*8
        const s5 = temp[x + 24]; // 3*8
        const s6 = temp[x + 40]; // 5*8
        const s7 = temp[x + 56]; // 7*8
        
        const t4 = s4 + s7;
        const t5 = s5 + s6;
        const t6 = s5 - s6;
        const t7 = s4 - s7;
        
        const v4 = t4 + t5;
        const v5 = (t6 + t7) * M_SQRT1_2;
        const v6 = t4 - t5;
        const v7 = t7;
        
        // Final output with level shift and clamping
        output[x + 0] = clampToI16((v0 + v4) / 8.0);
        output[x + 8] = clampToI16((v1 + v5) / 8.0);
        output[x + 16] = clampToI16((v2 + v6) / 8.0);
        output[x + 24] = clampToI16((v3 + v7) / 8.0);
        output[x + 32] = clampToI16((v3 - v7) / 8.0);
        output[x + 40] = clampToI16((v2 - v6) / 8.0);
        output[x + 48] = clampToI16((v1 - v5) / 8.0);
        output[x + 56] = clampToI16((v0 - v4) / 8.0);
    }
}

fn clampToI16(value: f32) i16 {
    if (value < -128.0) return -128;
    if (value > 127.0) return 127;
    return @intFromFloat(value);
}

/// Clamp value to 0-255 range
pub fn clampToByte(value: i32) u8 {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return @intCast(value);
}

/// YCbCr to RGB conversion
pub fn ycbcrToRGB(y: u8, cb: u8, cr: u8) [3]u8 {
    // JPEG YCbCr to RGB conversion (ITU-R BT.601)
    const y_val = @as(i32, y);
    const cb_val = @as(i32, cb) - 128;
    const cr_val = @as(i32, cr) - 128;
    
    const r = y_val + ((cr_val * 1436) >> 10);
    const g = y_val - ((cb_val * 352) >> 10) - ((cr_val * 731) >> 10);
    const b = y_val + ((cb_val * 1815) >> 10);
    
    return [3]u8{
        clampToByte(r),
        clampToByte(g),
        clampToByte(b),
    };
}

/// Zigzag scan order for 8x8 blocks
pub const ZIGZAG = [64]u8{
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

/// Dequantize DCT coefficients
pub fn dequantize(block: *Block, quant_table: *const QuantTable) void {
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        block[i] = @intCast(@as(i32, block[i]) * @as(i32, quant_table.table[i]));
    }
}

/// Decode Huffman symbol
pub fn decodeHuffmanSymbol(reader: *BitReader, table: *const HuffmanTable) !u8 {
    var code: u16 = 0;
    var i: usize = 0;
    
    while (i < 16) : (i += 1) {
        const bit = try reader.readBits(1);
        code = (code << 1) | bit;
        
        if (table.min_code[i] != -1 and code <= table.max_code[i]) {
            const index = @as(usize, @intCast(table.val_offset[i])) + 
                         @as(usize, @intCast(code - table.min_code[i]));
            return table.values[index];
        }
    }
    
    return error.InvalidHuffmanCode;
}

// FFI exports for Mojo integration
export fn nExtract_JPEG_decode(data: [*]const u8, len: usize) ?*JpegImage {
    const allocator = std.heap.c_allocator;
    var decoder = JpegDecoder.create(allocator);
    const image = decoder.decode(data[0..len]) catch return null;
    return image;
}

export fn nExtract_JPEG_destroy(image: *JpegImage) void {
    image.deinit();
}

export fn nExtract_JPEG_getWidth(image: *const JpegImage) u16 {
    return image.width;
}

export fn nExtract_JPEG_getHeight(image: *const JpegImage) u16 {
    return image.height;
}

export fn nExtract_JPEG_getPixels(image: *const JpegImage) [*]const u8 {
    return image.pixels.ptr;
}
