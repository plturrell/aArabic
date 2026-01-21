const std = @import("std");
const deflate = @import("deflate.zig");

// ZIP file format constants
const LOCAL_FILE_HEADER_SIGNATURE = 0x04034b50;
const CENTRAL_DIR_SIGNATURE = 0x02014b50;
const END_OF_CENTRAL_DIR_SIGNATURE = 0x06054b50;
const ZIP64_END_OF_CENTRAL_DIR_SIGNATURE = 0x06064b50;
const ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIGNATURE = 0x07064b50;
const DATA_DESCRIPTOR_SIGNATURE = 0x08074b50;

// Compression methods
pub const CompressionMethod = enum(u16) {
    Store = 0,
    Deflate = 8,
    _,
};

// File attributes
pub const FileAttributes = packed struct {
    read_only: bool = false,
    hidden: bool = false,
    system: bool = false,
    _reserved1: u1 = 0,
    directory: bool = false,
    archive: bool = false,
    _reserved2: u10 = 0,
};

// ZIP64 extended information
pub const Zip64ExtendedInfo = struct {
    uncompressed_size: ?u64 = null,
    compressed_size: ?u64 = null,
    relative_header_offset: ?u64 = null,
    disk_start_number: ?u32 = null,
};

// Local file header
pub const LocalFileHeader = struct {
    version_needed: u16,
    general_purpose_flags: u16,
    compression_method: CompressionMethod,
    last_mod_time: u16,
    last_mod_date: u16,
    crc32: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    file_name_length: u16,
    extra_field_length: u16,
    file_name: []const u8,
    extra_field: []const u8,

    pub fn deinit(self: *LocalFileHeader, allocator: std.mem.Allocator) void {
        allocator.free(self.file_name);
        allocator.free(self.extra_field);
    }
};

// Central directory file header
pub const CentralDirFileHeader = struct {
    version_made_by: u16,
    version_needed: u16,
    general_purpose_flags: u16,
    compression_method: CompressionMethod,
    last_mod_time: u16,
    last_mod_date: u16,
    crc32: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    file_name_length: u16,
    extra_field_length: u16,
    file_comment_length: u16,
    disk_number_start: u16,
    internal_file_attributes: u16,
    external_file_attributes: u32,
    relative_offset_of_local_header: u32,
    file_name: []const u8,
    extra_field: []const u8,
    file_comment: []const u8,
    zip64_info: Zip64ExtendedInfo,

    pub fn deinit(self: *CentralDirFileHeader, allocator: std.mem.Allocator) void {
        allocator.free(self.file_name);
        allocator.free(self.extra_field);
        allocator.free(self.file_comment);
    }
};

// End of central directory record
pub const EndOfCentralDir = struct {
    disk_number: u16,
    disk_with_central_dir: u16,
    num_entries_this_disk: u16,
    num_entries_total: u16,
    central_dir_size: u32,
    central_dir_offset: u32,
    comment_length: u16,
    comment: []const u8,

    pub fn deinit(self: *EndOfCentralDir, allocator: std.mem.Allocator) void {
        allocator.free(self.comment);
    }
};

// ZIP64 end of central directory record
pub const Zip64EndOfCentralDir = struct {
    size: u64,
    version_made_by: u16,
    version_needed: u16,
    disk_number: u32,
    disk_with_central_dir: u32,
    num_entries_this_disk: u64,
    num_entries_total: u64,
    central_dir_size: u64,
    central_dir_offset: u64,
};

// ZIP file entry
pub const ZipEntry = struct {
    file_name: []const u8,
    compression_method: CompressionMethod,
    compressed_size: u64,
    uncompressed_size: u64,
    crc32: u32,
    local_header_offset: u64,
    is_directory: bool,
    last_mod_time: u16,
    last_mod_date: u16,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ZipEntry) void {
        self.allocator.free(self.file_name);
    }
};

// ZIP archive
pub const ZipArchive = struct {
    entries: std.ArrayList(ZipEntry),
    allocator: std.mem.Allocator,
    data: []const u8,
    eocd: EndOfCentralDir,
    zip64_eocd: ?Zip64EndOfCentralDir,

    pub fn init(allocator: std.mem.Allocator) ZipArchive {
        return .{
            .entries = std.ArrayList(ZipEntry){},
            .allocator = allocator,
            .data = &[_]u8{},
            .eocd = undefined,
            .zip64_eocd = null,
        };
    }

    pub fn deinit(self: *ZipArchive) void {
        for (self.entries.items) |*entry| {
            entry.deinit();
        }
        self.entries.deinit(self.allocator);
        self.eocd.deinit(self.allocator);
    }

    pub fn open(allocator: std.mem.Allocator, data: []const u8) !ZipArchive {
        var archive = init(allocator);
        archive.data = data;

        // Find end of central directory record
        const eocd_offset = try findEndOfCentralDir(data);
        archive.eocd = try parseEndOfCentralDir(allocator, data[eocd_offset..]);

        // Check for ZIP64
        if (archive.eocd.central_dir_offset == 0xFFFFFFFF or 
            archive.eocd.central_dir_size == 0xFFFFFFFF or
            archive.eocd.num_entries_total == 0xFFFF) {
            archive.zip64_eocd = try findAndParseZip64EndOfCentralDir(data, eocd_offset);
        }

        // Parse central directory
        try archive.parseCentralDirectory();

        return archive;
    }

    fn parseCentralDirectory(self: *ZipArchive) !void {
        const cd_offset = if (self.zip64_eocd) |z64|
            z64.central_dir_offset
        else
            self.eocd.central_dir_offset;

        const num_entries = if (self.zip64_eocd) |z64|
            z64.num_entries_total
        else
            self.eocd.num_entries_total;

        var offset = cd_offset;
        var i: u64 = 0;
        while (i < num_entries) : (i += 1) {
            const header = try parseCentralDirFileHeader(self.allocator, self.data[offset..]);
            defer {
                var h = header;
                h.deinit(self.allocator);
            }

            const entry = ZipEntry{
                .file_name = try self.allocator.dupe(u8, header.file_name),
                .compression_method = header.compression_method,
                .compressed_size = if (header.zip64_info.compressed_size) |size|
                    size
                else
                    header.compressed_size,
                .uncompressed_size = if (header.zip64_info.uncompressed_size) |size|
                    size
                else
                    header.uncompressed_size,
                .crc32 = header.crc32,
                .local_header_offset = if (header.zip64_info.relative_header_offset) |off|
                    off
                else
                    header.relative_offset_of_local_header,
                .is_directory = (header.external_file_attributes & 0x10) != 0,
                .last_mod_time = header.last_mod_time,
                .last_mod_date = header.last_mod_date,
                .allocator = self.allocator,
            };

            try self.entries.append(self.allocator, entry);

            offset += 46 + header.file_name_length + header.extra_field_length + header.file_comment_length;
        }
    }

    pub fn extractFile(self: *ZipArchive, entry: *const ZipEntry, output: []u8) !usize {
        if (entry.is_directory) {
            return error.IsDirectory;
        }

        if (output.len < entry.uncompressed_size) {
            return error.BufferTooSmall;
        }

        // Read local file header
        const local_header = try parseLocalFileHeader(
            self.allocator,
            self.data[entry.local_header_offset..]
        );
        defer {
            var h = local_header;
            h.deinit(self.allocator);
        }

        // Calculate data offset
        const data_offset = entry.local_header_offset + 30 + 
            local_header.file_name_length + local_header.extra_field_length;

        const compressed_data = self.data[data_offset..][0..entry.compressed_size];

        switch (entry.compression_method) {
            .Store => {
                // No compression - direct copy
                @memcpy(output[0..entry.uncompressed_size], compressed_data);
            },
            .Deflate => {
                // DEFLATE decompression
                const decompressed = try deflate.decompress(self.allocator, compressed_data);
                defer self.allocator.free(decompressed);

                if (decompressed.len != entry.uncompressed_size) {
                    return error.DecompressionSizeMismatch;
                }

                @memcpy(output[0..decompressed.len], decompressed);
            },
            else => {
                return error.UnsupportedCompressionMethod;
            },
        }

        // Verify CRC32
        const calculated_crc = crc32(output[0..entry.uncompressed_size]);
        if (calculated_crc != entry.crc32) {
            return error.CRC32Mismatch;
        }

        return entry.uncompressed_size;
    }

    pub fn extractFileAlloc(self: *ZipArchive, entry: *const ZipEntry) ![]u8 {
        const buffer = try self.allocator.alloc(u8, entry.uncompressed_size);
        errdefer self.allocator.free(buffer);

        const size = try self.extractFile(entry, buffer);
        return buffer[0..size];
    }

    pub fn findEntry(self: *ZipArchive, name: []const u8) ?*ZipEntry {
        for (self.entries.items) |*entry| {
            if (std.mem.eql(u8, entry.file_name, name)) {
                return entry;
            }
        }
        return null;
    }
};

// CRC32 implementation
const crc32_table = blk: {
    @setEvalBranchQuota(10000);
    var table: [256]u32 = undefined;
    for (&table, 0..) |*entry, i| {
        var crc: u32 = @intCast(i);
        var j: u8 = 0;
        while (j < 8) : (j += 1) {
            if (crc & 1 != 0) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
        entry.* = crc;
    }
    break :blk table;
};

pub fn crc32(data: []const u8) u32 {
    var crc: u32 = 0xFFFFFFFF;
    for (data) |byte| {
        const index = @as(u8, @truncate(crc)) ^ byte;
        crc = (crc >> 8) ^ crc32_table[index];
    }
    return ~crc;
}

// Helper functions for parsing ZIP structures

fn findEndOfCentralDir(data: []const u8) !usize {
    // Search backwards for EOCD signature
    if (data.len < 22) return error.InvalidZipFile;

    var i = data.len - 22;  // Start from the earliest possible position
    while (true) {
        if (i + 4 > data.len) break;
        if (std.mem.readInt(u32, data[i..][0..4], .little) == END_OF_CENTRAL_DIR_SIGNATURE) {
            // Verify comment length matches
            if (i + 22 <= data.len) {
                const comment_len = std.mem.readInt(u16, data[i + 20..][0..2], .little);
                if (i + 22 + comment_len == data.len) {
                    return i;
                }
            }
        }
        if (i == 0) break;
        i -= 1;
    }

    return error.EndOfCentralDirNotFound;
}

fn parseEndOfCentralDir(allocator: std.mem.Allocator, data: []const u8) !EndOfCentralDir {
    if (data.len < 22) return error.InvalidEOCD;

    const signature = std.mem.readInt(u32, data[0..4], .little);
    if (signature != END_OF_CENTRAL_DIR_SIGNATURE) {
        return error.InvalidEOCDSignature;
    }

    const comment_len = std.mem.readInt(u16, data[20..22], .little);
    if (data.len < 22 + comment_len) return error.InvalidEOCD;

    const comment = try allocator.dupe(u8, data[22..][0..comment_len]);

    return EndOfCentralDir{
        .disk_number = std.mem.readInt(u16, data[4..6], .little),
        .disk_with_central_dir = std.mem.readInt(u16, data[6..8], .little),
        .num_entries_this_disk = std.mem.readInt(u16, data[8..10], .little),
        .num_entries_total = std.mem.readInt(u16, data[10..12], .little),
        .central_dir_size = std.mem.readInt(u32, data[12..16], .little),
        .central_dir_offset = std.mem.readInt(u32, data[16..20], .little),
        .comment_length = comment_len,
        .comment = comment,
    };
}

fn findAndParseZip64EndOfCentralDir(data: []const u8, eocd_offset: usize) !Zip64EndOfCentralDir {
    // Search backwards from EOCD for ZIP64 EOCD locator
    if (eocd_offset < 20) return error.Zip64EOCDNotFound;

    var i = eocd_offset;
    while (i >= 20) {
        i -= 1;
        if (std.mem.readInt(u32, data[i..][0..4], .little) == ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIGNATURE) {
            const zip64_eocd_offset = std.mem.readInt(u64, data[i + 8..][0..8], .little);
            return parseZip64EndOfCentralDir(data[zip64_eocd_offset..]);
        }
    }

    return error.Zip64EOCDNotFound;
}

fn parseZip64EndOfCentralDir(data: []const u8) !Zip64EndOfCentralDir {
    if (data.len < 56) return error.InvalidZip64EOCD;

    const signature = std.mem.readInt(u32, data[0..4], .little);
    if (signature != ZIP64_END_OF_CENTRAL_DIR_SIGNATURE) {
        return error.InvalidZip64EOCDSignature;
    }

    return Zip64EndOfCentralDir{
        .size = std.mem.readInt(u64, data[4..12], .little),
        .version_made_by = std.mem.readInt(u16, data[12..14], .little),
        .version_needed = std.mem.readInt(u16, data[14..16], .little),
        .disk_number = std.mem.readInt(u32, data[16..20], .little),
        .disk_with_central_dir = std.mem.readInt(u32, data[20..24], .little),
        .num_entries_this_disk = std.mem.readInt(u64, data[24..32], .little),
        .num_entries_total = std.mem.readInt(u64, data[32..40], .little),
        .central_dir_size = std.mem.readInt(u64, data[40..48], .little),
        .central_dir_offset = std.mem.readInt(u64, data[48..56], .little),
    };
}

fn parseCentralDirFileHeader(allocator: std.mem.Allocator, data: []const u8) !CentralDirFileHeader {
    if (data.len < 46) return error.InvalidCentralDirHeader;

    const signature = std.mem.readInt(u32, data[0..4], .little);
    if (signature != CENTRAL_DIR_SIGNATURE) {
        return error.InvalidCentralDirSignature;
    }

    const file_name_len = std.mem.readInt(u16, data[28..30], .little);
    const extra_field_len = std.mem.readInt(u16, data[30..32], .little);
    const file_comment_len = std.mem.readInt(u16, data[32..34], .little);

    if (data.len < 46 + file_name_len + extra_field_len + file_comment_len) {
        return error.InvalidCentralDirHeader;
    }

    const file_name = try allocator.dupe(u8, data[46..][0..file_name_len]);
    errdefer allocator.free(file_name);

    const extra_field = try allocator.dupe(u8, data[46 + file_name_len..][0..extra_field_len]);
    errdefer allocator.free(extra_field);

    const file_comment = try allocator.dupe(u8, data[46 + file_name_len + extra_field_len..][0..file_comment_len]);

    // Parse ZIP64 extended information if present
    var zip64_info = Zip64ExtendedInfo{};
    if (extra_field_len > 0) {
        zip64_info = parseZip64ExtraField(extra_field);
    }

    return CentralDirFileHeader{
        .version_made_by = std.mem.readInt(u16, data[4..6], .little),
        .version_needed = std.mem.readInt(u16, data[6..8], .little),
        .general_purpose_flags = std.mem.readInt(u16, data[8..10], .little),
        .compression_method = @enumFromInt(std.mem.readInt(u16, data[10..12], .little)),
        .last_mod_time = std.mem.readInt(u16, data[12..14], .little),
        .last_mod_date = std.mem.readInt(u16, data[14..16], .little),
        .crc32 = std.mem.readInt(u32, data[16..20], .little),
        .compressed_size = std.mem.readInt(u32, data[20..24], .little),
        .uncompressed_size = std.mem.readInt(u32, data[24..28], .little),
        .file_name_length = file_name_len,
        .extra_field_length = extra_field_len,
        .file_comment_length = file_comment_len,
        .disk_number_start = std.mem.readInt(u16, data[34..36], .little),
        .internal_file_attributes = std.mem.readInt(u16, data[36..38], .little),
        .external_file_attributes = std.mem.readInt(u32, data[38..42], .little),
        .relative_offset_of_local_header = std.mem.readInt(u32, data[42..46], .little),
        .file_name = file_name,
        .extra_field = extra_field,
        .file_comment = file_comment,
        .zip64_info = zip64_info,
    };
}

fn parseLocalFileHeader(allocator: std.mem.Allocator, data: []const u8) !LocalFileHeader {
    if (data.len < 30) return error.InvalidLocalFileHeader;

    const signature = std.mem.readInt(u32, data[0..4], .little);
    if (signature != LOCAL_FILE_HEADER_SIGNATURE) {
        return error.InvalidLocalFileHeaderSignature;
    }

    const file_name_len = std.mem.readInt(u16, data[26..28], .little);
    const extra_field_len = std.mem.readInt(u16, data[28..30], .little);

    if (data.len < 30 + file_name_len + extra_field_len) {
        return error.InvalidLocalFileHeader;
    }

    const file_name = try allocator.dupe(u8, data[30..][0..file_name_len]);
    errdefer allocator.free(file_name);

    const extra_field = try allocator.dupe(u8, data[30 + file_name_len..][0..extra_field_len]);

    return LocalFileHeader{
        .version_needed = std.mem.readInt(u16, data[4..6], .little),
        .general_purpose_flags = std.mem.readInt(u16, data[6..8], .little),
        .compression_method = @enumFromInt(std.mem.readInt(u16, data[8..10], .little)),
        .last_mod_time = std.mem.readInt(u16, data[10..12], .little),
        .last_mod_date = std.mem.readInt(u16, data[12..14], .little),
        .crc32 = std.mem.readInt(u32, data[14..18], .little),
        .compressed_size = std.mem.readInt(u32, data[18..22], .little),
        .uncompressed_size = std.mem.readInt(u32, data[22..26], .little),
        .file_name_length = file_name_len,
        .extra_field_length = extra_field_len,
        .file_name = file_name,
        .extra_field = extra_field,
    };
}

fn parseZip64ExtraField(extra_field: []const u8) Zip64ExtendedInfo {
    var info = Zip64ExtendedInfo{};
    var offset: usize = 0;

    while (offset + 4 <= extra_field.len) {
        const header_id = std.mem.readInt(u16, extra_field[offset..][0..2], .little);
        const data_size = std.mem.readInt(u16, extra_field[offset + 2..][0..2], .little);
        offset += 4;

        if (offset + data_size > extra_field.len) break;

        if (header_id == 0x0001) { // ZIP64 extended information
            var field_offset: usize = 0;
            
            if (field_offset + 8 <= data_size) {
                info.uncompressed_size = std.mem.readInt(u64, extra_field[offset + field_offset..][0..8], .little);
                field_offset += 8;
            }
            
            if (field_offset + 8 <= data_size) {
                info.compressed_size = std.mem.readInt(u64, extra_field[offset + field_offset..][0..8], .little);
                field_offset += 8;
            }
            
            if (field_offset + 8 <= data_size) {
                info.relative_header_offset = std.mem.readInt(u64, extra_field[offset + field_offset..][0..8], .little);
                field_offset += 8;
            }
            
            if (field_offset + 4 <= data_size) {
                info.disk_start_number = std.mem.readInt(u32, extra_field[offset + field_offset..][0..4], .little);
            }
        }

        offset += data_size;
    }

    return info;
}

// Export C-compatible API
export fn nExtract_ZIP_open(data: [*]const u8, len: usize) ?*ZipArchive {
    const allocator = std.heap.c_allocator;
    const archive = allocator.create(ZipArchive) catch return null;
    archive.* = ZipArchive.open(allocator, data[0..len]) catch {
        allocator.destroy(archive);
        return null;
    };
    return archive;
}

export fn nExtract_ZIP_close(archive: *ZipArchive) void {
    const allocator = archive.allocator;
    archive.deinit();
    allocator.destroy(archive);
}

export fn nExtract_ZIP_get_entry_count(archive: *ZipArchive) usize {
    return archive.entries.items.len;
}

export fn nExtract_ZIP_get_entry(archive: *ZipArchive, index: usize) ?*ZipEntry {
    if (index >= archive.entries.items.len) return null;
    return &archive.entries.items[index];
}

export fn nExtract_ZIP_find_entry(archive: *ZipArchive, name: [*:0]const u8) ?*ZipEntry {
    const name_slice = std.mem.span(name);
    return archive.findEntry(name_slice);
}

export fn nExtract_ZIP_extract_file(
    archive: *ZipArchive,
    entry: *const ZipEntry,
    output: [*]u8,
    output_len: usize
) isize {
    const size = archive.extractFile(entry, output[0..output_len]) catch {
        return -1;
    };
    return @intCast(size);
}
