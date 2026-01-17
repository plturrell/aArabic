const std = @import("std");

/// WAV file RIFF header - not packed to allow arrays
pub const RiffHeader = struct {
    chunk_id: [4]u8 = "RIFF".*,
    chunk_size: u32,
    format: [4]u8 = "WAVE".*,
};

/// WAV format chunk - not packed to allow proper alignment
pub const FmtChunk = struct {
    chunk_id: [4]u8 = "fmt ".*,
    chunk_size: u32 = 16,
    audio_format: u16 = 1, // PCM
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
};

/// WAV data chunk header
pub const DataChunkHeader = struct {
    chunk_id: [4]u8 = "data".*,
    chunk_size: u32,
};

/// Complete WAV header structure
pub const WavHeader = struct {
    riff: RiffHeader,
    fmt: FmtChunk,
    data: DataChunkHeader,

    pub fn init(num_samples: usize, sample_rate: u32, channels: u16, bits_per_sample: u16) WavHeader {
        const byte_rate = sample_rate * channels * (bits_per_sample / 8);
        const block_align = channels * (bits_per_sample / 8);
        const data_size = @as(u32, @intCast(num_samples * channels * (bits_per_sample / 8)));
        
        return WavHeader{
            .riff = RiffHeader{
                .chunk_size = 36 + data_size,
                .chunk_id = "RIFF".*,
                .format = "WAVE".*,
            },
            .fmt = FmtChunk{
                .chunk_id = "fmt ".*,
                .chunk_size = 16,
                .audio_format = 1,
                .num_channels = channels,
                .sample_rate = sample_rate,
                .byte_rate = byte_rate,
                .block_align = @intCast(block_align),
                .bits_per_sample = bits_per_sample,
            },
            .data = DataChunkHeader{
                .chunk_id = "data".*,
                .chunk_size = data_size,
            },
        };
    }

    pub fn validate(self: WavHeader) !void {
        if (!std.mem.eql(u8, &self.riff.chunk_id, "RIFF")) {
            return error.InvalidRiffHeader;
        }
        if (!std.mem.eql(u8, &self.riff.format, "WAVE")) {
            return error.InvalidWaveFormat;
        }
        if (!std.mem.eql(u8, &self.fmt.chunk_id, "fmt ")) {
            return error.InvalidFmtChunk;
        }
        if (self.fmt.audio_format != 1) {
            return error.UnsupportedAudioFormat;
        }
        if (!std.mem.eql(u8, &self.data.chunk_id, "data")) {
            return error.InvalidDataChunk;
        }
    }
};

/// Parse WAV file header from bytes
pub fn parseHeader(data: []const u8) !WavHeader {
    if (data.len < 44) return error.HeaderTooSmall;

    var header: WavHeader = undefined;

    // Parse RIFF header
    @memcpy(&header.riff.chunk_id, data[0..4]);
    header.riff.chunk_size = std.mem.readInt(u32, data[4..8], .little);
    @memcpy(&header.riff.format, data[8..12]);

    // Parse fmt chunk
    @memcpy(&header.fmt.chunk_id, data[12..16]);
    header.fmt.chunk_size = std.mem.readInt(u32, data[16..20], .little);
    header.fmt.audio_format = std.mem.readInt(u16, data[20..22], .little);
    header.fmt.num_channels = std.mem.readInt(u16, data[22..24], .little);
    header.fmt.sample_rate = std.mem.readInt(u32, data[24..28], .little);
    header.fmt.byte_rate = std.mem.readInt(u32, data[28..32], .little);
    header.fmt.block_align = std.mem.readInt(u16, data[32..34], .little);
    header.fmt.bits_per_sample = std.mem.readInt(u16, data[34..36], .little);

    // Parse data chunk header
    @memcpy(&header.data.chunk_id, data[36..40]);
    header.data.chunk_size = std.mem.readInt(u32, data[40..44], .little);

    try header.validate();

    return header;
}

/// Write WAV header to bytes
pub fn writeHeader(header: WavHeader, buffer: []u8) !void {
    if (buffer.len < 44) return error.BufferTooSmall;

    @memcpy(buffer[0..4], &header.riff.chunk_id);
    std.mem.writeInt(u32, buffer[4..8], header.riff.chunk_size, .little);
    @memcpy(buffer[8..12], &header.riff.format);

    @memcpy(buffer[12..16], &header.fmt.chunk_id);
    std.mem.writeInt(u32, buffer[16..20], header.fmt.chunk_size, .little);
    std.mem.writeInt(u16, buffer[20..22], header.fmt.audio_format, .little);
    std.mem.writeInt(u16, buffer[22..24], header.fmt.num_channels, .little);
    std.mem.writeInt(u32, buffer[24..28], header.fmt.sample_rate, .little);
    std.mem.writeInt(u32, buffer[28..32], header.fmt.byte_rate, .little);
    std.mem.writeInt(u16, buffer[32..34], header.fmt.block_align, .little);
    std.mem.writeInt(u16, buffer[34..36], header.fmt.bits_per_sample, .little);

    @memcpy(buffer[36..40], &header.data.chunk_id);
    std.mem.writeInt(u32, buffer[40..44], header.data.chunk_size, .little);
}

/// Read 24-bit sample from byte array (little-endian)
pub fn read24BitSample(bytes: *const [3]u8) i32 {
    var sample: i32 = 0;
    sample |= @as(i32, bytes[0]);
    sample |= @as(i32, bytes[1]) << 8;
    sample |= @as(i32, @as(i8, @bitCast(bytes[2]))) << 16;
    return sample;
}

/// Write 24-bit sample to byte array (little-endian)
pub fn write24BitSample(sample: i32, bytes: *[3]u8) void {
    bytes[0] = @truncate(@as(u32, @bitCast(sample)));
    bytes[1] = @truncate(@as(u32, @bitCast(sample)) >> 8);
    bytes[2] = @truncate(@as(u32, @bitCast(sample)) >> 16);
}

/// Read 16-bit sample from byte array (little-endian)
pub fn read16BitSample(bytes: *const [2]u8) i16 {
    return std.mem.readInt(i16, bytes, .little);
}

/// Write 16-bit sample to byte array (little-endian)
pub fn write16BitSample(sample: i16, bytes: *[2]u8) void {
    std.mem.writeInt(i16, bytes, sample, .little);
}

test "WAV header creation" {
    const header = WavHeader.init(48000, 48000, 2, 24);
    
    try std.testing.expectEqual(@as(u32, 48000), header.fmt.sample_rate);
    try std.testing.expectEqual(@as(u16, 2), header.fmt.num_channels);
    try std.testing.expectEqual(@as(u16, 24), header.fmt.bits_per_sample);
    
    try header.validate();
}

test "WAV header serialization" {
    const header = WavHeader.init(100, 48000, 2, 24);
    
    var buffer: [44]u8 = undefined;
    try writeHeader(header, &buffer);
    
    const parsed = try parseHeader(&buffer);
    
    try std.testing.expectEqual(header.fmt.sample_rate, parsed.fmt.sample_rate);
    try std.testing.expectEqual(header.fmt.num_channels, parsed.fmt.num_channels);
    try std.testing.expectEqual(header.fmt.bits_per_sample, parsed.fmt.bits_per_sample);
}

test "24-bit sample conversion" {
    const sample: i32 = 1234567;
    var bytes: [3]u8 = undefined;
    write24BitSample(sample, &bytes);
    const read_sample = read24BitSample(&bytes);
    
    try std.testing.expectEqual(sample, read_sample);
}
