const std = @import("std");
const audio_types = @import("audio_types.zig");
const wav_format = @import("wav_format.zig");

const AudioBuffer = audio_types.AudioBuffer;
const WavHeader = wav_format.WavHeader;

/// Read a WAV file and return an AudioBuffer
pub fn readWAV(path: []const u8, allocator: std.mem.Allocator) !AudioBuffer {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // Read entire file into memory
    const file_size = try file.getEndPos();
    const file_data = try allocator.alloc(u8, file_size);
    defer allocator.free(file_data);
    
    _ = try file.readAll(file_data);

    // Parse WAV header
    const header = try wav_format.parseHeader(file_data);

    // Calculate number of samples
    const bytes_per_sample = header.fmt.bits_per_sample / 8;
    const total_samples = header.data.chunk_size / bytes_per_sample;
    const num_frames = total_samples / header.fmt.num_channels;

    // Allocate buffer for float samples
    const buffer = try AudioBuffer.init(
        allocator,
        num_frames,
        header.fmt.sample_rate,
        @intCast(header.fmt.num_channels),
        @intCast(header.fmt.bits_per_sample),
    );

    // Read PCM data starting after header (44 bytes)
    const pcm_data = file_data[44..];

    // Convert PCM to float based on bit depth
    switch (header.fmt.bits_per_sample) {
        16 => try readPcm16(pcm_data, buffer.samples, header.fmt.num_channels),
        24 => try readPcm24(pcm_data, buffer.samples, header.fmt.num_channels),
        else => return error.UnsupportedBitDepth,
    }

    return buffer;
}

/// Write an AudioBuffer to a WAV file
pub fn writeWAV(buffer: AudioBuffer, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    const num_frames = buffer.frameCount();
    const channels = @as(u16, buffer.channels);
    const bits_per_sample = @as(u16, buffer.bit_depth);

    // Create WAV header
    const header = WavHeader.init(num_frames, buffer.sample_rate, channels, bits_per_sample);

    // Write header
    var header_bytes: [44]u8 = undefined;
    try wav_format.writeHeader(header, &header_bytes);
    try file.writeAll(&header_bytes);

    // Convert float samples to PCM and write
    const bytes_per_sample = bits_per_sample / 8;
    const pcm_size = num_frames * channels * bytes_per_sample;
    const pcm_data = try buffer.allocator.alloc(u8, pcm_size);
    defer buffer.allocator.free(pcm_data);

    switch (bits_per_sample) {
        16 => writePcm16(buffer.samples, pcm_data, channels),
        24 => writePcm24(buffer.samples, pcm_data, channels),
        else => return error.UnsupportedBitDepth,
    }

    try file.writeAll(pcm_data);
}

/// Read 16-bit PCM data into float buffer
fn readPcm16(pcm_data: []const u8, samples: []f32, _: u16) !void {
    if (pcm_data.len < samples.len * 2) return error.InsufficientData;

    var i: usize = 0;
    while (i < samples.len) : (i += 1) {
        const byte_offset = i * 2;
        const pcm_bytes = pcm_data[byte_offset .. byte_offset + 2];
        const pcm_sample = wav_format.read16BitSample(pcm_bytes[0..2]);
        samples[i] = audio_types.pcm16ToFloat(pcm_sample);
    }
}

/// Read 24-bit PCM data into float buffer
fn readPcm24(pcm_data: []const u8, samples: []f32, _: u16) !void {
    if (pcm_data.len < samples.len * 3) return error.InsufficientData;

    var i: usize = 0;
    while (i < samples.len) : (i += 1) {
        const byte_offset = i * 3;
        const pcm_bytes = pcm_data[byte_offset .. byte_offset + 3];
        const pcm_sample = wav_format.read24BitSample(pcm_bytes[0..3]);
        samples[i] = audio_types.pcm24ToFloat(pcm_sample);
    }
}

/// Write float buffer to 16-bit PCM data
fn writePcm16(samples: []const f32, pcm_data: []u8, _: u16) void {
    var i: usize = 0;
    while (i < samples.len) : (i += 1) {
        const pcm_sample = audio_types.floatToPcm16(samples[i]);
        const byte_offset = i * 2;
        var pcm_bytes = pcm_data[byte_offset .. byte_offset + 2];
        wav_format.write16BitSample(pcm_sample, pcm_bytes[0..2]);
    }
}

/// Write float buffer to 24-bit PCM data
fn writePcm24(samples: []const f32, pcm_data: []u8, _: u16) void {
    var i: usize = 0;
    while (i < samples.len) : (i += 1) {
        const pcm_sample = audio_types.floatToPcm24(samples[i]);
        const byte_offset = i * 3;
        var pcm_bytes = pcm_data[byte_offset .. byte_offset + 3];
        wav_format.write24BitSample(pcm_sample, pcm_bytes[0..3]);
    }
}

/// Generate a test tone (sine wave)
pub fn generateTestTone(
    allocator: std.mem.Allocator,
    frequency: f32,
    duration_seconds: f32,
    sample_rate: u32,
    channels: u8,
) !AudioBuffer {
    const num_frames = @as(usize, @intFromFloat(duration_seconds * @as(f32, @floatFromInt(sample_rate))));
    var buffer = try AudioBuffer.init(allocator, num_frames, sample_rate, channels, 24);

    const two_pi = 2.0 * std.math.pi;
    const angular_freq = two_pi * frequency;

    var frame: usize = 0;
    while (frame < num_frames) : (frame += 1) {
        const t = @as(f32, @floatFromInt(frame)) / @as(f32, @floatFromInt(sample_rate));
        const sample_value = @sin(angular_freq * t) * 0.5; // Half amplitude to avoid clipping

        // Write to all channels
        var ch: usize = 0;
        while (ch < channels) : (ch += 1) {
            buffer.samples[frame * channels + ch] = sample_value;
        }
    }

    return buffer;
}

/// Get audio file information without loading the entire file
pub fn getAudioInfo(path: []const u8, _: std.mem.Allocator) !AudioFileInfo {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var header_bytes: [44]u8 = undefined;
    _ = try file.readAll(&header_bytes);

    const header = try wav_format.parseHeader(&header_bytes);

    const bytes_per_sample = header.fmt.bits_per_sample / 8;
    const total_samples = header.data.chunk_size / bytes_per_sample;
    const num_frames = total_samples / header.fmt.num_channels;
    const duration = @as(f32, @floatFromInt(num_frames)) / @as(f32, @floatFromInt(header.fmt.sample_rate));

    return AudioFileInfo{
        .sample_rate = header.fmt.sample_rate,
        .channels = @intCast(header.fmt.num_channels),
        .bit_depth = @intCast(header.fmt.bits_per_sample),
        .num_frames = num_frames,
        .duration_seconds = duration,
        .file_size = try file.getEndPos(),
    };
}

pub const AudioFileInfo = struct {
    sample_rate: u32,
    channels: u8,
    bit_depth: u8,
    num_frames: usize,
    duration_seconds: f32,
    file_size: u64,

    pub fn format(self: AudioFileInfo, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print(
            "AudioFile: {d}Hz, {d}ch, {d}bit, {d} frames, {d:.2}s, {d} bytes",
            .{ self.sample_rate, self.channels, self.bit_depth, self.num_frames, self.duration_seconds, self.file_size },
        );
    }
};

test "generate and write test tone" {
    const allocator = std.testing.allocator;

    // Generate 1 second of 440Hz tone at 48kHz stereo
    var buffer = try generateTestTone(allocator, 440.0, 1.0, 48000, 2);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(u32, 48000), buffer.sample_rate);
    try std.testing.expectEqual(@as(u8, 2), buffer.channels);
    try std.testing.expectEqual(@as(usize, 48000), buffer.frameCount());

    // Verify samples are in valid range
    for (buffer.samples) |sample| {
        try std.testing.expect(sample >= -1.0 and sample <= 1.0);
    }
}
