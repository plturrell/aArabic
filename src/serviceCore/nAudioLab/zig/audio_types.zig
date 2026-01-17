const std = @import("std");

/// Professional audio buffer for 48kHz/24-bit stereo processing
pub const AudioBuffer = struct {
    samples: []f32,
    sample_rate: u32 = 48000,
    channels: u8 = 2,
    bit_depth: u8 = 24,
    allocator: std.mem.Allocator,

    /// Initialize an empty audio buffer
    pub fn init(allocator: std.mem.Allocator, num_samples: usize, sample_rate: u32, channels: u8, bit_depth: u8) !AudioBuffer {
        const samples = try allocator.alloc(f32, num_samples * channels);
        @memset(samples, 0.0);
        
        return AudioBuffer{
            .samples = samples,
            .sample_rate = sample_rate,
            .channels = channels,
            .bit_depth = bit_depth,
            .allocator = allocator,
        };
    }

    /// Initialize from existing samples
    pub fn fromSlice(allocator: std.mem.Allocator, samples: []const f32, sample_rate: u32, channels: u8, bit_depth: u8) !AudioBuffer {
        const buffer_samples = try allocator.alloc(f32, samples.len);
        @memcpy(buffer_samples, samples);
        
        return AudioBuffer{
            .samples = buffer_samples,
            .sample_rate = sample_rate,
            .channels = channels,
            .bit_depth = bit_depth,
            .allocator = allocator,
        };
    }

    /// Get the number of frames (samples per channel)
    pub fn frameCount(self: AudioBuffer) usize {
        return self.samples.len / self.channels;
    }

    /// Get duration in seconds
    pub fn duration(self: AudioBuffer) f32 {
        const frames = self.frameCount();
        return @as(f32, @floatFromInt(frames)) / @as(f32, @floatFromInt(self.sample_rate));
    }

    /// Get a specific channel's samples
    pub fn getChannel(self: AudioBuffer, allocator: std.mem.Allocator, channel: u8) ![]f32 {
        if (channel >= self.channels) return error.InvalidChannel;
        
        const frames = self.frameCount();
        const channel_samples = try allocator.alloc(f32, frames);
        
        var i: usize = 0;
        while (i < frames) : (i += 1) {
            channel_samples[i] = self.samples[i * self.channels + channel];
        }
        
        return channel_samples;
    }

    /// Convert to mono by averaging channels
    pub fn toMono(self: AudioBuffer, allocator: std.mem.Allocator) !AudioBuffer {
        if (self.channels == 1) {
            return try self.clone(allocator);
        }
        
        const frames = self.frameCount();
        const mono_samples = try allocator.alloc(f32, frames);
        
        var i: usize = 0;
        while (i < frames) : (i += 1) {
            var sum: f32 = 0.0;
            var ch: usize = 0;
            while (ch < self.channels) : (ch += 1) {
                sum += self.samples[i * self.channels + ch];
            }
            mono_samples[i] = sum / @as(f32, @floatFromInt(self.channels));
        }
        
        return AudioBuffer{
            .samples = mono_samples,
            .sample_rate = self.sample_rate,
            .channels = 1,
            .bit_depth = self.bit_depth,
            .allocator = allocator,
        };
    }

    /// Clone the audio buffer
    pub fn clone(self: AudioBuffer, allocator: std.mem.Allocator) !AudioBuffer {
        return try fromSlice(allocator, self.samples, self.sample_rate, self.channels, self.bit_depth);
    }

    /// Normalize samples to [-1.0, 1.0] range
    pub fn normalize(self: *AudioBuffer) void {
        var max_abs: f32 = 0.0;
        for (self.samples) |sample| {
            const abs_sample = @abs(sample);
            if (abs_sample > max_abs) {
                max_abs = abs_sample;
            }
        }
        
        if (max_abs > 0.0 and max_abs > 1.0) {
            const scale = 1.0 / max_abs;
            for (self.samples) |*sample| {
                sample.* *= scale;
            }
        }
    }

    /// Apply gain in dB
    pub fn applyGain(self: *AudioBuffer, gain_db: f32) void {
        const linear_gain = std.math.pow(f32, 10.0, gain_db / 20.0);
        for (self.samples) |*sample| {
            sample.* *= linear_gain;
        }
    }

    /// Free the audio buffer
    pub fn deinit(self: *AudioBuffer) void {
        self.allocator.free(self.samples);
    }
};

/// Audio format information
pub const AudioFormat = struct {
    sample_rate: u32,
    channels: u8,
    bit_depth: u8,
    
    pub const CD_QUALITY = AudioFormat{
        .sample_rate = 44100,
        .channels = 2,
        .bit_depth = 16,
    };
    
    pub const STUDIO_QUALITY = AudioFormat{
        .sample_rate = 48000,
        .channels = 2,
        .bit_depth = 24,
    };
    
    pub const HIGH_RES = AudioFormat{
        .sample_rate = 96000,
        .channels = 2,
        .bit_depth = 24,
    };
};

/// Convert 24-bit PCM to float32 [-1.0, 1.0]
pub fn pcm24ToFloat(pcm: i32) f32 {
    // 24-bit signed integer range: -8388608 to 8388607
    const max_val: f32 = 8388607.0;
    return @as(f32, @floatFromInt(pcm)) / max_val;
}

/// Convert float32 [-1.0, 1.0] to 24-bit PCM
pub fn floatToPcm24(sample: f32) i32 {
    const clamped = std.math.clamp(sample, -1.0, 1.0);
    const max_val: f32 = 8388607.0;
    return @as(i32, @intFromFloat(clamped * max_val));
}

/// Convert 16-bit PCM to float32 [-1.0, 1.0]
pub fn pcm16ToFloat(pcm: i16) f32 {
    const max_val: f32 = 32767.0;
    return @as(f32, @floatFromInt(pcm)) / max_val;
}

/// Convert float32 [-1.0, 1.0] to 16-bit PCM
pub fn floatToPcm16(sample: f32) i16 {
    const clamped = std.math.clamp(sample, -1.0, 1.0);
    const max_val: f32 = 32767.0;
    return @as(i16, @intFromFloat(clamped * max_val));
}

test "AudioBuffer initialization" {
    const allocator = std.testing.allocator;
    
    var buffer = try AudioBuffer.init(allocator, 100, 48000, 2, 24);
    defer buffer.deinit();
    
    try std.testing.expectEqual(@as(usize, 200), buffer.samples.len);
    try std.testing.expectEqual(@as(u32, 48000), buffer.sample_rate);
    try std.testing.expectEqual(@as(u8, 2), buffer.channels);
    try std.testing.expectEqual(@as(usize, 100), buffer.frameCount());
}

test "PCM conversions" {
    const f_sample: f32 = 0.5;
    const pcm24 = floatToPcm24(f_sample);
    const back_to_float = pcm24ToFloat(pcm24);
    
    try std.testing.expectApproxEqAbs(f_sample, back_to_float, 0.0001);
}
