// Zig FFI Bridge for Mojo-Zig Interoperability
// ============================================
//
// This file provides C-compatible FFI functions to bridge
// Mojo TTS inference engine with Zig audio processing.

const std = @import("std");
const dolby = @import("dolby_processor.zig");
const audio_io = @import("audio_io.zig");
const AudioBuffer = @import("audio_types.zig").AudioBuffer;

/// Process audio with Dolby processing
/// 
/// Parameters:
///   - samples_ptr: Pointer to audio samples (f32 array)
///   - length: Number of samples (mono samples, not stereo pairs)
///   - sample_rate: Sample rate in Hz
///   - channels: Number of audio channels (1=mono, 2=stereo)
/// 
/// Returns:
///   - 0 on success
///   - -1 on error
export fn process_audio_dolby(
    samples_ptr: [*]f32,
    length: usize,
    sample_rate: u32,
    channels: u8,
) callconv(.C) c_int {
    // Convert pointer to slice
    const samples = samples_ptr[0..length];
    
    // Create Dolby config
    const config = dolby.DolbyConfig{
        .target_lufs = -16.0,
        .compression_ratio = 3.0,
        .attack_ms = 5.0,
        .release_ms = 50.0,
        .enhancer_amount = 0.3,
    };
    
    // Process audio in-place
    dolby.processDolby(samples, sample_rate, channels, config) catch |err| {
        std.debug.print("Dolby processing error: {}\n", .{err});
        return -1;
    };
    
    return 0;
}

/// Save audio buffer to WAV file
/// 
/// Parameters:
///   - samples_ptr: Pointer to audio samples (f32 array)
///   - length: Number of samples
///   - sample_rate: Sample rate in Hz
///   - channels: Number of audio channels
///   - bit_depth: Bit depth (16 or 24)
///   - output_path: Null-terminated C string with output file path
/// 
/// Returns:
///   - 0 on success
///   - -1 on error
export fn save_audio_wav(
    samples_ptr: [*]f32,
    length: usize,
    sample_rate: u32,
    channels: u8,
    bit_depth: u8,
    output_path: [*:0]const u8,
) callconv(.C) c_int {
    // Get allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Convert pointer to slice
    const samples = samples_ptr[0..length];
    
    // Convert C string to Zig string
    const path = std.mem.span(output_path);
    
    // Create audio buffer
    // Note: We need to allocate and copy because AudioBuffer owns its memory
    const samples_copy = allocator.alloc(f32, length) catch return -1;
    defer allocator.free(samples_copy);
    @memcpy(samples_copy, samples);
    
    const buffer = AudioBuffer{
        .samples = samples_copy,
        .sample_rate = sample_rate,
        .channels = channels,
        .bit_depth = bit_depth,
    };
    
    // Write WAV file
    audio_io.writeWAV(buffer, path, allocator) catch |err| {
        std.debug.print("WAV write error: {}\n", .{err});
        return -1;
    };
    
    return 0;
}

/// Save audio buffer to MP3 file
/// 
/// Parameters:
///   - samples_ptr: Pointer to audio samples (f32 array)
///   - length: Number of samples
///   - sample_rate: Sample rate in Hz
///   - channels: Number of audio channels
///   - bitrate: MP3 bitrate in kbps (e.g., 320)
///   - output_path: Null-terminated C string with output file path
/// 
/// Returns:
///   - 0 on success
///   - -1 on error
export fn save_audio_mp3(
    samples_ptr: [*]f32,
    length: usize,
    sample_rate: u32,
    channels: u8,
    bitrate: u32,
    output_path: [*:0]const u8,
) callconv(.C) c_int {
    // Get allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Convert pointer to slice
    const samples = samples_ptr[0..length];
    
    // Convert C string to Zig string
    const path = std.mem.span(output_path);
    
    // Create audio buffer
    const samples_copy = allocator.alloc(f32, length) catch return -1;
    defer allocator.free(samples_copy);
    @memcpy(samples_copy, samples);
    
    const buffer = AudioBuffer{
        .samples = samples_copy,
        .sample_rate = sample_rate,
        .channels = channels,
        .bit_depth = 24, // MP3 doesn't use this, but set for consistency
    };
    
    // Write MP3 file
    audio_io.writeMP3(buffer, path, bitrate, allocator) catch |err| {
        std.debug.print("MP3 write error: {}\n", .{err});
        return -1;
    };
    
    return 0;
}

/// Load audio buffer from WAV file
/// 
/// Parameters:
///   - input_path: Null-terminated C string with input file path
///   - samples_ptr: Pointer to receive audio samples (allocated by caller)
///   - max_length: Maximum number of samples that can be written
///   - sample_rate_ptr: Pointer to receive sample rate
///   - channels_ptr: Pointer to receive channel count
///   - bit_depth_ptr: Pointer to receive bit depth
/// 
/// Returns:
///   - Number of samples loaded on success
///   - -1 on error
export fn load_audio_wav(
    input_path: [*:0]const u8,
    samples_ptr: [*]f32,
    max_length: usize,
    sample_rate_ptr: *u32,
    channels_ptr: *u8,
    bit_depth_ptr: *u8,
) callconv(.C) c_int {
    // Get allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Convert C string to Zig string
    const path = std.mem.span(input_path);
    
    // Read WAV file
    const buffer = audio_io.readWAV(path, allocator) catch |err| {
        std.debug.print("WAV read error: {}\n", .{err});
        return -1;
    };
    defer allocator.free(buffer.samples);
    
    // Check if buffer fits
    if (buffer.samples.len > max_length) {
        std.debug.print("Buffer too small: need {} samples, have {}\n", 
            .{buffer.samples.len, max_length});
        return -1;
    }
    
    // Copy samples
    const dest_slice = samples_ptr[0..buffer.samples.len];
    @memcpy(dest_slice, buffer.samples);
    
    // Set output parameters
    sample_rate_ptr.* = buffer.sample_rate;
    channels_ptr.* = buffer.channels;
    bit_depth_ptr.* = buffer.bit_depth;
    
    return @intCast(buffer.samples.len);
}

/// Get version information
/// 
/// Returns:
///   - Null-terminated C string with version
export fn get_version() callconv(.C) [*:0]const u8 {
    return "AudioLabShimmy FFI v1.0.0";
}

/// Test FFI connection
/// 
/// Parameters:
///   - test_value: A test value to echo back
/// 
/// Returns:
///   - The same test value + 1
export fn test_ffi_connection(test_value: i32) callconv(.C) i32 {
    return test_value + 1;
}

// Test function for C compatibility
test "FFI exports" {
    // Test that functions can be called
    const result = test_ffi_connection(42);
    try std.testing.expectEqual(43, result);
    
    const version = get_version();
    try std.testing.expect(std.mem.len(version) > 0);
}
