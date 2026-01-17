const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ========================================
    // Test Suite
    // ========================================
    
    // Test audio_types
    const audio_types_tests = b.addTest(.{
        .test_runner = b.path("zig/audio_types.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Test wav_format
    const wav_format_tests = b.addTest(.{
        .test_runner = b.path("zig/wav_format.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Test audio_io
    const audio_io_tests = b.addTest(.{
        .test_runner = b.path("zig/audio_io.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_audio_types_tests = b.addRunArtifact(audio_types_tests);
    const run_wav_format_tests = b.addRunArtifact(wav_format_tests);
    const run_audio_io_tests = b.addRunArtifact(audio_io_tests);
    
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_audio_types_tests.step);
    test_step.dependOn(&run_wav_format_tests.step);
    test_step.dependOn(&run_audio_io_tests.step);
}
