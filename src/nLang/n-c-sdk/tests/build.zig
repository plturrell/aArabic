const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main test step
    const test_step = b.step("test", "Run all tests");

    // Unit tests
    const unit_tests_step = b.step("test-unit", "Run unit tests");
    addUnitTests(b, unit_tests_step, target, optimize);
    test_step.dependOn(unit_tests_step);

    // Integration tests
    const integration_tests_step = b.step("test-integration", "Run integration tests");
    addIntegrationTests(b, integration_tests_step, target, optimize);
    test_step.dependOn(integration_tests_step);

    // Load tests
    const load_tests_step = b.step("test-load", "Run load tests");
    addLoadTests(b, load_tests_step, target, optimize);
    test_step.dependOn(load_tests_step);
}

fn addUnitTests(b: *std.Build, step: *std.Build.Step, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const hana_unit_tests = b.addTest(.{
        .name = "hana_unit_tests",
        .root_source_file = b.path("unit/hana_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    step.dependOn(&b.addRunArtifact(hana_unit_tests).step);
}

fn addIntegrationTests(b: *std.Build, step: *std.Build.Step, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const http_integration_tests = b.addTest(.{
        .name = "http_integration_tests",
        .root_source_file = b.path("integration/http_integration_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const hana_integration_tests = b.addTest(.{
        .name = "hana_integration_tests",
        .root_source_file = b.path("integration/hana_integration_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    step.dependOn(&b.addRunArtifact(http_integration_tests).step);
    step.dependOn(&b.addRunArtifact(hana_integration_tests).step);
}

fn addLoadTests(b: *std.Build, step: *std.Build.Step, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const load_tests = b.addTest(.{
        .name = "load_tests",
        .root_source_file = b.path("load/load_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    step.dependOn(&b.addRunArtifact(load_tests).step);
}