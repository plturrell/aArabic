// Build script for Mojo Package Manager
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Create root module from main.zig
    const root_module = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Create mojo-pkg executable
    const exe = b.addExecutable(.{
        .name = "mojo-pkg",
        .root_module = root_module,
    });
    
    b.installArtifact(exe);
    
    // Add run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    
    const run_step = b.step("run", "Run mojo-pkg");
    run_step.dependOn(&run_cmd.step);
    
    // Add test step for all modules
    const test_step = b.step("test", "Run all tests");
    
    const modules = [_][]const u8{
        "manifest.zig",
        "workspace.zig",
        "resolver.zig",
        "zig_bridge.zig",
        "cli.zig",
    };
    
    for (modules) |module| {
        const test_module = b.createModule(.{
            .root_source_file = b.path(module),
            .target = target,
            .optimize = optimize,
        });
        
        const module_tests = b.addTest(.{
            .root_module = test_module,
        });
        
        const run_module_tests = b.addRunArtifact(module_tests);
        test_step.dependOn(&run_module_tests.step);
    }
}
