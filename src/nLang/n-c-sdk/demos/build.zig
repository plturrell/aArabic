const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sdl_include_opt = b.option([]const u8, "sdl-include", "Path to SDL2 headers (optional)");
    const sdl_lib_opt = b.option([]const u8, "sdl-lib", "Path to SDL2 libraries (optional)");

    // Create SDL2 module with proper include paths
    const sdl_module = b.createModule(.{
        .root_source_file = b.path("sdl_bindings.zig"),
        .target = target,
        .optimize = optimize,
    });

    // We need to create a dummy executable to set include paths for C imports
    const sdl_dummy = b.addExecutable(.{
        .name = "sdl_dummy",
        .root_module = sdl_module,
    });

    if (sdl_include_opt) |include_path| {
        sdl_dummy.addIncludePath(.{ .cwd_relative = include_path });
    } else if (target.result.os.tag == .macos) {
        sdl_dummy.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include/SDL2" });
    }

    // Particle Physics Demo
    const particle_demo = b.addExecutable(.{
        .name = "particle_physics_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("particle_physics_demo.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(particle_demo);

    const run_cmd = b.addRunArtifact(particle_demo);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the particle physics demo");
    run_step.dependOn(&run_cmd.step);

    // Fractal Demo
    const fractal_demo = b.addExecutable(.{
        .name = "fractal_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("fractal_demo.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    fractal_demo.linkSystemLibrary("SDL2");
    fractal_demo.linkLibC();
    if (target.result.os.tag == .windows) {
        fractal_demo.linkSystemLibrary("gdi32");
        fractal_demo.linkSystemLibrary("user32");
        fractal_demo.linkSystemLibrary("shell32");
    } else if (target.result.os.tag == .linux) {
        fractal_demo.linkSystemLibrary("pthread");
    }

    if (sdl_include_opt) |include_path| {
        fractal_demo.addIncludePath(.{ .cwd_relative = include_path });
    } else if (target.result.os.tag == .macos) {
        fractal_demo.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include/SDL2" });
        fractal_demo.addIncludePath(.{ .cwd_relative = "/usr/local/include/SDL2" });
    }

    if (sdl_lib_opt) |lib_path| {
        fractal_demo.addLibraryPath(.{ .cwd_relative = lib_path });
    } else if (target.result.os.tag == .macos) {
        fractal_demo.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
        fractal_demo.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
    }

    b.installArtifact(fractal_demo);

    // Run command
    const run_fractal = b.addRunArtifact(fractal_demo);
    run_fractal.step.dependOn(b.getInstallStep());

    // Allow passing args to the run command
    if (b.args) |args| {
        run_fractal.addArgs(args);
    }

    const fractal_step = b.step("run-fractal", "Run the fractal visualization demo");
    fractal_step.dependOn(&run_fractal.step);

    // Benchmark Suite
    const benchmark = b.addExecutable(.{
        .name = "benchmark_suite",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmark_suite.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(benchmark);

    const run_bench = b.addRunArtifact(benchmark);
    run_bench.step.dependOn(b.getInstallStep());

    const bench_step = b.step("benchmark", "Run the performance benchmark suite");
    bench_step.dependOn(&run_bench.step);

    // SDL2 Visual Particle Demo (Phase 2 starter)
    const visual_demo = b.addExecutable(.{
        .name = "visual_particle_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("visual_particle_demo.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sdl", .module = sdl_module },
            },
        }),
    });

    visual_demo.linkSystemLibrary("SDL2");
    if (target.result.os.tag == .windows) visual_demo.linkSystemLibrary("SDL2main");

    if (sdl_include_opt) |include_path| {
        visual_demo.addIncludePath(.{ .cwd_relative = include_path });
    } else if (target.result.os.tag == .macos) {
        visual_demo.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include/SDL2" });
        visual_demo.addIncludePath(.{ .cwd_relative = "/usr/local/include/SDL2" });
    }

    if (sdl_lib_opt) |lib_path| {
        visual_demo.addLibraryPath(.{ .cwd_relative = lib_path });
    } else if (target.result.os.tag == .macos) {
        visual_demo.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
        visual_demo.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
    }

    b.installArtifact(visual_demo);

    const run_visual = b.addRunArtifact(visual_demo);
    run_visual.step.dependOn(b.getInstallStep());

    const visual_step = b.step("run-visual", "Run the SDL2 visual particle demo");
    visual_step.dependOn(&run_visual.step);

    // Multi-threaded Particle Physics Demo
    const particle_mt = b.addExecutable(.{
        .name = "particle_physics_mt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("particle_physics_mt.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(particle_mt);

    const run_mt = b.addRunArtifact(particle_mt);
    run_mt.step.dependOn(b.getInstallStep());

    const mt_step = b.step("run-mt", "Run the multi-threaded particle physics demo (1M particles)");
    mt_step.dependOn(&run_mt.step);

    // Dashboard
    const dashboard = b.addExecutable(.{
        .name = "demo_dashboard",
        .root_module = b.createModule(.{
            .root_source_file = b.path("demo_dashboard.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    dashboard.linkSystemLibrary("SDL2");
    dashboard.linkLibC();

    if (sdl_include_opt) |include_path| {
        dashboard.addIncludePath(.{ .cwd_relative = include_path });
    } else if (target.result.os.tag == .macos) {
        dashboard.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include/SDL2" });
    }

    if (sdl_lib_opt) |lib_path| {
        dashboard.addLibraryPath(.{ .cwd_relative = lib_path });
    } else if (target.result.os.tag == .macos) {
        dashboard.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    }

    b.installArtifact(dashboard);

    const run_dashboard = b.addRunArtifact(dashboard);
    run_dashboard.step.dependOn(b.getInstallStep());

    const dashboard_step = b.step("dashboard", "Run the demo dashboard (main entry point)");
    dashboard_step.dependOn(&run_dashboard.step);

    // Complete Visual Particle Demo
    const visual_complete = b.addExecutable(.{
        .name = "visual_particle_demo_complete",
        .root_module = b.createModule(.{
            .root_source_file = b.path("visual_particle_demo_complete.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    visual_complete.linkSystemLibrary("SDL2");
    visual_complete.linkLibC();

    if (sdl_include_opt) |include_path| {
        visual_complete.addIncludePath(.{ .cwd_relative = include_path });
    } else if (target.result.os.tag == .macos) {
        visual_complete.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include/SDL2" });
    }

    if (sdl_lib_opt) |lib_path| {
        visual_complete.addLibraryPath(.{ .cwd_relative = lib_path });
    } else if (target.result.os.tag == .macos) {
        visual_complete.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    }

    b.installArtifact(visual_complete);

    const run_visual_complete = b.addRunArtifact(visual_complete);
    run_visual_complete.step.dependOn(b.getInstallStep());

    const visual_complete_step = b.step("run-visual-complete", "Run the complete visual particle demo with metrics");
    visual_complete_step.dependOn(&run_visual_complete.step);

    // HPC Benchmark Suite
    const hpc_benchmark = b.addExecutable(.{
        .name = "hpc_benchmark_suite",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/hpc_benchmark_suite.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(hpc_benchmark);

    const run_hpc = b.addRunArtifact(hpc_benchmark);
    run_hpc.step.dependOn(b.getInstallStep());

    const hpc_step = b.step("hpc-benchmark", "Run HPC benchmark suite (STREAM, LINPACK, latency, scaling, SIMD)");
    hpc_step.dependOn(&run_hpc.step);

    // Sudoku HPC Benchmark
    const sudoku_benchmark = b.addExecutable(.{
        .name = "sudoku_hpc_benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/sudoku_hpc_benchmark.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(sudoku_benchmark);

    const run_sudoku = b.addRunArtifact(sudoku_benchmark);
    run_sudoku.step.dependOn(b.getInstallStep());

    const sudoku_step = b.step("sudoku-benchmark", "Run Sudoku HPC benchmark (algorithm comparison)");
    sudoku_step.dependOn(&run_sudoku.step);

    // Combined HPC benchmarks (run both and pipe to JSON)
    const combined_hpc_step = b.step("run-all-hpc", "Run all HPC benchmarks and generate combined JSON");
    combined_hpc_step.dependOn(&run_hpc.step);
}
