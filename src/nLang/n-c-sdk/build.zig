const std = @import("std");
const builtin = std.builtin;
const tests = @import("test/tests.zig");
const BufMap = std.BufMap;
const mem = std.mem;
const io = std.io;
const fs = std.fs;
const InstallDirectoryOptions = std.Build.InstallDirectoryOptions;
const assert = std.debug.assert;
const DevEnv = @import("src/dev.zig").Env;
const ValueInterpretMode = enum { direct, by_name };

const zig_version: std.SemanticVersion = .{ .major = 0, .minor = 15, .patch = 2 };
const stack_size = 46 * 1024 * 1024;

pub fn build(b: *std.Build) !void {
    const only_c = b.option(bool, "only-c", "Translate the Zig compiler to C code, with only the C backend enabled") orelse false;
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .ofmt = if (only_c) .c else null,
        },
    });
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseSafe,  // CUSTOM: Default to ReleaseSafe for production
    });

    const flat = b.option(bool, "flat", "Put files into the installation prefix in a manner suited for upstream distribution rather than a posix file system hierarchy standard") orelse false;
    const single_threaded = b.option(bool, "single-threaded", "Build artifacts that run in single threaded mode");
    const _use_zig_libcxx = b.option(bool, "use-zig-libcxx", "If libc++ is needed, use zig's bundled version, don't try to integrate with the system") orelse false;
    _ = _use_zig_libcxx;

    const test_step = b.step("test", "Run all the tests");
    const skip_install_lib_files = b.option(bool, "no-lib", "skip copying of lib/ files and langref to installation prefix. Useful for development") orelse false;
    const skip_install_langref = b.option(bool, "no-langref", "skip copying of langref to the installation prefix") orelse skip_install_lib_files;
    const std_docs = b.option(bool, "std-docs", "include standard library autodocs") orelse false;
    const no_bin = b.option(bool, "no-bin", "skip emitting compiler binary") orelse false;
    const enable_superhtml = b.option(bool, "enable-superhtml", "Check langref output HTML validity") orelse false;

    const langref_file = generateLangRef(b);
    const install_langref = b.addInstallFileWithDir(langref_file, .prefix, "doc/langref.html");
    const check_langref = superHtmlCheck(b, langref_file);
    if (enable_superhtml) install_langref.step.dependOn(check_langref);

    const check_autodocs = superHtmlCheck(b, b.path("lib/docs/index.html"));
    if (enable_superhtml) {
        test_step.dependOn(check_langref);
        test_step.dependOn(check_autodocs);
    }
    if (!skip_install_langref) {
        b.getInstallStep().dependOn(&install_langref.step);
    }

    const autodoc_test = b.addObject(.{
        .name = "std",
        .zig_lib_dir = b.path("lib"),
        .root_module = b.createModule(.{
            .root_source_file = b.path("lib/std/std.zig"),
            .target = target,
            .optimize = .Debug,
        }),
    });
    const install_std_docs = b.addInstallDirectory(.{
        .source_dir = autodoc_test.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "doc/std",
    });
    //if (enable_tidy) install_std_docs.step.dependOn(check_autodocs);
    if (std_docs) {
        b.getInstallStep().dependOn(&install_std_docs.step);
    }

    if (flat) {
        b.installFile("LICENSE", "LICENSE");
        b.installFile("README.md", "README.md");
    }

    const langref_step = b.step("langref", "Build and install the language reference");
    langref_step.dependOn(&install_langref.step);

    const std_docs_step = b.step("std-docs", "Build and install the standard library documentation");
    std_docs_step.dependOn(&install_std_docs.step);

    const docs_step = b.step("docs", "Build and install documentation");
    docs_step.dependOn(langref_step);
    docs_step.dependOn(std_docs_step);

    const skip_debug = b.option(bool, "skip-debug", "Main test suite skips debug builds") orelse false;
    const skip_release = b.option(bool, "skip-release", "Main test suite skips release builds") orelse false;
    const skip_release_small = b.option(bool, "skip-release-small", "Main test suite skips release-small builds") orelse skip_release;
    const skip_release_fast = b.option(bool, "skip-release-fast", "Main test suite skips release-fast builds") orelse skip_release;
    const skip_release_safe = b.option(bool, "skip-release-safe", "Main test suite skips release-safe builds") orelse skip_release;
    const skip_non_native = b.option(bool, "skip-non-native", "Main test suite skips non-native builds") orelse false;
    const skip_libc = b.option(bool, "skip-libc", "Main test suite skips tests that link libc") orelse false;
    const skip_single_threaded = b.option(bool, "skip-single-threaded", "Main test suite skips tests that are single-threaded") orelse false;
    const skip_compile_errors = b.option(bool, "skip-compile-errors", "Main test suite skips compile error tests") orelse false;
    const skip_translate_c = b.option(bool, "skip-translate-c", "Main test suite skips translate-c tests") orelse false;
    const skip_run_translated_c = b.option(bool, "skip-run-translated-c", "Main test suite skips run-translated-c tests") orelse false;
    const skip_freebsd = b.option(bool, "skip-freebsd", "Main test suite skips targets with freebsd OS") orelse false;
    const skip_netbsd = b.option(bool, "skip-netbsd", "Main test suite skips targets with netbsd OS") orelse false;
    const skip_windows = b.option(bool, "skip-windows", "Main test suite skips targets with windows OS") orelse false;
    const skip_macos = b.option(bool, "skip-macos", "Main test suite skips targets with macos OS") orelse false;
    const skip_linux = b.option(bool, "skip-linux", "Main test suite skips targets with linux OS") orelse false;
    const skip_llvm = b.option(bool, "skip-llvm", "Main test suite skips targets that use LLVM backend") orelse false;

    const only_install_lib_files = b.option(bool, "lib-files-only", "Only install library files") orelse false;

    // LLVM/Clang C++ dependencies have been removed - these options are kept for backwards compatibility
    // but are now always false/ignored
    const static_llvm = false;
    _ = b.option(bool, "static-llvm", "(DEPRECATED) LLVM has been removed - this option is ignored");
    const enable_llvm = false;
    _ = b.option(bool, "enable-llvm", "(DEPRECATED) LLVM has been removed - this option is ignored");
    const llvm_has_m68k = false;
    _ = b.option(bool, "llvm-has-m68k", "(DEPRECATED) LLVM has been removed - this option is ignored");
    const llvm_has_csky = false;
    _ = b.option(bool, "llvm-has-csky", "(DEPRECATED) LLVM has been removed - this option is ignored");
    const llvm_has_arc = false;
    _ = b.option(bool, "llvm-has-arc", "(DEPRECATED) LLVM has been removed - this option is ignored");
    const llvm_has_xtensa = false;
    _ = b.option(bool, "llvm-has-xtensa", "(DEPRECATED) LLVM has been removed - this option is ignored");
    const enable_ios_sdk = b.option(bool, "enable-ios-sdk", "Run tests requiring presence of iOS SDK and frameworks") orelse false;
    const enable_macos_sdk = b.option(bool, "enable-macos-sdk", "Run tests requiring presence of macOS SDK and frameworks") orelse enable_ios_sdk;
    const enable_symlinks_windows = b.option(bool, "enable-symlinks-windows", "Run tests requiring presence of symlinks on Windows") orelse false;
    const config_h_path_option = b.option([]const u8, "config_h", "Path to the generated config.h");

    if (!skip_install_lib_files) {
        b.installDirectory(.{
            .source_dir = b.path("lib"),
            .install_dir = if (flat) .prefix else .lib,
            .install_subdir = if (flat) "lib" else "zig",
            .exclude_extensions = &[_][]const u8{
                // exclude files from lib/std/compress/testdata
                ".gz",
                ".z.0",
                ".z.9",
                ".zst.3",
                ".zst.19",
                "rfc1951.txt",
                "rfc1952.txt",
                "rfc8478.txt",
                // exclude files from lib/std/compress/flate/testdata
                ".expect",
                ".expect-noinput",
                ".golden",
                ".input",
                "compress-e.txt",
                "compress-gettysburg.txt",
                "compress-pi.txt",
                "rfc1951.txt",
                // exclude files from lib/std/compress/lzma/testdata
                ".lzma",
                // exclude files from lib/std/compress/xz/testdata
                ".xz",
                // exclude files from lib/std/tz/
                ".tzif",
                // exclude files from lib/std/tar/testdata
                ".tar",
                // others
                "README.md",
            },
            .blank_extensions = &[_][]const u8{
                "test.zig",
            },
        });
    }

    if (only_install_lib_files)
        return;

    const entitlements = b.option([]const u8, "entitlements", "Path to entitlements file for hot-code swapping without sudo on macOS");
    // Tracy C++ integration has been removed - pure Zig profiling only
    // These options are kept for API compatibility but Tracy is always disabled
    _ = b.option([]const u8, "tracy", "[DEPRECATED] Tracy C++ integration has been removed");
    _ = b.option(bool, "tracy-callstack", "[DEPRECATED] Tracy C++ integration has been removed");
    _ = b.option(bool, "tracy-allocation", "[DEPRECATED] Tracy C++ integration has been removed");
    _ = b.option(u32, "tracy-callstack-depth", "[DEPRECATED] Tracy C++ integration has been removed");
    const debug_gpa = b.option(bool, "debug-allocator", "Force the compiler to use DebugAllocator") orelse false;
    const link_libc = b.option(bool, "force-link-libc", "Force self-hosted compiler to link libc") orelse only_c;
    const sanitize_thread = b.option(bool, "sanitize-thread", "Enable thread-sanitization") orelse false;
    const strip = b.option(bool, "strip", "Omit debug information");
    const valgrind = b.option(bool, "valgrind", "Enable valgrind integration");
    const pie = b.option(bool, "pie", "Produce a Position Independent Executable");
    const value_interpret_mode = b.option(ValueInterpretMode, "value-interpret-mode", "How the compiler translates between 'std.builtin' types and its internal datastructures") orelse .direct;
    const value_tracing = b.option(bool, "value-tracing", "Enable extra state tracking to help troubleshoot bugs in the compiler (using the std.debug.Trace API)") orelse false;

    const mem_leak_frames: u32 = b.option(u32, "mem-leak-frames", "How many stack frames to print when a memory leak occurs. Tests get 2x this amount.") orelse blk: {
        if (strip == true) break :blk @as(u32, 0);
        if (optimize != .Debug) break :blk 0;
        break :blk 4;
    };

    const exe = addCompilerStep(b, .{
        .optimize = optimize,
        .target = target,
        .strip = strip,
        .valgrind = valgrind,
        .sanitize_thread = sanitize_thread,
        .single_threaded = single_threaded,
    });
    exe.pie = pie;
    exe.entitlements = entitlements;

    const use_llvm = b.option(bool, "use-llvm", "Use the llvm backend");
    exe.use_llvm = use_llvm;
    exe.use_lld = use_llvm;

    // CUSTOM: Enable LTO for optimized builds (better performance)
    if (optimize != .Debug and use_llvm != false) {
        exe.want_lto = true;
    }

    // CUSTOM: ReleaseBalanced mode support
    // This mode enables PGO-guided optimization with safety checks
    const enable_balanced = b.option(bool, "balanced", "Enable ReleaseBalanced mode features (PGO + safety)") orelse false;
    const pgo_profile = b.option([]const u8, "pgo-profile", "Path to PGO profile data for balanced mode");
    
    if (enable_balanced) {
        std.debug.print("🎯 ReleaseBalanced mode enabled\n", .{});
        
        if (pgo_profile) |profile_path| {
            std.debug.print("📊 Using PGO profile: {s}\n", .{profile_path});
            // Note: PGO integration would happen in the compiler itself
            // This is a placeholder for build-time configuration
        } else {
            std.debug.print("⚠️  No PGO profile specified. Run with -Dpgo-profile=path/to/profile.pgo\n", .{});
        }
    }

    if (no_bin) {
        b.getInstallStep().dependOn(&exe.step);
    } else {
        const install_exe = b.addInstallArtifact(exe, .{
            .dest_dir = if (flat) .{ .override = .prefix } else .default,
        });
        b.getInstallStep().dependOn(&install_exe.step);
    }

    test_step.dependOn(&exe.step);

    const exe_options = b.addOptions();
    exe.root_module.addOptions("build_options", exe_options);

    exe_options.addOption(u32, "mem_leak_frames", mem_leak_frames);
    exe_options.addOption(bool, "skip_non_native", skip_non_native);
    exe_options.addOption(bool, "have_llvm", enable_llvm);
    exe_options.addOption(bool, "llvm_has_m68k", llvm_has_m68k);
    exe_options.addOption(bool, "llvm_has_csky", llvm_has_csky);
    exe_options.addOption(bool, "llvm_has_arc", llvm_has_arc);
    exe_options.addOption(bool, "llvm_has_xtensa", llvm_has_xtensa);
    exe_options.addOption(bool, "debug_gpa", debug_gpa);
    exe_options.addOption(DevEnv, "dev", b.option(DevEnv, "dev", "Build a compiler with a reduced feature set for development of specific features") orelse if (only_c) .bootstrap else .full);
    exe_options.addOption(ValueInterpretMode, "value_interpret_mode", value_interpret_mode);

    if (link_libc) {
        exe.root_module.link_libc = true;
    }

    const is_debug = optimize == .Debug;
    const enable_debug_extensions = b.option(bool, "debug-extensions", "Enable commands and options useful for debugging the compiler") orelse is_debug;
    const enable_logging = b.option(bool, "log", "Enable debug logging with --debug-log") orelse is_debug;
    const enable_link_snapshots = b.option(bool, "link-snapshot", "Whether to enable linker state snapshots") orelse false;

    const opt_version_string = b.option([]const u8, "version-string", "Override Zig version string. Default is to find out with git.");
    const version_slice = if (opt_version_string) |version| version else v: {
        if (!std.process.can_spawn) {
            std.debug.print("error: version info cannot be retrieved from git. Zig version must be provided using -Dversion-string\n", .{});
            std.process.exit(1);
        }
        const version_string = b.fmt("{d}.{d}.{d}", .{ zig_version.major, zig_version.minor, zig_version.patch });

        var code: u8 = undefined;
        const git_describe_untrimmed = b.runAllowFail(&[_][]const u8{
            "git",
            "-C", b.build_root.path orelse ".", // affects the --git-dir argument
            "--git-dir", ".git", // affected by the -C argument
            "describe", "--match",    "*.*.*", //
            "--tags",   "--abbrev=9",
        }, &code, .Ignore) catch {
            break :v version_string;
        };
        const git_describe = mem.trim(u8, git_describe_untrimmed, " \n\r");

        switch (mem.count(u8, git_describe, "-")) {
            0 => {
                // Tagged release version (e.g. 0.10.0).
                if (!mem.eql(u8, git_describe, version_string)) {
                    std.debug.print("Zig version '{s}' does not match Git tag '{s}'\n", .{ version_string, git_describe });
                    std.process.exit(1);
                }
                break :v version_string;
            },
            2 => {
                // Untagged development build (e.g. 0.10.0-dev.2025+ecf0050a9).
                var it = mem.splitScalar(u8, git_describe, '-');
                const tagged_ancestor = it.first();
                const commit_height = it.next().?;
                const commit_id = it.next().?;

                const ancestor_ver = try std.SemanticVersion.parse(tagged_ancestor);
                if (zig_version.order(ancestor_ver) != .gt) {
                    std.debug.print("Zig version '{f}' must be greater than tagged ancestor '{f}'\n", .{ zig_version, ancestor_ver });
                    std.process.exit(1);
                }

                // Check that the commit hash is prefixed with a 'g' (a Git convention).
                if (commit_id.len < 1 or commit_id[0] != 'g') {
                    std.debug.print("Unexpected `git describe` output: {s}\n", .{git_describe});
                    break :v version_string;
                }

                // The version is reformatted in accordance with the https://semver.org specification.
                break :v b.fmt("{s}-dev.{s}+{s}", .{ version_string, commit_height, commit_id[1..] });
            },
            else => {
                std.debug.print("Unexpected `git describe` output: {s}\n", .{git_describe});
                break :v version_string;
            },
        }
    };
    const version = try b.allocator.dupeZ(u8, version_slice);
    exe_options.addOption([:0]const u8, "version", version);

    // LLVM/Clang C++ dependencies have been removed - pure Zig implementation is now used
    _ = static_llvm; // Suppress unused variable warning
    _ = config_h_path_option; // Suppress unused variable warning

    const semver = try std.SemanticVersion.parse(version);
    exe_options.addOption(std.SemanticVersion, "semver", semver);

    exe_options.addOption(bool, "enable_debug_extensions", enable_debug_extensions);
    exe_options.addOption(bool, "enable_logging", enable_logging);
    exe_options.addOption(bool, "enable_link_snapshots", enable_link_snapshots);
    // Tracy C++ integration has been removed - always disabled
    exe_options.addOption(bool, "enable_tracy", false);
    exe_options.addOption(bool, "enable_tracy_callstack", false);
    exe_options.addOption(bool, "enable_tracy_allocation", false);
    exe_options.addOption(u32, "tracy_callstack_depth", 0);
    exe_options.addOption(bool, "value_tracing", value_tracing);

    const test_filters = b.option([]const []const u8, "test-filter", "Skip tests that do not match any filter") orelse &[0][]const u8{};
    const test_target_filters = b.option([]const []const u8, "test-target-filter", "Skip tests whose target triple do not match any filter") orelse &[0][]const u8{};
    const test_extra_targets = b.option(bool, "test-extra-targets", "Enable running module tests for additional targets") orelse false;

    var chosen_opt_modes_buf: [4]builtin.OptimizeMode = undefined;
    var chosen_mode_index: usize = 0;
    if (!skip_debug) {
        chosen_opt_modes_buf[chosen_mode_index] = builtin.OptimizeMode.Debug;
        chosen_mode_index += 1;
    }
    if (!skip_release_safe) {
        chosen_opt_modes_buf[chosen_mode_index] = builtin.OptimizeMode.ReleaseSafe;
        chosen_mode_index += 1;
    }
    if (!skip_release_fast) {
        chosen_opt_modes_buf[chosen_mode_index] = builtin.OptimizeMode.ReleaseFast;
        chosen_mode_index += 1;
    }
    if (!skip_release_small) {
        chosen_opt_modes_buf[chosen_mode_index] = builtin.OptimizeMode.ReleaseSmall;
        chosen_mode_index += 1;
    }
    const optimization_modes = chosen_opt_modes_buf[0..chosen_mode_index];

    const fmt_include_paths = &.{ "lib", "src", "test", "tools", "build.zig", "build.zig.zon" };
    const fmt_exclude_paths = &.{ "test/cases", "test/behavior/zon" };
    const do_fmt = b.addFmt(.{
        .paths = fmt_include_paths,
        .exclude_paths = fmt_exclude_paths,
    });
    b.step("fmt", "Modify source files in place to have conforming formatting").dependOn(&do_fmt.step);

    const check_fmt = b.step("test-fmt", "Check source files having conforming formatting");
    check_fmt.dependOn(&b.addFmt(.{
        .paths = fmt_include_paths,
        .exclude_paths = fmt_exclude_paths,
        .check = true,
    }).step);
    test_step.dependOn(check_fmt);

    const test_cases_step = b.step("test-cases", "Run the main compiler test cases");
    try tests.addCases(b, test_cases_step, target, .{
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
        .skip_compile_errors = skip_compile_errors,
        .skip_non_native = skip_non_native,
        .skip_freebsd = skip_freebsd,
        .skip_netbsd = skip_netbsd,
        .skip_windows = skip_windows,
        .skip_macos = skip_macos,
        .skip_linux = skip_linux,
        .skip_llvm = skip_llvm,
        .skip_libc = skip_libc,
    }, .{
        .skip_translate_c = skip_translate_c,
        .skip_run_translated_c = skip_run_translated_c,
    }, .{
        .enable_llvm = enable_llvm,
        .llvm_has_m68k = llvm_has_m68k,
        .llvm_has_csky = llvm_has_csky,
        .llvm_has_arc = llvm_has_arc,
        .llvm_has_xtensa = llvm_has_xtensa,
    });
    test_step.dependOn(test_cases_step);

    const test_modules_step = b.step("test-modules", "Run the per-target module tests");
    test_step.dependOn(test_modules_step);

    test_modules_step.dependOn(tests.addModuleTests(b, .{
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
        .test_extra_targets = test_extra_targets,
        .root_src = "test/behavior.zig",
        .name = "behavior",
        .desc = "Run the behavior tests",
        .optimize_modes = optimization_modes,
        .include_paths = &.{},
        .skip_single_threaded = skip_single_threaded,
        .skip_non_native = skip_non_native,
        .skip_freebsd = skip_freebsd,
        .skip_netbsd = skip_netbsd,
        .skip_windows = skip_windows,
        .skip_macos = skip_macos,
        .skip_linux = skip_linux,
        .skip_llvm = skip_llvm,
        .skip_libc = skip_libc,
        // 3888779264 was observed on an x86_64-linux-gnu host.
        .max_rss = 4000000000,
    }));

    test_modules_step.dependOn(tests.addModuleTests(b, .{
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
        .test_extra_targets = test_extra_targets,
        .root_src = "test/c_import.zig",
        .name = "c-import",
        .desc = "Run the @cImport tests",
        .optimize_modes = optimization_modes,
        .include_paths = &.{"test/c_import"},
        .skip_single_threaded = true,
        .skip_non_native = skip_non_native,
        .skip_freebsd = skip_freebsd,
        .skip_netbsd = skip_netbsd,
        .skip_windows = skip_windows,
        .skip_macos = skip_macos,
        .skip_linux = skip_linux,
        .skip_llvm = skip_llvm,
        .skip_libc = skip_libc,
    }));

    test_modules_step.dependOn(tests.addModuleTests(b, .{
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
        .test_extra_targets = test_extra_targets,
        .root_src = "lib/compiler_rt.zig",
        .name = "compiler-rt",
        .desc = "Run the compiler_rt tests",
        .optimize_modes = optimization_modes,
        .include_paths = &.{},
        .skip_single_threaded = true,
        .skip_non_native = skip_non_native,
        .skip_freebsd = skip_freebsd,
        .skip_netbsd = skip_netbsd,
        .skip_windows = skip_windows,
        .skip_macos = skip_macos,
        .skip_linux = skip_linux,
        .skip_llvm = skip_llvm,
        .skip_libc = true,
        .no_builtin = true,
    }));

    test_modules_step.dependOn(tests.addModuleTests(b, .{
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
        .test_extra_targets = test_extra_targets,
        .root_src = "lib/c.zig",
        .name = "zigc",
        .desc = "Run the zigc tests",
        .optimize_modes = optimization_modes,
        .include_paths = &.{},
        .skip_single_threaded = true,
        .skip_non_native = skip_non_native,
        .skip_freebsd = skip_freebsd,
        .skip_netbsd = skip_netbsd,
        .skip_windows = skip_windows,
        .skip_macos = skip_macos,
        .skip_linux = skip_linux,
        .skip_llvm = skip_llvm,
        .skip_libc = true,
        .no_builtin = true,
    }));

    test_modules_step.dependOn(tests.addModuleTests(b, .{
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
        .test_extra_targets = test_extra_targets,
        .root_src = "lib/std/std.zig",
        .name = "std",
        .desc = "Run the standard library tests",
        .optimize_modes = optimization_modes,
        .include_paths = &.{},
        .skip_single_threaded = skip_single_threaded,
        .skip_non_native = skip_non_native,
        .skip_freebsd = skip_freebsd,
        .skip_netbsd = skip_netbsd,
        .skip_windows = skip_windows,
        .skip_macos = skip_macos,
        .skip_linux = skip_linux,
        .skip_llvm = skip_llvm,
        .skip_libc = skip_libc,
        // I observed a value of 5605064704 on the M2 CI.
        .max_rss = 6165571174,
    }));

    const unit_tests_step = b.step("test-unit", "Run the compiler source unit tests");
    test_step.dependOn(unit_tests_step);

    const unit_tests = b.addTest(.{
        .root_module = addCompilerMod(b, .{
            .optimize = optimize,
            .target = target,
            .single_threaded = single_threaded,
        }),
        .filters = test_filters,
        .use_llvm = use_llvm,
        .use_lld = use_llvm,
        .zig_lib_dir = b.path("lib"),
    });
    if (link_libc) {
        unit_tests.root_module.link_libc = true;
    }
    unit_tests.root_module.addOptions("build_options", exe_options);
    unit_tests_step.dependOn(&b.addRunArtifact(unit_tests).step);

    test_step.dependOn(tests.addCompareOutputTests(b, test_filters, optimization_modes));
    test_step.dependOn(tests.addStandaloneTests(
        b,
        optimization_modes,
        enable_macos_sdk,
        enable_ios_sdk,
        enable_symlinks_windows,
    ));
    test_step.dependOn(tests.addCAbiTests(b, .{
        .test_target_filters = test_target_filters,
        .skip_non_native = skip_non_native,
        .skip_freebsd = skip_freebsd,
        .skip_netbsd = skip_netbsd,
        .skip_windows = skip_windows,
        .skip_macos = skip_macos,
        .skip_linux = skip_linux,
        .skip_llvm = skip_llvm,
        .skip_release = skip_release,
    }));
    test_step.dependOn(tests.addLinkTests(b, enable_macos_sdk, enable_ios_sdk, enable_symlinks_windows));
    test_step.dependOn(tests.addStackTraceTests(b, test_filters, optimization_modes));
    test_step.dependOn(tests.addCliTests(b));
    test_step.dependOn(tests.addAssembleAndLinkTests(b, test_filters, optimization_modes));
    if (tests.addDebuggerTests(b, .{
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
        .gdb = b.option([]const u8, "gdb", "path to gdb binary"),
        .lldb = b.option([]const u8, "lldb", "path to lldb binary"),
        .optimize_modes = optimization_modes,
        .skip_single_threaded = skip_single_threaded,
        .skip_libc = skip_libc,
    })) |test_debugger_step| test_step.dependOn(test_debugger_step);
    if (tests.addLlvmIrTests(b, .{
        .enable_llvm = enable_llvm,
        .test_filters = test_filters,
        .test_target_filters = test_target_filters,
    })) |test_llvm_ir_step| test_step.dependOn(test_llvm_ir_step);

    try addWasiUpdateStep(b, version);

    const update_mingw_step = b.step("update-mingw", "Update zig's bundled mingw");
    const opt_mingw_src_path = b.option([]const u8, "mingw-src", "path to mingw-w64 source directory");
    if (opt_mingw_src_path) |mingw_src_path| {
        const update_mingw_exe = b.addExecutable(.{
            .name = "update_mingw",
            .root_module = b.createModule(.{
                .target = b.graph.host,
                .root_source_file = b.path("tools/update_mingw.zig"),
            }),
        });
        const update_mingw_run = b.addRunArtifact(update_mingw_exe);
        update_mingw_run.addDirectoryArg(b.path("lib"));
        update_mingw_run.addDirectoryArg(.{ .cwd_relative = mingw_src_path });

        update_mingw_step.dependOn(&update_mingw_run.step);
    } else {
        update_mingw_step.dependOn(&b.addFail("The -Dmingw-src=... option is required for this step").step);
    }

    const test_incremental_step = b.step("test-incremental", "Run the incremental compilation test cases");
    try tests.addIncrementalTests(b, test_incremental_step);
    test_step.dependOn(test_incremental_step);
}

fn addWasiUpdateStep(b: *std.Build, version: [:0]const u8) !void {
    const semver = try std.SemanticVersion.parse(version);

    const exe = addCompilerStep(b, .{
        .optimize = .ReleaseSmall,
        .target = b.resolveTargetQuery(std.Target.Query.parse(.{
            .arch_os_abi = "wasm32-wasi",
            // * `extended_const` is not supported by the `wasm-opt` version in CI.
            // * `nontrapping_bulk_memory_len0` is supported by `wasm2c`.
            .cpu_features = "baseline-extended_const+nontrapping_bulk_memory_len0",
        }) catch unreachable),
    });

    const exe_options = b.addOptions();
    exe.root_module.addOptions("build_options", exe_options);

    exe_options.addOption(u32, "mem_leak_frames", 0);
    exe_options.addOption(bool, "have_llvm", false);
    exe_options.addOption(bool, "debug_gpa", false);
    exe_options.addOption([:0]const u8, "version", version);
    exe_options.addOption(std.SemanticVersion, "semver", semver);
    exe_options.addOption(bool, "enable_debug_extensions", false);
    exe_options.addOption(bool, "enable_logging", false);
    exe_options.addOption(bool, "enable_link_snapshots", false);
    exe_options.addOption(bool, "enable_tracy", false);
    exe_options.addOption(bool, "enable_tracy_callstack", false);
    exe_options.addOption(bool, "enable_tracy_allocation", false);
    exe_options.addOption(u32, "tracy_callstack_depth", 0);
    exe_options.addOption(bool, "value_tracing", false);
    exe_options.addOption(DevEnv, "dev", .bootstrap);

    // zig1 chooses to interpret values by name. The tradeoff is as follows:
    //
    // * We lose a small amount of performance. This is essentially irrelevant for zig1.
    //
    // * We lose the ability to perform trivial renames on certain `std.builtin` types without
    //   zig1.wasm updates. For instance, we cannot rename an enum from PascalCase fields to
    //   snake_case fields without an update.
    //
    // * We gain the ability to add and remove fields to and from `std.builtin` types without
    //   zig1.wasm updates. For instance, we can add a new tag to `CallingConvention` without
    //   an update.
    //
    // Because field renames only happen when we apply a breaking change to the language (which
    // is becoming progressively rarer), but tags may be added to or removed from target-dependent
    // types over time in response to new targets coming into use, we gain more than we lose here.
    exe_options.addOption(ValueInterpretMode, "value_interpret_mode", .by_name);

    const run_opt = b.addSystemCommand(&.{
        "wasm-opt",
        "-Oz",
        "--enable-bulk-memory",
        "--enable-mutable-globals",
        "--enable-nontrapping-float-to-int",
        "--enable-sign-ext",
    });
    run_opt.addArtifactArg(exe);
    run_opt.addArg("-o");
    run_opt.addFileArg(b.path("stage1/zig1.wasm"));

    const copy_zig_h = b.addUpdateSourceFiles();
    copy_zig_h.addCopyFileToSource(b.path("lib/zig.h"), "stage1/zig.h");

    const update_zig1_step = b.step("update-zig1", "Update stage1/zig1.wasm");
    update_zig1_step.dependOn(&run_opt.step);
    update_zig1_step.dependOn(&copy_zig_h.step);
}

const AddCompilerModOptions = struct {
    optimize: std.builtin.OptimizeMode,
    target: std.Build.ResolvedTarget,
    strip: ?bool = null,
    valgrind: ?bool = null,
    sanitize_thread: ?bool = null,
    single_threaded: ?bool = null,
};

fn addCompilerMod(b: *std.Build, options: AddCompilerModOptions) *std.Build.Module {
    const compiler_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = options.target,
        .optimize = options.optimize,
        .strip = options.strip,
        .sanitize_thread = options.sanitize_thread,
        .single_threaded = options.single_threaded,
        .valgrind = options.valgrind,
    });

    const aro_mod = b.createModule(.{
        .root_source_file = b.path("lib/compiler/aro/aro.zig"),
    });

    const aro_translate_c_mod = b.createModule(.{
        .root_source_file = b.path("lib/compiler/aro_translate_c.zig"),
    });

    aro_translate_c_mod.addImport("aro", aro_mod);
    compiler_mod.addImport("aro", aro_mod);
    compiler_mod.addImport("aro_translate_c", aro_translate_c_mod);

    return compiler_mod;
}

fn addCompilerStep(b: *std.Build, options: AddCompilerModOptions) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "zig",
        .max_rss = 7_800_000_000,
        .root_module = addCompilerMod(b, options),
    });
    exe.stack_size = stack_size;

    return exe;
}

// LLVM/Clang C++ dependencies have been removed
// All CMake configuration, C++ source files, and library linkage has been removed
// Pure Zig implementation is now used for all compiler functionality

fn generateLangRef(b: *std.Build) std.Build.LazyPath {
    const doctest_exe = b.addExecutable(.{
        .name = "doctest",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/doctest.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

    var dir = b.build_root.handle.openDir("doc/langref", .{ .iterate = true }) catch |err| {
        std.debug.panic("unable to open '{f}doc/langref' directory: {s}", .{
            b.build_root, @errorName(err),
        });
    };
    defer dir.close();

    var wf = b.addWriteFiles();

    var it = dir.iterateAssumeFirstIteration();
    while (it.next() catch @panic("failed to read dir")) |entry| {
        if (std.mem.startsWith(u8, entry.name, ".") or entry.kind != .file)
            continue;

        const out_basename = b.fmt("{s}.out", .{std.fs.path.stem(entry.name)});
        const cmd = b.addRunArtifact(doctest_exe);
        cmd.addArgs(&.{
            "--zig",        b.graph.zig_exe,
            // TODO: enhance doctest to use "--listen=-" rather than operating
            // in a temporary directory
            "--cache-root", b.cache_root.path orelse ".",
        });
        cmd.addArgs(&.{ "--zig-lib-dir", b.fmt("{f}", .{b.graph.zig_lib_directory}) });
        cmd.addArgs(&.{"-i"});
        cmd.addFileArg(b.path(b.fmt("doc/langref/{s}", .{entry.name})));

        cmd.addArgs(&.{"-o"});
        _ = wf.addCopyFile(cmd.addOutputFileArg(out_basename), out_basename);
    }

    const docgen_exe = b.addExecutable(.{
        .name = "docgen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/docgen.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

    const docgen_cmd = b.addRunArtifact(docgen_exe);
    docgen_cmd.addArgs(&.{"--code-dir"});
    docgen_cmd.addDirectoryArg(wf.getDirectory());

    docgen_cmd.addFileArg(b.path("doc/langref.html.in"));
    return docgen_cmd.addOutputFileArg("langref.html");
}

fn superHtmlCheck(b: *std.Build, html_file: std.Build.LazyPath) *std.Build.Step {
    const run_superhtml = b.addSystemCommand(&.{
        "superhtml", "check",
    });
    run_superhtml.addFileArg(html_file);
    run_superhtml.expectExitCode(0);
    return &run_superhtml.step;
}
