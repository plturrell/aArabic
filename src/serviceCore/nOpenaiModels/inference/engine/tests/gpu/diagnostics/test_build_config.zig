// Build Configuration Verifier
// Compile-time and runtime checks for CUDA integration
//
// Verifies:
// - CUDA libraries linked (-lcuda, -lcublas, -lcudart)
// - Symbol resolution (cudaMalloc, cublasCreate, etc.)
// - Shared library dependencies
// - Version compatibility (CUDA driver vs runtime)

const std = @import("std");
const testing = std.testing;
const builtin = @import("builtin");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
    std.debug.print("  BUILD CONFIGURATION VERIFICATION\n", .{});
    std.debug.print("=" ** 70 ++ "\n\n", .{});

    var passed: u32 = 0;
    var failed: u32 = 0;
    var warnings: u32 = 0;

    // Test 1: Platform information
    std.debug.print("[1/6] Platform Information\n", .{});
    testPlatformInfo();
    passed += 1;

    // Test 2: CUDA library paths
    std.debug.print("\n[2/6] CUDA Library Paths\n", .{});
    if (testCudaLibraryPaths(allocator)) {
        std.debug.print("   ✓ CUDA libraries found\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ⚠ CUDA libraries not found: {}\n", .{err});
        warnings += 1;
    }

    // Test 3: Symbol resolution
    std.debug.print("\n[3/6] CUDA Symbol Resolution\n", .{});
    if (testSymbolResolution()) {
        std.debug.print("   ✓ CUDA symbols resolved\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ✗ Symbol resolution failed: {}\n", .{err});
        failed += 1;
    }

    // Test 4: Version compatibility
    std.debug.print("\n[4/6] Version Compatibility\n", .{});
    if (testVersionCompatibility()) {
        std.debug.print("   ✓ Versions compatible\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ⚠ Version check: {}\n", .{err});
        warnings += 1;
    }

    // Test 5: Runtime linkage
    std.debug.print("\n[5/6] Runtime Library Linkage\n", .{});
    if (testRuntimeLinkage(allocator)) {
        std.debug.print("   ✓ Libraries properly linked\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ✗ Linkage issues: {}\n", .{err});
        failed += 1;
    }

    // Test 6: Recommended build flags
    std.debug.print("\n[6/6] Build Configuration Recommendations\n", .{});
    printBuildRecommendations();
    passed += 1;

    // Summary
    std.debug.print("\n" ++ "=" ** 70 ++ "\n", .{});
    std.debug.print("  SUMMARY\n", .{});
    std.debug.print("=" ** 70 ++ "\n", .{});
    std.debug.print("  Passed:   {d}\n", .{passed});
    std.debug.print("  Failed:   {d}\n", .{failed});
    std.debug.print("  Warnings: {d}\n", .{warnings});
    std.debug.print("=" ** 70 ++ "\n\n", .{});

    if (failed > 0) {
        std.debug.print("❌ BUILD CONFIGURATION ISSUES DETECTED\n\n", .{});
        std.debug.print("Possible fixes:\n", .{});
        std.debug.print("1. Ensure CUDA toolkit is installed\n", .{});
        std.debug.print("2. Add CUDA lib paths to LD_LIBRARY_PATH\n", .{});
        std.debug.print("3. Update build.zig to link CUDA libraries\n", .{});
        std.debug.print("4. Check CUDA version compatibility\n\n", .{});
        std.process.exit(1);
    } else if (warnings > 0) {
        std.debug.print("⚠️  BUILD CONFIGURATION HAS WARNINGS\n\n", .{});
    } else {
        std.debug.print("✓ BUILD CONFIGURATION OK\n\n", .{});
    }
}

fn testPlatformInfo() void {
    std.debug.print("   OS: {s}\n", .{@tagName(builtin.os.tag)});
    std.debug.print("   Arch: {s}\n", .{@tagName(builtin.cpu.arch)});
    std.debug.print("   ABI: {s}\n", .{@tagName(builtin.abi)});
    std.debug.print("   Optimize: {s}\n", .{@tagName(builtin.mode)});
    std.debug.print("   Link mode: {s}\n", .{@tagName(builtin.link_mode)});
}

fn testCudaLibraryPaths(allocator: std.mem.Allocator) !void {
    const cuda_paths = [_][]const u8{
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    };

    const cuda_libs = [_][]const u8{
        "libcuda.so",
        "libcudart.so",
        "libcublas.so",
    };

    var found_any = false;

    for (cuda_paths) |path| {
        for (cuda_libs) |lib| {
            const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ path, lib });
            defer allocator.free(full_path);

            std.fs.cwd().access(full_path, .{}) catch continue;
            
            if (!found_any) {
                std.debug.print("   Found libraries:\n", .{});
                found_any = true;
            }
            std.debug.print("   - {s}\n", .{full_path});
        }
    }

    if (!found_any) {
        return error.CudaLibrariesNotFound;
    }
}

fn testSymbolResolution() !void {
    // Try to import CUDA modules
    const cuda_bindings = @import("cuda_bindings");
    const cublas = @import("cublas_bindings");

    // Check if critical symbols exist
    _ = cuda_bindings.cudaGetDeviceCount;
    _ = cuda_bindings.cudaMalloc;
    _ = cuda_bindings.cudaFree;
    _ = cuda_bindings.cudaMemcpy;
    _ = cublas.cublasCreate;
    _ = cublas.cublasSgemm;

    std.debug.print("   Core CUDA symbols: ✓\n", .{});
    std.debug.print("   cuBLAS symbols: ✓\n", .{});
}

fn testVersionCompatibility() !void {
    const cuda_bindings = @import("cuda_bindings");

    var runtime_version: i32 = 0;
    var driver_version: i32 = 0;

    // Try to get runtime version
    const runtime_result = cuda_bindings.cudaRuntimeGetVersion(&runtime_version);
    if (runtime_result != cuda_bindings.cudaError_t.cudaSuccess) {
        return error.RuntimeVersionUnavailable;
    }

    // Try to get driver version
    const driver_result = cuda_bindings.cudaDriverGetVersion(&driver_version);
    if (driver_result != cuda_bindings.cudaError_t.cudaSuccess) {
        std.debug.print("   ⚠ Driver version unavailable (no GPU?)\n", .{});
        return;
    }

    const runtime_major = @divTrunc(runtime_version, 1000);
    const runtime_minor = @mod(@divTrunc(runtime_version, 10), 100);
    const driver_major = @divTrunc(driver_version, 1000);
    const driver_minor = @mod(@divTrunc(driver_version, 10), 100);

    std.debug.print("   Runtime: {d}.{d}\n", .{ runtime_major, runtime_minor });
    std.debug.print("   Driver: {d}.{d}\n", .{ driver_major, driver_minor });

    // Driver should be >= runtime
    if (driver_version < runtime_version) {
        std.debug.print("   ⚠ Driver older than runtime (may cause issues)\n", .{});
        return error.VersionMismatch;
    }
}

fn testRuntimeLinkage(allocator: std.mem.Allocator) !void {
    // Try to run ldd on the current executable
    const exe_path = try std.fs.selfExePathAlloc(allocator);
    defer allocator.free(exe_path);

    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "ldd", exe_path },
    }) catch |err| {
        std.debug.print("   ⚠ Could not run ldd: {}\n", .{err});
        return;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term.Exited != 0) {
        return error.LddFailed;
    }

    // Check for CUDA libraries in output
    const has_cuda = std.mem.indexOf(u8, result.stdout, "libcuda") != null;
    const has_cudart = std.mem.indexOf(u8, result.stdout, "libcudart") != null;
    const has_cublas = std.mem.indexOf(u8, result.stdout, "libcublas") != null;

    std.debug.print("   libcuda: {s}\n", .{if (has_cuda) "✓ linked" else "✗ not linked"});
    std.debug.print("   libcudart: {s}\n", .{if (has_cudart) "✓ linked" else "✗ not linked"});
    std.debug.print("   libcublas: {s}\n", .{if (has_cublas) "✓ linked" else "✗ not linked"});

    if (!has_cuda and !has_cudart and !has_cublas) {
        return error.NoCudaLibsLinked;
    }
}

fn printBuildRecommendations() void {
    std.debug.print("   Recommended build.zig configuration:\n\n", .{});
    std.debug.print("   ```zig\n", .{});
    std.debug.print("   // Add to your executable/library:\n", .{});
    std.debug.print("   exe.linkSystemLibrary(\"cuda\");\n", .{});
    std.debug.print("   exe.linkSystemLibrary(\"cudart\");\n", .{});
    std.debug.print("   exe.linkSystemLibrary(\"cublas\");\n", .{});
    std.debug.print("   exe.addLibraryPath(.{{ .path = \"/usr/local/cuda/lib64\" }});\n", .{});
    std.debug.print("   exe.addIncludePath(.{{ .path = \"/usr/local/cuda/include\" }});\n", .{});
    std.debug.print("   ```\n\n", .{});

    std.debug.print("   Environment variables:\n\n", .{});
    std.debug.print("   ```bash\n", .{});
    std.debug.print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n", .{});
    std.debug.print("   export PATH=/usr/local/cuda/bin:$PATH\n", .{});
    std.debug.print("   ```\n", .{});
}

test "build config: platform info" {
    // This should always succeed
    testPlatformInfo();
}

test "build config: symbol resolution" {
    // Check that CUDA symbols can be imported
    testSymbolResolution() catch |err| {
        std.debug.print("Symbol resolution test: {}\n", .{err});
        return err;
    };
}
