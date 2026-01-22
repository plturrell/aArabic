// GPU Diagnostics Test Suite
// Comprehensive GPU detection and capability testing
//
// This test verifies:
// 1. CUDA runtime availability
// 2. T4 GPU detection and properties
// 3. Tensor Core availability
// 4. Memory bandwidth measurement
// 5. cuBLAS initialization
// 6. Stream creation and synchronization

const std = @import("std");
const testing = std.testing;

// Import CUDA modules
const cuda_bindings = @import("cuda_bindings");
const cuda_context = @import("cuda_context");
const cuda_memory = @import("cuda_memory");
const cuda_streams = @import("cuda_streams");
const cublas = @import("cublas_bindings");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n" ++ "=" ** 60 ++ "\n", .{});
    std.debug.print("  GPU INTEGRATION DIAGNOSTICS\n", .{});
    std.debug.print("=" ** 60 ++ "\n\n", .{});

    var passed: u32 = 0;
    var failed: u32 = 0;
    var skipped: u32 = 0;

    // Test 1: CUDA Runtime Detection
    std.debug.print("[1/8] CUDA Runtime Detection\n", .{});
    if (testCudaRuntime()) {
        std.debug.print("   ✓ CUDA runtime available\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ✗ CUDA runtime not available: {}\n", .{err});
        failed += 1;
        std.debug.print("\n⚠️  GPU tests will be skipped (no CUDA runtime)\n\n", .{});
        printSummary(passed, failed, skipped + 7);
        return;
    }

    // Test 2: GPU Device Detection
    std.debug.print("\n[2/8] GPU Device Detection\n", .{});
    const device_count = testDeviceDetection() catch |err| {
        std.debug.print("   ✗ Failed to detect devices: {}\n", .{err});
        failed += 1;
        printSummary(passed, failed, skipped + 6);
        return;
    };
    std.debug.print("   ✓ Found {d} GPU device(s)\n", .{device_count});
    passed += 1;

    if (device_count == 0) {
        std.debug.print("\n⚠️  No GPU devices found\n\n", .{});
        printSummary(passed, failed, skipped + 6);
        return;
    }

    // Test 3: T4 GPU Detection
    std.debug.print("\n[3/8] T4 GPU Detection\n", .{});
    if (testT4Detection(allocator)) {
        std.debug.print("   ✓ T4 GPU detected\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ⚠ T4 not found (detected other GPU): {}\n", .{err});
        skipped += 1;
    }

    // Test 4: Tensor Core Availability
    std.debug.print("\n[4/8] Tensor Core Availability\n", .{});
    if (testTensorCores(allocator)) {
        std.debug.print("   ✓ Tensor Cores available\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ⚠ Tensor Cores not available: {}\n", .{err});
        skipped += 1;
    }

    // Test 5: Memory Bandwidth
    std.debug.print("\n[5/8] Memory Bandwidth Measurement\n", .{});
    if (testMemoryBandwidth(allocator)) |bandwidth| {
        std.debug.print("   ✓ Memory bandwidth: {d:.1} GB/s\n", .{bandwidth});
        passed += 1;
    } else |err| {
        std.debug.print("   ✗ Bandwidth test failed: {}\n", .{err});
        failed += 1;
    }

    // Test 6: cuBLAS Initialization
    std.debug.print("\n[6/8] cuBLAS Initialization\n", .{});
    if (testCublasInit()) {
        std.debug.print("   ✓ cuBLAS initialized successfully\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("   ✗ cuBLAS initialization failed: {}\n", .{err});
        failed += 1;
    }

    // Test 7: Stream Creation
    std.debug.print("\n[7/8] CUDA Stream Creation\n", .{});
    if (testStreamCreation(allocator)) |num_streams| {
        std.debug.print("   ✓ Created {d} CUDA streams\n", .{num_streams});
        passed += 1;
    } else |err| {
        std.debug.print("   ✗ Stream creation failed: {}\n", .{err});
        failed += 1;
    }

    // Test 8: GPU Memory Allocation
    std.debug.print("\n[8/8] GPU Memory Allocation\n", .{});
    if (testMemoryAllocation(allocator)) |size_mb| {
        std.debug.print("   ✓ Allocated {d} MB on GPU\n", .{size_mb});
        passed += 1;
    } else |err| {
        std.debug.print("   ✗ Memory allocation failed: {}\n", .{err});
        failed += 1;
    }

    // Print summary
    std.debug.print("\n", .{});
    printSummary(passed, failed, skipped);

    // Exit with error code if any tests failed
    if (failed > 0) {
        std.process.exit(1);
    }
}

fn testCudaRuntime() !void {
    var version: i32 = 0;
    try cuda_bindings.checkCudaError(
        cuda_bindings.cudaRuntimeGetVersion(&version),
        "get CUDA runtime version"
    );
    std.debug.print("      Runtime version: {d}.{d}\n", .{
        @divTrunc(version, 1000),
        @mod(@divTrunc(version, 10), 100),
    });
}

fn testDeviceDetection() !i32 {
    var device_count: i32 = 0;
    try cuda_bindings.checkCudaError(
        cuda_bindings.cudaGetDeviceCount(&device_count),
        "get device count"
    );
    return device_count;
}

fn testT4Detection(allocator: std.mem.Allocator) !void {
    const ctx = try cuda_context.CudaContext.init(allocator, 0);
    defer ctx.deinit();

    const props = ctx.properties;
    std.debug.print("      GPU: {s}\n", .{props.name});
    std.debug.print("      Compute: {d}.{d}\n", .{
        props.compute_capability.major,
        props.compute_capability.minor,
    });

    // T4 has compute capability 7.5
    if (props.compute_capability.major == 7 and props.compute_capability.minor == 5) {
        return;
    }

    return error.NotT4GPU;
}

fn testTensorCores(allocator: std.mem.Allocator) !void {
    const ctx = try cuda_context.CudaContext.init(allocator, 0);
    defer ctx.deinit();

    const props = ctx.properties;
    
    // Tensor Cores available on compute capability 7.0+
    if (props.compute_capability.major >= 7) {
        std.debug.print("      Compute capability: {d}.{d}\n", .{
            props.compute_capability.major,
            props.compute_capability.minor,
        });
        return;
    }

    return error.NoTensorCores;
}

fn testMemoryBandwidth(allocator: std.mem.Allocator) !f64 {
    const ctx = try cuda_context.CudaContext.init(allocator, 0);
    defer ctx.deinit();

    // Allocate test buffers (100 MB)
    const size: usize = 100 * 1024 * 1024;
    const host_data = try allocator.alloc(u8, size);
    defer allocator.free(host_data);

    // Fill with test pattern
    for (host_data, 0..) |*byte, i| {
        byte.* = @truncate(i);
    }

    var device_ptr: ?*anyopaque = null;
    try cuda_bindings.checkCudaError(
        cuda_bindings.cudaMalloc(&device_ptr, size),
        "allocate device memory"
    );
    defer _ = cuda_bindings.cudaFree(device_ptr);

    // Measure H2D transfer
    const start = std.time.nanoTimestamp();
    try cuda_bindings.checkCudaError(
        cuda_bindings.cudaMemcpy(
            device_ptr,
            host_data.ptr,
            size,
            cuda_bindings.cudaMemcpyKind.cudaMemcpyHostToDevice
        ),
        "copy host to device"
    );
    try cuda_bindings.checkCudaError(
        cuda_bindings.cudaDeviceSynchronize(),
        "sync after transfer"
    );
    const elapsed_ns = std.time.nanoTimestamp() - start;

    // Calculate bandwidth in GB/s
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const size_gb = @as(f64, @floatFromInt(size)) / (1024.0 * 1024.0 * 1024.0);
    const bandwidth = size_gb / elapsed_s;

    std.debug.print("      Transfer time: {d:.2}ms\n", .{elapsed_s * 1000.0});

    return bandwidth;
}

fn testCublasInit() !void {
    var handle: ?*anyopaque = null;
    try cublas.checkCublasError(
        cublas.cublasCreate(&handle),
        "create cuBLAS handle"
    );
    defer _ = cublas.cublasDestroy(handle);

    std.debug.print("      Handle: 0x{x}\n", .{@intFromPtr(handle)});
}

fn testStreamCreation(allocator: std.mem.Allocator) !u32 {
    const ctx = try cuda_context.CudaContext.init(allocator, 0);
    defer ctx.deinit();

    // Create pool of streams
    var streams: [4]?*anyopaque = undefined;
    for (&streams) |*stream| {
        try cuda_bindings.checkCudaError(
            cuda_bindings.cudaStreamCreate(stream),
            "create stream"
        );
    }

    // Clean up
    for (streams) |stream| {
        _ = cuda_bindings.cudaStreamDestroy(stream);
    }

    std.debug.print("      Pool size: {d}\n", .{streams.len});
    return @intCast(streams.len);
}

fn testMemoryAllocation(allocator: std.mem.Allocator) !u32 {
    const ctx = try cuda_context.CudaContext.init(allocator, 0);
    defer ctx.deinit();

    // Allocate 100 MB
    const size: usize = 100 * 1024 * 1024;
    var device_ptr: ?*anyopaque = null;
    
    try cuda_bindings.checkCudaError(
        cuda_bindings.cudaMalloc(&device_ptr, size),
        "allocate device memory"
    );
    defer _ = cuda_bindings.cudaFree(device_ptr);

    std.debug.print("      Address: 0x{x}\n", .{@intFromPtr(device_ptr)});
    return 100;
}

fn printSummary(passed: u32, failed: u32, skipped: u32) void {
    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("  SUMMARY\n", .{});
    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("  Passed:  {d}\n", .{passed});
    std.debug.print("  Failed:  {d}\n", .{failed});
    std.debug.print("  Skipped: {d}\n", .{skipped});
    std.debug.print("  Total:   {d}\n", .{passed + failed + skipped});
    std.debug.print("=" ** 60 ++ "\n\n", .{});

    if (failed == 0 and passed > 0) {
        std.debug.print("✓ GPU ACCELERATION VERIFIED\n\n", .{});
    } else if (failed > 0) {
        std.debug.print("✗ GPU INTEGRATION ISSUES DETECTED\n\n", .{});
    } else {
        std.debug.print("⚠ NO GPU AVAILABLE\n\n", .{});
    }
}
