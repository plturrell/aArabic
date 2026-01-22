// Unit tests for cuBLAS Bindings
// Tests cuBLAS initialization, GEMM operations, and Tensor Core paths
//
// Requirements: CUDA Toolkit with cuBLAS library

const std = @import("std");
const testing = std.testing;
const cublas = @import("cublas_bindings");
const cuda = @import("cuda_bindings");

// ============================================================================
// Helper Functions
// ============================================================================

fn hasGPU() bool {
    var device_count: c_int = 0;
    const result = cuda.cudaGetDeviceCount(&device_count);
    return result == cuda.cudaSuccess and device_count > 0;
}

fn hasTensorCores() bool {
    if (!hasGPU()) return false;

    var props: cuda.cudaDeviceProp = undefined;
    const result = cuda.cudaGetDeviceProperties(&props, 0);
    if (result != cuda.cudaSuccess) return false;

    // Tensor Cores available on Volta (7.0+)
    return props.major >= 7;
}

// ============================================================================
// cuBLAS Context Tests
// ============================================================================

test "cublas: context initialization" {
    std.debug.print("\n=== Testing cuBLAS Context Initialization ===\n", .{});

    if (!hasGPU()) {
        std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
        return;
    }

    var ctx = cublas.CublasContext.init(false) catch |err| {
        std.debug.print("❌ Failed to initialize cuBLAS: {}\n", .{err});
        return err;
    };
    defer ctx.deinit();

    std.debug.print("✅ cuBLAS context initialized successfully\n", .{});
    try testing.expect(ctx.handle != undefined);
}

test "cublas: context with tensor cores" {
    std.debug.print("\n=== Testing cuBLAS with Tensor Cores ===\n", .{});

    if (!hasGPU()) {
        std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
        return;
    }

    const use_tensor_cores = hasTensorCores();
    std.debug.print("   Tensor Cores available: {}\n", .{use_tensor_cores});

    var ctx = cublas.CublasContext.init(use_tensor_cores) catch |err| {
        std.debug.print("❌ Failed to initialize cuBLAS: {}\n", .{err});
        return err;
    };
    defer ctx.deinit();

    try testing.expect(ctx.use_tensor_cores == use_tensor_cores);
    std.debug.print("✅ cuBLAS Tensor Core config: {}\n", .{ctx.use_tensor_cores});
}

// ============================================================================
// SGEMM Tests (FP32)
// ============================================================================

test "cublas: sgemm small matrix" {
    std.debug.print("\n=== Testing cuBLAS SGEMM (FP32) ===\n", .{});

    if (!hasGPU()) {
        std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
        return;
    }

    var ctx = cublas.CublasContext.init(false) catch |err| {
        std.debug.print("❌ Failed to initialize cuBLAS: {}\n", .{err});
        return err;
    };
    defer ctx.deinit();

    const m: usize = 4;
    const n: usize = 4;
    const k: usize = 4;

    // Allocate host memory
    var h_a: [m * k]f32 = undefined;
    var h_b: [k * n]f32 = undefined;
    var h_c: [m * n]f32 = undefined;

    // Initialize matrices: A = I, B = 2*I => C = 2*I
    for (0..m) |i| {
        for (0..k) |j| {
            h_a[i * k + j] = if (i == j) 1.0 else 0.0;
        }
    }
    for (0..k) |i| {
        for (0..n) |j| {
            h_b[i * n + j] = if (i == j) 2.0 else 0.0;
        }
    }
    @memset(&h_c, 0);

    // Allocate device memory
    var d_a: *anyopaque = undefined;
    var d_b: *anyopaque = undefined;
    var d_c: *anyopaque = undefined;

    try cuda.checkCudaError(cuda.cudaMalloc(&d_a, m * k * @sizeOf(f32)), "cudaMalloc A");
    defer _ = cuda.cudaFree(d_a);

    try cuda.checkCudaError(cuda.cudaMalloc(&d_b, k * n * @sizeOf(f32)), "cudaMalloc B");
    defer _ = cuda.cudaFree(d_b);

    try cuda.checkCudaError(cuda.cudaMalloc(&d_c, m * n * @sizeOf(f32)), "cudaMalloc C");
    defer _ = cuda.cudaFree(d_c);

    // Copy to device
    try cuda.checkCudaError(cuda.cudaMemcpy(d_a, &h_a, m * k * @sizeOf(f32), cuda.cudaMemcpyHostToDevice), "cudaMemcpy A H2D");
    try cuda.checkCudaError(cuda.cudaMemcpy(d_b, &h_b, k * n * @sizeOf(f32), cuda.cudaMemcpyHostToDevice), "cudaMemcpy B H2D");
    try cuda.checkCudaError(cuda.cudaMemcpy(d_c, &h_c, m * n * @sizeOf(f32), cuda.cudaMemcpyHostToDevice), "cudaMemcpy C H2D");

    // Execute SGEMM: C = A @ B
    try ctx.sgemm(@ptrCast(d_c), @ptrCast(d_a), @ptrCast(d_b), m, n, k);

    // Synchronize
    try cuda.checkCudaError(cuda.cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy result back
    try cuda.checkCudaError(cuda.cudaMemcpy(&h_c, d_c, m * n * @sizeOf(f32), cuda.cudaMemcpyDeviceToHost), "cudaMemcpy C D2H");

    // Verify: C should be 2*I
    var correct = true;
    for (0..m) |i| {
        for (0..n) |j| {
            const expected: f32 = if (i == j) 2.0 else 0.0;
            const actual = h_c[i * n + j];
            if (@abs(actual - expected) > 0.001) {
                std.debug.print("❌ Mismatch at ({d},{d}): expected {d}, got {d}\n", .{ i, j, expected, actual });
                correct = false;
            }
        }
    }

    if (correct) {
        std.debug.print("✅ SGEMM result verified: I @ 2I = 2I\n", .{});
    }
    try testing.expect(correct);
}

// ============================================================================
// GemmEx Tests (Mixed Precision / Tensor Cores)
// ============================================================================

test "cublas: gemmEx FP16 tensor cores" {
    std.debug.print("\n=== Testing cuBLAS GemmEx (FP16 Tensor Cores) ===\n", .{});

    if (!hasGPU()) {
        std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
        return;
    }

    if (!hasTensorCores()) {
        std.debug.print("⚠️  Test skipped: No Tensor Core support (requires SM 7.0+)\n", .{});
        return;
    }

    var ctx = cublas.CublasContext.init(true) catch |err| {
        std.debug.print("❌ Failed to initialize cuBLAS: {}\n", .{err});
        return err;
    };
    defer ctx.deinit();

    // Use multiples of 8 for Tensor Core efficiency
    const m: usize = 16;
    const n: usize = 16;
    const k: usize = 16;

    // Allocate device memory for FP16 matrices
    var d_a: *anyopaque = undefined;
    var d_b: *anyopaque = undefined;
    var d_c: *anyopaque = undefined;

    try cuda.checkCudaError(cuda.cudaMalloc(&d_a, m * k * 2), "cudaMalloc A (FP16)"); // 2 bytes per f16
    defer _ = cuda.cudaFree(d_a);

    try cuda.checkCudaError(cuda.cudaMalloc(&d_b, k * n * 2), "cudaMalloc B (FP16)");
    defer _ = cuda.cudaFree(d_b);

    try cuda.checkCudaError(cuda.cudaMalloc(&d_c, m * n * @sizeOf(f32)), "cudaMalloc C (FP32)");
    defer _ = cuda.cudaFree(d_c);

    // Initialize with zeros (in real use, would copy FP16 data)
    try cuda.checkCudaError(cuda.cudaMemset(d_a, 0, m * k * 2), "cudaMemset A");
    try cuda.checkCudaError(cuda.cudaMemset(d_b, 0, k * n * 2), "cudaMemset B");
    try cuda.checkCudaError(cuda.cudaMemset(d_c, 0, m * n * @sizeOf(f32)), "cudaMemset C");

    // Execute GemmEx with FP16 inputs and FP32 output
    try ctx.gemmEx_fp16(d_c, d_a, d_b, m, n, k, cublas.CUDA_R_32F);

    // Synchronize
    try cuda.checkCudaError(cuda.cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std.debug.print("✅ GemmEx FP16→FP32 executed successfully (Tensor Cores)\n", .{});
    std.debug.print("   Matrix size: {d}x{d}x{d}\n", .{ m, n, k });
}

// ============================================================================
// Error Handling Tests
// ============================================================================

test "cublas: error string conversion" {
    std.debug.print("\n=== Testing cuBLAS Error Strings ===\n", .{});

    try testing.expectEqualStrings("CUBLAS_STATUS_SUCCESS", cublas.getCublasErrorString(cublas.CUBLAS_STATUS_SUCCESS));
    try testing.expectEqualStrings("CUBLAS_STATUS_NOT_INITIALIZED", cublas.getCublasErrorString(cublas.CUBLAS_STATUS_NOT_INITIALIZED));
    try testing.expectEqualStrings("CUBLAS_STATUS_INVALID_VALUE", cublas.getCublasErrorString(cublas.CUBLAS_STATUS_INVALID_VALUE));
    try testing.expectEqualStrings("CUBLAS_STATUS_ARCH_MISMATCH", cublas.getCublasErrorString(cublas.CUBLAS_STATUS_ARCH_MISMATCH));

    std.debug.print("✅ Error string conversion verified\n", .{});
}

// ============================================================================
// Performance Benchmark (Optional)
// ============================================================================

test "cublas: benchmark sgemm throughput" {
    std.debug.print("\n=== Benchmarking cuBLAS SGEMM Throughput ===\n", .{});

    if (!hasGPU()) {
        std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
        return;
    }

    var ctx = cublas.CublasContext.init(hasTensorCores()) catch |err| {
        std.debug.print("❌ Failed to initialize cuBLAS: {}\n", .{err});
        return err;
    };
    defer ctx.deinit();

    // Use a larger matrix for meaningful benchmark
    const m: usize = 1024;
    const n: usize = 1024;
    const k: usize = 1024;

    // Allocate device memory
    var d_a: *anyopaque = undefined;
    var d_b: *anyopaque = undefined;
    var d_c: *anyopaque = undefined;

    try cuda.checkCudaError(cuda.cudaMalloc(&d_a, m * k * @sizeOf(f32)), "cudaMalloc A");
    defer _ = cuda.cudaFree(d_a);

    try cuda.checkCudaError(cuda.cudaMalloc(&d_b, k * n * @sizeOf(f32)), "cudaMalloc B");
    defer _ = cuda.cudaFree(d_b);

    try cuda.checkCudaError(cuda.cudaMalloc(&d_c, m * n * @sizeOf(f32)), "cudaMalloc C");
    defer _ = cuda.cudaFree(d_c);

    // Warm up
    try ctx.sgemm(@ptrCast(d_c), @ptrCast(d_a), @ptrCast(d_b), m, n, k);
    try cuda.checkCudaError(cuda.cudaDeviceSynchronize(), "warmup sync");

    // Benchmark
    const iterations: usize = 10;
    const start = std.time.nanoTimestamp();

    for (0..iterations) |_| {
        try ctx.sgemm(@ptrCast(d_c), @ptrCast(d_a), @ptrCast(d_b), m, n, k);
    }
    try cuda.checkCudaError(cuda.cudaDeviceSynchronize(), "benchmark sync");

    const end = std.time.nanoTimestamp();
    const elapsed_ns = @as(u64, @intCast(end - start));
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const avg_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

    // Calculate TFLOPS: 2 * M * N * K FLOPs per GEMM
    const flops_per_gemm = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    const total_flops = flops_per_gemm * @as(f64, @floatFromInt(iterations));
    const tflops = total_flops / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e12;

    std.debug.print("✅ SGEMM Benchmark Results:\n", .{});
    std.debug.print("   Matrix size: {d}x{d}x{d}\n", .{ m, n, k });
    std.debug.print("   Iterations: {d}\n", .{iterations});
    std.debug.print("   Average time: {d:.3} ms\n", .{avg_ms});
    std.debug.print("   Throughput: {d:.2} TFLOPS\n", .{tflops});
}
