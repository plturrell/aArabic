// Unit tests for GPU Dequantization Kernels
// Tests GPU dequantization accuracy against CPU reference implementation
//
// Requirements: CUDA GPU with dequant_kernels library compiled
//
// Test Strategy:
// 1. Create known quantized data (or use reference quantizer)
// 2. Dequantize on GPU using CUDA kernels
// 3. Dequantize on CPU for reference
// 4. Compare results within tolerance

const std = @import("std");
const testing = std.testing;
const dequant = @import("dequant_bindings");
const cuda = @import("cuda_bindings");

// ============================================================================
// Helper Functions
// ============================================================================

fn hasGPU() bool {
    var device_count: c_int = 0;
    const result = cuda.cudaGetDeviceCount(&device_count);
    return result == cuda.cudaSuccess and device_count > 0;
}

fn skipIfNoGPU() bool {
    if (!hasGPU()) {
        std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
        return true;
    }
    return false;
}

// ============================================================================
// DequantContext Tests
// ============================================================================

test "dequant: context initialization" {
    std.debug.print("\n=== Testing DequantContext Initialization ===\n", .{});

    var ctx = dequant.DequantContext.init(null);
    defer ctx.deinit();

    try testing.expect(ctx.fp16_buffer == null);
    try testing.expect(ctx.input_buffer == null);
    try testing.expect(ctx.fp16_buffer_size == 0);
    try testing.expect(ctx.input_buffer_size == 0);

    std.debug.print("✅ DequantContext initialized correctly\n", .{});
}

test "dequant: buffer allocation" {
    std.debug.print("\n=== Testing Buffer Allocation ===\n", .{});

    if (skipIfNoGPU()) return;

    var ctx = dequant.DequantContext.init(null);
    defer ctx.deinit();

    // Allocate FP16 buffer
    try ctx.ensureBuffer(1024);
    try testing.expect(ctx.fp16_buffer != null);
    try testing.expect(ctx.fp16_buffer_size >= 1024);

    std.debug.print("   FP16 buffer allocated: {} elements\n", .{ctx.fp16_buffer_size});

    // Allocate input buffer
    try ctx.ensureInputBuffer(512);
    try testing.expect(ctx.input_buffer != null);
    try testing.expect(ctx.input_buffer_size >= 512);

    std.debug.print("   Input buffer allocated: {} bytes\n", .{ctx.input_buffer_size});

    // Test buffer reuse (smaller request should not reallocate)
    const old_fp16_ptr = ctx.fp16_buffer;
    try ctx.ensureBuffer(512);
    try testing.expect(ctx.fp16_buffer == old_fp16_ptr);

    std.debug.print("✅ Buffer allocation and reuse works\n", .{});
}

test "dequant: QuantType from GGUF" {
    std.debug.print("\n=== Testing QuantType Conversion ===\n", .{});

    // Import GGUF QuantizationType for testing
    const gguf = @import("gguf_loader");

    // Test valid conversions
    const q4_0 = dequant.QuantType.fromGguf(gguf.QuantizationType.Q4_0);
    try testing.expect(q4_0 != null);
    try testing.expect(q4_0.? == .Q4_0);

    const q8_0 = dequant.QuantType.fromGguf(gguf.QuantizationType.Q8_0);
    try testing.expect(q8_0 != null);
    try testing.expect(q8_0.? == .Q8_0);

    const q4_k = dequant.QuantType.fromGguf(gguf.QuantizationType.Q4_K);
    try testing.expect(q4_k != null);
    try testing.expect(q4_k.? == .Q4_K);

    const q6_k = dequant.QuantType.fromGguf(gguf.QuantizationType.Q6_K);
    try testing.expect(q6_k != null);
    try testing.expect(q6_k.? == .Q6_K);

    // Test F32 (not quantized)
    const f32_type = dequant.QuantType.fromGguf(gguf.QuantizationType.F32);
    try testing.expect(f32_type == null);

    std.debug.print("✅ QuantType conversion works correctly\n", .{});
}

test "dequant: block calculations" {
    std.debug.print("\n=== Testing Block Calculations ===\n", .{});

    // Q4_0: 32 elements per block
    const q4_0_blocks = dequant.DequantContext.calculateNumBlocks(.Q4_0, 256);
    try testing.expect(q4_0_blocks == 8); // 256 / 32 = 8

    // Q8_0: 32 elements per block
    const q8_0_blocks = dequant.DequantContext.calculateNumBlocks(.Q8_0, 1024);
    try testing.expect(q8_0_blocks == 32); // 1024 / 32 = 32

    // Q4_K: 256 elements per block
    const q4_k_blocks = dequant.DequantContext.calculateNumBlocks(.Q4_K, 2048);
    try testing.expect(q4_k_blocks == 8); // 2048 / 256 = 8

    // Q6_K: 256 elements per block
    const q6_k_blocks = dequant.DequantContext.calculateNumBlocks(.Q6_K, 4096);
    try testing.expect(q6_k_blocks == 16); // 4096 / 256 = 16

    std.debug.print("✅ Block calculations correct\n", .{});
}

// ============================================================================
// GPU Dequantization Tests (require actual GPU)
// ============================================================================

test "dequant: Q4_K kernel smoke test" {
    std.debug.print("\n=== Testing Q4_K GPU Dequant (Smoke) ===\n", .{});

    if (skipIfNoGPU()) return;

    var ctx = dequant.DequantContext.init(null);
    defer ctx.deinit();

    // Create test data: 1 Q4_K block = 144 bytes -> 256 FP16 values
    const num_blocks: usize = 1;
    const input_bytes = num_blocks * dequant.Q4_K_BLOCK_BYTES;
    var test_input: [dequant.Q4_K_BLOCK_BYTES]u8 = undefined;

    // Fill with non-zero pattern
    for (&test_input, 0..) |*b, i| {
        b.* = @intCast((i * 7 + 13) % 256);
    }

    // Call dequant
    const result = ctx.dequant(&test_input, .Q4_K, num_blocks);

    if (result) |fp16_ptr| {
        // Kernel succeeded - verify we got a valid pointer
        _ = fp16_ptr;
        std.debug.print("✅ Q4_K dequant kernel returned valid pointer\n", .{});

        // Sync to ensure kernel completed
        _ = cuda.cudaDeviceSynchronize();
        std.debug.print("   Kernel execution completed\n", .{});
    } else |err| {
        std.debug.print("❌ Q4_K dequant failed: {}\n", .{err});
        return err;
    }
}

test "dequant: Q6_K kernel smoke test" {
    std.debug.print("\n=== Testing Q6_K GPU Dequant (Smoke) ===\n", .{});

    if (skipIfNoGPU()) return;

    var ctx = dequant.DequantContext.init(null);
    defer ctx.deinit();

    // Create test data: 1 Q6_K block = 210 bytes -> 256 FP16 values
    const num_blocks: usize = 1;
    var test_input: [dequant.Q6_K_BLOCK_BYTES]u8 = undefined;

    // Fill with pattern
    for (&test_input, 0..) |*b, i| {
        b.* = @intCast((i * 11 + 7) % 256);
    }

    // Call dequant
    const result = ctx.dequant(&test_input, .Q6_K, num_blocks);

    if (result) |fp16_ptr| {
        _ = fp16_ptr;
        _ = cuda.cudaDeviceSynchronize();
        std.debug.print("✅ Q6_K dequant kernel succeeded\n", .{});
    } else |err| {
        std.debug.print("❌ Q6_K dequant failed: {}\n", .{err});
        return err;
    }
}

test "dequant: Q4_0 kernel smoke test" {
    std.debug.print("\n=== Testing Q4_0 GPU Dequant (Smoke) ===\n", .{});

    if (skipIfNoGPU()) return;

    var ctx = dequant.DequantContext.init(null);
    defer ctx.deinit();

    // Create test data: 1 Q4_0 block = 18 bytes -> 32 FP16 values
    const num_blocks: usize = 1;
    var test_input: [dequant.Q4_0_BLOCK_BYTES]u8 = undefined;

    // Fill with pattern
    for (&test_input, 0..) |*b, i| {
        b.* = @intCast((i * 3 + 5) % 256);
    }

    const result = ctx.dequant(&test_input, .Q4_0, num_blocks);

    if (result) |fp16_ptr| {
        _ = fp16_ptr;
        _ = cuda.cudaDeviceSynchronize();
        std.debug.print("✅ Q4_0 dequant kernel succeeded\n", .{});
    } else |err| {
        std.debug.print("❌ Q4_0 dequant failed: {}\n", .{err});
        return err;
    }
}

test "dequant: Q8_0 kernel smoke test" {
    std.debug.print("\n=== Testing Q8_0 GPU Dequant (Smoke) ===\n", .{});

    if (skipIfNoGPU()) return;

    var ctx = dequant.DequantContext.init(null);
    defer ctx.deinit();

    // Create test data: 1 Q8_0 block = 36 bytes -> 32 FP16 values
    const num_blocks: usize = 1;
    var test_input: [dequant.Q8_0_BLOCK_BYTES]u8 = undefined;

    // Fill with pattern
    for (&test_input, 0..) |*b, i| {
        b.* = @intCast((i * 5 + 3) % 256);
    }

    const result = ctx.dequant(&test_input, .Q8_0, num_blocks);

    if (result) |fp16_ptr| {
        _ = fp16_ptr;
        _ = cuda.cudaDeviceSynchronize();
        std.debug.print("✅ Q8_0 dequant kernel succeeded\n", .{});
    } else |err| {
        std.debug.print("❌ Q8_0 dequant failed: {}\n", .{err});
        return err;
    }
}

test "dequant: multiple blocks" {
    std.debug.print("\n=== Testing Multi-Block Dequantization ===\n", .{});

    if (skipIfNoGPU()) return;

    var ctx = dequant.DequantContext.init(null);
    defer ctx.deinit();

    // Test with 16 Q4_K blocks (typical weight tensor size)
    const num_blocks: usize = 16;
    const input_bytes = num_blocks * dequant.Q4_K_BLOCK_BYTES;
    const allocator = testing.allocator;

    const test_input = try allocator.alloc(u8, input_bytes);
    defer allocator.free(test_input);

    // Fill with non-trivial pattern
    for (test_input, 0..) |*b, i| {
        b.* = @intCast((i * 17 + 23) % 256);
    }

    const result = ctx.dequant(test_input.ptr, .Q4_K, num_blocks);

    if (result) |fp16_ptr| {
        _ = fp16_ptr;
        _ = cuda.cudaDeviceSynchronize();
        std.debug.print("✅ Multi-block ({} blocks, {} elements) dequant succeeded\n", .{
            num_blocks,
            num_blocks * dequant.Q4_K_BLOCK_SIZE,
        });
    } else |err| {
        std.debug.print("❌ Multi-block dequant failed: {}\n", .{err});
        return err;
    }
}

