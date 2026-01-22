// Test Suite for KV Cache Compression
// Validates compression algorithms, accuracy, and performance

const std = @import("std");
const compression = @import("kv_compression.zig");
const testing = std.testing;

// ============================================================================
// Test Helpers
// ============================================================================

fn createTestData(allocator: std.mem.Allocator, size: usize) ![]f32 {
    var data = try allocator.alloc(f32, size);
    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();
    
    for (data) |*val| {
        val.* = random.float(f32) * 2.0 - 1.0; // Range: [-1, 1]
    }
    
    return data;
}

fn calculateMeanError(original: []const f32, reconstructed: []const f32) f32 {
    var sum_error: f32 = 0.0;
    for (original, reconstructed) |orig, recon| {
        sum_error += @abs(orig - recon);
    }
    return sum_error / @as(f32, @floatFromInt(original.len));
}

fn calculateMaxError(original: []const f32, reconstructed: []const f32) f32 {
    var max_error: f32 = 0.0;
    for (original, reconstructed) |orig, recon| {
        const error = @abs(orig - recon);
        if (error > max_error) max_error = error;
    }
    return max_error;
}

// ============================================================================
// Algorithm Tests
// ============================================================================

test "compression_algorithm: compression ratios" {
    try testing.expect(compression.CompressionAlgorithm.none.compressionRatio() == 1.0);
    try testing.expect(compression.CompressionAlgorithm.fp16.compressionRatio() == 2.0);
    try testing.expect(compression.CompressionAlgorithm.int8_symmetric.compressionRatio() == 4.0);
    try testing.expect(compression.CompressionAlgorithm.int8_asymmetric.compressionRatio() == 4.0);
}

test "compression_algorithm: bytes per element" {
    try testing.expect(compression.CompressionAlgorithm.none.bytesPerElement() == 4);
    try testing.expect(compression.CompressionAlgorithm.fp16.bytesPerElement() == 2);
    try testing.expect(compression.CompressionAlgorithm.int8_symmetric.bytesPerElement() == 1);
    try testing.expect(compression.CompressionAlgorithm.int8_asymmetric.bytesPerElement() == 1);
}

// ============================================================================
// Quantization Parameter Tests
// ============================================================================

test "quantization_params: initialization" {
    const params = compression.QuantizationParams.init();
    
    try testing.expect(params.scale == 1.0);
    try testing.expect(params.zero_point == 0);
    try testing.expect(params.min_val == 0.0);
    try testing.expect(params.max_val == 0.0);
}

test "quantization_params: symmetric calibration" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 1000);
    defer allocator.free(data);
    
    const params = compression.QuantizationParams.calibrate(
        data,
        .int8_symmetric,
        0.9999,
    );
    
    // Symmetric should have zero_point = 0
    try testing.expect(params.zero_point == 0);
    try testing.expect(params.scale > 0);
    try testing.expect(params.min_val <= 0);
    try testing.expect(params.max_val >= 0);
}

test "quantization_params: asymmetric calibration" {
    const allocator = testing.allocator;
    
    // Create data with positive bias
    var data = try allocator.alloc(f32, 1000);
    defer allocator.free(data);
    
    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();
    for (data) |*val| {
        val.* = random.float(f32) * 5.0; // Range: [0, 5]
    }
    
    const params = compression.QuantizationParams.calibrate(
        data,
        .int8_asymmetric,
        0.9999,
    );
    
    // Asymmetric can have non-zero zero_point
    try testing.expect(params.scale > 0);
    try testing.expect(params.min_val >= 0);
    try testing.expect(params.max_val > params.min_val);
}

// ============================================================================
// FP16 Compression Tests
// ============================================================================

test "fp16_compression: no compression" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 100);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{
        .algorithm = .none,
    };
    
    const shape = [_]usize{data.len};
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    try testing.expect(compressed.getCompressionRatio() == 1.0);
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Should be exact match
    for (data, decompressed) |orig, recon| {
        try testing.expect(orig == recon);
    }
}

test "fp16_compression: basic compression" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 100);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{
        .algorithm = .fp16,
    };
    
    const shape = [_]usize{data.len};
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    // Should be 2x compression
    try testing.expect(compressed.getCompressionRatio() >= 1.9);
    try testing.expect(compressed.getCompressionRatio() <= 2.1);
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Check accuracy (FP16 has ~0.001 precision)
    const mean_error = calculateMeanError(data, decompressed);
    try testing.expect(mean_error < 0.01); // <1% error
}

test "fp16_compression: small values" {
    const allocator = testing.allocator;
    
    const small_values = [_]f32{ 0.001, 0.002, 0.003, 0.004, 0.005 };
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{small_values.len};
    
    const compressed = try compression.compress(allocator, &small_values, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Small values should have reasonable precision
    const mean_error = calculateMeanError(&small_values, decompressed);
    try testing.expect(mean_error < 0.001);
}

test "fp16_compression: large values" {
    const allocator = testing.allocator;
    
    const large_values = [_]f32{ 100.0, 200.0, 300.0, 400.0, 500.0 };
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{large_values.len};
    
    const compressed = try compression.compress(allocator, &large_values, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Large values should maintain relative precision
    const max_error = calculateMaxError(&large_values, decompressed);
    try testing.expect(max_error < 5.0); // <1% of 500
}

// ============================================================================
// INT8 Compression Tests
// ============================================================================

test "int8_compression: symmetric quantization" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 1000);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{
        .algorithm = .int8_symmetric,
    };
    
    const shape = [_]usize{data.len};
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    // Should be 4x compression
    try testing.expect(compressed.getCompressionRatio() >= 3.9);
    try testing.expect(compressed.getCompressionRatio() <= 4.1);
    
    // Check quantization params
    try testing.expect(compressed.quant_params.zero_point == 0);
    try testing.expect(compressed.quant_params.scale > 0);
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Check accuracy (INT8 has ~0.01 precision)
    const mean_error = calculateMeanError(data, decompressed);
    try testing.expect(mean_error < 0.05); // <5% error acceptable for INT8
}

test "int8_compression: asymmetric quantization" {
    const allocator = testing.allocator;
    
    // Create biased data (positive range)
    var data = try allocator.alloc(f32, 1000);
    defer allocator.free(data);
    
    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();
    for (data) |*val| {
        val.* = random.float(f32) * 10.0; // Range: [0, 10]
    }
    
    const config = compression.CompressionConfig{
        .algorithm = .int8_asymmetric,
    };
    
    const shape = [_]usize{data.len};
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    // Should be 4x compression
    try testing.expect(compressed.getCompressionRatio() >= 3.9);
    
    // Asymmetric should use non-zero zero_point for biased data
    try testing.expect(compressed.quant_params.scale > 0);
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    const mean_error = calculateMeanError(data, decompressed);
    try testing.expect(mean_error < 0.1); // <1% of range
}

test "int8_compression: zero values" {
    const allocator = testing.allocator;
    
    const zero_values = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const config = compression.CompressionConfig{ .algorithm = .int8_symmetric };
    const shape = [_]usize{zero_values.len};
    
    const compressed = try compression.compress(allocator, &zero_values, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Zero should remain zero
    for (decompressed) |val| {
        try testing.expect(@abs(val) < 0.001);
    }
}

// ============================================================================
// Compressed Tensor Tests
// ============================================================================

test "compressed_tensor: initialization" {
    const allocator = testing.allocator;
    const shape = [_]usize{ 10, 20 };
    
    const tensor = try compression.CompressedTensor.init(
        allocator,
        &shape,
        .fp16,
    );
    defer tensor.deinit();
    
    try testing.expect(tensor.element_count == 200);
    try testing.expect(tensor.algorithm == .fp16);
    try testing.expect(tensor.data.len == 400); // 200 elements * 2 bytes
}

test "compressed_tensor: size calculations" {
    const allocator = testing.allocator;
    const shape = [_]usize{1000};
    
    const tensor = try compression.CompressedTensor.init(
        allocator,
        &shape,
        .int8_symmetric,
    );
    defer tensor.deinit();
    
    try testing.expect(tensor.getOriginalSize() == 4000); // 1000 * 4 bytes
    try testing.expect(tensor.getCompressedSize() == 1000); // 1000 * 1 byte
    try testing.expect(tensor.getCompressionRatio() == 4.0);
}

// ============================================================================
// Compression Manager Tests
// ============================================================================

test "compression_manager: initialization" {
    const allocator = testing.allocator;
    const config = compression.CompressionConfig{};
    
    const manager = try compression.CompressionManager.init(allocator, config);
    defer manager.deinit();
    
    try testing.expect(manager.config.algorithm == .fp16);
    try testing.expect(manager.stats.compress_count == 0);
    try testing.expect(manager.stats.decompress_count == 0);
}

test "compression_manager: compress KV cache" {
    const allocator = testing.allocator;
    const config = compression.CompressionConfig{
        .algorithm = .fp16,
    };
    
    const manager = try compression.CompressionManager.init(allocator, config);
    defer manager.deinit();
    
    const keys = try createTestData(allocator, 128);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 128);
    defer allocator.free(values);
    
    const result = try manager.compressKVCache(keys, values);
    defer result[0].deinit();
    defer result[1].deinit();
    
    // Check stats updated
    try testing.expect(manager.stats.compress_count == 2);
    try testing.expect(manager.stats.total_original_bytes > 0);
    try testing.expect(manager.stats.total_compressed_bytes > 0);
}

test "compression_manager: decompress KV cache" {
    const allocator = testing.allocator;
    const config = compression.CompressionConfig{
        .algorithm = .fp16,
    };
    
    const manager = try compression.CompressionManager.init(allocator, config);
    defer manager.deinit();
    
    const keys = try createTestData(allocator, 128);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 128);
    defer allocator.free(values);
    
    // Compress
    const compressed = try manager.compressKVCache(keys, values);
    defer compressed[0].deinit();
    defer compressed[1].deinit();
    
    // Decompress
    const decompressed = try manager.decompressKVCache(compressed[0], compressed[1]);
    defer allocator.free(decompressed[0]);
    defer allocator.free(decompressed[1]);
    
    // Check stats
    try testing.expect(manager.stats.decompress_count == 2);
    
    // Check accuracy
    const keys_error = calculateMeanError(keys, decompressed[0]);
    const values_error = calculateMeanError(values, decompressed[1]);
    try testing.expect(keys_error < 0.01);
    try testing.expect(values_error < 0.01);
}

test "compression_manager: statistics" {
    const allocator = testing.allocator;
    const config = compression.CompressionConfig{
        .algorithm = .fp16,
    };
    
    const manager = try compression.CompressionManager.init(allocator, config);
    defer manager.deinit();
    
    // Perform multiple compressions
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const data = try createTestData(allocator, 256);
        defer allocator.free(data);
        
        const shape = [_]usize{data.len};
        const compressed = try compression.compress(allocator, data, &shape, config);
        defer compressed.deinit();
        
        const original_bytes = data.len * @sizeOf(f32);
        const compressed_bytes = compressed.getCompressedSize();
        manager.stats.addCompression(original_bytes, compressed_bytes, 100); // 100μs
    }
    
    // Check statistics
    try testing.expect(manager.stats.compress_count == 10);
    try testing.expect(manager.stats.getAverageCompressionRatio() >= 1.9);
    try testing.expect(manager.stats.getAverageCompressTime() == 100.0);
    try testing.expect(manager.stats.getThroughputMBps() > 0);
}

// ============================================================================
// Round-Trip Tests
// ============================================================================

test "round_trip: fp16 accuracy" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 1000);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{data.len};
    
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // FP16 should have <1% error
    const mean_error = calculateMeanError(data, decompressed);
    const max_error = calculateMaxError(data, decompressed);
    
    try testing.expect(mean_error < 0.01);
    try testing.expect(max_error < 0.1);
}

test "round_trip: int8_symmetric accuracy" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 1000);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{ .algorithm = .int8_symmetric };
    const shape = [_]usize{data.len};
    
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // INT8 should have <5% error
    const mean_error = calculateMeanError(data, decompressed);
    try testing.expect(mean_error < 0.05);
}

test "round_trip: int8_asymmetric accuracy" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 1000);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{ .algorithm = .int8_asymmetric };
    const shape = [_]usize{data.len};
    
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // INT8 should have <5% error
    const mean_error = calculateMeanError(data, decompressed);
    try testing.expect(mean_error < 0.05);
}

// ============================================================================
// Performance Tests
// ============================================================================

test "performance: compression speed" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 10000);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{data.len};
    
    const start = std.time.nanoTimestamp();
    const compressed = try compression.compress(allocator, data, &shape, config);
    const end = std.time.nanoTimestamp();
    
    defer compressed.deinit();
    
    const elapsed_us = @divTrunc(end - start, 1000);
    const throughput_mbps = (@as(f64, @floatFromInt(data.len * @sizeOf(f32))) / (1024.0 * 1024.0)) /
                           (@as(f64, @floatFromInt(elapsed_us)) / 1_000_000.0);
    
    // Should compress at >100 MB/s
    try testing.expect(throughput_mbps > 100.0);
}

test "performance: decompression speed" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 10000);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{data.len};
    
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    const start = std.time.nanoTimestamp();
    const decompressed = try compression.decompress(allocator, compressed);
    const end = std.time.nanoTimestamp();
    
    defer allocator.free(decompressed);
    
    const elapsed_us = @divTrunc(end - start, 1000);
    const throughput_mbps = (@as(f64, @floatFromInt(decompressed.len * @sizeOf(f32))) / (1024.0 * 1024.0)) /
                           (@as(f64, @floatFromInt(elapsed_us)) / 1_000_000.0);
    
    // Decompression should be faster than compression
    try testing.expect(throughput_mbps > 100.0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "edge_cases: empty data" {
    const allocator = testing.allocator;
    const data = [_]f32{};
    
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{0};
    
    const compressed = try compression.compress(allocator, &data, &shape, config);
    defer compressed.deinit();
    
    try testing.expect(compressed.element_count == 0);
    try testing.expect(compressed.data.len == 0);
}

test "edge_cases: single element" {
    const allocator = testing.allocator;
    const data = [_]f32{0.5};
    
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{1};
    
    const compressed = try compression.compress(allocator, &data, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    try testing.expect(decompressed.len == 1);
    try testing.expect(@abs(decompressed[0] - 0.5) < 0.01);
}

test "edge_cases: special values" {
    const allocator = testing.allocator;
    const special_values = [_]f32{
        0.0, -0.0, 1.0, -1.0,
        std.math.inf(f32), -std.math.inf(f32),
    };
    
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    const shape = [_]usize{special_values.len};
    
    const compressed = try compression.compress(allocator, &special_values, &shape, config);
    defer compressed.deinit();
    
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Check zeros and ones
    try testing.expect(@abs(decompressed[0]) < 0.001);
    try testing.expect(@abs(decompressed[1]) < 0.001);
    try testing.expect(@abs(decompressed[2] - 1.0) < 0.01);
    try testing.expect(@abs(decompressed[3] + 1.0) < 0.01);
    
    // Check infinities
    try testing.expect(std.math.isInf(decompressed[4]));
    try testing.expect(std.math.isInf(decompressed[5]));
}

// ============================================================================
// Compression Comparison Tests
// ============================================================================

test "comparison: algorithms" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 1000);
    defer allocator.free(data);
    
    const shape = [_]usize{data.len};
    
    // Test all algorithms
    const algorithms = [_]compression.CompressionAlgorithm{
        .none,
        .fp16,
        .int8_symmetric,
        .int8_asymmetric,
    };
    
    for (algorithms) |algo| {
        const config = compression.CompressionConfig{ .algorithm = algo };
        const compressed = try compression.compress(allocator, data, &shape, config);
        defer compressed.deinit();
        
        const expected_ratio = algo.compressionRatio();
        const actual_ratio = compressed.getCompressionRatio();
        
        // Allow 5% tolerance
        try testing.expect(@abs(actual_ratio - expected_ratio) < expected_ratio * 0.05);
    }
}

test "comparison: accuracy vs compression" {
    const allocator = testing.allocator;
    const data = try createTestData(allocator, 1000);
    defer allocator.free(data);
    
    const shape = [_]usize{data.len};
    
    // FP16: 2x compression, high accuracy
    {
        const config = compression.CompressionConfig{ .algorithm = .fp16 };
        const compressed = try compression.compress(allocator, data, &shape, config);
        defer compressed.deinit();
        
        const decompressed = try compression.decompress(allocator, compressed);
        defer allocator.free(decompressed);
        
        const error = calculateMeanError(data, decompressed);
        try testing.expect(compressed.getCompressionRatio() >= 1.9);
        try testing.expect(error < 0.01); // <1% error
    }
    
    // INT8: 4x compression, moderate accuracy
    {
        const config = compression.CompressionConfig{ .algorithm = .int8_symmetric };
        const compressed = try compression.compress(allocator, data, &shape, config);
        defer compressed.deinit();
        
        const decompressed = try compression.decompress(allocator, compressed);
        defer allocator.free(decompressed);
        
        const error = calculateMeanError(data, decompressed);
        try testing.expect(compressed.getCompressionRatio() >= 3.9);
        try testing.expect(error < 0.05); // <5% error
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

test "integration: multiple compress/decompress cycles" {
    const allocator = testing.allocator;
    const config = compression.CompressionConfig{ .algorithm = .fp16 };
    
    const manager = try compression.CompressionManager.init(allocator, config);
    defer manager.deinit();
    
    var keys = try createTestData(allocator, 256);
    defer allocator.free(keys);
    var values = try createTestData(allocator, 256);
    defer allocator.free(values);
    
    // Perform 5 compress/decompress cycles
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        const compressed = try manager.compressKVCache(keys, values);
        defer compressed[0].deinit();
        defer compressed[1].deinit();
        
        const decompressed = try manager.decompressKVCache(compressed[0], compressed[1]);
        defer allocator.free(decompressed[0]);
        defer allocator.free(decompressed[1]);
        
        // Verify accuracy maintained across cycles
        const error = calculateMeanError(keys, decompressed[0]);
        try testing.expect(error < 0.01);
    }
    
    // Check cumulative stats
    try testing.expect(manager.stats.compress_count == 10); // 5 cycles * 2 (K+V)
    try testing.expect(manager.stats.decompress_count == 10);
}

test "integration: large tensor compression" {
    const allocator = testing.allocator;
    const large_size: usize = 100000; // 100K elements
    
    const data = try createTestData(allocator, large_size);
    defer allocator.free(data);
    
    const config = compression.CompressionConfig{ .algorithm = .int8_symmetric };
    const shape = [_]usize{data.len};
    
    const compressed = try compression.compress(allocator, data, &shape, config);
    defer compressed.deinit();
    
    // Should save significant memory (4x)
    const saved_bytes = compressed.getOriginalSize() - compressed.getCompressedSize();
    const saved_mb = @as(f32, @floatFromInt(saved_bytes)) / (1024.0 * 1024.0);
    
    try testing.expect(saved_mb > 1.0); // At least 1MB saved
    try testing.expect(compressed.getCompressionRatio() >= 3.9);
    
    // Verify decompression works
    const decompressed = try compression.decompress(allocator, compressed);
    defer allocator.free(decompressed);
    
    try testing.expect(decompressed.len == data.len);
}
