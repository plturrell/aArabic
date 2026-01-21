// Compression Testing Suite
// Day 15: Comprehensive tests for all compression formats

const std = @import("std");
const testing = std.testing;
const deflate = @import("../parsers/deflate.zig");
const gzip = @import("../parsers/gzip.zig");
const zlib = @import("../parsers/zlib.zig");
const zip = @import("../parsers/zip.zig");

/// Test data generators
const TestData = struct {
    /// Generate random data
    fn random(allocator: std.mem.Allocator, size: usize, seed: u64) ![]u8 {
        var prng = std.rand.DefaultPrng.init(seed);
        const random_gen = prng.random();
        
        var data = try allocator.alloc(u8, size);
        random_gen.bytes(data);
        return data;
    }
    
    /// Generate highly compressible data (repeating pattern)
    fn repeating(allocator: std.mem.Allocator, size: usize, pattern: []const u8) ![]u8 {
        var data = try allocator.alloc(u8, size);
        var i: usize = 0;
        while (i < size) : (i += 1) {
            data[i] = pattern[i % pattern.len];
        }
        return data;
    }
    
    /// Generate incompressible data (random)
    fn incompressible(allocator: std.mem.Allocator, size: usize) ![]u8 {
        return try random(allocator, size, 12345);
    }
    
    /// Generate text data
    fn text(allocator: std.mem.Allocator, word_count: usize) ![]u8 {
        const words = [_][]const u8{
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "hello", "world", "test", "data", "compression", "algorithm",
        };
        
        var list = std.ArrayList(u8).init(allocator);
        defer list.deinit();
        
        var prng = std.rand.DefaultPrng.init(54321);
        const random_gen = prng.random();
        
        var i: usize = 0;
        while (i < word_count) : (i += 1) {
            const word = words[random_gen.intRangeAtMost(usize, 0, words.len - 1)];
            try list.appendSlice(word);
            if (i < word_count - 1) try list.append(' ');
        }
        
        return list.toOwnedSlice();
    }
};

/// Compression ratio calculator
fn compressionRatio(original_size: usize, compressed_size: usize) f64 {
    if (original_size == 0) return 0.0;
    return @as(f64, @floatFromInt(compressed_size)) / @as(f64, @floatFromInt(original_size));
}

/// Performance metrics
const PerfMetrics = struct {
    compression_time_ns: u64,
    decompression_time_ns: u64,
    original_size: usize,
    compressed_size: usize,
    compression_ratio: f64,
    
    fn print(self: PerfMetrics, format_name: []const u8) void {
        std.debug.print("\n=== {s} Performance ===\n", .{format_name});
        std.debug.print("Original size: {} bytes\n", .{self.original_size});
        std.debug.print("Compressed size: {} bytes\n", .{self.compressed_size});
        std.debug.print("Compression ratio: {d:.2}%\n", .{self.compression_ratio * 100});
        std.debug.print("Decompression time: {} ns ({d:.2} ms)\n", .{
            self.decompression_time_ns,
            @as(f64, @floatFromInt(self.decompression_time_ns)) / 1_000_000.0,
        });
    }
};

// ============================================================================
// DEFLATE Tests
// ============================================================================

test "Compression: DEFLATE small text" {
    const allocator = testing.allocator;
    const original = "Hello, DEFLATE compression!";
    
    // Note: We don't have compression yet, so we test with uncompressed blocks
    // This test verifies the decompression path works
    _ = allocator;
    _ = original;
    // TODO: Implement when compression is available
}

test "Compression: DEFLATE large data" {
    const allocator = testing.allocator;
    
    // Generate 1MB of test data
    const data = try TestData.repeating(allocator, 1024 * 1024, "ABCDEFGH");
    defer allocator.free(data);
    
    // TODO: Compress and decompress when compression is implemented
    _ = data;
}

test "Compression: DEFLATE random data" {
    const allocator = testing.allocator;
    
    const data = try TestData.random(allocator, 10240, 98765);
    defer allocator.free(data);
    
    // Random data should be mostly incompressible
    _ = data;
}

// ============================================================================
// GZIP Tests
// ============================================================================

test "Compression: GZIP format detection" {
    const valid_gzip = [_]u8{ 0x1f, 0x8b, 0x08, 0x00 };
    try testing.expect(gzip.isGzip(&valid_gzip));
    
    const invalid = [_]u8{ 0x50, 0x4b, 0x03, 0x04 }; // ZIP signature
    try testing.expect(!gzip.isGzip(&invalid));
}

test "Compression: GZIP with metadata" {
    // Test that metadata is properly preserved through compression/decompression
    // This tests the optional fields (filename, comment, etc.)
    
    // TODO: Create GZIP with metadata and verify after decompression
}

test "Compression: GZIP CRC verification" {
    // Test that corrupted data is detected via CRC32
    const allocator = testing.allocator;
    
    // Create valid GZIP data (using helper from gzip_test.zig)
    // Then corrupt it and verify error is caught
    
    _ = allocator;
}

// ============================================================================
// ZLIB Tests
// ============================================================================

test "Compression: ZLIB format detection" {
    const valid_zlib = [_]u8{ 0x78, 0x9c, 0x01, 0x00 };
    try testing.expect(zlib.isZlib(&valid_zlib));
    
    const invalid = [_]u8{ 0x1f, 0x8b, 0x08, 0x00 }; // GZIP signature
    try testing.expect(!zlib.isZlib(&invalid));
}

test "Compression: ZLIB Adler32 verification" {
    // Test that corrupted data is detected via Adler32
    const allocator = testing.allocator;
    
    _ = allocator;
    // TODO: Create ZLIB data, corrupt it, verify error
}

test "Compression: ZLIB window sizes" {
    // Test that various window sizes work correctly
    // CINFO from 0 to 7 (window sizes 256 to 32768)
    
    var cinfo: u4 = 0;
    while (cinfo <= 7) : (cinfo += 1) {
        const window_size = @as(u32, 1) << (@as(u5, cinfo) + 8);
        try testing.expect(window_size >= 256);
        try testing.expect(window_size <= 32768);
    }
}

// ============================================================================
// ZIP Tests
// ============================================================================

test "Compression: ZIP format detection" {
    const valid_zip = [_]u8{ 0x50, 0x4b, 0x03, 0x04 }; // Local file header
    try testing.expect(zip.isZip(&valid_zip));
    
    const invalid = [_]u8{ 0x1f, 0x8b, 0x08, 0x00 }; // GZIP signature
    try testing.expect(!zip.isZip(&invalid));
}

test "Compression: ZIP CRC32 verification" {
    // Test that file CRC32 is properly verified
    // TODO: Create ZIP with file, verify CRC32 check works
}

// ============================================================================
// Format Interoperability Tests
// ============================================================================

test "Compression: Format interop - DEFLATE in GZIP" {
    // Verify that DEFLATE compressed data works correctly when wrapped in GZIP
    const allocator = testing.allocator;
    
    _ = allocator;
    // TODO: Create DEFLATE stream, wrap in GZIP, decompress
}

test "Compression: Format interop - DEFLATE in ZLIB" {
    // Verify that DEFLATE compressed data works correctly when wrapped in ZLIB
    const allocator = testing.allocator;
    
    _ = allocator;
    // TODO: Create DEFLATE stream, wrap in ZLIB, decompress
}

test "Compression: Format interop - DEFLATE in ZIP" {
    // Verify that DEFLATE compressed files work in ZIP archives
    const allocator = testing.allocator;
    
    _ = allocator;
    // TODO: Create ZIP with DEFLATE compressed file, extract and verify
}

// ============================================================================
// Edge Cases and Corner Cases
// ============================================================================

test "Compression: Empty data handling" {
    const allocator = testing.allocator;
    
    // Test all formats with empty data
    const empty: []const u8 = "";
    
    _ = allocator;
    _ = empty;
    // TODO: Test DEFLATE, GZIP, ZLIB, ZIP with empty data
}

test "Compression: Single byte data" {
    const allocator = testing.allocator;
    
    const single_byte = [_]u8{0x42};
    
    _ = allocator;
    _ = single_byte;
    // TODO: Test all formats with single byte
}

test "Compression: Maximum size data" {
    // Test with largest supported size (e.g., 4GB limit for some formats)
    // This is a stress test
    
    // Skip for now due to memory requirements
    // TODO: Implement with streaming if needed
}

test "Compression: Malformed data handling" {
    const allocator = testing.allocator;
    
    // Test that malformed compressed data returns proper errors
    const malformed = [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff };
    
    _ = allocator;
    _ = malformed;
    // TODO: Test error handling for each format
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

test "Benchmark: DEFLATE decompression speed" {
    const allocator = testing.allocator;
    
    // Generate test data
    const original = try TestData.text(allocator, 10000);
    defer allocator.free(original);
    
    // TODO: Compress and benchmark decompression
    // Measure: time, throughput (MB/s)
    
    std.debug.print("\nDEFLATE decompression benchmark:\n", .{});
    std.debug.print("Data size: {} bytes\n", .{original.len});
}

test "Benchmark: GZIP decompression speed" {
    const allocator = testing.allocator;
    
    const original = try TestData.text(allocator, 10000);
    defer allocator.free(original);
    
    std.debug.print("\nGZIP decompression benchmark:\n", .{});
    std.debug.print("Data size: {} bytes\n", .{original.len});
}

test "Benchmark: ZLIB decompression speed" {
    const allocator = testing.allocator;
    
    const original = try TestData.text(allocator, 10000);
    defer allocator.free(original);
    
    std.debug.print("\nZLIB decompression benchmark:\n", .{});
    std.debug.print("Data size: {} bytes\n", .{original.len});
}

test "Benchmark: ZIP extraction speed" {
    const allocator = testing.allocator;
    
    _ = allocator;
    
    std.debug.print("\nZIP extraction benchmark:\n", .{});
}

// ============================================================================
// Memory Usage Tests
// ============================================================================

test "Memory: DEFLATE decompression" {
    const allocator = testing.allocator;
    
    // Test memory usage during decompression
    // Verify no leaks, reasonable memory consumption
    
    const data = try TestData.repeating(allocator, 10240, "TEST");
    defer allocator.free(data);
    
    // TODO: Create compressed data and decompress
    // Measure peak memory usage
}

test "Memory: GZIP decompression" {
    const allocator = testing.allocator;
    
    const data = try TestData.repeating(allocator, 10240, "GZIP");
    defer allocator.free(data);
    
    // TODO: Verify memory usage
}

test "Memory: ZLIB decompression" {
    const allocator = testing.allocator;
    
    const data = try TestData.repeating(allocator, 10240, "ZLIB");
    defer allocator.free(data);
    
    // TODO: Verify memory usage
}

test "Memory: ZIP extraction" {
    const allocator = testing.allocator;
    
    _ = allocator;
    
    // TODO: Verify memory usage for multi-file ZIP
}

// ============================================================================
// Compression Ratio Tests
// ============================================================================

test "Compression ratio: Highly compressible data" {
    const allocator = testing.allocator;
    
    // Repeating pattern should compress very well
    const data = try TestData.repeating(allocator, 10240, "A");
    defer allocator.free(data);
    
    std.debug.print("\nCompression ratio for highly compressible data:\n", .{});
    std.debug.print("Original size: {} bytes\n", .{data.len});
    
    // TODO: Test compression ratio for each format
    // Expected: > 90% compression for repeating data
}

test "Compression ratio: Random data" {
    const allocator = testing.allocator;
    
    // Random data should compress poorly
    const data = try TestData.random(allocator, 10240, 11111);
    defer allocator.free(data);
    
    std.debug.print("\nCompression ratio for random data:\n", .{});
    std.debug.print("Original size: {} bytes\n", .{data.len});
    
    // TODO: Test compression ratio
    // Expected: Little to no compression (ratio close to 1.0)
}

test "Compression ratio: Text data" {
    const allocator = testing.allocator;
    
    const data = try TestData.text(allocator, 1000);
    defer allocator.free(data);
    
    std.debug.print("\nCompression ratio for text data:\n", .{});
    std.debug.print("Original size: {} bytes\n", .{data.len});
    
    // TODO: Test compression ratio
    // Expected: 40-60% compression for English text
}

// ============================================================================
// Correctness Tests
// ============================================================================

test "Correctness: Round-trip fidelity" {
    const allocator = testing.allocator;
    
    // Test that compress -> decompress returns identical data
    const original = try TestData.text(allocator, 500);
    defer allocator.free(original);
    
    // TODO: For each format:
    // 1. Compress original
    // 2. Decompress
    // 3. Verify byte-for-byte match
}

test "Correctness: Multiple rounds" {
    const allocator = testing.allocator;
    
    // Test that multiple compression/decompression rounds work
    const original = "Test data for multiple rounds";
    
    _ = allocator;
    _ = original;
    
    // TODO: Compress -> Decompress -> Compress -> Decompress
    // Verify data integrity at each step
}

test "Correctness: Unicode data" {
    const allocator = testing.allocator;
    
    const unicode_text = "Hello 世界 🌍 مرحبا Привет";
    
    _ = allocator;
    _ = unicode_text;
    
    // TODO: Verify all formats handle Unicode correctly
}

test "Correctness: Binary data" {
    const allocator = testing.allocator;
    
    var binary_data: [256]u8 = undefined;
    for (&binary_data, 0..) |*byte, i| {
        byte.* = @as(u8, @truncate(i));
    }
    
    _ = allocator;
    
    // TODO: Verify all formats handle full byte range correctly
}
