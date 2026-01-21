// Fuzzing Infrastructure for Compression Formats
// Day 15: Continuous fuzzing for DEFLATE, GZIP, ZLIB, ZIP

const std = @import("std");
const deflate = @import("../parsers/deflate.zig");
const gzip = @import("../parsers/gzip.zig");
const zlib = @import("../parsers/zlib.zig");
const zip = @import("../parsers/zip.zig");

/// Fuzzing statistics
const FuzzStats = struct {
    iterations: u64 = 0,
    crashes: u64 = 0,
    hangs: u64 = 0,
    errors: u64 = 0,
    unique_crashes: std.ArrayList([]const u8),
    
    fn init(allocator: std.mem.Allocator) FuzzStats {
        return .{
            .unique_crashes = std.ArrayList([]const u8).init(allocator),
        };
    }
    
    fn deinit(self: *FuzzStats) void {
        for (self.unique_crashes.items) |crash| {
            self.unique_crashes.allocator.free(crash);
        }
        self.unique_crashes.deinit();
    }
    
    fn print(self: FuzzStats) void {
        std.debug.print("\n=== Fuzzing Statistics ===\n", .{});
        std.debug.print("Iterations: {}\n", .{self.iterations});
        std.debug.print("Crashes: {}\n", .{self.crashes});
        std.debug.print("Hangs: {}\n", .{self.hangs});
        std.debug.print("Errors: {}\n", .{self.errors});
        std.debug.print("Unique crashes: {}\n", .{self.unique_crashes.items.len});
    }
};

/// Fuzz input generator
const FuzzInput = struct {
    /// Generate random bytes
    fn random(allocator: std.mem.Allocator, size: usize, prng: *std.rand.Random) ![]u8 {
        var data = try allocator.alloc(u8, size);
        prng.bytes(data);
        return data;
    }
    
    /// Mutate existing bytes
    fn mutate(allocator: std.mem.Allocator, original: []const u8, prng: *std.rand.Random) ![]u8 {
        var data = try allocator.dupe(u8, original);
        
        // Apply random mutations
        const num_mutations = prng.intRangeAtMost(usize, 1, 10);
        var i: usize = 0;
        while (i < num_mutations) : (i += 1) {
            const mutation_type = prng.intRangeAtMost(u8, 0, 4);
            
            switch (mutation_type) {
                0 => { // Bit flip
                    if (data.len > 0) {
                        const idx = prng.intRangeAtMost(usize, 0, data.len - 1);
                        const bit = @as(u8, 1) << @as(u3, @truncate(prng.intRangeAtMost(u8, 0, 7)));
                        data[idx] ^= bit;
                    }
                },
                1 => { // Byte flip
                    if (data.len > 0) {
                        const idx = prng.intRangeAtMost(usize, 0, data.len - 1);
                        data[idx] = prng.int(u8);
                    }
                },
                2 => { // Insert byte
                    if (data.len < 1024 * 1024) { // Limit size
                        const new_data = try allocator.alloc(u8, data.len + 1);
                        const idx = prng.intRangeAtMost(usize, 0, data.len);
                        @memcpy(new_data[0..idx], data[0..idx]);
                        new_data[idx] = prng.int(u8);
                        if (idx < data.len) {
                            @memcpy(new_data[idx + 1 ..], data[idx..]);
                        }
                        allocator.free(data);
                        data = new_data;
                    }
                },
                3 => { // Delete byte
                    if (data.len > 1) {
                        const idx = prng.intRangeAtMost(usize, 0, data.len - 1);
                        const new_data = try allocator.alloc(u8, data.len - 1);
                        @memcpy(new_data[0..idx], data[0..idx]);
                        if (idx + 1 < data.len) {
                            @memcpy(new_data[idx..], data[idx + 1 ..]);
                        }
                        allocator.free(data);
                        data = new_data;
                    }
                },
                4 => { // Duplicate chunk
                    if (data.len > 2 and data.len < 512 * 1024) {
                        const chunk_size = prng.intRangeAtMost(usize, 1, @min(16, data.len));
                        const src_idx = prng.intRangeAtMost(usize, 0, data.len - chunk_size);
                        const new_data = try allocator.alloc(u8, data.len + chunk_size);
                        const dst_idx = prng.intRangeAtMost(usize, 0, data.len);
                        
                        @memcpy(new_data[0..dst_idx], data[0..dst_idx]);
                        @memcpy(new_data[dst_idx .. dst_idx + chunk_size], data[src_idx .. src_idx + chunk_size]);
                        @memcpy(new_data[dst_idx + chunk_size ..], data[dst_idx..]);
                        
                        allocator.free(data);
                        data = new_data;
                    }
                },
                else => unreachable,
            }
        }
        
        return data;
    }
    
    /// Generate valid-looking GZIP header
    fn gzipHeader(allocator: std.mem.Allocator, prng: *std.rand.Random) ![]u8 {
        var data = std.ArrayList(u8).init(allocator);
        
        // Magic bytes
        try data.append(0x1f);
        try data.append(0x8b);
        
        // Compression method
        try data.append(8);
        
        // Random flags
        try data.append(prng.int(u8) & 0x1f); // Only lower 5 bits valid
        
        // MTIME (random)
        try data.appendSlice(&[_]u8{ prng.int(u8), prng.int(u8), prng.int(u8), prng.int(u8) });
        
        // XFL
        try data.append(prng.int(u8));
        
        // OS
        try data.append(prng.int(u8));
        
        return data.toOwnedSlice();
    }
    
    /// Generate valid-looking ZLIB header
    fn zlibHeader(allocator: std.mem.Allocator, prng: *std.rand.Random) ![]u8 {
        var data = std.ArrayList(u8).init(allocator);
        
        // CMF: compression method (8) and CINFO (0-7)
        const cinfo = @as(u8, @truncate(prng.intRangeAtMost(u4, 0, 7)));
        const cmf = (cinfo << 4) | 8;
        try data.append(cmf);
        
        // FLG: compute valid FCHECK
        var flg: u8 = prng.int(u8) & 0xe0; // Keep upper bits random
        const check_value = (@as(u16, cmf) * 256 + @as(u16, flg)) % 31;
        if (check_value != 0) {
            flg += @as(u8, @truncate(31 - check_value));
        }
        try data.append(flg);
        
        return data.toOwnedSlice();
    }
};

/// Fuzz target for DEFLATE
pub fn fuzzDeflate(allocator: std.mem.Allocator, input: []const u8) !void {
    _ = deflate.decompress(input, allocator) catch |err| {
        // Expected errors are OK
        switch (err) {
            error.InvalidBlockType,
            error.InvalidHuffmanCode,
            error.InvalidDistance,
            error.UnexpectedEndOfData,
            error.OutOfMemory,
            => {},
            else => return err,
        }
    };
}

/// Fuzz target for GZIP
pub fn fuzzGzip(allocator: std.mem.Allocator, input: []const u8) !void {
    var result = gzip.decompress(input, allocator) catch |err| {
        // Expected errors are OK
        switch (err) {
            error.InvalidMagic,
            error.UnsupportedCompressionMethod,
            error.InvalidHeader,
            error.InvalidFooter,
            error.CrcMismatch,
            error.SizeMismatch,
            error.DecompressionFailed,
            error.OutOfMemory,
            => return,
            else => return err,
        }
        return;
    };
    defer result.deinit();
}

/// Fuzz target for ZLIB
pub fn fuzzZlib(allocator: std.mem.Allocator, input: []const u8) !void {
    var result = zlib.decompress(input, allocator) catch |err| {
        // Expected errors are OK
        switch (err) {
            error.InvalidHeader,
            error.InvalidChecksum,
            error.UnsupportedCompressionMethod,
            error.DictionaryRequired,
            error.InvalidWindowSize,
            error.DecompressionFailed,
            error.Adler32Mismatch,
            error.OutOfMemory,
            => return,
            else => return err,
        }
        return;
    };
    defer result.deinit();
}

/// Fuzz target for ZIP
pub fn fuzzZip(allocator: std.mem.Allocator, input: []const u8) !void {
    _ = zip.parse(input, allocator) catch |err| {
        // Expected errors are OK
        switch (err) {
            error.InvalidSignature,
            error.InvalidHeader,
            error.UnsupportedCompressionMethod,
            error.UnsupportedVersion,
            error.CrcMismatch,
            error.InvalidCentralDirectory,
            error.OutOfMemory,
            => return,
            else => return err,
        }
        return;
    };
}

/// Run fuzzing campaign
pub fn runFuzzCampaign(
    allocator: std.mem.Allocator,
    comptime target: fn (std.mem.Allocator, []const u8) anyerror!void,
    iterations: u64,
    seed: u64,
) !FuzzStats {
    var stats = FuzzStats.init(allocator);
    errdefer stats.deinit();
    
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();
    
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        stats.iterations += 1;
        
        // Generate random input
        const size = random.intRangeAtMost(usize, 0, 1024);
        const input = try FuzzInput.random(allocator, size, &random);
        defer allocator.free(input);
        
        // Run fuzz target
        target(allocator, input) catch |err| {
            stats.errors += 1;
            
            // Check if this is a crash (unexpected error)
            switch (err) {
                error.OutOfMemory => {}, // Expected
                else => {
                    stats.crashes += 1;
                    const crash_msg = try std.fmt.allocPrint(
                        allocator,
                        "Crash at iteration {}: {}",
                        .{ i, err },
                    );
                    try stats.unique_crashes.append(crash_msg);
                },
            }
        };
        
        // Print progress every 1000 iterations
        if (i > 0 and i % 1000 == 0) {
            std.debug.print(".", .{});
        }
    }
    
    std.debug.print("\n", .{});
    return stats;
}

/// Corpus-based fuzzing (use known inputs and mutate them)
pub fn runCorpusFuzzing(
    allocator: std.mem.Allocator,
    comptime target: fn (std.mem.Allocator, []const u8) anyerror!void,
    corpus: []const []const u8,
    iterations: u64,
    seed: u64,
) !FuzzStats {
    var stats = FuzzStats.init(allocator);
    errdefer stats.deinit();
    
    if (corpus.len == 0) return stats;
    
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();
    
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        stats.iterations += 1;
        
        // Select random corpus input
        const corpus_idx = random.intRangeAtMost(usize, 0, corpus.len - 1);
        const original = corpus[corpus_idx];
        
        // Mutate it
        const input = try FuzzInput.mutate(allocator, original, &random);
        defer allocator.free(input);
        
        // Run fuzz target
        target(allocator, input) catch |err| {
            stats.errors += 1;
            
            switch (err) {
                error.OutOfMemory => {},
                else => {
                    stats.crashes += 1;
                    const crash_msg = try std.fmt.allocPrint(
                        allocator,
                        "Crash at iteration {}: {}",
                        .{ i, err },
                    );
                    try stats.unique_crashes.append(crash_msg);
                },
            }
        };
        
        if (i > 0 and i % 1000 == 0) {
            std.debug.print(".", .{});
        }
    }
    
    std.debug.print("\n", .{});
    return stats;
}

// ============================================================================
// Test entry points
// ============================================================================

const testing = std.testing;

test "Fuzz: DEFLATE basic" {
    const allocator = testing.allocator;
    
    std.debug.print("\nFuzzing DEFLATE (10,000 iterations)...\n", .{});
    var stats = try runFuzzCampaign(allocator, fuzzDeflate, 10_000, 12345);
    defer stats.deinit();
    
    stats.print();
    try testing.expect(stats.crashes == 0);
}

test "Fuzz: GZIP basic" {
    const allocator = testing.allocator;
    
    std.debug.print("\nFuzzing GZIP (10,000 iterations)...\n", .{});
    var stats = try runFuzzCampaign(allocator, fuzzGzip, 10_000, 23456);
    defer stats.deinit();
    
    stats.print();
    try testing.expect(stats.crashes == 0);
}

test "Fuzz: ZLIB basic" {
    const allocator = testing.allocator;
    
    std.debug.print("\nFuzzing ZLIB (10,000 iterations)...\n", .{});
    var stats = try runFuzzCampaign(allocator, fuzzZlib, 10_000, 34567);
    defer stats.deinit();
    
    stats.print();
    try testing.expect(stats.crashes == 0);
}

test "Fuzz: ZIP basic" {
    const allocator = testing.allocator;
    
    std.debug.print("\nFuzzing ZIP (10,000 iterations)...\n", .{});
    var stats = try runFuzzCampaign(allocator, fuzzZip, 10_000, 45678);
    defer stats.deinit();
    
    stats.print();
    try testing.expect(stats.crashes == 0);
}

test "Fuzz: GZIP with valid headers" {
    const allocator = testing.allocator;
    
    std.debug.print("\nFuzzing GZIP with valid headers (5,000 iterations)...\n", .{});
    
    // Create corpus of valid GZIP headers
    var corpus = std.ArrayList([]const u8).init(allocator);
    defer {
        for (corpus.items) |item| {
            allocator.free(item);
        }
        corpus.deinit();
    }
    
    var prng = std.rand.DefaultPrng.init(11111);
    const random = prng.random();
    
    // Generate 10 valid headers
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const header = try FuzzInput.gzipHeader(allocator, &random);
        try corpus.append(header);
    }
    
    var stats = try runCorpusFuzzing(allocator, fuzzGzip, corpus.items, 5_000, 56789);
    defer stats.deinit();
    
    stats.print();
    try testing.expect(stats.crashes == 0);
}

test "Fuzz: ZLIB with valid headers" {
    const allocator = testing.allocator;
    
    std.debug.print("\nFuzzing ZLIB with valid headers (5,000 iterations)...\n", .{});
    
    // Create corpus of valid ZLIB headers
    var corpus = std.ArrayList([]const u8).init(allocator);
    defer {
        for (corpus.items) |item| {
            allocator.free(item);
        }
        corpus.deinit();
    }
    
    var prng = std.rand.DefaultPrng.init(22222);
    const random = prng.random();
    
    // Generate 10 valid headers
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const header = try FuzzInput.zlibHeader(allocator, &random);
        try corpus.append(header);
    }
    
    var stats = try runCorpusFuzzing(allocator, fuzzZlib, corpus.items, 5_000, 67890);
    defer stats.deinit();
    
    stats.print();
    try testing.expect(stats.crashes == 0);
}

test "Fuzz: Edge cases - empty input" {
    const allocator = testing.allocator;
    const empty: []const u8 = "";
    
    // All parsers should handle empty input gracefully
    try fuzzDeflate(allocator, empty);
    try fuzzGzip(allocator, empty);
    try fuzzZlib(allocator, empty);
    try fuzzZip(allocator, empty);
}

test "Fuzz: Edge cases - single byte" {
    const allocator = testing.allocator;
    
    var i: u8 = 0;
    while (true) : (i +%= 1) {
        const byte = [_]u8{i};
        
        try fuzzDeflate(allocator, &byte);
        try fuzzGzip(allocator, &byte);
        try fuzzZlib(allocator, &byte);
        try fuzzZip(allocator, &byte);
        
        if (i == 255) break;
    }
}

test "Fuzz: Stress test - large input" {
    const allocator = testing.allocator;
    
    // Test with 1MB of random data
    var prng = std.rand.DefaultPrng.init(99999);
    const random = prng.random();
    
    const large_input = try FuzzInput.random(allocator, 1024 * 1024, &random);
    defer allocator.free(large_input);
    
    std.debug.print("\nFuzzing with 1MB input...\n", .{});
    
    try fuzzDeflate(allocator, large_input);
    try fuzzGzip(allocator, large_input);
    try fuzzZlib(allocator, large_input);
    try fuzzZip(allocator, large_input);
}
