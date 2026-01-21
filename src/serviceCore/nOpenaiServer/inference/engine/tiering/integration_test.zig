// Tiering Integration Test
// Test that all tiering components work together

const std = @import("std");
const ssd = @import("ssd_tier.zig");
const tiered_kv = @import("tiered_kv_cache.zig");
const mmap_gguf = @import("mmap_gguf.zig");
const tiered_tensors = @import("tiered_tensors.zig");
const compression = @import("compression.zig");
const encryption = @import("encryption.zig");
const async_io = @import("async_io.zig");
// Note: distributed_tier and unified_tier have external dependencies
// They should be tested when integrated with the full project

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n" ++ "═" ** 70 ++ "\n", .{});
    std.debug.print("🧪 Tiering Integration Test\n", .{});
    std.debug.print("═" ** 70 ++ "\n\n", .{});
    
    var passed: u32 = 0;
    var failed: u32 = 0;
    
    // Test 1: SSD Storage
    {
        std.debug.print("1️⃣  Testing SSD Storage...\n", .{});
        const storage = ssd.SSDStorage.init(allocator, .{
            .ssd_path = "/tmp/shimmy_test.tier",
            .max_ssd_mb = 128,
        }) catch |err| {
            std.debug.print("   ❌ Failed to init: {}\n", .{err});
            failed += 1;
            return;
        };
        defer storage.deinit();
        
        try storage.open();
        
        // Write test data
        const test_data = "Hello, SSD tiering!";
        const offset = try storage.allocBlock(4096);
        try storage.write(offset, test_data);
        
        // Read back
        const read_data = try storage.read(offset, test_data.len);
        if (std.mem.eql(u8, read_data, test_data)) {
            std.debug.print("   ✅ SSD read/write works\n", .{});
            passed += 1;
        } else {
            std.debug.print("   ❌ Data mismatch\n", .{});
            failed += 1;
        }
        
        storage.close();
        std.fs.cwd().deleteFile("/tmp/shimmy_test.tier") catch {};
    }
    
    // Test 2: Tiered KV Cache
    {
        std.debug.print("\n2️⃣  Testing Tiered KV Cache...\n", .{});
        const kv = tiered_kv.TieredKVCache.init(allocator, .{
            .n_layers = 2,
            .n_heads = 4,
            .head_dim = 64,
            .max_seq_len = 1000,
            .hot_tokens = 100,
            .max_ssd_mb = 128,
            .ssd_path = "/tmp/shimmy_kv_test.tier",
        }) catch |err| {
            std.debug.print("   ❌ Failed to init: {}\n", .{err});
            failed += 1;
            return;
        };
        defer kv.deinit();
        
        const kv_dim: usize = 4 * 64; // n_heads * head_dim
        const keys = try allocator.alloc(f32, kv_dim);
        defer allocator.free(keys);
        const values = try allocator.alloc(f32, kv_dim);
        defer allocator.free(values);
        
        // Fill with test data
        for (0..kv_dim) |i| {
            keys[i] = @as(f32, @floatFromInt(i)) * 0.01;
            values[i] = @as(f32, @floatFromInt(i)) * 0.02;
        }
        
        // Store tokens
        for (0..50) |_| {
            try kv.store(0, keys, values);
            try kv.store(1, keys, values);
            kv.advance();
        }
        
        std.debug.print("   ✅ Stored 50 tokens\n", .{});
        std.debug.print("   Seq pos: {d}, Hot tokens: {d}\n", .{
            kv.seq_pos, @min(kv.seq_pos, kv.config.hot_tokens),
        });
        passed += 1;
        
        std.fs.cwd().deleteFile("/tmp/shimmy_kv_test.tier") catch {};
    }
    
    // Test 3: Hot/Cold eviction
    {
        std.debug.print("\n3️⃣  Testing Hot/Cold Eviction...\n", .{});
        const kv = tiered_kv.TieredKVCache.init(allocator, .{
            .n_layers = 1,
            .n_heads = 2,
            .head_dim = 32,
            .max_seq_len = 500,
            .hot_tokens = 50,
            .cold_block_tokens = 25,
            .max_ssd_mb = 128,
            .ssd_path = "/tmp/shimmy_evict_test.tier",
        }) catch |err| {
            std.debug.print("   ❌ Failed to init: {}\n", .{err});
            failed += 1;
            return;
        };
        defer kv.deinit();
        
        const kv_dim: usize = 2 * 32;
        const keys = try allocator.alloc(f32, kv_dim);
        defer allocator.free(keys);
        const values = try allocator.alloc(f32, kv_dim);
        defer allocator.free(values);
        
        @memset(keys, 1.0);
        @memset(values, 2.0);
        
        // Store enough tokens to trigger eviction
        for (0..100) |_| {
            try kv.store(0, keys, values);
            kv.advance();
        }
        
        const stats = kv.getStats();
        std.debug.print("   Evictions: {d}\n", .{stats.evictions});
        std.debug.print("   Cold blocks: {d}\n", .{stats.cold_blocks});
        std.debug.print("   SSD usage: {d} MB\n", .{stats.ssd_usage_mb});
        
        if (stats.evictions > 0) {
            std.debug.print("   ✅ Eviction to SSD works\n", .{});
            passed += 1;
        } else {
            std.debug.print("   ❌ Expected evictions\n", .{});
            failed += 1;
        }
        
        std.fs.cwd().deleteFile("/tmp/shimmy_evict_test.tier") catch {};
    }

    // Test 4: Compression
    {
        std.debug.print("\n4️⃣  Testing Compression...\n", .{});
        const compressor = compression.KVCompressor.init(allocator, .{}) catch |err| {
            std.debug.print("   ❌ Failed to init: {}\n", .{err});
            failed += 1;
            return;
        };
        defer compressor.deinit();

        // Create test data with repeating patterns (compressible)
        var test_data: [1024]f32 = undefined;
        for (0..1024) |i| {
            test_data[i] = @as(f32, @floatFromInt(i % 32)) * 0.1;
        }

        const compressed = try compressor.compress(&test_data);
        defer allocator.free(compressed);

        const ratio = @as(f64, @floatFromInt(test_data.len * 4)) / @as(f64, @floatFromInt(compressed.len));
        std.debug.print("   Compression ratio: {d:.2}x\n", .{ratio});

        // Decompress
        var decompressed: [1024]f32 = undefined;
        try compressor.decompress(compressed, &decompressed);

        // Verify
        var match = true;
        for (0..1024) |i| {
            if (test_data[i] != decompressed[i]) {
                match = false;
                break;
            }
        }

        if (match) {
            std.debug.print("   ✅ Compression/decompression works\n", .{});
            passed += 1;
        } else {
            std.debug.print("   ❌ Data mismatch after decompression\n", .{});
            failed += 1;
        }
    }

    // Test 5: Async I/O
    {
        std.debug.print("\n5️⃣  Testing Async I/O...\n", .{});
        const engine = async_io.AsyncIOEngine.init(allocator, .{}) catch |err| {
            std.debug.print("   ❌ Failed to init: {}\n", .{err});
            failed += 1;
            return;
        };
        defer engine.deinit();

        // Create test file
        const file = try std.fs.cwd().createFile("/tmp/shimmy_async_test.bin", .{ .read = true });
        defer {
            file.close();
            std.fs.cwd().deleteFile("/tmp/shimmy_async_test.bin") catch {};
        }

        // Write test data
        var write_buf = [_]u8{'X'} ** 4096;
        try engine.submitWrite(file, 0, &write_buf, null);
        try engine.waitAll();

        // Read it back
        var read_buf: [4096]u8 = undefined;
        try engine.submitRead(file, 0, &read_buf, null);
        try engine.waitAll();

        if (std.mem.eql(u8, &read_buf, &write_buf)) {
            std.debug.print("   ✅ Async read/write works\n", .{});
            std.debug.print("   Avg latency: {d:.1} µs\n", .{engine.stats.avgLatencyUs()});
            passed += 1;
        } else {
            std.debug.print("   ❌ Data mismatch\n", .{});
            failed += 1;
        }
    }

    // Summary
    std.debug.print("\n" ++ "═" ** 70 ++ "\n", .{});
    std.debug.print("📊 Results: {d} passed, {d} failed\n", .{passed, failed});
    std.debug.print("═" ** 70 ++ "\n", .{});
    
    if (failed > 0) {
        std.process.exit(1);
    }
}

