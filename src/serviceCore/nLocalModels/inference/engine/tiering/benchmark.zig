// Tiering Benchmark
// Tests SSD-backed KV cache and tensor loading performance

const std = @import("std");
const ssd = @import("ssd_tier.zig");
const tiered_kv = @import("tiered_kv_cache.zig");
const mmap_gguf = @import("mmap_gguf.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n" ++ "═" ** 70 ++ "\n", .{});
    std.debug.print("🏋️ Tiering Benchmark\n", .{});
    std.debug.print("═" ** 70 ++ "\n\n", .{});
    
    // Parse args
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        std.debug.print("Usage: benchmark <test>\n", .{});
        std.debug.print("  ssd      - SSD storage benchmark\n", .{});
        std.debug.print("  kv       - KV cache benchmark\n", .{});
        std.debug.print("  gguf     - GGUF mmap benchmark (requires model path)\n", .{});
        std.debug.print("  all      - Run all benchmarks\n", .{});
        return;
    }
    
    const test_name = args[1];
    
    if (std.mem.eql(u8, test_name, "ssd") or std.mem.eql(u8, test_name, "all")) {
        try benchmarkSSD(allocator);
    }
    
    if (std.mem.eql(u8, test_name, "kv") or std.mem.eql(u8, test_name, "all")) {
        try benchmarkKVCache(allocator);
    }
    
    if (std.mem.eql(u8, test_name, "gguf")) {
        if (args.len < 3) {
            std.debug.print("Usage: benchmark gguf <model.gguf>\n", .{});
            return;
        }
        try benchmarkGGUF(allocator, args[2]);
    }
    
    std.debug.print("\n✅ Benchmark complete\n", .{});
}

fn benchmarkSSD(allocator: std.mem.Allocator) !void {
    std.debug.print("\n📀 SSD Storage Benchmark\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});
    
    const config = ssd.TierConfig{
        .ssd_path = "/tmp/shimmy_bench.tier",
        .max_ssd_mb = 1024, // 1GB for benchmark
    };
    
    const storage = try ssd.SSDStorage.init(allocator, config);
    defer storage.deinit();
    
    try storage.open();
    
    // Benchmark write
    const block_sizes = [_]u32{ 4096, 16384, 65536, 262144, 1048576 };
    
    for (block_sizes) |block_size| {
        const data = try allocator.alloc(u8, block_size);
        defer allocator.free(data);
        @memset(data, 0xAB);
        
        const iterations: u32 = 1000;
        const start = std.time.nanoTimestamp();
        
        for (0..iterations) |_| {
            const offset = try storage.allocBlock(block_size);
            try storage.write(offset, data);
            storage.freeBlock(offset, block_size);
        }
        
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
        const mb_per_sec = ops_per_sec * @as(f64, @floatFromInt(block_size)) / (1024.0 * 1024.0);
        
        std.debug.print("   {d:>7} bytes: {d:>8.0} ops/s, {d:>8.1} MB/s\n", .{
            block_size, ops_per_sec, mb_per_sec,
        });
    }
    
    storage.close();
    
    // Clean up
    std.fs.cwd().deleteFile("/tmp/shimmy_bench.tier") catch {};
}

fn benchmarkKVCache(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🗄️ KV Cache Benchmark\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});
    
    const config = tiered_kv.TieredKVConfig{
        .n_layers = 32,
        .n_heads = 32,
        .head_dim = 128,
        .max_seq_len = 10000,
        .hot_tokens = 1024,
        .max_ssd_mb = 512,
        .ssd_path = "/tmp/shimmy_kv_bench.tier",
    };
    
    const kv = try tiered_kv.TieredKVCache.init(allocator, config);
    defer kv.deinit();
    
    const kv_dim = config.n_heads * config.head_dim;
    const keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Fill with test data
    for (0..kv_dim) |i| {
        keys[i] = @as(f32, @floatFromInt(i)) * 0.001;
        values[i] = @as(f32, @floatFromInt(i)) * 0.002;
    }
    
    // Benchmark store (hot path)
    {
        const iterations: u32 = 1000;
        const start = std.time.nanoTimestamp();
        
        for (0..iterations) |_| {
            for (0..config.n_layers) |layer| {
                try kv.store(@intCast(layer), keys, values);
            }
            kv.advance();
        }
        
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const tokens_per_sec = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
        
        std.debug.print("   Store (hot): {d:.0} tokens/s\n", .{tokens_per_sec});
    }
    
    // Print stats
    kv.printStatus();
    
    // Clean up
    std.fs.cwd().deleteFile("/tmp/shimmy_kv_bench.tier") catch {};
}

fn benchmarkGGUF(allocator: std.mem.Allocator, path: []const u8) !void {
    std.debug.print("\n📂 GGUF Mmap Benchmark\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});
    
    const gguf = try mmap_gguf.MmapGGUF.open(allocator, path);
    defer gguf.close();
    
    gguf.printInfo();
    
    // List first 10 tensors
    std.debug.print("\n   First 10 tensors:\n", .{});
    const names = try gguf.listTensors(allocator);
    defer allocator.free(names);
    
    for (names[0..@min(10, names.len)]) |name| {
        const desc = gguf.getTensorDesc(name).?;
        std.debug.print("   - {s}: {d}x{d}x{d}x{d} ({s})\n", .{
            name, desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3],
            @tagName(desc.dtype),
        });
    }
    
    // Benchmark tensor access
    std.debug.print("\n   Tensor access benchmark:\n", .{});
    
    const iterations: u32 = 100;
    var total_bytes: u64 = 0;
    const start = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        for (names) |name| {
            const data = try gguf.getTensorData(name);
            total_bytes += data.len;
        }
    }
    
    const elapsed_ns = std.time.nanoTimestamp() - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const gb_per_sec = @as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000.0);
    
    std.debug.print("   Throughput: {d:.2} GB/s\n", .{gb_per_sec});
    std.debug.print("   (Note: First access triggers page faults, subsequent are cached)\n", .{});
}

