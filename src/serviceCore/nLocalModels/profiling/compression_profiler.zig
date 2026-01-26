// Compression Profiler - Advanced Compression Algorithm Performance Analysis
// Profiles DEFLATE, LZ4, Zstandard and other compression methods

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const CompressionAlgorithm = enum {
    deflate,
    lz4,
    zstd,
    snappy,
    brotli,
    gzip,

    pub fn toString(self: CompressionAlgorithm) []const u8 {
        return switch (self) {
            .deflate => "DEFLATE",
            .lz4 => "LZ4",
            .zstd => "Zstandard",
            .snappy => "Snappy",
            .brotli => "Brotli",
            .gzip => "GZIP",
        };
    }
};

pub const CompressionLevel = enum(u8) {
    fastest = 1,
    fast = 3,
    default = 6,
    best = 9,
    ultra = 22, // For Zstandard

    pub fn toInt(self: CompressionLevel) u8 {
        return @intFromEnum(self);
    }
};

pub const CompressionStats = struct {
    algorithm: CompressionAlgorithm,
    level: CompressionLevel,
    original_size: usize,
    compressed_size: usize,
    compression_time_ns: i64,
    decompression_time_ns: i64,
    compression_ratio: f64,
    throughput_mb_s: f64,
    timestamp_ns: i64,
};

pub const CompressionProfile = struct {
    stats: std.ArrayList(CompressionStats),
    
    // Aggregate metrics per algorithm
    algorithm_metrics: std.AutoHashMap(CompressionAlgorithm, AlgorithmMetrics),
    
    total_original_bytes: u64,
    total_compressed_bytes: u64,
    total_compression_time_ns: i64,
    total_decompression_time_ns: i64,
    
    allocator: Allocator,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: Allocator) CompressionProfile {
        return .{
            .stats = std.ArrayList(CompressionStats){},
            .algorithm_metrics = std.AutoHashMap(CompressionAlgorithm, AlgorithmMetrics).init(allocator),
            .total_original_bytes = 0,
            .total_compressed_bytes = 0,
            .total_compression_time_ns = 0,
            .total_decompression_time_ns = 0,
            .allocator = allocator,
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *CompressionProfile) void {
        self.stats.deinit();
        self.algorithm_metrics.deinit();
    }

    pub fn addStats(self: *CompressionProfile, stats: CompressionStats) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.stats.append(stats);

        // Update totals
        self.total_original_bytes += stats.original_size;
        self.total_compressed_bytes += stats.compressed_size;
        self.total_compression_time_ns += stats.compression_time_ns;
        self.total_decompression_time_ns += stats.decompression_time_ns;

        // Update algorithm metrics
        const entry = try self.algorithm_metrics.getOrPut(stats.algorithm);
        if (!entry.found_existing) {
            entry.value_ptr.* = AlgorithmMetrics.init();
        }
        entry.value_ptr.update(stats);

        // Limit history
        if (self.stats.items.len > 10000) {
            _ = self.stats.orderedRemove(0);
        }
    }

    pub fn getOverallCompressionRatio(self: *const CompressionProfile) f64 {
        if (self.total_original_bytes == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_original_bytes)) / 
               @as(f64, @floatFromInt(self.total_compressed_bytes));
    }

    pub fn getAverageThroughput(self: *const CompressionProfile) f64 {
        if (self.total_compression_time_ns == 0) return 0.0;
        const mb = @as(f64, @floatFromInt(self.total_original_bytes)) / 1_048_576.0;
        const seconds = @as(f64, @floatFromInt(self.total_compression_time_ns)) / 1_000_000_000.0;
        return mb / seconds;
    }

    pub fn getBestAlgorithm(self: *CompressionProfile, optimize_for: OptimizationGoal) ?CompressionAlgorithm {
        self.mutex.lock();
        defer self.mutex.unlock();

        var best_algo: ?CompressionAlgorithm = null;
        var best_score: f64 = 0.0;

        var iter = self.algorithm_metrics.iterator();
        while (iter.next()) |entry| {
            const score = switch (optimize_for) {
                .ratio => entry.value_ptr.avg_compression_ratio,
                .speed => entry.value_ptr.avg_throughput_mb_s,
                .balanced => entry.value_ptr.avg_compression_ratio * 
                            (entry.value_ptr.avg_throughput_mb_s / 100.0),
            };

            if (best_algo == null or score > best_score) {
                best_algo = entry.key_ptr.*;
                best_score = score;
            }
        }

        return best_algo;
    }

    pub fn toJson(self: *CompressionProfile, writer: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try writer.writeAll("{");
        try writer.print("\"total_original_mb\":{d:.2},", .{
            @as(f64, @floatFromInt(self.total_original_bytes)) / 1_048_576.0
        });
        try writer.print("\"total_compressed_mb\":{d:.2},", .{
            @as(f64, @floatFromInt(self.total_compressed_bytes)) / 1_048_576.0
        });
        try writer.print("\"overall_compression_ratio\":{d:.2},", .{self.getOverallCompressionRatio()});
        try writer.print("\"avg_throughput_mb_s\":{d:.2},", .{self.getAverageThroughput()});

        // Algorithm breakdown
        try writer.writeAll("\"algorithms\":{");
        var first = true;
        var iter = self.algorithm_metrics.iterator();
        while (iter.next()) |entry| {
            if (!first) try writer.writeAll(",");
            first = false;

            try writer.print("\"{s}\":{{", .{entry.key_ptr.toString()});
            try writer.print("\"count\":{d},", .{entry.value_ptr.count});
            try writer.print("\"avg_ratio\":{d:.2},", .{entry.value_ptr.avg_compression_ratio});
            try writer.print("\"avg_throughput_mb_s\":{d:.2},", .{entry.value_ptr.avg_throughput_mb_s});
            try writer.print("\"total_saved_mb\":{d:.2}", .{
                @as(f64, @floatFromInt(entry.value_ptr.total_bytes_saved)) / 1_048_576.0
            });
            try writer.writeAll("}");
        }
        try writer.writeAll("}");

        try writer.writeAll("}");
    }
};

const AlgorithmMetrics = struct {
    count: u64,
    total_original_bytes: u64,
    total_compressed_bytes: u64,
    total_bytes_saved: u64,
    total_compression_time_ns: i64,
    avg_compression_ratio: f64,
    avg_throughput_mb_s: f64,

    pub fn init() AlgorithmMetrics {
        return .{
            .count = 0,
            .total_original_bytes = 0,
            .total_compressed_bytes = 0,
            .total_bytes_saved = 0,
            .total_compression_time_ns = 0,
            .avg_compression_ratio = 0.0,
            .avg_throughput_mb_s = 0.0,
        };
    }

    pub fn update(self: *AlgorithmMetrics, stats: CompressionStats) void {
        self.count += 1;
        self.total_original_bytes += stats.original_size;
        self.total_compressed_bytes += stats.compressed_size;
        self.total_bytes_saved += stats.original_size - stats.compressed_size;
        self.total_compression_time_ns += stats.compression_time_ns;

        // Update averages
        self.avg_compression_ratio = @as(f64, @floatFromInt(self.total_original_bytes)) / 
                                     @as(f64, @floatFromInt(self.total_compressed_bytes));
        
        const mb = @as(f64, @floatFromInt(self.total_original_bytes)) / 1_048_576.0;
        const seconds = @as(f64, @floatFromInt(self.total_compression_time_ns)) / 1_000_000_000.0;
        self.avg_throughput_mb_s = mb / seconds;
    }
};

pub const OptimizationGoal = enum {
    ratio,    // Best compression ratio
    speed,    // Fastest compression
    balanced, // Balance of both
};

pub const CompressionProfiler = struct {
    profile: CompressionProfile,
    is_running: std.atomic.Value(bool),
    allocator: Allocator,

    pub fn init(allocator: Allocator) !*CompressionProfiler {
        const profiler = try allocator.create(CompressionProfiler);
        profiler.* = .{
            .profile = CompressionProfile.init(allocator),
            .is_running = std.atomic.Value(bool).init(false),
            .allocator = allocator,
        };
        return profiler;
    }

    pub fn deinit(self: *CompressionProfiler) void {
        self.profile.deinit();
        self.allocator.destroy(self);
    }

    pub fn start(self: *CompressionProfiler) void {
        self.is_running.store(true, .release);
    }

    pub fn stop(self: *CompressionProfiler) void {
        self.is_running.store(false, .release);
    }

    pub fn getProfile(self: *CompressionProfiler) *CompressionProfile {
        return &self.profile;
    }

    pub fn profileCompression(
        self: *CompressionProfiler,
        algorithm: CompressionAlgorithm,
        level: CompressionLevel,
        data: []const u8,
    ) !CompressionStats {
        if (!self.is_running.load(.acquire)) {
            return error.ProfilerNotRunning;
        }

        // Compress
        const comp_start = std.time.nanoTimestamp();
        const compressed = try self.compress(algorithm, level, data);
        const comp_duration = std.time.nanoTimestamp() - comp_start;
        defer self.allocator.free(compressed);

        // Decompress (for verification)
        const decomp_start = std.time.nanoTimestamp();
        const decompressed = try self.decompress(algorithm, compressed);
        const decomp_duration = std.time.nanoTimestamp() - decomp_start;
        defer self.allocator.free(decompressed);

        // Verify
        if (!std.mem.eql(u8, data, decompressed)) {
            return error.DecompressionMismatch;
        }

        const ratio = @as(f64, @floatFromInt(data.len)) / @as(f64, @floatFromInt(compressed.len));
        const mb = @as(f64, @floatFromInt(data.len)) / 1_048_576.0;
        const seconds = @as(f64, @floatFromInt(comp_duration)) / 1_000_000_000.0;
        const throughput = mb / seconds;

        const stats = CompressionStats{
            .algorithm = algorithm,
            .level = level,
            .original_size = data.len,
            .compressed_size = compressed.len,
            .compression_time_ns = comp_duration,
            .decompression_time_ns = decomp_duration,
            .compression_ratio = ratio,
            .throughput_mb_s = throughput,
            .timestamp_ns = std.time.nanoTimestamp(),
        };

        try self.profile.addStats(stats);

        return stats;
    }

    pub fn compareAlgorithms(
        self: *CompressionProfiler,
        data: []const u8,
        algorithms: []const CompressionAlgorithm,
    ) ![]CompressionStats {
        var results = try self.allocator.alloc(CompressionStats, algorithms.len);
        errdefer self.allocator.free(results);

        for (algorithms, 0..) |algo, i| {
            results[i] = try self.profileCompression(algo, .default, data);
        }

        return results;
    }

    pub fn findOptimalLevel(
        self: *CompressionProfiler,
        algorithm: CompressionAlgorithm,
        data: []const u8,
        goal: OptimizationGoal,
    ) !CompressionLevel {
        const levels = [_]CompressionLevel{ .fastest, .fast, .default, .best };
        
        var best_level: CompressionLevel = .default;
        var best_score: f64 = 0.0;

        for (levels) |level| {
            const stats = try self.profileCompression(algorithm, level, data);
            
            const score = switch (goal) {
                .ratio => stats.compression_ratio,
                .speed => stats.throughput_mb_s,
                .balanced => stats.compression_ratio * (stats.throughput_mb_s / 100.0),
            };

            if (score > best_score) {
                best_score = score;
                best_level = level;
            }
        }

        return best_level;
    }

    pub fn generateRecommendation(self: *CompressionProfiler, use_case: UseCase) ![]const u8 {
        const profile = self.getProfile();

        return switch (use_case) {
            .kv_cache => try self.allocator.dupe(u8, 
                "For KV cache: Use LZ4 (fastest=1) for speed, or Zstandard (level=3) for balanced performance. " ++
                "Avoid DEFLATE (too slow). Consider Snappy for minimal overhead."
            ),
            .model_weights => try self.allocator.dupe(u8,
                "For model weights: Use Zstandard (level=6-9) for best ratio with good speed. " ++
                "For deployment: DEFLATE (level=9) for smallest size. " ++
                "For training: LZ4 for fastest checkpointing."
            ),
            .activations => try self.allocator.dupe(u8,
                "For activations: Use LZ4 (fastest=1) for real-time processing. " ++
                "Consider no compression if latency is critical. " ++
                "Use Snappy for balanced overhead."
            ),
            .embeddings => try self.allocator.dupe(u8,
                "For embeddings: Use Zstandard (level=3-6) for good ratio. " ++
                "For vector databases: Consider dimensionality reduction before compression. " ++
                "LZ4 for fast retrieval."
            ),
        };
    }

    // Compression implementations (simplified - use real libraries in production)
    fn compress(self: *CompressionProfiler, algo: CompressionAlgorithm, level: CompressionLevel, data: []const u8) ![]u8 {
        _ = level;
        
        return switch (algo) {
            .deflate => try self.compressDeflate(data),
            .lz4 => try self.compressLZ4(data),
            .zstd => try self.compressZstd(data),
            .snappy, .brotli, .gzip => try self.compressGeneric(data),
        };
    }

    fn decompress(self: *CompressionProfiler, algo: CompressionAlgorithm, data: []const u8) ![]u8 {
        return switch (algo) {
            .deflate => try self.decompressDeflate(data),
            .lz4 => try self.decompressLZ4(data),
            .zstd => try self.decompressZstd(data),
            .snappy, .brotli, .gzip => try self.decompressGeneric(data),
        };
    }

    fn compressDeflate(self: *CompressionProfiler, data: []const u8) ![]u8 {
        // Use std.compress.deflate when available
        // For now, simulate with simple compression
        return try self.simulateCompression(data, 0.65); // ~65% of original
    }

    fn compressLZ4(self: *CompressionProfiler, data: []const u8) ![]u8 {
        // Use LZ4 library bindings
        return try self.simulateCompression(data, 0.75); // ~75% (faster, less compression)
    }

    fn compressZstd(self: *CompressionProfiler, data: []const u8) ![]u8 {
        // Use Zstandard library bindings
        return try self.simulateCompression(data, 0.60); // ~60% (good ratio)
    }

    fn compressGeneric(self: *CompressionProfiler, data: []const u8) ![]u8 {
        return try self.simulateCompression(data, 0.70);
    }

    fn decompressDeflate(self: *CompressionProfiler, data: []const u8) ![]u8 {
        return try self.simulateDecompression(data);
    }

    fn decompressLZ4(self: *CompressionProfiler, data: []const u8) ![]u8 {
        return try self.simulateDecompression(data);
    }

    fn decompressZstd(self: *CompressionProfiler, data: []const u8) ![]u8 {
        return try self.simulateDecompression(data);
    }

    fn decompressGeneric(self: *CompressionProfiler, data: []const u8) ![]u8 {
        return try self.simulateDecompression(data);
    }

    // Simulation helpers (replace with real compression in production)
    fn simulateCompression(self: *CompressionProfiler, data: []const u8, ratio: f64) ![]u8 {
        const compressed_size = @as(usize, @intFromFloat(@as(f64, @floatFromInt(data.len)) * ratio));
        const result = try self.allocator.alloc(u8, compressed_size);
        @memset(result, 0xFF);
        return result;
    }

    fn simulateDecompression(self: *CompressionProfiler, data: []const u8) ![]u8 {
        const original_size = @as(usize, @intFromFloat(@as(f64, @floatFromInt(data.len)) / 0.65));
        const result = try self.allocator.alloc(u8, original_size);
        @memset(result, 0xAA);
        return result;
    }
};

pub const UseCase = enum {
    kv_cache,
    model_weights,
    activations,
    embeddings,
};

// Testing
test "CompressionProfiler basic" {
    const allocator = std.testing.allocator;

    var profiler = try CompressionProfiler.init(allocator);
    defer profiler.deinit();

    profiler.start();

    // Create test data
    var test_data = try allocator.alloc(u8, 1024);
    defer allocator.free(test_data);
    @memset(test_data, 0x42);

    // Profile compression
    const stats = try profiler.profileCompression(.lz4, .default, test_data);
    
    try std.testing.expect(stats.original_size == 1024);
    try std.testing.expect(stats.compressed_size < stats.original_size);
    try std.testing.expect(stats.compression_ratio > 1.0);
}
