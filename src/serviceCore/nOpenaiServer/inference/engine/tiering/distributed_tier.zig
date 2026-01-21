// Distributed Tiering with DragonflyDB
// Enables multi-node KV cache sharing and SSD-backed distributed storage
//
// Architecture:
// - Local hot tier (RAM)
// - Distributed warm tier (DragonflyDB with tiering)
// - Local cold tier (SSD)
//
// This enables:
// - Sharing KV cache across inference nodes
// - Prompt caching for repeated queries
// - Session state persistence
// - Horizontal scaling of context length

const std = @import("std");
const dragonfly = @import("../../../integrations/cache/dragonfly/dragonfly_client.zig");

// ============================================================================
// Configuration
// ============================================================================

pub const DistributedConfig = struct {
    // DragonflyDB connection
    dragonfly_host: []const u8 = "127.0.0.1",
    dragonfly_port: u16 = 6379,
    dragonfly_pool_size: u32 = 8,
    
    // Key prefixes
    kv_cache_prefix: []const u8 = "shimmy:kv:",
    tensor_prefix: []const u8 = "shimmy:tensor:",
    session_prefix: []const u8 = "shimmy:session:",
    
    // TTL settings (seconds)
    kv_cache_ttl: u32 = 3600,      // 1 hour
    tensor_ttl: u32 = 86400,       // 24 hours
    session_ttl: u32 = 1800,       // 30 minutes
    
    // Compression
    compress_threshold: u32 = 1024, // Compress values > 1KB
    
    // Batching
    batch_size: u32 = 100,         // Max keys per MGET/MSET
};

// ============================================================================
// Distributed KV Cache Tier
// ============================================================================

pub const DistributedKVTier = struct {
    allocator: std.mem.Allocator,
    config: DistributedConfig,
    client: *dragonfly.DragonflyClient,
    
    // Statistics
    stats: Stats,
    
    pub const Stats = struct {
        gets: u64 = 0,
        sets: u64 = 0,
        hits: u64 = 0,
        misses: u64 = 0,
        bytes_sent: u64 = 0,
        bytes_received: u64 = 0,
        latency_sum_us: u64 = 0,
        latency_count: u64 = 0,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: DistributedConfig) !*DistributedKVTier {
        std.debug.print("\n🌐 Initializing Distributed KV Tier\n", .{});
        std.debug.print("   DragonflyDB: {s}:{d}\n", .{config.dragonfly_host, config.dragonfly_port});
        
        const self = try allocator.create(DistributedKVTier);
        errdefer allocator.destroy(self);
        
        // Initialize DragonflyDB client
        const client = try dragonfly.DragonflyClient.init(
            allocator,
            config.dragonfly_host,
            config.dragonfly_port,
            config.dragonfly_pool_size,
        );
        errdefer client.deinit();
        
        self.* = DistributedKVTier{
            .allocator = allocator,
            .config = config,
            .client = client,
            .stats = .{},
        };
        
        // Test connection
        const pong = try client.ping();
        if (!std.mem.eql(u8, pong, "PONG")) {
            return error.ConnectionFailed;
        }
        
        std.debug.print("   ✅ Connected to DragonflyDB\n", .{});
        return self;
    }
    
    /// Store KV cache block for a session
    pub fn storeKVBlock(
        self: *DistributedKVTier,
        session_id: []const u8,
        layer: u32,
        start_pos: u32,
        end_pos: u32,
        keys: []const f32,
        values: []const f32,
    ) !void {
        const start_time = std.time.microTimestamp();
        
        // Build key: shimmy:kv:<session>:<layer>:<start>-<end>
        var key_buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}{s}:{d}:{d}-{d}", .{
            self.config.kv_cache_prefix,
            session_id,
            layer,
            start_pos,
            end_pos,
        });
        
        // Serialize keys and values together
        const keys_bytes = std.mem.sliceAsBytes(keys);
        const values_bytes = std.mem.sliceAsBytes(values);
        
        // Combine into single buffer
        const total_size = keys_bytes.len + values_bytes.len;
        const buffer = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(buffer);
        
        @memcpy(buffer[0..keys_bytes.len], keys_bytes);
        @memcpy(buffer[keys_bytes.len..], values_bytes);
        
        // Store with TTL
        try self.client.setex(key, buffer, self.config.kv_cache_ttl);
        
        self.stats.sets += 1;
        self.stats.bytes_sent += total_size;
        self.stats.latency_sum_us += @intCast(std.time.microTimestamp() - start_time);
        self.stats.latency_count += 1;
    }
    
    /// Load KV cache block for a session
    pub fn loadKVBlock(
        self: *DistributedKVTier,
        session_id: []const u8,
        layer: u32,
        start_pos: u32,
        end_pos: u32,
        keys_out: []f32,
        values_out: []f32,
    ) !bool {
        const start_time = std.time.microTimestamp();
        
        var key_buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}{s}:{d}:{d}-{d}", .{
            self.config.kv_cache_prefix,
            session_id,
            layer,
            start_pos,
            end_pos,
        });
        
        self.stats.gets += 1;
        
        const data = self.client.get(key) catch |err| {
            if (err == error.KeyNotFound) {
                self.stats.misses += 1;
                return false;
            }
            return err;
        };
        defer self.allocator.free(data);

        // Split data into keys and values
        const keys_bytes = std.mem.sliceAsBytes(keys_out);
        const values_bytes = std.mem.sliceAsBytes(values_out);

        if (data.len != keys_bytes.len + values_bytes.len) {
            return error.DataSizeMismatch;
        }

        @memcpy(keys_bytes, data[0..keys_bytes.len]);
        @memcpy(values_bytes, data[keys_bytes.len..]);

        self.stats.hits += 1;
        self.stats.bytes_received += data.len;
        self.stats.latency_sum_us += @intCast(std.time.microTimestamp() - start_time);
        self.stats.latency_count += 1;

        return true;
    }

    /// Store prompt embedding for caching
    pub fn storePromptCache(
        self: *DistributedKVTier,
        prompt_hash: []const u8,
        kv_state: []const u8,
    ) !void {
        var key_buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}prompt:{s}", .{
            self.config.kv_cache_prefix,
            prompt_hash,
        });

        try self.client.setex(key, kv_state, self.config.kv_cache_ttl);
        self.stats.sets += 1;
        self.stats.bytes_sent += kv_state.len;
    }

    /// Load cached prompt state
    pub fn loadPromptCache(
        self: *DistributedKVTier,
        prompt_hash: []const u8,
    ) !?[]const u8 {
        var key_buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}prompt:{s}", .{
            self.config.kv_cache_prefix,
            prompt_hash,
        });

        self.stats.gets += 1;

        const data = self.client.get(key) catch |err| {
            if (err == error.KeyNotFound) {
                self.stats.misses += 1;
                return null;
            }
            return err;
        };

        self.stats.hits += 1;
        self.stats.bytes_received += data.len;
        return data;
    }

    /// Store session state
    pub fn storeSession(
        self: *DistributedKVTier,
        session_id: []const u8,
        state: []const u8,
    ) !void {
        var key_buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}{s}", .{
            self.config.session_prefix,
            session_id,
        });

        try self.client.setex(key, state, self.config.session_ttl);
    }

    /// Load session state
    pub fn loadSession(
        self: *DistributedKVTier,
        session_id: []const u8,
    ) !?[]const u8 {
        var key_buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}{s}", .{
            self.config.session_prefix,
            session_id,
        });

        return self.client.get(key) catch |err| {
            if (err == error.KeyNotFound) return null;
            return err;
        };
    }

    /// Delete session and all associated KV cache
    pub fn deleteSession(self: *DistributedKVTier, session_id: []const u8) !void {
        // Delete session state
        var key_buf: [256]u8 = undefined;
        const session_key = try std.fmt.bufPrint(&key_buf, "{s}{s}", .{
            self.config.session_prefix,
            session_id,
        });
        _ = self.client.del(session_key) catch {};

        // Delete KV cache entries (would need SCAN in production)
        // For now, rely on TTL expiration
    }

    /// Get average latency in microseconds
    pub fn getAvgLatencyUs(self: *DistributedKVTier) u64 {
        if (self.stats.latency_count == 0) return 0;
        return self.stats.latency_sum_us / self.stats.latency_count;
    }

    /// Get hit rate
    pub fn getHitRate(self: *DistributedKVTier) f32 {
        const total = self.stats.hits + self.stats.misses;
        if (total == 0) return 0;
        return @as(f32, @floatFromInt(self.stats.hits)) / @as(f32, @floatFromInt(total));
    }

    /// Print status
    pub fn printStatus(self: *DistributedKVTier) void {
        std.debug.print("\n📊 Distributed KV Tier Status\n", .{});
        std.debug.print("   Gets: {d}, Sets: {d}\n", .{self.stats.gets, self.stats.sets});
        std.debug.print("   Hits: {d}, Misses: {d} ({d:.1}% hit rate)\n", .{
            self.stats.hits, self.stats.misses, self.getHitRate() * 100,
        });
        std.debug.print("   Bytes sent: {d:.1} MB, received: {d:.1} MB\n", .{
            @as(f64, @floatFromInt(self.stats.bytes_sent)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.stats.bytes_received)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Avg latency: {d} µs\n", .{self.getAvgLatencyUs()});
    }

    pub fn deinit(self: *DistributedKVTier) void {
        self.client.deinit();
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Prompt Hash Utility
// ============================================================================

pub fn hashPrompt(prompt: []const u8) [32]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(prompt, &hash, .{});
    return hash;
}

pub fn hashToHex(hash: [32]u8) [64]u8 {
    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{std.fmt.fmtSliceHexLower(&hash)}) catch unreachable;
    return hex;
}

