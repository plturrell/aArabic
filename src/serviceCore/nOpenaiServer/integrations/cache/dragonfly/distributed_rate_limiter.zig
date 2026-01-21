// Distributed Rate Limiter using DragonflyDB
// Implements sliding window rate limiting with cross-instance coordination
//
// Features:
// - Atomic increment and check using Redis INCR + EXPIRE
// - Per-client rate limiting (by API key, IP, etc.)
// - Burst allowance (token bucket style)
// - Graceful degradation if Dragonfly unavailable
// - HTTP header generation for rate limit responses
// - C ABI for Mojo integration

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const dragonfly = @import("dragonfly_client.zig");
const DragonflyClient = dragonfly.DragonflyClient;

/// Rate limit information returned by checkWithInfo
pub const RateLimitInfo = struct {
    allowed: bool,
    current_count: u32,
    limit: u32,
    remaining: u32,
    reset_at: i64, // Unix timestamp (ms) when window resets

    /// Generate X-RateLimit-Limit header value
    pub fn getLimitHeader(self: RateLimitInfo, buf: []u8) ![]u8 {
        return std.fmt.bufPrint(buf, "{d}", .{self.limit});
    }

    /// Generate X-RateLimit-Remaining header value
    pub fn getRemainingHeader(self: RateLimitInfo, buf: []u8) ![]u8 {
        return std.fmt.bufPrint(buf, "{d}", .{self.remaining});
    }

    /// Generate X-RateLimit-Reset header value (Unix timestamp in seconds)
    pub fn getResetHeader(self: RateLimitInfo, buf: []u8) ![]u8 {
        const reset_seconds = @divFloor(self.reset_at, 1000);
        return std.fmt.bufPrint(buf, "{d}", .{reset_seconds});
    }
};

/// HTTP headers for rate limiting
pub const RateLimitHeaders = struct {
    limit: [32]u8,
    limit_len: usize,
    remaining: [32]u8,
    remaining_len: usize,
    reset: [32]u8,
    reset_len: usize,

    pub fn fromInfo(info: RateLimitInfo) RateLimitHeaders {
        var headers = RateLimitHeaders{
            .limit = undefined,
            .limit_len = 0,
            .remaining = undefined,
            .remaining_len = 0,
            .reset = undefined,
            .reset_len = 0,
        };

        if (info.getLimitHeader(&headers.limit)) |slice| {
            headers.limit_len = slice.len;
        } else |_| {}

        if (info.getRemainingHeader(&headers.remaining)) |slice| {
            headers.remaining_len = slice.len;
        } else |_| {}

        if (info.getResetHeader(&headers.reset)) |slice| {
            headers.reset_len = slice.len;
        } else |_| {}

        return headers;
    }
};

/// Configuration for the rate limiter
pub const Config = struct {
    key_prefix: []const u8 = "rl",
    requests_per_second: u32 = 10,
    burst_size: u32 = 20,
    window_ms: u32 = 1000, // Default 1 second window
    fallback_to_local: bool = true, // Allow requests if Dragonfly unavailable
};

/// Metrics for monitoring rate limiting behavior
pub const Metrics = struct {
    requests_allowed: u64 = 0,
    requests_denied: u64 = 0,
    fallback_activations: u64 = 0,
    dragonfly_errors: u64 = 0,

    pub fn allowRate(self: *const Metrics) f64 {
        const total = self.requests_allowed + self.requests_denied;
        if (total == 0) return 1.0;
        return @as(f64, @floatFromInt(self.requests_allowed)) / @as(f64, @floatFromInt(total));
    }
};

/// Local fallback rate limiter (simple in-memory counter)
const LocalRateLimiter = struct {
    counts: std.StringHashMap(LocalEntry),
    allocator: Allocator,

    const LocalEntry = struct {
        count: u32,
        window_start: i64,
    };

    pub fn init(allocator: Allocator) LocalRateLimiter {
        return LocalRateLimiter{
            .counts = std.StringHashMap(LocalEntry).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LocalRateLimiter) void {
        var iter = self.counts.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.counts.deinit();
    }

    pub fn checkAndIncrement(self: *LocalRateLimiter, key: []const u8, limit: u32, window_ms: u32) !struct { allowed: bool, count: u32, window_start: i64 } {
        const now = std.time.milliTimestamp();
        const window_start = @divFloor(now, @as(i64, window_ms)) * @as(i64, window_ms);

        if (self.counts.get(key)) |entry| {
            if (entry.window_start == window_start) {
                // Same window, check limit
                if (entry.count >= limit) {
                    return .{ .allowed = false, .count = entry.count, .window_start = window_start };
                }
                // Increment
                const new_count = entry.count + 1;
                try self.counts.put(key, .{ .count = new_count, .window_start = window_start });
                return .{ .allowed = true, .count = new_count, .window_start = window_start };
            }
        }

        // New window or new key
        const key_copy = try self.allocator.dupe(u8, key);
        try self.counts.put(key_copy, .{ .count = 1, .window_start = window_start });
        return .{ .allowed = true, .count = 1, .window_start = window_start };
    }
};

/// Distributed Rate Limiter using DragonflyDB
pub const DistributedRateLimiter = struct {
    client: *DragonflyClient,
    allocator: Allocator,
    key_prefix: []const u8,
    requests_per_second: u32,
    burst_size: u32,
    window_ms: u32,
    fallback_to_local: bool,
    local_limiter: LocalRateLimiter,
    metrics: Metrics,

    /// Initialize the rate limiter with given configuration
    pub fn init(allocator: Allocator, client: *DragonflyClient, config: Config) DistributedRateLimiter {
        return DistributedRateLimiter{
            .client = client,
            .allocator = allocator,
            .key_prefix = config.key_prefix,
            .requests_per_second = config.requests_per_second,
            .burst_size = config.burst_size,
            .window_ms = config.window_ms,
            .fallback_to_local = config.fallback_to_local,
            .local_limiter = LocalRateLimiter.init(allocator),
            .metrics = Metrics{},
        };
    }

    pub fn deinit(self: *DistributedRateLimiter) void {
        self.local_limiter.deinit();
    }

    /// Build the Redis key for a given client ID and window
    fn buildKey(self: *DistributedRateLimiter, client_id: []const u8, window: i64, buf: []u8) ![]u8 {
        return std.fmt.bufPrint(buf, "ratelimit:{s}:{s}:{d}", .{ self.key_prefix, client_id, window });
    }

    /// Get current window identifier
    fn getCurrentWindow(self: *DistributedRateLimiter) i64 {
        const now = std.time.milliTimestamp();
        return @divFloor(now, @as(i64, self.window_ms));
    }

    /// Calculate effective limit (base rate + burst)
    fn getEffectiveLimit(self: *DistributedRateLimiter) u32 {
        return self.requests_per_second + self.burst_size;
    }

    /// Check if a request should be allowed (simple bool result)
    pub fn check(self: *DistributedRateLimiter, client_id: []const u8) !bool {
        const info = try self.checkWithInfo(client_id);
        return info.allowed;
    }

    /// Check rate limit and return detailed information
    pub fn checkWithInfo(self: *DistributedRateLimiter, client_id: []const u8) !RateLimitInfo {
        const window = self.getCurrentWindow();
        const limit = self.getEffectiveLimit();
        const window_ms_i64: i64 = @intCast(self.window_ms);
        const reset_at = (window + 1) * window_ms_i64;

        // Build the key
        var key_buf: [256]u8 = undefined;
        const key = try self.buildKey(client_id, window, &key_buf);

        // Try distributed check first
        const result = self.distributedCheckAndIncrement(key, limit) catch |err| {
            // Dragonfly unavailable, use fallback if configured
            self.metrics.dragonfly_errors += 1;

            if (self.fallback_to_local) {
                self.metrics.fallback_activations += 1;
                const local_result = try self.local_limiter.checkAndIncrement(
                    client_id,
                    limit,
                    self.window_ms,
                );
                const remaining = if (local_result.count >= limit) 0 else limit - local_result.count;
                if (local_result.allowed) {
                    self.metrics.requests_allowed += 1;
                } else {
                    self.metrics.requests_denied += 1;
                }
                return RateLimitInfo{
                    .allowed = local_result.allowed,
                    .current_count = local_result.count,
                    .limit = limit,
                    .remaining = remaining,
                    .reset_at = reset_at,
                };
            }
            return err;
        };

        const remaining = if (result.count >= limit) 0 else limit - result.count;
        if (result.allowed) {
            self.metrics.requests_allowed += 1;
        } else {
            self.metrics.requests_denied += 1;
        }

        return RateLimitInfo{
            .allowed = result.allowed,
            .current_count = result.count,
            .limit = limit,
            .remaining = remaining,
            .reset_at = reset_at,
        };
    }

    /// Perform atomic check and increment using Redis INCR + EXPIRE
    fn distributedCheckAndIncrement(self: *DistributedRateLimiter, key: []const u8, limit: u32) !struct { allowed: bool, count: u32 } {
        // Use INCR to atomically increment the counter
        const count_i64 = try self.client.incr(key);
        const count: u32 = @intCast(count_i64);

        // If this is the first request in the window, set expiry
        if (count == 1) {
            const ttl_seconds: u32 = @max(1, self.window_ms / 1000 + 1);
            _ = try self.client.expire(key, ttl_seconds);
        }

        return .{
            .allowed = count <= limit,
            .count = count,
        };
    }

    /// Reset rate limit for a specific client
    pub fn reset(self: *DistributedRateLimiter, client_id: []const u8) !void {
        const window = self.getCurrentWindow();
        var key_buf: [256]u8 = undefined;
        const key = try self.buildKey(client_id, window, &key_buf);

        const keys = [_][]const u8{key};
        _ = try self.client.del(&keys);
    }

    /// Get current metrics
    pub fn getMetrics(self: *DistributedRateLimiter) Metrics {
        return self.metrics;
    }

    /// Reset metrics
    pub fn resetMetrics(self: *DistributedRateLimiter) void {
        self.metrics = Metrics{};
    }
};

// ============================================================================
// C ABI Exports for Mojo Integration
// ============================================================================

const CLimiter = opaque {};

/// C-compatible rate limit info structure
pub const CRateLimitInfo = extern struct {
    allowed: bool,
    current_count: u32,
    limit: u32,
    remaining: u32,
    reset_at: i64,
};

/// Global allocator for C ABI
var global_allocator: Allocator = std.heap.c_allocator;

/// Create a new distributed rate limiter
export fn distributed_rl_create(
    host: [*:0]const u8,
    port: u16,
    requests_per_second: u32,
    burst_size: u32,
    prefix: [*:0]const u8,
) callconv(.c) ?*CLimiter {
    const host_slice = mem.span(host);
    const prefix_slice = mem.span(prefix);

    const client = DragonflyClient.init(global_allocator, host_slice, port) catch return null;

    const limiter = global_allocator.create(DistributedRateLimiter) catch {
        client.deinit();
        return null;
    };

    limiter.* = DistributedRateLimiter.init(global_allocator, client, .{
        .key_prefix = prefix_slice,
        .requests_per_second = requests_per_second,
        .burst_size = burst_size,
        .window_ms = 1000,
        .fallback_to_local = true,
    });

    return @ptrCast(limiter);
}

/// Check if request is allowed (returns true if allowed)
export fn distributed_rl_check(
    limiter: *CLimiter,
    client_id: [*]const u8,
    id_len: usize,
) callconv(.c) bool {
    const real_limiter: *DistributedRateLimiter = @ptrCast(@alignCast(limiter));
    const id_slice = client_id[0..id_len];

    return real_limiter.check(id_slice) catch false;
}

/// Get detailed rate limit info
export fn distributed_rl_get_info(
    limiter: *CLimiter,
    client_id: [*]const u8,
    id_len: usize,
    info_out: *CRateLimitInfo,
) callconv(.c) bool {
    const real_limiter: *DistributedRateLimiter = @ptrCast(@alignCast(limiter));
    const id_slice = client_id[0..id_len];

    const info = real_limiter.checkWithInfo(id_slice) catch return false;

    info_out.* = CRateLimitInfo{
        .allowed = info.allowed,
        .current_count = info.current_count,
        .limit = info.limit,
        .remaining = info.remaining,
        .reset_at = info.reset_at,
    };

    return true;
}

/// Reset rate limit for a client
export fn distributed_rl_reset(
    limiter: *CLimiter,
    client_id: [*]const u8,
    id_len: usize,
) callconv(.c) void {
    const real_limiter: *DistributedRateLimiter = @ptrCast(@alignCast(limiter));
    const id_slice = client_id[0..id_len];

    real_limiter.reset(id_slice) catch {};
}

/// Destroy the rate limiter
export fn distributed_rl_destroy(limiter: *CLimiter) callconv(.c) void {
    const real_limiter: *DistributedRateLimiter = @ptrCast(@alignCast(limiter));
    const client = real_limiter.client;

    real_limiter.deinit();
    global_allocator.destroy(real_limiter);
    client.deinit();
}

/// Get metrics - requests allowed count
export fn distributed_rl_get_allowed_count(limiter: *CLimiter) callconv(.c) u64 {
    const real_limiter: *DistributedRateLimiter = @ptrCast(@alignCast(limiter));
    return real_limiter.metrics.requests_allowed;
}

/// Get metrics - requests denied count
export fn distributed_rl_get_denied_count(limiter: *CLimiter) callconv(.c) u64 {
    const real_limiter: *DistributedRateLimiter = @ptrCast(@alignCast(limiter));
    return real_limiter.metrics.requests_denied;
}

// ============================================================================
// Tests
// ============================================================================

test "RateLimitInfo header generation" {
    const info = RateLimitInfo{
        .allowed = true,
        .current_count = 5,
        .limit = 100,
        .remaining = 95,
        .reset_at = 1705500000000, // Example timestamp
    };

    var limit_buf: [32]u8 = undefined;
    const limit_str = try info.getLimitHeader(&limit_buf);
    try std.testing.expectEqualStrings("100", limit_str);

    var remaining_buf: [32]u8 = undefined;
    const remaining_str = try info.getRemainingHeader(&remaining_buf);
    try std.testing.expectEqualStrings("95", remaining_str);

    var reset_buf: [32]u8 = undefined;
    const reset_str = try info.getResetHeader(&reset_buf);
    try std.testing.expectEqualStrings("1705500000", reset_str);
}

test "RateLimitHeaders fromInfo" {
    const info = RateLimitInfo{
        .allowed = false,
        .current_count = 100,
        .limit = 100,
        .remaining = 0,
        .reset_at = 1705500000000,
    };

    const headers = RateLimitHeaders.fromInfo(info);

    try std.testing.expect(headers.limit_len > 0);
    try std.testing.expect(headers.remaining_len > 0);
    try std.testing.expect(headers.reset_len > 0);
}

test "Config defaults" {
    const config = Config{};

    try std.testing.expectEqual(@as(u32, 10), config.requests_per_second);
    try std.testing.expectEqual(@as(u32, 20), config.burst_size);
    try std.testing.expectEqual(@as(u32, 1000), config.window_ms);
    try std.testing.expect(config.fallback_to_local);
}

test "Metrics calculation" {
    var metrics = Metrics{
        .requests_allowed = 80,
        .requests_denied = 20,
        .fallback_activations = 0,
        .dragonfly_errors = 0,
    };

    const rate = metrics.allowRate();
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), rate, 0.001);
}

test "Metrics empty rate" {
    const metrics = Metrics{};
    const rate = metrics.allowRate();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rate, 0.001);
}

test "LocalRateLimiter basic functionality" {
    const allocator = std.testing.allocator;
    var local = LocalRateLimiter.init(allocator);
    defer local.deinit();

    // First request should be allowed
    const result1 = try local.checkAndIncrement("client1", 5, 1000);
    try std.testing.expect(result1.allowed);
    try std.testing.expectEqual(@as(u32, 1), result1.count);

    // Subsequent requests should increment
    const result2 = try local.checkAndIncrement("client1", 5, 1000);
    try std.testing.expect(result2.allowed);
    try std.testing.expectEqual(@as(u32, 2), result2.count);
}

test "LocalRateLimiter limit enforcement" {
    const allocator = std.testing.allocator;
    var local = LocalRateLimiter.init(allocator);
    defer local.deinit();

    // Fill up to limit
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        const result = try local.checkAndIncrement("client2", 5, 60000);
        try std.testing.expect(result.allowed);
    }

    // Next request should be denied
    const denied = try local.checkAndIncrement("client2", 5, 60000);
    try std.testing.expect(!denied.allowed);
    try std.testing.expectEqual(@as(u32, 5), denied.count);
}

test "LocalRateLimiter multiple clients" {
    const allocator = std.testing.allocator;
    var local = LocalRateLimiter.init(allocator);
    defer local.deinit();

    // Different clients should have separate limits
    const result1 = try local.checkAndIncrement("clientA", 2, 60000);
    const result2 = try local.checkAndIncrement("clientB", 2, 60000);

    try std.testing.expect(result1.allowed);
    try std.testing.expect(result2.allowed);
    try std.testing.expectEqual(@as(u32, 1), result1.count);
    try std.testing.expectEqual(@as(u32, 1), result2.count);
}
