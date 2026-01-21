// Utility Components for nWorkflow
// Day 29: Langflow Component Parity (Part 2/3)
// Implements: RateLimiterNode, QueueNode, BatchNode, ThrottleNode

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// ============================================================================
// Rate Limiter Node - Control request rates
// ============================================================================

pub const RateLimitStrategy = enum {
    fixed_window,
    sliding_window,
    token_bucket,
    leaky_bucket,
};

pub const RateLimiterNode = struct {
    allocator: Allocator,
    max_requests: u32,
    window_ms: u64,
    strategy: RateLimitStrategy,
    request_times: ArrayList(i64),
    tokens: u32,
    last_refill: i64,
    
    pub fn init(
        allocator: Allocator,
        max_requests: u32,
        window_ms: u64,
        strategy: RateLimitStrategy,
    ) !RateLimiterNode {
        return RateLimiterNode{
            .allocator = allocator,
            .max_requests = max_requests,
            .window_ms = window_ms,
            .strategy = strategy,
            .request_times = ArrayList(i64){},
            .tokens = max_requests,
            .last_refill = std.time.milliTimestamp(),
        };
    }
    
    pub fn deinit(self: *RateLimiterNode) void {
        self.request_times.deinit(self.allocator);
    }
    
    pub fn allowRequest(self: *RateLimiterNode) !bool {
        const now = std.time.milliTimestamp();
        
        return switch (self.strategy) {
            .fixed_window => try self.allowFixedWindow(now),
            .sliding_window => try self.allowSlidingWindow(now),
            .token_bucket => try self.allowTokenBucket(now),
            .leaky_bucket => try self.allowLeakyBucket(now),
        };
    }
    
    fn allowFixedWindow(self: *RateLimiterNode, now: i64) !bool {
        const window_start = @divFloor(now, @as(i64, @intCast(self.window_ms))) * @as(i64, @intCast(self.window_ms));
        
        // Remove requests outside current window
        var i: usize = 0;
        while (i < self.request_times.items.len) {
            if (self.request_times.items[i] < window_start) {
                _ = self.request_times.orderedRemove(i);
            } else {
                i += 1;
            }
        }
        
        if (self.request_times.items.len < self.max_requests) {
            try self.request_times.append(self.allocator, now);
            return true;
        }
        
        return false;
    }
    
    fn allowSlidingWindow(self: *RateLimiterNode, now: i64) !bool {
        const window_start = now - @as(i64, @intCast(self.window_ms));
        
        // Remove old requests
        var i: usize = 0;
        while (i < self.request_times.items.len) {
            if (self.request_times.items[i] < window_start) {
                _ = self.request_times.orderedRemove(i);
            } else {
                i += 1;
            }
        }
        
        if (self.request_times.items.len < self.max_requests) {
            try self.request_times.append(self.allocator, now);
            return true;
        }
        
        return false;
    }
    
    fn allowTokenBucket(self: *RateLimiterNode, now: i64) !bool {
        // Refill tokens based on time elapsed
        const elapsed = now - self.last_refill;
        const refill_rate = @as(f64, @floatFromInt(self.max_requests)) / @as(f64, @floatFromInt(self.window_ms));
        const tokens_to_add = @as(u32, @intFromFloat(@as(f64, @floatFromInt(elapsed)) * refill_rate));
        
        if (tokens_to_add > 0) {
            self.tokens = @min(self.tokens + tokens_to_add, self.max_requests);
            self.last_refill = now;
        }
        
        if (self.tokens > 0) {
            self.tokens -= 1;
            return true;
        }
        
        return false;
    }
    
    fn allowLeakyBucket(self: *RateLimiterNode, now: i64) !bool {
        // Similar to token bucket but with constant leak rate
        const elapsed = now - self.last_refill;
        const leak_rate = @as(f64, @floatFromInt(self.max_requests)) / @as(f64, @floatFromInt(self.window_ms));
        const leaked = @as(u32, @intFromFloat(@as(f64, @floatFromInt(elapsed)) * leak_rate));
        
        if (leaked > 0) {
            if (self.tokens >= leaked) {
                self.tokens -= leaked;
            } else {
                self.tokens = 0;
            }
            self.last_refill = now;
        }
        
        if (self.tokens < self.max_requests) {
            self.tokens += 1;
            return true;
        }
        
        return false;
    }
    
    pub fn reset(self: *RateLimiterNode) void {
        self.request_times.clearRetainingCapacity();
        self.tokens = self.max_requests;
        self.last_refill = std.time.milliTimestamp();
    }
};

// ============================================================================
// Queue Node - Message queuing with FIFO/LIFO/Priority
// ============================================================================

pub const QueueType = enum {
    fifo,
    lifo,
    priority,
};

pub const QueueItem = struct {
    data: []const u8,
    priority: i32,
    timestamp: i64,
    
    pub fn lessThan(_: void, a: QueueItem, b: QueueItem) bool {
        return a.priority > b.priority; // Higher priority first
    }
};

pub const QueueNode = struct {
    allocator: Allocator,
    queue_type: QueueType,
    items: ArrayList(QueueItem),
    max_size: ?usize,
    
    pub fn init(allocator: Allocator, queue_type: QueueType, max_size: ?usize) !QueueNode {
        return QueueNode{
            .allocator = allocator,
            .queue_type = queue_type,
            .items = ArrayList(QueueItem){},
            .max_size = max_size,
        };
    }
    
    pub fn deinit(self: *QueueNode) void {
        for (self.items.items) |item| {
            self.allocator.free(item.data);
        }
        self.items.deinit(self.allocator);
    }
    
    pub fn enqueue(self: *QueueNode, data: []const u8, priority: i32) !void {
        if (self.max_size) |max| {
            if (self.items.items.len >= max) {
                return error.QueueFull;
            }
        }
        
        const item = QueueItem{
            .data = try self.allocator.dupe(u8, data),
            .priority = priority,
            .timestamp = std.time.milliTimestamp(),
        };
        
        try self.items.append(self.allocator, item);
        
        // Sort if priority queue
        if (self.queue_type == .priority) {
            std.mem.sort(QueueItem, self.items.items, {}, QueueItem.lessThan);
        }
    }
    
    pub fn dequeue(self: *QueueNode) !?[]const u8 {
        if (self.items.items.len == 0) {
            return null;
        }
        
        const index = switch (self.queue_type) {
            .fifo, .priority => 0,
            .lifo => self.items.items.len - 1,
        };
        
        const item = self.items.orderedRemove(index);
        return item.data;
    }
    
    pub fn peek(self: *const QueueNode) ?[]const u8 {
        if (self.items.items.len == 0) {
            return null;
        }
        
        const index = switch (self.queue_type) {
            .fifo, .priority => 0,
            .lifo => self.items.items.len - 1,
        };
        
        return self.items.items[index].data;
    }
    
    pub fn size(self: *const QueueNode) usize {
        return self.items.items.len;
    }
    
    pub fn isEmpty(self: *const QueueNode) bool {
        return self.items.items.len == 0;
    }
    
    pub fn isFull(self: *const QueueNode) bool {
        if (self.max_size) |max| {
            return self.items.items.len >= max;
        }
        return false;
    }
    
    pub fn clear(self: *QueueNode) void {
        for (self.items.items) |item| {
            self.allocator.free(item.data);
        }
        self.items.clearRetainingCapacity();
    }
};

// ============================================================================
// Batch Node - Batch processing with size and time windows
// ============================================================================

pub const BatchNode = struct {
    allocator: Allocator,
    batch_size: usize,
    timeout_ms: u64,
    current_batch: ArrayList([]const u8),
    batch_start_time: ?i64,
    
    pub fn init(allocator: Allocator, batch_size: usize, timeout_ms: u64) !BatchNode {
        return BatchNode{
            .allocator = allocator,
            .batch_size = batch_size,
            .timeout_ms = timeout_ms,
            .current_batch = ArrayList([]const u8){},
            .batch_start_time = null,
        };
    }
    
    pub fn deinit(self: *BatchNode) void {
        for (self.current_batch.items) |item| {
            self.allocator.free(item);
        }
        self.current_batch.deinit(self.allocator);
    }
    
    pub fn add(self: *BatchNode, data: []const u8) !?[][]const u8 {
        const now = std.time.milliTimestamp();
        
        // Initialize batch start time on first item
        if (self.batch_start_time == null) {
            self.batch_start_time = now;
        }
        
        const data_copy = try self.allocator.dupe(u8, data);
        try self.current_batch.append(self.allocator, data_copy);
        
        // Check if batch is ready
        if (self.isReady(now)) {
            return try self.flush();
        }
        
        return null;
    }
    
    fn isReady(self: *const BatchNode, now: i64) bool {
        // Size threshold met
        if (self.current_batch.items.len >= self.batch_size) {
            return true;
        }
        
        // Timeout threshold met
        if (self.batch_start_time) |start| {
            const elapsed = @as(u64, @intCast(now - start));
            if (elapsed >= self.timeout_ms) {
                return true;
            }
        }
        
        return false;
    }
    
    pub fn flush(self: *BatchNode) ![][]const u8 {
        if (self.current_batch.items.len == 0) {
            return &[_][]const u8{};
        }
        
        const batch = try self.current_batch.toOwnedSlice(self.allocator);
        self.batch_start_time = null;
        
        return batch;
    }
    
    pub fn forceFlush(self: *BatchNode) ![][]const u8 {
        return try self.flush();
    }
    
    pub fn size(self: *const BatchNode) usize {
        return self.current_batch.items.len;
    }
};

// ============================================================================
// Throttle Node - Limit execution rate
// ============================================================================

pub const ThrottleNode = struct {
    allocator: Allocator,
    min_interval_ms: u64,
    last_execution: ?i64,
    pending_data: ?[]const u8,
    
    pub fn init(allocator: Allocator, min_interval_ms: u64) !ThrottleNode {
        return ThrottleNode{
            .allocator = allocator,
            .min_interval_ms = min_interval_ms,
            .last_execution = null,
            .pending_data = null,
        };
    }
    
    pub fn deinit(self: *ThrottleNode) void {
        if (self.pending_data) |data| {
            self.allocator.free(data);
        }
    }
    
    pub fn execute(self: *ThrottleNode, data: []const u8) !?[]const u8 {
        const now = std.time.milliTimestamp();
        
        // Check if enough time has passed
        if (self.last_execution) |last| {
            const elapsed = @as(u64, @intCast(now - last));
            if (elapsed < self.min_interval_ms) {
                // Store pending data (overwrite if exists)
                if (self.pending_data) |old_data| {
                    self.allocator.free(old_data);
                }
                self.pending_data = try self.allocator.dupe(u8, data);
                return null;
            }
        }
        
        // Execute immediately
        self.last_execution = now;
        
        // Clear any pending data
        if (self.pending_data) |old_data| {
            self.allocator.free(old_data);
            self.pending_data = null;
        }
        
        return try self.allocator.dupe(u8, data);
    }
    
    pub fn hasPending(self: *const ThrottleNode) bool {
        return self.pending_data != null;
    }
    
    pub fn getPending(self: *ThrottleNode) ?[]const u8 {
        if (self.pending_data) |data| {
            self.pending_data = null;
            return data;
        }
        return null;
    }
    
    pub fn canExecute(self: *const ThrottleNode) bool {
        if (self.last_execution) |last| {
            const now = std.time.milliTimestamp();
            const elapsed = @as(u64, @intCast(now - last));
            return elapsed >= self.min_interval_ms;
        }
        return true;
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "RateLimiterNode: fixed window" {
    const allocator = std.testing.allocator;
    
    var limiter = try RateLimiterNode.init(allocator, 3, 1000, .fixed_window);
    defer limiter.deinit();
    
    // Should allow first 3 requests
    try std.testing.expect(try limiter.allowRequest());
    try std.testing.expect(try limiter.allowRequest());
    try std.testing.expect(try limiter.allowRequest());
    
    // Should deny 4th request
    try std.testing.expect(!try limiter.allowRequest());
}

test "RateLimiterNode: sliding window" {
    const allocator = std.testing.allocator;
    
    var limiter = try RateLimiterNode.init(allocator, 2, 1000, .sliding_window);
    defer limiter.deinit();
    
    try std.testing.expect(try limiter.allowRequest());
    try std.testing.expect(try limiter.allowRequest());
    try std.testing.expect(!try limiter.allowRequest());
}

test "RateLimiterNode: token bucket" {
    const allocator = std.testing.allocator;
    
    var limiter = try RateLimiterNode.init(allocator, 5, 1000, .token_bucket);
    defer limiter.deinit();
    
    try std.testing.expect(try limiter.allowRequest());
    try std.testing.expect(try limiter.allowRequest());
}

test "RateLimiterNode: reset" {
    const allocator = std.testing.allocator;
    
    var limiter = try RateLimiterNode.init(allocator, 2, 1000, .fixed_window);
    defer limiter.deinit();
    
    _ = try limiter.allowRequest();
    _ = try limiter.allowRequest();
    
    limiter.reset();
    
    try std.testing.expect(try limiter.allowRequest());
}

test "QueueNode: FIFO" {
    const allocator = std.testing.allocator;
    
    var queue = try QueueNode.init(allocator, .fifo, null);
    defer queue.deinit();
    
    try queue.enqueue("first", 0);
    try queue.enqueue("second", 0);
    try queue.enqueue("third", 0);
    
    const item1 = try queue.dequeue();
    try std.testing.expect(item1 != null);
    try std.testing.expectEqualStrings("first", item1.?);
    defer allocator.free(item1.?);
    
    const item2 = try queue.dequeue();
    try std.testing.expect(item2 != null);
    try std.testing.expectEqualStrings("second", item2.?);
    defer allocator.free(item2.?);
}

test "QueueNode: LIFO" {
    const allocator = std.testing.allocator;
    
    var queue = try QueueNode.init(allocator, .lifo, null);
    defer queue.deinit();
    
    try queue.enqueue("first", 0);
    try queue.enqueue("second", 0);
    
    const item1 = try queue.dequeue();
    try std.testing.expect(item1 != null);
    try std.testing.expectEqualStrings("second", item1.?);
    defer allocator.free(item1.?);
}

test "QueueNode: priority" {
    const allocator = std.testing.allocator;
    
    var queue = try QueueNode.init(allocator, .priority, null);
    defer queue.deinit();
    
    try queue.enqueue("low", 1);
    try queue.enqueue("high", 10);
    try queue.enqueue("medium", 5);
    
    const item1 = try queue.dequeue();
    try std.testing.expect(item1 != null);
    try std.testing.expectEqualStrings("high", item1.?);
    defer allocator.free(item1.?);
}

test "QueueNode: max size" {
    const allocator = std.testing.allocator;
    
    var queue = try QueueNode.init(allocator, .fifo, 2);
    defer queue.deinit();
    
    try queue.enqueue("first", 0);
    try queue.enqueue("second", 0);
    
    const result = queue.enqueue("third", 0);
    try std.testing.expectError(error.QueueFull, result);
}

test "QueueNode: peek" {
    const allocator = std.testing.allocator;
    
    var queue = try QueueNode.init(allocator, .fifo, null);
    defer queue.deinit();
    
    try queue.enqueue("data", 0);
    
    const peeked = queue.peek();
    try std.testing.expect(peeked != null);
    try std.testing.expectEqualStrings("data", peeked.?);
    
    try std.testing.expectEqual(@as(usize, 1), queue.size());
}

test "BatchNode: size threshold" {
    const allocator = std.testing.allocator;
    
    var batch = try BatchNode.init(allocator, 3, 10000);
    defer batch.deinit();
    
    _ = try batch.add("item1");
    _ = try batch.add("item2");
    
    const result = try batch.add("item3");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 3), result.?.len);
    
    for (result.?) |item| {
        allocator.free(item);
    }
    allocator.free(result.?);
}

test "BatchNode: manual flush" {
    const allocator = std.testing.allocator;
    
    var batch = try BatchNode.init(allocator, 10, 10000);
    defer batch.deinit();
    
    _ = try batch.add("item1");
    _ = try batch.add("item2");
    
    const result = try batch.forceFlush();
    try std.testing.expectEqual(@as(usize, 2), result.len);
    
    for (result) |item| {
        allocator.free(item);
    }
    allocator.free(result);
}

test "ThrottleNode: basic throttling" {
    const allocator = std.testing.allocator;
    
    var throttle = try ThrottleNode.init(allocator, 100);
    defer throttle.deinit();
    
    const result1 = try throttle.execute("data1");
    try std.testing.expect(result1 != null);
    defer if (result1) |r| allocator.free(r);
    
    // Should be throttled
    const result2 = try throttle.execute("data2");
    try std.testing.expect(result2 == null);
    try std.testing.expect(throttle.hasPending());
}

test "ThrottleNode: can execute check" {
    const allocator = std.testing.allocator;
    
    var throttle = try ThrottleNode.init(allocator, 50);
    defer throttle.deinit();
    
    try std.testing.expect(throttle.canExecute());
    
    const result = try throttle.execute("data");
    if (result) |r| {
        defer allocator.free(r);
    }
    try std.testing.expect(!throttle.canExecute());
}
