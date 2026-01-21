const std = @import("std");
const Allocator = std.mem.Allocator;
const data_packet = @import("data_packet.zig");
const DataPacket = data_packet.DataPacket;

/// Stream mode for data processing
pub const StreamMode = enum {
    push, // Producer pushes data to consumers
    pull, // Consumer pulls data from producer
    
    pub fn toString(self: StreamMode) []const u8 {
        return switch (self) {
            .push => "push",
            .pull => "pull",
        };
    }
    
    pub fn fromString(str: []const u8) !StreamMode {
        if (std.mem.eql(u8, str, "push")) return .push;
        if (std.mem.eql(u8, str, "pull")) return .pull;
        return error.InvalidStreamMode;
    }
};

/// Stream consumer callback function type
pub const StreamConsumer = *const fn (*DataPacket) anyerror!void;

/// Data stream for processing packets in sequence
pub const DataStream = struct {
    allocator: Allocator,
    id: []const u8,
    mode: StreamMode,
    consumers: std.ArrayList(StreamConsumer),
    buffer: std.ArrayList(*DataPacket),
    buffer_size: usize,
    is_closed: bool,
    backpressure_enabled: bool,
    
    pub fn init(allocator: Allocator, id: []const u8, mode: StreamMode, buffer_size: usize) !*DataStream {
        const stream = try allocator.create(DataStream);
        errdefer allocator.destroy(stream);
        
        stream.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .mode = mode,
            .consumers = std.ArrayList(StreamConsumer){},
            .buffer = std.ArrayList(*DataPacket){},
            .buffer_size = buffer_size,
            .is_closed = false,
            .backpressure_enabled = true,
        };
        
        return stream;
    }
    
    pub fn deinit(self: *DataStream) void {
        self.allocator.free(self.id);
        
        // Clean up buffered packets
        for (self.buffer.items) |packet| {
            packet.deinit();
        }
        self.buffer.deinit(self.allocator);
        
        self.consumers.deinit(self.allocator);
        self.allocator.destroy(self);
    }
    
    /// Add a consumer to the stream
    pub fn addConsumer(self: *DataStream, consumer: StreamConsumer) !void {
        try self.consumers.append(self.allocator, consumer);
    }
    
    /// Push a packet to the stream
    pub fn push(self: *DataStream, packet: *DataPacket) !void {
        if (self.is_closed) return error.StreamClosed;
        
        // Check backpressure
        if (self.backpressure_enabled and self.buffer.items.len >= self.buffer_size) {
            return error.StreamFull;
        }
        
        if (self.mode == .push) {
            // Push mode: immediately notify consumers
            for (self.consumers.items) |consumer| {
                try consumer(packet);
            }
        } else {
            // Pull mode: buffer the packet
            try self.buffer.append(self.allocator, packet);
        }
    }
    
    /// Pull a packet from the stream (pull mode only)
    pub fn pull(self: *DataStream) ?*DataPacket {
        if (self.mode != .pull) return null;
        if (self.buffer.items.len == 0) return null;
        
        return self.buffer.orderedRemove(0);
    }
    
    /// Close the stream
    pub fn close(self: *DataStream) void {
        self.is_closed = true;
    }
    
    /// Get stream statistics
    pub fn getStats(self: *const DataStream) StreamStats {
        return .{
            .buffered_packets = self.buffer.items.len,
            .consumer_count = self.consumers.items.len,
            .is_closed = self.is_closed,
            .buffer_capacity = self.buffer_size,
        };
    }
};

/// Stream statistics
pub const StreamStats = struct {
    buffered_packets: usize,
    consumer_count: usize,
    is_closed: bool,
    buffer_capacity: usize,
};

/// Memory pool for DataPackets to reduce allocation overhead
pub const DataPacketPool = struct {
    allocator: Allocator,
    available: std.ArrayList(*DataPacket),
    max_size: usize,
    total_allocated: usize,
    
    pub fn init(allocator: Allocator, max_size: usize) DataPacketPool {
        return .{
            .allocator = allocator,
            .available = std.ArrayList(*DataPacket){},
            .max_size = max_size,
            .total_allocated = 0,
        };
    }
    
    pub fn deinit(self: *DataPacketPool) void {
        // Clean up all pooled packets
        for (self.available.items) |packet| {
            packet.deinit();
        }
        self.available.deinit(self.allocator);
    }
    
    /// Acquire a packet from the pool or create a new one
    pub fn acquire(self: *DataPacketPool, id: []const u8, data_type: data_packet.DataType, value: std.json.Value) !*DataPacket {
        if (self.available.items.len > 0) {
            // Reuse from pool
            const packet = self.available.items[self.available.items.len - 1];
            _ = self.available.pop();
            
            // Reset packet fields
            self.allocator.free(packet.id);
            packet.id = try self.allocator.dupe(u8, id);
            packet.data_type = data_type;
            packet.value = value;
            packet.timestamp = std.time.milliTimestamp();
            
            // Clear metadata
            var it = packet.metadata.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            packet.metadata.clearAndFree();
            
            return packet;
        } else if (self.total_allocated < self.max_size) {
            // Create new packet
            const packet = try DataPacket.init(self.allocator, id, data_type, value);
            self.total_allocated += 1;
            return packet;
        } else {
            return error.PoolExhausted;
        }
    }
    
    /// Release a packet back to the pool
    pub fn release(self: *DataPacketPool, packet: *DataPacket) !void {
        if (self.available.items.len < self.max_size) {
            try self.available.append(self.allocator, packet);
        } else {
            // Pool is full, deallocate
            packet.deinit();
            self.total_allocated = if (self.total_allocated > 0) self.total_allocated - 1 else 0;
        }
    }
    
    /// Get pool statistics
    pub fn getStats(self: *const DataPacketPool) PoolStats {
        return .{
            .available_packets = self.available.items.len,
            .total_allocated = self.total_allocated,
            .max_size = self.max_size,
        };
    }
};

/// Pool statistics
pub const PoolStats = struct {
    available_packets: usize,
    total_allocated: usize,
    max_size: usize,
};

/// Batch processor for efficient bulk operations
pub const BatchProcessor = struct {
    allocator: Allocator,
    batch_size: usize,
    batch: std.ArrayList(*DataPacket),
    processor: *const fn ([]const *DataPacket) anyerror!void,
    
    pub fn init(allocator: Allocator, batch_size: usize, processor: *const fn ([]const *DataPacket) anyerror!void) BatchProcessor {
        return .{
            .allocator = allocator,
            .batch_size = batch_size,
            .batch = std.ArrayList(*DataPacket){},
            .processor = processor,
        };
    }
    
    pub fn deinit(self: *BatchProcessor) void {
        self.batch.deinit(self.allocator);
    }
    
    /// Add a packet to the batch
    pub fn add(self: *BatchProcessor, packet: *DataPacket) !void {
        try self.batch.append(self.allocator, packet);
        
        if (self.batch.items.len >= self.batch_size) {
            try self.flush();
        }
    }
    
    /// Flush the current batch
    pub fn flush(self: *BatchProcessor) !void {
        if (self.batch.items.len == 0) return;
        
        try self.processor(self.batch.items);
        self.batch.clearRetainingCapacity();
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "StreamMode string conversion" {
    try std.testing.expectEqualStrings("push", StreamMode.push.toString());
    try std.testing.expectEqualStrings("pull", StreamMode.pull.toString());
    
    try std.testing.expectEqual(StreamMode.push, try StreamMode.fromString("push"));
    try std.testing.expectEqual(StreamMode.pull, try StreamMode.fromString("pull"));
    try std.testing.expectError(error.InvalidStreamMode, StreamMode.fromString("invalid"));
}

test "DataStream creation and cleanup" {
    const allocator = std.testing.allocator;
    
    const stream = try DataStream.init(allocator, "stream-1", .push, 10);
    defer stream.deinit();
    
    try std.testing.expectEqualStrings("stream-1", stream.id);
    try std.testing.expectEqual(StreamMode.push, stream.mode);
    try std.testing.expectEqual(@as(usize, 10), stream.buffer_size);
}

var test_consumer_called = false;

fn testConsumer(packet: *DataPacket) !void {
    _ = packet;
    test_consumer_called = true;
}

test "DataStream push mode" {
    const allocator = std.testing.allocator;
    
    const stream = try DataStream.init(allocator, "stream-1", .push, 10);
    defer stream.deinit();
    
    try stream.addConsumer(testConsumer);
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    defer packet.deinit();
    
    test_consumer_called = false;
    try stream.push(packet);
    
    try std.testing.expect(test_consumer_called);
}

test "DataStream pull mode" {
    const allocator = std.testing.allocator;
    
    const stream = try DataStream.init(allocator, "stream-1", .pull, 10);
    defer stream.deinit();
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    
    try stream.push(packet);
    
    const pulled = stream.pull();
    try std.testing.expect(pulled != null);
    try std.testing.expectEqualStrings("p1", pulled.?.id);
    
    pulled.?.deinit();
}

test "DataStream backpressure" {
    const allocator = std.testing.allocator;
    
    const stream = try DataStream.init(allocator, "stream-1", .pull, 2);
    defer stream.deinit();
    
    const value1 = std.json.Value{ .string = "test1" };
    const packet1 = try DataPacket.init(allocator, "p1", .string, value1);
    try stream.push(packet1);
    
    const value2 = std.json.Value{ .string = "test2" };
    const packet2 = try DataPacket.init(allocator, "p2", .string, value2);
    try stream.push(packet2);
    
    const value3 = std.json.Value{ .string = "test3" };
    const packet3 = try DataPacket.init(allocator, "p3", .string, value3);
    defer packet3.deinit();
    
    try std.testing.expectError(error.StreamFull, stream.push(packet3));
}

test "DataStream close" {
    const allocator = std.testing.allocator;
    
    const stream = try DataStream.init(allocator, "stream-1", .push, 10);
    defer stream.deinit();
    
    stream.close();
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    defer packet.deinit();
    
    try std.testing.expectError(error.StreamClosed, stream.push(packet));
}

test "DataPacketPool acquire and release" {
    const allocator = std.testing.allocator;
    
    var pool = DataPacketPool.init(allocator, 10);
    defer pool.deinit();
    
    const value = std.json.Value{ .string = "test" };
    const packet = try pool.acquire("p1", .string, value);
    
    try std.testing.expectEqualStrings("p1", packet.id);
    try std.testing.expectEqual(@as(usize, 1), pool.total_allocated);
    
    try pool.release(packet);
    try std.testing.expectEqual(@as(usize, 1), pool.available.items.len);
}

test "DataPacketPool reuse" {
    const allocator = std.testing.allocator;
    
    var pool = DataPacketPool.init(allocator, 10);
    defer pool.deinit();
    
    const value1 = std.json.Value{ .string = "test1" };
    const packet1 = try pool.acquire("p1", .string, value1);
    try pool.release(packet1);
    
    const value2 = std.json.Value{ .string = "test2" };
    const packet2 = try pool.acquire("p2", .string, value2);
    
    try std.testing.expectEqualStrings("p2", packet2.id);
    try std.testing.expectEqual(@as(usize, 1), pool.total_allocated);
    
    try pool.release(packet2);
}

var test_batch_processed = false;
var test_batch_count: usize = 0;

fn testBatchProcessor(packets: []const *DataPacket) !void {
    test_batch_processed = true;
    test_batch_count = packets.len;
}

test "BatchProcessor batching" {
    const allocator = std.testing.allocator;
    
    var processor = BatchProcessor.init(allocator, 3, testBatchProcessor);
    defer processor.deinit();
    
    const value1 = std.json.Value{ .string = "test1" };
    const packet1 = try DataPacket.init(allocator, "p1", .string, value1);
    defer packet1.deinit();
    
    const value2 = std.json.Value{ .string = "test2" };
    const packet2 = try DataPacket.init(allocator, "p2", .string, value2);
    defer packet2.deinit();
    
    test_batch_processed = false;
    try processor.add(packet1);
    try processor.add(packet2);
    try std.testing.expect(!test_batch_processed);
    
    const value3 = std.json.Value{ .string = "test3" };
    const packet3 = try DataPacket.init(allocator, "p3", .string, value3);
    defer packet3.deinit();
    
    try processor.add(packet3);
    try std.testing.expect(test_batch_processed);
    try std.testing.expectEqual(@as(usize, 3), test_batch_count);
}

test "BatchProcessor manual flush" {
    const allocator = std.testing.allocator;
    
    var processor = BatchProcessor.init(allocator, 10, testBatchProcessor);
    defer processor.deinit();
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    defer packet.deinit();
    
    test_batch_processed = false;
    try processor.add(packet);
    try std.testing.expect(!test_batch_processed);
    
    try processor.flush();
    try std.testing.expect(test_batch_processed);
    try std.testing.expectEqual(@as(usize, 1), test_batch_count);
}
