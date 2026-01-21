const std = @import("std");
const testing = std.testing;
const cuda_streams = @import("cuda_streams.zig");
const CudaStream = cuda_streams.CudaStream;
const CudaEvent = cuda_streams.CudaEvent;
const StreamPool = cuda_streams.StreamPool;
const StreamTimer = cuda_streams.StreamTimer;

// ============================================================================
// Stream Tests
// ============================================================================

test "cuda_streams: basic stream creation" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    try testing.expect(@intFromPtr(stream.handle) != 0);
    try testing.expect(stream.priority == 0);
    
    std.debug.print("✓ Stream created successfully\n", .{});
}

test "cuda_streams: stream with priority" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.initWithPriority(allocator, -1) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    try testing.expect(@intFromPtr(stream.handle) != 0);
    std.debug.print("✓ Priority stream created (priority: {})\n", .{stream.priority});
}

test "cuda_streams: stream synchronization" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    try stream.synchronize();
    std.debug.print("✓ Stream synchronized successfully\n", .{});
}

test "cuda_streams: check stream completion" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    const is_complete = try stream.isComplete();
    std.debug.print("✓ Stream completion check: {}\n", .{is_complete});
}

test "cuda_streams: multiple streams" {
    const allocator = testing.allocator;
    
    var stream1 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream1.deinit();
    
    var stream2 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream2.deinit();
    
    try testing.expect(stream1.handle != stream2.handle);
    std.debug.print("✓ Multiple streams created with different handles\n", .{});
}

// ============================================================================
// Event Tests
// ============================================================================

test "cuda_streams: basic event creation" {
    const allocator = testing.allocator;
    
    var event = CudaEvent.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer event.deinit();
    
    try testing.expect(@intFromPtr(event.handle) != 0);
    std.debug.print("✓ Event created successfully\n", .{});
}

test "cuda_streams: event with flags" {
    const allocator = testing.allocator;
    const cuda = @import("cuda_bindings");

    var event = CudaEvent.initWithFlags(allocator, cuda.cudaEventDefault) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer event.deinit();
    
    try testing.expect(@intFromPtr(event.handle) != 0);
    std.debug.print("✓ Event with flags created successfully\n", .{});
}

test "cuda_streams: event record and synchronize" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    var event = CudaEvent.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer event.deinit();
    
    try event.record(&stream);
    try event.synchronize();
    
    std.debug.print("✓ Event recorded and synchronized\n", .{});
}

test "cuda_streams: event completion check" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    var event = CudaEvent.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer event.deinit();
    
    try event.record(&stream);
    try event.synchronize();
    
    const is_complete = try event.isComplete();
    try testing.expect(is_complete == true);
    
    std.debug.print("✓ Event completion verified\n", .{});
}

test "cuda_streams: measure elapsed time" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    var start_event = CudaEvent.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer start_event.deinit();
    
    var end_event = CudaEvent.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer end_event.deinit();
    
    try start_event.record(&stream);
    try end_event.record(&stream);
    try end_event.synchronize();
    
    const elapsed_ms = try CudaEvent.elapsedTime(&start_event, &end_event);
    try testing.expect(elapsed_ms >= 0.0);
    
    std.debug.print("✓ Elapsed time measured: {d:.3} ms\n", .{elapsed_ms});
}

// ============================================================================
// Stream Pool Tests
// ============================================================================

test "cuda_streams: stream pool creation" {
    const allocator = testing.allocator;
    
    var pool = StreamPool.init(allocator, 4) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer pool.deinit();
    
    const stats = pool.getStats();
    try testing.expect(stats.total == 4);
    try testing.expect(stats.available == 4);
    try testing.expect(stats.in_use == 0);
    
    std.debug.print("✓ Stream pool created: {} total streams\n", .{stats.total});
}

test "cuda_streams: stream pool acquire and release" {
    const allocator = testing.allocator;
    
    var pool = StreamPool.init(allocator, 4) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer pool.deinit();
    
    // Acquire a stream
    const stream1 = try pool.acquire();
    var stats = pool.getStats();
    try testing.expect(stats.available == 3);
    try testing.expect(stats.in_use == 1);
    
    // Acquire another stream
    const stream2 = try pool.acquire();
    stats = pool.getStats();
    try testing.expect(stats.available == 2);
    try testing.expect(stats.in_use == 2);
    
    // Release streams
    try pool.release(stream1);
    stats = pool.getStats();
    try testing.expect(stats.available == 3);
    try testing.expect(stats.in_use == 1);
    
    try pool.release(stream2);
    stats = pool.getStats();
    try testing.expect(stats.available == 4);
    try testing.expect(stats.in_use == 0);
    
    std.debug.print("✓ Stream pool acquire/release working correctly\n", .{});
}

test "cuda_streams: stream pool exhaustion" {
    const allocator = testing.allocator;
    
    var pool = StreamPool.init(allocator, 2) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer pool.deinit();
    
    // Acquire all streams
    const stream1 = try pool.acquire();
    const stream2 = try pool.acquire();
    
    // Try to acquire when pool is exhausted
    const result = pool.acquire();
    try testing.expectError(error.NoStreamsAvailable, result);
    
    // Release one and acquire again
    try pool.release(stream1);
    const stream3 = try pool.acquire();
    try testing.expect(stream3.handle == stream1.handle);
    
    try pool.release(stream2);
    try pool.release(stream3);
    
    std.debug.print("✓ Stream pool exhaustion handled correctly\n", .{});
}

// ============================================================================
// Stream Timer Tests
// ============================================================================

test "cuda_streams: stream timer" {
    const allocator = testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    var timer = StreamTimer.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer timer.deinit();
    
    try timer.start(&stream);
    // Some work would happen here
    try timer.stop(&stream);
    
    const elapsed_ms = try timer.getElapsedTime();
    try testing.expect(elapsed_ms >= 0.0);
    
    std.debug.print("✓ Stream timer measured: {d:.3} ms\n", .{elapsed_ms});
}

// ============================================================================
// Utility Function Tests
// ============================================================================

test "cuda_streams: synchronize multiple streams" {
    const allocator = testing.allocator;
    
    var stream1 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream1.deinit();
    
    var stream2 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream2.deinit();
    
    var streams = [_]CudaStream{ stream1, stream2 };
    try cuda_streams.synchronizeStreams(&streams);
    
    std.debug.print("✓ Multiple streams synchronized\n", .{});
}

test "cuda_streams: check all streams complete" {
    const allocator = testing.allocator;
    
    var stream1 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream1.deinit();
    
    var stream2 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream2.deinit();
    
    try stream1.synchronize();
    try stream2.synchronize();
    
    var streams = [_]CudaStream{ stream1, stream2 };
    const all_complete = try cuda_streams.allStreamsComplete(&streams);
    try testing.expect(all_complete == true);
    
    std.debug.print("✓ All streams completion verified\n", .{});
}

test "cuda_streams: stream wait for event" {
    const allocator = testing.allocator;
    
    var stream1 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream1.deinit();
    
    var stream2 = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream2.deinit();
    
    var event = CudaEvent.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer event.deinit();
    
    try event.record(&stream1);
    try cuda_streams.streamWaitEvent(&stream2, &event);
    try stream2.synchronize();
    
    std.debug.print("✓ Stream waited for event successfully\n", .{});
}

// ============================================================================
// Integration Tests
// ============================================================================

test "cuda_streams: comprehensive workflow" {
    const allocator = testing.allocator;
    
    std.debug.print("\n🔄 Running comprehensive stream workflow test...\n", .{});
    
    // Create stream pool
    var pool = StreamPool.init(allocator, 3) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer pool.deinit();
    
    // Acquire streams
    const stream1 = try pool.acquire();
    const stream2 = try pool.acquire();
    
    // Create events for synchronization
    var event1 = try CudaEvent.init(allocator);
    defer event1.deinit();
    
    var event2 = try CudaEvent.init(allocator);
    defer event2.deinit();
    
    // Create timer
    var timer = try StreamTimer.init(allocator);
    defer timer.deinit();
    
    // Start timing
    try timer.start(stream1);
    
    // Record events in streams
    try event1.record(stream1);
    try event2.record(stream2);
    
    // Make stream2 wait for stream1
    try cuda_streams.streamWaitEvent(stream2, &event1);
    
    // Stop timing
    try timer.stop(stream1);
    
    // Synchronize everything
    try stream1.synchronize();
    try stream2.synchronize();
    
    // Get timing
    const elapsed_ms = try timer.getElapsedTime();
    
    // Release streams back to pool
    try pool.release(stream1);
    try pool.release(stream2);
    
    // Verify pool state
    const stats = pool.getStats();
    try testing.expect(stats.available == 3);
    
    std.debug.print("✓ Comprehensive workflow completed in {d:.3} ms\n", .{elapsed_ms});
    std.debug.print("✓ Pool state verified: {}/{} streams available\n", .{stats.available, stats.total});
}
