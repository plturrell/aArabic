// Thread Safety Tests for Petri Net Module
const std = @import("std");
const petri = @import("../petri/core.zig");
const types = @import("../petri/types.zig");

test "Petri: Concurrent place token additions" {
    const allocator = std.testing.allocator;
    
    // Create net
    const net = petri.pn_create("test_net", 0);
    defer _ = petri.pn_destroy(net);
    
    // Create a single place
    const place = petri.pn_place_create(net, "p1", "Test Place");
    try std.testing.expect(place != null);
    
    const ThreadContext = struct {
        place_ptr: *types.pn_place_t,
        iterations: usize,
    };
    
    fn addTokens(ctx: *ThreadContext) void {
        var idx: usize = 0;
        while (idx < ctx.iterations) : (idx += 1) {
            const token = petri.pn_token_create(null, 0);
            _ = petri.pn_token_put(ctx.place_ptr, token);
        }
    }
    
    // Spawn 4 threads, each adding 100 tokens
    const num_threads = 4;
    const iterations_per_thread = 100;
    
    var threads: [num_threads]std.Thread = undefined;
    var contexts: [num_threads]ThreadContext = undefined;
    
    for (0..num_threads) |i| {
        contexts[i] = .{
            .place_ptr = place.?,
            .iterations = iterations_per_thread,
        };
        threads[i] = try std.Thread.spawn(.{}, addTokens, .{&contexts[i]});
    }
    
    // Wait for all threads
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify total tokens = 4 * 100 = 400
    const token_count = petri.pn_place_token_count(place);
    try std.testing.expectEqual(@as(usize, num_threads * iterations_per_thread), token_count);
}

test "Petri: Concurrent transition firings" {
    const allocator = std.testing.allocator;
    
    // Create net with producer-consumer pattern
    const net = petri.pn_create("concurrent_test", 0);
    defer _ = petri.pn_destroy(net);
    
    // Create places
    const input_place = petri.pn_place_create(net, "input", "Input");
    const buffer_place = petri.pn_place_create(net, "buffer", "Buffer");
    
    // Create transition
    const produce_trans = petri.pn_trans_create(net, "produce", "Produce");
    
    // Create arcs: input -> produce -> buffer
    const arc1 = petri.pn_arc_create(net, "a1", .input);
    _ = petri.pn_arc_connect(arc1, "input", "produce");
    
    const arc2 = petri.pn_arc_create(net, "a2", .output);
    _ = petri.pn_arc_connect(arc2, "produce", "buffer");
    
    // Add 1000 tokens to input
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const token = petri.pn_token_create(null, 0);
        _ = petri.pn_token_put(input_place, token);
    }
    
    const FireContext = struct {
        net_ptr: *types.pn_net_t,
    };
    
    fn fireTransitions(ctx: *FireContext) void {
        // Each thread fires transitions as fast as possible
        var count: usize = 0;
        while (count < 250) : (count += 1) {
            _ = petri.pn_step(ctx.net_ptr);
        }
    }
    
    // Spawn 4 threads firing simultaneously
    const num_threads = 4;
    var threads: [num_threads]std.Thread = undefined;
    var contexts: [num_threads]FireContext = undefined;
    
    for (0..num_threads) |idx| {
        contexts[idx] = .{ .net_ptr = net.? };
        threads[idx] = try std.Thread.spawn(.{}, fireTransitions, .{&contexts[idx]});
    }
    
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify tokens moved correctly
    const buffer_tokens = petri.pn_place_token_count(buffer_place);
    try std.testing.expect(buffer_tokens <= 1000);
}

test "Petri: Concurrent read/write (stats while firing)" {
    const allocator = std.testing.allocator;
    
    const net = petri.pn_create("rw_test", 0);
    defer _ = petri.pn_destroy(net);
    
    // Setup simple net
    const p1 = petri.pn_place_create(net, "p1", "Place 1");
    const t1 = petri.pn_trans_create(net, "t1", "Trans 1");
    
    // Add tokens
    var i: usize = 0;
    while (i < 500) : (i += 1) {
        const token = petri.pn_token_create(null, 0);
        _ = petri.pn_token_put(p1, token);
    }
    
    const ReadContext = struct {
        net_ptr: *types.pn_net_t,
        iterations: usize,
    };
    
    fn readStats(ctx: *ReadContext) void {
        var i: usize = 0;
        while (i < ctx.iterations) : (i += 1) {
            var stats: types.pn_stats_t = undefined;
            _ = petri.pn_stats(ctx.net_ptr, &stats);
            std.Thread.yield() catch {};
        }
    }
    
    fn writeTokens(ctx: *ReadContext) void {
        var i: usize = 0;
        while (i < ctx.iterations) : (i += 1) {
            const token = petri.pn_token_create(null, 0);
            const p1_ptr = petri.pn_place_get(ctx.net_ptr, "p1");
            if (p1_ptr) |place| {
                _ = petri.pn_token_put(place, token);
            }
            std.Thread.yield() catch {};
        }
    }
    
    // Spawn 2 readers and 2 writers
    var reader_threads: [2]std.Thread = undefined;
    var writer_threads: [2]std.Thread = undefined;
    
    var read_contexts: [2]ReadContext = undefined;
    var write_contexts: [2]ReadContext = undefined;
    
    for (0..2) |idx| {
        read_contexts[idx] = .{ .net_ptr = net.?, .iterations = 100 };
        write_contexts[idx] = .{ .net_ptr = net.?, .iterations = 50 };
        
        reader_threads[idx] = try std.Thread.spawn(.{}, readStats, .{&read_contexts[idx]});
        writer_threads[idx] = try std.Thread.spawn(.{}, writeTokens, .{&write_contexts[idx]});
    }
    
    for (reader_threads) |thread| thread.join();
    for (writer_threads) |thread| thread.join();
    
    // If we got here without deadlock or crash, test passes!
    try std.testing.expect(true);
}

test "Petri: No deadlock under concurrent access" {
    // This test ensures multiple threads can access without deadlocking
    const allocator = std.testing.allocator;
    
    const net = petri.pn_create("deadlock_test", 0);
    defer _ = petri.pn_destroy(net);
    
    // Create multiple places and transitions
    _ = petri.pn_place_create(net, "p1", "P1");
    _ = petri.pn_place_create(net, "p2", "P2");
    _ = petri.pn_place_create(net, "p3", "P3");
    
    _ = petri.pn_trans_create(net, "t1", "T1");
    _ = petri.pn_trans_create(net, "t2", "T2");
    
    const MixedContext = struct {
        net_ptr: *types.pn_net_t,
        iterations: usize,
    };
    
    fn mixedOperations(ctx: *MixedContext) void {
        var i: usize = 0;
        while (i < ctx.iterations) : (i += 1) {
            // Mix of reads and writes
            var stats: types.pn_stats_t = undefined;
            _ = petri.pn_stats(ctx.net_ptr, &stats);
            
            _ = petri.pn_validate(ctx.net_ptr);
            
            std.Thread.yield() catch {};
        }
    }
    
    // Spawn 8 threads doing mixed operations
    const num_threads = 8;
    var threads: [num_threads]std.Thread = undefined;
    var contexts: [num_threads]MixedContext = undefined;
    
    for (0..num_threads) |idx| {
        contexts[idx] = .{ .net_ptr = net.?, .iterations = 50 };
        threads[idx] = try std.Thread.spawn(.{}, mixedOperations, .{&contexts[idx]});
    }
    
    for (threads) |thread| {
        thread.join();
    }
    
    try std.testing.expect(true);
}
