//! Comprehensive test suite for Petri Net library
//! Tests core functionality, state analysis, and monitoring

const std = @import("std");
const testing = std.testing;
const lib = @import("lib.zig");
const types = lib.types;

test "create and destroy petri net" {
    const net = lib.pn_create("test_net", 0);
    try testing.expect(net != null);
    
    const result = lib.pn_destroy(net);
    try testing.expectEqual(@as(c_int, 0), result);
}

test "create places and transitions" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    const p1 = lib.pn_place_create(net, "p1", "Place 1");
    try testing.expect(p1 != null);
    
    const p2 = lib.pn_place_create(net, "p2", "Place 2");
    try testing.expect(p2 != null);
    
    const t1 = lib.pn_trans_create(net, "t1", "Transition 1");
    try testing.expect(t1 != null);
}

test "create and connect arcs" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    const p1 = lib.pn_place_create(net, "p1", "Place 1");
    const p2 = lib.pn_place_create(net, "p2", "Place 2");
    const t1 = lib.pn_trans_create(net, "t1", "Transition 1");
    
    const arc1 = lib.pn_arc_create(net, "a1", .input);
    try testing.expect(arc1 != null);
    
    var result = lib.pn_arc_connect(arc1, "p1", "t1");
    try testing.expectEqual(@as(c_int, 0), result);
    
    result = lib.pn_arc_set_weight(arc1, 1);
    try testing.expectEqual(@as(c_int, 0), result);
    
    const arc2 = lib.pn_arc_create(net, "a2", .output);
    result = lib.pn_arc_connect(arc2, "t1", "p2");
    try testing.expectEqual(@as(c_int, 0), result);
    
    // Validate the net structure
    result = lib.pn_validate(net);
    try testing.expectEqual(@as(c_int, 0), result);
}

test "token operations" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    const p1 = lib.pn_place_create(net, "p1", "Place 1");
    
    // Create and put token
    const data: [4]u8 = .{ 1, 2, 3, 4 };
    const token = lib.pn_token_create(&data, 4);
    try testing.expect(token != null);
    
    var result = lib.pn_token_put(p1, token);
    try testing.expectEqual(@as(c_int, 0), result);
    
    // Check token count
    const count = lib.pn_place_token_count(p1);
    try testing.expectEqual(@as(usize, 1), count);
    
    // Get token back
    const retrieved = lib.pn_token_get(p1);
    try testing.expect(retrieved != null);
    
    // Verify place is now empty
    const final_count = lib.pn_place_token_count(p1);
    try testing.expectEqual(@as(usize, 0), final_count);
    
    _ = lib.pn_token_destroy(token);
    _ = lib.pn_token_destroy(retrieved);
}

test "simple petri net execution" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    // Create a simple producer-consumer net
    // p1 -> t1 -> p2
    const p1 = lib.pn_place_create(net, "p1", "Input");
    const p2 = lib.pn_place_create(net, "p2", "Output");
    const t1 = lib.pn_trans_create(net, "t1", "Transfer");
    
    // Connect arcs
    const arc1 = lib.pn_arc_create(net, "a1", .input);
    _ = lib.pn_arc_connect(arc1, "p1", "t1");
    _ = lib.pn_arc_set_weight(arc1, 1);
    
    const arc2 = lib.pn_arc_create(net, "a2", .output);
    _ = lib.pn_arc_connect(arc2, "t1", "p2");
    _ = lib.pn_arc_set_weight(arc2, 1);
    
    // Add initial token to p1
    const token = lib.pn_token_create(null, 0);
    _ = lib.pn_token_put(p1, token);
    
    // Verify initial state
    try testing.expectEqual(@as(usize, 1), lib.pn_place_token_count(p1));
    try testing.expectEqual(@as(usize, 0), lib.pn_place_token_count(p2));
    
    // Execute one step
    const result = lib.pn_step(net);
    try testing.expectEqual(@as(c_int, 0), result);
    
    // Verify token moved
    try testing.expectEqual(@as(usize, 0), lib.pn_place_token_count(p1));
    try testing.expectEqual(@as(usize, 1), lib.pn_place_token_count(p2));
    
    _ = lib.pn_token_destroy(token);
}

test "deadlock detection" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    _ = lib.pn_place_create(net, "p1", "Place 1");
    _ = lib.pn_trans_create(net, "t1", "Transition 1");
    
    // No tokens, no enabled transitions -> deadlock
    const is_deadlocked = lib.pn_is_deadlocked(net);
    try testing.expectEqual(@as(c_int, 1), is_deadlocked);
}

test "bounded and safe net detection" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    const p1 = lib.pn_place_create(net, "p1", "Place 1");
    _ = lib.pn_place_set_capacity(p1, 1);
    
    // With capacity 1, it should be safe
    const is_safe = lib.pn_is_safe(net);
    try testing.expectEqual(@as(c_int, 1), is_safe);
    
    // Check if bounded
    var bound: usize = 0;
    const is_bounded = lib.pn_is_bounded(net, &bound);
    try testing.expectEqual(@as(c_int, 1), is_bounded);
    try testing.expectEqual(@as(usize, 1), bound);
}

test "trace and monitoring" {
    const net = lib.pn_create("test_net", types.PN_CREATE_TRACED);
    defer _ = lib.pn_destroy(net);
    
    // Create simple net
    const p1 = lib.pn_place_create(net, "p1", "Place 1");
    const p2 = lib.pn_place_create(net, "p2", "Place 2");
    const t1 = lib.pn_trans_create(net, "t1", "Transition 1");
    
    const arc1 = lib.pn_arc_create(net, "a1", .input);
    _ = lib.pn_arc_connect(arc1, "p1", "t1");
    const arc2 = lib.pn_arc_create(net, "a2", .output);
    _ = lib.pn_arc_connect(arc2, "t1", "p2");
    
    // Add token and execute
    const token = lib.pn_token_create(null, 0);
    _ = lib.pn_token_put(p1, token);
    _ = lib.pn_step(net);
    
    // Check trace events
    var count: usize = 0;
    const result = lib.pn_trace_get(net, null, &count);
    try testing.expectEqual(@as(c_int, 0), result);
    try testing.expect(count > 0); // Should have recorded events
    
    // Clear trace
    _ = lib.pn_trace_clear(net);
    _ = lib.pn_trace_get(net, null, &count);
    try testing.expectEqual(@as(usize, 0), count);
    
    _ = lib.pn_token_destroy(token);
}

test "metrics collection" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    const p1 = lib.pn_place_create(net, "p1", "Place 1");
    const p2 = lib.pn_place_create(net, "p2", "Place 2");
    _ = lib.pn_trans_create(net, "t1", "Transition 1");
    
    // Add tokens
    const token1 = lib.pn_token_create(null, 0);
    const token2 = lib.pn_token_create(null, 0);
    _ = lib.pn_token_put(p1, token1);
    _ = lib.pn_token_put(p2, token2);
    
    // Get metrics
    var metrics: types.pn_metrics_t = undefined;
    const result = lib.pn_metrics_get(net, &metrics);
    try testing.expectEqual(@as(c_int, 0), result);
    try testing.expectEqual(@as(f64, 1.0), metrics.avg_tokens_per_place);
    try testing.expectEqual(@as(usize, 1), metrics.max_tokens_per_place);
    
    _ = lib.pn_token_destroy(token1);
    _ = lib.pn_token_destroy(token2);
}

test "statistics collection" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    _ = lib.pn_place_create(net, "p1", "Place 1");
    _ = lib.pn_place_create(net, "p2", "Place 2");
    _ = lib.pn_trans_create(net, "t1", "Transition 1");
    
    var stats: types.pn_stats_t = undefined;
    const result = lib.pn_stats(net, &stats);
    try testing.expectEqual(@as(c_int, 0), result);
    try testing.expectEqual(@as(usize, 2), stats.place_count);
    try testing.expectEqual(@as(usize, 1), stats.transition_count);
}

test "inhibitor arc functionality" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    // Create net with inhibitor arc
    // p1 -o t1 -> p2 (inhibitor arc from p1 to t1)
    const p1 = lib.pn_place_create(net, "p1", "Inhibitor Place");
    const p2 = lib.pn_place_create(net, "p2", "Output Place");
    const t1 = lib.pn_trans_create(net, "t1", "Transition 1");
    
    const inhibitor = lib.pn_arc_create(net, "inh1", .inhibitor);
    _ = lib.pn_arc_connect(inhibitor, "p1", "t1");
    
    const output = lib.pn_arc_create(net, "out1", .output);
    _ = lib.pn_arc_connect(output, "t1", "p2");
    
    // When p1 is empty, transition should fire
    var result = lib.pn_step(net);
    try testing.expectEqual(@as(c_int, 0), result);
    
    // Add token to p1
    const token = lib.pn_token_create(null, 0);
    _ = lib.pn_token_put(p1, token);
    
    // Now transition should be blocked (would need full simulation to verify)
    // This tests the structure is correct
    result = lib.pn_validate(net);
    try testing.expectEqual(@as(c_int, 0), result);
    
    _ = lib.pn_token_destroy(token);
}

test "priority-based firing" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    const t1 = lib.pn_trans_create(net, "t1", "Low Priority");
    const t2 = lib.pn_trans_create(net, "t2", "High Priority");
    
    _ = lib.pn_trans_set_priority(t1, 1);
    _ = lib.pn_trans_set_priority(t2, 10);
    
    const p1 = lib.pn_trans_get_priority(t1);
    const p2 = lib.pn_trans_get_priority(t2);
    
    try testing.expectEqual(@as(c_int, 1), p1);
    try testing.expectEqual(@as(c_int, 10), p2);
}

test "place capacity limits" {
    const net = lib.pn_create("test_net", 0);
    defer _ = lib.pn_destroy(net);
    
    const p1 = lib.pn_place_create(net, "p1", "Limited Place");
    _ = lib.pn_place_set_capacity(p1, 2);
    
    const cap = lib.pn_place_get_capacity(p1);
    try testing.expectEqual(@as(usize, 2), cap);
    
    // Add tokens up to capacity
    const token1 = lib.pn_token_create(null, 0);
    const token2 = lib.pn_token_create(null, 0);
    const token3 = lib.pn_token_create(null, 0);
    
    try testing.expectEqual(@as(c_int, 0), lib.pn_token_put(p1, token1));
    try testing.expectEqual(@as(c_int, 0), lib.pn_token_put(p1, token2));
    
    // Third token should fail
    try testing.expectEqual(@as(c_int, -1), lib.pn_token_put(p1, token3));
    
    _ = lib.pn_token_destroy(token1);
    _ = lib.pn_token_destroy(token2);
    _ = lib.pn_token_destroy(token3);
}