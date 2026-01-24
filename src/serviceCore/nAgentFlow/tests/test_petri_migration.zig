//! Test suite for zig-libc Petri net migration
//! Verifies compatibility wrapper works correctly

const std = @import("std");
const petri_net = @import("../core/petri_net.zig");
const PetriNet = petri_net.PetriNet;

test "Petri Migration: Basic net creation" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Migration Test");
    defer net.deinit();
    
    try std.testing.expectEqualStrings("Migration Test", net.name);
}

test "Petri Migration: Place creation and token management" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Place Test");
    defer net.deinit();
    
    // Create places
    const p1 = try net.addPlace("p1", "Input Place", null);
    const p2 = try net.addPlace("p2", "Output Place", 10);
    
    try std.testing.expect(p1.tokenCount() == 0);
    try std.testing.expect(!p1.hasTokens());
    
    // Add tokens
    try net.addTokenToPlace("p1", "{}");
    try net.addTokenToPlace("p1", "{\"data\": \"test\"}");
    
    try std.testing.expect(p1.tokenCount() == 2);
    try std.testing.expect(p1.hasTokens());
    
    // Verify capacity constraint
    try std.testing.expect(p2.tokenCount() == 0);
}

test "Petri Migration: Transition and arc creation" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Transition Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input", null);
    _ = try net.addPlace("p2", "Output", null);
    
    const t1 = try net.addTransition("t1", "Process", 5);
    try std.testing.expectEqual(@as(i32, 5), t1.priority);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try std.testing.expect(net.arcs_list.items.len == 2);
}

test "Petri Migration: Transition firing" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Firing Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input", null);
    _ = try net.addPlace("p2", "Output", null);
    _ = try net.addTransition("t1", "Process", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    // Add token to enable transition
    try net.addTokenToPlace("p1", "{}");
    
    // Check enabled
    try std.testing.expect(net.isTransitionEnabled("t1"));
    
    // Fire transition
    try net.fireTransition("t1");
    
    // Verify token moved
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), marking.get("p1"));
    try std.testing.expectEqual(@as(usize, 1), marking.get("p2"));
}

test "Petri Migration: Deadlock detection" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Deadlock Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Place 1", null);
    _ = try net.addTransition("t1", "Trans 1", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    
    // No tokens, should be deadlocked
    try std.testing.expect(net.isDeadlocked());
    
    // Add token, should not be deadlocked
    try net.addTokenToPlace("p1", "{}");
    try std.testing.expect(!net.isDeadlocked());
}

test "Petri Migration: Statistics" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Stats Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Place 1", null);
    _ = try net.addPlace("p2", "Place 2", null);
    _ = try net.addTransition("t1", "Trans 1", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    
    try net.addTokenToPlace("p1", "{}");
    try net.addTokenToPlace("p1", "{}");
    
    const stats = net.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.place_count);
    try std.testing.expectEqual(@as(usize, 1), stats.transition_count);
    try std.testing.expectEqual(@as(usize, 1), stats.arc_count);
    try std.testing.expectEqual(@as(usize, 2), stats.total_tokens);
}

test "Petri Migration: Multiple enabled transitions" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Multiple Trans Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Place 1", null);
    _ = try net.addPlace("p2", "Place 2", null);
    _ = try net.addTransition("t1", "Trans 1", 0);
    _ = try net.addTransition("t2", "Trans 2", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .input, 1, "p2", "t2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var enabled = try net.getEnabledTransitions();
    defer enabled.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), enabled.items.len);
    try std.testing.expectEqualStrings("t1", enabled.items[0]);
}

test "Petri Migration: Inhibitor arc (zig-libc feature)" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Inhibitor Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input", null);
    _ = try net.addPlace("p2", "Inhibitor", null);
    _ = try net.addPlace("p3", "Output", null);
    _ = try net.addTransition("t1", "Trans", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .inhibitor, 1, "p2", "t1");
    _ = try net.addArc("a3", .output, 1, "t1", "p3");
    
    // Add token to p1, transition should be enabled
    try net.addTokenToPlace("p1", "{}");
    try std.testing.expect(net.isTransitionEnabled("t1"));
    
    // Add token to inhibitor place, transition should be disabled
    try net.addTokenToPlace("p2", "{}");
    try std.testing.expect(!net.isTransitionEnabled("t1"));
}

test "Petri Migration: Marking equality" {
    const allocator = std.testing.allocator;
    
    var m1 = petri_net.Marking.init(allocator);
    defer m1.deinit();
    var m2 = petri_net.Marking.init(allocator);
    defer m2.deinit();
    
    try m1.set("p1", 2);
    try m1.set("p2", 1);
    try m2.set("p1", 2);
    try m2.set("p2", 1);
    
    try std.testing.expect(m1.equals(&m2));
    
    try m2.set("p2", 3);
    try std.testing.expect(!m1.equals(&m2));
}

test "Petri Migration: Thread safety smoke test" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Thread Safety Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Shared Place", null);
    
    // Add multiple tokens (backend should be thread-safe with RwLock)
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try net.addTokenToPlace("p1", "{}");
    }
    
    const stats = net.getStats();
    try std.testing.expectEqual(@as(usize, 10), stats.total_tokens);
}
