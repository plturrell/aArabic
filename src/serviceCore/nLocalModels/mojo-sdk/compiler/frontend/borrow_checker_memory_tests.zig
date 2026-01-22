// Comprehensive Memory Safety Tests for BorrowChecker
// Phase 2: Unit tests for each core struct

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const lifetimes = @import("lifetimes.zig");
const borrow_checker = @import("borrow_checker.zig");

const Lifetime = lifetimes.Lifetime;
const BorrowPath = borrow_checker.BorrowPath;
const Borrow = borrow_checker.Borrow;
const BorrowState = borrow_checker.BorrowState;
const BorrowScope = borrow_checker.BorrowScope;
const BorrowChecker = borrow_checker.BorrowChecker;
const BorrowKind = borrow_checker.BorrowKind;

// ============================================================================
// Lifetime Memory Tests
// ============================================================================

test "Lifetime: init and deinit memory management" {
    const allocator = testing.allocator;
    
    var lt = try Lifetime.init(allocator, "test_lifetime", 42, .Named);
    defer lt.deinit(allocator);
    
    try testing.expectEqualStrings("test_lifetime", lt.name);
    try testing.expectEqual(@as(u32, 42), lt.id);
}

test "Lifetime: multiple lifetimes no leaks" {
    const allocator = testing.allocator;
    
    var lt1 = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt1.deinit(allocator);
    
    var lt2 = try Lifetime.init(allocator, "b", 1, .Named);
    defer lt2.deinit(allocator);
    
    var lt3 = try Lifetime.init(allocator, "c", 2, .Static);
    defer lt3.deinit(allocator);
    
    // All should clean up without leaks
}

test "Lifetime: empty name" {
    const allocator = testing.allocator;
    
    var lt = try Lifetime.init(allocator, "", 0, .Anonymous);
    defer lt.deinit(allocator);
    
    try testing.expectEqualStrings("", lt.name);
}

test "Lifetime: long name" {
    const allocator = testing.allocator;
    
    const long_name = "very_long_lifetime_name_that_tests_memory_allocation_" ** 10;
    var lt = try Lifetime.init(allocator, long_name, 999, .Named);
    defer lt.deinit(allocator);
    
    try testing.expectEqualStrings(long_name, lt.name);
}

// ============================================================================
// BorrowPath Memory Tests
// ============================================================================

test "BorrowPath: init and deinit" {
    const allocator = testing.allocator;
    
    var path = try BorrowPath.init(allocator, "variable");
    defer path.deinit(allocator);
    
    try testing.expectEqualStrings("variable", path.root);
    try testing.expectEqual(@as(usize, 0), path.projections.items.len);
}

test "BorrowPath: with multiple fields" {
    const allocator = testing.allocator;
    
    var path = try BorrowPath.init(allocator, "obj");
    defer path.deinit(allocator);
    
    try path.addField(allocator, "field1");
    try path.addField(allocator, "field2");
    try path.addField(allocator, "field3");
    
    try testing.expectEqual(@as(usize, 3), path.projections.items.len);
}

test "BorrowPath: empty root name" {
    const allocator = testing.allocator;
    
    var path = try BorrowPath.init(allocator, "");
    defer path.deinit(allocator);
    
    try testing.expectEqualStrings("", path.root);
}

test "BorrowPath: mixed projections" {
    const allocator = testing.allocator;
    
    var path = try BorrowPath.init(allocator, "array");
    defer path.deinit(allocator);
    
    try path.addField(allocator, "items");
    try path.projections.append(allocator, .{ .Index = 5 });
    try path.projections.append(allocator, .Deref);
    
    try testing.expectEqual(@as(usize, 3), path.projections.items.len);
}

test "BorrowPath: many fields stress test" {
    const allocator = testing.allocator;
    
    var path = try BorrowPath.init(allocator, "root");
    defer path.deinit(allocator);
    
    // Add 100 fields
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const field_name = try std.fmt.allocPrint(allocator, "field_{d}", .{i});
        defer allocator.free(field_name);
        try path.addField(allocator, field_name);
    }
    
    try testing.expectEqual(@as(usize, 100), path.projections.items.len);
}

// ============================================================================
// Borrow Memory Tests
// ============================================================================

test "Borrow: init and deinit" {
    const allocator = testing.allocator;
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    const path = try BorrowPath.init(allocator, "x");
    
    var borrow = Borrow.init(.Shared, lt, path, .{ .line = 1, .column = 1 });
    defer borrow.deinit(allocator);
    
    try testing.expectEqual(BorrowKind.Shared, borrow.kind);
}

test "Borrow: multiple borrows no leaks" {
    const allocator = testing.allocator;
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    const path1 = try BorrowPath.init(allocator, "x");
    var borrow1 = Borrow.init(.Shared, lt, path1, .{ .line = 1, .column = 1 });
    defer borrow1.deinit(allocator);
    
    const path2 = try BorrowPath.init(allocator, "y");
    var borrow2 = Borrow.init(.Mutable, lt, path2, .{ .line = 2, .column = 1 });
    defer borrow2.deinit(allocator);
    
    const path3 = try BorrowPath.init(allocator, "z");
    var borrow3 = Borrow.init(.Owned, lt, path3, .{ .line = 3, .column = 1 });
    defer borrow3.deinit(allocator);
}

test "Borrow: with complex path" {
    const allocator = testing.allocator;
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    var path = try BorrowPath.init(allocator, "obj");
    try path.addField(allocator, "nested");
    try path.addField(allocator, "field");
    
    var borrow = Borrow.init(.Mutable, lt, path, .{ .line = 10, .column = 5 });
    defer borrow.deinit(allocator);
}

// ============================================================================
// BorrowState Memory Tests
// ============================================================================

test "BorrowState: init and deinit" {
    const allocator = testing.allocator;
    
    const path = try BorrowPath.init(allocator, "var");
    var state = BorrowState.init(allocator, path);
    defer state.deinit(allocator);
    
    try testing.expectEqual(false, state.is_moved);
    try testing.expectEqual(@as(usize, 0), state.active_borrows.items.len);
}

test "BorrowState: with active borrows" {
    const allocator = testing.allocator;
    
    const path = try BorrowPath.init(allocator, "x");
    var state = BorrowState.init(allocator, path);
    defer state.deinit(allocator);
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Add several borrows
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        const borrow_path = try BorrowPath.init(allocator, "x");
        const borrow = Borrow.init(.Shared, lt, borrow_path, .{ .line = @intCast(i + 1), .column = 1 });
        try state.active_borrows.append(allocator, borrow);
    }
    
    try testing.expectEqual(@as(usize, 5), state.active_borrows.items.len);
}

test "BorrowState: moved state" {
    const allocator = testing.allocator;
    
    const path = try BorrowPath.init(allocator, "moved_var");
    var state = BorrowState.init(allocator, path);
    defer state.deinit(allocator);
    
    state.is_moved = true;
    state.move_location = .{ .line = 42, .column = 10 };
    
    try testing.expect(state.is_moved);
    try testing.expectEqual(@as(u32, 42), state.move_location.?.line);
}

// ============================================================================
// BorrowScope Memory Tests
// ============================================================================

test "BorrowScope: init and deinit" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    try testing.expectEqual(@as(usize, 0), scope.states.count());
}

test "BorrowScope: create and retrieve state" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "test_var");
    try scope.createState(path);
    
    const state = scope.getState("test_var");
    try testing.expect(state != null);
    try testing.expectEqualStrings("test_var", state.?.path.root);
}

test "BorrowScope: multiple states" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    // Create 10 states
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const var_name = try std.fmt.allocPrint(allocator, "var_{d}", .{i});
        defer allocator.free(var_name);
        
        const path = try BorrowPath.init(allocator, var_name);
        try scope.createState(path);
    }
    
    try testing.expectEqual(@as(usize, 10), scope.states.count());
}

test "BorrowScope: nested scopes" {
    const allocator = testing.allocator;
    
    var parent = BorrowScope.init(allocator, null);
    defer parent.deinit();
    
    const parent_path = try BorrowPath.init(allocator, "parent_var");
    try parent.createState(parent_path);
    
    var child = BorrowScope.init(allocator, &parent);
    defer child.deinit();
    
    const child_path = try BorrowPath.init(allocator, "child_var");
    try child.createState(child_path);
    
    // Child can see parent's variables
    try testing.expect(child.getState("parent_var") != null);
    try testing.expect(child.getState("child_var") != null);
    
    // Parent cannot see child's variables
    try testing.expect(parent.getState("child_var") == null);
}

// ============================================================================
// BorrowChecker Memory Tests
// ============================================================================

test "BorrowChecker: init and deinit" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    try testing.expectEqual(@as(usize, 0), checker.errors.items.len);
}

test "BorrowChecker: addBorrow copies path correctly" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const state_path = try BorrowPath.init(allocator, "x");
    try scope.createState(state_path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Create a borrow and add it
    var borrow_path = try BorrowPath.init(allocator, "x");
    defer borrow_path.deinit(allocator); // We still own this
    
    const borrow = Borrow.init(.Shared, lt, borrow_path, .{ .line = 1, .column = 1 });
    try checker.addBorrow(borrow);
    
    // The checker should have made a copy, so we can still clean up our original
    const state = scope.getState("x");
    try testing.expectEqual(@as(usize, 1), state.?.active_borrows.items.len);
}

test "BorrowChecker: multiple borrows stress test" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const state_path = try BorrowPath.init(allocator, "x");
    try scope.createState(state_path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Add 50 shared borrows
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var borrow_path = try BorrowPath.init(allocator, "x");
        defer borrow_path.deinit(allocator);
        
        const borrow = Borrow.init(.Shared, lt, borrow_path, .{ .line = @intCast(i + 1), .column = 1 });
        try checker.addBorrow(borrow);
    }
    
    const state = scope.getState("x");
    try testing.expectEqual(@as(usize, 50), state.?.active_borrows.items.len);
}

test "BorrowChecker: error accumulation" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Add mutable borrow
    var path1 = try BorrowPath.init(allocator, "x");
    defer path1.deinit(allocator);
    const borrow1 = Borrow.init(.Mutable, lt, path1, .{ .line = 1, .column = 1 });
    try checker.addBorrow(borrow1);
    
    // Try to add another mutable - should fail
    var path2 = try BorrowPath.init(allocator, "x");
    defer path2.deinit(allocator);
    const borrow2 = Borrow.init(.Mutable, lt, path2, .{ .line = 2, .column = 1 });
    const ok = checker.checkBorrow(borrow2);
    
    try testing.expect(!ok);
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
}

// ============================================================================
// Integration Memory Tests
// ============================================================================

test "Integration: full borrow lifecycle no leaks" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    // Create multiple variables
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        const var_name = try std.fmt.allocPrint(allocator, "var_{d}", .{i});
        defer allocator.free(var_name);
        
        const path = try BorrowPath.init(allocator, var_name);
        try scope.createState(path);
    }
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Borrow each variable
    i = 0;
    while (i < 5) : (i += 1) {
        const var_name = try std.fmt.allocPrint(allocator, "var_{d}", .{i});
        defer allocator.free(var_name);
        
        var path = try BorrowPath.init(allocator, var_name);
        defer path.deinit(allocator);
        
        const borrow = Borrow.init(.Shared, lt, path, .{ .line = @intCast(i + 1), .column = 1 });
        try checker.addBorrow(borrow);
    }
    
    // All should clean up without leaks
}

test "Integration: complex nested paths" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const root_path = try BorrowPath.init(allocator, "root");
    try scope.createState(root_path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Create deeply nested paths
    var depth: usize = 0;
    while (depth < 10) : (depth += 1) {
        var path = try BorrowPath.init(allocator, "root");
        defer path.deinit(allocator);
        
        var d: usize = 0;
        while (d <= depth) : (d += 1) {
            const field = try std.fmt.allocPrint(allocator, "level_{d}", .{d});
            defer allocator.free(field);
            try path.addField(allocator, field);
        }
        
        const borrow = Borrow.init(.Shared, lt, path, .{ .line = @intCast(depth + 1), .column = 1 });
        try checker.addBorrow(borrow);
    }
}
