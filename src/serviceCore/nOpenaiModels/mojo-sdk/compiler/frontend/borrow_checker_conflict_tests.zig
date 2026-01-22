// Comprehensive Conflict Detection Tests for BorrowChecker
// Phase 3: Test all borrow rule combinations and conflict detection

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
const BorrowRules = borrow_checker.BorrowRules;

// ============================================================================
// Shared Borrow Tests
// ============================================================================

test "Conflict: multiple shared borrows allowed" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const state_path = try BorrowPath.init(allocator, "x");
    try scope.createState(state_path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Add 3 shared borrows - all should succeed
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var path = try BorrowPath.init(allocator, "x");
        defer path.deinit(allocator);
        
        const borrow = Borrow.init(.Shared, lt, path, .{ .line = @intCast(i + 1), .column = 1 });
        try testing.expect(checker.checkBorrow(borrow));
        try checker.addBorrow(borrow);
    }
    
    try testing.expectEqual(@as(usize, 0), checker.errors.items.len);
}

test "Conflict: shared borrow of different variables allowed" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const x_path = try BorrowPath.init(allocator, "x");
    try scope.createState(x_path);
    
    const y_path = try BorrowPath.init(allocator, "y");
    try scope.createState(y_path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Borrow both x and y as shared
    var x_borrow_path = try BorrowPath.init(allocator, "x");
    defer x_borrow_path.deinit(allocator);
    const x_borrow = Borrow.init(.Shared, lt, x_borrow_path, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(x_borrow));
    try checker.addBorrow(x_borrow);
    
    var y_borrow_path = try BorrowPath.init(allocator, "y");
    defer y_borrow_path.deinit(allocator);
    const y_borrow = Borrow.init(.Shared, lt, y_borrow_path, .{ .line = 2, .column = 1 });
    try testing.expect(checker.checkBorrow(y_borrow));
    try checker.addBorrow(y_borrow);
    
    try testing.expectEqual(@as(usize, 0), checker.errors.items.len);
}

// ============================================================================
// Mutable Borrow Tests
// ============================================================================

test "Conflict: mutable borrow blocks shared borrow" {
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
    var mut_path = try BorrowPath.init(allocator, "x");
    defer mut_path.deinit(allocator);
    const mut_borrow = Borrow.init(.Mutable, lt, mut_path, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(mut_borrow));
    try checker.addBorrow(mut_borrow);
    
    // Try shared borrow - should fail
    var shared_path = try BorrowPath.init(allocator, "x");
    defer shared_path.deinit(allocator);
    const shared_borrow = Borrow.init(.Shared, lt, shared_path, .{ .line = 2, .column = 1 });
    try testing.expect(!checker.checkBorrow(shared_borrow));
    
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
    try testing.expectEqual(BorrowChecker.BorrowError.ErrorKind.SharedBorrowWhileMutable, checker.errors.items[0].kind);
}

test "Conflict: shared borrow blocks mutable borrow" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Add shared borrow
    var shared_path = try BorrowPath.init(allocator, "x");
    defer shared_path.deinit(allocator);
    const shared_borrow = Borrow.init(.Shared, lt, shared_path, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(shared_borrow));
    try checker.addBorrow(shared_borrow);
    
    // Try mutable borrow - should fail
    var mut_path = try BorrowPath.init(allocator, "x");
    defer mut_path.deinit(allocator);
    const mut_borrow = Borrow.init(.Mutable, lt, mut_path, .{ .line = 2, .column = 1 });
    try testing.expect(!checker.checkBorrow(mut_borrow));
    
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
    try testing.expectEqual(BorrowChecker.BorrowError.ErrorKind.MutableBorrowWhileShared, checker.errors.items[0].kind);
}

test "Conflict: double mutable borrow forbidden" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Add first mutable borrow
    var mut_path1 = try BorrowPath.init(allocator, "x");
    defer mut_path1.deinit(allocator);
    const mut_borrow1 = Borrow.init(.Mutable, lt, mut_path1, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(mut_borrow1));
    try checker.addBorrow(mut_borrow1);
    
    // Try second mutable borrow - should fail
    var mut_path2 = try BorrowPath.init(allocator, "x");
    defer mut_path2.deinit(allocator);
    const mut_borrow2 = Borrow.init(.Mutable, lt, mut_path2, .{ .line = 2, .column = 1 });
    try testing.expect(!checker.checkBorrow(mut_borrow2));
    
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
    try testing.expectEqual(BorrowChecker.BorrowError.ErrorKind.DoubleMutableBorrow, checker.errors.items[0].kind);
}

test "Conflict: mutable borrow of different variables allowed" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const x_path = try BorrowPath.init(allocator, "x");
    try scope.createState(x_path);
    
    const y_path = try BorrowPath.init(allocator, "y");
    try scope.createState(y_path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Borrow both x and y as mutable - different variables, should work
    var x_borrow_path = try BorrowPath.init(allocator, "x");
    defer x_borrow_path.deinit(allocator);
    const x_borrow = Borrow.init(.Mutable, lt, x_borrow_path, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(x_borrow));
    try checker.addBorrow(x_borrow);
    
    var y_borrow_path = try BorrowPath.init(allocator, "y");
    defer y_borrow_path.deinit(allocator);
    const y_borrow = Borrow.init(.Mutable, lt, y_borrow_path, .{ .line = 2, .column = 1 });
    try testing.expect(checker.checkBorrow(y_borrow));
    try checker.addBorrow(y_borrow);
    
    try testing.expectEqual(@as(usize, 0), checker.errors.items.len);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

test "Conflict: borrow after move forbidden" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Move the value
    var move_path = try BorrowPath.init(allocator, "x");
    defer move_path.deinit(allocator);
    const move_borrow = Borrow.init(.Owned, lt, move_path, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(move_borrow));
    try checker.addBorrow(move_borrow);
    
    // Try to borrow after move - should fail
    var borrow_path = try BorrowPath.init(allocator, "x");
    defer borrow_path.deinit(allocator);
    const borrow = Borrow.init(.Shared, lt, borrow_path, .{ .line = 2, .column = 1 });
    try testing.expect(!checker.checkBorrow(borrow));
    
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
    try testing.expectEqual(BorrowChecker.BorrowError.ErrorKind.UseAfterMove, checker.errors.items[0].kind);
}

test "Conflict: move while borrowed forbidden" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Borrow the value
    var borrow_path = try BorrowPath.init(allocator, "x");
    defer borrow_path.deinit(allocator);
    const borrow = Borrow.init(.Shared, lt, borrow_path, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(borrow));
    try checker.addBorrow(borrow);
    
    // Try to move while borrowed - should fail
    var move_path = try BorrowPath.init(allocator, "x");
    defer move_path.deinit(allocator);
    const move_borrow = Borrow.init(.Owned, lt, move_path, .{ .line = 2, .column = 1 });
    try testing.expect(!checker.checkBorrow(move_borrow));
    
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
    try testing.expectEqual(BorrowChecker.BorrowError.ErrorKind.MovedValueBorrowed, checker.errors.items[0].kind);
}

test "Conflict: double move forbidden" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // First move
    var move_path1 = try BorrowPath.init(allocator, "x");
    defer move_path1.deinit(allocator);
    const move1 = Borrow.init(.Owned, lt, move_path1, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(move1));
    try checker.addBorrow(move1);
    
    // Second move - should fail
    var move_path2 = try BorrowPath.init(allocator, "x");
    defer move_path2.deinit(allocator);
    const move2 = Borrow.init(.Owned, lt, move_path2, .{ .line = 2, .column = 1 });
    try testing.expect(!checker.checkBorrow(move2));
    
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
    try testing.expectEqual(BorrowChecker.BorrowError.ErrorKind.UseAfterMove, checker.errors.items[0].kind);
}

// ============================================================================
// Path Overlap Tests
// ============================================================================

test "Conflict: non-overlapping paths don't conflict" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "obj");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Borrow obj.field1 as mutable
    var path1 = try BorrowPath.init(allocator, "obj");
    try path1.addField(allocator, "field1");
    defer path1.deinit(allocator);
    const borrow1 = Borrow.init(.Mutable, lt, path1, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(borrow1));
    try checker.addBorrow(borrow1);
    
    // Borrow obj.field2 as mutable - different fields, should work
    var path2 = try BorrowPath.init(allocator, "obj");
    try path2.addField(allocator, "field2");
    defer path2.deinit(allocator);
    const borrow2 = Borrow.init(.Mutable, lt, path2, .{ .line = 2, .column = 1 });
    try testing.expect(checker.checkBorrow(borrow2));
    try checker.addBorrow(borrow2);
    
    try testing.expectEqual(@as(usize, 0), checker.errors.items.len);
}

test "Conflict: overlapping paths conflict" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "obj");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Borrow obj.field as mutable
    var path1 = try BorrowPath.init(allocator, "obj");
    try path1.addField(allocator, "field");
    defer path1.deinit(allocator);
    const borrow1 = Borrow.init(.Mutable, lt, path1, .{ .line = 1, .column = 1 });
    try testing.expect(checker.checkBorrow(borrow1));
    try checker.addBorrow(borrow1);
    
    // Borrow obj.field again - should fail (same path)
    var path2 = try BorrowPath.init(allocator, "obj");
    try path2.addField(allocator, "field");
    defer path2.deinit(allocator);
    const borrow2 = Borrow.init(.Shared, lt, path2, .{ .line = 2, .column = 1 });
    try testing.expect(!checker.checkBorrow(borrow2));
    
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
}

// ============================================================================
// BorrowRules Tests
// ============================================================================

test "BorrowRules: one mutable xor many shared" {
    var lt = try Lifetime.init(testing.allocator, "a", 0, .Named);
    defer lt.deinit(testing.allocator);
    
    var path = try BorrowPath.init(testing.allocator, "x");
    defer path.deinit(testing.allocator);
    
    // Only shared - OK
    var shared_only = [_]Borrow{
        Borrow.init(.Shared, lt, path, .{ .line = 1, .column = 1 }),
        Borrow.init(.Shared, lt, path, .{ .line = 2, .column = 1 }),
    };
    try testing.expect(BorrowRules.checkRule1(&shared_only));
    
    // Only one mutable - OK
    var mut_only = [_]Borrow{
        Borrow.init(.Mutable, lt, path, .{ .line = 1, .column = 1 }),
    };
    try testing.expect(BorrowRules.checkRule1(&mut_only));
    
    // Mixed - NOT OK
    var mixed = [_]Borrow{
        Borrow.init(.Mutable, lt, path, .{ .line = 1, .column = 1 }),
        Borrow.init(.Shared, lt, path, .{ .line = 2, .column = 1 }),
    };
    try testing.expect(!BorrowRules.checkRule1(&mixed));
}

test "BorrowRules: no multiple mutable borrows" {
    var lt = try Lifetime.init(testing.allocator, "a", 0, .Named);
    defer lt.deinit(testing.allocator);
    
    var path = try BorrowPath.init(testing.allocator, "x");
    defer path.deinit(testing.allocator);
    
    // One mutable - OK
    var one_mut = [_]Borrow{
        Borrow.init(.Mutable, lt, path, .{ .line = 1, .column = 1 }),
    };
    try testing.expect(BorrowRules.checkRule3(&one_mut));
    
    // Two mutable - NOT OK
    var two_mut = [_]Borrow{
        Borrow.init(.Mutable, lt, path, .{ .line = 1, .column = 1 }),
        Borrow.init(.Mutable, lt, path, .{ .line = 2, .column = 1 }),
    };
    try testing.expect(!BorrowRules.checkRule3(&two_mut));
}

test "BorrowRules: borrow lifetime must not outlive owner" {
    var owner_lt = try Lifetime.init(testing.allocator, "owner", 10, .Named);
    defer owner_lt.deinit(testing.allocator);
    
    var borrow_lt = try Lifetime.init(testing.allocator, "borrow", 5, .Named);
    defer borrow_lt.deinit(testing.allocator);
    
    var path = try BorrowPath.init(testing.allocator, "x");
    defer path.deinit(testing.allocator);
    
    const borrow = Borrow.init(.Shared, borrow_lt, path, .{ .line = 1, .column = 1 });
    
    // Borrow lifetime (5) <= owner lifetime (10) - OK
    try testing.expect(BorrowRules.checkRule2(borrow, owner_lt));
    
    // Borrow lifetime (5) > owner lifetime (3) - NOT OK
    var short_owner = try Lifetime.init(testing.allocator, "short", 3, .Named);
    defer short_owner.deinit(testing.allocator);
    try testing.expect(!BorrowRules.checkRule2(borrow, short_owner));
}

// ============================================================================
// Error Message Tests
// ============================================================================

test "Error messages contain useful information" {
    const allocator = testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "my_variable");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Create conflict
    var path1 = try BorrowPath.init(allocator, "my_variable");
    defer path1.deinit(allocator);
    const borrow1 = Borrow.init(.Mutable, lt, path1, .{ .line = 42, .column = 10 });
    try checker.addBorrow(borrow1);
    
    var path2 = try BorrowPath.init(allocator, "my_variable");
    defer path2.deinit(allocator);
    const borrow2 = Borrow.init(.Shared, lt, path2, .{ .line = 99, .column = 5 });
    _ = checker.checkBorrow(borrow2);
    
    // Check error message contains variable name and line number
    try testing.expectEqual(@as(usize, 1), checker.errors.items.len);
    const err = checker.errors.items[0];
    
    // Message should contain variable name
    try testing.expect(std.mem.indexOf(u8, err.message, "my_variable") != null);
    
    // Message should mention line 42 (where mutable borrow is)
    try testing.expect(std.mem.indexOf(u8, err.message, "42") != null);
    
    // Location should be at line 99 (where conflict occurs)
    try testing.expectEqual(@as(u32, 99), err.location.line);
}
