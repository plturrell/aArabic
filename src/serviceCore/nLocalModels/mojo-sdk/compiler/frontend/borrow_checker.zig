// Borrow Checker - Rust-style Memory Safety
// Day 59: Borrow tracking, mutable vs immutable, basic rules

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const lifetimes = @import("lifetimes.zig");
const patterns = @import("lifetime_patterns.zig");
const Lifetime = lifetimes.Lifetime;
const Type = patterns.Type;

// ============================================================================
// Borrow Types & Representation
// ============================================================================

/// Types of borrows in the system
pub const BorrowKind = enum {
    Shared, // Immutable borrow (&T)
    Mutable, // Mutable borrow (&mut T)
    Owned, // Owned value (moved)
    
    pub fn format(
        self: BorrowKind,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        const str = switch (self) {
            .Shared => "shared",
            .Mutable => "mutable",
            .Owned => "owned",
        };
        try writer.print("{s}", .{str});
    }
};

/// Represents a borrow of a value
pub const Borrow = struct {
    kind: BorrowKind,
    lifetime: Lifetime,
    path: BorrowPath, // What is being borrowed
    location: SourceLocation,
    
    pub const SourceLocation = struct {
        line: u32,
        column: u32,
    };
    
    pub fn init(kind: BorrowKind, lifetime: Lifetime, path: BorrowPath, location: SourceLocation) Borrow {
        return Borrow{
            .kind = kind,
            .lifetime = lifetime,
            .path = path,
            .location = location,
        };
    }
    
    pub fn deinit(self: *Borrow, allocator: Allocator) void {
        // Only cleanup the path - lifetime is borrowed from scope
        self.path.deinit(allocator);
    }
};

/// Path to a borrowed value (e.g., x, x.field, x[0])
pub const BorrowPath = struct {
    root: []const u8, // Variable name
    projections: ArrayList(Projection),
    
    pub const Projection = union(enum) {
        Field: []const u8, // .field
        Index: u32, // [index]
        Deref, // *ptr
    };
    
    pub fn init(allocator: Allocator, root: []const u8) !BorrowPath {
        return BorrowPath{
            .root = try allocator.dupe(u8, root),
            .projections = ArrayList(Projection){},
        };
    }
    
    pub fn deinit(self: *BorrowPath, allocator: Allocator) void {
        allocator.free(self.root);
        for (self.projections.items) |proj| {
            switch (proj) {
                .Field => |field| allocator.free(field),
                else => {},
            }
        }
        self.projections.deinit(allocator);
    }
    
    pub fn addField(self: *BorrowPath, allocator: Allocator, field: []const u8) !void {
        const duped = try allocator.dupe(u8, field);
        try self.projections.append(allocator, .{ .Field = duped });
    }
    
    pub fn overlaps(self: BorrowPath, other: BorrowPath) bool {
        // Check if two paths overlap (one is prefix of other)
        if (!std.mem.eql(u8, self.root, other.root)) {
            return false;
        }
        
        const min_len = @min(self.projections.items.len, other.projections.items.len);
        var i: usize = 0;
        while (i < min_len) : (i += 1) {
            const p1 = self.projections.items[i];
            const p2 = other.projections.items[i];
            
            switch (p1) {
                .Field => |f1| switch (p2) {
                    .Field => |f2| if (!std.mem.eql(u8, f1, f2)) return false,
                    else => return false,
                },
                .Index => |idx1| switch (p2) {
                    .Index => |idx2| if (idx1 != idx2) return false,
                    else => return false,
                },
                .Deref => switch (p2) {
                    .Deref => {},
                    else => return false,
                },
            }
        }
        
        return true;
    }
};

// ============================================================================
// Borrow State Tracking
// ============================================================================

/// Tracks the borrowing state of a variable
pub const BorrowState = struct {
    path: BorrowPath,
    active_borrows: ArrayList(Borrow),
    is_moved: bool,
    move_location: ?Borrow.SourceLocation,
    
    pub fn init(allocator: Allocator, path: BorrowPath) BorrowState {
        _ = allocator;
        return BorrowState{
            .path = path,
            .active_borrows = ArrayList(Borrow){},
            .is_moved = false,
            .move_location = null,
        };
    }
    
    pub fn deinit(self: *BorrowState, allocator: Allocator) void {
        self.path.deinit(allocator);
        // Clean up each borrow's path before freeing the list
        for (self.active_borrows.items) |*borrow| {
            borrow.deinit(allocator);
        }
        self.active_borrows.deinit(allocator);
    }
};

/// Tracks all borrow states in current scope
pub const BorrowScope = struct {
    allocator: Allocator,
    states: StringHashMap(BorrowState), // var_name -> state
    parent: ?*BorrowScope,
    
    pub fn init(allocator: Allocator, parent: ?*BorrowScope) BorrowScope {
        return BorrowScope{
            .allocator = allocator,
            .states = StringHashMap(BorrowState).init(allocator),
            .parent = parent,
        };
    }
    
    pub fn deinit(self: *BorrowScope) void {
        var it = self.states.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.states.deinit();
    }
    
    pub fn getState(self: *BorrowScope, var_name: []const u8) ?*BorrowState {
        if (self.states.getPtr(var_name)) |state| {
            return state;
        }
        if (self.parent) |parent| {
            return parent.getState(var_name);
        }
        return null;
    }
    
    pub fn createState(self: *BorrowScope, path: BorrowPath) !void {
        const key = try self.allocator.dupe(u8, path.root);
        try self.states.put(key, BorrowState.init(self.allocator, path));
    }
};

// ============================================================================
// Borrow Checker Core
// ============================================================================

/// Main borrow checker
pub const BorrowChecker = struct {
    allocator: Allocator,
    current_scope: *BorrowScope,
    errors: ArrayList(BorrowError),
    
    pub const BorrowError = struct {
        kind: ErrorKind,
        message: []const u8,
        location: Borrow.SourceLocation,
        
        pub const ErrorKind = enum {
            MutableBorrowWhileShared,
            SharedBorrowWhileMutable,
            UseAfterMove,
            DoubleMutableBorrow,
            MovedValueBorrowed,
            BorrowOutlivesOwner,
        };
    };
    
    pub fn init(allocator: Allocator, scope: *BorrowScope) BorrowChecker {
        return BorrowChecker{
            .allocator = allocator,
            .current_scope = scope,
            .errors = ArrayList(BorrowError){},
        };
    }
    
    pub fn deinit(self: *BorrowChecker) void {
        for (self.errors.items) |err| {
            self.allocator.free(err.message);
        }
        self.errors.deinit(self.allocator);
    }
    
    pub fn checkBorrow(
        self: *BorrowChecker,
        borrow: Borrow,
    ) bool {
        const state = self.current_scope.getState(borrow.path.root) orelse {
            // Variable not found - should be caught by earlier passes
            return true;
        };
        
        // Rule 0: Cannot borrow moved value
        if (state.is_moved) {
            self.errors.append(self.allocator, BorrowError{
                .kind = .UseAfterMove,
                .message = std.fmt.allocPrint(
                    self.allocator,
                    "Cannot borrow '{s}' as it was moved at line {d}",
                    .{ borrow.path.root, state.move_location.?.line },
                ) catch @panic("OOM during error reporting"),
                .location = borrow.location,
            }) catch @panic("OOM during error append");
            return false;
        }
        
        // Check against active borrows
        for (state.active_borrows.items) |active_borrow| {
            if (!borrow.path.overlaps(active_borrow.path)) {
                continue; // Different paths don't conflict
            }
            
            switch (borrow.kind) {
                .Mutable => {
                    // Rule 1: Cannot take mutable borrow while any borrow exists
                    if (active_borrow.kind == .Mutable or active_borrow.kind == .Shared) {
                        self.errors.append(self.allocator, BorrowError{
                            .kind = if (active_borrow.kind == .Mutable)
                                .DoubleMutableBorrow
                            else
                                .MutableBorrowWhileShared,
                            .message = std.fmt.allocPrint(
                                self.allocator,
                                "Cannot borrow '{s}' as mutable because it is already borrowed as {any} at line {d}",
                                .{ borrow.path.root, active_borrow.kind, active_borrow.location.line },
                            ) catch @panic("OOM during error reporting"),
                            .location = borrow.location,
                        }) catch @panic("OOM during error append");
                        return false;
                    }
                },
                .Shared => {
                    // Rule 2: Cannot take shared borrow while mutable borrow exists
                    if (active_borrow.kind == .Mutable) {
                        self.errors.append(self.allocator, BorrowError{
                            .kind = .SharedBorrowWhileMutable,
                            .message = std.fmt.allocPrint(
                                self.allocator,
                                "Cannot borrow '{s}' as shared because it is already borrowed as mutable at line {d}",
                                .{ borrow.path.root, active_borrow.location.line },
                            ) catch @panic("OOM during error reporting"),
                            .location = borrow.location,
                        }) catch @panic("OOM during error append");
                        return false;
                    }
                    // Multiple shared borrows are OK
                },
                .Owned => {
                    // Moving while borrowed
                    self.errors.append(self.allocator, BorrowError{
                        .kind = .MovedValueBorrowed,
                        .message = std.fmt.allocPrint(
                            self.allocator,
                            "Cannot move '{s}' because it is borrowed as {any} at line {d}",
                            .{ borrow.path.root, active_borrow.kind, active_borrow.location.line },
                        ) catch @panic("OOM during error reporting"),
                            .location = borrow.location,
                        }) catch @panic("OOM during error append");
                        return false;
                },
            }
        }
        return true;
    }
                        
    /// Add a borrow to tracking
    pub fn addBorrow(self: *BorrowChecker, borrow: Borrow) !void {
        const state = self.current_scope.getState(borrow.path.root) orelse return;
        
        if (borrow.kind == .Owned) {
            state.is_moved = true;
            state.move_location = borrow.location;
        } else {
            // Make a copy of the path - we need to own it
            var path_copy = try BorrowPath.init(self.allocator, borrow.path.root);
            for (borrow.path.projections.items) |proj| {
                switch (proj) {
                    .Field => |field| try path_copy.addField(self.allocator, field),
                    .Index => |idx| try path_copy.projections.append(self.allocator, .{ .Index = idx }),
                    .Deref => try path_copy.projections.append(self.allocator, .Deref),
                }
            }
            
            const borrow_copy = Borrow{
                .kind = borrow.kind,
                .lifetime = borrow.lifetime,
                .path = path_copy,
                .location = borrow.location,
            };
            try state.active_borrows.append(self.allocator, borrow_copy);
        }
    }
    
    /// Remove borrows that have ended (lifetime expired)
    pub fn endBorrow(self: *BorrowChecker, path: BorrowPath, lifetime: Lifetime) !void {
        const state = self.current_scope.getState(path.root) orelse return;
        
        var i: usize = 0;
        while (i < state.active_borrows.items.len) {
            if (state.active_borrows.items[i].lifetime.id == lifetime.id) {
                _ = state.active_borrows.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }
    
    /// Check if a path can be accessed (not borrowed mutably)
    pub fn checkAccess(self: *BorrowChecker, path: BorrowPath, location: Borrow.SourceLocation) !bool {
        const state = self.current_scope.getState(path.root) orelse return true;
        
        if (state.is_moved) {
            try self.errors.append(self.allocator, BorrowError{
                .kind = .UseAfterMove,
                .message = try std.fmt.allocPrint(
                    self.allocator,
                    "Cannot access '{s}' as it was moved at line {d}",
                    .{ path.root, state.move_location.?.line },
                ),
                .location = location,
            });
            return false;
        }
        
        // Check for conflicting mutable borrows
        for (state.active_borrows.items) |active| {
            if (active.kind == .Mutable and path.overlaps(active.path)) {
                try self.errors.append(self.allocator, BorrowError{
                    .kind = .SharedBorrowWhileMutable,
                    .message = try std.fmt.allocPrint(
                        self.allocator,
                        "Cannot access '{s}' because it is mutably borrowed at line {d}",
                        .{ path.root, active.location.line },
                    ),
                    .location = location,
                });
                return false;
            }
        }
        
        return true;
    }
};

// ============================================================================
// Borrow Rules Enforcer
// ============================================================================

/// Enforces the fundamental borrow rules
pub const BorrowRules = struct {
    /// Rule: At any given time, you can have EITHER
    /// - One mutable reference, OR
    /// - Any number of immutable references
    pub fn checkRule1(borrows: []const Borrow) bool {
        var has_mutable = false;
        var has_shared = false;
        
        for (borrows) |b| {
            switch (b.kind) {
                .Mutable => has_mutable = true,
                .Shared => has_shared = true,
                .Owned => {},
            }
        }
        
        return !(has_mutable and has_shared);
    }
    
    /// Rule: References must always be valid (no dangling)
    pub fn checkRule2(borrow: Borrow, owner_lifetime: Lifetime) bool {
        // Borrow lifetime must not outlive owner
        return borrow.lifetime.id <= owner_lifetime.id or owner_lifetime.isStatic();
    }
    
    /// Rule: Cannot have multiple mutable borrows
    pub fn checkRule3(borrows: []const Borrow) bool {
        var mutable_count: usize = 0;
        
        for (borrows) |b| {
            if (b.kind == .Mutable) {
                mutable_count += 1;
            }
        }
        
        return mutable_count <= 1;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "borrow kind formatting" {
    var buffer: [100]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    
    const kind = BorrowKind.Shared;
    try kind.format("", .{}, stream.writer());
    try std.testing.expectEqualStrings("shared", stream.getWritten());
}

test "borrow path creation" {
    const allocator = std.testing.allocator;
    
    var path = try BorrowPath.init(allocator, "x");
    defer path.deinit(allocator);
    
    try std.testing.expectEqualStrings("x", path.root);
    try std.testing.expectEqual(@as(usize, 0), path.projections.items.len);
}

test "borrow path with field" {
    const allocator = std.testing.allocator;
    
    const path = try BorrowPath.init(allocator, "x");
    var mut_path = path;
    defer mut_path.deinit(allocator);
    
    try mut_path.addField(allocator, "field");
    
    try std.testing.expectEqual(@as(usize, 1), mut_path.projections.items.len);
}

test "borrow path overlap detection" {
    const allocator = std.testing.allocator;
    
    var path1 = try BorrowPath.init(allocator, "x");
    defer path1.deinit(allocator);
    
    var path2 = try BorrowPath.init(allocator, "x");
    defer path2.deinit(allocator);
    
    try std.testing.expect(path1.overlaps(path2));
    
    var path3 = try BorrowPath.init(allocator, "y");
    defer path3.deinit(allocator);
    
    try std.testing.expect(!path1.overlaps(path3));
}

test "borrow scope state management" {
    const allocator = std.testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    const state = scope.getState("x");
    try std.testing.expect(state != null);
    try std.testing.expectEqualStrings("x", state.?.path.root);
}

test "borrow checker shared borrow ok" {
    const allocator = std.testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
        var path1 = try BorrowPath.init(allocator, "x");    defer path1.deinit(allocator);
    
    const borrow1 = Borrow{
        .kind = .Shared,
        .lifetime = lt,
        .path = path1,
        .location = .{ .line = 1, .column = 1 },
    };
    
    try std.testing.expect(checker.checkBorrow(borrow1));
}

test "borrow checker mutable while shared error" {
    const allocator = std.testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    var path1 = try BorrowPath.init(allocator, "x");
    defer path1.deinit(allocator);
    const borrow1 = Borrow{
        .kind = .Shared,
        .lifetime = lt,
        .path = path1,
        .location = .{ .line = 1, .column = 1 },
    };
    try checker.addBorrow(borrow1);
    
    // Try mutable borrow - should fail
    var path2 = try BorrowPath.init(allocator, "x");
    defer path2.deinit(allocator);
    
    const borrow2 = Borrow{
        .kind = .Mutable,
        .lifetime = lt,
        .path = path2,
        .location = .{ .line = 2, .column = 1 },
    };
    
    try std.testing.expect(!checker.checkBorrow(borrow2));
    try std.testing.expectEqual(@as(usize, 1), checker.errors.items.len);
}

test "borrow rules: one mutable or many shared" {
    var lt = try Lifetime.init(std.testing.allocator, "a", 0, .Named);
    defer lt.deinit(std.testing.allocator);
    
    var path = try BorrowPath.init(std.testing.allocator, "x");
    defer path.deinit(std.testing.allocator);
    
    var borrows = [_]Borrow{
        .{
            .kind = .Shared,
            .lifetime = lt,
            .path = path,
            .location = .{ .line = 1, .column = 1 },
        },
        .{
            .kind = .Shared,
            .lifetime = lt,
            .path = path,
            .location = .{ .line = 2, .column = 1 },
        },
    };
    
    // Multiple shared OK
    try std.testing.expect(BorrowRules.checkRule1(&borrows));
}

test "borrow rules: no mixed mutable and shared" {
    var lt = try Lifetime.init(std.testing.allocator, "a", 0, .Named);
    defer lt.deinit(std.testing.allocator);
    
    var path = try BorrowPath.init(std.testing.allocator, "x");
    defer path.deinit(std.testing.allocator);
    
    var borrows = [_]Borrow{
        .{
            .kind = .Mutable,
            .lifetime = lt,
            .path = path,
            .location = .{ .line = 1, .column = 1 },
        },
        .{
            .kind = .Shared,
            .lifetime = lt,
            .path = path,
            .location = .{ .line = 2, .column = 1 },
        },
    };
    
    // Mixed mutable and shared NOT OK
    try std.testing.expect(!BorrowRules.checkRule1(&borrows));
}

test "use after move detection" {
    const allocator = std.testing.allocator;
    
    var scope = BorrowScope.init(allocator, null);
    defer scope.deinit();
    
    const path = try BorrowPath.init(allocator, "x");
    try scope.createState(path);
    
    var checker = BorrowChecker.init(allocator, &scope);
    defer checker.deinit();
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    // Move value
    var path1 = try BorrowPath.init(allocator, "x");
    defer path1.deinit(allocator);
    const move_borrow = Borrow{
        .kind = .Owned,
        .lifetime = lt,
        .path = path1,
        .location = .{ .line = 1, .column = 1 },
    };
    try checker.addBorrow(move_borrow);
    
    // Try to borrow after move - should fail
    var path2 = try BorrowPath.init(allocator, "x");
    defer path2.deinit(allocator);
    
    const after_move = Borrow{
        .kind = .Shared,
        .lifetime = lt,
        .path = path2,
        .location = .{ .line = 2, .column = 1 },
    };
    
    try std.testing.expect(!checker.checkBorrow(after_move));
    try std.testing.expectEqual(@as(usize, 1), checker.errors.items.len);
}
