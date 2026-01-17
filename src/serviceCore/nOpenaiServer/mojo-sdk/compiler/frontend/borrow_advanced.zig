// Advanced Borrow Checker Features
// Day 60: Partial borrows, interior mutability, borrow splitting

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const borrow_checker = @import("borrow_checker.zig");
const lifetimes = @import("lifetimes.zig");
const BorrowKind = borrow_checker.BorrowKind;
const Borrow = borrow_checker.Borrow;
const BorrowPath = borrow_checker.BorrowPath;
const Lifetime = lifetimes.Lifetime;

// ============================================================================
// Partial Borrows (Field-Level Borrowing)
// ============================================================================

/// Allows borrowing individual struct fields independently
pub const PartialBorrowTracker = struct {
    allocator: Allocator,
    field_borrows: StringHashMap(FieldBorrowState),
    
    pub const FieldBorrowState = struct {
        struct_name: []const u8,
        field_name: []const u8,
        borrows: ArrayList(Borrow),
        
        pub fn init(allocator: Allocator, struct_name: []const u8, field_name: []const u8) !FieldBorrowState {
            return FieldBorrowState{
                .struct_name = try allocator.dupe(u8, struct_name),
                .field_name = try allocator.dupe(u8, field_name),
                .borrows = ArrayList(Borrow){},
            };
        }
        
        pub fn deinit(self: *FieldBorrowState, allocator: Allocator) void {
            allocator.free(self.struct_name);
            allocator.free(self.field_name);
            self.borrows.deinit(allocator);
        }
    };
    
    pub fn init(allocator: Allocator) PartialBorrowTracker {
        return PartialBorrowTracker{
            .allocator = allocator,
            .field_borrows = StringHashMap(FieldBorrowState).init(allocator),
        };
    }
    
    pub fn deinit(self: *PartialBorrowTracker) void {
        var it = self.field_borrows.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.field_borrows.deinit();
    }
    
    /// Check if can borrow a specific field
    pub fn canBorrowField(
        self: *PartialBorrowTracker,
        struct_var: []const u8,
        field: []const u8,
        kind: BorrowKind,
    ) bool {
        const key = self.makeKey(struct_var, field) catch return false;
        defer self.allocator.free(key);
        
        const state = self.field_borrows.get(key) orelse return true;
        
        // Check compatibility with existing field borrows
        for (state.borrows.items) |existing| {
            if (kind == .Mutable or existing.kind == .Mutable) {
                return false; // Cannot have mutable with anything else
            }
        }
        
        return true;
    }
    
    /// Add field borrow
    pub fn addFieldBorrow(
        self: *PartialBorrowTracker,
        struct_var: []const u8,
        field: []const u8,
        borrow: Borrow,
    ) !void {
        const key = try self.makeKey(struct_var, field);
        
        if (self.field_borrows.getPtr(key)) |state| {
            try state.borrows.append(self.allocator, borrow);
            self.allocator.free(key);
        } else {
            var new_state = try FieldBorrowState.init(self.allocator, struct_var, field);
            try new_state.borrows.append(self.allocator, borrow);
            try self.field_borrows.put(key, new_state);
        }
    }
    
    fn makeKey(self: *PartialBorrowTracker, struct_var: []const u8, field: []const u8) ![]const u8 {
        return try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ struct_var, field });
    }
};

// ============================================================================
// Interior Mutability (Cell/RefCell Pattern)
// ============================================================================

/// Types that allow interior mutability (like Rust's Cell/RefCell)
pub const InteriorMutability = enum {
    None, // Normal type (no interior mutability)
    Cell, // Allows mutation through shared reference
    RefCell, // Runtime borrow checking
    Atomic, // Atomic operations
    UnsafeCell, // Raw interior mutability
};

/// Tracks types with interior mutability
pub const InteriorMutabilityTracker = struct {
    allocator: Allocator,
    interior_mutable_types: StringHashMap(InteriorMutability),
    
    pub fn init(allocator: Allocator) InteriorMutabilityTracker {
        return InteriorMutabilityTracker{
            .allocator = allocator,
            .interior_mutable_types = StringHashMap(InteriorMutability).init(allocator),
        };
    }
    
    pub fn deinit(self: *InteriorMutabilityTracker) void {
        var it = self.interior_mutable_types.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.interior_mutable_types.deinit();
    }
    
    pub fn registerType(self: *InteriorMutabilityTracker, type_name: []const u8, kind: InteriorMutability) !void {
        const key = try self.allocator.dupe(u8, type_name);
        try self.interior_mutable_types.put(key, kind);
    }
    
    pub fn hasInteriorMutability(self: *InteriorMutabilityTracker, type_name: []const u8) bool {
        return self.interior_mutable_types.get(type_name) != null;
    }
    
    pub fn getKind(self: *InteriorMutabilityTracker, type_name: []const u8) ?InteriorMutability {
        return self.interior_mutable_types.get(type_name);
    }
};

// ============================================================================
// Borrow Splitting
// ============================================================================

/// Allows splitting borrows (e.g., borrowing different array slices)
pub const BorrowSplitter = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) BorrowSplitter {
        return BorrowSplitter{ .allocator = allocator };
    }
    
    /// Check if two borrows can coexist via splitting
    pub fn canSplit(self: *BorrowSplitter, borrow1: BorrowPath, borrow2: BorrowPath) bool {
        _ = self;
        
        // Different roots can always coexist
        if (!std.mem.eql(u8, borrow1.root, borrow2.root)) {
            return true;
        }
        
        // Same root but different fields -> can split
        if (borrow1.projections.items.len > 0 and borrow2.projections.items.len > 0) {
            const proj1 = borrow1.projections.items[0];
            const proj2 = borrow2.projections.items[0];
            
            switch (proj1) {
                .Field => |f1| switch (proj2) {
                    .Field => |f2| return !std.mem.eql(u8, f1, f2),
                    else => return false,
                },
                .Index => |idx1| switch (proj2) {
                    .Index => |idx2| return idx1 != idx2,
                    else => return false,
                },
                else => return false,
            }
        }
        
        return false;
    }
    
    /// Split array borrow into non-overlapping slices
    pub fn splitArrayBorrow(
        self: *BorrowSplitter,
        array_path: BorrowPath,
        range1: Range,
        range2: Range,
    ) bool {
        _ = self;
        _ = array_path;
        
        // Check if ranges don't overlap
        return range1.end <= range2.start or range2.end <= range1.start;
    }
    
    pub const Range = struct {
        start: usize,
        end: usize,
    };
};

// ============================================================================
// Move Semantics (Advanced)
// ============================================================================

/// Tracks moves and copy semantics
pub const MoveTracker = struct {
    allocator: Allocator,
    copyable_types: StringHashMap(bool), // Types that implement Copy
    moved_values: StringHashMap(MoveInfo),
    
    pub const MoveInfo = struct {
        location: Borrow.SourceLocation,
        destination: []const u8,
    };
    
    pub fn init(allocator: Allocator) MoveTracker {
        return MoveTracker{
            .allocator = allocator,
            .copyable_types = StringHashMap(bool).init(allocator),
            .moved_values = StringHashMap(MoveInfo).init(allocator),
        };
    }
    
    pub fn deinit(self: *MoveTracker) void {
        var it1 = self.copyable_types.iterator();
        while (it1.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.copyable_types.deinit();
        
        var it2 = self.moved_values.iterator();
        while (it2.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.destination);
        }
        self.moved_values.deinit();
    }
    
    pub fn registerCopyType(self: *MoveTracker, type_name: []const u8) !void {
        const key = try self.allocator.dupe(u8, type_name);
        try self.copyable_types.put(key, true);
    }
    
    pub fn isCopyable(self: *MoveTracker, type_name: []const u8) bool {
        return self.copyable_types.get(type_name) orelse false;
    }
    
    pub fn recordMove(
        self: *MoveTracker,
        var_name: []const u8,
        destination: []const u8,
        location: Borrow.SourceLocation,
    ) !void {
        const key = try self.allocator.dupe(u8, var_name);
        const dest = try self.allocator.dupe(u8, destination);
        try self.moved_values.put(key, MoveInfo{
            .location = location,
            .destination = dest,
        });
    }
    
    pub fn isMoved(self: *MoveTracker, var_name: []const u8) bool {
        return self.moved_values.contains(var_name);
    }
};

// ============================================================================
// Reborrow Support
// ============================================================================

/// Handles reborrowing (borrowing from a borrow)
pub const ReborrowTracker = struct {
    allocator: Allocator,
    reborrow_chain: ArrayList(ReborrowLink),
    
    pub const ReborrowLink = struct {
        original: BorrowPath,
        reborrow: BorrowPath,
        original_kind: BorrowKind,
        reborrow_kind: BorrowKind,
    };
    
    pub fn init(allocator: Allocator) ReborrowTracker {
        return ReborrowTracker{
            .allocator = allocator,
            .reborrow_chain = ArrayList(ReborrowLink){},
        };
    }
    
    pub fn deinit(self: *ReborrowTracker) void {
        for (self.reborrow_chain.items) |*link| {
            link.original.deinit(self.allocator);
            link.reborrow.deinit(self.allocator);
        }
        self.reborrow_chain.deinit(self.allocator);
    }
    
    /// Check if reborrow is valid
    pub fn canReborrow(
        self: *ReborrowTracker,
        original_kind: BorrowKind,
        reborrow_kind: BorrowKind,
    ) bool {
        _ = self;
        
        // Can reborrow shared from shared
        if (original_kind == .Shared and reborrow_kind == .Shared) {
            return true;
        }
        
        // Can reborrow shared from mutable
        if (original_kind == .Mutable and reborrow_kind == .Shared) {
            return true;
        }
        
        // Can reborrow mutable from mutable (but only one at a time)
        if (original_kind == .Mutable and reborrow_kind == .Mutable) {
            return true;
        }
        
        // Cannot reborrow mutable from shared
        return false;
    }
    
    pub fn addReborrow(
        self: *ReborrowTracker,
        original: BorrowPath,
        reborrow: BorrowPath,
        original_kind: BorrowKind,
        reborrow_kind: BorrowKind,
    ) !void {
        try self.reborrow_chain.append(self.allocator, ReborrowLink{
            .original = original,
            .reborrow = reborrow,
            .original_kind = original_kind,
            .reborrow_kind = reborrow_kind,
        });
    }
};

// ============================================================================
// Tests
// ============================================================================

test "partial borrow different fields" {
    const allocator = std.testing.allocator;
    
    var tracker = PartialBorrowTracker.init(allocator);
    defer tracker.deinit();
    
    // Can borrow different fields mutably
    try std.testing.expect(tracker.canBorrowField("point", "x", .Mutable));
    try std.testing.expect(tracker.canBorrowField("point", "y", .Mutable));
}

test "partial borrow same field conflict" {
    const allocator = std.testing.allocator;
    
    var tracker = PartialBorrowTracker.init(allocator);
    defer tracker.deinit();
    
    const lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt.name);
    
    var path = try BorrowPath.init(allocator, "point");
    defer path.deinit(allocator);
    
    const borrow1 = Borrow{
        .kind = .Mutable,
        .lifetime = lt,
        .path = path,
        .location = .{ .line = 1, .column = 1 },
    };
    
    try tracker.addFieldBorrow("point", "x", borrow1);
    
    // Cannot borrow same field mutably twice
    try std.testing.expect(!tracker.canBorrowField("point", "x", .Mutable));
}

test "interior mutability cell" {
    const allocator = std.testing.allocator;
    
    var tracker = InteriorMutabilityTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.registerType("Cell", .Cell);
    
    try std.testing.expect(tracker.hasInteriorMutability("Cell"));
    try std.testing.expectEqual(InteriorMutability.Cell, tracker.getKind("Cell").?);
}

test "interior mutability refcell" {
    const allocator = std.testing.allocator;
    
    var tracker = InteriorMutabilityTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.registerType("RefCell", .RefCell);
    
    const kind = tracker.getKind("RefCell");
    try std.testing.expect(kind != null);
    try std.testing.expectEqual(InteriorMutability.RefCell, kind.?);
}

test "borrow splitting different fields" {
    const allocator = std.testing.allocator;
    
    var splitter = BorrowSplitter.init(allocator);
    
    var path1 = try BorrowPath.init(allocator, "point");
    defer path1.deinit(allocator);
    try path1.addField(allocator, "x");
    
    var path2 = try BorrowPath.init(allocator, "point");
    defer path2.deinit(allocator);
    try path2.addField(allocator, "y");
    
    // Different fields can be split
    try std.testing.expect(splitter.canSplit(path1, path2));
}

test "borrow splitting same field fails" {
    const allocator = std.testing.allocator;
    
    var splitter = BorrowSplitter.init(allocator);
    
    var path1 = try BorrowPath.init(allocator, "point");
    defer path1.deinit(allocator);
    try path1.addField(allocator, "x");
    
    var path2 = try BorrowPath.init(allocator, "point");
    defer path2.deinit(allocator);
    try path2.addField(allocator, "x");
    
    // Same field cannot be split
    try std.testing.expect(!splitter.canSplit(path1, path2));
}

test "borrow splitting array ranges" {
    const allocator = std.testing.allocator;
    
    var splitter = BorrowSplitter.init(allocator);
    
    var path = try BorrowPath.init(allocator, "array");
    defer path.deinit(allocator);
    
    const range1 = BorrowSplitter.Range{ .start = 0, .end = 5 };
    const range2 = BorrowSplitter.Range{ .start = 5, .end = 10 };
    
    // Non-overlapping ranges can be split
    try std.testing.expect(splitter.splitArrayBorrow(path, range1, range2));
}

test "move tracker copyable types" {
    const allocator = std.testing.allocator;
    
    var tracker = MoveTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.registerCopyType("i32");
    
    try std.testing.expect(tracker.isCopyable("i32"));
    try std.testing.expect(!tracker.isCopyable("String"));
}

test "move tracker record move" {
    const allocator = std.testing.allocator;
    
    var tracker = MoveTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.recordMove("x", "y", .{ .line = 10, .column = 5 });
    
    try std.testing.expect(tracker.isMoved("x"));
    try std.testing.expect(!tracker.isMoved("y"));
}

test "reborrow shared from shared" {
    const allocator = std.testing.allocator;
    
    var tracker = ReborrowTracker.init(allocator);
    defer tracker.deinit();
    
    try std.testing.expect(tracker.canReborrow(.Shared, .Shared));
}

test "reborrow shared from mutable" {
    const allocator = std.testing.allocator;
    
    var tracker = ReborrowTracker.init(allocator);
    defer tracker.deinit();
    
    try std.testing.expect(tracker.canReborrow(.Mutable, .Shared));
}

test "reborrow mutable from mutable" {
    const allocator = std.testing.allocator;
    
    var tracker = ReborrowTracker.init(allocator);
    defer tracker.deinit();
    
    try std.testing.expect(tracker.canReborrow(.Mutable, .Mutable));
}

test "reborrow mutable from shared fails" {
    const allocator = std.testing.allocator;
    
    var tracker = ReborrowTracker.init(allocator);
    defer tracker.deinit();
    
    // Cannot reborrow mutable from shared
    try std.testing.expect(!tracker.canReborrow(.Shared, .Mutable));
}
