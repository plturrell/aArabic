// Mojo SDK - Memory Management
// Day 26: Ownership, borrowing, lifetimes, move semantics, and RAII

const std = @import("std");

// ============================================================================
// Ownership System
// ============================================================================

pub const Ownership = enum {
    Owned,      // Exclusive ownership
    Borrowed,   // Shared read access
    Mutable,    // Exclusive write access
};

pub const Owner = struct {
    name: []const u8,
    owned_values: std.StringHashMap(ValueInfo),
    allocator: std.mem.Allocator,
    
    pub const ValueInfo = struct {
        value_type: []const u8,
        ownership: Ownership,
        moved: bool = false,
    };
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Owner {
        return Owner{
            .name = name,
            .owned_values = std.StringHashMap(ValueInfo).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addValue(self: *Owner, value_name: []const u8, value_type: []const u8) !void {
        try self.owned_values.put(value_name, ValueInfo{
            .value_type = value_type,
            .ownership = .Owned,
        });
    }
    
    pub fn moveValue(self: *Owner, value_name: []const u8) !void {
        if (self.owned_values.getPtr(value_name)) |info| {
            if (info.moved) {
                return error.UseAfterMove;
            }
            info.moved = true;
        } else {
            return error.ValueNotFound;
        }
    }
    
    pub fn hasValue(self: *const Owner, value_name: []const u8) bool {
        if (self.owned_values.get(value_name)) |info| {
            return !info.moved;
        }
        return false;
    }
    
    pub fn deinit(self: *Owner) void {
        self.owned_values.deinit();
    }
};

// ============================================================================
// Borrowing and References
// ============================================================================

pub const Borrow = struct {
    value_name: []const u8,
    borrow_type: BorrowType,
    lifetime: Lifetime,
    
    pub const BorrowType = enum {
        Immutable,  // &T
        Mutable,    // &mut T
    };
    
    pub fn init(value_name: []const u8, borrow_type: BorrowType, lifetime: Lifetime) Borrow {
        return Borrow{
            .value_name = value_name,
            .borrow_type = borrow_type,
            .lifetime = lifetime,
        };
    }
    
    pub fn isImmutable(self: *const Borrow) bool {
        return self.borrow_type == .Immutable;
    }
    
    pub fn isMutable(self: *const Borrow) bool {
        return self.borrow_type == .Mutable;
    }
};

pub const BorrowChecker = struct {
    active_borrows: std.ArrayList(Borrow),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) BorrowChecker {
        return BorrowChecker{
            .active_borrows = std.ArrayList(Borrow){},
            .allocator = allocator,
        };
    }
    
    pub fn addBorrow(self: *BorrowChecker, borrow: Borrow) !void {
        // Check borrow rules
        for (self.active_borrows.items) |existing| {
            if (std.mem.eql(u8, existing.value_name, borrow.value_name)) {
                // Cannot have mutable borrow if any other borrow exists
                if (borrow.borrow_type == .Mutable or existing.borrow_type == .Mutable) {
                    return error.BorrowConflict;
                }
            }
        }
        try self.active_borrows.append(self.allocator, borrow);
    }
    
    pub fn removeBorrow(self: *BorrowChecker, value_name: []const u8) void {
        var i: usize = 0;
        while (i < self.active_borrows.items.len) {
            if (std.mem.eql(u8, self.active_borrows.items[i].value_name, value_name)) {
                _ = self.active_borrows.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }
    
    pub fn canBorrow(self: *const BorrowChecker, value_name: []const u8, borrow_type: Borrow.BorrowType) bool {
        for (self.active_borrows.items) |existing| {
            if (std.mem.eql(u8, existing.value_name, value_name)) {
                if (borrow_type == .Mutable or existing.borrow_type == .Mutable) {
                    return false;
                }
            }
        }
        return true;
    }
    
    pub fn deinit(self: *BorrowChecker) void {
        self.active_borrows.deinit(self.allocator);
    }
};

// ============================================================================
// Lifetime Tracking
// ============================================================================

pub const Lifetime = struct {
    name: []const u8,
    start_scope: usize,
    end_scope: usize,
    
    pub fn init(name: []const u8, start: usize, end: usize) Lifetime {
        return Lifetime{
            .name = name,
            .start_scope = start,
            .end_scope = end,
        };
    }
    
    pub fn contains(self: *const Lifetime, scope: usize) bool {
        return scope >= self.start_scope and scope < self.end_scope;
    }
    
    pub fn outlives(self: *const Lifetime, other: *const Lifetime) bool {
        return self.start_scope <= other.start_scope and self.end_scope >= other.end_scope;
    }
};

pub const LifetimeTracker = struct {
    lifetimes: std.StringHashMap(Lifetime),
    current_scope: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) LifetimeTracker {
        return LifetimeTracker{
            .lifetimes = std.StringHashMap(Lifetime).init(allocator),
            .current_scope = 0,
            .allocator = allocator,
        };
    }
    
    pub fn enterScope(self: *LifetimeTracker) void {
        self.current_scope += 1;
    }
    
    pub fn exitScope(self: *LifetimeTracker) void {
        if (self.current_scope > 0) {
            self.current_scope -= 1;
        }
    }
    
    pub fn addLifetime(self: *LifetimeTracker, name: []const u8, end_scope: usize) !void {
        const lifetime = Lifetime.init(name, self.current_scope, end_scope);
        try self.lifetimes.put(name, lifetime);
    }
    
    pub fn getLifetime(self: *const LifetimeTracker, name: []const u8) ?Lifetime {
        return self.lifetimes.get(name);
    }
    
    pub fn checkLifetime(self: *const LifetimeTracker, name: []const u8) bool {
        if (self.getLifetime(name)) |lifetime| {
            return lifetime.contains(self.current_scope);
        }
        return false;
    }
    
    pub fn deinit(self: *LifetimeTracker) void {
        self.lifetimes.deinit();
    }
};

// ============================================================================
// Move Semantics
// ============================================================================

pub const MoveSemantics = struct {
    moved_values: std.StringHashMap(MoveInfo),
    allocator: std.mem.Allocator,
    
    pub const MoveInfo = struct {
        from_owner: []const u8,
        to_owner: []const u8,
        move_scope: usize,
    };
    
    pub fn init(allocator: std.mem.Allocator) MoveSemantics {
        return MoveSemantics{
            .moved_values = std.StringHashMap(MoveInfo).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn recordMove(self: *MoveSemantics, value: []const u8, from: []const u8, to: []const u8, scope: usize) !void {
        try self.moved_values.put(value, MoveInfo{
            .from_owner = from,
            .to_owner = to,
            .move_scope = scope,
        });
    }
    
    pub fn isMoved(self: *const MoveSemantics, value: []const u8) bool {
        return self.moved_values.contains(value);
    }
    
    pub fn getNewOwner(self: *const MoveSemantics, value: []const u8) ?[]const u8 {
        if (self.moved_values.get(value)) |info| {
            return info.to_owner;
        }
        return null;
    }
    
    pub fn deinit(self: *MoveSemantics) void {
        self.moved_values.deinit();
    }
};

pub const CopySemantics = struct {
    copyable_types: std.StringHashMap(bool),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) CopySemantics {
        return CopySemantics{
            .copyable_types = std.StringHashMap(bool).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn registerCopyable(self: *CopySemantics, type_name: []const u8) !void {
        try self.copyable_types.put(type_name, true);
    }
    
    pub fn isCopyable(self: *const CopySemantics, type_name: []const u8) bool {
        return self.copyable_types.get(type_name) orelse false;
    }
    
    pub fn deinit(self: *CopySemantics) void {
        self.copyable_types.deinit();
    }
};

// ============================================================================
// RAII Patterns
// ============================================================================

pub const Resource = struct {
    name: []const u8,
    resource_type: ResourceType,
    acquired: bool,
    released: bool,
    
    pub const ResourceType = enum {
        Memory,
        File,
        Socket,
        Lock,
        Custom,
    };
    
    pub fn init(name: []const u8, resource_type: ResourceType) Resource {
        return Resource{
            .name = name,
            .resource_type = resource_type,
            .acquired = false,
            .released = false,
        };
    }
    
    pub fn acquire(self: *Resource) !void {
        if (self.acquired and !self.released) {
            return error.AlreadyAcquired;
        }
        self.acquired = true;
        self.released = false;
    }
    
    pub fn release(self: *Resource) !void {
        if (!self.acquired) {
            return error.NotAcquired;
        }
        if (self.released) {
            return error.AlreadyReleased;
        }
        self.released = true;
    }
    
    pub fn isLeaked(self: *const Resource) bool {
        return self.acquired and !self.released;
    }
};

pub const RAIITracker = struct {
    resources: std.ArrayList(Resource),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) RAIITracker {
        return RAIITracker{
            .resources = std.ArrayList(Resource){},
            .allocator = allocator,
        };
    }
    
    pub fn trackResource(self: *RAIITracker, resource: Resource) !void {
        try self.resources.append(self.allocator, resource);
    }
    
    pub fn acquireResource(self: *RAIITracker, name: []const u8) !void {
        for (self.resources.items) |*res| {
            if (std.mem.eql(u8, res.name, name)) {
                try res.acquire();
                return;
            }
        }
        return error.ResourceNotFound;
    }
    
    pub fn releaseResource(self: *RAIITracker, name: []const u8) !void {
        for (self.resources.items) |*res| {
            if (std.mem.eql(u8, res.name, name)) {
                try res.release();
                return;
            }
        }
        return error.ResourceNotFound;
    }
    
    pub fn checkLeaks(self: *const RAIITracker, allocator: std.mem.Allocator) std.ArrayList([]const u8) {
        var leaks = std.ArrayList([]const u8){};
        for (self.resources.items) |res| {
            if (res.isLeaked()) {
                leaks.append(allocator, res.name) catch {};
            }
        }
        return leaks;
    }
    
    pub fn deinit(self: *RAIITracker) void {
        self.resources.deinit(self.allocator);
    }
};

// ============================================================================
// Memory Safety
// ============================================================================

pub const SafetyChecker = struct {
    owner_tracker: Owner,
    borrow_checker: BorrowChecker,
    lifetime_tracker: LifetimeTracker,
    move_semantics: MoveSemantics,
    raii_tracker: RAIITracker,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) SafetyChecker {
        return SafetyChecker{
            .owner_tracker = Owner.init(allocator, "main"),
            .borrow_checker = BorrowChecker.init(allocator),
            .lifetime_tracker = LifetimeTracker.init(allocator),
            .move_semantics = MoveSemantics.init(allocator),
            .raii_tracker = RAIITracker.init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn checkSafety(self: *SafetyChecker) !bool {
        // Check for moved values
        var moved_count: usize = 0;
        var iter = self.move_semantics.moved_values.iterator();
        while (iter.next()) |_| {
            moved_count += 1;
        }
        
        // Check for resource leaks
        var leaks = self.raii_tracker.checkLeaks(self.allocator);
        defer leaks.deinit(self.allocator);
        
        return leaks.items.len == 0;
    }
    
    pub fn deinit(self: *SafetyChecker) void {
        self.owner_tracker.deinit();
        self.borrow_checker.deinit();
        self.lifetime_tracker.deinit();
        self.move_semantics.deinit();
        self.raii_tracker.deinit();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "memory: ownership" {
    const allocator = std.testing.allocator;
    var owner = Owner.init(allocator, "test");
    defer owner.deinit();
    
    try owner.addValue("x", "Int");
    try std.testing.expect(owner.hasValue("x"));
}

test "memory: move semantics" {
    const allocator = std.testing.allocator;
    var owner = Owner.init(allocator, "test");
    defer owner.deinit();
    
    try owner.addValue("x", "String");
    try owner.moveValue("x");
    
    try std.testing.expect(!owner.hasValue("x"));
}

test "memory: borrow checker immutable" {
    const allocator = std.testing.allocator;
    var checker = BorrowChecker.init(allocator);
    defer checker.deinit();
    
    const lifetime = Lifetime.init("a", 0, 10);
    const borrow1 = Borrow.init("x", .Immutable, lifetime);
    const borrow2 = Borrow.init("x", .Immutable, lifetime);
    
    try checker.addBorrow(borrow1);
    try checker.addBorrow(borrow2);  // Multiple immutable borrows OK
}

test "memory: borrow checker mutable conflict" {
    const allocator = std.testing.allocator;
    var checker = BorrowChecker.init(allocator);
    defer checker.deinit();
    
    const lifetime = Lifetime.init("a", 0, 10);
    const borrow1 = Borrow.init("x", .Immutable, lifetime);
    const borrow2 = Borrow.init("x", .Mutable, lifetime);
    
    try checker.addBorrow(borrow1);
    
    const result = checker.addBorrow(borrow2);
    try std.testing.expectError(error.BorrowConflict, result);
}

test "memory: lifetime tracking" {
    const allocator = std.testing.allocator;
    var tracker = LifetimeTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.addLifetime("a", 5);
    tracker.enterScope();
    
    try std.testing.expect(tracker.checkLifetime("a"));
}

test "memory: lifetime outlives" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    const outer = Lifetime.init("outer", 0, 10);
    const inner = Lifetime.init("inner", 2, 8);
    
    try std.testing.expect(outer.outlives(&inner));
    try std.testing.expect(!inner.outlives(&outer));
}

test "memory: resource acquisition" {
    const allocator = std.testing.allocator;
    var tracker = RAIITracker.init(allocator);
    defer tracker.deinit();
    
    const resource = Resource.init("file", .File);
    try tracker.trackResource(resource);
    try tracker.acquireResource("file");
    
    // Resource acquired
}

test "memory: resource leak detection" {
    const allocator = std.testing.allocator;
    var tracker = RAIITracker.init(allocator);
    defer tracker.deinit();
    
    var resource = Resource.init("memory", .Memory);
    try resource.acquire();
    try tracker.trackResource(resource);
    
    var leaks = tracker.checkLeaks(allocator);
    defer leaks.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 1), leaks.items.len);
}

test "memory: copy semantics" {
    const allocator = std.testing.allocator;
    var copy_sem = CopySemantics.init(allocator);
    defer copy_sem.deinit();
    
    try copy_sem.registerCopyable("Int");
    
    try std.testing.expect(copy_sem.isCopyable("Int"));
    try std.testing.expect(!copy_sem.isCopyable("String"));
}

test "memory: safety checker" {
    const allocator = std.testing.allocator;
    var checker = SafetyChecker.init(allocator);
    defer checker.deinit();
    
    const is_safe = try checker.checkSafety();
    try std.testing.expect(is_safe);
}
