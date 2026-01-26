// Advanced Lifetime Features
// Day 57: Lifetime bounds on generics, HRTB, lifetime variance

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const lifetimes = @import("lifetimes.zig");
const Lifetime = lifetimes.Lifetime;
const LifetimeConstraint = lifetimes.LifetimeConstraint;

// ============================================================================
// Lifetime Bounds on Generic Types
// ============================================================================

/// Generic type parameter with lifetime bounds
pub const GenericTypeParam = struct {
    name: []const u8,
    lifetime_bounds: ArrayList(Lifetime), // T: 'a + 'b
    trait_bounds: ArrayList([]const u8), // T: Display + Clone
    
    pub fn init(allocator: Allocator, name: []const u8) !GenericTypeParam {
        return GenericTypeParam{
            .name = try allocator.dupe(u8, name),
            .lifetime_bounds = ArrayList(Lifetime){},
            .trait_bounds = ArrayList([]const u8){},
        };
    }
    
    pub fn deinit(self: *GenericTypeParam, allocator: Allocator) void {
        allocator.free(self.name);
        self.lifetime_bounds.deinit(allocator);
        self.trait_bounds.deinit(allocator);
    }
    
    pub fn addLifetimeBound(self: *GenericTypeParam, allocator: Allocator, lifetime: Lifetime) !void {
        try self.lifetime_bounds.append(allocator, lifetime);
    }
    
    pub fn hasLifetimeBound(self: GenericTypeParam, lifetime: Lifetime) bool {
        for (self.lifetime_bounds.items) |bound| {
            if (bound.id == lifetime.id) {
                return true;
            }
        }
        return false;
    }
};

/// Struct with lifetime-bounded generic parameters
pub const GenericStruct = struct {
    name: []const u8,
    lifetime_params: ArrayList(Lifetime),
    type_params: ArrayList(GenericTypeParam),
    
    pub fn init(allocator: Allocator, name: []const u8) !GenericStruct {
        return GenericStruct{
            .name = try allocator.dupe(u8, name),
            .lifetime_params = ArrayList(Lifetime){},
            .type_params = ArrayList(GenericTypeParam){},
        };
    }
    
    pub fn deinit(self: *GenericStruct, allocator: Allocator) void {
        allocator.free(self.name);
        self.lifetime_params.deinit(allocator);
        for (self.type_params.items) |*param| {
            param.deinit(allocator);
        }
        self.type_params.deinit(allocator);
    }
    
    pub fn addLifetimeParam(self: *GenericStruct, allocator: Allocator, lifetime: Lifetime) !void {
        try self.lifetime_params.append(allocator, lifetime);
    }
    
    pub fn addTypeParam(self: *GenericStruct, allocator: Allocator, param: GenericTypeParam) !void {
        try self.type_params.append(allocator, param);
    }
};

// ============================================================================
// Higher-Rank Trait Bounds (HRTB)
// ============================================================================

/// Higher-rank trait bound: for<'a> T: Trait<'a>
pub const HigherRankBound = struct {
    quantified_lifetimes: ArrayList(Lifetime), // for<'a, 'b>
    trait_name: []const u8,
    trait_lifetimes: ArrayList(Lifetime), // Lifetimes used in trait
    
    pub fn init(allocator: Allocator, trait_name: []const u8) !HigherRankBound {
        return HigherRankBound{
            .quantified_lifetimes = ArrayList(Lifetime){},
            .trait_name = try allocator.dupe(u8, trait_name),
            .trait_lifetimes = ArrayList(Lifetime){},
        };
    }
    
    pub fn deinit(self: *HigherRankBound, allocator: Allocator) void {
        self.quantified_lifetimes.deinit(allocator);
        allocator.free(self.trait_name);
        self.trait_lifetimes.deinit(allocator);
    }
    
    pub fn addQuantifiedLifetime(self: *HigherRankBound, allocator: Allocator, lifetime: Lifetime) !void {
        try self.quantified_lifetimes.append(allocator, lifetime);
    }
    
    pub fn format(
        self: HigherRankBound,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        
        try writer.print("for<", .{});
        for (self.quantified_lifetimes.items, 0..) |lt, i| {
            if (i > 0) try writer.print(", ", .{});
            try writer.print("{any}", .{lt});
        }
        try writer.print("> {s}", .{self.trait_name});
    }
};

// ============================================================================
// Lifetime Variance
// ============================================================================

/// Variance describes how lifetime subtyping works
pub const Variance = enum {
    Covariant, // Can substitute longer lifetime for shorter
    Contravariant, // Can substitute shorter lifetime for longer (function args)
    Invariant, // No substitution allowed
    Bivariant, // Both covariant and contravariant (rare)
    
    pub fn format(
        self: Variance,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        
        const str = switch (self) {
            .Covariant => "covariant",
            .Contravariant => "contravariant",
            .Invariant => "invariant",
            .Bivariant => "bivariant",
        };
        try writer.print("{s}", .{str});
    }
};

/// Tracks variance for type parameters and lifetimes
pub const VarianceTracker = struct {
    allocator: Allocator,
    lifetime_variance: std.AutoHashMap(u32, Variance), // lifetime_id -> variance
    type_variance: std.StringHashMap(Variance), // type_name -> variance
    
    pub fn init(allocator: Allocator) VarianceTracker {
        return VarianceTracker{
            .allocator = allocator,
            .lifetime_variance = std.AutoHashMap(u32, Variance).init(allocator),
            .type_variance = std.StringHashMap(Variance).init(allocator),
        };
    }
    
    pub fn deinit(self: *VarianceTracker) void {
        self.lifetime_variance.deinit();
        self.type_variance.deinit();
    }
    
    pub fn setLifetimeVariance(self: *VarianceTracker, lifetime_id: u32, variance: Variance) !void {
        try self.lifetime_variance.put(lifetime_id, variance);
    }
    
    pub fn getLifetimeVariance(self: *VarianceTracker, lifetime_id: u32) ?Variance {
        return self.lifetime_variance.get(lifetime_id);
    }
    
    pub fn canSubstitute(
        self: *VarianceTracker,
        lifetime_id: u32,
        from: Lifetime,
        to: Lifetime,
    ) bool {
        const variance = self.getLifetimeVariance(lifetime_id) orelse return false;
        
        switch (variance) {
            .Covariant => {
                // Can use longer lifetime where shorter expected
                return from.id >= to.id or from.isStatic();
            },
            .Contravariant => {
                // Can use shorter lifetime where longer expected
                return from.id <= to.id;
            },
            .Invariant => {
                // Must be exact match
                return from.id == to.id;
            },
            .Bivariant => {
                // Any substitution allowed
                return true;
            },
        }
    }
};

// ============================================================================
// Lifetime Subtyping
// ============================================================================

/// Lifetime subtyping rules
pub const LifetimeSubtyping = struct {
    allocator: Allocator,
    variance_tracker: *VarianceTracker,
    
    pub fn init(allocator: Allocator, variance_tracker: *VarianceTracker) LifetimeSubtyping {
        return LifetimeSubtyping{
            .allocator = allocator,
            .variance_tracker = variance_tracker,
        };
    }
    
    /// Check if type1 is a subtype of type2 considering lifetimes
    /// For references: &'a T <: &'b T iff 'a outlives 'b (covariant)
    pub fn isSubtype(
        self: *LifetimeSubtyping,
        type1: TypeWithLifetime,
        type2: TypeWithLifetime,
    ) bool {
        // Check base type compatibility
        if (!std.mem.eql(u8, type1.base_name, type2.base_name)) {
            return false;
        }
        
        // Check lifetime relationship based on variance
        if (type1.lifetime) |lt1| {
            if (type2.lifetime) |lt2| {
                _ = self.variance_tracker.getLifetimeVariance(lt1.id) orelse .Covariant;
                return self.variance_tracker.canSubstitute(lt1.id, lt1, lt2);
            }
        }
        
        return true;
    }
    
    pub const TypeWithLifetime = struct {
        base_name: []const u8,
        lifetime: ?Lifetime,
    };
};

// ============================================================================
// Lifetime Elision Rules (Extended)
// ============================================================================

/// Advanced lifetime elision patterns
pub const ElisionRules = struct {
    /// Check if lifetimes can be elided for this function signature
    pub fn canElide(params: []const ParamInfo, return_has_ref: bool) bool {
        if (!return_has_ref) {
            // No lifetime needed if return type has no references
            return true;
        }
        
        var ref_count: usize = 0;
        var has_self: bool = false;
        
        for (params) |param| {
            if (param.is_reference) {
                ref_count += 1;
                if (param.is_self) {
                    has_self = true;
                }
            }
        }
        
        // Can elide if:
        // 1. Exactly one input reference, or
        // 2. Has &self/&mut self
        return ref_count == 1 or has_self;
    }
    
    pub const ParamInfo = struct {
        is_reference: bool,
        is_self: bool,
    };
};

// ============================================================================
// Lifetime Bounds Checker
// ============================================================================

/// Validates lifetime bounds on generic types
pub const BoundsChecker = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) BoundsChecker {
        return BoundsChecker{
            .allocator = allocator,
        };
    }
    
    /// Check if type satisfies lifetime bounds
    pub fn checkBounds(
        self: *BoundsChecker,
        type_param: GenericTypeParam,
        actual_lifetime: Lifetime,
    ) !bool {
        _ = self;
        // Check if actual_lifetime outlives all required bounds
        for (type_param.lifetime_bounds.items) |bound| {
            // Would check: actual_lifetime: bound
            // For now, simplified check
            if (actual_lifetime.id < bound.id and !actual_lifetime.isStatic()) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "generic type with lifetime bounds" {
    const allocator = std.testing.allocator;
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    var param = try GenericTypeParam.init(allocator, "T");
    defer param.deinit(allocator);
    
    try param.addLifetimeBound(allocator, lt_a);
    
    try std.testing.expect(param.hasLifetimeBound(lt_a));
}

test "higher rank trait bound" {
    const allocator = std.testing.allocator;
    
    var hrtb = try HigherRankBound.init(allocator, "Fn");
    defer hrtb.deinit(allocator);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    try hrtb.addQuantifiedLifetime(allocator, lt_a);
    
    try std.testing.expectEqual(@as(usize, 1), hrtb.quantified_lifetimes.items.len);
}

test "variance covariant" {
    const allocator = std.testing.allocator;
    
    var tracker = VarianceTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.setLifetimeVariance(0, .Covariant);
    
    const variance = tracker.getLifetimeVariance(0);
    try std.testing.expect(variance != null);
    try std.testing.expectEqual(Variance.Covariant, variance.?);
}

test "variance contravariant substitution" {
    const allocator = std.testing.allocator;
    
    var tracker = VarianceTracker.init(allocator);
    defer tracker.deinit();
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    const lt_b = try Lifetime.init(allocator, "b", 1, .Named);
    defer {
        allocator.free(lt_a.name);
        allocator.free(lt_b.name);
    }
    
    try tracker.setLifetimeVariance(0, .Contravariant);
    
    // Contravariant: can substitute shorter for longer
    try std.testing.expect(tracker.canSubstitute(0, lt_a, lt_b));
}

test "lifetime subtyping" {
    const allocator = std.testing.allocator;
    
    var tracker = VarianceTracker.init(allocator);
    defer tracker.deinit();
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    // Set up variance for the lifetime
    try tracker.setLifetimeVariance(0, .Covariant);
    
    var subtyping = LifetimeSubtyping.init(allocator, &tracker);
    
    const type1 = LifetimeSubtyping.TypeWithLifetime{
        .base_name = "i32",
        .lifetime = lt_a,
    };
    
    const type2 = LifetimeSubtyping.TypeWithLifetime{
        .base_name = "i32",
        .lifetime = lt_a,
    };
    
    try std.testing.expect(subtyping.isSubtype(type1, type2));
}

test "elision rules single reference" {
    var params = [_]ElisionRules.ParamInfo{
        .{ .is_reference = true, .is_self = false },
    };
    
    try std.testing.expect(ElisionRules.canElide(&params, true));
}

test "elision rules with self" {
    var params = [_]ElisionRules.ParamInfo{
        .{ .is_reference = true, .is_self = true },
        .{ .is_reference = true, .is_self = false },
    };
    
    try std.testing.expect(ElisionRules.canElide(&params, true));
}

test "bounds checker" {
    const allocator = std.testing.allocator;
    
    var checker = BoundsChecker.init(allocator);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    const lt_b = try Lifetime.init(allocator, "b", 1, .Named);
    defer {
        allocator.free(lt_a.name);
        allocator.free(lt_b.name);
    }
    
    var param = try GenericTypeParam.init(allocator, "T");
    defer param.deinit(allocator);
    
    try param.addLifetimeBound(allocator, lt_b);
    
    // lt_a (id=0) < lt_b (id=1), so should fail
    try std.testing.expect(!try checker.checkBounds(param, lt_a));
}

test "generic struct with lifetimes" {
    const allocator = std.testing.allocator;
    
    var strukt = try GenericStruct.init(allocator, "Container");
    defer strukt.deinit(allocator);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    try strukt.addLifetimeParam(allocator, lt_a);
    
    var type_param = try GenericTypeParam.init(allocator, "T");
    try type_param.addLifetimeBound(allocator, lt_a);
    
    try strukt.addTypeParam(allocator, type_param);
    
    try std.testing.expectEqual(@as(usize, 1), strukt.lifetime_params.items.len);
    try std.testing.expectEqual(@as(usize, 1), strukt.type_params.items.len);
}
