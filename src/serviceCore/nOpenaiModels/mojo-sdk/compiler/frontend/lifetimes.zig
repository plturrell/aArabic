// Lifetime System - Lifetime Annotations and Tracking
// Day 56: Lifetime syntax, parameters, and inference

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// ============================================================================
// Lifetime Representation
// ============================================================================

/// Represents a lifetime parameter (e.g., 'a, 'b, 'static)
pub const Lifetime = struct {
    name: []const u8,
    id: u32, // Unique identifier
    kind: LifetimeKind,
    
    pub const LifetimeKind = enum {
        Named, // User-defined lifetime (e.g., 'a)
        Anonymous, // Compiler-generated lifetime (e.g., '_)
        Static, // 'static - lives for entire program
        Elided, // Inferred/elided lifetime
    };
    
    pub fn init(allocator: Allocator, name: []const u8, id: u32, kind: LifetimeKind) !Lifetime {
        return Lifetime{
            .name = try allocator.dupe(u8, name),
            .id = id,
            .kind = kind,
        };
    }
    
    pub fn deinit(self: *Lifetime, allocator: Allocator) void {
        allocator.free(self.name);
    }
    
    pub fn isStatic(self: Lifetime) bool {
        return self.kind == .Static;
    }
    
    pub fn isAnonymous(self: Lifetime) bool {
        return self.kind == .Anonymous;
    }
    
    pub fn format(
        self: Lifetime,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        
        switch (self.kind) {
            .Named => try writer.print("'{s}", .{self.name}),
            .Anonymous => try writer.print("'_", .{}),
            .Static => try writer.print("'static", .{}),
            .Elided => try writer.print("'<elided>", .{}),
        }
    }
};

/// Lifetime parameter declaration
pub const LifetimeParam = struct {
    lifetime: Lifetime,
    bounds: ArrayList(Lifetime), // Outlives constraints (e.g., 'a: 'b)
    
    pub fn init(allocator: Allocator, lifetime: Lifetime) LifetimeParam {
        return LifetimeParam{
            .lifetime = lifetime,
            .bounds = ArrayList(Lifetime).init(allocator),
        };
    }
    
    pub fn deinit(self: *LifetimeParam) void {
        self.bounds.deinit();
    }
    
    pub fn addBound(self: *LifetimeParam, bound: Lifetime) !void {
        try self.bounds.append(bound);
    }
};

// ============================================================================
// Lifetime Context & Scope
// ============================================================================

/// Lifetime scope - tracks lifetimes in current context
pub const LifetimeScope = struct {
    parent: ?*LifetimeScope,
    lifetimes: StringHashMap(Lifetime),
    next_id: u32,
    
    pub fn init(allocator: Allocator, parent: ?*LifetimeScope) LifetimeScope {
        return LifetimeScope{
            .parent = parent,
            .lifetimes = StringHashMap(Lifetime).init(allocator),
            .next_id = if (parent) |p| p.next_id else 0,
        };
    }
    
    pub fn deinit(self: *LifetimeScope) void {
        self.lifetimes.deinit();
    }
    
    pub fn declare(self: *LifetimeScope, lifetime: Lifetime) !void {
        try self.lifetimes.put(lifetime.name, lifetime);
    }
    
    pub fn lookup(self: *LifetimeScope, name: []const u8) ?Lifetime {
        if (self.lifetimes.get(name)) |lt| {
            return lt;
        }
        
        if (self.parent) |parent| {
            return parent.lookup(name);
        }
        
        return null;
    }
    
    pub fn generateAnonymous(self: *LifetimeScope, allocator: Allocator) !Lifetime {
        const id = self.next_id;
        self.next_id += 1;
        
        const name = try std.fmt.allocPrint(allocator, "anon_{d}", .{id});
        return try Lifetime.init(allocator, name, id, .Anonymous);
    }
};

// ============================================================================
// Lifetime-Annotated Types
// ============================================================================

/// Type with lifetime annotations
pub const LifetimeAnnotatedType = struct {
    base_type: *Type, // Base type (e.g., &T)
    lifetimes: ArrayList(Lifetime), // Lifetime parameters
    
    pub fn init(allocator: Allocator, base_type: *Type) LifetimeAnnotatedType {
        return LifetimeAnnotatedType{
            .base_type = base_type,
            .lifetimes = ArrayList(Lifetime).init(allocator),
        };
    }
    
    pub fn deinit(self: *LifetimeAnnotatedType) void {
        self.lifetimes.deinit();
    }
    
    pub fn addLifetime(self: *LifetimeAnnotatedType, lifetime: Lifetime) !void {
        try self.lifetimes.append(lifetime);
    }
    
    pub fn format(
        self: LifetimeAnnotatedType,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        
        if (self.lifetimes.items.len > 0) {
            try writer.print("&<", .{});
            for (self.lifetimes.items, 0..) |lt, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{any}", .{lt});
            }
            try writer.print(">", .{});
        }
        try writer.print("{s}", .{self.base_type.name});
    }
};

// Placeholder Type struct (would reference actual type system)
const Type = struct {
    name: []const u8,
    kind: TypeKind,
    
    const TypeKind = enum {
        Int,
        Float,
        Bool,
        String,
        Reference,
        Pointer,
        Array,
        Struct,
    };
};

// ============================================================================
// Lifetime Constraints & Relations
// ============================================================================

/// Lifetime constraint: 'a outlives 'b (written as 'a: 'b)
pub const LifetimeConstraint = struct {
    longer: Lifetime, // Lifetime that must outlive
    shorter: Lifetime, // Lifetime that is outlived
    
    pub fn init(longer: Lifetime, shorter: Lifetime) LifetimeConstraint {
        return LifetimeConstraint{
            .longer = longer,
            .shorter = shorter,
        };
    }
    
    pub fn format(
        self: LifetimeConstraint,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{any} : {any}", .{ self.longer, self.shorter });
    }
};

/// Tracks lifetime relationships and constraints
pub const LifetimeConstraintGraph = struct {
    allocator: Allocator,
    constraints: ArrayList(LifetimeConstraint),
    
    pub fn init(allocator: Allocator) LifetimeConstraintGraph {
        return LifetimeConstraintGraph{
            .allocator = allocator,
            .constraints = ArrayList(LifetimeConstraint){},
        };
    }
    
    pub fn deinit(self: *LifetimeConstraintGraph) void {
        self.constraints.deinit(self.allocator);
    }
    
    pub fn addConstraint(self: *LifetimeConstraintGraph, constraint: LifetimeConstraint) !void {
        try self.constraints.append(self.allocator, constraint);
    }
    
    pub fn outlives(self: *LifetimeConstraintGraph, longer: Lifetime, shorter: Lifetime) bool {
        // Check if longer outlives shorter (directly or transitively)
        
        // Direct check
        for (self.constraints.items) |c| {
            if (c.longer.id == longer.id and c.shorter.id == shorter.id) {
                return true;
            }
        }
        
        // Static outlives everything
        if (longer.isStatic()) {
            return true;
        }
        
        // Transitive check (simplified - real implementation would use graph traversal)
        for (self.constraints.items) |c1| {
            if (c1.longer.id == longer.id) {
                for (self.constraints.items) |c2| {
                    if (c2.shorter.id == shorter.id and c1.shorter.id == c2.longer.id) {
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    pub fn checkConsistency(self: *LifetimeConstraintGraph) !bool {
        // Check for cycles in constraint graph
        // This is a simplified check - real implementation would use proper cycle detection
        
        for (self.constraints.items) |c1| {
            for (self.constraints.items) |c2| {
                // Check for direct cycle: 'a: 'b and 'b: 'a
                if (c1.longer.id == c2.shorter.id and c1.shorter.id == c2.longer.id) {
                    return false;
                }
            }
        }
        
        return true;
    }
};

// ============================================================================
// Lifetime Inference
// ============================================================================

/// Infers elided lifetimes based on rules
pub const LifetimeInference = struct {
    allocator: Allocator,
    scope: *LifetimeScope,
    constraints: *LifetimeConstraintGraph,
    
    pub fn init(
        allocator: Allocator,
        scope: *LifetimeScope,
        constraints: *LifetimeConstraintGraph,
    ) LifetimeInference {
        return LifetimeInference{
            .allocator = allocator,
            .scope = scope,
            .constraints = constraints,
        };
    }
    
    /// Infer lifetime for function with elided lifetimes
    /// Rules (similar to Rust):
    /// 1. Each elided input lifetime becomes a distinct lifetime
    /// 2. If there's exactly one input lifetime, it's assigned to all elided outputs
    /// 3. If there's a &self or &mut self, its lifetime is assigned to all elided outputs
    pub fn inferFunctionLifetimes(
        self: *LifetimeInference,
        params: []const FunctionParam,
        return_type: *Type,
    ) !InferredLifetimes {
        _ = return_type;
        var inferred = InferredLifetimes.init(self.allocator);
        
        // Count reference parameters with lifetimes
        var ref_param_count: usize = 0;
        var last_ref_lifetime: ?Lifetime = null;
        var self_lifetime: ?Lifetime = null;
        
        for (params) |param| {
            if (param.is_reference) {
                ref_param_count += 1;
                if (param.lifetime) |lt| {
                    last_ref_lifetime = lt;
                    if (param.is_self) {
                        self_lifetime = lt;
                    }
                }
            }
        }
        
        // Apply inference rules
        if (self_lifetime) |lt| {
            // Rule 3: Use &self lifetime for output
            try inferred.output_lifetime.append(self.allocator, lt);
        } else if (ref_param_count == 1 and last_ref_lifetime != null) {
            // Rule 2: Single input lifetime -> output lifetime
            try inferred.output_lifetime.append(self.allocator, last_ref_lifetime.?);
        } else {
            // Generate fresh lifetime for output
            const fresh = try self.scope.generateAnonymous(self.allocator);
            try inferred.output_lifetime.append(self.allocator, fresh);
        }
        
        return inferred;
    }
    
    pub const InferredLifetimes = struct {
        allocator: Allocator,
        output_lifetime: ArrayList(Lifetime),
        
        pub fn init(allocator: Allocator) InferredLifetimes {
            return InferredLifetimes{
                .allocator = allocator,
                .output_lifetime = ArrayList(Lifetime){},
            };
        }
        
        pub fn deinit(self: *InferredLifetimes) void {
            self.output_lifetime.deinit(self.allocator);
        }
    };
    
    const FunctionParam = struct {
        name: []const u8,
        is_reference: bool,
        is_self: bool,
        lifetime: ?Lifetime,
    };
};

// ============================================================================
// Lifetime Checker
// ============================================================================

/// Validates lifetime annotations and checks for errors
pub const LifetimeChecker = struct {
    allocator: Allocator,
    constraints: *LifetimeConstraintGraph,
    errors: ArrayList(LifetimeError),
    
    pub fn init(allocator: Allocator, constraints: *LifetimeConstraintGraph) LifetimeChecker {
        return LifetimeChecker{
            .allocator = allocator,
            .constraints = constraints,
            .errors = ArrayList(LifetimeError){},
        };
    }
    
    pub fn deinit(self: *LifetimeChecker) void {
        self.errors.deinit(self.allocator);
    }
    
    pub fn checkAssignment(
        self: *LifetimeChecker,
        target_lifetime: Lifetime,
        source_lifetime: Lifetime,
    ) !bool {
        // Target must outlive source
        if (!self.constraints.outlives(target_lifetime, source_lifetime)) {
            try self.errors.append(self.allocator, LifetimeError{
                .kind = .OutlivesViolation,
                .message = try std.fmt.allocPrint(
                    self.allocator,
                    "Lifetime {any} does not outlive {any}",
                    .{ source_lifetime, target_lifetime },
                ),
            });
            return false;
        }
        return true;
    }
    
    pub fn checkFunctionCall(
        self: *LifetimeChecker,
        arg_lifetimes: []const Lifetime,
        param_lifetimes: []const Lifetime,
    ) !bool {
        if (arg_lifetimes.len != param_lifetimes.len) {
            try self.errors.append(self.allocator, LifetimeError{
                .kind = .ArgumentCountMismatch,
                .message = try std.fmt.allocPrint(
                    self.allocator,
                    "Expected {} lifetime arguments, found {}",
                    .{ param_lifetimes.len, arg_lifetimes.len },
                ),
            });
            return false;
        }
        
        for (arg_lifetimes, param_lifetimes) |arg_lt, param_lt| {
            if (!try self.checkAssignment(param_lt, arg_lt)) {
                return false;
            }
        }
        
        return true;
    }
    
    pub const LifetimeError = struct {
        kind: ErrorKind,
        message: []const u8,
        
        const ErrorKind = enum {
            OutlivesViolation,
            ArgumentCountMismatch,
            UndefinedLifetime,
            CyclicConstraint,
        };
    };
};

// ============================================================================
// Tests
// ============================================================================

test "lifetime creation" {
    const allocator = std.testing.allocator;
    
    var lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer lt.deinit(allocator);
    
    try std.testing.expectEqualStrings("a", lt.name);
    try std.testing.expectEqual(@as(u32, 0), lt.id);
    try std.testing.expectEqual(Lifetime.LifetimeKind.Named, lt.kind);
}

test "lifetime scope" {
    const allocator = std.testing.allocator;
    
    var scope = LifetimeScope.init(allocator, null);
    defer scope.deinit();
    
    const lt = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt.name);
    try scope.declare(lt);
    
    const found = scope.lookup("a");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("a", found.?.name);
}

test "lifetime constraints" {
    const allocator = std.testing.allocator;
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    const lt_b = try Lifetime.init(allocator, "b", 1, .Named);
    
    defer {
        allocator.free(lt_a.name);
        allocator.free(lt_b.name);
    }
    
    var graph = LifetimeConstraintGraph.init(allocator);
    defer graph.deinit();
    
    const constraint = LifetimeConstraint.init(lt_a, lt_b);
    try graph.addConstraint(constraint);
    
    try std.testing.expect(graph.outlives(lt_a, lt_b));
    try std.testing.expect(!graph.outlives(lt_b, lt_a));
}

test "static lifetime outlives everything" {
    const allocator = std.testing.allocator;
    
    const lt_static = try Lifetime.init(allocator, "static", 0, .Static);
    const lt_a = try Lifetime.init(allocator, "a", 1, .Named);
    
    defer {
        allocator.free(lt_static.name);
        allocator.free(lt_a.name);
    }
    
    var graph = LifetimeConstraintGraph.init(allocator);
    defer graph.deinit();
    
    try std.testing.expect(graph.outlives(lt_static, lt_a));
}

test "lifetime inference single input" {
    const allocator = std.testing.allocator;
    
    var scope = LifetimeScope.init(allocator, null);
    defer scope.deinit();
    
    var graph = LifetimeConstraintGraph.init(allocator);
    defer graph.deinit();
    
    var inference = LifetimeInference.init(allocator, &scope, &graph);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    var params = [_]LifetimeInference.FunctionParam{
        .{
            .name = "x",
            .is_reference = true,
            .is_self = false,
            .lifetime = lt_a,
        },
    };
    
    var dummy_type = Type{ .name = "i32", .kind = .Int };
    var inferred = try inference.inferFunctionLifetimes(&params, &dummy_type);
    defer inferred.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), inferred.output_lifetime.items.len);
    try std.testing.expectEqual(lt_a.id, inferred.output_lifetime.items[0].id);
}

test "lifetime checker assignment" {
    const allocator = std.testing.allocator;
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    const lt_b = try Lifetime.init(allocator, "b", 1, .Named);
    
    defer {
        allocator.free(lt_a.name);
        allocator.free(lt_b.name);
    }
    
    var graph = LifetimeConstraintGraph.init(allocator);
    defer graph.deinit();
    
    // 'a outlives 'b
    try graph.addConstraint(LifetimeConstraint.init(lt_a, lt_b));
    
    var checker = LifetimeChecker.init(allocator, &graph);
    defer {
        // Free error messages
        for (checker.errors.items) |err| {
            allocator.free(err.message);
        }
        checker.deinit();
    }
    
    // Can assign 'b to 'a (since 'a outlives 'b)
    try std.testing.expect(try checker.checkAssignment(lt_a, lt_b));
    
    // Cannot assign 'a to 'b (would violate constraint)
    try std.testing.expect(!try checker.checkAssignment(lt_b, lt_a));
}
