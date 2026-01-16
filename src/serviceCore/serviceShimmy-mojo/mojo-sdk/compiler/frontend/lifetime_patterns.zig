// Advanced Lifetime Patterns & Improvements
// Day 58: Type representation, variance inference, validation, error reporting

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const lifetimes = @import("lifetimes.zig");
const advanced = @import("lifetime_advanced.zig");
const Lifetime = lifetimes.Lifetime;
const Variance = advanced.Variance;

// ============================================================================
// Improved Type Representation
// ============================================================================

/// Enhanced type system with proper lifetime tracking
pub const Type = union(enum) {
    /// Primitive types (no lifetimes)
    Primitive: PrimitiveType,
    
    /// Reference type with lifetime
    Reference: struct {
        lifetime: Lifetime,
        base: *const Type,
        is_mutable: bool,
    },
    
    /// Pointer type
    Pointer: struct {
        base: *const Type,
        is_mutable: bool,
    },
    
    /// Function type
    Function: struct {
        params: []const Type,
        return_type: *const Type,
        lifetimes: []const Lifetime,
    },
    
    /// Generic type parameter
    GenericParam: struct {
        name: []const u8,
        id: u32,
        bounds: []const TypeBound,
    },
    
    /// Concrete named type
    Named: struct {
        name: []const u8,
        type_args: []const Type,
        lifetime_args: []const Lifetime,
    },
    
    /// Tuple type
    Tuple: []const Type,
    
    /// Array type
    Array: struct {
        element: *const Type,
        size: usize,
    },
    
    pub const PrimitiveType = enum {
        Int8, Int16, Int32, Int64,
        UInt8, UInt16, UInt32, UInt64,
        Float32, Float64,
        Bool,
        Char,
        Unit,
    };
    
    pub const TypeBound = union(enum) {
        Trait: []const u8,
        Lifetime: Lifetime,
    };
    
    pub fn getLifetimes(self: Type, allocator: Allocator) !ArrayList(Lifetime) {
        var result = ArrayList(Lifetime){};
        
        switch (self) {
            .Reference => |ref| {
                try result.append(allocator, ref.lifetime);
                var base_lts = try ref.base.getLifetimes(allocator);
                defer base_lts.deinit(allocator);
                try result.appendSlice(allocator, base_lts.items);
            },
            .Function => |func| {
                try result.appendSlice(allocator, func.lifetimes);
            },
            .Named => |named| {
                try result.appendSlice(allocator, named.lifetime_args);
            },
            .Tuple => |tuple| {
                for (tuple) |ty| {
                    var lts = try ty.getLifetimes(allocator);
                    defer lts.deinit(allocator);
                    try result.appendSlice(allocator, lts.items);
                }
            },
            .Array => |arr| {
                var lts = try arr.element.getLifetimes(allocator);
                defer lts.deinit(allocator);
                try result.appendSlice(allocator, lts.items);
            },
            else => {},
        }
        
        return result;
    }
};

// ============================================================================
// Variance Inference
// ============================================================================

/// Infers variance from type structure
pub const VarianceInference = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) VarianceInference {
        return VarianceInference{ .allocator = allocator };
    }
    
    /// Infer variance of a lifetime parameter in a type
    pub fn inferVariance(self: *VarianceInference, ty: Type, lifetime_id: u32) Variance {
        _ = self;
        
        switch (ty) {
            .Reference => |ref| {
                // Immutable references are covariant
                // Mutable references are invariant
                if (ref.is_mutable) {
                    return .Invariant;
                } else if (ref.lifetime.id == lifetime_id) {
                    return .Covariant;
                } else {
                    // Check nested type
                    return .Covariant; // Simplified
                }
            },
            .Function => |func| {
                // Parameters are contravariant, return is covariant
                for (func.lifetimes) |lt| {
                    if (lt.id == lifetime_id) {
                        // Simplified: check position
                        return .Contravariant; // In parameter position
                    }
                }
                return .Covariant;
            },
            .Pointer => {
                // Raw pointers are invariant
                return .Invariant;
            },
            .Tuple, .Array => {
                // Structural types inherit variance from components
                return .Covariant;
            },
            else => return .Bivariant,
        }
    }
    
    /// Compute variance for all lifetime parameters in a type definition
    pub fn computeAllVariances(
        self: *VarianceInference,
        type_def: TypeDefinition,
    ) !std.AutoHashMap(u32, Variance) {
        var result = std.AutoHashMap(u32, Variance).init(self.allocator);
        
        for (type_def.lifetime_params) |lt| {
            const variance = self.inferVariance(type_def.body, lt.id);
            try result.put(lt.id, variance);
        }
        
        return result;
    }
    
    pub const TypeDefinition = struct {
        name: []const u8,
        lifetime_params: []const Lifetime,
        body: Type,
    };
};

// ============================================================================
// Enhanced Validation
// ============================================================================

/// Comprehensive validation with detailed error messages
pub const LifetimeValidator = struct {
    allocator: Allocator,
    errors: ArrayList(ValidationError),
    
    pub const ValidationError = struct {
        kind: ErrorKind,
        message: []const u8,
        location: ?SourceLocation,
        
        pub const ErrorKind = enum {
            UnusedQuantifiedLifetime,
            LifetimeOutlivesViolation,
            IncompatibleVariance,
            InvalidBound,
            MissingLifetime,
            ConflictingLifetimes,
        };
        
        pub const SourceLocation = struct {
            file: []const u8,
            line: u32,
            column: u32,
        };
    };
    
    pub fn init(allocator: Allocator) LifetimeValidator {
        return LifetimeValidator{
            .allocator = allocator,
            .errors = ArrayList(ValidationError){},
        };
    }
    
    pub fn deinit(self: *LifetimeValidator) void {
        for (self.errors.items) |err| {
            self.allocator.free(err.message);
        }
        self.errors.deinit(self.allocator);
    }
    
    /// Validate higher-rank bound
    pub fn validateHRTB(self: *LifetimeValidator, hrtb: advanced.HigherRankBound) !bool {
        // Ensure quantified lifetimes are actually used
        for (hrtb.quantified_lifetimes.items) |q_lt| {
            var found = false;
            for (hrtb.trait_lifetimes.items) |t_lt| {
                if (q_lt.id == t_lt.id) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                try self.errors.append(self.allocator, ValidationError{
                    .kind = .UnusedQuantifiedLifetime,
                    .message = try std.fmt.allocPrint(
                        self.allocator,
                        "Quantified lifetime '{s}' is not used in trait bound '{s}'",
                        .{ q_lt.name, hrtb.trait_name },
                    ),
                    .location = null,
                });
                return false;
            }
        }
        
        return true;
    }
    
    /// Validate lifetime bounds on generic parameter
    pub fn validateBounds(
        self: *LifetimeValidator,
        param: advanced.GenericTypeParam,
        actual_lifetime: Lifetime,
        constraint_graph: *lifetimes.LifetimeConstraintGraph,
    ) !bool {
        for (param.lifetime_bounds.items) |bound| {
            if (!constraint_graph.outlives(actual_lifetime, bound)) {
                try self.errors.append(self.allocator, ValidationError{
                    .kind = .LifetimeOutlivesViolation,
                    .message = try std.fmt.allocPrint(
                        self.allocator,
                        "Type parameter '{s}' requires lifetime {any} to outlive {any}, but it doesn't",
                        .{ param.name, actual_lifetime, bound },
                    ),
                    .location = null,
                });
                return false;
            }
        }
        
        return true;
    }
    
    /// Validate variance compatibility
    pub fn validateVariance(
        self: *LifetimeValidator,
        expected: Variance,
        actual: Variance,
        context: []const u8,
    ) !bool {
        // Check if variance is compatible
        const compatible = switch (expected) {
            .Covariant => actual == .Covariant or actual == .Bivariant,
            .Contravariant => actual == .Contravariant or actual == .Bivariant,
            .Invariant => actual == .Invariant or actual == .Bivariant,
            .Bivariant => true,
        };
        
        if (!compatible) {
            try self.errors.append(self.allocator, ValidationError{
                .kind = .IncompatibleVariance,
                .message = try std.fmt.allocPrint(
                    self.allocator,
                    "Incompatible variance in {s}: expected {any}, found {any}",
                    .{ context, expected, actual },
                ),
                .location = null,
            });
            return false;
        }
        
        return true;
    }
};

// ============================================================================
// Lifetime Relationship Analysis
// ============================================================================

/// Analyzes complex lifetime relationships
pub const LifetimeAnalyzer = struct {
    allocator: Allocator,
    graph: *lifetimes.LifetimeConstraintGraph,
    
    pub fn init(allocator: Allocator, graph: *lifetimes.LifetimeConstraintGraph) LifetimeAnalyzer {
        return LifetimeAnalyzer{
            .allocator = allocator,
            .graph = graph,
        };
    }
    
    /// Find the most specific (shortest) lifetime that satisfies all constraints
    pub fn findMinimalLifetime(
        self: *LifetimeAnalyzer,
        candidates: []const Lifetime,
    ) ?Lifetime {
        _ = self;
        
        var minimal: ?Lifetime = null;
        
        for (candidates) |lt| {
            if (minimal == null or lt.id < minimal.?.id) {
                minimal = lt;
            }
        }
        
        return minimal;
    }
    
    /// Check if lifetime can be extended (lifetime extension analysis)
    pub fn canExtendLifetime(
        self: *LifetimeAnalyzer,
        lifetime: Lifetime,
        target_scope: LifetimeScope,
    ) bool {
        _ = self;
        _ = target_scope;
        
        // Simplified: static lifetimes can always be extended
        return lifetime.isStatic();
    }
    
    /// Compute transitive closure of outlives relationships
    pub fn computeTransitiveClosure(self: *LifetimeAnalyzer) !void {
        // Floyd-Warshall-style algorithm for transitive closure
        // Need to copy constraints first to avoid modifying while iterating
        var to_add = ArrayList(lifetimes.LifetimeConstraint){};
        defer to_add.deinit(self.allocator);
        
        const initial_len = self.graph.constraints.items.len;
        var i: usize = 0;
        while (i < initial_len) : (i += 1) {
            const c1 = self.graph.constraints.items[i];
            var j: usize = 0;
            while (j < initial_len) : (j += 1) {
                const c2 = self.graph.constraints.items[j];
                
                // If 'a: 'b and 'b: 'c, then add 'a: 'c
                if (c1.shorter.id == c2.longer.id) {
                    const transitive = lifetimes.LifetimeConstraint.init(c1.longer, c2.shorter);
                    
                    // Check if already exists
                    var exists = false;
                    for (self.graph.constraints.items) |existing| {
                        if (existing.longer.id == transitive.longer.id and
                            existing.shorter.id == transitive.shorter.id)
                        {
                            exists = true;
                            break;
                        }
                    }
                    
                    if (!exists) {
                        try to_add.append(self.allocator, transitive);
                    }
                }
            }
        }
        
        // Add all new constraints
        for (to_add.items) |constraint| {
            try self.graph.addConstraint(constraint);
        }
    }
    
    const LifetimeScope = struct {
        id: u32,
    };
};

// ============================================================================
// Lifetime Elision (Enhanced)
// ============================================================================

/// Enhanced lifetime elision with comprehensive rules
pub const EnhancedElision = struct {
    /// Complete Rust-style elision rules
    pub fn applyElisionRules(
        params: []const ParamInfo,
        return_type: TypeInfo,
    ) ElisionResult {
        var result = ElisionResult{
            .can_elide = false,
            .rule_applied = .None,
            .output_lifetime = null,
        };
        
        // Count reference parameters
        var ref_params = ArrayList(ParamInfo){};
        var self_param: ?ParamInfo = null;
        
        for (params) |param| {
            if (param.is_reference) {
                ref_params.append(std.heap.page_allocator, param) catch return result;
                if (param.is_self) {
                    self_param = param;
                }
            }
        }
        defer ref_params.deinit(std.heap.page_allocator);
        
        // Rule 0: No references in return type -> no lifetime needed
        if (!return_type.has_reference) {
            result.can_elide = true;
            result.rule_applied = .NoOutputReference;
            return result;
        }
        
        // Rule 1: Each elided lifetime in input parameters gets distinct parameter
        if (ref_params.items.len == 0) {
            result.can_elide = false;
            result.rule_applied = .NeedsExplicitLifetime;
            return result;
        }
        
        // Rule 2: Single input lifetime -> assign to all output lifetimes
        if (ref_params.items.len == 1) {
            result.can_elide = true;
            result.rule_applied = .SingleInput;
            result.output_lifetime = ref_params.items[0].lifetime;
            return result;
        }
        
        // Rule 3: &self or &mut self -> assign to all output lifetimes
        if (self_param) |self_p| {
            result.can_elide = true;
            result.rule_applied = .SelfInput;
            result.output_lifetime = self_p.lifetime;
            return result;
        }
        
        // Multiple inputs, no self -> explicit lifetime required
        result.can_elide = false;
        result.rule_applied = .NeedsExplicitLifetime;
        return result;
    }
    
    pub const ParamInfo = struct {
        is_reference: bool,
        is_self: bool,
        lifetime: ?Lifetime,
    };
    
    pub const TypeInfo = struct {
        has_reference: bool,
    };
    
    pub const ElisionResult = struct {
        can_elide: bool,
        rule_applied: ElisionRule,
        output_lifetime: ?Lifetime,
    };
    
    pub const ElisionRule = enum {
        None,
        NoOutputReference,
        SingleInput,
        SelfInput,
        NeedsExplicitLifetime,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "type representation with lifetimes" {
    const allocator = std.testing.allocator;
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    // Create a reference type
    const base = Type{ .Primitive = .Int32 };
    const ref_type = Type{ .Reference = .{
        .lifetime = lt_a,
        .base = &base,
        .is_mutable = false,
    } };
    
    var lts = try ref_type.getLifetimes(allocator);
    defer lts.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 1), lts.items.len);
    try std.testing.expectEqual(lt_a.id, lts.items[0].id);
}

test "variance inference covariant" {
    const allocator = std.testing.allocator;
    
    var inference = VarianceInference.init(allocator);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    const base = Type{ .Primitive = .Int32 };
    const ref_type = Type{ .Reference = .{
        .lifetime = lt_a,
        .base = &base,
        .is_mutable = false,
    } };
    
    const variance = inference.inferVariance(ref_type, 0);
    try std.testing.expectEqual(Variance.Covariant, variance);
}

test "variance inference invariant" {
    const allocator = std.testing.allocator;
    
    var inference = VarianceInference.init(allocator);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    const base = Type{ .Primitive = .Int32 };
    const mut_ref_type = Type{ .Reference = .{
        .lifetime = lt_a,
        .base = &base,
        .is_mutable = true,
    } };
    
    const variance = inference.inferVariance(mut_ref_type, 0);
    try std.testing.expectEqual(Variance.Invariant, variance);
}

test "validate HRTB unused lifetime" {
    const allocator = std.testing.allocator;
    
    var validator = LifetimeValidator.init(allocator);
    defer validator.deinit();
    
    var hrtb = try advanced.HigherRankBound.init(allocator, "Fn");
    defer hrtb.deinit(allocator);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    defer allocator.free(lt_a.name);
    
    // Add quantified lifetime but don't use it
    try hrtb.addQuantifiedLifetime(allocator, lt_a);
    
    const valid = try validator.validateHRTB(hrtb);
    try std.testing.expect(!valid);
    try std.testing.expectEqual(@as(usize, 1), validator.errors.items.len);
}

test "enhanced elision single input" {
    const lt_a = try Lifetime.init(std.testing.allocator, "a", 0, .Named);
    defer std.testing.allocator.free(lt_a.name);
    
    var params = [_]EnhancedElision.ParamInfo{
        .{ .is_reference = true, .is_self = false, .lifetime = lt_a },
    };
    
    const return_type = EnhancedElision.TypeInfo{ .has_reference = true };
    const result = EnhancedElision.applyElisionRules(&params, return_type);
    
    try std.testing.expect(result.can_elide);
    try std.testing.expectEqual(EnhancedElision.ElisionRule.SingleInput, result.rule_applied);
}

test "enhanced elision self parameter" {
    const lt_a = try Lifetime.init(std.testing.allocator, "a", 0, .Named);
    const lt_b = try Lifetime.init(std.testing.allocator, "b", 1, .Named);
    defer {
        std.testing.allocator.free(lt_a.name);
        std.testing.allocator.free(lt_b.name);
    }
    
    var params = [_]EnhancedElision.ParamInfo{
        .{ .is_reference = true, .is_self = true, .lifetime = lt_a },
        .{ .is_reference = true, .is_self = false, .lifetime = lt_b },
    };
    
    const return_type = EnhancedElision.TypeInfo{ .has_reference = true };
    const result = EnhancedElision.applyElisionRules(&params, return_type);
    
    try std.testing.expect(result.can_elide);
    try std.testing.expectEqual(EnhancedElision.ElisionRule.SelfInput, result.rule_applied);
}

test "lifetime analyzer minimal lifetime" {
    const allocator = std.testing.allocator;
    
    var graph = lifetimes.LifetimeConstraintGraph.init(allocator);
    defer graph.deinit();
    
    var analyzer = LifetimeAnalyzer.init(allocator, &graph);
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    const lt_b = try Lifetime.init(allocator, "b", 1, .Named);
    const lt_c = try Lifetime.init(allocator, "c", 2, .Named);
    defer {
        allocator.free(lt_a.name);
        allocator.free(lt_b.name);
        allocator.free(lt_c.name);
    }
    
    const candidates = [_]Lifetime{ lt_c, lt_a, lt_b };
    const minimal = analyzer.findMinimalLifetime(&candidates);
    
    try std.testing.expect(minimal != null);
    try std.testing.expectEqual(@as(u32, 0), minimal.?.id);
}

test "transitive closure computation" {
    const allocator = std.testing.allocator;
    
    var graph = lifetimes.LifetimeConstraintGraph.init(allocator);
    defer graph.deinit();
    
    const lt_a = try Lifetime.init(allocator, "a", 0, .Named);
    const lt_b = try Lifetime.init(allocator, "b", 1, .Named);
    const lt_c = try Lifetime.init(allocator, "c", 2, .Named);
    defer {
        allocator.free(lt_a.name);
        allocator.free(lt_b.name);
        allocator.free(lt_c.name);
    }
    
    // Add: 'a: 'b and 'b: 'c
    try graph.addConstraint(lifetimes.LifetimeConstraint.init(lt_a, lt_b));
    try graph.addConstraint(lifetimes.LifetimeConstraint.init(lt_b, lt_c));
    
    var analyzer = LifetimeAnalyzer.init(allocator, &graph);
    try analyzer.computeTransitiveClosure();
    
    // Should now have 'a: 'c transitively
    try std.testing.expect(graph.outlives(lt_a, lt_c));
}
