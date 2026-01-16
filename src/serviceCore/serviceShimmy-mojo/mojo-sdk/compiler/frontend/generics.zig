// Mojo SDK - Advanced Generics
// Day 25: Generic type parameters, higher-kinded types, variadic generics, const generics

const std = @import("std");

// ============================================================================
// Generic Type Parameters
// ============================================================================

pub const TypeParam = struct {
    name: []const u8,
    bounds: std.ArrayList([]const u8),
    default_type: ?[]const u8,
    variance: Variance,
    allocator: std.mem.Allocator,
    
    pub const Variance = enum {
        Invariant,    // T
        Covariant,    // +T (allows subtypes)
        Contravariant, // -T (allows supertypes)
    };
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) TypeParam {
        return TypeParam{
            .name = name,
            .bounds = std.ArrayList([]const u8){},
            .default_type = null,
            .variance = .Invariant,
            .allocator = allocator,
        };
    }
    
    pub fn addBound(self: *TypeParam, bound: []const u8) !void {
        try self.bounds.append(self.allocator, bound);
    }
    
    pub fn withDefault(self: TypeParam, default_type: []const u8) TypeParam {
        return TypeParam{
            .name = self.name,
            .bounds = self.bounds,
            .default_type = default_type,
            .variance = self.variance,
            .allocator = self.allocator,
        };
    }
    
    pub fn withVariance(self: TypeParam, variance: Variance) TypeParam {
        return TypeParam{
            .name = self.name,
            .bounds = self.bounds,
            .default_type = self.default_type,
            .variance = variance,
            .allocator = self.allocator,
        };
    }
    
    pub fn hasDefault(self: *const TypeParam) bool {
        return self.default_type != null;
    }
    
    pub fn deinit(self: *TypeParam) void {
        self.bounds.deinit(self.allocator);
    }
};

pub const GenericFunction = struct {
    name: []const u8,
    type_params: std.ArrayList(TypeParam),
    parameters: std.ArrayList(Parameter),
    return_type: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) GenericFunction {
        return GenericFunction{
            .name = name,
            .type_params = std.ArrayList(TypeParam){},
            .parameters = std.ArrayList(Parameter){},
            .return_type = "void",
            .allocator = allocator,
        };
    }
    
    pub fn addTypeParam(self: *GenericFunction, param: TypeParam) !void {
        try self.type_params.append(self.allocator, param);
    }
    
    pub fn addParameter(self: *GenericFunction, param: Parameter) !void {
        try self.parameters.append(self.allocator, param);
    }
    
    pub fn deinit(self: *GenericFunction) void {
        for (self.type_params.items) |*tp| {
            tp.deinit();
        }
        self.type_params.deinit(self.allocator);
        self.parameters.deinit(self.allocator);
    }
};

pub const Parameter = struct {
    name: []const u8,
    param_type: []const u8,
    
    pub fn init(name: []const u8, param_type: []const u8) Parameter {
        return Parameter{
            .name = name,
            .param_type = param_type,
        };
    }
};

// ============================================================================
// Higher-Kinded Types
// ============================================================================

pub const TypeConstructor = struct {
    name: []const u8,
    arity: usize,  // Number of type parameters
    kind: Kind,
    allocator: std.mem.Allocator,
    
    pub const Kind = enum {
        Star,              // * (concrete type)
        Arrow,             // * -> * (type constructor)
        HigherOrder,       // (* -> *) -> * (higher-order)
    };
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, arity: usize) TypeConstructor {
        return TypeConstructor{
            .name = name,
            .arity = arity,
            .kind = if (arity == 0) .Star else .Arrow,
            .allocator = allocator,
        };
    }
    
    pub fn isMonadic(self: *const TypeConstructor) bool {
        return self.arity == 1 and self.kind == .Arrow;
    }
    
    pub fn apply(self: *const TypeConstructor, args: []const []const u8) ![]const u8 {
        if (args.len != self.arity) {
            return error.ArityMismatch;
        }
        // Simplified: would construct applied type
        return "AppliedType";
    }
};

pub const HigherKindedType = struct {
    constructor: TypeConstructor,
    type_args: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, constructor: TypeConstructor) HigherKindedType {
        return HigherKindedType{
            .constructor = constructor,
            .type_args = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn addTypeArg(self: *HigherKindedType, arg: []const u8) !void {
        try self.type_args.append(self.allocator, arg);
    }
    
    pub fn isFullyApplied(self: *const HigherKindedType) bool {
        return self.type_args.items.len == self.constructor.arity;
    }
    
    pub fn deinit(self: *HigherKindedType) void {
        self.type_args.deinit(self.allocator);
    }
};

// ============================================================================
// Variadic Generics
// ============================================================================

pub const VariadicTypeParam = struct {
    name: []const u8,
    min_args: usize,
    max_args: ?usize,  // null = unbounded
    element_bounds: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) VariadicTypeParam {
        return VariadicTypeParam{
            .name = name,
            .min_args = 0,
            .max_args = null,
            .element_bounds = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn withMinArgs(self: VariadicTypeParam, min: usize) VariadicTypeParam {
        return VariadicTypeParam{
            .name = self.name,
            .min_args = min,
            .max_args = self.max_args,
            .element_bounds = self.element_bounds,
            .allocator = self.allocator,
        };
    }
    
    pub fn withMaxArgs(self: VariadicTypeParam, max: usize) VariadicTypeParam {
        return VariadicTypeParam{
            .name = self.name,
            .min_args = self.min_args,
            .max_args = max,
            .element_bounds = self.element_bounds,
            .allocator = self.allocator,
        };
    }
    
    pub fn addElementBound(self: *VariadicTypeParam, bound: []const u8) !void {
        try self.element_bounds.append(self.allocator, bound);
    }
    
    pub fn acceptsArity(self: *const VariadicTypeParam, arity: usize) bool {
        if (arity < self.min_args) return false;
        if (self.max_args) |max| {
            return arity <= max;
        }
        return true;
    }
    
    pub fn deinit(self: *VariadicTypeParam) void {
        self.element_bounds.deinit(self.allocator);
    }
};

pub const TupleType = struct {
    elements: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) TupleType {
        return TupleType{
            .elements = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn addElement(self: *TupleType, element_type: []const u8) !void {
        try self.elements.append(self.allocator, element_type);
    }
    
    pub fn arity(self: *const TupleType) usize {
        return self.elements.items.len;
    }
    
    pub fn deinit(self: *TupleType) void {
        self.elements.deinit(self.allocator);
    }
};

// ============================================================================
// Const Generics
// ============================================================================

pub const ConstParam = struct {
    name: []const u8,
    const_type: ConstType,
    default_value: ?ConstValue,
    
    pub const ConstType = enum {
        Int,
        UInt,
        Bool,
        String,
    };
    
    pub const ConstValue = union(ConstType) {
        Int: i64,
        UInt: u64,
        Bool: bool,
        String: []const u8,
    };
    
    pub fn init(name: []const u8, const_type: ConstType) ConstParam {
        return ConstParam{
            .name = name,
            .const_type = const_type,
            .default_value = null,
        };
    }
    
    pub fn withDefault(self: ConstParam, value: ConstValue) ConstParam {
        return ConstParam{
            .name = self.name,
            .const_type = self.const_type,
            .default_value = value,
        };
    }
    
    pub fn hasDefault(self: *const ConstParam) bool {
        return self.default_value != null;
    }
};

pub const ArrayType = struct {
    element_type: []const u8,
    size: ConstParam,
    
    pub fn init(element_type: []const u8, size: ConstParam) ArrayType {
        return ArrayType{
            .element_type = element_type,
            .size = size,
        };
    }
};

// ============================================================================
// Generic Specialization
// ============================================================================

pub const Specialization = struct {
    generic_type: []const u8,
    specialized_for: std.ArrayList(TypeMatch),
    priority: i32,  // Higher priority specializations checked first
    allocator: std.mem.Allocator,
    
    pub const TypeMatch = struct {
        param_name: []const u8,
        concrete_type: []const u8,
    };
    
    pub fn init(allocator: std.mem.Allocator, generic_type: []const u8) Specialization {
        return Specialization{
            .generic_type = generic_type,
            .specialized_for = std.ArrayList(TypeMatch){},
            .priority = 0,
            .allocator = allocator,
        };
    }
    
    pub fn addMatch(self: *Specialization, param_name: []const u8, concrete_type: []const u8) !void {
        try self.specialized_for.append(self.allocator, TypeMatch{
            .param_name = param_name,
            .concrete_type = concrete_type,
        });
    }
    
    pub fn withPriority(self: Specialization, priority: i32) Specialization {
        return Specialization{
            .generic_type = self.generic_type,
            .specialized_for = self.specialized_for,
            .priority = priority,
            .allocator = self.allocator,
        };
    }
    
    pub fn matches(self: *const Specialization, type_args: []const []const u8) bool {
        if (type_args.len != self.specialized_for.items.len) return false;
        
        for (self.specialized_for.items, 0..) |match, i| {
            if (!std.mem.eql(u8, type_args[i], match.concrete_type)) {
                return false;
            }
        }
        return true;
    }
    
    pub fn deinit(self: *Specialization) void {
        self.specialized_for.deinit(self.allocator);
    }
};

pub const SpecializationRegistry = struct {
    specializations: std.ArrayList(Specialization),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) SpecializationRegistry {
        return SpecializationRegistry{
            .specializations = std.ArrayList(Specialization){},
            .allocator = allocator,
        };
    }
    
    pub fn register(self: *SpecializationRegistry, spec: Specialization) !void {
        try self.specializations.append(self.allocator, spec);
        // Sort by priority (higher first)
        std.mem.sort(Specialization, self.specializations.items, {}, comparePriority);
    }
    
    fn comparePriority(_: void, a: Specialization, b: Specialization) bool {
        return a.priority > b.priority;
    }
    
    pub fn findSpecialization(self: *const SpecializationRegistry, generic_type: []const u8, type_args: []const []const u8) ?*const Specialization {
        for (self.specializations.items) |*spec| {
            if (std.mem.eql(u8, spec.generic_type, generic_type) and spec.matches(type_args)) {
                return spec;
            }
        }
        return null;
    }
    
    pub fn deinit(self: *SpecializationRegistry) void {
        for (self.specializations.items) |*spec| {
            spec.deinit();
        }
        self.specializations.deinit(self.allocator);
    }
};

// ============================================================================
// Type-Level Computation
// ============================================================================

pub const TypeApplication = struct {
    constructor: []const u8,
    args: std.ArrayList(TypeLevelExpr),
    allocator: std.mem.Allocator,
};

pub const TypeLambdaExpr = struct {
    param: []const u8,
    body: *TypeLevelExpr,
    allocator: std.mem.Allocator,
};

pub const TypeLetExpr = struct {
    binding: []const u8,
    value: *TypeLevelExpr,
    in_expr: *TypeLevelExpr,
    allocator: std.mem.Allocator,
};

pub const TypeLevelExpr = union(enum) {
    TypeVar: []const u8,
    TypeApp: TypeApplication,
    TypeLambda: TypeLambdaExpr,
    TypeLet: TypeLetExpr,
};

pub const TypeEvaluator = struct {
    bindings: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) TypeEvaluator {
        return TypeEvaluator{
            .bindings = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn eval(self: *TypeEvaluator, expr: TypeLevelExpr) ![]const u8 {
        return switch (expr) {
            .TypeVar => |name| self.bindings.get(name) orelse name,
            .TypeApp => "AppliedType",  // Simplified
            .TypeLambda => "Lambda",     // Simplified
            .TypeLet => "LetBinding",    // Simplified
        };
    }
    
    pub fn bind(self: *TypeEvaluator, name: []const u8, type_val: []const u8) !void {
        try self.bindings.put(name, type_val);
    }
    
    pub fn deinit(self: *TypeEvaluator) void {
        self.bindings.deinit();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "generics: type parameter" {
    const allocator = std.testing.allocator;
    var param = TypeParam.init(allocator, "T");
    defer param.deinit();
    
    try param.addBound("Display");
    
    try std.testing.expectEqual(@as(usize, 1), param.bounds.items.len);
}

test "generics: type parameter with default" {
    const allocator = std.testing.allocator;
    var param = TypeParam.init(allocator, "T");
    defer param.deinit();
    
    const with_default = param.withDefault("Int");
    
    try std.testing.expect(with_default.hasDefault());
}

test "generics: variance" {
    const allocator = std.testing.allocator;
    var param = TypeParam.init(allocator, "T");
    defer param.deinit();
    
    const covariant = param.withVariance(.Covariant);
    
    try std.testing.expectEqual(TypeParam.Variance.Covariant, covariant.variance);
}

test "generics: generic function" {
    const allocator = std.testing.allocator;
    var func = GenericFunction.init(allocator, "map");
    defer func.deinit();
    
    const type_param = TypeParam.init(allocator, "T");
    try func.addTypeParam(type_param);
    
    try std.testing.expectEqual(@as(usize, 1), func.type_params.items.len);
}

test "generics: type constructor" {
    const allocator = std.testing.allocator;
    const vec = TypeConstructor.init(allocator, "Vec", 1);
    
    try std.testing.expect(vec.isMonadic());
}

test "generics: variadic type param" {
    const allocator = std.testing.allocator;
    var variadic = VariadicTypeParam.init(allocator, "Args");
    defer variadic.deinit();
    
    const with_min = variadic.withMinArgs(1);
    
    try std.testing.expect(with_min.acceptsArity(2));
    try std.testing.expect(!with_min.acceptsArity(0));
}

test "generics: const parameter" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    const size = ConstParam.init("N", .UInt);
    const with_default = size.withDefault(.{ .UInt = 10 });
    
    try std.testing.expect(with_default.hasDefault());
}

test "generics: specialization" {
    const allocator = std.testing.allocator;
    var spec = Specialization.init(allocator, "Vec");
    defer spec.deinit();
    
    try spec.addMatch("T", "Int");
    
    const type_args = [_][]const u8{"Int"};
    try std.testing.expect(spec.matches(&type_args));
}

test "generics: specialization registry" {
    const allocator = std.testing.allocator;
    var registry = SpecializationRegistry.init(allocator);
    defer registry.deinit();
    
    var spec = Specialization.init(allocator, "Vec");
    try spec.addMatch("T", "Int");
    
    try registry.register(spec);
    
    const type_args = [_][]const u8{"Int"};
    const found = registry.findSpecialization("Vec", &type_args);
    
    try std.testing.expect(found != null);
}

test "generics: type evaluator" {
    const allocator = std.testing.allocator;
    var evaluator = TypeEvaluator.init(allocator);
    defer evaluator.deinit();
    
    try evaluator.bind("T", "Int");
    
    const type_var = TypeLevelExpr{ .TypeVar = "T" };
    const result = try evaluator.eval(type_var);
    
    try std.testing.expect(std.mem.eql(u8, result, "Int"));
}
