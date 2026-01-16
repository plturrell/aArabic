// Mojo SDK - Enhanced Type System
// Day 22: Type system enhancements, inference, and constraints

const std = @import("std");

// ============================================================================
// Base Type System
// ============================================================================

pub const PrimitiveType = enum {
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float,
    Float32,
    Float64,
    Bool,
    String,
    Void,
    
    pub fn bitWidth(self: PrimitiveType) ?usize {
        return switch (self) {
            .Int8, .UInt8 => 8,
            .Int16, .UInt16 => 16,
            .Int32, .UInt32 => 32,
            .Int64, .UInt64 => 64,
            .Float32 => 32,
            .Float64 => 64,
            .Bool => 1,
            else => null,
        };
    }
    
    pub fn isSigned(self: PrimitiveType) bool {
        return switch (self) {
            .Int, .Int8, .Int16, .Int32, .Int64, .Float, .Float32, .Float64 => true,
            else => false,
        };
    }
    
    pub fn isInteger(self: PrimitiveType) bool {
        return switch (self) {
            .Int, .Int8, .Int16, .Int32, .Int64,
            .UInt, .UInt8, .UInt16, .UInt32, .UInt64 => true,
            else => false,
        };
    }
    
    pub fn isFloat(self: PrimitiveType) bool {
        return switch (self) {
            .Float, .Float32, .Float64 => true,
            else => false,
        };
    }
};

// ============================================================================
// Type Definitions
// ============================================================================

pub const Type = union(enum) {
    Primitive: PrimitiveType,
    Array: *ArrayType,
    Pointer: *PointerType,
    Function: *FunctionType,
    Struct: *StructType,
    Union: *UnionType,
    Option: *OptionType,
    Generic: *GenericType,
    TypeAlias: *TypeAlias,
    
    pub fn isPrimitive(self: *const Type) bool {
        return switch (self.*) {
            .Primitive => true,
            else => false,
        };
    }
    
    pub fn isPointer(self: *const Type) bool {
        return switch (self.*) {
            .Pointer => true,
            else => false,
        };
    }
    
    pub fn isFunction(self: *const Type) bool {
        return switch (self.*) {
            .Function => true,
            else => false,
        };
    }
};

pub const ArrayType = struct {
    element_type: *Type,
    size: ?usize,
    
    pub fn init(allocator: std.mem.Allocator, element: *Type, size: ?usize) !*ArrayType {
        const array_type = try allocator.create(ArrayType);
        array_type.* = ArrayType{
            .element_type = element,
            .size = size,
        };
        return array_type;
    }
    
    pub fn isDynamic(self: *const ArrayType) bool {
        return self.size == null;
    }
};

pub const PointerType = struct {
    pointee_type: *Type,
    is_mutable: bool = true,
    
    pub fn init(allocator: std.mem.Allocator, pointee: *Type, mutable: bool) !*PointerType {
        const ptr_type = try allocator.create(PointerType);
        ptr_type.* = PointerType{
            .pointee_type = pointee,
            .is_mutable = mutable,
        };
        return ptr_type;
    }
};

pub const FunctionType = struct {
    param_types: std.ArrayList(*Type),
    return_type: *Type,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, return_type: *Type) !*FunctionType {
        const func_type = try allocator.create(FunctionType);
        func_type.* = FunctionType{
            .param_types = std.ArrayList(*Type){},
            .return_type = return_type,
            .allocator = allocator,
        };
        return func_type;
    }
    
    pub fn addParam(self: *FunctionType, param_type: *Type) !void {
        try self.param_types.append(self.allocator, param_type);
    }
    
    pub fn deinit(self: *FunctionType) void {
        self.param_types.deinit(self.allocator);
    }
};

pub const StructType = struct {
    name: []const u8,
    fields: std.StringHashMap(*Type),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) !*StructType {
        const struct_type = try allocator.create(StructType);
        struct_type.* = StructType{
            .name = name,
            .fields = std.StringHashMap(*Type).init(allocator),
            .allocator = allocator,
        };
        return struct_type;
    }
    
    pub fn addField(self: *StructType, field_name: []const u8, field_type: *Type) !void {
        try self.fields.put(field_name, field_type);
    }
    
    pub fn getField(self: *StructType, field_name: []const u8) ?*Type {
        return self.fields.get(field_name);
    }
    
    pub fn deinit(self: *StructType) void {
        self.fields.deinit();
    }
};

// ============================================================================
// Union Types
// ============================================================================

pub const UnionType = struct {
    variants: std.ArrayList(*Type),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) !*UnionType {
        const union_type = try allocator.create(UnionType);
        union_type.* = UnionType{
            .variants = std.ArrayList(*Type){},
            .allocator = allocator,
        };
        return union_type;
    }
    
    pub fn addVariant(self: *UnionType, variant_type: *Type) !void {
        try self.variants.append(self.allocator, variant_type);
    }
    
    pub fn hasVariant(self: *const UnionType, variant_type: *const Type) bool {
        for (self.variants.items) |variant| {
            if (variant == variant_type) return true;
        }
        return false;
    }
    
    pub fn deinit(self: *UnionType) void {
        self.variants.deinit(self.allocator);
    }
};

// ============================================================================
// Option Types
// ============================================================================

pub const OptionType = struct {
    inner_type: *Type,
    
    pub fn init(allocator: std.mem.Allocator, inner: *Type) !*OptionType {
        const option_type = try allocator.create(OptionType);
        option_type.* = OptionType{
            .inner_type = inner,
        };
        return option_type;
    }
    
    pub fn isSome(self: *const OptionType) bool {
        _ = self;
        return true; // Simplified for now
    }
    
    pub fn isNone(self: *const OptionType) bool {
        return !self.isSome();
    }
};

// ============================================================================
// Generic Types
// ============================================================================

pub const GenericType = struct {
    name: []const u8,
    constraints: std.ArrayList(*TypeConstraint),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) !*GenericType {
        const generic_type = try allocator.create(GenericType);
        generic_type.* = GenericType{
            .name = name,
            .constraints = std.ArrayList(*TypeConstraint){},
            .allocator = allocator,
        };
        return generic_type;
    }
    
    pub fn addConstraint(self: *GenericType, constraint: *TypeConstraint) !void {
        try self.constraints.append(self.allocator, constraint);
    }
    
    pub fn deinit(self: *GenericType) void {
        self.constraints.deinit(self.allocator);
    }
};

pub const TypeConstraint = struct {
    constraint_type: ConstraintType,
    target_type: ?*Type,
    
    pub fn init(constraint_type: ConstraintType) TypeConstraint {
        return TypeConstraint{
            .constraint_type = constraint_type,
            .target_type = null,
        };
    }
    
    pub fn withTarget(self: TypeConstraint, target: *Type) TypeConstraint {
        return TypeConstraint{
            .constraint_type = self.constraint_type,
            .target_type = target,
        };
    }
};

pub const ConstraintType = enum {
    Numeric,
    Comparable,
    Equatable,
    Copyable,
    Movable,
};

// ============================================================================
// Type Aliases
// ============================================================================

pub const TypeAlias = struct {
    name: []const u8,
    actual_type: *Type,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, actual: *Type) !*TypeAlias {
        const alias = try allocator.create(TypeAlias);
        alias.* = TypeAlias{
            .name = name,
            .actual_type = actual,
        };
        return alias;
    }
};

// ============================================================================
// Type Inference
// ============================================================================

pub const TypeInference = struct {
    allocator: std.mem.Allocator,
    type_variables: std.StringHashMap(*Type),
    
    pub fn init(allocator: std.mem.Allocator) TypeInference {
        return TypeInference{
            .allocator = allocator,
            .type_variables = std.StringHashMap(*Type).init(allocator),
        };
    }
    
    pub fn deinit(self: *TypeInference) void {
        self.type_variables.deinit();
    }
    
    pub fn inferType(self: *TypeInference, expr: []const u8) !*Type {
        _ = expr;
        // Simplified: return Int type
        const int_type = try self.allocator.create(Type);
        int_type.* = Type{ .Primitive = .Int };
        return int_type;
    }
    
    pub fn unify(self: *TypeInference, type1: *Type, type2: *Type) !bool {
        _ = self;
        _ = type1;
        _ = type2;
        return true; // Simplified for now
    }
};

// ============================================================================
// Type Checker
// ============================================================================

pub const TypeChecker = struct {
    allocator: std.mem.Allocator,
    inference: TypeInference,
    
    pub fn init(allocator: std.mem.Allocator) TypeChecker {
        return TypeChecker{
            .allocator = allocator,
            .inference = TypeInference.init(allocator),
        };
    }
    
    pub fn deinit(self: *TypeChecker) void {
        self.inference.deinit();
    }
    
    pub fn checkType(self: *TypeChecker, expected: *Type, actual: *Type) !bool {
        return try self.inference.unify(expected, actual);
    }
    
    pub fn isCompatible(self: *TypeChecker, type1: *Type, type2: *Type) bool {
        _ = self;
        _ = type1;
        _ = type2;
        return true; // Simplified for now
    }
};

// ============================================================================
// Tests
// ============================================================================

test "types: primitive bit width" {
    try std.testing.expectEqual(@as(usize, 8), PrimitiveType.Int8.bitWidth().?);
    try std.testing.expectEqual(@as(usize, 32), PrimitiveType.Int32.bitWidth().?);
    try std.testing.expectEqual(@as(?usize, null), PrimitiveType.Int.bitWidth());
}

test "types: primitive signed" {
    try std.testing.expect(PrimitiveType.Int.isSigned());
    try std.testing.expect(!PrimitiveType.UInt.isSigned());
}

test "types: primitive categories" {
    try std.testing.expect(PrimitiveType.Int.isInteger());
    try std.testing.expect(PrimitiveType.Float.isFloat());
    try std.testing.expect(!PrimitiveType.Bool.isInteger());
}

test "types: type union primitive" {
    var int_type = Type{ .Primitive = .Int };
    try std.testing.expect(int_type.isPrimitive());
    try std.testing.expect(!int_type.isPointer());
}

test "types: array type" {
    const allocator = std.testing.allocator;
    
    const element = try allocator.create(Type);
    defer allocator.destroy(element);
    element.* = Type{ .Primitive = .Int };
    
    const array = try ArrayType.init(allocator, element, 10);
    defer allocator.destroy(array);
    
    try std.testing.expect(!array.isDynamic());
    try std.testing.expectEqual(@as(?usize, 10), array.size);
}

test "types: pointer type" {
    const allocator = std.testing.allocator;
    
    const pointee = try allocator.create(Type);
    defer allocator.destroy(pointee);
    pointee.* = Type{ .Primitive = .Int };
    
    const ptr = try PointerType.init(allocator, pointee, true);
    defer allocator.destroy(ptr);
    
    try std.testing.expect(ptr.is_mutable);
}

test "types: function type" {
    const allocator = std.testing.allocator;
    
    const return_type = try allocator.create(Type);
    defer allocator.destroy(return_type);
    return_type.* = Type{ .Primitive = .Void };
    
    const func = try FunctionType.init(allocator, return_type);
    defer {
        func.deinit();
        allocator.destroy(func);
    }
    
    const param_type = try allocator.create(Type);
    defer allocator.destroy(param_type);
    param_type.* = Type{ .Primitive = .Int };
    
    try func.addParam(param_type);
    try std.testing.expectEqual(@as(usize, 1), func.param_types.items.len);
}

test "types: union type" {
    const allocator = std.testing.allocator;
    
    const union_type = try UnionType.init(allocator);
    defer {
        union_type.deinit();
        allocator.destroy(union_type);
    }
    
    const variant = try allocator.create(Type);
    defer allocator.destroy(variant);
    variant.* = Type{ .Primitive = .Int };
    
    try union_type.addVariant(variant);
    try std.testing.expect(union_type.hasVariant(variant));
}

test "types: type inference" {
    const allocator = std.testing.allocator;
    var inference = TypeInference.init(allocator);
    defer inference.deinit();
    
    const inferred = try inference.inferType("42");
    defer allocator.destroy(inferred);
    
    try std.testing.expect(inferred.isPrimitive());
}

test "types: type checker" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();
    
    const type1 = try allocator.create(Type);
    defer allocator.destroy(type1);
    type1.* = Type{ .Primitive = .Int };
    
    const type2 = try allocator.create(Type);
    defer allocator.destroy(type2);
    type2.* = Type{ .Primitive = .Int };
    
    const compatible = checker.isCompatible(type1, type2);
    try std.testing.expect(compatible);
}
