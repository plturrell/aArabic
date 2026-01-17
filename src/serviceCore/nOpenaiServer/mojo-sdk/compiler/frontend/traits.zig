// Mojo SDK - Trait System
// Day 24: Traits, bounds, implementations, and associated types

const std = @import("std");

// ============================================================================
// Trait Definition
// ============================================================================

pub const TraitMethod = struct {
    name: []const u8,
    parameters: std.ArrayList(Parameter),
    return_type: ?[]const u8,
    default_impl: ?[]const u8,  // Default implementation
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) TraitMethod {
        return TraitMethod{
            .name = name,
            .parameters = std.ArrayList(Parameter){},
            .return_type = null,
            .default_impl = null,
            .allocator = allocator,
        };
    }
    
    pub fn addParameter(self: *TraitMethod, param: Parameter) !void {
        try self.parameters.append(self.allocator, param);
    }
    
    pub fn withReturnType(self: TraitMethod, return_type: []const u8) TraitMethod {
        return TraitMethod{
            .name = self.name,
            .parameters = self.parameters,
            .return_type = return_type,
            .default_impl = self.default_impl,
            .allocator = self.allocator,
        };
    }
    
    pub fn withDefaultImpl(self: TraitMethod, impl: []const u8) TraitMethod {
        return TraitMethod{
            .name = self.name,
            .parameters = self.parameters,
            .return_type = self.return_type,
            .default_impl = impl,
            .allocator = self.allocator,
        };
    }
    
    pub fn hasDefaultImpl(self: *const TraitMethod) bool {
        return self.default_impl != null;
    }
    
    pub fn deinit(self: *TraitMethod) void {
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

pub const AssociatedType = struct {
    name: []const u8,
    bounds: std.ArrayList([]const u8),
    default_type: ?[]const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) AssociatedType {
        return AssociatedType{
            .name = name,
            .bounds = std.ArrayList([]const u8){},
            .default_type = null,
            .allocator = allocator,
        };
    }
    
    pub fn addBound(self: *AssociatedType, bound: []const u8) !void {
        try self.bounds.append(self.allocator, bound);
    }
    
    pub fn withDefault(self: AssociatedType, default_type: []const u8) AssociatedType {
        return AssociatedType{
            .name = self.name,
            .bounds = self.bounds,
            .default_type = default_type,
            .allocator = self.allocator,
        };
    }
    
    pub fn hasDefault(self: *const AssociatedType) bool {
        return self.default_type != null;
    }
    
    pub fn deinit(self: *AssociatedType) void {
        self.bounds.deinit(self.allocator);
    }
};

pub const Trait = struct {
    name: []const u8,
    methods: std.ArrayList(TraitMethod),
    associated_types: std.ArrayList(AssociatedType),
    super_traits: std.ArrayList([]const u8),  // Trait inheritance
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Trait {
        return Trait{
            .name = name,
            .methods = std.ArrayList(TraitMethod){},
            .associated_types = std.ArrayList(AssociatedType){},
            .super_traits = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn addMethod(self: *Trait, method: TraitMethod) !void {
        try self.methods.append(self.allocator, method);
    }
    
    pub fn addAssociatedType(self: *Trait, assoc_type: AssociatedType) !void {
        try self.associated_types.append(self.allocator, assoc_type);
    }
    
    pub fn addSuperTrait(self: *Trait, super_trait: []const u8) !void {
        try self.super_traits.append(self.allocator, super_trait);
    }
    
    pub fn getMethod(self: *const Trait, name: []const u8) ?*const TraitMethod {
        for (self.methods.items) |*method| {
            if (std.mem.eql(u8, method.name, name)) {
                return method;
            }
        }
        return null;
    }
    
    pub fn hasMethod(self: *const Trait, name: []const u8) bool {
        return self.getMethod(name) != null;
    }
    
    pub fn deinit(self: *Trait) void {
        for (self.methods.items) |*method| {
            method.deinit();
        }
        self.methods.deinit(self.allocator);
        
        for (self.associated_types.items) |*assoc_type| {
            assoc_type.deinit();
        }
        self.associated_types.deinit(self.allocator);
        
        self.super_traits.deinit(self.allocator);
    }
};

// ============================================================================
// Trait Bounds
// ============================================================================

pub const TraitBound = struct {
    trait_name: []const u8,
    type_params: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, trait_name: []const u8) TraitBound {
        return TraitBound{
            .trait_name = trait_name,
            .type_params = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn addTypeParam(self: *TraitBound, param: []const u8) !void {
        try self.type_params.append(self.allocator, param);
    }
    
    pub fn deinit(self: *TraitBound) void {
        self.type_params.deinit(self.allocator);
    }
};

pub const BoundedType = struct {
    type_name: []const u8,
    bounds: std.ArrayList(TraitBound),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, type_name: []const u8) BoundedType {
        return BoundedType{
            .type_name = type_name,
            .bounds = std.ArrayList(TraitBound){},
            .allocator = allocator,
        };
    }
    
    pub fn addBound(self: *BoundedType, bound: TraitBound) !void {
        try self.bounds.append(self.allocator, bound);
    }
    
    pub fn hasBound(self: *const BoundedType, trait_name: []const u8) bool {
        for (self.bounds.items) |bound| {
            if (std.mem.eql(u8, bound.trait_name, trait_name)) {
                return true;
            }
        }
        return false;
    }
    
    pub fn deinit(self: *BoundedType) void {
        for (self.bounds.items) |*bound| {
            bound.deinit();
        }
        self.bounds.deinit(self.allocator);
    }
};

// ============================================================================
// Trait Implementation
// ============================================================================

pub const MethodImpl = struct {
    method_name: []const u8,
    implementation: []const u8,  // Method body
    
    pub fn init(method_name: []const u8, implementation: []const u8) MethodImpl {
        return MethodImpl{
            .method_name = method_name,
            .implementation = implementation,
        };
    }
};

pub const TraitImpl = struct {
    trait_name: []const u8,
    for_type: []const u8,
    method_impls: std.ArrayList(MethodImpl),
    type_mappings: std.StringHashMap([]const u8),  // Associated type mappings
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, trait_name: []const u8, for_type: []const u8) TraitImpl {
        return TraitImpl{
            .trait_name = trait_name,
            .for_type = for_type,
            .method_impls = std.ArrayList(MethodImpl){},
            .type_mappings = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addMethodImpl(self: *TraitImpl, impl: MethodImpl) !void {
        try self.method_impls.append(self.allocator, impl);
    }
    
    pub fn addTypeMapping(self: *TraitImpl, assoc_type: []const u8, concrete_type: []const u8) !void {
        try self.type_mappings.put(assoc_type, concrete_type);
    }
    
    pub fn getMethodImpl(self: *const TraitImpl, method_name: []const u8) ?*const MethodImpl {
        for (self.method_impls.items) |*impl| {
            if (std.mem.eql(u8, impl.method_name, method_name)) {
                return impl;
            }
        }
        return null;
    }
    
    pub fn hasMethodImpl(self: *const TraitImpl, method_name: []const u8) bool {
        return self.getMethodImpl(method_name) != null;
    }
    
    pub fn deinit(self: *TraitImpl) void {
        self.method_impls.deinit(self.allocator);
        self.type_mappings.deinit();
    }
};

// ============================================================================
// Trait Object
// ============================================================================

pub const TraitObject = struct {
    trait_name: []const u8,
    vtable: *VTable,
    data_ptr: *anyopaque,
    
    pub fn init(trait_name: []const u8, vtable: *VTable, data_ptr: *anyopaque) TraitObject {
        return TraitObject{
            .trait_name = trait_name,
            .vtable = vtable,
            .data_ptr = data_ptr,
        };
    }
    
    pub fn call(self: *TraitObject, method_name: []const u8, args: []const u8) []const u8 {
        _ = args;
        // Simplified: would dispatch to vtable
        _ = self;
        _ = method_name;
        return "result";
    }
};

pub const VTable = struct {
    methods: std.StringHashMap(*const fn (*anyopaque, []const u8) []const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) VTable {
        return VTable{
            .methods = std.StringHashMap(*const fn (*anyopaque, []const u8) []const u8).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addMethod(self: *VTable, name: []const u8, func: *const fn (*anyopaque, []const u8) []const u8) !void {
        try self.methods.put(name, func);
    }
    
    pub fn getMethod(self: *const VTable, name: []const u8) ?*const fn (*anyopaque, []const u8) []const u8 {
        return self.methods.get(name);
    }
    
    pub fn deinit(self: *VTable) void {
        self.methods.deinit();
    }
};

// ============================================================================
// Trait Checker
// ============================================================================

pub const TraitChecker = struct {
    traits: std.StringHashMap(Trait),
    impls: std.ArrayList(TraitImpl),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) TraitChecker {
        return TraitChecker{
            .traits = std.StringHashMap(Trait).init(allocator),
            .impls = std.ArrayList(TraitImpl){},
            .allocator = allocator,
        };
    }
    
    pub fn registerTrait(self: *TraitChecker, trait: Trait) !void {
        try self.traits.put(trait.name, trait);
    }
    
    pub fn registerImpl(self: *TraitChecker, impl: TraitImpl) !void {
        try self.impls.append(self.allocator, impl);
    }
    
    pub fn typeImplementsTrait(self: *const TraitChecker, type_name: []const u8, trait_name: []const u8) bool {
        for (self.impls.items) |impl| {
            if (std.mem.eql(u8, impl.for_type, type_name) and 
                std.mem.eql(u8, impl.trait_name, trait_name)) {
                return true;
            }
        }
        return false;
    }
    
    pub fn checkBounds(self: *const TraitChecker, bounded_type: *const BoundedType) bool {
        for (bounded_type.bounds.items) |bound| {
            if (!self.typeImplementsTrait(bounded_type.type_name, bound.trait_name)) {
                return false;
            }
        }
        return true;
    }
    
    pub fn deinit(self: *TraitChecker) void {
        var trait_iter = self.traits.valueIterator();
        while (trait_iter.next()) |trait| {
            var trait_copy = trait.*;
            trait_copy.deinit();
        }
        self.traits.deinit();
        
        for (self.impls.items) |*impl| {
            impl.deinit();
        }
        self.impls.deinit(self.allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "traits: trait definition" {
    const allocator = std.testing.allocator;
    var trait = Trait.init(allocator, "Display");
    defer trait.deinit();
    
    var method = TraitMethod.init(allocator, "display");
    defer method.deinit();
    
    try trait.addMethod(method);
    
    try std.testing.expectEqual(@as(usize, 1), trait.methods.items.len);
    try std.testing.expect(trait.hasMethod("display"));
}

test "traits: trait method with parameters" {
    const allocator = std.testing.allocator;
    var method = TraitMethod.init(allocator, "format");
    defer method.deinit();
    
    const param = Parameter.init("width", "Int");
    try method.addParameter(param);
    
    try std.testing.expectEqual(@as(usize, 1), method.parameters.items.len);
}

test "traits: default method implementation" {
    const allocator = std.testing.allocator;
    var method = TraitMethod.init(allocator, "default_method");
    defer method.deinit();
    
    const with_default = method.withDefaultImpl("default implementation");
    
    try std.testing.expect(with_default.hasDefaultImpl());
}

test "traits: associated types" {
    const allocator = std.testing.allocator;
    var assoc_type = AssociatedType.init(allocator, "Item");
    defer assoc_type.deinit();
    
    try assoc_type.addBound("Display");
    
    try std.testing.expectEqual(@as(usize, 1), assoc_type.bounds.items.len);
}

test "traits: trait bounds" {
    const allocator = std.testing.allocator;
    var bound = TraitBound.init(allocator, "Clone");
    defer bound.deinit();
    
    try bound.addTypeParam("T");
    
    try std.testing.expectEqual(@as(usize, 1), bound.type_params.items.len);
}

test "traits: bounded type" {
    const allocator = std.testing.allocator;
    var bounded = BoundedType.init(allocator, "T");
    defer bounded.deinit();
    
    var bound = TraitBound.init(allocator, "Display");
    defer bound.deinit();
    
    try bounded.addBound(bound);
    
    try std.testing.expect(bounded.hasBound("Display"));
}

test "traits: trait implementation" {
    const allocator = std.testing.allocator;
    var impl = TraitImpl.init(allocator, "Display", "String");
    defer impl.deinit();
    
    const method_impl = MethodImpl.init("display", "return self");
    try impl.addMethodImpl(method_impl);
    
    try std.testing.expect(impl.hasMethodImpl("display"));
}

test "traits: trait checker" {
    const allocator = std.testing.allocator;
    var checker = TraitChecker.init(allocator);
    defer checker.deinit();
    
    const trait = Trait.init(allocator, "Display");
    try checker.registerTrait(trait);
    
    const impl = TraitImpl.init(allocator, "Display", "String");
    try checker.registerImpl(impl);
    
    try std.testing.expect(checker.typeImplementsTrait("String", "Display"));
}

test "traits: check bounds" {
    const allocator = std.testing.allocator;
    var checker = TraitChecker.init(allocator);
    defer checker.deinit();
    
    // Register trait
    const trait = Trait.init(allocator, "Clone");
    try checker.registerTrait(trait);
    
    // Register implementation
    const impl = TraitImpl.init(allocator, "Clone", "Int");
    try checker.registerImpl(impl);
    
    // Create bounded type
    var bounded = BoundedType.init(allocator, "Int");
    defer bounded.deinit();
    
    var bound = TraitBound.init(allocator, "Clone");
    defer bound.deinit();
    
    try bounded.addBound(bound);
    
    // Check bounds
    try std.testing.expect(checker.checkBounds(&bounded));
}

test "traits: vtable" {
    const allocator = std.testing.allocator;
    var vtable = VTable.init(allocator);
    defer vtable.deinit();
    
    const dummy_fn = struct {
        fn call(ptr: *anyopaque, args: []const u8) []const u8 {
            _ = ptr;
            _ = args;
            return "result";
        }
    }.call;
    
    try vtable.addMethod("display", dummy_fn);
    
    try std.testing.expect(vtable.getMethod("display") != null);
}
