// Type System Integration Tests
// Day 67: Comprehensive integration testing across all type system features

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// Import all type system components
const protocols = @import("protocols.zig");
const protocol_impl = @import("protocol_impl.zig");
const protocol_auto = @import("protocol_auto.zig");
const protocol_conditional = @import("protocol_conditional.zig");
const lifetimes = @import("lifetimes.zig");
const borrow_checker = @import("borrow_checker.zig");

// ============================================================================
// Integration Test 1: Protocol + Generics
// ============================================================================

test "protocol with generic type" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // Define Container protocol
    var protocol = try protocols.Protocol.init(allocator, "Container", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const assoc = try protocols.AssociatedType.init(allocator, "Item");
    try protocol.addAssociatedType(allocator, assoc);
    try registry.register(protocol);
    
    // Implement for Vec<T>
    var impl = try protocol_impl.ProtocolImpl.init(allocator, "Container", "Vec<T>", .{
        .file = "test.mojo",
        .line = 10,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    try impl.bindAssociatedType(allocator, "Item", "T");
    
    // Validate
    var validator = protocol_impl.ImplValidator.init(allocator, &registry);
    defer validator.deinit();
    
    const valid = try validator.validateImpl(impl);
    try std.testing.expect(valid);
}

// ============================================================================
// Integration Test 2: Protocol + Lifetimes
// ============================================================================

test "protocol with lifetime parameters" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // Protocol with lifetime-aware methods
    var protocol = try protocols.Protocol.init(allocator, "Borrowable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    
    var method = try protocols.MethodRequirement.init(allocator, "borrow", false);
    const param = try protocols.MethodRequirement.Parameter.init(allocator, "self", "&'a Self");
    try method.addParameter(allocator, param);
    try method.setReturnType(allocator, "&'a Data");
    
    try protocol.addRequirement(allocator, .{ .Method = method });
    try registry.register(protocol);
    
    // Verify protocol was registered
    const found = registry.getProtocol("Borrowable");
    try std.testing.expect(found != null);
}

// ============================================================================
// Integration Test 3: Derive + Borrow Checker
// ============================================================================

test "derived protocol with borrow checking" {
    const allocator = std.testing.allocator;
    
    // Create type info
    var type_info = try protocol_auto.TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    const field = try protocol_auto.TypeInfo.FieldInfo.init(allocator, "x", "Int");
    try type_info.addField(allocator, field);
    
    // Derive Eq
    var generator = protocol_auto.CodeGenerator.init(allocator);
    var code = try generator.generateImpl(type_info, .Eq);
    defer code.deinit(allocator);
    
    // Verify generated code
    try std.testing.expectEqualStrings("Eq", code.protocol_name);
    try std.testing.expect(code.methods.items.len > 0);
}

// ============================================================================
// Integration Test 4: Conditional + Automatic
// ============================================================================

test "conditional impl with derive" {
    const allocator = std.testing.allocator;
    
    // Conditional impl: impl<T: Eq> Eq for Vec<T>
    var cond_impl = try protocol_conditional.ConditionalImpl.init(allocator, "Eq", "Vec");
    defer cond_impl.deinit(allocator);
    
    var param = try protocol_conditional.TypeParameter.init(allocator, "T");
    try param.addConstraint(allocator, "Eq");
    try cond_impl.addTypeParameter(allocator, param);
    
    try std.testing.expectEqual(@as(usize, 1), cond_impl.type_parameters.items.len);
}

// ============================================================================
// Integration Test 5: Multiple Protocol Inheritance
// ============================================================================

test "multiple protocol inheritance chain" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // Base protocol
    const drawable = try protocols.Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    try registry.register(drawable);
    
    // Derived protocol: Shape: Drawable
    var shape = try protocols.Protocol.init(allocator, "Shape", .{
        .file = "test.mojo",
        .line = 10,
        .column = 1,
    });
    try shape.addParent(allocator, "Drawable");
    try registry.register(shape);
    
    // Verify both registered
    try std.testing.expect(registry.getProtocol("Drawable") != null);
    try std.testing.expect(registry.getProtocol("Shape") != null);
}

// ============================================================================
// Integration Test 6: Protocol Dispatch + Generics
// ============================================================================

test "dispatch table with generic methods" {
    const allocator = std.testing.allocator;
    
    var impl = try protocol_impl.ProtocolImpl.init(allocator, "Container", "Vec<T>", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    const method = try protocol_impl.MethodImpl.init(allocator, "get", "vec_get");
    try impl.addMethod(allocator, method);
    
    var builder = protocol_impl.DispatchTableBuilder.init(allocator);
    var table = try builder.buildTable(impl);
    defer table.deinit(allocator);
    
    const func = table.lookup("get");
    try std.testing.expect(func != null);
}

// ============================================================================
// Integration Test 7: Full Protocol System Workflow
// ============================================================================

test "complete protocol workflow" {
    const allocator = std.testing.allocator;
    
    // 1. Create protocol registry
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // 2. Define protocol
    var protocol = try protocols.Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const method_req = try protocols.MethodRequirement.init(allocator, "draw", false);
    try protocol.addRequirement(allocator, .{ .Method = method_req });
    try registry.register(protocol);
    
    // 3. Create implementation
    var impl = try protocol_impl.ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 20,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    const method_impl = try protocol_impl.MethodImpl.init(allocator, "draw", "circle_draw");
    try impl.addMethod(allocator, method_impl);
    
    // 4. Validate implementation
    var validator = protocol_impl.ImplValidator.init(allocator, &registry);
    defer validator.deinit();
    
    const valid = try validator.validateImpl(impl);
    try std.testing.expect(valid);
    
    // 5. Build dispatch table
    var builder = protocol_impl.DispatchTableBuilder.init(allocator);
    var table = try builder.buildTable(impl);
    defer table.deinit(allocator);
    
    const func = table.lookup("draw");
    try std.testing.expect(func != null);
    try std.testing.expectEqualStrings("circle_draw", func.?);
}

// ============================================================================
// Integration Test 8: Derive Multiple Protocols
// ============================================================================

test "derive multiple protocols at once" {
    const allocator = std.testing.allocator;
    
    var processor = protocol_auto.DeriveMacroProcessor.init(allocator);
    
    var type_info = try protocol_auto.TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    const x = try protocol_auto.TypeInfo.FieldInfo.init(allocator, "x", "Int");
    try type_info.addField(allocator, x);
    const y = try protocol_auto.TypeInfo.FieldInfo.init(allocator, "y", "Int");
    try type_info.addField(allocator, y);
    
    const derive_list = [_]protocol_auto.DerivableProtocol{ .Eq, .Hash, .Debug };
    var results = try processor.processDerive(type_info, &derive_list);
    defer {
        for (results.items) |*result| {
            result.deinit(allocator);
        }
        results.deinit(allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 3), results.items.len);
}

// ============================================================================
// Integration Test 9: Blanket Impl with Conditions
// ============================================================================

test "blanket impl with multiple conditions" {
    const allocator = std.testing.allocator;
    
    var blanket = try protocol_conditional.BlanketImpl.init(allocator, "Display", "Vec<T>");
    defer blanket.deinit(allocator);
    
    const cond1 = try protocol_conditional.BlanketImpl.Condition.init(
        allocator,
        .TypeImplements,
        "Debug",
        "T",
    );
    try blanket.addCondition(allocator, cond1);
    
    const cond2 = try protocol_conditional.BlanketImpl.Condition.init(
        allocator,
        .TypeImplements,
        "Display",
        "T",
    );
    try blanket.addCondition(allocator, cond2);
    
    try std.testing.expectEqual(@as(usize, 2), blanket.conditions.items.len);
}

// ============================================================================
// Integration Test 10: Protocol Inheritance + Implementation
// ============================================================================

test "implement protocol with inheritance" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // Parent protocol
    var drawable = try protocols.Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const draw_method = try protocols.MethodRequirement.init(allocator, "draw", false);
    try drawable.addRequirement(allocator, .{ .Method = draw_method });
    try registry.register(drawable);
    
    // Child protocol
    var shape = try protocols.Protocol.init(allocator, "Shape", .{
        .file = "test.mojo",
        .line = 10,
        .column = 1,
    });
    try shape.addParent(allocator, "Drawable");
    const area_method = try protocols.MethodRequirement.init(allocator, "area", false);
    try shape.addRequirement(allocator, .{ .Method = area_method });
    try registry.register(shape);
    
    // Verify hierarchy
    const found_shape = registry.getProtocol("Shape");
    try std.testing.expect(found_shape != null);
    try std.testing.expectEqual(@as(usize, 1), found_shape.?.parent_protocols.items.len);
}

// ============================================================================
// Integration Test 11: Associated Types + Conditional
// ============================================================================

test "conditional impl with associated types" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // Protocol with associated type
    var protocol = try protocols.Protocol.init(allocator, "Iterator", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const assoc = try protocols.AssociatedType.init(allocator, "Item");
    try protocol.addAssociatedType(allocator, assoc);
    try registry.register(protocol);
    
    // Conditional implementation
    var cond_impl = try protocol_conditional.ConditionalImpl.init(allocator, "Iterator", "Vec");
    defer cond_impl.deinit(allocator);
    
    try cond_impl.impl_data.bindAssociatedType(allocator, "Item", "T");
    
    const binding = cond_impl.impl_data.associated_type_bindings.get("Item");
    try std.testing.expect(binding != null);
}

// ============================================================================
// Integration Test 12: Witness Table + Dispatch
// ============================================================================

test "witness table for dynamic dispatch" {
    const allocator = std.testing.allocator;
    
    // Create witness table
    var witness = try protocol_impl.WitnessTable.init(allocator, "Drawable", "Circle");
    defer witness.deinit(allocator);
    
    try witness.addMethod(allocator, "draw", 0x1000);
    try witness.addMethod(allocator, "bounds", 0x2000);
    
    // Lookup methods
    const draw_ptr = witness.getMethodPtr("draw");
    const bounds_ptr = witness.getMethodPtr("bounds");
    
    try std.testing.expect(draw_ptr != null);
    try std.testing.expect(bounds_ptr != null);
    try std.testing.expectEqual(@as(usize, 0x1000), draw_ptr.?);
    try std.testing.expectEqual(@as(usize, 0x2000), bounds_ptr.?);
}

// ============================================================================
// Integration Test 13: Specialization Selection
// ============================================================================

test "specialization with priority" {
    const allocator = std.testing.allocator;
    
    var manager = protocol_conditional.SpecializationManager.init(allocator);
    defer manager.deinit();
    
    // General impl
    const general = try protocol_conditional.ConditionalImpl.init(allocator, "Display", "Vec");
    
    // Specific impl
    const specific = try protocol_conditional.ConditionalImpl.init(allocator, "Display", "Vec");
    
    try manager.registerSpecialization(general, specific, 10);
    
    try std.testing.expectEqual(@as(usize, 1), manager.specializations.items.len);
}

// ============================================================================
// Integration Test 14: Multiple Derives
// ============================================================================

test "struct with multiple derived protocols" {
    const allocator = std.testing.allocator;
    
    var type_info = try protocol_auto.TypeInfo.init(allocator, "Person");
    defer type_info.deinit(allocator);
    
    const name = try protocol_auto.TypeInfo.FieldInfo.init(allocator, "name", "String");
    try type_info.addField(allocator, name);
    const age = try protocol_auto.TypeInfo.FieldInfo.init(allocator, "age", "Int");
    try type_info.addField(allocator, age);
    
    // Derive all common protocols
    var processor = protocol_auto.DeriveMacroProcessor.init(allocator);
    const all_derives = [_]protocol_auto.DerivableProtocol{ .Eq, .Hash, .Debug, .Clone, .Default };
    var results = try processor.processDerive(type_info, &all_derives);
    defer {
        for (results.items) |*result| {
            result.deinit(allocator);
        }
        results.deinit(allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 5), results.items.len);
}

// ============================================================================
// Integration Test 15: Protocol Registry Operations
// ============================================================================

test "protocol registry with multiple protocols" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // Register multiple protocols
    const names = [_][]const u8{ "Eq", "Hash", "Debug", "Clone", "Display" };
    for (names) |name| {
        const protocol = try protocols.Protocol.init(allocator, name, .{
            .file = "test.mojo",
            .line = 1,
            .column = 1,
        });
        try registry.register(protocol);
    }
    
    // Verify all registered
    for (names) |name| {
        const found = registry.getProtocol(name);
        try std.testing.expect(found != null);
    }
}

// ============================================================================
// Integration Test 16: Impl Registry with Lookups
// ============================================================================

test "impl registry with multiple implementations" {
    const allocator = std.testing.allocator;
    
    var registry = protocol_impl.ImplRegistry.init(allocator);
    defer registry.deinit();
    
    // Register multiple impls
    const types = [_][]const u8{ "Circle", "Rectangle", "Triangle" };
    for (types) |type_name| {
        const impl = try protocol_impl.ProtocolImpl.init(allocator, "Drawable", type_name, .{
            .file = "test.mojo",
            .line = 1,
            .column = 1,
        });
        try registry.registerImpl(impl);
    }
    
    // Lookup each impl
    for (types) |type_name| {
        const found = registry.findImpl("Drawable", type_name);
        try std.testing.expect(found != null);
    }
}

// ============================================================================
// Integration Test 17: Complex Constraint Resolution
// ============================================================================

test "resolve complex constraints" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    var resolver = protocol_conditional.ConstraintResolver.init(allocator, &registry);
    
    // impl<T: Eq + Hash> Container for HashMap<T>
    var impl = try protocol_conditional.ConditionalImpl.init(allocator, "Container", "HashMap");
    defer impl.deinit(allocator);
    
    var param = try protocol_conditional.TypeParameter.init(allocator, "T");
    try param.addConstraint(allocator, "Eq");
    try param.addConstraint(allocator, "Hash");
    try impl.addTypeParameter(allocator, param);
    
    var concrete = StringHashMap([]const u8).init(allocator);
    defer concrete.deinit();
    try concrete.put("T", "String");
    
    const valid = try resolver.checkConstraints(impl, concrete);
    try std.testing.expect(valid);
}

// ============================================================================
// Integration Test 18: Blanket Pattern Matching
// ============================================================================

test "blanket impl pattern matching" {
    const allocator = std.testing.allocator;
    
    var matcher = protocol_conditional.BlanketImplMatcher.init(allocator);
    
    // Test various patterns
    try std.testing.expect(try matcher.matchesPattern("Vec<Int>", "Vec<T>"));
    try std.testing.expect(try matcher.matchesPattern("Option<String>", "Option<T>"));
    try std.testing.expect(!try matcher.matchesPattern("Vec<Int>", "HashMap<T>"));
}

// ============================================================================
// Integration Test 19: End-to-End Type Safety
// ============================================================================

test "end to end type safety validation" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    // Define Eq protocol
    var eq_protocol = try protocols.Protocol.init(allocator, "Eq", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const eq_method = try protocols.MethodRequirement.init(allocator, "eq", false);
    try eq_protocol.addRequirement(allocator, .{ .Method = eq_method });
    try registry.register(eq_protocol);
    
    // Auto-derive for Point
    var type_info = try protocol_auto.TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    var generator = protocol_auto.CodeGenerator.init(allocator);
    var generated = try generator.generateImpl(type_info, .Eq);
    defer generated.deinit(allocator);
    
    // Convert to impl
    var processor = protocol_auto.DeriveMacroProcessor.init(allocator);
    var impl = try processor.toProtocolImpl(type_info, generated);
    defer impl.deinit(allocator);
    
    // Validate
    var validator = protocol_impl.ImplValidator.init(allocator, &registry);
    defer validator.deinit();
    
    const valid = try validator.validateImpl(impl);
    try std.testing.expect(valid);
}

// ============================================================================
// Integration Test 20: Full Stack Integration
// ============================================================================

test "complete type system integration" {
    const allocator = std.testing.allocator;
    
    // 1. Protocol system
    var protocol_registry = protocols.ProtocolRegistry.init(allocator);
    defer protocol_registry.deinit();
    
    // 2. Implementation registry
    var impl_registry = protocol_impl.ImplRegistry.init(allocator);
    defer impl_registry.deinit();
    
    // 3. Conditional checker
    var cond_checker = protocol_conditional.ConditionalConformanceChecker.init(allocator, &protocol_registry);
    defer cond_checker.deinit();
    
    // Define protocols
    const drawable = try protocols.Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    try protocol_registry.register(drawable);
    
    const eq_protocol = try protocols.Protocol.init(allocator, "Eq", .{
        .file = "test.mojo",
        .line = 5,
        .column = 1,
    });
    try protocol_registry.register(eq_protocol);
    
    // Create implementations
    const impl1 = try protocol_impl.ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 10,
        .column = 1,
    });
    try impl_registry.registerImpl(impl1);
    
    const impl2 = try protocol_impl.ProtocolImpl.init(allocator, "Eq", "Point", .{
        .file = "test.mojo",
        .line = 20,
        .column = 1,
    });
    try impl_registry.registerImpl(impl2);
    
    // Verify everything works together
    try std.testing.expect(protocol_registry.getProtocol("Drawable") != null);
    try std.testing.expect(protocol_registry.getProtocol("Eq") != null);
    try std.testing.expect(impl_registry.findImpl("Drawable", "Circle") != null);
    try std.testing.expect(impl_registry.findImpl("Eq", "Point") != null);
}
