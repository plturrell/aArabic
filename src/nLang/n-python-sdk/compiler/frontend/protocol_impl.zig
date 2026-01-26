// Protocol Implementation
// Day 64: Protocol implementations and dispatch

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const protocols = @import("protocols.zig");
const Protocol = protocols.Protocol;
const ProtocolRequirement = protocols.ProtocolRequirement;
const MethodRequirement = protocols.MethodRequirement;

// ============================================================================
// Protocol Implementation
// ============================================================================

/// Represents an implementation of a protocol for a type
pub const ProtocolImpl = struct {
    protocol_name: []const u8,
    type_name: []const u8,
    methods: ArrayList(MethodImpl),
    associated_type_bindings: StringHashMap([]const u8),
    source_location: SourceLocation,
    
    pub const SourceLocation = struct {
        file: []const u8,
        line: u32,
        column: u32,
    };
    
    pub fn init(
        allocator: Allocator,
        protocol_name: []const u8,
        type_name: []const u8,
        location: SourceLocation,
    ) !ProtocolImpl {
        return ProtocolImpl{
            .protocol_name = try allocator.dupe(u8, protocol_name),
            .type_name = try allocator.dupe(u8, type_name),
            .methods = ArrayList(MethodImpl){},
            .associated_type_bindings = StringHashMap([]const u8).init(allocator),
            .source_location = location,
        };
    }
    
    pub fn deinit(self: *ProtocolImpl, allocator: Allocator) void {
        allocator.free(self.protocol_name);
        allocator.free(self.type_name);
        for (self.methods.items) |*method| {
            method.deinit(allocator);
        }
        self.methods.deinit(allocator);
        var it = self.associated_type_bindings.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.associated_type_bindings.deinit();
    }
    
    pub fn addMethod(self: *ProtocolImpl, allocator: Allocator, method: MethodImpl) !void {
        try self.methods.append(allocator, method);
    }
    
    pub fn bindAssociatedType(
        self: *ProtocolImpl,
        allocator: Allocator,
        type_name: []const u8,
        concrete_type: []const u8,
    ) !void {
        const key = try allocator.dupe(u8, type_name);
        const value = try allocator.dupe(u8, concrete_type);
        try self.associated_type_bindings.put(key, value);
    }
};

/// Method implementation
pub const MethodImpl = struct {
    name: []const u8,
    function_name: []const u8, // Actual function implementing the method
    
    pub fn init(allocator: Allocator, name: []const u8, function_name: []const u8) !MethodImpl {
        return MethodImpl{
            .name = try allocator.dupe(u8, name),
            .function_name = try allocator.dupe(u8, function_name),
        };
    }
    
    pub fn deinit(self: *MethodImpl, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.function_name);
    }
};

// ============================================================================
// Implementation Validator
// ============================================================================

/// Validates protocol implementations
pub const ImplValidator = struct {
    allocator: Allocator,
    protocols: *protocols.ProtocolRegistry,
    errors: ArrayList(ValidationError),
    
    pub const ValidationError = struct {
        message: []const u8,
        location: ProtocolImpl.SourceLocation,
        
        pub fn init(allocator: Allocator, message: []const u8, location: ProtocolImpl.SourceLocation) !ValidationError {
            return ValidationError{
                .message = try allocator.dupe(u8, message),
                .location = location,
            };
        }
        
        pub fn deinit(self: *ValidationError, allocator: Allocator) void {
            allocator.free(self.message);
        }
    };
    
    pub fn init(allocator: Allocator, protocol_registry: *protocols.ProtocolRegistry) ImplValidator {
        return ImplValidator{
            .allocator = allocator,
            .protocols = protocol_registry,
            .errors = ArrayList(ValidationError){},
        };
    }
    
    pub fn deinit(self: *ImplValidator) void {
        for (self.errors.items) |*err| {
            err.deinit(self.allocator);
        }
        self.errors.deinit(self.allocator);
    }
    
    /// Validate that implementation satisfies protocol
    pub fn validateImpl(self: *ImplValidator, impl: ProtocolImpl) !bool {
        // Get protocol definition
        const protocol = self.protocols.getProtocol(impl.protocol_name) orelse {
            const err = try ValidationError.init(
                self.allocator,
                "Protocol not found",
                impl.source_location,
            );
            try self.errors.append(self.allocator, err);
            return false;
        };
        
        // Check all required methods are implemented
        if (!try self.checkRequiredMethods(impl, protocol)) {
            return false;
        }
        
        // Check associated type bindings
        if (!try self.checkAssociatedTypes(impl, protocol)) {
            return false;
        }
        
        return self.errors.items.len == 0;
    }
    
    fn checkRequiredMethods(self: *ImplValidator, impl: ProtocolImpl, protocol: Protocol) !bool {
        for (protocol.requirements.items) |req| {
            switch (req) {
                .Method => |method| {
                    var found = false;
                    for (impl.methods.items) |impl_method| {
                        if (std.mem.eql(u8, impl_method.name, method.name)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        const err = try ValidationError.init(
                            self.allocator,
                            "Missing required method implementation",
                            impl.source_location,
                        );
                        try self.errors.append(self.allocator, err);
                        return false;
                    }
                },
                else => {},
            }
        }
        return true;
    }
    
    fn checkAssociatedTypes(self: *ImplValidator, impl: ProtocolImpl, protocol: Protocol) !bool {
        for (protocol.associated_types.items) |assoc_type| {
            if (!impl.associated_type_bindings.contains(assoc_type.name)) {
                const err = try ValidationError.init(
                    self.allocator,
                    "Missing associated type binding",
                    impl.source_location,
                );
                try self.errors.append(self.allocator, err);
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Dispatch Table Builder
// ============================================================================

/// Builds method dispatch tables
pub const DispatchTableBuilder = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DispatchTableBuilder {
        return DispatchTableBuilder{ .allocator = allocator };
    }
    
    /// Build dispatch table for implementation
    pub fn buildTable(self: *DispatchTableBuilder, impl: ProtocolImpl) !DispatchTable {
        var table = DispatchTable.init(self.allocator);
        
        for (impl.methods.items) |method| {
            const entry = DispatchTable.DispatchEntry{
                .method_name = try self.allocator.dupe(u8, method.name),
                .function_name = try self.allocator.dupe(u8, method.function_name),
            };
            try table.addEntry(self.allocator, entry);
        }
        
        return table;
    }
};

/// Dispatch table for protocol methods
pub const DispatchTable = struct {
    entries: ArrayList(DispatchEntry),
    
    pub const DispatchEntry = struct {
        method_name: []const u8,
        function_name: []const u8,
    };
    
    pub fn init(allocator: Allocator) DispatchTable {
        _ = allocator;
        return DispatchTable{
            .entries = ArrayList(DispatchEntry){},
        };
    }
    
    pub fn deinit(self: *DispatchTable, allocator: Allocator) void {
        for (self.entries.items) |entry| {
            allocator.free(entry.method_name);
            allocator.free(entry.function_name);
        }
        self.entries.deinit(allocator);
    }
    
    pub fn addEntry(self: *DispatchTable, allocator: Allocator, entry: DispatchEntry) !void {
        try self.entries.append(allocator, entry);
    }
    
    pub fn lookup(self: *DispatchTable, method_name: []const u8) ?[]const u8 {
        for (self.entries.items) |entry| {
            if (std.mem.eql(u8, entry.method_name, method_name)) {
                return entry.function_name;
            }
        }
        return null;
    }
};

// ============================================================================
// Witness Table (Runtime Dispatch)
// ============================================================================

/// Witness table for dynamic protocol dispatch
pub const WitnessTable = struct {
    protocol_name: []const u8,
    type_name: []const u8,
    vtable: ArrayList(VTableEntry),
    
    pub const VTableEntry = struct {
        method_name: []const u8,
        function_ptr: usize, // Function pointer (placeholder)
    };
    
    pub fn init(allocator: Allocator, protocol_name: []const u8, type_name: []const u8) !WitnessTable {
        return WitnessTable{
            .protocol_name = try allocator.dupe(u8, protocol_name),
            .type_name = try allocator.dupe(u8, type_name),
            .vtable = ArrayList(VTableEntry){},
        };
    }
    
    pub fn deinit(self: *WitnessTable, allocator: Allocator) void {
        allocator.free(self.protocol_name);
        allocator.free(self.type_name);
        for (self.vtable.items) |entry| {
            allocator.free(entry.method_name);
        }
        self.vtable.deinit(allocator);
    }
    
    pub fn addMethod(self: *WitnessTable, allocator: Allocator, method_name: []const u8, function_ptr: usize) !void {
        const entry = VTableEntry{
            .method_name = try allocator.dupe(u8, method_name),
            .function_ptr = function_ptr,
        };
        try self.vtable.append(allocator, entry);
    }
    
    pub fn getMethodPtr(self: *WitnessTable, method_name: []const u8) ?usize {
        for (self.vtable.items) |entry| {
            if (std.mem.eql(u8, entry.method_name, method_name)) {
                return entry.function_ptr;
            }
        }
        return null;
    }
};

// ============================================================================
// Implementation Registry
// ============================================================================

/// Registry for all protocol implementations
pub const ImplRegistry = struct {
    allocator: Allocator,
    implementations: ArrayList(ProtocolImpl),
    dispatch_tables: ArrayList(DispatchTable),
    
    pub fn init(allocator: Allocator) ImplRegistry {
        return ImplRegistry{
            .allocator = allocator,
            .implementations = ArrayList(ProtocolImpl){},
            .dispatch_tables = ArrayList(DispatchTable){},
        };
    }
    
    pub fn deinit(self: *ImplRegistry) void {
        for (self.implementations.items) |*impl| {
            impl.deinit(self.allocator);
        }
        self.implementations.deinit(self.allocator);
        for (self.dispatch_tables.items) |*table| {
            table.deinit(self.allocator);
        }
        self.dispatch_tables.deinit(self.allocator);
    }
    
    pub fn registerImpl(self: *ImplRegistry, impl: ProtocolImpl) !void {
        try self.implementations.append(self.allocator, impl);
    }
    
    pub fn findImpl(self: *ImplRegistry, protocol_name: []const u8, type_name: []const u8) ?ProtocolImpl {
        for (self.implementations.items) |impl| {
            if (std.mem.eql(u8, impl.protocol_name, protocol_name) and
                std.mem.eql(u8, impl.type_name, type_name))
            {
                return impl;
            }
        }
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "protocol impl creation" {
    const allocator = std.testing.allocator;
    
    var impl = try ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    try std.testing.expectEqualStrings("Drawable", impl.protocol_name);
    try std.testing.expectEqualStrings("Circle", impl.type_name);
}

test "add method to impl" {
    const allocator = std.testing.allocator;
    
    var impl = try ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    const method = try MethodImpl.init(allocator, "draw", "circle_draw");
    try impl.addMethod(allocator, method);
    
    try std.testing.expectEqual(@as(usize, 1), impl.methods.items.len);
}

test "bind associated type" {
    const allocator = std.testing.allocator;
    
    var impl = try ProtocolImpl.init(allocator, "Container", "Vec", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    try impl.bindAssociatedType(allocator, "Item", "Int");
    
    const binding = impl.associated_type_bindings.get("Item");
    try std.testing.expect(binding != null);
    try std.testing.expectEqualStrings("Int", binding.?);
}

test "impl validator checks required methods" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    var protocol = try Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const method_req = try MethodRequirement.init(allocator, "draw", false);
    try protocol.addRequirement(allocator, .{ .Method = method_req });
    try registry.register(protocol);
    
    var validator = ImplValidator.init(allocator, &registry);
    defer validator.deinit();
    
    var impl = try ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 10,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    const method_impl = try MethodImpl.init(allocator, "draw", "circle_draw");
    try impl.addMethod(allocator, method_impl);
    
    const valid = try validator.validateImpl(impl);
    try std.testing.expect(valid);
}

test "impl validator detects missing methods" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    var protocol = try Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const method_req = try MethodRequirement.init(allocator, "draw", false);
    try protocol.addRequirement(allocator, .{ .Method = method_req });
    try registry.register(protocol);
    
    var validator = ImplValidator.init(allocator, &registry);
    defer validator.deinit();
    
    var impl = try ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 10,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    // No methods added - should fail
    const valid = try validator.validateImpl(impl);
    try std.testing.expect(!valid);
    try std.testing.expect(validator.errors.items.len > 0);
}

test "dispatch table builder" {
    const allocator = std.testing.allocator;
    
    var impl = try ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    const method = try MethodImpl.init(allocator, "draw", "circle_draw");
    try impl.addMethod(allocator, method);
    
    var builder = DispatchTableBuilder.init(allocator);
    var table = try builder.buildTable(impl);
    defer table.deinit(allocator);
    
    const func = table.lookup("draw");
    try std.testing.expect(func != null);
    try std.testing.expectEqualStrings("circle_draw", func.?);
}

test "witness table creation" {
    const allocator = std.testing.allocator;
    
    var witness = try WitnessTable.init(allocator, "Drawable", "Circle");
    defer witness.deinit(allocator);
    
    try witness.addMethod(allocator, "draw", 0x12345678);
    
    const ptr = witness.getMethodPtr("draw");
    try std.testing.expect(ptr != null);
    try std.testing.expectEqual(@as(usize, 0x12345678), ptr.?);
}

test "impl registry" {
    const allocator = std.testing.allocator;
    
    var registry = ImplRegistry.init(allocator);
    defer registry.deinit();
    
    const impl = try ProtocolImpl.init(allocator, "Drawable", "Circle", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    
    try registry.registerImpl(impl);
    
    const found = registry.findImpl("Drawable", "Circle");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("Circle", found.?.type_name);
}

test "impl validator checks associated types" {
    const allocator = std.testing.allocator;
    
    var registry = protocols.ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    var protocol = try Protocol.init(allocator, "Container", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    const assoc = try protocols.AssociatedType.init(allocator, "Item");
    try protocol.addAssociatedType(allocator, assoc);
    try registry.register(protocol);
    
    var validator = ImplValidator.init(allocator, &registry);
    defer validator.deinit();
    
    var impl = try ProtocolImpl.init(allocator, "Container", "Vec", .{
        .file = "test.mojo",
        .line = 10,
        .column = 1,
    });
    defer impl.deinit(allocator);
    
    try impl.bindAssociatedType(allocator, "Item", "Int");
    
    const valid = try validator.validateImpl(impl);
    try std.testing.expect(valid);
}
