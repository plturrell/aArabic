// Protocol Infrastructure
// Day 63: Protocol definition and checking infrastructure

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// ============================================================================
// Protocol Definition
// ============================================================================

/// Protocol definition (similar to Rust traits or Swift protocols)
pub const Protocol = struct {
    name: []const u8,
    requirements: ArrayList(ProtocolRequirement),
    parent_protocols: ArrayList([]const u8), // Protocol inheritance
    associated_types: ArrayList(AssociatedType),
    source_location: SourceLocation,
    
    pub const SourceLocation = struct {
        file: []const u8,
        line: u32,
        column: u32,
    };
    
    pub fn init(allocator: Allocator, name: []const u8, location: SourceLocation) !Protocol {
        return Protocol{
            .name = try allocator.dupe(u8, name),
            .requirements = ArrayList(ProtocolRequirement){},
            .parent_protocols = ArrayList([]const u8){},
            .associated_types = ArrayList(AssociatedType){},
            .source_location = location,
        };
    }
    
    pub fn deinit(self: *Protocol, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.requirements.items) |*req| {
            req.deinit(allocator);
        }
        self.requirements.deinit(allocator);
        for (self.parent_protocols.items) |parent| {
            allocator.free(parent);
        }
        self.parent_protocols.deinit(allocator);
        for (self.associated_types.items) |*assoc| {
            assoc.deinit(allocator);
        }
        self.associated_types.deinit(allocator);
    }
    
    pub fn addRequirement(self: *Protocol, allocator: Allocator, requirement: ProtocolRequirement) !void {
        try self.requirements.append(allocator, requirement);
    }
    
    pub fn addParent(self: *Protocol, allocator: Allocator, parent_name: []const u8) !void {
        const name = try allocator.dupe(u8, parent_name);
        try self.parent_protocols.append(allocator, name);
    }
    
    pub fn addAssociatedType(self: *Protocol, allocator: Allocator, assoc_type: AssociatedType) !void {
        try self.associated_types.append(allocator, assoc_type);
    }
};

/// Protocol requirements (methods, properties, etc.)
pub const ProtocolRequirement = union(enum) {
    Method: MethodRequirement,
    Property: PropertyRequirement,
    AssociatedType: []const u8,
    
    pub fn deinit(self: *ProtocolRequirement, allocator: Allocator) void {
        switch (self.*) {
            .Method => |*m| m.deinit(allocator),
            .Property => |*p| p.deinit(allocator),
            .AssociatedType => |name| allocator.free(name),
        }
    }
};

/// Method requirement in a protocol
pub const MethodRequirement = struct {
    name: []const u8,
    parameters: ArrayList(Parameter),
    return_type: ?[]const u8,
    is_mutating: bool, // Takes &mut self vs &self
    
    pub const Parameter = struct {
        name: []const u8,
        type_name: []const u8,
        
        pub fn init(allocator: Allocator, name: []const u8, type_name: []const u8) !Parameter {
            return Parameter{
                .name = try allocator.dupe(u8, name),
                .type_name = try allocator.dupe(u8, type_name),
            };
        }
        
        pub fn deinit(self: *Parameter, allocator: Allocator) void {
            allocator.free(self.name);
            allocator.free(self.type_name);
        }
    };
    
    pub fn init(allocator: Allocator, name: []const u8, is_mutating: bool) !MethodRequirement {
        return MethodRequirement{
            .name = try allocator.dupe(u8, name),
            .parameters = ArrayList(Parameter){},
            .return_type = null,
            .is_mutating = is_mutating,
        };
    }
    
    pub fn deinit(self: *MethodRequirement, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.parameters.items) |*param| {
            param.deinit(allocator);
        }
        self.parameters.deinit(allocator);
        if (self.return_type) |ret| {
            allocator.free(ret);
        }
    }
    
    pub fn addParameter(self: *MethodRequirement, allocator: Allocator, param: Parameter) !void {
        try self.parameters.append(allocator, param);
    }
    
    pub fn setReturnType(self: *MethodRequirement, allocator: Allocator, return_type: []const u8) !void {
        if (self.return_type) |old| {
            allocator.free(old);
        }
        self.return_type = try allocator.dupe(u8, return_type);
    }
};

/// Property requirement in a protocol
pub const PropertyRequirement = struct {
    name: []const u8,
    type_name: []const u8,
    is_readonly: bool,
    
    pub fn init(allocator: Allocator, name: []const u8, type_name: []const u8, is_readonly: bool) !PropertyRequirement {
        return PropertyRequirement{
            .name = try allocator.dupe(u8, name),
            .type_name = try allocator.dupe(u8, type_name),
            .is_readonly = is_readonly,
        };
    }
    
    pub fn deinit(self: *PropertyRequirement, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.type_name);
    }
};

/// Associated type in a protocol
pub const AssociatedType = struct {
    name: []const u8,
    constraints: ArrayList([]const u8), // Protocol constraints
    
    pub fn init(allocator: Allocator, name: []const u8) !AssociatedType {
        return AssociatedType{
            .name = try allocator.dupe(u8, name),
            .constraints = ArrayList([]const u8){},
        };
    }
    
    pub fn deinit(self: *AssociatedType, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.constraints.items) |constraint| {
            allocator.free(constraint);
        }
        self.constraints.deinit(allocator);
    }
    
    pub fn addConstraint(self: *AssociatedType, allocator: Allocator, constraint: []const u8) !void {
        const c = try allocator.dupe(u8, constraint);
        try self.constraints.append(allocator, c);
    }
};

// ============================================================================
// Protocol Checker
// ============================================================================

/// Validates protocol definitions
pub const ProtocolChecker = struct {
    allocator: Allocator,
    protocols: StringHashMap(Protocol),
    errors: ArrayList(CheckError),
    
    pub const CheckError = struct {
        message: []const u8,
        location: Protocol.SourceLocation,
        
        pub fn init(allocator: Allocator, message: []const u8, location: Protocol.SourceLocation) !CheckError {
            return CheckError{
                .message = try allocator.dupe(u8, message),
                .location = location,
            };
        }
        
        pub fn deinit(self: *CheckError, allocator: Allocator) void {
            allocator.free(self.message);
        }
    };
    
    pub fn init(allocator: Allocator) ProtocolChecker {
        return ProtocolChecker{
            .allocator = allocator,
            .protocols = StringHashMap(Protocol).init(allocator),
            .errors = ArrayList(CheckError){},
        };
    }
    
    pub fn deinit(self: *ProtocolChecker) void {
        var it = self.protocols.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.protocols.deinit();
        for (self.errors.items) |*err| {
            err.deinit(self.allocator);
        }
        self.errors.deinit(self.allocator);
    }
    
    /// Register a protocol
    pub fn registerProtocol(self: *ProtocolChecker, protocol: Protocol) !void {
        const key = try self.allocator.dupe(u8, protocol.name);
        try self.protocols.put(key, protocol);
    }
    
    /// Check if protocol is valid
    pub fn checkProtocol(self: *ProtocolChecker, protocol_name: []const u8) !bool {
        const protocol = self.protocols.get(protocol_name) orelse {
                    const err = try CheckError.init(
                        self.allocator,
                        "Protocol not found",
                        .{ .file = "unknown", .line = 0, .column = 0 },
                    );
                    try self.errors.append(self.allocator, err);
            return false;
        };
        
        // Check for duplicate method names
        if (!try self.checkUniqueMethodNames(protocol)) {
            return false;
        }
        
        // Check inheritance hierarchy
        if (!try self.checkInheritance(protocol)) {
            return false;
        }
        
        // Check associated types
        if (!try self.checkAssociatedTypes(protocol)) {
            return false;
        }
        
        return self.errors.items.len == 0;
    }
    
    fn checkUniqueMethodNames(self: *ProtocolChecker, protocol: Protocol) !bool {
        var seen = StringHashMap(void).init(self.allocator);
        defer {
            var it = seen.keyIterator();
            while (it.next()) |key| {
                self.allocator.free(key.*);
            }
            seen.deinit();
        }
        
        for (protocol.requirements.items) |req| {
            switch (req) {
                .Method => |method| {
                    if (seen.contains(method.name)) {
                        const err = try CheckError.init(
                            self.allocator,
                            "Duplicate method name in protocol",
                            protocol.source_location,
                        );
                        try self.errors.append(self.allocator, err);
                        return false;
                    }
                    const key = try self.allocator.dupe(u8, method.name);
                    try seen.put(key, {});
                },
                else => {},
            }
        }
        
        return true;
    }
    
    fn checkInheritance(self: *ProtocolChecker, protocol: Protocol) !bool {
        // Check for circular inheritance
        var visited = StringHashMap(void).init(self.allocator);
        defer {
            var it = visited.keyIterator();
            while (it.next()) |key| {
                self.allocator.free(key.*);
            }
            visited.deinit();
        }
        
        return try self.checkInheritanceCycle(protocol.name, &visited);
    }
    
    fn checkInheritanceCycle(
        self: *ProtocolChecker,
        protocol_name: []const u8,
        visited: *StringHashMap(void),
    ) !bool {
        if (visited.contains(protocol_name)) {
            const err = try CheckError.init(
                self.allocator,
                "Circular protocol inheritance detected",
                .{ .file = "unknown", .line = 0, .column = 0 },
            );
            try self.errors.append(self.allocator, err);
            return false;
        }
        
        const key = try self.allocator.dupe(u8, protocol_name);
        try visited.put(key, {});
        
        const protocol = self.protocols.get(protocol_name) orelse return true;
        
        for (protocol.parent_protocols.items) |parent_name| {
            if (!try self.checkInheritanceCycle(parent_name, visited)) {
                return false;
            }
        }
        
        return true;
    }
    
    fn checkAssociatedTypes(self: *ProtocolChecker, protocol: Protocol) !bool {
        _ = protocol;
        _ = self;
        // Associated type validation will be implemented
        return true;
    }
    
    /// Check if method signature is valid
    pub fn validateMethodSignature(self: *ProtocolChecker, method: MethodRequirement) bool {
        _ = self;
        
        // Check that method has a name
        if (method.name.len == 0) {
            return false;
        }
        
        // Check parameters are valid
        for (method.parameters.items) |param| {
            if (param.name.len == 0 or param.type_name.len == 0) {
                return false;
            }
        }
        
        return true;
    }
};

// ============================================================================
// Protocol Registry
// ============================================================================

/// Global protocol registry
pub const ProtocolRegistry = struct {
    checker: ProtocolChecker,
    
    pub fn init(allocator: Allocator) ProtocolRegistry {
        return ProtocolRegistry{
            .checker = ProtocolChecker.init(allocator),
        };
    }
    
    pub fn deinit(self: *ProtocolRegistry) void {
        self.checker.deinit();
    }
    
    pub fn register(self: *ProtocolRegistry, protocol: Protocol) !void {
        try self.checker.registerProtocol(protocol);
    }
    
    pub fn check(self: *ProtocolRegistry, protocol_name: []const u8) !bool {
        return try self.checker.checkProtocol(protocol_name);
    }
    
    pub fn getProtocol(self: *ProtocolRegistry, name: []const u8) ?Protocol {
        return self.checker.protocols.get(name);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "protocol creation" {
    const allocator = std.testing.allocator;
    
    var protocol = try Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer protocol.deinit(allocator);
    
    try std.testing.expectEqualStrings("Drawable", protocol.name);
    try std.testing.expectEqual(@as(usize, 0), protocol.requirements.items.len);
}

test "method requirement" {
    const allocator = std.testing.allocator;
    
    var method = try MethodRequirement.init(allocator, "draw", false);
    defer method.deinit(allocator);
    
    const param = try MethodRequirement.Parameter.init(allocator, "self", "Self");
    try method.addParameter(allocator, param);
    
    try std.testing.expectEqualStrings("draw", method.name);
    try std.testing.expectEqual(@as(usize, 1), method.parameters.items.len);
    try std.testing.expect(!method.is_mutating);
}

test "protocol with method" {
    const allocator = std.testing.allocator;
    
    var protocol = try Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer protocol.deinit(allocator);
    
    var method = try MethodRequirement.init(allocator, "draw", false);
    const param = try MethodRequirement.Parameter.init(allocator, "self", "Self");
    try method.addParameter(allocator, param);
    
    try protocol.addRequirement(allocator, .{ .Method = method });
    
    try std.testing.expectEqual(@as(usize, 1), protocol.requirements.items.len);
}

test "protocol inheritance" {
    const allocator = std.testing.allocator;
    
    var protocol = try Protocol.init(allocator, "Shape", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    defer protocol.deinit(allocator);
    
    try protocol.addParent(allocator, "Drawable");
    
    try std.testing.expectEqual(@as(usize, 1), protocol.parent_protocols.items.len);
    try std.testing.expectEqualStrings("Drawable", protocol.parent_protocols.items[0]);
}

test "associated type" {
    const allocator = std.testing.allocator;
    
    var assoc = try AssociatedType.init(allocator, "Item");
    defer assoc.deinit(allocator);
    
    try assoc.addConstraint(allocator, "Equatable");
    
    try std.testing.expectEqualStrings("Item", assoc.name);
    try std.testing.expectEqual(@as(usize, 1), assoc.constraints.items.len);
}

test "protocol checker validates unique methods" {
    const allocator = std.testing.allocator;
    
    var checker = ProtocolChecker.init(allocator);
    defer checker.deinit();
    
    var protocol = try Protocol.init(allocator, "Test", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    
    const method1 = try MethodRequirement.init(allocator, "foo", false);
    try protocol.addRequirement(allocator, .{ .Method = method1 });
    
    const method2 = try MethodRequirement.init(allocator, "bar", false);
    try protocol.addRequirement(allocator, .{ .Method = method2 });
    
    try checker.registerProtocol(protocol);
    const valid = try checker.checkProtocol("Test");
    
    try std.testing.expect(valid);
}

test "protocol checker detects duplicate methods" {
    const allocator = std.testing.allocator;
    
    var checker = ProtocolChecker.init(allocator);
    defer checker.deinit();
    
    var protocol = try Protocol.init(allocator, "Test", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    
    const method1 = try MethodRequirement.init(allocator, "foo", false);
    try protocol.addRequirement(allocator, .{ .Method = method1 });
    
    const method2 = try MethodRequirement.init(allocator, "foo", false);
    try protocol.addRequirement(allocator, .{ .Method = method2 });
    
    try checker.registerProtocol(protocol);
    const valid = try checker.checkProtocol("Test");
    
    try std.testing.expect(!valid);
    try std.testing.expect(checker.errors.items.len > 0);
}

test "protocol registry" {
    const allocator = std.testing.allocator;
    
    var registry = ProtocolRegistry.init(allocator);
    defer registry.deinit();
    
    const protocol = try Protocol.init(allocator, "Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
    });
    
    try registry.register(protocol);
    
    const found = registry.getProtocol("Drawable");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("Drawable", found.?.name);
}
