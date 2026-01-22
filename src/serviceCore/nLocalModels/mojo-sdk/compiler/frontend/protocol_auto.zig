// Automatic Protocol Conformance
// Day 65: Automatic implementation generation and derive macros

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const protocols = @import("protocols.zig");
const protocol_impl = @import("protocol_impl.zig");
const Protocol = protocols.Protocol;
const ProtocolImpl = protocol_impl.ProtocolImpl;
const MethodImpl = protocol_impl.MethodImpl;

// ============================================================================
// Derivable Protocols
// ============================================================================

/// Common protocols that can be auto-derived
pub const DerivableProtocol = enum {
    Eq,        // Equality comparison
    Hash,      // Hashing
    Debug,     // Debug formatting
    Clone,     // Deep copy
    Default,   // Default value
    
    pub fn toString(self: DerivableProtocol) []const u8 {
        return switch (self) {
            .Eq => "Eq",
            .Hash => "Hash",
            .Debug => "Debug",
            .Clone => "Clone",
            .Default => "Default",
        };
    }
    
    pub fn fromString(name: []const u8) ?DerivableProtocol {
        if (std.mem.eql(u8, name, "Eq")) return .Eq;
        if (std.mem.eql(u8, name, "Hash")) return .Hash;
        if (std.mem.eql(u8, name, "Debug")) return .Debug;
        if (std.mem.eql(u8, name, "Clone")) return .Clone;
        if (std.mem.eql(u8, name, "Default")) return .Default;
        return null;
    }
};

// ============================================================================
// Type Information for Derivation
// ============================================================================

/// Information about a type for auto-derivation
pub const TypeInfo = struct {
    name: []const u8,
    fields: ArrayList(FieldInfo),
    
    pub const FieldInfo = struct {
        name: []const u8,
        type_name: []const u8,
        
        pub fn init(allocator: Allocator, name: []const u8, type_name: []const u8) !FieldInfo {
            return FieldInfo{
                .name = try allocator.dupe(u8, name),
                .type_name = try allocator.dupe(u8, type_name),
            };
        }
        
        pub fn deinit(self: *FieldInfo, allocator: Allocator) void {
            allocator.free(self.name);
            allocator.free(self.type_name);
        }
    };
    
    pub fn init(allocator: Allocator, name: []const u8) !TypeInfo {
        return TypeInfo{
            .name = try allocator.dupe(u8, name),
            .fields = ArrayList(FieldInfo){},
        };
    }
    
    pub fn deinit(self: *TypeInfo, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.fields.items) |*field| {
            field.deinit(allocator);
        }
        self.fields.deinit(allocator);
    }
    
    pub fn addField(self: *TypeInfo, allocator: Allocator, field: FieldInfo) !void {
        try self.fields.append(allocator, field);
    }
};

// ============================================================================
// Auto-Conformance Analyzer
// ============================================================================

/// Analyzes types for automatic protocol conformance
pub const AutoConformanceAnalyzer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) AutoConformanceAnalyzer {
        return AutoConformanceAnalyzer{ .allocator = allocator };
    }
    
    /// Check if type can auto-derive a protocol
    pub fn canDerive(self: *AutoConformanceAnalyzer, type_info: TypeInfo, protocol: DerivableProtocol) bool {
        _ = self;
        return switch (protocol) {
            .Eq => canDeriveEq(type_info),
            .Hash => canDeriveHash(type_info),
            .Debug => true, // Debug can always be derived
            .Clone => canDeriveClone(type_info),
            .Default => canDeriveDefault(type_info),
        };
    }
    
    fn canDeriveEq(type_info: TypeInfo) bool {
        // Can derive Eq if all fields are comparable
        // In a real implementation, we'd check field types
        _ = type_info;
        return true;
    }
    
    fn canDeriveHash(type_info: TypeInfo) bool {
        // Can derive Hash if all fields are hashable
        _ = type_info;
        return true;
    }
    
    fn canDeriveClone(type_info: TypeInfo) bool {
        // Can derive Clone if all fields are cloneable
        _ = type_info;
        return true;
    }
    
    fn canDeriveDefault(type_info: TypeInfo) bool {
        // Can derive Default if all fields have defaults
        _ = type_info;
        return true;
    }
};

// ============================================================================
// Code Generator
// ============================================================================

/// Generates implementation code for auto-derived protocols
pub const CodeGenerator = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CodeGenerator {
        return CodeGenerator{ .allocator = allocator };
    }
    
    /// Generate implementation for a derivable protocol
    pub fn generateImpl(
        self: *CodeGenerator,
        type_info: TypeInfo,
        protocol: DerivableProtocol,
    ) !GeneratedCode {
        return switch (protocol) {
            .Eq => try self.generateEq(type_info),
            .Hash => try self.generateHash(type_info),
            .Debug => try self.generateDebug(type_info),
            .Clone => try self.generateClone(type_info),
            .Default => try self.generateDefault(type_info),
        };
    }
    
    fn generateEq(self: *CodeGenerator, type_info: TypeInfo) !GeneratedCode {
        var methods = ArrayList(GeneratedMethod){};
        
        // Generate eq method
        var code = ArrayList(u8){};
        try code.appendSlice(self.allocator, "fn eq(self, other: Self) -> Bool {\n");
        
        if (type_info.fields.items.len == 0) {
            try code.appendSlice(self.allocator, "    return true\n");
        } else {
            for (type_info.fields.items, 0..) |field, i| {
                if (i > 0) {
                    try code.appendSlice(self.allocator, " and ");
                } else {
                    try code.appendSlice(self.allocator, "    return ");
                }
                try code.appendSlice(self.allocator, "self.");
                try code.appendSlice(self.allocator, field.name);
                try code.appendSlice(self.allocator, " == other.");
                try code.appendSlice(self.allocator, field.name);
            }
            try code.appendSlice(self.allocator, "\n");
        }
        
        try code.appendSlice(self.allocator, "}\n");
        
        const method = GeneratedMethod{
            .name = try self.allocator.dupe(u8, "eq"),
            .code = try code.toOwnedSlice(self.allocator),
        };
        try methods.append(self.allocator, method);
        
        return GeneratedCode{
            .protocol_name = try self.allocator.dupe(u8, "Eq"),
            .methods = methods,
        };
    }
    
    fn generateHash(self: *CodeGenerator, type_info: TypeInfo) !GeneratedCode {
        var methods = ArrayList(GeneratedMethod){};
        
        var code = ArrayList(u8){};
        try code.appendSlice(self.allocator, "fn hash(self) -> UInt64 {\n");
        try code.appendSlice(self.allocator, "    var h: UInt64 = 0\n");
        
        for (type_info.fields.items) |field| {
            try code.appendSlice(self.allocator, "    h = h ^ hash(self.");
            try code.appendSlice(self.allocator, field.name);
            try code.appendSlice(self.allocator, ")\n");
        }
        
        try code.appendSlice(self.allocator, "    return h\n}\n");
        
        const method = GeneratedMethod{
            .name = try self.allocator.dupe(u8, "hash"),
            .code = try code.toOwnedSlice(self.allocator),
        };
        try methods.append(self.allocator, method);
        
        return GeneratedCode{
            .protocol_name = try self.allocator.dupe(u8, "Hash"),
            .methods = methods,
        };
    }
    
    fn generateDebug(self: *CodeGenerator, type_info: TypeInfo) !GeneratedCode {
        var methods = ArrayList(GeneratedMethod){};
        
        var code = ArrayList(u8){};
        try code.appendSlice(self.allocator, "fn debug(self) -> String {\n");
        try code.appendSlice(self.allocator, "    var s = \"");
        try code.appendSlice(self.allocator, type_info.name);
        try code.appendSlice(self.allocator, " { \"\n");
        
        for (type_info.fields.items, 0..) |field, i| {
            if (i > 0) {
                try code.appendSlice(self.allocator, "    s += \", \"\n");
            }
            try code.appendSlice(self.allocator, "    s += \"");
            try code.appendSlice(self.allocator, field.name);
            try code.appendSlice(self.allocator, ": \" + debug(self.");
            try code.appendSlice(self.allocator, field.name);
            try code.appendSlice(self.allocator, ")\n");
        }
        
        try code.appendSlice(self.allocator, "    s += \" }\"\n");
        try code.appendSlice(self.allocator, "    return s\n}\n");
        
        const method = GeneratedMethod{
            .name = try self.allocator.dupe(u8, "debug"),
            .code = try code.toOwnedSlice(self.allocator),
        };
        try methods.append(self.allocator, method);
        
        return GeneratedCode{
            .protocol_name = try self.allocator.dupe(u8, "Debug"),
            .methods = methods,
        };
    }
    
    fn generateClone(self: *CodeGenerator, type_info: TypeInfo) !GeneratedCode {
        var methods = ArrayList(GeneratedMethod){};
        
        var code = ArrayList(u8){};
        try code.appendSlice(self.allocator, "fn clone(self) -> Self {\n");
        try code.appendSlice(self.allocator, "    return ");
        try code.appendSlice(self.allocator, type_info.name);
        try code.appendSlice(self.allocator, " {\n");
        
        for (type_info.fields.items) |field| {
            try code.appendSlice(self.allocator, "        ");
            try code.appendSlice(self.allocator, field.name);
            try code.appendSlice(self.allocator, ": clone(self.");
            try code.appendSlice(self.allocator, field.name);
            try code.appendSlice(self.allocator, "),\n");
        }
        
        try code.appendSlice(self.allocator, "    }\n}\n");
        
        const method = GeneratedMethod{
            .name = try self.allocator.dupe(u8, "clone"),
            .code = try code.toOwnedSlice(self.allocator),
        };
        try methods.append(self.allocator, method);
        
        return GeneratedCode{
            .protocol_name = try self.allocator.dupe(u8, "Clone"),
            .methods = methods,
        };
    }
    
    fn generateDefault(self: *CodeGenerator, type_info: TypeInfo) !GeneratedCode {
        var methods = ArrayList(GeneratedMethod){};
        
        var code = ArrayList(u8){};
        try code.appendSlice(self.allocator, "fn default() -> Self {\n");
        try code.appendSlice(self.allocator, "    return ");
        try code.appendSlice(self.allocator, type_info.name);
        try code.appendSlice(self.allocator, " {\n");
        
        for (type_info.fields.items) |field| {
            try code.appendSlice(self.allocator, "        ");
            try code.appendSlice(self.allocator, field.name);
            try code.appendSlice(self.allocator, ": ");
            try code.appendSlice(self.allocator, field.type_name);
            try code.appendSlice(self.allocator, "::default(),\n");
        }
        
        try code.appendSlice(self.allocator, "    }\n}\n");
        
        const method = GeneratedMethod{
            .name = try self.allocator.dupe(u8, "default"),
            .code = try code.toOwnedSlice(self.allocator),
        };
        try methods.append(self.allocator, method);
        
        return GeneratedCode{
            .protocol_name = try self.allocator.dupe(u8, "Default"),
            .methods = methods,
        };
    }
};

/// Generated code for a protocol implementation
pub const GeneratedCode = struct {
    protocol_name: []const u8,
    methods: ArrayList(GeneratedMethod),
    
    pub fn deinit(self: *GeneratedCode, allocator: Allocator) void {
        allocator.free(self.protocol_name);
        for (self.methods.items) |*method| {
            method.deinit(allocator);
        }
        self.methods.deinit(allocator);
    }
};

/// Generated method code
pub const GeneratedMethod = struct {
    name: []const u8,
    code: []const u8,
    
    pub fn deinit(self: *GeneratedMethod, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.code);
    }
};

// ============================================================================
// Derive Macro System
// ============================================================================

/// Processes derive macros for types
pub const DeriveMacroProcessor = struct {
    allocator: Allocator,
    analyzer: AutoConformanceAnalyzer,
    generator: CodeGenerator,
    
    pub fn init(allocator: Allocator) DeriveMacroProcessor {
        return DeriveMacroProcessor{
            .allocator = allocator,
            .analyzer = AutoConformanceAnalyzer.init(allocator),
            .generator = CodeGenerator.init(allocator),
        };
    }
    
    /// Process derive macro for a type
    pub fn processDerive(
        self: *DeriveMacroProcessor,
        type_info: TypeInfo,
        protocol_list: []const DerivableProtocol,
    ) !ArrayList(GeneratedCode) {
        var results = ArrayList(GeneratedCode){};
        
        for (protocol_list) |protocol| {
            if (self.analyzer.canDerive(type_info, protocol)) {
                const code = try self.generator.generateImpl(type_info, protocol);
                try results.append(self.allocator, code);
            }
        }
        
        return results;
    }
    
    /// Convert generated code to ProtocolImpl
    pub fn toProtocolImpl(
        self: *DeriveMacroProcessor,
        type_info: TypeInfo,
        generated: GeneratedCode,
    ) !ProtocolImpl {
        var impl = try ProtocolImpl.init(
            self.allocator,
            generated.protocol_name,
            type_info.name,
            .{ .file = "generated", .line = 0, .column = 0 },
        );
        
        for (generated.methods.items) |method| {
            const method_impl = try MethodImpl.init(
                self.allocator,
                method.name,
                method.name, // Function name same as method name for now
            );
            try impl.addMethod(self.allocator, method_impl);
        }
        
        return impl;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "derivable protocol enum" {
    const eq = DerivableProtocol.Eq;
    try std.testing.expectEqualStrings("Eq", eq.toString());
    
    const from_str = DerivableProtocol.fromString("Hash");
    try std.testing.expect(from_str != null);
    try std.testing.expectEqual(DerivableProtocol.Hash, from_str.?);
}

test "type info creation" {
    const allocator = std.testing.allocator;
    
    var type_info = try TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    const field = try TypeInfo.FieldInfo.init(allocator, "x", "Int");
    try type_info.addField(allocator, field);
    
    try std.testing.expectEqualStrings("Point", type_info.name);
    try std.testing.expectEqual(@as(usize, 1), type_info.fields.items.len);
}

test "can derive eq" {
    const allocator = std.testing.allocator;
    
    var analyzer = AutoConformanceAnalyzer.init(allocator);
    
    var type_info = try TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    const field = try TypeInfo.FieldInfo.init(allocator, "x", "Int");
    try type_info.addField(allocator, field);
    
    try std.testing.expect(analyzer.canDerive(type_info, .Eq));
}

test "generate eq implementation" {
    const allocator = std.testing.allocator;
    
    var generator = CodeGenerator.init(allocator);
    
    var type_info = try TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    const field = try TypeInfo.FieldInfo.init(allocator, "x", "Int");
    try type_info.addField(allocator, field);
    
    var code = try generator.generateImpl(type_info, .Eq);
    defer code.deinit(allocator);
    
    try std.testing.expectEqualStrings("Eq", code.protocol_name);
    try std.testing.expectEqual(@as(usize, 1), code.methods.items.len);
    try std.testing.expectEqualStrings("eq", code.methods.items[0].name);
}

test "generate hash implementation" {
    const allocator = std.testing.allocator;
    
    var generator = CodeGenerator.init(allocator);
    
    var type_info = try TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    var code = try generator.generateImpl(type_info, .Hash);
    defer code.deinit(allocator);
    
    try std.testing.expectEqualStrings("Hash", code.protocol_name);
    try std.testing.expect(code.methods.items.len > 0);
}

test "generate debug implementation" {
    const allocator = std.testing.allocator;
    
    var generator = CodeGenerator.init(allocator);
    
    var type_info = try TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    const field = try TypeInfo.FieldInfo.init(allocator, "x", "Int");
    try type_info.addField(allocator, field);
    
    var code = try generator.generateImpl(type_info, .Debug);
    defer code.deinit(allocator);
    
    try std.testing.expectEqualStrings("Debug", code.protocol_name);
    try std.testing.expect(code.methods.items[0].code.len > 0);
}

test "derive macro processor" {
    const allocator = std.testing.allocator;
    
    var processor = DeriveMacroProcessor.init(allocator);
    
    var type_info = try TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    const field = try TypeInfo.FieldInfo.init(allocator, "x", "Int");
    try type_info.addField(allocator, field);
    
    const derive_list = [_]DerivableProtocol{ .Eq, .Hash };
    var results = try processor.processDerive(type_info, &derive_list);
    defer {
        for (results.items) |*result| {
            result.deinit(allocator);
        }
        results.deinit(allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 2), results.items.len);
}

test "convert to protocol impl" {
    const allocator = std.testing.allocator;
    
    var processor = DeriveMacroProcessor.init(allocator);
    
    var type_info = try TypeInfo.init(allocator, "Point");
    defer type_info.deinit(allocator);
    
    var generator = CodeGenerator.init(allocator);
    var generated = try generator.generateImpl(type_info, .Eq);
    defer generated.deinit(allocator);
    
    var impl = try processor.toProtocolImpl(type_info, generated);
    defer impl.deinit(allocator);
    
    try std.testing.expectEqualStrings("Eq", impl.protocol_name);
    try std.testing.expectEqualStrings("Point", impl.type_name);
    try std.testing.expect(impl.methods.items.len > 0);
}
