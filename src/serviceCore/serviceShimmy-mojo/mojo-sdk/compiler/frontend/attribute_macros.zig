// Attribute Macros - Day 128
// @attribute syntax and custom attribute processing

const std = @import("std");
const Allocator = std.mem.Allocator;
const macro_system = @import("macro_system.zig");
const ast = @import("ast.zig");
const TokenStream = macro_system.TokenStream;

// ============================================================================
// Attribute Types
// ============================================================================

pub const AttributeKind = enum {
    inline_hint,       // @inline
    no_inline,         // @noinline
    always_inline,     // @always_inline
    deprecated,        // @deprecated("message")
    test_attr,         // @test
    benchmark,         // @benchmark
    export_attr,       // @export
    extern_attr,       // @extern
    link_name,         // @link_name("c_function")
    align_attr,        // @align(8)
    packed,            // @packed
    repr,              // @repr(C)
    derive,            // @derive(Trait1, Trait2)
    custom,            // @custom_attr(args)
};

pub const Attribute = struct {
    kind: AttributeKind,
    name: []const u8,
    args: ?TokenStream,
    span: SourceSpan,
    
    pub const SourceSpan = struct {
        file: []const u8,
        line: usize,
        column: usize,
    };
    
    pub fn init(kind: AttributeKind, name: []const u8) Attribute {
        return .{
            .kind = kind,
            .name = name,
            .args = null,
            .span = .{ .file = "", .line = 0, .column = 0 },
        };
    }
    
    pub fn withArgs(self: Attribute, args: TokenStream) Attribute {
        var result = self;
        result.args = args;
        return result;
    }
    
    pub fn hasArgs(self: *const Attribute) bool {
        return self.args != null;
    }
};

// ============================================================================
// Attribute List
// ============================================================================

pub const AttributeList = struct {
    attributes: std.ArrayList(Attribute),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) AttributeList {
        return .{
            .attributes = std.ArrayList(Attribute).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *AttributeList) void {
        for (self.attributes.items) |*attr| {
            if (attr.args) |*args| {
                args.deinit();
            }
        }
        self.attributes.deinit();
    }
    
    pub fn add(self: *AttributeList, attr: Attribute) !void {
        try self.attributes.append(attr);
    }
    
    pub fn has(self: *const AttributeList, kind: AttributeKind) bool {
        for (self.attributes.items) |attr| {
            if (attr.kind == kind) return true;
        }
        return false;
    }
    
    pub fn get(self: *const AttributeList, kind: AttributeKind) ?Attribute {
        for (self.attributes.items) |attr| {
            if (attr.kind == kind) return attr;
        }
        return null;
    }
    
    pub fn len(self: *const AttributeList) usize {
        return self.attributes.items.len;
    }
};

// ============================================================================
// Attribute Parser
// ============================================================================

pub const AttributeParser = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) AttributeParser {
        return .{ .allocator = allocator };
    }
    
    pub fn parse(self: *AttributeParser, source: []const u8) !Attribute {
        // Parse @attribute_name or @attribute_name(args)
        
        if (source.len == 0 or source[0] != '@') {
            return error.InvalidAttribute;
        }
        
        // Find attribute name
        var name_end: usize = 1;
        while (name_end < source.len) : (name_end += 1) {
            const ch = source[name_end];
            if (!std.ascii.isAlphanumeric(ch) and ch != '_') break;
        }
        
        const name = source[1..name_end];
        const kind = try self.parseKind(name);
        
        var attr = Attribute.init(kind, name);
        
        // Check for arguments
        if (name_end < source.len and source[name_end] == '(') {
            // TODO: Parse arguments
            var args = TokenStream.init(self.allocator);
            attr.args = args;
        }
        
        return attr;
    }
    
    fn parseKind(self: *AttributeParser, name: []const u8) !AttributeKind {
        _ = self;
        
        if (std.mem.eql(u8, name, "inline")) return .inline_hint;
        if (std.mem.eql(u8, name, "noinline")) return .no_inline;
        if (std.mem.eql(u8, name, "always_inline")) return .always_inline;
        if (std.mem.eql(u8, name, "deprecated")) return .deprecated;
        if (std.mem.eql(u8, name, "test")) return .test_attr;
        if (std.mem.eql(u8, name, "benchmark")) return .benchmark;
        if (std.mem.eql(u8, name, "export")) return .export_attr;
        if (std.mem.eql(u8, name, "extern")) return .extern_attr;
        if (std.mem.eql(u8, name, "link_name")) return .link_name;
        if (std.mem.eql(u8, name, "align")) return .align_attr;
        if (std.mem.eql(u8, name, "packed")) return .packed;
        if (std.mem.eql(u8, name, "repr")) return .repr;
        if (std.mem.eql(u8, name, "derive")) return .derive;
        
        return .custom;
    }
};

// ============================================================================
// Attribute Processor
// ============================================================================

pub const AttributeProcessor = struct {
    allocator: Allocator,
    handlers: std.StringHashMap(AttributeHandler),
    
    pub const AttributeHandler = *const fn (
        allocator: Allocator,
        attr: Attribute,
        item: *ast.Node,
    ) anyerror!void;
    
    pub fn init(allocator: Allocator) AttributeProcessor {
        return .{
            .allocator = allocator,
            .handlers = std.StringHashMap(AttributeHandler).init(allocator),
        };
    }
    
    pub fn deinit(self: *AttributeProcessor) void {
        self.handlers.deinit();
    }
    
    pub fn registerHandler(
        self: *AttributeProcessor,
        name: []const u8,
        handler: AttributeHandler,
    ) !void {
        try self.handlers.put(name, handler);
    }
    
    pub fn process(
        self: *AttributeProcessor,
        attrs: AttributeList,
        item: *ast.Node,
    ) !void {
        for (attrs.attributes.items) |attr| {
            if (self.handlers.get(attr.name)) |handler| {
                try handler(self.allocator, attr, item);
            } else {
                // Unknown attribute - could be warning or error
                std.debug.print("Warning: Unknown attribute @{s}\n", .{attr.name});
            }
        }
    }
};

// ============================================================================
// Built-in Attribute Handlers
// ============================================================================

/// Handle @inline attribute
pub fn handleInline(allocator: Allocator, attr: Attribute, item: *ast.Node) !void {
    _ = allocator;
    _ = attr;
    
    // Mark function for inlining
    if (item.* != .function_decl) {
        return error.InlineOnNonFunction;
    }
    
    // TODO: Set inline hint on function
}

/// Handle @deprecated attribute
pub fn handleDeprecated(allocator: Allocator, attr: Attribute, item: *ast.Node) !void {
    _ = allocator;
    _ = item;
    
    // Extract deprecation message
    if (attr.args) |args| {
        _ = args;
        // TODO: Parse deprecation message
    }
    
    // TODO: Mark item as deprecated
}

/// Handle @export attribute
pub fn handleExport(allocator: Allocator, attr: Attribute, item: *ast.Node) !void {
    _ = allocator;
    _ = attr;
    
    if (item.* != .function_decl) {
        return error.ExportOnNonFunction;
    }
    
    // TODO: Mark function for export
}

/// Handle @link_name attribute
pub fn handleLinkName(allocator: Allocator, attr: Attribute, item: *ast.Node) !void {
    _ = allocator;
    
    if (item.* != .function_decl) {
        return error.LinkNameOnNonFunction;
    }
    
    // Extract link name from args
    if (attr.args == null) {
        return error.LinkNameRequiresArg;
    }
    
    // TODO: Set link name
}

/// Handle @repr attribute
pub fn handleRepr(allocator: Allocator, attr: Attribute, item: *ast.Node) !void {
    _ = allocator;
    
    if (item.* != .struct_decl) {
        return error.ReprOnNonStruct;
    }
    
    // Extract repr kind from args
    if (attr.args == null) {
        return error.ReprRequiresArg;
    }
    
    // TODO: Set struct representation
}

/// Handle @align attribute
pub fn handleAlign(allocator: Allocator, attr: Attribute, item: *ast.Node) !void {
    _ = allocator;
    _ = item;
    
    // Extract alignment from args
    if (attr.args == null) {
        return error.AlignRequiresArg;
    }
    
    // TODO: Set alignment
}

// ============================================================================
// Attribute Composition
// ============================================================================

pub const AttributeComposer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) AttributeComposer {
        return .{ .allocator = allocator };
    }
    
    pub fn compose(
        self: *AttributeComposer,
        attrs: []const Attribute,
    ) !AttributeList {
        var list = AttributeList.init(self.allocator);
        
        for (attrs) |attr| {
            try list.add(attr);
        }
        
        // Validate composition
        try self.validateComposition(&list);
        
        return list;
    }
    
    fn validateComposition(self: *AttributeComposer, list: *AttributeList) !void {
        _ = self;
        
        // Check for conflicting attributes
        const has_inline = list.has(.inline_hint);
        const has_no_inline = list.has(.no_inline);
        
        if (has_inline and has_no_inline) {
            return error.ConflictingAttributes;
        }
        
        // TODO: Add more validation rules
    }
};

// ============================================================================
// Attribute Macro System
// ============================================================================

pub const AttributeMacro = struct {
    name: []const u8,
    handler: Handler,
    
    pub const Handler = *const fn (
        allocator: Allocator,
        attr_args: TokenStream,
        item: TokenStream,
    ) anyerror!TokenStream;
    
    pub fn expand(
        self: *const AttributeMacro,
        allocator: Allocator,
        attr_args: TokenStream,
        item: TokenStream,
    ) !TokenStream {
        return try self.handler(allocator, attr_args, item);
    }
};

pub const AttributeMacroRegistry = struct {
    macros: std.StringHashMap(AttributeMacro),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) AttributeMacroRegistry {
        return .{
            .macros = std.StringHashMap(AttributeMacro).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *AttributeMacroRegistry) void {
        self.macros.deinit();
    }
    
    pub fn register(self: *AttributeMacroRegistry, macro_def: AttributeMacro) !void {
        try self.macros.put(macro_def.name, macro_def);
    }
    
    pub fn get(self: *const AttributeMacroRegistry, name: []const u8) ?AttributeMacro {
        return self.macros.get(name);
    }
};

// ============================================================================
// Example Attribute Macros
// ============================================================================

/// Example: @route attribute for web framework
/// Usage: @route(GET, "/users/:id")
pub fn routeAttributeMacro(
    allocator: Allocator,
    attr_args: TokenStream,
    item: TokenStream,
) !TokenStream {
    _ = attr_args;
    _ = item;
    
    var result = TokenStream.init(allocator);
    
    // TODO: Parse HTTP method and path
    // TODO: Generate route registration code
    
    return result;
}

/// Example: @cached attribute
/// Usage: @cached(ttl=60)
pub fn cachedAttributeMacro(
    allocator: Allocator,
    attr_args: TokenStream,
    item: TokenStream,
) !TokenStream {
    _ = attr_args;
    _ = item;
    
    var result = TokenStream.init(allocator);
    
    // TODO: Parse cache parameters
    // TODO: Wrap function with caching logic
    
    return result;
}

/// Example: @validate attribute
/// Usage: @validate(min=0, max=100)
pub fn validateAttributeMacro(
    allocator: Allocator,
    attr_args: TokenStream,
    item: TokenStream,
) !TokenStream {
    _ = attr_args;
    _ = item;
    
    var result = TokenStream.init(allocator);
    
    // TODO: Parse validation rules
    // TODO: Generate validation code
    
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "Attribute creation" {
    const attr = Attribute.init(.inline_hint, "inline");
    try std.testing.expectEqual(AttributeKind.inline_hint, attr.kind);
    try std.testing.expectEqualStrings("inline", attr.name);
    try std.testing.expect(!attr.hasArgs());
}

test "Attribute with args" {
    const allocator = std.testing.allocator;
    var args = TokenStream.init(allocator);
    defer args.deinit();
    
    const attr = Attribute.init(.deprecated, "deprecated").withArgs(args);
    try std.testing.expect(attr.hasArgs());
}

test "AttributeList" {
    const allocator = std.testing.allocator;
    var list = AttributeList.init(allocator);
    defer list.deinit();
    
    try list.add(Attribute.init(.inline_hint, "inline"));
    try std.testing.expectEqual(@as(usize, 1), list.len());
    try std.testing.expect(list.has(.inline_hint));
}

test "AttributeParser" {
    const allocator = std.testing.allocator;
    var parser = AttributeParser.init(allocator);
    
    const attr = try parser.parse("@inline");
    try std.testing.expectEqual(AttributeKind.inline_hint, attr.kind);
}

test "AttributeProcessor" {
    const allocator = std.testing.allocator;
    var processor = AttributeProcessor.init(allocator);
    defer processor.deinit();
    
    try processor.registerHandler("inline", handleInline);
    
    const has_handler = processor.handlers.contains("inline");
    try std.testing.expect(has_handler);
}

test "AttributeComposer" {
    const allocator = std.testing.allocator;
    var composer = AttributeComposer.init(allocator);
    
    const attrs = [_]Attribute{
        Attribute.init(.inline_hint, "inline"),
        Attribute.init(.export_attr, "export"),
    };
    
    var list = try composer.compose(&attrs);
    defer list.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), list.len());
}

test "AttributeMacroRegistry" {
    const allocator = std.testing.allocator;
    var registry = AttributeMacroRegistry.init(allocator);
    defer registry.deinit();
    
    const macro_def = AttributeMacro{
        .name = "route",
        .handler = routeAttributeMacro,
    };
    
    try registry.register(macro_def);
    const found = registry.get("route");
    try std.testing.expect(found != null);
}
