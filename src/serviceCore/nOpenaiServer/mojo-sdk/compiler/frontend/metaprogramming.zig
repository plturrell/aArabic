// Mojo SDK - Metaprogramming
// Day 28: Compile-time evaluation, macros, reflection, attributes

const std = @import("std");

// ============================================================================
// Compile-time Evaluation
// ============================================================================

pub const CompileTimeValue = union(enum) {
    Int: i64,
    Float: f64,
    Bool: bool,
    String: []const u8,
    Type: []const u8,
    
    pub fn asInt(self: *const CompileTimeValue) ?i64 {
        return switch (self.*) {
            .Int => |val| val,
            else => null,
        };
    }
    
    pub fn asString(self: *const CompileTimeValue) ?[]const u8 {
        return switch (self.*) {
            .String => |val| val,
            else => null,
        };
    }
};

pub const CompileTimeEvaluator = struct {
    constants: std.StringHashMap(CompileTimeValue),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) CompileTimeEvaluator {
        return CompileTimeEvaluator{
            .constants = std.StringHashMap(CompileTimeValue).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn defineConstant(self: *CompileTimeEvaluator, name: []const u8, value: CompileTimeValue) !void {
        try self.constants.put(name, value);
    }
    
    pub fn getConstant(self: *const CompileTimeEvaluator, name: []const u8) ?CompileTimeValue {
        return self.constants.get(name);
    }
    
    pub fn evalExpression(self: *CompileTimeEvaluator, expr: []const u8) !CompileTimeValue {
        // Simplified: would parse and evaluate expression
        _ = self;
        _ = expr;
        return CompileTimeValue{ .Int = 42 };
    }
    
    pub fn deinit(self: *CompileTimeEvaluator) void {
        self.constants.deinit();
    }
};

// ============================================================================
// Macros and Code Generation
// ============================================================================

pub const MacroParameter = struct {
    name: []const u8,
    param_type: []const u8,
    
    pub fn init(name: []const u8, param_type: []const u8) MacroParameter {
        return MacroParameter{
            .name = name,
            .param_type = param_type,
        };
    }
};

pub const Macro = struct {
    name: []const u8,
    parameters: std.ArrayList(MacroParameter),
    body: []const u8,
    is_hygenic: bool,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, body: []const u8) Macro {
        return Macro{
            .name = name,
            .parameters = std.ArrayList(MacroParameter){},
            .body = body,
            .is_hygenic = true,
            .allocator = allocator,
        };
    }
    
    pub fn addParameter(self: *Macro, param: MacroParameter) !void {
        try self.parameters.append(self.allocator, param);
    }
    
    pub fn expand(self: *const Macro, args: []const []const u8) ![]const u8 {
        // Simplified: would substitute parameters and expand
        _ = args;
        return self.body;
    }
    
    pub fn deinit(self: *Macro) void {
        self.parameters.deinit(self.allocator);
    }
};

pub const CodeGenerator = struct {
    templates: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) CodeGenerator {
        return CodeGenerator{
            .templates = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addTemplate(self: *CodeGenerator, name: []const u8, template: []const u8) !void {
        try self.templates.put(name, template);
    }
    
    pub fn generate(self: *const CodeGenerator, template_name: []const u8) ![]const u8 {
        if (self.templates.get(template_name)) |template| {
            return template;
        }
        return error.TemplateNotFound;
    }
    
    pub fn deinit(self: *CodeGenerator) void {
        self.templates.deinit();
    }
};

// ============================================================================
// Reflection and Introspection
// ============================================================================

pub const TypeInfo = struct {
    name: []const u8,
    kind: TypeKind,
    size: usize,
    fields: std.ArrayList(FieldInfo),
    methods: std.ArrayList(MethodInfo),
    allocator: std.mem.Allocator,
    
    pub const TypeKind = enum {
        Struct,
        Enum,
        Trait,
        Function,
        Primitive,
    };
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, kind: TypeKind) TypeInfo {
        return TypeInfo{
            .name = name,
            .kind = kind,
            .size = 0,
            .fields = std.ArrayList(FieldInfo){},
            .methods = std.ArrayList(MethodInfo){},
            .allocator = allocator,
        };
    }
    
    pub fn addField(self: *TypeInfo, field: FieldInfo) !void {
        try self.fields.append(self.allocator, field);
    }
    
    pub fn addMethod(self: *TypeInfo, method: MethodInfo) !void {
        try self.methods.append(self.allocator, method);
    }
    
    pub fn deinit(self: *TypeInfo) void {
        self.fields.deinit(self.allocator);
        self.methods.deinit(self.allocator);
    }
};

pub const FieldInfo = struct {
    name: []const u8,
    field_type: []const u8,
    offset: usize,
    
    pub fn init(name: []const u8, field_type: []const u8, offset: usize) FieldInfo {
        return FieldInfo{
            .name = name,
            .field_type = field_type,
            .offset = offset,
        };
    }
};

pub const MethodInfo = struct {
    name: []const u8,
    signature: []const u8,
    
    pub fn init(name: []const u8, signature: []const u8) MethodInfo {
        return MethodInfo{
            .name = name,
            .signature = signature,
        };
    }
};

pub const Reflector = struct {
    type_registry: std.StringHashMap(TypeInfo),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) Reflector {
        return Reflector{
            .type_registry = std.StringHashMap(TypeInfo).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn registerType(self: *Reflector, type_info: TypeInfo) !void {
        try self.type_registry.put(type_info.name, type_info);
    }
    
    pub fn getTypeInfo(self: *const Reflector, type_name: []const u8) ?TypeInfo {
        return self.type_registry.get(type_name);
    }
    
    pub fn hasType(self: *const Reflector, type_name: []const u8) bool {
        return self.type_registry.contains(type_name);
    }
    
    pub fn deinit(self: *Reflector) void {
        var iter = self.type_registry.valueIterator();
        while (iter.next()) |type_info| {
            var mutable_info = type_info.*;
            mutable_info.deinit();
        }
        self.type_registry.deinit();
    }
};

// ============================================================================
// Attribute System
// ============================================================================

pub const Attribute = struct {
    name: []const u8,
    arguments: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Attribute {
        return Attribute{
            .name = name,
            .arguments = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn addArgument(self: *Attribute, arg: []const u8) !void {
        try self.arguments.append(self.allocator, arg);
    }
    
    pub fn hasArgument(self: *const Attribute, arg: []const u8) bool {
        for (self.arguments.items) |a| {
            if (std.mem.eql(u8, a, arg)) return true;
        }
        return false;
    }
    
    pub fn deinit(self: *Attribute) void {
        self.arguments.deinit(self.allocator);
    }
};

pub const AttributeTarget = struct {
    target_name: []const u8,
    attributes: std.ArrayList(Attribute),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, target_name: []const u8) AttributeTarget {
        return AttributeTarget{
            .target_name = target_name,
            .attributes = std.ArrayList(Attribute){},
            .allocator = allocator,
        };
    }
    
    pub fn addAttribute(self: *AttributeTarget, attr: Attribute) !void {
        try self.attributes.append(self.allocator, attr);
    }
    
    pub fn hasAttribute(self: *const AttributeTarget, name: []const u8) bool {
        for (self.attributes.items) |attr| {
            if (std.mem.eql(u8, attr.name, name)) return true;
        }
        return false;
    }
    
    pub fn deinit(self: *AttributeTarget) void {
        for (self.attributes.items) |*attr| {
            attr.deinit();
        }
        self.attributes.deinit(self.allocator);
    }
};

// ============================================================================
// Conditional Compilation
// ============================================================================

pub const Condition = union(enum) {
    Flag: []const u8,
    Platform: []const u8,
    Feature: []const u8,
    Version: VersionCheck,
    
    pub const VersionCheck = struct {
        operator: []const u8,  // ">=", "<=", "==", etc.
        version: []const u8,
    };
};

pub const ConditionalBlock = struct {
    condition: Condition,
    code: []const u8,
    else_code: ?[]const u8,
    
    pub fn init(condition: Condition, code: []const u8) ConditionalBlock {
        return ConditionalBlock{
            .condition = condition,
            .code = code,
            .else_code = null,
        };
    }
    
    pub fn withElse(self: ConditionalBlock, else_code: []const u8) ConditionalBlock {
        return ConditionalBlock{
            .condition = self.condition,
            .code = self.code,
            .else_code = else_code,
        };
    }
    
    pub fn evaluate(self: *const ConditionalBlock, enabled_flags: []const []const u8) []const u8 {
        // Simplified: would evaluate condition against flags
        _ = enabled_flags;
        return switch (self.condition) {
            .Flag => self.code,
            else => self.code,
        };
    }
};

pub const ConditionalCompiler = struct {
    enabled_flags: std.StringHashMap(bool),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ConditionalCompiler {
        return ConditionalCompiler{
            .enabled_flags = std.StringHashMap(bool).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn enableFlag(self: *ConditionalCompiler, flag: []const u8) !void {
        try self.enabled_flags.put(flag, true);
    }
    
    pub fn isEnabled(self: *const ConditionalCompiler, flag: []const u8) bool {
        return self.enabled_flags.get(flag) orelse false;
    }
    
    pub fn shouldCompile(self: *const ConditionalCompiler, condition: Condition) bool {
        return switch (condition) {
            .Flag => |flag| self.isEnabled(flag),
            .Platform => true,  // Simplified
            .Feature => true,   // Simplified
            .Version => true,   // Simplified
        };
    }
    
    pub fn deinit(self: *ConditionalCompiler) void {
        self.enabled_flags.deinit();
    }
};

// ============================================================================
// Template Metaprogramming
// ============================================================================

pub const Template = struct {
    name: []const u8,
    type_params: std.ArrayList([]const u8),
    body: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, body: []const u8) Template {
        return Template{
            .name = name,
            .type_params = std.ArrayList([]const u8){},
            .body = body,
            .allocator = allocator,
        };
    }
    
    pub fn addTypeParam(self: *Template, param: []const u8) !void {
        try self.type_params.append(self.allocator, param);
    }
    
    pub fn instantiate(self: *const Template, args: []const []const u8) ![]const u8 {
        if (args.len != self.type_params.items.len) {
            return error.ArityMismatch;
        }
        // Simplified: would substitute type parameters
        return self.body;
    }
    
    pub fn deinit(self: *Template) void {
        self.type_params.deinit(self.allocator);
    }
};

pub const TemplateEngine = struct {
    templates: std.StringHashMap(Template),
    instantiations: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) TemplateEngine {
        return TemplateEngine{
            .templates = std.StringHashMap(Template).init(allocator),
            .instantiations = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn registerTemplate(self: *TemplateEngine, template: Template) !void {
        try self.templates.put(template.name, template);
    }
    
    pub fn instantiate(self: *TemplateEngine, name: []const u8, args: []const []const u8) ![]const u8 {
        if (self.templates.get(name)) |template| {
            const code = try template.instantiate(args);
            try self.instantiations.append(self.allocator, code);
            return code;
        }
        return error.TemplateNotFound;
    }
    
    pub fn deinit(self: *TemplateEngine) void {
        var iter = self.templates.valueIterator();
        while (iter.next()) |template| {
            var mutable_template = template.*;
            mutable_template.deinit();
        }
        self.templates.deinit();
        self.instantiations.deinit(self.allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "metaprogramming: compile time value" {
    const val = CompileTimeValue{ .Int = 42 };
    try std.testing.expectEqual(@as(i64, 42), val.asInt().?);
}

test "metaprogramming: compile time evaluator" {
    const allocator = std.testing.allocator;
    var evaluator = CompileTimeEvaluator.init(allocator);
    defer evaluator.deinit();
    
    const val = CompileTimeValue{ .Int = 100 };
    try evaluator.defineConstant("MAX_SIZE", val);
    
    const result = evaluator.getConstant("MAX_SIZE");
    try std.testing.expect(result != null);
}

test "metaprogramming: macro definition" {
    const allocator = std.testing.allocator;
    var macro = Macro.init(allocator, "print", "println($1)");
    defer macro.deinit();
    
    const param = MacroParameter.init("msg", "String");
    try macro.addParameter(param);
    
    try std.testing.expectEqual(@as(usize, 1), macro.parameters.items.len);
}

test "metaprogramming: code generator" {
    const allocator = std.testing.allocator;
    var gen = CodeGenerator.init(allocator);
    defer gen.deinit();
    
    try gen.addTemplate("getter", "fn get() -> T { self.field }");
    
    const code = try gen.generate("getter");
    try std.testing.expect(code.len > 0);
}

test "metaprogramming: type info" {
    const allocator = std.testing.allocator;
    var type_info = TypeInfo.init(allocator, "Point", .Struct);
    defer type_info.deinit();
    
    const field = FieldInfo.init("x", "Int", 0);
    try type_info.addField(field);
    
    try std.testing.expectEqual(@as(usize, 1), type_info.fields.items.len);
}

test "metaprogramming: reflector" {
    const allocator = std.testing.allocator;
    var reflector = Reflector.init(allocator);
    defer reflector.deinit();
    
    const type_info = TypeInfo.init(allocator, "String", .Primitive);
    try reflector.registerType(type_info);
    
    try std.testing.expect(reflector.hasType("String"));
}

test "metaprogramming: attribute" {
    const allocator = std.testing.allocator;
    var attr = Attribute.init(allocator, "derive");
    defer attr.deinit();
    
    try attr.addArgument("Clone");
    try attr.addArgument("Debug");
    
    try std.testing.expect(attr.hasArgument("Clone"));
}

test "metaprogramming: attribute target" {
    const allocator = std.testing.allocator;
    var target = AttributeTarget.init(allocator, "MyStruct");
    defer target.deinit();
    
    var attr = Attribute.init(allocator, "repr");
    try attr.addArgument("C");
    
    try target.addAttribute(attr);
    
    try std.testing.expect(target.hasAttribute("repr"));
}

test "metaprogramming: conditional compilation" {
    const allocator = std.testing.allocator;
    var compiler = ConditionalCompiler.init(allocator);
    defer compiler.deinit();
    
    try compiler.enableFlag("debug");
    
    try std.testing.expect(compiler.isEnabled("debug"));
}

test "metaprogramming: template engine" {
    const allocator = std.testing.allocator;
    var engine = TemplateEngine.init(allocator);
    defer engine.deinit();
    
    var template = Template.init(allocator, "Vec", "struct Vec<T> { }");
    try template.addTypeParam("T");
    
    try engine.registerTemplate(template);
    
    const type_args = [_][]const u8{"Int"};
    const code = try engine.instantiate("Vec", &type_args);
    try std.testing.expect(code.len > 0);
}
