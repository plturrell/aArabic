// Mojo Bindgen Tool (Zig -> Mojo)
// Days 101-104: Automated FFI Binding Generation
// Parses Zig exports and generates type-safe Mojo FFI wrappers

const std = @import("std");
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const Allocator = std.mem.Allocator;

// ============================================================================
// Type Definitions
// ============================================================================

/// Zig type categories for mapping
pub const TypeCategory = enum {
    Primitive,
    Pointer,
    Slice,
    Array,
    Optional,
    ErrorUnion,
    Struct,
    Enum,
    Function,
    Void,
    Unknown,
};

/// Parsed type information
pub const TypeInfo = struct {
    raw: []const u8,
    category: TypeCategory,
    inner_type: ?[]const u8,
    is_const: bool,
    array_size: ?usize,

    pub fn init(allocator: Allocator, raw: []const u8) !TypeInfo {
        var info = TypeInfo{
            .raw = try allocator.dupe(u8, raw),
            .category = .Unknown,
            .inner_type = null,
            .is_const = false,
            .array_size = null,
        };

        // Analyze type
        if (std.mem.eql(u8, raw, "void")) {
            info.category = .Void;
        } else if (isPrimitive(raw)) {
            info.category = .Primitive;
        } else if (std.mem.startsWith(u8, raw, "[*]") or std.mem.startsWith(u8, raw, "*")) {
            info.category = .Pointer;
            info.is_const = std.mem.indexOf(u8, raw, "const") != null;
            // Extract inner type
            if (std.mem.startsWith(u8, raw, "[*]const ")) {
                info.inner_type = try allocator.dupe(u8, raw["[*]const ".len..]);
            } else if (std.mem.startsWith(u8, raw, "[*]")) {
                info.inner_type = try allocator.dupe(u8, raw["[*]".len..]);
            } else if (std.mem.startsWith(u8, raw, "*const ")) {
                info.inner_type = try allocator.dupe(u8, raw["*const ".len..]);
            } else if (std.mem.startsWith(u8, raw, "*")) {
                info.inner_type = try allocator.dupe(u8, raw["*".len..]);
            }
        } else if (std.mem.startsWith(u8, raw, "[]")) {
            info.category = .Slice;
            if (std.mem.startsWith(u8, raw, "[]const ")) {
                info.is_const = true;
                info.inner_type = try allocator.dupe(u8, raw["[]const ".len..]);
            } else {
                info.inner_type = try allocator.dupe(u8, raw["[]".len..]);
            }
        } else if (std.mem.startsWith(u8, raw, "?")) {
            info.category = .Optional;
            info.inner_type = try allocator.dupe(u8, raw["?".len..]);
        } else if (std.mem.indexOf(u8, raw, "!") != null) {
            info.category = .ErrorUnion;
        }

        return info;
    }

    pub fn deinit(self: *TypeInfo, allocator: Allocator) void {
        allocator.free(self.raw);
        if (self.inner_type) |inner| {
            allocator.free(inner);
        }
    }

    fn isPrimitive(type_name: []const u8) bool {
        const primitives = [_][]const u8{
            "u8",    "u16",   "u32",    "u64",   "u128",  "usize",
            "i8",    "i16",   "i32",    "i64",   "i128",  "isize",
            "f16",   "f32",   "f64",    "f128",  "bool",  "c_int",
            "c_uint", "c_long", "c_ulong", "c_char",
        };
        for (primitives) |p| {
            if (std.mem.eql(u8, type_name, p)) return true;
        }
        return false;
    }
};

/// Represents a parsed Zig function export
pub const ExportedFunction = struct {
    name: []const u8,
    params: ArrayList(Parameter),
    return_type: TypeInfo,
    calling_conv: CallingConvention,
    doc_comment: ?[]const u8,

    pub const Parameter = struct {
        name: []const u8,
        type_info: TypeInfo,

        pub fn deinit(self: *Parameter, allocator: Allocator) void {
            allocator.free(self.name);
            self.type_info.deinit(allocator);
        }
    };

    pub const CallingConvention = enum {
        C,
        Zig,
        Inline,
    };

    pub fn init(allocator: Allocator, name: []const u8) !ExportedFunction {
        return ExportedFunction{
            .name = try allocator.dupe(u8, name),
            .params = ArrayList(Parameter){},
            .return_type = try TypeInfo.init(allocator, "void"),
            .calling_conv = .C,
            .doc_comment = null,
        };
    }

    pub fn deinit(self: *ExportedFunction, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.params.items) |*param| {
            param.deinit(allocator);
        }
        self.params.deinit(allocator);
        self.return_type.deinit(allocator);
        if (self.doc_comment) |doc| {
            allocator.free(doc);
        }
    }

    pub fn addParam(self: *ExportedFunction, allocator: Allocator, param: Parameter) !void {
        try self.params.append(allocator, param);
    }
};

/// Represents a parsed Zig struct
pub const ExportedStruct = struct {
    name: []const u8,
    fields: ArrayList(Field),
    is_packed: bool,
    is_extern: bool,

    pub const Field = struct {
        name: []const u8,
        type_info: TypeInfo,
        default_value: ?[]const u8,

        pub fn deinit(self: *Field, allocator: Allocator) void {
            allocator.free(self.name);
            self.type_info.deinit(allocator);
            if (self.default_value) |dv| {
                allocator.free(dv);
            }
        }
    };

    pub fn init(allocator: Allocator, name: []const u8) !ExportedStruct {
        return ExportedStruct{
            .name = try allocator.dupe(u8, name),
            .fields = ArrayList(Field){},
            .is_packed = false,
            .is_extern = false,
        };
    }

    pub fn deinit(self: *ExportedStruct, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.fields.items) |*field| {
            field.deinit(allocator);
        }
        self.fields.deinit(allocator);
    }
};

/// Represents a parsed Zig enum
pub const ExportedEnum = struct {
    name: []const u8,
    variants: ArrayList([]const u8),
    tag_type: ?[]const u8,

    pub fn init(allocator: Allocator, name: []const u8) !ExportedEnum {
        return ExportedEnum{
            .name = try allocator.dupe(u8, name),
            .variants = ArrayList([]const u8){},
            .tag_type = null,
        };
    }

    pub fn deinit(self: *ExportedEnum, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.variants.items) |v| {
            allocator.free(v);
        }
        self.variants.deinit(allocator);
        if (self.tag_type) |tt| {
            allocator.free(tt);
        }
    }
};

// ============================================================================
// Zig Parser
// ============================================================================

/// Parser for Zig source files to extract exports
pub const ZigParser = struct {
    allocator: Allocator,
    source: []const u8,
    cursor: usize,
    line: usize,
    column: usize,
    errors: ArrayList(ParseError),

    pub const ParseError = struct {
        message: []const u8,
        line: usize,
        column: usize,
    };

    pub fn init(allocator: Allocator, source: []const u8) ZigParser {
        return ZigParser{
            .allocator = allocator,
            .source = source,
            .cursor = 0,
            .line = 1,
            .column = 1,
            .errors = ArrayList(ParseError){},
        };
    }

    pub fn deinit(self: *ZigParser) void {
        for (self.errors.items) |err| {
            self.allocator.free(err.message);
        }
        self.errors.deinit(self.allocator);
    }

    /// Parse all exported items from the source
    pub fn parseAll(self: *ZigParser) !ParseResult {
        var result = ParseResult{
            .functions = ArrayList(ExportedFunction){},
            .structs = ArrayList(ExportedStruct){},
            .enums = ArrayList(ExportedEnum){},
        };

        while (self.cursor < self.source.len) {
            self.skipWhitespaceAndComments();
            if (self.cursor >= self.source.len) break;

            // Check for export keyword
            if (self.matchKeyword("export")) {
                self.skipWhitespace();
                if (self.matchKeyword("fn")) {
                    const func = try self.parseFunctionDeclaration();
                    try result.functions.append(self.allocator, func);
                }
            } else if (self.matchKeyword("pub")) {
                self.skipWhitespace();
                if (self.matchKeyword("const")) {
                    // Could be a struct or enum definition
                    self.skipWhitespace();
                    const name = self.parseIdentifier();
                    self.skipWhitespace();
                    if (self.source[self.cursor] == '=') {
                        self.cursor += 1;
                        self.skipWhitespace();
                        if (self.matchKeyword("extern") or self.matchKeyword("packed")) {
                            self.skipWhitespace();
                        }
                        if (self.matchKeyword("struct")) {
                            const s = try self.parseStructDefinition(name);
                            try result.structs.append(self.allocator, s);
                        } else if (self.matchKeyword("enum")) {
                            const e = try self.parseEnumDefinition(name);
                            try result.enums.append(self.allocator, e);
                        }
                    }
                }
            } else {
                self.cursor += 1;
            }
        }

        return result;
    }

    pub const ParseResult = struct {
        functions: ArrayList(ExportedFunction),
        structs: ArrayList(ExportedStruct),
        enums: ArrayList(ExportedEnum),

        pub fn deinit(self: *ParseResult, allocator: Allocator) void {
            for (self.functions.items) |*f| f.deinit(allocator);
            self.functions.deinit(allocator);
            for (self.structs.items) |*s| s.deinit(allocator);
            self.structs.deinit(allocator);
            for (self.enums.items) |*e| e.deinit(allocator);
            self.enums.deinit(allocator);
        }
    };

    /// Parse all exported functions (legacy method)
    pub fn parseExports(self: *ZigParser) !ArrayList(ExportedFunction) {
        var exports = ArrayList(ExportedFunction){};

        while (self.cursor < self.source.len) {
            const start = std.mem.indexOfPos(u8, self.source, self.cursor, "export fn");
            if (start) |s| {
                self.cursor = s + "export fn".len;
                const func = try self.parseFunctionDeclaration();
                try exports.append(self.allocator, func);
            } else {
                break;
            }
        }

        return exports;
    }

    fn parseFunctionDeclaration(self: *ZigParser) !ExportedFunction {
        self.skipWhitespace();

        // Parse function name
        const name = self.parseIdentifier();
        var func = try ExportedFunction.init(self.allocator, name);

        // Parse params
        self.skipWhitespace();
        if (self.cursor < self.source.len and self.source[self.cursor] != '(') {
            return error.ExpectedOpenParen;
        }
        self.cursor += 1;

        while (self.cursor < self.source.len and self.source[self.cursor] != ')') {
            self.skipWhitespace();
            if (self.source[self.cursor] == ')') break;

            // Parse param name
            const param_name = self.parseIdentifier();
            self.skipWhitespace();

            if (self.cursor >= self.source.len or self.source[self.cursor] != ':') {
                return error.ExpectedColon;
            }
            self.cursor += 1;
            self.skipWhitespace();

            // Parse param type
            const type_str = self.parseType();
            const type_info = try TypeInfo.init(self.allocator, type_str);

            try func.addParam(self.allocator, .{
                .name = try self.allocator.dupe(u8, param_name),
                .type_info = type_info,
            });

            self.skipWhitespace();
            if (self.cursor < self.source.len and self.source[self.cursor] == ',') {
                self.cursor += 1;
            }
        }

        if (self.cursor < self.source.len and self.source[self.cursor] == ')') {
            self.cursor += 1;
        }

        self.skipWhitespace();

        // Parse return type
        if (self.cursor < self.source.len and self.source[self.cursor] != '{') {
            const ret_type = self.parseType();
            func.return_type.deinit(self.allocator);
            func.return_type = try TypeInfo.init(self.allocator, ret_type);
        }

        // Skip function body
        self.skipUntilChar('}');
        if (self.cursor < self.source.len) self.cursor += 1;

        return func;
    }

    fn parseStructDefinition(self: *ZigParser, name: []const u8) !ExportedStruct {
        var s = try ExportedStruct.init(self.allocator, name);

        self.skipWhitespace();
        if (self.cursor < self.source.len and self.source[self.cursor] == '{') {
            self.cursor += 1;

            while (self.cursor < self.source.len and self.source[self.cursor] != '}') {
                self.skipWhitespaceAndComments();
                if (self.cursor >= self.source.len or self.source[self.cursor] == '}') break;

                // Parse field name
                const field_name = self.parseIdentifier();
                if (field_name.len == 0) {
                    self.skipUntilChar(',');
                    if (self.cursor < self.source.len) self.cursor += 1;
                    continue;
                }

                self.skipWhitespace();
                if (self.cursor >= self.source.len or self.source[self.cursor] != ':') {
                    self.skipUntilChar(',');
                    if (self.cursor < self.source.len) self.cursor += 1;
                    continue;
                }
                self.cursor += 1;
                self.skipWhitespace();

                // Parse field type
                const field_type = self.parseType();
                const type_info = try TypeInfo.init(self.allocator, field_type);

                try s.fields.append(self.allocator, .{
                    .name = try self.allocator.dupe(u8, field_name),
                    .type_info = type_info,
                    .default_value = null,
                });

                self.skipWhitespace();
                if (self.cursor < self.source.len and self.source[self.cursor] == ',') {
                    self.cursor += 1;
                }
            }

            if (self.cursor < self.source.len) self.cursor += 1;
        }

        return s;
    }

    fn parseEnumDefinition(self: *ZigParser, name: []const u8) !ExportedEnum {
        var e = try ExportedEnum.init(self.allocator, name);

        self.skipWhitespace();
        // Check for tag type
        if (self.cursor < self.source.len and self.source[self.cursor] == '(') {
            self.cursor += 1;
            const tag = self.parseType();
            e.tag_type = try self.allocator.dupe(u8, tag);
            self.skipUntilChar(')');
            if (self.cursor < self.source.len) self.cursor += 1;
        }

        self.skipWhitespace();
        if (self.cursor < self.source.len and self.source[self.cursor] == '{') {
            self.cursor += 1;

            while (self.cursor < self.source.len and self.source[self.cursor] != '}') {
                self.skipWhitespaceAndComments();
                if (self.cursor >= self.source.len or self.source[self.cursor] == '}') break;

                const variant = self.parseIdentifier();
                if (variant.len > 0) {
                    try e.variants.append(self.allocator, try self.allocator.dupe(u8, variant));
                }

                self.skipWhitespace();
                if (self.cursor < self.source.len and self.source[self.cursor] == ',') {
                    self.cursor += 1;
                }
            }

            if (self.cursor < self.source.len) self.cursor += 1;
        }

        return e;
    }

    fn parseIdentifier(self: *ZigParser) []const u8 {
        const start = self.cursor;
        while (self.cursor < self.source.len and isValidIdentChar(self.source[self.cursor])) {
            self.cursor += 1;
        }
        return self.source[start..self.cursor];
    }

    fn parseType(self: *ZigParser) []const u8 {
        const start = self.cursor;
        var depth: usize = 0;

        while (self.cursor < self.source.len) {
            const c = self.source[self.cursor];
            // Check for stopping characters at depth 0 first
            if (depth == 0 and (c == ',' or c == ')' or c == '{' or c == ';' or c == '}')) {
                break;
            }
            if (c == '[' or c == '(') {
                depth += 1;
            } else if (c == ']' or c == ')') {
                if (depth > 0) depth -= 1;
            }
            self.cursor += 1;
        }

        return std.mem.trim(u8, self.source[start..self.cursor], " \t\n\r");
    }

    fn matchKeyword(self: *ZigParser, keyword: []const u8) bool {
        if (self.cursor + keyword.len > self.source.len) return false;
        if (std.mem.eql(u8, self.source[self.cursor..][0..keyword.len], keyword)) {
            // Check it's not part of a longer identifier
            if (self.cursor + keyword.len < self.source.len) {
                const next = self.source[self.cursor + keyword.len];
                if (isValidIdentChar(next)) return false;
            }
            self.cursor += keyword.len;
            return true;
        }
        return false;
    }

    fn skipWhitespace(self: *ZigParser) void {
        while (self.cursor < self.source.len and std.ascii.isWhitespace(self.source[self.cursor])) {
            if (self.source[self.cursor] == '\n') {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.cursor += 1;
        }
    }

    fn skipWhitespaceAndComments(self: *ZigParser) void {
        while (self.cursor < self.source.len) {
            if (std.ascii.isWhitespace(self.source[self.cursor])) {
                self.skipWhitespace();
            } else if (self.cursor + 1 < self.source.len and
                self.source[self.cursor] == '/' and self.source[self.cursor + 1] == '/')
            {
                // Line comment
                self.skipUntilChar('\n');
            } else {
                break;
            }
        }
    }

    fn skipUntilChar(self: *ZigParser, char: u8) void {
        while (self.cursor < self.source.len and self.source[self.cursor] != char) {
            self.cursor += 1;
        }
    }

    fn isValidIdentChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '_';
    }
};

// ============================================================================
// Mojo Code Generator
// ============================================================================

/// Generates Mojo FFI code from parsed Zig exports
pub const MojoGenerator = struct {
    allocator: Allocator,
    type_mappings: StringHashMap([]const u8),
    config: Config,

    pub const Config = struct {
        module_name: []const u8 = "ZigLib",
        lib_path: []const u8 = "libzig.so",
        generate_docstrings: bool = true,
        use_typed_pointers: bool = true,
    };

    pub fn init(allocator: Allocator) MojoGenerator {
        var gen = MojoGenerator{
            .allocator = allocator,
            .type_mappings = StringHashMap([]const u8).init(allocator),
            .config = .{},
        };
        gen.initDefaultMappings() catch {};
        return gen;
    }

    pub fn deinit(self: *MojoGenerator) void {
        var it = self.type_mappings.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.type_mappings.deinit();
    }

    fn initDefaultMappings(self: *MojoGenerator) !void {
        const mappings = [_][2][]const u8{
            .{ "u8", "UInt8" },
            .{ "u16", "UInt16" },
            .{ "u32", "UInt32" },
            .{ "u64", "UInt64" },
            .{ "usize", "Int" },
            .{ "i8", "Int8" },
            .{ "i16", "Int16" },
            .{ "i32", "Int32" },
            .{ "i64", "Int64" },
            .{ "isize", "Int" },
            .{ "f32", "Float32" },
            .{ "f64", "Float64" },
            .{ "bool", "Bool" },
            .{ "void", "None" },
            .{ "c_int", "Int32" },
            .{ "c_uint", "UInt32" },
            .{ "c_char", "Int8" },
        };

        for (mappings) |m| {
            const key = try self.allocator.dupe(u8, m[0]);
            const value = try self.allocator.dupe(u8, m[1]);
            try self.type_mappings.put(key, value);
        }
    }

    /// Generate complete Mojo module from parse result
    pub fn generateModule(self: *MojoGenerator, result: ZigParser.ParseResult) ![]const u8 {
        var code = ArrayList(u8){};

        // Header
        try code.appendSlice(self.allocator, "# Auto-generated Mojo FFI bindings\n");
        try code.appendSlice(self.allocator, "# Generated by mojo-bindgen\n\n");
        try code.appendSlice(self.allocator, "from sys.ffi import external_call, DLHandle, OpaquePointer\n");
        try code.appendSlice(self.allocator, "from memory import UnsafePointer\n\n");

        // Generate struct definitions
        for (result.structs.items) |s| {
            try self.generateStruct(&code, s);
        }

        // Generate enum definitions
        for (result.enums.items) |e| {
            try self.generateEnum(&code, e);
        }

        // Generate main class
        try code.appendSlice(self.allocator, "@value\n");
        try code.appendSlice(self.allocator, "struct ");
        try code.appendSlice(self.allocator, self.config.module_name);
        try code.appendSlice(self.allocator, ":\n");
        try code.appendSlice(self.allocator, "    \"\"\"FFI wrapper for Zig library.\"\"\"\n");
        try code.appendSlice(self.allocator, "    var _handle: DLHandle\n\n");

        // Constructor
        try code.appendSlice(self.allocator, "    fn __init__(inout self, path: String = \"");
        try code.appendSlice(self.allocator, self.config.lib_path);
        try code.appendSlice(self.allocator, "\"):\n");
        try code.appendSlice(self.allocator, "        \"\"\"Load the Zig library.\"\"\"\n");
        try code.appendSlice(self.allocator, "        self._handle = DLHandle(path)\n\n");

        // Generate function wrappers
        for (result.functions.items) |func| {
            try self.generateFunction(&code, func);
        }

        return code.toOwnedSlice(self.allocator);
    }

    /// Generate Mojo FFI code (legacy method)
    pub fn generate(self: *MojoGenerator, exports: ArrayList(ExportedFunction)) ![]const u8 {
        var code = ArrayList(u8){};

        try code.appendSlice(self.allocator, "from sys.ffi import external_call, DLHandle, OpaquePointer\n\n");
        try code.appendSlice(self.allocator, "@value\nstruct ");
        try code.appendSlice(self.allocator, self.config.module_name);
        try code.appendSlice(self.allocator, ":\n");
        try code.appendSlice(self.allocator, "    var _handle: DLHandle\n\n");
        try code.appendSlice(self.allocator, "    fn __init__(inout self, path: String):\n");
        try code.appendSlice(self.allocator, "        self._handle = DLHandle(path)\n\n");

        for (exports.items) |func| {
            try self.generateFunction(&code, func);
        }

        return code.toOwnedSlice(self.allocator);
    }

    fn generateStruct(self: *MojoGenerator, code: *ArrayList(u8), s: ExportedStruct) !void {
        try code.appendSlice(self.allocator, "@value\n");
        try code.appendSlice(self.allocator, "struct ");
        try code.appendSlice(self.allocator, s.name);
        try code.appendSlice(self.allocator, ":\n");
        try code.appendSlice(self.allocator, "    \"\"\"FFI struct wrapper.\"\"\"\n");

        for (s.fields.items) |field| {
            try code.appendSlice(self.allocator, "    var ");
            try code.appendSlice(self.allocator, field.name);
            try code.appendSlice(self.allocator, ": ");
            try code.appendSlice(self.allocator, self.mapType(field.type_info));
            try code.appendSlice(self.allocator, "\n");
        }

        try code.appendSlice(self.allocator, "\n");
    }

    fn generateEnum(self: *MojoGenerator, code: *ArrayList(u8), e: ExportedEnum) !void {
        try code.appendSlice(self.allocator, "@value\n");
        try code.appendSlice(self.allocator, "struct ");
        try code.appendSlice(self.allocator, e.name);
        try code.appendSlice(self.allocator, ":\n");
        try code.appendSlice(self.allocator, "    \"\"\"FFI enum wrapper.\"\"\"\n");
        try code.appendSlice(self.allocator, "    var value: Int\n\n");

        var i: usize = 0;
        for (e.variants.items) |variant| {
            try code.appendSlice(self.allocator, "    alias ");
            try code.appendSlice(self.allocator, variant);
            try code.appendSlice(self.allocator, " = ");
            var buf: [20]u8 = undefined;
            const num_str = std.fmt.bufPrint(&buf, "{d}", .{i}) catch "0";
            try code.appendSlice(self.allocator, num_str);
            try code.appendSlice(self.allocator, "\n");
            i += 1;
        }

        try code.appendSlice(self.allocator, "\n");
    }

    fn generateFunction(self: *MojoGenerator, code: *ArrayList(u8), func: ExportedFunction) !void {
        // Function signature
        try code.appendSlice(self.allocator, "    fn ");
        try code.appendSlice(self.allocator, func.name);
        try code.appendSlice(self.allocator, "(self");

        for (func.params.items) |param| {
            try code.appendSlice(self.allocator, ", ");
            try code.appendSlice(self.allocator, param.name);
            try code.appendSlice(self.allocator, ": ");
            try code.appendSlice(self.allocator, self.mapType(param.type_info));
        }

        try code.appendSlice(self.allocator, ") -> ");
        try code.appendSlice(self.allocator, self.mapType(func.return_type));
        try code.appendSlice(self.allocator, ":\n");

        // Docstring
        if (self.config.generate_docstrings) {
            try code.appendSlice(self.allocator, "        \"\"\"Call ");
            try code.appendSlice(self.allocator, func.name);
            try code.appendSlice(self.allocator, " from Zig library.\"\"\"\n");
        }

        // Function body
        try code.appendSlice(self.allocator, "        return external_call[\"");
        try code.appendSlice(self.allocator, func.name);
        try code.appendSlice(self.allocator, "\", ");
        try code.appendSlice(self.allocator, self.mapType(func.return_type));
        try code.appendSlice(self.allocator, "](self._handle");

        for (func.params.items) |param| {
            try code.appendSlice(self.allocator, ", ");
            try code.appendSlice(self.allocator, param.name);
        }

        try code.appendSlice(self.allocator, ")\n\n");
    }

    fn mapType(self: *MojoGenerator, type_info: TypeInfo) []const u8 {
        // Check direct mapping
        if (self.type_mappings.get(type_info.raw)) |mapped| {
            return mapped;
        }

        // Handle pointer types
        if (type_info.category == .Pointer) {
            if (type_info.is_const) {
                if (type_info.inner_type) |inner| {
                    if (std.mem.eql(u8, inner, "u8")) {
                        return "String";
                    }
                }
            }
            return "UnsafePointer[UInt8]";
        }

        // Handle slice types
        if (type_info.category == .Slice) {
            return "List[UInt8]";
        }

        // Handle optional types
        if (type_info.category == .Optional) {
            return "Optional[Any]";
        }

        // Default
        return "Any";
    }
};

// ============================================================================
// Tests
// ============================================================================

test "parse zig export" {
    const allocator = std.testing.allocator;
    const source =
        \\export fn add(a: u32, b: u32) u32 {
        \\    return a + b;
        \\}
    ;

    var parser = ZigParser.init(allocator, source);
    defer parser.deinit();
    var exports = try parser.parseExports();
    defer {
        for (exports.items) |*e| e.deinit(allocator);
        exports.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), exports.items.len);
    try std.testing.expectEqualStrings("add", exports.items[0].name);
}

test "parse multiple exports" {
    const allocator = std.testing.allocator;
    const source =
        \\export fn foo(x: i32) void {
        \\}
        \\
        \\export fn bar(s: [*]const u8) usize {
        \\    return 0;
        \\}
    ;

    var parser = ZigParser.init(allocator, source);
    defer parser.deinit();
    var exports = try parser.parseExports();
    defer {
        for (exports.items) |*e| e.deinit(allocator);
        exports.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 2), exports.items.len);
    try std.testing.expectEqualStrings("foo", exports.items[0].name);
    try std.testing.expectEqualStrings("bar", exports.items[1].name);
}

test "type info parsing" {
    const allocator = std.testing.allocator;

    var info1 = try TypeInfo.init(allocator, "u32");
    defer info1.deinit(allocator);
    try std.testing.expectEqual(TypeCategory.Primitive, info1.category);

    var info2 = try TypeInfo.init(allocator, "[*]const u8");
    defer info2.deinit(allocator);
    try std.testing.expectEqual(TypeCategory.Pointer, info2.category);
    try std.testing.expect(info2.is_const);

    var info3 = try TypeInfo.init(allocator, "?i32");
    defer info3.deinit(allocator);
    try std.testing.expectEqual(TypeCategory.Optional, info3.category);
}

test "mojo generator type mapping" {
    const allocator = std.testing.allocator;

    var gen = MojoGenerator.init(allocator);
    defer gen.deinit();

    var type_u32 = try TypeInfo.init(allocator, "u32");
    defer type_u32.deinit(allocator);
    try std.testing.expectEqualStrings("UInt32", gen.mapType(type_u32));

    var type_ptr = try TypeInfo.init(allocator, "[*]const u8");
    defer type_ptr.deinit(allocator);
    try std.testing.expectEqualStrings("String", gen.mapType(type_ptr));
}

test "generate mojo code" {
    const allocator = std.testing.allocator;
    const source =
        \\export fn greet(name: [*]const u8) void {
        \\}
    ;

    var parser = ZigParser.init(allocator, source);
    defer parser.deinit();
    var exports = try parser.parseExports();
    defer {
        for (exports.items) |*e| e.deinit(allocator);
        exports.deinit(allocator);
    }

    var gen = MojoGenerator.init(allocator);
    defer gen.deinit();

    const code = try gen.generate(exports);
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "fn greet") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "external_call") != null);
}

test "parse struct definition" {
    const allocator = std.testing.allocator;
    const source =
        \\pub const Point = struct {
        \\    x: i32,
        \\    y: i32,
        \\};
    ;

    var parser = ZigParser.init(allocator, source);
    defer parser.deinit();
    var result = try parser.parseAll();
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), result.structs.items.len);
    try std.testing.expectEqualStrings("Point", result.structs.items[0].name);
    try std.testing.expectEqual(@as(usize, 2), result.structs.items[0].fields.items.len);
}

test "parse enum definition" {
    const allocator = std.testing.allocator;
    const source =
        \\pub const Color = enum {
        \\    Red,
        \\    Green,
        \\    Blue,
        \\};
    ;

    var parser = ZigParser.init(allocator, source);
    defer parser.deinit();
    var result = try parser.parseAll();
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), result.enums.items.len);
    try std.testing.expectEqualStrings("Color", result.enums.items[0].name);
    try std.testing.expectEqual(@as(usize, 3), result.enums.items[0].variants.items.len);
}

test "full module generation" {
    const allocator = std.testing.allocator;
    const source =
        \\pub const Vec2 = struct {
        \\    x: f32,
        \\    y: f32,
        \\};
        \\
        \\export fn vec2_add(a: *Vec2, b: *Vec2) Vec2 {
        \\    return Vec2{ .x = a.x + b.x, .y = a.y + b.y };
        \\}
    ;

    var parser = ZigParser.init(allocator, source);
    defer parser.deinit();
    var result = try parser.parseAll();
    defer result.deinit(allocator);

    var gen = MojoGenerator.init(allocator);
    defer gen.deinit();

    const code = try gen.generateModule(result);
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "struct Vec2") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "fn vec2_add") != null);
}
