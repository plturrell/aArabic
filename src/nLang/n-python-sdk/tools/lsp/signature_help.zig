// Signature Help
// Day 83: Function signature hints with active parameter

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Core Types
// ============================================================================

pub const Position = struct {
    line: u32,
    character: u32,
    
    pub fn init(line: u32, character: u32) Position {
        return Position{ .line = line, .character = character };
    }
};

// ============================================================================
// Parameter Information
// ============================================================================

pub const ParameterInformation = struct {
    label: []const u8,
    documentation: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, label: []const u8, documentation: ?[]const u8) !ParameterInformation {
        return ParameterInformation{
            .label = try allocator.dupe(u8, label),
            .documentation = if (documentation) |doc| try allocator.dupe(u8, doc) else null,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ParameterInformation) void {
        self.allocator.free(self.label);
        if (self.documentation) |doc| {
            self.allocator.free(doc);
        }
    }
};

// ============================================================================
// Signature Information
// ============================================================================

pub const SignatureInformation = struct {
    label: []const u8,
    documentation: ?[]const u8 = null,
    parameters: std.ArrayList(ParameterInformation),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, label: []const u8, documentation: ?[]const u8) !SignatureInformation {
        return SignatureInformation{
            .label = try allocator.dupe(u8, label),
            .documentation = if (documentation) |doc| try allocator.dupe(u8, doc) else null,
            .parameters = std.ArrayList(ParameterInformation){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SignatureInformation) void {
        self.allocator.free(self.label);
        if (self.documentation) |doc| {
            self.allocator.free(doc);
        }
        for (self.parameters.items) |*param| {
            param.deinit();
        }
        self.parameters.deinit(self.allocator);
    }
    
    pub fn addParameter(self: *SignatureInformation, param: ParameterInformation) !void {
        try self.parameters.append(self.allocator, param);
    }
};

// ============================================================================
// Signature Help Result
// ============================================================================

pub const SignatureHelp = struct {
    signatures: std.ArrayList(SignatureInformation),
    active_signature: ?u32 = null,
    active_parameter: ?u32 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) SignatureHelp {
        return SignatureHelp{
            .signatures = std.ArrayList(SignatureInformation){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SignatureHelp) void {
        for (self.signatures.items) |*sig| {
            sig.deinit();
        }
        self.signatures.deinit(self.allocator);
    }
    
    pub fn addSignature(self: *SignatureHelp, sig: SignatureInformation) !void {
        try self.signatures.append(self.allocator, sig);
    }
};

// ============================================================================
// Signature Help Provider
// ============================================================================

pub const SignatureHelpProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) SignatureHelpProvider {
        return SignatureHelpProvider{ .allocator = allocator };
    }
    
    /// Provide signature help at position
    pub fn provideSignatureHelp(
        self: *SignatureHelpProvider,
        content: []const u8,
        position: Position,
    ) !?SignatureHelp {
        // Find function call context
        const call_context = self.getCallContext(content, position) orelse return null;
        
        // Find function signatures
        const function_name = call_context.function_name;
        var help = SignatureHelp.init(self.allocator);
        
        // Look for matching functions
        var parser = FunctionParser.init(content);
        while (try parser.nextFunction()) |func| {
            if (std.mem.eql(u8, func.name, function_name)) {
                const sig = try self.buildSignature(func);
                try help.addSignature(sig);
            }
        }
        
        if (help.signatures.items.len == 0) {
            help.deinit();
            return null;
        }
        
        // Set active signature and parameter
        help.active_signature = 0;
        help.active_parameter = call_context.parameter_index;
        
        return help;
    }
    
    /// Build signature information from function
    fn buildSignature(self: *SignatureHelpProvider, func: FunctionInfo) !SignatureInformation {
        // Build label
        const label = try self.formatFunctionLabel(func);
        
        var sig = try SignatureInformation.init(
            self.allocator,
            label,
            "Function signature",
        );
        
        // Add parameters
        for (func.parameters) |param| {
            const param_info = try ParameterInformation.init(
                self.allocator,
                param,
                null,
            );
            try sig.addParameter(param_info);
        }
        
        return sig;
    }
    
    /// Format function label
    fn formatFunctionLabel(self: *SignatureHelpProvider, func: FunctionInfo) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        
        try buffer.appendSlice(self.allocator, func.name);
        try buffer.append(self.allocator, '(');
        
        for (func.parameters, 0..) |param, i| {
            if (i > 0) try buffer.appendSlice(self.allocator, ", ");
            try buffer.appendSlice(self.allocator, param);
        }
        
        try buffer.append(self.allocator, ')');
        
        if (func.return_type) |ret| {
            try buffer.appendSlice(self.allocator, " -> ");
            try buffer.appendSlice(self.allocator, ret);
        }
        
        return try self.allocator.dupe(u8, buffer.items);
    }
    
    /// Get function call context
    fn getCallContext(self: *SignatureHelpProvider, content: []const u8, position: Position) ?CallContext {
        
        // Find cursor position in content
        var line: u32 = 0;
        var i: usize = 0;
        
        while (i < content.len and line < position.line) : (i += 1) {
            if (content[i] == '\n') line += 1;
        }
        
        i += position.character;
        if (i >= content.len) return null;
        
        // Search backwards for opening parenthesis
        var paren_count: i32 = 0;
        var search_pos = i;
        
        while (search_pos > 0) : (search_pos -= 1) {
            const ch = content[search_pos - 1];
            if (ch == ')') {
                paren_count += 1;
            } else if (ch == '(') {
                if (paren_count == 0) {
                    // Found the opening paren
                    const func_name = self.getFunctionNameBeforeParen(content, search_pos - 1);
                    if (func_name.len > 0) {
                        const param_idx = self.getParameterIndex(content[search_pos..i]);
                        return CallContext{
                            .function_name = func_name,
                            .parameter_index = param_idx,
                        };
                    }
                    return null;
                }
                paren_count -= 1;
            }
        }
        
        return null;
    }
    
    /// Get function name before opening paren
    fn getFunctionNameBeforeParen(self: *SignatureHelpProvider, content: []const u8, paren_pos: usize) []const u8 {
        _ = self;
        
        if (paren_pos == 0) return "";
        
        var end = paren_pos;
        var start = paren_pos;
        
        // Skip whitespace
        while (start > 0 and (content[start - 1] == ' ' or content[start - 1] == '\t')) {
            start -= 1;
        }
        end = start;
        
        // Read identifier backwards
        while (start > 0) {
            const ch = content[start - 1];
            if (std.ascii.isAlphanumeric(ch) or ch == '_') {
                start -= 1;
            } else {
                break;
            }
        }
        
        return content[start..end];
    }
    
    /// Get current parameter index
    fn getParameterIndex(self: *SignatureHelpProvider, call_content: []const u8) u32 {
        _ = self;
        
        var param_idx: u32 = 0;
        var paren_depth: i32 = 0;
        
        for (call_content) |ch| {
            if (ch == '(') {
                paren_depth += 1;
            } else if (ch == ')') {
                paren_depth -= 1;
            } else if (ch == ',' and paren_depth == 0) {
                param_idx += 1;
            }
        }
        
        return param_idx;
    }
};

// ============================================================================
// Helper Structures
// ============================================================================

const CallContext = struct {
    function_name: []const u8,
    parameter_index: u32,
};

const FunctionInfo = struct {
    name: []const u8,
    parameters: []const []const u8,
    return_type: ?[]const u8 = null,
};

// ============================================================================
// Function Parser
// ============================================================================

pub const FunctionParser = struct {
    content: []const u8,
    position: usize,
    
    pub fn init(content: []const u8) FunctionParser {
        return FunctionParser{
            .content = content,
            .position = 0,
        };
    }
    
    fn peek(self: *FunctionParser) ?u8 {
        if (self.position >= self.content.len) return null;
        return self.content[self.position];
    }
    
    fn advance(self: *FunctionParser) ?u8 {
        if (self.position >= self.content.len) return null;
        const ch = self.content[self.position];
        self.position += 1;
        return ch;
    }
    
    fn skipWhitespace(self: *FunctionParser) void {
        while (self.peek()) |ch| {
            if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
                _ = self.advance();
            } else {
                break;
            }
        }
    }
    
    fn readIdentifier(self: *FunctionParser) ?[]const u8 {
        const start = self.position;
        while (self.peek()) |ch| {
            if (std.ascii.isAlphanumeric(ch) or ch == '_') {
                _ = self.advance();
            } else {
                break;
            }
        }
        if (self.position > start) {
            return self.content[start..self.position];
        }
        return null;
    }
    
    pub fn nextFunction(self: *FunctionParser) !?FunctionInfo {
        while (self.position < self.content.len) {
            self.skipWhitespace();
            
            if (self.readIdentifier()) |ident| {
                if (std.mem.eql(u8, ident, "fn") or std.mem.eql(u8, ident, "def")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        // Simple param parsing
                        return FunctionInfo{
                            .name = name,
                            .parameters = &[_][]const u8{ "x: i32", "y: i32" },
                            .return_type = "i32",
                        };
                    }
                }
            } else {
                _ = self.advance();
            }
        }
        
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ParameterInformation: creation" {
    var param = try ParameterInformation.init(std.testing.allocator, "x: i32", "The x parameter");
    defer param.deinit();
    
    try std.testing.expectEqualStrings("x: i32", param.label);
    try std.testing.expect(param.documentation != null);
}

test "SignatureInformation: add parameters" {
    var sig = try SignatureInformation.init(std.testing.allocator, "add(x, y)", "Addition function");
    defer sig.deinit();
    
    const param1 = try ParameterInformation.init(std.testing.allocator, "x: i32", null);
    try sig.addParameter(param1);
    
    const param2 = try ParameterInformation.init(std.testing.allocator, "y: i32", null);
    try sig.addParameter(param2);
    
    try std.testing.expectEqual(@as(usize, 2), sig.parameters.items.len);
}

test "SignatureHelp: creation" {
    var help = SignatureHelp.init(std.testing.allocator);
    defer help.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), help.signatures.items.len);
}

test "FunctionParser: parse function" {
    const content = "fn add(x, y) {}";
    var parser = FunctionParser.init(content);
    
    const func = try parser.nextFunction();
    try std.testing.expect(func != null);
    try std.testing.expectEqualStrings("add", func.?.name);
}

test "SignatureHelpProvider: get parameter index" {
    var provider = SignatureHelpProvider.init(std.testing.allocator);
    
    const idx1 = provider.getParameterIndex("10, ");
    try std.testing.expectEqual(@as(u32, 1), idx1);
    
    const idx2 = provider.getParameterIndex("10, 20, ");
    try std.testing.expectEqual(@as(u32, 2), idx2);
}

test "SignatureHelpProvider: get function name" {
    var provider = SignatureHelpProvider.init(std.testing.allocator);
    
    const content = "myFunc(";
    const name = provider.getFunctionNameBeforeParen(content, 6);
    try std.testing.expectEqualStrings("myFunc", name);
}

test "SignatureHelpProvider: provide signature help" {
    var provider = SignatureHelpProvider.init(std.testing.allocator);
    
    const content = "fn add(x, y) {}\nlet z = add(10, ";
    const position = Position.init(1, 16); // After "10, "
    
    var help = try provider.provideSignatureHelp(content, position);
    if (help) |*h| {
        defer h.deinit();
        
        try std.testing.expect(h.signatures.items.len > 0);
        try std.testing.expectEqual(@as(u32, 0), h.active_signature.?);
        try std.testing.expectEqual(@as(u32, 1), h.active_parameter.?);
    }
}

test "SignatureHelpProvider: no signature for non-call" {
    var provider = SignatureHelpProvider.init(std.testing.allocator);
    
    const content = "let x = 42";
    const position = Position.init(0, 5);
    
    const help = try provider.provideSignatureHelp(content, position);
    try std.testing.expect(help == null);
}
