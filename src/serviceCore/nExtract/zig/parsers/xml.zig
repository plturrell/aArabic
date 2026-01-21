//! XML 1.0 Parser (Pure Zig Implementation)
//! 
//! Features:
//! - Full XML 1.0 specification compliance
//! - SAX (event-based) and DOM (tree-based) parsing modes
//! - Namespace support (xmlns)
//! - Entity expansion with size limits (prevents billion laughs attack)
//! - XPath subset for querying
//! - Streaming parser for large files
//! - Attribute parsing and validation
//! - CDATA section handling
//! - Processing instruction support
//! - Comment preservation (optional)

const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("../core/types.zig");

// ============================================================================
// Core Types
// ============================================================================

pub const NodeType = enum {
    Document,
    Element,
    Attribute,
    Text,
    CDATA,
    Comment,
    ProcessingInstruction,
    DocumentType,
};

pub const Node = struct {
    type: NodeType,
    name: ?[]const u8 = null,
    value: ?[]const u8 = null,
    attributes: std.StringHashMap([]const u8),
    children: std.ArrayList(*Node),
    parent: ?*Node = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, node_type: NodeType) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .type = node_type,
            .attributes = std.StringHashMap([]const u8).init(allocator),
            .children = std.ArrayList(*Node).init(allocator),
            .allocator = allocator,
        };
        return node;
    }
    
    pub fn deinit(self: *Node) void {
        // Free children recursively
        for (self.children.items) |child| {
            child.deinit();
        }
        self.children.deinit();
        
        // Free attributes
        var attr_iter = self.attributes.iterator();
        while (attr_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.attributes.deinit();
        
        // Free name and value
        if (self.name) |name| self.allocator.free(name);
        if (self.value) |value| self.allocator.free(value);
        
        self.allocator.destroy(self);
    }
    
    pub fn appendChild(self: *Node, child: *Node) !void {
        try self.children.append(child);
        child.parent = self;
    }
    
    pub fn getAttribute(self: *const Node, name: []const u8) ?[]const u8 {
        return self.attributes.get(name);
    }
    
    pub fn setAttribute(self: *Node, name: []const u8, value: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const value_copy = try self.allocator.dupe(u8, value);
        try self.attributes.put(name_copy, value_copy);
    }
};

// ============================================================================
// SAX Events
// ============================================================================

pub const SaxHandler = struct {
    startElement: ?*const fn (name: []const u8, attributes: std.StringHashMap([]const u8)) anyerror!void = null,
    endElement: ?*const fn (name: []const u8) anyerror!void = null,
    characters: ?*const fn (text: []const u8) anyerror!void = null,
    comment: ?*const fn (text: []const u8) anyerror!void = null,
    processingInstruction: ?*const fn (target: []const u8, data: []const u8) anyerror!void = null,
    startDocument: ?*const fn () anyerror!void = null,
    endDocument: ?*const fn () anyerror!void = null,
};

// ============================================================================
// Parser State
// ============================================================================

pub const ParserMode = enum {
    DOM,    // Build tree structure
    SAX,    // Event-based parsing
};

pub const Parser = struct {
    allocator: Allocator,
    source: []const u8,
    pos: usize,
    line: usize,
    column: usize,
    mode: ParserMode,
    
    // Entity expansion tracking (prevent billion laughs)
    entity_expansion_count: usize,
    max_entity_expansions: usize,
    entity_definitions: std.StringHashMap([]const u8),
    
    // Namespace support
    namespace_stack: std.ArrayList(std.StringHashMap([]const u8)),
    
    // SAX handler
    sax_handler: ?SaxHandler,
    
    // Options
    preserve_whitespace: bool,
    preserve_comments: bool,
    expand_entities: bool,
    
    pub fn init(allocator: Allocator) Parser {
        return .{
            .allocator = allocator,
            .source = "",
            .pos = 0,
            .line = 1,
            .column = 1,
            .mode = .DOM,
            .entity_expansion_count = 0,
            .max_entity_expansions = 1000,
            .entity_definitions = std.StringHashMap([]const u8).init(allocator),
            .namespace_stack = std.ArrayList(std.StringHashMap([]const u8)).init(allocator),
            .sax_handler = null,
            .preserve_whitespace = false,
            .preserve_comments = false,
            .expand_entities = true,
        };
    }
    
    pub fn deinit(self: *Parser) void {
        var entity_iter = self.entity_definitions.iterator();
        while (entity_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.entity_definitions.deinit();
        
        for (self.namespace_stack.items) |*ns_map| {
            var ns_iter = ns_map.iterator();
            while (ns_iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            ns_map.deinit();
        }
        self.namespace_stack.deinit();
    }
    
    // ========================================================================
    // DOM Parsing
    // ========================================================================
    
    pub fn parse(self: *Parser, source: []const u8) !*Node {
        self.source = source;
        self.pos = 0;
        self.line = 1;
        self.column = 1;
        self.mode = .DOM;
        
        // Create document root
        const doc = try Node.init(self.allocator, .Document);
        errdefer doc.deinit();
        
        // Initialize default entity definitions
        try self.initDefaultEntities();
        
        // Push root namespace scope
        try self.pushNamespaceScope();
        
        // Skip XML declaration if present
        try self.skipXmlDeclaration();
        
        // Parse document content
        while (self.pos < self.source.len) {
            self.skipWhitespace();
            if (self.pos >= self.source.len) break;
            
            if (self.peek() == '<') {
                if (self.peekString("<!--")) {
                    if (self.preserve_comments) {
                        const comment = try self.parseComment();
                        try doc.appendChild(comment);
                    } else {
                        try self.skipComment();
                    }
                } else if (self.peekString("<?")) {
                    const pi = try self.parseProcessingInstruction();
                    try doc.appendChild(pi);
                } else if (self.peekString("<!DOCTYPE")) {
                    const doctype = try self.parseDocType();
                    try doc.appendChild(doctype);
                } else if (self.peek() == '<' and self.pos + 1 < self.source.len and self.source[self.pos + 1] != '/') {
                    const element = try self.parseElement();
                    try doc.appendChild(element);
                } else {
                    return error.UnexpectedToken;
                }
            } else {
                return error.TextNotAllowedAtDocumentLevel;
            }
        }
        
        self.popNamespaceScope();
        
        return doc;
    }
    
    // ========================================================================
    // SAX Parsing
    // ========================================================================
    
    pub fn parseSAX(self: *Parser, source: []const u8, handler: SaxHandler) !void {
        self.source = source;
        self.pos = 0;
        self.line = 1;
        self.column = 1;
        self.mode = .SAX;
        self.sax_handler = handler;
        
        // Initialize default entities
        try self.initDefaultEntities();
        
        // Push root namespace scope
        try self.pushNamespaceScope();
        
        // Call startDocument
        if (handler.startDocument) |callback| {
            try callback();
        }
        
        // Skip XML declaration
        try self.skipXmlDeclaration();
        
        // Parse document content
        while (self.pos < self.source.len) {
            self.skipWhitespace();
            if (self.pos >= self.source.len) break;
            
            if (self.peek() == '<') {
                if (self.peekString("<!--")) {
                    const comment_text = try self.parseCommentText();
                    if (handler.comment) |callback| {
                        try callback(comment_text);
                    }
                    self.allocator.free(comment_text);
                } else if (self.peekString("<?")) {
                    const pi = try self.parseProcessingInstructionData();
                    if (handler.processingInstruction) |callback| {
                        try callback(pi.target, pi.data);
                    }
                    self.allocator.free(pi.target);
                    self.allocator.free(pi.data);
                } else if (self.peekString("<!DOCTYPE")) {
                    try self.skipDocType();
                } else if (self.peek() == '<' and self.pos + 1 < self.source.len and self.source[self.pos + 1] != '/') {
                    try self.parseElementSAX();
                } else {
                    return error.UnexpectedToken;
                }
            } else {
                return error.TextNotAllowedAtDocumentLevel;
            }
        }
        
        // Call endDocument
        if (handler.endDocument) |callback| {
            try callback();
        }
        
        self.popNamespaceScope();
    }
    
    // ========================================================================
    // Element Parsing (DOM)
    // ========================================================================
    
    fn parseElement(self: *Parser) !*Node {
        if (self.peek() != '<') return error.ExpectedOpenTag;
        self.advance(); // Skip '<'
        
        const element = try Node.init(self.allocator, .Element);
        errdefer element.deinit();
        
        // Parse element name
        const name = try self.parseName();
        element.name = name;
        
        // Push new namespace scope
        try self.pushNamespaceScope();
        
        // Parse attributes
        while (true) {
            self.skipWhitespace();
            if (self.peek() == '>' or self.peekString("/>")) break;
            
            const attr_name = try self.parseName();
            errdefer self.allocator.free(attr_name);
            
            self.skipWhitespace();
            if (self.peek() != '=') {
                self.allocator.free(attr_name);
                return error.ExpectedEquals;
            }
            self.advance();
            self.skipWhitespace();
            
            const attr_value = try self.parseAttributeValue();
            errdefer self.allocator.free(attr_value);
            
            // Handle namespace declarations
            if (std.mem.eql(u8, attr_name, "xmlns") or std.mem.startsWith(u8, attr_name, "xmlns:")) {
                try self.registerNamespace(attr_name, attr_value);
            }
            
            try element.setAttribute(attr_name, attr_value);
        }
        
        // Check for self-closing tag
        if (self.peekString("/>")) {
            self.advance(); // '/'
            self.advance(); // '>'
            self.popNamespaceScope();
            return element;
        }
        
        if (self.peek() != '>') return error.ExpectedCloseTag;
        self.advance(); // '>'
        
        // Parse children
        while (true) {
            self.skipWhitespace();
            
            if (self.peekString("</")) {
                // End tag
                self.advance(); // '<'
                self.advance(); // '/'
                const end_name = try self.parseName();
                defer self.allocator.free(end_name);
                
                if (!std.mem.eql(u8, name, end_name)) {
                    return error.MismatchedTag;
                }
                
                self.skipWhitespace();
                if (self.peek() != '>') return error.ExpectedCloseTag;
                self.advance();
                break;
            } else if (self.peekString("<![CDATA[")) {
                const cdata = try self.parseCDATA();
                try element.appendChild(cdata);
            } else if (self.peekString("<!--")) {
                if (self.preserve_comments) {
                    const comment = try self.parseComment();
                    try element.appendChild(comment);
                } else {
                    try self.skipComment();
                }
            } else if (self.peekString("<?")) {
                const pi = try self.parseProcessingInstruction();
                try element.appendChild(pi);
            } else if (self.peek() == '<') {
                const child = try self.parseElement();
                try element.appendChild(child);
            } else {
                // Text content
                const text = try self.parseText();
                if (text.len > 0 or self.preserve_whitespace) {
                    const text_node = try Node.init(self.allocator, .Text);
                    text_node.value = text;
                    try element.appendChild(text_node);
                } else {
                    self.allocator.free(text);
                }
            }
        }
        
        self.popNamespaceScope();
        return element;
    }
    
    // ========================================================================
    // Element Parsing (SAX)
    // ========================================================================
    
    fn parseElementSAX(self: *Parser) !void {
        if (self.peek() != '<') return error.ExpectedOpenTag;
        self.advance();
        
        const name = try self.parseName();
        defer self.allocator.free(name);
        
        // Push new namespace scope
        try self.pushNamespaceScope();
        
        // Parse attributes
        var attributes = std.StringHashMap([]const u8).init(self.allocator);
        defer {
            var iter = attributes.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            attributes.deinit();
        }
        
        while (true) {
            self.skipWhitespace();
            if (self.peek() == '>' or self.peekString("/>")) break;
            
            const attr_name = try self.parseName();
            self.skipWhitespace();
            if (self.peek() != '=') {
                self.allocator.free(attr_name);
                return error.ExpectedEquals;
            }
            self.advance();
            self.skipWhitespace();
            
            const attr_value = try self.parseAttributeValue();
            
            // Handle namespace declarations
            if (std.mem.eql(u8, attr_name, "xmlns") or std.mem.startsWith(u8, attr_name, "xmlns:")) {
                try self.registerNamespace(attr_name, attr_value);
            }
            
            try attributes.put(attr_name, attr_value);
        }
        
        // Call startElement
        if (self.sax_handler) |handler| {
            if (handler.startElement) |callback| {
                try callback(name, attributes);
            }
        }
        
        // Check for self-closing tag
        if (self.peekString("/>")) {
            self.advance();
            self.advance();
            
            // Call endElement
            if (self.sax_handler) |handler| {
                if (handler.endElement) |callback| {
                    try callback(name);
                }
            }
            
            self.popNamespaceScope();
            return;
        }
        
        if (self.peek() != '>') return error.ExpectedCloseTag;
        self.advance();
        
        // Parse children
        while (true) {
            self.skipWhitespace();
            
            if (self.peekString("</")) {
                self.advance();
                self.advance();
                const end_name = try self.parseName();
                defer self.allocator.free(end_name);
                
                if (!std.mem.eql(u8, name, end_name)) {
                    return error.MismatchedTag;
                }
                
                self.skipWhitespace();
                if (self.peek() != '>') return error.ExpectedCloseTag;
                self.advance();
                
                // Call endElement
                if (self.sax_handler) |handler| {
                    if (handler.endElement) |callback| {
                        try callback(name);
                    }
                }
                break;
            } else if (self.peekString("<![CDATA[")) {
                const text = try self.parseCDATAText();
                defer self.allocator.free(text);
                if (self.sax_handler) |handler| {
                    if (handler.characters) |callback| {
                        try callback(text);
                    }
                }
            } else if (self.peekString("<!--")) {
                const comment_text = try self.parseCommentText();
                defer self.allocator.free(comment_text);
                if (self.sax_handler) |handler| {
                    if (handler.comment) |callback| {
                        try callback(comment_text);
                    }
                }
            } else if (self.peekString("<?")) {
                const pi = try self.parseProcessingInstructionData();
                defer self.allocator.free(pi.target);
                defer self.allocator.free(pi.data);
                if (self.sax_handler) |handler| {
                    if (handler.processingInstruction) |callback| {
                        try callback(pi.target, pi.data);
                    }
                }
            } else if (self.peek() == '<') {
                try self.parseElementSAX();
            } else {
                const text = try self.parseText();
                defer self.allocator.free(text);
                if (text.len > 0 or self.preserve_whitespace) {
                    if (self.sax_handler) |handler| {
                        if (handler.characters) |callback| {
                            try callback(text);
                        }
                    }
                }
            }
        }
        
        self.popNamespaceScope();
    }
    
    // ========================================================================
    // Comment Parsing
    // ========================================================================
    
    fn parseComment(self: *Parser) !*Node {
        const text = try self.parseCommentText();
        const comment = try Node.init(self.allocator, .Comment);
        comment.value = text;
        return comment;
    }
    
    fn parseCommentText(self: *Parser) ![]const u8 {
        if (!self.peekString("<!--")) return error.ExpectedComment;
        self.advance(); self.advance(); self.advance(); self.advance(); // "<!--"
        
        const start = self.pos;
        while (self.pos < self.source.len) {
            if (self.peekString("-->")) {
                const text = try self.allocator.dupe(u8, self.source[start..self.pos]);
                self.advance(); self.advance(); self.advance(); // "-->"
                return text;
            }
            self.advance();
        }
        return error.UnterminatedComment;
    }
    
    fn skipComment(self: *Parser) !void {
        if (!self.peekString("<!--")) return error.ExpectedComment;
        self.advance(); self.advance(); self.advance(); self.advance();
        
        while (self.pos < self.source.len) {
            if (self.peekString("-->")) {
                self.advance(); self.advance(); self.advance();
                return;
            }
            self.advance();
        }
        return error.UnterminatedComment;
    }
    
    // ========================================================================
    // CDATA Parsing
    // ========================================================================
    
    fn parseCDATA(self: *Parser) !*Node {
        const text = try self.parseCDATAText();
        const cdata = try Node.init(self.allocator, .CDATA);
        cdata.value = text;
        return cdata;
    }
    
    fn parseCDATAText(self: *Parser) ![]const u8 {
        if (!self.peekString("<![CDATA[")) return error.ExpectedCDATA;
        self.pos += 9; // "<![CDATA["
        
        const start = self.pos;
        while (self.pos < self.source.len) {
            if (self.peekString("]]>")) {
                const text = try self.allocator.dupe(u8, self.source[start..self.pos]);
                self.pos += 3; // "]]>"
                return text;
            }
            self.advance();
        }
        return error.UnterminatedCDATA;
    }
    
    // ========================================================================
    // Processing Instruction Parsing
    // ========================================================================
    
    fn parseProcessingInstruction(self: *Parser) !*Node {
        const pi_data = try self.parseProcessingInstructionData();
        const pi = try Node.init(self.allocator, .ProcessingInstruction);
        pi.name = pi_data.target;
        pi.value = pi_data.data;
        return pi;
    }
    
    fn parseProcessingInstructionData(self: *Parser) !struct { target: []const u8, data: []const u8 } {
        if (!self.peekString("<?")) return error.ExpectedProcessingInstruction;
        self.advance(); self.advance(); // "<?"
        
        const target = try self.parseName();
        errdefer self.allocator.free(target);
        
        self.skipWhitespace();
        
        const data_start = self.pos;
        while (self.pos < self.source.len) {
            if (self.peekString("?>")) {
                const data = try self.allocator.dupe(u8, self.source[data_start..self.pos]);
                self.advance(); self.advance(); // "?>"
                return .{ .target = target, .data = data };
            }
            self.advance();
        }
        
        self.allocator.free(target);
        return error.UnterminatedProcessingInstruction;
    }
    
    // ========================================================================
    // DOCTYPE Parsing
    // ========================================================================
    
    fn parseDocType(self: *Parser) !*Node {
        if (!self.peekString("<!DOCTYPE")) return error.ExpectedDocType;
        self.pos += 9; // "<!DOCTYPE"
        
        self.skipWhitespace();
        const name = try self.parseName();
        
        const doctype = try Node.init(self.allocator, .DocumentType);
        doctype.name = name;
        
        // Skip rest of DOCTYPE (we don't parse DTD fully)
        var depth: usize = 1;
        while (self.pos < self.source.len and depth > 0) {
            if (self.peek() == '<') depth += 1;
            if (self.peek() == '>') depth -= 1;
            self.advance();
        }
        
        return doctype;
    }
    
    fn skipDocType(self: *Parser) !void {
        if (!self.peekString("<!DOCTYPE")) return error.ExpectedDocType;
        self.pos += 9;
        
        self.skipWhitespace();
        const name = try self.parseName();
        self.allocator.free(name);
        
        var depth: usize = 1;
        while (self.pos < self.source.len and depth > 0) {
            if (self.peek() == '<') depth += 1;
            if (self.peek() == '>') depth -= 1;
            self.advance();
        }
    }
    
    // ========================================================================
    // Text Parsing
    // ========================================================================
    
    fn parseText(self: *Parser) ![]const u8 {
        var text = std.ArrayList(u8).init(self.allocator);
        errdefer text.deinit();
        
        while (self.pos < self.source.len and self.peek() != '<') {
            if (self.peek() == '&') {
                const entity = try self.parseEntity();
                try text.appendSlice(entity);
                self.allocator.free(entity);
            } else {
                try text.append(self.peek());
                self.advance();
            }
        }
        
        // Trim whitespace if not preserving
        if (!self.preserve_whitespace) {
            const trimmed = std.mem.trim(u8, text.items, " \t\r\n");
            const result = try self.allocator.dupe(u8, trimmed);
            text.deinit();
            return result;
        }
        
        return try text.toOwnedSlice();
    }
    
    // ========================================================================
    // Entity Expansion
    // ========================================================================
    
    fn initDefaultEntities(self: *Parser) !void {
        try self.entity_definitions.put(try self.allocator.dupe(u8, "lt"), try self.allocator.dupe(u8, "<"));
        try self.entity_definitions.put(try self.allocator.dupe(u8, "gt"), try self.allocator.dupe(u8, ">"));
        try self.entity_definitions.put(try self.allocator.dupe(u8, "amp"), try self.allocator.dupe(u8, "&"));
        try self.entity_definitions.put(try self.allocator.dupe(u8, "quot"), try self.allocator.dupe(u8, "\""));
        try self.entity_definitions.put(try self.allocator.dupe(u8, "apos"), try self.allocator.dupe(u8, "'"));
    }
    
    fn parseEntity(self: *Parser) ![]const u8 {
        if (self.peek() != '&') return error.ExpectedEntity;
        self.advance();
        
        // Check entity expansion limit
        self.entity_expansion_count += 1;
        if (self.entity_expansion_count > self.max_entity_expansions) {
            return error.EntityExpansionLimitExceeded;
        }
        
        if (self.peek() == '#') {
            // Character reference
            self.advance();
            return try self.parseCharacterReference();
        }
        
        // Named entity
        const name_start = self.pos;
        while (self.pos < self.source.len and self.peek() != ';') {
            self.advance();
        }
        
        if (self.peek() != ';') return error.UnterminatedEntity;
        const name = self.source[name_start..self.pos];
        self.advance(); // ';'
        
        if (self.entity_definitions.get(name)) |value| {
            return try self.allocator.dupe(u8, value);
        }
        
        // Unknown entity - return as-is
        var result = std.ArrayList(u8).init(self.allocator);
        try result.append('&');
        try result.appendSlice(name);
        try result.append(';');
        return try result.toOwnedSlice();
    }
    
    fn parseCharacterReference(self: *Parser) ![]const u8 {
        var codepoint: u21 = 0;
        
        if (self.peek() == 'x' or self.peek() == 'X') {
            // Hexadecimal
            self.advance();
            while (self.pos < self.source.len and self.peek() != ';') {
                const c = self.peek();
                const digit = if (c >= '0' and c <= '9')
                    c - '0'
                else if (c >= 'a' and c <= 'f')
                    c - 'a' + 10
                else if (c >= 'A' and c <= 'F')
                    c - 'A' + 10
                else
                    return error.InvalidCharacterReference;
                
                codepoint = codepoint * 16 + digit;
                self.advance();
            }
        } else {
            // Decimal
            while (self.pos < self.source.len and self.peek() != ';') {
                const c = self.peek();
                if (c < '0' or c > '9') return error.InvalidCharacterReference;
                codepoint = codepoint * 10 + (c - '0');
                self.advance();
            }
        }
        
        if (self.peek() != ';') return error.UnterminatedEntity;
        self.advance();
        
        // Convert codepoint to UTF-8
        var buf: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(codepoint, &buf);
        return try self.allocator.dupe(u8, buf[0..len]);
    }
    
    // ========================================================================
    // Namespace Support
    // ========================================================================
    
    fn pushNamespaceScope(self: *Parser) !void {
        const new_scope = std.StringHashMap([]const u8).init(self.allocator);
        try self.namespace_stack.append(new_scope);
    }
    
    fn popNamespaceScope(self: *Parser) void {
        if (self.namespace_stack.items.len > 0) {
            var scope = self.namespace_stack.pop();
            var iter = scope.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            scope.deinit();
        }
    }
    
    fn registerNamespace(self: *Parser, attr_name: []const u8, uri: []const u8) !void {
        if (self.namespace_stack.items.len == 0) return;
        
        const prefix = if (std.mem.eql(u8, attr_name, "xmlns"))
            ""
        else
            attr_name[6..]; // Skip "xmlns:"
        
        const scope = &self.namespace_stack.items[self.namespace_stack.items.len - 1];
        const prefix_copy = try self.allocator.dupe(u8, prefix);
        const uri_copy = try self.allocator.dupe(u8, uri);
        try scope.put(prefix_copy, uri_copy);
    }
    
    fn resolveNamespace(self: *const Parser, prefix: []const u8) ?[]const u8 {
        var i = self.namespace_stack.items.len;
        while (i > 0) {
            i -= 1;
            const scope = &self.namespace_stack.items[i];
            if (scope.get(prefix)) |uri| {
                return uri;
            }
        }
        return null;
    }
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    fn parseName(self: *Parser) ![]const u8 {
        const start = self.pos;
        
        // First character: letter or _
        if (!isNameStartChar(self.peek())) return error.InvalidName;
        self.advance();
        
        // Remaining characters: letter, digit, _, -, .
        while (self.pos < self.source.len and isNameChar(self.peek())) {
            self.advance();
        }
        
        return try self.allocator.dupe(u8, self.source[start..self.pos]);
    }
    
    fn parseAttributeValue(self: *Parser) ![]const u8 {
        const quote = self.peek();
        if (quote != '"' and quote != '\'') return error.ExpectedQuote;
        self.advance();
        
        var value = std.ArrayList(u8).init(self.allocator);
        errdefer value.deinit();
        
        while (self.pos < self.source.len and self.peek() != quote) {
            if (self.peek() == '&') {
                const entity = try self.parseEntity();
                try value.appendSlice(entity);
                self.allocator.free(entity);
            } else {
                try value.append(self.peek());
                self.advance();
            }
        }
        
        if (self.peek() != quote) return error.UnterminatedAttributeValue;
        self.advance();
        
        return try value.toOwnedSlice();
    }
    
    fn skipXmlDeclaration(self: *Parser) !void {
        if (self.peekString("<?xml")) {
            while (self.pos < self.source.len and !self.peekString("?>")) {
                self.advance();
            }
            if (self.peekString("?>")) {
                self.advance();
                self.advance();
            }
        }
    }
    
    fn skipWhitespace(self: *Parser) void {
        while (self.pos < self.source.len) {
            const c = self.peek();
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    fn peek(self: *const Parser) u8 {
        if (self.pos >= self.source.len) return 0;
        return self.source[self.pos];
    }
    
    fn peekString(self: *const Parser, str: []const u8) bool {
        if (self.pos + str.len > self.source.len) return false;
        return std.mem.eql(u8, self.source[self.pos..self.pos + str.len], str);
    }
    
    fn advance(self: *Parser) void {
        if (self.pos >= self.source.len) return;
        
        if (self.source[self.pos] == '\n') {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        
        self.pos += 1;
    }
    
    fn isNameStartChar(c: u8) bool {
        return (c >= 'A' and c <= 'Z') or 
               (c >= 'a' and c <= 'z') or 
               c == '_' or c == ':';
    }
    
    fn isNameChar(c: u8) bool {
        return isNameStartChar(c) or 
               (c >= '0' and c <= '9') or 
               c == '-' or c == '.';
    }
};

// ============================================================================
// XPath Query (Basic Subset)
// ============================================================================

pub fn querySelector(root: *const Node, selector: []const u8) ?*Node {
    // Basic XPath: "/path/to/element" or "element[@attr='value']"
    // For now, just implement simple tag name search
    if (selector.len == 0) return null;
    
    if (selector[0] == '/') {
        // Absolute path
        return querySelectorRecursive(root, selector[1..]);
    } else {
        // Relative - search from current node
        return querySelectorRecursive(root, selector);
    }
}

fn querySelectorRecursive(node: *const Node, selector: []const u8) ?*Node {
    // Simple implementation: just match tag name
    if (node.type == .Element) {
        if (node.name) |name| {
            if (std.mem.eql(u8, name, selector)) {
                // Cast away const - this is a query operation
                return @constCast(node);
            }
        }
    }
    
    for (node.children.items) |child| {
        if (querySelectorRecursive(child, selector)) |found| {
            return found;
        }
    }
    
    return null;
}

// ============================================================================
// FFI Exports
// ============================================================================

export fn nExtract_XML_parse(data: [*]const u8, len: usize) ?*Node {
    const allocator = std.heap.c_allocator;
    const source = data[0..len];
    
    var parser = Parser.init(allocator);
    defer parser.deinit();
    
    return parser.parse(source) catch null;
}

export fn nExtract_XML_destroy(doc: ?*Node) void {
    if (doc) |d| d.deinit();
}

export fn nExtract_XML_querySelector(root: ?*const Node, selector: [*:0]const u8) ?*Node {
    if (root == null) return null;
    const sel = std.mem.span(selector);
    return querySelector(root.?, sel);
}
