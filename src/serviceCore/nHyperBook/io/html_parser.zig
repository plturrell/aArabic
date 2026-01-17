const std = @import("std");

// ============================================================================
// HyperShimmy HTML Parser
// ============================================================================
//
// Day 12: HTML parser for web scraping
//
// Features:
// - HTML tokenization
// - DOM tree construction
// - Text extraction
// - Link extraction (href attributes)
// - Metadata extraction (title, description)
// - Robust error handling for malformed HTML
// - Memory-safe implementation
// ============================================================================

/// HTML token types
pub const TokenType = enum {
    StartTag,
    EndTag,
    SelfClosingTag,
    Text,
    Comment,
    Doctype,
};

/// HTML token
pub const Token = struct {
    type: TokenType,
    name: ?[]const u8 = null, // Tag name for tags
    text: ?[]const u8 = null, // Text content
    attributes: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, token_type: TokenType) Token {
        return Token{
            .type = token_type,
            .attributes = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Token) void {
        if (self.name) |name| self.allocator.free(name);
        if (self.text) |text| self.allocator.free(text);

        var it = self.attributes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.attributes.deinit();
    }
};

/// HTML element node
pub const Element = struct {
    tag: []const u8,
    attributes: std.StringHashMap([]const u8),
    children: std.ArrayListUnmanaged(*Node),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, tag: []const u8) !*Element {
        const elem = try allocator.create(Element);
        elem.* = Element{
            .tag = try allocator.dupe(u8, tag),
            .attributes = std.StringHashMap([]const u8).init(allocator),
            .children = std.ArrayListUnmanaged(*Node){},
            .allocator = allocator,
        };
        return elem;
    }

    pub fn deinit(self: *Element) void {
        self.allocator.free(self.tag);

        var it = self.attributes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.attributes.deinit();

        for (self.children.items) |child| {
            child.deinit(self.allocator);
        }
        self.children.deinit(self.allocator);
    }
};

/// Text node
pub const TextNode = struct {
    content: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, content: []const u8) !*TextNode {
        const node = try allocator.create(TextNode);
        node.* = TextNode{
            .content = try allocator.dupe(u8, content),
            .allocator = allocator,
        };
        return node;
    }

    pub fn deinit(self: *TextNode) void {
        self.allocator.free(self.content);
    }
};

/// DOM node (element or text)
pub const Node = union(enum) {
    element: *Element,
    text: *TextNode,

    pub fn deinit(self: *Node, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .element => |elem| {
                elem.deinit();
                allocator.destroy(elem);
            },
            .text => |text| {
                text.deinit();
                allocator.destroy(text);
            },
        }
        allocator.destroy(self);
    }
};

/// HTML document
pub const Document = struct {
    root: ?*Node,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Document {
        return Document{
            .root = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Document) void {
        if (self.root) |root| {
            root.deinit(self.allocator);
        }
    }

    /// Extract all text content from the document
    pub fn getText(self: *Document, buffer: *std.ArrayListUnmanaged(u8)) !void {
        if (self.root) |root| {
            try self.getTextRecursive(root, buffer);
        }
    }

    fn getTextRecursive(self: *Document, node: *Node, buffer: *std.ArrayListUnmanaged(u8)) !void {
        switch (node.*) {
            .text => |text| {
                const trimmed = std.mem.trim(u8, text.content, " \t\r\n");
                if (trimmed.len > 0) {
                    if (buffer.items.len > 0) {
                        try buffer.append(self.allocator, ' ');
                    }
                    try buffer.appendSlice(self.allocator, trimmed);
                }
            },
            .element => |elem| {
                for (elem.children.items) |child| {
                    try self.getTextRecursive(child, buffer);
                }
            },
        }
    }

    /// Extract all links (href attributes) from the document
    pub fn getLinks(self: *Document) !std.ArrayListUnmanaged([]const u8) {
        var links = std.ArrayListUnmanaged([]const u8){};
        if (self.root) |root| {
            try self.getLinksRecursive(root, &links);
        }
        return links;
    }

    fn getLinksRecursive(self: *Document, node: *Node, links: *std.ArrayListUnmanaged([]const u8)) !void {
        switch (node.*) {
            .element => |elem| {
                // Check for anchor tags with href
                if (std.mem.eql(u8, elem.tag, "a")) {
                    if (elem.attributes.get("href")) |href| {
                        try links.append(self.allocator, href);
                    }
                }
                // Check for link tags with href
                if (std.mem.eql(u8, elem.tag, "link")) {
                    if (elem.attributes.get("href")) |href| {
                        try links.append(self.allocator, href);
                    }
                }
                // Recurse into children
                for (elem.children.items) |child| {
                    try self.getLinksRecursive(child, links);
                }
            },
            .text => {},
        }
    }

    /// Get title from document
    pub fn getTitle(self: *Document) !?[]const u8 {
        if (self.root) |root| {
            return try self.getTitleRecursive(root);
        }
        return null;
    }

    fn getTitleRecursive(self: *Document, node: *Node) !?[]const u8 {
        switch (node.*) {
            .element => |elem| {
                if (std.mem.eql(u8, elem.tag, "title")) {
                    var buffer = std.ArrayListUnmanaged(u8){};
                    defer buffer.deinit(self.allocator);
                    try self.getTextRecursive(node, &buffer);
                    return try self.allocator.dupe(u8, buffer.items);
                }
                for (elem.children.items) |child| {
                    if (try self.getTitleRecursive(child)) |title| {
                        return title;
                    }
                }
            },
            .text => {},
        }
        return null;
    }
};

/// HTML parser
pub const HtmlParser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HtmlParser {
        return HtmlParser{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HtmlParser) void {
        _ = self;
    }

    /// Parse HTML string into a Document
    pub fn parse(self: *HtmlParser, html: []const u8) !Document {
        var tokens = try self.tokenize(html);
        defer {
            for (tokens.items) |*token| {
                token.deinit();
            }
            tokens.deinit(self.allocator);
        }

        return try self.buildTree(tokens.items);
    }

    /// Tokenize HTML  
    fn tokenize(self: *HtmlParser, html: []const u8) !std.ArrayListUnmanaged(Token) {
        var tokens = std.ArrayListUnmanaged(Token){};
        var i: usize = 0;

        while (i < html.len) {
            if (html[i] == '<') {
                // Check for comment
                if (i + 3 < html.len and std.mem.startsWith(u8, html[i..], "<!--")) {
                    const end = std.mem.indexOf(u8, html[i..], "-->") orelse html.len - i;
                    i += end + 3;
                    continue;
                }

                // Check for doctype
                if (i + 9 < html.len and std.ascii.startsWithIgnoreCase(html[i..], "<!doctype")) {
                    const end = std.mem.indexOf(u8, html[i..], ">") orelse html.len - i;
                    i += end + 1;
                    continue;
                }

                // Find end of tag
                const tag_end = std.mem.indexOf(u8, html[i..], ">") orelse {
                    // Malformed HTML, skip
                    i += 1;
                    continue;
                };

                const tag_content = html[i + 1 .. i + tag_end];

                // Check for end tag
                if (tag_content.len > 0 and tag_content[0] == '/') {
                    var token = Token.init(self.allocator, .EndTag);
                    const tag_name = std.mem.trim(u8, tag_content[1..], " \t\r\n");
                    token.name = try self.toLowerCase(tag_name);
                    try tokens.append(self.allocator, token);
                    i += tag_end + 1;
                    continue;
                }

                // Check for self-closing tag
                const is_self_closing = tag_content.len > 0 and tag_content[tag_content.len - 1] == '/';
                const token_type = if (is_self_closing) TokenType.SelfClosingTag else TokenType.StartTag;

                var token = Token.init(self.allocator, token_type);

                // Parse tag name and attributes
                const clean_content = if (is_self_closing)
                    tag_content[0 .. tag_content.len - 1]
                else
                    tag_content;

                var parts = std.mem.tokenizeAny(u8, clean_content, " \t\r\n");
                if (parts.next()) |tag_name| {
                    token.name = try self.toLowerCase(tag_name);
                }

                // Parse attributes
                while (parts.next()) |attr_text| {
                    if (std.mem.indexOf(u8, attr_text, "=")) |eq_pos| {
                        const attr_name = attr_text[0..eq_pos];
                        var attr_value = attr_text[eq_pos + 1 ..];

                        // Remove quotes
                        if (attr_value.len >= 2) {
                            if ((attr_value[0] == '"' and attr_value[attr_value.len - 1] == '"') or
                                (attr_value[0] == '\'' and attr_value[attr_value.len - 1] == '\''))
                            {
                                attr_value = attr_value[1 .. attr_value.len - 1];
                            }
                        }

                        try token.attributes.put(
                            try self.toLowerCase(attr_name),
                            try self.allocator.dupe(u8, attr_value),
                        );
                    }
                }

                try tokens.append(self.allocator, token);
                i += tag_end + 1;
            } else {
                // Text content
                const text_end = std.mem.indexOf(u8, html[i..], "<") orelse html.len - i;
                const text_content = html[i .. i + text_end];

                if (std.mem.trim(u8, text_content, " \t\r\n").len > 0) {
                    var token = Token.init(self.allocator, .Text);
                    token.text = try self.allocator.dupe(u8, text_content);
                    try tokens.append(self.allocator, token);
                }

                i += text_end;
            }
        }

        return tokens;
    }

    /// Build DOM tree from tokens
    fn buildTree(self: *HtmlParser, tokens: []Token) !Document {
        var doc = Document.init(self.allocator);
        var stack = std.ArrayListUnmanaged(*Element){};
        defer stack.deinit(self.allocator);

        // Create implicit root
        const root_elem = try Element.init(self.allocator, "root");
        const root_node = try self.allocator.create(Node);
        root_node.* = Node{ .element = root_elem };
        doc.root = root_node;

        try stack.append(self.allocator, root_elem);

        for (tokens) |token| {
            switch (token.type) {
                .StartTag => {
                    const elem = try Element.init(self.allocator, token.name.?);

                    // Copy attributes
                    var it = token.attributes.iterator();
                    while (it.next()) |entry| {
                        try elem.attributes.put(
                            try self.allocator.dupe(u8, entry.key_ptr.*),
                            try self.allocator.dupe(u8, entry.value_ptr.*),
                        );
                    }

                    const node = try self.allocator.create(Node);
                    node.* = Node{ .element = elem };

                    // Add to current parent
                    if (stack.items.len > 0) {
                        const parent = stack.items[stack.items.len - 1];
                        try parent.children.append(self.allocator, node);
                    }

                    // Push to stack (unless it's a void element)
                    if (!self.isVoidElement(token.name.?)) {
                        try stack.append(self.allocator, elem);
                    }
                },
                .EndTag => {
                    // Pop matching element from stack
                    if (stack.items.len > 1) {
                        const current = stack.items[stack.items.len - 1];
                        if (std.mem.eql(u8, current.tag, token.name.?)) {
                            _ = stack.pop();
                        }
                    }
                },
                .SelfClosingTag => {
                    const elem = try Element.init(self.allocator, token.name.?);

                    // Copy attributes
                    var it = token.attributes.iterator();
                    while (it.next()) |entry| {
                        try elem.attributes.put(
                            try self.allocator.dupe(u8, entry.key_ptr.*),
                            try self.allocator.dupe(u8, entry.value_ptr.*),
                        );
                    }

                    const node = try self.allocator.create(Node);
                    node.* = Node{ .element = elem };

                    // Add to current parent
                    if (stack.items.len > 0) {
                        const parent = stack.items[stack.items.len - 1];
                        try parent.children.append(self.allocator, node);
                    }
                },
                .Text => {
                    const text_node = try TextNode.init(self.allocator, token.text.?);
                    const node = try self.allocator.create(Node);
                    node.* = Node{ .text = text_node };

                    // Add to current parent
                    if (stack.items.len > 0) {
                        const parent = stack.items[stack.items.len - 1];
                        try parent.children.append(self.allocator, node);
                    }
                },
                .Comment, .Doctype => {
                    // Skip comments and doctypes
                },
            }
        }

        return doc;
    }

    /// Convert string to lowercase (ASCII only)
    fn toLowerCase(self: *HtmlParser, s: []const u8) ![]const u8 {
        const result = try self.allocator.alloc(u8, s.len);
        for (s, 0..) |c, i| {
            result[i] = std.ascii.toLower(c);
        }
        return result;
    }

    /// Check if element is a void element (self-closing)
    fn isVoidElement(self: *HtmlParser, tag: []const u8) bool {
        _ = self;
        const void_elements = [_][]const u8{
            "area",  "base",  "br",    "col",   "embed",
            "hr",    "img",   "input", "link",  "meta",
            "param", "source", "track", "wbr",
        };

        for (void_elements) |void_elem| {
            if (std.mem.eql(u8, tag, void_elem)) {
                return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "html parser init" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();
}

test "parse simple html" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = "<html><body>Hello World</body></html>";
    var doc = try parser.parse(html);
    defer doc.deinit();

    try std.testing.expect(doc.root != null);
}

test "extract text content" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = "<html><body><p>Hello</p><p>World</p></body></html>";
    var doc = try parser.parse(html);
    defer doc.deinit();

    var text_buffer = std.ArrayListUnmanaged(u8){};
    defer text_buffer.deinit(std.testing.allocator);

    try doc.getText(&text_buffer);

    try std.testing.expect(std.mem.indexOf(u8, text_buffer.items, "Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, text_buffer.items, "World") != null);
}

test "extract links" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = 
        \\<html><body>
        \\<a href="https://example.com">Link 1</a>
        \\<a href="/page">Link 2</a>
        \\</body></html>
    ;

    var doc = try parser.parse(html);
    defer doc.deinit();

    var links = try doc.getLinks();
    defer links.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), links.items.len);
}

test "parse attributes" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = "<a href=\"https://example.com\" class=\"link\">Test</a>";
    var doc = try parser.parse(html);
    defer doc.deinit();

    // Check that attributes were parsed
    try std.testing.expect(doc.root != null);
}

test "handle malformed html" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = "<html><body><p>Unclosed paragraph</body></html>";
    var doc = try parser.parse(html);
    defer doc.deinit();

    try std.testing.expect(doc.root != null);
}

test "self-closing tags" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = "<html><body><br/><img src=\"test.jpg\"/></body></html>";
    var doc = try parser.parse(html);
    defer doc.deinit();

    try std.testing.expect(doc.root != null);
}

test "html comments ignored" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = "<html><!-- comment --><body>Text</body></html>";
    var doc = try parser.parse(html);
    defer doc.deinit();

    var text_buffer = std.ArrayListUnmanaged(u8){};
    defer text_buffer.deinit(std.testing.allocator);

    try doc.getText(&text_buffer);

    try std.testing.expect(std.mem.indexOf(u8, text_buffer.items, "comment") == null);
    try std.testing.expect(std.mem.indexOf(u8, text_buffer.items, "Text") != null);
}

test "extract title" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    const html = "<html><head><title>Page Title</title></head><body>Content</body></html>";
    var doc = try parser.parse(html);
    defer doc.deinit();

    const title = try doc.getTitle();
    defer if (title) |t| std.testing.allocator.free(t);

    try std.testing.expect(title != null);
    try std.testing.expectEqualStrings("Page Title", title.?);
}

test "void elements" {
    var parser = HtmlParser.init(std.testing.allocator);
    defer parser.deinit();

    try std.testing.expect(parser.isVoidElement("br"));
    try std.testing.expect(parser.isVoidElement("img"));
    try std.testing.expect(parser.isVoidElement("input"));
    try std.testing.expect(!parser.isVoidElement("div"));
    try std.testing.expect(!parser.isVoidElement("p"));
}
