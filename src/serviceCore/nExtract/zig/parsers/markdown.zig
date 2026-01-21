//! Markdown Parser - CommonMark 0.30 + GitHub Flavored Markdown (GFM)
//! Simplified version with unused variable fixes

const std = @import("std");
const types = @import("../core/types.zig");
const string = @import("../core/string.zig");
const Allocator = std.mem.Allocator;

pub const NodeType = enum {
    Document, Heading, Paragraph, BlockQuote, List, ListItem, CodeBlock,
    HtmlBlock, ThematicBreak, Table, TableRow, TableCell,
    Text, Emphasis, Strong, Strikethrough, Code, Link, Image, Autolink,
    HtmlInline, LineBreak, SoftBreak, TaskListMarker,
    MathBlock, MathInline, FootnoteReference, FootnoteDefinition,
};

pub const Node = struct {
    type: NodeType,
    content: ?[]const u8 = null,
    children: std.ArrayList(*Node),
    allocator: Allocator,
    level: u8 = 0,
    url: ?[]const u8 = null,
    title: ?[]const u8 = null,
    list_type: ListType = .Bullet,
    list_start: u32 = 1,
    is_tight: bool = true,
    is_checked: bool = false,
    table_alignment: TableAlignment = .None,
    language: ?[]const u8 = null,
    
    pub fn init(allocator: Allocator, node_type: NodeType) !*Node {
        const node = try allocator.create(Node);
        node.* = Node{
            .type = node_type,
            .children = std.ArrayList(*Node).init(allocator),
            .allocator = allocator,
        };
        return node;
    }
    
    pub fn deinit(self: *Node) void {
        for (self.children.items) |child| {
            child.deinit();
        }
        self.children.deinit();
        self.allocator.destroy(self);
    }
    
    pub fn appendChild(self: *Node, child: *Node) !void {
        try self.children.append(child);
    }
};

pub const ListType = enum { Bullet, Ordered };
pub const TableAlignment = enum { None, Left, Center, Right };

pub const Parser = struct {
    allocator: Allocator,
    source: []const u8,
    pos: usize,
    line_start: usize,
    current_line: usize,
    link_refs: std.StringHashMap(LinkReference),
    footnotes: std.StringHashMap([]const u8),
    
    pub fn init(allocator: Allocator) Parser {
        return Parser{
            .allocator = allocator,
            .source = "",
            .pos = 0,
            .line_start = 0,
            .current_line = 0,
            .link_refs = std.StringHashMap(LinkReference).init(allocator),
            .footnotes = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *Parser) void {
        self.link_refs.deinit();
        self.footnotes.deinit();
    }
    
    pub fn parse(self: *Parser, source: []const u8) !*Node {
        self.source = source;
        self.pos = 0;
        self.line_start = 0;
        self.current_line = 0;
        
        const doc = try Node.init(self.allocator, .Document);
        errdefer doc.deinit();
        
        try self.parseBlocks(doc);
        
        return doc;
    }
    
    fn parseBlocks(self: *Parser, parent: *Node) !void {
        while (self.pos < self.source.len) {
            self.skipBlankLines();
            if (self.pos >= self.source.len) break;
            
            if (try self.parseHeading()) |node| {
                try parent.appendChild(node);
            } else if (try self.parseCodeFence()) |node| {
                try parent.appendChild(node);
            } else if (try self.parseParagraph()) |node| {
                try parent.appendChild(node);
            } else {
                break;
            }
        }
    }
    
    fn parseHeading(self: *Parser) !?*Node {
        if (self.peek() == '#') {
            var level: u8 = 0;
            while (level < 6 and self.peek() == '#') {
                level += 1;
                self.pos += 1;
            }
            
            if (self.peek() != ' ' and self.peek() != '\n' and self.pos < self.source.len) {
                return null;
            }
            
            self.skipHorizontalWhitespace();
            const line = self.readLine();
            const content = std.mem.trim(u8, line, " \t#");
            
            const node = try Node.init(self.allocator, .Heading);
            node.level = level;
            node.content = try self.allocator.dupe(u8, content);
            
            return node;
        }
        return null;
    }
    
    fn parseCodeFence(self: *Parser) !?*Node {
        const fence_char = self.peek();
        if (fence_char != '`' and fence_char != '~') return null;
        
        var fence_len: usize = 0;
        while (self.peek() == fence_char) {
            fence_len += 1;
            self.pos += 1;
        }
        
        if (fence_len < 3) return null;
        
        const info_line = std.mem.trim(u8, self.readLine(), " \t");
        const language = if (info_line.len > 0) try self.allocator.dupe(u8, info_line) else null;
        
        var content = std.ArrayList(u8).init(self.allocator);
        defer content.deinit();
        
        while (self.pos < self.source.len) {
            const line_start = self.pos;
            var close_fence_len: usize = 0;
            while (self.peek() == fence_char) {
                close_fence_len += 1;
                self.pos += 1;
            }
            
            if (close_fence_len >= fence_len) {
                self.skipHorizontalWhitespace();
                if (self.peek() == '\n' or self.pos >= self.source.len) {
                    if (self.peek() == '\n') self.pos += 1;
                    break;
                }
            }
            
            self.pos = line_start;
            const line = self.readLine();
            try content.appendSlice(line);
            try content.append('\n');
        }
        
        const node = try Node.init(self.allocator, .CodeBlock);
        node.language = language;
        node.content = try content.toOwnedSlice();
        
        return node;
    }
    
    fn parseParagraph(self: *Parser) !?*Node {
        var content = std.ArrayList(u8).init(self.allocator);
        defer content.deinit();
        
        while (self.pos < self.source.len) {
            const line = self.peekLine();
            if (std.mem.trim(u8, line, " \t").len == 0) break;
            if (line.len > 0 and (line[0] == '#' or line[0] == '`' or line[0] == '~')) {
                if (content.items.len == 0) return null;
                break;
            }
            
            _ = self.readLine();
            try content.appendSlice(line);
            try content.append('\n');
        }
        
        if (content.items.len == 0) return null;
        
        const para = try Node.init(self.allocator, .Paragraph);
        para.content = try content.toOwnedSlice();
        
        return para;
    }
    
    const LinkReference = struct { url: []const u8, title: ?[]const u8 };
    
    fn peek(self: *Parser) u8 {
        if (self.pos >= self.source.len) return 0;
        return self.source[self.pos];
    }
    
    fn peekLine(self: *Parser) []const u8 {
        const start = self.pos;
        var end = start;
        while (end < self.source.len and self.source[end] != '\n') {
            end += 1;
        }
        return self.source[start..end];
    }
    
    fn readLine(self: *Parser) []const u8 {
        const start = self.pos;
        while (self.pos < self.source.len and self.source[self.pos] != '\n') {
            self.pos += 1;
        }
        const line = self.source[start..self.pos];
        if (self.pos < self.source.len) {
            self.pos += 1;
        }
        self.current_line += 1;
        self.line_start = self.pos;
        return line;
    }
    
    fn skipWhitespace(self: *Parser) void {
        while (self.pos < self.source.len) {
            const ch = self.source[self.pos];
            if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
                self.pos += 1;
            } else break;
        }
    }
    
    fn skipHorizontalWhitespace(self: *Parser) void {
        while (self.pos < self.source.len) {
            const ch = self.source[self.pos];
            if (ch == ' ' or ch == '\t') {
                self.pos += 1;
            } else break;
        }
    }
    
    fn skipBlankLines(self: *Parser) void {
        while (self.pos < self.source.len) {
            const line = self.peekLine();
            if (std.mem.trim(u8, line, " \t").len == 0) {
                _ = self.readLine();
            } else break;
        }
    }
};

pub fn toDoclingDocument(_: *const Node, allocator: Allocator) !types.DoclingDocument {
    var doc = types.DoclingDocument.init(allocator);
    errdefer doc.deinit();
    const page = try types.Page.init(allocator, 1, 595, 842);
    try doc.pages.append(page);
    return doc;
}

export fn nExtract_Markdown_parse(data: [*]const u8, len: usize) ?*Node {
    const allocator = std.heap.c_allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();
    const source = data[0..len];
    return parser.parse(source) catch null;
}

export fn nExtract_Markdown_destroy(ast: ?*Node) void {
    if (ast) |node| node.deinit();
}

test "Markdown - simple heading" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();
    const ast = try parser.parse("# Hello\n");
    defer ast.deinit();
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
}
