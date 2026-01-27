//! ============================================================================
//! Simple YAML Parser for ODPS v4.1 Files
//! Handles basic YAML structure needed for ODPS metadata
//! ============================================================================
//!
//! [CODE:file=yaml_parser.zig]
//! [CODE:module=metadata]
//! [CODE:language=zig]
//!
//! [RELATION:called_by=CODE:odps_mapper.zig]
//! [RELATION:called_by=CODE:odps_quality_service.zig]
//!
//! Note: Infrastructure code - parses ODPS YAML files for metadata extraction.
//! Does not implement ODPS business rules directly.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// YAML parsing errors
pub const YAMLError = error{
    InvalidFormat,
    MissingRequiredField,
    InvalidIndentation,
    ParseError,
};

/// Parsed YAML key-value pair
pub const YAMLNode = struct {
    key: []const u8,
    value: []const u8,
    indent: usize,
    children: std.ArrayList(YAMLNode),

    pub fn init(allocator: Allocator, key: []const u8, value: []const u8, indent: usize) !YAMLNode {
        return .{
            .key = try allocator.dupe(u8, key),
            .value = try allocator.dupe(u8, value),
            .indent = indent,
            .children = std.ArrayList(YAMLNode){},
        };
    }

    pub fn deinit(self: *YAMLNode, allocator: Allocator) void {
        allocator.free(self.key);
        allocator.free(self.value);
        for (self.children.items) |*child| {
            child.deinit(allocator);
        }
        self.children.deinit(allocator);
    }

    pub fn findChild(self: *const YAMLNode, key: []const u8) ?*const YAMLNode {
        for (self.children.items) |*child| {
            if (std.mem.eql(u8, child.key, key)) {
                return child;
            }
        }
        return null;
    }
};

/// Simple YAML parser for ODPS files
pub const YAMLParser = struct {
    allocator: Allocator,
    root: YAMLNode,

    pub fn init(allocator: Allocator) !YAMLParser {
        return .{
            .allocator = allocator,
            .root = try YAMLNode.init(allocator, "root", "", 0),
        };
    }

    pub fn deinit(self: *YAMLParser) void {
        self.root.deinit(self.allocator);
    }

    /// Parse YAML content from string
    pub fn parse(self: *YAMLParser, content: []const u8) !void {
        var lines = std.mem.splitScalar(u8, content, '\n');
        var current_parent = &self.root;
        var parent_stack = std.ArrayList(*YAMLNode){};
        defer parent_stack.deinit(self.allocator);

        while (lines.next()) |line| {
            if (line.len == 0) continue;
            if (std.mem.startsWith(u8, std.mem.trimLeft(u8, line, " \t"), "#")) continue;

            const indent = countIndent(line);
            const trimmed = std.mem.trim(u8, line, " \t");
            
            if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
                const key = std.mem.trim(u8, trimmed[0..colon_pos], " \t\"");
                const value_part = if (colon_pos + 1 < trimmed.len) 
                    std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t\"")
                else 
                    "";

                // Adjust parent based on indent
                while (parent_stack.items.len > 0 and 
                       parent_stack.items[parent_stack.items.len - 1].indent >= indent) {
                    _ = parent_stack.pop();
                }
                
                if (parent_stack.items.len > 0) {
                    current_parent = parent_stack.items[parent_stack.items.len - 1];
                } else {
                    current_parent = &self.root;
                }

                const node = try YAMLNode.init(self.allocator, key, value_part, indent);
                try current_parent.children.append(self.allocator, node);

                // If no value, this node can have children
                if (value_part.len == 0 or std.mem.eql(u8, value_part, "")) {
                    try parent_stack.append(self.allocator, &current_parent.children.items[current_parent.children.items.len - 1]);
                }
            }
        }
    }

    /// Get value by path (e.g., "product.productID")
    pub fn getValue(self: *const YAMLParser, path: []const u8) ?[]const u8 {
        var parts = std.mem.splitScalar(u8, path, '.');
        var current: ?*const YAMLNode = &self.root;

        while (parts.next()) |part| {
            if (current) |node| {
                current = node.findChild(part);
            } else {
                return null;
            }
        }

        if (current) |node| {
            return node.value;
        }
        return null;
    }
};

/// Count leading spaces for indentation
fn countIndent(line: []const u8) usize {
    var count: usize = 0;
    for (line) |char| {
        if (char == ' ') {
            count += 1;
        } else if (char == '\t') {
            count += 2; // Tab counts as 2 spaces
        } else {
            break;
        }
    }
    return count;
}

// Tests
test "YAML parser basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var parser = try YAMLParser.init(allocator);
    defer parser.deinit();

    const yaml_content =
        \\product:
        \\  productID: "urn:uuid:test-v1"
        \\  name: "Test Product"
        \\  details:
        \\    type: "dataset"
        \\    category: "transactional"
    ;

    try parser.parse(yaml_content);

    if (parser.getValue("product.productID")) |value| {
        try testing.expectEqualStrings("urn:uuid:test-v1", value);
    } else {
        try testing.expect(false);
    }
}