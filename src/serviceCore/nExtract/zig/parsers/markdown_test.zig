const std = @import("std");
const markdown = @import("markdown.zig");

test "Markdown parser - simple heading" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "# Hello World\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.Heading, ast.children.items[0].type);
    try std.testing.expectEqual(@as(u8, 1), ast.children.items[0].level);
}

test "Markdown parser - paragraph" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "This is a paragraph.\nWith multiple lines.\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.Paragraph, ast.children.items[0].type);
}

test "Markdown parser - code block" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source =
        \\```python
        \\print("Hello")
        \\```
        \\
    ;
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.CodeBlock, ast.children.items[0].type);
}

test "Markdown parser - table (GFM)" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source =
        \\| Name | Age |
        \\|------|-----|
        \\| Alice| 30  |
        \\| Bob  | 25  |
        \\
    ;
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.Table, ast.children.items[0].type);
}

test "Markdown parser - emphasis and strong" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "This is *italic* and **bold** text.\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    const para = ast.children.items[0];
    try std.testing.expect(para.children.items.len >= 3);
}

test "Markdown parser - links" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "Check out [Example](https://example.com)\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
}

test "Markdown parser - task list (GFM)" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source =
        \\- [x] Completed task
        \\- [ ] Pending task
        \\
    ;
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.List, ast.children.items[0].type);
}

test "Markdown parser - blockquote" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "> This is a quote\n> with multiple lines\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.BlockQuote, ast.children.items[0].type);
}

test "Markdown parser - horizontal rule" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "---\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.ThematicBreak, ast.children.items[0].type);
}

test "Markdown parser - ordered list" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source =
        \\1. First item
        \\2. Second item
        \\3. Third item
        \\
    ;
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
    try std.testing.expectEqual(markdown.NodeType.List, ast.children.items[0].type);
    const list = ast.children.items[0];
    try std.testing.expectEqual(markdown.ListType.Ordered, list.list_type);
}

test "Markdown parser - strikethrough (GFM)" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "This is ~~strikethrough~~ text.\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
}

test "Markdown parser - inline code" {
    const allocator = std.testing.allocator;
    var parser = markdown.Parser.init(allocator);
    defer parser.deinit();
    
    const source = "This has `inline code` in it.\n";
    const ast = try parser.parse(source);
    defer ast.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), ast.children.items.len);
}
