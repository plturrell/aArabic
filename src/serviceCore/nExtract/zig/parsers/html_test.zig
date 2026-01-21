//! HTML Parser Test Suite
//!
//! Comprehensive tests for the HTML5 parser implementation

const std = @import("std");
const testing = std.testing;
const html = @import("html.zig");

test "parse simple HTML document" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<!DOCTYPE html>
        \\<html>
        \\<head>
        \\  <title>Test Page</title>
        \\</head>
        \\<body>
        \\  <h1>Hello World</h1>
        \\  <p>This is a test.</p>
        \\</body>
        \\</html>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    try testing.expect(doc.root.children.items.len > 0);
}

test "parse HTML with attributes" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div id="main" class="container">
        \\  <p class="text">Hello</p>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    // Find div by ID
    const div = doc.root.getElementById("main");
    try testing.expect(div != null);
    try testing.expectEqualStrings("div", div.?.tag);
    
    const class = div.?.getAttribute("class");
    try testing.expect(class != null);
    try testing.expectEqualStrings("container", class.?);
}

test "querySelector by ID" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div id="header">Header</div>
        \\<div id="content">Content</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const content_div = doc.querySelector("#content");
    try testing.expect(content_div != null);
    try testing.expectEqualStrings("div", content_div.?.tag);
}

test "querySelector by class" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div class="container">
        \\  <p class="text">Paragraph 1</p>
        \\  <p class="text">Paragraph 2</p>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const container = doc.querySelector(".container");
    try testing.expect(container != null);
    try testing.expectEqualStrings("div", container.?.tag);
}

test "querySelector by tag name" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <p>First paragraph</p>
        \\  <p>Second paragraph</p>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const p = doc.querySelector("p");
    try testing.expect(p != null);
    try testing.expectEqualStrings("p", p.?.tag);
}

test "querySelectorAll" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <p class="text">Para 1</p>
        \\  <p class="text">Para 2</p>
        \\  <p class="text">Para 3</p>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    var results = try doc.querySelectorAll(".text");
    defer results.deinit();
    
    try testing.expectEqual(@as(usize, 3), results.items.len);
}

test "self-closing tags" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <img src="test.png" />
        \\  <br />
        \\  <input type="text" />
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const div = doc.querySelector("div");
    try testing.expect(div != null);
    try testing.expect(div.?.children.items.len >= 3);
}

test "void elements" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <img src="test.png">
        \\  <br>
        \\  <hr>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const div = doc.querySelector("div");
    try testing.expect(div != null);
}

test "comments" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <!-- This is a comment -->
        \\  <p>Text</p>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const div = doc.querySelector("div");
    try testing.expect(div != null);
    try testing.expect(div.?.children.items.len >= 2);
}

test "nested elements" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <ul>
        \\    <li>Item 1</li>
        \\    <li>Item 2</li>
        \\    <li>Item 3</li>
        \\  </ul>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const ul = doc.querySelector("ul");
    try testing.expect(ul != null);
    try testing.expect(ul.?.children.items.len >= 3);
}

test "attribute with double quotes" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div data-value="test value">Content</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const div = doc.querySelector("div");
    try testing.expect(div != null);
    
    const attr = div.?.getAttribute("data-value");
    try testing.expect(attr != null);
    try testing.expectEqualStrings("test value", attr.?);
}

test "attribute with single quotes" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div data-value='test value'>Content</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const div = doc.querySelector("div");
    try testing.expect(div != null);
    
    const attr = div.?.getAttribute("data-value");
    try testing.expect(attr != null);
    try testing.expectEqualStrings("test value", attr.?);
}

test "attribute without quotes" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div data-value=test>Content</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const div = doc.querySelector("div");
    try testing.expect(div != null);
    
    const attr = div.?.getAttribute("data-value");
    try testing.expect(attr != null);
    try testing.expectEqualStrings("test", attr.?);
}

test "multiple attributes" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<input type="text" name="username" id="user" class="form-control" required>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const input = doc.querySelector("input");
    try testing.expect(input != null);
    
    try testing.expectEqualStrings("text", input.?.getAttribute("type").?);
    try testing.expectEqualStrings("username", input.?.getAttribute("name").?);
    try testing.expectEqualStrings("user", input.?.getAttribute("id").?);
    try testing.expectEqualStrings("form-control", input.?.getAttribute("class").?);
}

test "text content" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<p>This is some text content.</p>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const p = doc.querySelector("p");
    try testing.expect(p != null);
    try testing.expect(p.?.children.items.len > 0);
    
    const text_node = p.?.children.items[0];
    try testing.expect(text_node.type == .text);
    try testing.expect(std.mem.indexOf(u8, text_node.text, "text content") != null);
}

test "getElementsByTagName" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <p>Para 1</p>
        \\  <span>Span</span>
        \\  <p>Para 2</p>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    var results = std.ArrayList(*html.Node).init(allocator);
    defer results.deinit();
    
    try doc.root.getElementsByTagName("p", &results);
    try testing.expectEqual(@as(usize, 2), results.items.len);
}

test "getElementsByClassName" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <p class="highlight">Para 1</p>
        \\  <p>Para 2</p>
        \\  <p class="highlight">Para 3</p>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    var results = std.ArrayList(*html.Node).init(allocator);
    defer results.deinit();
    
    try doc.root.getElementsByClassName("highlight", &results);
    try testing.expectEqual(@as(usize, 2), results.items.len);
}

test "doctype parsing" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<!DOCTYPE html>
        \\<html>
        \\<body>Test</body>
        \\</html>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    try testing.expect(doc.doctype != null);
}

test "tag soup recovery - unclosed tags" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div>
        \\  <p>Paragraph 1
        \\  <p>Paragraph 2
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    // Should parse without error despite missing </p> tags
    const div = doc.querySelector("div");
    try testing.expect(div != null);
}

test "empty elements" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div></div>
        \\<p></p>
        \\<span></span>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    try testing.expect(doc.root.children.items.len >= 3);
}

test "whitespace handling" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div   id="test"    class = "container"  >
        \\  <p   >Text</p  >
        \\</div  >
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const div = doc.querySelector("#test");
    try testing.expect(div != null);
    try testing.expectEqualStrings("container", div.?.getAttribute("class").?);
}

test "case insensitivity" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<DIV ID="Test">
        \\  <P CLASS="Text">Content</P>
        \\</DIV>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    // Tags and attributes should be lowercased
    const div = doc.querySelector("div");
    try testing.expect(div != null);
    try testing.expectEqualStrings("div", div.?.tag);
}

test "complex nested structure" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<div class="wrapper">
        \\  <header>
        \\    <nav>
        \\      <ul>
        \\        <li><a href="#home">Home</a></li>
        \\        <li><a href="#about">About</a></li>
        \\      </ul>
        \\    </nav>
        \\  </header>
        \\  <main>
        \\    <article>
        \\      <h1>Title</h1>
        \\      <p>Content</p>
        \\    </article>
        \\  </main>
        \\</div>
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const wrapper = doc.querySelector(".wrapper");
    try testing.expect(wrapper != null);
    
    const nav = doc.querySelector("nav");
    try testing.expect(nav != null);
    
    const h1 = doc.querySelector("h1");
    try testing.expect(h1 != null);
}

test "attribute selector" {
    const allocator = testing.allocator;
    
    const html_content =
        \\<input type="text" name="username">
        \\<input type="password" name="password">
    ;
    
    const doc = try html.HtmlParser.parse(allocator, html_content);
    defer doc.deinit();
    
    const text_input = doc.querySelector("[type=text]");
    try testing.expect(text_input != null);
    try testing.expectEqualStrings("username", text_input.?.getAttribute("name").?);
}
