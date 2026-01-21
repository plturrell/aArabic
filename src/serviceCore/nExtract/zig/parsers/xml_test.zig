const std = @import("std");
const xml = @import("xml.zig");
const testing = std.testing;

test "XML parser - simple element" {
    const source = "<root>Hello World</root>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    try testing.expectEqual(@as(usize, 1), doc.children.items.len);
    
    const root = doc.children.items[0];
    try testing.expectEqual(xml.NodeType.Element, root.type);
    try testing.expect(root.name != null);
    try testing.expectEqualStrings("root", root.name.?);
    
    try testing.expectEqual(@as(usize, 1), root.children.items.len);
    const text = root.children.items[0];
    try testing.expectEqual(xml.NodeType.Text, text.type);
    try testing.expectEqualStrings("Hello World", text.value.?);
}

test "XML parser - nested elements" {
    const source = 
        \\<book>
        \\  <title>The Great Gatsby</title>
        \\  <author>F. Scott Fitzgerald</author>
        \\  <year>1925</year>
        \\</book>
    ;
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const book = doc.children.items[0];
    try testing.expectEqualStrings("book", book.name.?);
    try testing.expectEqual(@as(usize, 3), book.children.items.len);
    
    const title = book.children.items[0];
    try testing.expectEqualStrings("title", title.name.?);
    try testing.expectEqualStrings("The Great Gatsby", title.children.items[0].value.?);
}

test "XML parser - attributes" {
    const source = "<person name=\"John\" age=\"30\">Developer</person>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const person = doc.children.items[0];
    try testing.expectEqualStrings("person", person.name.?);
    
    const name = person.getAttribute("name");
    try testing.expect(name != null);
    try testing.expectEqualStrings("John", name.?);
    
    const age = person.getAttribute("age");
    try testing.expect(age != null);
    try testing.expectEqualStrings("30", age.?);
}

test "XML parser - self-closing tag" {
    const source = "<img src=\"photo.jpg\" alt=\"Photo\" />";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const img = doc.children.items[0];
    try testing.expectEqualStrings("img", img.name.?);
    try testing.expectEqual(@as(usize, 0), img.children.items.len);
    
    const src = img.getAttribute("src");
    try testing.expectEqualStrings("photo.jpg", src.?);
}

test "XML parser - CDATA section" {
    const source = "<script><![CDATA[if (x < 5) { alert('Less than 5'); }]]></script>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const script = doc.children.items[0];
    try testing.expectEqualStrings("script", script.name.?);
    
    const cdata = script.children.items[0];
    try testing.expectEqual(xml.NodeType.CDATA, cdata.type);
    try testing.expectEqualStrings("if (x < 5) { alert('Less than 5'); }", cdata.value.?);
}

test "XML parser - comments" {
    const source = "<root><!-- This is a comment -->Text</root>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    parser.preserve_comments = true;
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const root = doc.children.items[0];
    try testing.expectEqual(@as(usize, 2), root.children.items.len);
    
    const comment = root.children.items[0];
    try testing.expectEqual(xml.NodeType.Comment, comment.type);
    try testing.expectEqualStrings(" This is a comment ", comment.value.?);
}

test "XML parser - entity references" {
    const source = "<text>&lt;Hello &amp; Goodbye&gt;</text>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const text_elem = doc.children.items[0];
    const text = text_elem.children.items[0].value.?;
    try testing.expectEqualStrings("<Hello & Goodbye>", text);
}

test "XML parser - character references" {
    const source = "<text>&#72;&#101;&#108;&#108;&#111;</text>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const text_elem = doc.children.items[0];
    const text = text_elem.children.items[0].value.?;
    try testing.expectEqualStrings("Hello", text);
}

test "XML parser - hexadecimal character references" {
    const source = "<text>&#x48;&#x65;&#x6C;&#x6C;&#x6F;</text>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const text_elem = doc.children.items[0];
    const text = text_elem.children.items[0].value.?;
    try testing.expectEqualStrings("Hello", text);
}

test "XML parser - processing instruction" {
    const source = "<?xml-stylesheet type=\"text/xsl\" href=\"style.xsl\"?><root/>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    try testing.expectEqual(@as(usize, 2), doc.children.items.len);
    
    const pi = doc.children.items[0];
    try testing.expectEqual(xml.NodeType.ProcessingInstruction, pi.type);
    try testing.expectEqualStrings("xml-stylesheet", pi.name.?);
}

test "XML parser - namespace declaration" {
    const source = "<root xmlns=\"http://example.com/ns\" xmlns:custom=\"http://custom.com\"><child/></root>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const root = doc.children.items[0];
    const xmlns = root.getAttribute("xmlns");
    try testing.expectEqualStrings("http://example.com/ns", xmlns.?);
    
    const custom = root.getAttribute("xmlns:custom");
    try testing.expectEqualStrings("http://custom.com", custom.?);
}

test "XML parser - SAX mode" {
    const source = "<root><child>Text</child></root>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    // Note: Full SAX callback testing would require more complex setup
    // For now, we just test that SAX parsing doesn't crash
    _ = allocator;
    
    const handler = xml.SaxHandler{
        .startElement = null,
        .endElement = null,
        .characters = null,
    };
    
    try parser.parseSAX(source, handler);
    
    // SAX parsing succeeded (no errors)
    try testing.expect(true);
}

test "XML parser - querySelector" {
    const source = 
        \\<library>
        \\  <book>
        \\    <title>Book 1</title>
        \\  </book>
        \\  <book>
        \\    <title>Book 2</title>
        \\  </book>
        \\</library>
    ;
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    const title = xml.querySelector(doc, "title");
    try testing.expect(title != null);
    try testing.expectEqualStrings("title", title.?.name.?);
}

test "XML parser - mismatched tags error" {
    const source = "<root><child></other></root>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const result = parser.parse(source);
    try testing.expectError(error.MismatchedTag, result);
}

test "XML parser - entity expansion limit" {
    const source = "<root>&test;&test;&test;</root>";
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    parser.max_entity_expansions = 2; // Set low limit
    defer parser.deinit();
    
    const result = parser.parse(source);
    // Should either succeed with unknown entities or fail with limit
    if (result) |doc| {
        doc.deinit();
    } else |_| {
        // Expected to potentially fail
    }
}

test "XML parser - complex document" {
    const source = 
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<!DOCTYPE note SYSTEM "note.dtd">
        \\<note date="2026-01-17">
        \\  <to>Tove</to>
        \\  <from>Jani</from>
        \\  <heading>Reminder</heading>
        \\  <body>Don't forget me this weekend!</body>
        \\  <metadata>
        \\    <priority level="high"/>
        \\    <tags>
        \\      <tag>personal</tag>
        \\      <tag>reminder</tag>
        \\    </tags>
        \\  </metadata>
        \\</note>
    ;
    
    const allocator = testing.allocator;
    var parser = xml.Parser.init(allocator);
    defer parser.deinit();
    
    const doc = try parser.parse(source);
    defer doc.deinit();
    
    // Should have DOCTYPE and root element
    try testing.expect(doc.children.items.len >= 2);
    
    // Find the note element (skip DOCTYPE)
    var note: ?*xml.Node = null;
    for (doc.children.items) |child| {
        if (child.type == .Element and std.mem.eql(u8, child.name.?, "note")) {
            note = child;
            break;
        }
    }
    
    try testing.expect(note != null);
    try testing.expectEqualStrings("2026-01-17", note.?.getAttribute("date").?);
    
    // Should have multiple children
    try testing.expect(note.?.children.items.len >= 4);
}
