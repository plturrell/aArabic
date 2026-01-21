// Core data structures for nExtract
// This module defines the fundamental types used throughout the system

const std = @import("std");

// ========== Geometric Types ==========

/// 2D point with floating-point coordinates
pub const Point = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Point {
        return .{ .x = x, .y = y };
    }

    pub fn distance(self: Point, other: Point) f32 {
        const dx = self.x - other.x;
        const dy = self.y - other.y;
        return @sqrt(dx * dx + dy * dy);
    }
};

/// Size with width and height
pub const Size = struct {
    width: f32,
    height: f32,

    pub fn init(width: f32, height: f32) Size {
        return .{ .width = width, .height = height };
    }

    pub fn area(self: Size) f32 {
        return self.width * self.height;
    }
};

/// Bounding box representing a rectangular region
pub const BoundingBox = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,

    pub fn init(x: f32, y: f32, width: f32, height: f32) BoundingBox {
        return .{ .x = x, .y = y, .width = width, .height = height };
    }

    pub fn contains(self: BoundingBox, point: Point) bool {
        return point.x >= self.x and
            point.x <= self.x + self.width and
            point.y >= self.y and
            point.y <= self.y + self.height;
    }

    pub fn intersects(self: BoundingBox, other: BoundingBox) bool {
        return !(self.x + self.width < other.x or
            other.x + other.width < self.x or
            self.y + self.height < other.y or
            other.y + other.height < self.y);
    }

    pub fn area(self: BoundingBox) f32 {
        return self.width * self.height;
    }
};

// ========== Document Element Types ==========

/// Types of document elements
pub const ElementType = enum {
    Text,
    Heading,
    Paragraph,
    Table,
    Image,
    Code,
    Formula,
    List,
    ListItem,
    BlockQuote,
    HorizontalRule,
    PageBreak,

    pub fn toString(self: ElementType) []const u8 {
        return switch (self) {
            .Text => "text",
            .Heading => "heading",
            .Paragraph => "paragraph",
            .Table => "table",
            .Image => "image",
            .Code => "code",
            .Formula => "formula",
            .List => "list",
            .ListItem => "list_item",
            .BlockQuote => "blockquote",
            .HorizontalRule => "horizontal_rule",
            .PageBreak => "page_break",
        };
    }
};

/// Properties for document elements
pub const Properties = struct {
    allocator: std.mem.Allocator,
    font_family: ?[]const u8 = null,
    font_size: ?f32 = null,
    is_bold: bool = false,
    is_italic: bool = false,
    is_underline: bool = false,
    color: ?u32 = null, // RGB color
    heading_level: ?u8 = null, // 1-6 for headings

    pub fn init(allocator: std.mem.Allocator) Properties {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Properties) void {
        if (self.font_family) |font| {
            self.allocator.free(font);
        }
    }
};

/// Document element representing a single unit of content
pub const Element = struct {
    allocator: std.mem.Allocator,
    type: ElementType,
    bbox: BoundingBox,
    content: []const u8,
    properties: Properties,
    page_number: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, element_type: ElementType) !*Element {
        const element = try allocator.create(Element);
        element.* = .{
            .allocator = allocator,
            .type = element_type,
            .bbox = BoundingBox.init(0, 0, 0, 0),
            .content = "",
            .properties = Properties.init(allocator),
        };
        return element;
    }

    pub fn deinit(self: *Element) void {
        if (self.content.len > 0) {
            self.allocator.free(self.content);
        }
        self.properties.deinit();
        self.allocator.destroy(self);
    }

    pub fn setContent(self: *Element, content: []const u8) !void {
        if (self.content.len > 0) {
            self.allocator.free(self.content);
        }
        self.content = try self.allocator.dupe(u8, content);
    }
};

// ========== Document Structure ==========

/// Page metadata
pub const PageMetadata = struct {
    page_number: u32,
    width: f32,
    height: f32,
    rotation: u16 = 0, // 0, 90, 180, 270

    pub fn init(page_number: u32, width: f32, height: f32) PageMetadata {
        return .{
            .page_number = page_number,
            .width = width,
            .height = height,
        };
    }
};

/// Document page
pub const Page = struct {
    allocator: std.mem.Allocator,
    metadata: PageMetadata,
    elements: std.ArrayList(*Element),

    pub fn init(allocator: std.mem.Allocator, metadata: PageMetadata) Page {
        return .{
            .allocator = allocator,
            .metadata = metadata,
            .elements = std.ArrayList(*Element).init(allocator),
        };
    }

    pub fn deinit(self: *Page) void {
        for (self.elements.items) |element| {
            element.deinit();
        }
        self.elements.deinit();
    }

    pub fn addElement(self: *Page, element: *Element) !void {
        try self.elements.append(element);
    }
};

/// Document metadata
pub const Metadata = struct {
    allocator: std.mem.Allocator,
    title: ?[]const u8 = null,
    author: ?[]const u8 = null,
    subject: ?[]const u8 = null,
    keywords: ?[]const u8 = null,
    creator: ?[]const u8 = null,
    producer: ?[]const u8 = null,
    creation_date: ?i64 = null, // Unix timestamp
    modification_date: ?i64 = null,

    pub fn init(allocator: std.mem.Allocator) Metadata {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Metadata) void {
        if (self.title) |title| self.allocator.free(title);
        if (self.author) |author| self.allocator.free(author);
        if (self.subject) |subject| self.allocator.free(subject);
        if (self.keywords) |keywords| self.allocator.free(keywords);
        if (self.creator) |creator| self.allocator.free(creator);
        if (self.producer) |producer| self.allocator.free(producer);
    }
};

/// Main document structure (Docling-compatible)
pub const DoclingDocument = struct {
    allocator: std.mem.Allocator,
    pages: std.ArrayList(Page),
    metadata: Metadata,
    elements: std.ArrayList(*Element), // Flattened list of all elements

    pub fn init(allocator: std.mem.Allocator) DoclingDocument {
        return .{
            .allocator = allocator,
            .pages = std.ArrayList(Page).init(allocator),
            .metadata = Metadata.init(allocator),
            .elements = std.ArrayList(*Element).init(allocator),
        };
    }

    pub fn deinit(self: *DoclingDocument) void {
        for (self.pages.items) |*page| {
            page.deinit();
        }
        self.pages.deinit();
        self.metadata.deinit();
        // Elements are owned by pages, so we just clear the list
        self.elements.deinit();
    }

    pub fn addPage(self: *DoclingDocument, page: Page) !void {
        try self.pages.append(page);
    }

    pub fn pageCount(self: *const DoclingDocument) usize {
        return self.pages.items.len;
    }
};

// ========== FFI Exports ==========

/// Create a new document
export fn nExtract_Document_create() ?*DoclingDocument {
    const allocator = std.heap.c_allocator;
    const doc = allocator.create(DoclingDocument) catch return null;
    doc.* = DoclingDocument.init(allocator);
    return doc;
}

/// Destroy a document
export fn nExtract_Document_destroy(doc: ?*DoclingDocument) void {
    if (doc) |d| {
        d.deinit();
        d.allocator.destroy(d);
    }
}

/// Create a new element
export fn nExtract_Element_create(element_type: ElementType) ?*Element {
    const allocator = std.heap.c_allocator;
    return Element.init(allocator, element_type) catch null;
}

/// Destroy an element
export fn nExtract_Element_destroy(element: ?*Element) void {
    if (element) |e| {
        e.deinit();
    }
}

// ========== Tests ==========

test "Point creation and distance" {
    const p1 = Point.init(0, 0);
    const p2 = Point.init(3, 4);
    try std.testing.expectEqual(@as(f32, 5.0), p1.distance(p2));
}

test "BoundingBox intersection" {
    const box1 = BoundingBox.init(0, 0, 10, 10);
    const box2 = BoundingBox.init(5, 5, 10, 10);
    const box3 = BoundingBox.init(20, 20, 10, 10);

    try std.testing.expect(box1.intersects(box2));
    try std.testing.expect(!box1.intersects(box3));
}

test "Element creation and destruction" {
    const allocator = std.testing.allocator;
    const element = try Element.init(allocator, .Paragraph);
    defer element.deinit();

    try element.setContent("Test content");
    try std.testing.expectEqualStrings("Test content", element.content);
}

test "Document creation and page management" {
    const allocator = std.testing.allocator;
    var doc = DoclingDocument.init(allocator);
    defer doc.deinit();

    const page_meta = PageMetadata.init(1, 595, 842); // A4 size
    const page = Page.init(allocator, page_meta);
    try doc.addPage(page);

    try std.testing.expectEqual(@as(usize, 1), doc.pageCount());
}

test "FFI document create/destroy" {
    const doc = nExtract_Document_create();
    try std.testing.expect(doc != null);
    nExtract_Document_destroy(doc);
}
