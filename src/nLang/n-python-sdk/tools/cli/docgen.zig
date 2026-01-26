// Mojo CLI - Documentation Generator
// Generate HTML/Markdown documentation from source code

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const DocFormat = enum {
    html,
    markdown,
};

pub const DocOptions = struct {
    input_dir: []const u8 = ".",
    output_dir: []const u8 = "docs",
    format: DocFormat = .html,
    include_private: bool = false,
};

pub fn generateDocs(allocator: Allocator, options: DocOptions) !void {
    std.debug.print("Generating documentation...\n", .{});
    std.debug.print("Input:  {s}\n", .{options.input_dir});
    std.debug.print("Output: {s}\n", .{options.output_dir});
    std.debug.print("Format: {s}\n", .{@tagName(options.format)});

    // Create output directory
    try std.fs.cwd().makePath(options.output_dir);

    // Discover source files
    var source_files = try discoverSourceFiles(allocator, options.input_dir);
    defer {
        for (source_files.items) |file| {
            allocator.free(file);
        }
        source_files.deinit(allocator);
    }

    std.debug.print("Found {d} source files\n\n", .{source_files.items.len});

    // Parse and extract documentation
    var doc_items = try std.ArrayList(DocItem).initCapacity(allocator, 10);
    defer {
        for (doc_items.items) |*item| {
            item.deinit(allocator);
        }
        doc_items.deinit(allocator);
    }

    for (source_files.items) |file| {
        try extractDocsFromFile(allocator, file, &doc_items, options);
    }

    // Generate documentation files
    switch (options.format) {
        .html => try generateHTML(allocator, &doc_items, options),
        .markdown => try generateMarkdown(allocator, &doc_items, options),
    }

    std.debug.print("Documentation generated successfully!\n", .{});
    std.debug.print("Open: {s}/index.html\n", .{options.output_dir});
}

const DocItem = struct {
    name: []const u8,
    kind: DocKind,
    description: []const u8,
    signature: ?[]const u8 = null,
    file: []const u8,
    line: usize,
    is_public: bool,

    const DocKind = enum {
        module,
        function,
        struct_,
        trait,
        constant,
        type_alias,
    };

    fn deinit(self: *DocItem, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
        if (self.signature) |sig| {
            allocator.free(sig);
        }
        allocator.free(self.file);
    }
};

fn discoverSourceFiles(allocator: Allocator, dir_path: []const u8) !std.ArrayList([]const u8) {
    var files = try std.ArrayList([]const u8).initCapacity(allocator, 10);

    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".mojo")) continue;

        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, entry.path });
        try files.append(allocator, full_path);
    }

    return files;
}

fn extractDocsFromFile(allocator: Allocator, file_path: []const u8, doc_items: *std.ArrayList(DocItem), options: DocOptions) !void {
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(source);

    // Parse source file
    // In real implementation, would use compiler/frontend/parser.zig
    // to extract function definitions, struct definitions, etc.
    // and their associated doc comments
    
    // Example: Extract function with doc comment
    const item = DocItem{
        .name = try allocator.dupe(u8, "example_function"),
        .kind = .function,
        .description = try allocator.dupe(u8, "This is an example function"),
        .signature = try allocator.dupe(u8, "fn example_function(x: Int) -> Int"),
        .file = try allocator.dupe(u8, file_path),
        .line = 1,
        .is_public = true,
    };

    if (item.is_public or options.include_private) {
        try doc_items.append(allocator, item);
    }
}

fn generateHTML(allocator: Allocator, doc_items: *std.ArrayList(DocItem), options: DocOptions) !void {
    // Generate index.html
    const index_path = try std.fmt.allocPrint(allocator, "{s}/index.html", .{options.output_dir});
    defer allocator.free(index_path);

    var html = try std.ArrayList(u8).initCapacity(allocator, 1024);
    defer html.deinit(allocator);

    const writer = html.writer(allocator);

    // HTML header
    try writer.writeAll("<!DOCTYPE html>\n");
    try writer.writeAll("<html>\n<head>\n");
    try writer.writeAll("  <meta charset=\"UTF-8\">\n");
    try writer.writeAll("  <title>Mojo Documentation</title>\n");
    try writer.writeAll("  <style>\n");
    try writer.writeAll("    body { font-family: sans-serif; margin: 40px; }\n");
    try writer.writeAll("    .item { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }\n");
    try writer.writeAll("    .signature { background: #f5f5f5; padding: 10px; font-family: monospace; }\n");
    try writer.writeAll("  </style>\n");
    try writer.writeAll("</head>\n<body>\n");
    try writer.writeAll("  <h1>Mojo Documentation</h1>\n");

    // Generate items
    for (doc_items.items) |item| {
        try writer.writeAll("  <div class=\"item\">\n");
        try writer.print("    <h2>{s}</h2>\n", .{item.name});
        try writer.print("    <p><strong>Kind:</strong> {s}</p>\n", .{@tagName(item.kind)});
        
        if (item.signature) |sig| {
            try writer.print("    <div class=\"signature\">{s}</div>\n", .{sig});
        }
        
        try writer.print("    <p>{s}</p>\n", .{item.description});
        try writer.print("    <p><small>Defined in {s}:{d}</small></p>\n", .{ item.file, item.line });
        try writer.writeAll("  </div>\n");
    }

    try writer.writeAll("</body>\n</html>\n");

    // Write to file
    try std.fs.cwd().writeFile(.{ .sub_path = index_path, .data = html.items });
}

fn generateMarkdown(allocator: Allocator, doc_items: *std.ArrayList(DocItem), options: DocOptions) !void {
    // Generate README.md
    const index_path = try std.fmt.allocPrint(allocator, "{s}/README.md", .{options.output_dir});
    defer allocator.free(index_path);

    var md = try std.ArrayList(u8).initCapacity(allocator, 1024);
    defer md.deinit(allocator);

    const writer = md.writer(allocator);

    // Markdown header
    try writer.writeAll("# Mojo Documentation\n\n");
    try writer.writeAll("## Table of Contents\n\n");

    // Generate TOC
    for (doc_items.items) |item| {
        try writer.print("- [{s}](#{s})\n", .{ item.name, item.name });
    }

    try writer.writeAll("\n---\n\n");

    // Generate items
    for (doc_items.items) |item| {
        try writer.print("## {s}\n\n", .{item.name});
        try writer.print("**Kind:** {s}\n\n", .{@tagName(item.kind)});
        
        if (item.signature) |sig| {
            try writer.writeAll("```mojo\n");
            try writer.print("{s}\n", .{sig});
            try writer.writeAll("```\n\n");
        }
        
        try writer.print("{s}\n\n", .{item.description});
        try writer.print("*Defined in {s}:{d}*\n\n", .{ item.file, item.line });
        try writer.writeAll("---\n\n");
    }

    // Write to file
    try std.fs.cwd().writeFile(.{ .sub_path = index_path, .data = md.items });
}

// ============================================================================
// Tests
// ============================================================================

test "generate html docs" {
    const allocator = std.testing.allocator;
    
    const options = DocOptions{
        .input_dir = "src",
        .output_dir = "docs",
        .format = .html,
        .include_private = false,
    };
    
    _ = options;
    _ = allocator;
}

test "generate markdown docs" {
    const allocator = std.testing.allocator;
    
    const options = DocOptions{
        .input_dir = "src",
        .output_dir = "docs",
        .format = .markdown,
        .include_private = false,
    };
    
    _ = options;
    _ = allocator;
}

test "include private items" {
    const allocator = std.testing.allocator;
    
    const options = DocOptions{
        .input_dir = "src",
        .output_dir = "docs",
        .format = .html,
        .include_private = true,
    };
    
    _ = options;
    _ = allocator;
}
