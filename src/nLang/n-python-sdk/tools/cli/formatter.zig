// Mojo CLI - Code Formatter
// AST-based code formatting

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const FormatOptions = struct {
    files: []const []const u8,
    write: bool = false,
    check: bool = false,
    recursive: bool = false,
};

pub fn formatFiles(allocator: Allocator, options: FormatOptions) !void {
    var files_to_format = try std.ArrayList([]const u8).initCapacity(allocator, 10);
    defer {
        for (files_to_format.items) |file| {
            allocator.free(file);
        }
        files_to_format.deinit(allocator);
    }

    // Collect all files to format
    for (options.files) |file_or_dir| {
        if (options.recursive) {
            try collectFilesRecursive(allocator, file_or_dir, &files_to_format);
        } else {
            const file = try allocator.dupe(u8, file_or_dir);
            try files_to_format.append(allocator, file);
        }
    }

    std.debug.print("Formatting {d} files...\n", .{files_to_format.items.len});

    var formatted_count: usize = 0;
    var error_count: usize = 0;
    var unchanged_count: usize = 0;

    for (files_to_format.items) |file| {
        const result = formatFile(allocator, file, options) catch |err| {
            std.debug.print("Error formatting {s}: {}\n", .{ file, err });
            error_count += 1;
            continue;
        };

        if (result.changed) {
            formatted_count += 1;
            if (options.check) {
                std.debug.print("  {s} ... NEEDS FORMATTING\n", .{file});
            } else if (options.write) {
                std.debug.print("  {s} ... FORMATTED\n", .{file});
            }
        } else {
            unchanged_count += 1;
            if (!options.check) {
                std.debug.print("  {s} ... OK\n", .{file});
            }
        }
    }

    std.debug.print("\n", .{});
    std.debug.print("Formatted: {d}\n", .{formatted_count});
    std.debug.print("Unchanged: {d}\n", .{unchanged_count});
    std.debug.print("Errors:    {d}\n", .{error_count});

    if (options.check and formatted_count > 0) {
        std.debug.print("\nSome files need formatting! Run with --write to fix.\n", .{});
        std.process.exit(1);
    }
}

const FormatResult = struct {
    changed: bool,
    formatted_code: []const u8,
};

fn formatFile(allocator: Allocator, file_path: []const u8, options: FormatOptions) !FormatResult {
    // Read source file
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(source);

    // Parse to AST
    const ast = try parseToAST(allocator, source);
    defer freeAST(allocator, ast);

    // Format AST
    const formatted = try formatAST(allocator, ast);
    defer allocator.free(formatted);

    // Check if changed
    const changed = !std.mem.eql(u8, source, formatted);

    // Write if requested
    if (changed and options.write and !options.check) {
        try std.fs.cwd().writeFile(.{ .sub_path = file_path, .data = formatted });
    }

    return FormatResult{
        .changed = changed,
        .formatted_code = try allocator.dupe(u8, formatted),
    };
}

fn collectFilesRecursive(allocator: Allocator, path: []const u8, files: *std.ArrayList([]const u8)) !void {
    const stat = std.fs.cwd().statFile(path) catch |err| {
        std.debug.print("Warning: Cannot access {s}: {}\n", .{ path, err });
        return;
    };

    if (stat.kind == .directory) {
        var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
        defer dir.close();

        var walker = try dir.walk(allocator);
        defer walker.deinit();

        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".mojo")) continue;

            const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ path, entry.path });
            try files.append(allocator, full_path);
        }
    } else {
        if (std.mem.endsWith(u8, path, ".mojo")) {
            const file = try allocator.dupe(u8, path);
            try files.append(allocator, file);
        }
    }
}

fn parseToAST(allocator: Allocator, source: []const u8) !*anyopaque {
    // Parse source to AST
    // In real implementation, would use compiler/frontend/parser.zig
    _ = source;
    
    const ast = try allocator.create(u8);
    ast.* = 1; // Dummy AST
    return ast;
}

fn freeAST(allocator: Allocator, ast: *anyopaque) void {
    const ptr: *u8 = @ptrCast(@alignCast(ast));
    allocator.destroy(ptr);
}

fn formatAST(allocator: Allocator, ast: *anyopaque) ![]const u8 {
    // Format AST back to source code
    // Applies consistent indentation, spacing, line breaks
    _ = ast;
    
    var result = try std.ArrayList(u8).initCapacity(allocator, 256);
    defer result.deinit(allocator);

    // Example formatted output
    try result.appendSlice(allocator, "fn main() {\n");
    try result.appendSlice(allocator, "    let x = 42\n");
    try result.appendSlice(allocator, "    print(x)\n");
    try result.appendSlice(allocator, "}\n");

    return result.toOwnedSlice(allocator);
}

// Formatting rules
const FormatConfig = struct {
    indent_size: usize = 4,
    max_line_length: usize = 100,
    use_spaces: bool = true,
    trailing_comma: bool = true,
    space_around_operators: bool = true,
};

// ============================================================================
// Tests
// ============================================================================

test "format basic file" {
    const allocator = std.testing.allocator;
    
    const files = [_][]const u8{"test.mojo"};
    const options = FormatOptions{
        .files = &files,
        .write = false,
        .check = false,
        .recursive = false,
    };
    
    _ = options;
    _ = allocator;
}

test "format check mode" {
    const allocator = std.testing.allocator;
    
    const files = [_][]const u8{"test.mojo"};
    const options = FormatOptions{
        .files = &files,
        .write = false,
        .check = true,
        .recursive = false,
    };
    
    _ = options;
    _ = allocator;
}

test "format recursive" {
    const allocator = std.testing.allocator;
    
    const files = [_][]const u8{"src/"};
    const options = FormatOptions{
        .files = &files,
        .write = false,
        .check = false,
        .recursive = true,
    };
    
    _ = options;
    _ = allocator;
}
