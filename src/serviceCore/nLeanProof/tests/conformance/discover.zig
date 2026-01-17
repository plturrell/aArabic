const std = @import("std");

const json = std.json;
const mem = std.mem;

const Options = struct {
    root: []const u8,
    suite: ?[]const u8,
    limit: usize,
    json_output: bool,
    absolute: bool,
};

const Entry = struct {
    path: []const u8,
    suite: []const u8,
    has_expected: bool,
};

fn printUsage() void {
    const out = std.fs.File.stderr().deprecatedWriter();
    _ = out.write(
        "Usage: lean4-discover [--root PATH] [--suite NAME] [--limit N] [--json] [--absolute]\n" ++
        "Defaults:\n" ++
        "  --root  vendor/layerIntelligence/lean4/tests\n" ++
        "  --suite lean\n"
    ) catch {};
}

fn parseArgs(args: []const [:0]u8) !Options {
    var options = Options{
        .root = "vendor/layerIntelligence/lean4/tests",
        .suite = "lean",
        .limit = 0,
        .json_output = false,
        .absolute = false,
    };

    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i][0..args[i].len];
        if (mem.eql(u8, arg, "--root")) {
            if (i + 1 >= args.len) {
                printUsage();
                return error.InvalidArgs;
            }
            options.root = args[i + 1][0..args[i + 1].len];
            i += 2;
            continue;
        }
        if (mem.eql(u8, arg, "--suite")) {
            if (i + 1 >= args.len) {
                printUsage();
                return error.InvalidArgs;
            }
            options.suite = args[i + 1][0..args[i + 1].len];
            i += 2;
            continue;
        }
        if (mem.eql(u8, arg, "--limit")) {
            if (i + 1 >= args.len) {
                printUsage();
                return error.InvalidArgs;
            }
            options.limit = std.fmt.parseInt(usize, args[i + 1][0..args[i + 1].len], 10) catch {
                printUsage();
                return error.InvalidArgs;
            };
            i += 2;
            continue;
        }
        if (mem.eql(u8, arg, "--json")) {
            options.json_output = true;
            i += 1;
            continue;
        }
        if (mem.eql(u8, arg, "--absolute")) {
            options.absolute = true;
            i += 1;
            continue;
        }
        if (mem.eql(u8, arg, "--help") or mem.eql(u8, arg, "-h")) {
            printUsage();
            std.process.exit(0);
        }
        printUsage();
        return error.InvalidArgs;
    }

    return options;
}

fn isLeanFile(path: []const u8) bool {
    return mem.endsWith(u8, path, ".lean");
}

fn hasExpectedOut(dir: std.fs.Dir, allocator: std.mem.Allocator, rel_path: []const u8) bool {
    const expected_path = std.fmt.allocPrint(allocator, "{s}.expected.out", .{rel_path}) catch return false;
    defer allocator.free(expected_path);

    if (dir.access(expected_path, .{})) |_| {
        return true;
    } else |_| {
        return false;
    }
}

fn dirExists(path: []const u8) bool {
    if (std.fs.cwd().openDir(path, .{})) |dir| {
        var mutable_dir = dir;
        mutable_dir.close();
        return true;
    } else |_| {
        return false;
    }
}

fn resolveRoot(allocator: std.mem.Allocator, root: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(root)) {
        return allocator.dupe(u8, root);
    }
    if (dirExists(root)) {
        return allocator.dupe(u8, root);
    }

    var depth: usize = 1;
    while (depth <= 6) : (depth += 1) {
        var parts = std.ArrayList([]const u8).empty;
        defer parts.deinit(allocator);

        var i: usize = 0;
        while (i < depth) : (i += 1) {
            parts.append(allocator, "..") catch return allocator.dupe(u8, root);
        }
        parts.append(allocator, root) catch return allocator.dupe(u8, root);

        const candidate = std.fs.path.join(allocator, parts.items) catch return allocator.dupe(u8, root);
        if (dirExists(candidate)) {
            return candidate;
        }
        allocator.free(candidate);
    }

    return allocator.dupe(u8, root);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const options = parseArgs(args) catch return;

    const resolved_root = try resolveRoot(allocator, options.root);
    defer allocator.free(resolved_root);

    const scan_root = if (options.suite) |suite|
        try std.fs.path.join(allocator, &.{ resolved_root, suite })
    else
        try allocator.dupe(u8, resolved_root);
    defer allocator.free(scan_root);

    const suite_name = if (options.suite) |suite|
        suite
    else
        std.fs.path.basename(scan_root);

    var absolute_root: ?[]u8 = null;
    defer if (absolute_root) |root| allocator.free(root);
    if (options.absolute) {
        absolute_root = std.fs.cwd().realpathAlloc(allocator, scan_root) catch null;
    }

    var dir = try std.fs.cwd().openDir(scan_root, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    const stdout = std.fs.File.stdout().deprecatedWriter();
    var out_writer = stdout;

    if (options.json_output) {
        try out_writer.writeAll("[");
    }

    var count: usize = 0;
    var first = true;
    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!isLeanFile(entry.path)) continue;

        const expected = hasExpectedOut(dir, allocator, entry.path);
        const output_path = if (options.absolute and absolute_root != null)
            try std.fs.path.join(allocator, &.{ absolute_root.?, entry.path })
        else
            entry.path;
        defer if (options.absolute and absolute_root != null) allocator.free(output_path);

        if (options.json_output) {
            if (!first) {
                try out_writer.writeAll(",");
            }
            first = false;
            const item = Entry{
                .path = output_path,
                .suite = suite_name,
                .has_expected = expected,
            };
            const json_entry = try json.Stringify.valueAlloc(allocator, item, .{});
            defer allocator.free(json_entry);
            try out_writer.writeAll(json_entry);
        } else {
            try out_writer.print("{s}\n", .{output_path});
        }

        count += 1;
        if (options.limit != 0 and count >= options.limit) {
            break;
        }
    }

    if (options.json_output) {
        try out_writer.writeAll("]\n");
    }
}
