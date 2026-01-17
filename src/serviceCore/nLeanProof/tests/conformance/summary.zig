const std = @import("std");

const json = std.json;
const mem = std.mem;

const Options = struct {
    root: []const u8,
    suite: ?[]const u8,
    output: ?[]const u8,
};

const DirStat = struct {
    dir: []const u8,
    total: usize,
    with_expected: usize,
    without_expected: usize,
};

const Report = struct {
    root: []const u8,
    suite: []const u8,
    total: usize,
    with_expected: usize,
    without_expected: usize,
    expected_ratio: f64,
    by_dir: []const DirStat,
};

fn printUsage() void {
    const out = std.fs.File.stderr().deprecatedWriter();
    _ = out.write(
        "Usage: lean4-summary [--root PATH] [--suite NAME] [--output PATH]\n" ++
        "Defaults:\n" ++
        "  --root  vendor/layerIntelligence/lean4/tests\n" ++
        "  --suite lean\n"
    ) catch {};
}

fn parseArgs(args: []const [:0]u8) !Options {
    var options = Options{
        .root = "vendor/layerIntelligence/lean4/tests",
        .suite = "lean",
        .output = null,
    };

    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i][0..args[i].len];
        if (mem.eql(u8, arg, "--root")) {
            if (i + 1 >= args.len) return error.InvalidArgs;
            options.root = args[i + 1][0..args[i + 1].len];
            i += 2;
            continue;
        }
        if (mem.eql(u8, arg, "--suite")) {
            if (i + 1 >= args.len) return error.InvalidArgs;
            options.suite = args[i + 1][0..args[i + 1].len];
            i += 2;
            continue;
        }
        if (mem.eql(u8, arg, "--output")) {
            if (i + 1 >= args.len) return error.InvalidArgs;
            options.output = args[i + 1][0..args[i + 1].len];
            i += 2;
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

fn topLevelDir(path: []const u8) []const u8 {
    if (mem.indexOfScalar(u8, path, '/')) |idx| {
        return path[0..idx];
    }
    if (mem.indexOfScalar(u8, path, '\\')) |idx| {
        return path[0..idx];
    }
    return "root";
}

fn writeReport(writer: anytype, allocator: std.mem.Allocator, report: Report) !void {
    const payload = try json.Stringify.valueAlloc(allocator, report, .{});
    defer allocator.free(payload);
    try writer.writeAll(payload);
    try writer.writeAll("\n");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const options = parseArgs(args) catch {
        printUsage();
        return;
    };

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

    var dir = try std.fs.cwd().openDir(scan_root, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    var total: usize = 0;
    var with_expected: usize = 0;
    var without_expected: usize = 0;

    const DirCounts = struct {
        total: usize,
        with_expected: usize,
        without_expected: usize,
    };

    var by_dir = std.StringHashMap(DirCounts).init(allocator);
    defer {
        var it = by_dir.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        by_dir.deinit();
    }

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!isLeanFile(entry.path)) continue;

        total += 1;
        const expected = hasExpectedOut(dir, allocator, entry.path);
        if (expected) {
            with_expected += 1;
        } else {
            without_expected += 1;
        }

        const dir_key = topLevelDir(entry.path);
        if (by_dir.getEntry(dir_key)) |found| {
            found.value_ptr.total += 1;
            if (expected) {
                found.value_ptr.with_expected += 1;
            } else {
                found.value_ptr.without_expected += 1;
            }
        } else {
            const key_copy = try allocator.dupe(u8, dir_key);
            try by_dir.put(key_copy, DirCounts{
                .total = 1,
                .with_expected = if (expected) 1 else 0,
                .without_expected = if (expected) 0 else 1,
            });
        }
    }

    var stats = std.ArrayList(DirStat).empty;
    defer stats.deinit(allocator);

    var iter = by_dir.iterator();
    while (iter.next()) |entry| {
        try stats.append(allocator, DirStat{
            .dir = entry.key_ptr.*,
            .total = entry.value_ptr.total,
            .with_expected = entry.value_ptr.with_expected,
            .without_expected = entry.value_ptr.without_expected,
        });
    }

    std.mem.sort(DirStat, stats.items, {}, struct {
        fn lessThan(_: void, a: DirStat, b: DirStat) bool {
            return std.mem.lessThan(u8, a.dir, b.dir);
        }
    }.lessThan);

    const ratio = if (total == 0)
        0.0
    else
        @as(f64, @floatFromInt(with_expected)) / @as(f64, @floatFromInt(total));

    const report = Report{
        .root = resolved_root,
        .suite = suite_name,
        .total = total,
        .with_expected = with_expected,
        .without_expected = without_expected,
        .expected_ratio = ratio,
        .by_dir = stats.items,
    };

    if (options.output) |output_path| {
        var file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
        defer file.close();
        const writer = file.deprecatedWriter();
        try writeReport(writer, allocator, report);
    } else {
        const stdout = std.fs.File.stdout().deprecatedWriter();
        try writeReport(stdout, allocator, report);
    }
}
