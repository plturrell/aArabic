const std = @import("std");

const json = std.json;
const mem = std.mem;

const Options = struct {
    root: []const u8,
    suite: ?[]const u8,
    sample: usize,
    output: ?[]const u8,
    absolute: bool,
};

const SampleEntry = struct {
    path: []const u8,
    has_expected: bool,
};

const Report = struct {
    root: []const u8,
    suite: []const u8,
    total: usize,
    with_expected: usize,
    without_expected: usize,
    samples: []const SampleEntry,
};

fn printUsage() void {
    const out = std.fs.File.stderr().deprecatedWriter();
    _ = out.write(
        "Usage: lean4-baseline [--root PATH] [--suite NAME] [--sample N] [--output PATH] [--absolute]\n" ++
        "Defaults:\n" ++
        "  --root   vendor/layerIntelligence/lean4/tests\n" ++
        "  --suite  lean\n" ++
        "  --sample 20\n"
    ) catch {};
}

fn parseArgs(args: []const [:0]u8) !Options {
    var options = Options{
        .root = "vendor/layerIntelligence/lean4/tests",
        .suite = "lean",
        .sample = 20,
        .output = null,
        .absolute = false,
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
        if (mem.eql(u8, arg, "--sample")) {
            if (i + 1 >= args.len) return error.InvalidArgs;
            options.sample = std.fmt.parseInt(usize, args[i + 1][0..args[i + 1].len], 10) catch return error.InvalidArgs;
            i += 2;
            continue;
        }
        if (mem.eql(u8, arg, "--output")) {
            if (i + 1 >= args.len) return error.InvalidArgs;
            options.output = args[i + 1][0..args[i + 1].len];
            i += 2;
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

    var samples = std.ArrayList(SampleEntry).empty;
    defer {
        for (samples.items) |entry| {
            allocator.free(entry.path);
        }
        samples.deinit(allocator);
    }

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    var total: usize = 0;
    var with_expected: usize = 0;
    var without_expected: usize = 0;

    var absolute_root: ?[]u8 = null;
    defer if (absolute_root) |root| allocator.free(root);
    if (options.absolute) {
        absolute_root = std.fs.cwd().realpathAlloc(allocator, scan_root) catch null;
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

        if (samples.items.len < options.sample) {
            const output_path = if (options.absolute and absolute_root != null)
                try std.fs.path.join(allocator, &.{ absolute_root.?, entry.path })
            else
                try allocator.dupe(u8, entry.path);

            try samples.append(allocator, SampleEntry{
                .path = output_path,
                .has_expected = expected,
            });
        }
    }

    const report = Report{
        .root = resolved_root,
        .suite = suite_name,
        .total = total,
        .with_expected = with_expected,
        .without_expected = without_expected,
        .samples = samples.items,
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
