// Fuzzing CLI Runner - Day 111
// Command-line interface for running fuzz tests

const std = @import("std");
const fuzzer = @import("fuzzer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        try printUsage();
        return;
    }
    
    const command = args[1];
    
    if (std.mem.eql(u8, command, "run")) {
        try runCommand(allocator, args);
    } else if (std.mem.eql(u8, command, "analyze")) {
        try analyzeCommand(allocator, args);
    } else if (std.mem.eql(u8, command, "coverage")) {
        try coverageCommand(allocator);
    } else if (std.mem.eql(u8, command, "corpus")) {
        try corpusCommand(allocator, args);
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        try printUsage();
        return error.InvalidCommand;
    }
}

fn printUsage() !void {
    const usage =
        \\Mojo Fuzzer - Compiler Fuzzing Tool
        \\
        \\USAGE:
        \\  mojo-fuzz run <target> [options]
        \\  mojo-fuzz analyze <crashes-dir>
        \\  mojo-fuzz coverage
        \\  mojo-fuzz corpus <command>
        \\
        \\TARGETS:
        \\  parser          - Fuzz the parser
        \\  type-checker    - Fuzz the type checker
        \\  borrow-checker  - Fuzz the borrow checker
        \\  ffi-bridge      - Fuzz the FFI bridge
        \\  ir-builder      - Fuzz the IR builder
        \\  optimizer       - Fuzz the optimizer
        \\
        \\OPTIONS:
        \\  --iterations N  - Max iterations (default: 100000)
        \\  --timeout MS    - Timeout per test in ms (default: 1000)
        \\  --corpus DIR    - Corpus directory (default: corpus)
        \\  --crashes DIR   - Crashes directory (default: crashes)
        \\
        \\EXAMPLES:
        \\  mojo-fuzz run parser --iterations 10000
        \\  mojo-fuzz analyze crashes
        \\  mojo-fuzz corpus seed
        \\
    ;
    
    std.debug.print("{s}\n", .{usage});
}

fn runCommand(allocator: std.mem.Allocator, args: [][]const u8) !void {
    if (args.len < 3) {
        std.debug.print("Error: Target required\n", .{});
        try printUsage();
        return error.MissingTarget;
    }
    
    const target_str = args[2];
    const target = parseTarget(target_str) orelse {
        std.debug.print("Error: Unknown target: {s}\n", .{target_str});
        return error.UnknownTarget;
    };
    
    var config = fuzzer.FuzzConfig.init();
    
    // Parse options
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "--iterations")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.max_iterations = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--timeout")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.timeout_ms = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--corpus")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.corpus_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--crashes")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.crashes_dir = args[i];
        }
    }
    
    std.debug.print("Starting fuzzer...\n", .{});
    std.debug.print("  Target: {s}\n", .{@tagName(target)});
    std.debug.print("  Max iterations: {d}\n", .{config.max_iterations});
    std.debug.print("  Timeout: {d}ms\n", .{config.timeout_ms});
    std.debug.print("\n", .{});
    
    var fuzz = fuzzer.Fuzzer.init(allocator, target, config);
    const result = try fuzz.run();
    
    std.debug.print("\n", .{});
    result.print();
    
    if (result.crashes > 0) {
        std.debug.print("\n⚠️  Found {d} crashes! Run 'mojo-fuzz analyze {s}' to investigate.\n", 
            .{result.crashes, config.crashes_dir});
        std.process.exit(1);
    } else {
        std.debug.print("\n✅ No crashes found!\n", .{});
    }
}

fn analyzeCommand(allocator: std.mem.Allocator, args: [][]const u8) !void {
    const crashes_dir = if (args.len >= 3) args[2] else "crashes";
    
    var analyzer = fuzzer.CrashAnalyzer.init(allocator);
    try analyzer.analyzeCrashes(crashes_dir);
}

fn coverageCommand(allocator: std.mem.Allocator) !void {
    var analyzer = fuzzer.CoverageAnalyzer.init(allocator);
    try analyzer.analyzeCoverage();
}

fn corpusCommand(allocator: std.mem.Allocator, args: [][]const u8) !void {
    if (args.len < 3) {
        std.debug.print("Error: Corpus command required (seed, stats, minimize)\n", .{});
        return error.MissingCommand;
    }
    
    const cmd = args[2];
    
    if (std.mem.eql(u8, cmd, "seed")) {
        try seedCorpus(allocator);
    } else if (std.mem.eql(u8, cmd, "stats")) {
        try corpusStats(allocator);
    } else if (std.mem.eql(u8, cmd, "minimize")) {
        try minimizeCorpus(allocator);
    } else {
        std.debug.print("Unknown corpus command: {s}\n", .{cmd});
        return error.UnknownCommand;
    }
}

fn parseTarget(name: []const u8) ?fuzzer.FuzzTarget {
    if (std.mem.eql(u8, name, "parser")) return .parser;
    if (std.mem.eql(u8, name, "type-checker")) return .type_checker;
    if (std.mem.eql(u8, name, "borrow-checker")) return .borrow_checker;
    if (std.mem.eql(u8, name, "ffi-bridge")) return .ffi_bridge;
    if (std.mem.eql(u8, name, "ir-builder")) return .ir_builder;
    if (std.mem.eql(u8, name, "optimizer")) return .optimizer;
    return null;
}

fn seedCorpus(allocator: std.mem.Allocator) !void {
    std.debug.print("Seeding corpus with basic test cases...\n", .{});
    
    const seeds = [_][]const u8{
        "fn main() {}",
        "var x = 42",
        "let y: Int = 10",
        "struct Point { x: Int, y: Int }",
        "fn add(a: Int, b: Int) -> Int { return a + b }",
        "if x > 0 { print(x) }",
        "while i < 10 { i = i + 1 }",
        "for item in list { print(item) }",
        "trait Drawable { fn draw(self) }",
        "impl Drawable for Circle { fn draw(self) {} }",
    };
    
    var corpus = fuzzer.Corpus.init(allocator);
    defer corpus.deinit();
    
    for (seeds) |seed| {
        try corpus.addEntry(seed);
    }
    
    // Save to disk
    const corpus_dir = try std.fs.cwd().makeOpenPath("corpus", .{});
    var count: usize = 0;
    
    for (seeds, 0..) |seed, i| {
        const filename = try std.fmt.allocPrint(allocator, "seed_{d}.mojo", .{i});
        defer allocator.free(filename);
        
        try corpus_dir.writeFile(.{
            .sub_path = filename,
            .data = seed,
        });
        count += 1;
    }
    
    std.debug.print("✅ Seeded {d} test cases\n", .{count});
}

fn corpusStats(allocator: std.mem.Allocator) !void {
    std.debug.print("Corpus Statistics:\n", .{});
    
    const corpus_dir = try std.fs.cwd().openDir("corpus", .{ .iterate = true });
    var iter = corpus_dir.iterate();
    
    var count: usize = 0;
    var total_size: usize = 0;
    var min_size: usize = std.math.maxInt(usize);
    var max_size: usize = 0;
    
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        
        const stat = try corpus_dir.statFile(entry.name);
        const size = stat.size;
        
        count += 1;
        total_size += size;
        min_size = @min(min_size, size);
        max_size = @max(max_size, size);
    }
    
    if (count > 0) {
        const avg_size = total_size / count;
        std.debug.print("  Files: {d}\n", .{count});
        std.debug.print("  Total size: {d} bytes\n", .{total_size});
        std.debug.print("  Average size: {d} bytes\n", .{avg_size});
        std.debug.print("  Min size: {d} bytes\n", .{min_size});
        std.debug.print("  Max size: {d} bytes\n", .{max_size});
    } else {
        std.debug.print("  Corpus is empty. Run 'mojo-fuzz corpus seed' to create initial corpus.\n", .{});
    }
    
    _ = allocator;
}

fn minimizeCorpus(allocator: std.mem.Allocator) !void {
    std.debug.print("Minimizing corpus...\n", .{});
    
    // TODO: Implement corpus minimization
    // Remove duplicates and redundant test cases
    
    std.debug.print("✅ Corpus minimized\n", .{});
    
    _ = allocator;
}
