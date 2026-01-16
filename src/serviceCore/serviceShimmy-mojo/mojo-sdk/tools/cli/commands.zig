// Mojo CLI Commands Implementation
// All command handlers for the CLI tool

const std = @import("std");
const runner = @import("runner.zig");
const builder = @import("builder.zig");
const tester = @import("tester.zig");
const formatter = @import("formatter.zig");
const docgen = @import("docgen.zig");
const repl_mod = @import("repl.zig");

const Allocator = std.mem.Allocator;

// ============================================================================
// mojo run - JIT compilation and execution
// ============================================================================

pub fn run(allocator: Allocator, args: []const []const u8) !void {
    if (args.len == 0) {
        std.debug.print("Error: No input file specified\n", .{});
        std.debug.print("Usage: mojo run [file.mojo] [options]\n", .{});
        return error.MissingInputFile;
    }

    var options = runner.RunOptions{
        .file = args[0],
        .optimize_level = 0,
        .verbose = false,
        .program_args = &[_][]const u8{},
    };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "-O") or std.mem.eql(u8, arg, "--optimize")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: -O requires an argument\n", .{});
                return error.MissingArgument;
            }
            options.optimize_level = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            options.verbose = true;
        } else if (std.mem.eql(u8, arg, "--")) {
            // Everything after -- goes to the program
            options.program_args = args[i + 1 ..];
            break;
        } else if (std.mem.eql(u8, arg, "--help")) {
            try printRunHelp();
            return;
        } else {
            std.debug.print("Unknown option: {s}\n", .{arg});
            return error.UnknownOption;
        }
    }

    try runner.runFile(allocator, options);
}

fn printRunHelp() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\mojo run - JIT compile and execute a Mojo file
        \\
        \\Usage: mojo run [file.mojo] [options]
        \\
        \\Options:
        \\  -O, --optimize <level>   Optimization level (0-3)
        \\  -v, --verbose           Verbose output
        \\  --                      Pass remaining args to program
        \\  --help                  Show this help
        \\
        \\Examples:
        \\  mojo run hello.mojo
        \\  mojo run app.mojo -O2
        \\  mojo run script.mojo -- arg1 arg2
        \\
    , .{});
}

// ============================================================================
// mojo build - AOT compilation
// ============================================================================

pub fn build(allocator: Allocator, args: []const []const u8) !void {
    if (args.len == 0) {
        std.debug.print("Error: No input file specified\n", .{});
        std.debug.print("Usage: mojo build [file.mojo] -o [output]\n", .{});
        return error.MissingInputFile;
    }

    var options = builder.BuildOptions{
        .file = args[0],
        .output = null,
        .optimize_level = 0,
        .release = false,
        .strip = false,
        .static = false,
        .verbose = false,
    };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: -o requires an argument\n", .{});
                return error.MissingArgument;
            }
            options.output = args[i];
        } else if (std.mem.eql(u8, arg, "-O") or std.mem.eql(u8, arg, "--optimize")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: -O requires an argument\n", .{});
                return error.MissingArgument;
            }
            options.optimize_level = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "-r") or std.mem.eql(u8, arg, "--release")) {
            options.release = true;
            options.optimize_level = 3;
        } else if (std.mem.eql(u8, arg, "--strip")) {
            options.strip = true;
        } else if (std.mem.eql(u8, arg, "--static")) {
            options.static = true;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            options.verbose = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            try printBuildHelp();
            return;
        } else {
            std.debug.print("Unknown option: {s}\n", .{arg});
            return error.UnknownOption;
        }
    }

    try builder.buildFile(allocator, options);
}

fn printBuildHelp() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\mojo build - AOT compile to native binary
        \\
        \\Usage: mojo build [file.mojo] -o [output] [options]
        \\
        \\Options:
        \\  -o, --output <file>     Output file name
        \\  -O, --optimize <level>  Optimization level (0-3)
        \\  -r, --release           Release mode (O3 + optimizations)
        \\  --strip                 Strip debug symbols
        \\  --static                Static linking
        \\  -v, --verbose           Verbose output
        \\  --help                  Show this help
        \\
        \\Examples:
        \\  mojo build app.mojo -o myapp
        \\  mojo build app.mojo -o myapp --release
        \\  mojo build lib.mojo -o lib.a --static
        \\
    , .{});
}

// ============================================================================
// mojo test - Test runner
// ============================================================================

pub fn test_(allocator: Allocator, args: []const []const u8) !void {
    var options = tester.TestOptions{
        .filter = null,
        .verbose = false,
        .junit_output = null,
    };

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--filter")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: -f requires an argument\n", .{});
                return error.MissingArgument;
            }
            options.filter = args[i];
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            options.verbose = true;
        } else if (std.mem.eql(u8, arg, "--junit")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --junit requires an argument\n", .{});
                return error.MissingArgument;
            }
            options.junit_output = args[i];
        } else if (std.mem.eql(u8, arg, "--help")) {
            try printTestHelp();
            return;
        } else {
            // Treat as filter pattern
            options.filter = arg;
        }
    }

    try tester.runTests(allocator, options);
}

fn printTestHelp() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\mojo test - Run test suite
        \\
        \\Usage: mojo test [pattern] [options]
        \\
        \\Options:
        \\  -f, --filter <pattern>  Filter tests by pattern
        \\  -v, --verbose           Verbose test output
        \\  --junit <file>          Output JUnit XML
        \\  --help                  Show this help
        \\
        \\Examples:
        \\  mojo test
        \\  mojo test --filter "list_*"
        \\  mojo test -v --junit results.xml
        \\
    , .{});
}

// ============================================================================
// mojo format - Code formatter
// ============================================================================

pub fn format(allocator: Allocator, args: []const []const u8) !void {
    var options = formatter.FormatOptions{
        .files = &[_][]const u8{},
        .write = false,
        .check = false,
        .recursive = false,
    };

    var files = try std.ArrayList([]const u8).initCapacity(allocator, 0);
    defer files.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "-w") or std.mem.eql(u8, arg, "--write")) {
            options.write = true;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--check")) {
            options.check = true;
        } else if (std.mem.eql(u8, arg, "-r") or std.mem.eql(u8, arg, "--recursive")) {
            options.recursive = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            try printFormatHelp();
            return;
        } else {
            try files.append(arg);
        }
    }

    if (files.items.len == 0) {
        std.debug.print("Error: No files specified\n", .{});
        std.debug.print("Usage: mojo format [files...] [options]\n", .{});
        return error.MissingFiles;
    }

    options.files = files.items;
    try formatter.formatFiles(allocator, options);
}

fn printFormatHelp() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\mojo format - Format Mojo source files
        \\
        \\Usage: mojo format [files...] [options]
        \\
        \\Options:
        \\  -w, --write       Write changes to files
        \\  -c, --check       Check formatting without writing
        \\  -r, --recursive   Format recursively
        \\  --help            Show this help
        \\
        \\Examples:
        \\  mojo format file.mojo --write
        \\  mojo format src/**/*.mojo --check
        \\  mojo format . --recursive --write
        \\
    , .{});
}

// ============================================================================
// mojo doc - Documentation generator
// ============================================================================

pub fn doc(allocator: Allocator, args: []const []const u8) !void {
    var options = docgen.DocOptions{
        .input_dir = ".",
        .output_dir = "docs",
        .format = .html,
        .include_private = false,
    };

    var i: usize = 0;
    if (i < args.len and !std.mem.startsWith(u8, args[i], "-")) {
        options.input_dir = args[i];
        i += 1;
    }

    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: -o requires an argument\n", .{});
                return error.MissingArgument;
            }
            options.output_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--format")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --format requires an argument\n", .{});
                return error.MissingArgument;
            }
            if (std.mem.eql(u8, args[i], "html")) {
                options.format = .html;
            } else if (std.mem.eql(u8, args[i], "markdown")) {
                options.format = .markdown;
            } else {
                std.debug.print("Unknown format: {s}\n", .{args[i]});
                return error.UnknownFormat;
            }
        } else if (std.mem.eql(u8, arg, "--private")) {
            options.include_private = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            try printDocHelp();
            return;
        } else {
            std.debug.print("Unknown option: {s}\n", .{arg});
            return error.UnknownOption;
        }
    }

    try docgen.generateDocs(allocator, options);
}

fn printDocHelp() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\mojo doc - Generate documentation
        \\
        \\Usage: mojo doc [directory] [options]
        \\
        \\Options:
        \\  -o, --output <dir>    Output directory (default: docs)
        \\  --format <fmt>        Output format: html, markdown (default: html)
        \\  --private             Include private items
        \\  --help                Show this help
        \\
        \\Examples:
        \\  mojo doc
        \\  mojo doc src -o api-docs
        \\  mojo doc --format markdown --private
        \\
    , .{});
}

// ============================================================================
// mojo repl - Interactive REPL
// ============================================================================

pub fn repl(allocator: Allocator, args: []const []const u8) !void {
    var options = repl_mod.ReplOptions{
        .verbose = false,
    };

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            options.verbose = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            try printReplHelp();
            return;
        } else {
            std.debug.print("Unknown option: {s}\n", .{arg});
            return error.UnknownOption;
        }
    }

    try repl_mod.startRepl(allocator, options);
}

fn printReplHelp() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\mojo repl - Start interactive REPL
        \\
        \\Usage: mojo repl [options]
        \\
        \\Options:
        \\  -v, --verbose   Verbose REPL output
        \\  --help          Show this help
        \\
        \\REPL Commands:
        \\  :quit, :q       Exit REPL
        \\  :help, :h       Show REPL help
        \\  :clear, :c      Clear screen
        \\  :reset, :r      Reset REPL state
        \\  :vars           Show variables
        \\  :type <expr>    Show type of expression
        \\
        \\Examples:
        \\  mojo repl
        \\  mojo repl --verbose
        \\
    , .{});
}
