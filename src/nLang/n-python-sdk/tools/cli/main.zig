// Mojo CLI Tool - Main Entry Point
// Complete command-line interface for Mojo compiler
// With full internationalization support (33 languages)

const std = @import("std");
const builtin = @import("builtin");
const commands = @import("commands.zig");
const i18n_cli = @import("i18n_cli.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize i18n from system locale
    i18n_cli.initCli(allocator);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        std.process.exit(1);
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "run")) {
        return commands.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, command, "build")) {
        return commands.build(allocator, args[2..]);
    } else if (std.mem.eql(u8, command, "test")) {
        return commands.test_(allocator, args[2..]);
    } else if (std.mem.eql(u8, command, "format")) {
        return commands.format(allocator, args[2..]);
    } else if (std.mem.eql(u8, command, "doc")) {
        return commands.doc(allocator, args[2..]);
    } else if (std.mem.eql(u8, command, "repl")) {
        return commands.repl(allocator, args[2..]);
    } else if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version")) {
        try printVersion();
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        try printHelp();
    } else {
        i18n_cli.printError(.err_unknown_command, command);
        try printUsage();
        std.process.exit(1);
    }
}

fn printUsage() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\Usage: mojo <command> [options]
        \\
        \\Commands:
        \\  run      JIT compile and execute a Mojo file
        \\  build    AOT compile to native binary
        \\  test     Run test suite
        \\  format   Format Mojo source files
        \\  doc      Generate documentation
        \\  repl     Start interactive REPL
        \\  version  Show version information
        \\  help     Show this help message
        \\
        \\Use 'mojo <command> --help' for more information on a command.
        \\
    , .{});
}

fn printVersion() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print("Mojo version 0.1.0 (SDK {s})\n", .{@tagName(builtin.target.cpu.arch)});
    std.debug.print("Copyright (c) 2026 Mojo Project\n", .{});
}

fn printHelp() !void {
    // const stdout = std.io.getStdOut().writer();
    std.debug.print(
        \\Mojo - A high-performance systems programming language
        \\
        \\Usage: mojo <command> [options]
        \\
        \\Commands:
        \\
        \\  run [file.mojo]
        \\    JIT compile and execute a Mojo file
        \\    Options:
        \\      -O, --optimize <level>    Optimization level (0-3)
        \\      -v, --verbose            Verbose output
        \\      --                       Pass remaining args to program
        \\
        \\  build [file.mojo] -o [output]
        \\    AOT compile to native binary
        \\    Options:
        \\      -o, --output <file>      Output file name
        \\      -O, --optimize <level>   Optimization level (0-3)
        \\      -r, --release            Release mode (O3 + optimizations)
        \\      --strip                  Strip debug symbols
        \\      --static                 Static linking
        \\
        \\  test [pattern]
        \\    Run test suite
        \\    Options:
        \\      -v, --verbose            Verbose test output
        \\      -f, --filter <pattern>   Filter tests by pattern
        \\      --junit <file>           Output JUnit XML
        \\
        \\  format [files...]
        \\    Format Mojo source files
        \\    Options:
        \\      -w, --write              Write changes to files
        \\      -c, --check              Check formatting without writing
        \\      -r, --recursive          Format recursively
        \\
        \\  doc [directory]
        \\    Generate documentation
        \\    Options:
        \\      -o, --output <dir>       Output directory
        \\      --format <fmt>           Output format (html, markdown)
        \\      --private                Include private items
        \\
        \\  repl
        \\    Start interactive REPL
        \\    Options:
        \\      -v, --verbose            Verbose REPL output
        \\
        \\  version, --version
        \\    Show version information
        \\
        \\  help, --help
        \\    Show this help message
        \\
        \\Examples:
        \\  mojo run hello.mojo
        \\  mojo build app.mojo -o myapp --release
        \\  mojo test --filter "list_*"
        \\  mojo format -w src/**/*.mojo
        \\  mojo doc -o docs/
        \\  mojo repl
        \\
        \\For more information, visit: https://docs.mojo.dev
        \\
    , .{});
}
