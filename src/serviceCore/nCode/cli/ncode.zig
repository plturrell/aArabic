//! nCode CLI Tool (Zig implementation)
//! Command-line interface for the nCode SCIP code intelligence platform
//!
//! Commands:
//!   ncode index <path>           - Index a project
//!   ncode search <query>         - Semantic search across code
//!   ncode query <cypher>         - Run Cypher query on graph
//!   ncode definition <file:line> - Find symbol definition
//!   ncode references <file:line> - Find symbol references
//!   ncode symbols <file>         - List symbols in file
//!   ncode export <format>        - Export index data
//!   ncode health                 - Check server health
//!   ncode interactive            - Interactive mode

const std = @import("std");
const ncode = @import("../client/ncode_client.zig");

const Command = enum {
    index,
    search,
    query,
    definition,
    references,
    symbols,
    export,
    health,
    interactive,
    help,
};

const Config = struct {
    server_url: []const u8 = "http://localhost:18003",
    qdrant_url: []const u8 = "http://localhost:6333",
    memgraph_url: []const u8 = "bolt://localhost:7687",
    verbose: bool = false,
    json_output: bool = false,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printHelp();
        return;
    }

    var config = Config{};
    const command_str = args[1];
    
    // Parse command
    const command = std.meta.stringToEnum(Command, command_str) orelse {
        std.debug.print("Unknown command: {s}\n", .{command_str});
        printHelp();
        return error.InvalidCommand;
    };

    // Execute command
    switch (command) {
        .index => try cmdIndex(allocator, args[2..], config),
        .search => try cmdSearch(allocator, args[2..], config),
        .query => try cmdQuery(allocator, args[2..], config),
        .definition => try cmdDefinition(allocator, args[2..], config),
        .references => try cmdReferences(allocator, args[2..], config),
        .symbols => try cmdSymbols(allocator, args[2..], config),
        .export => try cmdExport(allocator, args[2..], config),
        .health => try cmdHealth(allocator, config),
        .interactive => try cmdInteractive(allocator, config),
        .help => printHelp(),
    }
}

fn cmdIndex(allocator: std.mem.Allocator, args: [][]const u8, config: Config) !void {
    if (args.len < 1) {
        std.debug.print("Usage: ncode index <path> [--language <lang>]\n", .{});
        return error.InvalidArgs;
    }

    const path = args[0];
    std.debug.print("Indexing: {s}\n", .{path});

    // Run indexer based on language detection
    const ext = std.fs.path.extension(path);
    const indexer = detectIndexer(ext);
    
    std.debug.print("Using indexer: {s}\n", .{indexer});
    
    // Execute indexer command
    var child = std.ChildProcess.init(&[_][]const u8{
        indexer,
        "index",
        path,
    }, allocator);
    
    const result = try child.spawnAndWait();
    if (result.Exited != 0) {
        return error.IndexingFailed;
    }

    // Load index into server
    const client = try ncode.NCodeClient.init(allocator, .{
        .base_url = config.server_url,
    });
    defer client.deinit();

    const load_result = try client.loadIndex("index.scip");
    std.debug.print("✓ Loaded {d} symbols from {d} documents\n", .{
        load_result.symbols orelse 0,
        load_result.documents orelse 0,
    });
}

fn cmdSearch(allocator: std.mem.Allocator, args: [][]const u8, config: Config) !void {
    if (args.len < 1) {
        std.debug.print("Usage: ncode search <query> [--limit <n>]\n", .{});
        return error.InvalidArgs;
    }

    const query = args[0];
    const limit: usize = if (args.len > 2 and std.mem.eql(u8, args[1], "--limit"))
        try std.fmt.parseInt(usize, args[2], 10)
    else
        10;

    std.debug.print("Searching for: {s}\n", .{query});

    const qdrant = try ncode.QdrantClient.init(
        allocator,
        config.qdrant_url,
        "ncode",
    );
    defer qdrant.deinit();

    const results = try qdrant.semanticSearch(query, limit);
    std.debug.print("Results: {s}\n", .{results});
}

fn cmdQuery(allocator: std.mem.Allocator, args: [][]const u8, config: Config) !void {
    if (args.len < 1) {
        std.debug.print("Usage: ncode query <cypher-query>\n", .{});
        return error.InvalidArgs;
    }

    const query = args[0];
    std.debug.print("Running query: {s}\n", .{query});

    const memgraph = try ncode.MemgraphClient.init(allocator, config.memgraph_url);
    defer memgraph.deinit();

    // Execute Cypher query
    std.debug.print("Query results: (implementation needed)\n", .{});
}

fn cmdDefinition(allocator: std.mem.Allocator, args: [][]const u8, config: Config) !void {
    if (args.len < 1) {
        std.debug.print("Usage: ncode definition <file:line:char>\n", .{});
        return error.InvalidArgs;
    }

    const location = try parseLocation(args[0]);
    
    const client = try ncode.NCodeClient.init(allocator, .{
        .base_url = config.server_url,
    });
    defer client.deinit();

    const result = try client.findDefinition(.{
        .file = location.file,
        .line = location.line,
        .character = location.character,
    });

    if (result.location) |loc| {
        std.debug.print("Definition: {s}:{d}:{d}\n", .{
            loc.uri,
            loc.range.start.line,
            loc.range.start.character,
        });
        if (result.symbol) |sym| {
            std.debug.print("Symbol: {s}\n", .{sym});
        }
    } else {
        std.debug.print("No definition found\n", .{});
    }
}

fn cmdReferences(allocator: std.mem.Allocator, args: [][]const u8, config: Config) !void {
    if (args.len < 1) {
        std.debug.print("Usage: ncode references <file:line:char>\n", .{});
        return error.InvalidArgs;
    }

    const location = try parseLocation(args[0]);
    
    const client = try ncode.NCodeClient.init(allocator, .{
        .base_url = config.server_url,
    });
    defer client.deinit();

    const result = try client.findReferences(.{
        .file = location.file,
        .line = location.line,
        .character = location.character,
    });

    std.debug.print("Found {d} references:\n", .{result.locations.len});
    for (result.locations) |loc| {
        std.debug.print("  {s}:{d}:{d}\n", .{
            loc.uri,
            loc.range.start.line,
            loc.range.start.character,
        });
    }
}

fn cmdSymbols(allocator: std.mem.Allocator, args: [][]const u8, config: Config) !void {
    if (args.len < 1) {
        std.debug.print("Usage: ncode symbols <file>\n", .{});
        return error.InvalidArgs;
    }

    const file = args[0];
    
    const client = try ncode.NCodeClient.init(allocator, .{
        .base_url = config.server_url,
    });
    defer client.deinit();

    const result = try client.getSymbols(file);
    
    std.debug.print("Symbols in {s}:\n", .{file});
    for (result.symbols) |sym| {
        std.debug.print("  {s} {s} at line {d}\n", .{
            sym.kind,
            sym.name,
            sym.range.start.line,
        });
    }
}

fn cmdExport(allocator: std.mem.Allocator, args: [][]const u8, config: Config) !void {
    if (args.len < 1) {
        std.debug.print("Usage: ncode export <format> [--output <file>]\n", .{});
        std.debug.print("Formats: json, csv, graphml\n", .{});
        return error.InvalidArgs;
    }

    const format = args[0];
    _ = format;
    _ = allocator;
    _ = config;
    
    std.debug.print("Export functionality coming soon\n", .{});
}

fn cmdHealth(allocator: std.mem.Allocator, config: Config) !void {
    const client = try ncode.NCodeClient.init(allocator, .{
        .base_url = config.server_url,
    });
    defer client.deinit();

    const health = try client.health();
    
    std.debug.print("nCode Server Health\n", .{});
    std.debug.print("  Status: {s}\n", .{health.status});
    std.debug.print("  Version: {s}\n", .{health.version});
    if (health.uptime_seconds) |uptime| {
        std.debug.print("  Uptime: {d:.2} seconds\n", .{uptime});
    }
    if (health.index_loaded) |loaded| {
        std.debug.print("  Index Loaded: {}\n", .{loaded});
    }
}

fn cmdInteractive(allocator: std.mem.Allocator, config: Config) !void {
    std.debug.print("nCode Interactive Mode\n", .{});
    std.debug.print("Type 'help' for commands, 'exit' to quit\n\n", .{});

    const stdin = std.io.getStdIn().reader();
    const stdout = std.io.getStdOut().writer();

    var buf: [1024]u8 = undefined;
    
    while (true) {
        try stdout.writeAll("> ");
        
        const line = (try stdin.readUntilDelimiterOrEof(&buf, '\n')) orelse break;
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit")) break;
        if (std.mem.eql(u8, trimmed, "help")) {
            printHelp();
            continue;
        }
        
        // Parse and execute command
        var iter = std.mem.split(u8, trimmed, " ");
        var cmd_args = std.ArrayList([]const u8).init(allocator);
        defer cmd_args.deinit();
        
        while (iter.next()) |arg| {
            try cmd_args.append(arg);
        }
        
        // Execute command (simplified)
        std.debug.print("Executing: {s}\n", .{trimmed});
    }
}

fn parseLocation(loc_str: []const u8) !struct { file: []const u8, line: i32, character: i32 } {
    var iter = std.mem.split(u8, loc_str, ":");
    
    const file = iter.next() orelse return error.InvalidLocation;
    const line_str = iter.next() orelse return error.InvalidLocation;
    const char_str = iter.next() orelse return error.InvalidLocation;
    
    return .{
        .file = file,
        .line = try std.fmt.parseInt(i32, line_str, 10),
        .character = try std.fmt.parseInt(i32, char_str, 10),
    };
}

fn detectIndexer(ext: []const u8) []const u8 {
    if (std.mem.eql(u8, ext, ".ts") or std.mem.eql(u8, ext, ".tsx")) {
        return "scip-typescript";
    } else if (std.mem.eql(u8, ext, ".py")) {
        return "scip-python";
    } else if (std.mem.eql(u8, ext, ".java")) {
        return "scip-java";
    } else if (std.mem.eql(u8, ext, ".rs")) {
        return "rust-analyzer";
    } else {
        return "ncode-treesitter";
    }
}

fn printHelp() void {
    const help_text =
        \\nCode CLI - SCIP Code Intelligence Platform
        \\
        \\Usage: ncode <command> [options]
        \\
        \\Commands:
        \\  index <path>              Index a project or file
        \\  search <query>            Semantic search across code
        \\  query <cypher>            Run Cypher query on graph database
        \\  definition <file:line:char>  Find symbol definition
        \\  references <file:line:char>  Find symbol references
        \\  symbols <file>            List symbols in file
        \\  export <format>           Export index data (json, csv, graphml)
        \\  health                    Check server health
        \\  interactive               Start interactive mode
        \\  help                      Show this help message
        \\
        \\Options:
        \\  --server <url>            nCode server URL (default: http://localhost:18003)
        \\  --qdrant <url>            Qdrant URL (default: http://localhost:6333)
        \\  --memgraph <url>          Memgraph URL (default: bolt://localhost:7687)
        \\  --verbose                 Enable verbose output
        \\  --json                    Output in JSON format
        \\
        \\Examples:
        \\  ncode index src/
        \\  ncode search "authentication function"
        \\  ncode definition src/main.zig:10:5
        \\  ncode references src/utils.zig:25:10
        \\  ncode symbols src/main.zig
        \\  ncode query "MATCH (n:Symbol) RETURN n LIMIT 10"
        \\  ncode export json --output index.json
        \\  ncode interactive
        \\
        \\For more information: https://github.com/ncode/docs
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
