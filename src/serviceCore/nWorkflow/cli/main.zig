// nWorkflow CLI - Command-Line Interface for Workflow Management
// Provides commands for workflow, execution, node, server, and config management

const std = @import("std");
const mem = std.mem;
const fs = std.fs;
const process = std.process;
const Allocator = mem.Allocator;
const json = std.json;

// ============================================================================
// Constants and Version Info
// ============================================================================

pub const VERSION = "1.0.0";
pub const PROGRAM_NAME = "nwf";
pub const DEFAULT_SERVER = "http://localhost:8090";
pub const DEFAULT_FORMAT = "table";
pub const CONFIG_DIR = ".nworkflow";
pub const CONFIG_FILE = "config.json";

// Exit codes
pub const EXIT_SUCCESS: u8 = 0;
pub const EXIT_ERROR: u8 = 1;
pub const EXIT_USAGE_ERROR: u8 = 2;

// ============================================================================
// Output Formats
// ============================================================================

pub const OutputFormat = enum {
    json,
    table,
    yaml,

    pub fn fromString(str: []const u8) ?OutputFormat {
        if (mem.eql(u8, str, "json")) return .json;
        if (mem.eql(u8, str, "table")) return .table;
        if (mem.eql(u8, str, "yaml")) return .yaml;
        return null;
    }
};

// ============================================================================
// Global Options
// ============================================================================

pub const GlobalOptions = struct {
    server: []const u8 = DEFAULT_SERVER,
    token: ?[]const u8 = null,
    format: OutputFormat = .table,
    help: bool = false,
    version: bool = false,

    pub fn fromArgs(allocator: Allocator, args: *ArgParser) !GlobalOptions {
        var opts = GlobalOptions{};

        // Load config file defaults first
        if (Config.load(allocator)) |config| {
            if (config.server) |s| opts.server = s;
            if (config.token) |t| opts.token = t;
            if (config.format) |f| {
                if (OutputFormat.fromString(f)) |fmt| opts.format = fmt;
            }
        } else |_| {}

        // Override with command line args
        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--server") or mem.eql(u8, flag, "-s")) {
                _ = args.next();
                opts.server = args.next() orelse return error.MissingServerValue;
            } else if (mem.eql(u8, flag, "--token") or mem.eql(u8, flag, "-t")) {
                _ = args.next();
                opts.token = args.next() orelse return error.MissingTokenValue;
            } else if (mem.eql(u8, flag, "--format") or mem.eql(u8, flag, "-f")) {
                _ = args.next();
                const fmt_str = args.next() orelse return error.MissingFormatValue;
                opts.format = OutputFormat.fromString(fmt_str) orelse return error.InvalidFormat;
            } else if (mem.eql(u8, flag, "--help") or mem.eql(u8, flag, "-h")) {
                _ = args.next();
                opts.help = true;
            } else if (mem.eql(u8, flag, "--version") or mem.eql(u8, flag, "-v")) {
                _ = args.next();
                opts.version = true;
            } else {
                break;
            }
        }

        return opts;
    }
};

// ============================================================================
// Argument Parser
// ============================================================================

pub const ArgParser = struct {
    args: []const [:0]const u8,
    index: usize = 0,

    pub fn init(args: []const [:0]const u8) ArgParser {
        return .{ .args = args, .index = 0 };
    }

    pub fn next(self: *ArgParser) ?[]const u8 {
        if (self.index >= self.args.len) return null;
        const arg = self.args[self.index];
        self.index += 1;
        return arg;
    }

    pub fn peek(self: *const ArgParser) ?[]const u8 {
        if (self.index >= self.args.len) return null;
        return self.args[self.index];
    }

    pub fn peekFlag(self: *const ArgParser) ?[]const u8 {
        const arg = self.peek() orelse return null;
        if (arg.len > 0 and arg[0] == '-') return arg;
        return null;
    }

    pub fn remaining(self: *const ArgParser) []const [:0]const u8 {
        return self.args[self.index..];
    }

    pub fn hasMore(self: *const ArgParser) bool {
        return self.index < self.args.len;
    }

    pub fn expectArg(self: *ArgParser, name: []const u8) ![]const u8 {
        return self.next() orelse {
            std.debug.print("Error: Missing required argument: {s}\n", .{name});
            return error.MissingArgument;
        };
    }

    pub fn getOption(self: *ArgParser, short: []const u8, long: []const u8) ?[]const u8 {
        const arg = self.peek() orelse return null;
        if (mem.eql(u8, arg, short) or mem.eql(u8, arg, long)) {
            _ = self.next();
            return self.next();
        }
        return null;
    }

    pub fn hasFlag(self: *ArgParser, short: []const u8, long: []const u8) bool {
        const arg = self.peek() orelse return false;
        if (mem.eql(u8, arg, short) or mem.eql(u8, arg, long)) {
            _ = self.next();
            return true;
        }
        return false;
    }
};

// ============================================================================
// Config File Management (~/.nworkflow/config.json)
// ============================================================================

pub const Config = struct {
    server: ?[]const u8 = null,
    token: ?[]const u8 = null,
    format: ?[]const u8 = null,

    // Additional config options
    timeout_ms: u32 = 30000,
    verify_ssl: bool = true,

    pub fn getConfigPath(allocator: Allocator) ![]const u8 {
        const home = std.posix.getenv("HOME") orelse return error.NoHomeDir;
        return try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ home, CONFIG_DIR, CONFIG_FILE });
    }

    pub fn getConfigDir(allocator: Allocator) ![]const u8 {
        const home = std.posix.getenv("HOME") orelse return error.NoHomeDir;
        return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, CONFIG_DIR });
    }

    pub fn load(allocator: Allocator) !Config {
        const path = try getConfigPath(allocator);
        defer allocator.free(path);

        const file = fs.cwd().openFile(path, .{}) catch return Config{};
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 1024 * 1024);
        defer allocator.free(content);

        const parsed = try json.parseFromSlice(Config, allocator, content, .{});
        return parsed.value;
    }

    pub fn save(self: *const Config, allocator: Allocator) !void {
        const dir_path = try getConfigDir(allocator);
        defer allocator.free(dir_path);

        // Ensure directory exists
        fs.cwd().makeDir(dir_path) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        const path = try getConfigPath(allocator);
        defer allocator.free(path);

        const file = try fs.cwd().createFile(path, .{});
        defer file.close();

        // Serialize config to JSON
        var buf: [4096]u8 = undefined;
        var file_writer = file.writer(&buf);
        try json.Stringify.value(self.*, .{ .whitespace = .indent_2 }, &file_writer.interface);
        try file_writer.interface.flush();
    }

    pub fn get(self: *const Config, key: []const u8) ?[]const u8 {
        if (mem.eql(u8, key, "server")) return self.server;
        if (mem.eql(u8, key, "token")) return self.token;
        if (mem.eql(u8, key, "format")) return self.format;
        return null;
    }

    pub fn set(self: *Config, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const val_copy = try allocator.dupe(u8, value);
        if (mem.eql(u8, key, "server")) {
            self.server = val_copy;
        } else if (mem.eql(u8, key, "token")) {
            self.token = val_copy;
        } else if (mem.eql(u8, key, "format")) {
            self.format = val_copy;
        } else {
            return error.UnknownConfigKey;
        }
    }
};

// ============================================================================
// HTTP Client for API Calls
// ============================================================================

pub const HttpClient = struct {
    allocator: Allocator,
    base_url: []const u8,
    auth_token: ?[]const u8,

    pub fn init(allocator: Allocator, base_url: []const u8, token: ?[]const u8) HttpClient {
        return .{
            .allocator = allocator,
            .base_url = base_url,
            .auth_token = token,
        };
    }

    pub const HttpResponse = struct {
        status: u16,
        body: []const u8,
        allocator: Allocator,

        pub fn deinit(self: *HttpResponse) void {
            self.allocator.free(self.body);
        }
    };

    pub fn get(self: *HttpClient, path: []const u8) !HttpResponse {
        return self.request(.GET, path, null);
    }

    pub fn post(self: *HttpClient, path: []const u8, body: ?[]const u8) !HttpResponse {
        return self.request(.POST, path, body);
    }

    pub fn put(self: *HttpClient, path: []const u8, body: ?[]const u8) !HttpResponse {
        return self.request(.PUT, path, body);
    }

    pub fn delete(self: *HttpClient, path: []const u8) !HttpResponse {
        return self.request(.DELETE, path, null);
    }

    fn request(self: *HttpClient, method: std.http.Method, path: []const u8, body: ?[]const u8) !HttpResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, path });
        defer self.allocator.free(url);

        // Create HTTP client
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        // Build extra headers
        var extra_headers: [2]std.http.Header = undefined;
        var header_count: usize = 0;

        if (self.auth_token) |token| {
            const auth_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
            defer self.allocator.free(auth_value);
            extra_headers[header_count] = .{ .name = "Authorization", .value = auth_value };
            header_count += 1;
        }
        extra_headers[header_count] = .{ .name = "Content-Type", .value = "application/json" };
        header_count += 1;

        // Response body buffer using allocating writer
        var alloc_writer: std.Io.Writer.Allocating = .init(self.allocator);
        defer alloc_writer.deinit();

        // Use fetch API
        const result = try client.fetch(.{
            .location = .{ .url = url },
            .method = method,
            .extra_headers = extra_headers[0..header_count],
            .payload = body,
            .response_writer = &alloc_writer.writer,
        });

        return HttpResponse{
            .status = @intFromEnum(result.status),
            .body = try alloc_writer.toOwnedSlice(),
            .allocator = self.allocator,
        };
    }
};


// ============================================================================
// Table Formatter for Output
// ============================================================================

pub const TableFormatter = struct {
    allocator: Allocator,
    columns: std.ArrayList(Column) = .empty,
    rows: std.ArrayList(Row) = .empty,

    const Column = struct {
        name: []const u8,
        width: usize,
    };

    const Row = []const []const u8;

    pub fn init(allocator: Allocator) TableFormatter {
        return .{
            .allocator = allocator,
            .columns = .empty,
            .rows = .empty,
        };
    }

    pub fn deinit(self: *TableFormatter) void {
        for (self.rows.items) |row| {
            self.allocator.free(row);
        }
        self.rows.deinit(self.allocator);
        self.columns.deinit(self.allocator);
    }

    pub fn addColumn(self: *TableFormatter, name: []const u8) !void {
        try self.columns.append(self.allocator, .{ .name = name, .width = name.len });
    }

    pub fn addRow(self: *TableFormatter, values: []const []const u8) !void {
        const row = try self.allocator.dupe([]const u8, values);

        // Update column widths
        for (values, 0..) |val, i| {
            if (i < self.columns.items.len) {
                if (val.len > self.columns.items[i].width) {
                    self.columns.items[i].width = val.len;
                }
            }
        }

        try self.rows.append(self.allocator, row);
    }

    pub fn print(self: *const TableFormatter, writer: *std.Io.Writer) !void {
        // Print header separator
        try self.printSeparator(writer);

        // Print header
        try writer.writeAll("│");
        for (self.columns.items) |col| {
            try writer.print(" {s:<[1]} │", .{ col.name, col.width });
        }
        try writer.writeAll("\n");

        // Print header separator
        try self.printSeparator(writer);

        // Print rows
        for (self.rows.items) |row| {
            try writer.writeAll("│");
            for (row, 0..) |val, i| {
                if (i < self.columns.items.len) {
                    try writer.print(" {s:<[1]} │", .{ val, self.columns.items[i].width });
                }
            }
            try writer.writeAll("\n");
        }

        // Print footer separator
        try self.printSeparator(writer);
    }

    fn printSeparator(self: *const TableFormatter, writer: *std.Io.Writer) !void {
        try writer.writeAll("├");
        for (self.columns.items, 0..) |col, i| {
            var j: usize = 0;
            while (j < col.width + 2) : (j += 1) {
                try writer.writeByte(0xe2); // UTF-8 for ─
                try writer.writeByte(0x94);
                try writer.writeByte(0x80);
            }
            if (i < self.columns.items.len - 1) {
                try writer.writeAll("┼");
            }
        }
        try writer.writeAll("┤\n");
    }
};

// ============================================================================
// JSON Pretty Printer
// ============================================================================

pub fn prettyPrintJson(allocator: Allocator, input: []const u8, writer: *std.Io.Writer) !void {
    const parsed = json.parseFromSlice(json.Value, allocator, input, .{}) catch {
        // If not valid JSON, print as-is
        try writer.writeAll(input);
        return;
    };
    defer parsed.deinit();

    try json.Stringify.value(parsed.value, .{ .whitespace = .indent_2 }, writer);
    try writer.writeAll("\n");
}

// ============================================================================
// YAML Output (Simple Implementation)
// ============================================================================

fn writeIndent(writer: *std.Io.Writer, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try writer.writeByte(' ');
    }
}

pub fn jsonToYaml(allocator: Allocator, input: []const u8, writer: *std.Io.Writer) !void {
    const parsed = json.parseFromSlice(json.Value, allocator, input, .{}) catch {
        try writer.writeAll(input);
        return;
    };
    defer parsed.deinit();

    try writeYamlValue(parsed.value, writer, 0);
}

fn writeYamlValue(value: json.Value, writer: *std.Io.Writer, indent: usize) !void {
    switch (value) {
        .null => try writer.writeAll("null"),
        .bool => |b| try writer.print("{}", .{b}),
        .integer => |i| try writer.print("{}", .{i}),
        .float => |f| try writer.print("{d}", .{f}),
        .string => |s| try writer.print("\"{s}\"", .{s}),
        .array => |arr| {
            if (arr.items.len == 0) {
                try writer.writeAll("[]");
            } else {
                try writer.writeAll("\n");
                for (arr.items) |item| {
                    try writeIndent(writer, indent);
                    try writer.writeAll("- ");
                    try writeYamlValue(item, writer, indent + 2);
                    try writer.writeAll("\n");
                }
            }
        },
        .object => |obj| {
            if (obj.count() == 0) {
                try writer.writeAll("{}");
            } else {
                try writer.writeAll("\n");
                var iter = obj.iterator();
                while (iter.next()) |entry| {
                    try writeIndent(writer, indent);
                    try writer.print("{s}: ", .{entry.key_ptr.*});
                    try writeYamlValue(entry.value_ptr.*, writer, indent + 2);
                    try writer.writeAll("\n");
                }
            }
        },
        .number_string => |s| try writer.writeAll(s),
    }
}



// ============================================================================
// Command Handlers
// ============================================================================

pub const Commands = struct {
    // -------------------------------------------------------------------------
    // Workflow Commands
    // -------------------------------------------------------------------------

    pub fn workflowList(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        var page: []const u8 = "1";
        var limit: []const u8 = "20";
        var status: ?[]const u8 = null;

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--page")) {
                _ = args.next();
                page = args.next() orelse "1";
            } else if (mem.eql(u8, flag, "--limit")) {
                _ = args.next();
                limit = args.next() orelse "20";
            } else if (mem.eql(u8, flag, "--status")) {
                _ = args.next();
                status = args.next();
            } else break;
        }

        const path = if (status) |s|
            try std.fmt.allocPrint(allocator, "/api/v1/workflows?page={s}&limit={s}&status={s}", .{ page, limit, s })
        else
            try std.fmt.allocPrint(allocator, "/api/v1/workflows?page={s}&limit={s}", .{ page, limit });
        defer allocator.free(path);

        var response = try client.get(path);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, &[_][]const u8{ "id", "name", "status", "created_at" });
    }

    pub fn workflowGet(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        const id = try args.expectArg("workflow-id");
        const path = try std.fmt.allocPrint(allocator, "/api/v1/workflows/{s}", .{id});
        defer allocator.free(path);

        var response = try client.get(path);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
    }

    pub fn workflowCreate(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        const name = try args.expectArg("name");
        var file_path: ?[]const u8 = null;

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--file")) {
                _ = args.next();
                file_path = args.next();
            } else break;
        }

        const file = file_path orelse {
            std.debug.print("Error: --file <path.json> is required\n", .{});
            return error.MissingFile;
        };

        // Read file content
        const content = try readFile(allocator, file);
        defer allocator.free(content);

        // Create workflow with name
        const body = try std.fmt.allocPrint(allocator,
            \\{{"name": "{s}", "definition": {s}}}
        , .{ name, content });
        defer allocator.free(body);

        var response = try client.post("/api/v1/workflows", body);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
        std.debug.print("Workflow '{s}' created successfully.\n", .{name});
    }

    pub fn workflowUpdate(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        const id = try args.expectArg("workflow-id");
        var file_path: ?[]const u8 = null;

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--file")) {
                _ = args.next();
                file_path = args.next();
            } else break;
        }

        const file = file_path orelse {
            std.debug.print("Error: --file <path.json> is required\n", .{});
            return error.MissingFile;
        };

        const content = try readFile(allocator, file);
        defer allocator.free(content);

        const path = try std.fmt.allocPrint(allocator, "/api/v1/workflows/{s}", .{id});
        defer allocator.free(path);

        var response = try client.put(path, content);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
        std.debug.print("Workflow '{s}' updated successfully.\n", .{id});
    }

    pub fn workflowDelete(allocator: Allocator, client: *HttpClient, args: *ArgParser, _: OutputFormat) !void {
        const id = try args.expectArg("workflow-id");
        const path = try std.fmt.allocPrint(allocator, "/api/v1/workflows/{s}", .{id});
        defer allocator.free(path);

        var response = try client.delete(path);
        defer response.deinit();

        if (response.status == 200 or response.status == 204) {
            std.debug.print("Workflow '{s}' deleted successfully.\n", .{id});
        } else {
            std.debug.print("Error deleting workflow: {s}\n", .{response.body});
        }
    }

    pub fn workflowExport(allocator: Allocator, client: *HttpClient, args: *ArgParser, _: OutputFormat) !void {
        const id = try args.expectArg("workflow-id");
        var output_path: ?[]const u8 = null;

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--output")) {
                _ = args.next();
                output_path = args.next();
            } else break;
        }

        const path = try std.fmt.allocPrint(allocator, "/api/v1/workflows/{s}/export", .{id});
        defer allocator.free(path);

        var response = try client.get(path);
        defer response.deinit();

        if (output_path) |out| {
            const file = try fs.cwd().createFile(out, .{});
            defer file.close();
            try file.writeAll(response.body);
            std.debug.print("Workflow exported to: {s}\n", .{out});
        } else {
            var stdout_buf: [4096]u8 = undefined;
            var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
            defer stdout_writer.interface.flush() catch {};
            try prettyPrintJson(allocator, response.body, &stdout_writer.interface);
        }
    }

    pub fn workflowImport(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        const file_path = try args.expectArg("path.json");
        const content = try readFile(allocator, file_path);
        defer allocator.free(content);

        var response = try client.post("/api/v1/workflows/import", content);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
        std.debug.print("Workflow imported successfully.\n", .{});
    }

    // -------------------------------------------------------------------------
    // Execute Commands
    // -------------------------------------------------------------------------

    pub fn execute(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        const workflow_id = try args.expectArg("workflow-id");
        var input_json: ?[]const u8 = null;
        var wait = false;

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--input")) {
                _ = args.next();
                input_json = args.next();
            } else if (mem.eql(u8, flag, "--wait")) {
                _ = args.next();
                wait = true;
            } else break;
        }

        const body = try std.fmt.allocPrint(allocator,
            \\{{"workflow_id": "{s}", "input": {s}, "wait": {s}}}
        , .{ workflow_id, input_json orelse "{}", if (wait) "true" else "false" });
        defer allocator.free(body);

        var response = try client.post("/api/v1/executions", body);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
    }

    pub fn executionList(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        var workflow_id: ?[]const u8 = null;
        var status: ?[]const u8 = null;

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--workflow")) {
                _ = args.next();
                workflow_id = args.next();
            } else if (mem.eql(u8, flag, "--status")) {
                _ = args.next();
                status = args.next();
            } else break;
        }

        var path_buf: std.ArrayList(u8) = .empty;
        defer path_buf.deinit(allocator);
        try path_buf.appendSlice(allocator, "/api/v1/executions?");

        if (workflow_id) |wid| {
            try path_buf.appendSlice(allocator, "workflow_id=");
            try path_buf.appendSlice(allocator, wid);
            try path_buf.append(allocator, '&');
        }
        if (status) |s| {
            try path_buf.appendSlice(allocator, "status=");
            try path_buf.appendSlice(allocator, s);
            try path_buf.append(allocator, '&');
        }

        var response = try client.get(path_buf.items);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, &[_][]const u8{ "id", "workflow_id", "status", "started_at" });
    }

    pub fn executionGet(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        const id = try args.expectArg("execution-id");
        const path = try std.fmt.allocPrint(allocator, "/api/v1/executions/{s}", .{id});
        defer allocator.free(path);

        var response = try client.get(path);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
    }

    pub fn executionLogs(allocator: Allocator, client: *HttpClient, args: *ArgParser, _: OutputFormat) !void {
        const id = try args.expectArg("execution-id");
        const path = try std.fmt.allocPrint(allocator, "/api/v1/executions/{s}/logs", .{id});
        defer allocator.free(path);

        var response = try client.get(path);
        defer response.deinit();

        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
        defer stdout_writer.interface.flush() catch {};
        try stdout_writer.interface.writeAll(response.body);
        try stdout_writer.interface.writeAll("\n");
    }

    pub fn executionCancel(allocator: Allocator, client: *HttpClient, args: *ArgParser, _: OutputFormat) !void {
        const id = try args.expectArg("execution-id");
        const path = try std.fmt.allocPrint(allocator, "/api/v1/executions/{s}/cancel", .{id});
        defer allocator.free(path);

        var response = try client.post(path, null);
        defer response.deinit();

        if (response.status == 200) {
            std.debug.print("Execution '{s}' cancelled successfully.\n", .{id});
        } else {
            std.debug.print("Error cancelling execution: {s}\n", .{response.body});
        }
    }

    // -------------------------------------------------------------------------
    // Node Commands
    // -------------------------------------------------------------------------

    pub fn nodeList(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        var category: ?[]const u8 = null;

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--category")) {
                _ = args.next();
                category = args.next();
            } else break;
        }

        const path = if (category) |cat|
            try std.fmt.allocPrint(allocator, "/api/v1/node-types?category={s}", .{cat})
        else
            try allocator.dupe(u8, "/api/v1/node-types");
        defer allocator.free(path);

        var response = try client.get(path);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, &[_][]const u8{ "id", "name", "category", "description" });
    }

    pub fn nodeInfo(allocator: Allocator, client: *HttpClient, args: *ArgParser, format: OutputFormat) !void {
        const node_type = try args.expectArg("type");
        const path = try std.fmt.allocPrint(allocator, "/api/v1/node-types/{s}", .{node_type});
        defer allocator.free(path);

        var response = try client.get(path);
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
    }

    // -------------------------------------------------------------------------
    // Server Commands
    // -------------------------------------------------------------------------

    pub fn serverStart(args: *ArgParser) !void {
        var port: []const u8 = "8090";

        while (args.peekFlag()) |flag| {
            if (mem.eql(u8, flag, "--port")) {
                _ = args.next();
                port = args.next() orelse "8090";
            } else break;
        }

        std.debug.print("Starting nWorkflow server on port {s}...\n", .{port});
        std.debug.print("Press Ctrl+C to stop.\n\n", .{});

        // Execute the server binary
        var child = std.process.Child.init(&[_][]const u8{
            "./zig-out/bin/nworkflow-server",
            "--port",
            port,
        }, std.heap.page_allocator);

        child.spawn() catch |err| {
            std.debug.print("Error starting server: {any}\n", .{err});
            std.debug.print("Hint: Build the server first with 'zig build'\n", .{});
            return err;
        };

        _ = child.wait() catch |err| {
            std.debug.print("Error waiting for server: {any}\n", .{err});
            return err;
        };
    }

    pub fn serverStatus(allocator: Allocator, client: *HttpClient, format: OutputFormat) !void {
        var response = client.get("/api/v1/info") catch {
            std.debug.print("Server is not running or not reachable.\n", .{});
            return;
        };
        defer response.deinit();

        if (response.status == 200) {
            std.debug.print("Server is running.\n", .{});
            try outputResponse(allocator, response.body, format, null);
        } else {
            std.debug.print("Server returned status: {d}\n", .{response.status});
        }
    }

    pub fn serverHealth(allocator: Allocator, client: *HttpClient, format: OutputFormat) !void {
        var response = client.get("/api/health") catch {
            std.debug.print("Health check failed: Server not reachable.\n", .{});
            return;
        };
        defer response.deinit();

        try outputResponse(allocator, response.body, format, null);
    }

    // -------------------------------------------------------------------------
    // Config Commands
    // -------------------------------------------------------------------------

    pub fn configSet(allocator: Allocator, args: *ArgParser) !void {
        const key = try args.expectArg("key");
        const value = try args.expectArg("value");

        var config = Config.load(allocator) catch Config{};
        try config.set(allocator, key, value);
        try config.save(allocator);

        std.debug.print("Configuration '{s}' set to '{s}'\n", .{ key, value });
    }

    pub fn configGet(allocator: Allocator, args: *ArgParser) !void {
        const key = try args.expectArg("key");

        const config = Config.load(allocator) catch {
            std.debug.print("No configuration found.\n", .{});
            return;
        };

        if (config.get(key)) |value| {
            std.debug.print("{s}\n", .{value});
        } else {
            std.debug.print("Key '{s}' not found in configuration.\n", .{key});
        }
    }

    pub fn configList(allocator: Allocator) !void {
        const config = Config.load(allocator) catch {
            std.debug.print("No configuration found.\n", .{});
            return;
        };

        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
        defer stdout_writer.interface.flush() catch {};
        const stdout = &stdout_writer.interface;
        try stdout.writeAll("Current configuration:\n");
        try stdout.writeAll("─────────────────────────────\n");

        if (config.server) |s| try stdout.print("server: {s}\n", .{s});
        if (config.token) |t| try stdout.print("token: {s}\n", .{t});
        if (config.format) |f| try stdout.print("format: {s}\n", .{f});
        try stdout.print("timeout_ms: {d}\n", .{config.timeout_ms});
        try stdout.print("verify_ssl: {}\n", .{config.verify_ssl});
    }
};


// ============================================================================
// Helper Functions
// ============================================================================

fn readFile(allocator: Allocator, path: []const u8) ![]const u8 {
    const file = fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("Error opening file '{s}': {any}\n", .{ path, err });
        return err;
    };
    defer file.close();

    const size = try file.getEndPos();
    return try file.readToEndAlloc(allocator, size);
}

fn outputResponse(allocator: Allocator, body: []const u8, format: OutputFormat, table_columns: ?[]const []const u8) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    defer stdout_writer.interface.flush() catch {};
    const stdout = &stdout_writer.interface;

    switch (format) {
        .json => try prettyPrintJson(allocator, body, stdout),
        .yaml => try jsonToYaml(allocator, body, stdout),
        .table => {
            if (table_columns) |cols| {
                try outputAsTable(allocator, body, cols, stdout);
            } else {
                try prettyPrintJson(allocator, body, stdout);
            }
        },
    }
}

fn outputAsTable(allocator: Allocator, body: []const u8, columns: []const []const u8, writer: anytype) !void {
    const parsed = json.parseFromSlice(json.Value, allocator, body, .{}) catch {
        try writer.writeAll(body);
        return;
    };
    defer parsed.deinit();

    var table = TableFormatter.init(allocator);
    defer table.deinit();

    // Add columns
    for (columns) |col| {
        try table.addColumn(col);
    }

    // Find array in response (might be nested)
    const items = findArrayInJson(parsed.value) orelse {
        try prettyPrintJson(allocator, body, writer);
        return;
    };

    // Add rows
    for (items.array.items) |item| {
        if (item != .object) continue;

        var row_values: std.ArrayList([]const u8) = .empty;
        defer row_values.deinit(allocator);

        for (columns) |col| {
            const val = item.object.get(col);
            if (val) |v| {
                const str = switch (v) {
                    .string => |s| s,
                    .integer => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
                    .bool => |b| if (b) "true" else "false",
                    .null => "null",
                    else => "-",
                };
                try row_values.append(allocator, str);
            } else {
                try row_values.append(allocator, "-");
            }
        }

        try table.addRow(row_values.items);
    }

    try table.print(writer);
}

fn findArrayInJson(value: json.Value) ?json.Value {
    switch (value) {
        .array => return value,
        .object => |obj| {
            var iter = obj.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.* == .array) {
                    return entry.value_ptr.*;
                }
            }
        },
        else => {},
    }
    return null;
}


// ============================================================================
// Help Text
// ============================================================================

const HELP_TEXT =
    \\nWorkflow CLI - Command-line interface for workflow management
    \\
    \\USAGE:
    \\    nwf [OPTIONS] <COMMAND> [ARGS...]
    \\
    \\OPTIONS:
    \\    -s, --server <URL>      API server URL (default: http://localhost:8090)
    \\    -t, --token <TOKEN>     Authentication token
    \\    -f, --format <FORMAT>   Output format: json, table, yaml (default: table)
    \\    -h, --help              Show this help message
    \\    -v, --version           Show version information
    \\
    \\COMMANDS:
    \\    workflow                Manage workflows
    \\    execute                 Execute a workflow
    \\    execution               Manage workflow executions
    \\    node                    List and inspect node types
    \\    server                  Server management
    \\    config                  CLI configuration
    \\
    \\WORKFLOW COMMANDS:
    \\    workflow list [--page N] [--limit N] [--status STATUS]
    \\        List all workflows with optional filtering
    \\
    \\    workflow get <id>
    \\        Get details of a specific workflow
    \\
    \\    workflow create <name> --file <path.json>
    \\        Create a new workflow from a JSON definition file
    \\
    \\    workflow update <id> --file <path.json>
    \\        Update an existing workflow
    \\
    \\    workflow delete <id>
    \\        Delete a workflow
    \\
    \\    workflow export <id> [--output <path>]
    \\        Export workflow definition to file or stdout
    \\
    \\    workflow import <path.json>
    \\        Import a workflow from a JSON file
    \\
    \\EXECUTE COMMANDS:
    \\    execute <workflow-id> [--input <json>] [--wait]
    \\        Execute a workflow with optional input and wait for completion
    \\
    \\EXECUTION COMMANDS:
    \\    execution list [--workflow <id>] [--status STATUS]
    \\        List executions with optional filtering
    \\
    \\    execution get <id>
    \\        Get execution details
    \\
    \\    execution logs <id>
    \\        Show execution logs
    \\
    \\    execution cancel <id>
    \\        Cancel a running execution
    \\
    \\NODE COMMANDS:
    \\    node list [--category CATEGORY]
    \\        List available node types
    \\
    \\    node info <type>
    \\        Show detailed information about a node type
    \\
    \\SERVER COMMANDS:
    \\    server start [--port PORT]
    \\        Start the nWorkflow server
    \\
    \\    server status
    \\        Check server status
    \\
    \\    server health
    \\        Run health check
    \\
    \\CONFIG COMMANDS:
    \\    config set <key> <value>
    \\        Set a configuration value
    \\
    \\    config get <key>
    \\        Get a configuration value
    \\
    \\    config list
    \\        List all configuration values
    \\
    \\EXAMPLES:
    \\    # List all workflows
    \\    nwf workflow list
    \\
    \\    # Create a workflow
    \\    nwf workflow create my-workflow --file workflow.json
    \\
    \\    # Execute a workflow and wait
    \\    nwf execute abc123 --input '{"key": "value"}' --wait
    \\
    \\    # Check server health
    \\    nwf server health
    \\
    \\    # Configure default server
    \\    nwf config set server http://api.example.com:8090
    \\
    \\For more information, visit: https://github.com/nworkflow/nworkflow
    \\
;

const WORKFLOW_HELP =
    \\Workflow Commands:
    \\    list [--page N] [--limit N] [--status STATUS]  List workflows
    \\    get <id>                                        Get workflow details
    \\    create <name> --file <path.json>               Create workflow
    \\    update <id> --file <path.json>                 Update workflow
    \\    delete <id>                                     Delete workflow
    \\    export <id> [--output <path>]                  Export workflow
    \\    import <path.json>                              Import workflow
    \\
;

const EXECUTION_HELP =
    \\Execution Commands:
    \\    list [--workflow <id>] [--status STATUS]       List executions
    \\    get <id>                                        Get execution details
    \\    logs <id>                                       Show execution logs
    \\    cancel <id>                                     Cancel execution
    \\
;

const NODE_HELP =
    \\Node Commands:
    \\    list [--category CATEGORY]                     List node types
    \\    info <type>                                     Get node type info
    \\
;

const SERVER_HELP =
    \\Server Commands:
    \\    start [--port PORT]                            Start server
    \\    status                                          Check server status
    \\    health                                          Run health check
    \\
;

const CONFIG_HELP =
    \\Config Commands:
    \\    set <key> <value>                              Set config value
    \\    get <key>                                       Get config value
    \\    list                                            List all config
    \\
    \\Available keys: server, token, format
    \\
;


// ============================================================================
// Main Entry Point
// ============================================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Get command line arguments
    const args = try process.argsAlloc(allocator);
    defer process.argsFree(allocator, args);

    // Skip program name
    if (args.len < 2) {
        std.debug.print("{s}", .{HELP_TEXT});
        process.exit(EXIT_SUCCESS);
    }

    var parser = ArgParser.init(args[1..]);

    // Parse global options
    const global_opts = GlobalOptions.fromArgs(allocator, &parser) catch |err| {
        std.debug.print("Error parsing options: {any}\n", .{err});
        process.exit(EXIT_USAGE_ERROR);
    };

    // Handle global flags
    if (global_opts.version) {
        std.debug.print("nwf version {s}\n", .{VERSION});
        process.exit(EXIT_SUCCESS);
    }

    if (global_opts.help) {
        std.debug.print("{s}", .{HELP_TEXT});
        process.exit(EXIT_SUCCESS);
    }

    // Get command
    const command = parser.next() orelse {
        std.debug.print("{s}", .{HELP_TEXT});
        process.exit(EXIT_USAGE_ERROR);
    };

    // Initialize HTTP client
    var client = HttpClient.init(allocator, global_opts.server, global_opts.token);

    // Route to command handler
    runCommand(allocator, command, &parser, &client, global_opts.format) catch |err| {
        std.debug.print("Error: {any}\n", .{err});
        process.exit(EXIT_ERROR);
    };
}

fn runCommand(allocator: Allocator, command: []const u8, args: *ArgParser, client: *HttpClient, format: OutputFormat) !void {
    if (mem.eql(u8, command, "workflow")) {
        const subcommand = args.next() orelse {
            std.debug.print("{s}", .{WORKFLOW_HELP});
            return;
        };

        if (mem.eql(u8, subcommand, "list")) {
            try Commands.workflowList(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "get")) {
            try Commands.workflowGet(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "create")) {
            try Commands.workflowCreate(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "update")) {
            try Commands.workflowUpdate(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "delete")) {
            try Commands.workflowDelete(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "export")) {
            try Commands.workflowExport(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "import")) {
            try Commands.workflowImport(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "help") or mem.eql(u8, subcommand, "--help")) {
            std.debug.print("{s}", .{WORKFLOW_HELP});
        } else {
            std.debug.print("Unknown workflow command: {s}\n{s}", .{ subcommand, WORKFLOW_HELP });
        }
    } else if (mem.eql(u8, command, "execute")) {
        try Commands.execute(allocator, client, args, format);
    } else if (mem.eql(u8, command, "execution")) {
        const subcommand = args.next() orelse {
            std.debug.print("{s}", .{EXECUTION_HELP});
            return;
        };

        if (mem.eql(u8, subcommand, "list")) {
            try Commands.executionList(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "get")) {
            try Commands.executionGet(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "logs")) {
            try Commands.executionLogs(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "cancel")) {
            try Commands.executionCancel(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "help") or mem.eql(u8, subcommand, "--help")) {
            std.debug.print("{s}", .{EXECUTION_HELP});
        } else {
            std.debug.print("Unknown execution command: {s}\n{s}", .{ subcommand, EXECUTION_HELP });
        }
    } else if (mem.eql(u8, command, "node")) {
        const subcommand = args.next() orelse {
            std.debug.print("{s}", .{NODE_HELP});
            return;
        };

        if (mem.eql(u8, subcommand, "list")) {
            try Commands.nodeList(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "info")) {
            try Commands.nodeInfo(allocator, client, args, format);
        } else if (mem.eql(u8, subcommand, "help") or mem.eql(u8, subcommand, "--help")) {
            std.debug.print("{s}", .{NODE_HELP});
        } else {
            std.debug.print("Unknown node command: {s}\n{s}", .{ subcommand, NODE_HELP });
        }
    } else if (mem.eql(u8, command, "server")) {
        const subcommand = args.next() orelse {
            std.debug.print("{s}", .{SERVER_HELP});
            return;
        };

        if (mem.eql(u8, subcommand, "start")) {
            try Commands.serverStart(args);
        } else if (mem.eql(u8, subcommand, "status")) {
            try Commands.serverStatus(allocator, client, format);
        } else if (mem.eql(u8, subcommand, "health")) {
            try Commands.serverHealth(allocator, client, format);
        } else if (mem.eql(u8, subcommand, "help") or mem.eql(u8, subcommand, "--help")) {
            std.debug.print("{s}", .{SERVER_HELP});
        } else {
            std.debug.print("Unknown server command: {s}\n{s}", .{ subcommand, SERVER_HELP });
        }
    } else if (mem.eql(u8, command, "config")) {
        const subcommand = args.next() orelse {
            std.debug.print("{s}", .{CONFIG_HELP});
            return;
        };

        if (mem.eql(u8, subcommand, "set")) {
            try Commands.configSet(allocator, args);
        } else if (mem.eql(u8, subcommand, "get")) {
            try Commands.configGet(allocator, args);
        } else if (mem.eql(u8, subcommand, "list")) {
            try Commands.configList(allocator);
        } else if (mem.eql(u8, subcommand, "help") or mem.eql(u8, subcommand, "--help")) {
            std.debug.print("{s}", .{CONFIG_HELP});
        } else {
            std.debug.print("Unknown config command: {s}\n{s}", .{ subcommand, CONFIG_HELP });
        }
    } else if (mem.eql(u8, command, "help") or mem.eql(u8, command, "--help")) {
        std.debug.print("{s}", .{HELP_TEXT});
    } else {
        std.debug.print("Unknown command: {s}\n\n{s}", .{ command, HELP_TEXT });
        process.exit(EXIT_USAGE_ERROR);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "ArgParser basic" {
    const testing = std.testing;

    const args = [_][:0]const u8{ "--server", "http://test:8080", "workflow", "list" };
    var parser = ArgParser.init(&args);

    try testing.expectEqualStrings("--server", parser.next().?);
    try testing.expectEqualStrings("http://test:8080", parser.next().?);
    try testing.expectEqualStrings("workflow", parser.next().?);
    try testing.expectEqualStrings("list", parser.next().?);
    try testing.expect(parser.next() == null);
}

test "OutputFormat fromString" {
    const testing = std.testing;

    try testing.expect(OutputFormat.fromString("json") == .json);
    try testing.expect(OutputFormat.fromString("table") == .table);
    try testing.expect(OutputFormat.fromString("yaml") == .yaml);
    try testing.expect(OutputFormat.fromString("invalid") == null);
}

test "TableFormatter" {
    const testing = std.testing;

    var table = TableFormatter.init(testing.allocator);
    defer table.deinit();

    try table.addColumn("ID");
    try table.addColumn("Name");
    try table.addColumn("Status");

    try table.addRow(&[_][]const u8{ "1", "Test Workflow", "active" });
    try table.addRow(&[_][]const u8{ "2", "Another One", "paused" });

    try testing.expect(table.rows.items.len == 2);
    try testing.expect(table.columns.items.len == 3);
}