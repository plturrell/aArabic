// nCode HTTP Server v2 - Code Intelligence API with Logging & Metrics
// Port: 18003 (configurable via NCODE_PORT env var)

const std = @import("std");
const net = std.net;
const mem = std.mem;
const logging = @import("logging.zig");
const metrics = @import("metrics.zig");

const DEFAULT_PORT: u16 = 18003;

fn getPort() u16 {
    const port_str = std.posix.getenv("NCODE_PORT") orelse return DEFAULT_PORT;
    return std.fmt.parseInt(u16, port_str, 10) catch DEFAULT_PORT;
}

const ScipIndex = struct {
    loaded: bool = false,
    path: ?[]const u8 = null,
    symbol_count: u64 = 0,

    pub fn load(self: *ScipIndex, allocator: mem.Allocator, index_path: []const u8) !void {
        self.path = try allocator.dupe(u8, index_path);
        self.loaded = true;
        self.symbol_count = 1000; // Placeholder - would parse actual SCIP file
    }

    pub fn deinit(self: *ScipIndex, allocator: mem.Allocator) void {
        if (self.path) |p| allocator.free(p);
        self.path = null;
        self.loaded = false;
        self.symbol_count = 0;
    }
};

var scip_index: ScipIndex = .{};
var global_logger: ?logging.Logger = null;
var global_metrics: ?*metrics.Metrics = null;

fn jsonOk(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    return try std.fmt.allocPrint(allocator,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
            "Access-Control-Allow-Headers: Content-Type\r\nServer: nCode/2.0.0\r\n\r\n{s}",
        .{ body.len, body },
    );
}

fn textOk(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    return try std.fmt.allocPrint(allocator,
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {d}\r\n" ++
            "Server: nCode/2.0.0\r\n\r\n{s}",
        .{ body.len, body },
    );
}

fn jsonError(allocator: mem.Allocator, code: u16, message: []const u8) ![]const u8 {
    const status = switch (code) {
        400 => "400 Bad Request",
        404 => "404 Not Found",
        else => "500 Internal Server Error",
    };
    const body = try std.fmt.allocPrint(allocator, "{{\"error\":{{\"code\":{d},\"message\":\"{s}\"}}}}", .{ code, message });
    defer allocator.free(body);
    return try std.fmt.allocPrint(allocator,
        "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\nServer: nCode/2.0.0\r\n\r\n{s}",
        .{ status, body.len, body },
    );
}

const ParsedRequest = struct { method: []const u8, path: []const u8, body: []const u8 };

fn parseRequest(data: []const u8) !ParsedRequest {
    var lines = mem.splitSequence(u8, data, "\r\n");
    const first_line = lines.next() orelse return error.InvalidRequest;
    var parts = mem.splitSequence(u8, first_line, " ");
    const method = parts.next() orelse return error.InvalidMethod;
    const path = parts.next() orelse return error.InvalidPath;
    const body = if (mem.indexOf(u8, data, "\r\n\r\n")) |idx| data[idx + 4 ..] else &[_]u8{};
    return .{ .method = method, .path = path, .body = body };
}

fn extractJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    const search_key = std.fmt.allocPrint(std.heap.page_allocator, "\"{s}\":", .{key}) catch return null;
    defer std.heap.page_allocator.free(search_key);
    const key_start = mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_start + search_key.len;
    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}
    if (pos >= json.len or json[pos] != '"') return null;
    pos += 1;
    const str_start = pos;
    while (pos < json.len and json[pos] != '"') : (pos += 1) {}
    return json[str_start..pos];
}

fn extractJsonInt(json: []const u8, key: []const u8) ?i64 {
    const search_key = std.fmt.allocPrint(std.heap.page_allocator, "\"{s}\":", .{key}) catch return null;
    defer std.heap.page_allocator.free(search_key);
    const key_start = mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_start + search_key.len;
    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}
    const num_start = pos;
    while (pos < json.len and (json[pos] >= '0' and json[pos] <= '9')) : (pos += 1) {}
    if (num_start == pos) return null;
    return std.fmt.parseInt(i64, json[num_start..pos], 10) catch null;
}

fn handleHealth(allocator: mem.Allocator) ![]const u8 {
    if (global_metrics) |m| {
        const body = try std.fmt.allocPrint(allocator,
            "{{\"status\":\"ok\",\"version\":\"2.0.0\",\"index_loaded\":{s},\"uptime_seconds\":{d}}}",
            .{ if (scip_index.loaded) "true" else "false", m.getUptime() });
        defer allocator.free(body);
        return try jsonOk(allocator, body);
    }
    const body = "{{\"status\":\"ok\",\"version\":\"2.0.0\"}}";
    return try jsonOk(allocator, body);
}

fn handleMetrics(allocator: mem.Allocator) ![]const u8 {
    if (global_metrics) |m| {
        const prometheus_body = try m.formatPrometheus(allocator);
        defer allocator.free(prometheus_body);
        return try textOk(allocator, prometheus_body);
    }
    return try textOk(allocator, "# No metrics available\n");
}

fn handleMetricsJson(allocator: mem.Allocator) ![]const u8 {
    if (global_metrics) |m| {
        const json_body = try m.formatJson(allocator);
        defer allocator.free(json_body);
        return try jsonOk(allocator, json_body);
    }
    return try jsonOk(allocator, "{{\"error\":\"No metrics available\"}}");
}

fn handleIndexLoad(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    const path = extractJsonString(body, "path") orelse return try jsonError(allocator, 400, "Missing 'path' field");
    
    if (global_logger) |logger| {
        const ctx = try std.fmt.allocPrint(allocator, "{{\"path\":\"{s}\"}}", .{path});
        defer allocator.free(ctx);
        logger.info("Loading SCIP index", ctx);
    }
    
    scip_index.load(allocator, path) catch |err| {
        if (global_logger) |logger| {
            const ctx = try std.fmt.allocPrint(allocator, "{{\"error\":\"{any}\",\"path\":\"{s}\"}}", .{ err, path });
            defer allocator.free(ctx);
            logger.err("Failed to load index", ctx);
        }
        const msg = try std.fmt.allocPrint(allocator, "Failed to load index: {any}", .{err});
        defer allocator.free(msg);
        return try jsonError(allocator, 500, msg);
    };
    
    if (global_metrics) |m| {
        m.recordIndexLoaded(scip_index.symbol_count);
    }
    
    const resp = try std.fmt.allocPrint(allocator, "{{\"success\":true,\"path\":\"{s}\",\"symbols\":{d}}}", .{ path, scip_index.symbol_count });
    defer allocator.free(resp);
    return try jsonOk(allocator, resp);
}

fn handleDefinition(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    _ = extractJsonString(body, "file") orelse return try jsonError(allocator, 400, "Missing 'file' field");
    const line = extractJsonInt(body, "line") orelse return try jsonError(allocator, 400, "Missing 'line' field");
    const char = extractJsonInt(body, "character") orelse return try jsonError(allocator, 400, "Missing 'character' field");
    
    // Simulate cache hit for demonstration
    if (global_metrics) |m| {
        m.recordCacheHit();
    }
    
    const resp = try std.fmt.allocPrint(allocator,
        "{{\"locations\":[{{\"file\":\"example.zig\",\"range\":{{\"start\":{{\"line\":{d},\"character\":{d}}},\"end\":{{\"line\":{d},\"character\":{d}}}}}}}]}}",
        .{ line, char, line, char + 10 });
    defer allocator.free(resp);
    return try jsonOk(allocator, resp);
}

fn handleReferences(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    if (extractJsonString(body, "symbol")) |symbol| {
        const resp = try std.fmt.allocPrint(allocator,
            "{{\"locations\":[{{\"file\":\"example.zig\",\"range\":{{\"start\":{{\"line\":1,\"character\":0}},\"end\":{{\"line\":1,\"character\":10}}}}}}],\"symbol\":\"{s}\"}}",
            .{symbol});
        defer allocator.free(resp);
        return try jsonOk(allocator, resp);
    }
    _ = extractJsonString(body, "file") orelse return try jsonError(allocator, 400, "Missing 'file' or 'symbol' field");
    _ = extractJsonInt(body, "line") orelse return try jsonError(allocator, 400, "Missing 'line' field");
    _ = extractJsonInt(body, "character") orelse return try jsonError(allocator, 400, "Missing 'character' field");
    return try jsonOk(allocator, "{\"locations\":[]}");
}

fn handleHover(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    _ = extractJsonString(body, "file") orelse return try jsonError(allocator, 400, "Missing 'file' field");
    const line = extractJsonInt(body, "line") orelse return try jsonError(allocator, 400, "Missing 'line' field");
    const char = extractJsonInt(body, "character") orelse return try jsonError(allocator, 400, "Missing 'character' field");
    const resp = try std.fmt.allocPrint(allocator,
        "{{\"contents\":\"Symbol documentation (stub)\",\"range\":{{\"start\":{{\"line\":{d},\"character\":{d}}},\"end\":{{\"line\":{d},\"character\":{d}}}}}}}",
        .{ line, char, line, char + 5 });
    defer allocator.free(resp);
    return try jsonOk(allocator, resp);
}

fn handleSymbols(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    _ = extractJsonString(body, "file") orelse return try jsonError(allocator, 400, "Missing 'file' field");
    return try jsonOk(allocator, "{\"symbols\":[{\"name\":\"main\",\"kind\":\"function\",\"range\":{\"start\":{\"line\":0,\"character\":0},\"end\":{\"line\":10,\"character\":0}}}]}");
}

fn handleDocumentSymbols(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    _ = extractJsonString(body, "file") orelse return try jsonError(allocator, 400, "Missing 'file' field");
    return try jsonOk(allocator, "{\"symbols\":[{\"name\":\"Module\",\"kind\":\"module\",\"range\":{\"start\":{\"line\":0,\"character\":0},\"end\":{\"line\":100,\"character\":0}},\"children\":[{\"name\":\"main\",\"kind\":\"function\",\"range\":{\"start\":{\"line\":10,\"character\":0},\"end\":{\"line\":20,\"character\":0}}}]}]}");
}

fn handleExportQdrant(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    if (global_metrics) |m| {
        m.recordDbOperation();
    }
    const host = extractJsonString(body, "host") orelse "localhost";
    const port = extractJsonInt(body, "port") orelse 6333;
    const collection = extractJsonString(body, "collection") orelse "code_symbols";
    const resp = try std.fmt.allocPrint(allocator,
        "{{\"status\":\"pending\",\"message\":\"Use Python loader for Qdrant export\"," ++
        "\"command\":\"python scripts/load_to_databases.py {s} --qdrant --qdrant-host {s} --qdrant-port {d} --qdrant-collection {s}\"," ++
        "\"host\":\"{s}\",\"port\":{d},\"collection\":\"{s}\"}}",
        .{ scip_index.path orelse "index.scip", host, port, collection, host, port, collection });
    defer allocator.free(resp);
    return try jsonOk(allocator, resp);
}

fn handleExportMemgraph(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    if (global_metrics) |m| {
        m.recordDbOperation();
    }
    const host = extractJsonString(body, "host") orelse "localhost";
    const port = extractJsonInt(body, "port") orelse 7687;
    const resp = try std.fmt.allocPrint(allocator,
        "{{\"status\":\"pending\",\"message\":\"Use Python loader for Memgraph export\"," ++
        "\"command\":\"python scripts/load_to_databases.py {s} --memgraph --memgraph-host {s} --memgraph-port {d}\"," ++
        "\"host\":\"{s}\",\"port\":{d}}}",
        .{ scip_index.path orelse "index.scip", host, port, host, port });
    defer allocator.free(resp);
    return try jsonOk(allocator, resp);
}

fn handleExportMarquez(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    if (global_metrics) |m| {
        m.recordDbOperation();
    }
    const url = extractJsonString(body, "url") orelse "http://localhost:5000";
    const project = extractJsonString(body, "project") orelse "ncode-project";
    const resp = try std.fmt.allocPrint(allocator,
        "{{\"status\":\"pending\",\"message\":\"Use Python loader for Marquez export\"," ++
        "\"command\":\"python scripts/load_to_databases.py {s} --marquez --marquez-url {s} --project {s}\"," ++
        "\"url\":\"{s}\",\"project\":\"{s}\"}}",
        .{ scip_index.path orelse "index.scip", url, project, url, project });
    defer allocator.free(resp);
    return try jsonOk(allocator, resp);
}

fn handleExportAll(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    if (!scip_index.loaded) return try jsonError(allocator, 400, "No index loaded");
    _ = body;
    if (global_metrics) |m| {
        m.recordDbOperation();
    }
    const resp = try std.fmt.allocPrint(allocator,
        "{{\"status\":\"pending\",\"message\":\"Use Python loader for all exports\"," ++
        "\"command\":\"python scripts/load_to_databases.py {s} --all\"," ++
        "\"databases\":[\"qdrant\",\"memgraph\",\"marquez\"]}}",
        .{scip_index.path orelse "index.scip"});
    defer allocator.free(resp);
    return try jsonOk(allocator, resp);
}

fn routeRequest(allocator: mem.Allocator, method: []const u8, path: []const u8, body: []const u8) ![]const u8 {
    if (mem.eql(u8, method, "OPTIONS")) {
        return try std.fmt.allocPrint(allocator,
            "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n" ++
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n\r\n",
            .{});
    }
    if (mem.eql(u8, method, "GET")) {
        if (mem.eql(u8, path, "/health")) return try handleHealth(allocator);
        if (mem.eql(u8, path, "/metrics")) return try handleMetrics(allocator);
        if (mem.eql(u8, path, "/metrics.json")) return try handleMetricsJson(allocator);
    }
    if (mem.eql(u8, method, "POST")) {
        if (mem.eql(u8, path, "/v1/index/load")) return try handleIndexLoad(allocator, body);
        if (mem.eql(u8, path, "/v1/definition")) return try handleDefinition(allocator, body);
        if (mem.eql(u8, path, "/v1/references")) return try handleReferences(allocator, body);
        if (mem.eql(u8, path, "/v1/hover")) return try handleHover(allocator, body);
        if (mem.eql(u8, path, "/v1/symbols")) return try handleSymbols(allocator, body);
        if (mem.eql(u8, path, "/v1/document-symbols")) return try handleDocumentSymbols(allocator, body);
        if (mem.eql(u8, path, "/v1/index/export/qdrant")) return try handleExportQdrant(allocator, body);
        if (mem.eql(u8, path, "/v1/index/export/memgraph")) return try handleExportMemgraph(allocator, body);
        if (mem.eql(u8, path, "/v1/index/export/marquez")) return try handleExportMarquez(allocator, body);
        if (mem.eql(u8, path, "/v1/index/export/all")) return try handleExportAll(allocator, body);
    }
    return try jsonError(allocator, 404, "Not Found");
}

fn handleConnection(conn: net.Server.Connection, allocator: mem.Allocator) !void {
    defer conn.stream.close();

    const start_time = std.time.milliTimestamp();
    
    var buffer: [8192]u8 = undefined;
    const bytes_read = try conn.stream.read(&buffer);
    if (bytes_read == 0) return;

    const request_data = buffer[0..bytes_read];
    const req = parseRequest(request_data) catch return;

    var status: u16 = 200;
    const response = routeRequest(allocator, req.method, req.path, req.body) catch |err| {
        status = 500;
        if (global_logger) |logger| {
            const ctx = try std.fmt.allocPrint(allocator, "{{\"error\":\"{any}\",\"path\":\"{s}\"}}", .{ err, req.path });
            defer allocator.free(ctx);
            logger.err("Handler error", ctx);
        }
        const err_resp = jsonError(allocator, 500, "Internal Server Error") catch return;
        defer allocator.free(err_resp);
        _ = conn.stream.writeAll(err_resp) catch {};
        return;
    };
    defer allocator.free(response);

    // Extract status from response
    if (mem.indexOf(u8, response, "200 OK")) |_| {
        status = 200;
    } else if (mem.indexOf(u8, response, "400")) |_| {
        status = 400;
    } else if (mem.indexOf(u8, response, "404")) |_| {
        status = 404;
    } else if (mem.indexOf(u8, response, "500")) |_| {
        status = 500;
    }

    _ = try conn.stream.writeAll(response);
    
    const duration = std.time.milliTimestamp() - start_time;
    
    if (global_logger) |logger| {
        logger.logRequest(req.method, req.path, status, duration);
    }
    
    if (global_metrics) |m| {
        m.recordRequest(req.path, status, @intCast(duration));
    }
}

pub fn startServer() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize logger
    const logger = logging.Logger.init(allocator);
    global_logger = logger;
    logger.info("Initializing nCode server", null);

    // Initialize metrics
    var m = try metrics.Metrics.init(allocator);
    defer m.deinit();
    global_metrics = &m;
    logger.info("Metrics system initialized", null);

    const port = getPort();
    const addr = try net.Address.parseIp("0.0.0.0", port);

    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("🚀 nCode Server v2.0 Started - Code Intelligence API\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  • Address:   0.0.0.0:{d}\n", .{port});
    std.debug.print("  • Log Level: {s}\n", .{logger.level.toString()});
    std.debug.print("\n", .{});
    std.debug.print("Core Endpoints:\n", .{});
    std.debug.print("  • Health:           GET  http://localhost:{d}/health\n", .{port});
    std.debug.print("  • Metrics (Prom):   GET  http://localhost:{d}/metrics\n", .{port});
    std.debug.print("  • Metrics (JSON):   GET  http://localhost:{d}/metrics.json\n", .{port});
    std.debug.print("  • Load Index:       POST http://localhost:{d}/v1/index/load\n", .{port});
    std.debug.print("  • Definition:       POST http://localhost:{d}/v1/definition\n", .{port});
    std.debug.print("  • References:       POST http://localhost:{d}/v1/references\n", .{port});
    std.debug.print("  • Hover:            POST http://localhost:{d}/v1/hover\n", .{port});
    std.debug.print("  • Symbols:          POST http://localhost:{d}/v1/symbols\n", .{port});
    std.debug.print("  • Document Symbols: POST http://localhost:{d}/v1/document-symbols\n", .{port});
    std.debug.print("\n", .{});
    std.debug.print("Features:\n", .{});
    std.debug.print("  ✓ Structured JSON logging\n", .{});
    std.debug.print("  ✓ Prometheus metrics\n", .{});
    std.debug.print("  ✓ Request tracking\n", .{});
    std.debug.print("  ✓ Performance monitoring\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("✓ Server ready! Press Ctrl+C to stop.\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("\n", .{});

    logger.info("Server started successfully", null);

    while (true) {
        const conn = try server.accept();
        handleConnection(conn, allocator) catch |err| {
            const ctx = try std.fmt.allocPrint(allocator, "{{\"error\":\"{any}\"}}", .{err});
            defer allocator.free(ctx);
            logger.err("Connection error", ctx);
        };
    }
}

pub fn main() !void {
    try startServer();
}
