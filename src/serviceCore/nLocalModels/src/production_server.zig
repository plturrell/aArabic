// Production Zig Reverse Proxy + Static File Server
// Serves webapp files + proxies API to openai_http_server:11434
// Pure Zig solution - no NGINX/Caddy dependencies

const std = @import("std");
const net = std.net;
const fs = std.fs;
const mem = std.mem;
const json = std.json;
const os = std.os;
const time = std.time;

fn getMimeType(path: []const u8) []const u8 {
    if (mem.endsWith(u8, path, ".html")) return "text/html; charset=utf-8";
    if (mem.endsWith(u8, path, ".css")) return "text/css";
    if (mem.endsWith(u8, path, ".js")) return "application/javascript";
    if (mem.endsWith(u8, path, ".json")) return "application/json";
    if (mem.endsWith(u8, path, ".xml")) return "application/xml";
    if (mem.endsWith(u8, path, ".png")) return "image/png";
    if (mem.endsWith(u8, path, ".jpg")) return "image/jpeg";
    if (mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    if (mem.endsWith(u8, path, ".ico")) return "image/x-icon";
    if (mem.endsWith(u8, path, ".woff")) return "font/woff";
    if (mem.endsWith(u8, path, ".woff2")) return "font/woff2";
    return "application/octet-stream";
}

fn sendResponse(stream: net.Stream, status: u16, content_type: []const u8, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        else => "OK",
    };
    
    var header_buf: [1024]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf,
        "HTTP/1.1 {d} {s}\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS, HEAD\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{ status, reason, content_type, body.len },
    );
    
    _ = try stream.writeAll(header);
    _ = try stream.writeAll(body);
}

fn sendJson(stream: net.Stream, status: u16, body: []const u8) !void {
    try sendResponse(stream, status, "application/json", body);
}

fn sendHeadResponse(stream: net.Stream, status: u16, content_type: []const u8, content_length: usize) !void {
    const reason = switch (status) {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        else => "OK",
    };

    var header_buf: [1024]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf,
        "HTTP/1.1 {d} {s}\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS, HEAD\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{ status, reason, content_type, content_length },
    );

    _ = try stream.writeAll(header);
}

const HanaConfig = struct {
    bridge_url: []const u8,
    schema: []const u8,
};

fn loadHanaConfig(allocator: std.mem.Allocator) !HanaConfig {
    const bridge = std.process.getEnvVarOwned(allocator, "HANA_BRIDGE_URL") catch try allocator.dupe(u8, "http://localhost:3001/sql");
    errdefer allocator.free(bridge);
    const schema = std.process.getEnvVarOwned(allocator, "HANA_SCHEMA") catch try allocator.dupe(u8, "DBADMIN");
    errdefer allocator.free(schema);
    return HanaConfig{
        .bridge_url = bridge,
        .schema = schema,
    };
}

fn freeHanaConfig(allocator: std.mem.Allocator, cfg: HanaConfig) void {
    allocator.free(cfg.bridge_url);
    allocator.free(cfg.schema);
}

fn escapeSql(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);
    for (value) |ch| {
        if (ch == '\'') {
            try out.append(allocator, '\'');
            try out.append(allocator, '\'');
        } else {
            try out.append(allocator, ch);
        }
    }
    return out.toOwnedSlice(allocator);
}

fn valueAsString(value: json.Value) []const u8 {
    return switch (value) {
        .string => |s| s,
        .number_string => |s| s,
        .float => "",
        .integer => "",
        else => "",
    };
}

fn valueAsInt(value: ?json.Value, default_val: i32) i32 {
    if (value) |v| {
        return switch (v) {
            .integer => |i| @intCast(i),
            .float => |f| @intFromFloat(f),
            .number_string => |s| std.fmt.parseInt(i32, s, 10) catch default_val,
            else => default_val,
        };
    }
    return default_val;
}

fn extractArrayAndCount(raw: []const u8, allocator: std.mem.Allocator) !struct { arr: []u8, count: usize } {
    var tree = try json.parseFromSlice(json.Value, allocator, raw, .{});
    defer tree.deinit();
    const obj = switch (tree.value) {
        .object => |o| o,
        else => return error.Invalid,
    };
    const result_val = obj.get("result") orelse return error.Invalid;
    const arr_json = try std.fmt.allocPrint(allocator, "{f}", .{json.fmt(result_val, .{})});
    const count_val = obj.get("rowCount");
    const count = if (count_val) |c| @as(usize, @intCast(valueAsInt(c, 0))) else switch (result_val) {
        .array => |a| a.items.len,
        else => 0,
    };
    return .{ .arr = arr_json, .count = count };
}

fn jsonEscape(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{f}", .{json.fmt(json.Value{ .string = value }, .{})});
}

fn formatComparisons(raw: []const u8, allocator: std.mem.Allocator) !struct { body: []u8, count: usize } {
    var tree = try json.parseFromSlice(json.Value, allocator, raw, .{});
    defer tree.deinit();
    const obj = switch (tree.value) {
        .object => |o| o,
        else => return error.Invalid,
    };
    const result_val = obj.get("result") orelse return error.Invalid;
    const arr = switch (result_val) {
        .array => |a| a,
        else => return error.Invalid,
    };

    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);
    try out.append(allocator, '[');
    var first = true;

    for (arr.items) |item| {
        const row = switch (item) {
            .object => |o| o,
            else => continue,
        };

        const id_val = row.get("COMPARISON_ID");
        const prompt_val = row.get("PROMPT_TEXT") orelse json.Value{ .string = "" };
        const model_a_val = row.get("MODEL_A") orelse json.Value{ .string = "" };
        const model_b_val = row.get("MODEL_B") orelse json.Value{ .string = "" };
        const winner_val = row.get("WINNER") orelse json.Value{ .string = "" };
        const created_val = row.get("CREATED_AT") orelse json.Value{ .string = "" };

        const prompt_json = try jsonEscape(allocator, valueAsString(prompt_val));
        defer allocator.free(prompt_json);
        const model_a_json = try jsonEscape(allocator, valueAsString(model_a_val));
        defer allocator.free(model_a_json);
        const model_b_json = try jsonEscape(allocator, valueAsString(model_b_val));
        defer allocator.free(model_b_json);
        const winner_json = try jsonEscape(allocator, valueAsString(winner_val));
        defer allocator.free(winner_json);
        const ts_json = try jsonEscape(allocator, valueAsString(created_val));
        defer allocator.free(ts_json);

        if (!first) try out.append(allocator, ',');
        first = false;

        const entry = try std.fmt.allocPrint(allocator,
            "{{\"id\":{d},\"timestamp\":{s},\"prompt\":{s},\"modelA\":{{\"id\":{s},\"display_name\":{s},\"latency_ms\":{d},\"tokens_per_second\":{d}}},\"modelB\":{{\"id\":{s},\"display_name\":{s},\"latency_ms\":{d},\"tokens_per_second\":{d}}},\"winner\":{s}}}",
            .{
                valueAsInt(id_val, 0),
                ts_json,
                prompt_json,
                model_a_json,
                model_a_json,
                valueAsInt(row.get("LATENCY_A_MS"), 0),
                valueAsInt(row.get("TOKENS_PER_SECOND_A"), 0),
                model_b_json,
                model_b_json,
                valueAsInt(row.get("LATENCY_B_MS"), 0),
                valueAsInt(row.get("TOKENS_PER_SECOND_B"), 0),
                winner_json,
            },
        );
        defer allocator.free(entry);
        try out.appendSlice(allocator, entry);
    }

    try out.append(allocator, ']');
    const body = try out.toOwnedSlice(allocator);
    const default_len: i32 = @intCast(arr.items.len);
    const bridge_count = if (obj.get("rowCount")) |c| valueAsInt(c, default_len) else default_len;
    const count = @as(usize, @intCast(bridge_count));
    return .{ .body = body, .count = count };
}

fn hanaBridgeRequest(cfg: HanaConfig, allocator: std.mem.Allocator, sql: []const u8) ![]u8 {
    const payload = try std.fmt.allocPrint(allocator, "{{\"sql\":\"{s}\",\"schema\":\"{s}\"}}", .{ sql, cfg.schema });
    defer allocator.free(payload);

    const args = [_][]const u8{
        "curl",
        "-s",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-d",
        payload,
        cfg.bridge_url,
    };

    var child = std.process.Child.init(&args, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;
    try child.spawn();

    const stdout = try child.stdout.?.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(stdout);

    const result = try child.wait();
    if (result != .Exited or result.Exited != 0) {
        return error.ExecutionFailed;
    }

    return try allocator.dupe(u8, stdout);
}

fn hanaExec(cfg: HanaConfig, allocator: std.mem.Allocator, sql: []const u8) !void {
    _ = try hanaBridgeRequest(cfg, allocator, sql);
}

fn hanaQueryAlloc(cfg: HanaConfig, allocator: std.mem.Allocator, sql: []const u8) ![]u8 {
    return try hanaBridgeRequest(cfg, allocator, sql);
}

fn parsePromptInput(allocator: std.mem.Allocator, body: []const u8) !struct {
    text: []const u8,
    mode_id: i32,
    model_name: []const u8,
    user_id: []const u8,
    tags: []const u8,
} {
    var tree = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer tree.deinit();
    const obj = switch (tree.value) {
        .object => |o| o,
        else => return error.Invalid,
    };

    const prompt_text = obj.get("prompt_text") orelse return error.Invalid;
    const prompt_mode_id = obj.get("prompt_mode_id");
    const model_name = obj.get("model_name") orelse obj.get("model_id");
    const user_id = obj.get("user_id");
    const tags = obj.get("tags");

    const text_copied = try allocator.dupe(u8, valueAsString(prompt_text));
    const model_copied = try allocator.dupe(u8, valueAsString(model_name orelse json.Value{ .string = "" }));
    const user_copied = try allocator.dupe(u8, valueAsString(user_id orelse json.Value{ .string = "anonymous" }));
    const tags_copied = try allocator.dupe(u8, valueAsString(tags orelse json.Value{ .string = "" }));
    const mode_val: i32 = valueAsInt(prompt_mode_id, 1);

    return .{
        .text = text_copied,
        .mode_id = mode_val,
        .model_name = model_copied,
        .user_id = user_copied,
        .tags = tags_copied,
    };
}
fn ensureDataDir() void {
    _ = fs.cwd().makePath("data/dashboard") catch {};
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = fs.cwd().openFile(path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            return allocator.alloc(u8, 0);
        }
        return err;
    };
    defer file.close();

    const size = try file.getEndPos();
    var buf = try allocator.alloc(u8, size);
    const read_len = try file.readAll(buf);
    return buf[0..read_len];
}

fn stripOuterBraces(body: []const u8) []const u8 {
    if (body.len >= 2 and body[0] == '{' and body[body.len - 1] == '}') {
        return body[1 .. body.len - 1];
    }
    return body;
}

fn appendNdjson(allocator: std.mem.Allocator, path: []const u8, line: []const u8) !void {
    ensureDataDir();
    const existing = try readFileAlloc(allocator, path);
    defer allocator.free(existing);

    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);

    if (existing.len > 0) {
        try out.appendSlice(allocator, existing);
        if (existing[existing.len - 1] != '\n') try out.append(allocator, '\n');
    }
    try out.appendSlice(allocator, line);
    try out.append(allocator, '\n');

    const data = try out.toOwnedSlice(allocator);
    defer allocator.free(data);
    try fs.cwd().writeFile(.{ .sub_path = path, .data = data });
}

fn ndjsonToArrayJson(allocator: std.mem.Allocator, buf: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);

    try out.append(allocator, '[');
    var first: bool = true;
    var it = mem.splitAny(u8, buf, "\n");
    while (it.next()) |line| {
        if (line.len == 0) continue;
        if (!first) try out.append(allocator, ',');
        first = false;
        try out.appendSlice(allocator, line);
    }
    try out.append(allocator, ']');
    return out.toOwnedSlice(allocator);
}

fn filterNdjsonById(allocator: std.mem.Allocator, buf: []const u8, id: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);
    var it = mem.splitAny(u8, buf, "\n");
    while (it.next()) |line| {
        if (line.len == 0) continue;
        if (mem.indexOf(u8, line, id) != null) {
            continue;
        }
        try out.appendSlice(allocator, line);
        try out.append(allocator, '\n');
    }
    return out.toOwnedSlice(allocator);
}

fn proxyToOpenAI(client_stream: net.Stream, allocator: std.mem.Allocator, method: []const u8, path: []const u8, body: []const u8) !void {
    // Connect to openai_http_server on localhost:11434
    const api_addr = try net.Address.parseIp("127.0.0.1", 11434);
    const api_stream = net.tcpConnectToAddress(api_addr) catch |err| {
        std.debug.print("❌ Failed to connect to OpenAI API: {}\n", .{err});
        try sendResponse(client_stream, 502, "application/json", "{\"error\":\"API server unavailable\"}");
        return;
    };
    defer api_stream.close();
    
    // Build proxy request
    const proxy_request = try std.fmt.allocPrint(allocator,
        "{s} {s} HTTP/1.1\r\n" ++
        "Host: localhost:11434\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{ method, path, body.len, body },
    );
    defer allocator.free(proxy_request);
    
    // Send to OpenAI API
    _ = try api_stream.writeAll(proxy_request);
    
    // Read full response from API
    var response_list = std.ArrayList(u8).empty;
    defer response_list.deinit(allocator);
    
    var temp_buf: [4096]u8 = undefined;
    while (true) {
        const bytes_read = api_stream.read(&temp_buf) catch |err| {
            if (err == error.ConnectionResetByPeer or err == error.BrokenPipe) break;
            return err;
        };
        if (bytes_read == 0) break;
        try response_list.appendSlice(allocator, temp_buf[0..bytes_read]);
    }
    
    const full_response = response_list.items;
    
    if (full_response.len > 0) {
        // Forward complete response to client
        _ = try client_stream.writeAll(full_response);
        std.debug.print("✅ Proxied {s} {s} → {d} bytes\n", .{ method, path, full_response.len });
    }
}

fn serveStaticFile(stream: net.Stream, allocator: std.mem.Allocator, requested_path: []const u8, is_head: bool) !void {
    // Security: block directory traversal
    if (mem.indexOf(u8, requested_path, "..") != null) {
        try sendResponse(stream, 404, "text/plain", "Not Found");
        return;
    }

    // Favicon fallback to avoid noisy 404s in the browser console
    if (mem.eql(u8, requested_path, "/favicon.ico")) {
        const icon_svg =
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
            ++ "<rect width='64' height='64' fill='#0a3d62'/>"
            ++ "<text x='12' y='44' font-size='32' font-family='Arial' fill='#fff'>AI</text>"
            ++ "</svg>";
        if (is_head) {
            try sendHeadResponse(stream, 200, "image/svg+xml", icon_svg.len);
        } else {
            try sendResponse(stream, 200, "image/svg+xml", icon_svg);
        }
        return;
    }
    
    // Map URL path to file path
    var file_path: []const u8 = undefined;
    if (mem.eql(u8, requested_path, "/")) {
        file_path = "webapp/index.html";
    } else {
        const clean_path = if (mem.startsWith(u8, requested_path, "/")) 
            requested_path[1..] else requested_path;
        file_path = try std.fmt.allocPrint(allocator, "webapp/{s}", .{clean_path});
        defer allocator.free(file_path);
        
        return serveFile(stream, allocator, file_path, is_head);
    }
    
    try serveFile(stream, allocator, file_path, is_head);
}

fn serveFile(stream: net.Stream, allocator: std.mem.Allocator, file_path: []const u8, is_head: bool) !void {
    const file = fs.cwd().openFile(file_path, .{}) catch {
        try sendResponse(stream, 404, "text/plain", "File Not Found");
        return;
    };
    defer file.close();
    
    const file_size = try file.getEndPos();
    const content = try allocator.alloc(u8, file_size);
    defer allocator.free(content);
    
    _ = try file.readAll(content);
    
    const mime_type = getMimeType(file_path);
    if (is_head) {
        try sendHeadResponse(stream, 200, mime_type, content.len);
        return;
    }
    try sendResponse(stream, 200, mime_type, content);
}

fn handleConnection(stream: net.Stream, allocator: std.mem.Allocator) !void {
    defer stream.close();
    
    // Read HTTP request
    var buffer: [8192]u8 = undefined;
    const n = stream.read(&buffer) catch |err| {
        std.debug.print("❌ Read error: {}\n", .{err});
        return;
    };
    
    if (n == 0) return;
    
    const request = buffer[0..n];
    
    // Parse request line
    const first_line_end = mem.indexOf(u8, request, "\r\n") orelse return;
    const request_line = request[0..first_line_end];
    
    var parts = mem.splitSequence(u8, request_line, " ");
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;
    
    std.debug.print("📥 {s} {s}\n", .{ method, path });
    
    const is_head = mem.eql(u8, method, "HEAD");

    // Handle OPTIONS for CORS
    if (mem.eql(u8, method, "OPTIONS")) {
        try sendResponse(stream, 200, "text/plain", "");
        return;
    }
    
    // Extract body if present
    var body: []const u8 = "";
    if (mem.indexOf(u8, request, "\r\n\r\n")) |idx| {
        body = request[idx + 4 ..];
    }

    // Minimal local endpoints to keep UI live without mock data
    if (mem.startsWith(u8, path, "/api/v1/ab-testing/comparisons")) {
        const hana_cfg = loadHanaConfig(allocator) catch {
            try sendJson(stream, 500, "{\"error\":\"HANA config missing\"}");
            return;
        };
        defer freeHanaConfig(allocator, hana_cfg);

        if (mem.eql(u8, method, "GET") or is_head) {
            const sql = "SELECT COMPARISON_ID, PROMPT_TEXT, MODEL_A, MODEL_B, WINNER, RESPONSE_A, RESPONSE_B, LATENCY_A_MS, LATENCY_B_MS, TOKENS_PER_SECOND_A, TOKENS_PER_SECOND_B, CREATED_AT FROM PROMPT_COMPARISONS ORDER BY CREATED_AT DESC LIMIT 50";
            const raw = hanaQueryAlloc(hana_cfg, allocator, sql) catch |err| {
                std.debug.print("HANA query failed: {}\n", .{err});
                try sendJson(stream, 500, "{\"comparisons\":[]}");
                return;
            };
            defer allocator.free(raw);

            const formatted = formatComparisons(raw, allocator) catch |err| {
                std.debug.print("format comparisons failed: {}\n", .{err});
                try sendJson(stream, 500, "{\"comparisons\":[]}");
                return;
            };
            defer allocator.free(formatted.body);

            const resp = try std.fmt.allocPrint(allocator, "{{\"comparisons\":{s},\"total\":{d}}}", .{ formatted.body, formatted.count });
            defer allocator.free(resp);
            try sendJson(stream, 200, resp);
            return;
        }

        if (mem.eql(u8, method, "POST")) {
            // Store minimal fields from JSON
            var tree = json.parseFromSlice(json.Value, allocator, body, .{}) catch |err| {
                std.debug.print("parse error: {}\n", .{err});
                try sendJson(stream, 400, "{\"error\":\"invalid json\"}");
                return;
            };
            defer tree.deinit();
            const obj = switch (tree.value) {
                .object => |o| o,
                else => {
                    try sendJson(stream, 400, "{\"error\":\"invalid json\"}");
                    return;
                },
            };
            const prompt = obj.get("prompt") orelse json.Value{ .string = "" };
            const modelA = obj.get("modelA");
            const modelB = obj.get("modelB");
            const winner = obj.get("winner") orelse json.Value{ .string = "" };
            const respA = obj.get("responseA") orelse json.Value{ .string = "" };
            const respB = obj.get("responseB") orelse json.Value{ .string = "" };

            var modelA_name: []const u8 = "";
            var modelB_name: []const u8 = "";
            var latencyA: i32 = 0;
            var latencyB: i32 = 0;
            var tpsA: i32 = 0;
            var tpsB: i32 = 0;

            if (modelA) |ma| {
                switch (ma) {
                    .object => |o| {
                        modelA_name = valueAsString(o.get("display_name") orelse o.get("id") orelse json.Value{ .string = "" });
                        latencyA = valueAsInt(o.get("latency_ms"), 0);
                        tpsA = valueAsInt(o.get("tokens_per_second"), 0);
                    },
                    else => modelA_name = valueAsString(ma),
                }
            }

            if (modelB) |mb| {
                switch (mb) {
                    .object => |o| {
                        modelB_name = valueAsString(o.get("display_name") orelse o.get("id") orelse json.Value{ .string = "" });
                        latencyB = valueAsInt(o.get("latency_ms"), 0);
                        tpsB = valueAsInt(o.get("tokens_per_second"), 0);
                    },
                    else => modelB_name = valueAsString(mb),
                }
            }

            const esc_prompt = try escapeSql(allocator, valueAsString(prompt));
            defer allocator.free(esc_prompt);
            const esc_modelA = try escapeSql(allocator, modelA_name);
            defer allocator.free(esc_modelA);
            const esc_modelB = try escapeSql(allocator, modelB_name);
            defer allocator.free(esc_modelB);
            const esc_winner = try escapeSql(allocator, valueAsString(winner));
            defer allocator.free(esc_winner);
            const esc_respA = try escapeSql(allocator, valueAsString(respA));
            defer allocator.free(esc_respA);
            const esc_respB = try escapeSql(allocator, valueAsString(respB));
            defer allocator.free(esc_respB);

            const sql = try std.fmt.allocPrint(allocator,
                "INSERT INTO PROMPT_COMPARISONS (PROMPT_TEXT, MODEL_A, MODEL_B, WINNER, RESPONSE_A, RESPONSE_B, LATENCY_A_MS, LATENCY_B_MS, TOKENS_PER_SECOND_A, TOKENS_PER_SECOND_B, CREATED_AT) VALUES ('{s}','{s}','{s}','{s}','{s}','{s}', {d}, {d}, {d}, {d}, CURRENT_TIMESTAMP)",
                .{ esc_prompt, esc_modelA, esc_modelB, esc_winner, esc_respA, esc_respB, latencyA, latencyB, tpsA, tpsB });
            defer allocator.free(sql);

            hanaExec(hana_cfg, allocator, sql) catch |err| {
                std.debug.print("HANA insert failed: {}\n", .{err});
                try sendJson(stream, 500, "{\"error\":\"db insert failed\"}");
                return;
            };

            try sendJson(stream, 200, "{\"status\":\"ok\"}");
            return;
        }
    }

    if (mem.startsWith(u8, path, "/v1/prompts/history")) {
        const hana_cfg = loadHanaConfig(allocator) catch {
            try sendJson(stream, 500, "{\"error\":\"HANA config missing\"}");
            return;
        };
        defer freeHanaConfig(allocator, hana_cfg);

        const sql = "SELECT PROMPT_ID, PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, TAGS, CREATED_AT FROM PROMPTS ORDER BY CREATED_AT DESC LIMIT 50";
        const raw = hanaQueryAlloc(hana_cfg, allocator, sql) catch |err| {
            std.debug.print("HANA history query failed: {}\n", .{err});
            try sendJson(stream, 500, "{\"history\":[],\"total\":0}");
            return;
        };
        defer allocator.free(raw);

        const unwrapped = extractArrayAndCount(raw, allocator) catch |err| {
            std.debug.print("unwrap history failed: {}\n", .{err});
            try sendJson(stream, 500, "{\"history\":[],\"total\":0}");
            return;
        };
        defer allocator.free(unwrapped.arr);

        const resp = try std.fmt.allocPrint(allocator, "{{\"history\":{s},\"total\":{d}}}", .{ unwrapped.arr, unwrapped.count });
        defer allocator.free(resp);
        try sendJson(stream, 200, resp);
        return;
    }

    if (mem.startsWith(u8, path, "/api/v1/prompts/search")) {
        const hana_cfg = loadHanaConfig(allocator) catch {
            try sendJson(stream, 500, "{\"error\":\"HANA config missing\"}");
            return;
        };
        defer freeHanaConfig(allocator, hana_cfg);

        var search_term: []const u8 = "";
        if (mem.indexOf(u8, path, "q=")) |idx| {
            search_term = path[idx + 2 ..];
            if (mem.indexOf(u8, search_term, "&")) |amp_idx| {
                search_term = search_term[0..amp_idx];
            }
        }
        const escaped = try escapeSql(allocator, search_term);
        defer allocator.free(escaped);
        const sql = try std.fmt.allocPrint(allocator,
            "SELECT PROMPT_ID, PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, TAGS, CREATED_AT FROM PROMPTS WHERE LOWER(PROMPT_TEXT) LIKE LOWER('%{s}%') ORDER BY CREATED_AT DESC LIMIT 50",
            .{escaped});
        defer allocator.free(sql);

        const raw = hanaQueryAlloc(hana_cfg, allocator, sql) catch |err| {
            std.debug.print("HANA search failed: {}\n", .{err});
            try sendJson(stream, 500, "{\"results\":[],\"total\":0}");
            return;
        };
        defer allocator.free(raw);

        const unwrapped = extractArrayAndCount(raw, allocator) catch |err| {
            std.debug.print("unwrap search failed: {}\n", .{err});
            try sendJson(stream, 500, "{\"results\":[],\"total\":0}");
            return;
        };
        defer allocator.free(unwrapped.arr);

        const resp = try std.fmt.allocPrint(allocator, "{{\"results\":{s},\"total\":{d}}}", .{ unwrapped.arr, unwrapped.count });
        defer allocator.free(resp);
        try sendJson(stream, 200, resp);
        return;
    }

    if (mem.startsWith(u8, path, "/api/v1/prompts")) {
        const hana_cfg = loadHanaConfig(allocator) catch {
            try sendJson(stream, 500, "{\"error\":\"HANA config missing\"}");
            return;
        };
        defer freeHanaConfig(allocator, hana_cfg);

        if (mem.eql(u8, method, "DELETE")) {
            if (mem.indexOf(u8, path, "/api/v1/prompts/")) |idx| {
                const id_part = path[idx + "/api/v1/prompts/".len ..];
                const sql = try std.fmt.allocPrint(allocator, "DELETE FROM PROMPTS WHERE PROMPT_ID = {s}", .{id_part});
                defer allocator.free(sql);
                hanaExec(hana_cfg, allocator, sql) catch |err| {
                    std.debug.print("HANA delete failed: {}\n", .{err});
                    try sendJson(stream, 500, "{\"deleted\":false}");
                    return;
                };
            }
            try sendJson(stream, 200, "{\"deleted\":true}");
            return;
        }

        if (mem.eql(u8, method, "POST")) {
            const parsed = parsePromptInput(allocator, body) catch {
                try sendJson(stream, 400, "{\"error\":\"invalid prompt payload\"}");
                return;
            };
            defer allocator.free(parsed.text);
            defer allocator.free(parsed.model_name);
            defer allocator.free(parsed.user_id);
            defer allocator.free(parsed.tags);

            const esc_text = try escapeSql(allocator, parsed.text);
            defer allocator.free(esc_text);
            const esc_model = try escapeSql(allocator, parsed.model_name);
            defer allocator.free(esc_model);
            const esc_user = try escapeSql(allocator, parsed.user_id);
            defer allocator.free(esc_user);
            const esc_tags = try escapeSql(allocator, parsed.tags);
            defer allocator.free(esc_tags);

            const sql = try std.fmt.allocPrint(allocator,
                "INSERT INTO PROMPTS (PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, TAGS, CREATED_AT) VALUES ('{s}', {d}, '{s}', '{s}', '{s}', CURRENT_TIMESTAMP)",
                .{ esc_text, parsed.mode_id, esc_model, esc_user, esc_tags });
            defer allocator.free(sql);

            hanaExec(hana_cfg, allocator, sql) catch |err| {
                std.debug.print("HANA insert failed: {}\n", .{err});
                try sendJson(stream, 500, "{\"error\":\"db insert failed\"}");
                return;
            };

            const response = try std.fmt.allocPrint(allocator, "{{\"prompt_id\":\"{d}\",\"user_id\":\"{s}\"}}", .{ time.milliTimestamp(), parsed.user_id });
            defer allocator.free(response);
            try sendJson(stream, 200, response);
            return;
        }

        if (mem.eql(u8, method, "GET") or is_head) {
            const sql = "SELECT PROMPT_ID, PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, TAGS, CREATED_AT FROM PROMPTS ORDER BY CREATED_AT DESC LIMIT 50";
            const raw = hanaQueryAlloc(hana_cfg, allocator, sql) catch |err| {
                std.debug.print("HANA history query failed: {}\n", .{err});
                try sendJson(stream, 500, "{\"history\":[],\"total\":0}");
                return;
            };
            defer allocator.free(raw);

            const unwrapped = extractArrayAndCount(raw, allocator) catch |err| {
                std.debug.print("unwrap prompts failed: {}\n", .{err});
                try sendJson(stream, 500, "{\"history\":[],\"total\":0}");
                return;
            };
            defer allocator.free(unwrapped.arr);

            const resp = try std.fmt.allocPrint(allocator, "{{\"history\":{s},\"total\":{d}}}", .{ unwrapped.arr, unwrapped.count });
            defer allocator.free(resp);
            try sendJson(stream, 200, resp);
            return;
        }
    }
    
    // Route based on path
    // API proxy - forward to openai_http_server:11434
    // Strip /api prefix if present
    var proxy_path = path;
    if (mem.startsWith(u8, path, "/api/v1/")) {
        proxy_path = path[4..]; // Remove "/api" prefix
    }
    
    if (mem.startsWith(u8, path, "/api/v1/") or 
        mem.startsWith(u8, path, "/v1/") or
        mem.eql(u8, path, "/health") or
        mem.eql(u8, path, "/metrics")) {
        try proxyToOpenAI(stream, allocator, method, proxy_path, body);
        return;
    }
    
    // WebSocket upgrade (proxy to OpenAI server)
    if (mem.eql(u8, path, "/ws")) {
        if (mem.indexOf(u8, request, "Upgrade:") != null) {
            try proxyToOpenAI(stream, allocator, method, path, body);
            return;
        }
    }
    
    // Serve static files for everything else
    if (mem.eql(u8, method, "GET") or is_head) {
        try serveStaticFile(stream, allocator, path, is_head);
        return;
    }
    
    try sendResponse(stream, 404, "text/plain", "Not Found");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const address = try net.Address.parseIp("0.0.0.0", 8080);
    var server = try address.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();
    
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("🚀 Production Zig Server - Static Files + API Proxy\n", .{});
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("🌐 Frontend: http://localhost:8080\n", .{});
    std.debug.print("📁 Static: webapp/*\n", .{});
    std.debug.print("🔌 Proxy: /api/v1/* → localhost:11434\n", .{});
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("✅ Ready! Listening on port 8080...\n\n", .{});
    
    while (true) {
        const connection = server.accept() catch |err| {
            std.debug.print("❌ Accept error: {}\n", .{err});
            continue;
        };
        
        handleConnection(connection.stream, allocator) catch |err| {
            std.debug.print("❌ Handler error: {}\n", .{err});
        };
    }
}
