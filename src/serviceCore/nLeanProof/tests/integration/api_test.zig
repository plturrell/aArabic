const std = @import("std");
const json = std.json;
const mem = std.mem;

const TestCase = struct {
    name: []const u8,
    endpoint: []const u8,
    payload: []const u8,
    expected_status: u16,
    expected_contains: ?[]const u8,
};

const test_cases = [_]TestCase{
    // OpenAI-compatible endpoints
    .{
        .name = "health_check",
        .endpoint = "/health",
        .payload = "",
        .expected_status = 200,
        .expected_contains = "ok",
    },
    .{
        .name = "chat_completions",
        .endpoint = "/v1/chat/completions",
        .payload = 
        \\{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":32}
        ,
        .expected_status = 200,
        .expected_contains = "choices",
    },
    .{
        .name = "completions",
        .endpoint = "/v1/completions",
        .payload = 
        \\{"model":"test","prompt":"hello","max_tokens":32}
        ,
        .expected_status = 200,
        .expected_contains = "choices",
    },
    .{
        .name = "embeddings",
        .endpoint = "/v1/embeddings",
        .payload = 
        \\{"model":"test","input":"hello world"}
        ,
        .expected_status = 200,
        .expected_contains = "embedding",
    },
    // Lean4-specific endpoints
    .{
        .name = "lean4_check",
        .endpoint = "/v1/lean4/check",
        .payload =
        \\{"source":"#check Nat"}
        ,
        .expected_status = 200,
        .expected_contains = "errors",
    },
    .{
        .name = "lean4_run",
        .endpoint = "/v1/lean4/run",
        .payload =
        \\{"source":"#eval 1 + 1"}
        ,
        .expected_status = 200,
        .expected_contains = "stdout",
    },
    .{
        .name = "lean4_elaborate",
        .endpoint = "/v1/lean4/elaborate",
        .payload =
        \\{"source":"def foo := 42"}
        ,
        .expected_status = 200,
        .expected_contains = "declarations",
    },
};

fn runTest(allocator: std.mem.Allocator, base_url: []const u8, tc: TestCase) !bool {
    const url = try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_url, tc.endpoint });
    defer allocator.free(url);

    const uri = try std.Uri.parse(url);

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var response_body: std.io.Writer.Allocating = .init(allocator);
    defer response_body.deinit();

    const method: std.http.Method = if (tc.payload.len > 0) .POST else .GET;
    const payload: ?[]const u8 = if (tc.payload.len > 0) tc.payload else null;

    const result = client.fetch(.{
        .location = .{ .uri = uri },
        .method = method,
        .payload = payload,
        .response_writer = &response_body.writer,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }) catch |err| {
        std.debug.print("  FAIL: {s} - connection error: {}\n", .{ tc.name, err });
        return false;
    };

    const status: u16 = @intFromEnum(result.status);
    if (status != tc.expected_status) {
        std.debug.print("  FAIL: {s} - expected status {d}, got {d}\n", .{ tc.name, tc.expected_status, status });
        return false;
    }

    if (tc.expected_contains) |expected| {
        const body = response_body.written();
        if (mem.indexOf(u8, body, expected) == null) {
            std.debug.print("  FAIL: {s} - response missing '{s}'\n", .{ tc.name, expected });
            return false;
        }
    }

    std.debug.print("  PASS: {s}\n", .{tc.name});
    return true;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var port: u16 = 18001;
    if (args.len > 1) {
        port = std.fmt.parseInt(u16, args[1], 10) catch 18001;
    }

    const base_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}", .{port});
    defer allocator.free(base_url);

    std.debug.print("Running integration tests against {s}\n\n", .{base_url});

    var passed: usize = 0;
    var failed: usize = 0;

    for (test_cases) |tc| {
        if (runTest(allocator, base_url, tc) catch false) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    std.debug.print("\nResults: {d} passed, {d} failed\n", .{ passed, failed });

    if (failed > 0) {
        std.process.exit(1);
    }
}
