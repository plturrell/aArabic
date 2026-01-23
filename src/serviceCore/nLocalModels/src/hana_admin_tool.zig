const std = @import("std");
const HanaAgent = @import("orchestration/agents/hana_agent.zig");

fn usage() void {
    std.debug.print(
        "HANA Admin Tool (bridge-backed)\n" ++
        "Env: HANA_BRIDGE_URL (default http://localhost:3001/sql), HANA_SCHEMA (default DBADMIN)\n" ++
        "Commands:\n" ++
        "  list-prompts [limit] [offset]\n" ++
        "  create-prompt <prompt_text> <model> <user_id> [tags]\n" ++
        "  delete-prompt <id>\n" ++
        "  list-comparisons [limit] [offset]\n" ++
        "  create-comparison <prompt> <modelA> <modelB> <winner> <respA> <respB>\n" ++
        "  delete-comparison <id>\n",
        .{},
    );
}

fn getEnvOrDefault(allocator: std.mem.Allocator, key: []const u8, default_val: []const u8) ![]u8 {
    return std.process.getEnvVarOwned(allocator, key) catch try allocator.dupe(u8, default_val);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2) {
        usage();
        return;
    }

    const cmd = args[1];

    const bridge = try getEnvOrDefault(allocator, "HANA_BRIDGE_URL", "http://localhost:3001/sql");
    defer allocator.free(bridge);
    const schema = try getEnvOrDefault(allocator, "HANA_SCHEMA", "DBADMIN");
    defer allocator.free(schema);

    var agent = HanaAgent.init(allocator, .{
        .bridge_url = bridge,
        .schema = schema,
    });

    if (std.mem.eql(u8, cmd, "list-prompts")) {
        const limit = if (args.len > 2) std.fmt.parseInt(usize, args[2], 10) catch 50 else 50;
        const offset = if (args.len > 3) std.fmt.parseInt(usize, args[3], 10) catch 0 else 0;
        const data = try agent.listPrompts(limit, offset);
        defer allocator.free(data);
        try std.io.getStdOut().writeAll(data);
        return;
    }

    if (std.mem.eql(u8, cmd, "create-prompt")) {
        if (args.len < 5) {
            usage();
            return error.Invalid;
        }
        const prompt_text = args[2];
        const model = args[3];
        const user_id = args[4];
        const tags = if (args.len > 5) args[5] else "";
        try agent.createPrompt(prompt_text, model, user_id, 1, tags);
        std.debug.print("OK\n", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "delete-prompt")) {
        if (args.len < 3) {
            usage();
            return error.Invalid;
        }
        const id = std.fmt.parseInt(i32, args[2], 10) catch return error.Invalid;
        try agent.deletePrompt(id);
        std.debug.print("OK\n", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "list-comparisons")) {
        const limit = if (args.len > 2) std.fmt.parseInt(usize, args[2], 10) catch 50 else 50;
        const offset = if (args.len > 3) std.fmt.parseInt(usize, args[3], 10) catch 0 else 0;
        const data = try agent.listComparisons(limit, offset);
        defer allocator.free(data);
        try std.io.getStdOut().writeAll(data);
        return;
    }

    if (std.mem.eql(u8, cmd, "create-comparison")) {
        if (args.len < 8) {
            usage();
            return error.Invalid;
        }
        const prompt = args[2];
        const modelA = args[3];
        const modelB = args[4];
        const winner = args[5];
        const respA = args[6];
        const respB = args[7];
        try agent.createComparison(prompt, modelA, modelB, winner, respA, respB, 0, 0, 0, 0);
        std.debug.print("OK\n", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "delete-comparison")) {
        if (args.len < 3) {
            usage();
            return error.Invalid;
        }
        const id = std.fmt.parseInt(i32, args[2], 10) catch return error.Invalid;
        try agent.deleteComparison(id);
        std.debug.print("OK\n", .{});
        return;
    }

    usage();
}
