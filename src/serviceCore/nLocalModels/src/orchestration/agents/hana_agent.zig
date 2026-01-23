const std = @import("std");
const HanaAgent = @This();

pub const Config = struct {
    bridge_url: []const u8 = "http://localhost:3001/sql",
    schema: []const u8 = "DBADMIN",
};

allocator: std.mem.Allocator,
config: Config,

pub fn init(allocator: std.mem.Allocator, config: Config) HanaAgent {
    return .{ .allocator = allocator, .config = config };
}

fn runSql(self: *HanaAgent, sql: []const u8) ![]u8 {
    const payload = try std.fmt.allocPrint(self.allocator, "{{\"sql\":\"{s}\",\"schema\":\"{s}\"}}", .{ sql, self.config.schema });
    defer self.allocator.free(payload);

    const args = [_][]const u8{
        "curl",
        "-s",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-d",
        payload,
        self.config.bridge_url,
    };

    var child = std.process.Child.init(&args, self.allocator);
    child.stdout_behavior = .Pipe;
    try child.spawn();

    const out = try child.stdout.?.readToEndAlloc(self.allocator, 1024 * 1024);
    defer self.allocator.free(out);
    _ = try child.wait();

    return try self.allocator.dupe(u8, out);
}

fn escape(a: std.mem.Allocator, v: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(a);
    for (v) |c| {
        if (c == '\'') {
            try out.appendSlice(a, "''");
        } else {
            try out.append(a, c);
        }
    }
    return try out.toOwnedSlice(a);
}

pub fn createPrompt(self: *HanaAgent, prompt_text: []const u8, model: []const u8, user_id: []const u8, mode_id: i32, tags: []const u8) !void {
    const t = try escape(self.allocator, prompt_text);
    defer self.allocator.free(t);
    const m = try escape(self.allocator, model);
    defer self.allocator.free(m);
    const u = try escape(self.allocator, user_id);
    defer self.allocator.free(u);
    const tg = try escape(self.allocator, tags);
    defer self.allocator.free(tg);

    const sql = try std.fmt.allocPrint(
        self.allocator,
        "INSERT INTO PROMPTS (PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, TAGS, CREATED_AT) VALUES ('{s}', {d}, '{s}', '{s}', '{s}', CURRENT_TIMESTAMP)",
        .{ t, mode_id, m, u, tg },
    );
    defer self.allocator.free(sql);
    _ = try self.runSql(sql);
}

pub fn listPrompts(self: *HanaAgent, top: usize, offset: usize) ![]u8 {
    const sql = try std.fmt.allocPrint(self.allocator, "SELECT PROMPT_ID, PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, TAGS, CREATED_AT FROM PROMPTS ORDER BY CREATED_AT DESC LIMIT {d} OFFSET {d}", .{ top, offset });
    defer self.allocator.free(sql);
    return try self.runSql(sql);
}

pub fn deletePrompt(self: *HanaAgent, id: i32) !void {
    const sql = try std.fmt.allocPrint(self.allocator, "DELETE FROM PROMPTS WHERE PROMPT_ID = {d}", .{id});
    defer self.allocator.free(sql);
    _ = try self.runSql(sql);
}

pub fn createComparison(self: *HanaAgent, prompt: []const u8, modelA: []const u8, modelB: []const u8, winner: []const u8, respA: []const u8, respB: []const u8, latA: i32, latB: i32, tpsA: i32, tpsB: i32) !void {
    const p = try escape(self.allocator, prompt);
    defer self.allocator.free(p);
    const ma = try escape(self.allocator, modelA);
    defer self.allocator.free(ma);
    const mb = try escape(self.allocator, modelB);
    defer self.allocator.free(mb);
    const w = try escape(self.allocator, winner);
    defer self.allocator.free(w);
    const ra = try escape(self.allocator, respA);
    defer self.allocator.free(ra);
    const rb = try escape(self.allocator, respB);
    defer self.allocator.free(rb);

    const sql = try std.fmt.allocPrint(
        self.allocator,
        "INSERT INTO PROMPT_COMPARISONS (PROMPT_TEXT, MODEL_A, MODEL_B, WINNER, RESPONSE_A, RESPONSE_B, LATENCY_A_MS, LATENCY_B_MS, TOKENS_PER_SECOND_A, TOKENS_PER_SECOND_B, CREATED_AT) VALUES ('{s}','{s}','{s}','{s}','{s}','{s}',{d},{d},{d},{d},CURRENT_TIMESTAMP)",
        .{ p, ma, mb, w, ra, rb, latA, latB, tpsA, tpsB },
    );
    defer self.allocator.free(sql);
    _ = try self.runSql(sql);
}

pub fn listComparisons(self: *HanaAgent, top: usize, offset: usize) ![]u8 {
    const sql = try std.fmt.allocPrint(self.allocator, "SELECT COMPARISON_ID, PROMPT_TEXT, MODEL_A, MODEL_B, WINNER, RESPONSE_A, RESPONSE_B, LATENCY_A_MS, LATENCY_B_MS, TOKENS_PER_SECOND_A, TOKENS_PER_SECOND_B, CREATED_AT FROM PROMPT_COMPARISONS ORDER BY CREATED_AT DESC LIMIT {d} OFFSET {d}", .{ top, offset });
    defer self.allocator.free(sql);
    return try self.runSql(sql);
}

pub fn deleteComparison(self: *HanaAgent, id: i32) !void {
    const sql = try std.fmt.allocPrint(self.allocator, "DELETE FROM PROMPT_COMPARISONS WHERE COMPARISON_ID = {d}", .{id});
    defer self.allocator.free(sql);
    _ = try self.runSql(sql);
}
