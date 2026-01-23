const std = @import("std");
const json = std.json;

pub const OAuthClient = struct {
    allocator: std.mem.Allocator,
    auth_url: []const u8,
    client_id: []const u8,
    client_secret: []const u8,
    cached_token: ?[]u8 = null,
    expires_at_ms: i64 = 0,

    pub fn init(allocator: std.mem.Allocator, auth_url: []const u8, client_id: []const u8, client_secret: []const u8) !OAuthClient {
        return .{
            .allocator = allocator,
            .auth_url = auth_url,
            .client_id = client_id,
            .client_secret = client_secret,
        };
    }

    pub fn deinit(self: *OAuthClient) void {
        if (self.cached_token) |t| self.allocator.free(t);
        self.cached_token = null;
    }

    pub fn getToken(self: *OAuthClient) ![]const u8 {
        const now_ms: i64 = @intCast(std.time.milliTimestamp());
        if (self.cached_token != null and now_ms < self.expires_at_ms) {
            return self.cached_token.?;
        }

        const payload = try std.fmt.allocPrint(self.allocator, "grant_type=client_credentials&client_id={s}&client_secret={s}", .{ self.client_id, self.client_secret });
        defer self.allocator.free(payload);

        const args = [_][]const u8{
            "curl",
            "-s",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/x-www-form-urlencoded",
            "-d",
            payload,
            self.auth_url,
        };

        var child = std.process.Child.init(&args, self.allocator);
        child.stdout_behavior = .Pipe;
        try child.spawn();

        const out = try child.stdout.?.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(out);
        _ = try child.wait();

        var parsed = try json.parseFromSlice(json.Value, self.allocator, out, .{});
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return error.InvalidResponse,
        };

        const token_val = obj.get("access_token") orelse return error.InvalidResponse;
        const expires_val = obj.get("expires_in");

        const tok = switch (token_val) {
            .string => |s| try self.allocator.dupe(u8, s),
            else => return error.InvalidResponse,
        };

        if (self.cached_token) |t| self.allocator.free(t);
        self.cached_token = tok;

        const expires_in = if (expires_val) |v| switch (v) {
            .integer => |i| i,
            .number_string => |s| std.fmt.parseInt(i64, s, 10) catch 3600,
            else => 3600,
        } else 3600;

        self.expires_at_ms = now_ms + (expires_in * 1000) - 30000; // refresh 30s early
        return tok;
    }
};
