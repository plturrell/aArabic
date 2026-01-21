// Structured logging module for nCode server
// Provides JSON-formatted logs with levels and context

const std = @import("std");

pub const LogLevel = enum {
    DEBUG,
    INFO,
    WARN,
    ERROR,

    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .DEBUG => "DEBUG",
            .INFO => "INFO",
            .WARN => "WARN",
            .ERROR => "ERROR",
        };
    }

    pub fn fromString(s: []const u8) ?LogLevel {
        if (std.mem.eql(u8, s, "DEBUG")) return .DEBUG;
        if (std.mem.eql(u8, s, "INFO")) return .INFO;
        if (std.mem.eql(u8, s, "WARN")) return .WARN;
        if (std.mem.eql(u8, s, "ERROR")) return .ERROR;
        return null;
    }
};

pub const Logger = struct {
    level: LogLevel,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Logger {
        const level_str = std.posix.getenv("NCODE_LOG_LEVEL") orelse "INFO";
        const level = LogLevel.fromString(level_str) orelse .INFO;
        return .{
            .level = level,
            .allocator = allocator,
        };
    }

    pub fn shouldLog(self: Logger, level: LogLevel) bool {
        return @intFromEnum(level) >= @intFromEnum(self.level);
    }

    fn getTimestamp(allocator: std.mem.Allocator) ![]const u8 {
        const timestamp = std.time.timestamp();
        return try std.fmt.allocPrint(allocator, "{d}", .{timestamp});
    }

    pub fn log(self: Logger, level: LogLevel, message: []const u8, context: ?[]const u8) void {
        if (!self.shouldLog(level)) return;

        const timestamp = self.getTimestamp(self.allocator) catch "0";
        defer self.allocator.free(timestamp);

        const ctx = context orelse "{}";
        const log_entry = std.fmt.allocPrint(
            self.allocator,
            "{{\"timestamp\":{s},\"level\":\"{s}\",\"message\":\"{s}\",\"context\":{s}}}\n",
            .{ timestamp, level.toString(), message, ctx },
        ) catch {
            std.debug.print("Failed to format log entry\n", .{});
            return;
        };
        defer self.allocator.free(log_entry);

        std.debug.print("{s}", .{log_entry});
    }

    pub fn debug(self: Logger, message: []const u8, context: ?[]const u8) void {
        self.log(.DEBUG, message, context);
    }

    pub fn info(self: Logger, message: []const u8, context: ?[]const u8) void {
        self.log(.INFO, message, context);
    }

    pub fn warn(self: Logger, message: []const u8, context: ?[]const u8) void {
        self.log(.WARN, message, context);
    }

    pub fn err(self: Logger, message: []const u8, context: ?[]const u8) void {
        self.log(.ERROR, message, context);
    }

    pub fn logRequest(self: Logger, method: []const u8, path: []const u8, status: u16, duration_ms: i64) void {
        const context = std.fmt.allocPrint(
            self.allocator,
            "{{\"method\":\"{s}\",\"path\":\"{s}\",\"status\":{d},\"duration_ms\":{d}}}",
            .{ method, path, status, duration_ms },
        ) catch return;
        defer self.allocator.free(context);

        const level: LogLevel = if (status >= 500) .ERROR else if (status >= 400) .WARN else .INFO;
        self.log(level, "HTTP request", context);
    }
};
