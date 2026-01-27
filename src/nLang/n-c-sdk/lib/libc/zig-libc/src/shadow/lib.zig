// shadow module - Shadow password - Phase 1.28
// Real implementation reading /etc/shadow
const std = @import("std");
const fs = std.fs;

pub const spwd = extern struct {
    sp_namp: [*:0]u8,
    sp_pwdp: [*:0]u8,
    sp_lstchg: c_long,
    sp_min: c_long,
    sp_max: c_long,
    sp_warn: c_long,
    sp_inact: c_long,
    sp_expire: c_long,
    sp_flag: c_ulong,
};

const SHADOW_FILE = "/etc/shadow";

// Internal state
var shadow_file: ?fs.File = null;
var current_entry: spwd = undefined;
var name_buf: [256]u8 = undefined;
var pwd_buf: [256]u8 = undefined;

fn parseLong(s: []const u8) c_long {
    if (s.len == 0) return -1;
    return std.fmt.parseInt(c_long, s, 10) catch -1;
}

fn parseUlong(s: []const u8) c_ulong {
    if (s.len == 0) return 0;
    return std.fmt.parseInt(c_ulong, s, 10) catch 0;
}

fn parseShadowLine(line: []const u8, entry: *spwd, name_buffer: []u8, pwd_buffer: []u8) bool {
    var fields: [9][]const u8 = undefined;
    var field_count: usize = 0;
    var start: usize = 0;

    // Split by ':'
    for (line, 0..) |c, i| {
        if (c == ':') {
            if (field_count < 9) {
                fields[field_count] = line[start..i];
                field_count += 1;
            }
            start = i + 1;
        }
    }
    if (field_count < 9 and start < line.len) {
        fields[field_count] = line[start..];
        field_count += 1;
    }

    if (field_count < 2) return false;

    // Copy name (field 0)
    const name_len = @min(fields[0].len, name_buffer.len - 1);
    @memcpy(name_buffer[0..name_len], fields[0][0..name_len]);
    name_buffer[name_len] = 0;
    entry.sp_namp = @ptrCast(name_buffer.ptr);

    // Copy password (field 1)
    const pwd_len = @min(fields[1].len, pwd_buffer.len - 1);
    @memcpy(pwd_buffer[0..pwd_len], fields[1][0..pwd_len]);
    pwd_buffer[pwd_len] = 0;
    entry.sp_pwdp = @ptrCast(pwd_buffer.ptr);

    // Parse numeric fields
    entry.sp_lstchg = if (field_count > 2) parseLong(fields[2]) else -1;
    entry.sp_min = if (field_count > 3) parseLong(fields[3]) else -1;
    entry.sp_max = if (field_count > 4) parseLong(fields[4]) else -1;
    entry.sp_warn = if (field_count > 5) parseLong(fields[5]) else -1;
    entry.sp_inact = if (field_count > 6) parseLong(fields[6]) else -1;
    entry.sp_expire = if (field_count > 7) parseLong(fields[7]) else -1;
    entry.sp_flag = if (field_count > 8) parseUlong(fields[8]) else 0;

    return true;
}

/// Get shadow entry by name
pub export fn getspnam(name: [*:0]const u8) ?*spwd {
    const target = std.mem.span(name);

    const file = fs.openFileAbsolute(SHADOW_FILE, .{}) catch return null;
    defer file.close();

    var line_buf: [1024]u8 = undefined;
    const reader = file.reader();

    while (reader.readUntilDelimiterOrEof(&line_buf, '\n') catch null) |line| {
        // Skip comments and empty lines
        if (line.len == 0 or line[0] == '#') continue;

        if (parseShadowLine(line, &current_entry, &name_buf, &pwd_buf)) {
            const entry_name = std.mem.span(current_entry.sp_namp);
            if (std.mem.eql(u8, entry_name, target)) {
                return &current_entry;
            }
        }
    }

    return null;
}

/// Get next shadow entry (sequential access)
pub export fn getspent() ?*spwd {
    if (shadow_file == null) {
        setspent();
    }

    if (shadow_file) |file| {
        var line_buf: [1024]u8 = undefined;
        const reader = file.reader();

        while (reader.readUntilDelimiterOrEof(&line_buf, '\n') catch null) |line| {
            if (line.len == 0 or line[0] == '#') continue;

            if (parseShadowLine(line, &current_entry, &name_buf, &pwd_buf)) {
                return &current_entry;
            }
        }
    }

    return null;
}

/// Open/rewind shadow file
pub export fn setspent() void {
    if (shadow_file) |*file| {
        file.seekTo(0) catch {};
    } else {
        shadow_file = fs.openFileAbsolute(SHADOW_FILE, .{}) catch null;
    }
}

/// Close shadow file
pub export fn endspent() void {
    if (shadow_file) |*file| {
        file.close();
        shadow_file = null;
    }
}

/// Get shadow entry from file stream
pub export fn fgetspent(stream: *anyopaque) ?*spwd {
    _ = stream;
    // Would need FILE* implementation
    return null;
}

/// Parse shadow entry from string
pub export fn sgetspent(s: [*:0]const u8) ?*spwd {
    const line = std.mem.span(s);
    if (parseShadowLine(line, &current_entry, &name_buf, &pwd_buf)) {
        return &current_entry;
    }
    return null;
}

/// Write shadow entry to stream
pub export fn putspent(p: *const spwd, stream: *anyopaque) c_int {
    _ = p;
    _ = stream;
    // Would need FILE* implementation for proper output
    return -1;
}

/// Thread-safe getspnam
pub export fn getspnam_r(name: [*:0]const u8, result_buf: *spwd, buffer: [*]u8, buflen: usize, result: *?*spwd) c_int {
    if (buflen < 512) {
        result.* = null;
        return 34; // ERANGE
    }

    const target = std.mem.span(name);

    const file = fs.openFileAbsolute(SHADOW_FILE, .{}) catch {
        result.* = null;
        return 2; // ENOENT
    };
    defer file.close();

    var line_buf: [1024]u8 = undefined;
    const reader = file.reader();

    // Split buffer for name and password
    const name_buffer = buffer[0..256];
    const pwd_buffer = buffer[256..512];

    while (reader.readUntilDelimiterOrEof(&line_buf, '\n') catch null) |line| {
        if (line.len == 0 or line[0] == '#') continue;

        if (parseShadowLine(line, result_buf, name_buffer, pwd_buffer)) {
            const entry_name = std.mem.span(result_buf.sp_namp);
            if (std.mem.eql(u8, entry_name, target)) {
                result.* = result_buf;
                return 0;
            }
        }
    }

    result.* = null;
    return 0;
}

/// Thread-safe getspent
pub export fn getspent_r(result_buf: *spwd, buffer: [*]u8, buflen: usize, result: *?*spwd) c_int {
    if (buflen < 512) {
        result.* = null;
        return 34; // ERANGE
    }

    if (shadow_file == null) {
        setspent();
    }

    if (shadow_file) |file| {
        var line_buf: [1024]u8 = undefined;
        const reader = file.reader();

        const name_buffer = buffer[0..256];
        const pwd_buffer = buffer[256..512];

        while (reader.readUntilDelimiterOrEof(&line_buf, '\n') catch null) |line| {
            if (line.len == 0 or line[0] == '#') continue;

            if (parseShadowLine(line, result_buf, name_buffer, pwd_buffer)) {
                result.* = result_buf;
                return 0;
            }
        }
    }

    result.* = null;
    return 0;
}

/// Thread-safe fgetspent
pub export fn fgetspent_r(stream: *anyopaque, result_buf: *spwd, buffer: [*]u8, buflen: usize, result: *?*spwd) c_int {
    _ = stream;
    _ = result_buf;
    _ = buffer;
    _ = buflen;
    result.* = null;
    return -1;
}

// File locking for shadow file (requires root)
var shadow_lock_fd: ?fs.File = null;

/// Lock shadow file for writing
pub export fn lckpwdf() c_int {
    shadow_lock_fd = fs.openFileAbsolute("/etc/.pwd.lock", .{ .mode = .write_only }) catch {
        // Try to create if doesn't exist
        shadow_lock_fd = fs.createFileAbsolute("/etc/.pwd.lock", .{}) catch return -1;
        return 0;
    };

    // Would need flock() implementation for proper locking
    return 0;
}

/// Unlock shadow file
pub export fn ulckpwdf() c_int {
    if (shadow_lock_fd) |*file| {
        file.close();
        shadow_lock_fd = null;
    }
    return 0;
}
