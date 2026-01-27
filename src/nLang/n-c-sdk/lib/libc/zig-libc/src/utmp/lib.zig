// utmp module - User accounting - Phase 1.32
// Real implementation reading /var/run/utmp (Linux) or /var/run/utmpx (macOS)
const std = @import("std");
const builtin = @import("builtin");
const fs = std.fs;

// utmp entry types
pub const EMPTY: c_short = 0;
pub const RUN_LVL: c_short = 1;
pub const BOOT_TIME: c_short = 2;
pub const NEW_TIME: c_short = 3;
pub const OLD_TIME: c_short = 4;
pub const INIT_PROCESS: c_short = 5;
pub const LOGIN_PROCESS: c_short = 6;
pub const USER_PROCESS: c_short = 7;
pub const DEAD_PROCESS: c_short = 8;
pub const ACCOUNTING: c_short = 9;

pub const utmp = extern struct {
    ut_type: c_short,
    ut_pid: c_int,
    ut_line: [32]u8,
    ut_id: [4]u8,
    ut_user: [32]u8,
    ut_host: [256]u8,
    ut_exit: extern struct {
        e_termination: c_short,
        e_exit: c_short,
    },
    ut_session: c_long,
    ut_tv: extern struct {
        tv_sec: i32,
        tv_usec: i32,
    },
    ut_addr_v6: [4]i32,
    __unused: [20]u8,
};

pub const utmpx = extern struct {
    ut_type: c_short,
    ut_pid: c_int,
    ut_line: [32]u8,
    ut_id: [4]u8,
    ut_user: [32]u8,
    ut_host: [256]u8,
    ut_exit: extern struct {
        e_termination: c_short,
        e_exit: c_short,
    },
    ut_session: c_long,
    ut_tv: extern struct {
        tv_sec: i64,
        tv_usec: i64,
    },
    ut_addr_v6: [4]i32,
    __unused: [20]u8,
};

// Default utmp file paths
const UTMP_FILE = if (builtin.os.tag == .macos) "/var/run/utmpx" else "/var/run/utmp";
const WTMP_FILE = if (builtin.os.tag == .macos) "/var/log/wtmp" else "/var/log/wtmp";

// Internal state
var utmp_file: ?fs.File = null;
var utmp_path: []const u8 = UTMP_FILE;
var current_entry: utmp = undefined;
var current_entryx: utmpx = undefined;

/// Open or rewind the utmp file
pub export fn setutent() void {
    if (utmp_file) |*file| {
        file.seekTo(0) catch {};
    } else {
        utmp_file = fs.openFileAbsolute(utmp_path, .{}) catch null;
    }
}

/// Close the utmp file
pub export fn endutent() void {
    if (utmp_file) |*file| {
        file.close();
        utmp_file = null;
    }
}

/// Read next utmp entry
pub export fn getutent() ?*utmp {
    if (utmp_file == null) {
        setutent();
    }

    if (utmp_file) |file| {
        const bytes = file.reader().readBytesNoEof(@sizeOf(utmp)) catch return null;
        current_entry = @bitCast(bytes);
        return &current_entry;
    }
    return null;
}

/// Find utmp entry by ID or type
pub export fn getutid(ut: *const utmp) ?*utmp {
    if (utmp_file == null) {
        setutent();
    }

    while (getutent()) |entry| {
        // Match by type for run-level/boot/time entries
        if (ut.ut_type == RUN_LVL or ut.ut_type == BOOT_TIME or
            ut.ut_type == NEW_TIME or ut.ut_type == OLD_TIME)
        {
            if (entry.ut_type == ut.ut_type) return entry;
        }
        // Match by ut_id for process entries
        else if (ut.ut_type == INIT_PROCESS or ut.ut_type == LOGIN_PROCESS or
            ut.ut_type == USER_PROCESS or ut.ut_type == DEAD_PROCESS)
        {
            if (entry.ut_type == INIT_PROCESS or entry.ut_type == LOGIN_PROCESS or
                entry.ut_type == USER_PROCESS or entry.ut_type == DEAD_PROCESS)
            {
                if (std.mem.eql(u8, &entry.ut_id, &ut.ut_id)) return entry;
            }
        }
    }
    return null;
}

/// Find utmp entry by line
pub export fn getutline(ut: *const utmp) ?*utmp {
    if (utmp_file == null) {
        setutent();
    }

    while (getutent()) |entry| {
        if (entry.ut_type == LOGIN_PROCESS or entry.ut_type == USER_PROCESS) {
            // Compare ut_line (null-terminated comparison)
            var match = true;
            for (ut.ut_line, entry.ut_line) |a, b| {
                if (a != b) {
                    match = false;
                    break;
                }
                if (a == 0) break;
            }
            if (match) return entry;
        }
    }
    return null;
}

/// Write utmp entry (requires write access to utmp file)
pub export fn pututline(ut: *const utmp) ?*utmp {
    // Try to open for writing
    const file = fs.openFileAbsolute(utmp_path, .{ .mode = .read_write }) catch return null;
    defer file.close();

    // Search for matching entry to update
    var found = false;
    var offset: u64 = 0;

    while (true) {
        const bytes = file.reader().readBytesNoEof(@sizeOf(utmp)) catch break;
        const entry: utmp = @bitCast(bytes);

        if (ut.ut_type == entry.ut_type and std.mem.eql(u8, &ut.ut_id, &entry.ut_id)) {
            found = true;
            break;
        }
        offset += @sizeOf(utmp);
    }

    if (!found) {
        // Append at end
        file.seekFromEnd(0) catch return null;
    } else {
        // Seek back to overwrite
        file.seekTo(offset) catch return null;
    }

    const ut_bytes: *const [@sizeOf(utmp)]u8 = @ptrCast(ut);
    _ = file.write(ut_bytes) catch return null;

    current_entry = ut.*;
    return &current_entry;
}

/// Set utmp file path
pub export fn utmpname(file: [*:0]const u8) c_int {
    endutent();
    utmp_path = std.mem.span(file);
    return 0;
}

// utmpx functions (extended utmp)
var utmpx_file: ?fs.File = null;
var utmpx_path: []const u8 = UTMP_FILE;

pub export fn setutxent() void {
    if (utmpx_file) |*file| {
        file.seekTo(0) catch {};
    } else {
        utmpx_file = fs.openFileAbsolute(utmpx_path, .{}) catch null;
    }
}

pub export fn endutxent() void {
    if (utmpx_file) |*file| {
        file.close();
        utmpx_file = null;
    }
}

pub export fn getutxent() ?*utmpx {
    if (utmpx_file == null) {
        setutxent();
    }

    if (utmpx_file) |file| {
        const bytes = file.reader().readBytesNoEof(@sizeOf(utmpx)) catch return null;
        current_entryx = @bitCast(bytes);
        return &current_entryx;
    }
    return null;
}

pub export fn getutxid(ut: *const utmpx) ?*utmpx {
    if (utmpx_file == null) {
        setutxent();
    }

    while (getutxent()) |entry| {
        if (ut.ut_type == RUN_LVL or ut.ut_type == BOOT_TIME or
            ut.ut_type == NEW_TIME or ut.ut_type == OLD_TIME)
        {
            if (entry.ut_type == ut.ut_type) return entry;
        } else if (ut.ut_type == INIT_PROCESS or ut.ut_type == LOGIN_PROCESS or
            ut.ut_type == USER_PROCESS or ut.ut_type == DEAD_PROCESS)
        {
            if (entry.ut_type == INIT_PROCESS or entry.ut_type == LOGIN_PROCESS or
                entry.ut_type == USER_PROCESS or entry.ut_type == DEAD_PROCESS)
            {
                if (std.mem.eql(u8, &entry.ut_id, &ut.ut_id)) return entry;
            }
        }
    }
    return null;
}

pub export fn getutxline(ut: *const utmpx) ?*utmpx {
    if (utmpx_file == null) {
        setutxent();
    }

    while (getutxent()) |entry| {
        if (entry.ut_type == LOGIN_PROCESS or entry.ut_type == USER_PROCESS) {
            var match = true;
            for (ut.ut_line, entry.ut_line) |a, b| {
                if (a != b) {
                    match = false;
                    break;
                }
                if (a == 0) break;
            }
            if (match) return entry;
        }
    }
    return null;
}

pub export fn pututxline(ut: *const utmpx) ?*utmpx {
    const file = fs.openFileAbsolute(utmpx_path, .{ .mode = .read_write }) catch return null;
    defer file.close();

    var found = false;
    var offset: u64 = 0;

    while (true) {
        const bytes = file.reader().readBytesNoEof(@sizeOf(utmpx)) catch break;
        const entry: utmpx = @bitCast(bytes);

        if (ut.ut_type == entry.ut_type and std.mem.eql(u8, &ut.ut_id, &entry.ut_id)) {
            found = true;
            break;
        }
        offset += @sizeOf(utmpx);
    }

    if (!found) {
        file.seekFromEnd(0) catch return null;
    } else {
        file.seekTo(offset) catch return null;
    }

    const ut_bytes: *const [@sizeOf(utmpx)]u8 = @ptrCast(ut);
    _ = file.write(ut_bytes) catch return null;

    current_entryx = ut.*;
    return &current_entryx;
}
