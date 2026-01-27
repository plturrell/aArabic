// ftw module - Phase 1.20
// File tree walk implementation
const std = @import("std");
const posix = std.posix;
const fs = std.fs;

// File type flags returned to callback
pub const FTW_F: c_int = 0; // Regular file
pub const FTW_D: c_int = 1; // Directory
pub const FTW_DNR: c_int = 2; // Directory that cannot be read
pub const FTW_NS: c_int = 3; // Could not stat
pub const FTW_SL: c_int = 4; // Symbolic link
pub const FTW_DP: c_int = 5; // Directory (post-order visit, FTW_DEPTH)
pub const FTW_SLN: c_int = 6; // Symbolic link naming non-existing file

// nftw flags
pub const FTW_PHYS: c_int = 1; // Don't follow symlinks
pub const FTW_MOUNT: c_int = 2; // Stay within same filesystem
pub const FTW_CHDIR: c_int = 4; // chdir to each directory
pub const FTW_DEPTH: c_int = 8; // Depth-first (visit dir after contents)

// FTW structure for nftw callback
pub const FTW = extern struct {
    base: c_int, // Offset of basename in path
    level: c_int, // Depth relative to starting point
};

// stat structure (simplified for C compatibility)
pub const stat_t = extern struct {
    st_dev: u64,
    st_ino: u64,
    st_mode: u32,
    st_nlink: u64,
    st_uid: u32,
    st_gid: u32,
    st_rdev: u64,
    st_size: i64,
    st_blksize: i64,
    st_blocks: i64,
    st_atime: i64,
    st_mtime: i64,
    st_ctime: i64,
};

// Callback types
pub const FtwCallback = *const fn ([*:0]const u8, ?*const stat_t, c_int) callconv(.C) c_int;
pub const NftwCallback = *const fn ([*:0]const u8, ?*const stat_t, c_int, ?*FTW) callconv(.C) c_int;

// Internal allocator for path buffer
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// ftw - traverse a file tree
pub export fn ftw(path: [*:0]const u8, callback: FtwCallback, fd_limit: c_int) c_int {
    _ = fd_limit; // Not enforced in this implementation

    const allocator = gpa.allocator();
    const path_slice = std.mem.span(path);

    // Wrap ftw callback as nftw callback
    const result = walkTree(allocator, path_slice, callback, null, 0, 0) catch |err| {
        _ = err;
        return -1;
    };

    return result;
}

/// nftw - traverse a file tree (extended)
pub export fn nftw(path: [*:0]const u8, callback: NftwCallback, fd_limit: c_int, flags: c_int) c_int {
    _ = fd_limit;

    const allocator = gpa.allocator();
    const path_slice = std.mem.span(path);

    const result = walkTreeNftw(allocator, path_slice, callback, flags, 0) catch |err| {
        _ = err;
        return -1;
    };

    return result;
}

fn walkTree(
    allocator: std.mem.Allocator,
    path: []const u8,
    callback: FtwCallback,
    _: ?NftwCallback,
    _: c_int,
    _: c_int,
) !c_int {
    // Build null-terminated path
    const path_z = try allocator.allocSentinel(u8, path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, path);

    // Stat the path
    var stat_buf: stat_t = undefined;
    const file_type = getFileType(path, &stat_buf, false);

    // Call callback
    const result = callback(path_z.ptr, &stat_buf, file_type);
    if (result != 0) return result;

    // If directory, recurse
    if (file_type == FTW_D) {
        var dir = fs.openDirAbsolute(path, .{ .iterate = true }) catch {
            return 0; // Continue on error
        };
        defer dir.close();

        var iter = dir.iterate();
        while (iter.next() catch null) |entry| {
            if (std.mem.eql(u8, entry.name, ".") or std.mem.eql(u8, entry.name, "..")) continue;

            // Build full path
            const full_path = try std.fs.path.join(allocator, &.{ path, entry.name });
            defer allocator.free(full_path);

            const sub_result = try walkTree(allocator, full_path, callback, null, 0, 0);
            if (sub_result != 0) return sub_result;
        }
    }

    return 0;
}

fn walkTreeNftw(
    allocator: std.mem.Allocator,
    path: []const u8,
    callback: NftwCallback,
    flags: c_int,
    level: c_int,
) !c_int {
    const depth_first = (flags & FTW_DEPTH) != 0;
    const follow_symlinks = (flags & FTW_PHYS) == 0;

    // Build null-terminated path
    const path_z = try allocator.allocSentinel(u8, path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, path);

    // Stat the path
    var stat_buf: stat_t = undefined;
    const file_type = getFileType(path, &stat_buf, follow_symlinks);

    // Build FTW structure
    var ftw_info = FTW{
        .base = @intCast(getBasenameOffset(path)),
        .level = level,
    };

    // Pre-order callback (unless depth-first)
    if (!depth_first) {
        const result = callback(path_z.ptr, &stat_buf, file_type, &ftw_info);
        if (result != 0) return result;
    }

    // If directory, recurse
    if (file_type == FTW_D or (depth_first and file_type == FTW_DP)) {
        var dir = fs.openDirAbsolute(path, .{ .iterate = true }) catch {
            if (!depth_first) return 0;
            // For depth-first, still call callback with FTW_DNR
            const result = callback(path_z.ptr, &stat_buf, FTW_DNR, &ftw_info);
            return result;
        };
        defer dir.close();

        var iter = dir.iterate();
        while (iter.next() catch null) |entry| {
            if (std.mem.eql(u8, entry.name, ".") or std.mem.eql(u8, entry.name, "..")) continue;

            const full_path = try std.fs.path.join(allocator, &.{ path, entry.name });
            defer allocator.free(full_path);

            const sub_result = try walkTreeNftw(allocator, full_path, callback, flags, level + 1);
            if (sub_result != 0) return sub_result;
        }
    }

    // Post-order callback (depth-first)
    if (depth_first and file_type == FTW_D) {
        const result = callback(path_z.ptr, &stat_buf, FTW_DP, &ftw_info);
        if (result != 0) return result;
    }

    return 0;
}

fn getFileType(path: []const u8, stat_buf: *stat_t, follow_symlinks: bool) c_int {
    _ = follow_symlinks;

    // Use Zig's fs to get file info
    const file = fs.openFileAbsolute(path, .{}) catch |err| {
        @memset(std.mem.asBytes(stat_buf), 0);
        return switch (err) {
            error.AccessDenied => FTW_DNR,
            else => FTW_NS,
        };
    };
    defer file.close();

    const stat = file.stat() catch {
        @memset(std.mem.asBytes(stat_buf), 0);
        return FTW_NS;
    };

    // Fill stat buffer
    stat_buf.st_size = @intCast(stat.size);
    stat_buf.st_mtime = @divTrunc(stat.mtime, std.time.ns_per_s);
    stat_buf.st_atime = @divTrunc(stat.atime, std.time.ns_per_s);
    stat_buf.st_ctime = @divTrunc(stat.ctime, std.time.ns_per_s);
    stat_buf.st_mode = @intCast(@intFromEnum(stat.kind));
    stat_buf.st_ino = stat.inode;

    return switch (stat.kind) {
        .directory => FTW_D,
        .sym_link => FTW_SL,
        else => FTW_F,
    };
}

fn getBasenameOffset(path: []const u8) usize {
    var i = path.len;
    while (i > 0) : (i -= 1) {
        if (path[i - 1] == '/') return i;
    }
    return 0;
}
