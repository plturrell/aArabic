// glob module - Phase 1.17 - Full Implementation with path splitting and error handling
const std = @import("std");
const dirent = @import("../dirent/lib.zig");
const fnmatch_mod = @import("../fnmatch/lib.zig");

pub const GLOB_ERR: c_int = 1 << 0;
pub const GLOB_MARK: c_int = 1 << 1;
pub const GLOB_NOSORT: c_int = 1 << 2;
pub const GLOB_DOOFFS: c_int = 1 << 3;
pub const GLOB_NOCHECK: c_int = 1 << 4;
pub const GLOB_APPEND: c_int = 1 << 5;
pub const GLOB_NOESCAPE: c_int = 1 << 6;

pub const GLOB_NOSPACE: c_int = 1;
pub const GLOB_ABORTED: c_int = 2;
pub const GLOB_NOMATCH: c_int = 3;

pub const glob_t = extern struct {
    gl_pathc: usize,
    gl_pathv: [*]?[*:0]u8,
    gl_offs: usize,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Check if a pattern contains wildcard characters
fn hasWildcard(pat: []const u8) bool {
    for (pat) |c| {
        if (c == '*' or c == '?' or c == '[') return true;
    }
    return false;
}

/// Check if path is a directory using stat
fn isDirectory(path: [*:0]const u8) bool {
    var stat_buf: std.posix.system.Stat = undefined;
    const rc = std.posix.system.stat(path, &stat_buf);
    if (rc != 0) return false;
    return (stat_buf.mode & std.posix.S.IFMT) == std.posix.S.IFDIR;
}

/// Split pattern into directory and base pattern components
/// Returns (dir_path, file_pattern)
fn splitPattern(pat: []const u8) struct { dir: []const u8, base: []const u8 } {
    // Find last slash
    var last_slash: ?usize = null;
    for (pat, 0..) |c, i| {
        if (c == '/') last_slash = i;
    }

    if (last_slash) |idx| {
        if (idx == 0) {
            // Pattern starts with /
            return .{ .dir = "/", .base = pat[1..] };
        }
        return .{ .dir = pat[0..idx], .base = pat[idx + 1 ..] };
    }
    return .{ .dir = ".", .base = pat };
}

/// Recursive glob implementation
fn globRecursive(
    dir_path: []const u8,
    remaining_pattern: []const u8,
    flags: c_int,
    errfunc: ?*const fn ([*:0]const u8, c_int) callconv(.C) c_int,
    matches: *std.ArrayList([*:0]u8),
) c_int {
    // Split the remaining pattern into first component and rest
    var first_end: usize = 0;
    while (first_end < remaining_pattern.len and remaining_pattern[first_end] != '/') {
        first_end += 1;
    }

    const first_component = remaining_pattern[0..first_end];
    const rest = if (first_end < remaining_pattern.len)
        remaining_pattern[first_end + 1 ..]
    else
        "";

    // Create null-terminated dir path for opendir
    const dir_path_z = allocator.dupeZ(u8, dir_path) catch return GLOB_NOSPACE;
    defer allocator.free(dir_path_z);

    // If first component has no wildcards, just check if it exists
    if (!hasWildcard(first_component)) {
        // Build full path
        const full_path = if (std.mem.eql(u8, dir_path, "."))
            allocator.dupeZ(u8, first_component) catch return GLOB_NOSPACE
        else if (std.mem.eql(u8, dir_path, "/"))
            std.fmt.allocPrintZ(allocator, "/{s}", .{first_component}) catch return GLOB_NOSPACE
        else
            std.fmt.allocPrintZ(allocator, "{s}/{s}", .{ dir_path, first_component }) catch return GLOB_NOSPACE;

        if (rest.len > 0) {
            // More pattern to match - recurse
            defer allocator.free(full_path);
            const full_path_slice = std.mem.span(full_path);
            return globRecursive(full_path_slice, rest, flags, errfunc, matches);
        } else {
            // Final component - check if exists and add to matches
            var stat_buf: std.posix.system.Stat = undefined;
            const rc = std.posix.system.stat(full_path, &stat_buf);
            if (rc == 0) {
                // Path exists - add it
                var result_path = full_path;
                // GLOB_MARK: append / to directories
                if ((flags & GLOB_MARK) != 0) {
                    if ((stat_buf.mode & std.posix.S.IFMT) == std.posix.S.IFDIR) {
                        const marked = std.fmt.allocPrintZ(allocator, "{s}/", .{std.mem.span(full_path)}) catch {
                            allocator.free(full_path);
                            return GLOB_NOSPACE;
                        };
                        allocator.free(full_path);
                        result_path = marked;
                    }
                }
                matches.append(result_path) catch {
                    allocator.free(result_path);
                    return GLOB_NOSPACE;
                };
            } else {
                allocator.free(full_path);
            }
            return 0;
        }
    }

    // First component has wildcards - scan directory
    const dir = dirent.opendir(dir_path_z) orelse {
        // Handle error callback
        if (errfunc) |ef| {
            const errno_ptr = @import("../errno/lib.zig").__errno_location();
            if (ef(dir_path_z, errno_ptr.*) != 0 or (flags & GLOB_ERR) != 0) {
                return GLOB_ABORTED;
            }
        } else if ((flags & GLOB_ERR) != 0) {
            return GLOB_ABORTED;
        }
        return 0; // Continue without this directory
    };
    defer _ = dirent.closedir(dir);

    // Create pattern for fnmatch (null-terminated)
    const pattern_z = allocator.dupeZ(u8, first_component) catch return GLOB_NOSPACE;
    defer allocator.free(pattern_z);

    // Set fnmatch flags
    var fn_flags: c_int = 0;
    if ((flags & GLOB_NOESCAPE) != 0) fn_flags |= fnmatch_mod.FNM_NOESCAPE;

    // Read directory entries
    while (dirent.readdir(dir)) |entry| {
        const name = std.mem.span(@as([*:0]const u8, @ptrCast(&entry.d_name)));

        // Skip . and ..
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;

        // Match against pattern
        if (fnmatch_mod.fnmatch(pattern_z, @ptrCast(&entry.d_name), fn_flags) == 0) {
            // Build full path for this match
            const full_path = if (std.mem.eql(u8, dir_path, "."))
                allocator.dupeZ(u8, name) catch return GLOB_NOSPACE
            else if (std.mem.eql(u8, dir_path, "/"))
                std.fmt.allocPrintZ(allocator, "/{s}", .{name}) catch return GLOB_NOSPACE
            else
                std.fmt.allocPrintZ(allocator, "{s}/{s}", .{ dir_path, name }) catch return GLOB_NOSPACE;

            if (rest.len > 0) {
                // More pattern to match - need directory, recurse
                defer allocator.free(full_path);
                if (isDirectory(full_path)) {
                    const full_path_slice = std.mem.span(full_path);
                    const rc = globRecursive(full_path_slice, rest, flags, errfunc, matches);
                    if (rc != 0 and rc != GLOB_NOMATCH) return rc;
                }
            } else {
                // Final component - add to matches
                var result_path = full_path;

                // GLOB_MARK: append / to directories
                if ((flags & GLOB_MARK) != 0 and entry.d_type == dirent.DT_DIR) {
                    const marked = std.fmt.allocPrintZ(allocator, "{s}/", .{std.mem.span(full_path)}) catch {
                        allocator.free(full_path);
                        return GLOB_NOSPACE;
                    };
                    allocator.free(full_path);
                    result_path = marked;
                }

                matches.append(result_path) catch {
                    allocator.free(result_path);
                    return GLOB_NOSPACE;
                };
            }
        }
    }

    return 0;
}

pub export fn glob(
    pattern: [*:0]const u8,
    flags: c_int,
    errfunc: ?*const fn ([*:0]const u8, c_int) callconv(.C) c_int,
    pglob: *glob_t,
) c_int {
    const pat_span = std.mem.span(pattern);

    var matches = std.ArrayList([*:0]u8).init(allocator);
    defer matches.deinit();

    // Handle GLOB_DOOFFS - reserve slots at beginning
    const offs: usize = if ((flags & GLOB_DOOFFS) != 0) pglob.gl_offs else 0;

    // If GLOB_APPEND, preserve existing matches
    if ((flags & GLOB_APPEND) != 0 and pglob.gl_pathc > 0) {
        var i: usize = 0;
        while (i < pglob.gl_pathc) : (i += 1) {
            if (pglob.gl_pathv[offs + i]) |p| {
                matches.append(p) catch return GLOB_NOSPACE;
            }
        }
        // Free old array (but not the strings, we're keeping them)
        const old_slice = pglob.gl_pathv[0 .. offs + pglob.gl_pathc + 1];
        allocator.free(old_slice);
    }

    // Handle absolute paths and relative paths
    const start_dir: []const u8 = if (pat_span.len > 0 and pat_span[0] == '/') "/" else ".";
    const pattern_to_match = if (pat_span.len > 0 and pat_span[0] == '/') pat_span[1..] else pat_span;

    // Perform recursive glob
    const rc = globRecursive(start_dir, pattern_to_match, flags, errfunc, &matches);
    if (rc != 0 and rc != GLOB_NOMATCH) {
        // Free any matches we collected
        for (matches.items) |m| {
            allocator.free(std.mem.span(m));
        }
        return rc;
    }

    // Handle no matches case
    if (matches.items.len == 0) {
        if ((flags & GLOB_NOCHECK) != 0) {
            const dup = allocator.dupeZ(u8, pat_span) catch return GLOB_NOSPACE;
            matches.append(dup) catch {
                allocator.free(dup);
                return GLOB_NOSPACE;
            };
        } else {
            return GLOB_NOMATCH;
        }
    }

    // Sort matches unless GLOB_NOSORT
    if ((flags & GLOB_NOSORT) == 0) {
        std.mem.sort([*:0]u8, matches.items, {}, struct {
            fn lessThan(_: void, a: [*:0]u8, b: [*:0]u8) bool {
                return std.mem.orderZ(u8, a, b) == .lt;
            }
        }.lessThan);
    }

    // Allocate result array (with offs slots + matches + NULL terminator)
    const result_len = offs + matches.items.len + 1;
    const result = allocator.alloc(?[*:0]u8, result_len) catch return GLOB_NOSPACE;

    // Initialize offset slots to null
    for (0..offs) |i| {
        result[i] = null;
    }

    // Copy matches
    for (matches.items, 0..) |m, i| {
        result[offs + i] = m;
    }
    result[offs + matches.items.len] = null;

    pglob.gl_pathc = matches.items.len;
    pglob.gl_pathv = result.ptr;
    if ((flags & GLOB_DOOFFS) == 0) {
        pglob.gl_offs = 0;
    }

    return 0;
}

pub export fn globfree(pglob: *glob_t) void {
    if (pglob.gl_pathc == 0) return;

    const offs = pglob.gl_offs;

    // Free each path string
    var i: usize = 0;
    while (i < pglob.gl_pathc) : (i += 1) {
        if (pglob.gl_pathv[offs + i]) |ptr| {
            const slice = std.mem.span(ptr);
            allocator.free(slice[0 .. slice.len + 1]); // +1 for null terminator
        }
    }

    // Free the array itself
    const total_len = offs + pglob.gl_pathc + 1;
    const slice = pglob.gl_pathv[0..total_len];
    allocator.free(slice);

    pglob.gl_pathc = 0;
    pglob.gl_offs = 0;
}
