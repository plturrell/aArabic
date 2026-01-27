// mntent module - Mount entries - Phase 1.32
// Reads mount entries from /etc/mtab, /etc/fstab, or /proc/mounts
const std = @import("std");
const fs = std.fs;

pub const mntent = extern struct {
    mnt_fsname: [*:0]u8, // Device or server for filesystem
    mnt_dir: [*:0]u8, // Directory mounted on
    mnt_type: [*:0]u8, // Type of filesystem
    mnt_opts: [*:0]u8, // Mount options
    mnt_freq: c_int, // Dump frequency (days)
    mnt_passno: c_int, // Pass number for fsck
};

// Internal stream structure
const MntStream = struct {
    file: fs.File,
    line_buf: [1024]u8,
    // Static buffers for current entry
    fsname_buf: [256]u8,
    dir_buf: [256]u8,
    type_buf: [64]u8,
    opts_buf: [256]u8,
    current: mntent,
};

// Global allocator for stream structures
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// Open mount table file
pub export fn setmntent(filename: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque {
    _ = mode; // Only read mode supported

    const allocator = gpa.allocator();
    const stream = allocator.create(MntStream) catch return null;

    const path = std.mem.span(filename);
    stream.file = fs.openFileAbsolute(path, .{}) catch {
        allocator.destroy(stream);
        return null;
    };

    return @ptrCast(stream);
}

/// Read next mount entry (not thread-safe)
pub export fn getmntent(stream_ptr: ?*anyopaque) ?*mntent {
    if (stream_ptr == null) return null;
    const stream: *MntStream = @ptrCast(@alignCast(stream_ptr));

    return readMntent(stream, &stream.current, &stream.fsname_buf, &stream.dir_buf, &stream.type_buf, &stream.opts_buf);
}

/// Read next mount entry (thread-safe, reentrant)
pub export fn getmntent_r(stream_ptr: ?*anyopaque, mntbuf: *mntent, buf: [*]u8, buflen: c_int) ?*mntent {
    if (stream_ptr == null) return null;
    const stream: *MntStream = @ptrCast(@alignCast(stream_ptr));

    if (buflen < 512) return null; // Need minimum buffer size

    // Split buffer into sections
    const buf_slice = buf[0..@intCast(buflen)];
    const quarter: usize = @intCast(@divFloor(buflen, 4));

    const fsname_buf: *[256]u8 = @ptrCast(buf_slice[0..256]);
    const dir_buf: *[256]u8 = @ptrCast(buf_slice[quarter .. quarter + 256]);
    const type_buf: *[64]u8 = @ptrCast(buf_slice[quarter * 2 .. quarter * 2 + 64]);
    const opts_buf: *[256]u8 = @ptrCast(buf_slice[quarter * 3 .. quarter * 3 + 256]);

    return readMntent(stream, mntbuf, fsname_buf, dir_buf, type_buf, opts_buf);
}

fn readMntent(stream: *MntStream, mntbuf: *mntent, fsname_buf: *[256]u8, dir_buf: *[256]u8, type_buf: *[64]u8, opts_buf: *[256]u8) ?*mntent {
    var buf_reader = std.io.bufferedReader(stream.file.reader());
    var reader = buf_reader.reader();

    while (reader.readUntilDelimiterOrEof(&stream.line_buf, '\n') catch null) |line| {
        const trim_line = std.mem.trim(u8, line, " \t\r");
        if (trim_line.len == 0 or trim_line[0] == '#') continue;

        // Parse: fsname dir type opts freq passno
        var iter = std.mem.tokenizeAny(u8, trim_line, " \t");

        const fsname = iter.next() orelse continue;
        const dir = iter.next() orelse continue;
        const fstype = iter.next() orelse continue;
        const opts = iter.next() orelse "defaults";
        const freq_str = iter.next() orelse "0";
        const passno_str = iter.next() orelse "0";

        // Copy to buffers
        const fsname_len = @min(fsname.len, 255);
        @memcpy(fsname_buf[0..fsname_len], fsname[0..fsname_len]);
        fsname_buf[fsname_len] = 0;

        const dir_len = @min(dir.len, 255);
        @memcpy(dir_buf[0..dir_len], dir[0..dir_len]);
        dir_buf[dir_len] = 0;

        const type_len = @min(fstype.len, 63);
        @memcpy(type_buf[0..type_len], fstype[0..type_len]);
        type_buf[type_len] = 0;

        const opts_len = @min(opts.len, 255);
        @memcpy(opts_buf[0..opts_len], opts[0..opts_len]);
        opts_buf[opts_len] = 0;

        mntbuf.mnt_fsname = @ptrCast(fsname_buf);
        mntbuf.mnt_dir = @ptrCast(dir_buf);
        mntbuf.mnt_type = @ptrCast(type_buf);
        mntbuf.mnt_opts = @ptrCast(opts_buf);
        mntbuf.mnt_freq = std.fmt.parseInt(c_int, freq_str, 10) catch 0;
        mntbuf.mnt_passno = std.fmt.parseInt(c_int, passno_str, 10) catch 0;

        return mntbuf;
    }

    return null;
}

/// Add mount entry to file (append mode)
pub export fn addmntent(stream_ptr: ?*anyopaque, mnt: *const mntent) c_int {
    if (stream_ptr == null) return 1;
    const stream: *MntStream = @ptrCast(@alignCast(stream_ptr));

    // Format: fsname dir type opts freq passno
    const fsname = std.mem.span(mnt.mnt_fsname);
    const dir = std.mem.span(mnt.mnt_dir);
    const fstype = std.mem.span(mnt.mnt_type);
    const opts = std.mem.span(mnt.mnt_opts);

    var buf: [1024]u8 = undefined;
    const written = std.fmt.bufPrint(&buf, "{s} {s} {s} {s} {d} {d}\n", .{
        fsname,
        dir,
        fstype,
        opts,
        mnt.mnt_freq,
        mnt.mnt_passno,
    }) catch return 1;

    _ = stream.file.write(written) catch return 1;
    return 0;
}

/// Close mount table file
pub export fn endmntent(stream_ptr: ?*anyopaque) c_int {
    if (stream_ptr == null) return 1;

    const allocator = gpa.allocator();
    const stream: *MntStream = @ptrCast(@alignCast(stream_ptr));

    stream.file.close();
    allocator.destroy(stream);

    return 1; // Always returns 1 per POSIX
}

/// Check if mount option is present
pub export fn hasmntopt(mnt: *const mntent, opt: [*:0]const u8) ?[*:0]u8 {
    const opts = std.mem.span(mnt.mnt_opts);
    const search = std.mem.span(opt);

    // Options are comma-separated
    var iter = std.mem.splitScalar(u8, opts, ',');
    while (iter.next()) |option| {
        if (std.mem.eql(u8, option, search)) {
            // Return pointer to option in original string
            const offset = @intFromPtr(option.ptr) - @intFromPtr(opts.ptr);
            return @ptrCast(@constCast(&mnt.mnt_opts[offset]));
        }
    }

    return null;
}
