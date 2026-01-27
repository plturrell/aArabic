// Utility functions for stdlib
const std = @import("std");

const page_allocator = std.heap.page_allocator;

// Memory alignment functions - use rawAlloc with proper alignment
pub export fn aligned_alloc(alignment: usize, size: usize) ?*anyopaque {
    if (size == 0) return null;
    // Alignment must be a power of two
    if (!std.math.isPowerOfTwo(alignment)) return null;
    const align_enum = std.mem.Alignment.fromByteUnits(alignment);
    const ptr = page_allocator.rawAlloc(size, align_enum, @returnAddress()) orelse return null;
    // Track allocation for proper freeing
    const memory = @import("memory.zig");
    if (!memory.rememberAllocation(ptr, size)) {
        page_allocator.rawFree(ptr[0..size], align_enum, @returnAddress());
        return null;
    }
    return @ptrCast(ptr);
}

pub export fn posix_memalign(memptr: *?*anyopaque, alignment: usize, size: usize) c_int {
    // Alignment must be a power of two and a multiple of sizeof(void*)
    if (!std.math.isPowerOfTwo(alignment) or alignment < @sizeOf(*anyopaque)) {
        return 22; // EINVAL
    }
    if (size == 0) {
        memptr.* = null;
        return 0;
    }
    const align_enum = std.mem.Alignment.fromByteUnits(alignment);
    const ptr = page_allocator.rawAlloc(size, align_enum, @returnAddress()) orelse return 12; // ENOMEM
    // Track allocation for proper freeing
    const memory = @import("memory.zig");
    if (!memory.rememberAllocation(ptr, size)) {
        page_allocator.rawFree(ptr[0..size], align_enum, @returnAddress());
        return 12; // ENOMEM
    }
    memptr.* = @ptrCast(ptr);
    return 0;
}

// Path utilities - FIXED: Proper NUL termination
// Static buffers for "." and "/" with NUL terminators
const dot_string: [2:0]u8 = [_:0]u8{ '.', 0 };
const slash_string: [2:0]u8 = [_:0]u8{ '/', 0 };

pub export fn basename(path: [*:0]u8) [*:0]u8 {
    var i = std.mem.len(path);
    if (i == 0) return @constCast(@as([*:0]u8, @ptrCast(@constCast(&dot_string))));
    
    // Remove trailing slashes
    while (i > 0 and path[i - 1] == '/') i -= 1;
    if (i == 0) return @constCast(@as([*:0]u8, @ptrCast(@constCast(&slash_string))));
    
    // Find last slash
    var j = i;
    while (j > 0 and path[j - 1] != '/') j -= 1;
    
    return @ptrCast(&path[j]);
}

pub export fn dirname(path: [*:0]u8) [*:0]u8 {
    var i = std.mem.len(path);
    if (i == 0) return @constCast(@as([*:0]u8, @ptrCast(@constCast(&dot_string))));
    
    // Remove trailing slashes
    while (i > 0 and path[i - 1] == '/') i -= 1;
    if (i == 0) return @constCast(@as([*:0]u8, @ptrCast(@constCast(&slash_string))));
    
    // Find last slash
    while (i > 0 and path[i - 1] != '/') i -= 1;
    if (i == 0) return @constCast(@as([*:0]u8, @ptrCast(@constCast(&dot_string))));
    
    // Remove trailing slashes from result
    while (i > 1 and path[i - 1] == '/') i -= 1;
    path[i] = 0;
    return path;
}

// System execution - execute shell command using std.process.Child
pub export fn system(command: ?[*:0]const u8) c_int {
    if (command == null) return 1; // Shell available

    const cmd_slice = std.mem.span(command.?);

    // Spawn /bin/sh -c <command>
    var child = std.process.Child.init(&.{ "/bin/sh", "-c", cmd_slice }, page_allocator);
    child.spawn() catch return -1;

    const term = child.wait() catch return -1;

    return switch (term) {
        .Exited => |code| @as(c_int, code),
        .Signal => |sig| @as(c_int, @intCast(sig)) + 128,
        .Stopped => |sig| @as(c_int, @intCast(sig)) + 128,
        .Unknown => |val| @as(c_int, @intCast(val)),
    };
}

// Temporary files
pub export fn tmpnam(s: ?[*:0]u8) ?[*:0]u8 {
    const template = "/tmp/tmpXXXXXX";
    if (s) |buf| {
        @memcpy(buf[0..template.len], template);
        buf[template.len] = 0;
        return buf;
    }
    return null;
}

// More numeric
pub export fn strtoimax(nptr: [*:0]const u8, endptr: ?*?[*:0]u8, base: c_int) isize {
    return @intCast(@import("conversion.zig").strtoll(nptr, endptr, base));
}

pub export fn strtoumax(nptr: [*:0]const u8, endptr: ?*?[*:0]u8, base: c_int) usize {
    return @intCast(@import("numeric.zig").strtoull(nptr, endptr, base));
}

// Sorting helper
pub export fn lsearch(
    key: ?*const anyopaque,
    base: ?*anyopaque,
    nelp: *usize,
    width: usize,
    compar: *const fn (?*const anyopaque, ?*const anyopaque) callconv(.c) c_int,
) ?*anyopaque {
    if (key == null or base == null or width == 0) return null;

    const bytes: [*]u8 = @ptrCast(base);
    var i: usize = 0;
    const nel: usize = nelp.*;

    while (i < nel) : (i += 1) {
        const elem = &bytes[i * width];
        if (compar(key, elem) == 0) {
            return @ptrCast(elem);
        }
    }

    // Not found: append key and increase count (caller must ensure space)
    const dest_slice = bytes[nel * width ..][0..width];
    const key_bytes: [*]const u8 = @ptrCast(key.?);
    const key_slice = key_bytes[0..width];
    std.mem.copyForwards(u8, dest_slice, key_slice);
    nelp.* = nel + 1;
    return @ptrCast(dest_slice.ptr);
}

// Array realloc - FIXED: Use correct old size (nmemb * size)
pub export fn reallocarray(ptr: ?*anyopaque, nmemb: usize, size: usize) ?*anyopaque {
    // Check for overflow
    const total = std.math.mul(usize, nmemb, size) catch return null;
    
    // Use stdlib realloc which now properly tracks sizes
    const memory = @import("memory.zig");
    return memory.realloc(ptr, total);
}

// Quick utilities
// Note: clearenv is implemented in environment.zig

pub export fn getloadavg(loadavg: [*]f64, nelem: c_int) c_int {
    if (nelem <= 0) return -1;
    var i: usize = 0;
    while (i < @as(usize, @intCast(nelem))) : (i += 1) {
        loadavg[i] = 0.0;
    }
    return @intCast(nelem);
}

// File operations
pub export fn realpath(path: [*:0]const u8, resolved_path: ?[*:0]u8) ?[*:0]u8 {
    const len = std.mem.len(path);
    if (resolved_path) |rp| {
        @memcpy(rp[0..len], path[0..len]);
        rp[len] = 0;
        return rp;
    }
    return null;
}

// Random characters for mkstemp template substitution
const mkstemp_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

pub export fn mkstemp(template: [*:0]u8) c_int {
    const len = std.mem.len(template);
    if (len < 6) return -1;

    // Verify last 6 characters are 'X'
    const suffix_start = len - 6;
    for (template[suffix_start..len]) |c| {
        if (c != 'X') return -1;
    }

    // Get random seed (use timestamp-based approach)
    var prng = std.Random.DefaultPrng.init(@intCast(@as(i128, @truncate(std.time.nanoTimestamp()))));
    const random = prng.random();

    // Try to create unique file with random suffix
    var attempts: usize = 0;
    while (attempts < 100) : (attempts += 1) {
        // Generate random suffix
        for (0..6) |i| {
            template[suffix_start + i] = mkstemp_chars[random.intRangeAtMost(usize, 0, mkstemp_chars.len - 1)];
        }

        // Try to create file with O_CREAT | O_EXCL | O_RDWR using std.fs
        const path_slice = template[0..len :0];
        const file = std.fs.cwd().createFile(path_slice, .{
            .exclusive = true,
            .mode = 0o600,
        }) catch |err| {
            // If error is not PathAlreadyExists, fail
            if (err != error.PathAlreadyExists) {
                return -1;
            }
            continue;
        };
        return @intCast(file.handle);
    }
    return -1;
}

pub const FILE = opaque {};

// Maximum number of concurrent popen streams
const MAX_POPEN_STREAMS = 64;

// Structure to track popen streams and their child PIDs
const PopenEntry = struct {
    file: ?*FILE,
    pid: c_int,
};

// Static storage for popen tracking (no dynamic allocation needed)
var popen_entries: [MAX_POPEN_STREAMS]PopenEntry = [_]PopenEntry{.{ .file = null, .pid = -1 }} ** MAX_POPEN_STREAMS;
var popen_mutex = std.Thread.Mutex{};

// Internal PopenFile structure that wraps a file descriptor
const PopenFile = struct {
    fd: c_int,
    is_read: bool,
    buffer: [4096]u8,
    buf_pos: usize,
    buf_end: usize,
};

// Allocator for PopenFile structures
var popen_files: [MAX_POPEN_STREAMS]PopenFile = undefined;
var popen_file_used: [MAX_POPEN_STREAMS]bool = [_]bool{false} ** MAX_POPEN_STREAMS;

fn allocPopenFile() ?*PopenFile {
    for (&popen_files, 0..) |*pf, i| {
        if (!popen_file_used[i]) {
            popen_file_used[i] = true;
            return pf;
        }
    }
    return null;
}

fn freePopenFile(pf: *PopenFile) void {
    const base = @intFromPtr(&popen_files[0]);
    const target = @intFromPtr(pf);
    const idx = (target - base) / @sizeOf(PopenFile);
    if (idx < MAX_POPEN_STREAMS) {
        popen_file_used[idx] = false;
    }
}

// popen - open pipe to/from a command
pub export fn popen(command: [*:0]const u8, mode: [*:0]const u8) ?*FILE {
    const is_read = mode[0] == 'r';

    // Create pipe
    var pipefd: [2]c_int = undefined;
    const pipe_rc = std.posix.system.pipe(&pipefd);
    if (pipe_rc < 0) {
        return null;
    }

    // Fork process
    const pid = std.posix.system.fork();
    if (pid < 0) {
        _ = std.posix.system.close(pipefd[0]);
        _ = std.posix.system.close(pipefd[1]);
        return null;
    }

    if (pid == 0) {
        // Child process
        if (is_read) {
            // Parent reads from child's stdout
            _ = std.posix.system.close(pipefd[0]); // Close read end
            _ = std.posix.system.dup2(pipefd[1], 1); // Redirect stdout to pipe write end
            _ = std.posix.system.close(pipefd[1]);
        } else {
            // Parent writes to child's stdin
            _ = std.posix.system.close(pipefd[1]); // Close write end
            _ = std.posix.system.dup2(pipefd[0], 0); // Redirect stdin from pipe read end
            _ = std.posix.system.close(pipefd[0]);
        }

        // Execute command via shell: execve("/bin/sh", ["/bin/sh", "-c", command], environ)
        const shell: [*:0]const u8 = "/bin/sh";
        const sh_c: [*:0]const u8 = "-c";
        const argv = [_]?[*:0]const u8{ shell, sh_c, command, null };
        _ = std.posix.system.execve(shell, @ptrCast(&argv), @ptrCast(std.os.environ.ptr));
        std.posix.system.exit(127); // Exit if exec fails
    }

    // Parent process
    const parent_fd = if (is_read) pipefd[0] else pipefd[1];
    const close_fd = if (is_read) pipefd[1] else pipefd[0];
    _ = std.posix.system.close(close_fd);

    // Allocate PopenFile structure
    const pf = allocPopenFile() orelse {
        _ = std.posix.system.close(parent_fd);
        return null;
    };

    pf.* = PopenFile{
        .fd = parent_fd,
        .is_read = is_read,
        .buffer = undefined,
        .buf_pos = 0,
        .buf_end = 0,
    };

    const file_ptr: *FILE = @ptrCast(pf);

    // Track this popen stream for pclose
    popen_mutex.lock();
    defer popen_mutex.unlock();

    for (&popen_entries) |*entry| {
        if (entry.file == null) {
            entry.file = file_ptr;
            entry.pid = pid;
            break;
        }
    }

    return file_ptr;
}

// pclose - close pipe and wait for child process
pub export fn pclose(stream: ?*FILE) c_int {
    if (stream == null) return -1;

    popen_mutex.lock();

    // Find the pid for this stream
    var pid: c_int = -1;
    for (&popen_entries) |*entry| {
        if (entry.file == stream) {
            pid = entry.pid;
            entry.file = null;
            entry.pid = -1;
            break;
        }
    }

    popen_mutex.unlock();

    // Close the file descriptor
    const pf: *PopenFile = @ptrCast(@alignCast(stream));
    _ = std.posix.system.close(pf.fd);
    freePopenFile(pf);

    if (pid < 0) {
        return -1; // Stream was not from popen
    }

    // Wait for child process
    var status: c_int = 0;
    const wait_rc = std.posix.system.waitpid(pid, &status, 0);
    if (wait_rc < 0) {
        return -1;
    }

    return status;
}
