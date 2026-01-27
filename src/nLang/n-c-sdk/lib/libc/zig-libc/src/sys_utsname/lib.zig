// sys/utsname module - Phase 1.13
// System identification using real syscalls
const std = @import("std");
const builtin = @import("builtin");
const fs = std.fs;

// Size differs between Linux and macOS
const UTSNAME_LENGTH = if (builtin.os.tag == .macos) 256 else 65;

pub const utsname = extern struct {
    sysname: [UTSNAME_LENGTH]u8, // OS name
    nodename: [UTSNAME_LENGTH]u8, // Network node hostname
    release: [UTSNAME_LENGTH]u8, // OS release
    version: [UTSNAME_LENGTH]u8, // OS version
    machine: [UTSNAME_LENGTH]u8, // Hardware identifier
};

/// Get system identification
pub export fn uname(buf: *utsname) c_int {
    // Zero out the buffer first
    @memset(&buf.sysname, 0);
    @memset(&buf.nodename, 0);
    @memset(&buf.release, 0);
    @memset(&buf.version, 0);
    @memset(&buf.machine, 0);

    if (builtin.os.tag == .linux) {
        return unameLinux(buf);
    } else if (builtin.os.tag == .macos or builtin.os.tag == .ios) {
        return unameMacos(buf);
    } else {
        return unameFallback(buf);
    }
}

fn unameLinux(buf: *utsname) c_int {
    // Linux: use uname syscall directly
    const rc = std.os.linux.syscall(.uname, .{@intFromPtr(buf)});
    if (@as(isize, @bitCast(rc)) < 0) {
        return unameFallback(buf);
    }
    return 0;
}

fn unameMacos(buf: *utsname) c_int {
    // macOS: use sysctl to get system info
    const sysname = getSysctl("kern.ostype") orelse "Darwin";
    const nodename = getSysctl("kern.hostname") orelse "localhost";
    const release = getSysctl("kern.osrelease") orelse "unknown";
    const version = getSysctl("kern.version") orelse "unknown";
    const machine = getMachine();

    copyToBuf(&buf.sysname, sysname);
    copyToBuf(&buf.nodename, nodename);
    copyToBuf(&buf.release, release);
    copyToBuf(&buf.version, version);
    copyToBuf(&buf.machine, machine);

    return 0;
}

fn unameFallback(buf: *utsname) c_int {
    // Compile-time known values
    const sysname = switch (builtin.os.tag) {
        .linux => "Linux",
        .macos => "Darwin",
        .windows => "Windows",
        .freebsd => "FreeBSD",
        .netbsd => "NetBSD",
        .openbsd => "OpenBSD",
        else => "Unknown",
    };

    const machine = switch (builtin.cpu.arch) {
        .x86_64 => "x86_64",
        .x86 => "i686",
        .aarch64 => "aarch64",
        .arm => "arm",
        .riscv64 => "riscv64",
        .powerpc64le => "ppc64le",
        else => "unknown",
    };

    copyToBuf(&buf.sysname, sysname);
    copyToBuf(&buf.nodename, getHostname());
    copyToBuf(&buf.release, "0.0.0");
    copyToBuf(&buf.version, "#1");
    copyToBuf(&buf.machine, machine);

    return 0;
}

fn copyToBuf(dest: []u8, src: []const u8) void {
    const len = @min(src.len, dest.len - 1);
    @memcpy(dest[0..len], src[0..len]);
    dest[len] = 0;
}

fn getHostname() []const u8 {
    // Try to read from /etc/hostname
    const file = fs.openFileAbsolute("/etc/hostname", .{}) catch return "localhost";
    defer file.close();

    var buf: [256]u8 = undefined;
    const len = file.read(&buf) catch return "localhost";
    if (len == 0) return "localhost";

    // Trim trailing newline
    var end = len;
    while (end > 0 and (buf[end - 1] == '\n' or buf[end - 1] == '\r')) {
        end -= 1;
    }

    if (end == 0) return "localhost";
    return buf[0..end];
}

fn getMachine() []const u8 {
    return switch (builtin.cpu.arch) {
        .x86_64 => "x86_64",
        .x86 => "i686",
        .aarch64 => "arm64",
        .arm => "arm",
        else => "unknown",
    };
}

fn getSysctl(name: []const u8) ?[]const u8 {
    // Would need sysctl implementation
    // For now return null to use defaults
    _ = name;
    return null;
}

/// Get hostname (gethostname compatibility)
pub export fn gethostname(name: [*]u8, len: usize) c_int {
    var buf: utsname = undefined;
    if (uname(&buf) != 0) return -1;

    var i: usize = 0;
    while (i < len - 1 and i < UTSNAME_LENGTH and buf.nodename[i] != 0) : (i += 1) {
        name[i] = buf.nodename[i];
    }
    name[i] = 0;

    return 0;
}

/// Set hostname (sethostname - requires root)
pub export fn sethostname(name: [*]const u8, len: usize) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.sethostname, .{
            @intFromPtr(name),
            len,
        });
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return 0;
    }
    return -1; // EPERM on non-Linux
}
