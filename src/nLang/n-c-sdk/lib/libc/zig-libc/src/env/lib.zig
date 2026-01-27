// env module - Phase 1.18 - Environment variables
// Core implementations are in stdlib/environment.zig
// This module only provides additional extensions

const std = @import("std");

// Import core environment functions from stdlib
const environment = @import("../stdlib/environment.zig");

// Re-export core functions (they are already exported from environment.zig)
// getenv, setenv, unsetenv, putenv, clearenv are in stdlib/environment.zig

/// Secure version of getenv - returns null if running with elevated privileges
/// This is a security feature to prevent environment variable injection attacks
pub export fn secure_getenv(name: [*:0]const u8) ?[*:0]u8 {
    // Check if running with elevated privileges (setuid/setgid)
    // On most Unix systems, getuid() != geteuid() or getgid() != getegid() indicates this
    const uid = std.posix.getuid();
    const euid = std.posix.geteuid();
    const gid = std.posix.getgid();
    const egid = std.posix.getegid();

    // If running with elevated privileges, don't trust environment
    if (uid != euid or gid != egid) {
        return null;
    }

    // Otherwise, use regular getenv
    return environment.getenv(name);
}
