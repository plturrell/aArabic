// Environment variable functions
const std = @import("std");

const allocator = std.heap.page_allocator;

// Static storage for environment pointers (max environment entries)
const MAX_ENV_ENTRIES = 1024;
var env_storage: [MAX_ENV_ENTRIES:null]?[*:0]u8 = [_:null]?[*:0]u8{null} ** MAX_ENV_ENTRIES;
var env_initialized = false;

/// Initialize our environment storage from std.os.environ
fn initEnvStorage() void {
    if (env_initialized) return;

    var i: usize = 0;
    for (std.os.environ) |entry| {
        if (i >= MAX_ENV_ENTRIES - 1) break;
        env_storage[i] = entry;
        i += 1;
    }
    env_storage[i] = null;
    std.os.environ = env_storage[0..i];
    env_initialized = true;
}

/// Find index of environment variable by name
fn findEnvIndex(name: [*:0]const u8) ?usize {
    initEnvStorage();
    const name_len = std.mem.len(name);

    var i: usize = 0;
    while (env_storage[i]) |entry| : (i += 1) {
        const entry_span = std.mem.span(entry);
        if (entry_span.len > name_len and
            std.mem.eql(u8, entry_span[0..name_len], std.mem.span(name)) and
            entry[name_len] == '=') {
            return i;
        }
    }
    return null;
}

/// Count current environment entries
fn countEnvEntries() usize {
    var i: usize = 0;
    while (env_storage[i] != null) : (i += 1) {}
    return i;
}

/// Get environment variable
pub export fn getenv(name: [*:0]const u8) ?[*:0]u8 {
    initEnvStorage();
    const name_len = std.mem.len(name);

    var i: usize = 0;
    while (env_storage[i]) |entry| : (i += 1) {
        const entry_span = std.mem.span(entry);
        if (entry_span.len > name_len and
            std.mem.eql(u8, entry_span[0..name_len], std.mem.span(name)) and
            entry[name_len] == '=') {
            return entry + name_len + 1;
        }
    }
    return null;
}

/// Set environment variable
pub export fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int {
    initEnvStorage();

    const name_len = std.mem.len(name);
    const value_len = std.mem.len(value);

    // Check for invalid name (empty or contains '=')
    if (name_len == 0) return -1;
    for (std.mem.span(name)) |c| {
        if (c == '=') return -1;
    }

    // Check if variable already exists
    if (findEnvIndex(name)) |idx| {
        if (overwrite == 0) return 0; // Don't overwrite, success

        // Create new "NAME=value" string
        const total_len = name_len + 1 + value_len + 1; // name + '=' + value + null
        const new_entry = allocator.alloc(u8, total_len) catch return -1;
        @memcpy(new_entry[0..name_len], std.mem.span(name));
        new_entry[name_len] = '=';
        @memcpy(new_entry[name_len + 1 .. name_len + 1 + value_len], std.mem.span(value));
        new_entry[total_len - 1] = 0;

        env_storage[idx] = @ptrCast(new_entry.ptr);
        return 0;
    }

    // Add new variable
    const count = countEnvEntries();
    if (count >= MAX_ENV_ENTRIES - 1) return -1; // No room

    const total_len = name_len + 1 + value_len + 1;
    const new_entry = allocator.alloc(u8, total_len) catch return -1;
    @memcpy(new_entry[0..name_len], std.mem.span(name));
    new_entry[name_len] = '=';
    @memcpy(new_entry[name_len + 1 .. name_len + 1 + value_len], std.mem.span(value));
    new_entry[total_len - 1] = 0;

    env_storage[count] = @ptrCast(new_entry.ptr);
    env_storage[count + 1] = null;
    std.os.environ = env_storage[0 .. count + 1];
    return 0;
}

/// Remove environment variable
pub export fn unsetenv(name: [*:0]const u8) c_int {
    initEnvStorage();

    const name_len = std.mem.len(name);

    // Check for invalid name (empty or contains '=')
    if (name_len == 0) return -1;
    for (std.mem.span(name)) |c| {
        if (c == '=') return -1;
    }

    if (findEnvIndex(name)) |idx| {
        // Shift remaining entries down
        var i = idx;
        while (env_storage[i + 1] != null) : (i += 1) {
            env_storage[i] = env_storage[i + 1];
        }
        env_storage[i] = null;

        if (i > 0) {
            std.os.environ = env_storage[0..i];
        } else {
            std.os.environ = &[_][*:0]u8{};
        }
    }
    return 0;
}

/// Add environment variable (name=value string)
pub export fn putenv(string: [*:0]u8) c_int {
    initEnvStorage();

    const str_span = std.mem.span(string);

    // Find '=' in string
    const eq_pos = std.mem.indexOfScalar(u8, str_span, '=') orelse return -1;
    if (eq_pos == 0) return -1; // Empty name

    // Extract name for lookup
    const name = str_span[0..eq_pos];

    // Check if variable exists
    var i: usize = 0;
    while (env_storage[i]) |entry| : (i += 1) {
        const entry_span = std.mem.span(entry);
        if (entry_span.len > eq_pos and
            std.mem.eql(u8, entry_span[0..eq_pos], name) and
            entry[eq_pos] == '=') {
            // Replace existing entry with the provided string pointer
            env_storage[i] = string;
            return 0;
        }
    }

    // Add new entry
    const count = countEnvEntries();
    if (count >= MAX_ENV_ENTRIES - 1) return -1;

    env_storage[count] = string;
    env_storage[count + 1] = null;
    std.os.environ = env_storage[0 .. count + 1];
    return 0;
}

/// Clear all environment variables
pub export fn clearenv() c_int {
    initEnvStorage();

    // Set first entry to null, making the environment empty
    env_storage[0] = null;
    std.os.environ = &[_][*:0]u8{};
    return 0;
}