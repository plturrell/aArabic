// Generic JSON Parser for Mojo SDK
// Zero Python dependencies - pure Zig std.json

const std = @import("std");

// Global allocator for memory management
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// Parse JSON file and return as minified string
/// Returns error string if parsing fails
export fn zig_json_parse_file(path: [*:0]const u8) [*:0]const u8 {
    const allocator = gpa.allocator();
    
    // Open and read file
    const file = std.fs.cwd().openFile(
        std.mem.span(path), 
        .{}
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: File not found") catch unreachable;
        return err.ptr;
    };
    defer file.close();
    
    const content = file.readToEndAlloc(
        allocator, 
        10_000_000  // 10MB max file size
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Failed to read file") catch unreachable;
        return err.ptr;
    };
    defer allocator.free(content);
    
    // Parse JSON
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Invalid JSON syntax") catch unreachable;
        return err.ptr;
    };
    defer parsed.deinit();
    
    // Stringify to minified JSON (validates structure)
    const result = std.json.stringifyAlloc(
        allocator,
        parsed.value,
        .{ .whitespace = .minified }
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Failed to stringify JSON") catch unreachable;
        return err.ptr;
    };
    
    // Convert to null-terminated string
    const result_z = allocator.dupeZ(u8, result) catch unreachable;
    allocator.free(result);
    
    return result_z.ptr;
}

/// Parse JSON string and return as minified string
/// Returns error string if parsing fails
export fn zig_json_parse_string(
    json_ptr: [*:0]const u8,
    len: usize
) [*:0]const u8 {
    const allocator = gpa.allocator();
    
    const content = json_ptr[0..len];
    
    // Parse JSON
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Invalid JSON syntax") catch unreachable;
        return err.ptr;
    };
    defer parsed.deinit();
    
    // Stringify to minified JSON
    const result = std.json.stringifyAlloc(
        allocator,
        parsed.value,
        .{ .whitespace = .minified }
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Failed to stringify JSON") catch unreachable;
        return err.ptr;
    };
    
    // Convert to null-terminated string
    const result_z = allocator.dupeZ(u8, result) catch unreachable;
    allocator.free(result);
    
    return result_z.ptr;
}

/// Validate JSON without parsing full structure
export fn zig_json_validate(json_ptr: [*:0]const u8, len: usize) bool {
    const allocator = gpa.allocator();
    const content = json_ptr[0..len];
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch return false;
    
    parsed.deinit();
    return true;
}

/// Free memory allocated by Zig
export fn zig_json_free(ptr: [*:0]const u8) void {
    const allocator = gpa.allocator();
    const slice = std.mem.span(ptr);
    allocator.free(slice);
}

/// Get JSON object value by key
export fn zig_json_get_value(
    json_ptr: [*:0]const u8,
    json_len: usize,
    key_ptr: [*:0]const u8
) [*:0]const u8 {
    const allocator = gpa.allocator();
    const content = json_ptr[0..json_len];
    const key = std.mem.span(key_ptr);
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Invalid JSON") catch unreachable;
        return err.ptr;
    };
    defer parsed.deinit();
    
    // Get object
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => {
            const err = allocator.dupeZ(u8, "ERROR: Not a JSON object") catch unreachable;
            return err.ptr;
        },
    };
    
    // Get value by key
    const value = obj.get(key) orelse {
        const err = allocator.dupeZ(u8, "ERROR: Key not found") catch unreachable;
        return err.ptr;
    };
    
    // Stringify value
    const result = std.json.stringifyAlloc(
        allocator,
        value,
        .{ .whitespace = .minified }
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Failed to stringify value") catch unreachable;
        return err.ptr;
    };
    
    const result_z = allocator.dupeZ(u8, result) catch unreachable;
    allocator.free(result);
    
    return result_z.ptr;
}

/// Test function for debugging
export fn zig_json_test() [*:0]const u8 {
    const allocator = gpa.allocator();
    const msg = allocator.dupeZ(u8, "JSON parser initialized") catch unreachable;
    return msg.ptr;
}

/// Get all keys from JSON object
/// Returns comma-separated list of keys
export fn zig_json_get_keys(
    json_ptr: [*:0]const u8,
    json_len: usize
) [*:0]const u8 {
    const allocator = gpa.allocator();
    const content = json_ptr[0..json_len];
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Invalid JSON") catch unreachable;
        return err.ptr;
    };
    defer parsed.deinit();
    
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => {
            const err = allocator.dupeZ(u8, "ERROR: Not a JSON object") catch unreachable;
            return err.ptr;
        },
    };
    
    // Build comma-separated list of keys
    var keys_list = std.ArrayList(u8).init(allocator);
    defer keys_list.deinit();
    
    var iterator = obj.iterator();
    var first = true;
    while (iterator.next()) |entry| {
        if (!first) {
            keys_list.appendSlice(",") catch unreachable;
        }
        keys_list.appendSlice(entry.key_ptr.*) catch unreachable;
        first = false;
    }
    
    const result = allocator.dupeZ(u8, keys_list.items) catch unreachable;
    return result.ptr;
}

/// Get number of keys in JSON object
export fn zig_json_get_key_count(
    json_ptr: [*:0]const u8,
    json_len: usize
) usize {
    const allocator = gpa.allocator();
    const content = json_ptr[0..json_len];
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch return 0;
    defer parsed.deinit();
    
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return 0,
    };
    
    return obj.count();
}

/// Check if JSON object has key
export fn zig_json_has_key(
    json_ptr: [*:0]const u8,
    json_len: usize,
    key_ptr: [*:0]const u8
) bool {
    const allocator = gpa.allocator();
    const content = json_ptr[0..json_len];
    const key = std.mem.span(key_ptr);
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch return false;
    defer parsed.deinit();
    
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return false,
    };
    
    return obj.contains(key);
}

/// Get array length
export fn zig_json_get_array_length(
    json_ptr: [*:0]const u8,
    json_len: usize
) usize {
    const allocator = gpa.allocator();
    const content = json_ptr[0..json_len];
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch return 0;
    defer parsed.deinit();
    
    const arr = switch (parsed.value) {
        .array => |a| a,
        else => return 0,
    };
    
    return arr.items.len;
}

/// Get array item at index
export fn zig_json_get_array_item(
    json_ptr: [*:0]const u8,
    json_len: usize,
    index: usize
) [*:0]const u8 {
    const allocator = gpa.allocator();
    const content = json_ptr[0..json_len];
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Invalid JSON") catch unreachable;
        return err.ptr;
    };
    defer parsed.deinit();
    
    const arr = switch (parsed.value) {
        .array => |a| a,
        else => {
            const err = allocator.dupeZ(u8, "ERROR: Not a JSON array") catch unreachable;
            return err.ptr;
        },
    };
    
    if (index >= arr.items.len) {
        const err = allocator.dupeZ(u8, "ERROR: Index out of bounds") catch unreachable;
        return err.ptr;
    }
    
    const result = std.json.stringifyAlloc(
        allocator,
        arr.items[index],
        .{ .whitespace = .minified }
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Failed to stringify item") catch unreachable;
        return err.ptr;
    };
    
    const result_z = allocator.dupeZ(u8, result) catch unreachable;
    allocator.free(result);
    
    return result_z.ptr;
}

/// Get nested value by path (e.g., "graphs.supply_chain.nodes")
export fn zig_json_get_nested_value(
    json_ptr: [*:0]const u8,
    json_len: usize,
    path_ptr: [*:0]const u8
) [*:0]const u8 {
    const allocator = gpa.allocator();
    const content = json_ptr[0..json_len];
    const path = std.mem.span(path_ptr);
    
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        content,
        .{}
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Invalid JSON") catch unreachable;
        return err.ptr;
    };
    defer parsed.deinit();
    
    // Split path by '.'
    var current_value = parsed.value;
    var path_parts = std.mem.split(u8, path, ".");
    
    while (path_parts.next()) |part| {
        current_value = switch (current_value) {
            .object => |obj| blk: {
                const val = obj.get(part) orelse {
                    const err = allocator.dupeZ(u8, "ERROR: Path not found") catch unreachable;
                    return err.ptr;
                };
                break :blk val.*;
            },
            else => {
                const err = allocator.dupeZ(u8, "ERROR: Not an object at path") catch unreachable;
                return err.ptr;
            },
        };
    }
    
    const result = std.json.stringifyAlloc(
        allocator,
        current_value,
        .{ .whitespace = .minified }
    ) catch {
        const err = allocator.dupeZ(u8, "ERROR: Failed to stringify value") catch unreachable;
        return err.ptr;
    };
    
    const result_z = allocator.dupeZ(u8, result) catch unreachable;
    allocator.free(result);
    
    return result_z.ptr;
}
