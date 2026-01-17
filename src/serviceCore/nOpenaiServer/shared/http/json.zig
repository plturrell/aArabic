// Zig JSON Parser for Mojo
// Provides JSON capabilities that Mojo stdlib doesn't have yet
// High-performance JSON parsing using Zig's std.json

const std = @import("std");
const json = std.json;
const mem = std.mem;

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Parse JSON string and extract field
/// Returns null-terminated string for C ABI
export fn zig_json_get_string(
    json_str: [*:0]const u8,
    field: [*:0]const u8
) callconv(.c) [*:0]const u8 {
    const json_data = mem.span(json_str);
    const field_name = mem.span(field);
    
    const result = jsonGetString(json_data, field_name) catch |err| {
        std.debug.print("JSON parse error: {any}\n", .{err});
        return "";
    };
    
    return result.ptr;
}

fn jsonGetString(json_data: []const u8, field: []const u8) ![:0]const u8 {
    // Parse JSON
    const parsed = try json.parseFromSlice(
        json.Value,
        allocator,
        json_data,
        .{}
    );
    defer parsed.deinit();
    
    // Get field
    const obj = parsed.value.object;
    const value = obj.get(field) orelse return error.FieldNotFound;
    
    // Convert to string
    const str = value.string;
    const result = try allocator.allocSentinel(u8, str.len, 0);
    @memcpy(result[0..str.len], str);
    
    return result;
}

/// Parse JSON and extract integer field
export fn zig_json_get_int(
    json_str: [*:0]const u8,
    field: [*:0]const u8
) callconv(.c) i64 {
    const json_data = mem.span(json_str);
    const field_name = mem.span(field);
    
    return jsonGetInt(json_data, field_name) catch 0;
}

fn jsonGetInt(json_data: []const u8, field: []const u8) !i64 {
    const parsed = try json.parseFromSlice(
        json.Value,
        allocator,
        json_data,
        .{}
    );
    defer parsed.deinit();
    
    const obj = parsed.value.object;
    const value = obj.get(field) orelse return error.FieldNotFound;
    
    return value.integer;
}

/// Parse JSON and extract float field
export fn zig_json_get_float(
    json_str: [*:0]const u8,
    field: [*:0]const u8
) callconv(.c) f64 {
    const json_data = mem.span(json_str);
    const field_name = mem.span(field);
    
    return jsonGetFloat(json_data, field_name) catch 0.0;
}

fn jsonGetFloat(json_data: []const u8, field: []const u8) !f64 {
    const parsed = try json.parseFromSlice(
        json.Value,
        allocator,
        json_data,
        .{}
    );
    defer parsed.deinit();
    
    const obj = parsed.value.object;
    const value = obj.get(field) orelse return error.FieldNotFound;
    
    return value.float;
}

/// Parse JSON array of floats (for embeddings)
export fn zig_json_get_array(
    json_str: [*:0]const u8,
    field: [*:0]const u8,
    out_len: *usize
) callconv(.c) [*]f32 {
    const json_data = mem.span(json_str);
    const field_name = mem.span(field);
    
    const result = jsonGetArray(json_data, field_name, out_len) catch {
        out_len.* = 0;
        return @constCast(@ptrCast(&[_]f32{}));
    };
    
    return result.ptr;
}

fn jsonGetArray(json_data: []const u8, field: []const u8, out_len: *usize) ![]f32 {
    const parsed = try json.parseFromSlice(
        json.Value,
        allocator,
        json_data,
        .{}
    );
    defer parsed.deinit();
    
    const obj = parsed.value.object;
    const value = obj.get(field) orelse return error.FieldNotFound;
    
    const arr = value.array;
    out_len.* = arr.items.len;
    
    // Allocate array
    const result = try allocator.alloc(f32, arr.items.len);
    
    // Convert to f32
    for (arr.items, 0..) |item, i| {
        result[i] = @floatCast(item.float);
    }
    
    return result;
}

/// Create JSON object (simple key-value) - Custom implementation for Zig 0.15.2
export fn zig_json_create_object(
    keys: [*][*:0]const u8,
    values: [*][*:0]const u8,
    count: usize
) callconv(.c) [*:0]const u8 {
    const result = jsonCreateObject(keys, values, count) catch {
        return "{}";
    };
    
    return result.ptr;
}

fn jsonCreateObject(
    keys: [*][*:0]const u8,
    values: [*][*:0]const u8,
    count: usize
) ![:0]const u8 {
    // Manual JSON construction to avoid Zig 0.15.2 API issues
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);
    
    try buffer.append(allocator, '{');
    
    for (0..count) |i| {
        if (i > 0) {
            try buffer.append(allocator, ',');
        }
        
        const key = mem.span(keys[i]);
        const value = mem.span(values[i]);
        
        // Add key
        try buffer.append(allocator, '"');
        try buffer.appendSlice(allocator, key);
        try buffer.appendSlice(allocator, "\":");
        
        // Add value (assume string for now)
        try buffer.append(allocator, '"');
        try buffer.appendSlice(allocator, value);
        try buffer.append(allocator, '"');
    }
    
    try buffer.append(allocator, '}');
    
    // Return as null-terminated string
    const result = try allocator.allocSentinel(u8, buffer.items.len, 0);
    @memcpy(result[0..buffer.items.len], buffer.items);
    
    return result;
}

/// Stringify JSON value - Simple custom implementation
export fn zig_json_stringify(value: *const json.Value) callconv(.c) [*:0]const u8 {
    const result = jsonStringifyCustom(value) catch {
        return "{}";
    };
    
    return result.ptr;
}

fn jsonStringifyCustom(value: *const json.Value) ![:0]const u8 {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);
    
    try stringifyValue(value.*, &buffer);
    
    const result = try allocator.allocSentinel(u8, buffer.items.len, 0);
    @memcpy(result[0..buffer.items.len], buffer.items);
    
    return result;
}

fn stringifyValue(value: json.Value, buffer: *std.ArrayList(u8)) !void {
    switch (value) {
        .null => try buffer.appendSlice(allocator, "null"),
        .bool => |b| try buffer.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| try std.fmt.format(buffer.writer(allocator), "{d}", .{i}),
        .float => |f| try std.fmt.format(buffer.writer(allocator), "{d}", .{f}),
        .number_string => |s| try buffer.appendSlice(allocator, s),
        .string => |s| {
            try buffer.append(allocator, '"');
            try buffer.appendSlice(allocator, s);
            try buffer.append(allocator, '"');
        },
        .array => |arr| {
            try buffer.append(allocator, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try buffer.append(allocator, ',');
                try stringifyValue(item, buffer);
            }
            try buffer.append(allocator, ']');
        },
        .object => |obj| {
            try buffer.append(allocator, '{');
            var it = obj.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try buffer.append(allocator, ',');
                first = false;
                
                try buffer.append(allocator, '"');
                try buffer.appendSlice(allocator, entry.key_ptr.*);
                try buffer.appendSlice(allocator, "\":");
                try stringifyValue(entry.value_ptr.*, buffer);
            }
            try buffer.append(allocator, '}');
        },
    }
}

// For testing
pub fn main() !void {
    std.debug.print("🧪 Zig JSON Library Test\n", .{});
    
    // Test parsing
    const test_json = 
        \\{"query":"test","top_k":10,"score":0.95}
    ;
    
    const query = try jsonGetString(test_json, "query");
    const top_k = try jsonGetInt(test_json, "top_k");
    const score = try jsonGetFloat(test_json, "score");
    
    std.debug.print("Query: {s}\n", .{query});
    std.debug.print("Top K: {d}\n", .{top_k});
    std.debug.print("Score: {d}\n", .{score});
    
    std.debug.print("\n✅ JSON parsing works!\n", .{});
    std.debug.print("Build with: zig build-lib zig_json.zig -dynamic\n", .{});
}
