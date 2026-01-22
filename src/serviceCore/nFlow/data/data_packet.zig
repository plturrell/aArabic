const std = @import("std");
const Allocator = std.mem.Allocator;

/// Data type enumeration for typed data packets
pub const DataType = enum {
    string,
    number,
    boolean,
    object,
    array,
    binary,
    null_type,

    pub fn toString(self: DataType) []const u8 {
        return switch (self) {
            .string => "string",
            .number => "number",
            .boolean => "boolean",
            .object => "object",
            .array => "array",
            .binary => "binary",
            .null_type => "null",
        };
    }

    pub fn fromString(str: []const u8) !DataType {
        if (std.mem.eql(u8, str, "string")) return .string;
        if (std.mem.eql(u8, str, "number")) return .number;
        if (std.mem.eql(u8, str, "boolean")) return .boolean;
        if (std.mem.eql(u8, str, "object")) return .object;
        if (std.mem.eql(u8, str, "array")) return .array;
        if (std.mem.eql(u8, str, "binary")) return .binary;
        if (std.mem.eql(u8, str, "null")) return .null_type;
        return error.InvalidDataType;
    }
};

/// Data packet representing typed data flowing through workflows
pub const DataPacket = struct {
    allocator: Allocator,
    id: []const u8,
    data_type: DataType,
    value: std.json.Value,
    metadata: std.StringHashMap([]const u8),
    timestamp: i64,
    owns_value: bool, // Track if we need to free the value

    /// Initialize a new data packet
    pub fn init(allocator: Allocator, id: []const u8, data_type: DataType, value: std.json.Value) !*DataPacket {
        const packet = try allocator.create(DataPacket);
        errdefer allocator.destroy(packet);

        packet.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .data_type = data_type,
            .value = value,
            .metadata = std.StringHashMap([]const u8).init(allocator),
            .timestamp = std.time.milliTimestamp(),
            .owns_value = false, // By default, we don't own the value
        };

        return packet;
    }

    /// Clean up resources
    pub fn deinit(self: *DataPacket) void {
        self.allocator.free(self.id);
        
        // Only free the JSON value if we own it (from deserialization)
        if (self.owns_value) {
            freeJsonValue(self.allocator, self.value);
        }
        
        // Free metadata
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
        
        self.allocator.destroy(self);
    }

    /// Set metadata key-value pair
    pub fn setMetadata(self: *DataPacket, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        
        // Free old value if exists
        if (self.metadata.get(key)) |old_value| {
            self.allocator.free(old_value);
        }
        
        try self.metadata.put(key_copy, value_copy);
    }

    /// Get metadata value by key
    pub fn getMetadata(self: *const DataPacket, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    /// Serialize to JSON string
    pub fn serialize(self: *const DataPacket) ![]const u8 {
        var string = std.ArrayList(u8){};
        defer string.deinit(self.allocator);

        var writer = string.writer(self.allocator);
        
        try writer.writeAll("{\"id\":\"");
        try writer.writeAll(self.id);
        try writer.writeAll("\",\"type\":\"");
        try writer.writeAll(self.data_type.toString());
        try writer.writeAll("\",\"value\":");
        
        // Stringify the JSON value
        const value_json = try std.json.Stringify.valueAlloc(self.allocator, self.value, .{});
        defer self.allocator.free(value_json);
        try writer.writeAll(value_json);
        
        try writer.writeAll(",\"metadata\":{");
        var first = true;
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            if (!first) try writer.writeAll(",");
            first = false;
            try writer.writeAll("\"");
            try writer.writeAll(entry.key_ptr.*);
            try writer.writeAll("\":\"");
            try writer.writeAll(entry.value_ptr.*);
            try writer.writeAll("\"");
        }
        try writer.writeAll("},\"timestamp\":");
        try writer.print("{d}", .{self.timestamp});
        try writer.writeAll("}");

        return string.toOwnedSlice(self.allocator);
    }

    /// Deserialize from JSON string
    pub fn deserialize(allocator: Allocator, data: []const u8) !*DataPacket {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, data, .{});
        defer parsed.deinit();

        const obj = parsed.value.object;
        
        const id = obj.get("id").?.string;
        const type_str = obj.get("type").?.string;
        const data_type = try DataType.fromString(type_str);
        const parsed_value = obj.get("value").?;
        
        // Deep copy the value to avoid use-after-free when parsed is deinit'd
        const value = try deepCopyJsonValue(allocator, parsed_value);
        
        // Handle number as float
        const timestamp_value = obj.get("timestamp").?;
        const timestamp = if (timestamp_value == .float) 
            @as(i64, @intFromFloat(timestamp_value.float))
        else if (timestamp_value == .integer)
            timestamp_value.integer
        else
            return error.InvalidTimestamp;

        const packet = try DataPacket.init(allocator, id, data_type, value);
        packet.timestamp = timestamp;
        packet.owns_value = true; // We own this value since we deep-copied it

        // Restore metadata
        if (obj.get("metadata")) |meta_obj| {
            var it = meta_obj.object.iterator();
            while (it.next()) |entry| {
                try packet.setMetadata(entry.key_ptr.*, entry.value_ptr.*.string);
            }
        }

        return packet;
    }
    
    /// Free JSON value's allocated memory
    fn freeJsonValue(allocator: Allocator, value: std.json.Value) void {
        switch (value) {
            .null, .bool, .integer, .float => {},
            .number_string => |ns| allocator.free(ns),
            .string => |s| allocator.free(s),
            .array => |arr| {
                for (arr.items) |item| {
                    freeJsonValue(allocator, item);
                }
                // Array has allocator set, so deinit() doesn't need it
                var mutable_arr = arr;
                mutable_arr.deinit();
            },
            .object => |obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    freeJsonValue(allocator, entry.value_ptr.*);
                }
                var mutable_obj = obj;
                mutable_obj.deinit();
            },
        }
    }
    
    /// Deep copy a JSON value to avoid memory lifetime issues
    fn deepCopyJsonValue(allocator: Allocator, value: std.json.Value) !std.json.Value {
        return switch (value) {
            .null => .null,
            .bool => |b| .{ .bool = b },
            .integer => |i| .{ .integer = i },
            .float => |f| .{ .float = f },
            .number_string => |ns| .{ .number_string = try allocator.dupe(u8, ns) },
            .string => |s| .{ .string = try allocator.dupe(u8, s) },
            .array => |arr| {
                var new_array = std.json.Array{
                    .items = &.{},
                    .capacity = 0,
                    .allocator = allocator,
                };
                for (arr.items) |item| {
                    const copied_item = try deepCopyJsonValue(allocator, item);
                    try new_array.append(copied_item);
                }
                return .{ .array = new_array };
            },
            .object => |obj| {
                var new_obj = std.json.ObjectMap.init(allocator);
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
                    const value_copy = try deepCopyJsonValue(allocator, entry.value_ptr.*);
                    try new_obj.put(key_copy, value_copy);
                }
                return .{ .object = new_obj };
            },
        };
    }

    /// Validate against schema
    pub fn validate(self: *const DataPacket, schema: *const DataSchema) !void {
        // Check type matches
        if (self.data_type != schema.type) {
            return error.TypeMismatch;
        }

        // Check required
        if (schema.required and self.value == .null) {
            return error.RequiredValueMissing;
        }

        // Apply constraints if present
        if (schema.constraints) |constraints| {
            try self.validateConstraints(constraints);
        }
    }

    fn validateConstraints(self: *const DataPacket, constraints: SchemaConstraints) !void {
        switch (constraints) {
            .string_constraints => |sc| {
                if (self.value != .string) return error.TypeMismatch;
                const str = self.value.string;
                
                if (sc.min_length) |min| {
                    if (str.len < min) return error.StringTooShort;
                }
                if (sc.max_length) |max| {
                    if (str.len > max) return error.StringTooLong;
                }
                // Pattern validation would require regex support
            },
            .number_constraints => |nc| {
                // Check for float or integer types
                const num = if (self.value == .float) 
                    self.value.float
                else if (self.value == .integer)
                    @as(f64, @floatFromInt(self.value.integer))
                else
                    return error.TypeMismatch;
                
                if (nc.min) |min| {
                    if (num < min) return error.NumberTooSmall;
                }
                if (nc.max) |max| {
                    if (num > max) return error.NumberTooLarge;
                }
            },
            .array_constraints => |ac| {
                if (self.value != .array) return error.TypeMismatch;
                const arr = self.value.array;
                
                if (ac.min_items) |min| {
                    if (arr.items.len < min) return error.ArrayTooSmall;
                }
                if (ac.max_items) |max| {
                    if (arr.items.len > max) return error.ArrayTooLarge;
                }
                // Item schema validation would require recursive validation
            },
            .object_constraints => |oc| {
                if (self.value != .object) return error.TypeMismatch;
                const obj = self.value.object;
                
                // Check required properties
                for (oc.required_properties) |prop| {
                    if (!obj.contains(prop)) return error.RequiredPropertyMissing;
                }
                // Property schema validation would require recursive validation
            },
        }
    }
};

/// Schema definition for data validation
pub const DataSchema = struct {
    type: DataType,
    required: bool,
    constraints: ?SchemaConstraints,

    pub fn init(data_type: DataType, required: bool, constraints: ?SchemaConstraints) DataSchema {
        return .{
            .type = data_type,
            .required = required,
            .constraints = constraints,
        };
    }
};

/// Schema constraints for different data types
pub const SchemaConstraints = union(enum) {
    string_constraints: StringConstraints,
    number_constraints: NumberConstraints,
    array_constraints: ArrayConstraints,
    object_constraints: ObjectConstraints,
};

pub const StringConstraints = struct {
    min_length: ?usize = null,
    max_length: ?usize = null,
    pattern: ?[]const u8 = null,
};

pub const NumberConstraints = struct {
    min: ?f64 = null,
    max: ?f64 = null,
};

pub const ArrayConstraints = struct {
    min_items: ?usize = null,
    max_items: ?usize = null,
    item_schema: ?*const DataSchema = null,
};

pub const ObjectConstraints = struct {
    properties: std.StringHashMap(DataSchema),
    required_properties: []const []const u8,
};

// ============================================================================
// TESTS
// ============================================================================

test "DataType toString and fromString" {
    try std.testing.expectEqualStrings("string", DataType.string.toString());
    try std.testing.expectEqualStrings("number", DataType.number.toString());
    try std.testing.expectEqualStrings("boolean", DataType.boolean.toString());
    
    try std.testing.expectEqual(DataType.string, try DataType.fromString("string"));
    try std.testing.expectEqual(DataType.number, try DataType.fromString("number"));
    try std.testing.expectEqual(DataType.boolean, try DataType.fromString("boolean"));
    try std.testing.expectError(error.InvalidDataType, DataType.fromString("invalid"));
}

test "DataPacket creation and cleanup" {
    const allocator = std.testing.allocator;
    
    const value = std.json.Value{ .string = "test value" };
    const packet = try DataPacket.init(allocator, "packet-1", .string, value);
    defer packet.deinit();
    
    try std.testing.expectEqualStrings("packet-1", packet.id);
    try std.testing.expectEqual(DataType.string, packet.data_type);
    try std.testing.expectEqualStrings("test value", packet.value.string);
}

test "DataPacket metadata operations" {
    const allocator = std.testing.allocator;
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "packet-1", .string, value);
    defer packet.deinit();
    
    try packet.setMetadata("source", "http_request");
    try packet.setMetadata("user_id", "user-123");
    
    try std.testing.expectEqualStrings("http_request", packet.getMetadata("source").?);
    try std.testing.expectEqualStrings("user-123", packet.getMetadata("user_id").?);
    try std.testing.expect(packet.getMetadata("nonexistent") == null);
}

test "DataPacket serialization" {
    const allocator = std.testing.allocator;
    
    const value = std.json.Value{ .string = "test value" };
    const packet = try DataPacket.init(allocator, "packet-1", .string, value);
    defer packet.deinit();
    
    try packet.setMetadata("key", "value");
    
    const serialized = try packet.serialize();
    defer allocator.free(serialized);
    
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"id\":\"packet-1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"type\":\"string\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "test value") != null);
}

test "DataPacket deserialization" {
    const allocator = std.testing.allocator;
    
    const json_str = "{\"id\":\"packet-1\",\"type\":\"string\",\"value\":\"test\",\"metadata\":{\"key\":\"value\"},\"timestamp\":1234567890}";
    
    const packet = try DataPacket.deserialize(allocator, json_str);
    defer packet.deinit();
    
    try std.testing.expectEqualStrings("packet-1", packet.id);
    try std.testing.expectEqual(DataType.string, packet.data_type);
    try std.testing.expectEqualStrings("test", packet.value.string);
    try std.testing.expectEqual(@as(i64, 1234567890), packet.timestamp);
    try std.testing.expectEqualStrings("value", packet.getMetadata("key").?);
}

test "DataSchema string validation - success" {
    const allocator = std.testing.allocator;
    
    const schema = DataSchema.init(.string, true, .{
        .string_constraints = .{
            .min_length = 3,
            .max_length = 10,
        },
    });
    
    const value = std.json.Value{ .string = "hello" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    defer packet.deinit();
    
    try packet.validate(&schema);
}

test "DataSchema string validation - too short" {
    const allocator = std.testing.allocator;
    
    const schema = DataSchema.init(.string, true, .{
        .string_constraints = .{
            .min_length = 5,
            .max_length = 10,
        },
    });
    
    const value = std.json.Value{ .string = "hi" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    defer packet.deinit();
    
    try std.testing.expectError(error.StringTooShort, packet.validate(&schema));
}

test "DataSchema number validation - success" {
    const allocator = std.testing.allocator;
    
    const schema = DataSchema.init(.number, true, .{
        .number_constraints = .{
            .min = 0.0,
            .max = 100.0,
        },
    });
    
    const value = std.json.Value{ .float = 50.0 };
    const packet = try DataPacket.init(allocator, "p1", .number, value);
    defer packet.deinit();
    
    try packet.validate(&schema);
}

test "DataSchema number validation - too large" {
    const allocator = std.testing.allocator;
    
    const schema = DataSchema.init(.number, true, .{
        .number_constraints = .{
            .min = 0.0,
            .max = 100.0,
        },
    });
    
    const value = std.json.Value{ .float = 150.0 };
    const packet = try DataPacket.init(allocator, "p1", .number, value);
    defer packet.deinit();
    
    try std.testing.expectError(error.NumberTooLarge, packet.validate(&schema));
}

test "DataSchema type mismatch" {
    const allocator = std.testing.allocator;
    
    const schema = DataSchema.init(.string, true, null);
    
    const value = std.json.Value{ .float = 123.0 };
    const packet = try DataPacket.init(allocator, "p1", .number, value);
    defer packet.deinit();
    
    try std.testing.expectError(error.TypeMismatch, packet.validate(&schema));
}
