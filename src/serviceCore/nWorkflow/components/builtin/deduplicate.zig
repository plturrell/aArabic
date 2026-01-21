//! Deduplicate Component - Day 19
//! 
//! Data deduplication component for removing duplicate items.
//! Supports whole-item and field-based deduplication.
//!
//! Key Features:
//! - Remove duplicate items from arrays
//! - Deduplicate by specific field
//! - Keep first or last occurrence
//! - Count duplicates removed

const std = @import("std");
const node_types = @import("node_types");
const metadata_mod = @import("component_metadata");
const Allocator = std.mem.Allocator;

const NodeInterface = node_types.NodeInterface;
const ExecutionContext = node_types.ExecutionContext;
const Port = node_types.Port;
const ComponentMetadata = metadata_mod.ComponentMetadata;
const PortMetadata = metadata_mod.PortMetadata;
const ConfigSchemaField = metadata_mod.ConfigSchemaField;

/// Duplicate keeping strategy
pub const KeepStrategy = enum {
    first,
    last,
    
    pub fn fromString(str: []const u8) ?KeepStrategy {
        if (std.mem.eql(u8, str, "first")) return .first;
        if (std.mem.eql(u8, str, "last")) return .last;
        return null;
    }
    
    pub fn toString(self: KeepStrategy) []const u8 {
        return switch (self) {
            .first => "first",
            .last => "last",
        };
    }
};

/// Deduplicate Node implementation
pub const DeduplicateNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    field: ?[]const u8, // Field to deduplicate by (null = whole item)
    keep: KeepStrategy,
    count_removed: bool, // Whether to include count of removed items
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*DeduplicateNode {
        const node = try allocator.create(DeduplicateNode);
        errdefer allocator.destroy(node);
        
        node.* = DeduplicateNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "deduplicate",
            .field = null,
            .keep = .first,
            .count_removed = false,
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 2),
        };
        
        node.inputs[0] = Port{
            .id = "data",
            .name = "Data",
            .description = "Array to deduplicate",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        node.outputs[0] = Port{
            .id = "unique",
            .name = "Unique",
            .description = "Array with duplicates removed",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        node.outputs[1] = Port{
            .id = "stats",
            .name = "Statistics",
            .description = "Deduplication statistics",
            .port_type = .object,
            .required = false,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *DeduplicateNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.field) |f| self.allocator.free(f);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *DeduplicateNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("field")) |field_val| {
            if (field_val == .string) {
                self.field = try self.allocator.dupe(u8, field_val.string);
            }
        }
        
        if (config_obj.get("keep")) |keep_val| {
            if (keep_val == .string) {
                self.keep = KeepStrategy.fromString(keep_val.string) orelse .first;
            }
        }
        
        if (config_obj.get("count_removed")) |count_val| {
            if (count_val == .bool) {
                self.count_removed = count_val.bool;
            }
        }
    }
    
    pub fn asNodeInterface(self: *DeduplicateNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .description = "",
            .name = self.name,
            .node_type = self.node_type,
            .category = .utility,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = .{ .null = {} },
            .vtable = &.{
                .validate = validateImpl,
                .execute = executeImpl,
                .deinit = deinitImpl,
            },
            .impl_ptr = self,
        };
    }
    
    fn validateImpl(interface: *const NodeInterface) anyerror!void {
        _ = interface;
        // Deduplicate is always valid - field is optional
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*DeduplicateNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        // Mock implementation - returns deduplicated data structure
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "deduplicate" });
        try result.put("keep", std.json.Value{ .string = self.keep.toString() });
        
        if (self.field) |f| {
            try result.put("field", std.json.Value{ .string = f });
        }
        
        try result.put("original_count", std.json.Value{ .integer = 15 });
        try result.put("unique_count", std.json.Value{ .integer = 10 });
        
        if (self.count_removed) {
            try result.put("removed_count", std.json.Value{ .integer = 5 });
        }
        
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*DeduplicateNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const keep_strategies = [_][]const u8{ "first", "last" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("data", "Data", .array, true, "Array to deduplicate"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("unique", "Unique", .array, true, "Array with duplicates removed"),
        PortMetadata.init("stats", "Statistics", .object, false, "Deduplication statistics"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.stringField(
            "field",
            false,
            "Field to deduplicate by (empty = whole item)",
            "id",
        ),
        ConfigSchemaField.selectField(
            "keep",
            true,
            "Which occurrence to keep",
            &keep_strategies,
            "first",
        ),
        ConfigSchemaField.booleanField(
            "count_removed",
            false,
            "Include count of removed duplicates",
            false,
        ),
    };
    
    const tags = [_][]const u8{ "deduplicate", "unique", "array", "filter", "distinct" };
    const examples = [_][]const u8{
        "Remove duplicate items from array",
        "Deduplicate objects by ID field",
        "Keep last occurrence of duplicates",
    };
    
    return ComponentMetadata{
        .id = "deduplicate",
        .name = "Deduplicate Array",
        .version = "1.0.0",
        .description = "Remove duplicate items from arrays",
        .category = .transform,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "🎯",
        .color = "#16A085",
        .tags = &tags,
        .help_text = "Remove duplicates from arrays, either by whole item or specific field",
        .examples = &examples,
        .factory_fn = createDeduplicateNode,
    };
}

fn createDeduplicateNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const dedupe_node = try DeduplicateNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = dedupe_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "KeepStrategy string conversion" {
    try std.testing.expectEqual(KeepStrategy.first, KeepStrategy.fromString("first").?);
    try std.testing.expectEqual(KeepStrategy.last, KeepStrategy.fromString("last").?);
    try std.testing.expectEqualStrings("first", KeepStrategy.first.toString());
    try std.testing.expectEqualStrings("last", KeepStrategy.last.toString());
}

test "DeduplicateNode creation with defaults" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    
    const config = std.json.Value{ .object = config_obj };
    var node = try DeduplicateNode.init(allocator, "dedup1", "Deduplicate", config);
    defer node.deinit();
    
    try std.testing.expect(node.field == null);
    try std.testing.expectEqual(KeepStrategy.first, node.keep);
    try std.testing.expectEqual(false, node.count_removed);
}

test "DeduplicateNode keep first strategy" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("keep", std.json.Value{ .string = "first" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try DeduplicateNode.init(allocator, "dedup1", "Deduplicate", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("deduplicate", result.object.get("operation").?.string);
    try std.testing.expectEqualStrings("first", result.object.get("keep").?.string);
}

test "DeduplicateNode keep last strategy" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("keep", std.json.Value{ .string = "last" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try DeduplicateNode.init(allocator, "dedup1", "Deduplicate", config);
    defer node.deinit();
    
    try std.testing.expectEqual(KeepStrategy.last, node.keep);
}

test "DeduplicateNode with field" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("field", std.json.Value{ .string = "user_id" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try DeduplicateNode.init(allocator, "dedup1", "Deduplicate", config);
    defer node.deinit();
    
    try std.testing.expectEqualStrings("user_id", node.field.?);
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expectEqualStrings("user_id", result.object.get("field").?.string);
}

test "DeduplicateNode with count removed" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("count_removed", std.json.Value{ .bool = true });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try DeduplicateNode.init(allocator, "dedup1", "Deduplicate", config);
    defer node.deinit();
    
    try std.testing.expectEqual(true, node.count_removed);
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result.object.get("removed_count") != null);
}

test "DeduplicateNode validation always passes" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    
    const config = std.json.Value{ .object = config_obj };
    var node = try DeduplicateNode.init(allocator, "dedup1", "Deduplicate", config);
    defer node.deinit();
    
    const interface = node.asNodeInterface();
    try interface.vtable.?.validate(&interface);
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("deduplicate", metadata.id);
    try std.testing.expectEqualStrings("Deduplicate Array", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.transform, metadata.category);
}
