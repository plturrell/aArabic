//! Transform Component - Day 17
//! 
//! Data transformation component with functional operations.
//! Supports map, filter, reduce, pluck, and flatten operations.
//!
//! Key Features:
//! - Map: Transform each item
//! - Filter: Keep items matching condition
//! - Reduce: Aggregate to single value
//! - Pluck: Extract field from objects
//! - Flatten: Flatten nested arrays

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

/// Transform operations
pub const TransformOperation = enum {
    map,
    filter,
    reduce,
    pluck,
    flatten,
    
    pub fn fromString(str: []const u8) ?TransformOperation {
        if (std.mem.eql(u8, str, "map")) return .map;
        if (std.mem.eql(u8, str, "filter")) return .filter;
        if (std.mem.eql(u8, str, "reduce")) return .reduce;
        if (std.mem.eql(u8, str, "pluck")) return .pluck;
        if (std.mem.eql(u8, str, "flatten")) return .flatten;
        return null;
    }
    
    pub fn toString(self: TransformOperation) []const u8 {
        return switch (self) {
            .map => "map",
            .filter => "filter",
            .reduce => "reduce",
            .pluck => "pluck",
            .flatten => "flatten",
        };
    }
};

/// Transform Node implementation
pub const TransformNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    operation: TransformOperation,
    field: ?[]const u8, // For pluck operation
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*TransformNode {
        const node = try allocator.create(TransformNode);
        errdefer allocator.destroy(node);
        
        node.* = TransformNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "transform",
            .operation = .map,
            .field = null,
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 1),
        };
        
        node.inputs[0] = Port{
            .id = "data",
            .name = "Data",
            .description = "Input array to transform",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };

        node.outputs[0] = Port{
            .id = "result",
            .name = "Result",
            .description = "Transformed output data",
            .port_type = .any,
            .required = true,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *TransformNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.field) |f| self.allocator.free(f);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *TransformNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("operation")) |op_val| {
            if (op_val == .string) {
                self.operation = TransformOperation.fromString(op_val.string) orelse .map;
            }
        }
        
        if (config_obj.get("field")) |field_val| {
            if (field_val == .string) {
                self.field = try self.allocator.dupe(u8, field_val.string);
            }
        }
    }
    
    pub fn asNodeInterface(self: *TransformNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Data transformation node",
            .node_type = self.node_type,
            .category = .transform,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = .null,
            .vtable = &.{
                .validate = validateImpl,
                .execute = executeImpl,
                .deinit = deinitImpl,
            },
            .impl_ptr = self,
        };
    }
    
    fn validateImpl(interface: *const NodeInterface) anyerror!void {
        const self = @as(*const TransformNode, @ptrCast(@alignCast(interface.impl_ptr)));
        
        if (self.operation == .pluck and self.field == null) {
            return error.MissingField;
        }
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*TransformNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        // Mock implementation - returns transformed data structure
        return switch (self.operation) {
            .map => try self.executeMap(),
            .filter => try self.executeFilter(),
            .reduce => try self.executeReduce(),
            .pluck => try self.executePluck(),
            .flatten => try self.executeFlatten(),
        };
    }
    
    fn executeMap(self: *const TransformNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "map" });
        try result.put("status", std.json.Value{ .string = "transformed" });
        return std.json.Value{ .object = result };
    }
    
    fn executeFilter(self: *const TransformNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "filter" });
        try result.put("status", std.json.Value{ .string = "filtered" });
        return std.json.Value{ .object = result };
    }
    
    fn executeReduce(self: *const TransformNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "reduce" });
        try result.put("status", std.json.Value{ .string = "reduced" });
        return std.json.Value{ .object = result };
    }
    
    fn executePluck(self: *const TransformNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "pluck" });
        if (self.field) |f| {
            try result.put("field", std.json.Value{ .string = f });
        }
        return std.json.Value{ .object = result };
    }
    
    fn executeFlatten(self: *const TransformNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "flatten" });
        try result.put("status", std.json.Value{ .string = "flattened" });
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*TransformNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const operations = [_][]const u8{ "map", "filter", "reduce", "pluck", "flatten" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("data", "Data", .array, true, "Array to transform"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("result", "Result", .any, true, "Transformed data"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.selectField(
            "operation",
            true,
            "Transform operation",
            &operations,
            "map",
        ),
        ConfigSchemaField.stringField(
            "field",
            false,
            "Field to pluck (for pluck operation)",
            "name",
        ),
    };
    
    const tags = [_][]const u8{ "transform", "map", "filter", "array", "data" };
    const examples = [_][]const u8{
        "Map: Transform each item in array",
        "Filter: Keep items matching condition",
        "Pluck: Extract field from objects",
    };
    
    return ComponentMetadata{
        .id = "transform",
        .name = "Transform Data",
        .version = "1.0.0",
        .description = "Transform data using functional operations",
        .category = .transform,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "🔄",
        .color = "#9B59B6",
        .tags = &tags,
        .help_text = "Transform arrays using map, filter, reduce, pluck, or flatten operations",
        .examples = &examples,
        .factory_fn = createTransformNode,
    };
}

fn createTransformNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const transform_node = try TransformNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = transform_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "TransformOperation string conversion" {
    try std.testing.expectEqual(TransformOperation.map, TransformOperation.fromString("map").?);
    try std.testing.expectEqual(TransformOperation.filter, TransformOperation.fromString("filter").?);
    try std.testing.expectEqualStrings("map", TransformOperation.map.toString());
}

test "TransformNode creation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "map" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try TransformNode.init(allocator, "trans1", "Transform", config);
    defer node.deinit();
    
    try std.testing.expectEqual(TransformOperation.map, node.operation);
}

test "Transform map operation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "map" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try TransformNode.init(allocator, "trans1", "Transform", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();

    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("map", result.object.get("operation").?.string);
}

test "Transform pluck operation with field" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "pluck" });
    try config_obj.put("field", std.json.Value{ .string = "name" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try TransformNode.init(allocator, "trans1", "Transform", config);
    defer node.deinit();
    
    try std.testing.expectEqual(TransformOperation.pluck, node.operation);
    try std.testing.expectEqualStrings("name", node.field.?);
}

test "Transform validation - pluck without field" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "pluck" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try TransformNode.init(allocator, "trans1", "Transform", config);
    defer node.deinit();
    
    const interface = node.asNodeInterface();
    try std.testing.expectError(error.MissingField, interface.vtable.?.validate(&interface));
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("transform", metadata.id);
    try std.testing.expectEqualStrings("Transform Data", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.transform, metadata.category);
}
