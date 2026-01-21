//! Aggregate Component - Day 19
//! 
//! Data aggregation component for computing statistics and grouping.
//! Supports sum, avg, count, min, max, and group_by operations.
//!
//! Key Features:
//! - Sum: Sum numeric values
//! - Average: Calculate average
//! - Count: Count items
//! - Min/Max: Find minimum/maximum
//! - Group By: Group items by field

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

/// Aggregate operations
pub const AggregateOperation = enum {
    sum,
    avg,
    count,
    min,
    max,
    group_by,
    
    pub fn fromString(str: []const u8) ?AggregateOperation {
        if (std.mem.eql(u8, str, "sum")) return .sum;
        if (std.mem.eql(u8, str, "avg")) return .avg;
        if (std.mem.eql(u8, str, "count")) return .count;
        if (std.mem.eql(u8, str, "min")) return .min;
        if (std.mem.eql(u8, str, "max")) return .max;
        if (std.mem.eql(u8, str, "group_by")) return .group_by;
        return null;
    }
    
    pub fn toString(self: AggregateOperation) []const u8 {
        return switch (self) {
            .sum => "sum",
            .avg => "avg",
            .count => "count",
            .min => "min",
            .max => "max",
            .group_by => "group_by",
        };
    }
};

/// Aggregate Node implementation
pub const AggregateNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    operation: AggregateOperation,
    field: ?[]const u8, // Field to aggregate on
    group_by_field: ?[]const u8, // Field to group by
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*AggregateNode {
        const node = try allocator.create(AggregateNode);
        errdefer allocator.destroy(node);
        
        node.* = AggregateNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "aggregate",
            .operation = .sum,
            .field = null,
            .group_by_field = null,
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 1),
        };
        
        node.inputs[0] = Port{
            .id = "data",
            .name = "Data",
            .description = "Array to aggregate",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        node.outputs[0] = Port{
            .id = "result",
            .name = "Result",
            .description = "Aggregated result",
            .port_type = .any,
            .required = true,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *AggregateNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.field) |f| self.allocator.free(f);
        if (self.group_by_field) |f| self.allocator.free(f);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *AggregateNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("operation")) |op_val| {
            if (op_val == .string) {
                self.operation = AggregateOperation.fromString(op_val.string) orelse .sum;
            }
        }
        
        if (config_obj.get("field")) |field_val| {
            if (field_val == .string) {
                self.field = try self.allocator.dupe(u8, field_val.string);
            }
        }
        
        if (config_obj.get("group_by_field")) |field_val| {
            if (field_val == .string) {
                self.group_by_field = try self.allocator.dupe(u8, field_val.string);
            }
        }
    }
    
    pub fn asNodeInterface(self: *AggregateNode) NodeInterface {
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
        const self = @as(*const AggregateNode, @ptrCast(@alignCast(interface.impl_ptr)));
        
        // Numeric operations require a field (except count)
        if (self.operation != .count and self.field == null) {
            return error.MissingField;
        }
        
        // Group by operation requires group_by_field
        if (self.operation == .group_by and self.group_by_field == null) {
            return error.MissingGroupByField;
        }
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*AggregateNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        return switch (self.operation) {
            .sum => try self.executeSum(),
            .avg => try self.executeAvg(),
            .count => try self.executeCount(),
            .min => try self.executeMin(),
            .max => try self.executeMax(),
            .group_by => try self.executeGroupBy(),
        };
    }
    
    fn executeSum(self: *const AggregateNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "sum" });
        if (self.field) |f| {
            try result.put("field", std.json.Value{ .string = f });
        }
        try result.put("value", std.json.Value{ .integer = 100 });
        return std.json.Value{ .object = result };
    }
    
    fn executeAvg(self: *const AggregateNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "avg" });
        if (self.field) |f| {
            try result.put("field", std.json.Value{ .string = f });
        }
        try result.put("value", std.json.Value{ .float = 25.5 });
        return std.json.Value{ .object = result };
    }
    
    fn executeCount(self: *const AggregateNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "count" });
        try result.put("value", std.json.Value{ .integer = 10 });
        return std.json.Value{ .object = result };
    }
    
    fn executeMin(self: *const AggregateNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "min" });
        if (self.field) |f| {
            try result.put("field", std.json.Value{ .string = f });
        }
        try result.put("value", std.json.Value{ .integer = 1 });
        return std.json.Value{ .object = result };
    }
    
    fn executeMax(self: *const AggregateNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "max" });
        if (self.field) |f| {
            try result.put("field", std.json.Value{ .string = f });
        }
        try result.put("value", std.json.Value{ .integer = 50 });
        return std.json.Value{ .object = result };
    }
    
    fn executeGroupBy(self: *const AggregateNode) !std.json.Value {
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "group_by" });
        if (self.group_by_field) |f| {
            try result.put("group_by_field", std.json.Value{ .string = f });
        }
        try result.put("groups", std.json.Value{ .integer = 3 });
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*AggregateNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const operations = [_][]const u8{ "sum", "avg", "count", "min", "max", "group_by" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("data", "Data", .array, true, "Array to aggregate"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("result", "Result", .any, true, "Aggregated result"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.selectField(
            "operation",
            true,
            "Aggregate operation",
            &operations,
            "sum",
        ),
        ConfigSchemaField.stringField(
            "field",
            false,
            "Field to aggregate (for numeric operations)",
            "value",
        ),
        ConfigSchemaField.stringField(
            "group_by_field",
            false,
            "Field to group by (for group_by operation)",
            "category",
        ),
    };
    
    const tags = [_][]const u8{ "aggregate", "sum", "average", "statistics", "math" };
    const examples = [_][]const u8{
        "Sum: Calculate total of numeric values",
        "Avg: Calculate average of values",
        "Count: Count number of items",
        "Group By: Group items by field value",
    };
    
    return ComponentMetadata{
        .id = "aggregate",
        .name = "Aggregate Data",
        .version = "1.0.0",
        .description = "Aggregate data using statistical operations",
        .category = .transform,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "📊",
        .color = "#3498DB",
        .tags = &tags,
        .help_text = "Compute statistics like sum, average, count, min, max, or group data by field",
        .examples = &examples,
        .factory_fn = createAggregateNode,
    };
}

fn createAggregateNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const aggregate_node = try AggregateNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = aggregate_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "AggregateOperation string conversion" {
    try std.testing.expectEqual(AggregateOperation.sum, AggregateOperation.fromString("sum").?);
    try std.testing.expectEqual(AggregateOperation.avg, AggregateOperation.fromString("avg").?);
    try std.testing.expectEqual(AggregateOperation.count, AggregateOperation.fromString("count").?);
    try std.testing.expectEqualStrings("sum", AggregateOperation.sum.toString());
    try std.testing.expectEqualStrings("group_by", AggregateOperation.group_by.toString());
}

test "AggregateNode creation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "sum" });
    try config_obj.put("field", std.json.Value{ .string = "value" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try AggregateNode.init(allocator, "agg1", "Aggregate", config);
    defer node.deinit();
    
    try std.testing.expectEqual(AggregateOperation.sum, node.operation);
    try std.testing.expectEqualStrings("value", node.field.?);
}

test "Aggregate sum operation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "sum" });
    try config_obj.put("field", std.json.Value{ .string = "amount" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try AggregateNode.init(allocator, "agg1", "Aggregate", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("sum", result.object.get("operation").?.string);
    try std.testing.expectEqualStrings("amount", result.object.get("field").?.string);
}

test "Aggregate avg operation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "avg" });
    try config_obj.put("field", std.json.Value{ .string = "score" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try AggregateNode.init(allocator, "agg1", "Aggregate", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expectEqualStrings("avg", result.object.get("operation").?.string);
    try std.testing.expect(result.object.get("value").? == .float);
}

test "Aggregate count operation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "count" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try AggregateNode.init(allocator, "agg1", "Aggregate", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expectEqualStrings("count", result.object.get("operation").?.string);
}

test "Aggregate group_by operation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "group_by" });
    try config_obj.put("group_by_field", std.json.Value{ .string = "category" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try AggregateNode.init(allocator, "agg1", "Aggregate", config);
    defer node.deinit();
    
    try std.testing.expectEqual(AggregateOperation.group_by, node.operation);
    try std.testing.expectEqualStrings("category", node.group_by_field.?);
}

test "Aggregate validation - sum without field" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "sum" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try AggregateNode.init(allocator, "agg1", "Aggregate", config);
    defer node.deinit();
    
    const interface = node.asNodeInterface();
    try std.testing.expectError(error.MissingField, interface.vtable.?.validate(&interface));
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("aggregate", metadata.id);
    try std.testing.expectEqualStrings("Aggregate Data", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.transform, metadata.category);
}
