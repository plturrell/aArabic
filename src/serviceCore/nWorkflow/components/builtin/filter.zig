//! Filter Component - Day 17
//! 
//! Filters data based on conditions.
//! Supports simple comparisons and basic expressions.
//!
//! Key Features:
//! - Simple mode: Basic value comparisons
//! - Expression mode: Field-based conditions
//! - Multiple output ports (passed/failed)
//! - Type-aware filtering

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

/// Filter modes
pub const FilterMode = enum {
    simple,
    expression,
    
    pub fn fromString(str: []const u8) ?FilterMode {
        if (std.mem.eql(u8, str, "simple")) return .simple;
        if (std.mem.eql(u8, str, "expression")) return .expression;
        return null;
    }
    
    pub fn toString(self: FilterMode) []const u8 {
        return switch (self) {
            .simple => "simple",
            .expression => "expression",
        };
    }
};

/// Filter Node implementation
pub const FilterNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    mode: FilterMode,
    condition: []const u8,
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*FilterNode {
        const node = try allocator.create(FilterNode);
        errdefer allocator.destroy(node);
        
        node.* = FilterNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "filter",
            .mode = .simple,
            .condition = "",
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 2),
        };
        
        node.inputs[0] = Port{
            .description = "",
            .id = "data",
            .name = "Data",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        node.outputs[0] = Port{
            .description = "",
            .id = "passed",
            .name = "Passed",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        node.outputs[1] = Port{
            .description = "",
            .id = "failed",
            .name = "Failed",
            .port_type = .array,
            .required = false,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *FilterNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.condition);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *FilterNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("mode")) |mode_val| {
            if (mode_val == .string) {
                self.mode = FilterMode.fromString(mode_val.string) orelse .simple;
            }
        }
        
        if (config_obj.get("condition")) |cond_val| {
            if (cond_val == .string) {
                self.condition = try self.allocator.dupe(u8, cond_val.string);
            }
        }
    }
    
    pub fn asNodeInterface(self: *FilterNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "",
            .node_type = self.node_type,
            .category = .transform,
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
        const self = @as(*const FilterNode, @ptrCast(@alignCast(interface.impl_ptr)));
        
        if (self.condition.len == 0) {
            return error.MissingCondition;
        }
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*FilterNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        // Mock implementation
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("mode", std.json.Value{ .string = self.mode.toString() });
        try result.put("condition", std.json.Value{ .string = self.condition });
        try result.put("status", std.json.Value{ .string = "filtered" });
        
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*FilterNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const modes = [_][]const u8{ "simple", "expression" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("data", "Data", .array, true, "Data to filter"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("passed", "Passed", .array, true, "Items that passed filter"),
        PortMetadata.init("failed", "Failed", .array, false, "Items that failed filter"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.selectField(
            "mode",
            true,
            "Filter mode",
            &modes,
            "simple",
        ),
        ConfigSchemaField.stringField(
            "condition",
            true,
            "Filter condition",
            "value > 10",
        ),
    };
    
    const tags = [_][]const u8{ "filter", "condition", "query", "data" };
    const examples = [_][]const u8{
        "Simple: value > 10",
        "Expression: item.age >= 18",
        "Expression: item.status == 'active'",
    };
    
    return ComponentMetadata{
        .id = "filter",
        .name = "Filter Data",
        .version = "1.0.0",
        .description = "Filter data based on conditions",
        .category = .transform,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "🔍",
        .color = "#E74C3C",
        .tags = &tags,
        .help_text = "Filter arrays based on conditions with passed/failed outputs",
        .examples = &examples,
        .factory_fn = createFilterNode,
    };
}

fn createFilterNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const filter_node = try FilterNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = filter_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "FilterMode string conversion" {
    try std.testing.expectEqual(FilterMode.simple, FilterMode.fromString("simple").?);
    try std.testing.expectEqual(FilterMode.expression, FilterMode.fromString("expression").?);
    try std.testing.expectEqualStrings("simple", FilterMode.simple.toString());
}

test "FilterNode creation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("mode", std.json.Value{ .string = "simple" });
    try config_obj.put("condition", std.json.Value{ .string = "value > 10" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try FilterNode.init(allocator, "filter1", "Filter", config);
    defer node.deinit();
    
    try std.testing.expectEqual(FilterMode.simple, node.mode);
    try std.testing.expectEqualStrings("value > 10", node.condition);
}

test "Filter validation - missing condition" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("mode", std.json.Value{ .string = "simple" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try FilterNode.init(allocator, "filter1", "Filter", config);
    defer node.deinit();
    
    const interface = node.asNodeInterface();
    try std.testing.expectError(error.MissingCondition, interface.vtable.?.validate(&interface));
}

test "Filter execute" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("mode", std.json.Value{ .string = "expression" });
    try config_obj.put("condition", std.json.Value{ .string = "age >= 18" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try FilterNode.init(allocator, "filter1", "Filter", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("expression", result.object.get("mode").?.string);
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("filter", metadata.id);
    try std.testing.expectEqualStrings("Filter Data", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.transform, metadata.category);
    try std.testing.expectEqual(@as(usize, 2), metadata.outputs.len);
}
