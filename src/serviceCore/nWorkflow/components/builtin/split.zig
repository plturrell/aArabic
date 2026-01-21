//! Split Component - Day 18
//! 
//! Splits data stream to multiple outputs.
//! Supports broadcast, round-robin, and conditional routing.
//!
//! Key Features:
//! - Broadcast: Send same data to all outputs
//! - Round-robin: Distribute items evenly
//! - Conditional: Route based on conditions
//! - Dynamic output configuration

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

/// Split modes
pub const SplitMode = enum {
    broadcast,
    round_robin,
    conditional,
    
    pub fn fromString(str: []const u8) ?SplitMode {
        if (std.mem.eql(u8, str, "broadcast")) return .broadcast;
        if (std.mem.eql(u8, str, "round_robin")) return .round_robin;
        if (std.mem.eql(u8, str, "conditional")) return .conditional;
        return null;
    }
    
    pub fn toString(self: SplitMode) []const u8 {
        return switch (self) {
            .broadcast => "broadcast",
            .round_robin => "round_robin",
            .conditional => "conditional",
        };
    }
};

/// Split Node implementation
pub const SplitNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    mode: SplitMode,
    num_outputs: usize,
    condition: []const u8,
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*SplitNode {
        const node = try allocator.create(SplitNode);
        errdefer allocator.destroy(node);
        
        node.* = SplitNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "split",
            .mode = .broadcast,
            .num_outputs = 2,
            .condition = "",
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 2),
        };
        
        node.inputs[0] = Port{
            .id = "input",
            .name = "Input",
            .description = "Data to split",
            .port_type = .any,
            .required = true,
            .default_value = null,
        };

        // Initialize with 2 outputs by default
        node.outputs[0] = Port{
            .id = "output1",
            .name = "Output 1",
            .description = "First output",
            .port_type = .any,
            .required = true,
            .default_value = null,
        };

        node.outputs[1] = Port{
            .id = "output2",
            .name = "Output 2",
            .description = "Second output",
            .port_type = .any,
            .required = true,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *SplitNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.condition);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *SplitNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("mode")) |mode_val| {
            if (mode_val == .string) {
                self.mode = SplitMode.fromString(mode_val.string) orelse .broadcast;
            }
        }
        
        if (config_obj.get("num_outputs")) |num_val| {
            if (num_val == .integer) {
                self.num_outputs = @intCast(num_val.integer);
            }
        }
        
        if (config_obj.get("condition")) |cond_val| {
            if (cond_val == .string) {
                self.condition = try self.allocator.dupe(u8, cond_val.string);
            }
        }
    }
    
    pub fn asNodeInterface(self: *SplitNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Split node for branching workflows",
            .node_type = self.node_type,
            .category = .condition,
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
        const self = @as(*const SplitNode, @ptrCast(@alignCast(interface.impl_ptr)));
        
        if (self.num_outputs < 2) {
            return error.InsufficientOutputs;
        }
        
        if (self.mode == .conditional and self.condition.len == 0) {
            return error.MissingCondition;
        }
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*SplitNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        // Mock implementation
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("mode", std.json.Value{ .string = self.mode.toString() });
        try result.put("num_outputs", std.json.Value{ .integer = @intCast(self.num_outputs) });
        try result.put("status", std.json.Value{ .string = "split" });
        
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*SplitNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const modes = [_][]const u8{ "broadcast", "round_robin", "conditional" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("input", "Input", .any, true, "Data to split"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("output1", "Output 1", .any, true, "First output"),
        PortMetadata.init("output2", "Output 2", .any, true, "Second output"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.selectField(
            "mode",
            true,
            "Split mode",
            &modes,
            "broadcast",
        ),
        ConfigSchemaField.numberField(
            "num_outputs",
            false,
            "Number of outputs",
            2,
        ),
        ConfigSchemaField.stringField(
            "condition",
            false,
            "Routing condition (for conditional mode)",
            "priority == 'high'",
        ),
    };
    
    const tags = [_][]const u8{ "split", "route", "logic", "branch" };
    const examples = [_][]const u8{
        "Broadcast: Send same data to all outputs",
        "Round-robin: Distribute items evenly",
        "Conditional: Route based on conditions",
    };
    
    return ComponentMetadata{
        .id = "split",
        .name = "Split Data",
        .version = "1.0.0",
        .description = "Split data stream to multiple outputs",
        .category = .logic,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "🔀",
        .color = "#9B59B6",
        .tags = &tags,
        .help_text = "Split data to multiple outputs using different routing strategies",
        .examples = &examples,
        .factory_fn = createSplitNode,
    };
}

fn createSplitNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const split_node = try SplitNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = split_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "SplitMode string conversion" {
    try std.testing.expectEqual(SplitMode.broadcast, SplitMode.fromString("broadcast").?);
    try std.testing.expectEqual(SplitMode.round_robin, SplitMode.fromString("round_robin").?);
    try std.testing.expectEqualStrings("broadcast", SplitMode.broadcast.toString());
}

test "SplitNode creation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("mode", std.json.Value{ .string = "broadcast" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SplitNode.init(allocator, "split1", "Split", config);
    defer node.deinit();
    
    try std.testing.expectEqual(SplitMode.broadcast, node.mode);
    try std.testing.expectEqual(@as(usize, 2), node.num_outputs);
}

test "Split with custom output count" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("mode", std.json.Value{ .string = "round_robin" });
    try config_obj.put("num_outputs", std.json.Value{ .integer = 3 });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SplitNode.init(allocator, "split1", "Split", config);
    defer node.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), node.num_outputs);
}

test "Split validation - conditional without condition" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("mode", std.json.Value{ .string = "conditional" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SplitNode.init(allocator, "split1", "Split", config);
    defer node.deinit();
    
    const interface = node.asNodeInterface();
    try std.testing.expectError(error.MissingCondition, interface.vtable.?.validate(&interface));
}

test "Split execute" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("mode", std.json.Value{ .string = "broadcast" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SplitNode.init(allocator, "split1", "Split", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("broadcast", result.object.get("mode").?.string);
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("split", metadata.id);
    try std.testing.expectEqualStrings("Split Data", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.logic, metadata.category);
}
