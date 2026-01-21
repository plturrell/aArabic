//! Node Factory System - Day 14
//! 
//! This module provides factory functions for creating workflow nodes
//! from configuration data. It bridges the workflow parser with the
//! node type system, enabling dynamic node instantiation.
//!
//! Key Features:
//! - Dynamic node creation from JSON configuration
//! - Node type registry
//! - Configuration validation
//! - Port schema validation
//! - Error handling with detailed messages

const std = @import("std");
const Allocator = std.mem.Allocator;
const node_types = @import("node_types");

const Port = node_types.Port;
const PortType = node_types.PortType;
const NodeInterface = node_types.NodeInterface;
const NodeCategory = node_types.NodeCategory;
const TriggerNode = node_types.TriggerNode;
const ActionNode = node_types.ActionNode;
const ConditionNode = node_types.ConditionNode;
const TransformNode = node_types.TransformNode;

/// Node type identifier
pub const NodeTypeId = enum {
    trigger,
    action,
    condition,
    transform,
    custom,
    
    pub fn fromString(str: []const u8) ?NodeTypeId {
        if (std.mem.eql(u8, str, "trigger")) return .trigger;
        if (std.mem.eql(u8, str, "action")) return .action;
        if (std.mem.eql(u8, str, "condition")) return .condition;
        if (std.mem.eql(u8, str, "transform")) return .transform;
        if (std.mem.eql(u8, str, "custom")) return .custom;
        return null;
    }
    
    pub fn toString(self: NodeTypeId) []const u8 {
        return switch (self) {
            .trigger => "trigger",
            .action => "action",
            .condition => "condition",
            .transform => "transform",
            .custom => "custom",
        };
    }
};

/// Node configuration from workflow definition
pub const NodeConfig = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    node_type: []const u8,
    config: std.json.Value,
    
    /// Parse from JSON object
    pub fn fromJson(allocator: Allocator, json: std.json.Value) !NodeConfig {
        if (json != .object) return error.InvalidNodeConfig;
        
        const obj = json.object;
        
        const id = if (obj.get("id")) |v| switch (v) {
            .string => |s| s,
            else => return error.InvalidNodeId,
        } else return error.MissingNodeId;
        
        const name = if (obj.get("name")) |v| switch (v) {
            .string => |s| s,
            else => return error.InvalidNodeName,
        } else return error.MissingNodeName;
        
        const description = if (obj.get("description")) |v| switch (v) {
            .string => |s| s,
            else => "",
        } else "";
        
        const node_type = if (obj.get("type")) |v| switch (v) {
            .string => |s| s,
            else => return error.InvalidNodeType,
        } else return error.MissingNodeType;
        
        const config = obj.get("config") orelse std.json.Value{ .object = std.json.ObjectMap.init(allocator) };
        
        return NodeConfig{
            .id = id,
            .name = name,
            .description = description,
            .node_type = node_type,
            .config = config,
        };
    }
};

/// Port configuration from JSON
pub const PortConfig = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    port_type: PortType,
    required: bool,
    default_value: ?[]const u8,
    
    pub fn fromJson(json: std.json.Value) !PortConfig {
        if (json != .object) return error.InvalidPortConfig;
        
        const obj = json.object;
        
        const id = if (obj.get("id")) |v| switch (v) {
            .string => |s| s,
            else => return error.InvalidPortId,
        } else return error.MissingPortId;
        
        const name = if (obj.get("name")) |v| switch (v) {
            .string => |s| s,
            else => return error.InvalidPortName,
        } else return error.MissingPortName;
        
        const description = if (obj.get("description")) |v| switch (v) {
            .string => |s| s,
            else => "",
        } else "";
        
        const type_str = if (obj.get("type")) |v| switch (v) {
            .string => |s| s,
            else => return error.InvalidPortType,
        } else return error.MissingPortType;
        
        const port_type = parsePortType(type_str) orelse return error.UnknownPortType;
        
        const required = if (obj.get("required")) |v| switch (v) {
            .bool => |b| b,
            else => false,
        } else false;
        
        const default_value = if (obj.get("default")) |v| switch (v) {
            .string => |s| s,
            else => null,
        } else null;
        
        return PortConfig{
            .id = id,
            .name = name,
            .description = description,
            .port_type = port_type,
            .required = required,
            .default_value = default_value,
        };
    }
    
    pub fn toPort(self: PortConfig) Port {
        return Port.init(
            self.id,
            self.name,
            self.description,
            self.port_type,
            self.required,
            self.default_value,
        );
    }
};

fn parsePortType(type_str: []const u8) ?PortType {
    if (std.mem.eql(u8, type_str, "string")) return .string;
    if (std.mem.eql(u8, type_str, "number")) return .number;
    if (std.mem.eql(u8, type_str, "boolean")) return .boolean;
    if (std.mem.eql(u8, type_str, "object")) return .object;
    if (std.mem.eql(u8, type_str, "array")) return .array;
    if (std.mem.eql(u8, type_str, "any")) return .any;
    return null;
}

/// Node factory for creating nodes from configuration
pub const NodeFactory = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) NodeFactory {
        return NodeFactory{
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *NodeFactory) void {
        _ = self;
        // Nothing to cleanup for now
    }
    
    /// Create a node from configuration
    pub fn createNode(self: *NodeFactory, config: NodeConfig) !*NodeInterface {
        const node_type_id = NodeTypeId.fromString(config.node_type) orelse return error.UnknownNodeType;
        
        return switch (node_type_id) {
            .trigger => try self.createTriggerNode(config),
            .action => try self.createActionNode(config),
            .condition => try self.createConditionNode(config),
            .transform => try self.createTransformNode(config),
            .custom => error.CustomNodesNotYetSupported,
        };
    }
    
    fn createTriggerNode(self: *NodeFactory, config: NodeConfig) !*NodeInterface {
        // Parse trigger-specific configuration
        const trigger_type = if (config.config == .object) blk: {
            const obj = config.config.object;
            if (obj.get("trigger_type")) |v| {
                break :blk switch (v) {
                    .string => |s| s,
                    else => "manual",
                };
            }
            break :blk "manual";
        } else "manual";
        
        // Parse output ports
        var outputs = std.ArrayList(Port){};
        defer outputs.deinit(self.allocator);
        
        if (config.config == .object) {
            const obj = config.config.object;
            if (obj.get("outputs")) |outputs_val| {
                if (outputs_val == .array) {
                    for (outputs_val.array.items) |port_val| {
                        const port_config = try PortConfig.fromJson(port_val);
                        try outputs.append(self.allocator, port_config.toPort());
                    }
                }
            }
        }
        
        // Default output if none specified
        if (outputs.items.len == 0) {
            try outputs.append(self.allocator, Port.init(
                "output",
                "Output",
                "Trigger output",
                .object,
                false,
                null,
            ));
        }
        
        const outputs_slice = try self.allocator.dupe(Port, outputs.items);
        
        var node = try self.allocator.create(TriggerNode);
        node.* = try TriggerNode.init(
            self.allocator,
            config.id,
            config.name,
            config.description,
            trigger_type,
            outputs_slice,
            config.config,
        );
        
        return @ptrCast(&node.base);
    }
    
    fn createActionNode(self: *NodeFactory, config: NodeConfig) !*NodeInterface {
        // Parse action-specific configuration
        const action_type = if (config.config == .object) blk: {
            const obj = config.config.object;
            if (obj.get("action_type")) |v| {
                break :blk switch (v) {
                    .string => |s| s,
                    else => "generic",
                };
            }
            break :blk "generic";
        } else "generic";
        
        // Parse input and output ports
        var inputs = std.ArrayList(Port){};
        defer inputs.deinit(self.allocator);
        var outputs = std.ArrayList(Port){};
        defer outputs.deinit(self.allocator);
        
        if (config.config == .object) {
            const obj = config.config.object;
            
            // Parse inputs
            if (obj.get("inputs")) |inputs_val| {
                if (inputs_val == .array) {
                    for (inputs_val.array.items) |port_val| {
                        const port_config = try PortConfig.fromJson(port_val);
                        try inputs.append(self.allocator, port_config.toPort());
                    }
                }
            }
            
            // Parse outputs
            if (obj.get("outputs")) |outputs_val| {
                if (outputs_val == .array) {
                    for (outputs_val.array.items) |port_val| {
                        const port_config = try PortConfig.fromJson(port_val);
                        try outputs.append(self.allocator, port_config.toPort());
                    }
                }
            }
        }
        
        // Default ports if none specified
        if (inputs.items.len == 0) {
            try inputs.append(self.allocator, Port.init(
                "input",
                "Input",
                "Action input",
                .any,
                true,
                null,
            ));
        }
        if (outputs.items.len == 0) {
            try outputs.append(self.allocator, Port.init(
                "output",
                "Output",
                "Action output",
                .any,
                false,
                null,
            ));
        }
        
        const inputs_slice = try self.allocator.dupe(Port, inputs.items);
        const outputs_slice = try self.allocator.dupe(Port, outputs.items);
        
        var node = try self.allocator.create(ActionNode);
        node.* = try ActionNode.init(
            self.allocator,
            config.id,
            config.name,
            config.description,
            action_type,
            inputs_slice,
            outputs_slice,
            config.config,
        );
        
        return @ptrCast(&node.base);
    }
    
    fn createConditionNode(self: *NodeFactory, config: NodeConfig) !*NodeInterface {
        // Parse condition expression
        const condition = if (config.config == .object) blk: {
            const obj = config.config.object;
            if (obj.get("condition")) |v| {
                break :blk switch (v) {
                    .string => |s| s,
                    else => "true",
                };
            }
            break :blk "true";
        } else "true";
        
        // Parse input and output ports
        var inputs = std.ArrayList(Port){};
        defer inputs.deinit(self.allocator);
        var outputs = std.ArrayList(Port){};
        defer outputs.deinit(self.allocator);
        
        if (config.config == .object) {
            const obj = config.config.object;
            
            if (obj.get("inputs")) |inputs_val| {
                if (inputs_val == .array) {
                    for (inputs_val.array.items) |port_val| {
                        const port_config = try PortConfig.fromJson(port_val);
                        try inputs.append(self.allocator, port_config.toPort());
                    }
                }
            }
            
            if (obj.get("outputs")) |outputs_val| {
                if (outputs_val == .array) {
                    for (outputs_val.array.items) |port_val| {
                        const port_config = try PortConfig.fromJson(port_val);
                        try outputs.append(self.allocator, port_config.toPort());
                    }
                }
            }
        }
        
        // Default ports for condition node
        if (inputs.items.len == 0) {
            try inputs.append(self.allocator, Port.init(
                "value",
                "Value",
                "Value to evaluate",
                .any,
                true,
                null,
            ));
        }
        if (outputs.items.len == 0) {
            try outputs.append(self.allocator, Port.init(
                "true",
                "True",
                "True branch",
                .any,
                false,
                null,
            ));
            try outputs.append(self.allocator, Port.init(
                "false",
                "False",
                "False branch",
                .any,
                false,
                null,
            ));
        }
        
        const inputs_slice = try self.allocator.dupe(Port, inputs.items);
        const outputs_slice = try self.allocator.dupe(Port, outputs.items);
        
        var node = try self.allocator.create(ConditionNode);
        node.* = try ConditionNode.init(
            self.allocator,
            config.id,
            config.name,
            config.description,
            condition,
            inputs_slice,
            outputs_slice,
            config.config,
        );
        
        return @ptrCast(&node.base);
    }
    
    fn createTransformNode(self: *NodeFactory, config: NodeConfig) !*NodeInterface {
        // Parse transform type
        const transform_type = if (config.config == .object) blk: {
            const obj = config.config.object;
            if (obj.get("transform_type")) |v| {
                break :blk switch (v) {
                    .string => |s| s,
                    else => "map",
                };
            }
            break :blk "map";
        } else "map";
        
        // Parse input and output ports
        var inputs = std.ArrayList(Port){};
        defer inputs.deinit(self.allocator);
        var outputs = std.ArrayList(Port){};
        defer outputs.deinit(self.allocator);
        
        if (config.config == .object) {
            const obj = config.config.object;
            
            if (obj.get("inputs")) |inputs_val| {
                if (inputs_val == .array) {
                    for (inputs_val.array.items) |port_val| {
                        const port_config = try PortConfig.fromJson(port_val);
                        try inputs.append(self.allocator, port_config.toPort());
                    }
                }
            }
            
            if (obj.get("outputs")) |outputs_val| {
                if (outputs_val == .array) {
                    for (outputs_val.array.items) |port_val| {
                        const port_config = try PortConfig.fromJson(port_val);
                        try outputs.append(self.allocator, port_config.toPort());
                    }
                }
            }
        }
        
        // Default ports
        if (inputs.items.len == 0) {
            try inputs.append(self.allocator, Port.init(
                "data",
                "Data",
                "Input data",
                .array,
                true,
                null,
            ));
        }
        if (outputs.items.len == 0) {
            try outputs.append(self.allocator, Port.init(
                "result",
                "Result",
                "Transformed data",
                .array,
                false,
                null,
            ));
        }
        
        const inputs_slice = try self.allocator.dupe(Port, inputs.items);
        const outputs_slice = try self.allocator.dupe(Port, outputs.items);
        
        var node = try self.allocator.create(TransformNode);
        node.* = try TransformNode.init(
            self.allocator,
            config.id,
            config.name,
            config.description,
            transform_type,
            inputs_slice,
            outputs_slice,
            config.config,
        );
        
        return @ptrCast(&node.base);
    }
    
    /// Create multiple nodes from an array of configurations
    pub fn createNodes(self: *NodeFactory, configs: []const NodeConfig) !std.ArrayList(*NodeInterface) {
        var nodes = std.ArrayList(*NodeInterface){};
        errdefer {
            for (nodes.items) |node| {
                self.destroyNode(node);
            }
            nodes.deinit(self.allocator);
        }
        
        for (configs) |config| {
            const node = try self.createNode(config);
            try nodes.append(self.allocator, node);
        }
        
        return nodes;
    }
    
    /// Destroy a node and free its memory
    pub fn destroyNode(self: *NodeFactory, node: *NodeInterface) void {
        // Free port arrays
        self.allocator.free(node.inputs);
        self.allocator.free(node.outputs);
        
        // Determine node type and free accordingly
        const node_type_id = NodeTypeId.fromString(node.node_type) orelse {
            // Unknown node type, can't safely free
            return;
        };
        
        switch (node_type_id) {
            .trigger => {
                const trigger_node: *TriggerNode = @alignCast(@ptrCast(node));
                self.allocator.destroy(trigger_node);
            },
            .action => {
                const action_node: *ActionNode = @alignCast(@ptrCast(node));
                self.allocator.destroy(action_node);
            },
            .condition => {
                const condition_node: *ConditionNode = @alignCast(@ptrCast(node));
                self.allocator.destroy(condition_node);
            },
            .transform => {
                const transform_node: *TransformNode = @alignCast(@ptrCast(node));
                self.allocator.destroy(transform_node);
            },
            .custom => {
                // Custom nodes not yet supported
            },
        }
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "NodeTypeId from/to string" {
    const trigger = NodeTypeId.fromString("trigger");
    try std.testing.expect(trigger != null);
    try std.testing.expectEqual(NodeTypeId.trigger, trigger.?);
    try std.testing.expectEqualStrings("trigger", trigger.?.toString());
    
    const invalid = NodeTypeId.fromString("invalid");
    try std.testing.expect(invalid == null);
}

test "Parse port type" {
    try std.testing.expectEqual(PortType.string, parsePortType("string").?);
    try std.testing.expectEqual(PortType.number, parsePortType("number").?);
    try std.testing.expectEqual(PortType.boolean, parsePortType("boolean").?);
    try std.testing.expectEqual(PortType.object, parsePortType("object").?);
    try std.testing.expectEqual(PortType.array, parsePortType("array").?);
    try std.testing.expectEqual(PortType.any, parsePortType("any").?);
    try std.testing.expect(parsePortType("invalid") == null);
}

test "NodeConfig from JSON" {
    const allocator = std.testing.allocator;
    
    var obj = std.json.ObjectMap.init(allocator);
    defer obj.deinit();
    
    try obj.put("id", std.json.Value{ .string = "node1" });
    try obj.put("name", std.json.Value{ .string = "Test Node" });
    try obj.put("type", std.json.Value{ .string = "trigger" });
    
    const json = std.json.Value{ .object = obj };
    const config = try NodeConfig.fromJson(allocator, json);
    
    try std.testing.expectEqualStrings("node1", config.id);
    try std.testing.expectEqualStrings("Test Node", config.name);
    try std.testing.expectEqualStrings("trigger", config.node_type);
}

test "PortConfig from JSON" {
    const allocator = std.testing.allocator;
    
    var obj = std.json.ObjectMap.init(allocator);
    defer obj.deinit();
    
    try obj.put("id", std.json.Value{ .string = "port1" });
    try obj.put("name", std.json.Value{ .string = "Test Port" });
    try obj.put("type", std.json.Value{ .string = "string" });
    try obj.put("required", std.json.Value{ .bool = true });
    
    const json = std.json.Value{ .object = obj };
    const config = try PortConfig.fromJson(json);
    
    try std.testing.expectEqualStrings("port1", config.id);
    try std.testing.expectEqualStrings("Test Port", config.name);
    try std.testing.expectEqual(PortType.string, config.port_type);
    try std.testing.expect(config.required);
}

test "NodeFactory create trigger node" {
    const allocator = std.testing.allocator;
    
    var factory = NodeFactory.init(allocator);
    defer factory.deinit();
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("trigger_type", std.json.Value{ .string = "manual" });
    
    const config = NodeConfig{
        .id = "trigger1",
        .name = "Manual Trigger",
        .description = "Test trigger",
        .node_type = "trigger",
        .config = std.json.Value{ .object = config_obj },
    };
    
    const node = try factory.createNode(config);
    defer factory.destroyNode(node);
    
    try std.testing.expectEqualStrings("trigger1", node.id);
    try std.testing.expectEqualStrings("Manual Trigger", node.name);
    try std.testing.expectEqual(NodeCategory.trigger, node.category);
}

test "NodeFactory create action node" {
    const allocator = std.testing.allocator;
    
    var factory = NodeFactory.init(allocator);
    defer factory.deinit();
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("action_type", std.json.Value{ .string = "http_request" });
    
    const config = NodeConfig{
        .id = "action1",
        .name = "HTTP Request",
        .description = "Test action",
        .node_type = "action",
        .config = std.json.Value{ .object = config_obj },
    };
    
    const node = try factory.createNode(config);
    defer factory.destroyNode(node);
    
    try std.testing.expectEqualStrings("action1", node.id);
    try std.testing.expectEqual(NodeCategory.action, node.category);
    try std.testing.expect(node.inputs.len > 0);
    try std.testing.expect(node.outputs.len > 0);
}

test "NodeFactory create condition node" {
    const allocator = std.testing.allocator;
    
    var factory = NodeFactory.init(allocator);
    defer factory.deinit();
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("condition", std.json.Value{ .string = "value > 10" });
    
    const config = NodeConfig{
        .id = "condition1",
        .name = "If Greater",
        .description = "Test condition",
        .node_type = "condition",
        .config = std.json.Value{ .object = config_obj },
    };
    
    const node = try factory.createNode(config);
    defer factory.destroyNode(node);
    
    try std.testing.expectEqualStrings("condition1", node.id);
    try std.testing.expectEqual(NodeCategory.condition, node.category);
    try std.testing.expect(node.outputs.len >= 2); // true and false branches
}

test "NodeFactory create transform node" {
    const allocator = std.testing.allocator;
    
    var factory = NodeFactory.init(allocator);
    defer factory.deinit();
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("transform_type", std.json.Value{ .string = "map" });
    
    const config = NodeConfig{
        .id = "transform1",
        .name = "Map Transform",
        .description = "Test transform",
        .node_type = "transform",
        .config = std.json.Value{ .object = config_obj },
    };
    
    const node = try factory.createNode(config);
    defer factory.destroyNode(node);
    
    try std.testing.expectEqualStrings("transform1", node.id);
    try std.testing.expectEqual(NodeCategory.transform, node.category);
}

test "NodeFactory create multiple nodes" {
    const allocator = std.testing.allocator;
    
    var factory = NodeFactory.init(allocator);
    defer factory.deinit();
    
    const configs = [_]NodeConfig{
        NodeConfig{
            .id = "node1",
            .name = "Node 1",
            .description = "",
            .node_type = "trigger",
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        },
        NodeConfig{
            .id = "node2",
            .name = "Node 2",
            .description = "",
            .node_type = "action",
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        },
    };
    
    var nodes = try factory.createNodes(&configs);
    defer {
        for (nodes.items) |node| {
            factory.destroyNode(node);
        }
        nodes.deinit(allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 2), nodes.items.len);
    try std.testing.expectEqualStrings("node1", nodes.items[0].id);
    try std.testing.expectEqualStrings("node2", nodes.items[1].id);
}

test "NodeFactory error handling" {
    const allocator = std.testing.allocator;
    
    var factory = NodeFactory.init(allocator);
    defer factory.deinit();
    
    // Invalid node type
    const invalid_config = NodeConfig{
        .id = "invalid",
        .name = "Invalid",
        .description = "",
        .node_type = "unknown",
        .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    };
    
    try std.testing.expectError(error.UnknownNodeType, factory.createNode(invalid_config));
}
