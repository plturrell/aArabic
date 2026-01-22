//! Variable Component - Day 18
//! 
//! Manages workflow variables for state and data sharing.
//! Supports get, set, and delete operations with scoped storage.
//!
//! Key Features:
//! - Get/set/delete operations
//! - Workflow and execution scopes
//! - Type preservation
//! - Default values

const std = @import("std");
const node_types = @import("node_types");
const metadata_mod = @import("component_metadata");
const Allocator = std.mem.Allocator;

const NodeInterface = node_types.NodeInterface;
const NodeCategory = node_types.NodeCategory;
const ExecutionContext = node_types.ExecutionContext;
const Port = node_types.Port;
const ComponentMetadata = metadata_mod.ComponentMetadata;
const PortMetadata = metadata_mod.PortMetadata;
const ConfigSchemaField = metadata_mod.ConfigSchemaField;

/// Variable operations
pub const VariableOperation = enum {
    get,
    set,
    delete,
    
    pub fn fromString(str: []const u8) ?VariableOperation {
        if (std.mem.eql(u8, str, "get")) return .get;
        if (std.mem.eql(u8, str, "set")) return .set;
        if (std.mem.eql(u8, str, "delete")) return .delete;
        return null;
    }
    
    pub fn toString(self: VariableOperation) []const u8 {
        return switch (self) {
            .get => "get",
            .set => "set",
            .delete => "delete",
        };
    }
};

/// Variable scopes
pub const VariableScope = enum {
    workflow,
    execution,
    
    pub fn fromString(str: []const u8) ?VariableScope {
        if (std.mem.eql(u8, str, "workflow")) return .workflow;
        if (std.mem.eql(u8, str, "execution")) return .execution;
        return null;
    }
    
    pub fn toString(self: VariableScope) []const u8 {
        return switch (self) {
            .workflow => "workflow",
            .execution => "execution",
        };
    }
};

/// Variable Node implementation
pub const VariableNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    operation: VariableOperation,
    scope: VariableScope,
    variable_name: []const u8,
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*VariableNode {
        const node = try allocator.create(VariableNode);
        errdefer allocator.destroy(node);
        
        node.* = VariableNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "variable",
            .operation = .get,
            .scope = .execution,
            .variable_name = "",
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 1),
        };
        
        node.inputs[0] = Port{
            .id = "value",
            .name = "Value",
            .description = "Value to set (for set operation)",
            .port_type = .any,
            .required = false,
            .default_value = null,
        };

        node.outputs[0] = Port{
            .id = "value",
            .name = "Value",
            .description = "Variable value (for get operation)",
            .port_type = .any,
            .required = false,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *VariableNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.variable_name);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *VariableNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("operation")) |op_val| {
            if (op_val == .string) {
                self.operation = VariableOperation.fromString(op_val.string) orelse .get;
            }
        }
        
        if (config_obj.get("scope")) |scope_val| {
            if (scope_val == .string) {
                self.scope = VariableScope.fromString(scope_val.string) orelse .execution;
            }
        }
        
        if (config_obj.get("name")) |name_val| {
            if (name_val == .string) {
                self.variable_name = try self.allocator.dupe(u8, name_val.string);
            }
        }
    }
    
    pub fn asNodeInterface(self: *VariableNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Variable node for workflow state management",
            .node_type = self.node_type,
            .category = .action,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = std.json.Value{ .null = {} },
            .vtable = &.{
                .validate = validateImpl,
                .execute = executeImpl,
                .deinit = deinitImpl,
            },
            .impl_ptr = self,
        };
    }
    
    fn validateImpl(interface: *const NodeInterface) anyerror!void {
        const self = @as(*const VariableNode, @ptrCast(@alignCast(interface.impl_ptr)));
        
        if (self.variable_name.len == 0) {
            return error.MissingVariableName;
        }
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*VariableNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        // Mock implementation
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = self.operation.toString() });
        try result.put("scope", std.json.Value{ .string = self.scope.toString() });
        try result.put("variable", std.json.Value{ .string = self.variable_name });
        
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*VariableNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const operations = [_][]const u8{ "get", "set", "delete" };
    const scopes = [_][]const u8{ "workflow", "execution" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("value", "Value", .any, false, "Value to set (for set operation)"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("value", "Value", .any, false, "Variable value (for get operation)"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.selectField(
            "operation",
            true,
            "Variable operation",
            &operations,
            "get",
        ),
        ConfigSchemaField.selectField(
            "scope",
            true,
            "Variable scope",
            &scopes,
            "execution",
        ),
        ConfigSchemaField.stringField(
            "name",
            true,
            "Variable name",
            "user_data",
        ),
    };
    
    const tags = [_][]const u8{ "variable", "state", "storage", "utility" };
    const examples = [_][]const u8{
        "Get: Retrieve variable value",
        "Set: Store variable value",
        "Delete: Remove variable",
    };
    
    return ComponentMetadata{
        .id = "variable",
        .name = "Variable",
        .version = "1.0.0",
        .description = "Get, set, or delete workflow variables",
        .category = .utility,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "💾",
        .color = "#3498DB",
        .tags = &tags,
        .help_text = "Manage workflow variables for state and data sharing",
        .examples = &examples,
        .factory_fn = createVariableNode,
    };
}

fn createVariableNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const var_node = try VariableNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = var_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "VariableOperation string conversion" {
    try std.testing.expectEqual(VariableOperation.get, VariableOperation.fromString("get").?);
    try std.testing.expectEqual(VariableOperation.set, VariableOperation.fromString("set").?);
    try std.testing.expectEqualStrings("get", VariableOperation.get.toString());
}

test "VariableScope string conversion" {
    try std.testing.expectEqual(VariableScope.workflow, VariableScope.fromString("workflow").?);
    try std.testing.expectEqual(VariableScope.execution, VariableScope.fromString("execution").?);
    try std.testing.expectEqualStrings("execution", VariableScope.execution.toString());
}

test "VariableNode creation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "get" });
    try config_obj.put("scope", std.json.Value{ .string = "execution" });
    try config_obj.put("name", std.json.Value{ .string = "my_var" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try VariableNode.init(allocator, "var1", "Variable", config);
    defer node.deinit();
    
    try std.testing.expectEqual(VariableOperation.get, node.operation);
    try std.testing.expectEqual(VariableScope.execution, node.scope);
    try std.testing.expectEqualStrings("my_var", node.variable_name);
}

test "Variable set operation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "set" });
    try config_obj.put("scope", std.json.Value{ .string = "workflow" });
    try config_obj.put("name", std.json.Value{ .string = "counter" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try VariableNode.init(allocator, "var1", "Variable", config);
    defer node.deinit();
    
    try std.testing.expectEqual(VariableOperation.set, node.operation);
    try std.testing.expectEqual(VariableScope.workflow, node.scope);
}

test "Variable validation - missing name" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "get" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try VariableNode.init(allocator, "var1", "Variable", config);
    defer node.deinit();
    
    const interface = node.asNodeInterface();
    try std.testing.expectError(error.MissingVariableName, interface.vtable.?.validate(&interface));
}

test "Variable execute" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("operation", std.json.Value{ .string = "delete" });
    try config_obj.put("scope", std.json.Value{ .string = "execution" });
    try config_obj.put("name", std.json.Value{ .string = "temp_data" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try VariableNode.init(allocator, "var1", "Variable", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("delete", result.object.get("operation").?.string);
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("variable", metadata.id);
    try std.testing.expectEqualStrings("Variable", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.utility, metadata.category);
}
