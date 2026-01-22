//! Node Type System - Day 13
//! 
//! This module provides the foundation for all workflow nodes in nWorkflow.
//! It defines:
//! - PortType: Type system for node inputs/outputs
//! - Port: Input/output port definition
//! - NodeInterface: Base interface for all nodes
//! - ExecutionContext: Runtime context for node execution
//! - Core node types: TriggerNode, ActionNode, ConditionNode, TransformNode
//!
//! Key Features:
//! - Type-safe port system
//! - Validation before execution
//! - Rich execution context
//! - Extensible node architecture

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Port data types supported by nWorkflow
pub const PortType = enum {
    string,
    number,
    boolean,
    object,
    array,
    any,
    
    /// Check if a value matches this port type
    pub fn matches(self: PortType, value: std.json.Value) bool {
        return switch (self) {
            .string => value == .string,
            .number => value == .number_string or value == .integer or value == .float,
            .boolean => value == .bool,
            .object => value == .object,
            .array => value == .array,
            .any => true,
        };
    }
    
    /// Get a human-readable name for this type
    pub fn name(self: PortType) []const u8 {
        return switch (self) {
            .string => "string",
            .number => "number",
            .boolean => "boolean",
            .object => "object",
            .array => "array",
            .any => "any",
        };
    }
};

/// Port definition for node inputs and outputs
pub const Port = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    port_type: PortType,
    required: bool,
    default_value: ?[]const u8,
    
    /// Create a new port
    pub fn init(
        id: []const u8,
        name: []const u8,
        description: []const u8,
        port_type: PortType,
        required: bool,
        default_value: ?[]const u8,
    ) Port {
        return Port{
            .id = id,
            .name = name,
            .description = description,
            .port_type = port_type,
            .required = required,
            .default_value = default_value,
        };
    }
    
    /// Validate that a value is compatible with this port
    pub fn validateValue(self: *const Port, value: ?std.json.Value) !void {
        if (value) |v| {
            if (!self.port_type.matches(v)) {
                return error.TypeMismatch;
            }
        } else {
            if (self.required and self.default_value == null) {
                return error.RequiredPortMissing;
            }
        }
    }
};

/// Service interface for external systems
pub const Service = struct {
    name: []const u8,
    connection_string: []const u8,
    metadata: std.StringHashMap([]const u8),
    context_data: ?*anyopaque = null,

    pub fn init(allocator: Allocator, name: []const u8, connection_string: []const u8) !Service {
        return Service{
            .name = name,
            .connection_string = connection_string,
            .metadata = std.StringHashMap([]const u8).init(allocator),
            .context_data = null,
        };
    }

    pub fn deinit(self: *Service) void {
        self.metadata.deinit();
    }

    pub fn setContextData(self: *Service, data: ?*anyopaque) void {
        self.context_data = data;
    }

    pub fn getContextData(self: *const Service) ?*anyopaque {
        return self.context_data;
    }
};

/// Execution context provided to nodes during execution
pub const ExecutionContext = struct {
    allocator: Allocator,
    workflow_id: []const u8,
    execution_id: []const u8,
    user_id: ?[]const u8,
    variables: std.StringHashMap([]const u8),
    services: std.StringHashMap(*Service),
    inputs: std.StringHashMap(std.json.Value),
    
    /// Create a new execution context
    pub fn init(
        allocator: Allocator,
        workflow_id: []const u8,
        execution_id: []const u8,
        user_id: ?[]const u8,
    ) ExecutionContext {
        return ExecutionContext{
            .allocator = allocator,
            .workflow_id = workflow_id,
            .execution_id = execution_id,
            .user_id = user_id,
            .variables = std.StringHashMap([]const u8).init(allocator),
            .services = std.StringHashMap(*Service).init(allocator),
            .inputs = std.StringHashMap(std.json.Value).init(allocator),
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *ExecutionContext) void {
        self.variables.deinit();
        self.services.deinit();
        self.inputs.deinit();
    }
    
    /// Get a variable value
    pub fn getVariable(self: *const ExecutionContext, key: []const u8) ?[]const u8 {
        return self.variables.get(key);
    }
    
    /// Set a variable value
    pub fn setVariable(self: *ExecutionContext, key: []const u8, value: []const u8) !void {
        try self.variables.put(key, value);
    }
    
    /// Get a service connection
    pub fn getService(self: *const ExecutionContext, name: []const u8) ?*Service {
        return self.services.get(name);
    }
    
    /// Register a service
    pub fn registerService(self: *ExecutionContext, name: []const u8, service: *Service) !void {
        try self.services.put(name, service);
    }
    
    /// Get an input value
    pub fn getInput(self: *const ExecutionContext, port_id: []const u8) ?std.json.Value {
        return self.inputs.get(port_id);
    }
    
    /// Set an input value
    pub fn setInput(self: *ExecutionContext, port_id: []const u8, value: std.json.Value) !void {
        try self.inputs.put(port_id, value);
    }
    
    /// Get input as string (convenience method)
    pub fn getInputString(self: *const ExecutionContext, port_id: []const u8) ?[]const u8 {
        if (self.getInput(port_id)) |value| {
            return switch (value) {
                .string => |s| s,
                else => null,
            };
        }
        return null;
    }
    
    /// Get input as number (convenience method)
    pub fn getInputNumber(self: *const ExecutionContext, port_id: []const u8) ?f64 {
        if (self.getInput(port_id)) |value| {
            return switch (value) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                .number_string => |s| std.fmt.parseFloat(f64, s) catch null,
                else => null,
            };
        }
        return null;
    }
    
    /// Get input as boolean (convenience method)
    pub fn getInputBool(self: *const ExecutionContext, port_id: []const u8) ?bool {
        if (self.getInput(port_id)) |value| {
            return switch (value) {
                .bool => |b| b,
                else => null,
            };
        }
        return null;
    }
};

/// Node category for UI organization
pub const NodeCategory = enum {
    trigger,
    action,
    condition,
    transform,
    data,
    integration,
    utility,
    
    pub fn name(self: NodeCategory) []const u8 {
        return switch (self) {
            .trigger => "Triggers",
            .action => "Actions",
            .condition => "Conditions",
            .transform => "Transforms",
            .data => "Data",
            .integration => "Integrations",
            .utility => "Utilities",
        };
    }
};

/// Base interface for all workflow nodes
pub const NodeInterface = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    node_type: []const u8,
    category: NodeCategory,
    inputs: []const Port,
    outputs: []const Port,
    config: std.json.Value,
    vtable: ?*const VTable = null,
    impl_ptr: ?*anyopaque = null,

    pub const VTable = struct {
        validate: *const fn (*const NodeInterface) anyerror!void,
        execute: *const fn (*NodeInterface, *ExecutionContext) anyerror!std.json.Value,
        deinit: *const fn (*NodeInterface) void,
    };

    /// Validate the node configuration
    pub fn validate(self: *const NodeInterface) !void {
        // Check that all required inputs are defined
        for (self.inputs) |input| {
            if (input.required and input.default_value == null) {
                // Will be checked at execution time
            }
        }
        
        // Node type-specific validation will be done by subclasses
    }
    
    /// Validate inputs before execution
    pub fn validateInputs(self: *const NodeInterface, ctx: *const ExecutionContext) !void {
        for (self.inputs) |input| {
            const value = ctx.getInput(input.id);
            try input.validateValue(value);
        }
    }
};

/// Trigger node - starts workflow execution
pub const TriggerNode = struct {
    base: NodeInterface,
    trigger_type: []const u8, // "webhook", "cron", "manual", "event"
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        description: []const u8,
        trigger_type: []const u8,
        outputs: []const Port,
        config: std.json.Value,
    ) !TriggerNode {
        const base = NodeInterface{
            .id = id,
            .name = name,
            .description = description,
            .node_type = "trigger",
            .category = .trigger,
            .inputs = &[_]Port{}, // Triggers have no inputs
            .outputs = outputs,
            .config = config,
        };
        
        return TriggerNode{
            .base = base,
            .trigger_type = trigger_type,
            .allocator = allocator,
        };
    }
    
    pub fn validate(self: *const TriggerNode) !void {
        try self.base.validate();
        
        // Ensure triggers have at least one output
        if (self.base.outputs.len == 0) {
            return error.TriggerMustHaveOutput;
        }
        
        // Validate trigger type
        const valid_types = [_][]const u8{ "webhook", "cron", "manual", "event" };
        var valid = false;
        for (valid_types) |t| {
            if (std.mem.eql(u8, self.trigger_type, t)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return error.InvalidTriggerType;
        }
    }
    
    pub fn execute(self: *TriggerNode, ctx: *ExecutionContext) !std.json.Value {
        try self.validate();
        
        // Trigger execution logic
        // In real implementation, this would:
        // - For webhook: parse HTTP request
        // - For cron: provide schedule metadata
        // - For manual: provide user metadata
        // - For event: provide event data
        
        var result = std.json.ObjectMap.init(self.allocator);
        errdefer result.deinit();
        
        try result.put("triggered", std.json.Value{ .bool = true });
        try result.put("trigger_type", std.json.Value{ .string = self.trigger_type });
        try result.put("workflow_id", std.json.Value{ .string = ctx.workflow_id });
        try result.put("execution_id", std.json.Value{ .string = ctx.execution_id });
        
        if (ctx.user_id) |uid| {
            try result.put("user_id", std.json.Value{ .string = uid });
        }
        
        return std.json.Value{ .object = result };
    }
};

/// Action node - performs operations
pub const ActionNode = struct {
    base: NodeInterface,
    action_type: []const u8, // "http_request", "db_query", "send_email", etc.
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        description: []const u8,
        action_type: []const u8,
        inputs: []const Port,
        outputs: []const Port,
        config: std.json.Value,
    ) !ActionNode {
        const base = NodeInterface{
            .id = id,
            .name = name,
            .description = description,
            .node_type = "action",
            .category = .action,
            .inputs = inputs,
            .outputs = outputs,
            .config = config,
        };
        
        return ActionNode{
            .base = base,
            .action_type = action_type,
            .allocator = allocator,
        };
    }
    
    pub fn validate(self: *const ActionNode) !void {
        try self.base.validate();
        
        // Action nodes must have at least one input and one output
        if (self.base.inputs.len == 0) {
            return error.ActionMustHaveInput;
        }
        if (self.base.outputs.len == 0) {
            return error.ActionMustHaveOutput;
        }
    }
    
    pub fn execute(self: *ActionNode, ctx: *ExecutionContext) !std.json.Value {
        try self.validate();
        try self.base.validateInputs(ctx);
        
        // Action execution logic
        // In real implementation, this would perform the action based on action_type
        // Examples:
        // - http_request: make HTTP call
        // - db_query: execute database query
        // - send_email: send email via SMTP
        
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("action_executed", std.json.Value{ .bool = true });
        try result.put("action_type", std.json.Value{ .string = self.action_type });
        try result.put("node_id", std.json.Value{ .string = self.base.id });
        
        return std.json.Value{ .object = result };
    }
};

/// Condition node - branching logic
pub const ConditionNode = struct {
    base: NodeInterface,
    condition: []const u8, // Boolean expression
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        description: []const u8,
        condition: []const u8,
        inputs: []const Port,
        outputs: []const Port,
        config: std.json.Value,
    ) !ConditionNode {
        const base = NodeInterface{
            .id = id,
            .name = name,
            .description = description,
            .node_type = "condition",
            .category = .condition,
            .inputs = inputs,
            .outputs = outputs,
            .config = config,
        };
        
        return ConditionNode{
            .base = base,
            .condition = condition,
            .allocator = allocator,
        };
    }
    
    pub fn validate(self: *const ConditionNode) !void {
        try self.base.validate();
        
        // Condition nodes must have at least one input
        if (self.base.inputs.len == 0) {
            return error.ConditionMustHaveInput;
        }
        
        // Must have at least 2 outputs (true/false branches)
        if (self.base.outputs.len < 2) {
            return error.ConditionMustHaveTwoBranches;
        }
        
        // Condition expression must not be empty
        if (self.condition.len == 0) {
            return error.EmptyCondition;
        }
    }
    
    pub fn execute(self: *ConditionNode, ctx: *ExecutionContext) !std.json.Value {
        try self.validate();
        try self.base.validateInputs(ctx);
        
        // Evaluate condition
        const result = try self.evaluateCondition(ctx);
        
        var output = std.json.ObjectMap.init(self.allocator);
        try output.put("condition_result", std.json.Value{ .bool = result });
        try output.put("branch", std.json.Value{ .string = if (result) "true" else "false" });
        
        return std.json.Value{ .object = output };
    }
    
    pub fn evaluateCondition(self: *const ConditionNode, ctx: *const ExecutionContext) !bool {
        // Simple condition evaluation
        // In production, this would use a proper expression parser
        
        // Support simple comparisons like: "input.value > 10"
        // For now, just check if we have a boolean input
        if (ctx.getInputBool("condition")) |result| {
            return result;
        }
        
        // Check for "input.value" existence
        if (self.condition.len > 0) {
            const input_value = ctx.getInput("value");
            if (input_value) |_| {
                return true;
            }
        }
        
        return false;
    }
};

/// Transform node - data transformation
pub const TransformNode = struct {
    base: NodeInterface,
    transform_type: []const u8, // "map", "filter", "reduce", "merge", "split"
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        description: []const u8,
        transform_type: []const u8,
        inputs: []const Port,
        outputs: []const Port,
        config: std.json.Value,
    ) !TransformNode {
        const base = NodeInterface{
            .id = id,
            .name = name,
            .description = description,
            .node_type = "transform",
            .category = .transform,
            .inputs = inputs,
            .outputs = outputs,
            .config = config,
        };
        
        return TransformNode{
            .base = base,
            .transform_type = transform_type,
            .allocator = allocator,
        };
    }
    
    pub fn validate(self: *const TransformNode) !void {
        try self.base.validate();
        
        // Transform nodes must have at least one input and output
        if (self.base.inputs.len == 0) {
            return error.TransformMustHaveInput;
        }
        if (self.base.outputs.len == 0) {
            return error.TransformMustHaveOutput;
        }
        
        // Validate transform type
        const valid_types = [_][]const u8{ "map", "filter", "reduce", "merge", "split" };
        var valid = false;
        for (valid_types) |t| {
            if (std.mem.eql(u8, self.transform_type, t)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return error.InvalidTransformType;
        }
    }
    
    pub fn execute(self: *TransformNode, ctx: *ExecutionContext) !std.json.Value {
        try self.validate();
        try self.base.validateInputs(ctx);
        
        // Get input data
        const input_data = ctx.getInput("data");
        if (input_data == null) {
            return error.MissingInputData;
        }
        
        // Perform transformation based on type
        return switch (self.transform_type[0]) {
            'm' => if (std.mem.eql(u8, self.transform_type, "map"))
                try self.transformMap(ctx, input_data.?)
            else if (std.mem.eql(u8, self.transform_type, "merge"))
                try self.transformMerge(ctx, input_data.?)
            else
                error.UnknownTransform,
            'f' => try self.transformFilter(ctx, input_data.?),
            'r' => try self.transformReduce(ctx, input_data.?),
            's' => try self.transformSplit(ctx, input_data.?),
            else => error.UnknownTransform,
        };
    }
    
    fn transformMap(self: *const TransformNode, ctx: *const ExecutionContext, data: std.json.Value) !std.json.Value {
        _ = ctx;
        // Map transformation: apply function to each element
        // For now, return input as-is
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("transform", std.json.Value{ .string = "map" });
        try result.put("data", data);
        return std.json.Value{ .object = result };
    }
    
    fn transformFilter(self: *const TransformNode, ctx: *const ExecutionContext, data: std.json.Value) !std.json.Value {
        _ = ctx;
        // Filter transformation: keep elements matching predicate
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("transform", std.json.Value{ .string = "filter" });
        try result.put("data", data);
        return std.json.Value{ .object = result };
    }
    
    fn transformReduce(self: *const TransformNode, ctx: *const ExecutionContext, data: std.json.Value) !std.json.Value {
        _ = ctx;
        // Reduce transformation: aggregate elements
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("transform", std.json.Value{ .string = "reduce" });
        try result.put("data", data);
        return std.json.Value{ .object = result };
    }
    
    fn transformMerge(self: *const TransformNode, ctx: *const ExecutionContext, data: std.json.Value) !std.json.Value {
        _ = ctx;
        // Merge transformation: combine multiple inputs
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("transform", std.json.Value{ .string = "merge" });
        try result.put("data", data);
        return std.json.Value{ .object = result };
    }
    
    fn transformSplit(self: *const TransformNode, ctx: *const ExecutionContext, data: std.json.Value) !std.json.Value {
        _ = ctx;
        // Split transformation: split into multiple outputs
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("transform", std.json.Value{ .string = "split" });
        try result.put("data", data);
        return std.json.Value{ .object = result };
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "PortType matches values correctly" {
    const string_val = std.json.Value{ .string = "hello" };
    const number_val = std.json.Value{ .integer = 42 };
    const bool_val = std.json.Value{ .bool = true };
    
    try std.testing.expect(PortType.string.matches(string_val));
    try std.testing.expect(!PortType.string.matches(number_val));
    
    try std.testing.expect(PortType.number.matches(number_val));
    try std.testing.expect(!PortType.number.matches(string_val));
    
    try std.testing.expect(PortType.boolean.matches(bool_val));
    try std.testing.expect(!PortType.boolean.matches(string_val));
    
    try std.testing.expect(PortType.any.matches(string_val));
    try std.testing.expect(PortType.any.matches(number_val));
    try std.testing.expect(PortType.any.matches(bool_val));
}

test "Port validation" {
    const port = Port.init(
        "input1",
        "Input 1",
        "Test input port",
        .string,
        true,
        null,
    );
    
    // Valid string value
    const string_val = std.json.Value{ .string = "test" };
    try port.validateValue(string_val);
    
    // Invalid number value for string port
    const number_val = std.json.Value{ .integer = 42 };
    try std.testing.expectError(error.TypeMismatch, port.validateValue(number_val));
    
    // Missing required value
    try std.testing.expectError(error.RequiredPortMissing, port.validateValue(null));
}

test "ExecutionContext variable management" {
    const allocator = std.testing.allocator;
    
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", "user123");
    defer ctx.deinit();
    
    // Set and get variables
    try ctx.setVariable("key1", "value1");
    try std.testing.expectEqualStrings("value1", ctx.getVariable("key1").?);
    
    // Non-existent variable
    try std.testing.expect(ctx.getVariable("nonexistent") == null);
}

test "ExecutionContext input management" {
    const allocator = std.testing.allocator;
    
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    // Set and get inputs
    try ctx.setInput("port1", std.json.Value{ .string = "test" });
    const val = ctx.getInput("port1");
    try std.testing.expect(val != null);
    try std.testing.expectEqualStrings("test", val.?.string);
    
    // Convenience methods
    try ctx.setInput("str", std.json.Value{ .string = "hello" });
    try std.testing.expectEqualStrings("hello", ctx.getInputString("str").?);
    
    try ctx.setInput("num", std.json.Value{ .integer = 42 });
    try std.testing.expectEqual(@as(f64, 42.0), ctx.getInputNumber("num").?);
    
    try ctx.setInput("bool", std.json.Value{ .bool = true });
    try std.testing.expect(ctx.getInputBool("bool").?);
}

test "TriggerNode creation and validation" {
    const allocator = std.testing.allocator;
    
    const outputs = [_]Port{
        Port.init("output", "Output", "Trigger output", .object, false, null),
    };
    
    var trigger = try TriggerNode.init(
        allocator,
        "trigger1",
        "Webhook Trigger",
        "Receives webhook events",
        "webhook",
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    try trigger.validate();
    try std.testing.expectEqualStrings("trigger", trigger.base.node_type);
    try std.testing.expectEqualStrings("webhook", trigger.trigger_type);
}

test "TriggerNode execution" {
    const allocator = std.testing.allocator;
    
    const outputs = [_]Port{
        Port.init("output", "Output", "Trigger output", .object, false, null),
    };
    
    var trigger = try TriggerNode.init(
        allocator,
        "trigger1",
        "Manual Trigger",
        "Manual execution",
        "manual",
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", "user123");
    defer ctx.deinit();
    
    var result = try trigger.execute(&ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expect(result.object.get("triggered").?.bool);
}

test "ActionNode creation and validation" {
    const allocator = std.testing.allocator;
    
    const inputs = [_]Port{
        Port.init("url", "URL", "Request URL", .string, true, null),
    };
    
    const outputs = [_]Port{
        Port.init("response", "Response", "HTTP response", .object, false, null),
    };
    
    var action = try ActionNode.init(
        allocator,
        "action1",
        "HTTP Request",
        "Make HTTP request",
        "http_request",
        &inputs,
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    try action.validate();
    try std.testing.expectEqualStrings("action", action.base.node_type);
    try std.testing.expectEqualStrings("http_request", action.action_type);
}

test "ConditionNode creation and validation" {
    const allocator = std.testing.allocator;
    
    const inputs = [_]Port{
        Port.init("value", "Value", "Value to check", .any, true, null),
    };
    
    const outputs = [_]Port{
        Port.init("true", "True", "True branch", .any, false, null),
        Port.init("false", "False", "False branch", .any, false, null),
    };
    
    var condition = try ConditionNode.init(
        allocator,
        "condition1",
        "If/Else",
        "Branch based on condition",
        "input.value > 10",
        &inputs,
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    try condition.validate();
    try std.testing.expectEqualStrings("condition", condition.base.node_type);
}

test "TransformNode creation and validation" {
    const allocator = std.testing.allocator;
    
    const inputs = [_]Port{
        Port.init("data", "Data", "Input data", .array, true, null),
    };
    
    const outputs = [_]Port{
        Port.init("result", "Result", "Transformed data", .array, false, null),
    };
    
    var transform = try TransformNode.init(
        allocator,
        "transform1",
        "Map",
        "Transform array elements",
        "map",
        &inputs,
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    try transform.validate();
    try std.testing.expectEqualStrings("transform", transform.base.node_type);
    try std.testing.expectEqualStrings("map", transform.transform_type);
}

test "Node lifecycle with ExecutionContext" {
    const allocator = std.testing.allocator;
    
    // Create trigger node
    const outputs = [_]Port{
        Port.init("output", "Output", "Trigger output", .object, false, null),
    };
    
    var trigger = try TriggerNode.init(
        allocator,
        "trigger1",
        "Manual Trigger",
        "Start workflow",
        "manual",
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    // Create execution context
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", "user123");
    defer ctx.deinit();
    
    // Execute trigger
    var result = try trigger.execute(&ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    
    // Store result as input for next node
    try ctx.setInput("trigger_data", result);
    
    // Verify stored data
    const stored = ctx.getInput("trigger_data");
    try std.testing.expect(stored != null);
}
