//! Logger Component - Day 18
//! 
//! Logs workflow data for debugging and monitoring.
//! Supports multiple log levels and pass-through mode.
//!
//! Key Features:
//! - Multiple log levels (debug, info, warn, error)
//! - Message formatting
//! - Pass-through output
//! - Workflow monitoring

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

/// Log levels
pub const LogLevel = enum {
    debug,
    info,
    warn,
    err,
    
    pub fn fromString(str: []const u8) ?LogLevel {
        if (std.mem.eql(u8, str, "debug")) return .debug;
        if (std.mem.eql(u8, str, "info")) return .info;
        if (std.mem.eql(u8, str, "warn")) return .warn;
        if (std.mem.eql(u8, str, "error")) return .err;
        return null;
    }
    
    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
        };
    }
};

/// Logger Node implementation
pub const LoggerNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    level: LogLevel,
    message_template: []const u8,
    pass_through: bool,
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*LoggerNode {
        const node = try allocator.create(LoggerNode);
        errdefer allocator.destroy(node);
        
        node.* = LoggerNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "logger",
            .level = .info,
            .message_template = "",
            .pass_through = true,
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 1),
        };
        
        node.inputs[0] = Port{
            .description = "",
            .id = "data",
            .name = "Data",
            .port_type = .any,
            .required = true,
            .default_value = null,
        };
        
        node.outputs[0] = Port{
            .description = "",
            .id = "output",
            .name = "Output",
            .port_type = .any,
            .required = false,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *LoggerNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.message_template);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *LoggerNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("level")) |level_val| {
            if (level_val == .string) {
                self.level = LogLevel.fromString(level_val.string) orelse .info;
            }
        }
        
        if (config_obj.get("message")) |msg_val| {
            if (msg_val == .string) {
                self.message_template = try self.allocator.dupe(u8, msg_val.string);
            }
        }
        
        if (config_obj.get("pass_through")) |pass_val| {
            if (pass_val == .bool) {
                self.pass_through = pass_val.bool;
            }
        }
    }
    
    pub fn asNodeInterface(self: *LoggerNode) NodeInterface {
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
    
    fn validateImpl(_: *const NodeInterface) anyerror!void {
        // Logger always valid
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*LoggerNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        // Mock implementation - simulates logging
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("level", std.json.Value{ .string = self.level.toString() });
        try result.put("logged", std.json.Value{ .bool = true });
        try result.put("pass_through", std.json.Value{ .bool = self.pass_through });
        
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*LoggerNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const levels = [_][]const u8{ "debug", "info", "warn", "error" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("data", "Data", .any, true, "Data to log"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("output", "Output", .any, false, "Pass-through data"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.selectField(
            "level",
            true,
            "Log level",
            &levels,
            "info",
        ),
        ConfigSchemaField.stringField(
            "message",
            false,
            "Message template",
            "Processing: {data}",
        ),
        ConfigSchemaField.booleanField(
            "pass_through",
            false,
            "Pass data through to output",
            true,
        ),
    };
    
    const tags = [_][]const u8{ "logger", "debug", "utility", "monitoring" };
    const examples = [_][]const u8{
        "Debug: Log detailed information",
        "Info: Log general information",
        "Error: Log errors and issues",
    };
    
    return ComponentMetadata{
        .id = "logger",
        .name = "Logger",
        .version = "1.0.0",
        .description = "Log workflow data for debugging",
        .category = .utility,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "📝",
        .color = "#95A5A6",
        .tags = &tags,
        .help_text = "Log data flowing through the workflow for debugging and monitoring",
        .examples = &examples,
        .factory_fn = createLoggerNode,
    };
}

fn createLoggerNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const logger_node = try LoggerNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = logger_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "LogLevel string conversion" {
    try std.testing.expectEqual(LogLevel.debug, LogLevel.fromString("debug").?);
    try std.testing.expectEqual(LogLevel.info, LogLevel.fromString("info").?);
    try std.testing.expectEqualStrings("INFO", LogLevel.info.toString());
}

test "LoggerNode creation" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("level", std.json.Value{ .string = "info" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try LoggerNode.init(allocator, "log1", "Logger", config);
    defer node.deinit();
    
    try std.testing.expectEqual(LogLevel.info, node.level);
    try std.testing.expectEqual(true, node.pass_through);
}

test "Logger with custom message" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("level", std.json.Value{ .string = "debug" });
    try config_obj.put("message", std.json.Value{ .string = "Debug: {data}" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try LoggerNode.init(allocator, "log1", "Logger", config);
    defer node.deinit();
    
    try std.testing.expectEqualStrings("Debug: {data}", node.message_template);
}

test "Logger without pass-through" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("level", std.json.Value{ .string = "warn" });
    try config_obj.put("pass_through", std.json.Value{ .bool = false });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try LoggerNode.init(allocator, "log1", "Logger", config);
    defer node.deinit();
    
    try std.testing.expectEqual(false, node.pass_through);
}

test "Logger execute" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("level", std.json.Value{ .string = "error" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try LoggerNode.init(allocator, "log1", "Logger", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    const vtable = interface.vtable orelse return error.NoVTable;
    var result = try vtable.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("ERROR", result.object.get("level").?.string);
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("logger", metadata.id);
    try std.testing.expectEqualStrings("Logger", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.utility, metadata.category);
}
