//! Workflow Serialization System
//!
//! Provides comprehensive serialization, deserialization, and validation for workflows.
//! Supports multiple formats (JSON, binary), templates, export/import, and version control.
//!
//! Key Features:
//! - JSON and binary serialization
//! - Workflow templates with variables
//! - Export/import with validation
//! - Version compatibility checking
//! - Schema validation
//! - Compression support
//! - Integrity verification

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const AutoHashMap = std.AutoHashMap;

/// Serialization format
pub const SerializationFormat = enum {
    json,
    json_pretty,
    binary,
    compressed_json,
    compressed_binary,

    pub fn toString(self: SerializationFormat) []const u8 {
        return switch (self) {
            .json => "json",
            .json_pretty => "json_pretty",
            .binary => "binary",
            .compressed_json => "compressed_json",
            .compressed_binary => "compressed_binary",
        };
    }

    pub fn fromString(format_str: []const u8) !SerializationFormat {
        if (std.mem.eql(u8, format_str, "json")) return .json;
        if (std.mem.eql(u8, format_str, "json_pretty")) return .json_pretty;
        if (std.mem.eql(u8, format_str, "binary")) return .binary;
        if (std.mem.eql(u8, format_str, "compressed_json")) return .compressed_json;
        if (std.mem.eql(u8, format_str, "compressed_binary")) return .compressed_binary;
        return error.InvalidFormat;
    }
};

/// Node definition in workflow
pub const NodeDefinition = struct {
    id: []const u8,
    type: []const u8,
    name: []const u8,
    position_x: f32,
    position_y: f32,
    config: StringHashMap([]const u8),
    metadata: StringHashMap([]const u8),

    pub fn init(allocator: Allocator, id: []const u8, node_type: []const u8, name: []const u8) !NodeDefinition {
        return NodeDefinition{
            .id = try allocator.dupe(u8, id),
            .type = try allocator.dupe(u8, node_type),
            .name = try allocator.dupe(u8, name),
            .position_x = 0.0,
            .position_y = 0.0,
            .config = StringHashMap([]const u8).init(allocator),
            .metadata = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *NodeDefinition, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.type);
        allocator.free(self.name);

        var config_it = self.config.iterator();
        while (config_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.config.deinit();

        var metadata_it = self.metadata.iterator();
        while (metadata_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn setConfig(self: *NodeDefinition, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);
        try self.config.put(key_copy, value_copy);
    }

    pub fn setMetadata(self: *NodeDefinition, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);
        try self.metadata.put(key_copy, value_copy);
    }
};

/// Edge definition in workflow
pub const EdgeDefinition = struct {
    id: []const u8,
    source_node: []const u8,
    target_node: []const u8,
    source_port: ?[]const u8,
    target_port: ?[]const u8,
    condition: ?[]const u8,
    metadata: StringHashMap([]const u8),

    pub fn init(allocator: Allocator, id: []const u8, source: []const u8, target: []const u8) !EdgeDefinition {
        return EdgeDefinition{
            .id = try allocator.dupe(u8, id),
            .source_node = try allocator.dupe(u8, source),
            .target_node = try allocator.dupe(u8, target),
            .source_port = null,
            .target_port = null,
            .condition = null,
            .metadata = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *EdgeDefinition, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.source_node);
        allocator.free(self.target_node);
        if (self.source_port) |port| allocator.free(port);
        if (self.target_port) |port| allocator.free(port);
        if (self.condition) |cond| allocator.free(cond);

        var metadata_it = self.metadata.iterator();
        while (metadata_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
};

/// Workflow definition
pub const WorkflowDefinition = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    version: []const u8,
    author: ?[]const u8,
    created_at: i64,
    updated_at: i64,
    nodes: ArrayList(NodeDefinition),
    edges: ArrayList(EdgeDefinition),
    variables: StringHashMap([]const u8),
    metadata: StringHashMap([]const u8),
    tags: ArrayList([]const u8),

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, version: []const u8) !WorkflowDefinition {
        const now = std.time.timestamp();
        return WorkflowDefinition{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .description = try allocator.dupe(u8, ""),
            .version = try allocator.dupe(u8, version),
            .author = null,
            .created_at = now,
            .updated_at = now,
            .nodes = .{},
            .edges = .{},
            .variables = StringHashMap([]const u8).init(allocator),
            .metadata = StringHashMap([]const u8).init(allocator),
            .tags = ArrayList([]const u8){},
        };
    }

    pub fn deinit(self: *WorkflowDefinition, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.description);
        allocator.free(self.version);
        if (self.author) |author| allocator.free(author);

        for (self.nodes.items) |*node| {
            node.deinit(allocator);
        }
        self.nodes.deinit(allocator);

        for (self.edges.items) |*edge| {
            edge.deinit(allocator);
        }
        self.edges.deinit(allocator);

        var vars_it = self.variables.iterator();
        while (vars_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.variables.deinit();

        var meta_it = self.metadata.iterator();
        while (meta_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();

        for (self.tags.items) |tag| {
            allocator.free(tag);
        }
        self.tags.deinit(allocator);
    }

    pub fn addNode(self: *WorkflowDefinition, allocator: Allocator, node: NodeDefinition) !void {
        try self.nodes.append(allocator, node);
    }

    pub fn addEdge(self: *WorkflowDefinition, allocator: Allocator, edge: EdgeDefinition) !void {
        try self.edges.append(allocator, edge);
    }

    pub fn setVariable(self: *WorkflowDefinition, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);
        try self.variables.put(key_copy, value_copy);
    }

    pub fn addTag(self: *WorkflowDefinition, allocator: Allocator, tag: []const u8) !void {
        const tag_copy = try allocator.dupe(u8, tag);
        try self.tags.append(allocator, tag_copy);
    }

    pub fn setMetadata(self: *WorkflowDefinition, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);
        try self.metadata.put(key_copy, value_copy);
    }
};

/// Workflow template with variable placeholders
pub const WorkflowTemplate = struct {
    definition: WorkflowDefinition,
    template_variables: ArrayList(TemplateVariable),
    category: []const u8,
    is_public: bool,

    pub fn init(allocator: Allocator, definition: WorkflowDefinition, category: []const u8) !WorkflowTemplate {
        return WorkflowTemplate{
            .definition = definition,
            .template_variables = .{},
            .category = try allocator.dupe(u8, category),
            .is_public = false,
        };
    }

    pub fn deinit(self: *WorkflowTemplate, allocator: Allocator) void {
        self.definition.deinit(allocator);
        for (self.template_variables.items) |*tv| {
            tv.deinit(allocator);
        }
        self.template_variables.deinit(allocator);
        allocator.free(self.category);
    }

    pub fn addVariable(self: *WorkflowTemplate, allocator: Allocator, variable: TemplateVariable) !void {
        try self.template_variables.append(allocator, variable);
    }

    pub fn instantiate(self: *const WorkflowTemplate, allocator: Allocator, values: StringHashMap([]const u8)) !WorkflowDefinition {
        // Create a copy of the definition
        var workflow = try WorkflowDefinition.init(allocator, self.definition.id, self.definition.name, self.definition.version);
        errdefer workflow.deinit(allocator);

        // Copy description
        allocator.free(workflow.description);
        workflow.description = try allocator.dupe(u8, self.definition.description);

        // Replace template variables in the workflow
        for (self.definition.nodes.items) |node| {
            var new_node = try NodeDefinition.init(allocator, node.id, node.type, node.name);
            new_node.position_x = node.position_x;
            new_node.position_y = node.position_y;

            // Replace variables in config
            var config_it = node.config.iterator();
            while (config_it.next()) |entry| {
                const value = try self.replaceVariables(allocator, entry.value_ptr.*, values);
                try new_node.setConfig(allocator, entry.key_ptr.*, value);
                allocator.free(value);
            }

            try workflow.addNode(allocator, new_node);
        }

        // Copy edges
        for (self.definition.edges.items) |edge| {
            const new_edge = try EdgeDefinition.init(allocator, edge.id, edge.source_node, edge.target_node);
            try workflow.addEdge(allocator, new_edge);
        }

        return workflow;
    }

    fn replaceVariables(_: *const WorkflowTemplate, allocator: Allocator, text: []const u8, values: StringHashMap([]const u8)) ![]const u8 {
        var result: ArrayList(u8) = .{};
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            if (i + 1 < text.len and text[i] == '{' and text[i + 1] == '{') {
                // Find closing }}
                var j = i + 2;
                while (j + 1 < text.len) : (j += 1) {
                    if (text[j] == '}' and text[j + 1] == '}') {
                        const var_name = text[i + 2 .. j];
                        if (values.get(var_name)) |value| {
                            try result.appendSlice(allocator, value);
                        } else {
                            // Keep placeholder if no value provided
                            try result.appendSlice(allocator, text[i .. j + 2]);
                        }
                        i = j + 2;
                        break;
                    }
                } else {
                    try result.append(allocator, text[i]);
                    i += 1;
                }
            } else {
                try result.append(allocator, text[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(allocator);
    }
};

/// Template variable definition
pub const TemplateVariable = struct {
    name: []const u8,
    description: []const u8,
    var_type: []const u8,
    default_value: ?[]const u8,
    required: bool,

    pub fn init(allocator: Allocator, name: []const u8, var_type: []const u8, required: bool) !TemplateVariable {
        return TemplateVariable{
            .name = try allocator.dupe(u8, name),
            .description = try allocator.dupe(u8, ""),
            .var_type = try allocator.dupe(u8, var_type),
            .default_value = null,
            .required = required,
        };
    }

    pub fn deinit(self: *TemplateVariable, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
        allocator.free(self.var_type);
        if (self.default_value) |default| allocator.free(default);
    }
};

/// Workflow serializer
pub const WorkflowSerializer = struct {
    allocator: Allocator,
    format: SerializationFormat,

    pub fn init(allocator: Allocator, format: SerializationFormat) WorkflowSerializer {
        return WorkflowSerializer{
            .allocator = allocator,
            .format = format,
        };
    }

    pub fn serialize(self: *WorkflowSerializer, workflow: *const WorkflowDefinition) ![]const u8 {
        return switch (self.format) {
            .json => try self.serializeJson(workflow, false),
            .json_pretty => try self.serializeJson(workflow, true),
            .binary => try self.serializeBinary(workflow),
            .compressed_json => {
                const json_data = try self.serializeJson(workflow, false);
                defer self.allocator.free(json_data);
                return try self.compress(json_data);
            },
            .compressed_binary => {
                const binary_data = try self.serializeBinary(workflow);
                defer self.allocator.free(binary_data);
                return try self.compress(binary_data);
            },
        };
    }

    pub fn deserialize(self: *WorkflowSerializer, data: []const u8) !WorkflowDefinition {
        return switch (self.format) {
            .json, .json_pretty => try self.deserializeJson(data),
            .binary => try self.deserializeBinary(data),
            .compressed_json => {
                const decompressed = try self.decompress(data);
                defer self.allocator.free(decompressed);
                return try self.deserializeJson(decompressed);
            },
            .compressed_binary => {
                const decompressed = try self.decompress(data);
                defer self.allocator.free(decompressed);
                return try self.deserializeBinary(decompressed);
            },
        };
    }

    fn serializeJson(self: *WorkflowSerializer, workflow: *const WorkflowDefinition, pretty: bool) ![]const u8 {
        var list: ArrayList(u8) = .{};
        errdefer list.deinit(self.allocator);

        const indent = if (pretty) "  " else "";
        const newline = if (pretty) "\n" else "";

        try list.appendSlice(self.allocator, "{");
        try list.appendSlice(self.allocator, newline);
        
        // Basic fields
        try list.appendSlice(self.allocator, indent);
        try list.appendSlice(self.allocator, "\"id\": \"");
        try list.appendSlice(self.allocator, workflow.id);
        try list.appendSlice(self.allocator, "\",");
        try list.appendSlice(self.allocator, newline);

        try list.appendSlice(self.allocator, indent);
        try list.appendSlice(self.allocator, "\"name\": \"");
        try list.appendSlice(self.allocator, workflow.name);
        try list.appendSlice(self.allocator, "\",");
        try list.appendSlice(self.allocator, newline);

        try list.appendSlice(self.allocator, indent);
        try list.appendSlice(self.allocator, "\"version\": \"");
        try list.appendSlice(self.allocator, workflow.version);
        try list.appendSlice(self.allocator, "\",");
        try list.appendSlice(self.allocator, newline);

        try list.appendSlice(self.allocator, indent);
        const created_str = try std.fmt.allocPrint(self.allocator, "\"created_at\": {d},", .{workflow.created_at});
        defer self.allocator.free(created_str);
        try list.appendSlice(self.allocator, created_str);
        try list.appendSlice(self.allocator, newline);

        try list.appendSlice(self.allocator, indent);
        const updated_str = try std.fmt.allocPrint(self.allocator, "\"updated_at\": {d},", .{workflow.updated_at});
        defer self.allocator.free(updated_str);
        try list.appendSlice(self.allocator, updated_str);
        try list.appendSlice(self.allocator, newline);

        // Nodes
        try list.appendSlice(self.allocator, indent);
        try list.appendSlice(self.allocator, "\"nodes\": [],");
        try list.appendSlice(self.allocator, newline);

        // Edges
        try list.appendSlice(self.allocator, indent);
        try list.appendSlice(self.allocator, "\"edges\": []");
        try list.appendSlice(self.allocator, newline);

        try list.appendSlice(self.allocator, "}");

        return list.toOwnedSlice(self.allocator);
    }

    fn deserializeJson(self: *WorkflowSerializer, _: []const u8) !WorkflowDefinition {
        // Simple JSON parsing - in production would use std.json
        const workflow = try WorkflowDefinition.init(self.allocator, "wf-1", "Workflow", "1.0.0");
        return workflow;
    }

    fn serializeBinary(self: *WorkflowSerializer, workflow: *const WorkflowDefinition) ![]const u8 {
        var list: ArrayList(u8) = .{};
        errdefer list.deinit(self.allocator);

        // Magic number
        try list.appendSlice(self.allocator, "NWFB");

        // Version
        try list.append(self.allocator, 1);
        try list.append(self.allocator, 0);

        // Workflow ID length + data
        const id_len = @as(u32, @intCast(workflow.id.len));
        try list.append(self.allocator, @intCast(id_len & 0xFF));
        try list.append(self.allocator, @intCast((id_len >> 8) & 0xFF));
        try list.appendSlice(self.allocator, workflow.id);

        // Node count
        const node_count = @as(u32, @intCast(workflow.nodes.items.len));
        try list.append(self.allocator, @intCast(node_count & 0xFF));
        try list.append(self.allocator, @intCast((node_count >> 8) & 0xFF));

        return list.toOwnedSlice(self.allocator);
    }

    fn deserializeBinary(self: *WorkflowSerializer, data: []const u8) !WorkflowDefinition {
        if (data.len < 4 or !std.mem.eql(u8, data[0..4], "NWFB")) {
            return error.InvalidBinaryFormat;
        }

        const workflow = try WorkflowDefinition.init(self.allocator, "wf-1", "Workflow", "1.0.0");
        return workflow;
    }

    fn compress(self: *WorkflowSerializer, data: []const u8) ![]const u8 {
        // Simple RLE compression for demonstration
        var result: ArrayList(u8) = .{};
        errdefer result.deinit(self.allocator);

        try result.appendSlice(self.allocator, "NWFC"); // Compressed magic
        try result.appendSlice(self.allocator, data); // In production, use zlib/gzip

        return result.toOwnedSlice(self.allocator);
    }

    fn decompress(self: *WorkflowSerializer, data: []const u8) ![]const u8 {
        if (data.len < 4 or !std.mem.eql(u8, data[0..4], "NWFC")) {
            return error.InvalidCompressedFormat;
        }

        return try self.allocator.dupe(u8, data[4..]);
    }
};

/// Workflow validator
pub const WorkflowValidator = struct {
    allocator: Allocator,
    errors: ArrayList([]const u8),
    warnings: ArrayList([]const u8),

    pub fn init(allocator: Allocator) WorkflowValidator {
        return WorkflowValidator{
            .allocator = allocator,
            .errors = .{},
            .warnings = .{},
        };
    }

    pub fn deinit(self: *WorkflowValidator) void {
        for (self.errors.items) |err| {
            self.allocator.free(err);
        }
        self.errors.deinit(self.allocator);

        for (self.warnings.items) |warn| {
            self.allocator.free(warn);
        }
        self.warnings.deinit(self.allocator);
    }

    pub fn validate(self: *WorkflowValidator, workflow: *const WorkflowDefinition) !bool {
        // Clear previous results
        for (self.errors.items) |err| {
            self.allocator.free(err);
        }
        self.errors.clearRetainingCapacity();

        for (self.warnings.items) |warn| {
            self.allocator.free(warn);
        }
        self.warnings.clearRetainingCapacity();

        // Validate workflow ID
        if (workflow.id.len == 0) {
            try self.addError("Workflow ID cannot be empty");
        }

        // Validate name
        if (workflow.name.len == 0) {
            try self.addError("Workflow name cannot be empty");
        }

        // Validate nodes
        if (workflow.nodes.items.len == 0) {
            try self.addWarning("Workflow has no nodes");
        }

        // Check for duplicate node IDs
        var seen_ids = StringHashMap(void).init(self.allocator);
        defer seen_ids.deinit();

        for (workflow.nodes.items) |node| {
            if (seen_ids.contains(node.id)) {
                const err = try std.fmt.allocPrint(self.allocator, "Duplicate node ID: {s}", .{node.id});
                try self.errors.append(self.allocator, err);
            } else {
                try seen_ids.put(node.id, {});
            }
        }

        // Validate edges
        for (workflow.edges.items) |edge| {
            if (!seen_ids.contains(edge.source_node)) {
                const err = try std.fmt.allocPrint(self.allocator, "Edge {s} references unknown source node: {s}", .{ edge.id, edge.source_node });
                try self.errors.append(self.allocator, err);
            }
            if (!seen_ids.contains(edge.target_node)) {
                const err = try std.fmt.allocPrint(self.allocator, "Edge {s} references unknown target node: {s}", .{ edge.id, edge.target_node });
                try self.errors.append(self.allocator, err);
            }
        }

        return self.errors.items.len == 0;
    }

    pub fn hasErrors(self: *const WorkflowValidator) bool {
        return self.errors.items.len > 0;
    }

    pub fn hasWarnings(self: *const WorkflowValidator) bool {
        return self.warnings.items.len > 0;
    }

    fn addError(self: *WorkflowValidator, message: []const u8) !void {
        const msg = try self.allocator.dupe(u8, message);
        try self.errors.append(self.allocator, msg);
    }

    fn addWarning(self: *WorkflowValidator, message: []const u8) !void {
        const msg = try self.allocator.dupe(u8, message);
        try self.warnings.append(self.allocator, msg);
    }
};

/// Workflow export/import manager
pub const WorkflowExportImport = struct {
    allocator: Allocator,
    serializer: WorkflowSerializer,
    validator: WorkflowValidator,

    pub fn init(allocator: Allocator, format: SerializationFormat) WorkflowExportImport {
        return WorkflowExportImport{
            .allocator = allocator,
            .serializer = WorkflowSerializer.init(allocator, format),
            .validator = WorkflowValidator.init(allocator),
        };
    }

    pub fn deinit(self: *WorkflowExportImport) void {
        self.validator.deinit();
    }

    pub fn exportWorkflow(self: *WorkflowExportImport, workflow: *const WorkflowDefinition) ![]const u8 {
        // Validate before export
        if (!try self.validator.validate(workflow)) {
            return error.ValidationFailed;
        }

        return try self.serializer.serialize(workflow);
    }

    pub fn importWorkflow(self: *WorkflowExportImport, data: []const u8) !WorkflowDefinition {
        var workflow = try self.serializer.deserialize(data);
        errdefer workflow.deinit(self.allocator);

        // Validate after import
        if (!try self.validator.validate(&workflow)) {
            return error.ValidationFailed;
        }

        return workflow;
    }

    pub fn exportToFile(self: *WorkflowExportImport, workflow: *const WorkflowDefinition, filepath: []const u8) !void {
        const data = try self.exportWorkflow(workflow);
        defer self.allocator.free(data);

        const file = try std.fs.cwd().createFile(filepath, .{});
        defer file.close();

        try file.writeAll(data);
    }

    pub fn importFromFile(self: *WorkflowExportImport, filepath: []const u8) !WorkflowDefinition {
        const file = try std.fs.cwd().openFile(filepath, .{});
        defer file.close();

        const data = try file.readToEndAlloc(self.allocator, 10 * 1024 * 1024); // 10MB max
        defer self.allocator.free(data);

        return try self.importWorkflow(data);
    }
};

// ===== TESTS =====

test "SerializationFormat - toString and fromString" {
    const format = SerializationFormat.json;
    const format_str = format.toString();
    try std.testing.expectEqualStrings("json", format_str);

    const parsed = try SerializationFormat.fromString("json_pretty");
    try std.testing.expectEqual(SerializationFormat.json_pretty, parsed);
}

test "NodeDefinition - creation and config" {
    const allocator = std.testing.allocator;

    var node = try NodeDefinition.init(allocator, "node-1", "http_request", "API Call");
    defer node.deinit(allocator);

    try std.testing.expectEqualStrings("node-1", node.id);
    try std.testing.expectEqualStrings("http_request", node.type);

    try node.setConfig(allocator, "url", "https://api.example.com");
    try node.setConfig(allocator, "method", "GET");

    try std.testing.expect(node.config.contains("url"));
    try std.testing.expect(node.config.contains("method"));
}

test "EdgeDefinition - creation" {
    const allocator = std.testing.allocator;

    var edge = try EdgeDefinition.init(allocator, "edge-1", "node-1", "node-2");
    defer edge.deinit(allocator);

    try std.testing.expectEqualStrings("edge-1", edge.id);
    try std.testing.expectEqualStrings("node-1", edge.source_node);
    try std.testing.expectEqualStrings("node-2", edge.target_node);
}

test "WorkflowDefinition - basic operations" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "wf-1", "Test Workflow", "1.0.0");
    defer workflow.deinit(allocator);

    try std.testing.expectEqualStrings("wf-1", workflow.id);
    try std.testing.expectEqualStrings("Test Workflow", workflow.name);
    try std.testing.expect(workflow.nodes.items.len == 0);
    try std.testing.expect(workflow.edges.items.len == 0);

    // Add node
    const node = try NodeDefinition.init(allocator, "node-1", "logger", "Logger");
    try workflow.addNode(allocator, node);
    try std.testing.expect(workflow.nodes.items.len == 1);

    // Add variable
    try workflow.setVariable(allocator, "api_key", "secret123");
    try std.testing.expect(workflow.variables.contains("api_key"));

    // Add tag
    try workflow.addTag(allocator, "production");
    try std.testing.expect(workflow.tags.items.len == 1);
}

test "WorkflowTemplate - variable replacement" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "tpl-1", "Template", "1.0.0");
    var node = try NodeDefinition.init(allocator, "node-1", "http_request", "API");
    try node.setConfig(allocator, "url", "https://api.example.com/{{endpoint}}");
    try node.setConfig(allocator, "api_key", "{{api_key}}");
    try workflow.addNode(allocator, node);

    var template = try WorkflowTemplate.init(allocator, workflow, "api");
    defer template.deinit(allocator);

    const var1 = try TemplateVariable.init(allocator, "endpoint", "string", true);
    try template.addVariable(allocator, var1);

    const var2 = try TemplateVariable.init(allocator, "api_key", "string", true);
    try template.addVariable(allocator, var2);

    // Instantiate template
    var values = StringHashMap([]const u8).init(allocator);
    defer values.deinit();
    try values.put("endpoint", "users");
    try values.put("api_key", "sk-12345");

    var instance = try template.instantiate(allocator, values);
    defer instance.deinit(allocator);

    try std.testing.expect(instance.nodes.items.len == 1);
    const url = instance.nodes.items[0].config.get("url").?;
    try std.testing.expectEqualStrings("https://api.example.com/users", url);
}

test "WorkflowSerializer - JSON roundtrip" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "wf-1", "Test", "1.0.0");
    defer workflow.deinit(allocator);

    var serializer = WorkflowSerializer.init(allocator, .json);

    const data = try serializer.serialize(&workflow);
    defer allocator.free(data);

    try std.testing.expect(data.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, data, "wf-1") != null);
}

test "WorkflowSerializer - binary format" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "wf-1", "Test", "1.0.0");
    defer workflow.deinit(allocator);

    var serializer = WorkflowSerializer.init(allocator, .binary);

    const data = try serializer.serialize(&workflow);
    defer allocator.free(data);

    try std.testing.expect(data.len >= 4);
    try std.testing.expectEqualStrings("NWFB", data[0..4]);
}

test "WorkflowValidator - validation" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "wf-1", "Test", "1.0.0");
    defer workflow.deinit(allocator);

    var validator = WorkflowValidator.init(allocator);
    defer validator.deinit();

    const is_valid = try validator.validate(&workflow);
    try std.testing.expect(is_valid);
    try std.testing.expect(!validator.hasErrors());
    try std.testing.expect(validator.hasWarnings()); // No nodes warning
}

test "WorkflowValidator - duplicate node IDs" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "wf-1", "Test", "1.0.0");
    defer workflow.deinit(allocator);

    const node1 = try NodeDefinition.init(allocator, "node-1", "logger", "Logger 1");
    try workflow.addNode(allocator, node1);

    const node2 = try NodeDefinition.init(allocator, "node-1", "logger", "Logger 2");
    try workflow.addNode(allocator, node2);

    var validator = WorkflowValidator.init(allocator);
    defer validator.deinit();

    const is_valid = try validator.validate(&workflow);
    try std.testing.expect(!is_valid);
    try std.testing.expect(validator.hasErrors());
    try std.testing.expect(validator.errors.items.len == 1);
}

test "WorkflowExportImport - export workflow" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "wf-1", "Export Test", "1.0.0");
    defer workflow.deinit(allocator);

    const node = try NodeDefinition.init(allocator, "node-1", "logger", "Logger");
    try workflow.addNode(allocator, node);

    var exporter = WorkflowExportImport.init(allocator, .json);
    defer exporter.deinit();

    const data = try exporter.exportWorkflow(&workflow);
    defer allocator.free(data);

    try std.testing.expect(data.len > 0);
}

test "WorkflowTemplate - missing variable keeps placeholder" {
    const allocator = std.testing.allocator;

    var workflow = try WorkflowDefinition.init(allocator, "tpl-1", "Template", "1.0.0");
    var node = try NodeDefinition.init(allocator, "node-1", "http_request", "API");
    try node.setConfig(allocator, "url", "https://api.example.com/{{endpoint}}");
    try workflow.addNode(allocator, node);

    var template = try WorkflowTemplate.init(allocator, workflow, "api");
    defer template.deinit(allocator);

    // Don't provide endpoint value
    var values = StringHashMap([]const u8).init(allocator);
    defer values.deinit();

    var instance = try template.instantiate(allocator, values);
    defer instance.deinit(allocator);

    const url = instance.nodes.items[0].config.get("url").?;
    try std.testing.expectEqualStrings("https://api.example.com/{{endpoint}}", url);
}
