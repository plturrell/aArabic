//! Component Metadata System - Day 16
//! 
//! This module defines the metadata structure that describes workflow components.
//! Components are reusable building blocks with defined inputs, outputs, and configuration.
//!
//! Key Features:
//! - Component identity (ID, name, version, category)
//! - Input/output port schema with type validation
//! - Configuration schema with field definitions
//! - UI metadata (icon, color, tags)
//! - Documentation (help text, examples)
//! - Factory function for node creation

const std = @import("std");
const node_types = @import("node_types");
const Allocator = std.mem.Allocator;

/// Component category for organization and filtering
pub const ComponentCategory = enum {
    trigger,      // Workflow triggers (webhooks, cron, manual)
    action,       // Actions/operations (API calls, database ops)
    transform,    // Data transformation (map, filter, reduce)
    logic,        // Conditional/control flow (if, switch, loop)
    integration,  // External service integration (HTTP, GraphQL)
    llm,         // LLM operations (chat, embed, prompt)
    data,        // Data sources/sinks (files, databases, caches)
    utility,     // Utilities (logger, delay, variable)
    
    pub fn toString(self: ComponentCategory) []const u8 {
        return switch (self) {
            .trigger => "trigger",
            .action => "action",
            .transform => "transform",
            .logic => "logic",
            .integration => "integration",
            .llm => "llm",
            .data => "data",
            .utility => "utility",
        };
    }
    
    pub fn fromString(str: []const u8) ?ComponentCategory {
        if (std.mem.eql(u8, str, "trigger")) return .trigger;
        if (std.mem.eql(u8, str, "action")) return .action;
        if (std.mem.eql(u8, str, "transform")) return .transform;
        if (std.mem.eql(u8, str, "logic")) return .logic;
        if (std.mem.eql(u8, str, "integration")) return .integration;
        if (std.mem.eql(u8, str, "llm")) return .llm;
        if (std.mem.eql(u8, str, "data")) return .data;
        if (std.mem.eql(u8, str, "utility")) return .utility;
        return null;
    }
};

/// Metadata for a single port (input or output)
pub const PortMetadata = struct {
    id: []const u8,
    name: []const u8,
    port_type: node_types.PortType,
    required: bool,
    default_value: ?std.json.Value,
    description: []const u8,
    
    pub fn init(
        id: []const u8,
        name: []const u8,
        port_type: node_types.PortType,
        required: bool,
        description: []const u8,
    ) PortMetadata {
        return PortMetadata{
            .id = id,
            .name = name,
            .port_type = port_type,
            .required = required,
            .default_value = null,
            .description = description,
        };
    }
    
    pub fn withDefault(
        id: []const u8,
        name: []const u8,
        port_type: node_types.PortType,
        required: bool,
        description: []const u8,
        default_value: std.json.Value,
    ) PortMetadata {
        return PortMetadata{
            .id = id,
            .name = name,
            .port_type = port_type,
            .required = required,
            .default_value = default_value,
            .description = description,
        };
    }
};

/// Configuration schema field definition
pub const ConfigSchemaField = struct {
    key: []const u8,
    field_type: []const u8, // "string", "number", "boolean", "select", "object", "array"
    required: bool,
    default_value: ?std.json.Value,
    options: ?[]const []const u8, // For select fields
    description: []const u8,
    placeholder: ?[]const u8,
    
    pub fn stringField(
        key: []const u8,
        required: bool,
        description: []const u8,
        placeholder: ?[]const u8,
    ) ConfigSchemaField {
        return ConfigSchemaField{
            .key = key,
            .field_type = "string",
            .required = required,
            .default_value = null,
            .options = null,
            .description = description,
            .placeholder = placeholder,
        };
    }
    
    pub fn numberField(
        key: []const u8,
        required: bool,
        description: []const u8,
        default_value: ?i64,
    ) ConfigSchemaField {
        const default_json = if (default_value) |v| 
            std.json.Value{ .integer = v } 
        else 
            null;
        
        return ConfigSchemaField{
            .key = key,
            .field_type = "number",
            .required = required,
            .default_value = default_json,
            .options = null,
            .description = description,
            .placeholder = null,
        };
    }
    
    pub fn booleanField(
        key: []const u8,
        required: bool,
        description: []const u8,
        default_value: bool,
    ) ConfigSchemaField {
        return ConfigSchemaField{
            .key = key,
            .field_type = "boolean",
            .required = required,
            .default_value = std.json.Value{ .bool = default_value },
            .options = null,
            .description = description,
            .placeholder = null,
        };
    }
    
    pub fn selectField(
        key: []const u8,
        required: bool,
        description: []const u8,
        options: []const []const u8,
        default_value: ?[]const u8,
    ) ConfigSchemaField {
        const default_json = if (default_value) |v| 
            std.json.Value{ .string = v } 
        else 
            null;
        
        return ConfigSchemaField{
            .key = key,
            .field_type = "select",
            .required = required,
            .default_value = default_json,
            .options = options,
            .description = description,
            .placeholder = null,
        };
    }
};

/// Complete component metadata
pub const ComponentMetadata = struct {
    // Identity
    id: []const u8,
    name: []const u8,
    version: []const u8,
    description: []const u8,
    category: ComponentCategory,
    
    // Schema
    inputs: []const PortMetadata,
    outputs: []const PortMetadata,
    config_schema: []const ConfigSchemaField,
    
    // UI
    icon: []const u8, // Icon name or emoji
    color: []const u8, // Hex color for UI (#RRGGBB)
    tags: []const []const u8,
    
    // Documentation
    help_text: []const u8,
    examples: []const []const u8,
    
    // Lifecycle - Factory function to create node instances
    factory_fn: *const fn (Allocator, []const u8, []const u8, std.json.Value) anyerror!*node_types.NodeInterface,
    
    pub fn validateConfig(self: *const ComponentMetadata, config: std.json.Value) !void {
        // Ensure config is an object
        if (config != .object) {
            return error.InvalidConfigType;
        }
        
        const config_obj = config.object;
        
        // Check required fields
        for (self.config_schema) |field| {
            if (field.required) {
                if (!config_obj.contains(field.key)) {
                    // Use default value if available
                    if (field.default_value == null) {
                        return error.MissingRequiredField;
                    }
                }
            }
            
            // Validate field type if present
            if (config_obj.get(field.key)) |value| {
                const valid = try self.validateFieldType(field, value);
                if (!valid) {
                    return error.InvalidFieldType;
                }
            }
        }
    }
    
    fn validateFieldType(
        self: *const ComponentMetadata,
        field: ConfigSchemaField,
        value: std.json.Value,
    ) !bool {
        _ = self;
        
        if (std.mem.eql(u8, field.field_type, "string")) {
            return value == .string;
        } else if (std.mem.eql(u8, field.field_type, "number")) {
            return value == .integer or value == .float;
        } else if (std.mem.eql(u8, field.field_type, "boolean")) {
            return value == .bool;
        } else if (std.mem.eql(u8, field.field_type, "object")) {
            return value == .object;
        } else if (std.mem.eql(u8, field.field_type, "array")) {
            return value == .array;
        } else if (std.mem.eql(u8, field.field_type, "select")) {
            // For select fields, validate value is in options
            if (value != .string) return false;
            if (field.options) |opts| {
                for (opts) |opt| {
                    if (std.mem.eql(u8, value.string, opt)) return true;
                }
                return false;
            }
            return true;
        }
        
        return true; // Unknown type, allow it
    }
    
    pub fn getConfigField(self: *const ComponentMetadata, key: []const u8) ?ConfigSchemaField {
        for (self.config_schema) |field| {
            if (std.mem.eql(u8, field.key, key)) {
                return field;
            }
        }
        return null;
    }
    
    pub fn hasTag(self: *const ComponentMetadata, tag: []const u8) bool {
        for (self.tags) |t| {
            if (std.mem.eql(u8, t, tag)) return true;
        }
        return false;
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "ComponentCategory string conversion" {
    try std.testing.expectEqualStrings("trigger", ComponentCategory.trigger.toString());
    try std.testing.expectEqualStrings("action", ComponentCategory.action.toString());
    try std.testing.expectEqualStrings("llm", ComponentCategory.llm.toString());
    
    try std.testing.expectEqual(ComponentCategory.trigger, ComponentCategory.fromString("trigger").?);
    try std.testing.expectEqual(ComponentCategory.action, ComponentCategory.fromString("action").?);
    try std.testing.expectEqual(@as(?ComponentCategory, null), ComponentCategory.fromString("invalid"));
}

test "PortMetadata creation" {
    const port = PortMetadata.init(
        "input1",
        "Input Data",
        .string,
        true,
        "Data to process",
    );
    
    try std.testing.expectEqualStrings("input1", port.id);
    try std.testing.expectEqualStrings("Input Data", port.name);
    try std.testing.expectEqual(node_types.PortType.string, port.port_type);
    try std.testing.expect(port.required);
    try std.testing.expectEqual(@as(?std.json.Value, null), port.default_value);
}

test "ConfigSchemaField string field" {
    const field = ConfigSchemaField.stringField(
        "api_key",
        true,
        "API key for authentication",
        "sk-...",
    );
    
    try std.testing.expectEqualStrings("api_key", field.key);
    try std.testing.expectEqualStrings("string", field.field_type);
    try std.testing.expect(field.required);
    try std.testing.expectEqualStrings("sk-...", field.placeholder.?);
}

test "ConfigSchemaField number field" {
    const field = ConfigSchemaField.numberField(
        "timeout",
        false,
        "Timeout in seconds",
        30,
    );
    
    try std.testing.expectEqualStrings("timeout", field.key);
    try std.testing.expectEqualStrings("number", field.field_type);
    try std.testing.expect(!field.required);
    try std.testing.expectEqual(@as(i64, 30), field.default_value.?.integer);
}

test "ConfigSchemaField select field" {
    const options = [_][]const u8{ "GET", "POST", "PUT", "DELETE" };
    const field = ConfigSchemaField.selectField(
        "method",
        true,
        "HTTP method",
        &options,
        "GET",
    );
    
    try std.testing.expectEqualStrings("method", field.key);
    try std.testing.expectEqualStrings("select", field.field_type);
    try std.testing.expectEqual(@as(usize, 4), field.options.?.len);
    try std.testing.expectEqualStrings("GET", field.default_value.?.string);
}

test "ComponentMetadata validation - valid config" {
    const allocator = std.testing.allocator;
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.stringField("url", true, "URL", null),
        ConfigSchemaField.numberField("timeout", false, "Timeout", 30),
    };
    
    const metadata = ComponentMetadata{
        .id = "test",
        .name = "Test Component",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &config_schema,
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "Help",
        .examples = &[_][]const u8{},
        .factory_fn = undefined,
    };
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    
    try config_obj.put("url", std.json.Value{ .string = "https://example.com" });
    try config_obj.put("timeout", std.json.Value{ .integer = 60 });
    
    const config = std.json.Value{ .object = config_obj };
    
    try metadata.validateConfig(config);
}

test "ComponentMetadata validation - missing required field" {
    const allocator = std.testing.allocator;
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.stringField("url", true, "URL", null),
    };
    
    const metadata = ComponentMetadata{
        .id = "test",
        .name = "Test Component",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &config_schema,
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "Help",
        .examples = &[_][]const u8{},
        .factory_fn = undefined,
    };
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    
    const config = std.json.Value{ .object = config_obj };
    
    try std.testing.expectError(error.MissingRequiredField, metadata.validateConfig(config));
}

test "ComponentMetadata hasTag" {
    const tags = [_][]const u8{ "http", "api", "rest" };
    
    const metadata = ComponentMetadata{
        .id = "test",
        .name = "Test Component",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🧪",
        .color = "#000000",
        .tags = &tags,
        .help_text = "Help",
        .examples = &[_][]const u8{},
        .factory_fn = undefined,
    };
    
    try std.testing.expect(metadata.hasTag("http"));
    try std.testing.expect(metadata.hasTag("api"));
    try std.testing.expect(!metadata.hasTag("graphql"));
}
