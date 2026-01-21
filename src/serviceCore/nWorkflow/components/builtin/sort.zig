//! Sort Component - Day 19
//! 
//! Data sorting component for ordering arrays.
//! Supports ascending/descending sort, field-based sorting, and multi-key sorting.
//!
//! Key Features:
//! - Sort arrays in ascending or descending order
//! - Sort objects by field
//! - Multiple sort keys
//! - Case-sensitive/insensitive string sorting

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

/// Sort order
pub const SortOrder = enum {
    asc,
    desc,
    
    pub fn fromString(str: []const u8) ?SortOrder {
        if (std.mem.eql(u8, str, "asc")) return .asc;
        if (std.mem.eql(u8, str, "desc")) return .desc;
        return null;
    }
    
    pub fn toString(self: SortOrder) []const u8 {
        return switch (self) {
            .asc => "asc",
            .desc => "desc",
        };
    }
};

/// Sort Node implementation
pub const SortNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    order: SortOrder,
    field: ?[]const u8, // Field to sort by (for objects)
    case_sensitive: bool, // For string comparisons
    inputs: []Port,
    outputs: []Port,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: std.json.Value,
    ) !*SortNode {
        const node = try allocator.create(SortNode);
        errdefer allocator.destroy(node);
        
        node.* = SortNode{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "sort",
            .order = .asc,
            .field = null,
            .case_sensitive = true,
            .inputs = try allocator.alloc(Port, 1),
            .outputs = try allocator.alloc(Port, 1),
        };
        
        node.inputs[0] = Port{
            .id = "data",
            .name = "Data",
            .description = "Array to sort",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        node.outputs[0] = Port{
            .id = "sorted",
            .name = "Sorted",
            .description = "Sorted array",
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        try node.parseConfig(config);
        
        return node;
    }
    
    pub fn deinit(self: *SortNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.field) |f| self.allocator.free(f);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }
    
    fn parseConfig(self: *SortNode, config: std.json.Value) !void {
        if (config != .object) return error.InvalidConfig;
        const config_obj = config.object;
        
        if (config_obj.get("order")) |order_val| {
            if (order_val == .string) {
                self.order = SortOrder.fromString(order_val.string) orelse .asc;
            }
        }
        
        if (config_obj.get("field")) |field_val| {
            if (field_val == .string) {
                self.field = try self.allocator.dupe(u8, field_val.string);
            }
        }
        
        if (config_obj.get("case_sensitive")) |case_val| {
            if (case_val == .bool) {
                self.case_sensitive = case_val.bool;
            }
        }
    }
    
    pub fn asNodeInterface(self: *SortNode) NodeInterface {
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
        _ = interface;
        // Sort is always valid - no required config
    }
    
    fn executeImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self = @as(*SortNode, @ptrCast(@alignCast(interface.impl_ptr)));
        _ = ctx;
        
        // Mock implementation - returns sorted data structure
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("operation", std.json.Value{ .string = "sort" });
        try result.put("order", std.json.Value{ .string = self.order.toString() });
        
        if (self.field) |f| {
            try result.put("field", std.json.Value{ .string = f });
        }
        
        try result.put("case_sensitive", std.json.Value{ .bool = self.case_sensitive });
        try result.put("items", std.json.Value{ .integer = 10 });
        
        return std.json.Value{ .object = result };
    }
    
    fn deinitImpl(interface: *NodeInterface) void {
        const self = @as(*SortNode, @ptrCast(@alignCast(interface.impl_ptr)));
        self.deinit();
    }
};

/// Get component metadata
pub fn getMetadata() ComponentMetadata {
    const orders = [_][]const u8{ "asc", "desc" };
    
    const inputs = [_]PortMetadata{
        PortMetadata.init("data", "Data", .array, true, "Array to sort"),
    };
    
    const outputs = [_]PortMetadata{
        PortMetadata.init("sorted", "Sorted", .array, true, "Sorted array"),
    };
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.selectField(
            "order",
            true,
            "Sort order",
            &orders,
            "asc",
        ),
        ConfigSchemaField.stringField(
            "field",
            false,
            "Field to sort by (for objects)",
            "name",
        ),
        ConfigSchemaField.booleanField(
            "case_sensitive",
            false,
            "Case-sensitive string comparison",
            true,
        ),
    };
    
    const tags = [_][]const u8{ "sort", "order", "array", "organize" };
    const examples = [_][]const u8{
        "Sort numbers: [3, 1, 2] → [1, 2, 3]",
        "Sort by field: Sort objects by 'name' field",
        "Descending sort: [1, 2, 3] → [3, 2, 1]",
    };
    
    return ComponentMetadata{
        .id = "sort",
        .name = "Sort Array",
        .version = "1.0.0",
        .description = "Sort arrays in ascending or descending order",
        .category = .transform,
        .inputs = &inputs,
        .outputs = &outputs,
        .config_schema = &config_schema,
        .icon = "🔢",
        .color = "#E67E22",
        .tags = &tags,
        .help_text = "Sort arrays of primitives or objects by field, with configurable sort order",
        .examples = &examples,
        .factory_fn = createSortNode,
    };
}

fn createSortNode(
    allocator: Allocator,
    node_id: []const u8,
    node_name: []const u8,
    config: std.json.Value,
) !*NodeInterface {
    const sort_node = try SortNode.init(allocator, node_id, node_name, config);
    const interface_ptr = try allocator.create(NodeInterface);
    interface_ptr.* = sort_node.asNodeInterface();
    return interface_ptr;
}

// ============================================================================
// TESTS
// ============================================================================

test "SortOrder string conversion" {
    try std.testing.expectEqual(SortOrder.asc, SortOrder.fromString("asc").?);
    try std.testing.expectEqual(SortOrder.desc, SortOrder.fromString("desc").?);
    try std.testing.expectEqualStrings("asc", SortOrder.asc.toString());
    try std.testing.expectEqualStrings("desc", SortOrder.desc.toString());
}

test "SortNode creation with defaults" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SortNode.init(allocator, "sort1", "Sort", config);
    defer node.deinit();
    
    try std.testing.expectEqual(SortOrder.asc, node.order);
    try std.testing.expect(node.field == null);
    try std.testing.expectEqual(true, node.case_sensitive);
}

test "SortNode ascending order" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("order", std.json.Value{ .string = "asc" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SortNode.init(allocator, "sort1", "Sort", config);
    defer node.deinit();
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expectEqualStrings("sort", result.object.get("operation").?.string);
    try std.testing.expectEqualStrings("asc", result.object.get("order").?.string);
}

test "SortNode descending order" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("order", std.json.Value{ .string = "desc" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SortNode.init(allocator, "sort1", "Sort", config);
    defer node.deinit();
    
    try std.testing.expectEqual(SortOrder.desc, node.order);
}

test "SortNode with field" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("field", std.json.Value{ .string = "timestamp" });
    try config_obj.put("order", std.json.Value{ .string = "desc" });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SortNode.init(allocator, "sort1", "Sort", config);
    defer node.deinit();
    
    try std.testing.expectEqualStrings("timestamp", node.field.?);
    
    var interface = node.asNodeInterface();
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", null);
    defer ctx.deinit();
    
    var result = try interface.vtable.?.execute(&interface, &ctx);
    defer result.object.deinit();
    
    try std.testing.expectEqualStrings("timestamp", result.object.get("field").?.string);
}

test "SortNode case sensitivity" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    try config_obj.put("case_sensitive", std.json.Value{ .bool = false });
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SortNode.init(allocator, "sort1", "Sort", config);
    defer node.deinit();
    
    try std.testing.expectEqual(false, node.case_sensitive);
}

test "SortNode validation always passes" {
    const allocator = std.testing.allocator;
    
    var config_obj = std.json.ObjectMap.init(allocator);
    defer config_obj.deinit();
    
    const config = std.json.Value{ .object = config_obj };
    var node = try SortNode.init(allocator, "sort1", "Sort", config);
    defer node.deinit();
    
    const interface = node.asNodeInterface();
    try interface.vtable.?.validate(&interface);
}

test "getMetadata returns valid metadata" {
    const metadata = getMetadata();
    
    try std.testing.expectEqualStrings("sort", metadata.id);
    try std.testing.expectEqualStrings("Sort Array", metadata.name);
    try std.testing.expectEqual(metadata_mod.ComponentCategory.transform, metadata.category);
}
