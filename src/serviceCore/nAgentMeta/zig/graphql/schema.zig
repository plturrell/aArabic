//! GraphQL Schema - Day 31
//!
//! GraphQL schema definition for nMetaData API.
//! Provides type-safe GraphQL query, mutation, and subscription support.
//!
//! Key Features:
//! - Type definitions
//! - Query resolvers
//! - Mutation resolvers
//! - Subscription support
//! - Schema introspection
//!
//! Example Schema:
//! ```graphql
//! type Dataset {
//!   id: ID!
//!   name: String!
//!   type: DatasetType!
//!   upstream: [Dataset!]!
//!   downstream: [Dataset!]!
//! }
//!
//! type Query {
//!   dataset(id: ID!): Dataset
//!   datasets(page: Int, limit: Int): DatasetConnection!
//!   lineage(id: ID!, depth: Int): LineageGraph!
//! }
//!
//! type Mutation {
//!   createDataset(input: CreateDatasetInput!): Dataset!
//!   updateDataset(id: ID!, input: UpdateDatasetInput!): Dataset!
//!   deleteDataset(id: ID!, force: Boolean): Boolean!
//! }
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// GraphQL Type Kind
pub const TypeKind = enum {
    scalar,
    object,
    interface,
    union_type,
    enum_type,
    input_object,
    list,
    non_null,
    
    pub fn toString(self: TypeKind) []const u8 {
        return switch (self) {
            .scalar => "SCALAR",
            .object => "OBJECT",
            .interface => "INTERFACE",
            .union_type => "UNION",
            .enum_type => "ENUM",
            .input_object => "INPUT_OBJECT",
            .list => "LIST",
            .non_null => "NON_NULL",
        };
    }
};

/// GraphQL Scalar Types
pub const ScalarType = enum {
    int,
    float,
    string,
    boolean,
    id,
    
    pub fn toString(self: ScalarType) []const u8 {
        return switch (self) {
            .int => "Int",
            .float => "Float",
            .string => "String",
            .boolean => "Boolean",
            .id => "ID",
        };
    }
};

/// GraphQL Field Definition
pub const Field = struct {
    name: []const u8,
    type_name: []const u8,
    description: ?[]const u8 = null,
    is_nullable: bool = true,
    is_list: bool = false,
    arguments: []Argument = &[_]Argument{},
    
    pub fn format(self: Field, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}: ", .{self.name});
        if (self.is_list) try writer.writeAll("[");
        try writer.writeAll(self.type_name);
        if (!self.is_nullable) try writer.writeAll("!");
        if (self.is_list) try writer.writeAll("]");
    }
};

/// GraphQL Argument Definition
pub const Argument = struct {
    name: []const u8,
    type_name: []const u8,
    description: ?[]const u8 = null,
    is_nullable: bool = true,
    default_value: ?[]const u8 = null,
};

/// GraphQL Object Type Definition
pub const ObjectType = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    fields: []Field,
    interfaces: [][]const u8 = &[_][]const u8{},
};

/// GraphQL Input Type Definition
pub const InputType = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    fields: []Field,
};

/// GraphQL Enum Type Definition
pub const EnumType = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    values: []EnumValue,
};

/// GraphQL Enum Value
pub const EnumValue = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    deprecated: bool = false,
};

/// GraphQL Schema
pub const Schema = struct {
    allocator: Allocator,
    query_type: []const u8,
    mutation_type: ?[]const u8 = null,
    subscription_type: ?[]const u8 = null,
    types: std.StringHashMap(ObjectType),
    input_types: std.StringHashMap(InputType),
    enum_types: std.StringHashMap(EnumType),
    
    pub fn init(allocator: Allocator) Schema {
        return Schema{
            .allocator = allocator,
            .query_type = "Query",
            .mutation_type = "Mutation",
            .subscription_type = null,
            .types = std.StringHashMap(ObjectType).init(allocator),
            .input_types = std.StringHashMap(InputType).init(allocator),
            .enum_types = std.StringHashMap(EnumType).init(allocator),
        };
    }
    
    pub fn deinit(self: *Schema) void {
        self.types.deinit();
        self.input_types.deinit();
        self.enum_types.deinit();
    }
    
    /// Register an object type
    pub fn addType(self: *Schema, object_type: ObjectType) !void {
        try self.types.put(object_type.name, object_type);
    }
    
    /// Register an input type
    pub fn addInputType(self: *Schema, input_type: InputType) !void {
        try self.input_types.put(input_type.name, input_type);
    }
    
    /// Register an enum type
    pub fn addEnumType(self: *Schema, enum_type: EnumType) !void {
        try self.enum_types.put(enum_type.name, enum_type);
    }
    
    /// Get type by name
    pub fn getType(self: *const Schema, name: []const u8) ?ObjectType {
        return self.types.get(name);
    }
    
    /// Generate SDL (Schema Definition Language)
    pub fn toSDL(self: *const Schema, writer: anytype) !void {
        // Write schema directive
        try writer.print("schema {{\n", .{});
        try writer.print("  query: {s}\n", .{self.query_type});
        if (self.mutation_type) |mutation| {
            try writer.print("  mutation: {s}\n", .{mutation});
        }
        if (self.subscription_type) |subscription| {
            try writer.print("  subscription: {s}\n", .{subscription});
        }
        try writer.print("}}\n\n", .{});
        
        // Write enum types
        var enum_iter = self.enum_types.iterator();
        while (enum_iter.next()) |entry| {
            const enum_type = entry.value_ptr.*;
            if (enum_type.description) |desc| {
                try writer.print("\"\"\"{s}\"\"\"\n", .{desc});
            }
            try writer.print("enum {s} {{\n", .{enum_type.name});
            for (enum_type.values) |value| {
                if (value.description) |desc| {
                    try writer.print("  \"\"\"{s}\"\"\"\n", .{desc});
                }
                try writer.print("  {s}\n", .{value.name});
            }
            try writer.print("}}\n\n", .{});
        }
        
        // Write input types
        var input_iter = self.input_types.iterator();
        while (input_iter.next()) |entry| {
            const input_type = entry.value_ptr.*;
            if (input_type.description) |desc| {
                try writer.print("\"\"\"{s}\"\"\"\n", .{desc});
            }
            try writer.print("input {s} {{\n", .{input_type.name});
            for (input_type.fields) |field| {
                try writer.print("  {}\n", .{field});
            }
            try writer.print("}}\n\n", .{});
        }
        
        // Write object types
        var type_iter = self.types.iterator();
        while (type_iter.next()) |entry| {
            const object_type = entry.value_ptr.*;
            if (object_type.description) |desc| {
                try writer.print("\"\"\"{s}\"\"\"\n", .{desc});
            }
            try writer.print("type {s}", .{object_type.name});
            if (object_type.interfaces.len > 0) {
                try writer.print(" implements", .{});
                for (object_type.interfaces, 0..) |interface, i| {
                    if (i > 0) try writer.print(" &", .{});
                    try writer.print(" {s}", .{interface});
                }
            }
            try writer.print(" {{\n", .{});
            for (object_type.fields) |field| {
                if (field.description) |desc| {
                    try writer.print("  \"\"\"{s}\"\"\"\n", .{desc});
                }
                try writer.print("  {}", .{field});
                if (field.arguments.len > 0) {
                    try writer.print("(", .{});
                    for (field.arguments, 0..) |arg, i| {
                        if (i > 0) try writer.print(", ", .{});
                        try writer.print("{s}: {s}", .{ arg.name, arg.type_name });
                        if (!arg.is_nullable) try writer.print("!", .{});
                    }
                    try writer.print(")", .{});
                }
                try writer.print("\n", .{});
            }
            try writer.print("}}\n\n", .{});
        }
    }
};

/// Build nMetaData GraphQL Schema
pub fn buildNMetaDataSchema(allocator: Allocator) !Schema {
    var schema = Schema.init(allocator);
    
    // Enum: DatasetType
    try schema.addEnumType(EnumType{
        .name = "DatasetType",
        .description = "Type of dataset",
        .values = &[_]EnumValue{
            .{ .name = "TABLE", .description = "Database table" },
            .{ .name = "VIEW", .description = "Database view" },
            .{ .name = "PIPELINE", .description = "Data pipeline" },
            .{ .name = "STREAM", .description = "Data stream" },
            .{ .name = "FILE", .description = "File-based dataset" },
        },
    });
    
    // Type: Dataset
    try schema.addType(ObjectType{
        .name = "Dataset",
        .description = "A dataset in the metadata system",
        .fields = &[_]Field{
            .{ .name = "id", .type_name = "ID", .is_nullable = false, .description = "Unique identifier" },
            .{ .name = "name", .type_name = "String", .is_nullable = false, .description = "Dataset name" },
            .{ .name = "type", .type_name = "DatasetType", .is_nullable = false, .description = "Dataset type" },
            .{ .name = "schema", .type_name = "String", .description = "Schema name" },
            .{ .name = "description", .type_name = "String", .description = "Dataset description" },
            .{ .name = "createdAt", .type_name = "String", .is_nullable = false },
            .{ .name = "updatedAt", .type_name = "String", .is_nullable = false },
            .{ .name = "upstream", .type_name = "Dataset", .is_list = true, .is_nullable = false, .description = "Upstream dependencies" },
            .{ .name = "downstream", .type_name = "Dataset", .is_list = true, .is_nullable = false, .description = "Downstream consumers" },
        },
    });
    
    // Type: Query
    try schema.addType(ObjectType{
        .name = "Query",
        .description = "Root query type",
        .fields = &[_]Field{
            .{
                .name = "dataset",
                .type_name = "Dataset",
                .description = "Get dataset by ID",
                .arguments = &[_]Argument{
                    .{ .name = "id", .type_name = "ID", .is_nullable = false },
                },
            },
            .{
                .name = "datasets",
                .type_name = "Dataset",
                .is_list = true,
                .is_nullable = false,
                .description = "List all datasets",
                .arguments = &[_]Argument{
                    .{ .name = "page", .type_name = "Int", .default_value = "1" },
                    .{ .name = "limit", .type_name = "Int", .default_value = "10" },
                },
            },
        },
    });
    
    // Input: CreateDatasetInput
    try schema.addInputType(InputType{
        .name = "CreateDatasetInput",
        .description = "Input for creating a dataset",
        .fields = &[_]Field{
            .{ .name = "name", .type_name = "String", .is_nullable = false },
            .{ .name = "type", .type_name = "DatasetType", .is_nullable = false },
            .{ .name = "schema", .type_name = "String" },
            .{ .name = "description", .type_name = "String" },
        },
    });
    
    // Type: Mutation
    try schema.addType(ObjectType{
        .name = "Mutation",
        .description = "Root mutation type",
        .fields = &[_]Field{
            .{
                .name = "createDataset",
                .type_name = "Dataset",
                .is_nullable = false,
                .description = "Create a new dataset",
                .arguments = &[_]Argument{
                    .{ .name = "input", .type_name = "CreateDatasetInput", .is_nullable = false },
                },
            },
        },
    });
    
    return schema;
}

// ============================================================================
// Tests
// ============================================================================

test "Schema: init and deinit" {
    const allocator = std.testing.allocator;
    
    var schema = Schema.init(allocator);
    defer schema.deinit();
    
    try std.testing.expectEqualStrings("Query", schema.query_type);
}

test "Schema: add and get type" {
    const allocator = std.testing.allocator;
    
    var schema = Schema.init(allocator);
    defer schema.deinit();
    
    const dataset_type = ObjectType{
        .name = "Dataset",
        .fields = &[_]Field{
            .{ .name = "id", .type_name = "ID", .is_nullable = false },
        },
    };
    
    try schema.addType(dataset_type);
    
    const retrieved = schema.getType("Dataset");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings("Dataset", retrieved.?.name);
}

test "Schema: build nMetaData schema" {
    const allocator = std.testing.allocator;
    
    var schema = try buildNMetaDataSchema(allocator);
    defer schema.deinit();
    
    try std.testing.expectEqualStrings("Query", schema.query_type);
    try std.testing.expectEqualStrings("Mutation", schema.mutation_type.?);
    
    const dataset = schema.getType("Dataset");
    try std.testing.expect(dataset != null);
}

test "Schema: SDL generation" {
    const allocator = std.testing.allocator;
    
    var schema = try buildNMetaDataSchema(allocator);
    defer schema.deinit();
    
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit();
    
    try schema.toSDL(buffer.writer());
    
    try std.testing.expect(buffer.items.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "type Dataset") != null);
}
