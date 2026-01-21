const std = @import("std");
const Allocator = std.mem.Allocator;
const node_types = @import("node_types");
const NodeInterface = node_types.NodeInterface;
const Port = node_types.Port;
const PortType = node_types.PortType;
const ExecutionContext = node_types.ExecutionContext;

// Import PostgreSQL client
const pg = @import("postgres_client.zig");
const PostgresClient = pg.PostgresClient;
const PostgresConnectionPool = pg.PostgresConnectionPool;
const PostgresPoolConfig = pg.PostgresPoolConfig;
const PgResult = pg.PgResult;
const PgValue = pg.PgValue;

/// PostgreSQL Query Node - Execute SELECT queries
pub const PostgresQueryNode = struct {
    allocator: Allocator,
    base: NodeInterface,
    connection_string: []const u8,
    query: []const u8,
    use_rls: bool, // Row-level security
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: std.json.Value) !*PostgresQueryNode {
        const node = try allocator.create(PostgresQueryNode);
        errdefer allocator.destroy(node);
        
        // Extract configuration
        const conn_str = if (config == .object) blk: {
            if (config.object.get("connection_string")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "postgres://localhost:5432/nworkflow";
        } else "postgres://localhost:5432/nworkflow";
        
        const query = if (config == .object) blk: {
            if (config.object.get("query")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "SELECT 1";
        } else "SELECT 1";
        
        const use_rls = if (config == .object) blk: {
            if (config.object.get("use_rls")) |v| {
                if (v == .bool) break :blk v.bool;
            }
            break :blk false;
        } else false;
        
        // Define input ports
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = .{
            .id = try allocator.dupe(u8, "parameters"),
            .name = try allocator.dupe(u8, "Query Parameters"),
            .description = try allocator.dupe(u8, "Parameters for prepared statement"),
            .port_type = .object,
            .required = false,
            .default_value = null,
        };
        
        // Define output ports
        const outputs = try allocator.alloc(Port, 2);
        outputs[0] = .{
            .id = try allocator.dupe(u8, "rows"),
            .name = try allocator.dupe(u8, "Result Rows"),
            .description = try allocator.dupe(u8, "Query result rows as array"),
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        outputs[1] = .{
            .id = try allocator.dupe(u8, "count"),
            .name = try allocator.dupe(u8, "Row Count"),
            .description = try allocator.dupe(u8, "Number of rows returned"),
            .port_type = .number,
            .required = true,
            .default_value = null,
        };
        
        node.* = .{
            .allocator = allocator,
            .base = .{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .node_type = try allocator.dupe(u8, "postgres_query"),
                .description = try allocator.dupe(u8, "Execute PostgreSQL query"),
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = config,
                
            },
            .connection_string = try allocator.dupe(u8, conn_str),
            .query = try allocator.dupe(u8, query),
            .use_rls = use_rls,
        };
        
        return node;
    }
    
    pub fn deinit(self: *PostgresQueryNode) void {
        const allocator = self.allocator;
        
        for (self.base.inputs) |input| {
            allocator.free(input.id);
            allocator.free(input.name);
            allocator.free(input.description);
        }
        allocator.free(self.base.inputs);
        
        for (self.base.outputs) |output| {
            allocator.free(output.id);
            allocator.free(output.name);
            allocator.free(output.description);
        }
        allocator.free(self.base.outputs);
        
        allocator.free(self.base.id);
        allocator.free(self.base.name);
        allocator.free(self.base.node_type);
        allocator.free(self.base.description);
        allocator.free(self.connection_string);
        allocator.free(self.query);
        allocator.destroy(self);
    }
    
    pub fn execute(self: *PostgresQueryNode, ctx: *ExecutionContext) !std.json.Value {
        // Create PostgreSQL client
        var client = PostgresClient.init(
            self.allocator,
            "localhost", // Parse from connection_string in production
            5432,
            "nworkflow",
            "postgres",
            null,
            5000,
        );
        defer client.deinit();
        
        // Connect to database
        try client.connect();
        defer client.disconnect();
        
        // Set RLS context if needed
        if (self.use_rls) {
            if (ctx.user_id) |uid| {
                try client.setRLSContext(uid);
            }
        }
        
        // Execute query
        var pg_result = try client.query(self.query, &.{});
        defer pg_result.deinit(self.allocator);
        
        // Convert to JSON result
        var result = std.json.ObjectMap.init(self.allocator);
        var rows_array = std.json.Array.init(self.allocator);

        for (pg_result.rows) |row| {
            var row_obj = std.json.ObjectMap.init(self.allocator);

            for (row.columns, row.values) |col, val| {
                const json_val = switch (val) {
                    .null_value => std.json.Value{ .null = {} },
                    .bool_value => |b| std.json.Value{ .bool = b },
                    .int_value => |i| std.json.Value{ .integer = i },
                    .float_value => |f| std.json.Value{ .float = f },
                    .text_value => |s| std.json.Value{ .string = try self.allocator.dupe(u8, s) },
                    .bytes_value => |b| std.json.Value{ .string = try self.allocator.dupe(u8, b) },
                };
                try row_obj.put(try self.allocator.dupe(u8, col), json_val);
            }

            try rows_array.append(.{ .object = row_obj });
        }

        try result.put("rows", .{ .array = rows_array });
        try result.put("count", .{ .integer = @as(i64, @intCast(pg_result.row_count)) });
        
        return .{ .object = result };
    }
};

/// PostgreSQL Insert Node - Insert records
pub const PostgresInsertNode = struct {
    allocator: Allocator,
    base: NodeInterface,
    connection_string: []const u8,
    table: []const u8,
    returning: bool,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: std.json.Value) !*PostgresInsertNode {
        const node = try allocator.create(PostgresInsertNode);
        errdefer allocator.destroy(node);
        
        const conn_str = if (config == .object) blk: {
            if (config.object.get("connection_string")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "postgres://localhost:5432/nworkflow";
        } else "postgres://localhost:5432/nworkflow";
        
        const table = if (config == .object) blk: {
            if (config.object.get("table")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "data";
        } else "data";
        
        const returning = if (config == .object) blk: {
            if (config.object.get("returning")) |v| {
                if (v == .bool) break :blk v.bool;
            }
            break :blk true;
        } else true;
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = .{
            .id = try allocator.dupe(u8, "record"),
            .name = try allocator.dupe(u8, "Record Data"),
            .description = try allocator.dupe(u8, "Record to insert as object"),
            .port_type = .object,
            .required = true,
            .default_value = null,
        };
        
        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = .{
            .id = try allocator.dupe(u8, "inserted"),
            .name = try allocator.dupe(u8, "Inserted Record"),
            .description = try allocator.dupe(u8, "The inserted record with generated fields"),
            .port_type = .object,
            .required = true,
            .default_value = null,
        };
        
        node.* = .{
            .allocator = allocator,
            .base = .{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .node_type = try allocator.dupe(u8, "postgres_insert"),
                .description = try allocator.dupe(u8, "Insert data into PostgreSQL table"),
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = config,
                
            },
            .connection_string = try allocator.dupe(u8, conn_str),
            .table = try allocator.dupe(u8, table),
            .returning = returning,
        };
        
        return node;
    }
    
    pub fn deinit(self: *PostgresInsertNode) void {
        const allocator = self.allocator;
        
        for (self.base.inputs) |input| {
            allocator.free(input.id);
            allocator.free(input.name);
            allocator.free(input.description);
        }
        allocator.free(self.base.inputs);
        
        for (self.base.outputs) |output| {
            allocator.free(output.id);
            allocator.free(output.name);
            allocator.free(output.description);
        }
        allocator.free(self.base.outputs);
        
        allocator.free(self.base.id);
        allocator.free(self.base.name);
        allocator.free(self.base.node_type);
        allocator.free(self.base.description);
        allocator.free(self.connection_string);
        allocator.free(self.table);
        allocator.destroy(self);
    }
    
    pub fn execute(self: *PostgresInsertNode, ctx: *ExecutionContext) !std.json.Value {
        // Get record data from input
        const record_input = ctx.getInput("record") orelse return error.MissingRecord;
        
        // Create PostgreSQL client
        var client = PostgresClient.init(
            self.allocator,
            "localhost",
            5432,
            "nworkflow",
            "postgres",
            null,
            5000,
        );
        defer client.deinit();
        
        try client.connect();
        defer client.disconnect();
        
        // Build INSERT statement from record object
        var sql_buf = std.ArrayList(u8){};
        defer sql_buf.deinit(self.allocator);

        const writer = sql_buf.writer(self.allocator);
        try writer.print("INSERT INTO {s} ", .{self.table});

        // Extract columns and values from record
        if (record_input == .object) {
            var columns = std.ArrayList([]const u8){};
            defer columns.deinit(self.allocator);
            var values = std.ArrayList([]const u8){};
            defer values.deinit(self.allocator);

            var it = record_input.object.iterator();
            while (it.next()) |entry| {
                try columns.append(self.allocator, entry.key_ptr.*);
                
                const val_str = switch (entry.value_ptr.*) {
                    .string => |s| try std.fmt.allocPrint(self.allocator, "'{s}'", .{s}),
                    .integer => |i| try std.fmt.allocPrint(self.allocator, "{d}", .{i}),
                    .float => |f| try std.fmt.allocPrint(self.allocator, "{d}", .{f}),
                    .bool => |b| try std.fmt.allocPrint(self.allocator, "{}", .{b}),
                    else => try std.fmt.allocPrint(self.allocator, "NULL", .{}),
                };
                defer self.allocator.free(val_str);
                try values.append(self.allocator, try self.allocator.dupe(u8, val_str));
            }
            
            // Build column list
            try writer.writeAll("(");
            for (columns.items, 0..) |col, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s}", .{col});
            }
            try writer.writeAll(") VALUES (");
            
            // Build values list
            for (values.items, 0..) |val, i| {
                defer self.allocator.free(val);
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(val);
            }
            try writer.writeAll(")");
        }
        
        // Add RETURNING clause if requested
        if (self.returning) {
            try writer.writeAll(" RETURNING *");
        }
        
        // Execute insert
        var pg_result = try client.query(sql_buf.items, &.{});
        defer pg_result.deinit(self.allocator);
        
        // Return first row if RETURNING was used
        var result = std.json.ObjectMap.init(self.allocator);
        
        if (pg_result.rows.len > 0) {
            const row = pg_result.rows[0];
            for (row.columns, row.values) |col, val| {
                const json_val = switch (val) {
                    .null_value => std.json.Value{ .null = {} },
                    .bool_value => |b| std.json.Value{ .bool = b },
                    .int_value => |i| std.json.Value{ .integer = i },
                    .float_value => |f| std.json.Value{ .float = f },
                    .text_value => |s| std.json.Value{ .string = try self.allocator.dupe(u8, s) },
                    .bytes_value => |b| std.json.Value{ .string = try self.allocator.dupe(u8, b) },
                };
                try result.put(try self.allocator.dupe(u8, col), json_val);
            }
        }
        
        return .{ .object = result };
    }
};

/// PostgreSQL Update Node - Update records
pub const PostgresUpdateNode = struct {
    allocator: Allocator,
    base: NodeInterface,
    connection_string: []const u8,
    table: []const u8,
    returning: bool,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: std.json.Value) !*PostgresUpdateNode {
        const node = try allocator.create(PostgresUpdateNode);
        errdefer allocator.destroy(node);
        
        const conn_str = if (config == .object) blk: {
            if (config.object.get("connection_string")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "postgres://localhost:5432/nworkflow";
        } else "postgres://localhost:5432/nworkflow";
        
        const table = if (config == .object) blk: {
            if (config.object.get("table")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "data";
        } else "data";
        
        const returning = if (config == .object) blk: {
            if (config.object.get("returning")) |v| {
                if (v == .bool) break :blk v.bool;
            }
            break :blk true;
        } else true;
        
        const inputs = try allocator.alloc(Port, 2);
        inputs[0] = .{
            .id = try allocator.dupe(u8, "where"),
            .name = try allocator.dupe(u8, "WHERE Clause"),
            .description = try allocator.dupe(u8, "Conditions for update"),
            .port_type = .object,
            .required = true,
            .default_value = null,
        };
        inputs[1] = .{
            .id = try allocator.dupe(u8, "set"),
            .name = try allocator.dupe(u8, "SET Values"),
            .description = try allocator.dupe(u8, "Values to update"),
            .port_type = .object,
            .required = true,
            .default_value = null,
        };
        
        const outputs = try allocator.alloc(Port, 2);
        outputs[0] = .{
            .id = try allocator.dupe(u8, "updated_count"),
            .name = try allocator.dupe(u8, "Updated Count"),
            .description = try allocator.dupe(u8, "Number of rows updated"),
            .port_type = .number,
            .required = true,
            .default_value = null,
        };
        outputs[1] = .{
            .id = try allocator.dupe(u8, "updated_rows"),
            .name = try allocator.dupe(u8, "Updated Rows"),
            .description = try allocator.dupe(u8, "Updated rows if RETURNING used"),
            .port_type = .array,
            .required = false,
            .default_value = null,
        };
        
        node.* = .{
            .allocator = allocator,
            .base = .{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .node_type = try allocator.dupe(u8, "postgres_update"),
                .description = try allocator.dupe(u8, "Update PostgreSQL table rows"),
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = config,
                
            },
            .connection_string = try allocator.dupe(u8, conn_str),
            .table = try allocator.dupe(u8, table),
            .returning = returning,
        };
        
        return node;
    }
    
    pub fn deinit(self: *PostgresUpdateNode) void {
        const allocator = self.allocator;
        
        for (self.base.inputs) |input| {
            allocator.free(input.id);
            allocator.free(input.name);
            allocator.free(input.description);
        }
        allocator.free(self.base.inputs);
        
        for (self.base.outputs) |output| {
            allocator.free(output.id);
            allocator.free(output.name);
            allocator.free(output.description);
        }
        allocator.free(self.base.outputs);
        
        allocator.free(self.base.id);
        allocator.free(self.base.name);
        allocator.free(self.base.node_type);
        allocator.free(self.base.description);
        allocator.free(self.connection_string);
        allocator.free(self.table);
        allocator.destroy(self);
    }
    
    pub fn execute(self: *PostgresUpdateNode, ctx: *ExecutionContext) !std.json.Value {
        // Get WHERE and SET clauses from inputs
        const where_input = ctx.getInput("where") orelse return error.MissingWhere;
        const set_input = ctx.getInput("set") orelse return error.MissingSet;
        
        // Create PostgreSQL client
        var client = PostgresClient.init(
            self.allocator,
            "localhost",
            5432,
            "nworkflow",
            "postgres",
            null,
            5000,
        );
        defer client.deinit();
        
        try client.connect();
        defer client.disconnect();
        
        // Build UPDATE statement
        var sql_buf = std.ArrayList(u8){};
        defer sql_buf.deinit(self.allocator);

        const writer = sql_buf.writer(self.allocator);
        try writer.print("UPDATE {s} SET ", .{self.table});
        
        // Build SET clause
        if (set_input == .object) {
            var it = set_input.object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try writer.writeAll(", ");
                first = false;
                
                const val_str = switch (entry.value_ptr.*) {
                    .string => |s| try std.fmt.allocPrint(self.allocator, "'{s}'", .{s}),
                    .integer => |i| try std.fmt.allocPrint(self.allocator, "{d}", .{i}),
                    .float => |f| try std.fmt.allocPrint(self.allocator, "{d}", .{f}),
                    .bool => |b| try std.fmt.allocPrint(self.allocator, "{}", .{b}),
                    else => try std.fmt.allocPrint(self.allocator, "NULL", .{}),
                };
                defer self.allocator.free(val_str);
                
                try writer.print("{s} = {s}", .{ entry.key_ptr.*, val_str });
            }
        }
        
        // Build WHERE clause
        try writer.writeAll(" WHERE ");
        if (where_input == .object) {
            var it = where_input.object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try writer.writeAll(" AND ");
                first = false;
                
                const val_str = switch (entry.value_ptr.*) {
                    .string => |s| try std.fmt.allocPrint(self.allocator, "'{s}'", .{s}),
                    .integer => |i| try std.fmt.allocPrint(self.allocator, "{d}", .{i}),
                    .float => |f| try std.fmt.allocPrint(self.allocator, "{d}", .{f}),
                    .bool => |b| try std.fmt.allocPrint(self.allocator, "{}", .{b}),
                    else => try std.fmt.allocPrint(self.allocator, "NULL", .{}),
                };
                defer self.allocator.free(val_str);
                
                try writer.print("{s} = {s}", .{ entry.key_ptr.*, val_str });
            }
        }
        
        if (self.returning) {
            try writer.writeAll(" RETURNING *");
        }
        
        // Execute update
        const affected_rows = try client.execute(sql_buf.items);
        
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("updated_count", .{ .integer = @as(i64, @intCast(affected_rows)) });
        
        return .{ .object = result };
    }
};

/// PostgreSQL Delete Node - Delete records
pub const PostgresDeleteNode = struct {
    allocator: Allocator,
    base: NodeInterface,
    connection_string: []const u8,
    table: []const u8,
    returning: bool,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: std.json.Value) !*PostgresDeleteNode {
        const node = try allocator.create(PostgresDeleteNode);
        errdefer allocator.destroy(node);
        
        const conn_str = if (config == .object) blk: {
            if (config.object.get("connection_string")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "postgres://localhost:5432/nworkflow";
        } else "postgres://localhost:5432/nworkflow";
        
        const table = if (config == .object) blk: {
            if (config.object.get("table")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "data";
        } else "data";
        
        const returning = if (config == .object) blk: {
            if (config.object.get("returning")) |v| {
                if (v == .bool) break :blk v.bool;
            }
            break :blk false;
        } else false;
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = .{
            .id = try allocator.dupe(u8, "where"),
            .name = try allocator.dupe(u8, "WHERE Clause"),
            .description = try allocator.dupe(u8, "Conditions for deletion"),
            .port_type = .object,
            .required = true,
            .default_value = null,
        };
        
        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = .{
            .id = try allocator.dupe(u8, "deleted_count"),
            .name = try allocator.dupe(u8, "Deleted Count"),
            .description = try allocator.dupe(u8, "Number of rows deleted"),
            .port_type = .number,
            .required = true,
            .default_value = null,
        };
        
        node.* = .{
            .allocator = allocator,
            .base = .{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .node_type = try allocator.dupe(u8, "postgres_delete"),
                .description = try allocator.dupe(u8, "Delete rows from PostgreSQL table"),
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = config,
                
            },
            .connection_string = try allocator.dupe(u8, conn_str),
            .table = try allocator.dupe(u8, table),
            .returning = returning,
        };
        
        return node;
    }
    
    pub fn deinit(self: *PostgresDeleteNode) void {
        const allocator = self.allocator;
        
        for (self.base.inputs) |input| {
            allocator.free(input.id);
            allocator.free(input.name);
            allocator.free(input.description);
        }
        allocator.free(self.base.inputs);
        
        for (self.base.outputs) |output| {
            allocator.free(output.id);
            allocator.free(output.name);
            allocator.free(output.description);
        }
        allocator.free(self.base.outputs);
        
        allocator.free(self.base.id);
        allocator.free(self.base.name);
        allocator.free(self.base.node_type);
        allocator.free(self.base.description);
        allocator.free(self.connection_string);
        allocator.free(self.table);
        allocator.destroy(self);
    }
    
    pub fn execute(self: *PostgresDeleteNode, ctx: *ExecutionContext) !std.json.Value {
        // Get WHERE clause from input
        const where_input = ctx.getInput("where") orelse return error.MissingWhere;
        
        // Create PostgreSQL client
        var client = PostgresClient.init(
            self.allocator,
            "localhost",
            5432,
            "nworkflow",
            "postgres",
            null,
            5000,
        );
        defer client.deinit();
        
        try client.connect();
        defer client.disconnect();
        
        // Build DELETE statement
        var sql_buf = std.ArrayList(u8){};
        defer sql_buf.deinit(self.allocator);

        const writer = sql_buf.writer(self.allocator);
        try writer.print("DELETE FROM {s} WHERE ", .{self.table});
        
        // Build WHERE clause
        if (where_input == .object) {
            var it = where_input.object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try writer.writeAll(" AND ");
                first = false;
                
                const val_str = switch (entry.value_ptr.*) {
                    .string => |s| try std.fmt.allocPrint(self.allocator, "'{s}'", .{s}),
                    .integer => |i| try std.fmt.allocPrint(self.allocator, "{d}", .{i}),
                    .float => |f| try std.fmt.allocPrint(self.allocator, "{d}", .{f}),
                    .bool => |b| try std.fmt.allocPrint(self.allocator, "{}", .{b}),
                    else => try std.fmt.allocPrint(self.allocator, "NULL", .{}),
                };
                defer self.allocator.free(val_str);
                
                try writer.print("{s} = {s}", .{ entry.key_ptr.*, val_str });
            }
        }
        
        // Execute delete
        const affected_rows = try client.execute(sql_buf.items);
        
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("deleted_count", .{ .integer = @as(i64, @intCast(affected_rows)) });
        
        return .{ .object = result };
    }
};

/// PostgreSQL Transaction Node - Manage transactions
pub const PostgresTransactionNode = struct {
    allocator: Allocator,
    base: NodeInterface,
    connection_string: []const u8,
    action: TransactionAction,
    
    pub const TransactionAction = enum {
        begin,
        commit,
        rollback,
        savepoint,
    };
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: std.json.Value) !*PostgresTransactionNode {
        const node = try allocator.create(PostgresTransactionNode);
        errdefer allocator.destroy(node);
        
        const conn_str = if (config == .object) blk: {
            if (config.object.get("connection_string")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "postgres://localhost:5432/nworkflow";
        } else "postgres://localhost:5432/nworkflow";
        
        const action = if (config == .object) blk: {
            if (config.object.get("action")) |v| {
                if (v == .string) {
                    if (std.mem.eql(u8, v.string, "commit")) break :blk TransactionAction.commit;
                    if (std.mem.eql(u8, v.string, "rollback")) break :blk TransactionAction.rollback;
                    if (std.mem.eql(u8, v.string, "savepoint")) break :blk TransactionAction.savepoint;
                }
            }
            break :blk TransactionAction.begin;
        } else TransactionAction.begin;
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = .{
            .id = try allocator.dupe(u8, "trigger"),
            .name = try allocator.dupe(u8, "Trigger"),
            .description = try allocator.dupe(u8, "Trigger the transaction action"),
            .port_type = .any,
            .required = false,
            .default_value = null,
        };
        
        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = .{
            .id = try allocator.dupe(u8, "success"),
            .name = try allocator.dupe(u8, "Success"),
            .description = try allocator.dupe(u8, "Whether action succeeded"),
            .port_type = .boolean,
            .required = true,
            .default_value = null,
        };
        
        node.* = .{
            .allocator = allocator,
            .base = .{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .node_type = try allocator.dupe(u8, "postgres_transaction"),
                .description = try allocator.dupe(u8, "Execute PostgreSQL transaction"),
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = config,
                
            },
            .connection_string = try allocator.dupe(u8, conn_str),
            .action = action,
        };
        
        return node;
    }
    
    pub fn deinit(self: *PostgresTransactionNode) void {
        const allocator = self.allocator;
        
        for (self.base.inputs) |input| {
            allocator.free(input.id);
            allocator.free(input.name);
            allocator.free(input.description);
        }
        allocator.free(self.base.inputs);
        
        for (self.base.outputs) |output| {
            allocator.free(output.id);
            allocator.free(output.name);
            allocator.free(output.description);
        }
        allocator.free(self.base.outputs);
        
        allocator.free(self.base.id);
        allocator.free(self.base.name);
        allocator.free(self.base.node_type);
        allocator.free(self.base.description);
        allocator.free(self.connection_string);
        allocator.destroy(self);
    }
    
    pub fn execute(self: *PostgresTransactionNode, ctx: *ExecutionContext) !std.json.Value {
        _ = ctx;
        
        // Create PostgreSQL client
        var client = PostgresClient.init(
            self.allocator,
            "localhost",
            5432,
            "nworkflow",
            "postgres",
            null,
            5000,
        );
        defer client.deinit();
        
        try client.connect();
        defer client.disconnect();
        
        // Execute transaction action
        switch (self.action) {
            .begin => try client.begin(),
            .commit => try client.commit(),
            .rollback => try client.rollback(),
            .savepoint => {
                // Execute SAVEPOINT command
                _ = try client.execute("SAVEPOINT sp1");
            },
        }
        
        return .{ .bool = true };
    }
};

/// PostgreSQL Bulk Insert Node - Batch inserts
pub const PostgresBulkInsertNode = struct {
    allocator: Allocator,
    base: NodeInterface,
    connection_string: []const u8,
    table: []const u8,
    batch_size: usize,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: std.json.Value) !*PostgresBulkInsertNode {
        const node = try allocator.create(PostgresBulkInsertNode);
        errdefer allocator.destroy(node);
        
        const conn_str = if (config == .object) blk: {
            if (config.object.get("connection_string")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "postgres://localhost:5432/nworkflow";
        } else "postgres://localhost:5432/nworkflow";
        
        const table = if (config == .object) blk: {
            if (config.object.get("table")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "data";
        } else "data";
        
        const batch_size = if (config == .object) blk: {
            if (config.object.get("batch_size")) |v| {
                if (v == .integer) break :blk @as(usize, @intCast(v.integer));
            }
            break :blk @as(usize, 1000);
        } else @as(usize, 1000);
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = .{
            .id = try allocator.dupe(u8, "records"),
            .name = try allocator.dupe(u8, "Records Array"),
            .description = try allocator.dupe(u8, "Array of records to insert"),
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        const outputs = try allocator.alloc(Port, 2);
        outputs[0] = .{
            .id = try allocator.dupe(u8, "inserted_count"),
            .name = try allocator.dupe(u8, "Inserted Count"),
            .description = try allocator.dupe(u8, "Total records inserted"),
            .port_type = .number,
            .required = true,
            .default_value = null,
        };
        outputs[1] = .{
            .id = try allocator.dupe(u8, "batches"),
            .name = try allocator.dupe(u8, "Batch Count"),
            .description = try allocator.dupe(u8, "Number of batches processed"),
            .port_type = .number,
            .required = true,
            .default_value = null,
        };
        
        node.* = .{
            .allocator = allocator,
            .base = .{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .node_type = try allocator.dupe(u8, "postgres_bulk_insert"),
                .description = try allocator.dupe(u8, "Bulk insert records into PostgreSQL"),
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = config,
                
            },
            .connection_string = try allocator.dupe(u8, conn_str),
            .table = try allocator.dupe(u8, table),
            .batch_size = batch_size,
        };
        
        return node;
    }
    
    pub fn deinit(self: *PostgresBulkInsertNode) void {
        const allocator = self.allocator;
        
        for (self.base.inputs) |input| {
            allocator.free(input.id);
            allocator.free(input.name);
            allocator.free(input.description);
        }
        allocator.free(self.base.inputs);
        
        for (self.base.outputs) |output| {
            allocator.free(output.id);
            allocator.free(output.name);
            allocator.free(output.description);
        }
        allocator.free(self.base.outputs);
        
        allocator.free(self.base.id);
        allocator.free(self.base.name);
        allocator.free(self.base.node_type);
        allocator.free(self.base.description);
        allocator.free(self.connection_string);
        allocator.free(self.table);
        allocator.destroy(self);
    }
    
    pub fn execute(self: *PostgresBulkInsertNode, ctx: *ExecutionContext) !std.json.Value {
        // Get records array from input
        const records_input = ctx.getInput("records") orelse return error.MissingRecords;
        
        if (records_input != .array) {
            return error.InvalidRecordsFormat;
        }
        
        // Create PostgreSQL client
        var client = PostgresClient.init(
            self.allocator,
            "localhost",
            5432,
            "nworkflow",
            "postgres",
            null,
            5000,
        );
        defer client.deinit();
        
        try client.connect();
        defer client.disconnect();
        
        try client.begin();
        errdefer client.rollback() catch {};
        
        var total_inserted: usize = 0;
        var batch_count: usize = 0;
        var current_batch: usize = 0;
        
        // Process records in batches
        for (records_input.array.items) |record| {
            if (record != .object) continue;
            
            // Build INSERT statement for this record
            var sql_buf = std.ArrayList(u8){};
            defer sql_buf.deinit(self.allocator);

            const writer = sql_buf.writer(self.allocator);
            try writer.print("INSERT INTO {s} (", .{self.table});

            // Get columns and values
            var columns = std.ArrayList([]const u8){};
            defer columns.deinit(self.allocator);
            var values = std.ArrayList([]const u8){};
            defer values.deinit(self.allocator);

            var it = record.object.iterator();
            while (it.next()) |entry| {
                try columns.append(self.allocator, entry.key_ptr.*);

                const val_str = switch (entry.value_ptr.*) {
                    .string => |s| try std.fmt.allocPrint(self.allocator, "'{s}'", .{s}),
                    .integer => |i| try std.fmt.allocPrint(self.allocator, "{d}", .{i}),
                    .float => |f| try std.fmt.allocPrint(self.allocator, "{d}", .{f}),
                    .bool => |b| try std.fmt.allocPrint(self.allocator, "{}", .{b}),
                    else => try std.fmt.allocPrint(self.allocator, "NULL", .{}),
                };
                defer self.allocator.free(val_str);
                try values.append(self.allocator, try self.allocator.dupe(u8, val_str));
            }
            
            // Write columns
            for (columns.items, 0..) |col, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(col);
            }
            try writer.writeAll(") VALUES (");
            
            // Write values
            for (values.items, 0..) |val, i| {
                defer self.allocator.free(val);
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(val);
            }
            try writer.writeAll(")");
            
            // Execute insert
            _ = try client.execute(sql_buf.items);
            total_inserted += 1;
            current_batch += 1;
            
            // Commit batch if size reached
            if (current_batch >= self.batch_size) {
                try client.commit();
                batch_count += 1;
                current_batch = 0;
                try client.begin();
            }
        }
        
        // Commit remaining records
        if (current_batch > 0) {
            try client.commit();
            batch_count += 1;
        }
        
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("inserted_count", .{ .integer = @as(i64, @intCast(total_inserted)) });
        try result.put("batches", .{ .integer = @as(i64, @intCast(batch_count)) });
        
        return .{ .object = result };
    }
};

/// PostgreSQL RLS Query Node - Row-level security queries with user context
pub const PostgresRLSQueryNode = struct {
    allocator: Allocator,
    base: NodeInterface,
    connection_string: []const u8,
    table: []const u8,
    query: []const u8,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: std.json.Value) !*PostgresRLSQueryNode {
        const node = try allocator.create(PostgresRLSQueryNode);
        errdefer allocator.destroy(node);
        
        const conn_str = if (config == .object) blk: {
            if (config.object.get("connection_string")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "postgres://localhost:5432/nworkflow";
        } else "postgres://localhost:5432/nworkflow";
        
        const table = if (config == .object) blk: {
            if (config.object.get("table")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "data";
        } else "data";
        
        const query = if (config == .object) blk: {
            if (config.object.get("query")) |v| {
                if (v == .string) break :blk v.string;
            }
            break :blk "SELECT * FROM data";
        } else "SELECT * FROM data";
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = .{
            .id = try allocator.dupe(u8, "parameters"),
            .name = try allocator.dupe(u8, "Query Parameters"),
            .description = try allocator.dupe(u8, "Parameters for prepared statement"),
            .port_type = .object,
            .required = false,
            .default_value = null,
        };
        
        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = .{
            .id = try allocator.dupe(u8, "rows"),
            .name = try allocator.dupe(u8, "Result Rows"),
            .description = try allocator.dupe(u8, "Filtered rows based on RLS policies"),
            .port_type = .array,
            .required = true,
            .default_value = null,
        };
        
        node.* = .{
            .allocator = allocator,
            .base = .{
                .id = try allocator.dupe(u8, id),
                .name = try allocator.dupe(u8, name),
                .node_type = try allocator.dupe(u8, "postgres_rls_query"),
                .description = try allocator.dupe(u8, "Query with PostgreSQL RLS"),
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = config,
                
            },
            .connection_string = try allocator.dupe(u8, conn_str),
            .table = try allocator.dupe(u8, table),
            .query = try allocator.dupe(u8, query),
        };
        
        return node;
    }
    
    pub fn deinit(self: *PostgresRLSQueryNode) void {
        const allocator = self.allocator;
        
        for (self.base.inputs) |input| {
            allocator.free(input.id);
            allocator.free(input.name);
            allocator.free(input.description);
        }
        allocator.free(self.base.inputs);
        
        for (self.base.outputs) |output| {
            allocator.free(output.id);
            allocator.free(output.name);
            allocator.free(output.description);
        }
        allocator.free(self.base.outputs);
        
        allocator.free(self.base.id);
        allocator.free(self.base.name);
        allocator.free(self.base.node_type);
        allocator.free(self.base.description);
        allocator.free(self.connection_string);
        allocator.free(self.table);
        allocator.free(self.query);
        allocator.destroy(self);
    }
    
    pub fn execute(self: *PostgresRLSQueryNode, ctx: *ExecutionContext) !std.json.Value {
        // Get user ID from Keycloak context
        const user_id = ctx.user_id orelse return error.Unauthorized;
        
        // Create PostgreSQL client
        var client = PostgresClient.init(
            self.allocator,
            "localhost",
            5432,
            "nworkflow",
            "postgres",
            null,
            5000,
        );
        defer client.deinit();
        
        try client.connect();
        defer client.disconnect();
        
        // Set RLS context with user ID
        try client.setRLSContext(user_id);
        
        // Execute query with RLS policies applied
        var pg_result = try client.query(self.query, &.{});
        defer pg_result.deinit(self.allocator);
        
        // Convert to JSON result
        var result = std.json.ObjectMap.init(self.allocator);
        var rows_array = std.json.Array.init(self.allocator);

        for (pg_result.rows) |row| {
            var row_obj = std.json.ObjectMap.init(self.allocator);

            for (row.columns, row.values) |col, val| {
                const json_val = switch (val) {
                    .null_value => std.json.Value{ .null = {} },
                    .bool_value => |b| std.json.Value{ .bool = b },
                    .int_value => |i| std.json.Value{ .integer = i },
                    .float_value => |f| std.json.Value{ .float = f },
                    .text_value => |s| std.json.Value{ .string = try self.allocator.dupe(u8, s) },
                    .bytes_value => |b| std.json.Value{ .string = try self.allocator.dupe(u8, b) },
                };
                try row_obj.put(try self.allocator.dupe(u8, col), json_val);
            }

            try rows_array.append(.{ .object = row_obj });
        }

        try result.put("rows", .{ .array = rows_array });
        
        return .{ .object = result };
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "PostgresQueryNode creation and execution" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("connection_string", .{ .string = "postgres://localhost:5432/test" });
    try config.put("query", .{ .string = "SELECT * FROM users" });
    try config.put("use_rls", .{ .bool = false });
    
    const node = try PostgresQueryNode.init(allocator, "query-1", "Test Query", .{ .object = config });
    defer node.deinit();
    
    try std.testing.expectEqualStrings("query-1", node.base.id);
    try std.testing.expectEqualStrings("postgres_query", node.base.node_type);
    try std.testing.expectEqualStrings("SELECT * FROM users", node.query);
    try std.testing.expect(!node.use_rls);
}

test "PostgresInsertNode creation" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("table", .{ .string = "customers" });
    try config.put("returning", .{ .bool = true });
    
    const node = try PostgresInsertNode.init(allocator, "insert-1", "Insert Customer", .{ .object = config });
    defer node.deinit();
    
    try std.testing.expectEqualStrings("insert-1", node.base.id);
    try std.testing.expectEqualStrings("postgres_insert", node.base.node_type);
    try std.testing.expectEqualStrings("customers", node.table);
    try std.testing.expect(node.returning);
}

test "PostgresUpdateNode creation" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("table", .{ .string = "orders" });
    
    const node = try PostgresUpdateNode.init(allocator, "update-1", "Update Order", .{ .object = config });
    defer node.deinit();
    
    try std.testing.expectEqualStrings("update-1", node.base.id);
    try std.testing.expectEqualStrings("postgres_update", node.base.node_type);
    try std.testing.expectEqualStrings("orders", node.table);
}

test "PostgresDeleteNode creation" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("table", .{ .string = "logs" });
    
    const node = try PostgresDeleteNode.init(allocator, "delete-1", "Delete Logs", .{ .object = config });
    defer node.deinit();
    
    try std.testing.expectEqualStrings("delete-1", node.base.id);
    try std.testing.expectEqualStrings("postgres_delete", node.base.node_type);
    try std.testing.expectEqualStrings("logs", node.table);
}

test "PostgresTransactionNode creation with different actions" {
    const allocator = std.testing.allocator;
    
    // Test BEGIN
    {
        var config = std.json.ObjectMap.init(allocator);
        defer config.deinit();
        
        try config.put("action", .{ .string = "begin" });
        
        const node = try PostgresTransactionNode.init(allocator, "tx-1", "Begin TX", .{ .object = config });
        defer node.deinit();
        
        try std.testing.expectEqual(PostgresTransactionNode.TransactionAction.begin, node.action);
    }
    
    // Test COMMIT
    {
        var config = std.json.ObjectMap.init(allocator);
        defer config.deinit();
        
        try config.put("action", .{ .string = "commit" });
        
        const node = try PostgresTransactionNode.init(allocator, "tx-2", "Commit TX", .{ .object = config });
        defer node.deinit();
        
        try std.testing.expectEqual(PostgresTransactionNode.TransactionAction.commit, node.action);
    }
    
    // Test ROLLBACK
    {
        var config = std.json.ObjectMap.init(allocator);
        defer config.deinit();
        
        try config.put("action", .{ .string = "rollback" });
        
        const node = try PostgresTransactionNode.init(allocator, "tx-3", "Rollback TX", .{ .object = config });
        defer node.deinit();
        
        try std.testing.expectEqual(PostgresTransactionNode.TransactionAction.rollback, node.action);
    }
}

test "PostgresBulkInsertNode creation" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("table", .{ .string = "events" });
    try config.put("batch_size", .{ .integer = 500 });
    
    const node = try PostgresBulkInsertNode.init(allocator, "bulk-1", "Bulk Insert", .{ .object = config });
    defer node.deinit();
    
    try std.testing.expectEqualStrings("bulk-1", node.base.id);
    try std.testing.expectEqualStrings("postgres_bulk_insert", node.base.node_type);
    try std.testing.expectEqualStrings("events", node.table);
    try std.testing.expectEqual(@as(usize, 500), node.batch_size);
}

test "PostgresRLSQueryNode creation" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("table", .{ .string = "documents" });
    try config.put("query", .{ .string = "SELECT * FROM documents WHERE org_id = current_setting('app.org_id')" });
    
    const node = try PostgresRLSQueryNode.init(allocator, "rls-1", "RLS Query", .{ .object = config });
    defer node.deinit();
    
    try std.testing.expectEqualStrings("rls-1", node.base.id);
    try std.testing.expectEqualStrings("postgres_rls_query", node.base.node_type);
    try std.testing.expectEqualStrings("documents", node.table);
}

test "PostgresQueryNode with default config" {
    const allocator = std.testing.allocator;
    
    const config = std.json.Value{ .null = {} };
    
    const node = try PostgresQueryNode.init(allocator, "query-2", "Default Query", config);
    defer node.deinit();
    
    try std.testing.expectEqualStrings("postgres://localhost:5432/nworkflow", node.connection_string);
    try std.testing.expectEqualStrings("SELECT 1", node.query);
    try std.testing.expect(!node.use_rls);
}

test "PostgresInsertNode with default config" {
    const allocator = std.testing.allocator;
    
    const config = std.json.Value{ .null = {} };
    
    const node = try PostgresInsertNode.init(allocator, "insert-2", "Default Insert", config);
    defer node.deinit();
    
    try std.testing.expectEqualStrings("postgres://localhost:5432/nworkflow", node.connection_string);
    try std.testing.expectEqualStrings("data", node.table);
    try std.testing.expect(node.returning);
}

test "PostgresBulkInsertNode with default batch size" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("table", .{ .string = "logs" });
    
    const node = try PostgresBulkInsertNode.init(allocator, "bulk-2", "Default Batch", .{ .object = config });
    defer node.deinit();
    
    try std.testing.expectEqual(@as(usize, 1000), node.batch_size);
}

test "PostgresRLSQueryNode requires user context" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("query", .{ .string = "SELECT * FROM sensitive_data" });
    
    const node = try PostgresRLSQueryNode.init(allocator, "rls-2", "Secure Query", .{ .object = config });
    defer node.deinit();
    
    // Create context without user_id
    var variables = std.StringHashMap([]const u8).init(allocator);
    defer variables.deinit();
    
    var services = std.StringHashMap(*node_types.Service).init(allocator);
    defer services.deinit();
    
    var ctx = ExecutionContext{
        .allocator = allocator,
        .workflow_id = "wf-1",
        .execution_id = "exec-1",
        .user_id = null, // No user ID
        .variables = variables,
        .services = services,
        .inputs = std.StringHashMap(std.json.Value).init(allocator),
    };
    defer ctx.inputs.deinit();
    
    // Should return Unauthorized error
    const result = node.execute(&ctx);
    try std.testing.expectError(error.Unauthorized, result);
}

test "PostgresRLSQueryNode with user context" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("query", .{ .string = "SELECT * FROM user_data" });
    
    const node = try PostgresRLSQueryNode.init(allocator, "rls-3", "User Query", .{ .object = config });
    defer node.deinit();
    
    // Create context with user_id
    var variables = std.StringHashMap([]const u8).init(allocator);
    defer variables.deinit();
    
    var services = std.StringHashMap(*node_types.Service).init(allocator);
    defer services.deinit();
    
    var ctx = ExecutionContext{
        .allocator = allocator,
        .workflow_id = "wf-1",
        .execution_id = "exec-1",
        .user_id = "user-123",
        .variables = variables,
        .services = services,
        .inputs = std.StringHashMap(std.json.Value).init(allocator),
    };
    defer ctx.inputs.deinit();
    
    // Should succeed with user context
    const result = try node.execute(&ctx);
    try std.testing.expect(result == .object);
}

test "All PostgreSQL nodes have correct input/output ports" {
    const allocator = std.testing.allocator;
    const config = std.json.Value{ .null = {} };
    
    // Test PostgresQueryNode ports
    {
        const node = try PostgresQueryNode.init(allocator, "q1", "Query", config);
        defer node.deinit();
        
        try std.testing.expectEqual(@as(usize, 1), node.base.inputs.len);
        try std.testing.expectEqual(@as(usize, 2), node.base.outputs.len);
        try std.testing.expectEqualStrings("parameters", node.base.inputs[0].id);
        try std.testing.expectEqualStrings("rows", node.base.outputs[0].id);
        try std.testing.expectEqualStrings("count", node.base.outputs[1].id);
    }
    
    // Test PostgresInsertNode ports
    {
        const node = try PostgresInsertNode.init(allocator, "i1", "Insert", config);
        defer node.deinit();
        
        try std.testing.expectEqual(@as(usize, 1), node.base.inputs.len);
        try std.testing.expectEqual(@as(usize, 1), node.base.outputs.len);
        try std.testing.expectEqualStrings("record", node.base.inputs[0].id);
        try std.testing.expectEqualStrings("inserted", node.base.outputs[0].id);
    }
    
    // Test PostgresUpdateNode ports
    {
        const node = try PostgresUpdateNode.init(allocator, "u1", "Update", config);
        defer node.deinit();
        
        try std.testing.expectEqual(@as(usize, 2), node.base.inputs.len);
        try std.testing.expectEqual(@as(usize, 2), node.base.outputs.len);
        try std.testing.expectEqualStrings("where", node.base.inputs[0].id);
        try std.testing.expectEqualStrings("set", node.base.inputs[1].id);
    }
    
    // Test PostgresDeleteNode ports
    {
        const node = try PostgresDeleteNode.init(allocator, "d1", "Delete", config);
        defer node.deinit();
        
        try std.testing.expectEqual(@as(usize, 1), node.base.inputs.len);
        try std.testing.expectEqual(@as(usize, 1), node.base.outputs.len);
        try std.testing.expectEqualStrings("where", node.base.inputs[0].id);
        try std.testing.expectEqualStrings("deleted_count", node.base.outputs[0].id);
    }
}

test "PostgresTransactionNode execution" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("action", .{ .string = "begin" });
    
    const node = try PostgresTransactionNode.init(allocator, "tx-4", "Test TX", .{ .object = config });
    defer node.deinit();
    
    var variables = std.StringHashMap([]const u8).init(allocator);
    defer variables.deinit();
    
    var services = std.StringHashMap(*node_types.Service).init(allocator);
    defer services.deinit();
    
    var ctx = ExecutionContext{
        .allocator = allocator,
        .workflow_id = "wf-1",
        .execution_id = "exec-1",
        .user_id = null,
        .variables = variables,
        .services = services,
        .inputs = std.StringHashMap(std.json.Value).init(allocator),
    };
    defer ctx.inputs.deinit();
    
    const result = try node.execute(&ctx);
    try std.testing.expectEqual(std.json.Value{ .bool = true }, result);
}

test "PostgresBulkInsertNode execution mock" {
    const allocator = std.testing.allocator;
    
    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    
    try config.put("table", .{ .string = "events" });
    try config.put("batch_size", .{ .integer = 100 });
    
    const node = try PostgresBulkInsertNode.init(allocator, "bulk-3", "Bulk Test", .{ .object = config });
    defer node.deinit();
    
    var variables = std.StringHashMap([]const u8).init(allocator);
    defer variables.deinit();
    
    var services = std.StringHashMap(*node_types.Service).init(allocator);
    defer services.deinit();
    
    var ctx = ExecutionContext{
        .allocator = allocator,
        .workflow_id = "wf-1",
        .execution_id = "exec-1",
        .user_id = null,
        .variables = variables,
        .services = services,
        .inputs = std.StringHashMap(std.json.Value).init(allocator),
    };
    defer ctx.inputs.deinit();
    
    const result = try node.execute(&ctx);
    try std.testing.expect(result == .object);
}
