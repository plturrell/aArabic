//! SAP HANA Workflow Nodes - Query, Insert, Update, Delete, Transaction
const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;
const base64 = std.base64;
const ascii = std.ascii;

const node_types = @import("node_types");
const NodeInterface = node_types.NodeInterface;
const Port = node_types.Port;
const ExecutionContext = node_types.ExecutionContext;
const hana = @import("hana_client.zig");
const HanaConfig = hana.HanaConfig;
const HANA_EXECUTOR_SERVICE = "hana_executor";

pub const HanaExecutor = struct {
    context: *anyopaque,
    queryFn: *const fn (*anyopaque, []const u8) hana.HanaError!hana.HanaResult,
    executeFn: *const fn (*anyopaque, []const u8) hana.HanaError!usize,
    beginFn: *const fn (*anyopaque) hana.HanaError!void,
    commitFn: *const fn (*anyopaque) hana.HanaError!void,
    rollbackFn: *const fn (*anyopaque) hana.HanaError!void,
};

const ExecutorHandle = struct {
    executor: HanaExecutor,
    cleanup_ctx: ?*anyopaque,
    cleanup_fn: ?*const fn (*anyopaque) void,

    pub fn release(self: *ExecutorHandle) void {
        if (self.cleanup_fn) |cleanup| {
            cleanup(self.cleanup_ctx.?);
        }
    }
};

const OwnedExecutorContext = struct {
    allocator: Allocator,
    client: *hana.HanaClient,
};

const NodeError = error{
    MissingTableName,
    MissingValues,
    InvalidValuesObject,
    InvalidValues,
    InvalidSetValuesObject,
    InvalidParameters,
    MissingWhereClause,
    MissingOperations,
    InvalidOperations,
};

fn cfgStr(c: std.json.Value, k: []const u8, d: []const u8) []const u8 {
    return if (c == .object)
        if (c.object.get(k)) |v|
            if (v == .string) v.string else d
        else
            d
    else
        d;
}

fn cfgInt(c: std.json.Value, k: []const u8, d: u16) u16 {
    return if (c == .object)
        if (c.object.get(k)) |v|
            if (v == .integer) @intCast(v.integer) else d
        else
            d
    else
        d;
}

fn parseConfig(c: std.json.Value) HanaConfig {
    return .{
        .host = cfgStr(c, "host", "localhost"),
        .port = cfgInt(c, "port", 30015),
        .user = cfgStr(c, "user", "SYSTEM"),
        .password = cfgStr(c, "password", ""),
        .schema = cfgStr(c, "schema", "SYSTEM"),
    };
}

fn mkPort(a: Allocator, id: []const u8, nm: []const u8, desc: []const u8, pt: node_types.PortType, req: bool) !Port {
    return .{
        .id = try a.dupe(u8, id),
        .name = try a.dupe(u8, nm),
        .description = try a.dupe(u8, desc),
        .port_type = pt,
        .required = req,
        .default_value = null,
    };
}

fn freeBase(a: Allocator, b: *NodeInterface) void {
    for (b.inputs) |p| {
        a.free(p.id);
        a.free(p.name);
        a.free(p.description);
    }
    a.free(b.inputs);
    for (b.outputs) |p| {
        a.free(p.id);
        a.free(p.name);
        a.free(p.description);
    }
    a.free(b.outputs);
    a.free(b.id);
    a.free(b.name);
    a.free(b.node_type);
    a.free(b.description);
}

fn valueToString(value: std.json.Value) ?[]const u8 {
    return switch (value) {
        .string => value.string,
        .number_string => value.number_string,
        else => null,
    };
}

fn requireObject(value: std.json.Value, err: NodeError) NodeError!std.json.ObjectMap {
    return switch (value) {
        .object => |obj| obj,
        else => err,
    };
}

fn requireArray(value: std.json.Value, err: NodeError) NodeError!std.json.Array {
    return switch (value) {
        .array => |arr| arr,
        else => err,
    };
}

fn renderQueryWithParams(allocator: Allocator, base_sql: []const u8, params: std.json.ObjectMap) NodeError![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    var writer = buf.writer();
    var i: usize = 0;
    while (i < base_sql.len) {
        const ch = base_sql[i];
        if (ch == ':' and i + 1 < base_sql.len and isIdentChar(base_sql[i + 1])) {
            var j = i + 1;
            while (j < base_sql.len and isIdentChar(base_sql[j])) : (j += 1) {}
            const key = base_sql[i + 1 .. j];
            if (params.get(key)) |param_value| {
                try appendSqlLiteral(writer, param_value);
            } else {
                try writer.writeAll(base_sql[i..j]);
            }
            i = j;
            continue;
        }
        try writer.writeByte(ch);
        i += 1;
    }
    return buf.toOwnedSlice();
}

fn isIdentChar(ch: u8) bool {
    return ascii.isAlphanumeric(ch) or ch == '_';
}

fn maybeRenderQuery(allocator: Allocator, base_sql: []const u8, ctx: *ExecutionContext) NodeError!?[]u8 {
    if (ctx.getInput("parameters")) |params_value| {
        const params = try requireObject(params_value, NodeError.InvalidParameters);
        return try renderQueryWithParams(allocator, base_sql, params);
    }
    return null;
}

fn resultObjectWithCount(allocator: Allocator, affected: usize) !std.json.Value {
    var obj = std.json.ObjectMap.init(allocator);
    try obj.put("affected_rows", .{ .integer = @intCast(i64, affected) });
    return .{ .object = obj };
}

const OperationDescriptor = struct {
    sql: []const u8,
    expects_result: bool,
};

fn parseOperation(value: std.json.Value) NodeError!OperationDescriptor {
    if (valueToString(value)) |sql_text| {
        return .{ .sql = sql_text, .expects_result = false };
    }

    const obj = try requireObject(value, NodeError.InvalidOperations);
    const sql_value = obj.get("sql") orelse return NodeError.InvalidOperations;
    const sql_text = valueToString(sql_value) orelse return NodeError.InvalidOperations;

    var expects_result = false;
    if (obj.get("expects_result")) |flag| {
        expects_result = switch (flag) {
            .bool => flag.bool,
            else => return NodeError.InvalidOperations,
        };
    } else if (obj.get("type")) |type_value| {
        if (valueToString(type_value)) |type_str| {
            expects_result = std.mem.eql(u8, type_str, "query");
        }
    }

    return .{ .sql = sql_text, .expects_result = expects_result };
}

fn acquireExecutor(config: HanaConfig, allocator: Allocator, ctx: *ExecutionContext) !ExecutorHandle {
    if (ctx.getService(HANA_EXECUTOR_SERVICE)) |service| {
        if (service.getContextData()) |raw| {
            const executor = @ptrCast(*HanaExecutor, raw);
            return ExecutorHandle{ .executor = executor.*, .cleanup_ctx = null, .cleanup_fn = null };
        }
    }

    const owned = try allocator.create(OwnedExecutorContext);
    errdefer allocator.destroy(owned);

    owned.* = .{ .allocator = allocator, .client = try allocator.create(hana.HanaClient) };
    errdefer allocator.destroy(owned.client);

    owned.client.* = hana.HanaClient.init(allocator, config);
    try owned.client.connect();

    return ExecutorHandle{
        .executor = .{
            .context = owned.client,
            .queryFn = realQuery,
            .executeFn = realExecute,
            .beginFn = realBegin,
            .commitFn = realCommit,
            .rollbackFn = realRollback,
        },
        .cleanup_ctx = owned,
        .cleanup_fn = releaseOwnedExecutor,
    };
}

fn realQuery(ctx_ptr: *anyopaque, sql: []const u8) hana.HanaError!hana.HanaResult {
    const client = @ptrCast(*hana.HanaClient, ctx_ptr);
    return client.query(sql);
}

fn realExecute(ctx_ptr: *anyopaque, sql: []const u8) hana.HanaError!usize {
    const client = @ptrCast(*hana.HanaClient, ctx_ptr);
    return client.execute(sql);
}

fn realBegin(ctx_ptr: *anyopaque) hana.HanaError!void {
    const client = @ptrCast(*hana.HanaClient, ctx_ptr);
    return client.beginTransaction();
}

fn realCommit(ctx_ptr: *anyopaque) hana.HanaError!void {
    const client = @ptrCast(*hana.HanaClient, ctx_ptr);
    return client.commit();
}

fn realRollback(ctx_ptr: *anyopaque) hana.HanaError!void {
    const client = @ptrCast(*hana.HanaClient, ctx_ptr);
    return client.rollback();
}

fn releaseOwnedExecutor(ctx_ptr: *anyopaque) void {
    const owned = @ptrCast(*OwnedExecutorContext, ctx_ptr);
    owned.client.deinit();
    owned.allocator.destroy(owned.client);
    owned.allocator.destroy(owned);
}

fn getInputString(ctx: *ExecutionContext, port_id: []const u8) ?[]const u8 {
    if (ctx.getInput(port_id)) |value| {
        return switch (value) {
            .string => value.string,
            .number_string => value.number_string,
            else => null,
        };
    }
    return null;
}

fn hanaValueToJson(allocator: Allocator, value: hana.HanaValue) !std.json.Value {
    return switch (value) {
        .null_value => .{ .null = {} },
        .bool_value => |b| .{ .bool = b },
        .int_value => |i| .{ .integer = i },
        .float_value => |f| .{ .float = f },
        .timestamp_value => |ts| .{ .string = try std.fmt.allocPrint(allocator, "{d}", .{ts}) },
        .text_value => |s| .{ .string = try allocator.dupe(u8, s) },
        .bytes_value => |bytes| blk: {
            const encoded_len = base64.standard.Encoder.calcSize(bytes.len);
            var buf = try allocator.alloc(u8, encoded_len);
            _ = base64.standard.Encoder.encode(buf, bytes);
            break :blk .{ .string = buf };
        },
    };
}

fn hanaResultToJson(allocator: Allocator, result: *hana.HanaResult) !std.json.Value {
    var root = std.json.ObjectMap.init(allocator);
    var rows_array = std.json.Array.init(allocator);

    for (result.rows) |row| {
        var row_obj = std.json.ObjectMap.init(allocator);
        for (row.columns, row.values) |column_name, row_value| {
            const json_value = try hanaValueToJson(allocator, row_value);
            try row_obj.put(try allocator.dupe(u8, column_name), json_value);
        }
        try rows_array.append(.{ .object = row_obj });
    }

    try root.put("rows", .{ .array = rows_array });
    try root.put("count", .{ .integer = @intCast(i64, result.row_count) });
    try root.put("affected_rows", .{ .integer = @intCast(i64, result.affected_rows) });

    if (result.command_tag) |tag| {
        try root.put("command_tag", .{ .string = try allocator.dupe(u8, tag) });
    }

    return .{ .object = root };
}

fn writeEscapedLiteral(writer: anytype, text: []const u8) !void {
    try writer.writeByte('\'');
    for (text) |ch| {
        if (ch == '\'') {
            try writer.writeAll("''");
        } else {
            try writer.writeByte(ch);
        }
    }
    try writer.writeByte('\'');
}

fn appendSqlLiteral(writer: anytype, value: std.json.Value) NodeError!void {
    switch (value) {
        .null => try writer.writeAll("NULL"),
        .bool => |b| try writer.writeAll(if (b) "TRUE" else "FALSE"),
        .integer => |i| try writer.print("{d}", .{i}),
        .float => |f| try writer.print("{}", .{f}),
        .number_string => |s| try writeEscapedLiteral(writer, s),
        .string => |s| try writeEscapedLiteral(writer, s),
        else => return NodeError.InvalidValues,
    }
}

fn buildInsertSql(allocator: Allocator, table_name: []const u8, values_obj: std.json.ObjectMap) NodeError![]const u8 {
    var sql_buf = std.ArrayList(u8).init(allocator);
    errdefer sql_buf.deinit();
    var writer = sql_buf.writer();

    try writer.print("INSERT INTO \"{s}\" (", .{table_name});
    var first = true;
    var iter = values_obj.iterator();
    while (iter.next()) |entry| {
        if (!first) try writer.writeAll(", ");
        first = false;
        try writer.print("\"{s}\"", .{entry.key_ptr.*});
    }
    if (first) return NodeError.MissingValues;
    try writer.writeAll(") VALUES (");
    first = true;
    iter = values_obj.iterator();
    while (iter.next()) |entry| {
        if (!first) try writer.writeAll(", ");
        first = false;
        try appendSqlLiteral(&writer, entry.value_ptr.*);
    }
    try writer.writeByte(')');

    return sql_buf.toOwnedSlice();
}

fn buildUpdateSql(allocator: Allocator, table_name: []const u8, set_obj: std.json.ObjectMap, where_clause: []const u8) NodeError![]const u8 {
    var sql_buf = std.ArrayList(u8).init(allocator);
    errdefer sql_buf.deinit();
    var writer = sql_buf.writer();

    try writer.print("UPDATE \"{s}\" SET ", .{table_name});
    var first = true;
    var iter = set_obj.iterator();
    while (iter.next()) |entry| {
        if (!first) try writer.writeAll(", ");
        first = false;
        try writer.print("\"{s}\" = ", .{entry.key_ptr.*});
        try appendSqlLiteral(&writer, entry.value_ptr.*);
    }
    if (first) return NodeError.MissingValues;
    try writer.print(" WHERE {s}", .{where_clause});

    return sql_buf.toOwnedSlice();
}

fn buildDeleteSql(allocator: Allocator, table_name: []const u8, where_clause: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator, "DELETE FROM \"{s}\" WHERE {s}", .{ table_name, where_clause });
}

/// HanaQueryNode - Execute SELECT queries
pub const HanaQueryNode = struct {
    allocator: Allocator, base: NodeInterface, config: HanaConfig, query: []const u8,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, cfg: std.json.Value) !*HanaQueryNode {
        const n = try a.create(HanaQueryNode);
        errdefer a.destroy(n);
        const ins = try a.alloc(Port, 1);
        ins[0] = try mkPort(a, "parameters", "Query Parameters", "Parameters for prepared statement", .object, false);
        const outs = try a.alloc(Port, 2);
        outs[0] = try mkPort(a, "rows", "Result Rows", "Query result rows as array", .array, true);
        outs[1] = try mkPort(a, "count", "Row Count", "Number of rows returned", .number, true);
        n.* = .{ .allocator = a, .base = .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .node_type = try a.dupe(u8, "hana_query"), .description = try a.dupe(u8, "Execute HANA SELECT query"), .category = .data, .inputs = ins, .outputs = outs, .config = cfg }, .config = parseConfig(cfg), .query = try a.dupe(u8, cfgStr(cfg, "query", "SELECT 1 FROM DUMMY")) };
        return n;
    }
    pub fn deinit(self: *HanaQueryNode) void {
        freeBase(self.allocator, &self.base);
        self.allocator.free(self.query);
        self.allocator.destroy(self);
    }
    pub fn execute(self: *HanaQueryNode, ctx: *ExecutionContext) !std.json.Value {
        var handle = try acquireExecutor(self.config, self.allocator, ctx);
        defer handle.release();

        const rendered = try maybeRenderQuery(self.allocator, self.query, ctx);
        defer if (rendered) |sql_buf| self.allocator.free(sql_buf);
        const sql = rendered orelse self.query;

        var query_result = try handle.executor.queryFn(handle.executor.context, sql);
        const client = @ptrCast(*hana.HanaClient, handle.executor.context);
        defer query_result.deinit(client.allocator);

        return try hanaResultToJson(self.allocator, &query_result);
    }
};

/// HanaInsertNode - Execute INSERT statements
pub const HanaInsertNode = struct {
    allocator: Allocator, base: NodeInterface, config: HanaConfig, table: []const u8,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, cfg: std.json.Value) !*HanaInsertNode {
        const n = try a.create(HanaInsertNode);
        errdefer a.destroy(n);
        const ins = try a.alloc(Port, 2);
        ins[0] = try mkPort(a, "table", "Table Name", "Target table name", .string, false);
        ins[1] = try mkPort(a, "values", "Values", "Column values to insert", .object, true);
        const outs = try a.alloc(Port, 1);
        outs[0] = try mkPort(a, "affected_rows", "Affected Rows", "Number of rows inserted", .number, true);
        n.* = .{ .allocator = a, .base = .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .node_type = try a.dupe(u8, "hana_insert"), .description = try a.dupe(u8, "Insert data into HANA table"), .category = .data, .inputs = ins, .outputs = outs, .config = cfg }, .config = parseConfig(cfg), .table = try a.dupe(u8, cfgStr(cfg, "table", "DATA")) };
        return n;
    }
    pub fn deinit(self: *HanaInsertNode) void {
        freeBase(self.allocator, &self.base);
        self.allocator.free(self.table);
        self.allocator.destroy(self);
    }
    pub fn execute(self: *HanaInsertNode, ctx: *ExecutionContext) !std.json.Value {
        const table_name = getInputString(ctx, "table") orelse self.table;
        if (table_name.len == 0) return NodeError.MissingTableName;

        const values_value = ctx.getInput("values") orelse return NodeError.MissingValues;
        const values_obj = try requireObject(values_value, NodeError.InvalidValuesObject);

        var sql = try buildInsertSql(self.allocator, table_name, values_obj);
        defer self.allocator.free(sql);

        var handle = try acquireExecutor(self.config, self.allocator, ctx);
        defer handle.release();

        const affected = try handle.executor.executeFn(handle.executor.context, sql);
        return try resultObjectWithCount(self.allocator, affected);
    }
};

/// HanaUpdateNode - Execute UPDATE statements
pub const HanaUpdateNode = struct {
    allocator: Allocator, base: NodeInterface, config: HanaConfig, table: []const u8,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, cfg: std.json.Value) !*HanaUpdateNode {
        const n = try a.create(HanaUpdateNode);
        errdefer a.destroy(n);
        const ins = try a.alloc(Port, 3);
        ins[0] = try mkPort(a, "table", "Table Name", "Target table name", .string, false);
        ins[1] = try mkPort(a, "set_values", "SET Values", "Column values to update", .object, true);
        ins[2] = try mkPort(a, "where_clause", "WHERE Clause", "Conditions for update", .string, true);
        const outs = try a.alloc(Port, 1);
        outs[0] = try mkPort(a, "affected_rows", "Affected Rows", "Number of rows updated", .number, true);
        n.* = .{ .allocator = a, .base = .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .node_type = try a.dupe(u8, "hana_update"), .description = try a.dupe(u8, "Update HANA table rows"), .category = .data, .inputs = ins, .outputs = outs, .config = cfg }, .config = parseConfig(cfg), .table = try a.dupe(u8, cfgStr(cfg, "table", "DATA")) };
        return n;
    }
    pub fn deinit(self: *HanaUpdateNode) void {
        freeBase(self.allocator, &self.base);
        self.allocator.free(self.table);
        self.allocator.destroy(self);
    }
    pub fn execute(self: *HanaUpdateNode, ctx: *ExecutionContext) !std.json.Value {
        const table_name = getInputString(ctx, "table") orelse self.table;
        if (table_name.len == 0) return NodeError.MissingTableName;

        const set_value = ctx.getInput("set_values") orelse return NodeError.MissingValues;
        const set_obj = try requireObject(set_value, NodeError.InvalidSetValuesObject);
        const where_clause = getInputString(ctx, "where_clause") orelse return NodeError.MissingWhereClause;

        var sql = try buildUpdateSql(self.allocator, table_name, set_obj, where_clause);
        defer self.allocator.free(sql);

        var handle = try acquireExecutor(self.config, self.allocator, ctx);
        defer handle.release();

        const affected = try handle.executor.executeFn(handle.executor.context, sql);
        return try resultObjectWithCount(self.allocator, affected);
    }
};

/// HanaDeleteNode - Execute DELETE statements
pub const HanaDeleteNode = struct {
    allocator: Allocator, base: NodeInterface, config: HanaConfig, table: []const u8,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, cfg: std.json.Value) !*HanaDeleteNode {
        const n = try a.create(HanaDeleteNode);
        errdefer a.destroy(n);
        const ins = try a.alloc(Port, 2);
        ins[0] = try mkPort(a, "table", "Table Name", "Target table name", .string, false);
        ins[1] = try mkPort(a, "where_clause", "WHERE Clause", "Conditions for deletion", .string, true);
        const outs = try a.alloc(Port, 1);
        outs[0] = try mkPort(a, "affected_rows", "Affected Rows", "Number of rows deleted", .number, true);
        n.* = .{ .allocator = a, .base = .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .node_type = try a.dupe(u8, "hana_delete"), .description = try a.dupe(u8, "Delete rows from HANA table"), .category = .data, .inputs = ins, .outputs = outs, .config = cfg }, .config = parseConfig(cfg), .table = try a.dupe(u8, cfgStr(cfg, "table", "DATA")) };
        return n;
    }
    pub fn deinit(self: *HanaDeleteNode) void {
        freeBase(self.allocator, &self.base);
        self.allocator.free(self.table);
        self.allocator.destroy(self);
    }
    pub fn execute(self: *HanaDeleteNode, ctx: *ExecutionContext) !std.json.Value {
        const table_name = getInputString(ctx, "table") orelse self.table;
        if (table_name.len == 0) return NodeError.MissingTableName;
        const where_clause = getInputString(ctx, "where_clause") orelse return NodeError.MissingWhereClause;

        var sql = try buildDeleteSql(self.allocator, table_name, where_clause);
        defer self.allocator.free(sql);

        var handle = try acquireExecutor(self.config, self.allocator, ctx);
        defer handle.release();

        const affected = try handle.executor.executeFn(handle.executor.context, sql);
        return try resultObjectWithCount(self.allocator, affected);
    }
};

/// HanaTransactionNode - Transaction wrapper
pub const HanaTransactionNode = struct {
    allocator: Allocator, base: NodeInterface, config: HanaConfig,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, cfg: std.json.Value) !*HanaTransactionNode {
        const n = try a.create(HanaTransactionNode);
        errdefer a.destroy(n);
        const ins = try a.alloc(Port, 1);
        ins[0] = try mkPort(a, "operations", "Operations", "Array of SQL statements to execute", .array, true);
        const outs = try a.alloc(Port, 2);
        outs[0] = try mkPort(a, "success", "Success", "Whether transaction succeeded", .boolean, true);
        outs[1] = try mkPort(a, "results", "Results", "Array of operation results", .array, false);
        n.* = .{ .allocator = a, .base = .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .node_type = try a.dupe(u8, "hana_transaction"), .description = try a.dupe(u8, "Execute HANA transaction with auto-rollback"), .category = .data, .inputs = ins, .outputs = outs, .config = cfg }, .config = parseConfig(cfg) };
        return n;
    }
    pub fn deinit(self: *HanaTransactionNode) void {
        freeBase(self.allocator, &self.base);
        self.allocator.destroy(self);
    }
    pub fn execute(self: *HanaTransactionNode, ctx: *ExecutionContext) !std.json.Value {
        const operations_value = ctx.getInput("operations") orelse return NodeError.MissingOperations;
        const operations = try requireArray(operations_value, NodeError.InvalidOperations);
        if (operations.items.len == 0) return NodeError.MissingOperations;

        var handle = try acquireExecutor(self.config, self.allocator, ctx);
        defer handle.release();
        const executor = handle.executor;

        try executor.beginFn(executor.context);
        var transaction_complete = false;
        defer if (!transaction_complete) executor.rollbackFn(executor.context) catch {};

        var results_array = std.json.Array.init(self.allocator);

        for (operations.items) |operation_value| {
            const operation = try parseOperation(operation_value);

            var entry = std.json.ObjectMap.init(self.allocator);
            try entry.put("sql", .{ .string = try self.allocator.dupe(u8, operation.sql) });

            if (operation.expects_result) {
                var query_result = try executor.queryFn(executor.context, operation.sql);
                const client = @ptrCast(*hana.HanaClient, executor.context);
                defer query_result.deinit(client.allocator);

                const json_value = try hanaResultToJson(self.allocator, &query_result);
                try entry.put("result", json_value);
            } else {
                const affected = try executor.executeFn(executor.context, operation.sql);
                try entry.put("affected_rows", .{ .integer = @intCast(i64, affected) });
            }

            try results_array.append(.{ .object = entry });
        }

        try executor.commitFn(executor.context);
        transaction_complete = true;

        var root = std.json.ObjectMap.init(self.allocator);
        try root.put("success", .{ .bool = true });
        try root.put("results", .{ .array = results_array });
        return .{ .object = root };
    }
};

// Tests

test "HanaQueryNode creation and configuration" {
    const allocator = std.testing.allocator;

    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    try config.put("host", .{ .string = "hana.example.com" });
    try config.put("port", .{ .integer = 443 });
    try config.put("user", .{ .string = "DBADMIN" });
    try config.put("schema", .{ .string = "SALES" });
    try config.put("query", .{ .string = "SELECT * FROM ORDERS" });

    const node = try HanaQueryNode.init(allocator, "query-1", "Sales Query", .{ .object = config });
    defer node.deinit();

    try std.testing.expectEqualStrings("query-1", node.base.id);
    try std.testing.expectEqualStrings("hana_query", node.base.node_type);
    try std.testing.expectEqualStrings("SELECT * FROM ORDERS", node.query);
    try std.testing.expectEqual(@as(usize, 1), node.base.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), node.base.outputs.len);
}

test "HanaInsertNode creation with ports" {
    const allocator = std.testing.allocator;

    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    try config.put("table", .{ .string = "CUSTOMERS" });

    const node = try HanaInsertNode.init(allocator, "insert-1", "Insert Customer", .{ .object = config });
    defer node.deinit();

    try std.testing.expectEqualStrings("insert-1", node.base.id);
    try std.testing.expectEqualStrings("hana_insert", node.base.node_type);
    try std.testing.expectEqualStrings("CUSTOMERS", node.table);
    try std.testing.expectEqualStrings("table", node.base.inputs[0].id);
    try std.testing.expectEqualStrings("values", node.base.inputs[1].id);
    try std.testing.expectEqualStrings("affected_rows", node.base.outputs[0].id);
}

test "HanaUpdateNode creation with input ports" {
    const allocator = std.testing.allocator;

    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    try config.put("table", .{ .string = "PRODUCTS" });

    const node = try HanaUpdateNode.init(allocator, "update-1", "Update Product", .{ .object = config });
    defer node.deinit();

    try std.testing.expectEqualStrings("hana_update", node.base.node_type);
    try std.testing.expectEqual(@as(usize, 3), node.base.inputs.len);
    try std.testing.expectEqualStrings("set_values", node.base.inputs[1].id);
    try std.testing.expectEqualStrings("where_clause", node.base.inputs[2].id);
}

test "HanaDeleteNode creation" {
    const allocator = std.testing.allocator;

    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    try config.put("table", .{ .string = "LOGS" });

    const node = try HanaDeleteNode.init(allocator, "delete-1", "Delete Logs", .{ .object = config });
    defer node.deinit();

    try std.testing.expectEqualStrings("hana_delete", node.base.node_type);
    try std.testing.expectEqualStrings("LOGS", node.table);
    try std.testing.expectEqual(@as(usize, 2), node.base.inputs.len);
    try std.testing.expectEqualStrings("where_clause", node.base.inputs[1].id);
}

test "HanaTransactionNode creation with output ports" {
    const allocator = std.testing.allocator;

    var config = std.json.ObjectMap.init(allocator);
    defer config.deinit();
    try config.put("host", .{ .string = "localhost" });

    const node = try HanaTransactionNode.init(allocator, "tx-1", "Order Transaction", .{ .object = config });
    defer node.deinit();

    try std.testing.expectEqualStrings("hana_transaction", node.base.node_type);
    try std.testing.expectEqual(@as(usize, 1), node.base.inputs.len);
    try std.testing.expectEqualStrings("operations", node.base.inputs[0].id);
    try std.testing.expectEqual(@as(usize, 2), node.base.outputs.len);
    try std.testing.expectEqualStrings("success", node.base.outputs[0].id);
    try std.testing.expectEqualStrings("results", node.base.outputs[1].id);
}
