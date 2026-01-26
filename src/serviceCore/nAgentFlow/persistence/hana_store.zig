//! SAP HANA Persistence Layer for nWorkflow
//!
//! Provides enterprise-grade column store persistence for workflow definitions and execution history.
//! Optimized for SAP HANA's in-memory capabilities with multi-tenant support.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

const hana = @import("hana_sdk");
const hana_client = hana.client;
const HanaConfig = hana.Config;
const HanaClient = hana.Client;
const HanaResult = hana.QueryResult;
const HanaRow = hana_client.Row;
const HanaValue = hana_client.Value;

/// Execution status for workflow runs
pub const ExecutionStatus = enum {
    pending,
    running,
    completed,
    failed,
    cancelled,

    pub fn toString(self: ExecutionStatus) []const u8 {
        return switch (self) {
            .pending => "pending",
            .running => "running",
            .completed => "completed",
            .failed => "failed",
            .cancelled => "cancelled",
        };
    }

    pub fn fromString(s: []const u8) !ExecutionStatus {
        if (std.mem.eql(u8, s, "pending")) return .pending;
        if (std.mem.eql(u8, s, "running")) return .running;
        if (std.mem.eql(u8, s, "completed")) return .completed;
        if (std.mem.eql(u8, s, "failed")) return .failed;
        if (std.mem.eql(u8, s, "cancelled")) return .cancelled;
        return error.InvalidStatus;
    }
};

/// Workflow summary for listing operations
pub const WorkflowSummary = struct {
    id: []const u8,
    name: []const u8,
    status: []const u8,
    version: u32,
    updated_at: i64,

    pub fn deinit(self: *WorkflowSummary, a: Allocator) void {
        a.free(self.id);
        a.free(self.name);
        a.free(self.status);
    }

    pub fn clone(self: *const WorkflowSummary, a: Allocator) !WorkflowSummary {
        return .{
            .id = try a.dupe(u8, self.id),
            .name = try a.dupe(u8, self.name),
            .status = try a.dupe(u8, self.status),
            .version = self.version,
            .updated_at = self.updated_at,
        };
    }
};

/// Execution record for tracking workflow runs
pub const ExecutionRecord = struct {
    id: []const u8,
    workflow_id: []const u8,
    status: []const u8,
    started_at: i64,
    completed_at: ?i64,
    result_json: ?[]const u8,

    pub fn init(a: Allocator, id: []const u8, wf_id: []const u8, status: []const u8) !ExecutionRecord {
        return .{
            .id = try a.dupe(u8, id),
            .workflow_id = try a.dupe(u8, wf_id),
            .status = try a.dupe(u8, status),
            .started_at = std.time.timestamp(),
            .completed_at = null,
            .result_json = null,
        };
    }

    pub fn deinit(self: *ExecutionRecord, a: Allocator) void {
        a.free(self.id);
        a.free(self.workflow_id);
        a.free(self.status);
        if (self.result_json) |r| a.free(r);
    }

    pub fn clone(self: *const ExecutionRecord, a: Allocator) !ExecutionRecord {
        return .{
            .id = try a.dupe(u8, self.id),
            .workflow_id = try a.dupe(u8, self.workflow_id),
            .status = try a.dupe(u8, self.status),
            .started_at = self.started_at,
            .completed_at = self.completed_at,
            .result_json = if (self.result_json) |r| try a.dupe(u8, r) else null,
        };
    }
};

/// Internal workflow storage record
pub const WorkflowRecord = struct {
    id: []const u8,
    name: []const u8,
    definition_json: []const u8,
    tenant_id: []const u8,
    status: []const u8,
    version: u32,
    created_at: i64,
    updated_at: i64,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, definition: []const u8, tenant_id: []const u8) !WorkflowRecord {
        const now = std.time.timestamp();
        return .{
            .id = try a.dupe(u8, id),
            .name = try a.dupe(u8, name),
            .definition_json = try a.dupe(u8, definition),
            .tenant_id = try a.dupe(u8, tenant_id),
            .status = try a.dupe(u8, "active"),
            .version = 1,
            .created_at = now,
            .updated_at = now,
        };
    }

    pub fn deinit(self: *WorkflowRecord, a: Allocator) void {
        a.free(self.id);
        a.free(self.name);
        a.free(self.definition_json);
        a.free(self.tenant_id);
        a.free(self.status);
    }

    pub fn clone(self: *const WorkflowRecord, a: Allocator) !WorkflowRecord {
        return .{
            .id = try a.dupe(u8, self.id),
            .name = try a.dupe(u8, self.name),
            .definition_json = try a.dupe(u8, self.definition_json),
            .tenant_id = try a.dupe(u8, self.tenant_id),
            .status = try a.dupe(u8, self.status),
            .version = self.version,
            .created_at = self.created_at,
            .updated_at = self.updated_at,
        };
    }
};

// ===== HANA SQL DDL SCHEMA =====
pub const CREATE_WORKFLOWS_TABLE: []const u8 =
    \\CREATE COLUMN TABLE IF NOT EXISTS {PREFIX}_workflows (
    \\    id NVARCHAR(36) PRIMARY KEY,
    \\    name NVARCHAR(255) NOT NULL,
    \\    definition_json NCLOB NOT NULL,
    \\    tenant_id NVARCHAR(36) NOT NULL,
    \\    status NVARCHAR(20) NOT NULL DEFAULT 'active',
    \\    version INTEGER NOT NULL DEFAULT 1,
    \\    created_at BIGINT NOT NULL,
    \\    updated_at BIGINT NOT NULL)
;
pub const CREATE_EXECUTIONS_TABLE: []const u8 =
    \\CREATE COLUMN TABLE IF NOT EXISTS {PREFIX}_executions (
    \\    id NVARCHAR(36) PRIMARY KEY,
    \\    workflow_id NVARCHAR(36) NOT NULL,
    \\    status NVARCHAR(20) NOT NULL DEFAULT 'pending',
    \\    started_at BIGINT NOT NULL,
    \\    completed_at BIGINT,
    \\    result_json NCLOB,
    \\    FOREIGN KEY (workflow_id) REFERENCES {PREFIX}_workflows(id) ON DELETE CASCADE)
;
pub const CREATE_VERSIONS_TABLE: []const u8 =
    \\CREATE COLUMN TABLE IF NOT EXISTS {PREFIX}_workflow_versions (
    \\    workflow_id NVARCHAR(36) NOT NULL,
    \\    version INTEGER NOT NULL,
    \\    data NCLOB,
    \\    created_at BIGINT NOT NULL,
    \\    PRIMARY KEY (workflow_id, version),
    \\    FOREIGN KEY (workflow_id) REFERENCES {PREFIX}_workflows(id) ON DELETE CASCADE)
;
pub const CREATE_INDEXES: []const u8 =
    \\CREATE INDEX IF NOT EXISTS idx_{PREFIX}_workflows_tenant ON {PREFIX}_workflows(tenant_id);
    \\CREATE INDEX IF NOT EXISTS idx_{PREFIX}_workflows_name ON {PREFIX}_workflows(name);
    \\CREATE INDEX IF NOT EXISTS idx_{PREFIX}_workflows_status ON {PREFIX}_workflows(status);
    \\CREATE INDEX IF NOT EXISTS idx_{PREFIX}_executions_workflow ON {PREFIX}_executions(workflow_id);
    \\CREATE INDEX IF NOT EXISTS idx_{PREFIX}_executions_status ON {PREFIX}_executions(status)
;

pub fn generateUuid(a: Allocator) ![]const u8 {
    var rb: [16]u8 = undefined;
    std.crypto.random.bytes(&rb);
    rb[6] = (rb[6] & 0x0F) | 0x40;
    rb[8] = (rb[8] & 0x3F) | 0x80;
    return std.fmt.allocPrint(a, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{ rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], rb[6], rb[7], rb[8], rb[9], rb[10], rb[11], rb[12], rb[13], rb[14], rb[15] });
}

// ===== HANA WORKFLOW STORE =====

pub const HanaWorkflowStore = struct {
    allocator: Allocator,
    config: HanaConfig,
    table_prefix: []const u8,
    client: *HanaClient,

    const Self = @This();

    pub fn init(allocator: Allocator, config: HanaConfig, table_prefix: []const u8) !*Self {
        const store = try allocator.create(Self);
        errdefer allocator.destroy(store);

        const client = try hana.connectWithAllocator(allocator, config);

        store.* = .{
            .allocator = allocator,
            .config = config,
            .table_prefix = try allocator.dupe(u8, table_prefix),
            .client = client,
        };
        return store;
    }

    pub fn deinit(self: *Self) void {
        self.client.deinit();
        self.allocator.free(self.table_prefix);
        self.allocator.destroy(self);
    }

    fn workflowTableName(self: *Self) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}_workflows", .{self.table_prefix});
    }

    fn executionTableName(self: *Self) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}_executions", .{self.table_prefix});
    }

    fn versionsTableName(self: *Self) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}_workflow_versions", .{self.table_prefix});
    }

    fn runQuery(self: *Self, sql: []const u8) !HanaResult {
        return self.client.query(sql, self.allocator);
    }

    fn runExecute(self: *Self, sql: []const u8) !void {
        return self.client.execute(sql);
    }

    fn replacePrefix(self: *Self, template: []const u8) ![]u8 {
        return std.mem.replaceOwned(u8, self.allocator, template, "{PREFIX}", self.table_prefix);
    }

    pub fn createSchema(self: *Self) !void {
        const ddl_workflows = try self.replacePrefix(CREATE_WORKFLOWS_TABLE);
        defer self.allocator.free(ddl_workflows);
        const ddl_executions = try self.replacePrefix(CREATE_EXECUTIONS_TABLE);
        defer self.allocator.free(ddl_executions);
        const ddl_versions = try self.replacePrefix(CREATE_VERSIONS_TABLE);
        defer self.allocator.free(ddl_versions);
        const ddl_indexes = try self.replacePrefix(CREATE_INDEXES);
        defer self.allocator.free(ddl_indexes);

        try self.runExecute(ddl_workflows);
        try self.runExecute(ddl_executions);
        try self.runExecute(ddl_versions);
        try self.runExecute(ddl_indexes);
    }

    pub fn saveWorkflow(self: *Self, workflow: WorkflowRecord) !void {
        const table = try self.workflowTableName();
        defer self.allocator.free(table);

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "INSERT INTO \"{s}\" (id, name, definition_json, tenant_id, status, version, created_at, updated_at) VALUES ('{s}', '{s}', '{s}', '{s}', '{s}', {d}, {d}, {d})",
            .{
                table,
                workflow.id,
                workflow.name,
                workflow.definition_json,
                workflow.tenant_id,
                workflow.status,
                workflow.version,
                workflow.created_at,
                workflow.updated_at,
            },
        );
        defer self.allocator.free(sql);
        try self.runExecute(sql);
    }

    pub fn loadWorkflow(self: *Self, id: []const u8) !?WorkflowRecord {
        const table = try self.workflowTableName();
        defer self.allocator.free(table);
        const sql = try std.fmt.allocPrint(self.allocator, "SELECT id, name, definition_json, tenant_id, status, version, created_at, updated_at FROM \"{s}\" WHERE id = '{s}' AND status <> 'deleted'", .{ table, id });
        defer self.allocator.free(sql);

        var result = try self.runQuery(sql);
        defer result.deinit();
        if (result.rows.len == 0) return null;
        return try self.rowToWorkflow(result.rows[0]);
    }

    fn rowToWorkflow(self: *Self, row: HanaRow) !WorkflowRecord {
        const id = try self.extractText(row, "id");
        const name = try self.extractText(row, "name");
        const definition = try self.extractText(row, "definition_json");
        const tenant = try self.extractText(row, "tenant_id");
        const status = try self.extractText(row, "status");
        const version = try self.extractInt(row, "version");
        const created_at = try self.extractInt(row, "created_at");
        const updated_at = try self.extractInt(row, "updated_at");

        return .{
            .id = id,
            .name = name,
            .definition_json = definition,
            .tenant_id = tenant,
            .status = status,
            .version = @intCast(version),
            .created_at = created_at,
            .updated_at = updated_at,
        };
    }

    fn extractText(self: *Self, row: HanaRow, column: []const u8) ![]const u8 {
        const value = row.getValue(column) orelse return error.InvalidField;
        if (value.asString()) |s| {
            return try self.allocator.dupe(u8, s);
        }
        return switch (value) {
            .null_value => try self.allocator.dupe(u8, ""),
            else => error.InvalidField,
        };
    }

    fn extractInt(self: *Self, row: HanaRow, column: []const u8) !i64 {
        _ = self;
        const value = row.getValue(column) orelse return error.InvalidField;
        if (value.asInt()) |v| return v;
        return error.InvalidField;
    }

    fn extractOptionalInt(self: *Self, row: HanaRow, column: []const u8) !?i64 {
        _ = self;
        const value = row.getValue(column) orelse return null;
        return switch (value) {
            .null_value => null,
            else => value.asInt() orelse error.InvalidField,
        };
    }

    fn extractOptionalText(self: *Self, row: HanaRow, column: []const u8) !?[]const u8 {
        const value = row.getValue(column) orelse return null;
        return switch (value) {
            .null_value => null,
            else => if (value.asString()) |s| try self.allocator.dupe(u8, s) else error.InvalidField,
        };
    }

    pub fn deleteWorkflow(self: *Self, id: []const u8) !void {
        const table = try self.workflowTableName();
        defer self.allocator.free(table);
        const sql = try std.fmt.allocPrint(self.allocator, "UPDATE \"{s}\" SET status = 'deleted', updated_at = {d} WHERE id = '{s}'", .{ table, std.time.timestamp(), id });
        defer self.allocator.free(sql);
        try self.runExecute(sql);
    }

    pub fn listWorkflows(self: *Self, tenant_id: ?[]const u8, limit: u32, offset: u32) ![]WorkflowSummary {
        const table = try self.workflowTableName();
        defer self.allocator.free(table);
        const filter = if (tenant_id) |tid| try std.fmt.allocPrint(self.allocator, "AND tenant_id = '{s}'", .{tid}) else try self.allocator.dupe(u8, "");
        defer self.allocator.free(filter);
        const sql = try std.fmt.allocPrint(self.allocator, "SELECT id, name, status, version, updated_at FROM \"{s}\" WHERE status <> 'deleted' {s} ORDER BY updated_at DESC LIMIT {d} OFFSET {d}", .{ table, filter, limit, offset });
        defer self.allocator.free(sql);

        var result = try self.runQuery(sql);
        defer result.deinit();

        var summaries = std.ArrayList(WorkflowSummary){};
        errdefer {
            for (summaries.items) |*summary| summary.deinit(self.allocator);
            summaries.deinit();
        }

        for (result.rows) |row| {
            try summaries.append(self.allocator, try self.rowToWorkflowSummary(row));
        }

        return summaries.toOwnedSlice(self.allocator);
    }

    fn rowToWorkflowSummary(self: *Self, row: HanaRow) !WorkflowSummary {
        return .{
            .id = try self.extractText(row, "id"),
            .name = try self.extractText(row, "name"),
            .status = try self.extractText(row, "status"),
            .version = @intCast(try self.extractInt(row, "version")),
            .updated_at = try self.extractInt(row, "updated_at"),
        };
    }

    pub fn saveExecution(self: *Self, execution: ExecutionRecord) !void {
        const table = try self.executionTableName();
        defer self.allocator.free(table);

        const completed_fragment = if (execution.completed_at) |ts|
            try std.fmt.allocPrint(self.allocator, "{d}", .{ts})
        else
            try self.allocator.dupe(u8, "NULL");
        defer self.allocator.free(completed_fragment);

        const result_fragment = if (execution.result_json) |payload|
            try std.fmt.allocPrint(self.allocator, "'{s}'", .{payload})
        else
            try self.allocator.dupe(u8, "NULL");
        defer self.allocator.free(result_fragment);

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "INSERT INTO \"{s}\" (id, workflow_id, status, started_at, completed_at, result_json) VALUES ('{s}', '{s}', '{s}', {d}, {s}, {s})",
            .{
                table,
                execution.id,
                execution.workflow_id,
                execution.status,
                execution.started_at,
                completed_fragment,
                result_fragment,
            },
        );
        defer self.allocator.free(sql);

        try self.runExecute(sql);
    }

    pub fn getExecution(self: *Self, id: []const u8) !?ExecutionRecord {
        const table = try self.executionTableName();
        defer self.allocator.free(table);
        const sql = try std.fmt.allocPrint(self.allocator, "SELECT id, workflow_id, status, started_at, completed_at, result_json FROM \"{s}\" WHERE id = '{s}'", .{ table, id });
        defer self.allocator.free(sql);

        var result = try self.runQuery(sql);
        defer result.deinit();
        if (result.rows.len == 0) return null;
        return try self.rowToExecution(result.rows[0]);
    }

    pub fn listExecutions(self: *Self, workflow_id: []const u8, limit: u32) ![]ExecutionRecord {
        const table = try self.executionTableName();
        defer self.allocator.free(table);
        const sql = try std.fmt.allocPrint(self.allocator, "SELECT id, workflow_id, status, started_at, completed_at, result_json FROM \"{s}\" WHERE workflow_id = '{s}' ORDER BY started_at DESC LIMIT {d}", .{ table, workflow_id, limit });
        defer self.allocator.free(sql);

        var result = try self.runQuery(sql);
        defer result.deinit();

        var executions = std.ArrayList(ExecutionRecord){};
        errdefer {
            for (executions.items) |*item| item.deinit(self.allocator);
            executions.deinit();
        }

        for (result.rows) |row| {
            try executions.append(self.allocator, try self.rowToExecution(row));
        }

        return executions.toOwnedSlice(self.allocator);
    }

    fn rowToExecution(self: *Self, row: HanaRow) !ExecutionRecord {
        return .{
            .id = try self.extractText(row, "id"),
            .workflow_id = try self.extractText(row, "workflow_id"),
            .status = try self.extractText(row, "status"),
            .started_at = try self.extractInt(row, "started_at"),
            .completed_at = try self.extractOptionalInt(row, "completed_at"),
            .result_json = try self.extractOptionalText(row, "result_json"),
        };
    }

    pub fn saveWorkflowVersion(self: *Self, workflow_id: []const u8, version: u32, data: []const u8) !void {
        const table = try self.versionsTableName();
        defer self.allocator.free(table);

        const delete_sql = try std.fmt.allocPrint(self.allocator, "DELETE FROM \"{s}\" WHERE workflow_id = '{s}' AND version = {d}", .{ table, workflow_id, version });
        defer self.allocator.free(delete_sql);
        try self.runExecute(delete_sql);

        const insert_sql = try std.fmt.allocPrint(self.allocator, "INSERT INTO \"{s}\" (workflow_id, version, data, created_at) VALUES ('{s}', {d}, '{s}', {d})", .{ table, workflow_id, version, data, std.time.timestamp() });
        defer self.allocator.free(insert_sql);
        try self.runExecute(insert_sql);
    }

    pub fn getWorkflowVersion(self: *Self, workflow_id: []const u8, version: u32) !?[]const u8 {
        const table = try self.versionsTableName();
        defer self.allocator.free(table);
        const sql = try std.fmt.allocPrint(self.allocator, "SELECT data FROM \"{s}\" WHERE workflow_id = '{s}' AND version = {d}", .{ table, workflow_id, version });
        defer self.allocator.free(sql);

        var result = try self.runQuery(sql);
        defer result.deinit();
        if (result.rows.len == 0) return null;

        if (result.rows[0].getValue("data")) |value| {
            if (value.asString()) |s| {
                return try self.allocator.dupe(u8, s);
            }
            if (value == .null_value) return null;
            return error.InvalidField;
        }
        return null;
    }

    pub fn getConfig(self: *const Self) HanaConfig {
        return self.config;
    }

    pub fn getTablePrefix(self: *const Self) []const u8 {
        return self.table_prefix;
    }

// ===== TESTS =====

fn testConfig() HanaConfig {
    return .{ .host = "test.hana.ondemand.com", .port = 443, .user = "DBADMIN", .password = "secret", .database = "WORKFLOW" };
}

test "ExecutionStatus - toString and fromString" {
    try std.testing.expectEqualStrings("running", ExecutionStatus.running.toString());
    try std.testing.expectEqual(ExecutionStatus.completed, try ExecutionStatus.fromString("completed"));
    try std.testing.expectError(error.InvalidStatus, ExecutionStatus.fromString("invalid"));
}

test "WorkflowSummary - clone and deinit" {
    const allocator = std.testing.allocator;
    var summary = WorkflowSummary{
        .id = try allocator.dupe(u8, "wf-123"),
        .name = try allocator.dupe(u8, "Test Workflow"),
        .status = try allocator.dupe(u8, "active"),
        .version = 1,
        .updated_at = 1700000000,
    };
    defer summary.deinit(allocator);
    var cloned = try summary.clone(allocator);
    defer cloned.deinit(allocator);
    try std.testing.expectEqualStrings("wf-123", cloned.id);
    try std.testing.expectEqualStrings("Test Workflow", cloned.name);
}

test "ExecutionRecord - creation and status" {
    const allocator = std.testing.allocator;
    var record = try ExecutionRecord.init(allocator, "exec-123", "wf-456", "pending");
    defer record.deinit(allocator);
    try std.testing.expectEqualStrings("exec-123", record.id);
    try std.testing.expectEqualStrings("pending", record.status);
    try std.testing.expect(record.completed_at == null);
}

test "WorkflowRecord - creation and clone" {
    const allocator = std.testing.allocator;
    var record = try WorkflowRecord.init(allocator, "wf-001", "Order Processing", "{\"nodes\":[]}", "tenant-abc");
    defer record.deinit(allocator);
    try std.testing.expectEqualStrings("wf-001", record.id);
    try std.testing.expectEqualStrings("Order Processing", record.name);
    try std.testing.expectEqual(@as(u32, 1), record.version);
    try std.testing.expectEqualStrings("active", record.status);
    var cloned = try record.clone(allocator);
    defer cloned.deinit(allocator);
    try std.testing.expectEqualStrings(record.id, cloned.id);
}

test "generateUuid - format validation" {
    const allocator = std.testing.allocator;
    const uuid = try generateUuid(allocator);
    defer allocator.free(uuid);
    try std.testing.expectEqual(@as(usize, 36), uuid.len);
    try std.testing.expectEqual(@as(u8, '-'), uuid[8]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[13]);
    try std.testing.expectEqual(@as(u8, '4'), uuid[14]);
}

test "HanaWorkflowStore - initialization" {
    const allocator = std.testing.allocator;
    const store = try HanaWorkflowStore.init(allocator, testConfig(), "nworkflow");
    defer store.deinit();
    try std.testing.expectEqualStrings("nworkflow", store.getTablePrefix());
    try std.testing.expectEqualStrings("test.hana.ondemand.com", store.getConfig().host);
}

test "HanaWorkflowStore - workflow CRUD" {
    const allocator = std.testing.allocator;
    const store = try HanaWorkflowStore.init(allocator, testConfig(), "test_wf");
    defer store.deinit();

    var workflow = try WorkflowRecord.init(allocator, "wf-crud-001", "CRUD Test", "{}", "tenant-1");
    defer workflow.deinit(allocator);
    try store.saveWorkflow(workflow);

    var loaded = (try store.loadWorkflow("wf-crud-001")).?;
    defer loaded.deinit(allocator);
    try std.testing.expectEqualStrings("CRUD Test", loaded.name);

    try store.deleteWorkflow("wf-crud-001");
    const after_delete = try store.loadWorkflow("wf-crud-001");
    try std.testing.expect(after_delete == null);
}

test "HanaWorkflowStore - execution operations" {
    const allocator = std.testing.allocator;
    const store = try HanaWorkflowStore.init(allocator, testConfig(), "test_exec");
    defer store.deinit();

    var exec = try ExecutionRecord.init(allocator, "exec-001", "wf-001", "pending");
    defer exec.deinit(allocator);
    try store.saveExecution(exec);

    var retrieved = (try store.getExecution("exec-001")).?;
    defer retrieved.deinit(allocator);
    try std.testing.expectEqualStrings("wf-001", retrieved.workflow_id);
    try std.testing.expectEqualStrings("pending", retrieved.status);
}

test "HanaWorkflowStore - versioning" {
    const allocator = std.testing.allocator;
    const store = try HanaWorkflowStore.init(allocator, testConfig(), "test_ver");
    defer store.deinit();

    try store.saveWorkflowVersion("wf-ver-001", 1, "version 1 data");
    try store.saveWorkflowVersion("wf-ver-001", 2, "version 2 data");

    const v1 = (try store.getWorkflowVersion("wf-ver-001", 1)).?;
    defer allocator.free(v1);
    try std.testing.expectEqualStrings("version 1 data", v1);

    const v2 = (try store.getWorkflowVersion("wf-ver-001", 2)).?;
    defer allocator.free(v2);
    try std.testing.expectEqualStrings("version 2 data", v2);

    const v3 = try store.getWorkflowVersion("wf-ver-001", 3);
    try std.testing.expect(v3 == null);
}

test "HanaWorkflowStore - list workflows with tenant filter" {
    const allocator = std.testing.allocator;
    const store = try HanaWorkflowStore.init(allocator, testConfig(), "test_list");
    defer store.deinit();

    var wf1 = try WorkflowRecord.init(allocator, "wf-list-1", "Workflow A", "{}", "tenant-a");
    defer wf1.deinit(allocator);
    try store.saveWorkflow(wf1);

    var wf2 = try WorkflowRecord.init(allocator, "wf-list-2", "Workflow B", "{}", "tenant-b");
    defer wf2.deinit(allocator);
    try store.saveWorkflow(wf2);

    const all = try store.listWorkflows(null, 10, 0);
    defer {
        for (all) |*item| { var m = item.*; m.deinit(allocator); }
        allocator.free(all);
    }
    try std.testing.expect(all.len >= 2);

    const tenant_a = try store.listWorkflows("tenant-a", 10, 0);
    defer {
        for (tenant_a) |*item| { var m = item.*; m.deinit(allocator); }
        allocator.free(tenant_a);
    }
    try std.testing.expectEqual(@as(usize, 1), tenant_a.len);
    try std.testing.expectEqualStrings("Workflow A", tenant_a[0].name);
}

test "SQL schema constants - HANA DDL syntax" {
    try std.testing.expect(CREATE_WORKFLOWS_TABLE.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_WORKFLOWS_TABLE, "COLUMN TABLE") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_WORKFLOWS_TABLE, "NVARCHAR") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_WORKFLOWS_TABLE, "NCLOB") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_EXECUTIONS_TABLE, "FOREIGN KEY") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_VERSIONS_TABLE, "VARBINARY") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_INDEXES, "CREATE INDEX") != null);
}

}
;
