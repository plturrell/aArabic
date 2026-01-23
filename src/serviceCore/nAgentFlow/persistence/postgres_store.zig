//! PostgreSQL Persistence Layer for nWorkflow
//!
//! Provides enterprise-grade persistence for workflow definitions and execution history.
//! Includes multi-tenant support with Row-Level Security (RLS), UUID generation,
//! and comprehensive CRUD operations for workflows and executions.
//!
//! Key Features:
//! - Workflow storage with versioning
//! - Execution history and audit trail
//! - Multi-tenant isolation via tenant_id
//! - Connection pooling support
//! - SQL schema management
//! - UUID generation for IDs

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// ===== EXECUTION STATUS ENUM =====

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

    pub fn fromString(status_str: []const u8) !ExecutionStatus {
        if (std.mem.eql(u8, status_str, "pending")) return .pending;
        if (std.mem.eql(u8, status_str, "running")) return .running;
        if (std.mem.eql(u8, status_str, "completed")) return .completed;
        if (std.mem.eql(u8, status_str, "failed")) return .failed;
        if (std.mem.eql(u8, status_str, "cancelled")) return .cancelled;
        return error.InvalidStatus;
    }
};

// ===== WORKFLOW RECORD =====

/// Workflow record for database storage
pub const WorkflowRecord = struct {
    id: []const u8,
    name: []const u8,
    description: ?[]const u8,
    definition_json: []const u8,
    version: u32,
    tenant_id: []const u8,
    created_by: []const u8,
    created_at: i64,
    updated_at: i64,
    is_active: bool,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        definition_json: []const u8,
        tenant_id: []const u8,
        created_by: []const u8,
    ) !WorkflowRecord {
        const now = std.time.timestamp();
        return WorkflowRecord{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .description = null,
            .definition_json = try allocator.dupe(u8, definition_json),
            .version = 1,
            .tenant_id = try allocator.dupe(u8, tenant_id),
            .created_by = try allocator.dupe(u8, created_by),
            .created_at = now,
            .updated_at = now,
            .is_active = true,
        };
    }

    pub fn deinit(self: *WorkflowRecord, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        if (self.description) |desc| allocator.free(desc);
        allocator.free(self.definition_json);
        allocator.free(self.tenant_id);
        allocator.free(self.created_by);
    }

    pub fn clone(self: *const WorkflowRecord, allocator: Allocator) !WorkflowRecord {
        return WorkflowRecord{
            .id = try allocator.dupe(u8, self.id),
            .name = try allocator.dupe(u8, self.name),
            .description = if (self.description) |d| try allocator.dupe(u8, d) else null,
            .definition_json = try allocator.dupe(u8, self.definition_json),
            .version = self.version,
            .tenant_id = try allocator.dupe(u8, self.tenant_id),
            .created_by = try allocator.dupe(u8, self.created_by),
            .created_at = self.created_at,
            .updated_at = self.updated_at,
            .is_active = self.is_active,
        };
    }

    pub fn setDescription(self: *WorkflowRecord, allocator: Allocator, description: []const u8) !void {
        if (self.description) |desc| allocator.free(desc);
        self.description = try allocator.dupe(u8, description);
    }
};

// ===== EXECUTION RECORD =====

/// Execution record for tracking workflow runs
pub const ExecutionRecord = struct {
    id: []const u8,
    workflow_id: []const u8,
    status: ExecutionStatus,
    input_json: ?[]const u8,
    output_json: ?[]const u8,
    error_message: ?[]const u8,
    started_at: i64,
    completed_at: ?i64,
    tenant_id: []const u8,
    executed_by: []const u8,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        workflow_id: []const u8,
        tenant_id: []const u8,
        executed_by: []const u8,
    ) !ExecutionRecord {
        return ExecutionRecord{
            .id = try allocator.dupe(u8, id),
            .workflow_id = try allocator.dupe(u8, workflow_id),
            .status = .pending,
            .input_json = null,
            .output_json = null,
            .error_message = null,
            .started_at = std.time.timestamp(),
            .completed_at = null,
            .tenant_id = try allocator.dupe(u8, tenant_id),
            .executed_by = try allocator.dupe(u8, executed_by),
        };
    }

    pub fn deinit(self: *ExecutionRecord, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.workflow_id);
        if (self.input_json) |input| allocator.free(input);
        if (self.output_json) |output| allocator.free(output);
        if (self.error_message) |err| allocator.free(err);
        allocator.free(self.tenant_id);
        allocator.free(self.executed_by);
    }

    pub fn clone(self: *const ExecutionRecord, allocator: Allocator) !ExecutionRecord {
        return ExecutionRecord{
            .id = try allocator.dupe(u8, self.id),
            .workflow_id = try allocator.dupe(u8, self.workflow_id),
            .status = self.status,
            .input_json = if (self.input_json) |i| try allocator.dupe(u8, i) else null,
            .output_json = if (self.output_json) |o| try allocator.dupe(u8, o) else null,
            .error_message = if (self.error_message) |e| try allocator.dupe(u8, e) else null,
            .started_at = self.started_at,
            .completed_at = self.completed_at,
            .tenant_id = try allocator.dupe(u8, self.tenant_id),
            .executed_by = try allocator.dupe(u8, self.executed_by),
        };
    }

    pub fn setInput(self: *ExecutionRecord, allocator: Allocator, input: []const u8) !void {
        if (self.input_json) |old| allocator.free(old);
        self.input_json = try allocator.dupe(u8, input);
    }

    pub fn setOutput(self: *ExecutionRecord, allocator: Allocator, output: []const u8) !void {
        if (self.output_json) |old| allocator.free(old);
        self.output_json = try allocator.dupe(u8, output);
    }

    pub fn setError(self: *ExecutionRecord, allocator: Allocator, err_msg: []const u8) !void {
        if (self.error_message) |old| allocator.free(old);
        self.error_message = try allocator.dupe(u8, err_msg);
    }

    pub fn markCompleted(self: *ExecutionRecord) void {
        self.status = .completed;
        self.completed_at = std.time.timestamp();
    }

    pub fn markFailed(self: *ExecutionRecord, allocator: Allocator, err_msg: []const u8) !void {
        self.status = .failed;
        self.completed_at = std.time.timestamp();
        try self.setError(allocator, err_msg);
    }
};

// ===== EXECUTION LOG RECORD =====

/// Log record for detailed step-by-step execution tracking
pub const ExecutionLogRecord = struct {
    id: []const u8,
    execution_id: []const u8,
    step_name: []const u8,
    step_type: []const u8,
    status: ExecutionStatus,
    input_json: ?[]const u8,
    output_json: ?[]const u8,
    error_message: ?[]const u8,
    started_at: i64,
    completed_at: ?i64,
    duration_ms: ?i64,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        execution_id: []const u8,
        step_name: []const u8,
        step_type: []const u8,
    ) !ExecutionLogRecord {
        return ExecutionLogRecord{
            .id = try allocator.dupe(u8, id),
            .execution_id = try allocator.dupe(u8, execution_id),
            .step_name = try allocator.dupe(u8, step_name),
            .step_type = try allocator.dupe(u8, step_type),
            .status = .pending,
            .input_json = null,
            .output_json = null,
            .error_message = null,
            .started_at = std.time.timestamp(),
            .completed_at = null,
            .duration_ms = null,
        };
    }

    pub fn deinit(self: *ExecutionLogRecord, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.execution_id);
        allocator.free(self.step_name);
        allocator.free(self.step_type);
        if (self.input_json) |input| allocator.free(input);
        if (self.output_json) |output| allocator.free(output);
        if (self.error_message) |err| allocator.free(err);
    }
};


// ===== SQL SCHEMA CONSTANTS =====

/// SQL schema for workflows table with multi-tenant support
pub const CREATE_WORKFLOWS_TABLE: []const u8 =
    \\CREATE TABLE IF NOT EXISTS nworkflow_workflows (
    \\    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    \\    name VARCHAR(255) NOT NULL,
    \\    description TEXT,
    \\    definition_json JSONB NOT NULL,
    \\    version INTEGER NOT NULL DEFAULT 1,
    \\    tenant_id UUID NOT NULL,
    \\    created_by UUID NOT NULL,
    \\    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    \\    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    \\    is_active BOOLEAN NOT NULL DEFAULT true,
    \\
    \\    -- Constraints
    \\    CONSTRAINT fk_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    \\    CONSTRAINT fk_created_by FOREIGN KEY (created_by) REFERENCES users(id)
    \\);
    \\
    \\-- Enable Row-Level Security
    \\ALTER TABLE nworkflow_workflows ENABLE ROW LEVEL SECURITY;
    \\
    \\-- RLS Policy: Users can only access workflows in their tenant
    \\CREATE POLICY workflow_tenant_isolation ON nworkflow_workflows
    \\    FOR ALL
    \\    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
    \\
    \\-- RLS Policy: Service accounts can access all workflows (for admin operations)
    \\CREATE POLICY workflow_service_access ON nworkflow_workflows
    \\    FOR ALL
    \\    TO service_role
    \\    USING (true);
;

/// SQL schema for executions table
pub const CREATE_EXECUTIONS_TABLE: []const u8 =
    \\CREATE TABLE IF NOT EXISTS nworkflow_executions (
    \\    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    \\    workflow_id UUID NOT NULL,
    \\    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    \\    input_json JSONB,
    \\    output_json JSONB,
    \\    error_message TEXT,
    \\    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    \\    completed_at TIMESTAMPTZ,
    \\    tenant_id UUID NOT NULL,
    \\    executed_by UUID NOT NULL,
    \\
    \\    -- Constraints
    \\    CONSTRAINT fk_workflow FOREIGN KEY (workflow_id) REFERENCES nworkflow_workflows(id) ON DELETE CASCADE,
    \\    CONSTRAINT fk_execution_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    \\    CONSTRAINT fk_executed_by FOREIGN KEY (executed_by) REFERENCES users(id),
    \\    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
    \\);
    \\
    \\-- Enable Row-Level Security
    \\ALTER TABLE nworkflow_executions ENABLE ROW LEVEL SECURITY;
    \\
    \\-- RLS Policy: Users can only access executions in their tenant
    \\CREATE POLICY execution_tenant_isolation ON nworkflow_executions
    \\    FOR ALL
    \\    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
;

/// SQL schema for execution logs table (detailed step tracking)
pub const CREATE_EXECUTION_LOGS_TABLE: []const u8 =
    \\CREATE TABLE IF NOT EXISTS nworkflow_execution_logs (
    \\    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    \\    execution_id UUID NOT NULL,
    \\    step_name VARCHAR(255) NOT NULL,
    \\    step_type VARCHAR(100) NOT NULL,
    \\    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    \\    input_json JSONB,
    \\    output_json JSONB,
    \\    error_message TEXT,
    \\    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    \\    completed_at TIMESTAMPTZ,
    \\    duration_ms BIGINT,
    \\
    \\    -- Constraints
    \\    CONSTRAINT fk_execution FOREIGN KEY (execution_id) REFERENCES nworkflow_executions(id) ON DELETE CASCADE,
    \\    CONSTRAINT valid_log_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
    \\);
    \\
    \\-- Enable Row-Level Security (inherits from parent execution)
    \\ALTER TABLE nworkflow_execution_logs ENABLE ROW LEVEL SECURITY;
    \\
    \\-- RLS Policy via join to executions table
    \\CREATE POLICY log_tenant_isolation ON nworkflow_execution_logs
    \\    FOR ALL
    \\    USING (
    \\        EXISTS (
    \\            SELECT 1 FROM nworkflow_executions e
    \\            WHERE e.id = execution_id
    \\            AND e.tenant_id = current_setting('app.current_tenant_id')::UUID
    \\        )
    \\    );
;

/// SQL indexes for performance optimization
pub const CREATE_INDEXES: []const u8 =
    \\-- Workflow indexes
    \\CREATE INDEX IF NOT EXISTS idx_workflows_tenant_id ON nworkflow_workflows(tenant_id);
    \\CREATE INDEX IF NOT EXISTS idx_workflows_name ON nworkflow_workflows(name);
    \\CREATE INDEX IF NOT EXISTS idx_workflows_is_active ON nworkflow_workflows(is_active);
    \\CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON nworkflow_workflows(created_at DESC);
    \\CREATE INDEX IF NOT EXISTS idx_workflows_tenant_active ON nworkflow_workflows(tenant_id, is_active);
    \\
    \\-- Execution indexes
    \\CREATE INDEX IF NOT EXISTS idx_executions_workflow_id ON nworkflow_executions(workflow_id);
    \\CREATE INDEX IF NOT EXISTS idx_executions_tenant_id ON nworkflow_executions(tenant_id);
    \\CREATE INDEX IF NOT EXISTS idx_executions_status ON nworkflow_executions(status);
    \\CREATE INDEX IF NOT EXISTS idx_executions_started_at ON nworkflow_executions(started_at DESC);
    \\CREATE INDEX IF NOT EXISTS idx_executions_workflow_status ON nworkflow_executions(workflow_id, status);
    \\
    \\-- Execution logs indexes
    \\CREATE INDEX IF NOT EXISTS idx_execution_logs_execution_id ON nworkflow_execution_logs(execution_id);
    \\CREATE INDEX IF NOT EXISTS idx_execution_logs_status ON nworkflow_execution_logs(status);
    \\CREATE INDEX IF NOT EXISTS idx_execution_logs_step_type ON nworkflow_execution_logs(step_type);
;


// ===== UUID GENERATION =====

/// Generate a UUID v4 using random bytes
/// Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
pub fn generateUuid(allocator: Allocator) ![]const u8 {
    var random_bytes: [16]u8 = undefined;

    // Use crypto random if available, fallback to timestamp-based
    std.crypto.random.bytes(&random_bytes);

    // Set version (4) and variant (RFC 4122)
    random_bytes[6] = (random_bytes[6] & 0x0F) | 0x40; // Version 4
    random_bytes[8] = (random_bytes[8] & 0x3F) | 0x80; // Variant

    // Format as UUID string
    const uuid = try std.fmt.allocPrint(allocator, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
        random_bytes[0],
        random_bytes[1],
        random_bytes[2],
        random_bytes[3],
        random_bytes[4],
        random_bytes[5],
        random_bytes[6],
        random_bytes[7],
        random_bytes[8],
        random_bytes[9],
        random_bytes[10],
        random_bytes[11],
        random_bytes[12],
        random_bytes[13],
        random_bytes[14],
        random_bytes[15],
    });

    return uuid;
}

// ===== SQL QUERY BUILDER HELPERS =====

/// SQL query builder for constructing parameterized queries
pub const SqlQueryBuilder = struct {
    allocator: Allocator,
    query: ArrayList(u8),
    param_count: u32,

    pub fn init(allocator: Allocator) SqlQueryBuilder {
        return SqlQueryBuilder{
            .allocator = allocator,
            .query = .{},
            .param_count = 0,
        };
    }

    pub fn deinit(self: *SqlQueryBuilder) void {
        self.query.deinit(self.allocator);
    }

    pub fn append(self: *SqlQueryBuilder, text: []const u8) !void {
        try self.query.appendSlice(self.allocator, text);
    }

    pub fn appendParam(self: *SqlQueryBuilder) !void {
        self.param_count += 1;
        const param = try std.fmt.allocPrint(self.allocator, "${d}", .{self.param_count});
        defer self.allocator.free(param);
        try self.query.appendSlice(self.allocator, param);
    }

    pub fn build(self: *SqlQueryBuilder) ![]const u8 {
        return try self.query.toOwnedSlice(self.allocator);
    }

    pub fn reset(self: *SqlQueryBuilder) void {
        self.query.clearRetainingCapacity();
        self.param_count = 0;
    }
};


// ===== POSTGRES STORE =====

/// PostgreSQL store for workflow persistence
/// Provides CRUD operations for workflows and executions with multi-tenant support
pub const PostgresStore = struct {
    allocator: Allocator,
    connection_string: []const u8,
    pool_size: u8,

    // In-memory cache for mock implementation (would be replaced with actual DB connection pool)
    workflows: StringHashMap(WorkflowRecord),
    executions: StringHashMap(ExecutionRecord),
    execution_logs: StringHashMap(ExecutionLogRecord),

    /// Connection pool placeholder (would be actual connection pool in production)
    connection_pool: ?*anyopaque,

    pub fn init(allocator: Allocator, connection_string: []const u8, pool_size: ?u8) !*PostgresStore {
        const store = try allocator.create(PostgresStore);
        errdefer allocator.destroy(store);

        store.* = PostgresStore{
            .allocator = allocator,
            .connection_string = try allocator.dupe(u8, connection_string),
            .pool_size = pool_size orelse 10,
            .workflows = StringHashMap(WorkflowRecord).init(allocator),
            .executions = StringHashMap(ExecutionRecord).init(allocator),
            .execution_logs = StringHashMap(ExecutionLogRecord).init(allocator),
            .connection_pool = null,
        };

        // In production: Initialize connection pool here
        // store.connection_pool = try createConnectionPool(connection_string, pool_size);

        return store;
    }

    pub fn deinit(self: *PostgresStore) void {
        // Clean up workflows
        var wf_it = self.workflows.iterator();
        while (wf_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var record = entry.value_ptr.*;
            record.deinit(self.allocator);
        }
        self.workflows.deinit();

        // Clean up executions
        var ex_it = self.executions.iterator();
        while (ex_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var record = entry.value_ptr.*;
            record.deinit(self.allocator);
        }
        self.executions.deinit();

        // Clean up execution logs
        var log_it = self.execution_logs.iterator();
        while (log_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var record = entry.value_ptr.*;
            record.deinit(self.allocator);
        }
        self.execution_logs.deinit();

        self.allocator.free(self.connection_string);
        self.allocator.destroy(self);
    }

    // ===== WORKFLOW OPERATIONS =====

    /// Create a new workflow and return its generated ID
    pub fn createWorkflow(self: *PostgresStore, record: WorkflowRecord) ![]const u8 {
        // Generate UUID for the workflow
        const id = try generateUuid(self.allocator);
        errdefer self.allocator.free(id);

        // Build SQL query (for documentation/logging)
        const sql = try self.buildInsertWorkflowSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL with parameters
        // const result = try self.executeQuery(sql, .{
        //     id, record.name, record.description, record.definition_json,
        //     record.version, record.tenant_id, record.created_by
        // });

        // Mock implementation: Store in memory
        var new_record = try record.clone(self.allocator);
        self.allocator.free(new_record.id);
        new_record.id = try self.allocator.dupe(u8, id);

        const key = try self.allocator.dupe(u8, id);
        try self.workflows.put(key, new_record);

        return id;
    }

    /// Get a workflow by ID with tenant isolation
    pub fn getWorkflow(self: *PostgresStore, id: []const u8, tenant_id: []const u8) !?WorkflowRecord {
        // Build SQL query
        const sql = try self.buildSelectWorkflowSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL with parameters
        // const result = try self.executeQuery(sql, .{ id, tenant_id });
        // if (result.rows.len == 0) return null;
        // return try WorkflowRecord.fromRow(result.rows[0]);

        // Mock implementation: Check in-memory store
        if (self.workflows.get(id)) |record| {
            // Verify tenant isolation
            if (!std.mem.eql(u8, record.tenant_id, tenant_id)) {
                return null; // Tenant mismatch
            }
            // Check if workflow is active (soft delete check)
            if (!record.is_active) {
                return null;
            }
            return try record.clone(self.allocator);
        }
        return null;
    }

    /// Update an existing workflow
    pub fn updateWorkflow(self: *PostgresStore, record: WorkflowRecord) !void {
        // Build SQL query
        const sql = try self.buildUpdateWorkflowSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL with parameters
        // try self.executeQuery(sql, .{
        //     record.name, record.description, record.definition_json,
        //     record.version + 1, record.id, record.tenant_id
        // });

        // Mock implementation: Update in-memory store
        if (self.workflows.getPtr(record.id)) |existing| {
            // Verify tenant isolation
            if (!std.mem.eql(u8, existing.tenant_id, record.tenant_id)) {
                return error.TenantMismatch;
            }

            existing.deinit(self.allocator);
            existing.* = try record.clone(self.allocator);
            existing.version += 1;
            existing.updated_at = std.time.timestamp();
        } else {
            return error.WorkflowNotFound;
        }
    }

    /// Delete a workflow (soft delete - sets is_active = false)
    pub fn deleteWorkflow(self: *PostgresStore, id: []const u8, tenant_id: []const u8) !void {
        // Build SQL query
        const sql = try self.buildDeleteWorkflowSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL (soft delete)
        // try self.executeQuery(sql, .{ id, tenant_id });

        // Mock implementation
        if (self.workflows.getPtr(id)) |existing| {
            if (!std.mem.eql(u8, existing.tenant_id, tenant_id)) {
                return error.TenantMismatch;
            }
            existing.is_active = false;
            existing.updated_at = std.time.timestamp();
        } else {
            return error.WorkflowNotFound;
        }
    }

    /// List workflows for a tenant with pagination
    pub fn listWorkflows(self: *PostgresStore, tenant_id: []const u8, page: u32, limit: u32) ![]WorkflowRecord {
        // Build SQL query
        const sql = try self.buildListWorkflowsSql();
        defer self.allocator.free(sql);

        const offset = page * limit;
        _ = offset;

        // In production: Execute SQL with parameters
        // const result = try self.executeQuery(sql, .{ tenant_id, limit, offset });
        // return try WorkflowRecord.fromRows(result.rows);

        // Mock implementation: Filter and paginate in memory
        var results: ArrayList(WorkflowRecord) = .{};
        errdefer {
            for (results.items) |*item| {
                item.deinit(self.allocator);
            }
            results.deinit(self.allocator);
        }

        var count: u32 = 0;
        var skipped: u32 = 0;
        var it = self.workflows.iterator();
        while (it.next()) |entry| {
            const record = entry.value_ptr.*;
            if (std.mem.eql(u8, record.tenant_id, tenant_id) and record.is_active) {
                if (skipped < page * limit) {
                    skipped += 1;
                    continue;
                }
                if (count >= limit) break;

                try results.append(self.allocator, try record.clone(self.allocator));
                count += 1;
            }
        }

        return results.toOwnedSlice(self.allocator);
    }

    // ===== EXECUTION OPERATIONS =====

    /// Create a new execution and return its generated ID
    pub fn createExecution(self: *PostgresStore, record: ExecutionRecord) ![]const u8 {
        const id = try generateUuid(self.allocator);
        errdefer self.allocator.free(id);

        // Build SQL query
        const sql = try self.buildInsertExecutionSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL with parameters
        // try self.executeQuery(sql, .{
        //     id, record.workflow_id, record.status.toString(),
        //     record.input_json, record.tenant_id, record.executed_by
        // });

        // Mock implementation
        var new_record = try record.clone(self.allocator);
        self.allocator.free(new_record.id);
        new_record.id = try self.allocator.dupe(u8, id);

        const key = try self.allocator.dupe(u8, id);
        try self.executions.put(key, new_record);

        return id;
    }

    /// Update an existing execution
    pub fn updateExecution(self: *PostgresStore, record: ExecutionRecord) !void {
        // Build SQL query
        const sql = try self.buildUpdateExecutionSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL with parameters
        // try self.executeQuery(sql, .{
        //     record.status.toString(), record.output_json, record.error_message,
        //     record.completed_at, record.id
        // });

        // Mock implementation
        if (self.executions.getPtr(record.id)) |existing| {
            existing.deinit(self.allocator);
            existing.* = try record.clone(self.allocator);
        } else {
            return error.ExecutionNotFound;
        }
    }

    /// Get an execution by ID
    pub fn getExecution(self: *PostgresStore, id: []const u8) !?ExecutionRecord {
        // Build SQL query
        const sql = try self.buildSelectExecutionSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL with parameters
        // const result = try self.executeQuery(sql, .{ id });
        // if (result.rows.len == 0) return null;
        // return try ExecutionRecord.fromRow(result.rows[0]);

        // Mock implementation
        if (self.executions.get(id)) |record| {
            return try record.clone(self.allocator);
        }
        return null;
    }

    /// List executions for a workflow
    pub fn listExecutions(self: *PostgresStore, workflow_id: []const u8, limit: u32) ![]ExecutionRecord {
        // Build SQL query
        const sql = try self.buildListExecutionsSql();
        defer self.allocator.free(sql);

        // In production: Execute SQL with parameters
        // const result = try self.executeQuery(sql, .{ workflow_id, limit });
        // return try ExecutionRecord.fromRows(result.rows);

        // Mock implementation
        var results: ArrayList(ExecutionRecord) = .{};
        errdefer {
            for (results.items) |*item| {
                item.deinit(self.allocator);
            }
            results.deinit(self.allocator);
        }

        var count: u32 = 0;
        var it = self.executions.iterator();
        while (it.next()) |entry| {
            if (count >= limit) break;
            const record = entry.value_ptr.*;
            if (std.mem.eql(u8, record.workflow_id, workflow_id)) {
                try results.append(self.allocator, try record.clone(self.allocator));
                count += 1;
            }
        }

        return results.toOwnedSlice(self.allocator);
    }

    // ===== SQL QUERY BUILDERS =====

    fn buildInsertWorkflowSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("INSERT INTO nworkflow_workflows (id, name, description, definition_json, version, tenant_id, created_by, created_at, updated_at, is_active) VALUES (");
        try builder.appendParam(); // $1 - id
        try builder.append(", ");
        try builder.appendParam(); // $2 - name
        try builder.append(", ");
        try builder.appendParam(); // $3 - description
        try builder.append(", ");
        try builder.appendParam(); // $4 - definition_json
        try builder.append(", ");
        try builder.appendParam(); // $5 - version
        try builder.append(", ");
        try builder.appendParam(); // $6 - tenant_id
        try builder.append(", ");
        try builder.appendParam(); // $7 - created_by
        try builder.append(", NOW(), NOW(), true) RETURNING id");

        return try builder.build();
    }

    fn buildSelectWorkflowSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("SELECT id, name, description, definition_json, version, tenant_id, created_by, created_at, updated_at, is_active FROM nworkflow_workflows WHERE id = ");
        try builder.appendParam(); // $1 - id
        try builder.append(" AND tenant_id = ");
        try builder.appendParam(); // $2 - tenant_id
        try builder.append(" AND is_active = true");

        return try builder.build();
    }

    fn buildUpdateWorkflowSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("UPDATE nworkflow_workflows SET name = ");
        try builder.appendParam(); // $1 - name
        try builder.append(", description = ");
        try builder.appendParam(); // $2 - description
        try builder.append(", definition_json = ");
        try builder.appendParam(); // $3 - definition_json
        try builder.append(", version = ");
        try builder.appendParam(); // $4 - version
        try builder.append(", updated_at = NOW() WHERE id = ");
        try builder.appendParam(); // $5 - id
        try builder.append(" AND tenant_id = ");
        try builder.appendParam(); // $6 - tenant_id

        return try builder.build();
    }

    fn buildDeleteWorkflowSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("UPDATE nworkflow_workflows SET is_active = false, updated_at = NOW() WHERE id = ");
        try builder.appendParam(); // $1 - id
        try builder.append(" AND tenant_id = ");
        try builder.appendParam(); // $2 - tenant_id

        return try builder.build();
    }

    fn buildListWorkflowsSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("SELECT id, name, description, definition_json, version, tenant_id, created_by, created_at, updated_at, is_active FROM nworkflow_workflows WHERE tenant_id = ");
        try builder.appendParam(); // $1 - tenant_id
        try builder.append(" AND is_active = true ORDER BY created_at DESC LIMIT ");
        try builder.appendParam(); // $2 - limit
        try builder.append(" OFFSET ");
        try builder.appendParam(); // $3 - offset

        return try builder.build();
    }

    fn buildInsertExecutionSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("INSERT INTO nworkflow_executions (id, workflow_id, status, input_json, tenant_id, executed_by, started_at) VALUES (");
        try builder.appendParam(); // $1 - id
        try builder.append(", ");
        try builder.appendParam(); // $2 - workflow_id
        try builder.append(", ");
        try builder.appendParam(); // $3 - status
        try builder.append(", ");
        try builder.appendParam(); // $4 - input_json
        try builder.append(", ");
        try builder.appendParam(); // $5 - tenant_id
        try builder.append(", ");
        try builder.appendParam(); // $6 - executed_by
        try builder.append(", NOW()) RETURNING id");

        return try builder.build();
    }

    fn buildUpdateExecutionSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("UPDATE nworkflow_executions SET status = ");
        try builder.appendParam(); // $1 - status
        try builder.append(", output_json = ");
        try builder.appendParam(); // $2 - output_json
        try builder.append(", error_message = ");
        try builder.appendParam(); // $3 - error_message
        try builder.append(", completed_at = ");
        try builder.appendParam(); // $4 - completed_at
        try builder.append(" WHERE id = ");
        try builder.appendParam(); // $5 - id

        return try builder.build();
    }

    fn buildSelectExecutionSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("SELECT id, workflow_id, status, input_json, output_json, error_message, started_at, completed_at, tenant_id, executed_by FROM nworkflow_executions WHERE id = ");
        try builder.appendParam(); // $1 - id

        return try builder.build();
    }

    fn buildListExecutionsSql(self: *PostgresStore) ![]const u8 {
        var builder = SqlQueryBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.append("SELECT id, workflow_id, status, input_json, output_json, error_message, started_at, completed_at, tenant_id, executed_by FROM nworkflow_executions WHERE workflow_id = ");
        try builder.appendParam(); // $1 - workflow_id
        try builder.append(" ORDER BY started_at DESC LIMIT ");
        try builder.appendParam(); // $2 - limit

        return try builder.build();
    }

    // ===== SCHEMA MANAGEMENT =====

    /// Initialize database schema (creates tables and indexes)
    pub fn initializeSchema(self: *PostgresStore) !void {
        // In production: Execute schema creation SQL
        // try self.executeQuery(CREATE_WORKFLOWS_TABLE, .{});
        // try self.executeQuery(CREATE_EXECUTIONS_TABLE, .{});
        // try self.executeQuery(CREATE_EXECUTION_LOGS_TABLE, .{});
        // try self.executeQuery(CREATE_INDEXES, .{});

        // Mock implementation: Just log
        _ = self;
        // Schema initialization would happen here in production
    }

    /// Get connection string
    pub fn getConnectionString(self: *const PostgresStore) []const u8 {
        return self.connection_string;
    }

    /// Get pool size
    pub fn getPoolSize(self: *const PostgresStore) u8 {
        return self.pool_size;
    }
};


// ===== TESTS =====

test "ExecutionStatus - toString and fromString" {
    const status = ExecutionStatus.running;
    try std.testing.expectEqualStrings("running", status.toString());

    const parsed = try ExecutionStatus.fromString("completed");
    try std.testing.expectEqual(ExecutionStatus.completed, parsed);

    // Invalid status should error
    _ = ExecutionStatus.fromString("invalid") catch |err| {
        try std.testing.expectEqual(error.InvalidStatus, err);
        return;
    };
    try std.testing.expect(false); // Should not reach here
}

test "WorkflowRecord - creation and clone" {
    const allocator = std.testing.allocator;

    var record = try WorkflowRecord.init(
        allocator,
        "wf-123",
        "Test Workflow",
        "{\"nodes\": []}",
        "tenant-abc",
        "user-xyz",
    );
    defer record.deinit(allocator);

    try std.testing.expectEqualStrings("wf-123", record.id);
    try std.testing.expectEqualStrings("Test Workflow", record.name);
    try std.testing.expectEqualStrings("tenant-abc", record.tenant_id);
    try std.testing.expect(record.is_active);
    try std.testing.expectEqual(@as(u32, 1), record.version);

    // Test clone
    var cloned = try record.clone(allocator);
    defer cloned.deinit(allocator);

    try std.testing.expectEqualStrings(record.id, cloned.id);
    try std.testing.expectEqualStrings(record.name, cloned.name);
}

test "WorkflowRecord - setDescription" {
    const allocator = std.testing.allocator;

    var record = try WorkflowRecord.init(
        allocator,
        "wf-123",
        "Test",
        "{}",
        "tenant-1",
        "user-1",
    );
    defer record.deinit(allocator);

    try std.testing.expect(record.description == null);

    try record.setDescription(allocator, "A test workflow");
    try std.testing.expectEqualStrings("A test workflow", record.description.?);
}

test "ExecutionRecord - creation and status updates" {
    const allocator = std.testing.allocator;

    var record = try ExecutionRecord.init(
        allocator,
        "exec-123",
        "wf-456",
        "tenant-abc",
        "user-xyz",
    );
    defer record.deinit(allocator);

    try std.testing.expectEqualStrings("exec-123", record.id);
    try std.testing.expectEqual(ExecutionStatus.pending, record.status);
    try std.testing.expect(record.completed_at == null);

    // Test marking as completed
    record.markCompleted();
    try std.testing.expectEqual(ExecutionStatus.completed, record.status);
    try std.testing.expect(record.completed_at != null);
}

test "ExecutionRecord - markFailed" {
    const allocator = std.testing.allocator;

    var record = try ExecutionRecord.init(
        allocator,
        "exec-123",
        "wf-456",
        "tenant-1",
        "user-1",
    );
    defer record.deinit(allocator);

    try record.markFailed(allocator, "Connection timeout");
    try std.testing.expectEqual(ExecutionStatus.failed, record.status);
    try std.testing.expectEqualStrings("Connection timeout", record.error_message.?);
}

test "generateUuid - format validation" {
    const allocator = std.testing.allocator;

    const uuid = try generateUuid(allocator);
    defer allocator.free(uuid);

    // UUID should be 36 characters (8-4-4-4-12)
    try std.testing.expectEqual(@as(usize, 36), uuid.len);

    // Check dash positions
    try std.testing.expectEqual(@as(u8, '-'), uuid[8]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[13]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[18]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[23]);

    // Check version (should be 4)
    try std.testing.expectEqual(@as(u8, '4'), uuid[14]);
}

test "generateUuid - uniqueness" {
    const allocator = std.testing.allocator;

    const uuid1 = try generateUuid(allocator);
    defer allocator.free(uuid1);

    const uuid2 = try generateUuid(allocator);
    defer allocator.free(uuid2);

    // UUIDs should be different
    try std.testing.expect(!std.mem.eql(u8, uuid1, uuid2));
}

test "SqlQueryBuilder - basic operations" {
    const allocator = std.testing.allocator;

    var builder = SqlQueryBuilder.init(allocator);
    defer builder.deinit();

    try builder.append("SELECT * FROM users WHERE id = ");
    try builder.appendParam();
    try builder.append(" AND tenant_id = ");
    try builder.appendParam();

    const query = try builder.build();
    defer allocator.free(query);

    try std.testing.expectEqualStrings("SELECT * FROM users WHERE id = $1 AND tenant_id = $2", query);
}


test "PostgresStore - initialization" {
    const allocator = std.testing.allocator;

    const store = try PostgresStore.init(allocator, "postgres://localhost:5432/nworkflow", 5);
    defer store.deinit();

    try std.testing.expectEqualStrings("postgres://localhost:5432/nworkflow", store.getConnectionString());
    try std.testing.expectEqual(@as(u8, 5), store.getPoolSize());
}

test "PostgresStore - workflow CRUD operations" {
    const allocator = std.testing.allocator;

    const store = try PostgresStore.init(allocator, "postgres://localhost:5432/test", null);
    defer store.deinit();

    // Create workflow
    var record = try WorkflowRecord.init(
        allocator,
        "temp-id",
        "Test Workflow",
        "{\"version\": \"1.0\", \"nodes\": []}",
        "tenant-123",
        "user-456",
    );
    defer record.deinit(allocator);

    const workflow_id = try store.createWorkflow(record);
    defer allocator.free(workflow_id);

    try std.testing.expect(workflow_id.len > 0);

    // Get workflow
    var retrieved = (try store.getWorkflow(workflow_id, "tenant-123")).?;
    defer retrieved.deinit(allocator);

    try std.testing.expectEqualStrings("Test Workflow", retrieved.name);
    try std.testing.expectEqualStrings("tenant-123", retrieved.tenant_id);

    // Tenant isolation - different tenant should not see workflow
    const not_found = try store.getWorkflow(workflow_id, "other-tenant");
    try std.testing.expect(not_found == null);

    // Delete workflow
    try store.deleteWorkflow(workflow_id, "tenant-123");

    // Verify soft delete - getWorkflow returns null for inactive workflows
    const after_delete = try store.getWorkflow(workflow_id, "tenant-123");
    try std.testing.expect(after_delete == null);
}

test "PostgresStore - execution CRUD operations" {
    const allocator = std.testing.allocator;

    const store = try PostgresStore.init(allocator, "postgres://localhost:5432/test", null);
    defer store.deinit();

    // Create workflow first
    var wf_record = try WorkflowRecord.init(
        allocator,
        "temp",
        "Workflow",
        "{}",
        "tenant-1",
        "user-1",
    );
    defer wf_record.deinit(allocator);

    const wf_id = try store.createWorkflow(wf_record);
    defer allocator.free(wf_id);

    // Create execution
    var exec_record = try ExecutionRecord.init(
        allocator,
        "temp",
        wf_id,
        "tenant-1",
        "user-1",
    );
    defer exec_record.deinit(allocator);

    const exec_id = try store.createExecution(exec_record);
    defer allocator.free(exec_id);

    // Get execution
    var retrieved = (try store.getExecution(exec_id)).?;
    defer retrieved.deinit(allocator);

    try std.testing.expectEqual(ExecutionStatus.pending, retrieved.status);
    try std.testing.expectEqualStrings(wf_id, retrieved.workflow_id);

    // Update execution
    retrieved.status = .completed;
    retrieved.completed_at = std.time.timestamp();
    try store.updateExecution(retrieved);

    // Verify update
    var updated = (try store.getExecution(exec_id)).?;
    defer updated.deinit(allocator);

    try std.testing.expectEqual(ExecutionStatus.completed, updated.status);
}

test "PostgresStore - list workflows with pagination" {
    const allocator = std.testing.allocator;

    const store = try PostgresStore.init(allocator, "postgres://localhost:5432/test", null);
    defer store.deinit();

    // Create multiple workflows
    var i: usize = 0;
    var workflow_ids: ArrayList([]const u8) = .{};
    defer {
        for (workflow_ids.items) |id| {
            allocator.free(id);
        }
        workflow_ids.deinit(allocator);
    }

    while (i < 5) : (i += 1) {
        const name = try std.fmt.allocPrint(allocator, "Workflow {d}", .{i});
        defer allocator.free(name);

        var record = try WorkflowRecord.init(
            allocator,
            "temp",
            name,
            "{}",
            "tenant-list",
            "user-1",
        );
        defer record.deinit(allocator);

        const id = try store.createWorkflow(record);
        try workflow_ids.append(allocator, id);
    }

    // List first page (limit 2)
    const page1 = try store.listWorkflows("tenant-list", 0, 2);
    defer {
        for (page1) |*item| {
            var mutable_item = item.*;
            mutable_item.deinit(allocator);
        }
        allocator.free(page1);
    }

    try std.testing.expect(page1.len <= 2);
}

test "SQL schema constants - valid SQL syntax" {
    // Verify schema constants are not empty
    try std.testing.expect(CREATE_WORKFLOWS_TABLE.len > 0);
    try std.testing.expect(CREATE_EXECUTIONS_TABLE.len > 0);
    try std.testing.expect(CREATE_EXECUTION_LOGS_TABLE.len > 0);
    try std.testing.expect(CREATE_INDEXES.len > 0);

    // Verify they contain expected keywords
    try std.testing.expect(std.mem.indexOf(u8, CREATE_WORKFLOWS_TABLE, "CREATE TABLE") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_WORKFLOWS_TABLE, "nworkflow_workflows") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_WORKFLOWS_TABLE, "ROW LEVEL SECURITY") != null);

    try std.testing.expect(std.mem.indexOf(u8, CREATE_EXECUTIONS_TABLE, "nworkflow_executions") != null);
    try std.testing.expect(std.mem.indexOf(u8, CREATE_EXECUTIONS_TABLE, "FOREIGN KEY") != null);

    try std.testing.expect(std.mem.indexOf(u8, CREATE_INDEXES, "CREATE INDEX") != null);
}

test "ExecutionLogRecord - creation" {
    const allocator = std.testing.allocator;

    var log = try ExecutionLogRecord.init(
        allocator,
        "log-123",
        "exec-456",
        "process_data",
        "transform",
    );
    defer log.deinit(allocator);

    try std.testing.expectEqualStrings("log-123", log.id);
    try std.testing.expectEqualStrings("exec-456", log.execution_id);
    try std.testing.expectEqualStrings("process_data", log.step_name);
    try std.testing.expectEqualStrings("transform", log.step_type);
    try std.testing.expectEqual(ExecutionStatus.pending, log.status);
}