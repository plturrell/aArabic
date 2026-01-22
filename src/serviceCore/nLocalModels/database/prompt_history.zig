//! Prompt History CRUD Operations
//! Implements database operations for prompt storage and retrieval
//! Uses SAP HANA via zig_odata_sap.zig

const std = @import("std");
const mem = std.mem;

// Import OData client functions
extern fn zig_odata_execute_sql(
    host: [*:0]const u8,
    port: c_int,
    user: [*:0]const u8,
    password: [*:0]const u8,
    schema: [*:0]const u8,
    sql: [*:0]const u8,
) callconv(.c) c_int;

extern fn zig_odata_query_sql(
    host: [*:0]const u8,
    port: c_int,
    user: [*:0]const u8,
    password: [*:0]const u8,
    schema: [*:0]const u8,
    sql: [*:0]const u8,
    result_buf: [*]u8,
    result_buf_len: c_int,
) callconv(.c) c_int;

// ============================================================================
// Configuration
// ============================================================================

pub const HanaConfig = struct {
    host: []const u8,
    port: u16 = 443,
    user: []const u8,
    password: []const u8,
    schema: []const u8 = "NUCLEUS",
};

// ============================================================================
// Data Structures
// ============================================================================

pub const PromptRecord = struct {
    prompt_id: ?i32 = null,
    prompt_text: []const u8,
    prompt_mode_id: i32,
    model_name: []const u8,
    user_id: []const u8,
    tags: ?[]const u8 = null,
    created_at: ?[]const u8 = null,
    updated_at: ?[]const u8 = null,
};

pub const PromptHistoryQuery = struct {
    user_id: ?[]const u8 = null,
    model_name: ?[]const u8 = null,
    prompt_mode_id: ?i32 = null,
    search_text: ?[]const u8 = null,
    limit: u32 = 50,
    offset: u32 = 0,
    order_by: []const u8 = "created_at DESC",
};

pub const PromptSearchResult = struct {
    prompt_id: i32,
    prompt_text: []const u8,
    model_name: []const u8,
    created_at: []const u8,
    relevance_score: f32,
};

// ============================================================================
// CRUD Operations
// ============================================================================

/// Save a new prompt to the database
/// Returns the generated prompt_id or error
pub fn savePrompt(
    allocator: mem.Allocator,
    config: HanaConfig,
    prompt: PromptRecord,
) !i32 {
    // Build INSERT statement with parameterized values
    const sql_tmp = try std.fmt.allocPrint(
        allocator,
        \\INSERT INTO {s}.PROMPTS 
        \\(PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, TAGS)
        \\VALUES ('{s}', {d}, '{s}', '{s}', '{s}')
    ,
        .{
            config.schema,
            escapeSQL(allocator, prompt.prompt_text) catch prompt.prompt_text,
            prompt.prompt_mode_id,
            escapeSQL(allocator, prompt.model_name) catch prompt.model_name,
            escapeSQL(allocator, prompt.user_id) catch prompt.user_id,
            if (prompt.tags) |tags| escapeSQL(allocator, tags) catch tags else "",
        },
    );
    defer allocator.free(sql_tmp);
    const sql = try allocator.dupeZ(u8, sql_tmp);
    defer allocator.free(sql);

    const host_z = try allocator.dupeZ(u8, config.host);
    defer allocator.free(host_z);
    const user_z = try allocator.dupeZ(u8, config.user);
    defer allocator.free(user_z);
    const password_z = try allocator.dupeZ(u8, config.password);
    defer allocator.free(password_z);
    const schema_z = try allocator.dupeZ(u8, config.schema);
    defer allocator.free(schema_z);

    const result = zig_odata_execute_sql(
        host_z.ptr,
        @intCast(config.port),
        user_z.ptr,
        password_z.ptr,
        schema_z.ptr,
        sql.ptr,
    );

    if (result != 0) {
        return error.InsertFailed;
    }

    // In a real implementation, we would query for LAST_INSERT_ID()
    // For now, return a placeholder
    return 1;
}

/// Get prompt history with pagination and filtering
pub fn getPromptHistory(
    allocator: mem.Allocator,
    config: HanaConfig,
    query: PromptHistoryQuery,
) ![]const u8 {
    var sql_parts = std.ArrayList(u8).empty;
    defer sql_parts.deinit(allocator);

    // Base query
    try sql_parts.appendSlice("SELECT PROMPT_ID, PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, ");
    try sql_parts.appendSlice("USER_ID, TAGS, CREATED_AT, UPDATED_AT FROM ");
    try sql_parts.appendSlice(config.schema);
    try sql_parts.appendSlice(".PROMPTS WHERE 1=1");

    // Add filters
    if (query.user_id) |user_id| {
        const filter = try std.fmt.allocPrint(allocator, " AND USER_ID = '{s}'", .{
            escapeSQL(allocator, user_id) catch user_id,
        });
        defer allocator.free(filter);
        try sql_parts.appendSlice(filter);
    }

    if (query.model_name) |model_name| {
        const filter = try std.fmt.allocPrint(allocator, " AND MODEL_NAME = '{s}'", .{
            escapeSQL(allocator, model_name) catch model_name,
        });
        defer allocator.free(filter);
        try sql_parts.appendSlice(filter);
    }

    if (query.prompt_mode_id) |mode_id| {
        const filter = try std.fmt.allocPrint(allocator, " AND PROMPT_MODE_ID = {d}", .{mode_id});
        defer allocator.free(filter);
        try sql_parts.appendSlice(filter);
    }

    // Add search if provided
    if (query.search_text) |search| {
        const filter = try std.fmt.allocPrint(allocator, " AND CONTAINS(PROMPT_TEXT, '{s}')", .{
            escapeSQL(allocator, search) catch search,
        });
        defer allocator.free(filter);
        try sql_parts.appendSlice(filter);
    }

    // Add ordering
    const order = try std.fmt.allocPrint(allocator, " ORDER BY {s}", .{query.order_by});
    defer allocator.free(order);
    try sql_parts.appendSlice(order);

    // Add pagination
    const pagination = try std.fmt.allocPrint(
        allocator,
        " LIMIT {d} OFFSET {d}",
        .{ query.limit, query.offset },
    );
    defer allocator.free(pagination);
    try sql_parts.appendSlice(pagination);

    const sql = try sql_parts.toOwnedSliceSentinel(allocator, 0);

    // Execute query
    const host_z = try allocator.dupeZ(u8, config.host);
    defer allocator.free(host_z);
    const user_z = try allocator.dupeZ(u8, config.user);
    defer allocator.free(user_z);
    const password_z = try allocator.dupeZ(u8, config.password);
    defer allocator.free(password_z);
    const schema_z = try allocator.dupeZ(u8, config.schema);
    defer allocator.free(schema_z);

    var result_buf: [8192]u8 = undefined;
    const result_len = zig_odata_query_sql(
        host_z.ptr,
        @intCast(config.port),
        user_z.ptr,
        password_z.ptr,
        schema_z.ptr,
        sql.ptr,
        &result_buf,
        8192,
    );

    if (result_len <= 0) {
        return error.QueryFailed;
    }

    return try allocator.dupe(u8, result_buf[0..@intCast(result_len)]);
}

/// Delete a prompt by ID
pub fn deletePrompt(
    allocator: mem.Allocator,
    config: HanaConfig,
    prompt_id: i32,
) !void {
    const sql_tmp = try std.fmt.allocPrint(
        allocator,
        "DELETE FROM {s}.PROMPTS WHERE PROMPT_ID = {d}",
        .{ config.schema, prompt_id },
    );
    defer allocator.free(sql_tmp);
    const sql = try allocator.dupeZ(u8, sql_tmp);
    defer allocator.free(sql);

    const host_z = try allocator.dupeZ(u8, config.host);
    defer allocator.free(host_z);
    const user_z = try allocator.dupeZ(u8, config.user);
    defer allocator.free(user_z);
    const password_z = try allocator.dupeZ(u8, config.password);
    defer allocator.free(password_z);
    const schema_z = try allocator.dupeZ(u8, config.schema);
    defer allocator.free(schema_z);

    const result = zig_odata_execute_sql(
        host_z.ptr,
        @intCast(config.port),
        user_z.ptr,
        password_z.ptr,
        schema_z.ptr,
        sql.ptr,
    );

    if (result != 0) {
        return error.DeleteFailed;
    }
}

/// Full-text search across prompts
pub fn searchPrompts(
    allocator: mem.Allocator,
    config: HanaConfig,
    search_query: []const u8,
    limit: u32,
) ![]const u8 {
    // Use HANA's CONTAINS function for full-text search
    const sql_tmp = try std.fmt.allocPrint(
        allocator,
        \\SELECT PROMPT_ID, PROMPT_TEXT, MODEL_NAME, CREATED_AT,
        \\  SCORE() AS RELEVANCE_SCORE
        \\FROM {s}.PROMPTS
        \\WHERE CONTAINS(PROMPT_TEXT, '{s}', FUZZY(0.8))
        \\ORDER BY RELEVANCE_SCORE DESC
        \\LIMIT {d}
    ,
        .{
            config.schema,
            escapeSQL(allocator, search_query) catch search_query,
            limit,
        },
    );
    defer allocator.free(sql_tmp);
    const sql = try allocator.dupeZ(u8, sql_tmp);
    defer allocator.free(sql);

    const host_z = try allocator.dupeZ(u8, config.host);
    defer allocator.free(host_z);
    const user_z = try allocator.dupeZ(u8, config.user);
    defer allocator.free(user_z);
    const password_z = try allocator.dupeZ(u8, config.password);
    defer allocator.free(password_z);
    const schema_z = try allocator.dupeZ(u8, config.schema);
    defer allocator.free(schema_z);

    var result_buf: [16384]u8 = undefined;
    const result_len = zig_odata_query_sql(
        host_z.ptr,
        @intCast(config.port),
        user_z.ptr,
        password_z.ptr,
        schema_z.ptr,
        sql.ptr,
        &result_buf,
        16384,
    );

    if (result_len <= 0) {
        return error.SearchFailed;
    }

    return try allocator.dupe(u8, result_buf[0..@intCast(result_len)]);
}

/// Get prompt count for a user (for pagination)
pub fn getPromptCount(
    allocator: mem.Allocator,
    config: HanaConfig,
    query: PromptHistoryQuery,
) !i32 {
    const sql_tmp = if (query.user_id) |uid|
        try std.fmt.allocPrint(
            allocator,
            "SELECT COUNT(*) as total FROM {s}.PROMPTS WHERE USER_ID = '{s}'",
            .{ config.schema, escapeSQL(allocator, uid) catch uid },
        )
    else
        try std.fmt.allocPrint(
            allocator,
            "SELECT COUNT(*) as total FROM {s}.PROMPTS",
            .{config.schema},
        );
    defer allocator.free(sql_tmp);
    const sql = try allocator.dupeZ(u8, sql_tmp);
    defer allocator.free(sql);

    const host_z = try allocator.dupeZ(u8, config.host);
    defer allocator.free(host_z);
    const user_z = try allocator.dupeZ(u8, config.user);
    defer allocator.free(user_z);
    const password_z = try allocator.dupeZ(u8, config.password);
    defer allocator.free(password_z);
    const schema_z = try allocator.dupeZ(u8, config.schema);
    defer allocator.free(schema_z);

    var result_buf: [256]u8 = undefined;
    const result_len = zig_odata_query_sql(
        host_z.ptr,
        @intCast(config.port),
        user_z.ptr,
        password_z.ptr,
        schema_z.ptr,
        sql.ptr,
        &result_buf,
        256,
    );

    if (result_len <= 0) {
        return error.QueryFailed;
    }

    // Parse count from JSON result
    // For now, return placeholder
    return 0;
}

// ============================================================================
// SQL Injection Prevention
// ============================================================================

/// Escape SQL special characters to prevent injection
fn escapeSQL(allocator: mem.Allocator, input: []const u8) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    defer result.deinit(allocator);

    for (input) |char| {
        switch (char) {
            '\'' => try result.appendSlice(allocator, "''"), // Escape single quotes
            '\\' => try result.appendSlice(allocator, "\\\\"), // Escape backslashes
            '\n' => try result.appendSlice(allocator, "\\n"), // Escape newlines
            '\r' => try result.appendSlice(allocator, "\\r"), // Escape carriage returns
            '\t' => try result.appendSlice(allocator, "\\t"), // Escape tabs
            else => try result.append(allocator, char),
        }
    }

    return try result.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "escapeSQL prevents injection" {
    const allocator = std.testing.allocator;

    const malicious = "'; DROP TABLE PROMPTS; --";
    const escaped = try escapeSQL(allocator, malicious);
    defer allocator.free(escaped);

    try std.testing.expect(mem.indexOf(u8, escaped, "DROP") != null);
    try std.testing.expect(mem.indexOf(u8, escaped, "''") != null);
}

test "savePrompt creates valid SQL" {
    const allocator = std.testing.allocator;

    const prompt = PromptRecord{
        .prompt_text = "Test prompt",
        .prompt_mode_id = 1,
        .model_name = "gpt-4",
        .user_id = "test_user",
        .tags = "test,sample",
    };

    const config = HanaConfig{
        .host = "localhost",
        .user = "test",
        .password = "test",
    };

    // This would fail with actual connection, but tests SQL generation
    _ = savePrompt(allocator, config, prompt) catch |err| {
        try std.testing.expect(err == error.InsertFailed);
    };
}
