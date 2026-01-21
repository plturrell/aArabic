// OData Handler for PROMPTS entity
// Implements full CRUD operations via HANA

const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;
const QueryOptions = @import("../service.zig").QueryOptions;
const QueryBuilder = @import("../query_builder.zig").QueryBuilder;

// Import HANA client (will be linked)
extern fn zig_odata_query_sql(
    host: [*:0]const u8,
    port: c_int,
    user: [*:0]const u8,
    password: [*:0]const u8,
    schema: [*:0]const u8,
    sql: [*:0]const u8,
    result_buf: [*]u8,
    result_buf_len: c_int,
) c_int;

extern fn zig_odata_execute_sql(
    host: [*:0]const u8,
    port: c_int,
    user: [*:0]const u8,
    password: [*:0]const u8,
    schema: [*:0]const u8,
    sql: [*:0]const u8,
) c_int;

pub const HanaConfig = struct {
    host: []const u8,
    port: u16,
    user: []const u8,
    password: []const u8,
    schema: []const u8,
};

pub const PromptsHandler = struct {
    allocator: Allocator,
    config: HanaConfig,
    
    pub fn init(allocator: Allocator, config: HanaConfig) PromptsHandler {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }
    
    /// GET /odata/v4/Prompts - List all prompts with query options
    pub fn list(self: *PromptsHandler, options: QueryOptions) ![]const u8 {
        // Build SQL query
        var qb = QueryBuilder.init(self.allocator);
        _ = qb.from("NUCLEUS.PROMPTS");
        _ = try qb.applyODataOptions(options);
        
        const sql = try qb.build();
        defer self.allocator.free(sql);
        
        // Execute query
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        // Format as OData JSON
        return try self.formatListResponse(result, options);
    }
    
    /// GET /odata/v4/Prompts(123) - Get single prompt by ID
    pub fn get(self: *PromptsHandler, id: i32) ![]const u8 {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT * FROM NUCLEUS.PROMPTS WHERE PROMPT_ID = {d}",
            .{id},
        );
        defer self.allocator.free(sql);
        
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        return try self.formatSingleResponse(result, id);
    }
    
    /// POST /odata/v4/Prompts - Create new prompt
    pub fn create(self: *PromptsHandler, json_body: []const u8) ![]const u8 {
        _ = json_body; // TODO: Parse JSON to extract fields
        // For now, stub with a simple INSERT
        const sql = try std.fmt.allocPrint(
            self.allocator,
            \\INSERT INTO NUCLEUS.PROMPTS 
            \\(PROMPT_TEXT, PROMPT_MODE_ID, MODEL_NAME, USER_ID, CREATED_AT)
            \\VALUES ('New prompt', 1, 'test-model', 'anonymous', CURRENT_TIMESTAMP)
            ,
            .{},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return created entity (stub)
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{"@odata.context":"$metadata#Prompts/$entity","PROMPT_ID":1,"PROMPT_TEXT":"New prompt"}}
            ,
            .{},
        );
    }
    
    /// PATCH /odata/v4/Prompts(123) - Update prompt
    pub fn update(self: *PromptsHandler, id: i32, json_body: []const u8) ![]const u8 {
        _ = json_body;
        
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPDATE NUCLEUS.PROMPTS SET UPDATED_AT = CURRENT_TIMESTAMP WHERE PROMPT_ID = {d}",
            .{id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return updated entity
        return try self.get(id);
    }
    
    /// DELETE /odata/v4/Prompts(123) - Delete prompt
    pub fn delete(self: *PromptsHandler, id: i32) !void {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM NUCLEUS.PROMPTS WHERE PROMPT_ID = {d}",
            .{id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
    }
    
    // Helper: Execute SQL query and return results
    fn executeQuery(self: *PromptsHandler, sql: []const u8) ![]const u8 {
        var result_buf: [1024 * 1024]u8 = undefined; // 1MB buffer
        
        // Convert to null-terminated strings for C ABI
        const host_z = try self.allocator.dupeZ(u8, self.config.host);
        defer self.allocator.free(host_z);
        
        const user_z = try self.allocator.dupeZ(u8, self.config.user);
        defer self.allocator.free(user_z);
        
        const password_z = try self.allocator.dupeZ(u8, self.config.password);
        defer self.allocator.free(password_z);
        
        const schema_z = try self.allocator.dupeZ(u8, self.config.schema);
        defer self.allocator.free(schema_z);
        
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);
        
        const result_len = zig_odata_query_sql(
            host_z.ptr,
            @intCast(self.config.port),
            user_z.ptr,
            password_z.ptr,
            schema_z.ptr,
            sql_z.ptr,
            &result_buf,
            result_buf.len,
        );
        
        if (result_len < 0) {
            return error.QueryFailed;
        }
        
        return try self.allocator.dupe(u8, result_buf[0..@intCast(result_len)]);
    }
    
    // Helper: Execute SQL statement (no results)
    fn executeSql(self: *PromptsHandler, sql: []const u8) !void {
        const host_z = try self.allocator.dupeZ(u8, self.config.host);
        defer self.allocator.free(host_z);
        
        const user_z = try self.allocator.dupeZ(u8, self.config.user);
        defer self.allocator.free(user_z);
        
        const password_z = try self.allocator.dupeZ(u8, self.config.password);
        defer self.allocator.free(password_z);
        
        const schema_z = try self.allocator.dupeZ(u8, self.config.schema);
        defer self.allocator.free(schema_z);
        
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);
        
        const result = zig_odata_execute_sql(
            host_z.ptr,
            @intCast(self.config.port),
            user_z.ptr,
            password_z.ptr,
            schema_z.ptr,
            sql_z.ptr,
        );
        
        if (result != 0) {
            return error.ExecutionFailed;
        }
    }
    
    // Format query results as OData collection
    fn formatListResponse(self: *PromptsHandler, hana_result: []const u8, options: QueryOptions) ![]const u8 {
        _ = options;
        
        // For now, wrap HANA result in OData envelope
        // TODO: Parse HANA result and properly format
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#Prompts\",\"value\":{s}}}",
            .{hana_result},
        );
    }
    
    // Format single entity as OData
    fn formatSingleResponse(self: *PromptsHandler, hana_result: []const u8, id: i32) ![]const u8 {
        _ = id;
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#Prompts/$entity\",\"value\":{s}}}",
            .{hana_result},
        );
    }
};
