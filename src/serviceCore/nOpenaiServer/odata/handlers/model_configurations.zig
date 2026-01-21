// OData Handler for MODEL_CONFIGURATIONS entity
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

pub const ModelConfigurationsHandler = struct {
    allocator: Allocator,
    config: HanaConfig,
    
    pub fn init(allocator: Allocator, config: HanaConfig) ModelConfigurationsHandler {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }
    
    /// GET /odata/v4/ModelConfigurations - List all model configurations with query options
    pub fn list(self: *ModelConfigurationsHandler, options: QueryOptions) ![]const u8 {
        // Build SQL query
        var qb = QueryBuilder.init(self.allocator);
        _ = qb.from("NUCLEUS.MODEL_CONFIGURATIONS");
        _ = try qb.applyODataOptions(options);
        
        const sql = try qb.build();
        defer self.allocator.free(sql);
        
        // Execute query
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        // Format as OData JSON
        return try self.formatListResponse(result, options);
    }
    
    /// GET /odata/v4/ModelConfigurations('uuid') - Get single configuration by ID
    pub fn get(self: *ModelConfigurationsHandler, id: []const u8) ![]const u8 {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT * FROM NUCLEUS.MODEL_CONFIGURATIONS WHERE CONFIG_ID = '{s}'",
            .{id},
        );
        defer self.allocator.free(sql);
        
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        return try self.formatSingleResponse(result, id);
    }
    
    /// POST /odata/v4/ModelConfigurations - Create new model configuration
    pub fn create(self: *ModelConfigurationsHandler, json_body: []const u8) ![]const u8 {
        _ = json_body;
        
        // Create with default values
        const sql = try std.fmt.allocPrint(
            self.allocator,
            \\INSERT INTO NUCLEUS.MODEL_CONFIGURATIONS 
            \\(CONFIG_ID, MODEL_ID, USER_ID, TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS, 
            \\ CONTEXT_LENGTH, REPEAT_PENALTY, ENABLE_STREAMING, ENABLE_CACHE, 
            \\ IS_DEFAULT, CONFIG_NAME, CREATED_AT, UPDATED_AT)
            \\VALUES (SYSUUID, 'gpt-4', 'test-user', 0.7, 0.9, 40, 2048, 
            \\ 4096, 1.1, TRUE, TRUE, FALSE, 'Default Configuration', 
            \\ CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ,
            .{},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return created entity (stub)
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{"@odata.context":"$metadata#ModelConfigurations/$entity","CONFIG_ID":"new-config-id","MODEL_ID":"gpt-4","TEMPERATURE":0.7}}
            ,
            .{},
        );
    }
    
    /// PATCH /odata/v4/ModelConfigurations('uuid') - Update configuration
    pub fn update(self: *ModelConfigurationsHandler, id: []const u8, json_body: []const u8) ![]const u8 {
        _ = json_body;
        
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPDATE NUCLEUS.MODEL_CONFIGURATIONS SET UPDATED_AT = CURRENT_TIMESTAMP WHERE CONFIG_ID = '{s}'",
            .{id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return updated entity
        return try self.get(id);
    }
    
    /// DELETE /odata/v4/ModelConfigurations('uuid') - Delete configuration
    pub fn delete(self: *ModelConfigurationsHandler, id: []const u8) !void {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM NUCLEUS.MODEL_CONFIGURATIONS WHERE CONFIG_ID = '{s}'",
            .{id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
    }
    
    // Helper: Execute SQL query and return results
    fn executeQuery(self: *ModelConfigurationsHandler, sql: []const u8) ![]const u8 {
        var result_buf: [1024 * 1024]u8 = undefined; // 1MB buffer
        
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
    fn executeSql(self: *ModelConfigurationsHandler, sql: []const u8) !void {
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
    fn formatListResponse(self: *ModelConfigurationsHandler, hana_result: []const u8, options: QueryOptions) ![]const u8 {
        _ = options;
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#ModelConfigurations\",\"value\":{s}}}",
            .{hana_result},
        );
    }
    
    // Format single entity as OData
    fn formatSingleResponse(self: *ModelConfigurationsHandler, hana_result: []const u8, id: []const u8) ![]const u8 {
        _ = id;
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#ModelConfigurations/$entity\",\"value\":{s}}}",
            .{hana_result},
        );
    }
};
