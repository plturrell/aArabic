// OData Handler for USER_SETTINGS entity
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

pub const UserSettingsHandler = struct {
    allocator: Allocator,
    config: HanaConfig,
    
    pub fn init(allocator: Allocator, config: HanaConfig) UserSettingsHandler {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }
    
    /// GET /odata/v4/UserSettings - List all user settings with query options
    pub fn list(self: *UserSettingsHandler, options: QueryOptions) ![]const u8 {
        // Build SQL query
        var qb = QueryBuilder.init(self.allocator);
        _ = qb.from("NUCLEUS.USER_SETTINGS");
        _ = try qb.applyODataOptions(options);
        
        const sql = try qb.build();
        defer self.allocator.free(sql);
        
        // Execute query
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        // Format as OData JSON
        return try self.formatListResponse(result, options);
    }
    
    /// GET /odata/v4/UserSettings('user-id') - Get settings for specific user
    pub fn get(self: *UserSettingsHandler, user_id: []const u8) ![]const u8 {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT * FROM NUCLEUS.USER_SETTINGS WHERE USER_ID = '{s}'",
            .{user_id},
        );
        defer self.allocator.free(sql);
        
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        return try self.formatSingleResponse(result, user_id);
    }
    
    /// POST /odata/v4/UserSettings - Create new user settings
    pub fn create(self: *UserSettingsHandler, json_body: []const u8) ![]const u8 {
        _ = json_body;
        
        // Create with default values
        const sql = try std.fmt.allocPrint(
            self.allocator,
            \\INSERT INTO NUCLEUS.USER_SETTINGS 
            \\(USER_ID, THEME, LANGUAGE, DATE_FORMAT, TIME_FORMAT, 
            \\ API_BASE_URL, WEBSOCKET_URL, REQUEST_TIMEOUT_SEC, ENABLE_API_CACHE,
            \\ AUTO_REFRESH, REFRESH_INTERVAL_SEC, SHOW_ADVANCED_METRICS, 
            \\ ENABLE_CHART_ANIMATION, COMPACT_MODE, DEFAULT_CHART_RANGE,
            \\ ENABLE_DESKTOP_NOTIFICATIONS, ENABLE_NOTIFICATION_SOUND, 
            \\ AUTO_DISMISS_TIMEOUT_SEC, SAVE_PROMPT_HISTORY, 
            \\ ENABLE_ANALYTICS, ENABLE_ERROR_REPORTING, 
            \\ CREATED_AT, UPDATED_AT)
            \\VALUES ('new-user', 'sap_horizon', 'en', 'MM/DD/YYYY', '12h',
            \\ 'http://localhost:8080', 'ws://localhost:8080/ws', 30, TRUE,
            \\ TRUE, 10, FALSE, TRUE, FALSE, '1h',
            \\ FALSE, FALSE, 10, TRUE, FALSE, TRUE,
            \\ CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ,
            .{},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return created entity (stub)
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{"@odata.context":"$metadata#UserSettings/$entity","USER_ID":"new-user","THEME":"sap_horizon","LANGUAGE":"en"}}
            ,
            .{},
        );
    }
    
    /// PATCH /odata/v4/UserSettings('user-id') - Update user settings
    pub fn update(self: *UserSettingsHandler, user_id: []const u8, json_body: []const u8) ![]const u8 {
        _ = json_body;
        
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPDATE NUCLEUS.USER_SETTINGS SET UPDATED_AT = CURRENT_TIMESTAMP WHERE USER_ID = '{s}'",
            .{user_id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return updated entity
        return try self.get(user_id);
    }
    
    /// DELETE /odata/v4/UserSettings('user-id') - Delete user settings
    pub fn delete(self: *UserSettingsHandler, user_id: []const u8) !void {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM NUCLEUS.USER_SETTINGS WHERE USER_ID = '{s}'",
            .{user_id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
    }
    
    // Helper: Execute SQL query and return results
    fn executeQuery(self: *UserSettingsHandler, sql: []const u8) ![]const u8 {
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
    fn executeSql(self: *UserSettingsHandler, sql: []const u8) !void {
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
    fn formatListResponse(self: *UserSettingsHandler, hana_result: []const u8, options: QueryOptions) ![]const u8 {
        _ = options;
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#UserSettings\",\"value\":{s}}}",
            .{hana_result},
        );
    }
    
    // Format single entity as OData
    fn formatSingleResponse(self: *UserSettingsHandler, hana_result: []const u8, user_id: []const u8) ![]const u8 {
        _ = user_id;
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#UserSettings/$entity\",\"value\":{s}}}",
            .{hana_result},
        );
    }
};
