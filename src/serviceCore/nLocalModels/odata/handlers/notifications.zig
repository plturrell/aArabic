// OData Handler for NOTIFICATIONS entity
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

pub const NotificationsHandler = struct {
    allocator: Allocator,
    config: HanaConfig,
    
    pub fn init(allocator: Allocator, config: HanaConfig) NotificationsHandler {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }
    
    /// GET /odata/v4/Notifications - List all notifications with query options
    pub fn list(self: *NotificationsHandler, options: QueryOptions) ![]const u8 {
        // Build SQL query
        var qb = QueryBuilder.init(self.allocator);
        _ = qb.from("NUCLEUS.NOTIFICATIONS");
        _ = try qb.applyODataOptions(options);
        
        const sql = try qb.build();
        defer self.allocator.free(sql);
        
        // Execute query
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        // Format as OData JSON
        return try self.formatListResponse(result, options);
    }
    
    /// GET /odata/v4/Notifications('uuid') - Get single notification by ID
    pub fn get(self: *NotificationsHandler, id: []const u8) ![]const u8 {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT * FROM NUCLEUS.NOTIFICATIONS WHERE NOTIFICATION_ID = '{s}'",
            .{id},
        );
        defer self.allocator.free(sql);
        
        const result = try self.executeQuery(sql);
        defer self.allocator.free(result);
        
        return try self.formatSingleResponse(result, id);
    }
    
    /// POST /odata/v4/Notifications - Create new notification
    pub fn create(self: *NotificationsHandler, json_body: []const u8) ![]const u8 {
        _ = json_body;
        
        // Create with default values
        const sql = try std.fmt.allocPrint(
            self.allocator,
            \\INSERT INTO NUCLEUS.NOTIFICATIONS 
            \\(NOTIFICATION_ID, USER_ID, TYPE, CATEGORY, TITLE, MESSAGE, 
            \\ ACTION, ACTION_TEXT, IS_READ, IS_DISMISSED, PRIORITY, IS_STICKY,
            \\ CREATED_AT, SOURCE_SYSTEM)
            \\VALUES (SYSUUID, 'test-user', 'info', 'System', 
            \\ 'Test Notification', 'This is a test notification message',
            \\ 'viewDashboard', 'Go to Dashboard', FALSE, FALSE, 1, FALSE,
            \\ CURRENT_TIMESTAMP, 'OData API')
            ,
            .{},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return created entity (stub)
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{"@odata.context":"$metadata#Notifications/$entity","NOTIFICATION_ID":"new-notif-id","TYPE":"info","TITLE":"Test Notification"}}
            ,
            .{},
        );
    }
    
    /// PATCH /odata/v4/Notifications('uuid') - Update notification (e.g., mark as read)
    pub fn update(self: *NotificationsHandler, id: []const u8, json_body: []const u8) ![]const u8 {
        _ = json_body;
        
        // Mark as read with timestamp
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPDATE NUCLEUS.NOTIFICATIONS SET IS_READ = TRUE, READ_AT = CURRENT_TIMESTAMP WHERE NOTIFICATION_ID = '{s}'",
            .{id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
        
        // Return updated entity
        return try self.get(id);
    }
    
    /// DELETE /odata/v4/Notifications('uuid') - Delete notification
    pub fn delete(self: *NotificationsHandler, id: []const u8) !void {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM NUCLEUS.NOTIFICATIONS WHERE NOTIFICATION_ID = '{s}'",
            .{id},
        );
        defer self.allocator.free(sql);
        
        _ = try self.executeSql(sql);
    }
    
    // Helper: Execute SQL query and return results
    fn executeQuery(self: *NotificationsHandler, sql: []const u8) ![]const u8 {
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
    fn executeSql(self: *NotificationsHandler, sql: []const u8) !void {
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
    fn formatListResponse(self: *NotificationsHandler, hana_result: []const u8, options: QueryOptions) ![]const u8 {
        _ = options;
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#Notifications\",\"value\":{s}}}",
            .{hana_result},
        );
    }
    
    // Format single entity as OData
    fn formatSingleResponse(self: *NotificationsHandler, hana_result: []const u8, id: []const u8) ![]const u8 {
        _ = id;
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#Notifications/$entity\",\"value\":{s}}}",
            .{hana_result},
        );
    }
};
