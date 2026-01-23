const std = @import("std");
const client = @import("client.zig");

/// Query builder for constructing SQL queries with dialect-specific optimizations
pub const QueryBuilder = struct {
    allocator: std.mem.Allocator,
    dialect: client.Dialect,
    
    // Query components
    select_fields: std.ArrayList([]const u8),
    from_table: ?[]const u8,
    joins: std.ArrayList(JoinClause),
    where_conditions: std.ArrayList(Condition),
    group_by: std.ArrayList([]const u8),
    having: ?Condition,
    order_by: std.ArrayList(OrderBy),
    limit_value: ?u32,
    offset_value: ?u32,
    cte_clauses: std.ArrayList(CTE),
    
    pub fn init(allocator: std.mem.Allocator, dialect: client.Dialect) QueryBuilder {
        return QueryBuilder{
            .allocator = allocator,
            .dialect = dialect,
            .select_fields = std.ArrayList([]const u8).init(allocator),
            .from_table = null,
            .joins = std.ArrayList(JoinClause).init(allocator),
            .where_conditions = std.ArrayList(Condition).init(allocator),
            .group_by = std.ArrayList([]const u8).init(allocator),
            .having = null,
            .order_by = std.ArrayList(OrderBy).init(allocator),
            .limit_value = null,
            .offset_value = null,
            .cte_clauses = std.ArrayList(CTE).init(allocator),
        };
    }
    
    pub fn deinit(self: *QueryBuilder) void {
        self.select_fields.deinit();
        self.joins.deinit();
        self.where_conditions.deinit();
        self.group_by.deinit();
        self.order_by.deinit();
        self.cte_clauses.deinit();
    }
    
    /// Add SELECT fields
    pub fn select(self: *QueryBuilder, fields: []const []const u8) !*QueryBuilder {
        for (fields) |field| {
            try self.select_fields.append(field);
        }
        return self;
    }
    
    /// Set FROM table
    pub fn from(self: *QueryBuilder, table: []const u8) *QueryBuilder {
        self.from_table = table;
        return self;
    }
    
    /// Add JOIN clause
    pub fn join(
        self: *QueryBuilder,
        join_type: JoinType,
        table: []const u8,
        on_condition: []const u8,
    ) !*QueryBuilder {
        try self.joins.append(.{
            .join_type = join_type,
            .table = table,
            .on_condition = on_condition,
        });
        return self;
    }
    
    /// Add WHERE condition
    pub fn where(self: *QueryBuilder, condition: Condition) !*QueryBuilder {
        try self.where_conditions.append(condition);
        return self;
    }
    
    /// Add GROUP BY clause
    pub fn groupBy(self: *QueryBuilder, fields: []const []const u8) !*QueryBuilder {
        for (fields) |field| {
            try self.group_by.append(field);
        }
        return self;
    }
    
    /// Add HAVING clause
    pub fn having(self: *QueryBuilder, condition: Condition) *QueryBuilder {
        self.having = condition;
        return self;
    }
    
    /// Add ORDER BY clause
    pub fn orderBy(self: *QueryBuilder, field: []const u8, direction: OrderDirection) !*QueryBuilder {
        try self.order_by.append(.{
            .field = field,
            .direction = direction,
        });
        return self;
    }
    
    /// Set LIMIT
    pub fn limit(self: *QueryBuilder, value: u32) *QueryBuilder {
        self.limit_value = value;
        return self;
    }
    
    /// Set OFFSET
    pub fn offset(self: *QueryBuilder, value: u32) *QueryBuilder {
        self.offset_value = value;
        return self;
    }
    
    /// Add Common Table Expression (CTE)
    pub fn withCTE(self: *QueryBuilder, name: []const u8, query: []const u8) !*QueryBuilder {
        try self.cte_clauses.append(.{
            .name = name,
            .query = query,
        });
        return self;
    }
    
    /// Build the final SQL query
    pub fn build(self: *QueryBuilder) ![]const u8 {
        return switch (self.dialect) {
            .PostgreSQL => try self.buildPostgreSQL(),
            .HANA => try self.buildHANA(),
            .SQLite => try self.buildSQLite(),
        };
    }
    
    /// Build PostgreSQL-optimized query
    fn buildPostgreSQL(self: *QueryBuilder) ![]const u8 {
        var query = std.ArrayList(u8).init(self.allocator);
        const writer = query.writer();
        
        // CTEs
        if (self.cte_clauses.items.len > 0) {
            try writer.writeAll("WITH ");
            for (self.cte_clauses.items, 0..) |cte, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s} AS ({s})", .{ cte.name, cte.query });
            }
            try writer.writeAll(" ");
        }
        
        // SELECT
        try writer.writeAll("SELECT ");
        if (self.select_fields.items.len == 0) {
            try writer.writeAll("*");
        } else {
            for (self.select_fields.items, 0..) |field, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(field);
            }
        }
        
        // FROM
        if (self.from_table) |table| {
            try writer.print(" FROM {s}", .{table});
        }
        
        // JOINs
        for (self.joins.items) |j| {
            try writer.print(" {s} JOIN {s} ON {s}", .{
                @tagName(j.join_type),
                j.table,
                j.on_condition,
            });
        }
        
        // WHERE
        if (self.where_conditions.items.len > 0) {
            try writer.writeAll(" WHERE ");
            for (self.where_conditions.items, 0..) |cond, i| {
                if (i > 0) try writer.writeAll(" AND ");
                try writer.writeAll(cond.expression);
            }
        }
        
        // GROUP BY
        if (self.group_by.items.len > 0) {
            try writer.writeAll(" GROUP BY ");
            for (self.group_by.items, 0..) |field, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(field);
            }
        }
        
        // HAVING
        if (self.having) |h| {
            try writer.print(" HAVING {s}", .{h.expression});
        }
        
        // ORDER BY
        if (self.order_by.items.len > 0) {
            try writer.writeAll(" ORDER BY ");
            for (self.order_by.items, 0..) |ord, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s} {s}", .{
                    ord.field,
                    @tagName(ord.direction),
                });
            }
        }
        
        // LIMIT/OFFSET (PostgreSQL syntax)
        if (self.limit_value) |lim| {
            try writer.print(" LIMIT {d}", .{lim});
        }
        if (self.offset_value) |off| {
            try writer.print(" OFFSET {d}", .{off});
        }
        
        return query.toOwnedSlice();
    }
    
    /// Build SAP HANA-optimized query
    fn buildHANA(self: *QueryBuilder) ![]const u8 {
        var query = std.ArrayList(u8).init(self.allocator);
        const writer = query.writer();
        
        // CTEs
        if (self.cte_clauses.items.len > 0) {
            try writer.writeAll("WITH ");
            for (self.cte_clauses.items, 0..) |cte, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s} AS ({s})", .{ cte.name, cte.query });
            }
            try writer.writeAll(" ");
        }
        
        // SELECT with HANA hints
        try writer.writeAll("SELECT ");
        
        // Add column store hint for better performance
        if (self.from_table != null) {
            try writer.writeAll("/*+ USE_CS */ ");
        }
        
        if (self.select_fields.items.len == 0) {
            try writer.writeAll("*");
        } else {
            for (self.select_fields.items, 0..) |field, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(field);
            }
        }
        
        // FROM
        if (self.from_table) |table| {
            try writer.print(" FROM {s}", .{table});
        }
        
        // JOINs
        for (self.joins.items) |j| {
            try writer.print(" {s} JOIN {s} ON {s}", .{
                @tagName(j.join_type),
                j.table,
                j.on_condition,
            });
        }
        
        // WHERE
        if (self.where_conditions.items.len > 0) {
            try writer.writeAll(" WHERE ");
            for (self.where_conditions.items, 0..) |cond, i| {
                if (i > 0) try writer.writeAll(" AND ");
                try writer.writeAll(cond.expression);
            }
        }
        
        // GROUP BY
        if (self.group_by.items.len > 0) {
            try writer.writeAll(" GROUP BY ");
            for (self.group_by.items, 0..) |field, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(field);
            }
        }
        
        // HAVING
        if (self.having) |h| {
            try writer.print(" HAVING {s}", .{h.expression});
        }
        
        // ORDER BY
        if (self.order_by.items.len > 0) {
            try writer.writeAll(" ORDER BY ");
            for (self.order_by.items, 0..) |ord, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s} {s}", .{
                    ord.field,
                    @tagName(ord.direction),
                });
            }
        }
        
        // LIMIT/OFFSET (HANA syntax)
        if (self.limit_value) |lim| {
            try writer.print(" LIMIT {d}", .{lim});
        }
        if (self.offset_value) |off| {
            try writer.print(" OFFSET {d}", .{off});
        }
        
        return query.toOwnedSlice();
    }
    
    /// Build SQLite-optimized query
    fn buildSQLite(self: *QueryBuilder) ![]const u8 {
        var query = std.ArrayList(u8).init(self.allocator);
        const writer = query.writer();
        
        // CTEs
        if (self.cte_clauses.items.len > 0) {
            try writer.writeAll("WITH ");
            for (self.cte_clauses.items, 0..) |cte, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s} AS ({s})", .{ cte.name, cte.query });
            }
            try writer.writeAll(" ");
        }
        
        // SELECT
        try writer.writeAll("SELECT ");
        if (self.select_fields.items.len == 0) {
            try writer.writeAll("*");
        } else {
            for (self.select_fields.items, 0..) |field, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(field);
            }
        }
        
        // FROM
        if (self.from_table) |table| {
            try writer.print(" FROM {s}", .{table});
        }
        
        // JOINs
        for (self.joins.items) |j| {
            try writer.print(" {s} JOIN {s} ON {s}", .{
                @tagName(j.join_type),
                j.table,
                j.on_condition,
            });
        }
        
        // WHERE
        if (self.where_conditions.items.len > 0) {
            try writer.writeAll(" WHERE ");
            for (self.where_conditions.items, 0..) |cond, i| {
                if (i > 0) try writer.writeAll(" AND ");
                try writer.writeAll(cond.expression);
            }
        }
        
        // GROUP BY
        if (self.group_by.items.len > 0) {
            try writer.writeAll(" GROUP BY ");
            for (self.group_by.items, 0..) |field, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(field);
            }
        }
        
        // HAVING
        if (self.having) |h| {
            try writer.print(" HAVING {s}", .{h.expression});
        }
        
        // ORDER BY
        if (self.order_by.items.len > 0) {
            try writer.writeAll(" ORDER BY ");
            for (self.order_by.items, 0..) |ord, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s} {s}", .{
                    ord.field,
                    @tagName(ord.direction),
                });
            }
        }
        
        // LIMIT/OFFSET (SQLite syntax)
        if (self.limit_value) |lim| {
            try writer.print(" LIMIT {d}", .{lim});
        }
        if (self.offset_value) |off| {
            try writer.print(" OFFSET {d}", .{off});
        }
        
        return query.toOwnedSlice();
    }
};

/// JOIN types
pub const JoinType = enum {
    INNER,
    LEFT,
    RIGHT,
    FULL,
    CROSS,
};

/// JOIN clause
pub const JoinClause = struct {
    join_type: JoinType,
    table: []const u8,
    on_condition: []const u8,
};

/// WHERE/HAVING condition
pub const Condition = struct {
    expression: []const u8,
};

/// ORDER BY direction
pub const OrderDirection = enum {
    ASC,
    DESC,
};

/// ORDER BY clause
pub const OrderBy = struct {
    field: []const u8,
    direction: OrderDirection,
};

/// Common Table Expression
pub const CTE = struct {
    name: []const u8,
    query: []const u8,
};

// ============================================================================
// Unit Tests
// ============================================================================

test "QueryBuilder - basic SELECT" {
    var qb = QueryBuilder.init(std.testing.allocator, .PostgreSQL);
    defer qb.deinit();
    
    const fields = [_][]const u8{ "id", "name" };
    _ = try qb.select(&fields);
    _ = qb.from("users");
    
    const sql = try qb.build();
    defer std.testing.allocator.free(sql);
    
    try std.testing.expect(std.mem.indexOf(u8, sql, "SELECT id, name FROM users") != null);
}

test "QueryBuilder - with WHERE" {
    var qb = QueryBuilder.init(std.testing.allocator, .PostgreSQL);
    defer qb.deinit();
    
    const fields = [_][]const u8{"*"};
    _ = try qb.select(&fields);
    _ = qb.from("users");
    _ = try qb.where(.{ .expression = "age > 18" });
    
    const sql = try qb.build();
    defer std.testing.allocator.free(sql);
    
    try std.testing.expect(std.mem.indexOf(u8, sql, "WHERE age > 18") != null);
}

test "QueryBuilder - with JOIN" {
    var qb = QueryBuilder.init(std.testing.allocator, .PostgreSQL);
    defer qb.deinit();
    
    const fields = [_][]const u8{"*"};
    _ = try qb.select(&fields);
    _ = qb.from("users");
    _ = try qb.join(.INNER, "orders", "users.id = orders.user_id");
    
    const sql = try qb.build();
    defer std.testing.allocator.free(sql);
    
    try std.testing.expect(std.mem.indexOf(u8, sql, "INNER JOIN orders") != null);
}

test "QueryBuilder - with LIMIT and OFFSET" {
    var qb = QueryBuilder.init(std.testing.allocator, .PostgreSQL);
    defer qb.deinit();
    
    const fields = [_][]const u8{"*"};
    _ = try qb.select(&fields);
    _ = qb.from("users");
    _ = qb.limit(10);
    _ = qb.offset(20);
    
    const sql = try qb.build();
    defer std.testing.allocator.free(sql);
    
    try std.testing.expect(std.mem.indexOf(u8, sql, "LIMIT 10") != null);
    try std.testing.expect(std.mem.indexOf(u8, sql, "OFFSET 20") != null);
}

test "QueryBuilder - HANA with hint" {
    var qb = QueryBuilder.init(std.testing.allocator, .HANA);
    defer qb.deinit();
    
    const fields = [_][]const u8{"*"};
    _ = try qb.select(&fields);
    _ = qb.from("users");
    
    const sql = try qb.build();
    defer std.testing.allocator.free(sql);
    
    try std.testing.expect(std.mem.indexOf(u8, sql, "/*+ USE_CS */") != null);
}

test "QueryBuilder - with CTE" {
    var qb = QueryBuilder.init(std.testing.allocator, .PostgreSQL);
    defer qb.deinit();
    
    _ = try qb.withCTE("active_users", "SELECT * FROM users WHERE active = true");
    const fields = [_][]const u8{"*"};
    _ = try qb.select(&fields);
    _ = qb.from("active_users");
    
    const sql = try qb.build();
    defer std.testing.allocator.free(sql);
    
    try std.testing.expect(std.mem.indexOf(u8, sql, "WITH active_users AS") != null);
}
