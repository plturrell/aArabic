// SQL Query Builder for OData v4
// Translates OData query options to SQL statements

const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;
const QueryOptions = @import("service.zig").QueryOptions;

pub const QueryBuilder = struct {
    allocator: Allocator,
    select_cols: ?[]const u8 = null,
    from_table: ?[]const u8 = null,
    where_clause: ?[]const u8 = null,
    orderby_clause: ?[]const u8 = null,
    limit_value: ?usize = null,
    offset_value: ?usize = null,
    
    pub fn init(allocator: Allocator) QueryBuilder {
        return .{ .allocator = allocator };
    }
    
    pub fn select(self: *QueryBuilder, columns: []const u8) *QueryBuilder {
        self.select_cols = columns;
        return self;
    }
    
    pub fn from(self: *QueryBuilder, table: []const u8) *QueryBuilder {
        self.from_table = table;
        return self;
    }
    
    pub fn where(self: *QueryBuilder, condition: []const u8) *QueryBuilder {
        self.where_clause = condition;
        return self;
    }
    
    pub fn orderBy(self: *QueryBuilder, order: []const u8) *QueryBuilder {
        self.orderby_clause = order;
        return self;
    }
    
    pub fn limit(self: *QueryBuilder, n: usize) *QueryBuilder {
        self.limit_value = n;
        return self;
    }
    
    pub fn offset(self: *QueryBuilder, n: usize) *QueryBuilder {
        self.offset_value = n;
        return self;
    }
    
    /// Build complete SQL statement
    pub fn build(self: *QueryBuilder) ![]const u8 {
        var sql: std.ArrayList(u8) = .{};
        errdefer sql.deinit(self.allocator);
        
        // SELECT clause
        try sql.appendSlice(self.allocator, "SELECT ");
        if (self.select_cols) |cols| {
            try sql.appendSlice(self.allocator, cols);
        } else {
            try sql.appendSlice(self.allocator, "*");
        }
        
        // FROM clause
        try sql.appendSlice(self.allocator, " FROM ");
        if (self.from_table) |table| {
            try sql.appendSlice(self.allocator, table);
        } else {
            return error.MissingFromClause;
        }
        
        // WHERE clause
        if (self.where_clause) |where_clause| {
            try sql.appendSlice(self.allocator, " WHERE ");
            try sql.appendSlice(self.allocator, where_clause);
        }
        
        // ORDER BY clause
        if (self.orderby_clause) |orderby| {
            try sql.appendSlice(self.allocator, " ORDER BY ");
            try sql.appendSlice(self.allocator, orderby);
        }
        
        // LIMIT clause (HANA uses TOP)
        if (self.limit_value) |n| {
            const limit_str = try std.fmt.allocPrint(self.allocator, " LIMIT {d}", .{n});
            defer self.allocator.free(limit_str);
            try sql.appendSlice(self.allocator, limit_str);
        }
        
        // OFFSET clause
        if (self.offset_value) |n| {
            const offset_str = try std.fmt.allocPrint(self.allocator, " OFFSET {d}", .{n});
            defer self.allocator.free(offset_str);
            try sql.appendSlice(self.allocator, offset_str);
        }
        
        return sql.toOwnedSlice(self.allocator);
    }
    
    /// Apply OData query options to builder
    pub fn applyODataOptions(self: *QueryBuilder, options: QueryOptions) !*QueryBuilder {
        if (options.select) |select_opt| {
            const cols = try translateODataSelect(self.allocator, select_opt);
            self.select_cols = cols;
        }
        
        if (options.filter) |filter_opt| {
            const where_condition = try translateODataFilter(self.allocator, filter_opt);
            self.where_clause = where_condition;
        }
        
        if (options.orderby) |orderby_opt| {
            const order = try translateODataOrderBy(self.allocator, orderby_opt);
            self.orderby_clause = order;
        }
        
        if (options.top) |top_opt| {
            self.limit_value = top_opt;
        }
        
        if (options.skip) |skip_opt| {
            self.offset_value = skip_opt;
        }
        
        return self;
    }
};

/// Translate OData $select to SQL column list
/// Example: "prompt_text,rating,created_at" → "prompt_text, rating, created_at"
fn translateODataSelect(allocator: Allocator, select: []const u8) ![]const u8 {
    // Simple case: just replace commas with ", "
    var result: std.ArrayList(u8) = .{};
    errdefer result.deinit(allocator);
    
    var parts = mem.splitSequence(u8, select, ",");
    var first = true;
    while (parts.next()) |part| {
        if (!first) try result.appendSlice(allocator, ", ");
        try result.appendSlice(allocator, mem.trim(u8, part, " "));
        first = false;
    }
    
    return result.toOwnedSlice(allocator);
}

/// Translate OData $filter to SQL WHERE clause
/// Examples:
///   "rating gt 3" → "rating > 3"
///   "model_name eq 'gpt-4'" → "model_name = 'gpt-4'"
///   "rating gt 3 and is_favorite eq true" → "rating > 3 AND is_favorite = true"
fn translateODataFilter(allocator: Allocator, filter: []const u8) ![]const u8 {
    var result: std.ArrayList(u8) = .{};
    errdefer result.deinit(allocator);
    
    // Split by spaces while preserving quoted strings
    var tokens: std.ArrayList([]const u8) = .{};
    defer tokens.deinit(allocator);
    
    var i: usize = 0;
    var token_start: usize = 0;
    var in_quote = false;
    
    while (i < filter.len) : (i += 1) {
        const c = filter[i];
        
        if (c == '\'') {
            in_quote = !in_quote;
        } else if (c == ' ' and !in_quote) {
            if (i > token_start) {
                try tokens.append(allocator, filter[token_start..i]);
            }
            token_start = i + 1;
        }
    }
    
    if (filter.len > token_start) {
        try tokens.append(allocator, filter[token_start..]);
    }
    
    // Translate tokens
    for (tokens.items) |token| {
        if (mem.eql(u8, token, "eq")) {
            try result.appendSlice(allocator, " = ");
        } else if (mem.eql(u8, token, "ne")) {
            try result.appendSlice(allocator, " != ");
        } else if (mem.eql(u8, token, "gt")) {
            try result.appendSlice(allocator, " > ");
        } else if (mem.eql(u8, token, "ge")) {
            try result.appendSlice(allocator, " >= ");
        } else if (mem.eql(u8, token, "lt")) {
            try result.appendSlice(allocator, " < ");
        } else if (mem.eql(u8, token, "le")) {
            try result.appendSlice(allocator, " <= ");
        } else if (mem.eql(u8, token, "and")) {
            try result.appendSlice(allocator, " AND ");
        } else if (mem.eql(u8, token, "or")) {
            try result.appendSlice(allocator, " OR ");
        } else if (mem.eql(u8, token, "not")) {
            try result.appendSlice(allocator, " NOT ");
        } else {
            try result.appendSlice(allocator, token);
        }
    }
    
    return result.toOwnedSlice(allocator);
}

/// Translate OData $orderby to SQL ORDER BY clause
/// Example: "created_at desc" → "created_at DESC"
fn translateODataOrderBy(allocator: Allocator, orderby: []const u8) ![]const u8 {
    var result: std.ArrayList(u8) = .{};
    errdefer result.deinit(allocator);
    
    var parts = mem.splitSequence(u8, orderby, ",");
    var first = true;
    
    while (parts.next()) |part| {
        if (!first) try result.appendSlice(allocator, ", ");
        
        const trimmed = mem.trim(u8, part, " ");
        
        // Check for desc/asc
        if (mem.endsWith(u8, trimmed, " desc")) {
            const col = trimmed[0 .. trimmed.len - 5];
            try result.appendSlice(allocator, col);
            try result.appendSlice(allocator, " DESC");
        } else if (mem.endsWith(u8, trimmed, " asc")) {
            const col = trimmed[0 .. trimmed.len - 4];
            try result.appendSlice(allocator, col);
            try result.appendSlice(allocator, " ASC");
        } else {
            try result.appendSlice(allocator, trimmed);
            try result.appendSlice(allocator, " ASC");
        }
        
        first = false;
    }
    
    return result.toOwnedSlice(allocator);
}

/// Build INSERT statement from JSON payload
pub fn buildInsert(allocator: Allocator, table: []const u8, json: []const u8) ![]const u8 {
    _ = json; // Will parse JSON and extract fields
    
    // Stub for now - will implement JSON parsing
    return try std.fmt.allocPrint(
        allocator,
        "INSERT INTO {s} (PROMPT_TEXT) VALUES ('test')",
        .{table},
    );
}

/// Build UPDATE statement from JSON payload
pub fn buildUpdate(allocator: Allocator, table: []const u8, id: i32, json: []const u8) ![]const u8 {
    _ = json; // Will parse JSON and extract fields
    
    // Stub for now
    return try std.fmt.allocPrint(
        allocator,
        "UPDATE {s} SET UPDATED_AT = CURRENT_TIMESTAMP WHERE PROMPT_ID = {d}",
        .{ table, id },
    );
}

/// Build DELETE statement
pub fn buildDelete(allocator: Allocator, table: []const u8, id: i32) ![]const u8 {
    return try std.fmt.allocPrint(
        allocator,
        "DELETE FROM {s} WHERE PROMPT_ID = {d}",
        .{ table, id },
    );
}
