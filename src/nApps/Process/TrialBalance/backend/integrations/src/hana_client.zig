//! ============================================================================
//! SAP HANA Cloud Client
//! Handles communication with SAP HANA Cloud database
//! ============================================================================
//!
//! [CODE:file=hana_client.zig]
//! [CODE:module=integrations]
//! [CODE:language=zig]
//!
//! [CONFIG:requires=HANA_HOST]
//! [CONFIG:requires=HANA_PORT]
//! [CONFIG:requires=HANA_USER]
//! [CONFIG:requires=HANA_PASSWORD]
//!
//! [ODPS:product=acdoca-journal-entries]
//! [ODPS:product=trial-balance-aggregated]
//!
//! [TABLE:reads=ACDOCA]
//! [TABLE:reads=SKA1]
//! [TABLE:reads=TCURR]
//!
//! [RELATION:uses=CODE:destination_client.zig]
//!
//! This client provides HANA Cloud connectivity for reading SAP financial data
//! including ACDOCA journal entries, account master, and exchange rates.

const std = @import("std");

/// HANA Connection Configuration
pub const HanaConfig = struct {
    host: []const u8,
    port: u16 = 443,
    user: []const u8,
    password: []const u8,
    database: []const u8 = "",
    schema: []const u8 = "SAPABAP1",
    encrypt: bool = true,
    validate_certificate: bool = true,
    
    /// Create from environment variables
    pub fn fromEnv(allocator: std.mem.Allocator) !HanaConfig {
        const host = std.process.getEnvVarOwned(allocator, "HANA_HOST") catch return error.MissingHanaHost;
        errdefer allocator.free(host);
        
        const port_str = std.process.getEnvVarOwned(allocator, "HANA_PORT") catch null;
        defer if (port_str) |p| allocator.free(p);
        const port: u16 = if (port_str) |p| std.fmt.parseInt(u16, p, 10) catch 443 else 443;
        
        const user = std.process.getEnvVarOwned(allocator, "HANA_USER") catch return error.MissingHanaUser;
        errdefer allocator.free(user);
        
        const password = std.process.getEnvVarOwned(allocator, "HANA_PASSWORD") catch return error.MissingHanaPassword;
        errdefer allocator.free(password);
        
        const database = std.process.getEnvVarOwned(allocator, "HANA_DATABASE") catch try allocator.dupe(u8, "");
        errdefer allocator.free(database);
        
        const schema = std.process.getEnvVarOwned(allocator, "HANA_SCHEMA") catch try allocator.dupe(u8, "SAPABAP1");
        
        return .{
            .host = host,
            .port = port,
            .user = user,
            .password = password,
            .database = database,
            .schema = schema,
        };
    }
    
    /// Get connection string for HANA
    pub fn getConnectionUrl(self: *const HanaConfig, allocator: std.mem.Allocator) ![]const u8 {
        if (self.database.len > 0) {
            return std.fmt.allocPrint(
                allocator,
                "https://{s}:{d}/sap/bc/sql?databaseName={s}",
                .{ self.host, self.port, self.database }
            );
        }
        return std.fmt.allocPrint(
            allocator,
            "https://{s}:{d}/sap/bc/sql",
            .{ self.host, self.port }
        );
    }
};

/// HANA Query Result Row
pub const QueryRow = struct {
    columns: [][]const u8,
    values: [][]const u8,
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *QueryRow) void {
        for (self.values) |v| self.allocator.free(v);
        self.allocator.free(self.values);
        for (self.columns) |c| self.allocator.free(c);
        self.allocator.free(self.columns);
    }
    
    pub fn get(self: *const QueryRow, column: []const u8) ?[]const u8 {
        for (self.columns, 0..) |col, i| {
            if (std.mem.eql(u8, col, column)) {
                return self.values[i];
            }
        }
        return null;
    }
};

/// HANA Query Result
pub const QueryResult = struct {
    rows: []QueryRow,
    row_count: usize,
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *QueryResult) void {
        for (self.rows) |*row| row.deinit();
        self.allocator.free(self.rows);
    }
};

/// SAP HANA Cloud Client
/// Uses HANA REST API (SQLScript Procedures) or OData
pub const HanaClient = struct {
    allocator: std.mem.Allocator,
    config: HanaConfig,
    http_client: std.http.Client,
    base_url: []const u8,
    auth_header: ?[]const u8,
    
    const Self = @This();
    
    /// Initialize HANA client
    pub fn init(allocator: std.mem.Allocator, config: HanaConfig) !Self {
        const base_url = try std.fmt.allocPrint(
            allocator,
            "https://{s}:{d}",
            .{ config.host, config.port }
        );
        
        // Create Basic Auth header
        const credentials = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ config.user, config.password });
        defer allocator.free(credentials);
        
        var encoder = std.base64.standard.Encoder;
        const encoded_len = std.base64.standard.Encoder.calcSize(credentials.len);
        const encoded = try allocator.alloc(u8, encoded_len);
        _ = std.base64.standard.Encoder.encode(encoded, credentials);
        
        const auth_header = try std.fmt.allocPrint(allocator, "Basic {s}", .{encoded});
        allocator.free(encoded);
        
        return .{
            .allocator = allocator,
            .config = config,
            .http_client = std.http.Client.init(allocator, .{}),
            .base_url = base_url,
            .auth_header = auth_header,
        };
    }
    
    /// Initialize from environment
    pub fn initFromEnv(allocator: std.mem.Allocator) !Self {
        const config = try HanaConfig.fromEnv(allocator);
        return Self.init(allocator, config);
    }
    
    pub fn deinit(self: *Self) void {
        if (self.auth_header) |h| self.allocator.free(h);
        self.allocator.free(self.base_url);
        self.http_client.deinit();
    }
    
    /// Execute SQL query via HANA SQL API
    pub fn executeQuery(self: *Self, sql: []const u8) !QueryResult {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/sap/bc/sql?sap-client=100",
            .{self.base_url}
        );
        defer self.allocator.free(url);
        
        const uri = try std.Uri.parse(url);
        var header_buffer: [8192]u8 = undefined;
        var request = try self.http_client.open(.POST, uri, .{ .server_header_buffer = &header_buffer });
        defer request.deinit();
        
        request.transfer_encoding = .chunked;
        try request.headers.append("Content-Type", "application/sql");
        try request.headers.append("Accept", "application/json");
        if (self.auth_header) |auth| {
            try request.headers.append("Authorization", auth);
        }
        
        try request.send();
        try request.writeAll(sql);
        try request.finish();
        try request.wait();
        
        if (request.response.status != .ok) {
            return error.QueryFailed;
        }
        
        const body = try request.reader().readAllAlloc(self.allocator, 100 * 1024 * 1024);
        defer self.allocator.free(body);
        
        return self.parseQueryResult(body);
    }
    
    /// Parse JSON query result
    fn parseQueryResult(self: *Self, json_body: []const u8) !QueryResult {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_body, .{});
        defer parsed.deinit();
        
        var rows = std.ArrayList(QueryRow).init(self.allocator);
        errdefer rows.deinit();
        
        // Parse column names
        const results = parsed.value.object.get("results") orelse return error.InvalidResponse;
        const columns_json = results.array.items[0].object.get("columns") orelse return error.InvalidResponse;
        
        var columns = std.ArrayList([]const u8).init(self.allocator);
        defer columns.deinit();
        for (columns_json.array.items) |col| {
            const col_name = col.object.get("name").?.string;
            try columns.append(try self.allocator.dupe(u8, col_name));
        }
        
        // Parse rows
        const rows_json = results.array.items[0].object.get("rows") orelse return error.InvalidResponse;
        for (rows_json.array.items) |row_json| {
            var values = std.ArrayList([]const u8).init(self.allocator);
            for (row_json.array.items) |val| {
                const str_val = switch (val) {
                    .string => |s| try self.allocator.dupe(u8, s),
                    .integer => |i| try std.fmt.allocPrint(self.allocator, "{d}", .{i}),
                    .float => |f| try std.fmt.allocPrint(self.allocator, "{d}", .{f}),
                    .null => try self.allocator.dupe(u8, ""),
                    else => try self.allocator.dupe(u8, ""),
                };
                try values.append(str_val);
            }
            
            const columns_owned = try self.allocator.alloc([]const u8, columns.items.len);
            for (columns.items, 0..) |c, i| {
                columns_owned[i] = try self.allocator.dupe(u8, c);
            }
            
            try rows.append(.{
                .columns = columns_owned,
                .values = try values.toOwnedSlice(),
                .allocator = self.allocator,
            });
        }
        
        return .{
            .rows = try rows.toOwnedSlice(),
            .row_count = rows.items.len,
            .allocator = self.allocator,
        };
    }
    
    // ========================================================================
    // Trial Balance Specific Queries
    // ========================================================================
    
    /// Get ACDOCA journal entries for a period
    pub fn getJournalEntries(
        self: *Self,
        company_code: []const u8,
        fiscal_year: []const u8,
        period: []const u8,
    ) !QueryResult {
        const sql = try std.fmt.allocPrint(self.allocator,
            \\SELECT 
            \\    RBUKRS as COMPANY_CODE,
            \\    RACCT as ACCOUNT,
            \\    GJAHR as FISCAL_YEAR,
            \\    POPER as PERIOD,
            \\    RHCUR as LOCAL_CURRENCY,
            \\    HSL as LOCAL_AMOUNT,
            \\    RKCUR as GROUP_CURRENCY,
            \\    KSL as GROUP_AMOUNT,
            \\    DRCRK as DEBIT_CREDIT,
            \\    BELNR as DOCUMENT_NUMBER,
            \\    BUZEI as LINE_ITEM
            \\FROM {s}.ACDOCA
            \\WHERE RBUKRS = '{s}'
            \\  AND GJAHR = '{s}'
            \\  AND POPER = '{s}'
            \\  AND RLDNR = '0L'
            \\ORDER BY RACCT, BELNR, BUZEI
        , .{ self.config.schema, company_code, fiscal_year, period });
        defer self.allocator.free(sql);
        
        return self.executeQuery(sql);
    }
    
    /// Get trial balance summary
    pub fn getTrialBalance(
        self: *Self,
        company_code: []const u8,
        fiscal_year: []const u8,
        period: []const u8,
    ) !QueryResult {
        const sql = try std.fmt.allocPrint(self.allocator,
            \\SELECT 
            \\    RACCT as ACCOUNT,
            \\    SUM(CASE WHEN DRCRK = 'S' THEN HSL ELSE 0 END) as DEBIT_LOCAL,
            \\    SUM(CASE WHEN DRCRK = 'H' THEN HSL ELSE 0 END) as CREDIT_LOCAL,
            \\    SUM(CASE WHEN DRCRK = 'S' THEN KSL ELSE 0 END) as DEBIT_GROUP,
            \\    SUM(CASE WHEN DRCRK = 'H' THEN KSL ELSE 0 END) as CREDIT_GROUP,
            \\    RHCUR as LOCAL_CURRENCY,
            \\    RKCUR as GROUP_CURRENCY
            \\FROM {s}.ACDOCA
            \\WHERE RBUKRS = '{s}'
            \\  AND GJAHR = '{s}'
            \\  AND POPER <= '{s}'
            \\  AND RLDNR = '0L'
            \\GROUP BY RACCT, RHCUR, RKCUR
            \\ORDER BY RACCT
        , .{ self.config.schema, company_code, fiscal_year, period });
        defer self.allocator.free(sql);
        
        return self.executeQuery(sql);
    }
    
    /// Get exchange rates from TCURR
    pub fn getExchangeRates(
        self: *Self,
        from_currency: []const u8,
        to_currency: []const u8,
        date: []const u8,
    ) !QueryResult {
        const sql = try std.fmt.allocPrint(self.allocator,
            \\SELECT 
            \\    FCURR as FROM_CURRENCY,
            \\    TCURR as TO_CURRENCY,
            \\    UKURS as EXCHANGE_RATE,
            \\    FFACT as FROM_FACTOR,
            \\    TFACT as TO_FACTOR,
            \\    GDATU as VALID_FROM
            \\FROM {s}.TCURR
            \\WHERE KURST = 'M'
            \\  AND FCURR = '{s}'
            \\  AND TCURR = '{s}'
            \\  AND GDATU <= '{s}'
            \\ORDER BY GDATU DESC
            \\LIMIT 1
        , .{ self.config.schema, from_currency, to_currency, date });
        defer self.allocator.free(sql);
        
        return self.executeQuery(sql);
    }
    
    /// Get account master data from SKA1
    pub fn getAccountMaster(self: *Self, chart_of_accounts: []const u8) !QueryResult {
        const sql = try std.fmt.allocPrint(self.allocator,
            \\SELECT 
            \\    SAKNR as ACCOUNT,
            \\    TXT50 as DESCRIPTION,
            \\    KTOKS as ACCOUNT_GROUP,
            \\    XBILK as BALANCE_SHEET_FLAG,
            \\    GVTYP as PL_TYPE
            \\FROM {s}.SKA1 S
            \\INNER JOIN {s}.SKAT T ON S.SAKNR = T.SAKNR AND T.SPRAS = 'E'
            \\WHERE S.KTOPL = '{s}'
            \\ORDER BY SAKNR
        , .{ self.config.schema, self.config.schema, chart_of_accounts });
        defer self.allocator.free(sql);
        
        return self.executeQuery(sql);
    }
    
    /// Test connection
    pub fn testConnection(self: *Self) !bool {
        const sql = "SELECT 1 FROM DUMMY";
        const result = self.executeQuery(sql) catch return false;
        defer @constCast(&result).deinit();
        return result.row_count > 0;
    }
};

/// Connection Pool for HANA
pub const HanaConnectionPool = struct {
    allocator: std.mem.Allocator,
    config: HanaConfig,
    pool: std.ArrayList(*HanaClient),
    max_connections: usize,
    mutex: std.Thread.Mutex,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, config: HanaConfig, max_connections: usize) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .pool = std.ArrayList(*HanaClient).init(allocator),
            .max_connections = max_connections,
            .mutex = .{},
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.pool.items) |client| {
            client.deinit();
            self.allocator.destroy(client);
        }
        self.pool.deinit();
    }
    
    /// Get a connection from the pool
    pub fn acquire(self: *Self) !*HanaClient {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.pool.items.len > 0) {
            return self.pool.pop();
        }
        
        if (self.pool.items.len < self.max_connections) {
            const client = try self.allocator.create(HanaClient);
            client.* = try HanaClient.init(self.allocator, self.config);
            return client;
        }
        
        return error.PoolExhausted;
    }
    
    /// Return a connection to the pool
    pub fn release(self: *Self, client: *HanaClient) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.pool.append(client) catch {
            client.deinit();
            self.allocator.destroy(client);
        };
    }
};

test "HanaClient init" {
    const allocator = std.testing.allocator;
    const config = HanaConfig{
        .host = "test.hana.trial.us10.hanacloud.ondemand.com",
        .port = 443,
        .user = "test",
        .password = "test",
        .schema = "SAPABAP1",
    };
    
    var client = try HanaClient.init(allocator, config);
    defer client.deinit();
}