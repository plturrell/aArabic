//! ============================================================================
//! SQLite Adapter for Trial Balance Development
//! Provides database access for local development and testing
//! ============================================================================
//!
//! [CODE:file=sqlite_adapter.zig]
//! [CODE:module=models/calculation]
//! [CODE:language=zig]
//!
//! [TABLE:manages=JOURNAL_ENTRIES,ACCOUNT_BALANCES,TRIAL_BALANCE_RESULTS]
//!
//! [RELATION:used_by=CODE:balance_engine.zig]
//! [RELATION:dev_replacement_for=HANA]
//!
//! Note: Development adapter for local testing without HANA.
//! Production deployments use HANA directly.

const std = @import("std");
const c = @cImport({
    @cInclude("sqlite3.h");
});

pub const SQLiteError = error{
    OpenFailed,
    PrepareFailed,
    ExecuteFailed,
    BindFailed,
    StepFailed,
};

pub const Database = struct {
    db: ?*c.sqlite3,

    pub fn init(path: [:0]const u8) !Database {
        var db: ?*c.sqlite3 = null;
        const result = c.sqlite3_open(path.ptr, &db);
        
        if (result != c.SQLITE_OK) {
            if (db) |d| {
                _ = c.sqlite3_close(d);
            }
            return SQLiteError.OpenFailed;
        }

        return Database{ .db = db };
    }

    pub fn deinit(self: *Database) void {
        if (self.db) |db| {
            _ = c.sqlite3_close(db);
            self.db = null;
        }
    }

    pub fn execute(self: *Database, sql: [:0]const u8) !void {
        var err_msg: [*c]u8 = undefined;
        const result = c.sqlite3_exec(self.db, sql.ptr, null, null, &err_msg);
        
        if (result != c.SQLITE_OK) {
            defer c.sqlite3_free(err_msg);
            return SQLiteError.ExecuteFailed;
        }
    }
};

pub const JournalEntry = struct {
    entry_id: []const u8,
    company_code: []const u8,
    fiscal_year: []const u8,
    period: []const u8,
    account: []const u8,
    debit_credit_indicator: u8,
    amount: f64,
    currency: []const u8,
};

pub const GLAccount = struct {
    account_id: []const u8,
    account_number: []const u8,
    description: []const u8,
    ifrs_schedule: []const u8,
    ifrs_category: []const u8,
    account_type: []const u8,
};

pub const DatabaseReader = struct {
    db: *Database,
    allocator: std.mem.Allocator,

    pub fn init(db: *Database, allocator: std.mem.Allocator) DatabaseReader {
        return DatabaseReader{
            .db = db,
            .allocator = allocator,
        };
    }

    pub fn read_journal_entries(
        self: *DatabaseReader,
        company_code: []const u8,
        fiscal_year: []const u8,
        period: []const u8,
    ) !std.ArrayList(JournalEntry) {
        var entries: std.ArrayList(JournalEntry) = .{};
        errdefer entries.deinit(self.allocator);

        // SQL query to read journal entries
        const sql =
            \\SELECT entry_id, rbukrs, gjahr, poper, racct, drcrk, hsl, rtcur
            \\FROM TB_JOURNAL_ENTRIES
            \\WHERE rbukrs = ? AND gjahr = ? AND poper = ? AND validated = 1
        ;

        var stmt: ?*c.sqlite3_stmt = null;
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);

        var result = c.sqlite3_prepare_v2(self.db.db, sql_z.ptr, -1, &stmt, null);
        if (result != c.SQLITE_OK) {
            return SQLiteError.PrepareFailed;
        }
        defer _ = c.sqlite3_finalize(stmt);

        // Bind parameters (using null for destructor means SQLite will copy the data)
        const cc_z = try self.allocator.dupeZ(u8, company_code);
        defer self.allocator.free(cc_z);
        result = c.sqlite3_bind_text(stmt, 1, cc_z.ptr, -1, null);
        if (result != c.SQLITE_OK) return SQLiteError.BindFailed;

        const fy_z = try self.allocator.dupeZ(u8, fiscal_year);
        defer self.allocator.free(fy_z);
        result = c.sqlite3_bind_text(stmt, 2, fy_z.ptr, -1, null);
        if (result != c.SQLITE_OK) return SQLiteError.BindFailed;

        const p_z = try self.allocator.dupeZ(u8, period);
        defer self.allocator.free(p_z);
        result = c.sqlite3_bind_text(stmt, 3, p_z.ptr, -1, null);
        if (result != c.SQLITE_OK) return SQLiteError.BindFailed;

        // Execute query and fetch results
        while (c.sqlite3_step(stmt) == c.SQLITE_ROW) {
            const entry_id = std.mem.span(c.sqlite3_column_text(stmt, 0));
            const rbukrs = std.mem.span(c.sqlite3_column_text(stmt, 1));
            const gjahr = std.mem.span(c.sqlite3_column_text(stmt, 2));
            const poper = std.mem.span(c.sqlite3_column_text(stmt, 3));
            const racct = std.mem.span(c.sqlite3_column_text(stmt, 4));
            const drcrk = std.mem.span(c.sqlite3_column_text(stmt, 5));
            const hsl = c.sqlite3_column_double(stmt, 6);
            const rtcur = std.mem.span(c.sqlite3_column_text(stmt, 7));

            const entry = JournalEntry{
                .entry_id = try self.allocator.dupe(u8, entry_id),
                .company_code = try self.allocator.dupe(u8, rbukrs),
                .fiscal_year = try self.allocator.dupe(u8, gjahr),
                .period = try self.allocator.dupe(u8, poper),
                .account = try self.allocator.dupe(u8, racct),
                .debit_credit_indicator = drcrk[0],
                .amount = hsl,
                .currency = try self.allocator.dupe(u8, rtcur),
            };

            try entries.append(self.allocator, entry);
        }

        return entries;
    }

    pub fn read_gl_accounts(self: *DatabaseReader) !std.ArrayList(GLAccount) {
        var accounts: std.ArrayList(GLAccount) = .{};
        errdefer accounts.deinit(self.allocator);

        const sql =
            \\SELECT account_id, saknr, txt50, ifrs_schedule, ifrs_category, account_type
            \\FROM TB_GL_ACCOUNTS
        ;

        var stmt: ?*c.sqlite3_stmt = null;
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);

        const result = c.sqlite3_prepare_v2(self.db.db, sql_z.ptr, -1, &stmt, null);
        if (result != c.SQLITE_OK) {
            return SQLiteError.PrepareFailed;
        }
        defer _ = c.sqlite3_finalize(stmt);

        while (c.sqlite3_step(stmt) == c.SQLITE_ROW) {
            const account_id = std.mem.span(c.sqlite3_column_text(stmt, 0));
            const saknr = std.mem.span(c.sqlite3_column_text(stmt, 1));
            const txt50 = std.mem.span(c.sqlite3_column_text(stmt, 2));
            const ifrs_schedule = std.mem.span(c.sqlite3_column_text(stmt, 3));
            const ifrs_category = std.mem.span(c.sqlite3_column_text(stmt, 4));
            const account_type = std.mem.span(c.sqlite3_column_text(stmt, 5));

            const account = GLAccount{
                .account_id = try self.allocator.dupe(u8, account_id),
                .account_number = try self.allocator.dupe(u8, saknr),
                .description = try self.allocator.dupe(u8, txt50),
                .ifrs_schedule = try self.allocator.dupe(u8, ifrs_schedule),
                .ifrs_category = try self.allocator.dupe(u8, ifrs_category),
                .account_type = try self.allocator.dupe(u8, account_type),
            };

            try accounts.append(self.allocator, account);
        }

        return accounts;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "database connection" {
    var db = try Database.init(":memory:");
    defer db.deinit();

    try db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)");
    try db.execute("INSERT INTO test (name) VALUES ('test')");
}