//! ============================================================================
//! Trial Balance API endpoints
//! Handles REST API requests for trial balance operations based on DOI specs
//! ============================================================================
//!
//! [CODE:file=trial_balance.zig]
//! [CODE:module=api]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated,variances]
//! [ODPS:rules=TB001,TB002,VAR001,VAR002,VAR003,VAR004,VAR005]
//!
//! [DOI:controls=VAL-001,MKR-CHK-001]
//! [DOI:thresholds=REQ-THRESH-001,REQ-THRESH-002,REQ-THRESH-003]
//!
//! [PETRI:stages=S02,S03,S04,S05,S06]
//!
//! [TABLE:reads=TB_TRIAL_BALANCE,TB_VARIANCE_DETAILS]
//! [TABLE:writes=TB_TRIAL_BALANCE,TB_VARIANCE_DETAILS]
//!
//! [API:produces=/api/v1/trial-balance,/api/v1/accounts,/api/v1/trial-balance/summary]
//!
//! [RELATION:calls=CODE:balance_engine.zig]
//! [RELATION:calls=CODE:destination_client.zig]
//! [RELATION:called_by=CODE:main.zig]
//!
//! This API implements DOI variance thresholds:
//! - BS: $100m AND >10% (REQ-THRESH-001)
//! - P&L: $3m AND >10% (REQ-THRESH-002)

const std = @import("std");
const DestinationClient = @import("../../../integrations/src/destination_client.zig");

/// Trial Balance API endpoints
/// Handles REST API requests for trial balance operations based on DOI specifications

pub const AccountType = enum {
    BalanceSheet,
    ProfitAndLoss,
};

pub const TrialBalanceEntry = struct {
    account_id: []const u8, // Using description as ID if ID is missing
    description: []const u8,
    account_type: AccountType,
    current_balance_usd: f64,
    previous_balance_usd: f64,
    
    // Calculated fields
    variance_abs: f64,
    variance_pct: f64,
    is_material: bool,
    commentary: ?[]const u8,
};

// JSON struct for Python script output
const RawEntry = struct {
    description: []const u8,
    current: f64,
    previous: f64,
    variance_abs: f64,
    variance_pct: f64,
    comments: ?[]const u8,
};

pub const VarianceThresholds = struct {
    bs_abs_threshold: f64 = 100_000_000.0, // $100m
    pl_abs_threshold: f64 = 3_000_000.0,   // $3m
    pct_threshold: f64 = 10.0,             // 10%
};

pub const TrialBalanceAPI = struct {
    allocator: std.mem.Allocator,
    ai_client: DestinationClient.LocalModelsClient,
    thresholds: VarianceThresholds,

    pub fn init(allocator: std.mem.Allocator) !TrialBalanceAPI {
        return .{
            .allocator = allocator,
            .ai_client = try DestinationClient.LocalModelsClient.init(allocator),
            .thresholds = .{},
        };
    }

    pub fn deinit(self: *TrialBalanceAPI) void {
        self.ai_client.deinit();
    }

    /// Process a Trial Balance file using the Python script for XLSB extraction
    pub fn processReviewFile(self: *TrialBalanceAPI, file_name: []const u8) ![]u8 {
        // Define paths
        // Assuming we are running from backend root
        const script_path = "../BusDocs/extract_xlsb.py";
        const venv_python = "../BusDocs/venv/bin/python3";
        const file_path_prefix = "../BusDocs/";
        
        const full_file_path = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{file_path_prefix, file_name});
        defer self.allocator.free(full_file_path);

        // Execute Python script
        const argv = &[_][]const u8{
            venv_python,
            script_path,
            full_file_path,
        };

        var child = std.process.Child.init(argv, self.allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        try child.spawn();
        
        // Read output (limited to 10MB)
        const stdout = try child.stdout.?.reader().readAllAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(stdout);
        
        const stderr = try child.stderr.?.reader().readAllAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(stderr);

        const term = try child.wait();
        
        if (term.Exited != 0) {
            std.debug.print("Python script failed: {s}\n", .{stderr});
            return error.ScriptExecutionFailed;
        }

        // Parse JSON output
        const parsed = try std.json.parseFromSlice([]RawEntry, self.allocator, stdout, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        
        var entries = std.ArrayList(TrialBalanceEntry).init(self.allocator);
        defer entries.deinit();

        // Convert raw entries to processed entries
        for (parsed.value) |raw| {
            // Determine type based on file name or description context
            // Simplified logic: Assume P&L if file has "PL"
            const acc_type: AccountType = if (std.mem.indexOf(u8, file_name, "PL") != null) .ProfitAndLoss else .BalanceSheet;

            var entry = TrialBalanceEntry{
                .account_id = raw.description, // Use desc as ID
                .description = raw.description,
                .account_type = acc_type,
                .current_balance_usd = raw.current,
                .previous_balance_usd = raw.previous,
                .variance_abs = raw.variance_abs,
                .variance_pct = raw.variance_pct,
                .is_material = false,
                .commentary = if (raw.comments) |c| try self.allocator.dupe(u8, c) else null,
            };

            self.analyzeVariance(&entry);

            // If material and no comment, generate AI comment
            if (entry.is_material and (entry.commentary == null or std.mem.eql(u8, entry.commentary.?, "Immaterial Variance"))) {
                 const context = try std.fmt.allocPrint(self.allocator, 
                    "Explain variance for {s}. Current: ${d}m, Previous: ${d}m. Variance: ${d}m ({d:.1}%). Context: HKG market expansion.",
                    .{
                        entry.description,
                        entry.current_balance_usd / 1_000_000.0,
                        entry.previous_balance_usd / 1_000_000.0,
                        entry.variance_abs / 1_000_000.0,
                        entry.variance_pct
                    }
                );
                defer self.allocator.free(context);
                
                // Call AI Service
                const ai_response = try self.ai_client.generateNarrative(context);
                // Clean up in real code
                entry.commentary = ai_response; 
            }
            
            try entries.append(entry);
        }

        // Serialize result
        var json_out = std.ArrayList(u8).init(self.allocator);
        try json_out.writer().print("{{\"file\": \"{s}\", \"entries\": [", .{file_name});
        
        for (entries.items, 0..) |entry, i| {
            if (i > 0) try json_out.appendSlice(",");
            
            // Handle optional commentary string safely
            const comment = if (entry.commentary) |c| c else "null";
            // We need to escape quotes in comments for valid JSON, keeping it simple here
            
            try json_out.writer().print(
                \\{{"desc": "{s}", "current": {d:.2}, "previous": {d:.2}, "variance_abs": {d:.2}, "variance_pct": {d:.2}, "is_material": {}, "commentary": "{s}"}}
                , .{
                    entry.description, 
                    entry.current_balance_usd,
                    entry.previous_balance_usd,
                    entry.variance_abs,
                    entry.variance_pct,
                    entry.is_material,
                    comment
                }
            );
        }
        try json_out.appendSlice("]}");
        
        return json_out.toOwnedSlice();
    }

    /// Calculate variance and check against DOI thresholds
    fn analyzeVariance(self: *TrialBalanceAPI, entry: *TrialBalanceEntry) void {
        // If variance is 0 from file, recalculate if we have balances
        if (entry.variance_abs == 0 and entry.current_balance_usd != 0) {
            entry.variance_abs = entry.current_balance_usd - entry.previous_balance_usd;
        }
        
        if (entry.variance_pct == 0 and entry.previous_balance_usd != 0) {
            entry.variance_pct = (entry.variance_abs / entry.previous_balance_usd) * 100.0;
        }

        const abs_variance = @abs(entry.variance_abs);
        const abs_pct = @abs(entry.variance_pct);

        const amount_threshold = switch (entry.account_type) {
            .BalanceSheet => self.thresholds.bs_abs_threshold,
            .ProfitAndLoss => self.thresholds.pl_abs_threshold,
        };

        // DOI Rule: Material if > Amount Threshold AND > Percentage Threshold
        if (abs_variance > amount_threshold and abs_pct > self.thresholds.pct_threshold) {
            entry.is_material = true;
        } else {
            entry.is_material = false;
        }
    }

    /// GET /api/v1/trial-balance
    /// Returns list of trial balance accounts
    pub fn getTrialBalance(self: *TrialBalanceAPI) ![]u8 {
        // Mock data
        const accounts = 
            \\[
            \\  {"id": "1000", "name": "Cash", "debit": 50000.00, "credit": 0.00, "currency": "USD"},
            \\  {"id": "1100", "name": "Accounts Receivable", "debit": 15000.50, "credit": 0.00, "currency": "USD"},
            \\  {"id": "2000", "name": "Accounts Payable", "debit": 0.00, "credit": 12000.00, "currency": "USD"},
            \\  {"id": "3000", "name": "Retained Earnings", "debit": 0.00, "credit": 35000.00, "currency": "USD"},
            \\  {"id": "4000", "name": "Revenue", "debit": 0.00, "credit": 60000.00, "currency": "USD"},
            \\  {"id": "5000", "name": "Expenses", "debit": 42000.00, "credit": 0.00, "currency": "USD"}
            \\]
        ;
        return try self.allocator.dupe(u8, accounts);
    }

    /// GET /api/v1/trial-balance/:accountId
    /// Returns specific account details
    pub fn getAccount(self: *TrialBalanceAPI, account_id: []const u8) ![]u8 {
        // Mock account detail
        const json = try std.fmt.allocPrint(self.allocator, 
            "{{\"id\": \"{s}\", \"name\": \"Account {s}\", \"transactions\": []}}", 
            .{account_id, account_id});
        return json;
    }

    /// POST /api/v1/trial-balance
    /// Creates new trial balance entry (Maker)
    pub fn createEntry(self: *TrialBalanceAPI, data: []const u8) ![]u8 {
        // TODO: Implement create with workflow
        _ = self;
        _ = data;
        return try self.allocator.dupe(u8, "{\"status\": \"created\", \"id\": \"new_id\"}");
    }

    /// PUT /api/v1/trial-balance/:accountId
    /// Updates trial balance entry (Maker)
    pub fn updateEntry(self: *TrialBalanceAPI, account_id: []const u8, data: []const u8) ![]u8 {
        // TODO: Implement update with workflow
        _ = self;
        _ = account_id;
        _ = data;
        return try self.allocator.dupe(u8, "{\"status\": \"updated\"}");
    }

    /// GET /api/v1/trial-balance/summary
    /// Returns summary totals
    pub fn getSummary(self: *TrialBalanceAPI) ![]u8 {
        const summary = 
            \\{"total_debit": 107000.50, "total_credit": 107000.00, "variance": 0.50, "status": "UNBALANCED"}
        ;
        return try self.allocator.dupe(u8, summary);
    }

    /// POST /api/v1/trial-balance/narrative
    /// Generates AI narrative for the current balance state
    pub fn getNarrative(self: *TrialBalanceAPI, context_data: []const u8) ![]u8 {
        // Call nLocalModels to generate narrative
        return self.ai_client.generateNarrative(context_data);
    }
};

test "TrialBalanceAPI init" {
    const allocator = std.testing.allocator;
    var api = try TrialBalanceAPI.init(allocator);
    defer api.deinit();
}