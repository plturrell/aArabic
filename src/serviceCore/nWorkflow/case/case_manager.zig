//! Case Management Module for nWorkflow
//! Implements dynamic case management similar to Appian Case Management Studio

const std = @import("std");
const Allocator = std.mem.Allocator;

// Case State enum
pub const CaseState = enum {
    DRAFT,
    OPEN,
    IN_PROGRESS,
    ON_HOLD,
    PENDING_REVIEW,
    RESOLVED,
    CLOSED,
    CANCELLED,
    REOPENED,

    pub fn toString(self: CaseState) []const u8 {
        return @tagName(self);
    }

    pub fn isTerminal(self: CaseState) bool {
        return self == .CLOSED or self == .CANCELLED;
    }

    pub fn isActive(self: CaseState) bool {
        return self == .OPEN or self == .IN_PROGRESS or self == .PENDING_REVIEW or self == .REOPENED;
    }
};

// Case Event for state transitions
pub const CaseEvent = enum {
    SUBMIT,
    ASSIGN,
    START_WORK,
    PUT_ON_HOLD,
    RESUME,
    SUBMIT_FOR_REVIEW,
    APPROVE,
    REJECT,
    RESOLVE,
    CLOSE,
    CANCEL,
    REOPEN,
};

// Case Priority
pub const CasePriority = enum {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL,

    pub fn toNumber(self: CasePriority) u8 {
        return switch (self) {
            .LOW => 1,
            .MEDIUM => 2,
            .HIGH => 3,
            .CRITICAL => 4,
        };
    }
};

// Case Type definition
pub const CaseType = struct {
    id: []const u8,
    name: []const u8,
    description: ?[]const u8 = null,
    default_priority: CasePriority = .MEDIUM,
    sla_hours: ?u32 = null,
    required_fields: ?[]const []const u8 = null,
    allowed_actions: ?[]const []const u8 = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8) !CaseType {
        return CaseType{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CaseType) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.description) |d| self.allocator.free(d);
    }
};

// Case Comment
pub const CaseComment = struct {
    id: []const u8,
    author_id: []const u8,
    content: []const u8,
    created_at: i64,
    is_internal: bool = false, // Internal comments not visible to customer
};

// Case Attachment
pub const CaseAttachment = struct {
    id: []const u8,
    filename: []const u8,
    mime_type: []const u8,
    size_bytes: u64,
    storage_path: []const u8,
    uploaded_by: []const u8,
    uploaded_at: i64,
};

// Case Activity Log Entry
pub const CaseActivity = struct {
    id: []const u8,
    case_id: []const u8,
    activity_type: []const u8,
    description: []const u8,
    performed_by: []const u8,
    timestamp: i64,
    metadata: ?[]const u8 = null, // JSON metadata
};

// Main Case struct
pub const Case = struct {
    id: []const u8,
    case_number: []const u8, // Human-readable case number
    case_type_id: []const u8,
    subject: []const u8,
    description: ?[]const u8 = null,

    state: CaseState = .DRAFT,
    priority: CasePriority = .MEDIUM,

    owner_id: ?[]const u8 = null,
    assignee_id: ?[]const u8 = null,
    team_id: ?[]const u8 = null,

    customer_id: ?[]const u8 = null,
    customer_email: ?[]const u8 = null,

    parent_case_id: ?[]const u8 = null, // For sub-cases
    related_case_ids: ?[]const []const u8 = null,

    workflow_instance_id: ?[]const u8 = null,

    // Data
    custom_fields: ?[]const u8 = null, // JSON custom fields

    // Timestamps
    created_at: i64,
    updated_at: i64,
    due_date: ?i64 = null,
    resolved_at: ?i64 = null,
    closed_at: ?i64 = null,

    // SLA tracking
    sla_breach_at: ?i64 = null,
    sla_breached: bool = false,

    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, case_number: []const u8, case_type_id: []const u8, subject: []const u8) !Case {
        const now = std.time.timestamp();
        return Case{
            .id = try allocator.dupe(u8, id),
            .case_number = try allocator.dupe(u8, case_number),
            .case_type_id = try allocator.dupe(u8, case_type_id),
            .subject = try allocator.dupe(u8, subject),
            .created_at = now,
            .updated_at = now,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Case) void {
        self.allocator.free(self.id);
        self.allocator.free(self.case_number);
        self.allocator.free(self.case_type_id);
        self.allocator.free(self.subject);
        if (self.description) |d| self.allocator.free(d);
        if (self.assignee_id) |a| self.allocator.free(a);
    }

    // State transition function
    pub fn transition(current: CaseState, event: CaseEvent) ?CaseState {
        return switch (current) {
            .DRAFT => switch (event) {
                .SUBMIT => .OPEN,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .OPEN => switch (event) {
                .ASSIGN => .OPEN,
                .START_WORK => .IN_PROGRESS,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .IN_PROGRESS => switch (event) {
                .PUT_ON_HOLD => .ON_HOLD,
                .SUBMIT_FOR_REVIEW => .PENDING_REVIEW,
                .RESOLVE => .RESOLVED,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .ON_HOLD => switch (event) {
                .RESUME => .IN_PROGRESS,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .PENDING_REVIEW => switch (event) {
                .APPROVE => .RESOLVED,
                .REJECT => .IN_PROGRESS,
                else => null,
            },
            .RESOLVED => switch (event) {
                .CLOSE => .CLOSED,
                .REOPEN => .REOPENED,
                else => null,
            },
            .REOPENED => switch (event) {
                .START_WORK => .IN_PROGRESS,
                .CLOSE => .CLOSED,
                else => null,
            },
            else => null,
        };
    }

    pub fn canTransition(self: *const Case, event: CaseEvent) bool {
        return transition(self.state, event) != null;
    }

    pub fn applyEvent(self: *Case, event: CaseEvent) !void {
        const new_state = transition(self.state, event) orelse return error.InvalidTransition;
        self.state = new_state;
        self.updated_at = std.time.timestamp();

        if (event == .RESOLVE) {
            self.resolved_at = self.updated_at;
        } else if (event == .CLOSE) {
            self.closed_at = self.updated_at;
        }
    }

    pub fn assign(self: *Case, assignee_id: []const u8) !void {
        if (self.assignee_id) |old_id| {
            self.allocator.free(old_id);
        }
        self.assignee_id = try self.allocator.dupe(u8, assignee_id);
        self.updated_at = std.time.timestamp();
    }

    pub fn checkSlaBreach(self: *Case) bool {
        if (self.sla_breach_at) |breach_time| {
            if (std.time.timestamp() > breach_time and !self.sla_breached) {
                self.sla_breached = true;
                return true;
            }
        }
        return false;
    }

    pub fn toJson(self: *const Case, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(allocator);

        var writer = buffer.writer(allocator);
        try writer.print(
            \\{{"id":"{s}","case_number":"{s}","subject":"{s}","state":"{s}","priority":{d},"created_at":{d}}}
        , .{
            self.id,
            self.case_number,
            self.subject,
            self.state.toString(),
            self.priority.toNumber(),
            self.created_at,
        });

        return buffer.toOwnedSlice(allocator);
    }
};


// Case Manager
pub const CaseManager = struct {
    cases: std.StringHashMap(*Case),
    cases_by_type: std.StringHashMap(std.ArrayList([]const u8)),
    cases_by_assignee: std.StringHashMap(std.ArrayList([]const u8)),
    case_types: std.StringHashMap(*CaseType),
    next_case_number: u64 = 1,
    allocator: Allocator,

    pub fn init(allocator: Allocator) CaseManager {
        return CaseManager{
            .cases = std.StringHashMap(*Case).init(allocator),
            .cases_by_type = std.StringHashMap(std.ArrayList([]const u8)).init(allocator),
            .cases_by_assignee = std.StringHashMap(std.ArrayList([]const u8)).init(allocator),
            .case_types = std.StringHashMap(*CaseType).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CaseManager) void {
        var iter = self.cases.valueIterator();
        while (iter.next()) |c| {
            c.*.deinit();
            self.allocator.destroy(c.*);
        }
        self.cases.deinit();

        var type_iter = self.case_types.valueIterator();
        while (type_iter.next()) |ct| {
            ct.*.deinit();
            self.allocator.destroy(ct.*);
        }
        self.case_types.deinit();

        // Deinit ArrayLists
        var by_type_iter = self.cases_by_type.valueIterator();
        while (by_type_iter.next()) |list| {
            list.deinit(self.allocator);
        }
        self.cases_by_type.deinit();

        var by_assignee_iter = self.cases_by_assignee.valueIterator();
        while (by_assignee_iter.next()) |list| {
            list.deinit(self.allocator);
        }
        self.cases_by_assignee.deinit();
    }

    pub fn registerCaseType(self: *CaseManager, case_type: *CaseType) !void {
        try self.case_types.put(case_type.id, case_type);
    }

    pub fn createCase(self: *CaseManager, case_type_id: []const u8, subject: []const u8) !*Case {
        // Generate IDs using counter to ensure uniqueness
        var id_buf: [64]u8 = undefined;
        const case_id = std.fmt.bufPrint(&id_buf, "case-{d}", .{self.next_case_number}) catch "case-0";

        var num_buf: [16]u8 = undefined;
        const case_number = std.fmt.bufPrint(&num_buf, "CASE-{d:0>6}", .{self.next_case_number}) catch "CASE-000000";
        self.next_case_number += 1;

        const case_ptr = try self.allocator.create(Case);
        case_ptr.* = try Case.init(self.allocator, case_id, case_number, case_type_id, subject);

        try self.cases.put(case_ptr.id, case_ptr);

        return case_ptr;
    }

    pub fn getCase(self: *CaseManager, case_id: []const u8) ?*Case {
        return self.cases.get(case_id);
    }

    pub fn getCaseByNumber(self: *CaseManager, case_number: []const u8) ?*Case {
        var iter = self.cases.valueIterator();
        while (iter.next()) |c| {
            if (std.mem.eql(u8, c.*.case_number, case_number)) {
                return c.*;
            }
        }
        return null;
    }

    pub fn getActiveCaseCount(self: *CaseManager) usize {
        var count: usize = 0;
        var iter = self.cases.valueIterator();
        while (iter.next()) |c| {
            if (c.*.state.isActive()) {
                count += 1;
            }
        }
        return count;
    }

    pub fn getCasesForAssignee(self: *CaseManager, assignee_id: []const u8, allocator: Allocator) ![]const *Case {
        var result = std.ArrayList(*Case){};
        var iter = self.cases.valueIterator();
        while (iter.next()) |c| {
            if (c.*.assignee_id) |aid| {
                if (std.mem.eql(u8, aid, assignee_id)) {
                    try result.append(allocator, c.*);
                }
            }
        }
        return result.toOwnedSlice(allocator);
    }

    pub fn getCasesByState(self: *CaseManager, state: CaseState, allocator: Allocator) ![]const *Case {
        var result = std.ArrayList(*Case){};
        var iter = self.cases.valueIterator();
        while (iter.next()) |c| {
            if (c.*.state == state) {
                try result.append(allocator, c.*);
            }
        }
        return result.toOwnedSlice(allocator);
    }

    pub fn getCasesByPriority(self: *CaseManager, priority: CasePriority, allocator: Allocator) ![]const *Case {
        var result = std.ArrayList(*Case){};
        var iter = self.cases.valueIterator();
        while (iter.next()) |c| {
            if (c.*.priority == priority) {
                try result.append(allocator, c.*);
            }
        }
        return result.toOwnedSlice(allocator);
    }

    pub fn getCaseCount(self: *CaseManager) usize {
        return self.cases.count();
    }

    pub fn deleteCase(self: *CaseManager, case_id: []const u8) bool {
        if (self.cases.fetchRemove(case_id)) |kv| {
            var case_ptr = kv.value;
            case_ptr.deinit();
            self.allocator.destroy(case_ptr);
            return true;
        }
        return false;
    }
};


// Tests
test "Case state transitions" {
    const allocator = std.testing.allocator;

    var case_obj = try Case.init(allocator, "case-1", "CASE-000001", "support", "Test Issue");
    defer case_obj.deinit();

    try std.testing.expectEqual(CaseState.DRAFT, case_obj.state);

    try case_obj.applyEvent(.SUBMIT);
    try std.testing.expectEqual(CaseState.OPEN, case_obj.state);

    try case_obj.applyEvent(.START_WORK);
    try std.testing.expectEqual(CaseState.IN_PROGRESS, case_obj.state);

    try case_obj.applyEvent(.RESOLVE);
    try std.testing.expectEqual(CaseState.RESOLVED, case_obj.state);
    try std.testing.expect(case_obj.resolved_at != null);

    try case_obj.applyEvent(.CLOSE);
    try std.testing.expectEqual(CaseState.CLOSED, case_obj.state);
}

test "CaseManager operations" {
    const allocator = std.testing.allocator;

    var manager = CaseManager.init(allocator);
    defer manager.deinit();

    const case_obj = try manager.createCase("support", "Test Case");
    try std.testing.expectEqualStrings("CASE-000001", case_obj.case_number);

    try std.testing.expectEqual(@as(usize, 0), manager.getActiveCaseCount());

    try case_obj.applyEvent(.SUBMIT);
    try std.testing.expectEqual(@as(usize, 1), manager.getActiveCaseCount());
}

test "Case priority ordering" {
    try std.testing.expect(CasePriority.LOW.toNumber() < CasePriority.MEDIUM.toNumber());
    try std.testing.expect(CasePriority.MEDIUM.toNumber() < CasePriority.HIGH.toNumber());
    try std.testing.expect(CasePriority.HIGH.toNumber() < CasePriority.CRITICAL.toNumber());
}

test "CaseState terminal and active checks" {
    try std.testing.expect(CaseState.CLOSED.isTerminal());
    try std.testing.expect(CaseState.CANCELLED.isTerminal());
    try std.testing.expect(!CaseState.OPEN.isTerminal());

    try std.testing.expect(CaseState.OPEN.isActive());
    try std.testing.expect(CaseState.IN_PROGRESS.isActive());
    try std.testing.expect(!CaseState.CLOSED.isActive());
    try std.testing.expect(!CaseState.DRAFT.isActive());
}

test "Case assignment" {
    const allocator = std.testing.allocator;

    var case_obj = try Case.init(allocator, "case-2", "CASE-000002", "support", "Assignment Test");
    defer case_obj.deinit();

    try case_obj.assign("user-123");
    try std.testing.expectEqualStrings("user-123", case_obj.assignee_id.?);

    // Reassignment should free old value
    try case_obj.assign("user-456");
    try std.testing.expectEqualStrings("user-456", case_obj.assignee_id.?);
}

test "Invalid state transitions" {
    const allocator = std.testing.allocator;

    var case_obj = try Case.init(allocator, "case-3", "CASE-000003", "support", "Invalid Transition Test");
    defer case_obj.deinit();

    // Cannot CLOSE from DRAFT
    const result = case_obj.applyEvent(.CLOSE);
    try std.testing.expectError(error.InvalidTransition, result);
}

test "Case JSON serialization" {
    const allocator = std.testing.allocator;

    var case_obj = try Case.init(allocator, "case-4", "CASE-000004", "support", "JSON Test");
    defer case_obj.deinit();

    const json = try case_obj.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "case-4") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "CASE-000004") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "JSON Test") != null);
}

test "CaseManager getCasesByState" {
    const allocator = std.testing.allocator;

    var manager = CaseManager.init(allocator);
    defer manager.deinit();

    const case1 = try manager.createCase("support", "Case 1");
    try case1.applyEvent(.SUBMIT);
    // case1 should now be in OPEN state
    try std.testing.expectEqual(CaseState.OPEN, case1.state);

    const case2 = try manager.createCase("support", "Case 2");
    try case2.applyEvent(.SUBMIT);
    try case2.applyEvent(.START_WORK);
    // case2 should now be in IN_PROGRESS state
    try std.testing.expectEqual(CaseState.IN_PROGRESS, case2.state);

    // Verify we have 2 cases total
    try std.testing.expectEqual(@as(usize, 2), manager.getCaseCount());

    const draft_cases = try manager.getCasesByState(.DRAFT, allocator);
    defer allocator.free(draft_cases);
    try std.testing.expectEqual(@as(usize, 0), draft_cases.len);

    const open_cases = try manager.getCasesByState(.OPEN, allocator);
    defer allocator.free(open_cases);
    try std.testing.expectEqual(@as(usize, 1), open_cases.len);

    const in_progress_cases = try manager.getCasesByState(.IN_PROGRESS, allocator);
    defer allocator.free(in_progress_cases);
    try std.testing.expectEqual(@as(usize, 1), in_progress_cases.len);
}

test "CaseType initialization" {
    const allocator = std.testing.allocator;

    var case_type = try CaseType.init(allocator, "incident", "Incident Report");
    defer case_type.deinit();

    try std.testing.expectEqualStrings("incident", case_type.id);
    try std.testing.expectEqualStrings("Incident Report", case_type.name);
    try std.testing.expectEqual(CasePriority.MEDIUM, case_type.default_priority);
}
