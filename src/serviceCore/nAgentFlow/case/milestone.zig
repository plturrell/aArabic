//! Milestone Management Module for nWorkflow Case Management
//! Implements milestone tracking with conditions and auto-complete logic

const std = @import("std");
const Allocator = std.mem.Allocator;

// Milestone State
pub const MilestoneState = enum {
    NOT_STARTED,
    IN_PROGRESS,
    COMPLETED,
    SKIPPED,

    pub fn toString(self: MilestoneState) []const u8 {
        return @tagName(self);
    }

    pub fn isTerminal(self: MilestoneState) bool {
        return self == .COMPLETED or self == .SKIPPED;
    }
};

// Condition Types for milestone completion
pub const ConditionType = enum {
    FIELD_EQUALS,
    FIELD_NOT_EMPTY,
    FIELD_CONTAINS,
    TASK_COMPLETED,
    ALL_TASKS_COMPLETED,
    ANY_TASK_COMPLETED,
    TIME_ELAPSED,
    CUSTOM_EXPRESSION,
};

// Milestone Condition
pub const MilestoneCondition = struct {
    condition_type: ConditionType,
    field_name: ?[]const u8 = null,
    expected_value: ?[]const u8 = null,
    task_ids: ?[]const []const u8 = null,
    time_seconds: ?i64 = null,
    expression: ?[]const u8 = null, // Custom expression for evaluation
    allocator: Allocator,

    pub fn init(allocator: Allocator, condition_type: ConditionType) MilestoneCondition {
        return MilestoneCondition{
            .condition_type = condition_type,
            .allocator = allocator,
        };
    }

    pub fn withFieldEquals(allocator: Allocator, field_name: []const u8, expected_value: []const u8) !MilestoneCondition {
        return MilestoneCondition{
            .condition_type = .FIELD_EQUALS,
            .field_name = try allocator.dupe(u8, field_name),
            .expected_value = try allocator.dupe(u8, expected_value),
            .allocator = allocator,
        };
    }

    pub fn withFieldNotEmpty(allocator: Allocator, field_name: []const u8) !MilestoneCondition {
        return MilestoneCondition{
            .condition_type = .FIELD_NOT_EMPTY,
            .field_name = try allocator.dupe(u8, field_name),
            .allocator = allocator,
        };
    }

    pub fn withTimeElapsed(allocator: Allocator, seconds: i64) MilestoneCondition {
        return MilestoneCondition{
            .condition_type = .TIME_ELAPSED,
            .time_seconds = seconds,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MilestoneCondition) void {
        if (self.field_name) |f| self.allocator.free(f);
        if (self.expected_value) |v| self.allocator.free(v);
        if (self.expression) |e| self.allocator.free(e);
    }

    // Evaluate condition against provided context
    pub fn evaluate(self: *const MilestoneCondition, context: *const EvaluationContext) bool {
        return switch (self.condition_type) {
            .FIELD_EQUALS => {
                if (self.field_name) |field| {
                    if (context.getField(field)) |value| {
                        if (self.expected_value) |expected| {
                            return std.mem.eql(u8, value, expected);
                        }
                    }
                }
                return false;
            },
            .FIELD_NOT_EMPTY => {
                if (self.field_name) |field| {
                    if (context.getField(field)) |value| {
                        return value.len > 0;
                    }
                }
                return false;
            },
            .TIME_ELAPSED => {
                if (self.time_seconds) |seconds| {
                    return context.elapsed_time >= seconds;
                }
                return false;
            },
            .TASK_COMPLETED => {
                if (self.task_ids) |ids| {
                    if (ids.len > 0) {
                        return context.isTaskCompleted(ids[0]);
                    }
                }
                return false;
            },
            .ALL_TASKS_COMPLETED => {
                if (self.task_ids) |ids| {
                    for (ids) |id| {
                        if (!context.isTaskCompleted(id)) return false;
                    }
                    return true;
                }
                return false;
            },
            .ANY_TASK_COMPLETED => {
                if (self.task_ids) |ids| {
                    for (ids) |id| {
                        if (context.isTaskCompleted(id)) return true;
                    }
                }
                return false;
            },
            else => false,
        };
    }
};

// Evaluation context for conditions
pub const EvaluationContext = struct {
    fields: std.StringHashMap([]const u8),
    completed_tasks: std.StringHashMap(bool),
    elapsed_time: i64 = 0,
    allocator: Allocator,

    pub fn init(allocator: Allocator) EvaluationContext {
        return EvaluationContext{
            .fields = std.StringHashMap([]const u8).init(allocator),
            .completed_tasks = std.StringHashMap(bool).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *EvaluationContext) void {
        self.fields.deinit();
        self.completed_tasks.deinit();
    }

    pub fn setField(self: *EvaluationContext, name: []const u8, value: []const u8) !void {
        try self.fields.put(name, value);
    }

    pub fn getField(self: *const EvaluationContext, name: []const u8) ?[]const u8 {
        return self.fields.get(name);
    }

    pub fn markTaskCompleted(self: *EvaluationContext, task_id: []const u8) !void {
        try self.completed_tasks.put(task_id, true);
    }

    pub fn isTaskCompleted(self: *const EvaluationContext, task_id: []const u8) bool {
        return self.completed_tasks.get(task_id) orelse false;
    }
};

// Milestone struct
pub const Milestone = struct {
    id: []const u8,
    name: []const u8,
    description: ?[]const u8 = null,
    case_id: []const u8,

    state: MilestoneState = .NOT_STARTED,
    order: u32 = 0, // For ordering milestones

    // Conditions for auto-completion
    conditions: std.ArrayList(MilestoneCondition),
    require_all_conditions: bool = true, // AND vs OR logic

    // Dependencies on other milestones
    dependencies: std.ArrayList([]const u8),

    // Timing
    due_date: ?i64 = null,
    started_at: ?i64 = null,
    completed_at: ?i64 = null,
    created_at: i64,

    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, case_id: []const u8) !Milestone {
        return Milestone{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .case_id = try allocator.dupe(u8, case_id),
            .conditions = std.ArrayList(MilestoneCondition){},
            .dependencies = std.ArrayList([]const u8){},
            .created_at = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Milestone) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.case_id);
        if (self.description) |d| self.allocator.free(d);

        for (self.conditions.items) |*cond| {
            cond.deinit();
        }
        self.conditions.deinit(self.allocator);

        for (self.dependencies.items) |dep| {
            self.allocator.free(dep);
        }
        self.dependencies.deinit(self.allocator);
    }

    pub fn addCondition(self: *Milestone, condition: MilestoneCondition) !void {
        try self.conditions.append(self.allocator, condition);
    }

    pub fn addDependency(self: *Milestone, milestone_id: []const u8) !void {
        try self.dependencies.append(self.allocator, try self.allocator.dupe(u8, milestone_id));
    }

    pub fn start(self: *Milestone) !void {
        if (self.state != .NOT_STARTED) return error.InvalidStateTransition;
        self.state = .IN_PROGRESS;
        self.started_at = std.time.timestamp();
    }

    pub fn complete(self: *Milestone) !void {
        if (self.state != .IN_PROGRESS) return error.InvalidStateTransition;
        self.state = .COMPLETED;
        self.completed_at = std.time.timestamp();
    }

    pub fn skip(self: *Milestone) !void {
        if (self.state.isTerminal()) return error.InvalidStateTransition;
        self.state = .SKIPPED;
        self.completed_at = std.time.timestamp();
    }

    // Check if all conditions are met for auto-completion
    pub fn checkAutoComplete(self: *Milestone, context: *const EvaluationContext) bool {
        if (self.conditions.items.len == 0) return false;

        if (self.require_all_conditions) {
            // AND logic - all conditions must be true
            for (self.conditions.items) |*cond| {
                if (!cond.evaluate(context)) return false;
            }
            return true;
        } else {
            // OR logic - any condition can be true
            for (self.conditions.items) |*cond| {
                if (cond.evaluate(context)) return true;
            }
            return false;
        }
    }

    pub fn isOverdue(self: *const Milestone) bool {
        if (self.due_date) |due| {
            return std.time.timestamp() > due and !self.state.isTerminal();
        }
        return false;
    }
};


// Milestone Tracker - manages multiple milestones for a case
pub const MilestoneTracker = struct {
    milestones: std.StringHashMap(*Milestone),
    milestones_by_case: std.StringHashMap(std.ArrayList([]const u8)),
    allocator: Allocator,

    pub fn init(allocator: Allocator) MilestoneTracker {
        return MilestoneTracker{
            .milestones = std.StringHashMap(*Milestone).init(allocator),
            .milestones_by_case = std.StringHashMap(std.ArrayList([]const u8)).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MilestoneTracker) void {
        var iter = self.milestones.valueIterator();
        while (iter.next()) |m| {
            m.*.deinit();
            self.allocator.destroy(m.*);
        }
        self.milestones.deinit();

        var case_iter = self.milestones_by_case.valueIterator();
        while (case_iter.next()) |list| {
            for (list.items) |id| {
                self.allocator.free(id);
            }
            list.deinit(self.allocator);
        }
        self.milestones_by_case.deinit();
    }

    pub fn addMilestone(self: *MilestoneTracker, milestone: *Milestone) !void {
        try self.milestones.put(milestone.id, milestone);

        // Track by case
        var entry = try self.milestones_by_case.getOrPut(milestone.case_id);
        if (!entry.found_existing) {
            entry.value_ptr.* = std.ArrayList([]const u8){};
        }
        try entry.value_ptr.append(self.allocator, try self.allocator.dupe(u8, milestone.id));
    }

    pub fn getMilestone(self: *MilestoneTracker, milestone_id: []const u8) ?*Milestone {
        return self.milestones.get(milestone_id);
    }

    pub fn getMilestonesForCase(self: *MilestoneTracker, case_id: []const u8, allocator: Allocator) ![]const *Milestone {
        var result = std.ArrayList(*Milestone){};
        if (self.milestones_by_case.get(case_id)) |ids| {
            for (ids.items) |id| {
                if (self.milestones.get(id)) |m| {
                    try result.append(m);
                }
            }
        }
        return result.toOwnedSlice();
    }

    // Check dependencies for a milestone
    pub fn areDependenciesMet(self: *MilestoneTracker, milestone: *const Milestone) bool {
        for (milestone.dependencies.items) |dep_id| {
            if (self.milestones.get(dep_id)) |dep| {
                if (dep.state != .COMPLETED) return false;
            } else {
                return false; // Dependency not found
            }
        }
        return true;
    }

    // Auto-complete milestones based on context
    pub fn processAutoComplete(self: *MilestoneTracker, case_id: []const u8, context: *const EvaluationContext) !u32 {
        var completed_count: u32 = 0;
        if (self.milestones_by_case.get(case_id)) |ids| {
            for (ids.items) |id| {
                if (self.milestones.get(id)) |m| {
                    if (m.state == .IN_PROGRESS and self.areDependenciesMet(m)) {
                        if (m.checkAutoComplete(context)) {
                            try m.complete();
                            completed_count += 1;
                        }
                    }
                }
            }
        }
        return completed_count;
    }

    // Get completion percentage for a case
    pub fn getCompletionPercentage(self: *MilestoneTracker, case_id: []const u8) f32 {
        var total: u32 = 0;
        var completed: u32 = 0;

        if (self.milestones_by_case.get(case_id)) |ids| {
            for (ids.items) |id| {
                if (self.milestones.get(id)) |m| {
                    if (m.state != .SKIPPED) {
                        total += 1;
                        if (m.state == .COMPLETED) completed += 1;
                    }
                }
            }
        }

        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(completed)) / @as(f32, @floatFromInt(total)) * 100.0;
    }

    pub fn getMilestoneCount(self: *MilestoneTracker) usize {
        return self.milestones.count();
    }
};

// Tests
test "MilestoneState transitions" {
    try std.testing.expect(MilestoneState.COMPLETED.isTerminal());
    try std.testing.expect(MilestoneState.SKIPPED.isTerminal());
    try std.testing.expect(!MilestoneState.IN_PROGRESS.isTerminal());
    try std.testing.expect(!MilestoneState.NOT_STARTED.isTerminal());
}

test "Milestone basic operations" {
    const allocator = std.testing.allocator;

    var milestone = try Milestone.init(allocator, "ms-1", "Initial Review", "case-1");
    defer milestone.deinit();

    try std.testing.expectEqual(MilestoneState.NOT_STARTED, milestone.state);

    try milestone.start();
    try std.testing.expectEqual(MilestoneState.IN_PROGRESS, milestone.state);
    try std.testing.expect(milestone.started_at != null);

    try milestone.complete();
    try std.testing.expectEqual(MilestoneState.COMPLETED, milestone.state);
    try std.testing.expect(milestone.completed_at != null);
}

test "MilestoneCondition FIELD_EQUALS" {
    const allocator = std.testing.allocator;

    var cond = try MilestoneCondition.withFieldEquals(allocator, "status", "approved");
    defer cond.deinit();

    var context = EvaluationContext.init(allocator);
    defer context.deinit();

    try context.setField("status", "pending");
    try std.testing.expect(!cond.evaluate(&context));

    try context.setField("status", "approved");
    try std.testing.expect(cond.evaluate(&context));
}

test "MilestoneCondition TIME_ELAPSED" {
    const allocator = std.testing.allocator;

    var cond = MilestoneCondition.withTimeElapsed(allocator, 3600); // 1 hour
    defer cond.deinit();

    var context = EvaluationContext.init(allocator);
    defer context.deinit();

    context.elapsed_time = 1800; // 30 minutes
    try std.testing.expect(!cond.evaluate(&context));

    context.elapsed_time = 7200; // 2 hours
    try std.testing.expect(cond.evaluate(&context));
}

test "MilestoneTracker operations" {
    const allocator = std.testing.allocator;

    var tracker = MilestoneTracker.init(allocator);
    defer tracker.deinit();

    var ms1 = try allocator.create(Milestone);
    ms1.* = try Milestone.init(allocator, "ms-1", "Milestone 1", "case-1");
    try tracker.addMilestone(ms1);

    var ms2 = try allocator.create(Milestone);
    ms2.* = try Milestone.init(allocator, "ms-2", "Milestone 2", "case-1");
    try ms2.addDependency("ms-1");
    try tracker.addMilestone(ms2);

    try std.testing.expectEqual(@as(usize, 2), tracker.getMilestoneCount());

    // ms2 depends on ms1, which is not completed
    try std.testing.expect(!tracker.areDependenciesMet(ms2));

    // Complete ms1
    try ms1.start();
    try ms1.complete();
    try std.testing.expect(tracker.areDependenciesMet(ms2));
}

test "MilestoneTracker completion percentage" {
    const allocator = std.testing.allocator;

    var tracker = MilestoneTracker.init(allocator);
    defer tracker.deinit();

    var ms1 = try allocator.create(Milestone);
    ms1.* = try Milestone.init(allocator, "ms-1", "Milestone 1", "case-1");
    try tracker.addMilestone(ms1);

    var ms2 = try allocator.create(Milestone);
    ms2.* = try Milestone.init(allocator, "ms-2", "Milestone 2", "case-1");
    try tracker.addMilestone(ms2);

    try std.testing.expectEqual(@as(f32, 0.0), tracker.getCompletionPercentage("case-1"));

    try ms1.start();
    try ms1.complete();
    try std.testing.expectEqual(@as(f32, 50.0), tracker.getCompletionPercentage("case-1"));

    try ms2.start();
    try ms2.complete();
    try std.testing.expectEqual(@as(f32, 100.0), tracker.getCompletionPercentage("case-1"));
}

test "Milestone auto-complete with conditions" {
    const allocator = std.testing.allocator;

    var milestone = try Milestone.init(allocator, "ms-1", "Auto Milestone", "case-1");
    defer milestone.deinit();

    const cond = try MilestoneCondition.withFieldEquals(allocator, "approval", "yes");
    try milestone.addCondition(cond);

    var context = EvaluationContext.init(allocator);
    defer context.deinit();

    try context.setField("approval", "no");
    try std.testing.expect(!milestone.checkAutoComplete(&context));

    try context.setField("approval", "yes");
    try std.testing.expect(milestone.checkAutoComplete(&context));
}

