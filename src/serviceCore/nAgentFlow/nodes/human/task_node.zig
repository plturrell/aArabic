//! Human Task Node for nWorkflow
//! Implements human-in-the-loop workflow tasks with forms, assignments, and deadlines
//!
//! This module provides:
//! - TaskPriority: Priority levels for human tasks
//! - TaskState: State machine states for task lifecycle
//! - TaskEvent: Events that trigger state transitions
//! - TaskAssignment: Assignment rules (user, group, role, pool)
//! - TaskDeadline: Deadline configuration with escalation
//! - HumanTask: Full task definition with state management
//! - HumanTaskNode: Workflow node that creates human tasks
//! - TaskManager: Manages all human tasks with indexing

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Task Priority
// ============================================================================

/// Task Priority levels
pub const TaskPriority = enum {
    LOW,
    NORMAL,
    HIGH,
    URGENT,

    pub fn toString(self: TaskPriority) []const u8 {
        return switch (self) {
            .LOW => "LOW",
            .NORMAL => "NORMAL",
            .HIGH => "HIGH",
            .URGENT => "URGENT",
        };
    }

    pub fn fromString(str: []const u8) ?TaskPriority {
        if (std.mem.eql(u8, str, "LOW")) return .LOW;
        if (std.mem.eql(u8, str, "NORMAL")) return .NORMAL;
        if (std.mem.eql(u8, str, "HIGH")) return .HIGH;
        if (std.mem.eql(u8, str, "URGENT")) return .URGENT;
        return null;
    }

    pub fn toInt(self: TaskPriority) u8 {
        return switch (self) {
            .LOW => 1,
            .NORMAL => 2,
            .HIGH => 3,
            .URGENT => 4,
        };
    }
};

// ============================================================================
// Task State (similar to Appian task states)
// ============================================================================

/// Task State for state machine
pub const TaskState = enum {
    PENDING, // Created but not assigned
    ASSIGNED, // Assigned to user/group
    CLAIMED, // Claimed by specific user
    IN_PROGRESS, // User working on it
    COMPLETED, // Successfully finished
    REJECTED, // Rejected by assignee
    ESCALATED, // Escalated to manager
    CANCELLED, // Cancelled by system/admin
    EXPIRED, // Deadline passed

    pub fn toString(self: TaskState) []const u8 {
        return @tagName(self);
    }

    pub fn isTerminal(self: TaskState) bool {
        return switch (self) {
            .COMPLETED, .CANCELLED, .EXPIRED => true,
            else => false,
        };
    }

    pub fn isActive(self: TaskState) bool {
        return switch (self) {
            .PENDING, .ASSIGNED, .CLAIMED, .IN_PROGRESS, .ESCALATED, .REJECTED => true,
            else => false,
        };
    }
};

// ============================================================================
// Task Event for state transitions
// ============================================================================

/// Task Event for state transitions
pub const TaskEvent = enum {
    ASSIGN,
    CLAIM,
    START,
    COMPLETE,
    REJECT,
    ESCALATE,
    CANCEL,
    EXPIRE,
    REASSIGN,
    RELEASE, // Release a claimed task back to pool

    pub fn toString(self: TaskEvent) []const u8 {
        return @tagName(self);
    }
};

// ============================================================================
// Task Assignment - who can work on the task
// ============================================================================

/// Task Assignment - who can work on the task
pub const TaskAssignment = struct {
    user_id: ?[]const u8 = null,
    group_id: ?[]const u8 = null,
    role_id: ?[]const u8 = null,
    pool_users: ?[]const []const u8 = null,

    pub fn isUserAssignment(self: TaskAssignment) bool {
        return self.user_id != null;
    }

    pub fn isGroupAssignment(self: TaskAssignment) bool {
        return self.group_id != null or self.role_id != null;
    }

    pub fn isPoolAssignment(self: TaskAssignment) bool {
        return self.pool_users != null and self.pool_users.?.len > 0;
    }

    pub fn isEmpty(self: TaskAssignment) bool {
        return self.user_id == null and self.group_id == null and self.role_id == null and (self.pool_users == null or self.pool_users.?.len == 0);
    }

    pub fn canUserClaim(self: TaskAssignment, user_id: []const u8) bool {
        // Direct user assignment
        if (self.user_id) |uid| {
            if (std.mem.eql(u8, uid, user_id)) return true;
        }
        // Pool assignment - check if user is in pool
        if (self.pool_users) |pool| {
            for (pool) |pool_user| {
                if (std.mem.eql(u8, pool_user, user_id)) return true;
            }
        }
        // Group/role assignments would need external resolution
        return self.group_id != null or self.role_id != null;
    }
};

// ============================================================================
// Task Deadline configuration
// ============================================================================

/// Task Deadline configuration
pub const TaskDeadline = struct {
    due_date: i64, // Unix timestamp
    warning_threshold_hours: u32 = 24, // Hours before due to warn
    auto_escalate: bool = false,
    escalate_to: ?[]const u8 = null, // User/group to escalate to
    auto_reassign: bool = false,
    reassign_to: ?[]const u8 = null,

    pub fn isOverdue(self: TaskDeadline) bool {
        return std.time.timestamp() > self.due_date;
    }

    pub fn isWarning(self: TaskDeadline) bool {
        const warning_time = self.due_date - @as(i64, @intCast(self.warning_threshold_hours)) * 3600;
        const now = std.time.timestamp();
        return now >= warning_time and now <= self.due_date;
    }

    pub fn remainingSeconds(self: TaskDeadline) i64 {
        return self.due_date - std.time.timestamp();
    }

    pub fn remainingHours(self: TaskDeadline) i64 {
        return @divFloor(self.remainingSeconds(), 3600);
    }
};

// ============================================================================
// Human Task definition
// ============================================================================

/// Human Task definition
pub const HumanTask = struct {
    id: []const u8,
    name: []const u8,
    description: ?[]const u8 = null,
    workflow_id: []const u8,
    node_id: []const u8,

    state: TaskState = .PENDING,
    priority: TaskPriority = .NORMAL,

    assignment: TaskAssignment,
    claimed_by: ?[]const u8 = null,

    deadline: ?TaskDeadline = null,

    form_id: ?[]const u8 = null, // Reference to form definition
    form_data: ?[]const u8 = null, // JSON form input data
    output_data: ?[]const u8 = null, // JSON output after completion

    created_at: i64,
    updated_at: i64,
    started_at: ?i64 = null,
    completed_at: ?i64 = null,

    parent_task_id: ?[]const u8 = null, // For subtasks
    metadata: ?[]const u8 = null, // Additional JSON metadata

    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        workflow_id: []const u8,
        node_id: []const u8,
        assignment: TaskAssignment,
    ) !HumanTask {
        const now = std.time.timestamp();
        return HumanTask{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .workflow_id = try allocator.dupe(u8, workflow_id),
            .node_id = try allocator.dupe(u8, node_id),
            .assignment = assignment,
            .created_at = now,
            .updated_at = now,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HumanTask) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.description) |d| self.allocator.free(d);
        self.allocator.free(self.workflow_id);
        self.allocator.free(self.node_id);
        if (self.claimed_by) |c| self.allocator.free(c);
        if (self.form_id) |f| self.allocator.free(f);
        if (self.form_data) |fd| self.allocator.free(fd);
        if (self.output_data) |od| self.allocator.free(od);
        if (self.parent_task_id) |p| self.allocator.free(p);
        if (self.metadata) |m| self.allocator.free(m);
    }

    /// State transition function - returns new state or null if invalid
    pub fn transition(current: TaskState, event: TaskEvent) ?TaskState {
        return switch (current) {
            .PENDING => switch (event) {
                .ASSIGN => .ASSIGNED,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .ASSIGNED => switch (event) {
                .CLAIM => .CLAIMED,
                .REASSIGN => .ASSIGNED,
                .CANCEL => .CANCELLED,
                .EXPIRE => .EXPIRED,
                .ESCALATE => .ESCALATED,
                else => null,
            },
            .CLAIMED => switch (event) {
                .START => .IN_PROGRESS,
                .REJECT => .REJECTED,
                .REASSIGN => .ASSIGNED,
                .RELEASE => .ASSIGNED,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .IN_PROGRESS => switch (event) {
                .COMPLETE => .COMPLETED,
                .REJECT => .REJECTED,
                .ESCALATE => .ESCALATED,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .ESCALATED => switch (event) {
                .ASSIGN => .ASSIGNED,
                .CANCEL => .CANCELLED,
                else => null,
            },
            .REJECTED => switch (event) {
                .REASSIGN => .ASSIGNED,
                .CANCEL => .CANCELLED,
                else => null,
            },
            else => null, // Terminal states
        };
    }

    pub fn canTransition(self: *const HumanTask, event: TaskEvent) bool {
        return transition(self.state, event) != null;
    }

    pub fn applyEvent(self: *HumanTask, event: TaskEvent) !void {
        const new_state = transition(self.state, event) orelse return error.InvalidTransition;
        self.state = new_state;
        self.updated_at = std.time.timestamp();

        if (event == .START) {
            self.started_at = self.updated_at;
        } else if (event == .COMPLETE) {
            self.completed_at = self.updated_at;
        }
    }

    pub fn claim(self: *HumanTask, user_id: []const u8) !void {
        if (self.state != .ASSIGNED) return error.TaskNotAssigned;
        if (self.claimed_by) |old| self.allocator.free(old);
        self.claimed_by = try self.allocator.dupe(u8, user_id);
        try self.applyEvent(.CLAIM);
    }

    pub fn start(self: *HumanTask) !void {
        if (self.state != .CLAIMED) return error.TaskNotClaimed;
        try self.applyEvent(.START);
    }

    pub fn complete(self: *HumanTask, output_data: []const u8) !void {
        if (self.state != .IN_PROGRESS) return error.TaskNotInProgress;
        if (self.output_data) |old| self.allocator.free(old);
        self.output_data = try self.allocator.dupe(u8, output_data);
        try self.applyEvent(.COMPLETE);
    }

    pub fn reject(self: *HumanTask, reason: ?[]const u8) !void {
        if (self.state != .CLAIMED and self.state != .IN_PROGRESS) return error.TaskNotClaimedOrInProgress;
        if (reason) |r| {
            if (self.metadata) |old| self.allocator.free(old);
            self.metadata = try self.allocator.dupe(u8, r);
        }
        try self.applyEvent(.REJECT);
    }

    pub fn isOverdue(self: *const HumanTask) bool {
        if (self.deadline) |deadline| {
            return deadline.isOverdue();
        }
        return false;
    }

    pub fn setDescription(self: *HumanTask, description: []const u8) !void {
        if (self.description) |old| self.allocator.free(old);
        self.description = try self.allocator.dupe(u8, description);
        self.updated_at = std.time.timestamp();
    }

    pub fn setFormId(self: *HumanTask, form_id: []const u8) !void {
        if (self.form_id) |old| self.allocator.free(old);
        self.form_id = try self.allocator.dupe(u8, form_id);
        self.updated_at = std.time.timestamp();
    }

    pub fn setFormData(self: *HumanTask, form_data: []const u8) !void {
        if (self.form_data) |old| self.allocator.free(old);
        self.form_data = try self.allocator.dupe(u8, form_data);
        self.updated_at = std.time.timestamp();
    }

    pub fn toJson(self: *const HumanTask, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(allocator);

        const writer = buffer.writer(allocator);
        try writer.print(
            \\{{"id":"{s}","name":"{s}","state":"{s}","priority":"{s}","workflow_id":"{s}","node_id":"{s}","created_at":{d},"updated_at":{d}
        , .{
            self.id,
            self.name,
            self.state.toString(),
            self.priority.toString(),
            self.workflow_id,
            self.node_id,
            self.created_at,
            self.updated_at,
        });

        if (self.claimed_by) |cb| {
            try writer.print(
                \\,"claimed_by":"{s}"
            , .{cb});
        }
        if (self.started_at) |sa| {
            try writer.print(
                \\,"started_at":{d}
            , .{sa});
        }
        if (self.completed_at) |ca| {
            try writer.print(
                \\,"completed_at":{d}
            , .{ca});
        }

        try writer.writeAll("}}");

        return buffer.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Human Task Node - workflow node that creates human tasks
// ============================================================================

/// Human Task Node - workflow node that creates human tasks
pub const HumanTaskNode = struct {
    id: []const u8,
    name: []const u8,
    task_name_template: []const u8,
    description_template: ?[]const u8 = null,

    default_assignment: TaskAssignment,
    default_priority: TaskPriority = .NORMAL,
    default_deadline_hours: ?u32 = null,

    form_id: ?[]const u8 = null,
    input_mapping: ?[]const u8 = null, // JSON path mapping
    output_mapping: ?[]const u8 = null,

    on_complete_node: ?[]const u8 = null,
    on_reject_node: ?[]const u8 = null,
    on_timeout_node: ?[]const u8 = null,

    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        task_name_template: []const u8,
        assignment: TaskAssignment,
    ) !HumanTaskNode {
        return HumanTaskNode{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .task_name_template = try allocator.dupe(u8, task_name_template),
            .default_assignment = assignment,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HumanTaskNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.task_name_template);
        if (self.description_template) |d| self.allocator.free(d);
        if (self.form_id) |f| self.allocator.free(f);
        if (self.input_mapping) |i| self.allocator.free(i);
        if (self.output_mapping) |o| self.allocator.free(o);
        if (self.on_complete_node) |c| self.allocator.free(c);
        if (self.on_reject_node) |r| self.allocator.free(r);
        if (self.on_timeout_node) |t| self.allocator.free(t);
    }

    pub fn setDescriptionTemplate(self: *HumanTaskNode, template: []const u8) !void {
        if (self.description_template) |old| self.allocator.free(old);
        self.description_template = try self.allocator.dupe(u8, template);
    }

    pub fn setFormId(self: *HumanTaskNode, form_id: []const u8) !void {
        if (self.form_id) |old| self.allocator.free(old);
        self.form_id = try self.allocator.dupe(u8, form_id);
    }

    pub fn createTask(
        self: *const HumanTaskNode,
        allocator: Allocator,
        task_id: []const u8,
        workflow_id: []const u8,
    ) !HumanTask {
        var task = try HumanTask.init(
            allocator,
            task_id,
            self.task_name_template,
            workflow_id,
            self.id,
            self.default_assignment,
        );
        task.priority = self.default_priority;

        if (self.form_id) |fid| {
            try task.setFormId(fid);
        }

        if (self.default_deadline_hours) |hours| {
            task.deadline = TaskDeadline{
                .due_date = std.time.timestamp() + @as(i64, @intCast(hours)) * 3600,
            };
        }

        return task;
    }
};

// ============================================================================
// Task Manager - manages all human tasks
// ============================================================================

/// Task Manager - manages all human tasks
pub const TaskManager = struct {
    tasks: std.StringHashMap(*HumanTask),
    tasks_by_workflow: std.StringHashMap(std.ArrayList([]const u8)),
    tasks_by_user: std.StringHashMap(std.ArrayList([]const u8)),
    task_counter: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) TaskManager {
        return TaskManager{
            .tasks = std.StringHashMap(*HumanTask).init(allocator),
            .tasks_by_workflow = std.StringHashMap(std.ArrayList([]const u8)){},
            .tasks_by_user = std.StringHashMap(std.ArrayList([]const u8)){},
            .task_counter = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TaskManager) void {
        // Free all tasks and their keys (keys are shared with workflow lists)
        var task_iter = self.tasks.iterator();
        while (task_iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.tasks.deinit();

        // Free workflow index lists (strings already freed via tasks hashmap keys)
        var wf_iter = self.tasks_by_workflow.valueIterator();
        while (wf_iter.next()) |list_ptr| {
            // Don't free items - they're the same pointers as task keys
            list_ptr.deinit(self.allocator);
        }
        self.tasks_by_workflow.deinit();

        // Free user index lists and their string contents (separate allocations)
        var user_iter = self.tasks_by_user.valueIterator();
        while (user_iter.next()) |list_ptr| {
            for (list_ptr.items) |id| {
                self.allocator.free(id);
            }
            list_ptr.deinit(self.allocator);
        }
        self.tasks_by_user.deinit();
    }

    fn generateTaskId(self: *TaskManager) ![]const u8 {
        self.task_counter += 1;
        const timestamp = std.time.timestamp();
        var buf: [64]u8 = undefined;
        const len = std.fmt.bufPrint(&buf, "task-{d}-{d}", .{ timestamp, self.task_counter }) catch return error.BufferTooSmall;
        return try self.allocator.dupe(u8, len);
    }

    pub fn createTask(self: *TaskManager, node: *const HumanTaskNode, workflow_id: []const u8) !*HumanTask {
        const task_id = try self.generateTaskId();
        defer self.allocator.free(task_id);

        const task = try self.allocator.create(HumanTask);
        errdefer self.allocator.destroy(task);

        task.* = try node.createTask(self.allocator, task_id, workflow_id);

        // Store task
        const id_copy = try self.allocator.dupe(u8, task.id);
        try self.tasks.put(id_copy, task);

        // Index by workflow
        const gop = try self.tasks_by_workflow.getOrPut(workflow_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList([]const u8){};
        }
        try gop.value_ptr.append(self.allocator, id_copy);

        return task;
    }

    pub fn createTaskWithId(
        self: *TaskManager,
        node: *const HumanTaskNode,
        workflow_id: []const u8,
        task_id: []const u8,
    ) !*HumanTask {
        const task = try self.allocator.create(HumanTask);
        errdefer self.allocator.destroy(task);

        task.* = try node.createTask(self.allocator, task_id, workflow_id);

        // Store task
        const id_copy = try self.allocator.dupe(u8, task.id);
        try self.tasks.put(id_copy, task);

        // Index by workflow
        const gop = try self.tasks_by_workflow.getOrPut(workflow_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList([]const u8){};
        }
        try gop.value_ptr.append(self.allocator, id_copy);

        return task;
    }

    pub fn getTask(self: *TaskManager, task_id: []const u8) ?*HumanTask {
        return self.tasks.get(task_id);
    }

    pub fn getTasksForWorkflow(self: *TaskManager, workflow_id: []const u8) ?[]const []const u8 {
        if (self.tasks_by_workflow.get(workflow_id)) |list| {
            return list.items;
        }
        return null;
    }

    pub fn getTasksForUser(self: *TaskManager, user_id: []const u8) ?[]const []const u8 {
        if (self.tasks_by_user.get(user_id)) |list| {
            return list.items;
        }
        return null;
    }

    pub fn assignTask(self: *TaskManager, task_id: []const u8) !void {
        const task = self.tasks.get(task_id) orelse return error.TaskNotFound;
        try task.applyEvent(.ASSIGN);
    }

    pub fn claimTask(self: *TaskManager, task_id: []const u8, user_id: []const u8) !void {
        const task = self.tasks.get(task_id) orelse return error.TaskNotFound;
        try task.claim(user_id);

        // Index by user
        const gop = try self.tasks_by_user.getOrPut(user_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList([]const u8){};
        }
        const id_copy = try self.allocator.dupe(u8, task_id);
        try gop.value_ptr.append(self.allocator, id_copy);
    }

    pub fn startTask(self: *TaskManager, task_id: []const u8) !void {
        const task = self.tasks.get(task_id) orelse return error.TaskNotFound;
        try task.start();
    }

    pub fn completeTask(self: *TaskManager, task_id: []const u8, output_data: []const u8) !void {
        const task = self.tasks.get(task_id) orelse return error.TaskNotFound;
        try task.complete(output_data);
    }

    pub fn rejectTask(self: *TaskManager, task_id: []const u8, reason: ?[]const u8) !void {
        const task = self.tasks.get(task_id) orelse return error.TaskNotFound;
        try task.reject(reason);
    }

    pub fn cancelTask(self: *TaskManager, task_id: []const u8) !void {
        const task = self.tasks.get(task_id) orelse return error.TaskNotFound;
        try task.applyEvent(.CANCEL);
    }

    pub fn getPendingTaskCount(self: *TaskManager) usize {
        var count: usize = 0;
        var iter = self.tasks.valueIterator();
        while (iter.next()) |task_ptr| {
            if (task_ptr.*.state.isActive()) {
                count += 1;
            }
        }
        return count;
    }

    pub fn getOverdueTasks(self: *TaskManager, allocator: Allocator) !std.ArrayList(*HumanTask) {
        var overdue = std.ArrayList(*HumanTask){};
        errdefer overdue.deinit(allocator);

        var iter = self.tasks.valueIterator();
        while (iter.next()) |task_ptr| {
            if (task_ptr.*.state.isActive() and task_ptr.*.isOverdue()) {
                try overdue.append(allocator, task_ptr.*);
            }
        }
        return overdue;
    }

    pub fn getTasksByState(self: *TaskManager, state: TaskState, allocator: Allocator) !std.ArrayList(*HumanTask) {
        var result = std.ArrayList(*HumanTask){};
        errdefer result.deinit(allocator);

        var iter = self.tasks.valueIterator();
        while (iter.next()) |task_ptr| {
            if (task_ptr.*.state == state) {
                try result.append(allocator, task_ptr.*);
            }
        }
        return result;
    }

    pub fn taskCount(self: *TaskManager) usize {
        return self.tasks.count();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "TaskPriority conversions" {
    try std.testing.expectEqualStrings("HIGH", TaskPriority.HIGH.toString());
    try std.testing.expectEqual(TaskPriority.HIGH, TaskPriority.fromString("HIGH").?);
    try std.testing.expectEqual(@as(u8, 3), TaskPriority.HIGH.toInt());
}

test "TaskState properties" {
    try std.testing.expect(TaskState.COMPLETED.isTerminal());
    try std.testing.expect(TaskState.CANCELLED.isTerminal());
    try std.testing.expect(!TaskState.PENDING.isTerminal());

    try std.testing.expect(TaskState.PENDING.isActive());
    try std.testing.expect(TaskState.IN_PROGRESS.isActive());
    try std.testing.expect(!TaskState.COMPLETED.isActive());
}

test "TaskAssignment checks" {
    const user_assignment = TaskAssignment{ .user_id = "user-123" };
    try std.testing.expect(user_assignment.isUserAssignment());
    try std.testing.expect(!user_assignment.isGroupAssignment());

    const group_assignment = TaskAssignment{ .group_id = "approvers" };
    try std.testing.expect(!group_assignment.isUserAssignment());
    try std.testing.expect(group_assignment.isGroupAssignment());

    const empty = TaskAssignment{};
    try std.testing.expect(empty.isEmpty());
}

test "HumanTask state transitions" {
    const allocator = std.testing.allocator;

    const assignment = TaskAssignment{ .group_id = "approvers" };
    var task = try HumanTask.init(allocator, "task-1", "Review Request", "wf-1", "node-1", assignment);
    defer task.deinit();

    try std.testing.expectEqual(TaskState.PENDING, task.state);

    // PENDING -> ASSIGNED
    try task.applyEvent(.ASSIGN);
    try std.testing.expectEqual(TaskState.ASSIGNED, task.state);

    // ASSIGNED -> CLAIMED
    try task.claim("user-123");
    try std.testing.expectEqual(TaskState.CLAIMED, task.state);
    try std.testing.expectEqualStrings("user-123", task.claimed_by.?);

    // CLAIMED -> IN_PROGRESS
    try task.start();
    try std.testing.expectEqual(TaskState.IN_PROGRESS, task.state);
    try std.testing.expect(task.started_at != null);

    // IN_PROGRESS -> COMPLETED
    try task.complete("{}");
    try std.testing.expectEqual(TaskState.COMPLETED, task.state);
    try std.testing.expect(task.completed_at != null);
}

test "HumanTask invalid transitions" {
    const allocator = std.testing.allocator;

    const assignment = TaskAssignment{ .user_id = "user-1" };
    var task = try HumanTask.init(allocator, "task-1", "Test", "wf-1", "node-1", assignment);
    defer task.deinit();

    // Can't claim from PENDING
    try std.testing.expectError(error.TaskNotAssigned, task.claim("user-1"));

    // Can't start from PENDING
    try std.testing.expectError(error.TaskNotClaimed, task.start());

    // Can't complete from PENDING
    try std.testing.expectError(error.TaskNotInProgress, task.complete("{}"));
}

test "HumanTask reject" {
    const allocator = std.testing.allocator;

    const assignment = TaskAssignment{ .user_id = "user-1" };
    var task = try HumanTask.init(allocator, "task-1", "Test", "wf-1", "node-1", assignment);
    defer task.deinit();

    try task.applyEvent(.ASSIGN);
    try task.claim("user-1");
    try task.reject("Not my job");

    try std.testing.expectEqual(TaskState.REJECTED, task.state);
}

test "HumanTaskNode creates task" {
    const allocator = std.testing.allocator;

    const assignment = TaskAssignment{ .role_id = "manager" };
    var node = try HumanTaskNode.init(allocator, "node-1", "Approval Node", "Approve Request: ${request_id}", assignment);
    defer node.deinit();

    node.default_priority = .HIGH;
    node.default_deadline_hours = 48;

    var task = try node.createTask(allocator, "task-1", "wf-1");
    defer task.deinit();

    try std.testing.expectEqualStrings("task-1", task.id);
    try std.testing.expectEqualStrings("wf-1", task.workflow_id);
    try std.testing.expectEqualStrings("node-1", task.node_id);
    try std.testing.expectEqual(TaskPriority.HIGH, task.priority);
    try std.testing.expect(task.deadline != null);
}

test "TaskManager operations" {
    const allocator = std.testing.allocator;

    var manager = TaskManager.init(allocator);
    defer manager.deinit();

    const assignment = TaskAssignment{ .user_id = "user-1" };
    var node = try HumanTaskNode.init(allocator, "node-1", "Task Node", "Review", assignment);
    defer node.deinit();

    // Create task
    const task = try manager.createTaskWithId(&node, "wf-1", "task-001");
    try std.testing.expectEqual(TaskState.PENDING, task.state);
    try std.testing.expectEqual(@as(usize, 1), manager.taskCount());

    // Assign
    try manager.assignTask("task-001");
    try std.testing.expectEqual(TaskState.ASSIGNED, task.state);

    // Claim
    try manager.claimTask("task-001", "user-1");
    try std.testing.expectEqual(TaskState.CLAIMED, task.state);

    // Start
    try manager.startTask("task-001");
    try std.testing.expectEqual(TaskState.IN_PROGRESS, task.state);

    // Complete
    try manager.completeTask("task-001", "{}");
    try std.testing.expectEqual(TaskState.COMPLETED, task.state);

    // No longer active
    try std.testing.expectEqual(@as(usize, 0), manager.getPendingTaskCount());
}

test "TaskManager multiple tasks" {
    const allocator = std.testing.allocator;

    var manager = TaskManager.init(allocator);
    defer manager.deinit();

    const assignment = TaskAssignment{ .group_id = "reviewers" };
    var node = try HumanTaskNode.init(allocator, "node-1", "Review Node", "Review Item", assignment);
    defer node.deinit();

    // Create multiple tasks
    _ = try manager.createTaskWithId(&node, "wf-1", "task-001");
    _ = try manager.createTaskWithId(&node, "wf-1", "task-002");
    _ = try manager.createTaskWithId(&node, "wf-2", "task-003");

    try std.testing.expectEqual(@as(usize, 3), manager.taskCount());
    try std.testing.expectEqual(@as(usize, 3), manager.getPendingTaskCount());

    // Get tasks by workflow
    const wf1_tasks = manager.getTasksForWorkflow("wf-1");
    try std.testing.expect(wf1_tasks != null);
    try std.testing.expectEqual(@as(usize, 2), wf1_tasks.?.len);

    const wf2_tasks = manager.getTasksForWorkflow("wf-2");
    try std.testing.expect(wf2_tasks != null);
    try std.testing.expectEqual(@as(usize, 1), wf2_tasks.?.len);
}

test "TaskManager task not found" {
    const allocator = std.testing.allocator;

    var manager = TaskManager.init(allocator);
    defer manager.deinit();

    try std.testing.expectError(error.TaskNotFound, manager.assignTask("nonexistent"));
    try std.testing.expectError(error.TaskNotFound, manager.claimTask("nonexistent", "user"));
    try std.testing.expectError(error.TaskNotFound, manager.completeTask("nonexistent", "{}"));
}

test "HumanTask toJson" {
    const allocator = std.testing.allocator;

    const assignment = TaskAssignment{ .user_id = "user-1" };
    var task = try HumanTask.init(allocator, "task-1", "Test Task", "wf-1", "node-1", assignment);
    defer task.deinit();

    const json = try task.toJson(allocator);
    defer allocator.free(json);

    // Check that JSON contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, json, "\"id\":\"task-1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"Test Task\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"state\":\"PENDING\"") != null);
}

