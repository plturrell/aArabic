//! Approval Node for nWorkflow
//!
//! This module provides multi-level approval capabilities:
//! - ApprovalType: Types of approval workflows (SINGLE, SEQUENTIAL, PARALLEL, etc.)
//! - ApprovalStep: Individual approval step in a chain
//! - ApprovalChain: Chain of approval steps
//! - ApprovalNode: Workflow node that extends HumanTaskNode for approvals
//! - ApprovalManager: Manages multi-level approval workflows

const std = @import("std");
const Allocator = std.mem.Allocator;
const task_node = @import("task_node.zig");
const HumanTask = task_node.HumanTask;
const HumanTaskNode = task_node.HumanTaskNode;
const TaskAssignment = task_node.TaskAssignment;
const TaskState = task_node.TaskState;
const TaskPriority = task_node.TaskPriority;
const TaskManager = task_node.TaskManager;

// ============================================================================
// Approval Type
// ============================================================================

/// Approval Type - defines how approvals are processed
pub const ApprovalType = enum {
    SINGLE, // Single approver
    SEQUENTIAL, // Approvers in sequence
    PARALLEL, // All approvers at once
    UNANIMOUS, // All must approve
    MAJORITY, // Majority must approve
    FIRST_RESPONSE, // First response wins

    pub fn toString(self: ApprovalType) []const u8 {
        return @tagName(self);
    }

    pub fn fromString(str: []const u8) ?ApprovalType {
        inline for (std.meta.fields(ApprovalType)) |field| {
            if (std.mem.eql(u8, str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }

    pub fn requiresAllApprovers(self: ApprovalType) bool {
        return switch (self) {
            .UNANIMOUS => true,
            else => false,
        };
    }

    pub fn isParallel(self: ApprovalType) bool {
        return switch (self) {
            .PARALLEL, .UNANIMOUS, .MAJORITY, .FIRST_RESPONSE => true,
            else => false,
        };
    }
};

// ============================================================================
// Approval Decision
// ============================================================================

/// Approval Decision
pub const ApprovalDecision = enum {
    PENDING,
    APPROVED,
    REJECTED,
    ABSTAINED,

    pub fn toString(self: ApprovalDecision) []const u8 {
        return @tagName(self);
    }

    pub fn isComplete(self: ApprovalDecision) bool {
        return self != .PENDING;
    }
};

// ============================================================================
// Approval Step
// ============================================================================

/// Approval Step - individual step in an approval chain
pub const ApprovalStep = struct {
    id: []const u8,
    name: []const u8,
    order: u32,
    assignment: TaskAssignment,
    decision: ApprovalDecision = .PENDING,
    decided_by: ?[]const u8 = null,
    decided_at: ?i64 = null,
    comments: ?[]const u8 = null,
    task_id: ?[]const u8 = null, // Reference to created task
    required: bool = true, // If false, can be skipped
    timeout_hours: ?u32 = null,
    auto_approve_on_timeout: bool = false,

    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        order: u32,
        assignment: TaskAssignment,
    ) !ApprovalStep {
        return ApprovalStep{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .order = order,
            .assignment = assignment,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ApprovalStep) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.decided_by) |d| self.allocator.free(d);
        if (self.comments) |c| self.allocator.free(c);
        if (self.task_id) |t| self.allocator.free(t);
    }

    pub fn approve(self: *ApprovalStep, user_id: []const u8, comments: ?[]const u8) !void {
        if (self.decided_by) |old| self.allocator.free(old);
        self.decided_by = try self.allocator.dupe(u8, user_id);
        self.decided_at = std.time.timestamp();
        self.decision = .APPROVED;
        if (comments) |c| {
            if (self.comments) |old| self.allocator.free(old);
            self.comments = try self.allocator.dupe(u8, c);
        }
    }

    pub fn reject(self: *ApprovalStep, user_id: []const u8, comments: ?[]const u8) !void {
        if (self.decided_by) |old| self.allocator.free(old);
        self.decided_by = try self.allocator.dupe(u8, user_id);
        self.decided_at = std.time.timestamp();
        self.decision = .REJECTED;
        if (comments) |c| {
            if (self.comments) |old| self.allocator.free(old);
            self.comments = try self.allocator.dupe(u8, c);
        }
    }

    pub fn isPending(self: *const ApprovalStep) bool {
        return self.decision == .PENDING;
    }

    pub fn isApproved(self: *const ApprovalStep) bool {
        return self.decision == .APPROVED;
    }

    pub fn isRejected(self: *const ApprovalStep) bool {
        return self.decision == .REJECTED;
    }
};

// ============================================================================
// Approval Chain
// ============================================================================

/// Approval Chain - chain of approval steps
pub const ApprovalChain = struct {
    id: []const u8,
    name: []const u8,
    approval_type: ApprovalType,
    steps: std.ArrayList(*ApprovalStep),
    current_step_index: usize = 0,
    created_at: i64,
    completed_at: ?i64 = null,
    final_decision: ApprovalDecision = .PENDING,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        approval_type: ApprovalType,
    ) !ApprovalChain {
        return ApprovalChain{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .approval_type = approval_type,
            .steps = std.ArrayList(*ApprovalStep){},
            .created_at = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ApprovalChain) void {
        for (self.steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.steps.deinit(self.allocator);
        self.allocator.free(self.id);
        self.allocator.free(self.name);
    }

    pub fn addStep(self: *ApprovalChain, step: *ApprovalStep) !void {
        try self.steps.append(self.allocator, step);
    }

    pub fn createAndAddStep(
        self: *ApprovalChain,
        id: []const u8,
        name: []const u8,
        assignment: TaskAssignment,
    ) !*ApprovalStep {
        const step = try self.allocator.create(ApprovalStep);
        errdefer self.allocator.destroy(step);

        const order: u32 = @intCast(self.steps.items.len);
        step.* = try ApprovalStep.init(self.allocator, id, name, order, assignment);
        try self.steps.append(self.allocator, step);
        return step;
    }

    pub fn getCurrentStep(self: *ApprovalChain) ?*ApprovalStep {
        if (self.current_step_index < self.steps.items.len) {
            return self.steps.items[self.current_step_index];
        }
        return null;
    }

    pub fn getStep(self: *ApprovalChain, step_id: []const u8) ?*ApprovalStep {
        for (self.steps.items) |step| {
            if (std.mem.eql(u8, step.id, step_id)) {
                return step;
            }
        }
        return null;
    }

    pub fn stepCount(self: *const ApprovalChain) usize {
        return self.steps.items.len;
    }

    pub fn completedStepCount(self: *const ApprovalChain) usize {
        var count: usize = 0;
        for (self.steps.items) |step| {
            if (step.decision.isComplete()) count += 1;
        }
        return count;
    }

    pub fn approvedStepCount(self: *const ApprovalChain) usize {
        var count: usize = 0;
        for (self.steps.items) |step| {
            if (step.isApproved()) count += 1;
        }
        return count;
    }

    pub fn rejectedStepCount(self: *const ApprovalChain) usize {
        var count: usize = 0;
        for (self.steps.items) |step| {
            if (step.isRejected()) count += 1;
        }
        return count;
    }

    /// Process a decision and update chain state
    pub fn processDecision(self: *ApprovalChain, step_id: []const u8, decision: ApprovalDecision, user_id: []const u8, comments: ?[]const u8) !void {
        const step = self.getStep(step_id) orelse return error.StepNotFound;

        switch (decision) {
            .APPROVED => try step.approve(user_id, comments),
            .REJECTED => try step.reject(user_id, comments),
            else => return error.InvalidDecision,
        }

        // Update chain state based on approval type
        try self.updateChainState();
    }

    fn updateChainState(self: *ApprovalChain) !void {
        switch (self.approval_type) {
            .SINGLE => {
                if (self.steps.items.len > 0) {
                    const step = self.steps.items[0];
                    if (step.decision.isComplete()) {
                        self.final_decision = step.decision;
                        self.completed_at = std.time.timestamp();
                    }
                }
            },
            .SEQUENTIAL => {
                // Check if current step is complete
                if (self.getCurrentStep()) |current| {
                    if (current.isRejected()) {
                        self.final_decision = .REJECTED;
                        self.completed_at = std.time.timestamp();
                    } else if (current.isApproved()) {
                        self.current_step_index += 1;
                        if (self.current_step_index >= self.steps.items.len) {
                            self.final_decision = .APPROVED;
                            self.completed_at = std.time.timestamp();
                        }
                    }
                }
            },
            .PARALLEL, .FIRST_RESPONSE => {
                // First response wins
                for (self.steps.items) |step| {
                    if (step.decision.isComplete()) {
                        self.final_decision = step.decision;
                        self.completed_at = std.time.timestamp();
                        break;
                    }
                }
            },
            .UNANIMOUS => {
                // All must approve
                const rejected = self.rejectedStepCount();
                if (rejected > 0) {
                    self.final_decision = .REJECTED;
                    self.completed_at = std.time.timestamp();
                } else if (self.approvedStepCount() == self.steps.items.len) {
                    self.final_decision = .APPROVED;
                    self.completed_at = std.time.timestamp();
                }
            },
            .MAJORITY => {
                // Majority must approve
                const total = self.steps.items.len;
                const approved = self.approvedStepCount();
                const rejected = self.rejectedStepCount();
                const majority = (total / 2) + 1;

                if (approved >= majority) {
                    self.final_decision = .APPROVED;
                    self.completed_at = std.time.timestamp();
                } else if (rejected >= majority) {
                    self.final_decision = .REJECTED;
                    self.completed_at = std.time.timestamp();
                }
            },
        }
    }

    pub fn isComplete(self: *const ApprovalChain) bool {
        return self.final_decision != .PENDING;
    }

    pub fn isApproved(self: *const ApprovalChain) bool {
        return self.final_decision == .APPROVED;
    }

    pub fn isRejected(self: *const ApprovalChain) bool {
        return self.final_decision == .REJECTED;
    }
};

// ============================================================================
// Approval Node - extends HumanTaskNode for approvals
// ============================================================================

/// Approval Node - workflow node for approval tasks
pub const ApprovalNode = struct {
    id: []const u8,
    name: []const u8,
    approval_type: ApprovalType,
    step_definitions: std.ArrayList(StepDefinition),
    on_approved_node: ?[]const u8 = null,
    on_rejected_node: ?[]const u8 = null,
    allocator: Allocator,

    pub const StepDefinition = struct {
        id: []const u8,
        name: []const u8,
        assignment: TaskAssignment,
        timeout_hours: ?u32 = null,
    };

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        approval_type: ApprovalType,
    ) !ApprovalNode {
        return ApprovalNode{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .approval_type = approval_type,
            .step_definitions = std.ArrayList(StepDefinition){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ApprovalNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        for (self.step_definitions.items) |def| {
            self.allocator.free(def.id);
            self.allocator.free(def.name);
        }
        self.step_definitions.deinit(self.allocator);
        if (self.on_approved_node) |n| self.allocator.free(n);
        if (self.on_rejected_node) |n| self.allocator.free(n);
    }

    pub fn addStepDefinition(
        self: *ApprovalNode,
        id: []const u8,
        name: []const u8,
        assignment: TaskAssignment,
    ) !void {
        try self.step_definitions.append(self.allocator, StepDefinition{
            .id = try self.allocator.dupe(u8, id),
            .name = try self.allocator.dupe(u8, name),
            .assignment = assignment,
        });
    }

    pub fn createApprovalChain(self: *const ApprovalNode, allocator: Allocator, chain_id: []const u8) !*ApprovalChain {
        const chain = try allocator.create(ApprovalChain);
        errdefer allocator.destroy(chain);

        chain.* = try ApprovalChain.init(allocator, chain_id, self.name, self.approval_type);

        // Create steps from definitions
        for (self.step_definitions.items) |def| {
            _ = try chain.createAndAddStep(def.id, def.name, def.assignment);
        }

        return chain;
    }
};

// ============================================================================
// Approval Manager
// ============================================================================

/// Approval Manager - manages multi-level approval workflows
pub const ApprovalManager = struct {
    chains: std.StringHashMap(*ApprovalChain),
    chains_by_workflow: std.StringHashMap(std.ArrayList([]const u8)),
    task_manager: *TaskManager,
    chain_counter: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, task_manager: *TaskManager) ApprovalManager {
        return ApprovalManager{
            .chains = std.StringHashMap(*ApprovalChain).init(allocator),
            .chains_by_workflow = std.StringHashMap(std.ArrayList([]const u8)).init(allocator),
            .task_manager = task_manager,
            .chain_counter = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ApprovalManager) void {
        // Free all chains and their keys (keys are shared with workflow lists)
        var chain_iter = self.chains.iterator();
        while (chain_iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.chains.deinit();

        // Free workflow index lists (strings already freed via chains hashmap keys)
        var wf_iter = self.chains_by_workflow.valueIterator();
        while (wf_iter.next()) |list_ptr| {
            // Don't free items - they're the same pointers as chain keys
            list_ptr.deinit(self.allocator);
        }
        self.chains_by_workflow.deinit();
    }

    fn generateChainId(self: *ApprovalManager) ![]const u8 {
        self.chain_counter += 1;
        const timestamp = std.time.timestamp();
        var buf: [64]u8 = undefined;
        const len = std.fmt.bufPrint(&buf, "chain-{d}-{d}", .{ timestamp, self.chain_counter }) catch return error.BufferTooSmall;
        return try self.allocator.dupe(u8, len);
    }

    pub fn createApprovalChain(
        self: *ApprovalManager,
        node: *const ApprovalNode,
        workflow_id: []const u8,
    ) !*ApprovalChain {
        const chain_id = try self.generateChainId();
        defer self.allocator.free(chain_id);

        const chain = try node.createApprovalChain(self.allocator, chain_id);

        // Store chain
        const id_copy = try self.allocator.dupe(u8, chain.id);
        try self.chains.put(id_copy, chain);

        // Index by workflow
        const gop = try self.chains_by_workflow.getOrPut(workflow_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList([]const u8){};
        }
        try gop.value_ptr.append(self.allocator, id_copy);

        return chain;
    }

    pub fn getChain(self: *ApprovalManager, chain_id: []const u8) ?*ApprovalChain {
        return self.chains.get(chain_id);
    }

    pub fn getChainsForWorkflow(self: *ApprovalManager, workflow_id: []const u8) ?[]const []const u8 {
        if (self.chains_by_workflow.get(workflow_id)) |list| {
            return list.items;
        }
        return null;
    }

    pub fn processApproval(
        self: *ApprovalManager,
        chain_id: []const u8,
        step_id: []const u8,
        decision: ApprovalDecision,
        user_id: []const u8,
        comments: ?[]const u8,
    ) !void {
        const chain = self.chains.get(chain_id) orelse return error.ChainNotFound;
        try chain.processDecision(step_id, decision, user_id, comments);
    }

    pub fn chainCount(self: *ApprovalManager) usize {
        return self.chains.count();
    }

    pub fn getPendingChainCount(self: *ApprovalManager) usize {
        var count: usize = 0;
        var iter = self.chains.valueIterator();
        while (iter.next()) |chain_ptr| {
            if (!chain_ptr.*.isComplete()) {
                count += 1;
            }
        }
        return count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ApprovalType properties" {
    try std.testing.expectEqualStrings("SEQUENTIAL", ApprovalType.SEQUENTIAL.toString());
    try std.testing.expectEqual(ApprovalType.SEQUENTIAL, ApprovalType.fromString("SEQUENTIAL").?);

    try std.testing.expect(ApprovalType.UNANIMOUS.requiresAllApprovers());
    try std.testing.expect(!ApprovalType.SINGLE.requiresAllApprovers());

    try std.testing.expect(ApprovalType.PARALLEL.isParallel());
    try std.testing.expect(!ApprovalType.SEQUENTIAL.isParallel());
}

test "ApprovalStep approve and reject" {
    const allocator = std.testing.allocator;

    const assignment = TaskAssignment{ .user_id = "manager-1" };
    var step = try ApprovalStep.init(allocator, "step-1", "Manager Approval", 0, assignment);
    defer step.deinit();

    try std.testing.expect(step.isPending());
    try std.testing.expect(!step.isApproved());

    try step.approve("manager-1", "Looks good");
    try std.testing.expect(step.isApproved());
    try std.testing.expect(!step.isPending());
    try std.testing.expectEqualStrings("manager-1", step.decided_by.?);
}

test "ApprovalChain sequential approval" {
    const allocator = std.testing.allocator;

    var chain = try ApprovalChain.init(allocator, "chain-1", "Expense Approval", .SEQUENTIAL);
    defer chain.deinit();

    const assignment1 = TaskAssignment{ .user_id = "manager-1" };
    const assignment2 = TaskAssignment{ .user_id = "director-1" };

    _ = try chain.createAndAddStep("step-1", "Manager Approval", assignment1);
    _ = try chain.createAndAddStep("step-2", "Director Approval", assignment2);

    try std.testing.expectEqual(@as(usize, 2), chain.stepCount());
    try std.testing.expect(!chain.isComplete());

    // First step approval
    try chain.processDecision("step-1", .APPROVED, "manager-1", null);
    try std.testing.expect(!chain.isComplete());
    try std.testing.expectEqual(@as(usize, 1), chain.current_step_index);

    // Second step approval
    try chain.processDecision("step-2", .APPROVED, "director-1", null);
    try std.testing.expect(chain.isComplete());
    try std.testing.expect(chain.isApproved());
}

test "ApprovalChain sequential rejection" {
    const allocator = std.testing.allocator;

    var chain = try ApprovalChain.init(allocator, "chain-1", "Expense Approval", .SEQUENTIAL);
    defer chain.deinit();

    const assignment1 = TaskAssignment{ .user_id = "manager-1" };
    const assignment2 = TaskAssignment{ .user_id = "director-1" };

    _ = try chain.createAndAddStep("step-1", "Manager Approval", assignment1);
    _ = try chain.createAndAddStep("step-2", "Director Approval", assignment2);

    // First step rejection
    try chain.processDecision("step-1", .REJECTED, "manager-1", "Budget exceeded");
    try std.testing.expect(chain.isComplete());
    try std.testing.expect(chain.isRejected());
}

test "ApprovalChain unanimous approval" {
    const allocator = std.testing.allocator;

    var chain = try ApprovalChain.init(allocator, "chain-1", "Board Vote", .UNANIMOUS);
    defer chain.deinit();

    _ = try chain.createAndAddStep("step-1", "Board Member 1", TaskAssignment{ .user_id = "member-1" });
    _ = try chain.createAndAddStep("step-2", "Board Member 2", TaskAssignment{ .user_id = "member-2" });
    _ = try chain.createAndAddStep("step-3", "Board Member 3", TaskAssignment{ .user_id = "member-3" });

    // All approve
    try chain.processDecision("step-1", .APPROVED, "member-1", null);
    try std.testing.expect(!chain.isComplete());

    try chain.processDecision("step-2", .APPROVED, "member-2", null);
    try std.testing.expect(!chain.isComplete());

    try chain.processDecision("step-3", .APPROVED, "member-3", null);
    try std.testing.expect(chain.isComplete());
    try std.testing.expect(chain.isApproved());
}

test "ApprovalChain unanimous rejection" {
    const allocator = std.testing.allocator;

    var chain = try ApprovalChain.init(allocator, "chain-1", "Board Vote", .UNANIMOUS);
    defer chain.deinit();

    _ = try chain.createAndAddStep("step-1", "Board Member 1", TaskAssignment{ .user_id = "member-1" });
    _ = try chain.createAndAddStep("step-2", "Board Member 2", TaskAssignment{ .user_id = "member-2" });

    // One rejection fails unanimous
    try chain.processDecision("step-1", .APPROVED, "member-1", null);
    try chain.processDecision("step-2", .REJECTED, "member-2", "Disagree");

    try std.testing.expect(chain.isComplete());
    try std.testing.expect(chain.isRejected());
}

test "ApprovalChain majority approval" {
    const allocator = std.testing.allocator;

    var chain = try ApprovalChain.init(allocator, "chain-1", "Committee Vote", .MAJORITY);
    defer chain.deinit();

    _ = try chain.createAndAddStep("step-1", "Member 1", TaskAssignment{ .user_id = "member-1" });
    _ = try chain.createAndAddStep("step-2", "Member 2", TaskAssignment{ .user_id = "member-2" });
    _ = try chain.createAndAddStep("step-3", "Member 3", TaskAssignment{ .user_id = "member-3" });

    // 2 out of 3 approve (majority)
    try chain.processDecision("step-1", .APPROVED, "member-1", null);
    try std.testing.expect(!chain.isComplete());

    try chain.processDecision("step-2", .APPROVED, "member-2", null);
    try std.testing.expect(chain.isComplete());
    try std.testing.expect(chain.isApproved());
}

test "ApprovalNode creates chain" {
    const allocator = std.testing.allocator;

    var node = try ApprovalNode.init(allocator, "node-1", "Expense Approval", .SEQUENTIAL);
    defer node.deinit();

    try node.addStepDefinition("step-1", "Manager", TaskAssignment{ .role_id = "manager" });
    try node.addStepDefinition("step-2", "Director", TaskAssignment{ .role_id = "director" });

    const chain = try node.createApprovalChain(allocator, "chain-1");
    defer {
        chain.deinit();
        allocator.destroy(chain);
    }

    try std.testing.expectEqual(@as(usize, 2), chain.stepCount());
    try std.testing.expectEqual(ApprovalType.SEQUENTIAL, chain.approval_type);
}

test "ApprovalManager operations" {
    const allocator = std.testing.allocator;

    var task_manager = TaskManager.init(allocator);
    defer task_manager.deinit();

    var manager = ApprovalManager.init(allocator, &task_manager);
    defer manager.deinit();

    var node = try ApprovalNode.init(allocator, "node-1", "Approval", .SINGLE);
    defer node.deinit();
    try node.addStepDefinition("step-1", "Approver", TaskAssignment{ .user_id = "approver-1" });

    // Create chain
    const chain = try manager.createApprovalChain(&node, "wf-1");
    try std.testing.expectEqual(@as(usize, 1), manager.chainCount());
    try std.testing.expectEqual(@as(usize, 1), manager.getPendingChainCount());

    // Process approval
    try manager.processApproval(chain.id, "step-1", .APPROVED, "approver-1", "Approved");
    try std.testing.expect(chain.isComplete());
    try std.testing.expectEqual(@as(usize, 0), manager.getPendingChainCount());
}

test "ApprovalManager chain not found" {
    const allocator = std.testing.allocator;

    var task_manager = TaskManager.init(allocator);
    defer task_manager.deinit();

    var manager = ApprovalManager.init(allocator, &task_manager);
    defer manager.deinit();

    try std.testing.expectError(error.ChainNotFound, manager.processApproval("nonexistent", "step-1", .APPROVED, "user", null));
}

