const std = @import("std");

/// Reusable Approval Engine for Process Apps
/// Provides common approval workflow patterns that can be used across different process applications

pub const ApprovalRole = enum {
    maker,
    checker,
    manager,
    approver,
    viewer,
};

pub const ApprovalAction = enum {
    submit,
    approve,
    reject,
    revise,
    cancel,
};

pub const ApprovalRecord = struct {
    id: []const u8,
    instance_id: []const u8,
    step_number: u32,
    role: ApprovalRole,
    user_id: []const u8,
    action: ApprovalAction,
    comments: []const u8,
    timestamp: i64,
    duration_ms: ?u64, // Time taken to approve
};

pub const ApprovalChain = struct {
    steps: []const ApprovalStep,
    parallel: bool, // If true, all steps must approve in parallel
    
    pub const ApprovalStep = struct {
        role: ApprovalRole,
        required: bool,
        timeout_hours: ?u32,
        escalation_role: ?ApprovalRole,
    };
};

pub const ApprovalEngine = struct {
    allocator: std.mem.Allocator,
    chain: ApprovalChain,
    records: std.ArrayList(ApprovalRecord),
    
    pub fn init(allocator: std.mem.Allocator, chain: ApprovalChain) !ApprovalEngine {
        return ApprovalEngine{
            .allocator = allocator,
            .chain = chain,
            .records = std.ArrayList(ApprovalRecord).init(allocator),
        };
    }
    
    pub fn deinit(self: *ApprovalEngine) void {
        self.records.deinit();
    }
    
    /// Process approval action
    pub fn processApproval(
        self: *ApprovalEngine,
        instance_id: []const u8,
        user_id: []const u8,
        role: ApprovalRole,
        action: ApprovalAction,
        comments: []const u8,
    ) !ApprovalResult {
        const step_number = try self.getCurrentStepNumber(instance_id);
        
        // Validate user has correct role for current step
        if (!try self.validateRole(step_number, role)) {
            return ApprovalResult{
                .success = false,
                .next_step = null,
                .completed = false,
                .message = "Invalid role for current approval step",
            };
        }
        
        // Create approval record
        const record = ApprovalRecord{
            .id = try self.generateRecordId(),
            .instance_id = try self.allocator.dupe(u8, instance_id),
            .step_number = step_number,
            .role = role,
            .user_id = try self.allocator.dupe(u8, user_id),
            .action = action,
            .comments = try self.allocator.dupe(u8, comments),
            .timestamp = std.time.timestamp(),
            .duration_ms = null,
        };
        
        try self.records.append(record);
        
        // Determine next step
        if (action == .reject) {
            return ApprovalResult{
                .success = true,
                .next_step = null,
                .completed = true,
                .message = "Entry rejected",
            };
        }
        
        if (action == .approve) {
            const next_step = step_number + 1;
            if (next_step >= self.chain.steps.len) {
                // All approvals complete
                return ApprovalResult{
                    .success = true,
                    .next_step = null,
                    .completed = true,
                    .message = "All approvals completed",
                };
            } else {
                // Move to next approval step
                return ApprovalResult{
                    .success = true,
                    .next_step = self.chain.steps[next_step].role,
                    .completed = false,
                    .message = "Approved, moving to next step",
                };
            }
        }
        
        return ApprovalResult{
            .success = false,
            .next_step = null,
            .completed = false,
            .message = "Unknown action",
        };
    }
    
    /// Get current approval step for instance
    fn getCurrentStepNumber(self: *ApprovalEngine, instance_id: []const u8) !u32 {
        var max_step: u32 = 0;
        
        for (self.records.items) |record| {
            if (std.mem.eql(u8, record.instance_id, instance_id)) {
                if (record.action == .approve and record.step_number > max_step) {
                    max_step = record.step_number;
                }
            }
        }
        
        // Next step is max_step + 1, but if nothing approved yet, it's step 0
        if (max_step == 0) {
            // Check if any record exists
            for (self.records.items) |record| {
                if (std.mem.eql(u8, record.instance_id, instance_id)) {
                    return max_step + 1;
                }
            }
            return 0; // First approval step
        }
        
        return max_step + 1;
    }
    
    fn validateRole(self: *ApprovalEngine, step_number: u32, role: ApprovalRole) !bool {
        if (step_number >= self.chain.steps.len) return false;
        
        const required_role = self.chain.steps[step_number].role;
        return role == required_role;
    }
    
    /// Get approval history for instance
    pub fn getApprovalHistory(
        self: *ApprovalEngine,
        instance_id: []const u8,
    ) ![]const ApprovalRecord {
        var history = std.ArrayList(ApprovalRecord).init(self.allocator);
        defer history.deinit();
        
        for (self.records.items) |record| {
            if (std.mem.eql(u8, record.instance_id, instance_id)) {
                try history.append(record);
            }
        }
        
        return history.toOwnedSlice();
    }
    
    /// Check if instance has pending approvals
    pub fn hasPendingApprovals(
        self: *ApprovalEngine,
        instance_id: []const u8,
    ) !bool {
        const step = try self.getCurrentStepNumber(instance_id);
        return step < self.chain.steps.len;
    }
    
    fn generateRecordId(self: *ApprovalEngine) ![]const u8 {
        const timestamp = std.time.timestamp();
        return try std.fmt.allocPrint(
            self.allocator,
            "apr_{d}",
            .{timestamp},
        );
    }
};

pub const ApprovalResult = struct {
    success: bool,
    next_step: ?ApprovalRole,
    completed: bool,
    message: []const u8,
};

// =========================================================================
// Predefined Approval Chains
// =========================================================================

/// Standard two-tier approval (Checker -> Manager)
pub fn createTwoTierChain() ApprovalChain {
    return ApprovalChain{
        .steps = &[_]ApprovalChain.ApprovalStep{
            .{ .role = .checker, .required = true, .timeout_hours = 24, .escalation_role = .manager },
            .{ .role = .manager, .required = true, .timeout_hours = 48, .escalation_role = null },
        },
        .parallel = false,
    };
}

/// Standard three-tier approval (Maker -> Checker -> Manager)
pub fn createThreeTierChain() ApprovalChain {
    return ApprovalChain{
        .steps = &[_]ApprovalChain.ApprovalStep{
            .{ .role = .maker, .required = true, .timeout_hours = null, .escalation_role = null },
            .{ .role = .checker, .required = true, .timeout_hours = 24, .escalation_role = .manager },
            .{ .role = .manager, .required = true, .timeout_hours = 48, .escalation_role = null },
        },
        .parallel = false,
    };
}

/// Parallel approval (all approvers must approve)
pub fn createParallelChain(roles: []const ApprovalRole) ApprovalChain {
    _ = roles;
    // TODO: Create dynamic approval chain from roles array
    return ApprovalChain{
        .steps = &[_]ApprovalChain.ApprovalStep{},
        .parallel = true,
    };
}

// =========================================================================
// Tests
// =========================================================================

test "ApprovalEngine: two-tier approval flow" {
    const allocator = std.testing.allocator;
    
    const chain = createTwoTierChain();
    var engine = try ApprovalEngine.init(allocator, chain);
    defer engine.deinit();
    
    // Checker approves
    const result1 = try engine.processApproval(
        "inst-001",
        "checker-user",
        .checker,
        .approve,
        "Approved by checker",
    );
    
    try std.testing.expectEqual(true, result1.success);
    try std.testing.expectEqual(false, result1.completed);
    try std.testing.expectEqual(ApprovalRole.manager, result1.next_step.?);
    
    // Manager approves
    const result2 = try engine.processApproval(
        "inst-001",
        "manager-user",
        .manager,
        .approve,
        "Approved by manager",
    );
    
    try std.testing.expectEqual(true, result2.success);
    try std.testing.expectEqual(true, result2.completed);
}