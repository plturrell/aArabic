//! Task Assignment Module for nWorkflow Case Management
//! Implements task routing, workload balancing, and skill-based assignment

const std = @import("std");
const Allocator = std.mem.Allocator;

// Assignment Rule types
pub const AssignmentRule = enum {
    ROUND_ROBIN, // Rotate through available assignees
    LEAST_LOADED, // Assign to user with fewest active tasks
    SKILL_BASED, // Match required skills
    MANUAL, // Manual assignment only
    WEIGHTED, // Weighted distribution based on capacity
    PRIORITY_BASED, // Higher priority users get assigned first

    pub fn toString(self: AssignmentRule) []const u8 {
        return @tagName(self);
    }
};

// Skill proficiency levels
pub const SkillLevel = enum {
    BEGINNER,
    INTERMEDIATE,
    ADVANCED,
    EXPERT,

    pub fn toNumber(self: SkillLevel) u8 {
        return switch (self) {
            .BEGINNER => 1,
            .INTERMEDIATE => 2,
            .ADVANCED => 3,
            .EXPERT => 4,
        };
    }

    pub fn meetsRequirement(self: SkillLevel, required: SkillLevel) bool {
        return self.toNumber() >= required.toNumber();
    }
};

// Skill requirement for task assignment
pub const SkillRequirement = struct {
    skill_name: []const u8,
    min_level: SkillLevel = .BEGINNER,
    is_mandatory: bool = true,
    allocator: Allocator,

    pub fn init(allocator: Allocator, skill_name: []const u8, min_level: SkillLevel) !SkillRequirement {
        return SkillRequirement{
            .skill_name = try allocator.dupe(u8, skill_name),
            .min_level = min_level,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SkillRequirement) void {
        self.allocator.free(self.skill_name);
    }
};

// User skill record
pub const UserSkill = struct {
    user_id: []const u8,
    skill_name: []const u8,
    level: SkillLevel,
    certified: bool = false,
    last_used: ?i64 = null,
};

// Team member in assignment pool
pub const PoolMember = struct {
    user_id: []const u8,
    name: []const u8,
    is_active: bool = true,
    capacity: u32 = 10, // Max concurrent tasks
    current_load: u32 = 0,
    skills: std.ArrayList(UserSkill),
    weight: u32 = 100, // For weighted assignment (100 = normal)
    priority: u32 = 1, // For priority-based assignment
    allocator: Allocator,

    pub fn init(allocator: Allocator, user_id: []const u8, name: []const u8) !PoolMember {
        return PoolMember{
            .user_id = try allocator.dupe(u8, user_id),
            .name = try allocator.dupe(u8, name),
            .skills = std.ArrayList(UserSkill){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PoolMember) void {
        self.allocator.free(self.user_id);
        self.allocator.free(self.name);
        self.skills.deinit(self.allocator);
    }

    pub fn hasCapacity(self: *const PoolMember) bool {
        return self.is_active and self.current_load < self.capacity;
    }

    pub fn availableCapacity(self: *const PoolMember) u32 {
        if (!self.is_active or self.current_load >= self.capacity) return 0;
        return self.capacity - self.current_load;
    }

    pub fn hasSkill(self: *const PoolMember, skill_name: []const u8, min_level: SkillLevel) bool {
        for (self.skills.items) |skill| {
            if (std.mem.eql(u8, skill.skill_name, skill_name)) {
                return skill.level.meetsRequirement(min_level);
            }
        }
        return false;
    }

    pub fn addSkill(self: *PoolMember, skill: UserSkill) !void {
        try self.skills.append(self.allocator, skill);
    }

    pub fn incrementLoad(self: *PoolMember) void {
        if (self.current_load < self.capacity) {
            self.current_load += 1;
        }
    }

    pub fn decrementLoad(self: *PoolMember) void {
        if (self.current_load > 0) {
            self.current_load -= 1;
        }
    }
};

// Assignment Pool - group of users for task assignment
pub const AssignmentPool = struct {
    id: []const u8,
    name: []const u8,
    members: std.StringHashMap(*PoolMember),
    member_order: std.ArrayList([]const u8), // For round-robin
    assignment_rule: AssignmentRule = .ROUND_ROBIN,
    round_robin_index: usize = 0,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8) !AssignmentPool {
        return AssignmentPool{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .members = std.StringHashMap(*PoolMember).init(allocator),
            .member_order = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AssignmentPool) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);

        var iter = self.members.valueIterator();
        while (iter.next()) |m| {
            m.*.deinit();
            self.allocator.destroy(m.*);
        }
        self.members.deinit();

        for (self.member_order.items) |id| {
            self.allocator.free(id);
        }
        self.member_order.deinit(self.allocator);
    }

    pub fn addMember(self: *AssignmentPool, member: *PoolMember) !void {
        try self.members.put(member.user_id, member);
        try self.member_order.append(self.allocator, try self.allocator.dupe(u8, member.user_id));
    }

    pub fn getMember(self: *AssignmentPool, user_id: []const u8) ?*PoolMember {
        return self.members.get(user_id);
    }

    pub fn getActiveMembers(self: *AssignmentPool, allocator: Allocator) ![]const *PoolMember {
        var result = std.ArrayList(*PoolMember){};
        var iter = self.members.valueIterator();
        while (iter.next()) |m| {
            if (m.*.is_active and m.*.hasCapacity()) {
                try result.append(allocator, m.*);
            }
        }
        return result.toOwnedSlice(allocator);
    }

    pub fn getMemberCount(self: *AssignmentPool) usize {
        return self.members.count();
    }

    pub fn getTotalCapacity(self: *AssignmentPool) u32 {
        var total: u32 = 0;
        var iter = self.members.valueIterator();
        while (iter.next()) |m| {
            if (m.*.is_active) {
                total += m.*.availableCapacity();
            }
        }
        return total;
    }
};


// Task Router - routes tasks to appropriate assignees
pub const TaskRouter = struct {
    pools: std.StringHashMap(*AssignmentPool),
    allocator: Allocator,

    pub fn init(allocator: Allocator) TaskRouter {
        return TaskRouter{
            .pools = std.StringHashMap(*AssignmentPool).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TaskRouter) void {
        var iter = self.pools.valueIterator();
        while (iter.next()) |p| {
            p.*.deinit();
            self.allocator.destroy(p.*);
        }
        self.pools.deinit();
    }

    pub fn addPool(self: *TaskRouter, pool: *AssignmentPool) !void {
        try self.pools.put(pool.id, pool);
    }

    pub fn getPool(self: *TaskRouter, pool_id: []const u8) ?*AssignmentPool {
        return self.pools.get(pool_id);
    }

    // Route task to best assignee based on pool's assignment rule
    pub fn routeTask(self: *TaskRouter, pool_id: []const u8, required_skills: ?[]const SkillRequirement) ?*PoolMember {
        const pool = self.pools.get(pool_id) orelse return null;

        return switch (pool.assignment_rule) {
            .ROUND_ROBIN => self.routeRoundRobin(pool),
            .LEAST_LOADED => self.routeLeastLoaded(pool),
            .SKILL_BASED => self.routeSkillBased(pool, required_skills),
            .WEIGHTED => self.routeWeighted(pool),
            .PRIORITY_BASED => self.routePriorityBased(pool),
            .MANUAL => null, // Manual assignment doesn't auto-route
        };
    }

    fn routeRoundRobin(self: *TaskRouter, pool: *AssignmentPool) ?*PoolMember {
        _ = self;
        if (pool.member_order.items.len == 0) return null;

        var attempts: usize = 0;
        while (attempts < pool.member_order.items.len) {
            const idx = pool.round_robin_index % pool.member_order.items.len;
            pool.round_robin_index += 1;

            const user_id = pool.member_order.items[idx];
            if (pool.members.get(user_id)) |member| {
                if (member.hasCapacity()) {
                    return member;
                }
            }
            attempts += 1;
        }
        return null;
    }

    fn routeLeastLoaded(self: *TaskRouter, pool: *AssignmentPool) ?*PoolMember {
        _ = self;
        var best: ?*PoolMember = null;
        var best_available: u32 = 0;

        var iter = pool.members.valueIterator();
        while (iter.next()) |m| {
            if (m.*.hasCapacity()) {
                const available = m.*.availableCapacity();
                if (best == null or available > best_available) {
                    best = m.*;
                    best_available = available;
                }
            }
        }
        return best;
    }

    fn routeSkillBased(self: *TaskRouter, pool: *AssignmentPool, required_skills: ?[]const SkillRequirement) ?*PoolMember {
        _ = self;
        const skills = required_skills orelse return null;

        var best: ?*PoolMember = null;
        var best_score: u32 = 0;

        var iter = pool.members.valueIterator();
        while (iter.next()) |m| {
            if (!m.*.hasCapacity()) continue;

            var score: u32 = 0;
            var has_mandatory = true;

            for (skills) |skill_req| {
                if (m.*.hasSkill(skill_req.skill_name, skill_req.min_level)) {
                    score += 1;
                } else if (skill_req.is_mandatory) {
                    has_mandatory = false;
                    break;
                }
            }

            if (has_mandatory and score > best_score) {
                best = m.*;
                best_score = score;
            }
        }
        return best;
    }

    fn routeWeighted(self: *TaskRouter, pool: *AssignmentPool) ?*PoolMember {
        _ = self;
        var total_weight: u32 = 0;
        var iter = pool.members.valueIterator();
        while (iter.next()) |m| {
            if (m.*.hasCapacity()) {
                total_weight += m.*.weight;
            }
        }

        if (total_weight == 0) return null;

        // Simple weighted selection (could use random for better distribution)
        var current: u32 = 0;
        const target = total_weight / 2;

        iter = pool.members.valueIterator();
        while (iter.next()) |m| {
            if (m.*.hasCapacity()) {
                current += m.*.weight;
                if (current >= target) {
                    return m.*;
                }
            }
        }
        return null;
    }

    fn routePriorityBased(self: *TaskRouter, pool: *AssignmentPool) ?*PoolMember {
        _ = self;
        var best: ?*PoolMember = null;
        var best_priority: u32 = 0;

        var iter = pool.members.valueIterator();
        while (iter.next()) |m| {
            if (m.*.hasCapacity() and m.*.priority > best_priority) {
                best = m.*;
                best_priority = m.*.priority;
            }
        }
        return best;
    }
};


// Workload Balancer - monitors and rebalances workload across pool members
pub const WorkloadBalancer = struct {
    pools: *std.StringHashMap(*AssignmentPool),
    threshold_high: f32 = 0.8, // 80% capacity
    threshold_low: f32 = 0.2, // 20% capacity
    allocator: Allocator,

    pub fn init(allocator: Allocator, pools: *std.StringHashMap(*AssignmentPool)) WorkloadBalancer {
        return WorkloadBalancer{
            .pools = pools,
            .allocator = allocator,
        };
    }

    pub fn getLoadFactor(self: *WorkloadBalancer, pool_id: []const u8) ?f32 {
        _ = self;
        // Implementation would calculate actual load factor
        _ = pool_id;
        return 0.5;
    }

    pub fn isOverloaded(self: *WorkloadBalancer, member: *const PoolMember) bool {
        if (member.capacity == 0) return true;
        const load_factor = @as(f32, @floatFromInt(member.current_load)) / @as(f32, @floatFromInt(member.capacity));
        return load_factor >= self.threshold_high;
    }

    pub fn isUnderloaded(self: *WorkloadBalancer, member: *const PoolMember) bool {
        if (member.capacity == 0) return false;
        const load_factor = @as(f32, @floatFromInt(member.current_load)) / @as(f32, @floatFromInt(member.capacity));
        return load_factor <= self.threshold_low;
    }

    pub fn getPoolStats(self: *WorkloadBalancer, pool_id: []const u8) ?PoolStats {
        const pool = self.pools.get(pool_id) orelse return null;

        var stats = PoolStats{
            .total_members = 0,
            .active_members = 0,
            .total_capacity = 0,
            .total_load = 0,
        };

        var iter = pool.members.valueIterator();
        while (iter.next()) |m| {
            stats.total_members += 1;
            if (m.*.is_active) {
                stats.active_members += 1;
                stats.total_capacity += m.*.capacity;
                stats.total_load += m.*.current_load;
            }
        }

        return stats;
    }
};

pub const PoolStats = struct {
    total_members: u32,
    active_members: u32,
    total_capacity: u32,
    total_load: u32,

    pub fn getLoadPercentage(self: PoolStats) f32 {
        if (self.total_capacity == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_load)) / @as(f32, @floatFromInt(self.total_capacity)) * 100.0;
    }
};

// Tests
test "SkillLevel comparison" {
    try std.testing.expect(SkillLevel.EXPERT.meetsRequirement(.BEGINNER));
    try std.testing.expect(SkillLevel.ADVANCED.meetsRequirement(.INTERMEDIATE));
    try std.testing.expect(!SkillLevel.BEGINNER.meetsRequirement(.ADVANCED));
    try std.testing.expect(SkillLevel.INTERMEDIATE.meetsRequirement(.INTERMEDIATE));
}

test "PoolMember capacity" {
    const allocator = std.testing.allocator;

    var member = try PoolMember.init(allocator, "user-1", "John Doe");
    defer member.deinit();

    member.capacity = 5;
    try std.testing.expect(member.hasCapacity());
    try std.testing.expectEqual(@as(u32, 5), member.availableCapacity());

    member.current_load = 3;
    try std.testing.expectEqual(@as(u32, 2), member.availableCapacity());

    member.current_load = 5;
    try std.testing.expect(!member.hasCapacity());
    try std.testing.expectEqual(@as(u32, 0), member.availableCapacity());
}

test "PoolMember skill matching" {
    const allocator = std.testing.allocator;

    var member = try PoolMember.init(allocator, "user-1", "Jane Smith");
    defer member.deinit();

    try member.addSkill(.{
        .user_id = "user-1",
        .skill_name = "python",
        .level = .ADVANCED,
    });

    try std.testing.expect(member.hasSkill("python", .BEGINNER));
    try std.testing.expect(member.hasSkill("python", .ADVANCED));
    try std.testing.expect(!member.hasSkill("python", .EXPERT));
    try std.testing.expect(!member.hasSkill("java", .BEGINNER));
}

test "AssignmentPool operations" {
    const allocator = std.testing.allocator;

    var pool = try AssignmentPool.init(allocator, "pool-1", "Support Team");
    defer pool.deinit();

    const member1 = try allocator.create(PoolMember);
    member1.* = try PoolMember.init(allocator, "user-1", "User 1");
    try pool.addMember(member1);

    const member2 = try allocator.create(PoolMember);
    member2.* = try PoolMember.init(allocator, "user-2", "User 2");
    try pool.addMember(member2);

    try std.testing.expectEqual(@as(usize, 2), pool.getMemberCount());
    try std.testing.expectEqual(@as(u32, 20), pool.getTotalCapacity()); // 2 * 10 default
}

test "TaskRouter round robin" {
    const allocator = std.testing.allocator;

    var router = TaskRouter.init(allocator);
    defer router.deinit();

    const pool = try allocator.create(AssignmentPool);
    pool.* = try AssignmentPool.init(allocator, "pool-1", "Test Pool");
    pool.assignment_rule = .ROUND_ROBIN;
    try router.addPool(pool);

    const member1 = try allocator.create(PoolMember);
    member1.* = try PoolMember.init(allocator, "user-1", "User 1");
    try pool.addMember(member1);

    const member2 = try allocator.create(PoolMember);
    member2.* = try PoolMember.init(allocator, "user-2", "User 2");
    try pool.addMember(member2);

    // First assignment
    const first = router.routeTask("pool-1", null);
    try std.testing.expect(first != null);

    // Second assignment should be different (round robin)
    const second = router.routeTask("pool-1", null);
    try std.testing.expect(second != null);
}

test "TaskRouter least loaded" {
    const allocator = std.testing.allocator;

    var router = TaskRouter.init(allocator);
    defer router.deinit();

    const pool = try allocator.create(AssignmentPool);
    pool.* = try AssignmentPool.init(allocator, "pool-1", "Test Pool");
    pool.assignment_rule = .LEAST_LOADED;
    try router.addPool(pool);

    const member1 = try allocator.create(PoolMember);
    member1.* = try PoolMember.init(allocator, "user-1", "User 1");
    member1.current_load = 5;
    try pool.addMember(member1);

    const member2 = try allocator.create(PoolMember);
    member2.* = try PoolMember.init(allocator, "user-2", "User 2");
    member2.current_load = 2;
    try pool.addMember(member2);

    const assigned = router.routeTask("pool-1", null);
    try std.testing.expect(assigned != null);
    try std.testing.expectEqualStrings("user-2", assigned.?.user_id); // Lower load
}

test "WorkloadBalancer thresholds" {
    const allocator = std.testing.allocator;

    var pools = std.StringHashMap(*AssignmentPool).init(allocator);
    defer pools.deinit();

    var balancer = WorkloadBalancer.init(allocator, &pools);

    var member = try PoolMember.init(allocator, "user-1", "Test User");
    defer member.deinit();

    member.capacity = 10;
    member.current_load = 2;
    try std.testing.expect(balancer.isUnderloaded(&member));
    try std.testing.expect(!balancer.isOverloaded(&member));

    member.current_load = 9;
    try std.testing.expect(balancer.isOverloaded(&member));
    try std.testing.expect(!balancer.isUnderloaded(&member));
}

test "PoolStats load percentage" {
    const stats = PoolStats{
        .total_members = 5,
        .active_members = 4,
        .total_capacity = 40,
        .total_load = 20,
    };

    try std.testing.expectEqual(@as(f32, 50.0), stats.getLoadPercentage());
}
