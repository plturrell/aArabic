const std = @import("std");

/// Agent Manager - Core agent orchestration (nLocalModels pattern)
/// Manages agent registry, assignments, and performance tracking
/// Integrates with HANA for persistence and nLocalModels for AI

pub const AgentManager = struct {
    allocator: std.mem.Allocator,
    agents: std.StringHashMap(Agent),
    assignments: std.ArrayList(Assignment),
    hana_cache: ?*HanaAgentCache,
    ai_selector: ?*AISelector,
    config: Config,
    
    pub const Config = struct {
        enable_hana: bool,
        enable_ai: bool,
        assignment_strategy: AssignmentStrategy,
        local_models_url: ?[]const u8,
        
        pub const AssignmentStrategy = enum {
            hungarian,
            greedy,
            ai_enhanced,
            round_robin,
        };
    };
    
    pub fn init(allocator: std.mem.Allocator, config: Config) !AgentManager {
        return AgentManager{
            .allocator = allocator,
            .agents = std.StringHashMap(Agent).init(allocator),
            .assignments = std.ArrayList(Assignment).init(allocator),
            .hana_cache = null,
            .ai_selector = null,
            .config = config,
        };
    }
    
    pub fn deinit(self: *AgentManager) void {
        self.agents.deinit();
        self.assignments.deinit();
    }
    
    // =========================================================================
    // Agent Registration & Management
    // =========================================================================
    
    /// Register agent in system
    pub fn registerAgent(self: *AgentManager, agent: Agent) !void {
        try self.agents.put(agent.id, agent);
        
        // Persist to HANA if enabled
        if (self.hana_cache) |cache| {
            try cache.storeAgent(agent);
        }
        
        std.log.info("Registered agent: {s} ({s})", .{ agent.name, agent.role });
    }
    
    /// Update agent availability
    pub fn updateAvailability(
        self: *AgentManager,
        agent_id: []const u8,
        availability: Agent.Availability,
    ) !void {
        var agent_entry = self.agents.getPtr(agent_id) orelse return error.AgentNotFound;
        agent_entry.availability = availability;
        
        // Update in HANA
        if (self.hana_cache) |cache| {
            try cache.updateAgentAvailability(agent_id, availability);
        }
    }
    
    /// Get agent by ID
    pub fn getAgent(self: *AgentManager, agent_id: []const u8) ?Agent {
        return self.agents.get(agent_id);
    }
    
    /// Get available agents for role/capability
    pub fn getAvailableAgents(
        self: *AgentManager,
        role: ?[]const u8,
        capability: ?[]const u8,
    ) ![]Agent {
        var available = std.ArrayList(Agent).init(self.allocator);
        defer available.deinit();
        
        var it = self.agents.valueIterator();
        while (it.next()) |agent| {
            // Check availability
            if (agent.availability != .available) continue;
            if (agent.current_load >= agent.capacity) continue;
            
            // Check role filter
            if (role) |r| {
                if (!std.mem.eql(u8, agent.role, r)) continue;
            }
            
            // Check capability filter
            if (capability) |cap| {
                var has_cap = false;
                for (agent.capabilities) |agent_cap| {
                    if (std.mem.eql(u8, agent_cap, cap)) {
                        has_cap = true;
                        break;
                    }
                }
                if (!has_cap) continue;
            }
            
            try available.append(agent.*);
        }
        
        return available.toOwnedSlice();
    }
    
    // =========================================================================
    // Task Assignment
    // =========================================================================
    
    /// Assign task to optimal agent
    pub fn assignTask(
        self: *AgentManager,
        task: Task,
    ) !Assignment {
        const available = try self.getAvailableAgents(null, task.required_capability);
        defer self.allocator.free(available);
        
        if (available.len == 0) {
            return error.NoAvailableAgents;
        }
        
        const assignment = switch (self.config.assignment_strategy) {
            .hungarian => try self.assignHungarian(task, available),
            .greedy => try self.assignGreedy(task, available),
            .ai_enhanced => try self.assignWithAI(task, available),
            .round_robin => try self.assignRoundRobin(task, available),
        };
        
        // Update agent load
        var agent_entry = self.agents.getPtr(assignment.agent_id) orelse return error.AgentNotFound;
        agent_entry.current_load += 1;
        
        // Store assignment
        try self.assignments.append(assignment);
        
        // Persist to HANA
        if (self.hana_cache) |cache| {
            try cache.storeAssignment(assignment);
        }
        
        std.log.info("Assigned task {s} to agent {s} (score: {d:.2})", .{
            task.id,
            assignment.agent_id,
            assignment.score,
        });
        
        return assignment;
    }
    
    fn assignHungarian(
        self: *AgentManager,
        task: Task,
        available: []Agent,
    ) !Assignment {
        // Build cost matrix
        var costs = try self.allocator.alloc(f32, available.len);
        defer self.allocator.free(costs);
        
        for (available, 0..) |agent, i| {
            costs[i] = self.calculateAssignmentCost(task, agent);
        }
        
        // Find minimum cost
        var min_cost: f32 = std.math.floatMax(f32);
        var best_idx: usize = 0;
        
        for (costs, 0..) |cost, i| {
            if (cost < min_cost) {
                min_cost = cost;
                best_idx = i;
            }
        }
        
        return Assignment{
            .id = try self.generateAssignmentId(),
            .task_id = task.id,
            .agent_id = available[best_idx].id,
            .score = 1.0 - (min_cost / 10.0),
            .method = "hungarian",
            .timestamp = std.time.timestamp(),
        };
    }
    
    fn assignGreedy(
        self: *AgentManager,
        task: Task,
        available: []Agent,
    ) !Assignment {
        return self.assignHungarian(task, available); // Same logic for single task
    }
    
    fn assignWithAI(
        self: *AgentManager,
        task: Task,
        available: []Agent,
    ) !Assignment {
        if (self.ai_selector) |selector| {
            const recommendation = try selector.recommend(task, available);
            
            return Assignment{
                .id = try self.generateAssignmentId(),
                .task_id = task.id,
                .agent_id = recommendation.agent_id,
                .score = recommendation.confidence,
                .method = "ai_enhanced",
                .timestamp = std.time.timestamp(),
            };
        }
        
        // Fallback to greedy
        return self.assignGreedy(task, available);
    }
    
    fn assignRoundRobin(
        self: *AgentManager,
        task: Task,
        available: []Agent,
    ) !Assignment {
        _ = self;
        
        return Assignment{
            .id = try self.generateAssignmentId(),
            .task_id = task.id,
            .agent_id = available[0].id,
            .score = 0.7,
            .method = "round_robin",
            .timestamp = std.time.timestamp(),
        };
    }
    
    fn calculateAssignmentCost(
        self: *AgentManager,
        task: Task,
        agent: Agent,
    ) f32 {
        _ = self;
        var cost: f32 = 1.0;
        
        // Capability match
        var has_capability = false;
        if (task.required_capability) |req_cap| {
            for (agent.capabilities) |cap| {
                if (std.mem.eql(u8, cap, req_cap)) {
                    has_capability = true;
                    break;
                }
            }
            if (!has_capability) cost += 5.0;
        }
        
        // Workload factor
        const utilization = @as(f32, @floatFromInt(agent.current_load)) / 
                          @as(f32, @floatFromInt(agent.capacity));
        cost += utilization * 0.5;
        
        // Performance factor
        cost += (1.0 - agent.performance_score) * 0.3;
        
        return cost;
    }
    
    fn generateAssignmentId(self: *AgentManager) ![]const u8 {
        const timestamp = std.time.timestamp();
        return try std.fmt.allocPrint(
            self.allocator,
            "assign_{d}",
            .{timestamp},
        );
    }
    
    // =========================================================================
    // Task Completion
    // =========================================================================
    
    /// Mark task as completed
    pub fn completeTask(
        self: *AgentManager,
        task_id: []const u8,
        success: bool,
        duration_ms: u64,
    ) !void {
        // Find assignment
        var assignment: ?*Assignment = null;
        for (self.assignments.items) |*a| {
            if (std.mem.eql(u8, a.task_id, task_id)) {
                assignment = a;
                break;
            }
        }
        
        if (assignment == null) return error.AssignmentNotFound;
        
        const agent_id = assignment.?.agent_id;
        
        // Update agent load
        var agent_entry = self.agents.getPtr(agent_id) orelse return error.AgentNotFound;
        if (agent_entry.current_load > 0) {
            agent_entry.current_load -= 1;
        }
        
        // Update performance score (exponential moving average)
        const performance_impact: f32 = if (success) 0.05 else -0.05;
        agent_entry.performance_score = std.math.clamp(
            agent_entry.performance_score + performance_impact,
            0.0,
            1.0,
        );
        
        // Record in HANA
        if (self.hana_cache) |cache| {
            try cache.recordCompletion(task_id, agent_id, success, duration_ms);
        }
        
        std.log.info("Task {s} completed by {s}: {s} ({d}ms)", .{
            task_id,
            agent_id,
            if (success) "success" else "failure",
            duration_ms,
        });
    }
    
    // =========================================================================
    // Analytics
    // =========================================================================
    
    /// Get agent statistics
    pub fn getStatistics(self: *AgentManager) Statistics {
        var stats = Statistics{
            .total_agents = self.agents.count(),
            .available_agents = 0,
            .busy_agents = 0,
            .total_assignments = self.assignments.items.len,
            .avg_utilization = 0.0,
        };
        
        var total_util: f32 = 0.0;
        var it = self.agents.valueIterator();
        
        while (it.next()) |agent| {
            if (agent.availability == .available and agent.current_load < agent.capacity) {
                stats.available_agents += 1;
            } else {
                stats.busy_agents += 1;
            }
            
            const util = @as(f32, @floatFromInt(agent.current_load)) / 
                        @as(f32, @floatFromInt(agent.capacity));
            total_util += util;
        }
        
        if (stats.total_agents > 0) {
            stats.avg_utilization = total_util / @as(f32, @floatFromInt(stats.total_agents));
        }
        
        return stats;
    }
};

// =========================================================================
// Supporting Types (Following nLocalModels patterns)
// =========================================================================

pub const Agent = struct {
    id: []const u8,
    name: []const u8,
    role: []const u8, // "maker", "checker", "manager"
    capabilities: []const []const u8,
    capacity: u32,
    current_load: u32,
    availability: Availability,
    performance_score: f32,
    
    pub const Availability = enum {
        available,
        busy,
        offline,
        on_leave,
    };
};

pub const Task = struct {
    id: []const u8,
    name: []const u8,
    required_capability: ?[]const u8,
    priority: u8,
    estimated_effort: u32,
    deadline: ?i64,
};

pub const Assignment = struct {
    id: []const u8,
    task_id: []const u8,
    agent_id: []const u8,
    score: f32,
    method: []const u8,
    timestamp: i64,
};

pub const Statistics = struct {
    total_agents: usize,
    available_agents: usize,
    busy_agents: usize,
    total_assignments: usize,
    avg_utilization: f32,
};

// Placeholder types for dependencies
const HanaAgentCache = struct {};
const AISelector = struct {};