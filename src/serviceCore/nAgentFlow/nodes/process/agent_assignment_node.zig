const std = @import("std");

/// Agent Assignment Node for nAgentFlow
/// Integrates with process engine agent logic and Hungarian algorithm
/// Provides intelligent task-to-agent assignment in workflows

pub const AgentAssignmentNode = struct {
    id: []const u8,
    name: []const u8,
    config: NodeConfig,
    allocator: std.mem.Allocator,
    
    pub const NodeConfig = struct {
        assignment_strategy: Strategy,
        capability_required: ?[]const u8,
        consider_workload: bool,
        consider_performance: bool,
        fallback_to_any: bool, // If no matching agent, assign to any available
        
        pub const Strategy = enum {
            hungarian, // Optimal assignment using Hungarian algorithm
            greedy, // Fast greedy assignment
            round_robin, // Round-robin distribution
            least_loaded, // Assign to agent with lowest load
            highest_performance, // Assign to best performing agent
        };
    };
    
    pub fn init(
        allocator: std.mem.Allocator,
        id: []const u8,
        name: []const u8,
        config: NodeConfig,
    ) !AgentAssignmentNode {
        return AgentAssignmentNode{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .config = config,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *AgentAssignmentNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
    }
    
    /// Execute node - assign task to optimal agent
    pub fn execute(self: *AgentAssignmentNode, input: NodeInput) !NodeOutput {
        std.log.info("Agent Assignment Node '{s}': Processing task assignment", .{self.name});
        
        // Parse input
        const task_data = input.task_data;
        const available_agents = input.available_agents;
        
        if (available_agents.len == 0) {
            return NodeOutput{
                .success = false,
                .assigned_agent_id = null,
                .assignment_score = 0.0,
                .message = "No available agents",
            };
        }
        
        // Build cost matrix
        const cost_matrix = try self.buildCostMatrix(task_data, available_agents);
        defer self.allocator.free(cost_matrix);
        
        // Run assignment algorithm
        const assignment = switch (self.config.assignment_strategy) {
            .hungarian => try self.assignHungarian(cost_matrix, available_agents),
            .greedy => try self.assignGreedy(cost_matrix, available_agents),
            .round_robin => try self.assignRoundRobin(available_agents),
            .least_loaded => try self.assignLeastLoaded(available_agents),
            .highest_performance => try self.assignHighestPerformance(available_agents),
        };
        
        return NodeOutput{
            .success = true,
            .assigned_agent_id = assignment.agent_id,
            .assignment_score = assignment.score,
            .message = assignment.reason,
        };
    }
    
    fn buildCostMatrix(
        self: *AgentAssignmentNode,
        task: TaskData,
        agents: []const AgentData,
    ) ![]f32 {
        var matrix = try self.allocator.alloc(f32, agents.len);
        
        for (agents, 0..) |agent, i| {
            // Calculate cost (lower is better)
            var cost: f32 = 1.0;
            
            // Factor 1: Capability match
            if (self.config.capability_required) |req_cap| {
                var has_capability = false;
                for (agent.capabilities) |cap| {
                    if (std.mem.eql(u8, cap, req_cap)) {
                        has_capability = true;
                        break;
                    }
                }
                if (!has_capability) {
                    cost += 10.0; // High penalty for missing capability
                }
            }
            
            // Factor 2: Current workload
            if (self.config.consider_workload) {
                const utilization = @as(f32, @floatFromInt(agent.current_load)) / 
                                  @as(f32, @floatFromInt(agent.capacity));
                cost += utilization * 0.5;
            }
            
            // Factor 3: Performance history
            if (self.config.consider_performance) {
                cost += (1.0 - agent.performance_score) * 0.3;
            }
            
            // Factor 4: Task urgency vs agent availability
            const urgency = @as(f32, @floatFromInt(task.priority)) / 10.0;
            if (agent.availability != .available) {
                cost += urgency * 2.0; // High penalty if urgent task and agent unavailable
            }
            
            matrix[i] = cost;
        }
        
        return matrix;
    }
    
    fn assignHungarian(
        self: *AgentAssignmentNode,
        cost_matrix: []const f32,
        agents: []const AgentData,
    ) !AssignmentResult {
        // For single task, just find minimum cost agent
        var min_cost: f32 = std.math.floatMax(f32);
        var best_idx: usize = 0;
        
        for (cost_matrix, 0..) |cost, i| {
            if (cost < min_cost) {
                min_cost = cost;
                best_idx = i;
            }
        }
        
        return AssignmentResult{
            .agent_id = agents[best_idx].id,
            .score = 1.0 - (min_cost / 10.0), // Convert cost to score
            .reason = "Optimal assignment (Hungarian algorithm)",
        };
    }
    
    fn assignGreedy(
        self: *AgentAssignmentNode,
        cost_matrix: []const f32,
        agents: []const AgentData,
    ) !AssignmentResult {
        _ = self;
        
        // Find agent with lowest cost
        var min_cost: f32 = std.math.floatMax(f32);
        var best_idx: usize = 0;
        
        for (cost_matrix, 0..) |cost, i| {
            if (cost < min_cost) {
                min_cost = cost;
                best_idx = i;
            }
        }
        
        return AssignmentResult{
            .agent_id = agents[best_idx].id,
            .score = 1.0 - (min_cost / 10.0),
            .reason = "Greedy assignment",
        };
    }
    
    fn assignRoundRobin(
        self: *AgentAssignmentNode,
        agents: []const AgentData,
    ) !AssignmentResult {
        _ = self;
        
        // Simple round-robin (use first available)
        return AssignmentResult{
            .agent_id = agents[0].id,
            .score = 0.7,
            .reason = "Round-robin assignment",
        };
    }
    
    fn assignLeastLoaded(
        self: *AgentAssignmentNode,
        agents: []const AgentData,
    ) !AssignmentResult {
        _ = self;
        
        var min_load: u32 = std.math.maxInt(u32);
        var best_idx: usize = 0;
        
        for (agents, 0..) |agent, i| {
            if (agent.current_load < min_load) {
                min_load = agent.current_load;
                best_idx = i;
            }
        }
        
        return AssignmentResult{
            .agent_id = agents[best_idx].id,
            .score = 0.8,
            .reason = "Least loaded agent",
        };
    }
    
    fn assignHighestPerformance(
        self: *AgentAssignmentNode,
        agents: []const AgentData,
    ) !AssignmentResult {
        _ = self;
        
        var best_score: f32 = 0.0;
        var best_idx: usize = 0;
        
        for (agents, 0..) |agent, i| {
            if (agent.performance_score > best_score) {
                best_score = agent.performance_score;
                best_idx = i;
            }
        }
        
        return AssignmentResult{
            .agent_id = agents[best_idx].id,
            .score = best_score,
            .reason = "Highest performing agent",
        };
    }
};

// =========================================================================
// Supporting Types
// =========================================================================

pub const NodeInput = struct {
    task_data: TaskData,
    available_agents: []const AgentData,
};

pub const TaskData = struct {
    id: []const u8,
    name: []const u8,
    priority: u8,
    required_capability: ?[]const u8,
    estimated_effort: u32,
    deadline: ?i64,
};

pub const AgentData = struct {
    id: []const u8,
    name: []const u8,
    role: []const u8,
    capabilities: []const []const u8,
    capacity: u32,
    current_load: u32,
    availability: enum { available, busy, offline, on_leave },
    performance_score: f32,
};

pub const NodeOutput = struct {
    success: bool,
    assigned_agent_id: ?[]const u8,
    assignment_score: f32,
    message: []const u8,
};

pub const AssignmentResult = struct {
    agent_id: []const u8,
    score: f32,
    reason: []const u8,
};

// =========================================================================
// Tests
// =========================================================================

test "AgentAssignmentNode: greedy strategy" {
    const allocator = std.testing.allocator;
    
    const config = AgentAssignmentNode.NodeConfig{
        .assignment_strategy = .greedy,
        .capability_required = "accounting",
        .consider_workload = true,
        .consider_performance = true,
        .fallback_to_any = false,
    };
    
    var node = try AgentAssignmentNode.init(
        allocator,
        "node-001",
        "Task Assignment",
        config,
    );
    defer node.deinit();
    
    const task = TaskData{
        .id = "task-001",
        .name = "Review Entry",
        .priority = 8,
        .required_capability = "accounting",
        .estimated_effort = 30,
        .deadline = null,
    };
    
    const agents = [_]AgentData{
        .{
            .id = "agent-001",
            .name = "John",
            .role = "checker",
            .capabilities = &[_][]const u8{"accounting"},
            .capacity = 5,
            .current_load = 2,
            .availability = .available,
            .performance_score = 0.85,
        },
    };
    
    const input = NodeInput{
        .task_data = task,
        .available_agents = &agents,
    };
    
    const output = try node.execute(input);
    try std.testing.expect(output.success);
}