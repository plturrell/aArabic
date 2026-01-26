const std = @import("std");

/// AI-Driven Agent Selection using nLocalModels
/// Enhances Hungarian algorithm with ML-based predictions for:
/// - Agent performance forecasting
/// - Task complexity estimation
/// - Workload prediction
/// - Optimal assignment recommendations

pub const IntelligentAgentSelector = struct {
    allocator: std.mem.Allocator,
    local_models_url: []const u8,
    model_name: []const u8,
    
    pub fn init(
        allocator: std.mem.Allocator,
        local_models_url: []const u8,
        model_name: []const u8,
    ) !IntelligentAgentSelector {
        return IntelligentAgentSelector{
            .allocator = allocator,
            .local_models_url = try allocator.dupe(u8, local_models_url),
            .model_name = try allocator.dupe(u8, model_name),
        };
    }
    
    pub fn deinit(self: *IntelligentAgentSelector) void {
        self.allocator.free(self.local_models_url);
        self.allocator.free(self.model_name);
    }
    
    // =========================================================================
    // AI-Enhanced Selection
    // =========================================================================
    
    /// Get AI recommendation for agent assignment
    pub fn getRecommendation(
        self: *IntelligentAgentSelector,
        task: TaskContext,
        agents: []const AgentContext,
    ) !AgentRecommendation {
        // Build prompt for LLM
        const prompt = try self.buildPrompt(task, agents);
        defer self.allocator.free(prompt);
        
        // Call nLocalModels inference API
        const ai_response = try self.callLocalModels(prompt);
        defer self.allocator.free(ai_response);
        
        // Parse AI response
        return try self.parseRecommendation(ai_response, agents);
    }
    
    fn buildPrompt(
        self: *IntelligentAgentSelector,
        task: TaskContext,
        agents: []const AgentContext,
    ) ![]u8 {
        var prompt = std.ArrayList(u8).init(self.allocator);
        const writer = prompt.writer();
        
        try writer.writeAll("# Agent Assignment Task\n\n");
        try writer.print("## Task Details\n", .{});
        try writer.print("- Name: {s}\n", .{task.name});
        try writer.print("- Type: {s}\n", .{task.task_type});
        try writer.print("- Priority: {d}/10\n", .{task.priority});
        try writer.print("- Complexity: {s}\n", .{task.complexity});
        try writer.print("- Estimated Effort: {d} minutes\n", .{task.estimated_effort});
        
        if (task.required_skills.len > 0) {
            try writer.writeAll("- Required Skills: ");
            for (task.required_skills, 0..) |skill, i| {
                try writer.print("{s}", .{skill});
                if (i < task.required_skills.len - 1) {
                    try writer.writeAll(", ");
                }
            }
            try writer.writeAll("\n");
        }
        
        try writer.writeAll("\n## Available Agents\n\n");
        for (agents, 0..) |agent, i| {
            try writer.print("{d}. {s} ({s})\n", .{ i + 1, agent.name, agent.role });
            try writer.print("   - Skills: ", .{});
            for (agent.capabilities, 0..) |cap, j| {
                try writer.print("{s}", .{cap});
                if (j < agent.capabilities.len - 1) {
                    try writer.writeAll(", ");
                }
            }
            try writer.writeAll("\n");
            try writer.print("   - Current Load: {d}/{d} tasks\n", .{ agent.current_load, agent.capacity });
            try writer.print("   - Performance: {d:.1}%\n", .{agent.performance_score * 100});
            try writer.print("   - Avg Completion Time: {d} mins\n", .{agent.avg_completion_time});
            try writer.writeAll("\n");
        }
        
        try writer.writeAll("\n## Instructions\n");
        try writer.writeAll("Analyze the task requirements and agent capabilities.\n");
        try writer.writeAll("Consider workload, performance history, and skill match.\n");
        try writer.writeAll("Recommend the best agent (1-based index) with reasoning.\n");
        try writer.writeAll("Format: {\"agent_index\": N, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}\n");
        
        return prompt.toOwnedSlice();
    }
    
    fn callLocalModels(self: *IntelligentAgentSelector, prompt: []const u8) ![]u8 {
        // Build request payload
        var payload = std.ArrayList(u8).init(self.allocator);
        defer payload.deinit();
        const writer = payload.writer();
        
        try writer.writeAll("{\n");
        try writer.print("  \"model\": \"{s}\",\n", .{self.model_name});
        try writer.print("  \"prompt\": \"{s}\",\n", .{prompt}); // TODO: Escape JSON
        try writer.writeAll("  \"max_tokens\": 500,\n");
        try writer.writeAll("  \"temperature\": 0.3\n");
        try writer.writeAll("}\n");
        
        // TODO: HTTP POST to nLocalModels /api/v1/inference
        // For now, return mock response
        
        const mock_response =
            \\{
            \\  "agent_index": 1,
            \\  "confidence": 0.87,
            \\  "reasoning": "Agent has matching skills, low current load, and strong performance history"
            \\}
        ;
        
        return try self.allocator.dupe(u8, mock_response);
    }
    
    fn parseRecommendation(
        self: *IntelligentAgentSelector,
        response: []const u8,
        agents: []const AgentContext,
    ) !AgentRecommendation {
        _ = self;
        
        // TODO: Parse JSON response properly
        // For now, return mock recommendation
        
        return AgentRecommendation{
            .agent_id = agents[0].id,
            .confidence = 0.87,
            .reasoning = "AI-recommended based on skills and performance",
            .alternative_agents = &[_][]const u8{},
        };
    }
    
    // =========================================================================
    // Performance Prediction
    // =========================================================================
    
    /// Predict task completion time for agent
    pub fn predictCompletionTime(
        self: *IntelligentAgentSelector,
        task: TaskContext,
        agent: AgentContext,
    ) !PredictionResult {
        // Build prediction prompt
        const prompt = try self.buildPredictionPrompt(task, agent);
        defer self.allocator.free(prompt);
        
        // Call AI model
        const ai_response = try self.callLocalModels(prompt);
        defer self.allocator.free(ai_response);
        
        // Parse prediction
        // TODO: Implement proper parsing
        
        return PredictionResult{
            .estimated_minutes = task.estimated_effort,
            .confidence = 0.75,
            .factors = &[_][]const u8{ "historical_performance", "current_workload" },
        };
    }
    
    fn buildPredictionPrompt(
        self: *IntelligentAgentSelector,
        task: TaskContext,
        agent: AgentContext,
    ) ![]u8 {
        _ = self;
        
        var prompt = std.ArrayList(u8).init(self.allocator);
        const writer = prompt.writer();
        
        try writer.writeAll("Predict task completion time:\n");
        try writer.print("Task: {s} ({s}), Est: {d} mins\n", .{
            task.name,
            task.task_type,
            task.estimated_effort,
        });
        try writer.print("Agent: {s}, Avg: {d} mins, Load: {d}/{d}\n", .{
            agent.name,
            agent.avg_completion_time,
            agent.current_load,
            agent.capacity,
        });
        
        return prompt.toOwnedSlice();
    }
    
    // =========================================================================
    // Workload Balancing
    // =========================================================================
    
    /// Predict future workload and recommend rebalancing
    pub fn recommendWorkloadBalance(
        self: *IntelligentAgentSelector,
        agents: []const AgentContext,
        pending_tasks: []const TaskContext,
    ) !BalanceRecommendation {
        _ = self;
        _ = agents;
        _ = pending_tasks;
        
        // TODO: Use AI to predict future workload and recommend assignments
        
        return BalanceRecommendation{
            .should_rebalance = false,
            .recommended_assignments = &[_]TaskAssignment{},
            .reasoning = "Workload is currently balanced",
        };
    }
};

// =========================================================================
// Supporting Types
// =========================================================================

pub const TaskContext = struct {
    id: []const u8,
    name: []const u8,
    task_type: []const u8,
    priority: u8,
    complexity: []const u8, // "low", "medium", "high"
    estimated_effort: u32,
    required_skills: []const []const u8,
    historical_data: ?HistoricalTaskData,
};

pub const AgentContext = struct {
    id: []const u8,
    name: []const u8,
    role: []const u8,
    capabilities: []const []const u8,
    capacity: u32,
    current_load: u32,
    performance_score: f32,
    avg_completion_time: u32,
    historical_data: ?HistoricalAgentData,
};

pub const HistoricalTaskData = struct {
    similar_tasks_completed: u32,
    avg_completion_time: u32,
    success_rate: f32,
};

pub const HistoricalAgentData = struct {
    tasks_completed: u32,
    avg_task_time: u32,
    success_rate: f32,
    specialization_areas: []const []const u8,
};

pub const AgentRecommendation = struct {
    agent_id: []const u8,
    confidence: f32, // 0.0-1.0
    reasoning: []const u8,
    alternative_agents: []const []const u8, // Backup recommendations
};

pub const PredictionResult = struct {
    estimated_minutes: u32,
    confidence: f32,
    factors: []const []const u8, // Factors influencing prediction
};

pub const BalanceRecommendation = struct {
    should_rebalance: bool,
    recommended_assignments: []const TaskAssignment,
    reasoning: []const u8,
};

pub const TaskAssignment = struct {
    task_id: []const u8,
    agent_id: []const u8,
    reason: []const u8,
};

// =========================================================================
// Tests
// =========================================================================

test "IntelligentAgentSelector: init" {
    const allocator = std.testing.allocator;
    
    var selector = try IntelligentAgentSelector.init(
        allocator,
        "http://localhost:8000",
        "qwen-2.5",
    );
    defer selector.deinit();
    
    try std.testing.expectEqualStrings("qwen-2.5", selector.model_name);
}