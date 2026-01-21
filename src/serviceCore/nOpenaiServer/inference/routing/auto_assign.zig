// ============================================================================
// Auto-Assignment Logic - Day 23 Implementation
// ============================================================================
// Purpose: Automatically assign models to agents based on capability scoring
// Week: Week 5 (Days 21-25) - Model Router Foundation
// Phase: Month 2 - Model Router & Orchestration
// ============================================================================

const std = @import("std");
const capability_scorer = @import("capability_scorer.zig");

// Type aliases for convenience
const ModelCapability = capability_scorer.ModelCapability;
const TaskType = capability_scorer.TaskType;
const ModelCapabilityProfile = capability_scorer.ModelCapabilityProfile;
const AgentCapabilityRequirements = capability_scorer.AgentCapabilityRequirements;
const CapabilityMatchResult = capability_scorer.CapabilityMatchResult;
const CapabilityScorer = capability_scorer.CapabilityScorer;

// ============================================================================
// AGENT REGISTRY
// ============================================================================

/// Agent topology information
pub const AgentInfo = struct {
    agent_id: []const u8,
    agent_name: []const u8,
    agent_type: AgentType,
    status: AgentStatus,
    endpoint: []const u8,
    capabilities: []const ModelCapability,
    preferred_capabilities: []const ModelCapability,
    min_context_length: u32,
    
    pub const AgentType = enum {
        inference,
        tool,
        orchestrator,
        
        pub fn toString(self: AgentType) []const u8 {
            return @tagName(self);
        }
    };
    
    pub const AgentStatus = enum {
        online,
        offline,
        maintenance,
        
        pub fn toString(self: AgentStatus) []const u8 {
            return @tagName(self);
        }
    };
    
    /// Convert AgentInfo to AgentCapabilityRequirements for scoring
    pub fn toCapabilityRequirements(self: *const AgentInfo, allocator: std.mem.Allocator) !AgentCapabilityRequirements {
        var requirements = AgentCapabilityRequirements.init(allocator, self.agent_id, self.agent_name);
        
        // Add required capabilities
        for (self.capabilities) |cap| {
            try requirements.required_capabilities.append(cap);
        }
        
        // Add preferred capabilities
        for (self.preferred_capabilities) |cap| {
            try requirements.preferred_capabilities.append(cap);
        }
        
        requirements.min_context_length = self.min_context_length;
        
        return requirements;
    }
};

/// Agent registry for managing available agents
pub const AgentRegistry = struct {
    allocator: std.mem.Allocator,
    agents: std.ArrayList(AgentInfo),
    
    pub fn init(allocator: std.mem.Allocator) AgentRegistry {
        return .{
            .allocator = allocator,
            .agents = std.ArrayList(AgentInfo).init(allocator),
        };
    }
    
    pub fn deinit(self: *AgentRegistry) void {
        self.agents.deinit();
    }
    
    /// Register a new agent
    pub fn registerAgent(self: *AgentRegistry, agent: AgentInfo) !void {
        try self.agents.append(agent);
    }
    
    /// Get all online agents
    pub fn getOnlineAgents(self: *const AgentRegistry, allocator: std.mem.Allocator) !std.ArrayList(AgentInfo) {
        var online = std.ArrayList(AgentInfo).init(allocator);
        
        for (self.agents.items) |agent| {
            if (agent.status == .online) {
                try online.append(agent);
            }
        }
        
        return online;
    }
    
    /// Get agent by ID
    pub fn getAgentById(self: *const AgentRegistry, agent_id: []const u8) ?AgentInfo {
        for (self.agents.items) |agent| {
            if (std.mem.eql(u8, agent.agent_id, agent_id)) {
                return agent;
            }
        }
        return null;
    }
};

// ============================================================================
// MODEL REGISTRY
// ============================================================================

/// Model registry for managing available models
pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    models: std.ArrayList(ModelCapabilityProfile),
    
    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{
            .allocator = allocator,
            .models = std.ArrayList(ModelCapabilityProfile).init(allocator),
        };
    }
    
    pub fn deinit(self: *ModelRegistry) void {
        for (self.models.items) |*model| {
            model.deinit();
        }
        self.models.deinit();
    }
    
    /// Register a new model
    pub fn registerModel(self: *ModelRegistry, model: ModelCapabilityProfile) !void {
        try self.models.append(model);
    }
    
    /// Get all models
    pub fn getAllModels(self: *const ModelRegistry) []const ModelCapabilityProfile {
        return self.models.items;
    }
    
    /// Get model by ID
    pub fn getModelById(self: *const ModelRegistry, model_id: []const u8) ?*const ModelCapabilityProfile {
        for (self.models.items) |*model| {
            if (std.mem.eql(u8, model.model_id, model_id)) {
                return model;
            }
        }
        return null;
    }
};

// ============================================================================
// ASSIGNMENT RESULT
// ============================================================================

/// Assignment decision for an agent
pub const AssignmentDecision = struct {
    agent_id: []const u8,
    agent_name: []const u8,
    model_id: []const u8,
    model_name: []const u8,
    match_score: f32,
    assignment_method: AssignmentMethod,
    capability_overlap: []const ModelCapability,
    missing_required: []const ModelCapability,
    
    pub const AssignmentMethod = enum {
        auto,
        manual,
        fallback,
        
        pub fn toString(self: AssignmentMethod) []const u8 {
            return @tagName(self);
        }
    };
};

// ============================================================================
// AUTO-ASSIGNMENT STRATEGIES
// ============================================================================

/// Strategy for auto-assigning models to agents
pub const AssignmentStrategy = enum {
    greedy,      // Assign best model to each agent independently
    optimal,     // Try to maximize overall assignment quality
    balanced,    // Balance between quality and distribution
    
    pub fn toString(self: AssignmentStrategy) []const u8 {
        return @tagName(self);
    }
};

// ============================================================================
// AUTO-ASSIGNER
// ============================================================================

pub const AutoAssigner = struct {
    allocator: std.mem.Allocator,
    scorer: CapabilityScorer,
    agent_registry: *AgentRegistry,
    model_registry: *ModelRegistry,
    
    pub fn init(
        allocator: std.mem.Allocator,
        agent_registry: *AgentRegistry,
        model_registry: *ModelRegistry,
    ) AutoAssigner {
        return .{
            .allocator = allocator,
            .scorer = CapabilityScorer.init(allocator),
            .agent_registry = agent_registry,
            .model_registry = model_registry,
        };
    }
    
    /// Perform auto-assignment using specified strategy
    pub fn assignAll(
        self: *AutoAssigner,
        strategy: AssignmentStrategy,
    ) !std.ArrayList(AssignmentDecision) {
        return switch (strategy) {
            .greedy => try self.assignGreedy(),
            .optimal => try self.assignOptimal(),
            .balanced => try self.assignBalanced(),
        };
    }
    
    /// Greedy assignment: Assign best model to each agent independently
    fn assignGreedy(self: *AutoAssigner) !std.ArrayList(AssignmentDecision) {
        var decisions = std.ArrayList(AssignmentDecision).init(self.allocator);
        
        // Get online agents
        var online_agents = try self.agent_registry.getOnlineAgents(self.allocator);
        defer online_agents.deinit();
        
        const models = self.model_registry.getAllModels();
        
        // For each agent, find the best matching model
        for (online_agents.items) |agent| {
            var best_score: f32 = 0.0;
            var best_model: ?*const ModelCapabilityProfile = null;
            var best_result: ?CapabilityMatchResult = null;
            
            // Convert agent to capability requirements
            var requirements = try agent.toCapabilityRequirements(self.allocator);
            defer requirements.deinit();
            
            // Score against all models
            for (models) |*model| {
                var result = try self.scorer.scoreMatch(&requirements, model);
                
                if (result.match_score > best_score) {
                    if (best_result) |*old_result| {
                        old_result.deinit();
                    }
                    best_score = result.match_score;
                    best_model = model;
                    best_result = result;
                } else {
                    result.deinit();
                }
            }
            
            // Create assignment decision
            if (best_model) |model| {
                if (best_result) |*result| {
                    defer result.deinit();
                    
                    const decision = AssignmentDecision{
                        .agent_id = agent.agent_id,
                        .agent_name = agent.agent_name,
                        .model_id = model.model_id,
                        .model_name = model.model_name,
                        .match_score = result.match_score,
                        .assignment_method = .auto,
                        .capability_overlap = result.capability_overlap.items,
                        .missing_required = result.missing_required.items,
                    };
                    
                    try decisions.append(decision);
                }
            }
        }
        
        return decisions;
    }
    
    /// Optimal assignment: Maximize overall assignment quality
    fn assignOptimal(self: *AutoAssigner) !std.ArrayList(AssignmentDecision) {
        // For now, use greedy as optimal (can be enhanced with Hungarian algorithm)
        return try self.assignGreedy();
    }
    
    /// Balanced assignment: Balance between quality and model distribution
    fn assignBalanced(self: *AutoAssigner) !std.ArrayList(AssignmentDecision) {
        var decisions = std.ArrayList(AssignmentDecision).init(self.allocator);
        
        var online_agents = try self.agent_registry.getOnlineAgents(self.allocator);
        defer online_agents.deinit();
        
        const models = self.model_registry.getAllModels();
        
        // Track model usage counts
        var model_usage = std.StringHashMap(u32).init(self.allocator);
        defer model_usage.deinit();
        
        for (models) |model| {
            try model_usage.put(model.model_id, 0);
        }
        
        // For each agent, find best model considering current distribution
        for (online_agents.items) |agent| {
            var best_adjusted_score: f32 = 0.0;
            var best_model: ?*const ModelCapabilityProfile = null;
            var best_result: ?CapabilityMatchResult = null;
            
            var requirements = try agent.toCapabilityRequirements(self.allocator);
            defer requirements.deinit();
            
            for (models) |*model| {
                var result = try self.scorer.scoreMatch(&requirements, model);
                
                // Adjust score based on current usage (penalize overused models)
                const usage = model_usage.get(model.model_id) orelse 0;
                const usage_penalty = @as(f32, @floatFromInt(usage)) * 5.0; // 5 points per usage
                const adjusted_score = result.match_score - usage_penalty;
                
                if (adjusted_score > best_adjusted_score) {
                    if (best_result) |*old_result| {
                        old_result.deinit();
                    }
                    best_adjusted_score = adjusted_score;
                    best_model = model;
                    best_result = result;
                } else {
                    result.deinit();
                }
            }
            
            if (best_model) |model| {
                if (best_result) |*result| {
                    defer result.deinit();
                    
                    // Update usage count
                    const current_usage = model_usage.get(model.model_id) orelse 0;
                    try model_usage.put(model.model_id, current_usage + 1);
                    
                    const decision = AssignmentDecision{
                        .agent_id = agent.agent_id,
                        .agent_name = agent.agent_name,
                        .model_id = model.model_id,
                        .model_name = model.model_name,
                        .match_score = result.match_score,
                        .assignment_method = .auto,
                        .capability_overlap = result.capability_overlap.items,
                        .missing_required = result.missing_required.items,
                    };
                    
                    try decisions.append(decision);
                }
            }
        }
        
        return decisions;
    }
    
    /// Assign a specific model to a specific agent (manual assignment)
    pub fn assignManual(
        self: *AutoAssigner,
        agent_id: []const u8,
        model_id: []const u8,
    ) !AssignmentDecision {
        const agent_info = self.agent_registry.getAgentById(agent_id) orelse return error.AgentNotFound;
        const model = self.model_registry.getModelById(model_id) orelse return error.ModelNotFound;
        
        // Score the manual assignment
        var requirements = try agent_info.toCapabilityRequirements(self.allocator);
        defer requirements.deinit();
        
        var result = try self.scorer.scoreMatch(&requirements, model);
        defer result.deinit();
        
        return AssignmentDecision{
            .agent_id = agent_id,
            .agent_name = agent_info.agent_name,
            .model_id = model_id,
            .model_name = model.model_name,
            .match_score = result.match_score,
            .assignment_method = .manual,
            .capability_overlap = result.capability_overlap.items,
            .missing_required = result.missing_required.items,
        };
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Create sample agent registry for testing
pub fn createSampleAgentRegistry(allocator: std.mem.Allocator) !AgentRegistry {
    var registry = AgentRegistry.init(allocator);
    
    // Agent 1: GPU inference agent (coding focused)
    const agent1_caps = [_]ModelCapability{ .coding, .reasoning };
    const agent1_prefs = [_]ModelCapability{ .high_accuracy, .long_context };
    try registry.registerAgent(.{
        .agent_id = "agent_gpu_1",
        .agent_name = "GPU Inference Agent 1",
        .agent_type = .inference,
        .status = .online,
        .endpoint = "http://localhost:8001",
        .capabilities = &agent1_caps,
        .preferred_capabilities = &agent1_prefs,
        .min_context_length = 4096,
    });
    
    // Agent 2: GPU inference agent (general purpose)
    const agent2_caps = [_]ModelCapability{ .general, .reasoning };
    const agent2_prefs = [_]ModelCapability{ .multilingual, .long_context };
    try registry.registerAgent(.{
        .agent_id = "agent_gpu_2",
        .agent_name = "GPU Inference Agent 2",
        .agent_type = .inference,
        .status = .online,
        .endpoint = "http://localhost:8002",
        .capabilities = &agent2_caps,
        .preferred_capabilities = &agent2_prefs,
        .min_context_length = 4096,
    });
    
    // Agent 3: CPU inference agent (lightweight)
    const agent3_caps = [_]ModelCapability{ .general };
    const agent3_prefs = [_]ModelCapability{ .low_latency };
    try registry.registerAgent(.{
        .agent_id = "agent_cpu_1",
        .agent_name = "CPU Inference Agent 1",
        .agent_type = .inference,
        .status = .online,
        .endpoint = "http://localhost:8003",
        .capabilities = &agent3_caps,
        .preferred_capabilities = &agent3_prefs,
        .min_context_length = 2048,
    });
    
    return registry;
}

/// Create sample model registry for testing
pub fn createSampleModelRegistry(allocator: std.mem.Allocator) !ModelRegistry {
    var registry = ModelRegistry.init(allocator);
    
    // Register predefined model profiles
    try registry.registerModel(try capability_scorer.createLlama3_70BProfile(allocator));
    try registry.registerModel(try capability_scorer.createMistral7BProfile(allocator));
    try registry.registerModel(try capability_scorer.createTinyLlama1BProfile(allocator));
    
    return registry;
}

// ============================================================================
// UNIT TESTS
// ============================================================================

test "AgentRegistry: register and retrieve agents" {
    const allocator = std.testing.allocator;
    
    var registry = AgentRegistry.init(allocator);
    defer registry.deinit();
    
    const caps = [_]ModelCapability{ .coding };
    const prefs = [_]ModelCapability{ .high_accuracy };
    
    try registry.registerAgent(.{
        .agent_id = "test_agent",
        .agent_name = "Test Agent",
        .agent_type = .inference,
        .status = .online,
        .endpoint = "http://localhost:8000",
        .capabilities = &caps,
        .preferred_capabilities = &prefs,
        .min_context_length = 2048,
    });
    
    const agent = registry.getAgentById("test_agent");
    try std.testing.expect(agent != null);
    try std.testing.expectEqualStrings("Test Agent", agent.?.agent_name);
}

test "ModelRegistry: register and retrieve models" {
    const allocator = std.testing.allocator;
    
    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();
    
    var model = ModelCapabilityProfile.init(allocator, "test-model", "Test Model");
    try model.addCapability(.coding, 0.8);
    
    try registry.registerModel(model);
    
    const retrieved = registry.getModelById("test-model");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings("Test Model", retrieved.?.model_name);
}

test "AutoAssigner: greedy assignment" {
    const allocator = std.testing.allocator;
    
    var agent_registry = try createSampleAgentRegistry(allocator);
    defer agent_registry.deinit();
    
    var model_registry = try createSampleModelRegistry(allocator);
    defer model_registry.deinit();
    
    var assigner = AutoAssigner.init(allocator, &agent_registry, &model_registry);
    
    var decisions = try assigner.assignGreedy();
    defer decisions.deinit();
    
    // Should have 3 assignments (one per online agent)
    try std.testing.expectEqual(@as(usize, 3), decisions.items.len);
    
    // All assignments should have scores > 0
    for (decisions.items) |decision| {
        try std.testing.expect(decision.match_score > 0.0);
    }
}

test "AutoAssigner: balanced assignment" {
    const allocator = std.testing.allocator;
    
    var agent_registry = try createSampleAgentRegistry(allocator);
    defer agent_registry.deinit();
    
    var model_registry = try createSampleModelRegistry(allocator);
    defer model_registry.deinit();
    
    var assigner = AutoAssigner.init(allocator, &agent_registry, &model_registry);
    
    var decisions = try assigner.assignBalanced();
    defer decisions.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), decisions.items.len);
    
    // Verify model distribution (should spread across different models)
    var model_counts = std.StringHashMap(u32).init(allocator);
    defer model_counts.deinit();
    
    for (decisions.items) |decision| {
        const count = model_counts.get(decision.model_id) orelse 0;
        try model_counts.put(decision.model_id, count + 1);
    }
    
    // Should use at least 2 different models (balanced distribution)
    try std.testing.expect(model_counts.count() >= 2);
}

test "AutoAssigner: manual assignment" {
    const allocator = std.testing.allocator;
    
    var agent_registry = try createSampleAgentRegistry(allocator);
    defer agent_registry.deinit();
    
    var model_registry = try createSampleModelRegistry(allocator);
    defer model_registry.deinit();
    
    var assigner = AutoAssigner.init(allocator, &agent_registry, &model_registry);
    
    const decision = try assigner.assignManual("agent_gpu_1", "llama3-70b");
    
    try std.testing.expectEqualStrings("agent_gpu_1", decision.agent_id);
    try std.testing.expectEqualStrings("llama3-70b", decision.model_id);
    try std.testing.expect(decision.assignment_method == .manual);
    try std.testing.expect(decision.match_score > 0.0);
}

test "AgentInfo: convert to capability requirements" {
    const allocator = std.testing.allocator;
    
    const caps = [_]ModelCapability{ .coding, .reasoning };
    const prefs = [_]ModelCapability{ .high_accuracy };
    
    const agent = AgentInfo{
        .agent_id = "test",
        .agent_name = "Test",
        .agent_type = .inference,
        .status = .online,
        .endpoint = "http://localhost:8000",
        .capabilities = &caps,
        .preferred_capabilities = &prefs,
        .min_context_length = 4096,
    };
    
    var requirements = try agent.toCapabilityRequirements(allocator);
    defer requirements.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), requirements.required_capabilities.items.len);
    try std.testing.expectEqual(@as(usize, 1), requirements.preferred_capabilities.items.len);
    try std.testing.expectEqual(@as(u32, 4096), requirements.min_context_length);
}
