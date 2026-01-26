// ============================================================================
// Capability Scorer - Day 22 Implementation
// ============================================================================
// Purpose: Score agent-model pairs based on capability matching
// Week: Week 5 (Days 21-25) - Model Router Foundation
// Phase: Month 2 - Model Router & Orchestration
// ============================================================================

const std = @import("std");

// ============================================================================
// ENUMS: Model Capabilities
// ============================================================================

/// Model capabilities that can be matched against task types
pub const ModelCapability = enum {
    coding,           // Code generation, debugging, refactoring
    math,            // Mathematical reasoning, calculations
    reasoning,       // Logical reasoning, problem solving
    arabic,          // Arabic language understanding and generation
    general,         // General-purpose text generation
    multilingual,    // Support for multiple languages
    long_context,    // Ability to handle long context windows (>8K tokens)
    low_latency,     // Optimized for fast inference
    high_accuracy,   // Optimized for accuracy over speed
    function_calling, // Tool/function calling support
    
    pub fn toString(self: ModelCapability) []const u8 {
        return @tagName(self);
    }
    
    pub fn fromString(str: []const u8) ?ModelCapability {
        inline for (@typeInfo(ModelCapability).Enum.fields) |field| {
            if (std.mem.eql(u8, str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }
};

/// Task types that can be routed to appropriate models
pub const TaskType = enum {
    coding,
    math,
    reasoning,
    arabic,
    general,
    translation,
    summarization,
    question_answering,
    
    pub fn toString(self: TaskType) []const u8 {
        return @tagName(self);
    }
    
    pub fn fromString(str: []const u8) ?TaskType {
        inline for (@typeInfo(TaskType).Enum.fields) |field| {
            if (std.mem.eql(u8, str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }
};

/// Specialized task categories used to benchmark agents within a type
pub const TaskCategory = enum {
    math,
    time_series,
    relational,
    graph,
    code,
    vector_search,
    ocr_extraction,
    reasoning,
    summarization,

    pub fn toString(self: TaskCategory) []const u8 {
        return @tagName(self);
    }
};

/// Benchmark/dataset metadata attached to agents
pub const TaskProfile = struct {
    category: TaskCategory,
    benchmark: []const u8,
    dataset: []const u8,
    score: f32 = 0.0,
};

// ============================================================================
// STRUCTS: Capability Definitions
// ============================================================================

/// Model capability profile with weighted scores
pub const ModelCapabilityProfile = struct {
    model_id: []const u8,
    model_name: []const u8,
    capabilities: std.AutoHashMap(ModelCapability, f32),
    context_length: u32,
    parameters_billions: f32,
    
    pub fn init(allocator: std.mem.Allocator, model_id: []const u8, model_name: []const u8) ModelCapabilityProfile {
        return .{
            .model_id = model_id,
            .model_name = model_name,
            .capabilities = std.AutoHashMap(ModelCapability, f32).init(allocator),
            .context_length = 4096,
            .parameters_billions = 7.0,
        };
    }
    
    pub fn deinit(self: *ModelCapabilityProfile) void {
        self.capabilities.deinit();
    }
    
    /// Add a capability with its strength (0.0 - 1.0)
    pub fn addCapability(self: *ModelCapabilityProfile, capability: ModelCapability, strength: f32) !void {
        try self.capabilities.put(capability, std.math.clamp(strength, 0.0, 1.0));
    }
    
    /// Get capability strength (returns 0.0 if not present)
    pub fn getCapabilityStrength(self: *const ModelCapabilityProfile, capability: ModelCapability) f32 {
        return self.capabilities.get(capability) orelse 0.0;
    }
};

/// Agent capability requirements
pub const AgentCapabilityRequirements = struct {
    agent_id: []const u8,
    agent_name: []const u8,
    required_capabilities: std.ArrayList(ModelCapability),
    preferred_capabilities: std.ArrayList(ModelCapability),
    min_context_length: u32,
    task_profiles: std.ArrayList(TaskProfile),
    
    pub fn init(allocator: std.mem.Allocator, agent_id: []const u8, agent_name: []const u8) AgentCapabilityRequirements {
        return .{
            .agent_id = agent_id,
            .agent_name = agent_name,
            .required_capabilities = try std.ArrayList(ModelCapability).initCapacity(allocator, 0),
            .preferred_capabilities = try std.ArrayList(ModelCapability).initCapacity(allocator, 0),
            .min_context_length = 2048,
            .task_profiles = try std.ArrayList(TaskProfile).initCapacity(allocator, 0),
        };
    }
    
    pub fn deinit(self: *AgentCapabilityRequirements) void {
        self.required_capabilities.deinit();
        self.preferred_capabilities.deinit();
        self.task_profiles.deinit();
    }
};

/// Capability match result
pub const CapabilityMatchResult = struct {
    agent_id: []const u8,
    model_id: []const u8,
    match_score: f32, // 0.0 - 100.0
    required_match_count: u32,
    preferred_match_count: u32,
    total_required: u32,
    total_preferred: u32,
    capability_overlap: std.ArrayList(ModelCapability),
    missing_required: std.ArrayList(ModelCapability),
    
    pub fn init(allocator: std.mem.Allocator, agent_id: []const u8, model_id: []const u8) CapabilityMatchResult {
        return .{
            .agent_id = agent_id,
            .model_id = model_id,
            .match_score = 0.0,
            .required_match_count = 0,
            .preferred_match_count = 0,
            .total_required = 0,
            .total_preferred = 0,
            .capability_overlap = try std.ArrayList(ModelCapability).initCapacity(allocator, 0),
            .missing_required = try std.ArrayList(ModelCapability).initCapacity(allocator, 0),
        };
    }
    
    pub fn deinit(self: *CapabilityMatchResult) void {
        self.capability_overlap.deinit();
        self.missing_required.deinit();
    }
};

// ============================================================================
// CAPABILITY SCORER: Core Logic
// ============================================================================

pub const CapabilityScorer = struct {
    allocator: std.mem.Allocator,
    
    /// Weights for scoring components
    const REQUIRED_CAPABILITY_WEIGHT: f32 = 0.70;
    const PREFERRED_CAPABILITY_WEIGHT: f32 = 0.20;
    const CONTEXT_LENGTH_WEIGHT: f32 = 0.10;
    
    pub fn init(allocator: std.mem.Allocator) CapabilityScorer {
        return .{ .allocator = allocator };
    }
    
    /// Calculate match score between agent requirements and model capabilities
    pub fn scoreMatch(
        self: *CapabilityScorer,
        agent: *const AgentCapabilityRequirements,
        model: *const ModelCapabilityProfile,
    ) !CapabilityMatchResult {
        var result = CapabilityMatchResult.init(self.allocator, agent.agent_id, model.model_id);
        
        result.total_required = @intCast(agent.required_capabilities.items.len);
        result.total_preferred = @intCast(agent.preferred_capabilities.items.len);
        
        // Score required capabilities
        var required_score: f32 = 0.0;
        for (agent.required_capabilities.items) |capability| {
            const strength = model.getCapabilityStrength(capability);
            if (strength > 0.0) {
                result.required_match_count += 1;
                required_score += strength;
                try result.capability_overlap.append(capability);
            } else {
                try result.missing_required.append(capability);
            }
        }
        
        // Normalize required score
        if (result.total_required > 0) {
            required_score = required_score / @as(f32, @floatFromInt(result.total_required));
        } else {
            required_score = 1.0; // If no requirements, give full score
        }
        
        // Score preferred capabilities
        var preferred_score: f32 = 0.0;
        for (agent.preferred_capabilities.items) |capability| {
            const strength = model.getCapabilityStrength(capability);
            if (strength > 0.0) {
                result.preferred_match_count += 1;
                preferred_score += strength;
                // Add to overlap if not already present
                var already_in_overlap = false;
                for (result.capability_overlap.items) |overlap_cap| {
                    if (overlap_cap == capability) {
                        already_in_overlap = true;
                        break;
                    }
                }
                if (!already_in_overlap) {
                    try result.capability_overlap.append(capability);
                }
            }
        }
        
        // Normalize preferred score
        if (result.total_preferred > 0) {
            preferred_score = preferred_score / @as(f32, @floatFromInt(result.total_preferred));
        } else {
            preferred_score = 1.0; // If no preferences, give full score
        }
        
        // Score context length compatibility
        const context_score: f32 = if (model.context_length >= agent.min_context_length) 1.0 else @as(f32, @floatFromInt(model.context_length)) / @as(f32, @floatFromInt(agent.min_context_length));
        
        // Calculate composite score (0.0 - 100.0)
        result.match_score = (
            (required_score * REQUIRED_CAPABILITY_WEIGHT) +
            (preferred_score * PREFERRED_CAPABILITY_WEIGHT) +
            (context_score * CONTEXT_LENGTH_WEIGHT)
        ) * 100.0;
        
        return result;
    }
    
    /// Score all agent-model pairs and return sorted results
    pub fn scoreAllPairs(
        self: *CapabilityScorer,
        agents: []const AgentCapabilityRequirements,
        models: []const ModelCapabilityProfile,
    ) !std.ArrayList(CapabilityMatchResult) {
        var results = try std.ArrayList(CapabilityMatchResult).initCapacity(self.allocator, 0);
        
        for (agents) |*agent| {
            for (models) |*model| {
                const match_result = try self.scoreMatch(agent, model);
                try results.append(match_result);
            }
        }
        
        // Sort by match score descending
        std.sort.pdq(CapabilityMatchResult, results.items, {}, compareMatchScores);
        
        return results;
    }
    
    /// Map task type to required capabilities
    pub fn getRequiredCapabilitiesForTask(task_type: TaskType, allocator: std.mem.Allocator) !std.ArrayList(ModelCapability) {
        var capabilities = try std.ArrayList(ModelCapability).initCapacity(allocator, 0);
        
        switch (task_type) {
            .coding => {
                try capabilities.append(.coding);
                try capabilities.append(.reasoning);
            },
            .math => {
                try capabilities.append(.math);
                try capabilities.append(.reasoning);
            },
            .reasoning => {
                try capabilities.append(.reasoning);
            },
            .arabic => {
                try capabilities.append(.arabic);
                try capabilities.append(.multilingual);
            },
            .general => {
                try capabilities.append(.general);
            },
            .translation => {
                try capabilities.append(.multilingual);
                try capabilities.append(.general);
            },
            .summarization => {
                try capabilities.append(.reasoning);
                try capabilities.append(.general);
            },
            .question_answering => {
                try capabilities.append(.reasoning);
                try capabilities.append(.general);
            },
        }
        
        return capabilities;
    }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn compareMatchScores(context: void, a: CapabilityMatchResult, b: CapabilityMatchResult) bool {
    _ = context;
    return a.match_score > b.match_score;
}

// ============================================================================
// PREDEFINED MODEL PROFILES
// ============================================================================

/// Create a profile for LLaMA 3 70B
pub fn createLlama3_70BProfile(allocator: std.mem.Allocator) !ModelCapabilityProfile {
    var profile = ModelCapabilityProfile.init(allocator, "llama3-70b", "LLaMA 3 70B");
    profile.context_length = 8192;
    profile.parameters_billions = 70.0;
    
    try profile.addCapability(.coding, 0.90);
    try profile.addCapability(.math, 0.85);
    try profile.addCapability(.reasoning, 0.95);
    try profile.addCapability(.general, 0.95);
    try profile.addCapability(.multilingual, 0.80);
    try profile.addCapability(.long_context, 0.85);
    try profile.addCapability(.high_accuracy, 0.90);
    
    return profile;
}

/// Create a profile for Mistral 7B
pub fn createMistral7BProfile(allocator: std.mem.Allocator) !ModelCapabilityProfile {
    var profile = ModelCapabilityProfile.init(allocator, "mistral-7b", "Mistral 7B");
    profile.context_length = 32768;
    profile.parameters_billions = 7.0;
    
    try profile.addCapability(.coding, 0.80);
    try profile.addCapability(.math, 0.75);
    try profile.addCapability(.reasoning, 0.85);
    try profile.addCapability(.general, 0.85);
    try profile.addCapability(.multilingual, 0.70);
    try profile.addCapability(.long_context, 0.95);
    try profile.addCapability(.low_latency, 0.85);
    
    return profile;
}

/// Create a profile for TinyLLaMA 1.1B
pub fn createTinyLlama1BProfile(allocator: std.mem.Allocator) !ModelCapabilityProfile {
    var profile = ModelCapabilityProfile.init(allocator, "tinyllama-1b", "TinyLLaMA 1.1B");
    profile.context_length = 2048;
    profile.parameters_billions = 1.1;
    
    try profile.addCapability(.general, 0.70);
    try profile.addCapability(.reasoning, 0.60);
    try profile.addCapability(.low_latency, 0.95);
    try profile.addCapability(.coding, 0.50);
    
    return profile;
}

// ============================================================================
// UNIT TESTS
// ============================================================================

test "ModelCapability enum to/from string" {
    const coding = ModelCapability.coding;
    const coding_str = coding.toString();
    try std.testing.expectEqualStrings("coding", coding_str);
    
    const parsed = ModelCapability.fromString("math");
    try std.testing.expect(parsed != null);
    try std.testing.expect(parsed.? == ModelCapability.math);
}

test "TaskType enum to/from string" {
    const task = TaskType.reasoning;
    const task_str = task.toString();
    try std.testing.expectEqualStrings("reasoning", task_str);
    
    const parsed = TaskType.fromString("coding");
    try std.testing.expect(parsed != null);
    try std.testing.expect(parsed.? == TaskType.coding);
}

test "ModelCapabilityProfile basic operations" {
    const allocator = std.testing.allocator;
    
    var profile = ModelCapabilityProfile.init(allocator, "test-model", "Test Model");
    defer profile.deinit();
    
    try profile.addCapability(.coding, 0.85);
    try profile.addCapability(.math, 0.90);
    
    const coding_strength = profile.getCapabilityStrength(.coding);
    try std.testing.expectApproxEqAbs(@as(f32, 0.85), coding_strength, 0.001);
    
    const reasoning_strength = profile.getCapabilityStrength(.reasoning);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), reasoning_strength, 0.001);
}

test "CapabilityScorer: perfect match" {
    const allocator = std.testing.allocator;
    
    var scorer = CapabilityScorer.init(allocator);
    
    // Create model with coding capability
    var model = ModelCapabilityProfile.init(allocator, "test-model", "Test Model");
    defer model.deinit();
    try model.addCapability(.coding, 1.0);
    model.context_length = 4096;
    
    // Create agent requiring coding
    var agent = AgentCapabilityRequirements.init(allocator, "test-agent", "Test Agent");
    defer agent.deinit();
    try agent.required_capabilities.append(.coding);
    agent.min_context_length = 2048;
    
    var result = try scorer.scoreMatch(&agent, &model);
    defer result.deinit();
    
    // Should be close to 100 (required match + context match)
    try std.testing.expect(result.match_score > 80.0);
    try std.testing.expectEqual(@as(u32, 1), result.required_match_count);
}

test "CapabilityScorer: no match" {
    const allocator = std.testing.allocator;
    
    var scorer = CapabilityScorer.init(allocator);
    
    // Create model with coding capability
    var model = ModelCapabilityProfile.init(allocator, "test-model", "Test Model");
    defer model.deinit();
    try model.addCapability(.coding, 1.0);
    
    // Create agent requiring math (not present in model)
    var agent = AgentCapabilityRequirements.init(allocator, "test-agent", "Test Agent");
    defer agent.deinit();
    try agent.required_capabilities.append(.math);
    
    var result = try scorer.scoreMatch(&agent, &model);
    defer result.deinit();
    
    // Should have low score due to missing required capability
    try std.testing.expect(result.match_score < 50.0);
    try std.testing.expectEqual(@as(u32, 0), result.required_match_count);
    try std.testing.expectEqual(@as(u32, 1), result.missing_required.items.len);
}

test "CapabilityScorer: task type mapping" {
    const allocator = std.testing.allocator;
    
    var coding_caps = try CapabilityScorer.getRequiredCapabilitiesForTask(.coding, allocator);
    defer coding_caps.deinit();
    
    try std.testing.expect(coding_caps.items.len >= 2);
    
    var has_coding = false;
    var has_reasoning = false;
    for (coding_caps.items) |cap| {
        if (cap == .coding) has_coding = true;
        if (cap == .reasoning) has_reasoning = true;
    }
    
    try std.testing.expect(has_coding);
    try std.testing.expect(has_reasoning);
}

test "Predefined model profiles" {
    const allocator = std.testing.allocator;
    
    var llama3 = try createLlama3_70BProfile(allocator);
    defer llama3.deinit();
    
    try std.testing.expect(llama3.context_length == 8192);
    try std.testing.expect(llama3.getCapabilityStrength(.reasoning) > 0.9);
    
    var mistral = try createMistral7BProfile(allocator);
    defer mistral.deinit();
    
    try std.testing.expect(mistral.context_length == 32768);
    try std.testing.expect(mistral.getCapabilityStrength(.long_context) > 0.9);
}
