// ============================================================================
// Adaptive Router - Day 27 Implementation
// ============================================================================
// Purpose: Feedback loop for adaptive model selection using performance data
// Week: Week 6 (Days 26-30) - Performance Monitoring & Feedback Loop
// Phase: Month 2 - Model Router & Orchestration
// ============================================================================

const std = @import("std");
const capability_scorer = @import("capability_scorer.zig");
const auto_assign = @import("auto_assign.zig");
const performance_metrics = @import("performance_metrics.zig");
const HanaClient = @import("../../hana/core/client.zig").HanaClient;
const hana_queries = @import("../../hana/core/queries.zig");

// Type aliases
const ModelCapabilityProfile = capability_scorer.ModelCapabilityProfile;
const AgentCapabilityRequirements = capability_scorer.AgentCapabilityRequirements;
const CapabilityMatchResult = capability_scorer.CapabilityMatchResult;
const CapabilityScorer = capability_scorer.CapabilityScorer;
const PerformanceTracker = performance_metrics.PerformanceTracker;
const ModelMetrics = performance_metrics.PerformanceTracker.ModelMetrics;

// ============================================================================
// ADAPTIVE SCORING
// ============================================================================

/// Adaptive scorer that combines capability matching with performance data
pub const AdaptiveScorer = struct {
    allocator: std.mem.Allocator,
    capability_scorer: CapabilityScorer,
    performance_tracker: *PerformanceTracker,
    
    // Weighting configuration
    config: AdaptiveConfig,
    
    pub const AdaptiveConfig = struct {
        capability_weight: f32 = 0.60,     // 60% from capability matching
        success_rate_weight: f32 = 0.25,   // 25% from success rate
        latency_weight: f32 = 0.15,        // 15% from latency
        
        // Performance thresholds
        min_acceptable_success_rate: f32 = 0.90,  // 90%
        max_acceptable_latency_ms: f32 = 500.0,   // 500ms
        
        // Minimum data requirement
        min_requests_for_performance: u64 = 10,
        
        pub fn validate(self: *const AdaptiveConfig) bool {
            const total_weight = self.capability_weight + 
                                self.success_rate_weight + 
                                self.latency_weight;
            return total_weight >= 0.99 and total_weight <= 1.01;
        }
    };
    
    pub fn init(
        allocator: std.mem.Allocator,
        performance_tracker: *PerformanceTracker,
        config: AdaptiveConfig,
    ) AdaptiveScorer {
        return .{
            .allocator = allocator,
            .capability_scorer = CapabilityScorer.init(allocator),
            .performance_tracker = performance_tracker,
            .config = config,
        };
    }
    
    /// Score agent-model match with performance feedback
    pub fn scoreWithFeedback(
        self: *AdaptiveScorer,
        requirements: *const AgentCapabilityRequirements,
        model: *const ModelCapabilityProfile,
    ) !AdaptiveMatchResult {
        // Get capability-based score (0-100)
        var cap_result = try self.capability_scorer.scoreMatch(requirements, model);
        defer cap_result.deinit();
        
        const cap_score = cap_result.match_score;
        
        // Get performance data for this model
        const perf_data = self.performance_tracker.model_metrics.get(model.model_id);
        
        var final_score: f32 = cap_score;
        var performance_adjustment: f32 = 0.0;
        var has_performance_data = false;
        
        if (perf_data) |metrics| {
            if (metrics.total_requests >= self.config.min_requests_for_performance) {
                has_performance_data = true;
                
                // Calculate performance score components
                const success_score = metrics.getSuccessRate() * 100.0;
                const latency_score = self.calculateLatencyScore(metrics.getAvgLatency());
                
                // Weighted combination
                final_score = (cap_score * self.config.capability_weight) +
                             (success_score * self.config.success_rate_weight) +
                             (latency_score * self.config.latency_weight);
                
                performance_adjustment = final_score - cap_score;
                
                // Apply penalty for poor performance
                if (metrics.getSuccessRate() < self.config.min_acceptable_success_rate) {
                    final_score *= 0.5; // 50% penalty
                }
                
                if (metrics.getAvgLatency() > self.config.max_acceptable_latency_ms) {
                    final_score *= 0.8; // 20% penalty
                }
            }
        }
        
        return AdaptiveMatchResult{
            .agent_id = requirements.agent_id,
            .model_id = model.model_id,
            .capability_score = cap_score,
            .performance_score = if (has_performance_data) final_score else cap_score,
            .performance_adjustment = performance_adjustment,
            .has_performance_data = has_performance_data,
            .success_rate = if (perf_data) |m| m.getSuccessRate() else null,
            .avg_latency_ms = if (perf_data) |m| m.getAvgLatency() else null,
        };
    }
    
    /// Calculate latency score (0-100, lower latency = higher score)
    fn calculateLatencyScore(self: *const AdaptiveScorer, latency_ms: f32) f32 {
        // Linear scoring: 0ms = 100, max_acceptable = 50, 2x max = 0
        const max_lat = self.config.max_acceptable_latency_ms;
        
        if (latency_ms <= 0) return 100.0;
        if (latency_ms >= max_lat * 2.0) return 0.0;
        
        // Linear interpolation
        const score = 100.0 - (latency_ms / (max_lat * 2.0)) * 100.0;
        return @max(0.0, @min(100.0, score));
    }
};

/// Result of adaptive scoring
pub const AdaptiveMatchResult = struct {
    agent_id: []const u8,
    model_id: []const u8,
    capability_score: f32,        // Original capability-based score
    performance_score: f32,       // Adjusted with performance data
    performance_adjustment: f32,  // Difference (can be negative)
    has_performance_data: bool,
    success_rate: ?f32,
    avg_latency_ms: ?f32,
};

// ============================================================================
// ADAPTIVE AUTO-ASSIGNER
// ============================================================================

/// Enhanced auto-assigner with performance feedback
pub const AdaptiveAutoAssigner = struct {
    allocator: std.mem.Allocator,
    adaptive_scorer: AdaptiveScorer,
    agent_registry: *auto_assign.AgentRegistry,
    model_registry: *auto_assign.ModelRegistry,
    hana_client: ?*HanaClient,
    
    pub fn init(
        allocator: std.mem.Allocator,
        agent_registry: *auto_assign.AgentRegistry,
        model_registry: *auto_assign.ModelRegistry,
        performance_tracker: *PerformanceTracker,
        config: AdaptiveScorer.AdaptiveConfig,
    ) AdaptiveAutoAssigner {
        return .{
            .allocator = allocator,
            .adaptive_scorer = AdaptiveScorer.init(allocator, performance_tracker, config),
            .agent_registry = agent_registry,
            .model_registry = model_registry,
            .hana_client = null,
        };
    }
    
    pub fn initWithHana(
        allocator: std.mem.Allocator,
        agent_registry: *auto_assign.AgentRegistry,
        model_registry: *auto_assign.ModelRegistry,
        performance_tracker: *PerformanceTracker,
        config: AdaptiveScorer.AdaptiveConfig,
        hana_client: *HanaClient,
    ) AdaptiveAutoAssigner {
        return .{
            .allocator = allocator,
            .adaptive_scorer = AdaptiveScorer.init(allocator, performance_tracker, config),
            .agent_registry = agent_registry,
            .model_registry = model_registry,
            .hana_client = hana_client,
        };
    }
    
    /// Assign models using adaptive scoring with performance feedback
    pub fn assignAdaptive(self: *AdaptiveAutoAssigner) !std.ArrayList(AdaptiveAssignmentDecision) {
        var decisions = std.ArrayList(AdaptiveAssignmentDecision).init(self.allocator);
        
        // Get online agents
        var online_agents = try self.agent_registry.getOnlineAgents(self.allocator);
        defer online_agents.deinit();
        
        const models = self.model_registry.getAllModels();
        
        // For each agent, find best model using adaptive scoring
        for (online_agents.items) |agent| {
            var best_score: f32 = 0.0;
            var best_model: ?*const ModelCapabilityProfile = null;
            var best_result: ?AdaptiveMatchResult = null;
            
            // Convert agent to requirements
            var requirements = try agent.toCapabilityRequirements(self.allocator);
            defer requirements.deinit();
            
            // Score against all models with performance feedback
            for (models) |*model| {
                const result = try self.adaptive_scorer.scoreWithFeedback(&requirements, model);
                
                if (result.performance_score > best_score) {
                    best_score = result.performance_score;
                    best_model = model;
                    best_result = result;
                }
            }
            
            // Create assignment decision
            if (best_model) |model| {
                if (best_result) |result| {
                    const decision = AdaptiveAssignmentDecision{
                        .agent_id = agent.agent_id,
                        .agent_name = agent.agent_name,
                        .model_id = model.model_id,
                        .model_name = model.model_name,
                        .capability_score = result.capability_score,
                        .performance_score = result.performance_score,
                        .performance_adjustment = result.performance_adjustment,
                        .has_performance_data = result.has_performance_data,
                        .success_rate = result.success_rate,
                        .avg_latency_ms = result.avg_latency_ms,
                    };
                    
                    try decisions.append(decision);
                    
                    // Persist routing decision to HANA
                    if (self.hana_client) |client| {
                        const routing_decision = hana_queries.RoutingDecision{
                            .id = try hana_queries.generateDecisionId(self.allocator),
                            .request_id = try std.fmt.allocPrint(self.allocator, "adaptive_{d}", .{std.time.milliTimestamp()}),
                            .task_type = "adaptive_assignment",
                            .agent_id = agent.agent_id,
                            .model_id = model.model_id,
                            .capability_score = result.capability_score,
                            .performance_score = result.performance_score,
                            .composite_score = result.performance_score,
                            .strategy_used = "adaptive",
                            .latency_ms = if (result.avg_latency_ms) |lat| @as(i32, @intFromFloat(lat)) else 0,
                            .success = true,
                            .fallback_used = false,
                            .timestamp = std.time.milliTimestamp(),
                        };
                        defer {
                            self.allocator.free(routing_decision.id);
                            self.allocator.free(routing_decision.request_id);
                        }
                        
                        hana_queries.saveRoutingDecision(client, routing_decision) catch |err| {
                            std.log.warn("Failed to save routing decision to HANA: {}", .{err});
                        };
                    }
                }
            }
        }
        
        return decisions;
    }
};

/// Enhanced assignment decision with performance feedback
pub const AdaptiveAssignmentDecision = struct {
    agent_id: []const u8,
    agent_name: []const u8,
    model_id: []const u8,
    model_name: []const u8,
    capability_score: f32,
    performance_score: f32,
    performance_adjustment: f32,
    has_performance_data: bool,
    success_rate: ?f32,
    avg_latency_ms: ?f32,
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Create sample performance tracker with data
pub fn createSamplePerformanceTracker(allocator: std.mem.Allocator) !PerformanceTracker {
    var tracker = PerformanceTracker.init(allocator, 100);
    
    // Add sample data for llama3-70b (good performance)
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "llama3-{d}", .{i});
        defer allocator.free(id);
        
        var decision = try performance_metrics.RoutingDecision.init(
            allocator, id, "agent-1", "llama3-70b", 92.0
        );
        decision.latency_ms = 100.0 + @as(f32, @floatFromInt(i * 5));
        decision.success = true;
        try tracker.recordDecision(decision);
    }
    
    // Add sample data for mistral-7b (poor performance)
    i = 0;
    while (i < 20) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "mistral-{d}", .{i});
        defer allocator.free(id);
        
        var decision = try performance_metrics.RoutingDecision.init(
            allocator, id, "agent-2", "mistral-7b", 88.0
        );
        decision.latency_ms = 400.0 + @as(f32, @floatFromInt(i * 10));
        decision.success = (i < 12); // 60% success rate
        try tracker.recordDecision(decision);
    }
    
    return tracker;
}

// ============================================================================
// UNIT TESTS
// ============================================================================

test "AdaptiveScorer: score with performance feedback" {
    const allocator = std.testing.allocator;
    
    var tracker = try createSamplePerformanceTracker(allocator);
    defer tracker.deinit();
    
    const config = AdaptiveScorer.AdaptiveConfig{};
    var scorer = AdaptiveScorer.init(allocator, &tracker, config);
    
    var model = ModelCapabilityProfile.init(allocator, "llama3-70b", "LLaMA 3 70B");
    defer model.deinit();
    try model.addCapability(.coding, 0.9);
    try model.addCapability(.reasoning, 0.85);
    
    var requirements = AgentCapabilityRequirements.init(allocator, "agent-1", "Test Agent");
    defer requirements.deinit();
    try requirements.required_capabilities.append(.coding);
    
    const result = try scorer.scoreWithFeedback(&requirements, &model);
    
    try std.testing.expect(result.has_performance_data);
    try std.testing.expect(result.performance_score > 0.0);
}

test "AdaptiveScorer: penalty for poor performance" {
    const allocator = std.testing.allocator;
    
    var tracker = try createSamplePerformanceTracker(allocator);
    defer tracker.deinit();
    
    const config = AdaptiveScorer.AdaptiveConfig{};
    var scorer = AdaptiveScorer.init(allocator, &tracker, config);
    
    // Model with poor performance (60% success rate)
    var model = ModelCapabilityProfile.init(allocator, "mistral-7b", "Mistral 7B");
    defer model.deinit();
    try model.addCapability(.general, 0.8);
    
    var requirements = AgentCapabilityRequirements.init(allocator, "agent-2", "Test Agent");
    defer requirements.deinit();
    try requirements.required_capabilities.append(.general);
    
    const result = try scorer.scoreWithFeedback(&requirements, &model);
    
    try std.testing.expect(result.has_performance_data);
    try std.testing.expect(result.success_rate.? < 0.90); // Below threshold
    // Score should be penalized
    try std.testing.expect(result.performance_score < result.capability_score);
}

test "AdaptiveScorer: latency score calculation" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    const config = AdaptiveScorer.AdaptiveConfig{};
    var scorer = AdaptiveScorer.init(allocator, &tracker, config);
    
    // Test latency scoring
    try std.testing.expectEqual(@as(f32, 100.0), scorer.calculateLatencyScore(0.0));
    try std.testing.expectEqual(@as(f32, 50.0), scorer.calculateLatencyScore(500.0));
    try std.testing.expectEqual(@as(f32, 0.0), scorer.calculateLatencyScore(1000.0));
}

test "AdaptiveAutoAssigner: adaptive assignment" {
    const allocator = std.testing.allocator;
    
    var agent_registry = try auto_assign.createSampleAgentRegistry(allocator);
    defer agent_registry.deinit();
    
    var model_registry = try auto_assign.createSampleModelRegistry(allocator);
    defer model_registry.deinit();
    
    var tracker = try createSamplePerformanceTracker(allocator);
    defer tracker.deinit();
    
    const config = AdaptiveScorer.AdaptiveConfig{};
    var assigner = AdaptiveAutoAssigner.init(
        allocator,
        &agent_registry,
        &model_registry,
        &tracker,
        config,
    );
    
    var decisions = try assigner.assignAdaptive();
    defer decisions.deinit();
    
    // Should have 3 assignments
    try std.testing.expectEqual(@as(usize, 3), decisions.items.len);
    
    // Verify decisions have performance data
    for (decisions.items) |decision| {
        try std.testing.expect(decision.performance_score > 0.0);
    }
}

test "AdaptiveConfig: weight validation" {
    const valid_config = AdaptiveScorer.AdaptiveConfig{
        .capability_weight = 0.60,
        .success_rate_weight = 0.25,
        .latency_weight = 0.15,
    };
    try std.testing.expect(valid_config.validate());
    
    const invalid_config = AdaptiveScorer.AdaptiveConfig{
        .capability_weight = 0.50,
        .success_rate_weight = 0.30,
        .latency_weight = 0.10,
        // Total = 0.90 (invalid)
    };
    try std.testing.expect(!invalid_config.validate());
}
