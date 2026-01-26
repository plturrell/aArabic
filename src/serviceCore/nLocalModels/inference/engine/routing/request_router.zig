//! Request Router - Day 14
//! Intelligent request routing for multi-model serving
//!
//! Features:
//! - Multiple routing strategies (round-robin, least-loaded, cache-aware, etc.)
//! - Health-aware routing (skip unhealthy models)
//! - Load balancing across model instances
//! - Cache hit optimization
//! - Quota-aware routing
//! - A/B testing support
//! - Request affinity (sticky routing)
//! - Comprehensive metrics

const std = @import("std");
const ModelRegistry = @import("../../shared/model_registry.zig").ModelRegistry;
const MultiModelCacheManager = @import("../tiering/multi_model_cache.zig").MultiModelCacheManager;
const ResourceQuotaManager = @import("../tiering/resource_quotas.zig").ResourceQuotaManager;
const log = @import("../tiering/structured_logging.zig");

// ============================================================================
// Routing Strategy
// ============================================================================

pub const RoutingStrategy = enum {
    round_robin,           // Rotate through models
    least_loaded,          // Route to least loaded model
    cache_aware,           // Prefer models with cache hits
    quota_aware,           // Avoid models near quota limits
    random,                // Random selection
    weighted_random,       // Random with weights
    latency_based,         // Prefer lowest latency models
    affinity_based,        // Sticky routing based on session
};

pub const RoutingConfig = struct {
    strategy: RoutingStrategy = .least_loaded,
    
    /// Strategy-specific settings
    enable_health_checks: bool = true,
    enable_quota_checks: bool = true,
    enable_cache_optimization: bool = true,
    
    /// Load balancing
    max_load_threshold: f32 = 0.9,     // 90% max load
    min_cache_hit_rate: f32 = 0.3,     // 30% min cache hit rate
    
    /// Affinity settings
    affinity_timeout_sec: u32 = 300,   // 5 minutes
    
    /// A/B testing
    ab_test_enabled: bool = false,
    ab_test_split: f32 = 0.5,          // 50/50 split
    ab_test_model_a: ?[]const u8 = null,
    ab_test_model_b: ?[]const u8 = null,
    
    /// Retry settings
    max_retries: u32 = 3,
    retry_delay_ms: u32 = 100,
};

// ============================================================================
// Routing Decision
// ============================================================================

pub const RoutingDecision = struct {
    model_id: []const u8,
    reason: []const u8,
    
    /// Scoring factors
    load_score: f32 = 0.0,
    cache_score: f32 = 0.0,
    quota_score: f32 = 0.0,
    health_score: f32 = 0.0,
    total_score: f32 = 0.0,
    
    /// Metadata
    attempted_models: u32 = 0,
    fallback_used: bool = false,
    
    pub fn format(
        self: RoutingDecision,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("RoutingDecision{{ model={s}, score={d:.2}, reason={s} }}", .{
            self.model_id,
            self.total_score,
            self.reason,
        });
    }
};

// ============================================================================
// Model Candidate
// ============================================================================

const ModelCandidate = struct {
    model_id: []const u8,
    score: f32,
    load: f32,
    cache_hit_rate: f32,
    quota_available: f32,
    is_healthy: bool,
};

// ============================================================================
// Request Router
// ============================================================================

pub const RequestRouter = struct {
    allocator: std.mem.Allocator,
    config: RoutingConfig,
    
    /// Integration points
    registry: ?*ModelRegistry = null,
    cache_manager: ?*MultiModelCacheManager = null,
    quota_manager: ?*ResourceQuotaManager = null,
    
    /// Routing state
    round_robin_index: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    
    /// Affinity tracking (session_id -> model_id)
    affinity_map: std.StringHashMap(AffinityEntry),
    affinity_mutex: std.Thread.Mutex = .{},
    
    /// Statistics
    stats: RoutingStats,
    stats_mutex: std.Thread.Mutex = .{},
    
    const AffinityEntry = struct {
        model_id: []const u8,
        last_access: i64,
        request_count: u64,
    };
    
    pub const RoutingStats = struct {
        total_routes: u64 = 0,
        successful_routes: u64 = 0,
        failed_routes: u64 = 0,
        fallback_routes: u64 = 0,
        
        /// Per-strategy counters
        round_robin_count: u64 = 0,
        least_loaded_count: u64 = 0,
        cache_aware_count: u64 = 0,
        quota_aware_count: u64 = 0,
        affinity_count: u64 = 0,
        
        /// Performance
        total_routing_time_us: u64 = 0,
        avg_routing_time_us: f32 = 0.0,
    };
    
    pub fn init(
        allocator: std.mem.Allocator,
        config: RoutingConfig,
    ) !*RequestRouter {
        log.info("Initializing Request Router: strategy={s}", .{@tagName(config.strategy)});
        
        const self = try allocator.create(RequestRouter);
        errdefer allocator.destroy(self);
        
        self.* = RequestRouter{
            .allocator = allocator,
            .config = config,
            .affinity_map = std.StringHashMap(AffinityEntry).init(allocator),
            .stats = .{},
        };
        
        log.info("Request Router initialized successfully", .{});
        return self;
    }
    
    pub fn deinit(self: *RequestRouter) void {
        // Clean up affinity map
        var iter = self.affinity_map.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.model_id);
        }
        self.affinity_map.deinit();
        
        self.allocator.destroy(self);
    }
    
    /// Set integration components
    pub fn setRegistry(self: *RequestRouter, registry: *ModelRegistry) void {
        self.registry = registry;
    }
    
    pub fn setCacheManager(self: *RequestRouter, cache_manager: *MultiModelCacheManager) void {
        self.cache_manager = cache_manager;
    }
    
    pub fn setQuotaManager(self: *RequestRouter, quota_manager: *ResourceQuotaManager) void {
        self.quota_manager = quota_manager;
    }
    
    /// Route a request to the best available model
    pub fn route(
        self: *RequestRouter,
        request: struct {
            session_id: ?[]const u8 = null,
            preferred_model: ?[]const u8 = null,
            estimated_tokens: u64 = 100,
            required_capabilities: []const []const u8 = &[_][]const u8{},
        },
    ) !RoutingDecision {
        const start_time = std.time.microTimestamp();
        
        self.stats_mutex.lock();
        self.stats.total_routes += 1;
        self.stats_mutex.unlock();
        
        // 1. Check for preferred model
        if (request.preferred_model) |model_id| {
            if (try self.isModelAvailable(model_id, request.estimated_tokens)) {
                return self.recordDecision(
                    model_id,
                    "Preferred model available",
                    1.0,
                    start_time,
                );
            }
        }
        
        // 2. Check affinity (sticky routing)
        if (request.session_id) |session_id| {
            if (try self.checkAffinity(session_id, request.estimated_tokens)) |model_id| {
                return self.recordDecision(
                    model_id,
                    "Affinity-based routing",
                    0.95,
                    start_time,
                );
            }
        }
        
        // 3. A/B testing
        if (self.config.ab_test_enabled) {
            if (try self.routeABTest(request.estimated_tokens)) |decision| {
                return self.recordDecisionWithTime(decision, start_time);
            }
        }
        
        // 4. Apply routing strategy
        const decision = switch (self.config.strategy) {
            .round_robin => try self.routeRoundRobin(request.estimated_tokens),
            .least_loaded => try self.routeLeastLoaded(request.estimated_tokens),
            .cache_aware => try self.routeCacheAware(request.estimated_tokens),
            .quota_aware => try self.routeQuotaAware(request.estimated_tokens),
            .random => try self.routeRandom(request.estimated_tokens),
            .weighted_random => try self.routeWeightedRandom(request.estimated_tokens),
            .latency_based => try self.routeLatencyBased(request.estimated_tokens),
            .affinity_based => try self.routeAffinityBased(request.session_id, request.estimated_tokens),
        };
        
        // 5. Set up affinity for future requests
        if (request.session_id) |session_id| {
            try self.setAffinity(session_id, decision.model_id);
        }
        
        return self.recordDecisionWithTime(decision, start_time);
    }
    
    // ========================================================================
    // Routing Strategies
    // ========================================================================
    
    fn routeRoundRobin(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
        const models = try self.getAvailableModels(estimated_tokens);
        defer self.allocator.free(models);
        
        if (models.len == 0) {
            return error.NoAvailableModels;
        }
        
        const index = self.round_robin_index.fetchAdd(1, .monotonic) % models.len;
        
        self.stats_mutex.lock();
        self.stats.round_robin_count += 1;
        self.stats_mutex.unlock();
        
        return RoutingDecision{
            .model_id = models[index],
            .reason = "Round-robin selection",
            .total_score = 1.0,
        };
    }
    
    fn routeLeastLoaded(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
        const candidates = try self.scoreModels(estimated_tokens);
        defer {
            for (candidates) |c| self.allocator.free(c.model_id);
            self.allocator.free(candidates);
        }
        
        if (candidates.len == 0) {
            return error.NoAvailableModels;
        }
        
        // Find model with lowest load
        var best = candidates[0];
        for (candidates[1..]) |candidate| {
            if (candidate.load < best.load) {
                best = candidate;
            }
        }
        
        self.stats_mutex.lock();
        self.stats.least_loaded_count += 1;
        self.stats_mutex.unlock();
        
        return RoutingDecision{
            .model_id = try self.allocator.dupe(u8, best.model_id),
            .reason = "Least loaded model",
            .load_score = 1.0 - best.load,
            .total_score = best.score,
        };
    }
    
    fn routeCacheAware(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
        const candidates = try self.scoreModels(estimated_tokens);
        defer {
            for (candidates) |c| self.allocator.free(c.model_id);
            self.allocator.free(candidates);
        }
        
        if (candidates.len == 0) {
            return error.NoAvailableModels;
        }
        
        // Find model with best cache hit rate
        var best = candidates[0];
        for (candidates[1..]) |candidate| {
            if (candidate.cache_hit_rate > best.cache_hit_rate) {
                best = candidate;
            }
        }
        
        self.stats_mutex.lock();
        self.stats.cache_aware_count += 1;
        self.stats_mutex.unlock();
        
        return RoutingDecision{
            .model_id = try self.allocator.dupe(u8, best.model_id),
            .reason = "Cache-aware routing",
            .cache_score = best.cache_hit_rate,
            .total_score = best.score,
        };
    }
    
    fn routeQuotaAware(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
        const candidates = try self.scoreModels(estimated_tokens);
        defer {
            for (candidates) |c| self.allocator.free(c.model_id);
            self.allocator.free(candidates);
        }
        
        if (candidates.len == 0) {
            return error.NoAvailableModels;
        }
        
        // Find model with most available quota
        var best = candidates[0];
        for (candidates[1..]) |candidate| {
            if (candidate.quota_available > best.quota_available) {
                best = candidate;
            }
        }
        
        self.stats_mutex.lock();
        self.stats.quota_aware_count += 1;
        self.stats_mutex.unlock();
        
        return RoutingDecision{
            .model_id = try self.allocator.dupe(u8, best.model_id),
            .reason = "Quota-aware routing",
            .quota_score = best.quota_available,
            .total_score = best.score,
        };
    }
    
    fn routeRandom(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
        const models = try self.getAvailableModels(estimated_tokens);
        defer self.allocator.free(models);
        
        if (models.len == 0) {
            return error.NoAvailableModels;
        }
        
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = prng.random();
        const index = random.intRangeAtMost(usize, 0, models.len - 1);
        
        return RoutingDecision{
            .model_id = models[index],
            .reason = "Random selection",
            .total_score = 1.0,
        };
    }
    
    fn routeWeightedRandom(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
        const candidates = try self.scoreModels(estimated_tokens);
        defer {
            for (candidates) |c| self.allocator.free(c.model_id);
            self.allocator.free(candidates);
        }
        
        if (candidates.len == 0) {
            return error.NoAvailableModels;
        }
        
        // Calculate total weight
        var total_weight: f32 = 0.0;
        for (candidates) |c| {
            total_weight += c.score;
        }
        
        // Random selection based on weights
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = prng.random();
        var target = random.float(f32) * total_weight;
        
        for (candidates) |c| {
            target -= c.score;
            if (target <= 0.0) {
                return RoutingDecision{
                    .model_id = try self.allocator.dupe(u8, c.model_id),
                    .reason = "Weighted random selection",
                    .total_score = c.score,
                };
            }
        }
        
        // Fallback to first candidate
        return RoutingDecision{
            .model_id = try self.allocator.dupe(u8, candidates[0].model_id),
            .reason = "Weighted random fallback",
            .total_score = candidates[0].score,
        };
    }
    
    fn routeLatencyBased(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
        // For now, use cache-aware as proxy for latency
        // TODO: Add actual latency tracking
        return self.routeCacheAware(estimated_tokens);
    }
    
    fn routeAffinityBased(
        self: *RequestRouter,
        session_id: ?[]const u8,
        estimated_tokens: u64,
    ) !RoutingDecision {
        if (session_id) |sid| {
            if (try self.checkAffinity(sid, estimated_tokens)) |model_id| {
                return RoutingDecision{
                    .model_id = model_id,
                    .reason = "Affinity-based routing",
                    .total_score = 1.0,
                };
            }
        }
        
        // Fallback to least loaded
        return self.routeLeastLoaded(estimated_tokens);
    }
    
    // ========================================================================
    // Helper Functions
    // ========================================================================
    
    fn getAvailableModels(self: *RequestRouter, estimated_tokens: u64) ![][]const u8 {
        const registry = self.registry orelse return error.RegistryNotSet;
        
        const all_models = try registry.listModels(self.allocator);
        var available = std.ArrayList([]const u8){};
        errdefer {
            for (available.items) |m| self.allocator.free(m);
            available.deinit();
        }
        
        for (all_models) |model_id| {
            if (try self.isModelAvailable(model_id, estimated_tokens)) {
                try available.append(try self.allocator.dupe(u8, model_id));
            }
        }
        
        // Clean up all_models
        for (all_models) |m| self.allocator.free(m);
        self.allocator.free(all_models);
        
        return available.toOwnedSlice();
    }
    
    fn isModelAvailable(self: *RequestRouter, model_id: []const u8, estimated_tokens: u64) !bool {
        // Check health
        if (self.config.enable_health_checks) {
            const registry = self.registry orelse return true;
            const config = registry.get(model_id) orelse return false;
            if (!config.is_active) return false;
        }
        
        // Check quota
        if (self.config.enable_quota_checks) {
            if (self.quota_manager) |qm| {
                const result = try qm.checkQuota(model_id, .{
                    .estimated_tokens = estimated_tokens,
                });
                if (!result.allowed) return false;
            }
        }
        
        return true;
    }
    
    fn scoreModels(self: *RequestRouter, estimated_tokens: u64) ![]ModelCandidate {
        const models = try self.getAvailableModels(estimated_tokens);
        defer {
            for (models) |m| self.allocator.free(m);
            self.allocator.free(models);
        }
        
        var candidates = std.ArrayList(ModelCandidate){};
        errdefer candidates.deinit();
        
        for (models) |model_id| {
            const candidate = try self.scoreModel(model_id, estimated_tokens);
            try candidates.append(candidate);
        }
        
        return candidates.toOwnedSlice();
    }
    
    fn scoreModel(self: *RequestRouter, model_id: []const u8, estimated_tokens: u64) !ModelCandidate {
        _ = estimated_tokens;
        
        var candidate = ModelCandidate{
            .model_id = try self.allocator.dupe(u8, model_id),
            .score = 0.0,
            .load = 0.0,
            .cache_hit_rate = 0.0,
            .quota_available = 1.0,
            .is_healthy = true,
        };
        
        // Score based on cache performance
        if (self.cache_manager) |cm| {
            const stats = try cm.getModelStats(model_id);
            const total_requests = stats.cache_hits + stats.cache_misses;
            if (total_requests > 0) {
                candidate.cache_hit_rate = @as(f32, @floatFromInt(stats.cache_hits)) /
                                          @as(f32, @floatFromInt(total_requests));
            }
        }
        
        // Score based on quota availability
        if (self.quota_manager) |qm| {
            if (qm.getModelReport(model_id)) |report| {
                // Use inverse of utilization as availability
                candidate.quota_available = (100.0 - report.hourly_quota_used) / 100.0;
                candidate.load = report.ram_utilization / 100.0;
            }
        }
        
        // Calculate composite score
        candidate.score = (candidate.cache_hit_rate * 0.4) +
                         ((1.0 - candidate.load) * 0.4) +
                         (candidate.quota_available * 0.2);
        
        return candidate;
    }
    
    fn checkAffinity(self: *RequestRouter, session_id: []const u8, estimated_tokens: u64) !?[]const u8 {
        self.affinity_mutex.lock();
        defer self.affinity_mutex.unlock();
        
        if (self.affinity_map.get(session_id)) |entry| {
            const now = std.time.milliTimestamp();
            const age_sec = @divFloor(now - entry.last_access, 1000);
            
            // Check if affinity is still valid
            if (age_sec < self.config.affinity_timeout_sec) {
                // Verify model is still available
                if (try self.isModelAvailable(entry.model_id, estimated_tokens)) {
                    return entry.model_id;
                }
            }
            
            // Expired or unavailable - remove
            self.allocator.free(entry.model_id);
            _ = self.affinity_map.remove(session_id);
        }
        
        return null;
    }
    
    fn setAffinity(self: *RequestRouter, session_id: []const u8, model_id: []const u8) !void {
        self.affinity_mutex.lock();
        defer self.affinity_mutex.unlock();
        
        const now = std.time.milliTimestamp();
        
        if (self.affinity_map.getPtr(session_id)) |entry| {
            entry.last_access = now;
            entry.request_count += 1;
        } else {
            const sid_key = try self.allocator.dupe(u8, session_id);
            const mid_value = try self.allocator.dupe(u8, model_id);
            
            try self.affinity_map.put(sid_key, .{
                .model_id = mid_value,
                .last_access = now,
                .request_count = 1,
            });
        }
    }
    
    fn routeABTest(self: *RequestRouter, estimated_tokens: u64) !?RoutingDecision {
        const model_a = self.config.ab_test_model_a orelse return null;
        const model_b = self.config.ab_test_model_b orelse return null;
        
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = prng.random();
        const use_a = random.float(f32) < self.config.ab_test_split;
        
        const selected = if (use_a) model_a else model_b;
        
        if (try self.isModelAvailable(selected, estimated_tokens)) {
            return RoutingDecision{
                .model_id = selected,
                .reason = if (use_a) "A/B test: variant A" else "A/B test: variant B",
                .total_score = 1.0,
            };
        }
        
        return null;
    }
    
    fn recordDecision(
        self: *RequestRouter,
        model_id: []const u8,
        reason: []const u8,
        score: f32,
        start_time: i64,
    ) RoutingDecision {
        const end_time = std.time.microTimestamp();
        const duration_us = @as(u64, @intCast(end_time - start_time));
        
        self.stats_mutex.lock();
        self.stats.successful_routes += 1;
        self.stats.total_routing_time_us += duration_us;
        self.stats.avg_routing_time_us = @as(f32, @floatFromInt(self.stats.total_routing_time_us)) /
                                         @as(f32, @floatFromInt(self.stats.total_routes));
        self.stats_mutex.unlock();
        
        return RoutingDecision{
            .model_id = model_id,
            .reason = reason,
            .total_score = score,
        };
    }
    
    fn recordDecisionWithTime(self: *RequestRouter, decision: RoutingDecision, start_time: i64) RoutingDecision {
        const end_time = std.time.microTimestamp();
        const duration_us = @as(u64, @intCast(end_time - start_time));
        
        self.stats_mutex.lock();
        self.stats.successful_routes += 1;
        self.stats.total_routing_time_us += duration_us;
        self.stats.avg_routing_time_us = @as(f32, @floatFromInt(self.stats.total_routing_time_us)) /
                                         @as(f32, @floatFromInt(self.stats.total_routes));
        self.stats_mutex.unlock();
        
        return decision;
    }
    
    /// Get routing statistics
    pub fn getStats(self: *RequestRouter) RoutingStats {
        self.stats_mutex.lock();
        defer self.stats_mutex.unlock();
        return self.stats;
    }
    
    /// Print routing status
    pub fn printStatus(self: *RequestRouter) void {
        self.stats_mutex.lock();
        defer self.stats_mutex.unlock();
        
        std.debug.print("\n📊 Request Router Status\n", .{});
        std.debug.print("=" ** 60 ++ "\n", .{});
        std.debug.print("Strategy: {s}\n", .{@tagName(self.config.strategy)});
        std.debug.print("Total routes: {d}\n", .{self.stats.total_routes});
        std.debug.print("Successful: {d}\n", .{self.stats.successful_routes});
        std.debug.print("Failed: {d}\n", .{self.stats.failed_routes});
        std.debug.print("Fallback: {d}\n", .{self.stats.fallback_routes});
        std.debug.print("Avg routing time: {d:.2}μs\n", .{self.stats.avg_routing_time_us});
        
        std.debug.print("\nPer-Strategy Counts:\n", .{});
        std.debug.print("  Round-robin: {d}\n", .{self.stats.round_robin_count});
        std.debug.print("  Least-loaded: {d}\n", .{self.stats.least_loaded_count});
        std.debug.print("  Cache-aware: {d}\n", .{self.stats.cache_aware_count});
        std.debug.print("  Quota-aware: {d}\n", .{self.stats.quota_aware_count});
        std.debug.print("  Affinity: {d}\n", .{self.stats.affinity_count});
        
        std.debug.print("\nAffinity entries: {d}\n", .{self.affinity_map.count()});
        std.debug.print("=" ** 60 ++ "\n", .{});
    }
};
