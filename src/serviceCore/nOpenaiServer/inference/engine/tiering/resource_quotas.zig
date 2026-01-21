//! Resource Quotas & Limits - Day 13
//! Enforces per-model resource limits, quotas, and rate limiting
//!
//! Features:
//! - Hard RAM/SSD limits per model
//! - Request rate limiting (requests/sec, tokens/sec)
//! - Token quota enforcement (hourly/daily limits)
//! - Resource isolation between models
//! - Quota violation handling
//! - Comprehensive monitoring and alerting

const std = @import("std");
const log = @import("structured_logging.zig");

// ============================================================================
// Resource Quota Configuration
// ============================================================================

pub const ResourceQuotaConfig = struct {
    /// Model identifier
    model_id: []const u8,
    
    /// Memory limits
    max_ram_mb: u64 = 2048,              // Hard RAM limit per model
    max_ssd_mb: u64 = 16384,             // Hard SSD limit per model (16GB)
    
    /// Rate limits
    max_requests_per_second: f32 = 100.0,     // Request rate limit
    max_tokens_per_second: f32 = 1000.0,      // Token throughput limit
    
    /// Quota limits (time-based)
    max_tokens_per_hour: u64 = 1_000_000,     // Hourly token quota
    max_tokens_per_day: u64 = 10_000_000,     // Daily token quota
    max_requests_per_hour: u64 = 10_000,      // Hourly request quota
    max_requests_per_day: u64 = 100_000,      // Daily request quota
    
    /// Burst allowance (short-term overages)
    burst_requests: u32 = 200,           // Allow burst of requests
    burst_tokens: u64 = 10_000,          // Allow burst of tokens
    
    /// Actions on quota violations
    on_violation: ViolationAction = .throttle,
    
    /// Grace period before hard enforcement
    grace_period_sec: u32 = 60,
};

pub const ViolationAction = enum {
    reject,          // Reject requests immediately
    throttle,        // Slow down requests (delay)
    warn,            // Log warning but allow
    queue,           // Queue requests for later
};

// ============================================================================
// Resource Usage Tracking
// ============================================================================

pub const ResourceUsage = struct {
    /// Memory usage
    current_ram_mb: u64 = 0,
    peak_ram_mb: u64 = 0,
    current_ssd_mb: u64 = 0,
    peak_ssd_mb: u64 = 0,
    
    /// Rate counters (current window)
    requests_this_second: u32 = 0,
    tokens_this_second: u64 = 0,
    
    /// Rolling quota counters
    tokens_this_hour: u64 = 0,
    tokens_this_day: u64 = 0,
    requests_this_hour: u64 = 0,
    requests_this_day: u64 = 0,
    
    /// Burst tracking
    burst_requests_used: u32 = 0,
    burst_tokens_used: u64 = 0,
    
    /// Timing
    last_reset_time: i64 = 0,            // Unix timestamp (ms)
    hour_window_start: i64 = 0,
    day_window_start: i64 = 0,
    second_window_start: i64 = 0,
    
    /// Violation tracking
    total_violations: u64 = 0,
    ram_violations: u32 = 0,
    ssd_violations: u32 = 0,
    rate_violations: u32 = 0,
    quota_violations: u32 = 0,
    
    pub fn init() ResourceUsage {
        const now = std.time.milliTimestamp();
        return ResourceUsage{
            .last_reset_time = now,
            .hour_window_start = now,
            .day_window_start = now,
            .second_window_start = now,
        };
    }
    
    pub fn updateMemory(self: *ResourceUsage, ram_mb: u64, ssd_mb: u64) void {
        self.current_ram_mb = ram_mb;
        self.current_ssd_mb = ssd_mb;
        self.peak_ram_mb = @max(self.peak_ram_mb, ram_mb);
        self.peak_ssd_mb = @max(self.peak_ssd_mb, ssd_mb);
    }
    
    pub fn recordRequest(self: *ResourceUsage, tokens: u64) void {
        const now = std.time.milliTimestamp();
        
        // Reset second window if needed
        if (now - self.second_window_start >= 1000) {
            self.requests_this_second = 0;
            self.tokens_this_second = 0;
            self.second_window_start = now;
        }
        
        // Reset hour window if needed
        if (now - self.hour_window_start >= 3_600_000) {
            self.tokens_this_hour = 0;
            self.requests_this_hour = 0;
            self.hour_window_start = now;
        }
        
        // Reset day window if needed
        if (now - self.day_window_start >= 86_400_000) {
            self.tokens_this_day = 0;
            self.requests_this_day = 0;
            self.day_window_start = now;
        }
        
        // Update counters
        self.requests_this_second += 1;
        self.tokens_this_second += tokens;
        self.requests_this_hour += 1;
        self.tokens_this_hour += tokens;
        self.requests_this_day += 1;
        self.tokens_this_day += tokens;
    }
    
    pub fn recordViolation(self: *ResourceUsage, violation_type: ViolationType) void {
        self.total_violations += 1;
        switch (violation_type) {
            .ram_limit => self.ram_violations += 1,
            .ssd_limit => self.ssd_violations += 1,
            .rate_limit => self.rate_violations += 1,
            .quota_limit => self.quota_violations += 1,
        }
    }
    
    pub fn useBurst(self: *ResourceUsage, tokens: u64) void {
        self.burst_tokens_used += tokens;
        self.burst_requests_used += 1;
    }
    
    pub fn resetBurst(self: *ResourceUsage) void {
        self.burst_tokens_used = 0;
        self.burst_requests_used = 0;
    }
};

pub const ViolationType = enum {
    ram_limit,
    ssd_limit,
    rate_limit,
    quota_limit,
};

// ============================================================================
// Quota Enforcement Result
// ============================================================================

pub const QuotaCheckResult = struct {
    allowed: bool,
    reason: ?[]const u8 = null,
    violation_type: ?ViolationType = null,
    retry_after_ms: ?u64 = null,         // Suggested retry delay
    current_usage: f32 = 0.0,            // Usage percentage
    
    pub fn allow() QuotaCheckResult {
        return .{ .allowed = true };
    }
    
    pub fn deny(reason: []const u8, vtype: ViolationType, retry_ms: ?u64) QuotaCheckResult {
        return .{
            .allowed = false,
            .reason = reason,
            .violation_type = vtype,
            .retry_after_ms = retry_ms,
        };
    }
};

// ============================================================================
// Resource Quota Manager
// ============================================================================

pub const ResourceQuotaManager = struct {
    allocator: std.mem.Allocator,
    
    /// Per-model quotas (HashMap: model_id -> quota config)
    quotas: std.StringHashMap(ResourceQuotaConfig),
    
    /// Per-model usage tracking
    usage: std.StringHashMap(ResourceUsage),
    
    /// Global settings
    enforcement_enabled: bool = true,
    grace_period_enabled: bool = true,
    
    /// Monitoring
    stats: QuotaStats,
    
    /// Thread safety
    mutex: std.Thread.Mutex,
    
    pub const QuotaStats = struct {
        total_checks: u64 = 0,
        total_violations: u64 = 0,
        rejected_requests: u64 = 0,
        throttled_requests: u64 = 0,
        queued_requests: u64 = 0,
        
        ram_violations: u64 = 0,
        ssd_violations: u64 = 0,
        rate_violations: u64 = 0,
        quota_violations: u64 = 0,
    };
    
    pub fn init(allocator: std.mem.Allocator) !*ResourceQuotaManager {
        log.info("Initializing Resource Quota Manager", .{});
        
        const self = try allocator.create(ResourceQuotaManager);
        errdefer allocator.destroy(self);
        
        self.* = ResourceQuotaManager{
            .allocator = allocator,
            .quotas = std.StringHashMap(ResourceQuotaConfig).init(allocator),
            .usage = std.StringHashMap(ResourceUsage).init(allocator),
            .stats = .{},
            .mutex = .{},
        };
        
        log.info("Resource Quota Manager initialized successfully", .{});
        return self;
    }
    
    pub fn deinit(self: *ResourceQuotaManager) void {
        // Free string keys
        var quota_iter = self.quotas.keyIterator();
        while (quota_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        
        var usage_iter = self.usage.keyIterator();
        while (usage_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        
        self.quotas.deinit();
        self.usage.deinit();
        self.allocator.destroy(self);
    }
    
    /// Set quota configuration for a model
    pub fn setQuota(self: *ResourceQuotaManager, config: ResourceQuotaConfig) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        log.info("Setting quota for model: id={s}, ram={d}MB, rate={d:.1}req/s", .{
            config.model_id, config.max_ram_mb, config.max_requests_per_second,
        });
        
        const model_id = try self.allocator.dupe(u8, config.model_id);
        errdefer self.allocator.free(model_id);
        
        // Store quota config
        try self.quotas.put(model_id, config);
        
        // Initialize usage tracking if not exists
        if (!self.usage.contains(model_id)) {
            const usage_key = try self.allocator.dupe(u8, config.model_id);
            try self.usage.put(usage_key, ResourceUsage.init());
        }
        
        log.info("Quota configured successfully for model: {s}", .{config.model_id});
    }
    
    /// Remove quota for a model
    pub fn removeQuota(self: *ResourceQuotaManager, model_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        log.info("Removing quota for model: {s}", .{model_id});
        
        if (self.quotas.fetchRemove(model_id)) |entry| {
            self.allocator.free(entry.key);
        }
        
        if (self.usage.fetchRemove(model_id)) |entry| {
            self.allocator.free(entry.key);
        }
    }
    
    /// Check if a request is allowed under current quotas
    pub fn checkQuota(
        self: *ResourceQuotaManager,
        model_id: []const u8,
        request: struct {
            estimated_tokens: u64,
            ram_needed_mb: u64 = 0,
            ssd_needed_mb: u64 = 0,
        },
    ) !QuotaCheckResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.stats.total_checks += 1;
        
        if (!self.enforcement_enabled) {
            return QuotaCheckResult.allow();
        }
        
        const quota = self.quotas.get(model_id) orelse {
            log.warn("No quota configured for model: {s}", .{model_id});
            return QuotaCheckResult.allow();
        };
        
        var usage = self.usage.getPtr(model_id) orelse {
            log.warn("No usage tracking for model: {s}", .{model_id});
            return QuotaCheckResult.allow();
        };
        
        // Check memory limits
        if (request.ram_needed_mb > 0) {
            if (usage.current_ram_mb + request.ram_needed_mb > quota.max_ram_mb) {
                usage.recordViolation(.ram_limit);
                self.stats.ram_violations += 1;
                self.stats.total_violations += 1;
                
                return self.handleViolation(
                    quota,
                    usage,
                    "RAM limit exceeded",
                    .ram_limit,
                    null,
                );
            }
        }
        
        if (request.ssd_needed_mb > 0) {
            if (usage.current_ssd_mb + request.ssd_needed_mb > quota.max_ssd_mb) {
                usage.recordViolation(.ssd_limit);
                self.stats.ssd_violations += 1;
                self.stats.total_violations += 1;
                
                return self.handleViolation(
                    quota,
                    usage,
                    "SSD limit exceeded",
                    .ssd_limit,
                    null,
                );
            }
        }
        
        // Check rate limits
        const now = std.time.milliTimestamp();
        if (now - usage.second_window_start < 1000) {
            const req_rate = @as(f32, @floatFromInt(usage.requests_this_second + 1));
            if (req_rate > quota.max_requests_per_second) {
                // Check burst allowance
                if (usage.burst_requests_used < quota.burst_requests) {
                    usage.useBurst(request.estimated_tokens);
                    log.warn("Using burst allowance: model={s}, burst={d}/{d}", .{
                        model_id, usage.burst_requests_used, quota.burst_requests,
                    });
                } else {
                    usage.recordViolation(.rate_limit);
                    self.stats.rate_violations += 1;
                    self.stats.total_violations += 1;
                    
                    const retry_ms = 1000 - @as(u64, @intCast(now - usage.second_window_start));
                    return self.handleViolation(
                        quota,
                        usage,
                        "Request rate limit exceeded",
                        .rate_limit,
                        retry_ms,
                    );
                }
            }
            
            const token_rate = @as(f32, @floatFromInt(usage.tokens_this_second + request.estimated_tokens));
            if (token_rate > quota.max_tokens_per_second) {
                usage.recordViolation(.rate_limit);
                self.stats.rate_violations += 1;
                self.stats.total_violations += 1;
                
                const retry_ms = 1000 - @as(u64, @intCast(now - usage.second_window_start));
                return self.handleViolation(
                    quota,
                    usage,
                    "Token rate limit exceeded",
                    .rate_limit,
                    retry_ms,
                );
            }
        }
        
        // Check quota limits (hourly)
        if (usage.tokens_this_hour + request.estimated_tokens > quota.max_tokens_per_hour) {
            usage.recordViolation(.quota_limit);
            self.stats.quota_violations += 1;
            self.stats.total_violations += 1;
            
            const time_until_reset = 3_600_000 - @as(u64, @intCast(now - usage.hour_window_start));
            return self.handleViolation(
                quota,
                usage,
                "Hourly token quota exceeded",
                .quota_limit,
                time_until_reset,
            );
        }
        
        if (usage.requests_this_hour + 1 > quota.max_requests_per_hour) {
            usage.recordViolation(.quota_limit);
            self.stats.quota_violations += 1;
            self.stats.total_violations += 1;
            
            const time_until_reset = 3_600_000 - @as(u64, @intCast(now - usage.hour_window_start));
            return self.handleViolation(
                quota,
                usage,
                "Hourly request quota exceeded",
                .quota_limit,
                time_until_reset,
            );
        }
        
        // Check quota limits (daily)
        if (usage.tokens_this_day + request.estimated_tokens > quota.max_tokens_per_day) {
            usage.recordViolation(.quota_limit);
            self.stats.quota_violations += 1;
            self.stats.total_violations += 1;
            
            const time_until_reset = 86_400_000 - @as(u64, @intCast(now - usage.day_window_start));
            return self.handleViolation(
                quota,
                usage,
                "Daily token quota exceeded",
                .quota_limit,
                time_until_reset,
            );
        }
        
        if (usage.requests_this_day + 1 > quota.max_requests_per_day) {
            usage.recordViolation(.quota_limit);
            self.stats.quota_violations += 1;
            self.stats.total_violations += 1;
            
            const time_until_reset = 86_400_000 - @as(u64, @intCast(now - usage.day_window_start));
            return self.handleViolation(
                quota,
                usage,
                "Daily request quota exceeded",
                .quota_limit,
                time_until_reset,
            );
        }
        
        // All checks passed - record the request
        usage.recordRequest(request.estimated_tokens);
        
        return QuotaCheckResult.allow();
    }
    
    fn handleViolation(
        self: *ResourceQuotaManager,
        quota: ResourceQuotaConfig,
        usage: *ResourceUsage,
        reason: []const u8,
        vtype: ViolationType,
        retry_ms: ?u64,
    ) QuotaCheckResult {
        switch (quota.on_violation) {
            .reject => {
                self.stats.rejected_requests += 1;
                log.warn("Request rejected: {s} for model: {s}", .{ reason, quota.model_id });
                return QuotaCheckResult.deny(reason, vtype, retry_ms);
            },
            .throttle => {
                self.stats.throttled_requests += 1;
                log.warn("Request throttled: {s} for model: {s}", .{ reason, quota.model_id });
                return QuotaCheckResult.deny(reason, vtype, retry_ms);
            },
            .warn => {
                log.warn("Quota violation (warning only): {s} for model: {s}", .{ reason, quota.model_id });
                return QuotaCheckResult.allow();
            },
            .queue => {
                self.stats.queued_requests += 1;
                log.info("Request queued: {s} for model: {s}", .{ reason, quota.model_id });
                return QuotaCheckResult.deny(reason, vtype, retry_ms);
            },
        }
    }
    
    /// Update memory usage for a model
    pub fn updateMemoryUsage(
        self: *ResourceQuotaManager,
        model_id: []const u8,
        ram_mb: u64,
        ssd_mb: u64,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.usage.getPtr(model_id)) |usage| {
            usage.updateMemory(ram_mb, ssd_mb);
        }
    }
    
    /// Get current usage for a model
    pub fn getUsage(self: *ResourceQuotaManager, model_id: []const u8) ?ResourceUsage {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return self.usage.get(model_id);
    }
    
    /// Get quota configuration for a model
    pub fn getQuota(self: *ResourceQuotaManager, model_id: []const u8) ?ResourceQuotaConfig {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return self.quotas.get(model_id);
    }
    
    /// Reset burst allowance for a model
    pub fn resetBurst(self: *ResourceQuotaManager, model_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.usage.getPtr(model_id)) |usage| {
            usage.resetBurst();
            log.info("Burst allowance reset for model: {s}", .{model_id});
        }
    }
    
    /// Get global quota statistics
    pub fn getStats(self: *ResourceQuotaManager) QuotaStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }
    
    /// Get detailed model report
    pub fn getModelReport(
        self: *ResourceQuotaManager,
        model_id: []const u8,
    ) ?struct {
        model_id: []const u8,
        quota: ResourceQuotaConfig,
        usage: ResourceUsage,
        ram_utilization: f32,
        ssd_utilization: f32,
        requests_per_second: f32,
        tokens_per_second: f32,
        hourly_quota_used: f32,
        daily_quota_used: f32,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const quota = self.quotas.get(model_id) orelse return null;
        const usage = self.usage.get(model_id) orelse return null;
        
        const ram_util = @as(f32, @floatFromInt(usage.current_ram_mb)) / 
                         @as(f32, @floatFromInt(quota.max_ram_mb)) * 100.0;
        const ssd_util = @as(f32, @floatFromInt(usage.current_ssd_mb)) / 
                         @as(f32, @floatFromInt(quota.max_ssd_mb)) * 100.0;
        
        const hourly_quota_pct = @as(f32, @floatFromInt(usage.tokens_this_hour)) / 
                                 @as(f32, @floatFromInt(quota.max_tokens_per_hour)) * 100.0;
        const daily_quota_pct = @as(f32, @floatFromInt(usage.tokens_this_day)) / 
                                @as(f32, @floatFromInt(quota.max_tokens_per_day)) * 100.0;
        
        return .{
            .model_id = model_id,
            .quota = quota,
            .usage = usage,
            .ram_utilization = ram_util,
            .ssd_utilization = ssd_util,
            .requests_per_second = @as(f32, @floatFromInt(usage.requests_this_second)),
            .tokens_per_second = @as(f32, @floatFromInt(usage.tokens_this_second)),
            .hourly_quota_used = hourly_quota_pct,
            .daily_quota_used = daily_quota_pct,
        };
    }
    
    /// Print comprehensive status report
    pub fn printStatus(self: *ResourceQuotaManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        std.debug.print("\n📊 Resource Quota Manager Status\n", .{});
        std.debug.print("=" ** 70 ++ "\n", .{});
        std.debug.print("Total checks: {d}\n", .{self.stats.total_checks});
        std.debug.print("Total violations: {d}\n", .{self.stats.total_violations});
        std.debug.print("  RAM violations: {d}\n", .{self.stats.ram_violations});
        std.debug.print("  SSD violations: {d}\n", .{self.stats.ssd_violations});
        std.debug.print("  Rate violations: {d}\n", .{self.stats.rate_violations});
        std.debug.print("  Quota violations: {d}\n", .{self.stats.quota_violations});
        std.debug.print("\nRequests:\n", .{});
        std.debug.print("  Rejected: {d}\n", .{self.stats.rejected_requests});
        std.debug.print("  Throttled: {d}\n", .{self.stats.throttled_requests});
        std.debug.print("  Queued: {d}\n", .{self.stats.queued_requests});
        
        std.debug.print("\nPer-Model Quotas:\n", .{});
        std.debug.print("-" ** 70 ++ "\n", .{});
        
        var iter = self.quotas.iterator();
        while (iter.next()) |entry| {
            const model_id = entry.key_ptr.*;
            const quota = entry.value_ptr.*;
            const usage = self.usage.get(model_id) orelse continue;
            
            const ram_util = @as(f32, @floatFromInt(usage.current_ram_mb)) / 
                            @as(f32, @floatFromInt(quota.max_ram_mb)) * 100.0;
            const ssd_util = @as(f32, @floatFromInt(usage.current_ssd_mb)) / 
                            @as(f32, @floatFromInt(quota.max_ssd_mb)) * 100.0;
            
            std.debug.print("\nModel: {s}\n", .{model_id});
            std.debug.print("  Memory:\n", .{});
            std.debug.print("    RAM: {d}/{d} MB ({d:.1}%)\n", .{
                usage.current_ram_mb, quota.max_ram_mb, ram_util,
            });
            std.debug.print("    SSD: {d}/{d} MB ({d:.1}%)\n", .{
                usage.current_ssd_mb, quota.max_ssd_mb, ssd_util,
            });
            std.debug.print("  Rate:\n", .{});
            std.debug.print("    Requests/sec: {d}/{d:.1}\n", .{
                usage.requests_this_second, quota.max_requests_per_second,
            });
            std.debug.print("    Tokens/sec: {d}/{d:.1}\n", .{
                usage.tokens_this_second, quota.max_tokens_per_second,
            });
            std.debug.print("  Quotas:\n", .{});
            std.debug.print("    Hourly: {d}/{d} tokens, {d}/{d} requests\n", .{
                usage.tokens_this_hour, quota.max_tokens_per_hour,
                usage.requests_this_hour, quota.max_requests_per_hour,
            });
            std.debug.print("    Daily: {d}/{d} tokens, {d}/{d} requests\n", .{
                usage.tokens_this_day, quota.max_tokens_per_day,
                usage.requests_this_day, quota.max_requests_per_day,
            });
            std.debug.print("  Violations: {d} total ({d} RAM, {d} SSD, {d} rate, {d} quota)\n", .{
                usage.total_violations, usage.ram_violations, usage.ssd_violations,
                usage.rate_violations, usage.quota_violations,
            });
        }
        
        std.debug.print("=" ** 70 ++ "\n", .{});
    }
};
