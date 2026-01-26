// ============================================================================
// Alert System - Day 28 Implementation
// ============================================================================
// Purpose: Threshold monitoring and alerting for model router performance
// Week: Week 6 (Days 26-30) - Performance Monitoring & Feedback Loop
// Phase: Month 2 - Model Router & Orchestration
// ============================================================================

const std = @import("std");
const performance_metrics = @import("performance_metrics.zig");

const PerformanceTracker = performance_metrics.PerformanceTracker;
const LatencyMetric = performance_metrics.LatencyMetric;
const SuccessRateMetric = performance_metrics.SuccessRateMetric;

// ============================================================================
// ALERT TYPES
// ============================================================================

pub const AlertSeverity = enum {
    info,
    warning,
    @"error",
    critical,
    
    pub fn toString(self: AlertSeverity) []const u8 {
        return switch (self) {
            .info => "INFO",
            .warning => "WARNING",
            .@"error" => "ERROR",
            .critical => "CRITICAL",
        };
    }
};

pub const AlertType = enum {
    high_latency,
    low_success_rate,
    model_failure,
    threshold_breach,
    no_models_available,
    rapid_failures,
    
    pub fn toString(self: AlertType) []const u8 {
        return switch (self) {
            .high_latency => "HIGH_LATENCY",
            .low_success_rate => "LOW_SUCCESS_RATE",
            .model_failure => "MODEL_FAILURE",
            .threshold_breach => "THRESHOLD_BREACH",
            .no_models_available => "NO_MODELS_AVAILABLE",
            .rapid_failures => "RAPID_FAILURES",
        };
    }
};

pub const Alert = struct {
    alert_id: []const u8,
    timestamp: i64,
    severity: AlertSeverity,
    alert_type: AlertType,
    message: []const u8,
    model_id: ?[]const u8,
    metric_value: ?f32,
    threshold_value: ?f32,
    
    pub fn init(
        allocator: std.mem.Allocator,
        alert_id: []const u8,
        severity: AlertSeverity,
        alert_type: AlertType,
        message: []const u8,
    ) !Alert {
        return .{
            .alert_id = try allocator.dupe(u8, alert_id),
            .timestamp = std.time.milliTimestamp(),
            .severity = severity,
            .alert_type = alert_type,
            .message = try allocator.dupe(u8, message),
            .model_id = null,
            .metric_value = null,
            .threshold_value = null,
        };
    }
    
    pub fn deinit(self: *Alert, allocator: std.mem.Allocator) void {
        allocator.free(self.alert_id);
        allocator.free(self.message);
        if (self.model_id) |model_id| {
            allocator.free(model_id);
        }
    }
};

// ============================================================================
// ALERT THRESHOLDS
// ============================================================================

pub const AlertThresholds = struct {
    // Latency thresholds (milliseconds)
    latency_warning_p95: f32 = 500.0,
    latency_error_p95: f32 = 1000.0,
    latency_critical_p99: f32 = 2000.0,
    
    // Success rate thresholds (0.0 - 1.0)
    success_rate_warning: f32 = 0.95,  // 95%
    success_rate_error: f32 = 0.90,    // 90%
    success_rate_critical: f32 = 0.80,  // 80%
    
    // Failure thresholds
    consecutive_failures_warning: u32 = 3,
    consecutive_failures_critical: u32 = 10,
    
    // Time windows (milliseconds)
    rapid_failure_window_ms: i64 = 60_000,  // 1 minute
    rapid_failure_count: u32 = 5,
    
    pub fn validate(self: *const AlertThresholds) bool {
        // Ensure warning < error < critical
        if (self.latency_warning_p95 >= self.latency_error_p95) return false;
        if (self.latency_error_p95 >= self.latency_critical_p99) return false;
        
        if (self.success_rate_warning <= self.success_rate_error) return false;
        if (self.success_rate_error <= self.success_rate_critical) return false;
        
        return true;
    }
};

// ============================================================================
// ALERT MANAGER
// ============================================================================

pub const AlertManager = struct {
    allocator: std.mem.Allocator,
    performance_tracker: *PerformanceTracker,
    thresholds: AlertThresholds,
    
    // Alert history
    active_alerts: std.ArrayList(Alert),
    alert_history: std.ArrayList(Alert),
    max_history: usize,
    
    // Alert suppression
    last_alert_times: std.StringHashMap(i64),
    alert_cooldown_ms: i64,
    
    pub fn init(
        allocator: std.mem.Allocator,
        performance_tracker: *PerformanceTracker,
        thresholds: AlertThresholds,
        max_history: usize,
    ) AlertManager {
        return .{
            .allocator = allocator,
            .performance_tracker = performance_tracker,
            .thresholds = thresholds,
            .active_alerts = std.ArrayList(Alert){},
            .alert_history = std.ArrayList(Alert){},
            .max_history = max_history,
            .last_alert_times = std.StringHashMap(i64).init(allocator),
            .alert_cooldown_ms = 300_000, // 5 minutes default
        };
    }
    
    pub fn deinit(self: *AlertManager) void {
        for (self.active_alerts.items) |*alert| {
            alert.deinit(self.allocator);
        }
        self.active_alerts.deinit();
        
        for (self.alert_history.items) |*alert| {
            alert.deinit(self.allocator);
        }
        self.alert_history.deinit();
        
        self.last_alert_times.deinit();
    }
    
    /// Check all metrics and generate alerts if needed
    pub fn checkMetrics(self: *AlertManager) !std.ArrayList(Alert) {
        var new_alerts = std.ArrayList(Alert){};
        
        // Check latency metrics
        const latency_metrics = try self.performance_tracker.getLatencyMetrics(self.allocator);
        try self.checkLatencyThresholds(latency_metrics, &new_alerts);
        
        // Check success rate
        const success_metrics = self.performance_tracker.getSuccessRate();
        try self.checkSuccessRateThresholds(success_metrics, &new_alerts);
        
        // Check per-model metrics
        try self.checkModelMetrics(&new_alerts);
        
        // Add new alerts to active and history
        for (new_alerts.items) |alert| {
            try self.active_alerts.append(alert);
            try self.addToHistory(alert);
        }
        
        return new_alerts;
    }
    
    /// Check latency against thresholds
    fn checkLatencyThresholds(
        self: *AlertManager,
        metrics: LatencyMetric,
        alerts: *std.ArrayList(Alert),
    ) !void {
        // Check P95 latency
        if (metrics.p95 >= self.thresholds.latency_critical_p99) {
            const alert_key = "latency_p95_critical";
            if (try self.shouldAlert(alert_key)) {
                const msg = try std.fmt.allocPrint(
                    self.allocator,
                    "CRITICAL: P95 latency {d:.1}ms exceeds critical threshold {d:.1}ms",
                    .{ metrics.p95, self.thresholds.latency_critical_p99 }
                );
                
                var alert = try Alert.init(
                    self.allocator,
                    try self.generateAlertId(),
                    .critical,
                    .high_latency,
                    msg,
                );
                alert.metric_value = metrics.p95;
                alert.threshold_value = self.thresholds.latency_critical_p99;
                
                try alerts.append(alert);
                try self.recordAlert(alert_key);
            }
        } else if (metrics.p95 >= self.thresholds.latency_error_p95) {
            const alert_key = "latency_p95_error";
            if (try self.shouldAlert(alert_key)) {
                const msg = try std.fmt.allocPrint(
                    self.allocator,
                    "ERROR: P95 latency {d:.1}ms exceeds error threshold {d:.1}ms",
                    .{ metrics.p95, self.thresholds.latency_error_p95 }
                );
                
                var alert = try Alert.init(
                    self.allocator,
                    try self.generateAlertId(),
                    .@"error",
                    .high_latency,
                    msg,
                );
                alert.metric_value = metrics.p95;
                alert.threshold_value = self.thresholds.latency_error_p95;
                
                try alerts.append(alert);
                try self.recordAlert(alert_key);
            }
        } else if (metrics.p95 >= self.thresholds.latency_warning_p95) {
            const alert_key = "latency_p95_warning";
            if (try self.shouldAlert(alert_key)) {
                const msg = try std.fmt.allocPrint(
                    self.allocator,
                    "WARNING: P95 latency {d:.1}ms exceeds warning threshold {d:.1}ms",
                    .{ metrics.p95, self.thresholds.latency_warning_p95 }
                );
                
                var alert = try Alert.init(
                    self.allocator,
                    try self.generateAlertId(),
                    .warning,
                    .high_latency,
                    msg,
                );
                alert.metric_value = metrics.p95;
                alert.threshold_value = self.thresholds.latency_warning_p95;
                
                try alerts.append(alert);
                try self.recordAlert(alert_key);
            }
        }
    }
    
    /// Check success rate against thresholds
    fn checkSuccessRateThresholds(
        self: *AlertManager,
        metrics: SuccessRateMetric,
        alerts: *std.ArrayList(Alert),
    ) !void {
        if (metrics.total_requests == 0) return;
        
        const rate = metrics.success_rate;
        
        if (rate <= self.thresholds.success_rate_critical) {
            const alert_key = "success_rate_critical";
            if (try self.shouldAlert(alert_key)) {
                const msg = try std.fmt.allocPrint(
                    self.allocator,
                    "CRITICAL: Success rate {d:.1}% below critical threshold {d:.1}%",
                    .{ rate * 100.0, self.thresholds.success_rate_critical * 100.0 }
                );
                
                var alert = try Alert.init(
                    self.allocator,
                    try self.generateAlertId(),
                    .critical,
                    .low_success_rate,
                    msg,
                );
                alert.metric_value = rate;
                alert.threshold_value = self.thresholds.success_rate_critical;
                
                try alerts.append(alert);
                try self.recordAlert(alert_key);
            }
        } else if (rate <= self.thresholds.success_rate_error) {
            const alert_key = "success_rate_error";
            if (try self.shouldAlert(alert_key)) {
                const msg = try std.fmt.allocPrint(
                    self.allocator,
                    "ERROR: Success rate {d:.1}% below error threshold {d:.1}%",
                    .{ rate * 100.0, self.thresholds.success_rate_error * 100.0 }
                );
                
                var alert = try Alert.init(
                    self.allocator,
                    try self.generateAlertId(),
                    .@"error",
                    .low_success_rate,
                    msg,
                );
                alert.metric_value = rate;
                alert.threshold_value = self.thresholds.success_rate_error;
                
                try alerts.append(alert);
                try self.recordAlert(alert_key);
            }
        } else if (rate <= self.thresholds.success_rate_warning) {
            const alert_key = "success_rate_warning";
            if (try self.shouldAlert(alert_key)) {
                const msg = try std.fmt.allocPrint(
                    self.allocator,
                    "WARNING: Success rate {d:.1}% below warning threshold {d:.1}%",
                    .{ rate * 100.0, self.thresholds.success_rate_warning * 100.0 }
                );
                
                var alert = try Alert.init(
                    self.allocator,
                    try self.generateAlertId(),
                    .warning,
                    .low_success_rate,
                    msg,
                );
                alert.metric_value = rate;
                alert.threshold_value = self.thresholds.success_rate_warning;
                
                try alerts.append(alert);
                try self.recordAlert(alert_key);
            }
        }
    }
    
    /// Check per-model metrics
    fn checkModelMetrics(
        self: *AlertManager,
        alerts: *std.ArrayList(Alert),
    ) !void {
        var iter = self.performance_tracker.model_metrics.iterator();
        
        while (iter.next()) |entry| {
            const model_id = entry.key_ptr.*;
            const metrics = entry.value_ptr.*;
            
            // Check model success rate
            if (metrics.total_requests >= 10) {
                const rate = metrics.getSuccessRate();
                
                if (rate < self.thresholds.success_rate_error) {
                    const alert_key = try std.fmt.allocPrint(
                        self.allocator,
                        "model_{s}_low_success",
                        .{model_id}
                    );
                    defer self.allocator.free(alert_key);
                    
                    if (try self.shouldAlert(alert_key)) {
                        const msg = try std.fmt.allocPrint(
                            self.allocator,
                            "Model {s} has low success rate: {d:.1}%",
                            .{ model_id, rate * 100.0 }
                        );
                        
                        var alert = try Alert.init(
                            self.allocator,
                            try self.generateAlertId(),
                            .@"error",
                            .model_failure,
                            msg,
                        );
                        alert.model_id = try self.allocator.dupe(u8, model_id);
                        alert.metric_value = rate;
                        
                        try alerts.append(alert);
                        try self.recordAlert(alert_key);
                    }
                }
            }
        }
    }
    
    /// Check if alert should be sent (respects cooldown)
    fn shouldAlert(self: *AlertManager, alert_key: []const u8) !bool {
        const now = std.time.milliTimestamp();
        
        if (self.last_alert_times.get(alert_key)) |last_time| {
            if (now - last_time < self.alert_cooldown_ms) {
                return false; // Still in cooldown
            }
        }
        
        return true;
    }
    
    /// Record alert time for cooldown
    fn recordAlert(self: *AlertManager, alert_key: []const u8) !void {
        const now = std.time.milliTimestamp();
        const key_copy = try self.allocator.dupe(u8, alert_key);
        try self.last_alert_times.put(key_copy, now);
    }
    
    /// Add alert to history (ring buffer)
    fn addToHistory(self: *AlertManager, alert: Alert) !void {
        if (self.alert_history.items.len >= self.max_history) {
            var old = self.alert_history.orderedRemove(0);
            old.deinit(self.allocator);
        }
        try self.alert_history.append(alert);
    }
    
    /// Clear resolved alerts
    pub fn clearResolvedAlerts(self: *AlertManager) void {
        const i: usize = 0;
        while (i < self.active_alerts.items.len) {
            // For now, clear all active alerts
            // In production, check if condition still exists
            _ = self.active_alerts.swapRemove(i);
        }
    }
    
    /// Generate unique alert ID
    fn generateAlertId(self: *AlertManager) ![]const u8 {
        const timestamp = std.time.milliTimestamp();
        const random = std.crypto.random.int(u32);
        return try std.fmt.allocPrint(
            self.allocator,
            "alert_{d}_{x}",
            .{ timestamp, random }
        );
    }
    
    /// Get active alerts
    pub fn getActiveAlerts(self: *const AlertManager) []const Alert {
        return self.active_alerts.items;
    }
    
    /// Get alert history
    pub fn getAlertHistory(self: *const AlertManager) []const Alert {
        return self.alert_history.items;
    }
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "AlertThresholds: validation" {
    const valid = AlertThresholds{};
    try std.testing.expect(valid.validate());
    
    const invalid = AlertThresholds{
        .latency_warning_p95 = 1000.0,
        .latency_error_p95 = 500.0,  // Error < Warning (invalid)
    };
    try std.testing.expect(!invalid.validate());
}

test "Alert: creation and cleanup" {
    const allocator = std.testing.allocator;
    
    var alert = try Alert.init(
        allocator,
        "test-alert-1",
        .warning,
        .high_latency,
        "Test alert message"
    );
    defer alert.deinit(allocator);
    
    try std.testing.expectEqual(AlertSeverity.warning, alert.severity);
    try std.testing.expectEqual(AlertType.high_latency, alert.alert_type);
}

test "AlertManager: latency threshold checking" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    // Add high latency data
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "decision-{d}", .{i});
        defer allocator.free(id);
        
        var decision = try performance_metrics.RoutingDecision.init(
            allocator, id, "agent-1", "model-1", 85.0
        );
        decision.latency_ms = 600.0; // Above warning threshold
        decision.success = true;
        try tracker.recordDecision(decision);
    }
    
    const thresholds = AlertThresholds{};
    var manager = AlertManager.init(allocator, &tracker, thresholds, 100);
    defer manager.deinit();
    
    var alerts = try manager.checkMetrics();
    defer alerts.deinit();
    
    // Should generate latency warning
    try std.testing.expect(alerts.items.len > 0);
}

test "AlertManager: success rate threshold checking" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    // Add low success rate data (50%)
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "decision-{d}", .{i});
        defer allocator.free(id);
        
        var decision = try performance_metrics.RoutingDecision.init(
            allocator, id, "agent-1", "model-1", 85.0
        );
        decision.latency_ms = 150.0;
        decision.success = (i < 10); // 50% success rate
        try tracker.recordDecision(decision);
    }
    
    const thresholds = AlertThresholds{};
    var manager = AlertManager.init(allocator, &tracker, thresholds, 100);
    defer manager.deinit();
    
    var alerts = try manager.checkMetrics();
    defer alerts.deinit();
    
    // Should generate success rate alert
    try std.testing.expect(alerts.items.len > 0);
    
    // Check severity (should be critical for 50%)
    var has_critical = false;
    for (alerts.items) |alert| {
        if (alert.severity == .critical) {
            has_critical = true;
        }
    }
    try std.testing.expect(has_critical);
}

test "AlertManager: cooldown mechanism" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    const thresholds = AlertThresholds{};
    var manager = AlertManager.init(allocator, &tracker, thresholds, 100);
    defer manager.deinit();
    
    manager.alert_cooldown_ms = 1000; // 1 second
    
    // First check should alert
    const should_alert_1 = try manager.shouldAlert("test_key");
    try std.testing.expect(should_alert_1);
    
    try manager.recordAlert("test_key");
    
    // Immediate second check should not alert (cooldown)
    const should_alert_2 = try manager.shouldAlert("test_key");
    try std.testing.expect(!should_alert_2);
}
