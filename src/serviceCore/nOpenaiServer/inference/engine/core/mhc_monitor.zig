// mHC Production Monitoring Framework
// Comprehensive monitoring infrastructure for mHC (manifold Hyperbolic Constraints)
//
// Core Components:
// - MetricsBuffer: Circular buffer for time-series metrics
// - GeometricSpeculationMonitor: Tracks geometric speculation performance
// - AlertLevel/AlertConfig: Alert threshold system
// - Alert Integration: PagerDuty and Slack payload formatting
// - Grafana Dashboard: Dashboard configuration generation
//
// Reference: docs/DAY_64_MONITORING_REPORT.md

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Constants
// ============================================================================

/// Default buffer size for time-series metrics
pub const DEFAULT_BUFFER_SIZE = 1000;

/// Metric name constants for consistency
pub const MetricName = struct {
    pub const ACCEPTANCE_RATE = "mhc_acceptance_rate";
    pub const CURVATURE = "mhc_curvature";
    pub const ENERGY = "mhc_energy";
    pub const LATENCY = "mhc_latency_ms";
    pub const STABILITY = "mhc_stability";
    pub const SINKHORN_ITERATIONS = "mhc_sinkhorn_iterations";
    pub const MANIFOLD_PROJECTION = "mhc_manifold_projection_ms";
    pub const GEOMETRIC_TYPE = "mhc_geometric_type";
};

// ============================================================================
// MetricsBuffer - Circular Buffer for Time-Series Metrics
// ============================================================================

/// Circular buffer for storing time-series metric values
/// Uses a ring buffer to maintain a fixed window of recent values
pub fn MetricsBuffer(comptime size: usize) type {
    return struct {
        const Self = @This();

        /// Ring buffer for metric values
        values: [size]f32,
        /// Current write position (head of the buffer)
        head: usize,
        /// Number of values currently in buffer
        count: usize,
        /// Sum of all values (for efficient mean calculation)
        sum: f64,
        /// Minimum value in buffer
        cached_min: f32,
        /// Maximum value in buffer
        cached_max: f32,
        /// Whether cached min/max are valid
        cache_valid: bool,

        /// Initialize an empty metrics buffer
        pub fn init() Self {
            return .{
                .values = [_]f32{0.0} ** size,
                .head = 0,
                .count = 0,
                .sum = 0.0,
                .cached_min = std.math.floatMax(f32),
                .cached_max = std.math.floatMin(f32),
                .cache_valid = false,
            };
        }

        /// Push a new value into the buffer
        /// Overwrites oldest value if buffer is full
        pub fn push(self: *Self, value: f32) void {
            // If buffer is full, subtract the value being overwritten from sum
            if (self.count == size) {
                self.sum -= @as(f64, self.values[self.head]);
            }

            // Add new value
            self.values[self.head] = value;
            self.sum += @as(f64, value);

            // Advance head
            self.head = (self.head + 1) % size;
            if (self.count < size) {
                self.count += 1;
            }

            // Invalidate cache
            self.cache_valid = false;
        }

        /// Get the number of values in the buffer
        pub fn len(self: *const Self) usize {
            return self.count;
        }

        /// Check if buffer is empty
        pub fn isEmpty(self: *const Self) bool {
            return self.count == 0;
        }

        /// Check if buffer is full
        pub fn isFull(self: *const Self) bool {
            return self.count == size;
        }

        /// Calculate the mean of all values in the buffer
        pub fn mean(self: *const Self) f32 {
            if (self.count == 0) return 0.0;
            return @floatCast(self.sum / @as(f64, @floatFromInt(self.count)));
        }

        /// Find the minimum value in the buffer
        pub fn min(self: *Self) f32 {
            if (self.count == 0) return 0.0;

            if (self.cache_valid) {
                return self.cached_min;
            }

            self.updateCache();
            return self.cached_min;
        }

        /// Find the maximum value in the buffer
        pub fn max(self: *Self) f32 {
            if (self.count == 0) return 0.0;

            if (self.cache_valid) {
                return self.cached_max;
            }

            self.updateCache();
            return self.cached_max;
        }

        /// Update cached min/max values
        fn updateCache(self: *Self) void {
            self.cached_min = std.math.floatMax(f32);
            self.cached_max = std.math.floatMin(f32);

            for (0..self.count) |i| {
                const idx = if (self.count == size)
                    (self.head + i) % size
                else
                    i;
                const val = self.values[idx];
                self.cached_min = @min(self.cached_min, val);
                self.cached_max = @max(self.cached_max, val);
            }

            self.cache_valid = true;
        }

        /// Calculate the percentile value (p should be 0-100)
        /// Uses linear interpolation between nearest ranks
        pub fn percentile(self: *Self, p: f32, allocator: std.mem.Allocator) !f32 {
            if (self.count == 0) return 0.0;
            if (p <= 0) return self.min();
            if (p >= 100) return self.max();

            // Copy values to temporary buffer for sorting
            const temp = try allocator.alloc(f32, self.count);
            defer allocator.free(temp);

            for (0..self.count) |i| {
                const idx = if (self.count == size)
                    (self.head + i) % size
                else
                    i;
                temp[i] = self.values[idx];
            }

            // Sort the temporary buffer
            std.mem.sort(f32, temp, {}, std.sort.asc(f32));

            // Calculate the rank
            const rank = (p / 100.0) * @as(f32, @floatFromInt(self.count - 1));
            const lower_idx = @as(usize, @intFromFloat(@floor(rank)));
            const upper_idx = @min(lower_idx + 1, self.count - 1);
            const fraction = rank - @as(f32, @floatFromInt(lower_idx));

            // Linear interpolation
            return temp[lower_idx] + fraction * (temp[upper_idx] - temp[lower_idx]);
        }

        /// Calculate the standard deviation
        pub fn stddev(self: *const Self) f32 {
            if (self.count <= 1) return 0.0;

            const avg = self.mean();
            var variance_sum: f64 = 0.0;

            for (0..self.count) |i| {
                const idx = if (self.count == size)
                    (self.head + i) % size
                else
                    i;
                const diff = self.values[idx] - avg;
                variance_sum += @as(f64, diff * diff);
            }

            return @floatCast(@sqrt(variance_sum / @as(f64, @floatFromInt(self.count))));
        }

        /// Get the most recent value
        pub fn latest(self: *const Self) f32 {
            if (self.count == 0) return 0.0;
            const idx = if (self.head == 0) size - 1 else self.head - 1;
            return self.values[idx];
        }

        /// Get all values as a slice (ordered from oldest to newest)
        pub fn getValues(self: *const Self, allocator: std.mem.Allocator) ![]f32 {
            const result = try allocator.alloc(f32, self.count);
            for (0..self.count) |i| {
                const idx = if (self.count == size)
                    (self.head + i) % size
                else
                    i;
                result[i] = self.values[idx];
            }
            return result;
        }

        /// Clear all values from the buffer
        pub fn clear(self: *Self) void {
            self.head = 0;
            self.count = 0;
            self.sum = 0.0;
            self.cache_valid = false;
            self.cached_min = std.math.floatMax(f32);
            self.cached_max = std.math.floatMin(f32);
        }
    };
}

/// Default metrics buffer type with 1000 samples
pub const DefaultMetricsBuffer = MetricsBuffer(DEFAULT_BUFFER_SIZE);

// ============================================================================
// Alert Level and Configuration
// ============================================================================

/// Alert severity levels for monitoring thresholds
pub const AlertLevel = enum(u8) {
    info = 0,
    warning = 1,
    critical = 2,
    emergency = 3,

    /// Get string representation
    pub fn toString(self: AlertLevel) []const u8 {
        return switch (self) {
            .info => "INFO",
            .warning => "WARNING",
            .critical => "CRITICAL",
            .emergency => "EMERGENCY",
        };
    }

    /// Get severity color for display
    pub fn getColor(self: AlertLevel) []const u8 {
        return switch (self) {
            .info => "#36a64f", // green
            .warning => "#ffcc00", // yellow
            .critical => "#ff6600", // orange
            .emergency => "#ff0000", // red
        };
    }

    /// Get PagerDuty severity
    pub fn toPagerDutySeverity(self: AlertLevel) []const u8 {
        return switch (self) {
            .info => "info",
            .warning => "warning",
            .critical => "error",
            .emergency => "critical",
        };
    }
};

/// Configuration for alert thresholds
pub const AlertConfig = struct {
    // Acceptance rate thresholds
    acceptance_rate_warning: f32 = 0.7,
    acceptance_rate_critical: f32 = 0.5,
    acceptance_rate_emergency: f32 = 0.3,

    // Curvature thresholds (absolute value)
    curvature_warning: f32 = 2.0,
    curvature_critical: f32 = 5.0,
    curvature_emergency: f32 = 10.0,

    // Energy thresholds
    energy_warning: f32 = 100.0,
    energy_critical: f32 = 500.0,
    energy_emergency: f32 = 1000.0,

    // Latency thresholds (milliseconds)
    latency_warning_ms: f32 = 50.0,
    latency_critical_ms: f32 = 100.0,
    latency_emergency_ms: f32 = 500.0,

    // Stability thresholds
    stability_warning: f32 = 0.9,
    stability_critical: f32 = 0.8,
    stability_emergency: f32 = 0.5,

    // Alert cooldown to prevent alert storms (seconds)
    cooldown_seconds: u32 = 60,

    /// Check acceptance rate and return alert level
    pub fn checkAcceptanceRate(self: *const AlertConfig, rate: f32) AlertLevel {
        if (rate < self.acceptance_rate_emergency) return .emergency;
        if (rate < self.acceptance_rate_critical) return .critical;
        if (rate < self.acceptance_rate_warning) return .warning;
        return .info;
    }

    /// Check curvature and return alert level
    pub fn checkCurvature(self: *const AlertConfig, curvature: f32) AlertLevel {
        const abs_curv = @abs(curvature);
        if (abs_curv > self.curvature_emergency) return .emergency;
        if (abs_curv > self.curvature_critical) return .critical;
        if (abs_curv > self.curvature_warning) return .warning;
        return .info;
    }

    /// Check energy and return alert level
    pub fn checkEnergy(self: *const AlertConfig, energy: f32) AlertLevel {
        if (energy > self.energy_emergency) return .emergency;
        if (energy > self.energy_critical) return .critical;
        if (energy > self.energy_warning) return .warning;
        return .info;
    }

    /// Check latency and return alert level
    pub fn checkLatency(self: *const AlertConfig, latency_ms: f32) AlertLevel {
        if (latency_ms > self.latency_emergency_ms) return .emergency;
        if (latency_ms > self.latency_critical_ms) return .critical;
        if (latency_ms > self.latency_warning_ms) return .warning;
        return .info;
    }

    /// Check stability and return alert level
    pub fn checkStability(self: *const AlertConfig, stability: f32) AlertLevel {
        if (stability < self.stability_emergency) return .emergency;
        if (stability < self.stability_critical) return .critical;
        if (stability < self.stability_warning) return .warning;
        return .info;
    }
};


// ============================================================================
// Alert Structure
// ============================================================================

/// Represents a triggered alert
pub const Alert = struct {
    /// Alert severity level
    level: AlertLevel,
    /// Name of the metric that triggered the alert
    metric: []const u8,
    /// Current value of the metric
    value: f32,
    /// Threshold that was exceeded
    threshold: f32,
    /// Unix timestamp when alert was triggered
    timestamp: i64,
    /// Optional message with context
    message: ?[]const u8,
    /// Source component
    source: []const u8,

    /// Create a new alert
    pub fn create(
        level: AlertLevel,
        metric: []const u8,
        value: f32,
        threshold: f32,
        message: ?[]const u8,
        source: []const u8,
    ) Alert {
        return .{
            .level = level,
            .metric = metric,
            .value = value,
            .threshold = threshold,
            .timestamp = std.time.timestamp(),
            .message = message,
            .source = source,
        };
    }

    /// Format alert for logging
    pub fn format(
        self: Alert,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("[{s}] {s}: {d:.4} (threshold: {d:.4}) - {s}", .{
            self.level.toString(),
            self.metric,
            self.value,
            self.threshold,
            self.message orelse "No additional context",
        });
    }
};

// ============================================================================
// PagerDuty Integration
// ============================================================================

/// Format an alert as a PagerDuty Events API v2 payload
pub fn formatPagerDutyPayload(
    alert: Alert,
    routing_key: []const u8,
    allocator: std.mem.Allocator,
) ![]const u8 {
    const timestamp_str = try std.fmt.allocPrint(allocator, "{d}", .{alert.timestamp});
    defer allocator.free(timestamp_str);

    const summary = if (alert.message) |msg|
        try std.fmt.allocPrint(allocator, "[mHC] {s}: {s}", .{ alert.metric, msg })
    else
        try std.fmt.allocPrint(allocator, "[mHC] {s} threshold exceeded: {d:.4}", .{ alert.metric, alert.value });
    defer allocator.free(summary);

    return std.fmt.allocPrint(allocator,
        \\{{
        \\  "routing_key": "{s}",
        \\  "event_action": "trigger",
        \\  "dedup_key": "mhc-{s}-{s}",
        \\  "payload": {{
        \\    "summary": "{s}",
        \\    "severity": "{s}",
        \\    "source": "{s}",
        \\    "timestamp": "{s}",
        \\    "component": "mhc-monitoring",
        \\    "group": "inference-engine",
        \\    "class": "performance",
        \\    "custom_details": {{
        \\      "metric": "{s}",
        \\      "value": {d:.6},
        \\      "threshold": {d:.6},
        \\      "alert_level": "{s}"
        \\    }}
        \\  }},
        \\  "client": "mHC Monitor",
        \\  "client_url": "https://metrics.example.com/mhc"
        \\}}
    , .{
        routing_key,
        alert.metric,
        timestamp_str,
        summary,
        alert.level.toPagerDutySeverity(),
        alert.source,
        timestamp_str,
        alert.metric,
        alert.value,
        alert.threshold,
        alert.level.toString(),
    });
}

// ============================================================================
// Slack Integration
// ============================================================================

/// Format an alert as a Slack webhook payload
pub fn formatSlackPayload(
    alert: Alert,
    channel: []const u8,
    allocator: std.mem.Allocator,
) ![]const u8 {
    const emoji = switch (alert.level) {
        .info => ":information_source:",
        .warning => ":warning:",
        .critical => ":rotating_light:",
        .emergency => ":fire:",
    };

    const message_text = if (alert.message) |msg|
        try std.fmt.allocPrint(allocator, "{s}", .{msg})
    else
        try std.fmt.allocPrint(allocator, "Threshold exceeded", .{});
    defer allocator.free(message_text);

    return std.fmt.allocPrint(allocator,
        \\{{
        \\  "channel": "{s}",
        \\  "username": "mHC Monitor",
        \\  "icon_emoji": "{s}",
        \\  "attachments": [
        \\    {{
        \\      "color": "{s}",
        \\      "title": "{s} Alert: {s}",
        \\      "text": "{s}",
        \\      "fields": [
        \\        {{
        \\          "title": "Metric",
        \\          "value": "{s}",
        \\          "short": true
        \\        }},
        \\        {{
        \\          "title": "Current Value",
        \\          "value": "{d:.4}",
        \\          "short": true
        \\        }},
        \\        {{
        \\          "title": "Threshold",
        \\          "value": "{d:.4}",
        \\          "short": true
        \\        }},
        \\        {{
        \\          "title": "Source",
        \\          "value": "{s}",
        \\          "short": true
        \\        }}
        \\      ],
        \\      "ts": {d}
        \\    }}
        \\  ]
        \\}}
    , .{
        channel,
        emoji,
        alert.level.getColor(),
        alert.level.toString(),
        alert.metric,
        message_text,
        alert.metric,
        alert.value,
        alert.threshold,
        alert.source,
        alert.timestamp,
    });
}


// ============================================================================
// Geometric Speculation Monitor
// ============================================================================

/// Comprehensive monitor for geometric speculation performance
/// Tracks acceptance rates, curvature, energy, and latency metrics
pub const GeometricSpeculationMonitor = struct {
    /// Buffer for acceptance rate metrics (0.0-1.0)
    acceptance_rate_buffer: DefaultMetricsBuffer,
    /// Buffer for curvature measurements
    curvature_buffer: DefaultMetricsBuffer,
    /// Buffer for energy values
    energy_buffer: DefaultMetricsBuffer,
    /// Buffer for latency measurements (ms)
    latency_buffer: DefaultMetricsBuffer,
    /// Buffer for stability scores
    stability_buffer: DefaultMetricsBuffer,
    /// Buffer for Sinkhorn iteration counts
    sinkhorn_iters_buffer: DefaultMetricsBuffer,

    /// Alert configuration
    alert_config: AlertConfig,
    /// Total number of speculations
    total_speculations: u64,
    /// Total accepted speculations
    accepted_speculations: u64,
    /// Last alert timestamp per metric (for cooldown)
    last_alert_time: std.StringHashMap(i64),
    /// Allocator for dynamic allocations
    allocator: std.mem.Allocator,

    /// Initialize a new monitor
    pub fn init(allocator: std.mem.Allocator, config: ?AlertConfig) GeometricSpeculationMonitor {
        return .{
            .acceptance_rate_buffer = DefaultMetricsBuffer.init(),
            .curvature_buffer = DefaultMetricsBuffer.init(),
            .energy_buffer = DefaultMetricsBuffer.init(),
            .latency_buffer = DefaultMetricsBuffer.init(),
            .stability_buffer = DefaultMetricsBuffer.init(),
            .sinkhorn_iters_buffer = DefaultMetricsBuffer.init(),
            .alert_config = config orelse AlertConfig{},
            .total_speculations = 0,
            .accepted_speculations = 0,
            .last_alert_time = std.StringHashMap(i64).init(allocator),
            .allocator = allocator,
        };
    }

    /// Deinitialize the monitor
    pub fn deinit(self: *GeometricSpeculationMonitor) void {
        self.last_alert_time.deinit();
    }

    /// Record an acceptance rate measurement
    pub fn recordAcceptance(self: *GeometricSpeculationMonitor, rate: f32) void {
        self.acceptance_rate_buffer.push(rate);
        self.total_speculations += 1;
        if (rate > 0.5) {
            self.accepted_speculations += 1;
        }
    }

    /// Record a curvature measurement
    pub fn recordCurvature(self: *GeometricSpeculationMonitor, curvature: f32) void {
        self.curvature_buffer.push(curvature);
    }

    /// Record an energy measurement
    pub fn recordEnergy(self: *GeometricSpeculationMonitor, energy: f32) void {
        self.energy_buffer.push(energy);
    }

    /// Record a latency measurement (in milliseconds)
    pub fn recordLatency(self: *GeometricSpeculationMonitor, latency_ms: f32) void {
        self.latency_buffer.push(latency_ms);
    }

    /// Record a stability score
    pub fn recordStability(self: *GeometricSpeculationMonitor, stability: f32) void {
        self.stability_buffer.push(stability);
    }

    /// Record Sinkhorn iteration count
    pub fn recordSinkhornIterations(self: *GeometricSpeculationMonitor, iterations: u32) void {
        self.sinkhorn_iters_buffer.push(@floatFromInt(iterations));
    }

    /// Check all thresholds and return the highest alert level
    pub fn checkThresholds(self: *GeometricSpeculationMonitor) AlertLevel {
        var max_level = AlertLevel.info;

        // Check acceptance rate
        if (!self.acceptance_rate_buffer.isEmpty()) {
            const level = self.alert_config.checkAcceptanceRate(self.acceptance_rate_buffer.mean());
            if (@intFromEnum(level) > @intFromEnum(max_level)) max_level = level;
        }

        // Check curvature
        if (!self.curvature_buffer.isEmpty()) {
            const level = self.alert_config.checkCurvature(self.curvature_buffer.mean());
            if (@intFromEnum(level) > @intFromEnum(max_level)) max_level = level;
        }

        // Check energy
        if (!self.energy_buffer.isEmpty()) {
            const level = self.alert_config.checkEnergy(self.energy_buffer.mean());
            if (@intFromEnum(level) > @intFromEnum(max_level)) max_level = level;
        }

        // Check latency
        if (!self.latency_buffer.isEmpty()) {
            const level = self.alert_config.checkLatency(self.latency_buffer.mean());
            if (@intFromEnum(level) > @intFromEnum(max_level)) max_level = level;
        }

        // Check stability
        if (!self.stability_buffer.isEmpty()) {
            const level = self.alert_config.checkStability(self.stability_buffer.mean());
            if (@intFromEnum(level) > @intFromEnum(max_level)) max_level = level;
        }

        return max_level;
    }

    /// Get alerts for all metrics exceeding thresholds
    pub fn getAlerts(self: *GeometricSpeculationMonitor, allocator: std.mem.Allocator) !std.ArrayList(Alert) {
        var alerts = std.ArrayList(Alert).init(allocator);
        const now = std.time.timestamp();

        // Check acceptance rate
        if (!self.acceptance_rate_buffer.isEmpty()) {
            const rate = self.acceptance_rate_buffer.mean();
            const level = self.alert_config.checkAcceptanceRate(rate);
            if (@intFromEnum(level) > @intFromEnum(AlertLevel.info)) {
                if (self.shouldAlert(MetricName.ACCEPTANCE_RATE, now)) {
                    try alerts.append(Alert.create(
                        level,
                        MetricName.ACCEPTANCE_RATE,
                        rate,
                        self.alert_config.acceptance_rate_warning,
                        "Low speculation acceptance rate",
                        "GeometricSpeculationMonitor",
                    ));
                }
            }
        }

        // Check curvature
        if (!self.curvature_buffer.isEmpty()) {
            const curvature = self.curvature_buffer.mean();
            const level = self.alert_config.checkCurvature(curvature);
            if (@intFromEnum(level) > @intFromEnum(AlertLevel.info)) {
                if (self.shouldAlert(MetricName.CURVATURE, now)) {
                    try alerts.append(Alert.create(
                        level,
                        MetricName.CURVATURE,
                        curvature,
                        self.alert_config.curvature_warning,
                        "High manifold curvature detected",
                        "GeometricSpeculationMonitor",
                    ));
                }
            }
        }

        return alerts;
    }

    /// Check if we should fire an alert (respects cooldown)
    fn shouldAlert(self: *GeometricSpeculationMonitor, metric: []const u8, now: i64) bool {
        const last_time = self.last_alert_time.get(metric) orelse 0;
        if (now - last_time >= @as(i64, self.alert_config.cooldown_seconds)) {
            self.last_alert_time.put(metric, now) catch {};
            return true;
        }
        return false;
    }

    /// Get overall acceptance rate
    pub fn getOverallAcceptanceRate(self: *const GeometricSpeculationMonitor) f32 {
        if (self.total_speculations == 0) return 0.0;
        return @as(f32, @floatFromInt(self.accepted_speculations)) /
            @as(f32, @floatFromInt(self.total_speculations));
    }

    /// Reset all metrics
    pub fn reset(self: *GeometricSpeculationMonitor) void {
        self.acceptance_rate_buffer.clear();
        self.curvature_buffer.clear();
        self.energy_buffer.clear();
        self.latency_buffer.clear();
        self.stability_buffer.clear();
        self.sinkhorn_iters_buffer.clear();
        self.total_speculations = 0;
        self.accepted_speculations = 0;
        self.last_alert_time.clearAndFree();
    }

    /// Get summary statistics
    pub fn getSummary(self: *GeometricSpeculationMonitor) MonitorSummary {
        return .{
            .acceptance_rate_mean = self.acceptance_rate_buffer.mean(),
            .curvature_mean = self.curvature_buffer.mean(),
            .energy_mean = self.energy_buffer.mean(),
            .latency_mean_ms = self.latency_buffer.mean(),
            .stability_mean = self.stability_buffer.mean(),
            .sinkhorn_iters_mean = self.sinkhorn_iters_buffer.mean(),
            .total_speculations = self.total_speculations,
            .accepted_speculations = self.accepted_speculations,
            .overall_acceptance_rate = self.getOverallAcceptanceRate(),
            .current_alert_level = self.checkThresholds(),
        };
    }
};

/// Summary of monitor metrics
pub const MonitorSummary = struct {
    acceptance_rate_mean: f32,
    curvature_mean: f32,
    energy_mean: f32,
    latency_mean_ms: f32,
    stability_mean: f32,
    sinkhorn_iters_mean: f32,
    total_speculations: u64,
    accepted_speculations: u64,
    overall_acceptance_rate: f32,
    current_alert_level: AlertLevel,
};


// ============================================================================
// Grafana Dashboard Configuration
// ============================================================================

/// Panel types for Grafana dashboards
pub const GrafanaPanelType = enum {
    graph,
    gauge,
    stat,
    table,
    heatmap,
    timeseries,

    pub fn toString(self: GrafanaPanelType) []const u8 {
        return switch (self) {
            .graph => "graph",
            .gauge => "gauge",
            .stat => "stat",
            .table => "table",
            .heatmap => "heatmap",
            .timeseries => "timeseries",
        };
    }
};

/// Generate a Grafana panel configuration for a specific metric
pub fn generateGrafanaPanel(
    metric_name: []const u8,
    panel_type: GrafanaPanelType,
    panel_id: u32,
    allocator: std.mem.Allocator,
) ![]const u8 {
    const title = try std.fmt.allocPrint(allocator, "mHC {s}", .{metric_name});
    defer allocator.free(title);

    const thresholds = getMetricThresholds(metric_name);

    return std.fmt.allocPrint(allocator,
        \\{{
        \\  "id": {d},
        \\  "type": "{s}",
        \\  "title": "{s}",
        \\  "gridPos": {{
        \\    "h": 8,
        \\    "w": 12,
        \\    "x": {d},
        \\    "y": {d}
        \\  }},
        \\  "datasource": "Prometheus",
        \\  "targets": [
        \\    {{
        \\      "expr": "{s}",
        \\      "legendFormat": "{{{{instance}}}}",
        \\      "refId": "A"
        \\    }}
        \\  ],
        \\  "fieldConfig": {{
        \\    "defaults": {{
        \\      "thresholds": {{
        \\        "mode": "absolute",
        \\        "steps": [
        \\          {{"color": "green", "value": null}},
        \\          {{"color": "yellow", "value": {d:.2}}},
        \\          {{"color": "orange", "value": {d:.2}}},
        \\          {{"color": "red", "value": {d:.2}}}
        \\        ]
        \\      }},
        \\      "unit": "{s}"
        \\    }}
        \\  }},
        \\  "options": {{
        \\    "legend": {{
        \\      "displayMode": "list",
        \\      "placement": "bottom"
        \\    }}
        \\  }}
        \\}}
    , .{
        panel_id,
        panel_type.toString(),
        title,
        (panel_id % 2) * 12,
        (panel_id / 2) * 8,
        metric_name,
        thresholds.warning,
        thresholds.critical,
        thresholds.emergency,
        getMetricUnit(metric_name),
    });
}

/// Threshold values for a metric
const MetricThresholds = struct {
    warning: f32,
    critical: f32,
    emergency: f32,
};

/// Get default thresholds for known metrics
fn getMetricThresholds(metric_name: []const u8) MetricThresholds {
    if (std.mem.eql(u8, metric_name, MetricName.ACCEPTANCE_RATE)) {
        return .{ .warning = 0.7, .critical = 0.5, .emergency = 0.3 };
    } else if (std.mem.eql(u8, metric_name, MetricName.CURVATURE)) {
        return .{ .warning = 2.0, .critical = 5.0, .emergency = 10.0 };
    } else if (std.mem.eql(u8, metric_name, MetricName.ENERGY)) {
        return .{ .warning = 100.0, .critical = 500.0, .emergency = 1000.0 };
    } else if (std.mem.eql(u8, metric_name, MetricName.LATENCY)) {
        return .{ .warning = 50.0, .critical = 100.0, .emergency = 500.0 };
    } else if (std.mem.eql(u8, metric_name, MetricName.STABILITY)) {
        return .{ .warning = 0.9, .critical = 0.8, .emergency = 0.5 };
    } else {
        return .{ .warning = 0.7, .critical = 0.8, .emergency = 0.9 };
    }
}

/// Get the unit for a metric
fn getMetricUnit(metric_name: []const u8) []const u8 {
    if (std.mem.eql(u8, metric_name, MetricName.LATENCY) or
        std.mem.eql(u8, metric_name, MetricName.MANIFOLD_PROJECTION))
    {
        return "ms";
    } else if (std.mem.eql(u8, metric_name, MetricName.ACCEPTANCE_RATE) or
        std.mem.eql(u8, metric_name, MetricName.STABILITY))
    {
        return "percentunit";
    } else {
        return "none";
    }
}

/// Generate a complete Grafana dashboard JSON configuration
pub fn generateDashboard(
    dashboard_title: []const u8,
    allocator: std.mem.Allocator,
) ![]const u8 {
    // Generate panels for all metrics
    const metrics = [_][]const u8{
        MetricName.ACCEPTANCE_RATE,
        MetricName.CURVATURE,
        MetricName.ENERGY,
        MetricName.LATENCY,
        MetricName.STABILITY,
        MetricName.SINKHORN_ITERATIONS,
    };

    var panels = std.ArrayListUnmanaged(u8){};
    defer panels.deinit(allocator);

    for (metrics, 0..) |metric, i| {
        if (i > 0) {
            try panels.appendSlice(allocator, ",\n    ");
        }
        const panel = try generateGrafanaPanel(
            metric,
            if (i < 2) .timeseries else .gauge,
            @intCast(i + 1),
            allocator,
        );
        defer allocator.free(panel);
        try panels.appendSlice(allocator, panel);
    }

    return std.fmt.allocPrint(allocator,
        \\{{
        \\  "dashboard": {{
        \\    "id": null,
        \\    "uid": "mhc-monitoring",
        \\    "title": "{s}",
        \\    "tags": ["mhc", "inference", "monitoring"],
        \\    "timezone": "browser",
        \\    "schemaVersion": 36,
        \\    "version": 1,
        \\    "refresh": "5s",
        \\    "time": {{
        \\      "from": "now-1h",
        \\      "to": "now"
        \\    }},
        \\    "panels": [
        \\    {s}
        \\    ],
        \\    "annotations": {{
        \\      "list": [
        \\        {{
        \\          "name": "Alerts",
        \\          "datasource": "-- Grafana --",
        \\          "enable": true,
        \\          "iconColor": "rgba(255, 96, 96, 1)",
        \\          "type": "dashboard"
        \\        }}
        \\      ]
        \\    }},
        \\    "templating": {{
        \\      "list": [
        \\        {{
        \\          "name": "instance",
        \\          "type": "query",
        \\          "datasource": "Prometheus",
        \\          "query": "label_values(mhc_acceptance_rate, instance)",
        \\          "refresh": 1,
        \\          "multi": true,
        \\          "includeAll": true
        \\        }}
        \\      ]
        \\    }}
        \\  }},
        \\  "overwrite": true
        \\}}
    , .{
        dashboard_title,
        panels.items,
    });
}


// ============================================================================
// Prometheus Export
// ============================================================================

/// Format metrics in Prometheus exposition format
pub fn formatPrometheusMetrics(
    monitor: *GeometricSpeculationMonitor,
    allocator: std.mem.Allocator,
) ![]const u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
    const writer = buffer.writer(allocator);

    // Acceptance Rate
    try writer.writeAll("# HELP mhc_acceptance_rate Geometric speculation acceptance rate\n");
    try writer.writeAll("# TYPE mhc_acceptance_rate gauge\n");
    try writer.print("mhc_acceptance_rate {d:.6}\n\n", .{monitor.acceptance_rate_buffer.mean()});

    // Curvature
    try writer.writeAll("# HELP mhc_curvature Current manifold curvature\n");
    try writer.writeAll("# TYPE mhc_curvature gauge\n");
    try writer.print("mhc_curvature {d:.6}\n\n", .{monitor.curvature_buffer.mean()});

    // Energy
    try writer.writeAll("# HELP mhc_energy Manifold energy level\n");
    try writer.writeAll("# TYPE mhc_energy gauge\n");
    try writer.print("mhc_energy {d:.6}\n\n", .{monitor.energy_buffer.mean()});

    // Latency
    try writer.writeAll("# HELP mhc_latency_ms Inference latency in milliseconds\n");
    try writer.writeAll("# TYPE mhc_latency_ms gauge\n");
    try writer.print("mhc_latency_ms {d:.6}\n\n", .{monitor.latency_buffer.mean()});

    // Stability
    try writer.writeAll("# HELP mhc_stability Manifold stability score\n");
    try writer.writeAll("# TYPE mhc_stability gauge\n");
    try writer.print("mhc_stability {d:.6}\n\n", .{monitor.stability_buffer.mean()});

    // Sinkhorn Iterations
    try writer.writeAll("# HELP mhc_sinkhorn_iterations Average Sinkhorn iterations\n");
    try writer.writeAll("# TYPE mhc_sinkhorn_iterations gauge\n");
    try writer.print("mhc_sinkhorn_iterations {d:.6}\n\n", .{monitor.sinkhorn_iters_buffer.mean()});

    // Total speculations
    try writer.writeAll("# HELP mhc_total_speculations Total number of speculations\n");
    try writer.writeAll("# TYPE mhc_total_speculations counter\n");
    try writer.print("mhc_total_speculations {d}\n\n", .{monitor.total_speculations});

    // Accepted speculations
    try writer.writeAll("# HELP mhc_accepted_speculations Total accepted speculations\n");
    try writer.writeAll("# TYPE mhc_accepted_speculations counter\n");
    try writer.print("mhc_accepted_speculations {d}\n\n", .{monitor.accepted_speculations});

    // Overall acceptance rate
    try writer.writeAll("# HELP mhc_overall_acceptance_rate Overall speculation acceptance rate\n");
    try writer.writeAll("# TYPE mhc_overall_acceptance_rate gauge\n");
    try writer.print("mhc_overall_acceptance_rate {d:.6}\n\n", .{monitor.getOverallAcceptanceRate()});

    return buffer.toOwnedSlice(allocator);
}

// ============================================================================
// Unit Tests
// ============================================================================

test "MetricsBuffer - init creates empty buffer" {
    var buffer = DefaultMetricsBuffer.init();
    try std.testing.expect(buffer.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), buffer.len());
    try std.testing.expectEqual(@as(f32, 0.0), buffer.mean());
}

test "MetricsBuffer - push adds values" {
    var buffer = DefaultMetricsBuffer.init();
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.push(3.0);

    try std.testing.expectEqual(@as(usize, 3), buffer.len());
    try std.testing.expect(!buffer.isEmpty());
}

test "MetricsBuffer - mean calculates correctly" {
    var buffer = DefaultMetricsBuffer.init();
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.push(3.0);
    buffer.push(4.0);

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), buffer.mean(), 0.001);
}

test "MetricsBuffer - min finds minimum" {
    var buffer = DefaultMetricsBuffer.init();
    buffer.push(5.0);
    buffer.push(2.0);
    buffer.push(8.0);
    buffer.push(1.0);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), buffer.min(), 0.001);
}

test "MetricsBuffer - max finds maximum" {
    var buffer = DefaultMetricsBuffer.init();
    buffer.push(5.0);
    buffer.push(2.0);
    buffer.push(8.0);
    buffer.push(1.0);

    try std.testing.expectApproxEqAbs(@as(f32, 8.0), buffer.max(), 0.001);
}

test "MetricsBuffer - latest returns most recent value" {
    var buffer = DefaultMetricsBuffer.init();
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.push(3.0);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), buffer.latest(), 0.001);
}

test "MetricsBuffer - stddev calculates correctly" {
    var buffer = DefaultMetricsBuffer.init();
    // Values: 2, 4, 4, 4, 5, 5, 7, 9 -> mean = 5, stddev ≈ 2
    buffer.push(2.0);
    buffer.push(4.0);
    buffer.push(4.0);
    buffer.push(4.0);
    buffer.push(5.0);
    buffer.push(5.0);
    buffer.push(7.0);
    buffer.push(9.0);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), buffer.stddev(), 0.1);
}

test "MetricsBuffer - percentile calculates correctly" {
    const allocator = std.testing.allocator;
    var buffer = DefaultMetricsBuffer.init();

    // Add values 1-10
    for (1..11) |i| {
        buffer.push(@floatFromInt(i));
    }

    // Median (50th percentile) should be around 5.5
    const p50 = try buffer.percentile(50, allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), p50, 0.5);

    // 90th percentile
    const p90 = try buffer.percentile(90, allocator);
    try std.testing.expect(p90 > 8.0);
}


test "MetricsBuffer - ring buffer overwrites oldest values" {
    // Use a small buffer for testing
    var buffer = MetricsBuffer(5).init();

    // Fill buffer
    for (1..6) |i| {
        buffer.push(@floatFromInt(i));
    }

    try std.testing.expect(buffer.isFull());
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), buffer.mean(), 0.001); // (1+2+3+4+5)/5

    // Push one more to overwrite oldest
    buffer.push(6.0);

    // Now should have 2,3,4,5,6 -> mean = 4.0
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), buffer.mean(), 0.001);
}

test "MetricsBuffer - clear resets buffer" {
    var buffer = DefaultMetricsBuffer.init();
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.clear();

    try std.testing.expect(buffer.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), buffer.len());
}

test "AlertLevel - toString returns correct strings" {
    try std.testing.expectEqualStrings("INFO", AlertLevel.info.toString());
    try std.testing.expectEqualStrings("WARNING", AlertLevel.warning.toString());
    try std.testing.expectEqualStrings("CRITICAL", AlertLevel.critical.toString());
    try std.testing.expectEqualStrings("EMERGENCY", AlertLevel.emergency.toString());
}

test "AlertLevel - getColor returns valid hex colors" {
    try std.testing.expect(AlertLevel.info.getColor().len > 0);
    try std.testing.expect(AlertLevel.warning.getColor()[0] == '#');
    try std.testing.expect(AlertLevel.critical.getColor()[0] == '#');
    try std.testing.expect(AlertLevel.emergency.getColor()[0] == '#');
}

test "AlertLevel - toPagerDutySeverity maps correctly" {
    try std.testing.expectEqualStrings("info", AlertLevel.info.toPagerDutySeverity());
    try std.testing.expectEqualStrings("warning", AlertLevel.warning.toPagerDutySeverity());
    try std.testing.expectEqualStrings("error", AlertLevel.critical.toPagerDutySeverity());
    try std.testing.expectEqualStrings("critical", AlertLevel.emergency.toPagerDutySeverity());
}

test "AlertConfig - checkAcceptanceRate returns correct levels" {
    const config = AlertConfig{};

    try std.testing.expectEqual(AlertLevel.info, config.checkAcceptanceRate(0.9));
    try std.testing.expectEqual(AlertLevel.warning, config.checkAcceptanceRate(0.6));
    try std.testing.expectEqual(AlertLevel.critical, config.checkAcceptanceRate(0.4));
    try std.testing.expectEqual(AlertLevel.emergency, config.checkAcceptanceRate(0.2));
}

test "AlertConfig - checkCurvature returns correct levels" {
    const config = AlertConfig{};

    try std.testing.expectEqual(AlertLevel.info, config.checkCurvature(1.0));
    try std.testing.expectEqual(AlertLevel.warning, config.checkCurvature(3.0));
    try std.testing.expectEqual(AlertLevel.critical, config.checkCurvature(7.0));
    try std.testing.expectEqual(AlertLevel.emergency, config.checkCurvature(15.0));
}

test "AlertConfig - checkLatency returns correct levels" {
    const config = AlertConfig{};

    try std.testing.expectEqual(AlertLevel.info, config.checkLatency(20.0));
    try std.testing.expectEqual(AlertLevel.warning, config.checkLatency(60.0));
    try std.testing.expectEqual(AlertLevel.critical, config.checkLatency(200.0));
    try std.testing.expectEqual(AlertLevel.emergency, config.checkLatency(600.0));
}

test "Alert - create initializes all fields" {
    const alert = Alert.create(
        .warning,
        "test_metric",
        0.5,
        0.7,
        "Test message",
        "TestSource",
    );

    try std.testing.expectEqual(AlertLevel.warning, alert.level);
    try std.testing.expectEqualStrings("test_metric", alert.metric);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), alert.value, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), alert.threshold, 0.001);
    try std.testing.expect(alert.timestamp > 0);
}

test "Alert - format produces valid output" {
    const alert = Alert.create(
        .critical,
        "test_metric",
        0.5,
        0.7,
        "Test message",
        "TestSource",
    );

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try alert.format("", .{}, fbs.writer());
    try std.testing.expect(fbs.pos > 0);
}

test "formatPagerDutyPayload - generates valid JSON" {
    const allocator = std.testing.allocator;
    const alert = Alert.create(
        .critical,
        "mhc_acceptance_rate",
        0.4,
        0.5,
        "Low acceptance rate",
        "Monitor",
    );

    const payload = try formatPagerDutyPayload(alert, "test_routing_key", allocator);
    defer allocator.free(payload);

    // Check it's valid JSON-ish (contains expected fields)
    try std.testing.expect(std.mem.indexOf(u8, payload, "routing_key") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "severity") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "mhc_acceptance_rate") != null);
}

test "formatSlackPayload - generates valid JSON" {
    const allocator = std.testing.allocator;
    const alert = Alert.create(
        .warning,
        "mhc_latency_ms",
        75.0,
        50.0,
        "High latency detected",
        "Monitor",
    );

    const payload = try formatSlackPayload(alert, "#alerts", allocator);
    defer allocator.free(payload);

    // Check it's valid JSON-ish (contains expected fields)
    try std.testing.expect(std.mem.indexOf(u8, payload, "channel") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "attachments") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "mhc_latency_ms") != null);
}


test "GeometricSpeculationMonitor - init and deinit" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    try std.testing.expectEqual(@as(u64, 0), monitor.total_speculations);
    try std.testing.expectEqual(@as(u64, 0), monitor.accepted_speculations);
}

test "GeometricSpeculationMonitor - recordAcceptance updates buffers" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    monitor.recordAcceptance(0.8);
    monitor.recordAcceptance(0.9);
    monitor.recordAcceptance(0.7);

    try std.testing.expectEqual(@as(u64, 3), monitor.total_speculations);
    try std.testing.expectEqual(@as(u64, 3), monitor.accepted_speculations);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), monitor.acceptance_rate_buffer.mean(), 0.01);
}

test "GeometricSpeculationMonitor - recordCurvature" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    monitor.recordCurvature(-1.0);
    monitor.recordCurvature(-1.5);

    try std.testing.expectApproxEqAbs(@as(f32, -1.25), monitor.curvature_buffer.mean(), 0.01);
}

test "GeometricSpeculationMonitor - recordEnergy" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    monitor.recordEnergy(50.0);
    monitor.recordEnergy(60.0);

    try std.testing.expectApproxEqAbs(@as(f32, 55.0), monitor.energy_buffer.mean(), 0.01);
}

test "GeometricSpeculationMonitor - recordLatency" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    monitor.recordLatency(10.0);
    monitor.recordLatency(20.0);
    monitor.recordLatency(30.0);

    try std.testing.expectApproxEqAbs(@as(f32, 20.0), monitor.latency_buffer.mean(), 0.01);
}

test "GeometricSpeculationMonitor - checkThresholds returns correct level" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    // With no data, should be info
    try std.testing.expectEqual(AlertLevel.info, monitor.checkThresholds());

    // Add low acceptance rate (should trigger warning)
    monitor.recordAcceptance(0.6);
    try std.testing.expectEqual(AlertLevel.warning, monitor.checkThresholds());
}

test "GeometricSpeculationMonitor - getSummary returns all metrics" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    monitor.recordAcceptance(0.85);
    monitor.recordCurvature(-1.0);
    monitor.recordEnergy(50.0);
    monitor.recordLatency(25.0);
    monitor.recordStability(0.95);
    monitor.recordSinkhornIterations(10);

    const summary = monitor.getSummary();

    try std.testing.expectApproxEqAbs(@as(f32, 0.85), summary.acceptance_rate_mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), summary.curvature_mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), summary.energy_mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), summary.latency_mean_ms, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), summary.stability_mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), summary.sinkhorn_iters_mean, 0.01);
}

test "GeometricSpeculationMonitor - reset clears all data" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    monitor.recordAcceptance(0.8);
    monitor.recordCurvature(-1.0);
    monitor.reset();

    try std.testing.expectEqual(@as(u64, 0), monitor.total_speculations);
    try std.testing.expect(monitor.acceptance_rate_buffer.isEmpty());
    try std.testing.expect(monitor.curvature_buffer.isEmpty());
}

test "GeometricSpeculationMonitor - getOverallAcceptanceRate" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    // Rate > 0.5 counts as accepted
    monitor.recordAcceptance(0.8); // accepted
    monitor.recordAcceptance(0.3); // not accepted
    monitor.recordAcceptance(0.9); // accepted
    monitor.recordAcceptance(0.6); // accepted

    // 3 out of 4 accepted
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), monitor.getOverallAcceptanceRate(), 0.01);
}

test "GrafanaPanelType - toString returns correct strings" {
    try std.testing.expectEqualStrings("graph", GrafanaPanelType.graph.toString());
    try std.testing.expectEqualStrings("gauge", GrafanaPanelType.gauge.toString());
    try std.testing.expectEqualStrings("timeseries", GrafanaPanelType.timeseries.toString());
}

test "generateGrafanaPanel - produces valid JSON" {
    const allocator = std.testing.allocator;

    const panel = try generateGrafanaPanel(
        MetricName.ACCEPTANCE_RATE,
        .timeseries,
        1,
        allocator,
    );
    defer allocator.free(panel);

    // Check for expected fields
    try std.testing.expect(std.mem.indexOf(u8, panel, "\"id\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, panel, "timeseries") != null);
    try std.testing.expect(std.mem.indexOf(u8, panel, "Prometheus") != null);
}

test "generateDashboard - produces complete dashboard JSON" {
    const allocator = std.testing.allocator;

    const dashboard = try generateDashboard("mHC Production Monitoring", allocator);
    defer allocator.free(dashboard);

    // Check for expected fields
    try std.testing.expect(std.mem.indexOf(u8, dashboard, "dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, dashboard, "panels") != null);
    try std.testing.expect(std.mem.indexOf(u8, dashboard, "mhc-monitoring") != null);
}

test "formatPrometheusMetrics - produces valid prometheus format" {
    const allocator = std.testing.allocator;
    var monitor = GeometricSpeculationMonitor.init(allocator, null);
    defer monitor.deinit();

    monitor.recordAcceptance(0.9);
    monitor.recordCurvature(-1.0);
    monitor.recordLatency(25.0);

    const metrics = try formatPrometheusMetrics(&monitor, allocator);
    defer allocator.free(metrics);

    // Check for expected fields
    try std.testing.expect(std.mem.indexOf(u8, metrics, "# HELP") != null);
    try std.testing.expect(std.mem.indexOf(u8, metrics, "# TYPE") != null);
    try std.testing.expect(std.mem.indexOf(u8, metrics, "mhc_acceptance_rate") != null);
    try std.testing.expect(std.mem.indexOf(u8, metrics, "mhc_curvature") != null);
}

test "MetricName constants are defined" {
    try std.testing.expect(MetricName.ACCEPTANCE_RATE.len > 0);
    try std.testing.expect(MetricName.CURVATURE.len > 0);
    try std.testing.expect(MetricName.ENERGY.len > 0);
    try std.testing.expect(MetricName.LATENCY.len > 0);
}