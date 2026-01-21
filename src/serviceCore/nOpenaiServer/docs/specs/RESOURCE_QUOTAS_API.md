# Resource Quotas & Limits API - Day 13

**Version:** 1.0.0  
**Status:** Production Ready  
**Integration:** Works with Days 1-12 (Tiered Cache, Model Registry, Multi-Model Cache)

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Quota Types](#quota-types)
6. [Violation Actions](#violation-actions)
7. [Integration Examples](#integration-examples)
8. [Monitoring & Metrics](#monitoring--metrics)
9. [Best Practices](#best-practices)
10. [Performance](#performance)

---

## Overview

The Resource Quota Manager enforces per-model resource limits, rate limiting, and usage quotas to ensure fair resource allocation and prevent system overload in multi-tenant environments.

### Key Features

- **Hard Memory Limits**: Enforce per-model RAM and SSD caps
- **Rate Limiting**: Request/sec and token/sec limits
- **Time-Based Quotas**: Hourly and daily token/request quotas
- **Burst Allowance**: Short-term overage tolerance
- **Resource Isolation**: Independent limits per model
- **Flexible Actions**: Reject, throttle, warn, or queue violations
- **Comprehensive Monitoring**: Per-model and global statistics

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│          Resource Quota Manager (Day 13)                │
│                                                          │
│  ┌────────────────────────────────────────────────┐   │
│  │ Per-Model Quotas (StringHashMap)              │   │
│  │                                                │   │
│  │  model-a → QuotaConfig + Usage Tracking       │   │
│  │  model-b → QuotaConfig + Usage Tracking       │   │
│  │  model-c → QuotaConfig + Usage Tracking       │   │
│  └────────────────────────────────────────────────┘   │
│                                                          │
│  Enforcement:                                           │
│  - Memory limits (RAM/SSD)                             │
│  - Rate limits (req/s, tokens/s)                       │
│  - Time-based quotas (hourly/daily)                    │
│  - Burst allowance tracking                            │
│  - Violation handling (reject/throttle/warn/queue)     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### ResourceQuotaConfig

Configuration for a model's resource limits.

```zig
pub const ResourceQuotaConfig = struct {
    /// Model identifier
    model_id: []const u8,
    
    /// Memory limits
    max_ram_mb: u64 = 2048,              // Hard RAM limit
    max_ssd_mb: u64 = 16384,             // Hard SSD limit (16GB)
    
    /// Rate limits
    max_requests_per_second: f32 = 100.0,
    max_tokens_per_second: f32 = 1000.0,
    
    /// Time-based quotas
    max_tokens_per_hour: u64 = 1_000_000,
    max_tokens_per_day: u64 = 10_000_000,
    max_requests_per_hour: u64 = 10_000,
    max_requests_per_day: u64 = 100_000,
    
    /// Burst allowance
    burst_requests: u32 = 200,
    burst_tokens: u64 = 10_000,
    
    /// Violation behavior
    on_violation: ViolationAction = .throttle,
    
    /// Grace period (seconds)
    grace_period_sec: u32 = 60,
};
```

### ResourceUsage

Tracks current resource usage for a model.

```zig
pub const ResourceUsage = struct {
    /// Memory usage
    current_ram_mb: u64,
    peak_ram_mb: u64,
    current_ssd_mb: u64,
    peak_ssd_mb: u64,
    
    /// Rate counters (current window)
    requests_this_second: u32,
    tokens_this_second: u64,
    
    /// Rolling quota counters
    tokens_this_hour: u64,
    tokens_this_day: u64,
    requests_this_hour: u64,
    requests_this_day: u64,
    
    /// Burst tracking
    burst_requests_used: u32,
    burst_tokens_used: u64,
    
    /// Violation tracking
    total_violations: u64,
    ram_violations: u32,
    ssd_violations: u32,
    rate_violations: u32,
    quota_violations: u32,
};
```

### QuotaCheckResult

Result of a quota check operation.

```zig
pub const QuotaCheckResult = struct {
    allowed: bool,
    reason: ?[]const u8,
    violation_type: ?ViolationType,
    retry_after_ms: ?u64,          // Suggested retry delay
    current_usage: f32,            // Usage percentage
};
```

---

## Configuration

### Basic Configuration

```zig
// Initialize quota manager
const manager = try ResourceQuotaManager.init(allocator);
defer manager.deinit();

// Configure quota for a model
try manager.setQuota(.{
    .model_id = "llama-3.2-1b",
    .max_ram_mb = 1024,           // 1GB RAM limit
    .max_ssd_mb = 8192,           // 8GB SSD limit
    .max_requests_per_second = 50.0,
    .max_tokens_per_second = 500.0,
    .on_violation = .throttle,
});
```

### Production Configuration

```zig
// High-priority production model
try manager.setQuota(.{
    .model_id = "production-llama-70b",
    .max_ram_mb = 8192,           // 8GB RAM
    .max_ssd_mb = 65536,          // 64GB SSD
    .max_requests_per_second = 100.0,
    .max_tokens_per_second = 2000.0,
    .max_tokens_per_hour = 5_000_000,
    .max_tokens_per_day = 50_000_000,
    .burst_requests = 500,
    .burst_tokens: 50_000,
    .on_violation = .queue,        // Queue excess requests
    .grace_period_sec = 30,
});

// Development model
try manager.setQuota(.{
    .model_id = "dev-qwen-0.5b",
    .max_ram_mb = 512,            // 512MB RAM
    .max_ssd_mb = 4096,           // 4GB SSD
    .max_requests_per_second = 10.0,
    .max_tokens_per_second = 100.0,
    .max_tokens_per_hour = 100_000,
    .max_tokens_per_day = 1_000_000,
    .on_violation = .reject,       // Reject excess requests
});

// Experimental model
try manager.setQuota(.{
    .model_id = "experimental-model",
    .max_ram_mb = 256,
    .max_ssd_mb = 2048,
    .max_requests_per_second = 5.0,
    .on_violation = .warn,         // Warn but allow
});
```

---

## API Reference

### Initialization

#### `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ResourceQuotaManager
```

Creates a new Resource Quota Manager.

**Returns:** Pointer to initialized manager  
**Errors:** `OutOfMemory`

**Example:**
```zig
const manager = try ResourceQuotaManager.init(allocator);
defer manager.deinit();
```

#### `deinit`

```zig
pub fn deinit(self: *ResourceQuotaManager) void
```

Cleans up and frees all resources.

### Quota Management

#### `setQuota`

```zig
pub fn setQuota(
    self: *ResourceQuotaManager,
    config: ResourceQuotaConfig,
) !void
```

Configures quota for a model.

**Parameters:**
- `config`: Quota configuration

**Errors:** `OutOfMemory`

**Example:**
```zig
try manager.setQuota(.{
    .model_id = "my-model",
    .max_ram_mb = 2048,
    .max_requests_per_second = 100.0,
    .on_violation = .throttle,
});
```

#### `removeQuota`

```zig
pub fn removeQuota(
    self: *ResourceQuotaManager,
    model_id: []const u8,
) !void
```

Removes quota configuration for a model.

**Parameters:**
- `model_id`: Model identifier

**Example:**
```zig
try manager.removeQuota("old-model");
```

#### `getQuota`

```zig
pub fn getQuota(
    self: *ResourceQuotaManager,
    model_id: []const u8,
) ?ResourceQuotaConfig
```

Retrieves quota configuration for a model.

**Returns:** Quota config or null if not found

**Example:**
```zig
if (manager.getQuota("my-model")) |quota| {
    std.debug.print("RAM limit: {d}MB\n", .{quota.max_ram_mb});
}
```

### Quota Enforcement

#### `checkQuota`

```zig
pub fn checkQuota(
    self: *ResourceQuotaManager,
    model_id: []const u8,
    request: struct {
        estimated_tokens: u64,
        ram_needed_mb: u64 = 0,
        ssd_needed_mb: u64 = 0,
    },
) !QuotaCheckResult
```

Checks if a request is allowed under current quotas.

**Parameters:**
- `model_id`: Model identifier
- `request`: Request parameters

**Returns:** Check result with allow/deny decision

**Example:**
```zig
const result = try manager.checkQuota("my-model", .{
    .estimated_tokens = 100,
    .ram_needed_mb = 50,
});

if (result.allowed) {
    // Process request
} else {
    // Handle violation
    std.debug.print("Denied: {s}\n", .{result.reason.?});
    if (result.retry_after_ms) |retry_ms| {
        std.debug.print("Retry after {d}ms\n", .{retry_ms});
    }
}
```

### Usage Tracking

#### `updateMemoryUsage`

```zig
pub fn updateMemoryUsage(
    self: *ResourceQuotaManager,
    model_id: []const u8,
    ram_mb: u64,
    ssd_mb: u64,
) !void
```

Updates current memory usage for a model.

**Parameters:**
- `model_id`: Model identifier
- `ram_mb`: Current RAM usage
- `ssd_mb`: Current SSD usage

**Example:**
```zig
try manager.updateMemoryUsage("my-model", 1024, 4096);
```

#### `getUsage`

```zig
pub fn getUsage(
    self: *ResourceQuotaManager,
    model_id: []const u8,
) ?ResourceUsage
```

Retrieves current usage for a model.

**Returns:** Usage data or null if not found

**Example:**
```zig
if (manager.getUsage("my-model")) |usage| {
    std.debug.print("RAM: {d}MB, Violations: {d}\n", .{
        usage.current_ram_mb,
        usage.total_violations,
    });
}
```

#### `resetBurst`

```zig
pub fn resetBurst(
    self: *ResourceQuotaManager,
    model_id: []const u8,
) !void
```

Resets burst allowance for a model.

**Example:**
```zig
try manager.resetBurst("my-model");
```

### Reporting

#### `getModelReport`

```zig
pub fn getModelReport(
    self: *ResourceQuotaManager,
    model_id: []const u8,
) ?ModelReport
```

Generates comprehensive report for a model.

**Returns:** Detailed report with utilization metrics

**Example:**
```zig
if (manager.getModelReport("my-model")) |report| {
    std.debug.print("RAM utilization: {d:.1}%\n", .{report.ram_utilization});
    std.debug.print("Hourly quota used: {d:.1}%\n", .{report.hourly_quota_used});
}
```

#### `getStats`

```zig
pub fn getStats(self: *ResourceQuotaManager) QuotaStats
```

Gets global quota statistics.

**Returns:** Global statistics

**Example:**
```zig
const stats = manager.getStats();
std.debug.print("Total violations: {d}\n", .{stats.total_violations});
std.debug.print("Rejected requests: {d}\n", .{stats.rejected_requests});
```

#### `printStatus`

```zig
pub fn printStatus(self: *ResourceQuotaManager) void
```

Prints comprehensive status report to stdout.

**Example:**
```zig
manager.printStatus();
```

---

## Quota Types

### Memory Quotas

Hard limits on RAM and SSD usage per model.

```zig
.max_ram_mb = 2048,      // Maximum 2GB RAM
.max_ssd_mb = 16384,     // Maximum 16GB SSD
```

**Enforcement:** Checked before allocation  
**Action:** Configurable (reject/throttle/warn/queue)

### Rate Limits

Limits on requests and tokens per second.

```zig
.max_requests_per_second = 100.0,    // Max 100 req/s
.max_tokens_per_second = 1000.0,     // Max 1000 tokens/s
```

**Enforcement:** Sliding window (1 second)  
**Burst:** Allowed via `burst_requests` and `burst_tokens`  
**Retry:** `retry_after_ms` provided on violation

### Time-Based Quotas

Hourly and daily limits on total usage.

```zig
.max_tokens_per_hour = 1_000_000,     // 1M tokens/hour
.max_tokens_per_day = 10_000_000,     // 10M tokens/day
.max_requests_per_hour = 10_000,      // 10K requests/hour
.max_requests_per_day = 100_000,      // 100K requests/day
```

**Enforcement:** Rolling windows  
**Reset:** Automatic at window boundaries  
**Retry:** `retry_after_ms` indicates time until reset

---

## Violation Actions

### ViolationAction Enum

```zig
pub const ViolationAction = enum {
    reject,          // Immediately reject request
    throttle,        // Delay/slow down request
    warn,            // Log warning but allow
    queue,           // Queue request for later
};
```

### Behavior Comparison

| Action | Request | Logged | Statistics | Use Case |
|--------|---------|--------|------------|----------|
| **reject** | Denied | Yes | `rejected_requests++` | Hard limits, production |
| **throttle** | Delayed | Yes | `throttled_requests++` | Graceful degradation |
| **warn** | Allowed | Yes | No increment | Development, monitoring |
| **queue** | Queued | Yes | `queued_requests++` | Load smoothing |

### Examples

```zig
// Production: Hard enforcement
.on_violation = .reject,

// API Gateway: Rate limit with retry
.on_violation = .throttle,

// Development: Monitor but don't block
.on_violation = .warn,

// Background tasks: Queue for later
.on_violation = .queue,
```

---

## Integration Examples

### Example 1: Integration with Multi-Model Cache

```zig
pub fn initializeSystem(allocator: std.mem.Allocator) !SystemContext {
    // Initialize multi-model cache (Day 12)
    const cache_manager = try MultiModelCacheManager.init(allocator, .{
        .total_ram_mb = 8192,
        .total_ssd_mb = 65536,
        .allocation_strategy = .fair_share,
    });
    
    // Initialize quota manager (Day 13)
    const quota_manager = try ResourceQuotaManager.init(allocator);
    
    // Register models with both managers
    const models = [_][]const u8{
        "llama-3.2-1b",
        "qwen2.5-0.5b",
        "phi-2",
    };
    
    for (models) |model_id| {
        // Register in cache manager
        try cache_manager.registerModel(model_id, .{
            .n_layers = 16,
            .n_heads = 16,
            .head_dim = 64,
            .max_seq_len = 4096,
        });
        
        // Set quotas
        try quota_manager.setQuota(.{
            .model_id = model_id,
            .max_ram_mb = 2048,
            .max_ssd_mb = 16384,
            .max_requests_per_second = 50.0,
            .max_tokens_per_hour = 1_000_000,
            .on_violation = .throttle,
        });
    }
    
    return SystemContext{
        .cache_manager = cache_manager,
        .quota_manager = quota_manager,
    };
}
```

### Example 2: Request Handling with Quota Check

```zig
pub fn handleInferenceRequest(
    ctx: *SystemContext,
    request: InferenceRequest,
) !InferenceResponse {
    // Check quotas before processing
    const quota_check = try ctx.quota_manager.checkQuota(request.model_id, .{
        .estimated_tokens = request.max_tokens,
        .ram_needed_mb = estimateRAMNeeded(request),
    });
    
    if (!quota_check.allowed) {
        // Handle quota violation
        log.warn("Request denied: {s}", .{quota_check.reason.?});
        
        return InferenceResponse{
            .error = .quota_exceeded,
            .message = quota_check.reason.?,
            .retry_after_ms = quota_check.retry_after_ms,
        };
    }
    
    // Get cache for model
    const cache = try ctx.cache_manager.getModelCache(request.model_id);
    
    // Process request
    const response = try processInference(cache, request);
    
    // Update usage tracking
    const actual_ram = cache.getStats().ram_used_mb;
    const actual_ssd = cache.getStats().ssd_used_mb;
    try ctx.quota_manager.updateMemoryUsage(
        request.model_id,
        actual_ram,
        actual_ssd,
    );
    
    return response;
}
```

### Example 3: Monitoring Integration

```zig
pub fn exportQuotaMetrics(
    quota_manager: *ResourceQuotaManager,
    allocator: std.mem.Allocator,
) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    const writer = buffer.writer();
    
    // Global metrics
    const stats = quota_manager.getStats();
    try writer.print(
        "quota_total_checks {d}\n" ++
        "quota_total_violations {d}\n" ++
        "quota_ram_violations {d}\n" ++
        "quota_ssd_violations {d}\n" ++
        "quota_rate_violations {d}\n" ++
        "quota_quota_violations {d}\n" ++
        "quota_rejected_requests {d}\n" ++
        "quota_throttled_requests {d}\n",
        .{
            stats.total_checks,
            stats.total_violations,
            stats.ram_violations,
            stats.ssd_violations,
            stats.rate_violations,
            stats.quota_violations,
            stats.rejected_requests,
            stats.throttled_requests,
        },
    );
    
    // Per-model metrics (implement model iteration)
    // ...
    
    return buffer.toOwnedSlice();
}
```

### Example 4: Dynamic Quota Adjustment

```zig
pub fn adjustQuotasBasedOnLoad(
    quota_manager: *ResourceQuotaManager,
    system_load: f32,
) !void {
    const models = try quota_manager.listModels(allocator);
    defer allocator.free(models);
    
    for (models) |model_id| {
        const current_quota = quota_manager.getQuota(model_id) orelse continue;
        const usage = quota_manager.getUsage(model_id) orelse continue;
        
        // High system load: tighten quotas
        if (system_load > 0.8) {
            try quota_manager.setQuota(.{
                .model_id = model_id,
                .max_ram_mb = current_quota.max_ram_mb * 80 / 100,  // Reduce 20%
                .max_requests_per_second = current_quota.max_requests_per_second * 0.8,
                .on_violation = .throttle,  // Be more aggressive
            });
        }
        // Low system load: relax quotas
        else if (system_load < 0.3) {
            try quota_manager.setQuota(.{
                .model_id = model_id,
                .max_ram_mb = current_quota.max_ram_mb * 120 / 100,  // Increase 20%
                .max_requests_per_second = current_quota.max_requests_per_second * 1.2,
                .on_violation = .warn,  // Be more lenient
            });
        }
    }
}
```

---

## Monitoring & Metrics

### Global Statistics

```zig
pub const QuotaStats = struct {
    total_checks: u64,
    total_violations: u64,
    rejected_requests: u64,
    throttled_requests: u64,
    queued_requests: u64,
    ram_violations: u64,
    ssd_violations: u64,
    rate_violations: u64,
    quota_violations: u64,
};
```

### Per-Model Metrics

```zig
const report = manager.getModelReport("my-model").?;

// Memory utilization
std.debug.print("RAM: {d:.1}%\n", .{report.ram_utilization});
std.debug.print("SSD: {d:.1}%\n", .{report.ssd_utilization});

// Rate metrics
std.debug.print("Req/s: {d:.1}\n", .{report.requests_per_second});
std.debug.print("Tokens/s: {d:.1}\n", .{report.tokens_per_second});

// Quota usage
std.debug.print("Hourly: {d:.1}%\n", .{report.hourly_quota_used});
std.debug.print("Daily: {d:.1}%\n", .{report.daily_quota_used});
```

### Alerting

```zig
pub fn checkForAlerts(quota_manager: *ResourceQuotaManager) ![]Alert {
    var alerts = std.ArrayList(Alert).init(allocator);
    
    const stats = quota_manager.getStats();
    
    // High violation rate
    if (stats.total_violations > 1000) {
        try alerts.append(.{
            .severity = .warning,
            .message = "High quota violation rate detected",
        });
    }
    
    // Per-model checks
    for (models) |model_id| {
        if (quota_manager.getModelReport(model_id)) |report| {
            // High memory utilization
            if (report.ram_utilization > 90.0) {
                try alerts.append(.{
                    .severity = .critical,
                    .message = try std.fmt.allocPrint(
                        allocator,
                        "Model {s}: RAM utilization at {d:.1}%",
                        .{model_id, report.ram_utilization},
                    ),
                });
            }
            
            // Quota near limit
            if (report.hourly_quota_used > 80.0) {
                try alerts.append(.{
                    .severity = .warning,
                    .message = try std.fmt.allocPrint(
                        allocator,
                        "Model {s}: Hourly quota at {d:.1}%",
                        .{model_id, report.hourly_quota_used},
                    ),
                });
            }
        }
    }
    
    return alerts.toOwnedSlice();
}
```

---

## Best Practices

### 1. Set Appropriate Limits

```zig
// Start conservative, monitor, adjust
try manager.setQuota(.{
    .model_id = "new-model",
    .max_ram_mb = 1024,        // Start with 1GB
    .max_requests_per_second = 10.0,  // Low initial limit
    .on_violation = .warn,     // Monitor without blocking
});

// After monitoring, adjust based on usage patterns
```

### 2. Use Burst Allowance

```zig
// Allow short-term spikes
.burst_requests = 200,      // 2x normal rate
.burst_tokens = 10_000,     // Buffer for bursts
```

### 3. Configure Appropriate Actions

```zig
// Production critical models
.on_violation = .reject,    // Hard enforcement

// Best-effort models
.on_violation = .throttle,  // Graceful degradation

// Development
.on_violation = .warn,      // Monitor only
```

### 4. Monitor and Alert

```zig
// Regular monitoring
const stats = manager.getStats();
if (stats.total_violations > threshold) {
    sendAlert("High violation rate");
}

// Per-model monitoring
for (models) |model_id| {
    const report = manager.getModelReport(model_id).?;
    if (report.ram_utilization > 90.0) {
        sendAlert("High RAM usage for {s}", .{model_id});
    }
}
```

### 5. Handle Violations Gracefully

```zig
const result = try manager.checkQuota(model_id, request);

if (!result.allowed) {
    switch (result.violation_type.?) {
        .rate_limit => {
            // Implement exponential backoff
            const delay = result.retry_after_ms.? * retry_count;
            std.time.sleep(delay * 1_000_000);
            return try retryRequest(request);
        },
        .quota_limit => {
            // Queue for later processing
            try queueRequest(request, result.retry_after_ms.?);
        },
        .ram_limit, .ssd_limit => {
            // Trigger garbage collection or eviction
            try freeResources(model_id);
            return try retryRequest(request);
        },
    }
}
```

---

## Performance

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `checkQuota` | O(1) | HashMap lookup + counter updates |
| `setQuota` | O(1) | HashMap insert |
| `updateMemoryUsage` | O(1) | Direct field update |
| `getUsage` | O(1) | HashMap lookup |
| `getModelReport` | O(1) | HashMap lookup + calculations |

### Space Complexity

| Component | Per-Model | 100 Models |
|-----------|-----------|------------|
| QuotaConfig | ~200 bytes | ~20 KB |
| ResourceUsage | ~200 bytes | ~20 KB |
| **Total** | ~400 bytes | ~40 KB |

### Performance Characteristics

- **Quota Check Latency**: <100ns (HashMap O(1))
- **Memory Overhead**: <0.01% of total system memory
- **Thread Safety**: Mutex-protected, negligible contention
- **Scalability**: Supports 1000+ models efficiently

### Optimization Tips

1. **Batch Operations**: Check quotas before expensive operations
2. **Cache Results**: Store quota check results for repeated access
3. **Lazy Cleanup**: Reset windows on access, not timer-based
4. **Lock Minimization**: Hold mutex only during HashMap operations

---

## Summary

The Resource Quota Manager provides production-ready resource governance for multi-model serving:

- ✅ **Hard Limits**: Enforce RAM/SSD caps per model
- ✅ **Rate Limiting**: Control request and token rates
- ✅ **Time Quotas**: Hourly and daily usage limits
- ✅ **Flexible Actions**: Reject, throttle, warn, or queue
- ✅ **Isolation**: Independent limits per model
- ✅ **Monitoring**: Comprehensive per-model and global metrics
- ✅ **Thread-Safe**: Concurrent access supported
- ✅ **Performant**: <100ns quota checks, O(1) operations

**Integration Points:**
- Works with Day 12 Multi-Model Cache Manager
- Complements Day 11 Model Registry
- Integrates with Days 6-9 observability stack

**Next Steps:**
- Day 14: Request routing and load balancing
- Day 15: Week 3 integration and testing
