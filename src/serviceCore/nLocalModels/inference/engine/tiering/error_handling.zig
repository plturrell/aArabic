// Error Handling & Circuit Breaker System for Production LLM Server
// Day 8: Resilient error handling with automatic recovery
//
// Features:
// - Circuit breaker pattern for SSD failures
// - Exponential backoff retry logic
// - Graceful degradation modes (RAM-only fallback)
// - Error metrics and alerting
// - Thread-safe state management

const std = @import("std");
const log = @import("structured_logging.zig");
const trace = @import("otel_tracing.zig");

/// Circuit breaker states following the standard pattern
pub const CircuitState = enum {
    closed,        // Normal operation, requests pass through
    open,          // Too many failures, reject requests immediately
    half_open,     // Testing if service recovered
    
    pub fn toString(self: CircuitState) []const u8 {
        return switch (self) {
            .closed => "CLOSED",
            .open => "OPEN",
            .half_open => "HALF_OPEN",
        };
    }
};

/// Error categories for classification
pub const ErrorCategory = enum {
    transient,      // Temporary, retry recommended
    permanent,      // Permanent, don't retry
    resource,       // Resource exhaustion (disk full, OOM)
    timeout,        // Operation timeout
    io_error,       // I/O operation failed
    
    pub fn shouldRetry(self: ErrorCategory) bool {
        return switch (self) {
            .transient, .timeout, .io_error => true,
            .permanent, .resource => false,
        };
    }
};

/// Circuit breaker configuration
pub const CircuitBreakerConfig = struct {
    /// Number of failures before opening circuit
    failure_threshold: u32 = 5,
    
    /// Time to wait before trying half_open (milliseconds)
    reset_timeout_ms: i64 = 30000,  // 30 seconds
    
    /// Number of successful requests in half_open before closing
    success_threshold: u32 = 2,
    
    /// Rolling window size for failure counting (milliseconds)
    window_size_ms: i64 = 60000,  // 1 minute
    
    /// Resource name for logging/metrics
    resource_name: []const u8 = "unknown",
};

/// Retry configuration with exponential backoff
pub const RetryConfig = struct {
    /// Maximum number of retry attempts
    max_attempts: u32 = 3,
    
    /// Initial backoff delay (milliseconds)
    initial_delay_ms: i64 = 100,
    
    /// Maximum backoff delay (milliseconds)
    max_delay_ms: i64 = 10000,  // 10 seconds
    
    /// Backoff multiplier
    multiplier: f64 = 2.0,
    
    /// Add jitter to prevent thundering herd
    jitter: bool = true,
};

/// Failure record for tracking
const FailureRecord = struct {
    timestamp: i64,
    error_category: ErrorCategory,
};

/// Circuit breaker implementation
pub const CircuitBreaker = struct {
    allocator: std.mem.Allocator,
    config: CircuitBreakerConfig,
    
    // State management
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: i64,
    state_change_time: i64,
    
    // Failure tracking
    failures: std.ArrayList(FailureRecord),
    
    // Thread safety
    mutex: std.Thread.Mutex,
    
    pub fn init(allocator: std.mem.Allocator, config: CircuitBreakerConfig) !*CircuitBreaker {
        const self = try allocator.create(CircuitBreaker);
        errdefer allocator.destroy(self);
        
        self.* = CircuitBreaker{
            .allocator = allocator,
            .config = config,
            .state = .closed,
            .failure_count = 0,
            .success_count = 0,
            .last_failure_time = 0,
            .state_change_time = std.time.milliTimestamp(),
            .failures = std.ArrayList(FailureRecord){},
            .mutex = .{},
        };
        
        log.info("Circuit breaker initialized: resource={s}, threshold={d}", .{
            config.resource_name,
            config.failure_threshold,
        });
        
        return self;
    }
    
    pub fn deinit(self: *CircuitBreaker) void {
        self.failures.deinit();
        self.allocator.destroy(self);
    }
    
    /// Check if request should be allowed
    pub fn allowRequest(self: *CircuitBreaker) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const now = std.time.milliTimestamp();
        
        switch (self.state) {
            .closed => return true,
            
            .open => {
                // Check if reset timeout elapsed
                if (now - self.state_change_time >= self.config.reset_timeout_ms) {
                    self.transitionToHalfOpen();
                    return true;
                }
                return false;
            },
            
            .half_open => return true,
        }
    }
    
    /// Record successful operation
    pub fn recordSuccess(self: *CircuitBreaker) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        switch (self.state) {
            .closed => {
                // Reset failure count on success in closed state
                self.failure_count = 0;
            },
            
            .half_open => {
                self.success_count += 1;
                if (self.success_count >= self.config.success_threshold) {
                    self.transitionToClosed();
                }
            },
            
            .open => {},  // Shouldn't happen, but safe to ignore
        }
    }
    
    /// Record failed operation
    pub fn recordFailure(self: *CircuitBreaker, category: ErrorCategory) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const now = std.time.milliTimestamp();
        
        // Add failure record
        self.failures.append(.{
            .timestamp = now,
            .error_category = category,
        }) catch {
            log.warn("Failed to record failure in circuit breaker", .{});
        };
        
        // Remove old failures outside window
        self.pruneOldFailures(now);
        
        self.last_failure_time = now;
        self.failure_count += 1;
        
        switch (self.state) {
            .closed => {
                if (self.failure_count >= self.config.failure_threshold) {
                    self.transitionToOpen();
                }
            },
            
            .half_open => {
                // Any failure in half_open immediately opens circuit
                self.transitionToOpen();
            },
            
            .open => {},  // Already open
        }
    }
    
    /// Get current circuit breaker metrics
    pub fn getMetrics(self: *CircuitBreaker) CircuitMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return .{
            .state = self.state,
            .failure_count = self.failure_count,
            .success_count = self.success_count,
            .recent_failures = @intCast(self.failures.items.len),
        };
    }
    
    // Private state transition methods
    
    fn transitionToClosed(self: *CircuitBreaker) void {
        const old_state = self.state;
        self.state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.state_change_time = std.time.milliTimestamp();
        
        log.info("Circuit breaker {s} → {s}: resource={s}", .{
            old_state.toString(),
            self.state.toString(),
            self.config.resource_name,
        });
    }
    
    fn transitionToOpen(self: *CircuitBreaker) void {
        const old_state = self.state;
        self.state = .open;
        self.success_count = 0;
        self.state_change_time = std.time.milliTimestamp();
        
        log.err("Circuit breaker {s} → {s}: resource={s}, failures={d}", .{
            old_state.toString(),
            self.state.toString(),
            self.config.resource_name,
            self.failure_count,
        });
    }
    
    fn transitionToHalfOpen(self: *CircuitBreaker) void {
        const old_state = self.state;
        self.state = .half_open;
        self.success_count = 0;
        self.state_change_time = std.time.milliTimestamp();
        
        log.info("Circuit breaker {s} → {s}: resource={s}, testing recovery", .{
            old_state.toString(),
            self.state.toString(),
            self.config.resource_name,
        });
    }
    
    fn pruneOldFailures(self: *CircuitBreaker, now: i64) void {
        const window_start = now - self.config.window_size_ms;
        
        var i: usize = 0;
        while (i < self.failures.items.len) {
            if (self.failures.items[i].timestamp < window_start) {
                _ = self.failures.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }
};

/// Circuit breaker metrics for monitoring
pub const CircuitMetrics = struct {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    recent_failures: u32,
};

/// Retry executor with exponential backoff
pub const RetryExecutor = struct {
    config: RetryConfig,
    
    pub fn init(config: RetryConfig) RetryExecutor {
        return .{ .config = config };
    }
    
    /// Execute operation with retry logic
    pub fn execute(
        self: *RetryExecutor,
        comptime T: type,
        operation: *const fn () anyerror!T,
        operation_name: []const u8,
    ) !T {
        var attempt: u32 = 0;
        var delay_ms: i64 = self.config.initial_delay_ms;
        
        while (attempt < self.config.max_attempts) : (attempt += 1) {
            const result = operation() catch |err| {
                const category = categorizeError(err);
                
                log.warn("Operation failed: name={s}, attempt={d}/{d}, error={s}, category={s}", .{
                    operation_name,
                    attempt + 1,
                    self.config.max_attempts,
                    @errorName(err),
                    @tagName(category),
                });
                
                // Don't retry permanent errors or resource errors
                if (!category.shouldRetry()) {
                    return err;
                }
                
                // Last attempt failed
                if (attempt == self.config.max_attempts - 1) {
                    return err;
                }
                
                // Calculate backoff delay
                const actual_delay = self.calculateBackoff(delay_ms);
                log.debug("Retrying after {d}ms backoff", .{actual_delay});
                
                std.time.sleep(@intCast(actual_delay * 1_000_000));
                delay_ms = @min(
                    @as(i64, @intFromFloat(@as(f64, @floatFromInt(delay_ms)) * self.config.multiplier)),
                    self.config.max_delay_ms,
                );
                
                continue;
            };
            
            if (attempt > 0) {
                log.info("Operation succeeded after {d} retries: name={s}", .{
                    attempt,
                    operation_name,
                });
            }
            
            return result;
        }
        
        return error.MaxRetriesExceeded;
    }
    
    fn calculateBackoff(self: *RetryExecutor, base_delay_ms: i64) i64 {
        if (!self.config.jitter) {
            return base_delay_ms;
        }
        
        // Add random jitter (±25%)
        const jitter_range = base_delay_ms / 4;
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const jitter = rng.random().intRangeAtMost(i64, -jitter_range, jitter_range);
        
        return base_delay_ms + jitter;
    }
};

/// Categorize error for retry decision
fn categorizeError(err: anyerror) ErrorCategory {
    return switch (err) {
        // Transient errors
        error.ConnectionRefused,
        error.ConnectionResetByPeer,
        error.BrokenPipe,
        error.NetworkUnreachable,
        => .transient,
        
        // Timeout errors
        error.Timeout,
        error.WouldBlock,
        => .timeout,
        
        // I/O errors
        error.InputOutput,
        error.FileNotFound,
        error.DeviceBusy,
        => .io_error,
        
        // Resource errors
        error.OutOfMemory,
        error.SystemResources,
        error.DiskQuota,
        error.NoSpaceLeft,
        => .resource,
        
        // Permanent errors
        error.AccessDenied,
        error.InvalidArgument,
        error.NotSupported,
        => .permanent,
        
        // Default to transient for unknown errors
        else => .transient,
    };
}

/// Graceful degradation mode
pub const DegradationMode = enum {
    normal,           // All tiers operational
    ssd_degraded,     // SSD failing, RAM-only mode
    memory_pressure,  // Low memory, aggressive eviction
    emergency,        // Severe issues, minimal functionality
    
    pub fn toString(self: DegradationMode) []const u8 {
        return switch (self) {
            .normal => "NORMAL",
            .ssd_degraded => "SSD_DEGRADED",
            .memory_pressure => "MEMORY_PRESSURE",
            .emergency => "EMERGENCY",
        };
    }
};

/// Degradation controller
pub const DegradationController = struct {
    mode: DegradationMode,
    mode_change_time: i64,
    mutex: std.Thread.Mutex,
    
    pub fn init() DegradationController {
        return .{
            .mode = .normal,
            .mode_change_time = std.time.milliTimestamp(),
            .mutex = .{},
        };
    }
    
    /// Get current degradation mode
    pub fn getMode(self: *DegradationController) DegradationMode {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.mode;
    }
    
    /// Set degradation mode
    pub fn setMode(self: *DegradationController, new_mode: DegradationMode) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.mode != new_mode) {
            const old_mode = self.mode;
            self.mode = new_mode;
            self.mode_change_time = std.time.milliTimestamp();
            
            log.warn("Degradation mode changed: {s} → {s}", .{
                old_mode.toString(),
                new_mode.toString(),
            });
        }
    }
    
    /// Check if SSD operations should be skipped
    pub fn shouldSkipSSD(self: *DegradationController) bool {
        return self.getMode() == .ssd_degraded or self.getMode() == .emergency;
    }
    
    /// Check if aggressive memory management needed
    pub fn needsAggressiveEviction(self: *DegradationController) bool {
        return self.getMode() == .memory_pressure or self.getMode() == .emergency;
    }
};

/// Error metrics for monitoring
pub const ErrorMetrics = struct {
    total_errors: std.atomic.Value(u64),
    transient_errors: std.atomic.Value(u64),
    permanent_errors: std.atomic.Value(u64),
    resource_errors: std.atomic.Value(u64),
    timeout_errors: std.atomic.Value(u64),
    io_errors: std.atomic.Value(u64),
    
    pub fn init() ErrorMetrics {
        return .{
            .total_errors = std.atomic.Value(u64).init(0),
            .transient_errors = std.atomic.Value(u64).init(0),
            .permanent_errors = std.atomic.Value(u64).init(0),
            .resource_errors = std.atomic.Value(u64).init(0),
            .timeout_errors = std.atomic.Value(u64).init(0),
            .io_errors = std.atomic.Value(u64).init(0),
        };
    }
    
    pub fn recordError(self: *ErrorMetrics, category: ErrorCategory) void {
        _ = self.total_errors.fetchAdd(1, .monotonic);
        
        switch (category) {
            .transient => _ = self.transient_errors.fetchAdd(1, .monotonic),
            .permanent => _ = self.permanent_errors.fetchAdd(1, .monotonic),
            .resource => _ = self.resource_errors.fetchAdd(1, .monotonic),
            .timeout => _ = self.timeout_errors.fetchAdd(1, .monotonic),
            .io_error => _ = self.io_errors.fetchAdd(1, .monotonic),
        }
    }
    
    pub fn getSnapshot(self: *ErrorMetrics) ErrorSnapshot {
        return .{
            .total = self.total_errors.load(.monotonic),
            .transient = self.transient_errors.load(.monotonic),
            .permanent = self.permanent_errors.load(.monotonic),
            .resource = self.resource_errors.load(.monotonic),
            .timeout = self.timeout_errors.load(.monotonic),
            .io_errors = self.io_errors.load(.monotonic),
        };
    }
};

pub const ErrorSnapshot = struct {
    total: u64,
    transient: u64,
    permanent: u64,
    resource: u64,
    timeout: u64,
    io_errors: u64,
};

// ============================================================================
// Global Error Handling Infrastructure
// ============================================================================

var global_error_metrics: ?*ErrorMetrics = null;
var global_degradation: ?*DegradationController = null;
var global_mutex: std.Thread.Mutex = .{};

/// Initialize global error handling
pub fn initGlobalErrorHandling(allocator: std.mem.Allocator) !void {
    global_mutex.lock();
    defer global_mutex.unlock();
    
    if (global_error_metrics != null) {
        return error.ErrorHandlingAlreadyInitialized;
    }
    
    const metrics = try allocator.create(ErrorMetrics);
    metrics.* = ErrorMetrics.init();
    global_error_metrics = metrics;
    
    const degradation = try allocator.create(DegradationController);
    degradation.* = DegradationController.init();
    global_degradation = degradation;
    
    log.info("Global error handling initialized", .{});
}

/// Deinitialize global error handling
pub fn deinitGlobalErrorHandling(allocator: std.mem.Allocator) void {
    global_mutex.lock();
    defer global_mutex.unlock();
    
    if (global_error_metrics) |metrics| {
        allocator.destroy(metrics);
        global_error_metrics = null;
    }
    
    if (global_degradation) |degradation| {
        allocator.destroy(degradation);
        global_degradation = null;
    }
}

/// Get global error metrics
pub fn getGlobalErrorMetrics() ?*ErrorMetrics {
    global_mutex.lock();
    defer global_mutex.unlock();
    return global_error_metrics;
}

/// Get global degradation controller
pub fn getGlobalDegradation() ?*DegradationController {
    global_mutex.lock();
    defer global_mutex.unlock();
    return global_degradation;
}
