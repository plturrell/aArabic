// Memory Pre-Check Module
// Validates available memory before model loading to prevent OOM crashes

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Memory Requirements Estimation
// ============================================================================

pub const MemoryRequirements = struct {
    model_size_mb: u64,       // Size of model weights in RAM
    kv_cache_mb: u64,         // KV cache for max context
    activation_mb: u64,       // Activation buffers
    overhead_mb: u64,         // Runtime overhead
    total_mb: u64,            // Total required

    pub fn fromModelConfig(config: ModelMemoryConfig) MemoryRequirements {
        // Model weights (after dequantization if needed)
        const model_mb = config.param_count * config.bytes_per_param / (1024 * 1024);

        // KV cache: 2 * n_layers * max_seq_len * n_heads * head_dim * sizeof(f32)
        const kv_cache_mb = 2 * config.n_layers * config.max_seq_len * config.n_heads *
            config.head_dim * 4 / (1024 * 1024);

        // Activation buffers (rough estimate: batch_size * hidden_dim * 4 * 10)
        const activation_mb = config.batch_size * config.hidden_dim * 4 * 10 / (1024 * 1024);

        // Overhead (allocator, temp buffers, etc.) - 10% of model
        const overhead_mb = model_mb / 10;

        return .{
            .model_size_mb = model_mb,
            .kv_cache_mb = kv_cache_mb,
            .activation_mb = activation_mb,
            .overhead_mb = overhead_mb,
            .total_mb = model_mb + kv_cache_mb + activation_mb + overhead_mb,
        };
    }
};

pub const ModelMemoryConfig = struct {
    param_count: u64 = 0,        // Total parameters
    bytes_per_param: u64 = 4,    // 4 for f32, 2 for f16
    n_layers: u64 = 32,
    max_seq_len: u64 = 4096,
    n_heads: u64 = 32,
    head_dim: u64 = 128,
    hidden_dim: u64 = 4096,
    batch_size: u64 = 1,
};

// ============================================================================
// System Memory Check
// ============================================================================

pub const SystemMemory = struct {
    total_mb: u64,
    available_mb: u64,
    used_mb: u64,

    pub fn get() SystemMemory {
        if (builtin.os.tag == .macos) {
            return getMacOSMemory();
        } else if (builtin.os.tag == .linux) {
            return getLinuxMemory();
        } else {
            // Fallback: assume 8GB total, 4GB available
            return .{ .total_mb = 8192, .available_mb = 4096, .used_mb = 4096 };
        }
    }
};

fn getMacOSMemory() SystemMemory {
    // Use sysctl for total memory
    var total_mem: u64 = 0;
    var size: usize = @sizeOf(u64);

    // hw.memsize
    const mib = [_]c_int{ 6, 24 }; // CTL_HW, HW_MEMSIZE
    std.posix.sysctl(&mib, @ptrCast(&total_mem), &size, null, 0) catch {
        total_mem = 16 * 1024 * 1024 * 1024; // Fallback: 16GB
    };

    const total_mb = total_mem / (1024 * 1024);

    // Get current process RSS for used estimate
    // Use C rusage directly
    var rusage: extern struct {
        ru_utime: extern struct { tv_sec: isize, tv_usec: isize },
        ru_stime: extern struct { tv_sec: isize, tv_usec: isize },
        ru_maxrss: isize,
        ru_ixrss: isize,
        ru_idrss: isize,
        ru_isrss: isize,
        ru_minflt: isize,
        ru_majflt: isize,
        ru_nswap: isize,
        ru_inblock: isize,
        ru_oublock: isize,
        ru_msgsnd: isize,
        ru_msgrcv: isize,
        ru_nsignals: isize,
        ru_nvcsw: isize,
        ru_nivcsw: isize,
    } = undefined;

    const RUSAGE_SELF: c_int = 0;
    _ = std.c.getrusage(RUSAGE_SELF, @ptrCast(&rusage));
    const rss_mb: u64 = @intCast(@max(0, @divFloor(rusage.ru_maxrss, 1024)));

    // Estimate available (total - current RSS - 2GB for system)
    const system_reserve: u64 = 2048;
    const available_mb = if (total_mb > rss_mb + system_reserve)
        total_mb - rss_mb - system_reserve
    else
        total_mb / 4;

    return .{
        .total_mb = total_mb,
        .available_mb = available_mb,
        .used_mb = rss_mb,
    };
}

fn getLinuxMemory() SystemMemory {
    // Read /proc/meminfo
    const file = std.fs.openFileAbsolute("/proc/meminfo", .{}) catch {
        return .{ .total_mb = 8192, .available_mb = 4096, .used_mb = 4096 };
    };
    defer file.close();

    var buf: [4096]u8 = undefined;
    const bytes = file.readAll(&buf) catch {
        return .{ .total_mb = 8192, .available_mb = 4096, .used_mb = 4096 };
    };

    var total_kb: u64 = 0;
    var available_kb: u64 = 0;

    var lines = std.mem.splitScalar(u8, buf[0..bytes], '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "MemTotal:")) {
            total_kb = parseMemInfoValue(line);
        } else if (std.mem.startsWith(u8, line, "MemAvailable:")) {
            available_kb = parseMemInfoValue(line);
        }
    }

    const total_mb = total_kb / 1024;
    const available_mb = available_kb / 1024;

    return .{
        .total_mb = total_mb,
        .available_mb = available_mb,
        .used_mb = total_mb - available_mb,
    };
}

fn parseMemInfoValue(line: []const u8) u64 {
    // Format: "MemTotal:       16384 kB"
    var it = std.mem.tokenizeAny(u8, line, ": \t");
    _ = it.next(); // Skip label
    const value_str = it.next() orelse return 0;
    return std.fmt.parseInt(u64, value_str, 10) catch 0;
}

// ============================================================================
// Memory Pre-Check API
// ============================================================================

pub const MemoryCheckResult = struct {
    ok: bool,
    system: SystemMemory,
    required: MemoryRequirements,
    margin_mb: i64, // Positive = headroom, Negative = shortage

    pub fn format(self: MemoryCheckResult, writer: anytype) !void {
        try writer.print("Memory Check: {s}\n", .{if (self.ok) "✅ PASS" else "❌ FAIL"});
        try writer.print("  System: {d} MB total, {d} MB available\n", .{ self.system.total_mb, self.system.available_mb });
        try writer.print("  Required: {d} MB (model={d}, kv={d}, act={d}, overhead={d})\n", .{
            self.required.total_mb,
            self.required.model_size_mb,
            self.required.kv_cache_mb,
            self.required.activation_mb,
            self.required.overhead_mb,
        });
        try writer.print("  Margin: {d} MB\n", .{self.margin_mb});
    }
};

/// Check if system has enough memory for model with given config
pub fn checkMemory(config: ModelMemoryConfig) MemoryCheckResult {
    const system = SystemMemory.get();
    const required = MemoryRequirements.fromModelConfig(config);

    const margin: i64 = @as(i64, @intCast(system.available_mb)) -
        @as(i64, @intCast(required.total_mb));

    return .{
        .ok = margin >= 0,
        .system = system,
        .required = required,
        .margin_mb = margin,
    };
}

/// Check memory for a GGUF file by estimating from file size
pub fn checkMemoryForGGUF(file_size_bytes: u64, quant_type: QuantType) MemoryCheckResult {
    // Estimate params from file size
    const bytes_per_param: u64 = switch (quant_type) {
        .F32 => 4,
        .F16 => 2,
        .Q8_0 => 1,
        .Q4_0, .Q4_K => 1, // ~0.5 bytes but dequantizes to more
    };

    // After dequantization, weights take 4 bytes (f32)
    const estimated_params = file_size_bytes / bytes_per_param;

    return checkMemory(.{
        .param_count = estimated_params,
        .bytes_per_param = 4, // After dequantization
    });
}

pub const QuantType = enum { F32, F16, Q8_0, Q4_0, Q4_K };

/// Print memory check to stderr
pub fn printMemoryCheck(config: ModelMemoryConfig) void {
    const result = checkMemory(config);

    std.debug.print("\n🧠 Memory Pre-Check\n", .{});
    std.debug.print("   System: {d} MB total, {d} MB available\n", .{
        result.system.total_mb, result.system.available_mb,
    });
    std.debug.print("   Required: {d} MB\n", .{result.required.total_mb});
    std.debug.print("     - Model: {d} MB\n", .{result.required.model_size_mb});
    std.debug.print("     - KV Cache: {d} MB\n", .{result.required.kv_cache_mb});
    std.debug.print("     - Activations: {d} MB\n", .{result.required.activation_mb});
    std.debug.print("   Margin: {d} MB\n", .{result.margin_mb});
    std.debug.print("   Status: {s}\n\n", .{if (result.ok) "✅ OK" else "❌ INSUFFICIENT"});
}

// ============================================================================
// Unit Tests
// ============================================================================

test "MemoryRequirements calculation" {
    const reqs = MemoryRequirements.fromModelConfig(.{
        .param_count = 1_000_000_000, // 1B params
        .bytes_per_param = 4,
        .n_layers = 32,
        .max_seq_len = 4096,
        .n_heads = 32,
        .head_dim = 128,
    });

    // 1B * 4 bytes = 4GB for model
    try std.testing.expect(reqs.model_size_mb > 3000);
    try std.testing.expect(reqs.model_size_mb < 5000);

    // KV cache should be significant
    try std.testing.expect(reqs.kv_cache_mb > 0);

    // Total should include all components
    try std.testing.expect(reqs.total_mb > reqs.model_size_mb);
}

test "SystemMemory.get returns reasonable values" {
    const mem = SystemMemory.get();

    // Should have at least 1GB total
    try std.testing.expect(mem.total_mb >= 1024);

    // Available should be less than or equal to total
    try std.testing.expect(mem.available_mb <= mem.total_mb);
}

test "checkMemory passes for small model" {
    const result = checkMemory(.{
        .param_count = 1_000_000, // 1M params (tiny)
        .bytes_per_param = 4,
    });

    // A 4MB model should fit on any modern system
    try std.testing.expect(result.ok);
    try std.testing.expect(result.margin_mb > 0);
}

test "checkMemoryForGGUF estimates correctly" {
    const result = checkMemoryForGGUF(
        1024 * 1024 * 1024, // 1GB file
        .Q4_0,
    );

    // Should produce a requirement estimate
    try std.testing.expect(result.required.total_mb > 0);
}

test "QuantType enum values" {
    // Verify all quant types are defined
    _ = QuantType.F32;
    _ = QuantType.F16;
    _ = QuantType.Q8_0;
    _ = QuantType.Q4_0;
    _ = QuantType.Q4_K;
}

