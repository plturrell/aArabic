// Industry Baselines - Reference Data Only
// These are published specifications and community benchmarks for comparison
// NOT used as actual measurements - only for gap analysis

const std = @import("std");

// ============================================================================
// GPU Hardware Specifications (Published by Manufacturers)
// ============================================================================

pub const T4Specs = struct {
    // NVIDIA Tesla T4 Official Specifications
    // Source: NVIDIA Data Sheet (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
    
    pub const name = "Tesla T4";
    pub const memory_gb: f32 = 16.0;
    pub const memory_bandwidth_gbs: f32 = 320.0;
    pub const fp32_tflops: f32 = 8.1;
    pub const fp16_tensor_tflops: f32 = 65.0;
    pub const int8_tensor_tops: f32 = 130.0;
    pub const compute_capability_major: i32 = 7;
    pub const compute_capability_minor: i32 = 5;
    pub const tdp_watts: u32 = 70;
};

pub const A100Specs = struct {
    // NVIDIA A100 - For reference (higher-end comparison)
    pub const name = "A100";
    pub const memory_gb: f32 = 40.0;
    pub const memory_bandwidth_gbs: f32 = 1555.0;
    pub const fp32_tflops: f32 = 19.5;
    pub const fp16_tensor_tflops: f32 = 312.0;
};

// ============================================================================
// Community Benchmarks (llama.cpp, vLLM, etc.)
// ============================================================================

pub const LlamaCppBenchmarks = struct {
    // Source: llama.cpp GitHub discussions and published benchmarks
    // These are typical performance numbers reported by the community
    
    pub const T4 = struct {
        // Matrix multiplication (cuBLAS SGEMM)
        pub const matmul_64_ms: f32 = 0.05;
        pub const matmul_256_ms: f32 = 0.3;
        pub const matmul_512_ms: f32 = 1.2;
        pub const matmul_1024_ms: f32 = 8.0;
        
        // Model inference (7B parameter model)
        pub const inference_7b_tokens_per_sec: f32 = 45.0;
        pub const inference_7b_batch8_tokens_per_sec: f32 = 280.0;
        
        // Memory usage
        pub const model_7b_vram_gb: f32 = 6.5;
        pub const model_13b_vram_gb: f32 = 12.0;
    };
    
    pub const CPU = struct {
        // Typical CPU performance (for reference)
        pub const matmul_256_ms: f32 = 180.0;  // ~600× slower than GPU
        pub const inference_7b_tokens_per_sec: f32 = 2.5;
    };
};

pub const vLLMBenchmarks = struct {
    // Source: vLLM published benchmarks
    pub const T4 = struct {
        pub const inference_7b_tokens_per_sec: f32 = 55.0;
        pub const inference_7b_batch8_tokens_per_sec: f32 = 350.0;
    };
};

// ============================================================================
// Expected Performance Ranges
// ============================================================================

pub const PerformanceRanges = struct {
    // What we should expect to see on T4 if GPU is properly utilized
    
    pub const ExpectedSpeedup = struct {
        pub const matmul_min: f32 = 50.0;    // Minimum acceptable
        pub const matmul_typical: f32 = 500.0;  // Typical with good optimization
        pub const matmul_max: f32 = 1500.0;  // Best case for large matrices
    };
    
    pub const ExpectedGFLOPS = struct {
        pub const cpu_typical: f32 = 2.0;     // Typical CPU GFLOPS
        pub const gpu_min: f32 = 100.0;       // Minimum if GPU is working
        pub const gpu_typical: f32 = 1000.0;  // Typical T4 GFLOPS
        pub const gpu_max: f32 = 2500.0;      // Near theoretical peak
    };
    
    pub const ExpectedMemoryBandwidth = struct {
        pub const t4_min_gbs: f32 = 250.0;   // 78% of peak
        pub const t4_typical_gbs: f32 = 290.0;  // 90% of peak
        pub const t4_max_gbs: f32 = 315.0;   // 98% of peak
    };
};

// ============================================================================
// Comparison Helpers
// ============================================================================

pub fn compareToBaseline(measured_speedup: f32) struct {
    status: []const u8,
    analysis: []const u8,
} {
    if (measured_speedup >= PerformanceRanges.ExpectedSpeedup.matmul_typical) {
        return .{
            .status = "EXCELLENT",
            .analysis = "Performance exceeds typical GPU acceleration",
        };
    } else if (measured_speedup >= PerformanceRanges.ExpectedSpeedup.matmul_min) {
        return .{
            .status = "GOOD",
            .analysis = "GPU acceleration active and performing well",
        };
    } else if (measured_speedup >= 10.0) {
        return .{
            .status = "SUBOPTIMAL",
            .analysis = "GPU detected but not fully utilized - check backend selection",
        };
    } else {
        return .{
            .status = "CPU_ONLY",
            .analysis = "CPU-only execution detected - GPU integration not active",
        };
    }
}

pub fn formatComparison(measured: f32, baseline: f32, unit: []const u8) ![]const u8 {
    const allocator = std.heap.page_allocator;
    const ratio = measured / baseline;
    
    if (ratio > 0.9 and ratio < 1.1) {
        return try std.fmt.allocPrint(allocator, 
            "{d:.2}{s} (matches baseline {d:.2}{s})", 
            .{measured, unit, baseline, unit}
        );
    } else if (ratio > 1.0) {
        return try std.fmt.allocPrint(allocator, 
            "{d:.2}{s} ({d:.1}× slower than baseline {d:.2}{s})", 
            .{measured, unit, ratio, baseline, unit}
        );
    } else {
        return try std.fmt.allocPrint(allocator, 
            "{d:.2}{s} ({d:.1}× faster than baseline {d:.2}{s})", 
            .{measured, unit, 1.0/ratio, baseline, unit}
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

test "baselines: T4 specs" {
    try std.testing.expectEqual(@as(f32, 320.0), T4Specs.memory_bandwidth_gbs);
    try std.testing.expectEqual(@as(f32, 8.1), T4Specs.fp32_tflops);
}

test "baselines: performance comparison" {
    const result = compareToBaseline(600.0);
    try std.testing.expectEqualStrings("EXCELLENT", result.status);
    
    const cpu_result = compareToBaseline(1.2);
    try std.testing.expectEqualStrings("CPU_ONLY", cpu_result.status);
}
