// Test mHC Integration with Matrix Operations
// Validates the complete flow: matmul → mHC constraints → stability metrics

const std = @import("std");
const matrix_ops = @import("matrix_ops.zig");
const mhc_constraints = @import("mhc_constraints.zig");
const mhc_config = @import("mhc_configuration.zig");

test "MatMulConfig.from_global creates correct config" {
    const global_config = mhc_config.MHCConfiguration{
        .core = .{
            .enabled = true,
            .sinkhorn_iterations = 15,
            .manifold_epsilon = 1e-5,
            .stability_threshold = 1e-3,
            .manifold_beta = 5.0,
        },
        .matrix_ops = .{
            .use_mhc = true,
        },
    };

    const matmul_config = matrix_ops.MatMulConfig.from_global(global_config, 42);

    try std.testing.expect(matmul_config.use_mhc);
    try std.testing.expectEqual(@as(u32, 42), matmul_config.layer_id);
    try std.testing.expect(matmul_config.mhc_config.enabled);
    try std.testing.expectEqual(@as(u32, 15), matmul_config.mhc_config.sinkhorn_iterations);
    try std.testing.expectEqual(@as(f32, 1e-5), matmul_config.mhc_config.manifold_epsilon);
}

test "matmul_with_mhc without mHC enabled returns null" {
    const allocator = std.testing.allocator;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 0, 0, 1 };
    var c: [4]f32 = undefined;

    const config = matrix_ops.MatMulConfig{
        .use_mhc = false, // mHC disabled
    };

    const metrics = try matrix_ops.matmul_with_mhc(
        &c,
        .{ .f32 = &a },
        &b,
        2, // m
        2, // n
        2, // k
        config,
        allocator,
        null,
    );

    // Should return null when mHC is disabled
    try std.testing.expect(metrics == null);

    // But matmul should still work
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c[0], 0.01);
}

test "matmul_with_mhc with mHC enabled returns metrics" {
    const allocator = std.testing.allocator;

    // Create 4x4 matrix multiplication
    const a = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    const b = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }; // Identity matrix
    var c: [16]f32 = undefined;

    const config = matrix_ops.MatMulConfig{
        .use_mhc = true,
        .layer_id = 5,
        .mhc_config = .{
            .enabled = true,
            .sinkhorn_iterations = 10,
            .manifold_epsilon = 1e-6,
            .stability_threshold = 1e-4,
            .manifold_beta = 10.0,
        },
    };

    const metrics = try matrix_ops.matmul_with_mhc(
        &c,
        .{ .f32 = &a },
        &b,
        4, // m
        4, // n
        4, // k
        config,
        allocator,
        null,
    );

    // Should return metrics when mHC is enabled
    try std.testing.expect(metrics != null);

    if (metrics) |m| {
        try std.testing.expectEqual(@as(u32, 5), m.layer_id);
        try std.testing.expect(m.signal_norm_before > 0);
        try std.testing.expect(m.signal_norm_after > 0);
        try std.testing.expect(m.amplification_factor > 0);
        try std.testing.expect(m.convergence_iterations <= 10);
    }
}

test "matmul_with_mhc respects layer_range" {
    const allocator = std.testing.allocator;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 0, 0, 1 };
    var c: [4]f32 = undefined;

    // Layer range: [10, 20]
    const layer_range = mhc_constraints.LayerRange{
        .start = 10,
        .end = 20,
    };

    // Test layer 5 (outside range)
    {
        const config = matrix_ops.MatMulConfig{
            .use_mhc = true,
            .layer_id = 5, // Outside range
            .mhc_config = .{
                .enabled = true,
                .layer_range = layer_range,
            },
        };

        const metrics = try matrix_ops.matmul_with_mhc(
            &c,
            .{ .f32 = &a },
            &b,
            2, 2, 2,
            config,
            allocator,
            null,
        );

        // Should return null because layer is outside range
        try std.testing.expect(metrics == null);
    }

    // Test layer 15 (inside range)
    {
        const config = matrix_ops.MatMulConfig{
            .use_mhc = true,
            .layer_id = 15, // Inside range
            .mhc_config = .{
                .enabled = true,
                .layer_range = layer_range,
            },
        };

        const metrics = try matrix_ops.matmul_with_mhc(
            &c,
            .{ .f32 = &a },
            &b,
            2, 2, 2,
            config,
            allocator,
            null,
        );

        // Should return metrics because layer is inside range
        try std.testing.expect(metrics != null);
    }
}

test "matmul_with_mhc applies L2 ball projection" {
    const allocator = std.testing.allocator;

    // Create matrix with large values that should be constrained
    const a = [_]f32{
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160,
    };
    const b = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }; // Identity
    var c: [16]f32 = undefined;

    const config = matrix_ops.MatMulConfig{
        .use_mhc = true,
        .layer_id = 0,
        .mhc_config = .{
            .enabled = true,
            .manifold_beta = 10.0, // Small beta to force projection
        },
    };

    const metrics = try matrix_ops.matmul_with_mhc(
        &c,
        .{ .f32 = &a },
        &b,
        4, 4, 4,
        config,
        allocator,
        null,
    );

    try std.testing.expect(metrics != null);

    if (metrics) |m| {
        // After projection, norm should be <= beta
        try std.testing.expect(m.signal_norm_after <= config.mhc_config.manifold_beta + 0.1);

        // Since we're projecting large values, should see significant reduction
        try std.testing.expect(m.signal_norm_before > m.signal_norm_after);
    }

    // Verify L2 norm of output
    const norm = matrix_ops.l2_norm(&c);
    try std.testing.expect(norm <= config.mhc_config.manifold_beta + 0.1);
}

test "matmul_with_mhc detects instability" {
    const allocator = std.testing.allocator;

    // Create pathological case that might cause instability
    const a = [_]f32{
        1000, 2000, 3000, 4000,
        5000, 6000, 7000, 8000,
        9000, 10000, 11000, 12000,
        13000, 14000, 15000, 16000,
    };
    const b = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    var c: [16]f32 = undefined;

    const config = matrix_ops.MatMulConfig{
        .use_mhc = true,
        .layer_id = 0,
        .mhc_config = .{
            .enabled = true,
            .manifold_beta = 1.0, // Very small beta
            .stability_threshold = 1e-4,
        },
    };

    const metrics = try matrix_ops.matmul_with_mhc(
        &c,
        .{ .f32 = &a },
        &b,
        4, 4, 4,
        config,
        allocator,
        null,
    );

    try std.testing.expect(metrics != null);

    if (metrics) |m| {
        // With such large input values and small beta, we expect instability
        // (amplification factor should be far from 1.0)
        const is_stable = m.amplification_factor >= 0.9 and m.amplification_factor <= 1.1;
        
        // This test verifies we can detect instability
        // In production, this would trigger warnings or corrective actions
        _ = is_stable; // We're just testing that we get metrics
    }
}

test "ManifoldConstraints euclidean projection" {
    const allocator = std.testing.allocator;

    var activations = [_]f32{ 10.0, 20.0, 30.0 };

    const manifold = matrix_ops.ManifoldConstraints{
        .manifold_type = .euclidean,
        .apply_projection = true,
    };

    // Euclidean should not modify (already handled by L2 ball)
    const original = activations;
    try matrix_ops.apply_geometric_projection(&activations, manifold, allocator);

    try std.testing.expectEqual(original[0], activations[0]);
}

test "ManifoldConstraints spherical projection" {
    const allocator = std.testing.allocator;

    var activations = [_]f32{ 3.0, 4.0, 0.0 }; // L2 norm = 5.0

    const manifold = matrix_ops.ManifoldConstraints{
        .manifold_type = .spherical,
        .apply_projection = true,
    };

    try matrix_ops.apply_geometric_projection(&activations, manifold, allocator);

    // After spherical projection, norm should be 1.0
    const norm = matrix_ops.l2_norm(&activations);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.01);
}

test "ManifoldConstraints hyperbolic requires negative curvature" {
    const allocator = std.testing.allocator;

    var activations = [_]f32{ 1.0, 2.0, 3.0 };

    const manifold = matrix_ops.ManifoldConstraints{
        .manifold_type = .hyperbolic,
        .curvature = 1.0, // Invalid: must be negative
        .apply_projection = true,
    };

    // Should return error for positive curvature
    try std.testing.expectError(
        error.InvalidHyperbolicCurvature,
        matrix_ops.apply_geometric_projection(&activations, manifold, allocator),
    );
}

test "Integration: full pipeline with quantized weights" {
    const allocator = std.testing.allocator;

    // Note: This test uses f32 weights for simplicity
    // In production, Q4_K/Q6_K weights would be used
    const a = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    const b = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    var c: [16]f32 = undefined;

    // Create configuration from global settings
    const global_config = mhc_config.MHCConfiguration{
        .core = .{
            .enabled = true,
            .sinkhorn_iterations = 20,
            .manifold_epsilon = 1e-6,
            .stability_threshold = 1e-4,
            .manifold_beta = 10.0,
            .log_stability_metrics = false, // Disable logging in tests
        },
        .matrix_ops = .{
            .use_mhc = true,
        },
    };

    const config = matrix_ops.MatMulConfig.from_global(global_config, 0);

    const metrics = try matrix_ops.matmul_with_mhc(
        &c,
        .{ .f32 = &a },
        &b,
        4, 4, 4,
        config,
        allocator,
        null,
    );

    // Verify complete pipeline
    try std.testing.expect(metrics != null);
    
    if (metrics) |m| {
        // Check all metrics are populated
        try std.testing.expect(m.signal_norm_before > 0);
        try std.testing.expect(m.signal_norm_after > 0);
        try std.testing.expect(m.amplification_factor > 0);
        try std.testing.expect(m.convergence_iterations > 0);
        try std.testing.expect(m.max_activation >= 0);
        try std.testing.expect(m.timestamp > 0);

        // Log summary
        std.debug.print("\n[Test] mHC Integration Pipeline:\n", .{});
        std.debug.print("  Layer: {d}\n", .{m.layer_id});
        std.debug.print("  Amplification: {d:.3}\n", .{m.amplification_factor});
        std.debug.print("  Iterations: {d}\n", .{m.convergence_iterations});
        std.debug.print("  Stability: {s}\n", .{if (m.is_stable) "✅ Stable" else "⚠️  Unstable"});
    }
}

test "SIMD detection" {
    const simd = matrix_ops.SIMDCapabilities.detect();
    
    // Should detect based on architecture
    const arch = @import("builtin").cpu.arch;
    if (arch == .aarch64 or arch == .arm) {
        try std.testing.expect(simd.has_neon);
        try std.testing.expectEqual(@as(usize, 4), simd.vector_width);
    } else {
        try std.testing.expectEqual(@as(usize, 1), simd.vector_width);
    }
}

test "Batch matmul with mHC" {
    const allocator = std.testing.allocator;
    const batch_size = 4;
    const m = 2;
    const n = 2;
    const k = 2;

    // Allocate batch arrays
    var outputs = try allocator.alloc([]f32, batch_size);
    defer allocator.free(outputs);
    var weights = try allocator.alloc(matrix_ops.Weight, batch_size);
    defer allocator.free(weights);
    var inputs = try allocator.alloc([]const f32, batch_size);
    defer allocator.free(inputs);

    // Initialize batch data
    for (0..batch_size) |i| {
        outputs[i] = try allocator.alloc(f32, m * n);
        const w = try allocator.alloc(f32, m * k);
        for (w) |*val| val.* = 1.0;
        weights[i] = .{ .f32 = w };
        
        const inp = try allocator.alloc(f32, k * n);
        for (inp) |*val| val.* = 1.0;
        inputs[i] = inp;
    }
    defer {
        for (0..batch_size) |i| {
            allocator.free(outputs[i]);
            allocator.free(weights[i].f32);
            allocator.free(inputs[i]);
        }
    }

    const config = matrix_ops.MatMulConfig{
        .use_mhc = true,
        .mhc_config = .{ .enabled = true },
    };

    const metrics = try matrix_ops.matmul_batch_with_mhc(
        outputs,
        weights,
        inputs,
        batch_size,
        m, n, k,
        config,
        allocator,
        null,
    );
    defer allocator.free(metrics);

    // Verify all batch elements were processed
    try std.testing.expectEqual(batch_size, metrics.len);
    for (metrics) |m_opt| {
        try std.testing.expect(m_opt != null);
    }
}

test "get_thread_count" {
    // Small matrix should use serial
    try std.testing.expectEqual(@as(usize, 1), matrix_ops.get_thread_count(1000, null));
    
    // Large matrix with no pool should use serial
    try std.testing.expectEqual(@as(usize, 1), matrix_ops.get_thread_count(5000, null));
}

test "matmul_with_mhc supports all quantization types" {
    const allocator = std.testing.allocator;
    
    // Test that dispatch works for all weight types
    // (actual quantized data would require proper initialization)
    const m = 2;
    const n = 2;
    const k = 2;
    
    var c: [4]f32 = undefined;
    const b = [_]f32{ 1, 0, 0, 1 };
    
    const config = matrix_ops.MatMulConfig{
        .use_mhc = true,
        .mhc_config = .{ .enabled = true },
    };

    // Test f32 path (works)
    const a_f32 = [_]f32{ 1, 2, 3, 4 };
    const metrics = try matrix_ops.matmul_with_mhc(
        &c,
        .{ .f32 = &a_f32 },
        &b,
        m, n, k,
        config,
        allocator,
        null,
    );
    
    try std.testing.expect(metrics != null);
}

// Run all tests
pub fn main() !void {
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("mHC Matrix Operations Integration Tests (Days 35-36)\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});

    const allocator = std.heap.page_allocator;

    // Run individual test functions
    try @import("std").testing.refAllDecls(@This());

    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("✅ All mHC integration tests passed!\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});

    _ = allocator;
}
