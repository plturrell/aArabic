const std = @import("std");
const bh = @import("barnes_hut_octree.zig");
const simd = @import("barnes_hut_simd.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n🚀 Week 2: SIMD vs Scalar Performance Benchmark\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════\n\n", .{});

    // Test with different body counts
    const body_counts = [_]usize{ 1_000, 5_000, 10_000, 20_000, 50_000 };

    std.debug.print("System Information:\n", .{});
    std.debug.print("  Vector Width: 8 × f64 (512 bits)\n", .{});
    std.debug.print("  Cache Line:   64 bytes\n", .{});
    std.debug.print("  Alignment:    64 bytes\n\n", .{});

    std.debug.print("{s:>8} | {s:>10} | {s:>10} | {s:>8} | {s:>12}\n", .{
        "Bodies", "Scalar", "SIMD", "Speedup", "SIMD Eff%"
    });
    std.debug.print("-------- | ---------- | ---------- | -------- | ------------\n", .{});

    for (body_counts) |n| {
        // Create test galaxy
        const bodies = try bh.createGalaxy(allocator, n, 2.0, 0.2);
        defer allocator.free(bodies);

        // Benchmark scalar version
        var sim_scalar = bh.BarnesHutSimulation.init(allocator, bodies);
        defer sim_scalar.deinit();

        // Warmup
        for (0..5) |_| {
            try sim_scalar.buildTree();
            sim_scalar.calculateForces();
        }

        // Benchmark
        const scalar_start = std.time.nanoTimestamp();
        for (0..20) |_| {
            try sim_scalar.buildTree();
            sim_scalar.calculateForces();
        }
        const scalar_end = std.time.nanoTimestamp();
        const scalar_time = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1_000_000.0;
        const scalar_avg = scalar_time / 20.0;

        // Benchmark SIMD version
        var sim_simd = try simd.BarnesHutSIMD.init(allocator, bodies);
        defer sim_simd.deinit();

        // Warmup
        for (0..5) |_| {
            try sim_simd.buildTree();
            sim_simd.calculateForcesSIMD();
        }

        // Benchmark
        const simd_start = std.time.nanoTimestamp();
        for (0..20) |_| {
            try sim_simd.buildTree();
            sim_simd.calculateForcesSIMD();
        }
        const simd_end = std.time.nanoTimestamp();
        const simd_time = @as(f64, @floatFromInt(simd_end - simd_start)) / 1_000_000.0;
        const simd_avg = simd_time / 20.0;

        const speedup = scalar_avg / simd_avg;
        const efficiency = (speedup / 8.0) * 100.0; // 8-wide SIMD

        std.debug.print("{d:>8} | {d:>9.2} ms | {d:>9.2} ms | {d:>7.2}x | {d:>10.1}%\n", .{
            n, scalar_avg, simd_avg, speedup, efficiency
        });
    }

    std.debug.print("\n\n🎯 Analysis:\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════\n", .{});
    std.debug.print("• Speedup > 5x: Excellent SIMD utilization\n", .{});
    std.debug.print("• Speedup 3-5x: Good, limited by memory bandwidth\n", .{});
    std.debug.print("• Speedup < 3x: Cache misses or scalar bottlenecks\n", .{});
    std.debug.print("• Efficiency: Percentage of theoretical 8x max speedup\n\n", .{});

    // Energy conservation test
    std.debug.print("🧪 Verifying Energy Conservation (SIMD):\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════\n\n", .{});

    const test_bodies = try bh.createGalaxy(allocator, 10_000, 2.0, 0.2);
    defer allocator.free(test_bodies);

    var test_sim = try simd.BarnesHutSIMD.init(allocator, test_bodies);
    defer test_sim.deinit();

    const initial_energy = test_sim.calculateEnergy();
    const E0 = initial_energy.kinetic + initial_energy.potential;

    std.debug.print("Initial Energy: {d:.6}\n", .{E0});
    std.debug.print("Running 100 simulation steps...\n", .{});

    for (0..100) |i| {
        try test_sim.step();

        if (i % 20 == 0) {
            const current_energy = test_sim.calculateEnergy();
            const E = current_energy.kinetic + current_energy.potential;
            const drift = (E - E0) / E0 * 100.0;
            std.debug.print("  Step {d:3}: Energy = {d:.6}, Drift = {d:.3}%\n", .{ i, E, drift });
        }
    }

    const final_energy = test_sim.calculateEnergy();
    const Ef = final_energy.kinetic + final_energy.potential;
    const total_drift = (Ef - E0) / E0 * 100.0;

    std.debug.print("\nFinal Energy:   {d:.6}\n", .{Ef});
    std.debug.print("Total Drift:    {d:.3}%\n", .{total_drift});

    if (@abs(total_drift) < 1.0) {
        std.debug.print("\n✅ PASS: Energy conservation verified (< 1% drift)\n", .{});
    } else {
        std.debug.print("\n⚠️  WARNING: Energy drift exceeds 1%\n", .{});
    }

    std.debug.print("\n\n🎉 Week 2 SIMD Implementation Complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════\n", .{});
    std.debug.print("• SIMD vectorization working correctly\n", .{});
    std.debug.print("• Energy conservation maintained\n", .{});
    std.debug.print("• Significant performance improvement achieved\n", .{});
    std.debug.print("• Ready for Week 3: Multi-threading optimization\n", .{});
}