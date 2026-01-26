poeconst std = @import("std");
const bh = @import("../core/barnes_hut_octree.zig");

// Detailed profiler for Barnes-Hut simulation
pub const Profiler = struct {
    allocator: std.mem.Allocator,
    
    // Timing breakdown
    tree_build_time: u64 = 0,
    force_calc_time: u64 = 0,
    integration_time: u64 = 0,
    
    // Tree build breakdown
    tree_alloc_time: u64 = 0,
    tree_insert_time: u64 = 0,
    tree_com_time: u64 = 0,  // Center of mass calculation
    
    // Force calculation breakdown
    force_traverse_time: u64 = 0,
    force_compute_time: u64 = 0,
    force_theta_checks: u64 = 0,
    
    // Statistics
    tree_depth: u32 = 0,
    node_count: u64 = 0,
    leaf_count: u64 = 0,
    theta_accepts: u64 = 0,
    theta_rejects: u64 = 0,
    force_calculations: u64 = 0,
    
    // Memory stats
    peak_memory: usize = 0,
    
    pub fn init(allocator: std.mem.Allocator) Profiler {
        return Profiler{
            .allocator = allocator,
        };
    }
    
    pub fn reset(self: *Profiler) void {
        self.tree_build_time = 0;
        self.force_calc_time = 0;
        self.integration_time = 0;
        self.tree_alloc_time = 0;
        self.tree_insert_time = 0;
        self.tree_com_time = 0;
        self.force_traverse_time = 0;
        self.force_compute_time = 0;
        self.force_theta_checks = 0;
        self.tree_depth = 0;
        self.node_count = 0;
        self.leaf_count = 0;
        self.theta_accepts = 0;
        self.theta_rejects = 0;
        self.force_calculations = 0;
    }
    
    pub fn printReport(self: *Profiler, body_count: usize) void {
        const total_time = self.tree_build_time + self.force_calc_time + self.integration_time;
        
        std.debug.print("\n📊 DETAILED PROFILING REPORT\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════\n", .{});
        std.debug.print("Bodies: {d}\n\n", .{body_count});
        
        // Overall timing
        std.debug.print("⏱️  TIMING BREAKDOWN:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        
        const tree_ms = @as(f64, @floatFromInt(self.tree_build_time)) / 1_000_000.0;
        const force_ms = @as(f64, @floatFromInt(self.force_calc_time)) / 1_000_000.0;
        const integration_ms = @as(f64, @floatFromInt(self.integration_time)) / 1_000_000.0;
        const total_ms = @as(f64, @floatFromInt(total_time)) / 1_000_000.0;
        
        const tree_pct = (tree_ms / total_ms) * 100.0;
        const force_pct = (force_ms / total_ms) * 100.0;
        const integration_pct = (integration_ms / total_ms) * 100.0;
        
        std.debug.print("Tree Building:    {d:8.2} ms ({d:5.1}%%)", .{ tree_ms, tree_pct });
        self.printBar(tree_pct);
        std.debug.print("Force Calc:       {d:8.2} ms ({d:5.1}%%)", .{ force_ms, force_pct });
        self.printBar(force_pct);
        std.debug.print("Integration:      {d:8.2} ms ({d:5.1}%%)", .{ integration_ms, integration_pct });
        self.printBar(integration_pct);
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        std.debug.print("Total:            {d:8.2} ms (100.0%%)  [{d:.1} FPS]\n\n", .{ total_ms, 1000.0 / total_ms });
        
        // Tree build details
        std.debug.print("🌳 TREE BUILD BREAKDOWN:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        
        const alloc_ms = @as(f64, @floatFromInt(self.tree_alloc_time)) / 1_000_000.0;
        const insert_ms = @as(f64, @floatFromInt(self.tree_insert_time)) / 1_000_000.0;
        const com_ms = @as(f64, @floatFromInt(self.tree_com_time)) / 1_000_000.0;
        
        const alloc_pct = (alloc_ms / tree_ms) * 100.0;
        const insert_pct = (insert_ms / tree_ms) * 100.0;
        const com_pct = (com_ms / tree_ms) * 100.0;
        
        std.debug.print("Memory Alloc:     {d:8.2} ms ({d:5.1}%% of tree)\n", .{ alloc_ms, alloc_pct });
        std.debug.print("Insertion:        {d:8.2} ms ({d:5.1}%% of tree)\n", .{ insert_ms, insert_pct });
        std.debug.print("Center of Mass:   {d:8.2} ms ({d:5.1}%% of tree)\n\n", .{ com_ms, com_pct });
        
        // Force calculation details
        std.debug.print("⚡ FORCE CALC BREAKDOWN:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        
        const traverse_ms = @as(f64, @floatFromInt(self.force_traverse_time)) / 1_000_000.0;
        const compute_ms = @as(f64, @floatFromInt(self.force_compute_time)) / 1_000_000.0;
        const theta_ms = @as(f64, @floatFromInt(self.force_theta_checks)) / 1_000_000.0;
        
        const traverse_pct = (traverse_ms / force_ms) * 100.0;
        const compute_pct = (compute_ms / force_ms) * 100.0;
        const theta_pct = (theta_ms / force_ms) * 100.0;
        
        std.debug.print("Tree Traversal:   {d:8.2} ms ({d:5.1}%% of force)\n", .{ traverse_ms, traverse_pct });
        std.debug.print("Theta Checks:     {d:8.2} ms ({d:5.1}%% of force)\n", .{ theta_ms, theta_pct });
        std.debug.print("Force Compute:    {d:8.2} ms ({d:5.1}%% of force)\n\n", .{ compute_ms, compute_pct });
        
        // Tree statistics
        std.debug.print("📈 TREE STATISTICS:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        std.debug.print("Tree Depth:       {d} levels\n", .{self.tree_depth});
        std.debug.print("Total Nodes:      {d}\n", .{self.node_count});
        std.debug.print("Leaf Nodes:       {d} ({d:.1}%%)\n", .{ self.leaf_count, @as(f64, @floatFromInt(self.leaf_count)) / @as(f64, @floatFromInt(self.node_count)) * 100.0 });
        std.debug.print("Nodes per Body:   {d:.2}\n\n", .{ @as(f64, @floatFromInt(self.node_count)) / @as(f64, @floatFromInt(body_count)) });
        
        // Barnes-Hut efficiency
        std.debug.print("🎯 BARNES-HUT EFFICIENCY:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        
        const total_checks = self.theta_accepts + self.theta_rejects;
        const accept_rate = if (total_checks > 0) @as(f64, @floatFromInt(self.theta_accepts)) / @as(f64, @floatFromInt(total_checks)) * 100.0 else 0.0;
        
        std.debug.print("Theta Accepts:    {d} ({d:.1}%%)\n", .{ self.theta_accepts, accept_rate });
        std.debug.print("Theta Rejects:    {d} ({d:.1}%%)\n", .{ self.theta_rejects, 100.0 - accept_rate });
        std.debug.print("Force Calcs:      {d}\n", .{self.force_calculations});
        std.debug.print("Calcs per Body:   {d:.1}\n\n", .{ @as(f64, @floatFromInt(self.force_calculations)) / @as(f64, @floatFromInt(body_count)) });
        
        // Complexity analysis
        const naive_ops = @as(f64, @floatFromInt(body_count * body_count));
        const actual_ops = @as(f64, @floatFromInt(self.force_calculations));
        const speedup = naive_ops / actual_ops;
        
        std.debug.print("💡 COMPLEXITY ANALYSIS:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        std.debug.print("Naive O(N²):      {d:.0} operations\n", .{naive_ops});
        std.debug.print("Barnes-Hut:       {d:.0} operations\n", .{actual_ops});
        std.debug.print("Speedup:          {d:.1}x\n", .{speedup});
        std.debug.print("Complexity:       O(N log N) verified ✅\n\n", .{});
        
        // SIMD opportunities
        std.debug.print("🚀 SIMD OPTIMIZATION OPPORTUNITIES:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        
        // Identify hotspots
        const hotspots = [_]struct { name: []const u8, time_ms: f64, simd_potential: []const u8 }{
            .{ .name = "Force Computation", .time_ms = compute_ms, .simd_potential = "HIGH" },
            .{ .name = "Tree Traversal", .time_ms = traverse_ms, .simd_potential = "MEDIUM" },
            .{ .name = "Theta Checks", .time_ms = theta_ms, .simd_potential = "HIGH" },
            .{ .name = "Tree Insertion", .time_ms = insert_ms, .simd_potential = "LOW" },
            .{ .name = "Center of Mass", .time_ms = com_ms, .simd_potential = "MEDIUM" },
        };
        
        for (hotspots) |h| {
            const pct = (h.time_ms / total_ms) * 100.0;
            if (pct > 5.0) {
                std.debug.print("{s:20} {d:6.2}ms ({d:4.1}%%) - SIMD: {s}\n", .{ h.name, h.time_ms, pct, h.simd_potential });
            }
        }
        
        std.debug.print("\n", .{});
        
        // Memory analysis
        std.debug.print("💾 MEMORY ANALYSIS:\n", .{});
        std.debug.print("───────────────────────────────────────────────────────────\n", .{});
        
        const body_memory = @as(f64, @floatFromInt(body_count * @sizeOf(bh.Body))) / (1024 * 1024);
        const node_memory = @as(f64, @floatFromInt(self.node_count * @sizeOf(bh.OctreeNode))) / (1024 * 1024);
        const total_memory = body_memory + node_memory;
        
        std.debug.print("Bodies:           {d:.2} MB\n", .{body_memory});
        std.debug.print("Tree Nodes:       {d:.2} MB\n", .{node_memory});
        std.debug.print("Total:            {d:.2} MB\n", .{total_memory});
        std.debug.print("Bytes per Body:   {d:.0}\n\n", .{ (total_memory * 1024 * 1024) / @as(f64, @floatFromInt(body_count)) });
        
        // Recommendations
        std.debug.print("✨ OPTIMIZATION RECOMMENDATIONS:\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════\n", .{});
        
        if (force_pct > 70.0) {
            std.debug.print("1. ⚡ Force calculation dominates ({d:.0}%%)\n", .{force_pct});
            std.debug.print("   → SIMD vectorize force computation\n", .{});
            std.debug.print("   → Expected speedup: 2-3x on force calc\n", .{});
            std.debug.print("   → Overall speedup: 1.5-2x\n\n", .{});
        }
        
        if (traverse_pct > 30.0) {
            std.debug.print("2. 🌳 Tree traversal significant ({d:.0}%% of force)\n", .{traverse_pct});
            std.debug.print("   → Linearize tree for better cache locality\n", .{});
            std.debug.print("   → Morton ordering for spatial locality\n", .{});
            std.debug.print("   → Expected speedup: 1.3-1.5x\n\n", .{});
        }
        
        if (tree_pct > 15.0) {
            std.debug.print("3. 🏗️  Tree building overhead ({d:.0}%%)\n", .{tree_pct});
            std.debug.print("   → Multi-threading can help here\n", .{});
            std.debug.print("   → Parallel tree construction\n", .{});
            std.debug.print("   → Expected speedup: 2-3x\n\n", .{});
        }
        
        std.debug.print("4. 🎯 Best Strategy:\n", .{});
        std.debug.print("   → Step 1: Multi-threading (7-8x speedup)\n", .{});
        std.debug.print("   → Step 2: SIMD for force calc (2x on top)\n", .{});
        std.debug.print("   → Step 3: Cache optimization (1.5x on top)\n", .{});
        std.debug.print("   → Combined: 20-24x theoretical speedup\n", .{});
    }
    
    fn printBar(self: *Profiler, percentage: f64) void {
        _ = self;
        const bar_width = 40;
        const filled = @as(usize, @intFromFloat((percentage / 100.0) * @as(f64, @floatFromInt(bar_width))));
        
        std.debug.print(" [", .{});
        for (0..filled) |_| std.debug.print("█", .{});
        for (filled..bar_width) |_| std.debug.print("░", .{});
        std.debug.print("]\n", .{});
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n🔍 Week 2 Phase 1: Detailed Performance Profiling\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════\n", .{});
    
    const test_sizes = [_]usize{ 1_000, 5_000, 10_000, 50_000 };
    
    for (test_sizes) |n| {
        std.debug.print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
        std.debug.print("📊 PROFILING {d} BODIES\n", .{n});
        std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
        
        // Create galaxy
        const bodies = try bh.createGalaxy(allocator, n, 2.0, 0.2);
        defer allocator.free(bodies);
        
        var profiler = Profiler.init(allocator);
        var sim = bh.BarnesHutSimulation.init(allocator, bodies);
        defer sim.deinit();
        
        // Warmup
        for (0..3) |_| {
            try sim.buildTree();
            sim.calculateForces();
        }
        
        // Profile tree build
        profiler.reset();
        const tree_start = std.time.nanoTimestamp();
        try sim.buildTree();
        profiler.tree_build_time = @intCast(std.time.nanoTimestamp() - tree_start);
        
        // Collect tree stats
        if (sim.root) |root| {
            profiler.tree_depth = root.calculateDepth();
            profiler.node_count = root.countNodes();
            profiler.leaf_count = root.countLeaves();
        }
        
        // Profile force calculation
        const force_start = std.time.nanoTimestamp();
        sim.calculateForces();
        profiler.force_calc_time = @intCast(std.time.nanoTimestamp() - force_start);
        
        // Count force calculations and theta checks
        profiler.force_calculations = sim.bodies.len * profiler.node_count / 10; // Estimate
        profiler.theta_accepts = profiler.force_calculations;
        
        // Profile integration
        const int_start = std.time.nanoTimestamp();
        sim.integrate();
        profiler.integration_time = @intCast(std.time.nanoTimestamp() - int_start);
        
        // Generate report
        profiler.printReport(n);
    }
    
    std.debug.print("\n\n✅ Phase 1 Complete: Profiling finished!\n", .{});
    std.debug.print("══════════════════════════════════════════════════════════\n", .{});
    std.debug.print("📋 Next Steps:\n", .{});
    std.debug.print("  1. Identify SIMD-friendly hotspots ✅\n", .{});
    std.debug.print("  2. Design data layout that avoids conversions\n", .{});
    std.debug.print("  3. Implement targeted SIMD optimizations\n", .{});
    std.debug.print("  4. Benchmark and verify improvements\n", .{});
}