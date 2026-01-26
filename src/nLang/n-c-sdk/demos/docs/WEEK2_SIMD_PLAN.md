# 🚀 Week 2: SIMD Vectorization Plan

**Goal:** 50K bodies @ 20 FPS (5x speedup from SIMD)  
**Current:** 10K bodies @ 19 FPS (baseline)  
**Strategy:** Vectorize force calculations to process 8 bodies simultaneously

---

## 📊 Current Performance Bottleneck

From Week 1 profiling:
```
Force Calculation: 39.79 ms (86.8% of frame time)
├─ Per body: 3.98 microseconds
├─ Operations: ~130K (N log N)
└─ Bottleneck: Scalar processing (1 body at a time)

Target with SIMD:
├─ 8x parallelism (AVX2/NEON)
├─ Expected: 39.79 ms → 8 ms
└─ Result: 45 ms → 13 ms total (77 FPS @ 10K!)
```

---

## 🎯 Week 2 Milestones

### Phase 1: Data Layout Transformation (Days 1-2)
- [ ] Convert from AoS (Array of Structures) to SoA (Structure of Arrays)
- [ ] Align data to cache lines (64 bytes)
- [ ] Verify correctness after transformation

### Phase 2: SIMD Force Calculation (Days 3-4)
- [ ] Implement 8-wide force calculation using @Vector
- [ ] Vectorize distance calculations
- [ ] Vectorize gravitational force computation
- [ ] Handle remainder bodies (< 8)

### Phase 3: Tree Traversal Optimization (Day 5)
- [ ] Vectorize theta criterion checks
- [ ] SIMD-friendly center-of-mass access
- [ ] Prefetch next nodes

### Phase 4: Testing & Validation (Days 6-7)
- [ ] Verify energy conservation maintained
- [ ] Benchmark: scalar vs SIMD
- [ ] Test with 50K bodies
- [ ] Profile cache behavior

---

## 🔬 Technical Approach

### 1. Structure of Arrays (SoA) Layout

**Current (AoS):**
```zig
const Body = struct {
    position: Vec3,    // x, y, z
    velocity: Vec3,    // vx, vy, vz
    acceleration: Vec3,
    mass: f64,
    id: u32,
};
bodies: []Body  // All data interleaved
```

**New (SoA):**
```zig
const BodiesSOA = struct {
    // Position components separate
    pos_x: []f64 align(64),
    pos_y: []f64 align(64),
    pos_z: []f64 align(64),
    
    // Velocity components separate
    vel_x: []f64 align(64),
    vel_y: []f64 align(64),
    vel_z: []f64 align(64),
    
    // Acceleration
    acc_x: []f64 align(64),
    acc_y: []f64 align(64),
    acc_z: []f64 align(64),
    
    mass: []f64 align(64),
    count: usize,
};
```

**Benefits:**
- ✅ SIMD loads 8 consecutive values
- ✅ Better cache utilization
- ✅ No structure padding waste

---

### 2. Vectorized Force Calculation

```zig
const Vec8f64 = @Vector(8, f64);

fn calculateForce8(
    bodies: *BodiesSOA,
    start_idx: usize,
    node: *OctreeNode,
) struct { fx: Vec8f64, fy: Vec8f64, fz: Vec8f64 } {
    // Load 8 body positions at once
    const pos_x: Vec8f64 = bodies.pos_x[start_idx..][0..8].*;
    const pos_y: Vec8f64 = bodies.pos_y[start_idx..][0..8].*;
    const pos_z: Vec8f64 = bodies.pos_z[start_idx..][0..8].*;
    
    // Vector from bodies to node center of mass
    const dx = @as(Vec8f64, @splat(node.center_of_mass.x)) - pos_x;
    const dy = @as(Vec8f64, @splat(node.center_of_mass.y)) - pos_y;
    const dz = @as(Vec8f64, @splat(node.center_of_mass.z)) - pos_z;
    
    // Distance squared (8 at once!)
    const dist_sq = dx * dx + dy * dy + dz * dz;
    
    // Softening parameter
    const softening: Vec8f64 = @splat(0.01 * 0.01);
    const dist_soft = dist_sq + softening;
    
    // Force magnitude: G * m1 * m2 / r²
    const masses: Vec8f64 = bodies.mass[start_idx..][0..8].*;
    const node_mass: Vec8f64 = @splat(node.total_mass);
    const G: Vec8f64 = @splat(1.0);
    
    const force_mag = (G * masses * node_mass) / dist_soft;
    
    // Normalize and scale
    const dist = @sqrt(dist_sq);
    const fx = (dx / dist) * force_mag;
    const fy = (dy / dist) * force_mag;
    const fz = (dz / dist) * force_mag;
    
    return .{ .fx = fx, .fy = fy, .fz = fz };
}
```

---

### 3. Vectorized Tree Traversal

```zig
fn calculateForcesVectorized(sim: *BarnesHutSimulation) void {
    const bodies = &sim.bodies_soa;
    const n = bodies.count;
    const vec_width = 8;
    
    // Process in chunks of 8
    var i: usize = 0;
    while (i + vec_width <= n) : (i += vec_width) {
        // Traverse tree for 8 bodies simultaneously
        const forces = traverseTree8(bodies, i, sim.root.?);
        
        // Store results
        const acc_x = forces.fx / bodies.mass[i..][0..8].*;
        const acc_y = forces.fy / bodies.mass[i..][0..8].*;
        const acc_z = forces.fz / bodies.mass[i..][0..8].*;
        
        @memcpy(bodies.acc_x[i..][0..8], &acc_x);
        @memcpy(bodies.acc_y[i..][0..8], &acc_y);
        @memcpy(bodies.acc_z[i..][0..8], &acc_z);
    }
    
    // Handle remainder (< 8 bodies)
    while (i < n) : (i += 1) {
        const force = sim.root.?.calculateForceScalar(bodies, i);
        bodies.acc_x[i] = force.x / bodies.mass[i];
        bodies.acc_y[i] = force.y / bodies.mass[i];
        bodies.acc_z[i] = force.z / bodies.mass[i];
    }
}
```

---

### 4. SIMD Theta Criterion

```zig
fn checkTheta8(
    bodies: *BodiesSOA,
    start_idx: usize,
    node: *OctreeNode,
    theta: f64,
) Vec8bool {
    // Distance to node
    const pos_x: Vec8f64 = bodies.pos_x[start_idx..][0..8].*;
    const pos_y: Vec8f64 = bodies.pos_y[start_idx..][0..8].*;
    const pos_z: Vec8f64 = bodies.pos_z[start_idx..][0..8].*;
    
    const dx = @as(Vec8f64, @splat(node.center_of_mass.x)) - pos_x;
    const dy = @as(Vec8f64, @splat(node.center_of_mass.y)) - pos_y;
    const dz = @as(Vec8f64, @splat(node.center_of_mass.z)) - pos_z;
    
    const dist = @sqrt(dx * dx + dy * dy + dz * dz);
    
    // s/d < theta?
    const size: Vec8f64 = @splat(node.size);
    const theta_vec: Vec8f64 = @splat(theta);
    const ratio = size / dist;
    
    return ratio < theta_vec;  // 8 boolean results!
}
```

---

## 📈 Expected Performance Gains

### Theoretical Speedup:
```
Operation        Scalar    SIMD    Speedup
──────────────────────────────────────────
Load positions   8 loads   1 load   8x
Distance calc    8 muls    1 mul    8x
Force magnitude  8 divs    1 div    8x
Total throughput 1 body/op 8 body/op 8x
```

### Real-World Expectations:
```
Ideal:           8x speedup
Memory bound:    ~5x speedup (realistic)
Cache misses:    ~4x speedup (conservative)

Current:  39.79 ms for 10K bodies
Target:    8.00 ms for 10K bodies (5x)
Result:   13.00 ms total frame (77 FPS!)
```

### Scaling Projections:
```
Bodies   Scalar    SIMD      FPS
─────────────────────────────────
10K      45 ms    13 ms     77
20K      90 ms    26 ms     38
50K     225 ms    65 ms     15 ⚠️
100K    450 ms   130 ms      8 ❌

Conclusion: Need threading for 50K+ (Week 3)
```

---

## 🛠️ Implementation Steps

### Step 1: Create SoA Data Structure
```zig
// File: barnes_hut_simd.zig

pub const BodiesSOA = struct {
    allocator: std.mem.Allocator,
    
    // Aligned arrays for SIMD
    pos_x: []f64,
    pos_y: []f64,
    pos_z: []f64,
    vel_x: []f64,
    vel_y: []f64,
    vel_z: []f64,
    acc_x: []f64,
    acc_y: []f64,
    acc_z: []f64,
    mass: []f64,
    id: []u32,
    
    count: usize,
    capacity: usize,
    
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !BodiesSOA {
        return BodiesSOA{
            .allocator = allocator,
            .pos_x = try allocator.alignedAlloc(f64, 64, capacity),
            .pos_y = try allocator.alignedAlloc(f64, 64, capacity),
            .pos_z = try allocator.alignedAlloc(f64, 64, capacity),
            .vel_x = try allocator.alignedAlloc(f64, 64, capacity),
            .vel_y = try allocator.alignedAlloc(f64, 64, capacity),
            .vel_z = try allocator.alignedAlloc(f64, 64, capacity),
            .acc_x = try allocator.alignedAlloc(f64, 64, capacity),
            .acc_y = try allocator.alignedAlloc(f64, 64, capacity),
            .acc_z = try allocator.alignedAlloc(f64, 64, capacity),
            .mass = try allocator.alignedAlloc(f64, 64, capacity),
            .id = try allocator.alignedAlloc(u32, 64, capacity),
            .count = 0,
            .capacity = capacity,
        };
    }
    
    pub fn deinit(self: *BodiesSOA) void {
        self.allocator.free(self.pos_x);
        self.allocator.free(self.pos_y);
        self.allocator.free(self.pos_z);
        self.allocator.free(self.vel_x);
        self.allocator.free(self.vel_y);
        self.allocator.free(self.vel_z);
        self.allocator.free(self.acc_x);
        self.allocator.free(self.acc_y);
        self.allocator.free(self.acc_z);
        self.allocator.free(self.mass);
        self.allocator.free(self.id);
    }
};
```

### Step 2: Convert AoS to SoA
```zig
pub fn convertToSOA(bodies: []Body, allocator: std.mem.Allocator) !BodiesSOA {
    var soa = try BodiesSOA.init(allocator, bodies.len);
    
    for (bodies, 0..) |body, i| {
        soa.pos_x[i] = body.position.x;
        soa.pos_y[i] = body.position.y;
        soa.pos_z[i] = body.position.z;
        soa.vel_x[i] = body.velocity.x;
        soa.vel_y[i] = body.velocity.y;
        soa.vel_z[i] = body.velocity.z;
        soa.mass[i] = body.mass;
        soa.id[i] = body.id;
    }
    
    soa.count = bodies.len;
    return soa;
}
```

### Step 3: SIMD Force Kernel
```zig
const Vec8f64 = @Vector(8, f64);

pub fn calculateForce8SIMD(
    soa: *BodiesSOA,
    idx: usize,
    com_x: f64,
    com_y: f64,
    com_z: f64,
    total_mass: f64,
    G: f64,
) struct { fx: Vec8f64, fy: Vec8f64, fz: Vec8f64 } {
    // Load 8 positions
    const px: Vec8f64 = soa.pos_x[idx..][0..8].*;
    const py: Vec8f64 = soa.pos_y[idx..][0..8].*;
    const pz: Vec8f64 = soa.pos_z[idx..][0..8].*;
    
    // Displacement vectors
    const dx = @as(Vec8f64, @splat(com_x)) - px;
    const dy = @as(Vec8f64, @splat(com_y)) - py;
    const dz = @as(Vec8f64, @splat(com_z)) - pz;
    
    // Distance calculation
    const dist_sq = dx * dx + dy * dy + dz * dz;
    const softening: Vec8f64 = @splat(0.0001);
    const dist_soft_sq = dist_sq + softening;
    const dist = @sqrt(dist_soft_sq);
    
    // Force magnitude
    const masses: Vec8f64 = soa.mass[idx..][0..8].*;
    const M: Vec8f64 = @splat(total_mass);
    const g: Vec8f64 = @splat(G);
    
    const force_mag = (g * masses * M) / dist_soft_sq;
    
    // Force components
    const fx = (dx / dist) * force_mag;
    const fy = (dy / dist) * force_mag;
    const fz = (dz / dist) * force_mag;
    
    return .{ .fx = fx, .fy = fy, .fz = fz };
}
```

---

## 🧪 Validation Strategy

### 1. Correctness Tests
```zig
test "SIMD matches scalar" {
    // Create small test case
    var bodies = try createTestBodies(allocator, 16);
    defer allocator.free(bodies);
    
    // Calculate forces scalar
    var forces_scalar = try calculateForcesScalar(bodies);
    defer allocator.free(forces_scalar);
    
    // Calculate forces SIMD
    var forces_simd = try calculateForcesSIMD(bodies);
    defer allocator.free(forces_simd);
    
    // Compare results
    for (forces_scalar, forces_simd) |scalar, simd| {
        try testing.expectApproxEqAbs(scalar.x, simd.x, 1e-10);
        try testing.expectApproxEqAbs(scalar.y, simd.y, 1e-10);
        try testing.expectApproxEqAbs(scalar.z, simd.z, 1e-10);
    }
}
```

### 2. Energy Conservation
```zig
test "SIMD preserves energy" {
    var sim = try createSimulation(allocator, 1000);
    defer sim.deinit();
    
    const initial_energy = sim.calculateEnergy();
    
    // Run 100 steps
    for (0..100) |_| {
        try sim.stepSIMD();
    }
    
    const final_energy = sim.calculateEnergy();
    const drift = @abs(final_energy - initial_energy) / initial_energy;
    
    try testing.expect(drift < 0.01); // < 1% drift
}
```

### 3. Performance Benchmarks
```zig
fn benchmarkSIMD(allocator: std.mem.Allocator) !void {
    const body_counts = [_]usize{ 1000, 5000, 10000, 50000 };
    
    for (body_counts) |n| {
        var sim = try createSimulation(allocator, n);
        defer sim.deinit();
        
        // Warmup
        for (0..10) |_| try sim.stepSIMD();
        
        // Benchmark
        const start = std.time.nanoTimestamp();
        for (0..100) |_| try sim.stepSIMD();
        const end = std.time.nanoTimestamp();
        
        const elapsed_ms = @as(f64, @floatFromInt(end - start)) / 1e6;
        const avg_frame = elapsed_ms / 100.0;
        const fps = 1000.0 / avg_frame;
        
        std.debug.print("{d:6} bodies: {d:6.2} ms/frame ({d:5.1} FPS)\n", 
            .{ n, avg_frame, fps });
    }
}
```

---

## 📋 Week 2 Checklist

### Day 1-2: Foundation
- [ ] Create `barnes_hut_simd.zig` with SoA structures
- [ ] Implement AoS → SoA conversion
- [ ] Test data layout correctness
- [ ] Benchmark memory access patterns

### Day 3-4: SIMD Implementation
- [ ] Implement 8-wide force calculation
- [ ] Vectorize distance computations
- [ ] Handle edge cases (remainder bodies)
- [ ] Test correctness vs scalar

### Day 5: Optimization
- [ ] Profile cache behavior
- [ ] Add prefetching hints
- [ ] Optimize tree traversal
- [ ] Tune theta criterion

### Day 6-7: Testing & Polish
- [ ] Run full test suite
- [ ] Verify energy conservation
- [ ] Benchmark at 10K, 20K, 50K bodies
- [ ] Create `galaxy_demo_v2` with SIMD
- [ ] Document performance gains

---

## 🎯 Success Criteria

### Must Have:
- ✅ 5x speedup on force calculation
- ✅ Energy drift < 1%
- ✅ 50K bodies at 15+ FPS
- ✅ No correctness regressions

### Nice to Have:
- ✅ 6x+ speedup (better than expected)
- ✅ 50K bodies at 20 FPS
- ✅ Cache-optimized traversal
- ✅ Detailed performance analysis

---

## 📊 Expected Final Results

```
Week 1 Baseline (Scalar):
├─ 10K bodies: 45 ms (19 FPS)
├─ 50K bodies: 225 ms (4 FPS) ❌
└─ Algorithm: O(N log N) but scalar

Week 2 Target (SIMD):
├─ 10K bodies: 13 ms (77 FPS) ✅
├─ 50K bodies: 65 ms (15 FPS) ✅
└─ Speedup: 5x on force calculation

Week 3 Preview (SIMD + Threading):
├─ 50K bodies: 8 ms (125 FPS) ⭐
├─ 100K bodies: 16 ms (62 FPS)
└─ 500K bodies: 80 ms (12 FPS)
```

---

## 🚀 Let's Build It!

Ready to implement Week 2 SIMD optimization. This will be the biggest single performance gain in the entire project!

**Next step:** Create `barnes_hut_simd.zig` with SoA layout and vectorized force calculations.