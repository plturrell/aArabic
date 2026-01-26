# 🌌 Barnes-Hut Galaxy Simulation - 4-Week Roadmap

## Project Goal
Create a real-time N-body gravity simulation capable of simulating **1 million bodies at 60 FPS** using the Barnes-Hut algorithm, demonstrating that Zig can compete with specialized scientific computing software and GPU implementations.

## Technical Approach

### Core Algorithm: Barnes-Hut Tree
- **Spatial partitioning:** Recursive octree for 3D space
- **Complexity:** O(N log N) instead of naive O(N²)
- **Far-field approximation:** Treat distant clusters as single masses
- **Theta criterion:** θ = 0.5 (adjustable for accuracy/speed tradeoff)

### Expected Performance
- **Naive O(N²):** 1M bodies = 1 trillion force calculations = impossible
- **Barnes-Hut:** 1M bodies × log₂(1M) ≈ 20M operations
- **With SIMD (8-wide):** 2.5M vector operations
- **At 5 GHz:** ~20 cycles per vector op → **fits in 16ms frame!**

---

## Week 1: Core Barnes-Hut Implementation (Jan 25 - Jan 31)

### Goals
- ✅ Implement octree data structure
- ✅ Build tree from particle positions
- ✅ Calculate center of mass for nodes
- ✅ Implement force calculation with theta criterion
- ✅ Target: 100K bodies at 30 FPS

### Deliverables
1. `barnes_hut_octree.zig` - Core octree structure
2. `gravity_simulation.zig` - Physics integration
3. `galaxy_demo_v1.zig` - Basic visualization
4. Performance comparison vs naive O(N²)

### Key Data Structures
```zig
const OctreeNode = struct {
    center: Vec3,           // Center of this cube
    size: f64,              // Width of cube
    center_of_mass: Vec3,   // Weighted average position
    total_mass: f64,        // Sum of all masses in subtree
    children: ?[8]*OctreeNode, // 8 octants (null if leaf)
    body: ?*Body,           // Single body if leaf node
};

const Body = struct {
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    mass: f64,
    id: u32,
};
```

### Success Metrics
- [x] Tree construction: <2ms for 100K bodies
- [ ] Force calculation: <30ms for 100K bodies
- [ ] Total frame time: <33ms (30 FPS)
- [ ] Visual verification: Particles cluster correctly

---

## Week 2: SIMD Optimization (Feb 1 - Feb 7)

### Goals
- [ ] Vectorize force calculations (AVX2/NEON)
- [ ] Optimize memory layout for cache efficiency
- [ ] Implement prefetching for tree traversal
- [ ] Target: 500K bodies at 30 FPS

### Deliverables
1. `simd_force_calculator.zig` - Vectorized operations
2. `aligned_allocators.zig` - Cache-line aligned memory
3. Performance profiling report
4. Comparison: scalar vs vectorized

### SIMD Optimizations
```zig
// Process 8 bodies simultaneously
const ForceAccumulator = struct {
    fx: @Vector(8, f64),
    fy: @Vector(8, f64),
    fz: @Vector(8, f64),
};

// Calculate force between 8 bodies and one node
fn calculateForce8(bodies: [8]Body, node: OctreeNode) ForceAccumulator;
```

### Success Metrics
- [ ] 4-8x speedup from vectorization
- [ ] Force calculation: <15ms for 500K bodies
- [ ] Memory bandwidth: >20 GB/s
- [ ] Cache miss rate: <5%

---

## Week 3: Multi-threading & Scaling (Feb 8 - Feb 14)

### Goals
- [ ] Parallel tree construction
- [ ] Thread-safe force calculation
- [ ] Work-stealing for load balancing
- [ ] Target: 1M bodies at 30 FPS

### Deliverables
1. `parallel_octree.zig` - Concurrent tree building
2. `work_stealing_scheduler.zig` - Load balancing
3. Scaling analysis (1-16 threads)
4. Comparison vs single-threaded

### Parallelization Strategy
```zig
// Phase 1: Parallel tree construction
// - Each thread builds subtree for its octant
// - Lock-free merge at boundaries

// Phase 2: Parallel force calculation
// - Partition bodies across threads
// - Each thread traverses full tree
// - No synchronization needed!

// Phase 3: Parallel integration
// - Independent velocity/position updates
```

### Success Metrics
- [ ] Linear scaling up to 8 threads
- [ ] Tree construction: <1ms for 1M bodies
- [ ] Force calculation: <10ms for 1M bodies
- [ ] Total frame time: <33ms (30 FPS)

---

## Week 4: Polish & Visualization (Feb 15 - Feb 21)

### Goals
- [ ] Beautiful SDL2 visualization
- [ ] Interactive controls (zoom, rotate, parameters)
- [ ] Real galaxy formation scenarios
- [ ] Target: 1M bodies at 60 FPS

### Deliverables
1. `galaxy_visualization.zig` - Advanced rendering
2. `galaxy_presets.zig` - Formation scenarios
3. `performance_dashboard.zig` - Real-time metrics
4. Demo video and documentation

### Visualization Features
- Real-time 3D rendering with SDL2
- Color-coded by velocity/density
- Particle trails for orbits
- Interactive camera controls
- Performance overlay with breakdown

### Galaxy Formation Scenarios
1. **Uniform Sphere Collapse** - Watch dark matter halo form
2. **Disk Galaxy** - Spiral arms emerge naturally
3. **Galaxy Merger** - Two galaxies colliding
4. **Cluster Formation** - Multiple galaxies interacting

### Success Metrics
- [ ] 1M bodies at 60 FPS sustained
- [ ] Frame time variance <10%
- [ ] All scenarios scientifically accurate
- [ ] Beautiful, publication-quality visuals

---

## Performance Targets Summary

| Week | Bodies | FPS | Frame Time | Operations/Frame |
|------|--------|-----|------------|------------------|
| 1    | 100K   | 30  | 33ms       | 2M              |
| 2    | 500K   | 30  | 33ms       | 10M             |
| 3    | 1M     | 30  | 33ms       | 20M             |
| 4    | 1M     | 60  | 16ms       | 20M             |

## Comparison Targets

### Academic Benchmarks
- **Dubinski (1996):** 1M bodies in 10 minutes on Cray T3D
- **Our target:** 1M bodies at 60 FPS (600x faster than Cray!)

### Commercial Software
- **Universe Sandbox 2:** ~100K bodies at 60 FPS (GPU)
- **Our target:** 10x more bodies on CPU

### GPU Implementations
- **CUDA N-body:** 1M bodies at ~200 FPS on RTX 4090
- **Our target:** Competitive performance on CPU (30% of GPU)

## Technical Innovations

1. **Cache-Oblivious Tree Traversal**
   - Morton ordering for spatial locality
   - Prefetch next nodes during computation

2. **Adaptive Theta**
   - Dynamic accuracy adjustment
   - Higher accuracy for close interactions
   - Lower accuracy for distant clusters

3. **Hierarchical Time Stepping**
   - Fast particles get small timesteps
   - Slow particles get large timesteps
   - Maintains accuracy while improving performance

4. **Lock-Free Parallel Tree**
   - No synchronization during force calculation
   - Atomic operations only for tree construction
   - Perfect scaling to many cores

## Success Criteria

### Technical Success
- ✅ 1M bodies at 60 FPS sustained
- ✅ O(N log N) complexity verified
- ✅ Multi-threaded scaling demonstrated
- ✅ SIMD utilization >90%

### Scientific Success
- ✅ Galaxy formation matches observations
- ✅ Energy conservation <1% error
- ✅ Angular momentum conserved
- ✅ Matches published simulations

### Impact Success
- ✅ Demonstrates Zig's HPC capabilities
- ✅ Shows CPU can compete with GPU
- ✅ Creates reusable scientific library
- ✅ Inspires further Zig adoption

---

## Getting Started (Week 1, Day 1)

### Today's Tasks
1. Create `barnes_hut_octree.zig` - Core data structure
2. Implement tree construction algorithm
3. Basic force calculation (no optimization yet)
4. Simple visualization with existing SDL2 code

### Expected Output
```
🌌 Barnes-Hut Galaxy Simulation v0.1
Bodies: 10,000
Tree depth: 13 levels
Tree build time: 0.5 ms
Force calc time: 15.2 ms
Total frame time: 25.8 ms (38.7 FPS)

✅ Proof of concept working!
```

Let's begin! 🚀