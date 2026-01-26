# Integrated Architecture: Petri Net + Multi-Threading + SIMD

## 🎯 Vision: Orchestrated High-Performance Galaxy Simulation

Combining three powerful techniques for maximum performance:
1. **Petri Net Orchestration** - Workflow management, thread pool, work distribution
2. **Multi-Threading** - Parallel execution across CPU cores
3. **SIMD Vectorization** - Process 4 operations simultaneously per core

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  PETRI NET ORCHESTRATION                    │
│         (nAgentFlow ExecutionStrategy.concurrent)           │
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐           │
│  │ Thread   │────▶│  Work    │────▶│ Results  │           │
│  │  Pool    │     │  Queue   │     │  Ready   │           │
│  │ (8 tok)  │     │ (N tok)  │     │ (0→N)    │           │
│  └──────────┘     └──────────┘     └──────────┘           │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴──────────┐
    │                   │
┌───▼──────────┐  ┌────▼──────────────┐
│ THREAD LAYER │  │   SIMD LAYER      │
│              │  │                   │
│ Parallel     │  │ Vector operations │
│ execution    │  │ (4 nodes at once) │
│ 7-8x speedup │  │ 2-3x speedup      │
└──────┬───────┘  └────┬──────────────┘
       │               │
       └───────┬───────┘
               │
    ┌──────────▼──────────┐
    │  Barnes-Hut Tree    │
    │    O(N log N)       │
    └─────────────────────┘
```

## 🏗️ Three-Layer Design

### Layer 1: Petri Net Orchestration

**Purpose**: High-level workflow control, thread pool management, work distribution

**Components**:
```zig
const GalaxySimulationNet = struct {
    net: PetriNet,
    thread_pool: ThreadPool,  // Persistent, managed by Petri net
    
    // Places
    places: struct {
        bodies_initialized: *Place,      // Start state
        tree_built: *Place,              // Tree ready
        work_queue: *Place,              // Body chunks waiting
        thread_available: [8]*Place,     // Per-thread availability
        forces_computed: *Place,         // All forces done
        frame_complete: *Place,          // Ready for next frame
    },
    
    // Transitions
    transitions: struct {
        build_tree: *Transition,         // Serial tree construction
        distribute_work: *Transition,    // Split into chunks
        compute_forces: *Transition,     // Parallel force calc (concurrent!)
        collect_results: *Transition,    // Gather all results
        integrate: *Transition,          // Update positions
    },
};
```

**Key Features**:
- **Persistent Thread Pool**: Threads created once, reused via token flow
- **Work Stealing**: Threads grab tokens from work_queue place
- **Automatic Synchronization**: Barrier built into Petri net semantics
- **Deadlock Detection**: Built-in via `isDeadlocked()`

### Layer 2: Multi-Threading Execution

**Purpose**: Parallel execution of force calculations across CPU cores

**Thread Pool Design**:
```zig
const PetriNetThreadPool = struct {
    petri_net: *PetriNet,
    threads: []Thread,
    work_contexts: []WorkContext,
    
    pub fn init(allocator: Allocator, net: *PetriNet, count: usize) !*Self {
        const threads = try allocator.alloc(Thread, count);
        
        // Initialize thread places in Petri net
        for (0..count) |i| {
            const place_id = try std.fmt.allocPrint(allocator, "thread_{d}", .{i});
            _ = try net.addPlace(place_id, "Thread Ready", 1);
            _ = try net.addTokenToPlace(place_id, &[_]u8{1}); // Mark ready
        }
        
        // Spawn persistent worker threads
        for (threads, 0..) |*thread, i| {
            thread.* = try Thread.spawn(.{}, workerThread, .{self, i});
        }
        
        return self;
    }
    
    fn workerThread(pool: *PetriNetThreadPool, thread_id: usize) void {
        while (true) {
            // Wait for work token
            const work = pool.getWork(thread_id) orelse {
                std.time.sleep(1_000_000); // 1ms
                continue;
            };
            
            // Process work (with SIMD if applicable)
            pool.processWork(work, thread_id);
            
            // Return result token
            pool.returnResult(work.result, thread_id);
        }
    }
    
    fn getWork(pool: *PetriNetThreadPool, thread_id: usize) ?WorkContext {
        const thread_place = try std.fmt.allocPrint(pool.allocator, "thread_{d}", .{thread_id});
        defer pool.allocator.free(thread_place);
        
        // Check if thread is available AND work exists
        if (pool.petri_net.isTransitionEnabled("assign_work")) {
            // Fire transition: thread_ready + work_queue → thread_working
            pool.petri_net.fireTransition("assign_work") catch return null;
            return pool.dequeueWork();
        }
        
        return null;
    }
};
```

**Benefits**:
- **Zero spawn overhead** (threads persist)
- **Work stealing** (via Petri net token flow)
- **Dynamic load balancing** (automatic via Petri net)

### Layer 3: SIMD Acceleration

**Purpose**: Vectorize force calculations for 2-3x speedup

**SIMD Strategy** (from the plan):
```zig
const SIMDNodeGroup = struct {
    // Process 4 nodes simultaneously
    centers_x: @Vector(4, f32),
    centers_y: @Vector(4, f32),
    centers_z: @Vector(4, f32),
    masses: @Vector(4, f32),
    sizes: @Vector(4, f32),
    
    // SIMD-accelerated distance check
    pub fn checkTheta(self: *SIMDNodeGroup, body_pos: Vec3, theta: f32) u4 {
        const body_x = @Vector(4, f32){@floatCast(body_pos.x)} ** 4;
        const body_y = @Vector(4, f32){@floatCast(body_pos.y)} ** 4;
        const body_z = @Vector(4, f32){@floatCast(body_pos.z)} ** 4;
        
        const dx = self.centers_x - body_x;
        const dy = self.centers_y - body_y;
        const dz = self.centers_z - body_z;
        
        const dist_sq = dx*dx + dy*dy + dz*dz;
        const dist = @sqrt(dist_sq);
        
        const ratio = self.sizes / dist;
        const theta_vec = @Vector(4, f32){theta} ** 4;
        
        return @bitCast(ratio < theta_vec);  // 4-bit mask
    }
};
```

**Benefits**:
- **No AoS→SoA conversion** (major win!)
- **4 nodes processed per cycle** (2-4x speedup)
- **Works with existing tree structure**

## 🎨 Complete Integration Example

```zig
pub const IntegratedGalaxySimulation = struct {
    allocator: Allocator,
    petri_net: *PetriNet,
    thread_pool: *PetriNetThreadPool,
    bodies: []Body,
    tree: *BarnesHutTree,
    simd_groups: []SIMDNodeGroup,
    
    pub fn init(allocator: Allocator, num_bodies: usize) !*Self {
        // 1. Create Petri net
        var net = try PetriNet.init(allocator, "galaxy_simulation");
        try buildSimulationWorkflow(&net);
        
        // 2. Create thread pool (managed by Petri net)
        var pool = try PetriNetThreadPool.init(allocator, &net, 8);
        
        // 3. Allocate bodies
        const bodies = try allocator.alloc(Body, num_bodies);
        
        // 4. Create tree
        const bounds = Bounds.init(
            Vec3.init(-2000, -2000, -200),
            Vec3.init(2000, 2000, 200),
        );
        var tree = try BarnesHutTree.init(allocator, bounds, 0.5);
        
        return self;
    }
    
    pub fn step(self: *Self, dt: f64) !void {
        // Phase 1: Build tree (serial) - controlled by Petri net
        if (self.petri_net.isTransitionEnabled("build_tree")) {
            try self.tree.build(self.bodies);
            try self.petri_net.fireTransition("build_tree");
            
            // Group nodes for SIMD
            self.simd_groups = try self.tree.groupNodesForSIMD(self.allocator);
        }
        
        // Phase 2: Compute forces (parallel + SIMD) - orchestrated by Petri net
        if (self.petri_net.isTransitionEnabled("compute_forces")) {
            // Petri net's concurrent strategy fires this across all threads!
            try self.petri_net.fireTransition("compute_forces");
            // This internally calls thread_pool.dispatch() with SIMD work
        }
        
        // Phase 3: Integrate (parallel)
        if (self.petri_net.isTransitionEnabled("integrate")) {
            try self.integrateParallel(dt);
            try self.petri_net.fireTransition("integrate");
        }
        
        // Frame complete
        if (self.petri_net.isTransitionEnabled("complete_frame")) {
            try self.petri_net.fireTransition("complete_frame");
        }
    }
    
    fn integrateParallel(self: *Self, dt: f64) !void {
        // Petri net distributes integration work
        const chunk_size = 1000;
        var i: usize = 0;
        
        while (i < self.bodies.len) : (i += chunk_size) {
            const end = @min(i + chunk_size, self.bodies.len);
            
            // Add work token to Petri net
            try self.petri_net.addTokenToPlace("integrate_work", 
                &IntegrateWorkToken{
                    .start = i,
                    .end = end,
                    .dt = dt,
                });
        }
        
        // Petri net automatically distributes to threads
        // Wait for all_integrated place to be marked
        while (!self.petri_net.isPlaceMarked("all_integrated")) {
            std.time.sleep(100_000); // 0.1ms
        }
    }
};

fn buildSimulationWorkflow(net: *PetriNet) !void {
    // Create places
    _ = try net.addPlace("bodies_init", "Bodies Initialized", null);
    _ = try net.addPlace("tree_built", "Tree Built", null);
    _ = try net.addPlace("forces_ready", "Forces Computed", null);
    _ = try net.addPlace("frame_complete", "Frame Complete", null);
    
    // Create thread pool places (8 threads)
    for (0..8) |i| {
        const id = try std.fmt.allocPrint(net.allocator, "thread_{d}_ready", .{i});
        _ = try net.addPlace(id, "Thread Ready", 1);
    }
    
    // Create transitions with priorities
    _ = try net.addTransition("build_tree", "Build Tree", 0);
    _ = try net.addTransition("compute_forces", "Compute Forces", 1);
    _ = try net.addTransition("integrate", "Integrate", 2);
    _ = try net.addTransition("complete_frame", "Complete Frame", 3);
    
    // Connect arcs
    _ = try net.addArc("arc1", .input, 1, "bodies_init", "build_tree");
    _ = try net.addArc("arc2", .output, 1, "build_tree", "tree_built");
    _ = try net.addArc("arc3", .input, 1, "tree_built", "compute_forces");
    _ = try net.addArc("arc4", .output, 1, "compute_forces", "forces_ready");
    
    // Add initial token
    try net.addTokenToPlace("bodies_init", &[_]u8{1});
}
```

## 📊 Performance Comparison

### Manual Threading (Phase 2A Current)
```
Overhead per frame:
  Thread spawn: 0.66ms (11 threads × 0.06ms)
  Synchronization: 0.05ms
  Total: 0.71ms

50K bodies:
  Force calc: 100ms (with 7x speedup)
  Overhead: 0.71ms
  Total: 100.71ms (9.9 FPS)
```

### Petri Net Threading (Proposed)
```
Overhead per frame:
  Thread spawn: 0ms (persistent pool!)
  Token routing: 0.02ms (negligible)
  Synchronization: 0.03ms
  Total: 0.05ms

50K bodies:
  Force calc: 100ms (same 7x speedup)
  Overhead: 0.05ms
  Total: 100.05ms (10.0 FPS)

Benefit: +0.66ms per frame = +0.7% performance
```

### Petri Net + SIMD (Full Integration)
```
50K bodies:
  Force calc: 50ms (SIMD 2x on top of threading)
  Overhead: 0.05ms
  Total: 50.05ms (20.0 FPS) ✅✅

Total speedup: 739ms → 50ms = 14.8x! 🎉
```

## 🔧 Implementation Phases

### Phase 2A-Extended: Add Petri Net Thread Pool

**Files to Create**:
- `advanced/petri_net_threading.zig` - Petri-net-controlled threading
- `core/thread_pool_petri.zig` - Persistent thread pool with Petri net integration

**Expected Results**:
- Same 7-8x speedup as manual threading
- 0.66ms less overhead
- Better load balancing (work stealing)
- Visual debugging capability

### Phase 2B: Add SIMD Vectorization

**Files to Create**:
- `advanced/simd_force_calculator.zig` - SIMD node group processing
- `core/simd_node_grouping.zig` - Group tree nodes for SIMD

**Expected Results**:
- Additional 2-3x speedup
- Total: 14-21x speedup
- 50K bodies at 20-30 FPS

### Phase 2C: Cache Optimization

**Already works with architecture**:
- Petri net work distribution respects cache boundaries
- SIMD naturally cache-friendly (sequential access)
- Thread-local buffers reduce cache contention

## 💡 Why This Architecture is Superior

### 1. **Separation of Concerns**
```
Petri Net Layer:    "What to do" (orchestration)
Threading Layer:    "How to parallelize" (execution)
SIMD Layer:         "How to vectorize" (optimization)
```

### 2. **Composability**
- Can disable threading (set thread_count = 1)
- Can disable SIMD (use scalar fallback)
- Each layer independently testable

### 3. **Observability**
```zig
// Get real-time stats
const stats = petri_net.getStats();
std.debug.print("Active threads: {d}\n", .{stats.tokens_in_place("thread_working")});
std.debug.print("Queue depth: {d}\n", .{stats.tokens_in_place("work_queue")});
std.debug.print("Completed: {d}/{d}\n", .{
    stats.tokens_in_place("results_ready"),
    total_work_units,
});
```

### 4. **Fault Tolerance**
- Petri net can detect stuck transitions
- Can timeout stalled threads
- Can redistribute work from failed threads

### 5. **Production-Ready**
- Uses proven `nAgentFlow` executor
- Thread-safe (RwLock)
- Already used at scale in `serviceCore`

## 🎯 Expected Final Performance

```
50,000 Bodies Performance Timeline:

Week 1 Baseline:         739ms (1.4 FPS)   ❌

Week 2 Phase 1:          739ms (1.4 FPS)   (profiling only)

Week 2 Phase 2A:         ~105ms (9.5 FPS)  ✅ Multi-threading
  ├─ Threading: 7x
  └─ Overhead: 0.71ms

Week 2 Phase 2A+:        ~100ms (10.0 FPS) ✅✅ Petri net pool
  ├─ Threading: 7x
  ├─ Pool: persistent
  └─ Overhead: 0.05ms (reduced!)

Week 2 Phase 2B:         ~50ms (20.0 FPS)  ✅✅✅ + SIMD
  ├─ Threading: 7x
  ├─ SIMD: 2x
  └─ Total: 14x

Week 2 Phase 2C:         ~33ms (30.0 FPS)  ✅✅✅✅ + Cache
  ├─ Threading: 7x
  ├─ SIMD: 2x
  ├─ Cache: 1.5x
  └─ Total: 21x

FINAL TARGET: 24ms (41.7 FPS) 🎯 Original goal!
```

## 📋 Implementation Checklist

### Petri Net Integration (Days 3-4)
- [ ] Create `GalaxySimulationNet` structure
- [ ] Implement persistent thread pool
- [ ] Add work distribution via tokens
- [ ] Test work stealing behavior
- [ ] Measure overhead reduction

### SIMD Integration (Days 5-6)
- [ ] Create `SIMDNodeGroup` structure
- [ ] Implement 4-node vectorized processing
- [ ] Add hybrid SIMD/scalar decision logic
- [ ] Benchmark SIMD speedup
- [ ] Validate correctness (forces match)

### Cache Optimization (Day 7)
- [ ] Memory layout analysis
- [ ] Prefetching hints
- [ ] Thread affinity tuning
- [ ] Final benchmarks

### Validation (Day 8)
- [ ] 1000-frame stability test
- [ ] Energy conservation check
- [ ] Visual verification (galaxy formation)
- [ ] Performance regression tests

## 🚀 Quick Start

Once implemented, usage will be simple:

```zig
// Initialize once
var sim = try IntegratedGalaxySimulation.init(allocator, 50_000);
defer sim.deinit();

// Game loop
while (running) {
    try sim.step(dt);  // Petri net handles everything!
    
    // Optional: monitoring
    sim.printPerformanceStats();
}
```

The Petri net **automatically**:
- Distributes work to threads
- Decides SIMD vs scalar
- Handles synchronization
- Provides monitoring

---

**Status**: 📐 Architecture Design Complete  
**Next**: Implement Petri net thread pool integration  
**Timeline**: 6 days to complete all phases