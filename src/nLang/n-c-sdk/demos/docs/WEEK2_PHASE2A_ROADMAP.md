# Week 2 Phase 2A: Multi-Threading Implementation Roadmap

## 🎯 Mission: Achieve 7-8x Speedup Through Parallelization

Based on profiling data showing **83-95% of time in force calculation**, multi-threading is our highest-impact optimization.

## 📊 Baseline Performance (From Profiling)

| Bodies | Current Time | Current FPS | Target FPS | Speedup Needed |
|--------|-------------|-------------|------------|----------------|
| 1,000  | 1.73ms | 578.4 | ✅ Already fast | - |
| 5,000  | 37.07ms | 27.0 | 60+ | 2.2x |
| 10,000 | 84.88ms | 11.8 | 60+ | 5.1x |
| 50,000 | 739.49ms | 1.4 | 30+ | 21.4x |

## 🏗️ Implementation Strategy

### Phase 2A Goals (Days 3-4)
1. **Primary**: Achieve 4-8x speedup through multi-threading
2. **Validate**: No data races, energy conservation, correct results
3. **Measure**: Thread scaling efficiency across core counts

### Three-Pronged Approach

#### Strategy 1: Simple Spatial Partitioning ✅ IMPLEMENTED
```zig
// Divide bodies equally among threads
const work_per_thread = bodies.len / thread_count;

// Each thread:
//   1. Processes its slice of bodies
//   2. Reads shared tree (read-only, no locks)
//   3. Writes to thread-local force buffer
```

**Pros:**
- Simple to implement
- No synchronization overhead
- Good cache locality

**Cons:**
- May have load imbalance
- Tree building still serial

**Expected Speedup:** 4-6x on 8 cores

#### Strategy 2: Work Stealing (Dynamic Load Balancing)
```zig
const WorkStealingQueue = struct {
    tasks: []Task,
    next_task: Atomic(usize),
    
    pub fn getNextTask() ?*Task {
        // Atomically claim next task
        // Threads steal work when done
    }
};
```

**Pros:**
- Handles uneven work distribution
- Better CPU utilization
- Scales better with more cores

**Cons:**
- Slightly more overhead
- More complex implementation

**Expected Speedup:** 6-8x on 8 cores

#### Strategy 3: Hybrid (If Needed)
- Spatial partitioning for force calculation
- Work stealing for tree construction
- Best of both worlds

## 📐 Technical Implementation Details

### 1. Cache-Line Alignment (Prevents False Sharing)
```zig
const Body = struct {
    position: Vec3 align(64),  // Cache line boundary
    velocity: Vec3,
    acceleration: Vec3,
    mass: f64,
    id: u32,
    _padding: [4]u8 = undefined,  // Pad to exactly 64 bytes
};

comptime {
    if (@sizeOf(Body) != 64) {
        @compileError("Body must be exactly 64 bytes");
    }
}
```

**Why 64 bytes?**
- Modern CPUs use 64-byte cache lines
- Prevents false sharing between threads
- Each body update doesn't invalidate other threads' caches

### 2. Thread-Local Force Accumulation
```zig
// Each thread accumulates forces independently
thread_local_forces: []Vec3,

// No atomic operations needed during accumulation
forces[body_idx].add(computed_force);

// Only combine at the end (serial, but fast)
for (all_threads) |thread_forces| {
    bodies[i].apply(thread_forces[i]);
}
```

### 3. Read-Only Tree Sharing
```zig
// Build tree once (can be parallelized later)
tree.buildFromBodies(bodies);

// Multiple threads read simultaneously (safe!)
for (bodies_slice) |body| {
    tree.computeForce(body);  // Read-only traversal
}
```

**No locks needed** because:
- Tree is immutable during force calculation
- Each thread reads independently
- No write conflicts

### 4. Leapfrog Integration (Energy Conserving)
```zig
pub fn update(self: *Body, dt: f64) void {
    // v_{1/2} = v_0 + a_0 * dt/2
    self.velocity += self.acceleration * (dt * 0.5);
    
    // x_1 = x_0 + v_{1/2} * dt
    self.position += self.velocity * dt;
    
    // v_1 = v_{1/2} + a_1 * dt/2
    // (next acceleration computed in next frame)
}
```

**Better than Euler because:**
- 2nd order accurate (Euler is 1st order)
- Energy conserving (important for long simulations)
- Same computational cost as Euler

## 📈 Expected Performance Results

### Best Case (8x Linear Scaling)
```
10K bodies:  84.88ms → 10.6ms   (94.3 FPS) ✅
50K bodies:  739.49ms → 92.4ms  (10.8 FPS) ✅
100K bodies: ~1.5s → 187.5ms    (5.3 FPS)  ✅
```

### Realistic Case (6x Scaling)
```
10K bodies:  84.88ms → 14.1ms   (70.9 FPS) ✅
50K bodies:  739.49ms → 123.2ms (8.1 FPS)  ✅
100K bodies: ~1.5s → 250ms      (4.0 FPS)  ⚠️
```

### Conservative Case (4x Scaling)
```
10K bodies:  84.88ms → 21.2ms   (47.2 FPS) ✅
50K bodies:  739.49ms → 184.9ms (5.4 FPS)  ⚠️
100K bodies: ~1.5s → 375ms      (2.7 FPS)  ❌
```

## 🔧 Build & Test Commands

```bash
# Navigate to demos directory
cd src/nLang/n-c-sdk/demos

# Build with optimizations
zig build-exe -femit-bin=week2_mt advanced/week2_phase2a_multithread.zig -OReleaseFast

# Run benchmarks
./week2_mt

# Test with thread sanitizer (debug mode)
zig build-exe advanced/week2_phase2a_multithread.zig -fsanitize=thread
./week2_phase2a_multithread

# Profile with perf (Linux)
perf record -g ./week2_mt
perf report
```

## ✅ Success Criteria

### Must Have (Primary Goals)
- [ ] 10K bodies at 60+ FPS (5.1x speedup)
- [ ] 50K bodies at 10+ FPS (7.1x speedup)
- [ ] 4-8x speedup demonstrated with 8 cores
- [ ] No data races (verified with thread sanitizer)
- [ ] Energy conservation (<0.1% drift per frame)

### Should Have (Secondary Goals)
- [ ] Thread efficiency >75% (speedup/cores)
- [ ] Scales linearly up to 8 cores
- [ ] Load balancing within 10% across threads
- [ ] Visual verification (galaxy forms correctly)

### Nice to Have (Stretch Goals)
- [ ] 100K bodies at 5+ FPS
- [ ] Work stealing implementation
- [ ] Parallel tree construction
- [ ] Dynamic thread pool adjustment

## 📊 Profiling Points

### Before Multi-threading
```
Force Calculation: 704.49ms (95.3% of 739.49ms)
├─ Tree Traversal:  ? ms
├─ Theta Checks:    ? ms
└─ Force Compute:   ? ms
```

### After Multi-threading (Target)
```
Force Calculation: ~88ms (target 8x speedup)
├─ Thread Spawn:    <1ms
├─ Parallel Force:  ~85ms (split across 8 cores)
└─ Synchronization: <2ms
```

## 🚧 Known Challenges

### Challenge 1: Load Imbalance
**Problem**: Dense regions have more work than sparse regions  
**Solution**: 
- Morton ordering for spatial locality
- Work stealing for dynamic balancing
- Chunk size tuning (1000-10000 bodies/task)

### Challenge 2: Tree Construction Bottleneck
**Problem**: Tree building is still serial (4.7% of time)  
**Solution**:
- Acceptable for now (small percentage)
- Can parallelize in Phase 2C if needed
- Focus on force calculation first

### Challenge 3: Memory Bandwidth
**Problem**: 8 threads reading same tree could saturate memory  
**Solution**:
- Cache-line alignment helps
- Prefetching hints
- Consider thread-local trees if needed

## 🔄 Iteration Plan

### Day 3: Initial Implementation
- [x] Create cache-line aligned Body struct
- [x] Implement simple spatial partitioning
- [ ] Basic benchmarking framework
- [ ] Thread safety validation

### Day 4: Optimization & Validation
- [ ] Profile thread utilization
- [ ] Implement work stealing if needed
- [ ] Tune chunk sizes
- [ ] Run full benchmark suite

### Day 5: Analysis & Documentation
- [ ] Measure scaling efficiency
- [ ] Document bottlenecks
- [ ] Prepare for Phase 2B (SIMD)
- [ ] Update roadmap based on results

## 📋 Next Phase Decision Tree

```
If speedup >= 7x:
    ✅ Move to Phase 2B (SIMD)
    Target: Additional 2-3x → Total 14-21x
    
If speedup = 4-7x:
    ⚠️ Implement work stealing
    Re-measure, then move to Phase 2B
    
If speedup < 4x:
    ❌ Debug thread efficiency
    Check for false sharing
    Profile memory bandwidth
    May need Phase 2C (cache) first
```

## 🎯 End Goal

**Target Performance for 50K Bodies:**
- Current: 739.49ms (1.4 FPS) ❌
- After Phase 2A: 92-123ms (8-11 FPS) ✅ Real-time
- After Phase 2B: 31-41ms (24-32 FPS) ✅✅ Smooth
- After Phase 2C: 23-31ms (32-43 FPS) ✅✅✅ Butter smooth

**Timeline:**
- Phase 2A (Multi-threading): Days 3-4 ← **WE ARE HERE**
- Phase 2B (SIMD): Days 5-6
- Phase 2C (Cache): Day 7
- Final validation: Day 8

---

**Status**: 🚧 In Progress  
**Last Updated**: January 25, 2026  
**Next Milestone**: Complete simple partitioning implementation