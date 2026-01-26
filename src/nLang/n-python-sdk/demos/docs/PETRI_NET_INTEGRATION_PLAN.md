# Petri Net Integration Plan for Galaxy Simulation

## 🎯 Using Your New zig-libc Petri Net Engine

Your `zig-libc` Petri net engine is **perfect** for this! Key features:
- ✅ **Thread-safe** (RwLock for concurrent access)
- ✅ **Production-grade** (100 functions, fully tested)
- ✅ **Event callbacks** (monitor thread state changes)
- ✅ **Trace support** (execution monitoring)
- ✅ **Multiple execution strategies** (sequential, concurrent, priority-based)

## 📐 Integration Architecture

```
┌──────────────────────────────────────────────────────┐
│         zig-libc Petri Net Engine                    │
│         (Thread-safe, Event-driven)                  │
│                                                       │
│  Places:                    Transitions:             │
│  • thread_pool (8 tokens)   • assign_work           │
│  • work_queue (N tokens)    • compute_forces        │
│  • results_ready (0→N)      • collect_results       │
└────────────────┬─────────────────────────────────────┘
                 │
      ┌──────────┴─────────┐
      │                    │
┌─────▼────────┐    ┌─────▼────────────┐
│   THREADS    │    │      SIMD        │
│   7-8x       │    │      2-3x        │
└──────────────┘    └──────────────────┘
```

## 🏗️ Implementation: Three Phases

### Phase 1: Monitoring Layer (Day 3) ⭐ START HERE
**Goal**: Add Petri net as observability layer without changing threading logic

**Benefits**:
- Zero risk (doesn't affect existing code)
- Real-time visibility into thread behavior
- Foundation for advanced features

**Implementation**:

```zig
// demos/advanced/petri_net_monitor.zig
const std = @import("std");
const petri = @import("zig_libc").petri;

pub const ThreadMonitor = struct {
    net: *petri.types.pn_net_t,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, thread_count: usize) !*ThreadMonitor {
        // Create Petri net
        const net = petri.core.pn_create("galaxy_monitor", petri.types.PN_CREATE_TRACED) 
            orelse return error.PetriNetCreationFailed;
        
        // Create places
        _ = petri.core.pn_place_create(net, "bodies_init", "Bodies Initialized");
        _ = petri.core.pn_place_create(net, "tree_built", "Tree Built");
        _ = petri.core.pn_place_create(net, "forces_ready", "Forces Ready");
        _ = petri.core.pn_place_create(net, "frame_complete", "Frame Complete");
        
        // Create thread availability places
        for (0..thread_count) |i| {
            const place_id = try std.fmt.allocPrintZ(allocator, "thread_{d}_ready", .{i});
            defer allocator.free(place_id);
            _ = petri.core.pn_place_create(net, place_id, place_id);
            
            // Add token = thread is ready
            const place = petri.core.pn_place_get(net, place_id);
            const token = petri.core.pn_token_create(null, 0);
            _ = petri.core.pn_token_put(place, token);
        }
        
        // Create transitions
        _ = petri.core.pn_trans_create(net, "build_tree", "Build Tree");
        _ = petri.core.pn_trans_create(net, "compute_forces", "Compute Forces");
        _ = petri.core.pn_trans_create(net, "integrate", "Integrate");
        
        // Connect arcs
        const arc1 = petri.core.pn_arc_create(net, "arc1", .input);
        _ = petri.core.pn_arc_connect(arc1, "bodies_init", "build_tree");
        
        const arc2 = petri.core.pn_arc_create(net, "arc2", .output);
        _ = petri.core.pn_arc_connect(arc2, "build_tree", "tree_built");
        
        // ... more arcs
        
        const monitor = try allocator.create(ThreadMonitor);
        monitor.* = .{
            .net = net,
            .allocator = allocator,
        };
        
        return monitor;
    }
    
    pub fn deinit(self: *ThreadMonitor) void {
        _ = petri.core.pn_destroy(self.net);
        self.allocator.destroy(self);
    }
    
    // Called when thread starts work
    pub fn onThreadStart(self: *ThreadMonitor, thread_id: usize) void {
        const place_id = std.fmt.allocPrintZ(self.allocator, "thread_{d}_ready", .{thread_id}) 
            catch return;
        defer self.allocator.free(place_id);
        
        // Remove token (thread is busy)
        const place = petri.core.pn_place_get(self.net, place_id);
        if (place) |p| {
            _ = petri.core.pn_token_get(p);
        }
    }
    
    // Called when thread completes work
    pub fn onThreadComplete(self: *ThreadMonitor, thread_id: usize) void {
        const place_id = std.fmt.allocPrintZ(self.allocator, "thread_{d}_ready", .{thread_id}) 
            catch return;
        defer self.allocator.free(place_id);
        
        // Add token (thread is ready again)
        const place = petri.core.pn_place_get(self.net, place_id);
        if (place) |p| {
            const token = petri.core.pn_token_create(null, 0);
            _ = petri.core.pn_token_put(p, token);
        }
    }
    
    // Get statistics
    pub fn getStats(self: *ThreadMonitor) !petri.types.pn_stats_t {
        var stats: petri.types.pn_stats_t = undefined;
        _ = petri.core.pn_stats(self.net, &stats);
        return stats;
    }
    
    // Export state as JSON
    pub fn exportState(self: *ThreadMonitor) ![]const u8 {
        // Use serialization module
        return petri.serialization.toJSON(self.allocator, self.net);
    }
};
```

**Usage in existing code**:

```zig
// In week2_phase2a_simple.zig
const ThreadMonitor = @import("petri_net_monitor.zig").ThreadMonitor;

pub fn main() !void {
    var monitor = try ThreadMonitor.init(allocator, 8);
    defer monitor.deinit();
    
    // Mark initial state
    monitor.onThreadStart(0); // Placeholder to mark activity
    
    // ... existing threading code ...
    
    // After frame
    const stats = try monitor.getStats();
    std.debug.print("Active threads: {d}/{d}\n", .{
        stats.place_count - getReadyThreadCount(monitor),
        stats.place_count,
    });
}
```

### Phase 2: Petri Net Work Distribution (Days 4-5)
**Goal**: Let Petri net control work assignment to threads

**Benefits**:
- Eliminates 0.66ms thread spawn overhead
- Work stealing via token flow
- Dynamic load balancing

**Implementation**:

```zig
// demos/advanced/petri_net_threading.zig
pub const PetriNetThreadPool = struct {
    net: *petri.types.pn_net_t,
    threads: []std.Thread,
    work_queue: std.ArrayList(WorkUnit),
    running: std.atomic.Value(bool),
    
    pub fn init(allocator: std.mem.Allocator, thread_count: usize) !*PetriNetThreadPool {
        const net = petri.core.pn_create("thread_pool", 0) 
            orelse return error.PetriNetCreationFailed;
        
        // Create work queue place
        const work_queue_place = petri.core.pn_place_create(net, "work_queue", "Work Queue");
        _ = petri.core.pn_place_set_capacity(work_queue_place, 10000); // Max work items
        
        // Create thread pool places
        for (0..thread_count) |i| {
            const place_id = try std.fmt.allocPrintZ(allocator, "thread_{d}", .{i});
            defer allocator.free(place_id);
            _ = petri.core.pn_place_create(net, place_id, place_id);
        }
        
        // Create transition: assign_work
        // Fires when: work_queue has token AND thread_X has token
        // Result: Remove tokens from both, thread becomes busy
        const assign_work = petri.core.pn_trans_create(net, "assign_work", "Assign Work");
        _ = petri.core.pn_trans_set_priority(assign_work, 10); // High priority
        
        // Spawn persistent worker threads
        var threads = try allocator.alloc(std.Thread, thread_count);
        for (threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{}, workerThread, .{self, i});
        }
        
        const pool = try allocator.create(PetriNetThreadPool);
        pool.* = .{
            .net = net,
            .threads = threads,
            .work_queue = std.ArrayList(WorkUnit).init(allocator),
            .running = std.atomic.Value(bool).init(true),
        };
        
        return pool;
    }
    
    fn workerThread(pool: *PetriNetThreadPool, thread_id: usize) void {
        while (pool.running.load(.acquire)) {
            // Try to grab work from Petri net
            const work = pool.tryGetWork(thread_id) orelse {
                std.time.sleep(1_000_000); // 1ms
                continue;
            };
            
            // Process work
            pool.processWork(work);
            
            // Mark thread as ready again (add token back)
            pool.returnThread(thread_id);
        }
    }
    
    fn tryGetWork(pool: *PetriNetThreadPool, thread_id: usize) ?WorkUnit {
        // Check if work available and thread is ready
        const place_id = std.fmt.allocPrintZ(pool.allocator, "thread_{d}", .{thread_id}) 
            catch return null;
        defer pool.allocator.free(place_id);
        
        const thread_place = petri.core.pn_place_get(pool.net, place_id) orelse return null;
        const work_place = petri.core.pn_place_get(pool.net, "work_queue") orelse return null;
        
        if (petri.core.pn_place_has_tokens(thread_place) == 1 and 
            petri.core.pn_place_has_tokens(work_place) == 1) {
            
            // Fire transition: assign work to this thread
            _ = petri.core.pn_trans_fire(pool.net, "assign_work");
            
            // Dequeue work
            return pool.work_queue.popOrNull();
        }
        
        return null;
    }
    
    pub fn submitWork(pool: *PetriNetThreadPool, work: WorkUnit) !void {
        // Add work to queue
        try pool.work_queue.append(work);
        
        // Add token to work_queue place
        const work_place = petri.core.pn_place_get(pool.net, "work_queue") orelse return error.PlaceNotFound;
        const token = petri.core.pn_token_create(null, 0) orelse return error.TokenCreationFailed;
        _ = petri.core.pn_token_put(work_place, token);
    }
};
```

### Phase 3: SIMD Integration (Days 6-7)
**Goal**: Petri net decides SIMD vs scalar based on node characteristics

**Benefits**:
- Automatic algorithm selection
- Optimal performance per data type
- Easy to tune thresholds

```zig
pub const SIMDDecisionNet = struct {
    net: *petri.types.pn_net_t,
    
    pub fn shouldUseSIMD(self: *SIMDDecisionNet, node: *OctreeNode) bool {
        // Add node info as token to "node_info" place
        const node_info_place = petri.core.pn_place_get(self.net, "node_info") orelse return false;
        
        const data = std.fmt.allocPrint(self.allocator, "{d}", .{node.body_count}) catch return false;
        defer self.allocator.free(data);
        
        const token = petri.core.pn_token_create(data.ptr, data.len) orelse return false;
        _ = petri.core.pn_token_put(node_info_place, token);
        
        // Fire transition: check_threshold
        // Guard function checks if body_count >= 8
        const check_trans = petri.core.pn_trans_get(self.net, "check_threshold");
        if (petri.core.pn_trans_is_enabled(check_trans) == 1) {
            _ = petri.core.pn_trans_fire(check_trans);
            
            // Check if token ended up in "use_simd" or "use_scalar" place
            const simd_place = petri.core.pn_place_get(self.net, "use_simd") orelse return false;
            return petri.core.pn_place_has_tokens(simd_place) == 1;
        }
        
        return false;
    }
};
```

## 📊 Expected Performance Improvements

### Phase 1: Monitoring (Day 3)
```
Performance: No change (monitoring only)
Benefits: Real-time thread visibility, execution traces
Overhead: <0.01ms per frame (negligible)
```

### Phase 2: Work Distribution (Days 4-5)
```
Before: 100.71ms (with 0.66ms thread spawn overhead)
After:  100.05ms (persistent thread pool)
Benefit: +0.66ms = +0.7% performance
```

### Phase 3: SIMD Integration (Days 6-7)
```
After SIMD: 50.05ms (2x additional speedup)
Total: 739ms → 50ms = 14.8x speedup!
```

## 🔧 Step-by-Step Implementation

### Day 3: Implement Monitoring
1. Create `petri_net_monitor.zig`
2. Integrate with `week2_phase2a_simple.zig`
3. Test: Verify Petri net captures thread events
4. Export state to JSON for visualization

### Days 4-5: Implement Work Distribution
1. Create `petri_net_threading.zig`
2. Implement persistent thread pool
3. Add work stealing via token flow
4. Benchmark: Should match 7-8x with 0.66ms less overhead

### Days 6-7: SIMD Integration
1. Create `simd_decision_net.zig`
2. Add guards for SIMD threshold (body_count >= 8)
3. Route work to SIMD or scalar handler
4. Benchmark: Expect 2-3x additional speedup

### Day 8: Validation
1. 1000-frame stability test
2. Verify no deadlocks (use `pn_is_deadlocked`)
3. Energy conservation check
4. Performance regression tests

## 🚀 Quick Start Example

```zig
// Simple integration example
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create Petri net monitor
    var monitor = try ThreadMonitor.init(allocator, 8);
    defer monitor.deinit();
    
    // Your existing simulation
    var sim = try IntegratedGalaxySimulation.init(allocator, 50_000);
    defer sim.deinit();
    
    // Simulation loop
    for (0..1000) |frame| {
        // Mark frame start
        monitor.onFrameStart(frame);
        
        // Run step (your existing code)
        try sim.step(dt);
        
        // Get Petri net stats
        const stats = try monitor.getStats();
        if (frame % 100 == 0) {
            std.debug.print("Frame {d}: Active threads: {d}\n", .{
                frame,
                countActiveThreads(&stats),
            });
        }
    }
    
    // Export execution trace
    const trace_json = try monitor.exportState();
    defer allocator.free(trace_json);
    
    // Save for visualization
    try std.fs.cwd().writeFile("execution_trace.json", trace_json);
}
```

## 💡 Key Advantages of This Approach

### 1. Incremental Adoption
- Phase 1: Monitor only (zero risk)
- Phase 2: Control threading (low risk)
- Phase 3: Add SIMD (high benefit)

### 2. Thread Safety Built-In
Your zig-libc engine has RwLock, so multiple threads can:
- Read state simultaneously (lockShared)
- Modify state safely (lock)
- No manual synchronization needed!

### 3. Observable System
```zig
// Real-time monitoring
const stats = petri.core.pn_stats(net);
std.debug.print("Deadlocked: {}\n", .{petri.core.pn_is_deadlocked(net) == 1});
std.debug.print("Tokens flowing: {d}\n", .{stats.total_tokens});
```

### 4. Event-Driven Architecture
```zig
// Set callback for transition fired
_ = petri.core.pn_callback_set(net, .transition_fired, onTransitionFired);

fn onTransitionFired(
    net: ?*petri.types.pn_net_t, 
    event: petri.types.pn_event_type_t, 
    ctx: ?*anyopaque
) void {
    std.debug.print("Transition fired!\n", .{});
}
```

### 5. Production-Ready
- 100 functions (fully implemented)
- Thread-safe (RwLock)
- Tested (thread safety tests included)
- Serialization support (JSON export)

## 📋 Implementation Checklist

### Phase 1: Monitoring (Day 3)
- [ ] Create ThreadMonitor structure
- [ ] Integrate onThreadStart/Complete callbacks
- [ ] Test with existing threading code
- [ ] Export state to JSON
- [ ] Verify overhead <0.01ms

### Phase 2: Work Distribution (Days 4-5)
- [ ] Create PetriNetThreadPool structure
- [ ] Implement persistent worker threads
- [ ] Add work queue with token-based assignment
- [ ] Test work stealing behavior
- [ ] Benchmark overhead reduction (0.66ms)

### Phase 3: SIMD Integration (Days 6-7)
- [ ] Create SIMDDecisionNet structure
- [ ] Add threshold guards (body_count >= 8)
- [ ] Route to SIMD or scalar handler
- [ ] Benchmark 2-3x speedup
- [ ] Validate correctness

### Validation (Day 8)
- [ ] 1000-frame stability test
- [ ] Deadlock detection verification
- [ ] Energy conservation check
- [ ] Performance regression tests
- [ ] Documentation and examples

---

**Status**: 📐 Implementation Plan Ready  
**Engine**: zig-libc Petri net (thread-safe, production-grade)  
**Timeline**: 6 days to full integration  
**Expected Result**: 14-21x speedup with robust orchestration