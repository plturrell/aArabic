# 🚀 Zig Performance Demos

A collection of high-performance simulations and benchmarks showcasing Zig's exceptional capabilities, featuring a complete Barnes-Hut N-body galaxy simulation.

## 📁 Project Structure

```
demos/
├── core/              # Core simulation libraries
│   ├── barnes_hut_octree.zig    # Barnes-Hut algorithm
│   ├── barnes_hut_simd.zig      # SIMD optimizations
│   └── sdl_bindings.zig         # SDL2 graphics bindings
│
├── simple/            # Educational demos
│   ├── fractal_demo.zig         # Mandelbrot visualizer
│   ├── particle_physics_demo.zig # Basic physics
│   └── visual_particle_demo.zig  # Particle visuals
│
├── advanced/          # Advanced simulations
│   ├── galaxy_demo_v1.zig       # Galaxy simulation
│   ├── particle_physics_mt.zig  # Multi-threaded physics
│   └── visual_particle_demo_complete.zig # Full particle system
│
├── benchmarks/        # Performance testing
│   ├── benchmark_simd.zig       # SIMD benchmarks
│   ├── benchmark_suite.zig      # Full suite
│   ├── profiler.zig             # Detailed profiler
│   └── test_stdin.zig           # I/O testing
│
├── dashboards/        # UI dashboards
│   ├── demo_dashboard.zig       # Demo launcher
│   └── main_dashboard.zig       # Control panel
│
└── docs/              # Documentation
    ├── GALAXY_SIMULATION_ROADMAP.md
    ├── WEEK1_COMPLETION_REPORT.md
    ├── WEEK2_LESSONS_LEARNED.md
    ├── WEEK2_PROFILING_ANALYSIS.md
    └── WEEK2_SIMD_PLAN.md
```

## 🌌 Featured: Galaxy Simulation

Our flagship demo - a real-time N-body galaxy simulation using the Barnes-Hut algorithm.

### Current Performance (50,000 bodies)
- **Current**: 739ms/frame (1.4 FPS)
- **Week 1 Goal**: ✅ Working implementation with Barnes-Hut O(N log N)
- **Week 2 Target**: 24ms/frame (41 FPS) - 30x speedup

### Key Features
- ✅ Barnes-Hut octree algorithm (O(N log N) instead of O(N²))
- ✅ Detailed profiling and analysis tools
- 🎯 Multi-threading optimization (in progress)
- 🎯 SIMD vectorization (planned)
- 🎯 Cache optimization (planned)

**Quick Start:**
```bash
cd src/nLang/n-c-sdk/demos
# Build from root with all dependencies
zig build-exe -femit-bin=galaxy_demo_v1 advanced/galaxy_demo_v1.zig -OReleaseFast
```

## 📊 Demos by Category

### 🎓 Simple Demos (Learning)

#### 1. Fractal Visualizer
**Location:** `simple/fractal_demo.zig`

Real-time Mandelbrot set computation:
- 800×600 pixel resolution
- Infinite zoom capability
- ASCII art rendering

```bash
zig build-exe -femit-bin=fractal_demo simple/fractal_demo.zig -OReleaseFast
```

#### 2. Basic Particle Physics
**Location:** `simple/particle_physics_demo.zig`

Simple N-body physics simulation:
- Real-time gravity
- Basic collision detection
- Performance metrics

```bash
zig build-exe -femit-bin=particle_physics_demo simple/particle_physics_demo.zig -OReleaseFast
```

### 🔬 Advanced Demos

#### 1. Galaxy Simulation v1
**Location:** `advanced/galaxy_demo_v1.zig`

Production-ready galaxy simulation:
- Barnes-Hut octree optimization
- 50,000+ body support
- Detailed performance analysis

**Dependencies:** `core/barnes_hut_octree.zig`

```bash
zig build-exe -femit-bin=galaxy_demo_v1 advanced/galaxy_demo_v1.zig -OReleaseFast
```

#### 2. Multi-Threaded Physics
**Location:** `advanced/particle_physics_mt.zig`

Parallel physics simulation:
- Thread pool implementation
- Lock-free algorithms
- 8-core optimization

```bash
zig build-exe -femit-bin=particle_physics_mt advanced/particle_physics_mt.zig -OReleaseFast
```

#### 3. Complete Particle System
**Location:** `advanced/visual_particle_demo_complete.zig`

Full-featured particle system with SDL2:
- Hardware-accelerated rendering
- Real-time interaction
- Visual effects

**Dependencies:** SDL2, `core/sdl_bindings.zig`

```bash
zig build-exe -femit-bin=visual_particle_demo_complete advanced/visual_particle_demo_complete.zig -lSDL2 -OReleaseFast
```

### ⚡ Benchmarks & Profiling

#### 1. Detailed Profiler
**Location:** `benchmarks/profiler.zig`

Comprehensive performance analysis:
- Tree build breakdown
- Force calculation timing
- Memory usage tracking
- Optimization recommendations

**Dependencies:** `core/barnes_hut_octree.zig`

```bash
zig build-exe -femit-bin=profiler benchmarks/profiler.zig -OReleaseFast
```

**Sample Output:**
```
📊 PROFILING 50000 BODIES
═══════════════════════════════════════════

⏱️  TIMING BREAKDOWN:
Tree Building:       34.83 ms (  4.7%)
Force Calc:         704.49 ms ( 95.3%)
Integration:          0.18 ms (  0.0%)
Total:              739.49 ms (1.4 FPS)

✨ OPTIMIZATION RECOMMENDATIONS:
1. ⚡ Force calculation dominates (95%)
   → Multi-threading: 7-8x speedup expected
   → SIMD: 2-3x additional speedup
   → Combined: 20-30x total speedup
```

#### 2. SIMD Benchmarks
**Location:** `benchmarks/benchmark_simd.zig`

SIMD performance testing:
- Vector vs scalar comparison
- Data layout analysis
- Conversion overhead measurement

**Dependencies:** `core/barnes_hut_simd.zig`

```bash
zig build-exe -femit-bin=benchmark_simd benchmarks/benchmark_simd.zig -OReleaseFast
```

#### 3. Full Benchmark Suite
**Location:** `benchmarks/benchmark_suite.zig`

Comprehensive benchmarks:
- Fibonacci computation
- Prime sieve
- Matrix multiplication
- Hash operations
- Memory allocation

```bash
zig build-exe -femit-bin=benchmark_suite benchmarks/benchmark_suite.zig -OReleaseFast
```

## 🏗️ Building

### Prerequisites
- **Zig 0.15.0+** (currently using 0.15.2)
- **SDL2** (optional, for visual demos)
- **Terminal** with ANSI color support

### Quick Build Commands

**Simple demos:**
```bash
cd src/nLang/n-c-sdk/demos
zig build-exe -femit-bin=fractal_demo simple/fractal_demo.zig -OReleaseFast
zig build-exe -femit-bin=particle_physics_demo simple/particle_physics_demo.zig -OReleaseFast
```

**Advanced demos:**
```bash
zig build-exe -femit-bin=galaxy_demo_v1 advanced/galaxy_demo_v1.zig -OReleaseFast
zig build-exe -femit-bin=particle_physics_mt advanced/particle_physics_mt.zig -OReleaseFast
```

**Benchmarks:**
```bash
zig build-exe -femit-bin=profiler benchmarks/profiler.zig -OReleaseFast
zig build-exe -femit-bin=benchmark_simd benchmarks/benchmark_simd.zig -OReleaseFast
```

**Optimization levels:**
- `-ODebug`: Debug mode (default)
- `-OReleaseSafe`: Optimized with safety checks
- `-OReleaseFast`: Maximum performance
- `-OReleaseSmall`: Optimized for size

## 📈 Performance Targets

### Galaxy Simulation Progress

| Phase | Target | Status | Performance |
|-------|--------|--------|-------------|
| Week 1: Basic Implementation | Working Barnes-Hut | ✅ Complete | 1.4 FPS (50K bodies) |
| Week 2 Phase 1: Profiling | Identify bottlenecks | ✅ Complete | Force calc = 95% |
| Week 2 Phase 2A: Multi-threading | 7-8x speedup | 🎯 In Progress | Target: 10.8 FPS |
| Week 2 Phase 2B: SIMD | 2-3x additional | 🎯 Planned | Target: 29.4 FPS |
| Week 2 Phase 2C: Cache | 1.3-1.5x final | 🎯 Planned | Target: 41.7 FPS |

### Benchmark Targets

- ✅ Fibonacci(40) < 1 second
- ✅ Prime sieve to 10M < 500ms
- ✅ Matrix multiply (500×500) < 200ms
- ✅ 10M hash operations < 100ms
- ✅ 100K allocations < 50ms

## 📚 Documentation

### Core Documentation
- **[Galaxy Simulation Roadmap](docs/GALAXY_SIMULATION_ROADMAP.md)** - Complete development plan
- **[Week 1 Report](docs/WEEK1_COMPLETION_REPORT.md)** - Initial implementation
- **[Week 2 Profiling Analysis](docs/WEEK2_PROFILING_ANALYSIS.md)** - Performance data

### Technical Guides
- **[SIMD Lessons Learned](docs/WEEK2_LESSONS_LEARNED.md)** - What works and what doesn't
- **[SIMD Plan](docs/WEEK2_SIMD_PLAN.md)** - Optimization strategy
- **[Reorganization Plan](docs/REORGANIZATION_PLAN.md)** - Project structure

## 💡 What These Demos Teach

### Performance Engineering
- Writing cache-friendly code
- SIMD vectorization techniques
- Multi-threading strategies
- Algorithm optimization (O(N²) → O(N log N))

### Zig-Specific Skills
- Zero-cost abstractions
- Compile-time code generation
- Manual memory management
- C interop with SDL2

### Real-World Applications
- N-body simulations (astronomy, molecular dynamics)
- Spatial data structures (octrees, quadtrees)
- High-performance computing
- Real-time systems

## 🎯 Quick Start Guide

**1. Try a simple demo:**
```bash
cd src/nLang/n-c-sdk/demos
zig build-exe -femit-bin=fractal_demo simple/fractal_demo.zig -OReleaseFast
./fractal_demo
```

**2. Run the galaxy simulation:**
```bash
zig build-exe -femit-bin=galaxy_demo_v1 advanced/galaxy_demo_v1.zig -OReleaseFast
./galaxy_demo_v1
```

**3. Profile performance:**
```bash
zig build-exe -femit-bin=profiler benchmarks/profiler.zig -OReleaseFast
./profiler
```

**4. Read the docs:**
```bash
cat docs/GALAXY_SIMULATION_ROADMAP.md
cat docs/WEEK2_PROFILING_ANALYSIS.md
```

## 🤝 Contributing

Want to add a demo or optimization?

1. Add your code to the appropriate directory
2. Follow the existing naming conventions
3. Include performance metrics
4. Document what it demonstrates
5. Update this README

## 📝 License

Same license as the parent project.

## 🎓 Learning Resources

- [Zig Language Reference](https://ziglang.org/documentation/master/)
- [Zig Standard Library](https://ziglang.org/documentation/master/std/)
- [Barnes-Hut Algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation)
- [SIMD Programming](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

---

**Made with ⚡ and 🚀 using Zig**

*Current Focus: Achieving 30x speedup on galaxy simulation through multi-threading, SIMD, and cache optimization.*