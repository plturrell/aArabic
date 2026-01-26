# Demos Folder Reorganization Plan

## Current Issues
- 26+ files in root directory
- Mix of source files, executables, docs, and build artifacts
- Hard to navigate and find specific demos
- No clear separation between core libraries and examples

## Proposed Structure

```
demos/
├── README.md                          # Main entry point
├── build.zig                          # Build configuration
├── RUN_DEMO.sh                        # Quick launch script
│
├── core/                              # Core simulation libraries
│   ├── barnes_hut_octree.zig         # Barnes-Hut algorithm
│   ├── barnes_hut_simd.zig           # SIMD optimized version
│   └── sdl_bindings.zig              # SDL2 bindings
│
├── simple/                            # Simple educational demos
│   ├── fractal_demo.zig              # Basic fractal
│   ├── particle_physics_demo.zig     # Simple physics
│   └── visual_particle_demo.zig      # Basic particle visual
│
├── advanced/                          # Advanced simulations
│   ├── galaxy_demo_v1.zig            # Galaxy simulation v1
│   ├── particle_physics_mt.zig       # Multi-threaded physics
│   └── visual_particle_demo_complete.zig  # Complete particle system
│
├── benchmarks/                        # Performance testing
│   ├── benchmark_simd.zig            # SIMD benchmarks
│   ├── benchmark_suite.zig           # Full benchmark suite
│   ├── profiler.zig                  # Detailed profiler
│   └── test_stdin.zig                # Input testing
│
├── dashboards/                        # UI dashboards
│   ├── demo_dashboard.zig            # Demo selector
│   └── main_dashboard.zig            # Main control panel
│
├── docs/                              # Documentation
│   ├── GALAXY_SIMULATION_ROADMAP.md
│   ├── WEEK1_COMPLETION_REPORT.md
│   ├── WEEK2_LESSONS_LEARNED.md
│   ├── WEEK2_PROFILING_ANALYSIS.md
│   └── WEEK2_SIMD_PLAN.md
│
└── build/                             # Build outputs (gitignored)
    ├── .zig-cache/
    └── zig-out/
```

## File Categorization

### Core Libraries (→ core/)
- `barnes_hut_octree.zig` - Main algorithm
- `barnes_hut_simd.zig` - SIMD optimizations
- `sdl_bindings.zig` - Graphics bindings

### Simple Demos (→ simple/)
- `fractal_demo.zig` - Educational fractal
- `particle_physics_demo.zig` - Basic physics
- `visual_particle_demo.zig` - Simple visuals

### Advanced Demos (→ advanced/)
- `galaxy_demo_v1.zig` - Galaxy simulation
- `particle_physics_mt.zig` - Multi-threading example
- `visual_particle_demo_complete.zig` - Full-featured

### Benchmarks (→ benchmarks/)
- `benchmark_simd.zig` - SIMD performance
- `benchmark_suite.zig` - Comprehensive tests
- `profiler.zig` - Profiling tool
- `test_stdin.zig` - I/O testing

### Dashboards (→ dashboards/)
- `demo_dashboard.zig` - Demo launcher
- `main_dashboard.zig` - Control interface

### Documentation (→ docs/)
- All `.md` files except root README.md

### Build Artifacts (→ build/ or delete)
- Executables: `benchmark_simd`, `galaxy_demo_v1`, `main_dashboard`, etc.
- Cache: `.zig-cache/`, `zig-out/`

## Migration Strategy

1. Create new directory structure
2. Move files to appropriate directories
3. Update import paths in source files
4. Update build.zig to reflect new structure
5. Update README.md with new organization
6. Clean up build artifacts
7. Test that everything still compiles

## Benefits

✅ **Clear organization** - Easy to find specific demos
✅ **Separation of concerns** - Libraries vs examples vs docs
✅ **Scalability** - Easy to add new demos
✅ **Professional** - Industry-standard structure
✅ **Documentation** - Centralized in docs/
✅ **Build isolation** - Artifacts in build/

## Implementation

Ready to execute this reorganization?