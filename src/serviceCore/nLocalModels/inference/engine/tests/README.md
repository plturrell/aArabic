# Inference Engine Test Suite

Organized test suite for comprehensive validation of the inference engine.

## Directory Structure

```
tests/
├── gpu/                          # GPU-specific testing
│   ├── diagnostics/             # Hardware detection & validation
│   │   ├── test_gpu_diagnostics.zig      # GPU hardware detection
│   │   └── test_build_config.zig         # CUDA library verification
│   │
│   ├── benchmarks/              # Performance benchmarking
│   │   └── benchmark_gpu_vs_cpu.zig      # CPU vs GPU comparison
│   │
│   └── production/              # Production readiness tests
│       ├── test_production_load.zig      # Concurrency & load testing
│       ├── test_quantization_comparison.zig  # Quantization analysis
│       └── test_model_format_comparison.zig  # SafeTensor vs GGUF
│
├── loaders/                     # Model format loaders
│   ├── test_gguf_loader.zig            # GGUF format loading
│   ├── test_huggingface_loader.zig     # HuggingFace models
│   ├── test_hf_to_llama.zig            # Format conversion
│   ├── test_safetensors.zig            # SafeTensors format
│   └── test_safetensors_sharded.zig    # Sharded SafeTensors
│
├── performance/                 # Performance benchmarks
│   ├── benchmark_tiered_cache.zig      # Cache performance
│   ├── benchmark_day5                   # Day 5 benchmarks
│   └── benchmark_day5_light.zig        # Lightweight benchmarks
│
├── integration/                 # Integration tests
│   ├── test_bpe_tokenizer.zig          # Tokenizer integration
│   ├── test_memory_pool.zig            # Memory management
│   ├── test_config_parser.zig          # Configuration parsing
│   └── test_advanced_sampler.zig       # Sampling strategies
│
├── smoke/                       # Quick smoke tests
│   ├── test_lfm2_smoke.zig             # LFM2 smoke test
│   ├── test_all_models.zig             # All models quick test
│   └── test_multi_model_api.zig        # Multi-model API test
│
├── daily_tests/                 # Daily development tests
│   ├── test_day2.zig                   # Day 2 features
│   ├── test_day3.zig                   # Day 3 features
│   └── ... (day-by-day tests)
│
├── utils/                       # Utilities & reporting
│   ├── industry_baselines.zig          # Reference benchmarks
│   ├── report_generator.zig            # Report generation
│   └── generate_report.zig             # Report CLI tool
│
└── README.md                    # This file
```

## Test Categories

### 🔧 GPU Tests (`gpu/`)

#### Diagnostics
- **Purpose:** Verify GPU hardware and CUDA integration
- **Run when:** Initial setup, after CUDA updates, debugging GPU issues
- **Key tests:**
  - GPU detection and properties
  - CUDA library linkage
  - Driver compatibility

#### Benchmarks
- **Purpose:** Measure GPU vs CPU performance
- **Run when:** Performance validation, optimization verification
- **Key metrics:**
  - Matrix multiplication speedup
  - Memory bandwidth
  - GPU utilization

#### Production
- **Purpose:** Validate production readiness
- **Run when:** Before deployment, capacity planning
- **Key tests:**
  - Concurrency scaling (1→64 users)
  - Quantization format comparison
  - SafeTensor vs GGUF analysis
  - Load testing scenarios

### 📦 Loaders (`loaders/`)

- **Purpose:** Test model loading and format compatibility
- **Run when:** Adding new model formats, debugging load issues
- **Formats tested:**
  - GGUF (all quantization levels)
  - SafeTensors (single and sharded)
  - HuggingFace models
  - Format conversion utilities

### ⚡ Performance (`performance/`)

- **Purpose:** Benchmark specific subsystems
- **Run when:** Optimizing components, regression testing
- **Areas tested:**
  - Cache performance
  - Memory access patterns
  - Throughput under various conditions

### 🔗 Integration (`integration/`)

- **Purpose:** Test component interactions
- **Run when:** Continuous integration, feature integration
- **Components tested:**
  - Tokenizer integration
  - Memory pool management
  - Configuration parsing
  - Sampling strategies

### 💨 Smoke Tests (`smoke/`)

- **Purpose:** Quick validation that basic functionality works
- **Run when:** After builds, before detailed testing
- **Characteristics:**
  - Fast execution (< 30 seconds)
  - High-level validation
  - Pass/fail only

### 📅 Daily Tests (`daily_tests/`)

- **Purpose:** Historical development tests
- **Run when:** Debugging specific features, regression testing
- **Note:** Organized by development day

### 🛠️ Utils (`utils/`)

- **Purpose:** Supporting utilities for testing and reporting
- **Contents:**
  - Industry baseline data
  - Report generation system
  - Test result aggregation

## Running Tests

### Run Specific Category

```bash
cd /home/ubuntu/aArabic/src/serviceCore/nOpenaiServer/inference/engine

# GPU diagnostics
zig build test --test-filter "gpu/diagnostics"

# Production load tests
zig build test --test-filter "gpu/production"

# All loader tests
zig build test --test-filter "loaders"

# Smoke tests only
zig build test --test-filter "smoke"
```

### Run Individual Test

```bash
# Direct execution
zig test tests/gpu/diagnostics/test_gpu_diagnostics.zig

# Through build system
zig build test-gpu-diagnostics
```

### Generate Reports

```bash
# Run all GPU tests and generate report
./scripts/gpu/generate_validation_report.sh

# Generate report from test results
zig run tests/utils/generate_report.zig -- test_results.json
```

## Test Development Guidelines

### Naming Conventions

- **Test files:** `test_<feature>.zig`
- **Benchmark files:** `benchmark_<feature>.zig`
- **Utilities:** `<name>.zig` (no prefix)

### File Location

1. **GPU-specific tests:** Place in `gpu/` subdirectory
2. **Model format tests:** Place in `loaders/`
3. **Performance tests:** Place in `performance/`
4. **Quick validation:** Place in `smoke/`
5. **Component integration:** Place in `integration/`

### Test Structure

```zig
// File: tests/category/test_feature.zig
const std = @import("std");
const testing = std.testing;

// Test functions
test "feature: basic functionality" {
    // Test code
}

test "feature: edge cases" {
    // Test code
}

// Main for standalone execution
pub fn main() !void {
    // Standalone test runner
}
```

## Continuous Integration

### Pre-commit Tests
```bash
# Fast smoke tests
zig build test --test-filter "smoke"
```

### Pull Request Tests
```bash
# Comprehensive integration tests
zig build test --test-filter "integration"
zig build test --test-filter "loaders"
```

### Nightly Tests
```bash
# Full production validation
./scripts/gpu/generate_validation_report.sh
zig build test --test-filter "gpu/production"
```

## Metrics & Reporting

### Report Generation

All production tests export JSON results for aggregation:

```bash
# Run tests
zig build test-production-load        # → concurrency_results.json
zig build test-quantization-comparison # → quantization_comparison.json
zig build test-format-comparison      # → format_comparison.json

# Generate unified report
zig run tests/utils/generate_report.zig -- results.json --format=md

# Output: gpu_validation_report.md
```

### Key Metrics

**Performance:**
- Throughput (tokens/sec)
- Latency (P50, P95, P99)
- GPU utilization (%)

**Capacity:**
- Concurrent users supported
- Models per GPU
- Daily token capacity

**Efficiency:**
- Tokens/sec per GB VRAM
- Compression ratio
- Loading time

## Troubleshooting

### GPU Tests Failing

1. Check GPU detection: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Run diagnostics: `zig test tests/gpu/diagnostics/test_gpu_diagnostics.zig`

### Performance Issues

1. Check baseline: `zig test tests/gpu/benchmarks/benchmark_gpu_vs_cpu.zig`
2. Profile operations: Review GPU utilization metrics
3. Compare to industry baselines in `tests/utils/industry_baselines.zig`

### Test Organization Issues

- All tests follow the directory structure above
- If a test doesn't fit, consider creating a new category
- Keep related tests together for easier maintenance

## Contributing

When adding new tests:

1. Place in appropriate directory based on category
2. Follow naming conventions
3. Include both unit tests and main() for standalone execution
4. Export JSON results for production tests
5. Update this README if adding new categories

## References

- GPU validation: `docs/GPU_VALIDATION_REPORT_SYSTEM.md`
- Test automation: `scripts/gpu/generate_validation_report.sh`
- Industry baselines: `tests/utils/industry_baselines.zig`

---

**Last Updated:** January 21, 2026  
**Test Suite Version:** 1.0
