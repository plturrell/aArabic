# Test Suite Quick Reference Index

**Location:** `/home/ubuntu/aArabic/src/serviceCore/nOpenaiServer/inference/engine/tests`

## 📁 Directory Organization

```
tests/
├── gpu/                    [GPU Testing & Validation]
│   ├── diagnostics/       → Hardware detection, CUDA verification
│   ├── benchmarks/        → Performance benchmarking
│   └── production/        → Production load & scaling tests
│
├── loaders/               [Model Loading]
├── performance/           [Subsystem Benchmarks]
├── integration/           [Component Integration]
├── smoke/                 [Quick Validation]
├── daily_tests/           [Historical Tests]
└── utils/                 [Testing Utilities]
```

## 🎯 Quick Access by Purpose

### "I need to validate GPU integration"
```bash
cd tests/gpu/diagnostics
zig test test_gpu_diagnostics.zig
zig test test_build_config.zig
```

### "I need to benchmark performance"
```bash
cd tests/gpu/benchmarks
zig test benchmark_gpu_vs_cpu.zig
```

### "I need production capacity estimates"
```bash
cd tests/gpu/production
zig test test_production_load.zig           # Concurrency & load
zig test test_quantization_comparison.zig   # Quant analysis
zig test test_model_format_comparison.zig   # SafeTensor vs GGUF
```

### "I need a validation report"
```bash
cd /home/ubuntu/aArabic
./scripts/gpu/generate_validation_report.sh
```

## 📊 Test Files by Category

### GPU Tests (6 files)

| File | Category | Purpose | Key Metrics |
|------|----------|---------|-------------|
| `gpu/diagnostics/test_gpu_diagnostics.zig` | Diagnostic | Detect GPU, query properties | GPU model, VRAM, compute capability |
| `gpu/diagnostics/test_build_config.zig` | Diagnostic | Verify CUDA linkage | Library versions, symbol resolution |
| `gpu/benchmarks/benchmark_gpu_vs_cpu.zig` | Benchmark | CPU vs GPU performance | Speedup, GFLOPS, bandwidth |
| `gpu/production/test_production_load.zig` | Production | Concurrency & scaling | Throughput, latency dist, utilization |
| `gpu/production/test_quantization_comparison.zig` | Production | Quantization analysis | All quant levels, efficiency |
| `gpu/production/test_model_format_comparison.zig` | Production | Format comparison | SafeTensor vs GGUF metrics |

### Utilities (3 files)

| File | Purpose | Usage |
|------|---------|-------|
| `utils/industry_baselines.zig` | Reference data | T4/A100 specs, llama.cpp benchmarks |
| `utils/report_generator.zig` | Report generation | Generate MD/JSON/CSV reports |
| `utils/generate_report.zig` | CLI tool | `generate_report results.json` |

### Legacy Tests (30+ files in other directories)

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `loaders/` | 5 files | Model format loading tests |
| `performance/` | 3 files | Subsystem benchmarks |
| `integration/` | 4 files | Component integration |
| `smoke/` | 3 files | Quick validation |
| `daily_tests/` | 20+ files | Day-by-day development tests |

## 🚀 Common Workflows

### Workflow 1: Initial GPU Validation
```bash
cd /home/ubuntu/aArabic/src/serviceCore/nOpenaiServer/inference/engine

# Step 1: Verify GPU hardware
zig test tests/gpu/diagnostics/test_gpu_diagnostics.zig

# Step 2: Verify build configuration
zig test tests/gpu/diagnostics/test_build_config.zig

# Step 3: Run basic benchmark
zig test tests/gpu/benchmarks/benchmark_gpu_vs_cpu.zig
```

### Workflow 2: Production Capacity Planning
```bash
# Run all production tests
zig test tests/gpu/production/test_production_load.zig
zig test tests/gpu/production/test_quantization_comparison.zig
zig test tests/gpu/production/test_model_format_comparison.zig

# Generate comprehensive report
zig run tests/utils/generate_report.zig -- results.json
```

### Workflow 3: Automated Full Validation
```bash
# One command to run everything
./scripts/gpu/generate_validation_report.sh --format=all

# Outputs:
# - gpu_validation_reports/gpu_validation_report_*.md
# - gpu_validation_reports/test_results_*.json
# - gpu_validation_reports/*.log
```

## 📈 Metrics Summary

### Production Tests Capture

**Concurrency Metrics:**
- Throughput scaling (1→64 concurrent)
- Latency distribution (P50/P95/P99)
- GPU utilization patterns
- Queue saturation point
- Stability coefficient

**Quantization Metrics:**
- All formats: Q4_0 → FP32
- VRAM efficiency (tokens/sec/GB)
- Compression ratios
- Batch scaling (1/4/8/16/32)
- Optimal format recommendation

**Format Comparison:**
- SafeTensor vs GGUF loading time
- Disk size vs VRAM usage
- Runtime performance
- Multi-model capacity
- Model switching speed

## 🔍 Finding Specific Tests

```bash
# List all GPU tests
find tests/gpu -name "*.zig" -type f

# List all production tests
ls tests/gpu/production/

# Search for specific functionality
grep -r "concurrency" tests/gpu/

# Find tests by metric
grep -r "tokens_per_sec" tests/
```

## 📝 Notes

- **All production tests export JSON** for report aggregation
- **No mock data** - measurements only (with industry baselines for reference)
- **Modular design** - Each test can run standalone or as part of suite
- **Organized by purpose** - Easy to find relevant tests

## 🆘 Quick Troubleshooting

| Issue | Check | Location |
|-------|-------|----------|
| GPU not detected | Run diagnostics | `tests/gpu/diagnostics/test_gpu_diagnostics.zig` |
| Low performance | Run benchmark | `tests/gpu/benchmarks/benchmark_gpu_vs_cpu.zig` |
| Build issues | Verify config | `tests/gpu/diagnostics/test_build_config.zig` |
| Need capacity info | Run production tests | `tests/gpu/production/test_production_load.zig` |

---

**Quick Navigation:** [README.md](./README.md) • [GPU Validation Docs](../../../../docs/GPU_VALIDATION_REPORT_SYSTEM.md)  
**Last Updated:** January 21, 2026
