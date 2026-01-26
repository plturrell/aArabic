# Zig SDK Performance Benchmark Suite

This directory contains benchmarks to validate the performance claims made in the SDK documentation.

## 📊 Benchmark Categories

### 1. Array Operations (`array_operations.zig`)
- Array sum (1M elements)
- Array multiply-add (1M elements)
- Array sorting (100K elements)

### 2. String Processing (`string_processing.zig`)
- String concatenation (100K operations)
- String search (1MB text)
- String to integer parsing (100K operations)

### 3. Computation (`computation.zig`)
- Fibonacci recursive (n=35)
- Prime number sieve (up to 1M)
- Matrix multiplication (100x100)
- Hash computation (1M operations)

### 4. Performance Profiler (`performance_profiler.zig`) ⭐ NEW
- Comprehensive optimization analysis
- Build mode detection and reporting
- LTO verification
- CPU-intensive benchmarks (Fibonacci 38)
- Memory-intensive benchmarks (Sort 500K)
- Mixed workload analysis
- Performance recommendations

## 🚀 Running Benchmarks

### Quick Start

```bash
# Run all benchmarks with Debug mode (baseline)
./run_benchmarks.sh debug

# Run all benchmarks with ReleaseSafe mode
./run_benchmarks.sh releasesafe

# Run all benchmarks with ReleaseFast mode
./run_benchmarks.sh releasefast

# Compare all modes
./run_benchmarks.sh compare
```

### Manual Execution

```bash
# Build benchmarks
zig build -Doptimize=Debug
zig build -Doptimize=ReleaseSafe

# Run individual benchmarks
./zig-out/bin/array_operations
./zig-out/bin/string_processing
./zig-out/bin/computation
```

## 📈 Interpreting Results

Each benchmark reports:
- **Median time**: Middle value of all runs (most stable)
- **Mean time**: Average of all runs
- **Iterations**: Number of test runs

### Expected Performance Differences

| Mode | Speed | Safety | Use Case |
|------|-------|--------|----------|
| **Debug** | 1.0x (baseline) | ✅ Full | Development |
| **ReleaseSafe** | 2-3x faster | ✅ Full | **Production (Default)** |
| **ReleaseFast** | 3-4x faster | ❌ None | Performance-critical |

## 🎯 Validation Goals

These benchmarks validate:
1. ✅ ReleaseSafe is significantly faster than Debug
2. ✅ LTO provides additional speedup
3. ✅ Safety checks don't severely impact performance
4. ✅ SDK modifications deliver real benefits

## 📝 Adding New Benchmarks

1. Create new file: `benchmarks/your_benchmark.zig`
2. Import framework: `const framework = @import("framework");`
3. Use `framework.benchmark()` to measure code
4. Add to `build.zig` following existing patterns
5. Update this README

## 🔧 Benchmark Framework

The framework (`framework.zig`) provides:
- Timing infrastructure with warmup
- Statistical analysis (median, mean, min, max)
- Consistent output formatting
- Memory leak detection (via GPA)

## ⚠️ Important Notes

- Benchmarks are synthetic and may not reflect real-world performance
- Results vary by hardware, OS, and system load
- Run multiple times and look at median values
- Disable turbo boost/power saving for consistent results
- Close other applications during benchmarking

## 📊 Benchmark Methodology

1. **Warmup Phase**: Run several iterations to warm caches
2. **Measurement Phase**: Time actual iterations
3. **Statistical Analysis**: Calculate median/mean/min/max
4. **Repeat**: Multiple runs for stability

## 🎓 Learning Resources

- [Zig Build System](https://ziglang.org/documentation/master/#Build-System)
- [Zig Optimization Modes](https://ziglang.org/documentation/master/#Build-Mode)
- [LTO Documentation](https://llvm.org/docs/LinkTimeOptimization.html)

## 📄 License

Same as parent project: MIT License
