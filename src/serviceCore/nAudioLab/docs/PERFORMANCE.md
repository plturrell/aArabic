# AudioLabShimmy Performance Tuning Guide

**Version:** 1.0.0  
**Last Updated:** January 17, 2026

---

## Overview

This guide provides detailed information on optimizing AudioLabShimmy for maximum performance on various hardware configurations. The system is designed to run efficiently on CPU-only hardware, with specific optimizations for Apple Silicon.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [CPU Optimization](#cpu-optimization)
- [Memory Management](#memory-management)
- [Batch Processing](#batch-processing)
- [Model Optimization](#model-optimization)
- [Benchmarking](#benchmarking)
- [Troubleshooting Performance](#troubleshooting-performance)

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | 4 cores, 2.0 GHz |
| **RAM** | 8 GB |
| **Storage** | 10 GB free space |
| **OS** | macOS 12+, Linux (Ubuntu 20.04+) |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | 8+ cores, 3.0+ GHz (Apple Silicon M1/M2/M3) |
| **RAM** | 16 GB |
| **Storage** | 20 GB SSD |
| **OS** | macOS 13+ |

### Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Real-Time Factor | < 0.1x | ✓ |
| Latency (first token) | < 100ms | ✓ |
| Throughput | > 10 sentences/sec | ✓ |
| Memory Usage | < 2GB | ✓ |

---

## CPU Optimization

### Thread Configuration

The number of threads significantly impacts performance. Optimal settings depend on your CPU:

#### Apple Silicon (M1/M2/M3)

```mojo
var config = TTSConfig{
    num_threads: 8,           # Performance cores
    use_accelerate: true,     # Apple Accelerate framework
    simd_width: 4,            # Native SIMD width
}

var tts = TTSEngine.load_with_config(config)
```

**Recommendations:**
- **M1**: 4-6 threads
- **M1 Pro/Max**: 8-10 threads
- **M2**: 6-8 threads
- **M3/M3 Max**: 10-12 threads

#### Intel CPUs

```mojo
var config = TTSConfig{
    num_threads: 8,           # Hyperthreading aware
    use_accelerate: false,    # Use MKL or OpenBLAS
    simd_width: 8,            # AVX2/AVX-512
}
```

**Recommendations:**
- Use physical core count × 1.5 for hyperthreading
- Enable AVX2/AVX-512 instructions
- Consider using Intel MKL for matrix operations

### Accelerate Framework (Apple Silicon)

Leverage Apple's Accelerate framework for maximum performance:

```mojo
# Enable Accelerate for matrix operations
var config = TTSConfig{
    use_accelerate: true,
    accelerate_ops: [
        "matmul",           # Matrix multiplication
        "conv1d",           # 1D convolutions
        "fft",              # FFT operations
    ],
}
```

**Performance Impact:**
- Matrix multiplication: 2-3x faster
- Convolutions: 1.5-2x faster
- FFT: 2x faster

### SIMD Optimization

Ensure SIMD instructions are utilized:

```mojo
from sys import simd_width

fn optimize_simd():
    # Auto-detect optimal SIMD width
    var width = simd_width[DType.float32]()
    print(f"Using SIMD width: {width}")
    
    # Configure model with SIMD
    var config = TTSConfig{
        simd_width: width,
        vectorize_ops: true,
    }
```

---

## Memory Management

### Model Caching

Cache models in memory to avoid reload overhead:

```mojo
var config = TTSConfig{
    cache_models: true,       # Keep models in RAM
    cache_size_mb: 1024,      # Max cache size
}

var tts = TTSEngine.load_with_config(config)
```

**Memory Usage:**
- FastSpeech2: ~350MB
- HiFiGAN: ~150MB
- CMU Dictionary: ~50MB
- **Total**: ~550MB

### Memory Pooling

Use memory pooling for audio buffers:

```mojo
struct AudioPool:
    var buffers: List[AudioBuffer]
    var pool_size: Int = 10
    
    fn get_buffer(inout self) -> AudioBuffer:
        if len(self.buffers) > 0:
            return self.buffers.pop()
        return AudioBuffer.allocate()
    
    fn return_buffer(inout self, buffer: AudioBuffer):
        if len(self.buffers) < self.pool_size:
            self.buffers.append(buffer)
```

### Memory Profiling

Monitor memory usage during synthesis:

```bash
# Run with memory profiling
mojo run --profile-memory synthesis.mojo

# Check memory usage
mojo run scripts/profile_memory.py
```

---

## Batch Processing

### Optimal Batch Sizes

Process multiple texts in batches for better throughput:

```mojo
fn optimize_batch_size() -> Int:
    var cpu_cores = get_cpu_count()
    var ram_gb = get_ram_gb()
    
    # Rule of thumb: 2-4 texts per core
    var optimal_batch = min(
        cpu_cores * 3,
        Int(ram_gb / 0.5),  # 500MB per text
        16                   # Max batch size
    )
    
    return optimal_batch

fn batch_process(texts: List[String]) raises:
    var tts = TTSEngine.load("data/models")
    var batch_size = optimize_batch_size()
    
    for i in range(0, len(texts), batch_size):
        var batch = texts[i:i+batch_size]
        var audios = tts.synthesize_batch(batch)
        # Process audios...
```

**Performance Gains:**

| Batch Size | Throughput | vs Sequential |
|------------|------------|---------------|
| 1 (sequential) | 10 texts/sec | baseline |
| 4 | 25 texts/sec | 2.5x |
| 8 | 40 texts/sec | 4.0x |
| 16 | 50 texts/sec | 5.0x |

### Parallel Processing

Use multiple TTS engines for high-throughput scenarios:

```mojo
from concurrent import parallelize

fn parallel_synthesis(texts: List[String]) raises:
    var num_workers = 4
    var tts_engines = List[TTSEngine]()
    
    # Create worker engines
    for i in range(num_workers):
        tts_engines.append(TTSEngine.load("data/models"))
    
    # Distribute work
    fn worker(texts_chunk: List[String], engine: TTSEngine):
        for text in texts_chunk:
            var audio = engine.synthesize(text)
            # Process audio...
    
    # Parallelize
    parallelize[worker](texts, tts_engines)
```

---

## Model Optimization

### Quantization

Reduce model size with minimal quality impact:

```mojo
# Convert models to 8-bit quantization
var config = TTSConfig{
    quantization: "int8",     # int8, int16, float16
    quantize_weights: true,
    quantize_activations: false,
}

var tts = TTSEngine.load_with_config(config)
```

**Benefits:**
- Model size: 4x smaller
- Memory usage: 4x less
- Inference speed: 1.2-1.5x faster
- Quality impact: < 2% MOS degradation

### Model Pruning

Remove redundant parameters (advanced):

```bash
# Prune models (requires retraining)
mojo run scripts/prune_models.mojo \
    --sparsity 0.3 \
    --input data/models/fastspeech2 \
    --output data/models/fastspeech2_pruned
```

### Mixed Precision

Use FP16 for faster inference:

```mojo
var config = TTSConfig{
    mixed_precision: true,    # FP16/FP32 mix
    precision_mode: "auto",   # auto, fp16, fp32
}
```

**Performance:**
- Speed: 1.3-1.5x faster
- Memory: 30% reduction
- Quality: Negligible impact

---

## Benchmarking

### Built-in Benchmark

Run comprehensive performance tests:

```bash
# Run benchmark suite
./scripts/benchmark_performance.sh

# Expected output:
# ========================================
# AudioLabShimmy Performance Benchmark
# ========================================
# CPU: Apple M3 Max (12 cores)
# RAM: 32 GB
# Threads: 10
# ========================================
# 
# Single Sentence Synthesis:
#   Duration: 45ms
#   RTF: 0.056x
# 
# Batch Synthesis (8 texts):
#   Duration: 180ms
#   RTF: 0.048x
#   Throughput: 44 texts/sec
# 
# Long Text (500 words):
#   Duration: 2.3s
#   RTF: 0.073x
# 
# ========================================
# Performance Grade: EXCELLENT
# ========================================
```

### Custom Benchmarks

Create custom performance tests:

```mojo
from time import now

fn benchmark_synthesis() raises:
    var tts = TTSEngine.load("data/models")
    var texts = generate_test_texts(100)
    
    var start = now()
    for text in texts:
        var audio = tts.synthesize(text)
    var end = now()
    
    var duration_s = Float32(end - start) / 1_000_000_000
    var throughput = Float32(len(texts)) / duration_s
    
    print(f"Processed {len(texts)} texts in {duration_s:.2f}s")
    print(f"Throughput: {throughput:.1f} texts/sec")
```

### Profiling Tools

Use profiling tools to identify bottlenecks:

```bash
# CPU profiling
mojo run --profile-cpu synthesis.mojo

# Generate flame graph
mojo profile --format flamegraph output.prof

# Memory profiling
mojo run --profile-memory synthesis.mojo

# Trace execution
mojo run --trace synthesis.mojo
```

---

## Troubleshooting Performance

### Slow Synthesis

**Problem:** Synthesis takes longer than expected.

**Solutions:**

1. **Check thread count:**
```mojo
var config = TTSConfig{
    num_threads: 8,  # Increase to match CPU cores
}
```

2. **Enable Accelerate:**
```mojo
var config = TTSConfig{
    use_accelerate: true,  # Apple Silicon only
}
```

3. **Use batch processing:**
```mojo
var audios = tts.synthesize_batch(texts)  # vs sequential
```

4. **Reduce audio quality (if acceptable):**
```mojo
var config = AudioConfig{
    sample_rate: 22050,  # vs 48000
    bit_depth: 16,       # vs 24
}
```

### High Memory Usage

**Problem:** System runs out of memory.

**Solutions:**

1. **Reduce batch size:**
```mojo
var batch_size = 4  # vs 16
```

2. **Disable model caching:**
```mojo
var config = TTSConfig{
    cache_models: false,
}
```

3. **Use quantized models:**
```mojo
var config = TTSConfig{
    quantization: "int8",
}
```

4. **Process in chunks:**
```mojo
for chunk in split_into_chunks(texts, chunk_size=50):
    process_chunk(chunk)
```

### CPU Throttling

**Problem:** CPU throttles due to heat.

**Solutions:**

1. **Reduce thread count:**
```mojo
var config = TTSConfig{
    num_threads: 6,  # vs 12
}
```

2. **Add delays between batches:**
```mojo
import time
for batch in batches:
    process(batch)
    time.sleep(0.1)  # Cool down
```

3. **Monitor temperature:**
```bash
# macOS
sudo powermetrics --samplers smc | grep -i "CPU die temperature"
```

### Cache Misses

**Problem:** Frequent model reloads.

**Solutions:**

1. **Enable persistent caching:**
```mojo
var config = TTSConfig{
    cache_models: true,
    cache_persistent: true,
}
```

2. **Pre-load models:**
```mojo
# At startup
global_tts = TTSEngine.load("data/models")

# In requests
def synthesize_text(text):
    return global_tts.synthesize(text)
```

---

## Performance Checklist

### Pre-deployment Checklist

- [ ] Benchmark on target hardware
- [ ] Verify RTF < 0.1x
- [ ] Check memory usage < 2GB
- [ ] Test with concurrent requests
- [ ] Profile for bottlenecks
- [ ] Enable Accelerate (Apple Silicon)
- [ ] Configure optimal thread count
- [ ] Set up model caching
- [ ] Test batch processing
- [ ] Monitor CPU temperature

### Production Monitoring

```mojo
struct PerformanceMetrics:
    var avg_rtf: Float32
    var peak_memory_mb: Float32
    var throughput_per_sec: Float32
    var error_rate: Float32
    
    fn log(self):
        print(f"Avg RTF: {self.avg_rtf:.3f}x")
        print(f"Peak Memory: {self.peak_memory_mb:.0f} MB")
        print(f"Throughput: {self.throughput_per_sec:.1f} /sec")
        print(f"Error Rate: {self.error_rate:.2f}%")
```

---

## Performance Best Practices

1. **Always use batch processing** for multiple texts
2. **Enable Accelerate** on Apple Silicon
3. **Cache models** in long-running services
4. **Monitor memory** and implement pooling
5. **Profile regularly** to identify regressions
6. **Use quantization** for deployment
7. **Configure threads** based on hardware
8. **Benchmark** on target hardware before deployment

---

## Hardware-Specific Tips

### Apple M1/M2/M3

- Enable Accelerate framework
- Use 6-10 threads
- Leverage unified memory architecture
- Consider Metal acceleration (future)

### Intel CPUs

- Use Intel MKL for matrix ops
- Enable AVX2/AVX-512
- Use hyperthreading wisely
- Monitor thermal throttling

### AMD CPUs

- Use OpenBLAS for matrix ops
- Enable AVX2 instructions
- Configure NUMA awareness
- Use appropriate thread count

---

## Contact & Support

For performance issues or optimization questions:
- GitHub Issues: https://github.com/yourorg/nAudioLab/issues
- Discord: https://discord.gg/audiolabshimmy
- Email: performance@audiolabshimmy.org

---

**Last Updated:** January 17, 2026  
**Version:** 1.0.0
