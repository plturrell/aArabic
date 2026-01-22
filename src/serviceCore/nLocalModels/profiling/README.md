# Performance Profiling Infrastructure

**Version**: 1.0  
**Last Updated**: January 23, 2026  
**Status**: Production Ready

## Overview

Integrated performance profiling tools for nLocalModels inference service, providing CPU profiling, memory analysis, flame graphs, bottleneck detection, and real-time dashboards.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Profiling Infrastructure                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ CPU Profiler │  │   Memory     │  │  GPU Monitor │     │
│  │  (Sampling)  │  │   Profiler   │  │  (CUDA/ROC)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                         │                                   │
│                ┌────────▼────────┐                          │
│                │  Data Collector │                          │
│                │   & Aggregator  │                          │
│                └────────┬────────┘                          │
│                         │                                   │
│         ┌───────────────┼───────────────┐                  │
│         │               │               │                  │
│  ┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐           │
│  │   Flame     │ │ Bottleneck │ │  Dashboard │           │
│  │   Graph     │ │  Detector  │ │  (Web UI)  │           │
│  │  Generator  │ │            │ │            │           │
│  └─────────────┘ └────────────┘ └────────────┘           │
│         │               │               │                  │
│         └───────────────┴───────────────┘                  │
│                         │                                   │
│                ┌────────▼────────┐                          │
│                │   HANA Cloud    │                          │
│                │   Persistence   │                          │
│                └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. CPU Profiler (`cpu_profiler.zig`)
- Statistical sampling profiler
- Call stack capture (backtrace)
- Function-level timing
- Thread-aware profiling
- Configurable sampling rate

### 2. Memory Profiler (`memory_profiler.zig`)
- Heap allocation tracking
- Memory leak detection
- Peak usage monitoring
- Allocation hotspots
- GC pressure analysis

### 3. GPU Monitor (`gpu_monitor.zig`)
- CUDA/ROCm metrics
- GPU utilization tracking
- Memory bandwidth monitoring
- Kernel execution timing
- Multi-GPU support

### 4. Flame Graph Generator (`flamegraph.zig`)
- SVG flame graph output
- Interactive HTML viewer
- Stack folding and aggregation
- Color-coded performance zones
- Zoom and filter capabilities

### 5. Bottleneck Detector (`bottleneck_detector.zig`)
- Automatic hotspot identification
- Critical path analysis
- Performance regression detection
- Alerting on thresholds
- Recommendation engine

### 6. Performance Dashboard (`dashboard/`)
- Real-time metrics visualization
- Historical trend analysis
- Comparative profiling
- Export capabilities
- Integration with HANA

### 7. Tokenizer Profiler (`tokenizer_profiler.zig`)
- Vocabulary lookup tracking
- BPE merge operation profiling
- Cache hit rate analysis
- Tokens per second metrics
- Automated optimization recommendations

### 8. Compression Profiler (`compression_profiler.zig`)
- DEFLATE compression benchmarking
- LZ4 fast compression profiling
- Zstandard performance analysis
- Algorithm comparison (Snappy, Brotli, GZIP)
- Compression level optimization
- Use-case specific recommendations
- Throughput and ratio metrics

## Quick Start

### Enable Profiling

```bash
# Start inference server with profiling
./start-zig.sh --profile --profile-rate=1000

# Or via environment variables
export NLOCAL_PROFILE_ENABLED=true
export NLOCAL_PROFILE_CPU=true
export NLOCAL_PROFILE_MEMORY=true
export NLOCAL_PROFILE_GPU=true
./start-zig.sh
```

### Collect Profiles

```bash
# CPU profile for 60 seconds
curl -X POST http://localhost:8080/admin/profile/cpu?duration=60

# Memory snapshot
curl -X POST http://localhost:8080/admin/profile/memory

# Generate flame graph
curl -X POST http://localhost:8080/admin/profile/flamegraph \
  -o flamegraph.svg

# Get bottleneck report
curl http://localhost:8080/admin/profile/bottlenecks
```

### View Dashboard

```bash
# Open performance dashboard
open http://localhost:8080/profiling/dashboard

# Or access specific views
open http://localhost:8080/profiling/cpu
open http://localhost:8080/profiling/memory
open http://localhost:8080/profiling/gpu
```

## Configuration

### `profiling/config.json`

```json
{
  "cpu": {
    "enabled": true,
    "sample_rate_hz": 1000,
    "max_stack_depth": 128,
    "capture_threads": true
  },
  "memory": {
    "enabled": true,
    "track_allocations": true,
    "sample_rate": 100,
    "leak_detection": true
  },
  "gpu": {
    "enabled": true,
    "poll_interval_ms": 100,
    "track_kernels": true
  },
  "flamegraph": {
    "min_width_percent": 0.1,
    "color_scheme": "hot",
    "interactive": true
  },
  "bottleneck": {
    "threshold_ms": 10,
    "auto_detect": true,
    "alert_on_regression": true
  },
  "storage": {
    "persist_to_hana": true,
    "retention_days": 30,
    "max_profiles": 1000
  }
}
```

## Usage Examples

### Example 1: Profile Inference Request

```bash
# Enable profiling for single request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Profile: true" \
  -d '{
    "model": "llama-3-70b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Get profile results
curl http://localhost:8080/admin/profile/last
```

### Example 2: Compare Before/After

```bash
# Profile baseline
curl -X POST http://localhost:8080/admin/profile/start \
  --data '{"name": "baseline"}'

# Run workload
./benchmark.sh

# Stop and save
curl -X POST http://localhost:8080/admin/profile/stop

# Make optimization changes
# ...

# Profile optimized
curl -X POST http://localhost:8080/admin/profile/start \
  --data '{"name": "optimized"}'

./benchmark.sh

curl -X POST http://localhost:8080/admin/profile/stop

# Compare results
curl http://localhost:8080/admin/profile/compare?a=baseline&b=optimized
```

### Example 3: Continuous Monitoring

```bash
# Enable continuous profiling
curl -X POST http://localhost:8080/admin/profile/continuous \
  --data '{
    "enabled": true,
    "interval_seconds": 300,
    "cpu": true,
    "memory": true,
    "gpu": true
  }'

# View live metrics
open http://localhost:8080/profiling/dashboard
```

## Integration with Existing System

### HANA Cloud Persistence

All profiling data is automatically stored in HANA Cloud:

```sql
-- CPU profile samples
CREATE TABLE PROFILING_CPU_SAMPLES (
    ID VARCHAR(36) PRIMARY KEY,
    TIMESTAMP TIMESTAMP NOT NULL,
    THREAD_ID INTEGER,
    FUNCTION_NAME VARCHAR(512),
    FILE_PATH VARCHAR(1024),
    LINE_NUMBER INTEGER,
    STACK_DEPTH INTEGER,
    SAMPLE_COUNT INTEGER
);

-- Memory allocations
CREATE TABLE PROFILING_MEMORY_ALLOCS (
    ID VARCHAR(36) PRIMARY KEY,
    TIMESTAMP TIMESTAMP NOT NULL,
    SIZE_BYTES BIGINT,
    ADDRESS VARCHAR(32),
    STACK_TRACE TEXT,
    FREED BOOLEAN DEFAULT FALSE,
    FREED_AT TIMESTAMP
);

-- GPU metrics
CREATE TABLE PROFILING_GPU_METRICS (
    ID VARCHAR(36) PRIMARY KEY,
    TIMESTAMP TIMESTAMP NOT NULL,
    DEVICE_ID INTEGER,
    UTILIZATION_PERCENT DECIMAL(5,2),
    MEMORY_USED_MB BIGINT,
    MEMORY_TOTAL_MB BIGINT,
    TEMPERATURE_C DECIMAL(5,2),
    POWER_WATTS DECIMAL(7,2)
);
```

### UI5 Dashboard Integration

The performance dashboard integrates seamlessly with the existing UI5 monitoring interface.

## Performance Impact

Profiling overhead (typical):
- **CPU Profiler**: ~2-5% overhead at 1000 Hz sampling
- **Memory Profiler**: ~3-7% overhead with allocation tracking
- **GPU Monitor**: <1% overhead (polling only)
- **Combined**: ~8-12% total overhead

Recommendations:
- Use lower sampling rates in production (100-500 Hz)
- Enable profiling on-demand rather than continuously
- Use sampling for memory profiling (not full tracking)
- Profile representative workloads, not production traffic

## Output Formats

### Flame Graph (SVG)
Interactive SVG with JavaScript for zooming/filtering

### JSON Report
```json
{
  "profile_id": "prof_20260123_014500",
  "duration_seconds": 60.0,
  "cpu": {
    "total_samples": 60000,
    "top_functions": [
      {
        "name": "llama_forward",
        "samples": 25000,
        "percent": 41.67,
        "file": "inference/engine/llama.zig",
        "line": 234
      }
    ]
  },
  "memory": {
    "peak_mb": 4096,
    "allocations": 1234567,
    "leaked_mb": 0.0
  },
  "bottlenecks": [
    {
      "function": "matmul_kernel",
      "avg_time_ms": 45.2,
      "call_count": 1000,
      "severity": "high"
    }
  ]
}
```

## Troubleshooting

### Issue: No stack traces captured

**Solution**: Ensure debug symbols are enabled:
```bash
zig build -Doptimize=ReleaseSafe
```

### Issue: High profiling overhead

**Solution**: Reduce sampling rate:
```json
{"cpu": {"sample_rate_hz": 100}}
```

### Issue: GPU metrics not available

**Solution**: Check CUDA toolkit installation:
```bash
nvidia-smi
export CUDA_HOME=/usr/local/cuda
```

## API Reference

### Profiling Endpoints

#### `POST /admin/profile/start`
Start profiling session

#### `POST /admin/profile/stop`
Stop profiling and return results

#### `GET /admin/profile/status`
Get current profiling status

#### `POST /admin/profile/cpu`
Capture CPU profile (duration in query param)

#### `POST /admin/profile/memory`
Capture memory snapshot

#### `POST /admin/profile/flamegraph`
Generate flame graph

#### `GET /admin/profile/bottlenecks`
Get bottleneck analysis

#### `GET /admin/profile/compare`
Compare two profiles

#### `GET /profiling/dashboard`
Performance dashboard UI

## Best Practices

1. **Profile Representative Workloads**: Use realistic models and prompts
2. **Multiple Runs**: Average results over multiple profiling sessions
3. **Isolate Variables**: Change one thing at a time when optimizing
4. **Document Baselines**: Save baseline profiles before making changes
5. **Monitor Regressions**: Set up alerts for performance degradation
6. **Focus on Hotspots**: Optimize functions that consume >5% of runtime
7. **Consider Trade-offs**: Balance latency, throughput, and memory usage

## Security Considerations

- Profiling endpoints require admin authentication
- Stack traces may expose internal implementation details
- Disable profiling in production or restrict to authorized users
- Sanitize exported profiles before sharing externally

## Roadmap

### Phase 1 (Complete)
- ✅ CPU profiling with sampling
- ✅ Memory profiling with allocation tracking
- ✅ GPU monitoring for CUDA
- ✅ Flame graph generation
- ✅ Bottleneck detection
- ✅ Performance dashboard

### Phase 2 (Future)
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Custom metric collection
- [ ] A/B testing framework
- [ ] Predictive performance modeling
- [ ] Auto-tuning recommendations

## Support

For issues or questions:
- Documentation: `/docs/profiling/`
- Examples: `/profiling/examples/`
- GitHub: Issue tracker

---

**Last Updated**: January 23, 2026  
**Maintainer**: nLocalModels Performance Team
