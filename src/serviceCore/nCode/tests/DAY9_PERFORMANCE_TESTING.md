# Day 9: Performance Testing & Optimization - nCode

**Date:** 2026-01-18 (Thursday)  
**Status:** ✅ Complete  
**Focus:** Comprehensive performance benchmarking and optimization recommendations

---

## Overview

Day 9 establishes a complete performance testing framework for nCode, measuring and documenting system performance across all critical operations. This includes SCIP parsing, database loading, API endpoints, and concurrent request handling.

## Deliverables

### 1. Performance Benchmark Suite ✅

**File:** `tests/performance_benchmark.py` (900+ lines)

A comprehensive Python-based benchmarking framework that measures:

- **SCIP Parsing Performance**
  - File parsing speed (symbols/second)
  - Memory consumption during parsing
  - Throughput metrics
  
- **Database Loading Performance**
  - Qdrant vector database insertion speed
  - Memgraph graph database loading
  - Marquez lineage tracking overhead
  
- **API Endpoint Performance**
  - Health check latency
  - Index loading time
  - Query endpoint response times (definition, references, hover, symbols)
  - Percentile latency analysis (p50, p95, p99)
  
- **Concurrent Request Handling**
  - Multi-threaded request testing
  - Throughput under load
  - Success rate under concurrent load

**Key Features:**
- Automatic timing and memory profiling
- JSON report generation with detailed metrics
- Graceful degradation when services unavailable
- Configurable test parameters
- Statistical analysis (mean, median, percentiles)

### 2. Test Runner Script ✅

**File:** `tests/run_performance_tests.sh` (200+ lines)

Automated test execution with:
- Prerequisite checking (Python, packages, services)
- Service availability detection
- Automatic package installation
- Color-coded output
- Result summarization
- JQ-based metrics extraction

**Usage:**
```bash
# Basic benchmarks
./tests/run_performance_tests.sh

# Include large-scale tests
./tests/run_performance_tests.sh --large

# Enable memory profiling
./tests/run_performance_tests.sh --profile

# Custom SCIP file
./tests/run_performance_tests.sh --scip-file path/to/index.scip
```

---

## Performance Benchmarks

### Expected Performance Targets

Based on Day 9 requirements and system design:

#### SCIP Parsing
- **Target:** < 5 minutes for 10K file project
- **Expected:** 1,000-5,000 symbols/second
- **Memory:** < 500 MB for medium projects
- **Throughput:** Depends on file complexity and symbol density

#### Database Loading

**Qdrant (Vector Search):**
- **Target:** < 2 minutes for medium projects
- **Batch size:** 100 symbols per batch
- **Expected throughput:** 50-200 symbols/second
- **Memory:** < 1 GB during loading

**Memgraph (Graph Database):**
- **Target:** < 2 minutes for medium projects
- **Expected throughput:** 100-500 nodes/second
- **Relationships:** Fast bulk insertion via Cypher
- **Memory:** < 500 MB during loading

**Marquez (Lineage Tracking):**
- **Target:** < 200ms per API call
- **Expected:** 40-100ms per event
- **Overhead:** Minimal impact on indexing
- **API latency:** < 200ms per request

#### API Endpoints

**Health Check:**
- **Target:** < 10ms average
- **Expected:** 2-5ms typical
- **Throughput:** > 1000 requests/second

**Index Loading:**
- **Target:** < 100ms for cache hit
- **Expected:** 50-200ms
- **First load:** May take longer (file I/O)

**Query Endpoints:**
- **Definition/References:** < 50ms
- **Hover information:** < 30ms
- **Symbol listing:** < 100ms
- **Document symbols:** < 100ms

#### Concurrent Requests
- **Target:** Handle 50+ concurrent requests
- **Expected:** > 100 requests/second sustained
- **Success rate:** > 99% under normal load
- **Latency impact:** < 2x increase under 10x load

---

## Test Architecture

### Benchmark Classes

```python
class PerformanceBenchmark:
    """Main benchmark suite coordinator"""
    
    # Core benchmarking methods
    - benchmark_scip_parsing()
    - benchmark_qdrant_loading()
    - benchmark_memgraph_loading()
    - benchmark_marquez_loading()
    - benchmark_api_health()
    - benchmark_api_endpoints()
    - benchmark_concurrent_requests()
    
    # Reporting
    - generate_report()
    - print_report()
    - save_report()
```

### Data Structures

```python
@dataclass
class BenchmarkResult:
    test_name: str
    duration_ms: float
    success: bool
    throughput: Optional[float]      # items/second
    memory_mb: Optional[float]       # MB used
    details: Optional[Dict]          # Test-specific data
    error: Optional[str]             # Error message if failed

@dataclass
class PerformanceReport:
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    results: List[BenchmarkResult]
    system_info: Dict                # CPU, memory, Python version
    summary: Dict                    # Aggregated statistics
```

---

## Running Performance Tests

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip3 install requests psutil protobuf qdrant-client neo4j
   ```

2. **Start nCode server:**
   ```bash
   cd src/serviceCore/nCode
   zig build run
   ```

3. **Start database services (optional but recommended):**
   ```bash
   # Qdrant
   docker run -p 6333:6333 qdrant/qdrant
   
   # Memgraph
   docker run -p 7687:7687 memgraph/memgraph-platform
   
   # Marquez
   docker run -p 5000:5000 marquezproject/marquez
   ```

### Quick Start

```bash
# Navigate to nCode directory
cd src/serviceCore/nCode

# Run basic benchmarks
./tests/run_performance_tests.sh

# Run with all options
./tests/run_performance_tests.sh --large --profile
```

### Output

The benchmark generates:

1. **Console output** with real-time progress and results
2. **JSON report** with detailed metrics: `tests/performance_report_<timestamp>.json`
3. **Summary statistics** showing key performance indicators

Example output:
```
================================================================================
🚀 nCode Performance Benchmark Report
================================================================================

Timestamp: 2026-01-18 06:45:00
Tests: 12 total, 11 passed, 1 failed

System Info:
  CPU Cores: 8
  Memory: 16.0 GB

--------------------------------------------------------------------------------
Test Results:
--------------------------------------------------------------------------------

✅ SCIP Parsing
   Duration: 1234.56 ms
   Throughput: 2500.00 items/sec
   Memory: 256.00 MB
   Details: {
      "file_size_mb": 5.2,
      "document_count": 150,
      "symbol_count": 3086,
      "symbols_per_second": 2500.0
   }

✅ API Health
   Duration: 1050.00 ms
   Throughput: 95.24 items/sec
   Details: {
      "iterations": 100,
      "avg_latency_ms": 4.5,
      "p50_latency_ms": 4.2,
      "p95_latency_ms": 6.8,
      "requests_per_second": 95.24
   }

--------------------------------------------------------------------------------
Summary:
--------------------------------------------------------------------------------
  Average Duration: 825.50 ms
  Total Duration: 9906.00 ms
  Average Throughput: 450.75 items/sec
  Total Memory: 512.50 MB
```

---

## Performance Analysis

### Profiling Tools

The benchmark suite includes:

1. **Timing Profiling**
   - High-resolution `time.perf_counter()` timing
   - Per-operation duration tracking
   - Statistical analysis of latencies

2. **Memory Profiling**
   - Process memory usage via `psutil`
   - Memory delta measurement
   - Peak memory tracking

3. **Throughput Analysis**
   - Items processed per second
   - Request rate calculation
   - Concurrent load handling

### Metrics Collected

For each test:
- **Duration:** Total execution time (ms)
- **Throughput:** Items/requests per second
- **Memory:** Memory consumed (MB)
- **Success rate:** Percentage of successful operations
- **Latency percentiles:** p50, p95, p99 (where applicable)
- **Error details:** Failure reasons and context

---

## Optimization Recommendations

### SCIP Parsing Optimizations

1. **Protobuf Parsing:**
   - Use streaming parsing for large files
   - Implement lazy loading of symbol details
   - Cache frequently accessed symbols

2. **Memory Management:**
   - Process documents in batches
   - Clear intermediate data structures
   - Use memory-mapped files for large indexes

3. **Parallelization:**
   - Parse multiple documents concurrently
   - Use thread pool for symbol processing
   - Batch symbol extraction operations

### Database Loading Optimizations

1. **Qdrant:**
   - Increase batch size (100 → 500)
   - Use gRPC instead of HTTP (faster)
   - Pre-compute embeddings in parallel
   - Use proper vector dimensions (384 vs 768)

2. **Memgraph:**
   - Use UNWIND for batch inserts
   - Create indexes before bulk loading
   - Use transactions for atomicity
   - Optimize Cypher queries

3. **Marquez:**
   - Batch lineage events
   - Use async API calls
   - Cache namespace/dataset creation
   - Minimize API round trips

### API Server Optimizations

1. **Request Handling:**
   - Implement connection pooling
   - Add request caching layer
   - Use async I/O where possible
   - Optimize JSON parsing

2. **Index Management:**
   - Lazy load index data
   - Cache symbol lookups
   - Use memory-efficient data structures
   - Implement index compression

3. **Concurrent Access:**
   - Add request queue management
   - Implement rate limiting
   - Use lock-free data structures
   - Optimize thread pool sizing

---

## Performance Monitoring

### Production Metrics

Key metrics to monitor in production:

1. **Indexing Performance:**
   - Time to index (seconds)
   - Symbols processed per second
   - Memory usage during indexing
   - Error rate

2. **API Performance:**
   - Request latency (p50, p95, p99)
   - Requests per second
   - Error rate by endpoint
   - Cache hit rate

3. **Database Performance:**
   - Query latency
   - Connection pool utilization
   - Transaction throughput
   - Disk I/O

4. **System Resources:**
   - CPU usage (%)
   - Memory usage (MB/GB)
   - Network throughput
   - Disk I/O

### Monitoring Integration

The nCode server v2 includes:
- **Prometheus metrics** endpoint: `GET /metrics`
- **JSON metrics** endpoint: `GET /metrics.json`
- **Structured logging** with performance data
- **Request tracking** with timing

---

## Continuous Performance Testing

### CI/CD Integration

Integrate performance tests into CI/CD:

```yaml
# Example GitHub Actions workflow
performance-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Setup services
      run: docker-compose up -d
    - name: Run performance tests
      run: ./tests/run_performance_tests.sh
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: performance-report
        path: tests/performance_report_*.json
```

### Regression Detection

1. **Baseline Establishment:**
   - Run benchmarks on known-good version
   - Save results as baseline
   - Track metrics over time

2. **Comparison:**
   - Compare new results against baseline
   - Alert on > 10% performance degradation
   - Track performance trends

3. **Automated Alerts:**
   - Fail CI if critical metrics degrade
   - Generate performance comparison reports
   - Notify team of regressions

---

## Troubleshooting

### Common Issues

1. **Tests Fail to Start:**
   - Check Python dependencies: `pip3 install -r requirements.txt`
   - Verify SCIP file exists
   - Ensure databases are running

2. **Low Throughput:**
   - Check system resources (CPU, memory)
   - Verify database connections are healthy
   - Review logs for errors/warnings

3. **High Memory Usage:**
   - Reduce batch sizes
   - Clear caches between tests
   - Check for memory leaks

4. **Inconsistent Results:**
   - Ensure system is idle during testing
   - Run multiple iterations
   - Use statistical analysis to account for variance

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
./tests/run_performance_tests.sh
```

---

## Next Steps

After performance testing (Day 10):

1. **Docker Compose Setup:**
   - Package all services for one-command deployment
   - Optimize container resource limits
   - Add health checks and monitoring

2. **Performance Tuning:**
   - Implement identified optimizations
   - Re-run benchmarks to verify improvements
   - Document performance characteristics

3. **Load Testing:**
   - Test with realistic workloads
   - Identify bottlenecks under load
   - Validate scalability assumptions

4. **Documentation:**
   - Update README with performance specs
   - Document optimization strategies
   - Create performance tuning guide

---

## Files Created

```
tests/
├── performance_benchmark.py        # 900+ lines - Complete benchmark suite
├── run_performance_tests.sh        # 200+ lines - Automated test runner
└── DAY9_PERFORMANCE_TESTING.md    # This file - Comprehensive documentation
```

---

## Summary

Day 9 successfully delivered:

✅ **Complete performance benchmarking framework**
- SCIP parsing, database loading, API endpoints, concurrent requests
- Memory and throughput profiling
- Statistical analysis and reporting

✅ **Automated test execution**
- Prerequisite checking and service detection
- Multiple test modes (basic, large, profiling)
- JSON report generation

✅ **Comprehensive documentation**
- Performance targets and expectations
- Optimization recommendations
- Integration guidelines

✅ **Production-ready metrics**
- Baseline establishment
- Regression detection capability
- CI/CD integration ready

**Total Implementation:** 1,100+ lines of code and documentation

The nCode system now has a robust performance testing framework that will ensure production quality and identify optimization opportunities throughout development.

---

**Day 9 Complete!** ✅

Ready for Day 10: Docker Compose Setup
