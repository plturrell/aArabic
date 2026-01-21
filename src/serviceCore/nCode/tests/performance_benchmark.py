#!/usr/bin/env python3
"""
nCode Performance Benchmark Suite - Day 9

Comprehensive performance testing covering:
1. SCIP index parsing performance
2. Database loading performance (Qdrant, Memgraph, Marquez)
3. API endpoint response times
4. Concurrent request handling
5. Memory usage profiling

Usage:
    python tests/performance_benchmark.py
    python tests/performance_benchmark.py --large-project
    python tests/performance_benchmark.py --profile
"""

import asyncio
import json
import logging
import os
import sys
import time
import psutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests not installed. API tests will be skipped.")

try:
    from loaders.scip_parser import load_scip_file
    from loaders.qdrant_loader import QdrantLoader
    from loaders.memgraph_loader import MemgraphLoader
    from loaders.marquez_loader import MarquezLoader
    LOADERS_AVAILABLE = True
except ImportError as e:
    LOADERS_AVAILABLE = False
    print(f"⚠️  Loader imports failed: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    test_name: str
    duration_ms: float
    success: bool
    throughput: Optional[float] = None  # items/second
    memory_mb: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class PerformanceReport:
    """Complete performance report"""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    results: List[BenchmarkResult]
    system_info: Dict[str, Any]
    summary: Dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmark suite for nCode"""
    
    def __init__(self, server_url: str = "http://localhost:18003"):
        self.server_url = server_url
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()
        
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def _time_operation(self, func, *args, **kwargs) -> tuple:
        """Time an operation and return (result, duration_ms, memory_mb)"""
        start_memory = self._get_memory_usage_mb()
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage_mb()
        
        duration_ms = (end_time - start_time) * 1000
        memory_delta_mb = end_memory - start_memory
        
        return result, duration_ms, memory_delta_mb, success, error
    
    async def _time_async_operation(self, coro) -> tuple:
        """Time an async operation"""
        start_memory = self._get_memory_usage_mb()
        start_time = time.perf_counter()
        
        try:
            result = await coro
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage_mb()
        
        duration_ms = (end_time - start_time) * 1000
        memory_delta_mb = end_memory - start_memory
        
        return result, duration_ms, memory_delta_mb, success, error
    
    # ==================== SCIP Parsing Benchmarks ====================
    
    def benchmark_scip_parsing(self, scip_file: str) -> BenchmarkResult:
        """Benchmark SCIP file parsing"""
        logger.info(f"📊 Benchmarking SCIP parsing: {scip_file}")
        
        if not LOADERS_AVAILABLE:
            return BenchmarkResult(
                test_name="SCIP Parsing",
                duration_ms=0,
                success=False,
                error="Loaders not available"
            )
        
        if not os.path.exists(scip_file):
            return BenchmarkResult(
                test_name="SCIP Parsing",
                duration_ms=0,
                success=False,
                error=f"File not found: {scip_file}"
            )
        
        file_size_mb = os.path.getsize(scip_file) / (1024 * 1024)
        
        result, duration_ms, memory_mb, success, error = self._time_operation(
            load_scip_file, scip_file
        )
        
        if success and result:
            symbol_count = sum(len(doc.symbols) for doc in result.documents)
            throughput = symbol_count / (duration_ms / 1000) if duration_ms > 0 else 0
            
            return BenchmarkResult(
                test_name="SCIP Parsing",
                duration_ms=duration_ms,
                success=True,
                throughput=throughput,
                memory_mb=memory_mb,
                details={
                    "file_size_mb": file_size_mb,
                    "document_count": len(result.documents),
                    "symbol_count": symbol_count,
                    "symbols_per_second": throughput
                }
            )
        else:
            return BenchmarkResult(
                test_name="SCIP Parsing",
                duration_ms=duration_ms,
                success=False,
                memory_mb=memory_mb,
                error=error
            )
    
    # ==================== Database Loading Benchmarks ====================
    
    async def benchmark_qdrant_loading(self, scip_file: str) -> BenchmarkResult:
        """Benchmark Qdrant database loading"""
        logger.info("📊 Benchmarking Qdrant loading")
        
        if not LOADERS_AVAILABLE:
            return BenchmarkResult(
                test_name="Qdrant Loading",
                duration_ms=0,
                success=False,
                error="Loaders not available"
            )
        
        try:
            loader = QdrantLoader(host="localhost", port=6333)
            loader.connect()
            
            # Create test collection
            collection_name = f"benchmark_{int(time.time())}"
            
            result, duration_ms, memory_mb, success, error = await self._time_async_operation(
                loader.load_scip_index(scip_file, collection_name, batch_size=100)
            )
            
            if success and result:
                throughput = result["symbols_loaded"] / (duration_ms / 1000) if duration_ms > 0 else 0
                
                # Clean up test collection
                try:
                    loader.client.delete_collection(collection_name)
                except:
                    pass
                
                return BenchmarkResult(
                    test_name="Qdrant Loading",
                    duration_ms=duration_ms,
                    success=True,
                    throughput=throughput,
                    memory_mb=memory_mb,
                    details={
                        "symbols_loaded": result["symbols_loaded"],
                        "documents": result["documents"],
                        "symbols_per_second": throughput
                    }
                )
            else:
                return BenchmarkResult(
                    test_name="Qdrant Loading",
                    duration_ms=duration_ms,
                    success=False,
                    memory_mb=memory_mb,
                    error=error
                )
        except Exception as e:
            return BenchmarkResult(
                test_name="Qdrant Loading",
                duration_ms=0,
                success=False,
                error=str(e)
            )
    
    async def benchmark_memgraph_loading(self, scip_file: str) -> BenchmarkResult:
        """Benchmark Memgraph database loading"""
        logger.info("📊 Benchmarking Memgraph loading")
        
        if not LOADERS_AVAILABLE:
            return BenchmarkResult(
                test_name="Memgraph Loading",
                duration_ms=0,
                success=False,
                error="Loaders not available"
            )
        
        try:
            loader = MemgraphLoader(host="localhost", port=7687)
            loader.connect()
            
            # Clear database first
            loader.clear_database()
            
            result, duration_ms, memory_mb, success, error = await self._time_async_operation(
                loader.load_scip_index(scip_file)
            )
            
            if success and result:
                total_items = result["documents"] + result["symbols"]
                throughput = total_items / (duration_ms / 1000) if duration_ms > 0 else 0
                
                return BenchmarkResult(
                    test_name="Memgraph Loading",
                    duration_ms=duration_ms,
                    success=True,
                    throughput=throughput,
                    memory_mb=memory_mb,
                    details={
                        "documents": result["documents"],
                        "symbols": result["symbols"],
                        "relationships": result["relationships"],
                        "items_per_second": throughput
                    }
                )
            else:
                return BenchmarkResult(
                    test_name="Memgraph Loading",
                    duration_ms=duration_ms,
                    success=False,
                    memory_mb=memory_mb,
                    error=error
                )
        except Exception as e:
            return BenchmarkResult(
                test_name="Memgraph Loading",
                duration_ms=0,
                success=False,
                error=str(e)
            )
    
    async def benchmark_marquez_loading(self, scip_file: str) -> BenchmarkResult:
        """Benchmark Marquez lineage tracking"""
        logger.info("📊 Benchmarking Marquez loading")
        
        if not LOADERS_AVAILABLE:
            return BenchmarkResult(
                test_name="Marquez Loading",
                duration_ms=0,
                success=False,
                error="Loaders not available"
            )
        
        try:
            loader = MarquezLoader(url="http://localhost:5000")
            
            result, duration_ms, memory_mb, success, error = await self._time_async_operation(
                loader.load_scip_index(scip_file, project="benchmark")
            )
            
            if success and result:
                return BenchmarkResult(
                    test_name="Marquez Loading",
                    duration_ms=duration_ms,
                    success=True,
                    memory_mb=memory_mb,
                    details={
                        "namespace": result["namespace"],
                        "job": result["job"],
                        "run_id": result["run_id"],
                        "status": result["status"]
                    }
                )
            else:
                return BenchmarkResult(
                    test_name="Marquez Loading",
                    duration_ms=duration_ms,
                    success=False,
                    memory_mb=memory_mb,
                    error=error
                )
        except Exception as e:
            return BenchmarkResult(
                test_name="Marquez Loading",
                duration_ms=0,
                success=False,
                error=str(e)
            )
    
    # ==================== API Endpoint Benchmarks ====================
    
    def benchmark_api_health(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark /health endpoint"""
        logger.info(f"📊 Benchmarking /health endpoint ({iterations} requests)")
        
        if not REQUESTS_AVAILABLE:
            return BenchmarkResult(
                test_name="API Health",
                duration_ms=0,
                success=False,
                error="requests library not available"
            )
        
        durations = []
        errors = 0
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            req_start = time.perf_counter()
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                req_duration = (time.perf_counter() - req_start) * 1000
                if response.status_code == 200:
                    durations.append(req_duration)
                else:
                    errors += 1
            except Exception:
                errors += 1
        
        total_duration_ms = (time.perf_counter() - start_time) * 1000
        
        if durations:
            return BenchmarkResult(
                test_name="API Health",
                duration_ms=total_duration_ms,
                success=True,
                throughput=len(durations) / (total_duration_ms / 1000),
                details={
                    "iterations": iterations,
                    "successful": len(durations),
                    "errors": errors,
                    "avg_latency_ms": statistics.mean(durations),
                    "min_latency_ms": min(durations),
                    "max_latency_ms": max(durations),
                    "p50_latency_ms": statistics.median(durations),
                    "p95_latency_ms": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations),
                    "requests_per_second": len(durations) / (total_duration_ms / 1000)
                }
            )
        else:
            return BenchmarkResult(
                test_name="API Health",
                duration_ms=total_duration_ms,
                success=False,
                error=f"All {iterations} requests failed"
            )
    
    def benchmark_api_load_index(self, scip_file: str) -> BenchmarkResult:
        """Benchmark /v1/index/load endpoint"""
        logger.info("📊 Benchmarking /v1/index/load endpoint")
        
        if not REQUESTS_AVAILABLE:
            return BenchmarkResult(
                test_name="API Load Index",
                duration_ms=0,
                success=False,
                error="requests library not available"
            )
        
        start_time = time.perf_counter()
        try:
            response = requests.post(
                f"{self.server_url}/v1/index/load",
                json={"path": scip_file},
                timeout=30
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return BenchmarkResult(
                    test_name="API Load Index",
                    duration_ms=duration_ms,
                    success=True,
                    details=data
                )
            else:
                return BenchmarkResult(
                    test_name="API Load Index",
                    duration_ms=duration_ms,
                    success=False,
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                test_name="API Load Index",
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
    
    def benchmark_api_endpoints(self) -> List[BenchmarkResult]:
        """Benchmark all API endpoints"""
        logger.info("📊 Benchmarking API endpoints")
        
        endpoints = [
            ("Definition", "/v1/definition", {"file": "test.zig", "line": 10, "character": 5}),
            ("References", "/v1/references", {"file": "test.zig", "line": 10, "character": 5}),
            ("Hover", "/v1/hover", {"file": "test.zig", "line": 10, "character": 5}),
            ("Symbols", "/v1/symbols", {"file": "test.zig"}),
            ("Document Symbols", "/v1/document-symbols", {"file": "test.zig"}),
        ]
        
        results = []
        for name, path, payload in endpoints:
            start_time = time.perf_counter()
            try:
                response = requests.post(
                    f"{self.server_url}{path}",
                    json=payload,
                    timeout=5
                )
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                results.append(BenchmarkResult(
                    test_name=f"API {name}",
                    duration_ms=duration_ms,
                    success=response.status_code == 200,
                    details={"status_code": response.status_code}
                ))
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                results.append(BenchmarkResult(
                    test_name=f"API {name}",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    # ==================== Concurrent Request Benchmarks ====================
    
    def benchmark_concurrent_requests(self, num_threads: int = 10, requests_per_thread: int = 10) -> BenchmarkResult:
        """Benchmark concurrent request handling"""
        logger.info(f"📊 Benchmarking concurrent requests ({num_threads} threads, {requests_per_thread} req/thread)")
        
        if not REQUESTS_AVAILABLE:
            return BenchmarkResult(
                test_name="Concurrent Requests",
                duration_ms=0,
                success=False,
                error="requests library not available"
            )
        
        def make_request():
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                return response.status_code == 200
            except:
                return False
        
        successful = 0
        total_requests = num_threads * requests_per_thread
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            for future in as_completed(futures):
                if future.result():
                    successful += 1
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        throughput = total_requests / (duration_ms / 1000)
        
        return BenchmarkResult(
            test_name="Concurrent Requests",
            duration_ms=duration_ms,
            success=successful > 0,
            throughput=throughput,
            details={
                "threads": num_threads,
                "requests_per_thread": requests_per_thread,
                "total_requests": total_requests,
                "successful": successful,
                "failed": total_requests - successful,
                "success_rate": (successful / total_requests) * 100,
                "requests_per_second": throughput
            }
        )
    
    # ==================== Report Generation ====================
    
    def generate_report(self) -> PerformanceReport:
        """Generate performance report"""
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed
        
        # Calculate summary statistics
        summary = {
            "avg_duration_ms": statistics.mean([r.duration_ms for r in self.results if r.duration_ms > 0]),
            "total_duration_ms": sum(r.duration_ms for r in self.results),
            "avg_throughput": statistics.mean([r.throughput for r in self.results if r.throughput]),
            "total_memory_mb": sum(r.memory_mb for r in self.results if r.memory_mb),
        }
        
        return PerformanceReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=len(self.results),
            passed=passed,
            failed=failed,
            results=self.results,
            system_info={
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": sys.version
            },
            summary=summary
        )
    
    def print_report(self, report: PerformanceReport):
        """Print formatted report"""
        print("\n" + "="*80)
        print("🚀 nCode Performance Benchmark Report")
        print("="*80)
        print(f"\nTimestamp: {report.timestamp}")
        print(f"Tests: {report.total_tests} total, {report.passed} passed, {report.failed} failed")
        print(f"\nSystem Info:")
        print(f"  CPU Cores: {report.system_info['cpu_count']}")
        print(f"  Memory: {report.system_info['memory_total_gb']:.1f} GB")
        
        print("\n" + "-"*80)
        print("Test Results:")
        print("-"*80)
        
        for result in report.results:
            status = "✅" if result.success else "❌"
            print(f"\n{status} {result.test_name}")
            print(f"   Duration: {result.duration_ms:.2f} ms")
            
            if result.throughput:
                print(f"   Throughput: {result.throughput:.2f} items/sec")
            if result.memory_mb:
                print(f"   Memory: {result.memory_mb:.2f} MB")
            if result.details:
                print(f"   Details: {json.dumps(result.details, indent=6)}")
            if result.error:
                print(f"   Error: {result.error}")
        
        print("\n" + "-"*80)
        print("Summary:")
        print("-"*80)
        print(f"  Average Duration: {report.summary['avg_duration_ms']:.2f} ms")
        print(f"  Total Duration: {report.summary['total_duration_ms']:.2f} ms")
        print(f"  Average Throughput: {report.summary['avg_throughput']:.2f} items/sec")
        print(f"  Total Memory: {report.summary['total_memory_mb']:.2f} MB")
        print("\n" + "="*80 + "\n")
    
    def save_report(self, report: PerformanceReport, output_file: str):
        """Save report to JSON file"""
        report_dict = asdict(report)
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"📄 Report saved to {output_file}")


async def run_benchmarks(scip_file: str, include_large: bool = False):
    """Run all performance benchmarks"""
    benchmark = PerformanceBenchmark()
    
    print("\n" + "="*80)
    print("🚀 Starting nCode Performance Benchmark Suite")
    print("="*80 + "\n")
    
    # 1. SCIP Parsing
    result = benchmark.benchmark_scip_parsing(scip_file)
    benchmark.results.append(result)
    
    # 2. Database Loading (if available)
    if LOADERS_AVAILABLE:
        result = await benchmark.benchmark_qdrant_loading(scip_file)
        benchmark.results.append(result)
        
        result = await benchmark.benchmark_memgraph_loading(scip_file)
        benchmark.results.append(result)
        
        result = await benchmark.benchmark_marquez_loading(scip_file)
        benchmark.results.append(result)
    
    # 3. API Endpoints (if server is running)
    if REQUESTS_AVAILABLE:
        try:
            requests.get(f"{benchmark.server_url}/health", timeout=2)
            
            result = benchmark.benchmark_api_health(iterations=100)
            benchmark.results.append(result)
            
            result = benchmark.benchmark_api_load_index(scip_file)
            benchmark.results.append(result)
            
            for result in benchmark.benchmark_api_endpoints():
                benchmark.results.append(result)
            
            result = benchmark.benchmark_concurrent_requests(num_threads=10, requests_per_thread=10)
            benchmark.results.append(result)
            
            if include_large:
                result = benchmark.benchmark_concurrent_requests(num_threads=50, requests_per_thread=10)
                benchmark.results.append(result)
        except Exception as e:
            logger.warning(f"⚠️  Server not available: {e}")
    
    # Generate and print report
    report = benchmark.generate_report()
    benchmark.print_report(report)
    
    # Save report
    output_file = f"tests/performance_report_{int(time.time())}.json"
    benchmark.save_report(report, output_file)
    
    return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="nCode Performance Benchmark Suite")
    parser.add_argument("--scip-file", default="tests/sample.scip", help="Path to SCIP index file")
    parser.add_argument("--large-project", action="store_true", help="Include large project benchmarks")
    parser.add_argument("--profile", action="store_true", help="Enable memory profiling")
    
    args = parser.parse_args()
    
    # Check if SCIP file exists
    if not os.path.exists(args.scip_file):
        logger.error(f"❌ SCIP file not found: {args.scip_file}")
        logger.info("💡 Run indexing first to generate SCIP file")
        sys.exit(1)
    
    # Run benchmarks
    asyncio.run(run_benchmarks(args.scip_file, args.large_project))


if __name__ == "__main__":
    main()
