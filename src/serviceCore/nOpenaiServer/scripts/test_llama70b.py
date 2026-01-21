#!/usr/bin/env python3
"""
Test client and benchmark script for Llama 3.3 70B server.
Uses only standard library (no external dependencies).

Usage:
    python3 test_llama70b.py --quick              # Quick smoke test
    python3 test_llama70b.py --benchmark          # Full benchmark suite
    python3 test_llama70b.py --benchmark --concurrent 4  # Concurrent benchmark
    python3 test_llama70b.py --host 192.168.1.100 --port 8080 --max-tokens 512
"""

import argparse
import json
import queue
import sys
import threading
import time
from http.client import HTTPConnection
from typing import Any
from urllib.parse import urljoin

# Sample prompts for testing
PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain quantum computing in 3 sentences.",
    "long": """You are an expert technical writer. Please analyze the following scenario and provide a comprehensive response.

A software development team is building a distributed system that needs to handle millions of requests per second. The system must be fault-tolerant, scalable, and maintain low latency. The team is considering various architectural patterns including microservices, event-driven architecture, and CQRS (Command Query Responsibility Segregation).

The current challenges include:
1. Database bottlenecks under high load
2. Inconsistent data across services
3. Difficulty in debugging distributed transactions
4. High operational complexity
5. Latency spikes during peak traffic

The team has experience with Python, Go, and Rust. They are currently using PostgreSQL as their primary database and Redis for caching. The infrastructure runs on Kubernetes in AWS.

Please provide:
1. A detailed analysis of which architectural pattern would best suit their needs
2. Specific recommendations for handling the database bottleneck
3. Strategies for maintaining data consistency
4. Best practices for observability and debugging
5. A phased migration plan if they need to transition from their current architecture

Consider trade-offs between complexity, performance, and maintainability. Include specific technology recommendations where appropriate, and explain the reasoning behind each suggestion. Also discuss potential pitfalls and how to avoid them.""",
}


class LlamaClient:
    """OpenAI-compatible client for Llama server."""

    def __init__(self, host: str = "localhost", port: int = 11434, timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
        """Make HTTP request and return (status_code, response_data)."""
        conn = HTTPConnection(self.host, self.port, timeout=self.timeout)
        headers = {"Content-Type": "application/json"}
        body_bytes = json.dumps(body).encode() if body else None
        if body_bytes:
            headers["Content-Length"] = str(len(body_bytes))
        try:
            conn.request(method, path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            data = resp.read().decode()
            try:
                return resp.status, json.loads(data)
            except json.JSONDecodeError:
                return resp.status, data
        finally:
            conn.close()

    def health_check(self) -> tuple[bool, str]:
        """Check server health."""
        try:
            status, data = self._request("GET", "/health")
            return status == 200, str(data)
        except Exception as e:
            return False, str(e)

    def list_models(self) -> tuple[bool, list[str] | str]:
        """List available models."""
        try:
            status, data = self._request("GET", "/v1/models")
            if status == 200 and isinstance(data, dict):
                models = [m.get("id", "unknown") for m in data.get("data", [])]
                return True, models
            return False, str(data)
        except Exception as e:
            return False, str(e)

    def chat_completion(
        self, prompt: str, model: str = "llama-3.3-70b", max_tokens: int = 256, temperature: float = 0.7
    ) -> tuple[bool, dict]:
        """Send chat completion request."""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        start = time.perf_counter()
        try:
            status, data = self._request("POST", "/v1/chat/completions", body)
            elapsed = time.perf_counter() - start
            if status == 200 and isinstance(data, dict):
                content = ""
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                return True, {
                    "content": content,
                    "elapsed": elapsed,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            return False, {"error": str(data), "elapsed": elapsed}
        except Exception as e:
            return False, {"error": str(e), "elapsed": time.perf_counter() - start}


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.latencies: list[float] = []
        self.ttfb: list[float] = []
        self.tokens_generated: list[int] = []
        self.errors: int = 0
        self.start_time: float = 0
        self.end_time: float = 0

    def add_success(self, latency: float, ttfb: float, tokens: int):
        self.latencies.append(latency)
        self.ttfb.append(ttfb)
        self.tokens_generated.append(tokens)

    def add_error(self):
        self.errors += 1

    @staticmethod
    def percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = int(round((pct / 100.0) * (len(values) - 1)))
        return values[k]

    def summary(self) -> dict[str, Any]:
        elapsed = self.end_time - self.start_time
        total_requests = len(self.latencies) + self.errors
        total_tokens = sum(self.tokens_generated)
        return {
            "name": self.name,
            "total_requests": total_requests,
            "successful": len(self.latencies),
            "errors": self.errors,
            "elapsed_seconds": round(elapsed, 2),
            "requests_per_second": round(len(self.latencies) / elapsed, 2) if elapsed > 0 else 0,
            "tokens_per_second": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
            "latency_p50_ms": round(self.percentile(self.latencies, 50) * 1000, 1),
            "latency_p95_ms": round(self.percentile(self.latencies, 95) * 1000, 1),
            "latency_p99_ms": round(self.percentile(self.latencies, 99) * 1000, 1),
            "ttfb_p50_ms": round(self.percentile(self.ttfb, 50) * 1000, 1),
            "ttfb_p95_ms": round(self.percentile(self.ttfb, 95) * 1000, 1),
            "avg_tokens_per_request": round(total_tokens / len(self.latencies), 1) if self.latencies else 0,
        }


def benchmark_worker(
    worker_id: int,
    client: LlamaClient,
    prompt: str,
    max_tokens: int,
    work_q: queue.Queue,
    result: BenchmarkResult,
    lock: threading.Lock,
):
    """Worker thread for concurrent benchmarking."""
    while True:
        try:
            _ = work_q.get_nowait()
        except queue.Empty:
            break
        start = time.perf_counter()
        success, data = client.chat_completion(prompt, max_tokens=max_tokens)
        ttfb = time.perf_counter() - start  # Without streaming, TTFB ≈ total time
        latency = data.get("elapsed", time.perf_counter() - start)
        with lock:
            if success:
                tokens = data.get("completion_tokens", 0)
                result.add_success(latency, ttfb, tokens)
            else:
                result.add_error()
        work_q.task_done()


def run_benchmark(
    client: LlamaClient,
    prompt: str,
    prompt_name: str,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
) -> BenchmarkResult:
    """Run benchmark with given parameters."""
    result = BenchmarkResult(f"{prompt_name}_c{concurrency}")
    work_q: queue.Queue = queue.Queue()
    lock = threading.Lock()

    for _ in range(num_requests):
        work_q.put(1)

    threads = []
    result.start_time = time.perf_counter()

    for i in range(concurrency):
        t = threading.Thread(
            target=benchmark_worker,
            args=(i, client, prompt, max_tokens, work_q, result, lock),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    result.end_time = time.perf_counter()
    return result


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_result(label: str, value: str, indent: int = 2):
    """Print formatted result line."""
    print(" " * indent + f"{label}: {value}")


def format_markdown_table(results: list[dict]) -> str:
    """Format results as markdown table."""
    if not results:
        return ""
    headers = ["Test", "Reqs", "OK", "Err", "Time(s)", "Req/s", "Tok/s", "Lat p50", "Lat p95", "TTFB p50"]
    rows = []
    for r in results:
        rows.append([
            r["name"],
            str(r["total_requests"]),
            str(r["successful"]),
            str(r["errors"]),
            str(r["elapsed_seconds"]),
            str(r["requests_per_second"]),
            str(r["tokens_per_second"]),
            f"{r['latency_p50_ms']}ms",
            f"{r['latency_p95_ms']}ms",
            f"{r['ttfb_p50_ms']}ms",
        ])

    col_widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    data_lines = ["| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for row in rows]

    return "\n".join([header_line, separator] + data_lines)


def quick_test(client: LlamaClient, max_tokens: int) -> bool:
    """Run quick smoke test."""
    print_header("Quick Smoke Test")
    all_passed = True

    # Health check
    print("\n[1/3] Health Check...")
    ok, msg = client.health_check()
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {status}: {msg}")
    all_passed = all_passed and ok

    # List models
    print("\n[2/3] List Models...")
    ok, models = client.list_models()
    status = "✓ PASS" if ok else "✗ FAIL"
    if ok:
        print(f"  {status}: Found {len(models)} model(s)")
        for m in models[:5]:
            print(f"    - {m}")
    else:
        print(f"  {status}: {models}")
    all_passed = all_passed and ok

    # Chat completion
    print("\n[3/3] Chat Completion (short prompt)...")
    ok, data = client.chat_completion(PROMPTS["short"], max_tokens=max_tokens)
    status = "✓ PASS" if ok else "✗ FAIL"
    if ok:
        print(f"  {status}")
        print_result("Response", data["content"][:100] + "..." if len(data["content"]) > 100 else data["content"])
        print_result("Elapsed", f"{data['elapsed']:.2f}s")
        print_result("Tokens", f"{data['completion_tokens']} completion, {data['prompt_tokens']} prompt")
        if data["elapsed"] > 0 and data["completion_tokens"] > 0:
            tps = data["completion_tokens"] / data["elapsed"]
            print_result("Tokens/sec", f"{tps:.1f}")
    else:
        print(f"  {status}: {data.get('error', 'Unknown error')}")
    all_passed = all_passed and ok

    print("\n" + "-" * 60)
    final_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
    print(f"  {final_status}")
    return all_passed


def run_full_benchmark(client: LlamaClient, max_tokens: int, concurrency_levels: list[int]) -> list[dict]:
    """Run full benchmark suite."""
    print_header("Full Benchmark Suite")
    results = []
    num_requests = 5  # Requests per test

    prompt_tests = [
        ("short", PROMPTS["short"]),
        ("medium", PROMPTS["medium"]),
        ("long", PROMPTS["long"]),
    ]

    total_tests = len(prompt_tests) * len(concurrency_levels)
    current = 0

    for prompt_name, prompt in prompt_tests:
        for conc in concurrency_levels:
            current += 1
            test_name = f"{prompt_name}_c{conc}"
            print(f"\n[{current}/{total_tests}] Running: {test_name}")
            print(f"  Prompt: {prompt_name} ({len(prompt)} chars)")
            print(f"  Concurrency: {conc}, Requests: {num_requests}, Max tokens: {max_tokens}")

            result = run_benchmark(client, prompt, prompt_name, num_requests, conc, max_tokens)
            summary = result.summary()
            results.append(summary)

            print(f"  Results:")
            print_result("Successful", f"{summary['successful']}/{summary['total_requests']}", 4)
            print_result("Elapsed", f"{summary['elapsed_seconds']}s", 4)
            print_result("Tokens/sec", str(summary["tokens_per_second"]), 4)
            print_result("Latency p50/p95", f"{summary['latency_p50_ms']}ms / {summary['latency_p95_ms']}ms", 4)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test client and benchmark for Llama 3.3 70B server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    Quick smoke test
  %(prog)s --benchmark                Full benchmark suite
  %(prog)s --benchmark --concurrent 4 Benchmark with 4 concurrent requests
  %(prog)s --host 192.168.1.100       Test remote server
  %(prog)s --json                     Output results as JSON
        """,
    )
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=11434, help="Server port (default: 11434)")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrency level (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds (default: 120)")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output results as markdown table")
    args = parser.parse_args()

    # Default to quick test if no mode specified
    if not args.benchmark and not args.quick:
        args.quick = True

    print(f"Llama 3.3 70B Test Client")
    print(f"Target: http://{args.host}:{args.port}")
    print(f"Max tokens: {args.max_tokens}, Timeout: {args.timeout}s")

    client = LlamaClient(host=args.host, port=args.port, timeout=args.timeout)
    results = []

    if args.quick:
        success = quick_test(client, args.max_tokens)
        if not success:
            sys.exit(1)

    if args.benchmark:
        # Parse concurrency levels: if single value, test 1,2,4,8 or just that value
        if args.concurrent == 1:
            concurrency_levels = [1, 2, 4, 8]
        else:
            concurrency_levels = [args.concurrent]

        results = run_full_benchmark(client, args.max_tokens, concurrency_levels)

        # Output results
        print_header("Benchmark Summary")

        if args.json:
            print(json.dumps(results, indent=2))
        elif args.markdown:
            print("\n" + format_markdown_table(results))
        else:
            # Pretty print summary
            print("\n" + format_markdown_table(results))

            # Overall statistics
            if results:
                total_tokens = sum(r["tokens_per_second"] * r["elapsed_seconds"] for r in results)
                total_time = sum(r["elapsed_seconds"] for r in results)
                avg_tps = total_tokens / total_time if total_time > 0 else 0
                print(f"\n  Overall average tokens/sec: {avg_tps:.1f}")
                print(f"  Total benchmark time: {total_time:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
