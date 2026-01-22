#!/usr/bin/env python3
import argparse
import json
import queue
import threading
import time
from http.client import HTTPConnection
from urllib.parse import urlparse


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((pct / 100.0) * (len(values) - 1)))
    return values[k]


def build_payload(path, model, prompt, max_tokens, temperature):
    if path.endswith("/v1/chat/completions"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    return json.dumps(payload)


def worker(worker_id, host, port, path, headers, body, timeout, work_q, results):
    conn = HTTPConnection(host, port, timeout=timeout)
    while True:
        try:
            _ = work_q.get_nowait()
        except queue.Empty:
            break
        start = time.perf_counter()
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            ttfb = time.perf_counter() - start
            _ = resp.read()
            end = time.perf_counter()
            latency = end - start
            results["latencies"].append(latency)
            results["ttfb"].append(ttfb)
            if resp.status != 200:
                results["errors"] += 1
        except Exception:
            results["errors"] += 1
        finally:
            work_q.task_done()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="LLM HTTP benchmark (QPS and latency)")
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1/completions")
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--prompt", default="Hello from Shimmy-Mojo.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    parsed = urlparse(args.url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/v1/completions"

    body = build_payload(path, args.model, args.prompt, args.max_tokens, args.temperature)
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "Connection": "keep-alive",
    }

    work_q = queue.Queue()
    for _ in range(args.requests):
        work_q.put(1)

    results = {"latencies": [], "ttfb": [], "errors": 0}
    threads = []

    start_time = time.perf_counter()
    for i in range(args.concurrency):
        t = threading.Thread(
            target=worker,
            args=(i, host, port, path, headers, body, args.timeout, work_q, results),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    total = args.requests
    success = total - results["errors"]
    qps = success / elapsed if elapsed > 0 else 0.0

    lat_ms = [v * 1000.0 for v in results["latencies"]]
    ttfb_ms = [v * 1000.0 for v in results["ttfb"]]

    print("Requests:", total, "Success:", success, "Errors:", results["errors"])
    print("Elapsed: %.2fs  QPS: %.2f" % (elapsed, qps))
    if lat_ms:
        print(
            "Latency ms (p50/p95/p99): %.1f / %.1f / %.1f"
            % (percentile(lat_ms, 50), percentile(lat_ms, 95), percentile(lat_ms, 99))
        )
    if ttfb_ms:
        print(
            "TTFB ms (p50/p95/p99): %.1f / %.1f / %.1f"
            % (percentile(ttfb_ms, 50), percentile(ttfb_ms, 95), percentile(ttfb_ms, 99))
        )
        print("Note: Without streaming, TTFB ~= full latency.")


if __name__ == "__main__":
    main()
