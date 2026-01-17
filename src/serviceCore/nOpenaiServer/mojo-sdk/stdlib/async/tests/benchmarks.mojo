# Async Benchmarks - Day 104
# Performance benchmarks for the async system

from async.async_io import AsyncFile, read_file, write_file
from async.async_channels import bounded_channel, unbounded_channel
from async.async_sync import AsyncMutex, AsyncSemaphore
from async.async_utils import spawn, join_all
from async.async_stream import stream_from_range
from collections import List
from time.time import Duration, Stopwatch

# ============================================================================
# Benchmark Framework
# ============================================================================

@value
struct BenchmarkResult:
    """Result of a benchmark."""
    var name: String
    var iterations: Int
    var total_time_ns: Int
    var avg_time_ns: Int
    var throughput: Float
    
    fn __init__(inout self, name: String, iterations: Int, total_time_ns: Int):
        self.name = name
        self.iterations = iterations
        self.total_time_ns = total_time_ns
        self.avg_time_ns = total_time_ns // iterations
        self.throughput = Float(iterations) / (Float(total_time_ns) / 1_000_000_000.0)
    
    fn print(self):
        """Print benchmark result."""
        print("\n" + self.name + ":")
        print("  Iterations: " + str(self.iterations))
        print("  Total time: " + str(self.total_time_ns / 1_000_000) + " ms")
        print("  Avg time: " + str(self.avg_time_ns) + " ns")
        print("  Throughput: " + str(self.throughput) + " ops/sec")


@value
struct BenchmarkSuite:
    """Collection of benchmarks."""
    var name: String
    var results: List[BenchmarkResult]
    
    fn __init__(inout self, name: String):
        self.name = name
        self.results = List[BenchmarkResult]()
    
    fn add_result(inout self, result: BenchmarkResult):
        """Add benchmark result."""
        self.results.append(result)
    
    fn print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print(self.name + " - Benchmark Summary")
        print("="*60)
        
        for result in self.results:
            result.print()


fn benchmark[F: Fn[None]](name: String, iterations: Int, func: F) -> BenchmarkResult:
    """Run a benchmark.
    
    Args:
        name: Benchmark name.
        iterations: Number of iterations.
        func: Function to benchmark.
    
    Returns:
        Benchmark result.
    """
    var timer = Stopwatch()
    timer.start()
    
    for i in range(iterations):
        func()
    
    let elapsed_ns = timer.elapsed_ns()
    return BenchmarkResult(name, iterations, elapsed_ns)


async fn async_benchmark[F: AsyncFn[None]](
    name: String,
    iterations: Int,
    func: F
) -> BenchmarkResult:
    """Run an async benchmark."""
    var timer = Stopwatch()
    timer.start()
    
    for i in range(iterations):
        await func()
    
    let elapsed_ns = timer.elapsed_ns()
    return BenchmarkResult(name, iterations, elapsed_ns)


# ============================================================================
# Benchmark 1: Task Spawning
# ============================================================================

async fn bench_spawn_tasks() -> BenchmarkResult:
    """Benchmark task spawning overhead."""
    let iterations = 10000
    var timer = Stopwatch()
    timer.start()
    
    for i in range(iterations):
        let handle = spawn(async { return 42 })
        _ = await handle.join()
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Task Spawning", iterations, elapsed)


async fn bench_spawn_and_join() -> BenchmarkResult:
    """Benchmark spawning and joining multiple tasks."""
    let iterations = 1000
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        var handles = List[JoinHandle[Int]]()
        for j in range(10):
            handles.append(spawn(async { return j }))
        _ = await join_all(handles)
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Spawn & Join 10 Tasks", iterations * 10, elapsed)


# ============================================================================
# Benchmark 2: Channel Throughput
# ============================================================================

async fn bench_bounded_channel_throughput() -> BenchmarkResult:
    """Benchmark bounded channel throughput."""
    let iterations = 10000
    let (tx, rx) = bounded_channel[Int](100)
    
    # Spawn producer
    let producer = spawn(async {
        for i in range(iterations):
            await tx.send(i)?
        return Ok(None)
    })
    
    # Spawn consumer
    var timer = Stopwatch()
    timer.start()
    
    let consumer = spawn(async {
        for i in range(iterations):
            _ = await rx.recv()?
        return Ok(None)
    })
    
    _ = await producer.join()
    _ = await consumer.join()
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Bounded Channel Throughput", iterations, elapsed)


async fn bench_unbounded_channel_throughput() -> BenchmarkResult:
    """Benchmark unbounded channel throughput."""
    let iterations = 10000
    let (tx, rx) = unbounded_channel[Int]()
    
    # Producer
    let producer = spawn(async {
        for i in range(iterations):
            await tx.send(i)?
        return Ok(None)
    })
    
    # Consumer with timing
    var timer = Stopwatch()
    timer.start()
    
    let consumer = spawn(async {
        for i in range(iterations):
            _ = await rx.recv()?
        return Ok(None)
    })
    
    _ = await producer.join()
    _ = await consumer.join()
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Unbounded Channel Throughput", iterations, elapsed)


# ============================================================================
# Benchmark 3: Mutex Contention
# ============================================================================

async fn bench_mutex_contention() -> BenchmarkResult:
    """Benchmark mutex under contention."""
    let iterations = 1000
    let num_tasks = 10
    let mutex = AsyncMutex[Int](0)
    
    var timer = Stopwatch()
    timer.start()
    
    var handles = List[JoinHandle[None]]()
    for _ in range(num_tasks):
        handles.append(spawn(async {
            for _ in range(iterations):
                let guard = await mutex.lock()
                guard.set(guard.get() + 1)
            return None
        }))
    
    _ = await join_all(handles)
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Mutex Contention", 
                          iterations * num_tasks, elapsed)


# ============================================================================
# Benchmark 4: Semaphore Throughput
# ============================================================================

async fn bench_semaphore_acquire_release() -> BenchmarkResult:
    """Benchmark semaphore acquire/release."""
    let iterations = 10000
    let sem = AsyncSemaphore(1)
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        let permit = await sem.acquire()
        # Permit auto-released
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Semaphore Acquire/Release", iterations, elapsed)


# ============================================================================
# Benchmark 5: Stream Processing
# ============================================================================

async fn bench_stream_map_filter() -> BenchmarkResult:
    """Benchmark stream map and filter."""
    let iterations = 100
    var total_processed = 0
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        var stream = stream_from_range(1, 1001)
            .filter(fn(x: Int) -> Bool { return x % 2 == 0 })
            .map(fn(x: Int) -> Int { return x * 2 })
        
        let results = await stream.collect()
        total_processed += len(results)
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Stream Map+Filter", total_processed, elapsed)


async fn bench_stream_fold() -> BenchmarkResult:
    """Benchmark stream fold operation."""
    let iterations = 1000
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        var stream = stream_from_range(1, 101)
        let sum = await stream.fold(0, fn(acc: Int, x: Int) -> Int {
            return acc + x
        })
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Stream Fold", iterations * 100, elapsed)


# ============================================================================
# Benchmark 6: Concurrent I/O
# ============================================================================

async fn bench_concurrent_file_reads() -> BenchmarkResult:
    """Benchmark concurrent file reading."""
    let num_files = 10
    let iterations = 100
    
    # Create test files (simulated)
    var files = List[String]()
    for i in range(num_files):
        files.append("/tmp/bench_file_" + str(i) + ".txt")
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        var handles = List[JoinHandle[Result[String, IOError]]]()
        for path in files:
            handles.append(spawn(async {
                return await read_file(path)
            }))
        
        _ = await join_all(handles)
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Concurrent File Reads",
                          iterations * num_files, elapsed)


# ============================================================================
# Benchmark 7: Task Switching
# ============================================================================

async fn bench_task_switching() -> BenchmarkResult:
    """Benchmark task switching overhead."""
    let iterations = 10000
    
    async fn yield_task():
        """Task that yields."""
        # Simulated yield
        pass
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        await yield_task()
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Task Switching", iterations, elapsed)


# ============================================================================
# Benchmark 8: Memory Allocation
# ============================================================================

async fn bench_future_allocation() -> BenchmarkResult:
    """Benchmark future allocation overhead."""
    let iterations = 10000
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        let fut = Future[Int].ready(42)
        _ = await fut.poll()
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Future Allocation", iterations, elapsed)


# ============================================================================
# Benchmark 9: Error Handling
# ============================================================================

async fn bench_error_propagation() -> BenchmarkResult:
    """Benchmark error propagation overhead."""
    let iterations = 10000
    
    async fn returns_error() -> Result[Int, String]:
        return Err("Error")
    
    async fn propagates_error() -> Result[Int, String]:
        let value = await returns_error()?
        return Ok(value)
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        _ = await propagates_error()
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Error Propagation", iterations, elapsed)


# ============================================================================
# Benchmark 10: Complex Workload
# ============================================================================

async fn bench_complex_workload() -> BenchmarkResult:
    """Benchmark complex real-world workload."""
    let iterations = 100
    
    async fn complex_task(id: Int) -> Int:
        """Simulated complex task."""
        let mutex = AsyncMutex[Int](0)
        let guard = await mutex.lock()
        guard.set(id)
        
        var stream = stream_from_range(1, 11)
        let sum = await stream.fold(0, fn(acc: Int, x: Int) -> Int {
            return acc + x
        })
        
        return sum + guard.get()
    
    var timer = Stopwatch()
    timer.start()
    
    for _ in range(iterations):
        var handles = List[JoinHandle[Int]]()
        for i in range(10):
            handles.append(spawn(async {
                return await complex_task(i)
            }))
        
        _ = await join_all(handles)
    
    let elapsed = timer.elapsed_ns()
    return BenchmarkResult("Complex Workload",
                          iterations * 10, elapsed)


# ============================================================================
# Benchmark Runner
# ============================================================================

async fn run_benchmarks() -> BenchmarkSuite:
    """Run all benchmarks."""
    var suite = BenchmarkSuite("Async Performance Benchmarks")
    
    print("Running benchmarks...")
    
    # Task spawning
    suite.add_result(await bench_spawn_tasks())
    suite.add_result(await bench_spawn_and_join())
    
    # Channels
    suite.add_result(await bench_bounded_channel_throughput())
    suite.add_result(await bench_unbounded_channel_throughput())
    
    # Synchronization
    suite.add_result(await bench_mutex_contention())
    suite.add_result(await bench_semaphore_acquire_release())
    
    # Streams
    suite.add_result(await bench_stream_map_filter())
    suite.add_result(await bench_stream_fold())
    
    # I/O
    suite.add_result(await bench_concurrent_file_reads())
    
    # Overhead
    suite.add_result(await bench_task_switching())
    suite.add_result(await bench_future_allocation())
    suite.add_result(await bench_error_propagation())
    
    # Complex
    suite.add_result(await bench_complex_workload())
    
    return suite


fn main():
    """Main benchmark runner."""
    print("=" * 60)
    print("Async System Performance Benchmarks")
    print("=" * 60)
    
    let suite = await run_benchmarks()
    suite.print_summary()
    
    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)
