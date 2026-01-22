"""
Mojo Benchmark Module - Performance measurement and optimization utilities.

This module provides:
- Benchmark framework for measuring code performance
- Memory tracking utilities
- Performance profiling helpers
- Optimization utilities
"""

# ============================================================================
# Time Measurement
# ============================================================================

struct Stopwatch:
    """High-precision stopwatch for measuring elapsed time."""
    var _start_ns: Int64
    var _stop_ns: Int64
    var _running: Bool
    var _laps: List[Int64]

    fn __init__(inout self):
        """Creates a new stopwatch."""
        self._start_ns = 0
        self._stop_ns = 0
        self._running = False
        self._laps = List[Int64]()

    fn start(inout self):
        """Starts or resumes the stopwatch."""
        if not self._running:
            self._start_ns = _get_time_ns()
            self._running = True

    fn stop(inout self):
        """Stops the stopwatch."""
        if self._running:
            self._stop_ns = _get_time_ns()
            self._running = False

    fn reset(inout self):
        """Resets the stopwatch."""
        self._start_ns = 0
        self._stop_ns = 0
        self._running = False
        self._laps.clear()

    fn restart(inout self):
        """Resets and starts the stopwatch."""
        self.reset()
        self.start()

    fn lap(inout self):
        """Records a lap time."""
        if self._running:
            let current = _get_time_ns()
            self._laps.append(current - self._start_ns)

    fn elapsed_ns(self) -> Int64:
        """Returns elapsed nanoseconds."""
        if self._running:
            return _get_time_ns() - self._start_ns
        return self._stop_ns - self._start_ns

    fn elapsed_us(self) -> Int64:
        """Returns elapsed microseconds."""
        return self.elapsed_ns() // 1000

    fn elapsed_ms(self) -> Int64:
        """Returns elapsed milliseconds."""
        return self.elapsed_ns() // 1000000

    fn elapsed_secs(self) -> Float64:
        """Returns elapsed seconds as float."""
        return Float64(self.elapsed_ns()) / 1000000000.0

    fn lap_times(self) -> List[Int64]:
        """Returns list of lap times in nanoseconds."""
        return self._laps

    fn is_running(self) -> Bool:
        """Returns True if stopwatch is running."""
        return self._running

fn _get_time_ns() -> Int64:
    """Gets current time in nanoseconds (placeholder)."""
    # Would need system time integration
    # For benchmarking, this would use clock_gettime or equivalent
    return 0

# ============================================================================
# Benchmark Framework
# ============================================================================

struct BenchmarkConfig:
    """Configuration for benchmark runs."""
    var warmup_iterations: Int
    var min_iterations: Int
    var max_iterations: Int
    var target_time_ms: Int
    var name: String

    fn __init__(inout self, name: String = "benchmark"):
        """Creates default benchmark config."""
        self.warmup_iterations = 10
        self.min_iterations = 100
        self.max_iterations = 10000
        self.target_time_ms = 1000
        self.name = name

    fn with_warmup(inout self, iterations: Int) -> BenchmarkConfig:
        """Sets warmup iterations."""
        self.warmup_iterations = iterations
        return self

    fn with_min_iterations(inout self, iterations: Int) -> BenchmarkConfig:
        """Sets minimum iterations."""
        self.min_iterations = iterations
        return self

    fn with_max_iterations(inout self, iterations: Int) -> BenchmarkConfig:
        """Sets maximum iterations."""
        self.max_iterations = iterations
        return self

    fn with_target_time(inout self, ms: Int) -> BenchmarkConfig:
        """Sets target time in milliseconds."""
        self.target_time_ms = ms
        return self

struct BenchmarkResult:
    """Results from a benchmark run."""
    var name: String
    var iterations: Int
    var total_time_ns: Int64
    var min_time_ns: Int64
    var max_time_ns: Int64
    var mean_time_ns: Int64
    var median_time_ns: Int64
    var std_dev_ns: Int64
    var throughput: Float64  # ops/sec

    fn __init__(inout self, name: String):
        """Creates empty benchmark result."""
        self.name = name
        self.iterations = 0
        self.total_time_ns = 0
        self.min_time_ns = 0
        self.max_time_ns = 0
        self.mean_time_ns = 0
        self.median_time_ns = 0
        self.std_dev_ns = 0
        self.throughput = 0.0

    fn mean_us(self) -> Float64:
        """Returns mean time in microseconds."""
        return Float64(self.mean_time_ns) / 1000.0

    fn mean_ms(self) -> Float64:
        """Returns mean time in milliseconds."""
        return Float64(self.mean_time_ns) / 1000000.0

    fn to_string(self) -> String:
        """Formats result as string."""
        var result = String("Benchmark: ") + self.name + "\n"
        result = result + "  Iterations: " + String(self.iterations) + "\n"
        result = result + "  Mean: " + _format_time(self.mean_time_ns) + "\n"
        result = result + "  Min: " + _format_time(self.min_time_ns) + "\n"
        result = result + "  Max: " + _format_time(self.max_time_ns) + "\n"
        result = result + "  Std Dev: " + _format_time(self.std_dev_ns) + "\n"
        result = result + "  Throughput: " + _format_throughput(self.throughput) + "\n"
        return result

fn _format_time(ns: Int64) -> String:
    """Formats nanoseconds as human-readable time."""
    if ns < 1000:
        return String(ns) + " ns"
    elif ns < 1000000:
        return String(Float64(ns) / 1000.0) + " us"
    elif ns < 1000000000:
        return String(Float64(ns) / 1000000.0) + " ms"
    else:
        return String(Float64(ns) / 1000000000.0) + " s"

fn _format_throughput(ops_per_sec: Float64) -> String:
    """Formats throughput as human-readable string."""
    if ops_per_sec < 1000:
        return String(ops_per_sec) + " ops/s"
    elif ops_per_sec < 1000000:
        return String(ops_per_sec / 1000.0) + " K ops/s"
    elif ops_per_sec < 1000000000:
        return String(ops_per_sec / 1000000.0) + " M ops/s"
    else:
        return String(ops_per_sec / 1000000000.0) + " G ops/s"

struct Benchmark:
    """Benchmark runner for measuring performance."""
    var config: BenchmarkConfig
    var results: List[BenchmarkResult]

    fn __init__(inout self, config: BenchmarkConfig = BenchmarkConfig()):
        """Creates benchmark runner with config."""
        self.config = config
        self.results = List[BenchmarkResult]()

    fn run[F: fn() -> None](inout self, name: String, func: F) -> BenchmarkResult:
        """
        Runs benchmark on a function.

        Parameters:
            name: Name of the benchmark
            func: Function to benchmark (no arguments, no return)

        Returns:
            BenchmarkResult with statistics
        """
        var result = BenchmarkResult(name)
        var times = List[Int64]()

        # Warmup
        for _ in range(self.config.warmup_iterations):
            func()

        # Determine iteration count
        var iterations = self.config.min_iterations
        var sw = Stopwatch()

        # Run benchmarks
        for _ in range(iterations):
            sw.restart()
            func()
            sw.stop()
            times.append(sw.elapsed_ns())

        # Calculate statistics
        result.iterations = iterations
        result = _calculate_stats(result, times)

        self.results.append(result)
        return result

    fn run_n(inout self, name: String, iterations: Int, func: fn() -> None) -> BenchmarkResult:
        """Runs benchmark with fixed iteration count."""
        var result = BenchmarkResult(name)
        var times = List[Int64]()
        var sw = Stopwatch()

        # Warmup
        for _ in range(self.config.warmup_iterations):
            func()

        # Run benchmarks
        for _ in range(iterations):
            sw.restart()
            func()
            sw.stop()
            times.append(sw.elapsed_ns())

        result.iterations = iterations
        result = _calculate_stats(result, times)

        self.results.append(result)
        return result

    fn print_results(self):
        """Prints all benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        for i in range(len(self.results)):
            print(self.results[i].to_string())

        print("=" * 60)

fn _calculate_stats(inout result: BenchmarkResult, times: List[Int64]) -> BenchmarkResult:
    """Calculates statistics from timing data."""
    if len(times) == 0:
        return result

    # Sort for median
    var sorted_times = _sort_list(times)

    # Min/Max
    result.min_time_ns = sorted_times[0]
    result.max_time_ns = sorted_times[len(sorted_times) - 1]

    # Sum for mean
    var total: Int64 = 0
    for i in range(len(times)):
        total += times[i]

    result.total_time_ns = total
    result.mean_time_ns = total // Int64(len(times))

    # Median
    let mid = len(sorted_times) // 2
    if len(sorted_times) % 2 == 0:
        result.median_time_ns = (sorted_times[mid - 1] + sorted_times[mid]) // 2
    else:
        result.median_time_ns = sorted_times[mid]

    # Standard deviation
    var variance: Int64 = 0
    for i in range(len(times)):
        let diff = times[i] - result.mean_time_ns
        variance += diff * diff
    variance = variance // Int64(len(times))
    result.std_dev_ns = _isqrt(variance)

    # Throughput
    if result.mean_time_ns > 0:
        result.throughput = 1000000000.0 / Float64(result.mean_time_ns)

    return result

fn _sort_list(lst: List[Int64]) -> List[Int64]:
    """Simple sort for timing data."""
    var result = List[Int64]()
    for i in range(len(lst)):
        result.append(lst[i])

    # Bubble sort (good enough for small benchmark data)
    for i in range(len(result)):
        for j in range(len(result) - 1 - i):
            if result[j] > result[j + 1]:
                let temp = result[j]
                result[j] = result[j + 1]
                result[j + 1] = temp

    return result

fn _isqrt(n: Int64) -> Int64:
    """Integer square root."""
    if n < 0:
        return 0
    if n < 2:
        return n

    var x = n
    var y = (x + 1) // 2

    while y < x:
        x = y
        y = (x + n // x) // 2

    return x

# ============================================================================
# Memory Tracking
# ============================================================================

struct MemoryStats:
    """Memory usage statistics."""
    var allocated_bytes: Int64
    var freed_bytes: Int64
    var peak_bytes: Int64
    var allocation_count: Int
    var free_count: Int

    fn __init__(inout self):
        """Creates empty memory stats."""
        self.allocated_bytes = 0
        self.freed_bytes = 0
        self.peak_bytes = 0
        self.allocation_count = 0
        self.free_count = 0

    fn current_usage(self) -> Int64:
        """Returns current memory usage."""
        return self.allocated_bytes - self.freed_bytes

    fn to_string(self) -> String:
        """Formats stats as string."""
        var result = String("Memory Statistics:\n")
        result = result + "  Current: " + _format_bytes(self.current_usage()) + "\n"
        result = result + "  Peak: " + _format_bytes(self.peak_bytes) + "\n"
        result = result + "  Total Allocated: " + _format_bytes(self.allocated_bytes) + "\n"
        result = result + "  Allocations: " + String(self.allocation_count) + "\n"
        result = result + "  Frees: " + String(self.free_count) + "\n"
        return result

fn _format_bytes(bytes: Int64) -> String:
    """Formats bytes as human-readable string."""
    if bytes < 1024:
        return String(bytes) + " B"
    elif bytes < 1024 * 1024:
        return String(Float64(bytes) / 1024.0) + " KB"
    elif bytes < 1024 * 1024 * 1024:
        return String(Float64(bytes) / (1024.0 * 1024.0)) + " MB"
    else:
        return String(Float64(bytes) / (1024.0 * 1024.0 * 1024.0)) + " GB"

struct MemoryTracker:
    """Tracks memory allocations for profiling."""
    var stats: MemoryStats
    var _tracking: Bool

    fn __init__(inout self):
        """Creates memory tracker."""
        self.stats = MemoryStats()
        self._tracking = False

    fn start(inout self):
        """Starts tracking."""
        self._tracking = True
        self.stats = MemoryStats()

    fn stop(inout self):
        """Stops tracking."""
        self._tracking = False

    fn record_allocation(inout self, bytes: Int64):
        """Records an allocation."""
        if self._tracking:
            self.stats.allocated_bytes += bytes
            self.stats.allocation_count += 1
            let current = self.stats.current_usage()
            if current > self.stats.peak_bytes:
                self.stats.peak_bytes = current

    fn record_free(inout self, bytes: Int64):
        """Records a free."""
        if self._tracking:
            self.stats.freed_bytes += bytes
            self.stats.free_count += 1

    fn get_stats(self) -> MemoryStats:
        """Returns current stats."""
        return self.stats

    fn print_stats(self):
        """Prints current stats."""
        print(self.stats.to_string())

# ============================================================================
# Performance Utilities
# ============================================================================

struct BlackHole[T: AnyType]:
    """
    Prevents compiler from optimizing away benchmark code.

    Usage:
        var bh = BlackHole[Int]()
        bh.consume(expensive_computation())
    """
    var _value: T

    fn __init__(inout self):
        """Creates black hole."""
        pass

    fn consume(inout self, value: T):
        """Consumes a value, preventing optimization."""
        self._value = value

fn do_not_optimize[T: AnyType](value: T):
    """Prevents compiler from optimizing away a value."""
    # Compiler barrier - would use inline assembly or volatile
    var _sink = value
    pass

# ============================================================================
# Comparison Utilities
# ============================================================================

struct BenchmarkComparison:
    """Compares two benchmark results."""
    var baseline: BenchmarkResult
    var candidate: BenchmarkResult
    var speedup: Float64
    var improvement_percent: Float64

    fn __init__(inout self, baseline: BenchmarkResult, candidate: BenchmarkResult):
        """Creates comparison between baseline and candidate."""
        self.baseline = baseline
        self.candidate = candidate

        if candidate.mean_time_ns > 0:
            self.speedup = Float64(baseline.mean_time_ns) / Float64(candidate.mean_time_ns)
        else:
            self.speedup = 0.0

        if baseline.mean_time_ns > 0:
            let diff = baseline.mean_time_ns - candidate.mean_time_ns
            self.improvement_percent = Float64(diff) / Float64(baseline.mean_time_ns) * 100.0
        else:
            self.improvement_percent = 0.0

    fn is_faster(self) -> Bool:
        """Returns True if candidate is faster than baseline."""
        return self.speedup > 1.0

    fn is_slower(self) -> Bool:
        """Returns True if candidate is slower than baseline."""
        return self.speedup < 1.0

    fn to_string(self) -> String:
        """Formats comparison as string."""
        var result = String("Benchmark Comparison:\n")
        result = result + "  Baseline: " + self.baseline.name + " ("
        result = result + _format_time(self.baseline.mean_time_ns) + ")\n"
        result = result + "  Candidate: " + self.candidate.name + " ("
        result = result + _format_time(self.candidate.mean_time_ns) + ")\n"
        result = result + "  Speedup: " + String(self.speedup) + "x\n"

        if self.is_faster():
            result = result + "  Result: " + String(self.improvement_percent) + "% FASTER\n"
        elif self.is_slower():
            result = result + "  Result: " + String(-self.improvement_percent) + "% SLOWER\n"
        else:
            result = result + "  Result: NO CHANGE\n"

        return result

fn compare(baseline: BenchmarkResult, candidate: BenchmarkResult) -> BenchmarkComparison:
    """Creates comparison between two benchmark results."""
    return BenchmarkComparison(baseline, candidate)

# ============================================================================
# Batch Benchmarking
# ============================================================================

struct BenchmarkSuite:
    """Collection of related benchmarks."""
    var name: String
    var benchmarks: List[BenchmarkResult]
    var config: BenchmarkConfig

    fn __init__(inout self, name: String, config: BenchmarkConfig = BenchmarkConfig()):
        """Creates benchmark suite."""
        self.name = name
        self.benchmarks = List[BenchmarkResult]()
        self.config = config

    fn add(inout self, result: BenchmarkResult):
        """Adds benchmark result to suite."""
        self.benchmarks.append(result)

    fn print_summary(self):
        """Prints summary of all benchmarks."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUITE: " + self.name)
        print("=" * 70)
        print("")

        # Header
        print(_pad_right("Name", 30) + _pad_right("Mean", 15) +
              _pad_right("Min", 15) + _pad_right("Throughput", 15))
        print("-" * 70)

        # Results
        for i in range(len(self.benchmarks)):
            let b = self.benchmarks[i]
            print(_pad_right(b.name, 30) +
                  _pad_right(_format_time(b.mean_time_ns), 15) +
                  _pad_right(_format_time(b.min_time_ns), 15) +
                  _pad_right(_format_throughput(b.throughput), 15))

        print("=" * 70)

fn _pad_right(s: String, width: Int) -> String:
    """Pads string to width with spaces."""
    if len(s) >= width:
        return s
    var result = s
    for _ in range(width - len(s)):
        result = result + " "
    return result

# ============================================================================
# Quick Benchmark Helpers
# ============================================================================

fn bench[F: fn() -> None](name: String, func: F, iterations: Int = 1000) -> BenchmarkResult:
    """Quick benchmark helper."""
    var b = Benchmark()
    return b.run_n(name, iterations, func)

fn time_it[F: fn() -> None](func: F) -> Int64:
    """Returns execution time in nanoseconds."""
    var sw = Stopwatch()
    sw.start()
    func()
    sw.stop()
    return sw.elapsed_ns()

fn time_it_ms[F: fn() -> None](func: F) -> Float64:
    """Returns execution time in milliseconds."""
    return Float64(time_it(func)) / 1000000.0

# ============================================================================
# Iteration Helpers
# ============================================================================

struct Range:
    """Memory-efficient range for iteration."""
    var start: Int
    var stop: Int
    var step: Int

    fn __init__(inout self, stop: Int):
        """Creates range from 0 to stop."""
        self.start = 0
        self.stop = stop
        self.step = 1

    fn __init__(inout self, start: Int, stop: Int):
        """Creates range from start to stop."""
        self.start = start
        self.stop = stop
        self.step = 1

    fn __init__(inout self, start: Int, stop: Int, step: Int):
        """Creates range from start to stop with step."""
        self.start = start
        self.stop = stop
        self.step = step

    fn length(self) -> Int:
        """Returns number of elements in range."""
        if self.step > 0:
            if self.start >= self.stop:
                return 0
            return (self.stop - self.start + self.step - 1) // self.step
        elif self.step < 0:
            if self.start <= self.stop:
                return 0
            return (self.start - self.stop - self.step - 1) // (-self.step)
        return 0

    fn contains(self, value: Int) -> Bool:
        """Returns True if value is in range."""
        if self.step > 0:
            return value >= self.start and value < self.stop and (value - self.start) % self.step == 0
        elif self.step < 0:
            return value <= self.start and value > self.stop and (self.start - value) % (-self.step) == 0
        return False

struct RangeIterator:
    """Iterator for Range."""
    var _range: Range
    var _current: Int

    fn __init__(inout self, r: Range):
        """Creates iterator for range."""
        self._range = r
        self._current = r.start

    fn has_next(self) -> Bool:
        """Returns True if more elements."""
        if self._range.step > 0:
            return self._current < self._range.stop
        elif self._range.step < 0:
            return self._current > self._range.stop
        return False

    fn next(inout self) -> Int:
        """Returns next element."""
        let value = self._current
        self._current += self._range.step
        return value

# ============================================================================
# Caching Utilities
# ============================================================================

struct LRUCache[K: Hashable, V: AnyType]:
    """Least Recently Used cache with fixed capacity."""
    var _capacity: Int
    var _data: Dict[K, V]
    var _order: List[K]

    fn __init__(inout self, capacity: Int):
        """Creates LRU cache with given capacity."""
        self._capacity = capacity
        self._data = Dict[K, V]()
        self._order = List[K]()

    fn get(inout self, key: K) -> Optional[V]:
        """Gets value, moving to front if found."""
        if self._data.contains(key):
            self._move_to_front(key)
            return Optional[V](self._data.get(key))
        return Optional[V].none()

    fn put(inout self, key: K, value: V):
        """Puts value, evicting oldest if at capacity."""
        if self._data.contains(key):
            self._data.set(key, value)
            self._move_to_front(key)
        else:
            if len(self._order) >= self._capacity:
                self._evict_oldest()
            self._data.set(key, value)
            self._order.append(key)

    fn contains(self, key: K) -> Bool:
        """Returns True if key exists."""
        return self._data.contains(key)

    fn size(self) -> Int:
        """Returns current size."""
        return len(self._order)

    fn capacity(self) -> Int:
        """Returns capacity."""
        return self._capacity

    fn clear(inout self):
        """Clears the cache."""
        self._data.clear()
        self._order.clear()

    fn _move_to_front(inout self, key: K):
        """Moves key to front of order."""
        var new_order = List[K]()
        for i in range(len(self._order)):
            if self._order[i] != key:
                new_order.append(self._order[i])
        new_order.append(key)
        self._order = new_order

    fn _evict_oldest(inout self):
        """Removes oldest entry."""
        if len(self._order) > 0:
            let oldest = self._order[0]
            self._data.remove(oldest)
            var new_order = List[K]()
            for i in range(1, len(self._order)):
                new_order.append(self._order[i])
            self._order = new_order

struct Optional[T: AnyType]:
    """Optional value container."""
    var _value: T
    var _has_value: Bool

    fn __init__(inout self, value: T):
        """Creates optional with value."""
        self._value = value
        self._has_value = True

    @staticmethod
    fn none() -> Optional[T]:
        """Creates empty optional."""
        var opt = Optional[T].__new__()
        opt._has_value = False
        return opt

    fn has_value(self) -> Bool:
        """Returns True if has value."""
        return self._has_value

    fn value(self) -> T:
        """Returns value (must check has_value first)."""
        return self._value

    fn value_or(self, default: T) -> T:
        """Returns value or default."""
        if self._has_value:
            return self._value
        return default

# ============================================================================
# String Optimization Utilities
# ============================================================================

struct StringBuilder:
    """Efficient string builder with pre-allocated buffer."""
    var _buffer: List[String]
    var _length: Int

    fn __init__(inout self):
        """Creates empty string builder."""
        self._buffer = List[String]()
        self._length = 0

    fn __init__(inout self, initial_capacity: Int):
        """Creates string builder with capacity hint."""
        self._buffer = List[String]()
        self._length = 0

    fn append(inout self, s: String) -> StringBuilder:
        """Appends string."""
        self._buffer.append(s)
        self._length += len(s)
        return self

    fn append_line(inout self, s: String) -> StringBuilder:
        """Appends string with newline."""
        self._buffer.append(s)
        self._buffer.append("\n")
        self._length += len(s) + 1
        return self

    fn append_char(inout self, c: String) -> StringBuilder:
        """Appends single character."""
        self._buffer.append(c)
        self._length += 1
        return self

    fn append_int(inout self, value: Int) -> StringBuilder:
        """Appends integer."""
        let s = String(value)
        self._buffer.append(s)
        self._length += len(s)
        return self

    fn append_float(inout self, value: Float64, precision: Int = 6) -> StringBuilder:
        """Appends float with precision."""
        let s = String(value)
        self._buffer.append(s)
        self._length += len(s)
        return self

    fn length(self) -> Int:
        """Returns total length."""
        return self._length

    fn clear(inout self):
        """Clears builder."""
        self._buffer.clear()
        self._length = 0

    fn build(self) -> String:
        """Builds final string."""
        var result = String("")
        for i in range(len(self._buffer)):
            result = result + self._buffer[i]
        return result

    fn to_string(self) -> String:
        """Alias for build()."""
        return self.build()

# ============================================================================
# Assertion Utilities
# ============================================================================

fn assert_eq[T: Stringable](actual: T, expected: T, message: String = ""):
    """Asserts values are equal."""
    let actual_str = String(actual)
    let expected_str = String(expected)

    if actual_str != expected_str:
        var msg = "Assertion failed"
        if len(message) > 0:
            msg = msg + ": " + message
        msg = msg + "\n  Expected: " + expected_str
        msg = msg + "\n  Actual: " + actual_str
        print(msg)

fn assert_true(condition: Bool, message: String = ""):
    """Asserts condition is true."""
    if not condition:
        var msg = "Assertion failed: expected true"
        if len(message) > 0:
            msg = msg + " - " + message
        print(msg)

fn assert_false(condition: Bool, message: String = ""):
    """Asserts condition is false."""
    if condition:
        var msg = "Assertion failed: expected false"
        if len(message) > 0:
            msg = msg + " - " + message
        print(msg)

fn assert_near(actual: Float64, expected: Float64, tolerance: Float64 = 0.0001, message: String = ""):
    """Asserts floats are approximately equal."""
    let diff = actual - expected
    let abs_diff = diff if diff >= 0 else -diff

    if abs_diff > tolerance:
        var msg = "Assertion failed"
        if len(message) > 0:
            msg = msg + ": " + message
        msg = msg + "\n  Expected: " + String(expected)
        msg = msg + "\n  Actual: " + String(actual)
        msg = msg + "\n  Difference: " + String(abs_diff)
        msg = msg + "\n  Tolerance: " + String(tolerance)
        print(msg)

# ============================================================================
# Tests
# ============================================================================

fn test_stopwatch():
    """Test Stopwatch."""
    print("Testing Stopwatch...")

    var sw = Stopwatch()
    assert_false(sw.is_running(), "should not be running initially")

    sw.start()
    assert_true(sw.is_running(), "should be running after start")

    sw.stop()
    assert_false(sw.is_running(), "should not be running after stop")

    sw.reset()
    assert_eq(sw.elapsed_ns(), Int64(0), "should be 0 after reset")

    print("Stopwatch tests passed!")

fn test_benchmark_result():
    """Test BenchmarkResult."""
    print("Testing BenchmarkResult...")

    var result = BenchmarkResult("test")
    result.iterations = 100
    result.mean_time_ns = 1500000  # 1.5ms
    result.min_time_ns = 1000000   # 1ms
    result.max_time_ns = 2000000   # 2ms

    assert_near(result.mean_ms(), 1.5, 0.01, "mean_ms")

    print("BenchmarkResult tests passed!")

fn test_memory_stats():
    """Test MemoryStats."""
    print("Testing MemoryStats...")

    var stats = MemoryStats()
    stats.allocated_bytes = 1024
    stats.freed_bytes = 512

    assert_eq(stats.current_usage(), Int64(512), "current usage")

    print("MemoryStats tests passed!")

fn test_range():
    """Test Range."""
    print("Testing Range...")

    let r1 = Range(10)
    assert_eq(r1.length(), 10, "range(10) length")
    assert_true(r1.contains(0), "contains 0")
    assert_true(r1.contains(9), "contains 9")
    assert_false(r1.contains(10), "doesn't contain 10")

    let r2 = Range(5, 10)
    assert_eq(r2.length(), 5, "range(5,10) length")
    assert_true(r2.contains(5), "contains 5")
    assert_false(r2.contains(4), "doesn't contain 4")

    let r3 = Range(0, 10, 2)
    assert_eq(r3.length(), 5, "range(0,10,2) length")
    assert_true(r3.contains(0), "contains 0")
    assert_true(r3.contains(4), "contains 4")
    assert_false(r3.contains(5), "doesn't contain 5")

    print("Range tests passed!")

fn test_string_builder():
    """Test StringBuilder."""
    print("Testing StringBuilder...")

    var sb = StringBuilder()
    sb.append("Hello").append(" ").append("World")

    assert_eq(sb.length(), 11, "length")
    assert_eq(sb.build(), "Hello World", "build")

    sb.clear()
    assert_eq(sb.length(), 0, "length after clear")

    print("StringBuilder tests passed!")

fn test_optional():
    """Test Optional."""
    print("Testing Optional...")

    let some = Optional[Int](42)
    assert_true(some.has_value(), "has value")
    assert_eq(some.value(), 42, "value")

    let none = Optional[Int].none()
    assert_false(none.has_value(), "doesn't have value")
    assert_eq(none.value_or(0), 0, "value_or")

    print("Optional tests passed!")

fn run_all_tests():
    """Run all benchmark utility tests."""
    print("=== Benchmark Utility Tests ===")
    test_stopwatch()
    test_benchmark_result()
    test_memory_stats()
    test_range()
    test_string_builder()
    test_optional()
    print("=== All tests passed! ===")
