# Testing/Testing - Test Framework
# Day 39: Test framework, assertions, benchmarking

from builtin import Int, Float64, Bool, String
from collections.list import List
from io import print
from math.random import Random


# Test result tracking

struct TestResult:
    """Result of a single test."""
    var name: String
    var passed: Bool
    var message: String
    var duration_ms: Float64
    
    fn __init__(inout self, name: String, passed: Bool, message: String = "", duration_ms: Float64 = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


struct TestSuite:
    """Collection of test results."""
    var name: String
    var results: List[TestResult]
    var setup_fn: fn() -> None
    var teardown_fn: fn() -> None
    
    fn __init__(inout self, name: String):
        self.name = name
        self.results = List[TestResult]()
        self.setup_fn = default_setup
        self.teardown_fn = default_teardown
    
    fn add_result(inout self, result: TestResult):
        """Add test result to suite."""
        self.results.append(result)
    
    fn passed_count(self) -> Int:
        """Count of passed tests."""
        var count = 0
        for i in range(len(self.results)):
            if self.results[i].passed:
                count += 1
        return count
    
    fn failed_count(self) -> Int:
        """Count of failed tests."""
        return len(self.results) - self.passed_count()
    
    fn total_duration(self) -> Float64:
        """Total duration of all tests."""
        var total = 0.0
        for i in range(len(self.results)):
            total += self.results[i].duration_ms
        return total
    
    fn print_summary(self):
        """Print test suite summary."""
        let total = len(self.results)
        let passed = self.passed_count()
        let failed = self.failed_count()
        let duration = self.total_duration()
        
        print("\n" + "=" * 60)
        print("Test Suite: " + self.name)
        print("=" * 60)
        print("Total:  " + String(total) + " tests")
        print("Passed: " + String(passed) + " tests")
        print("Failed: " + String(failed) + " tests")
        print("Time:   " + String(duration) + " ms")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed Tests:")
            for i in range(len(self.results)):
                if not self.results[i].passed:
                    print("  - " + self.results[i].name)
                    if self.results[i].message:
                        print("    " + self.results[i].message)


fn default_setup():
    """Default setup function (does nothing)."""
    pass


fn default_teardown():
    """Default teardown function (does nothing)."""
    pass


# Assertion functions

fn assert_true(condition: Bool, message: String = "Assertion failed"):
    """Assert condition is true.
    
    Args:
        condition: Condition to check
        message: Error message if assertion fails
    
    Raises:
        AssertionError if condition is false
    """
    if not condition:
        raise AssertionError(message)


fn assert_false(condition: Bool, message: String = "Assertion failed"):
    """Assert condition is false.
    
    Args:
        condition: Condition to check
        message: Error message if assertion fails
    """
    if condition:
        raise AssertionError(message)


fn assert_equal[T](actual: T, expected: T, message: String = ""):
    """Assert two values are equal.
    
    Args:
        actual: Actual value
        expected: Expected value
        message: Optional error message
    """
    if actual != expected:
        let msg = message if message else "Expected " + String(expected) + " but got " + String(actual)
        raise AssertionError(msg)


fn assert_not_equal[T](actual: T, unexpected: T, message: String = ""):
    """Assert two values are not equal.
    
    Args:
        actual: Actual value
        unexpected: Value that should not match
        message: Optional error message
    """
    if actual == unexpected:
        let msg = message if message else "Values should not be equal: " + String(actual)
        raise AssertionError(msg)


fn assert_less[T](actual: T, bound: T, message: String = ""):
    """Assert value is less than bound.
    
    Args:
        actual: Value to check
        bound: Upper bound
        message: Optional error message
    """
    if not (actual < bound):
        let msg = message if message else String(actual) + " should be less than " + String(bound)
        raise AssertionError(msg)


fn assert_less_equal[T](actual: T, bound: T, message: String = ""):
    """Assert value is less than or equal to bound.
    
    Args:
        actual: Value to check
        bound: Upper bound
        message: Optional error message
    """
    if not (actual <= bound):
        let msg = message if message else String(actual) + " should be <= " + String(bound)
        raise AssertionError(msg)


fn assert_greater[T](actual: T, bound: T, message: String = ""):
    """Assert value is greater than bound.
    
    Args:
        actual: Value to check
        bound: Lower bound
        message: Optional error message
    """
    if not (actual > bound):
        let msg = message if message else String(actual) + " should be greater than " + String(bound)
        raise AssertionError(msg)


fn assert_greater_equal[T](actual: T, bound: T, message: String = ""):
    """Assert value is greater than or equal to bound.
    
    Args:
        actual: Value to check
        bound: Lower bound
        message: Optional error message
    """
    if not (actual >= bound):
        let msg = message if message else String(actual) + " should be >= " + String(bound)
        raise AssertionError(msg)


fn assert_in_range[T](value: T, min_val: T, max_val: T, message: String = ""):
    """Assert value is in range [min_val, max_val].
    
    Args:
        value: Value to check
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        message: Optional error message
    """
    if not (min_val <= value <= max_val):
        let msg = message if message else String(value) + " not in range [" + String(min_val) + ", " + String(max_val) + "]"
        raise AssertionError(msg)


fn assert_close(actual: Float64, expected: Float64, tolerance: Float64 = 1e-6, message: String = ""):
    """Assert floats are approximately equal.
    
    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Maximum allowed difference
        message: Optional error message
    """
    let diff = abs(actual - expected)
    if diff > tolerance:
        let msg = message if message else "Expected " + String(expected) + " but got " + String(actual) + " (diff: " + String(diff) + ")"
        raise AssertionError(msg)


fn assert_raises[E](fn_to_test: fn() -> None, message: String = ""):
    """Assert function raises an exception.
    
    Args:
        fn_to_test: Function that should raise
        message: Optional error message
    """
    var raised = False
    try:
        fn_to_test()
    except:
        raised = True
    
    if not raised:
        let msg = message if message else "Expected exception to be raised"
        raise AssertionError(msg)


# Benchmarking

struct BenchmarkResult:
    """Result of a benchmark run."""
    var name: String
    var iterations: Int
    var total_time_ms: Float64
    var avg_time_ms: Float64
    var min_time_ms: Float64
    var max_time_ms: Float64
    
    fn __init__(inout self, name: String, iterations: Int, times: List[Float64]):
        self.name = name
        self.iterations = iterations
        self.total_time_ms = 0.0
        self.min_time_ms = times[0]
        self.max_time_ms = times[0]
        
        for i in range(len(times)):
            let t = times[i]
            self.total_time_ms += t
            if t < self.min_time_ms:
                self.min_time_ms = t
            if t > self.max_time_ms:
                self.max_time_ms = t
        
        self.avg_time_ms = self.total_time_ms / Float64(iterations)
    
    fn print(self):
        """Print benchmark results."""
        print("\nBenchmark: " + self.name)
        print("  Iterations: " + String(self.iterations))
        print("  Total time: " + String(self.total_time_ms) + " ms")
        print("  Avg time:   " + String(self.avg_time_ms) + " ms")
        print("  Min time:   " + String(self.min_time_ms) + " ms")
        print("  Max time:   " + String(self.max_time_ms) + " ms")


fn benchmark(name: String, fn_to_bench: fn() -> None, iterations: Int = 1000) -> BenchmarkResult:
    """Run benchmark on function.
    
    Args:
        name: Benchmark name
        fn_to_bench: Function to benchmark
        iterations: Number of iterations
    
    Returns:
        Benchmark results
    """
    var times = List[Float64]()
    
    for _ in range(iterations):
        let start = time_ms()
        fn_to_bench()
        let end = time_ms()
        times.append(end - start)
    
    return BenchmarkResult(name, iterations, times)


# Mock time function (would use real timer)
var _time_counter: Float64 = 0.0

fn time_ms() -> Float64:
    """Get current time in milliseconds."""
    _time_counter += 0.1  # Simulate time passing
    return _time_counter


fn abs(x: Float64) -> Float64:
    """Absolute value."""
    return x if x >= 0.0 else -x


# Test runner

fn run_test(name: String, test_fn: fn() -> None) -> TestResult:
    """Run a single test.
    
    Args:
        name: Test name
        test_fn: Test function
    
    Returns:
        Test result
    """
    let start = time_ms()
    var passed = True
    var message = ""
    
    try:
        test_fn()
    except e:
        passed = False
        message = "Test failed: " + str(e)
    
    let duration = time_ms() - start
    
    return TestResult(name, passed, message, duration)


fn run_suite(suite: TestSuite, tests: List[(String, fn() -> None)]):
    """Run all tests in suite.
    
    Args:
        suite: Test suite to add results to
        tests: List of (name, test_fn) pairs
    """
    suite.setup_fn()
    
    for i in range(len(tests)):
        let (name, test_fn) = tests[i]
        let result = run_test(name, test_fn)
        suite.add_result(result)
        
        if result.passed:
            print("✓ " + name)
        else:
            print("✗ " + name)
            if result.message:
                print("  " + result.message)
    
    suite.teardown_fn()
    suite.print_summary()


# Parametric testing

fn parametric_test[T](name: String, test_fn: fn(T) -> None, params: List[T]) -> List[TestResult]:
    """Run parametric test with multiple inputs.
    
    Args:
        name: Base test name
        test_fn: Test function taking parameter
        params: List of parameters to test
    
    Returns:
        List of test results
    """
    var results = List[TestResult]()
    
    for i in range(len(params)):
        let param = params[i]
        let test_name = name + "[" + String(i) + "]"
        
        let result = run_test(test_name, lambda: test_fn(param))
        results.append(result)
    
    return results


# Test fixtures

struct Fixture[T]:
    """Test fixture for setup/teardown.
    
    Examples:
        ```mojo
        var fixture = Fixture[Database]()
        fixture.setup = lambda: Database()
        fixture.teardown = lambda db: db.close()
        ```
    """
    var setup: fn() -> T
    var teardown: fn(T) -> None
    
    fn __init__(inout self):
        self.setup = default_fixture_setup
        self.teardown = default_fixture_teardown
    
    fn run_with[R](self, test_fn: fn(T) -> R) -> R:
        """Run test with fixture.
        
        Args:
            test_fn: Test function using fixture
        
        Returns:
            Test result
        """
        let resource = self.setup()
        let result = test_fn(resource)
        self.teardown(resource)
        return result


fn default_fixture_setup[T]() -> T:
    """Default fixture setup."""
    pass


fn default_fixture_teardown[T](resource: T):
    """Default fixture teardown."""
    pass


# ============================================================================
# Tests
# ============================================================================

test "assert_true passes":
    assert_true(True)

test "assert_false passes":
    assert_false(False)

test "assert_equal integers":
    assert_equal(5, 5)

test "assert_not_equal integers":
    assert_not_equal(5, 10)

test "assert_less integers":
    assert_less(5, 10)

test "assert_greater integers":
    assert_greater(10, 5)

test "assert_in_range integers":
    assert_in_range(5, 0, 10)

test "assert_close floats":
    assert_close(3.14159, 3.14160, 0.001)

test "test suite creation":
    var suite = TestSuite("My Tests")
    assert_equal(len(suite.results), 0)

test "benchmark runs":
    fn dummy_work():
        var sum = 0
        for i in range(100):
            sum += i
    
    let result = benchmark("dummy", dummy_work, 10)
    assert_equal(result.iterations, 10)
