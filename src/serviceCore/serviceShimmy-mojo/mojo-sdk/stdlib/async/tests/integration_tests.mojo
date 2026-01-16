# Async Integration Tests - Day 104
# Comprehensive end-to-end tests for the async system

from async.async_io import AsyncFile, AsyncTcpListener, AsyncTcpStream, read_file, write_file
from async.async_channels import bounded_channel, unbounded_channel, oneshot_channel, broadcast_channel
from async.async_sync import AsyncMutex, AsyncRwLock, AsyncSemaphore, AsyncBarrier
from async.async_utils import spawn, join2, join_all, CancellationToken
from async.async_stream import Stream, stream_from_range, stream_from_list
from collections import List
from time.time import Duration

# ============================================================================
# Test Framework
# ============================================================================

@value
struct TestResult:
    """Result of a test."""
    var name: String
    var passed: Bool
    var message: String
    
    fn __init__(inout self, name: String, passed: Bool, message: String = ""):
        self.name = name
        self.passed = passed
        self.message = message


@value
struct TestSuite:
    """Collection of tests."""
    var name: String
    var results: List[TestResult]
    
    fn __init__(inout self, name: String):
        self.name = name
        self.results = List[TestResult]()
    
    fn add_result(inout self, result: TestResult):
        """Add test result."""
        self.results.append(result)
    
    fn summary(self) -> (Int, Int):
        """Get (passed, total) counts."""
        var passed = 0
        for result in self.results:
            if result.passed:
                passed += 1
        return (passed, len(self.results))
    
    fn print_summary(self):
        """Print test summary."""
        let (passed, total) = self.summary()
        print("\n" + self.name + " Summary:")
        print(str(passed) + "/" + str(total) + " tests passed")
        
        if passed < total:
            print("\nFailed tests:")
            for result in self.results:
                if not result.passed:
                    print("  ❌ " + result.name + ": " + result.message)


# ============================================================================
# Test 1: Basic Async/Await
# ============================================================================

async fn test_basic_async_await() -> TestResult:
    """Test basic async/await syntax."""
    async fn compute() -> Int:
        return 42
    
    let result = await compute()
    let passed = result == 42
    return TestResult("basic_async_await", passed, 
                     "Expected 42, got " + str(result))


async fn test_async_chaining() -> TestResult:
    """Test chaining async calls."""
    async fn step1() -> Int:
        return 10
    
    async fn step2(x: Int) -> Int:
        return x * 2
    
    let result = await step2(await step1())
    let passed = result == 20
    return TestResult("async_chaining", passed)


# ============================================================================
# Test 2: Future Operations
# ============================================================================

async fn test_future_creation() -> TestResult:
    """Test future creation and resolution."""
    let fut = Future[Int].ready(42)
    let result = await fut.poll()
    let passed = result == 42
    return TestResult("future_creation", passed)


async fn test_future_error_handling() -> TestResult:
    """Test future error handling."""
    async fn failing_operation() -> Result[Int, String]:
        return Err("Expected error")
    
    let result = await failing_operation()
    let passed = result.is_err()
    return TestResult("future_error_handling", passed)


# ============================================================================
# Test 3: Task Spawning and Joining
# ============================================================================

async fn test_spawn_and_join() -> TestResult:
    """Test spawning and joining tasks."""
    let handle = spawn(async { return 42 })
    let result = await handle.join()
    let passed = result.is_ok()
    return TestResult("spawn_and_join", passed)


async fn test_multiple_joins() -> TestResult:
    """Test joining multiple tasks."""
    let h1 = spawn(async { return 1 })
    let h2 = spawn(async { return 2 })
    let h3 = spawn(async { return 3 })
    
    let (r1, r2) = await join2(h1, h2)?
    let r3 = await h3.join()?
    
    let sum = r1 + r2 + r3
    let passed = sum == 6
    return TestResult("multiple_joins", passed)


async fn test_join_all() -> TestResult:
    """Test joining all tasks in list."""
    var handles = List[JoinHandle[Int]]()
    for i in range(10):
        handles.append(spawn(async { return i }))
    
    let results = await join_all(handles)?
    let passed = len(results) == 10
    return TestResult("join_all", passed)


# ============================================================================
# Test 4: Channels
# ============================================================================

async fn test_bounded_channel() -> TestResult:
    """Test bounded channel communication."""
    let (tx, rx) = bounded_channel[Int](5)
    
    # Send values
    for i in range(5):
        await tx.send(i)?
    
    # Receive values
    var sum = 0
    for i in range(5):
        sum += await rx.recv()?
    
    let passed = sum == 10  # 0+1+2+3+4
    return TestResult("bounded_channel", passed)


async fn test_unbounded_channel() -> TestResult:
    """Test unbounded channel."""
    let (tx, rx) = unbounded_channel[String]()
    
    await tx.send("hello")?
    let msg = await rx.recv()?
    
    let passed = msg == "hello"
    return TestResult("unbounded_channel", passed)


async fn test_oneshot_channel() -> TestResult:
    """Test oneshot channel."""
    let (tx, rx) = oneshot_channel[Bool]()
    
    tx.send(True)?
    let result = await rx.recv()?
    
    let passed = result == True
    return TestResult("oneshot_channel", passed)


async fn test_broadcast_channel() -> TestResult:
    """Test broadcast channel."""
    let tx = broadcast_channel[Int](5)
    let rx1 = tx.subscribe()
    let rx2 = tx.subscribe()
    
    await tx.send(42)?
    
    let v1 = await rx1.recv()?
    let v2 = await rx2.recv()?
    
    let passed = v1 == 42 and v2 == 42
    return TestResult("broadcast_channel", passed)


# ============================================================================
# Test 5: Synchronization Primitives
# ============================================================================

async fn test_mutex() -> TestResult:
    """Test async mutex."""
    let mutex = AsyncMutex[Int](0)
    
    # Lock and modify
    let guard = await mutex.lock()
    guard.set(42)
    
    # Value should be updated
    let passed = guard.get() == 42
    return TestResult("mutex", passed)


async fn test_rwlock() -> TestResult:
    """Test async RwLock."""
    let rwlock = AsyncRwLock[String]("initial")
    
    # Read lock
    let read_guard = await rwlock.read()
    let value = read_guard.get()
    
    # Write lock
    let write_guard = await rwlock.write()
    write_guard.set("updated")
    
    let passed = write_guard.get() == "updated"
    return TestResult("rwlock", passed)


async fn test_semaphore() -> TestResult:
    """Test async semaphore."""
    let sem = AsyncSemaphore(3)
    
    # Acquire permits
    let p1 = await sem.acquire()
    let p2 = await sem.acquire()
    let p3 = await sem.acquire()
    
    let remaining = sem.available_permits()
    let passed = remaining == 0
    return TestResult("semaphore", passed)


async fn test_barrier() -> TestResult:
    """Test async barrier."""
    let barrier = AsyncBarrier(3)
    
    # Simulate 3 tasks waiting
    var tasks = List[JoinHandle[Bool]]()
    for i in range(3):
        tasks.append(spawn(async {
            let result = await barrier.wait()
            return result.is_leader
        }))
    
    let results = await join_all(tasks)?
    
    # One should be leader
    var leader_count = 0
    for result in results:
        if result:
            leader_count += 1
    
    let passed = leader_count == 1
    return TestResult("barrier", passed)


# ============================================================================
# Test 6: Stream Processing
# ============================================================================

async fn test_stream_map() -> TestResult:
    """Test stream map operation."""
    var stream = stream_from_range(1, 6)
        .map(fn(x: Int) -> Int { return x * 2 })
    
    let results = await stream.collect()
    let passed = len(results) == 5
    return TestResult("stream_map", passed)


async fn test_stream_filter() -> TestResult:
    """Test stream filter operation."""
    var stream = stream_from_range(1, 11)
        .filter(fn(x: Int) -> Bool { return x % 2 == 0 })
    
    let results = await stream.collect()
    let passed = len(results) == 5  # 2,4,6,8,10
    return TestResult("stream_filter", passed)


async fn test_stream_take() -> TestResult:
    """Test stream take operation."""
    var stream = stream_from_range(1, 100)
        .take(5)
    
    let results = await stream.collect()
    let passed = len(results) == 5
    return TestResult("stream_take", passed)


async fn test_stream_fold() -> TestResult:
    """Test stream fold operation."""
    var stream = stream_from_range(1, 6)
    
    let sum = await stream.fold(0, fn(acc: Int, x: Int) -> Int {
        return acc + x
    })
    
    let passed = sum == 15  # 1+2+3+4+5
    return TestResult("stream_fold", passed)


# ============================================================================
# Test 7: Error Handling
# ============================================================================

async fn test_error_propagation() -> TestResult:
    """Test error propagation with ?."""
    async fn may_fail() -> Result[Int, String]:
        return Err("Error")
    
    async fn caller() -> Result[Int, String]:
        let value = await may_fail()?
        return Ok(value)
    
    let result = await caller()
    let passed = result.is_err()
    return TestResult("error_propagation", passed)


async fn test_error_recovery() -> TestResult:
    """Test error recovery with match."""
    async fn may_fail() -> Result[Int, String]:
        return Err("Error")
    
    let result = await may_fail()
    let recovered = match result:
        case Ok(v): v
        case Err(_): 0
    
    let passed = recovered == 0
    return TestResult("error_recovery", passed)


# ============================================================================
# Test 8: Cancellation
# ============================================================================

async fn test_cancellation_token() -> TestResult:
    """Test cancellation token."""
    let token = CancellationToken()
    
    let handle = spawn(async {
        var iterations = 0
        while not token.is_cancelled():
            iterations += 1
            if iterations > 100:
                break
        return iterations
    })
    
    # Cancel immediately
    token.cancel()
    
    let result = await handle.join()?
    let passed = result <= 100
    return TestResult("cancellation_token", passed)


# ============================================================================
# Test Runner
# ============================================================================

async fn run_integration_tests() -> TestSuite:
    """Run all integration tests."""
    var suite = TestSuite("Async Integration Tests")
    
    # Test 1: Basic
    suite.add_result(await test_basic_async_await())
    suite.add_result(await test_async_chaining())
    
    # Test 2: Futures
    suite.add_result(await test_future_creation())
    suite.add_result(await test_future_error_handling())
    
    # Test 3: Tasks
    suite.add_result(await test_spawn_and_join())
    suite.add_result(await test_multiple_joins())
    suite.add_result(await test_join_all())
    
    # Test 4: Channels
    suite.add_result(await test_bounded_channel())
    suite.add_result(await test_unbounded_channel())
    suite.add_result(await test_oneshot_channel())
    suite.add_result(await test_broadcast_channel())
    
    # Test 5: Sync primitives
    suite.add_result(await test_mutex())
    suite.add_result(await test_rwlock())
    suite.add_result(await test_semaphore())
    suite.add_result(await test_barrier())
    
    # Test 6: Streams
    suite.add_result(await test_stream_map())
    suite.add_result(await test_stream_filter())
    suite.add_result(await test_stream_take())
    suite.add_result(await test_stream_fold())
    
    # Test 7: Errors
    suite.add_result(await test_error_propagation())
    suite.add_result(await test_error_recovery())
    
    # Test 8: Cancellation
    suite.add_result(await test_cancellation_token())
    
    return suite


fn main():
    """Main test runner."""
    print("Running Async Integration Tests...")
    let suite = await run_integration_tests()
    suite.print_summary()
    
    let (passed, total) = suite.summary()
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
