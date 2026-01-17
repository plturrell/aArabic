# Async Utilities Module - Day 103
# Provides join!, select!, spawn_local, and task-local storage utilities

from collections import List, Optional
from time.time import Duration

# ============================================================================
# Join Utilities - Concurrent Execution
# ============================================================================

@value
struct JoinHandle[T]:
    """Handle to a spawned task."""
    
    var _task_id: Int
    var _completed: Bool
    
    fn __init__(inout self, task_id: Int):
        """Initialize join handle."""
        self._task_id = task_id
        self._completed = False
    
    async fn join(inout self) -> Result[T, TaskError]:
        """Wait for task to complete and get result.
        
        Returns:
            Task result or error if task panicked.
        """
        # TODO: Implement join using runtime
        self._completed = True
        return Err(TaskError("Not implemented"))
    
    fn is_finished(self) -> Bool:
        """Check if task has completed."""
        return self._completed
    
    fn abort(inout self):
        """Cancel the task."""
        # TODO: Implement task cancellation
        self._completed = True


async fn spawn[T, F: AsyncFn[T]](func: F) -> JoinHandle[T]:
    """Spawn an async task.
    
    Args:
        func: Async function to execute.
    
    Returns:
        Join handle for the spawned task.
    
    Example:
        let handle = spawn(async { await compute() })
        let result = await handle.join()
    """
    # TODO: Integrate with executor
    return JoinHandle[T](0)


async fn spawn_local[T, F: AsyncFn[T]](func: F) -> JoinHandle[T]:
    """Spawn a local async task (non-Send).
    
    Tasks spawned with spawn_local can capture non-Send types
    but must complete on the same thread.
    
    Args:
        func: Async function to execute.
    
    Returns:
        Join handle for the spawned task.
    """
    # TODO: Implement thread-local task spawning
    return JoinHandle[T](0)


async fn join[T](handle: JoinHandle[T]) -> Result[T, TaskError]:
    """Wait for a task to complete.
    
    Args:
        handle: Join handle from spawn.
    
    Returns:
        Task result.
    """
    return await handle.join()


async fn join2[T1, T2](
    handle1: JoinHandle[T1],
    handle2: JoinHandle[T2]
) -> Result[(T1, T2), TaskError]:
    """Join two tasks concurrently.
    
    Args:
        handle1: First task handle.
        handle2: Second task handle.
    
    Returns:
        Tuple of both results.
    
    Example:
        let h1 = spawn(async { await task1() })
        let h2 = spawn(async { await task2() })
        let (r1, r2) = await join2(h1, h2)?
    """
    let r1 = await handle1.join()?
    let r2 = await handle2.join()?
    return Ok((r1, r2))


async fn join3[T1, T2, T3](
    handle1: JoinHandle[T1],
    handle2: JoinHandle[T2],
    handle3: JoinHandle[T3]
) -> Result[(T1, T2, T3), TaskError]:
    """Join three tasks concurrently."""
    let r1 = await handle1.join()?
    let r2 = await handle2.join()?
    let r3 = await handle3.join()?
    return Ok((r1, r2, r3))


async fn join_all[T](handles: List[JoinHandle[T]]) -> Result[List[T], TaskError]:
    """Join all tasks in a list.
    
    Args:
        handles: List of join handles.
    
    Returns:
        List of results in same order.
    """
    var results = List[T]()
    for handle in handles:
        let result = await handle.join()?
        results.append(result)
    return Ok(results)


# ============================================================================
# Select Utilities - Racing Futures
# ============================================================================

@value
struct SelectArm[T]:
    """Arm of a select expression."""
    
    var index: Int
    var ready: Bool
    
    fn __init__(inout self, index: Int):
        self.index = index
        self.ready = False


async fn select2[T1, T2](
    fut1: Future[T1],
    fut2: Future[T2]
) -> Result[(Int, Either[T1, T2]), SelectError]:
    """Select first completed future from two.
    
    Args:
        fut1: First future.
        fut2: Second future.
    
    Returns:
        Tuple of (index, value) where index is 0 or 1.
    
    Example:
        let result = await select2(
            async { await fetch_api1() },
            async { await fetch_api2() }
        )
        match result:
            case (0, Left(val)): print("API 1 won: " + val)
            case (1, Right(val)): print("API 2 won: " + val)
    """
    # TODO: Implement select using runtime
    return Err(SelectError("Not implemented"))


async fn select3[T1, T2, T3](
    fut1: Future[T1],
    fut2: Future[T2],
    fut3: Future[T3]
) -> Result[(Int, Either3[T1, T2, T3]), SelectError]:
    """Select first completed future from three."""
    # TODO: Implement 3-way select
    return Err(SelectError("Not implemented"))


async fn race[T](futures: List[Future[T]]) -> Result[(Int, T), SelectError]:
    """Race multiple futures, return first to complete.
    
    Args:
        futures: List of futures to race.
    
    Returns:
        Tuple of (index, value) of winning future.
    """
    # TODO: Implement dynamic race
    return Err(SelectError("Not implemented"))


# ============================================================================
# Either Types for Select Results
# ============================================================================

@value
struct Either[L, R]:
    """Either Left or Right value."""
    
    var is_left: Bool
    var _left: Optional[L]
    var _right: Optional[R]
    
    fn __init__(inout self, left: L):
        """Create Left value."""
        self.is_left = True
        self._left = Some(left)
        self._right = None
    
    fn __init__(inout self, right: R):
        """Create Right value."""
        self.is_left = False
        self._left = None
        self._right = Some(right)
    
    fn left(self) -> Optional[L]:
        """Get left value if present."""
        return self._left
    
    fn right(self) -> Optional[R]:
        """Get right value if present."""
        return self._right
    
    fn unwrap_left(self) -> L:
        """Unwrap left value (panics if Right)."""
        return self._left.unwrap()
    
    fn unwrap_right(self) -> R:
        """Unwrap right value (panics if Left)."""
        return self._right.unwrap()


@value
struct Either3[T1, T2, T3]:
    """Either of three values."""
    
    var index: Int
    var _val1: Optional[T1]
    var _val2: Optional[T2]
    var _val3: Optional[T3]
    
    fn __init__(inout self, val1: T1):
        self.index = 0
        self._val1 = Some(val1)
        self._val2 = None
        self._val3 = None
    
    fn __init__(inout self, val2: T2):
        self.index = 1
        self._val1 = None
        self._val2 = Some(val2)
        self._val3 = None
    
    fn __init__(inout self, val3: T3):
        self.index = 2
        self._val1 = None
        self._val2 = None
        self._val3 = Some(val3)


# ============================================================================
# Task-Local Storage
# ============================================================================

@value
struct TaskLocal[T]:
    """Task-local storage (per-task state).
    
    Similar to thread-local storage, but per async task.
    """
    
    var _key: Int
    var _default: Optional[T]
    
    fn __init__(inout self):
        """Initialize task-local without default."""
        self._key = 0  # TODO: Generate unique key
        self._default = None
    
    fn __init__(inout self, default: T):
        """Initialize task-local with default value."""
        self._key = 0
        self._default = Some(default)
    
    fn get(self) -> Optional[T]:
        """Get value for current task.
        
        Returns:
            Value if set, or default if provided, else None.
        """
        # TODO: Implement task-local lookup
        return self._default
    
    fn set(self, value: T):
        """Set value for current task."""
        # TODO: Implement task-local store
        pass
    
    fn remove(self):
        """Remove value for current task."""
        # TODO: Implement task-local clear
        pass
    
    fn with[R, F: Fn[R]](self, value: T, func: F) -> R:
        """Run function with temporary task-local value.
        
        Args:
            value: Temporary value.
            func: Function to run.
        
        Returns:
            Function result.
        """
        # TODO: Implement scoped task-local
        return func()


# ============================================================================
# Cancellation Support
# ============================================================================

@value
struct CancellationToken:
    """Token for cancelling async operations."""
    
    var _cancelled: Bool
    
    fn __init__(inout self):
        """Create cancellation token."""
        self._cancelled = False
    
    fn cancel(inout self):
        """Signal cancellation."""
        self._cancelled = True
    
    fn is_cancelled(self) -> Bool:
        """Check if cancelled."""
        return self._cancelled
    
    async fn cancelled(self):
        """Wait until cancelled."""
        # TODO: Implement async wait for cancellation
        while not self._cancelled:
            # Yield
            pass


@value
struct CancellableTask[T]:
    """Task with cancellation support."""
    
    var _handle: JoinHandle[T]
    var _token: CancellationToken
    
    fn __init__(inout self, handle: JoinHandle[T], token: CancellationToken):
        self._handle = handle
        self._token = token
    
    async fn join(inout self) -> Result[T, TaskError]:
        """Wait for task completion."""
        return await self._handle.join()
    
    fn cancel(inout self):
        """Cancel the task."""
        self._token.cancel()
        self._handle.abort()


async fn with_cancellation[T, F: AsyncFn[T]](
    token: CancellationToken,
    func: F
) -> Result[T, TaskError]:
    """Run async function with cancellation support.
    
    Args:
        token: Cancellation token.
        func: Async function to run.
    
    Returns:
        Function result or cancellation error.
    """
    # TODO: Implement cancellation-aware execution
    if token.is_cancelled():
        return Err(TaskError("Cancelled"))
    
    return await func()


# ============================================================================
# Async Combinators
# ============================================================================

async fn try_join[T](handle: JoinHandle[T]) -> Optional[T]:
    """Try to join task without propagating errors.
    
    Args:
        handle: Task handle.
    
    Returns:
        Some(result) if successful, None if error.
    """
    let result = await handle.join()
    if result.is_ok():
        return Some(result.unwrap())
    return None


async fn timeout_join[T](
    handle: JoinHandle[T],
    duration: Duration
) -> Result[T, TimeoutError]:
    """Join with timeout.
    
    Args:
        handle: Task handle.
        duration: Maximum wait time.
    
    Returns:
        Task result or timeout error.
    """
    # TODO: Implement timeout join using select
    return await handle.join().map_err(|e| TimeoutError(duration))


# ============================================================================
# Error Types
# ============================================================================

@value
struct TaskError:
    """Task execution error."""
    var message: String
    
    fn __init__(inout self, message: String):
        self.message = message


@value
struct SelectError:
    """Select operation error."""
    var message: String
    
    fn __init__(inout self, message: String):
        self.message = message


@value
struct TimeoutError:
    """Timeout error."""
    var duration: Duration
    
    fn __init__(inout self, duration: Duration):
        self.duration = duration


# ============================================================================
# Tests
# ============================================================================

fn test_join_handle_creation():
    """Test join handle creation."""
    let handle = JoinHandle[Int](42)
    assert_equal(handle._task_id, 42)
    assert_false(handle.is_finished())


fn test_join_handle_abort():
    """Test join handle abort."""
    var handle = JoinHandle[Int](1)
    handle.abort()
    assert_true(handle.is_finished())


fn test_either_left():
    """Test Either Left construction."""
    let either = Either[Int, String](42)
    assert_true(either.is_left)
    assert_true(either.left() is Some)
    assert_true(either.right() is None)


fn test_either_right():
    """Test Either Right construction."""
    let either = Either[Int, String]("hello")
    assert_false(either.is_left)
    assert_true(either.left() is None)
    assert_true(either.right() is Some)


fn test_either3_creation():
    """Test Either3 construction."""
    let e1 = Either3[Int, String, Bool](42)
    assert_equal(e1.index, 0)
    
    let e2 = Either3[Int, String, Bool]("test")
    assert_equal(e2.index, 1)
    
    let e3 = Either3[Int, String, Bool](True)
    assert_equal(e3.index, 2)


fn test_task_local_creation():
    """Test task-local creation."""
    let local = TaskLocal[Int]()
    assert_true(local.get() is None)


fn test_task_local_with_default():
    """Test task-local with default."""
    let local = TaskLocal[Int](42)
    let value = local.get()
    assert_true(value is Some)


fn test_cancellation_token():
    """Test cancellation token."""
    var token = CancellationToken()
    assert_false(token.is_cancelled())
    
    token.cancel()
    assert_true(token.is_cancelled())


fn test_cancellable_task():
    """Test cancellable task."""
    let handle = JoinHandle[Int](1)
    let token = CancellationToken()
    var task = CancellableTask[Int](handle, token)
    
    task.cancel()
    assert_true(token.is_cancelled())


fn test_task_error():
    """Test task error creation."""
    let error = TaskError("test error")
    assert_equal(error.message, "test error")


fn test_select_error():
    """Test select error creation."""
    let error = SelectError("select failed")
    assert_equal(error.message, "select failed")


fn test_timeout_error():
    """Test timeout error creation."""
    let duration = Duration.from_secs(5)
    let error = TimeoutError(duration)
    assert_equal(error.duration.seconds(), 5)


fn run_all_tests():
    """Run all async utility tests."""
    test_join_handle_creation()
    test_join_handle_abort()
    test_either_left()
    test_either_right()
    test_either3_creation()
    test_task_local_creation()
    test_task_local_with_default()
    test_cancellation_token()
    test_cancellable_task()
    test_task_error()
    test_select_error()
    test_timeout_error()
    print("All async utility tests passed! ✅")
