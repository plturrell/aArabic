# Async Synchronization Module - Day 102
# Provides async synchronization primitives (Mutex, RwLock, Semaphore, Barrier)

from collections import List, Optional
from time.time import Duration

# ============================================================================
# Async Mutex
# ============================================================================

@value
struct AsyncMutex[T]:
    """Async mutual exclusion lock."""
    
    var _locked: Bool
    var _data: T
    
    fn __init__(inout self, data: T):
        """Initialize mutex with protected data."""
        self._locked = False
        self._data = data
    
    async fn lock(inout self) -> MutexGuard[T]:
        """Acquire lock asynchronously.
        
        Returns:
            Guard providing exclusive access to data.
        """
        # TODO: Implement async lock acquisition
        # Should wait if already locked
        self._locked = True
        return MutexGuard[T](self)
    
    async fn try_lock(inout self) -> Optional[MutexGuard[T]]:
        """Try to acquire lock without waiting.
        
        Returns:
            Some(guard) if acquired, None if already locked.
        """
        if self._locked:
            return None
        
        self._locked = True
        return Some(MutexGuard[T](self))
    
    fn unlock(inout self):
        """Release the lock."""
        self._locked = False
    
    fn is_locked(self) -> Bool:
        """Check if mutex is currently locked."""
        return self._locked


@value
struct MutexGuard[T]:
    """RAII guard for mutex lock."""
    
    var _mutex: AsyncMutex[T]
    
    fn __init__(inout self, mutex: AsyncMutex[T]):
        self._mutex = mutex
    
    fn __del__(owned self):
        """Automatically unlock when guard is dropped."""
        self._mutex.unlock()
    
    fn get(inout self) -> T:
        """Get reference to protected data."""
        return self._mutex._data
    
    fn set(inout self, value: T):
        """Set protected data."""
        self._mutex._data = value


# ============================================================================
# Async RwLock
# ============================================================================

@value
struct AsyncRwLock[T]:
    """Async reader-writer lock."""
    
    var _readers: Int
    var _writer: Bool
    var _data: T
    
    fn __init__(inout self, data: T):
        """Initialize RwLock with protected data."""
        self._readers = 0
        self._writer = False
        self._data = data
    
    async fn read(inout self) -> ReadGuard[T]:
        """Acquire read lock (shared).
        
        Returns:
            Read guard providing shared access.
        """
        # TODO: Implement async read lock
        # Multiple readers allowed, but not with writer
        self._readers += 1
        return ReadGuard[T](self)
    
    async fn write(inout self) -> WriteGuard[T]:
        """Acquire write lock (exclusive).
        
        Returns:
            Write guard providing exclusive access.
        """
        # TODO: Implement async write lock
        # Only one writer, no readers
        self._writer = True
        return WriteGuard[T](self)
    
    async fn try_read(inout self) -> Optional[ReadGuard[T]]:
        """Try to acquire read lock without waiting."""
        if self._writer:
            return None
        
        self._readers += 1
        return Some(ReadGuard[T](self))
    
    async fn try_write(inout self) -> Optional[WriteGuard[T]]:
        """Try to acquire write lock without waiting."""
        if self._writer or self._readers > 0:
            return None
        
        self._writer = True
        return Some(WriteGuard[T](self))
    
    fn unlock_read(inout self):
        """Release a read lock."""
        if self._readers > 0:
            self._readers -= 1
    
    fn unlock_write(inout self):
        """Release the write lock."""
        self._writer = False


@value
struct ReadGuard[T]:
    """RAII guard for read lock."""
    
    var _lock: AsyncRwLock[T]
    
    fn __init__(inout self, lock: AsyncRwLock[T]):
        self._lock = lock
    
    fn __del__(owned self):
        """Automatically release read lock."""
        self._lock.unlock_read()
    
    fn get(self) -> T:
        """Get shared reference to data."""
        return self._lock._data


@value
struct WriteGuard[T]:
    """RAII guard for write lock."""
    
    var _lock: AsyncRwLock[T]
    
    fn __init__(inout self, lock: AsyncRwLock[T]):
        self._lock = lock
    
    fn __del__(owned self):
        """Automatically release write lock."""
        self._lock.unlock_write()
    
    fn get(inout self) -> T:
        """Get exclusive reference to data."""
        return self._lock._data
    
    fn set(inout self, value: T):
        """Modify protected data."""
        self._lock._data = value


# ============================================================================
# Async Semaphore
# ============================================================================

@value
struct AsyncSemaphore:
    """Async counting semaphore."""
    
    var _permits: Int
    var _max_permits: Int
    
    fn __init__(inout self, permits: Int):
        """Initialize semaphore with permit count."""
        self._permits = permits
        self._max_permits = permits
    
    async fn acquire(inout self) -> SemaphorePermit:
        """Acquire a permit.
        
        Returns:
            Permit that releases on drop.
        """
        # TODO: Implement async acquire (wait if no permits)
        if self._permits > 0:
            self._permits -= 1
        return SemaphorePermit(self)
    
    async fn acquire_many(inout self, n: Int) -> SemaphorePermit:
        """Acquire multiple permits.
        
        Args:
            n: Number of permits to acquire.
        
        Returns:
            Permit representing n permits.
        """
        # TODO: Implement async acquire_many
        self._permits -= n
        return SemaphorePermit(self, n)
    
    fn try_acquire(inout self) -> Optional[SemaphorePermit]:
        """Try to acquire permit without waiting."""
        if self._permits <= 0:
            return None
        
        self._permits -= 1
        return Some(SemaphorePermit(self))
    
    fn release(inout self, n: Int = 1):
        """Release permits."""
        self._permits = min(self._permits + n, self._max_permits)
    
    fn available_permits(self) -> Int:
        """Get number of available permits."""
        return self._permits


@value
struct SemaphorePermit:
    """RAII permit for semaphore."""
    
    var _semaphore: AsyncSemaphore
    var _count: Int
    
    fn __init__(inout self, semaphore: AsyncSemaphore, count: Int = 1):
        self._semaphore = semaphore
        self._count = count
    
    fn __del__(owned self):
        """Automatically release permit(s)."""
        self._semaphore.release(self._count)
    
    fn forget(owned self):
        """Consume permit without releasing."""
        # Permit is dropped without release
        pass


# ============================================================================
# Async Barrier
# ============================================================================

@value
struct AsyncBarrier:
    """Async synchronization barrier."""
    
    var _count: Int
    var _waiting: Int
    var _generation: Int
    
    fn __init__(inout self, count: Int):
        """Initialize barrier for count tasks.
        
        Args:
            count: Number of tasks that must wait before proceeding.
        """
        self._count = count
        self._waiting = 0
        self._generation = 0
    
    async fn wait(inout self) -> BarrierWaitResult:
        """Wait at the barrier.
        
        Returns:
            Result indicating if this is the leader.
        """
        # TODO: Implement async barrier wait
        self._waiting += 1
        let is_leader = self._waiting >= self._count
        
        if is_leader:
            # Reset for next use
            self._waiting = 0
            self._generation += 1
        
        return BarrierWaitResult(is_leader, self._generation)


@value
struct BarrierWaitResult:
    """Result from barrier wait."""
    
    var is_leader: Bool
    var generation: Int
    
    fn __init__(inout self, is_leader: Bool, generation: Int):
        self.is_leader = is_leader
        self.generation = generation


# ============================================================================
# Async Notify (Condition Variable)
# ============================================================================

@value
struct AsyncNotify:
    """Async condition variable / notification."""
    
    var _notified: Bool
    
    fn __init__(inout self):
        """Initialize notification."""
        self._notified = False
    
    async fn notified(inout self):
        """Wait for notification."""
        # TODO: Implement async wait for notification
        while not self._notified:
            # Yield and wait
            pass
        self._notified = False
    
    fn notify_one(inout self):
        """Notify one waiting task."""
        self._notified = True
    
    fn notify_all(inout self):
        """Notify all waiting tasks."""
        self._notified = True


# ============================================================================
# Async Once
# ============================================================================

@value
struct AsyncOnce[T]:
    """Ensures async initialization happens only once."""
    
    var _initialized: Bool
    var _value: Optional[T]
    
    fn __init__(inout self):
        """Initialize Once."""
        self._initialized = False
        self._value = None
    
    async fn call_once[F: AsyncFn[T]](inout self, init: F) -> T:
        """Call initialization function once.
        
        Args:
            init: Async function to initialize value.
        
        Returns:
            Initialized value.
        """
        if not self._initialized:
            # TODO: Implement proper once semantics with locking
            let value = await init()
            self._value = Some(value)
            self._initialized = True
        
        return self._value.unwrap()
    
    fn is_initialized(self) -> Bool:
        """Check if already initialized."""
        return self._initialized


# ============================================================================
# Tests
# ============================================================================

fn test_async_mutex_creation():
    """Test async mutex creation."""
    let mutex = AsyncMutex[Int](42)
    assert_false(mutex.is_locked())
    assert_equal(mutex._data, 42)


fn test_async_rwlock_creation():
    """Test async RwLock creation."""
    let rwlock = AsyncRwLock[String]("test")
    assert_equal(rwlock._readers, 0)
    assert_false(rwlock._writer)
    assert_equal(rwlock._data, "test")


fn test_semaphore_creation():
    """Test semaphore creation."""
    let sem = AsyncSemaphore(5)
    assert_equal(sem.available_permits(), 5)
    assert_equal(sem._max_permits, 5)


fn test_semaphore_try_acquire():
    """Test semaphore try_acquire."""
    var sem = AsyncSemaphore(2)
    let permit1 = sem.try_acquire()
    assert_true(permit1 is Some)
    assert_equal(sem.available_permits(), 1)
    
    let permit2 = sem.try_acquire()
    assert_true(permit2 is Some)
    assert_equal(sem.available_permits(), 0)
    
    let permit3 = sem.try_acquire()
    assert_true(permit3 is None)


fn test_semaphore_release():
    """Test semaphore release."""
    var sem = AsyncSemaphore(5)
    sem._permits = 2
    sem.release(3)
    assert_equal(sem.available_permits(), 5)
    
    # Should not exceed max
    sem.release(10)
    assert_equal(sem.available_permits(), 5)


fn test_barrier_creation():
    """Test barrier creation."""
    let barrier = AsyncBarrier(3)
    assert_equal(barrier._count, 3)
    assert_equal(barrier._waiting, 0)
    assert_equal(barrier._generation, 0)


fn test_notify_creation():
    """Test notify creation."""
    let notify = AsyncNotify()
    assert_false(notify._notified)


fn test_once_creation():
    """Test once creation."""
    let once = AsyncOnce[Int]()
    assert_false(once.is_initialized())
    assert_true(once._value is None)


fn test_mutex_lock_state():
    """Test mutex lock state changes."""
    var mutex = AsyncMutex[Int](0)
    assert_false(mutex.is_locked())
    
    mutex._locked = True
    assert_true(mutex.is_locked())
    
    mutex.unlock()
    assert_false(mutex.is_locked())


fn test_rwlock_reader_count():
    """Test RwLock reader counting."""
    var rwlock = AsyncRwLock[Int](0)
    assert_equal(rwlock._readers, 0)
    
    rwlock._readers = 3
    rwlock.unlock_read()
    assert_equal(rwlock._readers, 2)
    
    rwlock.unlock_read()
    rwlock.unlock_read()
    assert_equal(rwlock._readers, 0)


fn test_barrier_wait_result():
    """Test barrier wait result."""
    let result = BarrierWaitResult(True, 5)
    assert_true(result.is_leader)
    assert_equal(result.generation, 5)


fn run_all_tests():
    """Run all async sync tests."""
    test_async_mutex_creation()
    test_async_rwlock_creation()
    test_semaphore_creation()
    test_semaphore_try_acquire()
    test_semaphore_release()
    test_barrier_creation()
    test_notify_creation()
    test_once_creation()
    test_mutex_lock_state()
    test_rwlock_reader_count()
    test_barrier_wait_result()
    print("All async sync tests passed! ✅")
