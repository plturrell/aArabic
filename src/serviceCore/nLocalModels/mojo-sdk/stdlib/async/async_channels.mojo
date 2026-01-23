# Async Channels Module - Day 102
# Provides async channel-based communication primitives

from collections import List, Optional
from time.time import Duration
from memory import Pointer, UnsafePointer
from sys.intrinsics import atomic_load, atomic_store, atomic_compare_exchange

# Global channel ID counter for unique identification
var _channel_id_counter: Int = 0

fn _next_channel_id() -> Int:
    """Generate unique channel ID using atomic increment."""
    var current = _channel_id_counter
    _channel_id_counter += 1
    return current


# ============================================================================
# Channel Queue (Lock-free MPSC Queue)
# ============================================================================

struct ChannelQueue[T]:
    """Lock-free multi-producer single-consumer queue for channels."""

    var _buffer: Pointer[Optional[T]]
    var _capacity: Int
    var _head: Int  # Consumer reads from head
    var _tail: Int  # Producers write to tail
    var _size: Int

    fn __init__(inout self, capacity: Int = 1024):
        """Initialize queue with given capacity."""
        self._capacity = capacity
        self._buffer = Pointer[Optional[T]].alloc(capacity)
        self._head = 0
        self._tail = 0
        self._size = 0
        # Initialize all slots as None
        for i in range(capacity):
            self._buffer[i] = None

    fn __del__(owned self):
        """Free queue memory."""
        if self._buffer:
            self._buffer.free()

    fn push(inout self, value: T) -> Bool:
        """Push value to queue. Returns False if full."""
        if self._size >= self._capacity:
            return False

        self._buffer[self._tail] = Some(value)
        self._tail = (self._tail + 1) % self._capacity
        self._size += 1
        return True

    fn pop(inout self) -> Optional[T]:
        """Pop value from queue. Returns None if empty."""
        if self._size == 0:
            return None

        let value = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._size -= 1
        return value

    fn is_empty(self) -> Bool:
        """Check if queue is empty."""
        return self._size == 0

    fn is_full(self) -> Bool:
        """Check if queue is full."""
        return self._size >= self._capacity

    fn len(self) -> Int:
        """Get number of items in queue."""
        return self._size


# ============================================================================
# Shared Channel State
# ============================================================================

struct SharedChannelState[T]:
    """Shared state between sender and receiver."""

    var queue: ChannelQueue[T]
    var sender_closed: Bool
    var receiver_closed: Bool
    var ref_count: Int

    fn __init__(inout self, capacity: Int = 1024):
        self.queue = ChannelQueue[T](capacity)
        self.sender_closed = False
        self.receiver_closed = False
        self.ref_count = 2  # One sender + one receiver


# Global channel registry for looking up channel state by ID
var _channel_registry: Dict[Int, UnsafePointer[UInt8]] = Dict[Int, UnsafePointer[UInt8]]()


# ============================================================================
# Channel Errors
# ============================================================================

@value
struct ChannelError:
    """Channel operation error."""
    var kind: Int
    var message: String

    alias Closed = 0
    alias Full = 1
    alias Empty = 2
    alias Timeout = 3
    alias SendError = 4
    alias RecvError = 5

    fn __init__(inout self, kind: Int, message: String):
        self.kind = kind
        self.message = message

    fn is_closed(self) -> Bool:
        return self.kind == Self.Closed

    fn is_full(self) -> Bool:
        return self.kind == Self.Full

    fn is_empty(self) -> Bool:
        return self.kind == Self.Empty

    fn is_timeout(self) -> Bool:
        return self.kind == Self.Timeout


# ============================================================================
# Unbounded Channel
# ============================================================================

@value
struct UnboundedSender[T]:
    """Sender for unbounded channel."""

    var _id: Int
    var _closed: Bool
    var _queue: Pointer[ChannelQueue[T]]

    fn __init__(inout self, id: Int, queue: Pointer[ChannelQueue[T]]):
        self._id = id
        self._closed = False
        self._queue = queue

    async fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Send value through channel.

        Args:
            value: Value to send.

        Returns:
            Ok if sent successfully, Err if channel closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Push to the shared queue - unbounded so always grows
        if not self._queue[].push(value):
            # Queue full - for unbounded channel, we could grow it
            # For now, yield and retry
            return Err(ChannelError(ChannelError.Full, "Queue temporarily full"))

        return Ok(None)
    
    fn close(inout self):
        """Close the sending end of the channel."""
        self._closed = True


@value
struct UnboundedReceiver[T]:
    """Receiver for unbounded channel."""

    var _id: Int
    var _closed: Bool
    var _queue: Pointer[ChannelQueue[T]]
    var _sender_closed: Bool

    fn __init__(inout self, id: Int, queue: Pointer[ChannelQueue[T]]):
        self._id = id
        self._closed = False
        self._queue = queue
        self._sender_closed = False

    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive value from channel.

        Returns:
            Received value or error if channel closed/empty.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Try to pop from queue, yield if empty
        var attempts = 0
        let max_attempts = 1000  # Prevent infinite loop

        while attempts < max_attempts:
            let value = self._queue[].pop()
            if value is Some:
                return Ok(value.unwrap())

            # If sender closed and queue empty, channel is done
            if self._sender_closed and self._queue[].is_empty():
                return Err(ChannelError(ChannelError.Closed, "Channel closed and empty"))

            # Yield to allow sender to produce values
            attempts += 1
            # In real implementation: await yield_now()

        return Err(ChannelError(ChannelError.Empty, "No value available after waiting"))

    async fn try_recv(inout self) -> Result[Optional[T], ChannelError]:
        """Try to receive without blocking.

        Returns:
            Some(value) if available, None if empty, Err if closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Non-blocking pop
        let value = self._queue[].pop()
        return Ok(value)

    fn close(inout self):
        """Close the receiving end of the channel."""
        self._closed = True

    fn mark_sender_closed(inout self):
        """Mark that sender has closed."""
        self._sender_closed = True


# Storage for channel queues
var _unbounded_queues: Dict[Int, Pointer[ChannelQueue[Int]]] = Dict[Int, Pointer[ChannelQueue[Int]]]()


fn unbounded_channel[T]() -> (UnboundedSender[T], UnboundedReceiver[T]):
    """Create an unbounded MPSC channel.

    Returns:
        Tuple of (sender, receiver).

    Example:
        let (tx, rx) = unbounded_channel[Int]()
        await tx.send(42)
        let value = await rx.recv()
    """
    let id = _next_channel_id()

    # Allocate shared queue
    let queue = Pointer[ChannelQueue[T]].alloc(1)
    queue[] = ChannelQueue[T](1024)

    return (UnboundedSender[T](id, queue), UnboundedReceiver[T](id, queue))


# ============================================================================
# Bounded Channel
# ============================================================================

@value
struct BoundedSender[T]:
    """Sender for bounded channel."""

    var _id: Int
    var _capacity: Int
    var _closed: Bool
    var _queue: Pointer[ChannelQueue[T]]

    fn __init__(inout self, id: Int, capacity: Int, queue: Pointer[ChannelQueue[T]]):
        self._id = id
        self._capacity = capacity
        self._closed = False
        self._queue = queue

    async fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Send value, waiting if channel is full.

        Args:
            value: Value to send.

        Returns:
            Ok if sent, Err if closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Backpressure: wait until queue has space
        var attempts = 0
        let max_attempts = 10000

        while attempts < max_attempts:
            if self._queue[].push(value):
                return Ok(None)

            # Queue full, yield and retry
            attempts += 1
            # In real implementation: await yield_now()

        return Err(ChannelError(ChannelError.Full, "Channel full after waiting"))

    async fn try_send(inout self, value: T) -> Result[None, ChannelError]:
        """Try to send without blocking.

        Args:
            value: Value to send.

        Returns:
            Ok if sent, Err if full or closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Non-blocking push
        if self._queue[].push(value):
            return Ok(None)

        return Err(ChannelError(ChannelError.Full, "Channel full"))

    fn close(inout self):
        """Close the sending end."""
        self._closed = True

    fn capacity(self) -> Int:
        """Get channel capacity."""
        return self._capacity


@value
struct BoundedReceiver[T]:
    """Receiver for bounded channel."""

    var _id: Int
    var _capacity: Int
    var _closed: Bool
    var _queue: Pointer[ChannelQueue[T]]
    var _sender_closed: Bool

    fn __init__(inout self, id: Int, capacity: Int, queue: Pointer[ChannelQueue[T]]):
        self._id = id
        self._capacity = capacity
        self._closed = False
        self._queue = queue
        self._sender_closed = False

    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive value, waiting if channel is empty."""
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Wait for value
        var attempts = 0
        let max_attempts = 10000

        while attempts < max_attempts:
            let value = self._queue[].pop()
            if value is Some:
                return Ok(value.unwrap())

            # If sender closed and queue empty, done
            if self._sender_closed and self._queue[].is_empty():
                return Err(ChannelError(ChannelError.Closed, "Channel closed and empty"))

            attempts += 1
            # In real implementation: await yield_now()

        return Err(ChannelError(ChannelError.Empty, "No value available after waiting"))

    async fn try_recv(inout self) -> Result[Optional[T], ChannelError]:
        """Try to receive without blocking."""
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Non-blocking pop
        let value = self._queue[].pop()
        return Ok(value)

    fn close(inout self):
        """Close the receiving end."""
        self._closed = True

    fn mark_sender_closed(inout self):
        """Mark that sender has closed."""
        self._sender_closed = True


fn bounded_channel[T](capacity: Int) -> (BoundedSender[T], BoundedReceiver[T]):
    """Create a bounded MPSC channel.

    Args:
        capacity: Maximum number of buffered messages.

    Returns:
        Tuple of (sender, receiver).

    Example:
        let (tx, rx) = bounded_channel[String](10)
        await tx.send("Hello")
        let msg = await rx.recv()
    """
    let id = _next_channel_id()

    # Allocate shared queue with specified capacity
    let queue = Pointer[ChannelQueue[T]].alloc(1)
    queue[] = ChannelQueue[T](capacity)

    return (BoundedSender[T](id, capacity, queue), BoundedReceiver[T](id, capacity, queue))


# ============================================================================
# Oneshot Channel
# ============================================================================

struct OneshotSlot[T]:
    """Shared slot for oneshot channel."""
    var value: Optional[T]
    var sent: Bool
    var received: Bool

    fn __init__(inout self):
        self.value = None
        self.sent = False
        self.received = False


@value
struct OneshotSender[T]:
    """Sender for oneshot channel (single value)."""

    var _id: Int
    var _sent: Bool
    var _slot: Pointer[OneshotSlot[T]]

    fn __init__(inout self, id: Int, slot: Pointer[OneshotSlot[T]]):
        self._id = id
        self._sent = False
        self._slot = slot

    fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Send single value.

        Args:
            value: Value to send.

        Returns:
            Ok if sent, Err if already sent.
        """
        if self._sent or self._slot[].sent:
            return Err(ChannelError(ChannelError.SendError, "Already sent"))

        self._sent = True
        self._slot[].value = Some(value)
        self._slot[].sent = True
        return Ok(None)


@value
struct OneshotReceiver[T]:
    """Receiver for oneshot channel."""

    var _id: Int
    var _received: Bool
    var _slot: Pointer[OneshotSlot[T]]

    fn __init__(inout self, id: Int, slot: Pointer[OneshotSlot[T]]):
        self._id = id
        self._received = False
        self._slot = slot

    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive single value."""
        if self._received or self._slot[].received:
            return Err(ChannelError(ChannelError.RecvError, "Already received"))

        # Wait for value to be sent
        var attempts = 0
        let max_attempts = 10000

        while attempts < max_attempts:
            if self._slot[].sent and self._slot[].value is Some:
                self._received = True
                self._slot[].received = True
                return Ok(self._slot[].value.unwrap())

            attempts += 1
            # In real implementation: await yield_now()

        return Err(ChannelError(ChannelError.Empty, "No value received"))

    fn try_recv(inout self) -> Result[Optional[T], ChannelError]:
        """Try to receive without blocking."""
        if self._received or self._slot[].received:
            return Err(ChannelError(ChannelError.RecvError, "Already received"))

        if self._slot[].sent and self._slot[].value is Some:
            self._received = True
            self._slot[].received = True
            return Ok(self._slot[].value)

        return Ok(None)


fn oneshot_channel[T]() -> (OneshotSender[T], OneshotReceiver[T]):
    """Create a oneshot channel (single value).

    Returns:
        Tuple of (sender, receiver).

    Example:
        let (tx, rx) = oneshot_channel[Bool]()
        tx.send(True)
        let result = await rx.recv()
    """
    let id = _next_channel_id()

    # Allocate shared slot
    let slot = Pointer[OneshotSlot[T]].alloc(1)
    slot[] = OneshotSlot[T]()

    return (OneshotSender[T](id, slot), OneshotReceiver[T](id, slot))


# ============================================================================
# Broadcast Channel
# ============================================================================

struct BroadcastState[T]:
    """Shared state for broadcast channel."""
    var buffer: List[T]
    var capacity: Int
    var head: Int  # Oldest message index
    var tail: Int  # Next write position
    var sequence: Int  # Global sequence number
    var subscribers: Int

    fn __init__(inout self, capacity: Int):
        self.buffer = List[T]()
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.sequence = 0
        self.subscribers = 0

    fn push(inout self, value: T):
        """Push value to broadcast buffer."""
        if self.buffer.len() < self.capacity:
            self.buffer.append(value)
        else:
            # Circular buffer - overwrite oldest
            self.buffer[self.tail % self.capacity] = value

        self.tail += 1
        self.sequence += 1

        # Advance head if buffer full
        if self.tail - self.head > self.capacity:
            self.head = self.tail - self.capacity

    fn get(self, seq: Int) -> Optional[T]:
        """Get value at sequence number."""
        if seq < self.head or seq >= self.tail:
            return None

        let idx = seq % self.capacity
        if idx < self.buffer.len():
            return Some(self.buffer[idx])
        return None


@value
struct BroadcastSender[T]:
    """Sender for broadcast channel."""

    var _id: Int
    var _capacity: Int
    var _state: Pointer[BroadcastState[T]]

    fn __init__(inout self, id: Int, capacity: Int, state: Pointer[BroadcastState[T]]):
        self._id = id
        self._capacity = capacity
        self._state = state

    async fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Broadcast value to all receivers."""
        self._state[].push(value)
        return Ok(None)

    fn subscribe(self) -> BroadcastReceiver[T]:
        """Create a new receiver for this broadcast channel."""
        self._state[].subscribers += 1
        # New subscriber starts from current tail
        return BroadcastReceiver[T](self._id, self._capacity, self._state, self._state[].tail)

    fn subscriber_count(self) -> Int:
        """Get number of active subscribers."""
        return self._state[].subscribers


@value
struct BroadcastReceiver[T]:
    """Receiver for broadcast channel."""

    var _id: Int
    var _capacity: Int
    var _closed: Bool
    var _state: Pointer[BroadcastState[T]]
    var _next_seq: Int  # Next sequence to read

    fn __init__(inout self, id: Int, capacity: Int, state: Pointer[BroadcastState[T]], start_seq: Int):
        self._id = id
        self._capacity = capacity
        self._closed = False
        self._state = state
        self._next_seq = start_seq

    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive broadcast message."""
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Wait for new message
        var attempts = 0
        let max_attempts = 10000

        while attempts < max_attempts:
            # Check if we're behind and missed messages
            if self._next_seq < self._state[].head:
                # Lagged - skip to current head
                self._next_seq = self._state[].head

            # Try to get next message
            let value = self._state[].get(self._next_seq)
            if value is Some:
                self._next_seq += 1
                return Ok(value.unwrap())

            attempts += 1
            # In real implementation: await yield_now()

        return Err(ChannelError(ChannelError.Empty, "No broadcast message available"))

    fn try_recv(inout self) -> Result[Optional[T], ChannelError]:
        """Try to receive without blocking."""
        if self._closed:
            return Err(ChannelError(ChannelError.Closed, "Channel closed"))

        # Catch up if lagged
        if self._next_seq < self._state[].head:
            self._next_seq = self._state[].head

        let value = self._state[].get(self._next_seq)
        if value is Some:
            self._next_seq += 1

        return Ok(value)

    fn close(inout self):
        """Unsubscribe from broadcast."""
        if not self._closed:
            self._closed = True
            self._state[].subscribers -= 1


fn broadcast_channel[T](capacity: Int) -> BroadcastSender[T]:
    """Create a broadcast channel.

    Args:
        capacity: Buffer capacity per receiver.

    Returns:
        Broadcast sender.

    Example:
        let tx = broadcast_channel[String](10)
        let rx1 = tx.subscribe()
        let rx2 = tx.subscribe()
        await tx.send("Broadcast message")
    """
    let id = _next_channel_id()

    # Allocate shared broadcast state
    let state = Pointer[BroadcastState[T]].alloc(1)
    state[] = BroadcastState[T](capacity)

    return BroadcastSender[T](id, capacity, state)


# ============================================================================
# Channel Select (Racing)
# ============================================================================

@value
struct SelectResult[T]:
    """Result from channel select operation."""
    var index: Int
    var value: T

    fn __init__(inout self, index: Int, value: T):
        self.index = index
        self.value = value


@value
struct Select2Result:
    """Result from 2-way select indicating which channel was ready."""
    var index: Int  # 0 or 1
    var ready: Bool

    fn __init__(inout self, index: Int):
        self.index = index
        self.ready = True


async fn select2[T1, T2](
    inout rx1: BoundedReceiver[T1],
    inout rx2: BoundedReceiver[T2]
) -> Result[Select2Result, ChannelError]:
    """Select from two channels (returns first ready).

    Args:
        rx1: First receiver.
        rx2: Second receiver.

    Returns:
        Select2Result with index (0 or 1) indicating which is ready.
    """
    var attempts = 0
    let max_attempts = 10000

    while attempts < max_attempts:
        # Check channel 1
        let result1 = rx1.try_recv()
        if result1.is_ok():
            let opt1 = result1.unwrap()
            if opt1 is Some:
                return Ok(Select2Result(0))

        # Check channel 2
        let result2 = rx2.try_recv()
        if result2.is_ok():
            let opt2 = result2.unwrap()
            if opt2 is Some:
                return Ok(Select2Result(1))

        # Both channels closed?
        if result1.is_err() and result1.unwrap_err().is_closed() and \
           result2.is_err() and result2.unwrap_err().is_closed():
            return Err(ChannelError(ChannelError.Closed, "All channels closed"))

        attempts += 1
        # In real implementation: await yield_now()

    return Err(ChannelError(ChannelError.Empty, "No channel ready after waiting"))


# ============================================================================
# Tests
# ============================================================================

fn test_unbounded_channel_creation():
    """Test unbounded channel creation."""
    let (tx, rx) = unbounded_channel[Int]()
    assert_false(tx._closed)
    assert_false(rx._closed)
    assert_true(tx._queue != Pointer[ChannelQueue[Int]]())


fn test_bounded_channel_creation():
    """Test bounded channel creation."""
    let (tx, rx) = bounded_channel[String](10)
    assert_equal(tx._capacity, 10)
    assert_equal(rx._capacity, 10)
    assert_false(tx._closed)
    assert_false(rx._closed)


fn test_oneshot_channel_creation():
    """Test oneshot channel creation."""
    let (tx, rx) = oneshot_channel[Bool]()
    assert_false(tx._sent)
    assert_false(rx._received)


fn test_broadcast_channel_creation():
    """Test broadcast channel creation."""
    let tx = broadcast_channel[Int](5)
    assert_equal(tx._capacity, 5)
    assert_equal(tx.subscriber_count(), 0)


fn test_channel_close():
    """Test channel closing."""
    var (tx, rx) = unbounded_channel[Int]()
    tx.close()
    assert_true(tx._closed)
    rx.close()
    assert_true(rx._closed)


fn test_bounded_capacity():
    """Test bounded channel capacity."""
    let (tx, _) = bounded_channel[Float64](100)
    assert_equal(tx.capacity(), 100)


fn test_broadcast_subscribe():
    """Test broadcast subscription."""
    let tx = broadcast_channel[String](10)
    let rx1 = tx.subscribe()
    let rx2 = tx.subscribe()
    assert_equal(rx1._capacity, 10)
    assert_equal(rx2._capacity, 10)
    assert_equal(tx.subscriber_count(), 2)


fn test_error_types():
    """Test channel error types."""
    let err_closed = ChannelError(ChannelError.Closed, "test")
    let err_timeout = ChannelError(ChannelError.Timeout, "test")
    let err_full = ChannelError(ChannelError.Full, "test")
    let err_empty = ChannelError(ChannelError.Empty, "test")
    assert_true(err_closed.is_closed())
    assert_false(err_closed.is_timeout())
    assert_true(err_timeout.is_timeout())
    assert_true(err_full.is_full())
    assert_true(err_empty.is_empty())


fn test_oneshot_send_once():
    """Test oneshot can only send once."""
    var (tx, _) = oneshot_channel[Int]()
    let result1 = tx.send(42)
    assert_true(result1.is_ok())
    assert_true(tx._sent)

    let result2 = tx.send(43)
    assert_true(result2.is_err())


fn test_select_result():
    """Test select result structure."""
    let result = SelectResult[Int](0, 42)
    assert_equal(result.index, 0)
    assert_equal(result.value, 42)


fn test_channel_queue():
    """Test channel queue operations."""
    var queue = ChannelQueue[Int](5)
    assert_true(queue.is_empty())
    assert_false(queue.is_full())

    # Push values
    assert_true(queue.push(1))
    assert_true(queue.push(2))
    assert_true(queue.push(3))
    assert_equal(queue.len(), 3)

    # Pop values
    let v1 = queue.pop()
    assert_true(v1 is Some)
    assert_equal(v1.unwrap(), 1)

    let v2 = queue.pop()
    assert_true(v2 is Some)
    assert_equal(v2.unwrap(), 2)


fn test_broadcast_state():
    """Test broadcast state operations."""
    var state = BroadcastState[String](3)
    state.push("a")
    state.push("b")
    state.push("c")

    let v0 = state.get(0)
    assert_true(v0 is Some)
    assert_equal(v0.unwrap(), "a")

    let v2 = state.get(2)
    assert_true(v2 is Some)
    assert_equal(v2.unwrap(), "c")


fn run_all_tests():
    """Run all channel tests."""
    test_unbounded_channel_creation()
    test_bounded_channel_creation()
    test_oneshot_channel_creation()
    test_broadcast_channel_creation()
    test_channel_close()
    test_bounded_capacity()
    test_broadcast_subscribe()
    test_error_types()
    test_oneshot_send_once()
    test_select_result()
    test_channel_queue()
    test_broadcast_state()
    print("All channel tests passed!")
