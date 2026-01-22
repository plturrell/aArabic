# Async Channels Module - Day 102
# Provides async channel-based communication primitives

from collections import List, Optional
from time.time import Duration

# ============================================================================
# Channel Errors
# ============================================================================

@value
struct ChannelError:
    """Channel operation error."""
    var kind: ErrorKind
    var message: String
    
    @value
    struct ErrorKind:
        """Error types for channel operations."""
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
        return self.kind == ErrorKind.Closed
    
    fn is_timeout(self) -> Bool:
        return self.kind == ErrorKind.Timeout


# ============================================================================
# Unbounded Channel
# ============================================================================

@value
struct UnboundedSender[T]:
    """Sender for unbounded channel."""
    
    var _id: Int
    var _closed: Bool
    
    fn __init__(inout self, id: Int):
        self._id = id
        self._closed = False
    
    async fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Send value through channel.
        
        Args:
            value: Value to send.
        
        Returns:
            Ok if sent successfully, Err if channel closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement actual send logic
        return Ok(None)
    
    fn close(inout self):
        """Close the sending end of the channel."""
        self._closed = True


@value
struct UnboundedReceiver[T]:
    """Receiver for unbounded channel."""
    
    var _id: Int
    var _closed: Bool
    
    fn __init__(inout self, id: Int):
        self._id = id
        self._closed = False
    
    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive value from channel.
        
        Returns:
            Received value or error if channel closed/empty.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement actual recv logic
        # For now, return error - real implementation would wait for value
        return Err(ChannelError(ChannelError.ErrorKind.Empty, "No value available"))
    
    async fn try_recv(inout self) -> Result[Optional[T], ChannelError]:
        """Try to receive without blocking.
        
        Returns:
            Some(value) if available, None if empty, Err if closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement non-blocking recv
        return Ok(None)
    
    fn close(inout self):
        """Close the receiving end of the channel."""
        self._closed = True


fn unbounded_channel[T]() -> (UnboundedSender[T], UnboundedReceiver[T]):
    """Create an unbounded MPSC channel.
    
    Returns:
        Tuple of (sender, receiver).
    
    Example:
        let (tx, rx) = unbounded_channel[Int]()
        await tx.send(42)
        let value = await rx.recv()
    """
    let id = 0  # TODO: Generate unique ID
    return (UnboundedSender[T](id), UnboundedReceiver[T](id))


# ============================================================================
# Bounded Channel
# ============================================================================

@value
struct BoundedSender[T]:
    """Sender for bounded channel."""
    
    var _id: Int
    var _capacity: Int
    var _closed: Bool
    
    fn __init__(inout self, id: Int, capacity: Int):
        self._id = id
        self._capacity = capacity
        self._closed = False
    
    async fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Send value, waiting if channel is full.
        
        Args:
            value: Value to send.
        
        Returns:
            Ok if sent, Err if closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement bounded send with backpressure
        return Ok(None)
    
    async fn try_send(inout self, value: T) -> Result[None, ChannelError]:
        """Try to send without blocking.
        
        Args:
            value: Value to send.
        
        Returns:
            Ok if sent, Err if full or closed.
        """
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement non-blocking send
        return Err(ChannelError(ChannelError.ErrorKind.Full, "Channel full"))
    
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
    
    fn __init__(inout self, id: Int, capacity: Int):
        self._id = id
        self._capacity = capacity
        self._closed = False
    
    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive value, waiting if channel is empty."""
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement bounded recv
        return Err(ChannelError(ChannelError.ErrorKind.Empty, "No value available"))
    
    async fn try_recv(inout self) -> Result[Optional[T], ChannelError]:
        """Try to receive without blocking."""
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement non-blocking recv
        return Ok(None)
    
    fn close(inout self):
        """Close the receiving end."""
        self._closed = True


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
    let id = 0  # TODO: Generate unique ID
    return (BoundedSender[T](id, capacity), BoundedReceiver[T](id, capacity))


# ============================================================================
# Oneshot Channel
# ============================================================================

@value
struct OneshotSender[T]:
    """Sender for oneshot channel (single value)."""
    
    var _id: Int
    var _sent: Bool
    
    fn __init__(inout self, id: Int):
        self._id = id
        self._sent = False
    
    fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Send single value.
        
        Args:
            value: Value to send.
        
        Returns:
            Ok if sent, Err if already sent.
        """
        if self._sent:
            return Err(ChannelError(ChannelError.ErrorKind.SendError, "Already sent"))
        
        self._sent = True
        # TODO: Implement oneshot send
        return Ok(None)


@value
struct OneshotReceiver[T]:
    """Receiver for oneshot channel."""
    
    var _id: Int
    var _received: Bool
    
    fn __init__(inout self, id: Int):
        self._id = id
        self._received = False
    
    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive single value."""
        if self._received:
            return Err(ChannelError(ChannelError.ErrorKind.RecvError, "Already received"))
        
        self._received = True
        # TODO: Implement oneshot recv
        return Err(ChannelError(ChannelError.ErrorKind.Empty, "No value available"))


fn oneshot_channel[T]() -> (OneshotSender[T], OneshotReceiver[T]):
    """Create a oneshot channel (single value).
    
    Returns:
        Tuple of (sender, receiver).
    
    Example:
        let (tx, rx) = oneshot_channel[Bool]()
        tx.send(True)
        let result = await rx.recv()
    """
    let id = 0  # TODO: Generate unique ID
    return (OneshotSender[T](id), OneshotReceiver[T](id))


# ============================================================================
# Broadcast Channel
# ============================================================================

@value
struct BroadcastSender[T]:
    """Sender for broadcast channel."""
    
    var _id: Int
    var _capacity: Int
    
    fn __init__(inout self, id: Int, capacity: Int):
        self._id = id
        self._capacity = capacity
    
    async fn send(inout self, value: T) -> Result[None, ChannelError]:
        """Broadcast value to all receivers."""
        # TODO: Implement broadcast send
        return Ok(None)
    
    fn subscribe(self) -> BroadcastReceiver[T]:
        """Create a new receiver for this broadcast channel."""
        return BroadcastReceiver[T](self._id, self._capacity)


@value
struct BroadcastReceiver[T]:
    """Receiver for broadcast channel."""
    
    var _id: Int
    var _capacity: Int
    var _closed: Bool
    
    fn __init__(inout self, id: Int, capacity: Int):
        self._id = id
        self._capacity = capacity
        self._closed = False
    
    async fn recv(inout self) -> Result[T, ChannelError]:
        """Receive broadcast message."""
        if self._closed:
            return Err(ChannelError(ChannelError.ErrorKind.Closed, "Channel closed"))
        
        # TODO: Implement broadcast recv
        return Err(ChannelError(ChannelError.ErrorKind.Empty, "No value available"))
    
    fn close(inout self):
        """Unsubscribe from broadcast."""
        self._closed = True


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
    let id = 0  # TODO: Generate unique ID
    return BroadcastSender[T](id, capacity)


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


async fn select2[T1, T2](
    rx1: BoundedReceiver[T1],
    rx2: BoundedReceiver[T2]
) -> Result[SelectResult, ChannelError]:
    """Select from two channels (returns first ready).
    
    Args:
        rx1: First receiver.
        rx2: Second receiver.
    
    Returns:
        SelectResult with index (0 or 1) and value.
    """
    # TODO: Implement select using runtime support
    return Err(ChannelError(ChannelError.ErrorKind.Empty, "Not implemented"))


# ============================================================================
# Tests
# ============================================================================

fn test_unbounded_channel_creation():
    """Test unbounded channel creation."""
    let (tx, rx) = unbounded_channel[Int]()
    assert_equal(tx._id, 0)
    assert_equal(rx._id, 0)
    assert_false(tx._closed)
    assert_false(rx._closed)


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


fn test_channel_close():
    """Test channel closing."""
    var (tx, rx) = unbounded_channel[Int]()
    tx.close()
    assert_true(tx._closed)
    rx.close()
    assert_true(rx._closed)


fn test_bounded_capacity():
    """Test bounded channel capacity."""
    let (tx, _) = bounded_channel[Float](100)
    assert_equal(tx.capacity(), 100)


fn test_broadcast_subscribe():
    """Test broadcast subscription."""
    let tx = broadcast_channel[String](10)
    let rx1 = tx.subscribe()
    let rx2 = tx.subscribe()
    assert_equal(rx1._capacity, 10)
    assert_equal(rx2._capacity, 10)


fn test_error_types():
    """Test channel error types."""
    let err_closed = ChannelError(ChannelError.ErrorKind.Closed, "test")
    let err_timeout = ChannelError(ChannelError.ErrorKind.Timeout, "test")
    assert_true(err_closed.is_closed())
    assert_false(err_closed.is_timeout())
    assert_true(err_timeout.is_timeout())


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
    print("All channel tests passed! ✅")
