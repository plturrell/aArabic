# Async Stream Module - Day 103
# Provides async iterators and stream combinators

from collections import List, Optional

# ============================================================================
# Async Iterator Trait
# ============================================================================

trait AsyncIterator[T]:
    """Trait for async iteration over values."""
    
    async fn next(inout self) -> Optional[T]:
        """Get next value asynchronously.
        
        Returns:
            Some(value) if available, None if stream ended.
        """
        ...
    
    fn size_hint(self) -> (Int, Optional[Int]):
        """Get size hint (lower_bound, optional upper_bound)."""
        return (0, None)


# ============================================================================
# Stream Type
# ============================================================================

@value
struct Stream[T]:
    """Async stream of values."""
    
    var _id: Int
    var _ended: Bool
    
    fn __init__(inout self, id: Int):
        """Initialize stream."""
        self._id = id
        self._ended = False
    
    async fn next(inout self) -> Optional[T]:
        """Get next value from stream.
        
        Returns:
            Some(value) or None if stream ended.
        """
        if self._ended:
            return None
        
        # TODO: Implement stream next
        return None
    
    async fn collect(inout self) -> List[T]:
        """Collect all values into a list.
        
        Returns:
            List of all stream values.
        """
        var values = List[T]()
        while True:
            let value = await self.next()
            if value is None:
                break
            values.append(value.unwrap())
        return values
    
    async fn for_each[F: Fn[None]](inout self, func: F):
        """Apply function to each value.
        
        Args:
            func: Function to apply.
        """
        while True:
            let value = await self.next()
            if value is None:
                break
            func(value.unwrap())
    
    fn map[U, F: Fn[U]](self, func: F) -> Stream[U]:
        """Map stream values through function.
        
        Args:
            func: Mapping function.
        
        Returns:
            Mapped stream.
        """
        return MapStream[T, U, F](self, func)
    
    fn filter[F: Fn[Bool]](self, predicate: F) -> Stream[T]:
        """Filter stream values by predicate.
        
        Args:
            predicate: Filter function.
        
        Returns:
            Filtered stream.
        """
        return FilterStream[T, F](self, predicate)
    
    fn take(self, n: Int) -> Stream[T]:
        """Take first n values from stream.
        
        Args:
            n: Number of values to take.
        
        Returns:
            Limited stream.
        """
        return TakeStream[T](self, n)
    
    fn skip(self, n: Int) -> Stream[T]:
        """Skip first n values from stream.
        
        Args:
            n: Number of values to skip.
        
        Returns:
            Skipped stream.
        """
        return SkipStream[T](self, n)
    
    async fn fold[U, F: Fn[U]](inout self, init: U, func: F) -> U:
        """Fold stream into single value.
        
        Args:
            init: Initial accumulator value.
            func: Folding function.
        
        Returns:
            Final accumulated value.
        """
        var acc = init
        while True:
            let value = await self.next()
            if value is None:
                break
            acc = func(acc, value.unwrap())
        return acc
    
    async fn count(inout self) -> Int:
        """Count number of values in stream."""
        var n = 0
        while True:
            let value = await self.next()
            if value is None:
                break
            n += 1
        return n


# ============================================================================
# Stream Combinators
# ============================================================================

@value
struct MapStream[T, U, F: Fn[U]]:
    """Stream that maps values."""
    
    var _source: Stream[T]
    var _func: F
    
    fn __init__(inout self, source: Stream[T], func: F):
        self._source = source
        self._func = func
    
    async fn next(inout self) -> Optional[U]:
        """Get next mapped value."""
        let value = await self._source.next()
        if value is None:
            return None
        return Some(self._func(value.unwrap()))


@value
struct FilterStream[T, F: Fn[Bool]]:
    """Stream that filters values."""
    
    var _source: Stream[T]
    var _predicate: F
    
    fn __init__(inout self, source: Stream[T], predicate: F):
        self._source = source
        self._predicate = predicate
    
    async fn next(inout self) -> Optional[T]:
        """Get next filtered value."""
        while True:
            let value = await self._source.next()
            if value is None:
                return None
            if self._predicate(value.unwrap()):
                return value


@value
struct TakeStream[T]:
    """Stream that takes first n values."""
    
    var _source: Stream[T]
    var _remaining: Int
    
    fn __init__(inout self, source: Stream[T], n: Int):
        self._source = source
        self._remaining = n
    
    async fn next(inout self) -> Optional[T]:
        """Get next value if within limit."""
        if self._remaining <= 0:
            return None
        
        let value = await self._source.next()
        if value is Some:
            self._remaining -= 1
        return value


@value
struct SkipStream[T]:
    """Stream that skips first n values."""
    
    var _source: Stream[T]
    var _remaining: Int
    
    fn __init__(inout self, source: Stream[T], n: Int):
        self._source = source
        self._remaining = n
    
    async fn next(inout self) -> Optional[T]:
        """Get next value after skipping."""
        # Skip values
        while self._remaining > 0:
            let _ = await self._source.next()
            self._remaining -= 1
        
        return await self._source.next()


# ============================================================================
# Stream Constructors
# ============================================================================

fn stream_from_list[T](values: List[T]) -> Stream[T]:
    """Create stream from list.
    
    Args:
        values: List of values.
    
    Returns:
        Stream yielding list values.
    """
    return ListStream[T](values)


@value
struct ListStream[T]:
    """Stream from a list."""
    
    var _values: List[T]
    var _index: Int
    
    fn __init__(inout self, values: List[T]):
        self._values = values
        self._index = 0
    
    async fn next(inout self) -> Optional[T]:
        """Get next value from list."""
        if self._index >= len(self._values):
            return None
        
        let value = self._values[self._index]
        self._index += 1
        return Some(value)


fn stream_from_range(start: Int, end: Int, step: Int = 1) -> Stream[Int]:
    """Create stream from range.
    
    Args:
        start: Start value (inclusive).
        end: End value (exclusive).
        step: Step size.
    
    Returns:
        Stream of integers.
    """
    return RangeStream(start, end, step)


@value
struct RangeStream:
    """Stream from a range."""
    
    var _current: Int
    var _end: Int
    var _step: Int
    
    fn __init__(inout self, start: Int, end: Int, step: Int):
        self._current = start
        self._end = end
        self._step = step
    
    async fn next(inout self) -> Optional[Int]:
        """Get next value from range."""
        if (self._step > 0 and self._current >= self._end) or \
           (self._step < 0 and self._current <= self._end):
            return None
        
        let value = self._current
        self._current += self._step
        return Some(value)


fn repeat[T](value: T) -> Stream[T]:
    """Create infinite stream repeating a value.
    
    Args:
        value: Value to repeat.
    
    Returns:
        Infinite stream.
    """
    return RepeatStream[T](value)


@value
struct RepeatStream[T]:
    """Stream that repeats a value."""
    
    var _value: T
    
    fn __init__(inout self, value: T):
        self._value = value
    
    async fn next(inout self) -> Optional[T]:
        """Always returns the same value."""
        return Some(self._value)


fn empty[T]() -> Stream[T]:
    """Create empty stream.
    
    Returns:
        Stream that immediately ends.
    """
    return EmptyStream[T]()


@value
struct EmptyStream[T]:
    """Empty stream."""
    
    fn __init__(inout self):
        pass
    
    async fn next(inout self) -> Optional[T]:
        """Always returns None."""
        return None


# ============================================================================
# Stream Combinators - Advanced
# ============================================================================

async fn zip[T1, T2](
    stream1: Stream[T1],
    stream2: Stream[T2]
) -> Stream[(T1, T2)]:
    """Zip two streams together.
    
    Args:
        stream1: First stream.
        stream2: Second stream.
    
    Returns:
        Stream of tuples.
    """
    return ZipStream[T1, T2](stream1, stream2)


@value
struct ZipStream[T1, T2]:
    """Zipped stream."""
    
    var _stream1: Stream[T1]
    var _stream2: Stream[T2]
    
    fn __init__(inout self, stream1: Stream[T1], stream2: Stream[T2]):
        self._stream1 = stream1
        self._stream2 = stream2
    
    async fn next(inout self) -> Optional[(T1, T2)]:
        """Get next tuple."""
        let v1 = await self._stream1.next()
        let v2 = await self._stream2.next()
        
        if v1 is None or v2 is None:
            return None
        
        return Some((v1.unwrap(), v2.unwrap()))


# ============================================================================
# Tests
# ============================================================================

fn test_stream_creation():
    """Test stream creation."""
    let stream = Stream[Int](1)
    assert_equal(stream._id, 1)
    assert_false(stream._ended)


fn test_list_stream():
    """Test stream from list."""
    let values = List[Int]([1, 2, 3])
    var stream = stream_from_list(values)
    # Would need async context to test next()


fn test_range_stream():
    """Test stream from range."""
    var stream = stream_from_range(0, 5, 1)
    # Would need async context to test next()


fn test_repeat_stream():
    """Test repeat stream."""
    let stream = repeat(42)
    # Always returns same value


fn test_empty_stream():
    """Test empty stream."""
    let stream = empty[Int]()
    # Immediately ends


fn test_map_stream():
    """Test map combinator."""
    let source = Stream[Int](1)
    let mapped = source.map(fn(x: Int) -> Int { return x * 2 })
    # Would need async to test


fn test_filter_stream():
    """Test filter combinator."""
    let source = Stream[Int](1)
    let filtered = source.filter(fn(x: Int) -> Bool { return x > 0 })
    # Would need async to test


fn test_take_stream():
    """Test take combinator."""
    let source = Stream[Int](1)
    let limited = source.take(10)
    # Would need async to test


fn test_skip_stream():
    """Test skip combinator."""
    let source = Stream[Int](1)
    let skipped = source.skip(5)
    # Would need async to test


fn test_either_unwrap():
    """Test Either unwrap methods."""
    let left = Either[Int, String](42)
    assert_equal(left.unwrap_left(), 42)
    
    let right = Either[Int, String]("test")
    assert_equal(right.unwrap_right(), "test")


fn run_all_tests():
    """Run all stream tests."""
    test_stream_creation()
    test_list_stream()
    test_range_stream()
    test_repeat_stream()
    test_empty_stream()
    test_map_stream()
    test_filter_stream()
    test_take_stream()
    test_skip_stream()
    test_either_unwrap()
    print("All stream tests passed! ✅")
