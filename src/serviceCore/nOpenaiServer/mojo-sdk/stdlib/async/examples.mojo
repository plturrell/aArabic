# Async Examples - Day 103
# Real-world examples demonstrating async/await patterns

from async.async_io import AsyncFile, AsyncTcpListener, AsyncTcpStream, read_file, write_file
from async.async_channels import bounded_channel, unbounded_channel, broadcast_channel
from async.async_sync import AsyncMutex, AsyncRwLock, AsyncSemaphore, AsyncBarrier
from async.async_utils import spawn, join2, join_all, select2, CancellationToken
from async.async_stream import Stream, stream_from_range
from collections import List
from time.time import Duration

# ============================================================================
# Example 1: Concurrent File Processing
# ============================================================================

async fn process_multiple_files(paths: List[String]) -> Result[Int, IOError]:
    """Process multiple files concurrently.
    
    Demonstrates:
    - Spawning concurrent tasks
    - Joining results
    - Error handling across tasks
    """
    var handles = List[JoinHandle[Result[String, IOError]]]()
    
    # Spawn tasks for each file
    for path in paths:
        let handle = spawn(async {
            let content = await read_file(path)?
            return Ok(content.upper())
        })
        handles.append(handle)
    
    # Join all results
    let results = await join_all(handles)?
    
    # Count successful processing
    var count = 0
    for result in results:
        if result.is_ok():
            count += 1
    
    return Ok(count)


# ============================================================================
# Example 2: Producer-Consumer Pipeline
# ============================================================================

async fn producer_consumer_pipeline():
    """Demonstrates producer-consumer pattern with channels.
    
    Shows:
    - Bounded channels for backpressure
    - Multiple producers/consumers
    - Graceful shutdown
    """
    let (tx, rx) = bounded_channel[Int](10)
    
    # Producer task
    let producer = spawn(async {
        for i in range(100):
            await tx.send(i)?
        tx.close()
        return Ok(None)
    })
    
    # Consumer task
    let consumer = spawn(async {
        var sum = 0
        while True:
            let value = await rx.recv()
            if value.is_err():
                break
            sum += value.unwrap()
        return Ok(sum)
    })
    
    # Wait for completion
    _ = await producer.join()
    let result = await consumer.join()?
    
    print("Total sum: " + str(result))


# ============================================================================
# Example 3: Web Server with Shared State
# ============================================================================

@value
struct ServerState:
    """Shared server state."""
    var request_count: Int
    var active_connections: Int


async fn web_server():
    """Simple async web server.
    
    Demonstrates:
    - TCP listener with accept loop
    - Shared state with async mutex
    - Concurrent request handling
    """
    let state = AsyncMutex[ServerState](ServerState(0, 0))
    let addr = SocketAddress("0.0.0.0", 8080)
    var listener = AsyncTcpListener(addr)
    _ = await listener.bind()
    
    print("Server listening on :8080")
    
    while True:
        # Accept connection
        let stream = await listener.accept()?
        
        # Spawn handler
        spawn(async {
            # Update state
            let guard = await state.lock()
            guard.set(ServerState(
                guard.get().request_count + 1,
                guard.get().active_connections + 1
            ))
            
            # Handle request
            await handle_request(stream)
            
            # Update state
            let guard2 = await state.lock()
            guard2.set(ServerState(
                guard2.get().request_count,
                guard2.get().active_connections - 1
            ))
        })


async fn handle_request(stream: AsyncTcpStream):
    """Handle HTTP request."""
    let request = await stream.read(4096)?
    let response = "HTTP/1.1 200 OK\r\n\r\nHello, World!"
    _ = await stream.write_all(response)?
    _ = await stream.close()


# ============================================================================
# Example 4: Rate-Limited API Calls
# ============================================================================

async fn rate_limited_api_calls(urls: List[String]):
    """Make rate-limited API calls.
    
    Demonstrates:
    - Semaphore for rate limiting
    - Concurrent requests within limits
    - Error recovery
    """
    let semaphore = AsyncSemaphore(5)  # Max 5 concurrent requests
    var handles = List[JoinHandle[Result[String, NetworkError]]]()
    
    for url in urls:
        let handle = spawn(async {
            # Acquire permit (rate limit)
            let permit = await semaphore.acquire()
            
            # Make request
            let result = await http_get(url)
            
            # Permit automatically released on drop
            return result
        })
        handles.append(handle)
    
    # Collect results
    let results = await join_all(handles)?
    
    for result in results:
        match result:
            case Ok(data):
                print("Success: " + data[:50])
            case Err(error):
                print("Error: " + error.message)


async fn http_get(url: String) -> Result[String, NetworkError]:
    """Simple HTTP GET."""
    # TODO: Implement actual HTTP
    return Ok("Response data")


# ============================================================================
# Example 5: Parallel Data Processing
# ============================================================================

async fn parallel_map[T, U](
    values: List[T],
    func: Fn[U],
    concurrency: Int = 4
) -> List[U]:
    """Map function over list in parallel.
    
    Args:
        values: Input values.
        func: Transformation function.
        concurrency: Max parallel operations.
    
    Returns:
        Transformed values.
    """
    let semaphore = AsyncSemaphore(concurrency)
    var handles = List[JoinHandle[U]]()
    
    for value in values:
        let handle = spawn(async {
            let permit = await semaphore.acquire()
            return func(value)
        })
        handles.append(handle)
    
    return await join_all(handles)?


# ============================================================================
# Example 6: Timeout Pattern
# ============================================================================

async fn fetch_with_fallback(primary_url: String, backup_url: String) -> String:
    """Fetch from primary with timeout, fallback to backup.
    
    Demonstrates:
    - Timeout handling
    - Fallback strategies
    - Error recovery
    """
    # Try primary with 5s timeout
    let result = await with_timeout(
        Duration.from_secs(5),
        async { await http_get(primary_url) }
    )
    
    match result:
        case Ok(data):
            return data
        case Err(_):
            # Fallback to backup
            print("Primary timeout, using backup")
            return await http_get(backup_url).unwrap_or("Default")


# ============================================================================
# Example 7: Worker Pool
# ============================================================================

async fn worker_pool[T](tasks: List[T], num_workers: Int):
    """Process tasks with fixed worker pool.
    
    Demonstrates:
    - Worker pool pattern
    - Task distribution
    - Barrier synchronization
    """
    let (tx, rx) = bounded_channel[T](tasks.len())
    let barrier = AsyncBarrier(num_workers + 1)
    
    # Send all tasks
    for task in tasks:
        await tx.send(task)
    tx.close()
    
    # Spawn workers
    for i in range(num_workers):
        spawn(async {
            while True:
                let task = await rx.recv()
                if task.is_err():
                    break
                
                # Process task
                await process_task(task.unwrap())
            
            # Signal completion
            await barrier.wait()
        })
    
    # Wait for all workers
    await barrier.wait()
    print("All workers complete")


async fn process_task[T](task: T):
    """Process a single task."""
    # TODO: Implement task processing
    pass


# ============================================================================
# Example 8: Stream Processing Pipeline
# ============================================================================

async fn stream_pipeline():
    """Process stream with multiple transformations.
    
    Demonstrates:
    - Stream combinators
    - Lazy evaluation
    - Efficient processing
    """
    # Create stream
    var stream = stream_from_range(1, 100)
    
    # Apply transformations
    let processed = stream
        .filter(fn(x: Int) -> Bool { return x % 2 == 0 })  # Even numbers
        .map(fn(x: Int) -> Int { return x * x })            # Square
        .take(10)                                            # First 10
    
    # Collect results
    let results = await processed.collect()
    print("Processed " + str(len(results)) + " values")


# ============================================================================
# Example 9: Cancellable Long-Running Task
# ============================================================================

async fn cancellable_computation():
    """Long-running task with cancellation.
    
    Demonstrates:
    - Cancellation tokens
    - Graceful cancellation
    - Cleanup on cancellation
    """
    let token = CancellationToken()
    
    # Start computation
    let handle = spawn(async {
        for i in range(1000000):
            # Check for cancellation
            if token.is_cancelled():
                print("Cancelled at iteration " + str(i))
                return -1
            
            # Do work
            await expensive_operation()
        
        return 1000000
    })
    
    # Cancel after 100ms
    spawn(async {
        await sleep(Duration.from_millis(100))
        token.cancel()
    })
    
    let result = await handle.join()
    print("Result: " + str(result))


async fn expensive_operation():
    """Simulate expensive operation."""
    # TODO: Actual work
    pass


async fn sleep(duration: Duration):
    """Sleep for duration."""
    # TODO: Implement async sleep
    pass


# ============================================================================
# Example 10: Load Balancer
# ============================================================================

async fn load_balancer(requests: List[String]):
    """Simple load balancer distributing requests.
    
    Demonstrates:
    - Round-robin distribution
    - Broadcast for server pool
    - Health checking
    """
    let server_pool = broadcast_channel[String](10)
    
    # Spawn servers
    for i in range(3):
        let rx = server_pool.subscribe()
        spawn(async {
            while True:
                let request = await rx.recv()
                if request.is_err():
                    break
                
                await handle_server_request(i, request.unwrap())
        })
    
    # Distribute requests
    for request in requests:
        await server_pool.send(request)
    
    print("All requests distributed")


async fn handle_server_request(server_id: Int, request: String):
    """Handle request on specific server."""
    print("Server " + str(server_id) + " handling: " + request)


# ============================================================================
# Tests
# ============================================================================

fn test_example_types():
    """Test example type definitions."""
    let state = ServerState(0, 0)
    assert_equal(state.request_count, 0)
    assert_equal(state.active_connections, 0)


fn run_all_tests():
    """Run all example tests."""
    test_example_types()
    print("All example tests passed! ✅")
