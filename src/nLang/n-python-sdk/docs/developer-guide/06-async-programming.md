# Chapter 06: Async Programming

**Version:** 1.0.0  
**Audience:** Intermediate Developers  
**Prerequisites:** Basic Mojo, memory safety concepts  
**Estimated Time:** 60 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Async/Await Basics](#asyncawait-basics)
3. [Futures and Tasks](#futures-and-tasks)
4. [Channels](#channels)
5. [Concurrent Patterns](#concurrent-patterns)
6. [Error Handling](#error-handling)
7. [Runtime Internals](#runtime-internals)
8. [Best Practices](#best-practices)

---

## Introduction

### What is Async Programming?

Async programming allows you to write concurrent code that's efficient and easy to understand. Instead of blocking while waiting for I/O, async functions can yield control to other tasks.

```mojo
# Synchronous - blocks entire thread
fn fetch_sync(url: String) -> Data:
    let response = http.get(url)  # Blocks here
    return response.data

# Asynchronous - non-blocking
async fn fetch_async(url: String) -> Data:
    let response = await http.get(url)  # Yields here
    return response.data
```

### Why Async?

- ⚡ **Performance**: Handle thousands of concurrent operations
- 🎯 **Efficiency**: Don't waste CPU while waiting
- 🧩 **Composability**: Build complex workflows easily
- 🔒 **Safety**: Memory-safe concurrency (no data races)

### Mojo's Async Features

- `async` functions and blocks
- `await` expressions
- Channels for communication
- Futures for representing async operations
- Built-in async runtime (5,950 lines)
- Zero-cost abstractions

---

## Async/Await Basics

### 2.1 Async Functions

Define an async function:

```mojo
async fn fetch_data(url: String) -> Result[Data, Error]:
    let response = await http.get(url)?
    let data = await response.json()?
    return Ok(data)
```

**Key points:**
- Use `async fn` keyword
- Can `await` other async functions
- Return type is wrapped in Future implicitly
- Can use `?` for error propagation

### 2.2 Await Expressions

```mojo
async fn example():
    # Await a future
    let result = await some_async_fn()
    
    # Await with error handling
    let data = await fetch_data(url)?
    
    # Multiple awaits
    let a = await task1()
    let b = await task2()
    print(f"Results: {a}, {b}")
```

### 2.3 Running Async Code

```mojo
# Main function can be async
async fn main():
    let data = await fetch_data("https://api.example.com")
    print(data)

# Or use spawn to run async from sync context
fn main():
    let handle = spawn(async {
        await fetch_data("https://api.example.com")
    })
    
    # Wait for completion
    let result = handle.await()
}
```

### 2.4 Async Blocks

```mojo
fn main():
    # Create async block
    let future = async {
        let a = await task1()
        let b = await task2()
        return a + b
    }
    
    # Spawn it
    let handle = spawn(future)
    let result = handle.await()
}
```

---

## Futures and Tasks

### 3.1 Futures

A Future represents an async computation:

```mojo
from async import Future

async fn compute() -> Int:
    await sleep(1.0)
    return 42

fn main():
    let future: Future[Int] = compute()
    # Future hasn't started yet - lazy evaluation
    
    let handle = spawn(future)
    let result = handle.await()  # Block until complete
    print(result)  # 42
}
```

### 3.2 Spawning Tasks

```mojo
from async import spawn, sleep

async fn worker(id: Int):
    print(f"Worker {id} starting")
    await sleep(1.0)
    print(f"Worker {id} done")

async fn main():
    # Spawn multiple tasks
    let t1 = spawn(worker(1))
    let t2 = spawn(worker(2))
    let t3 = spawn(worker(3))
    
    # Wait for all
    await t1
    await t2
    await t3
    
    print("All workers done")
}
```

### 3.3 Task Handles

```mojo
async fn long_computation() -> Int:
    await sleep(5.0)
    return 42

async fn main():
    let handle = spawn(long_computation())
    
    # Do other work while task runs
    print("Doing other work...")
    await sleep(1.0)
    
    # Wait for result
    let result = await handle
    print(f"Result: {result}")
}
```

### 3.4 Joining Multiple Tasks

```mojo
from async import join, spawn

async fn main():
    let futures = [
        spawn(fetch("url1")),
        spawn(fetch("url2")),
        spawn(fetch("url3")),
    ]
    
    # Wait for all to complete
    let results = await join(futures)
    
    for result in results:
        print(result)
}
```

---

## Channels

### 4.1 Channel Basics

Channels enable communication between async tasks:

```mojo
from async import Channel

async fn producer(ch: Channel[Int]):
    for i in range(10):
        await ch.send(i)
        await sleep(0.1)
    ch.close()

async fn consumer(ch: Channel[Int]):
    while True:
        match await ch.recv():
            case Some(value):
                print(f"Received: {value}")
            case None:
                break  # Channel closed

async fn main():
    let ch = Channel[Int]::new()
    
    spawn(producer(ch.clone()))
    spawn(consumer(ch))
    
    await sleep(2.0)  # Let them run
}
```

### 4.2 Buffered Channels

```mojo
# Unbuffered - synchronous
let ch = Channel[Int]::new()

# Buffered - can send without receiver
let ch = Channel[Int]::with_capacity(10)

async fn example():
    let ch = Channel[String]::with_capacity(5)
    
    # These don't block (buffer has space)
    await ch.send("msg1")
    await ch.send("msg2")
    await ch.send("msg3")
    
    # Receive
    let msg = await ch.recv()
}
```

### 4.3 Select - Multiple Channels

```mojo
from async import select

async fn main():
    let ch1 = Channel[Int]::new()
    let ch2 = Channel[String]::new()
    
    # Wait on multiple channels
    match await select([ch1, ch2]):
        case 0, value:  # Received from ch1
            print(f"Int: {value}")
        case 1, value:  # Received from ch2
            print(f"String: {value}")
}
```

### 4.4 Channel Patterns

```mojo
# Fan-out: One producer, multiple consumers
async fn fan_out():
    let ch = Channel[Int]::new()
    
    spawn(producer(ch.clone()))
    spawn(consumer(ch.clone(), 1))
    spawn(consumer(ch.clone(), 2))
    spawn(consumer(ch.clone(), 3))

# Fan-in: Multiple producers, one consumer
async fn fan_in():
    let ch = Channel[Int]::new()
    
    spawn(producer(ch.clone(), 1))
    spawn(producer(ch.clone(), 2))
    spawn(producer(ch.clone(), 3))
    spawn(consumer(ch))
```

---

## Concurrent Patterns

### 5.1 Parallel Execution

```mojo
async fn parallel_fetch():
    # Start all fetches concurrently
    let f1 = spawn(fetch("url1"))
    let f2 = spawn(fetch("url2"))
    let f3 = spawn(fetch("url3"))
    
    # Await all results
    let r1 = await f1
    let r2 = await f2
    let r3 = await f3
    
    return [r1, r2, r3]
```

### 5.2 Race Condition

```mojo
from async import race

async fn main():
    # Wait for first to complete
    match await race([
        fetch("primary_server"),
        fetch("backup_server"),
    ]):
        case 0, result:  # Primary won
            print(f"Primary: {result}")
        case 1, result:  # Backup won
            print(f"Backup: {result}")
}
```

### 5.3 Timeout Pattern

```mojo
from async import timeout, sleep

async fn with_timeout():
    match await timeout(fetch("slow_server"), 5.0):
        case Ok(result):
            print(f"Success: {result}")
        case Err(TimeoutError):
            print("Request timed out")
}
```

### 5.4 Retry Pattern

```mojo
async fn retry[T](
    operation: async fn() -> Result[T, Error],
    max_attempts: Int
) -> Result[T, Error]:
    for attempt in 1..=max_attempts:
        match await operation():
            case Ok(result):
                return Ok(result)
            case Err(error):
                if attempt == max_attempts:
                    return Err(error)
                await sleep(1.0 * attempt)  # Exponential backoff
    
    return Err(Error("Max retries exceeded"))
```

### 5.5 Pipeline Pattern

```mojo
async fn pipeline():
    let ch1 = Channel[String]::new()
    let ch2 = Channel[String]::new()
    let ch3 = Channel[String]::new()
    
    # Stage 1: Fetch
    spawn(async {
        for url in urls:
            let data = await fetch(url)
            await ch1.send(data)
        ch1.close()
    })
    
    # Stage 2: Process
    spawn(async {
        while let Some(data) = await ch1.recv():
            let processed = await process(data)
            await ch2.send(processed)
        ch2.close()
    })
    
    # Stage 3: Store
    spawn(async {
        while let Some(data) = await ch2.recv():
            await store(data)
        ch3.close()
    })
}
```

---

## Error Handling

### 6.1 Async Result Types

```mojo
async fn fallible_operation() -> Result[Data, Error]:
    let response = await http.get(url)?  # Propagate error
    
    if response.status != 200:
        return Err(Error(f"Bad status: {response.status}"))
    
    let data = await response.json()?
    return Ok(data)
```

### 6.2 Try-Catch in Async

```mojo
async fn safe_operation():
    match await fallible_operation():
        case Ok(data):
            print(f"Success: {data}")
        case Err(error):
            print(f"Error: {error}")
            # Handle error...
}
```

### 6.3 Panic in Async

```mojo
async fn might_panic():
    let value = await compute()
    if value < 0:
        panic("Unexpected negative value")
    return value

# Panics propagate to the spawning task
async fn main():
    let handle = spawn(might_panic())
    
    match await handle.catch():
        case Ok(result):
            print(f"Success: {result}")
        case Err(panic_msg):
            print(f"Task panicked: {panic_msg}")
}
```

---

## Runtime Internals

### 7.1 Task Scheduling

The async runtime uses a work-stealing scheduler:

```
┌─────────────────────────────────────┐
│          Task Queue                  │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│  │Task│ │Task│ │Task│ │Task│ ...   │
│  └────┘ └────┘ └────┘ └────┘       │
└─────────────────────────────────────┘
         ↓       ↓       ↓       ↓
    ┌────────┐ ┌────────┐ ┌────────┐
    │Worker 1│ │Worker 2│ │Worker 3│
    └────────┘ └────────┘ └────────┘
```

**Features:**
- Work-stealing for load balancing
- Thread-per-core model
- Lock-free task queues
- Efficient task scheduling

### 7.2 Executor

```mojo
from async import Executor

fn main():
    # Create executor with 4 threads
    let executor = Executor::new(4)
    
    # Spawn tasks
    let handle = executor.spawn(async {
        await some_work()
    })
    
    # Block until complete
    let result = handle.await()
    
    # Shutdown executor
    executor.shutdown()
}
```

### 7.3 I/O Multiplexing

The runtime uses platform-specific I/O:

- **Linux**: io_uring
- **macOS**: kqueue
- **Windows**: IOCP

```mojo
# Efficient I/O handling
async fn handle_connections():
    let listener = TcpListener::bind("0.0.0.0:8080")?
    
    while True:
        let conn = await listener.accept()
        spawn(handle_client(conn))  # Non-blocking
}
```

### 7.4 Memory Management

```mojo
# Tasks have their own stack
async fn recursive_task(n: Int) -> Int:
    if n == 0:
        return 1
    return n * await recursive_task(n - 1)

# Runtime manages:
# - Task allocation
# - Stack growth
# - Cleanup on completion
```

---

## Best Practices

### 8.1 Avoid Blocking in Async

```mojo
# BAD - Blocks the entire thread
async fn bad_example():
    let data = blocking_io_call()  # Don't do this!
    return data

# GOOD - Use async I/O
async fn good_example():
    let data = await async_io_call()
    return data
```

### 8.2 Use Channels for Communication

```mojo
# GOOD - Channels are async-safe
async fn producer(ch: Channel[Int]):
    await ch.send(42)

# AVOID - Shared mutable state
var global_data: Int = 0  # Data race risk
async fn unsafe_increment():
    global_data += 1  # Not safe!
```

### 8.3 Structured Concurrency

```mojo
# GOOD - All tasks complete before function returns
async fn structured():
    let t1 = spawn(task1())
    let t2 = spawn(task2())
    
    await t1
    await t2
    # Both complete here
}

# AVOID - Fire-and-forget
async fn unstructured():
    spawn(task1())  # May not complete!
    # Function returns immediately
}
```

### 8.4 Timeout Long Operations

```mojo
from async import timeout

async fn with_timeout():
    match await timeout(slow_operation(), 5.0):
        case Ok(result):
            return result
        case Err(TimeoutError):
            return default_value()
}
```

### 8.5 Graceful Shutdown

```mojo
async fn server():
    let shutdown = Channel[()]::new()
    
    # Main loop
    spawn(async {
        while True:
            match await select([connections, shutdown]):
                case 0, conn:
                    handle_connection(conn)
                case 1, _:
                    break  # Shutdown signal
    })
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    await shutdown.send(())
}
```

---

## Examples

### Example 1: HTTP Server

```mojo
from async import TcpListener, spawn
from io import read_to_string, write_all

async fn handle_client(conn: TcpStream):
    let request = await read_to_string(&conn)?
    
    let response = "HTTP/1.1 200 OK\r\n\r\nHello, World!"
    await write_all(&conn, response)?
    
    conn.close()

async fn main():
    let listener = TcpListener::bind("127.0.0.1:8080")?
    print("Server listening on port 8080")
    
    while True:
        let (conn, addr) = await listener.accept()?
        print(f"Connection from {addr}")
        spawn(handle_client(conn))
}
```

### Example 2: Concurrent Downloads

```mojo
from async import spawn, join

async fn download(url: String) -> Result[Data, Error]:
    print(f"Downloading {url}")
    let response = await http.get(url)?
    let data = await response.bytes()?
    print(f"Downloaded {url}: {data.len()} bytes")
    return Ok(data)

async fn download_all(urls: List[String]) -> List[Data]:
    var tasks = List[Task[Data]]()
    
    # Start all downloads
    for url in urls:
        tasks.append(spawn(download(url)))
    
    # Wait for all
    var results = List[Data]()
    for task in tasks:
        match await task:
            case Ok(data):
                results.append(data)
            case Err(e):
                print(f"Download failed: {e}")
    
    return results
}
```

### Example 3: Producer-Consumer

```mojo
from async import Channel, spawn

async fn producer(ch: Channel[Int], n: Int):
    for i in range(n):
        print(f"Producing {i}")
        await ch.send(i)
        await sleep(0.1)
    ch.close()

async fn consumer(ch: Channel[Int], id: Int):
    while True:
        match await ch.recv():
            case Some(value):
                print(f"Consumer {id} got {value}")
                await sleep(0.2)  # Simulate work
            case None:
                print(f"Consumer {id} done")
                break

async fn main():
    let ch = Channel[Int]::with_capacity(5)
    
    spawn(producer(ch.clone(), 20))
    spawn(consumer(ch.clone(), 1))
    spawn(consumer(ch.clone(), 2))
    
    await sleep(3.0)
}
```

### Example 4: Rate Limiting

```mojo
from async import sleep, Channel

struct RateLimiter {
    tokens: Channel[()]
    rate: Float  # Tokens per second
    
    fn new(rate: Float) -> Self:
        let tokens = Channel[()]::with_capacity(rate as Int)
        spawn(Self::refill_loop(tokens.clone(), rate))
        return RateLimiter { tokens: tokens, rate: rate }
    
    async fn refill_loop(tokens: Channel[()], rate: Float):
        let interval = 1.0 / rate
        while True:
            await sleep(interval)
            let _ = tokens.try_send(())  # Add token
    
    async fn acquire(self):
        await self.tokens.recv()  # Wait for token
}

async fn rate_limited_requests():
    let limiter = RateLimiter::new(10.0)  # 10 req/sec
    
    for i in range(100):
        await limiter.acquire()
        spawn(make_request(i))
}
```

### Example 5: Async HTTP Client

```mojo
from async import http

async fn fetch_json(url: String) -> Result[JsonValue, Error]:
    let response = await http.get(url)?
    
    if response.status != 200:
        return Err(Error(f"HTTP {response.status}"))
    
    let data = await response.json()?
    return Ok(data)

async fn fetch_multiple(urls: List[String]):
    var tasks = List[Task]()
    
    for url in urls:
        tasks.append(spawn(fetch_json(url)))
    
    for task in tasks:
        match await task:
            case Ok(data):
                print(f"Success: {data}")
            case Err(error):
                print(f"Error: {error}")
}
```

---

## Summary

You've learned:
- ✅ `async`/`await` syntax and semantics
- ✅ Futures and task spawning
- ✅ Channels for communication
- ✅ Common concurrent patterns
- ✅ Error handling in async code
- ✅ Runtime internals
- ✅ Best practices for async programming

### Key Takeaways

1. **Non-blocking I/O** - Use `await` instead of blocking
2. **Spawn tasks** - Run operations concurrently
3. **Channels** - Safe communication between tasks
4. **Structured concurrency** - All tasks complete before return
5. **Error handling** - Use Result and match
6. **No data races** - Memory safety extends to concurrency

### Next Steps

- **Practice**: Build an async HTTP server
- **Explore**: [Metaprogramming](07-metaprogramming.md) for async macros
- **Read**: [Standard Library](03-stdlib-guide.md) async APIs
- **Study**: Runtime implementation in `stdlib/async/`

---

## Quick Reference

```mojo
# Async functions
async fn work() -> Result[T, E]

# Await
let result = await async_fn()

# Spawn tasks
let handle = spawn(async_fn())
await handle

# Channels
let ch = Channel[Int]::new()
await ch.send(42)
let value = await ch.recv()

# Multiple tasks
let results = await join([task1(), task2()])

# Timeout
await timeout(operation(), 5.0)?

# Select
match await select([ch1, ch2]):
    case 0, v: # From ch1
    case 1, v: # From ch2
```

---

**Next Chapter:** [Metaprogramming](07-metaprogramming.md)  
**Previous Chapter:** [Protocol System](05-protocol-system.md)

---

*Chapter 06: Async Programming*  
*Part of the Mojo SDK Developer Guide v1.0.0*  
*Last Updated: January 2026*
