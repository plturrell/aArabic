# Chapter 15: Tutorials

**Version:** 1.0.0  
**Audience:** All Levels  
**Prerequisites:** Basic Mojo knowledge recommended  
**Estimated Time:** Varies by tutorial

---

## Table of Contents

1. [Introduction](#introduction)
2. [Beginner Tutorials](#beginner-tutorials)
3. [Intermediate Tutorials](#intermediate-tutorials)
4. [Advanced Tutorials](#advanced-tutorials)
5. [Project Tutorials](#project-tutorials)

---

## Introduction

These hands-on tutorials will help you learn Mojo by building real projects. Each tutorial includes:
- ✅ Complete working code
- ✅ Step-by-step instructions
- ✅ Explanations of concepts
- ✅ Common pitfalls to avoid
- ✅ Next steps for learning

**How to use these tutorials:**
1. Create a new project directory
2. Follow along step-by-step
3. Experiment with the code
4. Complete the exercises at the end

---

## Beginner Tutorials

### Tutorial 1: Calculator (30 minutes)

**What you'll learn:**
- Functions and parameters
- Control flow (if/match)
- Error handling
- User input

#### Step 1: Create the project

```bash
mojo-pkg new calculator
cd calculator
```

#### Step 2: Basic operations (src/main.mojo)

```mojo
fn add(a: Float, b: Float) -> Float:
    return a + b

fn subtract(a: Float, b: Float) -> Float:
    return a - b

fn multiply(a: Float, b: Float) -> Float:
    return a * b

fn divide(a: Float, b: Float) -> Result[Float, String]:
    if b == 0.0:
        return Err("Cannot divide by zero")
    return Ok(a / b)

fn main():
    print("=== Simple Calculator ===")
    
    # Test operations
    print(f"10 + 5 = {add(10.0, 5.0)}")
    print(f"10 - 5 = {subtract(10.0, 5.0)}")
    print(f"10 * 5 = {multiply(10.0, 5.0)}")
    
    match divide(10.0, 5.0):
        case Ok(result):
            print(f"10 / 5 = {result}")
        case Err(error):
            print(f"Error: {error}")
    
    # Test error case
    match divide(10.0, 0.0):
        case Ok(result):
            print(f"10 / 0 = {result}")
        case Err(error):
            print(f"Error: {error}")
```

#### Step 3: Run it

```bash
mojo run src/main.mojo
```

Expected output:
```
=== Simple Calculator ===
10 + 5 = 15.0
10 - 5 = 5.0
10 * 5 = 50.0
10 / 5 = 2.0
Error: Cannot divide by zero
```

#### Step 4: Add interactive mode

```mojo
from io import input

fn parse_float(s: String) -> Result[Float, String]:
    # Simplified parsing - production code would be more robust
    # In real implementation, use standard library parsing
    match Float::from_string(s):
        case Ok(value):
            return Ok(value)
        case Err(_):
            return Err("Invalid number")

fn main():
    print("=== Interactive Calculator ===")
    print("Enter 'quit' to exit")
    
    while True:
        # Get first number
        print("\nEnter first number: ", end="")
        let num1_str = input()
        if num1_str == "quit":
            break
        
        let num1 = match parse_float(num1_str):
            case Ok(n): n
            case Err(e):
                print(f"Error: {e}")
                continue
        
        # Get operation
        print("Enter operation (+, -, *, /): ", end="")
        let op = input()
        
        # Get second number
        print("Enter second number: ", end="")
        let num2_str = input()
        
        let num2 = match parse_float(num2_str):
            case Ok(n): n
            case Err(e):
                print(f"Error: {e}")
                continue
        
        # Perform calculation
        match op:
            case "+":
                print(f"Result: {add(num1, num2)}")
            case "-":
                print(f"Result: {subtract(num1, num2)}")
            case "*":
                print(f"Result: {multiply(num1, num2)}")
            case "/":
                match divide(num1, num2):
                    case Ok(result):
                        print(f"Result: {result}")
                    case Err(error):
                        print(f"Error: {error}")
            case _:
                print("Unknown operation")
    
    print("Goodbye!")
```

#### Exercises

1. Add more operations (power, modulo, square root)
2. Add a history feature
3. Support multiple operations in one line
4. Add unit tests

---

### Tutorial 2: Todo List (45 minutes)

**What you'll learn:**
- Structs and methods
- Collections (List)
- File I/O
- Memory safety

#### Step 1: Define the Task struct

```mojo
struct Task {
    id: Int
    title: String
    completed: Bool
    
    fn __init__(inout self, id: Int, title: String):
        self.id = id
        self.title = title
        self.completed = false
    
    fn display(self):
        let status = "✓" if self.completed else " "
        print(f"[{status}] {self.id}. {self.title}")
}
```

#### Step 2: Create TodoList struct

```mojo
from collections import List

struct TodoList {
    tasks: List[Task]
    next_id: Int
    
    fn __init__(inout self):
        self.tasks = List[Task]()
        self.next_id = 1
    
    fn add_task(inout self, title: String):
        self.tasks.append(Task(self.next_id, title))
        self.next_id += 1
        print(f"Added task: {title}")
    
    fn complete_task(inout self, id: Int) -> Bool:
        for i in range(self.tasks.len()):
            if self.tasks[i].id == id:
                self.tasks[i].completed = true
                print(f"Completed task {id}")
                return true
        print(f"Task {id} not found")
        return false
    
    fn delete_task(inout self, id: Int) -> Bool:
        for i in range(self.tasks.len()):
            if self.tasks[i].id == id:
                self.tasks.remove(i)
                print(f"Deleted task {id}")
                return true
        print(f"Task {id} not found")
        return false
    
    fn list_tasks(self):
        if self.tasks.len() == 0:
            print("No tasks!")
            return
        
        print("\n=== Your Tasks ===")
        for task in self.tasks:
            task.display()
}
```

#### Step 3: Add main menu

```mojo
from io import input

fn main():
    var todo_list = TodoList()
    
    print("=== Todo List Manager ===")
    
    while True:
        print("\nCommands:")
        print("  add <task>    - Add a new task")
        print("  complete <id> - Mark task as complete")
        print("  delete <id>   - Delete a task")
        print("  list          - Show all tasks")
        print("  quit          - Exit")
        print("\n> ", end="")
        
        let command = input()
        let parts = command.split()
        
        if parts.len() == 0:
            continue
        
        match parts[0]:
            case "add":
                if parts.len() < 2:
                    print("Usage: add <task>")
                    continue
                let title = " ".join(parts[1:])
                todo_list.add_task(title)
            
            case "complete":
                if parts.len() < 2:
                    print("Usage: complete <id>")
                    continue
                match Int::from_string(parts[1]):
                    case Ok(id):
                        todo_list.complete_task(id)
                    case Err(_):
                        print("Invalid task ID")
            
            case "delete":
                if parts.len() < 2:
                    print("Usage: delete <id>")
                    continue
                match Int::from_string(parts[1]):
                    case Ok(id):
                        todo_list.delete_task(id)
                    case Err(_):
                        print("Invalid task ID")
            
            case "list":
                todo_list.list_tasks()
            
            case "quit":
                print("Goodbye!")
                break
            
            case _:
                print("Unknown command")
```

#### Step 4: Add persistence

```mojo
from io import File, read_file, write_file

impl TodoList {
    fn save_to_file(self, path: String) -> Result[(), String]:
        var content = String()
        
        for task in self.tasks:
            content += f"{task.id}|{task.title}|{task.completed}\n"
        
        match write_file(path, content):
            case Ok(_):
                print(f"Saved to {path}")
                return Ok(())
            case Err(e):
                return Err(f"Failed to save: {e}")
    
    fn load_from_file(inout self, path: String) -> Result[(), String]:
        let content = match read_file(path):
            case Ok(data): data
            case Err(e): return Err(f"Failed to load: {e}")
        
        self.tasks = List[Task]()
        self.next_id = 1
        
        for line in content.split('\n'):
            if line.is_empty():
                continue
            
            let parts = line.split('|')
            if parts.len() != 3:
                continue
            
            let id = match Int::from_string(parts[0]):
                case Ok(n): n
                case Err(_): continue
            
            let title = parts[1]
            let completed = parts[2] == "true"
            
            var task = Task(id, title)
            task.completed = completed
            self.tasks.append(task)
            
            if id >= self.next_id:
                self.next_id = id + 1
        
        print(f"Loaded {self.tasks.len()} tasks")
        return Ok(())
}
```

#### Exercises

1. Add due dates for tasks
2. Add priority levels
3. Add categories/tags
4. Add search functionality
5. Export to CSV

---

## Intermediate Tutorials

### Tutorial 3: HTTP Server (60 minutes)

**What you'll learn:**
- Async programming
- Network I/O
- Request/response handling
- Routing

#### Step 1: Basic server

```mojo
from async import TcpListener, spawn
from io import read_to_string, write_all

async fn handle_client(conn: TcpStream):
    # Read request
    let request = match await read_to_string(&conn):
        case Ok(data): data
        case Err(e):
            print(f"Read error: {e}")
            return
    
    print(f"Request:\n{request}")
    
    # Simple response
    let response = """HTTP/1.1 200 OK
Content-Type: text/html
Content-Length: 27

<h1>Hello from Mojo!</h1>"""
    
    match await write_all(&conn, response):
        case Ok(_):
            print("Response sent")
        case Err(e):
            print(f"Write error: {e}")
    
    conn.close()

async fn main():
    let listener = match TcpListener::bind("127.0.0.1:8080"):
        case Ok(l): l
        case Err(e):
            print(f"Failed to bind: {e}")
            return
    
    print("Server listening on http://127.0.0.1:8080")
    
    while True:
        let (conn, addr) = match await listener.accept():
            case Ok(c): c
            case Err(e):
                print(f"Accept error: {e}")
                continue
        
        print(f"Connection from {addr}")
        spawn(handle_client(conn))
```

#### Step 2: Add routing

```mojo
struct Request {
    method: String
    path: String
    headers: Dict[String, String]
    body: String
    
    fn parse(data: String) -> Result[Request, String]:
        let lines = data.split('\n')
        if lines.len() == 0:
            return Err("Empty request")
        
        # Parse request line
        let parts = lines[0].split()
        if parts.len() < 3:
            return Err("Invalid request line")
        
        let method = parts[0]
        let path = parts[1]
        
        # Parse headers
        var headers = Dict[String, String]()
        var i = 1
        while i < lines.len() and !lines[i].is_empty():
            let header_parts = lines[i].split(": ")
            if header_parts.len() == 2:
                headers[header_parts[0]] = header_parts[1]
            i += 1
        
        # Body is after empty line
        let body = if i + 1 < lines.len():
            "\n".join(lines[i+1:])
        else:
            ""
        
        return Ok(Request {
            method: method,
            path: path,
            headers: headers,
            body: body,
        })
}

struct Response {
    status: Int
    headers: Dict[String, String]
    body: String
    
    fn new(status: Int, body: String) -> Response:
        var headers = Dict[String, String]()
        headers["Content-Type"] = "text/html"
        headers["Content-Length"] = String(body.len())
        
        return Response {
            status: status,
            headers: headers,
            body: body,
        }
    
    fn to_string(self) -> String:
        var result = f"HTTP/1.1 {self.status} OK\r\n"
        
        for (key, value) in self.headers:
            result += f"{key}: {value}\r\n"
        
        result += "\r\n"
        result += self.body
        
        return result
}

async fn route(request: Request) -> Response:
    match request.path:
        case "/":
            return Response::new(200, "<h1>Home Page</h1>")
        
        case "/about":
            return Response::new(200, "<h1>About Page</h1>")
        
        case "/api/hello":
            return Response::new(200, '{"message": "Hello, API!"}')
        
        case _:
            return Response::new(404, "<h1>404 Not Found</h1>")
```

#### Step 3: Add JSON API

```mojo
from collections import Dict

struct JsonResponse {
    data: Dict[String, String]
    
    fn to_json(self) -> String:
        var result = "{\n"
        var first = true
        
        for (key, value) in self.data:
            if !first:
                result += ",\n"
            result += f'  "{key}": "{value}"'
            first = false
        
        result += "\n}"
        return result
}

async fn api_users() -> Response:
    var data = Dict[String, String]()
    data["id"] = "1"
    data["name"] = "Alice"
    data["email"] = "alice@example.com"
    
    let json = JsonResponse { data: data }.to_json()
    
    var response = Response::new(200, json)
    response.headers["Content-Type"] = "application/json"
    
    return response
```

#### Exercises

1. Add POST request handling
2. Add URL query parameters
3. Add static file serving
4. Add middleware (logging, auth)
5. Add WebSocket support

---

### Tutorial 4: CLI Tool (45 minutes)

**What you'll learn:**
- Command-line argument parsing
- File system operations
- Error handling patterns
- User interaction

#### Complete CLI tool

```mojo
from io import File, read_file, write_file, list_files
from sys import args

struct Config {
    verbose: Bool
    input_file: String
    output_file: String
    
    fn parse_args(args: List[String]) -> Result[Config, String]:
        var verbose = false
        var input_file = ""
        var output_file = ""
        
        var i = 1  # Skip program name
        while i < args.len():
            match args[i]:
                case "-v" | "--verbose":
                    verbose = true
                
                case "-i" | "--input":
                    if i + 1 >= args.len():
                        return Err("--input requires a value")
                    input_file = args[i + 1]
                    i += 1
                
                case "-o" | "--output":
                    if i + 1 >= args.len():
                        return Err("--output requires a value")
                    output_file = args[i + 1]
                    i += 1
                
                case _:
                    return Err(f"Unknown argument: {args[i]}")
            
            i += 1
        
        if input_file.is_empty():
            return Err("--input is required")
        
        return Ok(Config {
            verbose: verbose,
            input_file: input_file,
            output_file: output_file,
        })
}

fn process_file(config: Config) -> Result[(), String]:
    if config.verbose:
        print(f"Reading from {config.input_file}")
    
    let content = match read_file(config.input_file):
        case Ok(data): data
        case Err(e): return Err(f"Failed to read: {e}")
    
    # Process content (example: convert to uppercase)
    let processed = content.uppercase()
    
    let output_path = if config.output_file.is_empty():
        config.input_file + ".out"
    else:
        config.output_file
    
    if config.verbose:
        print(f"Writing to {output_path}")
    
    match write_file(output_path, processed):
        case Ok(_):
            print(f"Success! Output written to {output_path}")
            return Ok(())
        case Err(e):
            return Err(f"Failed to write: {e}")
}

fn main():
    let config = match Config::parse_args(args()):
        case Ok(c): c
        case Err(e):
            print(f"Error: {e}")
            print("\nUsage:")
            print("  program -i INPUT [-o OUTPUT] [-v]")
            print("\nOptions:")
            print("  -i, --input FILE    Input file (required)")
            print("  -o, --output FILE   Output file (optional)")
            print("  -v, --verbose       Verbose output")
            return
    
    match process_file(config):
        case Ok(_):
            # Success message already printed
            pass
        case Err(e):
            print(f"Error: {e}")
}
```

#### Exercises

1. Add multiple input files
2. Add progress bar
3. Add color output
4. Add config file support
5. Add streaming for large files

---

## Advanced Tutorials

### Tutorial 5: Concurrent Web Scraper (90 minutes)

**What you'll learn:**
- Advanced async patterns
- Channels for coordination
- Error recovery
- Rate limiting

```mojo
from async import Channel, spawn, sleep, timeout
from collections import List, Set

struct Scraper {
    max_concurrent: Int
    rate_limit: Float  # Requests per second
    visited: Set[String]
    
    fn new(max_concurrent: Int, rate_limit: Float) -> Scraper:
        return Scraper {
            max_concurrent: max_concurrent,
            rate_limit: rate_limit,
            visited: Set[String](),
        }
    
    async fn scrape_url(self, url: String) -> Result[String, String]:
        print(f"Scraping: {url}")
        
        # Simulate HTTP request
        match await timeout(http.get(url), 10.0):
            case Ok(Ok(response)):
                return Ok(response.body)
            case Ok(Err(e)):
                return Err(f"HTTP error: {e}")
            case Err(TimeoutError):
                return Err("Request timed out")
    
    async fn worker(
        self,
        id: Int,
        url_ch: Channel[String],
        result_ch: Channel[Result[String, String]]
    ):
        print(f"Worker {id} started")
        
        while True:
            match await url_ch.recv():
                case Some(url):
                    # Rate limiting
                    await sleep(1.0 / self.rate_limit)
                    
                    let result = await self.scrape_url(url)
                    await result_ch.send(result)
                
                case None:
                    print(f"Worker {id} done")
                    break
    
    async fn scrape_all(inout self, urls: List[String]):
        let url_ch = Channel[String]::with_capacity(100)
        let result_ch = Channel[Result[String, String]]::with_capacity(100)
        
        # Start workers
        for i in range(self.max_concurrent):
            spawn(self.worker(i, url_ch.clone(), result_ch.clone()))
        
        # Send URLs
        spawn(async {
            for url in urls:
                await url_ch.send(url)
            url_ch.close()
        })
        
        # Collect results
        var results = List[Result[String, String]]()
        for _ in range(urls.len()):
            match await result_ch.recv():
                case Some(result):
                    results.append(result)
                case None:
                    break
        
        # Print summary
        var success_count = 0
        var error_count = 0
        
        for result in results:
            match result:
                case Ok(data):
                    success_count += 1
                    print(f"Success: {data.len()} bytes")
                case Err(error):
                    error_count += 1
                    print(f"Error: {error}")
        
        print(f"\nSummary:")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")
}
```

---

## Project Tutorials

### Tutorial 6: Complete REST API (2-3 hours)

A full REST API with database, authentication, and testing.

**Features:**
- User authentication (JWT)
- Database integration
- CRUD operations
- Input validation
- Error handling
- Unit tests

[See complete tutorial in separate file]

---

## Summary

You've learned how to build:
- ✅ Command-line tools
- ✅ Web servers
- ✅ Concurrent applications
- ✅ File processors
- ✅ Network clients

### Next Steps

1. **Combine concepts** - Build larger projects
2. **Optimize** - Profile and improve performance
3. **Test** - Write comprehensive test suites
4. **Deploy** - Package and distribute your apps
5. **Contribute** - Share your projects with the community

### Additional Resources

- [Best Practices](16-best-practices.md)
- [API Reference](14-api-reference.md)
- [Contributing](13-contributing.md)
- Example projects in `examples/`

---

**Previous Chapter:** [API Reference](14-api-reference.md)  
**Next Chapter:** [Best Practices](16-best-practices.md)

---

*Chapter 15: Tutorials*  
*Part of the Mojo SDK Developer Guide v1.0.0*  
*Last Updated: January 2026*
