# Chapter 16: Best Practices

**Audience:** All Levels  
**Prerequisites:** [Chapter 1: Getting Started](01-getting-started.md)  
**Estimated Time:** 60-90 minutes  
**Version:** 1.0.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Code Organization](#code-organization)
3. [Naming Conventions](#naming-conventions)
4. [Memory Management](#memory-management)
5. [Error Handling](#error-handling)
6. [Performance](#performance)
7. [Concurrency](#concurrency)
8. [Testing](#testing)
9. [Documentation](#documentation)
10. [Common Pitfalls](#common-pitfalls)

---

## Introduction

This chapter collects best practices, patterns, and conventions for writing idiomatic, maintainable, and efficient Mojo code.

**What you'll learn:**
- Industry-standard coding practices
- Mojo-specific patterns
- Performance optimization techniques
- Common mistakes to avoid

**Why follow best practices:**
- Code is easier to read and maintain
- Fewer bugs and issues
- Better performance
- Easier collaboration

---

## Code Organization

### Project Structure

**Recommended layout:**

```
my-project/
├── mojo.toml              # Package configuration
├── README.md              # Project documentation
├── LICENSE                # License file
│
├── src/                   # Source code
│   ├── main.mojo         # Entry point
│   ├── lib.mojo          # Library code
│   └── modules/          # Submodules
│       ├── parser.mojo
│       └── utils.mojo
│
├── tests/                 # Test files
│   ├── test_parser.mojo
│   └── test_utils.mojo
│
├── examples/              # Example code
│   └── basic.mojo
│
├── docs/                  # Documentation
│   └── api.md
│
└── benchmarks/            # Performance tests
    └── bench_parser.mojo
```

### Module Organization

**✅ Good:**
```mojo
# parser.mojo - Single responsibility

struct Token {
    kind: TokenKind
    value: String
}

struct Lexer {
    source: String
    position: Int
}

struct Parser {
    lexer: Lexer
    current_token: Token
}
```

**❌ Bad:**
```mojo
# everything.mojo - Too many responsibilities

struct Token { ... }
struct Lexer { ... }
struct Parser { ... }
struct Evaluator { ... }
struct CodeGenerator { ... }
struct Optimizer { ... }
# ... everything in one file
```

### Separation of Concerns

**✅ Good:**
```mojo
# data.mojo
struct User {
    name: String
    email: String
}

# validation.mojo
fn validate_email(email: String) -> Result[(), String]:
    if !email.contains("@"):
        return Err("Invalid email")
    return Ok(())

# storage.mojo
struct UserStorage {
    fn save(self, user: User) -> Result[(), Error]
    fn load(self, id: Int) -> Result[User, Error]
}
```

**❌ Bad:**
```mojo
# All mixed together
struct User {
    name: String
    email: String
    
    fn validate_email(self) -> Bool:
        # Validation logic in data struct
    
    fn save_to_database(self):
        # Database logic in data struct
}
```

---

## Naming Conventions

### General Rules

| Item | Convention | Example |
|------|------------|---------|
| Structs | PascalCase | `HttpClient`, `UserAccount` |
| Functions | snake_case | `parse_input`, `calculate_total` |
| Variables | snake_case | `user_name`, `total_count` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_SIZE`, `DEFAULT_TIMEOUT` |
| Protocols | PascalCase | `Drawable`, `Comparable` |
| Type Parameters | Single Capital | `T`, `K`, `V` |

### Meaningful Names

**✅ Good:**
```mojo
fn calculate_monthly_payment(principal: Float, rate: Float, months: Int) -> Float:
    let monthly_rate = rate / 12.0
    let payment = principal * (monthly_rate * pow(1 + monthly_rate, months)) / 
                  (pow(1 + monthly_rate, months) - 1)
    return payment
```

**❌ Bad:**
```mojo
fn calc(p: Float, r: Float, m: Int) -> Float:
    let mr = r / 12.0
    let pmt = p * (mr * pow(1 + mr, m)) / (pow(1 + mr, m) - 1)
    return pmt
```

### Boolean Names

**✅ Good:**
```mojo
let is_valid = true
let has_permission = false
let can_edit = true
let should_retry = false
```

**❌ Bad:**
```mojo
let valid = true        # Ambiguous
let permission = false  # Not clearly boolean
let edit = true         # Verb, not boolean
```

### Avoid Hungarian Notation

**✅ Good:**
```mojo
let name: String = "Alice"
let count: Int = 42
let users: List[User] = List()
```

**❌ Bad:**
```mojo
let str_name = "Alice"    # Type in name
let int_count = 42
let list_users = List()
```

---

## Memory Management

### Ownership Patterns

#### Pattern 1: Pass by Borrow (Default)

**Use when:** Function needs read-only access

```mojo
fn print_user(borrowed user: User):
    print(f"{user.name} ({user.email})")

# Caller retains ownership
let user = User("Alice", "alice@example.com")
print_user(user)  # user still valid here
```

#### Pattern 2: Pass by Mutable Borrow

**Use when:** Function needs to modify

```mojo
fn update_email(inout user: User, new_email: String):
    user.email = new_email

var user = User("Alice", "old@example.com")
update_email(user, "new@example.com")
# user modified, still owned by caller
```

#### Pattern 3: Pass by Move

**Use when:** Transferring ownership

```mojo
fn take_ownership(owned user: User):
    # user owned by this function
    process(user)
}  # user destroyed here

let user = User("Alice", "alice@example.com")
take_ownership(user^)  # Explicit move with ^
# user no longer accessible
```

### RAII Pattern

**✅ Good - Resource management:**
```mojo
struct File {
    handle: FileHandle
    
    fn __init__(inout self, path: String):
        self.handle = open(path)
    
    fn __del__(owned self):
        close(self.handle)  # Automatic cleanup
}

fn process_file(path: String):
    let file = File(path)
    # Use file
}  # File automatically closed
```

**❌ Bad - Manual cleanup:**
```mojo
fn process_file(path: String):
    let handle = open(path)
    # Use file
    close(handle)  # Easy to forget
}
```

### Avoid Unnecessary Clones

**✅ Good:**
```mojo
fn process(borrowed data: List[Int]):
    for item in data:
        print(item)
    # No clone needed
```

**❌ Bad:**
```mojo
fn process(borrowed data: List[Int]):
    let copy = data.clone()  # Unnecessary
    for item in copy:
        print(item)
}
```

### Use Slices for Views

**✅ Good:**
```mojo
fn first_half(borrowed data: []Int) -> []Int:
    return data[..data.len() / 2]  # No allocation
```

**❌ Bad:**
```mojo
fn first_half(borrowed data: List[Int]) -> List[Int]:
    var result = List[Int]()
    for i in range(data.len() / 2):
        result.append(data[i])  # Unnecessary allocation
    return result^
}
```

---

## Error Handling

### Use Result Type

**✅ Good:**
```mojo
fn divide(a: Int, b: Int) -> Result[Int, String]:
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)

match divide(10, 2):
    case Ok(result):
        print(f"Result: {result}")
    case Err(error):
        print(f"Error: {error}")
```

**❌ Bad - Using panic:**
```mojo
fn divide(a: Int, b: Int) -> Int:
    if b == 0:
        panic("Division by zero")  # Crashes program
    return a / b
```

### Chain Error Handling

**✅ Good:**
```mojo
fn load_and_parse(path: String) -> Result[Data, Error]:
    let content = read_file(path)?  # Propagate error
    let data = parse_json(content)?
    return Ok(data)
}
```

**❌ Bad - Nested matching:**
```mojo
fn load_and_parse(path: String) -> Result[Data, Error]:
    match read_file(path):
        case Ok(content):
            match parse_json(content):
                case Ok(data):
                    return Ok(data)
                case Err(e):
                    return Err(e)
        case Err(e):
            return Err(e)
}
```

### Provide Context

**✅ Good:**
```mojo
fn load_config(path: String) -> Result[Config, String]:
    let content = read_file(path)
        .map_err(|e| f"Failed to read config from {path}: {e}")?
    
    let config = parse_config(content)
        .map_err(|e| f"Failed to parse config: {e}")?
    
    return Ok(config)
}
```

**❌ Bad:**
```mojo
fn load_config(path: String) -> Result[Config, String]:
    let content = read_file(path)?  # No context
    let config = parse_config(content)?  # What failed?
    return Ok(config)
}
```

### Use Option for Nullable Values

**✅ Good:**
```mojo
fn find_user(id: Int) -> Option[User]:
    if let user = database.get(id):
        return Some(user)
    return None

match find_user(123):
    case Some(user):
        process(user)
    case None:
        print("User not found")
```

**❌ Bad - Using null-like patterns:**
```mojo
fn find_user(id: Int) -> User:
    if let user = database.get(id):
        return user
    return User()  # Invalid "null" user
}
```

---

## Performance

### Minimize Allocations

**✅ Good:**
```mojo
fn sum_array(borrowed arr: []Int) -> Int:
    var total = 0
    for x in arr:
        total += x
    return total
    # No allocations
```

**❌ Bad:**
```mojo
fn sum_array(borrowed arr: []Int) -> Int:
    var result = List[Int]()  # Unnecessary allocation
    for x in arr:
        result.append(x)
    return result.iter().sum()
}
```

### Use Iterators

**✅ Good:**
```mojo
fn process_large_file(path: String):
    for line in File(path).lines():  # Lazy iteration
        process_line(line)
    # Memory usage: O(1)
```

**❌ Bad:**
```mojo
fn process_large_file(path: String):
    let all_lines = File(path).read_all()  # Load entire file
    for line in all_lines:
        process_line(line)
    # Memory usage: O(n)
}
```

### Inline Small Functions

**✅ Good:**
```mojo
@always_inline
fn square(x: Int) -> Int:
    return x * x  # Will be inlined
```

### Pre-allocate Collections

**✅ Good:**
```mojo
fn create_list(size: Int) -> List[Int]:
    var list = List[Int]::with_capacity(size)  # Pre-allocate
    for i in range(size):
        list.append(i)
    return list^
}
```

**❌ Bad:**
```mojo
fn create_list(size: Int) -> List[Int]:
    var list = List[Int]()  # Will resize multiple times
    for i in range(size):
        list.append(i)
    return list^
}
```

### Avoid Unnecessary Copies

**✅ Good:**
```mojo
fn find_max(borrowed numbers: []Int) -> Int:
    var max_val = numbers[0]
    for num in numbers[1..]:
        if num > max_val:
            max_val = num
    return max_val
```

**❌ Bad:**
```mojo
fn find_max(borrowed numbers: []Int) -> Int:
    let copy = numbers.to_list()  # Unnecessary copy
    var max_val = copy[0]
    for num in copy[1..]:
        if num > max_val:
            max_val = num
    return max_val
}
```

---

## Concurrency

### Use Channels for Communication

**✅ Good:**
```mojo
async fn producer(ch: Channel[Int]):
    for i in range(10):
        await ch.send(i)
    ch.close()

async fn consumer(ch: Channel[Int]):
    while let Some(value) = await ch.recv():
        process(value)

async fn main():
    let ch = Channel[Int]::new()
    spawn(producer(ch.clone()))
    spawn(consumer(ch))
}
```

### Avoid Shared Mutable State

**✅ Good - Message passing:**
```mojo
async fn worker(id: Int, inbox: Channel[Task], outbox: Channel[Result]):
    while let Some(task) = await inbox.recv():
        let result = process(task)
        await outbox.send(result)
}
```

**❌ Bad - Shared state:**
```mojo
var shared_counter = 0  # Data race!

async fn worker():
    shared_counter += 1  # Not thread-safe
}
```

### Use Mutex for Shared State

**When necessary:**
```mojo
struct SharedCounter {
    mutex: Mutex[Int]
    
    fn increment(inout self):
        let guard = self.mutex.lock()
        *guard += 1
    }  # Automatically unlocked
    
    fn get(borrowed self) -> Int:
        let guard = self.mutex.lock()
        return *guard
    }
}
```

### Structured Concurrency

**✅ Good:**
```mojo
async fn fetch_all(urls: []String) -> List[Result]:
    var tasks = List[Task]()
    
    for url in urls:
        tasks.append(spawn(fetch(url)))
    
    var results = List[Result]()
    for task in tasks:
        results.append(await task)
    
    return results^
}  # All tasks complete before return
```

**❌ Bad - Unstructured:**
```mojo
async fn fetch_all(urls: []String):
    for url in urls:
        spawn(fetch(url))  # Fire and forget - no tracking
}  # Function returns before tasks complete
```

---

## Testing

### Test Organization

**Structure:**
```
tests/
├── unit/              # Unit tests
│   ├── test_parser.mojo
│   └── test_lexer.mojo
├── integration/       # Integration tests
│   └── test_compiler.mojo
└── e2e/              # End-to-end tests
    └── test_cli.mojo
```

### Unit Test Pattern

**✅ Good:**
```mojo
from testing import assert_equal, assert_true, test

@test
fn test_addition():
    assert_equal(add(2, 3), 5)
    assert_equal(add(0, 0), 0)
    assert_equal(add(-1, 1), 0)

@test
fn test_division():
    match divide(10, 2):
        case Ok(result):
            assert_equal(result, 5)
        case Err(_):
            panic("Unexpected error")
    
    match divide(10, 0):
        case Ok(_):
            panic("Should have failed")
        case Err(error):
            assert_true(error.contains("zero"))
```

### Test Naming

**✅ Good:**
```mojo
@test fn test_empty_list_returns_none()
@test fn test_valid_email_passes_validation()
@test fn test_negative_number_raises_error()
```

**❌ Bad:**
```mojo
@test fn test1()
@test fn test_stuff()
@test fn test()
```

### Use Test Fixtures

**✅ Good:**
```mojo
struct TestFixture {
    temp_dir: TempDir
    test_file: String
    
    fn setup() -> TestFixture:
        let temp_dir = TempDir::new()
        let test_file = temp_dir.path() + "/test.txt"
        write_file(test_file, "test content")
        return TestFixture { temp_dir, test_file }
    
    fn teardown(owned self):
        # Cleanup automatic via TempDir::__del__
    }
}

@test
fn test_file_operations():
    let fixture = TestFixture::setup()
    
    # Test code here
    let content = read_file(fixture.test_file)
    assert_equal(content, "test content")
}  # Automatic cleanup
```

### Property-Based Testing

**✅ Good:**
```mojo
@property_test
fn test_reverse_twice_is_identity(input: List[Int]):
    let result = input.reverse().reverse()
    assert_equal(result, input)

@property_test
fn test_sort_is_idempotent(input: List[Int]):
    let once = input.sort()
    let twice = once.sort()
    assert_equal(once, twice)
```

---

## Documentation

### Module Documentation

**✅ Good:**
```mojo
"""
# Parser Module

This module provides lexical analysis and parsing functionality
for the Mojo language.

## Example

```mojo
let lexer = Lexer::new(source_code)
let parser = Parser::new(lexer)
let ast = parser.parse()
```

## Error Handling

All parsing functions return `Result` types. Handle errors
appropriately.
"""

# Module code here
```

### Function Documentation

**✅ Good:**
```mojo
"""
Calculates the monthly payment for a loan.

# Arguments
- `principal`: The loan amount in dollars
- `annual_rate`: The annual interest rate (e.g., 0.05 for 5%)
- `months`: The number of months for the loan

# Returns
The monthly payment amount.

# Example
```mojo
let payment = calculate_monthly_payment(100000.0, 0.05, 360)
print(f"Monthly payment: ${payment}")
```

# Panics
Panics if `months` is zero or negative.
"""
fn calculate_monthly_payment(
    principal: Float,
    annual_rate: Float,
    months: Int
) -> Float:
    # Implementation
```

### Inline Comments

**✅ Good - Explain why:**
```mojo
fn optimize_query(query: String) -> String:
    # Remove redundant whitespace to reduce parsing overhead
    let trimmed = query.trim()
    
    # Convert to lowercase for case-insensitive matching
    return trimmed.lowercase()
}
```

**❌ Bad - Explain what:**
```mojo
fn optimize_query(query: String) -> String:
    let trimmed = query.trim()  # Trim the string
    return trimmed.lowercase()  # Convert to lowercase
}
```

---

## Common Pitfalls

### Pitfall 1: Borrowing After Move

**❌ Wrong:**
```mojo
let data = vec![1, 2, 3]
process(data^)  # Move
print(data)     # Error: data moved
```

**✅ Fix:**
```mojo
let data = vec![1, 2, 3]
process(data.clone()^)  # Clone before move
print(data)             # OK: data still valid
```

### Pitfall 2: Infinite Loops

**❌ Wrong:**
```mojo
var i = 0
while i < 10:
    print(i)
    # Forgot to increment i
}
```

**✅ Fix:**
```mojo
for i in range(10):
    print(i)
# Or:
var i = 0
while i < 10:
    print(i)
    i += 1
}
```

### Pitfall 3: Integer Overflow

**❌ Wrong:**
```mojo
let x: Int8 = 127
let y = x + 1  # Overflow!
```

**✅ Fix:**
```mojo
let x: Int8 = 127
let y = x.checked_add(1) match:
    case Ok(result): result
    case Err(_): panic("Overflow")
}
```

### Pitfall 4: Comparing Floats

**❌ Wrong:**
```mojo
let a: Float = 0.1 + 0.2
if a == 0.3:  # May fail due to precision
    print("Equal")
}
```

**✅ Fix:**
```mojo
let a: Float = 0.1 + 0.2
let epsilon = 0.00001
if abs(a - 0.3) < epsilon:
    print("Equal")
}
```

### Pitfall 5: Forgetting to Handle Errors

**❌ Wrong:**
```mojo
let file = open("data.txt")  # Ignores Result
```

**✅ Fix:**
```mojo
match open("data.txt"):
    case Ok(file):
        process(file)
    case Err(error):
        print(f"Error: {error}")
}
```

### Pitfall 6: Unused Results

**❌ Wrong:**
```mojo
list.push(item)  # Returns Result, ignored
```

**✅ Fix:**
```mojo
match list.push(item):
    case Ok(_):
        # Success
    case Err(error):
        handle_error(error)
}
```

---

## Summary

You've learned:
- ✅ Code organization patterns
- ✅ Naming conventions
- ✅ Memory management best practices
- ✅ Error handling patterns
- ✅ Performance optimization techniques
- ✅ Concurrency patterns
- ✅ Testing strategies
- ✅ Documentation standards
- ✅ Common pitfalls to avoid

### Golden Rules

1. **Prefer borrowing over moving** - Move only when necessary
2. **Use Result for errors** - Never panic in library code
3. **Test everything** - Unit, integration, and property tests
4. **Document public APIs** - Clear, helpful documentation
5. **Optimize last** - Correctness first, performance second
6. **Follow conventions** - Consistent code is maintainable code

### Next Steps

1. **Practice** - Apply these patterns in your code
2. **Review** - Read others' code for patterns
3. **Refactor** - Improve existing code
4. **Share** - Teach these practices to others

### Additional Resources

- [Memory Safety](04-memory-safety.md) - Deep dive
- [Async Programming](06-async-programming.md) - Concurrency patterns
- [Tutorials](15-tutorials.md) - Hands-on practice
- [API Reference](14-api-reference.md) - Complete API docs

---

**Previous Chapter:** [Tutorials](15-tutorials.md)  
**Next Chapter:** [Migration Guides](17-migration-guides.md)

---

*Chapter 16: Best Practices*  
*Part of the Mojo SDK Developer Guide v1.0.0*  
*Last Updated: January 2026*
