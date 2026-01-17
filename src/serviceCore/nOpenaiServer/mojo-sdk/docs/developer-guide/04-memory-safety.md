# Chapter 04: Memory Safety

**Version:** 1.0.0  
**Audience:** All Developers  
**Prerequisites:** Basic Mojo syntax  
**Estimated Time:** 60 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Ownership System](#ownership-system)
3. [Borrowing Rules](#borrowing-rules)
4. [Lifetime Analysis](#lifetime-analysis)
5. [Move Semantics](#move-semantics)
6. [Common Patterns](#common-patterns)
7. [Error Messages](#error-messages)
8. [Best Practices](#best-practices)

---

## Introduction

### What is Memory Safety?

Memory safety means your program cannot:
- ❌ Access freed memory (use-after-free)
- ❌ Have dangling pointers
- ❌ Experience data races
- ❌ Leak memory (with proper patterns)
- ❌ Have null pointer dereferences

**All verified at compile time!**

### Why Memory Safety Matters

```mojo
# This would crash in C/C++:
fn unsafe_example():
    var ptr = allocate_memory()
    free(ptr)
    use(ptr)  # ❌ Use-after-free!

# In Mojo, this won't compile:
fn safe_example():
    var data = String("hello")
    take_ownership(data^)
    print(data)  # ✅ Compile error: value moved
```

### The Three Pillars

Mojo's memory safety is built on:

1. **Ownership** - Every value has a single owner
2. **Borrowing** - Temporary access without ownership transfer
3. **Lifetimes** - Ensuring references remain valid

---

## Ownership System

### 2.1 Basic Ownership

**Rule 1:** Each value has exactly one owner at a time.

```mojo
fn main():
    let s = String("hello")  # s owns the string
    # When s goes out of scope, the string is freed
```

**Rule 2:** When the owner goes out of scope, the value is dropped.

```mojo
fn example():
    {
        let s = String("temporary")
        print(s)
    }  # s dropped here - memory freed automatically
    # s no longer accessible
```

### 2.2 Ownership Transfer (Move)

```mojo
fn take_ownership(owned data: String):
    print(f"I own: {data}")
    # data is freed when function returns

fn main():
    var s = String("hello")
    take_ownership(s^)  # Ownership moved with ^
    # print(s)  # ✅ Compile error: s was moved
```

**The `^` operator** explicitly moves ownership.

### 2.3 Owned Parameters

```mojo
# Takes ownership
fn process(owned data: String) -> String:
    return data.uppercase()

# Keeps ownership  
fn peek(borrowed data: String):
    print(data)

# Modifies in place
fn modify(inout data: String):
    data += " modified"
```

### 2.4 Return Values

Ownership is transferred to the caller:

```mojo
fn create_string() -> String:
    let s = String("new")
    return s  # Ownership moved to caller

fn main():
    let my_string = create_string()
    # my_string now owns the string
```

### 2.5 Multiple Owners

Mojo prevents multiple owners:

```mojo
fn example():
    let s1 = String("hello")
    let s2 = s1  # ✅ Error: cannot copy, must move
    
    # Correct way:
    let s1 = String("hello")
    let s2 = s1^  # Move ownership
    # s1 no longer valid
```

---

## Borrowing Rules

### 3.1 Immutable Borrowing

```mojo
fn read_only(borrowed data: String):
    print(data)  # Can read
    # data += "x"  # ✅ Error: cannot mutate borrowed value

fn main():
    let s = String("hello")
    read_only(s)  # Borrow
    read_only(s)  # Can borrow multiple times
    print(s)      # Original still valid
```

**Rule:** Multiple immutable borrows are allowed simultaneously.

### 3.2 Mutable Borrowing

```mojo
fn append_text(inout data: String):
    data += " world"  # Can mutate

fn main():
    var s = String("hello")
    append_text(s)  # Mutable borrow
    print(s)        # "hello world"
```

**Rule:** Only one mutable borrow at a time.

### 3.3 Borrowing Rules Summary

| Situation | Allowed? | Example |
|-----------|----------|---------|
| Multiple immutable borrows | ✅ Yes | `f(s); g(s)` |
| Immutable + mutable borrow | ❌ No | `f(s); g(&mut s)` |
| Multiple mutable borrows | ❌ No | `f(&mut s); g(&mut s)` |
| Borrow + move | ❌ No | `f(s); take(s^)` |

### 3.4 Borrow Scopes

```mojo
fn example():
    var s = String("hello")
    
    {
        borrowed_ref(s)  # Borrow starts
    }  # Borrow ends
    
    modify(s)  # OK - no active borrows
```

### 3.5 The Borrow Checker

The compiler tracks all borrows:

```mojo
fn problematic():
    var s = String("hello")
    
    let r1 = &s      # Immutable borrow
    let r2 = &s      # OK - another immutable borrow
    let r3 = &mut s  # ✅ Error: cannot borrow as mutable
    
    print(r1)
```

Error message:
```
error[E0502]: cannot borrow `s` as mutable because it is also borrowed as immutable
  --> example.mojo:5:14
   |
3  |     let r1 = &s      # Immutable borrow
   |              -- immutable borrow occurs here
4  |     let r2 = &s      # OK - another immutable borrow
5  |     let r3 = &mut s  # Error: cannot borrow as mutable
   |              ^^^^^^ mutable borrow occurs here
6  |     
7  |     print(r1)
   |           -- immutable borrow later used here
```

---

## Lifetime Analysis

### 4.1 What are Lifetimes?

Lifetimes ensure references don't outlive the data they point to.

```mojo
fn dangling_reference() -> &String:
    let s = String("local")
    return &s  # ✅ Error: returning reference to local variable
}
```

### 4.2 Lifetime Annotations

```mojo
# Explicit lifetimes
fn longest['a](x: &'a String, y: &'a String) -> &'a String:
    if x.len() > y.len():
        return x
    return y

fn main():
    let s1 = String("short")
    let s2 = String("longer")
    let result = longest(&s1, &s2)
    print(result)  # "longer"
```

**Lifetime `'a`** means:
- Both inputs must live at least as long as `'a`
- The output lives exactly as long as `'a`
- `'a` is determined by the shortest input lifetime

### 4.3 Lifetime Elision

Often, lifetimes can be inferred:

```mojo
# Explicit
fn first['a](s: &'a String) -> &'a String:
    return s

# Elided (equivalent)
fn first(s: &String) -> &String:
    return s
```

**Elision Rules:**
1. Each parameter gets its own lifetime
2. If one input lifetime, it's assigned to output
3. If multiple inputs, explicit annotation required

### 4.4 Struct Lifetimes

```mojo
struct Wrapper['a] {
    data: &'a String
}

fn create_wrapper['a](s: &'a String) -> Wrapper['a]:
    return Wrapper { data: s }

fn main():
    let s = String("hello")
    let w = create_wrapper(&s)
    print(w.data)  # OK
    # drop(s)  # Would be error - w.data still references s
```

### 4.5 Multiple Lifetimes

```mojo
struct Context['a, 'b] {
    config: &'a Config
    data: &'b Data
}

fn process['a, 'b](
    ctx: &Context['a, 'b],
    input: &'a String
) -> &'b Data:
    # Can return ctx.data (lifetime 'b)
    # Cannot return input (lifetime 'a)
    return ctx.data
```

---

## Move Semantics

### 5.1 What is Moving?

Moving transfers ownership without copying:

```mojo
fn example():
    let s1 = String("hello")  # s1 owns data
    let s2 = s1^              # Ownership moved to s2
    # s1 is now invalid
    # print(s1)  # ✅ Compile error
    print(s2)    # OK
```

### 5.2 Move vs Copy

```mojo
# Move (transfer ownership)
let s1 = String("hello")
let s2 = s1^

# Copy (duplicate data) - requires Clone protocol
#[derive(Clone)]
struct Point { x: Int, y: Int }

let p1 = Point { x: 5, y: 10 }
let p2 = p1.clone()  # Both valid
print(p1.x)          # OK
print(p2.x)          # OK
```

### 5.3 Moving in Function Calls

```mojo
fn take(owned s: String):
    print(s)

fn main():
    let s = String("hello")
    take(s^)     # Move
    # take(s)    # ✅ Error: s already moved
```

### 5.4 Partial Moves

```mojo
struct Data {
    name: String
    age: Int
}

fn example():
    let d = Data { name: "Alice", age: 30 }
    
    let name = d.name^  # Move name out
    # print(d.name)     # ✅ Error: name was moved
    print(d.age)        # OK - age not moved
    # print(d)          # ✅ Error: d partially moved
```

### 5.5 Move and Drop

```mojo
struct Resource {
    handle: FileHandle
    
    fn __del__(owned self):
        # Called when Resource is dropped
        close(self.handle)
        print("Resource cleaned up")
}

fn main():
    let r = Resource { handle: open_file("data.txt") }
    # r automatically dropped at end of scope
}  # "Resource cleaned up" printed here
```

---

## Common Patterns

### 6.1 Builder Pattern

```mojo
struct ConfigBuilder {
    timeout: Int
    retries: Int
    
    fn new() -> Self:
        return ConfigBuilder {
            timeout: 30,
            retries: 3,
        }
    
    fn with_timeout(inout self, timeout: Int) -> &Self:
        self.timeout = timeout
        return &self
    
    fn with_retries(inout self, retries: Int) -> &Self:
        self.retries = retries
        return &self
    
    fn build(owned self) -> Config:
        return Config {
            timeout: self.timeout,
            retries: self.retries,
        }
}

# Usage
let config = ConfigBuilder::new()
    .with_timeout(60)
    .with_retries(5)
    .build()
```

### 6.2 RAII (Resource Acquisition Is Initialization)

```mojo
struct File {
    handle: FileHandle
    
    fn open(path: String) -> Result[File, Error]:
        match open_file(path):
            case Ok(handle):
                return Ok(File { handle: handle })
            case Err(e):
                return Err(e)
    
    fn __del__(owned self):
        close(self.handle)  # Automatic cleanup
}

fn process_file():
    let file = File::open("data.txt")?
    # Use file...
}  # File automatically closed here
```

### 6.3 Option and Result

```mojo
# Option - may or may not have a value
fn find(list: List[Int], target: Int) -> Option[Int]:
    for i in 0..list.len():
        if list[i] == target:
            return Some(i)
    return None

# Result - success or error
fn divide(a: Int, b: Int) -> Result[Int, String]:
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)
```

### 6.4 Interior Mutability

```mojo
struct RefCell[T] {
    value: T
    borrowed: BorrowState
    
    fn borrow(self) -> Ref[T]:
        # Runtime borrow checking
        if self.borrowed == .Mutable:
            panic("Already borrowed mutably")
        self.borrowed = .Immutable
        return Ref { value: &self.value }
    
    fn borrow_mut(inout self) -> RefMut[T]:
        if self.borrowed != .None:
            panic("Already borrowed")
        self.borrowed = .Mutable
        return RefMut { value: &self.value }
}
```

### 6.5 Smart Pointers

```mojo
# Box - heap allocation
struct Box[T] {
    ptr: *T
    
    fn new(value: T) -> Box[T]:
        let ptr = allocate[T]()
        ptr.* = value
        return Box { ptr: ptr }
    
    fn __del__(owned self):
        deallocate(self.ptr)
}

# Rc - reference counting
struct Rc[T] {
    ptr: *RcInner[T]
    
    fn clone(self) -> Self:
        self.ptr.ref_count += 1
        return Rc { ptr: self.ptr }
    
    fn __del__(owned self):
        self.ptr.ref_count -= 1
        if self.ptr.ref_count == 0:
            deallocate(self.ptr)
}
```

---

## Error Messages

### 7.1 Use After Move

```mojo
fn example():
    let s = String("hello")
    take_ownership(s^)
    print(s)  # Error
```

Error:
```
error[E0382]: use of moved value: `s`
  --> example.mojo:4:11
   |
2  |     let s = String("hello")
   |         - move occurs because `s` has type `String`
3  |     take_ownership(s^)
   |                    -- value moved here
4  |     print(s)
   |           ^ value used here after move
   |
help: consider cloning the value if the type implements Clone
   |
3  |     take_ownership(s.clone())
   |                      ++++++++
```

### 7.2 Borrow Conflicts

```mojo
fn example():
    var s = String("hello")
    let r1 = &s
    let r2 = &mut s  # Error
    print(r1)
```

Error:
```
error[E0502]: cannot borrow `s` as mutable because it is also borrowed as immutable
  --> example.mojo:4:14
   |
3  |     let r1 = &s
   |              -- immutable borrow occurs here
4  |     let r2 = &mut s
   |              ^^^^^^ mutable borrow occurs here
5  |     print(r1)
   |           -- immutable borrow later used here
```

### 7.3 Dangling References

```mojo
fn dangling() -> &String:
    let s = String("local")
    return &s  # Error
}
```

Error:
```
error[E0106]: missing lifetime specifier
  --> example.mojo:2:18
   |
2  | fn dangling() -> &String:
   |                  ^ expected lifetime parameter
   |
note: this function's return type contains a borrowed value, 
      but there is no value for it to be borrowed from
help: consider giving it a 'static lifetime
```

---

## Best Practices

### 8.1 Prefer Borrowing

```mojo
# GOOD - Borrow when you don't need ownership
fn print_length(borrowed s: String):
    print(s.len())

# AVOID - Taking ownership unnecessarily
fn print_length(owned s: String):
    print(s.len())
```

### 8.2 Use References for Large Data

```mojo
# GOOD
fn process_large_data(borrowed data: &LargeStruct):
    # Work with reference

# AVOID - Expensive copy
fn process_large_data(data: LargeStruct):
    # Full copy of data
```

### 8.3 Return Owned Values

```mojo
# GOOD
fn create_user(name: String) -> User:
    return User { name: name }

# AVOID - Returning references to locals
fn create_user(name: String) -> &User:
    let user = User { name: name }
    return &user  # ✅ Error: dangling reference
```

### 8.4 Clone When Needed

```mojo
#[derive(Clone)]
struct Data {
    value: Int
}

fn share_data(data: Data):
    let copy1 = data.clone()
    let copy2 = data.clone()
    process(copy1)
    process(copy2)
```

### 8.5 Use Option and Result

```mojo
# GOOD - Explicit error handling
fn divide(a: Int, b: Int) -> Result[Int, String]:
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)

# AVOID - Panic on error
fn divide(a: Int, b: Int) -> Int:
    if b == 0:
        panic("Division by zero")  # Crashes program
    return a / b
```

---

## Summary

### Key Concepts

✅ **Ownership** - Each value has exactly one owner  
✅ **Borrowing** - Temporary access without ownership  
✅ **Lifetimes** - References must remain valid  
✅ **Move Semantics** - Transfer ownership explicitly  
✅ **Compile-time Safety** - No runtime overhead  

### Memory Safety Guarantees

With Mojo's system, you get:
- ✅ No use-after-free
- ✅ No double-free
- ✅ No null pointer dereferences
- ✅ No data races
- ✅ No dangling pointers

### Rules to Remember

1. Each value has one owner
2. Value dropped when owner goes out of scope
3. Multiple immutable borrows OR one mutable borrow
4. References must not outlive their referent
5. Use `^` to explicitly move

### Next Steps

- **Practice**: Write programs using ownership and borrowing
- **Read**: [Protocol System](05-protocol-system.md) for advanced patterns
- **Explore**: [Async Programming](06-async-programming.md) with safe concurrency
- **Study**: Standard library implementations

---

## Quick Reference

```mojo
# Ownership
let s = String("hello")        # s owns the string
take(s^)                       # Move ownership

# Borrowing
fn read(borrowed s: String)    # Immutable borrow
fn modify(inout s: String)     # Mutable borrow

# Lifetimes
fn longest['a](x: &'a String, y: &'a String) -> &'a String

# Smart pointers
Box[T]::new(value)            # Heap allocation
Rc[T]::new(value)             # Reference counting

# Patterns
Result[T, E]                  # Error handling
Option[T]                     # Nullable values
#[derive(Clone)]              # Enable cloning
```

---

**Next Chapter:** [Protocol System](05-protocol-system.md)  
**Previous Chapter:** [Standard Library Guide](03-stdlib-guide.md)

---

*Chapter 04: Memory Safety*  
*Part of the Mojo SDK Developer Guide v1.0.0*  
*Last Updated: January 2026*
