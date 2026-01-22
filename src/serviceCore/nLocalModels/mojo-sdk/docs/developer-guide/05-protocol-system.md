# Chapter 05: Protocol System

**Version:** 1.0.0  
**Audience:** Intermediate Developers  
**Prerequisites:** Basic Mojo syntax, structs  
**Estimated Time:** 45 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Protocol Basics](#protocol-basics)
3. [Implementing Protocols](#implementing-protocols)
4. [Automatic Derivation](#automatic-derivation)
5. [Conditional Conformance](#conditional-conformance)
6. [Protocol Inheritance](#protocol-inheritance)
7. [Associated Types](#associated-types)
8. [Best Practices](#best-practices)
9. [Common Patterns](#common-patterns)

---

## Introduction

### What are Protocols?

Protocols (also called traits or interfaces in other languages) define contracts that types must fulfill. They enable **protocol-oriented programming** - a powerful paradigm for writing flexible, reusable code.

```mojo
# Define a contract
protocol Drawable {
    fn draw(self)
}

# Types that implement the protocol can be used interchangeably
fn render(item: &dyn Drawable) {
    item.draw()  # Guaranteed to work!
}
```

### Why Use Protocols?

- ✅ **Abstraction**: Define behavior without implementation
- ✅ **Polymorphism**: Different types, same interface
- ✅ **Extensibility**: Add behavior to existing types
- ✅ **Type Safety**: Compile-time guarantees
- ✅ **Code Reuse**: Write generic algorithms

### Protocol System Features

Mojo's protocol system includes:
- Protocol definitions with methods and properties
- Protocol implementations for types
- **Automatic derivation** (`#[derive(Eq, Hash, Debug)]`)
- **Conditional conformance** (generic implementations)
- **Protocol inheritance** (one protocol extends another)
- **Associated types** (generic protocols)

---

## Protocol Basics

### Defining a Protocol

```mojo
protocol Printable {
    fn print(self)
}

protocol Serializable {
    fn serialize(self) -> String
    fn deserialize(data: String) -> Self
}
```

**Key points:**
- Use `protocol` keyword
- Define required methods
- Methods can have parameters and return types
- Methods can be static or instance methods

### Implementing a Protocol

```mojo
struct User {
    name: String
    age: Int
}

impl Printable for User {
    fn print(self) {
        print(f"User: {self.name}, age {self.age}")
    }
}

# Usage
fn main() {
    let user = User { name: "Alice", age: 30 }
    user.print()  # Output: User: Alice, age 30
}
```

### Multiple Implementations

A type can implement multiple protocols:

```mojo
impl Printable for User {
    fn print(self) {
        print(f"User: {self.name}")
    }
}

impl Serializable for User {
    fn serialize(self) -> String {
        return f"User({self.name},{self.age})"
    }
    
    fn deserialize(data: String) -> Self {
        # Parse and create User
        # ... implementation ...
    }
}
```

### Protocol Methods

```mojo
protocol Shape {
    # Instance method
    fn area(self) -> Float
    
    # Method with parameters
    fn scale(inout self, factor: Float)
    
    # Static method
    fn default_shape() -> Self
}

impl Shape for Circle {
    fn area(self) -> Float {
        return 3.14159 * self.radius * self.radius
    }
    
    fn scale(inout self, factor: Float) {
        self.radius *= factor
    }
    
    fn default_shape() -> Self {
        return Circle { radius: 1.0 }
    }
}
```

---

## Implementing Protocols

### Basic Implementation

```mojo
# Define protocol
protocol Comparable {
    fn compare(self, other: Self) -> Int
}

# Implement for Int
impl Comparable for Int {
    fn compare(self, other: Int) -> Int {
        if self < other: return -1
        if self > other: return 1
        return 0
    }
}

# Usage
fn find_max[T: Comparable](a: T, b: T) -> T {
    if a.compare(b) > 0:
        return a
    return b
}
```

### Protocol Requirements

All methods must be implemented:

```mojo
protocol Vehicle {
    fn start(inout self)
    fn stop(inout self)
    fn speed(self) -> Float
}

impl Vehicle for Car {
    fn start(inout self) {
        self.engine_running = true
        print("Car started")
    }
    
    fn stop(inout self) {
        self.engine_running = false
        print("Car stopped")
    }
    
    fn speed(self) -> Float {
        return self.current_speed
    }
}
```

### Implementation Validation

The compiler checks that all required methods are present:

```mojo
impl Vehicle for Bike {
    fn start(inout self) {
        print("Bike started")
    }
    # ERROR: Missing required methods 'stop' and 'speed'
}
```

---

## Automatic Derivation

### The `#[derive]` Attribute

Mojo can automatically generate protocol implementations:

```mojo
#[derive(Eq, Hash, Debug, Clone, Default)]
struct Point {
    x: Int
    y: Int
}

# Automatically generates:
# - Eq: equality comparison (==, !=)
# - Hash: hashing for collections
# - Debug: debug formatting
# - Clone: deep copying
# - Default: default values
```

### Derivable Protocols

| Protocol | Purpose | Generated Methods |
|----------|---------|-------------------|
| **Eq** | Equality | `eq(self, other) -> Bool` |
| **Hash** | Hashing | `hash(self) -> UInt64` |
| **Debug** | Debug printing | `debug(self) -> String` |
| **Clone** | Deep copy | `clone(self) -> Self` |
| **Default** | Default value | `default() -> Self` |

### Eq Derivation

```mojo
#[derive(Eq)]
struct Person {
    name: String
    age: Int
}

# Generated implementation:
impl Eq for Person {
    fn eq(self, other: Self) -> Bool {
        return self.name == other.name and self.age == other.age
    }
}

# Usage
let p1 = Person { name: "Alice", age: 30 }
let p2 = Person { name: "Alice", age: 30 }
print(p1 == p2)  # true
```

### Hash Derivation

```mojo
#[derive(Hash)]
struct Coordinate {
    x: Int
    y: Int
}

# Generated implementation:
impl Hash for Coordinate {
    fn hash(self) -> UInt64 {
        var h: UInt64 = 0
        h = h ^ hash(self.x)
        h = h ^ hash(self.y)
        return h
    }
}

# Usage with Dict
var visited = Dict[Coordinate, Bool]()
visited[Coordinate{x: 5, y: 10}] = true
```

### Debug Derivation

```mojo
#[derive(Debug)]
struct Rectangle {
    width: Float
    height: Float
}

# Generated implementation:
impl Debug for Rectangle {
    fn debug(self) -> String {
        var s = "Rectangle { "
        s += "width: " + debug(self.width)
        s += ", height: " + debug(self.height)
        s += " }"
        return s
    }
}

# Usage
let rect = Rectangle { width: 10.0, height: 5.0 }
print(debug(rect))  # Rectangle { width: 10.0, height: 5.0 }
```

### Clone Derivation

```mojo
#[derive(Clone)]
struct Node {
    value: Int
    children: List[Node]
}

# Generated implementation:
impl Clone for Node {
    fn clone(self) -> Self {
        return Node {
            value: clone(self.value),
            children: clone(self.children),
        }
    }
}

# Usage
let node1 = Node { value: 42, children: List[Node]() }
let node2 = node1.clone()  # Deep copy
```

### Default Derivation

```mojo
#[derive(Default)]
struct Config {
    timeout: Int
    retries: Int
    verbose: Bool
}

# Generated implementation:
impl Default for Config {
    fn default() -> Self {
        return Config {
            timeout: Int::default(),  # 0
            retries: Int::default(),  # 0
            verbose: Bool::default(), # false
        }
    }
}

# Usage
let config = Config::default()
```

### Multiple Derivations

```mojo
#[derive(Eq, Hash, Debug, Clone, Default)]
struct User {
    id: Int
    username: String
    email: String
}

# All 5 protocols implemented automatically!

fn main() {
    let u1 = User::default()
    let u2 = u1.clone()
    print(u1 == u2)      # true
    print(debug(u1))     # User { id: 0, ... }
    let h = hash(u1)     # Works!
}
```

---

## Conditional Conformance

### Generic Implementations

Implement a protocol for a generic type with constraints:

```mojo
# Option<T> is Eq if T is Eq
impl<T: Eq> Eq for Option<T> {
    fn eq(self, other: Self) -> Bool {
        match (self, other) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false
        }
    }
}

# Now Option<Int> automatically has Eq!
let opt1 = Some(42)
let opt2 = Some(42)
print(opt1 == opt2)  # true
```

### Vec Conformance

```mojo
# Vec<T> is Eq if T is Eq
impl<T: Eq> Eq for Vec<T> {
    fn eq(self, other: Self) -> Bool {
        if self.len() != other.len() {
            return false
        }
        for i in 0..self.len() {
            if self[i] != other[i] {
                return false
            }
        }
        return true
    }
}

# Vec<T> is Clone if T is Clone
impl<T: Clone> Clone for Vec<T> {
    fn clone(self) -> Self {
        var result = Vec<T>::new()
        for item in self {
            result.append(item.clone())
        }
        return result
    }
}
```

### Blanket Implementations

Implement a protocol for all types that satisfy a constraint:

```mojo
# Any type with Debug automatically gets ToString
impl<T: Debug> ToString for T {
    fn to_string(self) -> String {
        return debug(self)
    }
}

# Now ALL Debug types have to_string() for free!
#[derive(Debug)]
struct Point { x: Int, y: Int }

let p = Point { x: 5, y: 10 }
print(p.to_string())  # Works automatically!
```

### Multiple Constraints

```mojo
# Implement Display for HashMap if both K and V are Display
impl<K: Eq + Hash, V: Display> Display for HashMap<K, V> {
    fn display(self) -> String {
        var s = "{"
        for (key, value) in self {
            s += display(value)
            s += ", "
        }
        s += "}"
        return s
    }
}
```

---

## Protocol Inheritance

### Inheriting from One Protocol

```mojo
protocol Drawable {
    fn draw(self)
}

# Shape extends Drawable
protocol Shape: Drawable {
    fn area(self) -> Float
}

# Circle must implement both draw() and area()
impl Shape for Circle {
    fn draw(self) {
        print(f"Drawing circle at {self.center}")
    }
    
    fn area(self) -> Float {
        return 3.14159 * self.radius * self.radius
    }
}
```

### Multiple Inheritance

```mojo
protocol Printable {
    fn print(self)
}

protocol Comparable {
    fn compare(self, other: Self) -> Int
}

# Sortable requires both Printable and Comparable
protocol Sortable: Printable, Comparable {
    fn sort_key(self) -> Int
}
```

### Inheritance Chain

```mojo
protocol Animal {
    fn make_sound(self)
}

protocol Mammal: Animal {
    fn feed_young(self)
}

protocol Pet: Mammal {
    fn play(self)
}

# Dog must implement all three protocols
impl Pet for Dog {
    fn make_sound(self) {
        print("Woof!")
    }
    
    fn feed_young(self) {
        print("Nursing puppies")
    }
    
    fn play(self) {
        print("Playing fetch")
    }
}
```

---

## Associated Types

### Basic Associated Types

```mojo
protocol Iterator {
    type Item              # Associated type
    
    fn next(inout self) -> Option<Item>
}

impl Iterator for RangeIter {
    type Item = Int        # Bind associated type
    
    fn next(inout self) -> Option<Int> {
        if self.current <= self.end {
            let val = self.current
            self.current += 1
            return Some(val)
        }
        return None
    }
}
```

### Container Protocol

```mojo
protocol Container {
    type Item
    type Iter: Iterator
    
    fn len(self) -> Int
    fn is_empty(self) -> Bool
    fn iter(self) -> Iter
}

impl Container for Vec<T> {
    type Item = T
    type Iter = VecIter<T>
    
    fn len(self) -> Int {
        return self.size
    }
    
    fn is_empty(self) -> Bool {
        return self.size == 0
    }
    
    fn iter(self) -> VecIter<T> {
        return VecIter { vec: self, index: 0 }
    }
}
```

### Generic Functions with Associated Types

```mojo
fn sum_all[C: Container](container: C) -> C.Item
    where C.Item: Add {
    var total = C.Item::default()
    for item in container.iter() {
        total = total + item
    }
    return total
}
```

---

## Best Practices

### 1. Protocol Design

**DO:**
```mojo
protocol Serializable {
    fn serialize(self) -> String
    fn deserialize(data: String) -> Result<Self, Error>
}
```

**DON'T:**
```mojo
protocol Serializable {
    fn serialize(self) -> String
    fn to_json(self) -> String      # Redundant
    fn as_string(self) -> String    # Redundant
}
```

### 2. Use Derivation When Possible

**DO:**
```mojo
#[derive(Eq, Hash, Debug)]
struct Point { x: Int, y: Int }
```

**DON'T:**
```mojo
struct Point { x: Int, y: Int }

impl Eq for Point {
    fn eq(self, other: Self) -> Bool {
        return self.x == other.x and self.y == other.y
    }
}
# ... manual implementations of Hash and Debug
```

### 3. Clear Naming

**DO:**
```mojo
protocol Drawable {
    fn draw(self)
    fn bounds(self) -> Rect
}
```

**DON'T:**
```mojo
protocol D {
    fn d(self)
    fn b(self) -> Rect
}
```

### 4. Appropriate Abstraction

**DO:**
```mojo
protocol Database {
    fn connect(inout self) -> Result<(), Error>
    fn query(self, sql: String) -> Result<ResultSet, Error>
}
```

**DON'T:**
```mojo
protocol Thing {
    fn do_stuff(self)  # Too vague
}
```

---

## Common Patterns

### Pattern 1: Builder Protocol

```mojo
protocol Builder {
    type Output
    
    fn build(self) -> Output
}

struct ConfigBuilder {
    timeout: Int
    retries: Int
}

impl Builder for ConfigBuilder {
    type Output = Config
    
    fn build(self) -> Config {
        return Config {
            timeout: self.timeout,
            retries: self.retries,
        }
    }
}
```

### Pattern 2: Strategy Pattern

```mojo
protocol CompressionStrategy {
    fn compress(self, data: String) -> String
    fn decompress(self, data: String) -> String
}

struct GzipStrategy {}
struct ZipStrategy {}

impl CompressionStrategy for GzipStrategy {
    fn compress(self, data: String) -> String {
        # Gzip compression
    }
    fn decompress(self, data: String) -> String {
        # Gzip decompression
    }
}

fn compress_data(data: String, strategy: &dyn CompressionStrategy) -> String {
    return strategy.compress(data)
}
```

### Pattern 3: Observer Pattern

```mojo
protocol Observer {
    fn update(inout self, event: Event)
}

protocol Subject {
    fn attach(inout self, observer: Box<dyn Observer>)
    fn notify(self, event: Event)
}
```

### Pattern 4: Polymorphic Collections

```mojo
protocol Animal {
    fn make_sound(self)
}

impl Animal for Dog {
    fn make_sound(self) { print("Woof!") }
}

impl Animal for Cat {
    fn make_sound(self) { print("Meow!") }
}

fn main() {
    var animals: List[Box<dyn Animal>] = List()
    animals.append(Box::new(Dog{}))
    animals.append(Box::new(Cat{}))
    
    for animal in animals {
        animal.make_sound()
    }
}
```

---

## Summary

You've learned:
- ✅ How to define and implement protocols
- ✅ Automatic derivation with `#[derive]`
- ✅ Conditional conformance for generic types
- ✅ Protocol inheritance
- ✅ Associated types for generic protocols
- ✅ Best practices and common patterns

### Next Steps

- **Practice**: Implement protocols for your own types
- **Explore**: Study standard library protocol implementations
- **Read**: [Metaprogramming Guide](07-metaprogramming.md) for custom derive macros
- **Build**: Create reusable abstractions with protocols

---

## Quick Reference

```mojo
# Define protocol
protocol Drawable {
    fn draw(self)
}

# Implement protocol
impl Drawable for Circle {
    fn draw(self) { /* ... */ }
}

# Automatic derivation
#[derive(Eq, Hash, Debug, Clone, Default)]
struct Point { x: Int, y: Int }

# Conditional conformance
impl<T: Eq> Eq for Option<T> { /* ... */ }

# Protocol inheritance
protocol Shape: Drawable {
    fn area(self) -> Float
}

# Associated types
protocol Iterator {
    type Item
    fn next(inout self) -> Option<Item>
}
```

---

**Next Chapter:** [Async Programming](06-async-programming.md)  
**Previous Chapter:** [Memory Safety](04-memory-safety.md)

---

*Chapter 05: Protocol System*  
*Part of the Mojo SDK Developer Guide v1.0.0*  
*Last Updated: January 2026*
