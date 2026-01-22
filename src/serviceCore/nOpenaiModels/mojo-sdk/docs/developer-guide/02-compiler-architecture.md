# Chapter 2: Compiler Architecture

**Audience:** Intermediate to Advanced  
**Prerequisites:** [Chapter 1: Getting Started](01-getting-started.md)  
**Estimated Time:** 45-60 minutes  
**Version:** 1.0.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Compilation Pipeline](#compilation-pipeline)
4. [Frontend Components](#frontend-components)
5. [Type System](#type-system)
6. [Memory Safety](#memory-safety)
7. [Backend & Code Generation](#backend--code-generation)
8. [Performance](#performance)
9. [Extending the Compiler](#extending-the-compiler)

---

## Introduction

This chapter provides a comprehensive overview of the Mojo compiler architecture. You'll learn how source code transforms into optimized machine code through a sophisticated multi-stage pipeline.

**What you'll learn:**
- How the compiler processes Mojo code
- The role of each compilation stage
- How type checking and memory safety work
- Performance characteristics
- How to extend the compiler

**Why this matters:**
- Understanding compiler internals helps you write better code
- Debug compilation errors more effectively
- Contribute to compiler development
- Optimize your programs

---

## Overview

### High-Level Architecture

```
Source Code (.mojo)
        ↓
    [Lexer] ────→ Tokens
        ↓
    [Parser] ───→ AST
        ↓
  [Type Checker] → Typed AST
        ↓
[Borrow Checker] → Verified AST
        ↓
  [MLIR Backend] → MLIR IR
        ↓
  [LLVM Backend] → Machine Code
        ↓
   Executable
```

### Key Principles

1. **Safety First**: Memory safety and type safety are enforced at compile time
2. **Zero Cost**: Safety features have no runtime overhead
3. **Performance**: LLVM-based optimization for native performance
4. **Modularity**: Clean separation between compilation phases
5. **Error Quality**: Clear, actionable error messages

### Implementation Language

The compiler is written in **Zig**, chosen for:
- Fast compilation
- Low-level control
- Direct LLVM/MLIR integration
- Memory safety without GC
- Compile-time execution

---

## Compilation Pipeline

### Stage 1: Lexical Analysis

**What it does:** Converts source text into tokens

**Input:**
```mojo
fn add(a: Int, b: Int) -> Int:
    return a + b
```

**Output:**
```
Token(Fn, "fn")
Token(Identifier, "add")
Token(LeftParen, "(")
Token(Identifier, "a")
Token(Colon, ":")
Token(Identifier, "Int")
...
```

**Performance:** ~1μs per token

### Stage 2: Syntax Analysis

**What it does:** Builds Abstract Syntax Tree (AST)

**Input:** Token stream

**Output:**
```
FunctionDecl
├── name: "add"
├── params: [a: Int, b: Int]
├── return_type: Int
└── body: Block
    └── Return(BinaryExpr(+, a, b))
```

**Performance:** ~5μs per AST node

### Stage 3: Semantic Analysis

**What it does:** Type checking and name resolution

**Checks:**
- Variable declarations
- Type compatibility
- Function signatures
- Protocol conformance

**Example:**
```mojo
let x: String = 42  # Error: Type mismatch
```

Error:
```
error[E0308]: mismatched types
  expected: String
  found: Int
```

**Performance:** ~10-20μs per node

### Stage 4: Borrow Checking

**What it does:** Enforces memory safety rules

**Checks:**
- Ownership tracking
- Borrow conflicts
- Lifetime validation
- Move semantics

**Example:**
```mojo
let r1 = &x
let r2 = &mut x  # Error: Cannot borrow mutably
```

**Performance:** ~3-5ms per 1000 LOC

### Stage 5: IR Generation

**What it does:** Generates intermediate representation

**Transformations:**
- Pattern matching lowering
- Async desugaring
- Generic instantiation
- Optimization passes

**Performance:** ~20-40ms per 1000 LOC

### Stage 6: Code Generation

**What it does:** Generates native machine code

**Optimizations:**
- Dead code elimination
- Inlining
- Register allocation
- Loop optimization

**Performance:** ~50-100ms per 1000 LOC

---

## Frontend Components

### Lexer

The lexer is the first stage of compilation.

**File:** `compiler/frontend/lexer.zig` (~800 LOC)

**Key Features:**
- Character-by-character scanning
- Token classification
- Position tracking
- Error recovery

**Token Types:**

| Category | Examples |
|----------|----------|
| Keywords | `fn`, `let`, `var`, `if`, `while` |
| Operators | `+`, `-`, `*`, `/`, `==`, `!=` |
| Delimiters | `(`, `)`, `{`, `}`, `,`, `;` |
| Literals | `42`, `3.14`, `"hello"` |
| Identifiers | Variable/function names |

**Example Usage:**

```zig
var lexer = Lexer.init(source, allocator);
while (true) {
    const token = try lexer.nextToken();
    if (token.kind == .Eof) break;
    print("{}\n", .{token});
}
```

### Parser

The parser builds the AST from tokens.

**File:** `compiler/frontend/parser.zig` (~2,500 LOC)

**Parsing Technique:**
- Recursive descent
- Operator precedence climbing
- LL(1) with lookahead

**Operator Precedence:**

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `or` | Left |
| 2 | `and` | Left |
| 3 | `==`, `!=` | Left |
| 4 | `<`, `>`, `<=`, `>=` | Left |
| 5 | `+`, `-` | Left |
| 6 | `*`, `/`, `%` | Left |
| 7 | `-`, `!` (unary) | Right |
| 8 | `.`, `[]`, `()` (postfix) | Left |

**Example:**

```mojo
1 + 2 * 3  # Parsed as: 1 + (2 * 3)
```

AST:
```
BinaryExpr(+)
├── Literal(1)
└── BinaryExpr(*)
    ├── Literal(2)
    └── Literal(3)
```

### AST

The Abstract Syntax Tree represents program structure.

**File:** `compiler/frontend/ast.zig` (~450 LOC)

**Node Types:**

```zig
pub const AstNode = union(enum) {
    // Declarations
    FunctionDecl,
    StructDecl,
    ProtocolDecl,
    
    // Statements
    VarDecl,
    LetDecl,
    IfStmt,
    WhileStmt,
    ReturnStmt,
    
    // Expressions
    BinaryExpr,
    UnaryExpr,
    CallExpr,
    LiteralExpr,
};
```

**Memory Management:**
- RAII-style cleanup
- Recursive deallocation
- No memory leaks

---

## Type System

### Type Hierarchy

```
Type
├── Primitive (Int, Float, Bool, String)
├── Array ([T], [n]T)
├── Pointer (*T, *mut T)
├── Function ((T1, T2) -> T3)
├── Struct (user-defined)
├── Protocol (trait)
├── Union (sum type)
├── Option (Option[T])
└── Generic (T where constraints)
```

### Type Checking

**File:** `compiler/frontend/types.zig` (~400 LOC)

**Process:**

1. **Type Inference**
```mojo
let x = 42        # Infer: Int
let y = 3.14      # Infer: Float
let s = "hello"   # Infer: String
```

2. **Type Compatibility**
```mojo
let x: Int32 = 100
let y: Int64 = x    # ✅ Widening OK
let z: Int32 = y    # ❌ Narrowing requires cast
```

3. **Generic Resolution**
```mojo
fn identity[T](x: T) -> T:  # T inferred from usage
    return x

let result = identity(42)  # T = Int
```

### Type Constraints

```mojo
# Numeric constraint
fn add[T: Numeric](a: T, b: T) -> T:
    return a + b

# Multiple constraints
fn process[T: Copyable + Movable](data: T) -> T:
    return data.clone()
```

**Available Constraints:**
- `Numeric` - Supports arithmetic
- `Comparable` - Supports comparison
- `Equatable` - Supports equality
- `Copyable` - Can be copied
- `Movable` - Can be moved

---

## Memory Safety

### Ownership Model

Every value has exactly one owner:

```mojo
fn main():
    let s = String("hello")  # s owns the string
    process(s^)              # Ownership moved
    # print(s)               # ❌ Error: s was moved
}
```

### Borrowing Rules

**Rule 1:** One mutable OR many immutable borrows

```mojo
var x = 42
let r1 = &x       # ✅ Immutable
let r2 = &x       # ✅ Multiple immutable OK
let r3 = &mut x   # ❌ Cannot borrow mutably
```

**Rule 2:** References must be valid

```mojo
fn dangling() -> &String:
    let s = String("local")
    return &s  # ❌ Reference to local
}
```

**Rule 3:** No simultaneous mutable borrows

```mojo
var x = 42
let r1 = &mut x  # ✅ First mutable
let r2 = &mut x  # ❌ Second mutable conflicts
```

### Borrow Checker

**File:** `compiler/frontend/borrow_checker.zig` (~500 LOC)

**Algorithm:**

```
For each borrow:
1. Check if value was moved → Error
2. Check against active borrows:
   - If mutable borrow + any active → Error
   - If shared borrow + mutable active → Error
   - Multiple shared borrows → OK
3. Track borrow in scope
4. Release borrow at end of scope
```

**Example Analysis:**

```mojo
var data = vec![1, 2, 3]

{
    let r1 = &data      # Shared borrow starts
    print(r1[0])
}  # r1 ends here

let r2 = &mut data      # ✅ OK, r1 ended
r2.push(4)
```

**Borrow Tracking:**
```
Line 3: Add shared borrow of 'data'
Line 5: End shared borrow of 'data'
Line 7: Add mutable borrow of 'data' ✅
Line 8: Use mutable borrow
```

---

## Backend & Code Generation

### MLIR Integration

**What is MLIR?**
Multi-Level Intermediate Representation - a compiler infrastructure for building optimizing compilers.

**Why MLIR?**
- High-level abstractions
- Dialect system
- Progressive lowering
- LLVM integration

**Mojo's MLIR Dialects:**

```
Mojo IR (high-level)
    ↓
Standard Dialect (mid-level)
    ↓
LLVM Dialect (low-level)
    ↓
LLVM IR
```

### LLVM Backend

**Optimizations:**

| Pass | Description | Impact |
|------|-------------|--------|
| Inlining | Inline small functions | High |
| DCE | Dead code elimination | Medium |
| CSE | Common subexpression | Medium |
| Loop Opt | Loop unrolling, fusion | High |
| Vectorization | SIMD instructions | Very High |

**Example:**

```mojo
fn sum(arr: []Int) -> Int:
    var total = 0
    for x in arr:
        total += x
    return total
```

Optimized to vectorized SIMD instructions.

### Code Generation

**Target Architectures:**
- x86_64
- ARM64
- WebAssembly (planned)

**Output Formats:**
- Native executable
- Static library (.a)
- Dynamic library (.so/.dylib)
- Object file (.o)

---

## Performance

### Compilation Speed

| File Size | Lex | Parse | Check | Total |
|-----------|-----|-------|-------|-------|
| 100 LOC | 1ms | 2ms | 5ms | ~10ms |
| 1K LOC | 2ms | 10ms | 30ms | ~50ms |
| 10K LOC | 10ms | 50ms | 100ms | ~200ms |

### Optimization Levels

```bash
# No optimization (fastest compile)
mojo build -O0 main.mojo

# Basic optimization
mojo build -O1 main.mojo

# Full optimization (default)
mojo build -O2 main.mojo

# Aggressive optimization
mojo build -O3 main.mojo
```

**Performance Impact:**

| Level | Compile Time | Runtime Speed | Binary Size |
|-------|--------------|---------------|-------------|
| O0 | 1x | 1x | 1x |
| O1 | 1.2x | 2x | 0.9x |
| O2 | 1.5x | 3x | 0.8x |
| O3 | 2x | 3.5x | 0.7x |

### Memory Usage

**Compiler Memory:**
- Small project: ~50MB
- Medium project: ~200MB
- Large project: ~500MB

**Techniques:**
- Incremental compilation
- Parallel processing
- Memory pooling
- Arena allocation

---

## Extending the Compiler

### Adding a New Feature

**Example: Add a new operator**

1. **Update Lexer** (`lexer.zig`):
```zig
pub const TokenKind = enum {
    // ... existing tokens
    Power,  // ** operator
};
```

2. **Update Parser** (`parser.zig`):
```zig
fn factor(self: *Parser) !*Expr {
    var expr = try self.unary();
    
    while (self.match(.Power)) {
        const op = self.previous;
        const right = try self.unary();
        expr = try Expr.binary(expr, op, right);
    }
    
    return expr;
}
```

3. **Update Type Checker** (`types.zig`):
```zig
fn checkBinaryOp(op: TokenKind, left: Type, right: Type) !Type {
    switch (op) {
        .Power => {
            if (!left.isNumeric() or !right.isNumeric()) {
                return error.InvalidOperands;
            }
            return left;  // or promote to Float
        },
        // ... other operators
    }
}
```

4. **Update Code Generator**:
```zig
fn codegen(expr: *BinaryExpr) !Value {
    switch (expr.op) {
        .Power => {
            // Generate LLVM pow intrinsic
            return builder.buildCall(pow_intrinsic, &[_]Value{
                try codegen(expr.left),
                try codegen(expr.right),
            });
        },
        // ... other operators
    }
}
```

### Testing

```zig
test "power operator" {
    const source = "2 ** 3";
    var lexer = Lexer.init(source, allocator);
    var parser = try Parser.init(&lexer, allocator);
    
    const expr = try parser.expression();
    
    // Should parse as: Binary(Power, 2, 3)
    try testing.expect(expr.* == .Binary);
    try testing.expect(expr.Binary.op == .Power);
}
```

### Contributing Guidelines

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/my-feature`
3. **Write tests**: Cover new functionality
4. **Run test suite**: `zig build test`
5. **Submit pull request**: With clear description

---

## Summary

You've learned:
- ✅ Complete compilation pipeline
- ✅ Role of each compiler stage
- ✅ Type system architecture
- ✅ Memory safety enforcement
- ✅ Backend optimization
- ✅ Performance characteristics
- ✅ How to extend the compiler

### Key Takeaways

1. **Compilation is multi-stage** - Each stage has a specific responsibility
2. **Safety is compile-time** - Zero runtime overhead
3. **Performance is prioritized** - LLVM-based optimization
4. **Extensibility is built-in** - Clean modular architecture

### Next Steps

1. **Dive deeper**: Read the [Technical Manual](../manual/MOJO_SDK_TECHNICAL_MANUAL.md)
2. **Explore source**: Study `compiler/frontend/`
3. **Write code**: Practice with examples
4. **Contribute**: Fix bugs or add features

### Additional Resources

- [Memory Safety](04-memory-safety.md) - Deep dive into ownership
- [Protocol System](05-protocol-system.md) - Protocol implementation
- [Contributing](13-contributing.md) - How to contribute
- [API Reference](14-api-reference.md) - Complete API docs

---

## Exercises

### Exercise 1: Trace Compilation

Trace how this code compiles:

```mojo
fn factorial(n: Int) -> Int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**Questions:**
1. What tokens are generated?
2. What does the AST look like?
3. What type checks are performed?
4. Are there any borrow checker considerations?

### Exercise 2: Error Analysis

Find the errors in this code:

```mojo
fn example():
    let x = 42
    let r1 = &x
    let r2 = &mut x
    print(r1)
```

**Tasks:**
1. Identify the error
2. Explain which compilation stage catches it
3. Suggest a fix

### Exercise 3: Optimization

Compare these implementations:

```mojo
# Version A
fn sum_array(arr: []Int) -> Int:
    var total = 0
    for i in range(arr.len()):
        total += arr[i]
    return total

# Version B
fn sum_array(arr: []Int) -> Int:
    var total = 0
    for x in arr:
        total += x
    return total
```

**Questions:**
1. Which is faster?
2. What optimizations apply?
3. Do they generate the same code?

---

**Previous Chapter:** [Getting Started](01-getting-started.md)  
**Next Chapter:** [Standard Library Guide](03-stdlib-guide.md)

---

*Chapter 2: Compiler Architecture*  
*Part of the Mojo SDK Developer Guide v1.0.0*  
*Last Updated: January 2026*
