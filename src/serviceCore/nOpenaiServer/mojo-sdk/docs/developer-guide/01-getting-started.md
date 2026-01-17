# Chapter 01: Getting Started with Mojo SDK

**Version:** 1.0.0  
**Audience:** Beginners  
**Prerequisites:** Basic programming knowledge  
**Estimated Time:** 30 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Your First Program](#your-first-program)
4. [Understanding the Basics](#understanding-the-basics)
5. [IDE Setup](#ide-setup)
6. [Project Structure](#project-structure)
7. [Common Commands](#common-commands)
8. [Next Steps](#next-steps)

---

## Introduction

Welcome to Mojo! This guide will help you install Mojo, write your first program, and understand the basics of the language.

### What is Mojo?

Mojo is a **memory-safe, high-performance programming language** that combines:
- **Memory safety** like Rust (ownership and borrowing)
- **Performance** like C++ (zero-cost abstractions)
- **Syntax** like Python (developer-friendly)
- **Modern features** like Swift (protocols, async/await)

### Why Mojo?

- ✅ **Safe**: Compile-time memory safety prevents crashes
- ⚡ **Fast**: LLVM backend generates optimized machine code
- 🎨 **Expressive**: Clean syntax with powerful features
- 🛠️ **Great Tools**: LSP, package manager, debugger included
- 🔄 **Modern**: Async/await, protocols, metaprogramming

---

## Installation

### Option 1: Binary Release (Recommended)

#### Linux / macOS

```bash
# Download latest release
curl -LO https://github.com/mojo-lang/mojo-sdk/releases/download/v1.0.0/mojo-sdk-1.0.0.tar.gz

# Extract
tar xzf mojo-sdk-1.0.0.tar.gz

# Install
cd mojo-sdk-1.0.0
sudo ./install.sh

# Verify installation
mojo --version
```

Expected output:
```
Mojo SDK v1.0.0
```

#### Windows

```powershell
# Download from releases page
# https://github.com/mojo-lang/mojo-sdk/releases

# Extract and run installer
.\mojo-sdk-1.0.0-installer.exe

# Verify installation
mojo --version
```

### Option 2: Build from Source

#### Prerequisites

- **Zig** 0.13.0 or later
- **LLVM** 17+ development libraries
- **Git**

#### Build Steps

```bash
# Clone repository
git clone https://github.com/mojo-lang/mojo-sdk.git
cd mojo-sdk

# Build compiler
zig build-exe compiler/main.zig -O ReleaseFast

# Build standard library
zig build stdlib

# Run tests (optional)
./test_runner

# Install
sudo zig build install --prefix /usr/local
```

### Verify Installation

```bash
# Check version
mojo --version

# Check compiler
mojo --help

# Check tools
mojo-lsp --version
mojo-pkg --version
```

---

## Your First Program

### Hello, World!

Create a file called `hello.mojo`:

```mojo
fn main():
    print("Hello, Mojo! 🔥")
```

Run it:

```bash
mojo hello.mojo
```

Output:
```
Hello, Mojo! 🔥
```

**Congratulations!** You've just run your first Mojo program!

### Understanding the Code

```mojo
fn main():              # Function definition
    print("Hello...")   # Function call
```

- `fn` declares a function
- `main()` is the entry point
- Indentation (4 spaces) defines blocks
- `print()` outputs to console

### Variables and Types

Create `variables.mojo`:

```mojo
fn main():
    # Immutable variable (default)
    let name = "Alice"
    let age = 30
    
    # Mutable variable
    var count = 0
    count = count + 1
    
    # Type annotations
    let score: Float = 98.5
    let is_active: Bool = true
    
    print(f"Name: {name}")
    print(f"Age: {age}")
    print(f"Count: {count}")
    print(f"Score: {score}")
    print(f"Active: {is_active}")
```

Run it:
```bash
mojo variables.mojo
```

Output:
```
Name: Alice
Age: 30
Count: 1
Score: 98.5
Active: true
```

### Functions

Create `functions.mojo`:

```mojo
fn add(a: Int, b: Int) -> Int:
    return a + b

fn greet(name: String) -> String:
    return f"Hello, {name}!"

fn main():
    let result = add(5, 3)
    print(f"5 + 3 = {result}")
    
    let message = greet("Bob")
    print(message)
```

Output:
```
5 + 3 = 8
Hello, Bob!
```

### Control Flow

Create `control_flow.mojo`:

```mojo
fn main():
    # If-else
    let x = 10
    
    if x > 0:
        print("Positive")
    elif x < 0:
        print("Negative")
    else:
        print("Zero")
    
    # While loop
    var i = 0
    while i < 5:
        print(f"i = {i}")
        i += 1
    
    # For loop
    for j in range(3):
        print(f"j = {j}")
```

Output:
```
Positive
i = 0
i = 1
i = 2
i = 3
i = 4
j = 0
j = 1
j = 2
```

---

## Understanding the Basics

### Basic Types

```mojo
# Integers
let a: Int = 42
let b: Int8 = 127
let c: UInt64 = 1000

# Floating point
let pi: Float = 3.14159
let e: Float64 = 2.71828

# Boolean
let is_true: Bool = true
let is_false: Bool = false

# String
let text: String = "Hello"

# Character
let ch: Char = 'A'
```

### Collections

```mojo
from collections import List, Dict, Set

fn main():
    # List
    var numbers = List[Int]()
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    print(f"List: {numbers}")
    
    # Dictionary
    var scores = Dict[String, Int]()
    scores["Alice"] = 95
    scores["Bob"] = 87
    print(f"Dict: {scores}")
    
    # Set
    var unique = Set[Int]()
    unique.add(1)
    unique.add(2)
    unique.add(1)  # Duplicate ignored
    print(f"Set: {unique}")
```

### Structs

```mojo
struct Point:
    x: Int
    y: Int
    
    fn __init__(inout self, x: Int, y: Int):
        self.x = x
        self.y = y
    
    fn distance_from_origin(self) -> Float:
        return sqrt(self.x * self.x + self.y * self.y)

fn main():
    let p = Point(3, 4)
    print(f"Point: ({p.x}, {p.y})")
    print(f"Distance: {p.distance_from_origin()}")
```

---

## IDE Setup

### Visual Studio Code (Recommended)

#### 1. Install VS Code

Download from: https://code.visualstudio.com/

#### 2. Install Mojo Extension

```bash
code --install-extension mojo-lang.mojo-vscode
```

Or search for "Mojo" in VS Code Extensions.

#### 3. Configure Settings

Create `.vscode/settings.json` in your project:

```json
{
    "mojo.lsp.path": "/usr/local/bin/mojo-lsp",
    "mojo.compiler.path": "/usr/local/bin/mojo",
    "editor.formatOnSave": true,
    "editor.tabSize": 4,
    "editor.insertSpaces": true
}
```

#### 4. Features Available

- ✅ **Syntax Highlighting**: Colors and formatting
- ✅ **Code Completion**: IntelliSense suggestions
- ✅ **Go to Definition**: Jump to declarations
- ✅ **Find References**: See all usages
- ✅ **Hover Documentation**: Inline docs
- ✅ **Error Diagnostics**: Real-time errors
- ✅ **Refactoring**: Rename, extract, etc.
- ✅ **Debugging**: Set breakpoints, inspect

#### 5. Keyboard Shortcuts

| Action | Windows/Linux | macOS |
|--------|--------------|-------|
| Go to Definition | F12 | F12 |
| Find References | Shift+F12 | Shift+F12 |
| Rename | F2 | F2 |
| Format Document | Shift+Alt+F | Shift+Option+F |
| Quick Fix | Ctrl+. | Cmd+. |

### Other IDEs

#### Vim/Neovim

Install vim-mojo plugin:
```vim
Plug 'mojo-lang/vim-mojo'
```

Configure LSP in your `init.vim`:
```vim
lua << EOF
require'lspconfig'.mojo_lsp.setup{}
EOF
```

#### Emacs

Install mojo-mode:
```elisp
(use-package mojo-mode
  :mode "\\.mojo\\'")
```

---

## Project Structure

### Creating a New Project

```bash
mojo-pkg new my-project
cd my-project
```

Generated structure:

```
my-project/
├── mojo.toml           # Project configuration
├── src/
│   └── main.mojo      # Entry point
├── tests/
│   └── test_main.mojo # Tests
└── README.md          # Documentation
```

### Project Configuration (`mojo.toml`)

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2026"
authors = ["Your Name <you@example.com>"]
description = "A sample Mojo project"

[dependencies]
# Add dependencies here
# http = "1.0.0"
# json = "0.5.0"

[dev-dependencies]
testing = "1.0.0"

[build]
optimization = "release"
```

### Adding Dependencies

```bash
# Add a dependency
mojo-pkg add http@1.0.0

# Remove a dependency
mojo-pkg remove http

# Update dependencies
mojo-pkg update
```

### Building Your Project

```bash
# Build in debug mode
mojo-pkg build

# Build in release mode
mojo-pkg build --release

# Run the project
mojo-pkg run

# Run tests
mojo-pkg test
```

---

## Common Commands

### Compiler Commands

```bash
# Run a file directly
mojo run main.mojo

# Compile to binary
mojo build main.mojo -o myapp

# Compile with optimizations
mojo build main.mojo -o myapp -O3

# Check syntax without building
mojo check main.mojo

# Format code
mojo fmt src/

# Show documentation
mojo doc MyStruct
```

### Package Manager Commands

```bash
# Create new project
mojo-pkg new project-name

# Initialize in existing directory
mojo-pkg init

# Add dependency
mojo-pkg add package-name

# Build project
mojo-pkg build

# Run project
mojo-pkg run

# Run tests
mojo-pkg test

# Publish to registry
mojo-pkg publish
```

### Debugging Commands

```bash
# Launch debugger
mojo debug main.mojo

# Run with debugger attached
mojo run --debug main.mojo
```

### Help Commands

```bash
# General help
mojo --help

# Command-specific help
mojo build --help
mojo-pkg --help
```

---

## Next Steps

### Learning Path

Now that you have Mojo installed and running, here's what to learn next:

#### 1. **Language Basics** (Essential)
- [ ] Read [Quick Reference](../manual/MOJO_SDK_TECHNICAL_MANUAL.md#4-quick-reference)
- [ ] Learn about ownership and borrowing
- [ ] Practice with control flow
- [ ] Understand structs and methods

#### 2. **Core Concepts** (Important)
- [ ] Memory safety and ownership
- [ ] Protocol-oriented programming
- [ ] Error handling with Result and Option
- [ ] Working with collections

#### 3. **Advanced Features** (When Ready)
- [ ] Async/await programming
- [ ] Metaprogramming with macros
- [ ] Generic programming
- [ ] Performance optimization

### Recommended Tutorials

1. **[Building a Calculator](15-tutorials.md#calculator)** (30 min)
   - Learn functions and operators
   - Practice control flow
   - Build a simple CLI tool

2. **[Todo List Application](15-tutorials.md#todo-list)** (1 hour)
   - Work with collections
   - File I/O operations
   - User input handling

3. **[HTTP Client](15-tutorials.md#http-client)** (1 hour)
   - Network programming
   - Async/await basics
   - Error handling

### Resources

#### Documentation
- **[Technical Manual](../manual/MOJO_SDK_TECHNICAL_MANUAL.md)** - Complete reference
- **[Developer Guides](00-index.md)** - Topic-specific guides
- **[API Reference](14-api-reference.md)** - Standard library docs

#### Community
- **Website**: https://mojo-lang.org
- **Forum**: https://forum.mojo-lang.org
- **Discord**: https://discord.gg/mojo-lang
- **GitHub**: https://github.com/mojo-lang/mojo-sdk

#### Examples
- Browse `examples/` in the SDK directory
- Check out community projects on GitHub
- Study the standard library source code

---

## Common Issues & Solutions

### Issue: Command not found

**Problem**: `mojo: command not found`

**Solution**:
```bash
# Check if PATH is set correctly
echo $PATH

# Add to PATH (Linux/macOS)
export PATH="/usr/local/bin:$PATH"

# Add to ~/.bashrc or ~/.zshrc to make permanent
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
```

### Issue: Compilation errors

**Problem**: Unexpected compilation errors

**Solution**:
1. Check syntax with `mojo check`
2. Read error messages carefully
3. Consult the [Error Codes Reference](../manual/MOJO_SDK_TECHNICAL_MANUAL.md#appendix-a-error-codes-reference)
4. Ask on Discord or Forum

### Issue: LSP not working in VS Code

**Problem**: No autocomplete or diagnostics

**Solution**:
1. Check LSP is installed: `mojo-lsp --version`
2. Verify settings.json path is correct
3. Restart VS Code
4. Check Output panel (View → Output → Mojo Language Server)

### Issue: Slow compilation

**Problem**: Compilation takes too long

**Solution**:
1. Use incremental compilation: `mojo-pkg build`
2. Reduce dependencies
3. Split large files into modules
4. Use release build only for final builds

---

## Quick Reference Card

### Syntax Cheat Sheet

```mojo
# Variables
let x = 42          # Immutable
var y = 10          # Mutable

# Functions
fn add(a: Int, b: Int) -> Int:
    return a + b

# Structs
struct Point:
    x: Int
    y: Int

# Control Flow
if condition:
    # code
elif other:
    # code
else:
    # code

while condition:
    # code

for item in collection:
    # code

# Collections
List[Int]()
Dict[String, Int]()
Set[Int]()

# Error Handling
Result[T, E]
Option[T]
```

---

## Summary

You've learned:
- ✅ How to install Mojo
- ✅ How to write and run programs
- ✅ Basic syntax and types
- ✅ How to set up your IDE
- ✅ Project structure and commands
- ✅ Where to go next

### Next Chapter

Ready for more? → Continue with [Chapter 02: Compiler Architecture](02-compiler-architecture.md)

Or explore:
- [Memory Safety](04-memory-safety.md) - Understand ownership
- [Standard Library Guide](03-stdlib-guide.md) - Learn the APIs
- [Tutorials](15-tutorials.md) - Build real projects

---

## Glossary

- **Mojo**: The programming language
- **SDK**: Software Development Kit (compiler + tools)
- **LSP**: Language Server Protocol (IDE integration)
- **mojo-pkg**: Package manager
- **Ownership**: Memory management concept
- **Borrowing**: Temporary access to data
- **Protocol**: Interface definition (like traits)

---

**Questions?** Ask on [Discord](https://discord.gg/mojo-lang) or the [Forum](https://forum.mojo-lang.org)

---

*Chapter 01: Getting Started*  
*Part of the Mojo SDK Developer Guide v1.0.0*  
*Last Updated: January 2026*
