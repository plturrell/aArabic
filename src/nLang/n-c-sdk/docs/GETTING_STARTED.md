# Getting Started with n-c-sdk

Welcome to the n-c-sdk! This high-performance SDK built with Zig will help you get up and running quickly.

## 📋 Prerequisites

- **Zig 0.15.2** or later
- Basic familiarity with Zig programming language
- A terminal/command line interface

## 🚀 Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/n-c-sdk.git
cd n-c-sdk
```

### Verify Installation

Run the test suite to ensure everything is working:

```bash
# Run all tests
cd tests
zig build test

# Run integration tests
zig build integration
```

Expected output:
```
Build Summary: 3/3 steps succeeded; 15/15 tests passed
test success
```

### Run Benchmarks

Execute the benchmark suite:

```bash
cd benchmarks
./run_benchmarks.sh
```

## 📖 Your First Program

Create a simple program using the SDK patterns:

```zig
const std = @import("std");

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a list
    var numbers = std.ArrayList(i32){};
    defer numbers.deinit(allocator);

    // Add some numbers
    try numbers.append(allocator, 10);
    try numbers.append(allocator, 20);
    try numbers.append(allocator, 30);

    // Process and print
    var sum: i32 = 0;
    for (numbers.items) |num| {
        sum += num;
    }

    std.debug.print("Sum: {}\n", .{sum});
}
```

Save as `hello.zig` and run:

```bash
zig run hello.zig
```

## 🎯 Key Concepts

### Memory Management

The SDK emphasizes safe memory management:

```zig
// Always use GeneralPurposeAllocator for leak detection
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit(); // Reports leaks on exit

const allocator = gpa.allocator();

// Allocate memory
const buffer = try allocator.alloc(u8, 1024);
defer allocator.free(buffer); // Always free!
```

### Data Structures

#### ArrayList

```zig
var list = std.ArrayList(i32){};
defer list.deinit(allocator);

try list.append(allocator, 42);
const value = list.items[0]; // Access items
```

#### HashMap

```zig
var map = std.StringHashMap(i32).init(allocator);
defer map.deinit();

try map.put("answer", 42);
const value = map.get("answer"); // Returns ?i32
```

### Error Handling

```zig
// Functions that can fail return errors
fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

// Use try to propagate errors
const result = try divide(10, 2);

// Or handle explicitly
const result = divide(10, 0) catch |err| {
    std.debug.print("Error: {}\n", .{err});
    return;
};
```

## 📚 Next Steps

1. **[API Reference](api/README.md)** - Detailed API documentation
2. **[User Guide](guides/USER_GUIDE.md)** - Comprehensive usage guide
3. **[Examples](examples/README.md)** - Code examples
4. **[Benchmarking Guide](../benchmarks/README.md)** - Performance measurement
5. **[Testing Guide](../tests/README.md)** - Writing tests

## 🔧 Common Tasks

### Building Your Project

```bash
# Debug build
zig build

# Release build with safety checks
zig build -Doptimize=ReleaseSafe

# Release build with maximum performance
zig build -Doptimize=ReleaseFast
```

### Running Tests

```bash
# Run specific test file
zig test src/my_module.zig

# Run with specific optimization
zig test -OReleaseSafe src/my_module.zig

# Run specific test by name
zig test --test-filter "ArrayList" src/my_module.zig
```

### Benchmarking Your Code

```zig
const bench = @import("benchmark_framework");

test "my benchmark" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const result = try bench.benchmark(
        allocator,
        "My Operation",
        100, // iterations
        10,  // warmup
        struct {
            fn run() void {
                // Your code here
            }
        }{}
    );

    bench.printResult(result);
}
```

## ❓ Troubleshooting

### Compilation Errors

**Issue**: `error: struct has no member named 'init'`

**Solution**: Ensure you're using the correct Zig 0.15.2 API:
```zig
// Correct
var list = std.ArrayList(i32){};

// Incorrect (old API)
var list = std.ArrayList(i32).init(allocator);
```

### Memory Leaks

**Issue**: Program reports memory leaks on exit

**Solution**: Ensure all allocations are freed:
```zig
const data = try allocator.alloc(u8, 100);
defer allocator.free(data); // Add this!
```

### Build Failures

**Issue**: `zig build` fails

**Solution**: Check Zig version:
```bash
zig version
# Should output: 0.15.2 or later
```

## 🤝 Getting Help

- **Documentation**: Check the [docs](.) directory
- **Examples**: See [examples](examples/) for working code
- **Issues**: Report bugs via GitHub Issues
- **Community**: Join the Zig community discussions

## 📝 Best Practices

1. **Always use `defer`** for cleanup
2. **Check for memory leaks** with GeneralPurposeAllocator
3. **Handle errors explicitly** with try/catch
4. **Write tests** for your code
5. **Use `const` when possible** for immutability
6. **Document your code** with doc comments

## 🎓 Learning Resources

- [Zig Language Reference](https://ziglang.org/documentation/master/)
- [Zig Learn](https://ziglearn.org/)
- [Zig Standard Library](https://ziglang.org/documentation/master/std/)

## ⚡ Performance Tips

1. Use `ReleaseFast` for maximum performance
2. Profile before optimizing
3. Prefer stack allocation when possible
4. Use appropriate data structures
5. Leverage comptime for zero-cost abstractions

---

**Ready to dive deeper?** Continue to the [User Guide](guides/USER_GUIDE.md) for comprehensive documentation.
