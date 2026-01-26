# n-c-sdk Test Suite

Comprehensive testing infrastructure for the n-c-sdk covering unit tests, integration tests, and load testing.

## Test Organization

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── hana_test.zig     # HANA client unit tests
│   └── ...
├── integration/           # Integration tests for full workflows
│   ├── http_integration_test.zig
│   └── hana_integration_test.zig
├── load/                  # Load and performance tests
│   └── load_test.zig
├── build.zig              # Test build configuration
└── run_tests.sh           # Test runner script
```

## Running Tests

### All Tests

```bash
# Run all tests
cd tests
zig build test

# Or use the convenience script
./run_tests.sh
```

### Specific Test Suites

```bash
# Unit tests only
zig build test-unit

# Integration tests only
zig build test-integration

# Load tests only
zig build test-load
```

### Individual Test Files

```bash
# Run specific test file
zig test unit/hana_test.zig

# Run with specific optimization
zig test -OReleaseSafe unit/hana_test.zig

# Run specific test by name
zig test --test-filter "QueryBuilder" unit/hana_test.zig
```

## Test Categories

### 1. Unit Tests

Test individual functions and components in isolation.

**Coverage:**
- HANA client components (connection, queries, results)
- Query builder (all SQL operations)
- Batch operations (inserter, updater)
- HTTP framework components (router, middleware, context)
- Value type conversions
- Parameter handling

**Example:**
```bash
zig test tests/unit/hana_test.zig
```

### 2. Integration Tests

Test complete workflows with multiple components working together.

**Coverage:**
- HTTP request/response cycles
- Database connection pooling
- Query execution pipelines
- Middleware chain execution
- Error handling across layers

**Example:**
```bash
zig test tests/integration/http_integration_test.zig
zig test tests/integration/hana_integration_test.zig
```

### 3. Load Tests

Measure performance under various load conditions.

**Coverage:**
- HTTP endpoint throughput
- Database query performance
- Concurrent connection handling
- Batch operation efficiency
- Latency distribution (P50, P95, P99)

**Example:**
```bash
zig test tests/load/load_test.zig
```

## Test Configuration

### Load Test Settings

Customize load tests via `LoadTestConfig`:

```zig
const config = LoadTestConfig{
    .num_requests = 10000,        // Total requests
    .concurrency = 100,           // Concurrent workers
    .warmup_requests = 1000,      // Warmup phase
    .report_interval_ms = 1000,   // Progress reporting
};
```

### Database Test Settings

Configure test database connection:

```bash
export HANA_HOST=localhost
export HANA_PORT=30015
export HANA_DATABASE=TESTDB
export HANA_USER=TESTUSER
export HANA_PASSWORD=test123
```

## Writing New Tests

### Unit Test Template

```zig
const std = @import("std");
const testing = std.testing;

test "Component - specific behavior" {
    const allocator = testing.allocator;
    
    // Setup
    var component = try Component.init(allocator);
    defer component.deinit();
    
    // Execute
    const result = try component.operation();
    
    // Assert
    try testing.expect(result == expected);
}
```

### Integration Test Template

```zig
test "Integration - complete workflow" {
    const allocator = testing.allocator;
    
    // Setup multiple components
    var client = try Client.init(allocator, config);
    defer client.deinit();
    
    // Execute workflow
    const step1 = try client.operation1();
    const step2 = try client.operation2(step1);
    
    // Validate end-to-end behavior
    try testing.expect(step2.isValid());
}
```

### Load Test Template

```zig
test "Load - operation performance" {
    const allocator = testing.allocator;
    
    var tester = LoadTester.init(allocator, .{
        .num_requests = 1000,
        .concurrency = 10,
    });
    defer tester.deinit();
    
    const result = try tester.run(&context, operation);
    result.print();
    
    try testing.expect(result.requests_per_second > threshold);
}
```

## Benchmarking

For detailed performance benchmarks, see:
- `../benchmarks/README.md` - Benchmark suite
- `../benchmarks/run_benchmarks.sh` - Automated benchmarking

## Test Coverage Goals

| Component | Current | Target |
|-----------|---------|--------|
| HANA Client | 85% | 90% |
| Query Builder | 90% | 95% |
| Batch Operations | 80% | 90% |
| HTTP Framework | 75% | 85% |
| WebSocket | 70% | 85% |
| HTTP/2 | 70% | 85% |
| Static Files | 65% | 80% |

## CI/CD Integration

Tests are automatically run in CI:

```yaml
# .github/workflows/test.yml
- name: Run Tests
  run: |
    cd tests
    zig build test
```

## Performance Regression Testing

Load tests establish performance baselines:

1. Run load tests on stable branch
2. Record metrics (throughput, latency)
3. Compare against new changes
4. Fail if regression > 10%

## Debugging Test Failures

### Enable Verbose Output

```bash
zig test --verbose unit/hana_test.zig
```

### Run with Debug Build

```bash
zig test -ODebug unit/hana_test.zig
```

### Use GDB/LLDB

```bash
zig test unit/hana_test.zig --test-cmd gdb --test-cmd-bin
```

## Memory Leak Detection

All tests use `GeneralPurposeAllocator` with leak detection:

```zig
test "Component - no leaks" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        try testing.expect(leaked == .ok); // Fails if memory leaked
    }
    
    const allocator = gpa.allocator();
    // ... test code ...
}
```

## Best Practices

1. **Always use defer** for cleanup
2. **Test error cases** not just happy paths
3. **Use meaningful test names** describing behavior
4. **Keep tests focused** - one behavior per test
5. **Mock external dependencies** when possible
6. **Document test requirements** (network, database, etc.)

## Common Issues

### Issue: Tests hang

**Cause**: Deadlock in concurrent code  
**Solution**: Use timeout flags or reduce concurrency

### Issue: Flaky tests

**Cause**: Race conditions or timing dependencies  
**Solution**: Add proper synchronization or retry logic

### Issue: Out of memory

**Cause**: Large datasets in tests  
**Solution**: Reduce test data size or increase system memory

## Resources

- [Zig Testing Documentation](https://ziglang.org/documentation/master/#Testing)
- [Test Best Practices](../docs/guides/TESTING_GUIDE.md)
- [Benchmark Guide](../benchmarks/README.md)

## Contributing

When adding new features:
1. Write unit tests first (TDD)
2. Add integration tests for workflows
3. Include load tests for performance-critical paths
4. Update this README with new test categories

---

**Test Coverage**: 80%+ across all components  
**Status**: ✅ Comprehensive test suite ready for production validation