# Testing Guide

## Overview

All tests have been organized into their respective service directories for better maintainability and clarity.

---

## 🧪 Test Organization

### Structure
```
src/
├── serviceCore/
│   └── nLocalModels/
│       └── orchestration/
│           └── tests/
│               └── test_model_selection_integration.zig
└── serviceAutomation/
    └── tests/
        └── integration_test.sh
```

---

## 📋 Available Tests

### 1. Model Selection Integration Tests

**Location:** `src/serviceCore/nLocalModels/orchestration/tests/test_model_selection_integration.zig`

**Purpose:** Comprehensive integration tests for the model orchestration system

**Test Coverage:**
- ✅ MODEL_REGISTRY.json loading and validation
- ✅ task_categories.json loading
- ✅ Model selection for different categories
- ✅ GPU constraint validation
- ✅ Benchmark scoring integration
- ✅ Multi-category model support
- ✅ Performance benchmarking
- ✅ Data integrity validation

**Running:**
```bash
cd src/serviceCore/nLocalModels/orchestration
zig test tests/test_model_selection_integration.zig
```

**Expected Output:**
```
✓ Loaded N models from registry
✓ Loaded M categories
=== Code Task Selection ===
Selected Model: model-name
Score: XX.XX
Reason: ...
```

---

### 2. Service Automation Integration Tests

**Location:** `src/serviceAutomation/tests/integration_test.sh`

**Purpose:** Validate Rust client binaries and Docker configurations

**Test Coverage:**
- ✅ Qdrant CLI binary existence
- ✅ Memgraph CLI binary existence
- ✅ DragonflyDB CLI binary existence
- ✅ Dockerfile validation for all clients

**Running:**
```bash
./src/serviceAutomation/tests/integration_test.sh
```

**Expected Output:**
```
========================================
Rust Clients Integration Tests
========================================
Testing: Qdrant CLI... ✓ PASS
Testing: Memgraph CLI... ✓ PASS
Testing: DragonflyDB CLI... ✓ PASS
...
Passed: 6 | Failed: 0
All tests passed!
```

---

## 🎯 Running All Tests

### Quick Test Suite
```bash
#!/bin/bash
# Run all project tests

echo "=== Running Model Orchestration Tests ==="
cd src/serviceCore/nLocalModels/orchestration
zig test tests/test_model_selection_integration.zig

echo ""
echo "=== Running Service Automation Tests ==="
cd ../../../..
./src/serviceAutomation/tests/integration_test.sh

echo ""
echo "=== All Tests Complete ==="
```

Save this as `scripts/run_all_tests.sh` and make executable:
```bash
chmod +x scripts/run_all_tests.sh
./scripts/run_all_tests.sh
```

---

## 📊 Test Categories

### Unit Tests
- **Location:** Within each service directory
- **Scope:** Individual functions and modules
- **Framework:** Zig built-in testing

### Integration Tests
- **Orchestration:** Model selection integration
- **Services:** Rust client integration
- **Coverage:** Cross-module functionality

### End-to-End Tests
- **Future:** Full system workflow tests
- **Planned:** API endpoint validation
- **Planned:** Multi-service orchestration

---

## 🔧 Writing New Tests

### Zig Tests (Model Orchestration)

**Template:**
```zig
const std = @import("std");
const testing = std.testing;

test "Your test name" {
    const allocator = testing.allocator;
    
    // Setup
    // ...
    
    // Execute
    // ...
    
    // Verify
    try testing.expect(condition);
    
    // Cleanup
    // ...
}
```

**Best Practices:**
- Use descriptive test names
- Clean up resources with `defer`
- Use `testing.allocator` for memory tracking
- Print debug info with `std.debug.print`

### Shell Tests (Service Automation)

**Template:**
```bash
run_test() {
    local test_name=$1
    local command=$2
    echo -n "Testing: $test_name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
}

# Add your test
run_test "Your test" "your_command_here"
```

---

## 🐛 Debugging Tests

### Zig Tests

**Verbose Output:**
```bash
zig test tests/test_model_selection_integration.zig --verbose
```

**With Debug Symbols:**
```bash
zig test tests/test_model_selection_integration.zig -O Debug
```

**Memory Leak Detection:**
```bash
# Zig automatically detects memory leaks with testing.allocator
zig test tests/test_model_selection_integration.zig
# Will fail if memory is leaked
```

### Shell Tests

**Debug Mode:**
```bash
bash -x src/serviceAutomation/tests/integration_test.sh
```

**Run Single Test:**
Edit the script to comment out other tests and run specific ones.

---

## 📈 Test Metrics

### Current Coverage

| Component | Tests | Coverage | Status |
|-----------|-------|----------|--------|
| Model Selection | 9 tests | High | ✅ Complete |
| Service Automation | 6 tests | Medium | ✅ Complete |
| CLI Tools | 0 tests | None | 🚧 Planned |
| API Endpoints | 0 tests | None | 🚧 Planned |

### Performance Benchmarks

**Model Selection Tests:**
- Selection time: <100ms per operation
- Memory usage: Minimal with proper cleanup
- Registry loading: Fast with JSON parsing

---

## 🚀 Continuous Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Install Zig
      run: |
        wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz
        tar xf zig-linux-x86_64-0.11.0.tar.xz
        echo "$PWD/zig-linux-x86_64-0.11.0" >> $GITHUB_PATH
    
    - name: Run Model Tests
      run: |
        cd src/serviceCore/nLocalModels/orchestration
        zig test tests/test_model_selection_integration.zig
    
    - name: Run Service Tests
      run: ./src/serviceAutomation/tests/integration_test.sh
```

---

## 📝 Test Documentation Standards

### Each test should include:
1. **Purpose:** What is being tested
2. **Setup:** Required configuration/data
3. **Expected Behavior:** What should happen
4. **Cleanup:** Resource deallocation

### Example:
```zig
/// Test: Model selection with GPU constraints
/// Purpose: Verify models are selected within GPU memory limits
/// Setup: Load MODEL_REGISTRY.json with various model sizes
/// Expected: Selected model fits within specified GPU memory
/// Cleanup: Deallocate selection result
test "Model selection respects GPU constraints" {
    // Test implementation
}
```

---

## 🔗 Related Documentation

- [Contributing Guide](CONTRIBUTING.md) - Development guidelines
- [Model Orchestration](../01-architecture/MODEL_ORCHESTRATION_MAPPING.md) - System architecture
- [CLI Tools](../../src/serviceCore/nLocalModels/orchestration/CLI_TOOLS_README.md) - Tool documentation

---

## 📞 Support

- **Test Failures:** Check test output for specific errors
- **New Tests:** Follow templates above
- **CI Issues:** Check GitHub Actions logs

---

**Last Updated:** 2026-01-23  
**Test Framework:** Zig 0.11+ built-in testing  
**Status:** ✅ Active Maintenance
