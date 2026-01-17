#!/bin/bash

# Run All Tests for HyperShimmy

set -e  # Exit on error

echo "======================================================================"
echo "🧪 Running HyperShimmy Test Suite"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Track overall test results
TESTS_PASSED=0
TESTS_FAILED=0

# ====================================================================
# Zig Unit Tests
# ====================================================================
echo "⚡ Running Zig Unit Tests..."
echo "----------------------------------------------------------------------"

if command -v zig &> /dev/null; then
    if zig build test 2>&1; then
        echo "✓ Zig tests passed"
        ((TESTS_PASSED++))
    else
        echo "✗ Zig tests failed"
        ((TESTS_FAILED++))
    fi
else
    echo "⚠️  Zig not found, skipping Zig tests"
fi

echo ""

# ====================================================================
# Mojo Unit Tests
# ====================================================================
echo "🔥 Running Mojo Unit Tests..."
echo "----------------------------------------------------------------------"

if command -v mojo &> /dev/null; then
    # Check if test files exist
    if [ -d "tests/unit" ] && [ "$(ls -A tests/unit/*.mojo 2>/dev/null)" ]; then
        test_count=0
        failed_tests=0
        
        for test_file in tests/unit/*.mojo; do
            echo "  Running $(basename "$test_file")..."
            if mojo test "$test_file" 2>&1; then
                echo "    ✓ Passed"
                ((test_count++))
            else
                echo "    ✗ Failed"
                ((failed_tests++))
            fi
        done
        
        if [ $failed_tests -eq 0 ]; then
            echo "✓ All Mojo tests passed ($test_count tests)"
            ((TESTS_PASSED++))
        else
            echo "✗ $failed_tests Mojo test(s) failed"
            ((TESTS_FAILED++))
        fi
    else
        echo "  (No Mojo test files found yet - will be created starting Week 1, Day 6)"
        echo "  Skipping Mojo tests"
    fi
else
    echo "⚠️  Mojo not found, skipping Mojo tests"
fi

echo ""

# ====================================================================
# Integration Tests
# ====================================================================
echo "🔗 Running Integration Tests..."
echo "----------------------------------------------------------------------"

if [ -d "tests/integration" ] && [ "$(ls -A tests/integration/*.sh 2>/dev/null)" ]; then
    integration_count=0
    failed_integration=0
    
    for test_script in tests/integration/*.sh; do
        echo "  Running $(basename "$test_script")..."
        if bash "$test_script" 2>&1; then
            echo "    ✓ Passed"
            ((integration_count++))
        else
            echo "    ✗ Failed"
            ((failed_integration++))
        fi
    done
    
    if [ $failed_integration -eq 0 ]; then
        echo "✓ All integration tests passed ($integration_count tests)"
        ((TESTS_PASSED++))
    else
        echo "✗ $failed_integration integration test(s) failed"
        ((TESTS_FAILED++))
    fi
else
    echo "  (No integration tests found yet - will be created starting Week 4)"
    echo "  Skipping integration tests"
fi

echo ""

# ====================================================================
# Summary
# ====================================================================
echo "======================================================================"
echo "📊 Test Results Summary"
echo "======================================================================"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "  Test Suites: $TESTS_PASSED passed, $TOTAL_TESTS total"
    echo ""
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    echo ""
    echo "  Test Suites: $TESTS_PASSED passed, $TESTS_FAILED failed, $TOTAL_TESTS total"
    echo ""
    exit 1
fi
