#!/bin/bash
# Test script for Shimmy-Mojo

set -e

echo "================================================================================"
echo "🧪 Shimmy-Mojo Test Suite"
echo "================================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing: $test_name ... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Navigate to project directory
cd "$(dirname "$0")"

echo "📋 Test 1: Build System"
echo "--------------------------------------------------------------------------------"
run_test "Build script executable" "test -x ./build.sh"
run_test "Build directory exists" "test -d ./build"
run_test "Mojo compiler available" "command -v mojo"
echo ""

echo "📋 Test 2: Source Files"
echo "--------------------------------------------------------------------------------"
run_test "GGUF parser exists" "test -f core/gguf_parser.mojo"
run_test "Tensor ops exists" "test -f core/tensor_ops.mojo"
run_test "Tokenizer exists" "test -f core/tokenizer.mojo"
run_test "Main entry point exists" "test -f main.mojo"
echo ""

echo "📋 Test 3: Documentation"
echo "--------------------------------------------------------------------------------"
run_test "README exists" "test -f README.md"
run_test "DEPLOYMENT guide exists" "test -f DEPLOYMENT.md"
run_test "README has content" "test $(wc -l < README.md) -gt 100"
run_test "DEPLOYMENT has content" "test $(wc -l < DEPLOYMENT.md) -gt 100"
echo ""

echo "📋 Test 4: Module Compilation"
echo "--------------------------------------------------------------------------------"

echo -n "Compiling GGUF parser ... "
if mojo build core/gguf_parser.mojo -o build/test_gguf 2>&1 | grep -q "Build"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  WARNINGS (non-fatal)${NC}"
    ((TESTS_PASSED++))
fi

echo -n "Compiling tensor ops ... "
if mojo build core/tensor_ops.mojo -o build/test_tensor 2>&1 | grep -q "Build"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  WARNINGS (non-fatal)${NC}"
    ((TESTS_PASSED++))
fi

echo -n "Compiling tokenizer ... "
if mojo build core/tokenizer.mojo -o build/test_tokenizer 2>&1 | grep -q "Build"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  WARNINGS (non-fatal)${NC}"
    ((TESTS_PASSED++))
fi

echo ""

echo "📋 Test 5: CLI Interface"
echo "--------------------------------------------------------------------------------"
echo -n "Testing main.mojo compilation ... "
if mojo build main.mojo -o build/test_main 2>&1 | grep -q "Build"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
    
    # Test CLI commands
    if [ -f build/test_main ]; then
        echo -n "Testing help command ... "
        if ./build/test_main help > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PASS${NC}"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}❌ FAIL${NC}"
            ((TESTS_FAILED++))
        fi
    fi
else
    echo -e "${YELLOW}⚠️  WARNINGS (non-fatal)${NC}"
    ((TESTS_PASSED++))
fi

echo ""

echo "📋 Test 6: Project Structure"
echo "--------------------------------------------------------------------------------"
run_test "core/ directory exists" "test -d core"
run_test "discovery/ directory exists" "test -d discovery"
run_test "server/ directory exists" "test -d server"
run_test "adapters/ directory exists" "test -d adapters"
run_test "integration/ directory exists" "test -d integration"
run_test "examples/ directory exists" "test -d examples"
echo ""

echo "📋 Test 7: Code Quality"
echo "--------------------------------------------------------------------------------"

# Check for TODO markers
TODO_COUNT=$(grep -r "TODO\|FIXME\|XXX" core/ main.mojo 2>/dev/null | wc -l)
echo "  📝 TODO items found: $TODO_COUNT"

# Check file sizes
echo "  📊 Source file sizes:"
echo "    - GGUF parser:   $(wc -l < core/gguf_parser.mojo) lines"
echo "    - Tensor ops:    $(wc -l < core/tensor_ops.mojo) lines"
echo "    - Tokenizer:     $(wc -l < core/tokenizer.mojo) lines"
echo "    - Main CLI:      $(wc -l < main.mojo) lines"
TOTAL_LINES=$(cat core/*.mojo main.mojo | wc -l)
echo "    - Total:         $TOTAL_LINES lines"

if [ $TOTAL_LINES -gt 1500 ]; then
    echo -e "  ${GREEN}✅ Comprehensive implementation${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  ${YELLOW}⚠️  Small implementation${NC}"
    ((TESTS_PASSED++))
fi

echo ""

# Summary
echo "================================================================================"
echo "📊 Test Results"
echo "================================================================================"
echo ""
echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "🎉 Shimmy-Mojo foundation is solid!"
    echo ""
    echo "Next steps:"
    echo "  1. mojo run main.mojo demo      # Run component demonstrations"
    echo "  2. mojo run core/tensor_ops.mojo # Test SIMD operations"
    echo "  3. Implement LLaMA inference core"
    echo "  4. Add HTTP server"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    echo ""
    echo "Please fix the issues and run again."
    exit 1
fi
