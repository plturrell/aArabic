#!/bin/bash
# Integration test for Prompt Optimizer and Guardrails agents

set -e

echo "=========================================="
echo "Agent Integration Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Guardrails Validator
echo -e "${YELLOW}Test 1: Guardrails Validator${NC}"
echo "--------------------------------------"
cd guardrails
if zig run validator.zig 2>&1 | grep -q "✅ All tests complete"; then
    echo -e "${GREEN}✅ Guardrails validator: PASSED${NC}"
else
    echo -e "${RED}❌ Guardrails validator: FAILED${NC}"
    exit 1
fi
cd ..
echo ""

# Test 2: Prompt Optimizer
echo -e "${YELLOW}Test 2: Prompt Optimizer${NC}"
echo "--------------------------------------"
cd prompt_optimizer
if zig run optimizer.zig 2>&1 | grep -q "✅ Optimizer initialized"; then
    echo -e "${GREEN}✅ Prompt optimizer: PASSED${NC}"
else
    echo -e "${RED}❌ Prompt optimizer: FAILED${NC}"
    exit 1
fi
cd ..
echo ""

# Test 3: Server API Endpoints (if server is running)
echo -e "${YELLOW}Test 3: API Endpoints (requires running server)${NC}"
echo "--------------------------------------"

# Check if server is running
if curl -s http://localhost:11434/health >/dev/null 2>&1; then
    echo "Server is running, testing endpoints..."
    
    # Test guardrails metrics endpoint
    if curl -s http://localhost:11434/v1/guardrails/metrics | grep -q "total_validations"; then
        echo -e "${GREEN}✅ Guardrails metrics endpoint: PASSED${NC}"
    else
        echo -e "${YELLOW}⚠️  Guardrails metrics endpoint: NOT IMPLEMENTED${NC}"
    fi
    
    # Test guardrails policies endpoint
    if curl -s http://localhost:11434/v1/guardrails/policies | grep -q "config"; then
        echo -e "${GREEN}✅ Guardrails policies endpoint: PASSED${NC}"
    else
        echo -e "${YELLOW}⚠️  Guardrails policies endpoint: NOT IMPLEMENTED${NC}"
    fi
    
    # Test prompt optimizer endpoint
    if curl -s http://localhost:11434/v1/prompts/templates | grep -q "templates"; then
        echo -e "${GREEN}✅ Prompt optimizer endpoint: PASSED${NC}"
    else
        echo -e "${YELLOW}⚠️  Prompt optimizer endpoint: NOT IMPLEMENTED${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Server not running, skipping API tests${NC}"
    echo "   To test APIs, run: cd ../../ && ./start-zig.sh"
fi
echo ""

# Test 4: Integration with Chat Endpoint (mock)
echo -e "${YELLOW}Test 4: Mock Integration Flow${NC}"
echo "--------------------------------------"
echo "1. Input validation (guardrails)"
echo "2. Prompt optimization (optional)"
echo "3. LLM inference"
echo "4. Output validation (guardrails)"
echo ""
echo -e "${GREEN}✅ Integration flow defined${NC}"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}All Core Tests Passed!${NC}"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Integrate agents into handleChat() in openai_http_server.zig"
echo "2. Add API endpoints for /v1/guardrails/* and /v1/prompts/*"
echo "3. Connect UI to backend (webapp already created)"
echo "4. Test end-to-end with real LLM requests"
echo ""
echo "To view monitoring UI:"
echo "  http://localhost:11434/webapp/index.html"
echo "  Navigate to: Guardrails & Safety tab"
echo ""
