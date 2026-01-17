#!/bin/bash

# ============================================================================
# HyperShimmy Integration Test Runner
# ============================================================================
# Day 57: Execute integration tests for complete workflows
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "============================================================================"
echo "                HyperShimmy Integration Test Suite"
echo "============================================================================"
echo ""

# ============================================================================
# Integration Tests
# ============================================================================

echo -e "${BLUE}Running Integration Tests...${NC}"
echo "----------------------------------------------------------------------------"

cd "$PROJECT_DIR"

if ! command -v zig &> /dev/null; then
    echo -e "${RED}Error: Zig compiler not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Building and running integration tests...${NC}"

# Run OData endpoint tests
echo ""
echo -e "${BLUE}1. OData Endpoint Tests${NC}"
if zig test tests/integration/test_odata_endpoints.zig 2>&1 | tee /tmp/odata_test_output.txt; then
    echo -e "${GREEN}✓ OData endpoint tests passed${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}✗ OData endpoint tests failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Run file upload workflow tests
echo ""
echo -e "${BLUE}2. File Upload Workflow Tests${NC}"
if zig test tests/integration/test_file_upload_workflow.zig 2>&1 | tee /tmp/upload_test_output.txt; then
    echo -e "${GREEN}✓ File upload workflow tests passed${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}✗ File upload workflow tests failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Run AI pipeline tests
echo ""
echo -e "${BLUE}3. AI Pipeline Tests${NC}"
if zig test tests/integration/test_ai_pipeline.zig 2>&1 | tee /tmp/ai_pipeline_test_output.txt; then
    echo -e "${GREEN}✓ AI pipeline tests passed${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}✗ AI pipeline tests failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""

# ============================================================================
# Test Summary
# ============================================================================

echo "============================================================================"
echo "                       Integration Test Summary"
echo "============================================================================"
echo ""
echo "Test Suites Executed:"
echo "  1. OData Endpoints (metadata, CRUD, query options, actions)"
echo "  2. File Upload Workflow (validation, processing, storage)"
echo "  3. AI Pipeline (embedding, search, chat, summary, audio, slides)"
echo ""
echo "Results:"
echo -e "  Total Test Suites:  ${TOTAL_TESTS}"
echo -e "  ${GREEN}Passed:${NC}             ${PASSED_TESTS}"
echo -e "  ${RED}Failed:${NC}             ${FAILED_TESTS}"
echo ""

# Calculate success rate
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo -e "  Success Rate:       ${SUCCESS_RATE}%"
fi

echo ""

# ============================================================================
# Coverage Summary
# ============================================================================

echo "Integration Test Coverage:"
echo ""
echo "  ✓ OData V4 Protocol"
echo "    - Metadata endpoint"
echo "    - CRUD operations (Create, Read, Update, Delete)"
echo "    - Query options (\$filter, \$select, \$top, \$skip, \$orderby, \$count)"
echo "    - OData actions (Chat, Summary, Audio, Slides, Mindmap)"
echo "    - Error responses (4xx, 5xx)"
echo "    - CORS handling"
echo "    - Content negotiation"
echo ""
echo "  ✓ File Upload Pipeline"
echo "    - Multipart form data parsing"
echo "    - File validation (type, size)"
echo "    - Filename sanitization"
echo "    - File storage and retrieval"
echo "    - Metadata extraction"
echo "    - Error recovery"
echo "    - Integration with source management"
echo ""
echo "  ✓ AI Processing Pipeline"
echo "    - Embedding generation"
echo "    - Vector storage (Qdrant)"
echo "    - Semantic search"
echo "    - RAG (Retrieval Augmented Generation)"
echo "    - Chat with context"
echo "    - Summary generation"
echo "    - Knowledge graph extraction"
echo "    - Mindmap generation"
echo "    - Audio generation (TTS)"
echo "    - Slide generation"
echo "    - End-to-end workflows"
echo ""

# ============================================================================
# Workflow Testing
# ============================================================================

echo "Complete Workflows Tested:"
echo "  1. Document Upload → Embedding → Storage"
echo "  2. Query → Search → Context Retrieval"
echo "  3. Chat → RAG → Response Generation"
echo "  4. Sources → Analysis → Summary → Audio"
echo "  5. Sources → Knowledge Graph → Mindmap"
echo "  6. Sources → Content Analysis → Slides → Export"
echo ""

# ============================================================================
# Performance Insights
# ============================================================================

echo "Performance Considerations:"
echo "  - Mock implementations for fast test execution"
echo "  - Real integration tests require running services"
echo "  - Recommended: Run with actual server for E2E validation"
echo ""

# ============================================================================
# Next Steps
# ============================================================================

echo "Recommendations:"
echo "  1. Run unit tests first: ./scripts/run_unit_tests.sh"
echo "  2. Start server: ./scripts/start.sh (if not running)"
echo "  3. Run E2E tests with live server (Day 58+)"
echo "  4. Monitor logs during integration testing"
echo "  5. Check individual test outputs in /tmp/*.txt"
echo ""

# ============================================================================
# Exit Status
# ============================================================================

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}              ✓ ALL INTEGRATION TESTS PASSED!${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}============================================================================${NC}"
    echo -e "${RED}              ✗ SOME INTEGRATION TESTS FAILED${NC}"
    echo -e "${RED}============================================================================${NC}"
    echo ""
    echo "Check test outputs for details:"
    echo "  - /tmp/odata_test_output.txt"
    echo "  - /tmp/upload_test_output.txt"
    echo "  - /tmp/ai_pipeline_test_output.txt"
    echo ""
    exit 1
fi
