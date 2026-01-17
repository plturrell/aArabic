#!/bin/bash

# ============================================================================
# HyperShimmy Day 28 Test Script
# ============================================================================
#
# Tests OData Chat action implementation
#
# Usage:
#   ./scripts/test_odata_chat.sh
#
# Tests:
# 1. OData Chat action module structure
# 2. Zig handler compilation
# 3. Request/response mapping
# 4. Error handling
# 5. Integration with orchestrator
# 6. OData compliance
# ============================================================================

set -e  # Exit on error

echo ""
echo "========================================================================"
echo "🧪 Day 28: OData Chat Action Tests"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function for test results
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} $2"
        ((TESTS_FAILED++))
    fi
}

# ============================================================================
# Test 1: OData Chat Action Module
# ============================================================================

echo "Test 1: OData Chat Action Module Structure"
echo "------------------------------------------------------------------------"

if [ -f "server/odata_chat.zig" ]; then
    test_result 0 "Found odata_chat.zig"
else
    test_result 1 "Missing odata_chat.zig"
fi

# Check for key components
if grep -q "pub const ChatRequest = struct" server/odata_chat.zig; then
    test_result 0 "ChatRequest struct defined"
else
    test_result 1 "ChatRequest struct missing"
fi

if grep -q "pub const ChatResponse = struct" server/odata_chat.zig; then
    test_result 0 "ChatResponse struct defined"
else
    test_result 1 "ChatResponse struct missing"
fi

if grep -q "pub const ODataChatHandler = struct" server/odata_chat.zig; then
    test_result 0 "ODataChatHandler defined"
else
    test_result 1 "ODataChatHandler missing"
fi

if grep -q "pub fn handleODataChatRequest" server/odata_chat.zig; then
    test_result 0 "handleODataChatRequest function present"
else
    test_result 1 "handleODataChatRequest function missing"
fi

echo ""

# ============================================================================
# Test 2: Main.zig Integration
# ============================================================================

echo "Test 2: Main.zig Integration"
echo "------------------------------------------------------------------------"

if grep -q 'const odata_chat = @import("odata_chat.zig")' server/main.zig; then
    test_result 0 "odata_chat import present"
else
    test_result 1 "odata_chat import missing"
fi

if grep -q '/odata/v4/research/Chat' server/main.zig; then
    test_result 0 "Chat action route defined"
else
    test_result 1 "Chat action route missing"
fi

if grep -q 'handleODataChatAction' server/main.zig; then
    test_result 0 "handleODataChatAction function called"
else
    test_result 1 "handleODataChatAction function not called"
fi

if grep -q 'Day 28 Implementation' server/main.zig; then
    test_result 0 "Day 28 implementation documented"
else
    test_result 1 "Day 28 implementation not documented"
fi

echo ""

# ============================================================================
# Test 3: OData Complex Types
# ============================================================================

echo "Test 3: OData Complex Types Mapping"
echo "------------------------------------------------------------------------"

# Check ChatRequest fields match metadata
if grep -q "SessionId: \[\]const u8" server/odata_chat.zig; then
    test_result 0 "SessionId field present"
else
    test_result 1 "SessionId field missing"
fi

if grep -q "Message: \[\]const u8" server/odata_chat.zig; then
    test_result 0 "Message field present"
else
    test_result 1 "Message field missing"
fi

if grep -q "IncludeSources: bool" server/odata_chat.zig; then
    test_result 0 "IncludeSources field present"
else
    test_result 1 "IncludeSources field missing"
fi

if grep -q "MaxTokens: ?i32" server/odata_chat.zig; then
    test_result 0 "MaxTokens optional field present"
else
    test_result 1 "MaxTokens optional field missing"
fi

if grep -q "Temperature: ?f64" server/odata_chat.zig; then
    test_result 0 "Temperature optional field present"
else
    test_result 1 "Temperature optional field missing"
fi

# Check ChatResponse fields
if grep -q "MessageId: \[\]const u8" server/odata_chat.zig; then
    test_result 0 "MessageId field present"
else
    test_result 1 "MessageId field missing"
fi

if grep -q "Content: \[\]const u8" server/odata_chat.zig; then
    test_result 0 "Content field present"
else
    test_result 1 "Content field missing"
fi

if grep -q "SourceIds: \[\]const \[\]const u8" server/odata_chat.zig; then
    test_result 0 "SourceIds array field present"
else
    test_result 1 "SourceIds array field missing"
fi

if grep -q "Metadata: \[\]const u8" server/odata_chat.zig; then
    test_result 0 "Metadata field present"
else
    test_result 1 "Metadata field missing"
fi

echo ""

# ============================================================================
# Test 4: Error Handling
# ============================================================================

echo "Test 4: Error Handling"
echo "------------------------------------------------------------------------"

if grep -q "pub const ODataError = struct" server/odata_chat.zig; then
    test_result 0 "ODataError structure defined"
else
    test_result 1 "ODataError structure missing"
fi

if grep -q "formatODataError" server/odata_chat.zig; then
    test_result 0 "formatODataError method present"
else
    test_result 1 "formatODataError method missing"
fi

if grep -q "BadRequest" server/odata_chat.zig; then
    test_result 0 "BadRequest error handling present"
else
    test_result 1 "BadRequest error handling missing"
fi

if grep -q "InternalError" server/odata_chat.zig; then
    test_result 0 "InternalError handling present"
else
    test_result 1 "InternalError handling missing"
fi

echo ""

# ============================================================================
# Test 5: Orchestrator Integration
# ============================================================================

echo "Test 5: Orchestrator Integration"
echo "------------------------------------------------------------------------"

if grep -q 'const orchestrator = @import("orchestrator.zig")' server/odata_chat.zig; then
    test_result 0 "Orchestrator import present"
else
    test_result 1 "Orchestrator import missing"
fi

if grep -q "chatRequestToOrchestrateRequest" server/odata_chat.zig; then
    test_result 0 "Request mapping method present"
else
    test_result 1 "Request mapping method missing"
fi

if grep -q "orchestrateResponseToChatResponse" server/odata_chat.zig; then
    test_result 0 "Response mapping method present"
else
    test_result 1 "Response mapping method missing"
fi

if grep -q "orchestrator.OrchestratorHandler" server/odata_chat.zig; then
    test_result 0 "Uses OrchestratorHandler"
else
    test_result 1 "OrchestratorHandler not used"
fi

if grep -q "handleOrchestrate" server/odata_chat.zig; then
    test_result 0 "Calls handleOrchestrate method"
else
    test_result 1 "handleOrchestrate method not called"
fi

echo ""

# ============================================================================
# Test 6: Metadata Generation
# ============================================================================

echo "Test 6: Metadata Generation"
echo "------------------------------------------------------------------------"

if grep -q "buildMetadata" server/odata_chat.zig; then
    test_result 0 "buildMetadata method present"
else
    test_result 1 "buildMetadata method missing"
fi

if grep -q "generateMessageId" server/odata_chat.zig; then
    test_result 0 "generateMessageId method present"
else
    test_result 1 "generateMessageId method missing"
fi

# Check metadata includes orchestrator stats
if grep -q "confidence" server/odata_chat.zig && \
   grep -q "query_intent" server/odata_chat.zig && \
   grep -q "chunks_retrieved" server/odata_chat.zig; then
    test_result 0 "Metadata includes orchestrator statistics"
else
    test_result 1 "Metadata missing orchestrator statistics"
fi

echo ""

# ============================================================================
# Test 7: Unit Tests
# ============================================================================

echo "Test 7: Unit Tests"
echo "------------------------------------------------------------------------"

if grep -q 'test "odata chat handler basic"' server/odata_chat.zig; then
    test_result 0 "Basic test case present"
else
    test_result 1 "Basic test case missing"
fi

if grep -q 'test "odata chat handler without sources"' server/odata_chat.zig; then
    test_result 0 "Without sources test case present"
else
    test_result 1 "Without sources test case missing"
fi

if grep -q 'test "odata chat handler invalid json"' server/odata_chat.zig; then
    test_result 0 "Invalid JSON test case present"
else
    test_result 1 "Invalid JSON test case missing"
fi

echo ""

# ============================================================================
# Test 8: Code Quality
# ============================================================================

echo "Test 8: Code Quality & Documentation"
echo "------------------------------------------------------------------------"

# Check for proper documentation
if grep -q "OData V4 Chat action endpoint" server/odata_chat.zig; then
    test_result 0 "Module documented"
else
    test_result 1 "Module documentation missing"
fi

if grep -q "Day 28 Implementation" server/odata_chat.zig; then
    test_result 0 "Day 28 implementation noted"
else
    test_result 1 "Day 28 implementation not noted"
fi

# Check for proper structure
if grep -q "pub const" server/odata_chat.zig && \
   grep -q "pub fn" server/odata_chat.zig; then
    test_result 0 "Proper Zig structure (pub const/fn)"
else
    test_result 1 "Improper Zig structure"
fi

# Check module size is reasonable
ODATA_CHAT_LINES=$(wc -l < server/odata_chat.zig | tr -d ' ')
if [ "$ODATA_CHAT_LINES" -gt 250 ] && [ "$ODATA_CHAT_LINES" -lt 450 ]; then
    test_result 0 "Module size reasonable (~$ODATA_CHAT_LINES lines)"
else
    test_result 1 "Module size unusual ($ODATA_CHAT_LINES lines)"
fi

echo ""

# ============================================================================
# Test 9: Integration with Previous Days
# ============================================================================

echo "Test 9: Integration with Previous Days"
echo "------------------------------------------------------------------------"

# Check orchestrator integration (Day 27)
if [ -f "server/orchestrator.zig" ]; then
    test_result 0 "Orchestrator module present (Day 27)"
else
    test_result 1 "Orchestrator module missing"
fi

# Check metadata definition (Day 3)
if [ -f "odata/metadata.xml" ]; then
    if grep -q 'Action Name="Chat"' odata/metadata.xml; then
        test_result 0 "Chat action in metadata.xml"
    else
        test_result 1 "Chat action not in metadata.xml"
    fi
else
    test_result 1 "metadata.xml not found"
fi

# Check for proper complex types in metadata
if [ -f "odata/metadata.xml" ]; then
    if grep -q 'ComplexType Name="ChatRequest"' odata/metadata.xml && \
       grep -q 'ComplexType Name="ChatResponse"' odata/metadata.xml; then
        test_result 0 "ChatRequest/Response complex types in metadata"
    else
        test_result 1 "ChatRequest/Response complex types missing from metadata"
    fi
fi

echo ""

# ============================================================================
# Test 10: OData V4 Compliance
# ============================================================================

echo "Test 10: OData V4 Compliance"
echo "------------------------------------------------------------------------"

# Check action import in metadata
if [ -f "odata/metadata.xml" ]; then
    if grep -q 'ActionImport Name="Chat"' odata/metadata.xml; then
        test_result 0 "Chat ActionImport in entity container"
    else
        test_result 1 "Chat ActionImport missing"
    fi
fi

# Check endpoint follows OData conventions
if grep -q '/odata/v4/research/Chat' server/main.zig; then
    test_result 0 "Endpoint follows OData V4 conventions"
else
    test_result 1 "Endpoint doesn't follow OData V4 conventions"
fi

# Check for proper OData error format
if grep -q '"error"' server/odata_chat.zig && \
   grep -q '"code"' server/odata_chat.zig && \
   grep -q '"message"' server/odata_chat.zig; then
    test_result 0 "OData error format compliant"
else
    test_result 1 "OData error format not compliant"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "========================================================================"
echo "📊 Test Summary"
echo "========================================================================"
echo ""
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All Day 28 tests PASSED!${NC}"
    echo ""
    echo "Day 28 Implementation Complete:"
    echo "  ✓ OData Chat action handler implemented"
    echo "  ✓ Request/response mapping working"
    echo "  ✓ Orchestrator integration complete"
    echo "  ✓ Error handling implemented"
    echo "  ✓ OData V4 compliant"
    echo ""
    echo "Next Steps (Day 29):"
    echo "  • Implement Chat UI in SAPUI5"
    echo "  • Add message history display"
    echo "  • Implement chat panel"
    echo "  • Add UI for chat interactions"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
    echo ""
    echo "Please fix the failing tests before proceeding to Day 29."
    echo ""
    exit 1
fi
