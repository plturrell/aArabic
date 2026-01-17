#!/bin/bash

# ============================================================================
# HyperShimmy Day 29 Test Script
# ============================================================================
#
# Tests Chat UI implementation
#
# Usage:
#   ./scripts/test_chat_ui.sh
#
# Tests:
# 1. Chat view structure
# 2. Chat controller implementation
# 3. OData integration
# 4. Message rendering
# 5. Metadata display
# 6. Source citations
# 7. CSS styling
# ============================================================================

set -e  # Exit on error

echo ""
echo "========================================================================"
echo "🧪 Day 29: Chat UI Tests"
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
# Test 1: Chat View Structure
# ============================================================================

echo "Test 1: Chat View Structure"
echo "------------------------------------------------------------------------"

if [ -f "webapp/view/Chat.view.xml" ]; then
    test_result 0 "Found Chat.view.xml"
else
    test_result 1 "Missing Chat.view.xml"
fi

if grep -q "chatMessagesContainer" webapp/view/Chat.view.xml; then
    test_result 0 "Chat messages container present"
else
    test_result 1 "Chat messages container missing"
fi

if grep -q "chatInput" webapp/view/Chat.view.xml; then
    test_result 0 "Chat input field present"
else
    test_result 1 "Chat input field missing"
fi

if grep -q "onSendMessage" webapp/view/Chat.view.xml; then
    test_result 0 "Send message handler present"
else
    test_result 1 "Send message handler missing"
fi

if grep -q "onClearChat" webapp/view/Chat.view.xml; then
    test_result 0 "Clear chat handler present"
else
    test_result 1 "Clear chat handler missing"
fi

echo ""

# ============================================================================
# Test 2: Chat Controller Implementation
# ============================================================================

echo "Test 2: Chat Controller Implementation"
echo "------------------------------------------------------------------------"

if [ -f "webapp/controller/Chat.controller.js" ]; then
    test_result 0 "Found Chat.controller.js"
else
    test_result 1 "Missing Chat.controller.js"
fi

# Check for session ID initialization
if grep -q "_sessionId" webapp/controller/Chat.controller.js; then
    test_result 0 "Session ID initialization present"
else
    test_result 1 "Session ID initialization missing"
fi

# Check for message rendering
if grep -q "_renderChatHistory" webapp/controller/Chat.controller.js; then
    test_result 0 "Chat history rendering method present"
else
    test_result 1 "Chat history rendering method missing"
fi

if grep -q "_createMessageBox" webapp/controller/Chat.controller.js; then
    test_result 0 "Message box creation method present"
else
    test_result 1 "Message box creation method missing"
fi

echo ""

# ============================================================================
# Test 3: OData Integration
# ============================================================================

echo "Test 3: OData Integration"
echo "------------------------------------------------------------------------"

# Check for OData action call
if grep -q "_callChatAction" webapp/controller/Chat.controller.js; then
    test_result 0 "OData Chat action method present"
else
    test_result 1 "OData Chat action method missing"
fi

# Check for correct endpoint
if grep -q "/odata/v4/research/Chat" webapp/controller/Chat.controller.js; then
    test_result 0 "Correct OData endpoint configured"
else
    test_result 1 "OData endpoint not configured"
fi

# Check for request payload structure
if grep -q "SessionId" webapp/controller/Chat.controller.js && \
   grep -q "Message" webapp/controller/Chat.controller.js && \
   grep -q "IncludeSources" webapp/controller/Chat.controller.js; then
    test_result 0 "Request payload structure correct"
else
    test_result 1 "Request payload structure incorrect"
fi

# Check for response handling
if grep -q "Content" webapp/controller/Chat.controller.js && \
   grep -q "SourceIds" webapp/controller/Chat.controller.js && \
   grep -q "Metadata" webapp/controller/Chat.controller.js; then
    test_result 0 "Response handling implemented"
else
    test_result 1 "Response handling missing"
fi

# Check for error handling
if grep -q "catch" webapp/controller/Chat.controller.js && \
   grep -q "error" webapp/controller/Chat.controller.js; then
    test_result 0 "Error handling present"
else
    test_result 1 "Error handling missing"
fi

echo ""

# ============================================================================
# Test 4: Message Formatting
# ============================================================================

echo "Test 4: Message Formatting"
echo "------------------------------------------------------------------------"

# Check for message content formatting
if grep -q "_formatMessageContent" webapp/controller/Chat.controller.js; then
    test_result 0 "Message content formatting method present"
else
    test_result 1 "Message content formatting method missing"
fi

# Check for HTML escaping
if grep -q "replace.*&amp;" webapp/controller/Chat.controller.js; then
    test_result 0 "HTML escaping implemented"
else
    test_result 1 "HTML escaping missing"
fi

# Check for markdown bold support
if grep -q "\\*\\*" webapp/controller/Chat.controller.js; then
    test_result 0 "Markdown bold formatting supported"
else
    test_result 1 "Markdown bold formatting missing"
fi

# Check for line break handling
if grep -q "replace.*\\\\n.*<br>" webapp/controller/Chat.controller.js; then
    test_result 0 "Line break handling present"
else
    test_result 1 "Line break handling missing"
fi

echo ""

# ============================================================================
# Test 5: Metadata Display
# ============================================================================

echo "Test 5: Metadata Display"
echo "------------------------------------------------------------------------"

# Check for metadata display method
if grep -q "_createMetadataDisplay" webapp/controller/Chat.controller.js; then
    test_result 0 "Metadata display method present"
else
    test_result 1 "Metadata display method missing"
fi

# Check for confidence display
if grep -q "confidence" webapp/controller/Chat.controller.js; then
    test_result 0 "Confidence indicator present"
else
    test_result 1 "Confidence indicator missing"
fi

# Check for query intent display
if grep -q "query_intent" webapp/controller/Chat.controller.js; then
    test_result 0 "Query intent display present"
else
    test_result 1 "Query intent display missing"
fi

# Check for performance info
if grep -q "total_time_ms" webapp/controller/Chat.controller.js; then
    test_result 0 "Performance info display present"
else
    test_result 1 "Performance info display missing"
fi

echo ""

# ============================================================================
# Test 6: Source Citations
# ============================================================================

echo "Test 6: Source Citations"
echo "------------------------------------------------------------------------"

# Check for sources display method
if grep -q "_createSourcesDisplay" webapp/controller/Chat.controller.js; then
    test_result 0 "Sources display method present"
else
    test_result 1 "Sources display method missing"
fi

# Check for source links
if grep -q "sapMLink" webapp/controller/Chat.controller.js || \
   grep -q "sap.m.Link" webapp/controller/Chat.controller.js; then
    test_result 0 "Source links implemented"
else
    test_result 1 "Source links missing"
fi

# Check for SourceIds handling
if grep -q "sourceIds" webapp/controller/Chat.controller.js; then
    test_result 0 "SourceIds handling present"
else
    test_result 1 "SourceIds handling missing"
fi

echo ""

# ============================================================================
# Test 7: CSS Styling
# ============================================================================

echo "Test 7: CSS Styling"
echo "------------------------------------------------------------------------"

if [ -f "webapp/css/style.css" ]; then
    test_result 0 "Found style.css"
else
    test_result 1 "Missing style.css"
fi

# Check for user message styling
if grep -q "chatMessageUser" webapp/css/style.css; then
    test_result 0 "User message styling present"
else
    test_result 1 "User message styling missing"
fi

# Check for assistant message styling
if grep -q "chatMessageAssistant" webapp/css/style.css; then
    test_result 0 "Assistant message styling present"
else
    test_result 1 "Assistant message styling missing"
fi

# Check for metadata styling
if grep -q "chatMetadata" webapp/css/style.css; then
    test_result 0 "Metadata styling present"
else
    test_result 1 "Metadata styling missing"
fi

# Check for sources styling
if grep -q "chatSources" webapp/css/style.css; then
    test_result 0 "Sources styling present"
else
    test_result 1 "Sources styling missing"
fi

# Check for input toolbar styling
if grep -q "chatInputToolbar" webapp/css/style.css; then
    test_result 0 "Input toolbar styling present"
else
    test_result 1 "Input toolbar styling missing"
fi

# Check for animations
if grep -q "@keyframes fadeIn" webapp/css/style.css; then
    test_result 0 "Message fade-in animation present"
else
    test_result 1 "Message fade-in animation missing"
fi

# Check for responsive design
if grep -q "@media.*max-width" webapp/css/style.css; then
    test_result 0 "Responsive design present"
else
    test_result 1 "Responsive design missing"
fi

echo ""

# ============================================================================
# Test 8: User Experience Features
# ============================================================================

echo "Test 8: User Experience Features"
echo "------------------------------------------------------------------------"

# Check for auto-scroll
if grep -q "_scrollToBottom" webapp/controller/Chat.controller.js; then
    test_result 0 "Auto-scroll to bottom present"
else
    test_result 1 "Auto-scroll to bottom missing"
fi

# Check for timestamp formatting
if grep -q "_formatTimestamp" webapp/controller/Chat.controller.js; then
    test_result 0 "Timestamp formatting present"
else
    test_result 1 "Timestamp formatting missing"
fi

# Check for busy state handling
if grep -q "busy" webapp/controller/Chat.controller.js; then
    test_result 0 "Busy state handling present"
else
    test_result 1 "Busy state handling missing"
fi

# Check for welcome message
if grep -q "chatWelcomeTitle\|chatWelcomeText" webapp/view/Chat.view.xml; then
    test_result 0 "Welcome message present"
else
    test_result 1 "Welcome message missing"
fi

echo ""

# ============================================================================
# Test 9: Integration with Previous Days
# ============================================================================

echo "Test 9: Integration with Previous Days"
echo "------------------------------------------------------------------------"

# Check Day 28 OData action integration
if [ -f "server/odata_chat.zig" ]; then
    test_result 0 "Day 28 OData action present"
else
    test_result 1 "Day 28 OData action missing"
fi

# Check that endpoint matches Day 28
if grep -q "/odata/v4/research/Chat" webapp/controller/Chat.controller.js && \
   grep -q "/odata/v4/research/Chat" server/main.zig; then
    test_result 0 "Endpoint matches Day 28 implementation"
else
    test_result 1 "Endpoint mismatch with Day 28"
fi

# Check for proper JSON model usage
if grep -q "JSONModel" webapp/controller/Chat.controller.js; then
    test_result 0 "JSON model imported"
else
    test_result 1 "JSON model not imported"
fi

echo ""

# ============================================================================
# Test 10: Code Quality
# ============================================================================

echo "Test 10: Code Quality & Documentation"
echo "------------------------------------------------------------------------"

# Check for JSDoc comments
if grep -q "@param\|@returns\|@private" webapp/controller/Chat.controller.js; then
    test_result 0 "JSDoc comments present"
else
    test_result 1 "JSDoc comments missing"
fi

# Check controller size is reasonable
CONTROLLER_LINES=$(wc -l < webapp/controller/Chat.controller.js | tr -d ' ')
if [ "$CONTROLLER_LINES" -gt 300 ] && [ "$CONTROLLER_LINES" -lt 600 ]; then
    test_result 0 "Controller size reasonable (~$CONTROLLER_LINES lines)"
else
    test_result 1 "Controller size unusual ($CONTROLLER_LINES lines)"
fi

# Check CSS size is reasonable
CSS_LINES=$(wc -l < webapp/css/style.css | tr -d ' ')
if [ "$CSS_LINES" -gt 100 ] && [ "$CSS_LINES" -lt 400 ]; then
    test_result 0 "CSS size reasonable (~$CSS_LINES lines)"
else
    test_result 1 "CSS size unusual ($CSS_LINES lines)"
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
    echo -e "${GREEN}✅ All Day 29 tests PASSED!${NC}"
    echo ""
    echo "Day 29 Implementation Complete:"
    echo "  ✓ Chat UI with message display"
    echo "  ✓ OData Chat action integration"
    echo "  ✓ Metadata and confidence display"
    echo "  ✓ Source citations"
    echo "  ✓ Rich formatting and styling"
    echo "  ✓ Error handling"
    echo "  ✓ Responsive design"
    echo ""
    echo "Next Steps (Day 30):"
    echo "  • Add streaming support"
    echo "  • Implement real-time updates"
    echo "  • Add typing indicators"
    echo "  • Enhance performance"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
    echo ""
    echo "Please fix the failing tests before proceeding to Day 30."
    echo ""
    exit 1
fi
