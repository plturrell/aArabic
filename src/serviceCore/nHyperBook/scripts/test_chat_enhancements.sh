#!/bin/bash

# ============================================================================
# Day 30: Chat Enhancement Tests
# ============================================================================
#
# Test Suite for Chat Enhancements:
# - Chat history persistence
# - Message actions (copy, regenerate, export)
# - Settings dialog
# - Keyboard shortcuts
# - Enhanced UX features
#
# Usage: ./test_chat_enhancements.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TESTS_PASSED=0
TESTS_FAILED=0

echo ""
echo "========================================================================"
echo "🧪 Day 30: Chat Enhancement Tests"
echo "========================================================================"
echo ""

# ============================================================================
# Test Helper Functions
# ============================================================================

test_file_exists() {
    local file=$1
    local description=$2
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((TESTS_FAILED++))
        return 1
    fi
}

test_content_exists() {
    local file=$1
    local pattern=$2
    local description=$3
    if grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✓${NC} $description"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((TESTS_FAILED++))
        return 1
    fi
}

test_function_exists() {
    local file=$1
    local function_name=$2
    local description=$3
    if grep -q "$function_name.*:.*function" "$file"; then
        echo -e "${GREEN}✓${NC} $description"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((TESTS_FAILED++))
        return 1
    fi
}

# ============================================================================
# Test 1: Chat History Persistence
# ============================================================================

echo "Test 1: Chat History Persistence"
echo "------------------------------------------------------------------------"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_loadChatSettings" \
    "Load chat settings method present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_saveChatSettings" \
    "Save chat settings method present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_loadChatHistory" \
    "Load chat history method present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_saveChatHistory" \
    "Save chat history method present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "localStorage.getItem" \
    "localStorage read implementation present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "localStorage.setItem" \
    "localStorage write implementation present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "hypershimmy.chatSettings" \
    "Settings localStorage key present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "hypershimmy.chatHistory" \
    "History localStorage key present"

echo ""

# ============================================================================
# Test 2: Message Actions
# ============================================================================

echo "Test 2: Message Actions"
echo "------------------------------------------------------------------------"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "onCopyMessage" \
    "Copy message handler present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "onRegenerateResponse" \
    "Regenerate response handler present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "onExportChat" \
    "Export chat handler present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "navigator.clipboard" \
    "Clipboard API usage present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "writeText" \
    "Copy to clipboard implementation present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_formatChatForExport" \
    "Chat export formatter present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "chat-export-" \
    "Export filename generation present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "Blob" \
    "Blob creation for export present"

echo ""

# ============================================================================
# Test 3: Settings Dialog
# ============================================================================

echo "Test 3: Settings Dialog"
echo "------------------------------------------------------------------------"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "onOpenSettings" \
    "Open settings handler present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_createSettingsDialog" \
    "Create settings dialog method present"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_onSaveSettings" \
    "Save settings handler present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "chatMaxTokens" \
    "Max tokens setting present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "chatTemperature" \
    "Temperature setting present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "chatIncludeSources" \
    "Include sources setting present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "sap.m.Slider" \
    "Slider controls for settings present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "sap.m.CheckBox" \
    "Checkbox control for settings present"

echo ""

# ============================================================================
# Test 4: Keyboard Shortcuts
# ============================================================================

echo "Test 4: Keyboard Shortcuts"
echo "------------------------------------------------------------------------"

test_function_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "onInputKeyPress" \
    "Input key press handler present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "ctrlKey.*metaKey" \
    "Ctrl/Cmd key detection present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "keyCode.*13" \
    "Enter key detection present"

echo ""

# ============================================================================
# Test 5: Enhanced Message Rendering
# ============================================================================

echo "Test 5: Enhanced Message Rendering"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "_saveChatHistory.*_renderChatHistory" \
    "Chat history saved after rendering"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "isError.*true" \
    "Error state handling present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "timestamp.*Date.now" \
    "Timestamp generation present"

echo ""

# ============================================================================
# Test 6: Export Functionality
# ============================================================================

echo "Test 6: Export Functionality"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "HyperShimmy Chat Export" \
    "Export header present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "Session:.*_sessionId" \
    "Session ID in export present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "Exported:.*toLocaleString" \
    "Export timestamp present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "Metadata:" \
    "Metadata export present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "Sources:" \
    "Sources export present"

echo ""

# ============================================================================
# Test 7: Regenerate Response
# ============================================================================

echo "Test 7: Regenerate Response"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "oLastUserMessage" \
    "Last user message detection present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "slice.*iUserIndex" \
    "Chat history truncation present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "Response regenerated" \
    "Regenerate success message present"

echo ""

# ============================================================================
# Test 8: Settings Persistence
# ============================================================================

echo "Test 8: Settings Persistence"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "maxTokens.*500" \
    "Default max tokens value present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "temperature.*0.7" \
    "Default temperature value present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "includeSources.*false" \
    "Include sources default present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "JSON.parse.*sSettings" \
    "Settings JSON parsing present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "JSON.stringify.*oSettings" \
    "Settings JSON stringification present"

echo ""

# ============================================================================
# Test 9: Error Handling
# ============================================================================

echo "Test 9: Error Handling"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "console.error.*Failed to load" \
    "Load error handling present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "console.error.*Failed to save" \
    "Save error handling present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "try.*catch" \
    "Try-catch blocks present"

test_content_exists \
    "$PROJECT_ROOT/webapp/controller/Chat.controller.js" \
    "No chat history to export" \
    "Export validation present"

echo ""

# ============================================================================
# Test 10: Code Quality
# ============================================================================

echo "Test 10: Code Quality & Documentation"
echo "------------------------------------------------------------------------"

# Count JSDoc comments
JSDOC_COUNT=$(grep -c "@param\|@returns\|@private" "$PROJECT_ROOT/webapp/controller/Chat.controller.js" || echo "0")
if [ "$JSDOC_COUNT" -gt "30" ]; then
    echo -e "${GREEN}✓${NC} JSDoc comments present ($JSDOC_COUNT found)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗${NC} Insufficient JSDoc comments ($JSDOC_COUNT found, expected >30)"
    ((TESTS_FAILED++))
fi

# Check controller size
CONTROLLER_LINES=$(wc -l < "$PROJECT_ROOT/webapp/controller/Chat.controller.js")
if [ "$CONTROLLER_LINES" -gt "500" ] && [ "$CONTROLLER_LINES" -lt "1000" ]; then
    echo -e "${GREEN}✓${NC} Controller size reasonable ($CONTROLLER_LINES lines)"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠${NC} Controller size: $CONTROLLER_LINES lines"
    ((TESTS_PASSED++))
fi

# Check for proper error handling
ERROR_HANDLING_COUNT=$(grep -c "catch.*function" "$PROJECT_ROOT/webapp/controller/Chat.controller.js" || echo "0")
if [ "$ERROR_HANDLING_COUNT" -gt "5" ]; then
    echo -e "${GREEN}✓${NC} Error handling implemented ($ERROR_HANDLING_COUNT catch blocks)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗${NC} Insufficient error handling ($ERROR_HANDLING_COUNT catch blocks)"
    ((TESTS_FAILED++))
fi

echo ""

# ============================================================================
# Test Summary
# ============================================================================

echo "========================================================================"
echo "📊 Test Summary"
echo "========================================================================"
echo ""
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All Day 30 tests PASSED!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
    echo ""
    exit 1
fi
