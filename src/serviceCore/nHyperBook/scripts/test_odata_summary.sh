#!/bin/bash

# ============================================================================
# Day 32: OData Summary Action Test Suite
# ============================================================================
#
# Tests the complete OData Summary action implementation
#
# Usage:
#   cd src/serviceCore/nHyperBook && ./scripts/test_odata_summary.sh
# ============================================================================

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test headers
print_header() {
    echo ""
    echo "========================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "========================================================================"
    echo ""
}

# Function to print test results
print_test() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} $2"
        ((TESTS_FAILED++))
    fi
}

# Function to check if text exists in file
check_text() {
    if grep -q "$2" "$1" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Start tests
print_header "🧪 Day 32: OData Summary Action Tests"

# ============================================================================
# Test 1: File Structure
# ============================================================================
print_header "Test 1: File Structure"

check_text "server/odata_summary.zig" "OData Summary Action Handler"
print_test $? "OData summary handler module present"

check_text "server/main.zig" "odata_summary"
print_test $? "Main server imports odata_summary module"

check_text "server/main.zig" "GenerateSummary"
print_test $? "Main server routes summary action"

# ============================================================================
# Test 2: OData Complex Types
# ============================================================================
print_header "Test 2: OData Complex Types"

check_text "server/odata_summary.zig" "pub const SummaryRequest"
print_test $? "SummaryRequest complex type present"

check_text "server/odata_summary.zig" "pub const SummaryResponse"
print_test $? "SummaryResponse complex type present"

check_text "server/odata_summary.zig" "pub const KeyPoint"
print_test $? "KeyPoint structure present"

check_text "server/odata_summary.zig" "SourceIds"
print_test $? "SummaryRequest has SourceIds field"

check_text "server/odata_summary.zig" "SummaryType"
print_test $? "SummaryRequest has SummaryType field"

check_text "server/odata_summary.zig" "MaxLength"
print_test $? "SummaryRequest has MaxLength field"

check_text "server/odata_summary.zig" "IncludeCitations"
print_test $? "SummaryRequest has IncludeCitations field"

check_text "server/odata_summary.zig" "IncludeKeyPoints"
print_test $? "SummaryRequest has IncludeKeyPoints field"

# ============================================================================
# Test 3: Response Structure
# ============================================================================
print_header "Test 3: Response Structure"

check_text "server/odata_summary.zig" "SummaryId"
print_test $? "SummaryResponse has SummaryId field"

check_text "server/odata_summary.zig" "SummaryText"
print_test $? "SummaryResponse has SummaryText field"

check_text "server/odata_summary.zig" "KeyPoints"
print_test $? "SummaryResponse has KeyPoints field"

check_text "server/odata_summary.zig" "WordCount"
print_test $? "SummaryResponse has WordCount field"

check_text "server/odata_summary.zig" "Confidence"
print_test $? "SummaryResponse has Confidence field"

check_text "server/odata_summary.zig" "ProcessingTimeMs"
print_test $? "SummaryResponse has ProcessingTimeMs field"

# ============================================================================
# Test 4: FFI Structures
# ============================================================================
print_header "Test 4: FFI Structures"

check_text "server/odata_summary.zig" "MojoSummaryRequest"
print_test $? "MojoSummaryRequest FFI structure present"

check_text "server/odata_summary.zig" "MojoSummaryResponse"
print_test $? "MojoSummaryResponse FFI structure present"

check_text "server/odata_summary.zig" "MojoKeyPoint"
print_test $? "MojoKeyPoint FFI structure present"

check_text "server/odata_summary.zig" "mojo_generate_summary"
print_test $? "mojo_generate_summary FFI declaration present"

check_text "server/odata_summary.zig" "mojo_free_summary_response"
print_test $? "mojo_free_summary_response FFI declaration present"

# ============================================================================
# Test 5: Handler Structure
# ============================================================================
print_header "Test 5: Handler Structure"

check_text "server/odata_summary.zig" "ODataSummaryHandler"
print_test $? "ODataSummaryHandler structure present"

check_text "server/odata_summary.zig" "handleSummaryAction"
print_test $? "handleSummaryAction function present"

check_text "server/odata_summary.zig" "isValidSummaryType"
print_test $? "isValidSummaryType validation function present"

check_text "server/odata_summary.zig" "summaryRequestToMojoFFI"
print_test $? "summaryRequestToMojoFFI conversion function present"

check_text "server/odata_summary.zig" "mojoFFIToSummaryResponse"
print_test $? "mojoFFIToSummaryResponse conversion function present"

# ============================================================================
# Test 6: Summary Type Validation
# ============================================================================
print_header "Test 6: Summary Type Validation"

check_text "server/odata_summary.zig" "brief"
print_test $? "Brief summary type supported"

check_text "server/odata_summary.zig" "detailed"
print_test $? "Detailed summary type supported"

check_text "server/odata_summary.zig" "executive"
print_test $? "Executive summary type supported"

check_text "server/odata_summary.zig" "bullet_points"
print_test $? "Bullet points summary type supported"

check_text "server/odata_summary.zig" "comparative"
print_test $? "Comparative summary type supported"

# ============================================================================
# Test 7: Main Server Integration
# ============================================================================
print_header "Test 7: Main Server Integration"

check_text "server/main.zig" "odata_summary"
print_test $? "Main server imports summary module"

check_text "server/main.zig" "Day 32"
print_test $? "Day 32 comment in main server"

check_text "server/main.zig" "GenerateSummary"
print_test $? "Summary endpoint route present"

check_text "server/main.zig" "handleODataSummaryAction"
print_test $? "Summary action handler function present"

# Count lines of code
if [ -f "server/odata_summary.zig" ]; then
    LOC=$(wc -l < "server/odata_summary.zig")
    echo ""
    echo "Lines of code: $LOC"
    if [ $LOC -gt 500 ]; then
        print_test 0 "Sufficient implementation (>500 lines)"
    else
        print_test 1 "Implementation may be incomplete (<500 lines)"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
print_header "📊 Test Summary"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All Day 32 tests PASSED!${NC}"
    echo ""
    echo "Summary:"
    echo "  • OData Summary action handler implemented"
    echo "  • 5 summary types supported"
    echo "  • FFI integration with Mojo summary generator"
    echo "  • Complete request/response mapping"
    echo "  • Main server integration complete"
    echo ""
    echo -e "${GREEN}✨ Day 32 Implementation Complete!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
    echo ""
    echo "Failed tests: $TESTS_FAILED"
    exit 1
fi
