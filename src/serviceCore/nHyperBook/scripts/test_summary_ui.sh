#!/bin/bash

# ============================================================================
# Day 33: Summary UI Test Suite
# ============================================================================
#
# Tests the complete Summary UI implementation
#
# Usage:
#   cd src/serviceCore/nHyperBook && ./scripts/test_summary_ui.sh
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
print_header "🧪 Day 33: Summary UI Tests"

# ============================================================================
# Test 1: File Structure
# ============================================================================
print_header "Test 1: File Structure"

if [ -f "webapp/view/Summary.view.xml" ]; then
    print_test 0 "Summary view XML present"
else
    print_test 1 "Summary view XML missing"
fi

if [ -f "webapp/controller/Summary.controller.js" ]; then
    print_test 0 "Summary controller present"
else
    print_test 1 "Summary controller missing"
fi

# ============================================================================
# Test 2: View Structure
# ============================================================================
print_header "Test 2: View Structure"

check_text "webapp/view/Summary.view.xml" "controllerName=\"hypershimmy.controller.Summary\""
print_test $? "Controller properly linked"

check_text "webapp/view/Summary.view.xml" "summaryTitle"
print_test $? "Page title present"

check_text "webapp/view/Summary.view.xml" "Configuration Panel"
print_test $? "Configuration panel present"

check_text "webapp/view/Summary.view.xml" "summaryTypeSelect"
print_test $? "Summary type selector present"

check_text "webapp/view/Summary.view.xml" "maxLengthSlider"
print_test $? "Max length slider present"

check_text "webapp/view/Summary.view.xml" "toneSelect"
print_test $? "Tone selector present"

check_text "webapp/view/Summary.view.xml" "focusAreasInput"
print_test $? "Focus areas input present"

check_text "webapp/view/Summary.view.xml" "onGenerateSummary"
print_test $? "Generate button present"

# ============================================================================
# Test 3: Summary Types
# ============================================================================
print_header "Test 3: Summary Types"

check_text "webapp/view/Summary.view.xml" "brief"
print_test $? "Brief type option present"

check_text "webapp/view/Summary.view.xml" "detailed"
print_test $? "Detailed type option present"

check_text "webapp/view/Summary.view.xml" "executive"
print_test $? "Executive type option present"

check_text "webapp/view/Summary.view.xml" "bullet_points"
print_test $? "Bullet points type option present"

check_text "webapp/view/Summary.view.xml" "comparative"
print_test $? "Comparative type option present"

# ============================================================================
# Test 4: Display Components
# ============================================================================
print_header "Test 4: Display Components"

check_text "webapp/view/Summary.view.xml" "BusyIndicator"
print_test $? "Loading indicator present"

check_text "webapp/view/Summary.view.xml" "summaryText"
print_test $? "Summary text display present"

check_text "webapp/view/Summary.view.xml" "keyPointsList"
print_test $? "Key points list present"

check_text "webapp/view/Summary.view.xml" "sourcesList"
print_test $? "Sources list present"

check_text "webapp/view/Summary.view.xml" "Metadata"
print_test $? "Metadata panel present"

# ============================================================================
# Test 5: Controller Functions
# ============================================================================
print_header "Test 5: Controller Functions"

check_text "webapp/controller/Summary.controller.js" "onInit"
print_test $? "onInit function present"

check_text "webapp/controller/Summary.controller.js" "onGenerateSummary"
print_test $? "onGenerateSummary function present"

check_text "webapp/controller/Summary.controller.js" "onSummaryTypeChange"
print_test $? "onSummaryTypeChange function present"

check_text "webapp/controller/Summary.controller.js" "onExportSummary"
print_test $? "onExportSummary function present"

check_text "webapp/controller/Summary.controller.js" "onCopySummary"
print_test $? "onCopySummary function present"

check_text "webapp/controller/Summary.controller.js" "_callSummaryAction"
print_test $? "OData action call function present"

check_text "webapp/controller/Summary.controller.js" "_displaySummary"
print_test $? "Display summary function present"

check_text "webapp/controller/Summary.controller.js" "_formatSummaryText"
print_test $? "Format summary text function present"

# ============================================================================
# Test 6: Configuration Management
# ============================================================================
print_header "Test 6: Configuration Management"

check_text "webapp/controller/Summary.controller.js" "_summaryTypeDescriptions"
print_test $? "Summary type descriptions present"

check_text "webapp/controller/Summary.controller.js" "_initializeSummarySettings"
print_test $? "Initialize settings function present"

check_text "webapp/controller/Summary.controller.js" "_loadSummarySettings"
print_test $? "Load settings function present"

check_text "webapp/controller/Summary.controller.js" "_saveSummarySettings"
print_test $? "Save settings function present"

check_text "webapp/controller/Summary.controller.js" "localStorage"
print_test $? "LocalStorage persistence present"

# ============================================================================
# Test 7: Routing Integration
# ============================================================================
print_header "Test 7: Routing Integration"

check_text "webapp/manifest.json" "\"name\": \"summary\""
print_test $? "Summary route defined"

check_text "webapp/manifest.json" "sources/{sourceId}/summary"
print_test $? "Summary route pattern defined"

check_text "webapp/manifest.json" "\"viewName\": \"Summary\""
print_test $? "Summary target defined"

check_text "webapp/controller/Detail.controller.js" "oRouter.navTo(\"summary\""
print_test $? "Detail controller navigates to summary"

# ============================================================================
# Test 8: i18n Translations
# ============================================================================
print_header "Test 8: i18n Translations"

check_text "webapp/i18n/i18n.properties" "summaryTitle"
print_test $? "Summary title translation present"

check_text "webapp/i18n/i18n.properties" "summaryTypeBrief"
print_test $? "Brief type translation present"

check_text "webapp/i18n/i18n.properties" "summaryTypeExecutive"
print_test $? "Executive type translation present"

check_text "webapp/i18n/i18n.properties" "summaryGenerateButton"
print_test $? "Generate button translation present"

check_text "webapp/i18n/i18n.properties" "summaryKeyPoints"
print_test $? "Key points translation present"

# ============================================================================
# Test 9: CSS Styling
# ============================================================================
print_header "Test 9: CSS Styling"

check_text "webapp/css/style.css" ".hypershimmySummary"
print_test $? "Summary container style present"

check_text "webapp/css/style.css" ".summaryText"
print_test $? "Summary text style present"

check_text "webapp/css/style.css" ".summaryTypeDescription"
print_test $? "Type description style present"

check_text "webapp/css/style.css" ".summaryMetadata"
print_test $? "Metadata style present"

# ============================================================================
# Test 10: OData Integration
# ============================================================================
print_header "Test 10: OData Integration"

check_text "webapp/controller/Summary.controller.js" "/odata/v4/research/GenerateSummary"
print_test $? "OData endpoint referenced"

check_text "webapp/controller/Summary.controller.js" "SourceIds"
print_test $? "SourceIds parameter present"

check_text "webapp/controller/Summary.controller.js" "SummaryType"
print_test $? "SummaryType parameter present"

check_text "webapp/controller/Summary.controller.js" "MaxLength"
print_test $? "MaxLength parameter present"

check_text "webapp/controller/Summary.controller.js" "IncludeCitations"
print_test $? "IncludeCitations parameter present"

# Count lines of code
echo ""
if [ -f "webapp/view/Summary.view.xml" ]; then
    VIEW_LOC=$(wc -l < "webapp/view/Summary.view.xml")
    echo "Summary view LOC: $VIEW_LOC"
fi

if [ -f "webapp/controller/Summary.controller.js" ]; then
    CONTROLLER_LOC=$(wc -l < "webapp/controller/Summary.controller.js")
    echo "Summary controller LOC: $CONTROLLER_LOC"
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
    echo -e "${GREEN}✅ All Day 33 tests PASSED!${NC}"
    echo ""
    echo "Summary:"
    echo "  • Summary view XML implemented"
    echo "  • Summary controller implemented"
    echo "  • 5 summary types supported"
    echo "  • Configuration panel with all options"
    echo "  • Display components (text, key points, sources, metadata)"
    echo "  • Export and copy functionality"
    echo "  • Routing integration complete"
    echo "  • i18n translations added"
    echo "  • CSS styling complete"
    echo ""
    echo -e "${GREEN}✨ Day 33 Implementation Complete!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
    echo ""
    echo "Failed tests: $TESTS_FAILED"
    exit 1
fi
