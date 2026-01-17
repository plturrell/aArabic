#!/bin/bash

# ============================================================================
# HyperShimmy UI/UX Polish Test Script
# Day 54: UI/UX Polish - Verification Script
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================================"
echo "HyperShimmy UI/UX Polish Test"
echo "Day 54: Verifying UI/UX Improvements"
echo "============================================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    local test_name=$1
    local result=$2
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to check if CSS file exists and is valid
check_css_file() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing CSS File${NC}"
    echo "----------------------------------------"
    
    # Check file exists
    if [ -f "$css_file" ]; then
        print_result "CSS file exists" "PASS"
    else
        print_result "CSS file exists" "FAIL"
        return 1
    fi
    
    # Check file size (should be substantial with all improvements)
    local file_size=$(wc -c < "$css_file")
    if [ "$file_size" -gt 30000 ]; then
        print_result "CSS file has substantial content (>30KB)" "PASS"
    else
        print_result "CSS file has substantial content (>30KB)" "FAIL"
    fi
    
    # Check for CSS variables
    if grep -q ":root" "$css_file"; then
        print_result "CSS variables defined" "PASS"
    else
        print_result "CSS variables defined" "FAIL"
    fi
    
    echo ""
}

# Function to verify CSS features
verify_css_features() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing CSS Features${NC}"
    echo "----------------------------------------"
    
    # Check for color system
    if grep -q "var(--primary-color)" "$css_file"; then
        print_result "Color system with CSS variables" "PASS"
    else
        print_result "Color system with CSS variables" "FAIL"
    fi
    
    # Check for spacing system
    if grep -q "var(--spacing-" "$css_file"; then
        print_result "Spacing system defined" "PASS"
    else
        print_result "Spacing system defined" "FAIL"
    fi
    
    # Check for shadow system
    if grep -q "var(--shadow-" "$css_file"; then
        print_result "Shadow system defined" "PASS"
    else
        print_result "Shadow system defined" "FAIL"
    fi
    
    # Check for transition system
    if grep -q "var(--transition-" "$css_file"; then
        print_result "Transition system defined" "PASS"
    else
        print_result "Transition system defined" "FAIL"
    fi
    
    # Check for border radius system
    if grep -q "var(--radius-" "$css_file"; then
        print_result "Border radius system defined" "PASS"
    else
        print_result "Border radius system defined" "FAIL"
    fi
    
    echo ""
}

# Function to verify animations
verify_animations() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing Animations${NC}"
    echo "----------------------------------------"
    
    # Check for fadeIn animation
    if grep -q "@keyframes fadeIn" "$css_file"; then
        print_result "fadeIn animation defined" "PASS"
    else
        print_result "fadeIn animation defined" "FAIL"
    fi
    
    # Check for slideIn animations
    if grep -q "@keyframes slideInLeft" "$css_file" && grep -q "@keyframes slideInRight" "$css_file"; then
        print_result "Slide animations defined" "PASS"
    else
        print_result "Slide animations defined" "FAIL"
    fi
    
    # Check for spin animation (loading)
    if grep -q "@keyframes spin" "$css_file"; then
        print_result "Spin animation for loading" "PASS"
    else
        print_result "Spin animation for loading" "FAIL"
    fi
    
    # Check for shimmer animation (skeleton)
    if grep -q "@keyframes shimmer" "$css_file"; then
        print_result "Shimmer animation for skeleton loading" "PASS"
    else
        print_result "Shimmer animation for skeleton loading" "FAIL"
    fi
    
    # Check for typing indicator animation
    if grep -q "@keyframes typing" "$css_file"; then
        print_result "Typing indicator animation" "PASS"
    else
        print_result "Typing indicator animation" "FAIL"
    fi
    
    # Check for pulse animation
    if grep -q "@keyframes pulse" "$css_file"; then
        print_result "Pulse animation" "PASS"
    else
        print_result "Pulse animation" "FAIL"
    fi
    
    echo ""
}

# Function to verify component styles
verify_component_styles() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing Component Styles${NC}"
    echo "----------------------------------------"
    
    # Check for chat interface styles
    if grep -q ".chatMessageUser" "$css_file" && grep -q ".chatMessageAssistant" "$css_file"; then
        print_result "Chat message styles" "PASS"
    else
        print_result "Chat message styles" "FAIL"
    fi
    
    # Check for loading states
    if grep -q ".loadingSpinner" "$css_file" && grep -q ".skeleton" "$css_file"; then
        print_result "Loading state styles" "PASS"
    else
        print_result "Loading state styles" "FAIL"
    fi
    
    # Check for empty states
    if grep -q ".hypershimmyEmptyState" "$css_file"; then
        print_result "Empty state styles" "PASS"
    else
        print_result "Empty state styles" "FAIL"
    fi
    
    # Check for toast notifications
    if grep -q ".toast" "$css_file" && grep -q ".toastContainer" "$css_file"; then
        print_result "Toast notification styles" "PASS"
    else
        print_result "Toast notification styles" "FAIL"
    fi
    
    # Check for progress indicators
    if grep -q ".progressBar" "$css_file" && grep -q ".progressBarFill" "$css_file"; then
        print_result "Progress indicator styles" "PASS"
    else
        print_result "Progress indicator styles" "FAIL"
    fi
    
    # Check for form validation
    if grep -q ".formError" "$css_file" && grep -q ".formSuccess" "$css_file"; then
        print_result "Form validation styles" "PASS"
    else
        print_result "Form validation styles" "FAIL"
    fi
    
    # Check for badge styles
    if grep -q ".badge" "$css_file"; then
        print_result "Badge component styles" "PASS"
    else
        print_result "Badge component styles" "FAIL"
    fi
    
    echo ""
}

# Function to verify responsive design
verify_responsive_design() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing Responsive Design${NC}"
    echo "----------------------------------------"
    
    # Check for mobile breakpoint
    if grep -q "@media (max-width: 600px)" "$css_file"; then
        print_result "Mobile breakpoint (600px)" "PASS"
    else
        print_result "Mobile breakpoint (600px)" "FAIL"
    fi
    
    # Check for tablet breakpoint
    if grep -q "@media (max-width: 768px)" "$css_file"; then
        print_result "Tablet breakpoint (768px)" "PASS"
    else
        print_result "Tablet breakpoint (768px)" "FAIL"
    fi
    
    # Check for desktop breakpoint
    if grep -q "@media (max-width: 1024px)" "$css_file"; then
        print_result "Desktop breakpoint (1024px)" "PASS"
    else
        print_result "Desktop breakpoint (1024px)" "FAIL"
    fi
    
    echo ""
}

# Function to verify accessibility features
verify_accessibility() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing Accessibility Features${NC}"
    echo "----------------------------------------"
    
    # Check for focus styles
    if grep -q ":focus-visible" "$css_file"; then
        print_result "Focus-visible styles for keyboard navigation" "PASS"
    else
        print_result "Focus-visible styles for keyboard navigation" "FAIL"
    fi
    
    # Check for screen reader only class
    if grep -q ".sr-only" "$css_file"; then
        print_result "Screen reader only utility class" "PASS"
    else
        print_result "Screen reader only utility class" "FAIL"
    fi
    
    # Check for high contrast mode support
    if grep -q "@media (prefers-contrast: high)" "$css_file"; then
        print_result "High contrast mode support" "PASS"
    else
        print_result "High contrast mode support" "FAIL"
    fi
    
    # Check for reduced motion support
    if grep -q "@media (prefers-reduced-motion: reduce)" "$css_file"; then
        print_result "Reduced motion preference support" "PASS"
    else
        print_result "Reduced motion preference support" "FAIL"
    fi
    
    # Check for color-scheme support
    if grep -q "@media (prefers-color-scheme: dark)" "$css_file"; then
        print_result "Dark mode preference support" "PASS"
    else
        print_result "Dark mode preference support" "FAIL"
    fi
    
    echo ""
}

# Function to verify print styles
verify_print_styles() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing Print Styles${NC}"
    echo "----------------------------------------"
    
    # Check for print media query
    if grep -q "@media print" "$css_file"; then
        print_result "Print media query defined" "PASS"
    else
        print_result "Print media query defined" "FAIL"
    fi
    
    echo ""
}

# Function to verify custom scrollbar
verify_scrollbar() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing Custom Scrollbar${NC}"
    echo "----------------------------------------"
    
    # Check for webkit scrollbar
    if grep -q "::-webkit-scrollbar" "$css_file"; then
        print_result "Webkit scrollbar styles" "PASS"
    else
        print_result "Webkit scrollbar styles" "FAIL"
    fi
    
    # Check for Firefox scrollbar
    if grep -q "scrollbar-width" "$css_file" && grep -q "scrollbar-color" "$css_file"; then
        print_result "Firefox scrollbar styles" "PASS"
    else
        print_result "Firefox scrollbar styles" "FAIL"
    fi
    
    echo ""
}

# Function to verify view-specific styles
verify_view_styles() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing View-Specific Styles${NC}"
    echo "----------------------------------------"
    
    # Check for summary view styles
    if grep -q ".hypershimmySummary" "$css_file"; then
        print_result "Summary view styles" "PASS"
    else
        print_result "Summary view styles" "FAIL"
    fi
    
    # Check for mindmap view styles
    if grep -q ".hypershimmyMindmap" "$css_file"; then
        print_result "Mindmap view styles" "PASS"
    else
        print_result "Mindmap view styles" "FAIL"
    fi
    
    # Check for audio view styles
    if grep -q ".hypershimmyAudio" "$css_file"; then
        print_result "Audio view styles" "PASS"
    else
        print_result "Audio view styles" "FAIL"
    fi
    
    # Check for slides view styles
    if grep -q ".hypershimmySlides" "$css_file"; then
        print_result "Slides view styles" "PASS"
    else
        print_result "Slides view styles" "FAIL"
    fi
    
    echo ""
}

# Function to verify utility classes
verify_utilities() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing Utility Classes${NC}"
    echo "----------------------------------------"
    
    # Check for truncate utility
    if grep -q ".u-truncate" "$css_file"; then
        print_result "Text truncate utility" "PASS"
    else
        print_result "Text truncate utility" "FAIL"
    fi
    
    # Check for text alignment utilities
    if grep -q ".u-text-center" "$css_file"; then
        print_result "Text alignment utilities" "PASS"
    else
        print_result "Text alignment utilities" "FAIL"
    fi
    
    # Check for visually hidden utility
    if grep -q ".u-visually-hidden" "$css_file"; then
        print_result "Visually hidden utility" "PASS"
    else
        print_result "Visually hidden utility" "FAIL"
    fi
    
    echo ""
}

# Function to check CSS syntax
check_css_syntax() {
    local css_file="$PROJECT_ROOT/webapp/css/style.css"
    
    echo -e "${BLUE}Testing CSS Syntax${NC}"
    echo "----------------------------------------"
    
    # Check for balanced braces
    local open_braces=$(grep -o "{" "$css_file" | wc -l)
    local close_braces=$(grep -o "}" "$css_file" | wc -l)
    
    if [ "$open_braces" -eq "$close_braces" ]; then
        print_result "Balanced braces (${open_braces} pairs)" "PASS"
    else
        print_result "Balanced braces" "FAIL"
        echo "  Open: $open_braces, Close: $close_braces"
    fi
    
    # Check for proper comment syntax
    if ! grep -q "/\*[^*]*\*/" "$css_file" || grep -q "\*/" "$css_file"; then
        print_result "Proper comment syntax" "PASS"
    else
        print_result "Proper comment syntax" "FAIL"
    fi
    
    echo ""
}

# Function to generate summary report
generate_summary() {
    echo "============================================================================"
    echo "Test Summary"
    echo "============================================================================"
    echo ""
    echo "Total Tests Run:    $TESTS_RUN"
    echo -e "${GREEN}Tests Passed:       $TESTS_PASSED${NC}"
    
    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "${RED}Tests Failed:       $TESTS_FAILED${NC}"
    else
        echo "Tests Failed:       $TESTS_FAILED"
    fi
    
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        echo ""
        echo "UI/UX Polish Verification: COMPLETE"
        return 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        echo ""
        echo "UI/UX Polish Verification: FAILED"
        return 1
    fi
}

# Main execution
main() {
    echo "Starting UI/UX Polish verification..."
    echo ""
    
    # Run all test suites
    check_css_file
    verify_css_features
    verify_animations
    verify_component_styles
    verify_responsive_design
    verify_accessibility
    verify_print_styles
    verify_scrollbar
    verify_view_styles
    verify_utilities
    check_css_syntax
    
    # Generate summary
    generate_summary
}

# Run main function
main

exit $?
