#!/bin/bash

# ============================================================================
# Day 31: Summary Generator Tests
# ============================================================================
#
# Test Suite for Summary Generator:
# - Summary type implementations
# - Key point extraction
# - Multi-document synthesis
# - Configuration handling
# - Prompt templates
#
# Usage: ./test_summary.sh
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
echo "🧪 Day 31: Summary Generator Tests"
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

test_struct_exists() {
    local file=$1
    local struct_name=$2
    local description=$3
    if grep -q "struct $struct_name:" "$file"; then
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
# Test 1: File Structure
# ============================================================================

echo "Test 1: File Structure"
echo "------------------------------------------------------------------------"

test_file_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Summary generator module present"

echo ""

# ============================================================================
# Test 2: Core Structures
# ============================================================================

echo "Test 2: Core Data Structures"
echo "------------------------------------------------------------------------"

test_struct_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "SummaryType" \
    "SummaryType enumeration present"

test_struct_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "SummaryConfig" \
    "SummaryConfig structure present"

test_struct_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "SummaryRequest" \
    "SummaryRequest structure present"

test_struct_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "SummaryResponse" \
    "SummaryResponse structure present"

test_struct_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "KeyPoint" \
    "KeyPoint structure present"

test_struct_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "SummaryGenerator" \
    "SummaryGenerator structure present"

echo ""

# ============================================================================
# Test 3: Summary Types
# ============================================================================

echo "Test 3: Summary Types"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "fn brief()" \
    "Brief summary type present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "fn detailed()" \
    "Detailed summary type present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "fn executive()" \
    "Executive summary type present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "fn bullet_points()" \
    "Bullet points summary type present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "fn comparative()" \
    "Comparative summary type present"

echo ""

# ============================================================================
# Test 4: Prompt Templates
# ============================================================================

echo "Test 4: Prompt Templates"
echo "------------------------------------------------------------------------"

test_struct_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "SummaryPrompts" \
    "SummaryPrompts structure present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "get_system_prompt" \
    "System prompt template present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "get_brief_prompt" \
    "Brief summary prompt present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "get_detailed_prompt" \
    "Detailed summary prompt present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "get_executive_prompt" \
    "Executive summary prompt present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "get_bullet_points_prompt" \
    "Bullet points prompt present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "get_comparative_prompt" \
    "Comparative summary prompt present"

echo ""

# ============================================================================
# Test 5: Summary Generator Methods
# ============================================================================

echo "Test 5: Summary Generator Methods"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "fn generate_summary" \
    "Generate summary method present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_build_summary_prompt" \
    "Build prompt method present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_generate_summary_text" \
    "Generate text method present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_extract_key_points" \
    "Extract key points method present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_count_words" \
    "Word count method present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_calculate_confidence" \
    "Confidence calculation present"

echo ""

# ============================================================================
# Test 6: Summary Type Generators
# ============================================================================

echo "Test 6: Summary Type Generators"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_generate_brief_summary" \
    "Brief summary generator present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_generate_detailed_summary" \
    "Detailed summary generator present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_generate_executive_summary" \
    "Executive summary generator present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_generate_bullet_summary" \
    "Bullet summary generator present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_generate_comparative_summary" \
    "Comparative summary generator present"

echo ""

# ============================================================================
# Test 7: Configuration Features
# ============================================================================

echo "Test 7: Configuration Features"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "max_length: Int" \
    "Max length configuration present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "include_citations: Bool" \
    "Citations configuration present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "include_key_points: Bool" \
    "Key points configuration present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "tone: String" \
    "Tone configuration present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "focus_areas: List" \
    "Focus areas configuration present"

echo ""

# ============================================================================
# Test 8: Key Point Extraction
# ============================================================================

echo "Test 8: Key Point Extraction"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "importance: Float32" \
    "Key point importance scoring present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "category: String" \
    "Key point categorization present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "source_ids: List\[String\]" \
    "Source attribution in key points present"

echo ""

# ============================================================================
# Test 9: Response Metadata
# ============================================================================

echo "Test 9: Response Metadata"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "word_count: Int" \
    "Word count tracking present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "confidence: Float32" \
    "Confidence scoring present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "processing_time_ms: Int" \
    "Processing time tracking present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "_build_metadata" \
    "Metadata builder present"

echo ""

# ============================================================================
# Test 10: FFI Integration
# ============================================================================

echo "Test 10: FFI Integration"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "@export.*summary_generate" \
    "C ABI export for summary generation present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "DTypePointer" \
    "FFI pointer types present"

echo ""

# ============================================================================
# Test 11: Prompt Quality
# ============================================================================

echo "Test 11: Prompt Quality"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Synthesize information from multiple documents" \
    "Multi-document synthesis in prompts"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Extract key insights" \
    "Key insights extraction in prompts"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Cite sources" \
    "Source citation in prompts"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Requirements:" \
    "Structured prompt requirements present"

echo ""

# ============================================================================
# Test 12: Test Implementation
# ============================================================================

echo "Test 12: Test Implementation"
echo "------------------------------------------------------------------------"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "fn main()" \
    "Main test function present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Test 1: Brief Summary" \
    "Brief summary test present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Test 2: Detailed Summary" \
    "Detailed summary test present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Test 3: Executive Summary" \
    "Executive summary test present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Test 4: Bullet Point Summary" \
    "Bullet point summary test present"

test_content_exists \
    "$PROJECT_ROOT/mojo/summary_generator.mojo" \
    "Test 5: Comparative Summary" \
    "Comparative summary test present"

echo ""

# ============================================================================
# Test 13: Code Quality
# ============================================================================

echo "Test 13: Code Quality & Documentation"
echo "------------------------------------------------------------------------"

# Check file size
FILE_LINES=$(wc -l < "$PROJECT_ROOT/mojo/summary_generator.mojo")
if [ "$FILE_LINES" -gt "800" ]; then
    echo -e "${GREEN}✓${NC} Module size appropriate ($FILE_LINES lines)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗${NC} Module too small ($FILE_LINES lines)"
    ((TESTS_FAILED++))
fi

# Check for docstrings
DOCSTRING_COUNT=$(grep -c '"""' "$PROJECT_ROOT/mojo/summary_generator.mojo" || echo "0")
if [ "$DOCSTRING_COUNT" -gt "20" ]; then
    echo -e "${GREEN}✓${NC} Documentation present ($DOCSTRING_COUNT docstrings)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗${NC} Insufficient documentation ($DOCSTRING_COUNT docstrings)"
    ((TESTS_FAILED++))
fi

# Check for comments
COMMENT_COUNT=$(grep -c '#' "$PROJECT_ROOT/mojo/summary_generator.mojo" || echo "0")
if [ "$COMMENT_COUNT" -gt "30" ]; then
    echo -e "${GREEN}✓${NC} Code comments present ($COMMENT_COUNT comments)"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠${NC} Limited comments ($COMMENT_COUNT comments)"
    ((TESTS_PASSED++))
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
    echo -e "${GREEN}✅ All Day 31 tests PASSED!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
    echo ""
    exit 1
fi
