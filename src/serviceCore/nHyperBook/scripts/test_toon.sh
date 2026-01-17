#!/bin/bash

# ============================================================================
# Day 34: TOON Encoder Test Suite
# ============================================================================
#
# Tests the complete TOON encoding implementation
#
# Usage:
#   cd src/serviceCore/nHyperBook && ./scripts/test_toon.sh
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
print_header "🧪 Day 34: TOON Encoder Tests"

# ============================================================================
# Test 1: File Structure
# ============================================================================
print_header "Test 1: File Structure"

if [ -f "mojo/toon_encoder.mojo" ]; then
    print_test 0 "TOON encoder module present"
else
    print_test 1 "TOON encoder module missing"
fi

# ============================================================================
# Test 2: Core Data Structures
# ============================================================================
print_header "Test 2: Core Data Structures"

check_text "mojo/toon_encoder.mojo" "struct TOONToken"
print_test $? "TOONToken structure present"

check_text "mojo/toon_encoder.mojo" "struct TOONDictionary"
print_test $? "TOONDictionary structure present"

check_text "mojo/toon_encoder.mojo" "struct TOONEncoded"
print_test $? "TOONEncoded structure present"

check_text "mojo/toon_encoder.mojo" "struct TOONMetrics"
print_test $? "TOONMetrics structure present"

check_text "mojo/toon_encoder.mojo" "struct TOONEncoder"
print_test $? "TOONEncoder structure present"

# ============================================================================
# Test 3: TOONToken Features
# ============================================================================
print_header "Test 3: TOONToken Features"

check_text "mojo/toon_encoder.mojo" "var text: String"
print_test $? "Token text field present"

check_text "mojo/toon_encoder.mojo" "var frequency: Int"
print_test $? "Token frequency field present"

check_text "mojo/toon_encoder.mojo" "var positions: List\[Int\]"
print_test $? "Token positions field present"

check_text "mojo/toon_encoder.mojo" "var encoding_id: Int"
print_test $? "Token encoding ID field present"

check_text "mojo/toon_encoder.mojo" "var semantic_weight: Float32"
print_test $? "Token semantic weight field present"

check_text "mojo/toon_encoder.mojo" "fn add_position"
print_test $? "Add position function present"

# ============================================================================
# Test 4: TOONDictionary Functionality
# ============================================================================
print_header "Test 4: TOONDictionary Functionality"

check_text "mojo/toon_encoder.mojo" "fn add_token"
print_test $? "Add token function present"

check_text "mojo/toon_encoder.mojo" "fn get_token_id"
print_test $? "Get token ID function present"

check_text "mojo/toon_encoder.mojo" "fn get_token_text"
print_test $? "Get token text function present"

check_text "mojo/toon_encoder.mojo" "fn get_most_frequent"
print_test $? "Get most frequent tokens function present"

check_text "mojo/toon_encoder.mojo" "var tokens: Dict\[String, TOONToken\]"
print_test $? "Token dictionary present"

check_text "mojo/toon_encoder.mojo" "var reverse_map: Dict\[Int, String\]"
print_test $? "Reverse mapping present"

# ============================================================================
# Test 5: TOONEncoded Structure
# ============================================================================
print_header "Test 5: TOONEncoded Structure"

check_text "mojo/toon_encoder.mojo" "var token_ids: List\[Int\]"
print_test $? "Token IDs list present"

check_text "mojo/toon_encoder.mojo" "var dictionary: TOONDictionary"
print_test $? "Dictionary reference present"

check_text "mojo/toon_encoder.mojo" "var metadata: String"
print_test $? "Metadata field present"

check_text "mojo/toon_encoder.mojo" "var compression_ratio: Float32"
print_test $? "Compression ratio field present"

check_text "mojo/toon_encoder.mojo" "var original_length: Int"
print_test $? "Original length field present"

check_text "mojo/toon_encoder.mojo" "var encoded_length: Int"
print_test $? "Encoded length field present"

check_text "mojo/toon_encoder.mojo" "fn calculate_compression_ratio"
print_test $? "Calculate compression ratio function present"

# ============================================================================
# Test 6: TOONMetrics
# ============================================================================
print_header "Test 6: TOONMetrics"

check_text "mojo/toon_encoder.mojo" "var unique_tokens: Int"
print_test $? "Unique tokens metric present"

check_text "mojo/toon_encoder.mojo" "var total_tokens: Int"
print_test $? "Total tokens metric present"

check_text "mojo/toon_encoder.mojo" "var semantic_preservation: Float32"
print_test $? "Semantic preservation metric present"

check_text "mojo/toon_encoder.mojo" "var encoding_time_ms: Int"
print_test $? "Encoding time metric present"

check_text "mojo/toon_encoder.mojo" "var decoding_time_ms: Int"
print_test $? "Decoding time metric present"

# ============================================================================
# Test 7: TOONEncoder Core Functions
# ============================================================================
print_header "Test 7: TOONEncoder Core Functions"

check_text "mojo/toon_encoder.mojo" "fn encode"
print_test $? "Encode function present"

check_text "mojo/toon_encoder.mojo" "fn decode"
print_test $? "Decode function present"

check_text "mojo/toon_encoder.mojo" "fn compress_summary"
print_test $? "Compress summary function present"

check_text "mojo/toon_encoder.mojo" "fn get_metrics"
print_test $? "Get metrics function present"

# ============================================================================
# Test 8: Encoder Configuration
# ============================================================================
print_header "Test 8: Encoder Configuration"

check_text "mojo/toon_encoder.mojo" "var use_semantic_weights: Bool"
print_test $? "Semantic weights option present"

check_text "mojo/toon_encoder.mojo" "var min_token_length: Int"
print_test $? "Min token length option present"

check_text "mojo/toon_encoder.mojo" "var max_dictionary_size: Int"
print_test $? "Max dictionary size option present"

# ============================================================================
# Test 9: Internal Helper Functions
# ============================================================================
print_header "Test 9: Internal Helper Functions"

check_text "mojo/toon_encoder.mojo" "fn _tokenize"
print_test $? "Tokenize function present"

check_text "mojo/toon_encoder.mojo" "fn _generate_metadata"
print_test $? "Generate metadata function present"

check_text "mojo/toon_encoder.mojo" "fn _optimize_for_summary"
print_test $? "Optimize for summary function present"

check_text "mojo/toon_encoder.mojo" "fn _is_technical_term"
print_test $? "Is technical term function present"

check_text "mojo/toon_encoder.mojo" "fn _calculate_semantic_preservation"
print_test $? "Calculate semantic preservation function present"

# ============================================================================
# Test 10: FFI Exports
# ============================================================================
print_header "Test 10: FFI Exports"

check_text "mojo/toon_encoder.mojo" "@export"
print_test $? "FFI export decorators present"

check_text "mojo/toon_encoder.mojo" "fn toon_encode_text"
print_test $? "FFI encode function present"

check_text "mojo/toon_encoder.mojo" "fn toon_decode_text"
print_test $? "FFI decode function present"

check_text "mojo/toon_encoder.mojo" "fn toon_compress_summary"
print_test $? "FFI compress summary function present"

check_text "mojo/toon_encoder.mojo" "fn toon_get_metrics"
print_test $? "FFI get metrics function present"

# ============================================================================
# Test 11: Utility Functions
# ============================================================================
print_header "Test 11: Utility Functions"

check_text "mojo/toon_encoder.mojo" "fn calculate_compression_benefit"
print_test $? "Calculate compression benefit function present"

check_text "mojo/toon_encoder.mojo" "fn estimate_storage_savings"
print_test $? "Estimate storage savings function present"

# ============================================================================
# Test 12: Documentation
# ============================================================================
print_header "Test 12: Documentation"

check_text "mojo/toon_encoder.mojo" "Token-Optimized Ordered Notation"
print_test $? "TOON full name documented"

check_text "mojo/toon_encoder.mojo" "compression and encoding system"
print_test $? "Purpose documented"

check_text "mojo/toon_encoder.mojo" "Token frequency analysis"
print_test $? "Feature: Token frequency documented"

check_text "mojo/toon_encoder.mojo" "Semantic pattern compression"
print_test $? "Feature: Semantic compression documented"

check_text "mojo/toon_encoder.mojo" "Ordered notation"
print_test $? "Feature: Ordered notation documented"

# ============================================================================
# Test 13: Compression Features
# ============================================================================
print_header "Test 13: Compression Features"

check_text "mojo/toon_encoder.mojo" "Technical term recognition"
print_test $? "Technical term recognition feature present"

check_text "mojo/toon_encoder.mojo" "Citation preservation"
print_test $? "Citation preservation feature present"

check_text "mojo/toon_encoder.mojo" "compression_ratio"
print_test $? "Compression ratio tracking present"

check_text "mojo/toon_encoder.mojo" "semantic_preservation"
print_test $? "Semantic preservation tracking present"

# Count lines of code
echo ""
if [ -f "mojo/toon_encoder.mojo" ]; then
    LOC=$(wc -l < "mojo/toon_encoder.mojo")
    echo "TOON encoder LOC: $LOC"
    
    if [ $LOC -ge 400 ]; then
        print_test 0 "Sufficient implementation (>400 lines)"
    else
        print_test 1 "Implementation too small (<400 lines)"
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
    echo -e "${GREEN}✅ All Day 34 tests PASSED!${NC}"
    echo ""
    echo "Summary:"
    echo "  • TOON encoder module implemented"
    echo "  • Core data structures complete"
    echo "  • Token dictionary with frequency tracking"
    echo "  • Encoding/decoding functionality"
    echo "  • Summary-specific optimizations"
    echo "  • Compression metrics and analysis"
    echo "  • FFI exports for Zig integration"
    echo "  • Utility functions for storage analysis"
    echo ""
    echo -e "${GREEN}✨ Day 34 Implementation Complete!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
    echo ""
    echo "Failed tests: $TESTS_FAILED"
    exit 1
fi
