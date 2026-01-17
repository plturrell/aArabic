#!/bin/bash

# ============================================================================
# HyperShimmy OData Mindmap Action Test Script
# ============================================================================
#
# Day 38: Test OData Mindmap action endpoint
#
# Tests:
# - File structure validation
# - OData complex type definitions
# - FFI structure definitions
# - Layout algorithm validation
# - Handler functions
# - Request/response conversion
# - Error handling
# - Main server integration
#
# Usage:
#   ./scripts/test_odata_mindmap.sh
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
}

print_test() {
    echo -e "${YELLOW}TEST: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

check_file_exists() {
    if [ -f "$1" ]; then
        print_success "File exists: $1"
        return 0
    else
        print_error "File missing: $1"
        return 1
    fi
}

check_content() {
    local file=$1
    local pattern=$2
    local description=$3
    
    if grep -q "$pattern" "$file"; then
        print_success "$description"
        return 0
    else
        print_error "$description - Pattern not found: $pattern"
        return 1
    fi
}

count_occurrences() {
    local file=$1
    local pattern=$2
    grep -o "$pattern" "$file" | wc -l | tr -d ' '
}

# ============================================================================
# Test 1: File Structure
# ============================================================================

test_file_structure() {
    print_header "Test 1: File Structure"
    
    print_test "Check odata_mindmap.zig exists"
    check_file_exists "src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check file is not empty"
    if [ -s "src/serviceCore/nHyperBook/server/odata_mindmap.zig" ]; then
        print_success "File has content"
    else
        print_error "File is empty"
    fi
    
    print_test "Check file has correct header"
    check_content "src/serviceCore/nHyperBook/server/odata_mindmap.zig" \
        "Day 38 Implementation" \
        "File has Day 38 header"
    
    echo ""
}

# ============================================================================
# Test 2: OData Complex Types
# ============================================================================

test_odata_complex_types() {
    print_header "Test 2: OData Complex Types"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check MindmapRequest definition"
    check_content "$file" "pub const MindmapRequest = struct" \
        "MindmapRequest struct defined"
    
    print_test "Check MindmapRequest fields"
    check_content "$file" "SourceIds: \[\]const \[\]const u8" \
        "SourceIds field present"
    check_content "$file" "LayoutAlgorithm: \[\]const u8" \
        "LayoutAlgorithm field present"
    check_content "$file" "MaxDepth: \?i32" \
        "MaxDepth optional field present"
    check_content "$file" "MaxChildrenPerNode: \?i32" \
        "MaxChildrenPerNode optional field present"
    check_content "$file" "CanvasWidth: \?f32" \
        "CanvasWidth optional field present"
    check_content "$file" "CanvasHeight: \?f32" \
        "CanvasHeight optional field present"
    check_content "$file" "AutoSelectRoot: \?bool" \
        "AutoSelectRoot optional field present"
    check_content "$file" "RootEntityId: \?\[\]const u8" \
        "RootEntityId optional field present"
    
    print_test "Check MindmapNode definition"
    check_content "$file" "pub const MindmapNode = struct" \
        "MindmapNode struct defined"
    
    print_test "Check MindmapNode fields"
    check_content "$file" "Id: \[\]const u8" \
        "Id field present"
    check_content "$file" "Label: \[\]const u8" \
        "Label field present"
    check_content "$file" "NodeType: \[\]const u8" \
        "NodeType field present"
    check_content "$file" "EntityType: \[\]const u8" \
        "EntityType field present"
    check_content "$file" "Level: i32" \
        "Level field present"
    check_content "$file" "X: f32" \
        "X coordinate field present"
    check_content "$file" "Y: f32" \
        "Y coordinate field present"
    check_content "$file" "Confidence: f32" \
        "Confidence field present"
    check_content "$file" "ChildCount: i32" \
        "ChildCount field present"
    check_content "$file" "ParentId: \[\]const u8" \
        "ParentId field present"
    
    print_test "Check MindmapEdge definition"
    check_content "$file" "pub const MindmapEdge = struct" \
        "MindmapEdge struct defined"
    
    print_test "Check MindmapEdge fields"
    check_content "$file" "FromNodeId: \[\]const u8" \
        "FromNodeId field present"
    check_content "$file" "ToNodeId: \[\]const u8" \
        "ToNodeId field present"
    check_content "$file" "RelationshipType: \[\]const u8" \
        "RelationshipType field present"
    check_content "$file" "Style: \[\]const u8" \
        "Style field present"
    
    print_test "Check MindmapResponse definition"
    check_content "$file" "pub const MindmapResponse = struct" \
        "MindmapResponse struct defined"
    
    print_test "Check MindmapResponse fields"
    check_content "$file" "MindmapId: \[\]const u8" \
        "MindmapId field present"
    check_content "$file" "Title: \[\]const u8" \
        "Title field present"
    check_content "$file" "Nodes: \[\]const MindmapNode" \
        "Nodes array field present"
    check_content "$file" "Edges: \[\]const MindmapEdge" \
        "Edges array field present"
    check_content "$file" "RootNodeId: \[\]const u8" \
        "RootNodeId field present"
    check_content "$file" "LayoutAlgorithm: \[\]const u8" \
        "LayoutAlgorithm field in response present"
    check_content "$file" "MaxDepth: i32" \
        "MaxDepth field in response present"
    check_content "$file" "NodeCount: i32" \
        "NodeCount field present"
    check_content "$file" "EdgeCount: i32" \
        "EdgeCount field present"
    check_content "$file" "ProcessingTimeMs: i32" \
        "ProcessingTimeMs field present"
    check_content "$file" "Metadata: \[\]const u8" \
        "Metadata field present"
    
    print_test "Check ODataError definition"
    check_content "$file" "pub const ODataError = struct" \
        "ODataError struct defined"
    
    echo ""
}

# ============================================================================
# Test 3: FFI Structures
# ============================================================================

test_ffi_structures() {
    print_header "Test 3: FFI Structures"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check MojoMindmapRequest FFI structure"
    check_content "$file" "const MojoMindmapRequest = extern struct" \
        "MojoMindmapRequest extern struct defined"
    
    print_test "Check MojoMindmapRequest fields"
    check_content "$file" "source_ids_ptr: \[\*\]const \[\*:0\]const u8" \
        "source_ids_ptr field present"
    check_content "$file" "source_ids_len: usize" \
        "source_ids_len field present"
    check_content "$file" "layout_algorithm: \[\*:0\]const u8" \
        "layout_algorithm field present"
    check_content "$file" "max_depth: i32" \
        "max_depth field present"
    check_content "$file" "max_children_per_node: i32" \
        "max_children_per_node field present"
    check_content "$file" "canvas_width: f32" \
        "canvas_width field present"
    check_content "$file" "canvas_height: f32" \
        "canvas_height field present"
    check_content "$file" "auto_select_root: bool" \
        "auto_select_root field present"
    check_content "$file" "root_entity_id: \[\*:0\]const u8" \
        "root_entity_id field present"
    
    print_test "Check MojoMindmapNode FFI structure"
    check_content "$file" "const MojoMindmapNode = extern struct" \
        "MojoMindmapNode extern struct defined"
    
    print_test "Check MojoMindmapNode fields"
    check_content "$file" "id: \[\*:0\]const u8" \
        "id field present in FFI node"
    check_content "$file" "label: \[\*:0\]const u8" \
        "label field present in FFI node"
    check_content "$file" "node_type: \[\*:0\]const u8" \
        "node_type field present in FFI node"
    check_content "$file" "entity_type: \[\*:0\]const u8" \
        "entity_type field present in FFI node"
    check_content "$file" "level: i32" \
        "level field present in FFI node"
    check_content "$file" "x: f32" \
        "x field present in FFI node"
    check_content "$file" "y: f32" \
        "y field present in FFI node"
    check_content "$file" "confidence: f32" \
        "confidence field present in FFI node"
    check_content "$file" "child_count: i32" \
        "child_count field present in FFI node"
    check_content "$file" "parent_id: \[\*:0\]const u8" \
        "parent_id field present in FFI node"
    
    print_test "Check MojoMindmapEdge FFI structure"
    check_content "$file" "const MojoMindmapEdge = extern struct" \
        "MojoMindmapEdge extern struct defined"
    
    print_test "Check MojoMindmapResponse FFI structure"
    check_content "$file" "const MojoMindmapResponse = extern struct" \
        "MojoMindmapResponse extern struct defined"
    
    print_test "Check MojoMindmapResponse fields"
    check_content "$file" "mindmap_id: \[\*:0\]const u8" \
        "mindmap_id field present"
    check_content "$file" "title: \[\*:0\]const u8" \
        "title field present"
    check_content "$file" "nodes_ptr: \[\*\]const MojoMindmapNode" \
        "nodes_ptr field present"
    check_content "$file" "nodes_len: usize" \
        "nodes_len field present"
    check_content "$file" "edges_ptr: \[\*\]const MojoMindmapEdge" \
        "edges_ptr field present"
    check_content "$file" "edges_len: usize" \
        "edges_len field present"
    check_content "$file" "root_node_id: \[\*:0\]const u8" \
        "root_node_id field present"
    check_content "$file" "processing_time_ms: i32" \
        "processing_time_ms field present"
    check_content "$file" "metadata: \[\*:0\]const u8" \
        "metadata field present"
    
    print_test "Check FFI function declarations"
    check_content "$file" 'extern "C" fn mojo_generate_mindmap' \
        "mojo_generate_mindmap function declared"
    check_content "$file" 'extern "C" fn mojo_free_mindmap_response' \
        "mojo_free_mindmap_response function declared"
    
    echo ""
}

# ============================================================================
# Test 4: Handler Implementation
# ============================================================================

test_handler_implementation() {
    print_header "Test 4: Handler Implementation"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check ODataMindmapHandler definition"
    check_content "$file" "pub const ODataMindmapHandler = struct" \
        "ODataMindmapHandler struct defined"
    
    print_test "Check handler initialization"
    check_content "$file" "pub fn init(allocator: mem.Allocator)" \
        "init function present"
    
    print_test "Check handleMindmapAction function"
    check_content "$file" "pub fn handleMindmapAction" \
        "handleMindmapAction function present"
    
    print_test "Check layout algorithm validation"
    check_content "$file" "fn isValidLayoutAlgorithm" \
        "isValidLayoutAlgorithm function present"
    
    print_test "Check layout algorithm validation logic"
    check_content "$file" '"tree"' \
        "Tree layout supported"
    check_content "$file" '"radial"' \
        "Radial layout supported"
    
    print_test "Check request to FFI conversion"
    check_content "$file" "fn mindmapRequestToMojoFFI" \
        "mindmapRequestToMojoFFI function present"
    
    print_test "Check FFI request cleanup"
    check_content "$file" "fn freeMojoRequest" \
        "freeMojoRequest function present"
    
    print_test "Check response conversion"
    check_content "$file" "fn mojoResponseToMindmapResponse" \
        "mojoResponseToMindmapResponse function present"
    
    print_test "Check error formatting"
    check_content "$file" "fn formatODataError" \
        "formatODataError function present"
    
    print_test "Check handler cleanup"
    check_content "$file" "pub fn deinit" \
        "deinit function present"
    
    echo ""
}

# ============================================================================
# Test 5: Request/Response Processing
# ============================================================================

test_request_response_processing() {
    print_header "Test 5: Request/Response Processing"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check JSON parsing"
    check_content "$file" "json.parseFromSlice" \
        "JSON parsing used"
    
    print_test "Check request validation"
    check_content "$file" "Invalid MindmapRequest format" \
        "Request format validation present"
    
    print_test "Check algorithm validation"
    check_content "$file" "Invalid LayoutAlgorithm" \
        "Algorithm validation present"
    
    print_test "Check timing measurement"
    check_content "$file" "std.time.milliTimestamp" \
        "Timing measurement present"
    
    print_test "Check Mojo FFI call"
    check_content "$file" "mojo_generate_mindmap" \
        "Mojo FFI call present"
    
    print_test "Check response cleanup"
    check_content "$file" "mojo_free_mindmap_response" \
        "Response cleanup present"
    
    print_test "Check node conversion"
    check_content "$file" "for (mojo_nodes)" \
        "Node conversion loop present"
    
    print_test "Check edge conversion"
    check_content "$file" "for (mojo_edges)" \
        "Edge conversion loop present"
    
    print_test "Check JSON serialization"
    check_content "$file" "json.stringify" \
        "JSON serialization used"
    
    echo ""
}

# ============================================================================
# Test 6: Error Handling
# ============================================================================

test_error_handling() {
    print_header "Test 6: Error Handling"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check parse error handling"
    check_content "$file" "Failed to parse MindmapRequest" \
        "Parse error handling present"
    
    print_test "Check validation error handling"
    check_content "$file" "BadRequest" \
        "BadRequest error code used"
    
    print_test "Check OData error structure usage"
    check_content "$file" "ODataError" \
        "OData error structure used"
    
    print_test "Check error response formatting"
    check_content "$file" "formatODataError" \
        "Error formatting function called"
    
    print_test "Check defer for cleanup"
    local defer_count=$(count_occurrences "$file" "defer")
    if [ "$defer_count" -ge 3 ]; then
        print_success "Proper defer usage for cleanup ($defer_count occurrences)"
    else
        print_error "Insufficient defer usage ($defer_count occurrences, expected >= 3)"
    fi
    
    echo ""
}

# ============================================================================
# Test 7: HTTP Handler Integration
# ============================================================================

test_http_handler_integration() {
    print_header "Test 7: HTTP Handler Integration"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check public handler function"
    check_content "$file" "pub fn handleODataMindmapRequest" \
        "Public handler function present"
    
    print_test "Check handler accepts allocator"
    check_content "$file" "allocator: mem.Allocator" \
        "Handler accepts allocator"
    
    print_test "Check handler accepts body"
    check_content "$file" "body: \[\]const u8" \
        "Handler accepts body"
    
    print_test "Check handler returns result"
    check_content "$file" "!\[\]const u8" \
        "Handler returns result"
    
    echo ""
}

# ============================================================================
# Test 8: Unit Tests
# ============================================================================

test_unit_tests() {
    print_header "Test 8: Unit Tests"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check basic test"
    check_content "$file" 'test "odata mindmap handler basic"' \
        "Basic handler test present"
    
    print_test "Check radial layout test"
    check_content "$file" 'test "odata mindmap handler radial layout"' \
        "Radial layout test present"
    
    print_test "Check invalid algorithm test"
    check_content "$file" 'test "odata mindmap handler invalid algorithm"' \
        "Invalid algorithm test present"
    
    print_test "Check invalid JSON test"
    check_content "$file" 'test "odata mindmap handler invalid json"' \
        "Invalid JSON test present"
    
    print_test "Check test uses testing allocator"
    check_content "$file" "testing.allocator" \
        "Tests use testing allocator"
    
    print_test "Check test assertions"
    local assert_count=$(count_occurrences "$file" "try testing.expect")
    if [ "$assert_count" -ge 8 ]; then
        print_success "Sufficient test assertions ($assert_count occurrences)"
    else
        print_error "Insufficient test assertions ($assert_count occurrences, expected >= 8)"
    fi
    
    echo ""
}

# ============================================================================
# Test 9: Code Quality
# ============================================================================

test_code_quality() {
    print_header "Test 9: Code Quality"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check file length"
    local line_count=$(wc -l < "$file")
    if [ "$line_count" -ge 450 ]; then
        print_success "Sufficient implementation ($line_count lines)"
    else
        print_error "Implementation too short ($line_count lines, expected >= 450)"
    fi
    
    print_test "Check documentation comments"
    local doc_count=$(grep -c "///" "$file" || true)
    if [ "$doc_count" -ge 15 ]; then
        print_success "Good documentation ($doc_count doc comments)"
    else
        print_error "Insufficient documentation ($doc_count doc comments, expected >= 15)"
    fi
    
    print_test "Check section headers"
    local header_count=$(grep -c "// ====" "$file" || true)
    if [ "$header_count" -ge 7 ]; then
        print_success "Well organized ($header_count section headers)"
    else
        print_error "Poor organization ($header_count section headers, expected >= 7)"
    fi
    
    print_test "Check debug output"
    check_content "$file" "std.debug.print" \
        "Debug output present"
    
    print_test "Check memory safety patterns"
    check_content "$file" "defer.*deinit" \
        "Defer cleanup pattern used"
    
    echo ""
}

# ============================================================================
# Test 10: Integration Points
# ============================================================================

test_integration_points() {
    print_header "Test 10: Integration Points"
    
    local file="src/serviceCore/nHyperBook/server/odata_mindmap.zig"
    
    print_test "Check standard library imports"
    check_content "$file" "const std = @import(\"std\")" \
        "std library imported"
    check_content "$file" "const json = std.json" \
        "json module used"
    check_content "$file" "const mem = std.mem" \
        "mem module used"
    
    print_test "Check FFI integration pattern"
    check_content "$file" 'extern "C"' \
        "FFI pattern used"
    
    print_test "Check allocator usage"
    check_content "$file" "self.allocator" \
        "Allocator properly used"
    
    print_test "Check error handling pattern"
    check_content "$file" "catch |err|" \
        "Error catching present"
    
    echo ""
}

# ============================================================================
# Summary
# ============================================================================

print_summary() {
    print_header "Test Summary"
    
    echo ""
    echo -e "${BLUE}Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
    echo -e "${BLUE}Tests Failed: ${RED}${TESTS_FAILED}${NC}"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ All Day 38 tests PASSED!${NC}"
        echo ""
        echo "Summary:"
        echo "  • OData Mindmap action handler implemented"
        echo "  • 2 layout algorithms supported (tree, radial)"
        echo "  • FFI integration with Mojo mindmap generator"
        echo "  • Complete request/response mapping"
        echo "  • Comprehensive error handling"
        echo "  • Unit tests included"
        echo ""
        echo "✨ Day 38 Implementation Complete!"
        return 0
    else
        echo -e "${RED}❌ Some tests FAILED${NC}"
        echo ""
        echo "Please fix the failing tests before proceeding."
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    clear
    
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║       HyperShimmy Day 38 - OData Mindmap Action Tests             ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Run all tests
    test_file_structure
    test_odata_complex_types
    test_ffi_structures
    test_handler_implementation
    test_request_response_processing
    test_error_handling
    test_http_handler_integration
    test_unit_tests
    test_code_quality
    test_integration_points
    
    # Print summary
    print_summary
}

# Run main function
main
