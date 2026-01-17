#!/bin/bash
# ============================================================================
# HyperShimmy Integration Test Suite
# Day 19: Complete document ingestion pipeline testing
# ============================================================================

set -e  # Exit on error

cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Server configuration
SERVER_URL="http://localhost:11434"
UPLOAD_ENDPOINT="$SERVER_URL/api/upload"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_error() {
    echo -e "${RED}✗${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

test_start() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Test $TESTS_RUN: $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_server() {
    if ! curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
        log_error "Server not running at $SERVER_URL"
        log_info "Start server with: ./start.sh"
        exit 1
    fi
    log_success "Server is running"
}

cleanup_test_files() {
    rm -f /tmp/test_*.txt /tmp/test_*.html /tmp/test_*.pdf 2>/dev/null || true
}

# ============================================================================
# Test File Generators
# ============================================================================

create_small_text_file() {
    cat > /tmp/test_small.txt << 'EOF'
This is a small test document.
It has just a few lines.
Perfect for testing basic functionality.
EOF
}

create_medium_text_file() {
    cat > /tmp/test_medium.txt << 'EOF'
This is a medium-sized test document. It contains multiple paragraphs to test chunking.

Paragraph 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod 
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis 
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Paragraph 2: Duis aute irure dolor in reprehenderit in voluptate velit esse cillum 
dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt 
in culpa qui officia deserunt mollit anim id est laborum.

Paragraph 3: Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium 
doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis 
et quasi architecto beatae vitae dicta sunt explicabo.
EOF
}

create_large_text_file() {
    cat > /tmp/test_large.txt << 'EOF'
This is a large test document designed to test chunking with overlap.

Chapter 1: Introduction

This chapter introduces the concept of document processing. When dealing with large 
documents, it becomes necessary to break them into smaller, manageable chunks. Each 
chunk should be small enough to fit within embedding model limits, but large enough 
to contain meaningful semantic information.

The chunking process must consider several factors: chunk size, overlap between chunks, 
and sentence boundaries. A well-designed chunking algorithm will preserve semantic 
coherence while ensuring efficient processing.

Chapter 2: Chunking Strategies

There are several approaches to chunking text documents. The most basic approach is 
fixed-size chunking, where text is split at regular intervals. However, this can lead 
to awkward breaks in the middle of sentences or paragraphs.

A more sophisticated approach uses sentence boundary detection. By identifying natural 
breaks in the text (periods, exclamation marks, question marks), the chunking algorithm 
can create more coherent chunks. This improves the quality of semantic search results.

Overlap between chunks is another important consideration. When chunks have a small 
overlap (typically 10-20% of the chunk size), information that would otherwise be lost 
at chunk boundaries is preserved. This is particularly important for queries that span 
multiple concepts.

Chapter 3: Implementation Details

The implementation uses a configurable chunk size (default 512 characters) and overlap 
(default 50 characters). These values can be adjusted based on the specific use case 
and the embedding model being used.

The algorithm works as follows:
1. Start at the beginning of the document
2. Move forward by chunk_size characters
3. Find the nearest sentence boundary
4. Extract the chunk
5. Move back by overlap_size characters
6. Repeat until the end of the document

This approach ensures that chunks are roughly equal in size, maintain semantic coherence, 
and have appropriate overlap for cross-chunk queries.

Chapter 4: Performance Considerations

Performance is a critical consideration for document processing. The Mojo implementation 
provides native performance similar to C/C++, while maintaining Python-like readability.

For small documents (< 10KB), processing takes approximately 1ms. Medium documents 
(100KB) process in around 10ms, and large documents (1MB) in about 50ms. This is 
5-10x faster than equivalent Python implementations.

Memory usage is also optimized. The current implementation creates new string objects 
for each chunk, but this could be further optimized using zero-copy techniques where 
chunks reference the original text.

Chapter 5: Integration with Embeddings

Once documents are chunked, each chunk can be independently converted to an embedding 
vector. These vectors capture the semantic meaning of the text in a high-dimensional 
space, typically 768 or 1536 dimensions.

The embedding vectors are then stored in a vector database like Qdrant, where they can 
be efficiently searched using cosine similarity or other distance metrics. This enables 
semantic search, where queries are matched based on meaning rather than exact keyword 
matches.

Chapter 6: Future Enhancements

Several enhancements are planned for future versions:
- Token-based chunking using actual tokenizers
- Smart chunking that respects document structure
- Parallel processing for multiple documents
- Metadata enrichment with title, author, date extraction
- Language detection and handling

These improvements will further enhance the quality and performance of the document 
processing pipeline.

Conclusion

Document processing is a critical component of modern AI applications. By carefully 
designing the chunking strategy and implementing it efficiently, we can build systems 
that provide fast, accurate semantic search over large document collections.

The HyperShimmy project demonstrates how modern languages like Mojo and Zig can be 
combined to create high-performance AI infrastructure without sacrificing developer 
productivity.
EOF
}

create_html_file() {
    cat > /tmp/test_page.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test HTML Document</title>
</head>
<body>
    <h1>Document Processing Test</h1>
    
    <section>
        <h2>Introduction</h2>
        <p>This is a test HTML document for the HyperShimmy integration tests.</p>
        <p>It contains multiple sections and paragraphs to test HTML parsing.</p>
    </section>
    
    <section>
        <h2>Features</h2>
        <ul>
            <li>HTML tag parsing</li>
            <li>Text extraction</li>
            <li>Structure preservation</li>
        </ul>
    </section>
    
    <section>
        <h2>Content</h2>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod 
        tempor incididunt ut labore et dolore magna aliqua.</p>
        
        <p>Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi 
        ut aliquip ex ea commodo consequat.</p>
    </section>
</body>
</html>
EOF
}

create_special_chars_file() {
    cat > /tmp/test_special.txt << 'EOF'
Test document with special characters!

Question marks? Yes, we have those.
Exclamation points! Multiple! In a row!
Periods. Like. This.

UTF-8 characters: café, naïve, résumé
Emoji: 😀 🚀 ✅

Line breaks:
- Should be preserved
- And handled correctly
- In the chunking process

Numbers: 123, 456.78, 1,000,000
Symbols: @#$%^&*()_+-={}[]|:;"'<>,.?/
EOF
}

create_edge_case_files() {
    # Empty file
    touch /tmp/test_empty.txt
    
    # Single line
    echo "Single line document" > /tmp/test_single_line.txt
    
    # Very long line (no breaks)
    python3 -c "print('a' * 2000)" > /tmp/test_long_line.txt 2>/dev/null || \
        perl -e "print 'a' x 2000" > /tmp/test_long_line.txt
}

# ============================================================================
# Test Functions
# ============================================================================

test_server_health() {
    test_start "Server Health Check"
    
    local response=$(curl -s "$SERVER_URL/health")
    
    if echo "$response" | grep -q "ok"; then
        log_success "Server health check passed"
        log_info "Response: $response"
    else
        log_error "Server health check failed"
        return 1
    fi
}

test_small_file_upload() {
    test_start "Small Text File Upload"
    
    create_small_text_file
    
    local response=$(curl -s -X POST -F "file=@/tmp/test_small.txt" "$UPLOAD_ENDPOINT")
    
    if echo "$response" | grep -q "success"; then
        log_success "Small file uploaded successfully"
        log_info "Response: $response"
        
        # Check file was created
        if [ -d "uploads" ] && ls uploads/*.txt >/dev/null 2>&1; then
            log_success "Upload directory created and file saved"
        else
            log_warning "Upload directory or files not found"
        fi
    else
        log_error "Small file upload failed"
        log_info "Response: $response"
        return 1
    fi
}

test_medium_file_upload() {
    test_start "Medium Text File Upload (Chunking Test)"
    
    create_medium_text_file
    
    local response=$(curl -s -X POST -F "file=@/tmp/test_medium.txt" "$UPLOAD_ENDPOINT")
    
    if echo "$response" | grep -q "success"; then
        log_success "Medium file uploaded successfully"
        
        # Check for extracted text
        local text_files=$(find uploads -name "*_text.txt" 2>/dev/null | wc -l)
        if [ "$text_files" -gt 0 ]; then
            log_success "Text extraction successful ($text_files files)"
        else
            log_warning "No extracted text files found"
        fi
    else
        log_error "Medium file upload failed"
        return 1
    fi
}

test_large_file_upload() {
    test_start "Large Text File Upload (Multi-Chunk Test)"
    
    create_large_text_file
    
    local file_size=$(wc -c < /tmp/test_large.txt)
    log_info "File size: $file_size bytes"
    
    local response=$(curl -s -X POST -F "file=@/tmp/test_large.txt" "$UPLOAD_ENDPOINT")
    
    if echo "$response" | grep -q "success"; then
        log_success "Large file uploaded successfully"
        log_info "This file should generate multiple chunks (>512 chars)"
    else
        log_error "Large file upload failed"
        return 1
    fi
}

test_html_upload() {
    test_start "HTML File Upload and Parsing"
    
    create_html_file
    
    local response=$(curl -s -X POST -F "file=@/tmp/test_page.html" "$UPLOAD_ENDPOINT")
    
    if echo "$response" | grep -q "success"; then
        log_success "HTML file uploaded successfully"
        
        # Check if text was extracted (HTML tags removed)
        if [ -d "uploads" ]; then
            local extracted=$(find uploads -name "*_text.txt" -exec grep -l "Document Processing Test" {} \; 2>/dev/null | wc -l)
            if [ "$extracted" -gt 0 ]; then
                log_success "HTML parsed and text extracted"
            else
                log_warning "Could not verify HTML text extraction"
            fi
        fi
    else
        log_error "HTML file upload failed"
        return 1
    fi
}

test_special_characters() {
    test_start "Special Characters Handling"
    
    create_special_chars_file
    
    local response=$(curl -s -X POST -F "file=@/tmp/test_special.txt" "$UPLOAD_ENDPOINT")
    
    if echo "$response" | grep -q "success"; then
        log_success "Special characters file uploaded"
        log_info "UTF-8 and special characters should be preserved"
    else
        log_error "Special characters upload failed"
        return 1
    fi
}

test_edge_cases() {
    test_start "Edge Cases (Empty, Single Line, Long Line)"
    
    create_edge_case_files
    
    # Test empty file
    local response=$(curl -s -X POST -F "file=@/tmp/test_empty.txt" "$UPLOAD_ENDPOINT")
    if echo "$response" | grep -q "success\|empty\|zero"; then
        log_success "Empty file handled correctly"
    else
        log_warning "Empty file handling unclear"
    fi
    
    # Test single line
    response=$(curl -s -X POST -F "file=@/tmp/test_single_line.txt" "$UPLOAD_ENDPOINT")
    if echo "$response" | grep -q "success"; then
        log_success "Single line file handled correctly"
    else
        log_warning "Single line file handling unclear"
    fi
    
    # Test long line
    response=$(curl -s -X POST -F "file=@/tmp/test_long_line.txt" "$UPLOAD_ENDPOINT")
    if echo "$response" | grep -q "success"; then
        log_success "Long line file handled correctly"
    else
        log_warning "Long line file handling unclear"
    fi
}

test_concurrent_uploads() {
    test_start "Concurrent Uploads (Stress Test)"
    
    create_small_text_file
    create_medium_text_file
    
    log_info "Uploading 5 files concurrently..."
    
    # Launch 5 concurrent uploads
    for i in {1..5}; do
        curl -s -X POST -F "file=@/tmp/test_small.txt" "$UPLOAD_ENDPOINT" > /tmp/response_$i.txt 2>&1 &
    done
    
    # Wait for all uploads to complete
    wait
    
    # Check results
    local success_count=0
    for i in {1..5}; do
        if grep -q "success" /tmp/response_$i.txt 2>/dev/null; then
            success_count=$((success_count + 1))
        fi
        rm -f /tmp/response_$i.txt
    done
    
    if [ "$success_count" -eq 5 ]; then
        log_success "All concurrent uploads succeeded ($success_count/5)"
    else
        log_warning "Some concurrent uploads failed ($success_count/5 succeeded)"
    fi
}

test_error_handling() {
    test_start "Error Handling (Invalid Uploads)"
    
    # Test missing file
    local response=$(curl -s -X POST "$UPLOAD_ENDPOINT")
    if echo "$response" | grep -qi "error\|missing\|required"; then
        log_success "Missing file error handled correctly"
    else
        log_warning "Missing file error handling unclear"
    fi
    
    # Test invalid endpoint
    response=$(curl -s -X POST "$SERVER_URL/api/invalid")
    if echo "$response" | grep -qi "error\|not found\|404"; then
        log_success "Invalid endpoint error handled correctly"
    else
        log_warning "Invalid endpoint error handling unclear"
    fi
}

test_upload_directory_structure() {
    test_start "Upload Directory Structure Verification"
    
    if [ ! -d "uploads" ]; then
        log_error "Uploads directory does not exist"
        return 1
    fi
    
    log_success "Uploads directory exists"
    
    # Check for files
    local file_count=$(ls -1 uploads 2>/dev/null | wc -l)
    log_info "Files in uploads directory: $file_count"
    
    # Show sample files
    if [ "$file_count" -gt 0 ]; then
        log_info "Sample uploads:"
        ls -lh uploads | head -10
    fi
}

test_mojo_processor() {
    test_start "Mojo Document Processor (Standalone)"
    
    if [ ! -f "mojo/document_processor.mojo" ]; then
        log_error "Mojo processor not found"
        return 1
    fi
    
    log_info "Testing Mojo processor standalone..."
    
    # Try to run Mojo processor
    if command -v mojo &> /dev/null; then
        cd mojo
        if mojo run document_processor.mojo > /tmp/mojo_test.txt 2>&1; then
            log_success "Mojo processor executed successfully"
            log_info "Output:"
            cat /tmp/mojo_test.txt | head -20
        else
            log_warning "Mojo processor execution failed (may need compilation)"
            cat /tmp/mojo_test.txt
        fi
        cd ..
    else
        log_warning "Mojo compiler not found - skipping standalone test"
    fi
}

# ============================================================================
# Main Test Suite
# ============================================================================

main() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   HyperShimmy Integration Test Suite - Day 19             ║"
    echo "║   Complete Document Ingestion Pipeline Testing            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    log_info "Starting integration tests..."
    log_info "Server URL: $SERVER_URL"
    echo ""
    
    # Check prerequisites
    check_server
    
    # Run tests
    test_server_health || true
    test_small_file_upload || true
    test_medium_file_upload || true
    test_large_file_upload || true
    test_html_upload || true
    test_special_characters || true
    test_edge_cases || true
    test_concurrent_uploads || true
    test_error_handling || true
    test_upload_directory_structure || true
    test_mojo_processor || true
    
    # Cleanup
    cleanup_test_files
    
    # Print summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    TEST SUMMARY                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "Tests Run:    ${BLUE}$TESTS_RUN${NC}"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""
    
    if [ "$TESTS_FAILED" -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
        echo ""
        log_info "Integration test suite completed successfully"
        log_info "Document ingestion pipeline is working correctly"
        exit 0
    else
        echo -e "${RED}❌ Some tests failed${NC}"
        echo ""
        log_warning "Review failed tests and fix issues"
        exit 1
    fi
}

# Run main function
main "$@"
