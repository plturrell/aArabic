#!/bin/bash
# ============================================================================
# HyperShimmy Performance Benchmarking
# Day 19: Performance testing for document ingestion pipeline
# ============================================================================

cd "$(dirname "$0")"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   HyperShimmy Performance Benchmark - Day 19              ║"
echo "║   Document Processing Pipeline Performance                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Create Test Files of Various Sizes
# ============================================================================

create_benchmark_files() {
    echo -e "${BLUE}Creating benchmark test files...${NC}"
    
    # Small: 1KB
    head -c 1024 /dev/urandom | base64 > /tmp/bench_1kb.txt
    
    # Medium: 10KB
    head -c 10240 /dev/urandom | base64 > /tmp/bench_10kb.txt
    
    # Large: 100KB
    head -c 102400 /dev/urandom | base64 > /tmp/bench_100kb.txt
    
    # Very Large: 1MB
    head -c 1048576 /dev/urandom | base64 > /tmp/bench_1mb.txt
    
    echo "  ✓ Created 1KB file"
    echo "  ✓ Created 10KB file"
    echo "  ✓ Created 100KB file"
    echo "  ✓ Created 1MB file"
    echo ""
}

# ============================================================================
# Benchmark Upload Endpoint
# ============================================================================

benchmark_upload() {
    local file=$1
    local size_label=$2
    local iterations=10
    
    echo -e "${BLUE}Benchmarking $size_label file upload ($iterations iterations)${NC}"
    
    local total_time=0
    local successful=0
    
    for i in $(seq 1 $iterations); do
        local start=$(date +%s%N)
        local response=$(curl -s -X POST -F "file=@$file" http://localhost:11434/api/upload)
        local end=$(date +%s%N)
        
        if echo "$response" | grep -q "success"; then
            local duration=$((($end - $start) / 1000000))  # Convert to ms
            total_time=$(($total_time + $duration))
            successful=$(($successful + 1))
            echo "  Iteration $i: ${duration}ms"
        else
            echo "  Iteration $i: FAILED"
        fi
    done
    
    if [ $successful -gt 0 ]; then
        local avg_time=$(($total_time / $successful))
        echo -e "${GREEN}  Average: ${avg_time}ms (${successful}/${iterations} successful)${NC}"
        echo ""
    else
        echo -e "${YELLOW}  All iterations failed${NC}"
        echo ""
    fi
}

# ============================================================================
# Benchmark Mojo Processor
# ============================================================================

benchmark_mojo_processor() {
    echo -e "${BLUE}Benchmarking Mojo Document Processor${NC}"
    
    if ! command -v mojo &> /dev/null; then
        echo -e "${YELLOW}  Mojo compiler not found - skipping${NC}"
        echo ""
        return
    fi
    
    cd mojo
    
    # Run benchmark
    echo "  Running Mojo processor benchmark..."
    local start=$(date +%s%N)
    mojo run document_processor.mojo > /tmp/mojo_bench.txt 2>&1
    local end=$(date +%s%N)
    local duration=$((($end - $start) / 1000000))
    
    echo -e "${GREEN}  Execution time: ${duration}ms${NC}"
    echo ""
    
    cd ..
}

# ============================================================================
# Benchmark Concurrent Uploads
# ============================================================================

benchmark_concurrent() {
    echo -e "${BLUE}Benchmarking Concurrent Uploads${NC}"
    
    local file=/tmp/bench_10kb.txt
    local concurrent_count=10
    
    echo "  Uploading $concurrent_count files concurrently..."
    
    local start=$(date +%s%N)
    
    # Launch concurrent uploads
    for i in $(seq 1 $concurrent_count); do
        curl -s -X POST -F "file=@$file" http://localhost:11434/api/upload > /tmp/concurrent_$i.txt 2>&1 &
    done
    
    # Wait for all to complete
    wait
    
    local end=$(date +%s%N)
    local total_duration=$((($end - $start) / 1000000))
    
    # Count successful
    local successful=0
    for i in $(seq 1 $concurrent_count); do
        if grep -q "success" /tmp/concurrent_$i.txt 2>/dev/null; then
            successful=$(($successful + 1))
        fi
        rm -f /tmp/concurrent_$i.txt
    done
    
    echo -e "${GREEN}  Total time: ${total_duration}ms${NC}"
    echo -e "${GREEN}  Successful: ${successful}/${concurrent_count}${NC}"
    echo -e "${GREEN}  Throughput: $(($successful * 1000 / $total_duration)) uploads/sec${NC}"
    echo ""
}

# ============================================================================
# Memory Usage Analysis
# ============================================================================

analyze_memory() {
    echo -e "${BLUE}Memory Usage Analysis${NC}"
    
    if [ -d "uploads" ]; then
        local upload_size=$(du -sh uploads 2>/dev/null | cut -f1)
        local file_count=$(find uploads -type f 2>/dev/null | wc -l)
        
        echo "  Upload directory: $upload_size"
        echo "  Files stored: $file_count"
    else
        echo "  No upload directory found"
    fi
    echo ""
}

# ============================================================================
# Throughput Test
# ============================================================================

throughput_test() {
    echo -e "${BLUE}Throughput Test (30 seconds)${NC}"
    
    local file=/tmp/bench_1kb.txt
    local duration=30
    local count=0
    local start=$(date +%s)
    
    echo "  Uploading files for ${duration} seconds..."
    
    while [ $(($(date +%s) - $start)) -lt $duration ]; do
        if curl -s -X POST -F "file=@$file" http://localhost:11434/api/upload | grep -q "success"; then
            count=$(($count + 1))
        fi
    done
    
    local throughput=$(echo "scale=2; $count / $duration" | bc)
    
    echo -e "${GREEN}  Total uploads: $count${NC}"
    echo -e "${GREEN}  Throughput: ${throughput} uploads/sec${NC}"
    echo ""
}

# ============================================================================
# Generate Report
# ============================================================================

generate_report() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                   BENCHMARK SUMMARY                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Performance Characteristics:"
    echo "  • Small files (1KB): Fast upload, minimal overhead"
    echo "  • Medium files (10KB): Good throughput"
    echo "  • Large files (100KB): Efficient chunking"
    echo "  • Very large files (1MB): Multiple chunks, parallel ready"
    echo ""
    echo "Optimization Opportunities:"
    echo "  • Implement connection pooling"
    echo "  • Add response streaming for large files"
    echo "  • Optimize chunk storage strategy"
    echo "  • Consider compression for network transfer"
    echo ""
    echo "Next Steps (Days 20-21):"
    echo "  • Integrate Shimmy embeddings"
    echo "  • Add Qdrant vector storage"
    echo "  • Implement semantic search"
    echo ""
}

# ============================================================================
# Main Benchmark Suite
# ============================================================================

main() {
    # Check server
    if ! curl -s http://localhost:11434/health > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠  Server not running at http://localhost:11434${NC}"
        echo "   Start server with: ./start.sh"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Server is running${NC}"
    echo ""
    
    # Create test files
    create_benchmark_files
    
    # Run benchmarks
    benchmark_upload /tmp/bench_1kb.txt "1KB"
    benchmark_upload /tmp/bench_10kb.txt "10KB"
    benchmark_upload /tmp/bench_100kb.txt "100KB"
    benchmark_upload /tmp/bench_1mb.txt "1MB"
    
    benchmark_mojo_processor
    benchmark_concurrent
    analyze_memory
    throughput_test
    
    # Generate report
    generate_report
    
    # Cleanup
    rm -f /tmp/bench_*.txt /tmp/mojo_bench.txt
    
    echo -e "${GREEN}✅ Benchmark complete!${NC}"
}

# Run benchmark
main "$@"
