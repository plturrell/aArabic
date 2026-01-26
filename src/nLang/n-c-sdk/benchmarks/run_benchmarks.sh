#!/bin/bash

set -e

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_DIR"

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "$1"
    echo "════════════════════════════════════════════════════════════"
    echo ""
}

run_mode() {
    local mode=$1
    print_header "Building benchmarks with -Doptimize=$mode"
    zig build -Doptimize="$mode"
    
    print_header "Running benchmarks (optimize=$mode)"
    echo "Array Operations:"
    ./zig-out/bin/array_operations
    echo ""
    echo "String Processing:"
    ./zig-out/bin/string_processing
    echo ""
    echo "Computation:"
    ./zig-out/bin/computation
    echo ""
}

case "${1:-compare}" in
    debug|Debug)
        run_mode "Debug"
        ;;
    releasesafe|ReleaseSafe)
        run_mode "ReleaseSafe"
        ;;
    releasefast|ReleaseFast)
        run_mode "ReleaseFast"
        ;;
    compare)
        print_header "🎯 Running Comprehensive Benchmark Comparison"
        echo "This will run benchmarks in Debug, ReleaseSafe, and ReleaseFast modes"
        echo "to demonstrate the performance impact of optimization settings."
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        
        run_mode "Debug"
        run_mode "ReleaseSafe"
        run_mode "ReleaseFast"
        
        print_header "✅ Benchmark Comparison Complete"
        echo "Review the results above to see the performance differences"
        echo "between Debug, ReleaseSafe, and ReleaseFast optimization modes."
        ;;
    *)
        echo "Usage: $0 {debug|releasesafe|releasefast|compare}"
        echo ""
        echo "Modes:"
        echo "  debug        - Run with Debug optimization (baseline)"
        echo "  releasesafe  - Run with ReleaseSafe optimization (SDK default)"
        echo "  releasefast  - Run with ReleaseFast optimization (maximum speed)"
        echo "  compare      - Run all three modes for comparison (default)"
        exit 1
        ;;
esac

print_header "Done!"
