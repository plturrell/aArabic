#!/bin/bash
# Fix Zig 0.15.2 API issues in nLocalModels orchestration

set -e

cd "$(dirname "$0")/../src/serviceCore/nLocalModels/orchestration"

echo "Applying Zig 0.15.2 API fixes..."

# Fix benchmark_cli.zig - std.io API
if [ -f "benchmark_cli.zig" ]; then
    echo "Fixing benchmark_cli.zig..."
    sed -i '' 's/std\.io\.getStdOut()/std.fs.File.stdout()/g' benchmark_cli.zig
    sed -i '' 's/\.writer()/\.deprecatedWriter()/g' benchmark_cli.zig
    sed -i '' 's/var sorted =/const sorted =/g' benchmark_cli.zig
fi

# Fix analytics.zig - std.io API and unused vars
if [ -f "analytics.zig" ]; then
    echo "Fixing analytics.zig..."
    sed -i '' 's/std\.io\.getStdErr()/std.fs.File.stderr()/g' analytics.zig
    # Add _ = to unused variables
    sed -i '' '471s/const epoch_day/_ = epoch_day; \/\/ const epoch_day/' analytics.zig || true
    sed -i '' '472s/const day_seconds/_ = day_seconds; \/\/ const day_seconds/' analytics.zig || true
fi

# Fix model_selector.zig - remove pointless discards
if [ -f "model_selector.zig" ]; then
    echo "Fixing model_selector.zig..."
    sed -i '' '/^        _ = self;$/d' model_selector.zig
    sed -i '' '/^        _ = task_category;$/d' model_selector.zig
fi

# Fix gpu_monitor.zig - ArrayList init (if it exists)
if [ -f "gpu_monitor.zig" ]; then
    echo "Fixing gpu_monitor.zig..."
    # This one is trickier - might need manual fix
    echo "  Note: gpu_monitor.zig may need manual ArrayList.init fix"
fi

echo "Fixes applied! Now run: zig build"
