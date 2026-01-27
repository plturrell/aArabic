#!/bin/bash
# Comprehensive Zig 0.15.1 API Migration Script
# Fixes ArrayList.init() across all serviceCore services

set -e

echo "=================================================="
echo "Zig 0.15.1 API Migration - All Services"
echo "=================================================="

BASE_DIR="$(dirname "$0")/../src/serviceCore"

# Function to fix ArrayList.init() in a service
fix_service() {
    local service=$1
    local service_path="$BASE_DIR/$service"
    
    if [ ! -d "$service_path" ]; then
        echo "⚠️  Service $service not found, skipping..."
        return
    fi
    
    echo ""
    echo "Fixing $service..."
    echo "-------------------"
    
    # Count occurrences before fix
    local count_before=$(grep -r "ArrayList.*\.init(" "$service_path" 2>/dev/null | wc -l | tr -d ' ')
    echo "Found $count_before ArrayList.init() calls"
    
    # Fix ArrayList(T).init(allocator) -> ArrayList(T){}
    # Using perl for more reliable regex replacement
    find "$service_path" -name "*.zig" -type f -exec perl -i -pe 's/ArrayList\(([^)]+)\)\.init\([^)]+\)/ArrayList($1){}/g' {} +
    
    # Count occurrences after fix
    local count_after=$(grep -r "ArrayList.*\.init(" "$service_path" 2>/dev/null | wc -l | tr -d ' ')
    local fixed=$((count_before - count_after))
    
    echo "✓ Fixed $fixed ArrayList.init() calls"
    
    if [ $count_after -gt 0 ]; then
        echo "⚠️  Warning: $count_after ArrayList.init() calls remaining (may need manual review)"
    fi
}

# Fix all services in order (simplest to most complex)
echo "Starting migration..."

fix_service "nGrounding"
fix_service "nAgentMeta" 
fix_service "nLocalModels"
fix_service "nAgentFlow"

echo ""
echo "=================================================="
echo "Migration Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review changes: git diff src/serviceCore/"
echo "2. Test builds service-by-service"
echo "3. Commit if successful"
echo ""
echo "Note: Some append() calls may need allocator parameter."
echo "Build errors will indicate which ones need manual fixes."