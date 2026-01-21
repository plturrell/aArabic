#!/bin/bash
# Build all Rust API clients
# This script builds all 17+ Rust CLI clients in release mode

set -e  # Exit on error

echo "🔨 Building all Rust API clients..."
echo "=================================="

cd "$(dirname "$0")/../src/serviceAutomation"

# Counter for tracking
total=0
success=0
failed=0

# List of all client directories
clients=(
    "gitea-api-client"
    "git-api-client"
    "postgres-api-client"
    "kafka-api-client"
    "apisix-api-client"
    "keycloak-api-client"
    "filesystem-api-client"
    "memory-api-client"
    "shimmy-api-client"
    "marquez-api-client"
    "n8n-api-client"
    "opencanvas-api-client"
    "hyperbook-api-client"
    "qdrant-api-client"
    "memgraph-api-client"
    "dragonflydb-api-client"
    "ncode-api-client"
    "lean4-api-client"
)

for client in "${clients[@]}"; do
    if [ -d "$client" ]; then
        total=$((total + 1))
        echo ""
        echo "Building $client..."
        
        if (cd "$client" && cargo build --release 2>&1 | tail -5); then
            success=$((success + 1))
            echo "✅ $client built successfully"
        else
            failed=$((failed + 1))
            echo "❌ $client build failed"
        fi
    else
        echo "⚠️  $client directory not found, skipping..."
    fi
done

echo ""
echo "=================================="
echo "📊 Build Summary:"
echo "   Total:   $total clients"
echo "   Success: $success clients"
echo "   Failed:  $failed clients"
echo "=================================="

if [ $failed -eq 0 ]; then
    echo "✅ All clients built successfully!"
    echo ""
    echo "Next step: Run ./install_rust_clients.sh to install CLIs"
    exit 0
else
    echo "❌ Some clients failed to build"
    exit 1
fi
