#!/bin/bash

# Qdrant Integration Test Runner
# Quick-start script to run all Qdrant tests

set -e

echo "🚀 nCode Qdrant Integration Test Runner"
echo "========================================"
echo ""

# Check Python version
echo "📋 Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python version: $PYTHON_VERSION"

# Check if qdrant-client is installed
echo "📦 Checking dependencies..."
if ! python3 -c "import qdrant_client" 2>/dev/null; then
    echo "⚠️  qdrant-client not installed"
    echo "   Installing qdrant-client..."
    pip3 install qdrant-client
fi
echo "✅ qdrant-client installed"

# Check if Qdrant is running
echo ""
echo "🔌 Checking Qdrant connectivity..."
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "✅ Qdrant is running on localhost:6333"
else
    echo "❌ Qdrant is not running!"
    echo ""
    echo "To start Qdrant with Docker:"
    echo "  docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant"
    echo ""
    echo "Or pull the image first:"
    echo "  docker pull qdrant/qdrant"
    echo "  docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant"
    echo ""
    read -p "Do you want to start Qdrant now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Starting Qdrant..."
        docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
        echo "⏳ Waiting for Qdrant to start..."
        sleep 5
        if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
            echo "✅ Qdrant started successfully"
        else
            echo "❌ Failed to start Qdrant. Please start it manually."
            exit 1
        fi
    else
        echo "Please start Qdrant manually and run this script again."
        exit 1
    fi
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run tests
echo ""
echo "🧪 Running integration tests..."
echo "========================================"
echo ""

cd "$SCRIPT_DIR"
python3 qdrant_integration_test.py

# Check exit code
TEST_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
    echo ""
    echo "📊 Test Summary:"
    echo "   - Connection: ✅"
    echo "   - Collection creation: ✅"
    echo "   - Data insertion: ✅"
    echo "   - Basic search: ✅"
    echo "   - Filtered search: ✅"
    echo "   - Multi-filter search: ✅"
    echo "   - Payload retrieval: ✅"
    echo "   - Performance benchmark: ✅"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Review test results above"
    echo "   2. Check performance benchmarks"
    echo "   3. Test with real SCIP index (see DAY4_QDRANT_TESTING.md)"
    echo "   4. Proceed to Day 5 (Memgraph testing)"
else
    echo "❌ Some tests failed!"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "   1. Check Qdrant logs: docker logs qdrant"
    echo "   2. Verify Qdrant is accessible: curl http://localhost:6333/collections"
    echo "   3. Review test output above for specific errors"
    echo "   4. See DAY4_QDRANT_TESTING.md for detailed troubleshooting"
fi

echo ""
echo "📚 Documentation:"
echo "   - Test details: tests/DAY4_QDRANT_TESTING.md"
echo "   - Database guide: docs/DATABASE_INTEGRATION.md"
echo "   - Troubleshooting: docs/TROUBLESHOOTING.md"
echo ""

exit $TEST_EXIT_CODE
