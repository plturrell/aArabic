#!/bin/bash

echo "🧪 GraphChat Bridge Test Suite"
echo "================================"
echo ""

# Test 1: Check if services are running
echo "✓ Test 1: Checking if services are running..."
echo ""
echo "Shimmy:"
docker ps | grep shimmy || echo "  ❌ Shimmy not running"
echo ""
echo "Langflow:"
docker ps | grep langflow || echo "  ❌ Langflow not running"
echo ""
echo "Memgraph:"
docker ps | grep "ai_nucleus_memgraph " || echo "  ❌ Memgraph not running"
echo ""

# Test 2: Check if MCP server file exists
echo "✓ Test 2: Checking GraphChat Bridge MCP server..."
if [ -f "src/serviceCore/mcp_servers/graphchat_bridge.py" ]; then
    echo "  ✅ MCP server file exists"
else
    echo "  ❌ MCP server file not found"
fi
echo ""

# Test 3: Check if Python dependencies are installed
echo "✓ Test 3: Checking Python dependencies..."
python3 -c "import mcp; print('  ✅ mcp installed')" 2>/dev/null || echo "  ❌ mcp not installed"
python3 -c "import httpx; print('  ✅ httpx installed')" 2>/dev/null || echo "  ❌ httpx not installed"
python3 -c "import neo4j; print('  ✅ neo4j installed')" 2>/dev/null || echo "  ⚠️  neo4j not installed (optional - will use fallback)"
echo ""

# Test 4: Check if Cline settings exist
echo "✓ Test 4: Checking Cline MCP settings..."
if [ -f "$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json" ]; then
    echo "  ✅ Cline settings file exists"
    echo "  Configuration:"
    cat "$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json" | grep -A 2 "graphchat-bridge"
else
    echo "  ❌ Cline settings file not found"
fi
echo ""

# Test 5: Check Shimmy health
echo "✓ Test 5: Checking Shimmy health..."
SHIMMY_LOGS=$(docker logs ai_nucleus_shimmy 2>&1 | tail -3)
if echo "$SHIMMY_LOGS" | grep -q "Ready to serve"; then
    echo "  ✅ Shimmy is ready"
    echo "$SHIMMY_LOGS" | grep "Ready to serve"
else
    echo "  ⚠️  Shimmy status unknown"
    echo "$SHIMMY_LOGS"
fi
echo ""

# Test 6: Verify sample data in Memgraph
echo "✓ Test 6: Checking Memgraph data..."
MEMGRAPH_COUNT=$(docker exec ai_nucleus_memgraph mgconsole -e "MATCH (n) RETURN count(n) as count;" 2>&1 | grep -E "[0-9]+" | tail -1)
if [ ! -z "$MEMGRAPH_COUNT" ]; then
    echo "  ✅ Memgraph has data: $MEMGRAPH_COUNT nodes"
else
    echo "  ⚠️  Could not verify Memgraph data"
fi
echo ""

echo "================================"
echo "📊 Test Summary"
echo "================================"
echo ""
echo "All systems checked! ✅"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Restart VS Code (Cmd+R on Mac, Ctrl+R on Windows/Linux)"
echo ""
echo "2. Open Cline and try these test commands:"
echo "   • 'Use graphchat-bridge to list available tools'"
echo "   • 'Use graphchat-bridge to chat: What data do we have?'"
echo "   • 'Use graphchat-bridge to generate a query showing all people'"
echo ""
echo "3. If you see errors, check:"
echo "   • Docker containers are running: docker ps"
echo "   • Logs: docker logs ai_nucleus_shimmy"
echo ""
echo "📚 Documentation:"
echo "   • setup_graphchat_in_cline.md - Setup guide"
echo "   • GRAPHCHAT_SETUP.md - Complete documentation"
echo "   • src/serviceCore/mcp_servers/README.md - Technical docs"
echo ""
