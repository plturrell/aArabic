# GraphChat Bridge MCP Server

This MCP server enables Memgraph Lab's GraphChat feature by providing:
- LLM-powered natural language to Cypher query generation
- Graph schema introspection
- Query explanation
- Interactive chat about your graph data

## Features

### Tools
1. **execute_cypher** - Execute Cypher queries against Memgraph
2. **generate_cypher** - Generate Cypher from natural language questions
3. **explain_query** - Explain what a Cypher query does
4. **chat_about_graph** - Have conversations about your graph

### Resources
1. **graph://schema** - Current graph schema
2. **graph://stats** - Graph statistics

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make the script executable
chmod +x graphchat_bridge.py
```

## Configuration

### For Cline/Claude Desktop

Add to your Cline MCP settings (`~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "graphchat-bridge": {
      "command": "python3",
      "args": [
        "/Users/user/Documents/arabic_folder/src/serviceCore/mcp_servers/graphchat_bridge.py"
      ],
      "env": {
        "MEMGRAPH_URI": "bolt://localhost:7687",
        "SHIMMY_URL": "http://localhost:3001",
        "DEFAULT_MODEL": "Qwen3-Coder-30B-A3B-Instruct"
      }
    }
  }
}
```

### For Memgraph Lab GraphChat

Memgraph Lab v3.7+ can connect to MCP servers directly through its UI:

1. Open Memgraph Lab at http://localhost:3001
2. Click on "GraphChat" in the left menu
3. Click "Connect to MCP server" or similar
4. Configure:
   - **Server Type**: stdio
   - **Command**: `python3`
   - **Args**: `[path to graphchat_bridge.py]`
   - **Environment Variables**:
     - `MEMGRAPH_URI=bolt://localhost:7687`
     - `SHIMMY_URL=http://localhost:3001`

## Usage Examples

### Generate Cypher from Natural Language

```
"Show me all people working on the AI Nucleus project"
→ MATCH (p:Person)-[:WORKS_ON]->(proj:Project {name: 'AI Nucleus'}) RETURN p
```

### Chat About Your Graph

```
User: "What projects do we have?"
AI: "Based on the graph, you have 4 projects: AI Nucleus, Arabic NLP, 
     Graph Analytics, and Workflow Automation."

User: "Who is working on Arabic NLP?"
AI: "Let me check... [generates and executes query] Bob and Charlie are 
     working on the Arabic NLP project."
```

### Explain Queries

```
Input: MATCH (p:Person)-[r:WORKS_ON]->(proj:Project) 
       WHERE r.hoursPerWeek > 30 
       RETURN p.name, proj.name

Output: "This query finds all people who work on projects for more than 
         30 hours per week, and returns their names along with the 
         project names."
```

## Architecture

```
┌─────────────────────┐
│  Memgraph Lab       │
│  (GraphChat UI)     │
└──────────┬──────────┘
           │
           │ MCP Protocol
           │
┌──────────▼──────────┐
│ GraphChat Bridge    │
│ MCP Server          │
├─────────────────────┤
│ - Cypher Generation │
│ - Query Execution   │
│ - Schema Access     │
│ - Chat Interface    │
└─────┬───────┬───────┘
      │       │
      │       │
┌─────▼────┐ ┌▼──────────┐
│ Memgraph │ │  Shimmy   │
│ Database │ │  (LLM)    │
└──────────┘ └───────────┘
```

## Troubleshooting

### Connection Issues

**Problem**: "Cannot connect to Memgraph"
**Solution**: Ensure Memgraph is running:
```bash
docker ps | grep memgraph
# Should show ai_nucleus_memgraph running
```

**Problem**: "Shimmy API error"
**Solution**: Check Shimmy is accessible:
```bash
curl http://localhost:3001/health
```

### MCP Server Not Starting

**Problem**: "Module 'mcp' not found"
**Solution**: Install dependencies:
```bash
cd src/serviceCore/mcp_servers
pip install -r requirements.txt
```

### GraphChat Not Finding Server

**Problem**: GraphChat setup shows no servers
**Solution**: 
1. Restart Memgraph Lab container
2. Check MCP server configuration in Lab settings
3. Verify the python script is executable and has correct path

## Development

### Testing the MCP Server

```bash
# Test directly via stdio
python3 graphchat_bridge.py

# The server will wait for MCP protocol messages on stdin
# Send a test message (in MCP JSON-RPC format):
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

### Adding New Tools

1. Add tool definition to `list_tools()`
2. Implement handler in `call_tool()`
3. Test with sample queries

## License

Part of the AI Nucleus Platform
