# Setting Up GraphChat Bridge in Cline

## Step 1: Locate Your Cline MCP Settings File

The file is located at:
```
~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
```

## Step 2: Add GraphChat Configuration

Open the file and add this configuration (or create the file if it doesn't exist):

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
        "SHIMMY_URL": "http://localhost:11434",
        "LANGFLOW_URL": "http://localhost:7860",
        "MODEL_SERVER_URL": "http://localhost:8000",
        "DEFAULT_MODEL": "Qwen3-Coder-30B-A3B-Instruct"
      }
    }
  }
}
```

**If the file already has content**, merge this into the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "existing-server": {
      ...existing config...
    },
    "graphchat-bridge": {
      "command": "python3",
      "args": [
        "/Users/user/Documents/arabic_folder/src/serviceCore/mcp_servers/graphchat_bridge.py"
      ],
      "env": {
        "MEMGRAPH_URI": "bolt://localhost:7687",
        "SHIMMY_URL": "http://localhost:11434",
        "LANGFLOW_URL": "http://localhost:7860",
        "MODEL_SERVER_URL": "http://localhost:8000",
        "DEFAULT_MODEL": "Qwen3-Coder-30B-A3B-Instruct"
      }
    }
  }
}
```

## Step 3: Restart Cline

1. Close all Cline chat windows
2. Reload VS Code window:
   - **Mac**: Press `Cmd+R`
   - **Windows/Linux**: Press `Ctrl+R`
3. Open Cline again

## Step 4: Test GraphChat

Once Cline restarts, you should see the graphchat-bridge server available. Try these commands:

### Test 1: List Available Tools
```
Use the graphchat-bridge server to list all available tools
```

**Expected**: Should show 4 tools (execute_cypher, generate_cypher, explain_query, chat_about_graph)

### Test 2: Generate a Simple Query
```
Use graphchat-bridge to generate a Cypher query that returns all people in the database
```

**Expected**: Should generate something like:
```cypher
MATCH (p:Person)
RETURN p.name, p.role, p.department
```

### Test 3: Execute a Query
```
Use graphchat-bridge to execute this query:
MATCH (n:Person)
RETURN n.name, n.role
LIMIT 5
```

**Expected**: Should return the list of people from your graph

### Test 4: Interactive Chat
```
Use graphchat-bridge to chat about the graph: "What kind of data do we have in this database?"
```

**Expected**: AI will analyze the schema and describe your graph structure

## Troubleshooting

### Issue: "graphchat-bridge server not found"

**Solution 1**: Check file location
```bash
ls -la ~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/
```

**Solution 2**: Create directory if missing
```bash
mkdir -p ~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/
```

### Issue: "Python module not found"

**Solution**: Ensure dependencies are installed
```bash
cd /Users/user/Documents/arabic_folder/src/serviceCore/mcp_servers
pip3 install -r requirements.txt
```

### Issue: "Cannot connect to Memgraph"

**Solution**: Verify Memgraph is running
```bash
docker ps | grep memgraph
```

### Issue: "All LLM services unavailable"

**Solution**: Check services are running
```bash
docker ps | grep -E "shimmy|langflow"
docker logs ai_nucleus_shimmy | tail -10
```

## What Happens Behind the Scenes

When you use graphchat-bridge in Cline:

1. **Cline** sends your request to the GraphChat Bridge MCP server
2. **GraphChat Bridge** tries to use **Shimmy** first (fastest)
3. If Shimmy fails, it tries **Model Server**
4. If Model Server fails, it tries **Langflow**
5. The result is sent back to Cline and shown to you

All of this happens automatically with intelligent fallback!

## Next Steps After Setup

Once GraphChat is working in Cline, you can:

1. **Explore your data**: "Show me all the relationships in the database"
2. **Find patterns**: "Which people work on multiple projects?"
3. **Get insights**: "What's the average team size per project?"
4. **Learn Cypher**: "Explain this query: MATCH (p)-[r]->(n) RETURN p, r, n"

Enjoy your AI-powered graph exploration! 🚀
