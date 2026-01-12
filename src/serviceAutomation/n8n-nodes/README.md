# 🎯 n8n Nodes for Rust API Clients

**Complete n8n workflow integration for all 17 Rust API clients**

## 📦 What's Included

### Unified Node Approach

We provide a **unified n8n node** that integrates all 17 Rust CLI clients through a single, flexible interface. This approach is more practical for n8n as it:
- Reduces complexity (1 node vs 17 nodes)
- Provides consistent interface
- Easier to maintain and extend
- Simpler installation

### 17 Rust Clients Available

| # | Client | CLI Name | Default URL |
|---|--------|----------|-------------|
| 1 | Langflow | `langflow-cli` | http://localhost:7860 |
| 2 | Gitea | `gitea-cli` | http://localhost:3000 |
| 3 | Git | `git-cli` | (local) |
| 4 | Glean | `glean-cli` | http://localhost:8080 |
| 5 | MarkItDown | `markitdown-cli` | http://localhost:8000 |
| 6 | Marquez | `marquez-cli` | http://localhost:5000 |
| 7 | PostgreSQL | `postgres-cli` | localhost:5432 |
| 8 | Hyperbook | `hyperbook-cli` | http://localhost:3000 |
| 9 | n8n | `n8n-cli` | http://localhost:5678 |
| 10 | OpenCanvas | `opencanvas-cli` | http://localhost:3000 |
| 11 | Kafka | `kafka-cli` | localhost:9092 |
| 12 | Shimmy AI | `shimmy-cli` | http://localhost:8000 |
| 13 | APISIX | `apisix-cli` | http://localhost:9180 |
| 14 | Keycloak | `keycloak-cli` | http://localhost:8080 |
| 15 | Filesystem | `fs-cli` | (local) |
| 16 | Memory | `memory-cli` | (in-memory) |
| 17 | Lean4 | `lean4-cli` | http://localhost:8080 |

## 🚀 Quick Start

### Prerequisites

1. **n8n installed**:
   ```bash
   npm install -g n8n
   ```

2. **Rust CLIs built and installed**:
   ```bash
   cd src/serviceAutomation
   for dir in *-api-client; do
     (cd $dir && cargo build --release)
     sudo cp $dir/target/release/*-cli /usr/local/bin/
   done
   ```

### Installation

The node is already included in the repository:
- **File**: `RustClients.node.ts`
- **Location**: `src/serviceAutomation/n8n-nodes/`

To use it in n8n:

```bash
# 1. Navigate to n8n custom nodes directory
cd ~/.n8n/custom

# 2. Copy the node file
cp /path/to/src/serviceAutomation/n8n-nodes/RustClients.node.ts .

# 3. Install dependencies
npm install n8n-workflow

# 4. Restart n8n
n8n start
```

## 📚 Usage Guide

### Node Configuration

The **Rust API Clients** node has 4 main parameters:

#### 1. Client (Dropdown)
Select which Rust CLI to use:
- Langflow
- Gitea
- Git
- Glean
- MarkItDown
- Marquez
- PostgreSQL
- Hyperbook
- n8n
- OpenCanvas
- Kafka
- Shimmy AI
- APISIX
- Keycloak
- Filesystem
- Memory
- Lean4

#### 2. Operation (String)
The operation to execute. Examples:
- `health` - Health check
- `list-flows` - List Langflow flows
- `list-repos` - List Gitea repositories
- `status` - Git status
- `list` - List files
- `get` - Get memory value
- etc.

#### 3. Arguments (String, Optional)
Additional space-separated arguments:
- `owner repo` - For Gitea operations
- `key` - For memory operations
- `path/to/file` - For filesystem operations

#### 4. URL (String, Optional)
Service URL. If empty, uses default for each client.

### Output Format

```json
{
  "client": "langflow",
  "operation": "health",
  "success": true,
  "stdout": "Service is healthy",
  "stderr": ""
}
```

## 🎯 Example Workflows

### Example 1: Automated Git Backup to Gitea

```
┌──────────────┐
│   Schedule   │ Trigger: Every hour
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Git, Operation: status
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Git, Operation: add, Args: .
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Git, Operation: commit, Args: "Auto backup"
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Gitea, Operation: list-repos
└──────────────┘
```

### Example 2: Document Processing Pipeline

```
┌──────────────┐
│   Webhook    │ Trigger: New document uploaded
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Filesystem, Operation: read, Args: {{$node.Webhook.path}}
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: MarkItDown, Operation: convert
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Hyperbook, Operation: create-page
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Memory, Operation: set, Args: processed-{{$json.id}}
└──────────────┘
```

### Example 3: AI-Powered Code Analysis

```
┌──────────────┐
│  Manual      │ Trigger: Button click
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Filesystem, Operation: read, Args: src/main.rs
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Glean, Operation: analyze
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: ShimmyAI, Operation: chat, Args: "Analyze this code"
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Memory, Operation: set, Args: analysis-result
└──────────────┘
```

### Example 4: Microservices Health Monitor

```
┌──────────────┐
│   Cron       │ Every 5 minutes
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: APISIX, Operation: health
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Keycloak, Operation: list-realms
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: PostgreSQL, Operation: health
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Kafka, Operation: health
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Memory, Operation: set, Args: health-status
└──────────────┘
```

### Example 5: Data Pipeline Orchestration

```
┌──────────────┐
│   Schedule   │ Daily at 2 AM
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: PostgreSQL, Operation: query, Args: "SELECT * FROM data"
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Marquez, Operation: list-datasets
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Kafka, Operation: produce, Args: analytics-topic
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Rust Clients │ Client: Memory, Operation: set, Args: pipeline-status
└──────────────┘
```

## 🛠️ Advanced Usage

### Chaining Operations

Connect multiple Rust Clients nodes together:

1. **Drag nodes** from sidebar
2. **Connect outputs** to next node inputs
3. **Pass data** using expressions: `{{$json.stdout}}`
4. **Handle errors** with IF nodes checking `success` field

### Dynamic Parameters

Use n8n expressions for dynamic values:

```javascript
// Use previous node output
Client: Git
Operation: commit
Args: {{$json["Webhook"].message}}

// Use workflow variables
Client: Filesystem
Operation: read
Args: {{$node.Start.path}}

// Combine multiple values
Client: Gitea
Operation: get-repo
Args: {{$json.owner}} {{$json.repo}}
```

### Error Handling

Each execution returns:
```json
{
  "client": "langflow",
  "operation": "health",
  "success": true/false,
  "stdout": "output here",
  "stderr": "errors here"
}
```

Add an IF node after Rust Clients:
- **Condition**: `{{$json.success}}` equals `true`
- **True branch**: Continue workflow
- **False branch**: Send notification/log error

### Parallel Execution

Use **Split in Batches** node to run multiple operations in parallel:

```
Input (Array of operations)
    ↓
Split in Batches
    ↓
Rust Clients (processes each)
    ↓
Merge (combines results)
```

## 📋 Operation Reference

### Langflow Operations
```
health          - Check service status
list-flows      - List all flows
get-flow       - Get flow details (requires Flow ID)
run-flow       - Execute flow (requires Flow ID)
```

### Gitea Operations
```
health         - Check service status
list-repos     - List repositories
create-repo    - Create repository
get-repo       - Get repo details (requires owner, repo)
list-issues    - List issues (requires owner, repo)
```

### Git Operations
```
status         - Show working tree status
init           - Initialize repository
clone          - Clone repository (requires URL)
add            - Stage changes
commit         - Commit changes (requires message)
push           - Push to remote
pull           - Pull from remote
```

### Filesystem Operations
```
list           - List directory contents
read           - Read file contents
write          - Write to file (requires content)
delete         - Delete file
copy           - Copy file (requires destination)
exists         - Check file existence
```

### Memory Operations
```
keys           - List all keys
get            - Get value (requires key)
set            - Set value (requires key, value)
delete         - Delete key (requires key)
clear          - Clear all keys
```

### APISIX Operations
```
health         - Gateway health check
list-routes    - List all routes
list-services  - List all services
list-upstreams - List upstreams
```

### Keycloak Operations
```
list-realms    - List all realms
list-users     - List users (requires realm)
list-roles     - List roles (requires realm)
list-groups    - List groups (requires realm)
```

### PostgreSQL Operations
```
health         - Database health
query          - Execute SELECT (requires SQL)
execute        - Execute statement (requires SQL)
list-databases - List databases
list-tables    - List tables
```

### Kafka Operations
```
health         - Broker health
list-topics    - List topics
produce        - Publish message (requires topic, message)
consume        - Consume messages (requires topic)
```

### Shimmy AI Operations
```
health         - Service health
chat           - Chat completion (requires prompt)
list-models    - List available models
```

(See individual CLIs for complete operation lists)

## 🔧 Troubleshooting

### Node Not Appearing

```bash
# Check n8n custom directory
ls -la ~/.n8n/custom/

# Verify file is present
cat ~/.n8n/custom/RustClients.node.ts

# Restart n8n
pkill -f n8n
n8n start
```

### CLI Not Found Errors

```bash
# Check which CLIs are installed
which langflow-cli gitea-cli git-cli

# Add to PATH if needed
export PATH=$PATH:/usr/local/bin

# Or install missing CLIs
cd src/serviceAutomation/<client>-api-client
cargo build --release
sudo cp target/release/*-cli /usr/local/bin/
```

### Execution Timeouts

The node has a 30-second timeout. For long-running operations:
1. Use async operations where available
2. Split into smaller tasks
3. Increase timeout in node code if needed

### Permission Errors

```bash
# Ensure CLIs are executable
chmod +x /usr/local/bin/*-cli

# Check CLI permissions
ls -la /usr/local/bin/*-cli
```

## 📝 Development

### Adding New Operations

Edit `RustClients.node.ts`:

```typescript
// Add new client to dropdown
{
  displayName: 'Client',
  name: 'client',
  type: 'options',
  options: [
    // ... existing clients
    { name: 'New Client', value: 'newclient' },
  ],
}
```

### Custom Node Logic

```typescript
// Modify execute function for custom behavior
async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
  // Custom pre-processing
  const client = this.getNodeParameter('client', i) as string;
  
  // Custom CLI building
  let command = `${client}-cli`;
  
  // Custom post-processing
  const result = await execAsync(command);
  return processCustomOutput(result);
}
```

### Testing Nodes

1. Save changes to `RustClients.node.ts`
2. Restart n8n
3. Create test workflow
4. Add Rust Clients node
5. Configure and execute
6. Check output

## 🎉 Summary

**Unified n8n Integration**
- ✅ 1 flexible node for all 17 clients
- ✅ Dropdown client selection
- ✅ Dynamic operation specification
- ✅ Error handling with success flag
- ✅ Comprehensive documentation
- ✅ 5 example workflows

**Ready for Production**
- Simple installation
- Consistent interface
- Error handling
- Timeout protection
- Extensive examples

## 📚 Additional Resources

- **Main README**: `../README.md`
- **Visual Workflow Guide**: `../VISUAL_WORKFLOW_INTEGRATION.md`
- **Langflow Components**: `../langflow-components/README.md`
- **Individual CLI docs**: Check each `*-api-client/` directory

---

**🎯 All 17 Rust clients accessible through n8n!**

**Happy workflow building! 🚀**
