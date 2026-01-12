# 🎨 Langflow Components for Rust API Clients

**Complete visual workflow integration for all 17 Rust API clients**

## 📦 What's Included

### 17 Production-Ready Components

| # | Component | Icon | Purpose |
|---|-----------|------|---------|
| 1 | Langflow Operations | 🔄 | Manage Langflow workflows |
| 2 | Gitea Git Hosting | 🦊 | Git repository management |
| 3 | Git Operations | 🌿 | Version control commands |
| 4 | Filesystem Operations | 📁 | File read/write/manage |
| 5 | Memory Cache | 🧠 | In-memory key-value storage |
| 6 | APISIX API Gateway | 🚪 | API routing and services |
| 7 | Keycloak Authentication | 🔐 | User and realm management |
| 8 | Glean Code Intelligence | 🔍 | Code search and analysis |
| 9 | MarkItDown Converter | 📄 | Document to Markdown |
| 10 | Marquez Data Lineage | 📊 | Pipeline tracking |
| 11 | PostgreSQL Database | 🐘 | SQL queries and tables |
| 12 | Hyperbook Documentation | 📚 | Interactive docs |
| 13 | n8n Workflow Automation | ⚡ | Workflow orchestration |
| 14 | OpenCanvas Collaboration | 🎨 | Real-time editing |
| 15 | Kafka Messaging | 📨 | Message streaming |
| 16 | Shimmy-AI Local Inference | 🤖 | Local AI models |
| 17 | Lean4 Theorem Prover | 🔬 | Formal verification |

## 🚀 Quick Start

### Installation

```bash
# 1. Navigate to langflow-components directory
cd src/serviceAutomation/langflow-components

# 2. Run installation script
chmod +x install.sh
./install.sh

# 3. Restart Langflow
langflow run
```

### Verification

1. Open Langflow UI: http://localhost:7860
2. Click "Custom" category in left sidebar
3. You should see all 17 components listed

## 📚 Component Documentation

### 1. Langflow Operations 🔄

**Purpose**: Manage Langflow workflows programmatically

**Operations**:
- `health` - Check Langflow service status
- `list-flows` - List all available flows
- `get-flow` - Get specific flow details
- `run-flow` - Execute a flow

**Parameters**:
- URL: Langflow instance URL (default: http://localhost:7860)
- Flow ID: Required for get-flow and run-flow operations

**Example Use Cases**:
- Health monitoring dashboards
- Flow orchestration
- Automated flow execution
- CI/CD pipeline integration

---

### 2. Gitea Git Hosting 🦊

**Purpose**: Manage Git repositories via Gitea API

**Operations**:
- `health` - Check Gitea status
- `list-repos` - List repositories
- `create-repo` - Create new repository
- `get-repo` - Get repository details
- `list-issues` - List issues
- `list-prs` - List pull requests

**Parameters**:
- URL: Gitea instance URL (default: http://localhost:3000)
- Owner: Repository owner username
- Repo: Repository name

**Example Use Cases**:
- Automated repository creation
- Issue tracking automation
- PR status monitoring
- Repository health checks

---

### 3. Git Operations 🌿

**Purpose**: Execute Git version control commands

**Operations**:
- `status` - Show working tree status
- `init` - Initialize repository
- `clone` - Clone repository
- `add` - Stage changes
- `commit` - Commit changes
- `push` - Push to remote
- `pull` - Pull from remote
- `branch` - Branch operations

**Parameters**:
- Path: Repository path (default: .)
- Message: Commit message (for commit)
- URL: Clone URL (for clone)

**Example Use Cases**:
- Automated commits
- CI/CD workflows
- Backup automation
- Multi-repo synchronization

---

### 4. Filesystem Operations 📁

**Purpose**: Read, write, and manage files

**Operations**:
- `list` - List directory contents
- `read` - Read file contents
- `write` - Write to file
- `delete` - Delete file
- `copy` - Copy file
- `exists` - Check file existence
- `mkdir` - Create directory

**Parameters**:
- Path: File/directory path
- Content: File content (for write)
- Dest: Destination path (for copy)

**Example Use Cases**:
- Configuration management
- Log file processing
- Report generation
- Data migration

---

### 5. Memory Cache 🧠

**Purpose**: In-memory key-value storage with TTL

**Operations**:
- `keys` - List all keys
- `get` - Get value
- `set` - Set value
- `set-ttl` - Set value with TTL
- `delete` - Delete key
- `clear` - Clear all keys
- `stats` - Cache statistics

**Parameters**:
- Key: Cache key
- Value: Cache value
- TTL: Time-to-live in seconds

**Example Use Cases**:
- Session management
- Temporary data storage
- Rate limiting
- Workflow state management

---

### 6. APISIX API Gateway 🚪

**Purpose**: Manage API routes and services

**Operations**:
- `health` - Gateway health check
- `list-routes` - List all routes
- `list-services` - List all services
- `list-upstreams` - List upstreams
- `create-route` - Create new route
- `delete-route` - Delete route

**Parameters**:
- URL: APISIX admin URL (default: http://localhost:9180)
- Route ID: For route operations

**Example Use Cases**:
- Dynamic routing
- API composition
- Traffic management
- Service health monitoring

---

### 7. Keycloak Authentication 🔐

**Purpose**: Manage users, roles, and realms

**Operations**:
- `list-realms` - List all realms
- `list-users` - List users in realm
- `list-roles` - List roles
- `list-groups` - List groups
- `create-user` - Create new user
- `delete-user` - Delete user

**Parameters**:
- URL: Keycloak URL (default: http://localhost:8080)
- Realm: Realm name (default: master)
- Username: For user operations

**Example Use Cases**:
- User provisioning
- Access control automation
- Security auditing
- Multi-tenant management

---

### 8. Glean Code Intelligence 🔍

**Purpose**: Search and analyze code

**Operations**:
- `health` - Service health
- `search` - Search code
- `get-definition` - Get symbol definition
- `find-references` - Find references
- `analyze` - Analyze file

**Parameters**:
- URL: Glean URL (default: http://localhost:8080)
- Query: Search query
- File: File path for analysis

**Example Use Cases**:
- Code search automation
- Refactoring assistance
- Documentation generation
- Code quality analysis

---

### 9. MarkItDown Converter 📄

**Purpose**: Convert documents to Markdown

**Operations**:
- `health` - Service status
- `convert` - Convert file
- `convert-file` - Convert with output
- `list-formats` - Supported formats

**Parameters**:
- URL: Service URL (default: http://localhost:8000)
- Input File: File to convert
- Output File: Output path

**Example Use Cases**:
- Documentation conversion
- Report generation
- Content migration
- Format standardization

---

### 10. Marquez Data Lineage 📊

**Purpose**: Track data pipelines and lineage

**Operations**:
- `health` - Service health
- `list-namespaces` - List namespaces
- `list-datasets` - List datasets
- `list-jobs` - List jobs
- `get-lineage` - Get data lineage
- `create-namespace` - Create namespace

**Parameters**:
- URL: Marquez URL (default: http://localhost:5000)
- Namespace: Namespace name
- Dataset: Dataset name

**Example Use Cases**:
- Pipeline monitoring
- Data governance
- Impact analysis
- Compliance tracking

---

### 11. PostgreSQL Database 🐘

**Purpose**: Execute SQL queries and manage databases

**Operations**:
- `health` - Database health
- `query` - Execute SELECT query
- `execute` - Execute SQL statement
- `list-databases` - List databases
- `list-tables` - List tables

**Parameters**:
- Host: Database host (default: localhost)
- Port: Database port (default: 5432)
- Database: Database name
- SQL: SQL query/statement

**Example Use Cases**:
- Data extraction
- Report generation
- Database migration
- Health monitoring

---

### 12. Hyperbook Documentation 📚

**Purpose**: Manage interactive documentation

**Operations**:
- `health` - Service health
- `list-books` - List books
- `get-book` - Get book details
- `create-page` - Create page
- `update-page` - Update page

**Parameters**:
- URL: Hyperbook URL (default: http://localhost:3000)
- Book ID: Book identifier
- Page ID: Page identifier
- Content: Page content

**Example Use Cases**:
- Documentation automation
- Content publishing
- Version management
- Interactive tutorials

---

### 13. n8n Workflow Automation ⚡

**Purpose**: Manage n8n workflows

**Operations**:
- `health` - Service health
- `list-workflows` - List workflows
- `get-workflow` - Get workflow details
- `activate` - Activate workflow
- `deactivate` - Deactivate workflow
- `execute` - Execute workflow

**Parameters**:
- URL: n8n URL (default: http://localhost:5678)
- Workflow ID: Workflow identifier

**Example Use Cases**:
- Workflow orchestration
- Automated activation
- Workflow monitoring
- Integration testing

---

### 14. OpenCanvas Collaboration 🎨

**Purpose**: Real-time collaborative editing

**Operations**:
- `health` - Service health
- `list-canvases` - List canvases
- `create-canvas` - Create canvas
- `get-canvas` - Get canvas
- `update-canvas` - Update canvas
- `delete-canvas` - Delete canvas

**Parameters**:
- URL: OpenCanvas URL (default: http://localhost:3000)
- Canvas ID: Canvas identifier
- Title: Canvas title
- Content: Canvas content

**Example Use Cases**:
- Collaborative design
- Real-time brainstorming
- Content creation
- Team collaboration

---

### 15. Kafka Messaging 📨

**Purpose**: Publish and consume Kafka messages

**Operations**:
- `health` - Broker health
- `list-topics` - List topics
- `create-topic` - Create topic
- `produce` - Produce message
- `consume` - Consume messages

**Parameters**:
- Broker: Kafka broker (default: localhost:9092)
- Topic: Topic name
- Message: Message content

**Example Use Cases**:
- Event streaming
- Message queue automation
- Data pipeline integration
- Real-time processing

---

### 16. Shimmy-AI Local Inference 🤖

**Purpose**: OpenAI-compatible local AI inference

**Operations**:
- `health` - Service health
- `chat` - Chat completion
- `list-models` - List models

**Parameters**:
- URL: Shimmy-AI URL (default: http://localhost:8000)
- Prompt: Chat prompt
- Model: Model name

**Example Use Cases**:
- AI-powered workflows
- Content generation
- Data analysis
- Automated responses

---

### 17. Lean4 Theorem Prover 🔬

**Purpose**: Formal verification with Lean4

**Operations**:
- `health` - Service health
- `check` - Check proof
- `prove` - Prove theorem
- `info` - Get information

**Parameters**:
- URL: Lean4 URL (default: http://localhost:8080)
- File: Lean file path
- Theorem: Theorem name

**Example Use Cases**:
- Formal verification
- Proof automation
- Mathematical validation
- Code correctness

---

## 🎯 Example Workflows

### Workflow 1: Automated Git Backup

```
┌─────────────┐
│  Filesystem │ List modified files
│  Operations │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│     Git     │ Add & Commit changes
│  Operations │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Gitea    │ Push to remote
│ Git Hosting │
└─────────────┘
```

### Workflow 2: Document Processing Pipeline

```
┌─────────────┐
│  Filesystem │ Read source document
│  Operations │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│ MarkItDown  │ Convert to Markdown
│  Converter  │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Hyperbook  │ Publish to docs
│Documentation│
└─────────────┘
```

### Workflow 3: AI-Powered Code Analysis

```
┌─────────────┐
│  Filesystem │ Read code file
│  Operations │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Glean    │ Analyze code
│    Code     │
│Intelligence │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Shimmy-AI  │ Generate insights
│Local Inference│
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Memory   │ Cache results
│    Cache    │
└─────────────┘
```

### Workflow 4: Data Pipeline Monitoring

```
┌─────────────┐
│  PostgreSQL │ Query data
│  Database   │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Marquez   │ Track lineage
│Data Lineage │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Kafka    │ Publish metrics
│  Messaging  │
└─────────────┘
```

### Workflow 5: Microservices Health Check

```
┌─────────────┐
│   APISIX    │ Check gateway
│ API Gateway │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Keycloak   │ Check auth
│Authentication│
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  PostgreSQL │ Check database
│  Database   │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Memory   │ Store status
│    Cache    │
└─────────────┘
```

## 🛠️ Advanced Usage

### Chaining Operations

Components can be chained together by connecting outputs to inputs:

1. **Drag components** onto canvas
2. **Connect output ports** to input ports
3. **Configure parameters** for each component
4. **Run workflow** to execute

### Error Handling

Each component returns:
```json
{
  "success": true/false,
  "stdout": "...",
  "stderr": "...",
  "returncode": 0
}
```

Use conditional branches based on `success` field.

### Best Practices

1. **Health Checks First**: Always start workflows with health checks
2. **Use Memory Cache**: Cache frequently accessed data
3. **Chain Operations**: Break complex tasks into small components
4. **Error Handling**: Always handle component failures
5. **Test Individually**: Test each component before chaining

## 🔧 Troubleshooting

### Components Not Appearing

```bash
# Check installation
ls -la ~/.langflow/components/rust_clients.py

# Verify Langflow is running
ps aux | grep langflow

# Restart Langflow
pkill -f langflow
langflow run
```

### CLI Not Found Errors

```bash
# Check which CLIs are installed
which langflow-cli gitea-cli git-cli

# Install missing CLIs
cd src/serviceAutomation
for dir in *-api-client; do
  (cd $dir && cargo build --release)
  sudo cp $dir/target/release/*-cli /usr/local/bin/
done
```

### Connection Errors

- Verify service URLs are correct
- Check services are running
- Test with direct CLI commands first

## 📝 Development

### Adding Custom Operations

Edit `rust_clients.py` to add new operations:

```python
class MyComponent(RustClientComponent):
    inputs = [
        DropdownInput(
            name="operation",
            options=["new-op1", "new-op2"],
        ),
        # Add more inputs
    ]
    
    def execute(self) -> Data:
        # Custom logic
        result = self.execute_cli("my-cli", args)
        return Data(data=result)
```

### Testing Components

1. Save changes to `rust_clients.py`
2. Restart Langflow
3. Drag component onto canvas
4. Configure and run

## 🎉 Summary

**17 Production-Ready Components**
- ✅ Complete Rust CLI integration
- ✅ Visual drag-and-drop interface
- ✅ Error handling and validation
- ✅ Comprehensive documentation
- ✅ Example workflows included

**Ready to build amazing workflows!**

For more information, see:
- [Main README](../README.md)
- [Visual Workflow Integration](../VISUAL_WORKFLOW_INTEGRATION.md)
- Individual Rust client documentation
