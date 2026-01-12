# MCP Servers - Layer Core

Model Context Protocol (MCP) servers provide tool and resource capabilities to the AI Nucleus platform.

## Installed MCP Servers

### 1. **Filesystem MCP** (`./filesystem`)
- **Purpose:** File system operations and document access
- **Use Cases:**
  - Read/write files for invoice processing
  - Access documentation and templates
  - Code analysis and manipulation
- **Integration:** Connect to backend API for document workflows

### 2. **Git MCP** (`./git`)
- **Purpose:** Git repository operations
- **Use Cases:**
  - Interface with Gitea service
  - Version control for workflows and configs
  - Repository management
- **Integration:** Works with ai_nucleus_gitea service

### 3. **Memory MCP** (`./memory`)
- **Purpose:** Persistent agent memory
- **Use Cases:**
  - Store conversation context
  - Remember user preferences
  - Maintain session state
- **Integration:** Can use Dragonfly or Postgres for storage

### 4. **Postgres MCP** (`./postgres`)
- **Purpose:** Database operations
- **Use Cases:**
  - Query all 5 Postgres databases
  - Data analysis and reporting
  - Schema management
- **Integration:** Connect to:
  - ai_nucleus_gitea_db
  - ai_nucleus_keycloak_db
  - ai_nucleus_n8n_db
  - ai_nucleus_marquez_db
  - ai_nucleus_registry_db

## Integration Points

### Backend API Integration
The backend service can expose MCP tools through its API:
- File operations via Filesystem MCP
- Database queries via Postgres MCP
- Git operations via Git MCP

### n8n Workflows
MCP tools can be wrapped as n8n custom nodes:
- Filesystem operations in document workflows
- Database queries in automation flows
- Git operations for deployment workflows

### Langflow Integration
MCP resources can be used as Langflow components:
- Memory for conversation persistence
- Postgres for data retrieval
- Filesystem for document loading

## Configuration

MCP servers can be configured via environment variables in docker-compose.yml or integrated directly into the backend service.

## Source
- Repository: https://github.com/modelcontextprotocol/servers
- License: MIT
- Version: Latest (Jan 2026)