# Memgraph API Client

Rust client library and CLI for interacting with Memgraph graph database.

## Features

- Execute Cypher queries with parameters
- Create nodes and edges
- Query graph schema
- Get node/edge counts
- Health monitoring
- Connection testing
- Database clearing (with safety confirmation)

## Installation

```bash
cargo build --release
```

The binary will be available at `target/release/aimo-memgraph-cli`.

## Usage

### As a Library

```rust
use memgraph_api_client::MemgraphClient;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create client
    let client = MemgraphClient::new("http://localhost:7687")?;
    
    // Execute query
    let response = client.query("MATCH (n) RETURN n LIMIT 10").await?;
    println!("Found {} nodes", response.data.len());
    
    // Create a node
    let properties = json!({
        "name": "Alice",
        "age": 30
    });
    client.create_node(&["Person"], properties).await?;
    
    // Get statistics
    let node_count = client.get_node_count().await?;
    let edge_count = client.get_edge_count().await?;
    println!("Nodes: {}, Edges: {}", node_count, edge_count);
    
    Ok(())
}
```

### CLI Usage

```bash
# Test connection
aimo-memgraph-cli --url http://localhost:7687 test

# Health check
aimo-memgraph-cli health

# Execute a query
aimo-memgraph-cli query --query "MATCH (n:Person) RETURN n.name"

# Query with parameters
aimo-memgraph-cli query \
  --query "MATCH (n:Person {name: $name}) RETURN n" \
  --params '{"name": "Alice"}'

# Get database schema
aimo-memgraph-cli schema

# Get counts
aimo-memgraph-cli node-count
aimo-memgraph-cli edge-count

# Create a node
aimo-memgraph-cli create-node \
  --labels "Person,Developer" \
  --properties '{"name": "Bob", "age": 25}'

# Create an edge
aimo-memgraph-cli create-edge \
  --from "1" \
  --to "2" \
  --edge-type "KNOWS" \
  --properties '{"since": 2020}'

# Clear database (requires confirmation)
aimo-memgraph-cli clear --confirm
```

## API Reference

### MemgraphClient

Main client for interacting with Memgraph.

**Methods:**

- `new(base_url)` - Create a new client
- `execute_query(query, parameters)` - Execute Cypher query with optional parameters
- `query(query)` - Execute simple query without parameters
- `create_node(labels, properties)` - Create a node with labels and properties
- `create_edge(from_id, to_id, edge_type, properties)` - Create an edge between nodes
- `get_schema()` - Get database schema
- `get_node_count()` - Get total number of nodes
- `get_edge_count()` - Get total number of edges
- `clear_database()` - Delete all data (use with caution!)
- `health_check()` - Check service health
- `test_connection()` - Test database connection

### Data Structures

**QueryResponse:**
```rust
pub struct QueryResponse {
    pub columns: Vec<String>,
    pub data: Vec<Vec<serde_json::Value>>,
    pub metadata: Option<QueryMetadata>,
}
```

**Node:**
```rust
pub struct Node {
    pub id: Option<String>,
    pub labels: Vec<String>,
    pub properties: serde_json::Map<String, serde_json::Value>,
}
```

**Edge:**
```rust
pub struct Edge {
    pub id: Option<String>,
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
}
```

## Configuration

Default Memgraph URL: `http://localhost:7687`

Override with `--url` flag:
```bash
aimo-memgraph-cli --url http://memgraph:7687 health
```

## Development

Run tests:
```bash
cargo test
```

Build for release:
```bash
cargo build --release
```

## Integration

This client integrates with:
- AIMO solver crates for graph-based reasoning
- Langflow components for workflow automation
- Knowledge graph construction pipelines
- Relationship discovery systems

## License

MIT License - See LICENSE file for details
