# nCode Examples & Tutorials

This directory contains practical examples and tutorials for using nCode to index projects and integrate with Qdrant, Memgraph, and Marquez databases.

## 📁 Directory Structure

```
examples/
├── README.md                          # This file
├── typescript_project/                # TypeScript indexing example
│   ├── README.md
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   └── run_example.sh
├── python_project/                    # Python indexing example
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.py
│   ├── src/
│   └── run_example.sh
├── marquez_lineage/                   # Marquez lineage tracking example
│   ├── README.md
│   ├── track_indexing.py
│   └── query_lineage.py
├── notebooks/                         # Jupyter notebook examples
│   ├── 01_basic_indexing.ipynb
│   ├── 02_qdrant_semantic_search.ipynb
│   ├── 03_memgraph_graph_queries.ipynb
│   └── 04_marquez_lineage.ipynb
└── tutorials/                         # Step-by-step tutorials
    ├── typescript_tutorial.md
    ├── python_tutorial.md
    ├── java_tutorial.md
    ├── rust_tutorial.md
    ├── go_tutorial.md
    └── data_languages_tutorial.md
```

## 🚀 Quick Start

### 1. TypeScript Project Example

Index a TypeScript project and load it into Qdrant for semantic search:

```bash
cd examples/typescript_project
./run_example.sh
```

This example demonstrates:
- Setting up a TypeScript project with types
- Generating SCIP index with scip-typescript
- Loading index into nCode server
- Exporting to Qdrant for semantic search
- Querying symbol definitions and references

### 2. Python Project Example

Index a Python project and query the code graph in Memgraph:

```bash
cd examples/python_project
./run_example.sh
```

This example demonstrates:
- Creating a Python package with modules
- Generating SCIP index with scip-python
- Loading index into nCode server
- Exporting to Memgraph for graph queries
- Finding implementations and call graphs

### 3. Marquez Lineage Example

Track code indexing runs and query lineage:

```bash
cd examples/marquez_lineage
python track_indexing.py
python query_lineage.py
```

This example demonstrates:
- Tracking SCIP indexing as OpenLineage events
- Recording source file → SCIP index lineage
- Querying lineage graph through Marquez API
- Visualizing data flow and dependencies

## 📚 Tutorials

### Language-Specific Tutorials

Each tutorial provides step-by-step instructions for indexing and querying code:

1. **TypeScript/JavaScript** - [typescript_tutorial.md](tutorials/typescript_tutorial.md)
   - Project setup and dependencies
   - Indexing with scip-typescript
   - Type information and navigation
   
2. **Python** - [python_tutorial.md](tutorials/python_tutorial.md)
   - Virtual environment setup
   - Indexing with scip-python
   - Module and class analysis

3. **Java** - [java_tutorial.md](tutorials/java_tutorial.md)
   - Maven/Gradle project setup
   - Indexing with scip-java
   - Class hierarchy queries

4. **Rust** - [rust_tutorial.md](tutorials/rust_tutorial.md)
   - Cargo project indexing
   - Trait and impl analysis
   - Cross-crate references

5. **Go** - [go_tutorial.md](tutorials/go_tutorial.md)
   - Module-based project indexing
   - Package dependency analysis
   - Interface implementation finding

6. **Data Languages** - [data_languages_tutorial.md](tutorials/data_languages_tutorial.md)
   - JSON, XML, YAML, SQL indexing
   - Schema extraction
   - Data structure navigation

## 📓 Jupyter Notebooks

Interactive notebooks for learning nCode:

### 01_basic_indexing.ipynb
- Load and query SCIP indexes
- Find definitions and references
- Navigate code structure

### 02_qdrant_semantic_search.ipynb
- Load code into Qdrant
- Perform semantic search with natural language
- Filter by language and symbol type

### 03_memgraph_graph_queries.ipynb
- Load code graph into Memgraph
- Write Cypher queries
- Visualize code relationships

### 04_marquez_lineage.ipynb
- Track indexing pipeline
- Query data lineage
- Analyze code dependencies

## 🎯 Use Cases

### Use Case 1: Find All Usages of an API

```python
# Query nCode API
response = requests.post(
    "http://localhost:18003/v1/references",
    json={"symbol": "mylib.api.MyClass#myMethod()."}
)

# Results include all files and locations
for ref in response.json()["references"]:
    print(f"{ref['file']}:{ref['line']} - {ref['snippet']}")
```

### Use Case 2: Semantic Code Search

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Search with natural language
results = client.search(
    collection_name="code_symbols",
    query_text="functions that parse JSON data",
    limit=10
)

for result in results:
    print(f"{result.payload['symbol']} - Score: {result.score}")
```

### Use Case 3: Analyze Dependencies

```cypher
// Find all dependencies of a module (Memgraph Cypher)
MATCH (src:Symbol {name: "my_module"})-[:REFERENCES*]->(dep:Symbol)
WHERE src.kind = "module" AND dep.kind = "module"
RETURN DISTINCT dep.name, dep.file
```

## 🛠️ Prerequisites

### Required Software

1. **nCode Server** (running on localhost:18003)
   ```bash
   cd src/serviceCore/nCode
   ./scripts/start.sh
   ```

2. **Language Indexers** (install as needed)
   ```bash
   # TypeScript
   npm install -g @sourcegraph/scip-typescript
   
   # Python
   pip install scip-python
   
   # Java
   # See https://sourcegraph.github.io/scip-java/
   
   # Rust
   rustup component add rust-analyzer
   
   # Go
   go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
   ```

3. **Database Services** (optional, for advanced examples)
   ```bash
   # Start with Docker Compose (from project root)
   docker-compose up -d qdrant memgraph marquez
   ```

### Python Dependencies

```bash
# Install Python dependencies for examples
pip install -r requirements.txt
```

Contents of requirements.txt:
```
requests>=2.31.0
qdrant-client>=1.7.0
gqlalchemy>=1.4.0
openlineage-python>=1.0.0
jupyter>=1.0.0
matplotlib>=3.7.0
networkx>=3.1
```

## 🏃 Running Examples

### All-in-One Demo

Run all examples in sequence:

```bash
# From examples directory
./run_all_examples.sh
```

This script will:
1. Start nCode server (if not running)
2. Run TypeScript example → Qdrant
3. Run Python example → Memgraph
4. Run Marquez lineage tracking
5. Display results and statistics

### Individual Examples

```bash
# TypeScript → Qdrant
cd typescript_project && ./run_example.sh

# Python → Memgraph
cd python_project && ./run_example.sh

# Marquez lineage
cd marquez_lineage && python track_indexing.py
```

## 📊 Expected Results

### TypeScript Example Output

```
✓ Project indexed: 15 files, 243 symbols
✓ Loaded to nCode server
✓ Exported to Qdrant: 243 vectors
✓ Semantic search test:
  Query: "class with constructor"
  Results:
    - UserService#constructor() (score: 0.94)
    - DatabaseConnection#constructor() (score: 0.89)
    - ApiClient#constructor() (score: 0.86)
```

### Python Example Output

```
✓ Project indexed: 8 modules, 156 symbols
✓ Loaded to nCode server
✓ Exported to Memgraph: 156 nodes, 342 relationships
✓ Graph query test:
  Finding implementations of 'BaseRepository':
    - UserRepository (src/repositories/user.py)
    - ProductRepository (src/repositories/product.py)
    - OrderRepository (src/repositories/order.py)
```

### Marquez Example Output

```
✓ Tracked indexing run: run-2026-01-17-19:45:32
✓ Recorded lineage: 23 source files → index.scip
✓ Lineage query results:
  Source files indexed:
    - src/main.py
    - src/models/user.py
    - src/services/auth.py
    ... (20 more files)
```

## 🐛 Troubleshooting

### Server Not Running

```bash
# Check if nCode server is running
curl http://localhost:18003/health

# If not, start it
cd ../../
./scripts/start.sh
```

### Database Connection Issues

```bash
# Check database status
docker ps | grep -E "qdrant|memgraph|marquez"

# Restart databases if needed
docker-compose restart qdrant memgraph marquez
```

### Indexer Not Found

```bash
# Install missing indexer
npm install -g @sourcegraph/scip-typescript  # TypeScript
pip install scip-python                       # Python
```

### Common Errors

See [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) for detailed solutions.

## 📖 Further Reading

- [nCode README](../README.md) - Project overview
- [Architecture Guide](../docs/ARCHITECTURE.md) - System design
- [API Reference](../docs/API.md) - HTTP API documentation
- [Database Integration](../docs/DATABASE_INTEGRATION.md) - Database setup
- [SCIP Protocol](https://github.com/sourcegraph/scip) - SCIP specification

## 🤝 Contributing Examples

Have an interesting use case? Contribute an example!

1. Create a new directory under `examples/`
2. Add a `README.md` explaining the example
3. Include runnable scripts
4. Add test data if needed
5. Submit a pull request

## 📝 License

These examples are part of the nCode project and are licensed under the MIT License.

---

**Last Updated:** 2026-01-17  
**Version:** 1.0  
**Status:** Day 3 Complete ✅
