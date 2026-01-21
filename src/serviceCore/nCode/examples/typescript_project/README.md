# TypeScript Project Example - Qdrant Integration

This example demonstrates how to index a TypeScript project and load it into Qdrant for semantic code search.

## 📋 Overview

This example includes:
- A sample TypeScript project with classes, interfaces, and functions
- SCIP index generation using `scip-typescript`
- Loading the index into nCode server
- Exporting to Qdrant for vector-based semantic search
- Example queries to find symbols using natural language

## 🏗️ Project Structure

```
typescript_project/
├── README.md                 # This file
├── package.json              # Project dependencies
├── tsconfig.json             # TypeScript configuration
├── run_example.sh            # Automated demo script
├── query_qdrant.py           # Python script to query Qdrant
└── src/
    ├── models/
    │   ├── user.ts          # User model with interface
    │   └── product.ts       # Product model
    ├── services/
    │   ├── auth.ts          # Authentication service
    │   └── database.ts      # Database connection
    ├── utils/
    │   └── helpers.ts       # Utility functions
    └── index.ts             # Main entry point
```

## 🚀 Quick Start

### Automated Demo

Run the complete example with one command:

```bash
./run_example.sh
```

This script will:
1. Install dependencies (`npm install`)
2. Generate SCIP index (`npx @sourcegraph/scip-typescript index`)
3. Load index into nCode server
4. Export to Qdrant
5. Perform example semantic search queries
6. Display results

### Manual Steps

#### 1. Install Dependencies

```bash
npm install
npm install -g @sourcegraph/scip-typescript
```

#### 2. Generate SCIP Index

```bash
npx @sourcegraph/scip-typescript index
```

This creates `index.scip` containing all symbol information.

#### 3. Load into nCode Server

```bash
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d '{"path": "index.scip"}'
```

#### 4. Export to Qdrant

```bash
cd ../../loaders
python -c "
from scip_parser import SCIPParser
from qdrant_loader import QdrantLoader

# Parse SCIP index
parser = SCIPParser('../examples/typescript_project/index.scip')
documents = parser.parse_documents()

# Load to Qdrant
loader = QdrantLoader(
    host='localhost',
    port=6333,
    collection_name='typescript_example'
)
loader.load_documents(documents)
print(f'Loaded {len(documents)} documents to Qdrant')
"
```

Or use the database loader script:

```bash
cd ../..
python scripts/load_to_databases.py \
  examples/typescript_project/index.scip \
  --qdrant \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --qdrant-collection typescript_example
```

#### 5. Query Semantic Search

```bash
python query_qdrant.py
```

## 🔍 Example Queries

### Find Classes with Constructors

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

results = client.search(
    collection_name="typescript_example",
    query_text="class with constructor and private properties",
    limit=5
)

for result in results:
    print(f"{result.payload['symbol']} - {result.payload['file']}")
```

### Find Authentication Functions

```python
results = client.search(
    collection_name="typescript_example",
    query_text="functions that verify user credentials",
    limit=5
)
```

### Find Database Operations

```python
results = client.search(
    collection_name="typescript_example",
    query_text="methods that connect to database",
    limit=5
)
```

## 📊 Expected Output

```
🚀 TypeScript Example - Qdrant Integration
==========================================

Step 1: Installing dependencies...
✓ Dependencies installed

Step 2: Generating SCIP index...
Indexing TypeScript project...
Found 6 files, 45 symbols
✓ Generated index.scip (123 KB)

Step 3: Loading to nCode server...
✓ Index loaded successfully

Step 4: Exporting to Qdrant...
Creating collection 'typescript_example'...
Embedding 45 symbols...
✓ Loaded 45 vectors to Qdrant

Step 5: Running semantic search queries...

Query: "class with constructor"
Results:
  1. UserService#constructor() (score: 0.94)
     File: src/services/auth.ts
     
  2. DatabaseConnection#constructor() (score: 0.89)
     File: src/services/database.ts
     
  3. Product#constructor() (score: 0.85)
     File: src/models/product.ts

Query: "functions that validate input"
Results:
  1. validateEmail() (score: 0.91)
     File: src/utils/helpers.ts
     
  2. validatePassword() (score: 0.88)
     File: src/utils/helpers.ts

✅ Example completed successfully!
```

## 🎯 Key Concepts

### SCIP Indexing
- `scip-typescript` analyzes TypeScript/JavaScript code
- Extracts symbols (classes, functions, interfaces, etc.)
- Captures relationships (implements, extends, calls)
- Stores in SCIP protobuf format

### Vector Embeddings
- Each symbol gets converted to a vector embedding
- Embeddings capture semantic meaning
- Similar code concepts have similar vectors
- Enables natural language search

### Semantic Search
- Query with plain English descriptions
- Returns most relevant code symbols
- Ranked by similarity score
- Much more flexible than text search

## 🐛 Troubleshooting

### Error: `scip-typescript` not found

```bash
npm install -g @sourcegraph/scip-typescript
```

### Error: nCode server not responding

```bash
# Check server status
curl http://localhost:18003/health

# Start server if needed
cd ../../
./scripts/start.sh
```

### Error: Qdrant connection refused

```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant

# Or check if already running
docker ps | grep qdrant
```

### Error: Cannot find module errors

```bash
# Install dependencies
npm install

# Check TypeScript configuration
cat tsconfig.json
```

## 📚 Next Steps

1. **Modify the code**: Add your own TypeScript files and re-index
2. **Try different queries**: Experiment with various natural language queries
3. **Explore filters**: Filter by file, language, or symbol kind
4. **Compare with text search**: See how semantic search differs from grep/ripgrep

## 🔗 Related Examples

- [Python Project Example](../python_project/) - Memgraph integration
- [Marquez Lineage Example](../marquez_lineage/) - Track indexing runs
- [Jupyter Notebooks](../notebooks/) - Interactive tutorials

## 📖 References

- [scip-typescript Documentation](https://github.com/sourcegraph/scip-typescript)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [nCode Database Integration Guide](../../docs/DATABASE_INTEGRATION.md)
- [SCIP Protocol Specification](https://github.com/sourcegraph/scip)

---

**Last Updated:** 2026-01-17  
**Version:** 1.0
