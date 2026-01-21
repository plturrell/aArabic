# TypeScript/JavaScript Indexing Tutorial

Complete guide to indexing TypeScript and JavaScript projects with nCode and scip-typescript.

## 📋 Prerequisites

- Node.js 16+ and npm
- TypeScript project with `tsconfig.json`
- nCode server running on localhost:18003

## 🚀 Quick Start

### Step 1: Install scip-typescript

```bash
# Global installation (recommended)
npm install -g @sourcegraph/scip-typescript

# Or use with npx
npx @sourcegraph/scip-typescript --version
```

### Step 2: Set Up Your TypeScript Project

```bash
# Create a new project (if needed)
mkdir my-typescript-project
cd my-typescript-project
npm init -y

# Install TypeScript
npm install --save-dev typescript @types/node

# Create tsconfig.json
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "declaration": true,
    "sourceMap": true
  },
  "include": ["src/**/*"]
}
EOF
```

### Step 3: Create Sample TypeScript Code

```typescript
// src/models/user.ts
export interface User {
  id: string;
  name: string;
  email: string;
}

export class UserService {
  private users: Map<string, User> = new Map();

  async getUser(id: string): Promise<User | null> {
    return this.users.get(id) || null;
  }

  async createUser(user: User): Promise<void> {
    this.users.set(user.id, user);
  }
}
```

### Step 4: Generate SCIP Index

```bash
# Using scip-typescript
scip-typescript index

# Or with npx
npx @sourcegraph/scip-typescript index

# Specify output location
scip-typescript index --output=index.scip
```

This will:
- Analyze all TypeScript files in your project
- Extract symbols (classes, functions, interfaces, etc.)
- Capture type information and relationships
- Generate `index.scip` file

### Step 5: Load Index into nCode

```bash
# Get absolute path
SCIP_PATH="$(pwd)/index.scip"

# Load into nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$SCIP_PATH\"}"
```

### Step 6: Query Code Intelligence

```bash
# Find definition of UserService class
curl -X POST http://localhost:18003/v1/definition \
  -H "Content-Type: application/json" \
  -d '{
    "file": "src/models/user.ts",
    "line": 8,
    "character": 13
  }'

# Find all references to getUser method
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{"symbol": "my_project.UserService#getUser()."}'

# Get symbols in a file
curl -X POST http://localhost:18003/v1/symbols \
  -H "Content-Type: application/json" \
  -d '{"file": "src/models/user.ts"}'
```

## 🔧 Advanced Configuration

### Customize scip-typescript

Create `.scip-typescript.json`:

```json
{
  "projectRoot": ".",
  "output": "index.scip",
  "inferTsconfig": true,
  "yarnWorkspaces": false,
  "pnpmWorkspaces": false,
  "progressBar": true
}
```

### Handle Monorepos

```bash
# Index specific package
cd packages/my-package
scip-typescript index

# Or index entire workspace
scip-typescript index --yarn-workspaces
```

### Include/Exclude Files

Modify `tsconfig.json`:

```json
{
  "include": ["src/**/*"],
  "exclude": [
    "node_modules",
    "**/*.test.ts",
    "**/*.spec.ts",
    "dist"
  ]
}
```

## 💾 Export to Databases

### Qdrant (Semantic Search)

```bash
# Export to Qdrant
python scripts/load_to_databases.py index.scip \
  --qdrant \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --qdrant-collection my_project

# Query with Python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
results = client.search(
    collection_name="my_project",
    query_text="functions that handle user authentication",
    limit=10
)
```

### Memgraph (Graph Queries)

```bash
# Export to Memgraph
python scripts/load_to_databases.py index.scip \
  --memgraph \
  --memgraph-host localhost \
  --memgraph-port 7687

# Query with Cypher
MATCH (m:Symbol {kind: 'method'})-[:REFERENCES]->(c:Symbol {kind: 'class'})
RETURN m.name, c.name
LIMIT 10
```

## 🎯 Common Use Cases

### 1. Find All Implementations of an Interface

```typescript
// Define interface
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  save(item: T): Promise<void>;
}

// Find all implementations in nCode
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{"symbol": "my_project.Repository#"}'
```

### 2. Navigate Type Hierarchies

```bash
# Find all classes extending BaseController
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{"symbol": "my_project.BaseController#"}'
```

### 3. Track Function Calls

```bash
# Find all places calling authenticate()
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{"symbol": "my_project.auth.authenticate()."}'
```

## 🐛 Troubleshooting

### Issue: "Cannot find tsconfig.json"

```bash
# Specify tsconfig location
scip-typescript index --tsconfig ./tsconfig.json

# Or create one
npx tsc --init
```

### Issue: "No symbols found"

Check that:
1. TypeScript files are in the project
2. `tsconfig.json` includes your source files
3. TypeScript compiles without errors

```bash
# Test TypeScript compilation
npx tsc --noEmit
```

### Issue: Large monorepo indexing is slow

```bash
# Index incrementally
scip-typescript index --inferTsconfig false

# Or index specific directories
cd packages/core && scip-typescript index
cd packages/api && scip-typescript index
```

### Issue: Memory errors with large projects

```bash
# Increase Node.js memory
NODE_OPTIONS="--max-old-space-size=4096" scip-typescript index
```

## 📚 Best Practices

1. **Commit SCIP files to Git** for consistent indexing across team
2. **Index after major refactors** to keep navigation up-to-date
3. **Use in CI/CD** to track code changes over time
4. **Combine with ESLint** for comprehensive code quality
5. **Export to Qdrant** for natural language code search

## 🔗 Related Resources

- [scip-typescript GitHub](https://github.com/sourcegraph/scip-typescript)
- [SCIP Protocol Spec](https://github.com/sourcegraph/scip)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/)
- [nCode API Reference](../../docs/API.md)

## 📝 Example Scripts

### Auto-index on file change

```bash
#!/bin/bash
# watch_and_index.sh

while inotifywait -r -e modify,create,delete ./src; do
    echo "Changes detected, re-indexing..."
    scip-typescript index
    curl -X POST http://localhost:18003/v1/index/load \
      -H "Content-Type: application/json" \
      -d "{\"path\": \"$(pwd)/index.scip\"}"
done
```

### Pre-commit hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Generating SCIP index..."
scip-typescript index --output=.scip/index.scip
git add .scip/index.scip
```

---

**Last Updated:** 2026-01-17  
**Version:** 1.0
