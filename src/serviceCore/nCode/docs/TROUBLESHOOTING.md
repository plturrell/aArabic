# nCode Troubleshooting Guide

Comprehensive troubleshooting guide for nCode and its database integrations.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [nCode Server Issues](#ncode-server-issues)
- [SCIP Indexing Issues](#scip-indexing-issues)
- [Database Connection Issues](#database-connection-issues)
- [Performance Issues](#performance-issues)
- [Data Quality Issues](#data-quality-issues)
- [Common Error Messages](#common-error-messages)
- [Debug Mode](#debug-mode)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

Run these commands to quickly identify issues:

```bash
# Check nCode server
curl http://localhost:18003/health

# Check Qdrant
curl http://localhost:6333/healthz

# Check Memgraph
docker ps | grep memgraph

# Check Marquez
curl http://localhost:5000/api/v1/namespaces

# Check nCode build
cd src/serviceCore/nCode && zig build test

# Check integration tests
./scripts/integration_test.sh
```

---

## nCode Server Issues

### Server Won't Start

**Symptoms**:
- `./scripts/start.sh` exits immediately
- Port 18003 not listening
- No response from `curl http://localhost:18003/health`

**Diagnosis**:
```bash
# Check if port is already in use
lsof -i :18003

# Try starting manually to see errors
cd src/serviceCore/nCode
./zig-out/bin/ncode-server

# Check if binary exists
ls -la zig-out/bin/ncode-server
```

**Solutions**:

1. **Port Already in Use**:
   ```bash
   # Kill existing process
   kill $(lsof -t -i:18003)
   
   # Or use different port (edit server/main.zig)
   const port = 18004;  // Change port
   zig build
   ```

2. **Binary Not Built**:
   ```bash
   cd src/serviceCore/nCode
   zig build
   ```

3. **Permission Issues**:
   ```bash
   chmod +x zig-out/bin/ncode-server
   chmod +x scripts/start.sh
   ```

4. **Zig Version Mismatch**:
   ```bash
   # Check version
   zig version  # Should be 0.15.2+
   
   # Upgrade if needed
   brew upgrade zig  # macOS
   ```

---

### Server Crashes on Index Load

**Symptoms**:
- Server crashes when calling `/v1/index/load`
- `Segmentation fault` or `Out of memory`

**Diagnosis**:
```bash
# Check SCIP file size
ls -lh index.scip

# Check system memory
free -h  # Linux
vm_stat  # macOS

# Run with debug info
cd src/serviceCore/nCode
zig build -Doptimize=Debug
./zig-out/bin/ncode-server 2>&1 | tee server.log
```

**Solutions**:

1. **File Too Large**:
   ```bash
   # Split large indexes
   # Index incrementally
   # Increase system memory
   ```

2. **Corrupt SCIP File**:
   ```bash
   # Verify protobuf format
   file index.scip
   # Should show: "data"
   
   # Try re-indexing
   rm index.scip
   npx @sourcegraph/scip-typescript index
   ```

3. **Memory Leak**:
   ```bash
   # Use valgrind (Linux)
   valgrind --leak-check=full ./zig-out/bin/ncode-server
   
   # Monitor memory
   watch -n 1 'ps aux | grep ncode-server'
   ```

---

### API Requests Hang

**Symptoms**:
- Requests to `/v1/definition` never return
- Client timeout errors
- High CPU usage

**Diagnosis**:
```bash
# Check server logs
tail -f /tmp/ncode-server.log

# Check for deadlocks
lldb ./zig-out/bin/ncode-server
(lldb) attach -p $(pgrep ncode-server)
(lldb) bt all

# Profile performance
instruments -t "Time Profiler" ./zig-out/bin/ncode-server
```

**Solutions**:

1. **Infinite Loop in Parser**:
   - Check SCIP file for malformed data
   - Re-index project
   - Update to latest nCode version

2. **Slow Symbol Lookup**:
   ```bash
   # Optimize hash map (in code)
   # Add caching layer
   # Reduce index size
   ```

3. **Too Many Concurrent Requests**:
   ```bash
   # Limit concurrent requests
   # Add rate limiting
   # Use load balancer
   ```

---

## SCIP Indexing Issues

### Indexer Not Found

**Symptoms**:
- `scip-typescript: command not found`
- `scip-python: command not found`

**Diagnosis**:
```bash
# Check if installed
which scip-typescript
which scip-python

# Check npm global packages
npm list -g --depth=0

# Check pip packages
pip list | grep scip
```

**Solutions**:

```bash
# Install indexers
cd src/serviceCore/nCode
./scripts/install_indexers.sh

# Or manually:
npm install -g @sourcegraph/scip-typescript
pip install scip-python
go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
```

---

### Indexing Fails

**Symptoms**:
- Indexer exits with error
- No `index.scip` file created
- Empty or corrupt index file

**Diagnosis**:
```bash
# Run indexer with verbose output
npx @sourcegraph/scip-typescript index --verbose

# Check project structure
ls -la

# Check for compilation errors
npx tsc --noEmit  # TypeScript
python -m py_compile *.py  # Python
```

**Solutions**:

1. **Missing Dependencies**:
   ```bash
   # Install project dependencies first
   npm install  # Node.js
   pip install -r requirements.txt  # Python
   go mod download  # Go
   ```

2. **Compilation Errors**:
   ```bash
   # Fix code errors before indexing
   # Ensure project compiles successfully
   ```

3. **Unsupported Language Features**:
   ```bash
   # Update indexer to latest version
   npm update -g @sourcegraph/scip-typescript
   ```

4. **Large Project**:
   ```bash
   # Increase memory
   NODE_OPTIONS="--max-old-space-size=8192" npx @sourcegraph/scip-typescript index
   ```

---

### Incomplete Index

**Symptoms**:
- Some files not indexed
- Missing symbols
- Fewer symbols than expected

**Diagnosis**:
```bash
# Check what files were indexed
strings index.scip | grep -E "\.ts|\.py|\.go"

# Count symbols
strings index.scip | grep "scip-" | wc -l

# Check indexer output
npx @sourcegraph/scip-typescript index --verbose 2>&1 | grep "indexed"
```

**Solutions**:

1. **Files Excluded by .gitignore**:
   ```bash
   # Check .gitignore
   cat .gitignore
   
   # Indexers typically respect .gitignore
   # Remove entries or use --no-git-ignore flag
   ```

2. **File Type Not Supported**:
   ```bash
   # Use tree-sitter indexer for data files
   ./zig-out/bin/ncode-treesitter index --language json .
   ```

3. **Parse Errors**:
   ```bash
   # Fix syntax errors in source files
   # Check indexer logs for parse failures
   ```

---

## Database Connection Issues

### Qdrant Connection Failed

**Symptoms**:
```
ConnectionError: Cannot connect to Qdrant at localhost:6333
```

**Diagnosis**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/healthz

# Check Docker
docker ps | grep qdrant

# Check logs
docker logs qdrant 2>&1 | tail -20
```

**Solutions**:

1. **Qdrant Not Running**:
   ```bash
   # Start Qdrant
   docker run -d -p 6333:6333 -p 6334:6334 \
       --name qdrant \
       qdrant/qdrant:latest
   ```

2. **Wrong Port**:
   ```bash
   # Check actual port
   docker port qdrant
   
   # Update loader command
   python scripts/load_to_databases.py index.scip \
       --qdrant --qdrant-port 6333
   ```

3. **Network Issues**:
   ```bash
   # Check firewall
   sudo ufw status
   
   # Test connectivity
   telnet localhost 6333
   ```

4. **Qdrant Full/Out of Space**:
   ```bash
   # Check disk space
   df -h
   
   # Clean up old collections
   curl -X DELETE http://localhost:6333/collections/old_collection
   ```

---

### Memgraph Connection Failed

**Symptoms**:
```
Neo4jError: Unable to connect to bolt://localhost:7687
```

**Diagnosis**:
```bash
# Check if Memgraph is running
docker ps | grep memgraph

# Check logs
docker logs memgraph 2>&1 | tail -20

# Test Bolt connection
echo "RETURN 1;" | cypher-shell -a bolt://localhost:7687
```

**Solutions**:

1. **Memgraph Not Running**:
   ```bash
   # Start Memgraph
   docker run -d -p 7687:7687 -p 7444:7444 \
       --name memgraph \
       memgraph/memgraph:latest
   ```

2. **Wrong Credentials**:
   ```python
   # Update loader (default: no auth)
   driver = GraphDatabase.driver(
       "bolt://localhost:7687",
       auth=("", "")  # Empty username/password
   )
   ```

3. **Bolt Protocol Version**:
   ```bash
   # Update neo4j driver
   pip install --upgrade neo4j
   ```

4. **Database Locked**:
   ```bash
   # Restart Memgraph
   docker restart memgraph
   ```

---

### Marquez API Errors

**Symptoms**:
```
aiohttp.ClientError: 500 Internal Server Error
```

**Diagnosis**:
```bash
# Check Marquez services
docker-compose ps

# Check API health
curl http://localhost:5000/api/v1/namespaces

# Check PostgreSQL
docker logs marquez-db 2>&1 | tail -20

# Check Marquez API logs
docker logs marquez-api 2>&1 | tail -50
```

**Solutions**:

1. **Services Not Running**:
   ```bash
   cd vendor/layerData/marquez
   docker-compose up -d
   ```

2. **Database Migration Needed**:
   ```bash
   # Run migrations
   docker exec marquez-api ./wait-for-it.sh marquez-db:5432 -- \
       java -jar marquez-api.jar db migrate marquez.yml
   ```

3. **Invalid OpenLineage Event**:
   ```python
   # Validate event structure
   import json
   import jsonschema
   
   # Download schema
   # curl -O https://openlineage.io/spec/1-0-5/OpenLineage.json
   
   # Validate
   with open('OpenLineage.json') as f:
       schema = json.load(f)
   jsonschema.validate(event, schema)
   ```

4. **PostgreSQL Connection Issues**:
   ```bash
   # Check PostgreSQL
   docker exec -it marquez-db psql -U marquez
   
   # Check connections
   SELECT * FROM pg_stat_activity;
   ```

---

## Performance Issues

### Slow Indexing

**Symptoms**:
- Indexing takes hours for medium projects
- High CPU/memory usage during indexing

**Solutions**:

1. **Optimize Indexer Settings**:
   ```bash
   # TypeScript: Use tsconfig for optimization
   {
     "compilerOptions": {
       "skipLibCheck": true,
       "incremental": true
     }
   }
   ```

2. **Parallel Processing**:
   ```bash
   # Index multiple projects in parallel
   for project in project1 project2 project3; do
     (cd $project && npx @sourcegraph/scip-typescript index) &
   done
   wait
   ```

3. **Incremental Indexing**:
   ```bash
   # Only re-index changed files
   # (Feature not yet implemented - see roadmap)
   ```

---

### Slow Database Loading

**Symptoms**:
- `load_to_databases.py` takes >30 minutes
- High memory usage during loading

**Solutions**:

1. **Increase Batch Size**:
   ```python
   # In qdrant_loader.py
   batch_size = 500  # Default: 100
   
   # In memgraph_loader.py
   BATCH_SIZE = 10000  # Default: 1000
   ```

2. **Use GPU for Embeddings**:
   ```python
   # In qdrant_loader.py
   model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
   ```

3. **Parallel Loading**:
   ```bash
   # Load to different databases in parallel
   python scripts/load_to_databases.py index.scip --qdrant &
   python scripts/load_to_databases.py index.scip --memgraph &
   python scripts/load_to_databases.py index.scip --marquez &
   wait
   ```

4. **Optimize Database Settings**:
   ```bash
   # Qdrant: Disable fsync for bulk load
   # Memgraph: Increase transaction size
   # Marquez: Batch events
   ```

---

### Slow Queries

**Symptoms**:
- API requests take >1 second
- Qdrant searches slow
- Memgraph traversals timeout

**Solutions**:

1. **Add Indexes (Memgraph)**:
   ```cypher
   CREATE INDEX ON :Symbol(symbol);
   CREATE INDEX ON :Symbol(kind);
   CREATE INDEX ON :Document(path);
   ```

2. **Optimize Query**:
   ```cypher
   // Bad: Undirected relationship
   MATCH (a)-[:REFERENCES]-(b)
   
   // Good: Directed relationship
   MATCH (a)-[:REFERENCES]->(b)
   ```

3. **Limit Result Size**:
   ```python
   # Always use LIMIT
   results = client.search(..., limit=10)
   ```

4. **Use Filters**:
   ```python
   # Filter before semantic search
   query_filter = Filter(
       must=[FieldCondition(key="language", match=MatchValue(value="python"))]
   )
   ```

---

## Data Quality Issues

### Missing Symbols

**Symptoms**:
- Expected symbols not in index
- Search doesn't find known functions

**Diagnosis**:
```bash
# Check if symbol exists in SCIP
strings index.scip | grep "myFunction"

# Check nCode server
curl -X POST http://localhost:18003/v1/symbols \
  -H "Content-Type: application/json" \
  -d '{"file": "src/main.ts"}'
```

**Solutions**:

1. **Re-index**:
   ```bash
   rm index.scip
   npx @sourcegraph/scip-typescript index
   ```

2. **Check Symbol Kind**:
   ```bash
   # Symbol might be there but different kind
   # Search by different criteria
   ```

3. **File Not Indexed**:
   ```bash
   # Ensure file is in project
   # Check .gitignore
   # Verify indexer configuration
   ```

---

### Duplicate Symbols

**Symptoms**:
- Same symbol appears multiple times
- Inconsistent results

**Diagnosis**:
```bash
# Check for duplicates in Qdrant
curl -X POST http://localhost:6333/collections/code_symbols/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 100}'
```

**Solutions**:

1. **Multiple Indexes Loaded**:
   ```bash
   # Clear and reload
   # Qdrant: Delete collection
   curl -X DELETE http://localhost:6333/collections/code_symbols
   
   # Memgraph: Clear graph
   echo "MATCH (n) DETACH DELETE n;" | cypher-shell
   
   # Reload single index
   python scripts/load_to_databases.py index.scip --all
   ```

2. **Overlapping Projects**:
   ```bash
   # Use separate collections/namespaces
   --qdrant-collection project1
   --qdrant-collection project2
   ```

---

### Incorrect Relationships

**Symptoms**:
- Call graph shows wrong connections
- Implementations not found

**Diagnosis**:
```cypher
// Check relationships
MATCH (a:Symbol)-[r]->(b:Symbol)
RETURN type(r), count(*) as count
GROUP BY type(r)
```

**Solutions**:

1. **Re-index Project**:
   ```bash
   # Ensure clean state
   rm index.scip
   npx @sourcegraph/scip-typescript index
   ```

2. **Update Indexer**:
   ```bash
   # Get latest version
   npm update -g @sourcegraph/scip-typescript
   ```

3. **Check SCIP Format**:
   ```bash
   # Verify relationships in SCIP file
   strings index.scip | grep "is_reference\|is_implementation"
   ```

---

## Common Error Messages

### `error: FileNotFound`

**Cause**: SCIP file doesn't exist at specified path

**Solution**:
```bash
# Check path
ls -la index.scip

# Use absolute path
curl -X POST http://localhost:18003/v1/index/load \
  -d '{"path": "/absolute/path/to/index.scip"}'
```

---

### `error: Invalid protobuf format`

**Cause**: Corrupt or non-SCIP file

**Solution**:
```bash
# Verify file type
file index.scip

# Re-generate index
rm index.scip
npx @sourcegraph/scip-typescript index
```

---

### `error: Symbol not found`

**Cause**: Symbol doesn't exist in loaded index

**Solution**:
```bash
# Verify symbol exists
curl -X POST http://localhost:18003/v1/symbols \
  -d '{"file": "src/main.ts"}'

# Check symbol format
# SCIP symbols are like: scip-typescript npm pkg src/file.ts Symbol#
```

---

### `MemoryError` or `Out of memory`

**Cause**: Index too large for available RAM

**Solution**:
```bash
# Increase system memory
# Split into multiple indexes
# Use pagination
# Implement lazy loading
```

---

## Debug Mode

### Enable Verbose Logging

```bash
# Set environment variable
export NCODE_LOG_LEVEL=DEBUG

# Run server
./scripts/start.sh

# Or run directly
./zig-out/bin/ncode-server --verbose
```

### Trace API Calls

```bash
# Use curl with verbose output
curl -v http://localhost:18003/health

# Use httpie for better formatting
http -v POST localhost:18003/v1/index/load path=index.scip
```

### Profile Performance

```bash
# macOS Instruments
instruments -t "Time Profiler" ./zig-out/bin/ncode-server

# Linux perf
perf record -g ./zig-out/bin/ncode-server
perf report

# Valgrind
valgrind --tool=callgrind ./zig-out/bin/ncode-server
```

---

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Review documentation**: README.md, ARCHITECTURE.md, API.md
3. **Search existing issues**: GitHub Issues
4. **Collect diagnostic information**:
   ```bash
   # System info
   uname -a
   zig version
   python --version
   
   # nCode version
   git rev-parse HEAD
   
   # Logs
   cat /tmp/ncode-server.log
   
   # Database status
   docker ps
   ```

### How to Report Issues

Create a GitHub issue with:

1. **Title**: Brief description
2. **Environment**: OS, Zig version, Python version
3. **Steps to Reproduce**: Exact commands run
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happened
6. **Logs**: Relevant log output
7. **SCIP File**: (if small, or sample)

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community help
- **Documentation**: All docs in `docs/` directory

---

## Checklist for Common Problems

Use this checklist when troubleshooting:

- [ ] Is nCode server running? (`curl http://localhost:18003/health`)
- [ ] Is index loaded? (Check API response)
- [ ] Are databases running? (Qdrant, Memgraph, Marquez)
- [ ] Is SCIP file valid? (`file index.scip`)
- [ ] Are dependencies installed? (`./scripts/install_indexers.sh`)
- [ ] Is project compiled? (No syntax errors)
- [ ] Is enough memory available? (`free -h` / `vm_stat`)
- [ ] Are ports available? (`lsof -i :18003`)
- [ ] Are logs showing errors? (`tail -f /tmp/ncode-server.log`)
- [ ] Is latest version? (`git pull`)

---

**Last Updated**: 2026-01-17  
**Version**: 1.0
