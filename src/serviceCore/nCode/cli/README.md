# nCode CLI Tools

Command-line interface for the nCode SCIP code intelligence platform. Available in **Zig**, **Mojo**, and **Shell script**.

## Quick Start

### Shell Script (Universal)
```bash
chmod +x ncode.sh
./ncode.sh health
./ncode.sh index src/
./ncode.sh symbols src/main.js
```

### Zig (Compiled, Fast)
```bash
zig build-exe ncode.zig -I../client
./ncode health
./ncode index src/
```

### Mojo (ML/Data Workflows)
```bash
mojo ncode.mojo health
mojo ncode.mojo index src/
```

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `index <path>` | Index project/file | `ncode index src/` |
| `search <query>` | Semantic search | `ncode search "auth function"` |
| `query <cypher>` | Run Cypher query | `ncode query "MATCH (n) RETURN n"` |
| `definition <loc>` | Find definition | `ncode definition src/main.js:10:5` |
| `references <loc>` | Find references | `ncode references src/utils.js:25:10` |
| `symbols <file>` | List symbols | `ncode symbols src/main.js` |
| `export <format>` | Export data | `ncode export json output.json` |
| `health` | Check server | `ncode health` |
| `interactive` | Interactive mode | `ncode interactive` |
| `help` | Show help | `ncode help` |

## Installation

### Shell Completion

**Bash:**
```bash
source cli/completions/ncode.bash
# Or add to ~/.bashrc:
echo 'source /path/to/ncode/cli/completions/ncode.bash' >> ~/.bashrc
```

**Zsh:**
```bash
# Add to ~/.zshrc:
fpath=(/path/to/ncode/cli/completions $fpath)
autoload -Uz compinit && compinit
```

### System-wide Installation

```bash
# Copy main CLI
sudo cp ncode.sh /usr/local/bin/ncode
sudo chmod +x /usr/local/bin/ncode

# Install completions
sudo mkdir -p /usr/share/bash-completion/completions
sudo cp completions/ncode.bash /usr/share/bash-completion/completions/ncode
```

## Environment Variables

```bash
export NCODE_SERVER="http://localhost:18003"
export QDRANT_SERVER="http://localhost:6333"
export MEMGRAPH_SERVER="bolt://localhost:7687"
export VERBOSE="true"
```

## Examples

### Basic Workflow
```bash
# Check server
ncode health

# Index a project
ncode index myproject/

# Find symbols
ncode symbols myproject/src/main.js

# Find definition
ncode definition myproject/src/main.js:15:8

# Find references
ncode references myproject/src/utils.js:42:12

# Search semantically
ncode search "authentication handler"
```

### Interactive Mode
```bash
ncode interactive
> health
> index src/
> symbols src/main.js
> exit
```

### Scripting
```bash
#!/bin/bash
# Batch index multiple projects
for project in project1 project2 project3; do
    echo "Indexing $project..."
    ncode index "$project"
done

# Search and save results
ncode search "api handler" > results.txt
```

## Language-Specific Features

### Zig CLI
- Native performance
- Small binary size
- Type-safe
- Best for: System tools, embedded

### Mojo CLI
- Python interop
- ML/AI workflows
- Modern syntax
- Best for: Data science, ML pipelines

### Shell CLI
- Universal compatibility
- No compilation needed
- Easy to modify
- Best for: Quick scripts, automation

## Implementation Comparison

| Feature | Zig | Mojo | Shell |
|---------|-----|------|-------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Portability | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Dependencies | None | Python | curl, jq |
| Binary Size | ~500KB | N/A | ~10KB |
| Startup Time | <1ms | ~50ms | ~10ms |

## Command Reference

### `index`
Index a project or file using appropriate SCIP indexer.

```bash
ncode index <path> [--language <lang>]

# Examples
ncode index src/
ncode index myfile.ts
ncode index . --language typescript
```

**Supported Languages:** TypeScript, Python, Java, Rust, Go, and 28+ others via tree-sitter.

### `search`
Perform semantic search across indexed code.

```bash
ncode search <query> [limit]

# Examples
ncode search "authentication function" 10
ncode search "database connection" 20
```

**Requires:** Qdrant running with loaded index.

### `query`
Execute Cypher queries on the code graph.

```bash
ncode query <cypher-query>

# Examples
ncode query "MATCH (n:Symbol) RETURN n LIMIT 10"
ncode query "MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.name, g.name"
```

**Requires:** Memgraph running with loaded graph.

### `definition`
Find where a symbol is defined.

```bash
ncode definition <file:line:char>

# Example
ncode definition src/main.js:10:5
# Output: Definition: src/utils.js:42:8
#         Symbol: myFunction
```

### `references`
Find all references to a symbol.

```bash
ncode references <file:line:char> [--include-declaration]

# Example
ncode references src/utils.js:42:8
# Output: Found 5 references:
#           src/main.js:10:5
#           src/app.js:25:12
#           ...
```

### `symbols`
List all symbols in a file.

```bash
ncode symbols <file>

# Example
ncode symbols src/main.js
# Output: Symbols in src/main.js:
#           class MyClass at line 5
#           function myFunction at line 15
#           ...
```

### `export`
Export index data in various formats.

```bash
ncode export <format> [output]

# Examples
ncode export json index.json
ncode export csv symbols.csv
ncode export graphml graph.xml
```

**Formats:** json, csv, graphml

### `health`
Check nCode server health and status.

```bash
ncode health

# Output: nCode Server Health
#           Status: ok
#           Version: 2.0
#           Uptime: 3600.5 seconds
#           Index Loaded: true
```

### `interactive`
Start interactive REPL mode.

```bash
ncode interactive
# > health
# > index src/
# > symbols src/main.js
# > exit
```

## Troubleshooting

### Server Not Running
```bash
$ ncode health
Error: Connection refused

# Solution: Start the server
cd src/serviceCore/nCode
./scripts/start.sh
```

### Command Not Found
```bash
$ ncode: command not found

# Solution: Use full path or install
./ncode.sh health
# Or install system-wide (see Installation)
```

### Missing Dependencies (Shell version)
```bash
$ ./ncode.sh: jq: command not found

# Solution: Install dependencies
# macOS
brew install jq curl

# Ubuntu/Debian
sudo apt-get install jq curl

# Fedora/RHEL
sudo dnf install jq curl
```

## Advanced Usage

### Custom Server URLs
```bash
# Temporary
NCODE_SERVER=http://remote:18003 ncode health

# Permanent
export NCODE_SERVER="http://remote:18003"
ncode health
```

### Batch Operations
```bash
# Index multiple projects
find . -name "src" -type d | xargs -I {} ncode index {}

# Export all symbols
ncode symbols src/*.js > all_symbols.txt
```

### Integration with Other Tools
```bash
# With grep
ncode symbols src/main.js | grep "function"

# With fzf
ncode symbols src/main.js | fzf

# With watch
watch -n 5 ncode health
```

## Performance Tips

1. **Use Zig for Production:** Fastest startup and execution
2. **Cache Index:** Don't re-index unnecessarily
3. **Batch Operations:** Index multiple files at once
4. **Interactive Mode:** Reduces startup overhead for multiple commands

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

---

**Version:** 1.0.0  
**Last Updated:** 2026-01-18  
**Status:** Production Ready ✅
