"""
nCode CLI Tool (Mojo implementation)
Command-line interface for the nCode SCIP code intelligence platform

Commands:
    ncode index <path>           - Index a project
    ncode search <query>         - Semantic search across code
    ncode query <cypher>         - Run Cypher query on graph
    ncode definition <file:line> - Find symbol definition
    ncode references <file:line> - Find symbol references
    ncode symbols <file>         - List symbols in file
    ncode export <format>        - Export index data
    ncode health                 - Check server health
    ncode interactive            - Interactive mode
"""

from python import Python
from sys import argv
import sys

struct Config:
    var server_url: String
    var qdrant_url: String
    var memgraph_url: String
    var verbose: Bool
    var json_output: Bool
    
    fn __init__(inout self):
        self.server_url = "http://localhost:18003"
        self.qdrant_url = "http://localhost:6333"
        self.memgraph_url = "bolt://localhost:7687"
        self.verbose = False
        self.json_output = False


fn cmd_index(path: String, config: Config) raises:
    """Index a project or file"""
    print("Indexing:", path)
    
    # Detect language and run appropriate indexer
    let ext = get_extension(path)
    let indexer = detect_indexer(ext)
    
    print("Using indexer:", indexer)
    
    # Run indexer (using Python subprocess)
    let py = Python()
    let subprocess = py.import_module("subprocess")
    
    let result = subprocess.run([indexer, "index", path], capture_output=True)
    
    if result.returncode != 0:
        print("Error: Indexing failed")
        raise Error("Indexing failed")
    
    # Load index into server
    from ncode_client import NCodeClient
    
    var client = NCodeClient(config.server_url)
    var load_result = client.load_index("index.scip")
    
    print("✓ Loaded", load_result.symbols, "symbols from", load_result.documents, "documents")


fn cmd_search(query: String, limit: Int, config: Config) raises:
    """Semantic search across code"""
    print("Searching for:", query)
    
    from ncode_client import QdrantClient
    
    var qdrant = QdrantClient(config.qdrant_url, "ncode")
    var results = qdrant.semantic_search(query, limit)
    
    print("Results:", results)


fn cmd_query(cypher: String, config: Config) raises:
    """Run Cypher query on graph database"""
    print("Running query:", cypher)
    
    from ncode_client import MemgraphClient
    
    var memgraph = MemgraphClient(config.memgraph_url)
    
    # Execute query (implementation depends on query type)
    print("Query results: (implementation needed)")
    
    memgraph.close()


fn cmd_definition(location: String, config: Config) raises:
    """Find symbol definition"""
    let parts = parse_location(location)
    let file = parts[0]
    let line = int(parts[1])
    let character = int(parts[2])
    
    from ncode_client import NCodeClient
    
    var client = NCodeClient(config.server_url)
    var result = client.find_definition(file, line, character)
    
    if "location" in result:
        let loc = result["location"]
        print("Definition:", loc["uri"] + ":" + str(loc["range"]["start"]["line"]) + 
              ":" + str(loc["range"]["start"]["character"]))
        
        if "symbol" in result:
            print("Symbol:", result["symbol"])
    else:
        print("No definition found")


fn cmd_references(location: String, config: Config) raises:
    """Find symbol references"""
    let parts = parse_location(location)
    let file = parts[0]
    let line = int(parts[1])
    let character = int(parts[2])
    
    from ncode_client import NCodeClient
    
    var client = NCodeClient(config.server_url)
    var result = client.find_references(file, line, character)
    
    let locations = result["locations"]
    print("Found", len(locations), "references:")
    
    for loc in locations:
        print("  " + loc["uri"] + ":" + str(loc["range"]["start"]["line"]) + 
              ":" + str(loc["range"]["start"]["character"]))


fn cmd_symbols(file: String, config: Config) raises:
    """List symbols in file"""
    from ncode_client import NCodeClient
    
    var client = NCodeClient(config.server_url)
    var result = client.get_symbols(file)
    
    print("Symbols in", file + ":")
    
    for sym in result["symbols"]:
        print("  " + sym["kind"], sym["name"], "at line", sym["range"]["start"]["line"])


fn cmd_export(format: String, output: String, config: Config) raises:
    """Export index data"""
    print("Exporting to format:", format)
    print("Output file:", output)
    print("Export functionality coming soon")


fn cmd_health(config: Config) raises:
    """Check server health"""
    from ncode_client import NCodeClient
    
    var client = NCodeClient(config.server_url)
    var health = client.health()
    
    print("nCode Server Health")
    print("  Status:", health.status)
    print("  Version:", health.version)
    print("  Uptime:", health.uptime_seconds, "seconds")
    print("  Index Loaded:", health.index_loaded)


fn cmd_interactive(config: Config) raises:
    """Start interactive mode"""
    print("nCode Interactive Mode")
    print("Type 'help' for commands, 'exit' to quit")
    print()
    
    let py = Python()
    let input_fn = py.builtins().input
    
    while True:
        try:
            let line = String(input_fn("> "))
            let trimmed = line.strip()
            
            if len(trimmed) == 0:
                continue
            
            if trimmed == "exit":
                break
            
            if trimmed == "help":
                print_help()
                continue
            
            # Parse and execute command
            print("Executing:", trimmed)
            
        except:
            break


fn parse_location(loc_str: String) -> List[String]:
    """Parse file:line:character format"""
    let parts = loc_str.split(":")
    if len(parts) != 3:
        raise Error("Invalid location format. Use file:line:character")
    return parts


fn get_extension(path: String) -> String:
    """Get file extension"""
    let parts = path.split(".")
    if len(parts) < 2:
        return ""
    return "." + parts[len(parts) - 1]


fn detect_indexer(ext: String) -> String:
    """Detect appropriate indexer based on file extension"""
    if ext == ".ts" or ext == ".tsx":
        return "scip-typescript"
    elif ext == ".py":
        return "scip-python"
    elif ext == ".java":
        return "scip-java"
    elif ext == ".rs":
        return "rust-analyzer"
    elif ext == ".go":
        return "scip-go"
    else:
        return "ncode-treesitter"


fn print_help():
    """Print help message"""
    print("""nCode CLI - SCIP Code Intelligence Platform

Usage: ncode <command> [options]

Commands:
  index <path>              Index a project or file
  search <query>            Semantic search across code
  query <cypher>            Run Cypher query on graph database
  definition <file:line:char>  Find symbol definition
  references <file:line:char>  Find symbol references
  symbols <file>            List symbols in file
  export <format>           Export index data (json, csv, graphml)
  health                    Check server health
  interactive               Start interactive mode
  help                      Show this help message

Options:
  --server <url>            nCode server URL (default: http://localhost:18003)
  --qdrant <url>            Qdrant URL (default: http://localhost:6333)
  --memgraph <url>          Memgraph URL (default: bolt://localhost:7687)
  --verbose                 Enable verbose output
  --json                    Output in JSON format

Examples:
  ncode index src/
  ncode search "authentication function"
  ncode definition src/main.mojo:10:5
  ncode references src/utils.mojo:25:10
  ncode symbols src/main.mojo
  ncode query "MATCH (n:Symbol) RETURN n LIMIT 10"
  ncode export json --output index.json
  ncode interactive

For more information: https://github.com/ncode/docs
""")


fn main() raises:
    """Main entry point"""
    let args = sys.argv()
    
    if len(args) < 2:
        print_help()
        return
    
    var config = Config()
    let command = args[1]
    
    # Execute command
    if command == "index":
        if len(args) < 3:
            print("Error: Missing path argument")
            return
        cmd_index(args[2], config)
    
    elif command == "search":
        if len(args) < 3:
            print("Error: Missing query argument")
            return
        let limit = 10 if len(args) < 4 else int(args[3])
        cmd_search(args[2], limit, config)
    
    elif command == "query":
        if len(args) < 3:
            print("Error: Missing Cypher query")
            return
        cmd_query(args[2], config)
    
    elif command == "definition":
        if len(args) < 3:
            print("Error: Missing location (file:line:char)")
            return
        cmd_definition(args[2], config)
    
    elif command == "references":
        if len(args) < 3:
            print("Error: Missing location (file:line:char)")
            return
        cmd_references(args[2], config)
    
    elif command == "symbols":
        if len(args) < 3:
            print("Error: Missing file path")
            return
        cmd_symbols(args[2], config)
    
    elif command == "export":
        if len(args) < 3:
            print("Error: Missing export format")
            return
        let output = "output." + args[2] if len(args) < 4 else args[3]
        cmd_export(args[2], output, config)
    
    elif command == "health":
        cmd_health(config)
    
    elif command == "interactive":
        cmd_interactive(config)
    
    elif command == "help":
        print_help()
    
    else:
        print("Unknown command:", command)
        print_help()
