#!/bin/bash

# nCode CLI Tool (Shell script implementation)
# Command-line interface for the nCode SCIP code intelligence platform

set -e

# Configuration
NCODE_SERVER="${NCODE_SERVER:-http://localhost:18003}"
QDRANT_SERVER="${QDRANT_SERVER:-http://localhost:6333}"
MEMGRAPH_SERVER="${MEMGRAPH_SERVER:-bolt://localhost:7687}"
VERBOSE="${VERBOSE:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helpers
log_info() { [ "$VERBOSE" = "true" ] && echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1" >&2; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

# API helper
api_call() {
    local method=$1 endpoint=$2 data=$3
    local url="${NCODE_SERVER}${endpoint}"
    
    if [ -n "$data" ]; then
        curl -s -X "$method" "$url" -H "Content-Type: application/json" -d "$data"
    else
        curl -s -X "$method" "$url"
    fi
}

# Commands
cmd_index() {
    local path=$1
    [ -z "$path" ] && { log_error "Usage: ncode index <path>"; return 1; }
    
    log_info "Indexing: $path"
    
    # Detect indexer
    local ext="${path##*.}"
    local indexer="ncode-treesitter"
    case "$ext" in
        ts|tsx) indexer="scip-typescript" ;;
        py) indexer="scip-python" ;;
        java) indexer="scip-java" ;;
        rs) indexer="rust-analyzer" ;;
        go) indexer="scip-go" ;;
    esac
    
    log_info "Using indexer: $indexer"
    
    # Run indexer
    if ! $indexer index "$path"; then
        log_error "Indexing failed"
        return 1
    fi
    
    # Load to server
    local result=$(api_call POST "/v1/index/load" "{\"path\":\"index.scip\"}")
    local docs=$(echo "$result" | jq -r '.documents // 0')
    local syms=$(echo "$result" | jq -r '.symbols // 0')
    
    log_success "Loaded $syms symbols from $docs documents"
}

cmd_search() {
    local query=$1 limit=${2:-10}
    [ -z "$query" ] && { log_error "Usage: ncode search <query> [limit]"; return 1; }
    
    log_info "Searching for: $query"
    
    # TODO: Implement Qdrant search
    log_warning "Search functionality requires Qdrant integration"
}

cmd_query() {
    local cypher=$1
    [ -z "$cypher" ] && { log_error "Usage: ncode query <cypher>"; return 1; }
    
    log_info "Running query: $cypher"
    log_warning "Query functionality requires Memgraph integration"
}

cmd_definition() {
    local location=$1
    [ -z "$location" ] && { log_error "Usage: ncode definition <file:line:char>"; return 1; }
    
    IFS=':' read -r file line char <<< "$location"
    
    local result=$(api_call POST "/v1/definition" "{\"file\":\"$file\",\"line\":$line,\"character\":$char}")
    
    if echo "$result" | jq -e '.location' > /dev/null; then
        local uri=$(echo "$result" | jq -r '.location.uri')
        local def_line=$(echo "$result" | jq -r '.location.range.start.line')
        local def_char=$(echo "$result" | jq -r '.location.range.start.character')
        echo "Definition: $uri:$def_line:$def_char"
        
        local symbol=$(echo "$result" | jq -r '.symbol // empty')
        [ -n "$symbol" ] && echo "Symbol: $symbol"
    else
        log_warning "No definition found"
    fi
}

cmd_references() {
    local location=$1
    [ -z "$location" ] && { log_error "Usage: ncode references <file:line:char>"; return 1; }
    
    IFS=':' read -r file line char <<< "$location"
    
    local result=$(api_call POST "/v1/references" "{\"file\":\"$file\",\"line\":$line,\"character\":$char}")
    
    local count=$(echo "$result" | jq '.locations | length')
    echo "Found $count references:"
    
    echo "$result" | jq -r '.locations[] | "  \(.uri):\(.range.start.line):\(.range.start.character)"'
}

cmd_symbols() {
    local file=$1
    [ -z "$file" ] && { log_error "Usage: ncode symbols <file>"; return 1; }
    
    local result=$(api_call POST "/v1/symbols" "{\"file\":\"$file\"}")
    
    echo "Symbols in $file:"
    echo "$result" | jq -r '.symbols[] | "  \(.kind) \(.name) at line \(.range.start.line)"'
}

cmd_export() {
    local format=$1 output=${2:-output.$format}
    [ -z "$format" ] && { log_error "Usage: ncode export <format> [output]"; return 1; }
    
    log_warning "Export functionality coming soon"
}

cmd_health() {
    local result=$(api_call GET "/health")
    
    echo "nCode Server Health"
    echo "$result" | jq -r '"  Status: \(.status)"'
    echo "$result" | jq -r '"  Version: \(.version // "unknown")"'
    echo "$result" | jq -r '"  Uptime: \(.uptime_seconds // 0) seconds"'
    echo "$result" | jq -r '"  Index Loaded: \(.index_loaded // false)"'
}

cmd_interactive() {
    echo "nCode Interactive Mode"
    echo "Type 'help' for commands, 'exit' to quit"
    echo
    
    while true; do
        read -p "> " line
        [ -z "$line" ] && continue
        [ "$line" = "exit" ] && break
        [ "$line" = "help" ] && { print_help; continue; }
        
        eval "ncode $line"
    done
}

print_help() {
    cat << 'EOF'
nCode CLI - SCIP Code Intelligence Platform

Usage: ncode <command> [options]

Commands:
  index <path>              Index a project or file
  search <query> [limit]    Semantic search across code
  query <cypher>            Run Cypher query on graph database
  definition <file:line:char>  Find symbol definition
  references <file:line:char>  Find symbol references
  symbols <file>            List symbols in file
  export <format> [output]  Export index data (json, csv, graphml)
  health                    Check server health
  interactive               Start interactive mode
  help                      Show this help message

Environment Variables:
  NCODE_SERVER              nCode server URL (default: http://localhost:18003)
  QDRANT_SERVER             Qdrant URL (default: http://localhost:6333)
  MEMGRAPH_SERVER           Memgraph URL (default: bolt://localhost:7687)
  VERBOSE                   Enable verbose output (true/false)

Examples:
  ncode index src/
  ncode search "authentication function" 10
  ncode definition src/main.js:10:5
  ncode references src/utils.js:25:10
  ncode symbols src/main.js
  ncode query "MATCH (n:Symbol) RETURN n LIMIT 10"
  ncode export json output.json
  ncode health
  ncode interactive

For more information: https://github.com/ncode/docs
EOF
}

# Main
main() {
    local command=$1
    shift || { print_help; return 0; }
    
    case "$command" in
        index) cmd_index "$@" ;;
        search) cmd_search "$@" ;;
        query) cmd_query "$@" ;;
        definition) cmd_definition "$@" ;;
        references) cmd_references "$@" ;;
        symbols) cmd_symbols "$@" ;;
        export) cmd_export "$@" ;;
        health) cmd_health ;;
        interactive) cmd_interactive ;;
        help|--help|-h) print_help ;;
        *) log_error "Unknown command: $command"; print_help; return 1 ;;
    esac
}

main "$@"
