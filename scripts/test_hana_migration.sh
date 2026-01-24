#!/bin/bash
# ============================================================================
# HANA Migration Integration Test Suite
# Created: January 24, 2026
# Purpose: Validate all services work with HANA
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration
# ============================================================================

HANA_HOST="${HANA_HOST:-localhost}"
HANA_PORT="${HANA_PORT:-30015}"
HANA_DATABASE="${HANA_DATABASE:-NOPENAI_DB}"
HANA_USER="${HANA_USER:-SHIMMY_USER}"
HANA_PASSWORD="${HANA_PASSWORD}"

if [ -z "$HANA_PASSWORD" ]; then
    echo -e "${RED}ERROR: HANA_PASSWORD environment variable not set${NC}"
    exit 1
fi

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

run_sql() {
    local query="$1"
    hdbsql -u "$HANA_USER" -p "$HANA_PASSWORD" \
           -n "$HANA_HOST:$HANA_PORT" -d "$HANA_DATABASE" \
           -j "$query" 2>&1
}

# ============================================================================
# Test 1: HANA Connection
# ============================================================================

test_hana_connection() {
    print_header "Test 1: HANA Connection"
    
    print_info "Testing connection to $HANA_HOST:$HANA_PORT..."
    
    if result=$(run_sql "SELECT 1 FROM DUMMY"); then
        print_success "HANA connection successful"
        return 0
    else
        print_error "HANA connection failed: $result"
        return 1
    fi
}

# ============================================================================
# Test 2: Schema Verification
# ============================================================================

test_schema_exists() {
    print_header "Test 2: Schema Verification"
    
    print_info "Checking if migration tables exist..."
    
    local expected_tables=(
        "WORKFLOW_STATE"
        "WORKFLOW_CACHE"
        "EXECUTION_LOG"
        "KV_CACHE"
        "PROMPT_CACHE"
        "SESSION_STATE"
        "TENSOR_STORAGE"
        "AGENT_METADATA"
        "GRAPH_NODES"
        "GRAPH_EDGES"
        "SYSTEM_CONFIG"
        "HEALTH_STATUS"
    )
    
    local found=0
    for table in "${expected_tables[@]}"; do
        if result=$(run_sql "SELECT COUNT(*) FROM $table" 2>&1); then
            print_success "Table exists: $table"
            ((found++))
        else
            print_error "Table missing: $table"
        fi
    done
    
    print_info "Found $found out of ${#expected_tables[@]} expected tables"
    
    if [ $found -eq ${#expected_tables[@]} ]; then
        print_success "All tables exist"
        return 0
    else
        print_error "Some tables are missing"
        return 1
    fi
}

# ============================================================================
# Test 3: nAgentFlow Operations
# ============================================================================

test_nagent_flow() {
    print_header "Test 3: nAgentFlow Operations"
    
    # Test workflow state
    print_info "Testing WORKFLOW_STATE..."
    local workflow_id="test_workflow_$(date +%s)"
    
    # Insert test workflow
    run_sql "INSERT INTO WORKFLOW_STATE (workflow_id, status, state_data) 
             VALUES ('$workflow_id', 'RUNNING', TO_BLOB('test'))" > /dev/null
    
    # Retrieve workflow
    if result=$(run_sql "SELECT workflow_id FROM WORKFLOW_STATE WHERE workflow_id='$workflow_id'"); then
        print_success "Workflow state insert/select works"
    else
        print_error "Workflow state operations failed"
        return 1
    fi
    
    # Test workflow cache
    print_info "Testing WORKFLOW_CACHE..."
    local cache_key="test_cache_$(date +%s)"
    
    run_sql "INSERT INTO WORKFLOW_CACHE (cache_key, cache_value, expires_at) 
             VALUES ('$cache_key', TO_BLOB('cached'), ADD_SECONDS(CURRENT_TIMESTAMP, 3600))" > /dev/null
    
    if result=$(run_sql "SELECT cache_key FROM WORKFLOW_CACHE WHERE cache_key='$cache_key'"); then
        print_success "Workflow cache insert/select works"
    else
        print_error "Workflow cache operations failed"
        return 1
    fi
    
    # Cleanup
    run_sql "DELETE FROM WORKFLOW_STATE WHERE workflow_id='$workflow_id'" > /dev/null
    run_sql "DELETE FROM WORKFLOW_CACHE WHERE cache_key='$cache_key'" > /dev/null
    
    print_success "nAgentFlow operations validated"
    return 0
}

# ============================================================================
# Test 4: nLocalModels Cache Operations
# ============================================================================

test_nlocal_models() {
    print_header "Test 4: nLocalModels Cache Operations"
    
    # Test KV cache
    print_info "Testing KV_CACHE..."
    local kv_key="test_kv_$(date +%s)"
    
    run_sql "INSERT INTO KV_CACHE (key, value, expires_at) 
             VALUES ('$kv_key', TO_BLOB('kv_value'), ADD_SECONDS(CURRENT_TIMESTAMP, 3600))" > /dev/null
    
    if result=$(run_sql "SELECT key FROM KV_CACHE WHERE key='$kv_key'"); then
        print_success "KV cache insert/select works"
    else
        print_error "KV cache operations failed"
        return 1
    fi
    
    # Test prompt cache
    print_info "Testing PROMPT_CACHE..."
    local prompt_hash=$(echo -n "test_prompt" | sha256sum | cut -d' ' -f1)
    
    run_sql "INSERT INTO PROMPT_CACHE (hash, state, expires_at) 
             VALUES ('$prompt_hash', TO_BLOB('prompt_state'), ADD_SECONDS(CURRENT_TIMESTAMP, 3600))" > /dev/null
    
    if result=$(run_sql "SELECT hash FROM PROMPT_CACHE WHERE hash='$prompt_hash'"); then
        print_success "Prompt cache insert/select works"
    else
        print_error "Prompt cache operations failed"
        return 1
    fi
    
    # Test session state
    print_info "Testing SESSION_STATE..."
    local session_id="test_session_$(date +%s)"
    
    run_sql "INSERT INTO SESSION_STATE (session_id, data, expires_at) 
             VALUES ('$session_id', TO_BLOB('session_data'), ADD_SECONDS(CURRENT_TIMESTAMP, 1800))" > /dev/null
    
    if result=$(run_sql "SELECT session_id FROM SESSION_STATE WHERE session_id='$session_id'"); then
        print_success "Session state insert/select works"
    else
        print_error "Session state operations failed"
        return 1
    fi
    
    # Cleanup
    run_sql "DELETE FROM KV_CACHE WHERE key='$kv_key'" > /dev/null
    run_sql "DELETE FROM PROMPT_CACHE WHERE hash='$prompt_hash'" > /dev/null
    run_sql "DELETE FROM SESSION_STATE WHERE session_id='$session_id'" > /dev/null
    
    print_success "nLocalModels cache operations validated"
    return 0
}

# ============================================================================
# Test 5: TTL Cleanup
# ============================================================================

test_ttl_cleanup() {
    print_header "Test 5: TTL Cleanup"
    
    print_info "Testing automatic TTL cleanup..."
    
    # Insert expired entries
    run_sql "INSERT INTO KV_CACHE (key, value, expires_at) 
             VALUES ('expired_key', TO_BLOB('old'), ADD_SECONDS(CURRENT_TIMESTAMP, -3600))" > /dev/null
    
    # Run cleanup procedure
    if run_sql "CALL CLEANUP_EXPIRED_CACHE()" > /dev/null; then
        print_success "TTL cleanup procedure executed"
    else
        print_error "TTL cleanup procedure failed"
        return 1
    fi
    
    # Verify expired entry was deleted
    if result=$(run_sql "SELECT COUNT(*) FROM KV_CACHE WHERE key='expired_key'"); then
        if [[ "$result" == *"0"* ]]; then
            print_success "Expired entries cleaned up correctly"
        else
            print_error "Expired entries not cleaned up"
            return 1
        fi
    fi
    
    print_success "TTL cleanup validated"
    return 0
}

# ============================================================================
# Test 6: Cache Statistics View
# ============================================================================

test_statistics_view() {
    print_header "Test 6: Monitoring Views"
    
    print_info "Testing V_CACHE_STATISTICS view..."
    
    if result=$(run_sql "SELECT * FROM V_CACHE_STATISTICS"); then
        print_success "Cache statistics view accessible"
        echo "$result"
    else
        print_error "Cache statistics view failed"
        return 1
    fi
    
    print_success "Monitoring views validated"
    return 0
}

# ============================================================================
# Test 7: Performance Benchmark
# ============================================================================

test_performance() {
    print_header "Test 7: Performance Benchmark"
    
    print_info "Running performance test (100 operations)..."
    
    local start_time=$(date +%s%N)
    
    for i in {1..100}; do
        run_sql "INSERT INTO KV_CACHE (key, value, expires_at) 
                 VALUES ('perf_test_$i', TO_BLOB('data'), ADD_SECONDS(CURRENT_TIMESTAMP, 60))" > /dev/null
    done
    
    local end_time=$(date +%s%N)
    local duration_ms=$(( (end_time - start_time) / 1000000 ))
    local ops_per_sec=$(( 100000 / duration_ms ))
    
    print_info "100 inserts completed in ${duration_ms}ms"
    print_info "Throughput: ~${ops_per_sec} ops/sec"
    
    # Cleanup
    run_sql "DELETE FROM KV_CACHE WHERE key LIKE 'perf_test_%'" > /dev/null
    
    if [ $duration_ms -lt 1000 ]; then
        print_success "Performance test passed (< 1s for 100 ops)"
        return 0
    else
        print_error "Performance test slow (${duration_ms}ms)"
        return 1
    fi
}

# ============================================================================
# Test 8: Graph Engine (nAgentMeta)
# ============================================================================

test_graph_engine() {
    print_header "Test 8: Graph Engine"
    
    print_info "Testing graph operations..."
    
    # Create test nodes
    run_sql "INSERT INTO GRAPH_NODES (node_id, node_type, properties) 
             VALUES ('node1', 'agent', '{\"name\":\"TestAgent\"}')" > /dev/null
    run_sql "INSERT INTO GRAPH_NODES (node_id, node_type, properties) 
             VALUES ('node2', 'task', '{\"name\":\"TestTask\"}')" > /dev/null
    
    # Create edge
    run_sql "INSERT INTO GRAPH_EDGES (edge_id, source_node_id, target_node_id, edge_type) 
             VALUES ('edge1', 'node1', 'node2', 'executes')" > /dev/null
    
    # Query graph
    if result=$(run_sql "SELECT COUNT(*) FROM GRAPH_EDGES WHERE source_node_id='node1'"); then
        print_success "Graph query successful"
    else
        print_error "Graph query failed"
        return 1
    fi
    
    # Cleanup
    run_sql "DELETE FROM GRAPH_EDGES WHERE edge_id='edge1'" > /dev/null
    run_sql "DELETE FROM GRAPH_NODES WHERE node_id IN ('node1', 'node2')" > /dev/null
    
    print_success "Graph engine validated"
    return 0
}

# ============================================================================
# Main Test Runner
# ============================================================================

main() {
    print_header "HANA Migration Integration Test Suite"
    
    echo -e "${BLUE}Configuration:${NC}"
    echo "  HANA Host: $HANA_HOST"
    echo "  HANA Port: $HANA_PORT"
    echo "  Database: $HANA_DATABASE"
    echo "  User: $HANA_USER"
    echo ""
    
    local tests_passed=0
    local tests_failed=0
    local tests_total=8
    
    # Run tests
    if test_hana_connection; then ((tests_passed++)); else ((tests_failed++)); fi
    if test_schema_exists; then ((tests_passed++)); else ((tests_failed++)); fi
    if test_nagent_flow; then ((tests_passed++)); else ((tests_failed++)); fi
    if test_nlocal_models; then ((tests_passed++)); else ((tests_failed++)); fi
    if test_ttl_cleanup; then ((tests_passed++)); else ((tests_failed++)); fi
    if test_statistics_view; then ((tests_passed++)); else ((tests_failed++)); fi
    if test_performance; then ((tests_passed++)); else ((tests_failed++)); fi
    if test_graph_engine; then ((tests_passed++)); else ((tests_failed++)); fi
    
    # Summary
    print_header "Test Summary"
    
    echo "Tests Passed: $tests_passed / $tests_total"
    echo "Tests Failed: $tests_failed / $tests_total"
    
    if [ $tests_failed -eq 0 ]; then
        print_success "ALL TESTS PASSED! 🎉"
        print_info "HANA migration is validated and ready for production"
        return 0
    else
        print_error "SOME TESTS FAILED"
        print_info "Please review errors and fix issues before deployment"
        return 1
    fi
}

# ============================================================================
# Execute
# ============================================================================

main "$@"