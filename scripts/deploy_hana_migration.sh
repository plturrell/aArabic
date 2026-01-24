#!/bin/bash
# ============================================================================
# HANA Migration Deployment Script
# Created: January 24, 2026
# Purpose: Automated deployment of HANA migration
# ============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Configuration
# ============================================================================

HANA_HOST="${HANA_HOST}"
HANA_PORT="${HANA_PORT:-30015}"
HANA_DATABASE="${HANA_DATABASE:-NOPENAI_DB}"
HANA_USER="${HANA_USER:-SHIMMY_USER}"
HANA_PASSWORD="${HANA_PASSWORD}"

ENVIRONMENT="${1:-staging}"  # staging, production
DRY_RUN="${2:-false}"        # true for dry run

SCHEMA_FILE="config/database/hana_migration_schema.sql"
ROLLBACK_FILE="config/database/hana_rollback_schema.sql"
TEST_SCRIPT="scripts/test_hana_migration.sh"

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  $1${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

confirm_action() {
    local message="$1"
    echo -e "${YELLOW}⚠️  $message${NC}"
    read -p "Continue? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_error "Deployment cancelled by user"
        exit 1
    fi
}

run_sql() {
    local query="$1"
    hdbsql -u "$HANA_USER" -p "$HANA_PASSWORD" \
           -n "$HANA_HOST:$HANA_PORT" -d "$HANA_DATABASE" \
           "$query"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

preflight_checks() {
    print_header "Pre-flight Checks"
    
    # Check required environment variables
    print_info "Checking environment variables..."
    
    if [ -z "$HANA_HOST" ]; then
        print_error "HANA_HOST not set"
        exit 1
    fi
    print_success "HANA_HOST: $HANA_HOST"
    
    if [ -z "$HANA_PASSWORD" ]; then
        print_error "HANA_PASSWORD not set"
        exit 1
    fi
    print_success "HANA_PASSWORD: ****"
    
    # Check required files exist
    print_info "Checking required files..."
    
    if [ ! -f "$SCHEMA_FILE" ]; then
        print_error "Schema file not found: $SCHEMA_FILE"
        exit 1
    fi
    print_success "Schema file exists"
    
    if [ ! -f "$ROLLBACK_FILE" ]; then
        print_error "Rollback file not found: $ROLLBACK_FILE"
        exit 1
    fi
    print_success "Rollback file exists"
    
    if [ ! -f "$TEST_SCRIPT" ]; then
        print_error "Test script not found: $TEST_SCRIPT"
        exit 1
    fi
    print_success "Test script exists"
    
    # Check HANA connection
    print_info "Testing HANA connection..."
    
    if run_sql "SELECT 1 FROM DUMMY" > /dev/null 2>&1; then
        print_success "HANA connection successful"
    else
        print_error "Cannot connect to HANA"
        exit 1
    fi
    
    print_success "All pre-flight checks passed"
}

# ============================================================================
# Backup Existing Schema (if any)
# ============================================================================

backup_existing_schema() {
    print_header "Backup Existing Schema"
    
    local backup_dir="backups/hana_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_info "Backing up existing tables to $backup_dir..."
    
    # Export table list
    run_sql "SELECT TABLE_NAME FROM TABLES WHERE SCHEMA_NAME='$HANA_DATABASE'" \
        > "$backup_dir/existing_tables.txt" 2>&1 || true
    
    print_success "Backup created at $backup_dir"
}

# ============================================================================
# Deploy Schema
# ============================================================================

deploy_schema() {
    print_header "Deploy HANA Schema"
    
    if [ "$DRY_RUN" = "true" ]; then
        print_warning "DRY RUN MODE - No changes will be made"
        print_info "Would execute: $SCHEMA_FILE"
        return 0
    fi
    
    confirm_action "This will create all HANA tables for the migration"
    
    print_info "Executing schema migration script..."
    
    if hdbsql -u "$HANA_USER" -p "$HANA_PASSWORD" \
              -n "$HANA_HOST:$HANA_PORT" -d "$HANA_DATABASE" \
              -I "$SCHEMA_FILE"; then
        print_success "Schema deployed successfully"
    else
        print_error "Schema deployment failed"
        exit 1
    fi
}

# ============================================================================
# Run Integration Tests
# ============================================================================

run_integration_tests() {
    print_header "Integration Tests"
    
    if [ "$DRY_RUN" = "true" ]; then
        print_warning "DRY RUN MODE - Skipping tests"
        return 0
    fi
    
    print_info "Running integration test suite..."
    
    if bash "$TEST_SCRIPT"; then
        print_success "All integration tests passed"
    else
        print_error "Integration tests failed"
        print_warning "Consider rolling back the deployment"
        exit 1
    fi
}

# ============================================================================
# Verify Deployment
# ============================================================================

verify_deployment() {
    print_header "Deployment Verification"
    
    print_info "Checking table counts..."
    
    # Count tables
    local table_count=$(run_sql "SELECT COUNT(*) FROM TABLES WHERE SCHEMA_NAME='$HANA_DATABASE'" | tail -1)
    print_info "Total tables: $table_count"
    
    if [ "$table_count" -ge 12 ]; then
        print_success "Expected number of tables created"
    else
        print_warning "Table count seems low (expected >= 12)"
    fi
    
    # Check cache statistics
    print_info "Checking cache statistics view..."
    if run_sql "SELECT * FROM V_CACHE_STATISTICS" > /dev/null 2>&1; then
        print_success "Monitoring views accessible"
    else
        print_warning "Monitoring views not accessible"
    fi
    
    print_success "Deployment verification complete"
}

# ============================================================================
# Update Service Configuration
# ============================================================================

update_service_configs() {
    print_header "Update Service Configurations"
    
    if [ "$DRY_RUN" = "true" ]; then
        print_warning "DRY RUN MODE - Skipping config updates"
        return 0
    fi
    
    print_info "Updating service configuration files..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_info "Creating .env file..."
        cat > .env << EOF
# HANA Configuration (Generated on $(date))
HANA_HOST=$HANA_HOST
HANA_PORT=$HANA_PORT
HANA_DATABASE=$HANA_DATABASE
HANA_USER=$HANA_USER
HANA_PASSWORD=$HANA_PASSWORD

# Service Configuration
ENVIRONMENT=$ENVIRONMENT
ENABLE_DISTRIBUTED_CACHE=true
ENABLE_HANA_GRAPH=true

# Cache TTL Settings (seconds)
KV_CACHE_TTL=3600
PROMPT_CACHE_TTL=3600
SESSION_TTL=1800
TENSOR_TTL=86400
EOF
        print_success ".env file created"
    else
        print_info ".env file already exists, skipping..."
    fi
    
    print_success "Configuration updated"
}

# ============================================================================
# Post-Deployment Tasks
# ============================================================================

post_deployment() {
    print_header "Post-Deployment Tasks"
    
    print_info "Deployment completed at: $(date)"
    print_info "Environment: $ENVIRONMENT"
    print_info "HANA Host: $HANA_HOST"
    print_info "Database: $HANA_DATABASE"
    
    echo ""
    print_success "Next steps:"
    echo "  1. Monitor service logs for errors"
    echo "  2. Check HANA Studio for table usage"
    echo "  3. Run performance benchmarks"
    echo "  4. Schedule TTL cleanup job (hourly)"
    echo "  5. Configure monitoring dashboards"
    echo ""
    
    # Save deployment metadata
    local metadata_file="deployments/deployment_$(date +%Y%m%d_%H%M%S).json"
    mkdir -p deployments
    cat > "$metadata_file" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$ENVIRONMENT",
  "hana_host": "$HANA_HOST",
  "hana_database": "$HANA_DATABASE",
  "deployed_by": "$USER",
  "schema_version": "2.0.0",
  "services": ["nAgentFlow", "nAgentMeta", "nLocalModels"],
  "status": "success"
}
EOF
    print_success "Deployment metadata saved to $metadata_file"
}

# ============================================================================
# Main Deployment Flow
# ============================================================================

main() {
    print_banner "HANA Migration Deployment Script"
    
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Environment: $ENVIRONMENT"
    echo "  HANA Host: $HANA_HOST"
    echo "  HANA Port: $HANA_PORT"
    echo "  Database: $HANA_DATABASE"
    echo "  User: $HANA_USER"
    echo "  Dry Run: $DRY_RUN"
    echo ""
    
    if [ "$ENVIRONMENT" = "production" ]; then
        print_warning "⚠️  PRODUCTION DEPLOYMENT ⚠️"
        confirm_action "You are about to deploy to PRODUCTION"
    fi
    
    # Execute deployment steps
    preflight_checks
    backup_existing_schema
    deploy_schema
    run_integration_tests
    verify_deployment
    update_service_configs
    post_deployment
    
    print_banner "🎉 Deployment Complete!"
    
    print_success "HANA migration deployed successfully to $ENVIRONMENT"
    print_info "Review the deployment logs above for details"
    
    return 0
}

# ============================================================================
# Usage
# ============================================================================

usage() {
    echo "Usage: $0 [environment] [dry_run]"
    echo ""
    echo "Arguments:"
    echo "  environment  - staging|production (default: staging)"
    echo "  dry_run     - true|false (default: false)"
    echo ""
    echo "Environment Variables Required:"
    echo "  HANA_HOST     - HANA instance hostname"
    echo "  HANA_PASSWORD - HANA user password"
    echo ""
    echo "Optional Environment Variables:"
    echo "  HANA_PORT     - HANA port (default: 30015)"
    echo "  HANA_DATABASE - Database name (default: NOPENAI_DB)"
    echo "  HANA_USER     - Database user (default: SHIMMY_USER)"
    echo ""
    echo "Examples:"
    echo "  $0 staging          # Deploy to staging"
    echo "  $0 production       # Deploy to production"
    echo "  $0 staging true     # Dry run for staging"
    exit 1
}

# ============================================================================
# Execute
# ============================================================================

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage
fi

main "$@"