#!/bin/bash
# ============================================================================
# Trial Balance Schema Deployment Script
# Deploys HANA schema to SAP HANA Cloud
# ============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_DIR="$(dirname "$SCRIPT_DIR")/hana"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/deployment_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v hdbsql &> /dev/null; then
        error "hdbsql not found. Please install SAP HANA Client."
        exit 1
    fi
    
    if [ -z "$HANA_HOST" ]; then
        error "HANA_HOST environment variable not set"
        exit 1
    fi
    
    if [ -z "$HANA_USER" ]; then
        error "HANA_USER environment variable not set"
        exit 1
    fi
    
    if [ -z "$HANA_PASSWORD" ]; then
        error "HANA_PASSWORD environment variable not set"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Execute SQL file
execute_sql() {
    local sql_file=$1
    local description=$2
    
    log "Executing: $description"
    log "File: $sql_file"
    
    hdbsql -n "$HANA_HOST:$HANA_PORT" \
           -u "$HANA_USER" \
           -p "$HANA_PASSWORD" \
           -d "$HANA_DATABASE" \
           -I "$sql_file" \
           >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully executed: $description"
        return 0
    else
        error "✗ Failed to execute: $description"
        return 1
    fi
}

# Create schema
create_schema() {
    log "Creating schema TB_SCHEMA..."
    
    hdbsql -n "$HANA_HOST:$HANA_PORT" \
           -u "$HANA_USER" \
           -p "$HANA_PASSWORD" \
           -d "$HANA_DATABASE" \
           "CREATE SCHEMA TB_SCHEMA;" \
           >> "$LOG_FILE" 2>&1 || true
    
    log "Schema created or already exists"
}

# Deploy tables
deploy_tables() {
    log "Deploying tables..."
    
    execute_sql "$SCHEMA_DIR/tables/01_tb_core_tables.sql" "Core Tables"
    execute_sql "$SCHEMA_DIR/tables/02_tb_audit_exception_tables.sql" "Audit & Exception Tables"
    
    log "Tables deployed successfully"
}

# Deploy calculation views
deploy_calc_views() {
    log "Deploying calculation views..."
    
    execute_sql "$SCHEMA_DIR/calc_views/cv_account_balances.sql" "Account Balances View"
    execute_sql "$SCHEMA_DIR/calc_views/cv_multicurrency_balances.sql" "Multi-Currency Balances View"
    execute_sql "$SCHEMA_DIR/calc_views/cv_ifrs_summary.sql" "IFRS Summary View"
    
    log "Calculation views deployed successfully"
}

# Deploy stored procedures
deploy_procedures() {
    log "Deploying stored procedures..."
    
    if [ -d "$SCHEMA_DIR/procedures" ] && [ "$(ls -A $SCHEMA_DIR/procedures/*.sql 2>/dev/null)" ]; then
        for proc_file in "$SCHEMA_DIR/procedures"/*.sql; do
            local proc_name=$(basename "$proc_file" .sql)
            execute_sql "$proc_file" "Procedure: $proc_name"
        done
        log "Stored procedures deployed successfully"
    else
        warn "No stored procedures found to deploy"
    fi
}

# Create users and grant permissions
setup_permissions() {
    log "Setting up permissions..."
    
    # Create application user if not exists
    hdbsql -n "$HANA_HOST:$HANA_PORT" \
           -u "$HANA_USER" \
           -p "$HANA_PASSWORD" \
           -d "$HANA_DATABASE" \
           "CREATE USER TB_APP_USER PASSWORD \"${TB_APP_PASSWORD:-ChangeMe123}\" NO FORCE_FIRST_PASSWORD_CHANGE;" \
           >> "$LOG_FILE" 2>&1 || true
    
    # Grant schema access
    hdbsql -n "$HANA_HOST:$HANA_PORT" \
           -u "$HANA_USER" \
           -p "$HANA_PASSWORD" \
           -d "$HANA_DATABASE" \
           "GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA TB_SCHEMA TO TB_APP_USER;" \
           >> "$LOG_FILE" 2>&1
    
    log "Permissions configured successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    local table_count=$(hdbsql -n "$HANA_HOST:$HANA_PORT" \
                               -u "$HANA_USER" \
                               -p "$HANA_PASSWORD" \
                               -d "$HANA_DATABASE" \
                               -C \
                               "SELECT COUNT(*) FROM TABLES WHERE SCHEMA_NAME = 'TB_SCHEMA';" \
                               2>> "$LOG_FILE" | tail -1)
    
    log "Tables created: $table_count"
    
    if [ "$table_count" -lt 9 ]; then
        warn "Expected at least 9 tables, found $table_count"
    else
        log "✓ Table count verification passed"
    fi
}

# Main deployment function
main() {
    log "====================================================================="
    log "Trial Balance Schema Deployment"
    log "====================================================================="
    log "Target: $HANA_HOST"
    log "Database: ${HANA_DATABASE:-DBADMIN}"
    log "Schema: TB_SCHEMA"
    log "====================================================================="
    
    # Set defaults
    export HANA_PORT="${HANA_PORT:-443}"
    export HANA_DATABASE="${HANA_DATABASE:-DBADMIN}"
    
    check_prerequisites
    
    log "Starting deployment..."
    
    create_schema
    deploy_tables
    deploy_calc_views
    deploy_procedures
    setup_permissions
    verify_deployment
    
    log "====================================================================="
    log "✓ Deployment completed successfully!"
    log "Log file: $LOG_FILE"
    log "====================================================================="
}

# Run main function
main "$@"