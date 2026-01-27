#!/bin/bash
# ============================================================================
# SQLite Development Database Initialization Script
# Creates and seeds a local SQLite database for development
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_FILE="$SCRIPT_DIR/trial_balance_dev.db"
LOG_FILE="$SCRIPT_DIR/init_dev_db.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if sqlite3 is available
if ! command -v sqlite3 &> /dev/null; then
    echo "Error: sqlite3 not found. Please install SQLite3."
    exit 1
fi

log "====================================================================="
log "Trial Balance SQLite Development Database Initialization"
log "====================================================================="

# Remove existing database if it exists
if [ -f "$DB_FILE" ]; then
    warn "Removing existing database: $DB_FILE"
    rm "$DB_FILE"
fi

log "Creating new database: $DB_FILE"

# Create tables
log "Creating core tables..."
sqlite3 "$DB_FILE" < "$SCRIPT_DIR/01_tb_core_tables.sql"

log "Creating audit and exception tables..."
sqlite3 "$DB_FILE" < "$SCRIPT_DIR/02_tb_audit_exception_tables.sql"

log "Creating views..."
sqlite3 "$DB_FILE" < "$SCRIPT_DIR/03_tb_views.sql"

# Insert sample data
log "Inserting sample data..."
sqlite3 "$DB_FILE" <<EOF
-- Sample GL Accounts
INSERT INTO TB_GL_ACCOUNTS (account_id, mandt, saknr, ktopl, txt50, ifrs_schedule, ifrs_category, account_type)
VALUES 
    ('ACC001', '100', '100000', 'IFRS', 'Cash', '1A', 'Cash & Central Bank', 'Asset'),
    ('ACC002', '100', '200000', 'IFRS', 'Accounts Payable', '02', 'Liabilities', 'Liability'),
    ('ACC003', '100', '300000', 'IFRS', 'Share Capital', '2O', 'Share Capital', 'Equity'),
    ('ACC004', '100', '400000', 'IFRS', 'Interest Income', '3D', 'Interest Income', 'Income'),
    ('ACC005', '100', '500000', 'IFRS', 'Operating Expenses', '3L', 'Operating Costs', 'Expense');

-- Sample Exchange Rates
INSERT INTO TB_EXCHANGE_RATES (rate_id, mandt, kurst, fcurr, tcurr, gdatu, ukurs)
VALUES 
    ('FX001', '100', 'M', 'EUR', 'USD', '2025-01-31', 1.0845),
    ('FX002', '100', 'M', 'GBP', 'USD', '2025-01-31', 1.2650),
    ('FX003', '100', 'M', 'SGD', 'USD', '2025-01-31', 0.7420),
    ('FX004', '100', 'M', 'HKD', 'USD', '2025-01-31', 0.1283);

-- Sample Journal Entries
INSERT INTO TB_JOURNAL_ENTRIES 
    (entry_id, mandt, rldnr, rbukrs, gjahr, belnr, buzei, budat, racct, drcrk, hsl, rtcur, poper, validated)
VALUES 
    ('JE001', '100', '0L', '1000', '2025', 'DOC001', '001', '2025-01-15', '100000', 'S', 1000000.00, 'USD', '001', 1),
    ('JE002', '100', '0L', '1000', '2025', 'DOC001', '002', '2025-01-15', '200000', 'H', 1000000.00, 'USD', '001', 1),
    ('JE003', '100', '0L', '1000', '2025', 'DOC002', '001', '2025-01-20', '400000', 'H', 50000.00, 'USD', '001', 1),
    ('JE004', '100', '0L', '1000', '2025', 'DOC002', '002', '2025-01-20', '100000', 'S', 50000.00, 'USD', '001', 1);
EOF

log "Sample data inserted successfully"

# Verify database
log "Verifying database..."
sqlite3 "$DB_FILE" <<EOF
.mode column
.headers on
SELECT 'Tables:' as Info;
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;

SELECT '' as Info;
SELECT 'Views:' as Info;
SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;

SELECT '' as Info;
SELECT 'GL Accounts:' as Info;
SELECT COUNT(*) as count FROM TB_GL_ACCOUNTS;

SELECT '' as Info;
SELECT 'Journal Entries:' as Info;
SELECT COUNT(*) as count FROM TB_JOURNAL_ENTRIES;

SELECT '' as Info;
SELECT 'Exchange Rates:' as Info;
SELECT COUNT(*) as count FROM TB_EXCHANGE_RATES;
EOF

log "====================================================================="
log "✓ Database initialized successfully!"
log "Database location: $DB_FILE"
log "Log file: $LOG_FILE"
log ""
log "To connect: sqlite3 $DB_FILE"
log "====================================================================="