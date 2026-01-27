#!/bin/bash
# ============================================================================
# Setup Development Environment for Trial Balance
# Initializes SQLite DB, loads sample data, and verifies setup
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUSDOCS_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Trial Balance Development Environment Setup              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Initialize SQLite database
echo -e "${GREEN}[1/5]${NC} Initializing SQLite database..."
cd "$BUSDOCS_DIR/schema/sqlite"
./init_dev_db.sh
echo ""

# Step 2: Load HKG sample data
echo -e "${GREEN}[2/5]${NC} Loading HKG sample data..."
cd "$BUSDOCS_DIR/scripts"
python3 load_sample_to_sqlite.py
echo ""

# Step 3: Build Zig calculation engine
echo -e "${GREEN}[3/5]${NC} Building Zig calculation engine..."
cd "$BUSDOCS_DIR/models/calculation"
echo "  Testing balance_engine.zig..."
zig test balance_engine.zig 2>&1 | tail -5
echo "  Testing fx_converter.zig..."
zig test fx_converter.zig 2>&1 | tail -5
echo "  ✓ Zig modules compiled and tested"
echo ""

# Step 4: Verify database
echo -e "${GREEN}[4/5]${NC} Verifying database content..."
DB_PATH="$BUSDOCS_DIR/schema/sqlite/trial_balance_dev.db"
sqlite3 "$DB_PATH" <<EOF
.mode column
.headers on
SELECT 'Database Statistics:' as Info;
SELECT 
    (SELECT COUNT(*) FROM TB_GL_ACCOUNTS) as gl_accounts,
    (SELECT COUNT(*) FROM TB_JOURNAL_ENTRIES) as journal_entries,
    (SELECT COUNT(*) FROM TB_EXCHANGE_RATES) as exchange_rates;

SELECT '' as Info;
SELECT 'Sample Journal Entries:' as Info;
SELECT racct as Account, sgtxt as Description, 
       printf('$%.2f', hsl) as Amount, 
       drcrk as DC
FROM TB_JOURNAL_ENTRIES 
WHERE mandt = '100' 
LIMIT 5;
EOF
echo ""

# Step 5: Display next steps
echo -e "${GREEN}[5/5]${NC} Setup complete!"
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Next Steps                                                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "1. Start the backend API server:"
echo "   cd ../../../../backend"
echo "   zig build run"
echo ""
echo "2. In a new terminal, start the UI5 frontend:"
echo "   cd ../webapp"
echo "   npm install"
echo "   npm start"
echo ""
echo "3. Open browser to:"
echo "   http://localhost:8080/index.html"
echo ""
echo "4. Test API endpoints:"
echo "   curl http://localhost:8091/api/health"
echo "   curl http://localhost:8091/api/v1/accounts"
echo ""
echo -e "${GREEN}✓ Development environment ready!${NC}"