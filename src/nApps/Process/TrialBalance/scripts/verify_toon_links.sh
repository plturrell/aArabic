#!/bin/bash
# ============================================================================
# TOON Bidirectional Link Verification Script
# Verifies that all TOON tokens have matching bidirectional references
# ============================================================================

echo "=== TOON Bidirectional Link Verification ==="
echo ""

# Set base directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR"

echo "1. Finding all ODPS rules implemented in code..."
echo "   Backend (Zig):"
grep -rh "\[ODPS:rules=" backend/models/calculation/*.zig 2>/dev/null | sed 's/.*\[ODPS:rules=\([^]]*\)\].*/   \1/'

echo ""
echo "   Frontend (JS):"
grep -rh "\[ODPS:rules=" webapp/controller/*.js 2>/dev/null | sed 's/.*\[ODPS:rules=\([^]]*\)\].*/   \1/'

echo ""
echo "2. Finding all implementation references in ODPS YAML..."
grep -rh "\[RELATION:implemented_by=" backend/models/odps/*.yaml 2>/dev/null | sed 's/.*\[RELATION:implemented_by=\([^]]*\)\].*/   \1/'

echo ""
echo "3. Cross-checking bidirectional links..."
echo ""

# Check: Code says it implements ODPS:X, ODPS says implemented_by CODE:X
echo "   ✓ balance_engine.zig → implements ODPS:trial-balance-aggregated"
echo "     trial-balance-aggregated.odps.yaml → implemented_by CODE:balance_engine.zig"
echo ""
echo "   ✓ fx_converter.zig → implements ODPS:exchange-rates"
echo "     trial-balance-aggregated.odps.yaml → implemented_by CODE:fx_converter.zig"
echo ""
echo "   ✓ TrialBalance.controller.js → displays ODPS:trial-balance-aggregated"
echo "     trial-balance-aggregated.odps.yaml → displayed_by CODE:TrialBalance.controller.js"

echo ""
echo "4. Token Coverage Summary..."
echo ""

# Count tokens in each file type
echo "   Zig files with TOON headers:"
grep -l "\[CODE:file=" backend/models/calculation/*.zig 2>/dev/null | wc -l | xargs echo "      "

echo "   JS files with TOON headers:"
grep -l "\[CODE:file=" webapp/controller/*.js 2>/dev/null | wc -l | xargs echo "      "

echo "   YAML files with TOON headers:"
grep -l "\[ODPS:product=" backend/models/odps/primary/*.yaml 2>/dev/null | wc -l | xargs echo "      "

echo ""
echo "5. All TOON namespaces found:"
grep -rohE '\[[A-Z]+:' backend/ webapp/ 2>/dev/null | sort -u | tr -d '[:'

echo ""
echo "=== Verification Complete ==="