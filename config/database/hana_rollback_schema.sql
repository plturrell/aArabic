-- ============================================================================
-- SAP HANA Schema Rollback Script
-- Created: January 24, 2026
-- Purpose: Clean rollback of migration if needed
-- ============================================================================

-- WARNING: This script will DROP all tables and data!
-- Only use this for:
-- 1. Rolling back a failed migration
-- 2. Cleaning up a test/staging environment
-- 3. Starting fresh after testing

-- ============================================================================
-- Pre-Rollback Verification
-- ============================================================================

-- Check current table count
SELECT 
    'PRE-ROLLBACK' as phase,
    COUNT(*) as table_count,
    SUM(RECORD_COUNT) as total_records
FROM TABLES 
WHERE SCHEMA_NAME = 'NOPENAI_DB';

-- List all tables that will be dropped
SELECT TABLE_NAME, RECORD_COUNT
FROM TABLES 
WHERE SCHEMA_NAME = 'NOPENAI_DB'
ORDER BY TABLE_NAME;

-- ============================================================================
-- Step 1: Drop Views (Must drop before tables)
-- ============================================================================

DROP VIEW IF EXISTS V_CACHE_STATISTICS CASCADE;
DROP VIEW IF EXISTS V_WORKFLOW_STATISTICS CASCADE;

-- ============================================================================
-- Step 2: Drop Procedures
-- ============================================================================

DROP PROCEDURE IF EXISTS CLEANUP_EXPIRED_CACHE;

-- ============================================================================
-- Step 3: Drop Tables (Order matters due to foreign keys)
-- ============================================================================

-- Drop graph tables first (have foreign keys)
DROP TABLE IF EXISTS GRAPH_EDGES CASCADE;
DROP TABLE IF EXISTS GRAPH_NODES CASCADE;

-- Drop nLocalModels tables
DROP TABLE IF EXISTS TENSOR_STORAGE CASCADE;
DROP TABLE IF EXISTS SESSION_STATE CASCADE;
DROP TABLE IF EXISTS PROMPT_CACHE CASCADE;
DROP TABLE IF EXISTS KV_CACHE CASCADE;

-- Drop nAgentFlow tables
DROP TABLE IF EXISTS EXECUTION_LOG CASCADE;
DROP TABLE IF EXISTS WORKFLOW_CACHE CASCADE;
DROP TABLE IF EXISTS WORKFLOW_STATE CASCADE;

-- Drop nAgentMeta tables
DROP TABLE IF EXISTS AGENT_METADATA CASCADE;

-- Drop common tables
DROP TABLE IF EXISTS HEALTH_STATUS CASCADE;
DROP TABLE IF EXISTS SYSTEM_CONFIG CASCADE;

-- ============================================================================
-- Post-Rollback Verification
-- ============================================================================

-- Verify all tables are dropped
SELECT 
    'POST-ROLLBACK' as phase,
    COUNT(*) as remaining_tables
FROM TABLES 
WHERE SCHEMA_NAME = 'NOPENAI_DB';

-- Should return 0 tables
SELECT TABLE_NAME
FROM TABLES 
WHERE SCHEMA_NAME = 'NOPENAI_DB';

-- ============================================================================
-- Rollback Complete Message
-- ============================================================================

SELECT 
    'ROLLBACK COMPLETE' as status,
    CURRENT_TIMESTAMP as completed_at,
    'All migration tables have been dropped' as message
FROM DUMMY;

-- ============================================================================
-- Next Steps After Rollback
-- ============================================================================

-- If rolling back due to issues:
-- 1. Review error logs
-- 2. Fix identified issues
-- 3. Re-run hana_migration_schema.sql
-- 4. Re-test integration

-- If cleaning up test environment:
-- 1. You can now re-run the migration script
-- 2. Or leave environment clean

-- End of rollback script