-- ============================================================================
-- SAP BTP HANA Cloud - Schema and User Creation
-- ============================================================================
-- Purpose: Create NUCLEUS schema and application user for OpenAI Server
-- Execute as: DBADMIN user
-- Instance: d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com
-- ============================================================================

-- Step 1: Create NUCLEUS schema
-- ============================================================================
CREATE SCHEMA NUCLEUS;

COMMENT ON SCHEMA NUCLEUS IS 'Nucleus OpenAI Server - Application schema for model configurations, prompts, and analytics';

-- Step 2: Create application user NUCLEUS_APP
-- ============================================================================
-- IMPORTANT: Change this password after first login!
CREATE USER NUCLEUS_APP PASSWORD "NucleusApp2026!" NO FORCE_FIRST_PASSWORD_CHANGE;

-- Step 3: Grant schema ownership and permissions
-- ============================================================================

-- Grant full access to NUCLEUS schema
GRANT CREATE ANY ON SCHEMA NUCLEUS TO NUCLEUS_APP;
GRANT ALTER ON SCHEMA NUCLEUS TO NUCLEUS_APP;
GRANT DROP ON SCHEMA NUCLEUS TO NUCLEUS_APP;

-- Grant object-level permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA NUCLEUS TO NUCLEUS_APP;
GRANT EXECUTE ON SCHEMA NUCLEUS TO NUCLEUS_APP;
GRANT DEBUG ON SCHEMA NUCLEUS TO NUCLEUS_APP;

-- Grant system privileges needed for table/view/procedure creation
GRANT CREATE TABLE TO NUCLEUS_APP;
GRANT CREATE VIEW TO NUCLEUS_APP;
GRANT CREATE SEQUENCE TO NUCLEUS_APP;
GRANT CREATE PROCEDURE TO NUCLEUS_APP;
GRANT CREATE FUNCTION TO NUCLEUS_APP;
GRANT CREATE TRIGGER TO NUCLEUS_APP;

-- Grant monitoring privileges (read-only system views)
GRANT CATALOG READ TO NUCLEUS_APP;

-- Step 4: Set default schema for NUCLEUS_APP
-- ============================================================================
ALTER USER NUCLEUS_APP SET SCHEMA NUCLEUS;

-- Step 5: Grant access to system tables for metadata queries
-- ============================================================================
GRANT SELECT ON SYS.M_TABLES TO NUCLEUS_APP;
GRANT SELECT ON SYS.M_INDEXES TO NUCLEUS_APP;
GRANT SELECT ON SYS.VIEWS TO NUCLEUS_APP;
GRANT SELECT ON SYS.PROCEDURES TO NUCLEUS_APP;
GRANT SELECT ON SYS.M_DATABASE TO NUCLEUS_APP;

-- Step 6: Create role for easier permission management (optional)
-- ============================================================================
CREATE ROLE NUCLEUS_APP_ROLE;

GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA NUCLEUS TO NUCLEUS_APP_ROLE;
GRANT EXECUTE ON SCHEMA NUCLEUS TO NUCLEUS_APP_ROLE;

-- Assign role to user
GRANT NUCLEUS_APP_ROLE TO NUCLEUS_APP;

-- Step 7: Verification queries
-- ============================================================================

-- Verify schema created
SELECT SCHEMA_NAME, OWNER_NAME, CREATE_TIME 
FROM SCHEMAS 
WHERE SCHEMA_NAME = 'NUCLEUS';

-- Verify user created
SELECT USER_NAME, CREATOR, CREATE_TIME, USER_DEACTIVATED 
FROM USERS 
WHERE USER_NAME = 'NUCLEUS_APP';

-- Verify permissions
SELECT GRANTEE, GRANTOR, PRIVILEGE, IS_GRANTABLE
FROM GRANTED_PRIVILEGES
WHERE GRANTEE = 'NUCLEUS_APP'
ORDER BY PRIVILEGE;

-- Verify schema access
SELECT GRANTEE, SCHEMA_NAME, PRIVILEGE
FROM GRANTED_SCHEMA_PRIVILEGES  
WHERE GRANTEE = 'NUCLEUS_APP'
ORDER BY SCHEMA_NAME, PRIVILEGE;

COMMIT;

-- ============================================================================
-- NEXT STEPS
-- ============================================================================
-- After running this script as DBADMIN:
-- 
-- 1. Verify schema and user creation with queries above
-- 2. Connect as NUCLEUS_APP to test permissions
-- 3. Run prompt_modes_schema.sql to create base tables
-- 4. Run nucleus_schema_extensions.sql to create additional tables
-- 5. Update .env file with NUCLEUS_APP credentials
-- 
-- Connection string for NUCLEUS_APP:
-- Host: d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com
-- Port: 443
-- User: NUCLEUS_APP
-- Password: NucleusApp2026!
-- Database: SYSTEMDB
-- Schema: NUCLEUS
-- SSL: Required
-- ============================================================================

-- Security recommendations:
-- 1. Change NUCLEUS_APP password after first login
-- 2. Enable password policy for regular password changes
-- 3. Consider certificate-based authentication for production
-- 4. Audit DBADMIN usage - use NUCLEUS_APP for application
-- 5. Enable audit logging for NUCLEUS schema
-- ============================================================================
