# SAP BTP HANA Cloud Setup Guide
**Nucleus OpenAI Server - Database Configuration**

---

## Overview

This guide provides step-by-step instructions for setting up the Nucleus OpenAI Server database schema on SAP BTP HANA Cloud.

**Your HANA Cloud Instance:**
- **Host:** `d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com`
- **Port:** 443 (HTTPS/SSL)
- **Region:** prod-us10 (US East)
- **Admin User:** DBADMIN
- **Admin Password:** Initial@1

---

## Prerequisites

1. SAP BTP HANA Cloud instance (already provisioned ✓)
2. DBADMIN credentials (provided ✓)
3. SQL client (choose one):
   - SAP HANA Database Explorer (recommended for BTP Cloud)
   - hdbsql CLI tool
   - DBeaver with HANA JDBC driver
   - SAP HANA Studio

---

## Step 1: Connect to HANA Cloud

### Option A: SAP HANA Database Explorer (Recommended)

1. Log into SAP BTP Cockpit: https://cockpit.hanatrial.ondemand.com/ (or your region)
2. Navigate to your subaccount
3. Go to **SAP HANA Cloud** → **Actions** → **Open in SAP HANA Database Explorer**
4. You'll be automatically authenticated

### Option B: hdbsql CLI

```bash
# Install SAP HANA Client (if not installed)
# Download from: https://tools.hana.ondemand.com/

# Connect to HANA Cloud
hdbsql \
  -n d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com:443 \
  -u DBADMIN \
  -p Initial@1 \
  -d SYSTEMDB \
  -e -sslprovider commoncrypto -ssltrustcert
```

### Option C: DBeaver

1. Install DBeaver: https://dbeaver.io/
2. Create new connection → SAP HANA
3. Connection settings:
   - **Host:** `d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com`
   - **Port:** 443
   - **Database:** SYSTEMDB
   - **User:** DBADMIN
   - **Password:** Initial@1
   - **Use SSL:** ✓ Yes
   - **Validate certificate:** ✓ Yes (production) / ✗ No (testing)

---

## Step 2: Create Schema and User

Execute the SQL script to create the NUCLEUS schema and application user:

```sql
-- File: config/database/00_create_schema.sql
-- Execute as DBADMIN user

-- This script will:
-- 1. Create NUCLEUS schema
-- 2. Create NUCLEUS_APP user (password: NucleusApp2026!)
-- 3. Grant all necessary permissions
-- 4. Set default schema
-- 5. Create application role
```

**Execution:**
1. Open `00_create_schema.sql` in your SQL client
2. Ensure you're connected as DBADMIN
3. Execute the entire script
4. Verify the verification queries at the end show successful creation

**Expected Output:**
```
Schema NUCLEUS created
User NUCLEUS_APP created
Grants successful
```

---

## Step 3: Create Base Tables (Prompt Modes)

Execute the prompt modes schema to create the foundation tables:

```sql
-- File: config/database/prompt_modes_schema.sql
-- Execute as NUCLEUS_APP or DBADMIN

-- This script creates:
-- - PROMPT_MODE_CONFIGS (30 columns)
-- - PROMPT_HISTORY (27 columns)
-- - MODEL_PERFORMANCE (22 columns)
-- - MODE_PRESETS (17 columns) with preset data
-- - 2 views for common queries
```

**Execution:**
1. Connect as NUCLEUS_APP user (recommended) or stay as DBADMIN
2. Set schema to NUCLEUS: `SET SCHEMA NUCLEUS;`
3. Open `prompt_modes_schema.sql`
4. Execute the entire script
5. Verify tables created: `SELECT TABLE_NAME FROM M_TABLES WHERE SCHEMA_NAME = 'NUCLEUS';`

**Expected Tables:**
- PROMPT_MODE_CONFIGS
- PROMPT_HISTORY
- MODEL_PERFORMANCE
- MODE_PRESETS

---

## Step 4: Create Extension Tables (Days 1-3 Features)

Execute the extension schema to create additional application tables:

```sql
-- File: config/database/nucleus_schema_extensions.sql
-- Execute as NUCLEUS_APP or DBADMIN

-- This script creates:
-- - MODEL_CONFIGURATIONS (18 columns)
-- - USER_SETTINGS (24 columns)
-- - NOTIFICATIONS (15 columns)
-- - PROMPT_COMPARISONS (30 columns)
-- - MODEL_VERSIONS (15 columns)
-- - MODEL_VERSION_COMPARISONS (25 columns)
-- - TRAINING_EXPERIMENTS (25 columns)
-- - TRAINING_EXPERIMENT_COMPARISONS (27 columns)
-- - AUDIT_LOG (12 columns)
-- - 4 additional views
-- - 3 stored procedures
-- - 2 triggers
```

**Execution:**
1. Ensure NUCLEUS schema is set
2. Open `nucleus_schema_extensions.sql`
3. Execute the entire script (may take 1-2 minutes)
4. Verify all objects created

**Expected Tables (Total: 13 tables):**
- MODEL_CONFIGURATIONS
- USER_SETTINGS
- NOTIFICATIONS
- PROMPT_COMPARISONS
- MODEL_VERSIONS
- MODEL_VERSION_COMPARISONS
- TRAINING_EXPERIMENTS
- TRAINING_EXPERIMENT_COMPARISONS
- AUDIT_LOG

---

## Step 5: Verify Installation

Run these verification queries to ensure everything is set up correctly:

```sql
-- 1. Verify all tables created (should return 13 rows)
SELECT TABLE_NAME, RECORD_COUNT 
FROM M_TABLES 
WHERE SCHEMA_NAME = 'NUCLEUS'
ORDER BY TABLE_NAME;

-- 2. Verify indexes created (should return ~30 rows)
SELECT TABLE_NAME, INDEX_NAME, INDEX_TYPE
FROM M_INDEXES
WHERE SCHEMA_NAME = 'NUCLEUS'
ORDER BY TABLE_NAME, INDEX_NAME;

-- 3. Verify views created (should return 6 views)
SELECT VIEW_NAME, CREATE_TIME
FROM VIEWS
WHERE SCHEMA_NAME = 'NUCLEUS'
ORDER BY VIEW_NAME;

-- 4. Verify procedures created (should return 3 procedures)
SELECT PROCEDURE_NAME, CREATE_TIME
FROM PROCEDURES
WHERE SCHEMA_NAME = 'NUCLEUS'
ORDER BY PROCEDURE_NAME;

-- 5. Verify preset data inserted (should return 4 rows)
SELECT MODE_NAME, DISPLAY_NAME, PRIORITY_ORDER
FROM NUCLEUS.MODE_PRESETS
ORDER BY PRIORITY_ORDER;

-- Expected preset data:
-- Fast Mode (Priority 1)
-- Normal Mode (Priority 2)
-- Expert Mode (Priority 3)
-- Research Mode (Priority 4)
```

---

## Step 6: Update Application Configuration

Update your `.env` file with the connection details:

```bash
# SAP BTP HANA Cloud Connection
HANA_HOST=d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com
HANA_PORT=443
HANA_DATABASE=SYSTEMDB
HANA_SCHEMA=NUCLEUS
HANA_USER=NUCLEUS_APP
HANA_PASSWORD=NucleusApp2026!
HANA_ENCRYPT=true

# Connection Pool
HANA_POOL_MIN=2
HANA_POOL_MAX=10
HANA_CONNECTION_TIMEOUT_MS=10000
HANA_QUERY_TIMEOUT_MS=60000
```

---

## Step 7: Test Application Connection

Use the connection test script to verify your application can connect:

```bash
# Test connection from application
python scripts/test_hana_connection.py

# Expected output:
# ✓ Connected to HANA Cloud
# ✓ Schema NUCLEUS accessible
# ✓ 13 tables found
# ✓ All permissions verified
# ✓ Connection successful!
```

---

## Security Best Practices

### Immediate Actions:

1. **Change Default Passwords:**
   ```sql
   -- Change DBADMIN password
   ALTER USER DBADMIN PASSWORD "NewSecurePassword123!";
   
   -- Change NUCLEUS_APP password
   ALTER USER NUCLEUS_APP PASSWORD "NewAppPassword456!";
   ```

2. **Enable Password Policy:**
   ```sql
   ALTER USER NUCLEUS_APP FORCE PASSWORD CHANGE;
   ```

3. **Restrict DBADMIN Usage:**
   - Use DBADMIN only for schema changes
   - Use NUCLEUS_APP for application runtime
   - Create read-only user for reporting/analytics

### Long-term Recommendations:

1. **Enable Audit Logging:**
   ```sql
   ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM') 
   SET ('auditing', 'audit_all_read') = 'true' WITH RECONFIGURE;
   ```

2. **Certificate-Based Authentication:**
   - Generate X.509 certificates
   - Configure HANA Cloud to use certificate auth
   - Disable password authentication for NUCLEUS_APP

3. **Network Security:**
   - Use SAP Cloud Connector for on-premise connectivity
   - Configure IP allowlists in BTP Cloud Foundry
   - Enable VPN/private link for production

4. **Backup Configuration:**
   - Enable automatic backups in BTP console
   - Test backup restoration process
   - Configure backup retention policy (30+ days)

5. **Monitoring:**
   - Enable SAP HANA Cloud Central monitoring
   - Configure alerts for:
     - Connection failures
     - High memory usage
     - Failed login attempts
     - Schema changes

---

## Troubleshooting

### Issue: Cannot connect to HANA Cloud

**Solution:**
1. Verify HANA Cloud instance is running (BTP Cockpit)
2. Check hostname is correct (no typos in UUID)
3. Ensure port 443 is not blocked by firewall
4. Verify SSL/TLS is enabled in connection settings
5. Check credentials are correct

```bash
# Test basic connectivity
curl -k https://d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com:443

# Expected: Connection established (may show error page, but connection works)
```

### Issue: Permission denied errors

**Solution:**
1. Ensure you're connected as NUCLEUS_APP user
2. Verify schema is set: `SET SCHEMA NUCLEUS;`
3. Check grants: `SELECT * FROM GRANTED_PRIVILEGES WHERE GRANTEE = 'NUCLEUS_APP';`
4. Re-run `00_create_schema.sql` if needed

### Issue: Table already exists error

**Solution:**
```sql
-- Drop and recreate (CAUTION: Data loss!)
DROP TABLE NUCLEUS.TABLE_NAME CASCADE;

-- Or check if table exists first
SELECT * FROM M_TABLES WHERE SCHEMA_NAME = 'NUCLEUS' AND TABLE_NAME = 'TABLE_NAME';
```

### Issue: SSL certificate validation fails

**Solution:**
```bash
# Option 1: Disable certificate validation (testing only)
hdbsql -n host:443 -u USER -p PASS -sslprovider commoncrypto -ssltrustcert

# Option 2: Add SAP BTP CA certificate to trust store
# Download from: https://help.sap.com/docs/connectivity/connectivity-cloud/cloud-connector-certificate
```

---

## Maintenance Tasks

### Regular Maintenance (Weekly):

```sql
-- 1. Check table sizes
SELECT TABLE_NAME, DISK_SIZE, RECORD_COUNT
FROM M_TABLES
WHERE SCHEMA_NAME = 'NUCLEUS'
ORDER BY DISK_SIZE DESC;

-- 2. Cleanup old audit logs (keeps last 90 days)
CALL NUCLEUS.SP_CLEANUP_AUDIT_LOGS();

-- 3. Expire old notifications
CALL NUCLEUS.SP_EXPIRE_OLD_NOTIFICATIONS();

-- 4. Check index fragmentation
SELECT TABLE_NAME, INDEX_NAME, FRAGMENTATION_PERCENTAGE
FROM M_INDEXES
WHERE SCHEMA_NAME = 'NUCLEUS' AND FRAGMENTATION_PERCENTAGE > 20;
```

### Monthly Tasks:

```sql
-- 1. Update statistics
UPDATE STATISTICS FOR TABLE NUCLEUS.PROMPT_HISTORY;
UPDATE STATISTICS FOR TABLE NUCLEUS.PROMPT_COMPARISONS;
UPDATE STATISTICS FOR TABLE NUCLEUS.TRAINING_EXPERIMENTS;

-- 2. Rebuild fragmented indexes (if fragmentation > 30%)
MERGE DELTA OF NUCLEUS.PROMPT_HISTORY;

-- 3. Backup verification
-- Check last backup in BTP Cockpit → HANA Cloud → Backups
```

---

## Migration from Local/Test to Production

If you're migrating from a local HANA instance to BTP Cloud:

1. **Export Schema:**
   ```bash
   # Export from local instance
   hdbsql -U LOCAL -o export.sql "EXPORT SCHEMA NUCLEUS"
   ```

2. **Modify Export File:**
   - Remove CREATE SCHEMA statement (already exists)
   - Check for incompatible features
   - Adjust resource limits if any

3. **Import to BTP Cloud:**
   ```bash
   hdbsql -U BTP_CLOUD -i import.sql
   ```

4. **Verify Data:**
   ```sql
   SELECT COUNT(*) FROM NUCLEUS.PROMPT_HISTORY;
   SELECT COUNT(*) FROM NUCLEUS.MODEL_PERFORMANCE;
   ```

---

## Support and Resources

- **SAP HANA Cloud Documentation:** https://help.sap.com/docs/hana-cloud
- **BTP Cockpit:** https://cockpit.hanatrial.ondemand.com/
- **SAP Community:** https://community.sap.com/topics/hana-cloud
- **SQL Reference:** https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide

---

## Appendix: Quick Command Reference

```sql
-- List all schemas
SELECT * FROM SCHEMAS;

-- List all users
SELECT * FROM USERS;

-- Check current user
SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY;

-- Grant additional privilege
GRANT CREATE TABLE TO NUCLEUS_APP;

-- Revoke privilege
REVOKE CREATE TABLE FROM NUCLEUS_APP;

-- Drop schema (CAUTION!)
DROP SCHEMA NUCLEUS CASCADE;

-- Drop user
DROP USER NUCLEUS_APP CASCADE;
```

---

**Document Version:** 1.0  
**Last Updated:** January 21, 2026  
**Author:** Nucleus OpenAI Server Team
