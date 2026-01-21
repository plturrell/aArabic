-- ============================================================================
-- nWorkflow Initial Database Schema Migration
-- Version: 001
-- Description: Creates the foundational schema for nWorkflow multi-tenant 
--              workflow execution platform
-- ============================================================================

-- ============================================================================
-- SECTION 1: EXTENSIONS
-- Enable required PostgreSQL extensions for UUID generation and encryption
-- ============================================================================

-- uuid-ossp: Provides functions to generate universally unique identifiers (UUIDs)
-- Used for generating primary keys across all tables
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- pgcrypto: Provides cryptographic functions for encryption/decryption
-- Used for encrypting sensitive data like API keys and credentials
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- SECTION 2: TABLES
-- Core tables for multi-tenant workflow management
-- ============================================================================

-- -----------------------------------------------------------------------------
-- Tenants Table
-- Stores organization/tenant information for multi-tenancy support
-- Each tenant has isolated data through Row-Level Security policies
-- -----------------------------------------------------------------------------
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,                    -- Display name of the tenant
    slug VARCHAR(100) UNIQUE NOT NULL,             -- URL-friendly unique identifier
    settings JSONB DEFAULT '{}',                   -- Tenant-specific configuration
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE tenants IS 'Multi-tenant organizations using the workflow platform';
COMMENT ON COLUMN tenants.slug IS 'URL-friendly unique identifier for the tenant';
COMMENT ON COLUMN tenants.settings IS 'JSON configuration for tenant-specific settings';

-- -----------------------------------------------------------------------------
-- Users Table
-- Stores user information synced from Keycloak identity provider
-- Users are scoped to tenants for proper isolation
-- -----------------------------------------------------------------------------
CREATE TABLE users (
    id UUID PRIMARY KEY,                           -- UUID from Keycloak (not auto-generated)
    tenant_id UUID REFERENCES tenants(id),         -- Tenant association
    username VARCHAR(255) NOT NULL,                -- Unique username within tenant
    email VARCHAR(255),                            -- User's email address
    roles TEXT[] DEFAULT '{}',                     -- Array of role names from Keycloak
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ                         -- Tracks last login timestamp
);

COMMENT ON TABLE users IS 'User accounts synced from Keycloak identity provider';
COMMENT ON COLUMN users.id IS 'UUID from Keycloak - not auto-generated';
COMMENT ON COLUMN users.roles IS 'Array of role names assigned to the user';

-- -----------------------------------------------------------------------------
-- Workflows Table
-- Stores workflow definitions with versioning support
-- Workflows contain the complete definition in JSONB format
-- -----------------------------------------------------------------------------
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) NOT NULL,
    name VARCHAR(255) NOT NULL,                    -- Human-readable workflow name
    description TEXT,                              -- Detailed workflow description
    definition JSONB NOT NULL,                     -- Complete workflow definition (nodes, edges, config)
    version INTEGER DEFAULT 1,                     -- Version number for tracking changes
    is_active BOOLEAN DEFAULT true,                -- Whether workflow can be executed
    created_by UUID REFERENCES users(id),          -- User who created the workflow
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE workflows IS 'Workflow definitions with versioning support';
COMMENT ON COLUMN workflows.definition IS 'JSONB containing nodes, edges, and configuration';
COMMENT ON COLUMN workflows.version IS 'Incremented on each workflow update';

-- -----------------------------------------------------------------------------
-- Executions Table
-- Tracks workflow execution instances with status and results
-- Each execution is associated with a workflow and tenant
-- -----------------------------------------------------------------------------
CREATE TABLE executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID REFERENCES workflows(id) NOT NULL,
    tenant_id UUID REFERENCES tenants(id) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',          -- pending, running, completed, failed, cancelled
    input JSONB,                                   -- Input data provided to the workflow
    output JSONB,                                  -- Output data from successful execution
    error_message TEXT,                            -- Error details if execution failed
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,                      -- Set when execution finishes
    executed_by UUID REFERENCES users(id)          -- User who triggered the execution
);

COMMENT ON TABLE executions IS 'Workflow execution instances with status tracking';
COMMENT ON COLUMN executions.status IS 'Execution state: pending, running, completed, failed, cancelled';

-- -----------------------------------------------------------------------------
-- Execution Logs Table
-- Detailed step-by-step logs for each node in an execution
-- Enables debugging and performance analysis of workflow runs
-- -----------------------------------------------------------------------------
CREATE TABLE execution_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID REFERENCES executions(id) NOT NULL,
    node_id VARCHAR(255) NOT NULL,                 -- Identifier of the executed node
    node_type VARCHAR(100),                        -- Type of node (e.g., 'llm', 'http', 'transform')
    status VARCHAR(50),                            -- Node execution status
    input JSONB,                                   -- Input data to the node
    output JSONB,                                  -- Output data from the node
    error_message TEXT,                            -- Error details if node failed
    duration_ms INTEGER,                           -- Execution time in milliseconds
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE execution_logs IS 'Detailed logs for each node execution in a workflow run';
COMMENT ON COLUMN execution_logs.duration_ms IS 'Node execution time for performance analysis';

-- -----------------------------------------------------------------------------
-- Audit Logs Table
-- Comprehensive audit trail for security and compliance
-- Tracks all significant actions across the platform
-- -----------------------------------------------------------------------------
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id),         -- Tenant context (nullable for system events)
    user_id UUID,                                  -- User who performed the action (nullable for system)
    event_type VARCHAR(100) NOT NULL,              -- Category: auth, workflow, execution, admin
    resource_type VARCHAR(100),                    -- Type of resource affected
    resource_id UUID,                              -- ID of the affected resource
    action VARCHAR(100) NOT NULL,                  -- Specific action: create, update, delete, execute
    details JSONB,                                 -- Additional context and metadata
    ip_address INET,                               -- Client IP address
    user_agent TEXT,                               -- Client user agent string
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE audit_logs IS 'Security and compliance audit trail';
COMMENT ON COLUMN audit_logs.event_type IS 'Event category: auth, workflow, execution, admin';
COMMENT ON COLUMN audit_logs.action IS 'Specific action performed on the resource';

-- ============================================================================
-- SECTION 3: INDEXES
-- Performance optimization indexes for common query patterns
-- ============================================================================

-- Users indexes
CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- Workflows indexes
CREATE INDEX idx_workflows_tenant_id ON workflows(tenant_id);
CREATE INDEX idx_workflows_created_by ON workflows(created_by);
CREATE INDEX idx_workflows_is_active ON workflows(is_active);
CREATE INDEX idx_workflows_name ON workflows(name);

-- Executions indexes - optimized for status queries and filtering
CREATE INDEX idx_executions_tenant_id ON executions(tenant_id);
CREATE INDEX idx_executions_workflow_id ON executions(workflow_id);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_executions_workflow_status ON executions(workflow_id, status);
CREATE INDEX idx_executions_started_at ON executions(started_at DESC);
CREATE INDEX idx_executions_executed_by ON executions(executed_by);

-- Execution logs indexes - for debugging and analysis
CREATE INDEX idx_execution_logs_execution_id ON execution_logs(execution_id);
CREATE INDEX idx_execution_logs_node_id ON execution_logs(node_id);
CREATE INDEX idx_execution_logs_status ON execution_logs(status);
CREATE INDEX idx_execution_logs_created_at ON execution_logs(created_at DESC);

-- Audit logs indexes - for security analysis and reporting
CREATE INDEX idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_event_type_created ON audit_logs(event_type, created_at DESC);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- ============================================================================
-- SECTION 4: ROW-LEVEL SECURITY (RLS)
-- Enables automatic tenant isolation at the database level
-- Uses session variable app.current_tenant_id to filter data
-- ============================================================================

-- Enable RLS on tenant-scoped tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE execution_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- -----------------------------------------------------------------------------
-- RLS Policies for Users Table
-- Users can only see other users in their tenant
-- -----------------------------------------------------------------------------
CREATE POLICY users_tenant_isolation ON users
    FOR ALL
    USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::UUID);

-- -----------------------------------------------------------------------------
-- RLS Policies for Workflows Table
-- Workflows are isolated by tenant
-- -----------------------------------------------------------------------------
CREATE POLICY workflows_tenant_isolation ON workflows
    FOR ALL
    USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::UUID);

-- -----------------------------------------------------------------------------
-- RLS Policies for Executions Table
-- Executions are isolated by tenant
-- -----------------------------------------------------------------------------
CREATE POLICY executions_tenant_isolation ON executions
    FOR ALL
    USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::UUID);

-- -----------------------------------------------------------------------------
-- RLS Policies for Execution Logs Table
-- Logs are accessible through their parent execution's tenant
-- -----------------------------------------------------------------------------
CREATE POLICY execution_logs_tenant_isolation ON execution_logs
    FOR ALL
    USING (
        execution_id IN (
            SELECT id FROM executions
            WHERE tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::UUID
        )
    );

-- -----------------------------------------------------------------------------
-- RLS Policies for Audit Logs Table
-- Audit logs are isolated by tenant (null tenant_id for system-wide events)
-- -----------------------------------------------------------------------------
CREATE POLICY audit_logs_tenant_isolation ON audit_logs
    FOR ALL
    USING (
        tenant_id IS NULL OR
        tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::UUID
    );

-- ============================================================================
-- SECTION 5: FUNCTIONS AND TRIGGERS
-- Utility functions for automated timestamp management
-- ============================================================================

-- -----------------------------------------------------------------------------
-- Function: update_updated_at()
-- Automatically updates the updated_at timestamp when a row is modified
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_updated_at() IS 'Trigger function to auto-update updated_at timestamp';

-- -----------------------------------------------------------------------------
-- Apply update_updated_at trigger to tables with updated_at column
-- -----------------------------------------------------------------------------

-- Trigger for tenants table
CREATE TRIGGER trigger_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Trigger for workflows table
CREATE TRIGGER trigger_workflows_updated_at
    BEFORE UPDATE ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- SECTION 6: SEED DATA (Optional)
-- Default tenant for development/testing
-- ============================================================================

-- Insert a default tenant for development purposes
-- Comment out or remove in production deployments
INSERT INTO tenants (name, slug, settings)
VALUES ('Default Organization', 'default', '{"features": {"advanced_analytics": true}}')
ON CONFLICT (slug) DO NOTHING;

-- ============================================================================
-- Migration Complete
--
-- To use RLS, applications must set the tenant context before queries:
--   SET app.current_tenant_id = 'tenant-uuid-here';
--
-- Or use a function to set it:
--   SELECT set_config('app.current_tenant_id', 'tenant-uuid-here', false);
-- ============================================================================

