#!/bin/bash
set -e

# nLocalModels - HANA Cloud Setup Script
# This script creates the required database tables in HANA Cloud

echo "=========================================="
echo "nLocalModels - HANA Cloud Setup"
echo "=========================================="
echo ""

# Check if hdbsql is available
if ! command -v hdbsql &> /dev/null; then
    echo "❌ Error: hdbsql not found!"
    echo ""
    echo "Please install HANA Client:"
    echo "  macOS: Download from https://tools.hana.ondemand.com/#hanatools"
    echo "  Linux: sudo apt-get install sap-hana-client"
    echo ""
    exit 1
fi

# Get credentials
echo "Please provide HANA Cloud credentials:"
echo ""
read -p "HANA Host (e.g., abc123.hanacloud.ondemand.com): " HANA_HOST
read -p "HANA Port [443]: " HANA_PORT
HANA_PORT=${HANA_PORT:-443}
read -p "Database Name [NOPENAI_DB]: " HANA_DB
HANA_DB=${HANA_DB:-NOPENAI_DB}
read -p "Username [NUCLEUS_APP]: " HANA_USER
HANA_USER=${HANA_USER:-NUCLEUS_APP}
read -sp "Password: " HANA_PASSWORD
echo ""
echo ""

# Test connection
echo "Testing connection to HANA Cloud..."
if hdbsql -n "${HANA_HOST}:${HANA_PORT}" -d "${HANA_DB}" -u "${HANA_USER}" -p "${HANA_PASSWORD}" -C "SELECT 1 FROM DUMMY" &> /dev/null; then
    echo "✅ Connection successful!"
else
    echo "❌ Connection failed! Please check your credentials."
    exit 1
fi

echo ""
echo "Creating HANA tables..."
echo ""

# Create tables
hdbsql -n "${HANA_HOST}:${HANA_PORT}" -d "${HANA_DB}" -u "${HANA_USER}" -p "${HANA_PASSWORD}" << 'EOF'

-- 1. KV Cache Metadata
CREATE COLUMN TABLE KV_CACHE_METADATA (
  SESSION_ID VARCHAR(128),
  LAYER INT,
  SIZE INT,
  COMPRESSION VARCHAR(16),
  OBJECT_KEY VARCHAR(256),
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  EXPIRES_AT TIMESTAMP,
  PRIMARY KEY (SESSION_ID, LAYER)
);

-- 2. Prompt Cache
CREATE COLUMN TABLE PROMPT_CACHE (
  HASH VARCHAR(64) PRIMARY KEY,
  STATE BLOB,
  EXPIRES_AT TIMESTAMP,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Session State
CREATE COLUMN TABLE SESSION_STATE (
  SESSION_ID VARCHAR(128) PRIMARY KEY,
  DATA BLOB,
  EXPIRES_AT TIMESTAMP,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Routing Decisions
CREATE COLUMN TABLE ROUTING_DECISIONS (
  DECISION_ID VARCHAR(128) PRIMARY KEY,
  REQUEST_ID VARCHAR(128),
  TASK_TYPE VARCHAR(64),
  AGENT_ID VARCHAR(128),
  MODEL_ID VARCHAR(128),
  CAPABILITY_SCORE DECIMAL(5,2),
  PERFORMANCE_SCORE DECIMAL(5,2),
  COMPOSITE_SCORE DECIMAL(5,2),
  STRATEGY_USED VARCHAR(64),
  LATENCY_MS INT,
  SUCCESS BOOLEAN,
  FALLBACK_USED BOOLEAN,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Inference Metrics
CREATE COLUMN TABLE INFERENCE_METRICS (
  METRIC_ID VARCHAR(128) PRIMARY KEY,
  MODEL_ID VARCHAR(128),
  LATENCY_MS INT,
  TTFT_MS INT,
  TOKENS_GENERATED INT,
  CACHE_HIT BOOLEAN,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_routing_agent ON ROUTING_DECISIONS(AGENT_ID);
CREATE INDEX idx_routing_model ON ROUTING_DECISIONS(MODEL_ID);
CREATE INDEX idx_routing_created ON ROUTING_DECISIONS(CREATED_AT);
CREATE INDEX idx_metrics_model ON INFERENCE_METRICS(MODEL_ID);
CREATE INDEX idx_metrics_created ON INFERENCE_METRICS(CREATED_AT);

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tables created successfully!"
    echo ""
    
    # Verify tables
    echo "Verifying tables..."
    hdbsql -n "${HANA_HOST}:${HANA_PORT}" -d "${HANA_DB}" -u "${HANA_USER}" -p "${HANA_PASSWORD}" \
        -C "SELECT TABLE_NAME FROM TABLES WHERE SCHEMA_NAME='${HANA_USER}' AND TABLE_NAME IN ('KV_CACHE_METADATA','PROMPT_CACHE','SESSION_STATE','ROUTING_DECISIONS','INFERENCE_METRICS')"
    
    echo ""
    echo "✅ HANA setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Update nlocalmodels-secret.yaml with these credentials"
    echo "2. Deploy to Kyma: kubectl apply -k ."
    echo ""
else
    echo ""
    echo "❌ Error creating tables. Please check the error messages above."
    exit 1
fi
