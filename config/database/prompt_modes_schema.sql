-- ============================================================================
-- HANA Database Schema for Prompt Modes System
-- ============================================================================
-- Created: 2026-01-20
-- Purpose: Store prompt mode configurations, user history, and performance metrics
-- ============================================================================

-- Drop existing tables (for clean reinstall)
DROP TABLE IF EXISTS PROMPT_HISTORY;
DROP TABLE IF EXISTS MODEL_PERFORMANCE;
DROP TABLE IF EXISTS PROMPT_MODE_CONFIGS;

-- ============================================================================
-- Table: PROMPT_MODE_CONFIGS
-- Purpose: Store predefined and custom prompt mode configurations
-- ============================================================================
CREATE COLUMN TABLE PROMPT_MODE_CONFIGS (
    CONFIG_ID VARCHAR(36) PRIMARY KEY,
    MODE_NAME VARCHAR(50) NOT NULL,           -- Fast/Normal/Expert/Research
    MODEL_ID VARCHAR(100) NOT NULL,
    DISPLAY_NAME VARCHAR(200),
    DESCRIPTION VARCHAR(500),
    
    -- Resource Allocation (percentages)
    GPU_PERCENT INTEGER CHECK (GPU_PERCENT >= 0 AND GPU_PERCENT <= 100),
    RAM_PERCENT INTEGER CHECK (RAM_PERCENT >= 0 AND RAM_PERCENT <= 100),
    SSD_PERCENT INTEGER CHECK (SSD_PERCENT >= 0 AND SSD_PERCENT <= 100),
    
    -- Memory Limits (MB)
    GPU_MEMORY_MB INTEGER,
    RAM_MEMORY_MB INTEGER,
    SSD_MEMORY_MB INTEGER,
    KV_CACHE_RAM_MB INTEGER,
    
    -- Model Configuration
    QUANTIZATION VARCHAR(20),
    ARCHITECTURE VARCHAR(50),
    FORMAT VARCHAR(20),
    
    -- Tier Configuration (JSON)
    TIER_CONFIG NCLOB,                        -- Stores full tier config as JSON
    
    -- Performance Targets
    TARGET_LATENCY_MS INTEGER,
    TARGET_TOKENS_PER_SEC INTEGER,
    MAX_CONCURRENT_REQUESTS INTEGER,
    
    -- Metadata
    IS_PRESET BOOLEAN DEFAULT FALSE,          -- TRUE for Fast/Normal/Expert/Research
    IS_ACTIVE BOOLEAN DEFAULT TRUE,
    CREATED_BY VARCHAR(100),
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT CHK_PERCENT_SUM CHECK (GPU_PERCENT + RAM_PERCENT + SSD_PERCENT <= 100)
);

-- Indexes for performance
CREATE INDEX IDX_MODE_NAME ON PROMPT_MODE_CONFIGS(MODE_NAME);
CREATE INDEX IDX_MODEL_ID ON PROMPT_MODE_CONFIGS(MODEL_ID);
CREATE INDEX IDX_IS_PRESET ON PROMPT_MODE_CONFIGS(IS_PRESET);

-- ============================================================================
-- Table: PROMPT_HISTORY
-- Purpose: Store all prompt/response pairs with performance metrics
-- ============================================================================
CREATE COLUMN TABLE PROMPT_HISTORY (
    PROMPT_ID VARCHAR(36) PRIMARY KEY,
    SESSION_ID VARCHAR(36),
    USER_ID VARCHAR(100),
    
    -- Configuration Reference
    CONFIG_ID VARCHAR(36),
    MODE_NAME VARCHAR(50) NOT NULL,
    MODEL_ID VARCHAR(100) NOT NULL,
    
    -- Prompt/Response
    PROMPT_TEXT NCLOB,
    RESPONSE_TEXT NCLOB,
    SYSTEM_PROMPT NCLOB,
    
    -- Performance Metrics
    LATENCY_MS INTEGER,
    TTFT_MS INTEGER,                          -- Time To First Token
    TOKENS_GENERATED INTEGER,
    TOKENS_PER_SECOND DECIMAL(10,2),
    PROMPT_TOKENS INTEGER,
    
    -- Tier Statistics (JSON)
    TIER_STATS NCLOB,                         -- Hit rates, cache performance
    
    -- Resource Usage
    GPU_MEMORY_USED_MB INTEGER,
    RAM_MEMORY_USED_MB INTEGER,
    SSD_MEMORY_USED_MB INTEGER,
    
    -- Quality Metrics (optional user feedback)
    USER_RATING INTEGER CHECK (USER_RATING BETWEEN 1 AND 5),
    USER_FEEDBACK VARCHAR(1000),
    
    -- Metadata
    TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    IP_ADDRESS VARCHAR(50),
    CLIENT_INFO VARCHAR(200),
    
    -- Error Handling
    HAS_ERROR BOOLEAN DEFAULT FALSE,
    ERROR_MESSAGE VARCHAR(1000),
    
    -- Foreign Key
    CONSTRAINT FK_CONFIG FOREIGN KEY (CONFIG_ID) 
        REFERENCES PROMPT_MODE_CONFIGS(CONFIG_ID) ON DELETE SET NULL
);

-- Indexes for queries
CREATE INDEX IDX_PROMPT_USER ON PROMPT_HISTORY(USER_ID);
CREATE INDEX IDX_PROMPT_MODE ON PROMPT_HISTORY(MODE_NAME);
CREATE INDEX IDX_PROMPT_MODEL ON PROMPT_HISTORY(MODEL_ID);
CREATE INDEX IDX_PROMPT_TIMESTAMP ON PROMPT_HISTORY(TIMESTAMP);
CREATE INDEX IDX_PROMPT_SESSION ON PROMPT_HISTORY(SESSION_ID);

-- ============================================================================
-- Table: MODEL_PERFORMANCE
-- Purpose: Aggregate performance metrics per model per mode
-- ============================================================================
CREATE COLUMN TABLE MODEL_PERFORMANCE (
    METRIC_ID VARCHAR(36) PRIMARY KEY,
    MODEL_ID VARCHAR(100) NOT NULL,
    MODE_NAME VARCHAR(50),
    
    -- Aggregated Metrics
    AVG_LATENCY_MS DECIMAL(10,2),
    P50_LATENCY_MS DECIMAL(10,2),
    P95_LATENCY_MS DECIMAL(10,2),
    P99_LATENCY_MS DECIMAL(10,2),
    
    AVG_TOKENS_PER_SEC DECIMAL(10,2),
    MAX_TOKENS_PER_SEC DECIMAL(10,2),
    MIN_TOKENS_PER_SEC DECIMAL(10,2),
    
    -- Cache Statistics
    CACHE_HIT_RATE DECIMAL(5,2),
    AVG_GPU_TIER_HIT_RATE DECIMAL(5,2),
    AVG_RAM_TIER_HIT_RATE DECIMAL(5,2),
    AVG_SSD_TIER_HIT_RATE DECIMAL(5,2),
    
    -- Volume Statistics
    TOTAL_REQUESTS INTEGER DEFAULT 0,
    TOTAL_TOKENS_GENERATED BIGINT DEFAULT 0,
    TOTAL_ERRORS INTEGER DEFAULT 0,
    
    -- Quality Metrics
    AVG_USER_RATING DECIMAL(3,2),
    TOTAL_RATINGS INTEGER DEFAULT 0,
    
    -- Time Window
    MEASUREMENT_START TIMESTAMP,
    MEASUREMENT_END TIMESTAMP,
    LAST_UPDATED TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT UQ_MODEL_MODE UNIQUE (MODEL_ID, MODE_NAME, MEASUREMENT_START)
);

-- Indexes
CREATE INDEX IDX_PERF_MODEL ON MODEL_PERFORMANCE(MODEL_ID);
CREATE INDEX IDX_PERF_MODE ON MODEL_PERFORMANCE(MODE_NAME);
CREATE INDEX IDX_PERF_UPDATED ON MODEL_PERFORMANCE(LAST_UPDATED);

-- ============================================================================
-- Table: MODE_PRESETS
-- Purpose: Define the 4 standard mode presets with their characteristics
-- ============================================================================
CREATE COLUMN TABLE MODE_PRESETS (
    MODE_NAME VARCHAR(50) PRIMARY KEY,
    DISPLAY_NAME VARCHAR(100),
    DESCRIPTION VARCHAR(500),
    ICON VARCHAR(50),
    COLOR VARCHAR(20),
    
    -- Default Resource Allocation
    DEFAULT_GPU_PERCENT INTEGER,
    DEFAULT_RAM_PERCENT INTEGER,
    DEFAULT_SSD_PERCENT INTEGER,
    
    -- Compatible Models (JSON array)
    COMPATIBLE_MODELS NCLOB,
    RECOMMENDED_MODELS NCLOB,
    EXCLUDED_MODELS NCLOB,
    
    -- Performance Characteristics
    EXPECTED_LATENCY_MS VARCHAR(50),          -- e.g., "50-100"
    EXPECTED_TPS VARCHAR(50),                 -- e.g., "40-60"
    USE_CASES NCLOB,                          -- JSON array
    
    -- Priority Settings
    PRIORITY_ORDER INTEGER,
    IS_ACTIVE BOOLEAN DEFAULT TRUE,
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Insert Default Mode Presets
-- ============================================================================
INSERT INTO MODE_PRESETS VALUES (
    'Fast',
    'Fast Mode',
    'Optimized for lowest latency and quick responses. Uses highly compressed models with minimal resource overhead.',
    'sap-icon://accelerated',
    '#00A600',
    65, 25, 10,
    '["lfm2.5-1.2b-q4_0", "hymt-1.5-7b-q4_k_m"]',
    '["lfm2.5-1.2b-q4_0"]',
    '["llama-3.3-70b", "deepseek-coder-33b"]',
    '50-150',
    '40-80',
    '["development", "testing", "real-time-chat", "low-latency"]',
    1,
    TRUE,
    CURRENT_TIMESTAMP
);

INSERT INTO MODE_PRESETS VALUES (
    'Normal',
    'Normal Mode',
    'Balanced performance and quality for production workloads. Good compromise between speed and accuracy.',
    'sap-icon://navigation-right-arrow',
    '#0070F2',
    45, 35, 20,
    '["lfm2.5-1.2b-q4_k_m", "hymt-1.5-7b-q6_k", "deepseek-coder-33b"]',
    '["lfm2.5-1.2b-q4_k_m", "hymt-1.5-7b-q6_k"]',
    '["llama-3.3-70b"]',
    '100-300',
    '25-50',
    '["production", "general-purpose", "balanced-performance"]',
    2,
    TRUE,
    CURRENT_TIMESTAMP
);

INSERT INTO MODE_PRESETS VALUES (
    'Expert',
    'Expert Mode',
    'High quality responses with optimized tiering. Uses higher precision quantization for better accuracy.',
    'sap-icon://complete',
    '#FF9500',
    35, 45, 20,
    '["lfm2.5-1.2b-f16", "hymt-1.5-7b-q8_0", "deepseek-coder-33b"]',
    '["lfm2.5-1.2b-f16", "deepseek-coder-33b"]',
    '[]',
    '200-500',
    '15-35',
    '["code-generation", "complex-reasoning", "quality-focused"]',
    3,
    TRUE,
    CURRENT_TIMESTAMP
);

INSERT INTO MODE_PRESETS VALUES (
    'Research',
    'Research Mode',
    'Maximum quality with full tiering support. Suitable for large models and benchmarking work.',
    'sap-icon://lab',
    '#DC143C',
    25, 35, 40,
    '["llama-3.3-70b", "deepseek-coder-33b", "lfm2.5-1.2b-f16", "microsoft-phi-2"]',
    '["llama-3.3-70b"]',
    '[]',
    '300-1000',
    '10-25',
    '["research", "benchmarking", "quality-validation", "large-models"]',
    4,
    TRUE,
    CURRENT_TIMESTAMP
);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View: Recent prompts per mode
CREATE VIEW V_RECENT_PROMPTS_BY_MODE AS
SELECT 
    MODE_NAME,
    MODEL_ID,
    COUNT(*) as TOTAL_PROMPTS,
    AVG(LATENCY_MS) as AVG_LATENCY,
    AVG(TOKENS_PER_SECOND) as AVG_TPS,
    MAX(TIMESTAMP) as LAST_USED
FROM PROMPT_HISTORY
WHERE TIMESTAMP > ADD_DAYS(CURRENT_TIMESTAMP, -7)
GROUP BY MODE_NAME, MODEL_ID;

-- View: Top performing models per mode
CREATE VIEW V_TOP_MODELS_BY_MODE AS
SELECT 
    MODE_NAME,
    MODEL_ID,
    AVG_TOKENS_PER_SEC,
    P50_LATENCY_MS,
    CACHE_HIT_RATE,
    TOTAL_REQUESTS,
    AVG_USER_RATING
FROM MODEL_PERFORMANCE
WHERE TOTAL_REQUESTS > 10
ORDER BY MODE_NAME, AVG_TOKENS_PER_SEC DESC;

-- ============================================================================
-- Grant Permissions (adjust schema/user as needed)
-- ============================================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON PROMPT_MODE_CONFIGS TO OPENAI_SERVER_USER;
-- GRANT SELECT, INSERT ON PROMPT_HISTORY TO OPENAI_SERVER_USER;
-- GRANT SELECT, INSERT, UPDATE ON MODEL_PERFORMANCE TO OPENAI_SERVER_USER;
-- GRANT SELECT ON MODE_PRESETS TO OPENAI_SERVER_USER;

COMMIT;
