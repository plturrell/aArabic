-- ==============================================================================
-- HANA Cloud Vector Table for TrialBalance RAG
-- ==============================================================================
-- 
-- Creates a vector-enabled table to store document chunks with embeddings
-- for semantic search and RAG (Retrieval-Augmented Generation)
--
-- Embedding Dimension: 768 (nomic-embed-text) or 1536 (text-embedding-3-small)
-- ==============================================================================

-- Create schema if not exists (adjust to your schema)
-- CREATE SCHEMA IF NOT EXISTS TRIALBALANCE;

-- Drop existing table if needed (uncomment for fresh start)
-- DROP TABLE TRIALBALANCE_VECTORS;

-- Create vector table
CREATE COLUMN TABLE TRIALBALANCE_VECTORS (
    -- Primary key
    ID NVARCHAR(256) PRIMARY KEY,
    
    -- Document content
    CONTENT NCLOB,
    
    -- Metadata
    FILE_PATH NVARCHAR(500),
    FILE_TYPE NVARCHAR(50),
    MODULE NVARCHAR(100),
    CHUNK_INDEX INTEGER,
    TOTAL_CHUNKS INTEGER,
    
    -- Vector embedding (768 dimensions for nomic-embed-text)
    EMBEDDING REAL_VECTOR(768),
    
    -- Timestamps
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for metadata filtering
CREATE INDEX IDX_TB_VECTORS_MODULE ON TRIALBALANCE_VECTORS(MODULE);
CREATE INDEX IDX_TB_VECTORS_FILE_TYPE ON TRIALBALANCE_VECTORS(FILE_TYPE);
CREATE INDEX IDX_TB_VECTORS_FILE_PATH ON TRIALBALANCE_VECTORS(FILE_PATH);

-- Comments
COMMENT ON TABLE TRIALBALANCE_VECTORS IS 'Vector store for TrialBalance codebase RAG';
COMMENT ON COLUMN TRIALBALANCE_VECTORS.EMBEDDING IS 'nomic-embed-text 768-dim vectors';

-- ==============================================================================
-- Vector Search Function
-- ==============================================================================
-- 
-- Example query to find similar documents:
-- 
-- SELECT TOP 5 
--     ID, FILE_PATH, MODULE, 
--     LEFT(CONTENT, 200) as PREVIEW,
--     COSINE_SIMILARITY(EMBEDDING, TO_REAL_VECTOR(?)) as SIMILARITY
-- FROM TRIALBALANCE_VECTORS
-- WHERE MODULE = 'backend'  -- Optional filter
-- ORDER BY COSINE_SIMILARITY(EMBEDDING, TO_REAL_VECTOR(?)) DESC;
--
-- ==============================================================================

-- Create a stored procedure for semantic search
CREATE OR REPLACE PROCEDURE TRIALBALANCE_SEMANTIC_SEARCH(
    IN query_embedding REAL_VECTOR(768),
    IN top_k INTEGER DEFAULT 5,
    IN module_filter NVARCHAR(100) DEFAULT NULL
)
LANGUAGE SQLSCRIPT
READS SQL DATA
AS
BEGIN
    SELECT TOP :top_k
        ID,
        FILE_PATH,
        FILE_TYPE,
        MODULE,
        CHUNK_INDEX,
        LEFT(CONTENT, 500) as CONTENT_PREVIEW,
        COSINE_SIMILARITY(EMBEDDING, :query_embedding) as SIMILARITY
    FROM TRIALBALANCE_VECTORS
    WHERE (:module_filter IS NULL OR MODULE = :module_filter)
    ORDER BY COSINE_SIMILARITY(EMBEDDING, :query_embedding) DESC;
END;

-- ==============================================================================
-- Sample Data Check
-- ==============================================================================
-- SELECT COUNT(*) as TOTAL_VECTORS, 
--        COUNT(DISTINCT FILE_PATH) as UNIQUE_FILES,
--        COUNT(DISTINCT MODULE) as MODULES
-- FROM TRIALBALANCE_VECTORS;

-- SELECT MODULE, COUNT(*) as COUNT 
-- FROM TRIALBALANCE_VECTORS 
-- GROUP BY MODULE 
-- ORDER BY COUNT DESC;