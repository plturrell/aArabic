-- SCIP (Source Code Intelligence Protocol) Database Schema for SAP HANA
-- 
-- This schema supports distributed code intelligence across services in the
-- arabic_folder project (nGrounding, nCode, nAgentMeta)
--
-- Created: January 24, 2026
-- Version: 1.0.0

-- =============================================================================
-- SCIP Indexes Table
-- =============================================================================
-- Stores complete SCIP index files for projects
CREATE COLUMN TABLE SCIP_INDEXES (
  -- Primary identifier
  project_id VARCHAR(128) PRIMARY KEY,
  
  -- Project metadata
  project_root VARCHAR(512) NOT NULL,
  project_name VARCHAR(256),
  
  -- Index data (protobuf-encoded SCIP index)
  index_data BLOB,
  
  -- Metadata (JSON format)
  metadata JSON,
  
  -- Statistics
  document_count INTEGER DEFAULT 0,
  symbol_count INTEGER DEFAULT 0,
  occurrence_count INTEGER DEFAULT 0,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  indexed_at TIMESTAMP,
  
  -- Tool information
  tool_name VARCHAR(128),
  tool_version VARCHAR(32)
);

COMMENT ON COLUMN SCIP_INDEXES.index_data IS 'Protobuf-encoded SCIP index';
COMMENT ON COLUMN SCIP_INDEXES.metadata IS 'JSON metadata: {language, encoding, version}';

-- =============================================================================
-- SCIP Symbols Table
-- =============================================================================
-- Denormalized symbol table for fast lookups
CREATE COLUMN TABLE SCIP_SYMBOLS (
  -- Primary identifier (SCIP symbol format)
  symbol_id VARCHAR(512) PRIMARY KEY,
  
  -- Foreign key to project
  project_id VARCHAR(128) NOT NULL,
  
  -- Symbol information
  name VARCHAR(256) NOT NULL,
  kind VARCHAR(32) NOT NULL,  -- function, class, struct, method, etc.
  scheme VARCHAR(32),          -- zig, python, typescript, etc.
  package VARCHAR(256),
  
  -- Location
  file_path VARCHAR(512),
  line INTEGER,
  column INTEGER,
  
  -- Documentation
  documentation CLOB,
  
  -- Metadata
  is_definition BOOLEAN DEFAULT FALSE,
  is_exported BOOLEAN DEFAULT FALSE,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- Foreign key
  FOREIGN KEY (project_id) REFERENCES SCIP_INDEXES(project_id) ON DELETE CASCADE
);

COMMENT ON COLUMN SCIP_SYMBOLS.symbol_id IS 'SCIP format: scip-{scheme}+{package}+{descriptors}';
COMMENT ON COLUMN SCIP_SYMBOLS.kind IS 'SymbolKind: function, method, struct, class, etc.';

-- =============================================================================
-- SCIP Occurrences Table
-- =============================================================================
-- Stores all symbol occurrences (usages) for find-references
CREATE COLUMN TABLE SCIP_OCCURRENCES (
  -- Primary key
  occurrence_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  
  -- Foreign keys
  project_id VARCHAR(128) NOT NULL,
  symbol_id VARCHAR(512) NOT NULL,
  
  -- Location
  file_path VARCHAR(512) NOT NULL,
  start_line INTEGER NOT NULL,
  start_column INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  end_column INTEGER NOT NULL,
  
  -- Role
  is_definition BOOLEAN DEFAULT FALSE,
  is_reference BOOLEAN DEFAULT TRUE,
  is_write BOOLEAN DEFAULT FALSE,
  is_read BOOLEAN DEFAULT TRUE,
  
  -- Syntax highlighting
  syntax_kind VARCHAR(32),
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- Foreign keys
  FOREIGN KEY (project_id) REFERENCES SCIP_INDEXES(project_id) ON DELETE CASCADE,
  FOREIGN KEY (symbol_id) REFERENCES SCIP_SYMBOLS(symbol_id) ON DELETE CASCADE
);

COMMENT ON COLUMN SCIP_OCCURRENCES.syntax_kind IS 'SyntaxKind for highlighting: keyword, identifier, etc.';

-- =============================================================================
-- SCIP Relationships Table
-- =============================================================================
-- Stores relationships between symbols (inheritance, implementation, etc.)
CREATE COLUMN TABLE SCIP_RELATIONSHIPS (
  -- Primary key
  relationship_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  
  -- Source and target symbols
  source_symbol_id VARCHAR(512) NOT NULL,
  target_symbol_id VARCHAR(512) NOT NULL,
  
  -- Relationship type
  relationship_type VARCHAR(32) NOT NULL,  -- implements, extends, references, etc.
  
  -- Project context
  project_id VARCHAR(128) NOT NULL,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- Foreign keys
  FOREIGN KEY (source_symbol_id) REFERENCES SCIP_SYMBOLS(symbol_id) ON DELETE CASCADE,
  FOREIGN KEY (target_symbol_id) REFERENCES SCIP_SYMBOLS(symbol_id) ON DELETE CASCADE,
  FOREIGN KEY (project_id) REFERENCES SCIP_INDEXES(project_id) ON DELETE CASCADE
);

COMMENT ON COLUMN SCIP_RELATIONSHIPS.relationship_type IS 'definition, reference, implementation, type_definition';

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Symbol lookups by name
CREATE INDEX idx_scip_symbols_name ON SCIP_SYMBOLS(name);
CREATE INDEX idx_scip_symbols_project ON SCIP_SYMBOLS(project_id);
CREATE INDEX idx_scip_symbols_file ON SCIP_SYMBOLS(file_path);
CREATE INDEX idx_scip_symbols_kind ON SCIP_SYMBOLS(kind);
CREATE INDEX idx_scip_symbols_package ON SCIP_SYMBOLS(package);

-- Occurrence lookups
CREATE INDEX idx_scip_occurrences_symbol ON SCIP_OCCURRENCES(symbol_id);
CREATE INDEX idx_scip_occurrences_project ON SCIP_OCCURRENCES(project_id);
CREATE INDEX idx_scip_occurrences_file ON SCIP_OCCURRENCES(file_path);
CREATE INDEX idx_scip_occurrences_location ON SCIP_OCCURRENCES(file_path, start_line, start_column);

-- Relationship lookups
CREATE INDEX idx_scip_relationships_source ON SCIP_RELATIONSHIPS(source_symbol_id);
CREATE INDEX idx_scip_relationships_target ON SCIP_RELATIONSHIPS(target_symbol_id);
CREATE INDEX idx_scip_relationships_project ON SCIP_RELATIONSHIPS(project_id);
CREATE INDEX idx_scip_relationships_type ON SCIP_RELATIONSHIPS(relationship_type);

-- Project lookups
CREATE INDEX idx_scip_indexes_name ON SCIP_INDEXES(project_name);
CREATE INDEX idx_scip_indexes_updated ON SCIP_INDEXES(updated_at);

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- View: Symbol definitions with full context
CREATE VIEW v_scip_definitions AS
SELECT 
    s.symbol_id,
    s.name,
    s.kind,
    s.file_path,
    s.line,
    s.column,
    s.documentation,
    i.project_name,
    i.project_root
FROM SCIP_SYMBOLS s
JOIN SCIP_INDEXES i ON s.project_id = i.project_id
WHERE s.is_definition = TRUE;

-- View: Symbol references with context
CREATE VIEW v_scip_references AS
SELECT 
    o.symbol_id,
    s.name as symbol_name,
    s.kind as symbol_kind,
    o.file_path,
    o.start_line,
    o.start_column,
    o.end_line,
    o.end_column,
    i.project_name
FROM SCIP_OCCURRENCES o
JOIN SCIP_SYMBOLS s ON o.symbol_id = s.symbol_id
JOIN SCIP_INDEXES i ON o.project_id = i.project_id
WHERE o.is_reference = TRUE;

-- View: Project statistics
CREATE VIEW v_scip_project_stats AS
SELECT 
    i.project_id,
    i.project_name,
    i.project_root,
    i.document_count,
    i.symbol_count,
    i.occurrence_count,
    COUNT(DISTINCT s.symbol_id) as actual_symbol_count,
    COUNT(DISTINCT o.occurrence_id) as actual_occurrence_count,
    i.updated_at,
    i.tool_name,
    i.tool_version
FROM SCIP_INDEXES i
LEFT JOIN SCIP_SYMBOLS s ON i.project_id = s.project_id
LEFT JOIN SCIP_OCCURRENCES o ON i.project_id = o.project_id
GROUP BY 
    i.project_id, i.project_name, i.project_root, 
    i.document_count, i.symbol_count, i.occurrence_count,
    i.updated_at, i.tool_name, i.tool_version;

-- =============================================================================
-- Stored Procedures
-- =============================================================================

-- Procedure: Find all references to a symbol
CREATE PROCEDURE sp_find_references(
    IN p_symbol_id VARCHAR(512)
)
LANGUAGE SQLSCRIPT AS
BEGIN
    SELECT 
        file_path,
        start_line,
        start_column,
        end_line,
        end_column,
        is_definition,
        is_write,
        syntax_kind
    FROM SCIP_OCCURRENCES
    WHERE symbol_id = p_symbol_id
    ORDER BY file_path, start_line, start_column;
END;

-- Procedure: Get symbol definition
CREATE PROCEDURE sp_get_definition(
    IN p_symbol_id VARCHAR(512)
)
LANGUAGE SQLSCRIPT AS
BEGIN
    SELECT 
        name,
        kind,
        file_path,
        line,
        column,
        documentation,
        package,
        scheme
    FROM SCIP_SYMBOLS
    WHERE symbol_id = p_symbol_id
    AND is_definition = TRUE;
END;

-- Procedure: Search symbols by name pattern
CREATE PROCEDURE sp_search_symbols(
    IN p_project_id VARCHAR(128),
    IN p_name_pattern VARCHAR(256)
)
LANGUAGE SQLSCRIPT AS
BEGIN
    SELECT 
        symbol_id,
        name,
        kind,
        file_path,
        line,
        column,
        package
    FROM SCIP_SYMBOLS
    WHERE project_id = p_project_id
    AND name LIKE p_name_pattern
    ORDER BY name, file_path;
END;

-- =============================================================================
-- Grants
-- =============================================================================

-- Grant access to the application user
GRANT SELECT, INSERT, UPDATE, DELETE ON SCIP_INDEXES TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON SCIP_SYMBOLS TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON SCIP_OCCURRENCES TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON SCIP_RELATIONSHIPS TO SHIMMY_USER;

GRANT SELECT ON v_scip_definitions TO SHIMMY_USER;
GRANT SELECT ON v_scip_references TO SHIMMY_USER;
GRANT SELECT ON v_scip_project_stats TO SHIMMY_USER;

GRANT EXECUTE ON sp_find_references TO SHIMMY_USER;
GRANT EXECUTE ON sp_get_definition TO SHIMMY_USER;
GRANT EXECUTE ON sp_search_symbols TO SHIMMY_USER;

-- =============================================================================
-- Sample Queries
-- =============================================================================

-- Example: Get all symbols in a file
-- SELECT * FROM SCIP_SYMBOLS 
-- WHERE project_id = 'my-project' 
-- AND file_path = 'src/main.zig';

-- Example: Find all references to a symbol
-- CALL sp_find_references('scip-zig+my-project+main.MyStruct#');

-- Example: Get symbol at specific location
-- SELECT * FROM SCIP_OCCURRENCES
-- WHERE file_path = 'src/main.zig'
-- AND start_line <= 10 AND end_line >= 10
-- AND start_column <= 5 AND end_column >= 5;

-- Example: Project statistics
-- SELECT * FROM v_scip_project_stats WHERE project_name = 'my-project';