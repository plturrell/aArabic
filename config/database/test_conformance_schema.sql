-- =============================================================================
-- Test Conformance Schema for nGrounding
-- Migrates .lean test files to HANA tables, vectors, and graphs
-- =============================================================================

-- Conformance Test Cases Table
CREATE COLUMN TABLE CONFORMANCE_TESTS (
    TEST_ID NVARCHAR(100) PRIMARY KEY,
    TEST_TYPE NVARCHAR(50) NOT NULL,  -- 'lexer', 'parser', 'elaboration', 'basic'
    TEST_CATEGORY NVARCHAR(50),       -- 'golden', 'conformance', 'integration'
    INPUT_CODE NCLOB NOT NULL,
    EXPECTED_OUTPUT NCLOB,
    EXPECTED_TOKENS NCLOB,            -- For lexer tests (JSON array)
    EXPECTED_SYNTAX NCLOB,            -- For parser tests (JSON)
    TEST_STATUS NVARCHAR(20) DEFAULT 'ACTIVE',
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    LAST_RUN TIMESTAMP,
    PASS_COUNT INTEGER DEFAULT 0,
    FAIL_COUNT INTEGER DEFAULT 0
);

-- Test Vectors for Similarity Search
CREATE COLUMN TABLE TEST_VECTORS (
    VECTOR_ID NVARCHAR(100) PRIMARY KEY,
    TEST_ID NVARCHAR(100) NOT NULL,
    EMBEDDING_VECTOR REAL_VECTOR(768),
    VECTOR_TYPE NVARCHAR(50),         -- 'input', 'output', 'syntax'
    METADATA NCLOB,                   -- JSON metadata
    FOREIGN KEY (TEST_ID) REFERENCES CONFORMANCE_TESTS(TEST_ID) ON DELETE CASCADE
);

-- Test Execution History
CREATE COLUMN TABLE TEST_EXECUTION_HISTORY (
    EXECUTION_ID NVARCHAR(100) PRIMARY KEY,
    TEST_ID NVARCHAR(100) NOT NULL,
    EXECUTION_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    RESULT NVARCHAR(20) NOT NULL,     -- 'PASS', 'FAIL', 'SKIP', 'ERROR'
    ACTUAL_OUTPUT NCLOB,
    ERROR_MESSAGE NCLOB,
    DURATION_MS INTEGER,
    FOREIGN KEY (TEST_ID) REFERENCES CONFORMANCE_TESTS(TEST_ID) ON DELETE CASCADE
);

-- Test Dependencies Graph
CREATE GRAPH WORKSPACE TEST_DEPENDENCIES
    EDGE TABLE TEST_EDGES
        SOURCE COLUMN SOURCE
        TARGET COLUMN TARGET
        KEY COLUMN EDGE_ID
    (
        EDGE_ID NVARCHAR(100) PRIMARY KEY,
        SOURCE NVARCHAR(100) NOT NULL,
        TARGET NVARCHAR(100) NOT NULL,
        DEPENDENCY_TYPE NVARCHAR(50) NOT NULL,  -- 'requires', 'extends', 'similar_to'
        WEIGHT DOUBLE DEFAULT 1.0,
        METADATA NCLOB
    )
    VERTEX TABLE TEST_NODES
        KEY COLUMN NODE_ID
    (
        NODE_ID NVARCHAR(100) PRIMARY KEY,
        NODE_TYPE NVARCHAR(50) NOT NULL,        -- 'test', 'definition', 'theorem'
        NODE_LABEL NVARCHAR(200),
        COMPLEXITY INTEGER DEFAULT 1,
        METADATA NCLOB
    );

-- Indexes for Performance
CREATE INDEX IDX_TEST_TYPE ON CONFORMANCE_TESTS(TEST_TYPE);
CREATE INDEX IDX_TEST_CATEGORY ON CONFORMANCE_TESTS(TEST_CATEGORY);
CREATE INDEX IDX_TEST_STATUS ON CONFORMANCE_TESTS(TEST_STATUS);
CREATE INDEX IDX_EXECUTION_TIME ON TEST_EXECUTION_HISTORY(EXECUTION_TIME);
CREATE INDEX IDX_TEST_RESULT ON TEST_EXECUTION_HISTORY(RESULT);

-- Views for Reporting
CREATE VIEW V_TEST_SUMMARY AS
SELECT 
    TEST_TYPE,
    TEST_CATEGORY,
    COUNT(*) AS TOTAL_TESTS,
    SUM(CASE WHEN TEST_STATUS = 'ACTIVE' THEN 1 ELSE 0 END) AS ACTIVE_TESTS,
    SUM(PASS_COUNT) AS TOTAL_PASSES,
    SUM(FAIL_COUNT) AS TOTAL_FAILURES,
    AVG(PASS_COUNT * 1.0 / (PASS_COUNT + FAIL_COUNT)) AS SUCCESS_RATE
FROM CONFORMANCE_TESTS
GROUP BY TEST_TYPE, TEST_CATEGORY;

CREATE VIEW V_RECENT_FAILURES AS
SELECT 
    t.TEST_ID,
    t.TEST_TYPE,
    t.TEST_CATEGORY,
    h.EXECUTION_TIME,
    h.ERROR_MESSAGE,
    h.DURATION_MS
FROM CONFORMANCE_TESTS t
JOIN TEST_EXECUTION_HISTORY h ON t.TEST_ID = h.TEST_ID
WHERE h.RESULT = 'FAIL'
ORDER BY h.EXECUTION_TIME DESC;

-- Initial Test Data Migration
-- Basic Lean4 tests
INSERT INTO CONFORMANCE_TESTS (TEST_ID, TEST_TYPE, TEST_CATEGORY, INPUT_CODE, EXPECTED_OUTPUT)
VALUES (
    'basic_001_hello',
    'basic',
    'conformance',
    'def hello := "Hello, World!"',
    '{ "type": "definition", "name": "hello", "value": "Hello, World!" }'
);

INSERT INTO CONFORMANCE_TESTS (TEST_ID, TEST_TYPE, TEST_CATEGORY, INPUT_CODE, EXPECTED_OUTPUT)
VALUES (
    'basic_002_add_one',
    'basic',
    'conformance',
    'def add_one (n : Nat) : Nat := n + 1',
    '{ "type": "definition", "name": "add_one", "params": [{"name": "n", "type": "Nat"}], "return_type": "Nat" }'
);

INSERT INTO CONFORMANCE_TESTS (TEST_ID, TEST_TYPE, TEST_CATEGORY, INPUT_CODE, EXPECTED_OUTPUT)
VALUES (
    'basic_003_is_even',
    'basic',
    'conformance',
    'def is_even (n : Nat) : Bool := n % 2 == 0',
    '{ "type": "definition", "name": "is_even", "params": [{"name": "n", "type": "Nat"}], "return_type": "Bool" }'
);

INSERT INTO CONFORMANCE_TESTS (TEST_ID, TEST_TYPE, TEST_CATEGORY, INPUT_CODE, EXPECTED_OUTPUT)
VALUES (
    'basic_004_theorem',
    'basic',
    'conformance',
    'theorem zero_add (n : Nat) : 0 + n = n := by simp',
    '{ "type": "theorem", "name": "zero_add", "statement": "0 + n = n" }'
);

-- Parser golden tests
INSERT INTO CONFORMANCE_TESTS (TEST_ID, TEST_TYPE, TEST_CATEGORY, INPUT_CODE, EXPECTED_SYNTAX)
VALUES (
    'parser_001_theorem',
    'parser',
    'golden',
    'theorem t : Nat := 0',
    '{ "kind": "theorem", "name": "t", "type": "Nat", "value": "0" }'
);

-- Test nodes for graph
INSERT INTO TEST_NODES (NODE_ID, NODE_TYPE, NODE_LABEL, COMPLEXITY)
VALUES 
    ('basic_001_hello', 'test', 'Basic hello definition', 1),
    ('basic_002_add_one', 'test', 'Function with parameters', 2),
    ('basic_003_is_even', 'test', 'Boolean function', 2),
    ('basic_004_theorem', 'test', 'Simple theorem', 3),
    ('parser_001_theorem', 'test', 'Parser theorem test', 2);

-- Test dependencies
INSERT INTO TEST_EDGES (EDGE_ID, SOURCE, TARGET, DEPENDENCY_TYPE, WEIGHT)
VALUES 
    ('edge_001', 'basic_002_add_one', 'basic_001_hello', 'requires', 1.0),
    ('edge_002', 'basic_004_theorem', 'basic_002_add_one', 'extends', 2.0),
    ('edge_003', 'parser_001_theorem', 'basic_004_theorem', 'similar_to', 0.8);

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON CONFORMANCE_TESTS TO <your_user>;
-- GRANT SELECT, INSERT ON TEST_EXECUTION_HISTORY TO <your_user>;
-- GRANT SELECT ON V_TEST_SUMMARY TO <your_user>;
