-- Presentation Entity Schema
-- Day 48: Slide Export (HTML)
-- Stores metadata for generated presentations

CREATE TABLE IF NOT EXISTS Presentation (
    PresentationId TEXT PRIMARY KEY,
    SourceId TEXT NOT NULL,
    Title TEXT NOT NULL,
    Author TEXT NOT NULL DEFAULT 'HyperShimmy',
    Theme TEXT NOT NULL DEFAULT 'professional',
    FilePath TEXT NOT NULL,
    FileSize INTEGER NOT NULL DEFAULT 0,
    NumSlides INTEGER NOT NULL DEFAULT 0,
    TargetAudience TEXT NOT NULL DEFAULT 'general',
    DetailLevel TEXT NOT NULL DEFAULT 'medium',
    GeneratedAt INTEGER NOT NULL,
    ProcessingTimeMs INTEGER,
    Status TEXT NOT NULL DEFAULT 'completed',
    ErrorMessage TEXT,
    Version INTEGER NOT NULL DEFAULT 1,
    ExportFormat TEXT NOT NULL DEFAULT 'html',
    FOREIGN KEY (SourceId) REFERENCES Source(SourceId) ON DELETE CASCADE
);

-- Individual slides metadata table
CREATE TABLE IF NOT EXISTS Slide (
    SlideId TEXT PRIMARY KEY,
    PresentationId TEXT NOT NULL,
    SlideNumber INTEGER NOT NULL,
    Layout TEXT NOT NULL,
    Title TEXT NOT NULL,
    Content TEXT NOT NULL,
    Subtitle TEXT,
    Notes TEXT,
    FOREIGN KEY (PresentationId) REFERENCES Presentation(PresentationId) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_presentation_source ON Presentation(SourceId);
CREATE INDEX IF NOT EXISTS idx_presentation_status ON Presentation(Status);
CREATE INDEX IF NOT EXISTS idx_presentation_generated ON Presentation(GeneratedAt DESC);
CREATE INDEX IF NOT EXISTS idx_presentation_theme ON Presentation(Theme);
CREATE INDEX IF NOT EXISTS idx_slide_presentation ON Slide(PresentationId);
CREATE INDEX IF NOT EXISTS idx_slide_number ON Slide(PresentationId, SlideNumber);

-- Presentation versions view (for tracking revisions)
CREATE VIEW IF NOT EXISTS PresentationVersions AS
SELECT 
    PresentationId,
    SourceId,
    Title,
    Version,
    GeneratedAt,
    NumSlides,
    Theme
FROM Presentation
ORDER BY SourceId, Version DESC;

-- Recent presentations view
CREATE VIEW IF NOT EXISTS RecentPresentations AS
SELECT 
    p.PresentationId,
    p.SourceId,
    p.Title,
    p.Author,
    p.Theme,
    p.NumSlides,
    p.FileSize,
    p.GeneratedAt,
    p.Status,
    s.Title as SourceTitle
FROM Presentation p
LEFT JOIN Source s ON p.SourceId = s.SourceId
ORDER BY p.GeneratedAt DESC
LIMIT 50;

-- Sample data for testing (optional)
-- INSERT INTO Presentation (
--     PresentationId, 
--     SourceId, 
--     Title, 
--     Author, 
--     Theme, 
--     FilePath, 
--     FileSize, 
--     NumSlides, 
--     GeneratedAt, 
--     Status
-- ) VALUES (
--     'pres_test_001',
--     'source_test_001',
--     'Test Presentation',
--     'Test Author',
--     'professional',
--     'data/presentations/test_001.html',
--     0,
--     7,
--     strftime('%s', 'now'),
--     'completed'
-- );

-- Sample slide data
-- INSERT INTO Slide (
--     SlideId,
--     PresentationId,
--     SlideNumber,
--     Layout,
--     Title,
--     Content
-- ) VALUES (
--     'slide_test_001_001',
--     'pres_test_001',
--     1,
--     'title',
--     'Test Presentation',
--     'Test Author'
-- );
