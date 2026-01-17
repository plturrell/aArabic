-- Load schema
.read server/schema_presentations.sql

-- Insert test presentation
INSERT INTO Presentation (
    PresentationId,
    SourceId,
    Title,
    Author,
    Theme,
    FilePath,
    FileSize,
    NumSlides,
    TargetAudience,
    DetailLevel,
    GeneratedAt,
    ProcessingTimeMs,
    Status,
    Version,
    ExportFormat
) VALUES (
    'pres_source_001_1737030000',
    'source_001',
    'AI Research Overview',
    'Research Team',
    'professional',
    'data/presentations/presentation_1737030000.html',
    15234,
    7,
    'executive',
    'medium',
    strftime('%s', 'now'),
    250,
    'completed',
    1,
    'html'
);

-- Insert slides
INSERT INTO Slide (SlideId, PresentationId, SlideNumber, Layout, Title, Content)
VALUES 
    ('slide_001_001', 'pres_source_001_1737030000', 1, 'title', 'AI Research Overview', 'Research Team'),
    ('slide_001_002', 'pres_source_001_1737030000', 2, 'content', 'Overview', 'This presentation...'),
    ('slide_001_003', 'pres_source_001_1737030000', 3, 'bullet_points', 'Key Concepts', 'Core Concepts...'),
    ('slide_001_004', 'pres_source_001_1737030000', 4, 'two_column', 'Methodology', 'Systematic approach...'),
    ('slide_001_005', 'pres_source_001_1737030000', 5, 'bullet_points', 'Key Findings', 'Major breakthrough...'),
    ('slide_001_006', 'pres_source_001_1737030000', 6, 'image', 'Architecture', 'System architecture...'),
    ('slide_001_007', 'pres_source_001_1737030000', 7, 'conclusion', 'Conclusion', 'Thank you...');

-- Query test
SELECT 'Presentation Created:' as Status;
SELECT * FROM Presentation;

SELECT 'Slides:' as Status;
SELECT SlideId, SlideNumber, Layout, Title FROM Slide ORDER BY SlideNumber;

-- Test views
SELECT 'Recent Presentations View:' as Status;
SELECT * FROM RecentPresentations;

SELECT 'Version History:' as Status;
SELECT * FROM PresentationVersions;
