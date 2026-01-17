-- Audio Entity Schema
-- Stores metadata for generated audio files

CREATE TABLE IF NOT EXISTS Audio (
    AudioId TEXT PRIMARY KEY,
    SourceId TEXT NOT NULL,
    Title TEXT NOT NULL,
    FilePath TEXT NOT NULL,
    FileSize INTEGER NOT NULL DEFAULT 0,
    DurationSeconds REAL NOT NULL DEFAULT 0.0,
    SampleRate INTEGER NOT NULL DEFAULT 48000,
    BitDepth INTEGER NOT NULL DEFAULT 24,
    Channels INTEGER NOT NULL DEFAULT 2,
    Provider TEXT NOT NULL DEFAULT 'audiolabshimmy',
    Voice TEXT NOT NULL DEFAULT 'default',
    GeneratedAt INTEGER NOT NULL,
    ProcessingTimeMs INTEGER,
    Status TEXT NOT NULL DEFAULT 'pending',
    ErrorMessage TEXT,
    FOREIGN KEY (SourceId) REFERENCES Source(SourceId) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_audio_source ON Audio(SourceId);
CREATE INDEX IF NOT EXISTS idx_audio_status ON Audio(Status);
CREATE INDEX IF NOT EXISTS idx_audio_generated ON Audio(GeneratedAt DESC);

-- Sample data for testing (optional)
-- INSERT INTO Audio (AudioId, SourceId, Title, FilePath, FileSize, DurationSeconds, GeneratedAt, Status)
-- VALUES (
--     'audio_test_001',
--     'source_test_001',
--     'Audio Overview: Test Document',
--     'data/audio/test_001.mp3',
--     0,
--     0.0,
--     strftime('%s', 'now'),
--     'pending_integration'
-- );
