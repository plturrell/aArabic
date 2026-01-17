# Day 42 Complete: TTS Integration Planning & AudioLabShimmy Bridge ✅

**Date:** January 16, 2026  
**Focus:** Week 9, Day 42 - TTS Integration Architecture  
**Status:** ✅ **COMPLETE**

---

## 📋 Objectives

Create integration architecture between HyperShimmy and AudioLabShimmy:
- ✅ Document integration approach
- ✅ Create placeholder TTS provider interface
- ✅ Set up AudioLabShimmy project reference
- ✅ Define FFI contract for future integration
- ✅ Create stub implementation for development

---

## 🎯 Decision: Custom TTS Engine

After Day 41 research, the decision was made to build a **custom TTS engine (AudioLabShimmy)** instead of using OpenAI's API:

### Why Custom TTS?

1. **100% Control** - Complete ownership of technology stack
2. **Zero API Costs** - No per-character fees
3. **Privacy** - All processing stays local
4. **Quality** - Dolby-grade audio (48kHz/24-bit)
5. **Alignment** - Fits "100% Mojo/Zig" philosophy

### AudioLabShimmy Project

A separate 40-day implementation project:
- **Location:** `src/serviceCore/nAudioLab/`
- **Technology:** Mojo (neural models) + Zig (audio I/O)
- **Architecture:** FastSpeech2 + HiFiGAN
- **Training:** CPU-optimized for Apple Silicon
- **Quality:** Professional-grade Dolby processing

See: `src/serviceCore/nAudioLab/README.md`

---

## 🏗️ Integration Architecture

### **System Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                      HyperShimmy                             │
│                                                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Zig OData Server                                   │    │
│  │  • GenerateAudio action (endpoint)                 │    │
│  │  • Audio entity CRUD                               │    │
│  │  • File serving                                    │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │ FFI Call                                 │
│                   ▼                                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Mojo TTS Bridge (hypershimmy_tts.mojo)            │    │
│  │  • Provider abstraction                            │    │
│  │  • Text preprocessing                              │    │
│  │  • Format conversion                               │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │ FFI Call                                 │
│                   ▼                                          │
└───────────────────┼──────────────────────────────────────────┘
                    │
┌───────────────────┼──────────────────────────────────────────┐
│                   ▼                                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  AudioLabShimmy Engine                              │    │
│  │  • TTSEngine.synthesize(text)                      │    │
│  │  • Returns 48kHz/24-bit audio                      │    │
│  │  • Dolby processing applied                        │    │
│  └────────────────────────────────────────────────────┘    │
│                   AudioLabShimmy                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 📄 Files Created

### **1. Mojo TTS Provider Interface**

**File:** `mojo/tts_provider.mojo`

```mojo
"""
TTS Provider abstraction for HyperShimmy.
Provides interface for audio generation that will be implemented
by AudioLabShimmy once that project is complete.
"""

struct AudioConfig:
    """Configuration for audio generation."""
    var sample_rate: Int
    var channels: Int
    var bit_depth: Int
    
    fn __init__(inout self):
        self.sample_rate = 48000  # Professional quality
        self.channels = 2          # Stereo
        self.bit_depth = 24        # Studio standard

struct AudioBuffer:
    """Container for generated audio data."""
    var samples: DTypePointer[DType.float32]
    var length: Int
    var config: AudioConfig
    
    fn duration_seconds(self) -> Float64:
        """Calculate audio duration in seconds."""
        return Float64(self.length) / Float64(self.config.sample_rate * self.config.channels)

trait TTSProvider:
    """
    Abstract TTS provider interface.
    
    This will be implemented by AudioLabShimmy's TTSEngine.
    For now, provides a stub implementation.
    """
    
    fn synthesize(self, text: String) raises -> AudioBuffer:
        """
        Convert text to speech.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            AudioBuffer containing generated speech
            
        Raises:
            Error if synthesis fails
        """
        ...
    
    fn synthesize_to_file(self, text: String, output_path: String) raises:
        """
        Convert text to speech and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Where to save the audio file (MP3)
        """
        ...

struct StubTTSProvider(TTSProvider):
    """
    Stub implementation for development.
    Returns empty audio buffer as placeholder.
    
    TODO: Replace with AudioLabShimmy integration once available.
    """
    
    fn __init__(inout self):
        pass
    
    fn synthesize(self, text: String) raises -> AudioBuffer:
        """Stub: Returns empty buffer."""
        var config = AudioConfig()
        var buffer = AudioBuffer(
            samples=DTypePointer[DType.float32](),
            length=0,
            config=config
        )
        
        print("[STUB] TTS synthesis requested for:", text[:50], "...")
        print("[STUB] AudioLabShimmy not yet integrated")
        
        return buffer
    
    fn synthesize_to_file(self, text: String, output_path: String) raises:
        """Stub: Creates empty file."""
        print("[STUB] Would generate audio file:", output_path)
        print("[STUB] Text length:", len(text), "characters")
        # TODO: Once AudioLabShimmy is ready, call:
        # let engine = AudioLabShimmy.TTSEngine.load()
        # engine.synthesize(text).save(output_path)

# Export for use in other modules
alias DefaultTTSProvider = StubTTSProvider
```

---

### **2. Zig Audio Handler (Placeholder)**

**File:** `server/audio_handler.zig`

```zig
const std = @import("std");

/// Audio file metadata
pub const AudioMetadata = struct {
    audio_id: []const u8,
    source_id: []const u8,
    title: []const u8,
    file_path: []const u8,
    file_size: u64,
    duration_seconds: f32,
    generated_at: i64,
    status: []const u8,
};

/// Audio generation request
pub const AudioRequest = struct {
    source_id: []const u8,
    text: []const u8,
    voice: []const u8 = "default",
    format: []const u8 = "mp3",
};

/// Generate audio from text using AudioLabShimmy
/// TODO: Implement once AudioLabShimmy is complete
pub fn generateAudio(
    allocator: std.mem.Allocator,
    request: AudioRequest
) !AudioMetadata {
    // Stub implementation
    std.log.info("Audio generation requested:", .{});
    std.log.info("  Source ID: {s}", .{request.source_id});
    std.log.info("  Text length: {} chars", .{request.text.len});
    std.log.info("  Voice: {s}", .{request.voice});
    std.log.info("  Format: {s}", .{request.format});
    
    // TODO: Once AudioLabShimmy is ready:
    // 1. Call Mojo FFI to AudioLabShimmy
    // 2. Generate audio file
    // 3. Save to data/audio/
    // 4. Return actual metadata
    
    // For now, return stub metadata
    const audio_id = try std.fmt.allocPrint(
        allocator,
        "audio_{s}_{d}",
        .{ request.source_id, std.time.timestamp() }
    );
    
    return AudioMetadata{
        .audio_id = audio_id,
        .source_id = request.source_id,
        .title = "Audio Overview (Pending AudioLabShimmy)",
        .file_path = "data/audio/stub.mp3",
        .file_size = 0,
        .duration_seconds = 0.0,
        .generated_at = std.time.timestamp(),
        .status = "pending_integration",
    };
}

/// Serve audio file
/// TODO: Implement file serving once audio generation works
pub fn serveAudioFile(
    audio_id: []const u8,
    writer: anytype
) !void {
    std.log.info("Audio file requested: {s}", .{audio_id});
    
    // TODO: Once AudioLabShimmy is ready:
    // 1. Look up audio_id in database
    // 2. Read file from data/audio/
    // 3. Stream to writer with proper headers
    
    // For now, return stub response
    try writer.writeAll("Audio file not yet available (AudioLabShimmy pending)");
}
```

---

## 🔗 FFI Contract Definition

### **Future Integration Contract**

When AudioLabShimmy is complete, the integration will work as follows:

```mojo
# In HyperShimmy (mojo/hypershimmy_tts.mojo)
from nAudioLab import TTSEngine

struct HyperShimmyAudioGenerator:
    var tts_engine: TTSEngine
    
    fn __init__(inout self):
        # Load AudioLabShimmy models
        self.tts_engine = TTSEngine.load("../nAudioLab/data/models")
    
    fn generate_audio_overview(self, source_id: String, text: String) -> String:
        """
        Generate audio overview for a research source.
        
        Args:
            source_id: Source document ID
            text: Summary text to convert to speech
            
        Returns:
            Path to generated audio file
        """
        # Generate audio using AudioLabShimmy
        let audio = self.tts_engine.synthesize(
            text=text,
            quality="dolby"  # 48kHz/24-bit with Dolby processing
        )
        
        # Save to file
        let output_path = f"data/audio/{source_id}_{timestamp()}.mp3"
        audio.save(output_path)
        
        return output_path
```

---

## 📊 Data Model

### **Audio Entity Schema**

```sql
CREATE TABLE Audio (
    AudioId TEXT PRIMARY KEY,
    SourceId TEXT NOT NULL,
    Title TEXT NOT NULL,
    FilePath TEXT NOT NULL,
    FileSize INTEGER NOT NULL,
    DurationSeconds REAL NOT NULL,
    SampleRate INTEGER DEFAULT 48000,
    BitDepth INTEGER DEFAULT 24,
    Channels INTEGER DEFAULT 2,
    Provider TEXT DEFAULT 'audiolabshimmy',
    Voice TEXT DEFAULT 'default',
    GeneratedAt INTEGER NOT NULL,
    ProcessingTimeMs INTEGER,
    Status TEXT NOT NULL,
    ErrorMessage TEXT,
    FOREIGN KEY (SourceId) REFERENCES Source(SourceId)
);

CREATE INDEX idx_audio_source ON Audio(SourceId);
CREATE INDEX idx_audio_status ON Audio(Status);
```

---

## 📁 Directory Structure

```
src/serviceCore/nHyperBook/
├── mojo/
│   └── tts_provider.mojo          ✅ Created (stub)
├── server/
│   └── audio_handler.zig          ✅ Created (stub)
└── data/
    └── audio/                     📁 Will store generated files

src/serviceCore/nAudioLab/    ✅ Project initialized
├── README.md
├── build.zig
├── docs/
│   └── implementation-plan.md     (40-day plan)
└── [Implementation pending]
```

---

## ⏭️ Next Steps

### **Day 43: Optional Piper TTS (Skipped)**

Since we're building AudioLabShimmy from scratch, the Piper TTS integration (Day 43) is superseded. We'll move directly to Day 44.

### **Day 44: Audio OData Action**

Create the OData endpoint that will call the TTS provider:

**Tasks:**
1. Add Audio entity to OData schema
2. Implement `GenerateAudio` action
3. Add audio file serving endpoint
4. Wire up to stub TTS provider
5. Test endpoint (will return stub data for now)

**Files to Create:**
- `server/odata_audio.zig` - OData audio endpoints
- Update OData schema with Audio entity
- Add routes for audio file serving

### **Day 45: Audio UI**

Create the SAPUI5 interface for audio generation:

**Tasks:**
1. Create `webapp/view/Audio.view.xml`
2. Create `webapp/controller/Audio.controller.js`
3. Add audio playback controls
4. Add "Generate Audio" button
5. Show stub message until AudioLabShimmy is ready

---

## 🔄 AudioLabShimmy Integration Timeline

### **Phase 1: Stub (Current - Day 42)**
- ✅ Interface defined
- ✅ Stub implementation
- ✅ Can develop UI/OData endpoints

### **Phase 2: AudioLabShimmy Development (Days 1-40)**
- Build custom TTS engine
- Train FastSpeech2 + HiFiGAN models
- Implement Dolby processing
- See: `nAudioLab/docs/implementation-plan.md`

### **Phase 3: Integration (Post AudioLabShimmy Day 40)**
- Replace stub with real AudioLabShimmy calls
- Test end-to-end audio generation
- Validate audio quality
- Performance optimization

---

## ✅ Completion Checklist

- [x] Document decision to build custom TTS
- [x] Create AudioLabShimmy project skeleton
- [x] Define TTS provider interface (Mojo)
- [x] Create stub implementation
- [x] Create audio handler (Zig stub)
- [x] Define FFI contract for future integration
- [x] Document data model
- [x] Plan Day 44-45 tasks
- [x] Write completion documentation

---

## 🎉 Summary

**Day 42 establishes the integration architecture between HyperShimmy and AudioLabShimmy!**

### Key Achievements:

1. **Architecture Defined** - Clear separation between projects
2. **Interface Created** - TTSProvider trait in Mojo
3. **Stub Implementation** - Can continue HyperShimmy development
4. **AudioLabShimmy Started** - 40-day project initialized

### Development Strategy:

1. **Parallel Development:**
   - HyperShimmy continues with stub TTS
   - AudioLabShimmy developed separately
   - Integration happens when AudioLabShimmy is ready

2. **Benefits:**
   - No blocking dependencies
   - UI/OData can be built and tested
   - Custom TTS provides superior quality
   - Zero API costs

3. **Future Integration:**
   - Simple: Replace `StubTTSProvider` with `AudioLabShimmy.TTSEngine`
   - Single line change once AudioLabShimmy is complete
   - FFI contract already defined

### Next:
- **Day 44:** Audio OData action (will use stub for now)
- **Day 45:** Audio UI (will show "pending" state)
- **Later:** Integrate real AudioLabShimmy engine

**Status:** ✅ Complete - Architecture defined, stub implementation ready  
**Confidence:** High - Clear path for future integration

---

*Completed: January 16, 2026*
