# AudioLabShimmy API Reference

**Version:** 1.0.0  
**Last Updated:** January 17, 2026

---

## Overview

AudioLabShimmy is a professional-grade Text-to-Speech system that produces Dolby-quality audio output. This document provides comprehensive API documentation for developers integrating the system into their applications.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core API](#core-api)
- [Audio Processing](#audio-processing)
- [Text Processing](#text-processing)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Quick Start

### Basic Usage

```mojo
from nAudioLab.inference.engine import TTSEngine

fn main() raises:
    # Initialize the TTS engine
    var tts = TTSEngine.load("data/models")
    
    # Synthesize speech
    var audio = tts.synthesize("Hello, world!")
    
    # Save to file
    audio.save("output.wav")
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourorg/nAudioLab.git
cd nAudioLab

# Install dependencies
./scripts/install_dependencies.sh

# Download pre-trained models
./scripts/download_models.sh
```

---

## Core API

### TTSEngine

The main interface for text-to-speech synthesis.

#### Constructor

```mojo
TTSEngine.load(model_dir: String) -> TTSEngine
```

Loads pre-trained models from the specified directory.

**Parameters:**
- `model_dir` (String): Path to directory containing FastSpeech2 and HiFiGAN models

**Returns:**
- `TTSEngine`: Initialized TTS engine instance

**Raises:**
- `FileNotFoundError`: If model files are not found
- `ModelLoadError`: If models fail to load

**Example:**
```mojo
var tts = TTSEngine.load("data/models")
```

---

#### synthesize()

```mojo
fn synthesize(
    self,
    text: String,
    speed: Float32 = 1.0,
    pitch_shift: Float32 = 0.0,
    energy_scale: Float32 = 1.0
) -> AudioBuffer
```

Converts text to speech audio.

**Parameters:**
- `text` (String): Input text to synthesize
- `speed` (Float32, optional): Speech rate multiplier (0.5-2.0). Default: 1.0
  - 0.5 = half speed (slow)
  - 1.0 = normal speed
  - 2.0 = double speed (fast)
- `pitch_shift` (Float32, optional): Pitch adjustment in semitones (-12 to +12). Default: 0.0
  - Negative values = lower pitch
  - Positive values = higher pitch
- `energy_scale` (Float32, optional): Volume scaling factor (0.5-2.0). Default: 1.0

**Returns:**
- `AudioBuffer`: Generated audio with properties:
  - `sample_rate`: 48000 Hz
  - `channels`: 2 (stereo)
  - `bit_depth`: 24-bit
  - `samples`: Float32 array in range [-1.0, 1.0]

**Raises:**
- `ValueError`: If parameters are out of valid ranges
- `TextProcessingError`: If text cannot be normalized
- `SynthesisError`: If synthesis fails

**Example:**
```mojo
# Normal synthesis
var audio = tts.synthesize("Hello, world!")

# Slow speech, lower pitch
var audio_slow = tts.synthesize(
    "This is slower and deeper",
    speed=0.75,
    pitch_shift=-3.0
)

# Fast speech, higher pitch
var audio_fast = tts.synthesize(
    "This is faster and higher",
    speed=1.5,
    pitch_shift=2.0
)
```

---

#### synthesize_batch()

```mojo
fn synthesize_batch(
    self,
    texts: List[String],
    speed: Float32 = 1.0,
    pitch_shift: Float32 = 0.0
) -> List[AudioBuffer]
```

Synthesizes multiple texts in batch for better performance.

**Parameters:**
- `texts` (List[String]): List of input texts
- `speed` (Float32, optional): Speech rate for all texts. Default: 1.0
- `pitch_shift` (Float32, optional): Pitch adjustment for all texts. Default: 0.0

**Returns:**
- `List[AudioBuffer]`: List of generated audio buffers

**Example:**
```mojo
var texts = List[String]()
texts.append("First sentence")
texts.append("Second sentence")
texts.append("Third sentence")

var audios = tts.synthesize_batch(texts)
```

---

### AudioBuffer

Represents audio data with metadata.

#### Properties

```mojo
struct AudioBuffer:
    var samples: DTypePointer[DType.float32]  # Audio samples
    var length: Int                            # Number of samples
    var sample_rate: Int                       # Samples per second (48000)
    var channels: Int                          # Number of channels (2)
    var bit_depth: Int                         # Bits per sample (24)
```

#### Methods

##### save()

```mojo
fn save(self, path: String, format: String = "wav") raises
```

Saves audio to file.

**Parameters:**
- `path` (String): Output file path
- `format` (String, optional): Output format ("wav", "mp3", "flac"). Default: "wav"

**Formats:**
- `"wav"`: Uncompressed WAV (48kHz/24-bit/stereo)
- `"mp3"`: MP3 with 320kbps bitrate
- `"flac"`: Lossless FLAC compression

**Example:**
```mojo
audio.save("output.wav")
audio.save("output.mp3", format="mp3")
audio.save("output.flac", format="flac")
```

---

##### get_duration()

```mojo
fn get_duration(self) -> Float32
```

Returns audio duration in seconds.

**Example:**
```mojo
var duration = audio.get_duration()
print(f"Audio duration: {duration:.2f} seconds")
```

---

##### resample()

```mojo
fn resample(self, target_rate: Int) -> AudioBuffer
```

Resamples audio to different sample rate.

**Parameters:**
- `target_rate` (Int): Target sample rate (e.g., 16000, 22050, 44100)

**Returns:**
- `AudioBuffer`: Resampled audio

**Example:**
```mojo
var audio_16k = audio.resample(16000)
```

---

## Audio Processing

### Dolby Processing

Post-processing pipeline for professional audio quality.

```mojo
from nAudioLab.zig.dolby_processor import DolbyProcessor, DolbyConfig

fn apply_dolby_processing(audio: AudioBuffer) -> AudioBuffer:
    var config = DolbyConfig{
        target_lufs: -16.0,           # Loudness target
        compression_ratio: 3.0,        # Compression ratio
        attack_ms: 5.0,                # Compressor attack
        release_ms: 50.0,              # Compressor release
        enhancer_amount: 0.3,          # Harmonic enhancement
    }
    
    var processor = DolbyProcessor(config)
    return processor.process(audio)
```

#### DolbyConfig

Configuration for Dolby audio processing.

**Parameters:**
- `target_lufs` (Float32): Target loudness in LUFS (-24 to -12). Default: -16.0
- `compression_ratio` (Float32): Dynamic range compression (1.0-10.0). Default: 3.0
- `attack_ms` (Float32): Compressor attack time in milliseconds (0.1-100). Default: 5.0
- `release_ms` (Float32): Compressor release time in milliseconds (10-1000). Default: 50.0
- `enhancer_amount` (Float32): Harmonic enhancement amount (0.0-1.0). Default: 0.3

---

## Text Processing

### Text Normalization

Converts raw text to speakable form.

```mojo
from nAudioLab.text.normalizer import TextNormalizer

var normalizer = TextNormalizer()
var normalized = normalizer.normalize("Dr. Smith paid $10.50 on 1/17/2026")
# Result: "Doctor Smith paid ten dollars and fifty cents on January seventeenth, twenty twenty six"
```

#### Supported Conversions

| Input | Output |
|-------|--------|
| Numbers | `42` → "forty two" |
| Decimals | `3.14` → "three point one four" |
| Currency | `$10.50` → "ten dollars and fifty cents" |
| Dates | `1/17/2026` → "January seventeenth, twenty twenty six" |
| Abbreviations | `Dr.` → "Doctor", `St.` → "Street" |
| Phone numbers | `1-800-555-1234` → "one eight hundred five five five one two three four" |
| URLs | `example.com` → "example dot com" |

---

### Phonemization

Converts text to phonemes for synthesis.

```mojo
from nAudioLab.text.phoneme import Phonemizer

var phonemizer = Phonemizer()
var phonemes = phonemizer.text_to_phonemes("hello world")
# Result: [HH, EH1, L, OW0, W, ER1, L, D]
```

---

## Configuration

### Model Configuration

Configure model behavior at runtime.

```mojo
var config = TTSConfig{
    model_dir: "data/models",
    use_dolby: true,
    cache_models: true,
    num_threads: 8,
    device: "cpu",
}

var tts = TTSEngine.load_with_config(config)
```

#### TTSConfig

**Parameters:**
- `model_dir` (String): Path to model directory
- `use_dolby` (Bool): Enable Dolby post-processing. Default: true
- `cache_models` (Bool): Cache models in memory. Default: true
- `num_threads` (Int): Number of CPU threads (1-16). Default: 8
- `device` (String): Compute device ("cpu", "metal"). Default: "cpu"

---

### Audio Output Settings

```mojo
var audio_config = AudioConfig{
    sample_rate: 48000,
    channels: 2,
    bit_depth: 24,
    output_format: "wav",
}
```

---

## Error Handling

### Exception Types

```mojo
# Model errors
ModelLoadError          # Failed to load model files
ModelInferenceError     # Inference failed

# Text processing errors
TextProcessingError     # Text normalization failed
PhonemeError           # Phonemization failed

# Audio errors
AudioIOError           # File read/write failed
AudioFormatError       # Unsupported format
ResamplingError        # Resampling failed

# Parameter errors
ValueError             # Invalid parameter value
FileNotFoundError      # File not found
```

### Error Handling Example

```mojo
try:
    var tts = TTSEngine.load("data/models")
    var audio = tts.synthesize("Hello!")
    audio.save("output.wav")
except e: ModelLoadError:
    print("Failed to load models:", e.message)
except e: SynthesisError:
    print("Synthesis failed:", e.message)
except e: AudioIOError:
    print("Failed to save audio:", e.message)
```

---

## Examples

### Example 1: Basic Synthesis

```mojo
from nAudioLab.inference.engine import TTSEngine

fn main() raises:
    var tts = TTSEngine.load("data/models")
    var audio = tts.synthesize("Welcome to AudioLabShimmy!")
    audio.save("welcome.wav")
    print("Audio saved to welcome.wav")
```

---

### Example 2: Voice Variations

```mojo
fn create_voice_variations() raises:
    var tts = TTSEngine.load("data/models")
    var text = "This demonstrates voice variations"
    
    # Normal voice
    var normal = tts.synthesize(text)
    normal.save("normal.wav")
    
    # Deep voice (slow + low pitch)
    var deep = tts.synthesize(text, speed=0.8, pitch_shift=-4.0)
    deep.save("deep.wav")
    
    # High voice (fast + high pitch)
    var high = tts.synthesize(text, speed=1.2, pitch_shift=4.0)
    high.save("high.wav")
```

---

### Example 3: Batch Processing

```mojo
fn process_book_chapter() raises:
    var tts = TTSEngine.load("data/models")
    
    var paragraphs = List[String]()
    paragraphs.append("First paragraph of the chapter.")
    paragraphs.append("Second paragraph continues the story.")
    paragraphs.append("Final paragraph concludes.")
    
    var audios = tts.synthesize_batch(paragraphs)
    
    # Concatenate all audio
    var combined = AudioBuffer.concatenate(audios)
    combined.save("chapter.mp3", format="mp3")
```

---

### Example 4: Custom Audio Processing

```mojo
fn custom_processing() raises:
    var tts = TTSEngine.load("data/models")
    var audio = tts.synthesize("Custom audio processing example")
    
    # Apply custom Dolby settings
    var config = DolbyConfig{
        target_lufs: -18.0,        # Quieter for audiobook
        compression_ratio: 2.5,    # Lighter compression
        enhancer_amount: 0.2,      # Subtle enhancement
    }
    
    var processor = DolbyProcessor(config)
    var processed = processor.process(audio)
    processed.save("custom.wav")
```

---

### Example 5: Real-time Streaming

```mojo
fn stream_synthesis() raises:
    var tts = TTSEngine.load("data/models")
    var text = "This is a long text that will be streamed..."
    
    # Split into sentences
    var sentences = text.split(".")
    
    for sentence in sentences:
        if len(sentence.strip()) > 0:
            var audio = tts.synthesize(sentence)
            # Play audio immediately
            audio.play()  # Non-blocking playback
```

---

### Example 6: Multilingual Support (Future)

```mojo
fn multilingual_synthesis() raises:
    var tts = TTSEngine.load("data/models")
    
    # English
    var en = tts.synthesize("Hello, world!", language="en")
    en.save("hello_en.wav")
    
    # Spanish (future support)
    # var es = tts.synthesize("Hola, mundo!", language="es")
    # es.save("hello_es.wav")
```

---

## Performance Considerations

### CPU Optimization

- **Threads**: Use 8-16 threads for best performance on modern CPUs
- **Batch Size**: Process 4-8 sentences at once for optimal throughput
- **Caching**: Keep models cached in memory (enabled by default)

### Memory Usage

- **Model Memory**: ~500MB for FastSpeech2 + HiFiGAN
- **Audio Buffer**: ~10MB per minute of audio (48kHz/24-bit/stereo)
- **Peak Memory**: ~1GB during synthesis

### Synthesis Speed

- **Real-time Factor**: <0.1x (10x faster than real-time)
- **Example**: 10-second audio synthesized in ~1 second
- **Batch Processing**: 2-3x faster than sequential

---

## Quality Metrics

AudioLabShimmy meets professional broadcast standards:

| Metric | Target | Achieved |
|--------|--------|----------|
| Sample Rate | 48kHz | ✓ |
| Bit Depth | 24-bit | ✓ |
| LUFS Loudness | -16 ± 1 dB | ✓ |
| THD+N | < 1% | ✓ |
| Dynamic Range | ≥ 6 dB | ✓ |
| Stereo Imaging | Balanced | ✓ |

---

## Support & Resources

- **Documentation**: https://docs.audiolabshimmy.org
- **Examples**: `/examples/` directory
- **Issues**: https://github.com/yourorg/nAudioLab/issues
- **Community**: https://discord.gg/audiolabshimmy

---

**API Version:** 1.0.0  
**Last Updated:** January 17, 2026  
**License:** MIT
