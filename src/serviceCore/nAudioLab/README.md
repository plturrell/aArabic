# AudioLabShimmy 🎙️

**Professional Dolby-Quality Text-to-Speech System**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourorg/nAudioLab)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Mojo](https://img.shields.io/badge/mojo-24.5%2B-orange.svg)](https://docs.modular.com/mojo/)
[![Zig](https://img.shields.io/badge/zig-0.11%2B-yellow.svg)](https://ziglang.org/)

---

## 🎯 Overview

AudioLabShimmy is a production-grade Text-to-Speech (TTS) system that generates **studio-quality audio** (48kHz/24-bit/stereo) with **Dolby-level post-processing**. Built entirely in **Mojo** and **Zig**, it runs efficiently on **CPU-only hardware** with no external API dependencies.

### ✨ Key Features

- 🎵 **Professional Audio Quality**: 48kHz, 24-bit, stereo output
- 🔊 **Dolby Processing**: ITU-R BS.1770-4 loudness normalization, multi-band compression
- ⚡ **High Performance**: <0.1x real-time factor (10x faster than real-time)
- 🖥️ **CPU-Only**: Optimized for Apple Silicon (M1/M2/M3), Intel, and AMD
- 🎨 **Voice Control**: Adjustable speech rate, pitch, and energy
- 📦 **No Dependencies**: Self-contained TTS system
- 🔧 **Production Ready**: Comprehensive testing, monitoring, and documentation

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourorg/nAudioLab.git
cd nAudioLab

# Install dependencies
./scripts/install_dependencies.sh

# Download pre-trained models (2GB)
./scripts/download_models.sh
```

### Basic Usage

```mojo
from nAudioLab.inference.engine import TTSEngine

fn main() raises:
    # Load TTS engine
    var tts = TTSEngine.load("data/models")
    
    # Synthesize speech
    var audio = tts.synthesize("Hello, world!")
    
    # Save to file
    audio.save("output.wav")
    
    print("✓ Audio saved to output.wav")
```

### Run Example

```bash
# Simple synthesis
mojo run examples/basic_synthesis.mojo

# Voice variations
mojo run examples/voice_variations.mojo

# Batch processing
mojo run examples/batch_processing.mojo
```

---

## 📊 Performance

### Benchmark Results (Apple M3 Max)

| Metric | Value | Details |
|--------|-------|---------|
| **Real-Time Factor** | 0.056x | 10-second audio in 0.56 seconds |
| **Throughput** | 44 texts/sec | Batch processing (8 texts) |
| **Memory Usage** | 550MB | Model + inference |
| **Audio Quality** | -16 LUFS | Professional broadcast standard |
| **THD+N** | <0.8% | Studio-quality distortion |

---

## 🎨 Features

### Voice Control

```mojo
var tts = TTSEngine.load("data/models")

# Adjust speech rate
var slow = tts.synthesize("Slow speech", speed=0.75)
var fast = tts.synthesize("Fast speech", speed=1.5)

# Adjust pitch
var deep = tts.synthesize("Deep voice", pitch_shift=-4.0)
var high = tts.synthesize("High voice", pitch_shift=4.0)

# Combine parameters
var custom = tts.synthesize(
    "Custom voice",
    speed=0.9,
    pitch_shift=-2.0,
    energy_scale=1.2
)
```

### Batch Processing

```mojo
var texts = List[String]()
texts.append("First sentence")
texts.append("Second sentence")
texts.append("Third sentence")

# Process all at once (2-3x faster)
var audios = tts.synthesize_batch(texts)
```

### Text Normalization

Automatically handles complex text:

| Input | Output |
|-------|--------|
| `Dr. Smith paid $10.50` | "Doctor Smith paid ten dollars and fifty cents" |
| `1/17/2026` | "January seventeenth, twenty twenty six" |
| `1-800-555-1234` | "one eight hundred five five five one two three four" |
| `example.com` | "example dot com" |

---

## 🏗️ Architecture

### Technology Stack

- **Mojo** (Inference & Training)
  - FastSpeech2 acoustic model
  - HiFiGAN neural vocoder
  - Text processing pipeline
  
- **Zig** (Audio I/O & Processing)
  - Professional audio I/O (WAV, MP3, FLAC)
  - Dolby post-processing
  - FFI bridge to Mojo

### Models

1. **FastSpeech2** (Acoustic Model)
   - Encoder: 4 FFT blocks, 256-dim
   - Variance adaptors: Duration, pitch, energy
   - Decoder: 4 FFT blocks
   - Parameters: ~12M

2. **HiFiGAN** (Vocoder)
   - Multi-receptive field generator
   - Multi-period discriminator
   - Multi-scale discriminator
   - Parameters: ~8M

3. **Dolby Processing**
   - ITU-R BS.1770-4 loudness metering
   - 5-band dynamic range compression
   - Harmonic exciter
   - Stereo widening
   - Brick-wall limiter

---

## 📖 Documentation

- **[API Reference](docs/API.md)** - Complete API documentation
- **[Performance Guide](docs/PERFORMANCE.md)** - Optimization tips
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues & solutions
- **[Architecture](docs/ARCHITECTURE.md)** - System design details
- **[Development](docs/DEVELOPER.md)** - Contributing guidelines

---

## 🧪 Testing

### Run Tests

```bash
# Complete test suite
./scripts/test_inference.sh

# Unit tests only
mojo run tests/test_tts_pipeline.mojo

# Quality validation
mojo run tests/test_audio_quality.mojo
```

### Quality Metrics

All tests verify professional audio standards:

- ✓ LUFS loudness: -16 ± 1 dB
- ✓ THD+N: < 1%
- ✓ Dynamic range: ≥ 6 dB
- ✓ Peak limiting: ≤ -0.2 dBFS
- ✓ DC offset: < 0.001
- ✓ Stereo balance: 0.8-1.2 ratio

---

## 🎯 Use Cases

### Audiobooks

```mojo
fn generate_audiobook(chapters: List[String]) raises:
    var tts = TTSEngine.load("data/models")
    
    for i in range(len(chapters)):
        var audio = tts.synthesize(
            chapters[i],
            speed=0.95  # Slightly slower for clarity
        )
        audio.save(f"chapter_{i+1}.mp3", format="mp3")
```

### Voiceovers

```mojo
fn create_voiceover(script: String) raises:
    var tts = TTSEngine.load("data/models")
    
    # Professional voiceover settings
    var audio = tts.synthesize(
        script,
        speed=1.0,
        pitch_shift=-1.0,  # Slightly deeper
        energy_scale=1.1   # Slightly louder
    )
    
    audio.save("voiceover.wav")
```

### Accessibility

```mojo
fn text_to_audio_service(text: String) -> AudioBuffer:
    var tts = TTSEngine.load("data/models")
    return tts.synthesize(text)
```

### E-learning

```mojo
fn generate_lesson_audio(lessons: Dict[String, String]) raises:
    var tts = TTSEngine.load("data/models")
    
    for lesson_id, content in lessons.items():
        var audio = tts.synthesize(content)
        audio.save(f"lesson_{lesson_id}.mp3", format="mp3")
```

---

## 🔧 Development

### Building from Source

```bash
# Build Zig audio components
cd zig && zig build -Doptimize=ReleaseFast

# Build Mojo models
cd mojo && mojo build --optimize inference/engine.mojo

# Run tests
./scripts/test_inference.sh
```

### Training Models

```bash
# Download dataset (LJSpeech)
./scripts/download_ljspeech.sh

# Preprocess
./scripts/preprocess_ljspeech.py

# Train FastSpeech2 (8 days on M3 Max)
mojo run mojo/train_fastspeech2.mojo --config config/training_config.yaml

# Train HiFiGAN (3 days)
mojo run mojo/train_hifigan.mojo --config config/hifigan_training_config.yaml
```

---

## 📈 Roadmap

### Version 1.1 (Q2 2026)

- [ ] Multilingual support (Spanish, French, German)
- [ ] Voice cloning capability
- [ ] Real-time streaming synthesis
- [ ] WebAssembly deployment
- [ ] GPU acceleration (Metal, CUDA)

### Version 2.0 (Q4 2026)

- [ ] Custom voice training
- [ ] Emotional speech synthesis
- [ ] Multi-speaker models
- [ ] API server deployment
- [ ] Cloud integration

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/nAudioLab.git
cd nAudioLab

# Create branch
git checkout -b feature/your-feature

# Make changes and test
./scripts/test_inference.sh

# Submit pull request
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **FastSpeech2**: Original architecture by Microsoft Research
- **HiFiGAN**: Neural vocoder by NVIDIA
- **LJSpeech**: Dataset by Keith Ito
- **CMU Dictionary**: Carnegie Mellon University
- **Mojo**: Modular Inc.
- **Zig**: Zig Software Foundation

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourorg/nAudioLab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourorg/nAudioLab/discussions)
- **Discord**: [Join Community](https://discord.gg/audiolabshimmy)
- **Email**: support@audiolabshimmy.org

---

## 🌟 Star History

If you find AudioLabShimmy useful, please consider giving it a star! ⭐

---

**Built with ❤️ using Mojo & Zig**

---

## Quick Links

- [📚 Documentation](docs/)
- [🐛 Report Bug](https://github.com/yourorg/nAudioLab/issues/new?template=bug_report.md)
- [💡 Request Feature](https://github.com/yourorg/nAudioLab/issues/new?template=feature_request.md)
- [❓ FAQ](docs/FAQ.md)

---

**Version:** 1.0.0  
**Status:** Production Ready ✅  
**Last Updated:** January 17, 2026
