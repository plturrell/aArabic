## Day 2: Mel-Spectrogram Extraction - Mojo Setup Required

**Date:** January 17, 2026  
**Focus:** High-resolution mel-spectrogram extraction in Mojo

---

## 🚨 Mojo Installation Required

Day 2 requires Mojo to be properly installed. Mojo is not currently available on this system.

### Installation Steps

1. **Visit the official Modular website:**
   ```
   https://developer.modular.com/download
   ```

2. **Sign up for a free Modular account**

3. **Download the installer for macOS (Apple Silicon)**

4. **Run the installer and follow prompts**

5. **Install Mojo:**
   ```bash
   modular install mojo
   ```

6. **Add Mojo to your PATH:**
   ```bash
   echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.zshrc
   echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

7. **Verify installation:**
   ```bash
   mojo --version
   ```

### Alternative: Use the installation script
```bash
bash scripts/install_mojo.sh
```

Then follow the manual steps provided by the script.

---

## 📁 Files Created (Ready for Mojo)

### Day 2 Mojo Modules

1. **`mojo/audio/types.mojo`** (~230 lines)
   - `AudioBuffer` struct for Mojo
   - `MelSpectrogram` struct
   - `STFTConfig` configuration
   - `MelFilterbankConfig` configuration
   - Hz ↔ Mel conversion functions
   - Hann window generation
   - Signal padding utilities

2. **`mojo/audio/fft.mojo`** (~235 lines)
   - `Complex` number struct
   - STFT computation
   - Power spectrum conversion
   - dB scale conversion
   - Frequency/time axis helpers
   - Python bridge for initial implementation
   - Notes for production optimization

3. **`mojo/audio/mel_features.mojo`** (~260 lines)
   - Mel filterbank creation
   - Mel filterbank application
   - `extract_mel_spectrogram()` main function
   - Griffin-Lim reconstruction (for validation)
   - Normalization functions
   - Statistics computation

### Supporting Files

4. **`mojoproject.toml`** - Mojo package configuration

5. **`scripts/install_mojo.sh`** - Installation helper

6. **`scripts/test_mel_extraction.py`** - Python validation script (works now)

---

## 🎯 Day 2 Objectives

Once Mojo is installed, the implementation will provide:

✅ 2048-point FFT for 48kHz audio  
✅ Hann window function  
✅ 128-bin mel filterbank  
✅ Log scaling (dB)  
✅ High-resolution spectrograms (~23Hz frequency resolution)  
✅ ~10.7ms time resolution  

---

## 🧪 Testing Without Mojo (Day 2 Validation)

While Mojo is being set up, you can validate the mel-spectrogram extraction approach using Python:

```bash
cd src/serviceCore/nAudioLab

# First, ensure Day 1 tests have run to generate audio files
cd test_output && ls *.wav

# Install required Python packages
pip install librosa matplotlib numpy scipy

# Run mel-spectrogram validation
python3 scripts/test_mel_extraction.py
```

This will:
- Load the WAV files from Day 1
- Extract 128-bin mel-spectrograms
- Generate visualizations
- Validate the approach matches our Mojo specs

---

## 📊 Technical Specifications

### STFT Configuration
```
FFT Size: 2048 points
Hop Length: 512 samples
Window: Hann
Sample Rate: 48kHz

Frequency Resolution: 48000 / 2048 = ~23.4 Hz per bin
Time Resolution: 512 / 48000 = ~10.7 ms per frame
```

### Mel Filterbank Configuration
```
Number of Bins: 128
Frequency Range: 0 - 24000 Hz (Nyquist for 48kHz)
Scale: Mel (HTK formula)
Norm: Slaney
Filter Shape: Triangular
```

### Output Format
```
Shape: [time_frames, 128]
Values: dB scale (log mel-spectrogram)
Dynamic Range: 80 dB
Reference: 1.0
Minimum: 1e-10
```

---

## 🔄 Hybrid Approach (Current State)

### What's Implemented:

1. **Mojo Interface Modules** (types, fft, mel_features)
   - Define the API and structure
   - Use Python bridges for complex operations (temp)
   - Ready for pure Mojo implementation when optimizing

2. **Python Validation Script**
   - Demonstrates exact algorithm
   - Generates visualizations
   - Validates numerical correctness
   - Can run immediately

### Migration Path:

```
Phase 1 (Day 2): Python bridge ← WE ARE HERE
  ↓
Phase 2 (Week 3-4): Pure Mojo with SIMD
  ↓
Phase 3 (Week 5+): Apple Accelerate integration
```

---

## 📝 Once Mojo is Installed

After installing Mojo, you can:

1. **Build the Mojo audio package:**
   ```bash
   cd src/serviceCore/nAudioLab
   mojo package mojo/audio -o libaudio.mojopkg
   ```

2. **Run Mojo tests:**
   ```bash
   mojo test mojo/test_mel_extraction.mojo
   ```

3. **Use in production:**
   ```mojo
   from audio.mel_features import extract_mel_spectrogram
   from audio.types import AudioBuffer
   
   # Load audio (via Zig FFI)
   let audio = load_audio_from_zig("input.wav")
   
   # Extract mel-spectrogram
   let mel = extract_mel_spectrogram(audio)
   
   # mel.data is [time, 128] tensor ready for FastSpeech2
   ```

---

## 🚀 Next Steps

### Immediate (Before Day 3):
1. Install Mojo using the guide above
2. Run `python3 scripts/test_mel_extraction.py` to validate approach
3. Verify mel-spectrogram extraction works correctly

### Day 3 Focus:
- F0 (pitch) extraction using YIN algorithm
- Energy feature extraction  
- Voiced/unvoiced detection
- Prosody feature integration

---

## 📈 Progress Status

**Day 1:** ✅ COMPLETE - Audio I/O in Zig  
**Day 2:** 🟡 PENDING - Awaiting Mojo installation  
**Day 3:** ⏳ READY - F0 & Prosody extraction planned

---

## 💡 Why This Approach?

The hybrid Python/Mojo approach allows us to:
1. ✅ Validate algorithms immediately
2. ✅ Define clean Mojo interfaces
3. ✅ Test numerical correctness
4. ⏳ Optimize to pure Mojo incrementally
5. ⏳ Add SIMD/Accelerate when ready

This is pragmatic engineering: working software first, optimization second.

---

**Status:** 🟡 BLOCKED ON MOJO INSTALLATION  
**Workaround:** Python validation script ready  
**Next:** Install Mojo, then complete pure Mojo implementation
