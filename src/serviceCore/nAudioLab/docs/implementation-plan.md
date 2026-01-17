# AudioLabShimmy: 40-Day Implementation Plan

**Dolby-Quality TTS from Scratch in Mojo/Zig**

---

## 📋 Overview

**Duration:** 40 working days (8 weeks)  
**Estimated LOC:** ~8,500 lines  
**Technologies:** Mojo (inference + training), Zig (audio I/O)  
**Hardware:** CPU-only (Apple Silicon optimized)  
**Audio Quality:** 48kHz/24-bit with Dolby processing

---

## 🎯 Project Goals

Build a production TTS system with:
- ✅ Studio-quality audio output (48kHz, 24-bit)
- ✅ CPU-only training and inference
- ✅ Dolby loudness & dynamics processing
- ✅ FastSpeech2 + HiFiGAN architecture
- ✅ 100% Mojo/Zig implementation
- ✅ No external API dependencies

---

## 📅 Week-by-Week Breakdown

### **Week 1: Audio Foundation (Days 1-5)**

#### **Day 1: Audio Data Structures & I/O (Zig)**
**Goal:** Professional audio handling infrastructure

**Files to Create:**
- `zig/audio_types.zig` (100 lines)
- `zig/audio_io.zig` (200 lines)
- `zig/wav_format.zig` (150 lines)

**Implementation:**
```zig
// audio_types.zig
pub const AudioBuffer = struct {
    samples: []f32,
    sample_rate: u32 = 48000,
    channels: u8 = 2,
    bit_depth: u8 = 24,
};

// audio_io.zig
pub fn readWAV(path: []const u8, allocator: std.mem.Allocator) !AudioBuffer {
    // Parse WAV header
    // Read PCM data
    // Convert to f32 [-1.0, 1.0]
}

pub fn writeWAV(buffer: AudioBuffer, path: []const u8) !void {
    // Write WAV header (48kHz, 24-bit, stereo)
    // Write PCM samples
}

pub fn writeMP3(buffer: AudioBuffer, path: []const u8, bitrate: u32) !void {
    // Use libmp3lame FFI
    // Encode to MP3 (320kbps for quality)
}
```

**Tasks:**
- [ ] WAV file format parsing (RIFF chunks)
- [ ] 24-bit PCM handling
- [ ] Stereo channel interleaving
- [ ] libmp3lame FFI bindings
- [ ] Test with sample audio files

**Deliverable:** Read/write 48kHz/24-bit audio files

---

#### **Day 2: Mel-Spectrogram Extraction (Mojo)**
**Goal:** High-resolution audio features

**Files to Create:**
- `mojo/audio/types.mojo` (150 lines)
- `mojo/audio/fft.mojo` (300 lines)
- `mojo/audio/mel_features.mojo` (250 lines)

**Implementation:**
```mojo
from tensor import Tensor
from complex import ComplexSIMD

struct MelSpectrogram:
    var data: Tensor[DType.float32]  # [time, 128]
    var sample_rate: Int = 48000
    var n_fft: Int = 2048
    var hop_length: Int = 512
    var n_mels: Int = 128

fn stft(audio: AudioBuffer, n_fft: Int, hop_length: Int) -> Tensor[DType.complex64]:
    # Apply Hann window
    # Compute FFT (2048-point for 48kHz)
    # Return complex spectrogram [time, freq]

fn create_mel_filterbank(n_mels: Int, n_fft: Int, sample_rate: Int) -> Tensor[DType.float32]:
    # Create mel-scale filter bank
    # Triangular filters
    # 128 filters for high resolution

fn mel_spectrogram(audio: AudioBuffer) -> MelSpectrogram:
    # STFT
    # Apply mel filterbank
    # Log scaling
    # Return [time, 128]
```

**Tasks:**
- [ ] FFT implementation (use FFTW via FFI or pure Mojo)
- [ ] Hann window function
- [ ] Mel filterbank (128 bins)
- [ ] Log scaling
- [ ] Test on LJSpeech samples

**Deliverable:** Extract 128-bin mel-spectrograms at 48kHz

---

#### **Day 3: F0 & Prosody Extraction (Mojo)**
**Goal:** Pitch and energy features for expressive speech

**Files to Create:**
- `mojo/audio/f0_extractor.mojo` (350 lines)
- `mojo/audio/prosody.mojo` (200 lines)

**Implementation:**
```mojo
struct ProsodyFeatures:
    var f0: Tensor[DType.float32]          # Pitch contour
    var energy: Tensor[DType.float32]      # Frame energy
    var voiced: Tensor[DType.bool]         # Voice/unvoiced
    var duration: Tensor[DType.float32]    # Phoneme durations

fn extract_f0_yin(audio: AudioBuffer) -> Tensor[DType.float32]:
    # YIN algorithm for F0 extraction
    # More accurate than autocorrelation
    # Handles pitch doubling/halving
    # Return F0 contour in Hz

fn extract_energy(mel_spec: MelSpectrogram) -> Tensor[DType.float32]:
    # Frame-level energy
    # RMS or peak
    
fn detect_voiced_regions(audio: AudioBuffer, f0: Tensor) -> Tensor[DType.bool]:
    # Voiced/unvoiced detection
    # Zero-crossing rate
    # Energy threshold
```

**Tasks:**
- [ ] YIN algorithm implementation (300 lines)
- [ ] Energy calculation
- [ ] Voiced/unvoiced detection
- [ ] Smooth F0 contours
- [ ] Test on speech samples

**Deliverable:** Extract F0, energy, V/UV features

---

#### **Day 4: Text Normalization (Mojo)**
**Goal:** Convert any text to pronounceable form

**Files to Create:**
- `mojo/text/normalizer.mojo` (400 lines)
- `mojo/text/number_expander.mojo` (250 lines)

**Implementation:**
```mojo
fn normalize_text(text: String) -> String:
    # Lowercase
    # Expand numbers
    # Expand dates
    # Expand currency
    # Expand abbreviations
    # Remove special characters
    # Handle URLs/emails

fn expand_number(num_str: String) -> String:
    # 42 → "forty two"
    # 1,234 → "one thousand two hundred thirty four"
    # 3.14 → "three point one four"
    # Handle negatives, decimals, ordinals

fn expand_date(date_str: String) -> String:
    # 1/16/2026 → "January sixteenth, twenty twenty six"
    # Handle various formats

fn expand_currency(amount: String) -> String:
    # $10.50 → "ten dollars and fifty cents"

struct AbbreviationDict:
    var mappings: Dict[String, String]
    # Dr. → Doctor
    # St. → Street
    # etc. → et cetera
```

**Tasks:**
- [ ] Number-to-text (cardinal, ordinal)
- [ ] Date expansion
- [ ] Currency expansion
- [ ] Abbreviation dictionary (500+ entries)
- [ ] Test with complex sentences

**Deliverable:** Robust text normalization

---

#### **Day 5: Phoneme System (Mojo)**
**Goal:** English phoneme representation

**Files to Create:**
- `mojo/text/phoneme.mojo` (200 lines)
- `mojo/text/cmu_dict.mojo` (150 lines)
- `data/phonemes/cmudict.txt` (download 11MB file)

**Implementation:**
```mojo
struct Phoneme:
    var symbol: String      # e.g., "AH0", "K", "AE1"
    var ipa: String        # IPA representation
    var features: PhonemeFeatures

struct PhonemeFeatures:
    var voicing: Bool      # Voiced or unvoiced
    var place: String      # Articulatory place
    var manner: String     # Articulatory manner
    var vowel: Bool        # Vowel or consonant

struct CMUDict:
    var entries: Dict[String, List[Phoneme]]
    
    fn load(path: String) raises:
        # Parse CMU Pronouncing Dictionary
        # 134,000+ entries
    
    fn lookup(word: String) -> List[Phoneme]:
        # Dictionary lookup
        # Handle multiple pronunciations

# ARPAbet to features mapping
const PHONEME_FEATURES = {
    "AH0": PhonemeFeatures(voicing=True, place="central", manner="vowel", vowel=True),
    "K": PhonemeFeatures(voicing=False, place="velar", manner="stop", vowel=False),
    # ... 39 phonemes total
}
```

**Tasks:**
- [ ] Download CMU dict
- [ ] Parse CMU dict format
- [ ] Phoneme feature definitions
- [ ] Word lookup function
- [ ] Test pronunciation lookups

**Deliverable:** CMU dictionary loaded, phoneme system ready

---

### **Week 2: Neural Architecture (Days 6-10)**

#### **Day 6: Transformer Building Blocks (Mojo)**
**Goal:** Reusable attention mechanisms

**Files to Create:**
- `mojo/models/attention.mojo` (400 lines)
- `mojo/models/feed_forward.mojo` (200 lines)
- `mojo/models/layer_norm.mojo` (100 lines)

**Implementation:**
```mojo
struct MultiHeadAttention:
    var n_heads: Int = 4
    var d_model: Int = 256
    var d_k: Int = 64  # d_model / n_heads
    
    var W_q: Tensor[DType.float32]  # Query weights
    var W_k: Tensor[DType.float32]  # Key weights
    var W_v: Tensor[DType.float32]  # Value weights
    var W_o: Tensor[DType.float32]  # Output weights
    
    fn forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Q = x @ W_q, K = x @ W_k, V = x @ W_v
        # Split into heads
        # Scaled dot-product attention
        # Concat heads
        # Output projection
        # Return [batch, seq_len, d_model]

struct FeedForwardNetwork:
    var d_model: Int = 256
    var d_ff: Int = 1024
    
    var W1: Tensor[DType.float32]
    var b1: Tensor[DType.float32]
    var W2: Tensor[DType.float32]
    var b2: Tensor[DType.float32]
    
    fn forward(self, x: Tensor) -> Tensor:
        # x → Linear → ReLU → Linear
        # Expand to d_ff, then back to d_model

struct LayerNorm:
    var eps: Float32 = 1e-5
    var gamma: Tensor[DType.float32]
    var beta: Tensor[DType.float32]
    
    fn forward(self, x: Tensor) -> Tensor:
        # Normalize: (x - mean) / sqrt(var + eps)
        # Scale: gamma * norm + beta
```

**Tasks:**
- [ ] Multi-head attention with masking
- [ ] Feed-forward networks
- [ ] Layer normalization
- [ ] Positional encoding
- [ ] Test attention patterns

**Deliverable:** Transformer components ready

---

#### **Day 7: FastSpeech2 Encoder (Mojo)**
**Goal:** Text encoding with FFT blocks

**Files to Create:**
- `mojo/models/fft_block.mojo` (300 lines)
- `mojo/models/fastspeech2_encoder.mojo` (350 lines)

**Implementation:**
```mojo
struct FFTBlock:
    # Feed-Forward Transformer Block
    var self_attention: MultiHeadAttention
    var conv1: Conv1D
    var conv2: Conv1D
    var layer_norm1: LayerNorm
    var layer_norm2: LayerNorm
    
    fn forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # Self-attention + residual
        # Conv blocks + residual
        # Layer norm after each

struct FastSpeech2Encoder:
    var phoneme_embedding: Embedding  # 70 phonemes → 256 dim
    var pos_encoding: PositionalEncoding
    var layers: List[FFTBlock]  # 4 layers
    var layer_norm: LayerNorm
    
    fn forward(self, phonemes: Tensor[DType.int32]) -> Tensor[DType.float32]:
        # phonemes: [batch, seq_len]
        # Embed phonemes
        # Add positional encoding
        # Pass through FFT blocks
        # Return: [batch, seq_len, 256]
```

**Tasks:**
- [ ] 1D convolution layers
- [ ] FFT block with residuals
- [ ] Phoneme embedding (70 symbols)
- [ ] Positional encoding
- [ ] Test encoder output shapes

**Deliverable:** Text encoder working

---

#### **Day 8: Variance Adaptors (Mojo)**
**Goal:** Duration, pitch, energy prediction

**Files to Create:**
- `mojo/models/duration_predictor.mojo` (250 lines)
- `mojo/models/pitch_predictor.mojo` (250 lines)
- `mojo/models/energy_predictor.mojo` (200 lines)
- `mojo/models/length_regulator.mojo` (150 lines)

**Implementation:**
```mojo
struct DurationPredictor:
    var conv_layers: List[Conv1D]  # 2 layers
    var linear: Linear
    var dropout: Dropout
    
    fn forward(self, encoder_output: Tensor) -> Tensor[DType.float32]:
        # encoder_output: [batch, seq_len, 256]
        # Conv → ReLU → LayerNorm → Dropout
        # Linear projection
        # Return durations: [batch, seq_len]

struct PitchPredictor:
    # Same architecture as duration
    # Predicts log F0 values

struct EnergyPredictor:
    # Same architecture
    # Predicts frame energy

struct LengthRegulator:
    fn regulate_length(
        self,
        encoder_output: Tensor,
        durations: Tensor
    ) -> Tensor:
        # Expand encoder output based on predicted durations
        # phoneme_seq [N] + durations [N] → mel_seq [M]
        # M = sum(durations)
        # Repeat each phoneme 'duration' times
```

**Tasks:**
- [ ] Conv1D with dilation
- [ ] Duration prediction head
- [ ] Pitch prediction head  
- [ ] Energy prediction head
- [ ] Length regulation algorithm
- [ ] Test upsampling

**Deliverable:** Variance adaptors functional

---

#### **Day 9: FastSpeech2 Decoder (Mojo)**
**Goal:** Mel-spectrogram generation

**Files to Create:**
- `mojo/models/fastspeech2_decoder.mojo` (300 lines)
- `mojo/models/fastspeech2.mojo` (400 lines)

**Implementation:**
```mojo
struct FastSpeech2Decoder:
    var layers: List[FFTBlock]  # 4 layers
    var mel_linear: Linear      # 256 → 128 mel bins
    
    fn forward(self, x: Tensor) -> Tensor[DType.float32]:
        # x: [batch, mel_len, 256] (upsampled by length regulator)
        # Pass through FFT blocks
        # Project to mel-space
        # Return: [batch, mel_len, 128]

struct FastSpeech2:
    var encoder: FastSpeech2Encoder
    var variance_adaptor: VarianceAdaptor
    var decoder: FastSpeech2Decoder
    
    fn forward(
        self,
        phonemes: Tensor,
        target_durations: Optional[Tensor] = None,
        target_pitch: Optional[Tensor] = None,
        target_energy: Optional[Tensor] = None
    ) -> TTSOutput:
        # Encode phonemes
        # Predict variances (or use targets in training)
        # Regulate length
        # Decode to mel-spectrogram
        # Return mel + predictions

struct TTSOutput:
    var mel: Tensor[DType.float32]
    var pred_duration: Tensor[DType.float32]
    var pred_pitch: Tensor[DType.float32]
    var pred_energy: Tensor[DType.float32]
```

**Tasks:**
- [ ] Decoder FFT blocks
- [ ] Mel projection layer (256 → 128)
- [ ] Connect encoder + variance + decoder
- [ ] Forward pass logic
- [ ] Test output shapes

**Deliverable:** Complete FastSpeech2 model

---

#### **Day 10: HiFiGAN Generator (Mojo)**
**Goal:** Neural vocoder architecture (Part 1)

**Files to Create:**
- `mojo/models/hifigan_generator.mojo` (450 lines)
- `mojo/models/hifigan_blocks.mojo` (300 lines)

**Implementation:**
```mojo
struct HiFiGANGenerator:
    var upsample_layers: List[ConvTranspose1D]
    var resblocks: List[MultiReceptiveField ResBlock]
    
    fn forward(self, mel: Tensor) -> Tensor[DType.float32]:
        # mel: [batch, 128, time]
        # Upsample: 128 → 512 → 1024 → ... → 48000 Hz
        # Apply MRF resblocks at each scale
        # Return: [batch, 1, audio_len]

struct MRFResBlock:
    # Multi-Receptive Field Residual Block
    var conv_blocks: List[ConvBlock]  # Different kernel sizes
    
    fn forward(self, x: Tensor) -> Tensor:
        # Parallel conv with kernels: [3, 7, 11]
        # Sum outputs
        # Residual connection

struct ConvTranspose1D:
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var weights: Tensor[DType.float32]
    
    fn forward(self, x: Tensor) -> Tensor:
        # Transposed convolution for upsampling
```

**Tasks:**
- [ ] Transposed 1D convolutions
- [ ] Multi-receptive field resblocks
- [ ] Upsampling stages (128 mel → 48kHz)
- [ ] Test generator output shape
- [ ] Verify audio waveform range [-1, 1]

**Deliverable:** HiFiGAN generator (vocoder part 1)

---

### **Week 2: Training Infrastructure (Days 11-15)**

#### **Day 11: HiFiGAN Discriminators (Mojo)**
**Goal:** Adversarial training components

**Files to Create:**
- `mojo/models/hifigan_discriminator.mojo` (500 lines)

**Implementation:**
```mojo
struct MultiPeriodDiscriminator:
    # Analyze audio at multiple periods
    var periods: List[Int] = [2, 3, 5, 7, 11]
    var sub_discriminators: List[PeriodDiscriminator]
    
    fn forward(self, audio: Tensor) -> List[Tensor]:
        # Reshape audio by each period
        # Run through each discriminator
        # Return feature maps

struct MultiScaleDiscriminator:
    # Analyze audio at multiple scales
    var scales: Int = 3
    var discriminators: List[ScaleDiscriminator]
    
    fn forward(self, audio: Tensor) -> List[Tensor]:
        # Downsample audio (avgpool)
        # Run through each scale discriminator
        # Return feature maps

struct PeriodDiscriminator:
    var conv_layers: List[Conv2D]
    
    fn forward(self, x: Tensor) -> (Tensor, List[Tensor]):
        # Conv layers with LeakyReLU
        # Return logits + intermediate features
```

**Tasks:**
- [ ] Period discriminator (2D conv)
- [ ] Scale discriminator (1D conv)
- [ ] Multi-period wrapper
- [ ] Multi-scale wrapper
- [ ] Test discriminator outputs

**Deliverable:** Complete HiFiGAN architecture

---

#### **Day 12: Loss Functions (Mojo)**
**Goal:** Training objectives

**Files to Create:**
- `mojo/training/losses.mojo` (450 lines)

**Implementation:**
```mojo
fn fastspeech2_loss(
    pred: TTSOutput,
    target_mel: Tensor,
    target_duration: Tensor,
    target_pitch: Tensor,
    target_energy: Tensor
) -> Tuple[Float32, Dict[String, Float32]]:
    # Mel loss (L1): primary objective
    mel_loss = l1_loss(pred.mel, target_mel)
    
    # Duration loss (MSE): alignment
    dur_loss = mse_loss(pred.pred_duration, target_duration)
    
    # Pitch loss (MSE): intonation
    pitch_loss = mse_loss(pred.pred_pitch, target_pitch)
    
    # Energy loss (MSE): dynamics
    energy_loss = mse_loss(pred.pred_energy, target_energy)
    
    # Weighted sum
    total = mel_loss + 0.1 * dur_loss + 0.1 * pitch_loss + 0.1 * energy_loss
    
    return total, {"mel": mel_loss, "dur": dur_loss, ...}

fn hifigan_generator_loss(
    pred_audio: Tensor,
    target_audio: Tensor,
    disc_fake_outputs: List[Tensor]
) -> Tuple[Float32, Dict]:
    # Multi-resolution STFT loss
    stft_loss = multi_resolution_stft_loss(pred_audio, target_audio)
    
    # Adversarial loss (fool discriminators)
    adv_loss = 0.0
    for fake_logits in disc_fake_outputs:
        adv_loss += mse_loss(fake_logits, ones_like(fake_logits))
    
    # Feature matching loss
    fm_loss = feature_matching_loss(disc_fake_features, disc_real_features)
    
    total = stft_loss + adv_loss + 2.0 * fm_loss
    return total, losses_dict

fn hifigan_discriminator_loss(
    disc_real_outputs: List[Tensor],
    disc_fake_outputs: List[Tensor]
) -> Float32:
    # Real should be 1, fake should be 0
    real_loss = mse_loss(disc_real_outputs, ones)
    fake_loss = mse_loss(disc_fake_outputs, zeros)
    return real_loss + fake_loss

fn multi_resolution_stft_loss(pred: Tensor, target: Tensor) -> Float32:
    # Compute STFT at multiple resolutions
    # FFT sizes: [512, 1024, 2048]
    # Hop lengths: [128, 256, 512]
    # L1 loss on magnitude + phase
```

**Tasks:**
- [ ] FastSpeech2 losses
- [ ] HiFiGAN generator losses
- [ ] HiFiGAN discriminator losses
- [ ] Multi-resolution STFT
- [ ] Feature matching loss
- [ ] Test loss computation

**Deliverable:** All loss functions implemented

---

#### **Day 13: Dataset Loader (Mojo)**
**Goal:** LJSpeech data pipeline

**Files to Create:**
- `mojo/training/dataset.mojo` (500 lines)
- `mojo/training/preprocessor.mojo` (350 lines)

**Implementation:**
```mojo
struct LJSpeechDataset:
    var audio_paths: List[String]     # 13,100 clips
    var transcripts: List[String]     # Text transcriptions
    var phonemes: List[List[Phoneme]] # Pre-computed
    var mels: List[MelSpectrogram]    # Pre-computed
    var f0: List[Tensor]              # Pre-computed
    var energy: List[Tensor]          # Pre-computed
    var durations: List[Tensor]       # Aligned durations
    
    fn load(data_dir: String) raises:
        # Read metadata.csv
        # Load all transcriptions
        # Convert text → phonemes
        # Load pre-processed features

fn preprocess_dataset(data_dir: String, output_dir: String):
    # For each audio file:
    #   1. Load 48kHz audio
    #   2. Extract mel-spectrogram (128 bins)
    #   3. Extract F0 (YIN)
    #   4. Extract energy
    #   5. Align phonemes to audio (MFA)
    #   6. Save preprocessed features
    # This runs once, saves 50GB preprocessed data

struct DataLoader:
    var dataset: LJSpeechDataset
    var batch_size: Int = 16  # CPU-friendly
    
    fn get_batch(self, idx: Int) -> TTSBatch:
        # Collate samples
        # Pad sequences to max length
        # Return batch tensors

struct TTSBatch:
    var phonemes: Tensor[DType.int32]      # [batch, max_pho_len]
    var mels: Tensor[DType.float32]        # [batch, max_mel_len, 128]
    var durations: Tensor[DType.float32]   # [batch, max_pho_len]
    var pitch: Tensor[DType.float32]       # [batch, max_mel_len]
    var energy: Tensor[DType.float32]      # [batch, max_mel_len]
    var pho_lengths: Tensor[DType.int32]   # [batch]
    var mel_lengths: Tensor[DType.int32]   # [batch]
```

**Tasks:**
- [ ] Metadata CSV parser
- [ ] Audio file loading
- [ ] Feature extraction for all 13k clips
- [ ] Phoneme alignment (Montreal Forced Aligner)
- [ ] Batch collation with padding
- [ ] Test data loading speed

**Deliverable:** LJSpeech dataset ready for training

---

#### **Day 14: CPU-Optimized Training (Mojo)**
**Goal:** Fast training on Apple Silicon

**Files to Create:**
- `mojo/training/cpu_optimizer.mojo` (300 lines)
- `mojo/training/accelerate_bindings.mojo` (200 lines)

**Implementation:**
```mojo
from sys import simd_width

struct CPUOptimizedAdam:
    var learning_rate: Float32 = 1e-4
    var beta1: Float32 = 0.9
    var beta2: Float32 = 0.999
    var eps: Float32 = 1e-8
    
    var m: Dict[String, Tensor]  # First moment
    var v: Dict[String, Tensor]  # Second moment
    var t: Int = 0               # Time step
    
    fn step(inout self, params: Dict[String, Tensor], grads: Dict):
        # Vectorized Adam update
        # Use SIMD for speed
        # Update all parameters

@always_inline
fn matmul_accelerate(a: Tensor, b: Tensor) -> Tensor:
    # Use Apple Accelerate framework
    # cblas_sgemm for matrix multiply
    # Much faster than naive loops

fn parallelize_training(dataset: DataLoader, model: FastSpeech2, num_threads: Int):
    # Split batches across CPU cores
    # Gradient accumulation
    # Parallel forward passes
```

**Tasks:**
- [ ] Accelerate framework FFI bindings
- [ ] SIMD vectorization for ops
- [ ] Multi-threaded data loading
- [ ] Gradient accumulation (effective batch 32)
- [ ] Mixed precision (FP16/FP32)
- [ ] Benchmark training speed

**Deliverable:** Optimized training infrastructure

---

#### **Day 15: Training Script (Mojo)**
**Goal:** Complete training loop

**Files to Create:**
- `mojo/training/trainer.mojo` (400 lines)
- `mojo/train_fastspeech2.mojo` (300 lines)

**Implementation:**
```mojo
struct FastSpeech2Trainer:
    var model: FastSpeech2
    var optimizer: CPUOptimizedAdam
    var scheduler: LRScheduler
    var dataset: DataLoader
    
    fn train_epoch(inout self) -> Dict[String, Float32]:
        var total_loss = 0.0
        
        for batch_idx in range(len(self.dataset)):
            var batch = self.dataset.get_batch(batch_idx)
            
            # Forward pass
            var output = self.model.forward(
                batch.phonemes,
                batch.durations,
                batch.pitch,
                batch.energy
            )
            
            # Compute loss
            var loss, loss_dict = fastspeech2_loss(
                output, batch.mels,
                batch.durations, batch.pitch, batch.energy
            )
            
            # Backward pass
            var grads = loss.backward()
            
            # Update weights
            self.optimizer.step(self.model.parameters(), grads)
            
            total_loss += loss
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss:.4f}")
        
        return {"loss": total_loss / len(self.dataset)}
    
    fn validate(self) -> Float32:
        # Validation loop
        # No gradient computation

fn main():
    # Load dataset
    var dataset = LJSpeechDataset.load("data/datasets/ljspeech")
    
    # Initialize model
    var model = FastSpeech2()
    
    # Initialize trainer
    var trainer = FastSpeech2Trainer(model, dataset)
    
    # Training loop
    for epoch in range(200):  # ~200k steps
        var metrics = trainer.train_epoch()
        print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            model.save(f"data/models/fastspeech2/checkpoint_{epoch}.mojo")
```

**Tasks:**
- [ ] Training loop with metrics
- [ ] Validation loop
- [ ] Checkpoint saving/loading
- [ ] Learning rate scheduling
- [ ] Progress logging
- [ ] Test training on small dataset

**Deliverable:** FastSpeech2 training ready

---

### **Week 3-5: Model Training (Days 16-30)**

#### **Days 16-18: Dataset Preprocessing**
**Goal:** Prepare all 13,100 LJSpeech samples

**Tasks:**
- [ ] Download LJSpeech (2.6GB)
- [ ] Extract audio files
- [ ] Convert all to 48kHz stereo
- [ ] Extract mel-spectrograms (13k × 128 bins)
- [ ] Extract F0 contours (YIN algorithm)
- [ ] Extract energy values
- [ ] Run Montreal Forced Aligner for durations
- [ ] Save preprocessed features (~50GB)

**Script:** `scripts/preprocess_all.sh`
```bash
#!/bin/bash
# Download dataset
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

# Preprocess all
mojo run mojo/training/preprocessor.mojo \
    --input LJSpeech-1.1 \
    --output data/datasets/ljspeech_processed \
    --sample-rate 48000 \
    --n-mels 128
```

**Deliverable:** 13k preprocessed training samples

---

#### **Days 19-26: Train FastSpeech2 (8 days)**
**Goal:** Train acoustic model on CPU

**Configuration:**
```yaml
# training_config.yaml
model:
  encoder_layers: 4
  decoder_layers: 4
  d_model: 256
  n_heads: 4
  dropout: 0.1

training:
  batch_size: 16          # CPU-friendly
  gradient_accumulation: 2  # Effective batch 32
  learning_rate: 1e-4
  warmup_steps: 4000
  max_steps: 200000       # ~8 days on M3 Max
  save_every: 10000
  validate_every: 5000

optimization:
  use_accelerate: true
  num_threads: 16
  mixed_precision: true   # FP16/FP32
```

**Training Command:**
```bash
mojo run mojo/train_fastspeech2.mojo \
    --config training_config.yaml \
    --data data/datasets/ljspeech_processed \
    --output data/models/fastspeech2
```

**Expected Progress:**
- Day 19: Steps 0-25k
- Day 20: Steps 25k-50k
- Day 21: Steps 50k-75k
- Day 22: Steps 75k-100k
- Day 23: Steps 100k-125k
- Day 24: Steps 125k-150k
- Day 25: Steps 150k-175k
- Day 26: Steps 175k-200k

**Deliverable:** Trained FastSpeech2 model

---

#### **Day 27: HiFiGAN Training Script**
**Goal:** Set up vocoder training

**Files to Create:**
- `mojo/train_hifigan.mojo` (350 lines)

**Implementation:**
```mojo
struct HiFiGANTrainer:
    var generator: HiFiGANGenerator
    var mpd: MultiPeriodDiscriminator
    var msd: MultiScaleDiscriminator
    var optim_g: Adam
    var optim_d: Adam
    
    fn train_step(inout self, batch: AudioBatch):
        # 1. Train Discriminators
        real_audio = batch.audio
        pred_audio = self.generator(batch.mel)
        
        # Discriminator forward
        real_mpd, real_mpd_feats = self.mpd(real_audio)
        fake_mpd, fake_mpd_feats = self.mpd(pred_audio.detach())
        
        # Discriminator loss
        d_loss = discriminator_loss(real_mpd, fake_mpd)
        d_loss.backward()
        self.optim_d.step()
        
        # 2. Train Generator
        fake_mpd_g, fake_mpd_feats_g = self.mpd(pred_audio)
        
        # Generator loss
        g_loss = generator_loss(pred_audio, real_audio, fake_mpd_g, ...)
        g_loss.backward()
        self.optim_g.step()
```

**Tasks:**
- [ ] GAN training logic
- [ ] Alternating G/D updates
- [ ] Loss balancing
- [ ] Gradient clipping
- [ ] Test GAN training

**Deliverable:** HiFiGAN training script

---

#### **Days 28-30: Train HiFiGAN (3 days initial)**
**Goal:** Start vocoder training

**Training:** 500k steps (will continue to Day 35)

---

### **Week 4-6: Continued Training (Days 31-40)**

#### **Days 31-35: Complete HiFiGAN Training**
Continue vocoder training to 500k steps

---

#### **Day 36: Dolby Audio Processing (Zig)**
**Goal:** Professional post-processing

**Files to Create:**
- `zig/dolby_processor.zig` (600 lines)

**Implementation:**
```zig
const DolbyConfig = struct {
    target_lufs: f32 = -16.0,
    compression_ratio: f32 = 3.0,
    attack_ms: f32 = 5.0,
    release_ms: f32 = 50.0,
    enhancer_amount: f32 = 0.3,
};

pub fn processDolby(samples: []f32, config: DolbyConfig) !void {
    // 1. Measure loudness (ITU-R BS.1770-4)
    const lufs = measureLUFS(samples);
    const gain = config.target_lufs - lufs;
    applyGain(samples, gain);
    
    // 2. Multi-band compression
    try multibandCompress(samples, config);
    
    // 3. Harmonic enhancement
    try harmonicExciter(samples, config.enhancer_amount);
    
    // 4. Stereo widening (subtle)
    try stereoWiden(samples, 1.2);
    
    // 5. Final limiting
    try brickWallLimit(samples, -0.3);
}

fn measureLUFS(samples: []const f32) f32 {
    // K-weighting filter
    // Gating at -70 LUFS
    // Calculate integrated loudness
}

fn multibandCompress(samples: []f32, config: DolbyConfig) !void {
    // Split into 5 bands: [0-100, 100-500, 500-2k, 2k-8k, 8k-24k]
    // Apply compression per band
    // Recombine
}
```

**Tasks:**
- [ ] LUFS metering (ITU-R BS.1770-4)
- [ ] 5-band compressor with crossover filters
- [ ] Harmonic exciter
- [ ] Stereo widening (Haas effect)
- [ ] Brick-wall limiter (lookahead)
- [ ] Test on speech samples

**Deliverable:** Dolby-grade audio processing

---

#### **Day 37: Inference Engine (Mojo)**
**Goal:** Production TTS pipeline

**Files to Create:**
- `mojo/inference/engine.mojo` (350 lines)
- `mojo/inference/pipeline.mojo` (200 lines)

**Implementation:**
```mojo
struct TTSEngine:
    var fastspeech2: FastSpeech2
    var hifigan: HiFiGANGenerator
    var phonemizer: Phonemizer
    var normalizer: TextNormalizer
    
    fn load(model_dir: String) -> TTSEngine:
        # Load FastSpeech2 checkpoint
        # Load HiFiGAN checkpoint
        # Initialize phonemizer
        # Return engine
    
    fn synthesize(
        self,
        text: String,
        speed: Float32 = 1.0,
        pitch_shift: Float32 = 0.0
    ) -> AudioBuffer:
        # 1. Normalize text
        normalized = self.normalizer.normalize(text)
        
        # 2. Text → Phonemes
        phonemes = self.phonemizer.text_to_phonemes(normalized)
        
        # 3. Phonemes → Mel
        mel = self.fastspeech2.infer(phonemes, speed, pitch_shift)
        
        # 4. Mel → Audio (48kHz)
        audio = self.hifigan.generate(mel)
        
        # 5. Dolby processing (via FFI to Zig)
        audio_dolby = apply_dolby_processing_ffi(audio)
        
        return audio_dolby
```

**Tasks:**
- [ ] Model loading from checkpoints
- [ ] Inference pipeline (no gradients)
- [ ] Speed/pitch control
- [ ] Memory-efficient inference
- [ ] Test with various texts

**Deliverable:** Complete TTS inference engine

---

#### **Day 38: Zig FFI Bridge**
**Goal:** Connect Mojo TTS to Zig audio processing

**Files to Create:**
- `zig/ffi_bridge.zig` (250 lines)
- `mojo/audio/zig_ffi.mojo` (150 lines)

**Implementation:**
```zig
// ffi_bridge.zig
export fn process_audio_dolby(
    samples_ptr: [*]f32,
    length: usize,
    sample_rate: u32
) callconv(.C) c_int {
    const samples = samples_ptr[0..length];
    const config = DolbyConfig{};
    processDolby(samples, config) catch return -1;
    return 0;
}

export fn encode_to_mp3(
    samples_ptr: [*]f32,
    length: usize,
    sample_rate: u32,
    output_path: [*:0]const u8
) callconv(.C) c_int {
    // MP3 encoding with libmp3lame
}
```

```mojo
// zig_ffi.mojo
@external("process_audio_dolby")
fn process_audio_dolby_external(
    samples: DTypePointer[DType.float32],
    length: Int,
    sample_rate: Int
) -> Int

fn apply_dolby_processing_ffi(audio: AudioBuffer) -> AudioBuffer:
    # Call Zig function
    var result = process_audio_dolby_external(
        audio.samples.address,
        audio.length,
        audio.sample_rate
    )
    return audio
```

**Tasks:**
- [ ] Export Zig functions with C calling convention
- [ ] Import in Mojo with @external
- [ ] Handle memory between Mojo/Zig
- [ ] Error handling
- [ ] Test FFI calls

**Deliverable:** Mojo ↔ Zig FFI working

---

#### **Day 39: Integration Testing**
**Goal:** End-to-end TTS testing

**Files to Create:**
- `tests/test_tts_pipeline.mojo` (200 lines)
- `tests/test_audio_quality.mojo` (150 lines)
- `scripts/test_inference.sh`

**Tests:**
```mojo
fn test_simple_sentence():
    var tts = TTSEngine.load("data/models")
    var audio = tts.synthesize("Hello world")
    assert audio.sample_rate == 48000
    assert audio.channels == 2
    assert audio.bit_depth == 24

fn test_long_text():
    var tts = TTSEngine.load("data/models")
    var text = read_file("test_data/long_text.txt")  # 5000 words
    var audio = tts.synthesize(text)
    assert audio.length > 0
    audio.save("test_output.mp3")

fn test_audio_quality():
    var audio = load_audio("test_output.mp3")
    
    # Check LUFS
    var lufs = measure_lufs(audio)
    assert lufs >= -17.0 and lufs <= -15.0
    
    # Check THD+N
    var thd = measure_thd(audio)
    assert thd < 0.01
```

**Tasks:**
- [ ] Unit tests for each component
- [ ] Integration tests for pipeline
- [ ] Audio quality validation
- [ ] Performance benchmarks
- [ ] Memory profiling

**Deliverable:** Comprehensive test suite

---

#### **Day 40: Documentation & Polish**
**Goal:** Production-ready system

**Files to Create:**
- `docs/api-reference.md`
- `docs/performance-tuning.md`
- `docs/troubleshooting.md`

**Tasks:**
- [ ] API documentation
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] Example scripts
- [ ] Quality validation report

**Deliverable:** Complete, documented TTS system

---

## 📊 Code Statistics (Target)

| Component | Lines of Code |
|-----------|---------------|
| Audio Processing (Mojo) | 1,200 |
| Text Processing (Mojo) | 1,000 |
| Neural Models (Mojo) | 3,500 |
| Training (Mojo) | 1,800 |
| Inference (Mojo) | 700 |
| Zig Audio I/O | 800 |
| Zig Dolby Processing | 600 |
| Tests | 500 |
| Scripts | 300 |
| **Total** | **~10,400** |

---

## 🎯 Milestones

### Milestone 1: Foundation (Day 5)
- ✅ Audio I/O working
- ✅ Feature extraction working
- ✅ Text processing working

### Milestone 2: Architecture (Day 15)
- ✅ FastSpeech2 model complete
- ✅ HiFiGAN model complete
- ✅ Training infrastructure ready

### Milestone 3: Training (Day 30)
- ✅ FastSpeech2 trained
- ✅ HiFiGAN trained
- ✅ Models generating audio

### Milestone 4: Quality (Day 40)
- ✅ Dolby processing applied
- ✅ Quality metrics validated
- ✅ Production ready

---

## 🚀 Next Steps

1. Review this implementation plan
2. Set up development environment
3. Start Day 1: Audio data structures
4. Follow day-by-day plan
5. Train models (Days 19-35)
6. Validate quality (Days 36-40)

---

**Last Updated:** January 16, 2026  
**Ready for:** Implementation phase
