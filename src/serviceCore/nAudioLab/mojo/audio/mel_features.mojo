"""
Mel-Spectrogram Extraction for TTS
High-resolution 128-bin mel-spectrograms for 48kHz audio
"""

from tensor import Tensor
from python import Python
from .types import MelSpectrogram, MelFilterbankConfig, hz_to_mel, mel_to_hz
from .fft import compute_stft, magnitude_to_power, power_to_db


fn create_mel_filterbank(config: MelFilterbankConfig) raises -> Tensor[DType.float32]:
    """
    Create mel-scale filterbank
    
    Creates triangular filters on the mel scale that map linear frequency
    bins to mel bins.
    
    Args:
        config: Mel filterbank configuration
    
    Returns:
        Filterbank matrix [n_mels, freq_bins]
    """
    let freq_bins = config.mel_bins()
    
    # Use Python librosa for mel filterbank creation
    # Production version would implement pure Mojo
    let librosa = Python.import_module("librosa")
    let np = Python.import_module("numpy")
    
    let mel_fb = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        fmin=config.f_min,
        fmax=config.f_max,
        norm=config.norm
    )
    
    # Convert to Mojo tensor
    var filterbank = Tensor[DType.float32](config.n_mels, freq_bins)
    
    for m in range(config.n_mels):
        for f in range(freq_bins):
            filterbank[m * freq_bins + f] = Float32(mel_fb[m][f])
    
    return filterbank


fn apply_mel_filterbank(
    spectrogram: Tensor[DType.float32],
    filterbank: Tensor[DType.float32],
    n_mels: Int
) raises -> Tensor[DType.float32]:
    """
    Apply mel filterbank to power spectrogram
    
    Args:
        spectrogram: Power spectrogram [freq_bins, time_frames]
        filterbank: Mel filterbank [n_mels, freq_bins]
        n_mels: Number of mel bins
    
    Returns:
        Mel-spectrogram [n_mels, time_frames]
    """
    # For Day 2, use Python for matrix multiplication
    # Production would use BLAS or SIMD-optimized matmul
    let np = Python.import_module("numpy")
    
    # Convert tensors to numpy
    let spec_shape = spectrogram.shape()
    let freq_bins = Int(spec_shape[0])
    let time_frames = Int(spec_shape[1])
    
    var spec_list = Python.evaluate("[]")
    for f in range(freq_bins):
        var row = Python.evaluate("[]")
        for t in range(time_frames):
            _ = row.append(spectrogram[f * time_frames + t])
        _ = spec_list.append(row)
    let np_spec = np.array(spec_list)
    
    var fb_list = Python.evaluate("[]")
    for m in range(n_mels):
        var row = Python.evaluate("[]")
        for f in range(freq_bins):
            _ = row.append(filterbank[m * freq_bins + f])
        _ = fb_list.append(row)
    let np_fb = np.array(fb_list)
    
    # Matrix multiplication: [n_mels, freq_bins] @ [freq_bins, time_frames]
    let mel_spec_np = np.dot(np_fb, np_spec)
    
    # Convert back to Mojo
    var mel_spec = Tensor[DType.float32](n_mels, time_frames)
    for m in range(n_mels):
        for t in range(time_frames):
            mel_spec[m * time_frames + t] = Float32(mel_spec_np[m][t])
    
    return mel_spec


fn extract_mel_spectrogram(
    audio: Tensor[DType.float32],
    sample_rate: Int = 48000,
    n_fft: Int = 2048,
    hop_length: Int = 512,
    n_mels: Int = 128,
    f_min: Float32 = 0.0,
    f_max: Float32 = 24000.0
) raises -> MelSpectrogram:
    """
    Extract mel-spectrogram from audio signal
    
    Complete pipeline:
    1. Compute STFT
    2. Convert to power
    3. Apply mel filterbank
    4. Convert to log scale (dB)
    
    Args:
        audio: Input audio signal [samples]
        sample_rate: Sample rate in Hz (48000 for studio quality)
        n_fft: FFT size (2048 gives ~23Hz resolution at 48kHz)
        hop_length: Hop length in samples (512 gives ~10.7ms frames)
        n_mels: Number of mel bins (128 for high resolution)
        f_min: Minimum frequency
        f_max: Maximum frequency (Nyquist frequency for 48kHz)
    
    Returns:
        MelSpectrogram object with data [time, n_mels]
    """
    # Step 1: Compute STFT
    let magnitude_spec = compute_stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann"
    )
    
    # Step 2: Convert magnitude to power
    let power_spec = magnitude_to_power(magnitude_spec)
    
    # Step 3: Create and apply mel filterbank
    let mel_config = MelFilterbankConfig(
        n_mels=n_mels,
        n_fft=n_fft,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        norm="slaney"
    )
    
    let filterbank = create_mel_filterbank(mel_config)
    let mel_power = apply_mel_filterbank(power_spec, filterbank, n_mels)
    
    # Step 4: Convert to log scale (dB)
    let mel_db = power_to_db(mel_power, ref=1.0, amin=1e-10, top_db=80.0)
    
    # Transpose to [time, n_mels] format for TTS models
    let spec_shape = mel_db.shape()
    let n_mel_bins = Int(spec_shape[0])
    let time_frames = Int(spec_shape[1])
    
    var transposed = Tensor[DType.float32](time_frames, n_mel_bins)
    for t in range(time_frames):
        for m in range(n_mel_bins):
            transposed[t * n_mel_bins + m] = mel_db[m * time_frames + t]
    
    return MelSpectrogram(
        data=transposed,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )


fn mel_spectrogram_to_audio(
    mel_spec: MelSpectrogram,
    n_iter: Int = 32
) raises -> Tensor[DType.float32]:
    """
    Convert mel-spectrogram back to audio using Griffin-Lim algorithm
    
    Note: This is for testing/validation purposes.
    Production TTS uses neural vocoder (HiFiGAN) instead.
    
    Args:
        mel_spec: Mel-spectrogram to convert
        n_iter: Number of Griffin-Lim iterations
    
    Returns:
        Audio signal [samples]
    """
    let librosa = Python.import_module("librosa")
    let np = Python.import_module("numpy")
    
    # Convert mel data to numpy
    let time_frames = mel_spec.time_steps()
    let n_mels = mel_spec.mel_bins()
    
    var mel_list = Python.evaluate("[]")
    for t in range(time_frames):
        var row = Python.evaluate("[]")
        for m in range(n_mels):
            _ = row.append(mel_spec.data[t * n_mels + m])
        _ = mel_list.append(row)
    let np_mel = np.array(mel_list).T  # Transpose back to [n_mels, time]
    
    # Convert dB back to power
    let mel_power = librosa.db_to_power(np_mel)
    
    # Inverse mel filterbank (approximate)
    let audio_np = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=mel_spec.sample_rate,
        n_fft=mel_spec.n_fft,
        hop_length=mel_spec.hop_length,
        n_iter=n_iter
    )
    
    # Convert back to Mojo
    let audio_len = Int(audio_np.shape[0])
    var audio = Tensor[DType.float32](audio_len)
    for i in range(audio_len):
        audio[i] = Float32(audio_np[i])
    
    return audio


fn normalize_mel_spectrogram(
    inout mel_spec: Tensor[DType.float32],
    mean: Float32 = 0.0,
    std: Float32 = 1.0
) raises:
    """
    Normalize mel-spectrogram for neural network input
    
    z = (x - mean) / std
    
    Args:
        mel_spec: Mel-spectrogram to normalize (modified in-place)
        mean: Target mean (typically 0.0)
        std: Target standard deviation (typically 1.0)
    """
    # Compute current mean and std
    var sum_val: Float32 = 0.0
    for i in range(mel_spec.num_elements()):
        sum_val += mel_spec[i]
    let current_mean = sum_val / Float32(mel_spec.num_elements())
    
    var sum_sq: Float32 = 0.0
    for i in range(mel_spec.num_elements()):
        let diff = mel_spec[i] - current_mean
        sum_sq += diff * diff
    let current_std = sqrt_approx(sum_sq / Float32(mel_spec.num_elements()))
    
    # Normalize
    for i in range(mel_spec.num_elements()):
        mel_spec[i] = (mel_spec[i] - current_mean) / current_std * std + mean


fn sqrt_approx(x: Float32) -> Float32:
    """Square root approximation"""
    let py = Python.import_module("math")
    return Float32(py.sqrt(x))


fn compute_mel_statistics(mel_spec: Tensor[DType.float32]) raises -> Tuple[Float32, Float32]:
    """
    Compute mean and standard deviation of mel-spectrogram
    
    Returns: (mean, std)
    """
    var sum_val: Float32 = 0.0
    for i in range(mel_spec.num_elements()):
        sum_val += mel_spec[i]
    let mean = sum_val / Float32(mel_spec.num_elements())
    
    var sum_sq: Float32 = 0.0
    for i in range(mel_spec.num_elements()):
        let diff = mel_spec[i] - mean
        sum_sq += diff * diff
    let std = sqrt_approx(sum_sq / Float32(mel_spec.num_elements()))
    
    return (mean, std)
