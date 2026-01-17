"""
Prosody Feature Extraction for TTS
Energy, voiced/unvoiced detection, and duration features
"""

from tensor import Tensor
from python import Python
from math import sqrt
from .f0_extractor import YINConfig, extract_f0_yin


@value
struct ProsodyFeatures:
    """Complete prosody feature set for speech synthesis"""
    var f0: Tensor[DType.float32]          # Pitch contour in Hz [n_frames]
    var energy: Tensor[DType.float32]      # Frame energy [n_frames]
    var voiced: Tensor[DType.bool]         # Voice/unvoiced mask [n_frames]
    var log_f0: Tensor[DType.float32]      # Log-scaled F0 [n_frames]
    var normalized_energy: Tensor[DType.float32]  # Normalized energy [n_frames]
    
    fn __init__(inout self,
                f0: Tensor[DType.float32],
                energy: Tensor[DType.float32],
                voiced: Tensor[DType.bool],
                log_f0: Tensor[DType.float32],
                normalized_energy: Tensor[DType.float32]):
        """Initialize prosody features"""
        self.f0 = f0
        self.energy = energy
        self.voiced = voiced
        self.log_f0 = log_f0
        self.normalized_energy = normalized_energy
    
    fn n_frames(self) -> Int:
        """Get number of frames"""
        return self.f0.num_elements()


fn extract_frame_energy(
    audio: Tensor[DType.float32],
    frame_length: Int,
    hop_length: Int,
    method: String = "rms"
) -> Tensor[DType.float32]:
    """
    Extract frame-level energy
    
    Args:
        audio: Audio signal [samples]
        frame_length: Frame size in samples
        hop_length: Hop size in samples
        method: "rms" (root mean square) or "peak"
    
    Returns:
        Energy values [n_frames]
    """
    let signal_len = audio.num_elements()
    let n_frames = (signal_len - frame_length) // hop_length + 1
    
    var energy = Tensor[DType.float32](n_frames)
    
    for frame_idx in range(n_frames):
        let start = frame_idx * hop_length
        let end = start + frame_length
        
        if method == "rms":
            # RMS energy
            var sum_squares: Float32 = 0.0
            var count: Int = 0
            
            for i in range(start, end):
                if i < signal_len:
                    sum_squares += audio[i] * audio[i]
                    count += 1
            
            if count > 0:
                let py = Python.import_module("math")
                energy[frame_idx] = Float32(py.sqrt(sum_squares / Float32(count)))
            else:
                energy[frame_idx] = 0.0
        else:  # peak
            # Peak amplitude in frame
            var max_abs: Float32 = 0.0
            
            for i in range(start, end):
                if i < signal_len:
                    let abs_val = abs(audio[i])
                    if abs_val > max_abs:
                        max_abs = abs_val
            
            energy[frame_idx] = max_abs
    
    return energy


fn extract_energy_from_mel(
    mel_spec: Tensor[DType.float32]
) -> Tensor[DType.float32]:
    """
    Extract energy from mel-spectrogram
    
    Energy is the mean across mel bins for each time frame
    
    Args:
        mel_spec: Mel-spectrogram [n_mels, time_frames] or [time_frames, n_mels]
    
    Returns:
        Energy values [time_frames]
    """
    let shape = mel_spec.shape()
    let dim0 = Int(shape[0])
    let dim1 = Int(shape[1])
    
    # Assume [time, n_mels] format
    let time_frames = dim0
    let n_mels = dim1
    
    var energy = Tensor[DType.float32](time_frames)
    
    for t in range(time_frames):
        var sum_val: Float32 = 0.0
        for m in range(n_mels):
            sum_val += mel_spec[t * n_mels + m]
        energy[t] = sum_val / Float32(n_mels)
    
    return energy


fn detect_voiced_frames(
    audio: Tensor[DType.float32],
    f0: Tensor[DType.float32],
    frame_length: Int,
    hop_length: Int,
    zcr_threshold: Float32 = 0.3,
    energy_threshold: Float32 = 0.01
) -> Tensor[DType.bool]:
    """
    Detect voiced/unvoiced frames using multiple criteria
    
    Combines:
    1. F0 detection (primary)
    2. Zero-crossing rate
    3. Energy threshold
    
    Args:
        audio: Audio signal [samples]
        f0: F0 contour [n_frames]
        frame_length: Frame size
        hop_length: Hop size
        zcr_threshold: Max ZCR for voiced (normalized)
        energy_threshold: Min energy for voiced
    
    Returns:
        Boolean mask [n_frames], True for voiced
    """
    let n_frames = f0.num_elements()
    var voiced = Tensor[DType.bool](n_frames)
    
    # Extract zero-crossing rate
    let zcr = compute_zero_crossing_rate(audio, frame_length, hop_length)
    
    # Extract energy
    let energy = extract_frame_energy(audio, frame_length, hop_length, "rms")
    
    # Normalize ZCR and energy
    var max_zcr: Float32 = 0.0
    var max_energy: Float32 = 0.0
    
    for i in range(n_frames):
        if zcr[i] > max_zcr:
            max_zcr = zcr[i]
        if energy[i] > max_energy:
            max_energy = energy[i]
    
    # Combine criteria
    for i in range(n_frames):
        let f0_voiced = f0[i] > 0.0
        let zcr_voiced = (zcr[i] / max_zcr) < zcr_threshold if max_zcr > 0.0 else False
        let energy_voiced = (energy[i] / max_energy) > energy_threshold if max_energy > 0.0 else False
        
        # Frame is voiced if F0 detected AND (low ZCR OR high energy)
        voiced[i] = f0_voiced and (zcr_voiced or energy_voiced)
    
    return voiced


fn compute_zero_crossing_rate(
    audio: Tensor[DType.float32],
    frame_length: Int,
    hop_length: Int
) -> Tensor[DType.float32]:
    """
    Compute zero-crossing rate per frame
    
    ZCR is the rate at which signal changes sign
    High ZCR indicates noise or unvoiced sounds
    Low ZCR indicates periodic/voiced sounds
    
    Args:
        audio: Audio signal [samples]
        frame_length: Frame size
        hop_length: Hop size
    
    Returns:
        ZCR values [n_frames] (rate per sample)
    """
    let signal_len = audio.num_elements()
    let n_frames = (signal_len - frame_length) // hop_length + 1
    
    var zcr = Tensor[DType.float32](n_frames)
    
    for frame_idx in range(n_frames):
        let start = frame_idx * hop_length
        let end = start + frame_length
        
        var crossings: Int = 0
        var prev_sign = 1 if audio[start] >= 0.0 else -1
        
        for i in range(start + 1, end):
            if i < signal_len:
                let curr_sign = 1 if audio[i] >= 0.0 else -1
                if curr_sign != prev_sign:
                    crossings += 1
                prev_sign = curr_sign
        
        zcr[frame_idx] = Float32(crossings) / Float32(frame_length)
    
    return zcr


fn normalize_energy(
    energy: Tensor[DType.float32],
    method: String = "minmax"
) -> Tensor[DType.float32]:
    """
    Normalize energy values for neural network input
    
    Args:
        energy: Raw energy values [n_frames]
        method: "minmax" or "zscore"
    
    Returns:
        Normalized energy [n_frames]
    """
    let n = energy.num_elements()
    var normalized = Tensor[DType.float32](n)
    
    if method == "minmax":
        # Min-max normalization to [0, 1]
        var min_val: Float32 = 1e9
        var max_val: Float32 = -1e9
        
        for i in range(n):
            if energy[i] < min_val:
                min_val = energy[i]
            if energy[i] > max_val:
                max_val = energy[i]
        
        let range_val = max_val - min_val
        if range_val > 0.0:
            for i in range(n):
                normalized[i] = (energy[i] - min_val) / range_val
        else:
            for i in range(n):
                normalized[i] = 0.0
    
    else:  # zscore
        # Z-score normalization (mean=0, std=1)
        var sum_val: Float32 = 0.0
        for i in range(n):
            sum_val += energy[i]
        let mean = sum_val / Float32(n)
        
        var sum_sq: Float32 = 0.0
        for i in range(n):
            let diff = energy[i] - mean
            sum_sq += diff * diff
        
        let py = Python.import_module("math")
        let std = Float32(py.sqrt(sum_sq / Float32(n)))
        
        if std > 0.0:
            for i in range(n):
                normalized[i] = (energy[i] - mean) / std
        else:
            for i in range(n):
                normalized[i] = 0.0
    
    return normalized


fn extract_prosody_features(
    audio: Tensor[DType.float32],
    sample_rate: Int = 48000,
    frame_length: Int = 2048,
    hop_length: Int = 512,
    f0_min: Float32 = 80.0,
    f0_max: Float32 = 400.0
) -> ProsodyFeatures:
    """
    Extract complete prosody feature set
    
    Combines:
    - F0 (pitch) via YIN algorithm
    - Frame energy (RMS)
    - Voiced/unvoiced detection
    - Normalized features for training
    
    Args:
        audio: Audio signal [samples]
        sample_rate: Sample rate in Hz
        frame_length: Analysis frame size
        hop_length: Hop size between frames
        f0_min: Minimum F0 to detect
        f0_max: Maximum F0 to detect
    
    Returns:
        Complete ProsodyFeatures struct
    """
    # Configure YIN for F0 extraction
    let yin_config = YINConfig(
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        f0_min=f0_min,
        f0_max=f0_max,
        threshold=0.1
    )
    
    # Extract F0
    let f0 = extract_f0_yin(audio, yin_config)
    
    # Extract energy
    let energy = extract_frame_energy(audio, frame_length, hop_length, "rms")
    
    # Detect voiced frames
    let voiced = detect_voiced_frames(
        audio, f0, frame_length, hop_length,
        zcr_threshold=0.3,
        energy_threshold=0.01
    )
    
    # Convert F0 to log scale
    let n_frames = f0.num_elements()
    var log_f0 = Tensor[DType.float32](n_frames)
    let py = Python.import_module("math")
    
    for i in range(n_frames):
        if f0[i] > 0.0:
            log_f0[i] = Float32(py.log(f0[i]))
        else:
            log_f0[i] = 0.0
    
    # Normalize energy
    let normalized_energy = normalize_energy(energy, "minmax")
    
    return ProsodyFeatures(
        f0=f0,
        energy=energy,
        voiced=voiced,
        log_f0=log_f0,
        normalized_energy=normalized_energy
    )


fn compute_prosody_statistics(
    features: ProsodyFeatures
) -> Dict[String, Float32]:
    """
    Compute statistics for prosody features
    
    Args:
        features: Prosody features
    
    Returns:
        Dictionary of statistics
    """
    let n_frames = features.n_frames()
    
    # F0 statistics (voiced frames only)
    var f0_sum: Float32 = 0.0
    var f0_count: Int = 0
    var f0_min: Float32 = 1e9
    var f0_max: Float32 = -1e9
    
    for i in range(n_frames):
        if features.f0[i] > 0.0:
            f0_sum += features.f0[i]
            f0_count += 1
            if features.f0[i] < f0_min:
                f0_min = features.f0[i]
            if features.f0[i] > f0_max:
                f0_max = features.f0[i]
    
    let f0_mean = f0_sum / Float32(f0_count) if f0_count > 0 else 0.0
    
    # Energy statistics
    var energy_sum: Float32 = 0.0
    var energy_min: Float32 = 1e9
    var energy_max: Float32 = -1e9
    
    for i in range(n_frames):
        energy_sum += features.energy[i]
        if features.energy[i] < energy_min:
            energy_min = features.energy[i]
        if features.energy[i] > energy_max:
            energy_max = features.energy[i]
    
    let energy_mean = energy_sum / Float32(n_frames)
    
    # Voiced percentage
    var voiced_count: Int = 0
    for i in range(n_frames):
        if features.voiced[i]:
            voiced_count += 1
    let voiced_pct = Float32(voiced_count) / Float32(n_frames) * 100.0
    
    # Create results dictionary (pseudo-code, Mojo Dict syntax may vary)
    var stats = Dict[String, Float32]()
    stats["f0_mean"] = f0_mean
    stats["f0_min"] = f0_min
    stats["f0_max"] = f0_max
    stats["energy_mean"] = energy_mean
    stats["energy_min"] = energy_min
    stats["energy_max"] = energy_max
    stats["voiced_pct"] = voiced_pct
    stats["n_frames"] = Float32(n_frames)
    
    return stats
