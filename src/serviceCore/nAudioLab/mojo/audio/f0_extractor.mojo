"""
F0 (Pitch) Extraction using YIN Algorithm
Accurate fundamental frequency estimation for speech synthesis
"""

from tensor import Tensor
from python import Python
from math import sqrt


struct YINConfig:
    """Configuration for YIN F0 extraction"""
    var sample_rate: Int
    var frame_length: Int
    var hop_length: Int
    var f0_min: Float32
    var f0_max: Float32
    var threshold: Float32
    
    fn __init__(inout self,
                sample_rate: Int = 48000,
                frame_length: Int = 2048,
                hop_length: Int = 512,
                f0_min: Float32 = 80.0,
                f0_max: Float32 = 400.0,
                threshold: Float32 = 0.1):
        """Initialize YIN configuration for speech F0 detection"""
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = threshold


fn compute_difference_function(frame: Tensor[DType.float32]) -> Tensor[DType.float32]:
    """
    Compute difference function (step 1 of YIN)
    
    d(τ) = Σ(x[j] - x[j+τ])²
    
    Args:
        frame: Audio frame [frame_length]
    
    Returns:
        Difference function [frame_length/2]
    """
    let frame_len = frame.num_elements()
    let max_lag = frame_len // 2
    
    var diff_fn = Tensor[DType.float32](max_lag)
    
    for lag in range(max_lag):
        var sum_sq_diff: Float32 = 0.0
        for j in range(frame_len - lag):
            let d = frame[j] - frame[j + lag]
            sum_sq_diff += d * d
        diff_fn[lag] = sum_sq_diff
    
    return diff_fn


fn cumulative_mean_normalized_difference(diff_fn: Tensor[DType.float32]) -> Tensor[DType.float32]:
    """
    Compute cumulative mean normalized difference (step 2 of YIN)
    
    d'(τ) = d(τ) / [(1/τ) * Σ(d(j) for j=1..τ)]
    d'(0) = 1
    
    Args:
        diff_fn: Difference function
    
    Returns:
        Normalized difference function
    """
    let n = diff_fn.num_elements()
    var cmnd = Tensor[DType.float32](n)
    
    # d'(0) = 1 by definition
    cmnd[0] = 1.0
    
    var cumulative_sum: Float32 = 0.0
    for tau in range(1, n):
        cumulative_sum += diff_fn[tau]
        if cumulative_sum > 0.0:
            cmnd[tau] = diff_fn[tau] * Float32(tau) / cumulative_sum
        else:
            cmnd[tau] = 1.0
    
    return cmnd


fn absolute_threshold(cmnd: Tensor[DType.float32], threshold: Float32, tau_min: Int) -> Int:
    """
    Find first tau below threshold (step 3 of YIN)
    
    Args:
        cmnd: Cumulative mean normalized difference
        threshold: Threshold value (typically 0.1)
        tau_min: Minimum tau to consider (based on f0_max)
    
    Returns:
        Tau value or -1 if not found
    """
    let n = cmnd.num_elements()
    
    for tau in range(tau_min, n):
        if cmnd[tau] < threshold:
            # Found first minimum below threshold
            return tau
    
    return -1


fn parabolic_interpolation(
    cmnd: Tensor[DType.float32],
    tau: Int
) -> Float32:
    """
    Refine tau estimate using parabolic interpolation (step 4 of YIN)
    
    Fit parabola to (tau-1, tau, tau+1) and find minimum
    
    Args:
        cmnd: Cumulative mean normalized difference
        tau: Coarse tau estimate
    
    Returns:
        Refined tau estimate
    """
    if tau <= 0 or tau >= cmnd.num_elements() - 1:
        return Float32(tau)
    
    let s0 = cmnd[tau - 1]
    let s1 = cmnd[tau]
    let s2 = cmnd[tau + 1]
    
    # Parabolic interpolation formula
    let delta = 0.5 * (s0 - s2) / (s0 - 2.0 * s1 + s2)
    
    return Float32(tau) + delta


fn extract_f0_yin(
    audio: Tensor[DType.float32],
    config: YINConfig
) -> Tensor[DType.float32]:
    """
    Extract F0 contour using YIN algorithm
    
    YIN (de Cheveigné & Kawahara, 2002):
    1. Compute difference function
    2. Cumulative mean normalized difference
    3. Absolute threshold
    4. Parabolic interpolation
    5. Convert tau to frequency
    
    Args:
        audio: Audio signal [samples]
        config: YIN configuration
    
    Returns:
        F0 contour in Hz [n_frames], 0.0 for unvoiced
    """
    let signal_len = audio.num_elements()
    let n_frames = (signal_len - config.frame_length) // config.hop_length + 1
    
    # Calculate tau bounds based on F0 range
    let tau_min = Int(Float32(config.sample_rate) / config.f0_max)
    let tau_max = Int(Float32(config.sample_rate) / config.f0_min)
    
    var f0_contour = Tensor[DType.float32](n_frames)
    
    for frame_idx in range(n_frames):
        let start = frame_idx * config.hop_length
        let end = start + config.frame_length
        
        # Extract frame
        var frame = Tensor[DType.float32](config.frame_length)
        for i in range(config.frame_length):
            if start + i < signal_len:
                frame[i] = audio[start + i]
            else:
                frame[i] = 0.0
        
        # Step 1: Difference function
        let diff_fn = compute_difference_function(frame)
        
        # Step 2: Cumulative mean normalized difference
        let cmnd = cumulative_mean_normalized_difference(diff_fn)
        
        # Step 3: Absolute threshold
        let tau = absolute_threshold(cmnd, config.threshold, tau_min)
        
        if tau == -1 or tau >= tau_max:
            # Unvoiced or out of range
            f0_contour[frame_idx] = 0.0
        else:
            # Step 4: Parabolic interpolation
            let refined_tau = parabolic_interpolation(cmnd, tau)
            
            # Step 5: Convert to frequency
            f0_contour[frame_idx] = Float32(config.sample_rate) / refined_tau
    
    return f0_contour


fn smooth_f0_contour(
    f0: Tensor[DType.float32],
    window_size: Int = 5
) -> Tensor[DType.float32]:
    """
    Smooth F0 contour using median filter
    
    Args:
        f0: Raw F0 contour [n_frames]
        window_size: Median filter window size (odd number)
    
    Returns:
        Smoothed F0 contour
    """
    let n = f0.num_elements()
    var smoothed = Tensor[DType.float32](n)
    let half_window = window_size // 2
    
    for i in range(n):
        var window = Python.evaluate("[]")
        
        for j in range(-half_window, half_window + 1):
            let idx = i + j
            if idx >= 0 and idx < n and f0[idx] > 0.0:
                _ = window.append(f0[idx])
        
        if len(window) > 0:
            let np = Python.import_module("numpy")
            smoothed[i] = Float32(np.median(window))
        else:
            smoothed[i] = 0.0
    
    return smoothed


fn interpolate_unvoiced(
    f0: Tensor[DType.float32]
) -> Tensor[DType.float32]:
    """
    Interpolate F0 values for unvoiced regions
    
    Args:
        f0: F0 contour with 0.0 for unvoiced [n_frames]
    
    Returns:
        F0 contour with interpolated values
    """
    let n = f0.num_elements()
    var interpolated = Tensor[DType.float32](n)
    
    # Copy original values
    for i in range(n):
        interpolated[i] = f0[i]
    
    # Find and interpolate gaps
    var last_voiced_idx: Int = -1
    var last_voiced_value: Float32 = 0.0
    
    for i in range(n):
        if f0[i] > 0.0:
            # Voiced frame
            if last_voiced_idx >= 0 and i - last_voiced_idx > 1:
                # Interpolate gap
                let gap_size = i - last_voiced_idx - 1
                let delta = (f0[i] - last_voiced_value) / Float32(gap_size + 1)
                
                for j in range(1, gap_size + 1):
                    interpolated[last_voiced_idx + j] = last_voiced_value + delta * Float32(j)
            
            last_voiced_idx = i
            last_voiced_value = f0[i]
    
    return interpolated


fn log_f0(f0: Tensor[DType.float32]) -> Tensor[DType.float32]:
    """
    Convert F0 to log scale for neural network input
    
    log(f0) is more perceptually relevant than linear f0
    
    Args:
        f0: F0 contour in Hz [n_frames]
    
    Returns:
        Log F0 contour [n_frames]
    """
    let n = f0.num_elements()
    var log_f0_contour = Tensor[DType.float32](n)
    
    let py = Python.import_module("math")
    for i in range(n):
        if f0[i] > 0.0:
            log_f0_contour[i] = Float32(py.log(f0[i]))
        else:
            log_f0_contour[i] = 0.0
    
    return log_f0_contour


fn extract_f0_statistics(
    f0: Tensor[DType.float32]
) -> Tuple[Float32, Float32, Float32, Float32]:
    """
    Compute F0 statistics for normalization
    
    Args:
        f0: F0 contour [n_frames]
    
    Returns:
        (mean, std, min, max) of voiced F0 values
    """
    var sum_val: Float32 = 0.0
    var count: Int = 0
    var min_val: Float32 = 1e9
    var max_val: Float32 = -1e9
    
    # Compute mean, min, max
    for i in range(f0.num_elements()):
        if f0[i] > 0.0:
            sum_val += f0[i]
            count += 1
            if f0[i] < min_val:
                min_val = f0[i]
            if f0[i] > max_val:
                max_val = f0[i]
    
    if count == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    let mean = sum_val / Float32(count)
    
    # Compute std
    var sum_sq: Float32 = 0.0
    for i in range(f0.num_elements()):
        if f0[i] > 0.0:
            let diff = f0[i] - mean
            sum_sq += diff * diff
    
    let py = Python.import_module("math")
    let std = Float32(py.sqrt(sum_sq / Float32(count)))
    
    return (mean, std, min_val, max_val)
