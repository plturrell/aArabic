"""
FFT Implementation for Audio Processing
Simplified implementation for 48kHz audio processing
For production, would integrate with optimized FFT library (FFTW, vDSP)
"""

from tensor import Tensor
from python import Python
from math import sqrt


@value
struct Complex:
    """Complex number for FFT calculations"""
    var real: Float32
    var imag: Float32
    
    fn __init__(inout self, real: Float32, imag: Float32):
        self.real = real
        self.imag = imag
    
    fn __init__(inout self, real: Float32):
        self.real = real
        self.imag = 0.0
    
    fn magnitude(self) -> Float32:
        """Compute magnitude |z|"""
        return sqrt(self.real * self.real + self.imag * self.imag)
    
    fn phase(self) -> Float32:
        """Compute phase angle"""
        let py = Python.import_module("math")
        return py.atan2(self.imag, self.real).to_float64().cast[DType.float32]()
    
    fn __add__(self, other: Complex) -> Complex:
        """Add two complex numbers"""
        return Complex(self.real + other.real, self.imag + other.imag)
    
    fn __sub__(self, other: Complex) -> Complex:
        """Subtract two complex numbers"""
        return Complex(self.real - other.real, self.imag - other.imag)
    
    fn __mul__(self, other: Complex) -> Complex:
        """Multiply two complex numbers"""
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )


fn stft_python_bridge(
    signal: Tensor[DType.float32],
    n_fft: Int,
    hop_length: Int,
    win_length: Int,
    window: String
) raises -> Tensor[DType.float32]:
    """
    Bridge to Python's librosa for STFT computation
    
    For Day 2 demo purposes, we use Python's librosa.
    Production version would use pure Mojo with SIMD optimizations.
    
    Returns: Magnitude spectrogram [freq_bins, time_frames]
    """
    let np = Python.import_module("numpy")
    let librosa = Python.import_module("librosa")
    
    # Convert Mojo tensor to numpy array
    var signal_list = Python.evaluate("[]")
    for i in range(signal.num_elements()):
        _ = signal_list.append(signal[i])
    let np_signal = np.array(signal_list)
    
    # Compute STFT using librosa
    let stft_complex = librosa.stft(
        np_signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True
    )
    
    # Get magnitude
    let magnitude = np.abs(stft_complex)
    
    # Convert back to Mojo tensor
    let shape = magnitude.shape
    let freq_bins = Int(shape[0])
    let time_frames = Int(shape[1])
    
    var result = Tensor[DType.float32](freq_bins, time_frames)
    for f in range(freq_bins):
        for t in range(time_frames):
            result[f * time_frames + t] = Float32(magnitude[f][t])
    
    return result


fn compute_stft(
    signal: Tensor[DType.float32],
    n_fft: Int = 2048,
    hop_length: Int = 512,
    win_length: Int = 2048,
    window: String = "hann"
) raises -> Tensor[DType.float32]:
    """
    Compute Short-Time Fourier Transform
    
    Args:
        signal: Input audio signal
        n_fft: FFT size (2048 for 48kHz gives ~23Hz resolution)
        hop_length: Number of samples between frames
        win_length: Window size
        window: Window function type
    
    Returns:
        Magnitude spectrogram [freq_bins, time_frames]
        freq_bins = n_fft // 2 + 1 (one-sided spectrum)
    """
    # For Day 2, use Python bridge
    # Production version would use pure Mojo with vectorization
    return stft_python_bridge(signal, n_fft, hop_length, win_length, window)


fn power_to_db(
    spectrogram: Tensor[DType.float32],
    ref: Float32 = 1.0,
    amin: Float32 = 1e-10,
    top_db: Float32 = 80.0
) raises -> Tensor[DType.float32]:
    """
    Convert power spectrogram to decibel scale
    
    dB = 10 * log10(S / ref)
    """
    var db = Tensor[DType.float32](spectrogram.shape())
    let log10_ref = log10_approx(ref)
    
    for i in range(spectrogram.num_elements()):
        var val = spectrogram[i]
        if val < amin:
            val = amin
        db[i] = 10.0 * (log10_approx(val) - log10_ref)
    
    # Clip to top_db range
    let max_db = db_max(db)
    let threshold = max_db - top_db
    
    for i in range(db.num_elements()):
        if db[i] < threshold:
            db[i] = threshold
    
    return db


fn log10_approx(x: Float32) -> Float32:
    """Approximate log10 using Python bridge"""
    let py = Python.import_module("math")
    return Float32(py.log10(x))


fn db_max(tensor: Tensor[DType.float32]) -> Float32:
    """Find maximum value in tensor"""
    var max_val = tensor[0]
    for i in range(1, tensor.num_elements()):
        if tensor[i] > max_val:
            max_val = tensor[i]
    return max_val


fn magnitude_to_power(magnitude: Tensor[DType.float32]) -> Tensor[DType.float32]:
    """Convert magnitude spectrogram to power (square)"""
    var power = Tensor[DType.float32](magnitude.shape())
    for i in range(magnitude.num_elements()):
        power[i] = magnitude[i] * magnitude[i]
    return power


fn get_stft_frequencies(n_fft: Int, sample_rate: Int) -> Tensor[DType.float32]:
    """
    Get frequency values for STFT bins
    
    Returns: Frequency values in Hz for each FFT bin
    """
    let freq_bins = n_fft // 2 + 1
    var frequencies = Tensor[DType.float32](freq_bins)
    
    for i in range(freq_bins):
        frequencies[i] = Float32(i * sample_rate) / Float32(n_fft)
    
    return frequencies


fn get_stft_times(
    signal_length: Int,
    n_fft: Int,
    hop_length: Int,
    sample_rate: Int
) -> Tensor[DType.float32]:
    """
    Get time values for STFT frames
    
    Returns: Time values in seconds for each frame
    """
    let n_frames = (signal_length - n_fft) // hop_length + 1
    var times = Tensor[DType.float32](n_frames)
    
    for i in range(n_frames):
        let sample_pos = i * hop_length + n_fft // 2
        times[i] = Float32(sample_pos) / Float32(sample_rate)
    
    return times


# Note: For production, we would implement:
# 1. Cooley-Tukey FFT algorithm with radix-2 decimation
# 2. SIMD vectorization for parallel computation
# 3. Apple Accelerate framework binding for vDSP_fft
# 4. Memory-efficient in-place FFT
# 5. Optimized window function application
#
# Example production FFT signature would be:
# fn fft_optimized(inout signal: DTypePointer[DType.float32], 
#                  inout output: DTypePointer[DType.float32],
#                  n: Int, stride: Int = 1) -> None
