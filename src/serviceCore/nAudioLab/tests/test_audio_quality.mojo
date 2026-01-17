"""
Audio quality validation tests.
Validates output meets professional audio quality standards.
"""

from tensor import Tensor, TensorShape
from math import sqrt, log10, abs, pow
from testing import assert_true, assert_almost_equal

from ..mojo.inference.engine import TTSEngine
from ..mojo.audio.types import AudioBuffer


fn measure_lufs(audio: AudioBuffer) -> Float32:
    """
    Measure integrated loudness using ITU-R BS.1770-4 algorithm.
    Simplified version for testing.
    """
    # K-weighting filter (simplified - normally needs proper filtering)
    var sum_squares: Float32 = 0.0
    var count = 0
    
    # Calculate RMS over 400ms blocks
    var block_size = int(audio.sample_rate * 0.4)  # 400ms
    var num_blocks = audio.length // block_size
    
    for block_idx in range(num_blocks):
        var block_sum: Float32 = 0.0
        var start = block_idx * block_size
        var end = start + block_size
        
        for i in range(start, end):
            var sample = audio.samples[i]
            block_sum += sample * sample
        
        var block_rms = sqrt(block_sum / Float32(block_size))
        
        # Gating: only count blocks above -70 LUFS
        if block_rms > 0.0001:  # Approximate -70 LUFS
            sum_squares += block_sum
            count += block_size
    
    if count == 0:
        return -999.0  # Silent audio
    
    var mean_square = sum_squares / Float32(count)
    var lufs = -0.691 + 10.0 * log10(mean_square)  # ITU-R BS.1770 formula
    
    return lufs


fn measure_thd(audio: AudioBuffer) -> Float32:
    """
    Measure Total Harmonic Distortion + Noise (THD+N).
    Simplified version for testing.
    """
    # For a simple test, measure the ratio of high-frequency content
    # Real THD requires FFT and harmonic analysis
    
    var signal_power: Float32 = 0.0
    var noise_power: Float32 = 0.0
    
    # Simple high-pass filter to estimate noise
    for i in range(1, audio.length):
        var signal = audio.samples[i]
        var diff = audio.samples[i] - audio.samples[i-1]
        
        signal_power += signal * signal
        noise_power += diff * diff
    
    if signal_power < 0.000001:
        return 0.0
    
    var thd = sqrt(noise_power / signal_power)
    return thd


fn measure_peak_level(audio: AudioBuffer) -> Float32:
    """Measure peak level in dBFS."""
    var max_sample: Float32 = 0.0
    
    for i in range(audio.length):
        var abs_sample = abs(audio.samples[i])
        if abs_sample > max_sample:
            max_sample = abs_sample
    
    if max_sample < 0.000001:
        return -999.0
    
    return 20.0 * log10(max_sample)


fn measure_dynamic_range(audio: AudioBuffer) -> Float32:
    """Measure dynamic range (difference between peak and RMS)."""
    var peak = measure_peak_level(audio)
    
    # Calculate RMS
    var sum_squares: Float32 = 0.0
    for i in range(audio.length):
        sum_squares += audio.samples[i] * audio.samples[i]
    
    var rms = sqrt(sum_squares / Float32(audio.length))
    var rms_db = 20.0 * log10(rms) if rms > 0.000001 else -999.0
    
    return peak - rms_db


fn test_lufs_target() raises:
    """Test that audio meets -16 LUFS target."""
    print("Testing LUFS loudness target...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "This is a loudness test for audio quality validation"
    var audio = tts.synthesize(text)
    
    var lufs = measure_lufs(audio)
    print(f"  Measured LUFS: {lufs:.2f}")
    
    # Target: -16 LUFS ± 1.0 dB
    assert_true(lufs >= -17.0 and lufs <= -15.0, 
                f"LUFS should be between -17 and -15, got {lufs:.2f}")
    
    print("✓ LUFS target test passed")


fn test_thd_limit() raises:
    """Test that THD+N is below 1%."""
    print("Testing THD+N limit...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Testing total harmonic distortion"
    var audio = tts.synthesize(text)
    
    var thd = measure_thd(audio)
    print(f"  Measured THD+N: {thd*100:.3f}%")
    
    # Should be below 1% for high quality
    assert_true(thd < 0.01, f"THD+N should be < 1%, got {thd*100:.3f}%")
    
    print("✓ THD+N test passed")


fn test_peak_limiting() raises:
    """Test that peaks are properly limited."""
    print("Testing peak limiting...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Peak limiting test with loud exclamations!"
    var audio = tts.synthesize(text)
    
    var peak_db = measure_peak_level(audio)
    print(f"  Peak level: {peak_db:.2f} dBFS")
    
    # Should be limited to -0.3 dBFS
    assert_true(peak_db <= -0.2, f"Peak should be <= -0.2 dBFS, got {peak_db:.2f}")
    
    print("✓ Peak limiting test passed")


fn test_dynamic_range() raises:
    """Test adequate dynamic range."""
    print("Testing dynamic range...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "This sentence has both quiet and loud sections for testing"
    var audio = tts.synthesize(text)
    
    var dr = measure_dynamic_range(audio)
    print(f"  Dynamic range: {dr:.2f} dB")
    
    # Should have at least 6 dB dynamic range
    assert_true(dr >= 6.0, f"Dynamic range should be >= 6 dB, got {dr:.2f}")
    
    print("✓ Dynamic range test passed")


fn test_no_clipping() raises:
    """Test that audio doesn't clip."""
    print("Testing for clipping...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Testing for digital clipping artifacts"
    var audio = tts.synthesize(text)
    
    var clip_count = 0
    var clip_threshold: Float32 = 0.99
    
    for i in range(audio.length):
        if abs(audio.samples[i]) > clip_threshold:
            clip_count += 1
    
    var clip_percent = Float32(clip_count) / Float32(audio.length) * 100.0
    print(f"  Clipping samples: {clip_percent:.3f}%")
    
    # Should have minimal clipping (< 0.1%)
    assert_true(clip_percent < 0.1, f"Clipping should be < 0.1%, got {clip_percent:.3f}%")
    
    print("✓ No clipping test passed")


fn test_dc_offset() raises:
    """Test that audio has no DC offset."""
    print("Testing DC offset...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Testing for DC offset"
    var audio = tts.synthesize(text)
    
    # Calculate mean (should be near zero)
    var sum: Float32 = 0.0
    for i in range(audio.length):
        sum += audio.samples[i]
    
    var mean = sum / Float32(audio.length)
    print(f"  DC offset: {mean:.6f}")
    
    # Should be very close to zero
    assert_true(abs(mean) < 0.001, f"DC offset should be < 0.001, got {abs(mean):.6f}")
    
    print("✓ DC offset test passed")


fn test_stereo_imaging() raises:
    """Test stereo image quality."""
    print("Testing stereo imaging...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Testing stereo field"
    var audio = tts.synthesize(text)
    
    # For stereo, channels should have slight differences but be correlated
    if audio.channels == 2:
        var left_power: Float32 = 0.0
        var right_power: Float32 = 0.0
        var correlation: Float32 = 0.0
        
        # Assuming interleaved samples: L, R, L, R, ...
        for i in range(0, audio.length, 2):
            var left = audio.samples[i]
            var right = audio.samples[i + 1]
            
            left_power += left * left
            right_power += right * right
            correlation += left * right
        
        var balance = left_power / right_power if right_power > 0 else 1.0
        print(f"  L/R balance ratio: {balance:.3f}")
        
        # Balance should be close to 1.0 (within 20%)
        assert_true(balance >= 0.8 and balance <= 1.2, 
                   f"Stereo balance should be 0.8-1.2, got {balance:.3f}")
    
    print("✓ Stereo imaging test passed")


fn test_frequency_response() raises:
    """Test frequency response coverage."""
    print("Testing frequency response...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Testing frequency response across the audible spectrum"
    var audio = tts.synthesize(text)
    
    # Simple test: check that audio has both low and high frequency content
    var low_freq_energy: Float32 = 0.0
    var high_freq_energy: Float32 = 0.0
    
    # Low freq: smooth changes (window of 100 samples)
    # High freq: rapid changes (window of 5 samples)
    var window_low = 100
    var window_high = 5
    
    for i in range(window_low, audio.length):
        var diff_low = abs(audio.samples[i] - audio.samples[i - window_low])
        low_freq_energy += diff_low
    
    for i in range(window_high, audio.length):
        var diff_high = abs(audio.samples[i] - audio.samples[i - window_high])
        high_freq_energy += diff_high
    
    print(f"  Low freq energy: {low_freq_energy:.2f}")
    print(f"  High freq energy: {high_freq_energy:.2f}")
    
    # Both should be present
    assert_true(low_freq_energy > 0.1, "Should have low frequency content")
    assert_true(high_freq_energy > 0.1, "Should have high frequency content")
    
    print("✓ Frequency response test passed")


fn test_silence_handling() raises:
    """Test proper silence between words."""
    print("Testing silence handling...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Word. Another. Sentence."  # Should have pauses
    var audio = tts.synthesize(text)
    
    # Count samples below threshold (silence)
    var silence_count = 0
    var threshold: Float32 = 0.01
    
    for i in range(audio.length):
        if abs(audio.samples[i]) < threshold:
            silence_count += 1
    
    var silence_percent = Float32(silence_count) / Float32(audio.length) * 100.0
    print(f"  Silence: {silence_percent:.2f}%")
    
    # Should have some silence (5-20%)
    assert_true(silence_percent >= 5.0 and silence_percent <= 20.0,
               f"Silence should be 5-20%, got {silence_percent:.2f}%")
    
    print("✓ Silence handling test passed")


fn test_sample_rate_accuracy() raises:
    """Test that sample rate is exactly 48kHz."""
    print("Testing sample rate accuracy...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Testing sample rate"
    var audio = tts.synthesize(text)
    
    assert_equal(audio.sample_rate, 48000, "Sample rate must be 48000 Hz")
    
    print("✓ Sample rate test passed")


fn run_all_quality_tests() raises:
    """Run all audio quality tests."""
    print("\n" + "="*60)
    print("Running Audio Quality Validation Tests")
    print("="*60 + "\n")
    
    test_lufs_target()
    test_thd_limit()
    test_peak_limiting()
    test_dynamic_range()
    test_no_clipping()
    test_dc_offset()
    test_stereo_imaging()
    test_frequency_response()
    test_silence_handling()
    test_sample_rate_accuracy()
    
    print("\n" + "="*60)
    print("All Audio Quality Tests Passed! ✓")
    print("="*60 + "\n")


fn main() raises:
    run_all_quality_tests()
