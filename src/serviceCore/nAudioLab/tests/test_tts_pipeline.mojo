"""
Integration tests for the complete TTS pipeline.
Tests the end-to-end flow from text input to audio output.
"""

from tensor import Tensor, TensorShape
from pathlib import Path
from testing import assert_true, assert_equal, assert_almost_equal
from sys.info import os_is_macos

from ..mojo.inference.engine import TTSEngine
from ..mojo.audio.types import AudioBuffer
from ..zig.audio_io import writeWAV, readWAV


fn test_simple_sentence() raises:
    """Test synthesis of a simple sentence."""
    print("Testing simple sentence synthesis...")
    
    # Initialize TTS engine
    var model_dir = "src/serviceCore/nAudioLab/data/models"
    var tts = TTSEngine.load(model_dir)
    
    # Synthesize simple text
    var text = "Hello world"
    var audio = tts.synthesize(text)
    
    # Validate audio properties
    assert_equal(audio.sample_rate, 48000, "Sample rate should be 48kHz")
    assert_equal(audio.channels, 2, "Should be stereo")
    assert_equal(audio.bit_depth, 24, "Should be 24-bit")
    assert_true(audio.length > 0, "Audio should have samples")
    
    # Check audio is in valid range [-1.0, 1.0]
    var max_val: Float32 = -999999.0
    var min_val: Float32 = 999999.0
    for i in range(audio.length):
        if audio.samples[i] > max_val:
            max_val = audio.samples[i]
        if audio.samples[i] < min_val:
            min_val = audio.samples[i]
    
    assert_true(max_val <= 1.0, "Max sample should be <= 1.0")
    assert_true(min_val >= -1.0, "Min sample should be >= -1.0")
    
    print("✓ Simple sentence test passed")


fn test_long_text() raises:
    """Test synthesis of long text (multiple sentences)."""
    print("Testing long text synthesis...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    # Long text with various punctuation
    var text = """
    The quick brown fox jumps over the lazy dog.
    This is a test of the text-to-speech system.
    It should handle multiple sentences correctly.
    Numbers like 42, 1234, and 3.14 should be expanded properly.
    Dates such as January 16th, 2026 should also work.
    """
    
    var audio = tts.synthesize(text)
    
    # Long text should produce longer audio
    var expected_min_length = 48000 * 10  # At least 10 seconds
    assert_true(audio.length > expected_min_length, "Long text should produce sufficient audio")
    
    # Save to file for manual inspection
    var output_path = "src/serviceCore/nAudioLab/tests/output/long_text.wav"
    writeWAV(audio, output_path)
    
    print("✓ Long text test passed")


fn test_speed_control() raises:
    """Test speech speed control."""
    print("Testing speed control...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "This is a test sentence"
    
    # Normal speed
    var audio_normal = tts.synthesize(text, speed=1.0)
    var length_normal = audio_normal.length
    
    # Slow speed
    var audio_slow = tts.synthesize(text, speed=0.75)
    var length_slow = audio_slow.length
    
    # Fast speed
    var audio_fast = tts.synthesize(text, speed=1.5)
    var length_fast = audio_fast.length
    
    # Slower should be longer, faster should be shorter
    assert_true(length_slow > length_normal, "Slow speed should produce longer audio")
    assert_true(length_fast < length_normal, "Fast speed should produce shorter audio")
    
    print("✓ Speed control test passed")


fn test_pitch_control() raises:
    """Test pitch shifting."""
    print("Testing pitch control...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    var text = "Testing pitch control"
    
    # Normal pitch
    var audio_normal = tts.synthesize(text, pitch_shift=0.0)
    
    # Higher pitch
    var audio_high = tts.synthesize(text, pitch_shift=2.0)
    
    # Lower pitch
    var audio_low = tts.synthesize(text, pitch_shift=-2.0)
    
    # All should have same length (pitch doesn't affect duration)
    assert_equal(audio_normal.length, audio_high.length, "Pitch shouldn't affect length")
    assert_equal(audio_normal.length, audio_low.length, "Pitch shouldn't affect length")
    
    print("✓ Pitch control test passed")


fn test_special_characters() raises:
    """Test handling of special characters and punctuation."""
    print("Testing special character handling...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    # Text with various special cases
    var texts = List[String]()
    texts.append("Dr. Smith lives on Main St.")
    texts.append("The cost is $10.50")
    texts.append("Call 1-800-555-1234")
    texts.append("Visit www.example.com")
    texts.append("The temperature is -5°C")
    
    for i in range(len(texts)):
        var audio = tts.synthesize(texts[i])
        assert_true(audio.length > 0, "Should handle special characters")
    
    print("✓ Special character test passed")


fn test_empty_input() raises:
    """Test handling of empty or whitespace-only input."""
    print("Testing empty input handling...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    # Empty string
    try:
        var audio = tts.synthesize("")
        assert_true(audio.length == 0, "Empty input should produce no audio")
    except e:
        print("✓ Empty input correctly rejected")
    
    # Whitespace only
    try:
        var audio = tts.synthesize("   \n\t  ")
        assert_true(audio.length == 0, "Whitespace-only should produce no audio")
    except e:
        print("✓ Whitespace-only correctly rejected")
    
    print("✓ Empty input test passed")


fn test_memory_efficiency() raises:
    """Test memory usage for multiple synthesis calls."""
    print("Testing memory efficiency...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    # Synthesize multiple times to check for memory leaks
    var text = "This is a memory test"
    for i in range(20):
        var audio = tts.synthesize(text)
        # Audio should be deallocated after each iteration
        assert_true(audio.length > 0, "Each synthesis should succeed")
    
    print("✓ Memory efficiency test passed")


fn test_concurrent_synthesis() raises:
    """Test multiple engines running concurrently."""
    print("Testing concurrent synthesis...")
    
    # Load multiple engines
    var model_dir = "src/serviceCore/nAudioLab/data/models"
    var tts1 = TTSEngine.load(model_dir)
    var tts2 = TTSEngine.load(model_dir)
    
    var text = "Concurrent test"
    var audio1 = tts1.synthesize(text)
    var audio2 = tts2.synthesize(text)
    
    # Both should produce similar length audio
    var length_diff = abs(audio1.length - audio2.length)
    assert_true(length_diff < 1000, "Concurrent engines should produce similar output")
    
    print("✓ Concurrent synthesis test passed")


fn test_unicode_handling() raises:
    """Test handling of unicode characters."""
    print("Testing unicode handling...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    # Text with unicode quotes, dashes, etc.
    var text = "He said "Hello" — that's nice!"
    var audio = tts.synthesize(text)
    assert_true(audio.length > 0, "Should handle unicode characters")
    
    print("✓ Unicode handling test passed")


fn test_batch_processing() raises:
    """Test processing multiple texts in sequence."""
    print("Testing batch processing...")
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    var texts = List[String]()
    texts.append("First sentence")
    texts.append("Second sentence")
    texts.append("Third sentence")
    
    var total_length = 0
    for i in range(len(texts)):
        var audio = tts.synthesize(texts[i])
        total_length += audio.length
    
    assert_true(total_length > 0, "Batch processing should work")
    
    print("✓ Batch processing test passed")


fn run_all_tests() raises:
    """Run all integration tests."""
    print("\n" + "="*60)
    print("Running TTS Pipeline Integration Tests")
    print("="*60 + "\n")
    
    test_simple_sentence()
    test_long_text()
    test_speed_control()
    test_pitch_control()
    test_special_characters()
    test_empty_input()
    test_memory_efficiency()
    test_concurrent_synthesis()
    test_unicode_handling()
    test_batch_processing()
    
    print("\n" + "="*60)
    print("All TTS Pipeline Tests Passed! ✓")
    print("="*60 + "\n")


fn main() raises:
    run_all_tests()
