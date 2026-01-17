"""
TTS Pipeline Utilities
=======================

Helper functions and batch processing for TTS inference.
"""

from tensor import Tensor, TensorShape
from python import Python
from time import now

# Import engine components
from .engine import TTSEngine, InferenceConfig
from ..audio.types import AudioBuffer


struct BatchRequest:
    """Request for batch TTS synthesis."""
    var texts: List[String]
    var output_paths: List[String]
    var config: InferenceConfig
    
    fn __init__(inout self):
        """Initialize empty batch request."""
        self.texts = List[String]()
        self.output_paths = List[String]()
        self.config = InferenceConfig()
    
    fn add(inout self, text: String, output_path: String):
        """Add text to batch."""
        self.texts.append(text)
        self.output_paths.append(output_path)


struct BatchResult:
    """Result of batch TTS synthesis."""
    var success_count: Int
    var failure_count: Int
    var total_duration: Float32
    var errors: List[String]
    
    fn __init__(inout self):
        """Initialize empty result."""
        self.success_count = 0
        self.failure_count = 0
        self.total_duration = 0.0
        self.errors = List[String]()
    
    fn report(self) -> String:
        """Generate human-readable report."""
        var report = "\nBatch Synthesis Report\n"
        report += "=" * 50 + "\n"
        report += "Total requests: " + String(self.success_count + self.failure_count) + "\n"
        report += "Successful: " + String(self.success_count) + "\n"
        report += "Failed: " + String(self.failure_count) + "\n"
        report += "Total audio duration: " + String(self.total_duration) + " seconds\n"
        
        if self.failure_count > 0:
            report += "\nErrors:\n"
            for i in range(len(self.errors)):
                report += "  - " + self.errors[i] + "\n"
        
        return report


fn synthesize_batch(
    engine: TTSEngine,
    request: BatchRequest
) raises -> BatchResult:
    """
    Process a batch of TTS requests.
    
    Args:
        engine: Loaded TTS engine
        request: Batch request with texts and output paths
        
    Returns:
        BatchResult with success/failure statistics
    """
    var result = BatchResult()
    
    print("\nProcessing batch of", len(request.texts), "requests...")
    
    # Process each request
    for i in range(len(request.texts)):
        let text = request.texts[i]
        let output_path = request.output_paths[i]
        
        print("\n[" + String(i + 1) + "/" + String(len(request.texts)) + "]", text[:50] + "...")
        
        try:
            # Synthesize audio
            let audio = engine.synthesize(text)
            
            # Save to file (via Zig FFI)
            audio.save(output_path)
            
            result.success_count += 1
            result.total_duration += Float32(audio.length) / Float32(audio.sample_rate)
            
            print("  ✓ Saved to", output_path)
            
        except e:
            result.failure_count += 1
            let error_msg = "Request " + String(i + 1) + ": " + str(e)
            result.errors.append(error_msg)
            print("  ✗ Error:", e)
    
    return result


fn synthesize_to_file(
    engine: TTSEngine,
    text: String,
    output_path: String,
    config: Optional[InferenceConfig] = None
) raises:
    """
    Synthesize text and save to file.
    
    Args:
        engine: Loaded TTS engine
        text: Input text
        output_path: Path to save audio file (.wav or .mp3)
        config: Optional inference configuration
    """
    # Apply config if provided
    if config:
        engine.set_config(config.value())
    
    # Synthesize
    let audio = engine.synthesize(text)
    
    # Save
    audio.save(output_path)
    
    print("\n✓ Audio saved to:", output_path)
    print("  Duration:", Float32(audio.length) / Float32(audio.sample_rate), "seconds")
    print("  Sample rate:", audio.sample_rate, "Hz")
    print("  Channels:", audio.channels)
    print("  Bit depth:", audio.bit_depth, "bits")


fn benchmark_inference(
    engine: TTSEngine,
    test_texts: List[String],
    iterations: Int = 5
) raises -> Dict[String, Float32]:
    """
    Benchmark inference performance.
    
    Args:
        engine: Loaded TTS engine
        test_texts: List of test texts
        iterations: Number of iterations per text
        
    Returns:
        Dictionary with timing statistics
    """
    print("\nBenchmarking TTS Inference")
    print("=" * 50)
    print("Test texts:", len(test_texts))
    print("Iterations per text:", iterations)
    
    var total_time = 0.0
    var total_chars = 0
    var total_audio_duration = 0.0
    
    for text in test_texts:
        print("\nText:", text[:50] + "...")
        print("Length:", len(text), "characters")
        
        var text_time = 0.0
        
        for iter in range(iterations):
            let start = now()
            let audio = engine.synthesize(text)
            let end = now()
            
            let elapsed = Float32(end - start) / 1e9  # Convert to seconds
            text_time += elapsed
            
            if iter == 0:
                # Only count audio duration once
                total_audio_duration += Float32(audio.length) / Float32(audio.sample_rate)
        
        let avg_time = text_time / Float32(iterations)
        print("  Average inference time:", avg_time, "seconds")
        
        total_time += text_time
        total_chars += len(text)
    
    # Calculate statistics
    let total_iterations = len(test_texts) * iterations
    let avg_per_inference = total_time / Float32(total_iterations)
    let chars_per_second = Float32(total_chars * iterations) / total_time
    let rtf = total_time / total_audio_duration  # Real-time factor
    
    print("\n" + "=" * 50)
    print("Benchmark Results:")
    print("  Total inference time:", total_time, "seconds")
    print("  Average per inference:", avg_per_inference, "seconds")
    print("  Characters per second:", chars_per_second)
    print("  Real-time factor (RTF):", rtf)
    
    var results = Dict[String, Float32]()
    results["total_time"] = total_time
    results["avg_per_inference"] = avg_per_inference
    results["chars_per_second"] = chars_per_second
    results["rtf"] = rtf
    
    return results


fn stream_synthesis(
    engine: TTSEngine,
    text: String,
    chunk_size: Int = 100
) raises -> List[AudioBuffer]:
    """
    Synthesize text in chunks for streaming.
    
    Args:
        engine: Loaded TTS engine
        text: Input text
        chunk_size: Maximum characters per chunk
        
    Returns:
        List of audio buffers (one per chunk)
    """
    print("\nStreaming synthesis for text of length", len(text))
    
    # Split text into sentences
    let sentences = split_into_sentences(text)
    print("Split into", len(sentences), "sentences")
    
    var audio_chunks = List[AudioBuffer]()
    
    # Process each sentence
    for i in range(len(sentences)):
        let sentence = sentences[i]
        print("\n[" + String(i + 1) + "/" + String(len(sentences)) + "]", sentence[:50] + "...")
        
        # Synthesize chunk
        let audio = engine.synthesize(sentence)
        audio_chunks.append(audio)
        
        print("  Generated", audio.length, "samples")
    
    return audio_chunks


fn split_into_sentences(text: String) -> List[String]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    var sentences = List[String]()
    var current = ""
    
    for i in range(len(text)):
        let char = text[i]
        current += char
        
        # Check for sentence boundaries
        if char == '.' or char == '!' or char == '?':
            # Add sentence if not empty
            if len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
    
    # Add remaining text
    if len(current.strip()) > 0:
        sentences.append(current.strip())
    
    return sentences


fn concatenate_audio(chunks: List[AudioBuffer]) raises -> AudioBuffer:
    """
    Concatenate multiple audio buffers.
    
    Args:
        chunks: List of audio buffers to concatenate
        
    Returns:
        Single concatenated audio buffer
    """
    if len(chunks) == 0:
        raise Error("No audio chunks to concatenate")
    
    # Calculate total length
    var total_length = 0
    for chunk in chunks:
        total_length += chunk.length
    
    # Create output buffer
    let sample_rate = chunks[0].sample_rate
    let channels = chunks[0].channels
    let bit_depth = chunks[0].bit_depth
    
    var output = AudioBuffer(
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth
    )
    
    output.samples = DTypePointer[DType.float32].alloc(total_length * channels)
    output.length = total_length
    
    # Copy all chunks
    var offset = 0
    for chunk in chunks:
        let chunk_samples = chunk.length * channels
        for i in range(chunk_samples):
            output.samples[offset + i] = chunk.samples[i]
        offset += chunk_samples
    
    return output


fn main():
    """Test pipeline utilities."""
    try:
        print("TTS Pipeline Utilities Test")
        print("=" * 50)
        
        # Create test batch request
        var request = BatchRequest()
        request.add("Hello world!", "output/test1.wav")
        request.add("This is a test.", "output/test2.wav")
        request.add("Goodbye!", "output/test3.wav")
        
        print("\nBatch request created with", len(request.texts), "items")
        
        # Test sentence splitting
        let long_text = "Hello world! How are you? I'm doing great. Thank you for asking."
        let sentences = split_into_sentences(long_text)
        print("\nSentence splitting test:")
        print("  Input:", long_text)
        print("  Sentences:", len(sentences))
        for i in range(len(sentences)):
            print("    -", sentences[i])
        
        print("\n✓ Pipeline utilities validated!")
        
    except e:
        print("Error:", e)
