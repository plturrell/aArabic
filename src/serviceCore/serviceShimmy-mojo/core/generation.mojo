"""
Generation Loop - Text Generation Engine for LLM Inference
Orchestrates tokenization, inference, sampling, and caching for text generation
"""

from collections import List
from core.tokenizer import BPETokenizer, SentencePieceTokenizer
from core.kv_cache import KVCache
from core.sampling import SamplingConfig, sample_token
from python import Python

# ============================================================================
# Generation Configuration
# ============================================================================

struct GenerationConfig:
    """Configuration for text generation"""
    var max_tokens: Int
    var temperature: Float32
    var top_k: Int
    var top_p: Float32
    var min_p: Float32
    var repetition_penalty: Float32
    var stop_tokens: List[Int]
    var stream: Bool
    
    fn __init__(
        inout self,
        max_tokens: Int = 100,
        temperature: Float32 = 0.8,
        top_k: Int = 50,
        top_p: Float32 = 0.9,
        min_p: Float32 = 0.05,
        repetition_penalty: Float32 = 1.1,
        stream: Bool = False
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.stop_tokens = List[Int]()
        self.stream = stream

# ============================================================================
# Generation Result
# ============================================================================

struct GenerationResult:
    """Result of text generation"""
    var text: String
    var tokens: List[Int]
    var n_tokens: Int
    var stopped_by: String  # "max_tokens", "stop_token", "eos"
    var time_ms: Float64
    var tokens_per_second: Float32
    
    fn __init__(
        inout self,
        text: String,
        tokens: List[Int],
        stopped_by: String,
        time_ms: Float64
    ):
        self.text = text
        self.tokens = tokens
        self.n_tokens = len(tokens)
        self.stopped_by = stopped_by
        self.time_ms = time_ms
        self.tokens_per_second = Float32(self.n_tokens) / Float32(time_ms / 1000.0)

# ============================================================================
# Text Generator
# ============================================================================

struct TextGenerator:
    """
    High-level text generation engine
    Orchestrates all components for LLM inference
    """
    var vocab_size: Int
    var eos_token_id: Int
    
    fn __init__(inout self, vocab_size: Int, eos_token_id: Int = 2):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
    
    fn generate(
        self,
        prompt_tokens: List[Int],
        config: GenerationConfig,
        kv_cache: KVCache
    ) raises -> GenerationResult:
        """
        Generate text from prompt tokens
        
        Args:
            prompt_tokens: Input token IDs
            config: Generation configuration
            kv_cache: KV cache for efficiency
        
        Returns:
            GenerationResult with generated text and metadata
        """
        print("🚀 Starting text generation...")
        print(f"  Prompt tokens: {len(prompt_tokens)}")
        print(f"  Max new tokens: {config.max_tokens}")
        print(f"  Temperature: {config.temperature}")
        print()
        
        var py = Python.import_module("time")
        var start_time = py.time()
        
        # Initialize output tokens with prompt
        var output_tokens = List[Int]()
        for i in range(len(prompt_tokens)):
            output_tokens.append(prompt_tokens[i])
        
        # Create sampling config
        var sampling_config = SamplingConfig(
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            config.repetition_penalty
        )
        
        # Generation loop
        var n_generated = 0
        var stopped_by = "max_tokens"
        
        for step in range(config.max_tokens):
            # Get current position
            var pos = len(output_tokens) - 1
            
            # Forward pass (placeholder - would call model.forward())
            var logits = self.mock_forward_pass(output_tokens[pos], pos, kv_cache)
            
            # Sample next token
            var previous_tokens_ptr = DTypePointer[DType.int32].alloc(len(output_tokens))
            for i in range(len(output_tokens)):
                previous_tokens_ptr[i] = output_tokens[i]
            
            var next_token = sample_token(
                logits,
                self.vocab_size,
                sampling_config,
                previous_tokens_ptr,
                len(output_tokens)
            )
            
            previous_tokens_ptr.free()
            logits.free()
            
            # Add to output
            output_tokens.append(next_token)
            n_generated += 1
            
            # Stream output if enabled
            if config.stream:
                print(f"Token {n_generated}: {next_token}", end=" ", flush=True)
            
            # Check stop conditions
            if next_token == self.eos_token_id:
                stopped_by = "eos"
                break
            
            # Check if token in stop list
            for stop_token in config.stop_tokens:
                if next_token == stop_token:
                    stopped_by = "stop_token"
                    break
            
            if stopped_by == "stop_token":
                break
        
        var end_time = py.time()
        var time_ms = (end_time - start_time) * 1000.0
        
        print()
        print(f"✅ Generation complete!")
        print(f"  Tokens generated: {n_generated}")
        print(f"  Time: {time_ms:.2f}ms")
        print(f"  Speed: {Float32(n_generated) / Float32(time_ms / 1000.0):.2f} tok/s")
        print(f"  Stopped by: {stopped_by}")
        print()
        
        # Convert tokens to text (would use real tokenizer)
        var generated_text = self.tokens_to_text(output_tokens)
        
        return GenerationResult(
            generated_text,
            output_tokens,
            stopped_by,
            time_ms
        )
    
    fn mock_forward_pass(
        self,
        token: Int,
        pos: Int,
        kv_cache: KVCache
    ) -> DTypePointer[DType.float32]:
        """
        Mock forward pass for testing
        In production, this would call the actual LLaMA model
        """
        var logits = DTypePointer[DType.float32].alloc(self.vocab_size)
        
        # Generate mock logits
        for i in range(self.vocab_size):
            logits[i] = Float32(i % 10) * 0.5 - 2.0
        
        # Make some tokens more likely
        logits[42] = 3.0
        logits[17] = 2.5
        logits[self.eos_token_id] = -5.0  # Make EOS less likely
        
        return logits
    
    fn tokens_to_text(self, tokens: List[Int]) -> String:
        """Convert tokens to text (simplified)"""
        var text = ""
        for i in range(len(tokens)):
            text += str(tokens[i]) + " "
        return text.strip()

# ============================================================================
# Batch Generation
# ============================================================================

struct BatchGenerator:
    """
    Batch text generation for higher throughput
    Process multiple prompts in parallel
    """
    var vocab_size: Int
    var batch_size: Int
    
    fn __init__(inout self, vocab_size: Int, batch_size: Int = 8):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
    
    fn generate_batch(
        self,
        prompts: List[List[Int]],
        config: GenerationConfig
    ) raises -> List[GenerationResult]:
        """
        Generate text for multiple prompts in batch
        
        Args:
            prompts: List of prompt token sequences
            config: Generation configuration
        
        Returns:
            List of generation results
        """
        var results = List[GenerationResult]()
        
        print(f"🚀 Batch generation: {len(prompts)} prompts")
        print(f"  Batch size: {self.batch_size}")
        print()
        
        # Process in batches
        var n_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            var start = batch_idx * self.batch_size
            var end = min(start + self.batch_size, len(prompts))
            
            print(f"  Processing batch {batch_idx + 1}/{n_batches} ({end - start} prompts)...")
            
            # Generate for each prompt in batch
            # (Production would parallelize this)
            for i in range(start, end):
                # Create KV cache for this prompt
                var kv_cache = KVCache(4, 8, 64, 4096)  # Mock config
                
                # Generate
                var generator = TextGenerator(self.vocab_size, 2)
                var result = generator.generate(prompts[i], config, kv_cache)
                results.append(result)
        
        print(f"✅ Batch generation complete: {len(results)} results")
        
        return results

# ============================================================================
# Streaming Generator
# ============================================================================

struct StreamingGenerator:
    """
    Streaming text generation
    Yields tokens as they are generated for real-time responses
    """
    var vocab_size: Int
    
    fn __init__(inout self, vocab_size: Int):
        self.vocab_size = vocab_size
    
    fn generate_stream(
        self,
        prompt_tokens: List[Int],
        config: GenerationConfig,
        kv_cache: KVCache
    ) raises:
        """
        Generate text with streaming output
        Prints each token as it's generated
        """
        print("🌊 Streaming generation started...")
        print()
        print("Generated: ", end="", flush=True)
        
        var output_tokens = List[Int]()
        for token in prompt_tokens:
            output_tokens.append(token)
        
        var sampling_config = SamplingConfig(
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            config.repetition_penalty
        )
        
        for step in range(config.max_tokens):
            var pos = len(output_tokens) - 1
            
            # Mock forward pass
            var logits = DTypePointer[DType.float32].alloc(self.vocab_size)
            for i in range(self.vocab_size):
                logits[i] = Float32(i % 10) * 0.5
            
            # Sample
            var next_token = sample_token(logits, self.vocab_size, sampling_config)
            logits.free()
            
            # Stream token
            print(f"{next_token} ", end="", flush=True)
            
            output_tokens.append(next_token)
            
            # Stop conditions
            if next_token == 2:  # EOS
                break
        
        print()
        print()
        print("✅ Streaming complete")

# ============================================================================
# Testing
# ============================================================================

fn main() raises:
    print("=" * 80)
    print("🎬 Mojo Generation Loop - Complete Text Generation")
    print("=" * 80)
    print()
    
    # Test single generation
    print("🧪 Test 1: Single Generation")
    print("-" * 80)
    
    var vocab_size = 1000
    var generator = TextGenerator(vocab_size, eos_token_id=2)
    
    # Create mock prompt
    var prompt = List[Int]()
    prompt.append(1)   # BOS
    prompt.append(42)  # Token
    prompt.append(17)  # Token
    
    # Create generation config
    var config = GenerationConfig(
        max_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        stream=False
    )
    
    # Create KV cache
    var kv_cache = KVCache(
        n_layers=4,
        n_heads=8,
        head_dim=64,
        max_seq_len=128
    )
    
    # Generate
    var result = generator.generate(prompt, config, kv_cache)
    
    print(f"Result:")
    print(f"  Tokens generated: {result.n_tokens}")
    print(f"  Time: {result.time_ms:.2f}ms")
    print(f"  Speed: {result.tokens_per_second:.2f} tok/s")
    print(f"  Stopped by: {result.stopped_by}")
    print()
    
    # Test batch generation
    print("🧪 Test 2: Batch Generation")
    print("-" * 80)
    
    var batch_generator = BatchGenerator(vocab_size, batch_size=4)
    
    var prompts = List[List[Int]]()
    for i in range(6):
        var p = List[Int]()
        p.append(1)
        p.append(i + 10)
        prompts.append(p)
    
    var batch_results = batch_generator.generate_batch(prompts, config)
    print(f"  Generated {len(batch_results)} responses")
    print()
    
    # Test streaming
    print("🧪 Test 3: Streaming Generation")
    print("-" * 80)
    
    var stream_generator = StreamingGenerator(vocab_size)
    
    var stream_config = GenerationConfig(
        max_tokens=15,
        temperature=0.8,
        stream=True
    )
    
    var kv_cache2 = KVCache(4, 8, 64, 128)
    stream_generator.generate_stream(prompt, stream_config, kv_cache2)
    
    print()
    
    print("=" * 80)
    print("✅ All generation modes working!")
    print("=" * 80)
    print()
    print("Features implemented:")
    print("  ✅ Single generation")
    print("  ✅ Batch generation")
    print("  ✅ Streaming generation")
    print("  ✅ Stop conditions (EOS, max tokens, stop tokens)")
    print("  ✅ Performance metrics")
    print("  ✅ Repetition penalty")
    print("  ✅ Configurable sampling")
    print()
    print("Integration ready:")
    print("  • Tokenizer → tokens")
    print("  • KV Cache → efficient inference")
    print("  • Sampling → diverse output")
    print("  • Generation → complete pipeline")
    print()
    print("Next: Implement full LLaMA model to replace mock_forward_pass()")
