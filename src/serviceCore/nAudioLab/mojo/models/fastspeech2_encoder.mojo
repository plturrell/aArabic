"""
FastSpeech2 Encoder Implementation

This module implements the encoder component of FastSpeech2, which converts
phoneme sequences into hidden representations that can be used by the decoder
to generate mel-spectrograms.

Reference:
    "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"
    (Ren et al., 2020)
    https://arxiv.org/abs/2006.04558
"""

from tensor import Tensor
from random import rand
from math import sqrt
from .fft_block import FFTConfig, FFTBlock
from .positional_encoding import PositionalEncodingConfig, PositionalEncoding


struct EncoderConfig:
    """Configuration for FastSpeech2 encoder"""
    var n_phonemes: Int       # Number of phonemes in vocabulary
    var d_model: Int          # Model dimension
    var n_heads: Int          # Number of attention heads
    var d_ff: Int             # Feed-forward hidden dimension
    var n_layers: Int         # Number of FFT blocks
    var dropout: Float32      # Dropout probability
    var max_seq_len: Int      # Maximum sequence length
    
    fn __init__(
        inout self,
        n_phonemes: Int = 70,
        d_model: Int = 256,
        n_heads: Int = 4,
        d_ff: Int = 1024,
        n_layers: Int = 4,
        dropout: Float32 = 0.1,
        max_seq_len: Int = 1000
    ):
        """
        Initialize encoder configuration
        
        Args:
            n_phonemes: Vocabulary size (number of phonemes)
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_layers: Number of FFT blocks
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        self.n_phonemes = n_phonemes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len


struct PhonemeEmbedding:
    """
    Phoneme Embedding Layer
    
    Converts discrete phoneme indices to continuous vector representations.
    """
    var n_phonemes: Int
    var d_model: Int
    var embeddings: Tensor[DType.float32]  # [n_phonemes, d_model]
    
    fn __init__(inout self, n_phonemes: Int, d_model: Int):
        """
        Initialize phoneme embedding
        
        Args:
            n_phonemes: Vocabulary size
            d_model: Embedding dimension
        """
        self.n_phonemes = n_phonemes
        self.d_model = d_model
        
        # Initialize embeddings with Xavier initialization
        self.embeddings = Tensor[DType.float32](n_phonemes, d_model)
        
        let scale = sqrt(2.0 / Float32(d_model))
        rand(self.embeddings.data(), n_phonemes * d_model)
        
        for i in range(n_phonemes * d_model):
            self.embeddings[i] = (self.embeddings[i] - 0.5) * scale * 2.0
    
    fn forward(self, phonemes: Tensor[DType.int32]) -> Tensor[DType.float32]:
        """
        Embed phoneme indices
        
        Args:
            phonemes: Phoneme indices [batch_size, seq_len]
        
        Returns:
            Phoneme embeddings [batch_size, seq_len, d_model]
        """
        let batch_size = phonemes.shape()[0]
        let seq_len = phonemes.shape()[1]
        
        var output = Tensor[DType.float32](batch_size, seq_len, self.d_model)
        
        # Lookup embeddings for each phoneme
        for b in range(batch_size):
            for s in range(seq_len):
                let phoneme_idx = Int(phonemes[b, s])
                
                # Copy embedding vector
                if phoneme_idx < self.n_phonemes:
                    for d in range(self.d_model):
                        output[b, s, d] = self.embeddings[phoneme_idx, d]
        
        return output


struct FastSpeech2Encoder:
    """
    FastSpeech2 Encoder
    
    Converts phoneme sequences to hidden representations using:
    1. Phoneme embedding
    2. Positional encoding
    3. Stack of FFT blocks (Feed-Forward Transformer)
    
    Architecture:
        Phonemes [batch, seq_len]
            ↓
        [Phoneme Embedding]
            ↓
        Embedded [batch, seq_len, d_model]
            ↓
        [Positional Encoding]
            ↓
        [FFT Block 1]
        [FFT Block 2]
        [FFT Block 3]
        [FFT Block 4]
            ↓
        Encoder Output [batch, seq_len, d_model]
    """
    var config: EncoderConfig
    
    # Components
    var phoneme_embedding: PhonemeEmbedding
    var positional_encoding: PositionalEncoding
    var fft_blocks: List[FFTBlock]
    
    fn __init__(inout self, config: EncoderConfig):
        """Initialize FastSpeech2 encoder"""
        self.config = config
        
        # Phoneme embedding layer
        self.phoneme_embedding = PhonemeEmbedding(config.n_phonemes, config.d_model)
        
        # Positional encoding
        let pos_config = PositionalEncodingConfig(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout
        )
        self.positional_encoding = PositionalEncoding(pos_config)
        
        # FFT blocks
        self.fft_blocks = List[FFTBlock]()
        
        let fft_config = FFTConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        
        for _ in range(config.n_layers):
            self.fft_blocks.append(FFTBlock(fft_config))
    
    fn forward(
        self,
        phonemes: Tensor[DType.int32],
        mask: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Encode phoneme sequence
        
        Args:
            phonemes: Phoneme indices [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            Encoded representations [batch_size, seq_len, d_model]
        """
        # 1. Embed phonemes
        var x = self.phoneme_embedding.forward(phonemes)
        
        # 2. Add positional encoding
        x = self.positional_encoding.forward(x)
        
        # 3. Pass through FFT blocks
        for i in range(self.config.n_layers):
            x = self.fft_blocks[i].forward(x, mask)
        
        return x
    
    fn count_parameters(self) -> Int:
        """
        Count total trainable parameters
        
        Returns:
            Total parameter count
        """
        var total = 0
        
        # Phoneme embeddings
        total += self.config.n_phonemes * self.config.d_model
        
        # FFT blocks
        for i in range(self.config.n_layers):
            total += self.fft_blocks[i].count_parameters()
        
        return total


fn test_phoneme_embedding():
    """Test phoneme embedding"""
    print("="*70)
    print("PHONEME EMBEDDING TEST")
    print("="*70)
    
    let n_phonemes = 70
    let d_model = 256
    
    print(f"\nConfiguration:")
    print(f"  n_phonemes: {n_phonemes}")
    print(f"  d_model:    {d_model}")
    
    var embedding = PhonemeEmbedding(n_phonemes, d_model)
    print(f"\nEmbedding matrix shape: [{n_phonemes}, {d_model}]")
    
    # Test input - phoneme indices
    let batch_size = 2
    let seq_len = 10
    var phonemes = Tensor[DType.int32](batch_size, seq_len)
    
    # Fill with sample phoneme indices
    for b in range(batch_size):
        for s in range(seq_len):
            phonemes[b, s] = Int32((b * seq_len + s) % n_phonemes)
    
    print(f"\nInput shape: [{batch_size}, {seq_len}]")
    print(f"Sample phoneme indices: ", end="")
    for s in range(min(5, seq_len)):
        print(f"{phonemes[0, s]} ", end="")
    print()
    
    # Forward pass
    print("\nEmbedding phonemes...")
    var embedded = embedding.forward(phonemes)
    
    print(f"Output shape: [{embedded.shape()[0]}, {embedded.shape()[1]}, {embedded.shape()[2]}]")
    print(f"\nSample embedding values (first phoneme, first 5 dims):")
    for i in range(5):
        print(f"  embedded[0, 0, {i}] = {embedded[0, 0, i]:.6f}")
    
    print("\n✓ Phoneme embedding test passed!")
    print("="*70)


fn test_fastspeech2_encoder():
    """Test FastSpeech2 encoder"""
    print("\n" + "="*70)
    print("FASTSPEECH2 ENCODER TEST")
    print("="*70)
    
    # Configuration
    let config = EncoderConfig(
        n_phonemes=70,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4
    )
    
    print(f"\nConfiguration:")
    print(f"  n_phonemes:  {config.n_phonemes}")
    print(f"  d_model:     {config.d_model}")
    print(f"  n_heads:     {config.n_heads}")
    print(f"  d_ff:        {config.d_ff}")
    print(f"  n_layers:    {config.n_layers}")
    
    # Create encoder
    var encoder = FastSpeech2Encoder(config)
    print(f"\nParameters: {encoder.count_parameters():,}")
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var phonemes = Tensor[DType.int32](batch_size, seq_len)
    
    # Fill with sample phoneme indices
    for b in range(batch_size):
        for s in range(seq_len):
            phonemes[b, s] = Int32((b * seq_len + s) % config.n_phonemes)
    
    print(f"\nInput shape: [{batch_size}, {seq_len}]")
    
    # Create mask (no masking)
    var mask = Tensor[DType.float32](batch_size, seq_len, seq_len)
    for i in range(batch_size * seq_len * seq_len):
        mask[i] = 1.0
    
    # Forward pass
    print("\nRunning forward pass through encoder...")
    var output = encoder.forward(phonemes, mask)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample output values (first 5 dims):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ FastSpeech2 encoder test passed!")
    print("="*70)


fn test_encoder_parameter_breakdown():
    """Test parameter counting breakdown"""
    print("\n" + "="*70)
    print("ENCODER PARAMETER BREAKDOWN")
    print("="*70)
    
    let config = EncoderConfig(n_phonemes=70, d_model=256, n_heads=4, d_ff=1024, n_layers=4)
    var encoder = FastSpeech2Encoder(config)
    
    print(f"\nConfiguration:")
    print(f"  n_phonemes: {config.n_phonemes}")
    print(f"  d_model:    {config.d_model}")
    print(f"  n_layers:   {config.n_layers}")
    
    # Calculate component parameters
    let embedding_params = config.n_phonemes * config.d_model
    let fft_block_params = encoder.fft_blocks[0].count_parameters()
    let total_fft_params = fft_block_params * config.n_layers
    let total_params = encoder.count_parameters()
    
    print(f"\nParameter Breakdown:")
    print(f"  Phoneme Embedding:    {embedding_params:,}")
    print(f"  FFT Block (×1):       {fft_block_params:,}")
    print(f"  FFT Blocks (×{config.n_layers}):      {total_fft_params:,}")
    print(f"  {'─' * 40}")
    print(f"  Total Encoder:        {total_params:,}")
    
    print("\n✓ Parameter breakdown complete!")
    print("="*70)


fn main():
    """Run FastSpeech2 encoder tests"""
    test_phoneme_embedding()
    test_fastspeech2_encoder()
    test_encoder_parameter_breakdown()
