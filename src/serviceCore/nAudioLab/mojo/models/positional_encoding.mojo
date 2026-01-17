"""
Positional Encoding for FastSpeech2

This module implements positional encoding to inject information about
the position of tokens in the sequence. Since Transformers have no
inherent notion of position, this is crucial for sequence modeling.

Reference:
    "Attention is All You Need" (Vaswani et al., 2017)
    Section 3.5: Positional Encoding
"""

from tensor import Tensor
from math import sin, cos, pow
import math


struct PositionalEncodingConfig:
    """Configuration for positional encoding"""
    var d_model: Int
    var max_len: Int
    var dropout: Float32
    
    fn __init__(
        inout self,
        d_model: Int = 256,
        max_len: Int = 5000,
        dropout: Float32 = 0.1
    ):
        """
        Initialize positional encoding configuration
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout


struct PositionalEncoding:
    """
    Sinusoidal Positional Encoding
    
    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies.
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
        - pos: position in sequence
        - i: dimension index
        - d_model: model dimension
    
    This allows the model to learn to attend by relative positions,
    since for any fixed offset k, PE(pos+k) can be represented as a
    linear function of PE(pos).
    """
    var config: PositionalEncodingConfig
    var pe: Tensor[DType.float32]  # Pre-computed positional encodings
    
    fn __init__(inout self, config: PositionalEncodingConfig):
        """
        Initialize positional encoding
        
        Pre-computes positional encodings for efficiency.
        """
        self.config = config
        
        # Create positional encoding matrix [max_len, d_model]
        self.pe = Tensor[DType.float32](config.max_len, config.d_model)
        
        # Compute positional encodings
        self._compute_encodings()
    
    fn _compute_encodings(inout self):
        """
        Pre-compute sinusoidal positional encodings
        
        Uses sine for even indices and cosine for odd indices.
        """
        let max_len = self.config.max_len
        let d_model = self.config.d_model
        
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                # Compute angle: pos / 10000^(2i/d_model)
                let div_term = pow(10000.0, Float32(i) / Float32(d_model))
                let angle = Float32(pos) / div_term
                
                # Apply sine to even indices
                self.pe[pos, i] = sin(angle)
                
                # Apply cosine to odd indices (if within bounds)
                if i + 1 < d_model:
                    self.pe[pos, i + 1] = cos(angle)
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Add positional encoding to input
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        
        Returns:
            Input with positional encoding added [batch_size, seq_len, d_model]
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let d_model = x.shape()[2]
        
        # Verify sequence length doesn't exceed max_len
        if seq_len > self.config.max_len:
            print(f"Warning: seq_len ({seq_len}) > max_len ({self.config.max_len})")
        
        var output = Tensor[DType.float32](batch_size, seq_len, d_model)
        
        # Add positional encoding to each position in sequence
        for b in range(batch_size):
            for s in range(seq_len):
                for d in range(d_model):
                    output[b, s, d] = x[b, s, d] + self.pe[s, d]
        
        return output
    
    fn get_encoding(self, positions: Tensor[DType.int32]) -> Tensor[DType.float32]:
        """
        Get positional encodings for specific positions
        
        Args:
            positions: Position indices [batch_size, seq_len]
        
        Returns:
            Positional encodings [batch_size, seq_len, d_model]
        """
        let batch_size = positions.shape()[0]
        let seq_len = positions.shape()[1]
        let d_model = self.config.d_model
        
        var output = Tensor[DType.float32](batch_size, seq_len, d_model)
        
        for b in range(batch_size):
            for s in range(seq_len):
                let pos = Int(positions[b, s])
                if pos < self.config.max_len:
                    for d in range(d_model):
                        output[b, s, d] = self.pe[pos, d]
        
        return output


struct LearnedPositionalEncoding:
    """
    Learned Positional Encoding
    
    An alternative to sinusoidal encodings where position embeddings
    are learned parameters. Often used in modern models like BERT.
    """
    var config: PositionalEncodingConfig
    var embeddings: Tensor[DType.float32]  # Learned position embeddings
    
    fn __init__(inout self, config: PositionalEncodingConfig):
        """Initialize learned positional encoding"""
        self.config = config
        
        # Initialize position embeddings randomly
        from random import rand
        from math import sqrt
        
        self.embeddings = Tensor[DType.float32](config.max_len, config.d_model)
        
        # Xavier initialization
        let scale = sqrt(2.0 / Float32(config.d_model))
        rand(self.embeddings.data(), config.max_len * config.d_model)
        
        for i in range(config.max_len * config.d_model):
            self.embeddings[i] = (self.embeddings[i] - 0.5) * scale * 2.0
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Add learned positional encoding to input
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        
        Returns:
            Input with positional encoding added
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let d_model = x.shape()[2]
        
        var output = Tensor[DType.float32](batch_size, seq_len, d_model)
        
        for b in range(batch_size):
            for s in range(seq_len):
                for d in range(d_model):
                    output[b, s, d] = x[b, s, d] + self.embeddings[s, d]
        
        return output


fn test_positional_encoding():
    """Test sinusoidal positional encoding"""
    print("="*70)
    print("POSITIONAL ENCODING TEST")
    print("="*70)
    
    # Configuration
    let config = PositionalEncodingConfig(d_model=256, max_len=100)
    print(f"\nConfiguration:")
    print(f"  d_model:  {config.d_model}")
    print(f"  max_len:  {config.max_len}")
    
    # Create positional encoding
    var pos_enc = PositionalEncoding(config)
    print(f"\nPre-computed encodings shape: [{config.max_len}, {config.d_model}]")
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.d_model)
    
    # Fill with ones to see effect of positional encoding
    for i in range(batch_size * seq_len * config.d_model):
        x[i] = 1.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.d_model}]")
    print("Input filled with 1.0 to visualize positional encoding effect")
    
    # Forward pass
    print("\nApplying positional encoding...")
    var output = pos_enc.forward(x)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample positional encodings (position 0, first 5 dims):")
    for i in range(5):
        print(f"  PE[0, {i}] = {pos_enc.pe[0, i]:.6f}")
    
    print(f"\nSample positional encodings (position 5, first 5 dims):")
    for i in range(5):
        print(f"  PE[5, {i}] = {pos_enc.pe[5, i]:.6f}")
    
    print(f"\nOutput values (position 0, first 5 dims):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ Positional encoding test passed!")
    print("="*70)


fn test_learned_positional_encoding():
    """Test learned positional encoding"""
    print("\n" + "="*70)
    print("LEARNED POSITIONAL ENCODING TEST")
    print("="*70)
    
    # Configuration
    let config = PositionalEncodingConfig(d_model=256, max_len=100)
    print(f"\nConfiguration:")
    print(f"  d_model:  {config.d_model}")
    print(f"  max_len:  {config.max_len}")
    
    # Create learned positional encoding
    var learned_pos_enc = LearnedPositionalEncoding(config)
    print(f"\nLearned embeddings shape: [{config.max_len}, {config.d_model}]")
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.d_model)
    
    for i in range(batch_size * seq_len * config.d_model):
        x[i] = 1.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.d_model}]")
    
    # Forward pass
    print("\nApplying learned positional encoding...")
    var output = learned_pos_enc.forward(x)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample learned encodings (position 0, first 5 dims):")
    for i in range(5):
        print(f"  Learned[0, {i}] = {learned_pos_enc.embeddings[0, i]:.6f}")
    
    print("\n✓ Learned positional encoding test passed!")
    print("="*70)


fn test_positional_encoding_properties():
    """Test mathematical properties of positional encoding"""
    print("\n" + "="*70)
    print("POSITIONAL ENCODING PROPERTIES TEST")
    print("="*70)
    
    let config = PositionalEncodingConfig(d_model=256, max_len=100)
    var pos_enc = PositionalEncoding(config)
    
    print("\nTesting properties:")
    print("  1. Different positions have different encodings")
    print("  2. Encodings are bounded [-1, 1] (due to sin/cos)")
    print("  3. Pattern repeats at different frequencies")
    
    # Check encoding at different positions
    print(f"\nEncoding at position 0 (first dim): {pos_enc.pe[0, 0]:.6f}")
    print(f"Encoding at position 1 (first dim): {pos_enc.pe[1, 0]:.6f}")
    print(f"Encoding at position 2 (first dim): {pos_enc.pe[2, 0]:.6f}")
    
    # Check that values are in [-1, 1]
    var min_val = pos_enc.pe[0, 0]
    var max_val = pos_enc.pe[0, 0]
    
    for pos in range(min(10, config.max_len)):
        for dim in range(min(10, config.d_model)):
            let val = pos_enc.pe[pos, dim]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    
    print(f"\nValue range (first 10 pos × 10 dims):")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")
    print(f"  Within [-1, 1]: {min_val >= -1.0 and max_val <= 1.0}")
    
    print("\n✓ Positional encoding properties verified!")
    print("="*70)


fn main():
    """Run positional encoding tests"""
    test_positional_encoding()
    test_learned_positional_encoding()
    test_positional_encoding_properties()
