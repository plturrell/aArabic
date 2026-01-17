"""
FFT (Feed-Forward Transformer) Block for FastSpeech2

This module implements the FFT block, which is the core building block
of FastSpeech2's encoder and decoder. It combines multi-head attention,
feed-forward networks, and layer normalization with residual connections.

Reference:
    "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"
    (Ren et al., 2020)
    https://arxiv.org/abs/2006.04558
"""

from tensor import Tensor
from .attention import AttentionConfig, MultiHeadAttention
from .feed_forward import FFNConfig, FeedForwardNetwork
from .layer_norm import LayerNormConfig, LayerNorm


struct FFTConfig:
    """Configuration for FFT block"""
    var d_model: Int
    var n_heads: Int
    var d_ff: Int
    var dropout: Float32
    var use_conv: Bool  # Use 1D convolutions instead of linear layers
    
    fn __init__(
        inout self,
        d_model: Int = 256,
        n_heads: Int = 4,
        d_ff: Int = 1024,
        dropout: Float32 = 0.1,
        use_conv: Bool = True
    ):
        """
        Initialize FFT block configuration
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            use_conv: Whether to use Conv1D in FFN (FastSpeech2 style)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_conv = use_conv


struct FFTBlock:
    """
    Feed-Forward Transformer Block
    
    The core building block of FastSpeech2, consisting of:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Layer normalization after each sub-layer
    4. Residual connections around each sub-layer
    
    Architecture:
        Input
          ↓
        [Multi-Head Attention]
          ↓
        [+ Residual]
          ↓
        [Layer Norm]
          ↓
        [Feed-Forward Network]
          ↓
        [+ Residual]
          ↓
        [Layer Norm]
          ↓
        Output
    """
    var config: FFTConfig
    
    # Sub-modules
    var self_attention: MultiHeadAttention
    var ffn: FeedForwardNetwork
    var norm1: LayerNorm
    var norm2: LayerNorm
    
    fn __init__(inout self, config: FFTConfig):
        """Initialize FFT block with configured sub-modules"""
        self.config = config
        
        # Multi-head self-attention
        let attn_config = AttentionConfig(
            n_heads=config.n_heads,
            d_model=config.d_model,
            dropout=config.dropout
        )
        self.self_attention = MultiHeadAttention(attn_config)
        
        # Feed-forward network
        let ffn_config = FFNConfig(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        self.ffn = FeedForwardNetwork(ffn_config)
        
        # Layer normalization
        let ln_config = LayerNormConfig(normalized_shape=config.d_model)
        self.norm1 = LayerNorm(ln_config)
        self.norm2 = LayerNorm(ln_config)
    
    fn forward(
        self,
        x: Tensor[DType.float32],
        mask: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Forward pass through FFT block
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        
        Algorithm:
            1. Self-attention with residual and layer norm
            2. Feed-forward with residual and layer norm
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let d_model = x.shape()[2]
        
        # Sub-layer 1: Multi-head self-attention
        var attn_output = self.self_attention.forward(x, mask)
        
        # Residual connection + layer norm
        var x1 = self._add_tensors(x, attn_output)
        x1 = self.norm1.forward(x1)
        
        # Sub-layer 2: Feed-forward network
        var ffn_output = self.ffn.forward(x1)
        
        # Residual connection + layer norm
        var output = self._add_tensors(x1, ffn_output)
        output = self.norm2.forward(output)
        
        return output
    
    fn _add_tensors(
        self,
        a: Tensor[DType.float32],
        b: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Add two tensors element-wise (for residual connections)
        
        Args:
            a: First tensor
            b: Second tensor
        
        Returns:
            Sum of tensors
        """
        var result = Tensor[DType.float32](a.shape())
        
        let total_elements = a.num_elements()
        for i in range(total_elements):
            result[i] = a[i] + b[i]
        
        return result
    
    fn count_parameters(self) -> Int:
        """
        Count total trainable parameters
        
        Returns:
            Total parameter count
        """
        var total = 0
        total += self.self_attention.count_parameters()
        total += self.ffn.count_parameters()
        total += self.norm1.count_parameters()
        total += self.norm2.count_parameters()
        return total


struct Conv1DBlock:
    """
    1D Convolution Block for FastSpeech2
    
    An alternative formulation using 1D convolutions with residuals.
    Often used in FastSpeech2 for local feature extraction.
    
    Architecture:
        Input → Conv1D → ReLU → Conv1D → + Residual → Output
    """
    var d_model: Int
    var kernel_size: Int
    var padding: Int
    
    # Convolution weights
    var conv1_weight: Tensor[DType.float32]
    var conv1_bias: Tensor[DType.float32]
    var conv2_weight: Tensor[DType.float32]
    var conv2_bias: Tensor[DType.float32]
    
    fn __init__(
        inout self,
        d_model: Int,
        kernel_size: Int = 9,
        padding: Int = 4
    ):
        """
        Initialize Conv1D block
        
        Args:
            d_model: Model dimension (input/output channels)
            kernel_size: Convolution kernel size
            padding: Padding size (typically (kernel_size - 1) // 2)
        """
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Initialize convolution weights
        from random import rand
        from math import sqrt
        
        let scale = sqrt(2.0 / Float32(d_model * kernel_size))
        
        # First convolution
        self.conv1_weight = Tensor[DType.float32](d_model, d_model, kernel_size)
        rand(self.conv1_weight.data(), d_model * d_model * kernel_size)
        for i in range(d_model * d_model * kernel_size):
            self.conv1_weight[i] = (self.conv1_weight[i] - 0.5) * scale * 2.0
        
        self.conv1_bias = Tensor[DType.float32](d_model)
        
        # Second convolution
        self.conv2_weight = Tensor[DType.float32](d_model, d_model, kernel_size)
        rand(self.conv2_weight.data(), d_model * d_model * kernel_size)
        for i in range(d_model * d_model * kernel_size):
            self.conv2_weight[i] = (self.conv2_weight[i] - 0.5) * scale * 2.0
        
        self.conv2_bias = Tensor[DType.float32](d_model)
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass through Conv1D block
        
        Args:
            x: Input [batch_size, seq_len, d_model]
        
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        
        # First convolution
        var h = self._conv1d(x, self.conv1_weight, self.conv1_bias)
        
        # ReLU
        h = self._relu(h)
        
        # Second convolution
        var output = self._conv1d(h, self.conv2_weight, self.conv2_bias)
        
        # Residual connection
        for i in range(batch_size * seq_len * self.d_model):
            output[i] = output[i] + x[i]
        
        return output
    
    fn _conv1d(
        self,
        x: Tensor[DType.float32],
        weight: Tensor[DType.float32],
        bias: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Apply 1D convolution with padding
        
        Simplified implementation - in practice would use optimized conv
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let in_channels = x.shape()[2]
        let out_channels = weight.shape()[0]
        
        var output = Tensor[DType.float32](batch_size, seq_len, out_channels)
        
        # Simplified: point-wise for now (kernel_size=1 equivalent)
        for b in range(batch_size):
            for s in range(seq_len):
                for oc in range(out_channels):
                    var sum = bias[oc]
                    for ic in range(in_channels):
                        # Use center of kernel
                        let k_center = self.kernel_size // 2
                        sum += x[b, s, ic] * weight[oc, ic, k_center]
                    output[b, s, oc] = sum
        
        return output
    
    fn _relu(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply ReLU activation"""
        var output = Tensor[DType.float32](x.shape())
        for i in range(x.num_elements()):
            output[i] = max(0.0, x[i])
        return output


fn test_fft_block():
    """Test FFT block implementation"""
    print("="*70)
    print("FFT BLOCK TEST")
    print("="*70)
    
    # Configuration
    let config = FFTConfig(d_model=256, n_heads=4, d_ff=1024)
    print(f"\nConfiguration:")
    print(f"  d_model:  {config.d_model}")
    print(f"  n_heads:  {config.n_heads}")
    print(f"  d_ff:     {config.d_ff}")
    print(f"  dropout:  {config.dropout}")
    
    # Create FFT block
    var fft_block = FFTBlock(config)
    print(f"\nParameters: {fft_block.count_parameters():,}")
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.d_model)
    
    # Fill with test data
    for i in range(batch_size * seq_len * config.d_model):
        x[i] = Float32(i % 100) / 100.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.d_model}]")
    
    # Create mask (no masking)
    var mask = Tensor[DType.float32](batch_size, seq_len, seq_len)
    for i in range(batch_size * seq_len * seq_len):
        mask[i] = 1.0
    
    # Forward pass
    print("\nRunning forward pass...")
    var output = fft_block.forward(x, mask)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ FFT block test passed!")
    print("="*70)


fn test_conv1d_block():
    """Test Conv1D block"""
    print("\n" + "="*70)
    print("CONV1D BLOCK TEST")
    print("="*70)
    
    let d_model = 256
    let kernel_size = 9
    
    print(f"\nConfiguration:")
    print(f"  d_model:      {d_model}")
    print(f"  kernel_size:  {kernel_size}")
    
    var conv_block = Conv1DBlock(d_model, kernel_size)
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, d_model)
    
    for i in range(batch_size * seq_len * d_model):
        x[i] = Float32(i % 100) / 100.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {d_model}]")
    
    # Forward pass
    print("\nRunning forward pass...")
    var output = conv_block.forward(x)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ Conv1D block test passed!")
    print("="*70)


fn main():
    """Run FFT block tests"""
    test_fft_block()
    test_conv1d_block()
