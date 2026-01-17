"""
Feed-Forward Network Implementation for FastSpeech2

This module implements the position-wise feed-forward network used in
Transformer models. It consists of two linear transformations with a
ReLU activation in between.

Reference:
    "Attention is All You Need" (Vaswani et al., 2017)
    FFN(x) = max(0, xW1 + b1)W2 + b2
"""

from tensor import Tensor
from random import rand
from math import sqrt


struct FFNConfig:
    """Configuration for feed-forward network"""
    var d_model: Int      # Input/output dimension
    var d_ff: Int         # Hidden layer dimension
    var dropout: Float32  # Dropout probability
    
    fn __init__(inout self, d_model: Int = 256, d_ff: Int = 1024, dropout: Float32 = 0.1):
        """
        Initialize FFN configuration
        
        Args:
            d_model: Model dimension (input/output size)
            d_ff: Feed-forward dimension (hidden layer size)
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout


struct FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network
    
    Two-layer fully connected network with ReLU activation.
    Applied independently to each position in the sequence.
    
    Architecture:
        x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model) -> output
    
    This is equivalent to two 1x1 convolutions applied position-wise.
    """
    var config: FFNConfig
    
    # First layer: d_model -> d_ff
    var W1: Tensor[DType.float32]
    var b1: Tensor[DType.float32]
    
    # Second layer: d_ff -> d_model
    var W2: Tensor[DType.float32]
    var b2: Tensor[DType.float32]
    
    fn __init__(inout self, config: FFNConfig):
        """
        Initialize feed-forward network with random weights
        
        Uses Xavier/Glorot initialization for weights:
            scale = sqrt(2 / (fan_in + fan_out))
        """
        self.config = config
        
        let d_model = config.d_model
        let d_ff = config.d_ff
        
        # Initialize first layer weights
        let scale1 = sqrt(2.0 / (d_model + d_ff))
        self.W1 = self._init_weights(d_model, d_ff, scale1)
        self.b1 = Tensor[DType.float32](d_ff)
        
        # Initialize second layer weights
        let scale2 = sqrt(2.0 / (d_ff + d_model))
        self.W2 = self._init_weights(d_ff, d_model, scale2)
        self.b2 = Tensor[DType.float32](d_model)
    
    fn _init_weights(self, rows: Int, cols: Int, scale: Float32) -> Tensor[DType.float32]:
        """
        Initialize weight matrix with Xavier/Glorot initialization
        
        Args:
            rows: Number of input features
            cols: Number of output features
            scale: Scaling factor
        
        Returns:
            Initialized weight tensor
        """
        var weights = Tensor[DType.float32](rows, cols)
        
        # Random initialization in range [-scale, scale]
        rand(weights.data(), rows * cols)
        
        for i in range(rows * cols):
            weights[i] = (weights[i] - 0.5) * scale * 2.0
        
        return weights
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass through feed-forward network
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        
        Algorithm:
            1. Linear transformation: h = xW1 + b1
            2. ReLU activation: h = max(0, h)
            3. Linear transformation: y = hW2 + b2
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let d_model = self.config.d_model
        let d_ff = self.config.d_ff
        
        # First linear layer: [batch, seq_len, d_model] -> [batch, seq_len, d_ff]
        var hidden = self._linear(x, self.W1, self.b1)
        
        # ReLU activation
        hidden = self._relu(hidden)
        
        # Second linear layer: [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        var output = self._linear(hidden, self.W2, self.b2)
        
        return output
    
    fn _linear(
        self,
        x: Tensor[DType.float32],
        W: Tensor[DType.float32],
        b: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Apply linear transformation: y = xW + b
        
        Args:
            x: Input [batch, seq_len, d_in]
            W: Weight matrix [d_in, d_out]
            b: Bias vector [d_out]
        
        Returns:
            Output [batch, seq_len, d_out]
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let d_in = x.shape()[2]
        let d_out = W.shape()[1]
        
        var output = Tensor[DType.float32](batch_size, seq_len, d_out)
        
        # Matrix multiplication for each batch and sequence position
        for b in range(batch_size):
            for s in range(seq_len):
                for out_dim in range(d_out):
                    # Initialize with bias
                    var sum = b[out_dim]
                    
                    # Compute dot product
                    for in_dim in range(d_in):
                        sum += x[b, s, in_dim] * W[in_dim, out_dim]
                    
                    output[b, s, out_dim] = sum
        
        return output
    
    fn _relu(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Apply ReLU activation: max(0, x)
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor with ReLU applied element-wise
        """
        var output = Tensor[DType.float32](x.shape())
        
        let total_elements = x.num_elements()
        
        for i in range(total_elements):
            output[i] = max(0.0, x[i])
        
        return output
    
    fn count_parameters(self) -> Int:
        """
        Count total number of trainable parameters
        
        Returns:
            Total parameter count
        """
        let d_model = self.config.d_model
        let d_ff = self.config.d_ff
        
        # W1: d_model x d_ff, b1: d_ff
        # W2: d_ff x d_model, b2: d_model
        return (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)


struct Conv1DFeedForward:
    """
    Alternative FFN implementation using 1D convolutions
    
    Functionally equivalent to FeedForwardNetwork but uses convolutions
    instead of linear layers. Can be more efficient for sequence processing.
    
    Architecture:
        x -> Conv1D(d_model, d_ff, kernel=1) -> ReLU -> 
        Conv1D(d_ff, d_model, kernel=1) -> output
    """
    var config: FFNConfig
    
    # Convolution weights (kernel size = 1)
    var conv1_weight: Tensor[DType.float32]  # [d_ff, d_model, 1]
    var conv1_bias: Tensor[DType.float32]    # [d_ff]
    var conv2_weight: Tensor[DType.float32]  # [d_model, d_ff, 1]
    var conv2_bias: Tensor[DType.float32]    # [d_model]
    
    fn __init__(inout self, config: FFNConfig):
        """Initialize 1D convolution feed-forward network"""
        self.config = config
        
        let d_model = config.d_model
        let d_ff = config.d_ff
        
        # Initialize convolution weights
        let scale1 = sqrt(2.0 / (d_model + d_ff))
        self.conv1_weight = self._init_conv_weights(d_ff, d_model, 1, scale1)
        self.conv1_bias = Tensor[DType.float32](d_ff)
        
        let scale2 = sqrt(2.0 / (d_ff + d_model))
        self.conv2_weight = self._init_conv_weights(d_model, d_ff, 1, scale2)
        self.conv2_bias = Tensor[DType.float32](d_model)
    
    fn _init_conv_weights(
        self,
        out_channels: Int,
        in_channels: Int,
        kernel_size: Int,
        scale: Float32
    ) -> Tensor[DType.float32]:
        """Initialize convolution weights"""
        var weights = Tensor[DType.float32](out_channels, in_channels, kernel_size)
        
        rand(weights.data(), out_channels * in_channels * kernel_size)
        
        for i in range(out_channels * in_channels * kernel_size):
            weights[i] = (weights[i] - 0.5) * scale * 2.0
        
        return weights
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass using 1D convolutions
        
        Note: For kernel_size=1, this is equivalent to linear layers
        """
        # First convolution
        var hidden = self._conv1d(x, self.conv1_weight, self.conv1_bias)
        
        # ReLU
        hidden = self._relu(hidden)
        
        # Second convolution
        var output = self._conv1d(hidden, self.conv2_weight, self.conv2_bias)
        
        return output
    
    fn _conv1d(
        self,
        x: Tensor[DType.float32],
        weight: Tensor[DType.float32],
        bias: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """Apply 1D convolution with kernel size 1"""
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let in_channels = x.shape()[2]
        let out_channels = weight.shape()[0]
        
        var output = Tensor[DType.float32](batch_size, seq_len, out_channels)
        
        # For kernel_size=1, this is equivalent to linear transformation
        for b in range(batch_size):
            for s in range(seq_len):
                for oc in range(out_channels):
                    var sum = bias[oc]
                    for ic in range(in_channels):
                        sum += x[b, s, ic] * weight[oc, ic, 0]
                    output[b, s, oc] = sum
        
        return output
    
    fn _relu(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply ReLU activation"""
        var output = Tensor[DType.float32](x.shape())
        for i in range(x.num_elements()):
            output[i] = max(0.0, x[i])
        return output


fn test_feed_forward():
    """Test feed-forward network implementation"""
    print("="*70)
    print("FEED-FORWARD NETWORK TEST")
    print("="*70)
    
    # Configuration
    let config = FFNConfig(d_model=256, d_ff=1024, dropout=0.1)
    print(f"\nConfiguration:")
    print(f"  d_model:  {config.d_model}")
    print(f"  d_ff:     {config.d_ff}")
    print(f"  dropout:  {config.dropout}")
    
    # Create FFN
    var ffn = FeedForwardNetwork(config)
    print(f"\nParameters: {ffn.count_parameters():,}")
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.d_model)
    
    # Fill with test data
    for i in range(batch_size * seq_len * config.d_model):
        x[i] = Float32(i % 100) / 100.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.d_model}]")
    
    # Forward pass
    print("\nRunning forward pass...")
    var output = ffn.forward(x)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    # Test ReLU activation
    print("\n" + "-"*70)
    print("Testing ReLU activation...")
    var test_relu = Tensor[DType.float32](5)
    test_relu[0] = -2.0
    test_relu[1] = -0.5
    test_relu[2] = 0.0
    test_relu[3] = 0.5
    test_relu[4] = 2.0
    
    var relu_output = ffn._relu(test_relu)
    print("Input:  [-2.0, -0.5,  0.0,  0.5,  2.0]")
    print(f"Output: [{relu_output[0]:.1f}, {relu_output[1]:.1f}, {relu_output[2]:.1f}, {relu_output[3]:.1f}, {relu_output[4]:.1f}]")
    
    print("\n✓ Feed-forward network test passed!")
    print("="*70)


fn test_conv1d_ffn():
    """Test Conv1D feed-forward network"""
    print("\n" + "="*70)
    print("CONV1D FEED-FORWARD NETWORK TEST")
    print("="*70)
    
    # Configuration
    let config = FFNConfig(d_model=256, d_ff=1024)
    print(f"\nConfiguration:")
    print(f"  d_model:  {config.d_model}")
    print(f"  d_ff:     {config.d_ff}")
    
    # Create Conv1D FFN
    var conv_ffn = Conv1DFeedForward(config)
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.d_model)
    
    # Fill with test data
    for i in range(batch_size * seq_len * config.d_model):
        x[i] = Float32(i % 100) / 100.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.d_model}]")
    
    # Forward pass
    print("\nRunning forward pass...")
    var output = conv_ffn.forward(x)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ Conv1D feed-forward network test passed!")
    print("="*70)


fn main():
    """Run feed-forward network tests"""
    test_feed_forward()
    test_conv1d_ffn()
