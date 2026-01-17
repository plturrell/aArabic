"""
Duration Predictor for FastSpeech2

Predicts phoneme durations from encoder output.
Uses convolutional layers with layer normalization and dropout.
"""

from tensor import Tensor
from algorithm import vectorize
from memory import memset_zero
from random import rand
import math


struct Conv1D:
    """1D Convolution layer for sequence processing."""
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var kernel_size: Int
    var in_channels: Int
    var out_channels: Int
    var padding: Int
    
    fn __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int, padding: Int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Initialize weights with He initialization
        let fan_in = in_channels * kernel_size
        let std_dev = math.sqrt(2.0 / Float32(fan_in))
        self.weights = Tensor[DType.float32](out_channels, in_channels, kernel_size)
        self.bias = Tensor[DType.float32](out_channels)
        
        # Random initialization
        for i in range(self.weights.num_elements()):
            self.weights[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
        
        memset_zero(self.bias.data(), self.bias.num_elements())
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass for 1D convolution.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
        
        Returns:
            Output tensor [batch, out_channels, seq_len]
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[2]
        let out_len = seq_len + 2 * self.padding - self.kernel_size + 1
        
        var output = Tensor[DType.float32](batch_size, self.out_channels, out_len)
        memset_zero(output.data(), output.num_elements())
        
        # Perform 1D convolution
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for t in range(out_len):
                    var sum_val: Float32 = 0.0
                    
                    for in_c in range(self.in_channels):
                        for k in range(self.kernel_size):
                            let input_t = t + k - self.padding
                            if input_t >= 0 and input_t < seq_len:
                                let x_idx = b * (self.in_channels * seq_len) + in_c * seq_len + input_t
                                let w_idx = out_c * (self.in_channels * self.kernel_size) + in_c * self.kernel_size + k
                                sum_val += x[x_idx] * self.weights[w_idx]
                    
                    output[b * (self.out_channels * out_len) + out_c * out_len + t] = sum_val + self.bias[out_c]
        
        return output


struct LayerNorm:
    """Layer normalization."""
    var gamma: Tensor[DType.float32]
    var beta: Tensor[DType.float32]
    var normalized_shape: Int
    var eps: Float32
    
    fn __init__(inout self, normalized_shape: Int, eps: Float32 = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor[DType.float32](normalized_shape)
        self.beta = Tensor[DType.float32](normalized_shape)
        
        # Initialize gamma to 1, beta to 0
        for i in range(normalized_shape):
            self.gamma[i] = 1.0
            self.beta[i] = 0.0
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply layer normalization."""
        var output = Tensor[DType.float32](x.shape())
        
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        
        # Normalize each sequence element
        for b in range(batch_size):
            for t in range(seq_len):
                # Compute mean
                var mean: Float32 = 0.0
                for d in range(self.normalized_shape):
                    mean += x[b * seq_len * self.normalized_shape + t * self.normalized_shape + d]
                mean /= Float32(self.normalized_shape)
                
                # Compute variance
                var variance: Float32 = 0.0
                for d in range(self.normalized_shape):
                    let val = x[b * seq_len * self.normalized_shape + t * self.normalized_shape + d]
                    let diff = val - mean
                    variance += diff * diff
                variance /= Float32(self.normalized_shape)
                
                # Normalize and scale
                let std_dev = math.sqrt(variance + self.eps)
                for d in range(self.normalized_shape):
                    let idx = b * seq_len * self.normalized_shape + t * self.normalized_shape + d
                    let normalized = (x[idx] - mean) / std_dev
                    output[idx] = self.gamma[d] * normalized + self.beta[d]
        
        return output


struct Dropout:
    """Dropout layer for regularization."""
    var p: Float32
    var training: Bool
    
    fn __init__(inout self, p: Float32 = 0.1):
        self.p = p
        self.training = True
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply dropout."""
        if not self.training or self.p == 0.0:
            return x
        
        var output = Tensor[DType.float32](x.shape())
        let scale = 1.0 / (1.0 - self.p)
        
        for i in range(x.num_elements()):
            if rand[DType.float32]() > self.p:
                output[i] = x[i] * scale
            else:
                output[i] = 0.0
        
        return output
    
    fn eval(inout self):
        """Set to evaluation mode."""
        self.training = False
    
    fn train(inout self):
        """Set to training mode."""
        self.training = True


struct DurationPredictor:
    """
    Duration Predictor for FastSpeech2.
    
    Predicts phoneme durations using convolutional layers.
    Architecture:
        - Conv1D (256 -> 256, kernel=3)
        - ReLU
        - LayerNorm
        - Dropout
        - Conv1D (256 -> 256, kernel=3)
        - ReLU
        - LayerNorm
        - Dropout
        - Linear (256 -> 1)
    """
    var conv1: Conv1D
    var conv2: Conv1D
    var layer_norm1: LayerNorm
    var layer_norm2: LayerNorm
    var dropout1: Dropout
    var dropout2: Dropout
    var linear: Tensor[DType.float32]  # Linear projection weights
    var linear_bias: Tensor[DType.float32]
    var d_model: Int
    
    fn __init__(inout self, d_model: Int = 256, kernel_size: Int = 3, dropout: Float32 = 0.1):
        """
        Initialize duration predictor.
        
        Args:
            d_model: Model dimension (default: 256)
            kernel_size: Convolution kernel size (default: 3)
            dropout: Dropout probability (default: 0.1)
        """
        self.d_model = d_model
        
        # Convolutional layers with padding to preserve sequence length
        let padding = (kernel_size - 1) // 2
        self.conv1 = Conv1D(d_model, d_model, kernel_size, padding)
        self.conv2 = Conv1D(d_model, d_model, kernel_size, padding)
        
        # Layer normalization
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # Linear projection to scalar duration
        self.linear = Tensor[DType.float32](1, d_model)
        self.linear_bias = Tensor[DType.float32](1)
        
        # Initialize linear layer
        let std_dev = math.sqrt(2.0 / Float32(d_model))
        for i in range(d_model):
            self.linear[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
        self.linear_bias[0] = 0.0
    
    fn forward(self, encoder_output: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Predict durations from encoder output.
        
        Args:
            encoder_output: Encoder output [batch, seq_len, d_model]
        
        Returns:
            Predicted durations [batch, seq_len] in log scale
        """
        let batch_size = encoder_output.shape()[0]
        let seq_len = encoder_output.shape()[1]
        
        # Transpose to [batch, d_model, seq_len] for conv
        var x = Tensor[DType.float32](batch_size, self.d_model, seq_len)
        for b in range(batch_size):
            for t in range(seq_len):
                for d in range(self.d_model):
                    x[b * self.d_model * seq_len + d * seq_len + t] = encoder_output[b * seq_len * self.d_model + t * self.d_model + d]
        
        # First conv block
        x = self.conv1.forward(x)
        x = self._relu(x)
        
        # Transpose back for layer norm [batch, seq_len, d_model]
        var x_norm = self._transpose_back(x, batch_size, seq_len)
        x_norm = self.layer_norm1.forward(x_norm)
        x_norm = self.dropout1.forward(x_norm)
        
        # Transpose again for second conv
        x = self._transpose_forward(x_norm, batch_size, seq_len)
        
        # Second conv block
        x = self.conv2.forward(x)
        x = self._relu(x)
        
        # Transpose back
        x_norm = self._transpose_back(x, batch_size, seq_len)
        x_norm = self.layer_norm2.forward(x_norm)
        x_norm = self.dropout2.forward(x_norm)
        
        # Linear projection to durations
        var durations = Tensor[DType.float32](batch_size, seq_len)
        for b in range(batch_size):
            for t in range(seq_len):
                var sum_val: Float32 = 0.0
                for d in range(self.d_model):
                    sum_val += x_norm[b * seq_len * self.d_model + t * self.d_model + d] * self.linear[d]
                durations[b * seq_len + t] = sum_val + self.linear_bias[0]
        
        return durations
    
    fn _relu(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply ReLU activation."""
        var output = Tensor[DType.float32](x.shape())
        for i in range(x.num_elements()):
            output[i] = max(0.0, x[i])
        return output
    
    fn _transpose_forward(self, x: Tensor[DType.float32], batch_size: Int, seq_len: Int) -> Tensor[DType.float32]:
        """Transpose from [batch, seq_len, d_model] to [batch, d_model, seq_len]."""
        var output = Tensor[DType.float32](batch_size, self.d_model, seq_len)
        for b in range(batch_size):
            for t in range(seq_len):
                for d in range(self.d_model):
                    output[b * self.d_model * seq_len + d * seq_len + t] = x[b * seq_len * self.d_model + t * self.d_model + d]
        return output
    
    fn _transpose_back(self, x: Tensor[DType.float32], batch_size: Int, seq_len: Int) -> Tensor[DType.float32]:
        """Transpose from [batch, d_model, seq_len] to [batch, seq_len, d_model]."""
        var output = Tensor[DType.float32](batch_size, seq_len, self.d_model)
        for b in range(batch_size):
            for t in range(seq_len):
                for d in range(self.d_model):
                    output[b * seq_len * self.d_model + t * self.d_model + d] = x[b * self.d_model * seq_len + d * seq_len + t]
        return output
    
    fn eval(inout self):
        """Set to evaluation mode."""
        self.dropout1.eval()
        self.dropout2.eval()
    
    fn train(inout self):
        """Set to training mode."""
        self.dropout1.train()
        self.dropout2.train()
