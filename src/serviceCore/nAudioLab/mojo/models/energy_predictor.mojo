"""
Energy Predictor for FastSpeech2

Predicts frame energy from encoder output.
Uses the same architecture as duration and pitch predictors.
"""

from tensor import Tensor
from memory import memset_zero
from random import rand
import math


struct EnergyPredictor:
    """
    Energy Predictor for FastSpeech2.
    
    Predicts frame energy using convolutional layers.
    Shares architecture with DurationPredictor and PitchPredictor.
    
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
    var conv1_weights: Tensor[DType.float32]
    var conv1_bias: Tensor[DType.float32]
    var conv2_weights: Tensor[DType.float32]
    var conv2_bias: Tensor[DType.float32]
    var layer_norm1_gamma: Tensor[DType.float32]
    var layer_norm1_beta: Tensor[DType.float32]
    var layer_norm2_gamma: Tensor[DType.float32]
    var layer_norm2_beta: Tensor[DType.float32]
    var linear: Tensor[DType.float32]
    var linear_bias: Tensor[DType.float32]
    var d_model: Int
    var kernel_size: Int
    var dropout_p: Float32
    var training: Bool
    var eps: Float32
    
    fn __init__(inout self, d_model: Int = 256, kernel_size: Int = 3, dropout: Float32 = 0.1):
        """
        Initialize energy predictor.
        
        Args:
            d_model: Model dimension (default: 256)
            kernel_size: Convolution kernel size (default: 3)
            dropout: Dropout probability (default: 0.1)
        """
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dropout_p = dropout
        self.training = True
        self.eps = 1e-5
        
        let padding = (kernel_size - 1) // 2
        let fan_in = d_model * kernel_size
        let std_dev = math.sqrt(2.0 / Float32(fan_in))
        
        # Conv1 weights
        self.conv1_weights = Tensor[DType.float32](d_model, d_model, kernel_size)
        self.conv1_bias = Tensor[DType.float32](d_model)
        for i in range(self.conv1_weights.num_elements()):
            self.conv1_weights[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
        memset_zero(self.conv1_bias.data(), d_model)
        
        # Conv2 weights
        self.conv2_weights = Tensor[DType.float32](d_model, d_model, kernel_size)
        self.conv2_bias = Tensor[DType.float32](d_model)
        for i in range(self.conv2_weights.num_elements()):
            self.conv2_weights[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
        memset_zero(self.conv2_bias.data(), d_model)
        
        # Layer norm parameters
        self.layer_norm1_gamma = Tensor[DType.float32](d_model)
        self.layer_norm1_beta = Tensor[DType.float32](d_model)
        self.layer_norm2_gamma = Tensor[DType.float32](d_model)
        self.layer_norm2_beta = Tensor[DType.float32](d_model)
        
        for i in range(d_model):
            self.layer_norm1_gamma[i] = 1.0
            self.layer_norm1_beta[i] = 0.0
            self.layer_norm2_gamma[i] = 1.0
            self.layer_norm2_beta[i] = 0.0
        
        # Linear projection
        self.linear = Tensor[DType.float32](1, d_model)
        self.linear_bias = Tensor[DType.float32](1)
        
        let linear_std = math.sqrt(2.0 / Float32(d_model))
        for i in range(d_model):
            self.linear[i] = (rand[DType.float32]() - 0.5) * linear_std * 2.0
        self.linear_bias[0] = 0.0
    
    fn forward(self, encoder_output: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Predict energy from encoder output.
        
        Args:
            encoder_output: Encoder output [batch, seq_len, d_model]
        
        Returns:
            Predicted energy values [batch, seq_len]
        """
        let batch_size = encoder_output.shape()[0]
        let seq_len = encoder_output.shape()[1]
        
        # Transpose to [batch, d_model, seq_len] for conv
        var x = self._transpose_forward(encoder_output, batch_size, seq_len)
        
        # First conv block
        x = self._conv1d(x, self.conv1_weights, self.conv1_bias, batch_size, seq_len)
        x = self._relu(x)
        
        # Transpose back for layer norm
        var x_norm = self._transpose_back(x, batch_size, seq_len)
        x_norm = self._layer_norm(x_norm, self.layer_norm1_gamma, self.layer_norm1_beta, batch_size, seq_len)
        x_norm = self._dropout(x_norm)
        
        # Transpose for second conv
        x = self._transpose_forward(x_norm, batch_size, seq_len)
        
        # Second conv block
        x = self._conv1d(x, self.conv2_weights, self.conv2_bias, batch_size, seq_len)
        x = self._relu(x)
        
        # Transpose back
        x_norm = self._transpose_back(x, batch_size, seq_len)
        x_norm = self._layer_norm(x_norm, self.layer_norm2_gamma, self.layer_norm2_beta, batch_size, seq_len)
        x_norm = self._dropout(x_norm)
        
        # Linear projection to energy
        var energy = Tensor[DType.float32](batch_size, seq_len)
        for b in range(batch_size):
            for t in range(seq_len):
                var sum_val: Float32 = 0.0
                for d in range(self.d_model):
                    sum_val += x_norm[b * seq_len * self.d_model + t * self.d_model + d] * self.linear[d]
                energy[b * seq_len + t] = sum_val + self.linear_bias[0]
        
        return energy
    
    fn _conv1d(self, x: Tensor[DType.float32], weights: Tensor[DType.float32], 
               bias: Tensor[DType.float32], batch_size: Int, seq_len: Int) -> Tensor[DType.float32]:
        """Apply 1D convolution with padding."""
        let padding = (self.kernel_size - 1) // 2
        var output = Tensor[DType.float32](batch_size, self.d_model, seq_len)
        memset_zero(output.data(), output.num_elements())
        
        for b in range(batch_size):
            for out_c in range(self.d_model):
                for t in range(seq_len):
                    var sum_val: Float32 = 0.0
                    for in_c in range(self.d_model):
                        for k in range(self.kernel_size):
                            let input_t = t + k - padding
                            if input_t >= 0 and input_t < seq_len:
                                sum_val += x[b * self.d_model * seq_len + in_c * seq_len + input_t] * \
                                          weights[out_c * self.d_model * self.kernel_size + in_c * self.kernel_size + k]
                    output[b * self.d_model * seq_len + out_c * seq_len + t] = sum_val + bias[out_c]
        
        return output
    
    fn _layer_norm(self, x: Tensor[DType.float32], gamma: Tensor[DType.float32],
                   beta: Tensor[DType.float32], batch_size: Int, seq_len: Int) -> Tensor[DType.float32]:
        """Apply layer normalization."""
        var output = Tensor[DType.float32](x.shape())
        
        for b in range(batch_size):
            for t in range(seq_len):
                var mean: Float32 = 0.0
                for d in range(self.d_model):
                    mean += x[b * seq_len * self.d_model + t * self.d_model + d]
                mean /= Float32(self.d_model)
                
                var variance: Float32 = 0.0
                for d in range(self.d_model):
                    let val = x[b * seq_len * self.d_model + t * self.d_model + d]
                    let diff = val - mean
                    variance += diff * diff
                variance /= Float32(self.d_model)
                
                let std_dev = math.sqrt(variance + self.eps)
                for d in range(self.d_model):
                    let idx = b * seq_len * self.d_model + t * self.d_model + d
                    let normalized = (x[idx] - mean) / std_dev
                    output[idx] = gamma[d] * normalized + beta[d]
        
        return output
    
    fn _dropout(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply dropout."""
        if not self.training or self.dropout_p == 0.0:
            return x
        
        var output = Tensor[DType.float32](x.shape())
        let scale = 1.0 / (1.0 - self.dropout_p)
        
        for i in range(x.num_elements()):
            if rand[DType.float32]() > self.dropout_p:
                output[i] = x[i] * scale
            else:
                output[i] = 0.0
        
        return output
    
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
        self.training = False
    
    fn train(inout self):
        """Set to training mode."""
        self.training = True
