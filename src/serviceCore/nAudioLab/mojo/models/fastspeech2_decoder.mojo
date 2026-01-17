"""
FastSpeech2 Decoder

Transforms variance-adapted representations into mel-spectrograms.
Uses FFT blocks (Feed-Forward Transformer) for decoding.
"""

from tensor import Tensor
from memory import memset_zero
from random import rand
import math


struct FastSpeech2Decoder:
    """
    FastSpeech2 Decoder using FFT blocks.
    
    Architecture:
        - 4 FFT (Feed-Forward Transformer) blocks
        - Final linear projection: 256 → 128 mel bins
    
    Takes variance-adapted encoder output and generates mel-spectrograms.
    """
    var fft_blocks: List[FFTBlock]
    var mel_linear_weights: Tensor[DType.float32]
    var mel_linear_bias: Tensor[DType.float32]
    var d_model: Int
    var n_layers: Int
    var n_mels: Int
    
    fn __init__(
        inout self,
        d_model: Int = 256,
        n_heads: Int = 4,
        d_ff: Int = 1024,
        n_layers: Int = 4,
        n_mels: Int = 128,
        kernel_size: Int = 9,
        dropout: Float32 = 0.1
    ):
        """
        Initialize FastSpeech2 decoder.
        
        Args:
            d_model: Model dimension (default: 256)
            n_heads: Number of attention heads (default: 4)
            d_ff: Feed-forward dimension (default: 1024)
            n_layers: Number of FFT blocks (default: 4)
            n_mels: Number of mel bins (default: 128)
            kernel_size: Conv kernel size (default: 9)
            dropout: Dropout probability (default: 0.1)
        """
        from .fft_block import FFTBlock
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_mels = n_mels
        
        # Initialize FFT blocks
        self.fft_blocks = List[FFTBlock]()
        for i in range(n_layers):
            var block = FFTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout
            )
            self.fft_blocks.append(block)
        
        # Mel projection layer: 256 → 128
        self.mel_linear_weights = Tensor[DType.float32](n_mels, d_model)
        self.mel_linear_bias = Tensor[DType.float32](n_mels)
        
        # Initialize mel projection with Xavier initialization
        let std_dev = math.sqrt(2.0 / Float32(d_model + n_mels))
        for i in range(self.mel_linear_weights.num_elements()):
            self.mel_linear_weights[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
        
        memset_zero(self.mel_linear_bias.data(), n_mels)
    
    fn forward(
        self,
        x: Tensor[DType.float32],
        mask: Tensor[DType.bool] = Tensor[DType.bool]()
    ) -> Tensor[DType.float32]:
        """
        Decode variance-adapted representation to mel-spectrogram.
        
        Args:
            x: Variance-adapted encoder output [batch, mel_len, d_model]
            mask: Optional attention mask [batch, mel_len]
        
        Returns:
            Mel-spectrogram [batch, mel_len, n_mels]
        """
        let batch_size = x.shape()[0]
        let mel_len = x.shape()[1]
        
        # Pass through FFT blocks
        var hidden = x
        for i in range(self.n_layers):
            hidden = self.fft_blocks[i].forward(hidden, mask)
        
        # Project to mel-spectrogram
        var mel = Tensor[DType.float32](batch_size, mel_len, self.n_mels)
        
        for b in range(batch_size):
            for t in range(mel_len):
                for m in range(self.n_mels):
                    var sum_val: Float32 = 0.0
                    for d in range(self.d_model):
                        let hidden_idx = b * mel_len * self.d_model + t * self.d_model + d
                        let weight_idx = m * self.d_model + d
                        sum_val += hidden[hidden_idx] * self.mel_linear_weights[weight_idx]
                    
                    mel[b * mel_len * self.n_mels + t * self.n_mels + m] = sum_val + self.mel_linear_bias[m]
        
        return mel
    
    fn eval(inout self):
        """Set to evaluation mode."""
        for i in range(self.n_layers):
            self.fft_blocks[i].eval()
    
    fn train(inout self):
        """Set to training mode."""
        for i in range(self.n_layers):
            self.fft_blocks[i].train()


struct PostNet:
    """
    PostNet for mel-spectrogram refinement (optional).
    
    5 layers of 1D convolution with batch normalization and tanh.
    Adds residual refinement to the predicted mel-spectrogram.
    """
    var conv_layers: List[Conv1DLayer]
    var n_mels: Int
    var n_layers: Int
    
    fn __init__(
        inout self,
        n_mels: Int = 128,
        n_channels: Int = 512,
        kernel_size: Int = 5,
        n_layers: Int = 5
    ):
        """
        Initialize PostNet.
        
        Args:
            n_mels: Number of mel bins
            n_channels: Number of channels in conv layers
            kernel_size: Convolution kernel size
            n_layers: Number of conv layers
        """
        self.n_mels = n_mels
        self.n_layers = n_layers
        self.conv_layers = List[Conv1DLayer]()
        
        # First layer: n_mels → n_channels
        var first_layer = Conv1DLayer(n_mels, n_channels, kernel_size)
        self.conv_layers.append(first_layer)
        
        # Middle layers: n_channels → n_channels
        for i in range(n_layers - 2):
            var layer = Conv1DLayer(n_channels, n_channels, kernel_size)
            self.conv_layers.append(layer)
        
        # Last layer: n_channels → n_mels
        var last_layer = Conv1DLayer(n_channels, n_mels, kernel_size)
        self.conv_layers.append(last_layer)
    
    fn forward(self, mel: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Refine mel-spectrogram using PostNet.
        
        Args:
            mel: Input mel-spectrogram [batch, mel_len, n_mels]
        
        Returns:
            Refined mel-spectrogram [batch, mel_len, n_mels]
        """
        let batch_size = mel.shape()[0]
        let mel_len = mel.shape()[1]
        
        # Transpose to [batch, n_mels, mel_len] for conv
        var x = Tensor[DType.float32](batch_size, self.n_mels, mel_len)
        for b in range(batch_size):
            for t in range(mel_len):
                for m in range(self.n_mels):
                    x[b * self.n_mels * mel_len + m * mel_len + t] = mel[b * mel_len * self.n_mels + t * self.n_mels + m]
        
        # Pass through conv layers
        for i in range(self.n_layers - 1):
            x = self.conv_layers[i].forward(x)
            x = self._tanh(x)
        
        # Last layer without activation
        x = self.conv_layers[self.n_layers - 1].forward(x)
        
        # Transpose back to [batch, mel_len, n_mels]
        var refined = Tensor[DType.float32](batch_size, mel_len, self.n_mels)
        for b in range(batch_size):
            for t in range(mel_len):
                for m in range(self.n_mels):
                    refined[b * mel_len * self.n_mels + t * self.n_mels + m] = x[b * self.n_mels * mel_len + m * mel_len + t]
        
        # Add residual connection
        for i in range(refined.num_elements()):
            refined[i] += mel[i]
        
        return refined
    
    fn _tanh(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply tanh activation."""
        var output = Tensor[DType.float32](x.shape())
        for i in range(x.num_elements()):
            let exp_2x = math.exp(2.0 * x[i])
            output[i] = (exp_2x - 1.0) / (exp_2x + 1.0)
        return output


struct Conv1DLayer:
    """Simple 1D convolution layer for PostNet."""
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    
    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights
        let fan_in = in_channels * kernel_size
        let std_dev = math.sqrt(2.0 / Float32(fan_in))
        self.weights = Tensor[DType.float32](out_channels, in_channels, kernel_size)
        self.bias = Tensor[DType.float32](out_channels)
        
        for i in range(self.weights.num_elements()):
            self.weights[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
        memset_zero(self.bias.data(), out_channels)
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply 1D convolution."""
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[2]
        let padding = (self.kernel_size - 1) // 2
        
        var output = Tensor[DType.float32](batch_size, self.out_channels, seq_len)
        memset_zero(output.data(), output.num_elements())
        
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for t in range(seq_len):
                    var sum_val: Float32 = 0.0
                    
                    for in_c in range(self.in_channels):
                        for k in range(self.kernel_size):
                            let input_t = t + k - padding
                            if input_t >= 0 and input_t < seq_len:
                                let x_idx = b * self.in_channels * seq_len + in_c * seq_len + input_t
                                let w_idx = out_c * self.in_channels * self.kernel_size + in_c * self.kernel_size + k
                                sum_val += x[x_idx] * self.weights[w_idx]
                    
                    output[b * self.out_channels * seq_len + out_c * seq_len + t] = sum_val + self.bias[out_c]
        
        return output
