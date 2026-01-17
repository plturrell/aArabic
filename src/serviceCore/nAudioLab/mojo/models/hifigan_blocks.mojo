"""
HiFiGAN Building Blocks
Multi-Receptive Field Residual Blocks and Upsampling Layers
"""

from tensor import Tensor, TensorShape
from random import rand
from math import sqrt


struct Conv1DLayer:
    """1D Convolution Layer with padding"""
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var dilation: Int
    var in_channels: Int
    var out_channels: Int
    
    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Xavier initialization
        let fan_in = in_channels * kernel_size
        let fan_out = out_channels * kernel_size
        let limit = sqrt(6.0 / Float32(fan_in + fan_out))
        
        self.weights = Tensor[DType.float32](out_channels, in_channels, kernel_size)
        self.bias = Tensor[DType.float32](out_channels)
        
        # Initialize weights
        rand(self.weights.data(), self.weights.num_elements())
        for i in range(self.weights.num_elements()):
            self.weights[i] = self.weights[i] * 2.0 * limit - limit
        
        # Initialize bias to zero
        for i in range(out_channels):
            self.bias[i] = 0.0
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass for 1D convolution
        Input: [batch, in_channels, length]
        Output: [batch, out_channels, output_length]
        """
        let batch_size = x.dim(0)
        let in_len = x.dim(2)
        
        # Calculate output length
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        let output_len = (in_len + 2 * self.padding - effective_kernel) // self.stride + 1
        
        var output = Tensor[DType.float32](batch_size, self.out_channels, output_len)
        
        # Simplified convolution (for demonstration)
        # In production, use optimized BLAS/Accelerate
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for out_idx in range(output_len):
                    var sum_val: Float32 = self.bias[oc]
                    
                    for ic in range(self.in_channels):
                        for k in range(self.kernel_size):
                            let in_idx = out_idx * self.stride + k * self.dilation - self.padding
                            if in_idx >= 0 and in_idx < in_len:
                                sum_val += x[b, ic, in_idx] * self.weights[oc, ic, k]
                    
                    output[b, oc, out_idx] = sum_val
        
        return output


struct ConvTranspose1D:
    """Transposed 1D Convolution for upsampling"""
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var output_padding: Int
    var in_channels: Int
    var out_channels: Int
    
    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 2,
        padding: Int = 0,
        output_padding: Int = 0
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Xavier initialization
        let fan_in = in_channels * kernel_size
        let fan_out = out_channels * kernel_size
        let limit = sqrt(6.0 / Float32(fan_in + fan_out))
        
        self.weights = Tensor[DType.float32](in_channels, out_channels, kernel_size)
        self.bias = Tensor[DType.float32](out_channels)
        
        # Initialize weights
        rand(self.weights.data(), self.weights.num_elements())
        for i in range(self.weights.num_elements()):
            self.weights[i] = self.weights[i] * 2.0 * limit - limit
        
        # Initialize bias to zero
        for i in range(out_channels):
            self.bias[i] = 0.0
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass for transposed convolution
        Input: [batch, in_channels, length]
        Output: [batch, out_channels, upsampled_length]
        """
        let batch_size = x.dim(0)
        let in_len = x.dim(2)
        
        # Calculate output length
        let output_len = (in_len - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        var output = Tensor[DType.float32](batch_size, self.out_channels, output_len)
        
        # Initialize with bias
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(output_len):
                    output[b, oc, i] = self.bias[oc]
        
        # Transposed convolution
        for b in range(batch_size):
            for ic in range(self.in_channels):
                for in_idx in range(in_len):
                    for k in range(self.kernel_size):
                        let out_idx = in_idx * self.stride + k - self.padding
                        if out_idx >= 0 and out_idx < output_len:
                            for oc in range(self.out_channels):
                                output[b, oc, out_idx] += x[b, ic, in_idx] * self.weights[ic, oc, k]
        
        return output


fn leaky_relu(x: Tensor[DType.float32], negative_slope: Float32 = 0.1) -> Tensor[DType.float32]:
    """LeakyReLU activation"""
    var result = Tensor[DType.float32](x.shape())
    
    for i in range(x.num_elements()):
        if x[i] > 0.0:
            result[i] = x[i]
        else:
            result[i] = negative_slope * x[i]
    
    return result


struct ResBlock:
    """Single Residual Block with dilated convolutions"""
    var conv_layers: List[Conv1DLayer]
    var channels: Int
    var kernel_size: Int
    var dilations: List[Int]
    
    fn __init__(inout self, channels: Int, kernel_size: Int):
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = List[Int](1, 3, 5)  # Dilation rates
        self.conv_layers = List[Conv1DLayer]()
        
        # Create conv layers with different dilations
        for i in range(3):
            let dilation = self.dilations[i]
            let padding = (kernel_size - 1) * dilation // 2
            self.conv_layers.append(
                Conv1DLayer(channels, channels, kernel_size, 1, padding, dilation)
            )
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass through residual block
        Input: [batch, channels, length]
        Output: [batch, channels, length]
        """
        var residual = x
        var h = x
        
        # Apply convolutions with LeakyReLU
        for i in range(3):
            h = self.conv_layers[i].forward(h)
            h = leaky_relu(h)
        
        # Add residual connection
        var output = Tensor[DType.float32](x.shape())
        for i in range(x.num_elements()):
            output[i] = x[i] + h[i]
        
        return output


struct MRFResBlock:
    """
    Multi-Receptive Field Residual Block
    Parallel residual blocks with different kernel sizes
    """
    var resblocks: List[ResBlock]
    var kernel_sizes: List[Int]
    var channels: Int
    
    fn __init__(inout self, channels: Int):
        self.channels = channels
        self.kernel_sizes = List[Int](3, 7, 11)  # Multiple receptive fields
        self.resblocks = List[ResBlock]()
        
        # Create parallel residual blocks
        for i in range(3):
            self.resblocks.append(ResBlock(channels, self.kernel_sizes[i]))
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass through MRF block
        Sum outputs from parallel paths
        Input: [batch, channels, length]
        Output: [batch, channels, length]
        """
        # Initialize output
        var output = Tensor[DType.float32](x.shape())
        for i in range(output.num_elements()):
            output[i] = 0.0
        
        # Sum outputs from all parallel blocks
        for i in range(3):
            let block_out = self.resblocks[i].forward(x)
            for j in range(output.num_elements()):
                output[j] += block_out[j]
        
        # Average the outputs
        for i in range(output.num_elements()):
            output[i] /= 3.0
        
        return output


struct UpsampleBlock:
    """
    Upsampling block with transposed convolution and MRF resblocks
    """
    var upsample: ConvTranspose1D
    var mrf_blocks: List[MRFResBlock]
    var in_channels: Int
    var out_channels: Int
    var upsample_rate: Int
    
    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        upsample_rate: Int,
        num_mrf_blocks: Int = 3
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_rate = upsample_rate
        
        # Transposed convolution for upsampling
        let kernel_size = upsample_rate * 2
        let padding = (kernel_size - upsample_rate) // 2
        self.upsample = ConvTranspose1D(
            in_channels,
            out_channels,
            kernel_size,
            upsample_rate,
            padding
        )
        
        # MRF residual blocks
        self.mrf_blocks = List[MRFResBlock]()
        for _ in range(num_mrf_blocks):
            self.mrf_blocks.append(MRFResBlock(out_channels))
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass through upsample block
        Input: [batch, in_channels, length]
        Output: [batch, out_channels, length * upsample_rate]
        """
        # Upsample
        var h = self.upsample.forward(x)
        h = leaky_relu(h)
        
        # Apply MRF blocks
        for i in range(len(self.mrf_blocks)):
            h = self.mrf_blocks[i].forward(h)
        
        return h


fn print_tensor_stats(name: String, x: Tensor[DType.float32]):
    """Helper function to print tensor statistics"""
    var min_val: Float32 = x[0]
    var max_val: Float32 = x[0]
    var sum_val: Float32 = 0.0
    
    for i in range(x.num_elements()):
        let val = x[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
        sum_val += val
    
    let mean_val = sum_val / Float32(x.num_elements())
    
    print(name, "shape:", x.shape(), "min:", min_val, "max:", max_val, "mean:", mean_val)
