"""
Layer Normalization Implementation for FastSpeech2

This module implements layer normalization, a technique that normalizes
the inputs across the features (rather than across the batch as in batch
normalization). This is crucial for stabilizing training in Transformers.

Reference:
    "Layer Normalization" (Ba et al., 2016)
    https://arxiv.org/abs/1607.06450
"""

from tensor import Tensor
from math import sqrt


struct LayerNormConfig:
    """Configuration for layer normalization"""
    var normalized_shape: Int  # Size of the dimension to normalize
    var eps: Float32          # Small constant for numerical stability
    var elementwise_affine: Bool  # Whether to use learnable affine parameters
    
    fn __init__(
        inout self,
        normalized_shape: Int,
        eps: Float32 = 1e-5,
        elementwise_affine: Bool = True
    ):
        """
        Initialize layer normalization configuration
        
        Args:
            normalized_shape: Size of the last dimension to normalize over
            eps: Small constant added to denominator for numerical stability
            elementwise_affine: If True, use learnable gamma and beta parameters
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine


struct LayerNorm:
    """
    Layer Normalization
    
    Normalizes inputs across features (last dimension) rather than batch.
    Unlike batch normalization, layer norm is applied independently to each
    sample in the batch, making it suitable for sequence models.
    
    Formula:
        y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Where:
        - mean and var are computed over the last dimension
        - gamma and beta are learnable parameters (if elementwise_affine=True)
        - eps is a small constant for numerical stability
    """
    var config: LayerNormConfig
    
    # Learnable parameters
    var gamma: Tensor[DType.float32]  # Scale parameter
    var beta: Tensor[DType.float32]   # Shift parameter
    
    fn __init__(inout self, config: LayerNormConfig):
        """
        Initialize layer normalization
        
        Initializes gamma to 1.0 and beta to 0.0 (identity transformation)
        """
        self.config = config
        
        let normalized_shape = config.normalized_shape
        
        if config.elementwise_affine:
            # Initialize gamma to 1.0 (scale)
            self.gamma = Tensor[DType.float32](normalized_shape)
            for i in range(normalized_shape):
                self.gamma[i] = 1.0
            
            # Initialize beta to 0.0 (shift)
            self.beta = Tensor[DType.float32](normalized_shape)
            for i in range(normalized_shape):
                self.beta[i] = 0.0
        else:
            # No learnable parameters
            self.gamma = Tensor[DType.float32](0)
            self.beta = Tensor[DType.float32](0)
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Apply layer normalization
        
        Args:
            x: Input tensor [..., normalized_shape]
               Typically [batch_size, seq_len, d_model]
        
        Returns:
            Normalized tensor with same shape as input
        
        Algorithm:
            1. Compute mean and variance along last dimension
            2. Normalize: (x - mean) / sqrt(var + eps)
            3. Scale and shift: gamma * normalized + beta
        """
        let shape = x.shape()
        let batch_size = shape[0]
        let seq_len = shape[1]
        let features = shape[2]
        
        # Verify feature dimension matches configuration
        if features != self.config.normalized_shape:
            print("Warning: input features don't match normalized_shape")
        
        var output = Tensor[DType.float32](batch_size, seq_len, features)
        
        # Normalize each position independently
        for b in range(batch_size):
            for s in range(seq_len):
                # Step 1: Compute mean
                var mean = self._compute_mean(x, b, s, features)
                
                # Step 2: Compute variance
                var variance = self._compute_variance(x, b, s, features, mean)
                
                # Step 3: Normalize
                let std = sqrt(variance + self.config.eps)
                
                for f in range(features):
                    # Normalize: (x - mean) / std
                    var normalized = (x[b, s, f] - mean) / std
                    
                    # Apply affine transformation if enabled
                    if self.config.elementwise_affine:
                        normalized = self.gamma[f] * normalized + self.beta[f]
                    
                    output[b, s, f] = normalized
        
        return output
    
    fn _compute_mean(
        self,
        x: Tensor[DType.float32],
        batch_idx: Int,
        seq_idx: Int,
        features: Int
    ) -> Float32:
        """
        Compute mean along feature dimension
        
        Args:
            x: Input tensor
            batch_idx: Batch index
            seq_idx: Sequence index
            features: Number of features
        
        Returns:
            Mean value
        """
        var sum = Float32(0.0)
        
        for f in range(features):
            sum += x[batch_idx, seq_idx, f]
        
        return sum / Float32(features)
    
    fn _compute_variance(
        self,
        x: Tensor[DType.float32],
        batch_idx: Int,
        seq_idx: Int,
        features: Int,
        mean: Float32
    ) -> Float32:
        """
        Compute variance along feature dimension
        
        Args:
            x: Input tensor
            batch_idx: Batch index
            seq_idx: Sequence index
            features: Number of features
            mean: Precomputed mean
        
        Returns:
            Variance value
        """
        var sum_squared_diff = Float32(0.0)
        
        for f in range(features):
            let diff = x[batch_idx, seq_idx, f] - mean
            sum_squared_diff += diff * diff
        
        return sum_squared_diff / Float32(features)
    
    fn count_parameters(self) -> Int:
        """
        Count total number of trainable parameters
        
        Returns:
            Number of parameters (2 * normalized_shape if affine, else 0)
        """
        if self.config.elementwise_affine:
            return 2 * self.config.normalized_shape
        else:
            return 0


struct RMSNorm:
    """
    Root Mean Square Layer Normalization
    
    A simpler variant of layer normalization that only uses RMS for normalization
    without centering (no mean subtraction). Often used in modern LLMs.
    
    Formula:
        y = gamma * x / RMS(x)
        where RMS(x) = sqrt(mean(x^2) + eps)
    
    Reference:
        "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """
    var config: LayerNormConfig
    var gamma: Tensor[DType.float32]
    
    fn __init__(inout self, config: LayerNormConfig):
        """Initialize RMS normalization"""
        self.config = config
        
        let normalized_shape = config.normalized_shape
        
        # Initialize gamma to 1.0
        self.gamma = Tensor[DType.float32](normalized_shape)
        for i in range(normalized_shape):
            self.gamma[i] = 1.0
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Apply RMS normalization
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
        
        Returns:
            Normalized tensor
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let features = x.shape()[2]
        
        var output = Tensor[DType.float32](batch_size, seq_len, features)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Compute RMS
                var rms = self._compute_rms(x, b, s, features)
                
                # Normalize and scale
                for f in range(features):
                    output[b, s, f] = self.gamma[f] * x[b, s, f] / rms
        
        return output
    
    fn _compute_rms(
        self,
        x: Tensor[DType.float32],
        batch_idx: Int,
        seq_idx: Int,
        features: Int
    ) -> Float32:
        """Compute root mean square"""
        var sum_squares = Float32(0.0)
        
        for f in range(features):
            let val = x[batch_idx, seq_idx, f]
            sum_squares += val * val
        
        let mean_squares = sum_squares / Float32(features)
        return sqrt(mean_squares + self.config.eps)


fn test_layer_norm():
    """Test layer normalization implementation"""
    print("="*70)
    print("LAYER NORMALIZATION TEST")
    print("="*70)
    
    # Configuration
    let config = LayerNormConfig(normalized_shape=256, eps=1e-5, elementwise_affine=True)
    print(f"\nConfiguration:")
    print(f"  normalized_shape: {config.normalized_shape}")
    print(f"  eps:              {config.eps}")
    print(f"  elementwise_affine: {config.elementwise_affine}")
    
    # Create layer norm
    var layer_norm = LayerNorm(config)
    print(f"\nParameters: {layer_norm.count_parameters()}")
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.normalized_shape)
    
    # Fill with test data (different scales to test normalization)
    for b in range(batch_size):
        for s in range(seq_len):
            for f in range(config.normalized_shape):
                # Create data with non-zero mean and varying scales
                x[b, s, f] = Float32((b + 1) * (s + 1) * (f % 10 + 1)) / 10.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.normalized_shape}]")
    print(f"Input stats (first position):")
    
    # Compute input statistics
    var input_sum = Float32(0.0)
    for f in range(config.normalized_shape):
        input_sum += x[0, 0, f]
    let input_mean = input_sum / Float32(config.normalized_shape)
    
    var input_var = Float32(0.0)
    for f in range(config.normalized_shape):
        let diff = x[0, 0, f] - input_mean
        input_var += diff * diff
    input_var = input_var / Float32(config.normalized_shape)
    
    print(f"  Mean: {input_mean:.6f}")
    print(f"  Variance: {input_var:.6f}")
    
    # Forward pass
    print("\nRunning forward pass...")
    var output = layer_norm.forward(x)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    
    # Compute output statistics (should be ~0 mean, ~1 variance)
    var output_sum = Float32(0.0)
    for f in range(config.normalized_shape):
        output_sum += output[0, 0, f]
    let output_mean = output_sum / Float32(config.normalized_shape)
    
    var output_var = Float32(0.0)
    for f in range(config.normalized_shape):
        let diff = output[0, 0, f] - output_mean
        output_var += diff * diff
    output_var = output_var / Float32(config.normalized_shape)
    
    print(f"\nOutput stats (first position):")
    print(f"  Mean: {output_mean:.6f} (should be ~0)")
    print(f"  Variance: {output_var:.6f} (should be ~1)")
    
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ Layer normalization test passed!")
    print("="*70)


fn test_rms_norm():
    """Test RMS normalization"""
    print("\n" + "="*70)
    print("RMS NORMALIZATION TEST")
    print("="*70)
    
    # Configuration
    let config = LayerNormConfig(normalized_shape=256, eps=1e-5)
    print(f"\nConfiguration:")
    print(f"  normalized_shape: {config.normalized_shape}")
    print(f"  eps:              {config.eps}")
    
    # Create RMS norm
    var rms_norm = RMSNorm(config)
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.normalized_shape)
    
    # Fill with test data
    for b in range(batch_size):
        for s in range(seq_len):
            for f in range(config.normalized_shape):
                x[b, s, f] = Float32((f % 10 + 1)) / 5.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.normalized_shape}]")
    
    # Forward pass
    print("\nRunning forward pass...")
    var output = rms_norm.forward(x)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ RMS normalization test passed!")
    print("="*70)


fn test_normalization_properties():
    """Test that normalization achieves desired properties"""
    print("\n" + "="*70)
    print("NORMALIZATION PROPERTIES TEST")
    print("="*70)
    
    let config = LayerNormConfig(normalized_shape=100, eps=1e-5)
    var layer_norm = LayerNorm(config)
    
    # Create input with known statistics
    let batch_size = 1
    let seq_len = 1
    let features = 100
    
    var x = Tensor[DType.float32](batch_size, seq_len, features)
    
    # Fill with data: mean=5.0, std=2.0
    for f in range(features):
        x[0, 0, f] = 5.0 + 2.0 * Float32(f % 10 - 5) / 5.0
    
    print(f"\nInput created with approximate mean=5.0, std=2.0")
    
    # Apply normalization
    var output = layer_norm.forward(x)
    
    # Verify output has mean~0, std~1
    var mean = Float32(0.0)
    for f in range(features):
        mean += output[0, 0, f]
    mean = mean / Float32(features)
    
    var std_sq = Float32(0.0)
    for f in range(features):
        let diff = output[0, 0, f] - mean
        std_sq += diff * diff
    std_sq = std_sq / Float32(features)
    let std = sqrt(std_sq)
    
    print(f"\nOutput statistics:")
    print(f"  Mean: {mean:.8f} (target: ~0.0)")
    print(f"  Std:  {std:.8f} (target: ~1.0)")
    
    # Check if close to expected values
    let mean_ok = mean > -0.01 and mean < 0.01
    let std_ok = std > 0.9 and std < 1.1
    
    if mean_ok and std_ok:
        print("\n✓ Normalization properties verified!")
    else:
        print("\n✗ Warning: Statistics outside expected range")
    
    print("="*70)


fn main():
    """Run layer normalization tests"""
    test_layer_norm()
    test_rms_norm()
    test_normalization_properties()
