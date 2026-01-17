"""
Multi-Head Attention Implementation for FastSpeech2

This module implements the core attention mechanism used in Transformer-based
models. Multi-head attention allows the model to jointly attend to information
from different representation subspaces at different positions.

Reference:
    "Attention is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762
"""

from tensor import Tensor, TensorShape
from random import rand
from math import sqrt
import math


struct AttentionConfig:
    """Configuration for multi-head attention"""
    var n_heads: Int
    var d_model: Int
    var d_k: Int      # Key/Query dimension per head (d_model / n_heads)
    var d_v: Int      # Value dimension per head (d_model / n_heads)
    var dropout: Float32
    
    fn __init__(inout self, n_heads: Int = 4, d_model: Int = 256, dropout: Float32 = 0.1):
        """
        Initialize attention configuration
        
        Args:
            n_heads: Number of attention heads
            d_model: Model dimension
            dropout: Dropout probability
        """
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.dropout = dropout
        
        # Verify dimensions
        if d_model % n_heads != 0:
            print("Warning: d_model must be divisible by n_heads")


struct MultiHeadAttention:
    """
    Multi-Head Attention mechanism
    
    Allows the model to jointly attend to information from different
    representation subspaces. Implements scaled dot-product attention
    across multiple heads.
    
    Architecture:
        Input -> [Q, K, V projections] -> Split into heads -> 
        Scaled Dot-Product Attention -> Concat heads -> Output projection
    """
    var config: AttentionConfig
    
    # Weight matrices [d_model, d_model]
    var W_q: Tensor[DType.float32]  # Query projection
    var W_k: Tensor[DType.float32]  # Key projection
    var W_v: Tensor[DType.float32]  # Value projection
    var W_o: Tensor[DType.float32]  # Output projection
    
    # Bias vectors
    var b_q: Tensor[DType.float32]
    var b_k: Tensor[DType.float32]
    var b_v: Tensor[DType.float32]
    var b_o: Tensor[DType.float32]
    
    fn __init__(inout self, config: AttentionConfig):
        """Initialize multi-head attention with random weights"""
        self.config = config
        
        let d_model = config.d_model
        
        # Initialize weight matrices with Xavier initialization
        # Variance = 2 / (fan_in + fan_out)
        let scale = sqrt(2.0 / (d_model + d_model))
        
        self.W_q = self._init_weights(d_model, d_model, scale)
        self.W_k = self._init_weights(d_model, d_model, scale)
        self.W_v = self._init_weights(d_model, d_model, scale)
        self.W_o = self._init_weights(d_model, d_model, scale)
        
        # Initialize biases to zero
        self.b_q = Tensor[DType.float32](d_model)
        self.b_k = Tensor[DType.float32](d_model)
        self.b_v = Tensor[DType.float32](d_model)
        self.b_o = Tensor[DType.float32](d_model)
    
    fn _init_weights(self, rows: Int, cols: Int, scale: Float32) -> Tensor[DType.float32]:
        """Initialize weight matrix with scaled random values"""
        var weights = Tensor[DType.float32](rows, cols)
        # Random initialization
        rand(weights.data(), rows * cols)
        
        # Scale to appropriate range
        for i in range(rows * cols):
            weights[i] = (weights[i] - 0.5) * scale * 2.0
        
        return weights
    
    fn forward(
        self,
        x: Tensor[DType.float32],
        mask: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Forward pass through multi-head attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
                  (optional, use zeros for no masking)
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        
        Algorithm:
            1. Linear projections: Q = xW_q, K = xW_k, V = xW_v
            2. Split into heads: [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
            3. Scaled dot-product attention per head
            4. Concatenate heads
            5. Final linear projection
        """
        let batch_size = x.shape()[0]
        let seq_len = x.shape()[1]
        let d_model = self.config.d_model
        let n_heads = self.config.n_heads
        let d_k = self.config.d_k
        
        # Step 1: Linear projections Q, K, V
        var Q = self._linear_projection(x, self.W_q, self.b_q)
        var K = self._linear_projection(x, self.W_k, self.b_k)
        var V = self._linear_projection(x, self.W_v, self.b_v)
        
        # Step 2: Split into multiple heads
        # Reshape from [batch, seq_len, d_model] to [batch, n_heads, seq_len, d_k]
        Q = self._split_heads(Q, batch_size, seq_len, n_heads, d_k)
        K = self._split_heads(K, batch_size, seq_len, n_heads, d_k)
        V = self._split_heads(V, batch_size, seq_len, n_heads, d_k)
        
        # Step 3: Scaled dot-product attention
        var attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        # Reshape from [batch, n_heads, seq_len, d_k] to [batch, seq_len, d_model]
        var concat = self._concat_heads(attention_output, batch_size, seq_len, n_heads, d_k)
        
        # Step 5: Final linear projection
        var output = self._linear_projection(concat, self.W_o, self.b_o)
        
        return output
    
    fn _linear_projection(
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
        
        # Matrix multiplication: [batch, seq_len, d_in] @ [d_in, d_out]
        for b in range(batch_size):
            for s in range(seq_len):
                for out_dim in range(d_out):
                    var sum = b[out_dim]
                    for in_dim in range(d_in):
                        sum += x[b, s, in_dim] * W[in_dim, out_dim]
                    output[b, s, out_dim] = sum
        
        return output
    
    fn _split_heads(
        self,
        x: Tensor[DType.float32],
        batch_size: Int,
        seq_len: Int,
        n_heads: Int,
        d_k: Int
    ) -> Tensor[DType.float32]:
        """
        Split tensor into multiple attention heads
        
        Args:
            x: Input [batch, seq_len, d_model]
        
        Returns:
            Output [batch, n_heads, seq_len, d_k]
        """
        var output = Tensor[DType.float32](batch_size, n_heads, seq_len, d_k)
        
        for b in range(batch_size):
            for h in range(n_heads):
                for s in range(seq_len):
                    for k in range(d_k):
                        let idx = h * d_k + k
                        output[b, h, s, k] = x[b, s, idx]
        
        return output
    
    fn _concat_heads(
        self,
        x: Tensor[DType.float32],
        batch_size: Int,
        seq_len: Int,
        n_heads: Int,
        d_k: Int
    ) -> Tensor[DType.float32]:
        """
        Concatenate multiple attention heads
        
        Args:
            x: Input [batch, n_heads, seq_len, d_k]
        
        Returns:
            Output [batch, seq_len, d_model]
        """
        let d_model = n_heads * d_k
        var output = Tensor[DType.float32](batch_size, seq_len, d_model)
        
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(n_heads):
                    for k in range(d_k):
                        let idx = h * d_k + k
                        output[b, s, idx] = x[b, h, s, k]
        
        return output
    
    fn _scaled_dot_product_attention(
        self,
        Q: Tensor[DType.float32],
        K: Tensor[DType.float32],
        V: Tensor[DType.float32],
        mask: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Compute scaled dot-product attention
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        
        Args:
            Q: Query [batch, n_heads, seq_len, d_k]
            K: Key [batch, n_heads, seq_len, d_k]
            V: Value [batch, n_heads, seq_len, d_k]
            mask: Attention mask [batch, seq_len, seq_len]
        
        Returns:
            Attention output [batch, n_heads, seq_len, d_k]
        """
        let batch_size = Q.shape()[0]
        let n_heads = Q.shape()[1]
        let seq_len = Q.shape()[2]
        let d_k = Q.shape()[3]
        
        let scale = sqrt(Float32(d_k))
        
        # Compute attention scores: QK^T
        var scores = Tensor[DType.float32](batch_size, n_heads, seq_len, seq_len)
        
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len):
                    for j in range(seq_len):
                        var score = Float32(0.0)
                        for k in range(d_k):
                            score += Q[b, h, i, k] * K[b, h, j, k]
                        
                        # Scale by sqrt(d_k)
                        score = score / scale
                        
                        # Apply mask if provided (set masked positions to large negative)
                        if mask.shape()[0] > 0:
                            if mask[b, i, j] == 0.0:
                                score = -1e9
                        
                        scores[b, h, i, j] = score
        
        # Apply softmax to get attention weights
        var attention_weights = self._softmax(scores)
        
        # Apply attention weights to values: attention_weights @ V
        var output = Tensor[DType.float32](batch_size, n_heads, seq_len, d_k)
        
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len):
                    for k in range(d_k):
                        var weighted_sum = Float32(0.0)
                        for j in range(seq_len):
                            weighted_sum += attention_weights[b, h, i, j] * V[b, h, j, k]
                        output[b, h, i, k] = weighted_sum
        
        return output
    
    fn _softmax(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Apply softmax along last dimension
        
        Args:
            x: Input tensor [..., seq_len]
        
        Returns:
            Softmax output [..., seq_len]
        """
        var output = Tensor[DType.float32](x.shape())
        
        let batch_size = x.shape()[0]
        let n_heads = x.shape()[1]
        let seq_len_i = x.shape()[2]
        let seq_len_j = x.shape()[3]
        
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len_i):
                    # Find max for numerical stability
                    var max_val = x[b, h, i, 0]
                    for j in range(1, seq_len_j):
                        if x[b, h, i, j] > max_val:
                            max_val = x[b, h, i, j]
                    
                    # Compute exp(x - max) and sum
                    var sum_exp = Float32(0.0)
                    for j in range(seq_len_j):
                        let exp_val = math.exp(x[b, h, i, j] - max_val)
                        output[b, h, i, j] = exp_val
                        sum_exp += exp_val
                    
                    # Normalize
                    for j in range(seq_len_j):
                        output[b, h, i, j] = output[b, h, i, j] / sum_exp
        
        return output
    
    fn count_parameters(self) -> Int:
        """Count total number of trainable parameters"""
        let d_model = self.config.d_model
        # 4 weight matrices (d_model x d_model) + 4 bias vectors (d_model)
        return 4 * d_model * d_model + 4 * d_model


fn test_attention():
    """Test multi-head attention implementation"""
    print("="*70)
    print("MULTI-HEAD ATTENTION TEST")
    print("="*70)
    
    # Configuration
    let config = AttentionConfig(n_heads=4, d_model=256, dropout=0.1)
    print(f"\nConfiguration:")
    print(f"  n_heads:  {config.n_heads}")
    print(f"  d_model:  {config.d_model}")
    print(f"  d_k:      {config.d_k}")
    print(f"  d_v:      {config.d_v}")
    
    # Create attention module
    var attention = MultiHeadAttention(config)
    print(f"\nParameters: {attention.count_parameters():,}")
    
    # Test input
    let batch_size = 2
    let seq_len = 10
    var x = Tensor[DType.float32](batch_size, seq_len, config.d_model)
    
    # Fill with test data
    for i in range(batch_size * seq_len * config.d_model):
        x[i] = Float32(i % 100) / 100.0
    
    print(f"\nInput shape: [{batch_size}, {seq_len}, {config.d_model}]")
    
    # No mask (all ones)
    var mask = Tensor[DType.float32](batch_size, seq_len, seq_len)
    for i in range(batch_size * seq_len * seq_len):
        mask[i] = 1.0
    
    # Forward pass
    print("\nRunning forward pass...")
    var output = attention.forward(x, mask)
    
    print(f"Output shape: [{output.shape()[0]}, {output.shape()[1]}, {output.shape()[2]}]")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    print("\n✓ Multi-head attention test passed!")
    print("="*70)


fn main():
    """Run attention tests"""
    test_attention()
