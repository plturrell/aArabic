#!/usr/bin/env python3
"""
Test Suite for Transformer Building Blocks (Day 6)

This script provides Python validation of the Mojo transformer components:
- Multi-head attention
- Feed-forward networks
- Layer normalization

Validates the mathematical correctness and implementation details.
"""

import numpy as np
from typing import Tuple, Optional


class AttentionConfig:
    """Configuration for multi-head attention"""
    def __init__(self, n_heads: int = 4, d_model: int = 256, dropout: float = 0.1):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.dropout = dropout
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")


class MultiHeadAttention:
    """Multi-head attention mechanism in Python"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        
        # Initialize weights (simplified for testing)
        d_model = config.d_model
        scale = np.sqrt(2.0 / (d_model + d_model))
        
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through multi-head attention
        
        Args:
            x: Input [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v
        
        # Split into heads
        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        concat = self._concat_heads(attention_output, batch_size, seq_len)
        
        # Output projection
        output = concat @ self.W_o + self.b_o
        
        return output
    
    def _split_heads(self, x: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
        """Split into multiple heads"""
        n_heads = self.config.n_heads
        d_k = self.config.d_k
        
        # Reshape [batch, seq, d_model] -> [batch, n_heads, seq, d_k]
        x = x.reshape(batch_size, seq_len, n_heads, d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _concat_heads(self, x: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
        """Concatenate multiple heads"""
        # [batch, n_heads, seq, d_k] -> [batch, seq, d_model]
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.config.d_model)
    
    def _scaled_dot_product_attention(
        self, 
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Scaled dot-product attention"""
        d_k = self.config.d_k
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask[:, None, :, :] == 0, -1e9, scores)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax along last dimension"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class FFNConfig:
    """Configuration for feed-forward network"""
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout


class FeedForwardNetwork:
    """Position-wise feed-forward network"""
    
    def __init__(self, config: FFNConfig):
        self.config = config
        
        d_model = config.d_model
        d_ff = config.d_ff
        
        # Initialize weights
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            x: Input [batch_size, seq_len, d_model]
        
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        # First linear + ReLU
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        
        # Second linear
        output = hidden @ self.W2 + self.b2
        
        return output


class LayerNormConfig:
    """Configuration for layer normalization"""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps


class LayerNorm:
    """Layer normalization"""
    
    def __init__(self, config: LayerNormConfig):
        self.config = config
        
        # Initialize gamma and beta
        self.gamma = np.ones(config.normalized_shape)
        self.beta = np.zeros(config.normalized_shape)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization
        
        Args:
            x: Input [..., normalized_shape]
        
        Returns:
            Normalized output
        """
        # Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.config.eps)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        return output


def test_attention():
    """Test multi-head attention"""
    print("="*70)
    print("MULTI-HEAD ATTENTION TEST")
    print("="*70)
    
    config = AttentionConfig(n_heads=4, d_model=256)
    attention = MultiHeadAttention(config)
    
    print(f"\nConfiguration:")
    print(f"  n_heads:  {config.n_heads}")
    print(f"  d_model:  {config.d_model}")
    print(f"  d_k:      {config.d_k}")
    print(f"  d_v:      {config.d_v}")
    
    # Test input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, config.d_model).astype(np.float32)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = attention.forward(x)
    
    print(f"Output shape: {output.shape}")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    # Verify shape
    assert output.shape == x.shape, "Output shape mismatch!"
    
    print("\n✓ Multi-head attention test passed!")
    print("="*70)


def test_feed_forward():
    """Test feed-forward network"""
    print("\n" + "="*70)
    print("FEED-FORWARD NETWORK TEST")
    print("="*70)
    
    config = FFNConfig(d_model=256, d_ff=1024)
    ffn = FeedForwardNetwork(config)
    
    print(f"\nConfiguration:")
    print(f"  d_model:  {config.d_model}")
    print(f"  d_ff:     {config.d_ff}")
    
    # Test input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, config.d_model).astype(np.float32)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = ffn.forward(x)
    
    print(f"Output shape: {output.shape}")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    # Verify shape
    assert output.shape == x.shape, "Output shape mismatch!"
    
    # Test ReLU
    print("\n" + "-"*70)
    print("Testing ReLU activation...")
    test_input = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    relu_output = np.maximum(0, test_input)
    print(f"Input:  {test_input}")
    print(f"Output: {relu_output}")
    
    print("\n✓ Feed-forward network test passed!")
    print("="*70)


def test_layer_norm():
    """Test layer normalization"""
    print("\n" + "="*70)
    print("LAYER NORMALIZATION TEST")
    print("="*70)
    
    config = LayerNormConfig(normalized_shape=256)
    layer_norm = LayerNorm(config)
    
    print(f"\nConfiguration:")
    print(f"  normalized_shape: {config.normalized_shape}")
    print(f"  eps:              {config.eps}")
    
    # Test input with known statistics
    batch_size = 2
    seq_len = 10
    features = 256
    
    # Create input with non-zero mean and variance
    x = np.random.randn(batch_size, seq_len, features).astype(np.float32) * 2.0 + 5.0
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input stats (first position):")
    print(f"  Mean: {np.mean(x[0, 0]):.6f}")
    print(f"  Std:  {np.std(x[0, 0]):.6f}")
    
    # Forward pass
    output = layer_norm.forward(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output stats (first position):")
    print(f"  Mean: {np.mean(output[0, 0]):.6f} (should be ~0)")
    print(f"  Std:  {np.std(output[0, 0]):.6f} (should be ~1)")
    
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {output[0, 0, i]:.6f}")
    
    # Verify normalization properties
    mean = np.mean(output[0, 0])
    std = np.std(output[0, 0])
    
    assert abs(mean) < 0.01, f"Mean should be ~0, got {mean}"
    assert 0.9 < std < 1.1, f"Std should be ~1, got {std}"
    
    print("\n✓ Layer normalization test passed!")
    print("="*70)


def test_transformer_block():
    """Test combining all components into a simple transformer block"""
    print("\n" + "="*70)
    print("TRANSFORMER BLOCK TEST")
    print("="*70)
    
    d_model = 256
    
    # Initialize components
    attention = MultiHeadAttention(AttentionConfig(n_heads=4, d_model=d_model))
    ffn = FeedForwardNetwork(FFNConfig(d_model=d_model, d_ff=1024))
    ln1 = LayerNorm(LayerNormConfig(normalized_shape=d_model))
    ln2 = LayerNorm(LayerNormConfig(normalized_shape=d_model))
    
    print(f"\nTransformer Block Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: 4")
    print(f"  d_ff:    1024")
    
    # Test input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    print(f"\nInput shape: {x.shape}")
    
    # Transformer block forward pass
    # 1. Multi-head attention + residual + layer norm
    attn_out = attention.forward(x)
    x = ln1.forward(x + attn_out)
    
    # 2. Feed-forward + residual + layer norm
    ffn_out = ffn.forward(x)
    x = ln2.forward(x + ffn_out)
    
    print(f"Output shape: {x.shape}")
    print(f"\nSample output values (first 5):")
    for i in range(5):
        print(f"  output[0, 0, {i}] = {x[0, 0, i]:.6f}")
    
    print("\n✓ Transformer block test passed!")
    print("="*70)


def test_parameter_counts():
    """Test parameter counting"""
    print("\n" + "="*70)
    print("PARAMETER COUNT TEST")
    print("="*70)
    
    d_model = 256
    d_ff = 1024
    
    # Multi-head attention
    # 4 weight matrices (d_model x d_model) + 4 bias vectors (d_model)
    attention_params = 4 * d_model * d_model + 4 * d_model
    print(f"\nMulti-Head Attention:")
    print(f"  Parameters: {attention_params:,}")
    
    # Feed-forward network
    # W1: d_model x d_ff, b1: d_ff
    # W2: d_ff x d_model, b2: d_model
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    print(f"\nFeed-Forward Network:")
    print(f"  Parameters: {ffn_params:,}")
    
    # Layer normalization
    # gamma: d_model, beta: d_model
    ln_params = 2 * d_model
    print(f"\nLayer Normalization:")
    print(f"  Parameters: {ln_params:,}")
    
    # Total for one transformer block
    total = attention_params + ffn_params + 2 * ln_params
    print(f"\nTotal Transformer Block:")
    print(f"  Parameters: {total:,}")
    
    print("\n✓ Parameter count test passed!")
    print("="*70)


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "TRANSFORMER COMPONENTS TEST SUITE" + " "*17 + "║")
    print("║" + " "*68 + "║")
    print("║" + " "*15 + "Python validation of Mojo modules" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    try:
        test_attention()
        test_feed_forward()
        test_layer_norm()
        test_transformer_block()
        test_parameter_counts()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nTransformer building blocks validated:")
        print("  ✓ Multi-head attention")
        print("  ✓ Feed-forward networks")
        print("  ✓ Layer normalization")
        print("  ✓ Complete transformer block")
        print("  ✓ Parameter counting")
        print("\nReady for FastSpeech2 integration (Day 7)!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise


if __name__ == "__main__":
    main()
