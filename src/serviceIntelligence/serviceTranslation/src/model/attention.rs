/// Multi-Head Attention for Transformer models
/// Implements scaled dot-product attention in Burn

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Dropout, DropoutConfig};
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    num_heads: usize,
    head_dim: usize,
    query_proj: Linear<B>,
    key_proj: Linear<B>,
    value_proj: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub dropout: f64,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert_eq!(
            self.d_model % self.num_heads,
            0,
            "d_model must be divisible by num_heads"
        );

        let head_dim = self.d_model / self.num_heads;

        MultiHeadAttention {
            num_heads: self.num_heads,
            head_dim,
            query_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            key_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            value_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Forward pass for self-attention
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = query.dims();

        // Project Q, K, V
        let q = self.query_proj.forward(query);
        let k = self.key_proj.forward(key);
        let v = self.value_proj.forward(value);

        // Reshape for multi-head: [batch, seq, d_model] -> [batch, heads, seq, head_dim]
        let q = self.split_heads(q, batch_size, seq_len);
        let k = self.split_heads(k, batch_size, seq_len);
        let v = self.split_heads(v, batch_size, seq_len);

        // Scaled dot-product attention
        let attn_output = self.scaled_dot_product_attention(q, k, v, mask);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        let attn_output = self.merge_heads(attn_output, batch_size, seq_len);

        // Output projection
        self.out_proj.forward(attn_output)
    }

    fn split_heads(&self, x: Tensor<B, 3>, batch_size: usize, seq_len: usize) -> Tensor<B, 4> {
        // Reshape [batch, seq, d_model] to [batch, seq, heads, head_dim]
        // then transpose to [batch, heads, seq, head_dim]
        x.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2)
    }

    fn merge_heads(&self, x: Tensor<B, 4>, batch_size: usize, seq_len: usize) -> Tensor<B, 3> {
        // Reverse of split_heads
        x.swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.num_heads * self.head_dim])
    }

    fn scaled_dot_product_attention(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 4> {
        // Q @ K^T / sqrt(d_k)
        let d_k = self.head_dim as f64;
        let scores = query.matmul(key.transpose()) / d_k.sqrt();

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            let mask_expanded = m.unsqueeze_dim(1); // Add head dimension
            scores + mask_expanded * -1e9
        } else {
            scores
        };

        // Softmax
        let attn_weights = burn::tensor::activation::softmax(scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        // Weighted sum of values
        attn_weights.matmul(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_attention_creation() {
        let device = Default::default();
        let config = MultiHeadAttentionConfig {
            d_model: 512,
            num_heads: 8,
            dropout: 0.1,
        };
        
        let _attn: MultiHeadAttention<TestBackend> = config.init(&device);
    }

    #[test]
    fn test_attention_forward() {
        let device = Default::default();
        let config = MultiHeadAttentionConfig {
            d_model: 512,
            num_heads: 8,
            dropout: 0.0, // No dropout for testing
        };
        
        let attn = config.init(&device);
        
        // Create dummy input [batch=2, seq=10, d_model=512]
        let input = Tensor::<TestBackend, 3>::random(
            [2, 10, 512],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        let output = attn.forward(input.clone(), input.clone(), input, None);
        
        // Check output shape
        assert_eq!(output.dims(), [2, 10, 512]);
    }
}
