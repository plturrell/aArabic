/// Transformer Decoder for M2M100

use burn::prelude::*;
use burn::nn::{LayerNorm, LayerNormConfig, Dropout, DropoutConfig};
use burn::tensor::backend::Backend;
use crate::model::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
use crate::model::encoder::{FeedForward, FeedForwardConfig};

#[derive(Module, Debug)]
pub struct TransformerDecoderLayer<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    cross_attn: MultiHeadAttention<B>,
    feed_forward: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct TransformerDecoderLayerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

impl TransformerDecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerDecoderLayer<B> {
        TransformerDecoderLayer {
            self_attn: MultiHeadAttentionConfig {
                d_model: self.d_model,
                num_heads: self.num_heads,
                dropout: self.dropout,
            }.init(device),
            cross_attn: MultiHeadAttentionConfig {
                d_model: self.d_model,
                num_heads: self.num_heads,
                dropout: self.dropout,
            }.init(device),
            feed_forward: FeedForwardConfig {
                d_model: self.d_model,
                d_ff: self.d_ff,
                dropout: self.dropout,
            }.init(device),
            norm1: LayerNormConfig::new(self.d_model).init(device),
            norm2: LayerNormConfig::new(self.d_model).init(device),
            norm3: LayerNormConfig::new(self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> TransformerDecoderLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        self_mask: Option<Tensor<B, 3>>,
        cross_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        // Self-attention (masked for autoregressive)
        let attn_output = self.self_attn.forward(x.clone(), x.clone(), x.clone(), self_mask);
        let attn_output = self.dropout.forward(attn_output);
        let x = self.norm1.forward(x + attn_output);

        // Cross-attention to encoder output
        let cross_output = self.cross_attn.forward(
            x.clone(),
            encoder_output.clone(),
            encoder_output,
            cross_mask,
        );
        let cross_output = self.dropout.forward(cross_output);
        let x = self.norm2.forward(x + cross_output);

        // Feed-forward
        let ff_output = self.feed_forward.forward(x.clone());
        let ff_output = self.dropout.forward(ff_output);
        self.norm3.forward(x + ff_output)
    }
}

#[derive(Module, Debug)]
pub struct TransformerDecoder<B: Backend> {
    layers: Vec<TransformerDecoderLayer<B>>,
    norm: LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct TransformerDecoderConfig {
    pub num_layers: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

impl TransformerDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerDecoder<B> {
        let layer_config = TransformerDecoderLayerConfig {
            d_model: self.d_model,
            num_heads: self.num_heads,
            d_ff: self.d_ff,
            dropout: self.dropout,
        };

        let layers = (0..self.num_layers)
            .map(|_| layer_config.init(device))
            .collect();

        TransformerDecoder {
            layers,
            norm: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> TransformerDecoder<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        self_mask: Option<Tensor<B, 3>>,
        cross_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let mut output = x;
        
        for layer in &self.layers {
            output = layer.forward(
                output,
                encoder_output.clone(),
                self_mask.clone(),
                cross_mask.clone(),
            );
        }
        
        self.norm.forward(output)
    }

    /// Create causal mask for autoregressive generation
    pub fn create_causal_mask(seq_len: usize, device: &B::Device) -> Tensor<B, 3> {
        // Create upper triangular matrix of -inf for masking future positions
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        
        Tensor::<B, 2>::from_floats(mask_data.as_slice(), device)
            .reshape([1, seq_len, seq_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_decoder_layer() {
        let device = Default::default();
        let config = TransformerDecoderLayerConfig {
            d_model: 512,
            num_heads: 8,
            d_ff: 2048,
            dropout: 0.1,
        };
        
        let layer = config.init(&device);
        
        let decoder_input = Tensor::<TestBackend, 3>::random(
            [2, 10, 512],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        let encoder_output = Tensor::<TestBackend, 3>::random(
            [2, 15, 512],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        let output = layer.forward(decoder_input, encoder_output, None, None);
        assert_eq!(output.dims(), [2, 10, 512]);
    }

    #[test]
    fn test_causal_mask() {
        let device = Default::default();
        let mask = TransformerDecoder::<TestBackend>::create_causal_mask(5, &device);
        
        assert_eq!(mask.dims(), [1, 5, 5]);
        // Verify it's upper triangular
    }
}
