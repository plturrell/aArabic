/// Transformer Encoder for M2M100

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Dropout, DropoutConfig};
use burn::tensor::backend::Backend;
use crate::model::attention::{MultiHeadAttention, MultiHeadAttentionConfig};

#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    feed_forward: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct TransformerEncoderLayerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

impl TransformerEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEncoderLayer<B> {
        TransformerEncoderLayer {
            self_attn: MultiHeadAttentionConfig {
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
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> TransformerEncoderLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        // Self-attention with residual
        let attn_output = self.self_attn.forward(x.clone(), x.clone(), x.clone(), mask);
        let attn_output = self.dropout.forward(attn_output);
        let x = self.norm1.forward(x + attn_output);

        // Feed-forward with residual
        let ff_output = self.feed_forward.forward(x.clone());
        let ff_output = self.dropout.forward(ff_output);
        self.norm2.forward(x + ff_output)
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            linear1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            linear2: LinearConfig::new(self.d_ff, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    layers: Vec<TransformerEncoderLayer<B>>,
    norm: LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct TransformerEncoderConfig {
    pub num_layers: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

impl TransformerEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEncoder<B> {
        let layer_config = TransformerEncoderLayerConfig {
            d_model: self.d_model,
            num_heads: self.num_heads,
            d_ff: self.d_ff,
            dropout: self.dropout,
        };

        let layers = (0..self.num_layers)
            .map(|_| layer_config.init(device))
            .collect();

        TransformerEncoder {
            layers,
            norm: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> TransformerEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let mut output = x;
        
        for layer in &self.layers {
            output = layer.forward(output, mask.clone());
        }
        
        self.norm.forward(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_encoder_layer() {
        let device = Default::default();
        let config = TransformerEncoderLayerConfig {
            d_model: 512,
            num_heads: 8,
            d_ff: 2048,
            dropout: 0.1,
        };
        
        let layer = config.init(&device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 10, 512],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        let output = layer.forward(input, None);
        assert_eq!(output.dims(), [2, 10, 512]);
    }
}
