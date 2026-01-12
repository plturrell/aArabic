/// Embedding and Positional Encoding layers

use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    embedding: Embedding<B>,
    scale: f64,
}

#[derive(Config, Debug)]
pub struct TokenEmbeddingConfig {
    pub vocab_size: usize,
    pub d_model: usize,
}

impl TokenEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TokenEmbedding<B> {
        TokenEmbedding {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            scale: (self.d_model as f64).sqrt(),
        }
    }
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(tokens);
        embedded * self.scale
    }
}

#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    encoding: Tensor<B, 2>,
}

#[derive(Config, Debug)]
pub struct PositionalEncodingConfig {
    pub max_len: usize,
    pub d_model: usize,
}

impl PositionalEncodingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionalEncoding<B> {
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        let mut pe_data = vec![0.0; self.max_len * self.d_model];
        
        for pos in 0..self.max_len {
            for i in 0..self.d_model / 2 {
                let angle = pos as f64 / 10000_f64.powf(2.0 * i as f64 / self.d_model as f64);
                pe_data[pos * self.d_model + 2 * i] = angle.sin() as f32;
                pe_data[pos * self.d_model + 2 * i + 1] = angle.cos() as f32;
            }
        }
        
        let encoding = Tensor::<B, 2>::from_floats(
            pe_data.as_slice(),
            device
        ).reshape([self.max_len, self.d_model]);
        
        PositionalEncoding { encoding }
    }
}

impl<B: Backend> PositionalEncoding<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, seq_len, _d_model] = x.dims();
        
        // Get positional encodings for sequence length
        let pos_enc = self.encoding
            .clone()
            .slice([0..seq_len])
            .unsqueeze_dim(0); // Add batch dimension
        
        x + pos_enc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_token_embedding() {
        let device = Default::default();
        let config = TokenEmbeddingConfig {
            vocab_size: 128112, // M2M100 vocab size
            d_model: 512,
        };
        
        let embedding = config.init(&device);
        
        // Test input: [batch=2, seq=5]
        let tokens = Tensor::<TestBackend, 2>::from_data(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            &device,
        );
        
        let output = embedding.forward(tokens);
        assert_eq!(output.dims(), [2, 5, 512]);
    }

    #[test]
    fn test_positional_encoding() {
        let device = Default::default();
        let config = PositionalEncodingConfig {
            max_len: 1024,
            d_model: 512,
        };
        
        let pos_enc = config.init(&device);
        
        // Test input: [batch=2, seq=10, d_model=512]
        let input = Tensor::<TestBackend, 3>::zeros([2, 10, 512], &device);
        
        let output = pos_enc.forward(input);
        assert_eq!(output.dims(), [2, 10, 512]);
    }
}
