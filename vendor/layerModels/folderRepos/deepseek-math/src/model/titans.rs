use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::{CrossEntropyLossConfig, CrossEntropyLoss},
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{backend::{Backend, AutodiffBackend}, Tensor, Int},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

pub mod memory;
use memory::{LongTermMemory, LongTermPattern, SurpriseGate};

#[derive(Config, Debug)]
pub struct TitansMathConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_mem: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    #[config(default = "0.3")]
    pub surprise_threshold: f32,
    #[config(default = "0.9")]
    pub surprise_decay: f32,
}

#[derive(Module, Debug)]
pub struct TitansMathModel<B: Backend> {
    embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    head: Linear<B>,
    memory_proj: Linear<B>, 
    loss: CrossEntropyLoss<B>,
    max_seq_len: usize,
}

impl<B: Backend> TitansMathModel<B> {
    pub fn new(config: &TitansMathConfig, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .init(device);
            
        let transformer = TransformerEncoderConfig::new(config.d_model, config.d_ff, config.n_heads, config.n_layers)
            .init(device);
            
        let head = LinearConfig::new(config.d_model, config.vocab_size)
            .init(device);
            
        let memory_proj = LinearConfig::new(config.d_model, config.d_mem)
            .init(device);

        Self {
            embedding,
            transformer,
            head,
            memory_proj,
            loss: CrossEntropyLossConfig::new().init(device),
            max_seq_len: config.max_seq_len,
        }
    }

    fn generate_causal_mask(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
        let mut mask = Tensor::<B, 2>::zeros([seq_len, seq_len], device);
        for i in 0..seq_len {
            for j in 0..=i {
                mask = mask.slice_assign([i..i+1, j..j+1], Tensor::ones([1, 1], device));
            }
        }
        mask
    }

    pub fn forward(&self, item: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [_batch_size, seq_len] = item.dims();
        let device = item.device();

        let x = self.embedding.forward(item);
        let mask = Self::generate_causal_mask(seq_len, &device).lower_equal(Tensor::ones([1, 1], &device));
        let input = TransformerEncoderInput::new(x).mask_pad(mask);
        
        let encoded = self.transformer.forward(input);
        
        // Memory projection [batch, seq_len, d_mem]
        let mem_embeddings = self.memory_proj.forward(encoded.clone());
        let summary_mem = mem_embeddings.mean_dim(1).squeeze(1);

        let logits = self.head.forward(encoded);
        
        (logits, summary_mem)
    }

    pub fn forward_classification(
        &self,
        item: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (ClassificationOutput<B>, Tensor<B, 2>) {
        let (logits, summary_mem) = self.forward(item);
        
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);

        let loss = self.loss.forward(logits_flat.clone(), targets_flat.clone());

        (
            ClassificationOutput {
                loss,
                output: logits_flat,
                targets: targets_flat,
            },
            summary_mem
        )
    }
}

/// Helper method for consolidation, can be called from outside the model.
pub fn consolidate_memory<B: Backend>(
    gate: &mut SurpriseGate,
    memory: &mut LongTermMemory<B>,
    loss_val: f32,
    embedding: Tensor<B, 1>
) {
    let surprise = gate.compute(loss_val, 1.0);  // dt=1.0 for discrete steps
    if gate.should_consolidate(&surprise) {
        let pattern_hash = rand::random::<u64>();
        let pattern = LongTermPattern::new(pattern_hash, "math")
            .with_surprise(surprise.z_score);  // Use z_score as the surprise value
        memory.store(pattern, embedding);
    }
}

#[derive(Clone, Debug)]
pub struct MathBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: AutodiffBackend> TrainStep<MathBatch<B>, ClassificationOutput<B>> for TitansMathModel<B> {
    fn step(&self, batch: MathBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let (output, _mem) = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<MathBatch<B>, ClassificationOutput<B>> for TitansMathModel<B> {
    fn step(&self, batch: MathBatch<B>) -> ClassificationOutput<B> {
        let (output, _mem) = self.forward_classification(batch.inputs, batch.targets);
        output
    }
}
