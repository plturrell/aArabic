//! Training binary for World-Class Time Series Forecasting on LOTSA.
//!
//! This trains the WorldClassForecaster model on the LOTSA dataset,
//! a large-scale time series archive for universal forecasting.
//!
//! Features:
//! - Cosine annealing learning rate scheduler with warmup
//! - Early stopping based on validation loss
//! - Gradient clipping for training stability
//! - Multi-dataset loading from LOTSA

use anyhow::Result;
use burn::{
    backend::{ndarray::NdArray, Autodiff},
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    grad_clipping::GradientClippingConfig,
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::{Module, AutodiffModule},
    optim::{AdamConfig, GradientsParams, Optimizer},
    lr_scheduler::LrScheduler,
    record::CompactRecorder,
    tensor::backend::Backend,
    tensor::{Tensor, ElementConversion},
    train::{
        LearnerBuilder,
        MetricEarlyStoppingStrategy,
        StoppingCondition,
        metric::LossMetric,
        metric::store::{Aggregate, Direction, Split},
    },
};
use clap::Parser;
use deepseek_math::data::lotsa::{LotsaLoader, LotsaConfig};
use deepseek_math::data::features::{FeatureExtractor, FeatureConfig};
use deepseek_math::model::forecasting::{
    WorldClassForecaster, WorldClassForecasterConfig, TimeSeriesBatch, DistributionType,
};
use deepseek_math::model::titans::memory::{
    SurpriseMemoryConfig, SurpriseTrainingState,
};
use std::path::PathBuf;

type MyBackend = Autodiff<NdArray>;
type MyInnerBackend = NdArray;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train WorldClassForecaster on LOTSA")]
struct Args {
    /// Output directory for checkpoints
    #[arg(short, long, default_value = "artifact/forecaster")]
    output: PathBuf,

    /// Optional cached data file (JSONL)
    #[arg(short, long)]
    data_file: Option<PathBuf>,

    /// Number of epochs
    #[arg(short, long, default_value_t = 50)]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Initial learning rate
    #[arg(short, long, default_value_t = 1e-4)]
    learning_rate: f64,

    /// Minimum learning rate (for cosine annealing)
    #[arg(long, default_value_t = 1e-6)]
    min_learning_rate: f64,

    /// Warmup epochs (linear warmup before cosine decay)
    #[arg(long, default_value_t = 5)]
    warmup_epochs: usize,

    /// Context length (history)
    #[arg(long, default_value_t = 512)]
    context_len: usize,

    /// Forecast horizon
    #[arg(long, default_value_t = 96)]
    horizon: usize,

    /// Model dimension
    #[arg(long, default_value_t = 128)]
    d_model: usize,

    /// Number of transformer layers
    #[arg(long, default_value_t = 4)]
    n_layers: usize,

    /// Number of attention heads
    #[arg(long, default_value_t = 4)]
    n_heads: usize,

    /// Samples to load per LOTSA config
    #[arg(long, default_value_t = 100)]
    samples_per_config: usize,

    /// Skip downloading, use only local data
    #[arg(long, default_value_t = false)]
    offline: bool,

    /// Gradient clipping max norm (0 = disabled)
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f64,

    /// Early stopping patience (epochs without improvement, 0 = disabled)
    #[arg(long, default_value_t = 10)]
    early_stopping_patience: usize,

    /// Dataset preset: "quick", "all", "transportation", "m4"
    #[arg(long, default_value = "quick")]
    dataset_preset: String,

    /// Enable surprise-weighted training with Hull-White calibration
    #[arg(long, default_value_t = false)]
    surprise_weighted: bool,

    /// Surprise z-threshold for significant events
    #[arg(long, default_value_t = 2.0)]
    surprise_threshold: f32,

    /// Minimum gradient weight for low-surprise samples
    #[arg(long, default_value_t = 0.1)]
    min_gradient_weight: f32,

    /// Maximum gradient weight for high-surprise samples
    #[arg(long, default_value_t = 3.0)]
    max_gradient_weight: f32,
}

/// Dataset item for time series training.
#[derive(Clone, Debug)]
struct TimeSeriesItem {
    context: Vec<f32>,
    target: Vec<f32>,
}

/// Batcher for training (with Autodiff backend) using feature engineering.
#[derive(Clone)]
struct TimeSeriesBatcherTrain {
    context_len: usize,
    horizon: usize,
    feature_extractor: FeatureExtractor,
    feature_dim: usize,
}

impl TimeSeriesBatcherTrain {
    fn new(context_len: usize, horizon: usize) -> Self {
        let config = FeatureConfig::minimal(); // Use minimal for speed during training
        let feature_extractor = FeatureExtractor::new(config);
        let feature_dim = feature_extractor.feature_dim();
        Self { context_len, horizon, feature_extractor, feature_dim }
    }

    fn with_feature_config(context_len: usize, horizon: usize, config: FeatureConfig) -> Self {
        let feature_extractor = FeatureExtractor::new(config);
        let feature_dim = feature_extractor.feature_dim();
        Self { context_len, horizon, feature_extractor, feature_dim }
    }
}

impl Batcher<TimeSeriesItem, TimeSeriesBatch<MyBackend>> for TimeSeriesBatcherTrain {
    fn batch(&self, items: Vec<TimeSeriesItem>) -> TimeSeriesBatch<MyBackend> {
        let batch_size = items.len();
        let device = <MyBackend as Backend>::Device::default();

        // Preallocate for features: [batch, context_len, feature_dim]
        let mut history_data = Vec::with_capacity(batch_size * self.context_len * self.feature_dim);
        let mut target_data = Vec::with_capacity(batch_size * self.horizon);
        let mut norm_params = Vec::with_capacity(batch_size * 2); // median, iqr per sample

        for item in &items {
            // Pad or truncate context
            let context = if item.context.len() >= self.context_len {
                item.context[item.context.len() - self.context_len..].to_vec()
            } else {
                let mut padded = vec![0.0f32; self.context_len - item.context.len()];
                padded.extend(&item.context);
                padded
            };

            // Extract features from context
            let features = self.feature_extractor.extract(&context);
            let dense = features.to_dense();

            // Store normalization params for denormalization during inference
            norm_params.push(features.norm_median);
            norm_params.push(features.norm_iqr);

            // Flatten dense features [context_len, feature_dim] -> flat
            for t in 0..self.context_len {
                if t < dense.len() {
                    for f in 0..self.feature_dim {
                        if f < dense[t].len() {
                            history_data.push(dense[t][f]);
                        } else {
                            history_data.push(0.0);
                        }
                    }
                } else {
                    // Padding
                    for _ in 0..self.feature_dim {
                        history_data.push(0.0);
                    }
                }
            }

            // Normalize target using same params as context for consistency
            let target_raw = if item.target.len() >= self.horizon {
                item.target[..self.horizon].to_vec()
            } else {
                let mut t = item.target.clone();
                t.resize(self.horizon, *item.target.last().unwrap_or(&0.0));
                t
            };

            // Normalize targets with the context's normalization params
            let median = features.norm_median;
            let iqr = features.norm_iqr;
            for &v in &target_raw {
                target_data.push((v - median) / iqr);
            }
        }

        // Create tensors: history is [batch, context_len, feature_dim]
        let history = Tensor::<MyBackend, 1>::from_floats(&history_data[..], &device)
            .reshape([batch_size, self.context_len, self.feature_dim]);
        let targets = Tensor::<MyBackend, 1>::from_floats(&target_data[..], &device)
            .reshape([batch_size, self.horizon]);

        TimeSeriesBatch::new(history, targets)
    }
}

/// Batcher for validation (without Autodiff) using feature engineering.
#[derive(Clone)]
struct TimeSeriesBatcherValid {
    context_len: usize,
    horizon: usize,
    feature_extractor: FeatureExtractor,
    feature_dim: usize,
}

impl TimeSeriesBatcherValid {
    fn new(context_len: usize, horizon: usize) -> Self {
        let config = FeatureConfig::minimal();
        let feature_extractor = FeatureExtractor::new(config);
        let feature_dim = feature_extractor.feature_dim();
        Self { context_len, horizon, feature_extractor, feature_dim }
    }
}

impl Batcher<TimeSeriesItem, TimeSeriesBatch<MyInnerBackend>> for TimeSeriesBatcherValid {
    fn batch(&self, items: Vec<TimeSeriesItem>) -> TimeSeriesBatch<MyInnerBackend> {
        let batch_size = items.len();
        let device = <MyInnerBackend as Backend>::Device::default();

        let mut history_data = Vec::with_capacity(batch_size * self.context_len * self.feature_dim);
        let mut target_data = Vec::with_capacity(batch_size * self.horizon);

        for item in &items {
            let context = if item.context.len() >= self.context_len {
                item.context[item.context.len() - self.context_len..].to_vec()
            } else {
                let mut padded = vec![0.0f32; self.context_len - item.context.len()];
                padded.extend(&item.context);
                padded
            };

            // Extract features
            let features = self.feature_extractor.extract(&context);
            let dense = features.to_dense();

            for t in 0..self.context_len {
                if t < dense.len() {
                    for f in 0..self.feature_dim {
                        if f < dense[t].len() {
                            history_data.push(dense[t][f]);
                        } else {
                            history_data.push(0.0);
                        }
                    }
                } else {
                    for _ in 0..self.feature_dim {
                        history_data.push(0.0);
                    }
                }
            }

            let target_raw = if item.target.len() >= self.horizon {
                item.target[..self.horizon].to_vec()
            } else {
                let mut t = item.target.clone();
                t.resize(self.horizon, *item.target.last().unwrap_or(&0.0));
                t
            };

            // Normalize targets
            let median = features.norm_median;
            let iqr = features.norm_iqr;
            for &v in &target_raw {
                target_data.push((v - median) / iqr);
            }
        }

        let history = Tensor::<MyInnerBackend, 1>::from_floats(&history_data[..], &device)
            .reshape([batch_size, self.context_len, self.feature_dim]);
        let targets = Tensor::<MyInnerBackend, 1>::from_floats(&target_data[..], &device)
            .reshape([batch_size, self.horizon]);

        TimeSeriesBatch::new(history, targets)
    }
}

/// Get dataset configs based on preset name.
fn get_dataset_configs(preset: &str) -> Vec<LotsaConfig> {
    match preset.to_lowercase().as_str() {
        "quick" => LotsaConfig::quick(),
        "all" => LotsaConfig::all(),
        "transportation" => LotsaConfig::transportation(),
        "m4" => LotsaConfig::m4(),
        _ => {
            eprintln!("Unknown preset '{}', using 'quick'", preset);
            LotsaConfig::quick()
        }
    }
}

/// Load data from LOTSA or local file.
async fn load_data(args: &Args) -> Result<Vec<TimeSeriesItem>> {
    let items: Vec<TimeSeriesItem>;

    if let Some(data_file) = &args.data_file {
        // Load from local JSONL file
        println!("Loading cached data from {:?}...", data_file);
        let samples = LotsaLoader::load_from_jsonl(data_file)?;
        items = samples
            .into_iter()
            .map(|s| TimeSeriesItem {
                context: s.context,
                target: s.target,
            })
            .collect();
    } else if args.offline {
        return Err(anyhow::anyhow!(
            "Offline mode requires --data-file. Download data first with --save-data."
        ));
    } else {
        // Download from LOTSA
        let configs = get_dataset_configs(&args.dataset_preset);
        println!("Downloading data from LOTSA ({} datasets)...", configs.len());
        for config in &configs {
            println!("  - {}", config.config_name());
        }

        let loader = LotsaLoader::with_params(args.context_len, args.horizon);
        let samples = loader.load_multi_dataset(&configs, args.samples_per_config).await?;

        // Optionally save for future use
        let cache_path = args.output.join("lotsa_cache.jsonl");
        println!("Caching data to {:?}...", cache_path);
        std::fs::create_dir_all(&args.output)?;
        LotsaLoader::save_to_jsonl(&samples, &cache_path)?;

        items = samples
            .into_iter()
            .map(|s| TimeSeriesItem {
                context: s.context,
                target: s.target,
            })
            .collect();
    }

    Ok(items)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     World-Class Time Series Forecaster Training              ║");
    println!("║                    on LOTSA Dataset                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Output:         {:?}", args.output);
    println!("  Epochs:         {}", args.epochs);
    println!("  Batch size:     {}", args.batch_size);
    println!("  Learning rate:  {} -> {} (cosine)", args.learning_rate, args.min_learning_rate);
    println!("  Warmup epochs:  {}", args.warmup_epochs);
    println!("  Context len:    {}", args.context_len);
    println!("  Horizon:        {}", args.horizon);
    println!("  d_model:        {}", args.d_model);
    println!("  n_layers:       {}", args.n_layers);
    println!("  n_heads:        {}", args.n_heads);
    println!("  Grad clip:      {}", if args.grad_clip > 0.0 { format!("{}", args.grad_clip) } else { "disabled".to_string() });
    println!("  Early stopping: {}", if args.early_stopping_patience > 0 { format!("{} epochs", args.early_stopping_patience) } else { "disabled".to_string() });
    println!("  Dataset preset: {}", args.dataset_preset);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Load dataset
    let items = load_data(&args).await?;
    println!("Loaded {} samples", items.len());

    if items.is_empty() {
        return Err(anyhow::anyhow!("No training data found."));
    }

    // Split into train/valid (90/10)
    let split_idx = (items.len() as f64 * 0.9) as usize;
    let (train_items, valid_items) = items.split_at(split_idx);
    println!("Train: {} samples, Valid: {} samples", train_items.len(), valid_items.len());

    let train_dataset = InMemDataset::new(train_items.to_vec());
    let valid_dataset = InMemDataset::new(valid_items.to_vec());

    // Create dataloaders with feature engineering
    let batcher_train = TimeSeriesBatcherTrain::new(args.context_len, args.horizon);
    let feature_dim = batcher_train.feature_dim; // Get feature dimension from extractor
    let batcher_valid = TimeSeriesBatcherValid::new(args.context_len, args.horizon);

    println!("Feature engineering: {} features per timestep", feature_dim);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(args.batch_size)
        .shuffle(42)
        .num_workers(2)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(args.batch_size)
        .num_workers(2)
        .build(valid_dataset);

    // Initialize device and model
    let device = <MyBackend as Backend>::Device::default();

    // Use feature_dim as max_variates so the model input projection matches
    let config = WorldClassForecasterConfig::with_defaults(
        args.d_model,       // d_model
        args.n_heads,       // n_heads
        args.n_layers,      // n_layers
        feature_dim,        // max_variates = feature_dim
        args.horizon,       // forecast_horizon
    );

    println!("\nInitializing WorldClassForecaster...");
    println!("  Input features: {}", feature_dim);
    println!("  Parameters: ~{:.2}M",
        (args.d_model * args.d_model * args.n_layers * 4) as f64 / 1_000_000.0
    );

    let model: WorldClassForecaster<MyBackend> = WorldClassForecaster::new(&config, &device);

    // Configure learning rate scheduler (cosine annealing)
    // num_iters is the number of iterations (batches) per epoch * epochs
    let batches_per_epoch = (train_items.len() / args.batch_size).max(1);
    let total_iters = batches_per_epoch * args.epochs;
    let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(
        args.learning_rate,
        total_iters,
    )
    .with_min_lr(args.min_learning_rate)
    .init()
    .expect("Failed to create learning rate scheduler");

    // Configure optimizer with gradient clipping
    let optimizer = if args.grad_clip > 0.0 {
        AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(args.grad_clip as f32)))
            .init()
    } else {
        AdamConfig::new().init()
    };

    // Train using either standard Burn learner or surprise-weighted custom loop
    println!("\nBuilding training pipeline...");

    let eval_model: WorldClassForecaster<MyInnerBackend> = if args.surprise_weighted {
        // ============ SURPRISE-WEIGHTED TRAINING ============
        println!("🧠 Using SURPRISE-WEIGHTED training with Hull-White calibration");
        println!("   Z-threshold: {}", args.surprise_threshold);
        println!("   Gradient weights: [{}, {}]", args.min_gradient_weight, args.max_gradient_weight);

        // Initialize surprise training state with Hull-White dynamics
        let surprise_config = SurpriseMemoryConfig::new(
            args.d_model,  // dim
            64,            // slots (memory slots for calibration)
        )
        .with_z_threshold(args.surprise_threshold)
        .with_min_weight(args.min_gradient_weight)
        .with_max_weight(args.max_gradient_weight);

        let mut surprise_state = SurpriseTrainingState::new(&surprise_config);
        let mut model = model;
        let mut optimizer = optimizer;
        let mut lr_scheduler = lr_scheduler;

        let mut best_valid_loss = f32::MAX;
        let mut epochs_without_improvement = 0;

        for epoch in 0..args.epochs {
            let mut train_loss_sum = 0.0f32;
            let mut train_batches = 0usize;

            // Training loop with surprise-weighted gradients
            for batch in dataloader_train.iter() {
                // Forward pass
                let (forecast, stats) = model.forward(
                    batch.history.clone(),
                    batch.static_covs.clone(),
                    batch.future_covs.clone(),
                    DistributionType::Gaussian,
                    None,
                );

                // Denormalize predictions
                let predictions = model.revin.denormalize(forecast.point, &stats);
                let targets = batch.targets.clone();
                let [batch_size, horizon] = predictions.dims();

                // Compute per-sample MSE (for surprise computation)
                let pred_flat = predictions.clone().reshape([batch_size, horizon]);
                let tgt_flat = targets.clone().reshape([batch_size, horizon]);
                let per_sample_mse: Tensor<MyBackend, 1> = (pred_flat.clone() - tgt_flat.clone())
                    .powf_scalar(2.0)
                    .mean_dim(1)
                    .squeeze(1);  // [batch]

                // Extract per-sample errors for surprise gate
                let errors: Vec<f32> = per_sample_mse.clone().into_data().to_vec().unwrap();

                // Compute surprise-based gradient weights
                let gradient_weights = surprise_state.process_batch(
                    &errors,
                    args.min_gradient_weight,
                    args.max_gradient_weight,
                );

                // Compute weighted loss
                let weights_tensor = Tensor::<MyBackend, 1>::from_floats(
                    gradient_weights.as_slice(), &device
                );
                let weight_sum: f32 = gradient_weights.iter().sum();
                let normalized_weights = weights_tensor.mul_scalar(batch_size as f32 / weight_sum.max(1e-6));

                let weighted_sample_loss = per_sample_mse * normalized_weights;
                let loss = weighted_sample_loss.mean();

                // Backward pass
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(lr_scheduler.step(), model, grads);

                train_loss_sum += errors.iter().sum::<f32>() / batch_size as f32;
                train_batches += 1;
            }

            // Validation
            let mut valid_loss_sum = 0.0f32;
            let mut valid_batches = 0usize;
            let valid_model: WorldClassForecaster<MyInnerBackend> = model.valid();

            for batch in dataloader_valid.iter() {
                // batch is already TimeSeriesBatch<MyInnerBackend> (no Autodiff)
                let (forecast, stats) = valid_model.forward(
                    batch.history.clone(),
                    batch.static_covs.clone(),
                    batch.future_covs.clone(),
                    DistributionType::Gaussian,
                    None,
                );

                let predictions = valid_model.revin.denormalize(forecast.point, &stats);
                let targets = batch.targets.clone();
                let [batch_size, horizon] = predictions.dims();

                let mse = (predictions - targets)
                    .powf_scalar(2.0)
                    .mean()
                    .into_scalar()
                    .elem::<f32>();

                valid_loss_sum += mse;
                valid_batches += 1;
            }

            let train_loss = train_loss_sum / train_batches.max(1) as f32;
            let valid_loss = valid_loss_sum / valid_batches.max(1) as f32;

            // Hull-White calibration stats
            let (hw_alpha, hw_theta, hw_sigma) = surprise_state.hull_white_params();
            let consolidation_rate = surprise_state.consolidation_rate();

            println!(
                "Epoch {:3}/{} | Train: {:.4} | Valid: {:.4} | HW(α={:.3}, θ={:.3}, σ={:.3}) | Consol: {:.1}%",
                epoch + 1, args.epochs, train_loss, valid_loss,
                hw_alpha, hw_theta, hw_sigma,
                consolidation_rate * 100.0
            );

            // Early stopping
            if valid_loss < best_valid_loss {
                best_valid_loss = valid_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if args.early_stopping_patience > 0 && epochs_without_improvement >= args.early_stopping_patience {
                println!("\n⛔ Early stopping at epoch {} (no improvement for {} epochs)",
                    epoch + 1, args.early_stopping_patience);
                break;
            }
        }

        println!("\n✅ Surprise-weighted training complete");
        println!("   Final Hull-White params: {:?}", surprise_state.hull_white_params());
        println!("   Consolidation rate: {:.1}%", surprise_state.consolidation_rate() * 100.0);
        println!("   Avg Z-score: {:.2}", surprise_state.avg_z_score());

        model.valid()
    } else {
        // ============ STANDARD BURN LEARNER ============
        let learner = if args.early_stopping_patience > 0 {
            LearnerBuilder::new(args.output.to_str().unwrap())
                .metric_train_numeric(LossMetric::new())
                .metric_valid_numeric(LossMetric::new())
                .with_file_checkpointer(CompactRecorder::new())
                .devices(vec![device.clone()])
                .num_epochs(args.epochs)
                .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<MyInnerBackend>>(
                    Aggregate::Mean,
                    Direction::Lowest,
                    Split::Valid,
                    StoppingCondition::NoImprovementSince { n_epochs: args.early_stopping_patience },
                ))
                .summary()
                .build(model, optimizer, lr_scheduler)
        } else {
            LearnerBuilder::new(args.output.to_str().unwrap())
                .metric_train_numeric(LossMetric::new())
                .metric_valid_numeric(LossMetric::new())
                .with_file_checkpointer(CompactRecorder::new())
                .devices(vec![device.clone()])
                .num_epochs(args.epochs)
                .summary()
                .build(model, optimizer, lr_scheduler)
        };

        // Train
        println!("\n🚀 Starting training...\n");
        let trained_model = learner.fit(dataloader_train, dataloader_valid);
        trained_model.valid()
    };

    // Save final model
    let final_path = args.output.join("forecaster_final");
    println!("\nSaving model to {:?}...", final_path);
    eval_model.clone()
        .save_file(final_path.to_str().unwrap(), &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("Failed to save model: {}", e))?;

    // Save config
    let config_path = args.output.join("config.json");
    config.save(config_path.to_str().unwrap())?;

    // ============ EVALUATION ============
    println!("\n📊 Evaluating forecasting accuracy on validation set...\n");

    // Compute metrics on validation set
    let mut total_mae = 0.0f32;
    let mut total_mse = 0.0f32;
    let mut total_mape = 0.0f32;
    let mut total_smape = 0.0f32;
    let mut count = 0usize;
    let mut total_samples = 0usize;

    // Create feature extractor for evaluation (MUST match training config)
    let eval_feature_extractor = FeatureExtractor::new(FeatureConfig::minimal());
    let eval_feature_dim = eval_feature_extractor.feature_dim();

    // Sanity check: eval feature dim must match training
    assert_eq!(eval_feature_dim, feature_dim,
        "Feature dimension mismatch: eval {} vs training {}", eval_feature_dim, feature_dim);

    for sample in valid_items.iter() {
        // Create batch of 1
        let target_data: Vec<f32> = sample.target.clone();

        // Pad or truncate context (same as training batcher)
        let context = if sample.context.len() >= args.context_len {
            sample.context[sample.context.len() - args.context_len..].to_vec()
        } else {
            let mut padded = vec![0.0f32; args.context_len - sample.context.len()];
            padded.extend(&sample.context);
            padded
        };

        // Apply feature engineering (same as training)
        let features = eval_feature_extractor.extract(&context);
        let dense = features.to_dense();

        // Build feature tensor [context_len, feature_dim] (same as training batcher)
        let mut feature_data = Vec::with_capacity(args.context_len * eval_feature_dim);
        for t in 0..args.context_len {
            if t < dense.len() {
                for f in 0..eval_feature_dim {
                    if f < dense[t].len() {
                        feature_data.push(dense[t][f]);
                    } else {
                        feature_data.push(0.0);
                    }
                }
            } else {
                // Padding
                for _ in 0..eval_feature_dim {
                    feature_data.push(0.0);
                }
            }
        }

        let history = Tensor::<MyInnerBackend, 1>::from_floats(feature_data.as_slice(), &device)
            .reshape([1, args.context_len, eval_feature_dim]);

        // Forward pass to get predictions
        let (forecast, _stats) = eval_model.forward(
            history,
            None, // static_covs
            None, // future_covs
            deepseek_math::model::forecasting::DistributionType::Gaussian,
            None, // state
        );

        // Get point predictions (point field from ProbabilisticForecast)
        let predictions = forecast.point.reshape([args.horizon]);
        let pred_data: Vec<f32> = predictions.to_data().to_vec().unwrap();

        // Denormalize predictions using context's normalization params
        // (same params used to normalize targets during training)
        let median = features.norm_median;
        let iqr = features.norm_iqr;

        // Compute metrics for this sample
        for i in 0..pred_data.len().min(target_data.len()).min(args.horizon) {
            // Denormalize prediction: pred_denorm = pred * iqr + median
            let pred = pred_data[i] * iqr + median;
            let actual = target_data[i];

            let error: f32 = pred - actual;
            let abs_error: f32 = error.abs();

            // MAE
            total_mae += abs_error;

            // MSE
            total_mse += error * error;

            // MAPE (skip if actual is near zero to avoid division by zero)
            if actual.abs() > 1e-6_f32 {
                total_mape += (abs_error / actual.abs()) * 100.0;
            }

            // sMAPE: 200 * |pred - actual| / (|pred| + |actual|)
            let denom: f32 = pred.abs() + actual.abs();
            if denom > 1e-6_f32 {
                total_smape += 200.0 * abs_error / denom;
            }

            count += 1;
        }
        total_samples += 1;
    }

    // Compute averages
    let n = count as f32;
    let mae = total_mae / n;
    let rmse = (total_mse / n).sqrt();
    let mape = total_mape / n;
    let smape = total_smape / n;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  Forecasting Accuracy Metrics                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Samples evaluated: {:>6}                                   ║", total_samples);
    println!("║  Forecast horizon:  {:>6}                                   ║", args.horizon);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  MAE  (Mean Absolute Error):      {:>12.4}               ║", mae);
    println!("║  RMSE (Root Mean Square Error):   {:>12.4}               ║", rmse);
    println!("║  MAPE (Mean Abs Percentage Err):  {:>12.2}%              ║", mape);
    println!("║  sMAPE (Symmetric MAPE):          {:>12.2}%              ║", smape);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Interpretation
    println!("\n📈 Interpretation:");
    if smape < 10.0 {
        println!("   ✅ Excellent: sMAPE < 10% indicates highly accurate forecasts");
    } else if smape < 20.0 {
        println!("   ✅ Good: sMAPE 10-20% indicates good forecasting performance");
    } else if smape < 30.0 {
        println!("   ⚠️  Fair: sMAPE 20-30% indicates moderate accuracy");
    } else if smape < 50.0 {
        println!("   ⚠️  Poor: sMAPE 30-50% indicates the model needs more training");
    } else {
        println!("   ❌ Needs work: sMAPE > 50% - consider more data/epochs/tuning");
    }

    println!("\n✅ Training complete!");
    println!("Model saved to: {:?}", args.output);

    Ok(())
}

