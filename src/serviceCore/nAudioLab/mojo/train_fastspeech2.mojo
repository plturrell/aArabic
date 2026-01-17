"""
FastSpeech2 Training Entry Point
=================================

Main script to train FastSpeech2 model.

Usage:
    mojo run mojo/train_fastspeech2.mojo \\
        --data-dir data/datasets/ljspeech_processed \\
        --output-dir data/models/fastspeech2 \\
        --num-epochs 200 \\
        --batch-size 16 \\
        --learning-rate 1e-4

    Or resume from checkpoint:
    mojo run mojo/train_fastspeech2.mojo \\
        --data-dir data/datasets/ljspeech_processed \\
        --output-dir data/models/fastspeech2 \\
        --resume-from checkpoint_epoch_50.mojo
"""

from python import Python
from sys import argv
from pathlib import Path
from random import seed

# Import training components
from training.trainer import FastSpeech2Trainer, TrainingConfig
from training.dataset import LJSpeechDataset, DataLoader
from models.fastspeech2 import FastSpeech2


fn print_banner():
    """Print training banner."""
    print("\n" + "="*70)
    print("   FastSpeech2 Training - AudioLabShimmy")
    print("   Dolby-Quality TTS from Scratch in Mojo")
    print("="*70 + "\n")


fn print_usage():
    """Print usage information."""
    print("Usage:")
    print("  mojo run mojo/train_fastspeech2.mojo [OPTIONS]")
    print("\nOptions:")
    print("  --data-dir PATH          Path to preprocessed dataset")
    print("                           (default: data/datasets/ljspeech_processed)")
    print("  --output-dir PATH        Path to save models and checkpoints")
    print("                           (default: data/models/fastspeech2)")
    print("  --resume-from FILE       Resume training from checkpoint")
    print("  --num-epochs INT         Number of training epochs (default: 200)")
    print("  --batch-size INT         Batch size (default: 16)")
    print("  --learning-rate FLOAT    Base learning rate (default: 1e-4)")
    print("  --warmup-steps INT       LR warmup steps (default: 4000)")
    print("  --accumulation-steps INT Gradient accumulation steps (default: 2)")
    print("  --seed INT               Random seed (default: 42)")
    print("  --help                   Show this help message")
    print("\nExample:")
    print("  mojo run mojo/train_fastspeech2.mojo \\")
    print("    --data-dir data/datasets/ljspeech_processed \\")
    print("    --num-epochs 200 \\")
    print("    --batch-size 16")


fn parse_args() -> TrainingConfig:
    """Parse command line arguments."""
    var config = TrainingConfig()
    var data_dir = "data/datasets/ljspeech_processed"
    var output_dir = "data/models/fastspeech2"
    var random_seed = 42
    
    # Parse arguments
    var i = 1  # Skip program name
    while i < len(argv()):
        var arg = argv()[i]
        
        if arg == "--help" or arg == "-h":
            print_usage()
            exit(0)
        elif arg == "--data-dir" and i + 1 < len(argv()):
            data_dir = argv()[i + 1]
            i += 2
        elif arg == "--output-dir" and i + 1 < len(argv()):
            output_dir = argv()[i + 1]
            i += 2
        elif arg == "--resume-from" and i + 1 < len(argv()):
            config.resume_from = argv()[i + 1]
            i += 2
        elif arg == "--num-epochs" and i + 1 < len(argv()):
            config.num_epochs = int(argv()[i + 1])
            i += 2
        elif arg == "--batch-size" and i + 1 < len(argv()):
            config.batch_size = int(argv()[i + 1])
            i += 2
        elif arg == "--learning-rate" and i + 1 < len(argv()):
            config.learning_rate = float(argv()[i + 1])
            i += 2
        elif arg == "--warmup-steps" and i + 1 < len(argv()):
            config.warmup_steps = int(argv()[i + 1])
            i += 2
        elif arg == "--accumulation-steps" and i + 1 < len(argv()):
            config.accumulation_steps = int(argv()[i + 1])
            i += 2
        elif arg == "--seed" and i + 1 < len(argv()):
            random_seed = int(argv()[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {arg}")
            print_usage()
            exit(1)
    
    # Set checkpoint directory
    config.checkpoint_dir = output_dir + "/checkpoints"
    
    return config, data_dir, output_dir, random_seed


fn create_directories(output_dir: String):
    """Create output directories if they don't exist."""
    try:
        var py = Python.import_module("os")
        
        # Create main output directory
        if not py.path.exists(output_dir):
            py.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Create checkpoint directory
        var checkpoint_dir = output_dir + "/checkpoints"
        if not py.path.exists(checkpoint_dir):
            py.makedirs(checkpoint_dir)
            print(f"Created directory: {checkpoint_dir}")
        
        # Create logs directory
        var logs_dir = output_dir + "/logs"
        if not py.path.exists(logs_dir):
            py.makedirs(logs_dir)
            print(f"Created directory: {logs_dir}")
    except e:
        print(f"Error creating directories: {e}")
        exit(1)


fn load_dataset(data_dir: String, batch_size: Int) -> (DataLoader, DataLoader):
    """Load training and validation datasets."""
    print(f"\nLoading dataset from: {data_dir}")
    
    try:
        # Load full dataset
        var dataset = LJSpeechDataset.load(data_dir)
        print(f"Loaded {len(dataset)} samples")
        
        # Split into train/val (90/10)
        var num_samples = len(dataset)
        var num_train = int(num_samples * 0.9)
        var num_val = num_samples - num_train
        
        print(f"Train samples: {num_train}")
        print(f"Val samples: {num_val}")
        
        # Create data loaders
        var train_dataset = dataset.subset(0, num_train)
        var val_dataset = dataset.subset(num_train, num_samples)
        
        var train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        var val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    except e:
        print(f"Error loading dataset: {e}")
        print(f"Make sure the dataset is preprocessed and located at: {data_dir}")
        exit(1)


fn initialize_model() -> FastSpeech2:
    """Initialize FastSpeech2 model."""
    print("\nInitializing FastSpeech2 model...")
    
    try:
        # Model configuration
        var model = FastSpeech2(
            num_phonemes=70,
            d_model=256,
            n_heads=4,
            encoder_layers=4,
            decoder_layers=4,
            fft_conv_kernel=3,
            dropout=0.1,
            n_mel_bins=128
        )
        
        var num_params = model.num_parameters()
        print(f"Model initialized with {num_params:,} parameters ({num_params * 4 / 1e6:.1f} MB)")
        
        return model
    except e:
        print(f"Error initializing model: {e}")
        exit(1)


fn main():
    """Main training function."""
    # Print banner
    print_banner()
    
    # Parse arguments
    var config, data_dir, output_dir, random_seed = parse_args()
    
    # Set random seed
    seed(random_seed)
    print(f"Random seed: {random_seed}")
    
    # Create output directories
    create_directories(output_dir)
    
    # Load dataset
    var train_loader, val_loader = load_dataset(data_dir, config.batch_size)
    
    # Initialize model
    var model = initialize_model()
    
    # Create trainer
    print("\nInitializing trainer...")
    var trainer = FastSpeech2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Start training
    print("\nStarting training...")
    print("Press Ctrl+C to stop and save checkpoint\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted.mojo")
        print("Checkpoint saved. You can resume with --resume-from interrupted.mojo")
    except e:
        print(f"\n\nTraining failed with error: {e}")
        print("Saving emergency checkpoint...")
        try:
            trainer.save_checkpoint("emergency.mojo")
            print("Emergency checkpoint saved")
        except:
            print("Failed to save emergency checkpoint")
        exit(1)
    
    # Training complete
    print("\n" + "="*70)
    print("Training completed successfully!")
    print(f"Best model saved at: {output_dir}/checkpoints/best.mojo")
    print(f"Final checkpoint at: {output_dir}/checkpoints/checkpoint_epoch_{config.num_epochs}.mojo")
    print("="*70 + "\n")


# Entry point
if __name__ == "__main__":
    main()
