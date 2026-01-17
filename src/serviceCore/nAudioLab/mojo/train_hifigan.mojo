"""
HiFiGAN Training Script - Day 27
AudioLabShimmy - Neural Vocoder Training (GAN)

Trains HiFiGAN generator and discriminators to convert mel-spectrograms to audio waveforms.
Uses adversarial training with multi-resolution STFT loss.
"""

from tensor import Tensor, TensorShape
from python import Python
from pathlib import Path
from sys import argv
from collections import Dict
from random import seed, random_float32

# Import model components (from Days 10-11)
from models.hifigan_generator import HiFiGANGenerator
from models.hifigan_discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator
)

# Import training infrastructure (from Days 12-14)
from training.losses import (
    hifigan_generator_loss,
    hifigan_discriminator_loss,
    multi_resolution_stft_loss,
    feature_matching_loss
)
from training.cpu_optimizer import CPUOptimizedAdam
from training.dataset import LJSpeechDataset, AudioBatch

struct HiFiGANTrainer:
    """HiFiGAN GAN trainer with alternating generator/discriminator updates"""
    
    var generator: HiFiGANGenerator
    var mpd: MultiPeriodDiscriminator
    var msd: MultiScaleDiscriminator
    var optim_g: CPUOptimizedAdam
    var optim_d: CPUOptimizedAdam
    var step: Int
    
    fn __init__(inout self, config: TrainingConfig):
        """Initialize HiFiGAN trainer"""
        # Initialize models
        self.generator = HiFiGANGenerator(
            n_mels=config.n_mels,
            upsample_rates=config.upsample_rates,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes
        )
        
        self.mpd = MultiPeriodDiscriminator(
            periods=[2, 3, 5, 7, 11]
        )
        
        self.msd = MultiScaleDiscriminator(
            scales=3
        )
        
        # Initialize optimizers
        self.optim_g = CPUOptimizedAdam(
            learning_rate=config.learning_rate_g,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2
        )
        
        self.optim_d = CPUOptimizedAdam(
            learning_rate=config.learning_rate_d,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2
        )
        
        self.step = 0
    
    fn train_step(inout self, batch: AudioBatch) -> Dict[String, Float32]:
        """Single training step with alternating G/D updates"""
        
        # Get batch data
        real_audio = batch.audio  # [batch, 1, audio_len]
        mel = batch.mel           # [batch, mel_len, 128]
        
        # ============================================
        # 1. TRAIN DISCRIMINATORS
        # ============================================
        
        # Generate fake audio (detached for discriminator training)
        fake_audio = self.generator.forward(mel)
        fake_audio_detached = fake_audio.detach()
        
        # Multi-Period Discriminator
        mpd_real_logits, mpd_real_feats = self.mpd.forward(real_audio)
        mpd_fake_logits, mpd_fake_feats = self.mpd.forward(fake_audio_detached)
        
        # Multi-Scale Discriminator
        msd_real_logits, msd_real_feats = self.msd.forward(real_audio)
        msd_fake_logits, msd_fake_feats = self.msd.forward(fake_audio_detached)
        
        # Discriminator loss
        loss_d_mpd = hifigan_discriminator_loss(mpd_real_logits, mpd_fake_logits)
        loss_d_msd = hifigan_discriminator_loss(msd_real_logits, msd_fake_logits)
        loss_d = loss_d_mpd + loss_d_msd
        
        # Backward and optimize discriminators
        grads_d = loss_d.backward()
        self.optim_d.step(self.get_discriminator_params(), grads_d)
        
        # ============================================
        # 2. TRAIN GENERATOR
        # ============================================
        
        # Generate fake audio (attached for generator training)
        fake_audio = self.generator.forward(mel)
        
        # Get discriminator outputs for fake audio
        mpd_fake_g_logits, mpd_fake_g_feats = self.mpd.forward(fake_audio)
        msd_fake_g_logits, msd_fake_g_feats = self.msd.forward(fake_audio)
        
        # Generator loss
        loss_g, g_loss_dict = hifigan_generator_loss(
            pred_audio=fake_audio,
            target_audio=real_audio,
            disc_fake_outputs_mpd=mpd_fake_g_logits,
            disc_fake_outputs_msd=msd_fake_g_logits,
            disc_real_feats_mpd=mpd_real_feats,
            disc_fake_feats_mpd=mpd_fake_g_feats,
            disc_real_feats_msd=msd_real_feats,
            disc_fake_feats_msd=msd_fake_g_feats
        )
        
        # Backward and optimize generator
        grads_g = loss_g.backward()
        self.optim_g.step(self.generator.parameters(), grads_g)
        
        # Update step counter
        self.step += 1
        
        # Return losses for logging
        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_d_mpd": loss_d_mpd.item(),
            "loss_d_msd": loss_d_msd.item(),
            "loss_stft": g_loss_dict["stft_loss"],
            "loss_adv": g_loss_dict["adv_loss"],
            "loss_fm": g_loss_dict["fm_loss"]
        }
    
    fn get_discriminator_params(self) -> Dict[String, Tensor]:
        """Get all discriminator parameters"""
        params = Dict[String, Tensor]()
        
        # Add MPD parameters
        for name, param in self.mpd.parameters().items():
            params["mpd_" + name] = param
        
        # Add MSD parameters
        for name, param in self.msd.parameters().items():
            params["msd_" + name] = param
        
        return params
    
    fn save_checkpoint(self, path: String):
        """Save training checkpoint"""
        checkpoint = {
            "generator": self.generator.state_dict(),
            "mpd": self.mpd.state_dict(),
            "msd": self.msd.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "step": self.step
        }
        
        # Save to file
        save_checkpoint_to_file(checkpoint, path)
    
    fn load_checkpoint(inout self, path: String):
        """Load training checkpoint"""
        checkpoint = load_checkpoint_from_file(path)
        
        self.generator.load_state_dict(checkpoint["generator"])
        self.mpd.load_state_dict(checkpoint["mpd"])
        self.msd.load_state_dict(checkpoint["msd"])
        self.optim_g.load_state_dict(checkpoint["optim_g"])
        self.optim_d.load_state_dict(checkpoint["optim_d"])
        self.step = checkpoint["step"]


struct TrainingConfig:
    """HiFiGAN training configuration"""
    
    # Model architecture
    var n_mels: Int = 128
    var upsample_rates: List[Int] = [8, 8, 2, 2]  # 128 mel bins → 48kHz
    var upsample_kernel_sizes: List[Int] = [16, 16, 4, 4]
    var resblock_kernel_sizes: List[Int] = [3, 7, 11]
    var resblock_dilation_sizes: List[List[Int]] = [[1,3,5], [1,3,5], [1,3,5]]
    
    # Training
    var batch_size: Int = 16
    var max_steps: Int = 500000
    var learning_rate_g: Float32 = 2e-4
    var learning_rate_d: Float32 = 2e-4
    var adam_beta1: Float32 = 0.8
    var adam_beta2: Float32 = 0.99
    var lr_decay: Float32 = 0.999
    
    # Optimization
    var num_workers: Int = 7
    var use_accelerate: Bool = True
    var mixed_precision: Bool = True
    
    # Logging
    var log_every: Int = 100
    var save_every: Int = 5000
    var validate_every: Int = 1000
    
    # Paths
    var data_dir: String = "data/datasets/ljspeech_processed"
    var checkpoint_dir: String = "data/models/hifigan/checkpoints"
    var log_dir: String = "data/models/hifigan/logs"
    var sample_dir: String = "data/models/hifigan/samples"
    
    # FastSpeech2 checkpoint (for generating ground-truth mels)
    var fastspeech2_checkpoint: String = "data/models/fastspeech2/checkpoints/checkpoint_200000.mojo"


fn load_config_from_yaml(path: String) -> TrainingConfig:
    """Load configuration from YAML file"""
    # Use Python to parse YAML
    try:
        py = Python.import_module("builtins")
        yaml = Python.import_module("yaml")
        
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # Create TrainingConfig from dict
        config = TrainingConfig()
        # ... populate config fields ...
        
        return config
    except:
        print("Error loading config, using defaults")
        return TrainingConfig()


fn main():
    """Main training loop"""
    
    print("=" * 70)
    print("  HiFiGAN Training - Day 27")
    print("  Neural Vocoder (Mel-Spectrogram → Audio)")
    print("  AudioLabShimmy TTS System")
    print("=" * 70)
    print()
    
    # Parse command-line arguments
    var config_path = "config/hifigan_training_config.yaml"
    var resume_checkpoint: Optional[String] = None
    var target_step: Int = 500000
    
    for i in range(len(argv())):
        if argv()[i] == "--config":
            config_path = argv()[i + 1]
        elif argv()[i] == "--resume":
            resume_checkpoint = argv()[i + 1]
        elif argv()[i] == "--target-step":
            target_step = int(argv()[i + 1])
    
    # Load configuration
    print("Loading configuration from:", config_path)
    config = load_config_from_yaml(config_path)
    print("✓ Configuration loaded")
    print()
    
    # Create output directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.sample_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset from:", config.data_dir)
    dataset = LJSpeechDataset.load(config.data_dir)
    print(f"✓ Loaded {len(dataset)} samples")
    print()
    
    # Initialize trainer
    print("Initializing HiFiGAN trainer...")
    var trainer = HiFiGANTrainer(config)
    print("✓ Trainer initialized")
    print()
    
    # Resume from checkpoint if specified
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
        print(f"✓ Resumed from step {trainer.step}")
        print()
    
    # Training loop
    print("=" * 70)
    print(f"  Starting Training")
    print(f"  Current Step: {trainer.step}")
    print(f"  Target Step: {target_step}")
    print(f"  Steps Remaining: {target_step - trainer.step}")
    print("=" * 70)
    print()
    
    var batch_idx = 0
    
    while trainer.step < target_step:
        # Get batch
        batch = dataset.get_audio_batch(batch_idx % len(dataset), config.batch_size)
        
        # Training step
        losses = trainer.train_step(batch)
        
        # Logging
        if trainer.step % config.log_every == 0:
            print(f"Step {trainer.step:6d} | "
                  f"G: {losses['loss_g']:.4f} | "
                  f"D: {losses['loss_d']:.4f} | "
                  f"STFT: {losses['loss_stft']:.4f} | "
                  f"Adv: {losses['loss_adv']:.4f} | "
                  f"FM: {losses['loss_fm']:.4f}")
        
        # Save checkpoint
        if trainer.step % config.save_every == 0:
            checkpoint_path = f"{config.checkpoint_dir}/checkpoint_{trainer.step}.mojo"
            print(f"\nSaving checkpoint: {checkpoint_path}")
            trainer.save_checkpoint(checkpoint_path)
            print("✓ Checkpoint saved\n")
        
        # Validation
        if trainer.step % config.validate_every == 0:
            print(f"\nRunning validation at step {trainer.step}...")
            validate(trainer, dataset, config)
            print("✓ Validation complete\n")
        
        batch_idx += 1
    
    # Final checkpoint
    final_checkpoint = f"{config.checkpoint_dir}/checkpoint_final_{trainer.step}.mojo"
    print(f"\nSaving final checkpoint: {final_checkpoint}")
    trainer.save_checkpoint(final_checkpoint)
    
    print()
    print("=" * 70)
    print("  Training Complete!")
    print(f"  Final Step: {trainer.step}")
    print(f"  Final Generator Loss: {losses['loss_g']:.4f}")
    print(f"  Final Discriminator Loss: {losses['loss_d']:.4f}")
    print("=" * 70)


fn validate(trainer: HiFiGANTrainer, dataset: LJSpeechDataset, config: TrainingConfig):
    """Run validation and generate audio samples"""
    
    # Get validation batch
    val_batch = dataset.get_validation_batch(10)  # 10 samples
    
    # Generate audio
    generated_audio = trainer.generator.forward(val_batch.mel)
    
    # Calculate validation loss (without discriminators)
    val_loss = multi_resolution_stft_loss(generated_audio, val_batch.audio)
    
    print(f"  Validation STFT Loss: {val_loss:.4f}")
    
    # Save audio samples
    sample_dir = f"{config.sample_dir}/step_{trainer.step}"
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(min(5, len(generated_audio))):
        audio_sample = generated_audio[i]
        output_path = f"{sample_dir}/sample_{i:03d}.wav"
        save_audio_wav(audio_sample, output_path, sample_rate=48000)
    
    print(f"  Saved samples to: {sample_dir}")


fn save_audio_wav(audio: Tensor, path: String, sample_rate: Int):
    """Save audio tensor to WAV file using Zig FFI"""
    # Call Zig audio_io functions via FFI
    # This will be implemented in Day 38 (Zig FFI bridge)
    pass


fn load_checkpoint_from_file(path: String) -> Dict:
    """Load checkpoint from file"""
    # Use Python pickle or custom format
    py = Python.import_module("builtins")
    pickle = Python.import_module("pickle")
    
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    
    return checkpoint


fn save_checkpoint_to_file(checkpoint: Dict, path: String):
    """Save checkpoint to file"""
    py = Python.import_module("builtins")
    pickle = Python.import_module("pickle")
    
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


# Entry point
if __name__ == "__main__":
    main()
