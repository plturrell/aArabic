"""
FastSpeech2 Trainer
==================

Complete training loop for FastSpeech2 TTS model with:
- Training and validation loops
- Checkpoint saving/loading
- Learning rate scheduling
- Gradient accumulation
- Progress logging
- Metrics tracking
"""

from tensor import Tensor
from python import Python
from time import now
from sys import exit

# Import model components
from ..models.fastspeech2 import FastSpeech2, TTSOutput
from .losses import fastspeech2_loss
from .cpu_optimizer import CPUOptimizedAdam, WarmupScheduler, GradientAccumulator
from .dataset import DataLoader, TTSBatch


struct TrainingConfig:
    """Configuration for training."""
    var batch_size: Int
    var num_epochs: Int
    var learning_rate: Float32
    var warmup_steps: Int
    var decay_factor: Float32
    var decay_steps: Int
    var accumulation_steps: Int
    var max_grad_norm: Float32
    var validate_every: Int
    var save_every: Int
    var log_every: Int
    var checkpoint_dir: String
    var resume_from: String
    
    fn __init__(inout self):
        """Initialize with default values."""
        self.batch_size = 16
        self.num_epochs = 200
        self.learning_rate = 1e-4
        self.warmup_steps = 4000
        self.decay_factor = 0.5
        self.decay_steps = 50000
        self.accumulation_steps = 2
        self.max_grad_norm = 1.0
        self.validate_every = 5000
        self.save_every = 10000
        self.log_every = 100
        self.checkpoint_dir = "data/models/fastspeech2/checkpoints"
        self.resume_from = ""


struct TrainingMetrics:
    """Metrics for a training step."""
    var total_loss: Float32
    var mel_loss: Float32
    var duration_loss: Float32
    var pitch_loss: Float32
    var energy_loss: Float32
    var learning_rate: Float32
    var grad_norm: Float32
    var step_time: Float64
    
    fn __init__(inout self):
        """Initialize metrics."""
        self.total_loss = 0.0
        self.mel_loss = 0.0
        self.duration_loss = 0.0
        self.pitch_loss = 0.0
        self.energy_loss = 0.0
        self.learning_rate = 0.0
        self.grad_norm = 0.0
        self.step_time = 0.0
    
    fn print(self, step: Int, epoch: Int):
        """Print metrics."""
        print(
            f"[Epoch {epoch}] Step {step}: "
            f"Loss={self.total_loss:.4f} "
            f"(mel={self.mel_loss:.4f}, "
            f"dur={self.duration_loss:.4f}, "
            f"pitch={self.pitch_loss:.4f}, "
            f"energy={self.energy_loss:.4f}) "
            f"LR={self.learning_rate:.6f} "
            f"GradNorm={self.grad_norm:.4f} "
            f"Time={self.step_time:.3f}s"
        )


struct FastSpeech2Trainer:
    """Trainer for FastSpeech2 model."""
    var model: FastSpeech2
    var optimizer: CPUOptimizedAdam
    var scheduler: WarmupScheduler
    var accumulator: GradientAccumulator
    var config: TrainingConfig
    var train_loader: DataLoader
    var val_loader: DataLoader
    var current_step: Int
    var current_epoch: Int
    var best_val_loss: Float32
    
    fn __init__(
        inout self,
        model: FastSpeech2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = 1e10
        
        # Initialize optimizer
        self.optimizer = CPUOptimizedAdam(
            learning_rate=config.learning_rate,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0
        )
        self.optimizer.initialize_moments(self.model.parameters())
        
        # Initialize scheduler
        self.scheduler = WarmupScheduler(
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            decay_steps=config.decay_steps
        )
        
        # Initialize gradient accumulator
        self.accumulator = GradientAccumulator(
            accumulation_steps=config.accumulation_steps
        )
        
        # Resume from checkpoint if specified
        if len(config.resume_from) > 0:
            self.load_checkpoint(config.resume_from)
    
    fn train_epoch(inout self) -> Float32:
        """Train for one epoch."""
        var epoch_loss = 0.0
        var num_batches = len(self.train_loader)
        
        print(f"\n=== Epoch {self.current_epoch + 1}/{self.config.num_epochs} ===")
        print(f"Training on {num_batches} batches...")
        
        # Training loop
        for batch_idx in range(num_batches):
            # Get batch
            var batch = self.train_loader.get_batch(batch_idx)
            
            # Training step
            var metrics = self.train_step(batch)
            epoch_loss += metrics.total_loss
            
            # Log progress
            if self.current_step % self.config.log_every == 0:
                metrics.print(self.current_step, self.current_epoch + 1)
            
            # Validate
            if self.current_step % self.config.validate_every == 0:
                var val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.mojo")
                    print(f"✓ New best model saved (loss={val_loss:.4f})")
            
            # Save checkpoint
            if self.current_step % self.config.save_every == 0:
                var checkpoint_name = f"checkpoint_step_{self.current_step}.mojo"
                self.save_checkpoint(checkpoint_name)
                print(f"✓ Checkpoint saved: {checkpoint_name}")
            
            self.current_step += 1
        
        # Average epoch loss
        var avg_loss = epoch_loss / Float32(num_batches)
        return avg_loss
    
    fn train_step(inout self, batch: TTSBatch) -> TrainingMetrics:
        """Perform a single training step."""
        var start_time = now()
        var metrics = TrainingMetrics()
        
        # Update learning rate
        metrics.learning_rate = self.scheduler.step()
        self.optimizer.set_lr(metrics.learning_rate)
        
        # Forward pass
        var output = self.model.forward(
            batch.phonemes,
            batch.durations,  # Use ground truth during training
            batch.pitch,
            batch.energy
        )
        
        # Compute loss
        var total_loss, loss_dict = fastspeech2_loss(
            output,
            batch.mels,
            batch.durations,
            batch.pitch,
            batch.energy
        )
        
        # Store loss components
        metrics.total_loss = total_loss
        metrics.mel_loss = loss_dict["mel"]
        metrics.duration_loss = loss_dict["duration"]
        metrics.pitch_loss = loss_dict["pitch"]
        metrics.energy_loss = loss_dict["energy"]
        
        # Backward pass (compute gradients)
        var grads = total_loss.backward()
        
        # Accumulate gradients
        self.accumulator.accumulate(grads)
        
        # Update weights if ready
        if self.accumulator.should_step():
            # Get averaged gradients
            var avg_grads = self.accumulator.get_averaged_grads()
            
            # Clip gradients
            metrics.grad_norm = clip_gradients_by_norm(
                avg_grads,
                self.config.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step(self.model.parameters(), avg_grads)
            
            # Reset accumulator
            self.accumulator.reset()
        
        # Compute step time
        var end_time = now()
        metrics.step_time = Float64(end_time - start_time) / 1e9  # Convert to seconds
        
        return metrics
    
    fn validate(inout self) -> Float32:
        """Run validation."""
        print("\nRunning validation...")
        
        var total_loss = 0.0
        var num_batches = len(self.val_loader)
        
        # Validation loop (no gradient computation)
        for batch_idx in range(num_batches):
            var batch = self.val_loader.get_batch(batch_idx)
            
            # Forward pass only
            var output = self.model.forward(
                batch.phonemes,
                batch.durations,
                batch.pitch,
                batch.energy
            )
            
            # Compute loss
            var loss, _ = fastspeech2_loss(
                output,
                batch.mels,
                batch.durations,
                batch.pitch,
                batch.energy
            )
            
            total_loss += loss
        
        # Average validation loss
        var avg_loss = total_loss / Float32(num_batches)
        return avg_loss
    
    fn save_checkpoint(self, filename: String):
        """Save training checkpoint."""
        var path = self.config.checkpoint_dir + "/" + filename
        
        # Create checkpoint dictionary
        var checkpoint = Dict[String, Any]()
        checkpoint["model_state"] = self.model.state_dict()
        checkpoint["optimizer_state"] = self.optimizer.state_dict()
        checkpoint["scheduler_state"] = self.scheduler.state_dict()
        checkpoint["current_step"] = self.current_step
        checkpoint["current_epoch"] = self.current_epoch
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["config"] = self.config
        
        # Save to file
        try:
            save_dict(checkpoint, path)
            print(f"Checkpoint saved to {path}")
        except e:
            print(f"Error saving checkpoint: {e}")
    
    fn load_checkpoint(inout self, filename: String):
        """Load training checkpoint."""
        var path = filename
        if not filename.startswith("/"):
            path = self.config.checkpoint_dir + "/" + filename
        
        try:
            # Load checkpoint
            var checkpoint = load_dict(path)
            
            # Restore model
            self.model.load_state_dict(checkpoint["model_state"])
            
            # Restore optimizer
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
            # Restore scheduler
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            
            # Restore training state
            self.current_step = checkpoint["current_step"]
            self.current_epoch = checkpoint["current_epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            
            print(f"Checkpoint loaded from {path}")
            print(f"Resuming from step {self.current_step}, epoch {self.current_epoch}")
        except e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    fn train(inout self):
        """Main training loop."""
        print("\n" + "="*60)
        print("FastSpeech2 Training")
        print("="*60)
        print(f"Configuration:")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Accumulation steps: {self.config.accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.accumulation_steps}")
        print(f"  Num epochs: {self.config.num_epochs}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"  Max gradient norm: {self.config.max_grad_norm}")
        print(f"  Checkpoint dir: {self.config.checkpoint_dir}")
        print("="*60 + "\n")
        
        # Training loop
        var start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            var epoch_loss = self.train_epoch()
            
            print(f"\nEpoch {epoch + 1} complete. Average loss: {epoch_loss:.4f}")
            
            # Validate at end of epoch
            var val_loss = self.validate()
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save epoch checkpoint
            var checkpoint_name = f"checkpoint_epoch_{epoch + 1}.mojo"
            self.save_checkpoint(checkpoint_name)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.mojo")
                print(f"✓ New best model saved (loss={val_loss:.4f})")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total steps: {self.current_step}")
        print("="*60)


fn clip_gradients_by_norm(grads: Dict[String, Tensor], max_norm: Float32) -> Float32:
    """Clip gradients by global norm."""
    # Compute global norm
    var total_norm = 0.0
    for name in grads:
        var grad = grads[name]
        var norm = tensor_norm(grad)
        total_norm += norm * norm
    total_norm = sqrt(total_norm)
    
    # Clip if needed
    if total_norm > max_norm:
        var clip_coef = max_norm / (total_norm + 1e-6)
        for name in grads:
            grads[name] = grads[name] * clip_coef
    
    return total_norm


fn tensor_norm(tensor: Tensor) -> Float32:
    """Compute L2 norm of a tensor."""
    var sum_squares = 0.0
    for i in range(tensor.num_elements()):
        var val = tensor[i]
        sum_squares += val * val
    return sqrt(sum_squares)


fn save_dict(data: Dict[String, Any], path: String) raises:
    """Save dictionary to file (placeholder)."""
    # TODO: Implement serialization
    print(f"Saving to {path}...")


fn load_dict(path: String) raises -> Dict[String, Any]:
    """Load dictionary from file (placeholder)."""
    # TODO: Implement deserialization
    print(f"Loading from {path}...")
    return Dict[String, Any]()
