"""
Length Regulator for FastSpeech2

Expands encoder output based on predicted durations to match mel-spectrogram length.
This is the key component that allows FastSpeech2 to control speech duration.
"""

from tensor import Tensor
from memory import memset_zero
import math


struct LengthRegulator:
    """
    Length Regulator for FastSpeech2.
    
    Expands phoneme-level encoder output to frame-level based on durations.
    Each phoneme is repeated 'duration' times to create the mel-spectrogram timeline.
    
    Example:
        Input: encoder_output [batch, phoneme_seq_len, d_model]
               durations [batch, phoneme_seq_len]
        Output: expanded [batch, mel_seq_len, d_model]
        where mel_seq_len = sum(durations)
    """
    
    fn __init__(inout self):
        """Initialize length regulator (stateless)."""
        pass
    
    fn regulate_length(
        self,
        encoder_output: Tensor[DType.float32],
        durations: Tensor[DType.float32],
        alpha: Float32 = 1.0
    ) -> Tensor[DType.float32]:
        """
        Expand encoder output based on durations.
        
        Args:
            encoder_output: Encoder output [batch, phoneme_len, d_model]
            durations: Predicted durations [batch, phoneme_len] (in log scale or frames)
            alpha: Speed control factor (1.0 = normal, <1 = faster, >1 = slower)
        
        Returns:
            Expanded output [batch, mel_len, d_model]
        """
        let batch_size = encoder_output.shape()[0]
        let phoneme_len = encoder_output.shape()[1]
        let d_model = encoder_output.shape()[2]
        
        # Convert log durations to frame counts and apply speed control
        var frame_durations = Tensor[DType.int32](batch_size, phoneme_len)
        var total_frames = 0
        
        for b in range(batch_size):
            var batch_total = 0
            for p in range(phoneme_len):
                # Convert from log scale and apply alpha
                let duration_val = durations[b * phoneme_len + p]
                
                # Handle both log and linear durations
                var frame_count: Int
                if duration_val > 10.0:  # Likely log scale
                    frame_count = int(math.exp(duration_val) / alpha + 0.5)
                else:
                    frame_count = int(duration_val / alpha + 0.5)
                
                # Ensure at least 1 frame per phoneme
                frame_count = max(1, frame_count)
                
                frame_durations[b * phoneme_len + p] = frame_count
                batch_total += frame_count
            
            # Track maximum length across batch
            if batch_total > total_frames:
                total_frames = batch_total
        
        # Create expanded output with padding
        var expanded = Tensor[DType.float32](batch_size, total_frames, d_model)
        memset_zero(expanded.data(), expanded.num_elements())
        
        # Expand each batch
        for b in range(batch_size):
            var current_frame = 0
            
            for p in range(phoneme_len):
                let duration = int(frame_durations[b * phoneme_len + p])
                
                # Repeat phoneme encoding 'duration' times
                for d in range(duration):
                    if current_frame < total_frames:
                        # Copy encoder output to expanded position
                        for dim in range(d_model):
                            let src_idx = b * phoneme_len * d_model + p * d_model + dim
                            let dst_idx = b * total_frames * d_model + current_frame * d_model + dim
                            expanded[dst_idx] = encoder_output[src_idx]
                        
                        current_frame += 1
        
        return expanded
    
    fn regulate_length_with_target(
        self,
        encoder_output: Tensor[DType.float32],
        target_durations: Tensor[DType.int32]
    ) -> Tensor[DType.float32]:
        """
        Expand encoder output using ground truth durations (for training).
        
        Args:
            encoder_output: Encoder output [batch, phoneme_len, d_model]
            target_durations: Ground truth durations [batch, phoneme_len] as frame counts
        
        Returns:
            Expanded output [batch, mel_len, d_model]
        """
        let batch_size = encoder_output.shape()[0]
        let phoneme_len = encoder_output.shape()[1]
        let d_model = encoder_output.shape()[2]
        
        # Calculate max length across batch
        var total_frames = 0
        for b in range(batch_size):
            var batch_total = 0
            for p in range(phoneme_len):
                batch_total += int(target_durations[b * phoneme_len + p])
            if batch_total > total_frames:
                total_frames = batch_total
        
        # Create expanded output
        var expanded = Tensor[DType.float32](batch_size, total_frames, d_model)
        memset_zero(expanded.data(), expanded.num_elements())
        
        # Expand each batch using target durations
        for b in range(batch_size):
            var current_frame = 0
            
            for p in range(phoneme_len):
                let duration = int(target_durations[b * phoneme_len + p])
                
                # Repeat phoneme encoding 'duration' times
                for d in range(duration):
                    if current_frame < total_frames:
                        # Copy encoder output to expanded position
                        for dim in range(d_model):
                            let src_idx = b * phoneme_len * d_model + p * d_model + dim
                            let dst_idx = b * total_frames * d_model + current_frame * d_model + dim
                            expanded[dst_idx] = encoder_output[src_idx]
                        
                        current_frame += 1
        
        return expanded
    
    fn get_output_length(
        self,
        durations: Tensor[DType.float32],
        alpha: Float32 = 1.0
    ) -> Tensor[DType.int32]:
        """
        Calculate output length for each batch given durations.
        
        Args:
            durations: Predicted durations [batch, phoneme_len]
            alpha: Speed control factor
        
        Returns:
            Output lengths [batch]
        """
        let batch_size = durations.shape()[0]
        let phoneme_len = durations.shape()[1]
        
        var lengths = Tensor[DType.int32](batch_size)
        
        for b in range(batch_size):
            var total = 0
            for p in range(phoneme_len):
                let duration_val = durations[b * phoneme_len + p]
                
                var frame_count: Int
                if duration_val > 10.0:
                    frame_count = int(math.exp(duration_val) / alpha + 0.5)
                else:
                    frame_count = int(duration_val / alpha + 0.5)
                
                frame_count = max(1, frame_count)
                total += frame_count
            
            lengths[b] = total
        
        return lengths


struct VarianceAdaptor:
    """
    Complete Variance Adaptor combining duration, pitch, and energy prediction
    with length regulation.
    """
    var duration_predictor: DurationPredictor
    var pitch_predictor: PitchPredictor
    var energy_predictor: EnergyPredictor
    var length_regulator: LengthRegulator
    var pitch_embedding: Tensor[DType.float32]  # Embedding weights for pitch
    var energy_embedding: Tensor[DType.float32]  # Embedding weights for energy
    var d_model: Int
    var pitch_bins: Int
    var energy_bins: Int
    
    fn __init__(
        inout self,
        d_model: Int = 256,
        kernel_size: Int = 3,
        dropout: Float32 = 0.1,
        pitch_bins: Int = 256,
        energy_bins: Int = 256
    ):
        """
        Initialize variance adaptor.
        
        Args:
            d_model: Model dimension
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            pitch_bins: Number of pitch quantization bins
            energy_bins: Number of energy quantization bins
        """
        from .duration_predictor import DurationPredictor
        from .pitch_predictor import PitchPredictor
        from .energy_predictor import EnergyPredictor
        
        self.d_model = d_model
        self.pitch_bins = pitch_bins
        self.energy_bins = energy_bins
        
        # Initialize predictors
        self.duration_predictor = DurationPredictor(d_model, kernel_size, dropout)
        self.pitch_predictor = PitchPredictor(d_model, kernel_size, dropout)
        self.energy_predictor = EnergyPredictor(d_model, kernel_size, dropout)
        self.length_regulator = LengthRegulator()
        
        # Initialize embeddings for pitch and energy conditioning
        from random import rand
        
        self.pitch_embedding = Tensor[DType.float32](pitch_bins, d_model)
        self.energy_embedding = Tensor[DType.float32](energy_bins, d_model)
        
        let std_dev = math.sqrt(2.0 / Float32(d_model))
        for i in range(self.pitch_embedding.num_elements()):
            self.pitch_embedding[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
        for i in range(self.energy_embedding.num_elements()):
            self.energy_embedding[i] = (rand[DType.float32]() - 0.5) * std_dev * 2.0
    
    fn forward(
        self,
        encoder_output: Tensor[DType.float32],
        target_durations: Tensor[DType.int32] = Tensor[DType.int32](),
        target_pitch: Tensor[DType.float32] = Tensor[DType.float32](),
        target_energy: Tensor[DType.float32] = Tensor[DType.float32](),
        alpha: Float32 = 1.0
    ) -> Tuple[Tensor[DType.float32], Tensor[DType.float32], Tensor[DType.float32], Tensor[DType.float32]]:
        """
        Apply variance adaptation.
        
        Args:
            encoder_output: Encoder output [batch, phoneme_len, d_model]
            target_durations: Target durations for training (optional)
            target_pitch: Target pitch for training (optional)
            target_energy: Target energy for training (optional)
            alpha: Speed control factor
        
        Returns:
            Tuple of (output, pred_duration, pred_pitch, pred_energy)
        """
        # Predict variances
        var pred_duration = self.duration_predictor.forward(encoder_output)
        var pred_pitch = self.pitch_predictor.forward(encoder_output)
        var pred_energy = self.energy_predictor.forward(encoder_output)
        
        # Use targets if provided (training), else use predictions (inference)
        var duration_to_use = pred_duration
        if target_durations.num_elements() > 0:
            # Convert int32 target durations to float32 for regulation
            duration_to_use = Tensor[DType.float32](target_durations.shape())
            for i in range(target_durations.num_elements()):
                duration_to_use[i] = Float32(target_durations[i])
        
        # Length regulation
        var output: Tensor[DType.float32]
        if target_durations.num_elements() > 0:
            output = self.length_regulator.regulate_length_with_target(encoder_output, target_durations)
        else:
            output = self.length_regulator.regulate_length(encoder_output, pred_duration, alpha)
        
        # Add pitch and energy conditioning to output
        # (In a complete implementation, we would quantize and embed pitch/energy here)
        
        return (output, pred_duration, pred_pitch, pred_energy)
    
    fn eval(inout self):
        """Set to evaluation mode."""
        self.duration_predictor.eval()
        self.pitch_predictor.eval()
        self.energy_predictor.eval()
    
    fn train(inout self):
        """Set to training mode."""
        self.duration_predictor.train()
        self.pitch_predictor.train()
        self.energy_predictor.train()
