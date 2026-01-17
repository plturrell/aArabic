"""
Complete FastSpeech2 Model

Combines encoder, variance adaptors, and decoder for text-to-mel synthesis.
"""

from tensor import Tensor
from memory import memset_zero


struct TTSOutput:
    """Output structure for FastSpeech2 forward pass."""
    var mel: Tensor[DType.float32]              # Predicted mel-spectrogram
    var mel_postnet: Tensor[DType.float32]      # Refined mel (optional)
    var pred_duration: Tensor[DType.float32]    # Predicted durations
    var pred_pitch: Tensor[DType.float32]       # Predicted pitch
    var pred_energy: Tensor[DType.float32]      # Predicted energy
    
    fn __init__(
        inout self,
        mel: Tensor[DType.float32],
        mel_postnet: Tensor[DType.float32],
        pred_duration: Tensor[DType.float32],
        pred_pitch: Tensor[DType.float32],
        pred_energy: Tensor[DType.float32]
    ):
        self.mel = mel
        self.mel_postnet = mel_postnet
        self.pred_duration = pred_duration
        self.pred_pitch = pred_pitch
        self.pred_energy = pred_energy


struct FastSpeech2:
    """
    Complete FastSpeech2 TTS Model.
    
    Architecture:
        1. Encoder: Phoneme embedding → FFT blocks → Encoded representation
        2. Variance Adaptor: Duration/Pitch/Energy prediction + Length regulation
        3. Decoder: FFT blocks → Mel-spectrogram projection
        4. PostNet (optional): Mel refinement
    
    Forward Pass:
        phonemes → encoder → variance_adaptor → decoder → mel-spectrogram
    """
    var encoder: FastSpeech2Encoder
    var variance_adaptor: VarianceAdaptor
    var decoder: FastSpeech2Decoder
    var postnet: PostNet
    var use_postnet: Bool
    
    fn __init__(
        inout self,
        n_phonemes: Int = 70,
        d_model: Int = 256,
        n_heads: Int = 4,
        d_ff: Int = 1024,
        encoder_layers: Int = 4,
        decoder_layers: Int = 4,
        n_mels: Int = 128,
        dropout: Float32 = 0.1,
        use_postnet: Bool = True
    ):
        """
        Initialize FastSpeech2 model.
        
        Args:
            n_phonemes: Number of phoneme symbols (default: 70)
            d_model: Model dimension (default: 256)
            n_heads: Number of attention heads (default: 4)
            d_ff: Feed-forward dimension (default: 1024)
            encoder_layers: Number of encoder FFT blocks (default: 4)
            decoder_layers: Number of decoder FFT blocks (default: 4)
            n_mels: Number of mel bins (default: 128)
            dropout: Dropout probability (default: 0.1)
            use_postnet: Whether to use PostNet refinement (default: True)
        """
        from .fastspeech2_encoder import FastSpeech2Encoder
        from .length_regulator import VarianceAdaptor
        from .fastspeech2_decoder import FastSpeech2Decoder, PostNet
        
        self.use_postnet = use_postnet
        
        # Initialize encoder
        self.encoder = FastSpeech2Encoder(
            n_phonemes=n_phonemes,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=encoder_layers,
            kernel_size=9,
            dropout=dropout
        )
        
        # Initialize variance adaptor
        self.variance_adaptor = VarianceAdaptor(
            d_model=d_model,
            kernel_size=3,
            dropout=dropout,
            pitch_bins=256,
            energy_bins=256
        )
        
        # Initialize decoder
        self.decoder = FastSpeech2Decoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=decoder_layers,
            n_mels=n_mels,
            kernel_size=9,
            dropout=dropout
        )
        
        # Initialize postnet (optional)
        self.postnet = PostNet(
            n_mels=n_mels,
            n_channels=512,
            kernel_size=5,
            n_layers=5
        )
    
    fn forward(
        self,
        phonemes: Tensor[DType.int32],
        target_durations: Tensor[DType.int32] = Tensor[DType.int32](),
        target_pitch: Tensor[DType.float32] = Tensor[DType.float32](),
        target_energy: Tensor[DType.float32] = Tensor[DType.float32](),
        alpha: Float32 = 1.0
    ) -> TTSOutput:
        """
        Forward pass: phonemes → mel-spectrogram.
        
        Args:
            phonemes: Phoneme indices [batch, phoneme_len]
            target_durations: Ground truth durations for training (optional)
            target_pitch: Ground truth pitch for training (optional)
            target_energy: Ground truth energy for training (optional)
            alpha: Speed control factor (1.0 = normal, <1 = faster, >1 = slower)
        
        Returns:
            TTSOutput containing mel-spectrogram and predictions
        """
        # 1. Encode phonemes
        var encoder_output = self.encoder.forward(phonemes)
        # Shape: [batch, phoneme_len, 256]
        
        # 2. Variance adaptation
        var (decoder_input, pred_duration, pred_pitch, pred_energy) = \
            self.variance_adaptor.forward(
                encoder_output,
                target_durations,
                target_pitch,
                target_energy,
                alpha
            )
        # Shape: [batch, mel_len, 256]
        
        # 3. Decode to mel-spectrogram
        var mel = self.decoder.forward(decoder_input)
        # Shape: [batch, mel_len, 128]
        
        # 4. Optional PostNet refinement
        var mel_postnet = mel
        if self.use_postnet:
            mel_postnet = self.postnet.forward(mel)
        
        return TTSOutput(
            mel=mel,
            mel_postnet=mel_postnet,
            pred_duration=pred_duration,
            pred_pitch=pred_pitch,
            pred_energy=pred_energy
        )
    
    fn infer(
        self,
        phonemes: Tensor[DType.int32],
        alpha: Float32 = 1.0
    ) -> Tensor[DType.float32]:
        """
        Inference: Generate mel-spectrogram from phonemes.
        
        Args:
            phonemes: Phoneme indices [batch, phoneme_len]
            alpha: Speed control factor
        
        Returns:
            Generated mel-spectrogram [batch, mel_len, n_mels]
        """
        # Forward pass without ground truth targets
        var output = self.forward(
            phonemes,
            Tensor[DType.int32](),  # No target durations
            Tensor[DType.float32](),  # No target pitch
            Tensor[DType.float32](),  # No target energy
            alpha
        )
        
        # Return refined mel if postnet is used, otherwise raw mel
        if self.use_postnet:
            return output.mel_postnet
        else:
            return output.mel
    
    fn eval(inout self):
        """Set model to evaluation mode."""
        self.encoder.eval()
        self.variance_adaptor.eval()
        self.decoder.eval()
    
    fn train(inout self):
        """Set model to training mode."""
        self.encoder.train()
        self.variance_adaptor.train()
        self.decoder.train()
    
    fn num_parameters(self) -> Int:
        """
        Calculate total number of parameters in the model.
        
        Returns:
            Total parameter count
        """
        # Encoder parameters
        var total = 0
        
        # Phoneme embedding: 70 × 256 = 17,920
        total += 70 * 256
        
        # Positional encoding: pre-computed, no learnable params
        
        # Encoder FFT blocks: 4 layers
        # Each FFT block has ~2.1M parameters (attention + conv + FFN)
        # Approximate: 4 × 2.1M = 8.4M
        total += 8_400_000
        
        # Variance adaptor: ~1.33M parameters
        total += 1_330_000
        
        # Decoder FFT blocks: 4 layers (~8.4M)
        total += 8_400_000
        
        # Mel projection: 128 × 256 = 32,768
        total += 32_768
        
        # PostNet: ~2.6M parameters (5 conv layers with 512 channels)
        if self.use_postnet:
            total += 2_600_000
        
        return total


struct FastSpeech2Config:
    """Configuration for FastSpeech2 model."""
    var n_phonemes: Int
    var d_model: Int
    var n_heads: Int
    var d_ff: Int
    var encoder_layers: Int
    var decoder_layers: Int
    var n_mels: Int
    var dropout: Float32
    var use_postnet: Bool
    
    fn __init__(
        inout self,
        n_phonemes: Int = 70,
        d_model: Int = 256,
        n_heads: Int = 4,
        d_ff: Int = 1024,
        encoder_layers: Int = 4,
        decoder_layers: Int = 4,
        n_mels: Int = 128,
        dropout: Float32 = 0.1,
        use_postnet: Bool = True
    ):
        """Initialize configuration with default values."""
        self.n_phonemes = n_phonemes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.n_mels = n_mels
        self.dropout = dropout
        self.use_postnet = use_postnet
    
    fn create_model(self) -> FastSpeech2:
        """Create FastSpeech2 model from configuration."""
        return FastSpeech2(
            n_phonemes=self.n_phonemes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            n_mels=self.n_mels,
            dropout=self.dropout,
            use_postnet=self.use_postnet
        )


fn print_model_summary(model: FastSpeech2):
    """
    Print model architecture summary.
    
    Args:
        model: FastSpeech2 model instance
    """
    print("=" * 60)
    print("FastSpeech2 Model Summary")
    print("=" * 60)
    print("")
    print("Architecture:")
    print("  Encoder:")
    print("    - Phoneme embedding: 70 → 256")
    print("    - FFT blocks: 4 layers")
    print("    - Output: [batch, phoneme_len, 256]")
    print("")
    print("  Variance Adaptor:")
    print("    - Duration predictor: Conv1D + Linear")
    print("    - Pitch predictor: Conv1D + Linear")
    print("    - Energy predictor: Conv1D + Linear")
    print("    - Length regulator: Sequence expansion")
    print("    - Output: [batch, mel_len, 256]")
    print("")
    print("  Decoder:")
    print("    - FFT blocks: 4 layers")
    print("    - Mel projection: 256 → 128")
    print("    - Output: [batch, mel_len, 128]")
    print("")
    
    if model.use_postnet:
        print("  PostNet:")
        print("    - Conv layers: 5 layers with 512 channels")
        print("    - Residual refinement")
        print("")
    
    let total_params = model.num_parameters()
    print("Total Parameters: ~" + str(total_params / 1_000_000) + "M")
    print("")
    print("=" * 60)
