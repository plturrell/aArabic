"""
HiFiGAN Generator
Neural vocoder for converting mel-spectrograms to audio waveforms
"""

from tensor import Tensor, TensorShape
from random import rand
from math import sqrt
from .hifigan_blocks import (
    Conv1DLayer,
    ConvTranspose1D,
    UpsampleBlock,
    MRFResBlock,
    leaky_relu,
    print_tensor_stats
)


struct HiFiGANConfig:
    """Configuration for HiFiGAN Generator"""
    var n_mels: Int
    var upsample_rates: List[Int]
    var upsample_initial_channel: Int
    var upsample_kernel_sizes: List[Int]
    var resblock_kernel_sizes: List[Int]
    var resblock_dilation_sizes: List[List[Int]]
    var sample_rate: Int
    
    fn __init__(inout self):
        """Default configuration for 48kHz audio"""
        self.n_mels = 128
        self.sample_rate = 48000
        
        # Upsample from mel rate to audio rate
        # Mel hop length = 512, so mel rate = 48000/512 = 93.75 Hz
        # Need to upsample by 8 × 8 × 2 × 4 = 512
        self.upsample_rates = List[Int](8, 8, 2, 4)
        self.upsample_initial_channel = 512
        
        self.upsample_kernel_sizes = List[Int](16, 16, 4, 8)
        self.resblock_kernel_sizes = List[Int](3, 7, 11)
        
        # Dilation patterns for residual blocks
        self.resblock_dilation_sizes = List[List[Int]]()
        let dilations1 = List[Int](1, 3, 5)
        let dilations2 = List[Int](1, 3, 5)
        let dilations3 = List[Int](1, 3, 5)
        self.resblock_dilation_sizes.append(dilations1)
        self.resblock_dilation_sizes.append(dilations2)
        self.resblock_dilation_sizes.append(dilations3)
    
    fn print_summary(self):
        """Print configuration summary"""
        print("HiFiGAN Configuration:")
        print("  Mel bins:", self.n_mels)
        print("  Sample rate:", self.sample_rate)
        print("  Upsample rates:", self.upsample_rates)
        print("  Initial channels:", self.upsample_initial_channel)
        print("  Total upsampling:", self.get_total_upsample_rate())
    
    fn get_total_upsample_rate(self) -> Int:
        """Calculate total upsampling factor"""
        var total: Int = 1
        for i in range(len(self.upsample_rates)):
            total *= self.upsample_rates[i]
        return total


struct HiFiGANGenerator:
    """
    HiFiGAN Generator Network
    Converts mel-spectrograms to audio waveforms
    """
    var config: HiFiGANConfig
    var input_conv: Conv1DLayer
    var upsample_blocks: List[UpsampleBlock]
    var output_conv: Conv1DLayer
    var num_upsamples: Int
    
    fn __init__(inout self, config: HiFiGANConfig):
        """Initialize HiFiGAN Generator"""
        self.config = config
        self.num_upsamples = len(config.upsample_rates)
        
        # Input convolution: mel_bins → initial_channel
        self.input_conv = Conv1DLayer(
            config.n_mels,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        )
        
        # Upsample blocks
        self.upsample_blocks = List[UpsampleBlock]()
        var current_channels = config.upsample_initial_channel
        
        for i in range(self.num_upsamples):
            let upsample_rate = config.upsample_rates[i]
            let next_channels = current_channels // 2
            
            let block = UpsampleBlock(
                current_channels,
                next_channels,
                upsample_rate,
                num_mrf_blocks=3
            )
            self.upsample_blocks.append(block)
            
            current_channels = next_channels
        
        # Output convolution: final_channel → 1 (mono audio)
        self.output_conv = Conv1DLayer(
            current_channels,
            1,
            kernel_size=7,
            stride=1,
            padding=3
        )
    
    fn forward(self, mel: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass through generator
        Input: mel-spectrogram [batch, n_mels, time]
        Output: audio waveform [batch, 1, audio_samples]
        """
        # Input convolution
        var x = self.input_conv.forward(mel)
        x = leaky_relu(x)
        
        # Upsample blocks
        for i in range(len(self.upsample_blocks)):
            x = self.upsample_blocks[i].forward(x)
        
        # Output convolution
        x = self.output_conv.forward(x)
        
        # Tanh activation to bound output to [-1, 1]
        x = tanh_activation(x)
        
        return x
    
    fn generate(self, mel: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Generate audio from mel-spectrogram (inference mode)
        Input: mel-spectrogram [batch, n_mels, time]
        Output: audio waveform [batch, 1, audio_samples]
        """
        return self.forward(mel)
    
    fn count_parameters(self) -> Int:
        """Count total number of parameters"""
        var total: Int = 0
        
        # Input conv
        total += self.input_conv.weights.num_elements()
        total += self.input_conv.bias.num_elements()
        
        # Upsample blocks (approximate - would need to traverse all layers)
        # Each upsample block has transposed conv + multiple MRF blocks
        # Rough estimate: ~10M parameters in upsample blocks
        total += 10_000_000
        
        # Output conv
        total += self.output_conv.weights.num_elements()
        total += self.output_conv.bias.num_elements()
        
        return total
    
    fn print_architecture(self):
        """Print network architecture summary"""
        print("\n" + "="*60)
        print("HiFiGAN Generator Architecture")
        print("="*60)
        
        self.config.print_summary()
        
        print("\nNetwork Layers:")
        print("  1. Input Conv: [batch,", self.config.n_mels, ", time] →",
              "[batch,", self.config.upsample_initial_channel, ", time]")
        
        var current_channels = self.config.upsample_initial_channel
        for i in range(self.num_upsamples):
            let rate = self.config.upsample_rates[i]
            let next_channels = current_channels // 2
            print("  ", i+2, ". Upsample Block", i+1, ":", 
                  "[batch,", current_channels, ", T] →",
                  "[batch,", next_channels, ", T×", rate, "]")
            current_channels = next_channels
        
        print("  ", self.num_upsamples+2, ". Output Conv:",
              "[batch,", current_channels, ", audio_len] →",
              "[batch, 1, audio_len]")
        
        print("\nTotal Parameters:", self.count_parameters())
        print("="*60 + "\n")


fn tanh_activation(x: Tensor[DType.float32]) -> Tensor[DType.float32]:
    """Tanh activation function"""
    var result = Tensor[DType.float32](x.shape())
    
    for i in range(x.num_elements()):
        # tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        let val = x[i]
        if val > 10.0:
            result[i] = 1.0
        elif val < -10.0:
            result[i] = -1.0
        else:
            # Simplified tanh approximation
            let exp_2x = exp(2.0 * val)
            result[i] = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    return result


fn exp(x: Float32) -> Float32:
    """Exponential function (simplified)"""
    # In production, use proper exp from math library
    # This is a placeholder
    if x > 10.0:
        return 22026.0  # e^10
    elif x < -10.0:
        return 0.0
    else:
        # Taylor series approximation for demonstration
        var result: Float32 = 1.0
        var term: Float32 = 1.0
        for i in range(1, 15):
            term *= x / Float32(i)
            result += term
        return result


fn transpose_mel(mel: Tensor[DType.float32]) -> Tensor[DType.float32]:
    """
    Transpose mel-spectrogram for HiFiGAN input
    Input: [batch, time, n_mels]
    Output: [batch, n_mels, time]
    """
    let batch_size = mel.dim(0)
    let time_steps = mel.dim(1)
    let n_mels = mel.dim(2)
    
    var output = Tensor[DType.float32](batch_size, n_mels, time_steps)
    
    for b in range(batch_size):
        for t in range(time_steps):
            for m in range(n_mels):
                output[b, m, t] = mel[b, t, m]
    
    return output


struct VocoderPipeline:
    """
    Complete vocoder pipeline from FastSpeech2 output to audio
    """
    var generator: HiFiGANGenerator
    
    fn __init__(inout self):
        """Initialize vocoder with default config"""
        let config = HiFiGANConfig()
        self.generator = HiFiGANGenerator(config)
    
    fn mel_to_audio(
        self,
        mel: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Convert mel-spectrogram to audio waveform
        Input: mel [batch, time, n_mels] from FastSpeech2
        Output: audio [batch, 1, samples]
        """
        # Transpose for HiFiGAN: [batch, time, mels] → [batch, mels, time]
        let mel_transposed = transpose_mel(mel)
        
        # Generate audio
        let audio = self.generator.generate(mel_transposed)
        
        return audio
    
    fn audio_to_numpy_shape(
        self,
        audio: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Convert audio tensor to numpy-compatible shape
        Input: [batch, 1, samples]
        Output: [batch, samples]
        """
        let batch_size = audio.dim(0)
        let num_samples = audio.dim(2)
        
        var output = Tensor[DType.float32](batch_size, num_samples)
        
        for b in range(batch_size):
            for s in range(num_samples):
                output[b, s] = audio[b, 0, s]
        
        return output
    
    fn synthesize(
        self,
        mel: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Complete synthesis pipeline
        Input: mel from FastSpeech2 [batch, time, n_mels]
        Output: audio waveform [batch, samples] in range [-1, 1]
        """
        let audio_3d = self.mel_to_audio(mel)
        let audio_2d = self.audio_to_numpy_shape(audio_3d)
        
        return audio_2d


fn create_default_generator() -> HiFiGANGenerator:
    """Factory function to create generator with default config"""
    let config = HiFiGANConfig()
    return HiFiGANGenerator(config)


fn test_upsampling_math():
    """Test that upsampling rates match mel hop length"""
    let config = HiFiGANConfig()
    let total_upsample = config.get_total_upsample_rate()
    
    print("\n" + "="*60)
    print("Upsampling Math Verification")
    print("="*60)
    print("Sample rate:", config.sample_rate, "Hz")
    print("Mel hop length: 512 samples")
    print("Mel frame rate:", config.sample_rate // 512, "Hz")
    print("\nUpsample rates:", config.upsample_rates)
    print("Total upsampling factor:", total_upsample)
    print("\nVerification:")
    print("  512 samples/frame = Total upsample factor?", 
          "✓" if total_upsample == 512 else "✗")
    print("="*60 + "\n")


fn main():
    """Test HiFiGAN Generator"""
    print("\n🎵 HiFiGAN Generator Test\n")
    
    # Test upsampling math
    test_upsampling_math()
    
    # Create generator
    let config = HiFiGANConfig()
    var generator = HiFiGANGenerator(config)
    
    # Print architecture
    generator.print_architecture()
    
    # Test forward pass
    print("Testing forward pass...")
    let batch_size = 2
    let mel_time_steps = 100  # 100 mel frames
    let n_mels = 128
    
    var mel = Tensor[DType.float32](batch_size, n_mels, mel_time_steps)
    
    # Fill with random mel values
    rand(mel.data(), mel.num_elements())
    for i in range(mel.num_elements()):
        mel[i] = mel[i] * 2.0 - 1.0  # Range [-1, 1]
    
    print("\nInput mel shape: [", batch_size, ",", n_mels, ",", mel_time_steps, "]")
    
    let audio = generator.forward(mel)
    
    print("Output audio shape: [", audio.dim(0), ",", audio.dim(1), ",", audio.dim(2), "]")
    
    # Expected output length
    let expected_audio_len = mel_time_steps * config.get_total_upsample_rate()
    print("Expected audio length:", expected_audio_len, "samples")
    print("Actual audio length:", audio.dim(2), "samples")
    
    # Print audio statistics
    print_tensor_stats("Generated audio", audio)
    
    # Verify audio is in [-1, 1] range
    var min_val: Float32 = audio[0, 0, 0]
    var max_val: Float32 = audio[0, 0, 0]
    for i in range(audio.num_elements()):
        if audio[i] < min_val:
            min_val = audio[i]
        if audio[i] > max_val:
            max_val = audio[i]
    
    print("\nAudio range verification:")
    print("  Min:", min_val, "(should be ≥ -1.0)")
    print("  Max:", max_val, "(should be ≤ 1.0)")
    print("  In range?", "✓" if min_val >= -1.0 and max_val <= 1.0 else "✗")
    
    # Test vocoder pipeline
    print("\n" + "="*60)
    print("Testing VocoderPipeline")
    print("="*60)
    
    var pipeline = VocoderPipeline()
    
    # Create mel in FastSpeech2 output format [batch, time, mels]
    var mel_fs2 = Tensor[DType.float32](batch_size, mel_time_steps, n_mels)
    rand(mel_fs2.data(), mel_fs2.num_elements())
    
    print("\nInput mel (FastSpeech2 format): [", 
          batch_size, ",", mel_time_steps, ",", n_mels, "]")
    
    let audio_synth = pipeline.synthesize(mel_fs2)
    
    print("Output audio: [", audio_synth.dim(0), ",", audio_synth.dim(1), "]")
    print_tensor_stats("Synthesized audio", audio_synth)
    
    print("\n✅ HiFiGAN Generator test complete!")
