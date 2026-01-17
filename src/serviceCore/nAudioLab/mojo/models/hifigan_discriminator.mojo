"""
HiFiGAN Discriminators
Multi-Period and Multi-Scale discriminators for adversarial training
"""

from tensor import Tensor, TensorShape
from random import rand
from math import sqrt
from .hifigan_blocks import Conv1DLayer, leaky_relu


struct Conv2DLayer:
    """2D Convolution Layer for period discriminator"""
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var kernel_size: Tuple[Int, Int]
    var stride: Tuple[Int, Int]
    var padding: Tuple[Int, Int]
    var in_channels: Int
    var out_channels: Int
    
    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Tuple[Int, Int],
        stride: Tuple[Int, Int] = (1, 1),
        padding: Tuple[Int, Int] = (0, 0)
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier initialization
        let k_h = kernel_size[0]
        let k_w = kernel_size[1]
        let fan_in = in_channels * k_h * k_w
        let fan_out = out_channels * k_h * k_w
        let limit = sqrt(6.0 / Float32(fan_in + fan_out))
        
        self.weights = Tensor[DType.float32](out_channels, in_channels, k_h, k_w)
        self.bias = Tensor[DType.float32](out_channels)
        
        # Initialize weights
        rand(self.weights.data(), self.weights.num_elements())
        for i in range(self.weights.num_elements()):
            self.weights[i] = self.weights[i] * 2.0 * limit - limit
        
        # Initialize bias to zero
        for i in range(out_channels):
            self.bias[i] = 0.0
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass for 2D convolution
        Input: [batch, in_channels, height, width]
        Output: [batch, out_channels, out_height, out_width]
        """
        let batch_size = x.dim(0)
        let in_h = x.dim(2)
        let in_w = x.dim(3)
        
        let k_h = self.kernel_size[0]
        let k_w = self.kernel_size[1]
        let s_h = self.stride[0]
        let s_w = self.stride[1]
        let p_h = self.padding[0]
        let p_w = self.padding[1]
        
        # Calculate output dimensions
        let out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        let out_w = (in_w + 2 * p_w - k_w) // s_w + 1
        
        var output = Tensor[DType.float32](batch_size, self.out_channels, out_h, out_w)
        
        # Simplified 2D convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        var sum_val: Float32 = self.bias[oc]
                        
                        for ic in range(self.in_channels):
                            for kh in range(k_h):
                                for kw in range(k_w):
                                    let ih = oh * s_h + kh - p_h
                                    let iw = ow * s_w + kw - p_w
                                    
                                    if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                                        sum_val += x[b, ic, ih, iw] * self.weights[oc, ic, kh, kw]
                        
                        output[b, oc, oh, ow] = sum_val
        
        return output


struct PeriodDiscriminator:
    """
    Single Period Discriminator
    Analyzes audio at a specific period (e.g., 2, 3, 5, 7, 11)
    """
    var period: Int
    var conv_layers: List[Conv2DLayer]
    var num_layers: Int
    
    fn __init__(inout self, period: Int):
        self.period = period
        self.num_layers = 5
        self.conv_layers = List[Conv2DLayer]()
        
        # Layer configuration
        # Input: 1 channel (reshaped audio)
        # Progressively increase channels: 32 → 128 → 512 → 1024 → 1024
        let channels = List[Int](1, 32, 128, 512, 1024, 1024)
        
        for i in range(self.num_layers):
            let in_ch = channels[i]
            let out_ch = channels[i + 1]
            
            # Different kernel and stride patterns
            let kernel: Tuple[Int, Int]
            let stride: Tuple[Int, Int]
            let padding: Tuple[Int, Int]
            
            if i == 0:
                kernel = (5, 1)
                stride = (3, 1)
                padding = (2, 0)
            elif i < self.num_layers - 1:
                kernel = (5, 1)
                stride = (3, 1)
                padding = (2, 0)
            else:
                kernel = (5, 1)
                stride = (1, 1)
                padding = (2, 0)
            
            self.conv_layers.append(Conv2DLayer(in_ch, out_ch, kernel, stride, padding))
        
        # Final output layer
        self.conv_layers.append(Conv2DLayer(1024, 1, (3, 1), (1, 1), (1, 0)))
    
    fn forward(self, x: Tensor[DType.float32]) -> Tuple[Tensor[DType.float32], List[Tensor[DType.float32]]]:
        """
        Forward pass through period discriminator
        Input: [batch, 1, time] audio
        Output: (logits, feature_maps)
        """
        # Reshape audio by period: [batch, 1, time] → [batch, 1, time//period, period]
        let batch_size = x.dim(0)
        let time_steps = x.dim(2)
        let new_time = time_steps // self.period
        
        # Reshape: [batch, 1, new_time, period]
        var h = Tensor[DType.float32](batch_size, 1, new_time, self.period)
        for b in range(batch_size):
            for t in range(new_time):
                for p in range(self.period):
                    let orig_idx = t * self.period + p
                    if orig_idx < time_steps:
                        h[b, 0, t, p] = x[b, 0, orig_idx]
        
        # Store intermediate feature maps
        var feature_maps = List[Tensor[DType.float32]]()
        
        # Apply conv layers with LeakyReLU
        for i in range(len(self.conv_layers) - 1):
            h = self.conv_layers[i].forward(h)
            h = leaky_relu(h)
            feature_maps.append(h)
        
        # Final layer (no activation)
        let output = self.conv_layers[len(self.conv_layers) - 1].forward(h)
        
        return (output, feature_maps)


struct MultiPeriodDiscriminator:
    """
    Multi-Period Discriminator (MPD)
    Multiple discriminators analyzing audio at different periods
    """
    var periods: List[Int]
    var discriminators: List[PeriodDiscriminator]
    
    fn __init__(inout self):
        # Prime number periods for diverse analysis
        self.periods = List[Int](2, 3, 5, 7, 11)
        self.discriminators = List[PeriodDiscriminator]()
        
        for i in range(len(self.periods)):
            self.discriminators.append(PeriodDiscriminator(self.periods[i]))
    
    fn forward(self, x: Tensor[DType.float32]) -> Tuple[List[Tensor[DType.float32]], List[List[Tensor[DType.float32]]]]:
        """
        Forward pass through all period discriminators
        Input: [batch, 1, time] audio
        Output: (logits_list, feature_maps_list)
        """
        var all_logits = List[Tensor[DType.float32]]()
        var all_features = List[List[Tensor[DType.float32]]]()
        
        for i in range(len(self.discriminators)):
            let (logits, features) = self.discriminators[i].forward(x)
            all_logits.append(logits)
            all_features.append(features)
        
        return (all_logits, all_features)


fn avg_pool_1d(x: Tensor[DType.float32], kernel_size: Int, stride: Int) -> Tensor[DType.float32]:
    """Average pooling for downsampling"""
    let batch_size = x.dim(0)
    let channels = x.dim(1)
    let length = x.dim(2)
    
    let out_len = (length - kernel_size) // stride + 1
    var output = Tensor[DType.float32](batch_size, channels, out_len)
    
    for b in range(batch_size):
        for c in range(channels):
            for i in range(out_len):
                var sum_val: Float32 = 0.0
                let start = i * stride
                
                for k in range(kernel_size):
                    let idx = start + k
                    if idx < length:
                        sum_val += x[b, c, idx]
                
                output[b, c, i] = sum_val / Float32(kernel_size)
    
    return output


struct ScaleDiscriminator:
    """
    Single Scale Discriminator
    Analyzes audio at a specific scale (resolution)
    """
    var conv_layers: List[Conv1DLayer]
    var scale_factor: Int
    
    fn __init__(inout self, scale_factor: Int = 1):
        self.scale_factor = scale_factor
        self.conv_layers = List[Conv1DLayer]()
        
        # Layer configuration
        # Progressively increase channels: 16 → 64 → 256 → 1024 → 1024 → 1024
        let channels = List[Int](1, 16, 64, 256, 1024, 1024, 1024)
        let kernel_sizes = List[Int](15, 41, 41, 41, 41, 5)
        let strides = List[Int](1, 4, 4, 4, 4, 1)
        let groups = List[Int](1, 4, 16, 64, 256, 1)
        
        for i in range(6):
            let in_ch = channels[i]
            let out_ch = channels[i + 1]
            let kernel = kernel_sizes[i]
            let stride = strides[i]
            let padding = kernel // 2
            
            self.conv_layers.append(Conv1DLayer(in_ch, out_ch, kernel, stride, padding))
        
        # Final output layer
        self.conv_layers.append(Conv1DLayer(1024, 1, 3, 1, 1))
    
    fn forward(self, x: Tensor[DType.float32]) -> Tuple[Tensor[DType.float32], List[Tensor[DType.float32]]]:
        """
        Forward pass through scale discriminator
        Input: [batch, 1, time] audio
        Output: (logits, feature_maps)
        """
        var h = x
        var feature_maps = List[Tensor[DType.float32]]()
        
        # Apply conv layers with LeakyReLU
        for i in range(len(self.conv_layers) - 1):
            h = self.conv_layers[i].forward(h)
            h = leaky_relu(h)
            feature_maps.append(h)
        
        # Final layer (no activation)
        let output = self.conv_layers[len(self.conv_layers) - 1].forward(h)
        
        return (output, feature_maps)


struct MultiScaleDiscriminator:
    """
    Multi-Scale Discriminator (MSD)
    Multiple discriminators analyzing audio at different resolutions
    """
    var discriminators: List[ScaleDiscriminator]
    var num_scales: Int
    
    fn __init__(inout self):
        self.num_scales = 3
        self.discriminators = List[ScaleDiscriminator]()
        
        # Create discriminators at 3 scales
        for i in range(self.num_scales):
            self.discriminators.append(ScaleDiscriminator(i + 1))
    
    fn forward(self, x: Tensor[DType.float32]) -> Tuple[List[Tensor[DType.float32]], List[List[Tensor[DType.float32]]]]:
        """
        Forward pass through all scale discriminators
        Input: [batch, 1, time] audio
        Output: (logits_list, feature_maps_list)
        """
        var all_logits = List[Tensor[DType.float32]]()
        var all_features = List[List[Tensor[DType.float32]]]()
        
        var audio_scales = x
        
        for i in range(self.num_scales):
            # Run discriminator at this scale
            let (logits, features) = self.discriminators[i].forward(audio_scales)
            all_logits.append(logits)
            all_features.append(features)
            
            # Downsample for next scale (except last)
            if i < self.num_scales - 1:
                audio_scales = avg_pool_1d(audio_scales, kernel_size=4, stride=2)
        
        return (all_logits, all_features)


struct HiFiGANDiscriminators:
    """
    Complete HiFiGAN Discriminator System
    Combines Multi-Period and Multi-Scale discriminators
    """
    var mpd: MultiPeriodDiscriminator
    var msd: MultiScaleDiscriminator
    
    fn __init__(inout self):
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
    
    fn forward(
        self,
        real_audio: Tensor[DType.float32],
        fake_audio: Tensor[DType.float32]
    ) -> Tuple[
        List[Tensor[DType.float32]],  # real_mpd_logits
        List[Tensor[DType.float32]],  # fake_mpd_logits
        List[List[Tensor[DType.float32]]],  # real_mpd_features
        List[List[Tensor[DType.float32]]],  # fake_mpd_features
        List[Tensor[DType.float32]],  # real_msd_logits
        List[Tensor[DType.float32]],  # fake_msd_logits
        List[List[Tensor[DType.float32]]],  # real_msd_features
        List[List[Tensor[DType.float32]]]   # fake_msd_features
    ]:
        """
        Forward pass through both discriminators
        Analyzes both real and fake audio
        """
        # Multi-Period Discriminator
        let (real_mpd_logits, real_mpd_features) = self.mpd.forward(real_audio)
        let (fake_mpd_logits, fake_mpd_features) = self.mpd.forward(fake_audio)
        
        # Multi-Scale Discriminator
        let (real_msd_logits, real_msd_features) = self.msd.forward(real_audio)
        let (fake_msd_logits, fake_msd_features) = self.msd.forward(fake_audio)
        
        return (
            real_mpd_logits, fake_mpd_logits,
            real_mpd_features, fake_mpd_features,
            real_msd_logits, fake_msd_logits,
            real_msd_features, fake_msd_features
        )
    
    fn count_parameters(self) -> Int:
        """Count total parameters in discriminators"""
        # Approximate counts
        let mpd_params = 5 * 2_000_000  # 5 period discriminators
        let msd_params = 3 * 3_000_000  # 3 scale discriminators
        return mpd_params + msd_params
    
    fn print_architecture(self):
        """Print discriminator architecture summary"""
        print("\n" + "="*60)
        print("HiFiGAN Discriminators Architecture")
        print("="*60)
        
        print("\nMulti-Period Discriminator (MPD):")
        print("  Periods:", self.mpd.periods)
        print("  Discriminators: 5 (one per period)")
        print("  Each analyzes audio reshaped by period")
        print("  2D convolutions on [time//period, period]")
        
        print("\nMulti-Scale Discriminator (MSD):")
        print("  Scales: 3 (original, 2×, 4× downsampled)")
        print("  Each scale has separate discriminator")
        print("  1D convolutions at different resolutions")
        
        print("\nTotal Parameters:", self.count_parameters())
        print("="*60 + "\n")


fn main():
    """Test HiFiGAN Discriminators"""
    print("\n🎯 HiFiGAN Discriminators Test\n")
    
    # Create discriminators
    var discriminators = HiFiGANDiscriminators()
    
    # Print architecture
    discriminators.print_architecture()
    
    # Test forward pass
    print("Testing discriminators...")
    let batch_size = 2
    let audio_length = 8192  # Short audio for testing
    
    # Create fake audio samples
    var real_audio = Tensor[DType.float32](batch_size, 1, audio_length)
    var fake_audio = Tensor[DType.float32](batch_size, 1, audio_length)
    
    rand(real_audio.data(), real_audio.num_elements())
    rand(fake_audio.data(), fake_audio.num_elements())
    
    # Normalize to [-1, 1]
    for i in range(real_audio.num_elements()):
        real_audio[i] = real_audio[i] * 2.0 - 1.0
        fake_audio[i] = fake_audio[i] * 2.0 - 1.0
    
    print("Real audio shape: [", batch_size, ", 1,", audio_length, "]")
    print("Fake audio shape: [", batch_size, ", 1,", audio_length, "]")
    
    # Forward pass
    let (
        real_mpd_logits, fake_mpd_logits,
        real_mpd_features, fake_mpd_features,
        real_msd_logits, fake_msd_logits,
        real_msd_features, fake_msd_features
    ) = discriminators.forward(real_audio, fake_audio)
    
    print("\nMulti-Period Discriminator outputs:")
    print("  Real logits:", len(real_mpd_logits), "tensors (one per period)")
    print("  Fake logits:", len(fake_mpd_logits), "tensors")
    print("  Feature maps per discriminator:", len(real_mpd_features[0]))
    
    print("\nMulti-Scale Discriminator outputs:")
    print("  Real logits:", len(real_msd_logits), "tensors (one per scale)")
    print("  Fake logits:", len(fake_msd_logits), "tensors")
    print("  Feature maps per discriminator:", len(real_msd_features[0]))
    
    print("\n✅ HiFiGAN Discriminators test complete!")
    print("\nKey Capabilities:")
    print("  • Multi-Period: Analyzes periodic structures (2,3,5,7,11)")
    print("  • Multi-Scale: Analyzes at multiple resolutions")
    print("  • Feature matching: Intermediate features for loss")
    print("  • Adversarial training: Real vs. fake discrimination")
