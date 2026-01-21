"""
Mixed Precision Utilities for CUDA
FP32/FP16 conversion and scaling utilities for mixed precision training/inference.

FFI Bindings: fp32_to_fp16_kernel, fp16_to_fp32_kernel, scale_tensor_kernel
"""

from sys.ffi import DLHandle, external_call
from memory import UnsafePointer

# Loss scale constants
alias INITIAL_LOSS_SCALE: Float32 = 65536.0
alias MIN_LOSS_SCALE: Float32 = 1.0
alias SCALE_FACTOR: Float32 = 2.0
alias SCALE_WINDOW: Int32 = 2000


struct MixedPrecisionManager:
    """Manages FP16/FP32 conversions and loss scaling for mixed precision."""
    var _lib: DLHandle
    var _loss_scale: Float32
    var _growth_interval: Int32
    var _steps_since_growth: Int32
    var _overflow_count: Int32

    fn __init__(inout self, lib_path: String = "libcuda_kernels.so") raises:
        self._lib = DLHandle(lib_path)
        self._loss_scale = INITIAL_LOSS_SCALE
        self._growth_interval = SCALE_WINDOW
        self._steps_since_growth = 0
        self._overflow_count = 0

    fn convert_fp32_to_fp16(self, src: UnsafePointer[Float32], dst: UnsafePointer[Float16],
                           count: Int, stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Convert FP32 tensor to FP16 on GPU."""
        return external_call["fp32_to_fp16_kernel", Int32](src, dst, Int32(count), stream)

    fn convert_fp16_to_fp32(self, src: UnsafePointer[Float16], dst: UnsafePointer[Float32],
                           count: Int, stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Convert FP16 tensor to FP32 on GPU."""
        return external_call["fp16_to_fp32_kernel", Int32](src, dst, Int32(count), stream)

    fn scale_tensor_fp16(self, tensor: UnsafePointer[Float16], count: Int, scale: Float32,
                        stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Apply scaling factor to FP16 tensor in-place."""
        return external_call["scale_tensor_fp16_kernel", Int32](tensor, Int32(count), scale, stream)

    fn scale_tensor_fp32(self, tensor: UnsafePointer[Float32], count: Int, scale: Float32,
                        stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Apply scaling factor to FP32 tensor in-place."""
        return external_call["scale_tensor_fp32_kernel", Int32](tensor, Int32(count), scale, stream)

    fn check_inf_nan_fp16(self, tensor: UnsafePointer[Float16], count: Int,
                         result: UnsafePointer[Int32],
                         stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Check FP16 tensor for inf/nan values (overflow detection)."""
        return external_call["check_inf_nan_fp16_kernel", Int32](tensor, Int32(count), result, stream)

    fn get_loss_scale(self) -> Float32:
        return self._loss_scale

    fn update_scale(inout self, found_inf: Bool):
        """Update loss scale based on overflow detection."""
        if found_inf:
            self._loss_scale = max(self._loss_scale / SCALE_FACTOR, MIN_LOSS_SCALE)
            self._steps_since_growth = 0
            self._overflow_count += 1
        else:
            self._steps_since_growth += 1
            if self._steps_since_growth >= self._growth_interval:
                self._loss_scale *= SCALE_FACTOR
                self._steps_since_growth = 0

    fn scale_loss(self, loss: Float32) -> Float32:
        return loss * self._loss_scale

    fn unscale_gradients(self, grads: UnsafePointer[Float32], count: Int,
                        stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Unscale gradients after backward pass."""
        return self.scale_tensor_fp32(grads, count, 1.0 / self._loss_scale, stream)


fn convert_model_to_fp16(weights: UnsafePointer[Float32], weights_fp16: UnsafePointer[Float16],
                        count: Int, manager: MixedPrecisionManager) raises -> Int32:
    """Convert model weights from FP32 to FP16."""
    return manager.convert_fp32_to_fp16(weights, weights_fp16, count)


fn convert_activations_to_fp32(activations: UnsafePointer[Float16], activations_fp32: UnsafePointer[Float32],
                              count: Int, manager: MixedPrecisionManager) raises -> Int32:
    """Convert activations from FP16 to FP32 for loss computation."""
    return manager.convert_fp16_to_fp32(activations, activations_fp32, count)
