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


# ============================================================================
# C-Compatible Export Functions for Zig FFI
# ============================================================================

# Global manager storage for FFI context management
var _global_manager: UnsafePointer[MixedPrecisionManager] = UnsafePointer[MixedPrecisionManager]()


@export
fn mojo_mixed_precision_create(lib_path: UnsafePointer[UInt8]) -> UnsafePointer[NoneType]:
    """
    C-compatible export: Create a MixedPrecisionManager.

    Called from Zig as: mojo_mixed_precision_create(lib_path_cstr)

    Args:
        lib_path: C-string path to CUDA kernels library

    Returns:
        Opaque pointer to MixedPrecisionManager, or null on failure
    """
    try:
        var path_str = String(lib_path)
        var manager_ptr = UnsafePointer[MixedPrecisionManager].alloc(1)
        manager_ptr.init_pointee_move(MixedPrecisionManager(path_str))
        return manager_ptr.bitcast[NoneType]()
    except:
        return UnsafePointer[NoneType]()


@export
fn mojo_mixed_precision_destroy(ctx: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Destroy a MixedPrecisionManager.

    Called from Zig as: mojo_mixed_precision_destroy(ctx)

    Args:
        ctx: Opaque pointer from mojo_mixed_precision_create

    Returns:
        0 on success, -1 on failure
    """
    if not ctx:
        return -1
    var manager_ptr = ctx.bitcast[MixedPrecisionManager]()
    manager_ptr.destroy_pointee()
    manager_ptr.free()
    return 0


@export
fn mojo_convert_fp32_to_fp16(ctx: UnsafePointer[NoneType],
                              src: UnsafePointer[Float32],
                              dst: UnsafePointer[Float16],
                              count: Int32,
                              stream: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Convert FP32 tensor to FP16.

    Called from Zig as: mojo_convert_fp32_to_fp16(ctx, src, dst, count, stream)

    Args:
        ctx: Opaque pointer to MixedPrecisionManager
        src: Pointer to FP32 source tensor
        dst: Pointer to FP16 destination tensor
        count: Number of elements to convert
        stream: CUDA stream pointer

    Returns:
        CUDA status code (0 = success)
    """
    if not ctx:
        return -1
    try:
        var manager_ptr = ctx.bitcast[MixedPrecisionManager]()
        return manager_ptr[].convert_fp32_to_fp16(src, dst, Int(count), stream)
    except:
        return -1


@export
fn mojo_convert_fp16_to_fp32(ctx: UnsafePointer[NoneType],
                              src: UnsafePointer[Float16],
                              dst: UnsafePointer[Float32],
                              count: Int32,
                              stream: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Convert FP16 tensor to FP32.

    Called from Zig as: mojo_convert_fp16_to_fp32(ctx, src, dst, count, stream)

    Args:
        ctx: Opaque pointer to MixedPrecisionManager
        src: Pointer to FP16 source tensor
        dst: Pointer to FP32 destination tensor
        count: Number of elements to convert
        stream: CUDA stream pointer

    Returns:
        CUDA status code (0 = success)
    """
    if not ctx:
        return -1
    try:
        var manager_ptr = ctx.bitcast[MixedPrecisionManager]()
        return manager_ptr[].convert_fp16_to_fp32(src, dst, Int(count), stream)
    except:
        return -1


@export
fn mojo_scale_tensor_fp16(ctx: UnsafePointer[NoneType],
                           tensor: UnsafePointer[Float16],
                           count: Int32,
                           scale: Float32,
                           stream: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Scale FP16 tensor in-place.

    Called from Zig as: mojo_scale_tensor_fp16(ctx, tensor, count, scale, stream)

    Args:
        ctx: Opaque pointer to MixedPrecisionManager
        tensor: Pointer to FP16 tensor (modified in-place)
        count: Number of elements
        scale: Scaling factor
        stream: CUDA stream pointer

    Returns:
        CUDA status code (0 = success)
    """
    if not ctx:
        return -1
    try:
        var manager_ptr = ctx.bitcast[MixedPrecisionManager]()
        return manager_ptr[].scale_tensor_fp16(tensor, Int(count), scale, stream)
    except:
        return -1


@export
fn mojo_check_inf_nan_fp16(ctx: UnsafePointer[NoneType],
                            tensor: UnsafePointer[Float16],
                            count: Int32,
                            result: UnsafePointer[Int32],
                            stream: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Check FP16 tensor for inf/nan values.

    Called from Zig as: mojo_check_inf_nan_fp16(ctx, tensor, count, result, stream)

    Args:
        ctx: Opaque pointer to MixedPrecisionManager
        tensor: Pointer to FP16 tensor to check
        count: Number of elements
        result: Output pointer (set to 1 if inf/nan found, 0 otherwise)
        stream: CUDA stream pointer

    Returns:
        CUDA status code (0 = success)
    """
    if not ctx:
        return -1
    try:
        var manager_ptr = ctx.bitcast[MixedPrecisionManager]()
        return manager_ptr[].check_inf_nan_fp16(tensor, Int(count), result, stream)
    except:
        return -1


@export
fn mojo_get_loss_scale(ctx: UnsafePointer[NoneType]) -> Float32:
    """
    C-compatible export: Get current loss scale value.

    Called from Zig as: mojo_get_loss_scale(ctx)

    Args:
        ctx: Opaque pointer to MixedPrecisionManager

    Returns:
        Current loss scale, or -1.0 on error
    """
    if not ctx:
        return -1.0
    var manager_ptr = ctx.bitcast[MixedPrecisionManager]()
    return manager_ptr[].get_loss_scale()


@export
fn mojo_update_loss_scale(ctx: UnsafePointer[NoneType], found_inf: Int32) -> Int32:
    """
    C-compatible export: Update dynamic loss scale based on overflow detection.

    Called from Zig as: mojo_update_loss_scale(ctx, found_inf)

    Args:
        ctx: Opaque pointer to MixedPrecisionManager
        found_inf: 1 if inf/nan was found in gradients, 0 otherwise

    Returns:
        0 on success, -1 on failure
    """
    if not ctx:
        return -1
    var manager_ptr = ctx.bitcast[MixedPrecisionManager]()
    manager_ptr[].update_scale(found_inf != 0)
    return 0
