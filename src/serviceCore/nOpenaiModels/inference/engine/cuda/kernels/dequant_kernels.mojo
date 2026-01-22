"""
CUDA Dequantization Kernels for GGUF Quantized Weights
Supports Q4_0, Q8_0, and Q4_K formats with FP16 output for Tensor Core consumption.

FFI Bindings: cuLaunchKernel for custom dequant kernels, cuModuleLoad, cuModuleGetFunction

Block Formats:
- Q4_0: 18 bytes -> 32 FP16 values (2B f16 scale + 16B packed 4-bit)
- Q8_0: 36 bytes -> 32 FP16 values (4B f32 scale + 32B int8)
- Q4_K: 144 bytes -> 256 FP16 values (complex K-quant format)
"""

from sys.ffi import DLHandle, external_call
from memory import UnsafePointer

# CUDA constants
alias CUDA_SUCCESS: Int32 = 0
alias Q4_0_BLOCK_SIZE: Int32 = 32
alias Q4_0_BLOCK_BYTES: Int32 = 18
alias Q8_0_BLOCK_SIZE: Int32 = 32
alias Q8_0_BLOCK_BYTES: Int32 = 36
alias Q4_K_BLOCK_SIZE: Int32 = 256
alias Q4_K_BLOCK_BYTES: Int32 = 144


struct CudaModule:
    """Wrapper for CUDA module and kernel functions."""
    var _module: UnsafePointer[NoneType]
    var _lib: DLHandle
    var _initialized: Bool

    fn __init__(inout self, lib_path: String = "libcuda.so") raises:
        """Initialize CUDA driver API handle."""
        self._lib = DLHandle(lib_path)
        self._module = UnsafePointer[NoneType]()
        self._initialized = False

    fn __del__(owned self):
        if self._initialized:
            _ = external_call["cuModuleUnload", Int32](self._module)

    fn load_module(inout self, ptx_path: String) raises -> Int32:
        """Load a PTX or cubin module."""
        var status = external_call["cuModuleLoad", Int32](
            UnsafePointer.address_of(self._module), ptx_path.unsafe_ptr())
        if status == CUDA_SUCCESS:
            self._initialized = True
        return status

    fn get_function(self, func_name: String) raises -> UnsafePointer[NoneType]:
        """Get kernel function from loaded module."""
        var func = UnsafePointer[NoneType]()
        var status = external_call["cuModuleGetFunction", Int32](
            UnsafePointer.address_of(func), self._module, func_name.unsafe_ptr())
        if status != CUDA_SUCCESS:
            raise Error("Failed to get function: " + func_name + " (error " + String(status) + ")")
        return func


struct DequantKernels:
    """CUDA dequantization kernel wrappers for GGUF formats."""
    var _lib: DLHandle
    var _module: CudaModule
    var _dequant_q4_0_func: UnsafePointer[NoneType]
    var _dequant_q8_0_func: UnsafePointer[NoneType]
    var _dequant_q4_k_func: UnsafePointer[NoneType]
    var _initialized: Bool

    fn __init__(inout self, cuda_lib: String = "libcuda.so",
                kernel_lib: String = "libdequant_kernels.so") raises:
        """Initialize dequantization kernel handles."""
        self._lib = DLHandle(kernel_lib)
        self._module = CudaModule(cuda_lib)
        self._dequant_q4_0_func = UnsafePointer[NoneType]()
        self._dequant_q8_0_func = UnsafePointer[NoneType]()
        self._dequant_q4_k_func = UnsafePointer[NoneType]()
        self._initialized = False

    fn load_kernels(inout self, ptx_path: String) raises:
        """Load dequantization kernels from PTX module."""
        var status = self._module.load_module(ptx_path)
        if status != CUDA_SUCCESS:
            raise Error("Failed to load PTX module: " + String(status))
        self._dequant_q4_0_func = self._module.get_function("dequant_q4_0_kernel")
        self._dequant_q8_0_func = self._module.get_function("dequant_q8_0_kernel")
        self._dequant_q4_k_func = self._module.get_function("dequant_q4_k_kernel")
        self._initialized = True

    fn dequant_q4_0(self, input: UnsafePointer[UInt8], output: UnsafePointer[Float16],
                   num_blocks: Int32, stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """
        Dequantize Q4_0 blocks to FP16.
        
        Q4_0 format (18 bytes per 32 values):
        - 2 bytes: f16 scale
        - 16 bytes: 32x 4-bit signed values (packed, 2 per byte)
        
        Dequantization: value = (int4 - 8) * scale
        
        Args:
            input: Pointer to Q4_0 quantized data
            output: Pointer to FP16 output buffer
            num_blocks: Number of Q4_0 blocks to process
            stream: CUDA stream for async execution
        """
        return external_call["dequant_q4_0_fp16", Int32](
            input, output, num_blocks, stream)

    fn dequant_q8_0(self, input: UnsafePointer[UInt8], output: UnsafePointer[Float16],
                   num_blocks: Int32, stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """
        Dequantize Q8_0 blocks to FP16.
        
        Q8_0 format (36 bytes per 32 values):
        - 4 bytes: f32 scale
        - 32 bytes: 32x int8 values
        
        Dequantization: value = int8 * scale
        
        Args:
            input: Pointer to Q8_0 quantized data
            output: Pointer to FP16 output buffer
            num_blocks: Number of Q8_0 blocks to process
            stream: CUDA stream for async execution
        """
        return external_call["dequant_q8_0_fp16", Int32](
            input, output, num_blocks, stream)

    fn dequant_q4_k(self, input: UnsafePointer[UInt8], output: UnsafePointer[Float16],
                   num_blocks: Int32, stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """
        Dequantize Q4_K blocks to FP16.

        Q4_K format (144 bytes per 256 values):
        - 2 bytes: f16 d (super-block scale)
        - 2 bytes: f16 dmin (super-block min)
        - 12 bytes: packed scales/mins for 8 sub-blocks
        - 128 bytes: 256x 4-bit values (packed)

        Args:
            input: Pointer to Q4_K quantized data
            output: Pointer to FP16 output buffer
            num_blocks: Number of Q4_K blocks to process
            stream: CUDA stream for async execution
        """
        return external_call["dequant_q4_k_fp16", Int32](
            input, output, num_blocks, stream)

    fn launch_dequant_kernel(self, func: UnsafePointer[NoneType],
                            grid_dim_x: Int32, grid_dim_y: Int32, grid_dim_z: Int32,
                            block_dim_x: Int32, block_dim_y: Int32, block_dim_z: Int32,
                            shared_mem: Int32, stream: UnsafePointer[NoneType],
                            kernel_params: UnsafePointer[UnsafePointer[NoneType]]) raises -> Int32:
        """Launch a CUDA kernel using cuLaunchKernel."""
        return external_call["cuLaunchKernel", Int32](
            func, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem, stream, kernel_params, UnsafePointer[NoneType]())


# ============================================================================
# C-Compatible Export Functions for Zig FFI
# ============================================================================

fn _calculate_grid_size(num_elements: Int32, block_size: Int32) -> Int32:
    """Calculate grid size for kernel launch."""
    return (num_elements + block_size - 1) // block_size


@export
fn mojo_dequant_q4_0_fp16(input_ptr: UnsafePointer[UInt8], output_ptr: UnsafePointer[Float16],
                          num_blocks: Int32, stream_ptr: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Dequantize Q4_0 to FP16.

    Called from Zig as: mojo_dequant_q4_0_fp16(input, output, num_blocks, stream)

    Block format (18 bytes -> 32 FP16):
    - scale: f16 (2 bytes)
    - qs: 16 bytes packed 4-bit values

    Each 4-bit value v in [0,15] maps to (v - 8) * scale
    """
    return external_call["dequant_q4_0_fp16", Int32](input_ptr, output_ptr, num_blocks, stream_ptr)


@export
fn mojo_dequant_q8_0_fp16(input_ptr: UnsafePointer[UInt8], output_ptr: UnsafePointer[Float16],
                          num_blocks: Int32, stream_ptr: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Dequantize Q8_0 to FP16.

    Called from Zig as: mojo_dequant_q8_0_fp16(input, output, num_blocks, stream)

    Block format (36 bytes -> 32 FP16):
    - scale: f32 (4 bytes)
    - qs: 32 bytes int8 values

    Each int8 value q maps to q * scale
    """
    return external_call["dequant_q8_0_fp16", Int32](input_ptr, output_ptr, num_blocks, stream_ptr)


@export
fn mojo_dequant_q4_k_fp16(input_ptr: UnsafePointer[UInt8], output_ptr: UnsafePointer[Float16],
                          num_blocks: Int32, stream_ptr: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Dequantize Q4_K to FP16.

    Called from Zig as: mojo_dequant_q4_k_fp16(input, output, num_blocks, stream)

    Block format (144 bytes -> 256 FP16):
    - d: f16 super-block scale (2 bytes)
    - dmin: f16 super-block min (2 bytes)
    - scales: 12 bytes packed sub-block scales/mins
    - qs: 128 bytes packed 4-bit values
    """
    return external_call["dequant_q4_k_fp16", Int32](input_ptr, output_ptr, num_blocks, stream_ptr)


@export
fn mojo_dequant_get_output_size(quant_type: Int32, num_blocks: Int32) -> Int32:
    """
    Get output buffer size in FP16 elements for dequantization.

    Args:
        quant_type: 2=Q4_0, 8=Q8_0, 12=Q4_K (matches GGUF enum)
        num_blocks: Number of quantized blocks

    Returns:
        Number of FP16 elements in output
    """
    if quant_type == 2:  # Q4_0
        return num_blocks * Q4_0_BLOCK_SIZE
    elif quant_type == 8:  # Q8_0
        return num_blocks * Q8_0_BLOCK_SIZE
    elif quant_type == 12:  # Q4_K
        return num_blocks * Q4_K_BLOCK_SIZE
    else:
        return 0


@export
fn mojo_dequant_get_input_size(quant_type: Int32, num_blocks: Int32) -> Int32:
    """
    Get input buffer size in bytes for quantized data.

    Args:
        quant_type: 2=Q4_0, 8=Q8_0, 12=Q4_K (matches GGUF enum)
        num_blocks: Number of quantized blocks

    Returns:
        Size in bytes of quantized input
    """
    if quant_type == 2:  # Q4_0
        return num_blocks * Q4_0_BLOCK_BYTES
    elif quant_type == 8:  # Q8_0
        return num_blocks * Q8_0_BLOCK_BYTES
    elif quant_type == 12:  # Q4_K
        return num_blocks * Q4_K_BLOCK_BYTES
    else:
        return 0


# ============================================================================
# Convenience Functions
# ============================================================================

fn dequant_tensor_q4_0(kernels: DequantKernels, input: UnsafePointer[UInt8],
                       output: UnsafePointer[Float16], num_elements: Int,
                       stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
    """Dequantize a Q4_0 tensor (convenience wrapper)."""
    var num_blocks = Int32((num_elements + 31) // 32)
    return kernels.dequant_q4_0(input, output, num_blocks, stream)


fn dequant_tensor_q8_0(kernels: DequantKernels, input: UnsafePointer[UInt8],
                       output: UnsafePointer[Float16], num_elements: Int,
                       stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
    """Dequantize a Q8_0 tensor (convenience wrapper)."""
    var num_blocks = Int32((num_elements + 31) // 32)
    return kernels.dequant_q8_0(input, output, num_blocks, stream)


fn dequant_tensor_q4_k(kernels: DequantKernels, input: UnsafePointer[UInt8],
                       output: UnsafePointer[Float16], num_elements: Int,
                       stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
    """Dequantize a Q4_K tensor (convenience wrapper)."""
    var num_blocks = Int32((num_elements + 255) // 256)
    return kernels.dequant_q4_k(input, output, num_blocks, stream)

