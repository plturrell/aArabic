"""
Unified Inference Bridge API

This module provides the high-level Mojo interface to the Zig inference engine.
All services (LLM, Embedding, Translation, RAG) use this bridge to access
the native inference capabilities.

FFI Exports from Zig (inference/engine/mojo_bridge.zig):
- inference_load_model(path: *c_char) -> i32
- inference_generate(prompt: *u8, len: usize, max_tokens: u32, temp: f32, buf: *u8, buf_size: usize) -> i32
- inference_is_loaded() -> i32
- inference_get_info(buf: *u8, buf_size: usize) -> i32
- inference_unload() -> void
"""

from memory import UnsafePointer
from sys.ffi import DLHandle, external_call


# Library path constants
alias LIB_PATH_RELEASE = "./inference/engine/zig-out/lib/libinference.dylib"
alias LIB_PATH_RELEASE_LINUX = "./inference/engine/zig-out/lib/libinference.so"
alias LIB_PATH_FALLBACK = "./inference/engine/zig-out/bin/test_mojo_bridge"

# Buffer sizes
alias INFO_BUFFER_SIZE = 512
alias RESPONSE_BUFFER_SIZE = 8192

alias DEFAULT_MODEL_DIR = "./models"
alias VENDOR_MODEL_DIR = "./vendor/layerModels/huggingFace"


fn _string_to_c_ptr(value: String) -> UnsafePointer[UInt8]:
    """Allocate a null-terminated C string copy."""
    var bytes = value.as_bytes()
    var ptr = UnsafePointer[UInt8].alloc(len(bytes) + 1)
    for i in range(len(bytes)):
        ptr[i] = bytes[i]
    ptr[len(bytes)] = 0
    return ptr


fn resolve_model_path(model_name: String) -> String:
    """
    Resolve a model identifier to a filesystem path.
    Accepts full paths and falls back to known model roots.
    """
    if "/" in model_name or model_name.startswith("."):
        return model_name

    if model_name.startswith("vendor:"):
        var vendor_suffix = model_name[7:]
        return String(VENDOR_MODEL_DIR) + String("/") + vendor_suffix

    return String(DEFAULT_MODEL_DIR) + String("/") + model_name


struct InferenceEngine:
    """High-level wrapper for the Zig inference engine."""

    var _lib_handle: DLHandle
    var _is_initialized: Bool
    var _model_loaded: Bool
    var _model_path: String
    var _lib_path: String

    fn __init__(out self):
        """Initialize the inference engine (lazy load)."""
        self._lib_handle = DLHandle()
        self._is_initialized = False
        self._model_loaded = False
        self._model_path = ""
        self._lib_path = ""

    fn load_library(mut self, path: String = "") raises:
        """Load the native inference library."""
        if path != "":
            self._lib_handle = DLHandle(path)
            self._lib_path = path
            self._is_initialized = True
            return

        try:
            self._lib_handle = DLHandle(LIB_PATH_RELEASE)
            self._lib_path = LIB_PATH_RELEASE
        except:
            try:
                self._lib_handle = DLHandle(LIB_PATH_RELEASE_LINUX)
                self._lib_path = LIB_PATH_RELEASE_LINUX
            except:
                self._lib_handle = DLHandle(LIB_PATH_FALLBACK)
                self._lib_path = LIB_PATH_FALLBACK

        self._is_initialized = True

    fn is_model_loaded(self) -> Bool:
        """Check if a model is currently loaded."""
        if not self._is_initialized:
            return False

        var result = external_call["inference_is_loaded", Int32](self._lib_handle)
        return result == 1

    fn load_model(mut self, model_path: String) raises -> Bool:
        """
        Load a model from the specified path.

        Args:
            model_path: Path to the model directory (HuggingFace format)

        Returns:
            True if model loaded successfully
        """
        if not self._is_initialized:
            self.load_library()

        var c_path = _string_to_c_ptr(model_path)
        var result = external_call["inference_load_model", Int32](self._lib_handle, c_path)
        c_path.free()

        if result == 0:
            self._model_loaded = True
            self._model_path = model_path
            return True

        return False

    fn generate(
        self,
        prompt: String,
        max_tokens: Int = 100,
        temperature: Float32 = 0.7,
    ) raises -> String:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        if not self._is_initialized:
            raise Error("Inference engine not initialized")

        if not self._model_loaded:
            raise Error("Model not loaded")

        var prompt_bytes = prompt.as_bytes()
        var c_prompt = UnsafePointer[UInt8].alloc(len(prompt_bytes) + 1)
        for i in range(len(prompt_bytes)):
            c_prompt[i] = prompt_bytes[i]
        c_prompt[len(prompt_bytes)] = 0

        var result_buffer = UnsafePointer[UInt8].alloc(RESPONSE_BUFFER_SIZE)
        var length = external_call["inference_generate", Int32](
            self._lib_handle,
            c_prompt,
            len(prompt_bytes),
            Int32(max_tokens),
            temperature,
            result_buffer,
            RESPONSE_BUFFER_SIZE
        )

        c_prompt.free()

        if length <= 0:
            result_buffer.free()
            return "Error: Generation failed"

        var result = String(result_buffer, length)
        result_buffer.free()
        return result

    fn get_info(self) raises -> String:
        """Get information about the loaded model."""
        if not self._is_initialized:
            return "Engine not initialized"

        var buffer = UnsafePointer[UInt8].alloc(INFO_BUFFER_SIZE)
        var length = external_call["inference_get_info", Int32](
            self._lib_handle,
            buffer,
            INFO_BUFFER_SIZE
        )

        if length <= 0:
            buffer.free()
            return "No model info available"

        var info = String(buffer, length)
        buffer.free()
        return info

    fn unload(mut self):
        """Unload the current model and free resources."""
        if not self._is_initialized:
            return

        external_call["inference_unload", NoneType](self._lib_handle)
        self._model_loaded = False
        self._model_path = ""


# Convenience functions for direct access

fn create_inference_engine() -> InferenceEngine:
    """Factory function to create an inference engine instance."""
    return InferenceEngine()


var _shared_engine = InferenceEngine()
var _shared_engine_ready: Bool = False
var _shared_model_path: String = ""


fn ensure_shared_engine() raises:
    """Initialize the shared engine once."""
    if not _shared_engine_ready:
        _shared_engine.load_library()
        _shared_engine_ready = True


fn ensure_model_loaded(model_path: String) raises -> Bool:
    """Load the requested model if not already loaded."""
    ensure_shared_engine()
    if _shared_model_path == model_path and _shared_engine.is_model_loaded():
        return True

    var loaded = _shared_engine.load_model(model_path)
    if loaded:
        _shared_model_path = model_path
    return loaded


fn shared_generate(prompt: String, max_tokens: Int = 100, temperature: Float32 = 0.7) raises -> String:
    """Generate text using the shared engine."""
    ensure_shared_engine()
    return _shared_engine.generate(prompt, max_tokens, temperature)


fn shared_is_model_loaded() -> Bool:
    """Check if the shared engine has a loaded model."""
    if not _shared_engine_ready:
        return False
    return _shared_engine.is_model_loaded()


fn shared_get_info() raises -> String:
    """Get model info from the shared engine."""
    ensure_shared_engine()
    return _shared_engine.get_info()


fn quick_generate(prompt: String, max_tokens: Int = 100) raises -> String:
    """
    Quick generation without managing engine lifecycle.

    Note: This creates a new engine instance each call.
    For production use, maintain a persistent InferenceEngine.
    """
    var engine = InferenceEngine()
    engine.load_library()
    return engine.generate(prompt, max_tokens)
