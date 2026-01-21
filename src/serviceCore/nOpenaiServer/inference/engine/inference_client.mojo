"""
HuggingFace Inference Client - Mojo wrapper for Zig inference engine
Calls C API from mojo_bridge.zig
"""

from sys.ffi import DLHandle, external_call
from memory import UnsafePointer
from pathlib import Path


struct InferenceClient:
    """Client for HuggingFace inference via Zig C API."""
    
    var lib: DLHandle
    var default_model: String
    
    fn __init__(inout self, lib_path: String, default_model: String = "default") raises:
        """Initialize inference client.
        
        Args:
            lib_path: Path to libzig_inference shared library.
        """
        self.lib = DLHandle(lib_path)
        self.default_model = default_model
        print("✅ Loaded inference library:", lib_path)
    
    fn load_model(inout self, model_id: String, model_path: String) raises -> Bool:
        """Load HuggingFace model.
        
        Args:
            model_id: Identifier for the model handle.
            model_path: Path to HuggingFace model directory.
            
        Returns:
            True if successful, False otherwise.
        """
        print("\n🔧 Loading model", model_id, "from:", model_path)
        
        # Convert string to C string
        var id_bytes = model_id.as_bytes()
        var c_id = UnsafePointer[UInt8].alloc(len(id_bytes) + 1)
        for i in range(len(id_bytes)):
            c_id[i] = id_bytes[i]
        c_id[len(id_bytes)] = 0

        var path_bytes = model_path.as_bytes()
        var c_path = UnsafePointer[UInt8].alloc(len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            c_path[i] = path_bytes[i]
        c_path[len(path_bytes)] = 0  # Null terminator
        
        # Call C API
        var result = external_call["inference_load_model_v2", Int32](
            self.lib,
            c_id,
            c_path
        )
        
        c_id.free()
        c_path.free()
        
        if result == 0:
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Failed to load model")
            return False
    
    fn is_loaded(self, model_id: String = "") -> Bool:
        """Check if model is loaded.
        
        Returns:
            True if model is loaded.
        """
        var id = model_id if model_id != "" else self.default_model
        var bytes = id.as_bytes()
        var c_id = UnsafePointer[UInt8].alloc(len(bytes) + 1)
        for i in range(len(bytes)):
            c_id[i] = bytes[i]
        c_id[len(bytes)] = 0

        var result = external_call["inference_is_loaded_v2", Int32](self.lib, c_id)
        c_id.free()
        return result == 1
    
    fn get_info(self, model_id: String = "") raises -> String:
        """Get model information.
        
        Returns:
            Model info string.
        """
        var id = model_id if model_id != "" else self.default_model
        var bytes = id.as_bytes()
        var c_id = UnsafePointer[UInt8].alloc(len(bytes) + 1)
        for i in range(len(bytes)):
            c_id[i] = bytes[i]
        c_id[len(bytes)] = 0

        var buffer = UnsafePointer[UInt8].alloc(512)
        
        var length = external_call["inference_get_info_v2", Int32](
            self.lib,
            c_id,
            buffer,
            512
        )
        
        c_id.free()
        if length <= 0:
            buffer.free()
            return "No model info available"
        
        # Convert C string to Mojo String
        var result = String(buffer, length)
        buffer.free()
        return result
    
    fn generate(
        self,
        prompt: String,
        model_id: String = "",
        max_tokens: Int = 50,
        temperature: Float32 = 0.7,
    ) raises -> String:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt.
            model_id: Target model identifier (defaults to client's default).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text.
        """
        var id = model_id if model_id != "" else self.default_model
        if not self.is_loaded(id):
            return "Error: Model not loaded"
        
        print("\n🔮 Generating response...")
        print("   Prompt:", prompt)
        print("   Max tokens:", max_tokens)
        print("   Temperature:", temperature)
        
        var id_bytes = id.as_bytes()
        var c_id = UnsafePointer[UInt8].alloc(len(id_bytes) + 1)
        for i in range(len(id_bytes)):
            c_id[i] = id_bytes[i]
        c_id[len(id_bytes)] = 0

        # Convert prompt to C string
        var prompt_bytes = prompt.as_bytes()
        var c_prompt = UnsafePointer[UInt8].alloc(len(prompt_bytes))
        for i in range(len(prompt_bytes)):
            c_prompt[i] = prompt_bytes[i]
        
        # Allocate result buffer
        var result_buffer = UnsafePointer[UInt8].alloc(2048)
        
        # Call C API
        var length = external_call["inference_generate_v2", Int32](
            self.lib,
            c_id,
            c_prompt,
            len(prompt_bytes),
            Int32(max_tokens),
            temperature,
            result_buffer,
            2048
        )
        
        c_id.free()
        c_prompt.free()
        
        if length <= 0:
            result_buffer.free()
            return "Error: Generation failed"
        
        # Convert result to String
        var result = String(result_buffer, length)
        result_buffer.free()
        
        print("✅ Generated", length, "bytes")
        return result
    
    fn unload(inout self, model_id: String = ""):
        """Unload model and cleanup."""
        var id = model_id if model_id != "" else self.default_model
        var bytes = id.as_bytes()
        var c_id = UnsafePointer[UInt8].alloc(len(bytes) + 1)
        for i in range(len(bytes)):
            c_id[i] = bytes[i]
        c_id[len(bytes)] = 0

        external_call["inference_unload_v2", NoneType](self.lib, c_id)
        c_id.free()
        print("✅ Model unloaded:", id)


fn print_banner():
    """Print banner."""
    print("=" * 80)
    print("🤖 HuggingFace Inference Client - Mojo → Zig Integration")
    print("=" * 80)
    print()


fn test_inference_client(model_path: String, lib_path: String) raises:
    """Test inference client with real model.
    
    Args:
        model_path: Path to HuggingFace model.
        lib_path: Path to shared library.
    """
    print_banner()
    
    # Create client
    var client = InferenceClient(lib_path)
    
    # Load model
    var loaded = client.load_model(model_path)
    
    if not loaded:
        print("❌ Failed to load model")
        return
    
    # Check status
    print("\n📊 Status Check:")
    print("   Model loaded:", client.is_loaded())
    
    # Get info
    print("\n📋 Model Info:")
    var info = client.get_info()
    print("  ", info)
    
    # Test generation
    print("\n🧪 Testing Generation:")
    var prompts = List[String]()
    prompts.append("Hello, how are you?")
    prompts.append("What is the capital of France?")
    prompts.append("Explain quantum computing")
    
    for i in range(len(prompts)):
        var prompt = prompts[i]
        print("\n" + "─" * 80)
        print("Test", i + 1, "of", len(prompts))
        
        var response = client.generate(prompt, max_tokens=50, temperature=0.7)
        
        print("\n💬 Response:")
        print("  ", response)
    
    # Cleanup
    print("\n" + "─" * 80)
    client.unload()
    
    print("\n✅ All tests complete!")
    print("=" * 80)


fn main() raises:
    """Main test function."""
    
    # Default paths
    var model_path = "/Users/user/Documents/arabic_folder/vendor/layerModels/huggingFace/Qwen/Qwen2.5-0.5B-Instruct"
    var lib_path = "./test_mojo_bridge"  # Our standalone binary acts as library
    
    print("🎯 Configuration:")
    print("   Model:", model_path)
    print("   Library:", lib_path)
    print()
    
    # Note: For now, we'll test the C API directly via standalone binary
    # In production, this would load a shared library (.dylib/.so)
    
    print("⚠️  Note: This test requires building the Zig bridge first:")
    print("   cd inference && zig build-exe mojo_bridge.zig -O ReleaseFast")
    print()
    print("   For Mojo FFI, build as shared library:")
    print("   zig build-lib mojo_bridge.zig -dynamic -O ReleaseFast")
    print()
    
    # For now, show what the API would look like
    print("📝 Example API Usage:")
    print()
    print("   var client = InferenceClient(\"./libzig_inference.dylib\")")
    print("   client.load_model(model_path)")
    print("   var response = client.generate(\"Hello!\", max_tokens=50)")
    print("   print(response)")
    print("   client.unload()")
    print()
