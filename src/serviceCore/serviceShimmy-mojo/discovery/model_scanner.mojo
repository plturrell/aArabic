"""
Model Discovery System
Auto-detects GGUF models from various sources:
- HuggingFace cache (~/.cache/huggingface)
- Ollama models (~/.ollama/models)
- Local directories (./models, ~/models)
"""

from pathlib import Path
from python import Python
from collections import List, Dict
from core.gguf_parser import GGUFParser

# ============================================================================
# Model Information Structure
# ============================================================================

struct ModelInfo:
    """Information about a discovered model."""
    var name: String
    var path: String
    var size_bytes: Int
    var format: String  # "gguf", "ggml", etc.
    var source: String  # "huggingface", "ollama", "local"
    var quantization: String  # "Q4_0", "Q8_0", "F16", etc.
    var architecture: String  # "llama", "phi", "mistral", etc.
    
    fn __init__(
        inout self,
        name: String,
        path: String,
        size_bytes: Int = 0,
        format: String = "gguf",
        source: String = "local",
        quantization: String = "unknown",
        architecture: String = "unknown"
    ):
        self.name = name
        self.path = path
        self.size_bytes = size_bytes
        self.format = format
        self.source = source
        self.quantization = quantization
        self.architecture = architecture
    
    fn size_mb(self) -> Float32:
        """Get size in MB."""
        return Float32(self.size_bytes) / (1024.0 * 1024.0)
    
    fn size_gb(self) -> Float32:
        """Get size in GB."""
        return self.size_mb() / 1024.0

# ============================================================================
# Model Scanner
# ============================================================================

struct ModelScanner:
    """
    Scans filesystem for GGUF models.
    Checks multiple common locations.
    """
    var models: List[ModelInfo]
    var scan_paths: List[String]
    
    fn __init__(inout self):
        self.models = List[ModelInfo]()
        self.scan_paths = List[String]()
        
        # Add default scan paths
        self.add_default_paths()
    
    fn add_default_paths(inout self):
        """Add standard model locations."""
        var py = Python.import_module("os")
        var home = str(py.path.expanduser("~"))
        
        # HuggingFace cache
        self.scan_paths.append(home + "/.cache/huggingface/hub")
        
        # Ollama models
        self.scan_paths.append(home + "/.ollama/models")
        
        # Local directories
        self.scan_paths.append("./models")
        self.scan_paths.append(home + "/models")
        self.scan_paths.append(home + "/Downloads")
        
        print("📁 Configured scan paths:")
        for i in range(len(self.scan_paths)):
            print(f"   {i+1}. {self.scan_paths[i]}")
    
    fn add_scan_path(inout self, path: String):
        """Add custom scan path."""
        self.scan_paths.append(path)
    
    fn scan(inout self) raises -> Int:
        """
        Scan all configured paths for GGUF models.
        Returns number of models found.
        """
        print()
        print("=" * 80)
        print("🔍 Scanning for GGUF Models")
        print("=" * 80)
        print()
        
        var py = Python.import_module("os")
        var total_found = 0
        
        for i in range(len(self.scan_paths)):
            var path = self.scan_paths[i]
            print(f"Scanning: {path}")
            
            # Check if path exists
            if not py.path.exists(path):
                print(f"  ⚠️  Path does not exist")
                continue
            
            # Scan directory
            var found = self.scan_directory(path)
            total_found += found
            
            if found > 0:
                print(f"  ✅ Found {found} model(s)")
            else:
                print(f"  📭 No models found")
        
        print()
        print("=" * 80)
        print(f"✅ Scan complete: {total_found} total models")
        print("=" * 80)
        print()
        
        return total_found
    
    fn scan_directory(inout self, directory: String) raises -> Int:
        """Scan a single directory for GGUF files."""
        var py = Python.import_module("os")
        var found = 0
        
        # Walk directory tree
        for root_tuple in py.walk(directory):
            var root = str(root_tuple[0])
            var files = root_tuple[2]
            
            for file_obj in files:
                var filename = str(file_obj)
                
                # Check if GGUF file
                if filename.endswith(".gguf") or filename.endswith(".ggml"):
                    var filepath = py.path.join(root, filename)
                    
                    # Get file size
                    var size_bytes = int(py.path.getsize(filepath))
                    
                    # Determine source
                    var source = "local"
                    if "huggingface" in root:
                        source = "huggingface"
                    elif "ollama" in root:
                        source = "ollama"
                    
                    # Extract info from filename
                    var quant = self.extract_quantization(filename)
                    var arch = self.extract_architecture(filename)
                    
                    # Create model info
                    var model = ModelInfo(
                        name=filename,
                        path=str(filepath),
                        size_bytes=size_bytes,
                        format="gguf" if filename.endswith(".gguf") else "ggml",
                        source=source,
                        quantization=quant,
                        architecture=arch
                    )
                    
                    self.models.append(model)
                    found += 1
                    
                    print(f"    ✓ {filename} ({model.size_gb():.2f} GB, {quant})")
        
        return found
    
    fn extract_quantization(self, filename: String) -> String:
        """Extract quantization from filename."""
        # Common patterns: Q4_0, Q4_K_M, Q8_0, F16, etc.
        if "Q4_0" in filename or "q4_0" in filename:
            return "Q4_0"
        elif "Q4_K_M" in filename or "q4_k_m" in filename:
            return "Q4_K_M"
        elif "Q4_K_S" in filename or "q4_k_s" in filename:
            return "Q4_K_S"
        elif "Q8_0" in filename or "q8_0" in filename:
            return "Q8_0"
        elif "F16" in filename or "f16" in filename:
            return "F16"
        elif "F32" in filename or "f32" in filename:
            return "F32"
        else:
            return "unknown"
    
    fn extract_architecture(self, filename: String) -> String:
        """Extract architecture from filename."""
        var lower = filename.lower()
        
        if "llama" in lower:
            return "llama"
        elif "phi" in lower:
            return "phi"
        elif "mistral" in lower:
            return "mistral"
        elif "qwen" in lower:
            return "qwen"
        elif "gemma" in lower:
            return "gemma"
        else:
            return "unknown"
    
    fn list_models(self) -> List[ModelInfo]:
        """Get list of discovered models."""
        return self.models
    
    fn get_model_by_name(self, name: String) raises -> ModelInfo:
        """Get model by name."""
        for i in range(len(self.models)):
            if self.models[i].name == name:
                return self.models[i]
        
        raise Error("Model not found: " + name)
    
    fn print_summary(self):
        """Print summary of discovered models."""
        print()
        print("=" * 80)
        print("📊 Model Discovery Summary")
        print("=" * 80)
        print()
        
        if len(self.models) == 0:
            print("📭 No models found")
            print()
            print("💡 To add models:")
            print("   • Download GGUF files to ./models/")
            print("   • Install via Ollama: ollama pull llama2")
            print("   • Download from HuggingFace")
            print()
            return
        
        # Group by source
        var hf_count = 0
        var ollama_count = 0
        var local_count = 0
        var total_size: Float32 = 0.0
        
        for i in range(len(self.models)):
            var model = self.models[i]
            total_size += model.size_gb()
            
            if model.source == "huggingface":
                hf_count += 1
            elif model.source == "ollama":
                ollama_count += 1
            else:
                local_count += 1
        
        print(f"Total Models: {len(self.models)}")
        print(f"Total Size:   {total_size:.2f} GB")
        print()
        print("By Source:")
        print(f"  HuggingFace: {hf_count}")
        print(f"  Ollama:      {ollama_count}")
        print(f"  Local:       {local_count}")
        print()
        
        # List models
        print("Models:")
        print()
        for i in range(len(self.models)):
            var model = self.models[i]
            print(f"{i+1}. {model.name}")
            print(f"   Path:   {model.path}")
            print(f"   Size:   {model.size_gb():.2f} GB")
            print(f"   Quant:  {model.quantization}")
            print(f"   Arch:   {model.architecture}")
            print(f"   Source: {model.source}")
            print()
        
        print("=" * 80)

# ============================================================================
# Model Registry
# ============================================================================

struct ModelRegistry:
    """
    Central registry of available models.
    Maps model names to file paths and metadata.
    """
    var models: Dict[String, ModelInfo]
    var default_model: String
    
    fn __init__(inout self):
        self.models = Dict[String, ModelInfo]()
        self.default_model = ""
    
    fn register(inout self, model: ModelInfo):
        """Register a model."""
        self.models[model.name] = model
        
        # Set as default if first model
        if self.default_model == "":
            self.default_model = model.name
    
    fn get(self, name: String) raises -> ModelInfo:
        """Get model by name."""
        if name in self.models:
            return self.models[name]
        raise Error("Model not found: " + name)
    
    fn list_names(self) -> List[String]:
        """List all registered model names."""
        var names = List[String]()
        for key in self.models.keys():
            names.append(key)
        return names
    
    fn set_default(inout self, name: String) raises:
        """Set default model."""
        if name not in self.models:
            raise Error("Model not found: " + name)
        self.default_model = name
    
    fn get_default(self) -> String:
        """Get default model name."""
        return self.default_model

# ============================================================================
# Testing
# ============================================================================

fn main() raises:
    print("=" * 80)
    print("🔍 Model Discovery System")
    print("=" * 80)
    print()
    
    # Create scanner
    var scanner = ModelScanner()
    
    # Scan for models
    var count = scanner.scan()
    
    # Print summary
    scanner.print_summary()
    
    # Create registry
    if count > 0:
        print("=" * 80)
        print("📋 Building Model Registry")
        print("=" * 80)
        print()
        
        var registry = ModelRegistry()
        var models = scanner.list_models()
        
        for i in range(len(models)):
            registry.register(models[i])
            print(f"  ✅ Registered: {models[i].name}")
        
        print()
        print(f"✅ Registry complete: {count} models")
        print(f"🎯 Default model: {registry.get_default()}")
        print()
    
    print("=" * 80)
    print("✅ Model discovery complete!")
    print("=" * 80)
    print()
    
    if count == 0:
        print("💡 Next steps:")
        print("   1. Download GGUF models")
        print("   2. Place in ./models/ directory")
        print("   3. Run discovery again")
        print()
