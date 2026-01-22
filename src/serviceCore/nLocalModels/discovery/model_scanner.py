#!/usr/bin/env python3
"""
Model Scanner - Auto-discover GGUF models from various sources
Scans HuggingFace cache, Ollama models, and custom directories
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# Common model directories
HUGGINGFACE_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
OLLAMA_MODELS = Path.home() / ".ollama" / "models"
LOCAL_MODELS = Path("./models")

# GGUF file extensions
GGUF_EXTENSIONS = [".gguf", ".ggml"]

# ============================================================================
# Model Metadata
# ============================================================================

@dataclass
class ModelMetadata:
    """Metadata for discovered model"""
    name: str
    path: str
    size_bytes: int
    size_mb: float
    format: str
    source: str  # "huggingface", "ollama", "local"
    model_id: Optional[str]  # e.g., "TheBloke/Phi-3-mini-128k-instruct-GGUF"
    filename: str
    modified: str
    quantization: Optional[str]  # e.g., "Q4_K_M", "Q8_0"
    parameter_count: Optional[str]  # e.g., "3.8B", "7B"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=2)

# ============================================================================
# Model Scanner
# ============================================================================

class ModelScanner:
    """
    Scans for GGUF models in various locations
    """
    
    def __init__(self):
        self.models: List[ModelMetadata] = []
    
    def scan_all(self) -> List[ModelMetadata]:
        """Scan all known locations for models"""
        print("🔍 Scanning for GGUF models...")
        print()
        
        # Clear existing
        self.models = []
        
        # Scan each location
        self._scan_huggingface()
        self._scan_ollama()
        self._scan_local()
        
        # Sort by size
        self.models.sort(key=lambda m: m.size_bytes, reverse=True)
        
        print()
        print(f"✅ Found {len(self.models)} model(s)")
        
        return self.models
    
    def _scan_huggingface(self):
        """Scan HuggingFace cache"""
        if not HUGGINGFACE_CACHE.exists():
            print(f"⏭  Skipping HuggingFace cache (not found)")
            return
        
        print(f"📦 Scanning HuggingFace cache: {HUGGINGFACE_CACHE}")
        
        found = 0
        for root, dirs, files in os.walk(HUGGINGFACE_CACHE):
            for file in files:
                if any(file.endswith(ext) for ext in GGUF_EXTENSIONS):
                    file_path = Path(root) / file
                    model = self._parse_model(file_path, "huggingface")
                    if model:
                        self.models.append(model)
                        found += 1
        
        print(f"   Found {found} model(s)")
    
    def _scan_ollama(self):
        """Scan Ollama models"""
        if not OLLAMA_MODELS.exists():
            print(f"⏭  Skipping Ollama models (not found)")
            return
        
        print(f"🦙 Scanning Ollama models: {OLLAMA_MODELS}")
        
        found = 0
        for root, dirs, files in os.walk(OLLAMA_MODELS):
            for file in files:
                if any(file.endswith(ext) for ext in GGUF_EXTENSIONS):
                    file_path = Path(root) / file
                    model = self._parse_model(file_path, "ollama")
                    if model:
                        self.models.append(model)
                        found += 1
        
        print(f"   Found {found} model(s)")
    
    def _scan_local(self):
        """Scan local models directory"""
        if not LOCAL_MODELS.exists():
            print(f"⏭  Skipping local models (directory not found)")
            return
        
        print(f"📁 Scanning local models: {LOCAL_MODELS}")
        
        found = 0
        for file in LOCAL_MODELS.glob("**/*"):
            if file.is_file() and any(str(file).endswith(ext) for ext in GGUF_EXTENSIONS):
                model = self._parse_model(file, "local")
                if model:
                    self.models.append(model)
                    found += 1
        
        print(f"   Found {found} model(s)")
    
    def _parse_model(self, path: Path, source: str) -> Optional[ModelMetadata]:
        """Parse model file into metadata"""
        try:
            stat = path.stat()
            size_bytes = stat.st_size
            size_mb = size_bytes / (1024 * 1024)
            
            # Extract model info from filename
            filename = path.name
            name = path.stem
            
            # Try to extract model_id from path (for HuggingFace)
            model_id = None
            if source == "huggingface":
                # Path like: .../models--TheBloke--Phi-3-mini.../snapshots/.../file.gguf
                parts = str(path).split("models--")
                if len(parts) > 1:
                    id_part = parts[1].split("/")[0]
                    model_id = id_part.replace("--", "/")
            
            # Extract quantization (e.g., Q4_K_M, Q8_0)
            quantization = self._extract_quantization(filename)
            
            # Extract parameter count (e.g., 3B, 7B)
            param_count = self._extract_param_count(filename)
            
            # Modified time
            modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            return ModelMetadata(
                name=name,
                path=str(path),
                size_bytes=size_bytes,
                size_mb=round(size_mb, 2),
                format="gguf",
                source=source,
                model_id=model_id,
                filename=filename,
                modified=modified,
                quantization=quantization,
                parameter_count=param_count
            )
        
        except Exception as e:
            print(f"   ⚠️  Error parsing {path}: {e}")
            return None
    
    def _extract_quantization(self, filename: str) -> Optional[str]:
        """Extract quantization format from filename"""
        # Common patterns: Q4_K_M, Q8_0, Q4_0, Q5_K_S, etc.
        import re
        patterns = [
            r'[Qq](\d+)_([Kk]_)?[MSL]',  # Q4_K_M, Q5_K_S
            r'[Qq](\d+)_(\d+)',           # Q4_0, Q8_0
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(0).upper()
        
        return None
    
    def _extract_param_count(self, filename: str) -> Optional[str]:
        """Extract parameter count from filename"""
        import re
        # Patterns like: 3B, 7B, 13B, 1.3B, 3.8B
        match = re.search(r'(\d+\.?\d*)[Bb]', filename)
        if match:
            return match.group(1) + "B"
        return None
    
    def get_by_name(self, name: str) -> Optional[ModelMetadata]:
        """Get model by name"""
        for model in self.models:
            if model.name == name:
                return model
        return None
    
    def get_by_source(self, source: str) -> List[ModelMetadata]:
        """Get models from specific source"""
        return [m for m in self.models if m.source == source]
    
    def print_summary(self):
        """Print summary of discovered models"""
        if not self.models:
            print("No models found")
            return
        
        print()
        print("=" * 80)
        print("📊 Model Discovery Summary")
        print("=" * 80)
        print()
        
        # Group by source
        by_source = {}
        for model in self.models:
            if model.source not in by_source:
                by_source[model.source] = []
            by_source[model.source].append(model)
        
        # Print by source
        for source, models in by_source.items():
            print(f"📁 {source.upper()}: {len(models)} model(s)")
            for model in models:
                quant = f" ({model.quantization})" if model.quantization else ""
                params = f" [{model.parameter_count}]" if model.parameter_count else ""
                print(f"   • {model.name}{params}{quant} - {model.size_mb:.1f} MB")
            print()
        
        # Total stats
        total_size_gb = sum(m.size_bytes for m in self.models) / (1024 * 1024 * 1024)
        print(f"📊 Total: {len(self.models)} model(s), {total_size_gb:.2f} GB")
        print()

# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Registry for managing discovered models
    """
    
    def __init__(self, cache_file: str = ".model_registry.json"):
        self.cache_file = Path(cache_file)
        self.models: Dict[str, ModelMetadata] = {}
        self.scanner = ModelScanner()
    
    def load_cache(self) -> bool:
        """Load cached registry"""
        if not self.cache_file.exists():
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            self.models = {
                name: ModelMetadata(**meta)
                for name, meta in data.items()
            }
            return True
        
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            return False
    
    def save_cache(self):
        """Save registry to cache"""
        try:
            data = {
                name: model.to_dict()
                for name, model in self.models.items()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Registry cached to {self.cache_file}")
        
        except Exception as e:
            print(f"⚠️  Error saving cache: {e}")
    
    def refresh(self) -> List[ModelMetadata]:
        """Refresh registry by scanning"""
