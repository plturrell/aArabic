#!/usr/bin/env python3
"""
Mojo Bridge - Python to Mojo Inference Integration
Enables Python HTTP server to call Pure Mojo inference engine
"""

import subprocess
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

MOJO_BINARY_PATH = "./build/shimmy-mojo"
MODELS_DIR = "./models"

# ============================================================================
# Mojo Inference Bridge
# ============================================================================

class MojoBridge:
    """
    Bridge between Python HTTP server and Mojo inference engine
    Manages communication via subprocess or shared memory
    """
    
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.mojo_available = self._check_mojo_available()
    
    def _check_mojo_available(self) -> bool:
        """Check if Mojo runtime is available"""
        try:
            result = subprocess.run(
                ["mojo", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def load_model(self, model_name: str, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model using Mojo
        
        Args:
            model_name: Model identifier
            model_path: Path to GGUF file
        
        Returns:
            Model metadata
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Resolve model path
        if model_path is None:
            model_path = self.models_dir / f"{model_name}.gguf"
        
        print(f"📦 Loading model via Mojo: {model_name}")
        print(f"   Path: {model_path}")
        
        # Load model metadata using Mojo
        try:
            # Call Mojo to probe model
            result = subprocess.run(
                ["mojo", "run", "main.mojo", "probe", str(model_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                # Parse model info from output
                model_info = {
                    "name": model_name,
                    "path": str(model_path),
                    "loaded": True,
                    "format": "gguf",
                    "engine": "mojo"
                }
                
                self.loaded_models[model_name] = model_info
                print(f"   ✅ Model loaded successfully")
                return model_info
            else:
                print(f"   ❌ Failed to load model: {result.stderr}")
                raise Exception(f"Model loading failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            raise Exception("Model loading timed out")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def generate(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        max_tokens: int = 100,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using Mojo inference
        
        Args:
            model_name: Model to use
            prompt: Input prompt
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Top-P sampling
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
        
        Returns:
            Generated text
        """
        # Ensure model is loaded
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        # Build command
        cmd = [
            "mojo", "run", "main.mojo", "generate",
            model_name,
            prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--top-k", str(top_k),
            "--top-p", str(top_p)
        ]
        
        if stop:
            for stop_seq in stop:
                cmd.extend(["--stop", stop_seq])
        
        try:
            # Call Mojo for generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                # Extract generated text from output
                output = result.stdout.strip()
                return output
            else:
                raise Exception(f"Generation failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            raise Exception("Generation timed out")
        except Exception as e:
            raise Exception(f"Error during generation: {str(e)}")
    
    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        max_tokens: int = 100
    ):
        """
        Generate text with streaming using Mojo
        
        Yields:
            Generated tokens as they are produced
        """
        # Ensure model is loaded
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        # Build command with streaming flag
        cmd = [
            "mojo", "run", "main.mojo", "generate",
            model_name,
            prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--top-k", str(top_k),
            "--top-p", str(top_p),
            "--stream"
        ]
        
        try:
            # Start Mojo process for streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=Path(__file__).parent.parent
            )
            
            # Stream output line by line
            for line in process.stdout:
                if line.strip():
                    yield line.strip()
            
            process.wait(timeout=60)
        
        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception("Streaming generation timed out")
        except Exception as e:
            raise Exception(f"Error during streaming: {str(e)}")
    
    def list_models(self) -> List[str]:
        """List available models"""
        models = []
        
        # Check models directory
        if self.models_dir.exists():
            for file in self.models_dir.glob("*.gguf"):
                models.append(file.stem)
        
        # Add default models
        defaults = ["phi-3-mini", "llama-3.2-1b", "llama-3.2-3b"]
        for model in defaults:
            if model not in models:
                models.append(model)
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model information"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        return {
            "name": model_name,
            "loaded": False,
            "available": model_name in self.list_models()
        }

# ============================================================================
# Singleton Instance
# ============================================================================

# Global bridge instance
_bridge_instance: Optional[MojoBridge] = None

def get_bridge() -> MojoBridge:
    """Get or create the global Mojo bridge"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MojoBridge()
    return _bridge_instance

# ============================================================================
# Testing
# ============================================================================

def test_bridge():
    """Test the Mojo bridge"""
    print("=" * 80)
    print("🌉 Testing Mojo Bridge")
    print("=" * 80)
    print()
    
    bridge = get_bridge()
    
    # Test 1: Check Mojo availability
    print("🧪 Test 1: Mojo Runtime")
    print(f"   Mojo available: {bridge.mojo_available}")
    print()
    
    # Test 2: List models
    print("🧪 Test 2: List Models")
    models = bridge.list_models()
    print(f"   Available models: {len(models)}")
    for model in models:
        print(f"     - {model}")
    print()
    
    # Test 3: Model info
    print("🧪 Test 3: Model Info")
    info = bridge.get_model_info("phi-3-mini")
    print(f"   Model: {info['name']}")
    print(f"   Loaded: {info['loaded']}")
    print(f"   Available: {info['available']}")
    print()
    
    print("=" * 80)
    print("✅ Bridge tests complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_bridge()
