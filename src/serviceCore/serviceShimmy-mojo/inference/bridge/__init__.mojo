"""
Inference Bridge Module

Provides the high-level Mojo interface to the Zig inference engine.

Usage:
    from inference.bridge import InferenceEngine, GenerationConfig, InferenceResult

    var engine = InferenceEngine()
    engine.load_model("/path/to/model")
    var result = engine.generate("Hello, how are you?")
"""

from .inference_api import InferenceEngine, create_inference_engine, quick_generate
from .types import (
    ModelConfig,
    GenerationConfig,
    InferenceResult,
    EmbeddingResult,
)
