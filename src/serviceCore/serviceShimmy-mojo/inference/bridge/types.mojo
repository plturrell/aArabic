"""
Shared types for the inference bridge.

These types are used across all services (LLM, Embedding, Translation, RAG)
to ensure consistent interfaces with the native inference engine.
"""


struct ModelConfig:
    """Configuration for a loaded model."""

    var model_type: String
    var vocab_size: Int
    var hidden_size: Int
    var num_layers: Int
    var num_attention_heads: Int
    var num_kv_heads: Int
    var max_position_embeddings: Int

    fn __init__(out self):
        self.model_type = ""
        self.vocab_size = 0
        self.hidden_size = 0
        self.num_layers = 0
        self.num_attention_heads = 0
        self.num_kv_heads = 0
        self.max_position_embeddings = 0

    fn __init__(
        out self,
        model_type: String,
        vocab_size: Int,
        hidden_size: Int,
        num_layers: Int,
        num_attention_heads: Int,
        num_kv_heads: Int,
        max_position_embeddings: Int,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings


struct GenerationConfig:
    """Configuration for text generation."""

    var max_tokens: Int
    var temperature: Float32
    var top_k: Int
    var top_p: Float32
    var repetition_penalty: Float32

    fn __init__(out self):
        self.max_tokens = 100
        self.temperature = 0.7
        self.top_k = 40
        self.top_p = 0.9
        self.repetition_penalty = 1.0

    fn __init__(
        out self,
        max_tokens: Int = 100,
        temperature: Float32 = 0.7,
        top_k: Int = 40,
        top_p: Float32 = 0.9,
        repetition_penalty: Float32 = 1.0,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty


struct InferenceResult:
    """Result from an inference call."""

    var text: String
    var tokens_generated: Int
    var generation_time_ms: Float64
    var success: Bool
    var error_message: String

    fn __init__(out self):
        self.text = ""
        self.tokens_generated = 0
        self.generation_time_ms = 0.0
        self.success = False
        self.error_message = ""

    fn __init__(out self, text: String, tokens: Int, time_ms: Float64):
        self.text = text
        self.tokens_generated = tokens
        self.generation_time_ms = time_ms
        self.success = True
        self.error_message = ""

    @staticmethod
    fn error(message: String) -> InferenceResult:
        var result = InferenceResult()
        result.success = False
        result.error_message = message
        return result


struct EmbeddingResult:
    """Result from an embedding call."""

    var embedding: List[Float32]
    var dimensions: Int
    var computation_time_ms: Float64
    var success: Bool
    var error_message: String

    fn __init__(out self):
        self.embedding = List[Float32]()
        self.dimensions = 0
        self.computation_time_ms = 0.0
        self.success = False
        self.error_message = ""

    fn __init__(out self, embedding: List[Float32], time_ms: Float64):
        self.embedding = embedding
        self.dimensions = len(embedding)
        self.computation_time_ms = time_ms
        self.success = True
        self.error_message = ""


# Model type aliases for clarity
alias LLMModel = String
alias EmbeddingModel = String
alias TranslationModel = String
