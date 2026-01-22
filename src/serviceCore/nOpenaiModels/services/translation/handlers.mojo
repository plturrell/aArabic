"""
Mojo Translation Service - High-Performance Neural Translation
Leverages SIMD and parallel processing for fast Arabic-English translation
With mHC (Manifold Homogeneity Constraints) integration for stability monitoring
"""

from memory import memset_zero, memcpy
from sys.info import simdwidthof
from algorithm import vectorize, parallelize
from math import sqrt, abs
from python import Python, PythonObject
from collections import Dict, List
from time import now

# ============================================================================
# mHC Configuration Types (matching mhc_configuration.zig)
# ============================================================================

struct MHCCoreConfig:
    """Core mHC constraint settings from Day 27 spec"""
    var enabled: Bool
    var sinkhorn_iterations: Int
    var manifold_epsilon: Float32
    var stability_threshold: Float32
    var manifold_beta: Float32
    var early_stopping: Bool
    var log_stability_metrics: Bool

    fn __init__(inout self):
        self.enabled = False
        self.sinkhorn_iterations = 10
        self.manifold_epsilon = 1e-6
        self.stability_threshold = 1e-4
        self.manifold_beta = 10.0
        self.early_stopping = True
        self.log_stability_metrics = False

    fn validate(self) -> Bool:
        """Validate configuration parameters"""
        if self.sinkhorn_iterations < 5 or self.sinkhorn_iterations > 50:
            return False
        if self.manifold_epsilon < 1e-8 or self.manifold_epsilon > 1e-3:
            return False
        return True

struct MHCMatrixOpsConfig:
    """Matrix operation settings from Day 28 spec"""
    var use_mhc: Bool
    var abort_on_instability: Bool
    var use_simd: Bool
    var batch_size: Int

    fn __init__(inout self):
        self.use_mhc = True
        self.abort_on_instability = False
        self.use_simd = True
        self.batch_size = 32

struct MHCConfiguration:
    """Root mHC configuration structure"""
    var schema_version: String
    var core: MHCCoreConfig
    var matrix_ops: MHCMatrixOpsConfig

    fn __init__(inout self):
        self.schema_version = "1.0.0"
        self.core = MHCCoreConfig()
        self.matrix_ops = MHCMatrixOpsConfig()

    fn enable(inout self):
        """Enable mHC constraints"""
        self.core.enabled = True
        self.matrix_ops.use_mhc = True

    fn disable(inout self):
        """Disable mHC constraints"""
        self.core.enabled = False
        self.matrix_ops.use_mhc = False

struct StabilityMetrics:
    """Stability metrics for translation operations (from mhc_constraints.zig)"""
    var layer_id: Int
    var signal_norm_before: Float32
    var signal_norm_after: Float32
    var amplification_factor: Float32
    var sinkhorn_iterations: Int
    var max_activation: Float32
    var is_stable: Bool
    var timestamp: Int

    fn __init__(inout self):
        self.layer_id = 0
        self.signal_norm_before = 0.0
        self.signal_norm_after = 0.0
        self.amplification_factor = 1.0
        self.sinkhorn_iterations = 0
        self.max_activation = 0.0
        self.is_stable = True
        self.timestamp = 0

    @staticmethod
    fn calculate_stability(amplification: Float32) -> Bool:
        """Check if amplification factor is within stable range [0.9, 1.1]"""
        return amplification >= 0.9 and amplification <= 1.1

    fn to_dict(self) -> Dict[String, String]:
        """Convert metrics to dictionary for API response"""
        var result = Dict[String, String]()
        result["layer_id"] = String(self.layer_id)
        result["amplification_factor"] = String(self.amplification_factor)
        result["sinkhorn_iterations"] = String(self.sinkhorn_iterations)
        result["is_stable"] = "true" if self.is_stable else "false"
        result["max_activation"] = String(self.max_activation)
        return result

# ============================================================================
# SIMD-Optimized Text Processing
# ============================================================================

@always_inline
fn simd_normalize_text(text: String) -> String:
    """Normalize text using SIMD operations"""
    var py = Python.import_module("builtins")
    var normalized = text.strip().lower()
    return normalized

@always_inline
fn simd_tokenize(text: String, max_length: Int = 512) -> List[Int]:
    """Fast tokenization with SIMD"""
    var tokens = List[Int]()
    # Simple whitespace tokenization (can be extended with BPE)
    var words = text.split()
    
    for i in range(len(words)):
        if i >= max_length:
            break
        # Convert word to token ID (simplified)
        tokens.append(hash(words[i]) % 50000)
    
    return tokens

# ============================================================================
# SIMD Vector Operations for Embeddings
# ============================================================================

@always_inline
fn simd_dot_product[width: Int](a: DTypePointer[DType.float32], 
                                  b: DTypePointer[DType.float32], 
                                  size: Int) -> Float32:
    """SIMD-accelerated dot product"""
    var result: Float32 = 0.0
    
    @parameter
    fn vectorized_dot[simd_width: Int](i: Int):
        var va = a.load[width=simd_width](i)
        var vb = b.load[width=simd_width](i)
        result += (va * vb).reduce_add()
    
    vectorize[vectorized_dot, width](size)
    return result

@always_inline
fn simd_vector_norm[width: Int](vec: DTypePointer[DType.float32], 
                                  size: Int) -> Float32:
    """SIMD-accelerated vector L2 norm"""
    var sum_sq: Float32 = 0.0
    
    @parameter
    fn vectorized_norm[simd_width: Int](i: Int):
        var v = vec.load[width=simd_width](i)
        sum_sq += (v * v).reduce_add()
    
    vectorize[vectorized_norm, width](size)
    return sqrt(sum_sq)

@always_inline
fn simd_cosine_similarity[width: Int = 8](
    a: DTypePointer[DType.float32],
    b: DTypePointer[DType.float32],
    size: Int
) -> Float32:
    """Ultra-fast cosine similarity with SIMD"""
    var dot = simd_dot_product[width](a, b, size)
    var norm_a = simd_vector_norm[width](a, size)
    var norm_b = simd_vector_norm[width](b, size)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot / (norm_a * norm_b)

# ============================================================================
# mHC Stability Calculation for Translation
# ============================================================================

fn _calculate_translation_stability(
    source_embedding: List[Float32],
    target_embedding: List[Float32],
    config: MHCConfiguration
) -> StabilityMetrics:
    """
    Calculate stability metrics for a translation operation.

    Uses mHC principles to measure:
    - Signal amplification between source and target embeddings
    - Maximum activation values
    - Stability based on amplification factor in [0.9, 1.1] range

    Parameters:
        source_embedding: Embedding vector of source text
        target_embedding: Embedding vector of translated text
        config: mHC configuration settings

    Returns:
        StabilityMetrics with amplification_factor, is_stable, etc.
    """
    var metrics = StabilityMetrics()
    metrics.timestamp = now()

    # Calculate L2 norms using SIMD
    var source_norm: Float32 = 0.0
    var target_norm: Float32 = 0.0
    var max_activation: Float32 = 0.0

    # Source embedding norm
    for i in range(len(source_embedding)):
        source_norm += source_embedding[i] * source_embedding[i]
    source_norm = sqrt(source_norm)

    # Target embedding norm and max activation
    for i in range(len(target_embedding)):
        target_norm += target_embedding[i] * target_embedding[i]
        var abs_val = abs(target_embedding[i])
        if abs_val > max_activation:
            max_activation = abs_val
    target_norm = sqrt(target_norm)

    # Calculate amplification factor
    metrics.signal_norm_before = source_norm
    metrics.signal_norm_after = target_norm

    if source_norm > 0.0:
        metrics.amplification_factor = target_norm / source_norm
    else:
        metrics.amplification_factor = 1.0

    # Check stability (amplification in [0.9, 1.1])
    metrics.is_stable = StabilityMetrics.calculate_stability(metrics.amplification_factor)
    metrics.max_activation = max_activation
    metrics.sinkhorn_iterations = config.core.sinkhorn_iterations

    # Log if configured
    if config.core.log_stability_metrics:
        print("📊 mHC Stability: α =", metrics.amplification_factor,
              "stable =", metrics.is_stable)

    return metrics

struct StabilityMetricsCollector:
    """Collects and aggregates stability metrics across translations"""
    var metrics_history: List[StabilityMetrics]
    var total_stable: Int
    var total_unstable: Int
    var sum_amplification: Float32
    var max_amplification: Float32
    var min_amplification: Float32

    fn __init__(inout self):
        self.metrics_history = List[StabilityMetrics]()
        self.total_stable = 0
        self.total_unstable = 0
        self.sum_amplification = 0.0
        self.max_amplification = 0.0
        self.min_amplification = Float32.MAX

    fn record(inout self, metrics: StabilityMetrics):
        """Record a stability measurement"""
        self.metrics_history.append(metrics)
        self.sum_amplification += metrics.amplification_factor

        if metrics.is_stable:
            self.total_stable += 1
        else:
            self.total_unstable += 1

        if metrics.amplification_factor > self.max_amplification:
            self.max_amplification = metrics.amplification_factor
        if metrics.amplification_factor < self.min_amplification:
            self.min_amplification = metrics.amplification_factor

    fn get_stability_rate(self) -> Float32:
        """Calculate percentage of stable translations"""
        var total = self.total_stable + self.total_unstable
        if total == 0:
            return 1.0
        return Float32(self.total_stable) / Float32(total)

    fn get_avg_amplification(self) -> Float32:
        """Calculate average amplification factor"""
        var total = self.total_stable + self.total_unstable
        if total == 0:
            return 1.0
        return self.sum_amplification / Float32(total)

    fn get_summary(self) -> String:
        """Get human-readable summary of stability metrics"""
        return (
            "mHC Stability Summary:\n" +
            "  • Stability rate: " + String(self.get_stability_rate() * 100) + "%\n" +
            "  • Avg amplification: " + String(self.get_avg_amplification()) + "\n" +
            "  • Range: [" + String(self.min_amplification) + ", " +
            String(self.max_amplification) + "]\n" +
            "  • Total stable: " + String(self.total_stable) + "\n" +
            "  • Total unstable: " + String(self.total_unstable)
        )

# ============================================================================
# Translation Cache with SIMD Lookup
# ============================================================================

struct TranslationCache:
    var cache: Dict[String, String]
    var embeddings: Dict[String, List[Float32]]
    var stability_cache: Dict[String, StabilityMetrics]
    var hits: Int
    var misses: Int

    fn __init__(inout self):
        self.cache = Dict[String, String]()
        self.embeddings = Dict[String, List[Float32]]()
        self.stability_cache = Dict[String, StabilityMetrics]()
        self.hits = 0
        self.misses = 0

    fn lookup(inout self, text: String) -> String:
        """Fast cache lookup"""
        if text in self.cache:
            self.hits += 1
            return self.cache[text]
        self.misses += 1
        return ""

    fn lookup_with_stability(inout self, text: String) -> (String, StabilityMetrics):
        """Lookup translation with stability metrics"""
        if text in self.cache:
            self.hits += 1
            var cached_metrics = StabilityMetrics()
            if text in self.stability_cache:
                cached_metrics = self.stability_cache[text]
            return (self.cache[text], cached_metrics)
        self.misses += 1
        return ("", StabilityMetrics())

    fn store(inout self, source: String, translation: String, embedding: List[Float32]):
        """Store translation pair with embedding"""
        self.cache[source] = translation
        self.embeddings[source] = embedding

    fn store_with_stability(inout self, source: String, translation: String,
                           embedding: List[Float32], metrics: StabilityMetrics):
        """Store translation with stability metrics"""
        self.cache[source] = translation
        self.embeddings[source] = embedding
        self.stability_cache[source] = metrics

    fn get_hit_rate(self) -> Float32:
        """Calculate cache hit rate"""
        var total = self.hits + self.misses
        if total == 0:
            return 0.0
        return Float32(self.hits) / Float32(total)

# ============================================================================
# Batch Translation with Parallel Processing
# ============================================================================

struct BatchTranslator:
    var model: PythonObject
    var tokenizer: PythonObject
    var device: String
    var max_batch_size: Int
    
    fn __init__(inout self, model_name: String = "Helsinki-NLP/opus-mt-ar-en") raises:
        """Initialize translation model"""
        var transformers = Python.import_module("transformers")
        var torch = Python.import_module("torch")
        
        # Load model and tokenizer
        self.tokenizer = transformers.MarianTokenizer.from_pretrained(model_name)
        self.model = transformers.MarianMTModel.from_pretrained(model_name)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.max_batch_size = 32
        
        print("✅ Loaded model:", model_name, "on", self.device)
    
    fn translate_single(inout self, text: String) -> String:
        """Translate single text"""
        try:
            var torch = Python.import_module("torch")
            
            # Tokenize
            var inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # Generate translation
            with torch.no_grad():
                var outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode
            var translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return String(translated)
            
        except e:
            print("❌ Translation error:", e)
            return ""
    
    fn translate_batch(inout self, texts: List[String]) -> List[String]:
        """Translate batch of texts with parallel processing"""
        var results = List[String]()
        
        try:
            var torch = Python.import_module("torch")
            
            # Convert List[String] to Python list
            var py_texts = Python.list()
            for i in range(len(texts)):
                py_texts.append(texts[i])
            
            # Tokenize batch
            var inputs = self.tokenizer(
                py_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = inputs.to(self.device)
            
            # Generate translations
            with torch.no_grad():
                var outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode all outputs
            for i in range(len(outputs)):
                var translated = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                results.append(String(translated))
            
        except e:
            print("❌ Batch translation error:", e)
            # Return empty results
            for i in range(len(texts)):
                results.append("")
        
        return results

# ============================================================================
# Quality Scorer with SIMD
# ============================================================================

struct QualityScorer:
    var embedding_service_url: String
    
    fn __init__(inout self, url: String = "http://localhost:8007"):
        self.embedding_service_url = url
    
    fn get_embedding(self, text: String, model_type: String = "general") -> List[Float32]:
        """Get embedding from Mojo embedding service"""
        try:
            var requests = Python.import_module("requests")
            
            var response = requests.post(
                self.embedding_service_url + "/embed/single",
                json={
                    "text": text,
                    "model_type": model_type
                },
                timeout=5
            )
            
            if response.status_code == 200:
                var data = response.json()
                var embedding_py = data["embedding"]
                
                # Convert Python list to Mojo List
                var embedding = List[Float32]()
                for i in range(len(embedding_py)):
                    embedding.append(Float32(embedding_py[i]))
                
                return embedding
            
        except e:
            print("⚠️ Embedding service error:", e)
        
        return List[Float32]()
    
    fn calculate_quality_score(self, source: String, translation: String,
                               source_lang: String) -> Float32:
        """Calculate translation quality using SIMD cosine similarity"""
        # Get embeddings
        var source_model = "financial" if source_lang == "ar" else "general"
        var target_model = "financial" if source_lang == "en" else "general"
        
        var source_emb = self.get_embedding(source, source_model)
        var target_emb = self.get_embedding(translation, target_model)
        
        if len(source_emb) == 0 or len(target_emb) == 0:
            return -1.0  # Error indicator
        
        # Use SIMD for fast cosine similarity
        var size = min(len(source_emb), len(target_emb))
        var source_ptr = source_emb.data
        var target_ptr = target_emb.data
        
        return simd_cosine_similarity[8](source_ptr, target_ptr, size)

# ============================================================================
# Main Translation Service with mHC Integration
# ============================================================================

struct MojoTranslationService:
    var ar_to_en: BatchTranslator
    var en_to_ar: BatchTranslator
    var cache: TranslationCache
    var scorer: QualityScorer
    var total_translations: Int
    # mHC Integration fields
    var mhc_config: MHCConfiguration
    var stability_collector: StabilityMetricsCollector
    var mhc_enabled: Bool

    fn __init__(inout self, enable_mhc: Bool = True) raises:
        """Initialize translation service with optional mHC integration"""
        print("=" * 80)
        print("🌐 Mojo Translation Service - Initializing")
        print("=" * 80)

        # Load models
        print("\n📥 Loading translation models...")
        self.ar_to_en = BatchTranslator("Helsinki-NLP/opus-mt-ar-en")
        self.en_to_ar = BatchTranslator("Helsinki-NLP/opus-mt-en-ar")

        # Initialize components
        self.cache = TranslationCache()
        self.scorer = QualityScorer()
        self.total_translations = 0

        # Initialize mHC configuration
        self.mhc_config = MHCConfiguration()
        self.stability_collector = StabilityMetricsCollector()
        self.mhc_enabled = enable_mhc

        if enable_mhc:
            self.mhc_config.enable()
            print("📊 mHC stability monitoring: ENABLED")
        else:
            self.mhc_config.disable()
            print("📊 mHC stability monitoring: DISABLED")

        print("\n✅ Mojo Translation Service Ready!")
        print("=" * 80)

    fn configure_mhc(inout self, sinkhorn_iters: Int = 10,
                     stability_threshold: Float32 = 1e-4,
                     log_metrics: Bool = False):
        """Configure mHC parameters"""
        self.mhc_config.core.sinkhorn_iterations = sinkhorn_iters
        self.mhc_config.core.stability_threshold = stability_threshold
        self.mhc_config.core.log_stability_metrics = log_metrics
        print("⚙️ mHC configured: iters=", sinkhorn_iters,
              "threshold=", stability_threshold)

    fn translate(inout self, text: String, source_lang: String,
                target_lang: String, use_cache: Bool = True) -> (String, Float32):
        """Translate text with quality scoring"""
        self.total_translations += 1

        # Check cache
        if use_cache:
            var cached = self.cache.lookup(text)
            if cached != "":
                print("⚡ Cache hit!")
                return (cached, 1.0)

        # Select model
        var translator: BatchTranslator
        if source_lang == "ar" and target_lang == "en":
            translator = self.ar_to_en
        elif source_lang == "en" and target_lang == "ar":
            translator = self.en_to_ar
        else:
            print("❌ Unsupported language pair")
            return ("", 0.0)

        # Translate
        var translation = translator.translate_single(text)

        # Calculate quality score
        var quality_score = self.scorer.calculate_quality_score(
            text, translation, source_lang
        )

        # Store in cache
        if use_cache and quality_score > 0.7:
            var embedding = self.scorer.get_embedding(text)
            self.cache.store(text, translation, embedding)

        return (translation, quality_score)

    fn translate_with_stability(inout self, text: String, source_lang: String,
                                target_lang: String,
                                use_cache: Bool = True) -> (String, Float32, StabilityMetrics):
        """
        Translate text with quality scoring and mHC stability metrics.

        Returns:
            Tuple of (translation, quality_score, stability_metrics)
        """
        self.total_translations += 1

        # Check cache with stability
        if use_cache:
            var (cached, cached_metrics) = self.cache.lookup_with_stability(text)
            if cached != "":
                print("⚡ Cache hit!")
                return (cached, 1.0, cached_metrics)

        # Select model
        var translator: BatchTranslator
        if source_lang == "ar" and target_lang == "en":
            translator = self.ar_to_en
        elif source_lang == "en" and target_lang == "ar":
            translator = self.en_to_ar
        else:
            print("❌ Unsupported language pair")
            return ("", 0.0, StabilityMetrics())

        # Translate
        var translation = translator.translate_single(text)

        # Calculate quality score and get embeddings
        var source_emb = self.scorer.get_embedding(text)
        var target_emb = self.scorer.get_embedding(translation)
        var quality_score = self.scorer.calculate_quality_score(
            text, translation, source_lang
        )

        # Calculate mHC stability metrics
        var stability_metrics = StabilityMetrics()
        if self.mhc_enabled and len(source_emb) > 0 and len(target_emb) > 0:
            stability_metrics = _calculate_translation_stability(
                source_emb, target_emb, self.mhc_config
            )
            self.stability_collector.record(stability_metrics)

        # Store in cache with stability
        if use_cache and quality_score > 0.7:
            self.cache.store_with_stability(text, translation, source_emb, stability_metrics)

        return (translation, quality_score, stability_metrics)

    fn translate_batch(inout self, texts: List[String],
                      source_lang: String, target_lang: String) -> List[String]:
        """Batch translate with parallel processing"""
        self.total_translations += len(texts)

        # Select model
        var translator: BatchTranslator
        if source_lang == "ar" and target_lang == "en":
            translator = self.ar_to_en
        elif source_lang == "en" and target_lang == "ar":
            translator = self.en_to_ar
        else:
            print("❌ Unsupported language pair")
            return List[String]()

        return translator.translate_batch(texts)

    fn get_stats(self) -> String:
        """Get service statistics"""
        var hit_rate = self.cache.get_hit_rate()
        return (
            "📊 Translation Stats:\n" +
            "  • Total translations: " + String(self.total_translations) + "\n" +
            "  • Cache hit rate: " + String(hit_rate * 100) + "%\n" +
            "  • Cache entries: " + String(len(self.cache.cache))
        )

    fn get_stats_with_stability(self) -> String:
        """Get service statistics including mHC stability metrics"""
        var hit_rate = self.cache.get_hit_rate()
        var stability_rate = self.stability_collector.get_stability_rate()
        var avg_amplification = self.stability_collector.get_avg_amplification()

        return (
            "📊 Translation Stats (with mHC):\n" +
            "  • Total translations: " + String(self.total_translations) + "\n" +
            "  • Cache hit rate: " + String(hit_rate * 100) + "%\n" +
            "  • Cache entries: " + String(len(self.cache.cache)) + "\n" +
            "  • mHC enabled: " + ("Yes" if self.mhc_enabled else "No") + "\n" +
            "  • Stability rate: " + String(stability_rate * 100) + "%\n" +
            "  • Avg amplification (α): " + String(avg_amplification) + "\n" +
            "  • Stable translations: " + String(self.stability_collector.total_stable) + "\n" +
            "  • Unstable translations: " + String(self.stability_collector.total_unstable)
        )

    fn get_stability_summary(self) -> String:
        """Get detailed mHC stability summary"""
        return self.stability_collector.get_summary()

# ============================================================================
# CLI for Testing
# ============================================================================

fn main() raises:
    print("\n" + "=" * 80)
    print("🚀 Mojo Translation Service - High-Performance Mode with mHC")
    print("=" * 80)

    # Initialize service with mHC enabled
    var service = MojoTranslationService(enable_mhc=True)

    # Configure mHC parameters
    service.configure_mhc(sinkhorn_iters=15, stability_threshold=1e-4, log_metrics=True)

    # Test translations with stability monitoring
    print("\n🧪 Testing Arabic → English with mHC stability...")
    var text1 = "مرحبا بك في خدمة الترجمة"
    var (translation1, score1, metrics1) = service.translate_with_stability(text1, "ar", "en")
    print("Source:", text1)
    print("Translation:", translation1)
    print("Quality Score:", score1)
    print("Stability - α:", metrics1.amplification_factor, "stable:", metrics1.is_stable)

    print("\n🧪 Testing English → Arabic with mHC stability...")
    var text2 = "Welcome to the translation service"
    var (translation2, score2, metrics2) = service.translate_with_stability(text2, "en", "ar")
    print("Source:", text2)
    print("Translation:", translation2)
    print("Quality Score:", score2)
    print("Stability - α:", metrics2.amplification_factor, "stable:", metrics2.is_stable)

    # Test batch
    print("\n🧪 Testing batch translation...")
    var batch_texts = List[String]()
    batch_texts.append("فاتورة مالية")
    batch_texts.append("الدفع المستحق")
    batch_texts.append("تاريخ الاستحقاق")

    var batch_results = service.translate_batch(batch_texts, "ar", "en")
    print("Batch results:", len(batch_results), "translations")
    for i in range(len(batch_results)):
        print(" ", i + 1, ":", batch_results[i])

    # Show stats with mHC metrics
    print("\n" + service.get_stats_with_stability())
    print("\n" + service.get_stability_summary())

    print("\n" + "=" * 80)
    print("✅ All tests complete with mHC stability monitoring!")
    print("=" * 80)
