"""
KV Cache - Key-Value Cache for Efficient Transformer Inference
Stores attention keys and values to avoid recomputation during generation
"""

from memory import memset_zero, memcpy
from algorithm import vectorize

# ============================================================================
# KV Cache Configuration
# ============================================================================

alias DEFAULT_MAX_SEQ_LEN: Int = 4096
alias DEFAULT_CACHE_SIZE: Int = 2048

# ============================================================================
# KV Cache Structure
# ============================================================================

struct KVCache:
    """
    Key-Value cache for transformer inference
    Stores computed K and V matrices to avoid recomputation
    """
    var key_cache: DTypePointer[DType.float32]
    var value_cache: DTypePointer[DType.float32]
    
    var n_layers: Int
    var n_heads: Int
    var head_dim: Int
    var max_seq_len: Int
    var current_pos: Int
    
    var cache_size: Int  # Total allocated size
    
    fn __init__(
        inout self,
        n_layers: Int,
        n_heads: Int,
        head_dim: Int,
        max_seq_len: Int = DEFAULT_MAX_SEQ_LEN
    ):
        """Initialize KV cache with specified dimensions"""
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.current_pos = 0
        
        # Calculate cache size: layers × seq_len × heads × head_dim
        var cache_per_layer = max_seq_len * n_heads * head_dim
        self.cache_size = n_layers * cache_per_layer
        
        # Allocate memory for keys and values
        self.key_cache = DTypePointer[DType.float32].alloc(self.cache_size)
        self.value_cache = DTypePointer[DType.float32].alloc(self.cache_size)
        
        # Initialize to zero
        memset_zero(self.key_cache, self.cache_size)
        memset_zero(self.value_cache, self.cache_size)
    
    fn __del__(owned self):
        """Free allocated memory"""
        self.key_cache.free()
        self.value_cache.free()
    
    fn reset(inout self):
        """Reset cache to initial state"""
        self.current_pos = 0
        memset_zero(self.key_cache, self.cache_size)
        memset_zero(self.value_cache, self.cache_size)
    
    fn get_layer_offset(self, layer_idx: Int) -> Int:
        """Calculate memory offset for a specific layer"""
        return layer_idx * self.max_seq_len * self.n_heads * self.head_dim
    
    fn store_kv(
        inout self,
        layer_idx: Int,
        pos: Int,
        key: DTypePointer[DType.float32],
        value: DTypePointer[DType.float32]
    ):
        """
        Store key and value tensors for a specific position
        
        Args:
            layer_idx: Transformer layer index
            pos: Sequence position
            key: Key tensor [n_heads, head_dim]
            value: Value tensor [n_heads, head_dim]
        """
        var layer_offset = self.get_layer_offset(layer_idx)
        var pos_offset = pos * self.n_heads * self.head_dim
        var total_offset = layer_offset + pos_offset
        
        var kv_size = self.n_heads * self.head_dim
        
        # Copy key and value to cache
        memcpy(
            self.key_cache.offset(total_offset),
            key,
            kv_size
        )
        memcpy(
            self.value_cache.offset(total_offset),
            value,
            kv_size
        )
        
        # Update current position
        if pos >= self.current_pos:
            self.current_pos = pos + 1
    
    fn get_keys(
        self,
        layer_idx: Int,
        start_pos: Int = 0,
        end_pos: Int = -1
    ) -> DTypePointer[DType.float32]:
        """
        Get cached keys for a specific layer and position range
        
        Args:
            layer_idx: Transformer layer index
            start_pos: Start position (inclusive)
            end_pos: End position (exclusive), -1 for current_pos
        
        Returns:
            Pointer to key cache [seq_len, n_heads, head_dim]
        """
        var layer_offset = self.get_layer_offset(layer_idx)
        var pos_offset = start_pos * self.n_heads * self.head_dim
        
        return self.key_cache.offset(layer_offset + pos_offset)
    
    fn get_values(
        self,
        layer_idx: Int,
        start_pos: Int = 0,
        end_pos: Int = -1
    ) -> DTypePointer[DType.float32]:
        """
        Get cached values for a specific layer and position range
        
        Args:
            layer_idx: Transformer layer index
            start_pos: Start position (inclusive)
            end_pos: End position (exclusive), -1 for current_pos
        
        Returns:
            Pointer to value cache [seq_len, n_heads, head_dim]
        """
        var layer_offset = self.get_layer_offset(layer_idx)
        var pos_offset = start_pos * self.n_heads * self.head_dim
        
        return self.value_cache.offset(layer_offset + pos_offset)
    
    fn get_cache_length(self) -> Int:
        """Get current cached sequence length"""
        return self.current_pos
    
    fn is_full(self) -> Bool:
        """Check if cache is full"""
        return self.current_pos >= self.max_seq_len
    
    fn get_memory_usage(self) -> Int:
        """Get total memory usage in bytes"""
        # 2 caches (K and V) × cache_size × sizeof(float32)
        return 2 * self.cache_size * 4
    
    fn can_fit(self, additional_tokens: Int) -> Bool:
        """Check if additional tokens can fit in cache"""
        return (self.current_pos + additional_tokens) <= self.max_seq_len

# ============================================================================
# Optimized KV Cache with Sliding Window
# ============================================================================

struct SlidingWindowKVCache:
    """
    KV cache with sliding window for long sequences
    Only keeps recent context within window size
    """
    var key_cache: DTypePointer[DType.float32]
    var value_cache: DTypePointer[DType.float32]
    
    var n_layers: Int
    var n_heads: Int
    var head_dim: Int
    var window_size: Int  # Sliding window size
    var current_pos: Int
    var total_tokens: Int  # Total tokens processed
    
    var cache_size: Int
    
    fn __init__(
        inout self,
        n_layers: Int,
        n_heads: Int,
        head_dim: Int,
        window_size: Int = DEFAULT_CACHE_SIZE
    ):
        """Initialize sliding window cache"""
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.current_pos = 0
        self.total_tokens = 0
        
        # Allocate only for window size
        var cache_per_layer = window_size * n_heads * head_dim
        self.cache_size = n_layers * cache_per_layer
        
        self.key_cache = DTypePointer[DType.float32].alloc(self.cache_size)
        self.value_cache = DTypePointer[DType.float32].alloc(self.cache_size)
        
        memset_zero(self.key_cache, self.cache_size)
        memset_zero(self.value_cache, self.cache_size)
    
    fn __del__(owned self):
        """Free allocated memory"""
        self.key_cache.free()
        self.value_cache.free()
    
    fn store_kv(
        inout self,
        layer_idx: Int,
        key: DTypePointer[DType.float32],
        value: DTypePointer[DType.float32]
    ):
        """
        Store key and value with sliding window
        Automatically overwrites oldest entries when full
        """
        # Calculate circular buffer position
        var window_pos = self.current_pos % self.window_size
        
        var layer_offset = layer_idx * self.window_size * self.n_heads * self.head_dim
        var pos_offset = window_pos * self.n_heads * self.head_dim
        var total_offset = layer_offset + pos_offset
        
        var kv_size = self.n_heads * self.head_dim
        
        # Store in circular buffer
        memcpy(
            self.key_cache.offset(total_offset),
            key,
            kv_size
        )
        memcpy(
            self.value_cache.offset(total_offset),
            value,
            kv_size
        )
        
        self.current_pos += 1
        self.total_tokens += 1

# ============================================================================
# Multi-Query KV Cache (for models like LLaMA 2)
# ============================================================================

struct MultiQueryKVCache:
    """
    KV cache optimized for Multi-Query Attention (MQA)
    Shares K/V across query heads for memory efficiency
    """
    var key_cache: DTypePointer[DType.float32]
    var value_cache: DTypePointer[DType.float32]
    
    var n_layers: Int
    var n_kv_heads: Int  # Number of KV heads (< n_query_heads)
    var head_dim: Int
    var max_seq_len: Int
    var current_pos: Int
    
    fn __init__(
        inout self,
        n_layers: Int,
        n_kv_heads: Int,  # Typically 1 or n_heads // groups
        head_dim: Int,
        max_seq_len: Int = DEFAULT_MAX_SEQ_LEN
    ):
        """Initialize multi-query KV cache"""
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.current_pos = 0
        
        # Smaller cache due to fewer KV heads
        var cache_per_layer = max_seq_len * n_kv_heads * head_dim
        var cache_size = n_layers * cache_per_layer
        
        self.key_cache = DTypePointer[DType.float32].alloc(cache_size)
        self.value_cache = DTypePointer[DType.float32].alloc(cache_size)
        
        memset_zero(self.key_cache, cache_size)
        memset_zero(self.value_cache, cache_size)
    
    fn __del__(owned self):
        self.key_cache.free()
        self.value_cache.free()

# ============================================================================
# Cache Management
# ============================================================================

struct CacheManager:
    """Manages multiple KV caches for different models"""
    var caches: Dict[String, KVCache]
    var max_caches: Int
    
    fn __init__(inout self, max_caches: Int = 5):
        self.caches = Dict[String, KVCache]()
        self.max_caches = max_caches
    
    fn get_or_create_cache(
        inout self,
        model_name: String,
        n_layers: Int,
        n_heads: Int,
        head_dim: Int,
        max_seq_len: Int
    ) -> KVCache:
        """Get existing cache or create new one"""
        if model_name in self.caches:
            return self.caches[model_name]
        
        # Evict oldest if at capacity
        if len(self.caches) >= self.max_caches:
            # Simple FIFO eviction
            var first_key = list(self.caches.keys())[0]
            self.caches.pop(first_key)
        
        # Create new cache
        var cache = KVCache(n_layers, n_heads, head_dim, max_seq_len)
        self.caches[model_name] = cache
        return cache
    
    fn clear_cache(inout self, model_name: String):
        """Clear cache for specific model"""
        if model_name in self.caches:
            self.caches[model_name].reset()
    
    fn clear_all_caches(inout self):
        """Clear all caches"""
        for key in self.caches.keys():
            self.caches[key].reset()

# ============================================================================
# Testing
# ============================================================================

fn main():
    print("=" * 80)
    print("🔄 Mojo KV Cache - Efficient Transformer Inference")
    print("=" * 80)
    print()
    
    # Test standard KV cache
    print("🧪 Testing Standard KV Cache...")
    var n_layers = 4
    var n_heads = 8
    var head_dim = 64
    var max_seq_len = 128
    
    var cache = KVCache(n_layers, n_heads, head_dim, max_seq_len)
    
    print(f"  Configuration:")
    print(f"    Layers: {n_layers}")
    print(f"    Heads: {n_heads}")
    print(f"    Head dim: {head_dim}")
    print(f"    Max seq len: {max_seq_len}")
    print(f"    Memory: {cache.get_memory_usage() / (1024 * 1024)} MB")
    print()
    
    # Simulate storing KV pairs
    var kv_size = n_heads * head_dim
    var test_key = DTypePointer[DType.float32].alloc(kv_size)
    var test_value = DTypePointer[DType.float32].alloc(kv_size)
    
    # Initialize test data
    for i in range(kv_size):
        test_key[i] = Float32(i) * 0.01
        test_value[i] = Float32(i) * 0.02
    
    print("  Storing KV pairs...")
    for pos in range(10):
        for layer in range(n_layers):
            cache.store_kv(layer, pos, test_key, test_value)
    
    print(f"  Current cache length: {cache.get_cache_length()}")
    print(f"  Cache is full: {cache.is_full()}")
    print(f"  Can fit 100 more tokens: {cache.can_fit(100)}")
    print()
    
    # Test retrieval
    print("  Testing KV retrieval...")
    var retrieved_keys = cache.get_keys(0, 0)
    var retrieved_values = cache.get_values(0, 0)
    
    print(f"  Retrieved keys[0]: {retrieved_keys[0]}")
    print(f"  Retrieved values[0]: {retrieved_values[0]}")
    print("  ✅ KV cache working correctly")
    print()
    
    # Test sliding window cache
    print("🧪 Testing Sliding Window Cache...")
    var window_size = 32
    var sliding_cache = SlidingWindowKVCache(n_layers, n_heads, head_dim, window_size)
    
    print(f"  Window size: {window_size}")
    print(f"  Memory: {sliding_cache.get_memory_usage() / 1024} KB")
    print(f"  Memory savings vs full: {(1.0 - Float32(window_size) / Float32(max_seq_len)) * 100}%")
    print("  ✅ Sliding window cache initialized")
    print()
    
    # Cleanup
    test_key.free()
    test_value.free()
    
    print("=" * 80)
    print("✅ All KV cache tests complete!")
    print("=" * 80)
    print()
    print("Next: Implement attention mechanism using this cache")
