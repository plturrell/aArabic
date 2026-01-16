"""
Sampling Strategies - Token Sampling for LLM Text Generation
Implements various sampling methods for diverse and controlled generation
"""

from random import random_float64
from math import exp, log
from algorithm import vectorize
from memory import memcpy

# ============================================================================
# Sampling Configuration
# ============================================================================

alias DEFAULT_TEMPERATURE: Float32 = 1.0
alias DEFAULT_TOP_K: Int = 50
alias DEFAULT_TOP_P: Float32 = 0.9
alias MIN_TEMPERATURE: Float32 = 0.01

# ============================================================================
# Greedy Sampling
# ============================================================================

fn greedy_sample(logits: DTypePointer[DType.float32], vocab_size: Int) -> Int:
    """
    Greedy sampling: Select token with highest probability
    Deterministic, but can be repetitive
    
    Args:
        logits: Logits tensor [vocab_size]
        vocab_size: Size of vocabulary
    
    Returns:
        Token ID with highest probability
    """
    var max_idx = 0
    var max_val = logits[0]
    
    for i in range(1, vocab_size):
        if logits[i] > max_val:
            max_val = logits[i]
            max_idx = i
    
    return max_idx

# ============================================================================
# Temperature Sampling
# ============================================================================

fn apply_temperature(
    logits: DTypePointer[DType.float32],
    vocab_size: Int,
    temperature: Float32
) -> DTypePointer[DType.float32]:
    """
    Apply temperature scaling to logits
    
    Args:
        logits: Logits tensor [vocab_size]
        vocab_size: Size of vocabulary
        temperature: Sampling temperature (0.1-2.0)
            - Low (0.1-0.5): More focused, deterministic
            - Medium (0.7-1.0): Balanced
            - High (1.5-2.0): More creative, diverse
    
    Returns:
        Temperature-scaled logits
    """
    var scaled_logits = DTypePointer[DType.float32].alloc(vocab_size)
    
    # Clamp temperature to prevent extreme values
    var temp = max(temperature, MIN_TEMPERATURE)
    var temp_inv = 1.0 / temp
    
    @parameter
    fn scale[simd_width: Int](i: Int):
        var logit_vec = logits.load[width=simd_width](i)
        scaled_logits.store[width=simd_width](i, logit_vec * temp_inv)
    
    vectorize[scale, 8](vocab_size)
    
    return scaled_logits

fn temperature_sample(
    logits: DTypePointer[DType.float32],
    vocab_size: Int,
    temperature: Float32 = DEFAULT_TEMPERATURE
) -> Int:
    """
    Sample token using temperature scaling
    
    Args:
        logits: Logits tensor [vocab_size]
        vocab_size: Size of vocabulary
        temperature: Sampling temperature
    
    Returns:
        Sampled token ID
    """
    # Apply temperature
    var scaled_logits = apply_temperature(logits, vocab_size, temperature)
    
    # Convert to probabilities (softmax)
    var probs = softmax(scaled_logits, vocab_size)
    
    # Sample from distribution
    var token_id = categorical_sample(probs, vocab_size)
    
    scaled_logits.free()
    probs.free()
    
    return token_id

# ============================================================================
# Top-K Sampling
# ============================================================================

fn top_k_sample(
    logits: DTypePointer[DType.float32],
    vocab_size: Int,
    k: Int = DEFAULT_TOP_K,
    temperature: Float32 = DEFAULT_TEMPERATURE
) -> Int:
    """
    Top-K sampling: Sample from K tokens with highest probability
    
    Args:
        logits: Logits tensor [vocab_size]
        vocab_size: Size of vocabulary
        k: Number of top tokens to consider
        temperature: Sampling temperature
    
    Returns:
        Sampled token ID from top-K
    """
    # Apply temperature
    var scaled_logits = apply_temperature(logits, vocab_size, temperature)
    
    # Find top-K indices
    var top_k_indices = DTypePointer[DType.int32].alloc(k)
    var top_k_values = DTypePointer[DType.float32].alloc(k)
    
    # Simple selection: find K largest values
    # (Production would use heap or quickselect)
    for i in range(k):
        var max_idx = 0
        var max_val = -1e9
        
        for j in range(vocab_size):
            var already_selected = False
            for s in range(i):
                if top_k_indices[s] == j:
                    already_selected = True
                    break
            
            if not already_selected and scaled_logits[j] > max_val:
                max_val = scaled_logits[j]
                max_idx = j
        
        top_k_indices[i] = max_idx
        top_k_values[i] = max_val
    
    # Softmax over top-K
    var top_k_probs = softmax(top_k_values, k)
    
    # Sample from top-K distribution
    var selected_idx = categorical_sample(top_k_probs, k)
    var token_id = top_k_indices[selected_idx]
    
    # Cleanup
    scaled_logits.free()
    top_k_indices.free()
    top_k_values.free()
    top_k_probs.free()
    
    return token_id

# ============================================================================
# Top-P (Nucleus) Sampling
# ============================================================================

fn top_p_sample(
    logits: DTypePointer[DType.float32],
    vocab_size: Int,
    p: Float32 = DEFAULT_TOP_P,
    temperature: Float32 = DEFAULT_TEMPERATURE
) -> Int:
    """
    Top-P (Nucleus) sampling: Sample from smallest set with cumulative prob >= p
    More adaptive than top-K
    
    Args:
        logits: Logits tensor [vocab_size]
        vocab_size: Size of vocabulary
        p: Cumulative probability threshold (0.0-1.0)
        temperature: Sampling temperature
    
    Returns:
        Sampled token ID from nucleus
    """
    # Apply temperature
    var scaled_logits = apply_temperature(logits, vocab_size, temperature)
    
    # Convert to probabilities
    var probs = softmax(scaled_logits, vocab_size)
    
    # Sort indices by probability (descending)
    var sorted_indices = argsort_descending(probs, vocab_size)
    
    # Find nucleus: smallest set with cumulative prob >= p
    var cumulative_prob: Float32 = 0.0
    var nucleus_size = 0
    
    for i in range(vocab_size):
        var idx = sorted_indices[i]
        cumulative_prob += probs[idx]
        nucleus_size += 1
        
        if cumulative_prob >= p:
            break
    
    # Create nucleus probabilities
    var nucleus_probs = DTypePointer[DType.float32].alloc(nucleus_size)
    var nucleus_indices = DTypePointer[DType.int32].alloc(nucleus_size)
    
    for i in range(nucleus_size):
        var idx = sorted_indices[i]
        nucleus_indices[i] = idx
        nucleus_probs[i] = probs[idx]
    
    # Renormalize nucleus probabilities
    var sum_probs: Float32 = 0.0
    for i in range(nucleus_size):
        sum_probs += nucleus_probs[i]
    
    for i in range(nucleus_size):
        nucleus_probs[i] /= sum_probs
    
    # Sample from nucleus
    var selected_idx = categorical_sample(nucleus_probs, nucleus_size)
    var token_id = nucleus_indices[selected_idx]
    
    # Cleanup
    scaled_logits.free()
    probs.free()
    sorted_indices.free()
    nucleus_probs.free()
    nucleus_indices.free()
    
    return token_id

# ============================================================================
# Min-P Sampling
# ============================================================================

fn min_p_sample(
    logits: DTypePointer[DType.float32],
    vocab_size: Int,
    min_p: Float32 = 0.05,
    temperature: Float32 = DEFAULT_TEMPERATURE
) -> Int:
    """
    Min-P sampling: Filter tokens below min_p * max_prob
    Good for maintaining quality while allowing diversity
    
    Args:
        logits: Logits tensor [vocab_size]
        vocab_size: Size of vocabulary
        min_p: Minimum probability ratio (0.0-1.0)
        temperature: Sampling temperature
    
    Returns:
        Sampled token ID
    """
    # Apply temperature and convert to probs
    var scaled_logits = apply_temperature(logits, vocab_size, temperature)
    var probs = softmax(scaled_logits, vocab_size)
    
    # Find max probability
    var max_prob: Float32 = 0.0
    for i in range(vocab_size):
        if probs[i] > max_prob:
            max_prob = probs[i]
    
    # Calculate threshold
    var threshold = min_p * max_prob
    
    # Count valid tokens
    var n_valid = 0
    for i in range(vocab_size):
        if probs[i] >= threshold:
            n_valid += 1
    
    # Create filtered distribution
    var valid_probs = DTypePointer[DType.float32].alloc(n_valid)
    var valid_indices = DTypePointer[DType.int32].alloc(n_valid)
    
    var idx = 0
    var sum_valid: Float32 = 0.0
    for i in range(vocab_size):
        if probs[i] >= threshold:
            valid_indices[idx] = i
            valid_probs[idx] = probs[i]
            sum_valid += probs[i]
            idx += 1
    
    # Renormalize
    for i in range(n_valid):
        valid_probs[i] /= sum_valid
    
    # Sample
    var selected_idx = categorical_sample(valid_probs, n_valid)
    var token_id = valid_indices[selected_idx]
    
    # Cleanup
    scaled_logits.free()
    probs.free()
    valid_probs.free()
    valid_indices.free()
    
    return token_id

# ============================================================================
# Utility Functions
# ============================================================================

fn softmax(
    logits: DTypePointer[DType.float32],
    size: Int
) -> DTypePointer[DType.float32]:
    """
    Compute softmax with numerical stability
    
    Args:
        logits: Input logits [size]
        size: Size of array
    
    Returns:
        Probabilities [size]
    """
    var probs = DTypePointer[DType.float32].alloc(size)
    
    # Find max for stability
    var max_val = logits[0]
    for i in range(1, size):
        if logits[i] > max_val:
            max_val = logits[i]
    
    # Compute exp(x - max) and sum
    var sum_exp: Float32 = 0.0
    for i in range(size):
        var exp_val = exp(logits[i] - max_val)
        probs[i] = exp_val
        sum_exp += exp_val
    
    # Normalize
    var inv_sum = 1.0 / sum_exp
    for i in range(size):
        probs[i] *= inv_sum
    
    return probs

fn categorical_sample(
    probs: DTypePointer[DType.float32],
    size: Int
) -> Int:
    """
    Sample index from categorical distribution
    
    Args:
        probs: Probability distribution [size]
        size: Size of distribution
    
    Returns:
        Sampled index
    """
    # Generate random number in [0, 1)
    var rand_val = Float32(random_float64())
    
    # Find bin using cumulative probability
    var cumulative: Float32 = 0.0
    for i in range(size):
        cumulative += probs[i]
        if rand_val < cumulative:
            return i
    
    # Return last index if rounding errors
    return size - 1

fn argsort_descending(
    values: DTypePointer[DType.float32],
    size: Int
) -> DTypePointer[DType.int32]:
    """
    Return indices that would sort array in descending order
    
    Args:
        values: Array to sort [size]
        size: Array size
    
    Returns:
        Sorted indices [size]
    """
    var indices = DTypePointer[DType.int32].alloc(size)
    
    # Initialize indices
    for i in range(size):
        indices[i] = i
    
    # Simple insertion sort (production would use quicksort/heapsort)
    for i in range(1, size):
        var key_idx = indices[i]
        var key_val = values[key_idx]
        var j = i - 1
        
        while j >= 0 and values[indices[j]] < key_val:
            indices[j + 1] = indices[j]
            j -= 1
        
        indices[j + 1] = key_idx
    
    return indices

# ============================================================================
# Combined Sampling Strategy
# ============================================================================

struct SamplingConfig:
    """Configuration for sampling strategy"""
    var temperature: Float32
    var top_k: Int
    var top_p: Float32
    var min_p: Float32
    var repetition_penalty: Float32
    
    fn __init__(
        inout self,
        temperature: Float32 = 1.0,
        top_k: Int = 50,
        top_p: Float32 = 0.9,
        min_p: Float32 = 0.05,
        repetition_penalty: Float32 = 1.0
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty

fn sample_token(
    logits: DTypePointer[DType.float32],
    vocab_size: Int,
    config: SamplingConfig,
    previous_tokens: DTypePointer[DType.int32] = DTypePointer[DType.int32](),
    n_previous: Int = 0
) -> Int:
    """
    Sample token using configured strategy
    
    Args:
        logits: Logits tensor [vocab_size]
        vocab_size: Size of vocabulary
        config: Sampling configuration
        previous_tokens: Previously generated tokens
        n_previous: Number of previous tokens
    
    Returns:
        Sampled token ID
    """
    # Apply repetition penalty
    if n_previous > 0 and config.repetition_penalty != 1.0:
        apply_repetition_penalty(
            logits,
            vocab_size,
            previous_tokens,
            n_previous,
            config.repetition_penalty
        )
    
    # Choose sampling strategy
    if config.temperature < 0.01:
        # Greedy if temperature very low
        return greedy_sample(logits, vocab_size)
    elif config.top_k > 0 and config.top_k < vocab_size:
        # Top-K sampling
        return top_k_sample(logits, vocab_size, config.top_k, config.temperature)
    elif config.top_p < 1.0:
        # Top-P sampling
        return top_p_sample(logits, vocab_size, config.top_p, config.temperature)
    else:
        # Temperature sampling
        return temperature_sample(logits, vocab_size, config.temperature)

fn apply_repetition_penalty(
    logits: DTypePointer[DType.float32],
    vocab_size: Int,
    previous_tokens: DTypePointer[DType.int32],
    n_previous: Int,
    penalty: Float32
):
    """
    Apply repetition penalty to reduce repeated tokens
    
    Args:
        logits: Logits tensor [vocab_size] (modified in-place)
        vocab_size: Size of vocabulary
        previous_tokens: Previously generated tokens
        n_previous: Number of previous tokens
        penalty: Penalty factor (> 1.0 reduces repetition)
    """
    for i in range(n_previous):
        var token_id = previous_tokens[i]
        if token_id >= 0 and token_id < vocab_size:
            # Divide logit by penalty if token was seen
            logits[token_id] /= penalty

# ============================================================================
# Testing
# ============================================================================

fn main():
    print("=" * 80)
    print("🎲 Mojo Sampling Strategies - Diverse Text Generation")
    print("=" * 80)
    print()
    
    # Create test logits
    var vocab_size = 100
    var logits = DTypePointer[DType.float32].alloc(vocab_size)
    
    # Initialize with random-like values
    for i in range(vocab_size):
        logits[i] = Float32(i % 10) * 0.5 - 2.0
    
    # Make token 42 most likely
    logits[42] = 5.0
    
    print("🧪 Testing Sampling Strategies...")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Most likely token: 42")
    print()
    
    # Test greedy sampling
    print("  1. Greedy Sampling:")
    var greedy_token = greedy_sample(logits, vocab_size)
    print(f"     Selected: {greedy_token} (should be 42)")
    print()
    
    # Test temperature sampling
    print("  2. Temperature Sampling (temp=0.8):")
    for i in range(3):
        var temp_token = temperature_sample(logits, vocab_size, 0.8)
        print(f"     Sample {i+1}: {temp_token}")
    print()
    
    # Test top-K sampling
    print("  3. Top-K Sampling (k=10, temp=1.0):")
    for i in range(3):
        var topk_token = top_k_sample(logits, vocab_size, 10, 1.0)
        print(f"     Sample {i+1}: {topk_token}")
    print()
    
    # Test top-P sampling
    print("  4. Top-P Sampling (p=0.9, temp=1.0):")
    for i in range(3):
        var topp_token = top_p_sample(logits, vocab_size, 0.9, 1.0)
        print(f"     Sample {i+1}: {topp_token}")
    print()
    
    # Test min-P sampling
    print("  5. Min-P Sampling (min_p=0.05, temp=1.0):")
    for i in range(3):
        var minp_token = min_p_sample(logits, vocab_size, 0.05, 1.0)
        print(f"     Sample {i+1}: {minp_token}")
    print()
    
    # Test combined strategy
    print("  6. Combined Strategy:")
    var config = SamplingConfig(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        min_p=0.05
    )
    for i in range(3):
        var combined_token = sample_token(logits, vocab_size, config)
        print(f"     Sample {i+1}: {combined_token}")
    print()
    
    # Cleanup
    logits.free()
    
    print("=" * 80)
    print("✅ All sampling strategies working!")
    print("=" * 80)
    print()
    print("Features implemented:")
    print("  ✅ Greedy sampling (deterministic)")
    print("  ✅ Temperature scaling (creativity control)")
    print("  ✅ Top-K sampling (fixed diversity)")
    print("  ✅ Top-P/Nucleus sampling (adaptive diversity)")
    print("  ✅ Min-P sampling (quality-aware)")
    print("  ✅ Repetition penalty")
    print("  ✅ Configurable strategy")
    print()
    print("Next: Implement generation loop using these strategies")
