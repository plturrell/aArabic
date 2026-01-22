"""
KTO Policy Network for Tool Orchestration
Reuses Titans transformer architecture for policy network

KTO (Kahneman-Tversky Optimization):
- Models loss aversion (λ = 2.25)
- Binary feedback (success/failure)
- Sample efficient learning
- Human-aligned preferences

Architecture:
State → Encoder → Titans Transformer → Action/Value Heads → Tool Selection

Day 43: mHC Integration
- Stability-weighted policy updates
- Manifold-constrained action selection
- Policy stability tracking metrics
"""

from collections import Dict, List
from ..state import OrchestrationState, StateEncoder
from ..execution import ExecutionStrategy
from ..registry import ToolRegistry


# ============================================================================
# mHC Stability Tracking for KTO Policy (Day 43)
# ============================================================================

@value
struct MHCPolicyConfig:
    """Configuration for mHC constraints in KTO policy"""
    var enabled: Bool
    var manifold_beta: Float32        # Maximum L2 norm bound for policy outputs
    var stability_threshold: Float32  # Threshold for amplification check [0.9, 1.1]
    var adaptive_beta: Bool           # Enable adaptive beta based on history

    fn __init__(out self):
        self.enabled = True
        self.manifold_beta = 10.0
        self.stability_threshold = 0.1  # 10% deviation allowed
        self.adaptive_beta = True


@value
struct PolicyStabilityMetrics:
    """Stability metrics for KTO policy updates with mHC"""
    var update_count: Int
    var norm_before: Float32
    var norm_after: Float32
    var amplification_factor: Float32
    var is_stable: Bool
    var constraint_violations: Int
    var avg_stability_score: Float32

    fn __init__(out self, norm_before: Float32, norm_after: Float32):
        self.update_count = 1
        self.norm_before = norm_before
        self.norm_after = norm_after
        self.amplification_factor = norm_after / norm_before if norm_before > 0 else 1.0
        # Stable if amplification in [0.9, 1.1] range
        self.is_stable = self.amplification_factor >= 0.9 and self.amplification_factor <= 1.1
        self.constraint_violations = 0 if self.is_stable else 1
        self.avg_stability_score = 1.0 if self.is_stable else 0.5


# ============================================================================
# Core KTO Policy Types
# ============================================================================

@value
struct ToolAction:
    """
    Action selected by KTO policy
    
    Represents a decision to execute a tool with specific parameters
    and strategy
    """
    var tool_name: String
    var parameters: Dict[String, String]
    var strategy: ExecutionStrategy
    var confidence: Float32  # 0.0 to 1.0
    var expected_value: Float32  # V(s,a)
    
    fn __init__(
        inout self,
        tool_name: String,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        confidence: Float32 = 0.5
    ):
        self.tool_name = tool_name
        self.parameters = Dict[String, String]()
        self.strategy = strategy
        self.confidence = confidence
        self.expected_value = 0.0


@value
struct PolicyOutput:
    """Output from policy network forward pass"""
    var action_logits: List[Float32]  # Raw logits for each action
    var value_estimate: Float32  # V(s)
    var action_probs: List[Float32]  # Softmax probabilities
    var num_actions: Int
    
    fn __init__(inout self, num_actions: Int = 10):
        self.action_logits = List[Float32]()
        self.value_estimate = 0.0
        self.action_probs = List[Float32]()
        self.num_actions = num_actions


# ============================================================================
# Placeholder for Titans Transformer Integration
# ============================================================================

# TODO: Import from Titans when integrated
# from titans.transformer.model import TransformerModel
# from titans.types import Tensor2D

# Placeholder transformer interface
struct TransformerModel:
    """
    Placeholder for Titans transformer
    
    In production, this would be imported from:
    /Users/user/Documents/competitions/aimo_3/mojo_solvers/titans/
    
    The actual Titans transformer provides:
    - Multi-head attention
    - Feed-forward layers
    - Layer normalization
    - SIMD-optimized operations
    - 30-100x faster than Rust baseline
    """
    var d_model: Int
    var n_heads: Int
    var n_layers: Int
    var d_ff: Int
    
    fn __init__(
        inout self,
        d_model: Int = 256,
        n_heads: Int = 8,
        n_layers: Int = 6,
        d_ff: Int = 1024
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
    
    fn forward(self, input: List[Float32]) -> List[Float32]:
        """
        Forward pass through transformer
        
        In production: Full Titans transformer implementation
        For now: Placeholder that returns identity
        """
        return input


# ============================================================================
# KTO Policy Network
# ============================================================================

struct KTOPolicy:
    """
    KTO-based policy network for tool orchestration

    Key Components:
    1. State Encoder: Converts OrchestrationState to embeddings
    2. Titans Transformer: Core reasoning engine (REUSED!)
    3. Action Head: Maps hidden states to tool actions
    4. Value Head: Estimates state value V(s)

    KTO Specific:
    - Loss aversion coefficient: λ = 2.25
    - Reference policy for KL regularization
    - Desirable/undesirable experience handling

    Day 43 mHC Integration:
    - mhc_stability_weight: Weight for stability in policy updates
    - Manifold-constrained action logits
    - Policy stability tracking
    """
    var transformer: TransformerModel
    var state_encoder: StateEncoder
    var action_head: ActionHead
    var value_head: ValueHead
    var registry: ToolRegistry

    # KTO hyperparameters
    var lambda_loss_aversion: Float32  # λ = 2.25 (standard KTO)
    var reference_policy_ema: Float32  # EMA coefficient for ref policy
    var temperature: Float32  # Sampling temperature

    # Day 43: mHC Integration Fields
    var mhc_stability_weight: Float32    # Weight for stability penalty in loss
    var mhc_config: MHCPolicyConfig      # mHC configuration
    var stability_history: List[Float32] # History of stability scores
    var total_updates: Int               # Total policy updates
    var stable_updates: Int              # Updates within stability bounds

    fn __init__(
        inout self,
        registry: ToolRegistry,
        d_model: Int = 256,
        n_heads: Int = 8,
        n_layers: Int = 6,
        mhc_stability_weight: Float32 = 0.1
    ):
        """
        Initialize KTO policy network

        Args:
            registry: Tool registry for action space
            d_model: Transformer embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            mhc_stability_weight: Weight for mHC stability in loss (Day 43)
        """
        self.registry = registry

        # Initialize Titans transformer (REUSE!)
        self.transformer = TransformerModel(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * 4
        )

        # State encoding
        self.state_encoder = StateEncoder(
            embedding_dim=d_model,
            max_tools=50
        )

        # Output heads
        self.action_head = ActionHead(
            input_dim=d_model,
            num_actions=registry.total_tools
        )
        self.value_head = ValueHead(input_dim=d_model)

        # KTO hyperparameters (from paper)
        self.lambda_loss_aversion = 2.25  # Standard value from psychology
        self.reference_policy_ema = 0.99  # Reference policy EMA
        self.temperature = 1.0  # Sampling temperature

        # Day 43: mHC Integration Initialization
        self.mhc_stability_weight = mhc_stability_weight
        self.mhc_config = MHCPolicyConfig()
        self.stability_history = List[Float32]()
        self.total_updates = 0
        self.stable_updates = 0
    
    
    # ========================================================================
    # Forward Pass
    # ========================================================================
    
    fn forward(self, state: OrchestrationState) -> PolicyOutput:
        """
        Forward pass through policy network
        
        Steps:
        1. Encode state to embedding
        2. Pass through Titans transformer
        3. Generate action logits (action head)
        4. Estimate value (value head)
        5. Compute action probabilities (softmax)
        
        Args:
            state: Current orchestration state
            
        Returns:
            PolicyOutput with action logits, value estimate, and probs
        """
        # 1. Encode state
        let state_features = self.state_encoder.encode(state)
        
        # 2. Pass through Titans transformer
        let hidden = self.transformer.forward(state_features)
        
        # 3. Action head: hidden → action logits
        let action_logits = self.action_head.forward(hidden)
        
        # 4. Value head: hidden → value estimate
        let value_estimate = self.value_head.forward(hidden)
        
        # 5. Softmax for action probabilities
        let action_probs = self._softmax(action_logits)
        
        var output = PolicyOutput(num_actions=len(action_logits))
        output.action_logits = action_logits
        output.value_estimate = value_estimate
        output.action_probs = action_probs
        
        return output
    
    fn select_action(
        self,
        state: OrchestrationState,
        greedy: Bool = False
    ) -> ToolAction:
        """
        Select action using policy
        
        Args:
            state: Current state
            greedy: If True, select argmax; if False, sample from distribution
            
        Returns:
            ToolAction with selected tool and strategy
        """
        let output = self.forward(state)
        
        # Select action index
        var action_idx: Int
        if greedy:
            action_idx = self._argmax(output.action_probs)
        else:
            action_idx = self._sample_categorical(output.action_probs)
        
        # Map action index to tool name
        let tool_name = self._index_to_tool_name(action_idx)
        
        # Create action with confidence
        let confidence = output.action_probs[action_idx] if action_idx < len(output.action_probs) else 0.5
        
        var action = ToolAction(
            tool_name=tool_name,
            strategy=ExecutionStrategy.RL_OPTIMIZED,
            confidence=confidence
        )
        action.expected_value = output.value_estimate
        
        return action
    
    
    # ========================================================================
    # KTO Loss Computation
    # ========================================================================
    
    fn compute_kto_loss(
        self,
        desirable_states: List[OrchestrationState],
        desirable_actions: List[ToolAction],
        undesirable_states: List[OrchestrationState],
        undesirable_actions: List[ToolAction],
        reference_policy: KTOPolicy
    ) -> Float32:
        """
        Compute KTO loss function
        
        L_KTO = E_desirable[KL(π_θ || π_ref) - λ * log π_θ(a|s)] +
                E_undesirable[KL(π_θ || π_ref) - (1/λ) * log π_θ(a|s)]
        
        Where:
        - π_θ = current policy (self)
        - π_ref = reference policy (EMA of past policies)
        - λ = 2.25 (loss aversion coefficient)
        
        Args:
            desirable_states: States where action succeeded
            desirable_actions: Actions that succeeded
            undesirable_states: States where action failed
            undesirable_actions: Actions that failed
            reference_policy: Reference policy for KL divergence
            
        Returns:
            Total KTO loss
        """
        var total_loss: Float32 = 0.0
        let n_desirable = len(desirable_states)
        let n_undesirable = len(undesirable_states)
        
        # Desirable experiences (successful executions)
        for i in range(n_desirable):
            let state = desirable_states[i]
            let action = desirable_actions[i]
            
            # Current policy log prob
            let output = self.forward(state)
            let log_prob = self._compute_log_prob(output, action)
            
            # Reference policy for KL divergence
            let ref_output = reference_policy.forward(state)
            let kl_div = self._compute_kl_divergence(output.action_probs, ref_output.action_probs)
            
            # Gain utility: standard weighting
            total_loss += kl_div - self.lambda_loss_aversion * log_prob
        
        # Undesirable experiences (failed executions)
        for i in range(n_undesirable):
            let state = undesirable_states[i]
            let action = undesirable_actions[i]
            
            let output = self.forward(state)
            let log_prob = self._compute_log_prob(output, action)
            
            let ref_output = reference_policy.forward(state)
            let kl_div = self._compute_kl_divergence(output.action_probs, ref_output.action_probs)
            
            # Loss aversion: asymmetric weighting (1/λ)
            total_loss += kl_div - (1.0 / self.lambda_loss_aversion) * log_prob
        
        # Average over all experiences
        return total_loss / Float32(n_desirable + n_undesirable)


    # ========================================================================
    # Day 43: mHC Integration Methods
    # ========================================================================

    fn compute_kto_loss_with_mhc(
        inout self,
        desirable_states: List[OrchestrationState],
        desirable_actions: List[ToolAction],
        undesirable_states: List[OrchestrationState],
        undesirable_actions: List[ToolAction],
        reference_policy: KTOPolicy
    ) -> Float32:
        """
        Compute KTO loss with mHC stability constraint (Day 43)

        L_KTO_mHC = L_KTO + α * L_stability

        Where:
        - L_KTO = standard KTO loss
        - α = mhc_stability_weight
        - L_stability = penalty for policy distribution instability
        """
        # Compute base KTO loss
        let base_loss = self.compute_kto_loss(
            desirable_states, desirable_actions,
            undesirable_states, undesirable_actions,
            reference_policy
        )

        if not self.mhc_config.enabled:
            return base_loss

        # Compute stability penalty
        let stability_penalty = self._compute_stability_penalty(
            desirable_states, undesirable_states
        )

        # Update metrics
        self.total_updates += 1
        let is_stable = stability_penalty < self.mhc_config.stability_threshold
        if is_stable:
            self.stable_updates += 1

        # Track stability history
        let stability_score = 1.0 - stability_penalty
        self.stability_history.append(stability_score)

        # Combined loss: KTO + mHC stability
        return base_loss + self.mhc_stability_weight * stability_penalty

    fn _compute_stability_penalty(
        self,
        desirable_states: List[OrchestrationState],
        undesirable_states: List[OrchestrationState]
    ) -> Float32:
        """
        Compute stability penalty based on action probability distribution variance

        Measures how much the policy output norms deviate from ideal range
        """
        var total_penalty: Float32 = 0.0
        var count: Int = 0

        # Check desirable states
        for i in range(len(desirable_states)):
            let output = self.forward(desirable_states[i])
            let norm = self._compute_prob_norm(output.action_probs)
            let penalty = self._norm_deviation_penalty(norm)
            total_penalty += penalty
            count += 1

        # Check undesirable states
        for i in range(len(undesirable_states)):
            let output = self.forward(undesirable_states[i])
            let norm = self._compute_prob_norm(output.action_probs)
            let penalty = self._norm_deviation_penalty(norm)
            total_penalty += penalty
            count += 1

        return total_penalty / Float32(count) if count > 0 else 0.0

    fn _norm_deviation_penalty(self, norm: Float32) -> Float32:
        """Compute penalty for norm deviation from manifold bound"""
        let beta = self.mhc_config.manifold_beta
        if norm > beta:
            # Exceeded bound: quadratic penalty
            let excess = norm - beta
            return excess * excess / (beta * beta)
        else:
            # Within bound: no penalty
            return 0.0

    fn _compute_prob_norm(self, probs: List[Float32]) -> Float32:
        """Compute L2 norm of probability distribution"""
        var sum_sq: Float32 = 0.0
        for i in range(len(probs)):
            sum_sq += probs[i] * probs[i]
        return sqrt(sum_sq)

    fn select_action_with_stability(
        self,
        state: OrchestrationState,
        greedy: Bool = False
    ) -> ToolAction:
        """
        Select action with mHC stability consideration (Day 43)

        Applies manifold constraints to action logits before selection
        """
        var output = self.forward(state)

        # Apply mHC manifold constraints to action logits
        if self.mhc_config.enabled:
            output.action_logits = self._apply_manifold_constraints(
                output.action_logits, self.mhc_config.manifold_beta
            )
            # Recompute probabilities after constraint
            output.action_probs = self._softmax(output.action_logits)

        # Select action index
        var action_idx: Int
        if greedy:
            action_idx = self._argmax(output.action_probs)
        else:
            action_idx = self._sample_categorical(output.action_probs)

        # Map action index to tool name
        let tool_name = self._index_to_tool_name(action_idx)

        # Create action with confidence
        let confidence = output.action_probs[action_idx] if action_idx < len(output.action_probs) else 0.5

        var action = ToolAction(
            tool_name=tool_name,
            strategy=ExecutionStrategy.RL_OPTIMIZED,
            confidence=confidence
        )
        action.expected_value = output.value_estimate

        return action

    fn _apply_manifold_constraints(
        self,
        logits: List[Float32],
        beta: Float32
    ) -> List[Float32]:
        """
        Apply mHC manifold constraints to logits
        Projects onto L2 ball: ||x||₂ ≤ β
        """
        # Compute L2 norm
        var norm_sq: Float32 = 0.0
        for i in range(len(logits)):
            norm_sq += logits[i] * logits[i]
        let norm = sqrt(norm_sq)

        # Project if exceeds bound
        if norm > beta:
            let scale = beta / norm
            var constrained = List[Float32]()
            for i in range(len(logits)):
                constrained.append(logits[i] * scale)
            return constrained
        else:
            return logits

    fn get_stability_metrics(self) -> PolicyStabilityMetrics:
        """Get current policy stability metrics (Day 43)"""
        let avg_stability = self._compute_average_stability()
        var metrics = PolicyStabilityMetrics(
            norm_before=1.0,
            norm_after=avg_stability
        )
        metrics.update_count = self.total_updates
        metrics.constraint_violations = self.total_updates - self.stable_updates
        metrics.avg_stability_score = avg_stability
        return metrics

    fn _compute_average_stability(self) -> Float32:
        """Compute average stability score from history"""
        if len(self.stability_history) == 0:
            return 1.0
        var sum: Float32 = 0.0
        for i in range(len(self.stability_history)):
            sum += self.stability_history[i]
        return sum / Float32(len(self.stability_history))

    fn reset_stability_tracking(inout self):
        """Reset stability tracking metrics"""
        self.stability_history = List[Float32]()
        self.total_updates = 0
        self.stable_updates = 0


    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    fn _softmax(self, logits: List[Float32]) -> List[Float32]:
        """Compute softmax probabilities"""
        var probs = List[Float32]()
        var max_logit: Float32 = -1e9
        
        # Find max for numerical stability
        for i in range(len(logits)):
            max_logit = max(max_logit, logits[i])
        
        # Compute exp and sum
        var sum_exp: Float32 = 0.0
        for i in range(len(logits)):
            let exp_val = exp(logits[i] - max_logit)
            probs.append(exp_val)
            sum_exp += exp_val
        
        # Normalize
        for i in range(len(probs)):
            probs[i] = probs[i] / sum_exp
        
        return probs
    
    fn _argmax(self, probs: List[Float32]) -> Int:
        """Find index of maximum probability"""
        var max_idx = 0
        var max_val = probs[0] if len(probs) > 0 else 0.0
        
        for i in range(1, len(probs)):
            if probs[i] > max_val:
                max_val = probs[i]
                max_idx = i
        
        return max_idx
    
    fn _sample_categorical(self, probs: List[Float32]) -> Int:
        """Sample from categorical distribution"""
        # Simplified sampling - full implementation would use proper RNG
        return self._argmax(probs)
    
    fn _compute_log_prob(self, output: PolicyOutput, action: ToolAction) -> Float32:
        """
        Compute log probability of action: log π_θ(a|s)

        Maps action to index and returns log of probability.
        """
        # Map tool name to action index
        let action_idx = self._tool_name_to_index(action.tool_name)

        # Bounds check
        if action_idx < 0 or action_idx >= len(output.action_probs):
            return -1e9  # Very low log prob for invalid actions

        # Get probability and compute log
        let prob = output.action_probs[action_idx]

        # Numerical stability: clamp probability
        let clamped_prob = max(prob, 1e-10)

        return log(clamped_prob)

    fn _tool_name_to_index(self, tool_name: String) -> Int:
        """
        Map tool name to action index using registry

        Returns action index or -1 if not found
        """
        let tools = self.registry.list_all_tools()

        for i in range(len(tools)):
            if tools[i] == tool_name:
                return i

        # Fallback: hash-based mapping if tool not in registry
        var hash_val: UInt64 = 0
        let bytes = tool_name.as_bytes()
        for i in range(len(bytes)):
            hash_val = hash_val * 31 + UInt64(bytes[i])

        return Int(hash_val % UInt64(max(1, self.registry.total_tools)))
    
    fn _compute_kl_divergence(
        self,
        p: List[Float32],
        q: List[Float32]
    ) -> Float32:
        """
        Compute KL divergence: KL(p || q)
        
        KL(p || q) = Σ p(x) * log(p(x) / q(x))
        """
        var kl: Float32 = 0.0
        
        for i in range(min(len(p), len(q))):
            if p[i] > 0 and q[i] > 0:
                kl += p[i] * log(p[i] / q[i])
        
        return kl
    
    fn _index_to_tool_name(self, idx: Int) -> String:
        """
        Map action index to tool name using registry

        Args:
            idx: Action index

        Returns:
            Tool name from registry, or fallback name if index out of bounds
        """
        let tools = self.registry.list_all_tools()

        if idx >= 0 and idx < len(tools):
            return tools[idx]

        # Fallback for out-of-bounds index
        return "unknown_tool_" + String(idx)


# ============================================================================
# Action and Value Heads
# ============================================================================

struct ActionHead:
    """
    Action head: maps transformer hidden states to action logits
    
    Architecture: Linear(d_model → num_actions)
    """
    var input_dim: Int
    var num_actions: Int
    
    fn __init__(inout self, input_dim: Int, num_actions: Int):
        self.input_dim = input_dim
        self.num_actions = num_actions
    
    fn forward(self, hidden: List[Float32]) -> List[Float32]:
        """
        Forward pass: hidden → action logits
        
        In production: Actual linear layer with learned weights
        For now: Identity mapping (placeholder)
        """
        var logits = List[Float32]()
        for i in range(self.num_actions):
            logits.append(0.0)  # Placeholder
        return logits


struct ValueHead:
    """
    Value head: maps transformer hidden states to value estimate
    
    Architecture: Linear(d_model → 1)
    """
    var input_dim: Int
    
    fn __init__(inout self, input_dim: Int):
        self.input_dim = input_dim
    
    fn forward(self, hidden: List[Float32]) -> Float32:
        """
        Forward pass: hidden → value estimate
        
        In production: Actual linear layer
        For now: Return 0.0 (placeholder)
        """
        return 0.0


# ============================================================================
# Utility Functions
# ============================================================================

fn exp(x: Float32) -> Float32:
    """
    Exponential function using Taylor series

    e^x = 1 + x + x^2/2! + x^3/3! + ...
    """
    # Handle overflow/underflow
    if x > 88.0:
        return 3.4028235e38  # Float32 max
    if x < -88.0:
        return 0.0

    # Taylor series for e^x
    var result: Float32 = 1.0
    var term: Float32 = 1.0

    for n in range(1, 25):  # 25 terms for good precision
        term *= x / Float32(n)
        result += term
        if _abs_f32(term) < 1e-10:
            break

    return result


fn log(x: Float32) -> Float32:
    """
    Natural logarithm using Newton-Raphson method

    Iteratively solves e^y = x for y
    """
    if x <= 0:
        return -1e9  # Undefined for non-positive values

    if x == 1.0:
        return 0.0

    # Initial guess using simple approximation
    var y: Float32 = 0.0
    if x > 1.0:
        y = x - 1.0  # Good for x near 1
    else:
        y = -(1.0 / x - 1.0)

    # Normalize for better convergence
    var mantissa = x
    var exponent: Int = 0

    while mantissa >= 2.0:
        mantissa /= 2.0
        exponent += 1

    while mantissa < 1.0:
        mantissa *= 2.0
        exponent -= 1

    # Compute ln(mantissa) using Taylor series for ln(1+t) where t = mantissa - 1
    let t = mantissa - 1.0
    var result: Float32 = 0.0
    var term = t
    var sign: Float32 = 1.0

    for n in range(1, 15):
        result += sign * term / Float32(n)
        term *= t
        sign = -sign

    # Add exponent contribution: ln(2) ≈ 0.693147
    let ln2: Float32 = 0.6931471805599453
    result += Float32(exponent) * ln2

    return result


fn _abs_f32(x: Float32) -> Float32:
    """Absolute value helper for Float32"""
    return x if x >= 0.0 else -x


fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers"""
    return a if a < b else b


fn min(a: Float32, b: Float32) -> Float32:
    """Minimum of two floats"""
    return a if a < b else b


fn max(a: Float32, b: Float32) -> Float32:
    """Maximum of two floats"""
    return a if a > b else b


fn max(a: Int, b: Int) -> Int:
    """Maximum of two integers"""
    return a if a > b else b


fn sqrt(x: Float32) -> Float32:
    """
    Square root using Newton-Raphson method

    Converges quadratically: x_{n+1} = (x_n + a/x_n) / 2
    """
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return 0.0

    var guess = x / 2.0
    if x < 1.0:
        guess = 1.0

    # Newton-Raphson iterations
    for _ in range(10):
        let new_guess = (guess + x / guess) / 2.0
        if _abs_f32(new_guess - guess) < 1e-7:
            break
        guess = new_guess

    return guess


fn clamp(x: Float32, min_val: Float32, max_val: Float32) -> Float32:
    """Clamp value to range [min_val, max_val]"""
    if x < min_val:
        return min_val
    if x > max_val:
        return max_val
    return x
