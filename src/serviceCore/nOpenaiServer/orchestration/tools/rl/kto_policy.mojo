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
"""

from collections import Dict, List
from ..state import OrchestrationState, StateEncoder
from ..execution import ExecutionStrategy
from ..registry import ToolRegistry


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
    
    fn __init__(
        inout self,
        registry: ToolRegistry,
        d_model: Int = 256,
        n_heads: Int = 8,
        n_layers: Int = 6
    ):
        """
        Initialize KTO policy network
        
        Args:
            registry: Tool registry for action space
            d_model: Transformer embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
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
        """Compute log probability of action"""
        # Simplified - would map action to index and get log prob
        return 0.0  # Placeholder
    
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
        """Map action index to tool name"""
        # Simplified - would use registry.list_all_tools()[idx]
        return "tool_" + String(idx)


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
    """Exponential function (placeholder - use actual math.exp)"""
    # Simplified - actual implementation would use proper exp
    return 1.0 + x

fn log(x: Float32) -> Float32:
    """Natural logarithm (placeholder - use actual math.log)"""
    # Simplified - actual implementation would use proper log
    return x - 1.0

fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers"""
    return a if a < b else b

fn max(a: Float32, b: Float32) -> Float32:
    """Maximum of two floats"""
    return a if a > b else b
