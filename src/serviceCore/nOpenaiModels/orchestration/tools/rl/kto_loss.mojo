"""
KTO Loss Function Implementation
Kahneman-Tversky Optimization loss for human-aligned learning

Mathematical Formulation:
L_KTO = E_desirable[KL(π_θ || π_ref) - λ * log π_θ(a|s)] +
        E_undesirable[KL(π_θ || π_ref) - (1/λ) * log π_θ(a|s)]

Where:
- π_θ: Current policy
- π_ref: Reference policy (EMA)
- λ = 2.25: Loss aversion coefficient from psychology
- desirable: Successful tool executions
- undesirable: Failed tool executions

Key Insight: Losses hurt more than gains feel good (asymmetric utility)
"""

from collections import List
from .kto_policy import KTOPolicy, PolicyOutput, ToolAction
from ..state import OrchestrationState
from .experience_buffer import Experience, BalancedBatch


# ============================================================================
# KTO Loss Components
# ============================================================================

struct KTOLoss:
    """
    Complete KTO loss function with all components
    
    Components:
    1. Desirable loss (gain utility)
    2. Undesirable loss (loss utility with asymmetry)
    3. KL divergence regularization
    4. Reference policy management
    """
    var lambda_loss_aversion: Float32
    var kl_weight: Float32  # Weight for KL regularization
    var clip_epsilon: Float32  # For policy clipping (optional)
    
    fn __init__(
        inout self,
        lambda_loss_aversion: Float32 = 2.25,
        kl_weight: Float32 = 1.0,
        clip_epsilon: Float32 = 0.2
    ):
        """
        Initialize KTO loss function
        
        Args:
            lambda_loss_aversion: Loss aversion coefficient (standard: 2.25)
            kl_weight: Weight for KL divergence term
            clip_epsilon: Clipping epsilon for policy ratio (optional)
        """
        self.lambda_loss_aversion = lambda_loss_aversion
        self.kl_weight = kl_weight
        self.clip_epsilon = clip_epsilon
    
    
    # ========================================================================
    # Main Loss Computation
    # ========================================================================
    
    fn compute_loss(
        self,
        policy: KTOPolicy,
        reference_policy: KTOPolicy,
        batch: BalancedBatch
    ) -> KTOLossOutput:
        """
        Compute complete KTO loss for a balanced batch
        
        Args:
            policy: Current policy π_θ
            reference_policy: Reference policy π_ref (EMA)
            batch: Balanced batch of desirable/undesirable experiences
            
        Returns:
            KTOLossOutput with total loss and component losses
        """
        # Compute desirable loss (successful executions)
        let desirable_loss = self.compute_desirable_loss(
            policy,
            reference_policy,
            batch.desirable
        )
        
        # Compute undesirable loss (failed executions)
        let undesirable_loss = self.compute_undesirable_loss(
            policy,
            reference_policy,
            batch.undesirable
        )
        
        # Total KTO loss
        let total_loss = desirable_loss + undesirable_loss
        
        return KTOLossOutput(
            total_loss=total_loss,
            desirable_loss=desirable_loss,
            undesirable_loss=undesirable_loss,
            n_desirable=len(batch.desirable),
            n_undesirable=len(batch.undesirable)
        )
    
    
    # ========================================================================
    # Desirable Loss (Gain Utility)
    # ========================================================================
    
    fn compute_desirable_loss(
        self,
        policy: KTOPolicy,
        reference_policy: KTOPolicy,
        desirable_experiences: List[Experience]
    ) -> Float32:
        """
        Compute loss for desirable experiences (successful executions)
        
        L_desirable = E[KL(π_θ || π_ref) - λ * log π_θ(a|s)]
        
        Standard weighting (λ) for gains
        
        Args:
            policy: Current policy
            reference_policy: Reference policy for KL
            desirable_experiences: List of successful experiences
            
        Returns:
            Average desirable loss
        """
        var total_loss: Float32 = 0.0
        let n = len(desirable_experiences)
        
        if n == 0:
            return 0.0
        
        for i in range(n):
            let exp = desirable_experiences[i]
            
            # Get policy outputs
            let output = policy.forward(exp.state)
            let ref_output = reference_policy.forward(exp.state)
            
            # Compute KL divergence: KL(π_θ || π_ref)
            let kl_div = self._compute_kl_divergence(
                output.action_probs,
                ref_output.action_probs
            )
            
            # Compute log probability: log π_θ(a|s)
            let log_prob = self._compute_log_prob_for_action(
                output,
                exp.action
            )
            
            # Desirable loss: KL - λ * log π
            # Higher log prob → lower loss (encourage successful actions)
            let loss = self.kl_weight * kl_div - self.lambda_loss_aversion * log_prob
            total_loss += loss
        
        return total_loss / Float32(n)
    
    
    # ========================================================================
    # Undesirable Loss (Loss Utility)
    # ========================================================================
    
    fn compute_undesirable_loss(
        self,
        policy: KTOPolicy,
        reference_policy: KTOPolicy,
        undesirable_experiences: List[Experience]
    ) -> Float32:
        """
        Compute loss for undesirable experiences (failed executions)
        
        L_undesirable = E[KL(π_θ || π_ref) - (1/λ) * log π_θ(a|s)]
        
        Asymmetric weighting (1/λ) for losses - models loss aversion
        Failed actions penalized more strongly than successful actions rewarded
        
        Args:
            policy: Current policy
            reference_policy: Reference policy for KL
            undesirable_experiences: List of failed experiences
            
        Returns:
            Average undesirable loss
        """
        var total_loss: Float32 = 0.0
        let n = len(undesirable_experiences)
        
        if n == 0:
            return 0.0
        
        for i in range(n):
            let exp = undesirable_experiences[i]
            
            # Get policy outputs
            let output = policy.forward(exp.state)
            let ref_output = reference_policy.forward(exp.state)
            
            # Compute KL divergence
            let kl_div = self._compute_kl_divergence(
                output.action_probs,
                ref_output.action_probs
            )
            
            # Compute log probability
            let log_prob = self._compute_log_prob_for_action(
                output,
                exp.action
            )
            
            # Undesirable loss: KL - (1/λ) * log π
            # Asymmetric: failed actions penalized with 1/λ ≈ 0.44 weight
            # This is LESS than desirable weight (λ = 2.25)
            # → Policy learns to avoid failures more strongly
            let loss = self.kl_weight * kl_div - (1.0 / self.lambda_loss_aversion) * log_prob
            total_loss += loss
        
        return total_loss / Float32(n)
    
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    fn _compute_kl_divergence(
        self,
        p: List[Float32],
        q: List[Float32]
    ) -> Float32:
        """
        Compute KL divergence: KL(p || q) = Σ p(x) * log(p(x) / q(x))
        
        Regularizes policy to not deviate too far from reference
        """
        var kl: Float32 = 0.0
        let n = min(len(p), len(q))
        
        for i in range(n):
            if p[i] > 1e-10 and q[i] > 1e-10:  # Numerical stability
                kl += p[i] * log(p[i] / q[i])
        
        return kl
    
    fn _compute_log_prob_for_action(
        self,
        output: PolicyOutput,
        action: ToolAction
    ) -> Float32:
        """
        Compute log probability of action: log π_θ(a|s)

        Maps action to index in the action probability distribution
        and returns the log probability.

        Args:
            output: Policy output containing action probabilities
            action: The action taken

        Returns:
            Log probability of the action
        """
        # Map tool name to action index
        let action_idx = self._tool_name_to_index(action.tool_name, output.num_actions)

        # Bounds check
        if action_idx < 0 or action_idx >= len(output.action_probs):
            return -1e9  # Very low log prob for invalid actions

        # Get probability and compute log
        let prob = output.action_probs[action_idx]

        # Numerical stability: clamp probability to avoid log(0)
        let clamped_prob = max(prob, 1e-10)

        return log(clamped_prob)

    fn _tool_name_to_index(self, tool_name: String, num_actions: Int) -> Int:
        """
        Map tool name to action index

        Uses a simple hash-based mapping for deterministic indexing.
        In production, this would use the tool registry for proper mapping.

        Args:
            tool_name: Name of the tool
            num_actions: Total number of actions in the policy

        Returns:
            Action index in [0, num_actions)
        """
        if num_actions <= 0:
            return 0

        # Simple hash function for tool name
        var hash_val: UInt64 = 0
        let bytes = tool_name.as_bytes()
        for i in range(len(bytes)):
            hash_val = hash_val * 31 + UInt64(bytes[i])

        # Map to action index
        return Int(hash_val % UInt64(num_actions))

    fn _compute_action_entropy(self, probs: List[Float32]) -> Float32:
        """
        Compute entropy of action distribution: H(π) = -Σ π(a) log π(a)

        Higher entropy = more exploration
        Lower entropy = more exploitation

        Args:
            probs: Action probability distribution

        Returns:
            Entropy value
        """
        var entropy: Float32 = 0.0

        for i in range(len(probs)):
            if probs[i] > 1e-10:
                entropy -= probs[i] * log(probs[i])

        return entropy
    
    fn _clip_policy_ratio(self, ratio: Float32) -> Float32:
        """
        Clip policy ratio for stability (optional, like PPO)
        
        Keeps ratio in [1-ε, 1+ε] range
        """
        return max(
            1.0 - self.clip_epsilon,
            min(1.0 + self.clip_epsilon, ratio)
        )


# ============================================================================
# Loss Output
# ============================================================================

@value
struct KTOLossOutput:
    """
    Complete KTO loss with component breakdown
    
    Used for monitoring and debugging
    """
    var total_loss: Float32
    var desirable_loss: Float32
    var undesirable_loss: Float32
    var n_desirable: Int
    var n_undesirable: Int
    
    fn __init__(
        inout self,
        total_loss: Float32 = 0.0,
        desirable_loss: Float32 = 0.0,
        undesirable_loss: Float32 = 0.0,
        n_desirable: Int = 0,
        n_undesirable: Int = 0
    ):
        self.total_loss = total_loss
        self.desirable_loss = desirable_loss
        self.undesirable_loss = undesirable_loss
        self.n_desirable = n_desirable
        self.n_undesirable = n_undesirable
    
    fn is_balanced(self) -> Bool:
        """Check if batch was balanced"""
        return self.n_desirable == self.n_undesirable
    
    fn get_balance_ratio(self) -> Float32:
        """Get ratio of desirable to undesirable samples"""
        if self.n_undesirable == 0:
            return Float32.MAX if self.n_desirable > 0 else 1.0
        return Float32(self.n_desirable) / Float32(self.n_undesirable)


# ============================================================================
# Utility Functions
# ============================================================================

fn compute_kto_loss_simple(
    policy: KTOPolicy,
    reference_policy: KTOPolicy,
    batch: BalancedBatch,
    lambda_loss_aversion: Float32 = 2.25
) -> Float32:
    """
    Simplified KTO loss computation (convenience function)
    
    Args:
        policy: Current policy
        reference_policy: Reference policy
        batch: Balanced batch of experiences
        lambda_loss_aversion: Loss aversion coefficient
        
    Returns:
        Total KTO loss
    """
    let loss_fn = KTOLoss(lambda_loss_aversion=lambda_loss_aversion)
    let output = loss_fn.compute_loss(policy, reference_policy, batch)
    return output.total_loss


fn log(x: Float32) -> Float32:
    """
    Natural logarithm using Taylor series approximation

    For x > 0, computes ln(x) using:
    ln(x) = ln(m * 2^e) = ln(m) + e*ln(2)

    Where m is the mantissa in [1, 2) and e is the exponent.
    Uses Taylor series for ln(1+y) where y = m - 1.
    """
    if x <= 0:
        return -1e9  # Undefined for non-positive values

    # Handle special cases
    if x == 1.0:
        return 0.0

    # Normalize to [1, 2) range
    var mantissa = x
    var exponent: Int = 0

    # Scale down if > 2
    while mantissa >= 2.0:
        mantissa /= 2.0
        exponent += 1

    # Scale up if < 1
    while mantissa < 1.0:
        mantissa *= 2.0
        exponent -= 1

    # Now mantissa is in [1, 2)
    # Compute ln(mantissa) using Taylor series for ln(1+y) where y = mantissa - 1
    let y = mantissa - 1.0

    # Taylor series: ln(1+y) = y - y^2/2 + y^3/3 - y^4/4 + ...
    var result: Float32 = 0.0
    var term = y
    var sign: Float32 = 1.0

    for n in range(1, 20):  # 20 terms for good precision
        result += sign * term / Float32(n)
        term *= y
        sign = -sign

    # Add exponent contribution: ln(2) ≈ 0.693147
    let ln2: Float32 = 0.6931471805599453
    result += Float32(exponent) * ln2

    return result


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

    for n in range(1, 30):  # 30 terms for good precision
        term *= x / Float32(n)
        result += term
        if abs(term) < 1e-10:
            break

    return result


fn sqrt(x: Float32) -> Float32:
    """
    Square root using Newton-Raphson method

    Iteratively computes: x_{n+1} = (x_n + a/x_n) / 2
    """
    if x < 0.0:
        return 0.0  # NaN equivalent for negative input
    if x == 0.0:
        return 0.0

    # Initial guess using bit manipulation approximation
    var guess = x / 2.0
    if x > 1.0:
        guess = x / 2.0
    else:
        guess = 1.0

    # Newton-Raphson iterations
    for _ in range(10):  # 10 iterations for Float32 precision
        let new_guess = (guess + x / guess) / 2.0
        if abs(new_guess - guess) < 1e-7:
            break
        guess = new_guess

    return guess


fn abs(x: Float32) -> Float32:
    """Absolute value of a float"""
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


fn pow(base: Float32, exponent: Float32) -> Float32:
    """
    Power function: base^exponent

    Uses: x^y = e^(y * ln(x))
    """
    if base <= 0.0:
        if exponent == 0.0:
            return 1.0
        return 0.0

    return exp(exponent * log(base))


fn sigmoid(x: Float32) -> Float32:
    """
    Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
    """
    if x >= 0:
        let ez = exp(-x)
        return 1.0 / (1.0 + ez)
    else:
        let ez = exp(x)
        return ez / (1.0 + ez)


fn tanh(x: Float32) -> Float32:
    """
    Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """
    let ep = exp(x)
    let em = exp(-x)
    return (ep - em) / (ep + em)
