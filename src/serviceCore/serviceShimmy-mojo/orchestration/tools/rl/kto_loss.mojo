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
        
        In production: Map action to index and get log prob
        For now: Placeholder
        """
        # Simplified - would map action.tool_name to index
        # and return log(output.action_probs[index])
        return 0.0  # Placeholder
    
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
    """Natural logarithm (placeholder)"""
    # In production: use actual math.log
    return x - 1.0 if x > 0 else -1e9


fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers"""
    return a if a < b else b


fn min(a: Float32, b: Float32) -> Float32:
    """Minimum of two floats"""
    return a if a < b else b


fn max(a: Float32, b: Float32) -> Float32:
    """Maximum of two floats"""
    return a if a > b else b
