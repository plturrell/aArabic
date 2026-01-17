"""
Value Network and Advantage Estimation
Estimates state values and advantages for KTO learning

Features:
- Value network for V(s) estimation
- Advantage calculation A(s,a) = Q(s,a) - V(s)
- Temporal difference learning
- Bootstrapping for credit assignment
"""

from collections import List
from .kto_policy import KTOPolicy, PolicyOutput, ToolAction
from ..state import OrchestrationState
from .experience_buffer import Experience


# ============================================================================
# Value Network
# ============================================================================

struct ValueNetwork:
    """
    Estimates state value V(s) for advantage calculation
    
    Architecture:
    - Shares transformer backbone with policy (parameter efficient)
    - Separate value head
    - Trained with TD(λ) or Monte Carlo returns
    
    Used for:
    - Advantage estimation: A(s,a) = Q(s,a) - V(s)
    - Bootstrapping in KTO
    - Credit assignment
    """
    var policy: KTOPolicy  # Shares transformer backbone
    var learning_rate: Float32
    var gamma: Float32  # Discount factor
    var lambda_gae: Float32  # GAE lambda for advantage estimation
    
    fn __init__(
        inout self,
        policy: KTOPolicy,
        learning_rate: Float32 = 0.001,
        gamma: Float32 = 0.99,
        lambda_gae: Float32 = 0.95
    ):
        """
        Initialize value network
        
        Args:
            policy: KTO policy (shares transformer)
            learning_rate: Learning rate for value updates
            gamma: Discount factor for future rewards
            lambda_gae: GAE lambda for advantage smoothing
        """
        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
    
    fn estimate_value(self, state: OrchestrationState) -> Float32:
        """
        Estimate value of state: V(s)
        
        Args:
            state: Current orchestration state
            
        Returns:
            Estimated value (expected cumulative reward)
        """
        let output = self.policy.forward(state)
        return output.value_estimate
    
    fn estimate_values_batch(
        self,
        states: List[OrchestrationState]
    ) -> List[Float32]:
        """Estimate values for batch of states"""
        var values = List[Float32]()
        
        for i in range(len(states)):
            values.append(self.estimate_value(states[i]))
        
        return values
    
    fn compute_td_error(
        self,
        state: OrchestrationState,
        reward: Float32,
        next_state: OrchestrationState,
        done: Bool
    ) -> Float32:
        """
        Compute temporal difference error: δ = r + γV(s') - V(s)
        
        Used for TD learning and advantage estimation
        
        Args:
            state: Current state
            reward: Immediate reward
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            TD error
        """
        let v_current = self.estimate_value(state)
        let v_next = 0.0 if done else self.estimate_value(next_state)
        
        let td_error = reward + self.gamma * v_next - v_current
        return td_error
    
    fn update_value(
        inout self,
        state: OrchestrationState,
        target_value: Float32
    ):
        """
        Update value network towards target
        
        Loss: MSE(V(s), target)
        Update: V ← V - α * ∇(V(s) - target)²
        
        Args:
            state: State to update
            target_value: Target value (from returns or bootstrapping)
        """
        let predicted_value = self.estimate_value(state)
        let error = predicted_value - target_value
        
        # In production: backprop through value head
        # For now: conceptual placeholder
        # self.policy.value_head.backward(error * self.learning_rate)


# ============================================================================
# Advantage Estimator
# ============================================================================

struct AdvantageEstimator:
    """
    Compute advantages using Generalized Advantage Estimation (GAE)
    
    GAE(λ):
    A^GAE(s,a) = Σ (γλ)^t * δ_t
    
    Where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Benefits:
    - Balances bias-variance tradeoff
    - Smoother advantage estimates
    - Better credit assignment
    """
    var value_network: ValueNetwork
    var gamma: Float32  # Discount factor
    var lambda_gae: Float32  # GAE smoothing parameter
    
    fn __init__(
        inout self,
        value_network: ValueNetwork,
        gamma: Float32 = 0.99,
        lambda_gae: Float32 = 0.95
    ):
        """
        Initialize advantage estimator
        
        Args:
            value_network: Value network for V(s) estimation
            gamma: Discount factor
            lambda_gae: GAE lambda (1.0 = Monte Carlo, 0.0 = TD(0))
        """
        self.value_network = value_network
        self.gamma = gamma
        self.lambda_gae = lambda_gae
    
    fn compute_advantages(
        self,
        experiences: List[Experience]
    ) -> List[Float32]:
        """
        Compute GAE advantages for trajectory of experiences
        
        Args:
            experiences: Trajectory of experiences (ordered)
            
        Returns:
            List of advantage estimates A(s,a) for each experience
        """
        var advantages = List[Float32]()
        let n = len(experiences)
        
        if n == 0:
            return advantages
        
        # Compute TD errors for all steps
        var td_errors = List[Float32]()
        for i in range(n):
            let exp = experiences[i]
            let td_error = self.value_network.compute_td_error(
                exp.state,
                exp.reward,
                exp.next_state,
                exp.done
            )
            td_errors.append(td_error)
        
        # Compute GAE advantages backwards
        var running_advantage: Float32 = 0.0
        
        for i in range(n - 1, -1, -1):  # Backward iteration
            let td_error = td_errors[i]
            
            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            running_advantage = td_error + self.gamma * self.lambda_gae * running_advantage
            
            # Prepend to list (we're going backwards)
            advantages.append(running_advantage)
        
        # Reverse to match original order
        return self._reverse_list(advantages)
    
    fn compute_advantage(
        self,
        state: OrchestrationState,
        action: ToolAction,
        reward: Float32,
        next_state: OrchestrationState
    ) -> Float32:
        """
        Compute single-step advantage (simplified TD advantage)
        
        A(s,a) = r + γV(s') - V(s)
        
        Args:
            state: Current state
            action: Action taken (unused in value-based advantage)
            reward: Reward received
            next_state: Next state
            
        Returns:
            Advantage estimate
        """
        let v_current = self.value_network.estimate_value(state)
        let v_next = self.value_network.estimate_value(next_state)
        
        return reward + self.gamma * v_next - v_current
    
    fn _reverse_list(self, lst: List[Float32]) -> List[Float32]:
        """Reverse a list (helper method)"""
        var reversed = List[Float32]()
        for i in range(len(lst) - 1, -1, -1):
            reversed.append(lst[i])
        return reversed


# ============================================================================
# Return Computation
# ============================================================================

fn compute_returns(
    experiences: List[Experience],
    gamma: Float32 = 0.99
) -> List[Float32]:
    """
    Compute discounted returns for trajectory
    
    G_t = Σ γ^k * r_{t+k}
    
    Args:
        experiences: Trajectory of experiences
        gamma: Discount factor
        
    Returns:
        List of discounted returns for each step
    """
    var returns = List[Float32]()
    let n = len(experiences)
    
    if n == 0:
        return returns
    
    # Compute returns backwards (more efficient)
    var running_return: Float32 = 0.0
    
    for i in range(n - 1, -1, -1):
        let reward = experiences[i].reward
        running_return = reward + gamma * running_return
        returns.append(running_return)
    
    # Reverse to match original order
    return reverse_list(returns)


fn compute_monte_carlo_returns(
    experiences: List[Experience]
) -> List[Float32]:
    """
    Compute undiscounted Monte Carlo returns (gamma = 1.0)
    
    Useful for episodic tasks where all rewards are equally important
    """
    return compute_returns(experiences, gamma=1.0)


fn reverse_list(lst: List[Float32]) -> List[Float32]:
    """Reverse a list of Float32"""
    var reversed = List[Float32]()
    for i in range(len(lst) - 1, -1, -1):
        reversed.append(lst[i])
    return reversed


# ============================================================================
# Advantage Normalization
# ============================================================================

fn normalize_advantages(advantages: List[Float32]) -> List[Float32]:
    """
    Normalize advantages to zero mean and unit variance
    
    Improves training stability
    
    A_normalized = (A - mean(A)) / (std(A) + ε)
    """
    let n = len(advantages)
    if n == 0:
        return advantages
    
    # Compute mean
    var sum_adv: Float32 = 0.0
    for i in range(n):
        sum_adv += advantages[i]
    let mean = sum_adv / Float32(n)
    
    # Compute std dev
    var sum_sq_diff: Float32 = 0.0
    for i in range(n):
        let diff = advantages[i] - mean
        sum_sq_diff += diff * diff
    let std = sqrt(sum_sq_diff / Float32(n)) + 1e-8  # Add epsilon for stability
    
    # Normalize
    var normalized = List[Float32]()
    for i in range(n):
        normalized.append((advantages[i] - mean) / std)
    
    return normalized


fn sqrt(x: Float32) -> Float32:
    """Square root (placeholder)"""
    # In production: use actual math.sqrt
    return x ** 0.5 if x > 0 else 0.0
