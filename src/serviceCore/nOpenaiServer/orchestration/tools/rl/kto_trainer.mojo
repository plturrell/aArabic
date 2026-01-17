"""
KTO Trainer - Online Learning for Tool Orchestration
Manages training loop, reference policy updates, and online learning

Features:
- Online KTO training from workflow executions
- Reference policy management (EMA)
- Gradient updates (placeholder for autodiff)
- Training metrics and monitoring
"""

from collections import List
from .kto_policy import KTOPolicy, ToolAction
from .kto_loss import KTOLoss, KTOLossOutput
from .experience_buffer import ExperienceBuffer, Experience, BalancedBatch
from .value_model import ValueNetwork, AdvantageEstimator
from ..state import OrchestrationState
from ..execution import ToolResult


# ============================================================================
# KTO Trainer
# ============================================================================

struct KTOTrainer:
    """
    Complete KTO training system for tool orchestration
    
    Components:
    1. Current policy π_θ
    2. Reference policy π_ref (EMA)
    3. Value network V(s)
    4. Experience buffer (desirable/undesirable)
    5. Loss function
    6. Optimizer (placeholder)
    
    Training Flow:
    1. Execute workflow → collect experience
    2. Categorize as desirable/undesirable
    3. Add to experience buffer
    4. Sample balanced batch
    5. Compute KTO loss
    6. Update policy via gradient descent
    7. Update reference policy (EMA)
    """
    var policy: KTOPolicy
    var reference_policy: KTOPolicy
    var value_network: ValueNetwork
    var experience_buffer: ExperienceBuffer
    var loss_fn: KTOLoss
    var advantage_estimator: AdvantageEstimator
    
    # Training hyperparameters
    var learning_rate: Float32
    var batch_size: Int
    var reference_ema_alpha: Float32  # EMA coefficient for reference policy
    var update_frequency: Int  # Steps between policy updates
    var steps_since_update: Int
    var total_training_steps: Int
    
    fn __init__(
        inout self,
        policy: KTOPolicy,
        learning_rate: Float32 = 0.001,
        batch_size: Int = 32,
        reference_ema_alpha: Float32 = 0.99,
        update_frequency: Int = 1
    ):
        """
        Initialize KTO trainer
        
        Args:
            policy: KTO policy to train
            learning_rate: Learning rate for gradient descent
            batch_size: Batch size for training (must be even for balanced sampling)
            reference_ema_alpha: EMA coefficient for reference policy (0.99 = slow update)
            update_frequency: Number of steps between policy updates
        """
        self.policy = policy
        
        # Clone policy for reference (EMA target)
        self.reference_policy = policy  # TODO: Actual cloning
        
        # Value network shares transformer with policy
        self.value_network = ValueNetwork(
            policy=policy,
            learning_rate=learning_rate
        )
        
        # Advantage estimation
        self.advantage_estimator = AdvantageEstimator(
            value_network=self.value_network
        )
        
        # Experience storage
        self.experience_buffer = ExperienceBuffer(
            max_size_per_category=10000
        )
        
        # Loss function
        self.loss_fn = KTOLoss(
            lambda_loss_aversion=2.25,  # Standard KTO
            kl_weight=1.0
        )
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reference_ema_alpha = reference_ema_alpha
        self.update_frequency = update_frequency
        self.steps_since_update = 0
        self.total_training_steps = 0
    
    
    # ========================================================================
    # Online Learning
    # ========================================================================
    
    fn learn_from_execution(
        inout self,
        state: OrchestrationState,
        action: ToolAction,
        result: ToolResult,
        next_state: OrchestrationState
    ):
        """
        Learn from a single tool execution (online learning)
        
        Flow:
        1. Create experience from execution result
        2. Compute reward based on outcome
        3. Add to experience buffer
        4. Train if enough experiences collected
        
        Args:
            state: State before execution
            action: Action taken
            result: Tool execution result
            next_state: State after execution
        """
        # Compute reward from result
        let reward = self._compute_reward(result)
        
        # Create experience
        let experience = Experience(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=next_state.is_complete(),
            success=result.success,
            timestamp=result.timestamp
        )
        
        # Add to buffer
        self.experience_buffer.add(experience)
        
        # Train if we have enough experiences
        self.steps_since_update += 1
        if self.steps_since_update >= self.update_frequency:
            if self.experience_buffer.can_sample(self.batch_size):
                self.train_step()
                self.steps_since_update = 0
    
    fn learn_from_workflow(
        inout self,
        experiences: List[Experience]
    ):
        """
        Learn from complete workflow execution
        
        Batch adds all experiences and trains
        """
        self.experience_buffer.add_batch(experiences)
        
        # Train multiple steps if we have enough data
        let num_possible_batches = self.experience_buffer.size() // self.batch_size
        let num_train_steps = min(num_possible_batches, 10)  # Cap at 10 steps
        
        for _ in range(num_train_steps):
            if self.experience_buffer.can_sample(self.batch_size):
                self.train_step()
    
    
    # ========================================================================
    # Training Step
    # ========================================================================
    
    fn train_step(inout self) -> TrainingMetrics:
        """
        Single KTO training step
        
        Flow:
        1. Sample balanced batch from buffer
        2. Compute KTO loss
        3. Compute gradients (placeholder)
        4. Update policy parameters
        5. Update reference policy (EMA)
        6. Update value network
        
        Returns:
            TrainingMetrics with loss components
        """
        # Sample balanced batch
        let batch = self.experience_buffer.sample_balanced(self.batch_size)
        
        # Compute KTO loss
        let loss_output = self.loss_fn.compute_loss(
            self.policy,
            self.reference_policy,
            batch
        )
        
        # Gradient descent update (placeholder for autodiff)
        # In production: loss_output.total_loss.backward()
        # self.optimizer.step()
        
        # Update reference policy using EMA
        self._update_reference_policy()
        
        # Update value network
        self._update_value_network(batch)
        
        # Track training metrics
        self.total_training_steps += 1
        
        return TrainingMetrics(
            step=self.total_training_steps,
            total_loss=loss_output.total_loss,
            desirable_loss=loss_output.desirable_loss,
            undesirable_loss=loss_output.undesirable_loss,
            buffer_size=self.experience_buffer.size(),
            balance_ratio=self.experience_buffer.get_balance_ratio()
        )
    
    fn _update_reference_policy(inout self):
        """
        Update reference policy using exponential moving average
        
        π_ref ← α * π_ref + (1 - α) * π_θ
        
        Where α = reference_ema_alpha (typically 0.99)
        """
        # In production: EMA parameter update
        # for param_ref, param in zip(ref_policy.parameters(), policy.parameters()):
        #     param_ref = α * param_ref + (1 - α) * param
        
        # Placeholder: Reference policy tracks current policy
        # TODO: Implement actual parameter EMA
        pass
    
    fn _update_value_network(inout self, batch: BalancedBatch):
        """
        Update value network using experiences
        
        Trains value head to predict returns
        """
        # Compute advantages for all experiences
        var all_experiences = List[Experience]()
        
        # Combine desirable and undesirable
        for i in range(len(batch.desirable)):
            all_experiences.append(batch.desirable[i])
        for i in range(len(batch.undesirable)):
            all_experiences.append(batch.undesirable[i])
        
        # Compute advantages (which trains value network)
        let advantages = self.advantage_estimator.compute_advantages(all_experiences)
        
        # Value network updated implicitly through advantage computation
    
    
    # ========================================================================
    # Reward Computation
    # ========================================================================
    
    fn _compute_reward(self, result: ToolResult) -> Float32:
        """
        Compute reward from tool execution result
        
        Components:
        1. Outcome reward: +1 for success, -1 for failure
        2. Efficiency reward: faster/cheaper is better
        3. Quality reward: based on result quality (if available)
        
        Final reward: weighted combination
        """
        # Outcome reward (binary)
        let outcome_reward = 1.0 if result.success else -1.0
        
        # Efficiency reward (normalized)
        let time_factor = max(0.0, 1.0 - result.execution_time / 5.0)  # Normalize to 5s
        let cost_factor = max(0.0, 1.0 - result.cost / 0.1)  # Normalize to $0.10
        let efficiency_reward = (time_factor + cost_factor) / 2.0
        
        # Weighted combination
        let reward = 0.7 * outcome_reward + 0.3 * efficiency_reward
        
        return reward
    
    
    # ========================================================================
    # Training Utilities
    # ========================================================================
    
    fn get_training_stats(self) -> TrainingStatistics:
        """Get comprehensive training statistics"""
        let buffer_stats = self.experience_buffer.get_statistics()
        
        return TrainingStatistics(
            total_steps=self.total_training_steps,
            buffer_size=self.experience_buffer.size(),
            desirable_count=buffer_stats.desirable_count,
            undesirable_count=buffer_stats.undesirable_count,
            balance_ratio=buffer_stats.balance_ratio,
            learning_rate=self.learning_rate
        )
    
    fn should_train(self) -> Bool:
        """Check if trainer should perform training step"""
        return (
            self.experience_buffer.can_sample(self.batch_size) and
            self.steps_since_update >= self.update_frequency
        )


# ============================================================================
# Training Output Types
# ============================================================================

@value
struct TrainingMetrics:
    """Metrics from a single training step"""
    var step: Int
    var total_loss: Float32
    var desirable_loss: Float32
    var undesirable_loss: Float32
    var buffer_size: Int
    var balance_ratio: Float32
    
    fn __init__(
        inout self,
        step: Int = 0,
        total_loss: Float32 = 0.0,
        desirable_loss: Float32 = 0.0,
        undesirable_loss: Float32 = 0.0,
        buffer_size: Int = 0,
        balance_ratio: Float32 = 1.0
    ):
        self.step = step
        self.total_loss = total_loss
        self.desirable_loss = desirable_loss
        self.undesirable_loss = undesirable_loss
        self.buffer_size = buffer_size
        self.balance_ratio = balance_ratio


@value
struct TrainingStatistics:
    """Overall training statistics"""
    var total_steps: Int
    var buffer_size: Int
    var desirable_count: Int
    var undesirable_count: Int
    var balance_ratio: Float32
    var learning_rate: Float32
    
    fn __init__(
        inout self,
        total_steps: Int = 0,
        buffer_size: Int = 0,
        desirable_count: Int = 0,
        undesirable_count: Int = 0,
        balance_ratio: Float32 = 1.0,
        learning_rate: Float32 = 0.001
    ):
        self.total_steps = total_steps
        self.buffer_size = buffer_size
        self.desirable_count = desirable_count
        self.undesirable_count = undesirable_count
        self.balance_ratio = balance_ratio
        self.learning_rate = learning_rate


# ============================================================================
# Utility Functions
# ============================================================================

fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers"""
    return a if a < b else b

fn max(a: Float32, b: Float32) -> Float32:
    """Maximum of two floats"""
    return a if a > b else b
