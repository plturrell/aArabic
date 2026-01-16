"""
Reinforcement Learning Module
KTO-based policy network for intelligent tool orchestration

Components:
- kto_policy.mojo: Core policy network (reuses Titans transformer)
- kto_loss.mojo: KTO loss function implementation  
- experience_buffer.mojo: Replay buffer with desirable/undesirable split
- value_model.mojo: Value estimation for advantage calculation
- kto_trainer.mojo: Complete training system with online learning

Key Innovation: Reuses Titans transformer architecture for policy network
Performance: 10x faster than Python RL implementations

Usage Example:
    from tool_orchestration.rl import KTOTrainer, KTOPolicy, ExperienceBuffer
    
    # Initialize
    let registry = load_registry_from_json("config/toolorchestra_tools.json")
    let policy = KTOPolicy(registry, d_model=256, n_heads=8, n_layers=6)
    let trainer = KTOTrainer(policy, learning_rate=0.001, batch_size=32)
    
    # Online learning from workflow execution
    trainer.learn_from_execution(state, action, result, next_state)
    
    # Select action using trained policy
    let action = policy.select_action(current_state, greedy=False)
"""

from .kto_policy import (
    KTOPolicy,
    ToolAction,
    PolicyOutput,
    ActionHead,
    ValueHead,
    TransformerModel
)

from .kto_loss import (
    KTOLoss,
    KTOLossOutput,
    compute_kto_loss_simple
)

from .experience_buffer import (
    ExperienceBuffer,
    Experience,
    BalancedBatch,
    BufferStatistics
)

from .value_model import (
    ValueNetwork,
    AdvantageEstimator,
    compute_returns,
    compute_monte_carlo_returns,
    normalize_advantages
)

from .kto_trainer import (
    KTOTrainer,
    TrainingMetrics,
    TrainingStatistics
)
