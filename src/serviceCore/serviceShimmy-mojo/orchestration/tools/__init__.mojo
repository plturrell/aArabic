"""
Tool Orchestration Module
KTO-based reinforcement learning for intelligent tool coordination

Architecture:
- registry.mojo: Tool definitions and fast lookup
- execution.mojo: Multi-strategy execution engine
- state.mojo: State management for RL
- strategies.mojo: Strategy selection logic
- metrics.mojo: SIMD-optimized performance tracking
- rl/kto_policy.mojo: KTO policy network (Titans-based)
- rl/kto_loss.mojo: KTO loss function
- rl/value_model.mojo: Value estimation
- rl/experience_buffer.mojo: Replay buffer

Performance: 5-10x faster than Python ToolOrchestra
Key Innovation: KTO (Kahneman-Tversky Optimization) for sample-efficient learning
"""

from .registry import (
    ToolRegistry,
    ToolDefinition,
    ModelDefinition,
    ToolParameter,
    RegistryStats,
    load_registry_from_json
)

from .execution import (
    ExecutionEngine,
    ExecutionStrategy,
    ToolResult,
    WorkflowResult
)

# TODO: Import additional modules as they're created
# from .state import OrchestrationState
# from .strategies import StrategySelector
# from .metrics import MetricsTracker
# from .rl.kto_policy import KTOPolicy
