"""
Orchestration State Management
Manages workflow state for RL-based decision making

Features:
- State encoding for neural networks
- History tracking
- Context management
- Serialization for checkpoints
"""

from collections import Dict, List
from time import now
from .execution import ToolResult, WorkflowResult
from .metrics import ToolMetrics


# ============================================================================
# Core State Types
# ============================================================================

@value
struct OrchestrationState:
    """
    Complete state of workflow orchestration
    Used by KTO policy for decision making
    """
    var workflow_id: String
    var current_step: Int
    var total_steps: Int
    var completed_tools: List[String]
    var pending_tools: List[String]
    var failed_tools: List[String]
    var context: Dict[String, String]  # Workflow context data
    var metrics: Dict[String, ToolMetrics]
    var timestamp: Float64
    var success_rate: Float32  # Running success rate

    fn __init__(inout self):
        """Default constructor - creates empty state"""
        self.workflow_id = ""
        self.current_step = 0
        self.total_steps = 0
        self.completed_tools = List[String]()
        self.pending_tools = List[String]()
        self.failed_tools = List[String]()
        self.context = Dict[String, String]()
        self.metrics = Dict[String, ToolMetrics]()
        self.timestamp = now()
        self.success_rate = 1.0

    fn __init__(
        inout self,
        workflow_id: String,
        total_steps: Int = 0
    ):
        self.workflow_id = workflow_id
        self.current_step = 0
        self.total_steps = total_steps
        self.completed_tools = List[String]()
        self.pending_tools = List[String]()
        self.failed_tools = List[String]()
        self.context = Dict[String, String]()
        self.metrics = Dict[String, ToolMetrics]()
        self.timestamp = now()
        self.success_rate = 1.0
    
    fn add_completed_tool(inout self, tool_name: String):
        """Mark tool as completed"""
        self.completed_tools.append(tool_name)
        self.current_step += 1
        self._update_success_rate()
    
    fn add_failed_tool(inout self, tool_name: String):
        """Mark tool as failed"""
        self.failed_tools.append(tool_name)
        self.current_step += 1
        self._update_success_rate()
    
    fn add_pending_tool(inout self, tool_name: String):
        """Add tool to pending queue"""
        self.pending_tools.append(tool_name)
    
    fn update_context(inout self, key: String, value: String):
        """Update workflow context"""
        self.context[key] = value

    fn set_context(inout self, key: String, value: String):
        """Set a context value (alias for update_context)"""
        self.context[key] = value

    fn get_context(self, key: String, default: String = "") -> String:
        """Get a context value with optional default"""
        if key in self.context:
            return self.context[key]
        return default

    fn mark_tool_completed(inout self, tool_name: String, success: Bool):
        """
        Mark a tool as completed with success/failure status

        Args:
            tool_name: Name of the tool
            success: Whether the tool execution succeeded
        """
        # Remove from pending if present
        var new_pending = List[String]()
        for i in range(len(self.pending_tools)):
            if self.pending_tools[i] != tool_name:
                new_pending.append(self.pending_tools[i])
        self.pending_tools = new_pending

        # Add to appropriate list
        if success:
            self.add_completed_tool(tool_name)
        else:
            self.add_failed_tool(tool_name)

    fn remove_pending_tool(inout self, tool_name: String):
        """Remove a tool from the pending queue"""
        var new_pending = List[String]()
        for i in range(len(self.pending_tools)):
            if self.pending_tools[i] != tool_name:
                new_pending.append(self.pending_tools[i])
        self.pending_tools = new_pending

    fn get_pending_count(self) -> Int:
        """Get number of pending tools"""
        return len(self.pending_tools)

    fn get_completed_count(self) -> Int:
        """Get number of completed tools"""
        return len(self.completed_tools)

    fn get_failed_count(self) -> Int:
        """Get number of failed tools"""
        return len(self.failed_tools)
    
    fn update_metrics(inout self, tool_name: String, metrics: ToolMetrics):
        """Update metrics for a tool"""
        self.metrics[tool_name] = metrics
    
    fn _update_success_rate(inout self):
        """Recalculate success rate"""
        let total = len(self.completed_tools) + len(self.failed_tools)
        if total > 0:
            self.success_rate = Float32(len(self.completed_tools)) / Float32(total)
    
    fn get_progress(self) -> Float32:
        """Get workflow completion progress (0.0 to 1.0)"""
        if self.total_steps == 0:
            return 0.0
        return Float32(self.current_step) / Float32(self.total_steps)
    
    fn is_complete(self) -> Bool:
        """Check if workflow is complete"""
        return self.current_step >= self.total_steps
    
    fn has_failures(self) -> Bool:
        """Check if any tools failed"""
        return len(self.failed_tools) > 0


# ============================================================================
# State Encoder for RL
# ============================================================================

struct StateEncoder:
    """
    Encode orchestration state into vector representation for neural networks
    
    Encoding strategy:
    - One-hot encode tool completion status
    - Normalize metrics (time, cost, success rate)
    - Embed context as learned embeddings
    - Position encoding for workflow step
    
    Output: Fixed-size vector suitable for transformer input
    """
    var embedding_dim: Int
    var max_tools: Int
    var feature_dim: Int
    
    fn __init__(
        inout self,
        embedding_dim: Int = 256,
        max_tools: Int = 50
    ):
        """
        Initialize state encoder
        
        Args:
            embedding_dim: Dimension of output embeddings
            max_tools: Maximum number of tools in a workflow
        """
        self.embedding_dim = embedding_dim
        self.max_tools = max_tools
        # Feature vector: [progress, success_rate, n_completed, n_failed, n_pending, ...]
        self.feature_dim = 5 + max_tools * 3  # Base features + tool one-hots
    
    fn encode(self, state: OrchestrationState) -> List[Float32]:
        """
        Encode state into fixed-size feature vector
        
        Args:
            state: Current orchestration state
            
        Returns:
            Feature vector of size feature_dim
            
        Features:
        [0]: Workflow progress (0.0 to 1.0)
        [1]: Success rate (0.0 to 1.0)
        [2]: Number of completed tools (normalized)
        [3]: Number of failed tools (normalized)
        [4]: Number of pending tools (normalized)
        [5:5+max_tools]: One-hot completed tools
        [5+max_tools:5+2*max_tools]: One-hot failed tools
        [5+2*max_tools:5+3*max_tools]: One-hot pending tools
        """
        var features = List[Float32]()
        
        # Base features
        features.append(state.get_progress())
        features.append(state.success_rate)
        features.append(Float32(len(state.completed_tools)) / Float32(self.max_tools))
        features.append(Float32(len(state.failed_tools)) / Float32(self.max_tools))
        features.append(Float32(len(state.pending_tools)) / Float32(self.max_tools))
        
        # One-hot encoding for tool states (simplified)
        # In full implementation, would use proper tool indexing
        for i in range(self.max_tools * 3):
            features.append(0.0)  # Placeholder
        
        return features
    
    fn encode_batch(
        self,
        states: List[OrchestrationState]
    ) -> List[List[Float32]]:
        """Encode batch of states"""
        var batch = List[List[Float32]]()
        for i in range(len(states)):
            batch.append(self.encode(states[i]))
        return batch
    
    fn get_output_dim(self) -> Int:
        """Get dimension of encoded features"""
        return self.feature_dim


# ============================================================================
# State History
# ============================================================================

struct StateHistory:
    """
    Track history of orchestration states for learning and analysis
    
    Features:
    - Efficient circular buffer
    - State transitions
    - Metrics aggregation
    - Checkpoint/restore
    """
    var history: List[OrchestrationState]
    var max_history_size: Int
    var current_index: Int
    
    fn __init__(inout self, max_size: Int = 1000):
        """
        Initialize state history
        
        Args:
            max_size: Maximum number of states to keep in history
        """
        self.history = List[OrchestrationState]()
        self.max_history_size = max_size
        self.current_index = 0
    
    fn add_state(inout self, state: OrchestrationState):
        """
        Add state to history
        
        Uses circular buffer to maintain max_size
        """
        if len(self.history) < self.max_history_size:
            self.history.append(state)
        else:
            # Circular buffer: overwrite oldest
            self.history[self.current_index % self.max_history_size] = state
        
        self.current_index += 1
    
    fn get_recent(self, n: Int) -> List[OrchestrationState]:
        """Get n most recent states"""
        var recent = List[OrchestrationState]()
        let start_idx = max(0, len(self.history) - n)
        
        for i in range(start_idx, len(self.history)):
            recent.append(self.history[i])
        
        return recent
    
    fn get_by_workflow_id(self, workflow_id: String) -> List[OrchestrationState]:
        """Get all states for a specific workflow"""
        var workflow_states = List[OrchestrationState]()
        
        for i in range(len(self.history)):
            if self.history[i].workflow_id == workflow_id:
                workflow_states.append(self.history[i])
        
        return workflow_states
    
    fn get_successful_workflows(self) -> List[String]:
        """Get workflow IDs that completed successfully"""
        var successful = List[String]()
        
        for i in range(len(self.history)):
            let state = self.history[i]
            if state.is_complete() and not state.has_failures():
                successful.append(state.workflow_id)
        
        return successful
    
    fn get_failed_workflows(self) -> List[String]:
        """Get workflow IDs that had failures"""
        var failed = List[String]()
        
        for i in range(len(self.history)):
            let state = self.history[i]
            if state.has_failures():
                failed.append(state.workflow_id)
        
        return failed
    
    fn clear(inout self):
        """Clear history"""
        self.history = List[OrchestrationState]()
        self.current_index = 0
    
    fn size(self) -> Int:
        """Get number of states in history"""
        return len(self.history)


# ============================================================================
# State Transition
# ============================================================================

@value
struct StateTransition:
    """
    Represents a state transition for RL learning
    
    (state, action, next_state, reward, done)
    """
    var state: OrchestrationState
    var action: String  # Tool name that was executed
    var next_state: OrchestrationState
    var reward: Float32
    var done: Bool  # Whether workflow completed
    var success: Bool  # Whether action succeeded
    
    fn __init__(
        inout self,
        state: OrchestrationState,
        action: String,
        next_state: OrchestrationState,
        reward: Float32 = 0.0,
        done: Bool = False,
        success: Bool = True
    ):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.success = success


# ============================================================================
# Utility Functions
# ============================================================================

fn create_state_from_workflow_result(
    workflow_id: String,
    result: WorkflowResult
) -> OrchestrationState:
    """
    Create final state from workflow execution result
    
    Args:
        workflow_id: Workflow identifier
        result: Workflow execution result
        
    Returns:
        OrchestrationState representing final state
    """
    var state = OrchestrationState(
        workflow_id=workflow_id,
        total_steps=len(result.tool_results)
    )
    
    # Populate from results
    for i in range(len(result.tool_results)):
        let tool_result = result.tool_results[i]
        
        if tool_result.success:
            state.add_completed_tool(tool_result.tool_name)
        else:
            state.add_failed_tool(tool_result.tool_name)
    
    return state
