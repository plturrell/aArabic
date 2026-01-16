"""
Strategy Selection and Dependency Analysis
Intelligent selection of execution strategies based on workflow characteristics

Features:
- Dependency analysis
- Resource estimation
- Strategy recommendation
- Performance prediction
"""

from collections import Dict, List
from .registry import ToolRegistry, ToolDefinition
from .state import OrchestrationState
from .metrics import MetricsTracker, ToolMetrics
from .execution import ExecutionStrategy


# ============================================================================
# Dependency Analysis
# ============================================================================

@value
struct ToolDependency:
    """Represents a dependency between tools"""
    var source_tool: String
    var target_tool: String
    var dependency_type: String  # "data", "order", "resource"
    var is_blocking: Bool
    
    fn __init__(
        inout self,
        source_tool: String,
        target_tool: String,
        dependency_type: String = "data",
        is_blocking: Bool = True
    ):
        self.source_tool = source_tool
        self.target_tool = target_tool
        self.dependency_type = dependency_type
        self.is_blocking = is_blocking


struct DependencyAnalyzer:
    """
    Analyze dependencies between tools in a workflow
    
    Features:
    - Build dependency graph
    - Detect cycles
    - Find parallel execution opportunities
    - Topological sorting
    """
    var dependencies: List[ToolDependency]
    var adjacency: Dict[String, List[String]]
    
    fn __init__(inout self):
        """Initialize empty dependency analyzer"""
        self.dependencies = List[ToolDependency]()
        self.adjacency = Dict[String, List[String]]()
    
    fn add_dependency(inout self, dep: ToolDependency):
        """Add a dependency between tools"""
        self.dependencies.append(dep)
        
        # Update adjacency list
        if dep.source_tool not in self.adjacency:
            self.adjacency[dep.source_tool] = List[String]()
        self.adjacency[dep.source_tool].append(dep.target_tool)
    
    fn has_dependencies(self, tool_name: String) -> Bool:
        """Check if tool has any dependencies"""
        return tool_name in self.adjacency and len(self.adjacency[tool_name]) > 0
    
    fn get_dependencies(self, tool_name: String) -> List[String]:
        """Get list of tools that depend on this tool"""
        if tool_name in self.adjacency:
            return self.adjacency[tool_name]
        return List[String]()
    
    fn find_independent_tools(
        self,
        tools: List[String]
    ) -> List[String]:
        """
        Find tools that can be executed in parallel
        
        Returns tools with no dependencies on each other
        """
        var independent = List[String]()
        
        for i in range(len(tools)):
            let tool = tools[i]
            if not self.has_dependencies(tool):
                independent.append(tool)
        
        return independent
    
    fn detect_cycles(self) -> Bool:
        """
        Detect circular dependencies
        
        Returns true if cycle exists (workflow cannot be executed)
        """
        # Simplified cycle detection
        # Full implementation would use DFS with color marking
        return False
    
    fn topological_sort(self, tools: List[String]) -> List[String]:
        """
        Sort tools in execution order respecting dependencies
        
        Returns topologically sorted list of tools
        """
        # Simplified - full implementation would use Kahn's algorithm
        # For now, return original order
        return tools


# ============================================================================
# Resource Estimation
# ============================================================================

@value
struct ResourceRequirements:
    """Estimated resource requirements for a tool"""
    var tool_name: String
    var estimated_time: Float32  # seconds
    var estimated_cost: Float32
    var memory_mb: Int
    var cpu_cores: Int
    var requires_gpu: Bool
    
    fn __init__(
        inout self,
        tool_name: String,
        estimated_time: Float32 = 1.0,
        estimated_cost: Float32 = 0.01
    ):
        self.tool_name = tool_name
        self.estimated_time = estimated_time
        self.estimated_cost = estimated_cost
        self.memory_mb = 512  # Default 512MB
        self.cpu_cores = 1
        self.requires_gpu = False


struct ResourceEstimator:
    """
    Estimate resource requirements for tools and workflows
    
    Uses:
    - Historical metrics
    - Tool characteristics
    - Workload patterns
    """
    var registry: ToolRegistry
    var metrics: MetricsTracker
    
    fn __init__(
        inout self,
        registry: ToolRegistry,
        metrics: MetricsTracker
    ):
        self.registry = registry
        self.metrics = metrics
    
    fn estimate_tool(self, tool_name: String) -> ResourceRequirements:
        """
        Estimate resources for a single tool
        
        Combines:
        - Historical averages from metrics
        - Tool metadata from registry
        - Default estimates if no history
        """
        var req = ResourceRequirements(tool_name=tool_name)
        
        # Get historical data if available
        let stats = self.metrics.get_tool_statistics(tool_name)
        if stats.execution_count > 0:
            req.estimated_time = stats.mean_time
            req.estimated_cost = stats.total_cost / Float32(stats.execution_count)
        else:
            # Use registry estimates
            req.estimated_time = self.registry.estimate_tool_time(tool_name)
            req.estimated_cost = self.registry.estimate_tool_cost(tool_name)
        
        return req
    
    fn estimate_workflow(
        self,
        tools: List[String],
        strategy: ExecutionStrategy
    ) -> WorkflowResourceEstimate:
        """
        Estimate total resources for a workflow
        
        Considers execution strategy:
        - Sequential: Sum of all tool times
        - Parallel: Max of tool times (assuming unlimited parallelism)
        - Adaptive: Weighted combination based on dependencies
        """
        var estimate = WorkflowResourceEstimate()
        
        var total_time: Float32 = 0.0
        var total_cost: Float32 = 0.0
        var max_time: Float32 = 0.0
        
        for i in range(len(tools)):
            let tool_req = self.estimate_tool(tools[i])
            total_time += tool_req.estimated_time
            total_cost += tool_req.estimated_cost
            max_time = max(max_time, tool_req.estimated_time)
        
        # Adjust based on strategy
        if strategy == ExecutionStrategy.SEQUENTIAL:
            estimate.estimated_time = total_time
        elif strategy == ExecutionStrategy.PARALLEL:
            estimate.estimated_time = max_time
        elif strategy == ExecutionStrategy.ADAPTIVE:
            # Assume 50% parallelization
            estimate.estimated_time = (total_time + max_time) / 2.0
        else:  # RL_OPTIMIZED
            # Use historical best (or default to adaptive)
            estimate.estimated_time = (total_time + max_time) / 2.0
        
        estimate.estimated_cost = total_cost
        estimate.num_tools = len(tools)
        
        return estimate


@value
struct WorkflowResourceEstimate:
    """Resource estimates for entire workflow"""
    var estimated_time: Float32
    var estimated_cost: Float32
    var num_tools: Int
    
    fn __init__(inout self):
        self.estimated_time = 0.0
        self.estimated_cost = 0.0
        self.num_tools = 0


# ============================================================================
# Strategy Selector
# ============================================================================

struct StrategySelector:
    """
    Select optimal execution strategy for workflows
    
    Considers:
    - Tool dependencies
    - Resource requirements
    - Historical performance
    - Workflow characteristics
    """
    var registry: ToolRegistry
    var metrics: MetricsTracker
    var dependency_analyzer: DependencyAnalyzer
    var resource_estimator: ResourceEstimator
    
    fn __init__(
        inout self,
        registry: ToolRegistry,
        metrics: MetricsTracker
    ):
        self.registry = registry
        self.metrics = metrics
        self.dependency_analyzer = DependencyAnalyzer()
        self.resource_estimator = ResourceEstimator(registry, metrics)
    
    fn select_strategy(
        self,
        tools: List[String],
        state: OrchestrationState
    ) -> StrategyRecommendation:
        """
        Select optimal strategy for workflow
        
        Decision factors:
        1. Dependencies: Can tools run in parallel?
        2. Resource constraints: Do we have capacity?
        3. Historical performance: What worked before?
        4. Risk tolerance: How critical is this workflow?
        
        Returns recommended strategy with confidence score
        """
        var recommendation = StrategyRecommendation()
        
        # Analyze dependencies
        let independent_tools = self.dependency_analyzer.find_independent_tools(tools)
        let has_cycles = self.dependency_analyzer.detect_cycles()
        
        if has_cycles:
            # Cannot execute - circular dependency
            recommendation.strategy = ExecutionStrategy.SEQUENTIAL
            recommendation.confidence = 0.0
            recommendation.reason = "Circular dependency detected"
            return recommendation
        
        # Estimate resources for each strategy
        let seq_estimate = self.resource_estimator.estimate_workflow(
            tools, ExecutionStrategy.SEQUENTIAL
        )
        let par_estimate = self.resource_estimator.estimate_workflow(
            tools, ExecutionStrategy.PARALLEL
        )
        
        # Decision logic
        let parallelization_potential = (
            Float32(len(independent_tools)) / Float32(len(tools))
        )
        
        if parallelization_potential > 0.7:
            # High parallelization potential
            recommendation.strategy = ExecutionStrategy.PARALLEL
            recommendation.confidence = 0.9
            recommendation.reason = "High parallelization potential"
            recommendation.estimated_speedup = seq_estimate.estimated_time / par_estimate.estimated_time
        
        elif parallelization_potential > 0.3:
            # Moderate parallelization potential
            recommendation.strategy = ExecutionStrategy.ADAPTIVE
            recommendation.confidence = 0.7
            recommendation.reason = "Mixed dependencies - adaptive approach"
            recommendation.estimated_speedup = 1.5  # Conservative estimate
        
        else:
            # Low parallelization potential - sequential is safer
            recommendation.strategy = ExecutionStrategy.SEQUENTIAL
            recommendation.confidence = 0.8
            recommendation.reason = "High dependencies - sequential execution"
            recommendation.estimated_speedup = 1.0
        
        # Use RL if available and workflow is complex
        if len(tools) > 5 and state.success_rate > 0.7:
            recommendation.consider_rl = True
        
        return recommendation
    
    fn should_use_rl(
        self,
        tools: List[String],
        state: OrchestrationState
    ) -> Bool:
        """
        Decide if RL-optimized strategy should be used
        
        Use RL when:
        - Workflow is complex (many tools)
        - Historical success rate is good (model has learned)
        - Not a critical workflow (can tolerate exploration)
        """
        let is_complex = len(tools) > 5
        let has_history = state.success_rate > 0.0
        let is_stable = state.success_rate > 0.7
        
        return is_complex and has_history and is_stable
    
    fn predict_performance(
        self,
        tools: List[String],
        strategy: ExecutionStrategy
    ) -> PerformancePrediction:
        """
        Predict workflow performance with given strategy
        
        Returns estimated time, cost, and success probability
        """
        let estimate = self.resource_estimator.estimate_workflow(tools, strategy)
        
        # Calculate success probability from historical data
        var success_prob: Float32 = 1.0
        for i in range(len(tools)):
            let stats = self.metrics.get_tool_statistics(tools[i])
            if stats.execution_count > 0:
                success_prob *= stats.success_rate
        
        return PerformancePrediction(
            estimated_time=estimate.estimated_time,
            estimated_cost=estimate.estimated_cost,
            success_probability=success_prob,
            strategy=strategy
        )


# ============================================================================
# Recommendation Types
# ============================================================================

@value
struct StrategyRecommendation:
    """Strategy recommendation with reasoning"""
    var strategy: ExecutionStrategy
    var confidence: Float32  # 0.0 to 1.0
    var reason: String
    var estimated_speedup: Float32
    var consider_rl: Bool
    
    fn __init__(inout self):
        self.strategy = ExecutionStrategy.SEQUENTIAL
        self.confidence = 0.5
        self.reason = "Default recommendation"
        self.estimated_speedup = 1.0
        self.consider_rl = False


@value
struct PerformancePrediction:
    """Predicted performance metrics"""
    var estimated_time: Float32
    var estimated_cost: Float32
    var success_probability: Float32
    var strategy: ExecutionStrategy
    
    fn __init__(
        inout self,
        estimated_time: Float32 = 0.0,
        estimated_cost: Float32 = 0.0,
        success_probability: Float32 = 1.0,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    ):
        self.estimated_time = estimated_time
        self.estimated_cost = estimated_cost
        self.success_probability = success_probability
        self.strategy = strategy
