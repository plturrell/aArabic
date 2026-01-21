# Mojo Recursive LLM Implementation
# Extracted from Python RLM research (alexzhang13/rlm)
# Pure Mojo implementation for zero-dependency recursive inference
# Day 44: Enhanced with mHC (modified Homological Continuity) constraints

from collections import List, Dict
from memory import UnsafePointer
from python import Python, PythonObject
from time import now

# ============================================================================
# mHC Configuration for Recursive LLM (Day 44)
# ============================================================================

@value
struct MHCRecursionConfig:
    """mHC configuration specifically for recursive LLM operations.

    Applies depth-based constraints where deeper recursion levels
    get stricter stability requirements to prevent divergence.
    """
    var enabled: Bool
    var mhc_recursion_threshold: Float32      # Base stability threshold
    var base_sinkhorn_iterations: Int          # Base Sinkhorn-Knopp iterations
    var stability_threshold: Float32           # Amplification stability range
    var manifold_beta: Float32                 # Maximum activation bound
    var depth_strictness_factor: Float32       # How much stricter per depth level
    var log_stability_metrics: Bool
    var abort_on_instability: Bool

    fn __init__(out self):
        self.enabled = True
        self.mhc_recursion_threshold = Float32(0.15)  # Recursion-specific threshold
        self.base_sinkhorn_iterations = 10
        self.stability_threshold = Float32(1e-4)
        self.manifold_beta = Float32(10.0)
        self.depth_strictness_factor = Float32(1.5)   # 50% stricter per depth
        self.log_stability_metrics = False
        self.abort_on_instability = False

    fn get_threshold_for_depth(self, depth: Int) -> Float32:
        """Get stricter threshold for deeper recursion levels.

        Deeper levels have tighter stability bounds to prevent
        cascading instabilities from propagating upward.
        """
        var strictness = Float32(1.0)
        for _ in range(depth):
            strictness *= self.depth_strictness_factor
        return self.mhc_recursion_threshold / strictness

    fn get_iterations_for_depth(self, depth: Int) -> Int:
        """Get more iterations for deeper levels (more convergence needed)."""
        return self.base_sinkhorn_iterations + (depth * 5)


@value
struct RecursionStabilityMetrics:
    """Stability metrics tracking across recursion levels."""
    var depth: Int
    var query_id: Int
    var stability_score: Float32
    var amplification_factor: Float32
    var is_stable: Bool
    var convergence_iterations: Int
    var timestamp: Float64

    fn __init__(out self, depth: Int, query_id: Int):
        self.depth = depth
        self.query_id = query_id
        self.stability_score = Float32(1.0)
        self.amplification_factor = Float32(1.0)
        self.is_stable = True
        self.convergence_iterations = 0
        self.timestamp = now()

    @staticmethod
    fn calculate_stability(amplification: Float32) -> Bool:
        """Check if amplification factor is within stable range [0.9, 1.1]."""
        return amplification >= Float32(0.9) and amplification <= Float32(1.1)


struct RecursionStabilityTracker:
    """Tracks mHC stability across all recursion levels.

    Aggregates metrics to detect systematic instabilities
    that might indicate problematic recursion patterns.
    """
    var metrics: List[RecursionStabilityMetrics]
    var total_queries: Int
    var stable_queries: Int
    var max_depth_reached: Int
    var depth_stability_rates: Dict[Int, Float32]

    fn __init__(out self):
        self.metrics = List[RecursionStabilityMetrics]()
        self.total_queries = 0
        self.stable_queries = 0
        self.max_depth_reached = 0
        self.depth_stability_rates = Dict[Int, Float32]()

    fn record(inout self, metric: RecursionStabilityMetrics):
        """Record a stability metric from a recursion level."""
        self.metrics.append(metric)
        self.total_queries += 1
        if metric.is_stable:
            self.stable_queries += 1
        if metric.depth > self.max_depth_reached:
            self.max_depth_reached = metric.depth
        self._update_depth_rate(metric.depth, metric.is_stable)

    fn _update_depth_rate(inout self, depth: Int, is_stable: Bool):
        """Update stability rate for a specific depth."""
        # Simple running average (could use exponential moving average)
        var current_rate = self.depth_stability_rates.get(depth, Float32(1.0))
        var new_sample = Float32(1.0) if is_stable else Float32(0.0)
        var updated_rate = (current_rate + new_sample) / Float32(2.0)
        self.depth_stability_rates[depth] = updated_rate

    fn get_overall_stability_rate(self) -> Float32:
        """Get overall stability rate across all levels."""
        if self.total_queries == 0:
            return Float32(1.0)
        return Float32(self.stable_queries) / Float32(self.total_queries)

    fn get_depth_stability_rate(self, depth: Int) -> Float32:
        """Get stability rate for a specific depth."""
        return self.depth_stability_rates.get(depth, Float32(1.0))

    fn is_recursion_stable(self) -> Bool:
        """Check if overall recursion is stable (>90% stable queries)."""
        return self.get_overall_stability_rate() >= Float32(0.9)


# ============================================================================
# Core Types
# ============================================================================

@value
struct Message:
    """Represents a single message in conversation history"""
    var role: String
    var content: String
    
    fn __init__(inout self, role: String, content: String):
        self.role = role
        self.content = content
    
    fn to_dict(self) -> Dict[String, String]:
        """Convert to dictionary for LLM API"""
        var result = Dict[String, String]()
        result["role"] = self.role
        result["content"] = self.content
        return result


@value
struct CodeBlock:
    """Represents an extracted code block with execution result"""
    var code: String
    var result: String
    var has_llm_query: Bool
    var llm_query_text: String
    
    fn __init__(inout self, code: String):
        self.code = code
        self.result = ""
        self.has_llm_query = False
        self.llm_query_text = ""


@value
struct RLMIteration:
    """Represents one iteration of the RLM loop"""
    var prompt: List[Message]
    var response: String
    var code_blocks: List[CodeBlock]
    var has_final_answer: Bool
    var final_answer: String
    var iteration_time: Float64
    
    fn __init__(inout self):
        self.prompt = List[Message]()
        self.response = ""
        self.code_blocks = List[CodeBlock]()
        self.has_final_answer = False
        self.final_answer = ""
        self.iteration_time = 0.0


@value
struct RLMCompletion:
    """Final result from recursive LLM completion"""
    var prompt: String
    var response: String
    var execution_time: Float64
    var iterations_used: Int
    var recursive_calls: Int
    
    fn __init__(inout self, prompt: String, response: String):
        self.prompt = prompt
        self.response = response
        self.execution_time = 0.0
        self.iterations_used = 0
        self.recursive_calls = 0


# ============================================================================
# System Prompts
# ============================================================================

alias RLM_SYSTEM_PROMPT = """You are a Recursive Language Model (RLM). You can solve complex tasks by:

1. **Decomposing** tasks into smaller subtasks
2. **Writing Python code** to process data and make decisions
3. **Making recursive calls** using llm_query("your question") to solve subtasks
4. **Combining results** to answer the original question

## Available Tools

### llm_query(prompt: str) -> str
Make a recursive call to solve a subtask. Use this to:
- Summarize individual documents
- Answer specific questions
- Process data that's too complex for one pass

Example:
```python
# Recursive summarization
papers = ["paper1.txt", "paper2.txt", "paper3.txt"]
summaries = []
for paper in papers:
    summary = llm_query(f"Summarize: {paper}")
    summaries.append(summary)
final = "\\n".join(summaries)
```

### FINAL_ANSWER: marker
When you have the complete answer, prefix it with FINAL_ANSWER:

Example:
```
FINAL_ANSWER: The top 3 papers are: 1) RAG..., 2) Dense Retrieval..., 3) ColBERT...
```

## Guidelines

- Break complex tasks into simpler recursive calls
- Each llm_query() spawns a new RLM (can recurse further!)
- Use code to orchestrate multiple recursive calls
- Mark your final answer clearly with FINAL_ANSWER:

Now solve the user's task recursively!
"""


# ============================================================================
# Recursive LLM Core
# ============================================================================

struct RecursiveLLM:
    """
    Mojo implementation of Recursive Language Model pattern.
    Enables LLMs to decompose tasks and make recursive sub-calls.

    Day 44: Enhanced with mHC (modified Homological Continuity) constraints
    for stability tracking across recursion levels.
    """
    var max_depth: Int
    var max_iterations: Int
    var current_depth: Int
    var verbose: Bool
    var mhc_config: MHCRecursionConfig
    var stability_tracker: RecursionStabilityTracker
    var query_counter: Int

    fn __init__(inout self, max_depth: Int = 1, max_iterations: Int = 30, verbose: Bool = True):
        """
        Initialize Recursive LLM

        Args:
            max_depth: Maximum recursion depth (0 = no recursion, 1 = one level, etc.)
            max_iterations: Maximum iterations per completion call
            verbose: Print debug information
        """
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.current_depth = 0
        self.verbose = verbose
        self.mhc_config = MHCRecursionConfig()
        self.stability_tracker = RecursionStabilityTracker()
        self.query_counter = 0

    fn __init__(inout self, max_depth: Int, max_iterations: Int,
                verbose: Bool, mhc_config: MHCRecursionConfig):
        """Initialize with custom mHC configuration."""
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.current_depth = 0
        self.verbose = verbose
        self.mhc_config = mhc_config
        self.stability_tracker = RecursionStabilityTracker()
        self.query_counter = 0

    fn configure_mhc(inout self, enabled: Bool = True,
                     threshold: Float32 = 0.15,
                     strictness_factor: Float32 = 1.5):
        """Configure mHC parameters for recursion."""
        self.mhc_config.enabled = enabled
        self.mhc_config.mhc_recursion_threshold = threshold
        self.mhc_config.depth_strictness_factor = strictness_factor
        if self.verbose:
            print("⚙️ mHC configured: enabled=", enabled,
                  "threshold=", threshold, "strictness=", strictness_factor)
    
    fn completion(inout self, prompt: String, depth: Int = 0) -> RLMCompletion:
        """
        Main recursive completion call.
        
        This is the core RLM algorithm:
        1. Check if at max depth (base case: direct LLM call)
        2. Otherwise, iterate with code execution:
           a. Generate LLM response
           b. Extract and execute code blocks
           c. Detect llm_query() calls and recurse
           d. Check for final answer
           e. Add results to history and continue
        
        Args:
            prompt: User query to solve recursively
            depth: Current recursion depth
            
        Returns:
            RLMCompletion with final answer and metadata
        """
        self.current_depth = depth
        
        if self.verbose:
            print("=" * 80)
            print("🔄 Recursive LLM - Depth", depth, "/", self.max_depth)
            print("=" * 80)
            print("Prompt:", prompt)
            print()
        
        # Base case: at max depth, just call LLM directly
        if depth >= self.max_depth:
            return self._fallback_completion(prompt)
        
        # Recursive case: iterate with code execution
        var message_history = List[Message]()
        
        # Add system prompt
        message_history.append(Message("system", RLM_SYSTEM_PROMPT))
        
        # Add user prompt
        message_history.append(Message("user", prompt))
        
        var iteration: Int = 0
        var recursive_calls: Int = 0
        
        # Main iteration loop
        for i in range(self.max_iterations):
            iteration = i + 1
            
            if self.verbose:
                print("\n" + "─" * 80)
                print("📝 Iteration", iteration, "/", self.max_iterations)
                print("─" * 80)
            
            # 1. Get LLM response
            var response = self._call_llm(message_history)
            
            if self.verbose:
                print("\n🤖 LLM Response:")
                print(response)
            
            # 2. Extract code blocks
            var code_blocks = self._extract_code_blocks(response)
            
            if len(code_blocks) > 0 and self.verbose:
                print("\n💻 Found", len(code_blocks), "code block(s)")
            
            # 3. Execute code blocks and detect recursive calls
            var execution_results = List[String]()
            
            for idx in range(len(code_blocks)):
                var code_block = code_blocks[idx]
                
                if self.verbose:
                    print("\n  📦 Code Block", idx + 1, ":")
                    print("  ", code_block.code[:100], "...")
                
                # Check for llm_query() calls
                if self._contains_llm_query(code_block.code):
                    if self.verbose:
                        print("  🔄 Contains llm_query() - will recurse!")

                    # Extract query and make recursive call
                    var queries = self._extract_llm_queries(code_block.code)

                    for query_idx in range(len(queries)):
                        var query = queries[query_idx]
                        if self.verbose:
                            print("\n  ↳ Recursive call:", query[:50], "...")

                        # Day 44: Apply mHC constraints before recursion
                        var mhc_allowed = self._check_mhc_recursion_constraints(depth + 1)
                        if not mhc_allowed and self.mhc_config.abort_on_instability:
                            if self.verbose:
                                print("  ⚠️ mHC: Recursion blocked at depth", depth + 1)
                            execution_results.append("[mHC: Recursion depth limit reached]")
                            continue

                        # RECURSIVE CALL with mHC tracking!
                        self.query_counter += 1
                        var query_id = self.query_counter
                        var sub_result = self.completion(query, depth + 1)
                        recursive_calls += 1

                        # Day 44: Record stability metrics for this recursion
                        self._record_recursion_stability(depth + 1, query_id, sub_result)

                        execution_results.append(sub_result.response)

                        if self.verbose:
                            print("  ↳ Result:", sub_result.response[:100], "...")
                else:
                    # Regular code execution (Python FFI)
                    var result = self._execute_python_code(code_block.code)
                    execution_results.append(result)
                    
                    if self.verbose:
                        print("  ✅ Executed:", result[:100], "...")
            
            # 4. Check for final answer
            var final_answer = self._extract_final_answer(response)
            
            if len(final_answer) > 0:
                if self.verbose:
                    print("\n✨ FINAL ANSWER FOUND!")
                    print(final_answer)
                    print("=" * 80)
                
                var result = RLMCompletion(prompt, final_answer)
                result.iterations_used = iteration
                result.recursive_calls = recursive_calls
                return result
            
            # 5. Add response and execution results to history
            message_history.append(Message("assistant", response))
            
            if len(execution_results) > 0:
                var results_text = "Code execution results:\n"
                for idx in range(len(execution_results)):
                    results_text += str(idx + 1) + ". " + execution_results[idx] + "\n"
                message_history.append(Message("system", results_text))
        
        # Ran out of iterations - generate fallback answer
        if self.verbose:
            print("\n⚠️  Max iterations reached, generating fallback answer...")
        
        var fallback = self._generate_fallback_answer(message_history)
        var result = RLMCompletion(prompt, fallback)
        result.iterations_used = iteration
        result.recursive_calls = recursive_calls
        return result
    
    fn _fallback_completion(self, prompt: String) -> RLMCompletion:
        """
        Base case: at max depth, just call LLM directly without recursion
        """
        if self.verbose:
            print("📍 At max depth - direct LLM call (no recursion)")
        
        var message_history = List[Message]()
        message_history.append(Message("user", prompt))
        
        var response = self._call_llm(message_history)
        
        return RLMCompletion(prompt, response)
    
    fn _call_llm(self, message_history: List[Message]) -> String:
        """
        Call the underlying LLM engine (Shimmy).
        TODO: Integrate with actual Shimmy inference engine
        """
        # Placeholder: will integrate with Shimmy LLM engine
        # For now, return mock response for testing structure
        
        var last_message = message_history[len(message_history) - 1]
        
        # Simple mock response for testing
        if "2+2" in last_message.content:
            return "The answer is 4."
        elif "summarize" in last_message.content.lower():
            return "FINAL_ANSWER: This is a summary of the requested content."
        else:
            return "I will solve this step by step.\n```python\nresult = llm_query('What is the answer?')\nprint(result)\n```\n"
    
    fn _extract_code_blocks(self, response: String) -> List[CodeBlock]:
        """
        Extract Python code blocks from LLM response.
        Looks for ```python ... ``` or ``` ... ``` markers
        """
        var blocks = List[CodeBlock]()
        
        # Simple extraction (will enhance in Phase 3)
        var in_block = False
        var current_code = ""
        var lines = response.split("\n")
        
        for line in lines:
            if "```python" in line or "```" in line:
                if in_block:
                    # End of block
                    if len(current_code) > 0:
                        blocks.append(CodeBlock(current_code.strip()))
                    current_code = ""
                    in_block = False
                else:
                    # Start of block
                    in_block = True
            elif in_block:
                current_code += line + "\n"
        
        return blocks
    
    fn _contains_llm_query(self, code: String) -> Bool:
        """Check if code contains llm_query() call"""
        return "llm_query" in code or "llm.query" in code
    
    fn _extract_llm_queries(self, code: String) -> List[String]:
        """
        Extract llm_query("...") calls from Python code.
        Returns list of query strings.
        """
        var queries = List[String]()
        
        # Simple extraction: find llm_query("...")
        # Will enhance in Phase 3 with proper parsing
        
        var lines = code.split("\n")
        for line in lines:
            if "llm_query" in line:
                # Extract string between quotes
                var start = line.find('"')
                if start >= 0:
                    var end = line.find('"', start + 1)
                    if end > start:
                        var query = line[start + 1:end]
                        queries.append(query)
        
        return queries
    
    fn _execute_python_code(self, code: String) -> String:
        """
        Execute Python code via FFI.
        Will implement in Phase 2 with proper Python integration.
        """
        # Placeholder: will implement Python FFI execution
        return "[Code executed successfully]"
    
    fn _extract_final_answer(self, response: String) -> String:
        """
        Extract final answer from response.
        Looks for FINAL_ANSWER: marker or similar patterns.
        """
        var lines = response.split("\n")
        
        for line in lines:
            if "FINAL_ANSWER:" in line:
                var idx = line.find("FINAL_ANSWER:")
                if idx >= 0:
                    return line[idx + 13:].strip()
            elif "FINAL ANSWER:" in line:
                var idx = line.find("FINAL ANSWER:")
                if idx >= 0:
                    return line[idx + 13:].strip()
        
        return ""
    
    fn _generate_fallback_answer(self, message_history: List[Message]) -> String:
        """
        Generate fallback answer when max iterations reached.
        Prompts LLM to synthesize answer from conversation history.
        """
        var fallback_message = Message(
            "system",
            "Please provide a final answer based on the conversation so far."
        )

        var history_copy = message_history
        history_copy.append(fallback_message)

        return self._call_llm(history_copy)

    # ========================================================================
    # mHC Integration Methods (Day 44)
    # ========================================================================

    fn _check_mhc_recursion_constraints(self, target_depth: Int) -> Bool:
        """Check if recursion to target depth is allowed by mHC constraints.

        Uses depth-based thresholds where deeper levels are stricter.
        Returns True if recursion is allowed, False if blocked.
        """
        if not self.mhc_config.enabled:
            return True  # mHC disabled, allow all recursion

        # Get depth-specific threshold (stricter at deeper levels)
        var threshold = self.mhc_config.get_threshold_for_depth(target_depth)

        # Check stability rate at current depth
        var current_stability = self.stability_tracker.get_depth_stability_rate(
            target_depth - 1
        )

        # Allow if current level is stable enough
        var is_allowed = current_stability >= Float32(1.0) - threshold

        if self.verbose and self.mhc_config.log_stability_metrics:
            print("  📊 mHC check: depth=", target_depth,
                  "threshold=", threshold,
                  "stability=", current_stability,
                  "allowed=", is_allowed)

        return is_allowed

    fn _record_recursion_stability(inout self, depth: Int, query_id: Int,
                                   result: RLMCompletion):
        """Record stability metrics for a recursive call.

        Computes stability based on result characteristics and
        tracks metrics for depth-based analysis.
        """
        if not self.mhc_config.enabled:
            return

        # Create metrics for this recursion
        var metrics = RecursionStabilityMetrics(depth, query_id)

        # Compute stability score based on result
        # Higher score = more stable (response quality indicators)
        var stability_score = self._compute_recursion_stability_score(result)
        metrics.stability_score = stability_score

        # Compute effective amplification factor
        # Based on iterations used vs expected (depth-scaled)
        var expected_iterations = Float32(self.mhc_config.get_iterations_for_depth(depth))
        var actual_iterations = Float32(result.iterations_used)
        var amplification = actual_iterations / expected_iterations if expected_iterations > 0 else Float32(1.0)
        metrics.amplification_factor = amplification
        metrics.convergence_iterations = result.iterations_used

        # Determine overall stability
        metrics.is_stable = RecursionStabilityMetrics.calculate_stability(amplification)

        # Record in tracker
        self.stability_tracker.record(metrics)

        if self.verbose and self.mhc_config.log_stability_metrics:
            print("  📊 mHC recorded: depth=", depth,
                  "query_id=", query_id,
                  "amplification=", amplification,
                  "stable=", metrics.is_stable)

    fn _compute_recursion_stability_score(self, result: RLMCompletion) -> Float32:
        """Compute stability score for a recursion result.

        Score is based on:
        - Response length (normalized)
        - Iterations used vs max
        - Recursive calls made (complexity indicator)
        """
        # Normalize response length (0-1 scale, cap at 1000 chars)
        var response_len = Float32(len(result.response))
        var length_score = min(response_len / Float32(1000.0), Float32(1.0))

        # Iteration efficiency (fewer iterations = more stable)
        var iter_ratio = Float32(result.iterations_used) / Float32(self.max_iterations)
        var iter_score = Float32(1.0) - min(iter_ratio, Float32(1.0))

        # Recursive complexity (fewer nested calls = more stable)
        var recursion_penalty = Float32(result.recursive_calls) * Float32(0.1)
        var recursion_score = max(Float32(1.0) - recursion_penalty, Float32(0.0))

        # Weighted combination
        return (length_score * Float32(0.3) +
                iter_score * Float32(0.4) +
                recursion_score * Float32(0.3))

    fn get_mhc_stability_report(self) -> String:
        """Get a summary report of mHC stability across all recursion levels."""
        var report = "=== mHC Recursion Stability Report ===\n"
        report += "Total queries: " + str(self.stability_tracker.total_queries) + "\n"
        report += "Stable queries: " + str(self.stability_tracker.stable_queries) + "\n"
        report += "Overall stability: " + str(
            self.stability_tracker.get_overall_stability_rate() * Float32(100.0)
        ) + "%\n"
        report += "Max depth reached: " + str(self.stability_tracker.max_depth_reached) + "\n"
        report += "Recursion stable: " + str(self.stability_tracker.is_recursion_stable()) + "\n"
        return report


# ============================================================================
# API Integration (to be called from Zig HTTP server)
# ============================================================================

fn create_recursive_llm(max_depth: Int, max_iterations: Int) -> RecursiveLLM:
    """Factory function for creating RecursiveLLM instances"""
    return RecursiveLLM(max_depth, max_iterations)


fn create_recursive_llm_with_mhc(
    max_depth: Int,
    max_iterations: Int,
    mhc_threshold: Float32 = 0.15,
    mhc_strictness: Float32 = 1.5,
    verbose: Bool = True
) -> RecursiveLLM:
    """Factory function for creating RecursiveLLM with mHC constraints.

    Day 44: Creates instance with mHC stability tracking enabled.

    Args:
        max_depth: Maximum recursion depth
        max_iterations: Maximum iterations per level
        mhc_threshold: Base mHC recursion threshold (default: 0.15)
        mhc_strictness: How much stricter per depth level (default: 1.5x)
        verbose: Print debug information

    Returns:
        RecursiveLLM instance with mHC enabled
    """
    var mhc_config = MHCRecursionConfig()
    mhc_config.enabled = True
    mhc_config.mhc_recursion_threshold = mhc_threshold
    mhc_config.depth_strictness_factor = mhc_strictness
    mhc_config.log_stability_metrics = verbose

    return RecursiveLLM(max_depth, max_iterations, verbose, mhc_config)


fn recursive_completion(
    prompt: String,
    max_depth: Int = 1,
    max_iterations: Int = 30,
    verbose: Bool = True
) -> RLMCompletion:
    """
    Convenience function for one-shot recursive completion.
    Can be called from Zig via FFI.
    """
    var rlm = RecursiveLLM(max_depth, max_iterations, verbose)
    return rlm.completion(prompt, 0)


fn recursive_completion_with_mhc(
    prompt: String,
    max_depth: Int = 1,
    max_iterations: Int = 30,
    mhc_threshold: Float32 = 0.15,
    verbose: Bool = True
) -> RLMCompletion:
    """
    Day 44: Recursive completion with mHC stability tracking.

    Applies depth-based constraints where deeper recursion levels
    have stricter stability requirements.

    Args:
        prompt: User query to solve recursively
        max_depth: Maximum recursion depth
        max_iterations: Maximum iterations per level
        mhc_threshold: Base stability threshold
        verbose: Print debug information

    Returns:
        RLMCompletion with final answer and mHC stability metadata
    """
    var rlm = create_recursive_llm_with_mhc(
        max_depth, max_iterations, mhc_threshold, Float32(1.5), verbose
    )
    var result = rlm.completion(prompt, 0)

    # Log stability report if verbose
    if verbose:
        print(rlm.get_mhc_stability_report())

    return result
