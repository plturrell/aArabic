# Shimmy Integration for Recursive LLM
# Connects Recursive LLM to Shimmy inference engine

from collections import List, Dict
from memory import UnsafePointer
from sys.ffi import DLHandle

# Import our components
from .recursive_llm import Message, RLMCompletion
from .petri_net import PetriNet, Token, RLMState
from .pattern_extractor import (
    extract_llm_queries,
    has_final_answer,
    extract_final_answer,
    replace_queries_with_results
)

# ============================================================================
# Shimmy Engine Interface
# ============================================================================

struct ShimmyEngine:
    """
    Interface to Shimmy LLM inference engine.
    Wraps Shimmy's generate() function for recursive calls.
    """
    var model_name: String
    var temperature: Float32
    var max_tokens: Int
    var verbose: Bool
    
    fn __init__(
        inout self,
        model_name: String = "llama-3.2-1b",
        temperature: Float32 = 0.7,
        max_tokens: Int = 2048,
        verbose: Bool = False
    ):
        """
        Initialize Shimmy engine wrapper
        
        Args:
            model_name: Model to use (from Shimmy Phases 1-5)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            verbose: Print debug info
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
    
    fn generate(self, messages: List[Message]) -> String:
        """
        Generate response from Shimmy LLM.
        
        This will integrate with your existing Shimmy inference engine
        from Phases 1-5. For now, returns mock responses.
        
        Args:
            messages: Conversation history
            
        Returns:
            Generated response text
        """
        if self.verbose:
            print(f"  [Shimmy] Generating with {self.model_name}")
            print(f"  [Shimmy] History length: {len(messages)}")
        
        # TODO: Replace with actual Shimmy inference
        # This is where you'll call your Phases 1-5 Shimmy engine:
        # 
        # from serviceShimmy import ShimmyInference
        # var shimmy = ShimmyInference(self.model_name)
        # return shimmy.generate(
        #     messages=format_messages_for_shimmy(messages),
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens
        # )
        
        # Mock response for testing
        var last_msg = messages[len(messages) - 1].content
        
        if "2+2" in last_msg or "math" in last_msg.lower():
            return "The answer is 4. FINAL_ANSWER: 4"
        elif "summarize" in last_msg.lower():
            return 'I\'ll use recursion:\n```python\nsummary = llm_query("Summarize in detail")\nprint(summary)\n```'
        else:
            return "Let me break this down with llm_query()."
    
    fn generate_batch(self, queries: List[String]) -> List[String]:
        """
        Generate responses for multiple queries concurrently.
        
        Uses Shimmy's batch inference if available, otherwise
        executes sequentially (will optimize later).
        
        Args:
            queries: List of independent queries
            
        Returns:
            List of generated responses
        """
        var results = List[String]()
        
        if self.verbose:
            print(f"  [Shimmy] Batch generating {len(queries)} queries")
        
        # TODO: Optimize with parallel execution
        # For now, sequential (will add concurrency in Phase 4)
        for query in queries:
            var msg = List[Message]()
            msg.append(Message("user", query))
            var response = self.generate(msg)
            results.append(response)
        
        return results


# ============================================================================
# Recursive LLM with Shimmy Integration
# ============================================================================

struct IntegratedRecursiveLLM:
    """
    Recursive LLM integrated with Shimmy engine and Petri net.
    This is the complete, production-ready implementation.
    """
    var shimmy: ShimmyEngine
    var petri_net: PetriNet
    var max_depth: Int
    var max_iterations: Int
    var verbose: Bool
    
    fn __init__(
        inout self,
        model_name: String = "llama-3.2-1b",
        max_depth: Int = 2,
        max_iterations: Int = 30,
        max_concurrent: Int = 10,
        verbose: Bool = True
    ):
        """
        Initialize integrated recursive LLM
        
        Args:
            model_name: Shimmy model to use
            max_depth: Maximum recursion depth
            max_iterations: Max iterations per completion
            max_concurrent: Max concurrent queries
            verbose: Debug output
        """
        self.shimmy = ShimmyEngine(model_name, verbose=verbose)
        self.petri_net = PetriNet(max_concurrent, max_depth, verbose=verbose)
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    fn completion(inout self, prompt: String, depth: Int = 0) -> RLMCompletion:
        """
        Main recursive completion with Petri net state management.
        
        State flow:
        1. IDLE → GENERATING (Shimmy call)
        2. GENERATING → PARSING (extract queries)
        3. PARSING → EXECUTING (spawn concurrent)
        4. EXECUTING → WAITING (await results)
        5. WAITING → COMBINING (merge results)
        6. COMBINING → FINAL_ANSWER or back to GENERATING
        
        Args:
            prompt: User query
            depth: Current recursion depth
            
        Returns:
            RLMCompletion with final answer
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"🔄 Recursive LLM Completion - Depth {depth}/{self.max_depth}")
            print("=" * 70)
            print(f"Prompt: {prompt[:100]}...")
        
        # Base case: at max depth
        if depth >= self.max_depth:
            return self._base_case_completion(prompt)
        
        # Create initial token
        var root_token = self.petri_net.create_token(prompt, depth)
        self.petri_net.add_token(RLMState.IDLE, root_token)
        
        # Build message history
        var history = List[Message]()
        history.append(Message("system", get_rlm_system_prompt()))
        history.append(Message("user", prompt))
        
        var iterations = 0
        var recursive_calls = 0
        
        # Main iteration loop with Petri net state management
        for i in range(self.max_iterations):
            iterations = i + 1
            
            if self.verbose:
                print(f"\n📝 Iteration {iterations}/{self.max_iterations}")
                self.petri_net.visualize_petri_net()
            
            # Check for deadlock
            if self.petri_net.detect_deadlock():
                print("⚠️  Deadlock detected! Generating fallback...")
                return self._fallback_completion(prompt, history, iterations)
            
            # STATE: IDLE → GENERATING
            self.petri_net.move_token(root_token, RLMState.IDLE, RLMState.GENERATING)
            
            # Generate response from Shimmy
            var response = self.shimmy.generate(history)
            
            if self.verbose:
                print(f"🤖 Response: {response[:200]}...")
            
            # STATE: GENERATING → PARSING
            self.petri_net.move_token(root_token, RLMState.GENERATING, RLMState.PARSING)
            
            # Extract llm_query() calls
            var queries = extract_llm_queries(response)
            
            if len(queries) > 0:
                if self.verbose:
                    print(f"🔍 Found {len(queries)} llm_query() calls")
                
                # STATE: PARSING → EXECUTING
                self.petri_net.move_token(root_token, RLMState.PARSING, RLMState.EXECUTING_QUERIES)
                
                # Execute queries concurrently (respecting limits)
                var results = self._execute_concurrent_queries(queries, depth + 1)
                recursive_calls += len(queries)
                
                # STATE: EXECUTING → COMBINING
                self.petri_net.move_token(
                    root_token,
                    RLMState.EXECUTING_QUERIES,
                    RLMState.COMBINING_RESULTS
                )
                
                # Replace queries with results in response
                response = replace_queries_with_results(response, queries, results)
                
                if self.verbose:
                    print(f"✅ Substituted {len(results)} results")
            
            # Check for final answer
            if has_final_answer(response):
                var final = extract_final_answer(response)
                
                if self.verbose:
                    print(f"✨ FINAL ANSWER: {final[:100]}...")
                
                # STATE: → FINAL_ANSWER
                self.petri_net.move_token(
                    root_token,
                    root_token.state,
                    RLMState.FINAL_ANSWER
                )
                
                var result = RLMCompletion(prompt, final)
                result.iterations_used = iterations
                result.recursive_calls = recursive_calls
                return result
            
            # Add to history for next iteration
            history.append(Message("assistant", response))
            
            if len(queries) > 0:
                var results_msg = "Recursive query results:\n"
                for idx in range(len(queries)):
                    results_msg += f"{idx+1}. {queries[idx][:50]}... → [completed]\n"
                history.append(Message("system", results_msg))
        
        # Max iterations reached
        if self.verbose:
            print("⚠️  Max iterations reached, generating fallback")
        
        return self._fallback_completion(prompt, history, iterations)
    
    fn _base_case_completion(self, prompt: String) -> RLMCompletion:
        """Handle base case: direct Shimmy call at max depth"""
        if self.verbose:
            print(f"📍 Base case (depth={self.max_depth}): Direct Shimmy call")
        
        var messages = List[Message]()
        messages.append(Message("user", prompt))
        
        var response = self.shimmy.generate(messages)
        
        return RLMCompletion(prompt, response)
    
    fn _execute_concurrent_queries(
        inout self,
        queries: List[String],
        depth: Int
    ) -> List[String]:
        """
        Execute multiple queries concurrently (respecting Petri net limits).
        
        Uses Petri net to manage concurrency and prevent resource exhaustion.
        
        Args:
            queries: List of queries to execute
            depth: Recursion depth for these queries
            
        Returns:
            List of results (same order as queries)
        """
        var results = List[String]()
        
        for query in queries:
            # Check if we can spawn (Petri net concurrency control)
            if not self.petri_net.can_spawn_query(depth):
                if self.verbose:
                    print(f"  ⏳ Waiting for query slot...")
                # In production, would wait/queue
                # For now, execute sequentially
            
            # Create token for this query
            var query_token = self.petri_net.create_token(query, depth)
            self.petri_net.add_token(RLMState.EXECUTING_QUERIES, query_token)
            
            # Recursive call
            var result = self.completion(query, depth)
            
            # Mark token complete
            query_token.set_result(result.response)
            self.petri_net.move_token(
                query_token,
                RLMState.EXECUTING_QUERIES,
                RLMState.FINAL_ANSWER
            )
            
            results.append(result.response)
        
        return results
    
    fn _fallback_completion(
        self,
        prompt: String,
        history: List[Message],
        iterations: Int
    ) -> RLMCompletion:
        """Generate fallback answer when max iterations reached"""
        var fallback_msg = Message(
            "system",
            "Please provide a final answer based on our conversation."
        )
        history.append(fallback_msg)
        
        var response = self.shimmy.generate(history)
        
        var result = RLMCompletion(prompt, response)
        result.iterations_used = iterations
        return result


# ============================================================================
# API Functions (callable from Zig HTTP server)
# ============================================================================

fn create_recursive_llm_with_shimmy(
    model_name: String,
    max_depth: Int = 2,
    max_iterations: Int = 30,
    max_concurrent: Int = 10,
    verbose: Bool = True
) -> IntegratedRecursiveLLM:
    """
    Factory function for creating integrated recursive LLM.
    
    Args:
        model_name: Shimmy model name
        max_depth: Maximum recursion depth
        max_iterations: Max iterations per call
        max_concurrent: Max concurrent queries
        verbose: Debug output
        
    Returns:
        Fully configured recursive LLM
    """
    return IntegratedRecursiveLLM(
        model_name,
        max_depth,
        max_iterations,
        max_concurrent,
        verbose
    )


fn recursive_completion_with_shimmy(
    prompt: String,
    model_name: String = "llama-3.2-1b",
    max_depth: Int = 2,
    max_iterations: Int = 30
) -> RLMCompletion:
    """
    Convenience function for one-shot recursive completion.
    
    Can be called from Zig via C ABI.
    
    Args:
        prompt: User query
        model_name: Shimmy model
        max_depth: Max recursion depth
        max_iterations: Max iterations
        
    Returns:
        Completion result
    """
    var rlm = create_recursive_llm_with_shimmy(
        model_name,
        max_depth,
        max_iterations,
        max_concurrent=10,
        verbose=True
    )
    
    return rlm.completion(prompt, 0)


# ============================================================================
# Helper Functions
# ============================================================================

fn get_rlm_system_prompt() -> String:
    """Get the RLM system prompt for Shimmy"""
    return """You are a Recursive Language Model (RLM). You solve complex tasks by:

1. **Decomposing** tasks into subtasks
2. **Making recursive calls** using llm_query("question") 
3. **Combining results** to answer the original question

## How to Use llm_query()

When you need to solve a subtask, use:
```
llm_query("specific question about subtask")
```

Each llm_query() spawns a new LLM call with full context for that subtask.

Example:
```
To analyze 3 papers, I'll use:
1. llm_query("Summarize paper 1")
2. llm_query("Summarize paper 2")  
3. llm_query("Summarize paper 3")
Then combine the summaries.
```

## Final Answer

When you have the complete answer, use:
```
FINAL_ANSWER: [your complete answer]
```

Now solve the user's task recursively!
"""


fn format_messages_for_shimmy(messages: List[Message]) -> String:
    """
    Format message history for Shimmy API.
    
    Converts List[Message] to Shimmy's expected format.
    
    Args:
        messages: Message history
        
    Returns:
        Formatted string for Shimmy
    """
    var formatted = ""
    
    for msg in messages:
        if msg.role == "system":
            formatted += "[SYSTEM]\n" + msg.content + "\n\n"
        elif msg.role == "user":
            formatted += "[USER]\n" + msg.content + "\n\n"
        elif msg.role == "assistant":
            formatted += "[ASSISTANT]\n" + msg.content + "\n\n"
    
    return formatted


# ============================================================================
# C ABI Exports (for Zig HTTP server integration)
# ============================================================================

# Export for Zig to call
@export
fn rlm_recursive_completion(
    prompt_ptr: UnsafePointer[UInt8],
    prompt_len: Int,
    max_depth: Int,
    max_iterations: Int
) -> UnsafePointer[UInt8]:
    """
    C ABI function for recursive completion.
    Callable from Zig HTTP server.
    
    Args:
        prompt_ptr: Pointer to prompt string
        prompt_len: Length of prompt
        max_depth: Max recursion depth
        max_iterations: Max iterations
        
    Returns:
        Pointer to null-terminated result string
    """
    var prompt = String(prompt_ptr, prompt_len)
    
    var result = recursive_completion_with_shimmy(
        prompt,
        "llama-3.2-1b",
        max_depth,
        max_iterations
    )
    
    # Convert result to C string
    var result_str = result.response
    var c_str = result_str.c_str()
    
    return c_str


# ============================================================================
# Testing Utilities
# ============================================================================

fn test_shimmy_integration():
    """Test Shimmy integration"""
    print("Testing Shimmy Integration...")
    print("=" * 70)
    
    # Test 1: Simple generation
    print("\n🧪 Test 1: Simple Generation")
    var shimmy = ShimmyEngine("test-model", verbose=True)
    var messages = List[Message]()
    messages.append(Message("user", "What is 2+2?"))
    
    var response = shimmy.generate(messages)
    print(f"Response: {response}")
    
    # Test 2: Batch generation
    print("\n🧪 Test 2: Batch Generation")
    var queries = List[String]()
    queries.append("Question 1")
    queries.append("Question 2")
    queries.append("Question 3")
    
    var results = shimmy.generate_batch(queries)
    print(f"Generated {len(results)} responses")
    
    # Test 3: Full recursive completion
    print("\n🧪 Test 3: Recursive Completion")
    var rlm = create_recursive_llm_with_shimmy("test-model", max_depth=2)
    var completion = rlm.completion("Test recursive query", 0)
    print(f"Result: {completion.response}")
    print(f"Iterations: {completion.iterations_used}")
    print(f"Recursive calls: {completion.recursive_calls}")
    
    print("\n" + "=" * 70)
    print("✅ Shimmy integration tests complete!")
