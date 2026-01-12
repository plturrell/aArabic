# Mojo Recursive LLM Implementation
# Extracted from Python RLM research (alexzhang13/rlm)
# Pure Mojo implementation for zero-dependency recursive inference

from collections import List, Dict
from memory import UnsafePointer
from python import Python, PythonObject

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
    """
    var max_depth: Int
    var max_iterations: Int
    var current_depth: Int
    var verbose: Bool
    
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
                        
                        # RECURSIVE CALL!
                        var sub_result = self.completion(query, depth + 1)
                        recursive_calls += 1
                        
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


# ============================================================================
# API Integration (to be called from Zig HTTP server)
# ============================================================================

fn create_recursive_llm(max_depth: Int, max_iterations: Int) -> RecursiveLLM:
    """Factory function for creating RecursiveLLM instances"""
    return RecursiveLLM(max_depth, max_iterations)


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
