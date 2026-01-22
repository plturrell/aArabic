# Pattern Extraction for Recursive LLM
# Detects llm_query() patterns in LLM responses (no Python execution needed!)

from collections import List

# ============================================================================
# Query Pattern Detection
# ============================================================================

fn extract_llm_queries(response: String) -> List[String]:
    """
    Extract llm_query() calls from LLM response.
    
    Supports patterns:
    - llm_query("text")
    - llm_query('text')
    - llm.query("text")
    - llm.query('text')
    
    Args:
        response: LLM-generated response text
        
    Returns:
        List of query strings found
    """
    var queries = List[String]()
    
    # Patterns to detect
    var patterns = [
        'llm_query("',
        "llm_query('",
        'llm.query("',
        "llm.query('"
    ]
    
    for pattern in patterns:
        var idx = 0
        while True:
            idx = response.find(pattern, idx)
            if idx < 0:
                break
            
            # Extract quoted string after pattern
            var quote_char = pattern[len(pattern) - 1]  # " or '
            var start_idx = idx + len(pattern)
            var end_idx = response.find(quote_char, start_idx)
            
            if end_idx > start_idx:
                var query = response[start_idx:end_idx]
                queries.append(query)
                idx = end_idx + 1
            else:
                idx += 1
    
    return queries


fn contains_llm_query(text: String) -> Bool:
    """
    Quick check if text contains any llm_query() pattern.
    
    Args:
        text: Text to check
        
    Returns:
        True if contains llm_query pattern
    """
    return ("llm_query" in text) or ("llm.query" in text)


fn extract_quoted_string(
    text: String,
    start_idx: Int,
    quote_char: String = '"'
) -> String:
    """
    Extract string between quotes starting at given index.
    
    Handles escaped quotes: \"
    
    Args:
        text: Source text
        start_idx: Index to start looking for closing quote
        quote_char: Quote character (" or ')
        
    Returns:
        Extracted string (empty if not found)
    """
    var result = ""
    var i = start_idx
    var escaped = False
    
    while i < len(text):
        var char = text[i]
        
        if escaped:
            result += char
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote_char:
            return result
        else:
            result += char
        
        i += 1
    
    return ""


# ============================================================================
# Final Answer Detection
# ============================================================================

fn has_final_answer(response: String) -> Bool:
    """
    Check if response contains a final answer marker.
    
    Looks for patterns:
    - FINAL_ANSWER:
    - FINAL ANSWER:
    - Final Answer:
    - [FINAL]
    
    Args:
        response: LLM response text
        
    Returns:
        True if final answer marker found
    """
    var markers = [
        "FINAL_ANSWER:",
        "FINAL ANSWER:",
        "Final Answer:",
        "final answer:",
        "[FINAL]",
        "[ANSWER]"
    ]
    
    for marker in markers:
        if marker in response:
            return True
    
    return False


fn extract_final_answer(response: String) -> String:
    """
    Extract final answer from response.
    
    Looks for answer after marker and returns rest of text.
    
    Args:
        response: LLM response with final answer
        
    Returns:
        Final answer text (empty if not found)
    """
    var markers = [
        "FINAL_ANSWER:",
        "FINAL ANSWER:",
        "Final Answer:",
        "final answer:"
    ]
    
    for marker in markers:
        var idx = response.find(marker)
        if idx >= 0:
            var answer_start = idx + len(marker)
            var answer = response[answer_start:].strip()
            return answer
    
    # Try bracket markers [FINAL], [ANSWER]
    var bracket_markers = ["[FINAL]", "[ANSWER]"]
    for marker in bracket_markers:
        var idx = response.find(marker)
        if idx >= 0:
            # Look for content after marker
            var answer_start = idx + len(marker)
            var answer = response[answer_start:].strip()
            return answer
    
    return ""


# ============================================================================
# Code Block Detection (Optional - for better context)
# ============================================================================

fn extract_code_blocks(response: String) -> List[String]:
    """
    Extract code blocks from LLM response.
    
    Looks for:
    - ```python ... ```
    - ```...```
    - ``` (no language specified)
    
    Args:
        response: LLM response text
        
    Returns:
        List of code block contents
    """
    var blocks = List[String]()
    var lines = response.split("\n")
    
    var in_block = False
    var current_block = ""
    
    for line in lines:
        var stripped = line.strip()
        
        if stripped.startswith("```"):
            if in_block:
                # End of block
                if len(current_block) > 0:
                    blocks.append(current_block.strip())
                current_block = ""
                in_block = False
            else:
                # Start of block
                in_block = True
        elif in_block:
            current_block += line + "\n"
    
    return blocks


fn contains_code_block(response: String) -> Bool:
    """
    Quick check if response contains code blocks.
    
    Args:
        response: Text to check
        
    Returns:
        True if contains code block markers
    """
    return "```" in response


# ============================================================================
# Query Replacement (for inserting results)
# ============================================================================

fn replace_queries_with_results(
    response: String,
    queries: List[String],
    results: List[String]
) -> String:
    """
    Replace llm_query() calls in response with their results.
    
    Replaces:
        llm_query("question")
    With:
        [RESULT: answer]
    
    Args:
        response: Original LLM response
        queries: List of queries that were executed
        results: List of results (same order as queries)
        
    Returns:
        Response with queries replaced by results
    """
    var modified = response
    
    # Replace each query with its result
    for i in range(min(len(queries), len(results))):
        var query = queries[i]
        var result = results[i]
        
        # Try different patterns
        var patterns = [
            'llm_query("' + query + '")',
            "llm_query('" + query + "')",
            'llm.query("' + query + '")',
            "llm.query('" + query + "')"
        ]
        
        for pattern in patterns:
            if pattern in modified:
                var replacement = "[RESULT: " + result + "]"
                modified = modified.replace(pattern, replacement)
    
    return modified


# ============================================================================
# Utility Functions
# ============================================================================

fn count_llm_queries(response: String) -> Int:
    """
    Count number of llm_query() calls in response.
    
    Args:
        response: LLM response text
        
    Returns:
        Number of llm_query() calls found
    """
    return len(extract_llm_queries(response))


fn validate_query(query: String) -> Bool:
    """
    Validate that a query is reasonable.
    
    Checks:
    - Not empty
    - Not too long (max 1000 chars)
    - Contains actual text (not just whitespace)
    
    Args:
        query: Query string to validate
        
    Returns:
        True if query is valid
    """
    if len(query) == 0:
        return False
    
    if len(query) > 1000:
        return False
    
    if len(query.strip()) == 0:
        return False
    
    return True


fn sanitize_query(query: String) -> String:
    """
    Sanitize query string for safe processing.
    
    - Trim whitespace
    - Remove newlines
    - Limit length
    
    Args:
        query: Raw query string
        
    Returns:
        Sanitized query
    """
    var sanitized = query.strip()
    
    # Replace newlines with spaces
    sanitized = sanitized.replace("\n", " ")
    sanitized = sanitized.replace("\r", " ")
    
    # Collapse multiple spaces
    while "  " in sanitized:
        sanitized = sanitized.replace("  ", " ")
    
    # Limit length
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000] + "..."
    
    return sanitized


# ============================================================================
# Pattern Statistics (for debugging)
# ============================================================================

struct PatternStats:
    """Statistics about patterns found in response"""
    var num_queries: Int
    var num_code_blocks: Int
    var has_final_answer: Bool
    var query_list: List[String]
    
    fn __init__(inout self, response: String):
        """Analyze response and collect statistics"""
        self.query_list = extract_llm_queries(response)
        self.num_queries = len(self.query_list)
        self.num_code_blocks = len(extract_code_blocks(response))
        self.has_final_answer = has_final_answer(response)
    
    fn to_string(self) -> String:
        """Convert stats to readable string"""
        var result = "Pattern Statistics:\n"
        result += f"  Queries: {self.num_queries}\n"
        result += f"  Code blocks: {self.num_code_blocks}\n"
        result += f"  Final answer: {self.has_final_answer}\n"
        
        if self.num_queries > 0:
            result += "  Query list:\n"
            for i in range(self.num_queries):
                var query = self.query_list[i]
                var preview = query[:50] if len(query) > 50 else query
                result += f"    {i+1}. {preview}...\n"
        
        return result


# ============================================================================
# Testing Utilities
# ============================================================================

fn test_pattern_extraction():
    """Test pattern extraction functionality"""
    print("Testing Pattern Extraction...")
    print("=" * 60)
    
    # Test 1: Single query
    var test1 = 'I will llm_query("What is 2+2?") to get the answer.'
    var queries1 = extract_llm_queries(test1)
    print("Test 1 - Single query:")
    print(f"  Found {len(queries1)} queries")
    if len(queries1) > 0:
        print(f"  Query: {queries1[0]}")
    print()
    
    # Test 2: Multiple queries
    var test2 = '''
    First llm_query("question 1")
    Then llm_query("question 2")
    Finally llm_query("question 3")
    '''
    var queries2 = extract_llm_queries(test2)
    print("Test 2 - Multiple queries:")
    print(f"  Found {len(queries2)} queries")
    for i in range(len(queries2)):
        print(f"  Query {i+1}: {queries2[i]}")
    print()
    
    # Test 3: Final answer detection
    var test3 = "After analysis, FINAL_ANSWER: The result is 42."
    print("Test 3 - Final answer:")
    print(f"  Has final answer: {has_final_answer(test3)}")
    print(f"  Answer: {extract_final_answer(test3)}")
    print()
    
    # Test 4: Code blocks
    var test4 = '''
    Here's the code:
    ```python
    result = llm_query("test")
    print(result)
    ```
    '''
    var blocks = extract_code_blocks(test4)
    print("Test 4 - Code blocks:")
    print(f"  Found {len(blocks)} blocks")
    print()
    
    print("=" * 60)
    print("Pattern extraction tests complete!")
