# Petri Net State Machine for Recursive LLM
# Provides formal state management, concurrency control, and deadlock detection

from collections import List, Dict
from time import now

# ============================================================================
# States (Places in Petri Net)
# ============================================================================

enum RLMState:
    """States in the Recursive LLM Petri net"""
    IDLE = 0
    GENERATING = 1
    PARSING = 2
    EXECUTING_QUERIES = 3
    WAITING_FOR_RESULTS = 4
    COMBINING_RESULTS = 5
    FINAL_ANSWER = 6
    ERROR = 7

fn state_to_string(state: RLMState) -> String:
    """Convert state enum to string for debugging"""
    if state == RLMState.IDLE:
        return "IDLE"
    elif state == RLMState.GENERATING:
        return "GENERATING"
    elif state == RLMState.PARSING:
        return "PARSING"
    elif state == RLMState.EXECUTING_QUERIES:
        return "EXECUTING_QUERIES"
    elif state == RLMState.WAITING_FOR_RESULTS:
        return "WAITING_FOR_RESULTS"
    elif state == RLMState.COMBINING_RESULTS:
        return "COMBINING_RESULTS"
    elif state == RLMState.FINAL_ANSWER:
        return "FINAL_ANSWER"
    else:
        return "ERROR"


# ============================================================================
# Token (Represents work in the Petri net)
# ============================================================================

@value
struct Token:
    """
    Represents a unit of work in the Petri net.
    Each token carries a query and metadata.
    """
    var id: Int
    var query: String
    var depth: Int
    var parent_id: Int
    var created_at: Float64
    var state: RLMState
    var result: String
    var error: String
    
    fn __init__(
        inout self,
        id: Int,
        query: String,
        depth: Int,
        parent_id: Int = -1
    ):
        self.id = id
        self.query = query
        self.depth = depth
        self.parent_id = parent_id
        self.created_at = now()
        self.state = RLMState.IDLE
        self.result = ""
        self.error = ""
    
    fn is_expired(self, timeout_ms: Float64) -> Bool:
        """Check if token has been waiting too long"""
        return (now() - self.created_at) > timeout_ms
    
    fn set_result(inout self, result: String):
        """Set result and mark as complete"""
        self.result = result
        self.state = RLMState.FINAL_ANSWER
    
    fn set_error(inout self, error: String):
        """Set error and mark as failed"""
        self.error = error
        self.state = RLMState.ERROR


# ============================================================================
# Petri Net Controller
# ============================================================================

struct PetriNet:
    """
    Petri net controller for recursive LLM state management.
    
    Manages:
    - State transitions
    - Concurrent query execution
    - Resource limits
    - Deadlock detection
    """
    var places: Dict[Int, List[Token]]  # State -> Tokens
    var max_concurrent: Int
    var max_depth: Int
    var timeout_ms: Float64
    var next_token_id: Int
    var transition_count: Dict[String, Int]
    var verbose: Bool
    
    fn __init__(
        inout self,
        max_concurrent: Int = 10,
        max_depth: Int = 3,
        timeout_ms: Float64 = 30000.0,  # 30 seconds
        verbose: Bool = False
    ):
        """
        Initialize Petri net
        
        Args:
            max_concurrent: Maximum concurrent query executions
            max_depth: Maximum recursion depth
            timeout_ms: Timeout for token execution (milliseconds)
            verbose: Print debug information
        """
        self.places = Dict[Int, List[Token]]()
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self.timeout_ms = timeout_ms
        self.next_token_id = 0
        self.transition_count = Dict[String, Int]()
        self.verbose = verbose
        
        # Initialize all places (states)
        for i in range(8):  # 8 states in RLMState enum
            self.places[i] = List[Token]()
    
    fn create_token(
        inout self,
        query: String,
        depth: Int,
        parent_id: Int = -1
    ) -> Token:
        """Create a new token with unique ID"""
        var token = Token(self.next_token_id, query, depth, parent_id)
        self.next_token_id += 1
        return token
    
    fn add_token(inout self, state: RLMState, token: Token):
        """Add token to a place (state)"""
        var state_int = int(state)
        self.places[state_int].append(token)
        
        if self.verbose:
            print("  [Petri Net] Token", token.id, "→", state_to_string(state))
    
    fn move_token(
        inout self,
        token: Token,
        from_state: RLMState,
        to_state: RLMState
    ) -> Bool:
        """
        Move token from one state to another (fire transition)
        
        Returns True if successful, False if token not found
        """
        var from_int = int(from_state)
        var to_int = int(to_state)
        
        # Find and remove token from source state
        var source_tokens = self.places[from_int]
        var found = False
        var found_idx = -1
        
        for i in range(len(source_tokens)):
            if source_tokens[i].id == token.id:
                found = True
                found_idx = i
                break
        
        if not found:
            return False
        
        # Remove from source
        var moved_token = source_tokens[found_idx]
        source_tokens.pop(found_idx)
        
        # Update token state
        moved_token.state = to_state
        
        # Add to destination
        self.places[to_int].append(moved_token)
        
        # Track transition
        var transition_name = state_to_string(from_state) + "_to_" + state_to_string(to_state)
        if transition_name in self.transition_count:
            self.transition_count[transition_name] += 1
        else:
            self.transition_count[transition_name] = 1
        
        if self.verbose:
            print("  [Transition]", state_to_string(from_state), "→", 
                  state_to_string(to_state), "| Token", token.id)
        
        return True
    
    fn get_tokens_in_state(self, state: RLMState) -> List[Token]:
        """Get all tokens in a specific state"""
        var state_int = int(state)
        return self.places[state_int]
    
    fn count_tokens_in_state(self, state: RLMState) -> Int:
        """Count tokens in a specific state"""
        var state_int = int(state)
        return len(self.places[state_int])
    
    fn can_spawn_query(self, depth: Int) -> Bool:
        """
        Check if we can spawn a new recursive query
        
        Considers:
        - Current concurrent executions
        - Depth limit
        """
        # Check depth limit
        if depth >= self.max_depth:
            if self.verbose:
                print("  [Limit] Depth limit reached:", depth, ">=", self.max_depth)
            return False
        
        # Check concurrent execution limit
        var executing = self.count_tokens_in_state(RLMState.EXECUTING_QUERIES)
        var waiting = self.count_tokens_in_state(RLMState.WAITING_FOR_RESULTS)
        var active = executing + waiting
        
        if active >= self.max_concurrent:
            if self.verbose:
                print("  [Limit] Concurrent limit reached:", active, ">=", self.max_concurrent)
            return False
        
        return True
    
    fn detect_deadlock(self) -> Bool:
        """
        Detect potential deadlock conditions
        
        Checks:
        - Circular dependencies (not yet implemented - needs call graph)
        - Starvation (tokens waiting too long)
        - Resource exhaustion
        """
        # Check for starvation
        for state_int in range(8):
            var tokens = self.places[state_int]
            for token in tokens:
                if token.is_expired(self.timeout_ms):
                    if self.verbose:
                        print("  [Deadlock] Token", token.id, "expired after",
                              now() - token.created_at, "ms")
                    return True
        
        # Check for resource exhaustion
        var total_active = 0
        total_active += self.count_tokens_in_state(RLMState.GENERATING)
        total_active += self.count_tokens_in_state(RLMState.EXECUTING_QUERIES)
        total_active += self.count_tokens_in_state(RLMState.WAITING_FOR_RESULTS)
        
        if total_active >= self.max_concurrent * 2:
            if self.verbose:
                print("  [Deadlock] Too many active tokens:", total_active)
            return True
        
        return False
    
    fn get_state_summary(self) -> String:
        """Get summary of current Petri net state for debugging"""
        var summary = "Petri Net State:\n"
        
        for i in range(8):
            var state = RLMState(i)
            var count = self.count_tokens_in_state(state)
            if count > 0:
                summary += "  " + state_to_string(state) + ": " + str(count) + " tokens\n"
        
        summary += "\nTransitions fired:\n"
        for transition_name in self.transition_count:
            var count = self.transition_count[transition_name]
            summary += "  " + transition_name + ": " + str(count) + "\n"
        
        return summary
    
    fn clear(inout self):
        """Clear all tokens from all states"""
        for i in range(8):
            self.places[i].clear()
        self.next_token_id = 0
        self.transition_count.clear()
    
    fn export_to_json(self) -> String:
        """
        Export Petri net state to JSON for visualization
        
        Useful for debugging and visual tools
        """
        var json = "{\n"
        json += '  "states": {\n'
        
        for i in range(8):
            var state = RLMState(i)
            var count = self.count_tokens_in_state(state)
            json += '    "' + state_to_string(state) + '": ' + str(count)
            if i < 7:
                json += ","
            json += "\n"
        
        json += "  },\n"
        json += '  "transitions": {\n'
        
        var first = True
        for transition_name in self.transition_count:
            if not first:
                json += ","
            first = False
            var count = self.transition_count[transition_name]
            json += '    "' + transition_name + '": ' + str(count) + "\n"
        
        json += "  },\n"
        json += '  "active_tokens": [\n'
        
        # Export token details
        first = True
        for i in range(8):
            var tokens = self.places[i]
            for token in tokens:
                if not first:
                    json += ","
                first = False
                json += "    {\n"
                json += '      "id": ' + str(token.id) + ",\n"
                json += '      "depth": ' + str(token.depth) + ",\n"
                json += '      "state": "' + state_to_string(token.state) + '",\n'
                json += '      "query": "' + token.query[:50] + '..."\n'
                json += "    }\n"
        
        json += "  ]\n"
        json += "}"
        
        return json


# ============================================================================
# Transition Guards (Conditions for state transitions)
# ============================================================================

fn can_transition_to_generating(petri_net: PetriNet) -> Bool:
    """Check if we can transition from IDLE to GENERATING"""
    return petri_net.count_tokens_in_state(RLMState.IDLE) > 0

fn can_transition_to_parsing(petri_net: PetriNet) -> Bool:
    """Check if we can transition from GENERATING to PARSING"""
    return petri_net.count_tokens_in_state(RLMState.GENERATING) > 0

fn can_transition_to_executing(petri_net: PetriNet) -> Bool:
    """Check if we can transition from PARSING to EXECUTING_QUERIES"""
    return petri_net.count_tokens_in_state(RLMState.PARSING) > 0

fn can_transition_to_combining(petri_net: PetriNet) -> Bool:
    """Check if we can transition to COMBINING_RESULTS"""
    # All child queries must be complete
    return petri_net.count_tokens_in_state(RLMState.WAITING_FOR_RESULTS) == 0


# ============================================================================
# Utility Functions
# ============================================================================

fn visualize_petri_net(petri_net: PetriNet):
    """Print ASCII visualization of Petri net state"""
    print("\n" + "=" * 60)
    print("🔄 Petri Net Visualization")
    print("=" * 60)
    
    var states = [
        RLMState.IDLE,
        RLMState.GENERATING,
        RLMState.PARSING,
        RLMState.EXECUTING_QUERIES,
        RLMState.WAITING_FOR_RESULTS,
        RLMState.COMBINING_RESULTS,
        RLMState.FINAL_ANSWER,
        RLMState.ERROR
    ]
    
    for state in states:
        var count = petri_net.count_tokens_in_state(state)
        var bar = "●" * count
        print(f"{state_to_string(state):25s} [{count:2d}] {bar}")
    
    print("=" * 60)
    print(f"Max Concurrent: {petri_net.max_concurrent}")
    print(f"Max Depth: {petri_net.max_depth}")
    print(f"Total Tokens: {petri_net.next_token_id}")
    print("=" * 60 + "\n")
