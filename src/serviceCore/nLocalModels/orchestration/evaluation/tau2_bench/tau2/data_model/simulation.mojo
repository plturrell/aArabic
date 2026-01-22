# simulation.mojo
# Migrated from simulation.py
# Simulation data structures for TAU2-Bench

from collections import Dict, List
from utils.utils import get_now

struct SimulationConfig:
    """Configuration for a simulation run"""
    var domain: String
    var task_id: String
    var agent_type: String
    var max_turns: Int
    var timeout_seconds: Int
    var enable_metrics: Bool
    var environment_config: Dict[String, String]
    
    fn __init__(out self):
        self.domain = ""
        self.task_id = ""
        self.agent_type = "local"
        self.max_turns = 20
        self.timeout_seconds = 300
        self.enable_metrics = True
        self.environment_config = Dict[String, String]()
    
    fn __init__(out self, domain: String, task_id: String):
        self.domain = domain
        self.task_id = task_id
        self.agent_type = "local"
        self.max_turns = 20
        self.timeout_seconds = 300
        self.enable_metrics = True
        self.environment_config = Dict[String, String]()

struct SimulationState:
    """Current state of a simulation"""
    var simulation_id: String
    var status: String  # "running", "completed", "failed", "timeout"
    var current_turn: Int
    var start_time: String
    var end_time: String
    var error_message: String
    
    fn __init__(out self):
        self.simulation_id = ""
        self.status = "initialized"
        self.current_turn = 0
        self.start_time = get_now()
        self.end_time = ""
        self.error_message = ""
    
    fn __init__(out self, simulation_id: String):
        self.simulation_id = simulation_id
        self.status = "initialized"
        self.current_turn = 0
        self.start_time = get_now()
        self.end_time = ""
        self.error_message = ""
    
    fn is_running(self) -> Bool:
        """Check if simulation is currently running"""
        return self.status == "running"
    
    fn is_completed(self) -> Bool:
        """Check if simulation completed successfully"""
        return self.status == "completed"
    
    fn is_failed(self) -> Bool:
        """Check if simulation failed or timed out"""
        return self.status == "failed" or self.status == "timeout"
    
    fn mark_running(mut self):
        """Mark simulation as running"""
        self.status = "running"
    
    fn mark_completed(mut self):
        """Mark simulation as completed"""
        self.status = "completed"
        self.end_time = get_now()
    
    fn mark_failed(mut self, error: String):
        """Mark simulation as failed with error message"""
        self.status = "failed"
        self.error_message = error
        self.end_time = get_now()
    
    fn mark_timeout(mut self):
        """Mark simulation as timed out"""
        self.status = "timeout"
        self.error_message = "Simulation exceeded timeout limit"
        self.end_time = get_now()
    
    fn increment_turn(mut self):
        """Increment the current turn counter"""
        self.current_turn += 1

struct SimulationResult:
    """Results from a completed simulation"""
    var simulation_id: String
    var config: SimulationConfig
    var state: SimulationState
    var total_turns: Int
    var success: Bool
    var completion_reason: String
    var metrics: Dict[String, String]
    
    fn __init__(out self):
        self.simulation_id = ""
        self.config = SimulationConfig()
        self.state = SimulationState()
        self.total_turns = 0
        self.success = False
        self.completion_reason = ""
        self.metrics = Dict[String, String]()
    
    fn __init__(out self, simulation_id: String, config: SimulationConfig, state: SimulationState):
        self.simulation_id = simulation_id
        self.config = config
        self.state = state
        self.total_turns = state.current_turn
        self.success = state.is_completed()
        self.completion_reason = state.status
        self.metrics = Dict[String, String]()
    
    fn add_metric(mut self, key: String, value: String):
        """Add a metric to the results"""
        self.metrics[key] = value
    
    fn get_metric(self, key: String) -> String:
        """Get a metric value"""
        if key in self.metrics:
            return self.metrics[key]
        return ""

struct TurnRecord:
    """Record of a single turn in the simulation"""
    var turn_number: Int
    var timestamp: String
    var agent_action: String
    var environment_response: String
    var tool_calls: List[String]
    var duration_ms: Int
    
    fn __init__(out self):
        self.turn_number = 0
        self.timestamp = get_now()
        self.agent_action = ""
        self.environment_response = ""
        self.tool_calls = List[String]()
        self.duration_ms = 0
    
    fn __init__(out self, turn_number: Int):
        self.turn_number = turn_number
        self.timestamp = get_now()
        self.agent_action = ""
        self.environment_response = ""
        self.tool_calls = List[String]()
        self.duration_ms = 0
    
    fn add_tool_call(mut self, tool_name: String):
        """Add a tool call to this turn"""
        self.tool_calls.append(tool_name)
    
    fn get_tool_count(self) -> Int:
        """Get number of tools called in this turn"""
        return len(self.tool_calls)

struct SimulationTrace:
    """Complete trace of a simulation execution"""
    var simulation_id: String
    var turns: List[TurnRecord]
    var metadata: Dict[String, String]
    
    fn __init__(out self):
        self.simulation_id = ""
        self.turns = List[TurnRecord]()
        self.metadata = Dict[String, String]()
    
    fn __init__(out self, simulation_id: String):
        self.simulation_id = simulation_id
        self.turns = List[TurnRecord]()
        self.metadata = Dict[String, String]()
    
    fn add_turn(mut self, turn: TurnRecord):
        """Add a turn record to the trace"""
        self.turns.append(turn)
    
    fn get_turn_count(self) -> Int:
        """Get total number of turns"""
        return len(self.turns)
    
    fn add_metadata(mut self, key: String, value: String):
        """Add metadata to the trace"""
        self.metadata[key] = value
