# Mojo FFI Bindings for nWorkflow Petri Net Engine
# Part of serviceCore nWorkflow
# Day 7: Pythonic API wrapping Zig C exports
#
# This module provides a high-level Mojo interface to the nWorkflow
# Petri Net engine, built on top of the Zig core via FFI.

from sys.ffi import DLHandle, external_call
from memory import UnsafePointer
from pathlib import Path


# Error codes matching c_api.zig ErrorCode enum
@value
struct ErrorCode:
    var code: Int
    
    alias SUCCESS = 0
    alias NULL_POINTER = 1
    alias ALLOCATION_FAILED = 2
    alias INVALID_ID = 3
    alias INVALID_PARAMETER = 4
    alias NOT_FOUND = 5
    alias ALREADY_EXISTS = 6
    alias DEADLOCK = 7
    alias UNKNOWN = 99
    
    fn __init__(inout self, code: Int):
        self.code = code
    
    fn is_success(self) -> Bool:
        return self.code == Self.SUCCESS
    
    fn __str__(self) -> String:
        if self.code == Self.SUCCESS:
            return "Success"
        elif self.code == Self.NULL_POINTER:
            return "Null pointer"
        elif self.code == Self.ALLOCATION_FAILED:
            return "Allocation failed"
        elif self.code == Self.INVALID_ID:
            return "Invalid ID"
        elif self.code == Self.INVALID_PARAMETER:
            return "Invalid parameter"
        elif self.code == Self.NOT_FOUND:
            return "Not found"
        elif self.code == Self.ALREADY_EXISTS:
            return "Already exists"
        elif self.code == Self.DEADLOCK:
            return "Deadlock"
        else:
            return "Unknown error"


# Execution strategy enum
@value
struct ExecutionStrategy:
    var value: UInt32
    
    alias SEQUENTIAL = 0
    alias CONCURRENT = 1
    alias PRIORITY_BASED = 2
    alias CUSTOM = 3
    
    fn __init__(inout self, value: UInt32):
        self.value = value
    
    @staticmethod
    fn sequential() -> Self:
        return ExecutionStrategy(Self.SEQUENTIAL)
    
    @staticmethod
    fn concurrent() -> Self:
        return ExecutionStrategy(Self.CONCURRENT)
    
    @staticmethod
    fn priority_based() -> Self:
        return ExecutionStrategy(Self.PRIORITY_BASED)


# Conflict resolution enum
@value
struct ConflictResolution:
    var value: UInt32
    
    alias PRIORITY = 0
    alias RANDOM = 1
    alias ROUND_ROBIN = 2
    alias WEIGHTED_RANDOM = 3
    
    fn __init__(inout self, value: UInt32):
        self.value = value
    
    @staticmethod
    fn priority() -> Self:
        return ConflictResolution(Self.PRIORITY)
    
    @staticmethod
    fn random() -> Self:
        return ConflictResolution(Self.RANDOM)
    
    @staticmethod
    fn round_robin() -> Self:
        return ConflictResolution(Self.ROUND_ROBIN)
    
    @staticmethod
    fn weighted_random() -> Self:
        return ConflictResolution(Self.WEIGHTED_RANDOM)


# Arc type enum
@value
struct ArcType:
    var value: UInt32
    
    alias INPUT = 0
    alias OUTPUT = 1
    alias INHIBITOR = 2
    
    fn __init__(inout self, value: UInt32):
        self.value = value
    
    @staticmethod
    fn input() -> Self:
        return ArcType(Self.INPUT)
    
    @staticmethod
    fn output() -> Self:
        return ArcType(Self.OUTPUT)
    
    @staticmethod
    fn inhibitor() -> Self:
        return ArcType(Self.INHIBITOR)


# Global library handle
var _lib_handle: DLHandle


fn _get_lib_path() -> String:
    """Get the path to the nWorkflow shared library."""
    # Assume library is in zig-out/lib relative to this file
    return "zig-out/lib/libnworkflow.dylib"


fn init_library() raises:
    """Initialize the nWorkflow library. Call this once at startup."""
    global _lib_handle
    var lib_path = _get_lib_path()
    _lib_handle = DLHandle(lib_path)
    
    # Call nworkflow_init
    var init_fn = _lib_handle.get_function[fn() -> Int]("nworkflow_init")
    var result = init_fn()
    if result != ErrorCode.SUCCESS:
        raise Error("Failed to initialize nWorkflow library")


fn cleanup_library() raises:
    """Cleanup the nWorkflow library. Call this once at shutdown."""
    var cleanup_fn = _lib_handle.get_function[fn() -> Int]("nworkflow_cleanup")
    var result = cleanup_fn()
    if result != ErrorCode.SUCCESS:
        raise Error("Failed to cleanup nWorkflow library")


@value
struct PetriNet:
    """
    A Petri Net is a mathematical model for distributed systems.
    
    It consists of:
    - Places: Hold tokens (represent state)
    - Transitions: Process tokens (represent actions)
    - Arcs: Connect places and transitions (represent flow)
    - Tokens: Carry data through the net
    
    Example:
        var net = PetriNet("My Workflow")
        net.add_place("start", "Start Place")
        net.add_place("end", "End Place")
        net.add_transition("process", "Process Data", 0)
        net.add_arc("a1", ArcType.input(), 1, "start", "process")
        net.add_arc("a2", ArcType.output(), 1, "process", "end")
        net.add_token("start", "{}")
        net.fire_transition("process")
    """
    var _handle: UInt64
    
    fn __init__(inout self, name: String) raises:
        """Create a new Petri Net with the given name."""
        var create_fn = _lib_handle.get_function[fn(UnsafePointer[UInt8]) -> UInt64]("nworkflow_create_net")
        self._handle = create_fn(name.unsafe_ptr())
        if self._handle == 0:
            raise Error("Failed to create Petri Net")
    
    fn __del__(owned self):
        """Destroy the Petri Net and free its resources."""
        var destroy_fn = _lib_handle.get_function[fn(UInt64) -> Int]("nworkflow_destroy_net")
        _ = destroy_fn(self._handle)
    
    fn add_place(inout self, place_id: String, name: String, capacity: Int = -1) raises:
        """Add a place to the Petri Net.
        
        Args:
            place_id: Unique identifier for the place
            name: Human-readable name
            capacity: Maximum tokens (-1 for unlimited)
        """
        var add_fn = _lib_handle.get_function[
            fn(UInt64, UnsafePointer[UInt8], UnsafePointer[UInt8], Int32) -> Int
        ]("nworkflow_add_place")
        
        var result = add_fn(
            self._handle,
            place_id.unsafe_ptr(),
            name.unsafe_ptr(),
            capacity
        )
        
        if result != ErrorCode.SUCCESS:
            raise Error("Failed to add place: " + ErrorCode(result).__str__())
    
    fn add_transition(inout self, transition_id: String, name: String, priority: Int = 0) raises:
        """Add a transition to the Petri Net.
        
        Args:
            transition_id: Unique identifier for the transition
            name: Human-readable name
            priority: Priority for conflict resolution (higher = more important)
        """
        var add_fn = _lib_handle.get_function[
            fn(UInt64, UnsafePointer[UInt8], UnsafePointer[UInt8], Int32) -> Int
        ]("nworkflow_add_transition")
        
        var result = add_fn(
            self._handle,
            transition_id.unsafe_ptr(),
            name.unsafe_ptr(),
            priority
        )
        
        if result != ErrorCode.SUCCESS:
            raise Error("Failed to add transition: " + ErrorCode(result).__str__())
    
    fn add_arc(
        inout self,
        arc_id: String,
        arc_type: ArcType,
        weight: Int,
        source_id: String,
        target_id: String
    ) raises:
        """Add an arc connecting a place and transition.
        
        Args:
            arc_id: Unique identifier for the arc
            arc_type: Type of arc (input, output, inhibitor)
            weight: Number of tokens consumed/produced
            source_id: ID of source (place or transition)
            target_id: ID of target (transition or place)
        """
        var add_fn = _lib_handle.get_function[
            fn(UInt64, UnsafePointer[UInt8], UInt32, UInt32, UnsafePointer[UInt8], UnsafePointer[UInt8]) -> Int
        ]("nworkflow_add_arc")
        
        var result = add_fn(
            self._handle,
            arc_id.unsafe_ptr(),
            arc_type.value,
            weight,
            source_id.unsafe_ptr(),
            target_id.unsafe_ptr()
        )
        
        if result != ErrorCode.SUCCESS:
            raise Error("Failed to add arc: " + ErrorCode(result).__str__())
    
    fn add_token(inout self, place_id: String, data: String = "{}") raises:
        """Add a token to a place.
        
        Args:
            place_id: ID of the place to add token to
            data: JSON data associated with the token
        """
        var add_fn = _lib_handle.get_function[
            fn(UInt64, UnsafePointer[UInt8], UnsafePointer[UInt8]) -> Int
        ]("nworkflow_add_token")
        
        var result = add_fn(
            self._handle,
            place_id.unsafe_ptr(),
            data.unsafe_ptr()
        )
        
        if result != ErrorCode.SUCCESS:
            raise Error("Failed to add token: " + ErrorCode(result).__str__())
    
    fn fire_transition(inout self, transition_id: String) raises:
        """Fire a transition, moving tokens through the net.
        
        Args:
            transition_id: ID of the transition to fire
        """
        var fire_fn = _lib_handle.get_function[
            fn(UInt64, UnsafePointer[UInt8]) -> Int
        ]("nworkflow_fire_transition")
        
        var result = fire_fn(self._handle, transition_id.unsafe_ptr())
        
        if result != ErrorCode.SUCCESS:
            raise Error("Failed to fire transition: " + ErrorCode(result).__str__())
    
    fn is_deadlocked(self) -> Bool:
        """Check if the net is in deadlock (no enabled transitions).
        
        Returns:
            True if deadlocked, False otherwise
        """
        var check_fn = _lib_handle.get_function[fn(UInt64) -> Bool]("nworkflow_is_deadlocked")
        return check_fn(self._handle)
    
    fn get_enabled_count(self) -> Int:
        """Get the number of currently enabled transitions.
        
        Returns:
            Number of enabled transitions (-1 on error)
        """
        var count_fn = _lib_handle.get_function[fn(UInt64) -> Int32]("nworkflow_get_enabled_count")
        return int(count_fn(self._handle))
    
    fn get_place_token_count(self, place_id: String) -> Int:
        """Get the number of tokens in a place.
        
        Args:
            place_id: ID of the place
            
        Returns:
            Number of tokens (-1 on error)
        """
        var count_fn = _lib_handle.get_function[
            fn(UInt64, UnsafePointer[UInt8]) -> Int32
        ]("nworkflow_get_place_token_count")
        return int(count_fn(self._handle, place_id.unsafe_ptr()))


@value
struct PetriNetExecutor:
    """
    Executor for running Petri Nets with different execution strategies.
    
    Supports:
    - Sequential execution (one transition at a time)
    - Concurrent execution (all enabled transitions in parallel)
    - Priority-based execution (highest priority first)
    - Custom execution strategies
    
    Example:
        var net = PetriNet("My Workflow")
        # ... build the net ...
        var executor = PetriNetExecutor(net, ExecutionStrategy.sequential())
        executor.run_until_complete()
        var stats = executor.get_stats_json()
        print(stats)
    """
    var _handle: UInt64
    var _net_handle: UInt64  # Keep reference to net
    
    fn __init__(inout self, net: PetriNet, strategy: ExecutionStrategy) raises:
        """Create an executor for the given Petri Net.
        
        Args:
            net: The Petri Net to execute
            strategy: Execution strategy to use
        """
        self._net_handle = net._handle
        
        var create_fn = _lib_handle.get_function[
            fn(UInt64, UInt32) -> UInt64
        ]("nworkflow_create_executor")
        
        self._handle = create_fn(self._net_handle, strategy.value)
        if self._handle == 0:
            raise Error("Failed to create executor")
    
    fn __del__(owned self):
        """Destroy the executor and free its resources."""
        var destroy_fn = _lib_handle.get_function[fn(UInt64) -> Int]("nworkflow_destroy_executor")
        _ = destroy_fn(self._handle)
    
    fn step(inout self) raises -> Bool:
        """Execute one step (fire one or more transitions based on strategy).
        
        Returns:
            True if execution continued, False if deadlocked
        """
        var step_fn = _lib_handle.get_function[fn(UInt64) -> Bool]("nworkflow_executor_step")
        return step_fn(self._handle)
    
    fn run(inout self, max_steps: Int) raises:
        """Run for a maximum number of steps or until deadlock.
        
        Args:
            max_steps: Maximum number of steps to execute
        """
        var run_fn = _lib_handle.get_function[
            fn(UInt64, UInt64) -> Int
        ]("nworkflow_executor_run")
        
        var result = run_fn(self._handle, max_steps)
        if result != ErrorCode.SUCCESS:
            raise Error("Execution failed: " + ErrorCode(result).__str__())
    
    fn run_until_complete(inout self) raises:
        """Run until no transitions are enabled (deadlock or completion).
        
        This is the most common execution mode for complete workflow execution.
        """
        var run_fn = _lib_handle.get_function[fn(UInt64) -> Int]("nworkflow_executor_run_until_complete")
        
        var result = run_fn(self._handle)
        if result != ErrorCode.SUCCESS:
            raise Error("Execution failed: " + ErrorCode(result).__str__())
    
    fn set_conflict_resolution(inout self, resolution: ConflictResolution) raises:
        """Set the conflict resolution strategy.
        
        Args:
            resolution: Strategy for resolving conflicts between enabled transitions
        """
        var set_fn = _lib_handle.get_function[
            fn(UInt64, UInt32) -> Int
        ]("nworkflow_executor_set_conflict_resolution")
        
        var result = set_fn(self._handle, resolution.value)
        if result != ErrorCode.SUCCESS:
            raise Error("Failed to set conflict resolution: " + ErrorCode(result).__str__())
    
    fn get_stats_json(self) raises -> String:
        """Get execution statistics as JSON.
        
        Returns:
            JSON string with execution metrics (steps, transitions fired, timing, etc.)
        """
        # Allocate buffer for JSON string
        var buffer = UnsafePointer[UInt8].alloc(4096)
        
        var stats_fn = _lib_handle.get_function[
            fn(UInt64, UnsafePointer[UInt8], UInt64) -> Int
        ]("nworkflow_executor_get_stats_json")
        
        var result = stats_fn(self._handle, buffer, 4096)
        if result != ErrorCode.SUCCESS:
            buffer.free()
            raise Error("Failed to get stats: " + ErrorCode(result).__str__())
        
        # Convert to string
        var json_str = String(buffer)
        buffer.free()
        return json_str


# Fluent API for workflow building
@value
struct WorkflowBuilder:
    """
    Fluent API for building Petri Net workflows in a Pythonic way.
    
    Example:
        var workflow = (
            WorkflowBuilder("Document Processing")
            .place("inbox", "Input Queue")
            .place("processing", "Processing")
            .place("done", "Complete")
            .transition("start", "Start")
            .transition("finish", "Finish")
            .flow("inbox", "start")
            .flow("start", "processing")
            .flow("processing", "finish")
            .flow("finish", "done")
            .token("inbox", '{"doc": "test.pdf"}')
            .build()
        )
        
        var executor = PetriNetExecutor(workflow, ExecutionStrategy.sequential())
        executor.run_until_complete()
        print("Workflow complete!")
    """
    var net: PetriNet
    var arc_counter: Int
    
    fn __init__(inout self, name: String) raises:
        """Create a workflow builder with the given name."""
        self.net = PetriNet(name)
        self.arc_counter = 0
    
    fn place(inout self, id: String, name: String, capacity: Int = -1) raises -> Self:
        """Add a place to the workflow."""
        self.net.add_place(id, name, capacity)
        return self
    
    fn transition(inout self, id: String, name: String, priority: Int = 0) raises -> Self:
        """Add a transition to the workflow."""
        self.net.add_transition(id, name, priority)
        return self
    
    fn flow(inout self, from_id: String, to_id: String, weight: Int = 1) raises -> Self:
        """Add a flow (arc) between two nodes."""
        var arc_id = "arc_" + str(self.arc_counter)
        self.arc_counter += 1
        self.net.add_arc(arc_id, ArcType.input(), weight, from_id, to_id)
        return self
    
    fn token(inout self, place_id: String, data: String = "{}") raises -> Self:
        """Add a token to a place."""
        self.net.add_token(place_id, data)
        return self
    
    fn build(owned self) -> PetriNet:
        """Build and return the Petri Net."""
        return self.net


fn get_version() raises -> String:
    """Get the version of the nWorkflow library."""
    var buffer = UnsafePointer[UInt8].alloc(256)
    
    var version_fn = _lib_handle.get_function[
        fn(UnsafePointer[UInt8], UInt64) -> Int
    ]("nworkflow_get_version")
    
    var result = version_fn(buffer, 256)
    if result != ErrorCode.SUCCESS:
        buffer.free()
        raise Error("Failed to get version")
    
    var version_str = String(buffer)
    buffer.free()
    return version_str
