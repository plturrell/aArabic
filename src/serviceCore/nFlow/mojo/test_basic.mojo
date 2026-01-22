#!/usr/bin/env mojo
# Basic test for nWorkflow Mojo bindings
# Day 7: Simple integration test

from petri_net import (
    init_library,
    cleanup_library,
    PetriNet,
    PetriNetExecutor,
    ExecutionStrategy,
    ArcType,
    WorkflowBuilder,
    get_version,
)


fn test_basic_workflow() raises:
    """Test basic workflow creation and execution."""
    print("=== Test 1: Basic Workflow ===\n")
    
    # Create a simple workflow
    var net = PetriNet("Test Workflow")
    
    # Add places
    net.add_place("start", "Start Place")
    net.add_place("middle", "Middle Place")
    net.add_place("end", "End Place")
    
    # Add transitions
    net.add_transition("t1", "First Step", 0)
    net.add_transition("t2", "Second Step", 0)
    
    # Add arcs
    net.add_arc("a1", ArcType.input(), 1, "start", "t1")
    net.add_arc("a2", ArcType.output(), 1, "t1", "middle")
    net.add_arc("a3", ArcType.input(), 1, "middle", "t2")
    net.add_arc("a4", ArcType.output(), 1, "t2", "end")
    
    # Add initial token
    net.add_token("start", '{"task": "process"}')
    
    print("Initial state:")
    print("  Tokens in 'start':", net.get_place_token_count("start"))
    print("  Tokens in 'end':", net.get_place_token_count("end"))
    print("  Enabled transitions:", net.get_enabled_count())
    print("  Is deadlocked:", net.is_deadlocked())
    print()
    
    # Execute workflow
    var executor = PetriNetExecutor(net, ExecutionStrategy.sequential())
    executor.run_until_complete()
    
    print("After execution:")
    print("  Tokens in 'start':", net.get_place_token_count("start"))
    print("  Tokens in 'end':", net.get_place_token_count("end"))
    print("  Is deadlocked:", net.is_deadlocked())
    print()
    
    # Get statistics
    var stats = executor.get_stats_json()
    print("Execution Statistics:")
    print(stats)
    print()


fn test_fluent_api() raises:
    """Test the fluent API workflow builder."""
    print("=== Test 2: Fluent API ===\n")
    
    # Build workflow using fluent API
    var builder = WorkflowBuilder("Document Processing")
    builder = builder.place("inbox", "Input Queue")
    builder = builder.place("processing", "Processing")
    builder = builder.place("done", "Complete")
    builder = builder.transition("start", "Start")
    builder = builder.transition("finish", "Finish")
    builder = builder.flow("inbox", "start")
    builder = builder.flow("start", "processing")
    builder = builder.flow("processing", "finish")
    builder = builder.flow("finish", "done")
    builder = builder.token("inbox", '{"doc": "test.pdf"}')
    
    var workflow = builder.build()
    
    print("Workflow created with fluent API")
    print("  Tokens in 'inbox':", workflow.get_place_token_count("inbox"))
    print("  Enabled transitions:", workflow.get_enabled_count())
    print()
    
    # Execute
    var executor = PetriNetExecutor(workflow, ExecutionStrategy.sequential())
    executor.run_until_complete()
    
    print("After execution:")
    print("  Tokens in 'done':", workflow.get_place_token_count("done"))
    print()


fn test_concurrent_execution() raises:
    """Test concurrent execution strategy."""
    print("=== Test 3: Concurrent Execution ===\n")
    
    # Create a workflow with parallel branches
    var net = PetriNet("Parallel Workflow")
    
    # Two parallel branches
    net.add_place("start", "Start")
    net.add_place("branch1", "Branch 1")
    net.add_place("branch2", "Branch 2")
    net.add_place("end1", "End 1")
    net.add_place("end2", "End 2")
    
    net.add_transition("t1", "Process 1", 0)
    net.add_transition("t2", "Process 2", 0)
    
    # Branch 1
    net.add_arc("a1", ArcType.input(), 1, "start", "t1")
    net.add_arc("a2", ArcType.output(), 1, "t1", "end1")
    
    # Branch 2
    net.add_arc("a3", ArcType.input(), 1, "branch1", "t2")
    net.add_arc("a4", ArcType.output(), 1, "t2", "end2")
    
    # Add tokens to both branches
    net.add_token("start", '{"data": "A"}')
    net.add_token("branch1", '{"data": "B"}')
    
    print("Initial state:")
    print("  Enabled transitions:", net.get_enabled_count())
    print()
    
    # Use concurrent execution strategy
    var executor = PetriNetExecutor(net, ExecutionStrategy.concurrent())
    _ = executor.step()  # Should fire both transitions
    
    print("After concurrent step:")
    print("  Tokens in 'end1':", net.get_place_token_count("end1"))
    print("  Tokens in 'end2':", net.get_place_token_count("end2"))
    print()


fn main() raises:
    """Main entry point for tests."""
    print("\n" + "="*60)
    print("nWorkflow Mojo FFI Bindings - Basic Tests")
    print("="*60 + "\n")
    
    # Initialize library
    init_library()
    
    # Show version
    var version = get_version()
    print("nWorkflow version:", version)
    print()
    
    # Run tests
    test_basic_workflow()
    test_fluent_api()
    test_concurrent_execution()
    
    # Cleanup
    cleanup_library()
    
    print("="*60)
    print("All tests completed successfully!")
    print("="*60 + "\n")
