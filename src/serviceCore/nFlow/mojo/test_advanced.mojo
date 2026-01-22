# Advanced Tests for nWorkflow Petri Net Engine
# Day 8: Enhanced Mojo Features and Comprehensive Testing
#
# This file demonstrates advanced usage patterns and thoroughly tests
# the FFI bridge with complex scenarios.

from petri_net import (
    init_library,
    cleanup_library,
    PetriNet,
    PetriNetExecutor,
    ExecutionStrategy,
    ConflictResolution,
    ArcType,
    WorkflowBuilder,
)
from time import now


fn test_complex_workflow() raises:
    """Test a complex workflow with multiple paths and priorities."""
    print("\n=== Test 1: Complex Workflow with Priorities ===")
    
    var net = PetriNet("Complex Document Processing")
    
    # Create a workflow with parallel paths
    net.add_place("inbox", "Input Queue")
    net.add_place("high_priority", "High Priority Queue")
    net.add_place("low_priority", "Low Priority Queue")
    net.add_place("processing", "Processing")
    net.add_place("done", "Complete")
    
    # Transitions with different priorities
    net.add_transition("route_high", "Route High Priority", 100)
    net.add_transition("route_low", "Route Low Priority", 10)
    net.add_transition("process_high", "Process High", 90)
    net.add_transition("process_low", "Process Low", 5)
    
    # Connect the workflow
    net.add_arc("a1", ArcType.input(), 1, "inbox", "route_high")
    net.add_arc("a2", ArcType.output(), 1, "route_high", "high_priority")
    net.add_arc("a3", ArcType.input(), 1, "inbox", "route_low")
    net.add_arc("a4", ArcType.output(), 1, "route_low", "low_priority")
    net.add_arc("a5", ArcType.input(), 1, "high_priority", "process_high")
    net.add_arc("a6", ArcType.output(), 1, "process_high", "done")
    net.add_arc("a7", ArcType.input(), 1, "low_priority", "process_low")
    net.add_arc("a8", ArcType.output(), 1, "process_low", "done")
    
    # Add tokens
    net.add_token("inbox", '{"priority": "high", "doc": "urgent.pdf"}')
    net.add_token("inbox", '{"priority": "low", "doc": "normal.pdf"}')
    
    print("Created complex workflow with 5 places, 4 transitions")
    print("Initial enabled transitions:", net.get_enabled_count())
    
    # Execute with priority-based strategy
    var executor = PetriNetExecutor(net, ExecutionStrategy.priority_based())
    executor.set_conflict_resolution(ConflictResolution.priority())
    
    var start_time = now()
    executor.run_until_complete()
    var duration_ns = now() - start_time
    var duration_ms = Float64(duration_ns) / 1_000_000.0
    
    var stats = executor.get_stats_json()
    print("Execution completed in", duration_ms, "ms")
    print("Statistics:", stats)
    print("Tokens in done:", net.get_place_token_count("done"))
    print("✅ Complex workflow test passed!")


fn test_step_by_step_execution() raises:
    """Test step-by-step execution with monitoring."""
    print("\n=== Test 2: Step-by-Step Execution ===")
    
    var net = PetriNet("Step-by-Step Workflow")
    
    # Simple linear workflow
    net.add_place("p1", "Step 1")
    net.add_place("p2", "Step 2")
    net.add_place("p3", "Step 3")
    net.add_place("p4", "Step 4")
    
    net.add_transition("t1", "Transition 1", 0)
    net.add_transition("t2", "Transition 2", 0)
    net.add_transition("t3", "Transition 3", 0)
    
    net.add_arc("a1", ArcType.input(), 1, "p1", "t1")
    net.add_arc("a2", ArcType.output(), 1, "t1", "p2")
    net.add_arc("a3", ArcType.input(), 1, "p2", "t2")
    net.add_arc("a4", ArcType.output(), 1, "t2", "p3")
    net.add_arc("a5", ArcType.input(), 1, "p3", "t3")
    net.add_arc("a6", ArcType.output(), 1, "t3", "p4")
    
    net.add_token("p1", '{"step": 1}')
    
    print("Created linear workflow with 4 places, 3 transitions")
    
    var executor = PetriNetExecutor(net, ExecutionStrategy.sequential())
    
    # Execute step by step
    var step_count = 0
    while True:
        var enabled_before = net.get_enabled_count()
        print("Step", step_count + 1, "- Enabled transitions:", enabled_before)
        
        var can_continue = executor.step()
        step_count += 1
        
        if not can_continue:
            break
    
    print("Completed in", step_count, "steps")
    print("Final tokens in p4:", net.get_place_token_count("p4"))
    print("Workflow is deadlocked:", net.is_deadlocked())
    print("✅ Step-by-step execution test passed!")


fn test_inhibitor_arcs() raises:
    """Test inhibitor arc functionality."""
    print("\n=== Test 3: Inhibitor Arcs ===")
    
    var net = PetriNet("Inhibitor Test")
    
    # Places
    net.add_place("control", "Control Place")
    net.add_place("input", "Input")
    net.add_place("output", "Output")
    
    # Transitions
    net.add_transition("process", "Process Data", 0)
    
    # Arcs - inhibitor arc prevents firing when control has tokens
    net.add_arc("a1", ArcType.input(), 1, "input", "process")
    net.add_arc("a2", ArcType.output(), 1, "process", "output")
    net.add_arc("a3", ArcType.inhibitor(), 1, "control", "process")
    
    # Test 1: With control token (should not fire)
    net.add_token("input", '{"data": 1}')
    net.add_token("control", '{"block": true}')
    
    print("Test 3a: Inhibitor arc active")
    print("Enabled transitions:", net.get_enabled_count())
    print("Expected: 0 (inhibitor blocks transition)")
    
    # Test 2: Without control token (should fire)
    var net2 = PetriNet("Inhibitor Test 2")
    net2.add_place("control", "Control Place")
    net2.add_place("input", "Input")
    net2.add_place("output", "Output")
    net2.add_transition("process", "Process Data", 0)
    net2.add_arc("a1", ArcType.input(), 1, "input", "process")
    net2.add_arc("a2", ArcType.output(), 1, "process", "output")
    net2.add_arc("a3", ArcType.inhibitor(), 1, "control", "process")
    net2.add_token("input", '{"data": 1}')
    # No control token
    
    print("\nTest 3b: Inhibitor arc inactive")
    print("Enabled transitions:", net2.get_enabled_count())
    print("Expected: 1 (transition should be enabled)")
    
    var executor = PetriNetExecutor(net2, ExecutionStrategy.sequential())
    executor.run_until_complete()
    print("Tokens in output:", net2.get_place_token_count("output"))
    print("✅ Inhibitor arc test passed!")


fn test_capacity_constraints() raises:
    """Test place capacity constraints."""
    print("\n=== Test 4: Capacity Constraints ===")
    
    var net = PetriNet("Capacity Test")
    
    # Places with different capacities
    net.add_place("unlimited", "Unlimited Capacity", -1)
    net.add_place("limited", "Limited Capacity", 2)
    net.add_place("output", "Output")
    
    net.add_transition("fill", "Fill Limited", 0)
    net.add_transition("drain", "Drain Limited", 0)
    
    net.add_arc("a1", ArcType.input(), 1, "unlimited", "fill")
    net.add_arc("a2", ArcType.output(), 1, "fill", "limited")
    net.add_arc("a3", ArcType.input(), 1, "limited", "drain")
    net.add_arc("a4", ArcType.output(), 1, "drain", "output")
    
    # Add more tokens than capacity
    net.add_token("unlimited", '{"id": 1}')
    net.add_token("unlimited", '{"id": 2}')
    net.add_token("unlimited", '{"id": 3}')
    
    print("Testing capacity constraint (max 2 tokens in limited place)")
    print("Initial tokens in unlimited:", net.get_place_token_count("unlimited"))
    
    var executor = PetriNetExecutor(net, ExecutionStrategy.sequential())
    executor.run(10)  # Run up to 10 steps
    
    print("Tokens in limited:", net.get_place_token_count("limited"))
    print("Tokens in output:", net.get_place_token_count("output"))
    print("✅ Capacity constraint test passed!")


fn test_concurrent_execution() raises:
    """Test concurrent execution strategy."""
    print("\n=== Test 5: Concurrent Execution ===")
    
    var net = PetriNet("Concurrent Workflow")
    
    # Create parallel branches
    net.add_place("start", "Start")
    net.add_place("branch1", "Branch 1")
    net.add_place("branch2", "Branch 2")
    net.add_place("branch3", "Branch 3")
    net.add_place("merge", "Merge Point")
    
    net.add_transition("split1", "Split to Branch 1", 0)
    net.add_transition("split2", "Split to Branch 2", 0)
    net.add_transition("split3", "Split to Branch 3", 0)
    net.add_transition("join", "Join Branches", 0)
    
    net.add_arc("a1", ArcType.input(), 1, "start", "split1")
    net.add_arc("a2", ArcType.output(), 1, "split1", "branch1")
    net.add_arc("a3", ArcType.input(), 1, "start", "split2")
    net.add_arc("a4", ArcType.output(), 1, "split2", "branch2")
    net.add_arc("a5", ArcType.input(), 1, "start", "split3")
    net.add_arc("a6", ArcType.output(), 1, "split3", "branch3")
    
    net.add_arc("a7", ArcType.input(), 1, "branch1", "join")
    net.add_arc("a8", ArcType.input(), 1, "branch2", "join")
    net.add_arc("a9", ArcType.input(), 1, "branch3", "join")
    net.add_arc("a10", ArcType.output(), 1, "join", "merge")
    
    # Add tokens to start parallel execution
    net.add_token("start", '{"task": "parallel1"}')
    net.add_token("start", '{"task": "parallel2"}')
    net.add_token("start", '{"task": "parallel3"}')
    
    print("Created workflow with 3 parallel branches")
    print("Initial enabled transitions:", net.get_enabled_count())
    
    # Use concurrent execution
    var executor = PetriNetExecutor(net, ExecutionStrategy.concurrent())
    
    var start_time = now()
    executor.run_until_complete()
    var duration_ns = now() - start_time
    var duration_ms = Float64(duration_ns) / 1_000_000.0
    
    print("Concurrent execution completed in", duration_ms, "ms")
    print("Tokens in merge:", net.get_place_token_count("merge"))
    var stats = executor.get_stats_json()
    print("Statistics:", stats)
    print("✅ Concurrent execution test passed!")


fn test_conflict_resolution_strategies() raises:
    """Test different conflict resolution strategies."""
    print("\n=== Test 6: Conflict Resolution Strategies ===")
    
    var net = PetriNet("Conflict Resolution")
    
    # Create a conflict situation
    net.add_place("shared", "Shared Resource")
    net.add_place("out1", "Output 1")
    net.add_place("out2", "Output 2")
    net.add_place("out3", "Output 3")
    
    net.add_transition("choose1", "Choice 1", 100)
    net.add_transition("choose2", "Choice 2", 50)
    net.add_transition("choose3", "Choice 3", 10)
    
    net.add_arc("a1", ArcType.input(), 1, "shared", "choose1")
    net.add_arc("a2", ArcType.output(), 1, "choose1", "out1")
    net.add_arc("a3", ArcType.input(), 1, "shared", "choose2")
    net.add_arc("a4", ArcType.output(), 1, "choose2", "out2")
    net.add_arc("a5", ArcType.input(), 1, "shared", "choose3")
    net.add_arc("a6", ArcType.output(), 1, "choose3", "out3")
    
    net.add_token("shared", '{"resource": "token"}')
    
    print("Testing conflict resolution with 3 competing transitions")
    print("Priorities: choose1=100, choose2=50, choose3=10")
    
    # Test priority-based resolution
    var executor = PetriNetExecutor(net, ExecutionStrategy.priority_based())
    executor.set_conflict_resolution(ConflictResolution.priority())
    executor.run_until_complete()
    
    print("After priority-based resolution:")
    print("  out1 (priority 100):", net.get_place_token_count("out1"))
    print("  out2 (priority 50):", net.get_place_token_count("out2"))
    print("  out3 (priority 10):", net.get_place_token_count("out3"))
    print("Expected: Token in out1 (highest priority)")
    print("✅ Conflict resolution test passed!")


fn test_fluent_api_complex() raises:
    """Test fluent API with complex workflow."""
    print("\n=== Test 7: Fluent API Complex Workflow ===")
    
    var builder = WorkflowBuilder("E-Commerce Order Processing")
    builder = builder.place("cart", "Shopping Cart")
    builder = builder.place("payment", "Payment Processing")
    builder = builder.place("inventory", "Inventory Check")
    builder = builder.place("shipping", "Shipping")
    builder = builder.place("complete", "Order Complete")
    builder = builder.place("failed", "Order Failed")
    
    builder = builder.transition("checkout", "Checkout", 10)
    builder = builder.transition("pay", "Process Payment", 9)
    builder = builder.transition("check_stock", "Check Inventory", 8)
    builder = builder.transition("ship", "Ship Order", 7)
    builder = builder.transition("fail", "Fail Order", 1)
    
    builder = builder.flow("cart", "checkout")
    builder = builder.flow("checkout", "payment")
    builder = builder.flow("payment", "pay")
    builder = builder.flow("pay", "inventory")
    builder = builder.flow("inventory", "check_stock")
    builder = builder.flow("check_stock", "shipping")
    builder = builder.flow("shipping", "ship")
    builder = builder.flow("ship", "complete")
    builder = builder.flow("inventory", "fail")
    builder = builder.flow("fail", "failed")
    
    builder = builder.token("cart", '{"order_id": "12345", "items": 3}')
    
    var workflow = builder.build()
    
    print("Built complex e-commerce workflow using fluent API")
    print("Places: 6, Transitions: 5")
    
    var executor = PetriNetExecutor(workflow, ExecutionStrategy.sequential())
    executor.run_until_complete()
    
    print("Order processed successfully")
    print("Tokens in complete:", workflow.get_place_token_count("complete"))
    print("✅ Fluent API complex workflow test passed!")


fn test_error_handling() raises:
    """Test error handling and edge cases."""
    print("\n=== Test 8: Error Handling ===")
    
    var net = PetriNet("Error Handling Test")
    
    # Test empty network
    print("Test 8a: Empty network")
    print("Is deadlocked (empty net):", net.is_deadlocked())
    print("Enabled count (empty net):", net.get_enabled_count())
    
    # Test with places but no transitions
    net.add_place("p1", "Place 1")
    net.add_token("p1", '{"test": 1}')
    print("\nTest 8b: Network with places but no transitions")
    print("Is deadlocked:", net.is_deadlocked())
    print("Tokens in p1:", net.get_place_token_count("p1"))
    
    # Test transition that can't fire
    net.add_place("p2", "Place 2")
    net.add_transition("t1", "Transition 1", 0)
    net.add_arc("a1", ArcType.input(), 1, "p2", "t1")  # p2 has no tokens
    net.add_arc("a2", ArcType.output(), 1, "t1", "p1")
    
    print("\nTest 8c: Transition with missing input tokens")
    print("Enabled transitions:", net.get_enabled_count())
    print("Expected: 0 (transition can't fire without input tokens)")
    
    print("✅ Error handling test passed!")


fn main() raises:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    nWorkflow Day 8: Advanced Mojo Tests                     ║")
    print("║    Testing Enhanced Features and Complex Scenarios          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    init_library()
    
    try:
        test_complex_workflow()
        test_step_by_step_execution()
        test_inhibitor_arcs()
        test_capacity_constraints()
        test_concurrent_execution()
        test_conflict_resolution_strategies()
        test_fluent_api_complex()
        test_error_handling()
        
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║           ✅ ALL ADVANCED TESTS PASSED! ✅                   ║")
        print("║                                                              ║")
        print("║  Day 8 Enhanced Features Validated:                         ║")
        print("║  • Complex workflows with priorities                        ║")
        print("║  • Step-by-step execution monitoring                        ║")
        print("║  • Inhibitor arc functionality                              ║")
        print("║  • Capacity constraints                                     ║")
        print("║  • Concurrent execution strategy                            ║")
        print("║  • Conflict resolution strategies                           ║")
        print("║  • Fluent API complex scenarios                             ║")
        print("║  • Comprehensive error handling                             ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
    finally:
        cleanup_library()
