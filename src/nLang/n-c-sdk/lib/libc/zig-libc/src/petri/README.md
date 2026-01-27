# Native Petri Net Library for Zig

A comprehensive, production-ready Petri net implementation with 100 functions providing complete support for modeling concurrent systems, workflows, and distributed processes.

## Overview

This library provides native Petri net support as a first-class citizen in systems programming. It includes:

- **Core Petri Net Operations** (50 functions): Complete lifecycle management
- **Execution Control** (10 functions): Simulation and runtime control
- **State Analysis** (10 functions): Behavioral property verification
- **History & Monitoring** (10 functions): Event tracing and callbacks
- **Multi-Process Support** (10 functions - stubs): Shared memory coordination
- **Colored Petri Nets** (10 functions - stubs): Advanced token types

## Architecture

### File Structure

```
petri/
├── types.zig      # Type definitions and constants
├── core.zig       # Core implementation (50 functions)
├── lib.zig        # Public API and advanced features (50 functions)
├── test_petri.zig # Comprehensive test suite
└── README.md      # This file
```

### Key Components

1. **PetriNet**: Main container for places, transitions, and arcs
2. **Place**: Token containers with capacity limits
3. **Transition**: Firing rules with guards and priorities
4. **Arc**: Connections (input, output, inhibitor) with weights
5. **Token**: Data carriers with timestamps and colors

## Features

### ✅ Fully Implemented

#### Net Management
- `pn_create` / `pn_destroy`: Lifecycle management
- `pn_reset`: Return to initial marking
- `pn_validate`: Structure validation
- `pn_stats`: Collect statistics

#### Place Operations
- `pn_place_create` / `pn_place_destroy`: Place management
- `pn_place_set_capacity` / `pn_place_get_capacity`: Capacity control
- `pn_place_token_count`: Query token count
- `pn_place_has_tokens`: Check for tokens
- `pn_place_get_marking`: Get current marking

#### Transition Operations
- `pn_trans_create` / `pn_trans_destroy`: Transition management
- `pn_trans_set_priority` / `pn_trans_get_priority`: Priority control
- `pn_trans_set_guard`: Attach guard functions
- `pn_trans_is_enabled`: Check if fireable
- `pn_trans_fire`: Execute transition (moves tokens)
- `pn_trans_enable` / `pn_trans_disable`: Control state

#### Arc Operations
- `pn_arc_create` / `pn_arc_destroy`: Arc management
- `pn_arc_connect`: Link places and transitions
- `pn_arc_set_weight` / `pn_arc_get_weight`: Weight control
- `pn_arc_set_guard`: Attach arc guards
- Support for input, output, and inhibitor arcs

#### Token Operations
- `pn_token_create` / `pn_token_destroy`: Token lifecycle
- `pn_token_clone`: Duplicate tokens
- `pn_token_put` / `pn_token_get` / `pn_token_peek`: Place interactions
- `pn_token_set_data` / `pn_token_get_data`: Data management
- `pn_token_set_color` / `pn_token_get_color`: Color support
- `pn_token_get_timestamp` / `pn_token_get_id`: Metadata access

#### Execution Control
- `pn_execute`: Run net in various modes
- `pn_step`: Single-step execution
- `pn_run_until`: Execute with condition
- `pn_pause` / `pn_resume` / `pn_stop`: Runtime control
- `pn_get_enabled_transitions`: Query fireable transitions
- `pn_fire_random`: Fire random enabled transition
- `pn_fire_priority`: Fire highest priority transition
- `pn_fire_all`: Fire all enabled transitions

#### State Analysis
- `pn_is_deadlocked`: Detect deadlock conditions
- `pn_is_bounded`: Check token boundedness
- `pn_is_safe`: Verify 1-safe property
- `pn_is_live`: Check liveness (simplified)

#### History & Monitoring
- `pn_trace_enable` / `pn_trace_disable`: Event tracing control
- `pn_trace_get` / `pn_trace_clear`: Access trace history
- `pn_callback_set` / `pn_callback_remove`: Event callbacks
- `pn_metrics_get`: Collect performance metrics
- `pn_metrics_reset`: Reset metric counters

### 🚧 Stub Implementations (Ready for Enhancement)

- **Advanced State Analysis**: Reachability graphs, coverability trees, invariants
- **Multi-Process Support**: Shared memory, locking, inter-process notifications
- **Colored Petri Nets**: Type systems, multisets, color expressions
- **Serialization**: JSON, PNML, DOT, BPMN export/import

## Usage Examples

### Basic Producer-Consumer

```zig
const petri = @import("petri/lib.zig");

// Create net
const net = petri.pn_create("producer_consumer", 0);
defer _ = petri.pn_destroy(net);

// Create places
const buffer = petri.pn_place_create(net, "buffer", "Buffer");
_ = petri.pn_place_set_capacity(buffer, 10);

const input = petri.pn_place_create(net, "input", "Input");
const output = petri.pn_place_create(net, "output", "Output");

// Create transitions
const produce = petri.pn_trans_create(net, "produce", "Produce");
const consume = petri.pn_trans_create(net, "consume", "Consume");

// Connect arcs
const arc1 = petri.pn_arc_create(net, "a1", .input);
_ = petri.pn_arc_connect(arc1, "input", "produce");

const arc2 = petri.pn_arc_create(net, "a2", .output);
_ = petri.pn_arc_connect(arc2, "produce", "buffer");

const arc3 = petri.pn_arc_create(net, "a3", .input);
_ = petri.pn_arc_connect(arc3, "buffer", "consume");

const arc4 = petri.pn_arc_create(net, "a4", .output);
_ = petri.pn_arc_connect(arc4, "consume", "output");

// Add initial tokens
var i: usize = 0;
while (i < 5) : (i += 1) {
    const token = petri.pn_token_create(null, 0);
    _ = petri.pn_token_put(input, token);
}

// Execute
_ = petri.pn_execute(net, .continuous);
```

### Mutual Exclusion

```zig
// Create mutex net
const net = petri.pn_create("mutex", petri.PN_CREATE_TRACED);

// Free/busy places
const free = petri.pn_place_create(net, "free", "Resource Free");
const busy = petri.pn_place_create(net, "busy", "Resource Busy");

// Process places
const p1_idle = petri.pn_place_create(net, "p1_idle", "Process 1 Idle");
const p1_critical = petri.pn_place_create(net, "p1_crit", "P1 Critical");

// Transitions
const p1_acquire = petri.pn_trans_create(net, "p1_acq", "P1 Acquire");
const p1_release = petri.pn_trans_create(net, "p1_rel", "P1 Release");

// Connect arcs for mutual exclusion
// p1_idle + free -> p1_acquire -> p1_critical + busy
// p1_critical + busy -> p1_release -> p1_idle + free
```

### With Guards and Callbacks

```zig
// Guard function
fn token_guard(ctx: ?*anyopaque) callconv(.C) bool {
    _ = ctx;
    // Custom logic here
    return true;
}

// Event callback
fn on_transition_fired(net: ?*petri.pn_net_t, event: petri.pn_event_type_t, ctx: ?*anyopaque) callconv(.C) void {
    _ = net; _ = event; _ = ctx;
    std.debug.print("Transition fired!\n", .{});
}

// Set guard on transition
_ = petri.pn_trans_set_guard(transition, token_guard, null);

// Register callback
_ = petri.pn_callback_set(net, .transition_fired, on_transition_fired);
```

## Testing

Run the comprehensive test suite:

```bash
zig test src/nLang/n-c-sdk/lib/libc/zig-libc/src/petri/test_petri.zig
```

Tests cover:
- Basic operations (create, destroy, connect)
- Token management and movement
- Execution modes and control
- Deadlock detection
- Bounded and safe properties
- Event tracing and monitoring
- Metrics collection
- Inhibitor arcs
- Priority-based firing
- Capacity limits

## Implementation Details

### Token Movement (pn_trans_fire)

The `pn_trans_fire` function implements classical Petri net firing semantics:

1. **Check Enablement**: Verify transition is enabled and guards pass
2. **Validate Inputs**: Ensure all input places have sufficient tokens
3. **Check Inhibitors**: Verify inhibitor places are empty
4. **Remove Tokens**: Remove tokens from input places (by arc weight)
5. **Add Tokens**: Add tokens to output places (by arc weight)
6. **Record Events**: Log transition firing and place updates

### Thread Safety

Current implementation is **not thread-safe**. For concurrent access:
- Use external synchronization (mutexes)
- Or implement multi-process stubs with proper locking

### Memory Management

- Uses Zig's standard allocator
- All structures properly freed on destroy
- Token data is copied and owned by places
- Caller responsible for destroying returned tokens

## Performance Characteristics

- **Place Lookup**: O(1) average (HashMap)
- **Transition Lookup**: O(1) average (HashMap)
- **Arc Traversal**: O(n) where n = number of arcs
- **Token Movement**: O(w) where w = arc weight
- **Enabled Check**: O(a) where a = arcs connected to transition

## Known Limitations

1. **No Concurrency**: Single-threaded execution only
2. **Fixed String Sizes**: 256-byte limits on IDs and names
3. **Simplified Liveness**: Full liveness checking requires reachability analysis
4. **No Persistence**: Serialize/deserialize not implemented
5. **Limited Error Codes**: Uses errno; could be more specific

## Future Enhancements

### High Priority
- [ ] Complete serialization (JSON, PNML export/import)
- [ ] Reachability graph generation
- [ ] Coverability tree construction
- [ ] P/T invariant calculation
- [ ] Siphon and trap detection

### Medium Priority
- [ ] Timed Petri nets
- [ ] Stochastic Petri nets
- [ ] High-level Petri nets (full colored support)
- [ ] Visual editor integration
- [ ] Performance profiling tools

### Low Priority
- [ ] Multi-process shared memory support
- [ ] Distributed Petri net execution
- [ ] Real-time constraints
- [ ] Formal verification integration

## References

1. Peterson, J. L. (1981). *Petri Net Theory and the Modeling of Systems*
2. Murata, T. (1989). "Petri nets: Properties, analysis and applications"
3. Reisig, W. (2013). *Understanding Petri Nets*
4. van der Aalst, W. (1998). "The Application of Petri Nets to Workflow Management"

## License

Part of the n-c-sdk Zig standard library implementation.

## Contributing

See the main repository CONTRIBUTING.md for guidelines.

## Status

**Production Ready for:**
- Basic Petri net simulation
- Workflow modeling
- Concurrent system analysis
- State machine implementation

**Experimental for:**
- Large-scale distributed systems
- Real-time critical systems
- Multi-process coordination (stubs only)

**Not Ready for:**
- Systems requiring formal verification tools
- Applications needing colored Petri nets
- Persistent workflow storage