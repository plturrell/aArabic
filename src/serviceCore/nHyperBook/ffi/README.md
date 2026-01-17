# HyperShimmy FFI Bridge

Foreign Function Interface (FFI) bridge between Zig and Mojo for cross-language interoperability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HyperShimmy System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         C ABI          ┌──────────────┐  │
│  │              │  ◄───────────────────►  │              │  │
│  │   Zig Side   │                         │  Mojo Side   │  │
│  │              │                         │              │  │
│  │  • HTTP      │  hypershimmy_ffi.h     │  • AI/ML     │  │
│  │  • OData     │                         │  • LLM       │  │
│  │  • I/O       │  mojo_bridge.zig       │  • Embed     │  │
│  │              │                         │              │  │
│  └──────────────┘                         └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. C ABI Header (`hypershimmy_ffi.h`)

Defines the interface contract between Zig and Mojo:
- Result codes and error types
- Data structures (HSContext, HSString, HSBuffer)
- Function signatures for all FFI operations
- Type enumerations (SourceType, SourceStatus)

### 2. Zig Wrapper (`mojo_bridge.zig`)

Type-safe Zig interface to the Mojo runtime:
- Context management
- String/Buffer conversions
- Error handling
- High-level API wrapper

### 3. Mojo Implementation (`hypershimmy_ffi.mojo`)

Implements the FFI functions on the Mojo side:
- Runtime initialization/cleanup
- Memory management
- Source management (Day 8)
- Embeddings (Week 5)
- LLM inference (Week 6)

### 4. Test Program (`test_ffi.zig`)

Comprehensive test suite for FFI bridge:
- Context initialization
- String/Buffer conversion
- Version checking
- Error handling
- Feature testing

## Data Flow

### Example: Creating a Source

```
1. Zig Code:
   ctx.createSource(allocator, title, type, url, content)
   
2. mojo_bridge.zig:
   - Converts Zig strings to FFIString
   - Calls C function hs_source_create()
   
3. C ABI Boundary:
   - Passes HSString structures
   - Returns HSResult code
   
4. hypershimmy_ffi.mojo:
   - Receives FFI call
   - Creates Mojo Source struct
   - Stores in context
   - Returns source ID
   
5. Return Path:
   - Mojo → C ABI → Zig wrapper → Zig app
```

## Memory Management

### String Allocation

**Zig → Mojo:**
```zig
const ffi_str = FFIString.init("Hello");
// Stack-allocated, points to Zig memory
```

**Mojo → Zig:**
```mojo
fn hs_string_alloc() -> HSString:
    # Allocates on Mojo heap
    # Zig must call hs_string_free() when done
```

### Rules

1. **Zig owns Zig memory** - No free needed for stack strings
2. **Mojo owns Mojo memory** - Always call `hs_string_free()`
3. **Cross-boundary strings** - Always duplicate with allocator
4. **Context lifecycle** - Init → Use → Deinit

## API Reference

### Lifecycle Functions

```zig
// Initialize Mojo runtime
pub fn init() !Context

// Cleanup Mojo runtime  
pub fn deinit(self: Context) void

// Check if initialized
pub fn isInitialized(self: Context) bool

// Get version string
pub fn getVersion(self: Context, allocator: Allocator) ![]const u8
```

### Error Handling

```zig
// Get last error message
pub fn getLastError(self: Context, allocator: Allocator) ![]const u8

// Clear error state
pub fn clearError(self: Context) !void
```

### Source Management (Day 8)

```zig
// Create new source
pub fn createSource(
    self: Context,
    allocator: Allocator,
    title: []const u8,
    source_type: SourceType,
    url: []const u8,
    content: []const u8,
) ![]const u8

// Delete source
pub fn deleteSource(self: Context, source_id: []const u8) !void
```

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | HS_SUCCESS | Operation successful |
| 1 | HS_ERROR_INVALID_ARGUMENT | Invalid function argument |
| 2 | HS_ERROR_OUT_OF_MEMORY | Memory allocation failed |
| 3 | HS_ERROR_NOT_INITIALIZED | Context not initialized |
| 4 | HS_ERROR_ALREADY_INITIALIZED | Context already initialized |
| 5 | HS_ERROR_INTERNAL | Internal Mojo error |
| 6 | HS_ERROR_NOT_IMPLEMENTED | Feature not yet implemented |

## Building

### Mojo Side

```bash
cd mojo
mojo build hypershimmy_ffi.mojo --output libhypershimmy_mojo.so
```

### Zig Side

```bash
zig build-exe test_ffi.zig \
  -I../ffi \
  -L../mojo \
  -lhypershimmy_mojo
```

## Testing

```bash
# Run FFI bridge tests
cd ffi
zig build test

# Run integration test
./test_ffi
```

Expected output:
```
=== HyperShimmy FFI Bridge Test ===

Test 1: Initializing Mojo context...
  ✅ Context initialized successfully

Test 2: Checking if context is initialized...
  ✅ Context is initialized

Test 3: Getting Mojo runtime version...
  ✅ Mojo runtime version: 0.1.0-mojo

...

=== All tests passed! ===
```

## Implementation Status

### Day 6 (✅ Complete)
- [x] C ABI header definition
- [x] Zig wrapper module
- [x] Mojo FFI implementation
- [x] Context lifecycle management
- [x] String/Buffer handling
- [x] Error handling
- [x] Test program

### Day 8 (⏳ Pending)
- [ ] Source CRUD implementation
- [ ] In-memory storage
- [ ] Source validation

### Week 5 (⏳ Pending)
- [ ] Embedding generation
- [ ] Vector operations

### Week 6 (⏳ Pending)
- [ ] LLM inference
- [ ] Chat completion
- [ ] Streaming responses

## Design Decisions

### Why C ABI?

1. **Universal compatibility** - Both Zig and Mojo support C FFI
2. **Stable interface** - C ABI is well-defined and stable
3. **Performance** - Zero-cost abstraction
4. **Portability** - Works across platforms

### Why Opaque Pointers?

```c
typedef struct HSContext HSContext;  // Opaque
```

**Benefits:**
- Encapsulation - Mojo internals hidden from Zig
- Flexibility - Can change Mojo implementation
- Safety - No direct memory access from Zig
- Type safety - Context pointers strongly typed

### Why Manual Memory Management?

- **Control** - Explicit ownership semantics
- **Performance** - No GC overhead
- **Predictability** - Know exactly when allocations happen
- **Cross-language** - Clear responsibility boundaries

## Future Enhancements

1. **Async Operations** - Non-blocking FFI calls
2. **Streaming** - Iterator-based data transfer
3. **Callbacks** - Mojo → Zig function calls
4. **Error Context** - Rich error information
5. **Performance** - Zero-copy operations where possible

## Troubleshooting

### Context initialization fails
- Check Mojo runtime is installed
- Verify library path is correct
- Ensure no existing context

### Memory leaks
- Always call `deinit()` on Context
- Free returned strings with allocator
- Check paired alloc/free calls

### Segmentation faults
- Verify C ABI header matches Mojo implementation
- Check pointer validity before dereferencing
- Ensure strings are null-terminated where required

## References

- [Zig C Interop](https://ziglang.org/documentation/master/#C)
- [Mojo FFI Documentation](https://docs.modular.com/mojo/manual/python/)
- [C ABI Best Practices](https://www.lurklurk.org/effective_cpp/c_abi.html)

---

**Status:** ✅ Day 6 Complete  
**Next:** Day 7 - Source Entity CRUD (Zig implementation)
