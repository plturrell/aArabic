"""
HyperShimmy FFI Implementation (Mojo side)
Implements the C ABI interface for Zig interoperability
"""

from sys.ffi import external_call
from memory import UnsafePointer
from collections import Dict, List
from utils.stringref import StringRef

# ============================================================================
# Context and State Management
# ============================================================================

struct HSContext:
    """Opaque context handle for the Mojo runtime"""
    var initialized: Bool
    var version: String
    var last_error: String
    var source_manager: SourceManager
    
    fn __init__(inout self):
        self.initialized = False
        self.version = "0.1.0-mojo"
        self.last_error = ""
        self.source_manager = SourceManager()
    
    fn set_error(inout self, error: String):
        """Set the last error message"""
        self.last_error = error
    
    fn clear_error(inout self):
        """Clear the last error message"""
        self.last_error = ""

struct Source:
    """Source data structure"""
    var id: String
    var title: String
    var source_type: Int32
    var url: String
    var content: String
    var status: Int32
    
    fn __init__(inout self, id: String, title: String, source_type: Int32, 
                url: String, content: String):
        self.id = id
        self.title = title
        self.source_type = source_type
        self.url = url
        self.content = content
        self.status = 2  # HS_STATUS_READY

# Global context pointer (managed in Mojo)
var _global_context: UnsafePointer[HSContext] = UnsafePointer[HSContext]()

# ============================================================================
# FFI Function Implementations
# ============================================================================

@export
fn hs_init(ctx_out: UnsafePointer[UnsafePointer[HSContext]]) -> Int32:
    """Initialize the Mojo runtime"""
    # Check if already initialized
    if _global_context:
        return 4  # HS_ERROR_ALREADY_INITIALIZED
    
    # Allocate context
    _global_context = UnsafePointer[HSContext].alloc(1)
    var ctx = HSContext()
    ctx.initialized = True
    _global_context.init_pointee_move(ctx^)
    
    # Return context pointer
    ctx_out[0] = _global_context
    
    return 0  # HS_SUCCESS

@export
fn hs_cleanup(ctx: UnsafePointer[HSContext]) -> Int32:
    """Cleanup the Mojo runtime"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    # Free the context
    ctx.free()
    _global_context = UnsafePointer[HSContext]()
    
    return 0  # HS_SUCCESS

@export
fn hs_is_initialized(ctx: UnsafePointer[HSContext]) -> Bool:
    """Check if the runtime is initialized"""
    if not ctx:
        return False
    return ctx[0].initialized

@export
fn hs_get_version(ctx: UnsafePointer[HSContext], 
                  version_out: UnsafePointer[HSString]) -> Int32:
    """Get the Mojo runtime version"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    var version_str = ctx[0].version
    var version_ptr = version_str.unsafe_ptr()
    
    # Set the output string
    version_out[0] = HSString(version_ptr, len(version_str))
    
    return 0  # HS_SUCCESS

@export
fn hs_get_last_error(ctx: UnsafePointer[HSContext],
                     error_out: UnsafePointer[HSString]) -> Int32:
    """Get the last error message"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    var error_str = ctx[0].last_error
    var error_ptr = error_str.unsafe_ptr()
    
    # Set the output string
    error_out[0] = HSString(error_ptr, len(error_str))
    
    return 0  # HS_SUCCESS

@export
fn hs_clear_error(ctx: UnsafePointer[HSContext]) -> Int32:
    """Clear the last error"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    ctx[0].clear_error()
    return 0  # HS_SUCCESS

# ============================================================================
# String and Memory Management
# ============================================================================

struct HSString:
    """FFI string structure"""
    var data: UnsafePointer[UInt8]
    var length: UInt64
    
    fn __init__(inout self, data: UnsafePointer[UInt8], length: Int):
        self.data = data
        self.length = length

@export
fn hs_string_alloc(ctx: UnsafePointer[HSContext],
                   data: UnsafePointer[UInt8],
                   length: UInt64,
                   str_out: UnsafePointer[HSString]) -> Int32:
    """Allocate a string on the Mojo side"""
    if not ctx or not data:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    # Allocate new memory for the string
    var new_data = UnsafePointer[UInt8].alloc(int(length))
    
    # Copy the data
    for i in range(int(length)):
        new_data[i] = data[i]
    
    # Set the output string
    str_out[0] = HSString(new_data, int(length))
    
    return 0  # HS_SUCCESS

@export
fn hs_string_free(ctx: UnsafePointer[HSContext],
                  str: UnsafePointer[HSString]) -> Int32:
    """Free a string allocated by Mojo"""
    if not ctx or not str:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    # Free the string data
    str[0].data.free()
    
    return 0  # HS_SUCCESS

struct HSBuffer:
    """FFI buffer structure"""
    var data: UnsafePointer[UInt8]
    var length: UInt64
    
    fn __init__(inout self, data: UnsafePointer[UInt8], length: Int):
        self.data = data
        self.length = length

@export
fn hs_buffer_alloc(ctx: UnsafePointer[HSContext],
                   data: UnsafePointer[UInt8],
                   length: UInt64,
                   buf_out: UnsafePointer[HSBuffer]) -> Int32:
    """Allocate a buffer on the Mojo side"""
    if not ctx or not data:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    # Allocate new memory for the buffer
    var new_data = UnsafePointer[UInt8].alloc(int(length))
    
    # Copy the data
    for i in range(int(length)):
        new_data[i] = data[i]
    
    # Set the output buffer
    buf_out[0] = HSBuffer(new_data, int(length))
    
    return 0  # HS_SUCCESS

@export
fn hs_buffer_free(ctx: UnsafePointer[HSContext],
                  buf: UnsafePointer[HSBuffer]) -> Int32:
    """Free a buffer allocated by Mojo"""
    if not ctx or not buf:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    # Free the buffer data
    buf[0].data.free()
    
    return 0  # HS_SUCCESS

# ============================================================================
# Helper Functions
# ============================================================================

fn string_from_hsstring(hs: HSString) -> String:
    """Convert HSString to Mojo String"""
    var result = String()
    for i in range(int(hs.length)):
        result += chr(int(hs.data[i]))
    return result

# ============================================================================
# Source Management
# ============================================================================

@export
fn hs_source_create(ctx: UnsafePointer[HSContext],
                    title: HSString,
                    source_type: Int32,
                    url: HSString,
                    content: HSString,
                    source_id_out: UnsafePointer[HSString]) -> Int32:
    """Create a new source"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    try:
        # Convert HSString to String
        var title_str = string_from_hsstring(title)
        var url_str = string_from_hsstring(url)
        var content_str = string_from_hsstring(content)
        
        # Create source via manager
        var source_type_enum = SourceType(source_type)
        var id = ctx[0].source_manager.create_source(
            title_str,
            source_type_enum,
            url_str,
            content_str
        )
        
        # Allocate string for ID
        var id_data = UnsafePointer[UInt8].alloc(len(id))
        for i in range(len(id)):
            id_data[i] = ord(id[i])
        
        source_id_out[0] = HSString(id_data, len(id))
        
        return 0  # HS_SUCCESS
    except e:
        ctx[0].set_error("Failed to create source: " + str(e))
        return 5  # HS_ERROR_INTERNAL

@export
fn hs_source_get(ctx: UnsafePointer[HSContext],
                 source_id: HSString,
                 title_out: UnsafePointer[HSString],
                 type_out: UnsafePointer[Int32],
                 url_out: UnsafePointer[HSString],
                 content_out: UnsafePointer[HSString],
                 status_out: UnsafePointer[Int32]) -> Int32:
    """Get source by ID (stub for Day 8)"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    ctx[0].set_error("Source retrieval will be implemented in Day 8")
    return 6  # HS_ERROR_NOT_IMPLEMENTED

@export
fn hs_source_delete(ctx: UnsafePointer[HSContext],
                    source_id: HSString) -> Int32:
    """Delete a source"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    try:
        # Convert HSString to String
        var id_str = string_from_hsstring(source_id)
        
        # Delete via manager
        ctx[0].source_manager.delete_source(id_str)
        
        return 0  # HS_SUCCESS
    except e:
        ctx[0].set_error("Failed to delete source: " + str(e))
        return 5  # HS_ERROR_INTERNAL

# ============================================================================
# Embedding and LLM Functions (Stubs for future weeks)
# ============================================================================

@export
fn hs_embed_text(ctx: UnsafePointer[HSContext],
                 text: HSString,
                 embedding_out: UnsafePointer[HSBuffer]) -> Int32:
    """Generate embeddings (stub for Week 5)"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    ctx[0].set_error("Embedding generation will be implemented in Week 5")
    return 6  # HS_ERROR_NOT_IMPLEMENTED

@export
fn hs_chat_complete(ctx: UnsafePointer[HSContext],
                    prompt: HSString,
                    context: HSString,
                    response_out: UnsafePointer[HSString]) -> Int32:
    """Generate chat completion"""
    if not ctx:
        return 1  # HS_ERROR_INVALID_ARGUMENT
    
    try:
        # Convert HSString to String
        var prompt_str = string_from_hsstring(prompt)
        var context_str = string_from_hsstring(context)
        
        # In production, would use actual LLM chat from llm_chat.mojo
        # For now, return mock response
        var response = "Based on your documents: " + prompt_str[:50] + "..."
        
        # Allocate string for response
        var response_data = UnsafePointer[UInt8].alloc(len(response))
        for i in range(len(response)):
            response_data[i] = ord(response[i])
        
        response_out[0] = HSString(response_data, len(response))
        
        return 0  # HS_SUCCESS
    except e:
        ctx[0].set_error("Failed to generate chat completion: " + str(e))
        return 5  # HS_ERROR_INTERNAL
