/**
 * HyperShimmy FFI Interface
 * C ABI for Zig ↔ Mojo interoperability
 * 
 * This header defines the FFI boundary between:
 * - Zig (HTTP server, OData, I/O)
 * - Mojo (AI/ML, embeddings, LLM inference)
 */

#ifndef HYPERSHIMMY_FFI_H
#define HYPERSHIMMY_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types and Structures
// ============================================================================

/**
 * Result codes for FFI operations
 */
typedef enum {
    HS_SUCCESS = 0,
    HS_ERROR_INVALID_ARGUMENT = 1,
    HS_ERROR_OUT_OF_MEMORY = 2,
    HS_ERROR_NOT_INITIALIZED = 3,
    HS_ERROR_ALREADY_INITIALIZED = 4,
    HS_ERROR_INTERNAL = 5,
    HS_ERROR_NOT_IMPLEMENTED = 6
} HSResult;

/**
 * Opaque handle to the Mojo runtime context
 */
typedef struct HSContext HSContext;

/**
 * String structure for passing text across FFI boundary
 */
typedef struct {
    const char* data;
    uint64_t length;
} HSString;

/**
 * Buffer structure for passing binary data
 */
typedef struct {
    const uint8_t* data;
    uint64_t length;
} HSBuffer;

/**
 * Source type enumeration
 */
typedef enum {
    HS_SOURCE_TYPE_URL = 0,
    HS_SOURCE_TYPE_PDF = 1,
    HS_SOURCE_TYPE_TEXT = 2,
    HS_SOURCE_TYPE_FILE = 3
} HSSourceType;

/**
 * Source status enumeration
 */
typedef enum {
    HS_STATUS_PENDING = 0,
    HS_STATUS_PROCESSING = 1,
    HS_STATUS_READY = 2,
    HS_STATUS_FAILED = 3
} HSSourceStatus;

// ============================================================================
// Lifecycle Functions
// ============================================================================

/**
 * Initialize the Mojo runtime and create a context
 * 
 * @param ctx_out Pointer to store the created context handle
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_init(HSContext** ctx_out);

/**
 * Shutdown the Mojo runtime and cleanup resources
 * 
 * @param ctx Context handle to cleanup
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_cleanup(HSContext* ctx);

/**
 * Check if the Mojo runtime is initialized
 * 
 * @param ctx Context handle
 * @return true if initialized, false otherwise
 */
bool hs_is_initialized(HSContext* ctx);

/**
 * Get the version string of the Mojo runtime
 * 
 * @param ctx Context handle
 * @param version_out Pointer to store the version string
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_get_version(HSContext* ctx, HSString* version_out);

// ============================================================================
// String and Memory Management
// ============================================================================

/**
 * Allocate a string on the Mojo side
 * 
 * @param ctx Context handle
 * @param data String data
 * @param length String length
 * @param str_out Pointer to store the allocated string
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_string_alloc(HSContext* ctx, const char* data, uint64_t length, HSString* str_out);

/**
 * Free a string allocated by Mojo
 * 
 * @param ctx Context handle
 * @param str String to free
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_string_free(HSContext* ctx, HSString* str);

/**
 * Allocate a buffer on the Mojo side
 * 
 * @param ctx Context handle
 * @param data Buffer data
 * @param length Buffer length
 * @param buf_out Pointer to store the allocated buffer
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_buffer_alloc(HSContext* ctx, const uint8_t* data, uint64_t length, HSBuffer* buf_out);

/**
 * Free a buffer allocated by Mojo
 * 
 * @param ctx Context handle
 * @param buf Buffer to free
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_buffer_free(HSContext* ctx, HSBuffer* buf);

// ============================================================================
// Source Management (Day 7-8)
// ============================================================================

/**
 * Create a new source in Mojo
 * (To be implemented in Day 8)
 * 
 * @param ctx Context handle
 * @param title Source title
 * @param source_type Source type
 * @param url Source URL
 * @param content Source content
 * @param source_id_out Pointer to store the created source ID
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_source_create(
    HSContext* ctx,
    HSString title,
    HSSourceType source_type,
    HSString url,
    HSString content,
    HSString* source_id_out
);

/**
 * Get source by ID
 * (To be implemented in Day 8)
 */
HSResult hs_source_get(
    HSContext* ctx,
    HSString source_id,
    HSString* title_out,
    HSSourceType* type_out,
    HSString* url_out,
    HSString* content_out,
    HSSourceStatus* status_out
);

/**
 * Delete a source
 * (To be implemented in Day 8)
 */
HSResult hs_source_delete(HSContext* ctx, HSString source_id);

// ============================================================================
// Embedding Functions (Week 5)
// ============================================================================

/**
 * Generate embeddings for text
 * (To be implemented in Week 5)
 */
HSResult hs_embed_text(
    HSContext* ctx,
    HSString text,
    HSBuffer* embedding_out
);

// ============================================================================
// LLM Functions (Week 6)
// ============================================================================

/**
 * Generate chat completion
 * (To be implemented in Week 6)
 */
HSResult hs_chat_complete(
    HSContext* ctx,
    HSString prompt,
    HSString context,
    HSString* response_out
);

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Get the last error message
 * 
 * @param ctx Context handle
 * @param error_out Pointer to store the error message
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_get_last_error(HSContext* ctx, HSString* error_out);

/**
 * Clear the last error
 * 
 * @param ctx Context handle
 * @return HS_SUCCESS on success, error code otherwise
 */
HSResult hs_clear_error(HSContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // HYPERSHIMMY_FFI_H
