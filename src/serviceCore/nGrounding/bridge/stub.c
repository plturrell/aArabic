#include <stdint.h>
#include <string.h>
#include <stdio.h>

int32_t lean4_check(const uint8_t* source_ptr, size_t source_len, uint8_t* result_buf, size_t buf_size) {
    const char* msg = "{"
        "\"success\": false, "
        "\"messages\": \"Mojo compiler not available (Stub)\", "
        "\"errors\": [{\"message\": \"Mojo compiler stub - check not available\"}], "
        "\"warnings\": [], "
        "\"error_count\": 1, "
        "\"warning_count\": 0, "
        "\"info_count\": 0"
    "}";
    size_t len = strlen(msg);
    if (len >= buf_size) len = buf_size - 1;
    memcpy(result_buf, msg, len);
    result_buf[len] = 0;
    return (int32_t)len;
}

int32_t lean4_run(const uint8_t* source_ptr, size_t source_len, uint8_t* result_buf, size_t buf_size) {
    const char* msg = "{"
        "\"success\": false, "
        "\"stdout\": \"\", "
        "\"stderr\": \"Mojo runtime not available (Stub)\", "
        "\"exit_code\": 1"
    "}";
    size_t len = strlen(msg);
    if (len >= buf_size) len = buf_size - 1;
    memcpy(result_buf, msg, len);
    result_buf[len] = 0;
    return (int32_t)len;
}

int32_t lean4_elaborate(const uint8_t* source_ptr, size_t source_len, uint8_t* result_buf, size_t buf_size) {
    const char* msg = "{"
        "\"success\": false, "
        "\"declarations\": [], "
        "\"environment_size\": 0, "
        "\"errors\": \"Mojo elaborator not available (Stub)\""
    "}";
    size_t len = strlen(msg);
    if (len >= buf_size) len = buf_size - 1;
    memcpy(result_buf, msg, len);
    result_buf[len] = 0;
    return (int32_t)len;
}