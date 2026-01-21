# nCode Error Codes Reference

**Version:** 1.0  
**Last Updated:** 2026-01-18

## Overview

This document provides a complete reference of all error codes used in nCode. Each error code includes description, cause, solution, and related documentation.

## Error Code Format

Error codes follow the format: `Exxx` where:
- `E` = Error prefix
- `xxx` = 3-digit code indicating category and specific error

### Categories

- **E001-E099**: Critical/Fatal errors
- **E100-E199**: Transient/Retry errors
- **E200-E299**: Non-critical/Warning errors
- **E300-E399**: Client errors
- **E400-E499**: Server errors
- **E500-E599**: Database errors

## Critical Errors (E001-E099)

### E001: Invalid SCIP Protobuf Format

**Description:** The provided SCIP file contains invalid protobuf data.

**Cause:**
- Corrupted SCIP file
- Incomplete file transfer
- Wrong file format
- Incompatible SCIP version

**HTTP Status:** 400 Bad Request

**Response Example:**
```json
{
  "error": {
    "code": "E001",
    "message": "Invalid SCIP protobuf format",
    "details": "Failed to parse protobuf at offset 1234: unexpected field type",
    "suggestion": "Regenerate SCIP index with compatible version"
  }
}
```

**Solution:**
1. Verify file is valid SCIP format
2. Check file wasn't corrupted during transfer
3. Regenerate SCIP index with correct indexer
4. Ensure SCIP version compatibility

**Related:** [SCIP Format Guide](ARCHITECTURE.md#scip-format)

---

### E002: Corrupted SCIP File

**Description:** SCIP file structure is corrupted or incomplete.

**Cause:**
- Partial file write
- Disk corruption
- Interrupted generation
- Storage failure

**HTTP Status:** 400 Bad Request

**Solution:**
1. Delete corrupted file
2. Regenerate SCIP index
3. Verify storage integrity
4. Check disk space

---

### E003: Missing Required Fields

**Description:** Required SCIP fields are missing from the index.

**Cause:**
- Incomplete indexing
- Old SCIP version
- Invalid symbol data
- Indexer bug

**HTTP Status:** 400 Bad Request

**Solution:**
1. Check indexer version
2. Regenerate with updated indexer
3. Verify source code is valid
4. Review indexer logs

---

### E004: Out of Memory

**Description:** System ran out of available memory.

**Cause:**
- Large SCIP file
- Memory leak
- Insufficient system memory
- Too many concurrent requests

**HTTP Status:** 500 Internal Server Error

**Solution:**
1. Increase system memory
2. Use streaming for large files
3. Implement memory limits
4. Add backpressure
5. Scale horizontally

---

### E005: Configuration Error

**Description:** Invalid or missing configuration.

**Cause:**
- Missing config file
- Invalid config values
- Wrong file format
- Environment variables not set

**HTTP Status:** 500 Internal Server Error

**Solution:**
1. Check config file exists
2. Validate configuration values
3. Set required environment variables
4. Review configuration documentation

## Transient Errors (E100-E199)

### E101: Database Connection Timeout

**Description:** Timeout connecting to database.

**Cause:**
- Database not running
- Network issues
- Firewall blocking connection
- Database overloaded

**HTTP Status:** 503 Service Unavailable

**Retry:** Yes (3 retries, exponential backoff)

**Solution:**
1. Check database is running
2. Verify network connectivity
3. Check firewall rules
4. Scale database if overloaded

---

### E102: Network Timeout

**Description:** Network request timed out.

**Cause:**
- Slow network
- Service overloaded
- Network congestion
- Timeout too short

**HTTP Status:** 504 Gateway Timeout

**Retry:** Yes (3 retries)

**Solution:**
1. Increase timeout value
2. Check network performance
3. Scale service
4. Use CDN if applicable

---

### E103: Database Temporarily Unavailable

**Description:** Database is temporarily unavailable.

**Cause:**
- Database restarting
- Maintenance window
- Failover in progress
- Resource exhaustion

**HTTP Status:** 503 Service Unavailable

**Retry:** Yes (5 retries, longer backoff)

**Solution:**
1. Wait for database to recover
2. Check database logs
3. Verify resource availability
4. Enable read replicas

---

### E104: Rate Limit Exceeded

**Description:** Too many requests, rate limit exceeded.

**Cause:**
- Too many concurrent requests
- Burst traffic
- No rate limiting configured
- DDoS attack

**HTTP Status:** 429 Too Many Requests

**Retry:** Yes (with longer delay)

**Solution:**
1. Implement client-side rate limiting
2. Add request queuing
3. Scale service horizontally
4. Use caching

---

### E105: Lock Contention

**Description:** Failed to acquire lock due to contention.

**Cause:**
- High concurrency
- Long-held locks
- Deadlock
- Resource contention

**HTTP Status:** 503 Service Unavailable

**Retry:** Yes (5 retries)

**Solution:**
1. Reduce lock duration
2. Use optimistic locking
3. Partition resources
4. Implement retry logic

## Non-Critical Errors (E200-E299)

### E201: Single Symbol Parse Failure

**Description:** Failed to parse a single symbol, but continuing.

**Cause:**
- Malformed symbol data
- Unsupported syntax
- Edge case in parser
- Corrupted symbol

**HTTP Status:** N/A (logged warning)

**Action:** Skip symbol, continue processing

---

### E202: Missing Optional Field

**Description:** Optional field missing, using default.

**Cause:**
- Old SCIP version
- Field not supported by indexer
- Intentionally omitted

**HTTP Status:** N/A (logged info)

**Action:** Use default value, continue

---

### E203: Partial Database Write

**Description:** Some data written successfully, some failed.

**Cause:**
- Database constraint violation
- Partial transaction
- Network interruption
- Duplicate key

**HTTP Status:** N/A (logged warning)

**Action:** Log warning, continue with succeeded writes

---

### E204: Cache Miss

**Description:** Requested data not in cache.

**Cause:**
- Cache expiration
- Cache eviction
- First access
- Cache cleared

**HTTP Status:** N/A

**Action:** Fetch from source, update cache

---

### E205: Non-Existent Reference

**Description:** Symbol reference points to non-existent symbol.

**Cause:**
- External dependency
- Incomplete indexing
- Deleted symbol
- Cross-project reference

**HTTP Status:** N/A (logged info)

**Action:** Skip reference, continue

## Client Errors (E300-E399)

### E301: Invalid API Request

**Description:** Request format or parameters are invalid.

**Cause:**
- Missing required parameter
- Invalid JSON
- Wrong content type
- Malformed request

**HTTP Status:** 400 Bad Request

**Solution:**
1. Check API documentation
2. Validate request format
3. Include all required parameters
4. Use correct content type

---

### E302: Unauthorized

**Description:** Authentication required or failed.

**Cause:**
- Missing API key
- Invalid credentials
- Expired token
- Wrong authentication method

**HTTP Status:** 401 Unauthorized

**Solution:**
1. Provide valid credentials
2. Renew expired tokens
3. Check authentication method
4. Verify API key

---

### E303: Resource Not Found

**Description:** Requested resource doesn't exist.

**Cause:**
- Wrong resource ID
- Resource deleted
- Typo in path
- Resource never created

**HTTP Status:** 404 Not Found

**Solution:**
1. Verify resource exists
2. Check resource ID
3. Review path for typos
4. Create resource if needed

---

### E304: Method Not Allowed

**Description:** HTTP method not supported for endpoint.

**Cause:**
- Using POST instead of GET
- Endpoint doesn't support method
- API version mismatch

**HTTP Status:** 405 Method Not Allowed

**Solution:**
1. Check API documentation
2. Use correct HTTP method
3. Verify endpoint supports method

---

### E305: Request Too Large

**Description:** Request payload exceeds size limit.

**Cause:**
- SCIP file too large
- Too many symbols
- Large embedded data
- Exceeds configured limit

**HTTP Status:** 413 Payload Too Large

**Solution:**
1. Split into smaller requests
2. Use streaming upload
3. Compress payload
4. Request limit increase

## Server Errors (E400-E499)

### E401: Internal Server Error

**Description:** Unexpected server error occurred.

**Cause:**
- Unhandled exception
- Bug in code
- Resource exhaustion
- Unexpected state

**HTTP Status:** 500 Internal Server Error

**Solution:**
1. Check server logs
2. Report bug if repeatable
3. Retry request
4. Contact support

---

### E402: Service Unavailable

**Description:** Service temporarily unavailable.

**Cause:**
- Maintenance mode
- Overloaded
- Startup in progress
- Shutting down

**HTTP Status:** 503 Service Unavailable

**Solution:**
1. Wait and retry
2. Check status page
3. Use backup service
4. Queue request

## Database Errors (E500-E599)

### E501: Qdrant Connection Failed

**Description:** Failed to connect to Qdrant.

**Cause:**
- Qdrant not running
- Wrong host/port
- Network issue
- Authentication failed

**HTTP Status:** 503 Service Unavailable

**Retry:** Yes

**Solution:**
1. Start Qdrant: `docker-compose up -d qdrant`
2. Check connection string
3. Verify network connectivity
4. Check credentials

---

### E502: Qdrant Collection Not Found

**Description:** Requested Qdrant collection doesn't exist.

**Cause:**
- Collection never created
- Collection deleted
- Wrong collection name
- Database reset

**HTTP Status:** 404 Not Found

**Solution:**
1. Create collection
2. Check collection name
3. Verify database state
4. Reload data if needed

---

### E503: Memgraph Connection Failed

**Description:** Failed to connect to Memgraph.

**Cause:**
- Memgraph not running
- Wrong bolt URI
- Network issue
- Authentication failed

**HTTP Status:** 503 Service Unavailable

**Retry:** Yes

**Solution:**
1. Start Memgraph: `docker-compose up -d memgraph`
2. Check bolt URI (bolt://localhost:7687)
3. Verify network connectivity
4. Check credentials

---

### E504: Memgraph Query Failed

**Description:** Cypher query execution failed.

**Cause:**
- Invalid Cypher syntax
- Query timeout
- Resource limit exceeded
- Constraint violation

**HTTP Status:** 400 Bad Request

**Solution:**
1. Validate Cypher syntax
2. Optimize query
3. Increase timeout
4. Check constraints

---

### E505: Marquez Connection Failed

**Description:** Failed to connect to Marquez API.

**Cause:**
- Marquez not running
- Wrong API URL
- Network issue
- API unavailable

**HTTP Status:** 503 Service Unavailable

**Retry:** Yes

**Solution:**
1. Start Marquez: `docker-compose up -d marquez`
2. Check API URL (http://localhost:5000)
3. Verify network connectivity
4. Check Marquez logs

---

### E506: Marquez Event Rejected

**Description:** OpenLineage event rejected by Marquez.

**Cause:**
- Invalid event format
- Missing required fields
- Wrong event type
- Schema validation failed

**HTTP Status:** 400 Bad Request

**Solution:**
1. Validate event format
2. Check OpenLineage schema
3. Include all required fields
4. Verify event type

## Error Handling Best Practices

### 1. Error Categorization

```python
def is_retryable(error_code: str) -> bool:
    """Check if error is retryable"""
    return error_code.startswith('E1')  # E100-E199

def is_critical(error_code: str) -> bool:
    """Check if error is critical"""
    return error_code.startswith('E0')  # E001-E099

def requires_user_action(error_code: str) -> bool:
    """Check if error requires user action"""
    return error_code.startswith('E3')  # E300-E399
```

### 2. Error Response Builder

```python
def build_error_response(
    code: str,
    message: str,
    details: str = None,
    suggestion: str = None
) -> dict:
    """Build standard error response"""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "suggestion": suggestion
        }
    }
```

### 3. Error Logging

```python
def log_error(error_code: str, context: dict):
    """Log error with context"""
    logger.error(
        "Error occurred",
        error_code=error_code,
        **context
    )
```

## Quick Reference Table

| Code | Category | Retry? | HTTP | Description |
|------|----------|--------|------|-------------|
| E001 | Critical | No | 400 | Invalid SCIP format |
| E002 | Critical | No | 400 | Corrupted file |
| E003 | Critical | No | 400 | Missing fields |
| E004 | Critical | No | 500 | Out of memory |
| E005 | Critical | No | 500 | Config error |
| E101 | Transient | Yes | 503 | DB timeout |
| E102 | Transient | Yes | 504 | Network timeout |
| E103 | Transient | Yes | 503 | DB unavailable |
| E104 | Transient | Yes | 429 | Rate limit |
| E105 | Transient | Yes | 503 | Lock contention |
| E201 | Warning | N/A | - | Symbol parse fail |
| E202 | Warning | N/A | - | Missing optional |
| E203 | Warning | N/A | - | Partial write |
| E204 | Info | N/A | - | Cache miss |
| E205 | Info | N/A | - | Non-existent ref |
| E301 | Client | No | 400 | Invalid request |
| E302 | Client | No | 401 | Unauthorized |
| E303 | Client | No | 404 | Not found |
| E304 | Client | No | 405 | Method not allowed |
| E305 | Client | No | 413 | Too large |
| E501 | Database | Yes | 503 | Qdrant connection |
| E502 | Database | No | 404 | Qdrant collection |
| E503 | Database | Yes | 503 | Memgraph connection |
| E504 | Database | No | 400 | Memgraph query |
| E505 | Database | Yes | 503 | Marquez connection |
| E506 | Database | No | 400 | Marquez event |

## Related Documentation

- [Error Handling Guide](ERROR_HANDLING.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [API Reference](API.md)
- [Runbook](RUNBOOK.md)

---

**Version:** 1.0  
**Last Updated:** 2026-01-18  
**Author:** nCode Development Team
