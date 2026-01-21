# nCode Error Handling & Resilience Guide

**Version:** 1.0  
**Last Updated:** 2026-01-18  
**Status:** Production Ready

## Overview

This document describes the comprehensive error handling and resilience mechanisms implemented in nCode to ensure reliable operation in production environments.

## Error Handling Philosophy

nCode follows these principles:

1. **Fail Fast for Critical Errors** - Invalid SCIP data, corrupt files
2. **Graceful Degradation** - Continue on non-critical errors
3. **Automatic Recovery** - Retry transient failures
4. **Clear Communication** - Provide actionable error messages
5. **Observable Failures** - Log all errors with context

## Error Categories

### 1. Critical Errors (Fail Fast)

These errors prevent the system from functioning and should halt execution:

| Error Code | Description | Response |
|------------|-------------|----------|
| E001 | Invalid SCIP protobuf format | Reject request, return 400 |
| E002 | Corrupted SCIP file | Reject request, return 400 |
| E003 | Missing required fields | Reject request, return 400 |
| E004 | Out of memory | Return 500, log critical |
| E005 | Configuration error | Fail startup, log critical |

**Example:**
```json
{
  "error": {
    "code": "E001",
    "message": "Invalid SCIP protobuf format",
    "details": "Failed to parse protobuf at offset 1234",
    "suggestion": "Verify SCIP index was generated correctly"
  }
}
```

### 2. Transient Errors (Retry)

These errors may succeed on retry:

| Error Code | Description | Retry Strategy |
|------------|-------------|----------------|
| E101 | Database connection timeout | Exponential backoff, 3 retries |
| E102 | Network timeout | Exponential backoff, 3 retries |
| E103 | Database temporarily unavailable | Exponential backoff, 5 retries |
| E104 | Rate limit exceeded | Wait and retry, 3 retries |
| E105 | Lock contention | Exponential backoff, 5 retries |

**Retry Configuration:**
```python
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,  # seconds
    "max_delay": 30.0,     # seconds
    "backoff_factor": 2.0,  # exponential
    "jitter": True          # add randomness
}
```

### 3. Non-Critical Errors (Continue)

These errors allow continued operation:

| Error Code | Description | Response |
|------------|-------------|----------|
| E201 | Single symbol parse failure | Skip symbol, continue |
| E202 | Missing optional field | Use default, continue |
| E203 | Partial database write | Log warning, continue |
| E204 | Cache miss | Fetch from source, continue |
| E205 | Non-existent reference | Skip reference, continue |

### 4. Client Errors (User Action Required)

These errors require user intervention:

| Error Code | Description | Response |
|------------|-------------|----------|
| E301 | Invalid API request | Return 400 with details |
| E302 | Unauthorized | Return 401 |
| E303 | Resource not found | Return 404 |
| E304 | Method not allowed | Return 405 |
| E305 | Request too large | Return 413 |

## Implementation Patterns

### 1. Retry Logic with Exponential Backoff

```python
import time
import random
from typing import Callable, TypeVar, Optional

T = TypeVar('T')

def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Multiplier for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exceptions to retry on
        
    Returns:
        Result of successful function call
        
    Raises:
        Last exception if all retries exhausted
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt >= max_retries:
                raise
            
            # Calculate delay with exponential backoff
            current_delay = min(delay * (backoff_factor ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                current_delay *= (0.5 + random.random())
            
            print(f"Retry {attempt + 1}/{max_retries} after {current_delay:.2f}s: {e}")
            time.sleep(current_delay)
    
    raise last_exception


# Usage example
def connect_to_database():
    """Connect to database with retry logic"""
    return retry_with_backoff(
        lambda: database.connect(),
        max_retries=5,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
```

### 2. Circuit Breaker Pattern

```python
import time
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                # Try to recover
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func()
            
            # Success - reset circuit
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            
            raise


# Usage example
qdrant_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60.0,
    expected_exception=ConnectionError
)

def search_qdrant(query: str):
    """Search Qdrant with circuit breaker protection"""
    return qdrant_breaker.call(
        lambda: qdrant_client.search(query)
    )
```

### 3. Graceful Degradation

```python
from typing import Optional, List

def load_to_databases(scip_index: str, skip_on_error: bool = True):
    """
    Load SCIP index to multiple databases with graceful degradation.
    
    Args:
        scip_index: Path to SCIP index file
        skip_on_error: Continue if one database fails
    
    Returns:
        Dictionary of database -> success status
    """
    results = {}
    
    # Load to Qdrant
    try:
        load_to_qdrant(scip_index)
        results['qdrant'] = {'success': True}
    except Exception as e:
        results['qdrant'] = {'success': False, 'error': str(e)}
        if not skip_on_error:
            raise
        print(f"Warning: Qdrant loading failed: {e}")
    
    # Load to Memgraph
    try:
        load_to_memgraph(scip_index)
        results['memgraph'] = {'success': True}
    except Exception as e:
        results['memgraph'] = {'success': False, 'error': str(e)}
        if not skip_on_error:
            raise
        print(f"Warning: Memgraph loading failed: {e}")
    
    # Load to Marquez
    try:
        load_to_marquez(scip_index)
        results['marquez'] = {'success': True}
    except Exception as e:
        results['marquez'] = {'success': False, 'error': str(e)}
        if not skip_on_error:
            raise
        print(f"Warning: Marquez loading failed: {e}")
    
    return results
```

### 4. Timeout Protection

```python
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    """Context manager to enforce timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# Usage example
try:
    with timeout(30):
        result = expensive_operation()
except TimeoutError as e:
    print(f"Operation timed out: {e}")
    # Handle timeout gracefully
```

## Database-Specific Error Handling

### Qdrant Errors

```python
from qdrant_client.http.exceptions import (
    UnexpectedResponse,
    RateLimitError,
    TimeoutException
)

def search_with_retry(collection: str, query: str) -> List[dict]:
    """Search Qdrant with comprehensive error handling"""
    
    def _search():
        try:
            return qdrant_client.search(
                collection_name=collection,
                query_vector=embed(query),
                limit=10
            )
        except TimeoutException:
            # Retry on timeout
            raise ConnectionError("Qdrant timeout")
        except RateLimitError:
            # Wait and retry
            time.sleep(1)
            raise ConnectionError("Rate limited")
        except UnexpectedResponse as e:
            if e.status_code >= 500:
                # Server error - retry
                raise ConnectionError(f"Qdrant server error: {e}")
            else:
                # Client error - don't retry
                raise ValueError(f"Invalid request: {e}")
    
    return retry_with_backoff(_search)
```

### Memgraph Errors

```python
from neo4j.exceptions import (
    ServiceUnavailable,
    TransientError,
    DatabaseError
)

def query_with_retry(cypher: str, params: dict = None) -> List[dict]:
    """Execute Cypher query with error handling"""
    
    def _query():
        try:
            with driver.session() as session:
                result = session.run(cypher, params or {})
                return list(result)
        except ServiceUnavailable:
            # Connection lost - retry
            raise ConnectionError("Memgraph unavailable")
        except TransientError as e:
            # Transient error - retry
            raise ConnectionError(f"Transient error: {e}")
        except DatabaseError as e:
            # Permanent error - don't retry
            raise ValueError(f"Query error: {e}")
    
    return retry_with_backoff(_query)
```

### Marquez Errors

```python
import requests

def post_event_with_retry(event: dict) -> dict:
    """Post OpenLineage event with error handling"""
    
    def _post():
        try:
            response = requests.post(
                "http://localhost:5000/api/v1/lineage",
                json=event,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            raise ConnectionError("Marquez timeout")
        except requests.ConnectionError:
            raise ConnectionError("Cannot connect to Marquez")
        except requests.HTTPError as e:
            if e.response.status_code >= 500:
                raise ConnectionError(f"Marquez server error: {e}")
            else:
                raise ValueError(f"Invalid event: {e}")
    
    return retry_with_backoff(_post)
```

## Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "E101",
    "message": "Database connection timeout",
    "details": "Failed to connect to Qdrant at localhost:6333",
    "timestamp": "2026-01-18T06:30:00Z",
    "request_id": "req_abc123",
    "suggestion": "Check that Qdrant is running and accessible"
  }
}
```

## Logging Best Practices

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Logger that outputs structured JSON logs"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log(self, level: str, message: str, **kwargs):
        """Log structured message"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        
        log_func = getattr(self.logger, level.lower())
        log_func(json.dumps(log_entry))
    
    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log("WARNING", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)


# Usage
logger = StructuredLogger("ncode")

try:
    result = connect_to_database()
except Exception as e:
    logger.error(
        "Database connection failed",
        error_code="E101",
        database="qdrant",
        host="localhost:6333",
        exception=str(e)
    )
```

## Health Checks

### Comprehensive Health Check

```python
from enum import Enum
from typing import Dict

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

def check_health() -> Dict:
    """Comprehensive health check"""
    checks = {}
    
    # Check Qdrant
    try:
        qdrant_client.get_collections()
        checks['qdrant'] = {'status': 'healthy', 'latency_ms': 10}
    except Exception as e:
        checks['qdrant'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check Memgraph
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        checks['memgraph'] = {'status': 'healthy', 'latency_ms': 5}
    except Exception as e:
        checks['memgraph'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check Marquez
    try:
        response = requests.get("http://localhost:5000/api/v1/namespaces")
        response.raise_for_status()
        checks['marquez'] = {'status': 'healthy', 'latency_ms': 20}
    except Exception as e:
        checks['marquez'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Determine overall status
    unhealthy = sum(1 for c in checks.values() if c['status'] == 'unhealthy')
    
    if unhealthy == 0:
        overall_status = HealthStatus.HEALTHY
    elif unhealthy < len(checks):
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.UNHEALTHY
    
    return {
        'status': overall_status.value,
        'timestamp': datetime.utcnow().isoformat(),
        'checks': checks
    }
```

## Testing Error Scenarios

### Test Suite for Error Handling

```python
import pytest
from unittest.mock import Mock, patch

def test_retry_on_timeout():
    """Test retry logic with timeout"""
    mock_func = Mock(side_effect=[TimeoutError(), TimeoutError(), "success"])
    
    result = retry_with_backoff(
        mock_func,
        max_retries=3,
        initial_delay=0.1
    )
    
    assert result == "success"
    assert mock_func.call_count == 3

def test_circuit_breaker_opens():
    """Test circuit breaker opens after threshold"""
    breaker = CircuitBreaker(failure_threshold=3)
    mock_func = Mock(side_effect=ConnectionError())
    
    # Trigger failures
    for _ in range(3):
        with pytest.raises(ConnectionError):
            breaker.call(mock_func)
    
    # Circuit should be open
    assert breaker.state == CircuitState.OPEN
    
    # Next call should fail immediately
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        breaker.call(mock_func)

def test_graceful_degradation():
    """Test continuing after non-critical error"""
    with patch('qdrant_client.search', side_effect=ConnectionError()):
        results = load_to_databases("index.scip", skip_on_error=True)
    
    assert results['qdrant']['success'] == False
    assert 'error' in results['qdrant']
```

## Production Deployment Checklist

- [ ] All critical errors have error codes
- [ ] Retry logic implemented for transient errors
- [ ] Circuit breakers configured for external services
- [ ] Graceful degradation enabled for non-critical failures
- [ ] Structured logging configured
- [ ] Health checks implemented and tested
- [ ] Error scenarios tested
- [ ] Monitoring alerts configured
- [ ] Error documentation complete
- [ ] Runbook created for common errors

## Common Error Scenarios & Solutions

### Scenario 1: Database Connection Lost

**Symptoms:** Connection errors, timeouts

**Solution:**
1. Retry with exponential backoff
2. If all retries fail, open circuit breaker
3. Return degraded status in health check
4. Alert operations team

### Scenario 2: Corrupt SCIP File

**Symptoms:** Protobuf parse errors

**Solution:**
1. Reject request with clear error message
2. Log error with file details
3. Suggest regenerating SCIP index
4. Don't retry (permanent error)

### Scenario 3: Memory Pressure

**Symptoms:** Out of memory errors

**Solution:**
1. Implement streaming for large files
2. Add memory limits
3. Implement backpressure
4. Return 503 if overloaded

### Scenario 4: Rate Limiting

**Symptoms:** 429 responses from databases

**Solution:**
1. Implement token bucket rate limiter
2. Add jitter to retries
3. Queue requests if possible
4. Return 503 if queue full

## Monitoring & Alerting

### Key Metrics to Monitor

1. **Error Rates**
   - Total errors per minute
   - Error rate by type
   - Error rate by endpoint

2. **Database Health**
   - Connection success rate
   - Query latency
   - Retry count

3. **Circuit Breaker State**
   - Number of open circuits
   - Half-open transition rate
   - Recovery time

4. **Resource Usage**
   - Memory usage
   - CPU usage
   - Connection pool size

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Error rate | >1% | >5% |
| Failed health checks | 1 | 2 |
| Open circuits | 1 | 3 |
| Response time | >500ms | >2s |
| Memory usage | >80% | >95% |

## References

- [Error Codes Reference](ERROR_CODES.md)
- [Runbook](RUNBOOK.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [API Documentation](API.md)

---

**Version:** 1.0  
**Last Updated:** 2026-01-18  
**Author:** nCode Development Team
