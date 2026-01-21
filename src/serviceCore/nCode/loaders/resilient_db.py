#!/usr/bin/env python3
"""
Resilient Database Connection Wrappers for nCode
Implements retry logic, circuit breakers, and graceful degradation
"""

import time
import random
import logging
from enum import Enum
from typing import Callable, TypeVar, Optional, Dict, Any, List
from datetime import datetime, timezone
from functools import wraps

# Third-party imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, TransientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class RetryConfig:
    """Configuration for retry logic"""
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    Automatically opens circuit after threshold failures,
    then tests recovery after timeout period.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection"""
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info(f"Circuit breaker {self.name}: Attempting recovery (HALF_OPEN)")
                self.state = CircuitState.HALF_OPEN
            else:
                remaining = self.timeout - (time.time() - self.last_failure_time)
                raise Exception(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Retry in {remaining:.1f}s"
                )
        
        try:
            result = func()
            
            # Success - reset circuit
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker {self.name}: Recovery successful (CLOSED)")
                self.state = CircuitState.CLOSED
            self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker {self.name}: Opening circuit "
                    f"({self.failure_count} failures)"
                )
                self.state = CircuitState.OPEN
            
            raise
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        logger.info(f"Circuit breaker {self.name}: Manually reset")


def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig = None,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        config: Retry configuration
        retryable_exceptions: Tuple of exceptions to retry on
        
    Returns:
        Result of successful function call
        
    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt >= config.max_retries:
                logger.error(
                    f"All {config.max_retries} retries exhausted",
                    extra={"exception": str(e)}
                )
                raise
            
            # Calculate delay with exponential backoff
            delay = config.initial_delay * (config.backoff_factor ** attempt)
            delay = min(delay, config.max_delay)
            
            # Add jitter to prevent thundering herd
            if config.jitter:
                delay *= (0.5 + random.random())
            
            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} after {delay:.2f}s",
                extra={"exception": str(e)}
            )
            time.sleep(delay)
    
    raise last_exception


class ResilientQdrantClient:
    """
    Qdrant client with retry logic and circuit breaker.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        retry_config: RetryConfig = None,
        circuit_breaker_config: Dict[str, Any] = None
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed")
        
        self.host = host
        self.port = port
        self.retry_config = retry_config or RetryConfig(max_retries=3)
        
        # Initialize circuit breaker
        cb_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            name="qdrant",
            failure_threshold=cb_config.get('failure_threshold', 5),
            timeout=cb_config.get('timeout', 60.0),
            expected_exception=ConnectionError
        )
        
        self._client = None
    
    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client
    
    def search(self, collection_name: str, **kwargs) -> List[Any]:
        """
        Search with retry and circuit breaker protection.
        
        Args:
            collection_name: Name of collection
            **kwargs: Search parameters
            
        Returns:
            Search results
        """
        def _search():
            try:
                return self.client.search(
                    collection_name=collection_name,
                    **kwargs
                )
            except UnexpectedResponse as e:
                if e.status_code >= 500:
                    raise ConnectionError(f"Qdrant server error: {e}")
                raise ValueError(f"Invalid request: {e}")
            except Exception as e:
                raise ConnectionError(f"Qdrant error: {e}")
        
        return self.circuit_breaker.call(
            lambda: retry_with_backoff(_search, self.retry_config)
        )
    
    def create_collection(self, collection_name: str, **kwargs):
        """Create collection with retry logic"""
        def _create():
            return self.client.create_collection(
                collection_name=collection_name,
                **kwargs
            )
        
        return retry_with_backoff(_create, self.retry_config)
    
    def upsert(self, collection_name: str, **kwargs):
        """Upsert points with retry logic"""
        def _upsert():
            return self.client.upsert(
                collection_name=collection_name,
                **kwargs
            )
        
        return retry_with_backoff(_upsert, self.retry_config)
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health"""
        try:
            collections = self.client.get_collections()
            return {
                'status': 'healthy',
                'collections': len(collections.collections),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


class ResilientMemgraphClient:
    """
    Memgraph client with retry logic and circuit breaker.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "",
        password: str = "",
        retry_config: RetryConfig = None,
        circuit_breaker_config: Dict[str, Any] = None
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not installed")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.retry_config = retry_config or RetryConfig(max_retries=3)
        
        # Initialize circuit breaker
        cb_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            name="memgraph",
            failure_threshold=cb_config.get('failure_threshold', 5),
            timeout=cb_config.get('timeout', 60.0),
            expected_exception=ConnectionError
        )
        
        self._driver = None
    
    @property
    def driver(self):
        """Get or create Memgraph driver"""
        if self._driver is None:
            if self.username:
                from neo4j import basic_auth
                auth = basic_auth(self.username, self.password)
            else:
                auth = None
            self._driver = GraphDatabase.driver(self.uri, auth=auth)
        return self._driver
    
    def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """
        Execute Cypher query with retry and circuit breaker protection.
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            
        Returns:
            Query results
        """
        def _query():
            try:
                with self.driver.session() as session:
                    result = session.run(cypher, params or {})
                    return [dict(record) for record in result]
            except ServiceUnavailable:
                raise ConnectionError("Memgraph unavailable")
            except TransientError as e:
                raise ConnectionError(f"Transient error: {e}")
            except Exception as e:
                # Don't retry syntax errors
                if "syntax" in str(e).lower():
                    raise ValueError(f"Query error: {e}")
                raise ConnectionError(f"Memgraph error: {e}")
        
        return self.circuit_breaker.call(
            lambda: retry_with_backoff(_query, self.retry_config)
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Check Memgraph health"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def close(self):
        """Close driver connection"""
        if self._driver:
            self._driver.close()


class ResilientMarquezClient:
    """
    Marquez client with retry logic and circuit breaker.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        retry_config: RetryConfig = None,
        circuit_breaker_config: Dict[str, Any] = None
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed")
        
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/v1/"
        self.retry_config = retry_config or RetryConfig(max_retries=3)
        
        # Initialize circuit breaker
        cb_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            name="marquez",
            failure_threshold=cb_config.get('failure_threshold', 5),
            timeout=cb_config.get('timeout', 60.0),
            expected_exception=ConnectionError
        )
    
    def post_event(self, event: Dict) -> Dict:
        """
        Post OpenLineage event with retry and circuit breaker protection.
        
        Args:
            event: OpenLineage event dict
            
        Returns:
            Response dict
        """
        def _post():
            try:
                response = requests.post(
                    f"{self.api_url}lineage",
                    json=event,
                    timeout=10
                )
                response.raise_for_status()
                return response.json() if response.text else {}
            except requests.Timeout:
                raise ConnectionError("Marquez timeout")
            except requests.ConnectionError:
                raise ConnectionError("Cannot connect to Marquez")
            except requests.HTTPError as e:
                if e.response.status_code >= 500:
                    raise ConnectionError(f"Marquez server error: {e}")
                raise ValueError(f"Invalid event: {e}")
        
        return self.circuit_breaker.call(
            lambda: retry_with_backoff(_post, self.retry_config)
        )
    
    def get_lineage(self, node_id: str) -> Dict:
        """Get lineage graph with retry logic"""
        def _get():
            response = requests.get(
                f"{self.api_url}lineage",
                params={'nodeId': node_id},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        return retry_with_backoff(_get, self.retry_config)
    
    def health_check(self) -> Dict[str, Any]:
        """Check Marquez health"""
        try:
            response = requests.get(
                f"{self.api_url}namespaces",
                timeout=5
            )
            response.raise_for_status()
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


class DatabaseHealthMonitor:
    """
    Monitor health of all database connections.
    """
    
    def __init__(
        self,
        qdrant_client: Optional[ResilientQdrantClient] = None,
        memgraph_client: Optional[ResilientMemgraphClient] = None,
        marquez_client: Optional[ResilientMarquezClient] = None
    ):
        self.clients = {}
        
        if qdrant_client:
            self.clients['qdrant'] = qdrant_client
        if memgraph_client:
            self.clients['memgraph'] = memgraph_client
        if marquez_client:
            self.clients['marquez'] = marquez_client
    
    def check_all(self) -> Dict[str, Any]:
        """Check health of all databases"""
        results = {}
        
        for name, client in self.clients.items():
            try:
                results[name] = client.health_check()
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        # Determine overall status
        unhealthy = sum(
            1 for r in results.values()
            if r.get('status') != 'healthy'
        )
        
        if unhealthy == 0:
            overall = 'healthy'
        elif unhealthy < len(results):
            overall = 'degraded'
        else:
            overall = 'unhealthy'
        
        return {
            'overall_status': overall,
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    print("nCode Resilient Database Clients")
    print("=" * 50)
    
    # Test Qdrant
    if QDRANT_AVAILABLE:
        print("\n[Qdrant] Testing resilient client...")
        try:
            qdrant = ResilientQdrantClient()
            health = qdrant.health_check()
            print(f"  Status: {health['status']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test Memgraph
    if NEO4J_AVAILABLE:
        print("\n[Memgraph] Testing resilient client...")
        try:
            memgraph = ResilientMemgraphClient()
            health = memgraph.health_check()
            print(f"  Status: {health['status']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test Marquez
    if REQUESTS_AVAILABLE:
        print("\n[Marquez] Testing resilient client...")
        try:
            marquez = ResilientMarquezClient()
            health = marquez.health_check()
            print(f"  Status: {health['status']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
