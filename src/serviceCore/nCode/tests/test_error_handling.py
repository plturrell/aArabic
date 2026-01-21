#!/usr/bin/env python3
"""
Error Handling Test Suite for nCode
Tests retry logic, circuit breakers, and failure scenarios
"""

import sys
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'loaders'))

from resilient_db import (
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    retry_with_backoff,
    ResilientQdrantClient,
    ResilientMemgraphClient,
    ResilientMarquezClient,
    DatabaseHealthMonitor
)

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


class TestRetryLogic:
    """Test retry with exponential backoff"""
    
    def test_retry_success_first_attempt(self):
        """Test successful operation on first attempt"""
        mock_func = Mock(return_value="success")
        
        result = retry_with_backoff(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 1
        print(f"{GREEN}✓ Test passed: Success on first attempt{RESET}")
    
    def test_retry_success_after_failures(self):
        """Test success after some failures"""
        mock_func = Mock(side_effect=[
            ConnectionError("fail 1"),
            ConnectionError("fail 2"),
            "success"
        ])
        
        config = RetryConfig(max_retries=3, initial_delay=0.1)
        result = retry_with_backoff(mock_func, config)
        
        assert result == "success"
        assert mock_func.call_count == 3
        print(f"{GREEN}✓ Test passed: Success after 2 failures{RESET}")
    
    def test_retry_exhausted(self):
        """Test all retries exhausted"""
        mock_func = Mock(side_effect=ConnectionError("always fails"))
        
        config = RetryConfig(max_retries=2, initial_delay=0.1)
        
        with pytest.raises(ConnectionError):
            retry_with_backoff(mock_func, config)
        
        assert mock_func.call_count == 3  # initial + 2 retries
        print(f"{GREEN}✓ Test passed: All retries exhausted{RESET}")
    
    def test_retry_non_retryable_exception(self):
        """Test non-retryable exception fails immediately"""
        mock_func = Mock(side_effect=ValueError("non-retryable"))
        
        config = RetryConfig(max_retries=3, initial_delay=0.1)
        
        with pytest.raises(ValueError):
            retry_with_backoff(mock_func, config)
        
        assert mock_func.call_count == 1  # No retries
        print(f"{GREEN}✓ Test passed: Non-retryable exception{RESET}")


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    def test_circuit_closed_allows_calls(self):
        """Test circuit breaker allows calls when closed"""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        mock_func = Mock(return_value="success")
        
        result = breaker.call(mock_func)
        
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        print(f"{GREEN}✓ Test passed: Closed circuit allows calls{RESET}")
    
    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=3,
            expected_exception=ConnectionError
        )
        mock_func = Mock(side_effect=ConnectionError("fail"))
        
        # Trigger failures
        for _ in range(3):
            with pytest.raises(ConnectionError):
                breaker.call(mock_func)
        
        assert breaker.state == CircuitState.OPEN
        assert mock_func.call_count == 3
        print(f"{GREEN}✓ Test passed: Circuit opens after threshold{RESET}")
    
    def test_circuit_open_rejects_calls(self):
        """Test open circuit rejects calls immediately"""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            expected_exception=ConnectionError
        )
        mock_func = Mock(side_effect=ConnectionError("fail"))
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                breaker.call(mock_func)
        
        # Next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            breaker.call(mock_func)
        
        assert mock_func.call_count == 2  # No additional call
        print(f"{GREEN}✓ Test passed: Open circuit rejects calls{RESET}")
    
    def test_circuit_half_open_recovery(self):
        """Test circuit transitions to half-open and recovers"""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            timeout=0.1,  # Short timeout for testing
            expected_exception=ConnectionError
        )
        
        # Open the circuit
        mock_func = Mock(side_effect=ConnectionError("fail"))
        for _ in range(2):
            with pytest.raises(ConnectionError):
                breaker.call(mock_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Next successful call should close circuit
        mock_func = Mock(return_value="success")
        result = breaker.call(mock_func)
        
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        print(f"{GREEN}✓ Test passed: Circuit recovers via half-open{RESET}")


class TestResilientQdrantClient:
    """Test Qdrant resilient client"""
    
    @patch('resilient_db.QDRANT_AVAILABLE', True)
    @patch('resilient_db.QdrantClient')
    def test_search_with_retry(self, mock_qdrant_class):
        """Test search with retry on failure"""
        mock_client = Mock()
        mock_client.search.side_effect = [
            ConnectionError("timeout"),
            ["result1", "result2"]
        ]
        mock_qdrant_class.return_value = mock_client
        
        config = RetryConfig(max_retries=2, initial_delay=0.1)
        client = ResilientQdrantClient(retry_config=config)
        client._client = mock_client
        
        results = client.search("test_collection", query_vector=[1, 2, 3])
        
        assert results == ["result1", "result2"]
        assert mock_client.search.call_count == 2
        print(f"{GREEN}✓ Test passed: Qdrant search with retry{RESET}")
    
    @patch('resilient_db.QDRANT_AVAILABLE', True)
    @patch('resilient_db.QdrantClient')
    def test_circuit_breaker_opens(self, mock_qdrant_class):
        """Test circuit breaker opens after failures"""
        mock_client = Mock()
        mock_client.search.side_effect = ConnectionError("fail")
        mock_qdrant_class.return_value = mock_client
        
        config = RetryConfig(max_retries=1, initial_delay=0.1)
        cb_config = {'failure_threshold': 2}
        client = ResilientQdrantClient(
            retry_config=config,
            circuit_breaker_config=cb_config
        )
        client._client = mock_client
        
        # Trigger failures
        for _ in range(2):
            with pytest.raises(ConnectionError):
                client.search("test", query_vector=[1, 2, 3])
        
        assert client.circuit_breaker.state == CircuitState.OPEN
        print(f"{GREEN}✓ Test passed: Qdrant circuit breaker opens{RESET}")


class TestResilientMemgraphClient:
    """Test Memgraph resilient client"""
    
    @patch('resilient_db.NEO4J_AVAILABLE', True)
    @patch('resilient_db.GraphDatabase')
    def test_query_with_retry(self, mock_graph_db):
        """Test Cypher query with retry"""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([{"value": 1}]))
        
        mock_session.run.side_effect = [
            ConnectionError("timeout"),
            mock_result
        ]
        
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_graph_db.driver.return_value = mock_driver
        
        config = RetryConfig(max_retries=2, initial_delay=0.1)
        client = ResilientMemgraphClient(retry_config=config)
        client._driver = mock_driver
        
        results = client.query("RETURN 1 as value")
        
        assert len(results) == 1
        assert mock_session.run.call_count == 2
        print(f"{GREEN}✓ Test passed: Memgraph query with retry{RESET}")


class TestDatabaseHealthMonitor:
    """Test database health monitoring"""
    
    def test_all_healthy(self):
        """Test all databases healthy"""
        mock_qdrant = Mock()
        mock_qdrant.health_check.return_value = {'status': 'healthy'}
        
        mock_memgraph = Mock()
        mock_memgraph.health_check.return_value = {'status': 'healthy'}
        
        monitor = DatabaseHealthMonitor(
            qdrant_client=mock_qdrant,
            memgraph_client=mock_memgraph
        )
        
        health = monitor.check_all()
        
        assert health['overall_status'] == 'healthy'
        assert len(health['checks']) == 2
        print(f"{GREEN}✓ Test passed: All databases healthy{RESET}")
    
    def test_degraded_state(self):
        """Test degraded state with one unhealthy database"""
        mock_qdrant = Mock()
        mock_qdrant.health_check.return_value = {'status': 'healthy'}
        
        mock_memgraph = Mock()
        mock_memgraph.health_check.return_value = {'status': 'unhealthy'}
        
        monitor = DatabaseHealthMonitor(
            qdrant_client=mock_qdrant,
            memgraph_client=mock_memgraph
        )
        
        health = monitor.check_all()
        
        assert health['overall_status'] == 'degraded'
        print(f"{GREEN}✓ Test passed: Degraded state detected{RESET}")
    
    def test_all_unhealthy(self):
        """Test all databases unhealthy"""
        mock_qdrant = Mock()
        mock_qdrant.health_check.return_value = {'status': 'unhealthy'}
        
        mock_memgraph = Mock()
        mock_memgraph.health_check.return_value = {'status': 'unhealthy'}
        
        monitor = DatabaseHealthMonitor(
            qdrant_client=mock_qdrant,
            memgraph_client=mock_memgraph
        )
        
        health = monitor.check_all()
        
        assert health['overall_status'] == 'unhealthy'
        print(f"{GREEN}✓ Test passed: All databases unhealthy{RESET}")


def run_all_tests():
    """Run all error handling tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}nCode Error Handling Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    test_classes = [
        TestRetryLogic,
        TestCircuitBreaker,
        TestResilientQdrantClient,
        TestResilientMemgraphClient,
        TestDatabaseHealthMonitor
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{BLUE}Running {test_class.__name__}...{RESET}")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [
            method for method in dir(test_instance)
            if method.startswith('test_')
        ]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                print(f"{RED}✗ Test failed: {method_name}{RESET}")
                print(f"  Error: {e}")
    
    # Print summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    color = GREEN if pass_rate == 100 else RED
    
    print(f"Results: {color}{passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%){RESET}\n")
    
    if passed_tests == total_tests:
        print(f"{GREEN}✓ All tests passed! Error handling is working correctly.{RESET}\n")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Review errors above.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
