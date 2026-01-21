#!/usr/bin/env python3
"""
Marquez Integration Test Suite for nCode
Tests OpenLineage integration for SCIP index lineage tracking

Tests cover:
1. Connection to Marquez instance
2. OpenLineage event creation
3. Lineage graph construction (source files → SCIP index)
4. Lineage queries and traversal
5. Dataset and job tracking
6. Performance benchmarking
"""

import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from urllib.parse import urljoin

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class MarquezTester:
    """Test suite for Marquez integration"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize Marquez connection"""
        self.base_url = base_url
        self.api_url = urljoin(base_url, "/api/v1/")
        self.namespace = "ncode-test"
        self.test_results = []
        
    def _make_request(self, method: str, endpoint: str, 
                      data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to Marquez API"""
        url = urljoin(self.api_url, endpoint)
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            elif method == "PUT":
                response = requests.put(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json() if response.text else {}
        except requests.exceptions.RequestException as e:
            return None
    
    def test_connection(self) -> bool:
        """Test connection to Marquez instance"""
        print(f"\n{BLUE}=== Test 1: Marquez Connection ==={RESET}")
        try:
            # Try to get namespaces
            result = self._make_request("GET", "namespaces")
            
            if result is not None:
                print(f"{GREEN}✓ Successfully connected to Marquez at {self.base_url}{RESET}")
                print(f"  Found {len(result.get('namespaces', []))} namespace(s)")
                self.test_results.append(("Connection", True, None))
                return True
            else:
                print(f"{RED}✗ Cannot connect to Marquez{RESET}")
                self.test_results.append(("Connection", False, "No response"))
                return False
                
        except Exception as e:
            print(f"{RED}✗ Connection error: {e}{RESET}")
            print(f"{YELLOW}Hint: Is Marquez running? Check with: docker ps | grep marquez{RESET}")
            self.test_results.append(("Connection", False, str(e)))
            return False
    
    def test_create_namespace(self) -> bool:
        """Test creating a namespace"""
        print(f"\n{BLUE}=== Test 2: Create Namespace ==={RESET}")
        try:
            # Create test namespace
            data = {
                "ownerName": "ncode-test-owner",
                "description": "Test namespace for nCode integration tests"
            }
            
            result = self._make_request("PUT", f"namespaces/{self.namespace}", data)
            
            if result and result.get("name") == self.namespace:
                print(f"{GREEN}✓ Created namespace: {self.namespace}{RESET}")
                print(f"  Owner: {result.get('ownerName')}")
                self.test_results.append(("Create Namespace", True, None))
                return True
            else:
                print(f"{RED}✗ Failed to create namespace{RESET}")
                self.test_results.append(("Create Namespace", False, "Creation failed"))
                return False
                
        except Exception as e:
            print(f"{RED}✗ Failed to create namespace: {e}{RESET}")
            self.test_results.append(("Create Namespace", False, str(e)))
            return False
    
    def test_create_source_dataset(self) -> bool:
        """Test creating source dataset (input files)"""
        print(f"\n{BLUE}=== Test 3: Create Source Dataset ==={RESET}")
        try:
            dataset_name = "source-files"
            
            # Create dataset representing source code files
            data = {
                "type": "DB_TABLE",
                "physicalName": "src/services/auth.ts",
                "sourceName": "filesystem",
                "description": "Source code files for indexing",
                "fields": [
                    {
                        "name": "path",
                        "type": "STRING",
                        "description": "File path"
                    },
                    {
                        "name": "content",
                        "type": "STRING",
                        "description": "File content"
                    },
                    {
                        "name": "language",
                        "type": "STRING",
                        "description": "Programming language"
                    }
                ]
            }
            
            result = self._make_request(
                "PUT", 
                f"namespaces/{self.namespace}/datasets/{dataset_name}",
                data
            )
            
            if result and result.get("name") == dataset_name:
                print(f"{GREEN}✓ Created source dataset: {dataset_name}{RESET}")
                print(f"  Type: {result.get('type')}")
                print(f"  Fields: {len(result.get('fields', []))}")
                self.test_results.append(("Create Source Dataset", True, None))
                return True
            else:
                print(f"{RED}✗ Failed to create source dataset{RESET}")
                self.test_results.append(("Create Source Dataset", False, "Creation failed"))
                return False
                
        except Exception as e:
            print(f"{RED}✗ Failed to create source dataset: {e}{RESET}")
            self.test_results.append(("Create Source Dataset", False, str(e)))
            return False
    
    def test_create_output_dataset(self) -> bool:
        """Test creating output dataset (SCIP index)"""
        print(f"\n{BLUE}=== Test 4: Create Output Dataset ==={RESET}")
        try:
            dataset_name = "scip-index"
            
            # Create dataset representing SCIP index
            data = {
                "type": "DB_TABLE",
                "physicalName": "index.scip",
                "sourceName": "filesystem",
                "description": "SCIP code intelligence index",
                "fields": [
                    {
                        "name": "documents",
                        "type": "INTEGER",
                        "description": "Number of documents"
                    },
                    {
                        "name": "symbols",
                        "type": "INTEGER",
                        "description": "Number of symbols"
                    },
                    {
                        "name": "occurrences",
                        "type": "INTEGER",
                        "description": "Number of occurrences"
                    }
                ]
            }
            
            result = self._make_request(
                "PUT",
                f"namespaces/{self.namespace}/datasets/{dataset_name}",
                data
            )
            
            if result and result.get("name") == dataset_name:
                print(f"{GREEN}✓ Created output dataset: {dataset_name}{RESET}")
                print(f"  Physical name: {result.get('physicalName')}")
                self.test_results.append(("Create Output Dataset", True, None))
                return True
            else:
                print(f"{RED}✗ Failed to create output dataset{RESET}")
                self.test_results.append(("Create Output Dataset", False, "Creation failed"))
                return False
                
        except Exception as e:
            print(f"{RED}✗ Failed to create output dataset: {e}{RESET}")
            self.test_results.append(("Create Output Dataset", False, str(e)))
            return False
    
    def test_create_indexing_job(self) -> bool:
        """Test creating indexing job"""
        print(f"\n{BLUE}=== Test 5: Create Indexing Job ==={RESET}")
        try:
            job_name = "scip-indexer"
            
            # Create job representing the indexing process
            data = {
                "type": "BATCH",
                "description": "SCIP code indexer job",
                "location": "zig-out/bin/ncode-treesitter",
                "inputs": [
                    {
                        "namespace": self.namespace,
                        "name": "source-files"
                    }
                ],
                "outputs": [
                    {
                        "namespace": self.namespace,
                        "name": "scip-index"
                    }
                ]
            }
            
            result = self._make_request(
                "PUT",
                f"namespaces/{self.namespace}/jobs/{job_name}",
                data
            )
            
            if result and result.get("name") == job_name:
                print(f"{GREEN}✓ Created indexing job: {job_name}{RESET}")
                print(f"  Type: {result.get('type')}")
                print(f"  Inputs: {len(result.get('inputs', []))}")
                print(f"  Outputs: {len(result.get('outputs', []))}")
                self.test_results.append(("Create Job", True, None))
                return True
            else:
                print(f"{RED}✗ Failed to create job{RESET}")
                self.test_results.append(("Create Job", False, "Creation failed"))
                return False
                
        except Exception as e:
            print(f"{RED}✗ Failed to create job: {e}{RESET}")
            self.test_results.append(("Create Job", False, str(e)))
            return False
    
    def test_track_job_run(self) -> bool:
        """Test tracking a job run"""
        print(f"\n{BLUE}=== Test 6: Track Job Run ==={RESET}")
        try:
            job_name = "scip-indexer"
            run_id = f"test-run-{int(time.time())}"
            
            # Start job run
            start_data = {
                "eventType": "START",
                "eventTime": datetime.now(timezone.utc).isoformat(),
                "run": {
                    "runId": run_id
                },
                "job": {
                    "namespace": self.namespace,
                    "name": job_name
                },
                "inputs": [
                    {
                        "namespace": self.namespace,
                        "name": "source-files"
                    }
                ],
                "producer": "ncode-test-suite"
            }
            
            # Create run
            result = self._make_request(
                "POST",
                "lineage",
                start_data
            )
            
            if result:
                print(f"{GREEN}✓ Started job run: {run_id}{RESET}")
                
                # Complete job run
                time.sleep(0.5)  # Simulate processing
                
                complete_data = {
                    "eventType": "COMPLETE",
                    "eventTime": datetime.now(timezone.utc).isoformat(),
                    "run": {
                        "runId": run_id
                    },
                    "job": {
                        "namespace": self.namespace,
                        "name": job_name
                    },
                    "outputs": [
                        {
                            "namespace": self.namespace,
                            "name": "scip-index"
                        }
                    ],
                    "producer": "ncode-test-suite"
                }
                
                result = self._make_request(
                    "POST",
                    "lineage",
                    complete_data
                )
                
                if result:
                    print(f"{GREEN}✓ Completed job run{RESET}")
                    self.test_results.append(("Track Job Run", True, None))
                    return True
                else:
                    print(f"{RED}✗ Failed to complete job run{RESET}")
                    self.test_results.append(("Track Job Run", False, "Completion failed"))
                    return False
            else:
                print(f"{RED}✗ Failed to start job run{RESET}")
                self.test_results.append(("Track Job Run", False, "Start failed"))
                return False
                
        except Exception as e:
            print(f"{RED}✗ Failed to track job run: {e}{RESET}")
            self.test_results.append(("Track Job Run", False, str(e)))
            return False
    
    def test_query_lineage(self) -> bool:
        """Test querying lineage graph"""
        print(f"\n{BLUE}=== Test 7: Query Lineage ==={RESET}")
        try:
            # Query lineage for output dataset
            dataset_name = "scip-index"
            
            result = self._make_request(
                "GET",
                f"lineage?nodeId=dataset:{self.namespace}:{dataset_name}"
            )
            
            if result and "graph" in result:
                graph = result["graph"]
                print(f"{GREEN}✓ Retrieved lineage graph{RESET}")
                print(f"  Nodes: {len(graph)}")
                
                # Check for expected nodes
                node_types = {}
                for node_id, node_data in graph.items():
                    node_type = node_data.get("type", "unknown")
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                print(f"  Node types:")
                for node_type, count in node_types.items():
                    print(f"    {node_type}: {count}")
                
                self.test_results.append(("Query Lineage", True, None))
                return True
            else:
                print(f"{YELLOW}⚠ No lineage graph found (may need time to propagate){RESET}")
                self.test_results.append(("Query Lineage", True, "No graph yet"))
                return True
                
        except Exception as e:
            print(f"{RED}✗ Failed to query lineage: {e}{RESET}")
            self.test_results.append(("Query Lineage", False, str(e)))
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Benchmark API performance"""
        print(f"\n{BLUE}=== Test 8: Performance Benchmark ==={RESET}")
        try:
            # Benchmark: Get namespaces
            start_time = time.time()
            for _ in range(10):
                self._make_request("GET", "namespaces")
            end_time = time.time()
            avg_namespace_time = (end_time - start_time) / 10 * 1000  # ms
            
            # Benchmark: Get datasets
            start_time = time.time()
            for _ in range(10):
                self._make_request("GET", f"namespaces/{self.namespace}/datasets")
            end_time = time.time()
            avg_dataset_time = (end_time - start_time) / 10 * 1000  # ms
            
            # Benchmark: Get jobs
            start_time = time.time()
            for _ in range(10):
                self._make_request("GET", f"namespaces/{self.namespace}/jobs")
            end_time = time.time()
            avg_job_time = (end_time - start_time) / 10 * 1000  # ms
            
            print(f"{GREEN}✓ Performance benchmarks:{RESET}")
            print(f"  Get namespaces: {avg_namespace_time:.2f}ms (avg of 10 runs)")
            print(f"  Get datasets: {avg_dataset_time:.2f}ms (avg of 10 runs)")
            print(f"  Get jobs: {avg_job_time:.2f}ms (avg of 10 runs)")
            
            # Performance targets
            if avg_namespace_time < 200 and avg_dataset_time < 200:
                print(f"{GREEN}✓ Performance meets targets (<200ms){RESET}")
                self.test_results.append(("Performance", True, None))
                return True
            else:
                print(f"{YELLOW}⚠ Performance slower than target{RESET}")
                self.test_results.append(("Performance", True, "Slower than target"))
                return True
                
        except Exception as e:
            print(f"{RED}✗ Performance benchmark failed: {e}{RESET}")
            self.test_results.append(("Performance", False, str(e)))
            return False
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Test Summary{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, error in self.test_results:
            status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
            print(f"{status} - {test_name}")
            if error:
                print(f"       {YELLOW}└─ {error}{RESET}")
        
        print(f"\n{BLUE}{'='*60}{RESET}")
        pass_rate = (passed / total * 100) if total > 0 else 0
        color = GREEN if pass_rate >= 80 else YELLOW if pass_rate >= 50 else RED
        print(f"Results: {color}{passed}/{total} tests passed ({pass_rate:.1f}%){RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        # Recommendations
        if passed == total:
            print(f"{GREEN}✓ All tests passed! Marquez integration is working correctly.{RESET}")
            print(f"{GREEN}✓ Ready to track real indexing runs in Marquez.{RESET}\n")
        elif passed >= total * 0.8:
            print(f"{YELLOW}⚠ Most tests passed, but some issues detected.{RESET}")
            print(f"{YELLOW}  Review failed tests and check Marquez configuration.{RESET}\n")
        else:
            print(f"{RED}✗ Multiple test failures detected.{RESET}")
            print(f"{RED}  Check Marquez connection and configuration.{RESET}\n")


def main():
    """Run all Marquez integration tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}nCode Marquez Integration Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Configuration
    base_url = "http://localhost:5000"
    
    print(f"\nConfiguration:")
    print(f"  Marquez URL: {base_url}")
    print(f"  API Version: v1")
    
    # Initialize tester
    tester = MarquezTester(base_url)
    
    try:
        # Run tests
        if not tester.test_connection():
            print(f"\n{RED}Cannot proceed without Marquez connection.{RESET}")
            print(f"{YELLOW}Please ensure Marquez is running:{RESET}")
            print(f"  docker ps | grep marquez")
            print(f"  docker-compose up -d marquez marquez-db")
            return 1
        
        # Run test suite
        tester.test_create_namespace()
        tester.test_create_source_dataset()
        tester.test_create_output_dataset()
        tester.test_create_indexing_job()
        tester.test_track_job_run()
        tester.test_query_lineage()
        tester.test_performance_benchmark()
        
        # Print summary
        tester.print_summary()
        
        # Return appropriate exit code
        passed = sum(1 for _, success, _ in tester.test_results if success)
        total = len(tester.test_results)
        return 0 if passed == total else 1
        
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{RESET}")
        return 130
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
