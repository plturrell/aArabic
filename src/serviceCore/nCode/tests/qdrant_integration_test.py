#!/usr/bin/env python3
"""
Qdrant Integration Test Suite for nCode
Tests semantic search, filtering, and performance benchmarks
"""

import time
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
except ImportError:
    print("❌ Qdrant client not installed. Install with: pip install qdrant-client")
    sys.exit(1)

class QdrantIntegrationTest:
    """Test suite for Qdrant integration with nCode"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client: Optional[QdrantClient] = None
        self.test_collection = "ncode_integration_test"
        self.results: Dict[str, Any] = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "benchmarks": {}
        }
    
    def connect(self) -> bool:
        """Test connection to Qdrant"""
        print("🔌 Testing Qdrant connection...")
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            collections = self.client.get_collections()
            print(f"✅ Connected to Qdrant at {self.host}:{self.port}")
            print(f"   Found {len(collections.collections)} existing collections")
            self.results["passed"].append("Connection established")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            self.results["failed"].append(f"Connection failed: {e}")
            return False
    
    def test_collection_creation(self) -> bool:
        """Test creating a collection with proper vector configuration"""
        print("\n📦 Testing collection creation...")
        try:
            # Delete test collection if it exists
            try:
                self.client.delete_collection(self.test_collection)
                print(f"   Cleaned up existing test collection")
            except:
                pass
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.test_collection,
                vectors_config=VectorParams(
                    size=384,  # Standard embedding size for sentence-transformers
                    distance=Distance.COSINE
                )
            )
            
            # Verify collection exists
            collection_info = self.client.get_collection(self.test_collection)
            print(f"✅ Collection '{self.test_collection}' created successfully")
            print(f"   Vector size: {collection_info.config.params.vectors.size}")
            print(f"   Distance metric: {collection_info.config.params.vectors.distance}")
            
            self.results["passed"].append("Collection creation")
            return True
        except Exception as e:
            print(f"❌ Collection creation failed: {e}")
            self.results["failed"].append(f"Collection creation: {e}")
            return False
    
    def test_data_insertion(self) -> bool:
        """Test inserting sample code symbol data"""
        print("\n📝 Testing data insertion...")
        try:
            # Sample code symbols
            sample_data = [
                {
                    "id": 1,
                    "vector": [0.1] * 384,  # Mock embedding
                    "payload": {
                        "symbol": "User#constructor()",
                        "kind": "method",
                        "file": "src/models/user.ts",
                        "language": "typescript",
                        "documentation": "Creates a new User instance",
                        "line": 23
                    }
                },
                {
                    "id": 2,
                    "vector": [0.2] * 384,
                    "payload": {
                        "symbol": "AuthService#authenticate()",
                        "kind": "method",
                        "file": "src/services/auth.ts",
                        "language": "typescript",
                        "documentation": "Authenticate user with credentials",
                        "line": 67
                    }
                },
                {
                    "id": 3,
                    "vector": [0.3] * 384,
                    "payload": {
                        "symbol": "DatabaseConnection#connect()",
                        "kind": "method",
                        "file": "src/services/database.ts",
                        "language": "typescript",
                        "documentation": "Connect to database",
                        "line": 45
                    }
                },
                {
                    "id": 4,
                    "vector": [0.15] * 384,
                    "payload": {
                        "symbol": "Product",
                        "kind": "class",
                        "file": "src/models/product.ts",
                        "language": "typescript",
                        "documentation": "Product class with inventory management",
                        "line": 28
                    }
                },
                {
                    "id": 5,
                    "vector": [0.25] * 384,
                    "payload": {
                        "symbol": "validateEmail()",
                        "kind": "function",
                        "file": "src/utils/helpers.ts",
                        "language": "typescript",
                        "documentation": "Validate email format",
                        "line": 10
                    }
                }
            ]
            
            # Insert data
            start_time = time.time()
            self.client.upsert(
                collection_name=self.test_collection,
                points=[
                    PointStruct(
                        id=item["id"],
                        vector=item["vector"],
                        payload=item["payload"]
                    )
                    for item in sample_data
                ]
            )
            insertion_time = time.time() - start_time
            
            # Verify insertion
            collection_info = self.client.get_collection(self.test_collection)
            print(f"✅ Inserted {len(sample_data)} code symbols")
            print(f"   Total points in collection: {collection_info.points_count}")
            print(f"   Insertion time: {insertion_time:.3f}s")
            
            self.results["passed"].append("Data insertion")
            self.results["benchmarks"]["insertion_time"] = insertion_time
            self.results["benchmarks"]["points_inserted"] = len(sample_data)
            return True
        except Exception as e:
            print(f"❌ Data insertion failed: {e}")
            self.results["failed"].append(f"Data insertion: {e}")
            return False
    
    def test_basic_search(self) -> bool:
        """Test basic vector search"""
        print("\n🔍 Testing basic semantic search...")
        try:
            # Search for similar vectors
            query_vector = [0.12] * 384  # Similar to User constructor
            
            start_time = time.time()
            search_results = self.client.search(
                collection_name=self.test_collection,
                query_vector=query_vector,
                limit=3
            )
            search_time = time.time() - start_time
            
            print(f"✅ Search completed in {search_time:.3f}s")
            print(f"   Found {len(search_results)} results:")
            for i, result in enumerate(search_results, 1):
                print(f"   {i}. {result.payload['symbol']} (score: {result.score:.3f})")
                print(f"      {result.payload['file']}")
            
            self.results["passed"].append("Basic search")
            self.results["benchmarks"]["basic_search_time"] = search_time
            return True
        except Exception as e:
            print(f"❌ Basic search failed: {e}")
            self.results["failed"].append(f"Basic search: {e}")
            return False
    
    def test_filtered_search(self) -> bool:
        """Test search with filters"""
        print("\n🎯 Testing filtered search...")
        try:
            # Search with language filter
            query_vector = [0.2] * 384
            
            search_results = self.client.search(
                collection_name=self.test_collection,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="kind",
                            match=MatchValue(value="method")
                        )
                    ]
                ),
                limit=5
            )
            
            print(f"✅ Filtered search completed")
            print(f"   Found {len(search_results)} methods:")
            for result in search_results:
                print(f"   - {result.payload['symbol']}")
                assert result.payload['kind'] == 'method', "Filter not working correctly"
            
            self.results["passed"].append("Filtered search")
            return True
        except Exception as e:
            print(f"❌ Filtered search failed: {e}")
            self.results["failed"].append(f"Filtered search: {e}")
            return False
    
    def test_multiple_filters(self) -> bool:
        """Test search with multiple filters"""
        print("\n🔬 Testing multi-filter search...")
        try:
            query_vector = [0.2] * 384
            
            search_results = self.client.search(
                collection_name=self.test_collection,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="language",
                            match=MatchValue(value="typescript")
                        ),
                        FieldCondition(
                            key="kind",
                            match=MatchValue(value="method")
                        )
                    ]
                ),
                limit=5
            )
            
            print(f"✅ Multi-filter search completed")
            print(f"   Found {len(search_results)} TypeScript methods")
            
            self.results["passed"].append("Multi-filter search")
            return True
        except Exception as e:
            print(f"❌ Multi-filter search failed: {e}")
            self.results["failed"].append(f"Multi-filter search: {e}")
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Benchmark search performance"""
        print("\n⚡ Running performance benchmarks...")
        try:
            query_vector = [0.15] * 384
            iterations = 100
            
            # Warm-up
            for _ in range(10):
                self.client.search(
                    collection_name=self.test_collection,
                    query_vector=query_vector,
                    limit=5
                )
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                self.client.search(
                    collection_name=self.test_collection,
                    query_vector=query_vector,
                    limit=5
                )
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            
            print(f"✅ Performance benchmark completed")
            print(f"   Iterations: {iterations}")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average time: {avg_time*1000:.2f}ms per search")
            print(f"   Throughput: {iterations/total_time:.1f} searches/second")
            
            self.results["passed"].append("Performance benchmark")
            self.results["benchmarks"]["avg_search_time_ms"] = avg_time * 1000
            self.results["benchmarks"]["searches_per_second"] = iterations / total_time
            
            # Check if performance meets targets
            if avg_time * 1000 < 50:  # Target: <50ms per search
                print(f"   ✅ Performance excellent (<50ms)")
            elif avg_time * 1000 < 100:
                print(f"   ⚠️  Performance acceptable (<100ms)")
                self.results["warnings"].append("Search performance acceptable but could be improved")
            else:
                print(f"   ❌ Performance below target (>100ms)")
                self.results["warnings"].append("Search performance needs optimization")
            
            return True
        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            self.results["failed"].append(f"Performance benchmark: {e}")
            return False
    
    def test_payload_retrieval(self) -> bool:
        """Test retrieving full payload data"""
        print("\n📄 Testing payload retrieval...")
        try:
            # Get specific point
            point = self.client.retrieve(
                collection_name=self.test_collection,
                ids=[1]
            )[0]
            
            print(f"✅ Payload retrieved successfully")
            print(f"   Symbol: {point.payload['symbol']}")
            print(f"   File: {point.payload['file']}")
            print(f"   Line: {point.payload['line']}")
            print(f"   Documentation: {point.payload['documentation']}")
            
            self.results["passed"].append("Payload retrieval")
            return True
        except Exception as e:
            print(f"❌ Payload retrieval failed: {e}")
            self.results["failed"].append(f"Payload retrieval: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Clean up test collection"""
        print("\n🧹 Cleaning up test collection...")
        try:
            self.client.delete_collection(self.test_collection)
            print(f"✅ Test collection deleted")
            return True
        except Exception as e:
            print(f"⚠️  Failed to delete test collection: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total = len(self.results["passed"]) + len(self.results["failed"])
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        
        print(f"\n✅ Passed: {passed}/{total}")
        for test in self.results["passed"]:
            print(f"   - {test}")
        
        if self.results["failed"]:
            print(f"\n❌ Failed: {failed}/{total}")
            for test in self.results["failed"]:
                print(f"   - {test}")
        
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings: {len(self.results['warnings'])}")
            for warning in self.results["warnings"]:
                print(f"   - {warning}")
        
        if self.results["benchmarks"]:
            print(f"\n⚡ Performance Benchmarks:")
            for metric, value in self.results["benchmarks"].items():
                if isinstance(value, float):
                    print(f"   - {metric}: {value:.3f}")
                else:
                    print(f"   - {metric}: {value}")
        
        print("\n" + "=" * 60)
        
        # Return exit code
        return 0 if failed == 0 else 1

def main():
    """Run all tests"""
    print("🚀 nCode Qdrant Integration Test Suite")
    print("=" * 60)
    
    tester = QdrantIntegrationTest()
    
    # Run tests
    if not tester.connect():
        return 1
    
    tests = [
        tester.test_collection_creation,
        tester.test_data_insertion,
        tester.test_basic_search,
        tester.test_filtered_search,
        tester.test_multiple_filters,
        tester.test_payload_retrieval,
        tester.test_performance_benchmark,
    ]
    
    for test in tests:
        if not test():
            print(f"\n⚠️  Test failed, but continuing with remaining tests...\n")
    
    # Cleanup
    tester.cleanup()
    
    # Print summary
    exit_code = tester.print_summary()
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
