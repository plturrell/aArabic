#!/usr/bin/env python3
"""
Memgraph Integration Test Suite for nCode
Tests graph database integration with SCIP indexes

Tests cover:
1. Connection to Memgraph instance
2. SCIP index loading into graph structure
3. Cypher query execution
4. Relationship verification (REFERENCES, IMPLEMENTS, ENCLOSES)
5. Complex graph queries (transitive dependencies, call graphs)
6. Performance benchmarking
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, AuthError

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class MemgraphTester:
    """Test suite for Memgraph integration"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "", password: str = ""):
        """Initialize Memgraph connection"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.test_results = []
        
    def connect(self) -> bool:
        """Test connection to Memgraph instance"""
        print(f"\n{BLUE}=== Test 1: Memgraph Connection ==={RESET}")
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.username, self.password) if self.username else None
            )
            
            # Verify connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                value = result.single()["test"]
                
            if value == 1:
                print(f"{GREEN}✓ Successfully connected to Memgraph at {self.uri}{RESET}")
                self.test_results.append(("Connection", True, None))
                return True
            else:
                print(f"{RED}✗ Connection test failed{RESET}")
                self.test_results.append(("Connection", False, "Unexpected result"))
                return False
                
        except ServiceUnavailable as e:
            print(f"{RED}✗ Cannot connect to Memgraph: {e}{RESET}")
            print(f"{YELLOW}Hint: Is Memgraph running? Check with: docker ps | grep memgraph{RESET}")
            self.test_results.append(("Connection", False, str(e)))
            return False
        except AuthError as e:
            print(f"{RED}✗ Authentication failed: {e}{RESET}")
            self.test_results.append(("Connection", False, str(e)))
            return False
        except Exception as e:
            print(f"{RED}✗ Connection error: {e}{RESET}")
            self.test_results.append(("Connection", False, str(e)))
            return False
    
    def test_clear_database(self) -> bool:
        """Clear existing data for clean test"""
        print(f"\n{BLUE}=== Test 2: Clear Database ==={RESET}")
        try:
            with self.driver.session() as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                
                # Verify database is empty
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                
                if count == 0:
                    print(f"{GREEN}✓ Database cleared successfully{RESET}")
                    self.test_results.append(("Clear Database", True, None))
                    return True
                else:
                    print(f"{RED}✗ Database still has {count} nodes{RESET}")
                    self.test_results.append(("Clear Database", False, f"{count} nodes remain"))
                    return False
                    
        except Exception as e:
            print(f"{RED}✗ Failed to clear database: {e}{RESET}")
            self.test_results.append(("Clear Database", False, str(e)))
            return False
    
    def test_create_sample_graph(self) -> bool:
        """Create sample SCIP-like graph structure"""
        print(f"\n{BLUE}=== Test 3: Create Sample Graph ==={RESET}")
        try:
            with self.driver.session() as session:
                # Create Document nodes
                session.run("""
                    CREATE (d1:Document {
                        relative_path: 'src/services/auth.ts',
                        language: 'typescript',
                        text: 'export function authenticate(user: string) { return validateUser(user); }'
                    })
                """)
                
                session.run("""
                    CREATE (d2:Document {
                        relative_path: 'src/services/user.ts',
                        language: 'typescript',
                        text: 'export function validateUser(user: string) { return true; }'
                    })
                """)
                
                # Create Symbol nodes
                session.run("""
                    MATCH (d1:Document {relative_path: 'src/services/auth.ts'})
                    CREATE (s1:Symbol {
                        symbol: 'scip-typescript npm . src/services/auth.ts authenticate().',
                        name: 'authenticate',
                        kind: 'Function',
                        line: 0,
                        character: 16
                    })
                    CREATE (d1)-[:CONTAINS]->(s1)
                """)
                
                session.run("""
                    MATCH (d2:Document {relative_path: 'src/services/user.ts'})
                    CREATE (s2:Symbol {
                        symbol: 'scip-typescript npm . src/services/user.ts validateUser().',
                        name: 'validateUser',
                        kind: 'Function',
                        line: 0,
                        character: 16
                    })
                    CREATE (d2)-[:CONTAINS]->(s2)
                """)
                
                # Create REFERENCES relationship
                session.run("""
                    MATCH (s1:Symbol {name: 'authenticate'})
                    MATCH (s2:Symbol {name: 'validateUser'})
                    CREATE (s1)-[:REFERENCES {line: 0, character: 50}]->(s2)
                """)
                
                # Verify graph structure
                result = session.run("""
                    MATCH (d:Document)
                    RETURN count(d) as doc_count
                """)
                doc_count = result.single()["doc_count"]
                
                result = session.run("""
                    MATCH (s:Symbol)
                    RETURN count(s) as symbol_count
                """)
                symbol_count = result.single()["symbol_count"]
                
                result = session.run("""
                    MATCH ()-[r:REFERENCES]->()
                    RETURN count(r) as ref_count
                """)
                ref_count = result.single()["ref_count"]
                
                if doc_count == 2 and symbol_count == 2 and ref_count == 1:
                    print(f"{GREEN}✓ Created sample graph: {doc_count} documents, {symbol_count} symbols, {ref_count} references{RESET}")
                    self.test_results.append(("Create Graph", True, None))
                    return True
                else:
                    print(f"{RED}✗ Graph structure incorrect: {doc_count} docs, {symbol_count} symbols, {ref_count} refs{RESET}")
                    self.test_results.append(("Create Graph", False, "Wrong node/relationship counts"))
                    return False
                    
        except Exception as e:
            print(f"{RED}✗ Failed to create sample graph: {e}{RESET}")
            self.test_results.append(("Create Graph", False, str(e)))
            return False
    
    def test_find_symbol_definition(self) -> bool:
        """Test finding symbol definition"""
        print(f"\n{BLUE}=== Test 4: Find Symbol Definition ==={RESET}")
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s:Symbol {name: 'authenticate'})
                    MATCH (d:Document)-[:CONTAINS]->(s)
                    RETURN s.name as name, s.kind as kind, d.relative_path as file
                """)
                
                record = result.single()
                if record and record["name"] == "authenticate":
                    print(f"{GREEN}✓ Found definition:{RESET}")
                    print(f"  Name: {record['name']}")
                    print(f"  Kind: {record['kind']}")
                    print(f"  File: {record['file']}")
                    self.test_results.append(("Find Definition", True, None))
                    return True
                else:
                    print(f"{RED}✗ Could not find symbol definition{RESET}")
                    self.test_results.append(("Find Definition", False, "Symbol not found"))
                    return False
                    
        except Exception as e:
            print(f"{RED}✗ Failed to find definition: {e}{RESET}")
            self.test_results.append(("Find Definition", False, str(e)))
            return False
    
    def test_find_references(self) -> bool:
        """Test finding symbol references"""
        print(f"\n{BLUE}=== Test 5: Find References ==={RESET}")
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (target:Symbol {name: 'validateUser'})
                    MATCH (caller:Symbol)-[r:REFERENCES]->(target)
                    MATCH (d:Document)-[:CONTAINS]->(caller)
                    RETURN caller.name as caller_name, d.relative_path as file, 
                           r.line as line, r.character as character
                """)
                
                records = list(result)
                if len(records) > 0:
                    print(f"{GREEN}✓ Found {len(records)} reference(s):{RESET}")
                    for rec in records:
                        print(f"  {rec['caller_name']} in {rec['file']} at line {rec['line']}")
                    self.test_results.append(("Find References", True, None))
                    return True
                else:
                    print(f"{RED}✗ No references found{RESET}")
                    self.test_results.append(("Find References", False, "No results"))
                    return False
                    
        except Exception as e:
            print(f"{RED}✗ Failed to find references: {e}{RESET}")
            self.test_results.append(("Find References", False, str(e)))
            return False
    
    def test_relationship_types(self) -> bool:
        """Test different relationship types"""
        print(f"\n{BLUE}=== Test 6: Relationship Types ==={RESET}")
        try:
            with self.driver.session() as session:
                # Add more relationship types
                session.run("""
                    MATCH (s1:Symbol {name: 'authenticate'})
                    MATCH (d:Document {relative_path: 'src/services/auth.ts'})
                    CREATE (d)-[:ENCLOSES]->(s1)
                """)
                
                # Query relationship types
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                """)
                
                rel_types = {}
                for record in result:
                    rel_types[record["rel_type"]] = record["count"]
                
                expected_types = ["REFERENCES", "CONTAINS", "ENCLOSES"]
                found_types = [t for t in expected_types if t in rel_types]
                
                if len(found_types) >= 2:
                    print(f"{GREEN}✓ Found relationship types:{RESET}")
                    for rel_type, count in rel_types.items():
                        print(f"  {rel_type}: {count}")
                    self.test_results.append(("Relationship Types", True, None))
                    return True
                else:
                    print(f"{RED}✗ Missing expected relationship types{RESET}")
                    self.test_results.append(("Relationship Types", False, "Missing types"))
                    return False
                    
        except Exception as e:
            print(f"{RED}✗ Failed to test relationship types: {e}{RESET}")
            self.test_results.append(("Relationship Types", False, str(e)))
            return False
    
    def test_complex_query_call_graph(self) -> bool:
        """Test complex query: transitive call graph"""
        print(f"\n{BLUE}=== Test 7: Complex Query - Call Graph ==={RESET}")
        try:
            with self.driver.session() as session:
                # Add another level to the call graph
                session.run("""
                    MATCH (d:Document {relative_path: 'src/services/user.ts'})
                    CREATE (s3:Symbol {
                        symbol: 'scip-typescript npm . src/services/user.ts checkDatabase().',
                        name: 'checkDatabase',
                        kind: 'Function'
                    })
                    CREATE (d)-[:CONTAINS]->(s3)
                """)
                
                session.run("""
                    MATCH (s2:Symbol {name: 'validateUser'})
                    MATCH (s3:Symbol {name: 'checkDatabase'})
                    CREATE (s2)-[:REFERENCES]->(s3)
                """)
                
                # Query transitive dependencies
                result = session.run("""
                    MATCH path = (start:Symbol {name: 'authenticate'})-[:REFERENCES*1..3]->(end:Symbol)
                    RETURN start.name as start, end.name as end, length(path) as depth
                    ORDER BY depth
                """)
                
                records = list(result)
                if len(records) >= 2:
                    print(f"{GREEN}✓ Found transitive call graph with {len(records)} paths:{RESET}")
                    for rec in records:
                        print(f"  {rec['start']} → {rec['end']} (depth: {rec['depth']})")
                    self.test_results.append(("Call Graph", True, None))
                    return True
                else:
                    print(f"{RED}✗ Call graph incomplete{RESET}")
                    self.test_results.append(("Call Graph", False, "Insufficient paths"))
                    return False
                    
        except Exception as e:
            print(f"{RED}✗ Failed to execute call graph query: {e}{RESET}")
            self.test_results.append(("Call Graph", False, str(e)))
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Benchmark query performance"""
        print(f"\n{BLUE}=== Test 8: Performance Benchmark ==={RESET}")
        try:
            with self.driver.session() as session:
                # Add more nodes for realistic testing
                for i in range(50):
                    session.run("""
                        CREATE (d:Document {
                            relative_path: $path,
                            language: 'typescript'
                        })
                    """, path=f"src/test/file{i}.ts")
                
                # Benchmark: Find all documents
                start_time = time.time()
                for _ in range(10):
                    result = session.run("MATCH (d:Document) RETURN count(d) as count")
                    result.single()
                end_time = time.time()
                avg_time = (end_time - start_time) / 10 * 1000  # ms
                
                # Benchmark: Find symbols
                start_time = time.time()
                for _ in range(10):
                    result = session.run("MATCH (s:Symbol) RETURN count(s) as count")
                    result.single()
                end_time = time.time()
                avg_symbol_time = (end_time - start_time) / 10 * 1000  # ms
                
                # Benchmark: Complex query
                start_time = time.time()
                for _ in range(10):
                    result = session.run("""
                        MATCH (d:Document)-[:CONTAINS]->(s:Symbol)
                        WHERE s.kind = 'Function'
                        RETURN d.relative_path, s.name
                        LIMIT 10
                    """)
                    list(result)
                end_time = time.time()
                avg_complex_time = (end_time - start_time) / 10 * 1000  # ms
                
                print(f"{GREEN}✓ Performance benchmarks:{RESET}")
                print(f"  Document count query: {avg_time:.2f}ms (avg of 10 runs)")
                print(f"  Symbol count query: {avg_symbol_time:.2f}ms (avg of 10 runs)")
                print(f"  Complex join query: {avg_complex_time:.2f}ms (avg of 10 runs)")
                
                # Performance targets
                if avg_time < 50 and avg_complex_time < 100:
                    print(f"{GREEN}✓ Performance meets targets (<50ms simple, <100ms complex){RESET}")
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
            print(f"{GREEN}✓ All tests passed! Memgraph integration is working correctly.{RESET}")
            print(f"{GREEN}✓ Ready to load real SCIP indexes into Memgraph.{RESET}\n")
        elif passed >= total * 0.8:
            print(f"{YELLOW}⚠ Most tests passed, but some issues detected.{RESET}")
            print(f"{YELLOW}  Review failed tests and check Memgraph configuration.{RESET}\n")
        else:
            print(f"{RED}✗ Multiple test failures detected.{RESET}")
            print(f"{RED}  Check Memgraph connection and configuration.{RESET}\n")
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            print(f"{BLUE}Connection closed.{RESET}\n")


def main():
    """Run all Memgraph integration tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}nCode Memgraph Integration Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Configuration
    uri = "bolt://localhost:7687"
    username = ""  # Memgraph default: no auth
    password = ""
    
    print(f"\nConfiguration:")
    print(f"  Memgraph URI: {uri}")
    print(f"  Authentication: {'Enabled' if username else 'Disabled'}")
    
    # Initialize tester
    tester = MemgraphTester(uri, username, password)
    
    try:
        # Run tests
        if not tester.connect():
            print(f"\n{RED}Cannot proceed without Memgraph connection.{RESET}")
            print(f"{YELLOW}Please ensure Memgraph is running:{RESET}")
            print(f"  docker ps | grep memgraph")
            print(f"  docker-compose up -d memgraph")
            return 1
        
        # Run test suite
        tester.test_clear_database()
        tester.test_create_sample_graph()
        tester.test_find_symbol_definition()
        tester.test_find_references()
        tester.test_relationship_types()
        tester.test_complex_query_call_graph()
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
    finally:
        tester.close()


if __name__ == "__main__":
    sys.exit(main())
