"""
Example: Using MemgraphClient

This example demonstrates how to:
1. Connect to Memgraph
2. Execute Cypher queries
3. Use Memgraph-specific features (triggers, procedures, streams)
"""

from ..lib.clients.memgraph_client import MemgraphClient
from collections import Dict

fn main() raises:
    print("=" * 80)
    print("Memgraph Client Example")
    print("=" * 80)
    print()
    
    # Create client
    var client = MemgraphClient(
        host="localhost",
        port=7687,
        username="",  # Memgraph doesn't require auth by default
        password=""
    )
    
    print("📡 Connecting to Memgraph...")
    client.connect()
    print("✅ Connected!")
    print()
    
    # Example 1: Create some nodes
    print("Example 1: Creating nodes")
    print("-" * 40)
    var create_query = """
    CREATE (p1:Person {name: 'Alice', age: 30})
    CREATE (p2:Person {name: 'Bob', age: 25})
    CREATE (p3:Person {name: 'Charlie', age: 35})
    RETURN p1, p2, p3
    """
    var result = client.execute_query(create_query, Dict[String, String]())
    print("✅ Created 3 Person nodes")
    print()
    
    # Example 2: Create relationships
    print("Example 2: Creating relationships")
    print("-" * 40)
    var rel_query = """
    MATCH (p1:Person {name: 'Alice'}), (p2:Person {name: 'Bob'})
    CREATE (p1)-[:KNOWS {since: 2020}]->(p2)
    RETURN p1, p2
    """
    result = client.execute_query(rel_query, Dict[String, String]())
    print("✅ Created KNOWS relationship")
    print()
    
    # Example 3: Query with parameters
    print("Example 3: Parameterized query")
    print("-" * 40)
    var params = Dict[String, String]()
    params["min_age"] = "28"
    var param_query = "MATCH (p:Person) WHERE p.age >= $min_age RETURN p.name, p.age"
    result = client.execute_query(param_query, params)
    print("✅ Found people aged 28 or older")
    print()
    
    # Example 4: Get storage info (Memgraph-specific)
    print("Example 4: Memgraph storage info")
    print("-" * 40)
    var storage = client.get_storage_info()
    print("✅ Retrieved storage information")
    print()
    
    # Example 5: Get schema (Memgraph-specific)
    print("Example 5: Database schema")
    print("-" * 40)
    var index_info = client.get_index_info()
    print("✅ Retrieved index information")
    print()
    
    # Example 6: Create a trigger (Memgraph-specific)
    print("Example 6: Creating trigger")
    print("-" * 40)
    try:
        var trigger_result = client.create_trigger(
            "person_created",
            "CREATE",
            "UNWIND createdVertices AS v SET v.created_at = timestamp()"
        )
        print("✅ Created trigger: person_created")
    except:
        print("⚠️  Trigger might already exist")
    print()
    
    # Example 7: Show triggers
    print("Example 7: List all triggers")
    print("-" * 40)
    var triggers = client.get_triggers()
    print("✅ Listed all triggers")
    print()
    
    # Example 8: Call a procedure (Memgraph-specific)
    print("Example 8: Calling query module procedure")
    print("-" * 40)
    try:
        var proc_params = Dict[String, String]()
        var proc_result = client.call_procedure("mg.procedures", proc_params)
        print("✅ Called mg.procedures()")
    except:
        print("⚠️  Procedure not available or workspace doesn't support it")
    print()
    
    # Cleanup
    print("🧹 Cleaning up...")
    var cleanup_query = "MATCH (n:Person) DETACH DELETE n"
    client.execute_query(cleanup_query, Dict[String, String]())
    print("✅ Cleaned up test data")
    print()
    
    # Disconnect
    print("👋 Disconnecting...")
    client.disconnect()
    print("✅ Disconnected!")
    print()
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
