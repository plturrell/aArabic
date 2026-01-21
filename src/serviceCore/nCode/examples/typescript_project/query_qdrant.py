#!/usr/bin/env python3
"""
Query Qdrant for semantic code search examples
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def main():
    """Run example semantic search queries"""
    
    print("🔍 Running Semantic Search Queries")
    print("=" * 50)
    print()
    
    # Connect to Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333)
        collection_name = "typescript_example"
        
        # Check if collection exists
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            print(f"❌ Collection '{collection_name}' not found")
            print("   Run the export step first: ./run_example.sh")
            return
        
        print(f"✅ Connected to Qdrant collection: {collection_name}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("   Make sure Qdrant is running on localhost:6333")
        return
    
    # Query 1: Find classes with constructors
    print("Query 1: Find classes with constructors")
    print("-" * 50)
    try:
        results = client.search(
            collection_name=collection_name,
            query_text="class with constructor and properties",
            limit=5
        )
        
        if results:
            for i, result in enumerate(results, 1):
                payload = result.payload
                print(f"  {i}. {payload.get('symbol', 'N/A')} (score: {result.score:.3f})")
                print(f"     File: {payload.get('file', 'N/A')}")
                print(f"     Kind: {payload.get('kind', 'N/A')}")
                if i < len(results):
                    print()
        else:
            print("  No results found")
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
    
    print()
    
    # Query 2: Find authentication functions
    print("Query 2: Find authentication and login functions")
    print("-" * 50)
    try:
        results = client.search(
            collection_name=collection_name,
            query_text="functions that authenticate users and verify credentials",
            limit=5
        )
        
        if results:
            for i, result in enumerate(results, 1):
                payload = result.payload
                print(f"  {i}. {payload.get('symbol', 'N/A')} (score: {result.score:.3f})")
                print(f"     File: {payload.get('file', 'N/A')}")
                print(f"     Doc: {payload.get('documentation', 'N/A')[:60]}...")
                if i < len(results):
                    print()
        else:
            print("  No results found")
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
    
    print()
    
    # Query 3: Find database operations
    print("Query 3: Find database connection and query methods")
    print("-" * 50)
    try:
        results = client.search(
            collection_name=collection_name,
            query_text="methods for database connection and executing queries",
            limit=5
        )
        
        if results:
            for i, result in enumerate(results, 1):
                payload = result.payload
                print(f"  {i}. {payload.get('symbol', 'N/A')} (score: {result.score:.3f})")
                print(f"     File: {payload.get('file', 'N/A')}")
                print(f"     Kind: {payload.get('kind', 'N/A')}")
                if i < len(results):
                    print()
        else:
            print("  No results found")
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
    
    print()
    
    # Query 4: Find validation utilities
    print("Query 4: Find validation and helper functions")
    print("-" * 50)
    try:
        results = client.search(
            collection_name=collection_name,
            query_text="functions that validate email and password input",
            limit=5
        )
        
        if results:
            for i, result in enumerate(results, 1):
                payload = result.payload
                print(f"  {i}. {payload.get('symbol', 'N/A')} (score: {result.score:.3f})")
                print(f"     File: {payload.get('file', 'N/A')}")
                if i < len(results):
                    print()
        else:
            print("  No results found")
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
    
    print()
    
    # Get collection stats
    print("📊 Collection Statistics")
    print("-" * 50)
    try:
        collection_info = client.get_collection(collection_name)
        print(f"  Total vectors: {collection_info.points_count}")
        print(f"  Vector dimension: {collection_info.config.params.vectors.size}")
        print(f"  Distance metric: {collection_info.config.params.vectors.distance}")
    except Exception as e:
        print(f"  ❌ Failed to get stats: {e}")
    
    print()
    print("✅ Semantic search queries completed!")
    print()
    print("💡 Tips:")
    print("   - Try your own natural language queries")
    print("   - Filter by file: add query_filter parameter")
    print("   - Adjust limit to see more/fewer results")
    print()


if __name__ == "__main__":
    main()
