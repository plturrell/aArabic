"""
Test file for Qdrant client with Mojo 0.26.1
Quick verification that syntax changes are correct
"""

from qdrant_client import QdrantClient, QdrantResult

fn main() raises:
    print("🧪 Testing Qdrant Client with Mojo 0.26.1")
    print("=" * 60)
    
    # Test 1: Client instantiation
    print("\n1. Testing client instantiation...")
    var client = QdrantClient()
    print("   ✅ Client created successfully")
    
    # Test 2: Client with custom host/port
    print("\n2. Testing client with custom parameters...")
    var client2 = QdrantClient(host="localhost", port=6333)
    print("   ✅ Client with custom params created")
    
    print("\n" + "=" * 60)
    print("✅ All syntax tests passed!")
    print("   Qdrant client is compatible with Mojo 0.26.1")
