"""
Test LLM Services - Chat and Completion
Tests the new chat.mojo and completion.mojo modules
"""

from services.llm.chat import handle_chat_request
from services.llm.completion import handle_completion_request


fn test_chat_service():
    """Test chat completion service."""
    print("=" * 80)
    print("TEST 1: Chat Completion Service")
    print("=" * 80)
    print()
    
    # Simple chat request (mock JSON)
    var request_body = String('{"messages":[{"role":"user","content":"Hello!"}]}')
    
    print("📤 Request:")
    print(request_body)
    print()
    
    # Call chat handler
    var response = handle_chat_request(request_body)
    
    print("📥 Response:")
    print(response)
    print()
    print("=" * 80)
    print()


fn test_completion_service():
    """Test text completion service."""
    print("=" * 80)
    print("TEST 2: Text Completion Service")
    print("=" * 80)
    print()
    
    # Simple completion request (mock JSON)
    var request_body = String('{"prompt":"Once upon a time","max_tokens":50}')
    
    print("📤 Request:")
    print(request_body)
    print()
    
    # Call completion handler
    var response = handle_completion_request(request_body)
    
    print("📥 Response:")
    print(response)
    print()
    print("=" * 80)
    print()


fn main():
    """Run all tests."""
    print()
    print("🧪 LLM Services Test Suite")
    print("Testing chat.mojo and completion.mojo modules")
    print()
    
    # Test 1: Chat service
    test_chat_service()
    
    # Test 2: Completion service  
    test_completion_service()
    
    print("✅ All tests completed!")
    print()
    print("Note: These tests use the inference engine API.")
    print("Build the Zig library first: cd inference/engine && zig build")
    print()
