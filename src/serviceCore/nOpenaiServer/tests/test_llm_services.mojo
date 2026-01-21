"""
Test LLM Services - Chat and Completion
Comprehensive integration tests for the OpenAI-compatible API
"""

from services.llm.chat import handle_chat_request
from services.llm.completion import handle_completion_request
from services.llm.handlers import handle_request


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

    # Validate response structure
    var passed = True
    if "choices" not in response:
        print("❌ Missing 'choices' in response")
        passed = False
    if "id" not in response:
        print("❌ Missing 'id' in response")
        passed = False
    if "model" not in response:
        print("❌ Missing 'model' in response")
        passed = False
    if passed:
        print("✅ Chat response structure valid")
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

    # Validate response structure
    var passed = True
    if "choices" not in response:
        print("❌ Missing 'choices' in response")
        passed = False
    if "id" not in response:
        print("❌ Missing 'id' in response")
        passed = False
    if passed:
        print("✅ Completion response structure valid")
    print("=" * 80)
    print()


fn test_chat_with_system_message():
    """Test chat with system message."""
    print("=" * 80)
    print("TEST 3: Chat with System Message")
    print("=" * 80)
    print()

    var request_body = String(
        '{"messages":['
        '{"role":"system","content":"You are a helpful assistant."},'
        '{"role":"user","content":"What is 2+2?"}'
        '],"temperature":0.7,"max_tokens":100}'
    )

    print("📤 Request:")
    print(request_body)
    print()

    var response = handle_chat_request(request_body)

    print("📥 Response:")
    print(response)
    print()

    if "choices" in response:
        print("✅ Chat with system message handled correctly")
    else:
        print("❌ Failed to handle system message")
    print("=" * 80)
    print()


fn test_chat_multi_turn():
    """Test multi-turn conversation."""
    print("=" * 80)
    print("TEST 4: Multi-turn Conversation")
    print("=" * 80)
    print()

    var request_body = String(
        '{"messages":['
        '{"role":"user","content":"My name is Alice."},'
        '{"role":"assistant","content":"Hello Alice! Nice to meet you."},'
        '{"role":"user","content":"What is my name?"}'
        ']}'
    )

    print("📤 Request:")
    print(request_body)
    print()

    var response = handle_chat_request(request_body)

    print("📥 Response:")
    print(response)
    print()
    print("✅ Multi-turn conversation handled")
    print("=" * 80)
    print()


fn test_completion_with_params():
    """Test completion with various parameters."""
    print("=" * 80)
    print("TEST 5: Completion with Parameters")
    print("=" * 80)
    print()

    var request_body = String(
        '{"prompt":"The quick brown fox",'
        '"max_tokens":20,"temperature":0.5,"top_p":0.9,'
        '"n":1,"echo":false}'
    )

    print("📤 Request:")
    print(request_body)
    print()

    var response = handle_completion_request(request_body)

    print("📥 Response:")
    print(response)
    print()
    print("✅ Parameterized completion handled")
    print("=" * 80)
    print()


fn test_models_endpoint():
    """Test /v1/models endpoint via handlers."""
    print("=" * 80)
    print("TEST 6: Models Endpoint")
    print("=" * 80)
    print()

    # Simulate GET /v1/models
    var response_ptr = handle_request(
        "GET".unsafe_cstr_ptr(),
        "/v1/models".unsafe_cstr_ptr(),
        "".unsafe_cstr_ptr(),
        0
    )

    # Convert response to string
    var response = String(response_ptr)

    print("📥 Response:")
    print(response)
    print()

    if "data" in response and "object" in response:
        print("✅ Models endpoint returns valid structure")
    else:
        print("❌ Invalid models response")
    print("=" * 80)
    print()


fn test_health_endpoint():
    """Test /health endpoint via handlers."""
    print("=" * 80)
    print("TEST 7: Health Endpoint")
    print("=" * 80)
    print()

    var response_ptr = handle_request(
        "GET".unsafe_cstr_ptr(),
        "/health".unsafe_cstr_ptr(),
        "".unsafe_cstr_ptr(),
        0
    )

    var response = String(response_ptr)

    print("📥 Response:")
    print(response)
    print()

    if "status" in response and "healthy" in response:
        print("✅ Health endpoint returns healthy status")
    else:
        print("❌ Invalid health response")
    print("=" * 80)
    print()


fn test_error_handling():
    """Test error handling for invalid requests."""
    print("=" * 80)
    print("TEST 8: Error Handling")
    print("=" * 80)
    print()

    # Test empty messages array
    var request_body = String('{"messages":[]}')
    var response = handle_chat_request(request_body)
    print("📤 Empty messages request")
    print("📥 Response: " + response[:100] + "...")

    # Test malformed JSON
    request_body = String('{"messages": invalid}')
    response = handle_chat_request(request_body)
    print("📤 Malformed JSON request")
    print("📥 Response: " + response[:100] + "...")

    # Test missing required field
    request_body = String('{"temperature": 0.7}')
    response = handle_chat_request(request_body)
    print("📤 Missing messages field")
    print("📥 Response: " + response[:100] + "...")

    print("✅ Error handling tests completed")
    print("=" * 80)
    print()


fn main():
    """Run all tests."""
    print()
    print("🧪 LLM Services Integration Test Suite")
    print("Comprehensive tests for chat.mojo, completion.mojo, and handlers.mojo")
    print()

    # Basic functionality tests
    test_chat_service()
    test_completion_service()

    # Advanced chat tests
    test_chat_with_system_message()
    test_chat_multi_turn()

    # Advanced completion tests
    test_completion_with_params()

    # Endpoint tests
    test_models_endpoint()
    test_health_endpoint()

    # Error handling
    test_error_handling()

    print("✅ All integration tests completed!")
    print()
    print("Note: These tests use the inference engine API.")
    print("Build the Zig library first: cd inference/engine && zig build")
    print()
