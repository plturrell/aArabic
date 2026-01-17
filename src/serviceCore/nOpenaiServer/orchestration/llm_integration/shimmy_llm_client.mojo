"""
Shimmy LLM Client
=================

OpenAI-compatible client for shimmy_openai_server.
Wraps the existing HTTPClient to communicate with your local LLM.

Architecture:
    ShimmyLLMClient
        ↓ (uses)
    HTTPClient (existing in graph-toolkit)
        ↓ (uses)
    zig_shimmy_post() (existing in zig_http_shimmy)
        ↓ (calls)
    shimmy_openai_server (your local LLM on port 11434)

Author: Shimmy-Mojo Team
Date: 2026-01-16
"""

from collections import Dict, List
from ...graph_toolkit.lib.protocols.http import HTTPClient


struct ChatMessage:
    """Represents a single chat message in the conversation."""
    
    var role: String
    var content: String
    
    fn __init__(inout self, role: String, content: String):
        self.role = role
        self.content = content
    
    fn to_json(self) -> String:
        """Convert to JSON string."""
        return '{"role":"' + self.role + '","content":"' + self.escape_json(self.content) + '"}'
    
    fn escape_json(self, text: String) -> String:
        """Escape special characters for JSON."""
        # Simple escape for quotes and backslashes
        var result = String("")
        for i in range(len(text)):
            let c = text[i]
            if c == '"':
                result += '\\"'
            elif c == '\\':
                result += '\\\\'
            elif c == '\n':
                result += '\\n'
            elif c == '\r':
                result += '\\r'
            elif c == '\t':
                result += '\\t'
            else:
                result += c
        return result


struct ShimmyLLMClient:
    """
    Client for shimmy_openai_server (local LLM).
    
    Provides OpenAI-compatible chat completions API.
    Uses the existing HTTPClient infrastructure to communicate
    with your shimmy_openai_server running on localhost:11434.
    
    Example:
        var client = ShimmyLLMClient()
        var messages = List[ChatMessage]()
        messages.append(ChatMessage("user", "What is 2+2?"))
        var response = client.chat_completion(messages)
        print(response)  # "4"
    """
    
    var http: HTTPClient
    var model: String
    var base_url: String
    var default_temperature: Float32
    var default_max_tokens: Int
    
    fn __init__(inout self, 
                base_url: String = "http://localhost:11434",
                model: String = "phi-3-mini"):
        """
        Initialize the Shimmy LLM client.
        
        Args:
            base_url: URL of shimmy_openai_server (default: http://localhost:11434)
            model: Model to use (phi-3-mini, llama-3.2-1b, llama-3.2-3b)
        """
        self.base_url = base_url
        self.model = model
        self.http = HTTPClient(base_url)
        self.default_temperature = 0.7
        self.default_max_tokens = 512
    
    fn chat_completion(inout self, 
                      messages: List[ChatMessage],
                      temperature: Float32 = -1.0,
                      max_tokens: Int = -1) raises -> String:
        """
        Call /v1/chat/completions endpoint.
        
        This method:
        1. Builds an OpenAI-compatible JSON request
        2. POSTs to shimmy_openai_server via HTTPClient
        3. Parses the JSON response
        4. Returns the assistant's response text
        
        Args:
            messages: List of chat messages (conversation history)
            temperature: Sampling temperature (0-2). Use -1 for default (0.7)
            max_tokens: Maximum tokens to generate. Use -1 for default (512)
            
        Returns:
            The assistant's response text
            
        Raises:
            Error if the request fails or response is invalid
        """
        # Use defaults if not specified
        let temp = temperature if temperature >= 0 else self.default_temperature
        let max_tok = max_tokens if max_tokens > 0 else self.default_max_tokens
        
        # Build request JSON
        let request_json = self.build_chat_request(messages, temp, max_tok)
        
        # POST to shimmy_openai_server
        # This uses your existing HTTPClient -> zig_shimmy_post -> shimmy_openai_server
        let response_json = self.http.post_json("/v1/chat/completions", request_json)
        
        # Extract the assistant's response from JSON
        return self.extract_chat_response(response_json)
    
    fn build_chat_request(self, 
                         messages: List[ChatMessage], 
                         temperature: Float32,
                         max_tokens: Int) -> String:
        """
        Build OpenAI-compatible chat completion request.
        
        Format:
        {
            "model": "phi-3-mini",
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
        """
        var json = String('{"model":"')
        json += self.model
        json += '","messages":['
        
        # Add messages
        for i in range(len(messages)):
            if i > 0:
                json += ','
            json += messages[i].to_json()
        
        json += '],"temperature":'
        json += str(temperature)
        json += ',"max_tokens":'
        json += str(max_tokens)
        json += '}'
        
        return json
    
    fn extract_chat_response(self, response_json: String) raises -> String:
        """
        Extract assistant's message from OpenAI-compatible response.
        
        Response format:
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is..."
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
        Args:
            response_json: JSON response from shimmy_openai_server
            
        Returns:
            The content field from the assistant's message
        """
        # Simple JSON parsing to extract content
        # Look for "content":"..." pattern
        let content_marker = '"content":"'
        let content_start_idx = response_json.find(content_marker)
        
        if content_start_idx < 0:
            raise Error("Invalid response: no content field found")
        
        # Start after the marker
        let content_start = content_start_idx + len(content_marker)
        
        # Find the closing quote (handle escaped quotes)
        var content_end = content_start
        var escaped = False
        
        for i in range(content_start, len(response_json)):
            if escaped:
                escaped = False
                continue
            
            if response_json[i] == '\\':
                escaped = True
                continue
            
            if response_json[i] == '"':
                content_end = i
                break
        
        if content_end <= content_start:
            raise Error("Invalid response: malformed content field")
        
        # Extract and unescape content
        let raw_content = response_json[content_start:content_end]
        return self.unescape_json(raw_content)
    
    fn unescape_json(self, text: String) -> String:
        """Unescape JSON string."""
        var result = String("")
        var i = 0
        
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text):
                let next_char = text[i + 1]
                if next_char == 'n':
                    result += '\n'
                    i += 2
                elif next_char == 'r':
                    result += '\r'
                    i += 2
                elif next_char == 't':
                    result += '\t'
                    i += 2
                elif next_char == '"':
                    result += '"'
                    i += 2
                elif next_char == '\\':
                    result += '\\'
                    i += 2
                else:
                    result += text[i]
                    i += 1
            else:
                result += text[i]
                i += 1
        
        return result
    
    fn health_check(inout self) raises -> Bool:
        """
        Check if shimmy_openai_server is running and ready.
        
        Returns:
            True if server is healthy and model is loaded
        """
        try:
            let response = self.http.get("/health")
            return response.find('"status":"ready"') >= 0
        except:
            return False
    
    fn list_models(inout self) raises -> String:
        """
        Get available models from shimmy_openai_server.
        
        Returns:
            JSON list of available models
        """
        return self.http.get("/v1/models")
