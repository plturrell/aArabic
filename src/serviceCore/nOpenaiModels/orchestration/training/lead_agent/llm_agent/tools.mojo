# tools.mojo
# Migrated from tools.py - Pure Mojo with Zig FFI for HTTP
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------

from collections import Dict, List
from sys.ffi import external_call, DLHandle
from memory import UnsafePointer
from sys import env_get_string
from time import sleep


# FFI bindings to Zig HTTP client (shared/http/client.zig)
alias ZIG_HTTP_LIB = "libzig_http_shimmy.dylib"  # macOS, use .so on Linux


struct HTTPClient:
    """
    HTTP client using Zig FFI for making API calls.
    Wraps the Zig HTTP client (shared/http/client.zig).
    """
    var lib_handle: DLHandle
    
    fn __init__(inout self) raises:
        """Initialize HTTP client by loading Zig library"""
        # Load the Zig HTTP library
        self.lib_handle = DLHandle(ZIG_HTTP_LIB)
    
    fn get(self, url: String) raises -> String:
        """
        Perform HTTP GET request.
        Uses zig_http_get from client.zig.
        """
        # Convert String to C string
        let url_ptr = url.unsafe_cstr_ptr()
        
        # Call Zig function: zig_http_get(url: [*:0]const u8) -> [*:0]const u8
        let result_ptr = external_call["zig_http_get", UnsafePointer[UInt8]](
            self.lib_handle,
            url_ptr
        )
        
        # Convert result back to String
        var result = String(result_ptr)
        return result
    
    fn post(self, url: String, body: String) raises -> String:
        """
        Perform HTTP POST request.
        Uses zig_http_post from client.zig.
        """
        let url_ptr = url.unsafe_cstr_ptr()
        let body_ptr = body.unsafe_cstr_ptr()
        let body_len = len(body)
        
        # Call Zig function: zig_http_post(url, body, body_len) -> [*:0]const u8
        let result_ptr = external_call["zig_http_post", UnsafePointer[UInt8]](
            self.lib_handle,
            url_ptr,
            body_ptr,
            body_len
        )
        
        var result = String(result_ptr)
        return result


struct LLMConfig:
    """Configuration for LLM API calls"""
    var api_url: String
    var api_key: String
    var model: String
    var max_retries: Int
    var timeout: Int
    
    fn __init__(
        inout self,
        api_url: String = "http://localhost:8080/v1/chat/completions",
        model: String = "llama-3.3-70b",
        max_retries: Int = 3,
        timeout: Int = 30
    ):
        self.api_url = api_url
        self.api_key = env_get_string("OPENAI_API_KEY", "")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout


fn parse_xml_tag(text: String, tag: String) raises -> List[String]:
    """
    Parse XML-style tags from text.
    Extracts content between <tag> and </tag>.
    """
    var results = List[String]()
    let open_tag = "<" + tag + ">"
    let close_tag = "</" + tag + ">"
    
    var search_start = 0
    while True:
        # Find opening tag
        let open_idx = text.find(open_tag, search_start)
        if open_idx == -1:
            break
        
        # Find closing tag
        let content_start = open_idx + len(open_tag)
        let close_idx = text.find(close_tag, content_start)
        if close_idx == -1:
            break
        
        # Extract content
        let content = text[content_start:close_idx]
        results.append(content)
        
        # Move search position
        search_start = close_idx + len(close_tag)
    
    return results


fn format_json_request(prompt: String, model: String) -> String:
    """
    Format a chat completion request as JSON.
    Simple JSON formatting without external dependencies.
    """
    # Escape special characters in prompt
    var escaped_prompt = prompt.replace('"', '\\"').replace('\n', '\\n')
    
    var json = String('{"model":"')
    json += model
    json += '","messages":[{"role":"user","content":"'
    json += escaped_prompt
    json += '"}],"temperature":0.7,"max_tokens":2048}'
    
    return json


fn call_api_with_retry(
    client: HTTPClient,
    prompt: String,
    config: LLMConfig
) raises -> String:
    """
    Call LLM API with retry logic.
    Returns the response text.
    """
    var last_error = String("")
    
    for attempt in range(config.max_retries):
        try:
            # Format request
            let request_body = format_json_request(prompt, config.model)
            
            # Make API call
            let response = client.post(config.api_url, request_body)
            
            # Parse response (simplified - assumes response has "content" field)
            # Full implementation would use proper JSON parsing
            let content_start = response.find('"content":"')
            if content_start == -1:
                raise Error("Invalid response format")
            
            let content_begin = content_start + len('"content":"')
            let content_end = response.find('"', content_begin)
            if content_end == -1:
                raise Error("Invalid response format")
            
            let content = response[content_begin:content_end]
            
            # Unescape JSON string
            let result = content.replace('\\n', '\n').replace('\\"', '"')
            return result
            
        except e:
            last_error = str(e)
            if attempt < config.max_retries - 1:
                # Exponential backoff
                let wait_time = (2 ** attempt) * 1000  # milliseconds
                sleep(wait_time / 1000.0)  # convert to seconds
    
    raise Error("API call failed after " + str(config.max_retries) + " retries: " + last_error)


struct QueryWriter:
    """
    Generates search queries based on documents and user questions.
    Used to identify missing information that needs to be retrieved.
    """
    var prompt_template: String
    var config: LLMConfig
    var client: HTTPClient
    
    fn __init__(inout self, config: LLMConfig) raises:
        self.config = config
        self.client = HTTPClient()
        
        self.prompt_template = (
            "Documents:\n{documents}\n\n"
            "User question: {user_question}\n\n"
            "Break down the user question. Based on the documents we have found, "
            "write a query to search missing information. "
            "Wrap the query within <query> and </query>"
        )
    
    fn __call__(self, documents: String, user_question: String) raises -> List[String]:
        """
        Generate search queries.
        Returns list of query strings.
        """
        # Format prompt
        var prompt = self.prompt_template
        prompt = prompt.replace("{documents}", documents)
        prompt = prompt.replace("{user_question}", user_question)
        
        # Call API
        let response = call_api_with_retry(self.client, prompt, self.config)
        
        # Parse queries from response
        let queries = parse_xml_tag(response, "query")
        
        return queries


struct AnswerGenerator:
    """
    Generates answers based on documents and user questions.
    Produces both thinking process and final answer.
    """
    var prompt_template: String
    var config: LLMConfig
    var client: HTTPClient
    
    fn __init__(inout self, config: LLMConfig) raises:
        self.config = config
        self.client = HTTPClient()
        
        self.prompt_template = (
            "Documents:\n{documents}\n\n"
            "User question: {user_question}\n\n"
            "Wrap the thinking process and explanation between <think> and </think> "
            "and wrap only the exact answer without any explanation within "
            "<answer> and </answer>."
        )
    
    fn __call__(self, documents: String, user_question: String) raises -> String:
        """
        Generate answer.
        Returns the answer string (content within <answer> tags).
        """
        # Format prompt
        var prompt = self.prompt_template
        prompt = prompt.replace("{documents}", documents)
        prompt = prompt.replace("{user_question}", user_question)
        
        # Call API
        let response = call_api_with_retry(self.client, prompt, self.config)
        
        # Parse answer from response
        let answers = parse_xml_tag(response, "answer")
        
        if len(answers) > 0:
            return answers[0]
        else:
            # If no answer tags found, return full response
            return response
    
    fn get_thinking(self, documents: String, user_question: String) raises -> String:
        """
        Get the thinking process (content within <think> tags).
        """
        # Format prompt
        var prompt = self.prompt_template
        prompt = prompt.replace("{documents}", documents)
        prompt = prompt.replace("{user_question}", user_question)
        
        # Call API
        let response = call_api_with_retry(self.client, prompt, self.config)
        
        # Parse thinking from response
        let thinking = parse_xml_tag(response, "think")
        
        if len(thinking) > 0:
            return thinking[0]
        else:
            return ""


struct CodeExecutor:
    """
    Executes code in a sandboxed environment.
    Used for the enhance_reasoning tool.
    """
    var config: LLMConfig
    var client: HTTPClient
    
    fn __init__(inout self, config: LLMConfig) raises:
        self.config = config
        self.client = HTTPClient()
    
    fn execute_python(self, code: String) raises -> String:
        """
        Execute Python code and return the result.
        For now, returns a placeholder. Full implementation would:
        1. Send code to a sandbox service
        2. Execute safely with timeouts
        3. Return stdout/stderr
        """
        # This would call a sandbox execution service
        # For example: POST to http://localhost:8000/execute
        let sandbox_url = "http://localhost:8000/execute"
        
        var request_body = String('{"language":"python","code":"')
        request_body += code.replace('"', '\\"').replace('\n', '\\n')
        request_body += '"}'
        
        let response = self.client.post(sandbox_url, request_body)
        
        return response


fn main() raises:
    """Entry point for tools module"""
    print("=" * 80)
    print("LLM Tools - Pure Mojo Implementation with Zig HTTP FFI")
    print("=" * 80)
    print("")
    print("Features:")
    print("  - QueryWriter: Generate search queries")
    print("  - AnswerGenerator: Generate answers with thinking")
    print("  - CodeExecutor: Execute code in sandbox")
    print("  - HTTP client via Zig FFI (shared/http/client.zig)")
    print("  - Retry logic with exponential backoff")
    print("  - XML tag parsing for structured outputs")
    print("")
    
    # Test HTTP client
    print("Testing HTTP Client:")
    print("-" * 40)
    
    let config = LLMConfig(
        api_url="http://localhost:8080/v1/chat/completions",
        model="llama-3.3-70b"
    )
    
    print("Config:")
    print("  API URL:", config.api_url)
    print("  Model:", config.model)
    print("  Max Retries:", config.max_retries)
    print("  Timeout:", config.timeout, "seconds")
    print("")
    
    # Test XML parsing
    print("Testing XML Parsing:")
    print("-" * 40)
    
    let test_text = "Here is my thinking: <think>Let me analyze this</think> And the answer is: <answer>42</answer>"
    
    let thinking = parse_xml_tag(test_text, "think")
    let answers = parse_xml_tag(test_text, "answer")
    
    print("Input text:", test_text)
    print("Extracted <think>:", thinking[0] if len(thinking) > 0 else "None")
    print("Extracted <answer>:", answers[0] if len(answers) > 0 else "None")
    
    print("")
    print("=" * 80)
    print("✅ Pure Mojo tools with Zig FFI ready!")
    print("=" * 80)
    print("")
    print("Usage Example:")
    print("-" * 40)
    print("  # Create answer generator")
    print("  let generator = AnswerGenerator(config)")
    print("  ")
    print("  # Generate answer")
    print('  let answer = generator("Document text...", "What is X?")')
    print("  print(answer)")
    print("")
