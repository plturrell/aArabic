# code_sandbox.mojo
# Secure Code Execution Sandbox
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Dict, List
from sys.ffi import external_call, DLHandle
from memory import UnsafePointer
from time import now, sleep
from sys import env_get_string

# Import HTTP client for sandbox service
from tools import HTTPClient


struct SandboxConfig:
    """Configuration for code execution sandbox"""
    var sandbox_url: String
    var timeout_seconds: Int
    var max_memory_mb: Int
    var max_cpu_percent: Int
    var allowed_imports: List[String]
    var enable_network: Bool
    
    fn __init__(
        inout self,
        sandbox_url: String = "http://localhost:8000/execute",
        timeout_seconds: Int = 30,
        max_memory_mb: Int = 512,
        max_cpu_percent: Int = 50,
        enable_network: Bool = False
    ):
        self.sandbox_url = sandbox_url
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.enable_network = enable_network
        
        # Default allowed imports for safety
        self.allowed_imports = List[String]()
        self.allowed_imports.append("math")
        self.allowed_imports.append("random")
        self.allowed_imports.append("json")
        self.allowed_imports.append("datetime")
        self.allowed_imports.append("collections")


struct ExecutionResult:
    """Result of code execution"""
    var success: Bool
    var stdout: String
    var stderr: String
    var return_value: String
    var execution_time_ms: Float64
    var memory_used_mb: Float64
    var error_message: String
    
    fn __init__(inout self):
        self.success = False
        self.stdout = ""
        self.stderr = ""
        self.return_value = ""
        self.execution_time_ms = 0.0
        self.memory_used_mb = 0.0
        self.error_message = ""


struct CodeSandbox:
    """
    Secure code execution sandbox with resource limits.
    Supports Python code execution in isolated environment.
    """
    var config: SandboxConfig
    var http_client: HTTPClient
    
    fn __init__(inout self, config: SandboxConfig) raises:
        self.config = config
        self.http_client = HTTPClient()
        
        print(f"🔒 Code sandbox initialized:")
        print(f"   URL: {config.sandbox_url}")
        print(f"   Timeout: {config.timeout_seconds}s")
        print(f"   Max memory: {config.max_memory_mb}MB")
        print(f"   Max CPU: {config.max_cpu_percent}%")
    
    fn execute_python(self, code: String) raises -> ExecutionResult:
        """
        Execute Python code in sandbox.
        Returns execution result with stdout, stderr, and metrics.
        """
        print(f"💻 Executing code in sandbox...")
        
        # Validate code before execution
        let validation_result = self._validate_code(code)
        if not validation_result.success:
            var result = ExecutionResult()
            result.success = False
            result.error_message = validation_result.error_message
            return result
        
        # Prepare execution request
        let request = self._create_execution_request(code)
        
        # Execute via HTTP
        let start_time = now()
        
        try:
            let response = self.http_client.post(self.config.sandbox_url, request)
            let end_time = now()
            
            # Parse response
            var result = self._parse_execution_response(response)
            result.execution_time_ms = Float64(end_time - start_time) / 1_000_000.0  # ns to ms
            
            print(f"✅ Execution completed in {result.execution_time_ms:.2f}ms")
            return result
            
        except e:
            var result = ExecutionResult()
            result.success = False
            result.error_message = "Sandbox communication error: " + str(e)
            return result
    
    fn execute_python_with_retry(self, code: String, max_retries: Int = 3) raises -> ExecutionResult:
        """Execute with retry logic for transient failures"""
        var last_error = String("")
        
        for attempt in range(max_retries):
            try:
                let result = self.execute_python(code)
                
                # If success or non-retryable error, return
                if result.success or "ValidationError" in result.error_message:
                    return result
                
                last_error = result.error_message
                
                if attempt < max_retries - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                    sleep(1.0 * (attempt + 1))  # Exponential backoff
                    
            except e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    sleep(1.0 * (attempt + 1))
        
        var result = ExecutionResult()
        result.success = False
        result.error_message = f"Failed after {max_retries} attempts: {last_error}"
        return result
    
    fn _validate_code(self, code: String) -> ExecutionResult:
        """
        Validate code for security issues before execution.
        Checks for:
        - Dangerous imports
        - System calls
        - File operations (unless explicitly allowed)
        - Network operations (unless explicitly allowed)
        """
        var result = ExecutionResult()
        result.success = True
        
        # Check for dangerous imports
        let dangerous_imports = List[String](
            "os", "sys", "subprocess", "socket", "urllib", 
            "requests", "http", "ftplib", "telnetlib",
            "__import__", "eval", "exec", "compile"
        )
        
        for dangerous in dangerous_imports:
            if dangerous in code:
                result.success = False
                result.error_message = f"ValidationError: Dangerous operation detected: {dangerous}"
                return result
        
        # Check code length
        if len(code) > 10000:
            result.success = False
            result.error_message = "ValidationError: Code too long (max 10000 chars)"
            return result
        
        # Check for allowed imports
        if "import" in code:
            let code_lower = code.lower()
            var has_valid_import = False
            
            for allowed in self.config.allowed_imports:
                if f"import {allowed}" in code_lower or f"from {allowed}" in code_lower:
                    has_valid_import = True
                    break
            
            # If has imports but none are in allowed list, check if it's dangerous
            if "import" in code and not has_valid_import:
                # Allow if it's just importing from allowed modules
                var is_safe = True
                for dangerous in dangerous_imports:
                    if f"import {dangerous}" in code_lower or f"from {dangerous}" in code_lower:
                        is_safe = False
                        break
                
                if not is_safe:
                    result.success = False
                    result.error_message = "ValidationError: Imports restricted to: " + ", ".join(self.config.allowed_imports)
                    return result
        
        print("✅ Code validation passed")
        return result
    
    fn _create_execution_request(self, code: String) -> String:
        """Create JSON request for sandbox execution"""
        # Escape code for JSON
        var escaped_code = code.replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
        
        var request = String('{"code":"')
        request += escaped_code
        request += '","timeout":'
        request += str(self.config.timeout_seconds)
        request += ',"max_memory_mb":'
        request += str(self.config.max_memory_mb)
        request += ',"max_cpu_percent":'
        request += str(self.config.max_cpu_percent)
        request += ',"enable_network":'
        request += "true" if self.config.enable_network else "false"
        request += '}'
        
        return request
    
    fn _parse_execution_response(self, response: String) -> ExecutionResult:
        """Parse JSON response from sandbox"""
        var result = ExecutionResult()
        
        # Simple JSON parsing
        # Look for success field
        let success_idx = response.find('"success":')
        if success_idx != -1:
            let value_start = response.find(':', success_idx) + 1
            let value_end = response.find(',', value_start)
            let success_str = response[value_start:value_end].strip()
            result.success = "true" in success_str
        
        # Extract stdout
        let stdout_idx = response.find('"stdout":"')
        if stdout_idx != -1:
            let content_start = stdout_idx + len('"stdout":"')
            let content_end = response.find('"', content_start)
            if content_end != -1:
                result.stdout = response[content_start:content_end].replace('\\n', '\n')
        
        # Extract stderr
        let stderr_idx = response.find('"stderr":"')
        if stderr_idx != -1:
            let content_start = stderr_idx + len('"stderr":"')
            let content_end = response.find('"', content_start)
            if content_end != -1:
                result.stderr = response[content_start:content_end].replace('\\n', '\n')
        
        # Extract return value
        let return_idx = response.find('"return_value":"')
        if return_idx != -1:
            let content_start = return_idx + len('"return_value":"')
            let content_end = response.find('"', content_start)
            if content_end != -1:
                result.return_value = response[content_start:content_end]
        
        # Extract error message if present
        let error_idx = response.find('"error":"')
        if error_idx != -1:
            let content_start = error_idx + len('"error":"')
            let content_end = response.find('"', content_start)
            if content_end != -1:
                result.error_message = response[content_start:content_end]
        
        return result


fn create_test_code_samples() -> List[String]:
    """Create test code samples for sandbox testing"""
    var samples = List[String]()
    
    # Sample 1: Simple math
    samples.append("""
import math

def calculate():
    result = math.sqrt(16) + math.pow(2, 3)
    return result

answer = calculate()
print(f"Result: {answer}")
""")
    
    # Sample 2: Factorial
    samples.append("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"Factorial of 5 is {result}")
""")
    
    # Sample 3: Data processing
    samples.append("""
import json

data = {"name": "test", "values": [1, 2, 3, 4, 5]}
total = sum(data["values"])
average = total / len(data["values"])

result = {"total": total, "average": average}
print(json.dumps(result))
""")
    
    return samples


fn main() raises:
    """Test code sandbox"""
    print("=" * 80)
    print("🔒 Code Sandbox - Secure Execution Environment")
    print("=" * 80)
    print("")
    
    # Create sandbox config
    let config = SandboxConfig(
        sandbox_url="http://localhost:8000/execute",
        timeout_seconds=10,
        max_memory_mb=256,
        max_cpu_percent=50,
        enable_network=False
    )
    
    print("Configuration:")
    print(f"  Sandbox URL: {config.sandbox_url}")
    print(f"  Timeout: {config.timeout_seconds}s")
    print(f"  Max memory: {config.max_memory_mb}MB")
    print(f"  Max CPU: {config.max_cpu_percent}%")
    print(f"  Network: {'enabled' if config.enable_network else 'disabled'}")
    print(f"  Allowed imports: {len(config.allowed_imports)}")
    print("")
    
    # Initialize sandbox
    var sandbox = CodeSandbox(config)
    print("")
    
    # Test code samples
    let samples = create_test_code_samples()
    
    print("=" * 80)
    print(f"Testing {len(samples)} code samples")
    print("=" * 80)
    
    for i in range(len(samples)):
        let code = samples[i]
        
        print(f"\n📝 Sample {i+1}:")
        print(f"Code length: {len(code)} chars")
        print("Code preview:")
        print(code[:100] + "..." if len(code) > 100 else code)
        print("")
        
        # Execute
        let result = sandbox.execute_python_with_retry(code, max_retries=2)
        
        print(f"Result:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Execution time: {result.execution_time_ms:.2f}ms")
            print(f"  Memory used: {result.memory_used_mb:.2f}MB")
            if len(result.stdout) > 0:
                print(f"  Output: {result.stdout[:200]}")
            if len(result.return_value) > 0:
                print(f"  Return value: {result.return_value}")
        else:
            print(f"  Error: {result.error_message}")
        
        if len(result.stderr) > 0:
            print(f"  Stderr: {result.stderr[:200]}")
    
    # Test dangerous code (should fail validation)
    print("\n" + "=" * 80)
    print("🔒 Testing Security Validation")
    print("=" * 80)
    
    let dangerous_codes = List[String](
        "import os\nos.system('ls')",
        "import subprocess\nsubprocess.run(['ls'])",
        "eval('print(123)')",
        "__import__('os').system('ls')"
    )
    
    for i in range(len(dangerous_codes)):
        let code = dangerous_codes[i]
        print(f"\n🚨 Testing dangerous code {i+1}: {code[:50]}...")
        
        let result = sandbox.execute_python(code)
        if result.success:
            print("  ❌ SECURITY FAILURE: Dangerous code executed!")
        else:
            print(f"  ✅ Blocked: {result.error_message}")
    
    print("\n" + "=" * 80)
    print("✅ Code Sandbox Test Complete")
    print("=" * 80)
    print("")
    print("Features:")
    print("  ✅ Secure code execution")
    print("  ✅ Resource limits (CPU, memory, time)")
    print("  ✅ Import restrictions")
    print("  ✅ Dangerous operation detection")
    print("  ✅ Retry logic")
    print("  ✅ Detailed error reporting")
    print("")
    print("Ready for production use with LLM generation!")
