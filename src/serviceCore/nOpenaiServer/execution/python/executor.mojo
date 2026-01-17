# Python Code Executor for Recursive LLM
# Executes Python code blocks via FFI and captures output

from python import Python, PythonObject
from collections import Dict, List
from memory import UnsafePointer

# ============================================================================
# Execution Result Types
# ============================================================================

@value
struct ExecutionResult:
    """Result from executing Python code"""
    var success: Bool
    var output: String
    var error: String
    var execution_time: Float64
    
    fn __init__(inout self):
        self.success = False
        self.output = ""
        self.error = ""
        self.execution_time = 0.0
    
    fn __init__(inout self, success: Bool, output: String, error: String = ""):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = 0.0


# ============================================================================
# Python Code Executor
# ============================================================================

struct PythonExecutor:
    """
    Executes Python code via Mojo's Python FFI.
    Provides safe execution with output capture and error handling.
    """
    var python: PythonObject
    var globals_dict: PythonObject
    var locals_dict: PythonObject
    var verbose: Bool
    
    fn __init__(inout self, verbose: Bool = False) raises:
        """
        Initialize Python executor with isolated namespace.
        
        Args:
            verbose: Print execution details
        """
        self.verbose = verbose
        
        # Initialize Python interpreter
        self.python = Python.import_module("builtins")
        
        # Create isolated global and local namespaces
        self.globals_dict = Python.dict()
        self.locals_dict = Python.dict()
        
        # Add safe builtins
        self._setup_safe_environment()
        
        if self.verbose:
            print("🐍 Python Executor initialized")
    
    fn _setup_safe_environment(inout self) raises:
        """
        Setup a safe Python environment with common modules.
        Restricts dangerous operations while allowing useful functionality.
        """
        # Import safe standard library modules
        try:
            var sys = Python.import_module("sys")
            var os = Python.import_module("os")
            var json = Python.import_module("json")
            var re = Python.import_module("re")
            var math = Python.import_module("math")
            var datetime = Python.import_module("datetime")
            
            # Add to globals
            self.globals_dict["sys"] = sys
            self.globals_dict["os"] = os
            self.globals_dict["json"] = json
            self.globals_dict["re"] = re
            self.globals_dict["math"] = math
            self.globals_dict["datetime"] = datetime
            
            # Add common functions
            self.globals_dict["print"] = self.python.print
            self.globals_dict["len"] = self.python.len
            self.globals_dict["str"] = self.python.str
            self.globals_dict["int"] = self.python.int
            self.globals_dict["float"] = self.python.float
            self.globals_dict["list"] = self.python.list
            self.globals_dict["dict"] = self.python.dict
            self.globals_dict["range"] = self.python.range
            self.globals_dict["enumerate"] = self.python.enumerate
            self.globals_dict["zip"] = self.python.zip
            
            if self.verbose:
                print("  ✅ Safe environment configured")
        except:
            print("  ⚠️  Warning: Could not import all standard modules")
    
    fn execute(inout self, code: String) -> ExecutionResult:
        """
        Execute Python code and capture output.
        
        Args:
            code: Python code string to execute
            
        Returns:
            ExecutionResult with success status, output, and any errors
        """
        if self.verbose:
            print("\n📝 Executing Python code:")
            print("─" * 40)
            print(code[:200])
            if len(code) > 200:
                print("... (truncated)")
            print("─" * 40)
        
        var result = ExecutionResult()
        
        try:
            # Capture stdout using StringIO
            var io = Python.import_module("io")
            var contextlib = Python.import_module("contextlib")
            
            var stdout_capture = io.StringIO()
            
            # Execute code with stdout capture
            with contextlib.redirect_stdout(stdout_capture):
                # Use exec() to execute code
                var exec_fn = Python.eval("exec", self.globals_dict, self.locals_dict)
                exec_fn(code, self.globals_dict, self.locals_dict)
            
            # Get captured output
            var output_str = stdout_capture.getvalue()
            
            result.success = True
            result.output = str(output_str)
            
            if self.verbose:
                print("✅ Execution successful")
                if len(result.output) > 0:
                    print("📤 Output:")
                    print(result.output[:500])
                    if len(result.output) > 500:
                        print("... (truncated)")
        
        except e:
            result.success = False
            result.error = str(e)
            
            if self.verbose:
                print("❌ Execution failed:")
                print(result.error)
        
        return result
    
    fn execute_with_context(inout self, code: String, context: Dict[String, String]) -> ExecutionResult:
        """
        Execute Python code with additional context variables.
        
        Args:
            code: Python code to execute
            context: Dictionary of variables to inject into execution context
            
        Returns:
            ExecutionResult with output
        """
        # Add context variables to locals
        for key in context.keys():
            try:
                self.locals_dict[key] = context[key]
            except:
                if self.verbose:
                    print("  ⚠️  Could not inject context:", key)
        
        return self.execute(code)
    
    fn evaluate_expression(inout self, expression: String) -> ExecutionResult:
        """
        Evaluate a Python expression and return the result.
        Uses eval() instead of exec() for expressions.
        
        Args:
            expression: Python expression to evaluate
            
        Returns:
            ExecutionResult with the expression result
        """
        var result = ExecutionResult()
        
        try:
            var eval_fn = Python.eval("eval", self.globals_dict, self.locals_dict)
            var value = eval_fn(expression, self.globals_dict, self.locals_dict)
            
            result.success = True
            result.output = str(value)
            
            if self.verbose:
                print("✅ Expression evaluated:", expression, "=", result.output)
        
        except e:
            result.success = False
            result.error = str(e)
            
            if self.verbose:
                print("❌ Evaluation failed:", result.error)
        
        return result
    
    fn register_function(inout self, name: String, function: PythonObject):
        """
        Register a custom function in the execution environment.
        Useful for adding helper functions like llm_query placeholder.
        
        Args:
            name: Function name to register
            function: Python callable object
        """
        try:
            self.globals_dict[name] = function
            
            if self.verbose:
                print("  ✅ Registered function:", name)
        except:
            if self.verbose:
                print("  ⚠️  Could not register function:", name)
    
    fn clear_namespace(inout self):
        """
        Clear local namespace (keep globals).
        Useful for isolating execution between code blocks.
        """
        try:
            self.locals_dict = Python.dict()
            
            if self.verbose:
                print("  🧹 Local namespace cleared")
        except:
            if self.verbose:
                print("  ⚠️  Could not clear namespace")
    
    fn get_variable(inout self, name: String) -> PythonObject:
        """
