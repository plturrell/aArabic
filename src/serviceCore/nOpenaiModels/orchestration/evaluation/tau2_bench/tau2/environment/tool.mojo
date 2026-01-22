# tau2/environment/tool.mojo
# Migrated from tau2/environment/tool.py
# Tool definition and execution framework

from collections import Dict, List

struct ToolParameter:
    """Parameter definition for a tool."""
    var name: String
    var type: String  # "string", "integer", "boolean", "object", "array"
    var description: String
    var required: Bool
    var default_value: String  # JSON string representation
    
    fn __init__(
        inout self,
        name: String,
        type: String = "string",
        description: String = "",
        required: Bool = False,
        default_value: String = ""
    ):
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.default_value = default_value

struct ToolSchema:
    """OpenAI-compatible tool schema."""
    var type: String  # Always "function"
    var function_name: String
    var function_description: String
    var parameters: String  # JSON schema string
    
    fn __init__(
        inout self,
        function_name: String,
        function_description: String = "",
        parameters: String = "{}"
    ):
        self.type = "function"
        self.function_name = function_name
        self.function_description = function_description
        self.parameters = parameters
    
    fn to_json(self) -> String:
        """Convert to JSON string for OpenAI API."""
        var result = "{"
        result += '"type": "function",'
        result += '"function": {'
        result += '"name": "' + self.function_name + '",'
        result += '"description": "' + self.function_description + '",'
        result += '"parameters": ' + self.parameters
        result += '}}'
        return result

struct Tool:
    """
    Tool that can be called by LLMs.
    Simplified from Python version - uses JSON for parameters and return values.
    """
    var name: String
    var short_desc: String
    var long_desc: String
    var parameters: List[ToolParameter]
    var returns_desc: String
    var examples: List[String]
    var use_short_desc: Bool
    
    fn __init__(
        inout self,
        name: String,
        short_desc: String = "",
        long_desc: String = "",
        parameters: List[ToolParameter] = List[ToolParameter](),
        returns_desc: String = "",
        examples: List[String] = List[String](),
        use_short_desc: Bool = False
    ):
        self.name = name
        self.short_desc = short_desc
        self.long_desc = long_desc
        self.parameters = parameters
        self.returns_desc = returns_desc
        self.examples = examples
        self.use_short_desc = use_short_desc
    
    fn get_description(self) -> String:
        """Get the tool description (short or full)."""
        if self.short_desc == "":
            return self.name
        
        if self.long_desc == "" or self.use_short_desc:
            return self.short_desc
        
        return self.short_desc + "\n\n" + self.long_desc
    
    fn get_openai_schema(self) -> ToolSchema:
        """
        Get the OpenAI-compatible schema for this tool.
        
        Returns:
            ToolSchema with function name, description, and parameters
        """
        # Build parameters JSON schema
        var params_json = '{"type": "object", "properties": {'
        var required_params = List[String]()
        
        for i in range(len(self.parameters)):
            var param = self.parameters[i]
            if i > 0:
                params_json += ", "
            
            params_json += '"' + param.name + '": {'
            params_json += '"type": "' + param.type + '"'
            if param.description != "":
                params_json += ', "description": "' + param.description + '"'
            params_json += '}'
            
            if param.required:
                required_params.append(param.name)
        
        params_json += '}'
        
        # Add required fields
        if len(required_params) > 0:
            params_json += ', "required": ['
            for i in range(len(required_params)):
                if i > 0:
                    params_json += ", "
                params_json += '"' + required_params[i] + '"'
            params_json += ']'
        
        params_json += '}'
        
        return ToolSchema(
            function_name=self.name,
            function_description=self.get_description(),
            parameters=params_json
        )
    
    fn call(self, arguments: String) raises -> String:
        """
        Execute the tool with given arguments.
        
        Args:
            arguments: JSON string of arguments
            
        Returns:
            JSON string of result
        """
        # TODO: Implement tool execution
        # This will need to dispatch to actual tool implementations
        return '{"result": "Tool execution placeholder for ' + self.name + '"}'
    
    fn to_string(self) -> String:
        """String representation of the tool."""
        var result = "Tool: " + self.name + "\n"
        result += "Description: " + self.get_description() + "\n"
        result += "Parameters: " + String(len(self.parameters)) + "\n"
        for i in range(len(self.parameters)):
            var param = self.parameters[i]
            result += "  - " + param.name + " (" + param.type + ")"
            if param.required:
                result += " [required]"
            result += "\n"
        return result

fn as_tool(
    name: String,
    short_desc: String = "",
    long_desc: String = "",
    parameters: List[ToolParameter] = List[ToolParameter]()
) -> Tool:
    """
    Create a tool from basic components.
    
    Args:
        name: Tool name
        short_desc: Short description
        long_desc: Long description
        parameters: List of parameters
        
    Returns:
        Tool instance
    """
    return Tool(
        name=name,
        short_desc=short_desc,
        long_desc=long_desc,
        parameters=parameters
    )
