# tau2/environment/toolkit.mojo
# Migrated from tau2/environment/toolkit.py
# Tool collection management

from collections import List, Dict
from tau2.environment.tool import Tool, ToolSchema

struct Toolkit:
    """
    Collection of tools available to an agent.
    Manages tool registration, lookup, and execution.
    """
    var tools: List[Tool]
    var tool_names: List[String]
    
    fn __init__(inout self):
        """Initialize empty toolkit."""
        self.tools = List[Tool]()
        self.tool_names = List[String]()
    
    fn add_tool(inout self, tool: Tool):
        """
        Add a tool to the toolkit.
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        self.tool_names.append(tool.name)
    
    fn get_tool(self, name: String) -> Tool:
        """
        Get a tool by name.
        
        Args:
            name: Tool name to lookup
            
        Returns:
            Tool if found, empty tool otherwise
        """
        for i in range(len(self.tools)):
            if self.tools[i].name == name:
                return self.tools[i]
        return Tool(name="not_found")
    
    fn has_tool(self, name: String) -> Bool:
        """
        Check if toolkit contains a tool with given name.
        
        Args:
            name: Tool name to check
            
        Returns:
            True if tool exists
        """
        for i in range(len(self.tool_names)):
            if self.tool_names[i] == name:
                return True
        return False
    
    fn get_openai_schemas(self) -> List[ToolSchema]:
        """
        Get OpenAI schemas for all tools.
        
        Returns:
            List of ToolSchema objects
        """
        var schemas = List[ToolSchema]()
        for i in range(len(self.tools)):
            schemas.append(self.tools[i].get_openai_schema())
        return schemas
    
    fn execute_tool(self, name: String, arguments: String) raises -> String:
        """
        Execute a tool by name with given arguments.
        
        Args:
            name: Tool name
            arguments: JSON string of arguments
            
        Returns:
            JSON string result
        """
        var tool = self.get_tool(name)
        if tool.name == "not_found":
            raise Error("Tool not found: " + name)
        return tool.call(arguments)
    
    fn count(self) -> Int:
        """Get number of tools in toolkit."""
        return len(self.tools)
    
    fn list_tool_names(self) -> List[String]:
        """Get list of all tool names."""
        return self.tool_names
    
    fn to_string(self) -> String:
        """String representation of toolkit."""
        var result = "Toolkit with " + String(len(self.tools)) + " tools:\n"
        for i in range(len(self.tool_names)):
            result += "  - " + self.tool_names[i] + "\n"
        return result
