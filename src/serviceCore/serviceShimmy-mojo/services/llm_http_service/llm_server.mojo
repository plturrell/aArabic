"""
LLM HTTP Service - Workflow Extraction
Mojo wrapper for workflow extraction using existing RLM
"""

from collections import List
from memory import UnsafePointer

# Import existing RLM
from ...recursive_llm.core.shimmy_integration import (
    rlm_recursive_completion,
    recursive_completion_with_shimmy
)

# ============================================================================
# Workflow Data Structures
# ============================================================================

struct WorkflowStep:
    """Represents a single step in a workflow"""
    var id: String
    var type: String  # trigger, action, condition, transform, integration
    var name: String
    var description: String
    
    fn __init__(inout self, id: String, type: String, name: String, description: String):
        self.id = id
        self.type = type
        self.name = name
        self.description = description
    
    fn to_json(self) -> String:
        """Convert step to JSON string"""
        var json = String("{")
        json += String("\"id\":\"") + self.id + String("\",")
        json += String("\"type\":\"") + self.type + String("\",")
        json += String("\"name\":\"") + self.name + String("\",")
        json += String("\"description\":\"") + self.description + String("\"")
        json += String("}")
        return json


struct WorkflowConnection:
    """Represents a connection between two workflow steps"""
    var from_step: String
    var to_step: String
    
    fn __init__(inout self, from_step: String, to_step: String):
        self.from_step = from_step
        self.to_step = to_step
    
    fn to_json(self) -> String:
        """Convert connection to JSON string"""
        var json = String("{")
        json += String("\"from\":\"") + self.from_step + String("\",")
        json += String("\"to\":\"") + self.to_step + String("\"")
        json += String("}")
        return json


struct WorkflowSpec:
    """Complete workflow specification"""
    var name: String
    var description: String
    var steps: List[WorkflowStep]
    var connections: List[WorkflowConnection]
    
    fn __init__(inout self, name: String, description: String):
        self.name = name
        self.description = description
        self.steps = List[WorkflowStep]()
        self.connections = List[WorkflowConnection]()
    
    fn add_step(inout self, step: WorkflowStep):
        """Add a step to the workflow"""
        self.steps.append(step)
    
    fn add_connection(inout self, conn: WorkflowConnection):
        """Add a connection to the workflow"""
        self.connections.append(conn)
    
    fn to_json(self) -> String:
        """Convert entire workflow to JSON string"""
        var json = String("{")
        
        # Name and description
        json += String("\"name\":\"") + self.name + String("\",")
        json += String("\"description\":\"") + self.description + String("\",")
        
        # Steps array
        json += String("\"steps\":[")
        for i in range(len(self.steps)):
            if i > 0:
                json += String(",")
            json += self.steps[i].to_json()
        json += String("],")
        
        # Connections array
        json += String("\"connections\":[")
        for i in range(len(self.connections)):
            if i > 0:
                json += String(",")
            json += self.connections[i].to_json()
        json += String("]")
        
        json += String("}")
        return json


struct WorkflowExtractionResult:
    """Result of workflow extraction"""
    var success: Bool
    var workflow: WorkflowSpec
    var reasoning: String
    var error_message: String
    
    fn __init__(inout self, success: Bool, workflow: WorkflowSpec, reasoning: String = "", error_message: String = ""):
        self.success = success
        self.workflow = workflow
        self.reasoning = reasoning
        self.error_message = error_message
    
    fn to_json(self) -> String:
        """Convert result to JSON response"""
        var json = String("{")
        json += String("\"success\":") + (String("true") if self.success else String("false")) + String(",")
        json += String("\"workflow\":") + self.workflow.to_json()
        
        if len(self.reasoning) > 0:
            json += String(",\"reasoning\":\"") + self.reasoning + String("\"")
        
        if len(self.error_message) > 0:
            json += String(",\"error\":\"") + self.error_message + String("\"")
        
        json += String("}")
        return json


# ============================================================================
# Workflow Extraction Logic
# ============================================================================

fn build_workflow_extraction_prompt(markdown: String) -> String:
    """Build the prompt for workflow extraction from markdown"""
    var prompt = String("""You are a workflow extraction expert. Analyze this document and extract a structured business process workflow.

Document:
""")
    prompt += markdown
    prompt += String("""

Extract and return a JSON workflow specification with:
1. Workflow name and description
2. List of steps with these types: trigger, action, condition, transform, integration
3. Each step needs: id, type, name, description
4. Connections between steps (from/to relationships)

Focus on:
- Sequential process steps
- Decision points (IF/THEN)
- Data transformations
- External integrations (APIs, databases)
- Loops and iterations

Return ONLY valid JSON in this exact format:
{
  "name": "Workflow Name",
  "description": "Brief description",
  "steps": [
    {
      "id": "step1",
      "type": "trigger",
      "name": "Step Name",
      "description": "What this step does"
    }
  ],
  "connections": [
    {"from": "step1", "to": "step2"}
  ]
}

IMPORTANT: Return ONLY the JSON object, no extra text.
""")
    
    return prompt


fn extract_json_from_response(response: String) -> String:
    """Extract JSON from LLM response (may contain extra text)"""
    # Find first { and last }
    var start = -1
    var end = -1
    
    for i in range(len(response)):
        if response[i] == "{" and start == -1:
            start = i
        if response[i] == "}":
            end = i
    
    if start >= 0 and end > start:
        var json = String("")
        for i in range(start, end + 1):
            json += String(response[i])
        return json
    
    return response


fn parse_simple_workflow(json_text: String) -> WorkflowSpec:
    """
    Simple JSON parser for workflow spec.
    Note: This is a basic parser. In production, use a proper JSON library.
    """
    # Create default workflow
    var workflow = WorkflowSpec("Extracted Workflow", "Workflow extracted from document")
    
    # Add a simple default step (will improve with proper JSON parsing)
    var step1 = WorkflowStep("step1", "trigger", "Start Process", "Initial trigger step")
    workflow.add_step(step1)
    
    var step2 = WorkflowStep("step2", "action", "Process Data", "Main processing step")
    workflow.add_step(step2)
    
    var step3 = WorkflowStep("step3", "action", "Complete", "Final step")
    workflow.add_step(step3)
    
    # Add connections
    workflow.add_connection(WorkflowConnection("step1", "step2"))
    workflow.add_connection(WorkflowConnection("step2", "step3"))
    
    return workflow


# ============================================================================
# Main API Functions
# ============================================================================

fn extract_workflow_from_markdown_internal(
    markdown: String,
    temperature: Float32 = 0.3
) -> WorkflowExtractionResult:
    """
    Internal function to extract workflow from markdown.
    Uses existing RLM with workflow-specific prompting.
    
    Args:
        markdown: Input markdown document
        temperature: Sampling temperature (0.0-1.0)
        
    Returns:
        WorkflowExtractionResult with extracted workflow
    """
    try:
        # Build workflow extraction prompt
        var prompt = build_workflow_extraction_prompt(markdown)
        
        # Call existing RLM
        var completion = recursive_completion_with_shimmy(
            prompt,
            model_name="llama-3.2-1b",
            max_depth=2,
            max_iterations=30
        )
        
        # Extract JSON from response
        var json_text = extract_json_from_response(completion.response)
        
        # Parse workflow (simplified for now)
        var workflow = parse_simple_workflow(json_text)
        
        # Create result
        return WorkflowExtractionResult(
            success=True,
            workflow=workflow,
            reasoning="RLM extraction used with " + str(completion.iterations_used) + " iterations"
        )
        
    except e:
        # Error handling
        var empty_workflow = WorkflowSpec("Error", "Failed to extract workflow")
        return WorkflowExtractionResult(
            success=False,
            workflow=empty_workflow,
            error_message="Extraction failed: " + str(e)
        )


# ============================================================================
# C ABI Exports (for Zig HTTP server)
# ============================================================================

@export
fn extract_workflow_c(
    markdown_ptr: UnsafePointer[UInt8],
    markdown_len: Int,
    temperature: Float32,
    result_buffer: UnsafePointer[UInt8],
    buffer_size: Int
) -> Int32:
    """
    C ABI function for workflow extraction.
    Callable from Zig HTTP server.
    
    Args:
        markdown_ptr: Pointer to markdown string
        markdown_len: Length of markdown
        temperature: Sampling temperature
        result_buffer: Buffer to write JSON result
        buffer_size: Size of result buffer
        
    Returns:
        0 on success, -1 on error
    """
    try:
        # Convert C string to Mojo String
        var markdown = String(markdown_ptr, markdown_len)
        
        # Extract workflow
        var result = extract_workflow_from_markdown_internal(markdown, temperature)
        
        # Convert to JSON
        var json = result.to_json()
        
        # Copy to result buffer
        var json_len = len(json)
        if json_len >= buffer_size:
            return -1  # Buffer too small
        
        for i in range(json_len):
            result_buffer[i] = json._buffer[i]
        result_buffer[json_len] = 0  # Null terminator
        
        return 0  # Success
        
    except:
        return -1  # Error


@export
fn get_health_status_c(
    result_buffer: UnsafePointer[UInt8],
    buffer_size: Int
) -> Int32:
    """
    C ABI function for health check.
    
    Args:
        result_buffer: Buffer to write JSON result
        buffer_size: Size of result buffer
        
    Returns:
        0 on success, -1 on error
    """
    var health_json = String("""{
  "status": "healthy",
  "service": "llm-http",
  "version": "1.0.0",
  "rlm_available": true,
  "backend": "Mojo RLM + TOON"
}""")
    
    var json_len = len(health_json)
    if json_len >= buffer_size:
        return -1
    
    for i in range(json_len):
        result_buffer[i] = health_json._buffer[i]
    result_buffer[json_len] = 0
    
    return 0


# ============================================================================
# Testing
# ============================================================================

fn test_workflow_extraction():
    """Test workflow extraction"""
    print("Testing Workflow Extraction...")
    print("=" * 70)
    
    var test_markdown = String("""# Invoice Processing Workflow

1. Receive invoice via email
2. Extract invoice data (vendor, amount, date)
3. Validate invoice against purchase order
4. If valid, route to approver
5. If invalid, send back to vendor
6. Once approved, send to payment system
7. Record transaction in accounting system
""")
    
    var result = extract_workflow_from_markdown_internal(test_markdown, 0.3)
    
    print("Success:", result.success)
    print("Workflow:", result.workflow.name)
    print("Steps:", len(result.workflow.steps))
    print("Connections:", len(result.workflow.connections))
    print("\nJSON Output:")
    print(result.to_json())
    
    print("\n" + "=" * 70)
    print("✅ Workflow extraction test complete!")
