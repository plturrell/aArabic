# Telecom Tasks Utilities - Pure Mojo Implementation
# Utility structures and functions for telecom tasks

from collections import Dict, List


struct EnvFunctionCall:
    """Represents an environment function call."""
    var env_type: String  # "user" or "assistant"
    var func_name: String
    var arguments: Dict[String, String]
    
    fn __init__(out self, env_type: String, func_name: String):
        self.env_type = env_type
        self.func_name = func_name
        self.arguments = Dict[String, String]()


struct EnvAssertion:
    """Represents an environment assertion."""
    var env_type: String  # "user" or "assistant"
    var func_name: String
    var arguments: Dict[String, String]
    var message: String
    
    fn __init__(out self, env_type: String, func_name: String):
        self.env_type = env_type
        self.func_name = func_name
        self.arguments = Dict[String, String]()
        self.message = ""


struct ToolCall:
    """Represents a tool call."""
    var requestor: String  # "user" or "assistant"
    var name: String
    var arguments: Dict[String, String]
    
    fn __init__(out self, requestor: String, name: String):
        self.requestor = requestor
        self.name = name
        self.arguments = Dict[String, String]()


struct BaseTask:
    """Represents a base task."""
    var name: String
    var description: String
    
    fn __init__(out self, name: String, description: String):
        self.name = name
        self.description = description


struct SelectionSet:
    """Represents a selection set of tasks."""
    var tasks: List[BaseTask]
    
    fn __init__(out self):
        self.tasks = List[BaseTask]()
    
    fn add_task(inout self, task: BaseTask):
        self.tasks.append(task)


struct ComposedTask:
    """Represents a composed task from multiple base tasks."""
    var name: String
    var description: String
    var composed_from: List[BaseTask]
    
    fn __init__(out self, name: String, description: String):
        self.name = name
        self.description = description
        self.composed_from = List[BaseTask]()
    
    fn add_base_task(inout self, task: BaseTask):
        self.composed_from.append(task)


fn get_intent_from_task_id(task_id: String) -> String:
    """Extract the intent from the task_id.
    task_id is of the form: [intent]action1|action2|...|actionk[PERSONA:persona]
    """
    # Find the first [ and ]
    var start_idx = -1
    var end_idx = -1
    for i in range(len(task_id)):
        if task_id[i] == '[' and start_idx == -1:
            start_idx = i
        elif task_id[i] == ']' and start_idx != -1:
            end_idx = i
            break
    
    if start_idx != -1 and end_idx != -1:
        return task_id[start_idx + 1:end_idx]
    return ""


fn get_persona_from_task_id(task_id: String) -> String:
    """Extract the persona from the task_id.
    task_id is of the form: [intent]action1|action2|...|actionk[PERSONA:persona]
    """
    # Find [PERSONA: and the closing ]
    var persona_prefix = "[PERSONA:"
    var start_idx = -1
    
    # Simple search for the persona prefix
    for i in range(len(task_id) - len(persona_prefix)):
        var match = True
        for j in range(len(persona_prefix)):
            if task_id[i + j] != persona_prefix[j]:
                match = False
                break
        if match:
            start_idx = i + len(persona_prefix)
            break
    
    if start_idx != -1:
        var end_idx = start_idx
        while end_idx < len(task_id) and task_id[end_idx] != ']':
            end_idx += 1
        return task_id[start_idx:end_idx]
    return ""


fn get_num_issues_from_task_id(task_id: String) -> Int:
    """Extract the number of issues from the task_id.
    task_id is of the form: [intent]action1|action2|...|actionk[PERSONA:persona]
    """
    var count = 1
    for i in range(len(task_id)):
        if task_id[i] == '|':
            count += 1
    return count

