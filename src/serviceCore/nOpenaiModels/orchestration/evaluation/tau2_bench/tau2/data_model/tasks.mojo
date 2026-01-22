# tasks.mojo
# Migrated from tasks.py
# Task definitions for TAU2-Bench

from collections import Dict, List

struct TaskParameter:
    """Parameter definition for a task"""
    var name: String
    var param_type: String
    var description: String
    var required: Bool
    var default_value: String
    
    fn __init__(out self):
        self.name = ""
        self.param_type = "string"
        self.description = ""
        self.required = False
        self.default_value = ""
    
    fn __init__(out self, name: String, param_type: String, description: String, required: Bool):
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required
        self.default_value = ""

struct TaskConstraint:
    """Constraint definition for a task"""
    var constraint_type: String  # "time", "resource", "dependency"
    var value: String
    var description: String
    
    fn __init__(out self):
        self.constraint_type = ""
        self.value = ""
        self.description = ""
    
    fn __init__(out self, constraint_type: String, value: String, description: String):
        self.constraint_type = constraint_type
        self.value = value
        self.description = description

struct TaskGoal:
    """Goal definition for a task"""
    var goal_id: String
    var description: String
    var success_criteria: List[String]
    var weight: Float32  # For weighted multi-goal tasks
    
    fn __init__(out self):
        self.goal_id = ""
        self.description = ""
        self.success_criteria = List[String]()
        self.weight = 1.0
    
    fn __init__(out self, goal_id: String, description: String):
        self.goal_id = goal_id
        self.description = description
        self.success_criteria = List[String]()
        self.weight = 1.0
    
    fn add_criterion(mut self, criterion: String):
        """Add a success criterion"""
        self.success_criteria.append(criterion)

struct TaskDefinition:
    """Complete definition of a task in TAU2-Bench"""
    var task_id: String
    var domain: String
    var name: String
    var description: String
    var difficulty: String  # "easy", "medium", "hard"
    var parameters: List[TaskParameter]
    var constraints: List[TaskConstraint]
    var goals: List[TaskGoal]
    var required_tools: List[String]
    var estimated_turns: Int
    var metadata: Dict[String, String]
    
    fn __init__(out self):
        self.task_id = ""
        self.domain = ""
        self.name = ""
        self.description = ""
        self.difficulty = "medium"
        self.parameters = List[TaskParameter]()
        self.constraints = List[TaskConstraint]()
        self.goals = List[TaskGoal]()
        self.required_tools = List[String]()
        self.estimated_turns = 10
        self.metadata = Dict[String, String]()
    
    fn __init__(out self, task_id: String, domain: String, name: String, description: String):
        self.task_id = task_id
        self.domain = domain
        self.name = name
        self.description = description
        self.difficulty = "medium"
        self.parameters = List[TaskParameter]()
        self.constraints = List[TaskConstraint]()
        self.goals = List[TaskGoal]()
        self.required_tools = List[String]()
        self.estimated_turns = 10
        self.metadata = Dict[String, String]()
    
    fn add_parameter(mut self, param: TaskParameter):
        """Add a task parameter"""
        self.parameters.append(param)
    
    fn add_constraint(mut self, constraint: TaskConstraint):
        """Add a task constraint"""
        self.constraints.append(constraint)
    
    fn add_goal(mut self, goal: TaskGoal):
        """Add a task goal"""
        self.goals.append(goal)
    
    fn add_required_tool(mut self, tool_name: String):
        """Add a required tool"""
        self.required_tools.append(tool_name)
    
    fn set_difficulty(mut self, difficulty: String):
        """Set task difficulty"""
        self.difficulty = difficulty
    
    fn set_estimated_turns(mut self, turns: Int):
        """Set estimated number of turns"""
        self.estimated_turns = turns
    
    fn add_metadata(mut self, key: String, value: String):
        """Add metadata"""
        self.metadata[key] = value
    
    fn get_parameter_count(self) -> Int:
        """Get number of parameters"""
        return len(self.parameters)
    
    fn get_goal_count(self) -> Int:
        """Get number of goals"""
        return len(self.goals)
    
    fn get_required_tool_count(self) -> Int:
        """Get number of required tools"""
        return len(self.required_tools)

struct TaskInstance:
    """Instantiated task with specific parameters"""
    var instance_id: String
    var task_def: TaskDefinition
    var parameter_values: Dict[String, String]
    var context: String
    var start_state: String
    var expected_end_state: String
    
    fn __init__(out self):
        self.instance_id = ""
        self.task_def = TaskDefinition()
        self.parameter_values = Dict[String, String]()
        self.context = ""
        self.start_state = ""
        self.expected_end_state = ""
    
    fn __init__(out self, instance_id: String, task_def: TaskDefinition):
        self.instance_id = instance_id
        self.task_def = task_def
        self.parameter_values = Dict[String, String]()
        self.context = ""
        self.start_state = ""
        self.expected_end_state = ""
    
    fn set_parameter(mut self, name: String, value: String):
        """Set a parameter value"""
        self.parameter_values[name] = value
    
    fn get_parameter(self, name: String) -> String:
        """Get a parameter value"""
        if name in self.parameter_values:
            return self.parameter_values[name]
        return ""
    
    fn set_context(mut self, context: String):
        """Set task context"""
        self.context = context
    
    fn set_states(mut self, start_state: String, expected_end_state: String):
        """Set start and expected end states"""
        self.start_state = start_state
        self.expected_end_state = expected_end_state

struct TaskEvaluation:
    """Evaluation result for a completed task"""
    var instance_id: String
    var success: Bool
    var goals_achieved: List[String]
    var goals_failed: List[String]
    var score: Float32  # 0.0 to 1.0
    var completion_time: String
    var turns_used: Int
    var feedback: String
    var detailed_metrics: Dict[String, String]
    
    fn __init__(out self):
        self.instance_id = ""
        self.success = False
        self.goals_achieved = List[String]()
        self.goals_failed = List[String]()
        self.score = 0.0
        self.completion_time = ""
        self.turns_used = 0
        self.feedback = ""
        self.detailed_metrics = Dict[String, String]()
    
    fn __init__(out self, instance_id: String):
        self.instance_id = instance_id
        self.success = False
        self.goals_achieved = List[String]()
        self.goals_failed = List[String]()
        self.score = 0.0
        self.completion_time = ""
        self.turns_used = 0
        self.feedback = ""
        self.detailed_metrics = Dict[String, String]()
    
    fn mark_goal_achieved(mut self, goal_id: String):
        """Mark a goal as achieved"""
        self.goals_achieved.append(goal_id)
    
    fn mark_goal_failed(mut self, goal_id: String):
        """Mark a goal as failed"""
        self.goals_failed.append(goal_id)
    
    fn calculate_score(mut self, total_goals: Int):
        """Calculate success score based on goals"""
        if total_goals > 0:
            let achieved = len(self.goals_achieved)
            self.score = Float32(achieved) / Float32(total_goals)
            self.success = self.score >= 0.8  # 80% threshold
        else:
            self.score = 0.0
            self.success = False
    
    fn add_metric(mut self, key: String, value: String):
        """Add a detailed metric"""
        self.detailed_metrics[key] = value
    
    fn set_feedback(mut self, feedback: String):
        """Set evaluation feedback"""
        self.feedback = feedback

struct TaskRegistry:
    """Registry of all available tasks"""
    var tasks: Dict[String, TaskDefinition]
    var domain_index: Dict[String, List[String]]  # domain -> task_ids
    var difficulty_index: Dict[String, List[String]]  # difficulty -> task_ids
    
    fn __init__(out self):
        self.tasks = Dict[String, TaskDefinition]()
        self.domain_index = Dict[String, List[String]]()
        self.difficulty_index = Dict[String, List[String]]()
    
    fn register_task(mut self, task: TaskDefinition):
        """Register a new task"""
        self.tasks[task.task_id] = task
        
        # Update domain index
        if task.domain not in self.domain_index:
            self.domain_index[task.domain] = List[String]()
        var domain_tasks = self.domain_index[task.domain]
        domain_tasks.append(task.task_id)
        self.domain_index[task.domain] = domain_tasks
        
        # Update difficulty index
        if task.difficulty not in self.difficulty_index:
            self.difficulty_index[task.difficulty] = List[String]()
        var diff_tasks = self.difficulty_index[task.difficulty]
        diff_tasks.append(task.task_id)
        self.difficulty_index[task.difficulty] = diff_tasks
    
    fn get_task(self, task_id: String) -> TaskDefinition:
        """Get a task by ID"""
        if task_id in self.tasks:
            return self.tasks[task_id]
        return TaskDefinition()
    
    fn get_tasks_by_domain(self, domain: String) -> List[String]:
        """Get all task IDs for a domain"""
        if domain in self.domain_index:
            return self.domain_index[domain]
        return List[String]()
    
    fn get_tasks_by_difficulty(self, difficulty: String) -> List[String]:
        """Get all task IDs for a difficulty level"""
        if difficulty in self.difficulty_index:
            return self.difficulty_index[difficulty]
        return List[String]()
    
    fn task_exists(self, task_id: String) -> Bool:
        """Check if a task exists"""
        return task_id in self.tasks
