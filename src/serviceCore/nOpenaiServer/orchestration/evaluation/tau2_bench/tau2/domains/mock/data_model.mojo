# Mock Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the mock domain

from collections import Dict, List

# Task status enum values
alias TASK_STATUS_PENDING = "pending"
alias TASK_STATUS_COMPLETED = "completed"


struct Task:
    """Represents a task in the mock domain."""
    
    var task_id: String
    var title: String
    var description: String
    var status: String  # "pending" or "completed"
    
    fn __init__(out self, task_id: String, title: String, description: String, status: String):
        """Initialize a task.
        
        Args:
            task_id: Unique identifier for the task.
            title: Title of the task.
            description: Description of the task.
            status: Status of the task ("pending" or "completed").
        """
        self.task_id = task_id
        self.title = title
        self.description = description
        self.status = status
    
    fn __init__(out self, task_id: String, title: String, status: String):
        """Initialize a task without description.
        
        Args:
            task_id: Unique identifier for the task.
            title: Title of the task.
            status: Status of the task.
        """
        self.task_id = task_id
        self.title = title
        self.description = ""
        self.status = status
    
    fn is_pending(self) -> Bool:
        """Check if the task is pending."""
        return self.status == TASK_STATUS_PENDING
    
    fn is_completed(self) -> Bool:
        """Check if the task is completed."""
        return self.status == TASK_STATUS_COMPLETED


struct User:
    """Represents a user in the mock domain."""
    
    var user_id: String
    var name: String
    var tasks: List[String]  # List of task IDs
    
    fn __init__(out self, user_id: String, name: String):
        """Initialize a user.
        
        Args:
            user_id: Unique identifier for the user.
            name: User's name.
        """
        self.user_id = user_id
        self.name = name
        self.tasks = List[String]()
    
    fn __init__(out self, user_id: String, name: String, tasks: List[String]):
        """Initialize a user with tasks.
        
        Args:
            user_id: Unique identifier for the user.
            name: User's name.
            tasks: List of task IDs assigned to the user.
        """
        self.user_id = user_id
        self.name = name
        self.tasks = tasks
    
    fn add_task(inout self, task_id: String):
        """Add a task to the user's task list."""
        self.tasks.append(task_id)
    
    fn task_count(self) -> Int:
        """Return the number of tasks assigned to this user."""
        return len(self.tasks)


struct MockDB:
    """Simple database with users and their tasks."""
    
    var tasks: Dict[String, Task]
    var users: Dict[String, User]
    
    fn __init__(out self):
        """Initialize an empty mock database."""
        self.tasks = Dict[String, Task]()
        self.users = Dict[String, User]()
    
    fn add_task(inout self, task: Task):
        """Add a task to the database."""
        self.tasks[task.task_id] = task
    
    fn add_user(inout self, user: User):
        """Add a user to the database."""
        self.users[user.user_id] = user
    
    fn get_task(self, task_id: String) raises -> Task:
        """Get a task by ID.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task.
            
        Raises:
            Error if task not found.
        """
        if task_id not in self.tasks:
            raise Error("Task " + task_id + " not found")
        return self.tasks[task_id]
    
    fn get_user(self, user_id: String) raises -> User:
        """Get a user by ID.
        
        Args:
            user_id: The user ID.
            
        Returns:
            The user.
            
        Raises:
            Error if user not found.
        """
        if user_id not in self.users:
            raise Error("User " + user_id + " not found")
        return self.users[user_id]
    
    fn task_count(self) -> Int:
        """Return total number of tasks."""
        return len(self.tasks)
    
    fn user_count(self) -> Int:
        """Return total number of users."""
        return len(self.users)

