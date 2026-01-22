# Mock Domain Tools - Pure Mojo Implementation
# Provides tools for the mock domain

from collections import Dict, List

# Tool type constants
alias TOOL_TYPE_READ = "read"
alias TOOL_TYPE_WRITE = "write"
alias TOOL_TYPE_GENERIC = "generic"


struct MockTools:
    """Simple tools for the mock domain."""
    
    var tasks: Dict[String, String]  # task_id -> status
    var task_titles: Dict[String, String]  # task_id -> title
    var task_descriptions: Dict[String, String]  # task_id -> description
    var users: Dict[String, String]  # user_id -> name
    var user_tasks: Dict[String, List[String]]  # user_id -> task_ids
    var task_counter: Int
    
    fn __init__(out self):
        """Initialize mock tools with empty data."""
        self.tasks = Dict[String, String]()
        self.task_titles = Dict[String, String]()
        self.task_descriptions = Dict[String, String]()
        self.users = Dict[String, String]()
        self.user_tasks = Dict[String, List[String]]()
        self.task_counter = 0
    
    fn add_user(inout self, user_id: String, name: String):
        """Add a user to the database."""
        self.users[user_id] = name
        self.user_tasks[user_id] = List[String]()
    
    fn create_task(inout self, user_id: String, title: String, description: String) raises -> String:
        """Create a new task for a user.
        
        Args:
            user_id: The ID of the user creating the task.
            title: The title of the task.
            description: Description of the task.
            
        Returns:
            The task ID of the created task.
            
        Raises:
            Error if user not found.
        """
        if user_id not in self.users:
            raise Error("User " + user_id + " not found")
        
        self.task_counter += 1
        var task_id = "task_" + String(self.task_counter)
        
        self.tasks[task_id] = "pending"
        self.task_titles[task_id] = title
        self.task_descriptions[task_id] = description
        
        # Add task to user's task list
        if user_id in self.user_tasks:
            self.user_tasks[user_id].append(task_id)
        
        return task_id
    
    fn get_users(self) -> List[String]:
        """Get all user IDs in the database."""
        var result = List[String]()
        for user_id in self.users.keys():
            result.append(user_id[])
        return result
    
    fn update_task_status(inout self, task_id: String, status: String) raises -> String:
        """Update the status of a task.
        
        Args:
            task_id: The ID of the task to update.
            status: The new status ("pending" or "completed").
            
        Returns:
            The updated task ID.
            
        Raises:
            Error if task not found.
        """
        if task_id not in self.tasks:
            raise Error("Task " + task_id + " not found")
        
        self.tasks[task_id] = status
        return task_id
    
    fn get_task_status(self, task_id: String) raises -> String:
        """Get the status of a task.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task status.
            
        Raises:
            Error if task not found.
        """
        if task_id not in self.tasks:
            raise Error("Task " + task_id + " not found")
        return self.tasks[task_id]
    
    fn assert_number_of_tasks(self, user_id: String, expected_number: Int) raises -> Bool:
        """Check if the number of tasks for a user is as expected.
        
        Args:
            user_id: The ID of the user.
            expected_number: The expected number of tasks.
            
        Returns:
            True if the number matches, False otherwise.
            
        Raises:
            Error if user not found.
        """
        if user_id not in self.users:
            raise Error("User " + user_id + " not found")
        
        if user_id not in self.user_tasks:
            return expected_number == 0
        
        return len(self.user_tasks[user_id]) == expected_number
    
    fn assert_task_status(self, task_id: String, expected_status: String) raises -> Bool:
        """Check if the status of a task is as expected.
        
        Args:
            task_id: The task ID.
            expected_status: The expected status.
            
        Returns:
            True if status matches, False otherwise.
            
        Raises:
            Error if task not found.
        """
        if task_id not in self.tasks:
            raise Error("Task " + task_id + " not found")
        return self.tasks[task_id] == expected_status
    
    fn transfer_to_human_agents(self, summary: String) -> String:
        """Transfer the user to a human agent.
        
        Args:
            summary: A summary of the user's issue.
            
        Returns:
            A message indicating successful transfer.
        """
        return "Transfer successful"

