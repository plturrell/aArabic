# base.mojo
# Migrated from base.py
# Base user interface and profile definitions

from collections import Dict, List

trait User:
    """Base trait for user implementations"""
    
    fn get_id(self) -> String:
        """Get user identifier"""
        ...
    
    fn get_profile(self) -> UserProfile:
        """Get user profile"""
        ...
    
    fn generate_query(mut self, context: String) -> String:
        """
        Generate a user query based on context
        
        Args:
            context: Current context/state
            
        Returns:
            User query string
        """
        ...
    
    fn provide_feedback(mut self, agent_response: String) -> String:
        """
        Provide feedback on agent's response
        
        Args:
            agent_response: The agent's response
            
        Returns:
            User feedback
        """
        ...
    
    fn is_satisfied(self) -> Bool:
        """
        Check if user is satisfied with current state
        
        Returns:
            True if satisfied, False otherwise
        """
        ...
    
    fn reset(mut self):
        """Reset user state"""
        ...

struct UserProfile:
    """Profile information for a user"""
    var user_id: String
    var name: String
    var domain: String
    var preferences: Dict[String, String]
    var interaction_style: String  # "direct", "conversational", "technical"
    var patience_level: Int  # 1-10, how many turns before dissatisfaction
    var knowledge_level: String  # "novice", "intermediate", "expert"
    var goals: List[String]
    var constraints: List[String]
    
    fn __init__(out self):
        self.user_id = ""
        self.name = "User"
        self.domain = "general"
        self.preferences = Dict[String, String]()
        self.interaction_style = "conversational"
        self.patience_level = 5
        self.knowledge_level = "intermediate"
        self.goals = List[String]()
        self.constraints = List[String]()
    
    fn __init__(out self, user_id: String, name: String, domain: String):
        self.user_id = user_id
        self.name = name
        self.domain = domain
        self.preferences = Dict[String, String]()
        self.interaction_style = "conversational"
        self.patience_level = 5
        self.knowledge_level = "intermediate"
        self.goals = List[String]()
        self.constraints = List[String]()
    
    fn add_preference(mut self, key: String, value: String):
        """Add a user preference"""
        self.preferences[key] = value
    
    fn get_preference(self, key: String) -> String:
        """Get a user preference"""
        if key in self.preferences:
            return self.preferences[key]
        return ""
    
    fn add_goal(mut self, goal: String):
        """Add a user goal"""
        self.goals.append(goal)
    
    fn add_constraint(mut self, constraint: String):
        """Add a user constraint"""
        self.constraints.append(constraint)
    
    fn set_interaction_style(mut self, style: String):
        """Set interaction style"""
        self.interaction_style = style
    
    fn set_patience_level(mut self, level: Int):
        """Set patience level (1-10)"""
        if level >= 1 and level <= 10:
            self.patience_level = level
    
    fn set_knowledge_level(mut self, level: String):
        """Set knowledge level"""
        self.knowledge_level = level
    
    fn is_direct_style(self) -> Bool:
        """Check if user prefers direct communication"""
        return self.interaction_style == "direct"
    
    fn is_conversational_style(self) -> Bool:
        """Check if user prefers conversational style"""
        return self.interaction_style == "conversational"
    
    fn is_technical_style(self) -> Bool:
        """Check if user prefers technical communication"""
        return self.interaction_style == "technical"
    
    fn is_novice(self) -> Bool:
        """Check if user is a novice"""
        return self.knowledge_level == "novice"
    
    fn is_expert(self) -> Bool:
        """Check if user is an expert"""
        return self.knowledge_level == "expert"

struct BaseUser(User):
    """Basic implementation of User trait"""
    var profile: UserProfile
    var current_goal: String
    var turn_count: Int
    var satisfied: Bool
    var last_query: String
    var interaction_history: List[String]
    
    fn __init__(out self):
        self.profile = UserProfile()
        self.current_goal = ""
        self.turn_count = 0
        self.satisfied = False
        self.last_query = ""
        self.interaction_history = List[String]()
    
    fn __init__(out self, profile: UserProfile):
        self.profile = profile
        self.current_goal = ""
        self.turn_count = 0
        self.satisfied = False
        self.last_query = ""
        self.interaction_history = List[String]()
    
    fn get_id(self) -> String:
        """Get user identifier"""
        return self.profile.user_id
    
    fn get_profile(self) -> UserProfile:
        """Get user profile"""
        return self.profile
    
    fn generate_query(mut self, context: String) -> String:
        """
        Generate a user query based on context
        
        Args:
            context: Current context/state
            
        Returns:
            User query string
        """
        self.turn_count += 1
        
        # Simple query generation based on profile
        var query = ""
        
        if self.current_goal == "":
            query = "I need help with " + self.profile.domain
        else:
            query = "Can you help me with: " + self.current_goal
        
        self.last_query = query
        self.interaction_history.append("USER: " + query)
        
        return query
    
    fn provide_feedback(mut self, agent_response: String) -> String:
        """
        Provide feedback on agent's response
        
        Args:
            agent_response: The agent's response
            
        Returns:
            User feedback
        """
        self.interaction_history.append("AGENT: " + agent_response)
        
        # Simple feedback logic
        var feedback = ""
        
        if len(agent_response) < 10:
            feedback = "That response was too brief. Can you provide more details?"
        elif self.turn_count > self.profile.patience_level:
            feedback = "This is taking too long. Can we speed this up?"
            self.satisfied = False
        else:
            feedback = "Thank you, that's helpful."
            if "complete" in agent_response.lower() or "done" in agent_response.lower():
                self.satisfied = True
        
        self.interaction_history.append("FEEDBACK: " + feedback)
        return feedback
    
    fn is_satisfied(self) -> Bool:
        """
        Check if user is satisfied with current state
        
        Returns:
            True if satisfied, False otherwise
        """
        return self.satisfied
    
    fn reset(mut self):
        """Reset user state"""
        self.current_goal = ""
        self.turn_count = 0
        self.satisfied = False
        self.last_query = ""
        self.interaction_history = List[String]()
    
    fn set_goal(mut self, goal: String):
        """Set the current goal"""
        self.current_goal = goal
    
    fn get_turn_count(self) -> Int:
        """Get number of interaction turns"""
        return self.turn_count
    
    fn get_interaction_history(self) -> List[String]:
        """Get complete interaction history"""
        return self.interaction_history

fn create_default_user(user_id: String, domain: String) -> BaseUser:
    """
    Create a default user with standard configuration
    
    Args:
        user_id: User identifier
        domain: Domain of interest
        
    Returns:
        Configured BaseUser instance
    """
    var profile = UserProfile(user_id, "User", domain)
    profile.set_interaction_style("conversational")
    profile.set_patience_level(5)
    profile.set_knowledge_level("intermediate")
    
    return BaseUser(profile)
