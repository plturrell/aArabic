# user_simulator.mojo
# Migrated from user_simulator.py
# LLM-based user simulator for TAU2-Bench

from collections import Dict, List
from user.base import User, UserProfile, BaseUser
from data_model.tasks import TaskDefinition, TaskInstance

struct UserSimulatorConfig:
    """Configuration for user simulator"""
    var model: String
    var temperature: Float32
    var max_tokens: Int
    var variability: Float32  # 0.0-1.0, how much to vary responses
    var realism_level: String  # "low", "medium", "high"
    
    fn __init__(out self):
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.8
        self.max_tokens = 512
        self.variability = 0.5
        self.realism_level = "medium"
    
    fn __init__(out self, model: String, temperature: Float32):
        self.model = model
        self.temperature = temperature
        self.max_tokens = 512
        self.variability = 0.5
        self.realism_level = "medium"

struct UserSimulator(User):
    """
    LLM-based user simulator that generates realistic user behavior
    """
    var profile: UserProfile
    var config: UserSimulatorConfig
    var task_instance: TaskInstance
    var conversation_history: List[String]
    var satisfaction_score: Float32
    var turn_count: Int
    var max_turns: Int
    var current_intent: String
    var emotional_state: String  # "neutral", "frustrated", "satisfied", "confused"
    
    fn __init__(out self):
        self.profile = UserProfile()
        self.config = UserSimulatorConfig()
        self.task_instance = TaskInstance()
        self.conversation_history = List[String]()
        self.satisfaction_score = 0.5
        self.turn_count = 0
        self.max_turns = 20
        self.current_intent = ""
        self.emotional_state = "neutral"
    
    fn __init__(out self, profile: UserProfile, config: UserSimulatorConfig, task: TaskInstance):
        self.profile = profile
        self.config = config
        self.task_instance = task
        self.conversation_history = List[String]()
        self.satisfaction_score = 0.5
        self.turn_count = 0
        self.max_turns = 20
        self.current_intent = ""
        self.emotional_state = "neutral"
    
    fn get_id(self) -> String:
        """Get user identifier"""
        return self.profile.user_id
    
    fn get_profile(self) -> UserProfile:
        """Get user profile"""
        return self.profile
    
    fn _build_user_persona_prompt(self) -> String:
        """Build a prompt describing the user persona"""
        var prompt = "You are simulating a user with the following characteristics:\n"
        prompt = prompt + "- Name: " + self.profile.name + "\n"
        prompt = prompt + "- Domain: " + self.profile.domain + "\n"
        prompt = prompt + "- Interaction style: " + self.profile.interaction_style + "\n"
        prompt = prompt + "- Knowledge level: " + self.profile.knowledge_level + "\n"
        prompt = prompt + "- Patience level: " + str(self.profile.patience_level) + "/10\n"
        prompt = prompt + "- Current emotional state: " + self.emotional_state + "\n"
        
        # Add task context
        prompt = prompt + "\nYou are trying to accomplish: " + self.task_instance.task_def.description + "\n"
        
        return prompt
    
    fn _update_emotional_state(mut self, agent_response: String):
        """Update emotional state based on interaction"""
        # Simple heuristic-based emotion update
        if self.turn_count > self.profile.patience_level:
            self.emotional_state = "frustrated"
            self.satisfaction_score = max(0.0, self.satisfaction_score - 0.2)
        elif "success" in agent_response.lower() or "complete" in agent_response.lower():
            self.emotional_state = "satisfied"
            self.satisfaction_score = min(1.0, self.satisfaction_score + 0.3)
        elif "error" in agent_response.lower() or "cannot" in agent_response.lower():
            self.emotional_state = "confused"
            self.satisfaction_score = max(0.0, self.satisfaction_score - 0.1)
        else:
            self.emotional_state = "neutral"
            self.satisfaction_score = min(1.0, self.satisfaction_score + 0.05)
    
    fn _call_llm_for_query(self, context: String) -> String:
        """Call LLM to generate user query"""
        # TODO: Integrate with actual LLM service
        # For now, return a template-based response
        
        var query = ""
        
        if self.turn_count == 0:
            # Initial query
            query = "Hi, I need help with " + self.task_instance.task_def.name
        elif self.emotional_state == "frustrated":
            query = "This is taking too long. Can you help me faster?"
        elif self.emotional_state == "confused":
            query = "I don't understand. Can you explain that differently?"
        else:
            query = "What should I do next?"
        
        return query
    
    fn _call_llm_for_feedback(self, agent_response: String) -> String:
        """Call LLM to generate user feedback"""
        # TODO: Integrate with actual LLM service
        # For now, return template-based feedback
        
        var feedback = ""
        
        if self.emotional_state == "satisfied":
            feedback = "Great, thank you! That helps."
        elif self.emotional_state == "frustrated":
            feedback = "I'm getting frustrated. This isn't working well."
        elif self.emotional_state == "confused":
            feedback = "I'm still confused about what to do."
        else:
            feedback = "Okay, I'll try that."
        
        return feedback
    
    fn generate_query(mut self, context: String) -> String:
        """
        Generate a user query based on current context
        
        Args:
            context: Current context/state
            
        Returns:
            User query string
        """
        self.turn_count += 1
        
        # Build persona and context
        let persona = self._build_user_persona_prompt()
        let full_context = persona + "\nCurrent context: " + context + "\n"
        
        # Generate query
        let query = self._call_llm_for_query(full_context)
        
        # Record in history
        self.conversation_history.append("USER: " + query)
        
        return query
    
    fn provide_feedback(mut self, agent_response: String) -> String:
        """
        Provide feedback on agent's response
        
        Args:
            agent_response: The agent's response
            
        Returns:
            User feedback
        """
        # Record agent response
        self.conversation_history.append("AGENT: " + agent_response)
        
        # Update emotional state
        self._update_emotional_state(agent_response)
        
        # Generate feedback
        let feedback = self._call_llm_for_feedback(agent_response)
        
        # Record feedback
        self.conversation_history.append("FEEDBACK: " + feedback)
        
        return feedback
    
    fn is_satisfied(self) -> Bool:
        """
        Check if user is satisfied with current state
        
        Returns:
            True if satisfied, False otherwise
        """
        return self.satisfaction_score >= 0.8 and self.emotional_state == "satisfied"
    
    fn reset(mut self):
        """Reset user simulator state"""
        self.conversation_history = List[String]()
        self.satisfaction_score = 0.5
        self.turn_count = 0
        self.current_intent = ""
        self.emotional_state = "neutral"
    
    fn get_satisfaction_score(self) -> Float32:
        """Get current satisfaction score (0.0-1.0)"""
        return self.satisfaction_score
    
    fn get_emotional_state(self) -> String:
        """Get current emotional state"""
        return self.emotional_state
    
    fn get_turn_count(self) -> Int:
        """Get number of interaction turns"""
        return self.turn_count
    
    fn set_max_turns(mut self, max_turns: Int):
        """Set maximum number of turns"""
        self.max_turns = max_turns
    
    fn has_exceeded_max_turns(self) -> Bool:
        """Check if maximum turns exceeded"""
        return self.turn_count >= self.max_turns

fn create_user_simulator(
    user_id: String,
    domain: String,
    task: TaskInstance,
    model: String = "gpt-3.5-turbo"
) -> UserSimulator:
    """
    Factory function to create a user simulator
    
    Args:
        user_id: User identifier
        domain: Domain of the task
        task: Task instance to accomplish
        model: LLM model to use for simulation
        
    Returns:
        Configured UserSimulator instance
    """
    var profile = UserProfile(user_id, "SimulatedUser", domain)
    profile.set_interaction_style("conversational")
    profile.set_patience_level(5)
    profile.set_knowledge_level("intermediate")
    
    var config = UserSimulatorConfig(model, 0.8)
    config.realism_level = "medium"
    config.variability = 0.5
    
    return UserSimulator(profile, config, task)

fn create_realistic_user_simulator(
    user_id: String,
    domain: String,
    task: TaskInstance,
    interaction_style: String,
    knowledge_level: String,
    patience: Int
) -> UserSimulator:
    """
    Create a user simulator with specific behavioral characteristics
    
    Args:
        user_id: User identifier
        domain: Domain of the task
        task: Task instance to accomplish
        interaction_style: User's preferred interaction style
        knowledge_level: User's knowledge level
        patience: User's patience level (1-10)
        
    Returns:
        Configured UserSimulator instance
    """
    var profile = UserProfile(user_id, "SimulatedUser", domain)
    profile.set_interaction_style(interaction_style)
    profile.set_patience_level(patience)
    profile.set_knowledge_level(knowledge_level)
    
    var config = UserSimulatorConfig("gpt-4", 0.9)
    config.realism_level = "high"
    config.variability = 0.7
    
    return UserSimulator(profile, config, task)
