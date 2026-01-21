# Telecom Tasks Constants - Pure Mojo Implementation
# Constants for telecom task definitions

# Tool call information check message
alias TOOL_CALL_INFO_CHECK = "If the tool call does not return updated status information, you might need to perform another tool call to get the updated status."

# Tool call grounding message
alias TOOL_CALL_GROUNDING = """
Whenever the agent asks you about your device, always ground your responses on the results of tool calls. 
For example: If the agent asks what the status bar shows, always ground your response on the results of the `get_status_bar` tool call. If the agent asks if you are able to send an MMS message, always ground your response on the results of the `can_send_mms` tool call.
Never make up the results of tool calls, always ground your responses on the results of tool calls.
If you are unsure about whether an action is necessary, always ask the agent for clarification.
"""

# Persona 1 - Easy difficulty
alias PERSONA_1 = """
As a 41-year-old office administrator, you use your cellphone daily for both work and personal tasks. While you're familiar with common phone functions, you wouldn't call yourself a tech enthusiast.

Your technical skills are average - you handle standard smartphone features like calls, texts, email, and basic apps with ease. You understand the fundamental settings, but prefer clear, step-by-step guidance when trying something new.

In interactions, you're naturally friendly and patient. When receiving help, you listen attentively and aren't afraid to ask questions. You make sure to confirm your understanding and provide detailed feedback on each instruction you receive.
"""

# Persona 2 - Hard difficulty
alias PERSONA_2 = """
At 64 years old, you're a retired librarian who keeps your phone use simple - mainly for calls, texts, and capturing photos of your grandchildren. Technology in general makes you feel uneasy and overwhelmed.

Your technical knowledge is quite limited. Step-by-step instructions often confuse you, and technical terms like "VPN" or "APN" might as well be a foreign language. You only share information when specifically asked.

When dealing with technology, you tend to get flustered quickly. You need constant reassurance and often interrupt with anxious questions. Simple requests like "reboot the phone" can trigger worries about losing precious photos.
"""


struct Personas:
    """Container for persona definitions."""
    
    fn __init__(out self):
        pass
    
    fn get_persona(self, name: String) -> String:
        """Get a persona by name."""
        if name == "Easy":
            return PERSONA_1
        elif name == "Hard":
            return PERSONA_2
        else:
            return ""
    
    fn get_persona_names(self) -> List[String]:
        """Get all persona names."""
        var names = List[String]()
        names.append("None")
        names.append("Easy")
        names.append("Hard")
        return names

