# Telecom Service Issues - Pure Mojo Implementation
# Service issue task definitions for the telecom domain

from collections import Dict, List

# Overdue bill ID constant
alias OVERDUE_BILL_ID = "B1234321"


struct ServiceIssueTask:
    """Represents a service issue task."""
    var name: String
    var description: String
    var is_fixable: Bool
    
    fn __init__(out self, name: String, description: String, is_fixable: Bool):
        self.name = name
        self.description = description
        self.is_fixable = is_fixable


# Base task definitions
fn get_airplane_mode_on_task() -> ServiceIssueTask:
    """Get the airplane mode on task."""
    return ServiceIssueTask("airplane_mode_on", "Airplane mode is on.", True)


fn get_unseat_sim_card_task() -> ServiceIssueTask:
    """Get the unseat SIM card task."""
    return ServiceIssueTask("unseat_sim_card", "SIM card is unseated.", True)


fn get_lock_sim_card_pin_task() -> ServiceIssueTask:
    """Get the lock SIM card PIN task."""
    return ServiceIssueTask("lock_sim_card_pin", "SIM card is locked with a PIN", False)


fn get_break_apn_settings_task() -> ServiceIssueTask:
    """Get the break APN settings task."""
    return ServiceIssueTask("break_apn_settings", "APN settings are broken", True)


fn get_suspend_line_for_overdue_bill_task() -> ServiceIssueTask:
    """Get the suspend line for overdue bill task."""
    return ServiceIssueTask("overdue_bill_suspension", 
                            "Line is suspended for an overdue bill", True)


fn get_suspend_line_for_contract_end_task() -> ServiceIssueTask:
    """Get the suspend line for contract end task."""
    return ServiceIssueTask("contract_end_suspension", 
                            "Line is suspended for an overdue bill and a contract end", False)


struct ServiceIssuesManager:
    """Manages service issue tasks."""
    
    var domain: String
    var name: String
    var purpose: String
    var task_instructions: String
    var reason_for_call: String
    var known_info: String
    var ticket: String
    
    fn __init__(out self):
        """Initialize the service issues manager."""
        self.domain = "telecom"
        self.name = "service_issue"
        self.purpose = "Test resolution path: No Service/Connection Issues."
        self.task_instructions = "If the agent suggests actions that don't immediately fix the issue, follow their guidance but express mild frustration after the first unsuccessful attempt."
        self.reason_for_call = "Your phone has been showing 'No Service' for the past few hours."
        self.known_info = "You are {name} with phone number {phone_number}."
        self.ticket = "The user is experiencing issues with their phone service."
    
    fn get_all_tasks(self) -> List[ServiceIssueTask]:
        """Get all service issue tasks."""
        var tasks = List[ServiceIssueTask]()
        tasks.append(get_airplane_mode_on_task())
        tasks.append(get_unseat_sim_card_task())
        tasks.append(get_lock_sim_card_pin_task())
        tasks.append(get_break_apn_settings_task())
        tasks.append(get_suspend_line_for_overdue_bill_task())
        tasks.append(get_suspend_line_for_contract_end_task())
        return tasks
    
    fn get_fixable_tasks(self) -> List[ServiceIssueTask]:
        """Get all fixable service issue tasks."""
        var all_tasks = self.get_all_tasks()
        var fixable = List[ServiceIssueTask]()
        for i in range(len(all_tasks)):
            if all_tasks[i].is_fixable:
                fixable.append(all_tasks[i])
        return fixable


# Selection sets
struct AirplaneModeIssues:
    """Selection set for airplane mode issues."""
    var tasks: List[ServiceIssueTask]
    
    fn __init__(out self):
        self.tasks = List[ServiceIssueTask]()
        self.tasks.append(get_airplane_mode_on_task())


struct UnseatSimCardIssues:
    """Selection set for unseat SIM card issues."""
    var tasks: List[ServiceIssueTask]
    
    fn __init__(out self):
        self.tasks = List[ServiceIssueTask]()
        self.tasks.append(get_unseat_sim_card_task())


struct LockSimCardIssues:
    """Selection set for lock SIM card issues."""
    var tasks: List[ServiceIssueTask]
    
    fn __init__(out self):
        self.tasks = List[ServiceIssueTask]()
        self.tasks.append(get_lock_sim_card_pin_task())


struct BreakApnSettingsIssues:
    """Selection set for break APN settings issues."""
    var tasks: List[ServiceIssueTask]
    
    fn __init__(out self):
        self.tasks = List[ServiceIssueTask]()
        self.tasks.append(get_break_apn_settings_task())


struct SuspendLineIssues:
    """Selection set for suspend line issues."""
    var tasks: List[ServiceIssueTask]
    
    fn __init__(out self):
        self.tasks = List[ServiceIssueTask]()
        self.tasks.append(get_suspend_line_for_overdue_bill_task())
        self.tasks.append(get_suspend_line_for_contract_end_task())

