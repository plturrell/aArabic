# Telecom Mobile Data Issues - Pure Mojo Implementation
# Mobile data issue task definitions for the telecom domain

from collections import Dict, List

# Expected internet speed constants
alias EXPECTED_INTERNET_SPEED: Int = 200
alias EXPECTED_INTERNET_DESC = "excellent"
alias INVALID_INTERNET_DESC = "poor, fair or good"


struct MobileDataIssueTask:
    """Represents a mobile data issue task."""
    var name: String
    var description: String
    var is_fixable: Bool
    
    fn __init__(out self, name: String, description: String, is_fixable: Bool):
        self.name = name
        self.description = description
        self.is_fixable = is_fixable


# Base task definitions
fn get_user_abroad_roaming_enabled_off_task() -> MobileDataIssueTask:
    """Get the user abroad roaming enabled off task."""
    return MobileDataIssueTask("user_abroad_roaming_enabled_off", 
                               "User is abroad and roaming is off", True)


fn get_user_abroad_roaming_disabled_on_task() -> MobileDataIssueTask:
    """Get the user abroad roaming disabled on task."""
    return MobileDataIssueTask("user_abroad_roaming_disabled_on", 
                               "User is abroad and roaming is off", True)


fn get_user_abroad_roaming_disabled_off_task() -> MobileDataIssueTask:
    """Get the user abroad roaming disabled off task."""
    return MobileDataIssueTask("user_abroad_roaming_disabled_off", 
                               "User is abroad and roaming is off", True)


fn get_data_mode_off_task() -> MobileDataIssueTask:
    """Get the data mode off task."""
    return MobileDataIssueTask("data_mode_off", "Data mode is off", True)


fn get_data_saver_mode_on_task() -> MobileDataIssueTask:
    """Get the data saver mode on task."""
    return MobileDataIssueTask("data_saver_mode_on", "Data saver mode is on", True)


fn get_bad_network_preference_task() -> MobileDataIssueTask:
    """Get the bad network preference task."""
    return MobileDataIssueTask("bad_network_preference", "Bad network preference", True)


fn get_bad_vpn_task() -> MobileDataIssueTask:
    """Get the bad VPN task."""
    return MobileDataIssueTask("bad_vpn", "Bad vpn", True)


fn get_data_usage_exceeded_task() -> MobileDataIssueTask:
    """Get the data usage exceeded task."""
    return MobileDataIssueTask("data_usage_exceeded", "Data usage exceeded", True)


fn get_data_usage_exceeded_no_refuel_task() -> MobileDataIssueTask:
    """Get the data usage exceeded no refuel task."""
    return MobileDataIssueTask("data_usage_exceeded_no_refuel", 
                               "Data usage exceeded", False)


struct MobileDataIssuesManager:
    """Manages mobile data issue tasks."""
    
    var domain: String
    var name: String
    var purpose: String
    var task_instructions: String
    var reason_for_call: String
    var known_info: String
    var ticket: String
    
    fn __init__(out self):
        """Initialize the mobile data issues manager."""
        self.domain = "telecom"
        self.name = "mobile_data_issue"
        self.purpose = "Test resolution path: Mobile Data/Slow Internet Issues."
        self.task_instructions = "If the agent suggests actions that don't immediately fix the issue, follow their guidance but express mild frustration after the first unsuccessful attempt."
        self.reason_for_call = "Your mobile data is not working properly."
        self.known_info = "You are {name} with phone number {phone_number}."
        self.ticket = "The user is experiencing issues with their mobile data."
    
    fn get_all_tasks(self) -> List[MobileDataIssueTask]:
        """Get all mobile data issue tasks."""
        var tasks = List[MobileDataIssueTask]()
        tasks.append(get_user_abroad_roaming_enabled_off_task())
        tasks.append(get_user_abroad_roaming_disabled_on_task())
        tasks.append(get_user_abroad_roaming_disabled_off_task())
        tasks.append(get_data_mode_off_task())
        tasks.append(get_data_saver_mode_on_task())
        tasks.append(get_bad_network_preference_task())
        tasks.append(get_bad_vpn_task())
        tasks.append(get_data_usage_exceeded_task())
        tasks.append(get_data_usage_exceeded_no_refuel_task())
        return tasks


# Selection sets
struct RoamingIssues:
    """Selection set for roaming issues."""
    var tasks: List[MobileDataIssueTask]
    
    fn __init__(out self):
        self.tasks = List[MobileDataIssueTask]()
        self.tasks.append(get_user_abroad_roaming_enabled_off_task())
        self.tasks.append(get_user_abroad_roaming_disabled_on_task())
        self.tasks.append(get_user_abroad_roaming_disabled_off_task())


struct DataModeIssues:
    """Selection set for data mode issues."""
    var tasks: List[MobileDataIssueTask]
    
    fn __init__(out self):
        self.tasks = List[MobileDataIssueTask]()
        self.tasks.append(get_data_mode_off_task())


struct DataUsageExceededIssues:
    """Selection set for data usage exceeded issues."""
    var tasks: List[MobileDataIssueTask]
    
    fn __init__(out self):
        self.tasks = List[MobileDataIssueTask]()
        self.tasks.append(get_data_usage_exceeded_task())
        self.tasks.append(get_data_usage_exceeded_no_refuel_task())


struct NetworkPreferenceIssues:
    """Selection set for network preference issues."""
    var tasks: List[MobileDataIssueTask]
    
    fn __init__(out self):
        self.tasks = List[MobileDataIssueTask]()
        self.tasks.append(get_bad_network_preference_task())

