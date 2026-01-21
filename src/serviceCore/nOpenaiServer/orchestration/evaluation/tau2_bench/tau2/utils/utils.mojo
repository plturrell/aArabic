# tau2/utils/utils.mojo
# Migrated from tau2/utils/utils.py

from time import now
from sys import env_get
from pathlib import Path
from collections import Dict
from json import dumps as json_dumps, loads as json_loads

# Data directory configuration
fn get_data_dir() -> String:
    """Get the data directory path from REPO_PATH environment variable."""
    var repo_path = env_get("REPO_PATH", "/Users/user/Documents/arabic_folder/src/serviceCore/nOpenaiServer/tools/toolorchestra")
    return String(repo_path) + "/evaluation/data_dir"

alias DATA_DIR: String = get_data_dir()

fn get_dict_hash(obj: String) raises -> String:
    """
    Generate a unique hash for a JSON string representation of a dict.
    Returns a hex string representation of the hash.
    
    Args:
        obj: JSON string representation of the dictionary
        
    Returns:
        Hex string hash
    """
    # TODO: Implement SHA256 hashing when crypto stdlib is available
    # For now, return a simple hash based on string length and content
    var hash_val: Int = 0
    for i in range(len(obj)):
        hash_val = (hash_val * 31 + ord(obj[i])) % 2147483647
    return String(hash_val)

fn get_now() -> String:
    """
    Returns the current date and time in ISO 8601 format.
    
    Returns:
        Current timestamp as ISO 8601 string
    """
    # Get current time in nanoseconds
    var current_ns = now()
    var seconds = current_ns // 1_000_000_000
    var nanos = current_ns % 1_000_000_000
    
    # Simple ISO 8601 format: YYYY-MM-DDTHH:MM:SS
    # For now, return seconds since epoch as placeholder
    # TODO: Implement proper datetime formatting when stdlib is available
    return String(seconds) + "." + String(nanos)

fn format_time(timestamp: Int) -> String:
    """
    Format the time in ISO 8601 format.
    
    Args:
        timestamp: Unix timestamp in seconds
        
    Returns:
        Formatted time string
    """
    return String(timestamp)

fn get_commit_hash() -> String:
    """
    Get the commit hash of the current directory.
    
    Returns:
        Git commit hash or "unknown"
    """
    # TODO: Implement git command execution when process execution is available
    return "unknown"

struct DictDiff:
    """Structure to represent differences between two dictionaries."""
    var added: String
    var removed: String
    var modified: String
    
    fn __init__(inout self):
        self.added = "{}"
        self.removed = "{}"
        self.modified = "{}"

fn show_dict_diff(dict1: String, dict2: String) raises -> DictDiff:
    """
    Show the difference between two dictionaries (as JSON strings).
    
    Args:
        dict1: First dictionary as JSON string
        dict2: Second dictionary as JSON string
        
    Returns:
        DictDiff structure with changes
    """
    # TODO: Implement deep diff when JSON parsing is fully available
    var diff = DictDiff()
    return diff
