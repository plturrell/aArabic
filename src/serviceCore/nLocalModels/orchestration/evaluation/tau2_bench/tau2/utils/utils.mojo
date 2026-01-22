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
        Hex string hash (using FNV-1a algorithm)
    """
    # FNV-1a 64-bit hash
    var hash_val: UInt64 = 14695981039346656037  # FNV offset basis
    let fnv_prime: UInt64 = 1099511628211

    let bytes = obj.as_bytes()
    for i in range(len(bytes)):
        hash_val ^= UInt64(bytes[i])
        hash_val *= fnv_prime

    # Convert to hex string
    return _uint64_to_hex(hash_val)


fn _uint64_to_hex(value: UInt64) -> String:
    """Convert UInt64 to hexadecimal string"""
    let hex_chars = "0123456789abcdef"
    var result = String()
    var v = value

    # Handle zero case
    if v == 0:
        return "0"

    # Extract hex digits in reverse order
    var digits = List[Int]()
    while v > 0:
        digits.append(Int(v % 16))
        v //= 16

    # Build string in correct order
    for i in range(len(digits) - 1, -1, -1):
        result += hex_chars[digits[i]]

    return result


fn get_now() -> String:
    """
    Returns the current date and time in ISO 8601 format.

    Returns:
        Current timestamp as ISO 8601 string
    """
    from python import Python

    try:
        let datetime = Python.import_module("datetime")
        let now_obj = datetime.datetime.now(datetime.timezone.utc)
        return String(now_obj.isoformat())
    except:
        # Fallback to epoch seconds
        var current_ns = now()
        var seconds = current_ns // 1_000_000_000
        return String(seconds)


fn format_time(timestamp: Int) -> String:
    """
    Format the time in ISO 8601 format.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        Formatted time string
    """
    from python import Python

    try:
        let datetime = Python.import_module("datetime")
        let dt = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
        return String(dt.isoformat())
    except:
        return String(timestamp)


fn get_commit_hash() -> String:
    """
    Get the commit hash of the current directory.

    Returns:
        Git commit hash or "unknown"
    """
    from python import Python

    try:
        let subprocess = Python.import_module("subprocess")
        let result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if int(result.returncode) == 0:
            return String(result.stdout.strip())
    except:
        pass

    return "unknown"


fn get_git_branch() -> String:
    """Get the current git branch name."""
    from python import Python

    try:
        let subprocess = Python.import_module("subprocess")
        let result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if int(result.returncode) == 0:
            return String(result.stdout.strip())
    except:
        pass

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

    fn has_changes(self) -> Bool:
        """Check if there are any differences"""
        return self.added != "{}" or self.removed != "{}" or self.modified != "{}"


fn show_dict_diff(dict1: String, dict2: String) raises -> DictDiff:
    """
    Show the difference between two dictionaries (as JSON strings).

    Args:
        dict1: First dictionary as JSON string
        dict2: Second dictionary as JSON string

    Returns:
        DictDiff structure with changes
    """
    from python import Python

    var diff = DictDiff()

    try:
        let json_mod = Python.import_module("json")

        let d1 = json_mod.loads(dict1)
        let d2 = json_mod.loads(dict2)

        # Find added keys (in d2 but not in d1)
        var added_dict = Python.dict()
        for key in d2.keys():
            if key not in d1:
                added_dict[key] = d2[key]
        diff.added = String(json_mod.dumps(added_dict))

        # Find removed keys (in d1 but not in d2)
        var removed_dict = Python.dict()
        for key in d1.keys():
            if key not in d2:
                removed_dict[key] = d1[key]
        diff.removed = String(json_mod.dumps(removed_dict))

        # Find modified keys (in both but different values)
        var modified_dict = Python.dict()
        for key in d1.keys():
            if key in d2:
                if str(d1[key]) != str(d2[key]):
                    var mod_entry = Python.dict()
                    mod_entry["old"] = d1[key]
                    mod_entry["new"] = d2[key]
                    modified_dict[key] = mod_entry
        diff.modified = String(json_mod.dumps(modified_dict))

    except e:
        pass

    return diff
