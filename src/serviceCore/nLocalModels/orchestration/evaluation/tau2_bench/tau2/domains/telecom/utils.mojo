# Telecom Domain Utilities - Pure Mojo Implementation
# Path constants for telecom domain data files

from sys import env

# Data directory paths
fn get_data_dir() -> String:
    """Get the base data directory path."""
    var repo_path = env.get("REPO_PATH", ".")
    return repo_path + "/data"

fn get_telecom_data_dir() -> String:
    """Get the telecom domain data directory path."""
    return get_data_dir() + "/tau2/domains/telecom"

fn get_telecom_db_path() -> String:
    """Get the path to the telecom domain database JSON file."""
    return get_telecom_data_dir() + "/db.json"

fn get_telecom_policy_path() -> String:
    """Get the path to the telecom domain policy markdown file."""
    return get_telecom_data_dir() + "/policy.md"

fn get_telecom_task_set_path() -> String:
    """Get the path to the telecom domain tasks JSON file."""
    return get_telecom_data_dir() + "/tasks.json"

