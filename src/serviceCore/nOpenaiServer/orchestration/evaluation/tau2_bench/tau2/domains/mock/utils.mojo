# Mock Domain Utilities - Pure Mojo Implementation
# Path constants for mock domain data files

from sys import env

# Data directory paths
fn get_data_dir() -> String:
    """Get the base data directory path."""
    var repo_path = env.get("REPO_PATH", ".")
    return repo_path + "/data"

fn get_mock_data_dir() -> String:
    """Get the mock domain data directory path."""
    return get_data_dir() + "/tau2/domains/mock"

fn get_mock_db_path() -> String:
    """Get the path to the mock domain database JSON file."""
    return get_mock_data_dir() + "/db.json"

fn get_mock_policy_path() -> String:
    """Get the path to the mock domain policy markdown file."""
    return get_mock_data_dir() + "/policy.md"

fn get_mock_policy_solo_path() -> String:
    """Get the path to the mock domain solo policy markdown file."""
    return get_mock_data_dir() + "/policy_solo.md"

fn get_mock_task_set_path() -> String:
    """Get the path to the mock domain tasks JSON file."""
    return get_mock_data_dir() + "/tasks.json"

