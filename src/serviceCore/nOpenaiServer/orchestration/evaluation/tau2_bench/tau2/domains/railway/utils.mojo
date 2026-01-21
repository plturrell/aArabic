# Railway Domain Utilities - Pure Mojo Implementation
# Path constants for railway domain data files

from sys import env

# Data directory paths
fn get_data_dir() -> String:
    """Get the base data directory path."""
    var repo_path = env.get("REPO_PATH", ".")
    return repo_path + "/data"

fn get_railway_data_dir() -> String:
    """Get the railway domain data directory path."""
    return get_data_dir() + "/tau2/domains/railway"

fn get_railway_db_path() -> String:
    """Get the path to the railway domain database JSON file."""
    return get_railway_data_dir() + "/db.json"

fn get_railway_policy_path() -> String:
    """Get the path to the railway domain policy markdown file."""
    return get_railway_data_dir() + "/policy.md"

fn get_railway_task_set_path() -> String:
    """Get the path to the railway domain tasks JSON file."""
    return get_railway_data_dir() + "/tasks.json"

