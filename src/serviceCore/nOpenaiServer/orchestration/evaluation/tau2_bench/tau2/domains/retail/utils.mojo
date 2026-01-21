# Retail Domain Utilities - Pure Mojo Implementation
# Path constants for retail domain data files

from sys import env

# Data directory paths
fn get_data_dir() -> String:
    """Get the base data directory path."""
    var repo_path = env.get("REPO_PATH", ".")
    return repo_path + "/data"

fn get_retail_data_dir() -> String:
    """Get the retail domain data directory path."""
    return get_data_dir() + "/tau2/domains/retail"

fn get_retail_db_path() -> String:
    """Get the path to the retail domain database JSON file."""
    return get_retail_data_dir() + "/db.json"

fn get_retail_policy_path() -> String:
    """Get the path to the retail domain policy markdown file."""
    return get_retail_data_dir() + "/policy.md"

fn get_retail_task_set_path() -> String:
    """Get the path to the retail domain tasks JSON file."""
    return get_retail_data_dir() + "/tasks.json"

