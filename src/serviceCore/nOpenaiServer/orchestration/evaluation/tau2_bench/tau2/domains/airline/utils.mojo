# Airline Domain Utilities - Pure Mojo Implementation
# Path constants for airline domain data files

from sys import env

# Data directory paths
fn get_data_dir() -> String:
    """Get the base data directory path."""
    var repo_path = env.get("REPO_PATH", ".")
    return repo_path + "/data"

fn get_airline_data_dir() -> String:
    """Get the airline domain data directory path."""
    return get_data_dir() + "/tau2/domains/airline"

fn get_airline_db_path() -> String:
    """Get the path to the airline domain database JSON file."""
    return get_airline_data_dir() + "/db.json"

fn get_airline_policy_path() -> String:
    """Get the path to the airline domain policy markdown file."""
    return get_airline_data_dir() + "/policy.md"

fn get_airline_task_set_path() -> String:
    """Get the path to the airline domain tasks JSON file."""
    return get_airline_data_dir() + "/tasks.json"

