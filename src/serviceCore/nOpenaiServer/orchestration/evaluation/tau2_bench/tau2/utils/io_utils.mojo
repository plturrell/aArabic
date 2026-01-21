# tau2/utils/io_utils.mojo
# Migrated from tau2/utils/io_utils.py

from pathlib import Path
from collections import Dict
from json import dumps as json_dumps, loads as json_loads

fn load_file(path: String) raises -> String:
    """
    Load the content of a file from a path based on the file extension.
    
    Args:
        path: The path to the file to load.
        
    Returns:
        The data loaded from the file as a string.
        For JSON/YAML/TOML, returns the raw content string.
        For TXT/MD, returns the text content.
    """
    # Determine file extension
    var path_str = path
    var ext = ""
    
    # Find the last dot to get extension
    for i in range(len(path_str) - 1, -1, -1):
        if path_str[i] == ".":
            ext = path_str[i:]
            break
    
    # Read file content
    var content: String
    with open(path_str, "r") as f:
        content = f.read()
    
    if ext == ".json" or ext == ".yaml" or ext == ".yml" or ext == ".toml":
        # Return JSON content as-is for now
        # TODO: Parse and return as structured data when JSON parser is available
        return content
    elif ext == ".txt" or ext == ".md":
        return content
    else:
        raise Error("Unsupported file extension: " + ext)

fn dump_file(path: String, data: String) raises:
    """
    Dump data content to a file based on the file extension.
    
    Args:
        path: The path to the file to dump the data to.
        data: The data string to dump to the file.
    """
    # Create parent directory if it doesn't exist
    # TODO: Implement directory creation when Path operations are available
    
    # Determine file extension
    var path_str = path
    var ext = ""
    
    # Find the last dot to get extension
    for i in range(len(path_str) - 1, -1, -1):
        if path_str[i] == ".":
            ext = path_str[i:]
            break
    
    # Write file content
    if ext == ".json" or ext == ".yaml" or ext == ".yml" or ext == ".toml":
        with open(path_str, "w") as f:
            _ = f.write(data)
    elif ext == ".txt" or ext == ".md":
        with open(path_str, "w") as f:
            _ = f.write(data)
    else:
        raise Error("Unsupported file extension: " + ext)

fn read_json(path: String) raises -> String:
    """
    Read a JSON file and return its content as a string.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        JSON content as string
    """
    return load_file(path)

fn write_json(path: String, data: String) raises:
    """
    Write a JSON string to a file.
    
    Args:
        path: Path to the JSON file
        data: JSON string to write
    """
    dump_file(path, data)

fn mkdir_if_not_exists(path: String) raises:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    # TODO: Implement when Path.mkdir() is available
    pass
