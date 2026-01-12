"""
Input validation utilities
"""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def validate_file_path(file_path: str, base_dir: Optional[str] = None) -> Path:
    """
    Validate and sanitize file paths to prevent directory traversal attacks
    
    Args:
        file_path: The file path to validate
        base_dir: Base directory to restrict paths to (optional)
    
    Returns:
        Validated Path object
    
    Raises:
        ValueError: If path is invalid or outside base_dir
    """
    # Resolve the path
    path = Path(file_path).resolve()
    
    # Check for directory traversal attempts
    if ".." in str(path):
        raise ValueError("Path contains directory traversal attempt")
    
    # If base_dir is provided, ensure path is within it
    if base_dir:
        base = Path(base_dir).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            raise ValueError(f"Path is outside allowed directory: {base_dir}")
    
    return path


def validate_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL string to validate
    
    Returns:
        True if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
    
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        raise ValueError("Value must be a string")
    
    # Remove null bytes and control characters
    sanitized = "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()

