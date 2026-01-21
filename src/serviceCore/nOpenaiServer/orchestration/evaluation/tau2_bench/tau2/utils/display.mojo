# display.mojo
# Migrated from display.py
# Display and formatting utilities for TAU2-Bench

from collections import List

struct DisplayConfig:
    """Configuration for display formatting"""
    var use_colors: Bool
    var max_width: Int
    var indent_size: Int
    var show_timestamps: Bool
    
    fn __init__(out self):
        self.use_colors = True
        self.max_width = 80
        self.indent_size = 2
        self.show_timestamps = True

fn format_header(text: String, width: Int = 80, char: String = "=") -> String:
    """
    Format a header with surrounding characters
    
    Args:
        text: Header text
        width: Total width of header
        char: Character to use for border
        
    Returns:
        Formatted header string
    """
    let padding = (width - len(text) - 2) // 2
    var header = char * width + "\n"
    header = header + char * padding + " " + text + " " + char * padding
    
    # Add extra char if width is odd
    if (width - len(text) - 2) % 2 != 0:
        header = header + char
    
    header = header + "\n" + char * width + "\n"
    return header

fn format_section(title: String, content: String, indent: Int = 0) -> String:
    """
    Format a section with title and content
    
    Args:
        title: Section title
        content: Section content
        indent: Indentation level
        
    Returns:
        Formatted section string
    """
    let indent_str = " " * indent
    var section = indent_str + title + ":\n"
    section = section + indent_str + "-" * len(title) + "\n"
    section = section + indent_str + content + "\n"
    return section

fn format_list(items: List[String], indent: Int = 0, numbered: Bool = False) -> String:
    """
    Format a list of items
    
    Args:
        items: List of strings to format
        indent: Indentation level
        numbered: Use numbers instead of bullets
        
    Returns:
        Formatted list string
    """
    let indent_str = " " * indent
    var result = ""
    
    for i in range(len(items)):
        if numbered:
            result = result + indent_str + str(i + 1) + ". " + items[i] + "\n"
        else:
            result = result + indent_str + "• " + items[i] + "\n"
    
    return result

fn format_table(headers: List[String], rows: List[List[String]], 
                col_widths: List[Int] = List[Int]()) -> String:
    """
    Format data as a table
    
    Args:
        headers: Column headers
        rows: Table rows
        col_widths: Column widths (auto-calculated if empty)
        
    Returns:
        Formatted table string
    """
    # Calculate column widths if not provided
    var widths = col_widths
    if len(widths) == 0:
        widths = List[Int]()
        for i in range(len(headers)):
            var max_width = len(headers[i])
            for j in range(len(rows)):
                if i < len(rows[j]):
                    max_width = max(max_width, len(rows[j][i]))
            widths.append(max_width + 2)
    
    # Format header
    var table = ""
    for i in range(len(headers)):
        let width = widths[i] if i < len(widths) else 10
        table = table + headers[i].ljust(width)
    table = table + "\n"
    
    # Header separator
    for i in range(len(headers)):
        let width = widths[i] if i < len(widths) else 10
        table = table + "-" * width
    table = table + "\n"
    
    # Format rows
    for row_idx in range(len(rows)):
        let row = rows[row_idx]
        for col_idx in range(len(row)):
            let width = widths[col_idx] if col_idx < len(widths) else 10
            table = table + row[col_idx].ljust(width)
        table = table + "\n"
    
    return table

fn format_key_value(key: String, value: String, width: Int = 40) -> String:
    """
    Format key-value pair
    
    Args:
        key: Key string
        value: Value string
        width: Total width
        
    Returns:
        Formatted key-value string
    """
    let dots = "." * max(1, width - len(key) - len(value) - 2)
    return key + " " + dots + " " + value + "\n"

fn format_progress_bar(current: Int, total: Int, width: Int = 50, 
                       show_percentage: Bool = True) -> String:
    """
    Format a progress bar
    
    Args:
        current: Current progress value
        total: Total value
        width: Bar width in characters
        show_percentage: Show percentage text
        
    Returns:
        Formatted progress bar string
    """
    let percentage = Float32(current) / Float32(total) if total > 0 else 0.0
    let filled = Int(percentage * Float32(width))
    let empty = width - filled
    
    var bar = "[" + "=" * filled + " " * empty + "]"
    
    if show_percentage:
        bar = bar + " " + str(Int(percentage * 100.0)) + "%"
    
    return bar

fn format_duration(milliseconds: Int) -> String:
    """
    Format duration in milliseconds to human-readable string
    
    Args:
        milliseconds: Duration in milliseconds
        
    Returns:
        Formatted duration string
    """
    if milliseconds < 1000:
        return str(milliseconds) + "ms"
    
    let seconds = milliseconds // 1000
    let ms = milliseconds % 1000
    
    if seconds < 60:
        return str(seconds) + "s " + str(ms) + "ms"
    
    let minutes = seconds // 60
    let secs = seconds % 60
    
    if minutes < 60:
        return str(minutes) + "m " + str(secs) + "s"
    
    let hours = minutes // 60
    let mins = minutes % 60
    
    return str(hours) + "h " + str(mins) + "m"

fn format_size(bytes: Int) -> String:
    """
    Format byte size to human-readable string
    
    Args:
        bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if bytes < 1024:
        return str(bytes) + "B"
    
    let kb = Float32(bytes) / 1024.0
    if kb < 1024.0:
        return str(Int(kb)) + "KB"
    
    let mb = kb / 1024.0
    if mb < 1024.0:
        return str(Int(mb)) + "MB"
    
    let gb = mb / 1024.0
    return str(Int(gb)) + "GB"

fn truncate_string(text: String, max_length: Int, suffix: String = "...") -> String:
    """
    Truncate string to maximum length
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    let truncate_at = max_length - len(suffix)
    if truncate_at <= 0:
        return suffix[:max_length]
    
    return text[:truncate_at] + suffix

fn wrap_text(text: String, width: Int = 80, indent: Int = 0) -> String:
    """
    Wrap text to specified width
    
    Args:
        text: Text to wrap
        width: Maximum line width
        indent: Indentation for wrapped lines
        
    Returns:
        Wrapped text
    """
    # Simple word wrapping implementation
    let words = text.split()
    var lines = List[String]()
    var current_line = ""
    let indent_str = " " * indent
    
    for i in range(len(words)):
        let word = words[i]
        let test_line = current_line + " " + word if current_line != "" else word
        
        if len(test_line) + indent <= width:
            current_line = test_line
        else:
            if current_line != "":
                lines.append(indent_str + current_line)
            current_line = word
    
    if current_line != "":
        lines.append(indent_str + current_line)
    
    var result = ""
    for i in range(len(lines)):
        result = result + lines[i] + "\n"
    
    return result

fn format_json_pretty(data: String, indent: Int = 2) -> String:
    """
    Format JSON string with pretty printing
    
    Args:
        data: JSON string
        indent: Indentation size
        
    Returns:
        Pretty-printed JSON string
    """
    # Simplified JSON formatting
    # In production, would use proper JSON parser
    return data

fn clear_screen() -> String:
    """
    Generate escape sequence to clear screen
    
    Returns:
        Clear screen escape sequence
    """
    return "\033[2J\033[H"

fn format_status(status: String, success: Bool) -> String:
    """
    Format status message with indicator
    
    Args:
        status: Status text
        success: Whether status indicates success
        
    Returns:
        Formatted status string
    """
    let indicator = "✓" if success else "✗"
    return indicator + " " + status
