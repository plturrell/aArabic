"""
String operations for Lean4 stdlib.
"""

from collections import List


fn lean_string(s: String) -> String:
    """Create a Lean string."""
    return s


fn string_append(a: String, b: String) -> String:
    """Append two strings."""
    return a + b


fn string_length(s: String) -> Int:
    """Get string length in characters."""
    return len(s)


fn string_push(s: String, c: String) -> String:
    """Push a character onto a string."""
    return s + c


fn string_is_empty(s: String) -> Bool:
    """Check if string is empty."""
    return len(s) == 0


fn string_front(s: String) -> Optional[String]:
    """Get the first character."""
    if len(s) > 0:
        return s[0]
    return None


fn string_drop(s: String, n: Int) -> String:
    """Drop first n characters."""
    if n >= len(s):
        return ""
    return s[n:]


fn string_take(s: String, n: Int) -> String:
    """Take first n characters."""
    if n >= len(s):
        return s
    return s[:n]


fn string_get(s: String, i: Int) -> Optional[String]:
    """Get character at index."""
    if i >= 0 and i < len(s):
        return s[i]
    return None


fn string_split(s: String, sep: String) -> List[String]:
    """Split string by separator."""
    var result = List[String]()
    var current = String("")
    var sep_len = len(sep)
    var i = 0
    while i < len(s):
        var match = True
        if i + sep_len <= len(s):
            for j in range(sep_len):
                if s[i + j] != sep[j]:
                    match = False
                    break
            if match:
                result.append(current)
                current = ""
                i += sep_len
                continue
        current += s[i]
        i += 1
    result.append(current)
    return result


fn string_join(parts: List[String], sep: String) -> String:
    """Join strings with separator."""
    var result = String("")
    for i in range(len(parts)):
        if i > 0:
            result += sep
        result += parts[i]
    return result


fn string_trim(s: String) -> String:
    """Trim whitespace from both ends."""
    var start = 0
    var end = len(s)
    while start < end:
        var c = s[start]
        if c != " " and c != "\t" and c != "\n" and c != "\r":
            break
        start += 1
    while end > start:
        var c = s[end - 1]
        if c != " " and c != "\t" and c != "\n" and c != "\r":
            break
        end -= 1
    return s[start:end]


fn string_to_lower(s: String) -> String:
    """Convert to lowercase."""
    return s.lower()


fn string_to_upper(s: String) -> String:
    """Convert to uppercase."""
    return s.upper()


fn string_contains(s: String, sub: String) -> Bool:
    """Check if string contains substring."""
    return sub in s


fn string_starts_with(s: String, prefix: String) -> Bool:
    """Check if string starts with prefix."""
    if len(prefix) > len(s):
        return False
    return s[:len(prefix)] == prefix


fn string_ends_with(s: String, suffix: String) -> Bool:
    """Check if string ends with suffix."""
    if len(suffix) > len(s):
        return False
    return s[len(s) - len(suffix):] == suffix
