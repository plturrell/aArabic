# String - Advanced String Operations
# Day 32: Comprehensive string manipulation and processing

from builtin import Int, Bool, String
from collections.list import List


struct StringBuilder:
    """Efficient string builder for constructing strings incrementally.
    
    StringBuilder provides a mutable buffer for building strings efficiently,
    avoiding the overhead of repeated string concatenation.
    
    Examples:
        ```mojo
        var sb = StringBuilder()
        sb.append("Hello")
        sb.append(" ")
        sb.append("World")
        let result = sb.to_string()  # "Hello World"
        ```
    """
    
    var buffer: List[String]
    var length: Int
    
    fn __init__(inout self):
        """Initialize an empty string builder."""
        self.buffer = List[String]()
        self.length = 0
    
    fn __init__(inout self, initial: String):
        """Initialize with an initial string.
        
        Args:
            initial: Initial string content
        """
        self.buffer = List[String]()
        self.length = 0
        self.append(initial)
    
    fn append(inout self, text: String):
        """Append a string to the builder.
        
        Args:
            text: String to append
        """
        self.buffer.append(text)
        self.length += len(text)
    
    fn append_char(inout self, ch: String):
        """Append a single character.
        
        Args:
            ch: Character to append (single-char string)
        """
        self.buffer.append(ch)
        self.length += 1
    
    fn append_int(inout self, value: Int):
        """Append an integer as a string.
        
        Args:
            value: Integer to append
        """
        self.append(str(value))
    
    fn append_line(inout self, text: String):
        """Append a string followed by a newline.
        
        Args:
            text: String to append
        """
        self.append(text)
        self.append("\n")
    
    fn to_string(self) -> String:
        """Convert the builder to a string.
        
        Returns:
            Concatenated string
        """
        var result = ""
        for i in range(len(self.buffer)):
            result += self.buffer[i]
        return result
    
    fn clear(inout self):
        """Clear the builder's contents."""
        self.buffer = List[String]()
        self.length = 0
    
    fn size(self) -> Int:
        """Get the current length.
        
        Returns:
            Total character count
        """
        return self.length
    
    fn __str__(self) -> String:
        """Return string representation."""
        return self.to_string()


struct StringView:
    """Non-owning view into a string (substring without copying).
    
    StringView provides a lightweight way to reference parts of strings
    without allocating new memory.
    """
    
    var data: String
    var start: Int
    var length: Int
    
    fn __init__(inout self, text: String):
        """Create a view of the entire string.
        
        Args:
            text: String to view
        """
        self.data = text
        self.start = 0
        self.length = len(text)
    
    fn __init__(inout self, text: String, start: Int, length: Int):
        """Create a view of a substring.
        
        Args:
            text: Source string
            start: Starting index
            length: Length of view
        """
        self.data = text
        self.start = start
        self.length = length
    
    fn to_string(self) -> String:
        """Convert view to an actual string.
        
        Returns:
            String copy of the viewed portion
        """
        return self.data[self.start:self.start + self.length]
    
    fn __len__(self) -> Int:
        """Get the length of the view."""
        return self.length
    
    fn __str__(self) -> String:
        """Return string representation."""
        return self.to_string()


# String utility functions

fn str_len(text: String) -> Int:
    """Get the length of a string.
    
    Args:
        text: Input string
    
    Returns:
        Number of characters
    """
    return len(text)


fn str_is_empty(text: String) -> Bool:
    """Check if a string is empty.
    
    Args:
        text: Input string
    
    Returns:
        True if empty, False otherwise
    """
    return len(text) == 0


fn str_contains(text: String, substring: String) -> Bool:
    """Check if string contains a substring.
    
    Args:
        text: String to search in
        substring: Substring to find
    
    Returns:
        True if substring is found, False otherwise
    """
    return substring in text


fn str_starts_with(text: String, prefix: String) -> Bool:
    """Check if string starts with a prefix.
    
    Args:
        text: String to check
        prefix: Prefix to look for
    
    Returns:
        True if string starts with prefix, False otherwise
    """
    if len(prefix) > len(text):
        return False
    return text[:len(prefix)] == prefix


fn str_ends_with(text: String, suffix: String) -> Bool:
    """Check if string ends with a suffix.
    
    Args:
        text: String to check
        suffix: Suffix to look for
    
    Returns:
        True if string ends with suffix, False otherwise
    """
    if len(suffix) > len(text):
        return False
    return text[len(text) - len(suffix):] == suffix


fn str_index_of(text: String, substring: String) -> Int:
    """Find the first occurrence of a substring.
    
    Args:
        text: String to search in
        substring: Substring to find
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    let text_len = len(text)
    let sub_len = len(substring)
    
    if sub_len > text_len:
        return -1
    
    for i in range(text_len - sub_len + 1):
        var match = True
        for j in range(sub_len):
            if text[i + j] != substring[j]:
                match = False
                break
        if match:
            return i
    
    return -1


fn str_last_index_of(text: String, substring: String) -> Int:
    """Find the last occurrence of a substring.
    
    Args:
        text: String to search in
        substring: Substring to find
    
    Returns:
        Index of last occurrence, or -1 if not found
    """
    let text_len = len(text)
    let sub_len = len(substring)
    
    if sub_len > text_len:
        return -1
    
    for i in range(text_len - sub_len, -1, -1):
        var match = True
        for j in range(sub_len):
            if text[i + j] != substring[j]:
                match = False
                break
        if match:
            return i
    
    return -1


fn str_count(text: String, substring: String) -> Int:
    """Count occurrences of a substring.
    
    Args:
        text: String to search in
        substring: Substring to count
    
    Returns:
        Number of non-overlapping occurrences
    """
    var count = 0
    var pos = 0
    let sub_len = len(substring)
    
    while pos <= len(text) - sub_len:
        if str_index_of(text[pos:], substring) != -1:
            count += 1
            pos += str_index_of(text[pos:], substring) + sub_len
        else:
            break
    
    return count


fn str_replace(text: String, old: String, new: String) -> String:
    """Replace all occurrences of a substring.
    
    Args:
        text: Input string
        old: Substring to replace
        new: Replacement string
    
    Returns:
        String with all occurrences replaced
    """
    var result = StringBuilder()
    var pos = 0
    let text_len = len(text)
    let old_len = len(old)
    
    while pos < text_len:
        let index = str_index_of(text[pos:], old)
        if index != -1:
            # Add text before match
            result.append(text[pos:pos + index])
            # Add replacement
            result.append(new)
            # Move past the old substring
            pos += index + old_len
        else:
            # Add remaining text
            result.append(text[pos:])
            break
    
    return result.to_string()


fn str_replace_first(text: String, old: String, new: String) -> String:
    """Replace the first occurrence of a substring.
    
    Args:
        text: Input string
        old: Substring to replace
        new: Replacement string
    
    Returns:
        String with first occurrence replaced
    """
    let index = str_index_of(text, old)
    if index == -1:
        return text
    
    var result = StringBuilder()
    result.append(text[:index])
    result.append(new)
    result.append(text[index + len(old):])
    return result.to_string()


fn str_to_upper(text: String) -> String:
    """Convert string to uppercase.
    
    Args:
        text: Input string
    
    Returns:
        Uppercase version
    """
    var result = StringBuilder()
    for i in range(len(text)):
        let ch = text[i]
        # Simple ASCII conversion
        if ch >= "a" and ch <= "z":
            let code = ord(ch)
            result.append_char(chr(code - 32))
        else:
            result.append_char(ch)
    return result.to_string()


fn str_to_lower(text: String) -> String:
    """Convert string to lowercase.
    
    Args:
        text: Input string
    
    Returns:
        Lowercase version
    """
    var result = StringBuilder()
    for i in range(len(text)):
        let ch = text[i]
        # Simple ASCII conversion
        if ch >= "A" and ch <= "Z":
            let code = ord(ch)
            result.append_char(chr(code + 32))
        else:
            result.append_char(ch)
    return result.to_string()


fn str_capitalize(text: String) -> String:
    """Capitalize the first character.
    
    Args:
        text: Input string
    
    Returns:
        String with first character capitalized
    """
    if str_is_empty(text):
        return text
    
    let first = str_to_upper(text[0])
    if len(text) == 1:
        return first
    
    return first + str_to_lower(text[1:])


fn str_title(text: String) -> String:
    """Convert to title case (capitalize each word).
    
    Args:
        text: Input string
    
    Returns:
        Title-cased string
    """
    var result = StringBuilder()
    var capitalize_next = True
    
    for i in range(len(text)):
        let ch = text[i]
        if ch == " " or ch == "\t" or ch == "\n":
            result.append_char(ch)
            capitalize_next = True
        else:
            if capitalize_next:
                result.append(str_to_upper(ch))
                capitalize_next = False
            else:
                result.append(str_to_lower(ch))
    
    return result.to_string()


fn str_trim(text: String) -> String:
    """Remove leading and trailing whitespace.
    
    Args:
        text: Input string
    
    Returns:
        Trimmed string
    """
    return str_trim_left(str_trim_right(text))


fn str_trim_left(text: String) -> String:
    """Remove leading whitespace.
    
    Args:
        text: Input string
    
    Returns:
        String without leading whitespace
    """
    var start = 0
    while start < len(text):
        let ch = text[start]
        if ch != " " and ch != "\t" and ch != "\n" and ch != "\r":
            break
        start += 1
    return text[start:]


fn str_trim_right(text: String) -> String:
    """Remove trailing whitespace.
    
    Args:
        text: Input string
    
    Returns:
        String without trailing whitespace
    """
    var end = len(text)
    while end > 0:
        let ch = text[end - 1]
        if ch != " " and ch != "\t" and ch != "\n" and ch != "\r":
            break
        end -= 1
    return text[:end]


fn str_split(text: String, separator: String) -> List[String]:
    """Split string by separator.
    
    Args:
        text: String to split
        separator: Separator string
    
    Returns:
        List of substrings
    """
    var result = List[String]()
    var pos = 0
    let sep_len = len(separator)
    
    while pos < len(text):
        let index = str_index_of(text[pos:], separator)
        if index != -1:
            result.append(text[pos:pos + index])
            pos += index + sep_len
        else:
            result.append(text[pos:])
            break
    
    return result


fn str_split_lines(text: String) -> List[String]:
    """Split string into lines.
    
    Args:
        text: String to split
    
    Returns:
        List of lines
    """
    return str_split(text, "\n")


fn str_split_whitespace(text: String) -> List[String]:
    """Split string by whitespace.
    
    Args:
        text: String to split
    
    Returns:
        List of non-whitespace substrings
    """
    var result = List[String]()
    var current = StringBuilder()
    
    for i in range(len(text)):
        let ch = text[i]
        if ch == " " or ch == "\t" or ch == "\n" or ch == "\r":
            if current.size() > 0:
                result.append(current.to_string())
                current = StringBuilder()
        else:
            current.append_char(ch)
    
    if current.size() > 0:
        result.append(current.to_string())
    
    return result


fn str_join(parts: List[String], separator: String) -> String:
    """Join strings with a separator.
    
    Args:
        parts: List of strings to join
        separator: Separator between parts
    
    Returns:
        Joined string
    """
    var result = StringBuilder()
    for i in range(len(parts)):
        result.append(parts[i])
        if i < len(parts) - 1:
            result.append(separator)
    return result.to_string()


fn str_reverse(text: String) -> String:
    """Reverse a string.
    
    Args:
        text: Input string
    
    Returns:
        Reversed string
    """
    var result = StringBuilder()
    for i in range(len(text) - 1, -1, -1):
        result.append_char(text[i])
    return result.to_string()


fn str_repeat(text: String, count: Int) -> String:
    """Repeat a string n times.
    
    Args:
        text: String to repeat
        count: Number of repetitions
    
    Returns:
        Repeated string
    """
    var result = StringBuilder()
    for _ in range(count):
        result.append(text)
    return result.to_string()


fn str_pad_left(text: String, width: Int, fill: String = " ") -> String:
    """Pad string on the left to a given width.
    
    Args:
        text: String to pad
        width: Target width
        fill: Fill character (default: space)
    
    Returns:
        Left-padded string
    """
    let padding = width - len(text)
    if padding <= 0:
        return text
    
    return str_repeat(fill, padding) + text


fn str_pad_right(text: String, width: Int, fill: String = " ") -> String:
    """Pad string on the right to a given width.
    
    Args:
        text: String to pad
        width: Target width
        fill: Fill character (default: space)
    
    Returns:
        Right-padded string
    """
    let padding = width - len(text)
    if padding <= 0:
        return text
    
    return text + str_repeat(fill, padding)


fn str_pad_center(text: String, width: Int, fill: String = " ") -> String:
    """Center string within a given width.
    
    Args:
        text: String to center
        width: Target width
        fill: Fill character (default: space)
    
    Returns:
        Centered string
    """
    let padding = width - len(text)
    if padding <= 0:
        return text
    
    let left = padding // 2
    let right = padding - left
    return str_repeat(fill, left) + text + str_repeat(fill, right)


fn str_is_alpha(text: String) -> Bool:
    """Check if string contains only alphabetic characters.
    
    Args:
        text: Input string
    
    Returns:
        True if all characters are alphabetic, False otherwise
    """
    if str_is_empty(text):
        return False
    
    for i in range(len(text)):
        let ch = text[i]
        if not ((ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z")):
            return False
    
    return True


fn str_is_digit(text: String) -> Bool:
    """Check if string contains only digits.
    
    Args:
        text: Input string
    
    Returns:
        True if all characters are digits, False otherwise
    """
    if str_is_empty(text):
        return False
    
    for i in range(len(text)):
        let ch = text[i]
        if not (ch >= "0" and ch <= "9"):
            return False
    
    return True


fn str_is_alnum(text: String) -> Bool:
    """Check if string contains only alphanumeric characters.
    
    Args:
        text: Input string
    
    Returns:
        True if all characters are alphanumeric, False otherwise
    """
    if str_is_empty(text):
        return False
    
    for i in range(len(text)):
        let ch = text[i]
        if not ((ch >= "a" and ch <= "z") or 
                (ch >= "A" and ch <= "Z") or 
                (ch >= "0" and ch <= "9")):
            return False
    
    return True


fn str_is_whitespace(text: String) -> Bool:
    """Check if string contains only whitespace.
    
    Args:
        text: Input string
    
    Returns:
        True if all characters are whitespace, False otherwise
    """
    if str_is_empty(text):
        return False
    
    for i in range(len(text)):
        let ch = text[i]
        if not (ch == " " or ch == "\t" or ch == "\n" or ch == "\r"):
            return False
    
    return True


fn str_compare(a: String, b: String) -> Int:
    """Compare two strings lexicographically.
    
    Args:
        a: First string
        b: Second string
    
    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    let min_len = min(len(a), len(b))
    
    for i in range(min_len):
        if a[i] < b[i]:
            return -1
        if a[i] > b[i]:
            return 1
    
    if len(a) < len(b):
        return -1
    if len(a) > len(b):
        return 1
    
    return 0


fn str_compare_ignore_case(a: String, b: String) -> Int:
    """Compare two strings lexicographically, ignoring case.
    
    Args:
        a: First string
        b: Second string
    
    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b (case-insensitive)
    """
    return str_compare(str_to_lower(a), str_to_lower(b))
