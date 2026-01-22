"""
Mojo Path Module - Filesystem path manipulation utilities.

This module provides comprehensive path handling including:
- Path struct for filesystem paths
- Path manipulation (join, parent, filename, extension)
- Path normalization and resolution
- Platform-specific path handling
- Glob pattern matching
- Path iteration
"""

# ============================================================================
# Platform Detection
# ============================================================================

struct Platform:
    """Platform detection for path handling."""

    alias UNIX = 0
    alias WINDOWS = 1
    alias MACOS = 2

    @staticmethod
    fn current() -> Int:
        """Returns the current platform."""
        # Would be determined at compile time
        # For now, default to Unix-like
        return Platform.UNIX

    @staticmethod
    fn is_windows() -> Bool:
        """Returns True if running on Windows."""
        return Platform.current() == Platform.WINDOWS

    @staticmethod
    fn is_unix() -> Bool:
        """Returns True if running on Unix-like system."""
        return Platform.current() == Platform.UNIX or Platform.current() == Platform.MACOS

    @staticmethod
    fn is_macos() -> Bool:
        """Returns True if running on macOS."""
        return Platform.current() == Platform.MACOS

# ============================================================================
# Path Separators
# ============================================================================

alias UNIX_SEP: String = "/"
alias WINDOWS_SEP: String = "\\"
alias UNIX_PATH_SEP: String = ":"     # For PATH environment variable
alias WINDOWS_PATH_SEP: String = ";"  # For PATH environment variable

fn get_separator() -> String:
    """Returns the path separator for the current platform."""
    if Platform.is_windows():
        return WINDOWS_SEP
    return UNIX_SEP

fn get_path_separator() -> String:
    """Returns the PATH environment variable separator."""
    if Platform.is_windows():
        return WINDOWS_PATH_SEP
    return UNIX_PATH_SEP

# ============================================================================
# Path Component
# ============================================================================

struct PathComponent:
    """Represents a single component of a path."""
    var value: String
    var is_root: Bool
    var is_current: Bool    # "."
    var is_parent: Bool     # ".."

    fn __init__(inout self, value: String):
        """Creates a path component from a string."""
        self.value = value
        self.is_root = (value == "/" or
                       (len(value) == 2 and value[1] == ":" and
                        ((value[0] >= "A" and value[0] <= "Z") or
                         (value[0] >= "a" and value[0] <= "z"))))
        self.is_current = value == "."
        self.is_parent = value == ".."

    fn is_normal(self) -> Bool:
        """Returns True if this is a normal path component."""
        return not self.is_root and not self.is_current and not self.is_parent

    fn to_string(self) -> String:
        """Returns the component as a string."""
        return self.value

# ============================================================================
# Path - Main Path Type
# ============================================================================

struct Path:
    """
    Represents a filesystem path.

    Path provides cross-platform path manipulation with support for
    both Unix and Windows path formats.
    """
    var _path: String

    # ========================================================================
    # Constructors
    # ========================================================================

    fn __init__(inout self):
        """Creates an empty path."""
        self._path = ""

    fn __init__(inout self, path: String):
        """Creates a path from a string."""
        self._path = path

    @staticmethod
    fn from_string(s: String) -> Path:
        """Creates a path from a string."""
        return Path(s)

    @staticmethod
    fn cwd() -> Path:
        """Returns the current working directory (placeholder)."""
        # Would need system call integration
        return Path(".")

    @staticmethod
    fn home() -> Path:
        """Returns the user's home directory (placeholder)."""
        # Would need environment variable or system call
        if Platform.is_windows():
            return Path("C:\\Users\\Default")
        return Path("/home/user")

    @staticmethod
    fn temp() -> Path:
        """Returns the system temp directory."""
        if Platform.is_windows():
            return Path("C:\\Temp")
        return Path("/tmp")

    # ========================================================================
    # Basic Properties
    # ========================================================================

    fn as_string(self) -> String:
        """Returns the path as a string."""
        return self._path

    fn is_empty(self) -> Bool:
        """Returns True if the path is empty."""
        return len(self._path) == 0

    fn is_absolute(self) -> Bool:
        """Returns True if the path is absolute."""
        if len(self._path) == 0:
            return False

        # Unix absolute path
        if self._path[0] == "/":
            return True

        # Windows absolute path (C:\, D:\, etc.)
        if len(self._path) >= 3:
            let first = self._path[0]
            if ((first >= "A" and first <= "Z") or (first >= "a" and first <= "z")):
                if self._path[1] == ":" and (self._path[2] == "/" or self._path[2] == "\\"):
                    return True

        # UNC path (\\server\share)
        if len(self._path) >= 2:
            if self._path[0] == "\\" and self._path[1] == "\\":
                return True

        return False

    fn is_relative(self) -> Bool:
        """Returns True if the path is relative."""
        return not self.is_absolute()

    fn length(self) -> Int:
        """Returns the length of the path string."""
        return len(self._path)

    # ========================================================================
    # Path Components
    # ========================================================================

    fn filename(self) -> String:
        """Returns the final component of the path (file or directory name)."""
        if len(self._path) == 0:
            return ""

        # Find last separator
        var last_sep = -1
        for i in range(len(self._path)):
            if self._path[i] == "/" or self._path[i] == "\\":
                last_sep = i

        if last_sep == -1:
            return self._path

        if last_sep == len(self._path) - 1:
            # Path ends with separator, find the one before
            var prev_sep = -1
            for i in range(last_sep):
                if self._path[i] == "/" or self._path[i] == "\\":
                    prev_sep = i
            if prev_sep == -1:
                return self._path[0:last_sep]
            return self._path[prev_sep + 1:last_sep]

        return self._path[last_sep + 1:]

    fn stem(self) -> String:
        """Returns the filename without the extension."""
        let name = self.filename()
        if len(name) == 0:
            return ""

        # Find last dot (but not if it's the first character)
        var last_dot = -1
        for i in range(1, len(name)):
            if name[i] == ".":
                last_dot = i

        if last_dot == -1:
            return name

        return name[0:last_dot]

    fn extension(self) -> String:
        """Returns the file extension (including the dot)."""
        let name = self.filename()
        if len(name) == 0:
            return ""

        # Find last dot
        var last_dot = -1
        for i in range(1, len(name)):
            if name[i] == ".":
                last_dot = i

        if last_dot == -1:
            return ""

        return name[last_dot:]

    fn extension_without_dot(self) -> String:
        """Returns the file extension without the leading dot."""
        let ext = self.extension()
        if len(ext) > 0 and ext[0] == ".":
            return ext[1:]
        return ext

    fn parent(self) -> Path:
        """Returns the parent directory of this path."""
        if len(self._path) == 0:
            return Path("..")

        # Find last separator
        var last_sep = -1
        for i in range(len(self._path)):
            if self._path[i] == "/" or self._path[i] == "\\":
                last_sep = i

        # Skip trailing separator
        if last_sep == len(self._path) - 1 and last_sep > 0:
            var prev_sep = -1
            for i in range(last_sep):
                if self._path[i] == "/" or self._path[i] == "\\":
                    prev_sep = i
            last_sep = prev_sep

        if last_sep == -1:
            return Path(".")

        if last_sep == 0:
            return Path("/")

        # Windows root (C:\)
        if last_sep == 2 and self._path[1] == ":":
            return Path(self._path[0:3])

        return Path(self._path[0:last_sep])

    fn root(self) -> Path:
        """Returns the root component of the path."""
        if not self.is_absolute():
            return Path("")

        # Unix root
        if self._path[0] == "/":
            return Path("/")

        # Windows root (C:\)
        if len(self._path) >= 3 and self._path[1] == ":":
            return Path(self._path[0:3])

        # UNC root
        if len(self._path) >= 2 and self._path[0] == "\\" and self._path[1] == "\\":
            # Find end of server name
            var third_sep = -1
            for i in range(2, len(self._path)):
                if self._path[i] == "\\":
                    third_sep = i
                    break
            if third_sep > 0:
                # Find end of share name
                var fourth_sep = -1
                for i in range(third_sep + 1, len(self._path)):
                    if self._path[i] == "\\":
                        fourth_sep = i
                        break
                if fourth_sep > 0:
                    return Path(self._path[0:fourth_sep])
                return Path(self._path)

        return Path("")

    fn components(self) -> List[PathComponent]:
        """Returns the path split into its components."""
        var result = List[PathComponent]()

        if len(self._path) == 0:
            return result

        var current = String("")
        var i = 0

        # Handle root
        if self.is_absolute():
            if self._path[0] == "/":
                result.append(PathComponent("/"))
                i = 1
            elif len(self._path) >= 3 and self._path[1] == ":":
                result.append(PathComponent(self._path[0:2]))
                i = 3

        # Split remaining path
        while i < len(self._path):
            let c = self._path[i]

            if c == "/" or c == "\\":
                if len(current) > 0:
                    result.append(PathComponent(current))
                    current = ""
            else:
                current = current + c

            i += 1

        # Add final component
        if len(current) > 0:
            result.append(PathComponent(current))

        return result

    fn depth(self) -> Int:
        """Returns the depth of the path (number of components)."""
        return len(self.components())

    # ========================================================================
    # Path Manipulation
    # ========================================================================

    fn join(self, other: String) -> Path:
        """Joins this path with another path component."""
        if len(other) == 0:
            return Path(self._path)

        if len(self._path) == 0:
            return Path(other)

        # If other is absolute, return it
        let other_path = Path(other)
        if other_path.is_absolute():
            return other_path

        # Get separator
        let sep = get_separator()

        # Check if we need to add separator
        let last_char = self._path[len(self._path) - 1]
        if last_char == "/" or last_char == "\\":
            return Path(self._path + other)

        return Path(self._path + sep + other)

    fn join(self, other: Path) -> Path:
        """Joins this path with another path."""
        return self.join(other._path)

    fn __truediv__(self, other: String) -> Path:
        """Operator / for joining paths."""
        return self.join(other)

    fn __truediv__(self, other: Path) -> Path:
        """Operator / for joining paths."""
        return self.join(other)

    fn with_filename(self, name: String) -> Path:
        """Returns a new path with the filename replaced."""
        return self.parent().join(name)

    fn with_extension(self, ext: String) -> Path:
        """Returns a new path with the extension replaced."""
        let stem_str = self.stem()
        let parent = self.parent()

        # Ensure extension starts with dot
        var new_ext = ext
        if len(ext) > 0 and ext[0] != ".":
            new_ext = "." + ext

        return parent.join(stem_str + new_ext)

    fn with_stem(self, new_stem: String) -> Path:
        """Returns a new path with the stem (filename without extension) replaced."""
        let ext = self.extension()
        let parent = self.parent()
        return parent.join(new_stem + ext)

    # ========================================================================
    # Normalization
    # ========================================================================

    fn normalize(self) -> Path:
        """
        Normalizes the path by resolving . and .. components.

        Does not resolve symlinks or access the filesystem.
        """
        let comps = self.components()
        var result = List[String]()
        var is_abs = self.is_absolute()

        for i in range(len(comps)):
            let comp = comps[i]

            if comp.is_root:
                result.append(comp.value)
            elif comp.is_current:
                # Skip "."
                pass
            elif comp.is_parent:
                # Go up one level if possible
                if len(result) > 0 and result[len(result) - 1] != "..":
                    let last = result[len(result) - 1]
                    # Don't pop root
                    if last != "/" and not (len(last) == 2 and last[1] == ":"):
                        _ = result.pop()
                    else:
                        # At root, ".." is ignored
                        pass
                elif not is_abs:
                    result.append("..")
            else:
                result.append(comp.value)

        # Build result path
        if len(result) == 0:
            return Path(".")

        var path_str = String("")
        let sep = get_separator()

        for i in range(len(result)):
            if i > 0:
                let prev = result[i - 1]
                # Don't add separator after root that already has one
                if not (prev == "/" or (len(prev) >= 2 and prev[len(prev) - 1] == "\\")):
                    path_str = path_str + sep
            path_str = path_str + result[i]

        return Path(path_str)

    fn to_unix(self) -> Path:
        """Converts path to Unix format (forward slashes)."""
        var result = String("")
        for i in range(len(self._path)):
            if self._path[i] == "\\":
                result = result + "/"
            else:
                result = result + self._path[i]
        return Path(result)

    fn to_windows(self) -> Path:
        """Converts path to Windows format (backslashes)."""
        var result = String("")
        for i in range(len(self._path)):
            if self._path[i] == "/":
                result = result + "\\"
            else:
                result = result + self._path[i]
        return Path(result)

    fn to_native(self) -> Path:
        """Converts path to native format for current platform."""
        if Platform.is_windows():
            return self.to_windows()
        return self.to_unix()

    # ========================================================================
    # Path Queries
    # ========================================================================

    fn starts_with(self, prefix: String) -> Bool:
        """Returns True if path starts with the given prefix."""
        if len(prefix) > len(self._path):
            return False
        return self._path[0:len(prefix)] == prefix

    fn starts_with(self, prefix: Path) -> Bool:
        """Returns True if path starts with the given path prefix."""
        let self_comps = self.components()
        let prefix_comps = prefix.components()

        if len(prefix_comps) > len(self_comps):
            return False

        for i in range(len(prefix_comps)):
            if self_comps[i].value != prefix_comps[i].value:
                return False

        return True

    fn ends_with(self, suffix: String) -> Bool:
        """Returns True if path ends with the given suffix."""
        if len(suffix) > len(self._path):
            return False
        let start = len(self._path) - len(suffix)
        return self._path[start:] == suffix

    fn contains(self, s: String) -> Bool:
        """Returns True if path contains the given string."""
        for i in range(len(self._path) - len(s) + 1):
            if self._path[i:i + len(s)] == s:
                return True
        return False

    fn has_extension(self, ext: String) -> Bool:
        """Returns True if path has the given extension."""
        let path_ext = self.extension_without_dot()

        # Normalize input (remove leading dot if present)
        var check_ext = ext
        if len(ext) > 0 and ext[0] == ".":
            check_ext = ext[1:]

        # Case-insensitive comparison
        return path_ext.lower() == check_ext.lower()

    fn is_hidden(self) -> Bool:
        """Returns True if the file/directory is hidden (starts with .)."""
        let name = self.filename()
        return len(name) > 0 and name[0] == "."

    # ========================================================================
    # Relative Path Operations
    # ========================================================================

    fn relative_to(self, base: Path) -> Path:
        """
        Returns this path relative to the base path.

        Both paths should be absolute or both should be relative.
        """
        let self_comps = self.components()
        let base_comps = base.components()

        # Find common prefix length
        var common_len = 0
        let min_len = len(self_comps) if len(self_comps) < len(base_comps) else len(base_comps)

        while common_len < min_len:
            if self_comps[common_len].value != base_comps[common_len].value:
                break
            common_len += 1

        # Build relative path
        var result = List[String]()

        # Add ".." for each remaining component in base
        for i in range(common_len, len(base_comps)):
            result.append("..")

        # Add remaining components from self
        for i in range(common_len, len(self_comps)):
            result.append(self_comps[i].value)

        if len(result) == 0:
            return Path(".")

        let sep = get_separator()
        var path_str = result[0]
        for i in range(1, len(result)):
            path_str = path_str + sep + result[i]

        return Path(path_str)

    fn strip_prefix(self, prefix: Path) -> Path:
        """Removes the prefix from this path if it matches."""
        if not self.starts_with(prefix):
            return Path(self._path)

        let self_comps = self.components()
        let prefix_comps = prefix.components()

        if len(prefix_comps) >= len(self_comps):
            return Path(".")

        let sep = get_separator()
        var result = self_comps[len(prefix_comps)].value

        for i in range(len(prefix_comps) + 1, len(self_comps)):
            result = result + sep + self_comps[i].value

        return Path(result)

    # ========================================================================
    # Comparison
    # ========================================================================

    fn __eq__(self, other: Path) -> Bool:
        """Checks if two paths are equal (string comparison)."""
        return self._path == other._path

    fn __ne__(self, other: Path) -> Bool:
        """Checks if two paths are not equal."""
        return self._path != other._path

    fn equals_normalized(self, other: Path) -> Bool:
        """Checks if two paths are equal after normalization."""
        return self.normalize()._path == other.normalize()._path

    fn __lt__(self, other: Path) -> Bool:
        """Lexicographic comparison."""
        return self._path < other._path

    fn __le__(self, other: Path) -> Bool:
        return self._path <= other._path

    fn __gt__(self, other: Path) -> Bool:
        return self._path > other._path

    fn __ge__(self, other: Path) -> Bool:
        return self._path >= other._path

    # ========================================================================
    # String Conversion
    # ========================================================================

    fn to_string(self) -> String:
        """Returns the path as a string."""
        return self._path

    fn __str__(self) -> String:
        """String conversion."""
        return self._path

    fn display(self) -> String:
        """Returns a display-friendly representation."""
        return self._path

# ============================================================================
# Glob Pattern Matching
# ============================================================================

struct GlobPattern:
    """
    Glob pattern for matching file paths.

    Supports:
    - * : matches any sequence of characters (not including separator)
    - ** : matches any sequence of characters (including separator)
    - ? : matches any single character
    - [...] : matches any character in brackets
    - [!...] : matches any character not in brackets
    """
    var pattern: String
    var _segments: List[String]

    fn __init__(inout self, pattern: String):
        """Creates a glob pattern."""
        self.pattern = pattern
        self._segments = self._parse_segments()

    fn _parse_segments(self) -> List[String]:
        """Parses the pattern into segments."""
        var result = List[String]()
        var current = String("")

        for i in range(len(self.pattern)):
            let c = self.pattern[i]
            if c == "/" or c == "\\":
                if len(current) > 0:
                    result.append(current)
                    current = ""
            else:
                current = current + c

        if len(current) > 0:
            result.append(current)

        return result

    fn matches(self, path: String) -> Bool:
        """Returns True if the path matches the glob pattern."""
        return self._match_pattern(self.pattern, path, 0, 0)

    fn matches(self, path: Path) -> Bool:
        """Returns True if the path matches the glob pattern."""
        return self.matches(path.as_string())

    fn _match_pattern(self, pattern: String, text: String,
                      pi: Int, ti: Int) -> Bool:
        """Recursive pattern matching."""
        var p_idx = pi
        var t_idx = ti

        while p_idx < len(pattern):
            if t_idx >= len(text):
                # Check if remaining pattern is just stars
                while p_idx < len(pattern) and pattern[p_idx] == "*":
                    p_idx += 1
                return p_idx >= len(pattern)

            let p_char = pattern[p_idx]

            if p_char == "*":
                # Check for **
                if p_idx + 1 < len(pattern) and pattern[p_idx + 1] == "*":
                    # ** matches any path
                    p_idx += 2

                    # Skip any following slashes
                    while p_idx < len(pattern) and (pattern[p_idx] == "/" or pattern[p_idx] == "\\"):
                        p_idx += 1

                    if p_idx >= len(pattern):
                        return True

                    # Try matching at every position
                    while t_idx <= len(text):
                        if self._match_pattern(pattern, text, p_idx, t_idx):
                            return True
                        t_idx += 1

                    return False
                else:
                    # * matches anything except separator
                    p_idx += 1

                    # Try matching at every position (not crossing separator)
                    let start_t = t_idx
                    while t_idx <= len(text):
                        if t_idx > start_t:
                            let prev = text[t_idx - 1]
                            if prev == "/" or prev == "\\":
                                return False

                        if self._match_pattern(pattern, text, p_idx, t_idx):
                            return True
                        t_idx += 1

                    return False

            elif p_char == "?":
                # ? matches any single character except separator
                let t_char = text[t_idx]
                if t_char == "/" or t_char == "\\":
                    return False
                p_idx += 1
                t_idx += 1

            elif p_char == "[":
                # Character class
                let result = self._match_bracket(pattern, text, p_idx, t_idx)
                if result.matched:
                    p_idx = result.new_p_idx
                    t_idx += 1
                else:
                    return False

            else:
                # Literal character (treat / and \ as equivalent)
                let t_char = text[t_idx]

                var matches_sep = (p_char == "/" or p_char == "\\") and (t_char == "/" or t_char == "\\")
                var matches_exact = p_char == t_char

                if not (matches_sep or matches_exact):
                    return False

                p_idx += 1
                t_idx += 1

        return t_idx >= len(text)

    fn _match_bracket(self, pattern: String, text: String,
                      p_idx: Int, t_idx: Int) -> BracketMatchResult:
        """Matches a bracket expression [...]."""
        var result = BracketMatchResult()
        result.matched = False
        result.new_p_idx = p_idx

        if p_idx >= len(pattern) or pattern[p_idx] != "[":
            return result

        var i = p_idx + 1
        var negate = False
        var char_matched = False

        if i < len(pattern) and (pattern[i] == "!" or pattern[i] == "^"):
            negate = True
            i += 1

        let t_char = text[t_idx]

        while i < len(pattern) and pattern[i] != "]":
            # Check for range (a-z)
            if i + 2 < len(pattern) and pattern[i + 1] == "-" and pattern[i + 2] != "]":
                let range_start = pattern[i]
                let range_end = pattern[i + 2]

                if t_char >= range_start and t_char <= range_end:
                    char_matched = True

                i += 3
            else:
                if pattern[i] == t_char:
                    char_matched = True
                i += 1

        if i < len(pattern) and pattern[i] == "]":
            result.new_p_idx = i + 1
            result.matched = char_matched != negate  # XOR with negate

        return result

struct BracketMatchResult:
    """Result of bracket matching."""
    var matched: Bool
    var new_p_idx: Int

    fn __init__(inout self):
        self.matched = False
        self.new_p_idx = 0

# ============================================================================
# Path Builder
# ============================================================================

struct PathBuilder:
    """Fluent builder for constructing paths."""
    var _parts: List[String]

    fn __init__(inout self):
        """Creates an empty path builder."""
        self._parts = List[String]()

    fn __init__(inout self, base: String):
        """Creates a path builder with a base path."""
        self._parts = List[String]()
        self._parts.append(base)

    fn __init__(inout self, base: Path):
        """Creates a path builder with a base path."""
        self._parts = List[String]()
        self._parts.append(base.as_string())

    fn push(inout self, component: String) -> PathBuilder:
        """Adds a path component."""
        self._parts.append(component)
        return self

    fn push(inout self, component: Path) -> PathBuilder:
        """Adds a path component."""
        self._parts.append(component.as_string())
        return self

    fn pop(inout self) -> PathBuilder:
        """Removes the last path component."""
        if len(self._parts) > 0:
            _ = self._parts.pop()
        return self

    fn build(self) -> Path:
        """Builds the final path."""
        if len(self._parts) == 0:
            return Path("")

        var result = Path(self._parts[0])
        for i in range(1, len(self._parts)):
            result = result.join(self._parts[i])

        return result

# ============================================================================
# Utility Functions
# ============================================================================

fn join_paths(parts: List[String]) -> Path:
    """Joins multiple path parts into a single path."""
    if len(parts) == 0:
        return Path("")

    var result = Path(parts[0])
    for i in range(1, len(parts)):
        result = result.join(parts[i])

    return result

fn split_path(path: String) -> List[String]:
    """Splits a path into its components."""
    var result = List[String]()
    var current = String("")

    for i in range(len(path)):
        let c = path[i]
        if c == "/" or c == "\\":
            if len(current) > 0:
                result.append(current)
                current = ""
        else:
            current = current + c

    if len(current) > 0:
        result.append(current)

    return result

fn common_path(paths: List[Path]) -> Path:
    """Returns the longest common path prefix."""
    if len(paths) == 0:
        return Path("")

    if len(paths) == 1:
        return paths[0]

    # Get components of first path
    let first_comps = paths[0].components()
    var common_len = len(first_comps)

    # Compare with each other path
    for i in range(1, len(paths)):
        let other_comps = paths[i].components()
        var match_len = 0
        let min_len = common_len if common_len < len(other_comps) else len(other_comps)

        while match_len < min_len:
            if first_comps[match_len].value != other_comps[match_len].value:
                break
            match_len += 1

        common_len = match_len

    # Build result
    if common_len == 0:
        return Path("")

    let sep = get_separator()
    var result = first_comps[0].value

    for i in range(1, common_len):
        result = result + sep + first_comps[i].value

    return Path(result)

fn expand_tilde(path: String) -> Path:
    """Expands ~ to user's home directory."""
    if len(path) == 0 or path[0] != "~":
        return Path(path)

    let home = Path.home()

    if len(path) == 1:
        return home

    if path[1] == "/" or path[1] == "\\":
        return home.join(path[2:])

    return Path(path)

fn is_valid_filename(name: String) -> Bool:
    """Checks if a string is a valid filename."""
    if len(name) == 0:
        return False

    # Check for reserved characters
    for i in range(len(name)):
        let c = name[i]
        if c == "/" or c == "\\" or c == ":" or c == "*" or c == "?" or c == "\"" or c == "<" or c == ">" or c == "|":
            return False
        # Check for control characters
        if ord(c) < 32:
            return False

    # Check for reserved names on Windows
    let lower = name.lower()
    let reserved = ["con", "prn", "aux", "nul",
                    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
                    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"]

    for reserved_name in reserved:
        if lower == reserved_name:
            return False

    return True

fn sanitize_filename(name: String) -> String:
    """Removes or replaces invalid characters from a filename."""
    var result = String("")

    for i in range(len(name)):
        let c = name[i]

        # Replace invalid characters with underscore
        if c == "/" or c == "\\" or c == ":" or c == "*" or c == "?" or c == "\"" or c == "<" or c == ">" or c == "|":
            result = result + "_"
        elif ord(c) < 32:
            # Skip control characters
            pass
        else:
            result = result + c

    return result

# ============================================================================
# Path Iteration Helpers
# ============================================================================

struct PathIterator:
    """Iterator over path components."""
    var _components: List[PathComponent]
    var _index: Int

    fn __init__(inout self, path: Path):
        """Creates an iterator for a path."""
        self._components = path.components()
        self._index = 0

    fn has_next(self) -> Bool:
        """Returns True if there are more components."""
        return self._index < len(self._components)

    fn next(inout self) -> PathComponent:
        """Returns the next component."""
        let comp = self._components[self._index]
        self._index += 1
        return comp

    fn reset(inout self):
        """Resets the iterator to the beginning."""
        self._index = 0

struct AncestorIterator:
    """Iterator over a path's ancestors (parent directories)."""
    var _current: Path
    var _done: Bool

    fn __init__(inout self, path: Path):
        """Creates an ancestor iterator."""
        self._current = path
        self._done = False

    fn has_next(self) -> Bool:
        """Returns True if there are more ancestors."""
        return not self._done

    fn next(inout self) -> Path:
        """Returns the next ancestor."""
        let result = self._current
        let parent = self._current.parent()

        if parent == self._current or parent.is_empty():
            self._done = True
        else:
            self._current = parent

        return result

# ============================================================================
# Tests
# ============================================================================

fn test_path_basics():
    """Test basic path operations."""
    print("Testing Path basics...")

    let p = Path("/home/user/documents/file.txt")

    assert_true(p.is_absolute(), "is_absolute")
    assert_true(p.filename() == "file.txt", "filename")
    assert_true(p.stem() == "file", "stem")
    assert_true(p.extension() == ".txt", "extension")
    assert_true(p.extension_without_dot() == "txt", "extension_without_dot")

    let parent = p.parent()
    assert_true(parent.as_string() == "/home/user/documents", "parent")

    print("Path basics tests passed!")

fn test_path_join():
    """Test path joining."""
    print("Testing Path join...")

    let base = Path("/home/user")
    let joined = base.join("documents").join("file.txt")
    assert_true(joined.as_string() == "/home/user/documents/file.txt", "join")

    # Using operator /
    let p2 = base / "downloads" / "archive.zip"
    assert_true(p2.filename() == "archive.zip", "operator /")

    print("Path join tests passed!")

fn test_path_normalize():
    """Test path normalization."""
    print("Testing Path normalize...")

    let p1 = Path("/home/user/../user/./documents")
    let normalized = p1.normalize()
    assert_true(normalized.as_string() == "/home/user/documents", "normalize with ..")

    let p2 = Path("./foo/bar/../baz")
    let n2 = p2.normalize()
    assert_true(n2.as_string() == "foo/baz", "normalize relative")

    print("Path normalize tests passed!")

fn test_path_windows():
    """Test Windows path handling."""
    print("Testing Windows paths...")

    let win_path = Path("C:\\Users\\Admin\\Documents")
    assert_true(win_path.is_absolute(), "Windows absolute")
    assert_true(win_path.filename() == "Documents", "Windows filename")

    let unix_converted = win_path.to_unix()
    assert_true(unix_converted.as_string() == "C:/Users/Admin/Documents", "to_unix")

    print("Windows path tests passed!")

fn test_glob_pattern():
    """Test glob pattern matching."""
    print("Testing Glob patterns...")

    let pattern1 = GlobPattern("*.txt")
    assert_true(pattern1.matches("file.txt"), "simple wildcard")
    assert_true(not pattern1.matches("file.pdf"), "no match")

    let pattern2 = GlobPattern("src/**/*.py")
    assert_true(pattern2.matches("src/module/file.py"), "double star")
    assert_true(pattern2.matches("src/a/b/c/file.py"), "deep double star")

    let pattern3 = GlobPattern("test?.txt")
    assert_true(pattern3.matches("test1.txt"), "single char wildcard")
    assert_true(not pattern3.matches("test12.txt"), "too many chars")

    print("Glob pattern tests passed!")

fn test_path_builder():
    """Test path builder."""
    print("Testing PathBuilder...")

    var builder = PathBuilder("/home")
    let path = builder.push("user").push("documents").push("file.txt").build()

    assert_true(path.as_string() == "/home/user/documents/file.txt", "builder")

    print("PathBuilder tests passed!")

fn test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")

    # Valid filename
    assert_true(is_valid_filename("file.txt"), "valid filename")
    assert_true(not is_valid_filename("file/name.txt"), "invalid slash")
    assert_true(not is_valid_filename("con"), "reserved name")

    # Sanitize filename
    let sanitized = sanitize_filename("my:file<name>.txt")
    assert_true(not sanitized.contains(":"), "sanitized colon")
    assert_true(not sanitized.contains("<"), "sanitized lt")

    # Expand tilde
    let expanded = expand_tilde("~/documents")
    assert_true(expanded.contains("home") or expanded.contains("Users"), "expand tilde")

    print("Utility function tests passed!")

fn assert_true(condition: Bool, message: String):
    """Assert helper."""
    if not condition:
        print("ASSERTION FAILED: " + message)

fn run_all_tests():
    """Run all path module tests."""
    print("=== Path Module Tests ===")
    test_path_basics()
    test_path_join()
    test_path_normalize()
    test_path_windows()
    test_glob_pattern()
    test_path_builder()
    test_utility_functions()
    print("=== All tests passed! ===")
