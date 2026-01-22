# IO - Input/Output Operations
# Day 34: File I/O, console I/O, streams, and buffering

from builtin import Int, Bool, String
from collections.list import List


# Console I/O

fn print(value: String):
    """Print a value to stdout with newline.
    
    Args:
        value: Value to print
    """
    # Calls underlying C printf or similar
    __print_impl(value + "\n")


fn print_no_newline(value: String):
    """Print without adding newline.
    
    Args:
        value: Value to print
    """
    __print_impl(value)


fn println(value: String):
    """Print with newline (alias for print).
    
    Args:
        value: Value to print
    """
    print(value)


fn eprint(value: String):
    """Print to stderr.
    
    Args:
        value: Value to print to stderr
    """
    __eprint_impl(value + "\n")


fn input(prompt: String = "") -> String:
    """Read a line from stdin.
    
    Args:
        prompt: Optional prompt to display
    
    Returns:
        Line read from stdin (without newline)
    """
    if len(prompt) > 0:
        print_no_newline(prompt)
    return __input_impl()


# Internal implementation stubs
fn __print_impl(text: String):
    pass  # Would call libc printf or write syscall

fn __eprint_impl(text: String):
    pass  # Would call libc fprintf(stderr, ...) or write syscall

fn __input_impl() -> String:
    pass  # Would call libc fgets or read syscall
    return ""


# File I/O

struct File:
    """File handle for reading and writing files.
    
    Provides buffered file I/O operations with support for reading,
    writing, and seeking.
    
    Examples:
        ```mojo
        # Reading
        var file = File.open("data.txt", "r")
        let content = file.read()
        file.close()
        
        # Writing
        var out = File.open("output.txt", "w")
        out.write("Hello, World!")
        out.close()
        
        # With context manager (auto-close)
        with File.open("data.txt", "r") as file:
            let lines = file.read_lines()
        ```
    """
    
    var path: String
    var mode: String
    var handle: Int  # File descriptor
    var is_open: Bool
    var buffer: String
    var position: Int
    
    fn __init__(inout self, path: String, mode: String):
        """Initialize file handle.
        
        Args:
            path: File path
            mode: Open mode ("r", "w", "a", "r+", "w+", "a+")
        """
        self.path = path
        self.mode = mode
        self.handle = -1
        self.is_open = False
        self.buffer = ""
        self.position = 0
    
    @staticmethod
    fn open(path: String, mode: String = "r") -> File:
        """Open a file.
        
        Args:
            path: File path
            mode: Open mode (default: "r")
                - "r": Read
                - "w": Write (truncate)
                - "a": Append
                - "r+": Read/write
                - "w+": Read/write (truncate)
                - "a+": Read/append
        
        Returns:
            File handle
        """
        var file = File(path, mode)
        file._open()
        return file
    
    fn _open(inout self):
        """Internal: Open the file."""
        # Would call libc fopen or open syscall
        self.handle = __file_open(self.path, self.mode)
        self.is_open = True
    
    fn close(inout self):
        """Close the file."""
        if self.is_open:
            __file_close(self.handle)
            self.is_open = False
    
    fn read(inout self) -> String:
        """Read entire file contents.
        
        Returns:
            File contents as string
        """
        if not self.is_open:
            return ""
        return __file_read_all(self.handle)
    
    fn read_bytes(inout self, count: Int) -> String:
        """Read specified number of bytes.
        
        Args:
            count: Number of bytes to read
        
        Returns:
            Read bytes as string
        """
        if not self.is_open:
            return ""
        return __file_read_bytes(self.handle, count)
    
    fn read_line(inout self) -> String:
        """Read a single line.
        
        Returns:
            Line without newline character
        """
        if not self.is_open:
            return ""
        return __file_read_line(self.handle)
    
    fn read_lines(inout self) -> List[String]:
        """Read all lines from file.
        
        Returns:
            List of lines (without newlines)
        """
        var lines = List[String]()
        while self.is_open:
            let line = self.read_line()
            if len(line) == 0:
                break
            lines.append(line)
        return lines
    
    fn write(inout self, text: String):
        """Write string to file.
        
        Args:
            text: Text to write
        """
        if self.is_open:
            __file_write(self.handle, text)
    
    fn write_line(inout self, text: String):
        """Write string followed by newline.
        
        Args:
            text: Text to write
        """
        self.write(text + "\n")
    
    fn write_lines(inout self, lines: List[String]):
        """Write multiple lines.
        
        Args:
            lines: Lines to write (newlines added automatically)
        """
        for line in lines:
            self.write_line(line)
    
    fn flush(inout self):
        """Flush write buffer to disk."""
        if self.is_open:
            __file_flush(self.handle)
    
    fn seek(inout self, position: Int):
        """Seek to position in file.
        
        Args:
            position: Byte offset from start
        """
        if self.is_open:
            self.position = position
            __file_seek(self.handle, position)
    
    fn tell(self) -> Int:
        """Get current file position.
        
        Returns:
            Current byte offset
        """
        return self.position
    
    fn size(self) -> Int:
        """Get file size.
        
        Returns:
            File size in bytes
        """
        if not self.is_open:
            return 0
        return __file_size(self.handle)
    
    fn __enter__(inout self) -> File:
        """Context manager entry."""
        return self
    
    fn __exit__(inout self):
        """Context manager exit (auto-close)."""
        self.close()


# File operation utilities

fn read_file(path: String) -> String:
    """Read entire file contents.
    
    Args:
        path: File path
    
    Returns:
        File contents
    """
    var file = File.open(path, "r")
    let content = file.read()
    file.close()
    return content


fn write_file(path: String, content: String):
    """Write content to file (overwrite).
    
    Args:
        path: File path
        content: Content to write
    """
    var file = File.open(path, "w")
    file.write(content)
    file.close()


fn append_file(path: String, content: String):
    """Append content to file.
    
    Args:
        path: File path
        content: Content to append
    """
    var file = File.open(path, "a")
    file.write(content)
    file.close()


fn file_exists(path: String) -> Bool:
    """Check if file exists.
    
    Args:
        path: File path
    
    Returns:
        True if file exists, False otherwise
    """
    return __file_exists(path)


fn delete_file(path: String) -> Bool:
    """Delete a file.
    
    Args:
        path: File path
    
    Returns:
        True if deleted successfully, False otherwise
    """
    return __file_delete(path)


fn copy_file(source: String, destination: String) -> Bool:
    """Copy a file.
    
    Args:
        source: Source file path
        destination: Destination file path
    
    Returns:
        True if copied successfully, False otherwise
    """
    let content = read_file(source)
    write_file(destination, content)
    return True


fn move_file(source: String, destination: String) -> Bool:
    """Move/rename a file.
    
    Args:
        source: Source file path
        destination: Destination file path
    
    Returns:
        True if moved successfully, False otherwise
    """
    return __file_move(source, destination)


# Stream I/O

struct InputStream:
    """Abstract input stream for reading data."""
    
    var source: String
    var position: Int
    
    fn __init__(inout self, source: String):
        self.source = source
        self.position = 0
    
    fn read_char(inout self) -> String:
        """Read a single character.
        
        Returns:
            Next character or empty string if EOF
        """
        if self.position >= len(self.source):
            return ""
        let ch = self.source[self.position]
        self.position += 1
        return ch
    
    fn peek_char(self) -> String:
        """Peek at next character without advancing.
        
        Returns:
            Next character or empty string if EOF
        """
        if self.position >= len(self.source):
            return ""
        return self.source[self.position]
    
    fn has_next(self) -> Bool:
        """Check if more data available.
        
        Returns:
            True if more data available, False otherwise
        """
        return self.position < len(self.source)
    
    fn reset(inout self):
        """Reset stream to beginning."""
        self.position = 0


struct OutputStream:
    """Abstract output stream for writing data."""
    
    var buffer: String
    
    fn __init__(inout self):
        self.buffer = ""
    
    fn write_char(inout self, ch: String):
        """Write a single character.
        
        Args:
            ch: Character to write
        """
        self.buffer += ch
    
    fn write(inout self, text: String):
        """Write a string.
        
        Args:
            text: Text to write
        """
        self.buffer += text
    
    fn write_line(inout self, text: String):
        """Write a line with newline.
        
        Args:
            text: Text to write
        """
        self.buffer += text + "\n"
    
    fn to_string(self) -> String:
        """Get buffered content.
        
        Returns:
            All written content
        """
        return self.buffer
    
    fn clear(inout self):
        """Clear the buffer."""
        self.buffer = ""


struct BufferedReader:
    """Buffered reader for efficient file reading."""
    
    var file: File
    var buffer: String
    var buffer_pos: Int
    var buffer_size: Int
    
    fn __init__(inout self, file: File, buffer_size: Int = 4096):
        """Initialize buffered reader.
        
        Args:
            file: File to read from
            buffer_size: Buffer size in bytes (default: 4096)
        """
        self.file = file
        self.buffer = ""
        self.buffer_pos = 0
        self.buffer_size = buffer_size
    
    fn read_line(inout self) -> String:
        """Read a line efficiently.
        
        Returns:
            Line without newline
        """
        # Fill buffer if empty
        if self.buffer_pos >= len(self.buffer):
            self.buffer = self.file.read_bytes(self.buffer_size)
            self.buffer_pos = 0
            if len(self.buffer) == 0:
                return ""
        
        # Find newline in buffer
        var line = ""
        while self.buffer_pos < len(self.buffer):
            let ch = self.buffer[self.buffer_pos]
            self.buffer_pos += 1
            if ch == "\n":
                break
            line += ch
        
        return line


struct BufferedWriter:
    """Buffered writer for efficient file writing."""
    
    var file: File
    var buffer: String
    var buffer_size: Int
    
    fn __init__(inout self, file: File, buffer_size: Int = 4096):
        """Initialize buffered writer.
        
        Args:
            file: File to write to
            buffer_size: Buffer size in bytes (default: 4096)
        """
        self.file = file
        self.buffer = ""
        self.buffer_size = buffer_size
    
    fn write(inout self, text: String):
        """Write text with buffering.
        
        Args:
            text: Text to write
        """
        self.buffer += text
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    fn write_line(inout self, text: String):
        """Write line with buffering.
        
        Args:
            text: Text to write
        """
        self.write(text + "\n")
    
    fn flush(inout self):
        """Flush buffer to file."""
        if len(self.buffer) > 0:
            self.file.write(self.buffer)
            self.buffer = ""
    
    fn close(inout self):
        """Flush and close."""
        self.flush()
        self.file.close()


# Path operations

fn path_join(parts: List[String]) -> String:
    """Join path components.
    
    Args:
        parts: Path components
    
    Returns:
        Joined path
    """
    var result = ""
    for i in range(len(parts)):
        result += parts[i]
        if i < len(parts) - 1 and not result.endswith("/"):
            result += "/"
    return result


fn path_dirname(path: String) -> String:
    """Get directory name from path.
    
    Args:
        path: File path
    
    Returns:
        Directory portion
    """
    let last_slash = path.rfind("/")
    if last_slash == -1:
        return "."
    return path[:last_slash]


fn path_basename(path: String) -> String:
    """Get base name from path.
    
    Args:
        path: File path
    
    Returns:
        File name portion
    """
    let last_slash = path.rfind("/")
    if last_slash == -1:
        return path
    return path[last_slash + 1:]


fn path_extension(path: String) -> String:
    """Get file extension.
    
    Args:
        path: File path
    
    Returns:
        Extension including dot, or empty string
    """
    let basename = path_basename(path)
    let last_dot = basename.rfind(".")
    if last_dot == -1:
        return ""
    return basename[last_dot:]


fn path_without_extension(path: String) -> String:
    """Get path without extension.
    
    Args:
        path: File path
    
    Returns:
        Path without extension
    """
    let ext = path_extension(path)
    if len(ext) == 0:
        return path
    return path[:len(path) - len(ext)]


# Directory operations

fn list_dir(path: String) -> List[String]:
    """List directory contents.
    
    Args:
        path: Directory path
    
    Returns:
        List of file/directory names
    """
    return __list_dir_impl(path)


fn make_dir(path: String) -> Bool:
    """Create a directory.
    
    Args:
        path: Directory path
    
    Returns:
        True if created successfully, False otherwise
    """
    return __make_dir_impl(path)


fn remove_dir(path: String) -> Bool:
    """Remove a directory.
    
    Args:
        path: Directory path
    
    Returns:
        True if removed successfully, False otherwise
    """
    return __remove_dir_impl(path)


fn is_dir(path: String) -> Bool:
    """Check if path is a directory.
    
    Args:
        path: Path to check
    
    Returns:
        True if directory, False otherwise
    """
    return __is_dir_impl(path)


fn is_file(path: String) -> Bool:
    """Check if path is a file.
    
    Args:
        path: Path to check
    
    Returns:
        True if file, False otherwise
    """
    return __is_file_impl(path)


# Binary I/O

struct BinaryReader:
    """Reader for binary data."""
    
    var file: File
    
    fn __init__(inout self, file: File):
        self.file = file
    
    fn read_byte(inout self) -> Int:
        """Read a single byte.
        
        Returns:
            Byte value (0-255) or -1 if EOF
        """
        let data = self.file.read_bytes(1)
        if len(data) == 0:
            return -1
        return ord(data[0])
    
    fn read_int(inout self) -> Int:
        """Read a 4-byte integer (little-endian).
        
        Returns:
            Integer value
        """
        let bytes = self.file.read_bytes(4)
        var result = 0
        for i in range(4):
            result |= ord(bytes[i]) << (i * 8)
        return result
    
    fn read_float(inout self) -> Float64:
        """Read an 8-byte float (little-endian).
        
        Returns:
            Float value
        """
        # Would use proper float conversion
        let bytes = self.file.read_bytes(8)
        return 0.0  # Placeholder


struct BinaryWriter:
    """Writer for binary data."""
    
    var file: File
    
    fn __init__(inout self, file: File):
        self.file = file
    
    fn write_byte(inout self, value: Int):
        """Write a single byte.
        
        Args:
            value: Byte value (0-255)
        """
        self.file.write(chr(value % 256))
    
    fn write_int(inout self, value: Int):
        """Write a 4-byte integer (little-endian).
        
        Args:
            value: Integer to write
        """
        for i in range(4):
            self.write_byte((value >> (i * 8)) & 0xFF)
    
    fn write_float(inout self, value: Float64):
        """Write an 8-byte float (little-endian).
        
        Args:
            value: Float to write
        """
        # Would use proper float conversion
        pass


# Internal stubs (would be implemented in Zig/C)

fn __file_open(path: String, mode: String) -> Int:
    return 0  # File descriptor

fn __file_close(handle: Int):
    pass

fn __file_read_all(handle: Int) -> String:
    return ""

fn __file_read_bytes(handle: Int, count: Int) -> String:
    return ""

fn __file_read_line(handle: Int) -> String:
    return ""

fn __file_write(handle: Int, text: String):
    pass

fn __file_flush(handle: Int):
    pass

fn __file_seek(handle: Int, position: Int):
    pass

fn __file_size(handle: Int) -> Int:
    return 0

fn __file_exists(path: String) -> Bool:
    return False

fn __file_delete(path: String) -> Bool:
    return False

fn __file_move(source: String, dest: String) -> Bool:
    return False

fn __list_dir_impl(path: String) -> List[String]:
    return List[String]()

fn __make_dir_impl(path: String) -> Bool:
    return False

fn __remove_dir_impl(path: String) -> Bool:
    return False

fn __is_dir_impl(path: String) -> Bool:
    return False

fn __is_file_impl(path: String) -> Bool:
    return False
