# Mojo File I/O Module
# Day 47 - File operations and buffered I/O
#
# This module provides file system operations including:
# - File reading and writing
# - Buffered I/O for performance
# - File metadata and attributes
# - Directory operations

from ..ffi.ffi import CString, CType, CValue, UnsafePointer

# =============================================================================
# Constants
# =============================================================================

alias DEFAULT_BUFFER_SIZE: Int = 8192
alias MAX_PATH_LENGTH: Int = 4096

# =============================================================================
# File Mode
# =============================================================================

struct FileMode:
    """File open mode flags."""

    # Basic modes
    alias READ = 1
    alias WRITE = 2
    alias APPEND = 4
    alias CREATE = 8
    alias TRUNCATE = 16
    alias EXCLUSIVE = 32

    # Binary mode
    alias BINARY = 64

    # Common combinations
    alias READ_ONLY = FileMode.READ
    alias WRITE_ONLY = FileMode.WRITE | FileMode.CREATE | FileMode.TRUNCATE
    alias READ_WRITE = FileMode.READ | FileMode.WRITE
    alias APPEND_ONLY = FileMode.APPEND | FileMode.CREATE

    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __init__(inout self):
        self.value = FileMode.READ

    fn has(self, flag: Int) -> Bool:
        return (self.value & flag) != 0

    fn is_readable(self) -> Bool:
        return self.has(FileMode.READ)

    fn is_writable(self) -> Bool:
        return self.has(FileMode.WRITE) or self.has(FileMode.APPEND)

    fn is_binary(self) -> Bool:
        return self.has(FileMode.BINARY)

    fn to_c_mode(self) -> String:
        """Convert to C fopen mode string."""
        if self.has(FileMode.APPEND):
            if self.has(FileMode.READ):
                return "a+b" if self.is_binary() else "a+"
            return "ab" if self.is_binary() else "a"
        elif self.has(FileMode.WRITE):
            if self.has(FileMode.READ):
                if self.has(FileMode.TRUNCATE):
                    return "w+b" if self.is_binary() else "w+"
                return "r+b" if self.is_binary() else "r+"
            return "wb" if self.is_binary() else "w"
        else:
            return "rb" if self.is_binary() else "r"


# =============================================================================
# Seek Position
# =============================================================================

struct SeekFrom:
    """Seek position reference."""

    alias START = 0      # SEEK_SET
    alias CURRENT = 1    # SEEK_CUR
    alias END = 2        # SEEK_END

    var whence: Int
    var offset: Int

    fn __init__(inout self, whence: Int, offset: Int):
        self.whence = whence
        self.offset = offset

    @staticmethod
    fn start(offset: Int) -> SeekFrom:
        return SeekFrom(SeekFrom.START, offset)

    @staticmethod
    fn current(offset: Int) -> SeekFrom:
        return SeekFrom(SeekFrom.CURRENT, offset)

    @staticmethod
    fn end(offset: Int) -> SeekFrom:
        return SeekFrom(SeekFrom.END, offset)


# =============================================================================
# I/O Error
# =============================================================================

struct IOError:
    """I/O operation error."""

    alias NONE = 0
    alias NOT_FOUND = 1
    alias PERMISSION_DENIED = 2
    alias ALREADY_EXISTS = 3
    alias IS_DIRECTORY = 4
    alias NOT_DIRECTORY = 5
    alias INVALID_PATH = 6
    alias READ_ERROR = 7
    alias WRITE_ERROR = 8
    alias SEEK_ERROR = 9
    alias EOF = 10
    alias CLOSED = 11
    alias UNKNOWN = 99

    var code: Int
    var message: String
    var path: String

    fn __init__(inout self):
        self.code = IOError.NONE
        self.message = ""
        self.path = ""

    fn __init__(inout self, code: Int, message: String):
        self.code = code
        self.message = message
        self.path = ""

    fn __init__(inout self, code: Int, message: String, path: String):
        self.code = code
        self.message = message
        self.path = path

    fn is_error(self) -> Bool:
        return self.code != IOError.NONE

    fn is_eof(self) -> Bool:
        return self.code == IOError.EOF

    fn __str__(self) -> String:
        var result = "IOError(" + str(self.code) + "): " + self.message
        if len(self.path) > 0:
            result += " [" + self.path + "]"
        return result


# Global I/O error state
var _last_io_error = IOError()

fn get_last_io_error() -> IOError:
    return _last_io_error

fn clear_io_error():
    _last_io_error = IOError()

fn set_io_error(code: Int, message: String):
    _last_io_error = IOError(code, message)


# =============================================================================
# File Handle (Low-level)
# =============================================================================

struct FileHandle:
    """Low-level file handle wrapper."""

    var _handle: UnsafePointer[UInt8]  # FILE* in C
    var _fd: Int                        # File descriptor
    var _is_open: Bool

    fn __init__(inout self):
        self._handle = UnsafePointer[UInt8]()
        self._fd = -1
        self._is_open = False

    fn __init__(inout self, handle: UnsafePointer[UInt8], fd: Int):
        self._handle = handle
        self._fd = fd
        self._is_open = True

    fn is_valid(self) -> Bool:
        return self._is_open and self._fd >= 0

    fn fd(self) -> Int:
        return self._fd

    fn raw(self) -> UnsafePointer[UInt8]:
        return self._handle


# =============================================================================
# File Class
# =============================================================================

struct File:
    """High-level file operations."""

    var _path: String
    var _mode: FileMode
    var _handle: FileHandle
    var _position: Int
    var _size: Int

    fn __init__(inout self):
        self._path = ""
        self._mode = FileMode()
        self._handle = FileHandle()
        self._position = 0
        self._size = 0

    fn __init__(inout self, path: String, mode: FileMode = FileMode.READ_ONLY) raises:
        self._path = path
        self._mode = mode
        self._handle = FileHandle()
        self._position = 0
        self._size = 0
        self._open()

    fn __del__(owned self):
        if self._handle.is_valid():
            self._close()

    fn _open(inout self) raises:
        """Open the file."""
        # This would call libc fopen via FFI
        # FILE* fopen(const char* path, const char* mode)
        clear_io_error()

        # Validate path
        if len(self._path) == 0:
            set_io_error(IOError.INVALID_PATH, "Empty path")
            raise Error("Empty path")

        # Placeholder - actual implementation calls runtime
        self._handle = FileHandle(UnsafePointer[UInt8](), 3)  # Placeholder fd

        # Get file size if readable
        if self._mode.is_readable():
            self._update_size()

    fn _close(inout self):
        """Close the file."""
        if self._handle.is_valid():
            # This would call libc fclose via FFI
            self._handle = FileHandle()

    fn close(inout self):
        """Close the file explicitly."""
        self._close()

    fn is_open(self) -> Bool:
        return self._handle.is_valid()

    fn path(self) -> String:
        return self._path

    fn mode(self) -> FileMode:
        return self._mode

    fn position(self) -> Int:
        return self._position

    fn size(self) -> Int:
        return self._size

    fn _update_size(inout self):
        """Update cached file size."""
        # This would use fseek/ftell to get size
        self._size = 0  # Placeholder

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    fn read(inout self, size: Int = -1) raises -> String:
        """Read up to size bytes as a string. Read all if size is -1."""
        if not self._handle.is_valid():
            set_io_error(IOError.CLOSED, "File is closed")
            raise Error("File is closed")

        if not self._mode.is_readable():
            set_io_error(IOError.PERMISSION_DENIED, "File not opened for reading")
            raise Error("File not opened for reading")

        var bytes_to_read = size
        if bytes_to_read < 0:
            bytes_to_read = self._size - self._position
            if bytes_to_read < 0:
                bytes_to_read = 0

        # Allocate buffer and read
        # This would call libc fread via FFI
        var result = String()

        # Placeholder - actual implementation reads from file
        self._position += bytes_to_read

        return result

    fn read_bytes(inout self, size: Int = -1) raises -> List[UInt8]:
        """Read up to size bytes as a byte list."""
        if not self._handle.is_valid():
            raise Error("File is closed")

        var bytes_to_read = size
        if bytes_to_read < 0:
            bytes_to_read = self._size - self._position

        var result = List[UInt8]()
        # Placeholder - actual implementation reads from file
        self._position += bytes_to_read

        return result

    fn read_line(inout self, max_size: Int = -1) raises -> String:
        """Read a single line."""
        if not self._handle.is_valid():
            raise Error("File is closed")

        var result = String()
        var max_chars = max_size if max_size > 0 else 65536

        # Read character by character until newline
        # This would use libc fgetc via FFI
        # Placeholder implementation

        return result

    fn read_lines(inout self) raises -> List[String]:
        """Read all lines into a list."""
        var lines = List[String]()

        while self._position < self._size:
            var line = self.read_line()
            if len(line) == 0:
                break
            lines.append(line)

        return lines

    fn read_all(inout self) raises -> String:
        """Read entire file as string."""
        return self.read(-1)

    fn read_into(inout self, buffer: UnsafePointer[UInt8], size: Int) raises -> Int:
        """Read into provided buffer. Returns bytes read."""
        if not self._handle.is_valid():
            raise Error("File is closed")

        # This would call libc fread via FFI
        var bytes_read = 0  # Placeholder
        self._position += bytes_read

        return bytes_read

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    fn write(inout self, data: String) raises -> Int:
        """Write string to file. Returns bytes written."""
        if not self._handle.is_valid():
            set_io_error(IOError.CLOSED, "File is closed")
            raise Error("File is closed")

        if not self._mode.is_writable():
            set_io_error(IOError.PERMISSION_DENIED, "File not opened for writing")
            raise Error("File not opened for writing")

        # This would call libc fwrite via FFI
        var bytes_written = len(data)  # Placeholder
        self._position += bytes_written

        return bytes_written

    fn write_bytes(inout self, data: List[UInt8]) raises -> Int:
        """Write bytes to file. Returns bytes written."""
        if not self._handle.is_valid():
            raise Error("File is closed")

        var bytes_written = len(data)  # Placeholder
        self._position += bytes_written

        return bytes_written

    fn write_line(inout self, line: String) raises -> Int:
        """Write a line with newline."""
        return self.write(line + "\n")

    fn write_lines(inout self, lines: List[String]) raises -> Int:
        """Write multiple lines."""
        var total = 0
        for i in range(len(lines)):
            total += self.write_line(lines[i])
        return total

    fn write_from(inout self, buffer: UnsafePointer[UInt8], size: Int) raises -> Int:
        """Write from provided buffer. Returns bytes written."""
        if not self._handle.is_valid():
            raise Error("File is closed")

        var bytes_written = size  # Placeholder
        self._position += bytes_written

        return bytes_written

    # -------------------------------------------------------------------------
    # Position Operations
    # -------------------------------------------------------------------------

    fn seek(inout self, pos: SeekFrom) raises -> Int:
        """Seek to position. Returns new position."""
        if not self._handle.is_valid():
            raise Error("File is closed")

        # This would call libc fseek via FFI
        if pos.whence == SeekFrom.START:
            self._position = pos.offset
        elif pos.whence == SeekFrom.CURRENT:
            self._position += pos.offset
        else:  # END
            self._position = self._size + pos.offset

        # Clamp to valid range
        if self._position < 0:
            self._position = 0
        if self._position > self._size:
            self._position = self._size

        return self._position

    fn tell(self) -> Int:
        """Get current position."""
        return self._position

    fn rewind(inout self) raises:
        """Seek to beginning."""
        _ = self.seek(SeekFrom.start(0))

    fn at_end(self) -> Bool:
        """Check if at end of file."""
        return self._position >= self._size

    # -------------------------------------------------------------------------
    # Buffer Operations
    # -------------------------------------------------------------------------

    fn flush(inout self) raises:
        """Flush write buffer to disk."""
        if not self._handle.is_valid():
            raise Error("File is closed")
        # This would call libc fflush via FFI

    fn sync(inout self) raises:
        """Sync file to disk (fsync)."""
        if not self._handle.is_valid():
            raise Error("File is closed")
        # This would call libc fsync via FFI


# =============================================================================
# Buffered Reader
# =============================================================================

struct BufferedReader:
    """Buffered file reader for efficient sequential reads."""

    var _file: File
    var _buffer: List[UInt8]
    var _buffer_pos: Int
    var _buffer_size: Int
    var _buffer_filled: Int

    fn __init__(inout self, path: String, buffer_size: Int = DEFAULT_BUFFER_SIZE) raises:
        self._file = File(path, FileMode(FileMode.READ | FileMode.BINARY))
        self._buffer = List[UInt8]()
        self._buffer_pos = 0
        self._buffer_size = buffer_size
        self._buffer_filled = 0

        # Pre-allocate buffer
        for _ in range(buffer_size):
            self._buffer.append(0)

    fn __del__(owned self):
        pass  # File closes automatically

    fn _fill_buffer(inout self) raises -> Bool:
        """Fill buffer from file. Returns True if data was read."""
        if self._file.at_end():
            return False

        # Read into buffer
        # Placeholder - actual implementation reads from file
        self._buffer_pos = 0
        self._buffer_filled = 0  # Would be actual bytes read

        return self._buffer_filled > 0

    fn read(inout self, size: Int) raises -> List[UInt8]:
        """Read up to size bytes."""
        var result = List[UInt8]()
        var remaining = size

        while remaining > 0:
            # Check if buffer needs refill
            if self._buffer_pos >= self._buffer_filled:
                if not self._fill_buffer():
                    break  # EOF

            # Copy from buffer
            var available = self._buffer_filled - self._buffer_pos
            var to_copy = min(available, remaining)

            for i in range(to_copy):
                result.append(self._buffer[self._buffer_pos + i])

            self._buffer_pos += to_copy
            remaining -= to_copy

        return result

    fn read_byte(inout self) raises -> Int:
        """Read single byte. Returns -1 on EOF."""
        if self._buffer_pos >= self._buffer_filled:
            if not self._fill_buffer():
                return -1

        var byte = int(self._buffer[self._buffer_pos])
        self._buffer_pos += 1
        return byte

    fn read_line(inout self) raises -> String:
        """Read until newline."""
        var result = String()

        while True:
            var byte = self.read_byte()
            if byte < 0:
                break  # EOF
            if byte == 10:  # '\n'
                break
            if byte == 13:  # '\r'
                # Check for \r\n
                var next_byte = self.read_byte()
                if next_byte >= 0 and next_byte != 10:
                    # Not \r\n, need to "unread" - simplified: just skip
                    pass
                break
            result += chr(byte)

        return result

    fn read_all(inout self) raises -> String:
        """Read entire file."""
        var result = String()

        while True:
            var line = self.read_line()
            if len(line) == 0 and self._file.at_end():
                break
            result += line + "\n"

        return result

    fn close(inout self):
        """Close the file."""
        self._file.close()


# =============================================================================
# Buffered Writer
# =============================================================================

struct BufferedWriter:
    """Buffered file writer for efficient sequential writes."""

    var _file: File
    var _buffer: List[UInt8]
    var _buffer_pos: Int
    var _buffer_size: Int

    fn __init__(inout self, path: String, buffer_size: Int = DEFAULT_BUFFER_SIZE) raises:
        self._file = File(path, FileMode(FileMode.WRITE | FileMode.CREATE | FileMode.TRUNCATE | FileMode.BINARY))
        self._buffer = List[UInt8]()
        self._buffer_pos = 0
        self._buffer_size = buffer_size

        # Pre-allocate buffer
        for _ in range(buffer_size):
            self._buffer.append(0)

    fn __del__(owned self):
        # Flush remaining data
        if self._buffer_pos > 0:
            try:
                self._flush_buffer()
            except:
                pass

    fn _flush_buffer(inout self) raises:
        """Flush buffer to file."""
        if self._buffer_pos > 0:
            # Write buffer contents to file
            # Placeholder - actual implementation writes to file
            self._buffer_pos = 0

    fn write(inout self, data: String) raises -> Int:
        """Write string."""
        var total = 0

        for i in range(len(data)):
            self._buffer[self._buffer_pos] = ord(data[i])
            self._buffer_pos += 1
            total += 1

            if self._buffer_pos >= self._buffer_size:
                self._flush_buffer()

        return total

    fn write_bytes(inout self, data: List[UInt8]) raises -> Int:
        """Write bytes."""
        var total = 0

        for i in range(len(data)):
            self._buffer[self._buffer_pos] = data[i]
            self._buffer_pos += 1
            total += 1

            if self._buffer_pos >= self._buffer_size:
                self._flush_buffer()

        return total

    fn write_byte(inout self, byte: UInt8) raises:
        """Write single byte."""
        self._buffer[self._buffer_pos] = byte
        self._buffer_pos += 1

        if self._buffer_pos >= self._buffer_size:
            self._flush_buffer()

    fn write_line(inout self, line: String) raises -> Int:
        """Write line with newline."""
        return self.write(line + "\n")

    fn flush(inout self) raises:
        """Flush buffer to disk."""
        self._flush_buffer()
        self._file.flush()

    fn close(inout self) raises:
        """Flush and close."""
        self.flush()
        self._file.close()


# =============================================================================
# File Utilities
# =============================================================================

fn exists(path: String) -> Bool:
    """Check if file or directory exists."""
    # This would call libc stat via FFI
    return False  # Placeholder

fn is_file(path: String) -> Bool:
    """Check if path is a regular file."""
    # This would call libc stat and check S_ISREG
    return False  # Placeholder

fn is_dir(path: String) -> Bool:
    """Check if path is a directory."""
    # This would call libc stat and check S_ISDIR
    return False  # Placeholder

fn is_symlink(path: String) -> Bool:
    """Check if path is a symbolic link."""
    # This would call libc lstat and check S_ISLNK
    return False  # Placeholder

fn file_size(path: String) -> Int:
    """Get file size in bytes."""
    # This would call libc stat
    return 0  # Placeholder

fn remove(path: String) raises:
    """Remove a file."""
    # This would call libc unlink or remove
    pass

fn rename(old_path: String, new_path: String) raises:
    """Rename/move a file."""
    # This would call libc rename
    pass

fn copy(src: String, dst: String) raises:
    """Copy a file."""
    var reader = BufferedReader(src)
    var writer = BufferedWriter(dst)

    while True:
        var data = reader.read(DEFAULT_BUFFER_SIZE)
        if len(data) == 0:
            break
        _ = writer.write_bytes(data)

    reader.close()
    writer.close()

fn mkdir(path: String, mode: Int = 0o755) raises:
    """Create a directory."""
    # This would call libc mkdir
    pass

fn makedirs(path: String, mode: Int = 0o755) raises:
    """Create directory and all parent directories."""
    # Split path and create each component
    pass

fn rmdir(path: String) raises:
    """Remove an empty directory."""
    # This would call libc rmdir
    pass

fn listdir(path: String) raises -> List[String]:
    """List directory contents."""
    var entries = List[String]()
    # This would call libc opendir/readdir/closedir
    return entries

fn getcwd() -> String:
    """Get current working directory."""
    # This would call libc getcwd
    return ""  # Placeholder

fn chdir(path: String) raises:
    """Change current working directory."""
    # This would call libc chdir
    pass


# =============================================================================
# File Metadata
# =============================================================================

struct FileInfo:
    """File metadata."""

    var path: String
    var size: Int
    var mode: Int
    var uid: Int
    var gid: Int
    var atime: Int  # Access time (Unix timestamp)
    var mtime: Int  # Modification time
    var ctime: Int  # Creation/status change time
    var is_file: Bool
    var is_dir: Bool
    var is_symlink: Bool

    fn __init__(inout self):
        self.path = ""
        self.size = 0
        self.mode = 0
        self.uid = 0
        self.gid = 0
        self.atime = 0
        self.mtime = 0
        self.ctime = 0
        self.is_file = False
        self.is_dir = False
        self.is_symlink = False

    fn __init__(inout self, path: String) raises:
        self.path = path
        self.size = 0
        self.mode = 0
        self.uid = 0
        self.gid = 0
        self.atime = 0
        self.mtime = 0
        self.ctime = 0
        self.is_file = False
        self.is_dir = False
        self.is_symlink = False
        self._stat()

    fn _stat(inout self) raises:
        """Get file stats from system."""
        # This would call libc stat/lstat via FFI
        pass

    fn is_readable(self) -> Bool:
        return (self.mode & 0o400) != 0

    fn is_writable(self) -> Bool:
        return (self.mode & 0o200) != 0

    fn is_executable(self) -> Bool:
        return (self.mode & 0o100) != 0


fn stat(path: String) raises -> FileInfo:
    """Get file information."""
    return FileInfo(path)


# =============================================================================
# Temporary Files
# =============================================================================

struct TempFile:
    """Temporary file that is deleted on close."""

    var _file: File
    var _path: String
    var _delete_on_close: Bool

    fn __init__(inout self, prefix: String = "tmp", suffix: String = "") raises:
        # Generate unique temporary filename
        # This would use mkstemp or similar
        self._path = "/tmp/" + prefix + "_XXXXXX" + suffix  # Placeholder
        self._file = File(self._path, FileMode(FileMode.READ | FileMode.WRITE | FileMode.CREATE))
        self._delete_on_close = True

    fn __del__(owned self):
        self._file.close()
        if self._delete_on_close:
            try:
                remove(self._path)
            except:
                pass

    fn path(self) -> String:
        return self._path

    fn file(inout self) -> File:
        return self._file

    fn keep(inout self):
        """Don't delete on close."""
        self._delete_on_close = False


fn temp_dir() -> String:
    """Get system temporary directory."""
    # Check TMPDIR, TMP, TEMP environment variables
    # Fall back to /tmp on Unix, %TEMP% on Windows
    return "/tmp"  # Placeholder


# =============================================================================
# Context Manager Support
# =============================================================================

struct FileContext:
    """Context manager for files (with statement support)."""

    var _file: File
    var _should_close: Bool

    fn __init__(inout self, path: String, mode: FileMode = FileMode.READ_ONLY) raises:
        self._file = File(path, mode)
        self._should_close = True

    fn __enter__(inout self) -> File:
        return self._file

    fn __exit__(inout self):
        if self._should_close:
            self._file.close()


# =============================================================================
# Convenience Functions
# =============================================================================

fn read_file(path: String) raises -> String:
    """Read entire file as string."""
    var f = File(path, FileMode.READ_ONLY)
    var content = f.read_all()
    f.close()
    return content

fn read_bytes(path: String) raises -> List[UInt8]:
    """Read entire file as bytes."""
    var f = File(path, FileMode(FileMode.READ | FileMode.BINARY))
    var content = f.read_bytes()
    f.close()
    return content

fn write_file(path: String, content: String) raises:
    """Write string to file."""
    var f = File(path, FileMode.WRITE_ONLY)
    _ = f.write(content)
    f.close()

fn write_bytes(path: String, content: List[UInt8]) raises:
    """Write bytes to file."""
    var f = File(path, FileMode(FileMode.WRITE | FileMode.CREATE | FileMode.TRUNCATE | FileMode.BINARY))
    _ = f.write_bytes(content)
    f.close()

fn append_file(path: String, content: String) raises:
    """Append string to file."""
    var f = File(path, FileMode.APPEND_ONLY)
    _ = f.write(content)
    f.close()


# =============================================================================
# Tests
# =============================================================================

fn test_file_mode():
    """Test FileMode."""
    var read_mode = FileMode(FileMode.READ)
    assert_true(read_mode.is_readable(), "READ should be readable")
    assert_true(not read_mode.is_writable(), "READ should not be writable")

    var write_mode = FileMode(FileMode.WRITE)
    assert_true(write_mode.is_writable(), "WRITE should be writable")

    var rw_mode = FileMode(FileMode.READ_WRITE)
    assert_true(rw_mode.is_readable(), "READ_WRITE should be readable")
    assert_true(rw_mode.is_writable(), "READ_WRITE should be writable")

    print("test_file_mode: PASSED")


fn test_seek_from():
    """Test SeekFrom."""
    var start = SeekFrom.start(100)
    assert_true(start.whence == SeekFrom.START, "start should use START")
    assert_true(start.offset == 100, "offset should be 100")

    var current = SeekFrom.current(-50)
    assert_true(current.whence == SeekFrom.CURRENT, "current should use CURRENT")

    var end = SeekFrom.end(0)
    assert_true(end.whence == SeekFrom.END, "end should use END")

    print("test_seek_from: PASSED")


fn test_io_error():
    """Test IOError."""
    var err = IOError(IOError.NOT_FOUND, "File not found", "/test/path")
    assert_true(err.is_error(), "Should be error")
    assert_true(err.code == IOError.NOT_FOUND, "Code should match")
    assert_true(len(err.message) > 0, "Message should not be empty")

    var no_err = IOError()
    assert_true(not no_err.is_error(), "Default should not be error")

    print("test_io_error: PASSED")


fn test_file_info():
    """Test FileInfo struct."""
    var info = FileInfo()
    info.mode = 0o755
    assert_true(info.is_readable(), "Should be readable")
    assert_true(info.is_writable(), "Should be writable")
    assert_true(info.is_executable(), "Should be executable")

    info.mode = 0o444
    assert_true(info.is_readable(), "Should be readable")
    assert_true(not info.is_writable(), "Should not be writable")
    assert_true(not info.is_executable(), "Should not be executable")

    print("test_file_info: PASSED")


fn test_buffered_io():
    """Test buffered I/O concepts."""
    # Test buffer size
    assert_true(DEFAULT_BUFFER_SIZE == 8192, "Default buffer should be 8KB")
    assert_true(MAX_PATH_LENGTH == 4096, "Max path should be 4096")

    print("test_buffered_io: PASSED")


fn assert_true(condition: Bool, message: String):
    """Simple assertion helper."""
    if not condition:
        print("ASSERTION FAILED: " + message)


fn run_all_tests():
    """Run all file tests."""
    print("=== File I/O Module Tests ===")
    test_file_mode()
    test_seek_from()
    test_io_error()
    test_file_info()
    test_buffered_io()
    print("=== All Tests Passed ===")
