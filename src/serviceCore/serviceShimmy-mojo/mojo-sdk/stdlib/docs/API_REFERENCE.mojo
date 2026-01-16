"""
Mojo Standard Library - API Reference Documentation

This documentation covers the Phase 2 I/O & Networking modules:
- stdlib/ffi/ffi.mojo - Foreign Function Interface
- stdlib/io/file.mojo - File Operations
- stdlib/io/network.mojo - Networking & HTTP
- stdlib/io/json.mojo - JSON Parsing & Serialization
- stdlib/time/time.mojo - Date/Time Handling
- stdlib/sys/path.mojo - Path Manipulation

================================================================================
TABLE OF CONTENTS
================================================================================

1. FFI Module (ffi.mojo)
   1.1 C Types
   1.2 C Values
   1.3 External Functions
   1.4 Dynamic Libraries
   1.5 Struct Marshalling

2. File Module (file.mojo)
   2.1 File Modes
   2.2 File Operations
   2.3 Buffered I/O
   2.4 Directory Operations
   2.5 File Utilities

3. Network Module (network.mojo)
   3.1 IP Addresses
   3.2 Sockets
   3.3 TCP Client/Server
   3.4 UDP Sockets
   3.5 HTTP Client

4. JSON Module (json.mojo)
   4.1 JSON Values
   4.2 JSON Parser
   4.3 JSON Serialization
   4.4 JSON Builder

5. Time Module (time.mojo)
   5.1 Duration
   5.2 Date
   5.3 Time
   5.4 DateTime
   5.5 Timezone

6. Path Module (path.mojo)
   6.1 Path Operations
   6.2 Path Components
   6.3 Glob Patterns
   6.4 Path Builder

================================================================================
1. FFI MODULE (stdlib/ffi/ffi.mojo)
================================================================================

The FFI module provides seamless interoperability with C libraries.

------------------------------------------------------------------------------
1.1 C Types (CType)
------------------------------------------------------------------------------

CType defines C-compatible primitive types for FFI operations.

    alias VOID      # void
    alias CHAR      # char (8-bit)
    alias SCHAR     # signed char
    alias UCHAR     # unsigned char
    alias SHORT     # short (16-bit)
    alias USHORT    # unsigned short
    alias INT       # int (32-bit)
    alias UINT      # unsigned int
    alias LONG      # long (32/64-bit platform dependent)
    alias ULONG     # unsigned long
    alias LONGLONG  # long long (64-bit)
    alias ULONGLONG # unsigned long long
    alias FLOAT     # float (32-bit)
    alias DOUBLE    # double (64-bit)
    alias SIZE_T    # size_t
    alias SSIZE_T   # ssize_t
    alias INTPTR    # intptr_t
    alias UINTPTR   # uintptr_t

Example:
    let type_info = CType.INT
    print(type_info.name())       # "int"
    print(type_info.size())       # 4
    print(type_info.alignment())  # 4

------------------------------------------------------------------------------
1.2 C Values (CValue)
------------------------------------------------------------------------------

CValue wraps values for C interop with automatic type conversion.

Constructors:
    CValue.from_int(value: Int) -> CValue
    CValue.from_float(value: Float64) -> CValue
    CValue.from_bool(value: Bool) -> CValue
    CValue.from_pointer(ptr: Int) -> CValue
    CValue.null() -> CValue

Accessors:
    fn as_int(self) -> Int
    fn as_float(self) -> Float64
    fn as_bool(self) -> Bool
    fn as_pointer(self) -> Int
    fn is_null(self) -> Bool

Example:
    let val = CValue.from_int(42)
    let ptr = CValue.from_pointer(0x7fff1234)
    let null_val = CValue.null()

    if not null_val.is_null():
        print(val.as_int())  # 42

------------------------------------------------------------------------------
1.3 C Strings (CString)
------------------------------------------------------------------------------

CString handles null-terminated C strings with automatic memory management.

Constructors:
    CString(s: String) -> CString
    CString.from_ptr(ptr: Int, length: Int) -> CString

Methods:
    fn as_ptr(self) -> Int              # Get raw pointer
    fn length(self) -> Int              # String length (excluding null)
    fn to_string(self) -> String        # Convert to Mojo String
    fn is_null(self) -> Bool            # Check if null pointer

Static Methods:
    CString.null() -> CString           # Create null CString

Example:
    let cstr = CString("Hello, C!")
    let ptr = cstr.as_ptr()
    let back = cstr.to_string()
    print(back)  # "Hello, C!"

------------------------------------------------------------------------------
1.4 External Functions (ExternalFunction)
------------------------------------------------------------------------------

ExternalFunction wraps C function pointers for safe calling.

Constructor:
    ExternalFunction(name: String, address: Int, signature: FunctionSignature)

Methods:
    fn call(self, args: List[CValue]) -> CValue
    fn name(self) -> String
    fn address(self) -> Int

FunctionSignature Builder:
    var sig = FunctionSignature()
    sig.return_type(CType.INT)
    sig.add_param(CType.POINTER)
    sig.add_param(CType.SIZE_T)
    sig.set_variadic(False)

Example:
    # Wrap C's strlen function
    var sig = FunctionSignature()
    sig.return_type(CType.SIZE_T)
    sig.add_param(CType.POINTER)

    let strlen_fn = ExternalFunction("strlen", strlen_addr, sig)
    let result = strlen_fn.call([CValue.from_pointer(str_ptr)])
    print("Length:", result.as_int())

------------------------------------------------------------------------------
1.5 Dynamic Libraries (DynamicLibrary)
------------------------------------------------------------------------------

DynamicLibrary loads shared libraries (.so, .dylib, .dll) at runtime.

Constructors:
    DynamicLibrary(path: String) -> DynamicLibrary
    DynamicLibrary.load(path: String) -> Result[DynamicLibrary, FFIError]

Methods:
    fn get_symbol(self, name: String) -> Int    # Get symbol address
    fn has_symbol(self, name: String) -> Bool   # Check if symbol exists
    fn close(inout self)                        # Unload library
    fn is_loaded(self) -> Bool                  # Check if loaded
    fn path(self) -> String                     # Get library path

Example:
    let lib = DynamicLibrary("/usr/lib/libm.so")
    if lib.is_loaded():
        let cos_addr = lib.get_symbol("cos")
        if cos_addr != 0:
            # Call cos function...
            pass
        lib.close()

------------------------------------------------------------------------------
1.6 Struct Marshalling (CStruct)
------------------------------------------------------------------------------

CStructDef defines C struct layouts for marshalling.

CStructDef Methods:
    fn add_field(inout self, name: String, type: CType)
    fn size(self) -> Int
    fn alignment(self) -> Int
    fn field_offset(self, name: String) -> Int

CStruct Methods:
    fn set_int(inout self, field: String, value: Int)
    fn set_float(inout self, field: String, value: Float64)
    fn set_pointer(inout self, field: String, ptr: Int)
    fn get_int(self, field: String) -> Int
    fn get_float(self, field: String) -> Float64
    fn get_pointer(self, field: String) -> Int

Example:
    # Define a C struct: struct Point { int x; int y; }
    var point_def = CStructDef("Point")
    point_def.add_field("x", CType.INT)
    point_def.add_field("y", CType.INT)

    var point = CStruct(point_def)
    point.set_int("x", 10)
    point.set_int("y", 20)

    print(point.get_int("x"))  # 10

================================================================================
2. FILE MODULE (stdlib/io/file.mojo)
================================================================================

The File module provides comprehensive file system operations.

------------------------------------------------------------------------------
2.1 File Modes (FileMode)
------------------------------------------------------------------------------

FileMode flags control how files are opened.

    FileMode.READ       # Open for reading
    FileMode.WRITE      # Open for writing
    FileMode.APPEND     # Append to existing content
    FileMode.CREATE     # Create if doesn't exist
    FileMode.TRUNCATE   # Truncate existing content
    FileMode.BINARY     # Binary mode (no text conversion)

Modes can be combined with |:
    let mode = FileMode.READ | FileMode.WRITE | FileMode.CREATE

------------------------------------------------------------------------------
2.2 File Operations (File)
------------------------------------------------------------------------------

File class for reading and writing files.

Opening Files:
    File.open(path: String, mode: Int) -> Result[File, IOError]

    # Convenience constructors
    File.open_read(path: String) -> Result[File, IOError]
    File.open_write(path: String) -> Result[File, IOError]
    File.open_append(path: String) -> Result[File, IOError]

Reading:
    fn read(inout self, size: Int) -> String     # Read up to size bytes
    fn read_all(inout self) -> String            # Read entire file
    fn read_line(inout self) -> String           # Read single line
    fn read_lines(inout self) -> List[String]   # Read all lines

Writing:
    fn write(inout self, data: String) -> Int    # Write data, return bytes written
    fn write_line(inout self, line: String)      # Write line with newline
    fn flush(inout self)                         # Flush buffers to disk

Seeking:
    fn seek(inout self, offset: Int, whence: SeekFrom) -> Int
    fn tell(self) -> Int                         # Current position
    fn rewind(inout self)                        # Seek to beginning

Control:
    fn close(inout self)
    fn is_open(self) -> Bool

Example:
    # Write to file
    var file = File.open_write("/tmp/example.txt").unwrap()
    file.write("Hello, World!\n")
    file.write_line("Line 2")
    file.close()

    # Read from file
    var reader = File.open_read("/tmp/example.txt").unwrap()
    let content = reader.read_all()
    reader.close()
    print(content)

------------------------------------------------------------------------------
2.3 Buffered I/O
------------------------------------------------------------------------------

BufferedReader - Efficient sequential reading with internal buffer.

    BufferedReader(file: File, buffer_size: Int = 8192)

    fn read_line(inout self) -> String
    fn read_lines(inout self) -> List[String]
    fn read_until(inout self, delimiter: String) -> String
    fn peek(self, size: Int) -> String
    fn is_eof(self) -> Bool

BufferedWriter - Efficient sequential writing with internal buffer.

    BufferedWriter(file: File, buffer_size: Int = 8192)

    fn write(inout self, data: String)
    fn write_line(inout self, line: String)
    fn flush(inout self)

Example:
    var file = File.open_read("large_file.txt").unwrap()
    var reader = BufferedReader(file, 16384)

    while not reader.is_eof():
        let line = reader.read_line()
        process(line)

------------------------------------------------------------------------------
2.4 Directory Operations
------------------------------------------------------------------------------

Functions for directory manipulation:

    mkdir(path: String) -> Bool                  # Create directory
    makedirs(path: String) -> Bool               # Create nested directories
    rmdir(path: String) -> Bool                  # Remove empty directory
    listdir(path: String) -> List[String]        # List directory contents

    exists(path: String) -> Bool                 # Check if path exists
    is_file(path: String) -> Bool                # Check if regular file
    is_dir(path: String) -> Bool                 # Check if directory

Example:
    makedirs("/tmp/project/src/modules")

    for entry in listdir("/tmp/project"):
        if is_dir("/tmp/project/" + entry):
            print("DIR: " + entry)
        else:
            print("FILE: " + entry)

------------------------------------------------------------------------------
2.5 File Utilities
------------------------------------------------------------------------------

Convenience functions for common operations:

    read_file(path: String) -> String            # Read entire file
    write_file(path: String, content: String)    # Write entire file
    append_file(path: String, content: String)   # Append to file

    copy(src: String, dst: String) -> Bool       # Copy file
    rename(src: String, dst: String) -> Bool     # Rename/move file
    remove(path: String) -> Bool                 # Delete file

    file_size(path: String) -> Int               # Get file size

FileInfo - File metadata:

    FileInfo.from_path(path: String) -> FileInfo

    fn size(self) -> Int
    fn is_file(self) -> Bool
    fn is_dir(self) -> Bool
    fn is_symlink(self) -> Bool
    fn created(self) -> DateTime
    fn modified(self) -> DateTime
    fn accessed(self) -> DateTime

TempFile - Temporary file with automatic cleanup:

    TempFile.create() -> TempFile
    TempFile.create_in(dir: String) -> TempFile

    fn path(self) -> String
    fn write(inout self, data: String)
    fn read_all(self) -> String

Example:
    # Quick file operations
    write_file("/tmp/config.json", '{"key": "value"}')
    let config = read_file("/tmp/config.json")

    # File info
    let info = FileInfo.from_path("/tmp/config.json")
    print("Size:", info.size(), "bytes")
    print("Modified:", info.modified().to_iso_string())

================================================================================
3. NETWORK MODULE (stdlib/io/network.mojo)
================================================================================

The Network module provides TCP/UDP sockets and HTTP client functionality.

------------------------------------------------------------------------------
3.1 IP Addresses
------------------------------------------------------------------------------

IPv4Address:
    IPv4Address(a: Int, b: Int, c: Int, d: Int)
    IPv4Address.parse(s: String) -> IPv4Address
    IPv4Address.localhost() -> IPv4Address       # 127.0.0.1
    IPv4Address.any() -> IPv4Address             # 0.0.0.0
    IPv4Address.broadcast() -> IPv4Address       # 255.255.255.255

    fn to_string(self) -> String
    fn is_loopback(self) -> Bool
    fn is_private(self) -> Bool
    fn is_multicast(self) -> Bool

IPv6Address:
    IPv6Address.parse(s: String) -> IPv6Address
    IPv6Address.localhost() -> IPv6Address       # ::1
    IPv6Address.any() -> IPv6Address             # ::

    fn to_string(self) -> String

SocketAddress - IP + Port combination:
    SocketAddress(ip: IPv4Address, port: Int)
    SocketAddress.parse(s: String) -> SocketAddress  # "192.168.1.1:8080"

    fn ip(self) -> IPv4Address
    fn port(self) -> Int
    fn to_string(self) -> String

Example:
    let addr = IPv4Address.parse("192.168.1.100")
    let socket_addr = SocketAddress(addr, 8080)
    print(socket_addr.to_string())  # "192.168.1.100:8080"

------------------------------------------------------------------------------
3.2 Socket Base Class
------------------------------------------------------------------------------

Socket options and common functionality:

    fn set_timeout(inout self, timeout_ms: Int)
    fn set_reuse_address(inout self, reuse: Bool)
    fn set_keepalive(inout self, enabled: Bool)
    fn set_nodelay(inout self, enabled: Bool)     # TCP_NODELAY
    fn set_buffer_size(inout self, send: Int, recv: Int)

    fn local_address(self) -> SocketAddress
    fn is_connected(self) -> Bool
    fn close(inout self)

------------------------------------------------------------------------------
3.3 TCP Client/Server
------------------------------------------------------------------------------

TcpSocket - TCP client socket:

    TcpSocket.connect(address: SocketAddress) -> Result[TcpSocket, NetworkError]
    TcpSocket.connect(host: String, port: Int) -> Result[TcpSocket, NetworkError]

    fn send(inout self, data: String) -> Int      # Returns bytes sent
    fn recv(inout self, size: Int) -> String      # Receive up to size bytes
    fn send_all(inout self, data: String)         # Send all data
    fn recv_exact(inout self, size: Int) -> String # Receive exactly size bytes

TcpListener - TCP server socket:

    TcpListener.bind(address: SocketAddress) -> Result[TcpListener, NetworkError]
    TcpListener.bind(host: String, port: Int) -> Result[TcpListener, NetworkError]

    fn listen(inout self, backlog: Int = 128)
    fn accept(inout self) -> Result[TcpSocket, NetworkError]
    fn local_address(self) -> SocketAddress

Example - TCP Client:
    let socket = TcpSocket.connect("example.com", 80).unwrap()
    socket.send_all("GET / HTTP/1.0\r\nHost: example.com\r\n\r\n")
    let response = socket.recv(4096)
    socket.close()
    print(response)

Example - TCP Server:
    let listener = TcpListener.bind("0.0.0.0", 8080).unwrap()
    listener.listen()

    while True:
        let client = listener.accept().unwrap()
        let request = client.recv(1024)
        client.send_all("HTTP/1.0 200 OK\r\n\r\nHello!")
        client.close()

------------------------------------------------------------------------------
3.4 UDP Sockets
------------------------------------------------------------------------------

UdpSocket - Connectionless UDP communication:

    UdpSocket.bind(address: SocketAddress) -> Result[UdpSocket, NetworkError]
    UdpSocket.bind(host: String, port: Int) -> Result[UdpSocket, NetworkError]

    fn send_to(inout self, data: String, address: SocketAddress) -> Int
    fn recv_from(inout self, size: Int) -> Tuple[String, SocketAddress]
    fn connect(inout self, address: SocketAddress)  # Set default destination
    fn send(inout self, data: String) -> Int        # Send to connected address
    fn recv(inout self, size: Int) -> String        # Receive from any

Example:
    let socket = UdpSocket.bind("0.0.0.0", 5000).unwrap()

    # Send to specific address
    let target = SocketAddress(IPv4Address.parse("192.168.1.10"), 5001)
    socket.send_to("Hello UDP!", target)

    # Receive from anyone
    let (data, sender) = socket.recv_from(1024)
    print("Received from", sender.to_string(), ":", data)

------------------------------------------------------------------------------
3.5 HTTP Client
------------------------------------------------------------------------------

URL - Parse and manipulate URLs:

    URL.parse(s: String) -> URL

    fn scheme(self) -> String       # "http", "https"
    fn host(self) -> String         # "example.com"
    fn port(self) -> Int            # 80, 443, etc.
    fn path(self) -> String         # "/api/users"
    fn query(self) -> String        # "page=1&limit=10"
    fn fragment(self) -> String     # "section1"
    fn to_string(self) -> String

HttpHeaders - HTTP header collection:

    HttpHeaders()

    fn set(inout self, name: String, value: String)
    fn get(self, name: String) -> String
    fn has(self, name: String) -> Bool
    fn remove(inout self, name: String)
    fn to_string(self) -> String

HttpClient - Make HTTP requests:

    HttpClient()
    HttpClient.with_timeout(timeout_ms: Int)

    fn get(self, url: String) -> HttpResponse
    fn post(self, url: String, body: String) -> HttpResponse
    fn put(self, url: String, body: String) -> HttpResponse
    fn delete(self, url: String) -> HttpResponse

    fn set_header(inout self, name: String, value: String)
    fn set_timeout(inout self, timeout_ms: Int)

HttpResponse - HTTP response data:

    fn status_code(self) -> Int
    fn status_text(self) -> String
    fn headers(self) -> HttpHeaders
    fn body(self) -> String
    fn is_success(self) -> Bool      # 200-299
    fn is_redirect(self) -> Bool     # 300-399
    fn is_error(self) -> Bool        # 400+

Example:
    var client = HttpClient()
    client.set_header("User-Agent", "MojoApp/1.0")
    client.set_header("Accept", "application/json")

    # GET request
    let response = client.get("https://api.example.com/users")
    if response.is_success():
        let data = response.body()
        print("Users:", data)

    # POST request
    let post_response = client.post(
        "https://api.example.com/users",
        '{"name": "Alice", "email": "alice@example.com"}'
    )
    print("Created:", post_response.status_code())

DNS Resolution:

    resolve_host(hostname: String) -> IPv4Address
    resolve_all(hostname: String) -> List[IPv4Address]

Example:
    let ip = resolve_host("example.com")
    print("example.com resolves to:", ip.to_string())

    let all_ips = resolve_all("google.com")
    for ip in all_ips:
        print("  -", ip.to_string())

================================================================================
4. JSON MODULE (stdlib/io/json.mojo)
================================================================================

The JSON module provides full JSON parsing, serialization, and manipulation.

------------------------------------------------------------------------------
4.1 JSON Values (JsonValue)
------------------------------------------------------------------------------

JsonValue represents any JSON value type.

Type Checking:
    fn is_null(self) -> Bool
    fn is_bool(self) -> Bool
    fn is_number(self) -> Bool
    fn is_string(self) -> Bool
    fn is_array(self) -> Bool
    fn is_object(self) -> Bool
    fn json_type(self) -> JsonType

Value Access:
    fn as_bool(self) -> Bool
    fn as_number(self) -> Float64
    fn as_int(self) -> Int
    fn as_string(self) -> String
    fn as_array(self) -> List[JsonValue]
    fn as_object(self) -> Dict[String, JsonValue]

Constructors:
    JsonValue.null() -> JsonValue
    JsonValue.from_bool(value: Bool) -> JsonValue
    JsonValue.from_number(value: Float64) -> JsonValue
    JsonValue.from_int(value: Int) -> JsonValue
    JsonValue.from_string(value: String) -> JsonValue
    JsonValue.from_array(values: List[JsonValue]) -> JsonValue
    JsonValue.from_object(obj: Dict[String, JsonValue]) -> JsonValue

Object/Array Access:
    fn get(self, key: String) -> JsonValue       # Object key access
    fn get(self, index: Int) -> JsonValue        # Array index access
    fn get_path(self, path: String) -> JsonValue # Dot-notation path
    fn length(self) -> Int                       # Array/object length
    fn keys(self) -> List[String]                # Object keys
    fn values(self) -> List[JsonValue]           # Object/array values

Example:
    let null_val = JsonValue.null()
    let num_val = JsonValue.from_number(3.14)
    let str_val = JsonValue.from_string("hello")

    print(num_val.as_number())  # 3.14
    print(str_val.as_string())  # "hello"

------------------------------------------------------------------------------
4.2 JSON Parser (JsonParser)
------------------------------------------------------------------------------

JsonParser parses JSON strings into JsonValue objects.

    JsonParser(input: String)

    fn parse(inout self) -> Result[JsonValue, JsonError]

JsonError - Parsing error information:
    fn message(self) -> String
    fn line(self) -> Int
    fn column(self) -> Int

Example:
    let json_str = '{"name": "Alice", "age": 30, "tags": ["admin", "user"]}'

    var parser = JsonParser(json_str)
    let result = parser.parse()

    if result.is_ok():
        let root = result.unwrap()
        print(root.get("name").as_string())      # "Alice"
        print(root.get("age").as_int())          # 30
        print(root.get("tags").get(0).as_string())  # "admin"

        # Path access
        print(root.get_path("tags.1").as_string())  # "user"
    else:
        let error = result.error()
        print("Parse error at line", error.line(), ":", error.message())

------------------------------------------------------------------------------
4.3 JSON Serialization
------------------------------------------------------------------------------

Convert JsonValue back to JSON string:

    fn to_string(self) -> String           # Compact JSON
    fn to_pretty_string(self, indent: Int = 2) -> String  # Pretty-printed

Example:
    var obj = JsonValue.from_object(Dict[String, JsonValue]())
    # ... build object ...

    let compact = obj.to_string()
    # {"name":"Alice","age":30}

    let pretty = obj.to_pretty_string()
    # {
    #   "name": "Alice",
    #   "age": 30
    # }

------------------------------------------------------------------------------
4.4 JSON Builder (JsonBuilder)
------------------------------------------------------------------------------

JsonBuilder provides a fluent API for constructing JSON.

Methods:
    fn start_object(inout self)
    fn start_object(inout self, key: String)     # Nested object with key
    fn end_object(inout self)

    fn start_array(inout self)
    fn start_array(inout self, key: String)      # Nested array with key
    fn end_array(inout self)

    fn add_null(inout self, key: String)
    fn add_bool(inout self, key: String, value: Bool)
    fn add_number(inout self, key: String, value: Float64)
    fn add_string(inout self, key: String, value: String)

    fn add_array_null(inout self)                # Add null to current array
    fn add_array_bool(inout self, value: Bool)
    fn add_array_number(inout self, value: Float64)
    fn add_array_string(inout self, value: String)

    fn to_string(self) -> String
    fn to_pretty_string(self, indent: Int = 2) -> String
    fn build(self) -> JsonValue

Example:
    var builder = JsonBuilder()

    builder.start_object()
    builder.add_string("name", "Project Alpha")
    builder.add_number("version", 1.5)
    builder.add_bool("active", True)

    builder.start_array("contributors")
    builder.add_array_string("Alice")
    builder.add_array_string("Bob")
    builder.end_array()

    builder.start_object("config")
    builder.add_bool("debug", False)
    builder.add_number("timeout", 30)
    builder.end_object()

    builder.end_object()

    print(builder.to_pretty_string())
    # {
    #   "name": "Project Alpha",
    #   "version": 1.5,
    #   "active": true,
    #   "contributors": ["Alice", "Bob"],
    #   "config": {
    #     "debug": false,
    #     "timeout": 30
    #   }
    # }

================================================================================
5. TIME MODULE (stdlib/time/time.mojo)
================================================================================

The Time module provides comprehensive date/time handling.

------------------------------------------------------------------------------
5.1 Duration
------------------------------------------------------------------------------

Duration represents a span of time with nanosecond precision.

Constructors:
    Duration(nanos: Int64)
    Duration.from_nanos(nanos: Int64) -> Duration
    Duration.from_micros(micros: Int64) -> Duration
    Duration.from_millis(millis: Int64) -> Duration
    Duration.from_secs(secs: Int64) -> Duration
    Duration.from_mins(mins: Int64) -> Duration
    Duration.from_hours(hours: Int64) -> Duration
    Duration.from_days(days: Int64) -> Duration
    Duration.from_hms(hours: Int, mins: Int, secs: Int) -> Duration

Constants:
    Duration.ZERO
    Duration.NANOSECOND
    Duration.MICROSECOND
    Duration.MILLISECOND
    Duration.SECOND
    Duration.MINUTE
    Duration.HOUR
    Duration.DAY

Accessors:
    fn total_nanos(self) -> Int64
    fn total_micros(self) -> Int64
    fn total_millis(self) -> Int64
    fn total_secs(self) -> Int64
    fn total_mins(self) -> Int64
    fn total_hours(self) -> Int64
    fn total_days(self) -> Int64
    fn as_secs_f64(self) -> Float64

Arithmetic:
    fn __add__(self, other: Duration) -> Duration
    fn __sub__(self, other: Duration) -> Duration
    fn __mul__(self, scalar: Int64) -> Duration
    fn __truediv__(self, divisor: Int64) -> Duration
    fn __neg__(self) -> Duration
    fn abs(self) -> Duration

Example:
    let timeout = Duration.from_secs(30)
    let retry_delay = Duration.from_millis(500)

    let total = timeout + retry_delay * 3
    print("Total wait:", total.as_secs_f64(), "seconds")

    let work_day = Duration.from_hms(8, 30, 0)
    print("Work day:", work_day.to_string())  # "8h 30m 0s"

------------------------------------------------------------------------------
5.2 Date
------------------------------------------------------------------------------

Date represents a calendar date.

Constructors:
    Date(year: Int, month: Int, day: Int)
    Date.today() -> Date
    Date.from_ordinal(year: Int, day_of_year: Int) -> Date
    Date.from_timestamp(timestamp: Int64) -> Date

Properties:
    var year: Int
    var month: Int   # 1-12
    var day: Int     # 1-31

Methods:
    fn is_leap_year(self) -> Bool
    fn days_in_month(self) -> Int
    fn days_in_year(self) -> Int
    fn day_of_year(self) -> Int             # 1-366
    fn weekday(self) -> Weekday             # Monday=0, Sunday=6
    fn week_of_year(self) -> Int            # ISO week number
    fn quarter(self) -> Int                 # 1-4
    fn is_valid(self) -> Bool

Arithmetic:
    fn add_days(self, days: Int) -> Date
    fn add_months(self, months: Int) -> Date
    fn add_years(self, years: Int) -> Date
    fn days_until(self, other: Date) -> Int

Formatting:
    fn to_iso_string(self) -> String        # "2026-01-15"
    fn to_string(self) -> String            # "January 15, 2026"
    fn format(self, pattern: String) -> String

Format Patterns:
    %Y - 4-digit year
    %m - 2-digit month (01-12)
    %d - 2-digit day (01-31)
    %B - Full month name
    %b - Abbreviated month name
    %A - Full weekday name
    %a - Abbreviated weekday name
    %j - Day of year (001-366)
    %W - Week number (01-53)

Example:
    let date = Date(2026, 1, 15)

    print(date.to_iso_string())              # "2026-01-15"
    print(date.format("%B %d, %Y"))          # "January 15, 2026"
    print(date.weekday().name())             # "Thursday"

    let next_month = date.add_months(1)
    let days_between = date.days_until(next_month)
    print("Days until next month:", days_between)

Weekday Enumeration:
    Weekday.MONDAY, .TUESDAY, .WEDNESDAY, .THURSDAY, .FRIDAY, .SATURDAY, .SUNDAY

    fn name(self) -> String          # "Monday"
    fn short_name(self) -> String    # "Mon"
    fn is_weekend(self) -> Bool
    fn is_weekday(self) -> Bool

Month Enumeration:
    Month.JANUARY through Month.DECEMBER

    fn name(self) -> String          # "January"
    fn short_name(self) -> String    # "Jan"
    fn days(self, is_leap: Bool) -> Int

------------------------------------------------------------------------------
5.3 Time
------------------------------------------------------------------------------

Time represents a time of day with nanosecond precision.

Constructors:
    Time(hour: Int, minute: Int, second: Int = 0, nanosecond: Int = 0)
    Time.midnight() -> Time                  # 00:00:00
    Time.noon() -> Time                      # 12:00:00
    Time.from_secs_since_midnight(secs: Int) -> Time
    Time.from_nanos_since_midnight(nanos: Int64) -> Time

Properties:
    var hour: Int        # 0-23
    var minute: Int      # 0-59
    var second: Int      # 0-59
    var nanosecond: Int  # 0-999,999,999

Methods:
    fn to_secs_since_midnight(self) -> Int
    fn to_nanos_since_midnight(self) -> Int64
    fn is_am(self) -> Bool
    fn is_pm(self) -> Bool
    fn hour_12(self) -> Int                  # 1-12
    fn is_valid(self) -> Bool

Arithmetic:
    fn add(self, duration: Duration) -> Time
    fn subtract(self, duration: Duration) -> Time
    fn duration_until(self, other: Time) -> Duration

Formatting:
    fn to_iso_string(self) -> String         # "14:30:45"
    fn to_string_12h(self) -> String         # " 2:30 PM"
    fn format(self, pattern: String) -> String

Format Patterns:
    %H - 24-hour (00-23)
    %I - 12-hour (01-12)
    %M - Minutes (00-59)
    %S - Seconds (00-59)
    %f - Microseconds (000000-999999)
    %p - AM/PM

Example:
    let time = Time(14, 30, 45)

    print(time.to_iso_string())      # "14:30:45"
    print(time.to_string_12h())      # " 2:30 PM"
    print(time.format("%I:%M %p"))   # "02:30 PM"

    let later = time.add(Duration.from_hours(2))
    print(later.to_iso_string())     # "16:30:45"

------------------------------------------------------------------------------
5.4 DateTime
------------------------------------------------------------------------------

DateTime combines Date and Time.

Constructors:
    DateTime(date: Date, time: Time)
    DateTime(year: Int, month: Int, day: Int, hour: Int = 0,
             minute: Int = 0, second: Int = 0, nanosecond: Int = 0)
    DateTime.now() -> DateTime
    DateTime.from_timestamp(timestamp: Int64) -> DateTime
    DateTime.from_timestamp_millis(timestamp_ms: Int64) -> DateTime

Properties:
    var date: Date
    var time: Time

Accessors:
    fn year(self) -> Int
    fn month(self) -> Int
    fn day(self) -> Int
    fn hour(self) -> Int
    fn minute(self) -> Int
    fn second(self) -> Int
    fn nanosecond(self) -> Int
    fn weekday(self) -> Weekday
    fn day_of_year(self) -> Int

Timestamps:
    fn to_timestamp(self) -> Int64           # Unix seconds
    fn to_timestamp_millis(self) -> Int64    # Unix milliseconds

Arithmetic:
    fn add(self, duration: Duration) -> DateTime
    fn subtract(self, duration: Duration) -> DateTime
    fn add_days(self, days: Int) -> DateTime
    fn add_months(self, months: Int) -> DateTime
    fn add_years(self, years: Int) -> DateTime
    fn duration_since(self, other: DateTime) -> Duration

Formatting:
    fn to_iso_string(self) -> String         # "2026-01-15T14:30:45"
    fn to_string(self) -> String             # "January 15, 2026 14:30:45"
    fn format(self, pattern: String) -> String

Example:
    let dt = DateTime(2026, 1, 15, 14, 30, 0)

    print(dt.to_iso_string())        # "2026-01-15T14:30:45"

    # Add 2 days and 3 hours
    let later = dt.add_days(2).add(Duration.from_hours(3))

    # Calculate duration
    let meeting_start = DateTime(2026, 1, 15, 10, 0, 0)
    let meeting_end = DateTime(2026, 1, 15, 11, 30, 0)
    let duration = meeting_end.duration_since(meeting_start)
    print("Meeting duration:", duration.total_mins(), "minutes")

------------------------------------------------------------------------------
5.5 Timezone
------------------------------------------------------------------------------

Timezone represents UTC offset.

Constructors:
    Timezone(offset_minutes: Int, name: String = "")
    Timezone.from_hours(hours: Int) -> Timezone
    Timezone.from_hm(hours: Int, minutes: Int) -> Timezone

Constants:
    Timezone.UTC    # +00:00
    Timezone.EST    # -05:00
    Timezone.EDT    # -04:00
    Timezone.CST    # -06:00
    Timezone.CDT    # -05:00
    Timezone.MST    # -07:00
    Timezone.MDT    # -06:00
    Timezone.PST    # -08:00
    Timezone.PDT    # -07:00
    Timezone.GMT    # +00:00
    Timezone.CET    # +01:00
    Timezone.CEST   # +02:00
    Timezone.JST    # +09:00
    Timezone.AST    # +03:00 (Arabia)

Methods:
    fn offset_duration(self) -> Duration
    fn offset_string(self) -> String         # "+09:00" or "-05:00"
    fn to_string(self) -> String

Example:
    let jst = Timezone.JST
    print(jst.offset_string())       # "+09:00"
    print(jst.to_string())           # "JST (+09:00)"

    let custom = Timezone.from_hm(5, 30)  # +05:30 (India)
    print(custom.offset_string())    # "+05:30"

Parsing (DateTimeParser):
    DateTimeParser.parse_date(s: String) -> Date
    DateTimeParser.parse_time(s: String) -> Time
    DateTimeParser.parse_datetime(s: String) -> DateTime

Supported Formats:
    - ISO 8601: "2026-01-15", "14:30:45", "2026-01-15T14:30:45"
    - US Date: "01/15/2026"
    - European: "15.01.2026"

Example:
    let dt = DateTimeParser.parse_datetime("2026-01-15T14:30:00")
    let date = DateTimeParser.parse_date("01/15/2026")
    let time = DateTimeParser.parse_time("14:30:45")

================================================================================
6. PATH MODULE (stdlib/sys/path.mojo)
================================================================================

The Path module provides cross-platform path manipulation.

------------------------------------------------------------------------------
6.1 Path Operations
------------------------------------------------------------------------------

Path - Main path type:

Constructors:
    Path()
    Path(path: String)
    Path.from_string(s: String) -> Path
    Path.cwd() -> Path                       # Current working directory
    Path.home() -> Path                      # User home directory
    Path.temp() -> Path                      # System temp directory

Properties:
    fn as_string(self) -> String
    fn is_empty(self) -> Bool
    fn is_absolute(self) -> Bool
    fn is_relative(self) -> Bool
    fn length(self) -> Int

Components:
    fn filename(self) -> String              # "file.txt"
    fn stem(self) -> String                  # "file"
    fn extension(self) -> String             # ".txt"
    fn extension_without_dot(self) -> String # "txt"
    fn parent(self) -> Path
    fn root(self) -> Path

Manipulation:
    fn join(self, other: String) -> Path
    fn join(self, other: Path) -> Path
    fn __truediv__(self, other: String) -> Path  # path / "subdir"
    fn with_filename(self, name: String) -> Path
    fn with_extension(self, ext: String) -> Path
    fn with_stem(self, stem: String) -> Path

Normalization:
    fn normalize(self) -> Path               # Resolve . and ..
    fn to_unix(self) -> Path                 # Forward slashes
    fn to_windows(self) -> Path              # Backslashes
    fn to_native(self) -> Path               # Platform-appropriate

Queries:
    fn starts_with(self, prefix: String) -> Bool
    fn starts_with(self, prefix: Path) -> Bool
    fn ends_with(self, suffix: String) -> Bool
    fn contains(self, s: String) -> Bool
    fn has_extension(self, ext: String) -> Bool
    fn is_hidden(self) -> Bool               # Starts with .

Relative Paths:
    fn relative_to(self, base: Path) -> Path
    fn strip_prefix(self, prefix: Path) -> Path

Example:
    let path = Path("/home/user/documents/report.pdf")

    print(path.filename())           # "report.pdf"
    print(path.stem())               # "report"
    print(path.extension())          # ".pdf"
    print(path.parent().as_string()) # "/home/user/documents"

    # Path joining
    let base = Path("/home/user")
    let full = base / "documents" / "file.txt"

    # Modification
    let new_path = path.with_extension(".docx")
    print(new_path.filename())       # "report.docx"

    # Normalization
    let messy = Path("/home/user/../user/./documents")
    print(messy.normalize().as_string())  # "/home/user/documents"

------------------------------------------------------------------------------
6.2 Path Components
------------------------------------------------------------------------------

PathComponent - Individual path segment:

    var value: String
    var is_root: Bool
    var is_current: Bool     # "."
    var is_parent: Bool      # ".."

    fn is_normal(self) -> Bool

Path.components() -> List[PathComponent]:

Example:
    let path = Path("/home/user/documents")
    let components = path.components()

    for comp in components:
        print(comp.value)
    # Output: "/", "home", "user", "documents"

PathIterator - Iterate over components:

    PathIterator(path: Path)
    fn has_next(self) -> Bool
    fn next(inout self) -> PathComponent
    fn reset(inout self)

AncestorIterator - Iterate over parent directories:

    AncestorIterator(path: Path)
    fn has_next(self) -> Bool
    fn next(inout self) -> Path

Example:
    let path = Path("/home/user/documents/file.txt")
    var ancestors = AncestorIterator(path)

    while ancestors.has_next():
        print(ancestors.next().as_string())
    # Output:
    # /home/user/documents/file.txt
    # /home/user/documents
    # /home/user
    # /home
    # /

------------------------------------------------------------------------------
6.3 Glob Patterns (GlobPattern)
------------------------------------------------------------------------------

GlobPattern matches file paths using wildcards.

Pattern Syntax:
    *      - Match any characters (not including separator)
    **     - Match any characters (including separator) - recursive
    ?      - Match single character
    [abc]  - Match any character in set
    [a-z]  - Match character range
    [!abc] - Match any character NOT in set

Constructor:
    GlobPattern(pattern: String)

Methods:
    fn matches(self, path: String) -> Bool
    fn matches(self, path: Path) -> Bool

Example:
    # Match all .txt files
    let txt_pattern = GlobPattern("*.txt")
    print(txt_pattern.matches("file.txt"))     # true
    print(txt_pattern.matches("file.pdf"))     # false

    # Match all Python files in src directory (recursive)
    let py_pattern = GlobPattern("src/**/*.py")
    print(py_pattern.matches("src/main.py"))           # true
    print(py_pattern.matches("src/utils/helpers.py"))  # true
    print(py_pattern.matches("tests/test.py"))         # false

    # Match single character wildcard
    let log_pattern = GlobPattern("app?.log")
    print(log_pattern.matches("app1.log"))     # true
    print(log_pattern.matches("app12.log"))    # false

    # Character class
    let class_pattern = GlobPattern("[abc].txt")
    print(class_pattern.matches("a.txt"))      # true
    print(class_pattern.matches("d.txt"))      # false

------------------------------------------------------------------------------
6.4 Path Builder
------------------------------------------------------------------------------

PathBuilder - Fluent path construction:

    PathBuilder()
    PathBuilder(base: String)
    PathBuilder(base: Path)

    fn push(inout self, component: String) -> PathBuilder
    fn push(inout self, component: Path) -> PathBuilder
    fn pop(inout self) -> PathBuilder
    fn build(self) -> Path

Example:
    var builder = PathBuilder("/home")
    let path = builder
        .push("user")
        .push("documents")
        .push("project")
        .push("src")
        .build()

    print(path.as_string())  # "/home/user/documents/project/src"

    # Pop last component
    var builder2 = PathBuilder(path)
    let parent = builder2.pop().build()
    print(parent.as_string())  # "/home/user/documents/project"

Utility Functions:

    join_paths(parts: List[String]) -> Path
    split_path(path: String) -> List[String]
    common_path(paths: List[Path]) -> Path
    expand_tilde(path: String) -> Path

    is_valid_filename(name: String) -> Bool
    sanitize_filename(name: String) -> String

Example:
    # Find common prefix
    let paths = [
        Path("/home/user/documents/a.txt"),
        Path("/home/user/documents/b.txt"),
        Path("/home/user/pictures/c.png")
    ]
    let common = common_path(paths)
    print(common.as_string())  # "/home/user"

    # Expand tilde
    let expanded = expand_tilde("~/documents")
    print(expanded.as_string())  # "/home/user/documents"

    # Validate filename
    print(is_valid_filename("report.pdf"))    # true
    print(is_valid_filename("file/name.txt")) # false

    # Sanitize filename
    let safe = sanitize_filename("my:file<name>.txt")
    print(safe)  # "my_file_name_.txt"

================================================================================
END OF API REFERENCE
================================================================================

For more information:
- See individual module source files for implementation details
- Run module tests with: mojo test stdlib/tests/
- Report issues at: https://github.com/mojo-sdk/issues

Version: 1.0.0
Last Updated: January 2026
"""
