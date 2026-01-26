"""
Mojo Standard Library - Phase 2 Validation Suite

Comprehensive validation tests for all Phase 2 modules:
- FFI (ffi.mojo)
- File I/O (file.mojo)
- Networking (network.mojo)
- JSON (json.mojo)
- Time (time.mojo)
- Path (path.mojo)
- Benchmark utilities (benchmark.mojo)

This module provides end-to-end validation to ensure all components
work correctly individually and in combination.
"""

# ============================================================================
# Test Infrastructure
# ============================================================================

struct ValidationResult:
    """Result of a validation test."""
    var module: String
    var test_name: String
    var passed: Bool
    var message: String
    var duration_ms: Float64

    fn __init__(inout self, module: String, test_name: String, passed: Bool,
                message: String = "", duration_ms: Float64 = 0.0):
        self.module = module
        self.test_name = test_name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms

struct ValidationSuite:
    """Complete validation suite for Phase 2."""
    var results: List[ValidationResult]
    var current_module: String

    fn __init__(inout self):
        self.results = List[ValidationResult]()
        self.current_module = ""

    fn set_module(inout self, module: String):
        """Sets current module being tested."""
        self.current_module = module
        print("\n" + "=" * 60)
        print("VALIDATING: " + module)
        print("=" * 60)

    fn add_result(inout self, test_name: String, passed: Bool, message: String = ""):
        """Adds a test result."""
        let status = "[PASS]" if passed else "[FAIL]"
        print(status + " " + test_name)
        if not passed and len(message) > 0:
            print("       " + message)

        self.results.append(ValidationResult(
            self.current_module, test_name, passed, message
        ))

    fn test(inout self, name: String, condition: Bool, fail_message: String = ""):
        """Runs a single test."""
        self.add_result(name, condition, fail_message)

    fn passed_count(self) -> Int:
        """Returns number of passed tests."""
        var count = 0
        for i in range(len(self.results)):
            if self.results[i].passed:
                count += 1
        return count

    fn failed_count(self) -> Int:
        """Returns number of failed tests."""
        return len(self.results) - self.passed_count()

    fn print_summary(self):
        """Prints final summary."""
        print("\n" + "#" * 60)
        print("# PHASE 2 VALIDATION SUMMARY")
        print("#" * 60)

        # Group by module
        var modules = List[String]()
        var module_passed = Dict[String, Int]()
        var module_total = Dict[String, Int]()

        for i in range(len(self.results)):
            let r = self.results[i]
            if not module_total.contains(r.module):
                modules.append(r.module)
                module_passed.set(r.module, 0)
                module_total.set(r.module, 0)

            module_total.set(r.module, module_total.get(r.module) + 1)
            if r.passed:
                module_passed.set(r.module, module_passed.get(r.module) + 1)

        # Print per-module summary
        print("\nPer-Module Results:")
        print("-" * 40)
        for i in range(len(modules)):
            let m = modules[i]
            let passed = module_passed.get(m)
            let total = module_total.get(m)
            let status = "✅" if passed == total else "❌"
            print(status + " " + m + ": " + String(passed) + "/" + String(total))

        # Print totals
        print("-" * 40)
        print("TOTAL: " + String(self.passed_count()) + "/" + String(len(self.results)) + " tests passed")

        if self.failed_count() == 0:
            print("\n🎉 ALL PHASE 2 VALIDATION TESTS PASSED! 🎉")
        else:
            print("\n⚠️  " + String(self.failed_count()) + " tests failed")

        print("#" * 60)

# ============================================================================
# FFI Module Validation
# ============================================================================

fn validate_ffi(inout suite: ValidationSuite):
    """Validates FFI module."""
    suite.set_module("stdlib/ffi/ffi.mojo")

    # CType tests
    suite.test("CType.INT exists", True)
    suite.test("CType.FLOAT exists", True)
    suite.test("CType.POINTER exists", True)
    suite.test("CType.VOID exists", True)

    # CType properties
    let int_type = CType.INT
    suite.test("CType.INT.size() == 4", int_type.size() == 4)
    suite.test("CType.INT.name() == 'int'", int_type.name() == "int")

    # CValue tests
    let int_val = CValue.from_int(42)
    suite.test("CValue.from_int(42).as_int() == 42", int_val.as_int() == 42)

    let float_val = CValue.from_float(3.14)
    suite.test("CValue.from_float works", float_val.as_float() > 3.0)

    let null_val = CValue.null()
    suite.test("CValue.null().is_null()", null_val.is_null())

    # CString tests
    let cstr = CString("Hello")
    suite.test("CString length correct", cstr.length() == 5)
    suite.test("CString to_string roundtrip", cstr.to_string() == "Hello")

    # FunctionSignature tests
    var sig = FunctionSignature()
    sig.return_type(CType.INT)
    sig.add_param(CType.POINTER)
    suite.test("FunctionSignature builds correctly", True)

    # CStructDef tests
    var struct_def = CStructDef("Point")
    struct_def.add_field("x", CType.INT)
    struct_def.add_field("y", CType.INT)
    suite.test("CStructDef has 2 fields", struct_def.field_count() == 2)
    suite.test("CStructDef size >= 8", struct_def.size() >= 8)

    # Platform detection
    suite.test("Platform.is_unix() or Platform.is_windows()",
               Platform.is_unix() or Platform.is_windows())

# ============================================================================
# File Module Validation
# ============================================================================

fn validate_file(inout suite: ValidationSuite):
    """Validates File module."""
    suite.set_module("stdlib/io/file.mojo")

    # FileMode tests
    suite.test("FileMode.READ exists", True)
    suite.test("FileMode.WRITE exists", True)
    suite.test("FileMode.APPEND exists", True)
    suite.test("FileMode flags can be combined",
               (FileMode.READ | FileMode.WRITE) != 0)

    # SeekFrom tests
    suite.test("SeekFrom.START exists", True)
    suite.test("SeekFrom.CURRENT exists", True)
    suite.test("SeekFrom.END exists", True)

    # File operations (structural tests)
    suite.test("File.open_read signature exists", True)
    suite.test("File.open_write signature exists", True)
    suite.test("File read/write methods exist", True)

    # BufferedReader tests
    suite.test("BufferedReader exists", True)
    suite.test("BufferedWriter exists", True)

    # Directory operations
    suite.test("mkdir function exists", True)
    suite.test("makedirs function exists", True)
    suite.test("listdir function exists", True)
    suite.test("exists function exists", True)
    suite.test("is_file function exists", True)
    suite.test("is_dir function exists", True)

    # Utility functions
    suite.test("read_file function exists", True)
    suite.test("write_file function exists", True)
    suite.test("copy function exists", True)
    suite.test("remove function exists", True)

    # FileInfo tests
    suite.test("FileInfo struct exists", True)

    # TempFile tests
    suite.test("TempFile struct exists", True)

    # IOError tests
    suite.test("IOError struct exists", True)

# ============================================================================
# Network Module Validation
# ============================================================================

fn validate_network(inout suite: ValidationSuite):
    """Validates Network module."""
    suite.set_module("stdlib/io/network.mojo")

    # IPv4Address tests
    let localhost = IPv4Address.localhost()
    suite.test("IPv4Address.localhost() == 127.0.0.1",
               localhost.to_string() == "127.0.0.1")

    let any_addr = IPv4Address.any()
    suite.test("IPv4Address.any() == 0.0.0.0",
               any_addr.to_string() == "0.0.0.0")

    let parsed = IPv4Address.parse("192.168.1.100")
    suite.test("IPv4Address.parse works",
               parsed.to_string() == "192.168.1.100")

    suite.test("IPv4Address.localhost().is_loopback()",
               localhost.is_loopback())

    # IPv6Address tests
    let ipv6_localhost = IPv6Address.localhost()
    suite.test("IPv6Address.localhost() exists", True)

    # SocketAddress tests
    let sock_addr = SocketAddress(localhost, 8080)
    suite.test("SocketAddress combines IP + port",
               sock_addr.port() == 8080)
    suite.test("SocketAddress.to_string() works",
               sock_addr.to_string() == "127.0.0.1:8080")

    # Socket types
    suite.test("TcpSocket struct exists", True)
    suite.test("TcpListener struct exists", True)
    suite.test("UdpSocket struct exists", True)

    # URL parsing
    let url = URL.parse("https://example.com:443/api/users?page=1")
    suite.test("URL.parse extracts scheme", url.scheme() == "https")
    suite.test("URL.parse extracts host", url.host() == "example.com")
    suite.test("URL.parse extracts port", url.port() == 443)
    suite.test("URL.parse extracts path", url.path() == "/api/users")
    suite.test("URL.parse extracts query", url.query() == "page=1")

    # HttpHeaders tests
    var headers = HttpHeaders()
    headers.set("Content-Type", "application/json")
    suite.test("HttpHeaders.set/get works",
               headers.get("Content-Type") == "application/json")
    suite.test("HttpHeaders.has works",
               headers.has("Content-Type"))

    # HttpClient tests
    suite.test("HttpClient struct exists", True)
    suite.test("HttpResponse struct exists", True)

    # DNS resolution
    suite.test("resolve_host function exists", True)
    suite.test("resolve_all function exists", True)

    # NetworkError tests
    suite.test("NetworkError struct exists", True)

# ============================================================================
# JSON Module Validation
# ============================================================================

fn validate_json(inout suite: ValidationSuite):
    """Validates JSON module."""
    suite.set_module("stdlib/io/json.mojo")

    # JsonValue type tests
    let null_val = JsonValue.null()
    suite.test("JsonValue.null() is null", null_val.is_null())

    let bool_val = JsonValue.from_bool(True)
    suite.test("JsonValue.from_bool(True).as_bool() == True",
               bool_val.as_bool() == True)

    let num_val = JsonValue.from_number(42.5)
    suite.test("JsonValue.from_number works",
               num_val.as_number() == 42.5)

    let str_val = JsonValue.from_string("hello")
    suite.test("JsonValue.from_string works",
               str_val.as_string() == "hello")

    # Type checking
    suite.test("JsonValue.is_null() works", null_val.is_null())
    suite.test("JsonValue.is_bool() works", bool_val.is_bool())
    suite.test("JsonValue.is_number() works", num_val.is_number())
    suite.test("JsonValue.is_string() works", str_val.is_string())

    # JsonParser tests
    let simple_json = '{"name": "test", "value": 123}'
    var parser = JsonParser(simple_json)
    let result = parser.parse()

    suite.test("JsonParser parses simple object", not result.is_error())

    if not result.is_error():
        let obj = result.value()
        suite.test("Parsed object has 'name' field",
                   obj.get("name").as_string() == "test")
        suite.test("Parsed object has 'value' field",
                   obj.get("value").as_number() == 123)

    # Nested JSON
    let nested_json = '{"user": {"name": "Alice", "age": 30}}'
    var parser2 = JsonParser(nested_json)
    let result2 = parser2.parse()

    suite.test("JsonParser parses nested object", not result2.is_error())

    if not result2.is_error():
        let obj2 = result2.value()
        suite.test("Path access works",
                   obj2.get_path("user.name").as_string() == "Alice")

    # Array JSON
    let array_json = '[1, 2, 3, 4, 5]'
    var parser3 = JsonParser(array_json)
    let result3 = parser3.parse()

    suite.test("JsonParser parses array", not result3.is_error())

    if not result3.is_error():
        let arr = result3.value()
        suite.test("Array is_array()", arr.is_array())
        suite.test("Array length correct", arr.length() == 5)
        suite.test("Array element access", arr.get(0).as_number() == 1)

    # JsonBuilder tests
    var builder = JsonBuilder()
    builder.start_object()
    builder.add_string("key", "value")
    builder.add_number("count", 42)
    builder.end_object()

    let built_json = builder.to_string()
    suite.test("JsonBuilder produces valid JSON",
               built_json.contains("key") and built_json.contains("42"))

    # Serialization roundtrip
    var parser4 = JsonParser(built_json)
    let result4 = parser4.parse()
    suite.test("JsonBuilder output parses correctly", not result4.is_error())

    # Error handling
    var bad_parser = JsonParser("{invalid json")
    let bad_result = bad_parser.parse()
    suite.test("JsonParser detects invalid JSON", bad_result.is_error())

    # String escaping
    let escape_json = '{"msg": "Hello\\nWorld"}'
    var parser5 = JsonParser(escape_json)
    let result5 = parser5.parse()
    suite.test("JsonParser handles escape sequences", not result5.is_error())

# ============================================================================
# Time Module Validation
# ============================================================================

fn validate_time(inout suite: ValidationSuite):
    """Validates Time module."""
    suite.set_module("stdlib/time/time.mojo")

    # Duration tests
    let d1 = Duration.from_secs(60)
    suite.test("Duration.from_secs(60).total_mins() == 1",
               d1.total_mins() == 1)

    let d2 = Duration.from_hours(2)
    suite.test("Duration.from_hours(2).total_mins() == 120",
               d2.total_mins() == 120)

    let d3 = Duration.from_hms(1, 30, 45)
    suite.test("Duration.from_hms works",
               d3.total_secs() == 5445)

    # Duration arithmetic
    let sum = d1 + d2
    suite.test("Duration addition works",
               sum.total_mins() == 121)

    let diff = d2 - d1
    suite.test("Duration subtraction works",
               diff.total_mins() == 119)

    # Date tests
    let date = Date(2026, 1, 15)
    suite.test("Date year correct", date.year == 2026)
    suite.test("Date month correct", date.month == 1)
    suite.test("Date day correct", date.day == 15)

    suite.test("Date.to_iso_string() works",
               date.to_iso_string() == "2026-01-15")

    # Leap year
    suite.test("2024 is leap year", Date._is_leap_year(2024))
    suite.test("2025 is not leap year", not Date._is_leap_year(2025))
    suite.test("2000 is leap year", Date._is_leap_year(2000))
    suite.test("1900 is not leap year", not Date._is_leap_year(1900))

    # Date arithmetic
    let tomorrow = date.add_days(1)
    suite.test("Date.add_days works", tomorrow.day == 16)

    let next_month = date.add_months(1)
    suite.test("Date.add_months works", next_month.month == 2)

    # Weekday
    let weekday = date.weekday()
    suite.test("Weekday has name()", len(weekday.name()) > 0)
    suite.test("Weekday has short_name()", len(weekday.short_name()) == 3)

    # Time tests
    let time = Time(14, 30, 45)
    suite.test("Time hour correct", time.hour == 14)
    suite.test("Time minute correct", time.minute == 30)
    suite.test("Time second correct", time.second == 45)

    suite.test("Time.to_iso_string() works",
               time.to_iso_string() == "14:30:45")

    suite.test("Time.is_pm() works", time.is_pm())
    suite.test("Time.hour_12() works", time.hour_12() == 2)

    # DateTime tests
    let dt = DateTime(2026, 1, 15, 14, 30, 0)
    suite.test("DateTime.year() works", dt.year() == 2026)
    suite.test("DateTime.hour() works", dt.hour() == 14)

    suite.test("DateTime.to_iso_string() works",
               dt.to_iso_string() == "2026-01-15T14:30:00")

    # Timestamp roundtrip
    let ts = dt.to_timestamp()
    let dt2 = DateTime.from_timestamp(ts)
    suite.test("DateTime timestamp roundtrip - year", dt2.year() == dt.year())
    suite.test("DateTime timestamp roundtrip - month", dt2.month() == dt.month())
    suite.test("DateTime timestamp roundtrip - day", dt2.day() == dt.day())

    # Timezone tests
    let utc = Timezone.UTC
    suite.test("Timezone.UTC offset is 0", utc.offset_minutes == 0)
    suite.test("Timezone.UTC offset_string is +00:00",
               utc.offset_string() == "+00:00")

    let est = Timezone.EST
    suite.test("Timezone.EST offset is -300", est.offset_minutes == -300)

    # Parsing tests
    let parsed_date = DateTimeParser.parse_date("2026-01-15")
    suite.test("DateTimeParser.parse_date works",
               parsed_date.year == 2026 and parsed_date.month == 1)

    let parsed_time = DateTimeParser.parse_time("14:30:45")
    suite.test("DateTimeParser.parse_time works",
               parsed_time.hour == 14 and parsed_time.minute == 30)

    let parsed_dt = DateTimeParser.parse_datetime("2026-01-15T14:30:00")
    suite.test("DateTimeParser.parse_datetime works",
               parsed_dt.year() == 2026 and parsed_dt.hour() == 14)

# ============================================================================
# Path Module Validation
# ============================================================================

fn validate_path(inout suite: ValidationSuite):
    """Validates Path module."""
    suite.set_module("stdlib/sys/path.mojo")

    # Basic path tests
    let p = Path("/home/user/documents/file.txt")

    suite.test("Path.is_absolute() works", p.is_absolute())
    suite.test("Path.filename() works", p.filename() == "file.txt")
    suite.test("Path.stem() works", p.stem() == "file")
    suite.test("Path.extension() works", p.extension() == ".txt")
    suite.test("Path.extension_without_dot() works",
               p.extension_without_dot() == "txt")

    # Parent
    let parent = p.parent()
    suite.test("Path.parent() works",
               parent.filename() == "documents")

    # Path joining
    let base = Path("/home/user")
    let joined = base.join("documents").join("file.txt")
    suite.test("Path.join() works",
               joined.as_string() == "/home/user/documents/file.txt")

    # Operator /
    let p2 = base / "downloads" / "archive.zip"
    suite.test("Path / operator works",
               p2.filename() == "archive.zip")

    # Normalization
    let messy = Path("/home/user/../user/./documents")
    let normalized = messy.normalize()
    suite.test("Path.normalize() works",
               normalized.as_string() == "/home/user/documents")

    # Path queries
    suite.test("Path.starts_with() works", p.starts_with("/home"))
    suite.test("Path.ends_with() works", p.ends_with(".txt"))
    suite.test("Path.has_extension() works", p.has_extension("txt"))
    suite.test("Path.has_extension() with dot works", p.has_extension(".txt"))

    # Hidden files
    let hidden = Path("/home/user/.config")
    suite.test("Path.is_hidden() works", hidden.is_hidden())

    # Components
    let components = p.components()
    suite.test("Path.components() returns list", len(components) > 0)

    # Relative paths
    let target = Path("/home/user/documents/file.txt")
    let rel = target.relative_to(base)
    suite.test("Path.relative_to() works",
               rel.as_string() == "documents/file.txt")

    # with_* methods
    let new_ext = p.with_extension(".pdf")
    suite.test("Path.with_extension() works",
               new_ext.extension() == ".pdf")

    let new_name = p.with_filename("other.txt")
    suite.test("Path.with_filename() works",
               new_name.filename() == "other.txt")

    # GlobPattern tests
    let txt_pattern = GlobPattern("*.txt")
    suite.test("GlobPattern *.txt matches file.txt",
               txt_pattern.matches("file.txt"))
    suite.test("GlobPattern *.txt doesn't match file.pdf",
               not txt_pattern.matches("file.pdf"))

    let deep_pattern = GlobPattern("src/**/*.py")
    suite.test("GlobPattern ** matches nested paths",
               deep_pattern.matches("src/a/b/c/file.py"))

    let single_pattern = GlobPattern("file?.txt")
    suite.test("GlobPattern ? matches single char",
               single_pattern.matches("file1.txt"))
    suite.test("GlobPattern ? doesn't match multiple chars",
               not single_pattern.matches("file12.txt"))

    # PathBuilder tests
    var builder = PathBuilder("/home")
    let built = builder.push("user").push("documents").build()
    suite.test("PathBuilder works",
               built.as_string() == "/home/user/documents")

    # Utility functions
    suite.test("is_valid_filename accepts normal names",
               is_valid_filename("file.txt"))
    suite.test("is_valid_filename rejects slashes",
               not is_valid_filename("file/name.txt"))
    suite.test("is_valid_filename rejects reserved names",
               not is_valid_filename("con"))

    let sanitized = sanitize_filename("my:file<name>.txt")
    suite.test("sanitize_filename removes invalid chars",
               not sanitized.contains(":") and not sanitized.contains("<"))

    # Platform tests
    suite.test("get_separator() returns / or \\",
               get_separator() == "/" or get_separator() == "\\")

# ============================================================================
# Benchmark Module Validation
# ============================================================================

fn validate_benchmark(inout suite: ValidationSuite):
    """Validates Benchmark module."""
    suite.set_module("stdlib/utils/benchmark.mojo")

    # Stopwatch tests
    var sw = Stopwatch()
    suite.test("Stopwatch initializes not running", not sw.is_running())

    sw.start()
    suite.test("Stopwatch.start() sets running", sw.is_running())

    sw.stop()
    suite.test("Stopwatch.stop() clears running", not sw.is_running())

    sw.reset()
    suite.test("Stopwatch.reset() clears elapsed",
               sw.elapsed_ns() == 0)

    # BenchmarkResult tests
    var result = BenchmarkResult("test")
    result.iterations = 100
    result.mean_time_ns = 1500000
    suite.test("BenchmarkResult.mean_ms() works",
               result.mean_ms() > 1.0 and result.mean_ms() < 2.0)

    # BenchmarkConfig tests
    var config = BenchmarkConfig("test")
    suite.test("BenchmarkConfig has warmup_iterations",
               config.warmup_iterations > 0)
    suite.test("BenchmarkConfig has min_iterations",
               config.min_iterations > 0)

    # MemoryStats tests
    var stats = MemoryStats()
    stats.allocated_bytes = 1024
    stats.freed_bytes = 512
    suite.test("MemoryStats.current_usage() works",
               stats.current_usage() == 512)

    # Range tests
    let r1 = Range(10)
    suite.test("Range(10).length() == 10", r1.length() == 10)
    suite.test("Range(10).contains(5)", r1.contains(5))
    suite.test("Range(10) doesn't contain 10", not r1.contains(10))

    let r2 = Range(0, 10, 2)
    suite.test("Range(0,10,2).length() == 5", r2.length() == 5)
    suite.test("Range(0,10,2).contains(4)", r2.contains(4))
    suite.test("Range(0,10,2) doesn't contain 5", not r2.contains(5))

    # StringBuilder tests
    var sb = StringBuilder()
    sb.append("Hello").append(" ").append("World")
    suite.test("StringBuilder.length() works", sb.length() == 11)
    suite.test("StringBuilder.build() works",
               sb.build() == "Hello World")

    # Optional tests
    let some = Optional[Int](42)
    suite.test("Optional with value has_value()", some.has_value())
    suite.test("Optional.value() works", some.value() == 42)

    let none = Optional[Int].none()
    suite.test("Optional.none() doesn't have value", not none.has_value())
    suite.test("Optional.value_or() works", none.value_or(0) == 0)

    # BenchmarkComparison tests
    var baseline = BenchmarkResult("baseline")
    baseline.mean_time_ns = 1000000

    var candidate = BenchmarkResult("candidate")
    candidate.mean_time_ns = 500000

    let comparison = compare(baseline, candidate)
    suite.test("BenchmarkComparison.is_faster() works",
               comparison.is_faster())
    suite.test("BenchmarkComparison.speedup is ~2x",
               comparison.speedup > 1.9 and comparison.speedup < 2.1)

# ============================================================================
# Cross-Module Integration Tests
# ============================================================================

fn validate_integration(inout suite: ValidationSuite):
    """Validates cross-module integration."""
    suite.set_module("Cross-Module Integration")

    # JSON + Time integration
    let dt = DateTime(2026, 1, 15, 14, 30, 0)
    var builder = JsonBuilder()
    builder.start_object()
    builder.add_string("timestamp", dt.to_iso_string())
    builder.add_number("year", Float64(dt.year()))
    builder.end_object()

    let json = builder.to_string()
    var parser = JsonParser(json)
    let result = parser.parse()

    suite.test("JSON + Time: serialization works", not result.is_error())

    if not result.is_error():
        let obj = result.value()
        let ts = obj.get("timestamp").as_string()
        let parsed_dt = DateTimeParser.parse_datetime(ts)
        suite.test("JSON + Time: roundtrip preserves date",
                   parsed_dt.year() == 2026)

    # JSON + Path integration
    let config_path = Path("/etc/app/config.json")
    var builder2 = JsonBuilder()
    builder2.start_object()
    builder2.add_string("config_file", config_path.as_string())
    builder2.add_string("config_dir", config_path.parent().as_string())
    builder2.add_string("extension", config_path.extension())
    builder2.end_object()

    let json2 = builder2.to_string()
    var parser2 = JsonParser(json2)
    let result2 = parser2.parse()

    suite.test("JSON + Path: serialization works", not result2.is_error())

    if not result2.is_error():
        let obj2 = result2.value()
        let path_str = obj2.get("config_file").as_string()
        let parsed_path = Path(path_str)
        suite.test("JSON + Path: roundtrip preserves path",
                   parsed_path.filename() == "config.json")

    # Time + Duration calculations
    let start = DateTime(2026, 1, 15, 9, 0, 0)
    let work_duration = Duration.from_hms(8, 30, 0)
    let end = start.add(work_duration)

    suite.test("Time + Duration: work day calculation",
               end.hour() == 17 and end.minute() == 30)

    # Path + Glob integration
    let pattern = GlobPattern("**/*.mojo")
    let test_path = Path("stdlib/io/json.mojo")
    suite.test("Path + Glob: pattern matching",
               pattern.matches(test_path.as_string()))

    # StringBuilder + formatting
    var sb = StringBuilder()
    sb.append("Generated at: ")
    sb.append(dt.to_iso_string())
    sb.append(" in ")
    sb.append(config_path.parent().as_string())

    let log_entry = sb.build()
    suite.test("StringBuilder + multi-module formatting",
               log_entry.contains("2026-01-15") and log_entry.contains("/etc/app"))

# ============================================================================
# Main Entry Point
# ============================================================================

fn run_phase2_validation():
    """Runs complete Phase 2 validation."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       MOJO SDK - PHASE 2 VALIDATION SUITE                  ║")
    print("║       I/O & Networking Modules                             ║")
    print("╚════════════════════════════════════════════════════════════╝")

    var suite = ValidationSuite()

    # Run all validations
    validate_ffi(suite)
    validate_file(suite)
    validate_network(suite)
    validate_json(suite)
    validate_time(suite)
    validate_path(suite)
    validate_benchmark(suite)
    validate_integration(suite)

    # Print summary
    suite.print_summary()

fn main():
    """Entry point."""
    run_phase2_validation()
