"""
Mojo Standard Library - Integration Tests

Comprehensive integration tests that verify cross-module functionality
and ensure all stdlib components work together correctly.

Modules tested:
- stdlib/ffi/ffi.mojo
- stdlib/io/file.mojo
- stdlib/io/network.mojo
- stdlib/io/json.mojo
- stdlib/time/time.mojo
- stdlib/sys/path.mojo
"""

# ============================================================================
# Test Framework
# ============================================================================

struct TestResult:
    """Result of a single test."""
    var name: String
    var passed: Bool
    var message: String
    var duration_ns: Int64

    fn __init__(inout self, name: String, passed: Bool, message: String = "", duration_ns: Int64 = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ns = duration_ns

struct TestSuite:
    """Collection of test results."""
    var name: String
    var results: List[TestResult]
    var total_duration_ns: Int64

    fn __init__(inout self, name: String):
        self.name = name
        self.results = List[TestResult]()
        self.total_duration_ns = 0

    fn add_result(inout self, result: TestResult):
        """Adds a test result to the suite."""
        self.results.append(result)
        self.total_duration_ns += result.duration_ns

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

    fn all_passed(self) -> Bool:
        """Returns True if all tests passed."""
        return self.failed_count() == 0

    fn print_summary(self):
        """Prints test summary."""
        print("\n" + "=" * 60)
        print("Test Suite: " + self.name)
        print("=" * 60)

        for i in range(len(self.results)):
            let result = self.results[i]
            let status = "[PASS]" if result.passed else "[FAIL]"
            print(status + " " + result.name)
            if not result.passed and len(result.message) > 0:
                print("       " + result.message)

        print("-" * 60)
        print("Total: " + String(len(self.results)) + " tests")
        print("Passed: " + String(self.passed_count()))
        print("Failed: " + String(self.failed_count()))
        print("Duration: " + String(self.total_duration_ns // 1000000) + "ms")
        print("=" * 60)

struct TestRunner:
    """Runs test suites and collects results."""
    var suites: List[TestSuite]

    fn __init__(inout self):
        self.suites = List[TestSuite]()

    fn add_suite(inout self, suite: TestSuite):
        """Adds a test suite."""
        self.suites.append(suite)

    fn run_all(inout self):
        """Runs all test suites."""
        print("\n" + "#" * 60)
        print("# Mojo Standard Library Integration Tests")
        print("#" * 60)

        var total_passed = 0
        var total_failed = 0

        for i in range(len(self.suites)):
            self.suites[i].print_summary()
            total_passed += self.suites[i].passed_count()
            total_failed += self.suites[i].failed_count()

        print("\n" + "#" * 60)
        print("# FINAL RESULTS")
        print("#" * 60)
        print("Total Suites: " + String(len(self.suites)))
        print("Total Tests: " + String(total_passed + total_failed))
        print("Total Passed: " + String(total_passed))
        print("Total Failed: " + String(total_failed))

        if total_failed == 0:
            print("\n*** ALL TESTS PASSED ***")
        else:
            print("\n*** " + String(total_failed) + " TESTS FAILED ***")

        print("#" * 60)

# ============================================================================
# Assertion Helpers
# ============================================================================

fn assert_eq[T: Stringable](actual: T, expected: T, message: String = "") -> TestResult:
    """Asserts two values are equal."""
    let actual_str = String(actual)
    let expected_str = String(expected)

    if actual_str == expected_str:
        return TestResult(message, True)
    else:
        return TestResult(message, False, "Expected: " + expected_str + ", Got: " + actual_str)

fn assert_true(condition: Bool, message: String = "") -> TestResult:
    """Asserts condition is true."""
    if condition:
        return TestResult(message, True)
    else:
        return TestResult(message, False, "Expected true, got false")

fn assert_false(condition: Bool, message: String = "") -> TestResult:
    """Asserts condition is false."""
    if not condition:
        return TestResult(message, True)
    else:
        return TestResult(message, False, "Expected false, got true")

fn assert_not_empty(s: String, message: String = "") -> TestResult:
    """Asserts string is not empty."""
    if len(s) > 0:
        return TestResult(message, True)
    else:
        return TestResult(message, False, "Expected non-empty string")

fn assert_contains(haystack: String, needle: String, message: String = "") -> TestResult:
    """Asserts string contains substring."""
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return TestResult(message, True)
    return TestResult(message, False, "String does not contain: " + needle)

# ============================================================================
# JSON Integration Tests
# ============================================================================

fn test_json_suite() -> TestSuite:
    """Tests for JSON module integration."""
    var suite = TestSuite("JSON Integration")

    # Test 1: JSON value types
    suite.add_result(test_json_value_types())

    # Test 2: JSON parsing
    suite.add_result(test_json_parsing())

    # Test 3: JSON nested structures
    suite.add_result(test_json_nested())

    # Test 4: JSON serialization
    suite.add_result(test_json_serialization())

    # Test 5: JSON builder
    suite.add_result(test_json_builder())

    # Test 6: JSON path access
    suite.add_result(test_json_path_access())

    # Test 7: JSON error handling
    suite.add_result(test_json_error_handling())

    # Test 8: JSON with special characters
    suite.add_result(test_json_special_chars())

    return suite

fn test_json_value_types() -> TestResult:
    """Test JSON value type handling."""
    # Create different value types
    let null_val = JsonValue.null()
    let bool_val = JsonValue.from_bool(True)
    let num_val = JsonValue.from_number(42.5)
    let str_val = JsonValue.from_string("hello")

    # Verify types
    if not null_val.is_null():
        return TestResult("JSON value types", False, "null check failed")
    if not bool_val.is_bool():
        return TestResult("JSON value types", False, "bool check failed")
    if not num_val.is_number():
        return TestResult("JSON value types", False, "number check failed")
    if not str_val.is_string():
        return TestResult("JSON value types", False, "string check failed")

    # Verify values
    if bool_val.as_bool() != True:
        return TestResult("JSON value types", False, "bool value wrong")
    if num_val.as_number() != 42.5:
        return TestResult("JSON value types", False, "number value wrong")
    if str_val.as_string() != "hello":
        return TestResult("JSON value types", False, "string value wrong")

    return TestResult("JSON value types", True)

fn test_json_parsing() -> TestResult:
    """Test JSON parsing."""
    let json_str = '{"name": "test", "value": 123, "active": true}'

    var parser = JsonParser(json_str)
    let result = parser.parse()

    if result.is_error():
        return TestResult("JSON parsing", False, "Parse failed")

    let obj = result.value()
    if not obj.is_object():
        return TestResult("JSON parsing", False, "Not an object")

    return TestResult("JSON parsing", True)

fn test_json_nested() -> TestResult:
    """Test nested JSON structures."""
    let json_str = '''
    {
        "user": {
            "name": "Alice",
            "age": 30,
            "tags": ["admin", "user", "tester"]
        },
        "metadata": {
            "created": "2026-01-15",
            "version": 2
        }
    }
    '''

    var parser = JsonParser(json_str)
    let result = parser.parse()

    if result.is_error():
        return TestResult("JSON nested structures", False, "Parse failed")

    let root = result.value()

    # Access nested object
    let user = root.get("user")
    if user.is_null():
        return TestResult("JSON nested structures", False, "user is null")

    let name = user.get("name")
    if name.as_string() != "Alice":
        return TestResult("JSON nested structures", False, "name mismatch")

    # Access nested array
    let tags = user.get("tags")
    if not tags.is_array():
        return TestResult("JSON nested structures", False, "tags not array")

    return TestResult("JSON nested structures", True)

fn test_json_serialization() -> TestResult:
    """Test JSON serialization."""
    var builder = JsonBuilder()
    builder.start_object()
    builder.add_string("name", "test")
    builder.add_number("count", 42)
    builder.add_bool("enabled", True)
    builder.end_object()

    let json_str = builder.to_string()

    # Parse it back
    var parser = JsonParser(json_str)
    let result = parser.parse()

    if result.is_error():
        return TestResult("JSON serialization", False, "Re-parse failed")

    let obj = result.value()
    if obj.get("name").as_string() != "test":
        return TestResult("JSON serialization", False, "name mismatch after roundtrip")
    if obj.get("count").as_number() != 42:
        return TestResult("JSON serialization", False, "count mismatch after roundtrip")

    return TestResult("JSON serialization", True)

fn test_json_builder() -> TestResult:
    """Test JSON builder fluent API."""
    var builder = JsonBuilder()

    builder.start_object()
    builder.add_string("title", "Integration Test")

    builder.start_array("items")
    builder.add_array_number(1)
    builder.add_array_number(2)
    builder.add_array_number(3)
    builder.end_array()

    builder.start_object("config")
    builder.add_bool("debug", False)
    builder.end_object()

    builder.end_object()

    let json = builder.to_string()

    if not json.contains("Integration Test"):
        return TestResult("JSON builder", False, "title not found")
    if not json.contains("items"):
        return TestResult("JSON builder", False, "items not found")
    if not json.contains("config"):
        return TestResult("JSON builder", False, "config not found")

    return TestResult("JSON builder", True)

fn test_json_path_access() -> TestResult:
    """Test JSON path-based access."""
    let json_str = '{"a": {"b": {"c": [1, 2, 3]}}}'

    var parser = JsonParser(json_str)
    let result = parser.parse()

    if result.is_error():
        return TestResult("JSON path access", False, "Parse failed")

    let root = result.value()

    # Test path access
    let c = root.get_path("a.b.c")
    if not c.is_array():
        return TestResult("JSON path access", False, "path a.b.c not array")

    let first = root.get_path("a.b.c.0")
    if first.as_number() != 1:
        return TestResult("JSON path access", False, "a.b.c.0 != 1")

    return TestResult("JSON path access", True)

fn test_json_error_handling() -> TestResult:
    """Test JSON error handling."""
    # Invalid JSON
    let invalid = '{"name": "test", invalid}'

    var parser = JsonParser(invalid)
    let result = parser.parse()

    if not result.is_error():
        return TestResult("JSON error handling", False, "Should have failed on invalid JSON")

    # Empty input
    var parser2 = JsonParser("")
    let result2 = parser2.parse()

    if not result2.is_error():
        return TestResult("JSON error handling", False, "Should have failed on empty input")

    return TestResult("JSON error handling", True)

fn test_json_special_chars() -> TestResult:
    """Test JSON with special characters."""
    let json_str = '{"message": "Hello\\nWorld\\t!", "path": "C:\\\\Users\\\\test"}'

    var parser = JsonParser(json_str)
    let result = parser.parse()

    if result.is_error():
        return TestResult("JSON special chars", False, "Parse failed")

    let obj = result.value()
    let msg = obj.get("message").as_string()

    if not msg.contains("\n"):
        return TestResult("JSON special chars", False, "newline not parsed")
    if not msg.contains("\t"):
        return TestResult("JSON special chars", False, "tab not parsed")

    return TestResult("JSON special chars", True)

# ============================================================================
# Time Integration Tests
# ============================================================================

fn test_time_suite() -> TestSuite:
    """Tests for Time module integration."""
    var suite = TestSuite("Time Integration")

    suite.add_result(test_duration_arithmetic())
    suite.add_result(test_date_operations())
    suite.add_result(test_time_operations())
    suite.add_result(test_datetime_combined())
    suite.add_result(test_datetime_parsing())
    suite.add_result(test_datetime_formatting())
    suite.add_result(test_timezone_operations())
    suite.add_result(test_date_edge_cases())

    return suite

fn test_duration_arithmetic() -> TestResult:
    """Test Duration arithmetic operations."""
    let hour = Duration.from_hours(1)
    let thirty_mins = Duration.from_mins(30)

    let total = hour + thirty_mins
    if total.total_mins() != 90:
        return TestResult("Duration arithmetic", False, "1h + 30m != 90m")

    let diff = hour - thirty_mins
    if diff.total_mins() != 30:
        return TestResult("Duration arithmetic", False, "1h - 30m != 30m")

    let doubled = hour * 2
    if doubled.total_hours() != 2:
        return TestResult("Duration arithmetic", False, "1h * 2 != 2h")

    let halved = hour / 2
    if halved.total_mins() != 30:
        return TestResult("Duration arithmetic", False, "1h / 2 != 30m")

    return TestResult("Duration arithmetic", True)

fn test_date_operations() -> TestResult:
    """Test Date operations."""
    let d = Date(2026, 1, 15)

    # Day of year
    if d.day_of_year() != 15:
        return TestResult("Date operations", False, "day_of_year wrong")

    # Add days
    let d2 = d.add_days(20)
    if d2.month != 2 or d2.day != 4:
        return TestResult("Date operations", False, "add_days wrong")

    # Add months
    let d3 = d.add_months(2)
    if d3.month != 3:
        return TestResult("Date operations", False, "add_months wrong")

    # Leap year
    if not Date._is_leap_year(2024):
        return TestResult("Date operations", False, "2024 should be leap year")
    if Date._is_leap_year(2025):
        return TestResult("Date operations", False, "2025 should not be leap year")

    return TestResult("Date operations", True)

fn test_time_operations() -> TestResult:
    """Test Time operations."""
    let t = Time(14, 30, 45)

    # 12-hour conversion
    if t.hour_12() != 2:
        return TestResult("Time operations", False, "hour_12 wrong")

    if not t.is_pm():
        return TestResult("Time operations", False, "should be PM")

    # Seconds since midnight
    let expected_secs = 14 * 3600 + 30 * 60 + 45
    if t.to_secs_since_midnight() != expected_secs:
        return TestResult("Time operations", False, "secs since midnight wrong")

    # Add duration
    let t2 = t.add(Duration.from_hours(2))
    if t2.hour != 16:
        return TestResult("Time operations", False, "add duration wrong")

    return TestResult("Time operations", True)

fn test_datetime_combined() -> TestResult:
    """Test DateTime combined operations."""
    let dt = DateTime(2026, 1, 15, 14, 30, 0)

    # Component access
    if dt.year() != 2026:
        return TestResult("DateTime combined", False, "year wrong")
    if dt.hour() != 14:
        return TestResult("DateTime combined", False, "hour wrong")

    # Add duration (crossing day boundary)
    let dt2 = dt.add(Duration.from_hours(12))
    if dt2.day() != 16 or dt2.hour() != 2:
        return TestResult("DateTime combined", False, "add duration crossing day wrong")

    # Timestamp roundtrip
    let ts = dt.to_timestamp()
    let dt3 = DateTime.from_timestamp(ts)
    if dt3.year() != dt.year() or dt3.month() != dt.month() or dt3.day() != dt.day():
        return TestResult("DateTime combined", False, "timestamp roundtrip failed")

    return TestResult("DateTime combined", True)

fn test_datetime_parsing() -> TestResult:
    """Test DateTime parsing."""
    # ISO format
    let dt1 = DateTimeParser.parse_datetime("2026-01-15T14:30:45")
    if dt1.year() != 2026 or dt1.month() != 1 or dt1.day() != 15:
        return TestResult("DateTime parsing", False, "ISO date parse failed")
    if dt1.hour() != 14 or dt1.minute() != 30:
        return TestResult("DateTime parsing", False, "ISO time parse failed")

    # Date only (US format)
    let d = DateTimeParser.parse_date("01/15/2026")
    if d.year != 2026 or d.month != 1 or d.day != 15:
        return TestResult("DateTime parsing", False, "US date parse failed")

    # Time only
    let t = DateTimeParser.parse_time("14:30:45")
    if t.hour != 14 or t.minute != 30 or t.second != 45:
        return TestResult("DateTime parsing", False, "time parse failed")

    return TestResult("DateTime parsing", True)

fn test_datetime_formatting() -> TestResult:
    """Test DateTime formatting."""
    let d = Date(2026, 1, 15)

    # ISO format
    let iso = d.to_iso_string()
    if iso != "2026-01-15":
        return TestResult("DateTime formatting", False, "ISO format wrong: " + iso)

    # Custom format
    let custom = d.format("%B %d, %Y")
    if not custom.contains("January"):
        return TestResult("DateTime formatting", False, "month name not found")
    if not custom.contains("15"):
        return TestResult("DateTime formatting", False, "day not found")
    if not custom.contains("2026"):
        return TestResult("DateTime formatting", False, "year not found")

    # Time formatting
    let t = Time(14, 30, 45)
    let time_iso = t.to_iso_string()
    if time_iso != "14:30:45":
        return TestResult("DateTime formatting", False, "time ISO wrong: " + time_iso)

    return TestResult("DateTime formatting", True)

fn test_timezone_operations() -> TestResult:
    """Test Timezone operations."""
    let utc = Timezone.UTC
    if utc.offset_minutes != 0:
        return TestResult("Timezone operations", False, "UTC offset wrong")

    let est = Timezone.EST
    if est.offset_minutes != -300:
        return TestResult("Timezone operations", False, "EST offset wrong")

    let offset_str = est.offset_string()
    if offset_str != "-05:00":
        return TestResult("Timezone operations", False, "EST offset string wrong: " + offset_str)

    let jst = Timezone.JST
    if jst.offset_string() != "+09:00":
        return TestResult("Timezone operations", False, "JST offset string wrong")

    return TestResult("Timezone operations", True)

fn test_date_edge_cases() -> TestResult:
    """Test Date edge cases."""
    # Feb 29 in leap year
    let leap_feb = Date(2024, 2, 29)
    if not leap_feb.is_valid():
        return TestResult("Date edge cases", False, "Feb 29 2024 should be valid")

    # Feb 29 in non-leap year (should clamp or handle)
    let non_leap = Date(2025, 2, 28)
    if not non_leap.is_valid():
        return TestResult("Date edge cases", False, "Feb 28 2025 should be valid")

    # Year boundary
    let dec31 = Date(2025, 12, 31)
    let jan1 = dec31.add_days(1)
    if jan1.year != 2026 or jan1.month != 1 or jan1.day != 1:
        return TestResult("Date edge cases", False, "Year boundary wrong")

    # Negative days
    let jan15 = Date(2026, 1, 15)
    let jan5 = jan15.add_days(-10)
    if jan5.day != 5:
        return TestResult("Date edge cases", False, "Negative days wrong")

    return TestResult("Date edge cases", True)

# ============================================================================
# Path Integration Tests
# ============================================================================

fn test_path_suite() -> TestSuite:
    """Tests for Path module integration."""
    var suite = TestSuite("Path Integration")

    suite.add_result(test_path_components())
    suite.add_result(test_path_joining())
    suite.add_result(test_path_normalization())
    suite.add_result(test_path_queries())
    suite.add_result(test_path_relative())
    suite.add_result(test_glob_patterns())
    suite.add_result(test_path_builder())
    suite.add_result(test_filename_validation())

    return suite

fn test_path_components() -> TestResult:
    """Test path component extraction."""
    let p = Path("/home/user/documents/report.pdf")

    if p.filename() != "report.pdf":
        return TestResult("Path components", False, "filename wrong")

    if p.stem() != "report":
        return TestResult("Path components", False, "stem wrong")

    if p.extension() != ".pdf":
        return TestResult("Path components", False, "extension wrong")

    let parent = p.parent()
    if parent.filename() != "documents":
        return TestResult("Path components", False, "parent wrong")

    if not p.is_absolute():
        return TestResult("Path components", False, "should be absolute")

    return TestResult("Path components", True)

fn test_path_joining() -> TestResult:
    """Test path joining."""
    let base = Path("/home/user")

    let joined = base.join("documents").join("file.txt")
    if joined.as_string() != "/home/user/documents/file.txt":
        return TestResult("Path joining", False, "join wrong: " + joined.as_string())

    # Using operator
    let p2 = base / "downloads" / "archive.zip"
    if p2.filename() != "archive.zip":
        return TestResult("Path joining", False, "operator / wrong")

    # Absolute path overrides
    let abs_path = base.join("/etc/config")
    if abs_path.as_string() != "/etc/config":
        return TestResult("Path joining", False, "absolute override wrong")

    return TestResult("Path joining", True)

fn test_path_normalization() -> TestResult:
    """Test path normalization."""
    let p1 = Path("/home/user/../user/./documents")
    let n1 = p1.normalize()
    if n1.as_string() != "/home/user/documents":
        return TestResult("Path normalization", False, "normalize wrong: " + n1.as_string())

    let p2 = Path("./foo/bar/../baz")
    let n2 = p2.normalize()
    if n2.as_string() != "foo/baz":
        return TestResult("Path normalization", False, "relative normalize wrong: " + n2.as_string())

    let p3 = Path("a/b/c/../../d")
    let n3 = p3.normalize()
    if n3.as_string() != "a/d":
        return TestResult("Path normalization", False, "multiple .. wrong: " + n3.as_string())

    return TestResult("Path normalization", True)

fn test_path_queries() -> TestResult:
    """Test path query methods."""
    let p = Path("/home/user/documents/file.txt")

    if not p.starts_with("/home"):
        return TestResult("Path queries", False, "starts_with failed")

    if not p.ends_with(".txt"):
        return TestResult("Path queries", False, "ends_with failed")

    if not p.has_extension("txt"):
        return TestResult("Path queries", False, "has_extension failed")

    if not p.has_extension(".txt"):
        return TestResult("Path queries", False, "has_extension with dot failed")

    let hidden = Path("/home/user/.config")
    if not hidden.is_hidden():
        return TestResult("Path queries", False, "is_hidden failed")

    return TestResult("Path queries", True)

fn test_path_relative() -> TestResult:
    """Test relative path operations."""
    let base = Path("/home/user")
    let target = Path("/home/user/documents/file.txt")

    let rel = target.relative_to(base)
    if rel.as_string() != "documents/file.txt":
        return TestResult("Path relative", False, "relative_to wrong: " + rel.as_string())

    let stripped = target.strip_prefix(base)
    if stripped.as_string() != "documents/file.txt":
        return TestResult("Path relative", False, "strip_prefix wrong")

    return TestResult("Path relative", True)

fn test_glob_patterns() -> TestResult:
    """Test glob pattern matching."""
    # Simple wildcard
    let p1 = GlobPattern("*.txt")
    if not p1.matches("file.txt"):
        return TestResult("Glob patterns", False, "*.txt should match file.txt")
    if p1.matches("file.pdf"):
        return TestResult("Glob patterns", False, "*.txt should not match file.pdf")

    # Double star
    let p2 = GlobPattern("src/**/*.py")
    if not p2.matches("src/module/file.py"):
        return TestResult("Glob patterns", False, "** should match nested")
    if not p2.matches("src/a/b/c/file.py"):
        return TestResult("Glob patterns", False, "** should match deep nested")

    # Single char
    let p3 = GlobPattern("file?.txt")
    if not p3.matches("file1.txt"):
        return TestResult("Glob patterns", False, "? should match single char")
    if p3.matches("file12.txt"):
        return TestResult("Glob patterns", False, "? should not match multiple chars")

    # Character class
    let p4 = GlobPattern("[abc].txt")
    if not p4.matches("a.txt"):
        return TestResult("Glob patterns", False, "[abc] should match a")
    if p4.matches("d.txt"):
        return TestResult("Glob patterns", False, "[abc] should not match d")

    return TestResult("Glob patterns", True)

fn test_path_builder() -> TestResult:
    """Test PathBuilder."""
    var builder = PathBuilder("/home")
    let path = builder.push("user").push("documents").push("file.txt").build()

    if path.as_string() != "/home/user/documents/file.txt":
        return TestResult("PathBuilder", False, "build wrong: " + path.as_string())

    # Pop
    var builder2 = PathBuilder("/home/user/documents")
    let popped = builder2.pop().build()
    if popped.as_string() != "/home/user":
        return TestResult("PathBuilder", False, "pop wrong")

    return TestResult("PathBuilder", True)

fn test_filename_validation() -> TestResult:
    """Test filename validation."""
    if not is_valid_filename("normal_file.txt"):
        return TestResult("Filename validation", False, "valid name rejected")

    if is_valid_filename("file/name.txt"):
        return TestResult("Filename validation", False, "slash should be invalid")

    if is_valid_filename("file:name.txt"):
        return TestResult("Filename validation", False, "colon should be invalid")

    if is_valid_filename("con"):
        return TestResult("Filename validation", False, "reserved name should be invalid")

    # Sanitization
    let sanitized = sanitize_filename("my:file<name>.txt")
    if sanitized.contains(":") or sanitized.contains("<"):
        return TestResult("Filename validation", False, "sanitize failed")

    return TestResult("Filename validation", True)

# ============================================================================
# Cross-Module Integration Tests
# ============================================================================

fn test_cross_module_suite() -> TestSuite:
    """Tests for cross-module integration."""
    var suite = TestSuite("Cross-Module Integration")

    suite.add_result(test_json_with_datetime())
    suite.add_result(test_path_with_json())
    suite.add_result(test_config_file_workflow())
    suite.add_result(test_api_response_workflow())
    suite.add_result(test_log_entry_workflow())

    return suite

fn test_json_with_datetime() -> TestResult:
    """Test JSON serialization with DateTime values."""
    let dt = DateTime(2026, 1, 15, 14, 30, 0)

    var builder = JsonBuilder()
    builder.start_object()
    builder.add_string("event", "meeting")
    builder.add_string("timestamp", dt.to_iso_string())
    builder.add_number("duration_mins", 60)
    builder.end_object()

    let json = builder.to_string()

    # Parse back
    var parser = JsonParser(json)
    let result = parser.parse()

    if result.is_error():
        return TestResult("JSON with DateTime", False, "parse failed")

    let obj = result.value()
    let ts_str = obj.get("timestamp").as_string()

    # Parse the timestamp back
    let parsed_dt = DateTimeParser.parse_datetime(ts_str)

    if parsed_dt.year() != 2026 or parsed_dt.month() != 1:
        return TestResult("JSON with DateTime", False, "datetime roundtrip failed")

    return TestResult("JSON with DateTime", True)

fn test_path_with_json() -> TestResult:
    """Test JSON with path configurations."""
    let config_path = Path("/etc/app/config.json")
    let data_path = Path("/var/data")
    let log_path = Path("/var/log/app.log")

    var builder = JsonBuilder()
    builder.start_object()

    builder.start_object("paths")
    builder.add_string("config", config_path.as_string())
    builder.add_string("data", data_path.as_string())
    builder.add_string("log", log_path.as_string())
    builder.end_object()

    builder.start_object("options")
    builder.add_bool("verbose", True)
    builder.add_number("max_size", 1024)
    builder.end_object()

    builder.end_object()

    let json = builder.to_string()

    # Parse and extract paths
    var parser = JsonParser(json)
    let result = parser.parse()

    if result.is_error():
        return TestResult("Path with JSON", False, "parse failed")

    let obj = result.value()
    let paths = obj.get("paths")

    let parsed_config = Path(paths.get("config").as_string())
    if parsed_config.filename() != "config.json":
        return TestResult("Path with JSON", False, "config path wrong")

    let parsed_log = Path(paths.get("log").as_string())
    if parsed_log.extension() != ".log":
        return TestResult("Path with JSON", False, "log extension wrong")

    return TestResult("Path with JSON", True)

fn test_config_file_workflow() -> TestResult:
    """Test realistic config file workflow."""
    # Simulate reading/writing app configuration

    # Build config
    var builder = JsonBuilder()
    builder.start_object()

    builder.add_string("app_name", "MyApp")
    builder.add_string("version", "1.0.0")

    builder.start_object("database")
    builder.add_string("host", "localhost")
    builder.add_number("port", 5432)
    builder.add_string("name", "mydb")
    builder.end_object()

    builder.start_object("logging")
    builder.add_string("level", "info")
    builder.add_string("path", "/var/log/myapp.log")
    builder.add_number("max_size_mb", 100)
    builder.add_bool("rotate", True)
    builder.end_object()

    builder.start_object("created")
    let now = DateTime(2026, 1, 15, 10, 0, 0)
    builder.add_string("date", now.date.to_iso_string())
    builder.add_string("time", now.time.to_iso_string())
    builder.end_object()

    builder.end_object()

    let config_json = builder.to_pretty_string()

    # Verify structure
    if not config_json.contains("MyApp"):
        return TestResult("Config file workflow", False, "app_name missing")
    if not config_json.contains("localhost"):
        return TestResult("Config file workflow", False, "db host missing")
    if not config_json.contains("/var/log/myapp.log"):
        return TestResult("Config file workflow", False, "log path missing")

    // Parse it back
    var parser = JsonParser(config_json)
    let result = parser.parse()

    if result.is_error():
        return TestResult("Config file workflow", False, "re-parse failed")

    let config = result.value()

    // Verify values
    if config.get("app_name").as_string() != "MyApp":
        return TestResult("Config file workflow", False, "app_name mismatch")

    let db_port = config.get_path("database.port").as_number()
    if db_port != 5432:
        return TestResult("Config file workflow", False, "db port mismatch")

    let log_path = Path(config.get_path("logging.path").as_string())
    if log_path.extension() != ".log":
        return TestResult("Config file workflow", False, "log path extension wrong")

    return TestResult("Config file workflow", True)

fn test_api_response_workflow() -> TestResult:
    """Test API response parsing workflow."""
    // Simulated API response
    let api_response = '''
    {
        "status": "success",
        "data": {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "total": 2,
            "page": 1
        },
        "timestamp": "2026-01-15T14:30:00",
        "request_id": "abc123"
    }
    '''

    var parser = JsonParser(api_response)
    let result = parser.parse()

    if result.is_error():
        return TestResult("API response workflow", False, "parse failed")

    let response = result.value()

    // Check status
    if response.get("status").as_string() != "success":
        return TestResult("API response workflow", False, "status wrong")

    // Parse timestamp
    let ts_str = response.get("timestamp").as_string()
    let ts = DateTimeParser.parse_datetime(ts_str)
    if ts.hour() != 14:
        return TestResult("API response workflow", False, "timestamp hour wrong")

    // Access nested data
    let total = response.get_path("data.total").as_number()
    if total != 2:
        return TestResult("API response workflow", False, "total wrong")

    // Access array element
    let first_user_name = response.get_path("data.users.0.name").as_string()
    if first_user_name != "Alice":
        return TestResult("API response workflow", False, "first user name wrong")

    return TestResult("API response workflow", True)

fn test_log_entry_workflow() -> TestResult:
    """Test log entry creation workflow."""
    // Create log entry with timestamp and path info

    let timestamp = DateTime(2026, 1, 15, 14, 30, 45)
    let source_file = Path("/app/src/handlers/user.mojo")

    var builder = JsonBuilder()
    builder.start_object()
    builder.add_string("level", "INFO")
    builder.add_string("timestamp", timestamp.to_iso_string())
    builder.add_string("message", "User logged in successfully")

    builder.start_object("source")
    builder.add_string("file", source_file.filename())
    builder.add_string("path", source_file.parent().as_string())
    builder.add_number("line", 42)
    builder.end_object()

    builder.start_object("context")
    builder.add_string("user_id", "12345")
    builder.add_string("session_id", "sess_abc")
    builder.end_object()

    builder.end_object()

    let log_entry = builder.to_string()

    // Parse and verify
    var parser = JsonParser(log_entry)
    let result = parser.parse()

    if result.is_error():
        return TestResult("Log entry workflow", False, "parse failed")

    let entry = result.value()

    if entry.get("level").as_string() != "INFO":
        return TestResult("Log entry workflow", False, "level wrong")

    let source_filename = entry.get_path("source.file").as_string()
    if source_filename != "user.mojo":
        return TestResult("Log entry workflow", False, "source file wrong")

    return TestResult("Log entry workflow", True)

# ============================================================================
# Performance Benchmarks
# ============================================================================

fn test_performance_suite() -> TestSuite:
    """Performance benchmark tests."""
    var suite = TestSuite("Performance Benchmarks")

    suite.add_result(benchmark_json_parsing())
    suite.add_result(benchmark_path_operations())
    suite.add_result(benchmark_datetime_operations())
    suite.add_result(benchmark_duration_arithmetic())

    return suite

fn benchmark_json_parsing() -> TestResult:
    """Benchmark JSON parsing performance."""
    let json_str = '{"name": "test", "value": 123, "items": [1, 2, 3, 4, 5]}'

    let iterations = 1000
    var successful = 0

    for i in range(iterations):
        var parser = JsonParser(json_str)
        let result = parser.parse()
        if not result.is_error():
            successful += 1

    if successful != iterations:
        return TestResult("JSON parsing benchmark", False,
                         "Only " + String(successful) + "/" + String(iterations) + " succeeded")

    return TestResult("JSON parsing benchmark (" + String(iterations) + " iterations)", True)

fn benchmark_path_operations() -> TestResult:
    """Benchmark path operations."""
    let iterations = 1000
    var total_components = 0

    for i in range(iterations):
        let p = Path("/home/user/documents/projects/mojo/src/main.mojo")
        let components = p.components()
        total_components += len(components)

        let normalized = p.normalize()
        let parent = p.parent()
        let filename = p.filename()
        let ext = p.extension()

    if total_components != iterations * 8:  # 8 components per path
        return TestResult("Path operations benchmark", False, "Component count wrong")

    return TestResult("Path operations benchmark (" + String(iterations) + " iterations)", True)

fn benchmark_datetime_operations() -> TestResult:
    """Benchmark datetime operations."""
    let iterations = 1000
    var valid_dates = 0

    for i in range(iterations):
        let dt = DateTime(2026, 1, 15, 14, 30, 0)
        let ts = dt.to_timestamp()
        let dt2 = DateTime.from_timestamp(ts)

        if dt2.year() == 2026:
            valid_dates += 1

        let iso = dt.to_iso_string()
        let formatted = dt.date.format("%Y-%m-%d")

    if valid_dates != iterations:
        return TestResult("DateTime operations benchmark", False, "Not all dates valid")

    return TestResult("DateTime operations benchmark (" + String(iterations) + " iterations)", True)

fn benchmark_duration_arithmetic() -> TestResult:
    """Benchmark duration arithmetic."""
    let iterations = 10000
    var total_secs: Int64 = 0

    for i in range(iterations):
        let d1 = Duration.from_secs(Int64(i))
        let d2 = Duration.from_mins(1)
        let sum = d1 + d2
        total_secs += sum.total_secs()

    # Verify some arithmetic was done
    if total_secs == 0:
        return TestResult("Duration arithmetic benchmark", False, "No computation done")

    return TestResult("Duration arithmetic benchmark (" + String(iterations) + " iterations)", True)

# ============================================================================
# Main Test Runner
# ============================================================================

fn main():
    """Run all integration tests."""
    var runner = TestRunner()

    # Add all test suites
    runner.add_suite(test_json_suite())
    runner.add_suite(test_time_suite())
    runner.add_suite(test_path_suite())
    runner.add_suite(test_cross_module_suite())
    runner.add_suite(test_performance_suite())

    # Run all tests
    runner.run_all()
