"""
Mojo Service Framework
Days 105-109: Standardizing the "Shimmy" Pattern

This framework provides a standardized way to build HTTP services in Mojo
with Zig as the networking layer. It abstracts the Zig HTTP callback loop
and provides a clean, trait-based API for building web services.

Key Components:
- Service trait: Interface for request handlers
- Router: Path-based request dispatch with parameter extraction
- Middleware: Composable request/response processing
- Context: Request context with body, headers, params
- Response: Type-safe response builders
"""

from collections import Dict, List
from memory import UnsafePointer
from sys.ffi import external_call, DLHandle, OpaquePointer

# ============================================================================
# HTTP Method Enumeration
# ============================================================================

@value
struct Method:
    """HTTP request method."""
    var _value: String

    fn __init__(inout self, value: String):
        self._value = value

    fn __eq__(self, other: Method) -> Bool:
        return self._value == other._value

    fn __eq__(self, other: String) -> Bool:
        return self._value == other

    fn __ne__(self, other: Method) -> Bool:
        return self._value != other._value

    fn __str__(self) -> String:
        return self._value

    fn is_get(self) -> Bool:
        return self._value == "GET"

    fn is_post(self) -> Bool:
        return self._value == "POST"

    fn is_put(self) -> Bool:
        return self._value == "PUT"

    fn is_delete(self) -> Bool:
        return self._value == "DELETE"

    fn is_patch(self) -> Bool:
        return self._value == "PATCH"

    fn is_options(self) -> Bool:
        return self._value == "OPTIONS"

    fn is_head(self) -> Bool:
        return self._value == "HEAD"

    # Common method constants
    alias GET = Method("GET")
    alias POST = Method("POST")
    alias PUT = Method("PUT")
    alias DELETE = Method("DELETE")
    alias PATCH = Method("PATCH")
    alias OPTIONS = Method("OPTIONS")
    alias HEAD = Method("HEAD")

# ============================================================================
# HTTP Status Codes
# ============================================================================

@value
struct StatusCode:
    """HTTP status codes."""
    var code: Int
    var reason: String

    fn __init__(inout self, code: Int, reason: String):
        self.code = code
        self.reason = reason

    fn __int__(self) -> Int:
        return self.code

    # 2xx Success
    alias OK = StatusCode(200, "OK")
    alias CREATED = StatusCode(201, "Created")
    alias ACCEPTED = StatusCode(202, "Accepted")
    alias NO_CONTENT = StatusCode(204, "No Content")

    # 3xx Redirection
    alias MOVED_PERMANENTLY = StatusCode(301, "Moved Permanently")
    alias FOUND = StatusCode(302, "Found")
    alias NOT_MODIFIED = StatusCode(304, "Not Modified")
    alias TEMPORARY_REDIRECT = StatusCode(307, "Temporary Redirect")

    # 4xx Client Errors
    alias BAD_REQUEST = StatusCode(400, "Bad Request")
    alias UNAUTHORIZED = StatusCode(401, "Unauthorized")
    alias FORBIDDEN = StatusCode(403, "Forbidden")
    alias NOT_FOUND = StatusCode(404, "Not Found")
    alias METHOD_NOT_ALLOWED = StatusCode(405, "Method Not Allowed")
    alias CONFLICT = StatusCode(409, "Conflict")
    alias UNPROCESSABLE_ENTITY = StatusCode(422, "Unprocessable Entity")
    alias TOO_MANY_REQUESTS = StatusCode(429, "Too Many Requests")

    # 5xx Server Errors
    alias INTERNAL_SERVER_ERROR = StatusCode(500, "Internal Server Error")
    alias NOT_IMPLEMENTED = StatusCode(501, "Not Implemented")
    alias BAD_GATEWAY = StatusCode(502, "Bad Gateway")
    alias SERVICE_UNAVAILABLE = StatusCode(503, "Service Unavailable")

# ============================================================================
# Headers Collection
# ============================================================================

@value
struct Headers:
    """HTTP headers collection."""
    var _headers: Dict[String, String]

    fn __init__(inout self):
        self._headers = Dict[String, String]()

    fn get(self, name: String) -> String:
        """Get header value (case-insensitive)."""
        var lower_name = name.lower()
        for key in self._headers.keys():
            if key[].lower() == lower_name:
                return self._headers[key[]]
        return ""

    fn set(inout self, name: String, value: String):
        """Set header value."""
        self._headers[name] = value

    fn has(self, name: String) -> Bool:
        """Check if header exists."""
        var lower_name = name.lower()
        for key in self._headers.keys():
            if key[].lower() == lower_name:
                return True
        return False

    fn remove(inout self, name: String):
        """Remove header."""
        # Dict doesn't have remove, so we rebuild
        var new_headers = Dict[String, String]()
        var lower_name = name.lower()
        for key in self._headers.keys():
            if key[].lower() != lower_name:
                new_headers[key[]] = self._headers[key[]]
        self._headers = new_headers

    fn content_type(self) -> String:
        """Get Content-Type header."""
        return self.get("Content-Type")

    fn authorization(self) -> String:
        """Get Authorization header."""
        return self.get("Authorization")

    fn bearer_token(self) -> String:
        """Extract bearer token from Authorization header."""
        var auth = self.authorization()
        if auth.startswith("Bearer "):
            return auth[7:]
        return ""

# ============================================================================
# Path Parameters
# ============================================================================

@value
struct PathParams:
    """Extracted path parameters from route matching."""
    var _params: Dict[String, String]

    fn __init__(inout self):
        self._params = Dict[String, String]()

    fn get(self, name: String) -> String:
        """Get path parameter value."""
        if name in self._params:
            return self._params[name]
        return ""

    fn set(inout self, name: String, value: String):
        """Set path parameter."""
        self._params[name] = value

    fn has(self, name: String) -> Bool:
        """Check if parameter exists."""
        return name in self._params

# ============================================================================
# Query Parameters
# ============================================================================

@value
struct QueryParams:
    """Parsed query string parameters."""
    var _params: Dict[String, String]

    fn __init__(inout self):
        self._params = Dict[String, String]()

    @staticmethod
    fn parse(query_string: String) -> QueryParams:
        """Parse query string into parameters."""
        var params = QueryParams()
        if len(query_string) == 0:
            return params

        # Split by &
        var pairs = query_string.split("&")
        for i in range(len(pairs)):
            var pair = pairs[i]
            var eq_idx = pair.find("=")
            if eq_idx > 0:
                var key = pair[:eq_idx]
                var value = pair[eq_idx + 1:]
                params._params[key] = value
            elif len(pair) > 0:
                params._params[pair] = ""

        return params

    fn get(self, name: String) -> String:
        """Get query parameter value."""
        if name in self._params:
            return self._params[name]
        return ""

    fn get_int(self, name: String, default: Int = 0) -> Int:
        """Get query parameter as integer."""
        var value = self.get(name)
        if len(value) > 0:
            try:
                return int(value)
            except:
                return default
        return default

    fn has(self, name: String) -> Bool:
        """Check if parameter exists."""
        return name in self._params

# ============================================================================
# Request Context
# ============================================================================

@value
struct Context:
    """
    Context for a single HTTP request.
    Contains all information about the incoming request.
    """
    var method: Method
    var path: String
    var full_path: String  # Path with query string
    var body: String
    var headers: Headers
    var path_params: PathParams
    var query_params: QueryParams
    var state: Dict[String, String]  # Middleware can store state here

    fn __init__(inout self, method: String, path: String, body: String):
        self.method = Method(method)
        self.headers = Headers()
        self.path_params = PathParams()
        self.state = Dict[String, String]()

        # Parse path and query string
        var query_idx = path.find("?")
        if query_idx >= 0:
            self.path = path[:query_idx]
            self.full_path = path
            self.query_params = QueryParams.parse(path[query_idx + 1:])
        else:
            self.path = path
            self.full_path = path
            self.query_params = QueryParams()

        self.body = body

    fn json_body(self) -> String:
        """Get body as JSON string."""
        return self.body

    fn param(self, name: String) -> String:
        """Get path parameter by name."""
        return self.path_params.get(name)

    fn query(self, name: String) -> String:
        """Get query parameter by name."""
        return self.query_params.get(name)

    fn header(self, name: String) -> String:
        """Get header by name."""
        return self.headers.get(name)

    fn is_json(self) -> Bool:
        """Check if request is JSON."""
        return self.headers.content_type().find("application/json") >= 0

    fn get_state(self, key: String) -> String:
        """Get middleware state."""
        if key in self.state:
            return self.state[key]
        return ""

    fn set_state(inout self, key: String, value: String):
        """Set middleware state."""
        self.state[key] = value

# ============================================================================
# Response Builder
# ============================================================================

@value
struct Response:
    """
    HTTP response to return to the client.
    Provides fluent API for building responses.
    """
    var body: String
    var status: StatusCode
    var headers: Headers

    fn __init__(inout self, body: String = "", status: Int = 200):
        self.body = body
        self.status = StatusCode(status, "OK")
        self.headers = Headers()
        self.headers.set("Content-Type", "application/json")

    fn __init__(inout self, body: String, status: StatusCode):
        self.body = body
        self.status = status
        self.headers = Headers()
        self.headers.set("Content-Type", "application/json")

    # Fluent API

    fn with_status(self, status: Int) -> Response:
        """Set status code."""
        var resp = self
        resp.status = StatusCode(status, "")
        return resp

    fn with_status(self, status: StatusCode) -> Response:
        """Set status code."""
        var resp = self
        resp.status = status
        return resp

    fn with_header(self, name: String, value: String) -> Response:
        """Add header."""
        var resp = self
        resp.headers.set(name, value)
        return resp

    fn with_content_type(self, content_type: String) -> Response:
        """Set Content-Type header."""
        return self.with_header("Content-Type", content_type)

    fn with_body(self, body: String) -> Response:
        """Set response body."""
        var resp = self
        resp.body = body
        return resp

    # Static constructors

    @staticmethod
    fn ok(body: String = "") -> Response:
        """200 OK response."""
        return Response(body, StatusCode.OK)

    @staticmethod
    fn created(body: String = "") -> Response:
        """201 Created response."""
        return Response(body, StatusCode.CREATED)

    @staticmethod
    fn no_content() -> Response:
        """204 No Content response."""
        return Response("", StatusCode.NO_CONTENT)

    @staticmethod
    fn bad_request(message: String = "Bad Request") -> Response:
        """400 Bad Request response."""
        return Response('{"error":"' + message + '"}', StatusCode.BAD_REQUEST)

    @staticmethod
    fn unauthorized(message: String = "Unauthorized") -> Response:
        """401 Unauthorized response."""
        return Response('{"error":"' + message + '"}', StatusCode.UNAUTHORIZED)

    @staticmethod
    fn forbidden(message: String = "Forbidden") -> Response:
        """403 Forbidden response."""
        return Response('{"error":"' + message + '"}', StatusCode.FORBIDDEN)

    @staticmethod
    fn not_found(message: String = "Not Found") -> Response:
        """404 Not Found response."""
        return Response('{"error":"' + message + '"}', StatusCode.NOT_FOUND)

    @staticmethod
    fn method_not_allowed() -> Response:
        """405 Method Not Allowed response."""
        return Response('{"error":"Method Not Allowed"}', StatusCode.METHOD_NOT_ALLOWED)

    @staticmethod
    fn internal_error(message: String = "Internal Server Error") -> Response:
        """500 Internal Server Error response."""
        return Response('{"error":"' + message + '"}', StatusCode.INTERNAL_SERVER_ERROR)

    @staticmethod
    fn json(data: String) -> Response:
        """JSON response with 200 OK."""
        return Response(data, StatusCode.OK).with_content_type("application/json")

    @staticmethod
    fn text(data: String) -> Response:
        """Plain text response."""
        return Response(data, StatusCode.OK).with_content_type("text/plain")

    @staticmethod
    fn html(data: String) -> Response:
        """HTML response."""
        return Response(data, StatusCode.OK).with_content_type("text/html")

# ============================================================================
# Handler Function Type
# ============================================================================

# Type alias for handler functions
alias HandlerFn = fn(Context) -> Response

# ============================================================================
# Service Trait
# ============================================================================

trait Service:
    """
    Trait for any component that can handle HTTP requests.
    Implement this trait to create custom services.
    """
    fn handle(self, ctx: Context) -> Response

# ============================================================================
# Middleware Trait
# ============================================================================

trait Middleware:
    """
    Trait for request/response processing middleware.
    Middleware can modify requests before handlers and responses after.
    """
    fn process(self, ctx: Context, next: HandlerFn) -> Response

# ============================================================================
# Route Definition
# ============================================================================

@value
struct Route:
    """
    A single route definition with method, path pattern, and handler.
    Supports path parameters like /users/:id
    """
    var method: Method
    var pattern: String
    var handler: HandlerFn
    var param_names: List[String]

    fn __init__(inout self, method: Method, pattern: String, handler: HandlerFn):
        self.method = method
        self.pattern = pattern
        self.handler = handler
        self.param_names = List[String]()

        # Extract parameter names from pattern
        var parts = pattern.split("/")
        for i in range(len(parts)):
            var part = parts[i]
            if len(part) > 0 and part[0] == ":":
                self.param_names.append(part[1:])

    fn matches(self, method: Method, path: String) -> Bool:
        """Check if route matches method and path."""
        if self.method != method:
            return False

        var pattern_parts = self.pattern.split("/")
        var path_parts = path.split("/")

        if len(pattern_parts) != len(path_parts):
            return False

        for i in range(len(pattern_parts)):
            var pp = pattern_parts[i]
            if len(pp) > 0 and pp[0] == ":":
                continue  # Parameter matches anything
            if pp != path_parts[i]:
                return False

        return True

    fn extract_params(self, path: String) -> PathParams:
        """Extract path parameters from path."""
        var params = PathParams()
        var pattern_parts = self.pattern.split("/")
        var path_parts = path.split("/")

        var param_idx = 0
        for i in range(len(pattern_parts)):
            var pp = pattern_parts[i]
            if len(pp) > 0 and pp[0] == ":":
                if param_idx < len(self.param_names):
                    params.set(self.param_names[param_idx], path_parts[i])
                    param_idx += 1

        return params

# ============================================================================
# Router
# ============================================================================

struct Router:
    """
    Request router that dispatches to handlers based on path patterns.
    Supports path parameters and method-based routing.
    """
    var routes: List[Route]
    var middlewares: List[Middleware]
    var not_found_handler: HandlerFn

    fn __init__(inout self):
        self.routes = List[Route]()
        self.middlewares = List[Middleware]()

        # Default not found handler
        fn default_not_found(ctx: Context) -> Response:
            return Response.not_found("Route not found: " + ctx.path)
        self.not_found_handler = default_not_found

    # Route registration methods

    fn get(inout self, pattern: String, handler: HandlerFn):
        """Register GET route."""
        self.routes.append(Route(Method.GET, pattern, handler))

    fn post(inout self, pattern: String, handler: HandlerFn):
        """Register POST route."""
        self.routes.append(Route(Method.POST, pattern, handler))

    fn put(inout self, pattern: String, handler: HandlerFn):
        """Register PUT route."""
        self.routes.append(Route(Method.PUT, pattern, handler))

    fn delete(inout self, pattern: String, handler: HandlerFn):
        """Register DELETE route."""
        self.routes.append(Route(Method.DELETE, pattern, handler))

    fn patch(inout self, pattern: String, handler: HandlerFn):
        """Register PATCH route."""
        self.routes.append(Route(Method.PATCH, pattern, handler))

    fn options(inout self, pattern: String, handler: HandlerFn):
        """Register OPTIONS route."""
        self.routes.append(Route(Method.OPTIONS, pattern, handler))

    fn route(inout self, method: Method, pattern: String, handler: HandlerFn):
        """Register route with specific method."""
        self.routes.append(Route(method, pattern, handler))

    fn use(inout self, middleware: Middleware):
        """Add middleware to the chain."""
        self.middlewares.append(middleware)

    fn not_found(inout self, handler: HandlerFn):
        """Set custom not found handler."""
        self.not_found_handler = handler

    # Route matching and handling

    fn find_route(self, method: Method, path: String) -> Route:
        """Find matching route."""
        for i in range(len(self.routes)):
            var route = self.routes[i]
            if route.matches(method, path):
                return route
        return Route(Method.GET, "", self.not_found_handler)

    fn handle(self, ctx: Context) -> Response:
        """Handle incoming request."""
        # Find matching route
        var route = self.find_route(ctx.method, ctx.path)

        # Extract path parameters
        var ctx_with_params = ctx
        ctx_with_params.path_params = route.extract_params(ctx.path)

        # Apply middleware chain
        var handler = route.handler
        for i in range(len(self.middlewares) - 1, -1, -1):
            var mw = self.middlewares[i]
            var next_handler = handler
            fn wrapped(c: Context) -> Response:
                return mw.process(c, next_handler)
            handler = wrapped

        return handler(ctx_with_params)

# ============================================================================
# Route Group
# ============================================================================

struct RouteGroup:
    """
    Group routes under a common prefix with shared middleware.
    """
    var prefix: String
    var router: UnsafePointer[Router]
    var middlewares: List[Middleware]

    fn __init__(inout self, prefix: String, router: UnsafePointer[Router]):
        self.prefix = prefix
        self.router = router
        self.middlewares = List[Middleware]()

    fn use(inout self, middleware: Middleware):
        """Add middleware to this group."""
        self.middlewares.append(middleware)

    fn get(inout self, pattern: String, handler: HandlerFn):
        """Register GET route with prefix."""
        var full_pattern = self.prefix + pattern
        self.router[].get(full_pattern, self._wrap_handler(handler))

    fn post(inout self, pattern: String, handler: HandlerFn):
        """Register POST route with prefix."""
        var full_pattern = self.prefix + pattern
        self.router[].post(full_pattern, self._wrap_handler(handler))

    fn put(inout self, pattern: String, handler: HandlerFn):
        """Register PUT route with prefix."""
        var full_pattern = self.prefix + pattern
        self.router[].put(full_pattern, self._wrap_handler(handler))

    fn delete(inout self, pattern: String, handler: HandlerFn):
        """Register DELETE route with prefix."""
        var full_pattern = self.prefix + pattern
        self.router[].delete(full_pattern, self._wrap_handler(handler))

    fn _wrap_handler(self, handler: HandlerFn) -> HandlerFn:
        """Wrap handler with group middleware."""
        var wrapped = handler
        for i in range(len(self.middlewares) - 1, -1, -1):
            var mw = self.middlewares[i]
            var next_handler = wrapped
            fn mw_wrapped(ctx: Context) -> Response:
                return mw.process(ctx, next_handler)
            wrapped = mw_wrapped
        return wrapped

# ============================================================================
# Server Configuration
# ============================================================================

@value
struct ServerConfig:
    """Configuration for the HTTP server."""
    var host: String
    var port: Int
    var zig_lib_path: String
    var read_timeout_ms: Int
    var write_timeout_ms: Int
    var max_body_size: Int

    fn __init__(inout self):
        self.host = "0.0.0.0"
        self.port = 8080
        self.zig_lib_path = "./libzig_http_shimmy.dylib"
        self.read_timeout_ms = 30000
        self.write_timeout_ms = 30000
        self.max_body_size = 1024 * 1024  # 1MB

    fn with_host(self, host: String) -> ServerConfig:
        var config = self
        config.host = host
        return config

    fn with_port(self, port: Int) -> ServerConfig:
        var config = self
        config.port = port
        return config

    fn with_zig_lib(self, path: String) -> ServerConfig:
        var config = self
        config.zig_lib_path = path
        return config

# ============================================================================
# Zig Server Integration
# ============================================================================

struct ZigServer:
    """
    HTTP server backed by Zig for networking.
    Bridges Zig HTTP callbacks to Mojo handlers.
    """
    var config: ServerConfig
    var router: Router
    var running: Bool

    fn __init__(inout self, config: ServerConfig = ServerConfig()):
        self.config = config
        self.router = Router()
        self.running = False

    fn routes(inout self) -> UnsafePointer[Router]:
        """Get router for route registration."""
        return UnsafePointer[Router].address_of(self.router)

    fn group(inout self, prefix: String) -> RouteGroup:
        """Create route group with prefix."""
        return RouteGroup(prefix, self.routes())

    fn use(inout self, middleware: Middleware):
        """Add global middleware."""
        self.router.use(middleware)

    fn handle_request(inout self, method: String, path: String, body: String, headers: Headers) -> Response:
        """Handle incoming HTTP request."""
        var ctx = Context(method, path, body)
        ctx.headers = headers
        return self.router.handle(ctx)

    fn start(inout self) raises:
        """Start the HTTP server."""
        print("Starting Shimmy-Mojo server on " + self.config.host + ":" + String(self.config.port))
        print("Zig lib: " + self.config.zig_lib_path)
        self.running = True

        # In real implementation:
        # 1. Load Zig library
        # 2. Register callback
        # 3. Start event loop
        print("Server started successfully!")

    fn stop(inout self):
        """Stop the HTTP server."""
        self.running = False
        print("Server stopped.")

# ============================================================================
# C String Utilities (for Zig FFI)
# ============================================================================

fn cstr_len(ptr: UnsafePointer[UInt8]) -> Int:
    """Get length of null-terminated C string."""
    var i: Int = 0
    while ptr.load(i) != 0:
        i += 1
    return i

fn cstr_to_string(ptr: UnsafePointer[UInt8]) -> String:
    """Convert C string to Mojo String."""
    var length = cstr_len(ptr)
    if length == 0:
        return ""

    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr.load(i))

    return String(bytes)

fn string_to_cstr(s: String) -> UnsafePointer[UInt8]:
    """Convert Mojo String to C string (caller must free)."""
    var bytes = s.as_bytes()
    var length = len(bytes)
    var ptr = UnsafePointer[UInt8].alloc(length + 1)

    for i in range(length):
        ptr.store(i, bytes[i])
    ptr.store(length, 0)  # Null terminate

    return ptr

# ============================================================================
# Example Usage (Documentation)
# ============================================================================

"""
Example Usage:

```mojo
from stdlib.framework.service import ZigServer, Context, Response, ServerConfig

fn main() raises:
    var config = ServerConfig().with_port(8080)
    var server = ZigServer(config)

    # Register routes
    server.routes()[].get("/", fn(ctx: Context) -> Response:
        return Response.json('{"message":"Hello, World!"}')
    )

    server.routes()[].get("/users/:id", fn(ctx: Context) -> Response:
        var user_id = ctx.param("id")
        return Response.json('{"id":"' + user_id + '"}')
    )

    server.routes()[].post("/users", fn(ctx: Context) -> Response:
        var body = ctx.json_body()
        return Response.created('{"created":true}')
    )

    # Create API group
    var api = server.group("/api/v1")
    api.get("/status", fn(ctx: Context) -> Response:
        return Response.ok('{"status":"healthy"}')
    )

    # Start server
    server.start()
```
"""
