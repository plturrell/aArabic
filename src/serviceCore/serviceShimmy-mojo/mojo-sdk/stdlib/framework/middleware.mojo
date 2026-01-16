"""
Mojo Service Framework - Middleware
Days 105-109: Common Middleware Implementations

This module provides ready-to-use middleware for common HTTP concerns:
- Logging: Request/response logging with timing
- Auth: Bearer token authentication
- CORS: Cross-Origin Resource Sharing
- RateLimit: Request rate limiting
- Recovery: Panic recovery (error handling)
"""

from collections import Dict, List
from .service import Context, Response, Middleware, HandlerFn, StatusCode

# ============================================================================
# Logging Middleware
# ============================================================================

struct LoggingMiddleware(Middleware):
    """
    Logs all incoming requests and outgoing responses.
    Includes timing information.
    """
    var log_body: Bool
    var log_headers: Bool

    fn __init__(inout self, log_body: Bool = False, log_headers: Bool = False):
        self.log_body = log_body
        self.log_headers = log_headers

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Log request, call handler, log response."""
        # Log request
        var log_msg = "[" + String(ctx.method) + "] " + ctx.path
        print(log_msg)

        if self.log_headers:
            print("  Headers: (headers logging)")

        if self.log_body and len(ctx.body) > 0:
            var body_preview = ctx.body
            if len(body_preview) > 100:
                body_preview = body_preview[:100] + "..."
            print("  Body: " + body_preview)

        # Call next handler
        var response = next(ctx)

        # Log response
        print("  -> " + String(response.status.code) + " " + response.status.reason)

        return response

# ============================================================================
# Authentication Middleware
# ============================================================================

struct AuthMiddleware(Middleware):
    """
    Bearer token authentication middleware.
    Validates Authorization header and extracts user info.
    """
    var validate_fn: fn(String) -> Bool
    var skip_paths: List[String]

    fn __init__(inout self, validate_fn: fn(String) -> Bool):
        self.validate_fn = validate_fn
        self.skip_paths = List[String]()

    fn skip(inout self, path: String):
        """Add path to skip authentication."""
        self.skip_paths.append(path)

    fn should_skip(self, path: String) -> Bool:
        """Check if path should skip auth."""
        for i in range(len(self.skip_paths)):
            if self.skip_paths[i] == path:
                return True
            # Support wildcard prefix
            var skip_path = self.skip_paths[i]
            if skip_path.endswith("*"):
                var prefix = skip_path[:-1]
                if path.startswith(prefix):
                    return True
        return False

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Validate token and proceed or reject."""
        # Check if path should skip auth
        if self.should_skip(ctx.path):
            return next(ctx)

        # Get bearer token
        var token = ctx.headers.bearer_token()

        if len(token) == 0:
            return Response.unauthorized("Missing authentication token")

        # Validate token
        if not self.validate_fn(token):
            return Response.unauthorized("Invalid authentication token")

        # Token valid, set user info in context
        var ctx_with_auth = ctx
        ctx_with_auth.set_state("auth_token", token)

        return next(ctx_with_auth)

# ============================================================================
# API Key Authentication
# ============================================================================

struct ApiKeyMiddleware(Middleware):
    """
    API key authentication via header or query parameter.
    """
    var valid_keys: List[String]
    var header_name: String
    var query_param: String

    fn __init__(inout self, header_name: String = "X-API-Key", query_param: String = "api_key"):
        self.valid_keys = List[String]()
        self.header_name = header_name
        self.query_param = query_param

    fn add_key(inout self, key: String):
        """Add valid API key."""
        self.valid_keys.append(key)

    fn is_valid(self, key: String) -> Bool:
        """Check if key is valid."""
        for i in range(len(self.valid_keys)):
            if self.valid_keys[i] == key:
                return True
        return False

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Validate API key and proceed or reject."""
        # Check header first
        var key = ctx.headers.get(self.header_name)

        # Fall back to query param
        if len(key) == 0:
            key = ctx.query(self.query_param)

        if len(key) == 0:
            return Response.unauthorized("Missing API key")

        if not self.is_valid(key):
            return Response.unauthorized("Invalid API key")

        return next(ctx)

# ============================================================================
# CORS Middleware
# ============================================================================

struct CorsMiddleware(Middleware):
    """
    Cross-Origin Resource Sharing (CORS) middleware.
    Handles preflight requests and adds CORS headers.
    """
    var allowed_origins: List[String]
    var allowed_methods: List[String]
    var allowed_headers: List[String]
    var allow_credentials: Bool
    var max_age: Int

    fn __init__(inout self):
        self.allowed_origins = List[String]()
        self.allowed_methods = List[String]()
        self.allowed_headers = List[String]()
        self.allow_credentials = False
        self.max_age = 86400  # 24 hours

        # Default methods
        self.allowed_methods.append("GET")
        self.allowed_methods.append("POST")
        self.allowed_methods.append("PUT")
        self.allowed_methods.append("DELETE")
        self.allowed_methods.append("OPTIONS")

        # Default headers
        self.allowed_headers.append("Content-Type")
        self.allowed_headers.append("Authorization")
        self.allowed_headers.append("X-Requested-With")

    fn allow_origin(inout self, origin: String):
        """Add allowed origin."""
        self.allowed_origins.append(origin)

    fn allow_all_origins(inout self):
        """Allow all origins (*)."""
        self.allowed_origins = List[String]()
        self.allowed_origins.append("*")

    fn allow_method(inout self, method: String):
        """Add allowed method."""
        self.allowed_methods.append(method)

    fn allow_header(inout self, header: String):
        """Add allowed header."""
        self.allowed_headers.append(header)

    fn with_credentials(inout self):
        """Enable credentials support."""
        self.allow_credentials = True

    fn _join_list(self, items: List[String], sep: String) -> String:
        """Join list items with separator."""
        var result = String("")
        for i in range(len(items)):
            if i > 0:
                result += sep
            result += items[i]
        return result

    fn _is_origin_allowed(self, origin: String) -> Bool:
        """Check if origin is allowed."""
        for i in range(len(self.allowed_origins)):
            if self.allowed_origins[i] == "*" or self.allowed_origins[i] == origin:
                return True
        return False

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Add CORS headers and handle preflight."""
        var origin = ctx.headers.get("Origin")

        # No origin header, proceed normally
        if len(origin) == 0:
            return next(ctx)

        # Check if origin allowed
        if not self._is_origin_allowed(origin):
            return Response.forbidden("Origin not allowed")

        # Handle preflight (OPTIONS)
        if ctx.method.is_options():
            var response = Response.no_content()
            response = response.with_header("Access-Control-Allow-Origin", origin)
            response = response.with_header("Access-Control-Allow-Methods", self._join_list(self.allowed_methods, ", "))
            response = response.with_header("Access-Control-Allow-Headers", self._join_list(self.allowed_headers, ", "))
            response = response.with_header("Access-Control-Max-Age", String(self.max_age))
            if self.allow_credentials:
                response = response.with_header("Access-Control-Allow-Credentials", "true")
            return response

        # Normal request - add CORS headers to response
        var response = next(ctx)
        response = response.with_header("Access-Control-Allow-Origin", origin)
        if self.allow_credentials:
            response = response.with_header("Access-Control-Allow-Credentials", "true")

        return response

# ============================================================================
# Rate Limiting Middleware
# ============================================================================

struct RateLimitEntry:
    """Tracks request count for a client."""
    var count: Int
    var window_start: Int  # Unix timestamp

    fn __init__(inout self):
        self.count = 0
        self.window_start = 0

struct RateLimitMiddleware(Middleware):
    """
    Rate limiting middleware using sliding window algorithm.
    Limits requests per client (by IP or token).
    """
    var max_requests: Int
    var window_seconds: Int
    var entries: Dict[String, RateLimitEntry]
    var by_token: Bool  # If true, rate limit by auth token instead of IP

    fn __init__(inout self, max_requests: Int = 100, window_seconds: Int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.entries = Dict[String, RateLimitEntry]()
        self.by_token = False

    fn by_auth_token(inout self):
        """Rate limit by authentication token instead of IP."""
        self.by_token = True

    fn _get_client_key(self, ctx: Context) -> String:
        """Get client identifier for rate limiting."""
        if self.by_token:
            return ctx.headers.bearer_token()
        # Fall back to X-Forwarded-For or assume localhost
        var forwarded = ctx.headers.get("X-Forwarded-For")
        if len(forwarded) > 0:
            # Take first IP in chain
            var comma_idx = forwarded.find(",")
            if comma_idx > 0:
                return forwarded[:comma_idx]
            return forwarded
        return "127.0.0.1"

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Check rate limit and proceed or reject."""
        var client_key = self._get_client_key(ctx)

        if len(client_key) == 0:
            return next(ctx)  # Can't rate limit without identifier

        # Get or create entry
        var entry: RateLimitEntry
        if client_key in self.entries:
            entry = self.entries[client_key]
        else:
            entry = RateLimitEntry()

        # Check if window expired (simplified - would use actual time)
        var current_time = 0  # Would get actual time
        if current_time - entry.window_start > self.window_seconds:
            entry.count = 0
            entry.window_start = current_time

        # Check limit
        if entry.count >= self.max_requests:
            var response = Response('{"error":"Rate limit exceeded"}', StatusCode.TOO_MANY_REQUESTS)
            response = response.with_header("X-RateLimit-Limit", String(self.max_requests))
            response = response.with_header("X-RateLimit-Remaining", "0")
            response = response.with_header("Retry-After", String(self.window_seconds))
            return response

        # Increment count
        entry.count += 1
        self.entries[client_key] = entry

        # Proceed with request
        var response = next(ctx)

        # Add rate limit headers
        var remaining = self.max_requests - entry.count
        response = response.with_header("X-RateLimit-Limit", String(self.max_requests))
        response = response.with_header("X-RateLimit-Remaining", String(remaining))

        return response

# ============================================================================
# Recovery Middleware (Error Handling)
# ============================================================================

struct RecoveryMiddleware(Middleware):
    """
    Catches panics/errors and returns 500 response.
    Prevents server crashes from unhandled errors.
    """
    var log_errors: Bool
    var include_stack: Bool

    fn __init__(inout self, log_errors: Bool = True, include_stack: Bool = False):
        self.log_errors = log_errors
        self.include_stack = include_stack

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Wrap handler with error recovery."""
        # In Mojo, we can't catch panics like in Go/Rust
        # But we can handle expected errors
        try:
            return next(ctx)
        except e:
            if self.log_errors:
                print("[ERROR] Request failed: " + ctx.path)
                print("  Error: " + String(e))

            if self.include_stack:
                return Response.internal_error("Internal server error: " + String(e))
            else:
                return Response.internal_error()

# ============================================================================
# Request ID Middleware
# ============================================================================

struct RequestIdMiddleware(Middleware):
    """
    Adds unique request ID to each request.
    Useful for tracing and debugging.
    """
    var header_name: String
    var counter: Int

    fn __init__(inout self, header_name: String = "X-Request-Id"):
        self.header_name = header_name
        self.counter = 0

    fn _generate_id(inout self) -> String:
        """Generate unique request ID."""
        self.counter += 1
        # Simple counter-based ID (would use UUID in production)
        return "req-" + String(self.counter)

    fn process(inout self, ctx: Context, next: HandlerFn) -> Response:
        """Add request ID to context and response."""
        # Check if request already has ID
        var request_id = ctx.headers.get(self.header_name)

        if len(request_id) == 0:
            request_id = self._generate_id()

        # Add to context state
        var ctx_with_id = ctx
        ctx_with_id.set_state("request_id", request_id)

        # Call handler
        var response = next(ctx_with_id)

        # Add to response headers
        return response.with_header(self.header_name, request_id)

# ============================================================================
# Compression Middleware (Stub)
# ============================================================================

struct CompressionMiddleware(Middleware):
    """
    Compresses response body using gzip.
    Only compresses if client accepts gzip and body is large enough.
    """
    var min_size: Int
    var content_types: List[String]

    fn __init__(inout self, min_size: Int = 1024):
        self.min_size = min_size
        self.content_types = List[String]()
        self.content_types.append("application/json")
        self.content_types.append("text/html")
        self.content_types.append("text/plain")
        self.content_types.append("text/css")
        self.content_types.append("application/javascript")

    fn _should_compress(self, ctx: Context, response: Response) -> Bool:
        """Check if response should be compressed."""
        # Check Accept-Encoding
        var accept_encoding = ctx.headers.get("Accept-Encoding")
        if accept_encoding.find("gzip") < 0:
            return False

        # Check body size
        if len(response.body) < self.min_size:
            return False

        # Check content type
        var content_type = response.headers.content_type()
        for i in range(len(self.content_types)):
            if content_type.find(self.content_types[i]) >= 0:
                return True

        return False

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Compress response if appropriate."""
        var response = next(ctx)

        if self._should_compress(ctx, response):
            # Would compress body here
            # For now, just add header indicating compression would happen
            response = response.with_header("X-Compression", "would-compress")

        return response

# ============================================================================
# Timeout Middleware (Stub)
# ============================================================================

struct TimeoutMiddleware(Middleware):
    """
    Enforces request timeout.
    Returns 504 Gateway Timeout if handler takes too long.
    """
    var timeout_ms: Int

    fn __init__(inout self, timeout_ms: Int = 30000):
        self.timeout_ms = timeout_ms

    fn process(self, ctx: Context, next: HandlerFn) -> Response:
        """Execute handler with timeout."""
        # In real implementation, would use async/channels for timeout
        # For now, just proceed normally
        return next(ctx)

# ============================================================================
# Middleware Chain Builder
# ============================================================================

struct MiddlewareChain:
    """
    Fluent builder for creating middleware chains.
    """
    var middlewares: List[Middleware]

    fn __init__(inout self):
        self.middlewares = List[Middleware]()

    fn use(inout self, middleware: Middleware) -> MiddlewareChain:
        """Add middleware to chain."""
        self.middlewares.append(middleware)
        return self

    fn logging(inout self, log_body: Bool = False) -> MiddlewareChain:
        """Add logging middleware."""
        return self.use(LoggingMiddleware(log_body))

    fn cors(inout self) -> MiddlewareChain:
        """Add CORS middleware with defaults."""
        var cors = CorsMiddleware()
        cors.allow_all_origins()
        return self.use(cors)

    fn recovery(inout self) -> MiddlewareChain:
        """Add recovery middleware."""
        return self.use(RecoveryMiddleware())

    fn request_id(inout self) -> MiddlewareChain:
        """Add request ID middleware."""
        return self.use(RequestIdMiddleware())

    fn rate_limit(inout self, max_requests: Int = 100, window_seconds: Int = 60) -> MiddlewareChain:
        """Add rate limiting middleware."""
        return self.use(RateLimitMiddleware(max_requests, window_seconds))

    fn build(self) -> List[Middleware]:
        """Get the middleware list."""
        return self.middlewares
