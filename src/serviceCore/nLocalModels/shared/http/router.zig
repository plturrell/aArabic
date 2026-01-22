// Unified HTTP Router
// Routes requests to appropriate service handlers (LLM, Embedding, Translation, RAG)
// Supports both v1 (legacy) and v2 (unified) API endpoints

const std = @import("std");

// ============================================================================
// Service Types
// ============================================================================

pub const ServiceType = enum {
    llm,
    embedding,
    translation,
    rag,
    health,
    info,
    unknown,
};

// ============================================================================
// Route Definition
// ============================================================================

pub const Route = struct {
    method: []const u8,      // "GET", "POST", etc.
    path: []const u8,        // "/v1/chat/completions"
    service: ServiceType,    // Which service handles this
    
    pub fn matches(self: Route, method: []const u8, path: []const u8) bool {
        return std.mem.eql(u8, self.method, method) and 
               std.mem.eql(u8, self.path, path);
    }
};

// ============================================================================
// Route Table
// ============================================================================

pub const routes = [_]Route{
    // ========================================================================
    // LLM Service (Port 8010)
    // ========================================================================
    
    // OpenAI v1 API (legacy)
    .{ .method = "POST", .path = "/v1/chat/completions", .service = .llm },
    .{ .method = "POST", .path = "/v1/completions", .service = .llm },
    .{ .method = "GET", .path = "/v1/models", .service = .llm },
    
    // Ollama-compatible API
    .{ .method = "GET", .path = "/api/tags", .service = .llm },
    .{ .method = "POST", .path = "/api/generate", .service = .llm },
    .{ .method = "POST", .path = "/api/chat", .service = .llm },
    
    // v2 API (unified)
    .{ .method = "POST", .path = "/api/v2/llm/chat", .service = .llm },
    .{ .method = "POST", .path = "/api/v2/llm/completion", .service = .llm },
    .{ .method = "GET", .path = "/api/v2/llm/models", .service = .llm },
    
    // ========================================================================
    // Embedding Service (Port 8007 or /api/v2/embed/*)
    // ========================================================================
    
    // v1 API (legacy)
    .{ .method = "POST", .path = "/embed/single", .service = .embedding },
    .{ .method = "POST", .path = "/embed/batch", .service = .embedding },
    .{ .method = "POST", .path = "/embed/workflow", .service = .embedding },
    .{ .method = "POST", .path = "/embed/invoice", .service = .embedding },
    .{ .method = "POST", .path = "/embed/document", .service = .embedding },
    .{ .method = "GET", .path = "/embed/models", .service = .embedding },
    .{ .method = "GET", .path = "/embed/metrics", .service = .embedding },
    
    // v2 API (unified)
    .{ .method = "POST", .path = "/api/v2/embed/single", .service = .embedding },
    .{ .method = "POST", .path = "/api/v2/embed/batch", .service = .embedding },
    .{ .method = "POST", .path = "/api/v2/embed/workflow", .service = .embedding },
    .{ .method = "POST", .path = "/api/v2/embed/invoice", .service = .embedding },
    .{ .method = "POST", .path = "/api/v2/embed/document", .service = .embedding },
    .{ .method = "GET", .path = "/api/v2/embed/models", .service = .embedding },
    
    // ========================================================================
    // Translation Service (Port 8008 or /api/v2/translate/*)
    // ========================================================================
    
    // v1 API (legacy)
    .{ .method = "POST", .path = "/translate", .service = .translation },
    .{ .method = "POST", .path = "/translate/batch", .service = .translation },
    .{ .method = "POST", .path = "/translate/quality", .service = .translation },
    .{ .method = "GET", .path = "/translate/models", .service = .translation },
    .{ .method = "GET", .path = "/translate/metrics", .service = .translation },
    
    // v2 API (unified)
    .{ .method = "POST", .path = "/api/v2/translate", .service = .translation },
    .{ .method = "POST", .path = "/api/v2/translate/batch", .service = .translation },
    .{ .method = "POST", .path = "/api/v2/translate/quality", .service = .translation },
    .{ .method = "GET", .path = "/api/v2/translate/models", .service = .translation },
    
    // ========================================================================
    // RAG Service (Port 8009 or /api/v2/rag/*)
    // ========================================================================
    
    // v1 API (legacy)
    .{ .method = "POST", .path = "/rag/search", .service = .rag },
    .{ .method = "POST", .path = "/rag/index", .service = .rag },
    .{ .method = "POST", .path = "/rag/rerank", .service = .rag },
    .{ .method = "GET", .path = "/rag/collections", .service = .rag },
    .{ .method = "GET", .path = "/rag/metrics", .service = .rag },
    
    // v2 API (unified)
    .{ .method = "POST", .path = "/api/v2/rag/search", .service = .rag },
    .{ .method = "POST", .path = "/api/v2/rag/index", .service = .rag },
    .{ .method = "POST", .path = "/api/v2/rag/rerank", .service = .rag },
    .{ .method = "GET", .path = "/api/v2/rag/collections", .service = .rag },
    
    // ========================================================================
    // Health & Info
    // ========================================================================
    
    .{ .method = "GET", .path = "/", .service = .info },
    .{ .method = "GET", .path = "/health", .service = .health },
    .{ .method = "GET", .path = "/api/v2/health", .service = .health },
    .{ .method = "GET", .path = "/api/v2/llm/health", .service = .health },
    .{ .method = "GET", .path = "/api/v2/embed/health", .service = .health },
    .{ .method = "GET", .path = "/api/v2/translate/health", .service = .health },
    .{ .method = "GET", .path = "/api/v2/rag/health", .service = .health },
};

// ============================================================================
// Route Matching
// ============================================================================

pub fn route(method: []const u8, path: []const u8) ServiceType {
    """
    Match HTTP request to service type.
    Uses linear search through route table.
    
    Performance: O(n) where n = number of routes (~50)
    For high-performance needs, could use hash map or trie.
    """
    
    // Check exact matches first (most common case)
    for (routes) |r| {
        if (r.matches(method, path)) {
            return r.service;
        }
    }
    
    // Check prefix matches for wildcard routes
    // e.g., /api/v2/*/health matches /api/v2/llm/health
    if (std.mem.startsWith(u8, path, "/api/v2/") and 
        std.mem.endsWith(u8, path, "/health")) {
        return .health;
    }
    
    // No match found
    return .unknown;
}

// ============================================================================
// Route Information
// ============================================================================

pub fn getServiceName(service: ServiceType) []const u8 {
    """Get human-readable service name."""
    return switch (service) {
        .llm => "LLM",
        .embedding => "Embedding",
        .translation => "Translation",
        .rag => "RAG",
        .health => "Health",
        .info => "Info",
        .unknown => "Unknown",
    };
}

pub fn getServicePort(service: ServiceType) u16 {
    """Get default port for service (for multi-port mode)."""
    return switch (service) {
        .llm => 8010,
        .embedding => 8007,
        .translation => 8008,
        .rag => 8009,
        .health => 8010,  // Same as LLM
        .info => 8010,    // Same as LLM
        .unknown => 0,
    };
}

// ============================================================================
// API Version Detection
// ============================================================================

pub const ApiVersion = enum {
    v1,      // Legacy endpoints (no prefix or /v1/*)
    v2,      // Unified endpoints (/api/v2/*)
    ollama,  // Ollama-compatible (/api/*)
    unknown,
};

pub fn getApiVersion(path: []const u8) ApiVersion {
    """Detect API version from path."""
    if (std.mem.startsWith(u8, path, "/api/v2/")) {
        return .v2;
    } else if (std.mem.startsWith(u8, path, "/v1/")) {
        return .v1;
    } else if (std.mem.startsWith(u8, path, "/api/")) {
        return .ollama;
    } else if (std.mem.eql(u8, path, "/") or 
               std.mem.eql(u8, path, "/health")) {
        return .v1;  // Core endpoints
    } else if (std.mem.startsWith(u8, path, "/embed/") or
               std.mem.startsWith(u8, path, "/translate/") or
               std.mem.startsWith(u8, path, "/rag/")) {
        return .v1;  // Legacy service endpoints
    }
    return .unknown;
}

// ============================================================================
// Route Statistics (for monitoring)
// ============================================================================

pub const RouteStats = struct {
    service: ServiceType,
    count: u64,
    total_time_ms: u64,
    
    pub fn init(service: ServiceType) RouteStats {
        return RouteStats{
            .service = service,
            .count = 0,
            .total_time_ms = 0,
        };
    }
    
    pub fn record(self: *RouteStats, duration_ms: u64) void {
        self.count += 1;
        self.total_time_ms += duration_ms;
    }
    
    pub fn avgTimeMs(self: RouteStats) f64 {
        if (self.count == 0) return 0.0;
        return @intToFloat(f64, self.total_time_ms) / @intToFloat(f64, self.count);
    }
};

// ============================================================================
// Response Headers
// ============================================================================

pub fn addApiVersionHeaders(
    allocator: std.mem.Allocator,
    headers: *std.StringHashMap([]const u8),
    version: ApiVersion
) !void {
    """Add API version headers to response."""
    
    const version_str = switch (version) {
        .v1 => "1.0",
        .v2 => "2.0",
        .ollama => "ollama",
        .unknown => "unknown",
    };
    
    try headers.put("X-API-Version", version_str);
    try headers.put("X-Service", "shimmy-unified");
    
    // Add deprecation warning for v1
    if (version == .v1) {
        try headers.put("X-Deprecated", "true");
        try headers.put("X-Sunset", "2026-04-01");
        try headers.put("Link", "</api/v2/*>; rel=\"successor-version\"");
    }
}

// ============================================================================
// Route Testing
// ============================================================================

test "route LLM endpoints" {
    const testing = std.testing;
    
    try testing.expectEqual(ServiceType.llm, route("POST", "/v1/chat/completions"));
    try testing.expectEqual(ServiceType.llm, route("POST", "/v1/completions"));
    try testing.expectEqual(ServiceType.llm, route("GET", "/v1/models"));
    try testing.expectEqual(ServiceType.llm, route("POST", "/api/v2/llm/chat"));
}

test "route Embedding endpoints" {
    const testing = std.testing;
    
    try testing.expectEqual(ServiceType.embedding, route("POST", "/embed/single"));
    try testing.expectEqual(ServiceType.embedding, route("POST", "/api/v2/embed/batch"));
}

test "route Translation endpoints" {
    const testing = std.testing;
    
    try testing.expectEqual(ServiceType.translation, route("POST", "/translate"));
    try testing.expectEqual(ServiceType.translation, route("POST", "/api/v2/translate"));
}

test "route RAG endpoints" {
    const testing = std.testing;
    
    try testing.expectEqual(ServiceType.rag, route("POST", "/rag/search"));
    try testing.expectEqual(ServiceType.rag, route("POST", "/api/v2/rag/search"));
}

test "route health endpoints" {
    const testing = std.testing;
    
    try testing.expectEqual(ServiceType.health, route("GET", "/health"));
    try testing.expectEqual(ServiceType.health, route("GET", "/api/v2/health"));
}

test "detect API versions" {
    const testing = std.testing;
    
    try testing.expectEqual(ApiVersion.v1, getApiVersion("/v1/chat/completions"));
    try testing.expectEqual(ApiVersion.v2, getApiVersion("/api/v2/llm/chat"));
    try testing.expectEqual(ApiVersion.ollama, getApiVersion("/api/tags"));
    try testing.expectEqual(ApiVersion.v1, getApiVersion("/health"));
}

test "unknown route" {
    const testing = std.testing;
    
    try testing.expectEqual(ServiceType.unknown, route("GET", "/nonexistent"));
    try testing.expectEqual(ServiceType.unknown, route("POST", "/api/v3/future"));
}
