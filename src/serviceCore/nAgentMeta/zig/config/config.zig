// nMetaData Configuration System
// Provides JSON-based configuration with environment variable support

const std = @import("std");
const json = std.json;

/// Configuration error types
pub const ConfigError = error{
    InvalidJson,
    MissingRequiredField,
    InvalidValue,
    EnvironmentVariableNotFound,
    FileNotFound,
    ValidationFailed,
};

/// Server configuration
pub const ServerConfig = struct {
    host: []const u8 = "0.0.0.0",
    port: u16 = 8080,
    workers: u32 = 4,
    read_timeout_ms: u64 = 30000,
    write_timeout_ms: u64 = 30000,
    max_request_size_bytes: u64 = 10 * 1024 * 1024, // 10MB
};

/// Database type enum
pub const DatabaseType = enum {
    postgres,
    hana,
    sqlite,

    pub fn fromString(s: []const u8) !DatabaseType {
        if (std.mem.eql(u8, s, "postgres")) return .postgres;
        if (std.mem.eql(u8, s, "hana")) return .hana;
        if (std.mem.eql(u8, s, "sqlite")) return .sqlite;
        return error.InvalidValue;
    }
};

/// Connection pool configuration
pub const PoolConfig = struct {
    min_size: u32 = 5,
    max_size: u32 = 20,
    acquire_timeout_ms: u64 = 5000,
    idle_timeout_ms: u64 = 300000, // 5 minutes
    max_lifetime_ms: u64 = 1800000, // 30 minutes
    health_check_interval_ms: u64 = 30000, // 30 seconds
};

/// Database configuration
pub const DatabaseConfig = struct {
    type: DatabaseType,
    connection: []const u8,
    pool: PoolConfig = .{},
};

/// Authentication configuration
pub const AuthConfig = struct {
    jwt_secret: []const u8,
    token_expiry_minutes: u32 = 60,
    refresh_token_expiry_days: u32 = 30,
    bcrypt_rounds: u32 = 12,
};

/// Log level enum
pub const LogLevel = enum {
    debug,
    info,
    warn,
    @"error",

    pub fn fromString(s: []const u8) !LogLevel {
        if (std.mem.eql(u8, s, "debug")) return .debug;
        if (std.mem.eql(u8, s, "info")) return .info;
        if (std.mem.eql(u8, s, "warn")) return .warn;
        if (std.mem.eql(u8, s, "error")) return .@"error";
        return error.InvalidValue;
    }
};

/// Log format enum
pub const LogFormat = enum {
    json,
    text,

    pub fn fromString(s: []const u8) !LogFormat {
        if (std.mem.eql(u8, s, "json")) return .json;
        if (std.mem.eql(u8, s, "text")) return .text;
        return error.InvalidValue;
    }
};

/// Logging configuration
pub const LoggingConfig = struct {
    level: LogLevel = .info,
    format: LogFormat = .json,
    output: []const u8 = "stdout",
};

/// Metrics configuration
pub const MetricsConfig = struct {
    enabled: bool = true,
    port: u16 = 9090,
    path: []const u8 = "/metrics",
};

/// CORS configuration
pub const CorsConfig = struct {
    enabled: bool = true,
    allow_origins: []const []const u8 = &[_][]const u8{"*"},
    allow_methods: []const []const u8 = &[_][]const u8{ "GET", "POST", "PUT", "DELETE", "OPTIONS" },
    allow_headers: []const []const u8 = &[_][]const u8{ "Content-Type", "Authorization" },
    max_age_seconds: u32 = 86400,
};

/// Complete application configuration
pub const Config = struct {
    server: ServerConfig = .{},
    database: DatabaseConfig,
    auth: AuthConfig,
    logging: LoggingConfig = .{},
    metrics: MetricsConfig = .{},
    cors: CorsConfig = .{},

    allocator: std.mem.Allocator,

    /// Free allocated memory
    pub fn deinit(self: *Config) void {
        // Free any allocated strings
        self.allocator.free(self.database.connection);
        self.allocator.free(self.auth.jwt_secret);
        if (!std.mem.eql(u8, self.logging.output, "stdout")) {
            self.allocator.free(self.logging.output);
        }
    }

    /// Validate configuration
    pub fn validate(self: *const Config) !void {
        // Validate server config
        if (self.server.port == 0) {
            return error.ValidationFailed;
        }
        if (self.server.workers == 0) {
            return error.ValidationFailed;
        }

        // Validate database config
        if (self.database.connection.len == 0) {
            return error.ValidationFailed;
        }
        if (self.database.pool.min_size > self.database.pool.max_size) {
            return error.ValidationFailed;
        }

        // Validate auth config
        if (self.auth.jwt_secret.len < 32) {
            std.log.warn("JWT secret is less than 32 characters, consider using a stronger secret", .{});
        }
        if (self.auth.token_expiry_minutes == 0) {
            return error.ValidationFailed;
        }
    }
};

/// Environment variable resolver
const EnvResolver = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) EnvResolver {
        return .{ .allocator = allocator };
    }

    /// Resolve environment variables in a string
    /// Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax
    pub fn resolve(self: *EnvResolver, input: []const u8) ![]const u8 {
        if (std.mem.indexOf(u8, input, "${") == null) {
            // No env vars, return copy
            return try self.allocator.dupe(u8, input);
        }

        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        var i: usize = 0;
        while (i < input.len) {
            if (i + 1 < input.len and input[i] == '$' and input[i + 1] == '{') {
                // Find closing brace
                const start = i + 2;
                var end = start;
                while (end < input.len and input[end] != '}') : (end += 1) {}

                if (end >= input.len) {
                    return error.InvalidValue;
                }

                const var_expr = input[start..end];

                // Check for default value syntax: VAR_NAME:-default
                var var_name: []const u8 = undefined;
                var default_value: ?[]const u8 = null;

                if (std.mem.indexOf(u8, var_expr, ":-")) |sep_pos| {
                    var_name = var_expr[0..sep_pos];
                    default_value = var_expr[sep_pos + 2 ..];
                } else {
                    var_name = var_expr;
                }

                // Get environment variable
                const value = std.process.getEnvVarOwned(self.allocator, var_name) catch |err| {
                    if (err == error.EnvironmentVariableNotFound) {
                        if (default_value) |default| {
                            try result.appendSlice(default);
                            i = end + 1;
                            continue;
                        }
                    }
                    return error.EnvironmentVariableNotFound;
                };
                defer self.allocator.free(value);

                try result.appendSlice(value);
                i = end + 1;
            } else {
                try result.append(input[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice();
    }
};

/// Load configuration from JSON file
pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Config {
    // Read file
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.log.err("Failed to open config file '{s}': {}", .{ path, err });
        return error.FileNotFound;
    };
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 1024 * 1024); // Max 1MB config
    defer allocator.free(content);

    return try loadFromString(allocator, content);
}

/// Load configuration from JSON string
pub fn loadFromString(allocator: std.mem.Allocator, content: []const u8) !Config {
    var resolver = EnvResolver.init(allocator);

    // Parse JSON
    const parsed = json.parseFromSlice(json.Value, allocator, content, .{}) catch |err| {
        std.log.err("Failed to parse JSON config: {}", .{err});
        return error.InvalidJson;
    };
    defer parsed.deinit();

    const root = parsed.value.object;

    // Parse server config
    var server_config = ServerConfig{};
    if (root.get("server")) |server_obj| {
        const server = server_obj.object;
        if (server.get("host")) |v| {
            server_config.host = try allocator.dupe(u8, v.string);
        }
        if (server.get("port")) |v| {
            server_config.port = @intCast(v.integer);
        }
        if (server.get("workers")) |v| {
            server_config.workers = @intCast(v.integer);
        }
        if (server.get("read_timeout_ms")) |v| {
            server_config.read_timeout_ms = @intCast(v.integer);
        }
        if (server.get("write_timeout_ms")) |v| {
            server_config.write_timeout_ms = @intCast(v.integer);
        }
        if (server.get("max_request_size_bytes")) |v| {
            server_config.max_request_size_bytes = @intCast(v.integer);
        }
    }

    // Parse database config
    const db_obj = root.get("database") orelse return error.MissingRequiredField;
    const db = db_obj.object;

    const db_type_str = db.get("type") orelse return error.MissingRequiredField;
    const db_type = try DatabaseType.fromString(db_type_str.string);

    const db_conn_str = db.get("connection") orelse return error.MissingRequiredField;
    const db_connection = try resolver.resolve(db_conn_str.string);

    var pool_config = PoolConfig{};
    if (db.get("pool")) |pool_obj| {
        const pool = pool_obj.object;
        if (pool.get("min_size")) |v| pool_config.min_size = @intCast(v.integer);
        if (pool.get("max_size")) |v| pool_config.max_size = @intCast(v.integer);
        if (pool.get("acquire_timeout_ms")) |v| pool_config.acquire_timeout_ms = @intCast(v.integer);
        if (pool.get("idle_timeout_ms")) |v| pool_config.idle_timeout_ms = @intCast(v.integer);
        if (pool.get("max_lifetime_ms")) |v| pool_config.max_lifetime_ms = @intCast(v.integer);
        if (pool.get("health_check_interval_ms")) |v| pool_config.health_check_interval_ms = @intCast(v.integer);
    }

    const database_config = DatabaseConfig{
        .type = db_type,
        .connection = db_connection,
        .pool = pool_config,
    };

    // Parse auth config
    const auth_obj = root.get("auth") orelse return error.MissingRequiredField;
    const auth = auth_obj.object;

    const jwt_secret_str = auth.get("jwt_secret") orelse return error.MissingRequiredField;
    const jwt_secret = try resolver.resolve(jwt_secret_str.string);

    var auth_config = AuthConfig{
        .jwt_secret = jwt_secret,
    };
    if (auth.get("token_expiry_minutes")) |v| {
        auth_config.token_expiry_minutes = @intCast(v.integer);
    }
    if (auth.get("refresh_token_expiry_days")) |v| {
        auth_config.refresh_token_expiry_days = @intCast(v.integer);
    }
    if (auth.get("bcrypt_rounds")) |v| {
        auth_config.bcrypt_rounds = @intCast(v.integer);
    }

    // Parse logging config
    var logging_config = LoggingConfig{};
    if (root.get("logging")) |logging_obj| {
        const logging = logging_obj.object;
        if (logging.get("level")) |v| {
            logging_config.level = try LogLevel.fromString(v.string);
        }
        if (logging.get("format")) |v| {
            logging_config.format = try LogFormat.fromString(v.string);
        }
        if (logging.get("output")) |v| {
            logging_config.output = try allocator.dupe(u8, v.string);
        }
    }

    // Parse metrics config
    var metrics_config = MetricsConfig{};
    if (root.get("metrics")) |metrics_obj| {
        const metrics = metrics_obj.object;
        if (metrics.get("enabled")) |v| metrics_config.enabled = v.bool;
        if (metrics.get("port")) |v| metrics_config.port = @intCast(v.integer);
        if (metrics.get("path")) |v| {
            metrics_config.path = try allocator.dupe(u8, v.string);
        }
    }

    // Parse CORS config
    var cors_config = CorsConfig{};
    if (root.get("cors")) |cors_obj| {
        const cors = cors_obj.object;
        if (cors.get("enabled")) |v| cors_config.enabled = v.bool;
        // Note: For simplicity, using default arrays for origins/methods/headers
        // In production, these should be properly parsed from JSON arrays
    }

    var config = Config{
        .server = server_config,
        .database = database_config,
        .auth = auth_config,
        .logging = logging_config,
        .metrics = metrics_config,
        .cors = cors_config,
        .allocator = allocator,
    };

    // Validate configuration
    try config.validate();

    return config;
}

/// Create default configuration for testing
pub fn createDefault(allocator: std.mem.Allocator) !Config {
    const jwt_secret = try allocator.dupe(u8, "test-secret-key-min-32-characters-long");
    const db_connection = try allocator.dupe(u8, ":memory:");

    return Config{
        .database = .{
            .type = .sqlite,
            .connection = db_connection,
        },
        .auth = .{
            .jwt_secret = jwt_secret,
        },
        .allocator = allocator,
    };
}
