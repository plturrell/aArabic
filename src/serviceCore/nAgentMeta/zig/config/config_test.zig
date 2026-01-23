// Configuration System Tests

const std = @import("std");
const testing = std.testing;
const config = @import("config.zig");

test "Config: parse minimal valid configuration" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "sqlite",
        \\    "connection": ":memory:"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    try testing.expectEqual(config.DatabaseType.sqlite, cfg.database.type);
    try testing.expectEqualStrings(":memory:", cfg.database.connection);
    try testing.expectEqualStrings("my-super-secret-key-that-is-at-least-32-chars", cfg.auth.jwt_secret);

    // Check defaults
    try testing.expectEqual(@as(u16, 8080), cfg.server.port);
    try testing.expectEqual(@as(u32, 4), cfg.server.workers);
    try testing.expectEqual(config.LogLevel.info, cfg.logging.level);
}

test "Config: parse full configuration" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "server": {
        \\    "host": "127.0.0.1",
        \\    "port": 3000,
        \\    "workers": 8,
        \\    "read_timeout_ms": 60000,
        \\    "write_timeout_ms": 60000
        \\  },
        \\  "database": {
        \\    "type": "postgres",
        \\    "connection": "postgresql://user:pass@localhost:5432/db",
        \\    "pool": {
        \\      "min_size": 10,
        \\      "max_size": 50,
        \\      "acquire_timeout_ms": 10000
        \\    }
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars",
        \\    "token_expiry_minutes": 120,
        \\    "refresh_token_expiry_days": 60
        \\  },
        \\  "logging": {
        \\    "level": "debug",
        \\    "format": "text",
        \\    "output": "stdout"
        \\  },
        \\  "metrics": {
        \\    "enabled": true,
        \\    "port": 9091,
        \\    "path": "/prometheus"
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    // Server config
    try testing.expectEqualStrings("127.0.0.1", cfg.server.host);
    try testing.expectEqual(@as(u16, 3000), cfg.server.port);
    try testing.expectEqual(@as(u32, 8), cfg.server.workers);

    // Database config
    try testing.expectEqual(config.DatabaseType.postgres, cfg.database.type);
    try testing.expectEqual(@as(u32, 10), cfg.database.pool.min_size);
    try testing.expectEqual(@as(u32, 50), cfg.database.pool.max_size);

    // Auth config
    try testing.expectEqual(@as(u32, 120), cfg.auth.token_expiry_minutes);
    try testing.expectEqual(@as(u32, 60), cfg.auth.refresh_token_expiry_days);

    // Logging config
    try testing.expectEqual(config.LogLevel.debug, cfg.logging.level);
    try testing.expectEqual(config.LogFormat.text, cfg.logging.format);

    // Metrics config
    try testing.expect(cfg.metrics.enabled);
    try testing.expectEqual(@as(u16, 9091), cfg.metrics.port);
}

test "Config: environment variable substitution" {
    const allocator = testing.allocator;

    // Set test environment variables
    try std.process.putEnv("TEST_DB_PASSWORD", "secret123");
    try std.process.putEnv("TEST_JWT_SECRET", "my-jwt-secret-key-at-least-32-chars");

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "postgres",
        \\    "connection": "postgresql://user:${TEST_DB_PASSWORD}@localhost:5432/db"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "${TEST_JWT_SECRET}"
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    try testing.expect(std.mem.indexOf(u8, cfg.database.connection, "secret123") != null);
    try testing.expectEqualStrings("my-jwt-secret-key-at-least-32-chars", cfg.auth.jwt_secret);
}

test "Config: environment variable with default value" {
    const allocator = testing.allocator;

    // Don't set MISSING_VAR - it should use default
    const json_config =
        \\{
        \\  "database": {
        \\    "type": "sqlite",
        \\    "connection": "${MISSING_VAR:-:memory:}"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "${ALSO_MISSING:-default-secret-key-at-least-32-chars}"
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    try testing.expectEqualStrings(":memory:", cfg.database.connection);
    try testing.expectEqualStrings("default-secret-key-at-least-32-chars", cfg.auth.jwt_secret);
}

test "Config: validation catches invalid port" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "server": {
        \\    "port": 0
        \\  },
        \\  "database": {
        \\    "type": "sqlite",
        \\    "connection": ":memory:"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    const result = config.loadFromString(allocator, json_config);
    try testing.expectError(error.ValidationFailed, result);
}

test "Config: validation catches invalid pool config" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "postgres",
        \\    "connection": "postgresql://localhost/db",
        \\    "pool": {
        \\      "min_size": 50,
        \\      "max_size": 10
        \\    }
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    const result = config.loadFromString(allocator, json_config);
    try testing.expectError(error.ValidationFailed, result);
}

test "Config: missing required database field" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    const result = config.loadFromString(allocator, json_config);
    try testing.expectError(error.MissingRequiredField, result);
}

test "Config: missing required auth field" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "sqlite",
        \\    "connection": ":memory:"
        \\  }
        \\}
    ;

    const result = config.loadFromString(allocator, json_config);
    try testing.expectError(error.MissingRequiredField, result);
}

test "Config: invalid JSON" {
    const allocator = testing.allocator;

    const json_config = "{ invalid json }";

    const result = config.loadFromString(allocator, json_config);
    try testing.expectError(error.InvalidJson, result);
}

test "Config: invalid database type" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "mongodb",
        \\    "connection": "mongodb://localhost"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    const result = config.loadFromString(allocator, json_config);
    try testing.expectError(error.InvalidValue, result);
}

test "Config: invalid log level" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "sqlite",
        \\    "connection": ":memory:"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  },
        \\  "logging": {
        \\    "level": "trace"
        \\  }
        \\}
    ;

    const result = config.loadFromString(allocator, json_config);
    try testing.expectError(error.InvalidValue, result);
}

test "Config: DatabaseType.fromString" {
    try testing.expectEqual(config.DatabaseType.postgres, try config.DatabaseType.fromString("postgres"));
    try testing.expectEqual(config.DatabaseType.hana, try config.DatabaseType.fromString("hana"));
    try testing.expectEqual(config.DatabaseType.sqlite, try config.DatabaseType.fromString("sqlite"));
    try testing.expectError(error.InvalidValue, config.DatabaseType.fromString("mysql"));
}

test "Config: LogLevel.fromString" {
    try testing.expectEqual(config.LogLevel.debug, try config.LogLevel.fromString("debug"));
    try testing.expectEqual(config.LogLevel.info, try config.LogLevel.fromString("info"));
    try testing.expectEqual(config.LogLevel.warn, try config.LogLevel.fromString("warn"));
    try testing.expectEqual(config.LogLevel.@"error", try config.LogLevel.fromString("error"));
    try testing.expectError(error.InvalidValue, config.LogLevel.fromString("trace"));
}

test "Config: LogFormat.fromString" {
    try testing.expectEqual(config.LogFormat.json, try config.LogFormat.fromString("json"));
    try testing.expectEqual(config.LogFormat.text, try config.LogFormat.fromString("text"));
    try testing.expectError(error.InvalidValue, config.LogFormat.fromString("xml"));
}

test "Config: createDefault" {
    const allocator = testing.allocator;

    var cfg = try config.createDefault(allocator);
    defer cfg.deinit();

    try testing.expectEqual(config.DatabaseType.sqlite, cfg.database.type);
    try testing.expectEqualStrings(":memory:", cfg.database.connection);
    try testing.expect(cfg.auth.jwt_secret.len >= 32);
}

test "Config: multiple environment variables in same string" {
    const allocator = testing.allocator;

    try std.process.putEnv("TEST_HOST", "localhost");
    try std.process.putEnv("TEST_PORT", "5432");
    try std.process.putEnv("TEST_DB", "mydb");

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "postgres",
        \\    "connection": "postgresql://user:pass@${TEST_HOST}:${TEST_PORT}/${TEST_DB}"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    try testing.expect(std.mem.indexOf(u8, cfg.database.connection, "localhost") != null);
    try testing.expect(std.mem.indexOf(u8, cfg.database.connection, "5432") != null);
    try testing.expect(std.mem.indexOf(u8, cfg.database.connection, "mydb") != null);
}

test "Config: HANA database type" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "hana",
        \\    "connection": "hana://user:pass@hana-host:30015/SYSTEMDB",
        \\    "pool": {
        \\      "min_size": 20,
        \\      "max_size": 100
        \\    }
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    try testing.expectEqual(config.DatabaseType.hana, cfg.database.type);
    try testing.expectEqual(@as(u32, 20), cfg.database.pool.min_size);
    try testing.expectEqual(@as(u32, 100), cfg.database.pool.max_size);
}

test "Config: all pool configuration options" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "postgres",
        \\    "connection": "postgresql://localhost/db",
        \\    "pool": {
        \\      "min_size": 5,
        \\      "max_size": 25,
        \\      "acquire_timeout_ms": 3000,
        \\      "idle_timeout_ms": 600000,
        \\      "max_lifetime_ms": 3600000,
        \\      "health_check_interval_ms": 60000
        \\    }
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    try testing.expectEqual(@as(u32, 5), cfg.database.pool.min_size);
    try testing.expectEqual(@as(u32, 25), cfg.database.pool.max_size);
    try testing.expectEqual(@as(u64, 3000), cfg.database.pool.acquire_timeout_ms);
    try testing.expectEqual(@as(u64, 600000), cfg.database.pool.idle_timeout_ms);
    try testing.expectEqual(@as(u64, 3600000), cfg.database.pool.max_lifetime_ms);
    try testing.expectEqual(@as(u64, 60000), cfg.database.pool.health_check_interval_ms);
}

test "Config: CORS configuration" {
    const allocator = testing.allocator;

    const json_config =
        \\{
        \\  "database": {
        \\    "type": "sqlite",
        \\    "connection": ":memory:"
        \\  },
        \\  "auth": {
        \\    "jwt_secret": "my-super-secret-key-that-is-at-least-32-chars"
        \\  },
        \\  "cors": {
        \\    "enabled": false
        \\  }
        \\}
    ;

    var cfg = try config.loadFromString(allocator, json_config);
    defer cfg.deinit();

    try testing.expect(!cfg.cors.enabled);
}
