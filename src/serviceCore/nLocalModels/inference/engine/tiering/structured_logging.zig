// Structured Logging System for Production LLM Server
// Day 6: JSON structured logging with levels, rotation, and aggregation support
//
// Features:
// - JSON structured output for log aggregation (Loki/ELK)
// - Multiple log levels (DEBUG/INFO/WARN/ERROR/FATAL)
// - Performance-optimized buffering
// - Thread-safe logging
// - Automatic log rotation support
// - Context propagation (request ID, trace ID)

const std = @import("std");
const builtin = @import("builtin");

/// Log levels following industry standards
pub const LogLevel = enum(u8) {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4,
    
    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .DEBUG => "DEBUG",
            .INFO => "INFO",
            .WARN => "WARN",
            .ERROR => "ERROR",
            .FATAL => "FATAL",
        };
    }
    
    pub fn fromString(s: []const u8) ?LogLevel {
        if (std.mem.eql(u8, s, "DEBUG")) return .DEBUG;
        if (std.mem.eql(u8, s, "INFO")) return .INFO;
        if (std.mem.eql(u8, s, "WARN")) return .WARN;
        if (std.mem.eql(u8, s, "ERROR")) return .ERROR;
        if (std.mem.eql(u8, s, "FATAL")) return .FATAL;
        return null;
    }
};

/// Logger configuration
pub const LoggerConfig = struct {
    /// Minimum log level to output
    min_level: LogLevel = .INFO,
    
    /// Enable JSON structured output
    json_format: bool = true,
    
    /// Buffer size for async logging (0 = synchronous)
    buffer_size: usize = 4096,
    
    /// Log file path (null = stdout only)
    file_path: ?[]const u8 = null,
    
    /// Enable log rotation
    enable_rotation: bool = true,
    
    /// Max log file size before rotation (bytes)
    max_file_size: u64 = 100 * 1024 * 1024, // 100MB
    
    /// Max number of rotated log files to keep
    max_backup_files: u32 = 10,
    
    /// Service name for logging context
    service_name: []const u8 = "llm-server",
    
    /// Environment (dev/staging/prod)
    environment: []const u8 = "dev",
};

/// Log context for request tracing
pub const LogContext = struct {
    request_id: ?[]const u8 = null,
    trace_id: ?[]const u8 = null,
    span_id: ?[]const u8 = null,
    user_id: ?[]const u8 = null,
    model_name: ?[]const u8 = null,
    operation: ?[]const u8 = null,
};

/// Structured log entry
pub const LogEntry = struct {
    timestamp: i64,
    level: LogLevel,
    message: []const u8,
    context: LogContext,
    fields: std.StringHashMap([]const u8),
    
    /// Format as JSON for structured logging
    pub fn toJson(self: LogEntry, allocator: std.mem.Allocator) ![]u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit();
        
        const writer = buffer.writer();
        
        try writer.writeAll("{");
        
        // Timestamp (ISO 8601)
        try writer.print("\"timestamp\":\"{d}\"", .{self.timestamp});
        
        // Level
        try writer.print(",\"level\":\"{}\"", .{self.level.toString()});
        
        // Message
        try writer.print(",\"message\":\"{}\"", .{self.message});
        
        // Context fields
        if (self.context.request_id) |req_id| {
            try writer.print(",\"request_id\":\"{}\"", .{req_id});
        }
        if (self.context.trace_id) |trace_id| {
            try writer.print(",\"trace_id\":\"{}\"", .{trace_id});
        }
        if (self.context.span_id) |span_id| {
            try writer.print(",\"span_id\":\"{}\"", .{span_id});
        }
        if (self.context.user_id) |user_id| {
            try writer.print(",\"user_id\":\"{}\"", .{user_id});
        }
        if (self.context.model_name) |model| {
            try writer.print(",\"model\":\"{}\"", .{model});
        }
        if (self.context.operation) |op| {
            try writer.print(",\"operation\":\"{}\"", .{op});
        }
        
        // Additional fields
        var it = self.fields.iterator();
        while (it.next()) |entry| {
            try writer.print(",\"{}\":\"{}\"", .{ entry.key_ptr.*, entry.value_ptr.* });
        }
        
        try writer.writeAll("}");
        
        return buffer.toOwnedSlice();
    }
};

/// Main logger instance
pub const Logger = struct {
    allocator: std.mem.Allocator,
    config: LoggerConfig,
    mutex: std.Thread.Mutex,
    file: ?std.fs.File,
    current_file_size: std.atomic.Value(u64),
    context: LogContext,
    
    pub fn init(allocator: std.mem.Allocator, config: LoggerConfig) !*Logger {
        const self = try allocator.create(Logger);
        errdefer allocator.destroy(self);
        
        var file: ?std.fs.File = null;
        if (config.file_path) |path| {
            file = try std.fs.cwd().createFile(path, .{
                .read = false,
                .truncate = false,
            });
        }
        
        self.* = Logger{
            .allocator = allocator,
            .config = config,
            .mutex = .{},
            .file = file,
            .current_file_size = std.atomic.Value(u64).init(0),
            .context = .{},
        };
        
        return self;
    }
    
    pub fn deinit(self: *Logger) void {
        if (self.file) |file| {
            file.close();
        }
        self.allocator.destroy(self);
    }
    
    /// Set logging context for request tracing
    pub fn setContext(self: *Logger, context: LogContext) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.context = context;
    }
    
    /// Clear logging context
    pub fn clearContext(self: *Logger) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.context = .{};
    }
    
    /// Log a message with specified level
    pub fn log(
        self: *Logger,
        level: LogLevel,
        comptime fmt: []const u8,
        args: anytype,
    ) void {
        // Check if level should be logged
        if (@intFromEnum(level) < @intFromEnum(self.config.min_level)) {
            return;
        }
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Format message
        var msg_buffer: [2048]u8 = undefined;
        const message = std.fmt.bufPrint(&msg_buffer, fmt, args) catch {
            // Fallback if message too long
            const truncated = "...[truncated]";
            const max_len = msg_buffer.len - truncated.len;
            @memcpy(msg_buffer[max_len..][0..truncated.len], truncated);
            msg_buffer[0..msg_buffer.len].*;
        };
        
        // Create log entry
        var entry = LogEntry{
            .timestamp = std.time.milliTimestamp(),
            .level = level,
            .message = message,
            .context = self.context,
            .fields = std.StringHashMap([]const u8).init(self.allocator),
        };
        defer entry.fields.deinit();
        
        // Add service metadata
        entry.fields.put("service", self.config.service_name) catch {};
        entry.fields.put("environment", self.config.environment) catch {};
        
        // Write log entry
        self.writeEntry(entry) catch |write_err| {
            // Fallback to stderr if logging fails
            std.debug.print("Logging error: {}\n", .{write_err});
        };
    }
    
    /// Log with additional structured fields
    pub fn logFields(
        self: *Logger,
        level: LogLevel,
        comptime fmt: []const u8,
        args: anytype,
        fields: std.StringHashMap([]const u8),
    ) void {
        if (@intFromEnum(level) < @intFromEnum(self.config.min_level)) {
            return;
        }
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var msg_buffer: [2048]u8 = undefined;
        const message = std.fmt.bufPrint(&msg_buffer, fmt, args) catch msg_buffer[0..];
        
        const entry = LogEntry{
            .timestamp = std.time.milliTimestamp(),
            .level = level,
            .message = message,
            .context = self.context,
            .fields = fields,
        };
        
        self.writeEntry(entry) catch |write_err| {
            std.debug.print("Logging error: {}\n", .{write_err});
        };
    }
    
    /// Write entry to outputs
    fn writeEntry(self: *Logger, entry: LogEntry) !void {
        // Check if rotation needed
        if (self.config.enable_rotation and self.file != null) {
            const current_size = self.current_file_size.load(.monotonic);
            if (current_size >= self.config.max_file_size) {
                try self.rotateLog();
            }
        }
        
        if (self.config.json_format) {
            // JSON structured output
            const json = try entry.toJson(self.allocator);
            defer self.allocator.free(json);
            
            // Write to stdout
            try std.io.getStdOut().writer().print("{s}\n", .{json});
            
            // Write to file if configured
            if (self.file) |file| {
                try file.writer().print("{s}\n", .{json});
                _ = self.current_file_size.fetchAdd(json.len + 1, .monotonic);
            }
        } else {
            // Human-readable format
            const timestamp = entry.timestamp;
            const level_str = entry.level.toString();
            
            // Write to stdout
            try std.io.getStdOut().writer().print(
                "[{d}] {s}: {s}\n",
                .{ timestamp, level_str, entry.message }
            );
            
            // Write to file if configured
            if (self.file) |file| {
                const line = try std.fmt.allocPrint(
                    self.allocator,
                    "[{d}] {s}: {s}\n",
                    .{ timestamp, level_str, entry.message }
                );
                defer self.allocator.free(line);
                
                try file.writer().writeAll(line);
                _ = self.current_file_size.fetchAdd(line.len, .monotonic);
            }
        }
    }
    
    /// Rotate log files
    fn rotateLog(self: *Logger) !void {
        if (self.file == null or self.config.file_path == null) return;
        
        const base_path = self.config.file_path.?;
        
        // Close current file
        self.file.?.close();
        
        // Rotate existing backups
        var i: u32 = self.config.max_backup_files - 1;
        while (i > 0) : (i -= 1) {
            const old_name = try std.fmt.allocPrint(
                self.allocator,
                "{s}.{d}",
                .{ base_path, i }
            );
            defer self.allocator.free(old_name);
            
            const new_name = try std.fmt.allocPrint(
                self.allocator,
                "{s}.{d}",
                .{ base_path, i + 1 }
            );
            defer self.allocator.free(new_name);
            
            std.fs.cwd().rename(old_name, new_name) catch {};
        }
        
        // Rename current to .1
        const backup_name = try std.fmt.allocPrint(
            self.allocator,
            "{s}.1",
            .{base_path}
        );
        defer self.allocator.free(backup_name);
        
        try std.fs.cwd().rename(base_path, backup_name);
        
        // Create new file
        self.file = try std.fs.cwd().createFile(base_path, .{
            .read = false,
            .truncate = true,
        });
        
        self.current_file_size.store(0, .monotonic);
    }
    
    // Convenience methods for each log level
    
    pub fn debug(self: *Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.DEBUG, fmt, args);
    }
    
    pub fn info(self: *Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.INFO, fmt, args);
    }
    
    pub fn warn(self: *Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.WARN, fmt, args);
    }
    
    pub fn err(self: *Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.ERROR, fmt, args);
    }
    
    pub fn fatal(self: *Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.FATAL, fmt, args);
    }
};

/// Global logger instance (singleton pattern)
var global_logger: ?*Logger = null;
var global_logger_mutex: std.Thread.Mutex = .{};

/// Initialize global logger
pub fn initGlobalLogger(allocator: std.mem.Allocator, config: LoggerConfig) !void {
    global_logger_mutex.lock();
    defer global_logger_mutex.unlock();
    
    if (global_logger != null) {
        return error.LoggerAlreadyInitialized;
    }
    
    global_logger = try Logger.init(allocator, config);
}

/// Deinitialize global logger
pub fn deinitGlobalLogger() void {
    global_logger_mutex.lock();
    defer global_logger_mutex.unlock();
    
    if (global_logger) |logger| {
        logger.deinit();
        global_logger = null;
    }
}

/// Get global logger instance
pub fn getGlobalLogger() ?*Logger {
    global_logger_mutex.lock();
    defer global_logger_mutex.unlock();
    return global_logger;
}

// ============================================================================
// Convenience Functions for Global Logger
// ============================================================================

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (getGlobalLogger()) |logger| {
        logger.debug(fmt, args);
    }
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    if (getGlobalLogger()) |logger| {
        logger.info(fmt, args);
    }
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    if (getGlobalLogger()) |logger| {
        logger.warn(fmt, args);
    }
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    if (getGlobalLogger()) |logger| {
        logger.err(fmt, args);
    }
}

pub fn fatal(comptime fmt: []const u8, args: anytype) void {
    if (getGlobalLogger()) |logger| {
        logger.fatal(fmt, args);
    }
}

pub fn setContext(context: LogContext) void {
    if (getGlobalLogger()) |logger| {
        logger.setContext(context);
    }
}

pub fn clearContext() void {
    if (getGlobalLogger()) |logger| {
        logger.clearContext();
    }
}
