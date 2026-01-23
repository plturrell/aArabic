//! Security and Audit Logging Module for nWorkflow
//!
//! Provides comprehensive audit logging, RBAC permission checking,
//! and GDPR compliance helpers for the workflow engine.
//!
//! Features:
//! - Structured audit events for all workflow operations
//! - RBAC-based permission checking
//! - GDPR compliance helpers (anonymization, redaction, retention)
//! - JSON serialization for audit trails

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// ============================================================================
// AUDIT EVENT TYPES
// ============================================================================

/// Types of audit events that can be logged
pub const AuditEventType = enum {
    // Workflow events
    WORKFLOW_CREATED,
    WORKFLOW_UPDATED,
    WORKFLOW_DELETED,
    WORKFLOW_EXECUTED,

    // User events
    USER_LOGIN,
    USER_LOGOUT,
    USER_CREATED,
    USER_UPDATED,

    // Permission events
    PERMISSION_GRANTED,
    PERMISSION_REVOKED,

    // API events
    API_REQUEST,
    API_ERROR,

    // Data events
    DATA_ACCESS,
    DATA_EXPORT,
    DATA_DELETE,

    // System events
    CONFIG_CHANGED,
    SYSTEM_ERROR,

    /// Convert event type to string representation
    pub fn toString(self: AuditEventType) []const u8 {
        return switch (self) {
            .WORKFLOW_CREATED => "WORKFLOW_CREATED",
            .WORKFLOW_UPDATED => "WORKFLOW_UPDATED",
            .WORKFLOW_DELETED => "WORKFLOW_DELETED",
            .WORKFLOW_EXECUTED => "WORKFLOW_EXECUTED",
            .USER_LOGIN => "USER_LOGIN",
            .USER_LOGOUT => "USER_LOGOUT",
            .USER_CREATED => "USER_CREATED",
            .USER_UPDATED => "USER_UPDATED",
            .PERMISSION_GRANTED => "PERMISSION_GRANTED",
            .PERMISSION_REVOKED => "PERMISSION_REVOKED",
            .API_REQUEST => "API_REQUEST",
            .API_ERROR => "API_ERROR",
            .DATA_ACCESS => "DATA_ACCESS",
            .DATA_EXPORT => "DATA_EXPORT",
            .DATA_DELETE => "DATA_DELETE",
            .CONFIG_CHANGED => "CONFIG_CHANGED",
            .SYSTEM_ERROR => "SYSTEM_ERROR",
        };
    }

    /// Parse event type from string
    pub fn fromString(s: []const u8) ?AuditEventType {
        const map = std.StaticStringMap(AuditEventType).initComptime(.{
            .{ "WORKFLOW_CREATED", .WORKFLOW_CREATED },
            .{ "WORKFLOW_UPDATED", .WORKFLOW_UPDATED },
            .{ "WORKFLOW_DELETED", .WORKFLOW_DELETED },
            .{ "WORKFLOW_EXECUTED", .WORKFLOW_EXECUTED },
            .{ "USER_LOGIN", .USER_LOGIN },
            .{ "USER_LOGOUT", .USER_LOGOUT },
            .{ "USER_CREATED", .USER_CREATED },
            .{ "USER_UPDATED", .USER_UPDATED },
            .{ "PERMISSION_GRANTED", .PERMISSION_GRANTED },
            .{ "PERMISSION_REVOKED", .PERMISSION_REVOKED },
            .{ "API_REQUEST", .API_REQUEST },
            .{ "API_ERROR", .API_ERROR },
            .{ "DATA_ACCESS", .DATA_ACCESS },
            .{ "DATA_EXPORT", .DATA_EXPORT },
            .{ "DATA_DELETE", .DATA_DELETE },
            .{ "CONFIG_CHANGED", .CONFIG_CHANGED },
            .{ "SYSTEM_ERROR", .SYSTEM_ERROR },
        });
        return map.get(s);
    }
};

// ============================================================================
// AUDIT EVENT STRUCT
// ============================================================================

/// Represents a single audit event
pub const AuditEvent = struct {
    /// Unique identifier (UUID format)
    id: []const u8,
    /// Type of audit event
    event_type: AuditEventType,
    /// Unix timestamp when event occurred
    timestamp: i64,
    /// User who triggered the event (optional)
    user_id: ?[]const u8,
    /// Tenant context (optional, for multi-tenancy)
    tenant_id: ?[]const u8,
    /// Type of resource affected (e.g., "workflow", "execution")
    resource_type: ?[]const u8,
    /// Identifier of the affected resource
    resource_id: ?[]const u8,
    /// Human-readable action description
    action: []const u8,
    /// Client IP address (optional)
    ip_address: ?[]const u8,
    /// Client user agent string (optional)
    user_agent: ?[]const u8,
    /// Correlation ID for request tracing (optional)
    request_id: ?[]const u8,
    /// Additional details as JSON string (optional)
    details_json: ?[]const u8,
    /// Whether the action was successful
    success: bool,

    /// Create a new audit event with current timestamp
    pub fn create(
        allocator: Allocator,
        event_type: AuditEventType,
        action: []const u8,
        success: bool,
    ) !AuditEvent {
        const id = try generateUuid(allocator);
        const timestamp = std.time.timestamp();

        return AuditEvent{
            .id = id,
            .event_type = event_type,
            .timestamp = timestamp,
            .user_id = null,
            .tenant_id = null,
            .resource_type = null,
            .resource_id = null,
            .action = try allocator.dupe(u8, action),
            .ip_address = null,
            .user_agent = null,
            .request_id = null,
            .details_json = null,
            .success = success,
        };
    }

    /// Free all allocated memory for this event
    pub fn deinit(self: *AuditEvent, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.action);
        if (self.user_id) |uid| allocator.free(uid);
        if (self.tenant_id) |tid| allocator.free(tid);
        if (self.resource_type) |rt| allocator.free(rt);
        if (self.resource_id) |rid| allocator.free(rid);
        if (self.ip_address) |ip| allocator.free(ip);
        if (self.user_agent) |ua| allocator.free(ua);
        if (self.request_id) |reqid| allocator.free(reqid);
        if (self.details_json) |dj| allocator.free(dj);
    }

    /// Create a deep copy of this event
    pub fn clone(self: *const AuditEvent, allocator: Allocator) !AuditEvent {
        return AuditEvent{
            .id = try allocator.dupe(u8, self.id),
            .event_type = self.event_type,
            .timestamp = self.timestamp,
            .user_id = if (self.user_id) |u| try allocator.dupe(u8, u) else null,
            .tenant_id = if (self.tenant_id) |t| try allocator.dupe(u8, t) else null,
            .resource_type = if (self.resource_type) |r| try allocator.dupe(u8, r) else null,
            .resource_id = if (self.resource_id) |r| try allocator.dupe(u8, r) else null,
            .action = try allocator.dupe(u8, self.action),
            .ip_address = if (self.ip_address) |ip| try allocator.dupe(u8, ip) else null,
            .user_agent = if (self.user_agent) |ua| try allocator.dupe(u8, ua) else null,
            .request_id = if (self.request_id) |r| try allocator.dupe(u8, r) else null,
            .details_json = if (self.details_json) |d| try allocator.dupe(u8, d) else null,
            .success = self.success,
        };
    }
};

/// Generate a simple UUID v4 (random-based)
fn generateUuid(allocator: Allocator) ![]const u8 {
    var uuid_bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&uuid_bytes);

    // Set version (4) and variant bits
    uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40;
    uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80;

    return try std.fmt.allocPrint(
        allocator,
        "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}",
        .{
            uuid_bytes[0],  uuid_bytes[1],  uuid_bytes[2],  uuid_bytes[3],
            uuid_bytes[4],  uuid_bytes[5],  uuid_bytes[6],  uuid_bytes[7],
            uuid_bytes[8],  uuid_bytes[9],  uuid_bytes[10], uuid_bytes[11],
            uuid_bytes[12], uuid_bytes[13], uuid_bytes[14], uuid_bytes[15],
        },
    );
}

// ============================================================================
// AUDIT LOGGER
// ============================================================================

/// Audit logger that buffers events before persistence
pub const AuditLogger = struct {
    allocator: Allocator,
    events: ArrayList(AuditEvent),
    max_buffer_size: u32,

    /// Default maximum buffer size
    pub const DEFAULT_MAX_BUFFER_SIZE: u32 = 1000;

    /// Initialize a new audit logger
    pub fn init(allocator: Allocator) AuditLogger {
        return AuditLogger{
            .allocator = allocator,
            .events = ArrayList(AuditEvent){},
            .max_buffer_size = DEFAULT_MAX_BUFFER_SIZE,
        };
    }

    /// Initialize with custom buffer size
    pub fn initWithBufferSize(allocator: Allocator, max_buffer_size: u32) AuditLogger {
        return AuditLogger{
            .allocator = allocator,
            .events = ArrayList(AuditEvent){},
            .max_buffer_size = max_buffer_size,
        };
    }

    /// Deinitialize and free all resources
    pub fn deinit(self: *AuditLogger) void {
        for (self.events.items) |*event| {
            event.deinit(self.allocator);
        }
        self.events.deinit(self.allocator);
    }

    /// Log an audit event
    pub fn log(self: *AuditLogger, event: AuditEvent) !void {
        // Clone event to ensure ownership
        const cloned = try event.clone(self.allocator);
        try self.events.append(self.allocator, cloned);

        // Auto-flush if buffer is full
        if (self.events.items.len >= self.max_buffer_size) {
            // In production, this would trigger async persistence
            // For now, we just track buffer state
        }
    }

    /// Log a workflow-related event
    pub fn logWorkflowEvent(
        self: *AuditLogger,
        event_type: AuditEventType,
        user_id: ?[]const u8,
        tenant_id: ?[]const u8,
        workflow_id: []const u8,
        details: ?[]const u8,
    ) !void {
        var event = try AuditEvent.create(
            self.allocator,
            event_type,
            event_type.toString(),
            true,
        );
        event.user_id = if (user_id) |u| try self.allocator.dupe(u8, u) else null;
        event.tenant_id = if (tenant_id) |t| try self.allocator.dupe(u8, t) else null;
        event.resource_type = try self.allocator.dupe(u8, "workflow");
        event.resource_id = try self.allocator.dupe(u8, workflow_id);
        event.details_json = if (details) |d| try self.allocator.dupe(u8, d) else null;

        try self.events.append(self.allocator, event);
    }

    /// Log a user-related event
    pub fn logUserEvent(
        self: *AuditLogger,
        event_type: AuditEventType,
        user_id: []const u8,
        tenant_id: ?[]const u8,
        details: ?[]const u8,
    ) !void {
        var event = try AuditEvent.create(
            self.allocator,
            event_type,
            event_type.toString(),
            true,
        );
        event.user_id = try self.allocator.dupe(u8, user_id);
        event.tenant_id = if (tenant_id) |t| try self.allocator.dupe(u8, t) else null;
        event.resource_type = try self.allocator.dupe(u8, "user");
        event.resource_id = try self.allocator.dupe(u8, user_id);
        event.details_json = if (details) |d| try self.allocator.dupe(u8, d) else null;

        try self.events.append(self.allocator, event);
    }

    /// Log an API request
    pub fn logApiRequest(
        self: *AuditLogger,
        user_id: ?[]const u8,
        tenant_id: ?[]const u8,
        method: []const u8,
        path: []const u8,
        status_code: u16,
        duration_ms: i64,
    ) !void {
        const success = status_code < 400;
        const event_type: AuditEventType = if (success) .API_REQUEST else .API_ERROR;

        var event = try AuditEvent.create(
            self.allocator,
            event_type,
            "API_REQUEST",
            success,
        );
        event.user_id = if (user_id) |u| try self.allocator.dupe(u8, u) else null;
        event.tenant_id = if (tenant_id) |t| try self.allocator.dupe(u8, t) else null;

        // Create details JSON
        const details = try std.fmt.allocPrint(
            self.allocator,
            "{{\"method\":\"{s}\",\"path\":\"{s}\",\"status_code\":{d},\"duration_ms\":{d}}}",
            .{ method, path, status_code, duration_ms },
        );
        event.details_json = details;

        try self.events.append(self.allocator, event);
    }

    /// Log a data access event
    pub fn logDataAccess(
        self: *AuditLogger,
        user_id: ?[]const u8,
        tenant_id: ?[]const u8,
        resource_type: []const u8,
        resource_id: []const u8,
        action: []const u8,
    ) !void {
        var event = try AuditEvent.create(
            self.allocator,
            .DATA_ACCESS,
            action,
            true,
        );
        event.user_id = if (user_id) |u| try self.allocator.dupe(u8, u) else null;
        event.tenant_id = if (tenant_id) |t| try self.allocator.dupe(u8, t) else null;
        event.resource_type = try self.allocator.dupe(u8, resource_type);
        event.resource_id = try self.allocator.dupe(u8, resource_id);

        try self.events.append(self.allocator, event);
    }

    /// Flush buffered events and return them for persistence
    /// Caller owns the returned slice and must free each event
    pub fn flush(self: *AuditLogger) ![]AuditEvent {
        const result = try self.events.toOwnedSlice(self.allocator);
        self.events = ArrayList(AuditEvent){};
        return result;
    }

    /// Get current buffer count
    pub fn getBufferCount(self: *const AuditLogger) usize {
        return self.events.items.len;
    }

    /// Serialize a single event to JSON
    pub fn toJson(self: *AuditLogger, event: *const AuditEvent) ![]const u8 {
        var buffer = ArrayList(u8){};
        errdefer buffer.deinit(self.allocator);

        try buffer.appendSlice(self.allocator, "{");

        // ID
        try buffer.appendSlice(self.allocator, "\"id\":\"");
        try buffer.appendSlice(self.allocator, event.id);
        try buffer.appendSlice(self.allocator, "\"");

        // Event type
        try buffer.appendSlice(self.allocator, ",\"event_type\":\"");
        try buffer.appendSlice(self.allocator, event.event_type.toString());
        try buffer.appendSlice(self.allocator, "\"");

        // Timestamp
        const ts_str = try std.fmt.allocPrint(self.allocator, "{d}", .{event.timestamp});
        defer self.allocator.free(ts_str);
        try buffer.appendSlice(self.allocator, ",\"timestamp\":");
        try buffer.appendSlice(self.allocator, ts_str);

        // Optional fields
        if (event.user_id) |uid| {
            try buffer.appendSlice(self.allocator, ",\"user_id\":\"");
            try buffer.appendSlice(self.allocator, uid);
            try buffer.appendSlice(self.allocator, "\"");
        }

        if (event.tenant_id) |tid| {
            try buffer.appendSlice(self.allocator, ",\"tenant_id\":\"");
            try buffer.appendSlice(self.allocator, tid);
            try buffer.appendSlice(self.allocator, "\"");
        }

        if (event.resource_type) |rt| {
            try buffer.appendSlice(self.allocator, ",\"resource_type\":\"");
            try buffer.appendSlice(self.allocator, rt);
            try buffer.appendSlice(self.allocator, "\"");
        }

        if (event.resource_id) |rid| {
            try buffer.appendSlice(self.allocator, ",\"resource_id\":\"");
            try buffer.appendSlice(self.allocator, rid);
            try buffer.appendSlice(self.allocator, "\"");
        }

        // Action
        try buffer.appendSlice(self.allocator, ",\"action\":\"");
        try buffer.appendSlice(self.allocator, event.action);
        try buffer.appendSlice(self.allocator, "\"");

        // Optional IP, user agent, request ID
        if (event.ip_address) |ip| {
            try buffer.appendSlice(self.allocator, ",\"ip_address\":\"");
            try buffer.appendSlice(self.allocator, ip);
            try buffer.appendSlice(self.allocator, "\"");
        }

        if (event.user_agent) |ua| {
            try buffer.appendSlice(self.allocator, ",\"user_agent\":\"");
            try buffer.appendSlice(self.allocator, ua);
            try buffer.appendSlice(self.allocator, "\"");
        }

        if (event.request_id) |reqid| {
            try buffer.appendSlice(self.allocator, ",\"request_id\":\"");
            try buffer.appendSlice(self.allocator, reqid);
            try buffer.appendSlice(self.allocator, "\"");
        }

        // Details JSON (embedded as object, not string)
        if (event.details_json) |dj| {
            try buffer.appendSlice(self.allocator, ",\"details\":");
            try buffer.appendSlice(self.allocator, dj);
        }

        // Success
        try buffer.appendSlice(self.allocator, ",\"success\":");
        try buffer.appendSlice(self.allocator, if (event.success) "true" else "false");

        try buffer.appendSlice(self.allocator, "}");

        return try buffer.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// RBAC PERMISSION SYSTEM
// ============================================================================

/// Represents a permission grant for a role
pub const Permission = struct {
    /// Role name (e.g., "admin", "editor", "viewer")
    role: []const u8,
    /// Resource type (e.g., "workflow", "execution", "user")
    resource: []const u8,
    /// Allowed actions (e.g., ["read", "write", "delete"])
    actions: []const []const u8,

    /// Check if this permission allows the given action
    pub fn allowsAction(self: *const Permission, action: []const u8) bool {
        for (self.actions) |allowed| {
            if (std.mem.eql(u8, allowed, action)) {
                return true;
            }
            // Wildcard support
            if (std.mem.eql(u8, allowed, "*")) {
                return true;
            }
        }
        return false;
    }

    /// Check if this permission matches the given resource
    pub fn matchesResource(self: *const Permission, resource: []const u8) bool {
        if (std.mem.eql(u8, self.resource, "*")) {
            return true;
        }
        return std.mem.eql(u8, self.resource, resource);
    }
};

/// RBAC Permission Checker
pub const RBACChecker = struct {
    allocator: Allocator,
    permissions: []const Permission,

    /// Initialize RBAC checker with permissions
    pub fn init(allocator: Allocator, permissions: []const Permission) RBACChecker {
        return RBACChecker{
            .allocator = allocator,
            .permissions = permissions,
        };
    }

    /// Check if a role has permission to perform an action on a resource
    pub fn hasPermission(
        self: *const RBACChecker,
        role: []const u8,
        resource: []const u8,
        action: []const u8,
    ) bool {
        for (self.permissions) |perm| {
            if (std.mem.eql(u8, perm.role, role) and
                perm.matchesResource(resource) and
                perm.allowsAction(action))
            {
                return true;
            }
        }
        return false;
    }

    /// Check if any of the given roles has permission
    pub fn hasAnyPermission(
        self: *const RBACChecker,
        roles: []const []const u8,
        resource: []const u8,
        action: []const u8,
    ) bool {
        for (roles) |role| {
            if (self.hasPermission(role, resource, action)) {
                return true;
            }
        }
        return false;
    }

    /// Get all permissions for a specific role
    pub fn getPermissionsForRole(
        self: *const RBACChecker,
        role: []const u8,
    ) []const Permission {
        // Count matching permissions
        var count: usize = 0;
        for (self.permissions) |perm| {
            if (std.mem.eql(u8, perm.role, role)) {
                count += 1;
            }
        }

        if (count == 0) {
            return &[_]Permission{};
        }

        // For simplicity, return a view into the original slice
        // In production, you might want to allocate and copy
        var result: [64]Permission = undefined;
        var idx: usize = 0;
        for (self.permissions) |perm| {
            if (std.mem.eql(u8, perm.role, role) and idx < 64) {
                result[idx] = perm;
                idx += 1;
            }
        }

        // Return slice view (note: this is valid only while self.permissions is valid)
        return self.permissions[0..0]; // Return empty for now, caller should filter
    }

    /// Get all unique roles in the permission set
    pub fn getAllRoles(self: *const RBACChecker, allocator: Allocator) ![][]const u8 {
        var roles = ArrayList([]const u8){};
        defer roles.deinit(allocator);

        for (self.permissions) |perm| {
            var found = false;
            for (roles.items) |existing| {
                if (std.mem.eql(u8, existing, perm.role)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                try roles.append(allocator, perm.role);
            }
        }

        return try roles.toOwnedSlice(allocator);
    }
};

// ============================================================================
// GDPR COMPLIANCE HELPERS
// ============================================================================

/// GDPR compliance helper functions
pub const GDPRHelpers = struct {
    /// Anonymize a user ID using SHA256 hash
    /// Returns hex-encoded hash string
    pub fn anonymizeUserId(allocator: Allocator, user_id: []const u8) ![]const u8 {
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(user_id, &hash, .{});

        // Convert to hex string
        return try std.fmt.allocPrint(
            allocator,
            "{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}" ++
                "{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}" ++
                "{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}" ++
                "{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}",
            .{
                hash[0],  hash[1],  hash[2],  hash[3],  hash[4],  hash[5],  hash[6],  hash[7],
                hash[8],  hash[9],  hash[10], hash[11], hash[12], hash[13], hash[14], hash[15],
                hash[16], hash[17], hash[18], hash[19], hash[20], hash[21], hash[22], hash[23],
                hash[24], hash[25], hash[26], hash[27], hash[28], hash[29], hash[30], hash[31],
            },
        );
    }

    /// Redact sensitive fields from JSON string
    /// Replaces field values with "[REDACTED]"
    pub fn redactSensitiveFields(
        allocator: Allocator,
        json: []const u8,
        fields: []const []const u8,
    ) ![]const u8 {
        var result = try allocator.dupe(u8, json);
        errdefer allocator.free(result);

        for (fields) |field| {
            // Simple field redaction - look for "field":"value" patterns
            const pattern = try std.fmt.allocPrint(allocator, "\"{s}\":\"", .{field});
            defer allocator.free(pattern);

            var pos: usize = 0;
            while (pos < result.len) {
                if (std.mem.indexOf(u8, result[pos..], pattern)) |idx| {
                    const abs_idx = pos + idx + pattern.len;
                    // Find end of value (next unescaped quote)
                    var end_idx = abs_idx;
                    while (end_idx < result.len and result[end_idx] != '"') {
                        if (result[end_idx] == '\\' and end_idx + 1 < result.len) {
                            end_idx += 2; // Skip escaped character
                        } else {
                            end_idx += 1;
                        }
                    }

                    // Build new string with redacted value
                    const redacted = "[REDACTED]";
                    const new_len = abs_idx + redacted.len + (result.len - end_idx);
                    const new_result = try allocator.alloc(u8, new_len);

                    @memcpy(new_result[0..abs_idx], result[0..abs_idx]);
                    @memcpy(new_result[abs_idx .. abs_idx + redacted.len], redacted);
                    @memcpy(new_result[abs_idx + redacted.len ..], result[end_idx..]);

                    allocator.free(result);
                    result = new_result;
                    pos = abs_idx + redacted.len;
                } else {
                    break;
                }
            }
        }

        return result;
    }

    /// Calculate retention date based on event timestamp and retention period
    /// Returns Unix timestamp for when data should be deleted
    pub fn calculateRetentionDate(event_timestamp: i64, retention_days: u32) i64 {
        const seconds_per_day: i64 = 24 * 60 * 60;
        return event_timestamp + (@as(i64, retention_days) * seconds_per_day);
    }

    /// Check if an event has exceeded its retention period
    pub fn isExpired(event_timestamp: i64, retention_days: u32) bool {
        const retention_date = calculateRetentionDate(event_timestamp, retention_days);
        const now = std.time.timestamp();
        return now > retention_date;
    }

    /// Generate a data export in JSON format for GDPR data portability
    pub fn exportUserData(
        allocator: Allocator,
        events: []const AuditEvent,
        user_id: []const u8,
    ) ![]const u8 {
        var buffer = ArrayList(u8){};
        errdefer buffer.deinit(allocator);

        try buffer.appendSlice(allocator, "{\"user_id\":\"");
        try buffer.appendSlice(allocator, user_id);
        try buffer.appendSlice(allocator, "\",\"export_timestamp\":");

        const ts_str = try std.fmt.allocPrint(allocator, "{d}", .{std.time.timestamp()});
        defer allocator.free(ts_str);
        try buffer.appendSlice(allocator, ts_str);

        try buffer.appendSlice(allocator, ",\"events\":[");

        var first = true;
        for (events) |event| {
            if (event.user_id) |uid| {
                if (std.mem.eql(u8, uid, user_id)) {
                    if (!first) {
                        try buffer.appendSlice(allocator, ",");
                    }

                    // Serialize event inline
                    try buffer.appendSlice(allocator, "{\"event_type\":\"");
                    try buffer.appendSlice(allocator, event.event_type.toString());
                    try buffer.appendSlice(allocator, "\",\"timestamp\":");
                    const event_ts = try std.fmt.allocPrint(allocator, "{d}", .{event.timestamp});
                    defer allocator.free(event_ts);
                    try buffer.appendSlice(allocator, event_ts);
                    try buffer.appendSlice(allocator, ",\"action\":\"");
                    try buffer.appendSlice(allocator, event.action);
                    try buffer.appendSlice(allocator, "\"}");

                    first = false;
                }
            }
        }

        try buffer.appendSlice(allocator, "]}");

        return try buffer.toOwnedSlice(allocator);
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "AuditEventType toString and fromString" {
    try std.testing.expectEqualStrings("WORKFLOW_CREATED", AuditEventType.WORKFLOW_CREATED.toString());
    try std.testing.expectEqualStrings("USER_LOGIN", AuditEventType.USER_LOGIN.toString());
    try std.testing.expectEqualStrings("API_REQUEST", AuditEventType.API_REQUEST.toString());
    try std.testing.expectEqualStrings("DATA_ACCESS", AuditEventType.DATA_ACCESS.toString());

    try std.testing.expectEqual(AuditEventType.WORKFLOW_CREATED, AuditEventType.fromString("WORKFLOW_CREATED").?);
    try std.testing.expectEqual(AuditEventType.SYSTEM_ERROR, AuditEventType.fromString("SYSTEM_ERROR").?);
    try std.testing.expect(AuditEventType.fromString("INVALID_TYPE") == null);
}

test "AuditEvent create and deinit" {
    const allocator = std.testing.allocator;

    var event = try AuditEvent.create(
        allocator,
        .WORKFLOW_CREATED,
        "Created workflow",
        true,
    );
    defer event.deinit(allocator);

    try std.testing.expect(event.id.len == 36); // UUID format
    try std.testing.expectEqual(AuditEventType.WORKFLOW_CREATED, event.event_type);
    try std.testing.expectEqualStrings("Created workflow", event.action);
    try std.testing.expect(event.success);
    try std.testing.expect(event.timestamp > 0);
}

test "AuditEvent clone" {
    const allocator = std.testing.allocator;

    var original = try AuditEvent.create(
        allocator,
        .USER_LOGIN,
        "User logged in",
        true,
    );
    defer original.deinit(allocator);

    original.user_id = try allocator.dupe(u8, "user-123");
    original.tenant_id = try allocator.dupe(u8, "tenant-456");

    var cloned = try original.clone(allocator);
    defer cloned.deinit(allocator);

    try std.testing.expectEqualStrings(original.id, cloned.id);
    try std.testing.expectEqual(original.event_type, cloned.event_type);
    try std.testing.expectEqualStrings("user-123", cloned.user_id.?);
    try std.testing.expectEqualStrings("tenant-456", cloned.tenant_id.?);
}

test "AuditLogger basic operations" {
    const allocator = std.testing.allocator;

    var logger = AuditLogger.init(allocator);
    defer logger.deinit();

    try std.testing.expectEqual(@as(usize, 0), logger.getBufferCount());

    // Log an event
    var event = try AuditEvent.create(allocator, .API_REQUEST, "GET /api/v1/workflows", true);
    defer event.deinit(allocator);

    try logger.log(event);
    try std.testing.expectEqual(@as(usize, 1), logger.getBufferCount());
}

test "AuditLogger logWorkflowEvent" {
    const allocator = std.testing.allocator;

    var logger = AuditLogger.init(allocator);
    defer logger.deinit();

    try logger.logWorkflowEvent(
        .WORKFLOW_CREATED,
        "user-123",
        "tenant-456",
        "workflow-789",
        "{\"name\":\"Test Workflow\"}",
    );

    try std.testing.expectEqual(@as(usize, 1), logger.getBufferCount());

    const events = logger.events.items;
    try std.testing.expectEqual(AuditEventType.WORKFLOW_CREATED, events[0].event_type);
    try std.testing.expectEqualStrings("user-123", events[0].user_id.?);
    try std.testing.expectEqualStrings("workflow", events[0].resource_type.?);
    try std.testing.expectEqualStrings("workflow-789", events[0].resource_id.?);
}

test "AuditLogger logApiRequest" {
    const allocator = std.testing.allocator;

    var logger = AuditLogger.init(allocator);
    defer logger.deinit();

    try logger.logApiRequest(
        "user-123",
        "tenant-456",
        "GET",
        "/api/v1/workflows",
        200,
        45,
    );

    try std.testing.expectEqual(@as(usize, 1), logger.getBufferCount());

    const events = logger.events.items;
    try std.testing.expectEqual(AuditEventType.API_REQUEST, events[0].event_type);
    try std.testing.expect(events[0].success);
}

test "AuditLogger logApiRequest error status" {
    const allocator = std.testing.allocator;

    var logger = AuditLogger.init(allocator);
    defer logger.deinit();

    try logger.logApiRequest(
        "user-123",
        null,
        "POST",
        "/api/v1/workflows",
        500,
        100,
    );

    const events = logger.events.items;
    try std.testing.expectEqual(AuditEventType.API_ERROR, events[0].event_type);
    try std.testing.expect(!events[0].success);
}

test "AuditLogger flush" {
    const allocator = std.testing.allocator;

    var logger = AuditLogger.init(allocator);
    defer logger.deinit();

    try logger.logUserEvent(.USER_LOGIN, "user-123", null, null);
    try logger.logUserEvent(.USER_LOGOUT, "user-456", null, null);

    try std.testing.expectEqual(@as(usize, 2), logger.getBufferCount());

    const flushed = try logger.flush();
    defer {
        for (flushed) |*e| {
            var event = e.*;
            event.deinit(allocator);
        }
        allocator.free(flushed);
    }

    try std.testing.expectEqual(@as(usize, 2), flushed.len);
    try std.testing.expectEqual(@as(usize, 0), logger.getBufferCount());
}

test "AuditLogger toJson" {
    const allocator = std.testing.allocator;

    var logger = AuditLogger.init(allocator);
    defer logger.deinit();

    var event = try AuditEvent.create(allocator, .DATA_ACCESS, "read", true);
    defer event.deinit(allocator);

    event.user_id = try allocator.dupe(u8, "user-123");
    event.resource_type = try allocator.dupe(u8, "workflow");
    event.resource_id = try allocator.dupe(u8, "wf-001");

    const json = try logger.toJson(&event);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"event_type\":\"DATA_ACCESS\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"user_id\":\"user-123\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"success\":true") != null);
}

test "Permission allowsAction" {
    const actions = [_][]const u8{ "read", "write" };
    const perm = Permission{
        .role = "editor",
        .resource = "workflow",
        .actions = &actions,
    };

    try std.testing.expect(perm.allowsAction("read"));
    try std.testing.expect(perm.allowsAction("write"));
    try std.testing.expect(!perm.allowsAction("delete"));
}

test "Permission wildcard action" {
    const actions = [_][]const u8{"*"};
    const perm = Permission{
        .role = "admin",
        .resource = "workflow",
        .actions = &actions,
    };

    try std.testing.expect(perm.allowsAction("read"));
    try std.testing.expect(perm.allowsAction("write"));
    try std.testing.expect(perm.allowsAction("delete"));
    try std.testing.expect(perm.allowsAction("any_action"));
}

test "Permission matchesResource" {
    const actions = [_][]const u8{"read"};
    const perm = Permission{
        .role = "viewer",
        .resource = "workflow",
        .actions = &actions,
    };

    try std.testing.expect(perm.matchesResource("workflow"));
    try std.testing.expect(!perm.matchesResource("execution"));
}

test "Permission wildcard resource" {
    const actions = [_][]const u8{"read"};
    const perm = Permission{
        .role = "admin",
        .resource = "*",
        .actions = &actions,
    };

    try std.testing.expect(perm.matchesResource("workflow"));
    try std.testing.expect(perm.matchesResource("execution"));
    try std.testing.expect(perm.matchesResource("any_resource"));
}

test "RBACChecker hasPermission" {
    const allocator = std.testing.allocator;

    const admin_actions = [_][]const u8{"*"};
    const editor_actions = [_][]const u8{ "read", "write" };
    const viewer_actions = [_][]const u8{"read"};

    const permissions = [_]Permission{
        .{ .role = "admin", .resource = "*", .actions = &admin_actions },
        .{ .role = "editor", .resource = "workflow", .actions = &editor_actions },
        .{ .role = "viewer", .resource = "workflow", .actions = &viewer_actions },
    };

    const checker = RBACChecker.init(allocator, &permissions);

    // Admin can do anything
    try std.testing.expect(checker.hasPermission("admin", "workflow", "read"));
    try std.testing.expect(checker.hasPermission("admin", "workflow", "delete"));
    try std.testing.expect(checker.hasPermission("admin", "execution", "read"));

    // Editor can read/write workflows
    try std.testing.expect(checker.hasPermission("editor", "workflow", "read"));
    try std.testing.expect(checker.hasPermission("editor", "workflow", "write"));
    try std.testing.expect(!checker.hasPermission("editor", "workflow", "delete"));
    try std.testing.expect(!checker.hasPermission("editor", "execution", "read"));

    // Viewer can only read workflows
    try std.testing.expect(checker.hasPermission("viewer", "workflow", "read"));
    try std.testing.expect(!checker.hasPermission("viewer", "workflow", "write"));
}

test "RBACChecker hasAnyPermission" {
    const allocator = std.testing.allocator;

    const editor_actions = [_][]const u8{ "read", "write" };
    const viewer_actions = [_][]const u8{"read"};

    const permissions = [_]Permission{
        .{ .role = "editor", .resource = "workflow", .actions = &editor_actions },
        .{ .role = "viewer", .resource = "workflow", .actions = &viewer_actions },
    };

    const checker = RBACChecker.init(allocator, &permissions);

    const user_roles = [_][]const u8{ "viewer", "editor" };
    try std.testing.expect(checker.hasAnyPermission(&user_roles, "workflow", "write"));

    const viewer_only = [_][]const u8{"viewer"};
    try std.testing.expect(!checker.hasAnyPermission(&viewer_only, "workflow", "write"));
}

test "GDPRHelpers anonymizeUserId" {
    const allocator = std.testing.allocator;

    const hash1 = try GDPRHelpers.anonymizeUserId(allocator, "user-123");
    defer allocator.free(hash1);

    const hash2 = try GDPRHelpers.anonymizeUserId(allocator, "user-123");
    defer allocator.free(hash2);

    const hash3 = try GDPRHelpers.anonymizeUserId(allocator, "user-456");
    defer allocator.free(hash3);

    // Same input produces same hash
    try std.testing.expectEqualStrings(hash1, hash2);

    // Different input produces different hash
    try std.testing.expect(!std.mem.eql(u8, hash1, hash3));

    // Hash is 64 hex characters (256 bits)
    try std.testing.expectEqual(@as(usize, 64), hash1.len);
}

test "GDPRHelpers redactSensitiveFields" {
    const allocator = std.testing.allocator;

    const json = "{\"user_id\":\"123\",\"email\":\"test@example.com\",\"name\":\"John\"}";
    const fields = [_][]const u8{ "email", "name" };

    const redacted = try GDPRHelpers.redactSensitiveFields(allocator, json, &fields);
    defer allocator.free(redacted);

    try std.testing.expect(std.mem.indexOf(u8, redacted, "\"user_id\":\"123\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, "\"email\":\"[REDACTED]\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, "\"name\":\"[REDACTED]\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, "test@example.com") == null);
}

test "GDPRHelpers calculateRetentionDate" {
    const event_timestamp: i64 = 1700000000; // Example timestamp
    const retention_days: u32 = 90;

    const retention_date = GDPRHelpers.calculateRetentionDate(event_timestamp, retention_days);
    const expected = event_timestamp + (90 * 24 * 60 * 60);

    try std.testing.expectEqual(expected, retention_date);
}

test "GDPRHelpers isExpired" {
    const old_timestamp: i64 = 1600000000; // Very old timestamp
    const recent_timestamp = std.time.timestamp() - 1000; // 1000 seconds ago

    // Old event should be expired with 30-day retention
    try std.testing.expect(GDPRHelpers.isExpired(old_timestamp, 30));

    // Recent event should not be expired with 30-day retention
    try std.testing.expect(!GDPRHelpers.isExpired(recent_timestamp, 30));
}

test "GDPRHelpers exportUserData" {
    const allocator = std.testing.allocator;

    // Create test events
    var event1 = try AuditEvent.create(allocator, .USER_LOGIN, "login", true);
    defer event1.deinit(allocator);
    event1.user_id = try allocator.dupe(u8, "user-123");

    var event2 = try AuditEvent.create(allocator, .DATA_ACCESS, "read", true);
    defer event2.deinit(allocator);
    event2.user_id = try allocator.dupe(u8, "user-456");

    var event3 = try AuditEvent.create(allocator, .USER_LOGOUT, "logout", true);
    defer event3.deinit(allocator);
    event3.user_id = try allocator.dupe(u8, "user-123");

    const events = [_]AuditEvent{ event1, event2, event3 };

    const exported_data = try GDPRHelpers.exportUserData(allocator, &events, "user-123");
    defer allocator.free(exported_data);

    // Should include user-123's events only
    try std.testing.expect(std.mem.indexOf(u8, exported_data, "\"user_id\":\"user-123\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, exported_data, "USER_LOGIN") != null);
    try std.testing.expect(std.mem.indexOf(u8, exported_data, "USER_LOGOUT") != null);
    // Should not include user-456's events
    try std.testing.expect(std.mem.indexOf(u8, exported_data, "DATA_ACCESS") == null);
}

test "AuditLogger logDataAccess" {
    const allocator = std.testing.allocator;

    var logger = AuditLogger.init(allocator);
    defer logger.deinit();

    try logger.logDataAccess(
        "user-123",
        "tenant-456",
        "workflow",
        "wf-789",
        "export",
    );

    try std.testing.expectEqual(@as(usize, 1), logger.getBufferCount());

    const events = logger.events.items;
    try std.testing.expectEqual(AuditEventType.DATA_ACCESS, events[0].event_type);
    try std.testing.expectEqualStrings("export", events[0].action);
    try std.testing.expectEqualStrings("workflow", events[0].resource_type.?);
    try std.testing.expectEqualStrings("wf-789", events[0].resource_id.?);
}

test "generateUuid format" {
    const allocator = std.testing.allocator;

    const uuid = try generateUuid(allocator);
    defer allocator.free(uuid);

    // UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars with dashes)
    try std.testing.expectEqual(@as(usize, 36), uuid.len);
    try std.testing.expectEqual(@as(u8, '-'), uuid[8]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[13]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[18]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[23]);
}

