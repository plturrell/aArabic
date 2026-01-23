// OData v4 Service Layer for nOpenAI Server
// Provides RESTful API for all HANA tables

const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;

/// OData query options from URL parameters
pub const QueryOptions = struct {
    filter: ?[]const u8 = null,     // $filter
    select: ?[]const u8 = null,     // $select
    orderby: ?[]const u8 = null,    // $orderby
    top: ?usize = null,             // $top
    skip: ?usize = null,            // $skip
    count: bool = false,            // $count
    expand: ?[]const u8 = null,     // $expand
};

/// Entity set definition
pub const EntitySet = struct {
    name: []const u8,           // e.g., "Prompts"
    entity_type: []const u8,    // e.g., "Prompt"
    table_name: []const u8,     // e.g., "NUCLEUS.PROMPTS"
};

// Import handler types
const PromptsHandler = @import("handlers/prompts.zig").PromptsHandler;
const ModelConfigurationsHandler = @import("handlers/model_configurations.zig").ModelConfigurationsHandler;
const UserSettingsHandler = @import("handlers/user_settings.zig").UserSettingsHandler;
const NotificationsHandler = @import("handlers/notifications.zig").NotificationsHandler;

/// OData service registry
pub const ODataService = struct {
    allocator: Allocator,
    entity_sets: []const EntitySet,
    
    // Handler references (optional - will be set if available)
    prompts_handler: ?*PromptsHandler,
    model_configs_handler: ?*ModelConfigurationsHandler,
    user_settings_handler: ?*UserSettingsHandler,
    notifications_handler: ?*NotificationsHandler,
    
    pub fn init(allocator: Allocator) !ODataService {
        // Define all 13 entity sets
        const entity_sets = [_]EntitySet{
            .{ .name = "Prompts", .entity_type = "Prompt", .table_name = "PROMPTS" },
            .{ .name = "ModelConfigurations", .entity_type = "ModelConfiguration", .table_name = "MODEL_CONFIGURATIONS" },
            .{ .name = "UserSettings", .entity_type = "UserSetting", .table_name = "USER_SETTINGS" },
            .{ .name = "Notifications", .entity_type = "Notification", .table_name = "NOTIFICATIONS" },
            .{ .name = "PromptComparisons", .entity_type = "PromptComparison", .table_name = "PROMPT_COMPARISONS" },
            .{ .name = "ModelVersionComparisons", .entity_type = "ModelVersionComparison", .table_name = "MODEL_VERSION_COMPARISONS" },
            .{ .name = "TrainingExperimentComparisons", .entity_type = "TrainingExperimentComparison", .table_name = "TRAINING_EXPERIMENT_COMPARISONS" },
            .{ .name = "PromptModeConfigs", .entity_type = "PromptModeConfig", .table_name = "PROMPT_MODE_CONFIGS" },
            .{ .name = "ModePresets", .entity_type = "ModePreset", .table_name = "MODE_PRESETS" },
            .{ .name = "ModelPerformance", .entity_type = "ModelPerformanceRecord", .table_name = "MODEL_PERFORMANCE" },
            .{ .name = "ModelVersions", .entity_type = "ModelVersion", .table_name = "MODEL_VERSIONS" },
            .{ .name = "TrainingExperiments", .entity_type = "TrainingExperiment", .table_name = "TRAINING_EXPERIMENTS" },
            .{ .name = "AuditLog", .entity_type = "AuditLogEntry", .table_name = "AUDIT_LOG" },
        };
        
        const sets = try allocator.dupe(EntitySet, &entity_sets);
        
        return ODataService{
            .allocator = allocator,
            .entity_sets = sets,
            .prompts_handler = null,
            .model_configs_handler = null,
            .user_settings_handler = null,
            .notifications_handler = null,
        };
    }
    
    /// Set handler references (called from server after handlers are initialized)
    pub fn setHandlers(
        self: *ODataService,
        prompts: ?*PromptsHandler,
        model_configs: ?*ModelConfigurationsHandler,
        user_settings: ?*UserSettingsHandler,
        notifications: ?*NotificationsHandler,
    ) void {
        self.prompts_handler = prompts;
        self.model_configs_handler = model_configs;
        self.user_settings_handler = user_settings;
        self.notifications_handler = notifications;
    }
    
    pub fn deinit(self: *ODataService) void {
        self.allocator.free(self.entity_sets);
    }
    
    /// Route OData request to appropriate handler
    pub fn handleRequest(
        self: *ODataService,
        method: []const u8,
        path: []const u8,
        body: ?[]const u8,
    ) ![]const u8 {
        
        // Remove /odata/v4/ prefix
        // Normalize path: strip /odata/v4 prefix and leading slash
        var odata_path = if (mem.startsWith(u8, path, "/odata/v4/"))
            path[10..]
        else if (mem.startsWith(u8, path, "/odata/v4"))
            path[9..]
        else
            path;

        if (mem.startsWith(u8, odata_path, "/")) {
            odata_path = odata_path[1..];
        }

        // Service document
        if (odata_path.len == 0) {
            return self.generateServiceDocument();
        }

        // Handle $metadata
        if (mem.eql(u8, odata_path, "$metadata")) {
            return self.generateMetadata();
        }

        // Parse entity set name and key
        var path_parts = mem.splitSequence(u8, odata_path, "/");
        const entity_set_name = path_parts.next() orelse return error.InvalidPath;
        
        // Find entity set
        const entity_set = self.findEntitySet(entity_set_name) orelse {
            const err_msg = try std.fmt.allocPrint(
                self.allocator,
                "{{\"error\":{{\"code\":\"NotFound\",\"message\":\"Entity set '{s}' not found\"}}}}",
                .{entity_set_name},
            );
            return err_msg;
        };
        
        // Parse query options from URL
        const query_opts = try self.parseQueryOptions(path);
        
        // Route based on method
        if (mem.eql(u8, method, "GET")) {
            // Check if single entity request: EntitySet(key)
            if (mem.indexOf(u8, entity_set_name, "(")) |_| {
                return self.handleGetSingle(entity_set, entity_set_name);
            } else {
                return self.handleList(entity_set, query_opts);
            }
        } else if (mem.eql(u8, method, "POST")) {
            return self.handleCreate(entity_set, body orelse "");
        } else if (mem.eql(u8, method, "PATCH") or mem.eql(u8, method, "PUT")) {
            return self.handleUpdate(entity_set, entity_set_name, body orelse "");
        } else if (mem.eql(u8, method, "DELETE")) {
            return self.handleDelete(entity_set, entity_set_name);
        }
        
        return error.MethodNotAllowed;
    }
    
    fn findEntitySet(self: *ODataService, name: []const u8) ?EntitySet {
        // Extract entity set name without key if present
        const entity_name = if (mem.indexOf(u8, name, "(")) |idx|
            name[0..idx]
        else
            name;
        
        for (self.entity_sets) |set| {
            if (mem.eql(u8, set.name, entity_name)) {
                return set;
            }
        }
        return null;
    }
    
    fn parseQueryOptions(self: *ODataService, path: []const u8) !QueryOptions {
        _ = self;
        var options = QueryOptions{};
        
        // Find query string start
        const query_start = mem.indexOf(u8, path, "?") orelse return options;
        const query_string = path[query_start + 1 ..];
        
        // Parse query parameters
        var params = mem.splitSequence(u8, query_string, "&");
        while (params.next()) |param| {
            if (mem.indexOf(u8, param, "=")) |eq_idx| {
                const key = param[0..eq_idx];
                const value = param[eq_idx + 1 ..];
                
                if (mem.eql(u8, key, "$filter") or mem.eql(u8, key, "%24filter")) {
                    options.filter = value;
                } else if (mem.eql(u8, key, "$select") or mem.eql(u8, key, "%24select")) {
                    options.select = value;
                } else if (mem.eql(u8, key, "$orderby") or mem.eql(u8, key, "%24orderby")) {
                    options.orderby = value;
                } else if (mem.eql(u8, key, "$top") or mem.eql(u8, key, "%24top")) {
                    options.top = std.fmt.parseInt(usize, value, 10) catch null;
                } else if (mem.eql(u8, key, "$skip") or mem.eql(u8, key, "%24skip")) {
                    options.skip = std.fmt.parseInt(usize, value, 10) catch null;
                } else if (mem.eql(u8, key, "$count") or mem.eql(u8, key, "%24count")) {
                    options.count = mem.eql(u8, value, "true");
                } else if (mem.eql(u8, key, "$expand") or mem.eql(u8, key, "%24expand")) {
                    options.expand = value;
                }
            }
        }
        
        return options;
    }

    fn generateServiceDocument(self: *ODataService) ![]const u8 {
        var body: std.ArrayList(u8) = .{};
        errdefer body.deinit(self.allocator);

        try body.appendSlice(self.allocator, "{\"@odata.context\":\"$metadata\",\"value\":[");
        var first = true;
        for (self.entity_sets) |set| {
            if (!first) try body.append(self.allocator, ',');
            first = false;
            try body.appendSlice(self.allocator, "{\"name\":\"");
            try body.appendSlice(self.allocator, set.name);
            try body.appendSlice(self.allocator, "\",\"kind\":\"EntitySet\",\"url\":\"");
            try body.appendSlice(self.allocator, set.name);
            try body.appendSlice(self.allocator, "\"}");
        }
        try body.appendSlice(self.allocator, "]}");

        return body.toOwnedSlice(self.allocator);
    }
    
    fn generateMetadata(self: *ODataService) ![]const u8 {
        var metadata: std.ArrayList(u8) = .{};
        errdefer metadata.deinit(self.allocator);
        
        try metadata.appendSlice(self.allocator, 
            \\<?xml version="1.0" encoding="UTF-8"?>
            \\<edmx:Edmx xmlns:edmx="http://docs.oasis-open.org/odata/ns/edmx" Version="4.0">
            \\  <edmx:DataServices>
            \\    <Schema xmlns="http://docs.oasis-open.org/odata/ns/edm" Namespace="nucleus.odata.v1">
            \\      <EntityContainer Name="EntityContainer">
            \\
        );
        
        // Add entity set references
        for (self.entity_sets) |set| {
            const entity_set_xml = try std.fmt.allocPrint(
                self.allocator,
                "        <EntitySet Name=\"{s}\" EntityType=\"nucleus.odata.v1.{s}\"/>\n",
                .{ set.name, set.entity_type },
            );
            try metadata.appendSlice(self.allocator, entity_set_xml);
            self.allocator.free(entity_set_xml);
        }
        
        try metadata.appendSlice(self.allocator,
            \\      </EntityContainer>
            \\    </Schema>
            \\  </edmx:DataServices>
            \\</edmx:Edmx>
            \\
        );
        
        return metadata.toOwnedSlice(self.allocator);
    }
    
    // Route list requests to appropriate handler
    fn handleList(self: *ODataService, entity_set: EntitySet, options: QueryOptions) ![]const u8 {
        // Route to handler if available
        if (mem.eql(u8, entity_set.name, "Prompts")) {
            if (self.prompts_handler) |handler| {
                return try handler.list(options);
            }
        } else if (mem.eql(u8, entity_set.name, "ModelConfigurations")) {
            if (self.model_configs_handler) |handler| {
                return try handler.list(options);
            }
        } else if (mem.eql(u8, entity_set.name, "UserSettings")) {
            if (self.user_settings_handler) |handler| {
                return try handler.list(options);
            }
        } else if (mem.eql(u8, entity_set.name, "Notifications")) {
            if (self.notifications_handler) |handler| {
                return try handler.list(options);
            }
        }
        
        // Fallback: return empty stub for unimplemented entity sets
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#{s}\",\"value\":[]}}",
            .{entity_set.name},
        );
    }
    
    fn handleGetSingle(self: *ODataService, entity_set: EntitySet, path: []const u8) ![]const u8 {
        // Extract key from path: EntitySet(key)
        const key_start = mem.indexOf(u8, path, "(") orelse return error.InvalidKey;
        const key_end = mem.indexOf(u8, path[key_start..], ")") orelse return error.InvalidKey;
        const key = path[key_start + 1 .. key_start + key_end];
        
        // Route to handler if available
        if (mem.eql(u8, entity_set.name, "Prompts")) {
            if (self.prompts_handler) |handler| {
                const id = std.fmt.parseInt(i32, key, 10) catch return error.InvalidKey;
                return try handler.get(id);
            }
        } else if (mem.eql(u8, entity_set.name, "ModelConfigurations")) {
            if (self.model_configs_handler) |handler| {
                return try handler.get(key);
            }
        } else if (mem.eql(u8, entity_set.name, "UserSettings")) {
            if (self.user_settings_handler) |handler| {
                return try handler.get(key);
            }
        } else if (mem.eql(u8, entity_set.name, "Notifications")) {
            if (self.notifications_handler) |handler| {
                return try handler.get(key);
            }
        }
        
        // Fallback
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#{s}/$entity\",\"error\":\"Not implemented\"}}",
            .{entity_set.name},
        );
    }
    
    fn handleCreate(self: *ODataService, entity_set: EntitySet, body: []const u8) ![]const u8 {
        // Route to handler if available
        if (mem.eql(u8, entity_set.name, "Prompts")) {
            if (self.prompts_handler) |handler| {
                return try handler.create(body);
            }
        } else if (mem.eql(u8, entity_set.name, "ModelConfigurations")) {
            if (self.model_configs_handler) |handler| {
                return try handler.create(body);
            }
        } else if (mem.eql(u8, entity_set.name, "UserSettings")) {
            if (self.user_settings_handler) |handler| {
                return try handler.create(body);
            }
        } else if (mem.eql(u8, entity_set.name, "Notifications")) {
            if (self.notifications_handler) |handler| {
                return try handler.create(body);
            }
        }
        
        // Fallback
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#{s}/$entity\",\"error\":\"Not implemented\"}}",
            .{entity_set.name},
        );
    }
    
    fn handleUpdate(self: *ODataService, entity_set: EntitySet, path: []const u8, body: []const u8) ![]const u8 {
        // Extract key from path: EntitySet(key)
        const key_start = mem.indexOf(u8, path, "(") orelse return error.InvalidKey;
        const key_end = mem.indexOf(u8, path[key_start..], ")") orelse return error.InvalidKey;
        const key = path[key_start + 1 .. key_start + key_end];
        
        // Route to handler if available
        if (mem.eql(u8, entity_set.name, "Prompts")) {
            if (self.prompts_handler) |handler| {
                const id = std.fmt.parseInt(i32, key, 10) catch return error.InvalidKey;
                return try handler.update(id, body);
            }
        } else if (mem.eql(u8, entity_set.name, "ModelConfigurations")) {
            if (self.model_configs_handler) |handler| {
                return try handler.update(key, body);
            }
        } else if (mem.eql(u8, entity_set.name, "UserSettings")) {
            if (self.user_settings_handler) |handler| {
                return try handler.update(key, body);
            }
        } else if (mem.eql(u8, entity_set.name, "Notifications")) {
            if (self.notifications_handler) |handler| {
                return try handler.update(key, body);
            }
        }
        
        // Fallback
        return try std.fmt.allocPrint(
            self.allocator,
            "{{\"@odata.context\":\"$metadata#{s}/$entity\",\"error\":\"Not implemented\"}}",
            .{entity_set.name},
        );
    }
    
    fn handleDelete(self: *ODataService, entity_set: EntitySet, path: []const u8) ![]const u8 {
        // Extract key from path: EntitySet(key)
        const key_start = mem.indexOf(u8, path, "(") orelse return error.InvalidKey;
        const key_end = mem.indexOf(u8, path[key_start..], ")") orelse return error.InvalidKey;
        const key = path[key_start + 1 .. key_start + key_end];
        
        // Route to handler if available
        if (mem.eql(u8, entity_set.name, "Prompts")) {
            if (self.prompts_handler) |handler| {
                const id = std.fmt.parseInt(i32, key, 10) catch return error.InvalidKey;
                try handler.delete(id);
                return try self.allocator.dupe(u8, ""); // 204 No Content
            }
        } else if (mem.eql(u8, entity_set.name, "ModelConfigurations")) {
            if (self.model_configs_handler) |handler| {
                try handler.delete(key);
                return try self.allocator.dupe(u8, "");
            }
        } else if (mem.eql(u8, entity_set.name, "UserSettings")) {
            if (self.user_settings_handler) |handler| {
                try handler.delete(key);
                return try self.allocator.dupe(u8, "");
            }
        } else if (mem.eql(u8, entity_set.name, "Notifications")) {
            if (self.notifications_handler) |handler| {
                try handler.delete(key);
                return try self.allocator.dupe(u8, "");
            }
        }
        
        // Fallback: not implemented
        return error.NotImplemented;
    }
};
