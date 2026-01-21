//! Marquez/OpenLineage Client for nWorkflow Data Lineage Tracking
//!
//! Implements the OpenLineage specification (https://openlineage.io/)
//! for tracking data lineage in workflow executions.
//!
//! Features:
//! - OpenLineage event structures (RunEvent, Job, Dataset, Facets)
//! - JSON serialization for events
//! - HTTP client for Marquez API integration
//! - Workflow tracking helpers

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// CONSTANTS
// ============================================================================

/// OpenLineage producer identifier
pub const OPENLINEAGE_PRODUCER = "nworkflow";

/// OpenLineage schema URL
pub const SCHEMA_URL = "https://openlineage.io/spec/1-0-5/OpenLineage.json";

/// Default Marquez API URL
pub const DEFAULT_API_URL = "http://localhost:5000/api/v1";

/// Default namespace
pub const DEFAULT_NAMESPACE = "nworkflow";

// ============================================================================
// EVENT TYPES
// ============================================================================

/// OpenLineage event types
pub const EventType = enum {
    START,
    RUNNING,
    COMPLETE,
    FAIL,
    ABORT,

    pub fn toString(self: EventType) []const u8 {
        return switch (self) {
            .START => "START",
            .RUNNING => "RUNNING",
            .COMPLETE => "COMPLETE",
            .FAIL => "FAIL",
            .ABORT => "ABORT",
        };
    }

    pub fn fromString(str: []const u8) !EventType {
        if (std.mem.eql(u8, str, "START")) return .START;
        if (std.mem.eql(u8, str, "RUNNING")) return .RUNNING;
        if (std.mem.eql(u8, str, "COMPLETE")) return .COMPLETE;
        if (std.mem.eql(u8, str, "FAIL")) return .FAIL;
        if (std.mem.eql(u8, str, "ABORT")) return .ABORT;
        return error.InvalidEventType;
    }
};

// ============================================================================
// FACETS
// ============================================================================

/// Parent run facet for nested workflows
pub const ParentRunFacet = struct {
    _producer: []const u8 = OPENLINEAGE_PRODUCER,
    _schemaURL: []const u8 = SCHEMA_URL ++ "#/$defs/ParentRunFacet",
    run: struct {
        runId: []const u8,
    },
    job: struct {
        namespace: []const u8,
        name: []const u8,
    },
};

/// Nominal time facet
pub const NominalTimeFacet = struct {
    _producer: []const u8 = OPENLINEAGE_PRODUCER,
    _schemaURL: []const u8 = SCHEMA_URL ++ "#/$defs/NominalTimeFacet",
    nominalStartTime: []const u8,
    nominalEndTime: ?[]const u8 = null,
};

/// Documentation facet for job description
pub const DocumentationFacet = struct {
    _producer: []const u8 = OPENLINEAGE_PRODUCER,
    _schemaURL: []const u8 = SCHEMA_URL ++ "#/$defs/DocumentationJobFacet",
    description: []const u8,
};

/// Source code location facet
pub const SourceCodeLocationFacet = struct {
    _producer: []const u8 = OPENLINEAGE_PRODUCER,
    _schemaURL: []const u8 = SCHEMA_URL ++ "#/$defs/SourceCodeLocationJobFacet",
    type: []const u8 = "git",
    url: []const u8,
    repoUrl: ?[]const u8 = null,
    path: ?[]const u8 = null,
    version: ?[]const u8 = null,
    tag: ?[]const u8 = null,
    branch: ?[]const u8 = null,
};

/// Error message facet for failed runs
pub const ErrorMessageFacet = struct {
    _producer: []const u8 = OPENLINEAGE_PRODUCER,
    _schemaURL: []const u8 = SCHEMA_URL ++ "#/$defs/ErrorMessageRunFacet",
    message: []const u8,
    programmingLanguage: []const u8 = "zig",
    stackTrace: ?[]const u8 = null,
};

/// Schema facet for datasets
pub const SchemaFacet = struct {
    _producer: []const u8 = OPENLINEAGE_PRODUCER,
    _schemaURL: []const u8 = SCHEMA_URL ++ "#/$defs/SchemaDatasetFacet",
    fields: []const SchemaField,
};

/// Schema field definition
pub const SchemaField = struct {
    name: []const u8,
    type: []const u8,
    description: ?[]const u8 = null,
};

/// Run facets container
pub const RunFacets = struct {
    parent: ?ParentRunFacet = null,
    nominalTime: ?NominalTimeFacet = null,
    errorMessage: ?ErrorMessageFacet = null,
};

/// Job facets container
pub const JobFacets = struct {
    documentation: ?DocumentationFacet = null,
    sourceCodeLocation: ?SourceCodeLocationFacet = null,
};

/// Dataset facets container
pub const DatasetFacets = struct {
    schema: ?SchemaFacet = null,
};

// ============================================================================
// CORE STRUCTURES
// ============================================================================

/// Run identifier
pub const Run = struct {
    runId: []const u8,
    facets: ?RunFacets = null,
};

/// Job identifier
pub const Job = struct {
    namespace: []const u8,
    name: []const u8,
    facets: ?JobFacets = null,
};

/// Dataset identifier
pub const Dataset = struct {
    namespace: []const u8,
    name: []const u8,
    facets: ?DatasetFacets = null,
};

/// OpenLineage Run Event
pub const RunEvent = struct {
    eventType: EventType,
    eventTime: []const u8,
    run: Run,
    job: Job,
    inputs: []const Dataset,
    outputs: []const Dataset,
    producer: []const u8 = OPENLINEAGE_PRODUCER,
    schemaURL: []const u8 = SCHEMA_URL,
};

// ============================================================================
// MARQUEZ CLIENT
// ============================================================================

/// Marquez/OpenLineage API client
pub const MarquezClient = struct {
    allocator: Allocator,
    api_url: []const u8,
    namespace: []const u8,
    http_client: std.http.Client,

    /// Initialize a new MarquezClient
    pub fn init(allocator: Allocator) MarquezClient {
        return initWithConfig(allocator, DEFAULT_API_URL, DEFAULT_NAMESPACE);
    }

    /// Initialize with custom configuration
    pub fn initWithConfig(
        allocator: Allocator,
        api_url: []const u8,
        namespace: []const u8,
    ) MarquezClient {
        return MarquezClient{
            .allocator = allocator,
            .api_url = api_url,
            .namespace = namespace,
            .http_client = std.http.Client{ .allocator = allocator },
        };
    }

    /// Clean up resources
    pub fn deinit(self: *MarquezClient) void {
        self.http_client.deinit();
    }

    /// Emit an OpenLineage event to the Marquez API
    pub fn emitEvent(self: *MarquezClient, event: RunEvent) !void {
        const json_body = try self.serializeEvent(event);
        defer self.allocator.free(json_body);

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/lineage",
            .{self.api_url},
        );
        defer self.allocator.free(url);

        try self.postJson(url, json_body);
    }

    /// Create a RunEvent with the given parameters
    pub fn createRunEvent(
        self: *MarquezClient,
        event_type: EventType,
        run_id: []const u8,
        job_name: []const u8,
        inputs: []const Dataset,
        outputs: []const Dataset,
    ) !RunEvent {
        const event_time = try self.formatIso8601(std.time.timestamp());

        return RunEvent{
            .eventType = event_type,
            .eventTime = event_time,
            .run = Run{ .runId = run_id },
            .job = Job{ .namespace = self.namespace, .name = job_name },
            .inputs = inputs,
            .outputs = outputs,
        };
    }

    /// Format timestamp as ISO8601 string
    pub fn formatIso8601(self: *MarquezClient, timestamp: i64) ![]const u8 {
        const epoch_seconds = @as(u64, @intCast(timestamp));

        // Calculate date/time components from epoch
        const days_since_epoch = epoch_seconds / 86400;
        const seconds_today = epoch_seconds % 86400;

        const hours = seconds_today / 3600;
        const minutes = (seconds_today % 3600) / 60;
        const seconds = seconds_today % 60;

        // Simplified year calculation (approximate for 2024-2030)
        const year: u32 = 1970 + @as(u32, @intCast(days_since_epoch / 365));
        const day_of_year = days_since_epoch % 365;
        const month: u32 = @as(u32, @intCast(day_of_year / 30)) + 1;
        const day: u32 = @as(u32, @intCast(day_of_year % 30)) + 1;

        return try std.fmt.allocPrint(
            self.allocator,
            "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z",
            .{ year, month, day, hours, minutes, seconds },
        );
    }

    /// Generate a UUID v4
    pub fn generateUuid(self: *MarquezClient) ![]const u8 {
        var random_bytes: [16]u8 = undefined;
        std.crypto.random.bytes(&random_bytes);

        // Set version (4) and variant (RFC 4122)
        random_bytes[6] = (random_bytes[6] & 0x0F) | 0x40;
        random_bytes[8] = (random_bytes[8] & 0x3F) | 0x80;

        return try std.fmt.allocPrint(
            self.allocator,
            "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}",
            .{
                random_bytes[0], random_bytes[1], random_bytes[2], random_bytes[3],
                random_bytes[4], random_bytes[5], random_bytes[6], random_bytes[7],
                random_bytes[8], random_bytes[9], random_bytes[10], random_bytes[11],
                random_bytes[12], random_bytes[13], random_bytes[14], random_bytes[15],
            },
        );
    }

    // ========================================================================
    // WORKFLOW TRACKING HELPERS
    // ========================================================================

    /// Track workflow start event
    pub fn trackWorkflowStart(
        self: *MarquezClient,
        workflow_id: []const u8,
        workflow_name: []const u8,
        input_datasets: []const Dataset,
    ) !void {
        const event = try self.createRunEvent(
            .START,
            workflow_id,
            workflow_name,
            input_datasets,
            &[_]Dataset{},
        );
        try self.emitEvent(event);
        self.allocator.free(event.eventTime);
    }

    /// Track workflow completion event
    pub fn trackWorkflowComplete(
        self: *MarquezClient,
        workflow_id: []const u8,
        workflow_name: []const u8,
        output_datasets: []const Dataset,
    ) !void {
        const event = try self.createRunEvent(
            .COMPLETE,
            workflow_id,
            workflow_name,
            &[_]Dataset{},
            output_datasets,
        );
        try self.emitEvent(event);
        self.allocator.free(event.eventTime);
    }

    /// Track workflow failure event
    pub fn trackWorkflowFail(
        self: *MarquezClient,
        workflow_id: []const u8,
        workflow_name: []const u8,
        error_message: []const u8,
    ) !void {
        const event_time = try self.formatIso8601(std.time.timestamp());
        defer self.allocator.free(event_time);

        const event = RunEvent{
            .eventType = .FAIL,
            .eventTime = event_time,
            .run = Run{
                .runId = workflow_id,
                .facets = RunFacets{
                    .errorMessage = ErrorMessageFacet{
                        .message = error_message,
                    },
                },
            },
            .job = Job{ .namespace = self.namespace, .name = workflow_name },
            .inputs = &[_]Dataset{},
            .outputs = &[_]Dataset{},
        };
        try self.emitEvent(event);
    }

    /// Track individual node execution within a workflow
    pub fn trackNodeExecution(
        self: *MarquezClient,
        workflow_id: []const u8,
        node_name: []const u8,
        inputs: []const Dataset,
        outputs: []const Dataset,
    ) !void {
        const node_run_id = try self.generateUuid();
        defer self.allocator.free(node_run_id);

        const event_time = try self.formatIso8601(std.time.timestamp());
        defer self.allocator.free(event_time);

        // Create job name as workflow.node
        const job_name = try std.fmt.allocPrint(
            self.allocator,
            "{s}.{s}",
            .{ workflow_id, node_name },
        );
        defer self.allocator.free(job_name);

        const event = RunEvent{
            .eventType = .COMPLETE,
            .eventTime = event_time,
            .run = Run{
                .runId = node_run_id,
                .facets = RunFacets{
                    .parent = ParentRunFacet{
                        .run = .{ .runId = workflow_id },
                        .job = .{ .namespace = self.namespace, .name = workflow_id },
                    },
                },
            },
            .job = Job{ .namespace = self.namespace, .name = job_name },
            .inputs = inputs,
            .outputs = outputs,
        };
        try self.emitEvent(event);
    }

    // ========================================================================
    // JSON SERIALIZATION
    // ========================================================================

    /// Serialize a RunEvent to JSON
    pub fn serializeEvent(self: *MarquezClient, event: RunEvent) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(self.allocator);

        const writer = buffer.writer(self.allocator);

        try writer.writeAll("{");

        // Schema URL and producer
        try writer.print("\"schemaURL\":\"{s}\",", .{event.schemaURL});
        try writer.print("\"producer\":\"{s}\",", .{event.producer});

        // Event type and time
        try writer.print("\"eventType\":\"{s}\",", .{event.eventType.toString()});
        try writer.print("\"eventTime\":\"{s}\",", .{event.eventTime});

        // Run
        try writer.writeAll("\"run\":{");
        try writer.print("\"runId\":\"{s}\"", .{event.run.runId});
        if (event.run.facets) |facets| {
            try self.serializeRunFacets(writer, facets);
        }
        try writer.writeAll("},");

        // Job
        try writer.writeAll("\"job\":{");
        try writer.print("\"namespace\":\"{s}\",", .{event.job.namespace});
        try writer.print("\"name\":\"{s}\"", .{event.job.name});
        if (event.job.facets) |facets| {
            try self.serializeJobFacets(writer, facets);
        }
        try writer.writeAll("},");

        // Inputs
        try writer.writeAll("\"inputs\":[");
        for (event.inputs, 0..) |dataset, i| {
            if (i > 0) try writer.writeAll(",");
            try self.serializeDataset(writer, dataset);
        }
        try writer.writeAll("],");

        // Outputs
        try writer.writeAll("\"outputs\":[");
        for (event.outputs, 0..) |dataset, i| {
            if (i > 0) try writer.writeAll(",");
            try self.serializeDataset(writer, dataset);
        }
        try writer.writeAll("]");

        try writer.writeAll("}");

        return buffer.toOwnedSlice(self.allocator);
    }

    /// Serialize run facets
    fn serializeRunFacets(self: *MarquezClient, writer: anytype, facets: RunFacets) !void {
        _ = self;
        try writer.writeAll(",\"facets\":{");
        var first = true;

        if (facets.parent) |parent| {
            if (!first) try writer.writeAll(",");
            first = false;
            try writer.writeAll("\"parent\":{");
            try writer.print("\"_producer\":\"{s}\",", .{parent._producer});
            try writer.print("\"_schemaURL\":\"{s}\",", .{parent._schemaURL});
            try writer.print("\"run\":{{\"runId\":\"{s}\"}},", .{parent.run.runId});
            try writer.print("\"job\":{{\"namespace\":\"{s}\",\"name\":\"{s}\"}}", .{
                parent.job.namespace,
                parent.job.name,
            });
            try writer.writeAll("}");
        }

        if (facets.nominalTime) |nominal| {
            if (!first) try writer.writeAll(",");
            first = false;
            try writer.writeAll("\"nominalTime\":{");
            try writer.print("\"_producer\":\"{s}\",", .{nominal._producer});
            try writer.print("\"_schemaURL\":\"{s}\",", .{nominal._schemaURL});
            try writer.print("\"nominalStartTime\":\"{s}\"", .{nominal.nominalStartTime});
            if (nominal.nominalEndTime) |end_time| {
                try writer.print(",\"nominalEndTime\":\"{s}\"", .{end_time});
            }
            try writer.writeAll("}");
        }

        if (facets.errorMessage) |err_msg| {
            if (!first) try writer.writeAll(",");
            // first = false;
            try writer.writeAll("\"errorMessage\":{");
            try writer.print("\"_producer\":\"{s}\",", .{err_msg._producer});
            try writer.print("\"_schemaURL\":\"{s}\",", .{err_msg._schemaURL});
            try writer.print("\"message\":\"{s}\",", .{err_msg.message});
            try writer.print("\"programmingLanguage\":\"{s}\"", .{err_msg.programmingLanguage});
            if (err_msg.stackTrace) |st| {
                try writer.print(",\"stackTrace\":\"{s}\"", .{st});
            }
            try writer.writeAll("}");
        }

        try writer.writeAll("}");
    }

    /// Serialize job facets
    fn serializeJobFacets(self: *MarquezClient, writer: anytype, facets: JobFacets) !void {
        _ = self;
        try writer.writeAll(",\"facets\":{");
        var first = true;

        if (facets.documentation) |doc| {
            if (!first) try writer.writeAll(",");
            first = false;
            try writer.writeAll("\"documentation\":{");
            try writer.print("\"_producer\":\"{s}\",", .{doc._producer});
            try writer.print("\"_schemaURL\":\"{s}\",", .{doc._schemaURL});
            try writer.print("\"description\":\"{s}\"", .{doc.description});
            try writer.writeAll("}");
        }

        if (facets.sourceCodeLocation) |src| {
            if (!first) try writer.writeAll(",");
            // first = false;
            try writer.writeAll("\"sourceCodeLocation\":{");
            try writer.print("\"_producer\":\"{s}\",", .{src._producer});
            try writer.print("\"_schemaURL\":\"{s}\",", .{src._schemaURL});
            try writer.print("\"type\":\"{s}\",", .{src.type});
            try writer.print("\"url\":\"{s}\"", .{src.url});
            if (src.repoUrl) |repo| {
                try writer.print(",\"repoUrl\":\"{s}\"", .{repo});
            }
            if (src.path) |path| {
                try writer.print(",\"path\":\"{s}\"", .{path});
            }
            if (src.version) |ver| {
                try writer.print(",\"version\":\"{s}\"", .{ver});
            }
            if (src.tag) |tag| {
                try writer.print(",\"tag\":\"{s}\"", .{tag});
            }
            if (src.branch) |branch| {
                try writer.print(",\"branch\":\"{s}\"", .{branch});
            }
            try writer.writeAll("}");
        }

        try writer.writeAll("}");
    }

    /// Serialize a dataset
    fn serializeDataset(self: *MarquezClient, writer: anytype, dataset: Dataset) !void {
        _ = self;
        try writer.writeAll("{");
        try writer.print("\"namespace\":\"{s}\",", .{dataset.namespace});
        try writer.print("\"name\":\"{s}\"", .{dataset.name});

        if (dataset.facets) |facets| {
            if (facets.schema) |schema| {
                try writer.writeAll(",\"facets\":{\"schema\":{");
                try writer.print("\"_producer\":\"{s}\",", .{schema._producer});
                try writer.print("\"_schemaURL\":\"{s}\",", .{schema._schemaURL});
                try writer.writeAll("\"fields\":[");
                for (schema.fields, 0..) |field, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.writeAll("{");
                    try writer.print("\"name\":\"{s}\",", .{field.name});
                    try writer.print("\"type\":\"{s}\"", .{field.type});
                    if (field.description) |desc| {
                        try writer.print(",\"description\":\"{s}\"", .{desc});
                    }
                    try writer.writeAll("}");
                }
                try writer.writeAll("]}}");
            }
        }

        try writer.writeAll("}");
    }

    // ========================================================================
    // HTTP HELPERS
    // ========================================================================

    /// POST JSON to the specified URL
    fn postJson(self: *MarquezClient, url: []const u8, body: []const u8) !void {
        const uri = try std.Uri.parse(url);

        var header_buffer: [4096]u8 = undefined;
        var req = try self.http_client.open(.POST, uri, .{
            .server_header_buffer = &header_buffer,
        });
        defer req.deinit();

        // Set Content-Type header
        req.headers.content_type = .{ .override = "application/json" };

        try req.send();
        try req.writeAll(body);
        try req.finish();
        try req.wait();

        // Check response status
        const status = @intFromEnum(req.response.status);
        if (status >= 400) {
            return error.HttpError;
        }
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "EventType toString and fromString" {
    try std.testing.expectEqualStrings("START", EventType.START.toString());
    try std.testing.expectEqualStrings("RUNNING", EventType.RUNNING.toString());
    try std.testing.expectEqualStrings("COMPLETE", EventType.COMPLETE.toString());
    try std.testing.expectEqualStrings("FAIL", EventType.FAIL.toString());
    try std.testing.expectEqualStrings("ABORT", EventType.ABORT.toString());

    try std.testing.expectEqual(EventType.START, try EventType.fromString("START"));
    try std.testing.expectEqual(EventType.COMPLETE, try EventType.fromString("COMPLETE"));
    try std.testing.expectError(error.InvalidEventType, EventType.fromString("INVALID"));
}

test "MarquezClient initialization" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    try std.testing.expectEqualStrings(DEFAULT_API_URL, client.api_url);
    try std.testing.expectEqualStrings(DEFAULT_NAMESPACE, client.namespace);
}

test "MarquezClient initWithConfig" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.initWithConfig(
        allocator,
        "http://custom:8080/api/v1",
        "custom-namespace",
    );
    defer client.deinit();

    try std.testing.expectEqualStrings("http://custom:8080/api/v1", client.api_url);
    try std.testing.expectEqualStrings("custom-namespace", client.namespace);
}

test "MarquezClient generateUuid format" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    const uuid = try client.generateUuid();
    defer allocator.free(uuid);

    // UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (36 chars)
    try std.testing.expectEqual(@as(usize, 36), uuid.len);
    try std.testing.expectEqual(@as(u8, '-'), uuid[8]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[13]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[18]);
    try std.testing.expectEqual(@as(u8, '-'), uuid[23]);
}

test "MarquezClient formatIso8601" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    // Test with a known timestamp (2024-01-01 00:00:00 UTC approx)
    const timestamp: i64 = 1704067200;
    const iso_time = try client.formatIso8601(timestamp);
    defer allocator.free(iso_time);

    // Should end with 'Z' for UTC
    try std.testing.expect(iso_time.len > 0);
    try std.testing.expectEqual(@as(u8, 'Z'), iso_time[iso_time.len - 1]);
    // Should contain 'T' separator
    try std.testing.expect(std.mem.indexOf(u8, iso_time, "T") != null);
}

test "MarquezClient serializeEvent basic" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    const event = RunEvent{
        .eventType = .START,
        .eventTime = "2024-01-01T00:00:00Z",
        .run = Run{ .runId = "test-run-123" },
        .job = Job{ .namespace = "test-ns", .name = "test-job" },
        .inputs = &[_]Dataset{},
        .outputs = &[_]Dataset{},
    };

    const json = try client.serializeEvent(event);
    defer allocator.free(json);

    // Verify JSON contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, json, "\"eventType\":\"START\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"runId\":\"test-run-123\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"namespace\":\"test-ns\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"test-job\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"schemaURL\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"producer\":\"nworkflow\"") != null);
}

test "MarquezClient serializeEvent with datasets" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    const inputs = [_]Dataset{
        Dataset{ .namespace = "db", .name = "users_table" },
        Dataset{ .namespace = "db", .name = "orders_table" },
    };

    const outputs = [_]Dataset{
        Dataset{ .namespace = "warehouse", .name = "user_orders_summary" },
    };

    const event = RunEvent{
        .eventType = .COMPLETE,
        .eventTime = "2024-01-01T12:00:00Z",
        .run = Run{ .runId = "workflow-456" },
        .job = Job{ .namespace = "analytics", .name = "aggregate-job" },
        .inputs = &inputs,
        .outputs = &outputs,
    };

    const json = try client.serializeEvent(event);
    defer allocator.free(json);

    // Verify inputs and outputs
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"users_table\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"orders_table\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"user_orders_summary\"") != null);
}

test "MarquezClient createRunEvent" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    const event = try client.createRunEvent(
        .START,
        "run-789",
        "my-workflow",
        &[_]Dataset{},
        &[_]Dataset{},
    );
    defer allocator.free(event.eventTime);

    try std.testing.expectEqual(EventType.START, event.eventType);
    try std.testing.expectEqualStrings("run-789", event.run.runId);
    try std.testing.expectEqualStrings("my-workflow", event.job.name);
    try std.testing.expectEqualStrings(DEFAULT_NAMESPACE, event.job.namespace);
}

test "Dataset with schema facet serialization" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    const fields = [_]SchemaField{
        SchemaField{ .name = "id", .type = "integer", .description = "Primary key" },
        SchemaField{ .name = "name", .type = "string", .description = null },
    };

    const dataset = Dataset{
        .namespace = "postgres",
        .name = "users",
        .facets = DatasetFacets{
            .schema = SchemaFacet{
                .fields = &fields,
            },
        },
    };

    const event = RunEvent{
        .eventType = .COMPLETE,
        .eventTime = "2024-01-01T00:00:00Z",
        .run = Run{ .runId = "test-run" },
        .job = Job{ .namespace = "test", .name = "test-job" },
        .inputs = &[_]Dataset{},
        .outputs = &[_]Dataset{dataset},
    };

    const json = try client.serializeEvent(event);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"schema\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"fields\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"id\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"integer\"") != null);
}

test "RunEvent with error facet" {
    const allocator = std.testing.allocator;

    var client = MarquezClient.init(allocator);
    defer client.deinit();

    const event = RunEvent{
        .eventType = .FAIL,
        .eventTime = "2024-01-01T00:00:00Z",
        .run = Run{
            .runId = "failed-run",
            .facets = RunFacets{
                .errorMessage = ErrorMessageFacet{
                    .message = "Connection timeout",
                    .stackTrace = "at line 42",
                },
            },
        },
        .job = Job{ .namespace = "test", .name = "failing-job" },
        .inputs = &[_]Dataset{},
        .outputs = &[_]Dataset{},
    };

    const json = try client.serializeEvent(event);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"errorMessage\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"message\":\"Connection timeout\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"programmingLanguage\":\"zig\"") != null);
}

test "Constants are correct" {
    try std.testing.expectEqualStrings("nworkflow", OPENLINEAGE_PRODUCER);
    try std.testing.expectEqualStrings("https://openlineage.io/spec/1-0-5/OpenLineage.json", SCHEMA_URL);
    try std.testing.expectEqualStrings("http://localhost:5000/api/v1", DEFAULT_API_URL);
    try std.testing.expectEqualStrings("nworkflow", DEFAULT_NAMESPACE);
}

