//! GraphQL Handler - Day 31
//!
//! HTTP handler for GraphQL endpoint.
//! Provides GraphQL query and mutation support alongside REST API.

const std = @import("std");
const Request = @import("../http/types.zig").Request;
const Response = @import("../http/types.zig").Response;
const handlers = @import("handlers.zig");

/// GraphQL endpoint handler
pub fn graphqlHandler(req: *Request, resp: *Response) !void {
    // Only accept POST for GraphQL queries
    if (req.method != .POST) {
        try resp.error_(405, "Method not allowed. Use POST for GraphQL queries");
        return;
    }
    
    // Parse GraphQL request
    const GraphQLRequest = struct {
        query: []const u8,
        variables: ?std.json.Value = null,
        operationName: ?[]const u8 = null,
    };
    
    const body = req.jsonBody(GraphQLRequest) catch {
        try resp.error_(400, "Invalid GraphQL request");
        return;
    };
    
    // Simple query routing based on operation name
    const query_lower = std.ascii.lowerString(req.allocator, body.query) catch body.query;
    
    if (std.mem.indexOf(u8, query_lower, "dataset(") != null) {
        // Single dataset query
        try handleDatasetQuery(req, resp, body.variables);
    } else if (std.mem.indexOf(u8, query_lower, "datasets") != null) {
        // List datasets query
        try handlers.listDatasets(req, resp);
    } else if (std.mem.indexOf(u8, query_lower, "createdataset") != null) {
        // Create dataset mutation
        try handlers.createDataset(req, resp);
    } else {
        try resp.error_(400, "Unknown GraphQL operation");
    }
}

/// Handle dataset query
fn handleDatasetQuery(req: *Request, resp: *Response, variables: ?std.json.Value) !void {
    // Extract ID from variables
    if (variables) |vars| {
        if (vars.object.get("id")) |id_value| {
            const id = id_value.string;
            try req.params.put(
                try req.allocator.dupe(u8, "id"),
                try req.allocator.dupe(u8, id),
            );
            try handlers.getDataset(req, resp);
            return;
        }
    }
    
    try resp.error_(400, "Missing required variable: id");
}

/// GraphiQL playground (development only)
pub fn graphiqlHandler(req: *Request, resp: *Response) !void {
    _ = req;
    
    const graphiql_html =
        \\<!DOCTYPE html>
        \\<html>
        \\<head>
        \\  <title>nMetaData GraphiQL</title>
        \\  <style>
        \\    body {
        \\      height: 100vh;
        \\      margin: 0;
        \\      font-family: Arial, sans-serif;
        \\    }
        \\    #graphiql {
        \\      height: 100vh;
        \\    }
        \\    .placeholder {
        \\      padding: 20px;
        \\      text-align: center;
        \\    }
        \\  </style>
        \\</head>
        \\<body>
        \\  <div class="placeholder">
        \\    <h1>🚀 nMetaData GraphiQL</h1>
        \\    <p>GraphQL playground for nMetaData API</p>
        \\    <p><strong>Endpoint:</strong> POST /api/v1/graphql</p>
        \\    <h2>Example Query:</h2>
        \\    <pre style="text-align: left; max-width: 600px; margin: 20px auto; padding: 15px; background: #f5f5f5;">
        \\{
        \\  "query": "query { dataset(id: \"ds-001\") { id name type } }",
        \\  "variables": { "id": "ds-001" }
        \\}</pre>
        \\    <p><a href="/api/v1/info">Back to API</a></p>
        \\  </div>
        \\</body>
        \\</html>
    ;
    
    resp.status = 200;
    try resp.html(graphiql_html);
}

/// Schema introspection handler
pub fn schemaHandler(req: *Request, resp: *Response) !void {
    _ = req;
    
    const SchemaInfo = struct {
        types: []const []const u8,
        queries: []const []const u8,
        mutations: []const []const u8,
    };
    
    resp.status = 200;
    try resp.json(SchemaInfo{
        .types = &[_][]const u8{ "Dataset", "DatasetType", "LineageEdge" },
        .queries = &[_][]const u8{ "dataset", "datasets", "lineage" },
        .mutations = &[_][]const u8{ "createDataset", "updateDataset", "deleteDataset", "createLineageEdge" },
    });
}
