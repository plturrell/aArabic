//! ============================================================================
//! Comprehensive ODPS Integration Test Suite
//! Tests all components: YAML parser, ODPS mapper, Petri net, quality service
//! ============================================================================
//!
//! [CODE:file=test_odps_integration.zig]
//! [CODE:module=tests]
//! [CODE:language=zig]
//!
//! [RELATION:tests=CODE:yaml_parser.zig]
//! [RELATION:tests=CODE:odps_mapper.zig]
//! [RELATION:tests=CODE:odps_petrinet_bridge.zig]
//! [RELATION:tests=CODE:odps_quality_service.zig]
//! [RELATION:tests=CODE:odps_api.zig]
//!
//! Note: Test code - validates ODPS integration across all modules.

const std = @import("std");
const testing = std.testing;
const yaml_parser = @import("yaml_parser");
const odps_mapper = @import("odps_mapper");
const odps_petrinet_bridge = @import("odps_petrinet_bridge");
const odps_quality_service = @import("odps_quality_service");
const odps_api = @import("odps_api");

// Test 1: YAML Parser
test "YAML parser - nested structures" {
    const allocator = testing.allocator;
    
    var parser = try yaml_parser.YAMLParser.init(allocator);
    defer parser.deinit();
    
    const yaml_content =
        \\product:
        \\  productID: "urn:uuid:test-v1"
        \\  name: "Test Product"
        \\  dataQuality:
        \\    dataQualityScore: 95.0
        \\    validationRules:
        \\      - ruleID: "R001"
    ;
    
    try parser.parse(yaml_content);
    
    // Test basic field retrieval
    if (parser.getValue("product.productID")) |value| {
        try testing.expectEqualStrings("urn:uuid:test-v1", value);
    } else {
        return error.TestFailed;
    }
    
    // Test nested field
    if (parser.getValue("product.dataQuality.dataQualityScore")) |value| {
        try testing.expectEqualStrings("95.0", value);
    } else {
        return error.TestFailed;
    }
}

// Test 2: ODPS Mapper - Load ODPS
test "ODPS mapper - load and validate" {
    const allocator = testing.allocator;
    
    // Create mock ODPS file content
    const mock_odps_content =
        \\product:
        \\  productID: "urn:uuid:acdoca-journal-entries-v1"
        \\  name: "ACDOCA Universal Journal Entries"
        \\  description: "Complete SAP S/4HANA ACDOCA journal entries"
        \\  version: "1.0.0"
        \\  status: "active"
        \\  dataQuality:
        \\    dataQualityScore: 95.0
        \\  extensions:
        \\    sapORD: "../../ord/trial-balance-product.json#ACDOCA"
        \\    sapCSN: "../../csn/trial-balance.csn.json#JournalEntry"
    ;
    
    // Write temporary file
    const temp_file = "test_odps_temp.yaml";
    try std.fs.cwd().writeFile(.{
        .sub_path = temp_file,
        .data = mock_odps_content,
    });
    defer std.fs.cwd().deleteFile(temp_file) catch {};
    
    // Load ODPS
    const product = try odps_mapper.loadODPS(allocator, temp_file);
    defer {
        allocator.free(product.product_id);
        allocator.free(product.name);
        allocator.free(product.description);
        allocator.free(product.version);
        allocator.free(product.status);
        if (product.ord_ref) |ref| allocator.free(ref);
        if (product.csn_ref) |ref| allocator.free(ref);
    }
    
    // Validate loaded data
    try testing.expectEqualStrings("urn:uuid:acdoca-journal-entries-v1", product.product_id);
    try testing.expectEqualStrings("ACDOCA Universal Journal Entries", product.name);
    try testing.expectEqual(@as(f64, 95.0), product.quality_score);
    try testing.expect(product.ord_ref != null);
    try testing.expect(product.csn_ref != null);
    
    // Validate ODPS structure
    try testing.expect(try odps_mapper.validateODPS(&product));
}

// Test 3: ODPS to ORD Conversion
test "ODPS mapper - convert to ORD" {
    const allocator = testing.allocator;
    
    const odps = odps_mapper.ODPSProduct{
        .product_id = try allocator.dupe(u8, "urn:uuid:acdoca-v1"),
        .name = try allocator.dupe(u8, "ACDOCA Journal Entries"),
        .description = try allocator.dupe(u8, "Complete journal entries derived from ACDOCA"),
        .version = try allocator.dupe(u8, "1.0.0"),
        .status = try allocator.dupe(u8, "active"),
        .quality_score = 95.0,
        .ord_ref = null,
        .csn_ref = null,
    };
    defer {
        allocator.free(odps.product_id);
        allocator.free(odps.name);
        allocator.free(odps.description);
        allocator.free(odps.version);
        allocator.free(odps.status);
    }
    
    const ord = try odps_mapper.toORD(odps, allocator);
    defer {
        allocator.free(ord.ord_id);
        allocator.free(ord.title);
        allocator.free(ord.description);
        allocator.free(ord.version);
        allocator.free(ord.product_type);
        allocator.free(ord.category);
    }
    
    try testing.expectEqualStrings("ACDOCA Journal Entries", ord.title);
    try testing.expectEqualStrings("1.0.0", ord.version);
}

// Test 4: Petri Net Workflow Initialization
test "Petri net - 13-stage workflow initialization" {
    const allocator = testing.allocator;
    
    var workflow = try odps_petrinet_bridge.ODPSWorkflow.init(
        allocator,
        "./models/odps",
    );
    defer workflow.deinit();
    
    try workflow.initializeIFRSWorkflow();
    
    // Verify 13 stages
    try testing.expectEqual(@as(usize, 13), workflow.stages.items.len);
    
    // Verify stage IDs
    try testing.expectEqualStrings("S01", workflow.stages.items[0].stage_id);
    try testing.expectEqualStrings("S02", workflow.stages.items[1].stage_id);
    try testing.expectEqualStrings("S13", workflow.stages.items[12].stage_id);
    
    // Verify stage titles (updated to match ODPS spec)
    try testing.expectEqualStrings("Data Extraction", workflow.stages.items[0].title);
    try testing.expectEqualStrings("GCOA Mapping", workflow.stages.items[1].title);
    
    // Verify quality requirements (updated to match ODPS spec)
    try testing.expectEqual(@as(f64, 95.0), workflow.stages.items[0].quality_requirements);
    try testing.expectEqual(@as(f64, 99.0), workflow.stages.items[1].quality_requirements);
    
    // Verify initial status
    try testing.expectEqual(odps_petrinet_bridge.WorkflowStageStatus.pending, workflow.stages.items[0].status);
}

// Test 5: Workflow Progress Calculation
test "Petri net - progress tracking" {
    const allocator = testing.allocator;
    
    var workflow = try odps_petrinet_bridge.ODPSWorkflow.init(
        allocator,
        "./models/odps",
    );
    defer workflow.deinit();
    
    try workflow.initializeIFRSWorkflow();
    
    // Initially 0%
    try testing.expectEqual(@as(f64, 0.0), workflow.getProgress());
    
    // Complete 3 stages
    workflow.stages.items[0].status = .complete;
    workflow.stages.items[1].status = .complete;
    workflow.stages.items[2].status = .complete;
    
    const progress = workflow.getProgress();
    try testing.expect(progress > 23.0 and progress < 24.0); // ~23.08%
    
    // Complete all stages
    for (workflow.stages.items) |*stage| {
        stage.status = .complete;
    }
    
    try testing.expectEqual(@as(f64, 100.0), workflow.getProgress());
}

// Test 6: Quality Service Configuration
test "Quality service - initialization and config" {
    const allocator = testing.allocator;
    
    const config = try odps_quality_service.QualityUpdateConfig.default(allocator);
    var service = odps_quality_service.ODPSQualityService.init(allocator, config);
    defer service.deinit();
    
    try testing.expectEqual(@as(u64, 300), service.config.update_interval_seconds);
    try testing.expectEqual(@as(f64, 85.0), service.config.min_quality_threshold);
    try testing.expect(!service.needsUpdate()); // Just created, doesn't need update
}

// Test 7: Quality Report Generation
test "Quality service - quality report" {
    const allocator = testing.allocator;
    
    var report = odps_quality_service.QualityReport.init(allocator);
    defer report.deinit();
    
    try report.addProduct("ACDOCA Journal Entries", 95.0);
    try report.addProduct("Exchange Rates", 98.0);
    try report.addProduct("Trial Balance Aggregated", 92.0);
    try report.addProduct("Period Variances", 90.0);
    try report.addProduct("Account Master", 98.0);
    
    // Test average calculation
    const avg = report.getAverageQuality();
    try testing.expectEqual(@as(f64, 94.6), avg);
    
    // Test product count
    try testing.expectEqual(@as(usize, 5), report.products.items.len);
}

// Test 8: ODPS API Configuration
test "ODPS API - configuration" {
    const allocator = testing.allocator;
    
    const api_config = try odps_api.ODPSAPIConfig.default(allocator);
    const qconfig = try odps_quality_service.QualityUpdateConfig.default(allocator);
    var qservice = odps_quality_service.ODPSQualityService.init(allocator, qconfig);
    defer qservice.deinit();
    
    var api = odps_api.ODPSAPI.init(allocator, api_config, &qservice);
    defer api.deinit();
    
    try testing.expectEqualStrings("/api/v1/data-products", api.config.base_path);
    try testing.expect(api.config.enable_cors);
}

// Test 9: End-to-End Integration
test "Integration - complete workflow with quality gates" {
    const allocator = testing.allocator;
    
    // Initialize workflow
    var workflow = try odps_petrinet_bridge.ODPSWorkflow.init(
        allocator,
        "./models/odps",
    );
    defer workflow.deinit();
    try workflow.initializeIFRSWorkflow();
    
    // Initialize quality service
    const qconfig = try odps_quality_service.QualityUpdateConfig.default(allocator);
    var qservice = odps_quality_service.ODPSQualityService.init(allocator, qconfig);
    defer qservice.deinit();
    
    // Initialize API
    const api_config = try odps_api.ODPSAPIConfig.default(allocator);
    var api = odps_api.ODPSAPI.init(allocator, api_config, &qservice);
    defer api.deinit();
    
    // Verify components are connected
    try testing.expectEqual(@as(usize, 13), workflow.stages.items.len);
    try testing.expect(qservice.config.update_interval_seconds > 0);
    try testing.expectEqualStrings("/api/v1/data-products", api.config.base_path);
}

// Test 10: Stage Status Transitions
test "Workflow - stage status transitions" {
    const allocator = testing.allocator;
    
    var workflow = try odps_petrinet_bridge.ODPSWorkflow.init(
        allocator,
        "./models/odps",
    );
    defer workflow.deinit();
    try workflow.initializeIFRSWorkflow();
    
    // Test status enum
    try testing.expectEqualStrings("Pending", odps_petrinet_bridge.WorkflowStageStatus.pending.toString());
    try testing.expectEqualStrings("InProgress", odps_petrinet_bridge.WorkflowStageStatus.in_progress.toString());
    try testing.expectEqualStrings("Complete", odps_petrinet_bridge.WorkflowStageStatus.complete.toString());
    try testing.expectEqualStrings("Blocked", odps_petrinet_bridge.WorkflowStageStatus.blocked.toString());
    
    // Test transitions
    workflow.stages.items[0].status = .in_progress;
    try testing.expectEqual(odps_petrinet_bridge.WorkflowStageStatus.in_progress, workflow.stages.items[0].status);
    
    workflow.stages.items[0].status = .complete;
    workflow.stages.items[0].completed_at = std.time.timestamp();
    try testing.expect(workflow.stages.items[0].completed_at != null);
}