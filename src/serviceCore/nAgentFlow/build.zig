const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // zig-libc Configuration
    const zig_libc_options = b.addOptions();
    zig_libc_options.addOption(bool, "use_zig_libc", true);
    const zig_libc_config = zig_libc_options.createModule();

    // zig-libc Module (Rooted at src/lib.zig to allow internal relative imports)
    const zig_libc_mod = b.addModule("zig_libc", .{
        .root_source_file = b.path("../../nLang/n-c-sdk/lib/libc/zig-libc/src/lib.zig"),
    });
    zig_libc_mod.addImport("config", zig_libc_config);

    // Centralized SAP HANA SDK module shared across nAgentFlow
    const hana_sdk_mod = b.addModule("hana_sdk", .{
        .root_source_file = b.path("../../nLang/n-c-sdk/lib/hana/hana.zig"),
    });

    // Core Petri Net module (compatibility wrapper)
    const petri_net_mod = b.addModule("petri_net", .{
        .root_source_file = b.path("core/petri_net.zig"),
    });
    petri_net_mod.addImport("zig_libc", zig_libc_mod);

    // Create a single test step that will run all tests
    const test_step = b.step("test", "Run unit tests");

    // Tests for Petri Net
    // Tests for Petri Net - Create module first to add dependencies
    const petri_net_test_mod = b.createModule(.{
        .root_source_file = b.path("core/petri_net.zig"),
        .target = target,
        .optimize = optimize,
    });
    petri_net_test_mod.addImport("zig_libc", zig_libc_mod);

    const petri_net_tests = b.addTest(.{
        .root_module = petri_net_test_mod,
    });
    const run_petri_net_tests = b.addRunArtifact(petri_net_tests);
    test_step.dependOn(&run_petri_net_tests.step);
    
    // Focused test step for verification
    const test_petri_step = b.step("test-petri", "Run Petri net tests");
    test_petri_step.dependOn(&run_petri_net_tests.step);

    // Tests for Executor
    const executor_test_mod = b.createModule(.{
        .root_source_file = b.path("core/executor.zig"),
        .target = target,
        .optimize = optimize,
    });
    executor_test_mod.addImport("petri_net", petri_net_mod);
    const executor_tests = b.addTest(.{
        .root_module = executor_test_mod,
    });
    const run_executor_tests = b.addRunArtifact(executor_tests);
    test_step.dependOn(&run_executor_tests.step);

    // Tests for Workflow Parser
    const parser_test_mod = b.createModule(.{
        .root_source_file = b.path("core/workflow_parser.zig"),
        .target = target,
        .optimize = optimize,
    });
    parser_test_mod.addImport("petri_net", petri_net_mod);
    const parser_tests = b.addTest(.{
        .root_module = parser_test_mod,
    });
    const run_parser_tests = b.addRunArtifact(parser_tests);
    test_step.dependOn(&run_parser_tests.step);

    // Tests for Node Types
    const node_types_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("nodes/node_types.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_node_types_tests = b.addRunArtifact(node_types_tests);
    test_step.dependOn(&run_node_types_tests.step);

    // Shared modules for node factory and bridge
    const node_types_mod = b.addModule("node_types", .{
        .root_source_file = b.path("nodes/node_types.zig"),
    });
    const node_factory_mod = b.addModule("node_factory", .{
        .root_source_file = b.path("nodes/node_factory.zig"),
    });
    node_factory_mod.addImport("node_types", node_types_mod);
    
    // Tests for Node Factory
    const node_factory_test_mod = b.createModule(.{
        .root_source_file = b.path("nodes/node_factory.zig"),
        .target = target,
        .optimize = optimize,
    });
    node_factory_test_mod.addImport("node_types", node_types_mod);
    
    const node_factory_tests = b.addTest(.{
        .root_module = node_factory_test_mod,
    });
    const run_node_factory_tests = b.addRunArtifact(node_factory_tests);
    test_step.dependOn(&run_node_factory_tests.step);

    // Tests for Workflow-Node Bridge
    const bridge_test_mod = b.createModule(.{
        .root_source_file = b.path("integration/workflow_node_bridge.zig"),
        .target = target,
        .optimize = optimize,
    });
    bridge_test_mod.addImport("node_factory", node_factory_mod);
    bridge_test_mod.addImport("node_types", node_types_mod);
    
    const bridge_tests = b.addTest(.{
        .root_module = bridge_test_mod,
    });
    const run_bridge_tests = b.addRunArtifact(bridge_tests);
    test_step.dependOn(&run_bridge_tests.step);

    // Shared modules for integration (Day 15)
    const workflow_node_bridge_mod = b.addModule("workflow_node_bridge", .{
        .root_source_file = b.path("integration/workflow_node_bridge.zig"),
    });
    workflow_node_bridge_mod.addImport("node_factory", node_factory_mod);
    workflow_node_bridge_mod.addImport("node_types", node_types_mod);

    // Executor module (depends on petri_net)
    const executor_mod = b.addModule("executor", .{
        .root_source_file = b.path("core/executor.zig"),
    });
    executor_mod.addImport("petri_net", petri_net_mod);
    
    // Component system modules (Day 16)
    const component_metadata_mod = b.addModule("component_metadata", .{
        .root_source_file = b.path("components/component_metadata.zig"),
    });
    component_metadata_mod.addImport("node_types", node_types_mod);
    
    const component_registry_mod = b.addModule("registry", .{
        .root_source_file = b.path("components/registry.zig"),
    });
    component_registry_mod.addImport("node_types", node_types_mod);
    component_registry_mod.addImport("component_metadata", component_metadata_mod);
    
    const http_request_mod = b.addModule("http_request", .{
        .root_source_file = b.path("components/builtin/http_request.zig"),
    });
    http_request_mod.addImport("node_types", node_types_mod);
    http_request_mod.addImport("component_metadata", component_metadata_mod);
    
    // Day 17 component modules
    const transform_mod = b.addModule("transform", .{
        .root_source_file = b.path("components/builtin/transform.zig"),
    });
    transform_mod.addImport("node_types", node_types_mod);
    transform_mod.addImport("component_metadata", component_metadata_mod);
    
    const merge_mod = b.addModule("merge", .{
        .root_source_file = b.path("components/builtin/merge.zig"),
    });
    merge_mod.addImport("node_types", node_types_mod);
    merge_mod.addImport("component_metadata", component_metadata_mod);
    
    const filter_mod = b.addModule("filter", .{
        .root_source_file = b.path("components/builtin/filter.zig"),
    });
    filter_mod.addImport("node_types", node_types_mod);
    filter_mod.addImport("component_metadata", component_metadata_mod);
    
    // Day 18 component modules
    const split_mod = b.addModule("split", .{
        .root_source_file = b.path("components/builtin/split.zig"),
    });
    split_mod.addImport("node_types", node_types_mod);
    split_mod.addImport("component_metadata", component_metadata_mod);
    
    const logger_mod = b.addModule("logger", .{
        .root_source_file = b.path("components/builtin/logger.zig"),
    });
    logger_mod.addImport("node_types", node_types_mod);
    logger_mod.addImport("component_metadata", component_metadata_mod);
    
    const variable_mod = b.addModule("variable", .{
        .root_source_file = b.path("components/builtin/variable.zig"),
    });
    variable_mod.addImport("node_types", node_types_mod);
    variable_mod.addImport("component_metadata", component_metadata_mod);
    
    // Day 19 component modules
    const aggregate_mod = b.addModule("aggregate", .{
        .root_source_file = b.path("components/builtin/aggregate.zig"),
    });
    aggregate_mod.addImport("node_types", node_types_mod);
    aggregate_mod.addImport("component_metadata", component_metadata_mod);
    
    const sort_mod = b.addModule("sort", .{
        .root_source_file = b.path("components/builtin/sort.zig"),
    });
    sort_mod.addImport("node_types", node_types_mod);
    sort_mod.addImport("component_metadata", component_metadata_mod);
    
    const deduplicate_mod = b.addModule("deduplicate", .{
        .root_source_file = b.path("components/builtin/deduplicate.zig"),
    });
    deduplicate_mod.addImport("node_types", node_types_mod);
    deduplicate_mod.addImport("component_metadata", component_metadata_mod);
    
    // Day 20 data flow modules
    const data_packet_mod = b.addModule("data_packet", .{
        .root_source_file = b.path("data/data_packet.zig"),
    });
    
    const data_flow_mod = b.addModule("data_flow", .{
        .root_source_file = b.path("data/data_flow.zig"),
    });
    data_flow_mod.addImport("data_packet", data_packet_mod);
    
    // Day 21 streaming and pooling
    const data_stream_mod = b.addModule("data_stream", .{
        .root_source_file = b.path("data/data_stream.zig"),
    });
    data_stream_mod.addImport("data_packet", data_packet_mod);
    
    const data_pipeline_mod = b.addModule("data_pipeline", .{
        .root_source_file = b.path("data/data_pipeline.zig"),
    });
    data_pipeline_mod.addImport("data_packet", data_packet_mod);
    data_pipeline_mod.addImport("data_stream", data_stream_mod);
    
    const layerdata_integration_mod = b.addModule("layerdata_integration", .{
        .root_source_file = b.path("data/layerdata_integration.zig"),
    });
    layerdata_integration_mod.addImport("data_packet", data_packet_mod);
    layerdata_integration_mod.addImport("data_pipeline", data_pipeline_mod);
    
    // Day 22 LLM nodes module
    const llm_nodes_mod = b.addModule("llm_nodes", .{
        .root_source_file = b.path("nodes/llm/llm_nodes.zig"),
    });
    llm_nodes_mod.addImport("node_types", node_types_mod);
    llm_nodes_mod.addImport("data_packet", data_packet_mod);
    
    // Day 24 Advanced LLM module
    const llm_advanced_mod = b.addModule("llm_advanced", .{
        .root_source_file = b.path("nodes/llm/llm_advanced.zig"),
    });
    llm_advanced_mod.addImport("node_types", node_types_mod);
    llm_advanced_mod.addImport("data_packet", data_packet_mod);

    // Tests for Petri Node Executor (Day 15)
    const petri_executor_test_mod = b.createModule(.{
        .root_source_file = b.path("integration/petri_node_executor.zig"),
        .target = target,
        .optimize = optimize,
    });
    petri_executor_test_mod.addImport("petri_net", petri_net_mod);
    petri_executor_test_mod.addImport("executor", executor_mod);
    petri_executor_test_mod.addImport("node_types", node_types_mod);
    petri_executor_test_mod.addImport("workflow_node_bridge", workflow_node_bridge_mod);
    petri_executor_test_mod.addImport("node_factory", node_factory_mod);
    
    const petri_executor_tests = b.addTest(.{
        .root_module = petri_executor_test_mod,
    });
    const run_petri_executor_tests = b.addRunArtifact(petri_executor_tests);
    test_step.dependOn(&run_petri_executor_tests.step);

    // Shared module for Petri Node Executor
    const petri_node_executor_mod = b.addModule("petri_node_executor", .{
        .root_source_file = b.path("integration/petri_node_executor.zig"),
    });
    petri_node_executor_mod.addImport("petri_net", petri_net_mod);
    petri_node_executor_mod.addImport("executor", executor_mod);
    petri_node_executor_mod.addImport("node_types", node_types_mod);
    petri_node_executor_mod.addImport("workflow_node_bridge", workflow_node_bridge_mod);
    petri_node_executor_mod.addImport("node_factory", node_factory_mod);

    // Workflow parser module
    const workflow_parser_mod = b.addModule("workflow_parser", .{
        .root_source_file = b.path("core/workflow_parser.zig"),
    });
    workflow_parser_mod.addImport("petri_net", petri_net_mod);

    // BPMN Exporter module
    const bpmn_exporter_mod = b.addModule("bpmn_exporter", .{
        .root_source_file = b.path("bpmn/bpmn_exporter.zig"),
    });
    bpmn_exporter_mod.addImport("workflow_parser", workflow_parser_mod);

    // Tests for BPMN Exporter
    const bpmn_exporter_test_mod = b.createModule(.{
        .root_source_file = b.path("bpmn/bpmn_exporter.zig"),
        .target = target,
        .optimize = optimize,
    });
    bpmn_exporter_test_mod.addImport("workflow_parser", workflow_parser_mod);

    const bpmn_exporter_tests = b.addTest(.{
        .root_module = bpmn_exporter_test_mod,
    });
    const run_bpmn_exporter_tests = b.addRunArtifact(bpmn_exporter_tests);
    test_step.dependOn(&run_bpmn_exporter_tests.step);

    // ARIS EPC Parser module
    const epc_parser_mod = b.addModule("epc_parser", .{
        .root_source_file = b.path("aris/epc_parser.zig"),
    });
    epc_parser_mod.addImport("workflow_parser", workflow_parser_mod);

    // Tests for ARIS EPC Parser
    const epc_parser_test_mod = b.createModule(.{
        .root_source_file = b.path("aris/epc_parser.zig"),
        .target = target,
        .optimize = optimize,
    });
    epc_parser_test_mod.addImport("workflow_parser", workflow_parser_mod);

    const epc_parser_tests = b.addTest(.{
        .root_module = epc_parser_test_mod,
    });
    const run_epc_parser_tests = b.addRunArtifact(epc_parser_tests);
    test_step.dependOn(&run_epc_parser_tests.step);

    // Tests for Workflow Engine (Day 15)
    const workflow_engine_test_mod = b.createModule(.{
        .root_source_file = b.path("integration/workflow_engine.zig"),
        .target = target,
        .optimize = optimize,
    });
    workflow_engine_test_mod.addImport("workflow_parser", workflow_parser_mod);
    workflow_engine_test_mod.addImport("workflow_node_bridge", workflow_node_bridge_mod);
    workflow_engine_test_mod.addImport("petri_node_executor", petri_node_executor_mod);
    workflow_engine_test_mod.addImport("node_factory", node_factory_mod);
    workflow_engine_test_mod.addImport("node_types", node_types_mod);
    
    const workflow_engine_tests = b.addTest(.{
        .root_module = workflow_engine_test_mod,
    });
    const run_workflow_engine_tests = b.addRunArtifact(workflow_engine_tests);
    test_step.dependOn(&run_workflow_engine_tests.step);
    
    // Tests for Component Metadata (Day 16)
    const component_metadata_test_mod = b.createModule(.{
        .root_source_file = b.path("components/component_metadata.zig"),
        .target = target,
        .optimize = optimize,
    });
    component_metadata_test_mod.addImport("node_types", node_types_mod);
    
    const component_metadata_tests = b.addTest(.{
        .root_module = component_metadata_test_mod,
    });
    const run_component_metadata_tests = b.addRunArtifact(component_metadata_tests);
    test_step.dependOn(&run_component_metadata_tests.step);
    
    // Tests for Component Registry (Day 16)
    const component_registry_test_mod = b.createModule(.{
        .root_source_file = b.path("components/registry.zig"),
        .target = target,
        .optimize = optimize,
    });
    component_registry_test_mod.addImport("node_types", node_types_mod);
    component_registry_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const component_registry_tests = b.addTest(.{
        .root_module = component_registry_test_mod,
    });
    const run_component_registry_tests = b.addRunArtifact(component_registry_tests);
    test_step.dependOn(&run_component_registry_tests.step);
    
    // Tests for HTTP Request Component (Day 16)
    const http_request_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/http_request.zig"),
        .target = target,
        .optimize = optimize,
    });
    http_request_test_mod.addImport("node_types", node_types_mod);
    http_request_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const http_request_tests = b.addTest(.{
        .root_module = http_request_test_mod,
    });
    const run_http_request_tests = b.addRunArtifact(http_request_tests);
    test_step.dependOn(&run_http_request_tests.step);
    
    // Tests for Transform Component (Day 17)
    const transform_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/transform.zig"),
        .target = target,
        .optimize = optimize,
    });
    transform_test_mod.addImport("node_types", node_types_mod);
    transform_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const transform_tests = b.addTest(.{
        .root_module = transform_test_mod,
    });
    const run_transform_tests = b.addRunArtifact(transform_tests);
    test_step.dependOn(&run_transform_tests.step);
    
    // Tests for Merge Component (Day 17)
    const merge_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/merge.zig"),
        .target = target,
        .optimize = optimize,
    });
    merge_test_mod.addImport("node_types", node_types_mod);
    merge_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const merge_tests = b.addTest(.{
        .root_module = merge_test_mod,
    });
    const run_merge_tests = b.addRunArtifact(merge_tests);
    test_step.dependOn(&run_merge_tests.step);
    
    // Tests for Filter Component (Day 17)
    const filter_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/filter.zig"),
        .target = target,
        .optimize = optimize,
    });
    filter_test_mod.addImport("node_types", node_types_mod);
    filter_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const filter_tests = b.addTest(.{
        .root_module = filter_test_mod,
    });
    const run_filter_tests = b.addRunArtifact(filter_tests);
    test_step.dependOn(&run_filter_tests.step);
    
    // Tests for Split Component (Day 18)
    const split_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/split.zig"),
        .target = target,
        .optimize = optimize,
    });
    split_test_mod.addImport("node_types", node_types_mod);
    split_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const split_tests = b.addTest(.{
        .root_module = split_test_mod,
    });
    const run_split_tests = b.addRunArtifact(split_tests);
    test_step.dependOn(&run_split_tests.step);
    
    // Tests for Logger Component (Day 18)
    const logger_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/logger.zig"),
        .target = target,
        .optimize = optimize,
    });
    logger_test_mod.addImport("node_types", node_types_mod);
    logger_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const logger_tests = b.addTest(.{
        .root_module = logger_test_mod,
    });
    const run_logger_tests = b.addRunArtifact(logger_tests);
    test_step.dependOn(&run_logger_tests.step);
    
    // Tests for Variable Component (Day 18)
    const variable_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/variable.zig"),
        .target = target,
        .optimize = optimize,
    });
    variable_test_mod.addImport("node_types", node_types_mod);
    variable_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const variable_tests = b.addTest(.{
        .root_module = variable_test_mod,
    });
    const run_variable_tests = b.addRunArtifact(variable_tests);
    test_step.dependOn(&run_variable_tests.step);
    
    // Tests for Aggregate Component (Day 19)
    const aggregate_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/aggregate.zig"),
        .target = target,
        .optimize = optimize,
    });
    aggregate_test_mod.addImport("node_types", node_types_mod);
    aggregate_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const aggregate_tests = b.addTest(.{
        .root_module = aggregate_test_mod,
    });
    const run_aggregate_tests = b.addRunArtifact(aggregate_tests);
    test_step.dependOn(&run_aggregate_tests.step);
    
    // Tests for Sort Component (Day 19)
    const sort_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/sort.zig"),
        .target = target,
        .optimize = optimize,
    });
    sort_test_mod.addImport("node_types", node_types_mod);
    sort_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const sort_tests = b.addTest(.{
        .root_module = sort_test_mod,
    });
    const run_sort_tests = b.addRunArtifact(sort_tests);
    test_step.dependOn(&run_sort_tests.step);
    
    // Tests for Deduplicate Component (Day 19)
    const deduplicate_test_mod = b.createModule(.{
        .root_source_file = b.path("components/builtin/deduplicate.zig"),
        .target = target,
        .optimize = optimize,
    });
    deduplicate_test_mod.addImport("node_types", node_types_mod);
    deduplicate_test_mod.addImport("component_metadata", component_metadata_mod);
    
    const deduplicate_tests = b.addTest(.{
        .root_module = deduplicate_test_mod,
    });
    const run_deduplicate_tests = b.addRunArtifact(deduplicate_tests);
    test_step.dependOn(&run_deduplicate_tests.step);
    
    // Tests for Data Packet (Day 20)
    const data_packet_test_mod = b.createModule(.{
        .root_source_file = b.path("data/data_packet.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const data_packet_tests = b.addTest(.{
        .root_module = data_packet_test_mod,
    });
    const run_data_packet_tests = b.addRunArtifact(data_packet_tests);
    test_step.dependOn(&run_data_packet_tests.step);
    
    // Tests for Data Flow (Day 20)
    const data_flow_test_mod = b.createModule(.{
        .root_source_file = b.path("data/data_flow.zig"),
        .target = target,
        .optimize = optimize,
    });
    data_flow_test_mod.addImport("data_packet", data_packet_mod);
    
    const data_flow_tests = b.addTest(.{
        .root_module = data_flow_test_mod,
    });
    const run_data_flow_tests = b.addRunArtifact(data_flow_tests);
    test_step.dependOn(&run_data_flow_tests.step);
    
    // Tests for Data Stream (Day 21)
    const data_stream_test_mod = b.createModule(.{
        .root_source_file = b.path("data/data_stream.zig"),
        .target = target,
        .optimize = optimize,
    });
    data_stream_test_mod.addImport("data_packet", data_packet_mod);
    
    const data_stream_tests = b.addTest(.{
        .root_module = data_stream_test_mod,
    });
    const run_data_stream_tests = b.addRunArtifact(data_stream_tests);
    test_step.dependOn(&run_data_stream_tests.step);
    
    // Tests for Data Pipeline (Day 21)
    const data_pipeline_test_mod = b.createModule(.{
        .root_source_file = b.path("data/data_pipeline.zig"),
        .target = target,
        .optimize = optimize,
    });
    data_pipeline_test_mod.addImport("data_packet", data_packet_mod);
    data_pipeline_test_mod.addImport("data_stream", data_stream_mod);
    
    const data_pipeline_tests = b.addTest(.{
        .root_module = data_pipeline_test_mod,
    });
    const run_data_pipeline_tests = b.addRunArtifact(data_pipeline_tests);
    test_step.dependOn(&run_data_pipeline_tests.step);
    
    // Tests for LayerData Integration (Day 21)
    const layerdata_integration_test_mod = b.createModule(.{
        .root_source_file = b.path("data/layerdata_integration.zig"),
        .target = target,
        .optimize = optimize,
    });
    layerdata_integration_test_mod.addImport("data_packet", data_packet_mod);
    layerdata_integration_test_mod.addImport("data_pipeline", data_pipeline_mod);
    
    const layerdata_integration_tests = b.addTest(.{
        .root_module = layerdata_integration_test_mod,
    });
    const run_layerdata_integration_tests = b.addRunArtifact(layerdata_integration_tests);
    test_step.dependOn(&run_layerdata_integration_tests.step);
    
    // Tests for LLM Nodes (Day 22)
    const llm_nodes_test_mod = b.createModule(.{
        .root_source_file = b.path("nodes/llm/llm_nodes.zig"),
        .target = target,
        .optimize = optimize,
    });
    llm_nodes_test_mod.addImport("node_types", node_types_mod);
    llm_nodes_test_mod.addImport("data_packet", data_packet_mod);
    
    const llm_nodes_tests = b.addTest(.{
        .root_module = llm_nodes_test_mod,
    });
    const run_llm_nodes_tests = b.addRunArtifact(llm_nodes_tests);
    test_step.dependOn(&run_llm_nodes_tests.step);
    
    // Day 23 error recovery modules
    const error_recovery_mod = b.addModule("error_recovery", .{
        .root_source_file = b.path("error/error_recovery.zig"),
    });
    
    const node_error_handler_mod = b.addModule("node_error_handler", .{
        .root_source_file = b.path("error/node_error_handler.zig"),
    });
    node_error_handler_mod.addImport("error_recovery", error_recovery_mod);
    
    // Tests for Error Recovery (Day 23)
    const error_recovery_test_mod = b.createModule(.{
        .root_source_file = b.path("error/error_recovery.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const error_recovery_tests = b.addTest(.{
        .root_module = error_recovery_test_mod,
    });
    const run_error_recovery_tests = b.addRunArtifact(error_recovery_tests);
    test_step.dependOn(&run_error_recovery_tests.step);
    
    // Tests for Node Error Handler (Day 23)
    const node_error_handler_test_mod = b.createModule(.{
        .root_source_file = b.path("error/node_error_handler.zig"),
        .target = target,
        .optimize = optimize,
    });
    node_error_handler_test_mod.addImport("error_recovery", error_recovery_mod);
    
    const node_error_handler_tests = b.addTest(.{
        .root_module = node_error_handler_test_mod,
    });
    const run_node_error_handler_tests = b.addRunArtifact(node_error_handler_tests);
    test_step.dependOn(&run_node_error_handler_tests.step);
    
    // Tests for Advanced LLM (Day 24)
    const llm_advanced_test_mod = b.createModule(.{
        .root_source_file = b.path("nodes/llm/llm_advanced.zig"),
        .target = target,
        .optimize = optimize,
    });
    llm_advanced_test_mod.addImport("node_types", node_types_mod);
    llm_advanced_test_mod.addImport("data_packet", data_packet_mod);
    
    const llm_advanced_tests = b.addTest(.{
        .root_module = llm_advanced_test_mod,
    });
    const run_llm_advanced_tests = b.addRunArtifact(llm_advanced_tests);
    test_step.dependOn(&run_llm_advanced_tests.step);
    
    // Tests for State Manager (Day 25)
    const state_manager_test_mod = b.createModule(.{
        .root_source_file = b.path("memory/state_manager.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const state_manager_tests = b.addTest(.{
        .root_module = state_manager_test_mod,
    });
    const run_state_manager_tests = b.addRunArtifact(state_manager_tests);
    test_step.dependOn(&run_state_manager_tests.step);
    
    // Tests for State Versioning (Day 26)
    const state_versioning_test_mod = b.createModule(.{
        .root_source_file = b.path("memory/state_versioning.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const state_versioning_tests = b.addTest(.{
        .root_module = state_versioning_test_mod,
    });
    const run_state_versioning_tests = b.addRunArtifact(state_versioning_tests);
    test_step.dependOn(&run_state_versioning_tests.step);
    
    // Tests for Workflow Serialization (Day 27)
    const workflow_serialization_test_mod = b.createModule(.{
        .root_source_file = b.path("persistence/workflow_serialization.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const workflow_serialization_tests = b.addTest(.{
        .root_module = workflow_serialization_test_mod,
    });
    const run_workflow_serialization_tests = b.addRunArtifact(workflow_serialization_tests);
    test_step.dependOn(&run_workflow_serialization_tests.step);
    
    // Day 28 Langflow component modules
    const text_splitter_mod = b.addModule("text_splitter", .{
        .root_source_file = b.path("components/langflow/text_splitter.zig"),
    });
    text_splitter_mod.addImport("component_metadata", component_metadata_mod);
    text_splitter_mod.addImport("data_packet", data_packet_mod);
    
    _ = b.addModule("text_cleaner", .{
        .root_source_file = b.path("components/langflow/text_cleaner.zig"),
    });
    
    _ = b.addModule("control_flow", .{
        .root_source_file = b.path("components/langflow/control_flow.zig"),
    });
    
    _ = b.addModule("file_utils", .{
        .root_source_file = b.path("components/langflow/file_utils.zig"),
    });
    
    // Tests for Text Splitter (Day 28)
    const text_splitter_test_mod = b.createModule(.{
        .root_source_file = b.path("components/langflow/text_splitter.zig"),
        .target = target,
        .optimize = optimize,
    });
    text_splitter_test_mod.addImport("component_metadata", component_metadata_mod);
    text_splitter_test_mod.addImport("data_packet", data_packet_mod);
    
    const text_splitter_tests = b.addTest(.{
        .root_module = text_splitter_test_mod,
    });
    const run_text_splitter_tests = b.addRunArtifact(text_splitter_tests);
    test_step.dependOn(&run_text_splitter_tests.step);
    
    // Tests for Text Cleaner (Day 28)
    const text_cleaner_test_mod = b.createModule(.{
        .root_source_file = b.path("components/langflow/text_cleaner.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const text_cleaner_tests = b.addTest(.{
        .root_module = text_cleaner_test_mod,
    });
    const run_text_cleaner_tests = b.addRunArtifact(text_cleaner_tests);
    test_step.dependOn(&run_text_cleaner_tests.step);
    
    // Tests for Control Flow (Day 28)
    const control_flow_test_mod = b.createModule(.{
        .root_source_file = b.path("components/langflow/control_flow.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const control_flow_tests = b.addTest(.{
        .root_module = control_flow_test_mod,
    });
    const run_control_flow_tests = b.addRunArtifact(control_flow_tests);
    test_step.dependOn(&run_control_flow_tests.step);
    
    // Tests for File Utils (Day 28)
    const file_utils_test_mod = b.createModule(.{
        .root_source_file = b.path("components/langflow/file_utils.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const file_utils_tests = b.addTest(.{
        .root_module = file_utils_test_mod,
    });
    const run_file_utils_tests = b.addRunArtifact(file_utils_tests);
    test_step.dependOn(&run_file_utils_tests.step);
    
    // Day 29 Langflow component modules
    _ = b.addModule("api_connectors", .{
        .root_source_file = b.path("components/langflow/api_connectors.zig"),
    });
    
    _ = b.addModule("utilities", .{
        .root_source_file = b.path("components/langflow/utilities.zig"),
    });
    
    // Tests for API Connectors (Day 29)
    const api_connectors_test_mod = b.createModule(.{
        .root_source_file = b.path("components/langflow/api_connectors.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const api_connectors_tests = b.addTest(.{
        .root_module = api_connectors_test_mod,
    });
    const run_api_connectors_tests = b.addRunArtifact(api_connectors_tests);
    test_step.dependOn(&run_api_connectors_tests.step);
    
    // Tests for Utilities (Day 29)
    const utilities_test_mod = b.createModule(.{
        .root_source_file = b.path("components/langflow/utilities.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const utilities_tests = b.addTest(.{
        .root_module = utilities_test_mod,
    });
    const run_utilities_tests = b.addRunArtifact(utilities_tests);
    test_step.dependOn(&run_utilities_tests.step);
    
    // Day 30 Vector Store modules
    _ = b.addModule("vector_stores", .{
        .root_source_file = b.path("components/langflow/vector_stores.zig"),
    });
    
    // Tests for Vector Stores (Day 30)
    const vector_stores_test_mod = b.createModule(.{
        .root_source_file = b.path("components/langflow/vector_stores.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const vector_stores_tests = b.addTest(.{
        .root_module = vector_stores_test_mod,
    });
    const run_vector_stores_tests = b.addRunArtifact(vector_stores_tests);
    test_step.dependOn(&run_vector_stores_tests.step);

    // HANA nodes module
    const hana_nodes_mod = b.addModule("hana_nodes", .{
        .root_source_file = b.path("nodes/hana/hana_nodes.zig"),
    });
    hana_nodes_mod.addImport("node_types", node_types_mod);
    hana_nodes_mod.addImport("hana_sdk", hana_sdk_mod);

    // Tests for HANA Nodes
    const hana_nodes_test_mod = b.createModule(.{
        .root_source_file = b.path("nodes/hana/hana_nodes.zig"),
        .target = target,
        .optimize = optimize,
    });
    hana_nodes_test_mod.addImport("node_types", node_types_mod);
    hana_nodes_test_mod.addImport("hana_sdk", hana_sdk_mod);

    const hana_nodes_tests = b.addTest(.{
        .root_module = hana_nodes_test_mod,
    });
    const run_hana_nodes_tests = b.addRunArtifact(hana_nodes_tests);
    test_step.dependOn(&run_hana_nodes_tests.step);

    // Day 34 Identity modules
    const http_client_mod = b.addModule("http_client", .{
        .root_source_file = b.path("identity/http_client.zig"),
    });
    
    const keycloak_types_mod = b.addModule("keycloak_types", .{
        .root_source_file = b.path("identity/keycloak_types.zig"),
    });
    
    const keycloak_config_mod = b.addModule("keycloak_config", .{
        .root_source_file = b.path("identity/keycloak_config.zig"),
    });
    
    const keycloak_client_mod = b.addModule("keycloak_client", .{
        .root_source_file = b.path("identity/keycloak_client.zig"),
    });
    keycloak_client_mod.addImport("http_client", http_client_mod);
    keycloak_client_mod.addImport("keycloak_types", keycloak_types_mod);
    keycloak_client_mod.addImport("keycloak_config", keycloak_config_mod);
    
    // Tests for HTTP Client (Day 34)
    const http_client_test_mod = b.createModule(.{
        .root_source_file = b.path("identity/http_client.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const http_client_tests = b.addTest(.{
        .root_module = http_client_test_mod,
    });
    const run_http_client_tests = b.addRunArtifact(http_client_tests);
    test_step.dependOn(&run_http_client_tests.step);
    
    // Tests for Keycloak Types (Day 34)
    const keycloak_types_test_mod = b.createModule(.{
        .root_source_file = b.path("identity/keycloak_types.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const keycloak_types_tests = b.addTest(.{
        .root_module = keycloak_types_test_mod,
    });
    const run_keycloak_types_tests = b.addRunArtifact(keycloak_types_tests);
    test_step.dependOn(&run_keycloak_types_tests.step);
    
    // Tests for Keycloak Config (Day 34)
    const keycloak_config_test_mod = b.createModule(.{
        .root_source_file = b.path("identity/keycloak_config.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const keycloak_config_tests = b.addTest(.{
        .root_module = keycloak_config_test_mod,
    });
    const run_keycloak_config_tests = b.addRunArtifact(keycloak_config_tests);
    test_step.dependOn(&run_keycloak_config_tests.step);
    
    // Tests for Keycloak Client (Day 34)
    const keycloak_client_test_mod = b.createModule(.{
        .root_source_file = b.path("identity/keycloak_client.zig"),
        .target = target,
        .optimize = optimize,
    });
    keycloak_client_test_mod.addImport("http_client", http_client_mod);
    keycloak_client_test_mod.addImport("keycloak_types", keycloak_types_mod);
    keycloak_client_test_mod.addImport("keycloak_config", keycloak_config_mod);
    
    const keycloak_client_tests = b.addTest(.{
        .root_module = keycloak_client_test_mod,
    });
    const run_keycloak_client_tests = b.addRunArtifact(keycloak_client_tests);
    test_step.dependOn(&run_keycloak_client_tests.step);
    
    // Day 38 Keycloak Integration module
    const keycloak_integration_mod = b.addModule("keycloak_integration", .{
        .root_source_file = b.path("identity/keycloak_integration.zig"),
    });
    keycloak_integration_mod.addImport("http_client", http_client_mod);
    keycloak_integration_mod.addImport("keycloak_types", keycloak_types_mod);
    keycloak_integration_mod.addImport("keycloak_config", keycloak_config_mod);
    keycloak_integration_mod.addImport("keycloak_client", keycloak_client_mod);
    
    // Tests for Keycloak Integration (Day 38)
    const keycloak_integration_test_mod = b.createModule(.{
        .root_source_file = b.path("identity/keycloak_integration.zig"),
        .target = target,
        .optimize = optimize,
    });
    keycloak_integration_test_mod.addImport("http_client", http_client_mod);
    keycloak_integration_test_mod.addImport("keycloak_types", keycloak_types_mod);
    keycloak_integration_test_mod.addImport("keycloak_config", keycloak_config_mod);
    keycloak_integration_test_mod.addImport("keycloak_client", keycloak_client_mod);
    
    const keycloak_integration_tests = b.addTest(.{
        .root_module = keycloak_integration_test_mod,
    });
    const run_keycloak_integration_tests = b.addRunArtifact(keycloak_integration_tests);
    test_step.dependOn(&run_keycloak_integration_tests.step);

    // Lean4 nLeanProof Integration Nodes
    const lean_nodes_mod = b.addModule("lean_nodes", .{
        .root_source_file = b.path("nodes/lean/lean_nodes.zig"),
    });
    lean_nodes_mod.addImport("node_types", node_types_mod);

    // Tests for Lean Nodes (nLeanProof Integration)
    const lean_nodes_test_mod = b.createModule(.{
        .root_source_file = b.path("nodes/lean/lean_nodes.zig"),
        .target = target,
        .optimize = optimize,
    });
    lean_nodes_test_mod.addImport("node_types", node_types_mod);

    const lean_nodes_tests = b.addTest(.{
        .root_module = lean_nodes_test_mod,
    });
    const run_lean_nodes_tests = b.addRunArtifact(lean_nodes_tests);
    test_step.dependOn(&run_lean_nodes_tests.step);

    // Phase 2: Case Management modules
    const case_manager_mod = b.addModule("case_manager", .{
        .root_source_file = b.path("case/case_manager.zig"),
    });

    const milestone_mod = b.addModule("milestone", .{
        .root_source_file = b.path("case/milestone.zig"),
    });

    const task_assignment_mod = b.addModule("task_assignment", .{
        .root_source_file = b.path("case/task_assignment.zig"),
    });

    // Tests for Case Manager
    const case_manager_test_mod = b.createModule(.{
        .root_source_file = b.path("case/case_manager.zig"),
        .target = target,
        .optimize = optimize,
    });

    const case_manager_tests = b.addTest(.{
        .root_module = case_manager_test_mod,
    });
    const run_case_manager_tests = b.addRunArtifact(case_manager_tests);
    test_step.dependOn(&run_case_manager_tests.step);

    // Tests for Milestone
    const milestone_test_mod = b.createModule(.{
        .root_source_file = b.path("case/milestone.zig"),
        .target = target,
        .optimize = optimize,
    });

    const milestone_tests = b.addTest(.{
        .root_module = milestone_test_mod,
    });
    const run_milestone_tests = b.addRunArtifact(milestone_tests);
    test_step.dependOn(&run_milestone_tests.step);

    // Tests for Task Assignment
    const task_assignment_test_mod = b.createModule(.{
        .root_source_file = b.path("case/task_assignment.zig"),
        .target = target,
        .optimize = optimize,
    });

    const task_assignment_tests = b.addTest(.{
        .root_module = task_assignment_test_mod,
    });
    const run_task_assignment_tests = b.addRunArtifact(task_assignment_tests);
    test_step.dependOn(&run_task_assignment_tests.step);

    // Suppress unused variable warnings for case modules
    _ = case_manager_mod;
    _ = milestone_mod;
    _ = task_assignment_mod;

    // Example executable
    const example_exe = b.addExecutable(.{
        .name = "nworkflow_example",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic_workflow.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    example_exe.root_module.addImport("petri_net", petri_net_mod);
    b.installArtifact(example_exe);

    const run_cmd = b.addRunArtifact(example_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);

    // Tests for Server Auth
    const auth_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("server/auth.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_auth_tests = b.addRunArtifact(auth_tests);
    test_step.dependOn(&run_auth_tests.step);

    // Tests for HANA Cache
    const hana_cache_test_mod = b.createModule(.{
        .root_source_file = b.path("cache/hana_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    hana_cache_test_mod.addImport("hana_sdk", hana_sdk_mod);

    const hana_cache_tests = b.addTest(.{
        .root_module = hana_cache_test_mod,
    });
    const run_hana_cache_tests = b.addRunArtifact(hana_cache_tests);
    test_step.dependOn(&run_hana_cache_tests.step);

    // Tests for HANA Store
    const hana_store_test_mod = b.createModule(.{
        .root_source_file = b.path("persistence/hana_store.zig"),
        .target = target,
        .optimize = optimize,
    });
    hana_store_test_mod.addImport("hana_sdk", hana_sdk_mod);

    const hana_store_tests = b.addTest(.{
        .root_module = hana_store_test_mod,
    });
    const run_hana_store_tests = b.addRunArtifact(hana_store_tests);
    test_step.dependOn(&run_hana_store_tests.step);

    // Tests for Marquez Client
    const marquez_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("lineage/marquez_client.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_marquez_tests = b.addRunArtifact(marquez_tests);
    test_step.dependOn(&run_marquez_tests.step);

    // Tests for Security Audit
    const audit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("security/audit.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_audit_tests = b.addRunArtifact(audit_tests);
    test_step.dependOn(&run_audit_tests.step);

    // HTTP Server executable
    const server_exe = b.addExecutable(.{
        .name = "nworkflow_server",
        .root_module = b.createModule(.{
            .root_source_file = b.path("server/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(server_exe);

    const server_run_cmd = b.addRunArtifact(server_exe);
    server_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        server_run_cmd.addArgs(args);
    }

    const server_step = b.step("serve", "Run the HTTP server");
    server_step.dependOn(&server_run_cmd.step);

    // Benchmark executable
    const benchmark_exe = b.addExecutable(.{
        .name = "nworkflow_benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/benchmark.zig"),
            .target = target,
            .optimize = .ReleaseFast, // Always optimize benchmarks
        }),
    });
    b.installArtifact(benchmark_exe);

    const benchmark_run_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        benchmark_run_cmd.addArgs(args);
    }

    const bench_step = b.step("bench", "Run performance benchmarks");
    bench_step.dependOn(&benchmark_run_cmd.step);

    // Benchmark tests
    const benchmark_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/benchmark.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_benchmark_tests = b.addRunArtifact(benchmark_tests);
    test_step.dependOn(&run_benchmark_tests.step);

    // Tests for WebSocket
    const websocket_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("server/websocket.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_websocket_tests = b.addRunArtifact(websocket_tests);
    test_step.dependOn(&run_websocket_tests.step);

    // Tests for Integration Tests
    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);
    test_step.dependOn(&run_integration_tests.step);

    // =========================================================================
    // Human Task Management Modules (Phase 1)
    // =========================================================================

    // Human Task Node module
    const task_node_mod = b.addModule("task_node", .{
        .root_source_file = b.path("nodes/human/task_node.zig"),
    });

    // Tests for Human Task Node
    const task_node_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("nodes/human/task_node.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_task_node_tests = b.addRunArtifact(task_node_tests);
    test_step.dependOn(&run_task_node_tests.step);

    // Form Engine module
    const form_engine_mod = b.addModule("form_engine", .{
        .root_source_file = b.path("components/forms/form_engine.zig"),
    });
    _ = form_engine_mod;

    // Tests for Form Engine
    const form_engine_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("components/forms/form_engine.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_form_engine_tests = b.addRunArtifact(form_engine_tests);
    test_step.dependOn(&run_form_engine_tests.step);

    // Approval Node module
    const approval_node_mod = b.addModule("approval_node", .{
        .root_source_file = b.path("nodes/human/approval_node.zig"),
    });
    approval_node_mod.addImport("task_node", task_node_mod);

    // Tests for Approval Node
    const approval_node_test_mod = b.createModule(.{
        .root_source_file = b.path("nodes/human/approval_node.zig"),
        .target = target,
        .optimize = optimize,
    });
    approval_node_test_mod.addImport("task_node", task_node_mod);

    const approval_node_tests = b.addTest(.{
        .root_module = approval_node_test_mod,
    });
    const run_approval_node_tests = b.addRunArtifact(approval_node_tests);
    test_step.dependOn(&run_approval_node_tests.step);

    // CLI executable
    const cli_exe = b.addExecutable(.{
        .name = "nwf",
        .root_module = b.createModule(.{
            .root_source_file = b.path("cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(cli_exe);

    // CLI tests
    const cli_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_cli_tests = b.addRunArtifact(cli_tests);
    test_step.dependOn(&run_cli_tests.step);

    // ============================================
    // IDP (Intelligent Document Processing) Modules
    // ============================================

    // IDP OCR Node module
    const ocr_node_mod = b.addModule("ocr_node", .{
        .root_source_file = b.path("idp/ocr_node.zig"),
    });

    // Tests for OCR Node
    const ocr_node_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("idp/ocr_node.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_ocr_node_tests = b.addRunArtifact(ocr_node_tests);
    test_step.dependOn(&run_ocr_node_tests.step);

    // IDP Classifier Node module
    const classifier_node_mod = b.addModule("classifier_node", .{
        .root_source_file = b.path("idp/classifier_node.zig"),
    });

    // Tests for Classifier Node
    const classifier_node_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("idp/classifier_node.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_classifier_node_tests = b.addRunArtifact(classifier_node_tests);
    test_step.dependOn(&run_classifier_node_tests.step);

    // IDP Extractor Node module
    const extractor_node_mod = b.addModule("extractor_node", .{
        .root_source_file = b.path("idp/extractor_node.zig"),
    });

    // Tests for Extractor Node
    const extractor_node_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("idp/extractor_node.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_extractor_node_tests = b.addRunArtifact(extractor_node_tests);
    test_step.dependOn(&run_extractor_node_tests.step);

    // IDP Validator Node module
    const validator_node_mod = b.addModule("validator_node", .{
        .root_source_file = b.path("idp/validator_node.zig"),
    });

    // Tests for Validator Node
    const validator_node_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("idp/validator_node.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_validator_node_tests = b.addRunArtifact(validator_node_tests);
    test_step.dependOn(&run_validator_node_tests.step);

    // IDP Pipeline Node module (depends on other IDP modules)
    const pipeline_node_mod = b.addModule("pipeline_node", .{
        .root_source_file = b.path("idp/pipeline_node.zig"),
    });
    pipeline_node_mod.addImport("ocr_node", ocr_node_mod);
    pipeline_node_mod.addImport("classifier_node", classifier_node_mod);
    pipeline_node_mod.addImport("extractor_node", extractor_node_mod);
    pipeline_node_mod.addImport("validator_node", validator_node_mod);

    // Tests for Pipeline Node
    const pipeline_node_test_mod = b.createModule(.{
        .root_source_file = b.path("idp/pipeline_node.zig"),
        .target = target,
        .optimize = optimize,
    });
    pipeline_node_test_mod.addImport("ocr_node", ocr_node_mod);
    pipeline_node_test_mod.addImport("classifier_node", classifier_node_mod);
    pipeline_node_test_mod.addImport("extractor_node", extractor_node_mod);
    pipeline_node_test_mod.addImport("validator_node", validator_node_mod);

    const pipeline_node_tests = b.addTest(.{
        .root_module = pipeline_node_test_mod,
    });
    const run_pipeline_node_tests = b.addRunArtifact(pipeline_node_tests);
    test_step.dependOn(&run_pipeline_node_tests.step);

    // ============================================
    // RPA (Robotic Process Automation) Modules
    // ============================================

    // RPA Bot Node module
    const bot_node_mod = b.addModule("bot_node", .{
        .root_source_file = b.path("rpa/bot_node.zig"),
    });

    // Tests for Bot Node
    const bot_node_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("rpa/bot_node.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_bot_node_tests = b.addRunArtifact(bot_node_tests);
    test_step.dependOn(&run_bot_node_tests.step);

    // RPA UI Automation module
    const ui_automation_mod = b.addModule("ui_automation", .{
        .root_source_file = b.path("rpa/ui_automation.zig"),
    });

    // Tests for UI Automation
    const ui_automation_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("rpa/ui_automation.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_ui_automation_tests = b.addRunArtifact(ui_automation_tests);
    test_step.dependOn(&run_ui_automation_tests.step);

    // RPA Recorder module
    const recorder_mod = b.addModule("recorder", .{
        .root_source_file = b.path("rpa/recorder.zig"),
    });

    // Tests for Recorder
    const recorder_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("rpa/recorder.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_recorder_tests = b.addRunArtifact(recorder_tests);
    test_step.dependOn(&run_recorder_tests.step);

    // RPA Scheduler module
    const scheduler_mod = b.addModule("scheduler", .{
        .root_source_file = b.path("rpa/scheduler.zig"),
    });

    // Tests for Scheduler
    const scheduler_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("rpa/scheduler.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_scheduler_tests = b.addRunArtifact(scheduler_tests);
    test_step.dependOn(&run_scheduler_tests.step);

    // Mark modules as used (for future integration)
    _ = bot_node_mod;
    _ = ui_automation_mod;
    _ = recorder_mod;
    _ = scheduler_mod;

    // =========================================================================
    // Process Intelligence Modules
    // =========================================================================

    // Event Collector module
    const event_collector_mod = b.addModule("event_collector", .{
        .root_source_file = b.path("intelligence/event_collector.zig"),
    });

    // Tests for Event Collector
    const event_collector_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("intelligence/event_collector.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_event_collector_tests = b.addRunArtifact(event_collector_tests);
    test_step.dependOn(&run_event_collector_tests.step);

    // Process Miner module
    const process_miner_mod = b.addModule("process_miner", .{
        .root_source_file = b.path("intelligence/process_miner.zig"),
    });
    process_miner_mod.addImport("event_collector", event_collector_mod);

    // Tests for Process Miner
    const process_miner_test_mod = b.createModule(.{
        .root_source_file = b.path("intelligence/process_miner.zig"),
        .target = target,
        .optimize = optimize,
    });
    process_miner_test_mod.addImport("event_collector", event_collector_mod);
    const process_miner_tests = b.addTest(.{
        .root_module = process_miner_test_mod,
    });
    const run_process_miner_tests = b.addRunArtifact(process_miner_tests);
    test_step.dependOn(&run_process_miner_tests.step);

    // Bottleneck Analyzer module
    const bottleneck_analyzer_mod = b.addModule("bottleneck_analyzer", .{
        .root_source_file = b.path("intelligence/bottleneck_analyzer.zig"),
    });
    bottleneck_analyzer_mod.addImport("event_collector", event_collector_mod);

    // Tests for Bottleneck Analyzer
    const bottleneck_analyzer_test_mod = b.createModule(.{
        .root_source_file = b.path("intelligence/bottleneck_analyzer.zig"),
        .target = target,
        .optimize = optimize,
    });
    bottleneck_analyzer_test_mod.addImport("event_collector", event_collector_mod);
    const bottleneck_analyzer_tests = b.addTest(.{
        .root_module = bottleneck_analyzer_test_mod,
    });
    const run_bottleneck_analyzer_tests = b.addRunArtifact(bottleneck_analyzer_tests);
    test_step.dependOn(&run_bottleneck_analyzer_tests.step);

    // KPI Calculator module
    const kpi_calculator_mod = b.addModule("kpi_calculator", .{
        .root_source_file = b.path("intelligence/kpi_calculator.zig"),
    });
    kpi_calculator_mod.addImport("event_collector", event_collector_mod);

    // Tests for KPI Calculator
    const kpi_calculator_test_mod = b.createModule(.{
        .root_source_file = b.path("intelligence/kpi_calculator.zig"),
        .target = target,
        .optimize = optimize,
    });
    kpi_calculator_test_mod.addImport("event_collector", event_collector_mod);
    const kpi_calculator_tests = b.addTest(.{
        .root_module = kpi_calculator_test_mod,
    });
    const run_kpi_calculator_tests = b.addRunArtifact(kpi_calculator_tests);
    test_step.dependOn(&run_kpi_calculator_tests.step);
}
