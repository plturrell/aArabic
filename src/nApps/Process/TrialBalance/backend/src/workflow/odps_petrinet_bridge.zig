//! ============================================================================
//! ODPS-Petri Net Bridge - Integrate workflow metadata with ODPS data products
//! Links checklist items to ODPS operational metadata and tracks data quality
//! ============================================================================
//!
//! [CODE:file=odps_petrinet_bridge.zig]
//! [CODE:module=workflow]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-process,checklist-items]
//!
//! [DOI:controls=MKR-CHK-001,MKR-CHK-002]
//!
//! [PETRI:process=TB_PROCESS_petrinet.pnml]
//! [PETRI:stages=S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12,S13]
//!
//! [TABLE:reads=TB_WORKFLOW_STATUS,TB_CHECKLIST_ITEMS]
//! [TABLE:writes=TB_WORKFLOW_STATUS,TB_CHECKLIST_ITEMS]
//!
//! [RELATION:orchestrates=CODE:balance_engine.zig]
//! [RELATION:orchestrates=CODE:fx_converter.zig]
//! [RELATION:calls=CODE:odps_mapper.zig]
//! [RELATION:called_by=CODE:main.zig]
//!
//! This module bridges ODPS data products with Petri net workflow execution,
//! ensuring data quality gates are met before workflow stages proceed.

const std = @import("std");
const Allocator = std.mem.Allocator;
const odps_mapper = @import("odps_mapper");

/// Workflow stage status tracked in ODPS
pub const WorkflowStageStatus = enum {
    pending,
    in_progress,
    complete,
    blocked,

    pub fn toString(self: WorkflowStageStatus) []const u8 {
        return switch (self) {
            .pending => "Pending",
            .in_progress => "InProgress",
            .complete => "Complete",
            .blocked => "Blocked",
        };
    }
};

/// Workflow stage with ODPS metadata linkage
pub const ODPSWorkflowStage = struct {
    stage_id: []const u8,
    title: []const u8,
    status: WorkflowStageStatus,
    data_product_ids: []const []const u8, // ODPS product IDs affected by this stage
    quality_requirements: f64, // Minimum quality score required
    assigned_to: ?[]const u8,
    completed_at: ?i64,
    
    pub fn init(
        allocator: Allocator,
        stage_id: []const u8,
        title: []const u8,
        product_ids: []const []const u8,
        min_quality: f64,
    ) !ODPSWorkflowStage {
        return .{
            .stage_id = try allocator.dupe(u8, stage_id),
            .title = try allocator.dupe(u8, title),
            .status = .pending,
            .data_product_ids = try allocator.dupe([]const u8, product_ids),
            .quality_requirements = min_quality,
            .assigned_to = null,
            .completed_at = null,
        };
    }
};

/// Petri net workflow with ODPS integration
pub const ODPSWorkflow = struct {
    allocator: Allocator,
    stages: std.ArrayList(ODPSWorkflowStage),
    odps_dir: []const u8,
    
    pub fn init(allocator: Allocator, odps_directory: []const u8) !ODPSWorkflow {
        return .{
            .allocator = allocator,
            .stages = std.ArrayList(ODPSWorkflowStage){},
            .odps_dir = try allocator.dupe(u8, odps_directory),
        };
    }
    
    pub fn deinit(self: *ODPSWorkflow) void {
        for (self.stages.items) |stage| {
            self.allocator.free(stage.stage_id);
            self.allocator.free(stage.title);
            self.allocator.free(stage.data_product_ids);
            if (stage.assigned_to) |assigned| {
                self.allocator.free(assigned);
            }
        }
        self.stages.deinit(self.allocator);
        self.allocator.free(self.odps_dir);
    }
    
    /// Initialize 13-stage IFRS workflow with ODPS product mappings
    /// Aligned with ODPS trial-balance-petrinet.odps.yaml specification
    pub fn initializeIFRSWorkflow(self: *ODPSWorkflow) !void {
        // Stage 1: Data Extraction (T_EXTRACT)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S01",
            "Data Extraction",
            &[_][]const u8{ "urn:uuid:acdoca-journal-entries-v1" },
            95.0, // From ODPS primary product quality target
        ));
        
        // Stage 2: GCOA Mapping (T_MAP_GCOA)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S02",
            "GCOA Mapping",
            &[_][]const u8{ "urn:uuid:account-master-v1" },
            99.0, // From ODPS: TB003, TB005 validation
        ));
        
        // Stage 3: Currency Conversion (T_CONVERT_FX)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S03",
            "Currency Conversion",
            &[_][]const u8{ "urn:uuid:exchange-rates-v1" },
            98.0, // From ODPS: FX005, FX006 validation
        ));
        
        // Stage 4: Trial Balance Aggregation (T_AGGREGATE)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S04",
            "Trial Balance Aggregation",
            &[_][]const u8{ "urn:uuid:trial-balance-aggregated-v1", "urn:uuid:data-lineage-v1" },
            92.0, // From ODPS: TB001, TB002, TB004, TB006 validation
        ));
        
        // Stage 5: Variance Calculation (T_CALC_VARIANCE)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S05",
            "Variance Calculation",
            &[_][]const u8{ "urn:uuid:variances-v1" },
            90.0, // From ODPS: VAR001, VAR002 validation
        ));
        
        // Stage 6: Threshold Application (T_APPLY_THRESH)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S06",
            "Threshold Application",
            &[_][]const u8{ "urn:uuid:variances-v1" },
            90.0, // From ODPS: VAR003, VAR004 validation (MKR-CHK-001)
        ));
        
        // Stage 7: Commentary Initiation (T_INIT_COMMENTARY)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S07",
            "Commentary Initiation",
            &[_][]const u8{ "urn:uuid:checklist-items-v1" },
            85.0,
        ));
        
        // Stage 8: Commentary Collection (T_COLLECT_COMMENTARY)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S08",
            "Commentary Collection",
            &[_][]const u8{ "urn:uuid:checklist-items-v1" },
            85.0,
        ));
        
        // Stage 9: Driver Analysis (T_IDENTIFY_DRIVERS)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S09",
            "Driver Analysis",
            &[_][]const u8{ "urn:uuid:variances-v1" },
            90.0, // From ODPS: VAR008 validation (REQ-THRESH-004)
        ));
        
        // Stage 10: Coverage Verification (T_VERIFY_COVERAGE)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S10",
            "Coverage Verification",
            &[_][]const u8{ "urn:uuid:variances-v1", "urn:uuid:checklist-items-v1" },
            90.0, // From ODPS: VAR005, VAR006 validation (90% coverage)
        ));
        
        // Stage 11: Maker Review (T_MAKER_REVIEW)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S11",
            "Maker Review",
            &[_][]const u8{ "urn:uuid:checklist-items-v1" },
            85.0,
        ));
        
        // Stage 12: Checker Review (T_CHECKER_REVIEW)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S12",
            "Checker Review",
            &[_][]const u8{ "urn:uuid:checklist-items-v1", "urn:uuid:trial-balance-aggregated-v1" },
            92.0, // From ODPS: TB001, TB002 validation (VAL-001)
        ));
        
        // Stage 13: Submission & Archive (T_SUBMIT, T_ARCHIVE)
        try self.stages.append(self.allocator, try ODPSWorkflowStage.init(
            self.allocator,
            "S13",
            "Submission & Archive",
            &[_][]const u8{ "urn:uuid:workflow-execution-log-v1", "urn:uuid:data-lineage-v1" },
            98.0,
        ));
    }
    
    /// Check if stage can proceed based on ODPS product quality
    pub fn canProceed(self: *ODPSWorkflow, stage_index: usize) !bool {
        if (stage_index >= self.stages.items.len) return false;
        
        const stage = &self.stages.items[stage_index];
        
        // Check all required data products meet quality requirements
        for (stage.data_product_ids) |product_id| {
            const quality = try self.getProductQuality(product_id);
            if (quality < stage.quality_requirements) {
                return false;
            }
        }
        
        return true;
    }
    
    /// Get current quality score for an ODPS data product
    fn getProductQuality(self: *ODPSWorkflow, product_id: []const u8) !f64 {
        // Extract product name from URN
        var product_name: []const u8 = "";
        if (std.mem.indexOf(u8, product_id, "acdoca")) |_| {
            product_name = "acdoca-journal-entries";
        } else if (std.mem.indexOf(u8, product_id, "exchange-rates")) |_| {
            product_name = "exchange-rates";
        } else if (std.mem.indexOf(u8, product_id, "trial-balance")) |_| {
            product_name = "trial-balance-aggregated";
        } else if (std.mem.indexOf(u8, product_id, "variances")) |_| {
            product_name = "variances";
        } else {
            return 100.0; // Default if not found
        }
        
        // Build file path
        const file_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/primary/{s}.odps.yaml",
            .{ self.odps_dir, product_name },
        );
        defer self.allocator.free(file_path);
        
        // Load ODPS and get quality score
        const product = try odps_mapper.loadODPS(self.allocator, file_path);
        defer {
            self.allocator.free(product.product_id);
            self.allocator.free(product.name);
            self.allocator.free(product.description);
            self.allocator.free(product.version);
            self.allocator.free(product.status);
            if (product.ord_ref) |ref| self.allocator.free(ref);
            if (product.csn_ref) |ref| self.allocator.free(ref);
        }
        
        return product.quality_score;
    }
    
    /// Complete a workflow stage and update ODPS checklist metadata
    pub fn completeStage(self: *ODPSWorkflow, stage_index: usize) !void {
        if (stage_index >= self.stages.items.len) return error.InvalidStageIndex;
        
        var stage = &self.stages.items[stage_index];
        stage.status = .complete;
        stage.completed_at = std.time.timestamp();
        
        // Update ODPS checklist metadata
        const checklist_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/operational/checklist-items.odps.yaml",
            .{self.odps_dir},
        );
        defer self.allocator.free(checklist_path);
        
        // TODO: Update checklist ODPS file with completion status
        // This would modify the YAML to mark stage as complete
    }
    
    /// Get workflow progress (percentage complete)
    pub fn getProgress(self: *const ODPSWorkflow) f64 {
        var completed: usize = 0;
        for (self.stages.items) |stage| {
            if (stage.status == .complete) {
                completed += 1;
            }
        }
        return @as(f64, @floatFromInt(completed)) / @as(f64, @floatFromInt(self.stages.items.len)) * 100.0;
    }
};

// Tests
test "ODPS workflow initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var workflow = try ODPSWorkflow.init(allocator, "./models/odps");
    defer workflow.deinit();
    
    try workflow.initializeIFRSWorkflow();
    
    try testing.expectEqual(@as(usize, 13), workflow.stages.items.len);
    try testing.expectEqualStrings("S01", workflow.stages.items[0].stage_id);
    try testing.expectEqualStrings("S13", workflow.stages.items[12].stage_id);
}

test "workflow progress calculation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var workflow = try ODPSWorkflow.init(allocator, "./models/odps");
    defer workflow.deinit();
    
    try workflow.initializeIFRSWorkflow();
    
    // Initially 0%
    try testing.expectEqual(@as(f64, 0.0), workflow.getProgress());
    
    // Complete first stage
    workflow.stages.items[0].status = .complete;
    const progress = workflow.getProgress();
    try testing.expect(progress > 7.0 and progress < 8.0); // ~7.69%
}
