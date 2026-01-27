//! ============================================================================
//! ODPS Quality Update Service
//! Automatically updates ODPS quality scores based on runtime data quality metrics
//! ============================================================================
//!
//! [CODE:file=odps_quality_service.zig]
//! [CODE:module=services]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated,exchange-rates,acdoca-journal-entries,variances]
//!
//! [TABLE:reads=TB_TRIAL_BALANCE,TB_EXCHANGE_RATES,TB_JOURNAL_ENTRIES]
//! [TABLE:writes=TB_DATA_QUALITY_SCORES]
//!
//! [API:produces=/api/v1/data-products/quality-report]
//!
//! [RELATION:updates=ODPS:trial-balance-aggregated]
//! [RELATION:updates=ODPS:exchange-rates]
//! [RELATION:calls=CODE:odps_mapper.zig]
//! [RELATION:calls=CODE:data_quality.zig]
//! [RELATION:called_by=CODE:odps_api.zig]
//! [RELATION:called_by=CODE:main.zig]
//!
//! This service monitors data quality and updates ODPS quality scores in real-time.

const std = @import("std");
const Allocator = std.mem.Allocator;
const odps_mapper = @import("odps_mapper");
const data_quality = @import("data_quality");
const acdoca_table = @import("acdoca_table");

/// Quality update job configuration
pub const QualityUpdateConfig = struct {
    odps_directory: []const u8,
    update_interval_seconds: u64, // How often to update
    min_quality_threshold: f64, // Alert if below this
    
    pub fn default(allocator: Allocator) !QualityUpdateConfig {
        return .{
            .odps_directory = try allocator.dupe(u8, "./models/odps"),
            .update_interval_seconds = 300, // 5 minutes
            .min_quality_threshold = 85.0,
        };
    }
};

/// Quality update service
pub const ODPSQualityService = struct {
    allocator: Allocator,
    config: QualityUpdateConfig,
    last_update: i64,
    alert_callback: ?*const fn ([]const u8, f64) void,
    
    pub fn init(allocator: Allocator, config: QualityUpdateConfig) ODPSQualityService {
        return .{
            .allocator = allocator,
            .config = config,
            .last_update = std.time.timestamp(),
            .alert_callback = null,
        };
    }
    
    pub fn deinit(self: *ODPSQualityService) void {
        self.allocator.free(self.config.odps_directory);
    }
    
    /// Set callback for quality alerts
    pub fn setAlertCallback(self: *ODPSQualityService, callback: *const fn ([]const u8, f64) void) void {
        self.alert_callback = callback;
    }
    
    /// Update ACDOCA product quality from ACDOCATable
    pub fn updateACDOCAQuality(self: *ODPSQualityService, table: *const acdoca_table.ACDOCATable) !void {
        const quality_score = table.getQualityScore();
        const stats = table.getStatistics();
        
        // Build file path
        const file_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/primary/acdoca-journal-entries.odps.yaml",
            .{self.config.odps_directory},
        );
        defer self.allocator.free(file_path);
        
        // Update ODPS file
        try odps_mapper.updateODPSQuality(
            self.allocator,
            file_path,
            quality_score,
            stats.verified_entries,
            stats.total_entries,
        );
        
        // Alert if below threshold
        if (quality_score < self.config.min_quality_threshold) {
            if (self.alert_callback) |callback| {
                callback("ACDOCA Journal Entries", quality_score);
            }
        }
        
        self.last_update = std.time.timestamp();
    }
    
    /// Update all data products' quality scores
    pub fn updateAllProducts(self: *ODPSQualityService, runtime_metrics: *const RuntimeMetrics) !void {
        // Update ACDOCA
        if (runtime_metrics.acdoca_quality) |quality| {
            const file_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/primary/acdoca-journal-entries.odps.yaml",
                .{self.config.odps_directory},
            );
            defer self.allocator.free(file_path);
            
            try odps_mapper.updateODPSQuality(
                self.allocator,
                file_path,
                quality.score,
                quality.verified,
                quality.total,
            );
        }
        
        // Update Exchange Rates
        if (runtime_metrics.fx_quality) |quality| {
            const file_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/primary/exchange-rates.odps.yaml",
                .{self.config.odps_directory},
            );
            defer self.allocator.free(file_path);
            
            try odps_mapper.updateODPSQuality(
                self.allocator,
                file_path,
                quality.score,
                quality.verified,
                quality.total,
            );
        }
        
        // Update Trial Balance
        if (runtime_metrics.tb_quality) |quality| {
            const file_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/primary/trial-balance-aggregated.odps.yaml",
                .{self.config.odps_directory},
            );
            defer self.allocator.free(file_path);
            
            try odps_mapper.updateODPSQuality(
                self.allocator,
                file_path,
                quality.score,
                quality.verified,
                quality.total,
            );
        }
        
        self.last_update = std.time.timestamp();
    }
    
    /// Check if update is needed based on interval
    pub fn needsUpdate(self: *const ODPSQualityService) bool {
        const now = std.time.timestamp();
        const elapsed = now - self.last_update;
        return @as(u64, @intCast(elapsed)) >= self.config.update_interval_seconds;
    }
    
    /// Generate quality report for all ODPS products
    pub fn generateQualityReport(self: *ODPSQualityService) !QualityReport {
        var report = QualityReport.init(self.allocator);
        
        // Scan all ODPS files
        const products = [_][]const u8{
            "primary/acdoca-journal-entries.odps.yaml",
            "primary/exchange-rates.odps.yaml",
            "primary/trial-balance-aggregated.odps.yaml",
            "primary/variances.odps.yaml",
            "primary/account-master.odps.yaml",
        };
        
        for (products) |product_file| {
            const file_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}",
                .{ self.config.odps_directory, product_file },
            );
            defer self.allocator.free(file_path);
            
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
            
            try report.addProduct(product.name, product.quality_score);
        }
        
        return report;
    }
};

/// Runtime metrics from data processing
pub const RuntimeMetrics = struct {
    acdoca_quality: ?QualityMetric,
    fx_quality: ?QualityMetric,
    tb_quality: ?QualityMetric,
    variance_quality: ?QualityMetric,
    
    pub const QualityMetric = struct {
        score: f64,
        verified: usize,
        total: usize,
    };
};

/// Quality report aggregating all products
pub const QualityReport = struct {
    allocator: Allocator,
    products: std.ArrayList(ProductQuality),
    generated_at: i64,
    
    const ProductQuality = struct {
        name: []const u8,
        quality_score: f64,
    };
    
    pub fn init(allocator: Allocator) QualityReport {
        return .{
            .allocator = allocator,
            .products = std.ArrayList(ProductQuality){},
            .generated_at = std.time.timestamp(),
        };
    }
    
    pub fn deinit(self: *QualityReport) void {
        for (self.products.items) |product| {
            self.allocator.free(product.name);
        }
        self.products.deinit(self.allocator);
    }
    
    pub fn addProduct(self: *QualityReport, name: []const u8, quality: f64) !void {
        try self.products.append(self.allocator, .{
            .name = try self.allocator.dupe(u8, name),
            .quality_score = quality,
        });
    }
    
    pub fn getAverageQuality(self: *const QualityReport) f64 {
        if (self.products.items.len == 0) return 0.0;
        
        var total: f64 = 0.0;
        for (self.products.items) |product| {
            total += product.quality_score;
        }
        return total / @as(f64, @floatFromInt(self.products.items.len));
    }
};

// Tests
test "quality service initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const config = try QualityUpdateConfig.default(allocator);
    var service = ODPSQualityService.init(allocator, config);
    defer service.deinit();
    
    try testing.expectEqual(@as(u64, 300), service.config.update_interval_seconds);
    try testing.expectEqual(@as(f64, 85.0), service.config.min_quality_threshold);
}

test "quality report generation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var report = QualityReport.init(allocator);
    defer report.deinit();
    
    try report.addProduct("Product A", 95.0);
    try report.addProduct("Product B", 90.0);
    try report.addProduct("Product C", 85.0);
    
    const avg = report.getAverageQuality();
    try testing.expectEqual(@as(f64, 90.0), avg);
}