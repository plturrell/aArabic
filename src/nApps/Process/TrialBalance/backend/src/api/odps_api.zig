//! ============================================================================
//! ODPS REST API - Serve ODPS v4.1 metadata via HTTP endpoints
//! Provides vendor-neutral data product discovery
//! ============================================================================
//!
//! [CODE:file=odps_api.zig]
//! [CODE:module=api]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated,exchange-rates,variances,acdoca-journal-entries]
//!
//! [TABLE:reads=TB_TRIAL_BALANCE,TB_EXCHANGE_RATES,TB_VARIANCE_DETAILS]
//!
//! [API:produces=/api/v1/data-products,/api/v1/data-products/catalog,/api/v1/data-products/quality-report]
//!
//! [RELATION:serves=ODPS:trial-balance-aggregated]
//! [RELATION:calls=CODE:odps_mapper.zig]
//! [RELATION:calls=CODE:odps_quality_service.zig]
//! [RELATION:called_by=CODE:main.zig]
//! [RELATION:called_by=CODE:TrialBalance.controller.js]
//!
//! This API serves ODPS metadata for data product discovery and quality reporting.

const std = @import("std");
const Allocator = std.mem.Allocator;
const odps_mapper = @import("odps_mapper");
const odps_quality_service = @import("odps_quality_service");

/// ODPS API configuration
pub const ODPSAPIConfig = struct {
    odps_directory: []const u8,
    base_path: []const u8, // e.g., "/api/v1/data-products"
    enable_cors: bool,
    
    pub fn default(allocator: Allocator) !ODPSAPIConfig {
        return .{
            .odps_directory = try allocator.dupe(u8, "./models/odps"),
            .base_path = try allocator.dupe(u8, "/api/v1/data-products"),
            .enable_cors = true,
        };
    }
};

/// ODPS API handler
pub const ODPSAPI = struct {
    allocator: Allocator,
    config: ODPSAPIConfig,
    quality_service: *odps_quality_service.ODPSQualityService,
    
    pub fn init(
        allocator: Allocator,
        config: ODPSAPIConfig,
        quality_service: *odps_quality_service.ODPSQualityService,
    ) ODPSAPI {
        return .{
            .allocator = allocator,
            .config = config,
            .quality_service = quality_service,
        };
    }
    
    pub fn deinit(self: *ODPSAPI) void {
        self.allocator.free(self.config.odps_directory);
        self.allocator.free(self.config.base_path);
    }
    
    /// List all available ODPS data products
    /// GET /api/v1/data-products
    pub fn listProducts(self: *ODPSAPI) ![]const u8 {
        var json = std.ArrayList(u8).init(self.allocator);
        defer json.deinit();
        
        const writer = json.writer();
        try writer.writeAll("{\"products\":[");
        
        const products = [_]struct { file: []const u8, category: []const u8 }{
            .{ .file = "primary/acdoca-journal-entries.odps.yaml", .category = "primary" },
            .{ .file = "primary/exchange-rates.odps.yaml", .category = "primary" },
            .{ .file = "primary/trial-balance-aggregated.odps.yaml", .category = "primary" },
            .{ .file = "primary/variances.odps.yaml", .category = "primary" },
            .{ .file = "primary/account-master.odps.yaml", .category = "primary" },
            .{ .file = "metadata/data-lineage.odps.yaml", .category = "metadata" },
            .{ .file = "metadata/dataset-metadata.odps.yaml", .category = "metadata" },
            .{ .file = "operational/checklist-items.odps.yaml", .category = "operational" },
        };
        
        for (products, 0..) |prod, i| {
            if (i > 0) try writer.writeAll(",");
            
            const file_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}",
                .{ self.config.odps_directory, prod.file },
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
            
            try writer.print(
                \\{{"productID":"{s}","name":"{s}","version":"{s}","category":"{s}","qualityScore":{d:.1}}}
            ,
                .{
                    product.product_id,
                    product.name,
                    product.version,
                    prod.category,
                    product.quality_score,
                },
            );
        }
        
        try writer.writeAll("]}");
        return json.toOwnedSlice();
    }
    
    /// Get specific ODPS product by ID
    /// GET /api/v1/data-products/:productID
    pub fn getProduct(self: *ODPSAPI, product_id: []const u8) ![]const u8 {
        // Map product ID to file path
        const file_name = try self.productIDToFileName(product_id);
        defer self.allocator.free(file_name);
        
        const file_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}",
            .{ self.config.odps_directory, file_name },
        );
        defer self.allocator.free(file_path);
        
        // Return raw YAML content
        return try std.fs.cwd().readFileAlloc(self.allocator, file_path, 10 * 1024 * 1024);
    }
    
    /// Get quality report for all products
    /// GET /api/v1/data-products/quality-report
    pub fn getQualityReport(self: *ODPSAPI) ![]const u8 {
        var report = try self.quality_service.generateQualityReport();
        defer report.deinit();
        
        var json = std.ArrayList(u8).init(self.allocator);
        defer json.deinit();
        
        const writer = json.writer();
        try writer.writeAll("{");
        try writer.print("\"generatedAt\":{d},", .{report.generated_at});
        try writer.print("\"averageQuality\":{d:.1},", .{report.getAverageQuality()});
        try writer.writeAll("\"products\":[");
        
        for (report.products.items, 0..) |product, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.print(
                \\{{"name":"{s}","qualityScore":{d:.1}}}
            , .{ product.name, product.quality_score });
        }
        
        try writer.writeAll("]}");
        return json.toOwnedSlice();
    }
    
    /// Get ODPS catalog metadata
    /// GET /api/v1/data-products/catalog
    pub fn getCatalogMetadata(self: *ODPSAPI) ![]const u8 {
        var json = std.ArrayList(u8).init(self.allocator);
        defer json.deinit();
        
        const writer = json.writer();
        try writer.writeAll(
            \\{
            \\  "catalog": {
            \\    "name": "Trial Balance Data Products",
            \\    "version": "1.0.0",
            \\    "specification": "ODPS v4.1",
            \\    "organization": "nApps Process - Trial Balance Team",
            \\    "productCount": 8,
            \\    "categories": ["primary", "metadata", "operational"],
            \\    "standards": ["IFRS", "US-GAAP", "ISO-4217", "IAS-21"],
            \\    "endpoints": {
            \\      "list": "/api/v1/data-products",
            \\      "get": "/api/v1/data-products/:productID",
            \\      "quality": "/api/v1/data-products/quality-report",
            \\      "catalog": "/api/v1/data-products/catalog"
            \\    }
            \\  }
            \\}
        );
        
        return json.toOwnedSlice();
    }
    
    /// Map product ID to file path
    fn productIDToFileName(self: *ODPSAPI, product_id: []const u8) ![]const u8 {
        if (std.mem.indexOf(u8, product_id, "acdoca")) |_| {
            return try self.allocator.dupe(u8, "primary/acdoca-journal-entries.odps.yaml");
        } else if (std.mem.indexOf(u8, product_id, "exchange-rates")) |_| {
            return try self.allocator.dupe(u8, "primary/exchange-rates.odps.yaml");
        } else if (std.mem.indexOf(u8, product_id, "trial-balance")) |_| {
            return try self.allocator.dupe(u8, "primary/trial-balance-aggregated.odps.yaml");
        } else if (std.mem.indexOf(u8, product_id, "variances")) |_| {
            return try self.allocator.dupe(u8, "primary/variances.odps.yaml");
        } else if (std.mem.indexOf(u8, product_id, "account-master")) |_| {
            return try self.allocator.dupe(u8, "primary/account-master.odps.yaml");
        } else if (std.mem.indexOf(u8, product_id, "data-lineage")) |_| {
            return try self.allocator.dupe(u8, "metadata/data-lineage.odps.yaml");
        } else if (std.mem.indexOf(u8, product_id, "dataset-metadata")) |_| {
            return try self.allocator.dupe(u8, "metadata/dataset-metadata.odps.yaml");
        } else if (std.mem.indexOf(u8, product_id, "checklist")) |_| {
            return try self.allocator.dupe(u8, "operational/checklist-items.odps.yaml");
        }
        return error.ProductNotFound;
    }
};

// Tests
test "ODPS API list products" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const config = try ODPSAPIConfig.default(allocator);
    const qconfig = try odps_quality_service.QualityUpdateConfig.default(allocator);
    var qservice = odps_quality_service.ODPSQualityService.init(allocator, qconfig);
    defer qservice.deinit();
    
    var api = ODPSAPI.init(allocator, config, &qservice);
    defer api.deinit();
    
    // This would require actual ODPS files to exist
    // const json = try api.listProducts();
    // defer allocator.free(json);
    // try testing.expect(std.mem.indexOf(u8, json, "products") != null);
}