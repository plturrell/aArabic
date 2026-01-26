//! Multi-Category Routing Analytics Module
//! Analyzes model selection metrics to evaluate routing effectiveness
//!
//! Features:
//! - Load metrics from CSV files
//! - Analyze category-level statistics
//! - Track multi-category model utilization
//! - Generate effectiveness reports
//! - Export to JSON or Markdown

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Selection metric record
pub const SelectionMetric = struct {
    timestamp: []const u8,
    task_category: []const u8,
    selected_model: []const u8,
    primary_category: bool,
    confidence_score: f32,
    final_score: f64,
    gpu_id: u32,
    selection_duration_ms: i64,
    
    pub fn deinit(self: *SelectionMetric, allocator: Allocator) void {
        allocator.free(self.timestamp);
        allocator.free(self.task_category);
        allocator.free(self.selected_model);
    }
};

/// Category statistics
pub const CategoryStats = struct {
    total_requests: usize,
    models_used: std.StringHashMap(void),
    avg_score: f64,
    avg_confidence: f32,
    avg_duration_ms: f64,
    
    pub fn init(allocator: Allocator) CategoryStats {
        return .{
            .total_requests = 0,
            .models_used = std.StringHashMap(void).init(allocator),
            .avg_score = 0.0,
            .avg_confidence = 0.0,
            .avg_duration_ms = 0.0,
        };
    }
    
    pub fn deinit(self: *CategoryStats) void {
        var it = self.models_used.iterator();
        while (it.next()) |entry| {
            self.models_used.allocator.free(entry.key_ptr.*);
        }
        self.models_used.deinit();
    }
};

/// Model statistics
pub const ModelStats = struct {
    total_requests: usize,
    categories_served: std.StringHashMap(void),
    avg_score: f64,
    primary_category_count: usize,
    secondary_category_count: usize,
    
    pub fn init(allocator: Allocator) ModelStats {
        return .{
            .total_requests = 0,
            .categories_served = std.StringHashMap(void).init(allocator),
            .avg_score = 0.0,
            .primary_category_count = 0,
            .secondary_category_count = 0,
        };
    }
    
    pub fn deinit(self: *ModelStats) void {
        var it = self.categories_served.iterator();
        while (it.next()) |entry| {
            self.categories_served.allocator.free(entry.key_ptr.*);
        }
        self.categories_served.deinit();
    }
};

/// Overall effectiveness metrics
pub const EffectivenessMetrics = struct {
    total_requests: usize,
    unique_models_used: usize,
    multi_category_models: usize,
    secondary_category_usage: usize,
    secondary_category_percentage: f64,
    avg_secondary_confidence: f32,
    gpu_distribution: std.AutoHashMap(u32, usize),
    avg_selection_duration_ms: f64,
    
    pub fn deinit(self: *EffectivenessMetrics) void {
        self.gpu_distribution.deinit();
    }
};

/// Multi-Category Analytics Engine
pub const Analytics = struct {
    allocator: Allocator,
    metrics_file: []const u8,
    metrics: std.ArrayList(SelectionMetric),
    category_stats: std.StringHashMap(CategoryStats),
    model_stats: std.StringHashMap(ModelStats),
    
    pub fn init(allocator: Allocator, metrics_file: []const u8) !*Analytics {
        const self = try allocator.create(Analytics);
        self.* = .{
            .allocator = allocator,
            .metrics_file = try allocator.dupe(u8, metrics_file),
            .metrics = try std.ArrayList(SelectionMetric).initCapacity(allocator, 0),
            .category_stats = std.StringHashMap(CategoryStats).init(allocator),
            .model_stats = std.StringHashMap(ModelStats).init(allocator),
        };
        return self;
    }
    
    pub fn deinit(self: *Analytics) void {
        for (self.metrics.items) |*metric| {
            metric.deinit(self.allocator);
        }
        self.metrics.deinit(self.allocator);
        
        var cat_it = self.category_stats.iterator();
        while (cat_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.category_stats.deinit();
        
        var model_it = self.model_stats.iterator();
        while (model_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.model_stats.deinit();
        
        self.allocator.free(self.metrics_file);
        self.allocator.destroy(self);
    }
    
    /// Load metrics from CSV file
    pub fn loadMetrics(self: *Analytics) !void {
        const file = try std.fs.cwd().openFile(self.metrics_file, .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(self.allocator, 100 * 1024 * 1024); // 100MB max
        defer self.allocator.free(content);
        
        var lines = std.mem.splitScalar(u8, content, '\n');
        
        // Skip header
        _ = lines.next();
        
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            
            const metric = try self.parseCSVLine(line);
            try self.metrics.append(self.allocator, metric);
        }
    }
    
    fn parseCSVLine(self: *Analytics, line: []const u8) !SelectionMetric {
        var fields = std.mem.splitScalar(u8, line, ',');
        
        const timestamp = fields.next() orelse return error.InvalidCSV;
        const task_category = fields.next() orelse return error.InvalidCSV;
        const selected_model = fields.next() orelse return error.InvalidCSV;
        const primary_category_str = fields.next() orelse return error.InvalidCSV;
        const confidence_score_str = fields.next() orelse return error.InvalidCSV;
        const final_score_str = fields.next() orelse return error.InvalidCSV;
        const gpu_id_str = fields.next() orelse return error.InvalidCSV;
        const duration_str = fields.next() orelse return error.InvalidCSV;
        
        return SelectionMetric{
            .timestamp = try self.allocator.dupe(u8, timestamp),
            .task_category = try self.allocator.dupe(u8, task_category),
            .selected_model = try self.allocator.dupe(u8, selected_model),
            .primary_category = std.mem.eql(u8, primary_category_str, "true"),
            .confidence_score = try std.fmt.parseFloat(f32, confidence_score_str),
            .final_score = try std.fmt.parseFloat(f64, final_score_str),
            .gpu_id = try std.fmt.parseInt(u32, gpu_id_str, 10),
            .selection_duration_ms = try std.fmt.parseInt(i64, duration_str, 10),
        };
    }
    
    /// Analyze category-level statistics
    pub fn analyzeCategories(self: *Analytics) !void {
        for (self.metrics.items) |metric| {
            const cat = metric.task_category;
            
            const result = try self.category_stats.getOrPut(cat);
            if (!result.found_existing) {
                result.key_ptr.* = try self.allocator.dupe(u8, cat);
                result.value_ptr.* = CategoryStats.init(self.allocator);
            }
            
            var stats = result.value_ptr;
            stats.total_requests += 1;
            stats.avg_score += metric.final_score;
            stats.avg_confidence += metric.confidence_score;
            stats.avg_duration_ms += @as(f64, @floatFromInt(metric.selection_duration_ms));
            
            // Track model usage
            const model_result = try stats.models_used.getOrPut(metric.selected_model);
            if (!model_result.found_existing) {
                model_result.key_ptr.* = try self.allocator.dupe(u8, metric.selected_model);
            }
        }
        
        // Calculate averages
        var cat_it = self.category_stats.iterator();
        while (cat_it.next()) |entry| {
            const count = @as(f64, @floatFromInt(entry.value_ptr.total_requests));
            if (count > 0) {
                entry.value_ptr.avg_score /= count;
                entry.value_ptr.avg_confidence /= @as(f32, @floatCast(count));
                entry.value_ptr.avg_duration_ms /= count;
            }
        }
    }
    
    /// Analyze model-level statistics
    pub fn analyzeModels(self: *Analytics) !void {
        for (self.metrics.items) |metric| {
            const model = metric.selected_model;
            
            const result = try self.model_stats.getOrPut(model);
            if (!result.found_existing) {
                result.key_ptr.* = try self.allocator.dupe(u8, model);
                result.value_ptr.* = ModelStats.init(self.allocator);
            }
            
            var stats = result.value_ptr;
            stats.total_requests += 1;
            stats.avg_score += metric.final_score;
            
            if (metric.primary_category) {
                stats.primary_category_count += 1;
            } else {
                stats.secondary_category_count += 1;
            }
            
            // Track category usage
            const cat_result = try stats.categories_served.getOrPut(metric.task_category);
            if (!cat_result.found_existing) {
                cat_result.key_ptr.* = try self.allocator.dupe(u8, metric.task_category);
            }
        }
        
        // Calculate averages
        var model_it = self.model_stats.iterator();
        while (model_it.next()) |entry| {
            const count = @as(f64, @floatFromInt(entry.value_ptr.total_requests));
            if (count > 0) {
                entry.value_ptr.avg_score /= count;
            }
        }
    }
    
    /// Calculate overall effectiveness
    pub fn calculateEffectiveness(self: *Analytics) !EffectivenessMetrics {
        const total_requests = self.metrics.items.len;
        if (total_requests == 0) {
            return EffectivenessMetrics{
                .total_requests = 0,
                .unique_models_used = 0,
                .multi_category_models = 0,
                .secondary_category_usage = 0,
                .secondary_category_percentage = 0.0,
                .avg_secondary_confidence = 0.0,
                .gpu_distribution = std.AutoHashMap(u32, usize).init(self.allocator),
                .avg_selection_duration_ms = 0.0,
            };
        }
        
        // Count multi-category models
        var multi_cat_models: usize = 0;
        var model_it = self.model_stats.iterator();
        while (model_it.next()) |entry| {
            if (entry.value_ptr.categories_served.count() > 1) {
                multi_cat_models += 1;
            }
        }
        
        // Count secondary usage
        var secondary_usage: usize = 0;
        var secondary_confidence_sum: f32 = 0.0;
        for (self.metrics.items) |metric| {
            if (!metric.primary_category) {
                secondary_usage += 1;
                secondary_confidence_sum += metric.confidence_score;
            }
        }
        
        const secondary_percentage = (@as(f64, @floatFromInt(secondary_usage)) / 
                                     @as(f64, @floatFromInt(total_requests))) * 100.0;
        
        const avg_secondary_confidence = if (secondary_usage > 0)
            secondary_confidence_sum / @as(f32, @floatFromInt(secondary_usage))
        else
            0.0;
        
        // GPU distribution
        var gpu_dist = std.AutoHashMap(u32, usize).init(self.allocator);
        for (self.metrics.items) |metric| {
            const result = try gpu_dist.getOrPut(metric.gpu_id);
            if (!result.found_existing) {
                result.value_ptr.* = 0;
            }
            result.value_ptr.* += 1;
        }
        
        // Average duration
        var total_duration: i64 = 0;
        for (self.metrics.items) |metric| {
            total_duration += metric.selection_duration_ms;
        }
        const avg_duration = @as(f64, @floatFromInt(total_duration)) / 
                            @as(f64, @floatFromInt(total_requests));
        
        return EffectivenessMetrics{
            .total_requests = total_requests,
            .unique_models_used = self.model_stats.count(),
            .multi_category_models = multi_cat_models,
            .secondary_category_usage = secondary_usage,
            .secondary_category_percentage = secondary_percentage,
            .avg_secondary_confidence = avg_secondary_confidence,
            .gpu_distribution = gpu_dist,
            .avg_selection_duration_ms = avg_duration,
        };
    }
    
    /// Generate markdown report
    pub fn generateMarkdownReport(self: *Analytics, writer: anytype) !void {
        var effectiveness = try self.calculateEffectiveness();
        defer effectiveness.deinit();
        
        try writer.print("# Multi-Category Routing Analytics Report\n\n", .{});
        try writer.print("**Generated:** {s}\n", .{try self.getTimestamp()});
        try writer.print("**Metrics File:** {s}\n", .{self.metrics_file});
        try writer.print("**Total Requests:** {d}\n\n", .{effectiveness.total_requests});
        
        try writer.print("## Overall Effectiveness\n\n", .{});
        try writer.print("- **Unique Models Used:** {d}\n", .{effectiveness.unique_models_used});
        try writer.print("- **Multi-Category Models:** {d}\n", .{effectiveness.multi_category_models});
        try writer.print("- **Secondary Category Usage:** {d} ({d:.1}%)\n", 
            .{effectiveness.secondary_category_usage, effectiveness.secondary_category_percentage});
        try writer.print("- **Avg Secondary Confidence:** {d:.2}\n", .{effectiveness.avg_secondary_confidence});
        try writer.print("- **Avg Selection Duration:** {d:.1}ms\n\n", .{effectiveness.avg_selection_duration_ms});
        
        try writer.print("### GPU Distribution\n\n", .{});
        var gpu_it = effectiveness.gpu_distribution.iterator();
        while (gpu_it.next()) |entry| {
            const percentage = (@as(f64, @floatFromInt(entry.value_ptr.*)) / 
                               @as(f64, @floatFromInt(effectiveness.total_requests))) * 100.0;
            try writer.print("- GPU {d}: {d} requests ({d:.1}%)\n", 
                .{entry.key_ptr.*, entry.value_ptr.*, percentage});
        }
        try writer.print("\n", .{});
        
        try writer.print("## Performance by Category\n\n", .{});
        try writer.print("| Category | Requests | Models Used | Avg Score | Avg Confidence | Avg Duration |\n", .{});
        try writer.print("|----------|----------|-------------|-----------|----------------|-------------|\n", .{});
        
        var cat_it = self.category_stats.iterator();
        while (cat_it.next()) |entry| {
            try writer.print("| {s} | {d} | {d} | {d:.2} | {d:.2} | {d:.1}ms |\n",
                .{entry.key_ptr.*, entry.value_ptr.total_requests, 
                  entry.value_ptr.models_used.count(), entry.value_ptr.avg_score,
                  entry.value_ptr.avg_confidence, entry.value_ptr.avg_duration_ms});
        }
        try writer.print("\n", .{});
        
        try writer.print("## Multi-Category Model Utilization\n\n", .{});
        try writer.print("| Model | Total Requests | Categories | Primary | Secondary | Avg Score |\n", .{});
        try writer.print("|-------|----------------|------------|---------|-----------|----------|\n", .{});
        
        var model_it = self.model_stats.iterator();
        while (model_it.next()) |entry| {
            const cat_count = entry.value_ptr.categories_served.count();
            if (cat_count > 1) {
                try writer.print("| {s} | {d} | {d} | {d} | {d} | {d:.2} |\n",
                    .{entry.key_ptr.*, entry.value_ptr.total_requests, cat_count,
                      entry.value_ptr.primary_category_count, 
                      entry.value_ptr.secondary_category_count,
                      entry.value_ptr.avg_score});
            }
        }
        try writer.print("\n", .{});
        
        // Recommendations
        try writer.print("## Recommendations\n\n", .{});
        
        if (effectiveness.secondary_category_percentage > 30.0) {
            try writer.print("✓ **Good multi-category utilization** - Models are being effectively used across multiple categories\n\n", .{});
        } else {
            try writer.print("⚠ **Low multi-category utilization** - Consider reviewing confidence scores or model assignments\n\n", .{});
        }
        
        if (effectiveness.avg_secondary_confidence < 0.75) {
            try writer.print("⚠ **Low secondary category confidence** - Consider increasing confidence scores\n\n", .{});
        } else {
            try writer.print("✓ **Good secondary category confidence** - Models show strong confidence in secondary categories\n\n", .{});
        }
    }
    
    /// Print summary to stdout
    pub fn printSummary(self: *Analytics) !void {
        var effectiveness = try self.calculateEffectiveness();
        defer effectiveness.deinit();
        
        const stdout_file = std.fs.File.stdout();
        const stdout = stdout_file.deprecatedWriter();
        
        try stdout.print("\n", .{});
        try stdout.print("============================================================\n", .{});
        try stdout.print("MULTI-CATEGORY ROUTING SUMMARY\n", .{});
        try stdout.print("============================================================\n\n", .{});
        
        try stdout.print("Total Requests: {d}\n", .{effectiveness.total_requests});
        try stdout.print("Unique Models: {d}\n", .{effectiveness.unique_models_used});
        try stdout.print("Multi-Category Models: {d}\n\n", .{effectiveness.multi_category_models});
        
        try stdout.print("Secondary Category Usage: {d} ({d:.1}%)\n",
            .{effectiveness.secondary_category_usage, effectiveness.secondary_category_percentage});
        try stdout.print("Avg Secondary Confidence: {d:.2}\n", .{effectiveness.avg_secondary_confidence});
        try stdout.print("Avg Selection Duration: {d:.1}ms\n\n", .{effectiveness.avg_selection_duration_ms});
        
        try stdout.print("Top Models by Usage:\n", .{});
        // Sort and print top 5 models
        var models_list = try std.ArrayList(struct { name: []const u8, count: usize, categories: usize }).initCapacity(self.allocator, 0);
        defer models_list.deinit(self.allocator);
        
        var model_it = self.model_stats.iterator();
        while (model_it.next()) |entry| {
            try models_list.append(self.allocator, .{
                .name = entry.key_ptr.*,
                .count = entry.value_ptr.total_requests,
                .categories = entry.value_ptr.categories_served.count(),
            });
        }
        
        // Simple bubble sort for top 5
        if (models_list.items.len > 0) {
            for (0..@min(5, models_list.items.len)) |i| {
                for (i + 1..models_list.items.len) |j| {
                    if (models_list.items[j].count > models_list.items[i].count) {
                        const temp = models_list.items[i];
                        models_list.items[i] = models_list.items[j];
                        models_list.items[j] = temp;
                    }
                }
            }
            
            for (models_list.items[0..@min(5, models_list.items.len)]) |model| {
                try stdout.print("  {s}: {d} requests, {d} categories\n",
                    .{model.name, model.count, model.categories});
            }
        }
        
        try stdout.print("\n", .{});
        try stdout.print("============================================================\n\n", .{});
        try stdout.print("\n", .{});
        try stdout.print("=" ** 60, .{});
        try stdout.print("\n\n", .{});
    }
    
    fn getTimestamp(self: *Analytics) ![]const u8 {
        _ = self;
        var buf: [64]u8 = undefined;
        const now = std.time.timestamp();
        
        // Simple timestamp format
        return try std.fmt.bufPrint(&buf, "{d}", .{now});
    }
};

// CLI entry point
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        const stderr_file = std.fs.File.stderr();
        const msg = try std.fmt.allocPrint(allocator, "Usage: {s} <metrics_file> [--report] [--format markdown|json]\n", .{args[0]});
        defer allocator.free(msg);
        try stderr_file.writeAll(msg);
        std.process.exit(1);
    }
    
    const metrics_file = args[1];
    var generate_report = false;
    var format: []const u8 = "markdown";
    
    // Parse options
    for (args[2..]) |arg| {
        if (std.mem.eql(u8, arg, "--report")) {
            generate_report = true;
        } else if (std.mem.startsWith(u8, arg, "--format=")) {
            format = arg[9..];
        }
    }
    
    // Create analytics instance
    const analytics = try Analytics.init(allocator, metrics_file);
    defer analytics.deinit();
    
    // Load and analyze
    try analytics.loadMetrics();
    try analytics.analyzeCategories();
    try analytics.analyzeModels();
    
    // Print summary
    try analytics.printSummary();
    
    // Generate report if requested
    if (generate_report) {
        if (std.mem.eql(u8, format, "markdown")) {
            const stdout_file = std.fs.File.stdout();
            const stdout = stdout_file.deprecatedWriter();
            try analytics.generateMarkdownReport(stdout);
        } else if (std.mem.eql(u8, format, "json")) {
            // TODO: Implement JSON output
            std.debug.print("JSON format not yet implemented\n", .{});
        }
    }
}
