# break_down_metrics.mojo
# Migrated from break_down_metrics.py
# Detailed breakdown of metrics by category

from collections import Dict, List

alias MetricCategory = String

struct CategoryMetrics:
    """Metrics for a specific category"""
    var category_name: String
    var count: Int
    var success_count: Int
    var failure_count: Int
    var total_duration_ms: Int
    var avg_duration_ms: Float32
    var values: List[Float32]
    
    fn __init__(out self):
        self.category_name = ""
        self.count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_duration_ms = 0
        self.avg_duration_ms = 0.0
        self.values = List[Float32]()
    
    fn __init__(out self, category_name: String):
        self.category_name = category_name
        self.count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_duration_ms = 0
        self.avg_duration_ms = 0.0
        self.values = List[Float32]()
    
    fn record_event(mut self, success: Bool, duration_ms: Int, value: Float32 = 0.0):
        """Record an event in this category"""
        self.count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        self.total_duration_ms += duration_ms
        if self.count > 0:
            self.avg_duration_ms = Float32(self.total_duration_ms) / Float32(self.count)
        
        if value != 0.0:
            self.values.append(value)
    
    fn get_success_rate(self) -> Float32:
        """Calculate success rate"""
        if self.count == 0:
            return 0.0
        return Float32(self.success_count) / Float32(self.count)
    
    fn get_average_value(self) -> Float32:
        """Calculate average of recorded values"""
        if len(self.values) == 0:
            return 0.0
        
        var total = 0.0
        for i in range(len(self.values)):
            total += self.values[i]
        
        return total / Float32(len(self.values))
    
    fn get_min_value(self) -> Float32:
        """Get minimum value"""
        if len(self.values) == 0:
            return 0.0
        
        var min_val = self.values[0]
        for i in range(1, len(self.values)):
            if self.values[i] < min_val:
                min_val = self.values[i]
        
        return min_val
    
    fn get_max_value(self) -> Float32:
        """Get maximum value"""
        if len(self.values) == 0:
            return 0.0
        
        var max_val = self.values[0]
        for i in range(1, len(self.values)):
            if self.values[i] > max_val:
                max_val = self.values[i]
        
        return max_val

struct BreakDownMetrics:
    """Detailed breakdown of metrics by category"""
    var categories: Dict[String, CategoryMetrics]
    var total_events: Int
    var total_successes: Int
    var total_failures: Int
    
    fn __init__(out self):
        self.categories = Dict[String, CategoryMetrics]()
        self.total_events = 0
        self.total_successes = 0
        self.total_failures = 0
    
    fn register_category(mut self, category_name: String):
        """Register a new category"""
        if category_name not in self.categories:
            self.categories[category_name] = CategoryMetrics(category_name)
    
    fn record_event(mut self, category: String, success: Bool, duration_ms: Int, value: Float32 = 0.0):
        """Record an event in a specific category"""
        # Ensure category exists
        self.register_category(category)
        
        # Update category metrics
        var cat_metrics = self.categories[category]
        cat_metrics.record_event(success, duration_ms, value)
        self.categories[category] = cat_metrics
        
        # Update totals
        self.total_events += 1
        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
    
    fn get_category_metrics(self, category: String) -> CategoryMetrics:
        """Get metrics for a specific category"""
        if category in self.categories:
            return self.categories[category]
        return CategoryMetrics()
    
    fn get_overall_success_rate(self) -> Float32:
        """Calculate overall success rate"""
        if self.total_events == 0:
            return 0.0
        return Float32(self.total_successes) / Float32(self.total_events)
    
    fn get_category_names(self) -> List[String]:
        """Get list of all category names"""
        var names = List[String]()
        for category in self.categories:
            names.append(category)
        return names
    
    fn get_category_count(self) -> Int:
        """Get number of categories"""
        return len(self.categories)
    
    fn get_top_categories(self, n: Int) -> List[String]:
        """Get top N categories by event count"""
        # Simple implementation - would need sorting in production
        var top = List[String]()
        var added = 0
        
        for category in self.categories:
            if added < n:
                top.append(category)
                added += 1
        
        return top
    
    fn reset(mut self):
        """Reset all metrics"""
        self.categories = Dict[String, CategoryMetrics]()
        self.total_events = 0
        self.total_successes = 0
        self.total_failures = 0
    
    fn to_summary(self) -> String:
        """Generate a summary string"""
        var summary = "Breakdown Metrics Summary\n"
        summary = summary + "=" * 50 + "\n"
        summary = summary + "Total Events: " + str(self.total_events) + "\n"
        summary = summary + "Total Successes: " + str(self.total_successes) + "\n"
        summary = summary + "Total Failures: " + str(self.total_failures) + "\n"
        summary = summary + "Overall Success Rate: " + str(self.get_overall_success_rate() * 100.0) + "%\n"
        summary = summary + "Total Categories: " + str(self.get_category_count()) + "\n"
        summary = summary + "=" * 50 + "\n\n"
        
        # Add category details
        for category in self.categories:
            let metrics = self.categories[category]
            summary = summary + "Category: " + metrics.category_name + "\n"
            summary = summary + "  Count: " + str(metrics.count) + "\n"
            summary = summary + "  Success: " + str(metrics.success_count) + "\n"
            summary = summary + "  Failure: " + str(metrics.failure_count) + "\n"
            summary = summary + "  Success Rate: " + str(metrics.get_success_rate() * 100.0) + "%\n"
            summary = summary + "  Avg Duration: " + str(metrics.avg_duration_ms) + "ms\n"
            
            if len(metrics.values) > 0:
                summary = summary + "  Avg Value: " + str(metrics.get_average_value()) + "\n"
                summary = summary + "  Min Value: " + str(metrics.get_min_value()) + "\n"
                summary = summary + "  Max Value: " + str(metrics.get_max_value()) + "\n"
            
            summary = summary + "\n"
        
        return summary

struct MetricsComparison:
    """Compare metrics across different runs or configurations"""
    var baseline_metrics: BreakDownMetrics
    var comparison_metrics: BreakDownMetrics
    var baseline_name: String
    var comparison_name: String
    
    fn __init__(out self):
        self.baseline_metrics = BreakDownMetrics()
        self.comparison_metrics = BreakDownMetrics()
        self.baseline_name = "Baseline"
        self.comparison_name = "Comparison"
    
    fn __init__(out self, baseline: BreakDownMetrics, comparison: BreakDownMetrics, 
                baseline_name: String, comparison_name: String):
        self.baseline_metrics = baseline
        self.comparison_metrics = comparison
        self.baseline_name = baseline_name
        self.comparison_name = comparison_name
    
    fn compare_success_rates(self) -> Float32:
        """Calculate difference in success rates"""
        let baseline_rate = self.baseline_metrics.get_overall_success_rate()
        let comparison_rate = self.comparison_metrics.get_overall_success_rate()
        return comparison_rate - baseline_rate
    
    fn compare_category(self, category: String) -> Float32:
        """Compare a specific category's success rate"""
        let baseline_cat = self.baseline_metrics.get_category_metrics(category)
        let comparison_cat = self.comparison_metrics.get_category_metrics(category)
        
        return comparison_cat.get_success_rate() - baseline_cat.get_success_rate()
    
    fn generate_comparison_report(self) -> String:
        """Generate a comparison report"""
        var report = "Metrics Comparison Report\n"
        report = report + "=" * 60 + "\n"
        report = report + "Baseline: " + self.baseline_name + "\n"
        report = report + "Comparison: " + self.comparison_name + "\n"
        report = report + "=" * 60 + "\n\n"
        
        # Overall comparison
        let success_diff = self.compare_success_rates()
        report = report + "Overall Success Rate Difference: "
        if success_diff > 0:
            report = report + "+" + str(success_diff * 100.0) + "% (improved)\n"
        else:
            report = report + str(success_diff * 100.0) + "% (decreased)\n"
        
        report = report + "\n"
        
        # Category-by-category comparison
        let categories = self.baseline_metrics.get_category_names()
        for i in range(len(categories)):
            let category = categories[i]
            let diff = self.compare_category(category)
            
            report = report + "Category: " + category + "\n"
            report = report + "  Difference: "
            if diff > 0:
                report = report + "+" + str(diff * 100.0) + "%\n"
            else:
                report = report + str(diff * 100.0) + "%\n"
        
        return report

fn create_breakdown_metrics() -> BreakDownMetrics:
    """Factory function to create breakdown metrics"""
    return BreakDownMetrics()
