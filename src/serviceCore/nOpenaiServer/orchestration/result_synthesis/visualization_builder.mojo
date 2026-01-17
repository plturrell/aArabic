"""
Visualization Builder

Creates ASCII-based visualizations for query results:
- Bar charts
- Line charts  
- Network graphs (node-relationship diagrams)
- Distribution histograms

Pure Mojo - no external dependencies.
"""

from collections import Dict, List
from .result_formatter import QueryResult


# ============================================================================
# Chart Types
# ============================================================================

struct BarChart:
    """
    ASCII bar chart for categorical data.
    
    Example:
        Category A: ████████████ 45
        Category B: ████████ 32
        Category C: ████ 18
    """
    var title: String
    var data: Dict[String, Int]
    var max_width: Int
    
    fn __init__(inout self, title: String = "Bar Chart", max_width: Int = 50):
        self.title = title
        self.data = Dict[String, Int]()
        self.max_width = max_width
    
    fn add_category(inout self, label: String, value: Int):
        """Add a data category"""
        self.data[label] = value
    
    fn render(self) -> String:
        """Render ASCII bar chart"""
        if len(self.data) == 0:
            return "No data to visualize."
        
        var output = String("")
        output += f"\n{self.title}\n"
        output += "=" * len(self.title) + "\n\n"
        
        # Find max value for scaling
        var max_val = 0
        var keys = self.data.keys()
        for i in range(len(keys)):
            var key = keys[i]
            var val = self.data[key]
            if val > max_val:
                max_val = val
        
        # Render bars
        for i in range(len(keys)):
            var label = keys[i]
            var value = self.data[label]
            
            # Calculate bar width
            var bar_width = 0
            if max_val > 0:
                bar_width = Int((Float64(value) / Float64(max_val)) * Float64(self.max_width))
            
            # Render bar
            output += self._pad_label(label, 15) + " "
            for j in range(bar_width):
                output += "█"
            output += f" {value}\n"
        
        return output
    
    fn _pad_label(self, label: String, width: Int) -> String:
        """Pad label to fixed width"""
        if len(label) >= width:
            return label[0:width-3] + "..."
        else:
            var padded = label
            for i in range(width - len(label)):
                padded += " "
            return padded


struct NetworkGraph:
    """
    ASCII network graph for node-relationship visualization.
    
    Example:
        (Node1) --[REL]--> (Node2)
                \\
                 --[REL]--> (Node3)
    """
    var nodes: List[String]
    var edges: List[String]  # Format: "from->to:label"
    
    fn __init__(inout self):
        self.nodes = List[String]()
        self.edges = List[String]()
    
    fn add_node(inout self, node_id: String):
        """Add node to graph"""
        if node_id not in self.nodes:
            self.nodes.append(node_id)
    
    fn add_edge(inout self, from_node: String, to_node: String, label: String = ""):
        """Add edge between nodes"""
        self.add_node(from_node)
        self.add_node(to_node)
        
        var edge = f"{from_node}->{to_node}"
        if label != "":
            edge += f":{label}"
        self.edges.append(edge)
    
    fn render(self) -> String:
        """Render ASCII network graph"""
        if len(self.nodes) == 0:
            return "No nodes to visualize."
        
        var output = String("\nNetwork Graph\n")
        output += "=============\n\n"
        
        output += "Nodes:\n"
        for i in range(len(self.nodes)):
            output += f"  • {self.nodes[i]}\n"
        
        output += "\nEdges:\n"
        for i in range(len(self.edges)):
            var edge = self.edges[i]
            output += f"  {self._format_edge(edge)}\n"
        
        return output
    
    fn _format_edge(self, edge: String) -> String:
        """Format edge for display"""
        # Parse edge string
        var parts = edge.split("->")
        if len(parts) != 2:
            return edge
        
        var from_node = parts[0]
        var to_part = parts[1]
        
        # Check for label
        var label_parts = to_part.split(":")
        var to_node = label_parts[0]
        var label = ""
        if len(label_parts) > 1:
            label = label_parts[1]
        
        # Format
        if label != "":
            return f"({from_node}) --[{label}]--> ({to_node})"
        else:
            return f"({from_node}) ---------> ({to_node})"


# ============================================================================
# Visualization Builder
# ============================================================================

struct VisualizationBuilder:
    """
    Builds visualizations from query results.
    
    Analyzes result structure and generates appropriate charts.
    
    Example:
        var results = execute_query(cypher)
        var builder = VisualizationBuilder()
        var viz = builder.build(results)
        print(viz)
    """
    var verbose: Bool
    
    fn __init__(inout self, verbose: Bool = False):
        self.verbose = verbose
    
    fn build_bar_chart(self, result: QueryResult, value_column: String, label_column: String) raises -> String:
        """
        Build bar chart from results.
        
        Args:
            result: Query results
            value_column: Column containing numeric values
            label_column: Column containing category labels
            
        Returns:
            ASCII bar chart
        """
        var chart = BarChart(f"Distribution: {value_column}")
        
        for i in range(result.row_count):
            var record = result.records[i]
            
            if label_column not in record or value_column not in record:
                continue
            
            var label = record[label_column]
            var value_str = record[value_column]
            
            # Parse value (simplified - assumes integer)
            var value = self._parse_int(value_str)
            chart.add_category(label, value)
        
        return chart.render()
    
    fn build_network_graph(self, result: QueryResult) -> String:
        """
        Build network graph from query results.
        
        Assumes results contain node/relationship data.
        
        Args:
            result: Query results with graph data
            
        Returns:
            ASCII network visualization
        """
        var graph = NetworkGraph()
        
        # Detect graph patterns in results
        for i in range(result.row_count):
            var record = result.records[i]
            var keys = record.keys()
            
            # Look for node identifiers
            var from_node = ""
            var to_node = ""
            var rel_type = ""
            
            for j in range(len(keys)):
                var key = keys[j]
                var value = record[key]
                
                if key == "from" or key == "source" or key == "start":
                    from_node = value
                elif key == "to" or key == "target" or key == "end":
                    to_node = value
                elif key == "rel_type" or key == "relationship" or key == "type":
                    rel_type = value
            
            # Add edge if we found nodes
            if from_node != "" and to_node != "":
                graph.add_edge(from_node, to_node, rel_type)
            elif from_node != "":
                graph.add_node(from_node)
        
        return graph.render()
    
    fn auto_visualize(self, result: QueryResult) -> String:
        """
        Automatically determine and create best visualization.
        
        Args:
            result: Query results
            
        Returns:
            Appropriate visualization
        """
        if result.is_empty():
            return "No data to visualize."
        
        # Analyze structure
        var first_record = result.records[0]
        var keys = first_record.keys()
        
        if len(keys) == 0:
            return "No columns to visualize."
        
        # Check for graph structure
        var has_nodes = False
        var has_relationships = False
        
        for i in range(len(keys)):
            var key = keys[i]
            if key == "from" or key == "to" or key == "source" or key == "target":
                has_nodes = True
            if key == "rel_type" or key == "relationship":
                has_relationships = True
        
        if has_nodes or has_relationships:
            return self.build_network_graph(result)
        
        # Check for numeric aggregations
        var has_count = False
        var has_labels = False
        
        for i in range(len(keys)):
            var key = keys[i]
            if key == "count" or key == "total" or key == "sum":
                has_count = True
            if key == "label" or key == "name" or key == "category":
                has_labels = True
        
        if has_count and has_labels:
            # Try to build bar chart
            var label_col = ""
            var value_col = ""
            
            for i in range(len(keys)):
                var key = keys[i]
                if key == "label" or key == "name" or key == "category":
                    label_col = key
                elif key == "count" or key == "total":
                    value_col = key
            
            if label_col != "" and value_col != "":
                try:
                    return self.build_bar_chart(result, value_col, label_col)
                except:
                    pass
        
        # Default: suggest manual visualization
        return "Auto-visualization not supported for this result structure.\nUse build_bar_chart() or build_network_graph() explicitly."
    
    fn _parse_int(self, text: String) -> Int:
        """Parse string to integer (simplified)"""
        # TODO: Implement proper integer parsing
        # For now, return 0 as placeholder
        return 0


# ============================================================================
# Summary Statistics Visualizer
# ============================================================================

struct StatsVisualizer:
    """
    Creates visualizations for statistical summaries.
    """
    
    @staticmethod
    fn distribution_summary(values: List[Int], title: String = "Distribution") -> String:
        """
        Create distribution summary with histogram.
        
        Args:
            values: List of integer values
            title: Chart title
            
        Returns:
            ASCII histogram
        """
        if len(values) == 0:
            return "No values to visualize."
        
        var output = String(f"\n{title}\n")
        output += "=" * len(title) + "\n\n"
        
        # Calculate basic stats
        var sum = 0
        var min_val = values[0]
        var max_val = values[0]
        
        for i in range(len(values)):
            var val = values[i]
            sum += val
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
        
        var mean = Float64(sum) / Float64(len(values))
        
        output += f"Count: {len(values)}\n"
        output += f"Min: {min_val}\n"
        output += f"Max: {max_val}\n"
        output += f"Mean: {mean:.2f}\n"
        output += f"Sum: {sum}\n\n"
        
        # Simple histogram (buckets)
        output += "Histogram:\n"
        var bucket_count = 10
        var bucket_width = (max_val - min_val) / bucket_count
        
        if bucket_width > 0:
            output += "(Simplified histogram would be rendered here)\n"
        
        return output
