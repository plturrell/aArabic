"""
Result Synthesis Module

Formats and visualizes query results in human-readable formats.

Exports:
    - QueryResult: Raw query results structure
    - FormattedResult: Formatted result structure
    - ResultFormatter: Main formatter
    - ResponseGenerator: Natural language responses
    - BarChart: ASCII bar charts
    - NetworkGraph: Network visualizations
    - VisualizationBuilder: Auto-visualization
    - StatsVisualizer: Statistical summaries

Usage:
    from orchestration.result_synthesis import (
        QueryResult,
        ResultFormatter,
        VisualizationBuilder,
        ResponseGenerator
    )
    
    # Execute query and get results
    var results = QueryResult(records, "supply_chain", cypher)
    
    # Format results
    var formatter = ResultFormatter(schema)
    var formatted = formatter.format(results)
    
    # Print human-readable output
    print(formatted.summary)
    print(formatted.table)
    
    # Generate natural language response
    var generator = ResponseGenerator(schema)
    var response = generator.generate_response(query, results, formatted)
    print(response)
    
    # Create visualization
    var viz_builder = VisualizationBuilder()
    var viz = viz_builder.auto_visualize(results)
    print(viz)
"""

from .result_formatter import (
    QueryResult,
    FormattedResult,
    ResultFormatter,
    ResponseGenerator,
    export_to_csv,
    export_to_json_file
)

from .visualization_builder import (
    BarChart,
    NetworkGraph,
    VisualizationBuilder,
    StatsVisualizer
)

__all__ = [
    # Core types
    "QueryResult",
    "FormattedResult",
    
    # Formatters
    "ResultFormatter",
    "ResponseGenerator",
    
    # Visualizations
    "BarChart",
    "NetworkGraph",
    "VisualizationBuilder",
    "StatsVisualizer",
    
    # Export utilities
    "export_to_csv",
    "export_to_json_file"
]
