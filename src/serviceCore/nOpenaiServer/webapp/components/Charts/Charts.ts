/**
 * Charts Suite - Custom chart components for ML training visualization
 * Built with TypeScript, compiled with Bun
 * 
 * Features:
 * - RadialChart: Circular progress for percentages (win rate, accuracy)
 * - GaugeChart: Semi-circular gauge with zones (KL divergence)
 * - LineChart: Multi-series line chart for training curves
 * - BarChart: Grouped/stacked bar chart for comparisons
 * - SankeyDiagram: Flow visualization for data pipelines
 */

export { RadialChart } from './RadialChart';
export { GaugeChart } from './GaugeChart';
export { LineChart } from './LineChart';
export { BarChart } from './BarChart';
export { SankeyDiagram } from './SankeyDiagram';

// Export all types
export * from './types';

// Convenience factory for creating charts
export function createChart(
    type: 'radial' | 'gauge' | 'line' | 'bar' | 'sankey',
    container: HTMLElement | string,
    config: any
) {
    switch (type) {
        case 'radial':
            return new (require('./RadialChart').RadialChart)(container, config);
        case 'gauge':
            return new (require('./GaugeChart').GaugeChart)(container, config);
        case 'line':
            return new (require('./LineChart').LineChart)(container, config);
        case 'bar':
            return new (require('./BarChart').BarChart)(container, config);
        case 'sankey':
            return new (require('./SankeyDiagram').SankeyDiagram)(container, config);
        default:
            throw new Error(`Unknown chart type: ${type}`);
    }
}

