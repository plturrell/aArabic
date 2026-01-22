/**
 * Chart Suite Types
 * Custom chart components for training metrics visualization
 */

// Color palette matching SAP Fiori
export const CHART_COLORS = {
    primary: '#0a6ed1',      // SAP Blue
    success: '#107e3e',      // Green
    warning: '#e9730c',      // Orange
    error: '#bb0000',        // Red
    neutral: '#6a6d70',      // Gray
    
    // Gradients for charts
    gradient: {
        blue: ['#0a6ed1', '#1a9fff'],
        green: ['#107e3e', '#2ecc71'],
        orange: ['#e9730c', '#f39c12'],
        red: ['#bb0000', '#e74c3c']
    },
    
    // Series colors for multi-line charts
    series: [
        '#0a6ed1', '#107e3e', '#e9730c', '#9b59b6', 
        '#1abc9c', '#e74c3c', '#3498db', '#f39c12'
    ]
};

// Base chart configuration
export interface ChartConfig {
    width?: number;
    height?: number;
    margin?: { top: number; right: number; bottom: number; left: number };
    animate?: boolean;
    animationDuration?: number;
    responsive?: boolean;
}

// Radial/Gauge chart
export interface RadialChartConfig extends ChartConfig {
    value: number;           // 0-100
    maxValue?: number;
    minValue?: number;
    label?: string;
    unit?: string;
    thresholds?: { value: number; color: string }[];
    arcWidth?: number;
    showValue?: boolean;
    showLabel?: boolean;
}

// Line chart for training curves
export interface LineChartConfig extends ChartConfig {
    series: SeriesData[];
    xAxisLabel?: string;
    yAxisLabel?: string;
    showGrid?: boolean;
    showLegend?: boolean;
    showTooltip?: boolean;
    xAxisType?: 'linear' | 'time' | 'category';
    yAxisType?: 'linear' | 'log';
}

export interface SeriesData {
    name: string;
    data: DataPoint[];
    color?: string;
    lineWidth?: number;
    showPoints?: boolean;
    dashed?: boolean;
}

export interface DataPoint {
    x: number | string | Date;
    y: number;
    label?: string;
}

// Bar chart
export interface BarChartConfig extends ChartConfig {
    data: BarData[];
    orientation?: 'vertical' | 'horizontal';
    showValues?: boolean;
    grouped?: boolean;
    stacked?: boolean;
    xAxisLabel?: string;
    yAxisLabel?: string;
}

export interface BarData {
    category: string;
    values: { name: string; value: number; color?: string }[];
}

// Sankey diagram
export interface SankeyConfig extends ChartConfig {
    nodes: SankeyNode[];
    links: SankeyLink[];
    nodeWidth?: number;
    nodePadding?: number;
    showLabels?: boolean;
    showValues?: boolean;
}

export interface SankeyNode {
    id: string;
    name: string;
    color?: string;
    // Computed
    x?: number;
    y?: number;
    height?: number;
    value?: number;
}

export interface SankeyLink {
    source: string;
    target: string;
    value: number;
    color?: string;
    // Computed
    path?: string;
    width?: number;
}

// Gauge chart (similar to speedometer)
export interface GaugeChartConfig extends ChartConfig {
    value: number;
    min?: number;
    max?: number;
    label?: string;
    unit?: string;
    zones?: GaugeZone[];
    showNeedle?: boolean;
    needleColor?: string;
}

export interface GaugeZone {
    min: number;
    max: number;
    color: string;
    label?: string;
}

// Default configurations
export const DEFAULT_CHART_CONFIG: ChartConfig = {
    width: 400,
    height: 300,
    margin: { top: 20, right: 20, bottom: 40, left: 50 },
    animate: true,
    animationDuration: 300,
    responsive: true
};

export const DEFAULT_RADIAL_CONFIG: Partial<RadialChartConfig> = {
    maxValue: 100,
    minValue: 0,
    arcWidth: 20,
    showValue: true,
    showLabel: true,
    thresholds: [
        { value: 33, color: CHART_COLORS.error },
        { value: 66, color: CHART_COLORS.warning },
        { value: 100, color: CHART_COLORS.success }
    ]
};

export const DEFAULT_GAUGE_CONFIG: Partial<GaugeChartConfig> = {
    min: 0,
    max: 100,
    showNeedle: true,
    needleColor: '#333',
    zones: [
        { min: 0, max: 33, color: CHART_COLORS.success, label: 'Good' },
        { min: 33, max: 66, color: CHART_COLORS.warning, label: 'Warning' },
        { min: 66, max: 100, color: CHART_COLORS.error, label: 'Critical' }
    ]
};

