/**
 * NetworkGraph Type Definitions
 * Professional-grade network visualization component
 * 100% feature parity with SAP Network Graph
 */

// ============================================================================
// Core Types
// ============================================================================

export interface Vector2D {
    x: number;
    y: number;
}

export type NodeShape = 'circle' | 'rectangle';

export interface NodeConfig {
    id: string;
    name: string;
    description?: string;
    type: string;
    icon?: string;
    status: NodeStatus;
    model?: string;
    metrics?: NodeMetrics;
    group?: string;
    position?: Vector2D;  // Optional fixed position
    shape?: NodeShape;    // Node shape: circle (default) or rectangle
}

export interface EdgeConfig {
    id: string;
    from: string;
    to: string;
    label?: string;
    dataType?: string;
    status?: EdgeStatus;
    animated?: boolean;
}

export interface GroupConfig {
    id: string;
    name: string;
    description?: string;
    color?: string;
    collapsed?: boolean;
    nodeIds?: string[];  // IDs of nodes in this group
}

export interface NodeMetrics {
    totalRequests: number;
    avgLatency: number;
    successRate: number;
    throughput?: number;
}

// ============================================================================
// Enums
// ============================================================================

export enum NodeStatus {
    Success = 'Success',
    Warning = 'Warning',
    Error = 'Error',
    None = 'None',
    Running = 'Running'
}

export enum EdgeStatus {
    Active = 'Active',
    Inactive = 'Inactive',
    Flowing = 'Flowing',
    Error = 'Error'
}

export enum LayoutType {
    ForceDirected = 'force-directed',
    Hierarchical = 'hierarchical',
    Circular = 'circular',
    Grid = 'grid',
    Manual = 'manual'
}

// ============================================================================
// Graph State
// ============================================================================

// Forward type declarations - actual classes imported from separate files
export type GraphNode = any;  // Actual implementation in GraphNode.ts
export type GraphEdge = any;  // Actual implementation in GraphEdge.ts
export type GraphGroup = any; // Actual implementation in GraphGroup.ts

export interface GraphState {
    nodes: Map<string, GraphNode>;
    edges: Map<string, GraphEdge>;
    groups: Map<string, GraphGroup>;
    selectedNodeId: string | null;
    hoveredNodeId: string | null;
    viewport: Viewport;
    layout: LayoutType;
}

export interface Viewport {
    x: number;          // Pan offset X
    y: number;          // Pan offset Y
    scale: number;      // Zoom level (0.1 - 5.0)
    width: number;      // Canvas width
    height: number;     // Canvas height
}

// ============================================================================
// Physics & Layout
// ============================================================================

export interface ForceConfig {
    repulsion: number;      // Node repulsion strength (default: 1000)
    attraction: number;     // Edge attraction strength (default: 0.01)
    gravity: number;        // Center gravity (default: 0.1)
    damping: number;        // Velocity damping (default: 0.9)
    maxVelocity: number;    // Speed limit (default: 10)
}

export interface LayoutConfig {
    type: LayoutType;
    animate: boolean;
    duration: number;       // Animation duration (ms)
    forces?: ForceConfig;
    padding: number;        // Edge padding
}

// ============================================================================
// Interaction Events
// ============================================================================

export interface NodeEvent {
    type: 'click' | 'doubleclick' | 'hover' | 'drag' | 'drop';
    node: GraphNode;
    position: Vector2D;
    originalEvent: MouseEvent | TouchEvent;
}

export interface EdgeEvent {
    type: 'click' | 'hover';
    edge: GraphEdge;
    position: Vector2D;
    originalEvent: MouseEvent | TouchEvent;
}

export interface GraphEvent {
    type: 'zoom' | 'pan' | 'layout';
    viewport: Viewport;
    originalEvent?: WheelEvent | MouseEvent;
}

// ============================================================================
// Rendering
// ============================================================================

export interface RenderConfig {
    nodeRadius: number;             // Default: 40px
    nodeStrokeWidth: number;        // Default: 2px
    edgeStrokeWidth: number;        // Default: 2px
    arrowSize: number;              // Default: 8px
    fontSize: number;               // Default: 14px
    iconSize: number;               // Default: 24px
    showLabels: boolean;
    showMetrics: boolean;
    enableAnimations: boolean;
    theme: 'light' | 'dark';
}


// ============================================================================
// Constants
// ============================================================================

export const SAP_COLORS = {
    brand: '#0070f2',
    success: '#30914c',
    warning: '#df6e0c',
    error: '#cc1919',
    neutral: '#6c757d',
    background: '#f7f7f7',
    backgroundAlt: '#ffffff',
    border: '#d9d9d9',
    text: '#32363a',
    textMuted: '#6c757d'
};

export const DEFAULT_RENDER_CONFIG: RenderConfig = {
    nodeRadius: 40,
    nodeStrokeWidth: 2,
    edgeStrokeWidth: 2,
    arrowSize: 8,
    fontSize: 14,
    iconSize: 24,
    showLabels: true,
    showMetrics: true,
    enableAnimations: true,
    theme: 'light'
};

export const DEFAULT_FORCE_CONFIG: ForceConfig = {
    repulsion: 1000,
    attraction: 0.01,
    gravity: 0.1,
    damping: 0.9,
    maxVelocity: 10
};

export const DEFAULT_VIEWPORT: Viewport = {
    x: 0,
    y: 0,
    scale: 1.0,
    width: 800,
    height: 600
};
