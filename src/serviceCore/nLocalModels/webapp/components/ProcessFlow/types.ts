/**
 * ProcessFlow Types
 * 100% SAP Commercial Quality - Exact styling and behavior
 */

// ============================================================================
// Core Types
// ============================================================================

export interface ProcessFlowNode {
    id: string;
    lane: string;
    title: string;
    titleAbbreviation?: string;
    state: ProcessFlowNodeState;
    stateText?: string;
    texts?: string[];
    children?: string[];
    focused?: boolean;
    highlighted?: boolean;
    position?: number;
    isTitleClickable?: boolean;
    icon?: string;
    // Folded corners - document-style visual (SAP Fiori signature style)
    foldedCorners?: boolean;
    // Aggregation properties (stacked nodes)
    isAggregated?: boolean;
    aggregatedCount?: number;
    aggregatedItems?: ProcessFlowNode[];
}

export interface ProcessFlowLane {
    id: string;
    label: string;
    position: number;
    state?: ProcessFlowLaneState;
    icon?: string;
}

export interface ProcessFlowConnection {
    from: string;
    to: string;
    state?: ProcessFlowConnectionState;
    type?: 'normal' | 'planned';
}

// ============================================================================
// Enums - Exact SAP Values
// ============================================================================

export enum ProcessFlowNodeState {
    Positive = 'Positive',          // Green - Success
    Negative = 'Negative',          // Red - Error
    Critical = 'Critical',          // Orange - Warning
    Planned = 'Planned',            // Gray - Not started
    PlannedNegative = 'PlannedNegative',  // Gray with red border
    Neutral = 'Neutral'             // Blue - Running/Info
}

export enum ProcessFlowLaneState {
    value = 'value',
    warning = 'warning',
    error = 'error'
}

export enum ProcessFlowConnectionState {
    Normal = 'Normal',
    Highlighted = 'Highlighted',
    Dimmed = 'Dimmed'
}

export enum ProcessFlowDisplayState {
    Regular = 'Regular',
    Highlighted = 'Highlighted',
    HighlightedNonAdj = 'HighlightedNonAdj',  // Adjacent nodes dimmed
    Dimmed = 'Dimmed',
    Selected = 'Selected'
}

export enum ProcessFlowZoomLevel {
    One = 'One',      // Largest - header, status, 2 attributes
    Two = 'Two',      // Standard (auto for screens >1024px) - header, status, 1 attribute
    Three = 'Three',  // Reduced (auto for 600-1023px) - header and status only
    Four = 'Four'     // Smallest (auto for <600px) - status icon only
}

// SAP Fiori Semantic Zoom Level Configuration
export interface ZoomLevelConfig {
    scale: number;
    nodeWidth: number;
    nodeHeight: number;
    showHeader: boolean;
    showStatus: boolean;
    showAttr1: boolean;
    showAttr2: boolean;
}

export const ZOOM_LEVEL_CONFIG: Record<ProcessFlowZoomLevel, ZoomLevelConfig> = {
    [ProcessFlowZoomLevel.One]: {
        scale: 1.2,
        nodeWidth: 200,
        nodeHeight: 100,
        showHeader: true,
        showStatus: true,
        showAttr1: true,
        showAttr2: true
    },
    [ProcessFlowZoomLevel.Two]: {
        scale: 1.0,
        nodeWidth: 160,
        nodeHeight: 80,
        showHeader: true,
        showStatus: true,
        showAttr1: true,
        showAttr2: false
    },
    [ProcessFlowZoomLevel.Three]: {
        scale: 0.8,
        nodeWidth: 120,
        nodeHeight: 60,
        showHeader: true,
        showStatus: true,
        showAttr1: false,
        showAttr2: false
    },
    [ProcessFlowZoomLevel.Four]: {
        scale: 0.6,
        nodeWidth: 60,
        nodeHeight: 40,
        showHeader: false,
        showStatus: true,
        showAttr1: false,
        showAttr2: false
    }
};

// ============================================================================
// SAP Fiori Colors - Exact Commercial Palette
// ============================================================================

export const PROCESS_FLOW_COLORS = {
    // Node states
    positive: {
        background: '#107e3e',      // SAP Semantic Success Dark
        border: '#0a6534',
        text: '#ffffff'
    },
    negative: {
        background: '#bb0000',      // SAP Semantic Error
        border: '#a20000',
        text: '#ffffff'
    },
    critical: {
        background: '#e9730c',      // SAP Semantic Warning
        border: '#c9630a',
        text: '#ffffff'
    },
    planned: {
        background: '#ededed',      // SAP Gray 2
        border: '#d9d9d9',
        text: '#32363a'
    },
    plannedNegative: {
        background: '#ededed',
        border: '#bb0000',
        text: '#32363a'
    },
    neutral: {
        background: '#0a6ed1',      // SAP Semantic Information
        border: '#0854a0',
        text: '#ffffff'
    },
    
    // Connection states
    connection: {
        normal: '#6a6d70',          // SAP Gray 7
        highlighted: '#0a6ed1',     // SAP Blue
        dimmed: '#d9d9d9'           // SAP Gray 3
    },
    
    // Lane backgrounds
    lane: {
        default: '#ffffff',
        alternate: '#fafafa'        // SAP Background
    },
    
    // Text colors
    text: {
        primary: '#32363a',         // SAP Text Color
        secondary: '#6a6d70',       // SAP Gray 7
        light: '#89919a'            // SAP Gray 6
    },
    
    // Borders and dividers
    border: '#d9d9d9',              // SAP Gray 3
    
    // Hover and selection
    hover: 'rgba(10, 110, 209, 0.1)',
    selected: 'rgba(10, 110, 209, 0.2)',
    focus: '#0a6ed1'
};

// ============================================================================
// Layout Configuration - SAP Standard
// ============================================================================

export const PROCESS_FLOW_LAYOUT = {
    // Node dimensions
    node: {
        width: 160,                 // Standard SAP width
        height: 80,                 // Standard SAP height
        cornerRadius: 4,            // SAP corner radius
        borderWidth: 2,
        padding: 12,
        iconSize: 24,
        titleFontSize: 14,
        textFontSize: 12
    },
    
    // Spacing
    spacing: {
        horizontal: 80,             // Between nodes horizontally
        vertical: 100,              // Between lanes
        laneHeader: 120,            // Lane label width
        topMargin: 20,
        bottomMargin: 20,
        leftMargin: 20,
        rightMargin: 20
    },
    
    // Connection lines
    connection: {
        strokeWidth: 2,
        arrowSize: 8,
        cornerRadius: 8,            // Rounded corners for connections
        dashArray: '5,5'            // For planned connections
    },
    
    // Zoom levels - SAP standard
    zoom: {
        one: {
            scale: 1.0,
            showTexts: true,
            showIcons: true
        },
        two: {
            scale: 0.75,
            showTexts: true,
            showIcons: true
        },
        three: {
            scale: 0.5,
            showTexts: false,
            showIcons: true
        },
        four: {
            scale: 0.25,
            showTexts: false,
            showIcons: false
        }
    },
    
    // Animation timing
    animation: {
        duration: 300,              // ms
        easing: 'cubic-bezier(0.4, 0, 0.2, 1)'  // SAP standard easing
    }
};

// ============================================================================
// Event Types
// ============================================================================

export interface ProcessFlowNodeEvent {
    type: 'click' | 'press' | 'hover';
    node: ProcessFlowNode;
    originalEvent: MouseEvent | TouchEvent;
}

export interface ProcessFlowLaneEvent {
    type: 'click' | 'press';
    lane: ProcessFlowLane;
    originalEvent: MouseEvent | TouchEvent;
}

// ============================================================================
// Configuration
// ============================================================================

export interface ProcessFlowConfig {
    showLabels?: boolean;
    scrollable?: boolean;
    foldedCorners?: boolean;      // SAP signature style
    wheelZoomable?: boolean;
    optimizeDisplay?: boolean;     // Performance mode for large flows
    zoomLevel?: ProcessFlowZoomLevel;
}

export const DEFAULT_PROCESS_FLOW_CONFIG: ProcessFlowConfig = {
    showLabels: true,
    scrollable: true,
    foldedCorners: true,           // SAP signature folded corner
    wheelZoomable: true,
    optimizeDisplay: true,
    zoomLevel: ProcessFlowZoomLevel.One
};
