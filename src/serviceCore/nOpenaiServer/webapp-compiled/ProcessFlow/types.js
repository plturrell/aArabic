/**
 * ProcessFlow Types
 * 100% SAP Commercial Quality - Exact styling and behavior
 */
// ============================================================================
// Enums - Exact SAP Values
// ============================================================================
export var ProcessFlowNodeState;
(function (ProcessFlowNodeState) {
    ProcessFlowNodeState["Positive"] = "Positive";
    ProcessFlowNodeState["Negative"] = "Negative";
    ProcessFlowNodeState["Critical"] = "Critical";
    ProcessFlowNodeState["Planned"] = "Planned";
    ProcessFlowNodeState["PlannedNegative"] = "PlannedNegative";
    ProcessFlowNodeState["Neutral"] = "Neutral"; // Blue - Running/Info
})(ProcessFlowNodeState || (ProcessFlowNodeState = {}));
export var ProcessFlowLaneState;
(function (ProcessFlowLaneState) {
    ProcessFlowLaneState["value"] = "value";
    ProcessFlowLaneState["warning"] = "warning";
    ProcessFlowLaneState["error"] = "error";
})(ProcessFlowLaneState || (ProcessFlowLaneState = {}));
export var ProcessFlowConnectionState;
(function (ProcessFlowConnectionState) {
    ProcessFlowConnectionState["Normal"] = "Normal";
    ProcessFlowConnectionState["Highlighted"] = "Highlighted";
    ProcessFlowConnectionState["Dimmed"] = "Dimmed";
})(ProcessFlowConnectionState || (ProcessFlowConnectionState = {}));
export var ProcessFlowDisplayState;
(function (ProcessFlowDisplayState) {
    ProcessFlowDisplayState["Regular"] = "Regular";
    ProcessFlowDisplayState["Highlighted"] = "Highlighted";
    ProcessFlowDisplayState["HighlightedNonAdj"] = "HighlightedNonAdj";
    ProcessFlowDisplayState["Dimmed"] = "Dimmed";
    ProcessFlowDisplayState["Selected"] = "Selected";
})(ProcessFlowDisplayState || (ProcessFlowDisplayState = {}));
export var ProcessFlowZoomLevel;
(function (ProcessFlowZoomLevel) {
    ProcessFlowZoomLevel["One"] = "One";
    ProcessFlowZoomLevel["Two"] = "Two";
    ProcessFlowZoomLevel["Three"] = "Three";
    ProcessFlowZoomLevel["Four"] = "Four"; // Smallest (auto for <600px) - status icon only
})(ProcessFlowZoomLevel || (ProcessFlowZoomLevel = {}));
export const ZOOM_LEVEL_CONFIG = {
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
        background: '#107e3e', // SAP Semantic Success Dark
        border: '#0a6534',
        text: '#ffffff'
    },
    negative: {
        background: '#bb0000', // SAP Semantic Error
        border: '#a20000',
        text: '#ffffff'
    },
    critical: {
        background: '#e9730c', // SAP Semantic Warning
        border: '#c9630a',
        text: '#ffffff'
    },
    planned: {
        background: '#ededed', // SAP Gray 2
        border: '#d9d9d9',
        text: '#32363a'
    },
    plannedNegative: {
        background: '#ededed',
        border: '#bb0000',
        text: '#32363a'
    },
    neutral: {
        background: '#0a6ed1', // SAP Semantic Information
        border: '#0854a0',
        text: '#ffffff'
    },
    // Connection states
    connection: {
        normal: '#6a6d70', // SAP Gray 7
        highlighted: '#0a6ed1', // SAP Blue
        dimmed: '#d9d9d9' // SAP Gray 3
    },
    // Lane backgrounds
    lane: {
        default: '#ffffff',
        alternate: '#fafafa' // SAP Background
    },
    // Text colors
    text: {
        primary: '#32363a', // SAP Text Color
        secondary: '#6a6d70', // SAP Gray 7
        light: '#89919a' // SAP Gray 6
    },
    // Borders and dividers
    border: '#d9d9d9', // SAP Gray 3
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
        width: 160, // Standard SAP width
        height: 80, // Standard SAP height
        cornerRadius: 4, // SAP corner radius
        borderWidth: 2,
        padding: 12,
        iconSize: 24,
        titleFontSize: 14,
        textFontSize: 12
    },
    // Spacing
    spacing: {
        horizontal: 80, // Between nodes horizontally
        vertical: 100, // Between lanes
        laneHeader: 120, // Lane label width
        topMargin: 20,
        bottomMargin: 20,
        leftMargin: 20,
        rightMargin: 20
    },
    // Connection lines
    connection: {
        strokeWidth: 2,
        arrowSize: 8,
        cornerRadius: 8, // Rounded corners for connections
        dashArray: '5,5' // For planned connections
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
        duration: 300, // ms
        easing: 'cubic-bezier(0.4, 0, 0.2, 1)' // SAP standard easing
    }
};
export const DEFAULT_PROCESS_FLOW_CONFIG = {
    showLabels: true,
    scrollable: true,
    foldedCorners: true, // SAP signature folded corner
    wheelZoomable: true,
    optimizeDisplay: true,
    zoomLevel: ProcessFlowZoomLevel.One
};
//# sourceMappingURL=types.js.map