/**
 * NetworkGraph Type Definitions
 * Professional-grade network visualization component
 * 100% feature parity with SAP Network Graph
 */
// ============================================================================
// Enums
// ============================================================================
export var NodeStatus;
(function (NodeStatus) {
    NodeStatus["Success"] = "Success";
    NodeStatus["Warning"] = "Warning";
    NodeStatus["Error"] = "Error";
    NodeStatus["None"] = "None";
    NodeStatus["Running"] = "Running";
})(NodeStatus || (NodeStatus = {}));
export var EdgeStatus;
(function (EdgeStatus) {
    EdgeStatus["Active"] = "Active";
    EdgeStatus["Inactive"] = "Inactive";
    EdgeStatus["Flowing"] = "Flowing";
    EdgeStatus["Error"] = "Error";
})(EdgeStatus || (EdgeStatus = {}));
export var LayoutType;
(function (LayoutType) {
    LayoutType["ForceDirected"] = "force-directed";
    LayoutType["Hierarchical"] = "hierarchical";
    LayoutType["Circular"] = "circular";
    LayoutType["Grid"] = "grid";
    LayoutType["Manual"] = "manual";
})(LayoutType || (LayoutType = {}));
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
export const DEFAULT_RENDER_CONFIG = {
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
export const DEFAULT_FORCE_CONFIG = {
    repulsion: 1000,
    attraction: 0.01,
    gravity: 0.1,
    damping: 0.9,
    maxVelocity: 10
};
export const DEFAULT_VIEWPORT = {
    x: 0,
    y: 0,
    scale: 1.0,
    width: 800,
    height: 600
};
//# sourceMappingURL=types.js.map