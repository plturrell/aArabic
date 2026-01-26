/**
 * WorkflowBuilder Type Definitions
 * Visual drag-and-drop workflow creation component
 */

export enum NodeType {
    ROUTER = 'router',
    CODE = 'code',
    TRANSLATION = 'translation',
    RAG = 'rag',
    VALIDATION = 'validation',
    ORCHESTRATOR = 'orchestrator',
    CUSTOM = 'custom',
    MATH = 'math',
    CREATIVE = 'creative',
    SUMMARIZER = 'summarizer',
    CLASSIFIER = 'classifier',
    EMBEDDER = 'embedder',
    SEARCH = 'search',
    MEMORY = 'memory',
    PLANNER = 'planner'
}

export enum PortType {
    INPUT = 'input',
    OUTPUT = 'output'
}

export interface Port {
    id: string;
    type: PortType;
    label?: string;
    connected: boolean;
}

export interface WorkflowNode {
    id: string;
    type: NodeType;
    name: string;
    description?: string;
    x: number;
    y: number;
    width: number;
    height: number;
    inputPorts: Port[];
    outputPorts: Port[];
    properties: Record<string, any>;
    modelId?: string;
}

export interface WorkflowConnection {
    id: string;
    sourceNodeId: string;
    sourcePortId: string;
    targetNodeId: string;
    targetPortId: string;
    label?: string;
}

export interface WorkflowDefinition {
    id: string;
    name: string;
    description?: string;
    nodes: WorkflowNode[];
    connections: WorkflowConnection[];
    createdAt: Date;
    updatedAt: Date;
    version: number;
}

export interface NodeTemplate {
    type: NodeType;
    name: string;
    icon: string;
    color: string;
    defaultWidth: number;
    defaultHeight: number;
    inputPorts: number;
    outputPorts: number;
}

export interface ValidationResult {
    valid: boolean;
    errors: ValidationError[];
    warnings: ValidationWarning[];
}

export interface ValidationError {
    type: 'no_start' | 'no_end' | 'orphan_node' | 'cycle' | 'unconnected_port';
    nodeId?: string;
    message: string;
}

export interface ValidationWarning {
    type: 'single_path' | 'dead_end' | 'unused_port';
    nodeId?: string;
    message: string;
}

export interface DragState {
    isDragging: boolean;
    dragType: 'node' | 'connection' | 'canvas' | null;
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
    draggedNodeId?: string;
    draggedPortId?: string;
}

export interface WorkflowBuilderConfig {
    canvasWidth: number;
    canvasHeight: number;
    gridSize: number;
    snapToGrid: boolean;
    showGrid: boolean;
    zoomMin: number;
    zoomMax: number;
    defaultZoom: number;
}

export const DEFAULT_CONFIG: WorkflowBuilderConfig = {
    canvasWidth: 2000,
    canvasHeight: 1500,
    gridSize: 20,
    snapToGrid: true,
    showGrid: true,
    zoomMin: 0.25,
    zoomMax: 2,
    defaultZoom: 1
};

export const NODE_TEMPLATES: NodeTemplate[] = [
    { type: NodeType.ROUTER, name: 'Router', icon: '⚡', color: '#0a84ff', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 3 },
    { type: NodeType.CODE, name: 'Code Agent', icon: '💻', color: '#30d158', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.TRANSLATION, name: 'Translation', icon: '🌐', color: '#ff9f0a', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.RAG, name: 'RAG Engine', icon: '🔍', color: '#bf5af2', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.VALIDATION, name: 'Validation', icon: '✓', color: '#64d2ff', defaultWidth: 140, defaultHeight: 80, inputPorts: 2, outputPorts: 1 },
    { type: NodeType.ORCHESTRATOR, name: 'Orchestrator', icon: '🎯', color: '#ff453a', defaultWidth: 160, defaultHeight: 100, inputPorts: 1, outputPorts: 4 },
    { type: NodeType.MATH, name: 'Math Agent', icon: '🔢', color: '#ff6b6b', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.CREATIVE, name: 'Creative Agent', icon: '✨', color: '#f06595', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.SUMMARIZER, name: 'Summarizer', icon: '📝', color: '#845ef7', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.CLASSIFIER, name: 'Classifier', icon: '🏷️', color: '#20c997', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 2 },
    { type: NodeType.EMBEDDER, name: 'Embedder', icon: '🧬', color: '#fab005', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.SEARCH, name: 'Search Agent', icon: '🔎', color: '#15aabf', defaultWidth: 140, defaultHeight: 80, inputPorts: 1, outputPorts: 1 },
    { type: NodeType.MEMORY, name: 'Memory Agent', icon: '🧠', color: '#e64980', defaultWidth: 140, defaultHeight: 80, inputPorts: 2, outputPorts: 1 },
    { type: NodeType.PLANNER, name: 'Planner', icon: '📋', color: '#7950f2', defaultWidth: 160, defaultHeight: 100, inputPorts: 1, outputPorts: 3 }
];

