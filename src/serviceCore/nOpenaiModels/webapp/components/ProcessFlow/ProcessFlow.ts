/**
 * ProcessFlow - Main SAP Process Flow Component
 * 100% Commercial Quality - Exact SAP Fiori behavior
 */

import {
    ProcessFlowNode as NodeConfig,
    ProcessFlowLane as LaneConfig,
    ProcessFlowConnection as ConnectionConfig,
    ProcessFlowConfig,
    ProcessFlowZoomLevel,
    ProcessFlowNodeEvent,
    ProcessFlowLaneEvent,
    ProcessFlowDisplayState,
    DEFAULT_PROCESS_FLOW_CONFIG,
    PROCESS_FLOW_LAYOUT,
    ZOOM_LEVEL_CONFIG
} from './types';
import { ProcessFlowNode } from './ProcessFlowNode';
import { ProcessFlowLane } from './ProcessFlowLane';
import { ProcessFlowConnection } from './ProcessFlowConnection';

export class ProcessFlow {
    private container: HTMLElement;
    private svg!: SVGSVGElement;
    private contentGroup!: SVGGElement;
    private lanesGroup!: SVGGElement;
    private connectionsGroup!: SVGGElement;
    private nodesGroup!: SVGGElement;
    
    // Data
    private nodes: Map<string, ProcessFlowNode> = new Map();
    private lanes: Map<string, ProcessFlowLane> = new Map();
    private connections: Map<string, ProcessFlowConnection> = new Map();
    
    // Configuration
    private config: ProcessFlowConfig;
    
    // Layout
    private lanePositions: Map<string, number> = new Map();
    private columnPositions: number[] = [];
    
    // State
    private selectedNodeId: string | null = null;
    private hoveredNodeId: string | null = null;

    // Events
    private eventListeners: Map<string, Function[]> = new Map();

    // Zoom level management
    private currentZoomLevel: ProcessFlowZoomLevel = ProcessFlowZoomLevel.Two;
    private isAutoZoom: boolean = true;  // Auto-detect zoom based on container size
    private resizeObserver: ResizeObserver | null = null;

    // Overflow indicators
    private leftOverflowIndicator: HTMLDivElement | null = null;
    private rightOverflowIndicator: HTMLDivElement | null = null;

    constructor(container: HTMLElement | string, config?: Partial<ProcessFlowConfig>) {
        // Get container
        if (typeof container === 'string') {
            const el = document.querySelector(container);
            if (!el) throw new Error(`Container ${container} not found`);
            this.container = el as HTMLElement;
        } else {
            this.container = container;
        }
        
        // Merge config
        this.config = { ...DEFAULT_PROCESS_FLOW_CONFIG, ...config };
        
        // Initialize
        this.init();
    }
    
    // ========================================================================
    // Initialization
    // ========================================================================
    
    private init(): void {
        // Create SVG
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('class', 'process-flow-svg');
        this.svg.style.width = '100%';
        this.svg.style.height = '100%';
        this.svg.style.fontFamily = '"72", "72full", Arial, Helvetica, sans-serif';
        
        // Create content group (for zoom/pan)
        this.contentGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.contentGroup.setAttribute('class', 'process-flow-content');
        this.svg.appendChild(this.contentGroup);
        
        // Create layers
        this.lanesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.lanesGroup.setAttribute('class', 'process-flow-lanes');
        this.contentGroup.appendChild(this.lanesGroup);
        
        this.connectionsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.connectionsGroup.setAttribute('class', 'process-flow-connections');
        this.contentGroup.appendChild(this.connectionsGroup);
        
        this.nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.nodesGroup.setAttribute('class', 'process-flow-nodes');
        this.contentGroup.appendChild(this.nodesGroup);
        
        // Add to container
        this.container.appendChild(this.svg);

        // Create overflow indicators
        this.createOverflowIndicators();

        // Setup interactions
        this.setupInteractions();

        // Setup ResizeObserver for auto zoom detection
        this.setupResizeObserver();

        // Set initial zoom level based on config or auto-detect
        if (this.config.zoomLevel) {
            this.isAutoZoom = false;
            this.setZoomLevel(this.config.zoomLevel);
        } else {
            this.isAutoZoom = true;
            this.updateAutoZoomLevel();
        }
    }

    /**
     * Creates overflow indicators that show when process flow extends beyond visible area
     * Format: "< 2" on left, "5 >" on right, showing count of hidden steps
     */
    private createOverflowIndicators(): void {
        // Left overflow indicator
        this.leftOverflowIndicator = document.createElement('div');
        this.leftOverflowIndicator.className = 'process-flow-overflow-indicator process-flow-overflow-left';
        this.leftOverflowIndicator.style.cssText = `
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(10, 110, 209, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 0 4px 4px 0;
            font-family: "72", "72full", Arial, Helvetica, sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: none;
            z-index: 100;
            box-shadow: 2px 0 8px rgba(0,0,0,0.15);
            transition: background 0.2s ease;
        `;
        this.leftOverflowIndicator.addEventListener('mouseenter', () => {
            this.leftOverflowIndicator!.style.background = 'rgba(10, 110, 209, 1)';
        });
        this.leftOverflowIndicator.addEventListener('mouseleave', () => {
            this.leftOverflowIndicator!.style.background = 'rgba(10, 110, 209, 0.9)';
        });
        this.container.appendChild(this.leftOverflowIndicator);

        // Right overflow indicator
        this.rightOverflowIndicator = document.createElement('div');
        this.rightOverflowIndicator.className = 'process-flow-overflow-indicator process-flow-overflow-right';
        this.rightOverflowIndicator.style.cssText = `
            position: absolute;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(10, 110, 209, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 4px 0 0 4px;
            font-family: "72", "72full", Arial, Helvetica, sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: none;
            z-index: 100;
            box-shadow: -2px 0 8px rgba(0,0,0,0.15);
            transition: background 0.2s ease;
        `;
        this.rightOverflowIndicator.addEventListener('mouseenter', () => {
            this.rightOverflowIndicator!.style.background = 'rgba(10, 110, 209, 1)';
        });
        this.rightOverflowIndicator.addEventListener('mouseleave', () => {
            this.rightOverflowIndicator!.style.background = 'rgba(10, 110, 209, 0.9)';
        });
        this.container.appendChild(this.rightOverflowIndicator);

        // Make container position relative for indicators
        if (getComputedStyle(this.container).position === 'static') {
            this.container.style.position = 'relative';
        }
    }

    /**
     * Sets up ResizeObserver to auto-detect zoom level based on container size
     */
    private setupResizeObserver(): void {
        this.resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                if (entry.target === this.container) {
                    if (this.isAutoZoom) {
                        this.updateAutoZoomLevel();
                    }
                    this.updateOverflowIndicators();
                }
            }
        });
        this.resizeObserver.observe(this.container);
    }

    /**
     * Detects and returns the appropriate zoom level based on container width
     * SAP Fiori responsive breakpoints:
     * - >= 1024px: Level Two (Standard)
     * - 600-1023px: Level Three (Reduced)
     * - < 600px: Level Four (Smallest - icon only)
     */
    private detectZoomLevel(): ProcessFlowZoomLevel {
        const width = this.container.clientWidth;
        if (width >= 1024) return ProcessFlowZoomLevel.Two;
        if (width >= 600) return ProcessFlowZoomLevel.Three;
        return ProcessFlowZoomLevel.Four;
    }

    /**
     * Updates zoom level based on auto-detection
     */
    private updateAutoZoomLevel(): void {
        const detectedLevel = this.detectZoomLevel();
        if (detectedLevel !== this.currentZoomLevel) {
            this.applyZoomLevel(detectedLevel);
        }
    }
    
    private setupInteractions(): void {
        // Node click handling
        this.svg.addEventListener('click', (e) => {
            const target = e.target as SVGElement;
            const nodeEl = target.closest('.process-flow-node') as SVGGElement;
            
            if (nodeEl) {
                const nodeId = nodeEl.getAttribute('data-node-id');
                if (nodeId) {
                    this.handleNodeClick(nodeId, e);
                }
            }
        });
        
        // Node hover
        this.svg.addEventListener('mouseover', (e) => {
            const target = e.target as SVGElement;
            const nodeEl = target.closest('.process-flow-node') as SVGGElement;
            
            if (nodeEl) {
                const nodeId = nodeEl.getAttribute('data-node-id');
                if (nodeId) {
                    this.handleNodeHover(nodeId);
                }
            }
        });
        
        this.svg.addEventListener('mouseout', (e) => {
            const target = e.target as SVGElement;
            const nodeEl = target.closest('.process-flow-node') as SVGGElement;
            
            if (nodeEl) {
                this.handleNodeLeave();
            }
        });
        
        // Zoom with mouse wheel (if enabled)
        if (this.config.wheelZoomable) {
            this.svg.addEventListener('wheel', (e) => {
                e.preventDefault();
                this.handleWheel(e);
            });
        }
    }
    
    // ========================================================================
    // Data Loading
    // ========================================================================
    
    public setLanes(lanesConfig: LaneConfig[]): void {
        // Clear existing lanes
        this.lanes.clear();
        this.lanePositions.clear();
        
        // Sort by position
        lanesConfig.sort((a, b) => a.position - b.position);
        
        // Create lanes
        for (const config of lanesConfig) {
            const lane = new ProcessFlowLane(config);
            this.lanes.set(config.id, lane);
            this.lanesGroup.appendChild(lane.element);
        }
        
        this.updateLayout();
    }
    
    public setNodes(nodesConfig: NodeConfig[]): void {
        // Clear existing nodes
        this.nodes.clear();
        
        // Create nodes
        for (const config of nodesConfig) {
            const node = new ProcessFlowNode(config);
            this.nodes.set(config.id, node);
            this.nodesGroup.appendChild(node.element);
        }
        
        this.updateLayout();
    }
    
    public setConnections(connectionsConfig: ConnectionConfig[]): void {
        // Clear existing connections
        this.connections.clear();
        
        // Create connections
        for (const config of connectionsConfig) {
            const connection = new ProcessFlowConnection(config);
            const sourceNode = this.nodes.get(config.from);
            const targetNode = this.nodes.get(config.to);
            
            if (sourceNode && targetNode) {
                connection.setNodes(sourceNode, targetNode);
                this.connections.set(connection.id, connection);
                this.connectionsGroup.appendChild(connection.element);
            }
        }
    }
    
    public loadData(data: { 
        lanes: LaneConfig[], 
        nodes: NodeConfig[], 
        connections: ConnectionConfig[] 
    }): void {
        this.setLanes(data.lanes);
        this.setNodes(data.nodes);
        this.setConnections(data.connections);
    }
    
    // ========================================================================
    // Layout Engine - SAP Standard
    // ========================================================================
    
    private updateLayout(): void {
        // Calculate lane positions
        this.calculateLanePositions();
        
        // Calculate column positions
        this.calculateColumnPositions();
        
        // Position nodes
        this.positionNodes();
        
        // Update connections
        this.updateConnections();
        
        // Update SVG size
        this.updateSVGSize();
    }
    
    private calculateLanePositions(): void {
        const lanesArray = Array.from(this.lanes.values());
        lanesArray.sort((a, b) => a.position - b.position);
        
        let y = PROCESS_FLOW_LAYOUT.spacing.topMargin;
        
        for (const lane of lanesArray) {
            this.lanePositions.set(lane.id, y);
            lane.setPosition(y);
            y += PROCESS_FLOW_LAYOUT.node.height + PROCESS_FLOW_LAYOUT.spacing.vertical;
        }
    }
    
    private calculateColumnPositions(): void {
        // Group nodes by position (column)
        const columns = new Map<number, NodeConfig[]>();
        
        for (const node of this.nodes.values()) {
            const pos = node.position;
            if (!columns.has(pos)) {
                columns.set(pos, []);
            }
            columns.get(pos)!.push(node.toJSON());
        }
        
        // Calculate X positions
        this.columnPositions = [];
        let x = PROCESS_FLOW_LAYOUT.spacing.laneHeader + PROCESS_FLOW_LAYOUT.spacing.leftMargin;
        
        const maxColumn = Math.max(...Array.from(columns.keys()));
        for (let i = 0; i <= maxColumn; i++) {
            this.columnPositions.push(x);
            x += PROCESS_FLOW_LAYOUT.node.width + PROCESS_FLOW_LAYOUT.spacing.horizontal;
        }
    }
    
    private positionNodes(): void {
        for (const node of this.nodes.values()) {
            const laneY = this.lanePositions.get(node.lane);
            const columnX = this.columnPositions[node.position];
            
            if (laneY !== undefined && columnX !== undefined) {
                node.setPosition(columnX, laneY);
            }
        }
    }
    
    private updateConnections(): void {
        for (const connection of this.connections.values()) {
            connection.updatePath();
        }
    }
    
    private updateSVGSize(): void {
        const maxColumn = Math.max(...Array.from(this.nodes.values()).map(n => n.position));
        const width = this.columnPositions[maxColumn] + PROCESS_FLOW_LAYOUT.node.width + PROCESS_FLOW_LAYOUT.spacing.rightMargin;
        
        const maxLane = Array.from(this.lanes.values()).length;
        const height = maxLane * (PROCESS_FLOW_LAYOUT.node.height + PROCESS_FLOW_LAYOUT.spacing.vertical) + 
                      PROCESS_FLOW_LAYOUT.spacing.topMargin + 
                      PROCESS_FLOW_LAYOUT.spacing.bottomMargin;
        
        this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
    
    // ========================================================================
    // Interaction Handlers
    // ========================================================================
    
    private handleNodeClick(nodeId: string, event: MouseEvent): void {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        // Update selection
        if (this.selectedNodeId === nodeId) {
            this.selectedNodeId = null;
            this.clearSelection();
        } else {
            this.selectedNodeId = nodeId;
            this.highlightNode(nodeId);
        }
        
        // Emit event
        this.emit('nodeClick', { type: 'click', node: node.toJSON(), originalEvent: event });
    }
    
    private handleNodeHover(nodeId: string): void {
        if (this.hoveredNodeId === nodeId) return;
        
        this.hoveredNodeId = nodeId;
        const node = this.nodes.get(nodeId);
        if (node) {
            node.setDisplayState(ProcessFlowDisplayState.Highlighted);
        }
    }
    
    private handleNodeLeave(): void {
        if (this.hoveredNodeId) {
            const node = this.nodes.get(this.hoveredNodeId);
            if (node && this.selectedNodeId !== this.hoveredNodeId) {
                node.setDisplayState(ProcessFlowDisplayState.Regular);
            }
            this.hoveredNodeId = null;
        }
    }
    
    private handleWheel(event: WheelEvent): void {
        // Zoom in/out based on wheel direction
        const delta = event.deltaY > 0 ? -1 : 1;
        this.adjustZoom(delta);
    }
    
    // ========================================================================
    // Highlighting & Selection
    // ========================================================================
    
    private highlightNode(nodeId: string): void {
        const selectedNode = this.nodes.get(nodeId);
        if (!selectedNode) return;
        
        // Get connected nodes
        const connectedNodeIds = this.getConnectedNodes(nodeId);
        
        // Update all nodes
        for (const [id, node] of this.nodes) {
            if (id === nodeId) {
                node.setDisplayState(ProcessFlowDisplayState.Selected);
            } else if (connectedNodeIds.has(id)) {
                node.setDisplayState(ProcessFlowDisplayState.Highlighted);
            } else {
                node.setDisplayState(ProcessFlowDisplayState.Dimmed);
            }
        }
    }
    
    private clearSelection(): void {
        for (const node of this.nodes.values()) {
            node.setDisplayState(ProcessFlowDisplayState.Regular);
        }
    }
    
    private getConnectedNodes(nodeId: string): Set<string> {
        const connected = new Set<string>();

        for (const connection of this.connections.values()) {
            if (connection.from === nodeId) {
                connected.add(connection.to);
            }
            if (connection.to === nodeId) {
                connected.add(connection.from);
            }
        }

        return connected;
    }

    // ========================================================================
    // Path Highlighting - Search/Filter Results
    // ========================================================================

    /**
     * Highlights a path of nodes and their connecting lines
     * @param nodeIds Array of node IDs that form the path to highlight
     */
    public highlightPath(nodeIds: string[]): void {
        const pathNodeSet = new Set(nodeIds);

        // First, dim all non-path elements
        this.dimNonPathElements(nodeIds);

        // Highlight the path nodes
        for (const nodeId of nodeIds) {
            const node = this.nodes.get(nodeId);
            if (node) {
                node.setHighlighted(true);
            }
        }

        // Highlight connections between path nodes
        for (const connection of this.connections.values()) {
            if (pathNodeSet.has(connection.from) && pathNodeSet.has(connection.to)) {
                connection.setHighlighted(true);
            }
        }
    }

    /**
     * Removes all path highlighting, restoring normal display
     */
    public clearHighlight(): void {
        // Clear highlight from all nodes
        for (const node of this.nodes.values()) {
            node.setHighlighted(false);
            node.setDimmed(false);
        }

        // Clear highlight from all connections
        for (const connection of this.connections.values()) {
            connection.setHighlighted(false);
            connection.setDimmed(false);
        }
    }

    /**
     * Dims elements that are not part of the highlighted path
     * @param pathNodeIds Array of node IDs that are in the path (should not be dimmed)
     */
    public dimNonPathElements(pathNodeIds: string[]): void {
        const pathNodeSet = new Set(pathNodeIds);

        // Dim nodes not in the path
        for (const [nodeId, node] of this.nodes) {
            if (!pathNodeSet.has(nodeId)) {
                node.setDimmed(true);
            } else {
                node.setDimmed(false);
            }
        }

        // Dim connections not between path nodes
        for (const connection of this.connections.values()) {
            const isPathConnection = pathNodeSet.has(connection.from) && pathNodeSet.has(connection.to);
            if (!isPathConnection) {
                connection.setDimmed(true);
            } else {
                connection.setDimmed(false);
            }
        }
    }
    
    // ========================================================================
    // Zoom Control - SAP Fiori Semantic Zoom
    // ========================================================================

    /**
     * Sets the zoom level manually, disabling auto-detection
     * @param level The zoom level to set (One, Two, Three, or Four)
     */
    public setZoomLevel(level: ProcessFlowZoomLevel): void {
        this.isAutoZoom = false;
        this.applyZoomLevel(level);
    }

    /**
     * Gets the current zoom level
     */
    public getZoomLevel(): ProcessFlowZoomLevel {
        return this.currentZoomLevel;
    }

    /**
     * Enables auto zoom detection based on container size
     */
    public enableAutoZoom(): void {
        this.isAutoZoom = true;
        this.updateAutoZoomLevel();
    }

    /**
     * Applies the zoom level to all nodes and updates layout
     */
    private applyZoomLevel(level: ProcessFlowZoomLevel): void {
        this.currentZoomLevel = level;
        this.config.zoomLevel = level;

        const zoomConfig = ZOOM_LEVEL_CONFIG[level];

        // Update all nodes with new zoom level
        for (const node of this.nodes.values()) {
            node.setZoomLevel(level);
        }

        // Apply scale transform
        this.contentGroup.setAttribute('transform', `scale(${zoomConfig.scale})`);
        this.contentGroup.setAttribute('data-zoom', level);

        // Re-layout with new node sizes
        this.updateLayout();

        // Update overflow indicators
        this.updateOverflowIndicators();

        // Emit zoom change event
        this.emit('zoomChange', { level, config: zoomConfig });
    }

    /**
     * Cycles through zoom levels using mouse wheel
     * Levels cycle: Four -> Three -> Two -> One (zoom in)
     *              One -> Two -> Three -> Four (zoom out)
     */
    private adjustZoom(delta: number): void {
        const levels = [
            ProcessFlowZoomLevel.Four,
            ProcessFlowZoomLevel.Three,
            ProcessFlowZoomLevel.Two,
            ProcessFlowZoomLevel.One
        ];

        const currentIndex = levels.indexOf(this.currentZoomLevel);
        const newIndex = Math.max(0, Math.min(levels.length - 1, currentIndex + delta));

        if (levels[newIndex] !== this.currentZoomLevel) {
            this.isAutoZoom = false;  // Manual zoom disables auto
            this.applyZoomLevel(levels[newIndex]);
        }
    }

    /**
     * Updates overflow indicators to show count of hidden nodes
     */
    private updateOverflowIndicators(): void {
        if (!this.leftOverflowIndicator || !this.rightOverflowIndicator) return;

        const containerRect = this.container.getBoundingClientRect();
        const zoomConfig = ZOOM_LEVEL_CONFIG[this.currentZoomLevel];
        const scale = zoomConfig.scale;

        // Calculate visible area in SVG coordinates
        const visibleLeft = 0;
        const visibleRight = containerRect.width / scale;

        let hiddenLeft = 0;
        let hiddenRight = 0;

        // Count nodes outside visible area
        for (const node of this.nodes.values()) {
            const nodeRight = node.x + node.getWidth();
            const nodeLeft = node.x;

            if (nodeRight < visibleLeft) {
                hiddenLeft++;
            } else if (nodeLeft > visibleRight) {
                hiddenRight++;
            }
        }

        // Update left indicator
        if (hiddenLeft > 0) {
            this.leftOverflowIndicator.textContent = `< ${hiddenLeft}`;
            this.leftOverflowIndicator.style.display = 'block';
        } else {
            this.leftOverflowIndicator.style.display = 'none';
        }

        // Update right indicator
        if (hiddenRight > 0) {
            this.rightOverflowIndicator.textContent = `${hiddenRight} >`;
            this.rightOverflowIndicator.style.display = 'block';
        } else {
            this.rightOverflowIndicator.style.display = 'none';
        }
    }
    
    // ========================================================================
    // Event System
    // ========================================================================
    
    public on(event: string, callback: Function): void {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event)!.push(callback);
    }
    
    public off(event: string, callback?: Function): void {
        if (!this.eventListeners.has(event)) return;
        
        if (callback) {
            const callbacks = this.eventListeners.get(event)!;
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        } else {
            this.eventListeners.delete(event);
        }
    }
    
    private emit(event: string, data: any): void {
        if (!this.eventListeners.has(event)) return;
        
        for (const callback of this.eventListeners.get(event)!) {
            callback(data);
        }
    }
    
    // ========================================================================
    // Public API
    // ========================================================================
    
    public getNode(nodeId: string): ProcessFlowNode | undefined {
        return this.nodes.get(nodeId);
    }
    
    public selectNode(nodeId: string | null): void {
        if (nodeId === null) {
            this.selectedNodeId = null;
            this.clearSelection();
        } else {
            this.selectedNodeId = nodeId;
            this.highlightNode(nodeId);
        }
    }
    
    public exportData(): any {
        return {
            lanes: Array.from(this.lanes.values()).map(l => ({ id: l.id, label: l.label, position: l.position })),
            nodes: Array.from(this.nodes.values()).map(n => n.toJSON()),
            connections: Array.from(this.connections.values()).map(c => ({ from: c.from, to: c.to, state: c.state, type: c.type }))
        };
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    public destroy(): void {
        // Disconnect ResizeObserver
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }

        // Remove overflow indicators
        if (this.leftOverflowIndicator && this.leftOverflowIndicator.parentNode) {
            this.leftOverflowIndicator.parentNode.removeChild(this.leftOverflowIndicator);
        }
        if (this.rightOverflowIndicator && this.rightOverflowIndicator.parentNode) {
            this.rightOverflowIndicator.parentNode.removeChild(this.rightOverflowIndicator);
        }

        // Clear data
        for (const node of this.nodes.values()) {
            node.destroy();
        }
        for (const lane of this.lanes.values()) {
            lane.destroy();
        }
        for (const connection of this.connections.values()) {
            connection.destroy();
        }

        this.nodes.clear();
        this.lanes.clear();
        this.connections.clear();

        // Remove SVG
        if (this.svg && this.svg.parentNode) {
            this.svg.parentNode.removeChild(this.svg);
        }
    }
}
