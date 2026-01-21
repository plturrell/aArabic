/**
 * NetworkGraph - Main component orchestrator
 * Professional-grade network visualization with 100% SAP feature parity
 */
import { LayoutType, NodeStatus, EdgeStatus, DEFAULT_VIEWPORT, SAP_COLORS } from './types';
import { GraphNode } from './GraphNode';
import { GraphEdge } from './GraphEdge';
import { LayoutEngine } from './LayoutEngine';
import { InteractionHandler } from './InteractionHandler';
import { Minimap } from './Minimap';
import { SearchFilter } from './SearchFilter';
import { HistoryManager } from './HistoryManager';
import { MultiSelectHandler } from './MultiSelectHandler';
export class NetworkGraph {
    constructor(container) {
        // Sub-components
        this.minimap = null;
        // WebSocket for real-time updates
        this.ws = null;
        // Event callbacks
        this.eventHandlers = new Map();
        // Get container element
        if (typeof container === 'string') {
            const element = document.querySelector(container);
            if (!element)
                throw new Error(`Container not found: ${container}`);
            this.container = element;
        }
        else {
            this.container = container;
        }
        // Initialize state
        this.state = {
            nodes: new Map(),
            edges: new Map(),
            groups: new Map(),
            selectedNodeId: null,
            hoveredNodeId: null,
            viewport: { ...DEFAULT_VIEWPORT },
            layout: LayoutType.ForceDirected
        };
        // Create SVG canvas
        this.svg = this.createSVG();
        this.defsElement = this.createDefs();
        this.graphContainer = this.createGraphContainer();
        this.svg.appendChild(this.defsElement);
        this.svg.appendChild(this.graphContainer);
        this.container.appendChild(this.svg);
        // Initialize subsystems
        this.layoutEngine = new LayoutEngine();
        this.interactionHandler = new InteractionHandler(this.svg, this.state.viewport);
        // Setup event handlers
        this.setupEventHandlers();
        // Initialize sub-components
        this.searchFilter = new SearchFilter();
        this.historyManager = new HistoryManager();
        this.multiSelectHandler = new MultiSelectHandler(this.svg);
        // Wire up sub-component callbacks
        this.setupSubComponentCallbacks();
        // Update viewport size
        this.updateViewportSize();
        window.addEventListener('resize', () => this.updateViewportSize());
    }
    // ========================================================================
    // Initialization
    // ========================================================================
    createSVG() {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'network-graph');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.background = SAP_COLORS.background;
        svg.style.cursor = 'grab';
        return svg;
    }
    createDefs() {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        // Drop shadow filter for nodes
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'nodeShadow');
        filter.setAttribute('x', '-50%');
        filter.setAttribute('y', '-50%');
        filter.setAttribute('width', '200%');
        filter.setAttribute('height', '200%');
        const feGaussianBlur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
        feGaussianBlur.setAttribute('in', 'SourceAlpha');
        feGaussianBlur.setAttribute('stdDeviation', '3');
        const feOffset = document.createElementNS('http://www.w3.org/2000/svg', 'feOffset');
        feOffset.setAttribute('dx', '0');
        feOffset.setAttribute('dy', '2');
        feOffset.setAttribute('result', 'offsetblur');
        const feComponentTransfer = document.createElementNS('http://www.w3.org/2000/svg', 'feComponentTransfer');
        const feFuncA = document.createElementNS('http://www.w3.org/2000/svg', 'feFuncA');
        feFuncA.setAttribute('type', 'linear');
        feFuncA.setAttribute('slope', '0.3');
        feComponentTransfer.appendChild(feFuncA);
        const feMerge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
        const feMergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        const feMergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        feMergeNode2.setAttribute('in', 'SourceGraphic');
        feMerge.appendChild(feMergeNode1);
        feMerge.appendChild(feMergeNode2);
        filter.appendChild(feGaussianBlur);
        filter.appendChild(feOffset);
        filter.appendChild(feComponentTransfer);
        filter.appendChild(feMerge);
        defs.appendChild(filter);
        return defs;
    }
    createGraphContainer() {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'graph-container');
        return g;
    }
    setupEventHandlers() {
        // Node interactions
        this.interactionHandler.on('nodeClick', (event) => {
            this.selectNode(event.node.id);
            this.emit('nodeClick', event);
        });
        this.interactionHandler.on('nodeDrag', (event) => {
            this.emit('nodeDrag', event);
        });
        this.interactionHandler.on('viewportChange', (event) => {
            // Update minimap when viewport changes
            if (this.minimap) {
                this.minimap.setViewport(event.viewport);
                this.minimap.render();
            }
            this.emit('viewportChange', event);
        });
    }
    setupSubComponentCallbacks() {
        // History change callback
        this.historyManager.onChange(() => {
            this.emit('historyChange', {
                canUndo: this.historyManager.canUndo(),
                canRedo: this.historyManager.canRedo()
            });
        });
        // Multi-select callback
        this.multiSelectHandler.onSelection((selectedIds) => {
            this.emit('multiSelectionChange', { selectedIds });
        });
        // Search filter callbacks
        this.searchFilter.onFilter((filteredIds) => {
            this.emit('filterChange', { filteredIds });
        });
        this.searchFilter.onHighlight((highlightedIds) => {
            this.emit('highlightChange', { highlightedIds });
        });
    }
    // ========================================================================
    // Public API - Data Management
    // ========================================================================
    addNode(config) {
        const node = new GraphNode(config);
        this.state.nodes.set(node.id, node);
        this.graphContainer.appendChild(node.element);
        // Update interaction handler
        this.interactionHandler.setNodes(this.state.nodes);
        // Update sub-components
        this.updateSubComponentsData();
        this.emit('nodeAdded', { node });
    }
    addEdge(config) {
        const edge = new GraphEdge(config);
        this.state.edges.set(edge.id, edge);
        // Insert edge before nodes (so nodes are on top)
        const firstNode = this.graphContainer.querySelector('.graph-node');
        if (firstNode) {
            this.graphContainer.insertBefore(edge.element, firstNode);
        }
        else {
            this.graphContainer.appendChild(edge.element);
        }
        // Link to nodes
        const sourceNode = this.state.nodes.get(edge.from);
        const targetNode = this.state.nodes.get(edge.to);
        if (sourceNode && targetNode) {
            edge.setNodes(sourceNode, targetNode);
        }
        // Update interaction handler
        this.interactionHandler.setEdges(this.state.edges);
        // Update sub-components
        this.updateSubComponentsData();
        this.emit('edgeAdded', { edge });
    }
    removeNode(nodeId) {
        const node = this.state.nodes.get(nodeId);
        if (!node)
            return;
        // Remove connected edges
        for (const [edgeId, edge] of this.state.edges) {
            if (edge.from === nodeId || edge.to === nodeId) {
                this.removeEdge(edgeId);
            }
        }
        // Remove node
        node.destroy();
        this.state.nodes.delete(nodeId);
        // Update sub-components
        this.updateSubComponentsData();
        this.emit('nodeRemoved', { nodeId });
    }
    removeEdge(edgeId) {
        const edge = this.state.edges.get(edgeId);
        if (!edge)
            return;
        edge.destroy();
        this.state.edges.delete(edgeId);
        // Update sub-components
        this.updateSubComponentsData();
        this.emit('edgeRemoved', { edgeId });
    }
    clear() {
        // Clear all nodes
        for (const node of this.state.nodes.values()) {
            node.destroy();
        }
        this.state.nodes.clear();
        // Clear all edges
        for (const edge of this.state.edges.values()) {
            edge.destroy();
        }
        this.state.edges.clear();
        // Update sub-components
        this.updateSubComponentsData();
        this.emit('cleared', {});
    }
    updateSubComponentsData() {
        const nodesArray = Array.from(this.state.nodes.values());
        const edgesArray = Array.from(this.state.edges.values());
        // Update searchFilter
        this.searchFilter.setNodes(nodesArray);
        this.searchFilter.setEdges(edgesArray);
        // Update minimap if exists
        if (this.minimap) {
            this.minimap.setNodes(nodesArray);
            this.minimap.setEdges(edgesArray);
            this.minimap.render();
        }
    }
    // ========================================================================
    // Public API - Layout
    // ========================================================================
    setLayout(layoutType) {
        this.state.layout = layoutType;
        this.layoutEngine.setConfig({ type: layoutType });
        this.applyLayout();
    }
    applyLayout() {
        this.layoutEngine.setNodes(Array.from(this.state.nodes.values()));
        this.layoutEngine.setEdges(Array.from(this.state.edges.values()));
        this.layoutEngine.start();
        this.emit('layoutApplied', { layout: this.state.layout });
    }
    centerGraph() {
        this.layoutEngine.setNodes(Array.from(this.state.nodes.values()));
        this.layoutEngine.setEdges(Array.from(this.state.edges.values()));
        this.layoutEngine.centerGraph();
    }
    fitToView() {
        this.interactionHandler.fitToView();
    }
    // ========================================================================
    // Public API - Node Operations
    // ========================================================================
    updateNodeStatus(nodeId, status) {
        const node = this.state.nodes.get(nodeId);
        if (node) {
            node.setStatus(status);
            this.emit('nodeStatusChanged', { nodeId, status });
        }
    }
    selectNode(nodeId) {
        // Deselect previous
        if (this.state.selectedNodeId) {
            const prevNode = this.state.nodes.get(this.state.selectedNodeId);
            if (prevNode)
                prevNode.setSelected(false);
        }
        // Select new
        this.state.selectedNodeId = nodeId;
        if (nodeId) {
            const node = this.state.nodes.get(nodeId);
            if (node)
                node.setSelected(true);
        }
        this.emit('selectionChanged', { nodeId });
    }
    getNode(nodeId) {
        return this.state.nodes.get(nodeId);
    }
    getSelectedNode() {
        return this.state.selectedNodeId
            ? this.state.nodes.get(this.state.selectedNodeId) || null
            : null;
    }
    // ========================================================================
    // Public API - Edge Operations
    // ========================================================================
    updateEdgeStatus(edgeId, status) {
        const edge = this.state.edges.get(edgeId);
        if (edge) {
            edge.setStatus(status);
            this.emit('edgeStatusChanged', { edgeId, status });
        }
    }
    getEdge(edgeId) {
        return this.state.edges.get(edgeId);
    }
    // ========================================================================
    // Public API - Minimap
    // ========================================================================
    enableMinimap() {
        if (this.minimap)
            return;
        this.minimap = new Minimap(this.container, this.state.viewport);
        // Set initial data
        this.minimap.setNodes(Array.from(this.state.nodes.values()));
        this.minimap.setEdges(Array.from(this.state.edges.values()));
        // Wire up viewport change callback
        this.minimap.onViewportChange((viewport) => {
            this.interactionHandler.setViewport(viewport);
            this.emit('viewportChange', { viewport });
        });
        this.minimap.show();
        this.minimap.render();
    }
    disableMinimap() {
        if (!this.minimap)
            return;
        this.minimap.destroy();
        this.minimap = null;
    }
    // ========================================================================
    // Public API - Search & Filter
    // ========================================================================
    search(query) {
        return this.searchFilter.search(query);
    }
    filter(criteria) {
        this.searchFilter.applyFilter(criteria);
    }
    clearFilter() {
        this.searchFilter.clearFilter();
    }
    // ========================================================================
    // Public API - History (Undo/Redo)
    // ========================================================================
    undo() {
        this.historyManager.undo();
    }
    redo() {
        this.historyManager.redo();
    }
    canUndo() {
        return this.historyManager.canUndo();
    }
    canRedo() {
        return this.historyManager.canRedo();
    }
    // ========================================================================
    // Public API - Multi-Select
    // ========================================================================
    getMultiSelectHandler() {
        return this.multiSelectHandler;
    }
    getHistoryManager() {
        return this.historyManager;
    }
    getSearchFilter() {
        return this.searchFilter;
    }
    // ========================================================================
    // Public API - Viewport
    // ========================================================================
    zoomIn() {
        this.interactionHandler.zoomIn();
    }
    zoomOut() {
        this.interactionHandler.zoomOut();
    }
    resetZoom() {
        this.interactionHandler.resetZoom();
    }
    getViewport() {
        return { ...this.state.viewport };
    }
    // ========================================================================
    // Data Loading
    // ========================================================================
    async loadFromAPI(apiUrl) {
        try {
            const response = await fetch(apiUrl);
            if (!response.ok)
                throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            this.loadData(data);
            this.emit('dataLoaded', { source: apiUrl });
        }
        catch (error) {
            console.error('Failed to load from API:', error);
            this.emit('error', { error, source: 'loadFromAPI' });
        }
    }
    loadData(data) {
        this.clear();
        // Load nodes
        if (data.agents) {
            for (const agentData of data.agents) {
                this.addNode({
                    id: agentData.id,
                    name: agentData.name,
                    description: agentData.description,
                    type: agentData.type,
                    status: this.mapStatus(agentData.status),
                    model: agentData.model_id,
                    metrics: {
                        totalRequests: agentData.total_requests || 0,
                        avgLatency: agentData.avg_latency || 0,
                        successRate: agentData.success_rate || 0
                    }
                });
            }
        }
        // Load edges from next_agents
        if (data.agents) {
            for (const agentData of data.agents) {
                if (agentData.next_agents) {
                    for (const targetId of agentData.next_agents) {
                        const edgeId = `${agentData.id}-${targetId}`;
                        this.addEdge({
                            id: edgeId,
                            from: agentData.id,
                            to: targetId,
                            status: EdgeStatus.Active
                        });
                    }
                }
            }
        }
        // Apply layout
        setTimeout(() => {
            this.applyLayout();
            setTimeout(() => this.fitToView(), 1000);
        }, 100);
    }
    mapStatus(status) {
        const statusMap = {
            'healthy': NodeStatus.Success,
            'warning': NodeStatus.Warning,
            'error': NodeStatus.Error,
            'running': NodeStatus.Running,
            'busy': NodeStatus.Warning
        };
        return statusMap[status?.toLowerCase()] || NodeStatus.None;
    }
    // ========================================================================
    // WebSocket Real-Time Updates
    // ========================================================================
    connectWebSocket(url) {
        if (this.ws) {
            this.ws.close();
        }
        this.ws = new WebSocket(url);
        this.ws.onopen = () => {
            console.log('✅ NetworkGraph WebSocket connected');
            this.emit('wsConnected', {});
        };
        this.ws.onmessage = (event) => {
            try {
                const update = JSON.parse(event.data);
                this.handleWSUpdate(update);
            }
            catch (error) {
                console.error('Failed to parse WS message:', error);
            }
        };
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('wsError', { error });
        };
        this.ws.onclose = () => {
            console.log('WebSocket closed');
            this.emit('wsDisconnected', {});
        };
    }
    handleWSUpdate(update) {
        if (update.type === 'agent_status') {
            this.updateNodeStatus(update.agent_id, this.mapStatus(update.status));
        }
        else if (update.type === 'workflow_step') {
            const edgeId = `${update.from}-${update.to}`;
            const edge = this.state.edges.get(edgeId);
            if (edge) {
                edge.flash();
            }
        }
        this.emit('wsUpdate', update);
    }
    disconnectWebSocket() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
    // ========================================================================
    // Event System
    // ========================================================================
    on(event, callback) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(callback);
    }
    off(event, callback) {
        if (!callback) {
            this.eventHandlers.delete(event);
        }
        else {
            const handlers = this.eventHandlers.get(event);
            if (handlers) {
                const index = handlers.indexOf(callback);
                if (index !== -1)
                    handlers.splice(index, 1);
            }
        }
    }
    emit(event, data) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            for (const handler of handlers) {
                handler(data);
            }
        }
    }
    // ========================================================================
    // Export/Import
    // ========================================================================
    exportData() {
        return {
            nodes: Array.from(this.state.nodes.values()).map(n => n.toJSON()),
            edges: Array.from(this.state.edges.values()).map(e => e.toJSON()),
            viewport: this.state.viewport,
            layout: this.state.layout
        };
    }
    exportImage() {
        // Convert SVG to data URL
        const svgData = new XMLSerializer().serializeToString(this.svg);
        return 'data:image/svg+xml;base64,' + btoa(svgData);
    }
    // ========================================================================
    // Utilities
    // ========================================================================
    updateViewportSize() {
        const rect = this.container.getBoundingClientRect();
        this.state.viewport.width = rect.width;
        this.state.viewport.height = rect.height;
        this.interactionHandler.setViewport(this.state.viewport);
    }
    getStats() {
        return {
            nodeCount: this.state.nodes.size,
            edgeCount: this.state.edges.size,
            viewport: this.state.viewport,
            layout: this.state.layout
        };
    }
    // ========================================================================
    // Cleanup
    // ========================================================================
    destroy() {
        this.disconnectWebSocket();
        this.clear();
        this.interactionHandler.destroy();
        this.layoutEngine.stop();
        // Destroy sub-components
        if (this.minimap) {
            this.minimap.destroy();
            this.minimap = null;
        }
        this.multiSelectHandler.destroy();
        if (this.svg.parentNode) {
            this.svg.parentNode.removeChild(this.svg);
        }
        this.eventHandlers.clear();
    }
}
// Export for global use
if (typeof window !== 'undefined') {
    window.NetworkGraph = NetworkGraph;
}
//# sourceMappingURL=NetworkGraph.js.map