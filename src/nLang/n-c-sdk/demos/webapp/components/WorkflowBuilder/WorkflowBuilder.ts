/**
 * WorkflowBuilder - Visual drag-and-drop workflow creation component
 * Uses D3.js for rendering
 */

import * as d3 from 'd3';
import {
    WorkflowNode, WorkflowConnection, WorkflowDefinition, NodeType, PortType, Port,
    ValidationResult, ValidationError, ValidationWarning, DragState,
    WorkflowBuilderConfig, DEFAULT_CONFIG, NODE_TEMPLATES, NodeTemplate
} from './types';

export class WorkflowBuilder {
    private container: HTMLElement;
    private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
    private canvas: d3.Selection<SVGGElement, unknown, null, undefined>;
    private config: WorkflowBuilderConfig;
    private nodes: WorkflowNode[] = [];
    private connections: WorkflowConnection[] = [];
    private selectedNodeId: string | null = null;
    private dragState: DragState = { isDragging: false, dragType: null, startX: 0, startY: 0, currentX: 0, currentY: 0 };
    private zoom: d3.ZoomBehavior<SVGSVGElement, unknown>;
    private onChangeCallback: ((workflow: WorkflowDefinition) => void) | null = null;
    private onSelectCallback: ((node: WorkflowNode | null) => void) | null = null;

    constructor(container: HTMLElement, config: Partial<WorkflowBuilderConfig> = {}) {
        this.container = container;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.svg = d3.select(container).append('svg')
            .attr('width', '100%').attr('height', '100%')
            .style('background', '#1c1c1e');
        this.canvas = this.svg.append('g').attr('class', 'canvas');
        this.zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([this.config.zoomMin, this.config.zoomMax])
            .on('zoom', (event) => this.canvas.attr('transform', event.transform));
        this.svg.call(this.zoom);
        this._initGrid();
        this._initConnectionLayer();
        this._initNodeLayer();
        this._initDragHandlers();
    }

    private _initGrid(): void {
        if (!this.config.showGrid) return;
        const defs = this.svg.append('defs');
        defs.append('pattern').attr('id', 'grid').attr('width', this.config.gridSize).attr('height', this.config.gridSize)
            .attr('patternUnits', 'userSpaceOnUse')
            .append('path').attr('d', `M ${this.config.gridSize} 0 L 0 0 0 ${this.config.gridSize}`)
            .attr('fill', 'none').attr('stroke', '#2c2c2e').attr('stroke-width', 0.5);
        this.canvas.append('rect').attr('width', this.config.canvasWidth).attr('height', this.config.canvasHeight)
            .attr('fill', 'url(#grid)');
    }

    private _initConnectionLayer(): void { this.canvas.append('g').attr('class', 'connections'); }
    private _initNodeLayer(): void { this.canvas.append('g').attr('class', 'nodes'); }

    private _initDragHandlers(): void {
        this.svg.on('mouseup', () => this._endDrag());
        this.svg.on('mousemove', (event) => this._onMouseMove(event));
    }

    public addNode(type: NodeType, x: number, y: number): WorkflowNode {
        const template = NODE_TEMPLATES.find(t => t.type === type) || NODE_TEMPLATES[0];
        const snappedX = this.config.snapToGrid ? Math.round(x / this.config.gridSize) * this.config.gridSize : x;
        const snappedY = this.config.snapToGrid ? Math.round(y / this.config.gridSize) * this.config.gridSize : y;
        const node: WorkflowNode = {
            id: `node-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            type, name: template.name, x: snappedX, y: snappedY,
            width: template.defaultWidth, height: template.defaultHeight,
            inputPorts: Array.from({ length: template.inputPorts }, (_, i) => ({ id: `in-${i}`, type: PortType.INPUT, connected: false })),
            outputPorts: Array.from({ length: template.outputPorts }, (_, i) => ({ id: `out-${i}`, type: PortType.OUTPUT, connected: false })),
            properties: {}
        };
        this.nodes.push(node);
        this._renderNode(node);
        this._notifyChange();
        return node;
    }

    private _renderNode(node: WorkflowNode): void {
        const template = NODE_TEMPLATES.find(t => t.type === node.type);
        const nodeGroup = this.canvas.select('.nodes').append('g')
            .attr('class', 'node').attr('data-id', node.id)
            .attr('transform', `translate(${node.x}, ${node.y})`);
        nodeGroup.append('rect').attr('width', node.width).attr('height', node.height)
            .attr('rx', 8).attr('fill', template?.color || '#444').attr('stroke', '#555').attr('stroke-width', 2);
        nodeGroup.append('text').attr('x', node.width / 2).attr('y', 20)
            .attr('text-anchor', 'middle').attr('fill', '#fff').attr('font-size', '12px').text(template?.icon || '⚙');
        nodeGroup.append('text').attr('x', node.width / 2).attr('y', 40)
            .attr('text-anchor', 'middle').attr('fill', '#fff').attr('font-size', '11px').attr('font-weight', 'bold').text(node.name);
        node.inputPorts.forEach((port, i) => this._renderPort(nodeGroup, node, port, i, 'input'));
        node.outputPorts.forEach((port, i) => this._renderPort(nodeGroup, node, port, i, 'output'));
        nodeGroup.call(d3.drag<SVGGElement, unknown>()
            .on('start', (event) => this._startNodeDrag(event, node))
            .on('drag', (event) => this._onNodeDrag(event, node))
            .on('end', () => this._endDrag()));
        nodeGroup.on('click', (event) => { event.stopPropagation(); this.selectNode(node.id); });
    }

    private _renderPort(group: d3.Selection<SVGGElement, unknown, null, undefined>, node: WorkflowNode, port: Port, index: number, type: 'input' | 'output'): void {
        const x = type === 'input' ? 0 : node.width;
        const y = 25 + index * 20;
        const portGroup = group.append('g').attr('class', `port ${type}`).attr('data-port-id', port.id);
        portGroup.append('circle').attr('cx', x).attr('cy', y).attr('r', 6).attr('fill', '#fff').attr('stroke', '#333');
        portGroup.on('mousedown', (event) => { event.stopPropagation(); this._startConnectionDrag(event, node.id, port.id, type); });
    }

    public connect(sourceNodeId: string, sourcePortId: string, targetNodeId: string, targetPortId: string): WorkflowConnection | null {
        if (sourceNodeId === targetNodeId) return null;
        const connection: WorkflowConnection = { id: `conn-${Date.now()}`, sourceNodeId, sourcePortId, targetNodeId, targetPortId };
        this.connections.push(connection);
        this._renderConnection(connection);
        this._notifyChange();
        return connection;
    }

    private _renderConnection(connection: WorkflowConnection): void {
        const sourceNode = this.nodes.find(n => n.id === connection.sourceNodeId);
        const targetNode = this.nodes.find(n => n.id === connection.targetNodeId);
        if (!sourceNode || !targetNode) return;
        const sourcePortIndex = sourceNode.outputPorts.findIndex(p => p.id === connection.sourcePortId);
        const targetPortIndex = targetNode.inputPorts.findIndex(p => p.id === connection.targetPortId);
        const x1 = sourceNode.x + sourceNode.width, y1 = sourceNode.y + 25 + sourcePortIndex * 20;
        const x2 = targetNode.x, y2 = targetNode.y + 25 + targetPortIndex * 20;
        const midX = (x1 + x2) / 2;
        this.canvas.select('.connections').append('path').attr('data-id', connection.id)
            .attr('d', `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`)
            .attr('stroke', '#0a84ff').attr('stroke-width', 2).attr('fill', 'none');
    }

    public validate(): ValidationResult { const errors: ValidationError[] = [], warnings: ValidationWarning[] = [];
        const startNodes = this.nodes.filter(n => n.inputPorts.every(p => !this.connections.some(c => c.targetNodeId === n.id && c.targetPortId === p.id)));
        const endNodes = this.nodes.filter(n => n.outputPorts.every(p => !this.connections.some(c => c.sourceNodeId === n.id && c.sourcePortId === p.id)));
        if (startNodes.length === 0) errors.push({ type: 'no_start', message: 'Workflow has no start node' });
        if (endNodes.length === 0) errors.push({ type: 'no_end', message: 'Workflow has no end node' });
        const connected = new Set<string>(); const visit = (id: string) => { if (connected.has(id)) return; connected.add(id);
            this.connections.filter(c => c.sourceNodeId === id).forEach(c => visit(c.targetNodeId)); };
        startNodes.forEach(n => visit(n.id));
        this.nodes.filter(n => !connected.has(n.id)).forEach(n => errors.push({ type: 'orphan_node', nodeId: n.id, message: `Node "${n.name}" is not connected to workflow` }));
        return { valid: errors.length === 0, errors, warnings }; }

    public exportWorkflow(): WorkflowDefinition { return { id: `wf-${Date.now()}`, name: 'New Workflow', nodes: [...this.nodes], connections: [...this.connections], createdAt: new Date(), updatedAt: new Date(), version: 1 }; }
    public loadWorkflow(workflow: WorkflowDefinition): void { this.clear(); this.nodes = [...workflow.nodes]; this.connections = [...workflow.connections]; this.nodes.forEach(n => this._renderNode(n)); this.connections.forEach(c => this._renderConnection(c)); }
    public clear(): void { this.nodes = []; this.connections = []; this.canvas.select('.nodes').selectAll('*').remove(); this.canvas.select('.connections').selectAll('*').remove(); this.selectedNodeId = null; this._notifyChange(); }
    public selectNode(nodeId: string | null): void { this.selectedNodeId = nodeId; this.canvas.selectAll('.node').classed('selected', false); if (nodeId) this.canvas.select(`[data-id="${nodeId}"]`).classed('selected', true); if (this.onSelectCallback) this.onSelectCallback(this.nodes.find(n => n.id === nodeId) || null); }
    public deleteSelected(): void { if (!this.selectedNodeId) return; this.connections = this.connections.filter(c => c.sourceNodeId !== this.selectedNodeId && c.targetNodeId !== this.selectedNodeId); this.nodes = this.nodes.filter(n => n.id !== this.selectedNodeId); this.canvas.select(`[data-id="${this.selectedNodeId}"]`).remove(); this.canvas.select('.connections').selectAll(`[data-id]`).filter((_, i, nodes) => { const c = this.connections.find(conn => nodes[i].getAttribute('data-id') === conn.id); return !c; }).remove(); this.selectedNodeId = null; this._notifyChange(); }
    public onChange(callback: (workflow: WorkflowDefinition) => void): void { this.onChangeCallback = callback; }
    public onSelect(callback: (node: WorkflowNode | null) => void): void { this.onSelectCallback = callback; }
    private _startNodeDrag(event: d3.D3DragEvent<SVGGElement, unknown, unknown>, node: WorkflowNode): void { this.dragState = { isDragging: true, dragType: 'node', startX: event.x, startY: event.y, currentX: event.x, currentY: event.y, draggedNodeId: node.id }; }
    private _startConnectionDrag(event: MouseEvent, nodeId: string, portId: string, type: string): void { if (type === 'output') this.dragState = { isDragging: true, dragType: 'connection', startX: event.clientX, startY: event.clientY, currentX: event.clientX, currentY: event.clientY, draggedNodeId: nodeId, draggedPortId: portId }; }
    private _onNodeDrag(event: d3.D3DragEvent<SVGGElement, unknown, unknown>, node: WorkflowNode): void { node.x = this.config.snapToGrid ? Math.round(event.x / this.config.gridSize) * this.config.gridSize : event.x; node.y = this.config.snapToGrid ? Math.round(event.y / this.config.gridSize) * this.config.gridSize : event.y; this.canvas.select(`[data-id="${node.id}"]`).attr('transform', `translate(${node.x}, ${node.y})`); this._updateConnections(node.id); }
    private _onMouseMove(event: MouseEvent): void { if (!this.dragState.isDragging) return; this.dragState.currentX = event.clientX; this.dragState.currentY = event.clientY; }
    private _endDrag(): void { this.dragState = { isDragging: false, dragType: null, startX: 0, startY: 0, currentX: 0, currentY: 0 }; this._notifyChange(); }
    private _updateConnections(nodeId: string): void { this.connections.filter(c => c.sourceNodeId === nodeId || c.targetNodeId === nodeId).forEach(c => { this.canvas.select(`.connections [data-id="${c.id}"]`).remove(); this._renderConnection(c); }); }
    private _notifyChange(): void { if (this.onChangeCallback) this.onChangeCallback(this.exportWorkflow()); }
    public getNodeTemplates(): NodeTemplate[] { return NODE_TEMPLATES; }
    public destroy(): void { this.svg.remove(); }
}

export default WorkflowBuilder;

