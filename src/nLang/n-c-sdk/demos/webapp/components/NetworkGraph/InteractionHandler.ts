/**
 * InteractionHandler - Mouse and touch interactions
 * Drag & drop, zoom, pan, and selection
 */

import { Vector2D, Viewport, NodeEvent, GraphEvent } from './types';
import { GraphNode } from './GraphNode';
import { GraphEdge } from './GraphEdge';

export class InteractionHandler {
    private svg: SVGSVGElement;
    private viewport: Viewport;
    private nodes: Map<string, GraphNode> = new Map();
    private edges: Map<string, GraphEdge> = new Map();
    
    // Interaction state
    private isDragging: boolean = false;
    private isPanning: boolean = false;
    private draggedNode: GraphNode | null = null;
    private dragStart: Vector2D = { x: 0, y: 0 };
    private panStart: Vector2D = { x: 0, y: 0 };
    private lastMousePos: Vector2D = { x: 0, y: 0 };
    
    // Event callbacks
    private onNodeClick: ((event: NodeEvent) => void) | null = null;
    private onNodeDrag: ((event: NodeEvent) => void) | null = null;
    private onViewportChange: ((event: GraphEvent) => void) | null = null;
    
    constructor(svg: SVGSVGElement, viewport: Viewport) {
        this.svg = svg;
        this.viewport = viewport;
        this.setupEventListeners();
    }
    
    // ========================================================================
    // Setup
    // ========================================================================
    
    private setupEventListeners(): void {
        // Mouse events
        this.svg.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.svg.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.svg.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.svg.addEventListener('wheel', this.onWheel.bind(this), { passive: false });
        this.svg.addEventListener('mouseleave', this.onMouseLeave.bind(this));
        
        // Touch events
        this.svg.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: false });
        this.svg.addEventListener('touchmove', this.onTouchMove.bind(this), { passive: false });
        this.svg.addEventListener('touchend', this.onTouchEnd.bind(this));
        
        // Prevent context menu
        this.svg.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    public setNodes(nodes: Map<string, GraphNode>): void {
        this.nodes = nodes;
    }
    
    public setEdges(edges: Map<string, GraphEdge>): void {
        this.edges = edges;
    }
    
    public setViewport(viewport: Viewport): void {
        this.viewport = viewport;
    }
    
    // ========================================================================
    // Event Callbacks
    // ========================================================================
    
    public on(event: string, callback: any): void {
        switch (event) {
            case 'nodeClick':
                this.onNodeClick = callback;
                break;
            case 'nodeDrag':
                this.onNodeDrag = callback;
                break;
            case 'viewportChange':
                this.onViewportChange = callback;
                break;
        }
    }
    
    // ========================================================================
    // Mouse Events
    // ========================================================================
    
    private onMouseDown(event: MouseEvent): void {
        const worldPos = this.screenToWorld({ x: event.clientX, y: event.clientY });
        
        // Check if clicking on a node
        const clickedNode = this.getNodeAtPosition(worldPos);
        
        if (clickedNode) {
            // Start dragging node
            this.isDragging = true;
            this.draggedNode = clickedNode;
            this.dragStart = { ...worldPos };
            clickedNode.setDragging(true);
            
            // Notify
            if (this.onNodeClick) {
                this.onNodeClick({
                    type: 'click',
                    node: clickedNode,
                    position: worldPos,
                    originalEvent: event
                });
            }
        } else {
            // Start panning
            this.isPanning = true;
            this.panStart = {
                x: event.clientX - this.viewport.x,
                y: event.clientY - this.viewport.y
            };
        }
        
        this.lastMousePos = { x: event.clientX, y: event.clientY };
        event.preventDefault();
    }
    
    private onMouseMove(event: MouseEvent): void {
        const worldPos = this.screenToWorld({ x: event.clientX, y: event.clientY });
        
        if (this.isDragging && this.draggedNode) {
            // Drag node
            this.draggedNode.position.x = worldPos.x;
            this.draggedNode.position.y = worldPos.y;
            this.draggedNode.updatePosition();
            
            // Update connected edges
            this.updateEdgesForNode(this.draggedNode);
            
            // Notify
            if (this.onNodeDrag) {
                this.onNodeDrag({
                    type: 'drag',
                    node: this.draggedNode,
                    position: worldPos,
                    originalEvent: event
                });
            }
        } else if (this.isPanning) {
            // Pan viewport
            this.viewport.x = event.clientX - this.panStart.x;
            this.viewport.y = event.clientY - this.panStart.y;
            
            this.updateSVGTransform();
            
            // Notify
            if (this.onViewportChange) {
                this.onViewportChange({
                    type: 'pan',
                    viewport: this.viewport,
                    originalEvent: event
                });
            }
        } else {
            // Hover detection
            const hoveredNode = this.getNodeAtPosition(worldPos);
            
            // Update hover states
            for (const node of this.nodes.values()) {
                node.setHovered(node === hoveredNode);
            }
            
            // Update cursor
            this.svg.style.cursor = hoveredNode ? 'pointer' : (this.isPanning ? 'grabbing' : 'grab');
        }
        
        this.lastMousePos = { x: event.clientX, y: event.clientY };
    }
    
    private onMouseUp(event: MouseEvent): void {
        if (this.draggedNode) {
            this.draggedNode.setDragging(false);
            this.draggedNode.fixed = false;  // Re-enable physics
            this.draggedNode = null;
        }
        
        this.isDragging = false;
        this.isPanning = false;
        this.svg.style.cursor = 'grab';
    }
    
    private onMouseLeave(event: MouseEvent): void {
        this.onMouseUp(event);
        
        // Clear all hover states
        for (const node of this.nodes.values()) {
            node.setHovered(false);
        }
    }
    
    private onWheel(event: WheelEvent): void {
        event.preventDefault();
        
        // Zoom toward mouse position
        const mousePos = { x: event.clientX, y: event.clientY };
        const worldBefore = this.screenToWorld(mousePos);
        
        // Update scale
        const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
        this.viewport.scale = Math.max(0.1, Math.min(5.0, this.viewport.scale * zoomFactor));
        
        // Adjust pan to keep mouse position fixed
        const worldAfter = this.screenToWorld(mousePos);
        this.viewport.x += (worldAfter.x - worldBefore.x) * this.viewport.scale;
        this.viewport.y += (worldAfter.y - worldBefore.y) * this.viewport.scale;
        
        this.updateSVGTransform();
        
        // Notify
        if (this.onViewportChange) {
            this.onViewportChange({
                type: 'zoom',
                viewport: this.viewport,
                originalEvent: event
            });
        }
    }
    
    // ========================================================================
    // Touch Events
    // ========================================================================
    
    private onTouchStart(event: TouchEvent): void {
        event.preventDefault();
        
        if (event.touches.length === 1) {
            // Single touch - treat as mouse down
            const touch = event.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.onMouseDown(mouseEvent);
        } else if (event.touches.length === 2) {
            // Two finger pinch/pan
            this.isPanning = true;
            const center = this.getTouchCenter(event.touches);
            this.panStart = {
                x: center.x - this.viewport.x,
                y: center.y - this.viewport.y
            };
        }
    }
    
    private onTouchMove(event: TouchEvent): void {
        event.preventDefault();
        
        if (event.touches.length === 1 && (this.isDragging || this.isPanning)) {
            // Single touch - treat as mouse move
            const touch = event.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.onMouseMove(mouseEvent);
        } else if (event.touches.length === 2) {
            // Two finger pinch/zoom
            const center = this.getTouchCenter(event.touches);
            
            if (this.isPanning) {
                // Pan
                this.viewport.x = center.x - this.panStart.x;
                this.viewport.y = center.y - this.panStart.y;
                this.updateSVGTransform();
            }
        }
    }
    
    private onTouchEnd(event: TouchEvent): void {
        if (event.touches.length === 0) {
            this.onMouseUp(new MouseEvent('mouseup'));
        }
    }
    
    private getTouchCenter(touches: TouchList): Vector2D {
        let x = 0, y = 0;
        for (let i = 0; i < touches.length; i++) {
            x += touches[i].clientX;
            y += touches[i].clientY;
        }
        return {
            x: x / touches.length,
            y: y / touches.length
        };
    }
    
    // ========================================================================
    // Coordinate Transformation
    // ========================================================================
    
    private screenToWorld(screenPos: Vector2D): Vector2D {
        const rect = this.svg.getBoundingClientRect();
        
        // Convert screen coordinates to SVG coordinates
        return {
            x: (screenPos.x - rect.left - this.viewport.x - rect.width / 2) / this.viewport.scale,
            y: (screenPos.y - rect.top - this.viewport.y - rect.height / 2) / this.viewport.scale
        };
    }
    
    private worldToScreen(worldPos: Vector2D): Vector2D {
        const rect = this.svg.getBoundingClientRect();
        
        return {
            x: worldPos.x * this.viewport.scale + this.viewport.x + rect.width / 2,
            y: worldPos.y * this.viewport.scale + this.viewport.y + rect.height / 2
        };
    }
    
    // ========================================================================
    // Helper Methods
    // ========================================================================
    
    private getNodeAtPosition(worldPos: Vector2D): GraphNode | null {
        // Check nodes in reverse order (top to bottom)
        const nodesArray = Array.from(this.nodes.values());
        
        for (let i = nodesArray.length - 1; i >= 0; i--) {
            if (nodesArray[i].containsPoint(worldPos)) {
                return nodesArray[i];
            }
        }
        
        return null;
    }
    
    private updateEdgesForNode(node: GraphNode): void {
        // Update all edges connected to this node
        for (const edge of this.edges.values()) {
            if (edge.from === node.id || edge.to === node.id) {
                edge.updatePath();
            }
        }
    }
    
    private updateSVGTransform(): void {
        // Apply transform to container group (not SVG itself)
        const container = this.svg.querySelector('.graph-container');
        if (container) {
            (container as SVGGElement).setAttribute(
                'transform',
                `translate(${this.viewport.x}, ${this.viewport.y}) scale(${this.viewport.scale})`
            );
        }
    }
    
    // ========================================================================
    // Public Methods
    // ========================================================================
    
    public zoomIn(): void {
        this.viewport.scale = Math.min(5.0, this.viewport.scale * 1.2);
        this.updateSVGTransform();
        
        if (this.onViewportChange) {
            this.onViewportChange({
                type: 'zoom',
                viewport: this.viewport
            });
        }
    }
    
    public zoomOut(): void {
        this.viewport.scale = Math.max(0.1, this.viewport.scale / 1.2);
        this.updateSVGTransform();
        
        if (this.onViewportChange) {
            this.onViewportChange({
                type: 'zoom',
                viewport: this.viewport
            });
        }
    }
    
    public resetZoom(): void {
        this.viewport.scale = 1.0;
        this.viewport.x = 0;
        this.viewport.y = 0;
        this.updateSVGTransform();
        
        if (this.onViewportChange) {
            this.onViewportChange({
                type: 'zoom',
                viewport: this.viewport
            });
        }
    }
    
    public fitToView(): void {
        if (this.nodes.size === 0) return;
        
        // Calculate bounding box
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        for (const node of this.nodes.values()) {
            minX = Math.min(minX, node.position.x - node.radius);
            maxX = Math.max(maxX, node.position.x + node.radius);
            minY = Math.min(minY, node.position.y - node.radius);
            maxY = Math.max(maxY, node.position.y + node.radius);
        }
        
        const rect = this.svg.getBoundingClientRect();
        const graphWidth = maxX - minX;
        const graphHeight = maxY - minY;
        
        // Calculate scale to fit with padding
        const padding = 100;
        const scaleX = (rect.width - 2 * padding) / graphWidth;
        const scaleY = (rect.height - 2 * padding) / graphHeight;
        
        this.viewport.scale = Math.min(scaleX, scaleY, 2.0);
        
        // Center the graph
        this.viewport.x = -(minX + maxX) / 2 * this.viewport.scale;
        this.viewport.y = -(minY + maxY) / 2 * this.viewport.scale;
        
        this.updateSVGTransform();
        
        if (this.onViewportChange) {
            this.onViewportChange({
                type: 'zoom',
                viewport: this.viewport
            });
        }
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    public destroy(): void {
        // Remove event listeners
        this.svg.removeEventListener('mousedown', this.onMouseDown.bind(this));
        this.svg.removeEventListener('mousemove', this.onMouseMove.bind(this));
        this.svg.removeEventListener('mouseup', this.onMouseUp.bind(this));
        this.svg.removeEventListener('wheel', this.onWheel.bind(this));
        this.svg.removeEventListener('mouseleave', this.onMouseLeave.bind(this));
        this.svg.removeEventListener('touchstart', this.onTouchStart.bind(this));
        this.svg.removeEventListener('touchmove', this.onTouchMove.bind(this));
        this.svg.removeEventListener('touchend', this.onTouchEnd.bind(this));
    }
}
