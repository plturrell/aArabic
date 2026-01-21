/**
 * Minimap - Overview navigation widget
 * Shows entire graph with viewport indicator
 */
import { SAP_COLORS } from './types';
export class Minimap {
    constructor(container, viewport) {
        this.width = 200;
        this.height = 150;
        this.visible = true;
        // Graph data
        this.nodes = [];
        this.edges = [];
        // Interaction
        this.isDragging = false;
        // Callbacks
        this.onViewportMove = null;
        this.container = container;
        this.viewport = viewport;
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.canvas.className = 'minimap-canvas';
        this.canvas.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            border: 2px solid ${SAP_COLORS.border};
            border-radius: 4px;
            background: ${SAP_COLORS.backgroundAlt};
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            cursor: pointer;
        `;
        const ctx = this.canvas.getContext('2d');
        if (!ctx)
            throw new Error('Canvas 2D context not available');
        this.ctx = ctx;
        this.container.appendChild(this.canvas);
        // Setup interactions
        this.setupEventListeners();
    }
    // ========================================================================
    // Event Listeners
    // ========================================================================
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));
    }
    onMouseDown(event) {
        this.isDragging = true;
        this.updateViewportFromClick(event);
    }
    onMouseMove(event) {
        if (this.isDragging) {
            this.updateViewportFromClick(event);
        }
    }
    onMouseUp(event) {
        this.isDragging = false;
    }
    updateViewportFromClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;
        // Convert to world coordinates
        const bounds = this.calculateGraphBounds();
        const worldX = bounds.minX + (clickX / this.width) * (bounds.maxX - bounds.minX);
        const worldY = bounds.minY + (clickY / this.height) * (bounds.maxY - bounds.minY);
        // Update viewport to center on clicked point
        this.viewport.x = -worldX * this.viewport.scale;
        this.viewport.y = -worldY * this.viewport.scale;
        if (this.onViewportMove) {
            this.onViewportMove(this.viewport);
        }
        this.render();
    }
    // ========================================================================
    // Data Update
    // ========================================================================
    setNodes(nodes) {
        this.nodes = nodes;
    }
    setEdges(edges) {
        this.edges = edges;
    }
    setViewport(viewport) {
        this.viewport = viewport;
    }
    // ========================================================================
    // Rendering
    // ========================================================================
    render() {
        if (!this.visible)
            return;
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);
        if (this.nodes.length === 0)
            return;
        // Calculate graph bounds
        const bounds = this.calculateGraphBounds();
        // Draw edges
        this.ctx.strokeStyle = SAP_COLORS.neutral;
        this.ctx.lineWidth = 1;
        for (const edge of this.edges) {
            const source = this.nodes.find(n => n.id === edge.from);
            const target = this.nodes.find(n => n.id === edge.to);
            if (source && target) {
                const startX = this.worldToMinimapX(source.position.x, bounds);
                const startY = this.worldToMinimapY(source.position.y, bounds);
                const endX = this.worldToMinimapX(target.position.x, bounds);
                const endY = this.worldToMinimapY(target.position.y, bounds);
                this.ctx.beginPath();
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(endX, endY);
                this.ctx.stroke();
            }
        }
        // Draw nodes
        for (const node of this.nodes) {
            const x = this.worldToMinimapX(node.position.x, bounds);
            const y = this.worldToMinimapY(node.position.y, bounds);
            const radius = 3;
            // Node color based on status
            this.ctx.fillStyle = this.getStatusColor(node);
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            this.ctx.fill();
        }
        // Draw viewport rectangle
        this.drawViewportRect(bounds);
    }
    drawViewportRect(bounds) {
        // Calculate visible area in world coordinates
        const viewWidth = this.viewport.width / this.viewport.scale;
        const viewHeight = this.viewport.height / this.viewport.scale;
        const centerX = -this.viewport.x / this.viewport.scale;
        const centerY = -this.viewport.y / this.viewport.scale;
        const x1 = this.worldToMinimapX(centerX - viewWidth / 2, bounds);
        const y1 = this.worldToMinimapY(centerY - viewHeight / 2, bounds);
        const x2 = this.worldToMinimapX(centerX + viewWidth / 2, bounds);
        const y2 = this.worldToMinimapY(centerY + viewHeight / 2, bounds);
        // Draw rectangle
        this.ctx.strokeStyle = SAP_COLORS.brand;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    // ========================================================================
    // Coordinate Conversion
    // ========================================================================
    calculateGraphBounds() {
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (const node of this.nodes) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
        }
        // Add padding
        const padding = 50;
        return {
            minX: minX - padding,
            maxX: maxX + padding,
            minY: minY - padding,
            maxY: maxY + padding
        };
    }
    worldToMinimapX(worldX, bounds) {
        const rangeX = bounds.maxX - bounds.minX;
        return ((worldX - bounds.minX) / rangeX) * this.width;
    }
    worldToMinimapY(worldY, bounds) {
        const rangeY = bounds.maxY - bounds.minY;
        return ((worldY - bounds.minY) / rangeY) * this.height;
    }
    getStatusColor(node) {
        // Return simplified colors for minimap
        return SAP_COLORS.brand;
    }
    // ========================================================================
    // Visibility
    // ========================================================================
    show() {
        this.visible = true;
        this.canvas.style.display = 'block';
        this.render();
    }
    hide() {
        this.visible = false;
        this.canvas.style.display = 'none';
    }
    toggle() {
        if (this.visible) {
            this.hide();
        }
        else {
            this.show();
        }
    }
    // ========================================================================
    // Callbacks
    // ========================================================================
    onViewportChange(callback) {
        this.onViewportMove = callback;
    }
    // ========================================================================
    // Cleanup
    // ========================================================================
    destroy() {
        if (this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}
//# sourceMappingURL=Minimap.js.map