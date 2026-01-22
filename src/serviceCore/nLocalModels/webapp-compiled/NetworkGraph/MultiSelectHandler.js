/**
 * MultiSelectHandler - Advanced selection features
 * Lasso tool, rubber band, shift+click multi-select
 */
export class MultiSelectHandler {
    constructor(svg) {
        this.selectedNodes = new Set();
        // Lasso selection
        this.lassoActive = false;
        this.lassoPath = null;
        this.lassoPoints = [];
        // Rubber band selection
        this.rubberBandActive = false;
        this.rubberBandRect = null;
        this.rubberBandStart = { x: 0, y: 0 };
        // Selection modes
        this.selectionMode = 'single';
        // Callbacks
        this.onSelectionChange = null;
        this.svg = svg;
        this.createSelectionElements();
    }
    // ========================================================================
    // Initialization
    // ========================================================================
    createSelectionElements() {
        // Create lasso path
        this.lassoPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this.lassoPath.setAttribute('class', 'lasso-path');
        this.lassoPath.setAttribute('fill', 'rgba(0, 112, 242, 0.1)');
        this.lassoPath.setAttribute('stroke', '#0070f2');
        this.lassoPath.setAttribute('stroke-width', '2');
        this.lassoPath.setAttribute('stroke-dasharray', '5,5');
        this.lassoPath.setAttribute('visibility', 'hidden');
        this.svg.appendChild(this.lassoPath);
        // Create rubber band rectangle
        this.rubberBandRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        this.rubberBandRect.setAttribute('class', 'rubber-band');
        this.rubberBandRect.setAttribute('fill', 'rgba(0, 112, 242, 0.1)');
        this.rubberBandRect.setAttribute('stroke', '#0070f2');
        this.rubberBandRect.setAttribute('stroke-width', '2');
        this.rubberBandRect.setAttribute('stroke-dasharray', '5,5');
        this.rubberBandRect.setAttribute('visibility', 'hidden');
        this.svg.appendChild(this.rubberBandRect);
    }
    // ========================================================================
    // Selection Mode
    // ========================================================================
    setMode(mode) {
        this.selectionMode = mode;
        // Update cursor
        switch (mode) {
            case 'lasso':
                this.svg.style.cursor = 'crosshair';
                break;
            case 'rubberband':
                this.svg.style.cursor = 'crosshair';
                break;
            default:
                this.svg.style.cursor = 'grab';
        }
    }
    getMode() {
        return this.selectionMode;
    }
    // ========================================================================
    // Lasso Selection
    // ========================================================================
    startLasso(point) {
        this.lassoActive = true;
        this.lassoPoints = [point];
        if (this.lassoPath) {
            this.lassoPath.setAttribute('visibility', 'visible');
        }
    }
    updateLasso(point) {
        if (!this.lassoActive || !this.lassoPath)
            return;
        this.lassoPoints.push(point);
        // Create path data
        if (this.lassoPoints.length > 1) {
            let pathData = `M ${this.lassoPoints[0].x} ${this.lassoPoints[0].y}`;
            for (let i = 1; i < this.lassoPoints.length; i++) {
                pathData += ` L ${this.lassoPoints[i].x} ${this.lassoPoints[i].y}`;
            }
            this.lassoPath.setAttribute('d', pathData);
        }
    }
    endLasso(nodes) {
        if (!this.lassoActive)
            return;
        this.lassoActive = false;
        // Close the path
        if (this.lassoPath && this.lassoPoints.length > 2) {
            let pathData = this.lassoPath.getAttribute('d') || '';
            pathData += ' Z'; // Close path
            this.lassoPath.setAttribute('d', pathData);
            // Find nodes inside lasso
            const selected = this.getNodesInLasso(nodes);
            this.setSelection(selected);
        }
        // Hide lasso
        if (this.lassoPath) {
            this.lassoPath.setAttribute('visibility', 'hidden');
        }
        this.lassoPoints = [];
    }
    getNodesInLasso(nodes) {
        const selected = new Set();
        for (const node of nodes) {
            if (this.isPointInPolygon(node.position, this.lassoPoints)) {
                selected.add(node.id);
            }
        }
        return selected;
    }
    isPointInPolygon(point, polygon) {
        // Ray casting algorithm
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i].x, yi = polygon[i].y;
            const xj = polygon[j].x, yj = polygon[j].y;
            const intersect = ((yi > point.y) !== (yj > point.y)) &&
                (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
            if (intersect)
                inside = !inside;
        }
        return inside;
    }
    // ========================================================================
    // Rubber Band Selection
    // ========================================================================
    startRubberBand(point) {
        this.rubberBandActive = true;
        this.rubberBandStart = point;
        if (this.rubberBandRect) {
            this.rubberBandRect.setAttribute('visibility', 'visible');
            this.rubberBandRect.setAttribute('x', point.x.toString());
            this.rubberBandRect.setAttribute('y', point.y.toString());
            this.rubberBandRect.setAttribute('width', '0');
            this.rubberBandRect.setAttribute('height', '0');
        }
    }
    updateRubberBand(point) {
        if (!this.rubberBandActive || !this.rubberBandRect)
            return;
        const x = Math.min(this.rubberBandStart.x, point.x);
        const y = Math.min(this.rubberBandStart.y, point.y);
        const width = Math.abs(point.x - this.rubberBandStart.x);
        const height = Math.abs(point.y - this.rubberBandStart.y);
        this.rubberBandRect.setAttribute('x', x.toString());
        this.rubberBandRect.setAttribute('y', y.toString());
        this.rubberBandRect.setAttribute('width', width.toString());
        this.rubberBandRect.setAttribute('height', height.toString());
    }
    endRubberBand(nodes, currentPoint) {
        if (!this.rubberBandActive)
            return;
        this.rubberBandActive = false;
        // Find nodes in rectangle
        const x1 = Math.min(this.rubberBandStart.x, currentPoint.x);
        const y1 = Math.min(this.rubberBandStart.y, currentPoint.y);
        const x2 = Math.max(this.rubberBandStart.x, currentPoint.x);
        const y2 = Math.max(this.rubberBandStart.y, currentPoint.y);
        const selected = new Set();
        for (const node of nodes) {
            if (node.position.x >= x1 && node.position.x <= x2 &&
                node.position.y >= y1 && node.position.y <= y2) {
                selected.add(node.id);
            }
        }
        this.setSelection(selected);
        // Hide rubber band
        if (this.rubberBandRect) {
            this.rubberBandRect.setAttribute('visibility', 'hidden');
        }
    }
    // ========================================================================
    // Selection Management
    // ========================================================================
    setSelection(nodeIds) {
        this.selectedNodes = new Set(nodeIds);
        if (this.onSelectionChange) {
            this.onSelectionChange(this.selectedNodes);
        }
    }
    addToSelection(nodeId) {
        this.selectedNodes.add(nodeId);
        if (this.onSelectionChange) {
            this.onSelectionChange(this.selectedNodes);
        }
    }
    removeFromSelection(nodeId) {
        this.selectedNodes.delete(nodeId);
        if (this.onSelectionChange) {
            this.onSelectionChange(this.selectedNodes);
        }
    }
    toggleSelection(nodeId) {
        if (this.selectedNodes.has(nodeId)) {
            this.removeFromSelection(nodeId);
        }
        else {
            this.addToSelection(nodeId);
        }
    }
    clearSelection() {
        this.selectedNodes.clear();
        if (this.onSelectionChange) {
            this.onSelectionChange(this.selectedNodes);
        }
    }
    getSelection() {
        return new Set(this.selectedNodes);
    }
    selectAll(nodes) {
        this.selectedNodes = new Set(nodes.map(n => n.id));
        if (this.onSelectionChange) {
            this.onSelectionChange(this.selectedNodes);
        }
    }
    invertSelection(nodes) {
        const allIds = new Set(nodes.map(n => n.id));
        const newSelection = new Set();
        for (const id of allIds) {
            if (!this.selectedNodes.has(id)) {
                newSelection.add(id);
            }
        }
        this.setSelection(newSelection);
    }
    // ========================================================================
    // Event Handlers
    // ========================================================================
    onSelection(callback) {
        this.onSelectionChange = callback;
    }
    // ========================================================================
    // Cleanup
    // ========================================================================
    destroy() {
        if (this.lassoPath && this.lassoPath.parentNode) {
            this.lassoPath.parentNode.removeChild(this.lassoPath);
        }
        if (this.rubberBandRect && this.rubberBandRect.parentNode) {
            this.rubberBandRect.parentNode.removeChild(this.rubberBandRect);
        }
        this.selectedNodes.clear();
    }
}
//# sourceMappingURL=MultiSelectHandler.js.map