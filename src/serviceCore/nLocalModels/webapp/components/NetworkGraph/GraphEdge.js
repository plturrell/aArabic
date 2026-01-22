/**
 * GraphEdge - Edge rendering with arrows and flow animations
 * Professional-grade edge with Bezier curves and SAP styling
 */

import { 
    EdgeConfig, 
    EdgeStatus, 
    Vector2D,
    SAP_COLORS,
    DEFAULT_RENDER_CONFIG 
} from './types';
import { GraphNode } from './GraphNode';

export class GraphEdge {
    // Core properties
    public id: string;
    public from: string;
    public to: string;
    public label: string;
    public status: EdgeStatus;
    public animated: boolean;
    
    // Rendering
    public element: SVGGElement;
    private path: SVGPathElement;
    private arrowHead: SVGPolygonElement;
    private labelText: SVGTextElement;
    private flowCircle: SVGCircleElement;
    
    // References to nodes
    private sourceNode: GraphNode | null = null;
    private targetNode: GraphNode | null = null;
    
    // Animation
    private flowAnimation: Animation | null = null;
    
    constructor(config: EdgeConfig) {
        this.id = config.id;
        this.from = config.from;
        this.to = config.to;
        this.label = config.label || '';
        this.status = config.status || EdgeStatus.Inactive;
        this.animated = config.animated || false;
        
        // Create SVG elements
        this.element = this.createElement();
    }
    
    // ========================================================================
    // Rendering
    // ========================================================================
    
    private createElement(): SVGGElement {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'graph-edge');
        g.setAttribute('data-edge-id', this.id);
        g.setAttribute('data-from', this.from);
        g.setAttribute('data-to', this.to);
        
        // Create path (the line)
        this.path = this.createPath();
        g.appendChild(this.path);
        
        // Create arrow head
        this.arrowHead = this.createArrowHead();
        g.appendChild(this.arrowHead);
        
        // Create flow indicator (animated circle)
        this.flowCircle = this.createFlowCircle();
        g.appendChild(this.flowCircle);
        
        // Create label (optional)
        if (this.label) {
            this.labelText = this.createLabel();
            g.appendChild(this.labelText);
        }
        
        return g;
    }
    
    private createPath(): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'edge-path');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', this.getStatusColor());
        path.setAttribute('stroke-width', DEFAULT_RENDER_CONFIG.edgeStrokeWidth.toString());
        path.setAttribute('stroke-linecap', 'round');
        path.style.transition = 'stroke 0.3s ease';
        
        // Dashed for inactive
        if (this.status === EdgeStatus.Inactive) {
            path.setAttribute('stroke-dasharray', '5,5');
        }
        
        return path;
    }
    
    private createArrowHead(): SVGPolygonElement {
        const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        arrow.setAttribute('class', 'edge-arrow');
        arrow.setAttribute('fill', this.getStatusColor());
        arrow.style.transition = 'fill 0.3s ease';
        return arrow;
    }
    
    private createFlowCircle(): SVGCircleElement {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('class', 'edge-flow');
        circle.setAttribute('r', '4');
        circle.setAttribute('fill', SAP_COLORS.brand);
        circle.setAttribute('opacity', '0');
        return circle;
    }
    
    private createLabel(): SVGTextElement {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('class', 'edge-label');
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', '10');
        label.setAttribute('fill', SAP_COLORS.textMuted);
        label.setAttribute('pointer-events', 'none');
        label.textContent = this.label;
        return label;
    }
    
    // ========================================================================
    // Position Update
    // ========================================================================
    
    public setNodes(source: GraphNode, target: GraphNode): void {
        this.sourceNode = source;
        this.targetNode = target;
        this.updatePath();
    }
    
    public updatePath(): void {
        if (!this.sourceNode || !this.targetNode) return;
        
        const start = this.sourceNode.position;
        const end = this.targetNode.position;
        
        // Calculate angle between nodes
        const angle = Math.atan2(end.y - start.y, end.x - start.x);
        
        // Offset start/end by node radius (so edge doesn't overlap node)
        const startOffset = {
            x: start.x + this.sourceNode.radius * Math.cos(angle),
            y: start.y + this.sourceNode.radius * Math.sin(angle)
        };
        
        const endOffset = {
            x: end.x - this.targetNode.radius * Math.cos(angle),
            y: end.y - this.targetNode.radius * Math.sin(angle)
        };
        
        // Create Bezier curve for smooth edges
        const pathData = this.createBezierPath(startOffset, endOffset);
        this.path.setAttribute('d', pathData);
        
        // Position arrow at end
        this.positionArrowHead(endOffset, angle);
        
        // Position label at midpoint
        if (this.labelText) {
            const midX = (startOffset.x + endOffset.x) / 2;
            const midY = (startOffset.y + endOffset.y) / 2;
            this.labelText.setAttribute('x', midX.toString());
            this.labelText.setAttribute('y', (midY - 10).toString());
        }
    }
    
    private createBezierPath(start: Vector2D, end: Vector2D): string {
        // Calculate control points for smooth Bezier curve
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Curve factor (how much the curve bends)
        const curvature = 0.2;
        const offset = distance * curvature;
        
        // Control points perpendicular to the line
        const angle = Math.atan2(dy, dx);
        const perpAngle = angle + Math.PI / 2;
        
        const cp1x = start.x + dx * 0.33 + offset * Math.cos(perpAngle);
        const cp1y = start.y + dy * 0.33 + offset * Math.sin(perpAngle);
        
        const cp2x = start.x + dx * 0.67 + offset * Math.cos(perpAngle);
        const cp2y = start.y + dy * 0.67 + offset * Math.sin(perpAngle);
        
        // For straight edges (close nodes), use simple line
        if (distance < 100) {
            return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
        }
        
        // Cubic Bezier curve
        return `M ${start.x} ${start.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${end.x} ${end.y}`;
    }
    
    private positionArrowHead(point: Vector2D, angle: number): void {
        const arrowSize = DEFAULT_RENDER_CONFIG.arrowSize;
        
        // Calculate arrow points (triangle)
        const tip = point;
        const base1 = {
            x: tip.x - arrowSize * Math.cos(angle - Math.PI / 6),
            y: tip.y - arrowSize * Math.sin(angle - Math.PI / 6)
        };
        const base2 = {
            x: tip.x - arrowSize * Math.cos(angle + Math.PI / 6),
            y: tip.y - arrowSize * Math.sin(angle + Math.PI / 6)
        };
        
        const points = `${tip.x},${tip.y} ${base1.x},${base1.y} ${base2.x},${base2.y}`;
        this.arrowHead.setAttribute('points', points);
    }
    
    // ========================================================================
    // Visual State Changes
    // ========================================================================
    
    public setStatus(status: EdgeStatus): void {
        this.status = status;
        const color = this.getStatusColor();
        
        this.path.setAttribute('stroke', color);
        this.arrowHead.setAttribute('fill', color);
        
        // Update dash pattern
        if (status === EdgeStatus.Inactive) {
            this.path.setAttribute('stroke-dasharray', '5,5');
        } else {
            this.path.removeAttribute('stroke-dasharray');
        }
        
        // Start/stop flow animation
        if (status === EdgeStatus.Flowing || status === EdgeStatus.Active) {
            this.startFlowAnimation();
        } else {
            this.stopFlowAnimation();
        }
    }
    
    public setHighlighted(highlighted: boolean): void {
        if (highlighted) {
            this.path.setAttribute('stroke-width', '4');
            this.element.style.filter = 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))';
        } else {
            this.path.setAttribute('stroke-width', DEFAULT_RENDER_CONFIG.edgeStrokeWidth.toString());
            this.element.style.filter = '';
        }
    }
    
    // ========================================================================
    // Animations
    // ========================================================================
    
    public startFlowAnimation(): void {
        if (this.flowAnimation) return;  // Already animating
        
        this.flowCircle.setAttribute('opacity', '1');
        
        // Animate circle along the path
        this.flowAnimation = this.flowCircle.animate([
            { offsetDistance: '0%' },
            { offsetDistance: '100%' }
        ], {
            duration: 2000,
            iterations: Infinity,
            easing: 'linear'
        });
        
        // Use CSS offset-path for smooth animation along curve
        this.flowCircle.style.offsetPath = `path("${this.path.getAttribute('d')}")`;
    }
    
    public stopFlowAnimation(): void {
        if (this.flowAnimation) {
            this.flowAnimation.cancel();
            this.flowAnimation = null;
        }
        this.flowCircle.setAttribute('opacity', '0');
    }
    
    public pulse(): void {
        // Pulse the entire edge
        this.element.animate([
            { opacity: 1.0 },
            { opacity: 0.5, offset: 0.5 },
            { opacity: 1.0 }
        ], {
            duration: 800,
            easing: 'ease-in-out'
        });
    }
    
    public flash(): void {
        // Quick flash for data transmission
        this.path.animate([
            { stroke: this.getStatusColor() },
            { stroke: SAP_COLORS.brand },
            { stroke: this.getStatusColor() }
        ], {
            duration: 400,
            easing: 'ease-out'
        });
    }
    
    // ========================================================================
    // Utilities
    // ========================================================================
    
    private getStatusColor(): string {
        switch (this.status) {
            case EdgeStatus.Active:
                return SAP_COLORS.brand;
            case EdgeStatus.Flowing:
                return SAP_COLORS.success;
            case EdgeStatus.Error:
                return SAP_COLORS.error;
            case EdgeStatus.Inactive:
            default:
                return SAP_COLORS.neutral;
        }
    }
    
    public getLength(): number {
        if (!this.sourceNode || !this.targetNode) return 0;
        
        const dx = this.targetNode.position.x - this.sourceNode.position.x;
        const dy = this.targetNode.position.y - this.sourceNode.position.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    public getMidpoint(): Vector2D {
        if (!this.sourceNode || !this.targetNode) {
            return { x: 0, y: 0 };
        }
        
        return {
            x: (this.sourceNode.position.x + this.targetNode.position.x) / 2,
            y: (this.sourceNode.position.y + this.targetNode.position.y) / 2
        };
    }
    
    public containsPoint(point: Vector2D, threshold: number = 5): boolean {
        // Check if point is near the edge path (for click detection)
        if (!this.sourceNode || !this.targetNode) return false;
        
        const start = this.sourceNode.position;
        const end = this.targetNode.position;
        
        // Distance from point to line segment
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const lengthSquared = dx * dx + dy * dy;
        
        if (lengthSquared === 0) {
            const distX = point.x - start.x;
            const distY = point.y - start.y;
            return Math.sqrt(distX * distX + distY * distY) <= threshold;
        }
        
        const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
        
        const projX = start.x + t * dx;
        const projY = start.y + t * dy;
        
        const distX = point.x - projX;
        const distY = point.y - projY;
        const distance = Math.sqrt(distX * distX + distY * distY);
        
        return distance <= threshold;
    }
    
    // ========================================================================
    // Data Export
    // ========================================================================
    
    public toJSON(): EdgeConfig {
        return {
            id: this.id,
            from: this.from,
            to: this.to,
            label: this.label,
            status: this.status,
            animated: this.animated
        };
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    public destroy(): void {
        this.stopFlowAnimation();
        
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }
}
