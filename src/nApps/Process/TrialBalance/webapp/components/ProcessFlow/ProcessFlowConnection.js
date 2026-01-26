/**
 * ProcessFlowConnection - SAP-styled connections
 * Rounded corners, arrows, exact SAP styling
 */

import {
    ProcessFlowConnection as ConnectionConfig,
    ProcessFlowConnectionState,
    PROCESS_FLOW_COLORS,
    PROCESS_FLOW_LAYOUT
} from './types';
import { ProcessFlowNode } from './ProcessFlowNode';

export class ProcessFlowConnection {
    public id: string;
    public from: string;
    public to: string;
    public state: ProcessFlowConnectionState;
    public type: 'normal' | 'planned';
    
    // Rendering
    public element: SVGGElement;
    private path: SVGPathElement;
    private arrow: SVGPolygonElement;
    
    // Node references
    private sourceNode: ProcessFlowNode | null = null;
    private targetNode: ProcessFlowNode | null = null;
    
    constructor(config: ConnectionConfig) {
        this.id = `${config.from}-${config.to}`;
        this.from = config.from;
        this.to = config.to;
        this.state = config.state || ProcessFlowConnectionState.Normal;
        this.type = config.type || 'normal';
        
        // Create SVG element
        this.element = this.createElement();
    }
    
    // ========================================================================
    // Rendering - SAP Rounded Connections
    // ========================================================================
    
    private createElement(): SVGGElement {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'process-flow-connection');
        g.setAttribute('data-from', this.from);
        g.setAttribute('data-to', this.to);
        
        // Create path (the connection line)
        this.path = this.createPath();
        g.appendChild(this.path);
        
        // Create arrow
        this.arrow = this.createArrow();
        g.appendChild(this.arrow);
        
        return g;
    }
    
    private createPath(): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'connection-path');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', this.getStateColor());
        path.setAttribute('stroke-width', PROCESS_FLOW_LAYOUT.connection.strokeWidth.toString());
        path.setAttribute('stroke-linecap', 'round');
        path.setAttribute('stroke-linejoin', 'round');
        
        // Dashed for planned connections
        if (this.type === 'planned') {
            path.setAttribute('stroke-dasharray', PROCESS_FLOW_LAYOUT.connection.dashArray);
        }
        
        path.style.transition = 'stroke 0.3s ease';
        
        return path;
    }
    
    private createArrow(): SVGPolygonElement {
        const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        arrow.setAttribute('class', 'connection-arrow');
        arrow.setAttribute('fill', this.getStateColor());
        arrow.style.transition = 'fill 0.3s ease';
        
        return arrow;
    }
    
    // ========================================================================
    // Path Calculation - SAP Rounded Style
    // ========================================================================
    
    public setNodes(source: ProcessFlowNode, target: ProcessFlowNode): void {
        this.sourceNode = source;
        this.targetNode = target;
        this.updatePath();
    }
    
    public updatePath(): void {
        if (!this.sourceNode || !this.targetNode) return;
        
        const sourceX = this.sourceNode.x + this.sourceNode.getWidth();
        const sourceY = this.sourceNode.y + this.sourceNode.getHeight() / 2;
        
        const targetX = this.targetNode.x;
        const targetY = this.targetNode.y + this.targetNode.getHeight() / 2;
        
        // Create path with rounded corners (SAP style)
        const pathData = this.createRoundedPath(
            { x: sourceX, y: sourceY },
            { x: targetX, y: targetY }
        );
        
        this.path.setAttribute('d', pathData);
        
        // Position arrow at target
        this.positionArrow({ x: targetX, y: targetY });
    }
    
    /**
     * Creates SAP-style rounded connection path
     * Uses cubic bezier curves for smooth rounded corners
     */
    private createRoundedPath(start: { x: number; y: number }, end: { x: number; y: number }): string {
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const cornerRadius = PROCESS_FLOW_LAYOUT.connection.cornerRadius;
        
        // Horizontal then vertical connection with rounded corners
        if (Math.abs(dy) < 2) {
            // Straight horizontal line
            return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
        }
        
        // Calculate control points for smooth curves
        const midX = start.x + dx / 2;
        
        if (dy > 0) {
            // Going down
            const cp1 = Math.min(cornerRadius, Math.abs(dx) / 2);
            const cp2 = Math.min(cornerRadius, Math.abs(dy) / 2);
            
            return `
                M ${start.x} ${start.y}
                L ${midX - cp1} ${start.y}
                Q ${midX} ${start.y}, ${midX} ${start.y + cp2}
                L ${midX} ${end.y - cp2}
                Q ${midX} ${end.y}, ${midX + cp1} ${end.y}
                L ${end.x} ${end.y}
            `.trim().replace(/\s+/g, ' ');
        } else {
            // Going up
            const cp1 = Math.min(cornerRadius, Math.abs(dx) / 2);
            const cp2 = Math.min(cornerRadius, Math.abs(dy) / 2);
            
            return `
                M ${start.x} ${start.y}
                L ${midX - cp1} ${start.y}
                Q ${midX} ${start.y}, ${midX} ${start.y - cp2}
                L ${midX} ${end.y + cp2}
                Q ${midX} ${end.y}, ${midX + cp1} ${end.y}
                L ${end.x} ${end.y}
            `.trim().replace(/\s+/g, ' ');
        }
    }
    
    private positionArrow(point: { x: number; y: number }): void {
        const arrowSize = PROCESS_FLOW_LAYOUT.connection.arrowSize;
        
        // Arrow pointing right (since connections go left-to-right)
        const tip = point;
        const base1 = { x: tip.x - arrowSize, y: tip.y - arrowSize / 2 };
        const base2 = { x: tip.x - arrowSize, y: tip.y + arrowSize / 2 };
        
        const points = `${tip.x},${tip.y} ${base1.x},${base1.y} ${base2.x},${base2.y}`;
        this.arrow.setAttribute('points', points);
    }
    
    // ========================================================================
    // State Management
    // ========================================================================
    
    public setState(state: ProcessFlowConnectionState): void {
        this.state = state;
        const color = this.getStateColor();
        
        this.path.setAttribute('stroke', color);
        this.arrow.setAttribute('fill', color);
    }
    
    private getStateColor(): string {
        switch (this.state) {
            case ProcessFlowConnectionState.Highlighted:
                return PROCESS_FLOW_COLORS.connection.highlighted;
            case ProcessFlowConnectionState.Dimmed:
                return PROCESS_FLOW_COLORS.connection.dimmed;
            case ProcessFlowConnectionState.Normal:
            default:
                return PROCESS_FLOW_COLORS.connection.normal;
        }
    }
    
    // ========================================================================
    // Animation
    // ========================================================================
    
    public animate(): void {
        // Pulse animation for active flow
        this.path.animate([
            { strokeWidth: PROCESS_FLOW_LAYOUT.connection.strokeWidth },
            { strokeWidth: PROCESS_FLOW_LAYOUT.connection.strokeWidth + 2 },
            { strokeWidth: PROCESS_FLOW_LAYOUT.connection.strokeWidth }
        ], {
            duration: 800,
            easing: PROCESS_FLOW_LAYOUT.animation.easing
        });
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    public destroy(): void {
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }
}
