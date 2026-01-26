/**
 * ProcessFlowConnection - SAP-styled connections
 * Rounded corners, arrows, exact SAP styling
 * Supports labels with semantic colors (SAP Fiori specification)
 */

import {
    ProcessFlowConnection as ConnectionConfig,
    ProcessFlowConnectionState,
    PROCESS_FLOW_COLORS,
    PROCESS_FLOW_LAYOUT
} from './types';
import { ProcessFlowNode } from './ProcessFlowNode';

// ============================================================================
// Label Configuration
// ============================================================================

export type ConnectionLabelState = 'Positive' | 'Negative' | 'Neutral' | 'Critical';

export interface ConnectionLabelConfig {
    text: string;
    state: ConnectionLabelState;
    icon?: string;
}

// Label colors based on SAP Fiori semantic colors
const LABEL_COLORS: Record<ConnectionLabelState, string> = {
    Positive: '#107e3e',   // Green
    Negative: '#bb0000',   // Red
    Neutral: '#0070f2',    // Blue
    Critical: '#df6e0c'    // Orange/Yellow
};

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

    // Label support
    private label: ConnectionLabelConfig | null = null;
    private labelElement: SVGGElement | null = null;
    private labelClickCallback: ((connection: ProcessFlowConnection) => void) | null = null;

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
    // Label Support - SAP Fiori Connection Labels
    // ========================================================================

    /**
     * Sets the label for this connection
     * @param label Label configuration or null to remove
     */
    public setLabel(label: ConnectionLabelConfig | null): void {
        this.label = label;

        if (label) {
            this.renderLabel();
        } else {
            this.removeLabel();
        }
    }

    /**
     * Gets the current label configuration
     */
    public getLabel(): ConnectionLabelConfig | null {
        return this.label;
    }

    /**
     * Sets callback for label click events
     */
    public onLabelClick(callback: ((connection: ProcessFlowConnection) => void) | null): void {
        this.labelClickCallback = callback;
    }

    /**
     * Renders the label element (pill-shaped badge)
     */
    private renderLabel(): void {
        if (!this.label) return;

        // Remove existing label
        this.removeLabel();

        // Create label group
        this.labelElement = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.labelElement.setAttribute('class', 'connection-label');
        this.labelElement.style.cursor = 'pointer';

        // Calculate label dimensions
        const text = this.label.text;
        const hasIcon = !!this.label.icon;
        const fontSize = 11;
        const paddingX = 8;
        const paddingY = 4;
        const iconWidth = hasIcon ? 14 : 0;
        const textWidth = text.length * 6; // Approximate width
        const labelWidth = textWidth + iconWidth + paddingX * 2 + (hasIcon ? 4 : 0);
        const labelHeight = fontSize + paddingY * 2;
        const cornerRadius = labelHeight / 2; // Pill shape

        // Get label color
        const bgColor = LABEL_COLORS[this.label.state];

        // Create background (pill shape)
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', (-labelWidth / 2).toString());
        rect.setAttribute('y', (-labelHeight / 2).toString());
        rect.setAttribute('width', labelWidth.toString());
        rect.setAttribute('height', labelHeight.toString());
        rect.setAttribute('rx', cornerRadius.toString());
        rect.setAttribute('ry', cornerRadius.toString());
        rect.setAttribute('fill', bgColor);
        rect.style.filter = 'drop-shadow(0 1px 2px rgba(0,0,0,0.2))';
        this.labelElement.appendChild(rect);

        // Create icon if present
        let iconOffset = 0;
        if (hasIcon && this.label.icon) {
            iconOffset = -textWidth / 2 - 2;
            const iconText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            iconText.setAttribute('x', iconOffset.toString());
            iconText.setAttribute('y', '4');
            iconText.setAttribute('font-family', 'SAP-icons');
            iconText.setAttribute('font-size', '12');
            iconText.setAttribute('fill', 'white');
            iconText.setAttribute('text-anchor', 'middle');
            iconText.textContent = this.label.icon;
            this.labelElement.appendChild(iconText);
        }

        // Create text
        const textEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        const textX = hasIcon ? iconOffset + iconWidth + 2 : 0;
        textEl.setAttribute('x', textX.toString());
        textEl.setAttribute('y', '4');
        textEl.setAttribute('font-family', '"72", "72full", Arial, Helvetica, sans-serif');
        textEl.setAttribute('font-size', fontSize.toString());
        textEl.setAttribute('font-weight', '600');
        textEl.setAttribute('fill', 'white');
        textEl.setAttribute('text-anchor', hasIcon ? 'start' : 'middle');
        textEl.textContent = text;
        this.labelElement.appendChild(textEl);

        // Add click handler
        this.labelElement.addEventListener('click', (e) => {
            e.stopPropagation();
            if (this.labelClickCallback) {
                this.labelClickCallback(this);
            }
        });

        // Add hover effect
        this.labelElement.addEventListener('mouseenter', () => {
            rect.style.filter = 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))';
            rect.setAttribute('opacity', '0.9');
        });
        this.labelElement.addEventListener('mouseleave', () => {
            rect.style.filter = 'drop-shadow(0 1px 2px rgba(0,0,0,0.2))';
            rect.setAttribute('opacity', '1');
        });

        // Add to connection element
        this.element.appendChild(this.labelElement);

        // Position at midpoint
        this.positionLabel();
    }

    /**
     * Positions the label at the center of the connection path
     */
    private positionLabel(): void {
        if (!this.labelElement || !this.sourceNode || !this.targetNode) return;

        // Calculate midpoint of the connection
        const sourceX = this.sourceNode.x + this.sourceNode.getWidth();
        const sourceY = this.sourceNode.y + this.sourceNode.getHeight() / 2;
        const targetX = this.targetNode.x;
        const targetY = this.targetNode.y + this.targetNode.getHeight() / 2;

        // For orthogonal paths, the midpoint is at the vertical segment center
        const midX = sourceX + (targetX - sourceX) / 2;
        const midY = (sourceY + targetY) / 2;

        this.labelElement.setAttribute('transform', `translate(${midX}, ${midY})`);
    }

    /**
     * Removes the label element
     */
    private removeLabel(): void {
        if (this.labelElement && this.labelElement.parentNode) {
            this.labelElement.parentNode.removeChild(this.labelElement);
            this.labelElement = null;
        }
    }

    // ========================================================================
    // Path Highlighting
    // ========================================================================

    /**
     * Sets the highlighted state for path highlighting
     * @param highlighted Whether the connection should be highlighted
     */
    public setHighlighted(highlighted: boolean): void {
        if (highlighted) {
            this.element.classList.add('highlighted');
            this.element.classList.remove('dimmed');
            this.path.setAttribute('stroke', PROCESS_FLOW_COLORS.connection.highlighted);
            this.path.setAttribute('stroke-width', '3');
            this.arrow.setAttribute('fill', PROCESS_FLOW_COLORS.connection.highlighted);
            this.element.style.filter = 'drop-shadow(0 0 4px rgba(10, 110, 209, 0.5))';
        } else {
            this.element.classList.remove('highlighted');
            this.path.setAttribute('stroke', this.getStateColor());
            this.path.setAttribute('stroke-width', PROCESS_FLOW_LAYOUT.connection.strokeWidth.toString());
            this.arrow.setAttribute('fill', this.getStateColor());
            this.element.style.filter = '';
        }
    }

    /**
     * Sets the dimmed state for path highlighting
     * @param dimmed Whether the connection should be dimmed
     */
    public setDimmed(dimmed: boolean): void {
        if (dimmed) {
            this.element.classList.add('dimmed');
            this.element.classList.remove('highlighted');
            this.path.setAttribute('stroke', PROCESS_FLOW_COLORS.connection.dimmed);
            this.arrow.setAttribute('fill', PROCESS_FLOW_COLORS.connection.dimmed);
            this.element.style.opacity = '0.3';
        } else {
            this.element.classList.remove('dimmed');
            this.path.setAttribute('stroke', this.getStateColor());
            this.arrow.setAttribute('fill', this.getStateColor());
            this.element.style.opacity = '1';
        }
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    public destroy(): void {
        this.removeLabel();
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }
}
