/**
 * GraphNode - Individual node rendering and physics
 * Professional-grade node with SAP Fiori styling
 */

import {
    NodeConfig,
    NodeStatus,
    NodeShape,
    Vector2D,
    NodeMetrics,
    SAP_COLORS,
    DEFAULT_RENDER_CONFIG
} from './types';

export class GraphNode {
    // Core properties
    public id: string;
    public name: string;
    public description: string;
    public type: string;
    public status: NodeStatus;
    public model: string;
    public metrics: NodeMetrics;
    public group: string | null;
    public shape: NodeShape;

    // Physics properties
    public position: Vector2D;
    public velocity: Vector2D;
    public force: Vector2D;
    public mass: number;
    public radius: number;
    public fixed: boolean;  // If true, ignore physics

    // Rectangle dimensions (used when shape is 'rectangle')
    private rectWidth: number = 120;
    private rectHeight: number = 80;

    // Rendering
    public element: SVGGElement;
    private circle: SVGCircleElement | null = null;
    private rect: SVGRectElement | null = null;
    private icon: SVGTextElement;
    private label: SVGTextElement;
    private statusIndicator: SVGCircleElement;
    private halo: SVGCircleElement | SVGRectElement;
    private expandButton: SVGGElement | null = null;

    // State
    private isSelected: boolean = false;
    private isHovered: boolean = false;
    private isDragging: boolean = false;

    // Expand/Collapse state
    public expandState: 'expanded' | 'partial' | 'collapsed' = 'collapsed';
    public hasChildren: boolean = false;
    private onExpandClickCallback: ((nodeId: string, currentState: 'expanded' | 'partial' | 'collapsed') => void) | null = null;
    
    constructor(config: NodeConfig) {
        // Initialize core properties
        this.id = config.id;
        this.name = config.name;
        this.description = config.description || '';
        this.type = config.type;
        this.status = config.status;
        this.model = config.model || 'N/A';
        this.metrics = config.metrics || {
            totalRequests: 0,
            avgLatency: 0,
            successRate: 0
        };
        this.group = config.group || null;
        this.shape = config.shape || 'circle';

        // Initialize physics
        this.position = config.position || this.randomPosition();
        this.velocity = { x: 0, y: 0 };
        this.force = { x: 0, y: 0 };
        this.mass = 1.0;
        this.radius = DEFAULT_RENDER_CONFIG.nodeRadius;
        this.fixed = false;

        // Create SVG elements
        this.element = this.createElement();
    }
    
    // ========================================================================
    // Rendering
    // ========================================================================
    
    private createElement(): SVGGElement {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'graph-node');
        g.setAttribute('data-node-id', this.id);
        g.setAttribute('data-node-type', this.type);
        g.setAttribute('data-node-shape', this.shape);

        // Create halo (for selection/hover)
        this.halo = this.createHalo();
        g.appendChild(this.halo);

        // Create main shape (circle or rectangle)
        if (this.shape === 'rectangle') {
            this.rect = this.createRect();
            g.appendChild(this.rect);
        } else {
            this.circle = this.createCircle();
            g.appendChild(this.circle);
        }

        // Create status indicator (small circle in corner)
        this.statusIndicator = this.createStatusIndicator();
        g.appendChild(this.statusIndicator);

        // Create icon (Unicode or SVG path)
        this.icon = this.createIcon();
        g.appendChild(this.icon);

        // Create label (node name)
        this.label = this.createLabel();
        g.appendChild(this.label);

        // Create expand button (only rendered if hasChildren is true)
        this.expandButton = this.renderExpandButton();
        g.appendChild(this.expandButton);

        // Set initial position
        this.updatePosition();

        return g;
    }

    private createHalo(): SVGCircleElement | SVGRectElement {
        if (this.shape === 'rectangle') {
            const halo = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            halo.setAttribute('class', 'node-halo');
            halo.setAttribute('x', (-(this.rectWidth / 2) - 8).toString());
            halo.setAttribute('y', (-(this.rectHeight / 2) - 8).toString());
            halo.setAttribute('width', (this.rectWidth + 16).toString());
            halo.setAttribute('height', (this.rectHeight + 16).toString());
            halo.setAttribute('rx', '12');
            halo.setAttribute('ry', '12');
            halo.setAttribute('fill', 'none');
            halo.setAttribute('stroke', SAP_COLORS.brand);
            halo.setAttribute('stroke-width', '3');
            halo.setAttribute('opacity', '0');
            return halo;
        } else {
            const halo = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            halo.setAttribute('class', 'node-halo');
            halo.setAttribute('r', (this.radius + 8).toString());
            halo.setAttribute('fill', 'none');
            halo.setAttribute('stroke', SAP_COLORS.brand);
            halo.setAttribute('stroke-width', '3');
            halo.setAttribute('opacity', '0');
            return halo;
        }
    }

    private createRect(): SVGRectElement {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('class', 'node-rect graph-node-rect');
        rect.setAttribute('x', (-this.rectWidth / 2).toString());
        rect.setAttribute('y', (-this.rectHeight / 2).toString());
        rect.setAttribute('width', this.rectWidth.toString());
        rect.setAttribute('height', this.rectHeight.toString());
        rect.setAttribute('rx', '8');
        rect.setAttribute('ry', '8');
        rect.setAttribute('fill', this.getStatusColor());
        rect.setAttribute('stroke', '#ffffff');
        rect.setAttribute('stroke-width', '2');
        rect.setAttribute('filter', 'url(#nodeRectShadow)');
        rect.style.cursor = 'pointer';
        rect.style.transition = 'all 0.3s ease';
        return rect;
    }

    private createCircle(): SVGCircleElement {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('class', 'node-circle');
        circle.setAttribute('r', this.radius.toString());
        circle.setAttribute('fill', this.getStatusColor());
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', '2');
        circle.setAttribute('filter', 'url(#nodeShadow)');
        circle.style.cursor = 'pointer';
        circle.style.transition = 'all 0.3s ease';
        return circle;
    }
    
    private createStatusIndicator(): SVGCircleElement {
        const indicator = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        indicator.setAttribute('class', 'status-indicator');
        indicator.setAttribute('r', '8');
        indicator.setAttribute('fill', this.getStatusColor());
        indicator.setAttribute('stroke', '#ffffff');
        indicator.setAttribute('stroke-width', '2');
        return indicator;
    }
    
    private createIcon(): SVGTextElement {
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        icon.setAttribute('class', 'node-icon');
        icon.setAttribute('text-anchor', 'middle');
        icon.setAttribute('dominant-baseline', 'central');
        icon.setAttribute('font-size', '24');
        icon.setAttribute('fill', '#ffffff');
        icon.setAttribute('pointer-events', 'none');
        icon.textContent = this.getIconText();
        return icon;
    }
    
    private createLabel(): SVGTextElement {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('class', 'node-label');
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', '12');
        label.setAttribute('font-weight', 'bold');
        label.setAttribute('fill', SAP_COLORS.text);
        label.setAttribute('pointer-events', 'none');
        label.textContent = this.name;
        return label;
    }

    private renderExpandButton(): SVGGElement {
        const buttonGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        buttonGroup.setAttribute('class', 'node-expand-button');

        // Position at bottom-right of node (45 degrees, bottom-right)
        const angle = Math.PI / 4;  // 45 degrees bottom-right
        const offsetX = this.radius * Math.cos(angle);
        const offsetY = this.radius * Math.sin(angle);
        buttonGroup.setAttribute('transform', `translate(${offsetX}, ${offsetY})`);

        // Button background circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('class', 'expand-button-bg');
        circle.setAttribute('r', '10');
        circle.setAttribute('fill', SAP_COLORS.brand);
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', '2');
        circle.style.cursor = 'pointer';
        buttonGroup.appendChild(circle);

        // Button icon
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        icon.setAttribute('class', 'expand-button-icon');
        icon.setAttribute('text-anchor', 'middle');
        icon.setAttribute('dominant-baseline', 'central');
        icon.setAttribute('font-size', '10');
        icon.setAttribute('fill', '#ffffff');
        icon.setAttribute('pointer-events', 'none');
        icon.textContent = this.getExpandIcon();
        buttonGroup.appendChild(icon);

        // Initially hidden (shown only when hasChildren is true)
        buttonGroup.style.display = this.hasChildren ? 'block' : 'none';

        // Click handler
        buttonGroup.addEventListener('click', (e: Event) => {
            e.stopPropagation();
            if (this.onExpandClickCallback) {
                this.onExpandClickCallback(this.id, this.expandState);
            }
        });

        return buttonGroup;
    }

    private getExpandIcon(): string {
        switch (this.expandState) {
            case 'expanded':
                return '▼';
            case 'partial':
                return '◐';
            case 'collapsed':
            default:
                return '▶';
        }
    }

    public setExpandState(state: 'expanded' | 'partial' | 'collapsed'): void {
        this.expandState = state;

        // Update icon
        if (this.expandButton) {
            const icon = this.expandButton.querySelector('.expand-button-icon');
            if (icon) {
                icon.textContent = this.getExpandIcon();
            }

            // Update button background color based on state
            const bg = this.expandButton.querySelector('.expand-button-bg');
            if (bg) {
                switch (state) {
                    case 'expanded':
                        bg.setAttribute('fill', SAP_COLORS.success);
                        break;
                    case 'partial':
                        bg.setAttribute('fill', SAP_COLORS.warning);
                        break;
                    case 'collapsed':
                    default:
                        bg.setAttribute('fill', SAP_COLORS.brand);
                        break;
                }
            }

            // Animate the state change
            if (DEFAULT_RENDER_CONFIG.enableAnimations) {
                this.expandButton.animate([
                    { transform: `translate(${this.radius * Math.cos(Math.PI / 4)}, ${this.radius * Math.sin(Math.PI / 4)}) scale(1.3)` },
                    { transform: `translate(${this.radius * Math.cos(Math.PI / 4)}, ${this.radius * Math.sin(Math.PI / 4)}) scale(1.0)` }
                ], {
                    duration: 200,
                    easing: 'ease-out'
                });
            }
        }
    }

    public setHasChildren(hasChildren: boolean): void {
        this.hasChildren = hasChildren;
        if (this.expandButton) {
            this.expandButton.style.display = hasChildren ? 'block' : 'none';
        }
    }

    public onExpandClick(callback: (nodeId: string, currentState: 'expanded' | 'partial' | 'collapsed') => void): void {
        this.onExpandClickCallback = callback;
    }

    // ========================================================================
    // Physics & Movement
    // ========================================================================
    
    public updatePosition(): void {
        if (!this.element) return;

        // Update SVG transform
        this.element.setAttribute('transform', `translate(${this.position.x}, ${this.position.y})`);

        // Update status indicator position (top-right corner)
        if (this.shape === 'rectangle') {
            const offsetX = this.rectWidth / 2 - 8;
            const offsetY = -this.rectHeight / 2 + 8;
            this.statusIndicator.setAttribute('cx', offsetX.toString());
            this.statusIndicator.setAttribute('cy', offsetY.toString());
        } else {
            const angle = -Math.PI / 4;  // 45 degrees top-right
            const offsetX = this.radius * Math.cos(angle);
            const offsetY = this.radius * Math.sin(angle);
            this.statusIndicator.setAttribute('cx', offsetX.toString());
            this.statusIndicator.setAttribute('cy', offsetY.toString());
        }
    }
    
    public applyForce(fx: number, fy: number): void {
        if (this.fixed) return;
        
        this.force.x += fx;
        this.force.y += fy;
    }
    
    public updatePhysics(dt: number = 1.0): void {
        if (this.fixed) return;
        
        // F = ma, a = F/m
        const ax = this.force.x / this.mass;
        const ay = this.force.y / this.mass;
        
        // Update velocity
        this.velocity.x += ax * dt;
        this.velocity.y += ay * dt;
        
        // Apply damping
        this.velocity.x *= 0.9;
        this.velocity.y *= 0.9;
        
        // Limit max velocity
        const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
        if (speed > 10) {
            this.velocity.x = (this.velocity.x / speed) * 10;
            this.velocity.y = (this.velocity.y / speed) * 10;
        }
        
        // Update position
        this.position.x += this.velocity.x * dt;
        this.position.y += this.velocity.y * dt;
        
        // Reset forces
        this.force.x = 0;
        this.force.y = 0;
        
        // Update visual position
        this.updatePosition();
    }
    
    // ========================================================================
    // Visual State Changes
    // ========================================================================
    
    public setStatus(status: NodeStatus): void {
        this.status = status;
        const color = this.getStatusColor();

        const shapeElement = this.getShapeElement();
        if (shapeElement) {
            shapeElement.setAttribute('fill', color);
        }
        this.statusIndicator.setAttribute('fill', color);

        // Pulse animation on status change
        if (DEFAULT_RENDER_CONFIG.enableAnimations) {
            this.pulse();
        }
    }

    public setSelected(selected: boolean): void {
        this.isSelected = selected;
        const shapeElement = this.getShapeElement();

        if (selected) {
            this.halo.setAttribute('opacity', '1');
            if (shapeElement) {
                shapeElement.setAttribute('stroke-width', '4');
            }
            this.element.style.filter = 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))';
        } else {
            this.halo.setAttribute('opacity', '0');
            if (shapeElement) {
                shapeElement.setAttribute('stroke-width', '2');
            }
            this.element.style.filter = '';
        }
    }

    public setHovered(hovered: boolean): void {
        this.isHovered = hovered;
        const shapeElement = this.getShapeElement();

        if (hovered) {
            this.halo.setAttribute('opacity', '0.5');
            if (shapeElement) {
                shapeElement.style.transform = 'scale(1.1)';
            }
        } else if (!this.isSelected) {
            this.halo.setAttribute('opacity', '0');
            if (shapeElement) {
                shapeElement.style.transform = 'scale(1.0)';
            }
        }
    }

    public setDragging(dragging: boolean): void {
        this.isDragging = dragging;
        this.fixed = dragging;  // Fix position while dragging
        const shapeElement = this.getShapeElement();

        if (dragging) {
            this.element.style.cursor = 'grabbing';
            if (shapeElement) {
                shapeElement.style.opacity = '0.8';
            }
        } else {
            this.element.style.cursor = 'pointer';
            if (shapeElement) {
                shapeElement.style.opacity = '1.0';
            }
        }
    }

    /**
     * Get the main shape element (circle or rectangle)
     */
    private getShapeElement(): SVGCircleElement | SVGRectElement | null {
        return this.shape === 'rectangle' ? this.rect : this.circle;
    }

    // ========================================================================
    // Animations
    // ========================================================================

    public pulse(): void {
        const shapeElement = this.getShapeElement();
        if (!shapeElement) return;

        shapeElement.animate([
            { transform: 'scale(1.0)' },
            { transform: 'scale(1.2)', offset: 0.5 },
            { transform: 'scale(1.0)' }
        ], {
            duration: 600,
            easing: 'ease-out'
        });
    }
    
    public highlight(): void {
        // Add glowing effect
        this.element.style.filter = 'drop-shadow(0 0 10px ' + this.getStatusColor() + ')';
        
        setTimeout(() => {
            this.element.style.filter = '';
        }, 1000);
    }
    
    public startPulseLoop(): void {
        // Continuous pulse for running status
        if (this.status === NodeStatus.Running) {
            const animate = () => {
                if (this.status === NodeStatus.Running) {
                    this.pulse();
                    setTimeout(animate, 1500);
                }
            };
            animate();
        }
    }
    
    // ========================================================================
    // Utilities
    // ========================================================================
    
    private getStatusColor(): string {
        switch (this.status) {
            case NodeStatus.Success:
                return SAP_COLORS.success;
            case NodeStatus.Warning:
                return SAP_COLORS.warning;
            case NodeStatus.Error:
                return SAP_COLORS.error;
            case NodeStatus.Running:
                return SAP_COLORS.brand;
            case NodeStatus.None:
            default:
                return SAP_COLORS.neutral;
        }
    }
    
    private getIconText(): string {
        // Map node types to Unicode icons
        const iconMap: Record<string, string> = {
            'code_intelligence': '📝',
            'vector_search': '🔍',
            'graph_database': '🕸️',
            'verification': '✓',
            'workflow': '⚙️',
            'lineage': '📊',
            'orchestrator': '🎯',
            'router': '🔀',
            'translation': '🌐',
            'rag': '📚'
        };
        
        return iconMap[this.type] || '⚡';
    }
    
    private randomPosition(): Vector2D {
        // Random position in a circle (for initial layout)
        const angle = Math.random() * 2 * Math.PI;
        const distance = Math.random() * 200 + 100;
        
        return {
            x: Math.cos(angle) * distance,
            y: Math.sin(angle) * distance
        };
    }
    
    public distanceTo(other: GraphNode): number {
        const dx = this.position.x - other.position.x;
        const dy = this.position.y - other.position.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    public angleTo(other: GraphNode): number {
        const dx = other.position.x - this.position.x;
        const dy = other.position.y - this.position.y;
        return Math.atan2(dy, dx);
    }
    
    // ========================================================================
    // Collision Detection
    // ========================================================================

    public overlaps(other: GraphNode): boolean {
        const minDistance = this.radius + other.radius;
        return this.distanceTo(other) < minDistance;
    }

    public containsPoint(point: Vector2D): boolean {
        const dx = point.x - this.position.x;
        const dy = point.y - this.position.y;

        if (this.shape === 'rectangle') {
            // Check if point is within rectangle bounds
            return Math.abs(dx) <= this.rectWidth / 2 && Math.abs(dy) <= this.rectHeight / 2;
        } else {
            // Circle: check distance from center
            const distance = Math.sqrt(dx * dx + dy * dy);
            return distance <= this.radius;
        }
    }

    // ========================================================================
    // Dimension Accessors
    // ========================================================================

    /**
     * Get the width of the node (for layout calculations)
     */
    public getWidth(): number {
        return this.shape === 'rectangle' ? this.rectWidth : this.radius * 2;
    }

    /**
     * Get the height of the node (for layout calculations)
     */
    public getHeight(): number {
        return this.shape === 'rectangle' ? this.rectHeight : this.radius * 2;
    }
    
    // ========================================================================
    // Data Export
    // ========================================================================
    
    public toJSON(): NodeConfig {
        return {
            id: this.id,
            name: this.name,
            description: this.description,
            type: this.type,
            status: this.status,
            model: this.model,
            metrics: this.metrics,
            group: this.group || undefined,
            position: { ...this.position },
            shape: this.shape
        };
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
