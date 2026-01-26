/**
 * GraphNode - Individual node rendering and physics
 * Professional-grade node with SAP Fiori styling
 */

import { 
    NodeConfig, 
    NodeStatus, 
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
    
    // Physics properties
    public position: Vector2D;
    public velocity: Vector2D;
    public force: Vector2D;
    public mass: number;
    public radius: number;
    public fixed: boolean;  // If true, ignore physics
    
    // Rendering
    public element: SVGGElement;
    private circle: SVGCircleElement;
    private icon: SVGTextElement;
    private label: SVGTextElement;
    private statusIndicator: SVGCircleElement;
    private halo: SVGCircleElement;
    
    // State
    private isSelected: boolean = false;
    private isHovered: boolean = false;
    private isDragging: boolean = false;
    
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
        
        // Create halo (for selection/hover)
        this.halo = this.createHalo();
        g.appendChild(this.halo);
        
        // Create main circle
        this.circle = this.createCircle();
        g.appendChild(this.circle);
        
        // Create status indicator (small circle in corner)
        this.statusIndicator = this.createStatusIndicator();
        g.appendChild(this.statusIndicator);
        
        // Create icon (Unicode or SVG path)
        this.icon = this.createIcon();
        g.appendChild(this.icon);
        
        // Create label (node name)
        this.label = this.createLabel();
        g.appendChild(this.label);
        
        // Set initial position
        this.updatePosition();
        
        return g;
    }
    
    private createHalo(): SVGCircleElement {
        const halo = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        halo.setAttribute('class', 'node-halo');
        halo.setAttribute('r', (this.radius + 8).toString());
        halo.setAttribute('fill', 'none');
        halo.setAttribute('stroke', SAP_COLORS.brand);
        halo.setAttribute('stroke-width', '3');
        halo.setAttribute('opacity', '0');
        return halo;
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
    
    // ========================================================================
    // Physics & Movement
    // ========================================================================
    
    public updatePosition(): void {
        if (!this.element) return;
        
        // Update SVG transform
        this.element.setAttribute('transform', `translate(${this.position.x}, ${this.position.y})`);
        
        // Update status indicator position (top-right corner)
        const angle = -Math.PI / 4;  // 45 degrees top-right
        const offsetX = this.radius * Math.cos(angle);
        const offsetY = this.radius * Math.sin(angle);
        this.statusIndicator.setAttribute('cx', offsetX.toString());
        this.statusIndicator.setAttribute('cy', offsetY.toString());
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
        
        this.circle.setAttribute('fill', color);
        this.statusIndicator.setAttribute('fill', color);
        
        // Pulse animation on status change
        if (DEFAULT_RENDER_CONFIG.enableAnimations) {
            this.pulse();
        }
    }
    
    public setSelected(selected: boolean): void {
        this.isSelected = selected;
        
        if (selected) {
            this.halo.setAttribute('opacity', '1');
            this.circle.setAttribute('stroke-width', '4');
            this.element.style.filter = 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))';
        } else {
            this.halo.setAttribute('opacity', '0');
            this.circle.setAttribute('stroke-width', '2');
            this.element.style.filter = '';
        }
    }
    
    public setHovered(hovered: boolean): void {
        this.isHovered = hovered;
        
        if (hovered) {
            this.halo.setAttribute('opacity', '0.5');
            this.circle.style.transform = 'scale(1.1)';
        } else if (!this.isSelected) {
            this.halo.setAttribute('opacity', '0');
            this.circle.style.transform = 'scale(1.0)';
        }
    }
    
    public setDragging(dragging: boolean): void {
        this.isDragging = dragging;
        this.fixed = dragging;  // Fix position while dragging
        
        if (dragging) {
            this.element.style.cursor = 'grabbing';
            this.circle.style.opacity = '0.8';
        } else {
            this.element.style.cursor = 'pointer';
            this.circle.style.opacity = '1.0';
        }
    }
    
    // ========================================================================
    // Animations
    // ========================================================================
    
    public pulse(): void {
        const animation = this.circle.animate([
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
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance <= this.radius;
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
            position: { ...this.position }
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
