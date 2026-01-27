/**
 * ProcessFlowNode - SAP-styled process step
 * 100% Commercial Quality - Exact SAP Fiori styling with folded corners
 */

import {
    ProcessFlowNode as NodeConfig,
    ProcessFlowNodeState,
    ProcessFlowDisplayState,
    PROCESS_FLOW_COLORS,
    PROCESS_FLOW_LAYOUT
} from './types';

export class ProcessFlowNode {
    public id: string;
    public lane: string;
    public title: string;
    public state: ProcessFlowNodeState;
    public texts: string[];
    public children: string[];
    public position: number;
    
    // Rendering
    public element: SVGGElement;
    private background: SVGPathElement;        // Path for folded corner
    private borderPath: SVGPathElement;
    private titleText: SVGTextElement;
    private stateText: SVGTextElement;
    private detailTexts: SVGTextElement[] = [];
    private icon: SVGTextElement | null = null;
    
    // Display state
    private displayState: ProcessFlowDisplayState = ProcessFlowDisplayState.Regular;
    
    // Position
    public x: number = 0;
    public y: number = 0;
    
    constructor(config: NodeConfig) {
        this.id = config.id;
        this.lane = config.lane;
        this.title = config.title;
        this.state = config.state;
        this.texts = config.texts || [];
        this.children = config.children || [];
        this.position = config.position || 0;
        
        // Create SVG element
        this.element = this.createElement();
    }
    
    // ========================================================================
    // Rendering - SAP Signature Style
    // ========================================================================
    
    private createElement(): SVGGElement {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'process-flow-node');
        g.setAttribute('data-node-id', this.id);
        g.setAttribute('data-state', this.state);
        
        const layout = PROCESS_FLOW_LAYOUT.node;
        
        // Create background with folded corner (SAP signature style)
        this.background = this.createFoldedCornerPath();
        g.appendChild(this.background);
        
        // Create border
        this.borderPath = this.createBorderPath();
        g.appendChild(this.borderPath);
        
        // Create icon (if needed)
        if (this.needsIcon()) {
            this.icon = this.createIcon();
            g.appendChild(this.icon);
        }
        
        // Create title
        this.titleText = this.createTitle();
        g.appendChild(this.titleText);
        
        // Create state text (e.g., "Completed", "Failed")
        this.stateText = this.createStateText();
        g.appendChild(this.stateText);
        
        // Create detail texts
        for (let i = 0; i < this.texts.length; i++) {
            const text = this.createDetailText(this.texts[i], i);
            this.detailTexts.push(text);
            g.appendChild(text);
        }
        
        // Add hover effects
        g.style.cursor = 'pointer';
        g.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        
        return g;
    }
    
    /**
     * Creates SAP signature folded corner path
     * This is the distinctive SAP Process Flow visual style
     */
    private createFoldedCornerPath(): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'node-background');
        
        const w = PROCESS_FLOW_LAYOUT.node.width;
        const h = PROCESS_FLOW_LAYOUT.node.height;
        const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
        const foldSize = 12;  // Folded corner size
        
        // Create path with folded top-right corner (SAP signature)
        const pathData = `
            M ${r} 0
            L ${w - foldSize} 0
            L ${w} ${foldSize}
            L ${w} ${h - r}
            Q ${w} ${h}, ${w - r} ${h}
            L ${r} ${h}
            Q 0 ${h}, 0 ${h - r}
            L 0 ${r}
            Q 0 0, ${r} 0
            Z
        `.trim().replace(/\s+/g, ' ');
        
        path.setAttribute('d', pathData);
        path.setAttribute('fill', this.getStateColors().background);
        path.style.transition = 'fill 0.3s ease';
        
        return path;
    }
    
    private createBorderPath(): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'node-border');
        
        const w = PROCESS_FLOW_LAYOUT.node.width;
        const h = PROCESS_FLOW_LAYOUT.node.height;
        const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
        const foldSize = 12;
        
        // Same path as background
        const pathData = `
            M ${r} 0
            L ${w - foldSize} 0
            L ${w} ${foldSize}
            L ${w} ${h - r}
            Q ${w} ${h}, ${w - r} ${h}
            L ${r} ${h}
            Q 0 ${h}, 0 ${h - r}
            L 0 ${r}
            Q 0 0, ${r} 0
            Z
        `.trim().replace(/\s+/g, ' ');
        
        path.setAttribute('d', pathData);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', this.getStateColors().border);
        path.setAttribute('stroke-width', PROCESS_FLOW_LAYOUT.node.borderWidth.toString());
        path.style.transition = 'stroke 0.3s ease';
        
        return path;
    }
    
    private createIcon(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('class', 'node-icon');
        text.setAttribute('x', (PROCESS_FLOW_LAYOUT.node.padding).toString());
        text.setAttribute('y', (PROCESS_FLOW_LAYOUT.node.padding + 18).toString());
        text.setAttribute('font-size', PROCESS_FLOW_LAYOUT.node.iconSize.toString());
        text.setAttribute('fill', this.getStateColors().text);
        text.textContent = this.getIconForState();
        return text;
    }
    
    private createTitle(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('class', 'node-title');
        
        const xOffset = this.icon ? 40 : PROCESS_FLOW_LAYOUT.node.padding;
        text.setAttribute('x', xOffset.toString());
        text.setAttribute('y', (PROCESS_FLOW_LAYOUT.node.padding + 16).toString());
        text.setAttribute('font-size', PROCESS_FLOW_LAYOUT.node.titleFontSize.toString());
        text.setAttribute('font-weight', 'bold');
        text.setAttribute('fill', this.getStateColors().text);
        text.textContent = this.truncateText(this.title, 18);
        
        return text;
    }
    
    private createStateText(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('class', 'node-state-text');
        text.setAttribute('x', PROCESS_FLOW_LAYOUT.node.padding.toString());
        text.setAttribute('y', (PROCESS_FLOW_LAYOUT.node.padding + 36).toString());
        text.setAttribute('font-size', PROCESS_FLOW_LAYOUT.node.textFontSize.toString());
        text.setAttribute('fill', this.getStateColors().text);
        text.setAttribute('opacity', '0.9');
        text.textContent = this.getStateLabel();
        
        return text;
    }
    
    private createDetailText(content: string, index: number): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('class', 'node-detail-text');
        text.setAttribute('x', PROCESS_FLOW_LAYOUT.node.padding.toString());
        text.setAttribute('y', (PROCESS_FLOW_LAYOUT.node.padding + 52 + index * 14).toString());
        text.setAttribute('font-size', (PROCESS_FLOW_LAYOUT.node.textFontSize - 1).toString());
        text.setAttribute('fill', this.getStateColors().text);
        text.setAttribute('opacity', '0.8');
        text.textContent = this.truncateText(content, 22);
        
        return text;
    }
    
    // ========================================================================
    // Position Management
    // ========================================================================
    
    public setPosition(x: number, y: number): void {
        this.x = x;
        this.y = y;
        this.element.setAttribute('transform', `translate(${x}, ${y})`);
    }
    
    public getWidth(): number {
        return PROCESS_FLOW_LAYOUT.node.width;
    }
    
    public getHeight(): number {
        return PROCESS_FLOW_LAYOUT.node.height;
    }
    
    // ========================================================================
    // State Management
    // ========================================================================
    
    public setState(state: ProcessFlowNodeState): void {
        this.state = state;
        this.updateColors();
        this.stateText.textContent = this.getStateLabel();
    }
    
    public setDisplayState(state: ProcessFlowDisplayState): void {
        this.displayState = state;
        
        switch (state) {
            case ProcessFlowDisplayState.Highlighted:
                this.element.style.filter = 'drop-shadow(0 4px 8px rgba(10,110,209,0.4))';
                this.element.style.transform = 'scale(1.05)';
                break;
            case ProcessFlowDisplayState.Dimmed:
                this.element.style.opacity = '0.4';
                break;
            case ProcessFlowDisplayState.Selected:
                this.element.style.filter = 'drop-shadow(0 0 12px rgba(10,110,209,0.6))';
                this.borderPath.setAttribute('stroke-width', '3');
                break;
            case ProcessFlowDisplayState.Regular:
            default:
                this.element.style.filter = '';
                this.element.style.transform = 'scale(1.0)';
                this.element.style.opacity = '1.0';
                this.borderPath.setAttribute('stroke-width', PROCESS_FLOW_LAYOUT.node.borderWidth.toString());
        }
    }
    
    private updateColors(): void {
        const colors = this.getStateColors();
        this.background.setAttribute('fill', colors.background);
        this.borderPath.setAttribute('stroke', colors.border);
        this.titleText.setAttribute('fill', colors.text);
        this.stateText.setAttribute('fill', colors.text);
        
        for (const text of this.detailTexts) {
            text.setAttribute('fill', colors.text);
        }
        
        if (this.icon) {
            this.icon.setAttribute('fill', colors.text);
        }
    }
    
    // ========================================================================
    // Utilities
    // ========================================================================
    
    private getStateColors() {
        switch (this.state) {
            case ProcessFlowNodeState.Positive:
                return PROCESS_FLOW_COLORS.positive;
            case ProcessFlowNodeState.Negative:
                return PROCESS_FLOW_COLORS.negative;
            case ProcessFlowNodeState.Critical:
                return PROCESS_FLOW_COLORS.critical;
            case ProcessFlowNodeState.Planned:
                return PROCESS_FLOW_COLORS.planned;
            case ProcessFlowNodeState.PlannedNegative:
                return PROCESS_FLOW_COLORS.plannedNegative;
            case ProcessFlowNodeState.Neutral:
            default:
                return PROCESS_FLOW_COLORS.neutral;
        }
    }
    
    private getStateLabel(): string {
        switch (this.state) {
            case ProcessFlowNodeState.Positive:
                return 'Completed';
            case ProcessFlowNodeState.Negative:
                return 'Failed';
            case ProcessFlowNodeState.Critical:
                return 'Warning';
            case ProcessFlowNodeState.Planned:
                return 'Planned';
            case ProcessFlowNodeState.PlannedNegative:
                return 'Planned (Issue)';
            case ProcessFlowNodeState.Neutral:
                return 'In Progress';
            default:
                return '';
        }
    }
    
    private getIconForState(): string {
        switch (this.state) {
            case ProcessFlowNodeState.Positive:
                return '✓';
            case ProcessFlowNodeState.Negative:
                return '✗';
            case ProcessFlowNodeState.Critical:
                return '⚠';
            case ProcessFlowNodeState.Neutral:
                return '▶';
            default:
                return '';
        }
    }
    
    private needsIcon(): boolean {
        return this.state !== ProcessFlowNodeState.Planned &&
               this.state !== ProcessFlowNodeState.PlannedNegative;
    }
    
    private truncateText(text: string, maxLength: number): string {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }
    
    // ========================================================================
    // Export
    // ========================================================================
    
    public toJSON(): NodeConfig {
        return {
            id: this.id,
            lane: this.lane,
            title: this.title,
            state: this.state,
            texts: this.texts,
            children: this.children,
            position: this.position
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
