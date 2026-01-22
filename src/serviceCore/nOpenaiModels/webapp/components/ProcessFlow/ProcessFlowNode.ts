/**
 * ProcessFlowNode - SAP-styled process step
 * 100% Commercial Quality - Exact SAP Fiori styling with folded corners
 */

import {
    ProcessFlowNode as NodeConfig,
    ProcessFlowNodeState,
    ProcessFlowDisplayState,
    ProcessFlowZoomLevel,
    PROCESS_FLOW_COLORS,
    PROCESS_FLOW_LAYOUT,
    ZOOM_LEVEL_CONFIG,
    ZoomLevelConfig
} from './types';

export class ProcessFlowNode {
    public id: string;
    public lane: string;
    public title: string;
    public state: ProcessFlowNodeState;
    public texts: string[];
    public children: string[];
    public position: number;

    // Folded corners property - document-style visual
    public foldedCorners: boolean;

    // Aggregation properties
    public isAggregated: boolean;
    public aggregatedCount: number;
    public aggregatedItems: NodeConfig[];
    private isExpanded: boolean = false;

    // Rendering
    public element: SVGGElement;
    private background: SVGPathElement;        // Path for folded corner or regular rect
    private borderPath: SVGPathElement;
    private foldTriangle: SVGPathElement | null = null;  // Folded corner triangle
    private titleText: SVGTextElement;
    private stateText: SVGTextElement;
    private detailTexts: SVGTextElement[] = [];
    private icon: SVGTextElement | null = null;
    private statusIcon: SVGTextElement | null = null;  // Icon-only mode for Level 4
    private stackElements: SVGPathElement[] = [];      // Stack shadow elements
    private counterBadge: SVGGElement | null = null;   // Counter badge group

    // Display state
    private displayState: ProcessFlowDisplayState = ProcessFlowDisplayState.Regular;

    // Zoom level
    private currentZoomLevel: ProcessFlowZoomLevel = ProcessFlowZoomLevel.Two;
    private currentZoomConfig: ZoomLevelConfig = ZOOM_LEVEL_CONFIG[ProcessFlowZoomLevel.Two];

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

        // Folded corners property - defaults to true for SAP signature style
        this.foldedCorners = config.foldedCorners !== undefined ? config.foldedCorners : true;

        // Aggregation properties
        this.isAggregated = config.isAggregated || false;
        this.aggregatedCount = config.aggregatedCount || 0;
        this.aggregatedItems = config.aggregatedItems || [];

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

        // Add aggregated class if this is an aggregated node
        if (this.isAggregated) {
            g.classList.add('aggregated-node');
        }

        // Add folded-corner class if enabled
        if (this.foldedCorners) {
            g.classList.add('folded-corner');
        }

        const layout = PROCESS_FLOW_LAYOUT.node;

        // Render stack layers first (behind the main node) if aggregated
        if (this.isAggregated) {
            this.renderStack(g);
        }

        // Create background (with or without folded corner)
        this.background = this.foldedCorners
            ? this.createFoldedCornerPath()
            : this.createRegularPath();
        g.appendChild(this.background);

        // Create border
        this.borderPath = this.createBorderPath();
        g.appendChild(this.borderPath);

        // Render folded corner triangle if enabled
        if (this.foldedCorners) {
            this.foldTriangle = this.renderFoldedCorner();
            g.appendChild(this.foldTriangle);
        }

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

        // Create centered status icon for Level 4 (icon-only mode)
        this.statusIcon = this.createStatusIcon();
        g.appendChild(this.statusIcon);
        this.statusIcon.style.display = 'none';  // Hidden by default

        // Render counter badge on top-right if aggregated
        if (this.isAggregated && this.aggregatedCount > 1) {
            this.renderCounter(g);
        }

        // Add hover effects
        g.style.cursor = 'pointer';
        g.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';

        return g;
    }

    /**
     * Creates a large centered status icon for Level 4 zoom (icon-only mode)
     */
    private createStatusIcon(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('class', 'node-status-icon');
        // Center in level 4 node size (60x40)
        text.setAttribute('x', '30');
        text.setAttribute('y', '28');
        text.setAttribute('font-size', '24');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'middle');
        text.setAttribute('fill', this.getStateColors().text);
        text.textContent = this.getIconForState();
        return text;
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

    /**
     * Creates a regular rectangular path (no folded corner)
     */
    private createRegularPath(): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'node-background');

        const w = PROCESS_FLOW_LAYOUT.node.width;
        const h = PROCESS_FLOW_LAYOUT.node.height;
        const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;

        // Create regular rounded rectangle path
        const pathData = `
            M ${r} 0
            L ${w - r} 0
            Q ${w} 0, ${w} ${r}
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

    /**
     * Renders the folded corner triangle element
     * Creates a triangle in the top-right corner to simulate a paper fold
     * The fold is approximately 12px x 12px
     */
    private renderFoldedCorner(): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'node-fold-triangle');

        const w = PROCESS_FLOW_LAYOUT.node.width;
        const foldSize = 12;  // Folded corner size (12px x 12px)

        // Create the fold triangle in the top-right corner
        // Triangle points: top-right cut corner start, fold apex, and right edge
        const pathData = `
            M ${w - foldSize} 0
            L ${w - foldSize} ${foldSize}
            L ${w} ${foldSize}
            Z
        `.trim().replace(/\s+/g, ' ');

        path.setAttribute('d', pathData);

        // Use a lighter shade of the background color to simulate paper fold shadow
        const colors = this.getStateColors();
        path.setAttribute('fill', this.getLighterShade(colors.background));
        path.setAttribute('stroke', colors.border);
        path.setAttribute('stroke-width', '1');
        path.style.transition = 'fill 0.3s ease';
        path.style.filter = 'drop-shadow(1px 1px 1px rgba(0, 0, 0, 0.15))';

        return path;
    }

    /**
     * Returns a lighter shade of the given hex color for the fold effect
     */
    private getLighterShade(hexColor: string): string {
        // Parse hex color
        const hex = hexColor.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);

        // Lighten by 30%
        const factor = 0.3;
        const newR = Math.min(255, Math.round(r + (255 - r) * factor));
        const newG = Math.min(255, Math.round(g + (255 - g) * factor));
        const newB = Math.min(255, Math.round(b + (255 - b) * factor));

        // Helper to pad hex values (compatible with older ES targets)
        const toHex = (n: number): string => {
            const hex = n.toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        };

        return '#' + toHex(newR) + toHex(newG) + toHex(newB);
    }

    private createBorderPath(): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'node-border');

        const w = PROCESS_FLOW_LAYOUT.node.width;
        const h = PROCESS_FLOW_LAYOUT.node.height;
        const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
        const foldSize = 12;

        // Create path based on folded corners setting
        let pathData: string;
        if (this.foldedCorners) {
            // Path with folded top-right corner
            pathData = `
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
        } else {
            // Regular rounded rectangle
            pathData = `
                M ${r} 0
                L ${w - r} 0
                Q ${w} 0, ${w} ${r}
                L ${w} ${h - r}
                Q ${w} ${h}, ${w - r} ${h}
                L ${r} ${h}
                Q 0 ${h}, 0 ${h - r}
                L 0 ${r}
                Q 0 0, ${r} 0
                Z
            `.trim().replace(/\s+/g, ' ');
        }

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
    // Aggregation (Stack) Rendering
    // ========================================================================

    /**
     * Renders offset rectangles behind the main node to create a stacked cards effect.
     * The stack depth is based on the aggregated count (max 3 visible layers).
     */
    private renderStack(container: SVGGElement): void {
        const w = PROCESS_FLOW_LAYOUT.node.width;
        const h = PROCESS_FLOW_LAYOUT.node.height;
        const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
        const foldSize = 12;

        // Determine number of visible stack layers (max 3 for visual clarity)
        const stackLayers = Math.min(this.aggregatedCount - 1, 3);
        const stackOffset = 4;  // Pixel offset between stack layers

        // Create stack layers from back to front
        for (let i = stackLayers; i >= 1; i--) {
            const offsetX = i * stackOffset;
            const offsetY = i * stackOffset;

            const stackPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            stackPath.setAttribute('class', 'aggregate-stack');

            // Create the same folded corner path but offset
            const pathData = `
                M ${r + offsetX} ${offsetY}
                L ${w - foldSize + offsetX} ${offsetY}
                L ${w + offsetX} ${foldSize + offsetY}
                L ${w + offsetX} ${h - r + offsetY}
                Q ${w + offsetX} ${h + offsetY}, ${w - r + offsetX} ${h + offsetY}
                L ${r + offsetX} ${h + offsetY}
                Q ${offsetX} ${h + offsetY}, ${offsetX} ${h - r + offsetY}
                L ${offsetX} ${r + offsetY}
                Q ${offsetX} ${offsetY}, ${r + offsetX} ${offsetY}
                Z
            `.trim().replace(/\s+/g, ' ');

            stackPath.setAttribute('d', pathData);

            // Use a lighter version of the state color for stack layers
            const colors = this.getStateColors();
            stackPath.setAttribute('fill', colors.background);
            stackPath.setAttribute('stroke', colors.border);
            stackPath.setAttribute('stroke-width', '1');
            stackPath.setAttribute('opacity', (0.3 + (stackLayers - i) * 0.15).toString());
            stackPath.style.transition = 'opacity 0.3s ease';

            container.appendChild(stackPath);
            this.stackElements.push(stackPath);
        }
    }

    /**
     * Renders a count badge on the top-right corner showing the number of aggregated items.
     */
    private renderCounter(container: SVGGElement): void {
        const w = PROCESS_FLOW_LAYOUT.node.width;
        const badgeRadius = 12;
        const badgeX = w - 6;
        const badgeY = -6;

        // Create badge group
        this.counterBadge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.counterBadge.setAttribute('class', 'aggregate-counter');

        // Badge circle background
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', badgeX.toString());
        circle.setAttribute('cy', badgeY.toString());
        circle.setAttribute('r', badgeRadius.toString());
        circle.setAttribute('fill', '#0a6ed1');  // SAP Blue
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', '2');
        this.counterBadge.appendChild(circle);

        // Badge text (count)
        const countText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        countText.setAttribute('x', badgeX.toString());
        countText.setAttribute('y', (badgeY + 1).toString());
        countText.setAttribute('text-anchor', 'middle');
        countText.setAttribute('dominant-baseline', 'middle');
        countText.setAttribute('font-size', '10');
        countText.setAttribute('font-weight', 'bold');
        countText.setAttribute('fill', '#ffffff');
        countText.setAttribute('font-family', '"72", "72full", Arial, Helvetica, sans-serif');

        // Display count with + for larger numbers
        const displayCount = this.aggregatedCount > 99 ? '99+' : this.aggregatedCount.toString();
        countText.textContent = displayCount;
        this.counterBadge.appendChild(countText);

        container.appendChild(this.counterBadge);
    }

    /**
     * Expands the aggregated node to show individual items.
     * Emits an event that the parent ProcessFlow can handle.
     * @returns The list of aggregated items if available
     */
    public expandAggregate(): NodeConfig[] {
        if (!this.isAggregated || this.isExpanded) {
            return [];
        }

        this.isExpanded = true;
        this.element.classList.add('aggregate-expanded');

        // Hide stack elements when expanded
        for (const stackEl of this.stackElements) {
            stackEl.style.opacity = '0';
        }

        // Hide counter badge when expanded
        if (this.counterBadge) {
            this.counterBadge.style.opacity = '0';
        }

        return this.aggregatedItems;
    }

    /**
     * Collapses the expanded node back to aggregated view.
     */
    public collapseAggregate(): void {
        if (!this.isAggregated || !this.isExpanded) {
            return;
        }

        this.isExpanded = false;
        this.element.classList.remove('aggregate-expanded');

        // Show stack elements when collapsed
        const stackLayers = Math.min(this.aggregatedCount - 1, 3);
        for (let i = 0; i < this.stackElements.length; i++) {
            const stackEl = this.stackElements[i];
            stackEl.style.opacity = (0.3 + (stackLayers - i - 1) * 0.15).toString();
        }

        // Show counter badge when collapsed
        if (this.counterBadge) {
            this.counterBadge.style.opacity = '1';
        }
    }

    /**
     * Returns whether the aggregate is currently expanded.
     */
    public isAggregateExpanded(): boolean {
        return this.isExpanded;
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
        return this.currentZoomConfig.nodeWidth;
    }

    public getHeight(): number {
        return this.currentZoomConfig.nodeHeight;
    }

    // ========================================================================
    // Zoom Level Management - SAP Fiori Semantic Zoom
    // ========================================================================

    /**
     * Sets the zoom level and updates element visibility according to SAP Fiori standards:
     * - Level 1: Largest nodes - header, status, 2 attributes
     * - Level 2: Standard (auto for screens >1024px) - header, status, 1 attribute
     * - Level 3: Reduced (auto for 600-1023px) - header and status only
     * - Level 4: Smallest (auto for <600px) - status icon only
     */
    public setZoomLevel(level: ProcessFlowZoomLevel): void {
        this.currentZoomLevel = level;
        this.currentZoomConfig = ZOOM_LEVEL_CONFIG[level];

        const config = this.currentZoomConfig;

        // Update node shape/size
        this.updateNodeShape(config.nodeWidth, config.nodeHeight);

        // Show/hide elements based on zoom level
        this.updateElementVisibility(config);

        // Update status icon position for current size
        if (this.statusIcon) {
            this.statusIcon.setAttribute('x', (config.nodeWidth / 2).toString());
            this.statusIcon.setAttribute('y', (config.nodeHeight / 2 + 4).toString());
        }
    }

    public getZoomLevel(): ProcessFlowZoomLevel {
        return this.currentZoomLevel;
    }

    /**
     * Updates the node background and border paths for the new dimensions
     */
    private updateNodeShape(width: number, height: number): void {
        const r = Math.min(PROCESS_FLOW_LAYOUT.node.cornerRadius, width / 10);
        const foldSize = Math.min(12, width / 13);

        let pathData: string;
        if (this.foldedCorners) {
            // Path with folded top-right corner
            pathData = `
                M ${r} 0
                L ${width - foldSize} 0
                L ${width} ${foldSize}
                L ${width} ${height - r}
                Q ${width} ${height}, ${width - r} ${height}
                L ${r} ${height}
                Q 0 ${height}, 0 ${height - r}
                L 0 ${r}
                Q 0 0, ${r} 0
                Z
            `.trim().replace(/\s+/g, ' ');
        } else {
            // Regular rounded rectangle
            pathData = `
                M ${r} 0
                L ${width - r} 0
                Q ${width} 0, ${width} ${r}
                L ${width} ${height - r}
                Q ${width} ${height}, ${width - r} ${height}
                L ${r} ${height}
                Q 0 ${height}, 0 ${height - r}
                L 0 ${r}
                Q 0 0, ${r} 0
                Z
            `.trim().replace(/\s+/g, ' ');
        }

        this.background.setAttribute('d', pathData);
        this.borderPath.setAttribute('d', pathData);

        // Update fold triangle position if it exists
        if (this.foldTriangle && this.foldedCorners) {
            const foldPathData = `
                M ${width - foldSize} 0
                L ${width - foldSize} ${foldSize}
                L ${width} ${foldSize}
                Z
            `.trim().replace(/\s+/g, ' ');
            this.foldTriangle.setAttribute('d', foldPathData);
        }
    }

    /**
     * Updates visibility of elements based on zoom level configuration
     */
    private updateElementVisibility(config: ZoomLevelConfig): void {
        // Header/title visibility
        this.titleText.style.display = config.showHeader ? 'block' : 'none';

        // Status text visibility
        this.stateText.style.display = config.showStatus && config.showHeader ? 'block' : 'none';

        // Icon visibility (small icon next to title)
        if (this.icon) {
            this.icon.style.display = config.showHeader ? 'block' : 'none';
        }

        // Detail texts / attributes
        for (let i = 0; i < this.detailTexts.length; i++) {
            if (i === 0) {
                this.detailTexts[i].style.display = config.showAttr1 ? 'block' : 'none';
            } else if (i === 1) {
                this.detailTexts[i].style.display = config.showAttr2 ? 'block' : 'none';
            } else {
                // Additional attributes follow attr2 visibility
                this.detailTexts[i].style.display = config.showAttr2 ? 'block' : 'none';
            }
        }

        // Status icon for Level 4 (icon-only mode)
        if (this.statusIcon) {
            this.statusIcon.style.display = !config.showHeader && config.showStatus ? 'block' : 'none';
        }
    }

    // ========================================================================
    // State Management
    // ========================================================================

    public setState(state: ProcessFlowNodeState): void {
        this.state = state;
        this.updateColors();
        this.stateText.textContent = this.getStateLabel();
        if (this.statusIcon) {
            this.statusIcon.textContent = this.getIconForState();
        }
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

        if (this.statusIcon) {
            this.statusIcon.setAttribute('fill', colors.text);
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
    // Path Highlighting
    // ========================================================================

    /**
     * Sets the highlighted state for path highlighting
     * @param highlighted Whether the node should be highlighted
     */
    public setHighlighted(highlighted: boolean): void {
        if (highlighted) {
            this.element.classList.add('highlighted');
            this.element.classList.remove('dimmed');
            this.element.style.filter = 'drop-shadow(0 0 8px rgba(10, 110, 209, 0.6))';
            this.borderPath.setAttribute('stroke-width', '3');
        } else {
            this.element.classList.remove('highlighted');
            this.element.style.filter = '';
            this.borderPath.setAttribute('stroke-width', PROCESS_FLOW_LAYOUT.node.borderWidth.toString());
        }
    }

    /**
     * Sets the dimmed state for path highlighting
     * @param dimmed Whether the node should be dimmed
     */
    public setDimmed(dimmed: boolean): void {
        if (dimmed) {
            this.element.classList.add('dimmed');
            this.element.classList.remove('highlighted');
            this.element.style.opacity = '0.3';
            this.element.style.filter = 'grayscale(50%)';
        } else {
            this.element.classList.remove('dimmed');
            this.element.style.opacity = '1';
            this.element.style.filter = '';
        }
    }

    // ========================================================================
    // Export
    // ========================================================================

    public toJSON(): NodeConfig {
        const result: NodeConfig = {
            id: this.id,
            lane: this.lane,
            title: this.title,
            state: this.state,
            texts: this.texts,
            children: this.children,
            position: this.position,
            foldedCorners: this.foldedCorners
        };

        // Include aggregation properties if this is an aggregated node
        if (this.isAggregated) {
            result.isAggregated = this.isAggregated;
            result.aggregatedCount = this.aggregatedCount;
            result.aggregatedItems = this.aggregatedItems;
        }

        return result;
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
