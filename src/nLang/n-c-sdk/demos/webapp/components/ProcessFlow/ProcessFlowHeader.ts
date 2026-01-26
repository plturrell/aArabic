/**
 * ProcessFlowHeader - SAP Fiori Header with Circular Progress Charts
 * Shows lane icons with status distribution as circular arcs
 */

import { PROCESS_FLOW_COLORS, PROCESS_FLOW_LAYOUT } from './types';

// ============================================================================
// Types
// ============================================================================

export interface StatusCounts {
    positive: number;   // Success/green
    negative: number;   // Error/red
    neutral: number;    // In progress/blue
    critical: number;   // Warning/yellow
    planned: number;    // Planned/gray
}

export interface LaneHeaderConfig {
    laneId: string;
    title: string;
    icon?: string;  // Icon character or SVG path
    statusCounts: StatusCounts;
}

// ============================================================================
// Constants
// ============================================================================

const HEADER_HEIGHT = 80;
const CIRCLE_RADIUS = 24;
const ARC_RADIUS = 28;
const ARC_THICKNESS = 4;
const ICON_FONT_SIZE = 16;
const LABEL_FONT_SIZE = 12;

const STATUS_COLORS: Record<keyof StatusCounts, string> = {
    positive: PROCESS_FLOW_COLORS.positive.background,  // Green
    negative: PROCESS_FLOW_COLORS.negative.background,  // Red
    neutral: PROCESS_FLOW_COLORS.neutral.background,    // Blue
    critical: PROCESS_FLOW_COLORS.critical.background,  // Yellow/Orange
    planned: PROCESS_FLOW_COLORS.planned.border         // Gray
};

const DEFAULT_ICONS: Record<string, string> = {
    document: '◷',
    approval: '✓',
    process: '⚙',
    default: '●'
};

// ============================================================================
// ProcessFlowHeader Class
// ============================================================================

export class ProcessFlowHeader {
    private container: HTMLElement;
    private headerElement: HTMLDivElement;
    private svgElement: SVGSVGElement;
    private lanes: LaneHeaderConfig[] = [];
    private laneWidth: number = 160;
    private laneOffset: number = PROCESS_FLOW_LAYOUT.spacing.laneHeader + PROCESS_FLOW_LAYOUT.spacing.leftMargin;
    
    // Elements per lane
    private laneElements: Map<string, {
        group: SVGGElement;
        circle: SVGCircleElement;
        arcs: SVGPathElement[];
        icon: SVGTextElement;
        label: SVGTextElement;
    }> = new Map();

    // Tooltip element
    private tooltip: HTMLDivElement | null = null;

    // Event callbacks
    private onLaneClick: ((laneId: string) => void) | null = null;

    constructor(container: HTMLElement) {
        this.container = container;
        this.headerElement = this.createHeaderElement();
        this.svgElement = this.createSVGElement();
        this.headerElement.appendChild(this.svgElement);
        this.container.insertBefore(this.headerElement, this.container.firstChild);
        this.createTooltip();
    }

    // ========================================================================
    // Element Creation
    // ========================================================================

    private createHeaderElement(): HTMLDivElement {
        const header = document.createElement('div');
        header.className = 'process-flow-header';
        header.style.cssText = `
            height: ${HEADER_HEIGHT}px;
            background: #f7f7f7;
            border-bottom: 1px solid #d9d9d9;
            position: relative;
            overflow: hidden;
            font-family: "72", "72full", Arial, Helvetica, sans-serif;
        `;
        return header;
    }

    private createSVGElement(): SVGSVGElement {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'process-flow-header-svg');
        svg.style.width = '100%';
        svg.style.height = '100%';
        return svg;
    }

    private createTooltip(): void {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'process-flow-header-tooltip';
        this.tooltip.style.cssText = `
            position: absolute;
            background: rgba(50, 54, 58, 0.95);
            color: #ffffff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-family: "72", "72full", Arial, Helvetica, sans-serif;
            white-space: nowrap;
            pointer-events: none;
            z-index: 1000;
            display: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        document.body.appendChild(this.tooltip);
    }

    // ========================================================================
    // Lane Management
    // ========================================================================

    public setLanes(lanes: LaneHeaderConfig[]): void {
        this.lanes = lanes;
        this.render();
    }

    public updateLane(laneId: string, config: Partial<LaneHeaderConfig>): void {
        const lane = this.lanes.find(l => l.laneId === laneId);
        if (lane) {
            Object.assign(lane, config);
            this.updateLaneElement(laneId);
        }
    }

    public setLaneWidth(width: number): void {
        this.laneWidth = width;
        this.render();
    }

    public setLaneOffset(offset: number): void {
        this.laneOffset = offset;
        this.render();
    }

    public setOnLaneClick(callback: (laneId: string) => void): void {
        this.onLaneClick = callback;
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    private render(): void {
        // Clear existing content
        while (this.svgElement.firstChild) {
            this.svgElement.removeChild(this.svgElement.firstChild);
        }
        this.laneElements.clear();

        // Render each lane header
        this.lanes.forEach((lane, index) => {
            const x = this.laneOffset + (index * this.laneWidth) + (this.laneWidth / 2);
            const y = HEADER_HEIGHT / 2;
            const laneGroup = this.renderLaneElement(lane, x, y);
            this.svgElement.appendChild(laneGroup);
        });
    }

    private renderLaneElement(lane: LaneHeaderConfig, cx: number, cy: number): SVGGElement {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'process-flow-header-lane');
        group.setAttribute('data-lane-id', lane.laneId);
        group.style.cursor = 'pointer';

        // Create background circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', cx.toString());
        circle.setAttribute('cy', cy.toString());
        circle.setAttribute('r', CIRCLE_RADIUS.toString());
        circle.setAttribute('fill', '#ffffff');
        circle.setAttribute('stroke', '#d9d9d9');
        circle.setAttribute('stroke-width', '1');
        circle.setAttribute('class', 'lane-circle');
        group.appendChild(circle);

        // Create status arcs
        const arcs = this.createStatusArcs(lane.statusCounts, cx, cy);
        arcs.forEach(arc => group.appendChild(arc));

        // Create icon text
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        icon.setAttribute('x', cx.toString());
        icon.setAttribute('y', (cy + 5).toString());
        icon.setAttribute('text-anchor', 'middle');
        icon.setAttribute('dominant-baseline', 'middle');
        icon.setAttribute('font-size', ICON_FONT_SIZE.toString());
        icon.setAttribute('fill', '#32363a');
        icon.setAttribute('class', 'lane-icon');
        icon.textContent = lane.icon || DEFAULT_ICONS[lane.laneId] || DEFAULT_ICONS.default;
        group.appendChild(icon);

        // Create label below circle
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', cx.toString());
        label.setAttribute('y', (cy + CIRCLE_RADIUS + 16).toString());
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', LABEL_FONT_SIZE.toString());
        label.setAttribute('font-family', '"72", "72full", Arial, Helvetica, sans-serif');
        label.setAttribute('font-weight', '600');
        label.setAttribute('fill', '#32363a');
        label.setAttribute('class', 'lane-label-text');
        label.textContent = this.truncateLabel(lane.title, 14);
        group.appendChild(label);

        // Store elements for updates
        this.laneElements.set(lane.laneId, {
            group,
            circle,
            arcs,
            icon,
            label
        });

        // Add event listeners
        group.addEventListener('click', (e) => this.handleLaneClick(e, lane.laneId));
        group.addEventListener('mouseenter', (e) => this.handleLaneHover(e, lane, true));
        group.addEventListener('mouseleave', (e) => this.handleLaneHover(e, lane, false));

        return group;
    }

    private createStatusArcs(statusCounts: StatusCounts, cx: number, cy: number): SVGPathElement[] {
        const arcs: SVGPathElement[] = [];
        const total = statusCounts.positive + statusCounts.negative +
                     statusCounts.neutral + statusCounts.critical + statusCounts.planned;

        if (total === 0) return arcs;

        const statusOrder: (keyof StatusCounts)[] = ['positive', 'neutral', 'critical', 'negative', 'planned'];
        let currentAngle = -90; // Start from top

        statusOrder.forEach(status => {
            const count = statusCounts[status];
            if (count === 0) return;

            const percentage = count / total;
            const sweepAngle = percentage * 360;
            const endAngle = currentAngle + sweepAngle;

            const arc = this.createArcPath(cx, cy, ARC_RADIUS, currentAngle, endAngle, STATUS_COLORS[status]);
            arc.setAttribute('class', `status-arc status-arc-${status}`);
            arcs.push(arc);

            currentAngle = endAngle;
        });

        return arcs;
    }

    private createArcPath(cx: number, cy: number, radius: number, startAngle: number, endAngle: number, color: string): SVGPathElement {
        const startRad = (startAngle * Math.PI) / 180;
        const endRad = (endAngle * Math.PI) / 180;

        const x1 = cx + radius * Math.cos(startRad);
        const y1 = cy + radius * Math.sin(startRad);
        const x2 = cx + radius * Math.cos(endRad);
        const y2 = cy + radius * Math.sin(endRad);

        const largeArc = (endAngle - startAngle) > 180 ? 1 : 0;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const d = `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
        path.setAttribute('d', d);
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', ARC_THICKNESS.toString());
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke-linecap', 'round');

        return path;
    }

    private truncateLabel(text: string, maxLength: number): string {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 1) + '…';
    }

    // ========================================================================
    // Update Methods
    // ========================================================================

    private updateLaneElement(laneId: string): void {
        const lane = this.lanes.find(l => l.laneId === laneId);
        const elements = this.laneElements.get(laneId);

        if (!lane || !elements) return;

        // Update icon
        elements.icon.textContent = lane.icon || DEFAULT_ICONS[lane.laneId] || DEFAULT_ICONS.default;

        // Update label
        elements.label.textContent = this.truncateLabel(lane.title, 14);

        // Update arcs - remove old and create new
        const cx = parseFloat(elements.circle.getAttribute('cx') || '0');
        const cy = parseFloat(elements.circle.getAttribute('cy') || '0');

        // Remove old arcs
        elements.arcs.forEach(arc => arc.remove());

        // Create new arcs
        const newArcs = this.createStatusArcs(lane.statusCounts, cx, cy);
        newArcs.forEach(arc => {
            elements.group.insertBefore(arc, elements.icon);
        });

        elements.arcs = newArcs;
    }

    // ========================================================================
    // Event Handlers
    // ========================================================================

    private handleLaneClick(event: MouseEvent, laneId: string): void {
        event.stopPropagation();
        if (this.onLaneClick) {
            this.onLaneClick(laneId);
        }
    }

    private handleLaneHover(event: MouseEvent, lane: LaneHeaderConfig, isEnter: boolean): void {
        const elements = this.laneElements.get(lane.laneId);
        if (!elements) return;

        if (isEnter) {
            // Highlight circle on hover
            elements.circle.setAttribute('stroke', '#0a6ed1');
            elements.circle.setAttribute('stroke-width', '2');

            // Show tooltip
            this.showTooltip(event, lane);
        } else {
            // Reset circle
            elements.circle.setAttribute('stroke', '#d9d9d9');
            elements.circle.setAttribute('stroke-width', '1');

            // Hide tooltip
            this.hideTooltip();
        }
    }

    // ========================================================================
    // Tooltip
    // ========================================================================

    private showTooltip(event: MouseEvent, lane: LaneHeaderConfig): void {
        if (!this.tooltip) return;

        const counts = lane.statusCounts;
        const total = counts.positive + counts.negative + counts.neutral + counts.critical + counts.planned;

        // Build tooltip content
        const lines: string[] = [`<strong>${lane.title}</strong>`];

        if (counts.positive > 0) {
            lines.push(`<span style="color: ${STATUS_COLORS.positive}">●</span> Success: ${counts.positive} (${Math.round(counts.positive / total * 100)}%)`);
        }
        if (counts.neutral > 0) {
            lines.push(`<span style="color: ${STATUS_COLORS.neutral}">●</span> In Progress: ${counts.neutral} (${Math.round(counts.neutral / total * 100)}%)`);
        }
        if (counts.critical > 0) {
            lines.push(`<span style="color: ${STATUS_COLORS.critical}">●</span> Warning: ${counts.critical} (${Math.round(counts.critical / total * 100)}%)`);
        }
        if (counts.negative > 0) {
            lines.push(`<span style="color: ${STATUS_COLORS.negative}">●</span> Error: ${counts.negative} (${Math.round(counts.negative / total * 100)}%)`);
        }
        if (counts.planned > 0) {
            lines.push(`<span style="color: ${STATUS_COLORS.planned}">●</span> Planned: ${counts.planned} (${Math.round(counts.planned / total * 100)}%)`);
        }

        this.tooltip.innerHTML = lines.join('<br>');
        this.tooltip.style.display = 'block';

        // Position tooltip near mouse
        const offsetX = 15;
        const offsetY = 10;
        this.tooltip.style.left = `${event.pageX + offsetX}px`;
        this.tooltip.style.top = `${event.pageY + offsetY}px`;

        // Ensure tooltip stays within viewport
        const rect = this.tooltip.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            this.tooltip.style.left = `${event.pageX - rect.width - offsetX}px`;
        }
        if (rect.bottom > window.innerHeight) {
            this.tooltip.style.top = `${event.pageY - rect.height - offsetY}px`;
        }
    }

    private hideTooltip(): void {
        if (this.tooltip) {
            this.tooltip.style.display = 'none';
        }
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    public destroy(): void {
        // Remove tooltip from DOM
        if (this.tooltip && this.tooltip.parentNode) {
            this.tooltip.parentNode.removeChild(this.tooltip);
            this.tooltip = null;
        }

        // Clear event listeners by removing elements
        this.laneElements.forEach((elements) => {
            if (elements.group.parentNode) {
                elements.group.parentNode.removeChild(elements.group);
            }
        });
        this.laneElements.clear();

        // Remove header element
        if (this.headerElement.parentNode) {
            this.headerElement.parentNode.removeChild(this.headerElement);
        }

        // Clear references
        this.lanes = [];
        this.onLaneClick = null;
    }
}
