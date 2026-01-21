/**
 * ProcessFlowLane - SAP-styled lane header
 * Exact SAP Fiori lane styling
 */
import { PROCESS_FLOW_COLORS, PROCESS_FLOW_LAYOUT } from './types';
export class ProcessFlowLane {
    constructor(config) {
        this.icon = null;
        // Position
        this.y = 0;
        this.height = PROCESS_FLOW_LAYOUT.node.height;
        this.id = config.id;
        this.label = config.label;
        this.position = config.position;
        this.state = config.state;
        // Create SVG element
        this.element = this.createElement();
    }
    // ========================================================================
    // Rendering
    // ========================================================================
    createElement() {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'process-flow-lane');
        g.setAttribute('data-lane-id', this.id);
        // Create background
        this.background = this.createBackground();
        g.appendChild(this.background);
        // Create label
        this.labelText = this.createLabel();
        g.appendChild(this.labelText);
        return g;
    }
    createBackground() {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('class', 'lane-background');
        rect.setAttribute('width', PROCESS_FLOW_LAYOUT.spacing.laneHeader.toString());
        rect.setAttribute('height', this.height.toString());
        rect.setAttribute('fill', this.position % 2 === 0 ?
            PROCESS_FLOW_COLORS.lane.default :
            PROCESS_FLOW_COLORS.lane.alternate);
        rect.setAttribute('stroke', PROCESS_FLOW_COLORS.border);
        rect.setAttribute('stroke-width', '1');
        return rect;
    }
    createLabel() {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('class', 'lane-label');
        text.setAttribute('x', (PROCESS_FLOW_LAYOUT.spacing.laneHeader / 2).toString());
        text.setAttribute('y', (this.height / 2).toString());
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'middle');
        text.setAttribute('font-size', '13');
        text.setAttribute('font-weight', '600');
        text.setAttribute('fill', PROCESS_FLOW_COLORS.text.primary);
        text.textContent = this.label;
        return text;
    }
    // ========================================================================
    // Position Management
    // ========================================================================
    setPosition(y) {
        this.y = y;
        this.element.setAttribute('transform', `translate(0, ${y})`);
    }
    setHeight(height) {
        this.height = height;
        this.background.setAttribute('height', height.toString());
        this.labelText.setAttribute('y', (height / 2).toString());
    }
    // ========================================================================
    // Cleanup
    // ========================================================================
    destroy() {
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }
}
//# sourceMappingURL=ProcessFlowLane.js.map