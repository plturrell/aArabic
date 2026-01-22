/**
 * GraphGroup - Visual grouping of nodes with collapse/expand functionality
 * Part of NetworkGraph component
 * SAP Fiori compliant with smooth transitions
 */
// Animation duration for SAP Fiori transitions (in ms)
const ANIMATION_DURATION = 300;
const HEADER_HEIGHT = 32;
const COLLAPSED_WIDTH = 160;
const COLLAPSED_HEIGHT = 60;
export class GraphGroup {
    constructor(config) {
        this.headerElement = null;
        this.contentElement = null;
        this.toggleButton = null;
        this.isAnimating = false;
        this.onToggleCallback = null;
        this.id = config.id;
        this.name = config.name;
        this.description = config.description;
        this.color = config.color || '#0070f2'; // SAP brand color default
        this.collapsed = config.collapsed || false;
        this.nodeIds = new Set(config.nodeIds || []);
        this.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.bounds = { x: 0, y: 0, width: 0, height: 0 };
        this.collapsedBounds = { x: 0, y: 0, width: COLLAPSED_WIDTH, height: COLLAPSED_HEIGHT };
        this.createElement();
    }
    createElement() {
        this.element.setAttribute('class', `graph-group ${this.collapsed ? 'collapsed' : 'expanded'}`);
        this.element.setAttribute('data-group-id', this.id);
        this.element.setAttribute('role', 'group');
        this.element.setAttribute('aria-label', this.name);
        this.element.setAttribute('aria-expanded', String(!this.collapsed));
    }
    onToggle(callback) {
        this.onToggleCallback = callback;
    }
    addNode(nodeId) {
        this.nodeIds.add(nodeId);
    }
    removeNode(nodeId) {
        this.nodeIds.delete(nodeId);
    }
    hasNode(nodeId) {
        return this.nodeIds.has(nodeId);
    }
    getNodeCount() {
        return this.nodeIds.size;
    }
    updateBounds(x, y, width, height) {
        this.bounds = { x, y, width, height };
        // Update collapsed bounds center position
        this.collapsedBounds.x = x + (width - COLLAPSED_WIDTH) / 2;
        this.collapsedBounds.y = y + (height - COLLAPSED_HEIGHT) / 2;
        this.render();
    }
    render() {
        // Clear existing content
        while (this.element.firstChild) {
            this.element.removeChild(this.element.firstChild);
        }
        const padding = 20;
        const activeBounds = this.collapsed ? this.collapsedBounds : this.bounds;
        const { x, y, width, height } = activeBounds;
        // Create content container for animation
        this.contentElement = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.contentElement.setAttribute('class', 'graph-group-content');
        // Background rectangle
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('class', 'graph-group-background');
        rect.setAttribute('x', String(x - padding));
        rect.setAttribute('y', String(y - padding));
        rect.setAttribute('width', String(width + padding * 2));
        rect.setAttribute('height', String(height + padding * 2 + HEADER_HEIGHT));
        rect.setAttribute('rx', '8');
        rect.setAttribute('fill', this.color);
        rect.setAttribute('fill-opacity', this.collapsed ? '0.15' : '0.08');
        rect.setAttribute('stroke', this.color);
        rect.setAttribute('stroke-width', '2');
        if (!this.collapsed) {
            rect.setAttribute('stroke-dasharray', '5,5');
        }
        this.contentElement.appendChild(rect);
        // Render header bar
        this.renderHeader(x - padding, y - padding, width + padding * 2);
        // Add node count badge when collapsed
        if (this.collapsed) {
            this.renderCollapsedBadge(x + width / 2, y + height / 2 + 10);
        }
        this.element.appendChild(this.contentElement);
        this.element.appendChild(this.headerElement);
    }
    renderHeader(x, y, width) {
        this.headerElement = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.headerElement.setAttribute('class', 'graph-group-header');
        // Header background
        const headerBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        headerBg.setAttribute('class', 'graph-group-header-bg');
        headerBg.setAttribute('x', String(x));
        headerBg.setAttribute('y', String(y));
        headerBg.setAttribute('width', String(width));
        headerBg.setAttribute('height', String(HEADER_HEIGHT));
        headerBg.setAttribute('rx', '8');
        headerBg.setAttribute('fill', this.color);
        headerBg.setAttribute('fill-opacity', '0.9');
        this.headerElement.appendChild(headerBg);
        // Clip bottom corners of header
        const headerClip = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        headerClip.setAttribute('x', String(x));
        headerClip.setAttribute('y', String(y + HEADER_HEIGHT - 8));
        headerClip.setAttribute('width', String(width));
        headerClip.setAttribute('height', '8');
        headerClip.setAttribute('fill', this.color);
        headerClip.setAttribute('fill-opacity', '0.9');
        this.headerElement.appendChild(headerClip);
        // Create toggle button
        this.toggleButton = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.toggleButton.setAttribute('class', 'graph-group-toggle');
        this.toggleButton.setAttribute('role', 'button');
        this.toggleButton.setAttribute('aria-label', this.collapsed ? 'Expand group' : 'Collapse group');
        this.toggleButton.setAttribute('tabindex', '0');
        this.toggleButton.style.cursor = 'pointer';
        // Toggle button background (hit area)
        const toggleBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        toggleBg.setAttribute('x', String(x + 4));
        toggleBg.setAttribute('y', String(y + 4));
        toggleBg.setAttribute('width', '24');
        toggleBg.setAttribute('height', '24');
        toggleBg.setAttribute('rx', '4');
        toggleBg.setAttribute('fill', 'transparent');
        toggleBg.setAttribute('class', 'toggle-hitarea');
        this.toggleButton.appendChild(toggleBg);
        // Toggle icon (▼ expanded, ▶ collapsed)
        const toggleIcon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        toggleIcon.setAttribute('class', 'graph-group-toggle-icon');
        toggleIcon.setAttribute('x', String(x + 16));
        toggleIcon.setAttribute('y', String(y + 22));
        toggleIcon.setAttribute('text-anchor', 'middle');
        toggleIcon.setAttribute('font-size', '12');
        toggleIcon.setAttribute('fill', 'white');
        toggleIcon.textContent = this.collapsed ? '▶' : '▼';
        this.toggleButton.appendChild(toggleIcon);
        // Add click handler
        this.toggleButton.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggle();
        });
        // Add keyboard handler
        this.toggleButton.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                e.stopPropagation();
                this.toggle();
            }
        });
        this.headerElement.appendChild(this.toggleButton);
        // Group label in header
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('class', 'graph-group-label');
        label.setAttribute('x', String(x + 32));
        label.setAttribute('y', String(y + 21));
        label.setAttribute('font-size', '13');
        label.setAttribute('font-weight', '600');
        label.setAttribute('fill', 'white');
        label.textContent = this.name;
        this.headerElement.appendChild(label);
        // Node count indicator in header
        const countBadge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        countBadge.setAttribute('class', 'graph-group-count');
        countBadge.setAttribute('x', String(x + width - 12));
        countBadge.setAttribute('y', String(y + 21));
        countBadge.setAttribute('text-anchor', 'end');
        countBadge.setAttribute('font-size', '11');
        countBadge.setAttribute('fill', 'rgba(255,255,255,0.8)');
        countBadge.textContent = `(${this.nodeIds.size})`;
        this.headerElement.appendChild(countBadge);
    }
    renderCollapsedBadge(cx, cy) {
        // Large node count display when collapsed
        const badge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        badge.setAttribute('class', 'graph-group-collapsed-badge');
        const badgeCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        badgeCircle.setAttribute('cx', String(cx));
        badgeCircle.setAttribute('cy', String(cy));
        badgeCircle.setAttribute('r', '18');
        badgeCircle.setAttribute('fill', this.color);
        badgeCircle.setAttribute('fill-opacity', '0.2');
        badgeCircle.setAttribute('stroke', this.color);
        badgeCircle.setAttribute('stroke-width', '2');
        badge.appendChild(badgeCircle);
        const countText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        countText.setAttribute('x', String(cx));
        countText.setAttribute('y', String(cy + 5));
        countText.setAttribute('text-anchor', 'middle');
        countText.setAttribute('font-size', '14');
        countText.setAttribute('font-weight', 'bold');
        countText.setAttribute('fill', this.color);
        countText.textContent = String(this.nodeIds.size);
        badge.appendChild(countText);
        this.contentElement?.appendChild(badge);
    }
    toggle() {
        if (this.isAnimating)
            return;
        this.isAnimating = true;
        const wasCollapsed = this.collapsed;
        this.collapsed = !this.collapsed;
        // Update ARIA state
        this.element.setAttribute('aria-expanded', String(!this.collapsed));
        this.element.classList.toggle('collapsed', this.collapsed);
        this.element.classList.toggle('expanded', !this.collapsed);
        // Add animation class
        this.element.classList.add('animating');
        // Trigger callback to notify parent of visibility changes
        if (this.onToggleCallback) {
            this.onToggleCallback({
                groupId: this.id,
                collapsed: this.collapsed,
                nodeIds: Array.from(this.nodeIds)
            });
        }
        // Re-render with animation
        this.render();
        // Remove animation class after transition
        setTimeout(() => {
            this.element.classList.remove('animating');
            this.isAnimating = false;
        }, ANIMATION_DURATION);
    }
    expand() {
        if (!this.collapsed)
            return;
        this.toggle();
    }
    collapse() {
        if (this.collapsed)
            return;
        this.toggle();
    }
    setCollapsed(collapsed) {
        if (this.collapsed !== collapsed) {
            this.toggle();
        }
    }
    destroy() {
        // Clean up event listeners
        if (this.toggleButton) {
            this.toggleButton.replaceWith(this.toggleButton.cloneNode(true));
        }
        this.element.remove();
        this.nodeIds.clear();
        this.onToggleCallback = null;
        this.headerElement = null;
        this.contentElement = null;
        this.toggleButton = null;
    }
}
//# sourceMappingURL=GraphGroup.js.map