/**
 * NodePopover - Details and Links Popover for NetworkGraph nodes
 * SAP Fiori Design System compliant
 */
import { NodeStatus, SAP_COLORS } from './types';
export class NodePopover {
    constructor(container) {
        this.popoverElement = null;
        this.node = null;
        this.visible = false;
        this.activeTab = 'details';
        // Callbacks
        this.onNavigateCallback = null;
        this.onExpandCallback = null;
        this.onCollapseCallback = null;
        this.onViewDetailsCallback = null;
        this.container = container;
        this.boundHandleClickOutside = this.handleClickOutside.bind(this);
        this.boundHandleKeyDown = this.handleKeyDown.bind(this);
    }
    show(config) {
        // Hide existing popover if any
        this.hide();
        this.node = config.node;
        this.visible = true;
        this.activeTab = 'details';
        // Create popover element
        this.popoverElement = this.createElement(config);
        this.container.appendChild(this.popoverElement);
        // Position with smart viewport collision detection
        this.positionPopover(config.position);
        // Add event listeners
        document.addEventListener('mousedown', this.boundHandleClickOutside);
        document.addEventListener('keydown', this.boundHandleKeyDown);
        // Trigger fade-in animation
        requestAnimationFrame(() => {
            if (this.popoverElement) {
                this.popoverElement.classList.add('visible');
            }
        });
    }
    hide() {
        if (!this.visible || !this.popoverElement)
            return;
        this.visible = false;
        this.popoverElement.classList.remove('visible');
        // Remove after animation
        const element = this.popoverElement;
        setTimeout(() => {
            if (element && element.parentNode) {
                element.parentNode.removeChild(element);
            }
        }, 200);
        this.popoverElement = null;
        this.node = null;
        // Remove event listeners
        document.removeEventListener('mousedown', this.boundHandleClickOutside);
        document.removeEventListener('keydown', this.boundHandleKeyDown);
    }
    isVisible() {
        return this.visible;
    }
    on(event, callback) {
        switch (event) {
            case 'navigate':
                this.onNavigateCallback = callback;
                break;
            case 'expand':
                this.onExpandCallback = callback;
                break;
            case 'collapse':
                this.onCollapseCallback = callback;
                break;
            case 'viewDetails':
                this.onViewDetailsCallback = callback;
                break;
        }
    }
    destroy() {
        this.hide();
        this.onNavigateCallback = null;
        this.onExpandCallback = null;
        this.onCollapseCallback = null;
        this.onViewDetailsCallback = null;
    }
    // ========================================================================
    // Private Methods
    // ========================================================================
    createElement(config) {
        const popover = document.createElement('div');
        popover.className = 'node-popover';
        popover.innerHTML = this.buildPopoverHTML(config);
        // Setup event handlers
        this.setupPopoverEvents(popover, config);
        return popover;
    }
    buildPopoverHTML(config) {
        const { node, connectedNodes } = config;
        const statusColor = this.getStatusColor(node.status);
        const statusText = this.getStatusText(node.status);
        return `
            <div class="popover-header">
                <div class="popover-header-content">
                    <span class="popover-icon">${this.getIconText(node.type)}</span>
                    <div class="popover-title-section">
                        <h3 class="popover-title">${this.escapeHtml(node.name)}</h3>
                        <span class="popover-status" style="background-color: ${statusColor}">
                            ${statusText}
                        </span>
                    </div>
                </div>
                <button class="popover-close" aria-label="Close">&times;</button>
            </div>

            <div class="popover-tabs">
                <button class="popover-tab active" data-tab="details">Details</button>
                <button class="popover-tab" data-tab="links">Links (${connectedNodes.length})</button>
            </div>

            <div class="popover-content">
                ${this.buildDetailsTab(node)}
                ${this.buildLinksTab(connectedNodes)}
            </div>

            <div class="popover-footer">
                <button class="popover-action secondary" data-action="expand">Expand</button>
                <button class="popover-action primary" data-action="viewDetails">View Details</button>
            </div>
        `;
    }
    buildDetailsTab(node) {
        const metrics = node.metrics || { totalRequests: 0, avgLatency: 0, successRate: 0 };
        return `
            <div class="popover-tab-content" data-content="details">
                <div class="details-form">
                    <div class="detail-row">
                        <span class="detail-label">ID</span>
                        <span class="detail-value">${this.escapeHtml(node.id)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Type</span>
                        <span class="detail-value">${this.escapeHtml(node.type)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Model</span>
                        <span class="detail-value">${this.escapeHtml(node.model || 'N/A')}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Description</span>
                        <span class="detail-value">${this.escapeHtml(node.description || 'No description')}</span>
                    </div>
                    <div class="details-divider"></div>
                    <div class="details-section-title">Metrics</div>
                    <div class="detail-row">
                        <span class="detail-label">Total Requests</span>
                        <span class="detail-value">${metrics.totalRequests.toLocaleString()}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Avg Latency</span>
                        <span class="detail-value">${metrics.avgLatency.toFixed(2)} ms</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Success Rate</span>
                        <span class="detail-value">${(metrics.successRate * 100).toFixed(1)}%</span>
                    </div>
                    ${metrics.throughput !== undefined ? `
                    <div class="detail-row">
                        <span class="detail-label">Throughput</span>
                        <span class="detail-value">${metrics.throughput.toFixed(2)} req/s</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    buildLinksTab(connectedNodes) {
        const incoming = connectedNodes.filter(c => c.direction === 'incoming');
        const outgoing = connectedNodes.filter(c => c.direction === 'outgoing');
        const buildLinkList = (nodes, direction) => {
            if (nodes.length === 0) {
                return `<div class="links-empty">No ${direction} connections</div>`;
            }
            return nodes.map(({ node }) => `
                <div class="link-item" data-node-id="${this.escapeHtml(node.id)}">
                    <span class="link-icon">${this.getIconText(node.type)}</span>
                    <div class="link-info">
                        <span class="link-name">${this.escapeHtml(node.name)}</span>
                        <span class="link-type">${this.escapeHtml(node.type)}</span>
                    </div>
                    <span class="link-status" style="background-color: ${this.getStatusColor(node.status)}"></span>
                </div>
            `).join('');
        };
        return `
            <div class="popover-tab-content hidden" data-content="links">
                <div class="links-section">
                    <div class="links-section-header">
                        <span class="links-direction-icon">←</span>
                        <span class="links-section-title">Incoming (${incoming.length})</span>
                    </div>
                    <div class="links-list">
                        ${buildLinkList(incoming, 'incoming')}
                    </div>
                </div>
                <div class="links-section">
                    <div class="links-section-header">
                        <span class="links-direction-icon">→</span>
                        <span class="links-section-title">Outgoing (${outgoing.length})</span>
                    </div>
                    <div class="links-list">
                        ${buildLinkList(outgoing, 'outgoing')}
                    </div>
                </div>
            </div>
        `;
    }
    getStatusColor(status) {
        switch (status) {
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
    getStatusText(status) {
        switch (status) {
            case NodeStatus.Success:
                return 'Active';
            case NodeStatus.Warning:
                return 'Warning';
            case NodeStatus.Error:
                return 'Error';
            case NodeStatus.Running:
                return 'Running';
            case NodeStatus.None:
            default:
                return 'Inactive';
        }
    }
    getIconText(type) {
        const iconMap = {
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
        return iconMap[type] || '⚡';
    }
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    setupPopoverEvents(popover, config) {
        // Close button
        const closeBtn = popover.querySelector('.popover-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }
        // Tab switching
        const tabs = popover.querySelectorAll('.popover-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const target = e.currentTarget;
                const tabName = target.getAttribute('data-tab');
                // Update active tab button
                tabs.forEach(t => t.classList.remove('active'));
                target.classList.add('active');
                // Show/hide content
                const contents = popover.querySelectorAll('.popover-tab-content');
                contents.forEach(content => {
                    const contentName = content.getAttribute('data-content');
                    if (contentName === tabName) {
                        content.classList.remove('hidden');
                    }
                    else {
                        content.classList.add('hidden');
                    }
                });
                this.activeTab = tabName;
            });
        });
        // Action buttons
        const expandBtn = popover.querySelector('[data-action="expand"]');
        if (expandBtn) {
            expandBtn.addEventListener('click', () => {
                if (this.onExpandCallback && this.node) {
                    this.onExpandCallback(this.node.id);
                }
            });
        }
        const viewDetailsBtn = popover.querySelector('[data-action="viewDetails"]');
        if (viewDetailsBtn) {
            viewDetailsBtn.addEventListener('click', () => {
                if (this.onViewDetailsCallback && this.node) {
                    this.onViewDetailsCallback(this.node.id);
                }
            });
        }
        // Link navigation
        const linkItems = popover.querySelectorAll('.link-item');
        linkItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const target = e.currentTarget;
                const nodeId = target.getAttribute('data-node-id');
                if (nodeId && this.onNavigateCallback) {
                    this.hide();
                    this.onNavigateCallback(nodeId);
                }
            });
        });
    }
    positionPopover(position) {
        if (!this.popoverElement)
            return;
        const containerRect = this.container.getBoundingClientRect();
        const popoverWidth = 320; // Expected popover width
        const popoverHeight = 400; // Expected popover height
        const offset = 15; // Offset from click position
        let x = position.x + offset;
        let y = position.y + offset;
        // Viewport collision detection - right edge
        if (x + popoverWidth > containerRect.width) {
            x = position.x - popoverWidth - offset;
        }
        // Viewport collision detection - bottom edge
        if (y + popoverHeight > containerRect.height) {
            y = containerRect.height - popoverHeight - offset;
        }
        // Ensure popover stays within left and top bounds
        x = Math.max(offset, x);
        y = Math.max(offset, y);
        this.popoverElement.style.left = `${x}px`;
        this.popoverElement.style.top = `${y}px`;
    }
    handleClickOutside(e) {
        if (!this.popoverElement || !this.visible)
            return;
        const target = e.target;
        if (!this.popoverElement.contains(target)) {
            this.hide();
        }
    }
    handleKeyDown(e) {
        if (!this.visible)
            return;
        if (e.key === 'Escape') {
            this.hide();
        }
    }
}
//# sourceMappingURL=NodePopover.js.map