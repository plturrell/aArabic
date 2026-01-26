/**
 * SankeyDiagram - Flow visualization for data pipelines
 * Shows flow of data through training stages, model routing, etc.
 */

import { SankeyConfig, SankeyNode, SankeyLink, CHART_COLORS, DEFAULT_CHART_CONFIG } from './types';

interface ComputedNode extends SankeyNode {
    x: number;
    y: number;
    height: number;
    value: number;
    sourceLinks: ComputedLink[];
    targetLinks: ComputedLink[];
}

interface ComputedLink extends SankeyLink {
    sourceNode: ComputedNode;
    targetNode: ComputedNode;
    width: number;
    sy: number;  // Source y offset
    ty: number;  // Target y offset
}

export class SankeyDiagram {
    private container: HTMLElement;
    private svg: SVGSVGElement;
    private config: SankeyConfig;
    
    // Computed data
    private computedNodes: Map<string, ComputedNode> = new Map();
    private computedLinks: ComputedLink[] = [];
    
    // Groups
    private linksGroup: SVGGElement;
    private nodesGroup: SVGGElement;
    private labelsGroup: SVGGElement;
    
    constructor(container: HTMLElement | string, config: Partial<SankeyConfig>) {
        if (typeof container === 'string') {
            const el = document.querySelector(container);
            if (!el) throw new Error(`Container not found: ${container}`);
            this.container = el as HTMLElement;
        } else {
            this.container = container;
        }
        
        this.config = {
            ...DEFAULT_CHART_CONFIG,
            nodeWidth: 20,
            nodePadding: 10,
            showLabels: true,
            showValues: true,
            ...config,
            nodes: config.nodes || [],
            links: config.links || []
        } as SankeyConfig;
        
        this.svg = this.createSVG();
        this.container.appendChild(this.svg);
        
        this.linksGroup = this.createGroup('links');
        this.nodesGroup = this.createGroup('nodes');
        this.labelsGroup = this.createGroup('labels');
        
        this.computeLayout();
        this.render();
        
        if (this.config.responsive) {
            this.setupResizeObserver();
        }
    }
    
    private createSVG(): SVGSVGElement {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', String(this.config.width));
        svg.setAttribute('height', String(this.config.height));
        svg.style.overflow = 'visible';
        return svg;
    }
    
    private createGroup(className: string): SVGGElement {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', className);
        this.svg.appendChild(g);
        return g;
    }
    
    private computeLayout(): void {
        const m = this.config.margin!;
        const width = this.config.width! - m.left - m.right;
        const height = this.config.height! - m.top - m.bottom;
        
        // Initialize nodes
        this.computedNodes.clear();
        for (const node of this.config.nodes) {
            this.computedNodes.set(node.id, {
                ...node,
                x: 0,
                y: 0,
                height: 0,
                value: 0,
                sourceLinks: [],
                targetLinks: []
            });
        }
        
        // Calculate node values from links
        for (const link of this.config.links) {
            const source = this.computedNodes.get(link.source);
            const target = this.computedNodes.get(link.target);
            if (source && target) {
                source.value += link.value;
                target.value += link.value;
            }
        }
        
        // Assign nodes to columns (layers)
        const columns: ComputedNode[][] = [];
        const visited = new Set<string>();
        
        // Find source nodes (no incoming links)
        const sourceNodes: ComputedNode[] = [];
        for (const [id, node] of this.computedNodes) {
            const hasIncoming = this.config.links.some(l => l.target === id);
            if (!hasIncoming) {
                sourceNodes.push(node);
            }
        }
        
        // BFS to assign columns
        let currentColumn = sourceNodes;
        while (currentColumn.length > 0) {
            columns.push([...currentColumn]);
            currentColumn.forEach(n => visited.add(n.id));
            
            const nextColumn: ComputedNode[] = [];
            for (const node of currentColumn) {
                for (const link of this.config.links) {
                    if (link.source === node.id) {
                        const target = this.computedNodes.get(link.target);
                        if (target && !visited.has(target.id) && !nextColumn.includes(target)) {
                            nextColumn.push(target);
                        }
                    }
                }
            }
            currentColumn = nextColumn;
        }
        
        // Position nodes
        const numColumns = columns.length;
        const columnWidth = numColumns > 1 ? width / (numColumns - 1) : width;
        
        columns.forEach((column, colIdx) => {
            const x = m.left + colIdx * columnWidth;
            
            // Calculate total height needed
            const totalValue = column.reduce((sum, n) => sum + n.value, 0);
            const availableHeight = height - (column.length - 1) * this.config.nodePadding!;

            let y = m.top;
            for (const node of column) {
                node.x = x;
                node.y = y;
                node.height = totalValue > 0 ? (node.value / totalValue) * availableHeight : 20;
                y += node.height + this.config.nodePadding!;
            }
        });

        // Compute links
        this.computedLinks = [];
        for (const link of this.config.links) {
            const sourceNode = this.computedNodes.get(link.source);
            const targetNode = this.computedNodes.get(link.target);

            if (sourceNode && targetNode) {
                const computedLink: ComputedLink = {
                    ...link,
                    sourceNode,
                    targetNode,
                    width: 0,
                    sy: 0,
                    ty: 0
                };

                sourceNode.sourceLinks.push(computedLink);
                targetNode.targetLinks.push(computedLink);
                this.computedLinks.push(computedLink);
            }
        }

        // Calculate link positions
        for (const node of this.computedNodes.values()) {
            let sy = 0;
            for (const link of node.sourceLinks) {
                link.width = node.height * (link.value / node.value);
                link.sy = node.y + sy;
                sy += link.width;
            }

            let ty = 0;
            for (const link of node.targetLinks) {
                link.ty = node.y + ty;
                ty += link.width || (node.height * (link.value / node.value));
            }
        }
    }

    private renderLinks(): void {
        this.linksGroup.innerHTML = '';

        for (const link of this.computedLinks) {
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

            const x0 = link.sourceNode.x + this.config.nodeWidth!;
            const x1 = link.targetNode.x;
            const y0 = link.sy + link.width / 2;
            const y1 = link.ty + link.width / 2;

            const curvature = 0.5;
            const xi = (x0 + x1) * curvature;

            const d = `
                M ${x0} ${y0}
                C ${xi} ${y0}, ${xi} ${y1}, ${x1} ${y1}
            `;

            const color = link.color || CHART_COLORS.primary;

            path.setAttribute('d', d);
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', color);
            path.setAttribute('stroke-width', String(Math.max(1, link.width)));
            path.setAttribute('opacity', '0.5');

            // Hover effect
            path.addEventListener('mouseenter', () => {
                path.setAttribute('opacity', '0.8');
            });
            path.addEventListener('mouseleave', () => {
                path.setAttribute('opacity', '0.5');
            });

            this.linksGroup.appendChild(path);
        }
    }

    private renderNodes(): void {
        this.nodesGroup.innerHTML = '';

        let colorIdx = 0;
        for (const node of this.computedNodes.values()) {
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            const color = node.color || CHART_COLORS.series[colorIdx % CHART_COLORS.series.length];

            rect.setAttribute('x', String(node.x));
            rect.setAttribute('y', String(node.y));
            rect.setAttribute('width', String(this.config.nodeWidth));
            rect.setAttribute('height', String(Math.max(1, node.height)));
            rect.setAttribute('fill', color);
            rect.setAttribute('rx', '2');
            rect.style.cursor = 'pointer';

            // Hover effect
            rect.addEventListener('mouseenter', () => {
                rect.setAttribute('opacity', '0.8');
            });
            rect.addEventListener('mouseleave', () => {
                rect.setAttribute('opacity', '1');
            });

            this.nodesGroup.appendChild(rect);
            colorIdx++;
        }
    }

    private renderLabels(): void {
        this.labelsGroup.innerHTML = '';
        if (!this.config.showLabels) return;

        for (const node of this.computedNodes.values()) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');

            // Position label to the right of node if in left half, else to left
            const midX = this.config.width! / 2;
            const isLeft = node.x < midX;

            text.setAttribute('x', String(isLeft ? node.x + this.config.nodeWidth! + 6 : node.x - 6));
            text.setAttribute('y', String(node.y + node.height / 2));
            text.setAttribute('text-anchor', isLeft ? 'start' : 'end');
            text.setAttribute('dominant-baseline', 'middle');
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', '#333');

            let labelText = node.name;
            if (this.config.showValues) {
                labelText += ` (${node.value})`;
            }
            text.textContent = labelText;

            this.labelsGroup.appendChild(text);
        }
    }

    private render(): void {
        this.renderLinks();
        this.renderNodes();
        this.renderLabels();
    }

    private setupResizeObserver(): void {
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this.config.width = width;
                    this.config.height = height;
                    this.svg.setAttribute('width', String(width));
                    this.svg.setAttribute('height', String(height));
                    this.computeLayout();
                    this.render();
                }
            }
        });
        observer.observe(this.container);
    }

    // Public API
    public setData(nodes: SankeyNode[], links: SankeyLink[]): void {
        this.config.nodes = nodes;
        this.config.links = links;
        this.computeLayout();
        this.render();
    }

    public destroy(): void {
        this.container.removeChild(this.svg);
    }
}

