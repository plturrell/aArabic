/**
 * LineChart - Multi-series line chart for training curves
 * Displays loss, accuracy, and other metrics over training steps
 */

import { LineChartConfig, SeriesData, DataPoint, CHART_COLORS, DEFAULT_CHART_CONFIG } from './types';

export class LineChart {
    private container: HTMLElement;
    private svg: SVGSVGElement;
    private config: LineChartConfig;
    
    // Elements
    private chartArea: SVGGElement;
    private xAxisGroup: SVGGElement;
    private yAxisGroup: SVGGElement;
    private gridGroup: SVGGElement;
    private linesGroup: SVGGElement;
    private tooltipDiv: HTMLDivElement | null = null;
    private legendDiv: HTMLDivElement | null = null;
    
    // Scales
    private xScale: (val: number) => number = () => 0;
    private yScale: (val: number) => number = () => 0;
    private xDomain: [number, number] = [0, 100];
    private yDomain: [number, number] = [0, 1];
    
    constructor(container: HTMLElement | string, config: Partial<LineChartConfig>) {
        if (typeof container === 'string') {
            const el = document.querySelector(container);
            if (!el) throw new Error(`Container not found: ${container}`);
            this.container = el as HTMLElement;
        } else {
            this.container = container;
        }
        
        this.config = {
            ...DEFAULT_CHART_CONFIG,
            showGrid: true,
            showLegend: true,
            showTooltip: true,
            xAxisType: 'linear',
            yAxisType: 'linear',
            ...config,
            series: config.series || []
        } as LineChartConfig;
        
        // Create wrapper for relative positioning
        this.container.style.position = 'relative';
        
        this.svg = this.createSVG();
        this.container.appendChild(this.svg);
        
        // Create groups in correct order (back to front)
        this.gridGroup = this.createGroup('grid');
        this.linesGroup = this.createGroup('lines');
        this.xAxisGroup = this.createGroup('x-axis');
        this.yAxisGroup = this.createGroup('y-axis');
        this.chartArea = this.createChartArea();
        
        if (this.config.showTooltip) {
            this.tooltipDiv = this.createTooltip();
        }
        
        if (this.config.showLegend) {
            this.legendDiv = this.createLegend();
        }
        
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
    
    private createChartArea(): SVGGElement {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        const m = this.config.margin!;
        g.setAttribute('transform', `translate(${m.left}, ${m.top})`);
        this.svg.appendChild(g);
        return g;
    }
    
    private createTooltip(): HTMLDivElement {
        const div = document.createElement('div');
        div.style.cssText = `
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            z-index: 1000;
        `;
        this.container.appendChild(div);
        return div;
    }
    
    private createLegend(): HTMLDivElement {
        const div = document.createElement('div');
        div.style.cssText = `
            display: flex;
            gap: 16px;
            justify-content: center;
            margin-top: 8px;
            font-size: 12px;
        `;
        this.container.appendChild(div);
        return div;
    }
    
    private calculateDomains(): void {
        if (this.config.series.length === 0) return;
        
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;
        
        for (const series of this.config.series) {
            for (const point of series.data) {
                const x = typeof point.x === 'number' ? point.x : 0;
                xMin = Math.min(xMin, x);
                xMax = Math.max(xMax, x);
                yMin = Math.min(yMin, point.y);
                yMax = Math.max(yMax, point.y);
            }
        }
        
        // Add padding
        const yPadding = (yMax - yMin) * 0.1 || 0.1;
        this.xDomain = [xMin, xMax];
        this.yDomain = [yMin - yPadding, yMax + yPadding];
    }
    
    private calculateScales(): void {
        const m = this.config.margin!;
        const width = this.config.width! - m.left - m.right;
        const height = this.config.height! - m.top - m.bottom;
        
        this.xScale = (val: number) => {
            const range = this.xDomain[1] - this.xDomain[0] || 1;
            return ((val - this.xDomain[0]) / range) * width;
        };
        
        this.yScale = (val: number) => {
            const range = this.yDomain[1] - this.yDomain[0] || 1;
            return height - ((val - this.yDomain[0]) / range) * height;
        };
    }

    private renderGrid(): void {
        this.gridGroup.innerHTML = '';
        if (!this.config.showGrid) return;

        const m = this.config.margin!;
        const width = this.config.width! - m.left - m.right;
        const height = this.config.height! - m.top - m.bottom;

        // Horizontal grid lines
        for (let i = 0; i <= 5; i++) {
            const y = m.top + (height / 5) * i;
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', String(m.left));
            line.setAttribute('y1', String(y));
            line.setAttribute('x2', String(m.left + width));
            line.setAttribute('y2', String(y));
            line.setAttribute('stroke', '#e0e0e0');
            line.setAttribute('stroke-dasharray', '3,3');
            this.gridGroup.appendChild(line);
        }

        // Vertical grid lines
        for (let i = 0; i <= 5; i++) {
            const x = m.left + (width / 5) * i;
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', String(x));
            line.setAttribute('y1', String(m.top));
            line.setAttribute('x2', String(x));
            line.setAttribute('y2', String(m.top + height));
            line.setAttribute('stroke', '#e0e0e0');
            line.setAttribute('stroke-dasharray', '3,3');
            this.gridGroup.appendChild(line);
        }
    }

    private renderAxes(): void {
        this.xAxisGroup.innerHTML = '';
        this.yAxisGroup.innerHTML = '';

        const m = this.config.margin!;
        const width = this.config.width! - m.left - m.right;
        const height = this.config.height! - m.top - m.bottom;

        // X-axis
        const xAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        xAxisLine.setAttribute('x1', String(m.left));
        xAxisLine.setAttribute('y1', String(m.top + height));
        xAxisLine.setAttribute('x2', String(m.left + width));
        xAxisLine.setAttribute('y2', String(m.top + height));
        xAxisLine.setAttribute('stroke', '#333');
        this.xAxisGroup.appendChild(xAxisLine);

        // X-axis ticks and labels
        for (let i = 0; i <= 5; i++) {
            const val = this.xDomain[0] + ((this.xDomain[1] - this.xDomain[0]) / 5) * i;
            const x = m.left + (width / 5) * i;

            const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            tick.setAttribute('x1', String(x));
            tick.setAttribute('y1', String(m.top + height));
            tick.setAttribute('x2', String(x));
            tick.setAttribute('y2', String(m.top + height + 5));
            tick.setAttribute('stroke', '#333');
            this.xAxisGroup.appendChild(tick);

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', String(x));
            label.setAttribute('y', String(m.top + height + 18));
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '10');
            label.setAttribute('fill', '#666');
            label.textContent = String(Math.round(val));
            this.xAxisGroup.appendChild(label);
        }

        // Y-axis
        const yAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        yAxisLine.setAttribute('x1', String(m.left));
        yAxisLine.setAttribute('y1', String(m.top));
        yAxisLine.setAttribute('x2', String(m.left));
        yAxisLine.setAttribute('y2', String(m.top + height));
        yAxisLine.setAttribute('stroke', '#333');
        this.yAxisGroup.appendChild(yAxisLine);

        // Y-axis ticks and labels
        for (let i = 0; i <= 5; i++) {
            const val = this.yDomain[0] + ((this.yDomain[1] - this.yDomain[0]) / 5) * i;
            const y = m.top + height - (height / 5) * i;

            const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            tick.setAttribute('x1', String(m.left - 5));
            tick.setAttribute('y1', String(y));
            tick.setAttribute('x2', String(m.left));
            tick.setAttribute('y2', String(y));
            tick.setAttribute('stroke', '#333');
            this.yAxisGroup.appendChild(tick);

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', String(m.left - 8));
            label.setAttribute('y', String(y + 3));
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('font-size', '10');
            label.setAttribute('fill', '#666');
            label.textContent = val.toFixed(2);
            this.yAxisGroup.appendChild(label);
        }

        // Axis labels
        if (this.config.xAxisLabel) {
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', String(m.left + width / 2));
            label.setAttribute('y', String(this.config.height! - 5));
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '12');
            label.setAttribute('fill', '#333');
            label.textContent = this.config.xAxisLabel;
            this.xAxisGroup.appendChild(label);
        }

        if (this.config.yAxisLabel) {
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('transform', `translate(12, ${m.top + height / 2}) rotate(-90)`);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '12');
            label.setAttribute('fill', '#333');
            label.textContent = this.config.yAxisLabel;
            this.yAxisGroup.appendChild(label);
        }
    }

    private renderLines(): void {
        this.linesGroup.innerHTML = '';
        const m = this.config.margin!;

        this.config.series.forEach((series, idx) => {
            if (series.data.length === 0) return;

            const color = series.color || CHART_COLORS.series[idx % CHART_COLORS.series.length];
            const lineWidth = series.lineWidth || 2;

            // Create path
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            const d = series.data.map((point, i) => {
                const x = m.left + this.xScale(typeof point.x === 'number' ? point.x : i);
                const y = m.top + this.yScale(point.y);
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
            }).join(' ');

            path.setAttribute('d', d);
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', color);
            path.setAttribute('stroke-width', String(lineWidth));
            if (series.dashed) {
                path.setAttribute('stroke-dasharray', '5,5');
            }
            this.linesGroup.appendChild(path);

            // Add points if enabled
            if (series.showPoints !== false) {
                series.data.forEach((point, i) => {
                    const x = m.left + this.xScale(typeof point.x === 'number' ? point.x : i);
                    const y = m.top + this.yScale(point.y);

                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.setAttribute('cx', String(x));
                    circle.setAttribute('cy', String(y));
                    circle.setAttribute('r', '4');
                    circle.setAttribute('fill', color);
                    circle.setAttribute('stroke', 'white');
                    circle.setAttribute('stroke-width', '2');
                    circle.style.cursor = 'pointer';

                    // Tooltip interaction
                    if (this.tooltipDiv) {
                        circle.addEventListener('mouseenter', (e) => {
                            this.tooltipDiv!.innerHTML = `
                                <strong>${series.name}</strong><br>
                                X: ${point.x}<br>
                                Y: ${point.y.toFixed(4)}
                            `;
                            this.tooltipDiv!.style.opacity = '1';
                            this.tooltipDiv!.style.left = `${(e as MouseEvent).offsetX + 10}px`;
                            this.tooltipDiv!.style.top = `${(e as MouseEvent).offsetY - 30}px`;
                        });
                        circle.addEventListener('mouseleave', () => {
                            this.tooltipDiv!.style.opacity = '0';
                        });
                    }

                    this.linesGroup.appendChild(circle);
                });
            }
        });
    }

    private renderLegend(): void {
        if (!this.legendDiv) return;
        this.legendDiv.innerHTML = '';

        this.config.series.forEach((series, idx) => {
            const color = series.color || CHART_COLORS.series[idx % CHART_COLORS.series.length];

            const item = document.createElement('div');
            item.style.cssText = 'display: flex; align-items: center; gap: 4px;';

            const swatch = document.createElement('div');
            swatch.style.cssText = `width: 12px; height: 12px; background: ${color}; border-radius: 2px;`;

            const label = document.createElement('span');
            label.textContent = series.name;

            item.appendChild(swatch);
            item.appendChild(label);
            this.legendDiv.appendChild(item);
        });
    }

    private render(): void {
        this.calculateDomains();
        this.calculateScales();
        this.renderGrid();
        this.renderAxes();
        this.renderLines();
        this.renderLegend();
    }

    private setupResizeObserver(): void {
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this.config.width = width;
                    this.config.height = height - (this.legendDiv ? 30 : 0);
                    this.svg.setAttribute('width', String(this.config.width));
                    this.svg.setAttribute('height', String(this.config.height));
                    this.render();
                }
            }
        });
        observer.observe(this.container);
    }

    // Public API
    public setData(series: SeriesData[]): void {
        this.config.series = series;
        this.render();
    }

    public addPoint(seriesIndex: number, point: DataPoint): void {
        if (this.config.series[seriesIndex]) {
            this.config.series[seriesIndex].data.push(point);
            this.render();
        }
    }

    public destroy(): void {
        this.container.removeChild(this.svg);
        if (this.tooltipDiv) this.container.removeChild(this.tooltipDiv);
        if (this.legendDiv) this.container.removeChild(this.legendDiv);
    }
}
