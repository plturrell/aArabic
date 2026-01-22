/**
 * BarChart - Vertical/Horizontal bar chart
 * For comparing metrics across algorithms, models, experiments
 */

import { BarChartConfig, BarData, CHART_COLORS, DEFAULT_CHART_CONFIG } from './types';

export class BarChart {
    private container: HTMLElement;
    private svg: SVGSVGElement;
    private config: BarChartConfig;
    
    // Groups
    private barsGroup: SVGGElement;
    private xAxisGroup: SVGGElement;
    private yAxisGroup: SVGGElement;
    private tooltipDiv: HTMLDivElement | null = null;
    
    // Scales
    private maxValue: number = 0;
    
    constructor(container: HTMLElement | string, config: Partial<BarChartConfig>) {
        if (typeof container === 'string') {
            const el = document.querySelector(container);
            if (!el) throw new Error(`Container not found: ${container}`);
            this.container = el as HTMLElement;
        } else {
            this.container = container;
        }
        
        this.config = {
            ...DEFAULT_CHART_CONFIG,
            orientation: 'vertical',
            showValues: true,
            grouped: false,
            stacked: false,
            ...config,
            data: config.data || []
        } as BarChartConfig;
        
        this.container.style.position = 'relative';
        
        this.svg = this.createSVG();
        this.container.appendChild(this.svg);
        
        this.barsGroup = this.createGroup('bars');
        this.xAxisGroup = this.createGroup('x-axis');
        this.yAxisGroup = this.createGroup('y-axis');
        
        this.tooltipDiv = this.createTooltip();
        
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
    
    private calculateMaxValue(): void {
        this.maxValue = 0;
        for (const item of this.config.data) {
            if (this.config.stacked) {
                const sum = item.values.reduce((s, v) => s + v.value, 0);
                this.maxValue = Math.max(this.maxValue, sum);
            } else {
                for (const val of item.values) {
                    this.maxValue = Math.max(this.maxValue, val.value);
                }
            }
        }
        this.maxValue *= 1.1; // Add 10% padding
    }
    
    private renderVerticalBars(): void {
        const m = this.config.margin!;
        const width = this.config.width! - m.left - m.right;
        const height = this.config.height! - m.top - m.bottom;
        
        const categories = this.config.data;
        const categoryWidth = width / categories.length;
        const barPadding = categoryWidth * 0.2;
        const numSeries = categories[0]?.values.length || 1;
        const barWidth = this.config.grouped 
            ? (categoryWidth - barPadding * 2) / numSeries 
            : categoryWidth - barPadding * 2;
        
        categories.forEach((category, catIdx) => {
            const baseX = m.left + catIdx * categoryWidth + barPadding;
            
            if (this.config.stacked) {
                let stackY = 0;
                category.values.forEach((val, valIdx) => {
                    const barHeight = (val.value / this.maxValue) * height;
                    const color = val.color || CHART_COLORS.series[valIdx % CHART_COLORS.series.length];
                    
                    const rect = this.createBar(
                        baseX,
                        m.top + height - stackY - barHeight,
                        barWidth,
                        barHeight,
                        color,
                        val.name,
                        val.value
                    );
                    this.barsGroup.appendChild(rect);
                    
                    stackY += barHeight;
                });
            } else if (this.config.grouped) {
                category.values.forEach((val, valIdx) => {
                    const barHeight = (val.value / this.maxValue) * height;
                    const color = val.color || CHART_COLORS.series[valIdx % CHART_COLORS.series.length];
                    const x = baseX + valIdx * barWidth;
                    
                    const rect = this.createBar(
                        x,
                        m.top + height - barHeight,
                        barWidth - 2,
                        barHeight,
                        color,
                        val.name,
                        val.value
                    );
                    this.barsGroup.appendChild(rect);
                });
            } else {
                // Single bar per category
                const val = category.values[0];
                if (val) {
                    const barHeight = (val.value / this.maxValue) * height;
                    const color = val.color || CHART_COLORS.series[catIdx % CHART_COLORS.series.length];

                    const rect = this.createBar(
                        baseX,
                        m.top + height - barHeight,
                        barWidth,
                        barHeight,
                        color,
                        category.category,
                        val.value
                    );
                    this.barsGroup.appendChild(rect);
                }
            }

            // Category label
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', String(baseX + barWidth / 2 * (this.config.grouped ? numSeries : 1)));
            label.setAttribute('y', String(m.top + height + 18));
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '11');
            label.setAttribute('fill', '#666');
            label.textContent = category.category;
            this.xAxisGroup.appendChild(label);
        });
    }

    private createBar(x: number, y: number, width: number, height: number, color: string, name: string, value: number): SVGRectElement {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', String(x));
        rect.setAttribute('y', String(y));
        rect.setAttribute('width', String(Math.max(0, width)));
        rect.setAttribute('height', String(Math.max(0, height)));
        rect.setAttribute('fill', color);
        rect.setAttribute('rx', '2');
        rect.style.cursor = 'pointer';
        rect.style.transition = 'opacity 0.15s';

        // Value label
        if (this.config.showValues && height > 20) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', String(x + width / 2));
            text.setAttribute('y', String(y + 15));
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '10');
            text.setAttribute('fill', 'white');
            text.setAttribute('font-weight', 'bold');
            text.textContent = value.toFixed(1);
            this.barsGroup.appendChild(text);
        }

        // Hover effects
        rect.addEventListener('mouseenter', (e) => {
            rect.setAttribute('opacity', '0.8');
            if (this.tooltipDiv) {
                this.tooltipDiv.innerHTML = `<strong>${name}</strong><br>Value: ${value.toFixed(2)}`;
                this.tooltipDiv.style.opacity = '1';
                this.tooltipDiv.style.left = `${(e as MouseEvent).offsetX + 10}px`;
                this.tooltipDiv.style.top = `${(e as MouseEvent).offsetY - 30}px`;
            }
        });
        rect.addEventListener('mouseleave', () => {
            rect.setAttribute('opacity', '1');
            if (this.tooltipDiv) {
                this.tooltipDiv.style.opacity = '0';
            }
        });

        return rect;
    }

    private renderAxes(): void {
        const m = this.config.margin!;
        const height = this.config.height! - m.top - m.bottom;

        // Y-axis line
        const yAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        yAxisLine.setAttribute('x1', String(m.left));
        yAxisLine.setAttribute('y1', String(m.top));
        yAxisLine.setAttribute('x2', String(m.left));
        yAxisLine.setAttribute('y2', String(m.top + height));
        yAxisLine.setAttribute('stroke', '#333');
        this.yAxisGroup.appendChild(yAxisLine);

        // Y-axis ticks
        for (let i = 0; i <= 5; i++) {
            const val = (this.maxValue / 5) * i;
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
            label.setAttribute('y', String(y + 4));
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('font-size', '10');
            label.setAttribute('fill', '#666');
            label.textContent = val.toFixed(1);
            this.yAxisGroup.appendChild(label);
        }

        // Axis labels
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

    private render(): void {
        this.barsGroup.innerHTML = '';
        this.xAxisGroup.innerHTML = '';
        this.yAxisGroup.innerHTML = '';

        this.calculateMaxValue();

        if (this.config.orientation === 'vertical') {
            this.renderVerticalBars();
        }
        // TODO: Add horizontal bar rendering

        this.renderAxes();
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
                    this.render();
                }
            }
        });
        observer.observe(this.container);
    }

    // Public API
    public setData(data: BarData[]): void {
        this.config.data = data;
        this.render();
    }

    public destroy(): void {
        this.container.removeChild(this.svg);
        if (this.tooltipDiv) this.container.removeChild(this.tooltipDiv);
    }
}

