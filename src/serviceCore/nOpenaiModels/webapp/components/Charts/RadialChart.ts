/**
 * RadialChart - Circular progress chart
 * Used for displaying percentages like Win Rate, Accuracy, etc.
 */

import { RadialChartConfig, DEFAULT_RADIAL_CONFIG, CHART_COLORS, DEFAULT_CHART_CONFIG } from './types';

export class RadialChart {
    private container: HTMLElement;
    private svg: SVGSVGElement;
    private config: RadialChartConfig;
    
    // Elements
    private backgroundArc: SVGPathElement;
    private valueArc: SVGPathElement;
    private valueText: SVGTextElement;
    private labelText: SVGTextElement;
    
    constructor(container: HTMLElement | string, config: Partial<RadialChartConfig>) {
        // Get container
        if (typeof container === 'string') {
            const el = document.querySelector(container);
            if (!el) throw new Error(`Container not found: ${container}`);
            this.container = el as HTMLElement;
        } else {
            this.container = container;
        }
        
        // Merge config with defaults
        this.config = {
            ...DEFAULT_CHART_CONFIG,
            ...DEFAULT_RADIAL_CONFIG,
            ...config,
            value: config.value ?? 0
        } as RadialChartConfig;
        
        // Create SVG
        this.svg = this.createSVG();
        this.container.appendChild(this.svg);
        
        // Create elements
        this.backgroundArc = this.createArc('background');
        this.valueArc = this.createArc('value');
        this.valueText = this.createValueText();
        this.labelText = this.createLabelText();
        
        // Initial render
        this.render();
        
        // Setup resize observer if responsive
        if (this.config.responsive) {
            this.setupResizeObserver();
        }
    }
    
    private createSVG(): SVGSVGElement {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', String(this.config.width));
        svg.setAttribute('height', String(this.config.height));
        svg.setAttribute('viewBox', `0 0 ${this.config.width} ${this.config.height}`);
        svg.style.overflow = 'visible';
        return svg;
    }
    
    private createArc(type: 'background' | 'value'): SVGPathElement {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke-linecap', 'round');
        
        if (type === 'background') {
            path.setAttribute('stroke', '#e0e0e0');
            path.setAttribute('stroke-width', String(this.config.arcWidth));
        } else {
            path.setAttribute('stroke-width', String(this.config.arcWidth! + 2));
        }
        
        this.svg.appendChild(path);
        return path;
    }
    
    private createValueText(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'middle');
        text.setAttribute('font-size', '28');
        text.setAttribute('font-weight', 'bold');
        text.setAttribute('fill', '#333');
        this.svg.appendChild(text);
        return text;
    }
    
    private createLabelText(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '14');
        text.setAttribute('fill', '#666');
        this.svg.appendChild(text);
        return text;
    }
    
    private getColorForValue(value: number): string {
        const thresholds = this.config.thresholds || DEFAULT_RADIAL_CONFIG.thresholds!;
        for (const threshold of thresholds) {
            if (value <= threshold.value) {
                return threshold.color;
            }
        }
        return CHART_COLORS.primary;
    }
    
    private describeArc(cx: number, cy: number, radius: number, startAngle: number, endAngle: number): string {
        const start = this.polarToCartesian(cx, cy, radius, endAngle);
        const end = this.polarToCartesian(cx, cy, radius, startAngle);
        const largeArcFlag = endAngle - startAngle <= 180 ? 0 : 1;
        
        return [
            'M', start.x, start.y,
            'A', radius, radius, 0, largeArcFlag, 0, end.x, end.y
        ].join(' ');
    }
    
    private polarToCartesian(cx: number, cy: number, radius: number, angleInDegrees: number) {
        const angleInRadians = (angleInDegrees - 90) * Math.PI / 180;
        return {
            x: cx + radius * Math.cos(angleInRadians),
            y: cy + radius * Math.sin(angleInRadians)
        };
    }
    
    private render(): void {
        const width = this.config.width!;
        const height = this.config.height!;
        const cx = width / 2;
        const cy = height / 2;
        const radius = Math.min(width, height) / 2 - this.config.arcWidth! - 10;
        
        // Background arc (full circle)
        this.backgroundArc.setAttribute('d', this.describeArc(cx, cy, radius, 0, 359.99));
        
        // Value arc
        const percentage = (this.config.value - this.config.minValue!) / (this.config.maxValue! - this.config.minValue!);
        const endAngle = Math.max(0.01, percentage * 360);
        this.valueArc.setAttribute('d', this.describeArc(cx, cy, radius, 0, endAngle));
        this.valueArc.setAttribute('stroke', this.getColorForValue(this.config.value));
        
        // Animate if enabled
        if (this.config.animate) {
            this.valueArc.style.transition = `stroke-dashoffset ${this.config.animationDuration}ms ease-out`;
        }
        
        // Value text
        if (this.config.showValue) {
            const displayValue = this.config.unit 
                ? `${Math.round(this.config.value)}${this.config.unit}`
                : String(Math.round(this.config.value));
            this.valueText.textContent = displayValue;
            this.valueText.setAttribute('x', String(cx));
            this.valueText.setAttribute('y', String(cy));
        }
        
        // Label text
        if (this.config.showLabel && this.config.label) {
            this.labelText.textContent = this.config.label;
            this.labelText.setAttribute('x', String(cx));
            this.labelText.setAttribute('y', String(cy + 25));
        }
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
                    this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
                    this.render();
                }
            }
        });
        observer.observe(this.container);
    }

    // Public API
    public setValue(value: number): void {
        this.config.value = Math.max(this.config.minValue!, Math.min(this.config.maxValue!, value));
        this.render();
    }

    public getValue(): number {
        return this.config.value;
    }

    public setLabel(label: string): void {
        this.config.label = label;
        this.render();
    }

    public destroy(): void {
        this.container.removeChild(this.svg);
    }
}
