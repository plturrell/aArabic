/**
 * GaugeChart - Semi-circular gauge with needle
 * Used for KL Divergence, temperature, and bounded metrics
 */

import { GaugeChartConfig, DEFAULT_GAUGE_CONFIG, CHART_COLORS, DEFAULT_CHART_CONFIG } from './types';

export class GaugeChart {
    private container: HTMLElement;
    private svg: SVGSVGElement;
    private config: GaugeChartConfig;
    
    // Elements
    private zones: SVGPathElement[] = [];
    private needle: SVGGElement;
    private valueText: SVGTextElement;
    private labelText: SVGTextElement;
    
    constructor(container: HTMLElement | string, config: Partial<GaugeChartConfig>) {
        if (typeof container === 'string') {
            const el = document.querySelector(container);
            if (!el) throw new Error(`Container not found: ${container}`);
            this.container = el as HTMLElement;
        } else {
            this.container = container;
        }
        
        this.config = {
            ...DEFAULT_CHART_CONFIG,
            ...DEFAULT_GAUGE_CONFIG,
            ...config,
            value: config.value ?? 0
        } as GaugeChartConfig;
        
        this.svg = this.createSVG();
        this.container.appendChild(this.svg);
        
        this.createZones();
        this.needle = this.createNeedle();
        this.valueText = this.createValueText();
        this.labelText = this.createLabelText();
        
        this.render();
        
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
    
    private createZones(): void {
        const zonesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        
        for (const zone of this.config.zones || []) {
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('fill', zone.color);
            path.setAttribute('opacity', '0.8');
            this.zones.push(path);
            zonesGroup.appendChild(path);
        }
        
        this.svg.appendChild(zonesGroup);
    }
    
    private createNeedle(): SVGGElement {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        
        // Needle body
        const needle = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        needle.setAttribute('fill', this.config.needleColor || '#333');
        group.appendChild(needle);
        
        // Center circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('fill', this.config.needleColor || '#333');
        circle.setAttribute('r', '8');
        group.appendChild(circle);
        
        this.svg.appendChild(group);
        return group;
    }
    
    private createValueText(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '24');
        text.setAttribute('font-weight', 'bold');
        text.setAttribute('fill', '#333');
        this.svg.appendChild(text);
        return text;
    }
    
    private createLabelText(): SVGTextElement {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '12');
        text.setAttribute('fill', '#666');
        this.svg.appendChild(text);
        return text;
    }
    
    private describeArc(cx: number, cy: number, innerR: number, outerR: number, startAngle: number, endAngle: number): string {
        const startOuter = this.polarToCartesian(cx, cy, outerR, endAngle);
        const endOuter = this.polarToCartesian(cx, cy, outerR, startAngle);
        const startInner = this.polarToCartesian(cx, cy, innerR, startAngle);
        const endInner = this.polarToCartesian(cx, cy, innerR, endAngle);
        const largeArc = endAngle - startAngle <= 180 ? 0 : 1;
        
        return [
            'M', startOuter.x, startOuter.y,
            'A', outerR, outerR, 0, largeArc, 0, endOuter.x, endOuter.y,
            'L', startInner.x, startInner.y,
            'A', innerR, innerR, 0, largeArc, 1, endInner.x, endInner.y,
            'Z'
        ].join(' ');
    }
    
    private polarToCartesian(cx: number, cy: number, radius: number, angleInDegrees: number) {
        const angleInRadians = (angleInDegrees - 90) * Math.PI / 180;
        return {
            x: cx + radius * Math.cos(angleInRadians),
            y: cy + radius * Math.sin(angleInRadians)
        };
    }
    
    private valueToAngle(value: number): number {
        const range = this.config.max! - this.config.min!;
        const normalized = (value - this.config.min!) / range;
        // Gauge spans from -135° to 135° (270° total)
        return -135 + normalized * 270;
    }

    private render(): void {
        const width = this.config.width!;
        const height = this.config.height!;
        const cx = width / 2;
        const cy = height * 0.65;
        const outerR = Math.min(width, height) / 2 - 20;
        const innerR = outerR - 30;

        // Render zones
        const zones = this.config.zones || [];
        zones.forEach((zone, i) => {
            const startAngle = this.valueToAngle(zone.min);
            const endAngle = this.valueToAngle(zone.max);
            this.zones[i].setAttribute('d', this.describeArc(cx, cy, innerR, outerR, startAngle, endAngle));
        });

        // Render needle
        if (this.config.showNeedle) {
            const angle = this.valueToAngle(this.config.value);
            const needleLength = innerR - 10;

            const tip = this.polarToCartesian(cx, cy, needleLength, angle);
            const baseLeft = this.polarToCartesian(cx, cy, 10, angle - 90);
            const baseRight = this.polarToCartesian(cx, cy, 10, angle + 90);

            const needle = this.needle.querySelector('polygon')!;
            needle.setAttribute('points', `${tip.x},${tip.y} ${baseLeft.x},${baseLeft.y} ${baseRight.x},${baseRight.y}`);

            const circle = this.needle.querySelector('circle')!;
            circle.setAttribute('cx', String(cx));
            circle.setAttribute('cy', String(cy));

            // Animate
            if (this.config.animate) {
                this.needle.style.transition = `transform ${this.config.animationDuration}ms ease-out`;
            }
        }

        // Value text
        const displayValue = this.config.unit
            ? `${this.config.value.toFixed(1)}${this.config.unit}`
            : this.config.value.toFixed(1);
        this.valueText.textContent = displayValue;
        this.valueText.setAttribute('x', String(cx));
        this.valueText.setAttribute('y', String(cy + 40));

        // Label text
        if (this.config.label) {
            this.labelText.textContent = this.config.label;
            this.labelText.setAttribute('x', String(cx));
            this.labelText.setAttribute('y', String(cy + 60));
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
        this.config.value = Math.max(this.config.min!, Math.min(this.config.max!, value));
        this.render();
    }

    public getValue(): number {
        return this.config.value;
    }

    public destroy(): void {
        this.container.removeChild(this.svg);
    }
}

