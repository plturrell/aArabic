/**
 * PerformanceMonitor - FPS and performance metrics tracking
 * Real-time monitoring for optimization
 */

export class PerformanceMonitor {
    private enabled: boolean = false;
    private displayElement: HTMLDivElement | null = null;
    
    // FPS tracking
    private frameCount: number = 0;
    private lastTime: number = performance.now();
    private fps: number = 60;
    private fpsHistory: number[] = [];
    private maxHistorySize: number = 60;  // 1 second at 60fps
    
    // Performance metrics
    private renderTime: number = 0;
    private layoutTime: number = 0;
    private updateTime: number = 0;
    
    // Memory (if available)
    private memoryUsage: number = 0;
    
    // Thresholds
    private readonly FPS_WARNING = 30;
    private readonly FPS_CRITICAL = 15;
    
    constructor() {
        this.createDisplay();
    }
    
    // ========================================================================
    // Display
    // ========================================================================
    
    private createDisplay(): void {
        this.displayElement = document.createElement('div');
        this.displayElement.className = 'performance-monitor';
        this.displayElement.style.cssText = `
            position: fixed;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: #00ff00;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            padding: 10px;
            border-radius: 4px;
            z-index: 10000;
            min-width: 200px;
            display: none;
        `;
        document.body.appendChild(this.displayElement);
    }
    
    private updateDisplay(): void {
        if (!this.displayElement || !this.enabled) return;
        
        const avgFps = this.getAverageFPS();
        const fpsColor = this.getFPSColor(avgFps);
        
        const html = `
            <div style="color: ${fpsColor}; font-weight: bold; font-size: 14px;">
                FPS: ${avgFps.toFixed(1)}
            </div>
            <div style="margin-top: 5px;">
                Render: ${this.renderTime.toFixed(2)}ms
            </div>
            <div>
                Layout: ${this.layoutTime.toFixed(2)}ms
            </div>
            <div>
                Update: ${this.updateTime.toFixed(2)}ms
            </div>
            ${this.memoryUsage > 0 ? `
            <div>
                Memory: ${(this.memoryUsage / 1024 / 1024).toFixed(2)}MB
            </div>
            ` : ''}
            <div style="margin-top: 5px; height: 20px; background: #333;">
                ${this.renderFPSChart()}
            </div>
        `;
        
        this.displayElement.innerHTML = html;
    }
    
    private renderFPSChart(): string {
        const bars: string[] = [];
        const maxFps = 60;
        
        for (let i = Math.max(0, this.fpsHistory.length - 30); i < this.fpsHistory.length; i++) {
            const fps = this.fpsHistory[i];
            const height = (fps / maxFps) * 100;
            const color = this.getFPSColor(fps);
            
            bars.push(`<div style="
                display: inline-block;
                width: 3px;
                height: ${height}%;
                background: ${color};
                margin-right: 1px;
                vertical-align: bottom;
            "></div>`);
        }
        
        return bars.join('');
    }
    
    private getFPSColor(fps: number): string {
        if (fps >= this.FPS_WARNING) return '#00ff00';  // Green
        if (fps >= this.FPS_CRITICAL) return '#ffaa00';  // Yellow
        return '#ff0000';  // Red
    }
    
    // ========================================================================
    // FPS Tracking
    // ========================================================================
    
    public startFrame(): void {
        if (!this.enabled) return;
        
        const now = performance.now();
        const delta = now - this.lastTime;
        
        if (delta >= 1000) {
            this.fps = (this.frameCount * 1000) / delta;
            this.fpsHistory.push(this.fps);
            
            if (this.fpsHistory.length > this.maxHistorySize) {
                this.fpsHistory.shift();
            }
            
            this.frameCount = 0;
            this.lastTime = now;
            
            this.updateDisplay();
        }
        
        this.frameCount++;
    }
    
    public measureRender<T>(fn: () => T): T {
        const start = performance.now();
        const result = fn();
        this.renderTime = performance.now() - start;
        return result;
    }
    
    public measureLayout<T>(fn: () => T): T {
        const start = performance.now();
        const result = fn();
        this.layoutTime = performance.now() - start;
        return result;
    }
    
    public measureUpdate<T>(fn: () => T): T {
        const start = performance.now();
        const result = fn();
        this.updateTime = performance.now() - start;
        return result;
    }
    
    // ========================================================================
    // Metrics
    // ========================================================================
    
    public getAverageFPS(): number {
        if (this.fpsHistory.length === 0) return this.fps;
        
        const sum = this.fpsHistory.reduce((a, b) => a + b, 0);
        return sum / this.fpsHistory.length;
    }
    
    public getMinFPS(): number {
        if (this.fpsHistory.length === 0) return this.fps;
        return Math.min(...this.fpsHistory);
    }
    
    public getMaxFPS(): number {
        if (this.fpsHistory.length === 0) return this.fps;
        return Math.max(...this.fpsHistory);
    }
    
    public getCurrentFPS(): number {
        return this.fps;
    }
    
    public getRenderTime(): number {
        return this.renderTime;
    }
    
    public getLayoutTime(): number {
        return this.layoutTime;
    }
    
    public getUpdateTime(): number {
        return this.updateTime;
    }
    
    public getTotalTime(): number {
        return this.renderTime + this.layoutTime + this.updateTime;
    }
    
    // ========================================================================
    // Memory
    // ========================================================================
    
    public updateMemoryUsage(): void {
        // @ts-ignore - performance.memory is non-standard
        if (performance.memory) {
            // @ts-ignore
            this.memoryUsage = performance.memory.usedJSHeapSize;
        }
    }
    
    public getMemoryUsage(): number {
        return this.memoryUsage;
    }
    
    // ========================================================================
    // Control
    // ========================================================================
    
    public enable(): void {
        this.enabled = true;
        if (this.displayElement) {
            this.displayElement.style.display = 'block';
        }
        this.reset();
    }
    
    public disable(): void {
        this.enabled = false;
        if (this.displayElement) {
            this.displayElement.style.display = 'none';
        }
    }
    
    public toggle(): void {
        if (this.enabled) {
            this.disable();
        } else {
            this.enable();
        }
    }
    
    public isEnabled(): boolean {
        return this.enabled;
    }
    
    public reset(): void {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fpsHistory = [];
        this.renderTime = 0;
        this.layoutTime = 0;
        this.updateTime = 0;
    }
    
    // ========================================================================
    // Warnings
    // ========================================================================
    
    public getPerformanceWarnings(): string[] {
        const warnings: string[] = [];
        
        const avgFps = this.getAverageFPS();
        if (avgFps < this.FPS_CRITICAL) {
            warnings.push(`Critical: FPS ${avgFps.toFixed(1)} (target: 60)`);
        } else if (avgFps < this.FPS_WARNING) {
            warnings.push(`Warning: FPS ${avgFps.toFixed(1)} (target: 60)`);
        }
        
        if (this.renderTime > 16) {
            warnings.push(`Render time ${this.renderTime.toFixed(2)}ms exceeds 16ms budget`);
        }
        
        if (this.layoutTime > 16) {
            warnings.push(`Layout time ${this.layoutTime.toFixed(2)}ms exceeds 16ms budget`);
        }
        
        if (this.memoryUsage > 100 * 1024 * 1024) {
            warnings.push(`Memory usage ${(this.memoryUsage / 1024 / 1024).toFixed(2)}MB is high`);
        }
        
        return warnings;
    }
    
    // ========================================================================
    // Report
    // ========================================================================
    
    public getReport(): any {
        return {
            fps: {
                current: this.fps,
                average: this.getAverageFPS(),
                min: this.getMinFPS(),
                max: this.getMaxFPS()
            },
            timing: {
                render: this.renderTime,
                layout: this.layoutTime,
                update: this.updateTime,
                total: this.getTotalTime()
            },
            memory: {
                used: this.memoryUsage,
                usedMB: this.memoryUsage / 1024 / 1024
            },
            warnings: this.getPerformanceWarnings()
        };
    }
    
    public logReport(): void {
        console.table(this.getReport());
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    public destroy(): void {
        if (this.displayElement && this.displayElement.parentNode) {
            this.displayElement.parentNode.removeChild(this.displayElement);
        }
    }
}
