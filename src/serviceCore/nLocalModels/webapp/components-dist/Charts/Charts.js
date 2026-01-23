var __defProp = Object.defineProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __moduleCache = /* @__PURE__ */ new WeakMap;
var __toCommonJS = (from) => {
  var entry = __moduleCache.get(from), desc;
  if (entry)
    return entry;
  entry = __defProp({}, "__esModule", { value: true });
  if (from && typeof from === "object" || typeof from === "function")
    __getOwnPropNames(from).map((key) => !__hasOwnProp.call(entry, key) && __defProp(entry, key, {
      get: () => from[key],
      enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable
    }));
  __moduleCache.set(from, entry);
  return entry;
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true,
      configurable: true,
      set: (newValue) => all[name] = () => newValue
    });
};
var __esm = (fn, res) => () => (fn && (res = fn(fn = 0)), res);

// Charts/types.ts
var CHART_COLORS, DEFAULT_CHART_CONFIG, DEFAULT_RADIAL_CONFIG, DEFAULT_GAUGE_CONFIG;
var init_types = __esm(() => {
  CHART_COLORS = {
    primary: "#0a6ed1",
    success: "#107e3e",
    warning: "#e9730c",
    error: "#bb0000",
    neutral: "#6a6d70",
    gradient: {
      blue: ["#0a6ed1", "#1a9fff"],
      green: ["#107e3e", "#2ecc71"],
      orange: ["#e9730c", "#f39c12"],
      red: ["#bb0000", "#e74c3c"]
    },
    series: [
      "#0a6ed1",
      "#107e3e",
      "#e9730c",
      "#9b59b6",
      "#1abc9c",
      "#e74c3c",
      "#3498db",
      "#f39c12"
    ]
  };
  DEFAULT_CHART_CONFIG = {
    width: 400,
    height: 300,
    margin: { top: 20, right: 20, bottom: 40, left: 50 },
    animate: true,
    animationDuration: 300,
    responsive: true
  };
  DEFAULT_RADIAL_CONFIG = {
    maxValue: 100,
    minValue: 0,
    arcWidth: 20,
    showValue: true,
    showLabel: true,
    thresholds: [
      { value: 33, color: CHART_COLORS.error },
      { value: 66, color: CHART_COLORS.warning },
      { value: 100, color: CHART_COLORS.success }
    ]
  };
  DEFAULT_GAUGE_CONFIG = {
    min: 0,
    max: 100,
    showNeedle: true,
    needleColor: "#333",
    zones: [
      { min: 0, max: 33, color: CHART_COLORS.success, label: "Good" },
      { min: 33, max: 66, color: CHART_COLORS.warning, label: "Warning" },
      { min: 66, max: 100, color: CHART_COLORS.error, label: "Critical" }
    ]
  };
});

// Charts/RadialChart.ts
var exports_RadialChart = {};
__export(exports_RadialChart, {
  RadialChart: () => RadialChart
});

class RadialChart {
  container;
  svg;
  config;
  backgroundArc;
  valueArc;
  valueText;
  labelText;
  constructor(container, config) {
    if (typeof container === "string") {
      const el = document.querySelector(container);
      if (!el)
        throw new Error(`Container not found: ${container}`);
      this.container = el;
    } else {
      this.container = container;
    }
    this.config = {
      ...DEFAULT_CHART_CONFIG,
      ...DEFAULT_RADIAL_CONFIG,
      ...config,
      value: config.value ?? 0
    };
    this.svg = this.createSVG();
    this.container.appendChild(this.svg);
    this.backgroundArc = this.createArc("background");
    this.valueArc = this.createArc("value");
    this.valueText = this.createValueText();
    this.labelText = this.createLabelText();
    this.render();
    if (this.config.responsive) {
      this.setupResizeObserver();
    }
  }
  createSVG() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(this.config.width));
    svg.setAttribute("height", String(this.config.height));
    svg.setAttribute("viewBox", `0 0 ${this.config.width} ${this.config.height}`);
    svg.style.overflow = "visible";
    return svg;
  }
  createArc(type) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke-linecap", "round");
    if (type === "background") {
      path.setAttribute("stroke", "#e0e0e0");
      path.setAttribute("stroke-width", String(this.config.arcWidth));
    } else {
      path.setAttribute("stroke-width", String(this.config.arcWidth + 2));
    }
    this.svg.appendChild(path);
    return path;
  }
  createValueText() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dominant-baseline", "middle");
    text.setAttribute("font-size", "28");
    text.setAttribute("font-weight", "bold");
    text.setAttribute("fill", "#333");
    this.svg.appendChild(text);
    return text;
  }
  createLabelText() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "14");
    text.setAttribute("fill", "#666");
    this.svg.appendChild(text);
    return text;
  }
  getColorForValue(value) {
    const thresholds = this.config.thresholds || DEFAULT_RADIAL_CONFIG.thresholds;
    for (const threshold of thresholds) {
      if (value <= threshold.value) {
        return threshold.color;
      }
    }
    return CHART_COLORS.primary;
  }
  describeArc(cx, cy, radius, startAngle, endAngle) {
    const start = this.polarToCartesian(cx, cy, radius, endAngle);
    const end = this.polarToCartesian(cx, cy, radius, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? 0 : 1;
    return [
      "M",
      start.x,
      start.y,
      "A",
      radius,
      radius,
      0,
      largeArcFlag,
      0,
      end.x,
      end.y
    ].join(" ");
  }
  polarToCartesian(cx, cy, radius, angleInDegrees) {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180;
    return {
      x: cx + radius * Math.cos(angleInRadians),
      y: cy + radius * Math.sin(angleInRadians)
    };
  }
  render() {
    const width = this.config.width;
    const height = this.config.height;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) / 2 - this.config.arcWidth - 10;
    this.backgroundArc.setAttribute("d", this.describeArc(cx, cy, radius, 0, 359.99));
    const percentage = (this.config.value - this.config.minValue) / (this.config.maxValue - this.config.minValue);
    const endAngle = Math.max(0.01, percentage * 360);
    this.valueArc.setAttribute("d", this.describeArc(cx, cy, radius, 0, endAngle));
    this.valueArc.setAttribute("stroke", this.getColorForValue(this.config.value));
    if (this.config.animate) {
      this.valueArc.style.transition = `stroke-dashoffset ${this.config.animationDuration}ms ease-out`;
    }
    if (this.config.showValue) {
      const displayValue = this.config.unit ? `${Math.round(this.config.value)}${this.config.unit}` : String(Math.round(this.config.value));
      this.valueText.textContent = displayValue;
      this.valueText.setAttribute("x", String(cx));
      this.valueText.setAttribute("y", String(cy));
    }
    if (this.config.showLabel && this.config.label) {
      this.labelText.textContent = this.config.label;
      this.labelText.setAttribute("x", String(cx));
      this.labelText.setAttribute("y", String(cy + 25));
    }
  }
  setupResizeObserver() {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          this.config.width = width;
          this.config.height = height;
          this.svg.setAttribute("width", String(width));
          this.svg.setAttribute("height", String(height));
          this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
          this.render();
        }
      }
    });
    observer.observe(this.container);
  }
  setValue(value) {
    this.config.value = Math.max(this.config.minValue, Math.min(this.config.maxValue, value));
    this.render();
  }
  getValue() {
    return this.config.value;
  }
  setLabel(label) {
    this.config.label = label;
    this.render();
  }
  destroy() {
    this.container.removeChild(this.svg);
  }
}
var init_RadialChart = __esm(() => {
  init_types();
});

// Charts/GaugeChart.ts
var exports_GaugeChart = {};
__export(exports_GaugeChart, {
  GaugeChart: () => GaugeChart
});

class GaugeChart {
  container;
  svg;
  config;
  zones = [];
  needle;
  valueText;
  labelText;
  constructor(container, config) {
    if (typeof container === "string") {
      const el = document.querySelector(container);
      if (!el)
        throw new Error(`Container not found: ${container}`);
      this.container = el;
    } else {
      this.container = container;
    }
    this.config = {
      ...DEFAULT_CHART_CONFIG,
      ...DEFAULT_GAUGE_CONFIG,
      ...config,
      value: config.value ?? 0
    };
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
  createSVG() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(this.config.width));
    svg.setAttribute("height", String(this.config.height));
    svg.setAttribute("viewBox", `0 0 ${this.config.width} ${this.config.height}`);
    svg.style.overflow = "visible";
    return svg;
  }
  createZones() {
    const zonesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    for (const zone of this.config.zones || []) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("fill", zone.color);
      path.setAttribute("opacity", "0.8");
      this.zones.push(path);
      zonesGroup.appendChild(path);
    }
    this.svg.appendChild(zonesGroup);
  }
  createNeedle() {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const needle = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    needle.setAttribute("fill", this.config.needleColor || "#333");
    group.appendChild(needle);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("fill", this.config.needleColor || "#333");
    circle.setAttribute("r", "8");
    group.appendChild(circle);
    this.svg.appendChild(group);
    return group;
  }
  createValueText() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "24");
    text.setAttribute("font-weight", "bold");
    text.setAttribute("fill", "#333");
    this.svg.appendChild(text);
    return text;
  }
  createLabelText() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "12");
    text.setAttribute("fill", "#666");
    this.svg.appendChild(text);
    return text;
  }
  describeArc(cx, cy, innerR, outerR, startAngle, endAngle) {
    const startOuter = this.polarToCartesian(cx, cy, outerR, endAngle);
    const endOuter = this.polarToCartesian(cx, cy, outerR, startAngle);
    const startInner = this.polarToCartesian(cx, cy, innerR, startAngle);
    const endInner = this.polarToCartesian(cx, cy, innerR, endAngle);
    const largeArc = endAngle - startAngle <= 180 ? 0 : 1;
    return [
      "M",
      startOuter.x,
      startOuter.y,
      "A",
      outerR,
      outerR,
      0,
      largeArc,
      0,
      endOuter.x,
      endOuter.y,
      "L",
      startInner.x,
      startInner.y,
      "A",
      innerR,
      innerR,
      0,
      largeArc,
      1,
      endInner.x,
      endInner.y,
      "Z"
    ].join(" ");
  }
  polarToCartesian(cx, cy, radius, angleInDegrees) {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180;
    return {
      x: cx + radius * Math.cos(angleInRadians),
      y: cy + radius * Math.sin(angleInRadians)
    };
  }
  valueToAngle(value) {
    const range = this.config.max - this.config.min;
    const normalized = (value - this.config.min) / range;
    return -135 + normalized * 270;
  }
  render() {
    const width = this.config.width;
    const height = this.config.height;
    const cx = width / 2;
    const cy = height * 0.65;
    const outerR = Math.min(width, height) / 2 - 20;
    const innerR = outerR - 30;
    const zones = this.config.zones || [];
    zones.forEach((zone, i) => {
      const startAngle = this.valueToAngle(zone.min);
      const endAngle = this.valueToAngle(zone.max);
      this.zones[i].setAttribute("d", this.describeArc(cx, cy, innerR, outerR, startAngle, endAngle));
    });
    if (this.config.showNeedle) {
      const angle = this.valueToAngle(this.config.value);
      const needleLength = innerR - 10;
      const tip = this.polarToCartesian(cx, cy, needleLength, angle);
      const baseLeft = this.polarToCartesian(cx, cy, 10, angle - 90);
      const baseRight = this.polarToCartesian(cx, cy, 10, angle + 90);
      const needle = this.needle.querySelector("polygon");
      needle.setAttribute("points", `${tip.x},${tip.y} ${baseLeft.x},${baseLeft.y} ${baseRight.x},${baseRight.y}`);
      const circle = this.needle.querySelector("circle");
      circle.setAttribute("cx", String(cx));
      circle.setAttribute("cy", String(cy));
      if (this.config.animate) {
        this.needle.style.transition = `transform ${this.config.animationDuration}ms ease-out`;
      }
    }
    const displayValue = this.config.unit ? `${this.config.value.toFixed(1)}${this.config.unit}` : this.config.value.toFixed(1);
    this.valueText.textContent = displayValue;
    this.valueText.setAttribute("x", String(cx));
    this.valueText.setAttribute("y", String(cy + 40));
    if (this.config.label) {
      this.labelText.textContent = this.config.label;
      this.labelText.setAttribute("x", String(cx));
      this.labelText.setAttribute("y", String(cy + 60));
    }
  }
  setupResizeObserver() {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          this.config.width = width;
          this.config.height = height;
          this.svg.setAttribute("width", String(width));
          this.svg.setAttribute("height", String(height));
          this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
          this.render();
        }
      }
    });
    observer.observe(this.container);
  }
  setValue(value) {
    this.config.value = Math.max(this.config.min, Math.min(this.config.max, value));
    this.render();
  }
  getValue() {
    return this.config.value;
  }
  destroy() {
    this.container.removeChild(this.svg);
  }
}
var init_GaugeChart = __esm(() => {
  init_types();
});

// Charts/LineChart.ts
var exports_LineChart = {};
__export(exports_LineChart, {
  LineChart: () => LineChart
});

class LineChart {
  container;
  svg;
  config;
  chartArea;
  xAxisGroup;
  yAxisGroup;
  gridGroup;
  linesGroup;
  tooltipDiv = null;
  legendDiv = null;
  xScale = () => 0;
  yScale = () => 0;
  xDomain = [0, 100];
  yDomain = [0, 1];
  constructor(container, config) {
    if (typeof container === "string") {
      const el = document.querySelector(container);
      if (!el)
        throw new Error(`Container not found: ${container}`);
      this.container = el;
    } else {
      this.container = container;
    }
    this.config = {
      ...DEFAULT_CHART_CONFIG,
      showGrid: true,
      showLegend: true,
      showTooltip: true,
      xAxisType: "linear",
      yAxisType: "linear",
      ...config,
      series: config.series || []
    };
    this.container.style.position = "relative";
    this.svg = this.createSVG();
    this.container.appendChild(this.svg);
    this.gridGroup = this.createGroup("grid");
    this.linesGroup = this.createGroup("lines");
    this.xAxisGroup = this.createGroup("x-axis");
    this.yAxisGroup = this.createGroup("y-axis");
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
  createSVG() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(this.config.width));
    svg.setAttribute("height", String(this.config.height));
    svg.style.overflow = "visible";
    return svg;
  }
  createGroup(className) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", className);
    this.svg.appendChild(g);
    return g;
  }
  createChartArea() {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const m = this.config.margin;
    g.setAttribute("transform", `translate(${m.left}, ${m.top})`);
    this.svg.appendChild(g);
    return g;
  }
  createTooltip() {
    const div = document.createElement("div");
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
  createLegend() {
    const div = document.createElement("div");
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
  calculateDomains() {
    if (this.config.series.length === 0)
      return;
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    for (const series of this.config.series) {
      for (const point of series.data) {
        const x = typeof point.x === "number" ? point.x : 0;
        xMin = Math.min(xMin, x);
        xMax = Math.max(xMax, x);
        yMin = Math.min(yMin, point.y);
        yMax = Math.max(yMax, point.y);
      }
    }
    const yPadding = (yMax - yMin) * 0.1 || 0.1;
    this.xDomain = [xMin, xMax];
    this.yDomain = [yMin - yPadding, yMax + yPadding];
  }
  calculateScales() {
    const m = this.config.margin;
    const width = this.config.width - m.left - m.right;
    const height = this.config.height - m.top - m.bottom;
    this.xScale = (val) => {
      const range = this.xDomain[1] - this.xDomain[0] || 1;
      return (val - this.xDomain[0]) / range * width;
    };
    this.yScale = (val) => {
      const range = this.yDomain[1] - this.yDomain[0] || 1;
      return height - (val - this.yDomain[0]) / range * height;
    };
  }
  renderGrid() {
    this.gridGroup.innerHTML = "";
    if (!this.config.showGrid)
      return;
    const m = this.config.margin;
    const width = this.config.width - m.left - m.right;
    const height = this.config.height - m.top - m.bottom;
    for (let i = 0;i <= 5; i++) {
      const y = m.top + height / 5 * i;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(m.left));
      line.setAttribute("y1", String(y));
      line.setAttribute("x2", String(m.left + width));
      line.setAttribute("y2", String(y));
      line.setAttribute("stroke", "#e0e0e0");
      line.setAttribute("stroke-dasharray", "3,3");
      this.gridGroup.appendChild(line);
    }
    for (let i = 0;i <= 5; i++) {
      const x = m.left + width / 5 * i;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(x));
      line.setAttribute("y1", String(m.top));
      line.setAttribute("x2", String(x));
      line.setAttribute("y2", String(m.top + height));
      line.setAttribute("stroke", "#e0e0e0");
      line.setAttribute("stroke-dasharray", "3,3");
      this.gridGroup.appendChild(line);
    }
  }
  renderAxes() {
    this.xAxisGroup.innerHTML = "";
    this.yAxisGroup.innerHTML = "";
    const m = this.config.margin;
    const width = this.config.width - m.left - m.right;
    const height = this.config.height - m.top - m.bottom;
    const xAxisLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    xAxisLine.setAttribute("x1", String(m.left));
    xAxisLine.setAttribute("y1", String(m.top + height));
    xAxisLine.setAttribute("x2", String(m.left + width));
    xAxisLine.setAttribute("y2", String(m.top + height));
    xAxisLine.setAttribute("stroke", "#333");
    this.xAxisGroup.appendChild(xAxisLine);
    for (let i = 0;i <= 5; i++) {
      const val = this.xDomain[0] + (this.xDomain[1] - this.xDomain[0]) / 5 * i;
      const x = m.left + width / 5 * i;
      const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tick.setAttribute("x1", String(x));
      tick.setAttribute("y1", String(m.top + height));
      tick.setAttribute("x2", String(x));
      tick.setAttribute("y2", String(m.top + height + 5));
      tick.setAttribute("stroke", "#333");
      this.xAxisGroup.appendChild(tick);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(x));
      label.setAttribute("y", String(m.top + height + 18));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", "10");
      label.setAttribute("fill", "#666");
      label.textContent = String(Math.round(val));
      this.xAxisGroup.appendChild(label);
    }
    const yAxisLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    yAxisLine.setAttribute("x1", String(m.left));
    yAxisLine.setAttribute("y1", String(m.top));
    yAxisLine.setAttribute("x2", String(m.left));
    yAxisLine.setAttribute("y2", String(m.top + height));
    yAxisLine.setAttribute("stroke", "#333");
    this.yAxisGroup.appendChild(yAxisLine);
    for (let i = 0;i <= 5; i++) {
      const val = this.yDomain[0] + (this.yDomain[1] - this.yDomain[0]) / 5 * i;
      const y = m.top + height - height / 5 * i;
      const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tick.setAttribute("x1", String(m.left - 5));
      tick.setAttribute("y1", String(y));
      tick.setAttribute("x2", String(m.left));
      tick.setAttribute("y2", String(y));
      tick.setAttribute("stroke", "#333");
      this.yAxisGroup.appendChild(tick);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(m.left - 8));
      label.setAttribute("y", String(y + 3));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("font-size", "10");
      label.setAttribute("fill", "#666");
      label.textContent = val.toFixed(2);
      this.yAxisGroup.appendChild(label);
    }
    if (this.config.xAxisLabel) {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(m.left + width / 2));
      label.setAttribute("y", String(this.config.height - 5));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", "12");
      label.setAttribute("fill", "#333");
      label.textContent = this.config.xAxisLabel;
      this.xAxisGroup.appendChild(label);
    }
    if (this.config.yAxisLabel) {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("transform", `translate(12, ${m.top + height / 2}) rotate(-90)`);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", "12");
      label.setAttribute("fill", "#333");
      label.textContent = this.config.yAxisLabel;
      this.yAxisGroup.appendChild(label);
    }
  }
  renderLines() {
    this.linesGroup.innerHTML = "";
    const m = this.config.margin;
    this.config.series.forEach((series, idx) => {
      if (series.data.length === 0)
        return;
      const color = series.color || CHART_COLORS.series[idx % CHART_COLORS.series.length];
      const lineWidth = series.lineWidth || 2;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const d = series.data.map((point, i) => {
        const x = m.left + this.xScale(typeof point.x === "number" ? point.x : i);
        const y = m.top + this.yScale(point.y);
        return `${i === 0 ? "M" : "L"} ${x} ${y}`;
      }).join(" ");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-width", String(lineWidth));
      if (series.dashed) {
        path.setAttribute("stroke-dasharray", "5,5");
      }
      this.linesGroup.appendChild(path);
      if (series.showPoints !== false) {
        series.data.forEach((point, i) => {
          const x = m.left + this.xScale(typeof point.x === "number" ? point.x : i);
          const y = m.top + this.yScale(point.y);
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", String(x));
          circle.setAttribute("cy", String(y));
          circle.setAttribute("r", "4");
          circle.setAttribute("fill", color);
          circle.setAttribute("stroke", "white");
          circle.setAttribute("stroke-width", "2");
          circle.style.cursor = "pointer";
          if (this.tooltipDiv) {
            circle.addEventListener("mouseenter", (e) => {
              this.tooltipDiv.innerHTML = `
                                <strong>${series.name}</strong><br>
                                X: ${point.x}<br>
                                Y: ${point.y.toFixed(4)}
                            `;
              this.tooltipDiv.style.opacity = "1";
              this.tooltipDiv.style.left = `${e.offsetX + 10}px`;
              this.tooltipDiv.style.top = `${e.offsetY - 30}px`;
            });
            circle.addEventListener("mouseleave", () => {
              this.tooltipDiv.style.opacity = "0";
            });
          }
          this.linesGroup.appendChild(circle);
        });
      }
    });
  }
  renderLegend() {
    if (!this.legendDiv)
      return;
    this.legendDiv.innerHTML = "";
    this.config.series.forEach((series, idx) => {
      const color = series.color || CHART_COLORS.series[idx % CHART_COLORS.series.length];
      const item = document.createElement("div");
      item.style.cssText = "display: flex; align-items: center; gap: 4px;";
      const swatch = document.createElement("div");
      swatch.style.cssText = `width: 12px; height: 12px; background: ${color}; border-radius: 2px;`;
      const label = document.createElement("span");
      label.textContent = series.name;
      item.appendChild(swatch);
      item.appendChild(label);
      this.legendDiv.appendChild(item);
    });
  }
  render() {
    this.calculateDomains();
    this.calculateScales();
    this.renderGrid();
    this.renderAxes();
    this.renderLines();
    this.renderLegend();
  }
  setupResizeObserver() {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          this.config.width = width;
          this.config.height = height - (this.legendDiv ? 30 : 0);
          this.svg.setAttribute("width", String(this.config.width));
          this.svg.setAttribute("height", String(this.config.height));
          this.render();
        }
      }
    });
    observer.observe(this.container);
  }
  setData(series) {
    this.config.series = series;
    this.render();
  }
  addPoint(seriesIndex, point) {
    if (this.config.series[seriesIndex]) {
      this.config.series[seriesIndex].data.push(point);
      this.render();
    }
  }
  destroy() {
    this.container.removeChild(this.svg);
    if (this.tooltipDiv)
      this.container.removeChild(this.tooltipDiv);
    if (this.legendDiv)
      this.container.removeChild(this.legendDiv);
  }
}
var init_LineChart = __esm(() => {
  init_types();
});

// Charts/BarChart.ts
var exports_BarChart = {};
__export(exports_BarChart, {
  BarChart: () => BarChart
});

class BarChart {
  container;
  svg;
  config;
  barsGroup;
  xAxisGroup;
  yAxisGroup;
  tooltipDiv = null;
  maxValue = 0;
  constructor(container, config) {
    if (typeof container === "string") {
      const el = document.querySelector(container);
      if (!el)
        throw new Error(`Container not found: ${container}`);
      this.container = el;
    } else {
      this.container = container;
    }
    this.config = {
      ...DEFAULT_CHART_CONFIG,
      orientation: "vertical",
      showValues: true,
      grouped: false,
      stacked: false,
      ...config,
      data: config.data || []
    };
    this.container.style.position = "relative";
    this.svg = this.createSVG();
    this.container.appendChild(this.svg);
    this.barsGroup = this.createGroup("bars");
    this.xAxisGroup = this.createGroup("x-axis");
    this.yAxisGroup = this.createGroup("y-axis");
    this.tooltipDiv = this.createTooltip();
    this.render();
    if (this.config.responsive) {
      this.setupResizeObserver();
    }
  }
  createSVG() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(this.config.width));
    svg.setAttribute("height", String(this.config.height));
    svg.style.overflow = "visible";
    return svg;
  }
  createGroup(className) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", className);
    this.svg.appendChild(g);
    return g;
  }
  createTooltip() {
    const div = document.createElement("div");
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
  calculateMaxValue() {
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
    this.maxValue *= 1.1;
  }
  renderVerticalBars() {
    const m = this.config.margin;
    const width = this.config.width - m.left - m.right;
    const height = this.config.height - m.top - m.bottom;
    const categories = this.config.data;
    const categoryWidth = width / categories.length;
    const barPadding = categoryWidth * 0.2;
    const numSeries = categories[0]?.values.length || 1;
    const barWidth = this.config.grouped ? (categoryWidth - barPadding * 2) / numSeries : categoryWidth - barPadding * 2;
    categories.forEach((category, catIdx) => {
      const baseX = m.left + catIdx * categoryWidth + barPadding;
      if (this.config.stacked) {
        let stackY = 0;
        category.values.forEach((val, valIdx) => {
          const barHeight = val.value / this.maxValue * height;
          const color = val.color || CHART_COLORS.series[valIdx % CHART_COLORS.series.length];
          const rect = this.createBar(baseX, m.top + height - stackY - barHeight, barWidth, barHeight, color, val.name, val.value);
          this.barsGroup.appendChild(rect);
          stackY += barHeight;
        });
      } else if (this.config.grouped) {
        category.values.forEach((val, valIdx) => {
          const barHeight = val.value / this.maxValue * height;
          const color = val.color || CHART_COLORS.series[valIdx % CHART_COLORS.series.length];
          const x = baseX + valIdx * barWidth;
          const rect = this.createBar(x, m.top + height - barHeight, barWidth - 2, barHeight, color, val.name, val.value);
          this.barsGroup.appendChild(rect);
        });
      } else {
        const val = category.values[0];
        if (val) {
          const barHeight = val.value / this.maxValue * height;
          const color = val.color || CHART_COLORS.series[catIdx % CHART_COLORS.series.length];
          const rect = this.createBar(baseX, m.top + height - barHeight, barWidth, barHeight, color, category.category, val.value);
          this.barsGroup.appendChild(rect);
        }
      }
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(baseX + barWidth / 2 * (this.config.grouped ? numSeries : 1)));
      label.setAttribute("y", String(m.top + height + 18));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", "11");
      label.setAttribute("fill", "#666");
      label.textContent = category.category;
      this.xAxisGroup.appendChild(label);
    });
  }
  createBar(x, y, width, height, color, name, value) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(Math.max(0, width)));
    rect.setAttribute("height", String(Math.max(0, height)));
    rect.setAttribute("fill", color);
    rect.setAttribute("rx", "2");
    rect.style.cursor = "pointer";
    rect.style.transition = "opacity 0.15s";
    if (this.config.showValues && height > 20) {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(x + width / 2));
      text.setAttribute("y", String(y + 15));
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("font-size", "10");
      text.setAttribute("fill", "white");
      text.setAttribute("font-weight", "bold");
      text.textContent = value.toFixed(1);
      this.barsGroup.appendChild(text);
    }
    rect.addEventListener("mouseenter", (e) => {
      rect.setAttribute("opacity", "0.8");
      if (this.tooltipDiv) {
        this.tooltipDiv.innerHTML = `<strong>${name}</strong><br>Value: ${value.toFixed(2)}`;
        this.tooltipDiv.style.opacity = "1";
        this.tooltipDiv.style.left = `${e.offsetX + 10}px`;
        this.tooltipDiv.style.top = `${e.offsetY - 30}px`;
      }
    });
    rect.addEventListener("mouseleave", () => {
      rect.setAttribute("opacity", "1");
      if (this.tooltipDiv) {
        this.tooltipDiv.style.opacity = "0";
      }
    });
    return rect;
  }
  renderAxes() {
    const m = this.config.margin;
    const height = this.config.height - m.top - m.bottom;
    const yAxisLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    yAxisLine.setAttribute("x1", String(m.left));
    yAxisLine.setAttribute("y1", String(m.top));
    yAxisLine.setAttribute("x2", String(m.left));
    yAxisLine.setAttribute("y2", String(m.top + height));
    yAxisLine.setAttribute("stroke", "#333");
    this.yAxisGroup.appendChild(yAxisLine);
    for (let i = 0;i <= 5; i++) {
      const val = this.maxValue / 5 * i;
      const y = m.top + height - height / 5 * i;
      const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tick.setAttribute("x1", String(m.left - 5));
      tick.setAttribute("y1", String(y));
      tick.setAttribute("x2", String(m.left));
      tick.setAttribute("y2", String(y));
      tick.setAttribute("stroke", "#333");
      this.yAxisGroup.appendChild(tick);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(m.left - 8));
      label.setAttribute("y", String(y + 4));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("font-size", "10");
      label.setAttribute("fill", "#666");
      label.textContent = val.toFixed(1);
      this.yAxisGroup.appendChild(label);
    }
    if (this.config.yAxisLabel) {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("transform", `translate(12, ${m.top + height / 2}) rotate(-90)`);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", "12");
      label.setAttribute("fill", "#333");
      label.textContent = this.config.yAxisLabel;
      this.yAxisGroup.appendChild(label);
    }
  }
  render() {
    this.barsGroup.innerHTML = "";
    this.xAxisGroup.innerHTML = "";
    this.yAxisGroup.innerHTML = "";
    this.calculateMaxValue();
    if (this.config.orientation === "vertical") {
      this.renderVerticalBars();
    }
    this.renderAxes();
  }
  setupResizeObserver() {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          this.config.width = width;
          this.config.height = height;
          this.svg.setAttribute("width", String(width));
          this.svg.setAttribute("height", String(height));
          this.render();
        }
      }
    });
    observer.observe(this.container);
  }
  setData(data) {
    this.config.data = data;
    this.render();
  }
  destroy() {
    this.container.removeChild(this.svg);
    if (this.tooltipDiv)
      this.container.removeChild(this.tooltipDiv);
  }
}
var init_BarChart = __esm(() => {
  init_types();
});

// Charts/SankeyDiagram.ts
var exports_SankeyDiagram = {};
__export(exports_SankeyDiagram, {
  SankeyDiagram: () => SankeyDiagram
});

class SankeyDiagram {
  container;
  svg;
  config;
  computedNodes = new Map;
  computedLinks = [];
  linksGroup;
  nodesGroup;
  labelsGroup;
  constructor(container, config) {
    if (typeof container === "string") {
      const el = document.querySelector(container);
      if (!el)
        throw new Error(`Container not found: ${container}`);
      this.container = el;
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
    };
    this.svg = this.createSVG();
    this.container.appendChild(this.svg);
    this.linksGroup = this.createGroup("links");
    this.nodesGroup = this.createGroup("nodes");
    this.labelsGroup = this.createGroup("labels");
    this.computeLayout();
    this.render();
    if (this.config.responsive) {
      this.setupResizeObserver();
    }
  }
  createSVG() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(this.config.width));
    svg.setAttribute("height", String(this.config.height));
    svg.style.overflow = "visible";
    return svg;
  }
  createGroup(className) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", className);
    this.svg.appendChild(g);
    return g;
  }
  computeLayout() {
    const m = this.config.margin;
    const width = this.config.width - m.left - m.right;
    const height = this.config.height - m.top - m.bottom;
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
    for (const link of this.config.links) {
      const source = this.computedNodes.get(link.source);
      const target = this.computedNodes.get(link.target);
      if (source && target) {
        source.value += link.value;
        target.value += link.value;
      }
    }
    const columns = [];
    const visited = new Set;
    const sourceNodes = [];
    for (const [id, node] of this.computedNodes) {
      const hasIncoming = this.config.links.some((l) => l.target === id);
      if (!hasIncoming) {
        sourceNodes.push(node);
      }
    }
    let currentColumn = sourceNodes;
    while (currentColumn.length > 0) {
      columns.push([...currentColumn]);
      currentColumn.forEach((n) => visited.add(n.id));
      const nextColumn = [];
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
    const numColumns = columns.length;
    const columnWidth = numColumns > 1 ? width / (numColumns - 1) : width;
    columns.forEach((column, colIdx) => {
      const x = m.left + colIdx * columnWidth;
      const totalValue = column.reduce((sum, n) => sum + n.value, 0);
      const availableHeight = height - (column.length - 1) * this.config.nodePadding;
      let y = m.top;
      for (const node of column) {
        node.x = x;
        node.y = y;
        node.height = totalValue > 0 ? node.value / totalValue * availableHeight : 20;
        y += node.height + this.config.nodePadding;
      }
    });
    this.computedLinks = [];
    for (const link of this.config.links) {
      const sourceNode = this.computedNodes.get(link.source);
      const targetNode = this.computedNodes.get(link.target);
      if (sourceNode && targetNode) {
        const computedLink = {
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
        ty += link.width || node.height * (link.value / node.value);
      }
    }
  }
  renderLinks() {
    this.linksGroup.innerHTML = "";
    for (const link of this.computedLinks) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const x0 = link.sourceNode.x + this.config.nodeWidth;
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
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-width", String(Math.max(1, link.width)));
      path.setAttribute("opacity", "0.5");
      path.addEventListener("mouseenter", () => {
        path.setAttribute("opacity", "0.8");
      });
      path.addEventListener("mouseleave", () => {
        path.setAttribute("opacity", "0.5");
      });
      this.linksGroup.appendChild(path);
    }
  }
  renderNodes() {
    this.nodesGroup.innerHTML = "";
    let colorIdx = 0;
    for (const node of this.computedNodes.values()) {
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const color = node.color || CHART_COLORS.series[colorIdx % CHART_COLORS.series.length];
      rect.setAttribute("x", String(node.x));
      rect.setAttribute("y", String(node.y));
      rect.setAttribute("width", String(this.config.nodeWidth));
      rect.setAttribute("height", String(Math.max(1, node.height)));
      rect.setAttribute("fill", color);
      rect.setAttribute("rx", "2");
      rect.style.cursor = "pointer";
      rect.addEventListener("mouseenter", () => {
        rect.setAttribute("opacity", "0.8");
      });
      rect.addEventListener("mouseleave", () => {
        rect.setAttribute("opacity", "1");
      });
      this.nodesGroup.appendChild(rect);
      colorIdx++;
    }
  }
  renderLabels() {
    this.labelsGroup.innerHTML = "";
    if (!this.config.showLabels)
      return;
    for (const node of this.computedNodes.values()) {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      const midX = this.config.width / 2;
      const isLeft = node.x < midX;
      text.setAttribute("x", String(isLeft ? node.x + this.config.nodeWidth + 6 : node.x - 6));
      text.setAttribute("y", String(node.y + node.height / 2));
      text.setAttribute("text-anchor", isLeft ? "start" : "end");
      text.setAttribute("dominant-baseline", "middle");
      text.setAttribute("font-size", "12");
      text.setAttribute("fill", "#333");
      let labelText = node.name;
      if (this.config.showValues) {
        labelText += ` (${node.value})`;
      }
      text.textContent = labelText;
      this.labelsGroup.appendChild(text);
    }
  }
  render() {
    this.renderLinks();
    this.renderNodes();
    this.renderLabels();
  }
  setupResizeObserver() {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          this.config.width = width;
          this.config.height = height;
          this.svg.setAttribute("width", String(width));
          this.svg.setAttribute("height", String(height));
          this.computeLayout();
          this.render();
        }
      }
    });
    observer.observe(this.container);
  }
  setData(nodes, links) {
    this.config.nodes = nodes;
    this.config.links = links;
    this.computeLayout();
    this.render();
  }
  destroy() {
    this.container.removeChild(this.svg);
  }
}
var init_SankeyDiagram = __esm(() => {
  init_types();
});

// Charts/Charts.ts
init_RadialChart();
init_GaugeChart();
init_LineChart();
init_BarChart();
init_SankeyDiagram();
init_types();
function createChart(type, container, config) {
  switch (type) {
    case "radial":
      return new ((init_RadialChart(), __toCommonJS(exports_RadialChart))).RadialChart(container, config);
    case "gauge":
      return new ((init_GaugeChart(), __toCommonJS(exports_GaugeChart))).GaugeChart(container, config);
    case "line":
      return new ((init_LineChart(), __toCommonJS(exports_LineChart))).LineChart(container, config);
    case "bar":
      return new ((init_BarChart(), __toCommonJS(exports_BarChart))).BarChart(container, config);
    case "sankey":
      return new ((init_SankeyDiagram(), __toCommonJS(exports_SankeyDiagram))).SankeyDiagram(container, config);
    default:
      throw new Error(`Unknown chart type: ${type}`);
  }
}
export {
  createChart,
  SankeyDiagram,
  RadialChart,
  LineChart,
  GaugeChart,
  DEFAULT_RADIAL_CONFIG,
  DEFAULT_GAUGE_CONFIG,
  DEFAULT_CHART_CONFIG,
  CHART_COLORS,
  BarChart
};

//# debugId=1B2983747A4368E864756E2164756E21
