// ProcessFlow/types.ts
var ZOOM_LEVEL_CONFIG = {
  ["One" /* One */]: {
    scale: 1.2,
    nodeWidth: 200,
    nodeHeight: 100,
    showHeader: true,
    showStatus: true,
    showAttr1: true,
    showAttr2: true
  },
  ["Two" /* Two */]: {
    scale: 1,
    nodeWidth: 160,
    nodeHeight: 80,
    showHeader: true,
    showStatus: true,
    showAttr1: true,
    showAttr2: false
  },
  ["Three" /* Three */]: {
    scale: 0.8,
    nodeWidth: 120,
    nodeHeight: 60,
    showHeader: true,
    showStatus: true,
    showAttr1: false,
    showAttr2: false
  },
  ["Four" /* Four */]: {
    scale: 0.6,
    nodeWidth: 60,
    nodeHeight: 40,
    showHeader: false,
    showStatus: true,
    showAttr1: false,
    showAttr2: false
  }
};
var PROCESS_FLOW_COLORS = {
  positive: {
    background: "#107e3e",
    border: "#0a6534",
    text: "#ffffff"
  },
  negative: {
    background: "#bb0000",
    border: "#a20000",
    text: "#ffffff"
  },
  critical: {
    background: "#e9730c",
    border: "#c9630a",
    text: "#ffffff"
  },
  planned: {
    background: "#ededed",
    border: "#d9d9d9",
    text: "#32363a"
  },
  plannedNegative: {
    background: "#ededed",
    border: "#bb0000",
    text: "#32363a"
  },
  neutral: {
    background: "#0a6ed1",
    border: "#0854a0",
    text: "#ffffff"
  },
  connection: {
    normal: "#6a6d70",
    highlighted: "#0a6ed1",
    dimmed: "#d9d9d9"
  },
  lane: {
    default: "#ffffff",
    alternate: "#fafafa"
  },
  text: {
    primary: "#32363a",
    secondary: "#6a6d70",
    light: "#89919a"
  },
  border: "#d9d9d9",
  hover: "rgba(10, 110, 209, 0.1)",
  selected: "rgba(10, 110, 209, 0.2)",
  focus: "#0a6ed1"
};
var PROCESS_FLOW_LAYOUT = {
  node: {
    width: 160,
    height: 80,
    cornerRadius: 4,
    borderWidth: 2,
    padding: 12,
    iconSize: 24,
    titleFontSize: 14,
    textFontSize: 12
  },
  spacing: {
    horizontal: 80,
    vertical: 100,
    laneHeader: 120,
    topMargin: 20,
    bottomMargin: 20,
    leftMargin: 20,
    rightMargin: 20
  },
  connection: {
    strokeWidth: 2,
    arrowSize: 8,
    cornerRadius: 8,
    dashArray: "5,5"
  },
  zoom: {
    one: {
      scale: 1,
      showTexts: true,
      showIcons: true
    },
    two: {
      scale: 0.75,
      showTexts: true,
      showIcons: true
    },
    three: {
      scale: 0.5,
      showTexts: false,
      showIcons: true
    },
    four: {
      scale: 0.25,
      showTexts: false,
      showIcons: false
    }
  },
  animation: {
    duration: 300,
    easing: "cubic-bezier(0.4, 0, 0.2, 1)"
  }
};
var DEFAULT_PROCESS_FLOW_CONFIG = {
  showLabels: true,
  scrollable: true,
  foldedCorners: true,
  wheelZoomable: true,
  optimizeDisplay: true,
  zoomLevel: "One" /* One */
};

// ProcessFlow/ProcessFlowNode.ts
class ProcessFlowNode {
  id;
  lane;
  title;
  state;
  texts;
  children;
  position;
  foldedCorners;
  isAggregated;
  aggregatedCount;
  aggregatedItems;
  isExpanded = false;
  element;
  background;
  borderPath;
  foldTriangle = null;
  titleText;
  stateText;
  detailTexts = [];
  icon = null;
  statusIcon = null;
  stackElements = [];
  counterBadge = null;
  displayState = "Regular" /* Regular */;
  currentZoomLevel = "Two" /* Two */;
  currentZoomConfig = ZOOM_LEVEL_CONFIG["Two" /* Two */];
  x = 0;
  y = 0;
  constructor(config) {
    this.id = config.id;
    this.lane = config.lane;
    this.title = config.title;
    this.state = config.state;
    this.texts = config.texts || [];
    this.children = config.children || [];
    this.position = config.position || 0;
    this.foldedCorners = config.foldedCorners !== undefined ? config.foldedCorners : true;
    this.isAggregated = config.isAggregated || false;
    this.aggregatedCount = config.aggregatedCount || 0;
    this.aggregatedItems = config.aggregatedItems || [];
    this.element = this.createElement();
  }
  createElement() {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "process-flow-node");
    g.setAttribute("data-node-id", this.id);
    g.setAttribute("data-state", this.state);
    if (this.isAggregated) {
      g.classList.add("aggregated-node");
    }
    if (this.foldedCorners) {
      g.classList.add("folded-corner");
    }
    const layout = PROCESS_FLOW_LAYOUT.node;
    if (this.isAggregated) {
      this.renderStack(g);
    }
    this.background = this.foldedCorners ? this.createFoldedCornerPath() : this.createRegularPath();
    g.appendChild(this.background);
    this.borderPath = this.createBorderPath();
    g.appendChild(this.borderPath);
    if (this.foldedCorners) {
      this.foldTriangle = this.renderFoldedCorner();
      g.appendChild(this.foldTriangle);
    }
    if (this.needsIcon()) {
      this.icon = this.createIcon();
      g.appendChild(this.icon);
    }
    this.titleText = this.createTitle();
    g.appendChild(this.titleText);
    this.stateText = this.createStateText();
    g.appendChild(this.stateText);
    for (let i = 0;i < this.texts.length; i++) {
      const text = this.createDetailText(this.texts[i], i);
      this.detailTexts.push(text);
      g.appendChild(text);
    }
    this.statusIcon = this.createStatusIcon();
    g.appendChild(this.statusIcon);
    this.statusIcon.style.display = "none";
    if (this.isAggregated && this.aggregatedCount > 1) {
      this.renderCounter(g);
    }
    g.style.cursor = "pointer";
    g.style.transition = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)";
    return g;
  }
  createStatusIcon() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("class", "node-status-icon");
    text.setAttribute("x", "30");
    text.setAttribute("y", "28");
    text.setAttribute("font-size", "24");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dominant-baseline", "middle");
    text.setAttribute("fill", this.getStateColors().text);
    text.textContent = this.getIconForState();
    return text;
  }
  createFoldedCornerPath() {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "node-background");
    const w = PROCESS_FLOW_LAYOUT.node.width;
    const h = PROCESS_FLOW_LAYOUT.node.height;
    const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
    const foldSize = 12;
    const pathData = `
            M ${r} 0
            L ${w - foldSize} 0
            L ${w} ${foldSize}
            L ${w} ${h - r}
            Q ${w} ${h}, ${w - r} ${h}
            L ${r} ${h}
            Q 0 ${h}, 0 ${h - r}
            L 0 ${r}
            Q 0 0, ${r} 0
            Z
        `.trim().replace(/\s+/g, " ");
    path.setAttribute("d", pathData);
    path.setAttribute("fill", this.getStateColors().background);
    path.style.transition = "fill 0.3s ease";
    return path;
  }
  createRegularPath() {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "node-background");
    const w = PROCESS_FLOW_LAYOUT.node.width;
    const h = PROCESS_FLOW_LAYOUT.node.height;
    const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
    const pathData = `
            M ${r} 0
            L ${w - r} 0
            Q ${w} 0, ${w} ${r}
            L ${w} ${h - r}
            Q ${w} ${h}, ${w - r} ${h}
            L ${r} ${h}
            Q 0 ${h}, 0 ${h - r}
            L 0 ${r}
            Q 0 0, ${r} 0
            Z
        `.trim().replace(/\s+/g, " ");
    path.setAttribute("d", pathData);
    path.setAttribute("fill", this.getStateColors().background);
    path.style.transition = "fill 0.3s ease";
    return path;
  }
  renderFoldedCorner() {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "node-fold-triangle");
    const w = PROCESS_FLOW_LAYOUT.node.width;
    const foldSize = 12;
    const pathData = `
            M ${w - foldSize} 0
            L ${w - foldSize} ${foldSize}
            L ${w} ${foldSize}
            Z
        `.trim().replace(/\s+/g, " ");
    path.setAttribute("d", pathData);
    const colors = this.getStateColors();
    path.setAttribute("fill", this.getLighterShade(colors.background));
    path.setAttribute("stroke", colors.border);
    path.setAttribute("stroke-width", "1");
    path.style.transition = "fill 0.3s ease";
    path.style.filter = "drop-shadow(1px 1px 1px rgba(0, 0, 0, 0.15))";
    return path;
  }
  getLighterShade(hexColor) {
    const hex = hexColor.replace("#", "");
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    const factor = 0.3;
    const newR = Math.min(255, Math.round(r + (255 - r) * factor));
    const newG = Math.min(255, Math.round(g + (255 - g) * factor));
    const newB = Math.min(255, Math.round(b + (255 - b) * factor));
    const toHex = (n) => {
      const hex2 = n.toString(16);
      return hex2.length === 1 ? "0" + hex2 : hex2;
    };
    return "#" + toHex(newR) + toHex(newG) + toHex(newB);
  }
  createBorderPath() {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "node-border");
    const w = PROCESS_FLOW_LAYOUT.node.width;
    const h = PROCESS_FLOW_LAYOUT.node.height;
    const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
    const foldSize = 12;
    let pathData;
    if (this.foldedCorners) {
      pathData = `
                M ${r} 0
                L ${w - foldSize} 0
                L ${w} ${foldSize}
                L ${w} ${h - r}
                Q ${w} ${h}, ${w - r} ${h}
                L ${r} ${h}
                Q 0 ${h}, 0 ${h - r}
                L 0 ${r}
                Q 0 0, ${r} 0
                Z
            `.trim().replace(/\s+/g, " ");
    } else {
      pathData = `
                M ${r} 0
                L ${w - r} 0
                Q ${w} 0, ${w} ${r}
                L ${w} ${h - r}
                Q ${w} ${h}, ${w - r} ${h}
                L ${r} ${h}
                Q 0 ${h}, 0 ${h - r}
                L 0 ${r}
                Q 0 0, ${r} 0
                Z
            `.trim().replace(/\s+/g, " ");
    }
    path.setAttribute("d", pathData);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", this.getStateColors().border);
    path.setAttribute("stroke-width", PROCESS_FLOW_LAYOUT.node.borderWidth.toString());
    path.style.transition = "stroke 0.3s ease";
    return path;
  }
  createIcon() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("class", "node-icon");
    text.setAttribute("x", PROCESS_FLOW_LAYOUT.node.padding.toString());
    text.setAttribute("y", (PROCESS_FLOW_LAYOUT.node.padding + 18).toString());
    text.setAttribute("font-size", PROCESS_FLOW_LAYOUT.node.iconSize.toString());
    text.setAttribute("fill", this.getStateColors().text);
    text.textContent = this.getIconForState();
    return text;
  }
  createTitle() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("class", "node-title");
    const xOffset = this.icon ? 40 : PROCESS_FLOW_LAYOUT.node.padding;
    text.setAttribute("x", xOffset.toString());
    text.setAttribute("y", (PROCESS_FLOW_LAYOUT.node.padding + 16).toString());
    text.setAttribute("font-size", PROCESS_FLOW_LAYOUT.node.titleFontSize.toString());
    text.setAttribute("font-weight", "bold");
    text.setAttribute("fill", this.getStateColors().text);
    text.textContent = this.truncateText(this.title, 18);
    return text;
  }
  createStateText() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("class", "node-state-text");
    text.setAttribute("x", PROCESS_FLOW_LAYOUT.node.padding.toString());
    text.setAttribute("y", (PROCESS_FLOW_LAYOUT.node.padding + 36).toString());
    text.setAttribute("font-size", PROCESS_FLOW_LAYOUT.node.textFontSize.toString());
    text.setAttribute("fill", this.getStateColors().text);
    text.setAttribute("opacity", "0.9");
    text.textContent = this.getStateLabel();
    return text;
  }
  createDetailText(content, index) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("class", "node-detail-text");
    text.setAttribute("x", PROCESS_FLOW_LAYOUT.node.padding.toString());
    text.setAttribute("y", (PROCESS_FLOW_LAYOUT.node.padding + 52 + index * 14).toString());
    text.setAttribute("font-size", (PROCESS_FLOW_LAYOUT.node.textFontSize - 1).toString());
    text.setAttribute("fill", this.getStateColors().text);
    text.setAttribute("opacity", "0.8");
    text.textContent = this.truncateText(content, 22);
    return text;
  }
  renderStack(container) {
    const w = PROCESS_FLOW_LAYOUT.node.width;
    const h = PROCESS_FLOW_LAYOUT.node.height;
    const r = PROCESS_FLOW_LAYOUT.node.cornerRadius;
    const foldSize = 12;
    const stackLayers = Math.min(this.aggregatedCount - 1, 3);
    const stackOffset = 4;
    for (let i = stackLayers;i >= 1; i--) {
      const offsetX = i * stackOffset;
      const offsetY = i * stackOffset;
      const stackPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
      stackPath.setAttribute("class", "aggregate-stack");
      const pathData = `
                M ${r + offsetX} ${offsetY}
                L ${w - foldSize + offsetX} ${offsetY}
                L ${w + offsetX} ${foldSize + offsetY}
                L ${w + offsetX} ${h - r + offsetY}
                Q ${w + offsetX} ${h + offsetY}, ${w - r + offsetX} ${h + offsetY}
                L ${r + offsetX} ${h + offsetY}
                Q ${offsetX} ${h + offsetY}, ${offsetX} ${h - r + offsetY}
                L ${offsetX} ${r + offsetY}
                Q ${offsetX} ${offsetY}, ${r + offsetX} ${offsetY}
                Z
            `.trim().replace(/\s+/g, " ");
      stackPath.setAttribute("d", pathData);
      const colors = this.getStateColors();
      stackPath.setAttribute("fill", colors.background);
      stackPath.setAttribute("stroke", colors.border);
      stackPath.setAttribute("stroke-width", "1");
      stackPath.setAttribute("opacity", (0.3 + (stackLayers - i) * 0.15).toString());
      stackPath.style.transition = "opacity 0.3s ease";
      container.appendChild(stackPath);
      this.stackElements.push(stackPath);
    }
  }
  renderCounter(container) {
    const w = PROCESS_FLOW_LAYOUT.node.width;
    const badgeRadius = 12;
    const badgeX = w - 6;
    const badgeY = -6;
    this.counterBadge = document.createElementNS("http://www.w3.org/2000/svg", "g");
    this.counterBadge.setAttribute("class", "aggregate-counter");
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", badgeX.toString());
    circle.setAttribute("cy", badgeY.toString());
    circle.setAttribute("r", badgeRadius.toString());
    circle.setAttribute("fill", "#0a6ed1");
    circle.setAttribute("stroke", "#ffffff");
    circle.setAttribute("stroke-width", "2");
    this.counterBadge.appendChild(circle);
    const countText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    countText.setAttribute("x", badgeX.toString());
    countText.setAttribute("y", (badgeY + 1).toString());
    countText.setAttribute("text-anchor", "middle");
    countText.setAttribute("dominant-baseline", "middle");
    countText.setAttribute("font-size", "10");
    countText.setAttribute("font-weight", "bold");
    countText.setAttribute("fill", "#ffffff");
    countText.setAttribute("font-family", '"72", "72full", Arial, Helvetica, sans-serif');
    const displayCount = this.aggregatedCount > 99 ? "99+" : this.aggregatedCount.toString();
    countText.textContent = displayCount;
    this.counterBadge.appendChild(countText);
    container.appendChild(this.counterBadge);
  }
  expandAggregate() {
    if (!this.isAggregated || this.isExpanded) {
      return [];
    }
    this.isExpanded = true;
    this.element.classList.add("aggregate-expanded");
    for (const stackEl of this.stackElements) {
      stackEl.style.opacity = "0";
    }
    if (this.counterBadge) {
      this.counterBadge.style.opacity = "0";
    }
    return this.aggregatedItems;
  }
  collapseAggregate() {
    if (!this.isAggregated || !this.isExpanded) {
      return;
    }
    this.isExpanded = false;
    this.element.classList.remove("aggregate-expanded");
    const stackLayers = Math.min(this.aggregatedCount - 1, 3);
    for (let i = 0;i < this.stackElements.length; i++) {
      const stackEl = this.stackElements[i];
      stackEl.style.opacity = (0.3 + (stackLayers - i - 1) * 0.15).toString();
    }
    if (this.counterBadge) {
      this.counterBadge.style.opacity = "1";
    }
  }
  isAggregateExpanded() {
    return this.isExpanded;
  }
  setPosition(x, y) {
    this.x = x;
    this.y = y;
    this.element.setAttribute("transform", `translate(${x}, ${y})`);
  }
  getWidth() {
    return this.currentZoomConfig.nodeWidth;
  }
  getHeight() {
    return this.currentZoomConfig.nodeHeight;
  }
  setZoomLevel(level) {
    this.currentZoomLevel = level;
    this.currentZoomConfig = ZOOM_LEVEL_CONFIG[level];
    const config = this.currentZoomConfig;
    this.updateNodeShape(config.nodeWidth, config.nodeHeight);
    this.updateElementVisibility(config);
    if (this.statusIcon) {
      this.statusIcon.setAttribute("x", (config.nodeWidth / 2).toString());
      this.statusIcon.setAttribute("y", (config.nodeHeight / 2 + 4).toString());
    }
  }
  getZoomLevel() {
    return this.currentZoomLevel;
  }
  updateNodeShape(width, height) {
    const r = Math.min(PROCESS_FLOW_LAYOUT.node.cornerRadius, width / 10);
    const foldSize = Math.min(12, width / 13);
    let pathData;
    if (this.foldedCorners) {
      pathData = `
                M ${r} 0
                L ${width - foldSize} 0
                L ${width} ${foldSize}
                L ${width} ${height - r}
                Q ${width} ${height}, ${width - r} ${height}
                L ${r} ${height}
                Q 0 ${height}, 0 ${height - r}
                L 0 ${r}
                Q 0 0, ${r} 0
                Z
            `.trim().replace(/\s+/g, " ");
    } else {
      pathData = `
                M ${r} 0
                L ${width - r} 0
                Q ${width} 0, ${width} ${r}
                L ${width} ${height - r}
                Q ${width} ${height}, ${width - r} ${height}
                L ${r} ${height}
                Q 0 ${height}, 0 ${height - r}
                L 0 ${r}
                Q 0 0, ${r} 0
                Z
            `.trim().replace(/\s+/g, " ");
    }
    this.background.setAttribute("d", pathData);
    this.borderPath.setAttribute("d", pathData);
    if (this.foldTriangle && this.foldedCorners) {
      const foldPathData = `
                M ${width - foldSize} 0
                L ${width - foldSize} ${foldSize}
                L ${width} ${foldSize}
                Z
            `.trim().replace(/\s+/g, " ");
      this.foldTriangle.setAttribute("d", foldPathData);
    }
  }
  updateElementVisibility(config) {
    this.titleText.style.display = config.showHeader ? "block" : "none";
    this.stateText.style.display = config.showStatus && config.showHeader ? "block" : "none";
    if (this.icon) {
      this.icon.style.display = config.showHeader ? "block" : "none";
    }
    for (let i = 0;i < this.detailTexts.length; i++) {
      if (i === 0) {
        this.detailTexts[i].style.display = config.showAttr1 ? "block" : "none";
      } else if (i === 1) {
        this.detailTexts[i].style.display = config.showAttr2 ? "block" : "none";
      } else {
        this.detailTexts[i].style.display = config.showAttr2 ? "block" : "none";
      }
    }
    if (this.statusIcon) {
      this.statusIcon.style.display = !config.showHeader && config.showStatus ? "block" : "none";
    }
  }
  setState(state) {
    this.state = state;
    this.updateColors();
    this.stateText.textContent = this.getStateLabel();
    if (this.statusIcon) {
      this.statusIcon.textContent = this.getIconForState();
    }
  }
  setDisplayState(state) {
    this.displayState = state;
    switch (state) {
      case "Highlighted" /* Highlighted */:
        this.element.style.filter = "drop-shadow(0 4px 8px rgba(10,110,209,0.4))";
        this.element.style.transform = "scale(1.05)";
        break;
      case "Dimmed" /* Dimmed */:
        this.element.style.opacity = "0.4";
        break;
      case "Selected" /* Selected */:
        this.element.style.filter = "drop-shadow(0 0 12px rgba(10,110,209,0.6))";
        this.borderPath.setAttribute("stroke-width", "3");
        break;
      case "Regular" /* Regular */:
      default:
        this.element.style.filter = "";
        this.element.style.transform = "scale(1.0)";
        this.element.style.opacity = "1.0";
        this.borderPath.setAttribute("stroke-width", PROCESS_FLOW_LAYOUT.node.borderWidth.toString());
    }
  }
  updateColors() {
    const colors = this.getStateColors();
    this.background.setAttribute("fill", colors.background);
    this.borderPath.setAttribute("stroke", colors.border);
    this.titleText.setAttribute("fill", colors.text);
    this.stateText.setAttribute("fill", colors.text);
    for (const text of this.detailTexts) {
      text.setAttribute("fill", colors.text);
    }
    if (this.icon) {
      this.icon.setAttribute("fill", colors.text);
    }
    if (this.statusIcon) {
      this.statusIcon.setAttribute("fill", colors.text);
    }
  }
  getStateColors() {
    switch (this.state) {
      case "Positive" /* Positive */:
        return PROCESS_FLOW_COLORS.positive;
      case "Negative" /* Negative */:
        return PROCESS_FLOW_COLORS.negative;
      case "Critical" /* Critical */:
        return PROCESS_FLOW_COLORS.critical;
      case "Planned" /* Planned */:
        return PROCESS_FLOW_COLORS.planned;
      case "PlannedNegative" /* PlannedNegative */:
        return PROCESS_FLOW_COLORS.plannedNegative;
      case "Neutral" /* Neutral */:
      default:
        return PROCESS_FLOW_COLORS.neutral;
    }
  }
  getStateLabel() {
    switch (this.state) {
      case "Positive" /* Positive */:
        return "Completed";
      case "Negative" /* Negative */:
        return "Failed";
      case "Critical" /* Critical */:
        return "Warning";
      case "Planned" /* Planned */:
        return "Planned";
      case "PlannedNegative" /* PlannedNegative */:
        return "Planned (Issue)";
      case "Neutral" /* Neutral */:
        return "In Progress";
      default:
        return "";
    }
  }
  getIconForState() {
    switch (this.state) {
      case "Positive" /* Positive */:
        return "✓";
      case "Negative" /* Negative */:
        return "✗";
      case "Critical" /* Critical */:
        return "⚠";
      case "Neutral" /* Neutral */:
        return "▶";
      default:
        return "";
    }
  }
  needsIcon() {
    return this.state !== "Planned" /* Planned */ && this.state !== "PlannedNegative" /* PlannedNegative */;
  }
  truncateText(text, maxLength) {
    if (text.length <= maxLength)
      return text;
    return text.substring(0, maxLength - 3) + "...";
  }
  setHighlighted(highlighted) {
    if (highlighted) {
      this.element.classList.add("highlighted");
      this.element.classList.remove("dimmed");
      this.element.style.filter = "drop-shadow(0 0 8px rgba(10, 110, 209, 0.6))";
      this.borderPath.setAttribute("stroke-width", "3");
    } else {
      this.element.classList.remove("highlighted");
      this.element.style.filter = "";
      this.borderPath.setAttribute("stroke-width", PROCESS_FLOW_LAYOUT.node.borderWidth.toString());
    }
  }
  setDimmed(dimmed) {
    if (dimmed) {
      this.element.classList.add("dimmed");
      this.element.classList.remove("highlighted");
      this.element.style.opacity = "0.3";
      this.element.style.filter = "grayscale(50%)";
    } else {
      this.element.classList.remove("dimmed");
      this.element.style.opacity = "1";
      this.element.style.filter = "";
    }
  }
  toJSON() {
    const result = {
      id: this.id,
      lane: this.lane,
      title: this.title,
      state: this.state,
      texts: this.texts,
      children: this.children,
      position: this.position,
      foldedCorners: this.foldedCorners
    };
    if (this.isAggregated) {
      result.isAggregated = this.isAggregated;
      result.aggregatedCount = this.aggregatedCount;
      result.aggregatedItems = this.aggregatedItems;
    }
    return result;
  }
  destroy() {
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

// ProcessFlow/ProcessFlowLane.ts
class ProcessFlowLane {
  id;
  label;
  position;
  state;
  element;
  background;
  labelText;
  icon = null;
  y = 0;
  height = PROCESS_FLOW_LAYOUT.node.height;
  constructor(config) {
    this.id = config.id;
    this.label = config.label;
    this.position = config.position;
    this.state = config.state;
    this.element = this.createElement();
  }
  createElement() {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "process-flow-lane");
    g.setAttribute("data-lane-id", this.id);
    this.background = this.createBackground();
    g.appendChild(this.background);
    this.labelText = this.createLabel();
    g.appendChild(this.labelText);
    return g;
  }
  createBackground() {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("class", "lane-background");
    rect.setAttribute("width", PROCESS_FLOW_LAYOUT.spacing.laneHeader.toString());
    rect.setAttribute("height", this.height.toString());
    rect.setAttribute("fill", this.position % 2 === 0 ? PROCESS_FLOW_COLORS.lane.default : PROCESS_FLOW_COLORS.lane.alternate);
    rect.setAttribute("stroke", PROCESS_FLOW_COLORS.border);
    rect.setAttribute("stroke-width", "1");
    return rect;
  }
  createLabel() {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("class", "lane-label");
    text.setAttribute("x", (PROCESS_FLOW_LAYOUT.spacing.laneHeader / 2).toString());
    text.setAttribute("y", (this.height / 2).toString());
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dominant-baseline", "middle");
    text.setAttribute("font-size", "13");
    text.setAttribute("font-weight", "600");
    text.setAttribute("fill", PROCESS_FLOW_COLORS.text.primary);
    text.textContent = this.label;
    return text;
  }
  setPosition(y) {
    this.y = y;
    this.element.setAttribute("transform", `translate(0, ${y})`);
  }
  setHeight(height) {
    this.height = height;
    this.background.setAttribute("height", height.toString());
    this.labelText.setAttribute("y", (height / 2).toString());
  }
  destroy() {
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

// ProcessFlow/ProcessFlowConnection.ts
var LABEL_COLORS = {
  Positive: "#107e3e",
  Negative: "#bb0000",
  Neutral: "#0070f2",
  Critical: "#df6e0c"
};

class ProcessFlowConnection {
  id;
  from;
  to;
  state;
  type;
  element;
  path;
  arrow;
  label = null;
  labelElement = null;
  labelClickCallback = null;
  sourceNode = null;
  targetNode = null;
  constructor(config) {
    this.id = `${config.from}-${config.to}`;
    this.from = config.from;
    this.to = config.to;
    this.state = config.state || "Normal" /* Normal */;
    this.type = config.type || "normal";
    this.element = this.createElement();
  }
  createElement() {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "process-flow-connection");
    g.setAttribute("data-from", this.from);
    g.setAttribute("data-to", this.to);
    this.path = this.createPath();
    g.appendChild(this.path);
    this.arrow = this.createArrow();
    g.appendChild(this.arrow);
    return g;
  }
  createPath() {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "connection-path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", this.getStateColor());
    path.setAttribute("stroke-width", PROCESS_FLOW_LAYOUT.connection.strokeWidth.toString());
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    if (this.type === "planned") {
      path.setAttribute("stroke-dasharray", PROCESS_FLOW_LAYOUT.connection.dashArray);
    }
    path.style.transition = "stroke 0.3s ease";
    return path;
  }
  createArrow() {
    const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    arrow.setAttribute("class", "connection-arrow");
    arrow.setAttribute("fill", this.getStateColor());
    arrow.style.transition = "fill 0.3s ease";
    return arrow;
  }
  setNodes(source, target) {
    this.sourceNode = source;
    this.targetNode = target;
    this.updatePath();
  }
  updatePath() {
    if (!this.sourceNode || !this.targetNode)
      return;
    const sourceX = this.sourceNode.x + this.sourceNode.getWidth();
    const sourceY = this.sourceNode.y + this.sourceNode.getHeight() / 2;
    const targetX = this.targetNode.x;
    const targetY = this.targetNode.y + this.targetNode.getHeight() / 2;
    const pathData = this.createRoundedPath({ x: sourceX, y: sourceY }, { x: targetX, y: targetY });
    this.path.setAttribute("d", pathData);
    this.positionArrow({ x: targetX, y: targetY });
  }
  createRoundedPath(start, end) {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const cornerRadius = PROCESS_FLOW_LAYOUT.connection.cornerRadius;
    if (Math.abs(dy) < 2) {
      return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
    }
    const midX = start.x + dx / 2;
    if (dy > 0) {
      const cp1 = Math.min(cornerRadius, Math.abs(dx) / 2);
      const cp2 = Math.min(cornerRadius, Math.abs(dy) / 2);
      return `
                M ${start.x} ${start.y}
                L ${midX - cp1} ${start.y}
                Q ${midX} ${start.y}, ${midX} ${start.y + cp2}
                L ${midX} ${end.y - cp2}
                Q ${midX} ${end.y}, ${midX + cp1} ${end.y}
                L ${end.x} ${end.y}
            `.trim().replace(/\s+/g, " ");
    } else {
      const cp1 = Math.min(cornerRadius, Math.abs(dx) / 2);
      const cp2 = Math.min(cornerRadius, Math.abs(dy) / 2);
      return `
                M ${start.x} ${start.y}
                L ${midX - cp1} ${start.y}
                Q ${midX} ${start.y}, ${midX} ${start.y - cp2}
                L ${midX} ${end.y + cp2}
                Q ${midX} ${end.y}, ${midX + cp1} ${end.y}
                L ${end.x} ${end.y}
            `.trim().replace(/\s+/g, " ");
    }
  }
  positionArrow(point) {
    const arrowSize = PROCESS_FLOW_LAYOUT.connection.arrowSize;
    const tip = point;
    const base1 = { x: tip.x - arrowSize, y: tip.y - arrowSize / 2 };
    const base2 = { x: tip.x - arrowSize, y: tip.y + arrowSize / 2 };
    const points = `${tip.x},${tip.y} ${base1.x},${base1.y} ${base2.x},${base2.y}`;
    this.arrow.setAttribute("points", points);
  }
  setState(state) {
    this.state = state;
    const color = this.getStateColor();
    this.path.setAttribute("stroke", color);
    this.arrow.setAttribute("fill", color);
  }
  getStateColor() {
    switch (this.state) {
      case "Highlighted" /* Highlighted */:
        return PROCESS_FLOW_COLORS.connection.highlighted;
      case "Dimmed" /* Dimmed */:
        return PROCESS_FLOW_COLORS.connection.dimmed;
      case "Normal" /* Normal */:
      default:
        return PROCESS_FLOW_COLORS.connection.normal;
    }
  }
  animate() {
    this.path.animate([
      { strokeWidth: PROCESS_FLOW_LAYOUT.connection.strokeWidth },
      { strokeWidth: PROCESS_FLOW_LAYOUT.connection.strokeWidth + 2 },
      { strokeWidth: PROCESS_FLOW_LAYOUT.connection.strokeWidth }
    ], {
      duration: 800,
      easing: PROCESS_FLOW_LAYOUT.animation.easing
    });
  }
  setLabel(label) {
    this.label = label;
    if (label) {
      this.renderLabel();
    } else {
      this.removeLabel();
    }
  }
  getLabel() {
    return this.label;
  }
  onLabelClick(callback) {
    this.labelClickCallback = callback;
  }
  renderLabel() {
    if (!this.label)
      return;
    this.removeLabel();
    this.labelElement = document.createElementNS("http://www.w3.org/2000/svg", "g");
    this.labelElement.setAttribute("class", "connection-label");
    this.labelElement.style.cursor = "pointer";
    const text = this.label.text;
    const hasIcon = !!this.label.icon;
    const fontSize = 11;
    const paddingX = 8;
    const paddingY = 4;
    const iconWidth = hasIcon ? 14 : 0;
    const textWidth = text.length * 6;
    const labelWidth = textWidth + iconWidth + paddingX * 2 + (hasIcon ? 4 : 0);
    const labelHeight = fontSize + paddingY * 2;
    const cornerRadius = labelHeight / 2;
    const bgColor = LABEL_COLORS[this.label.state];
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", (-labelWidth / 2).toString());
    rect.setAttribute("y", (-labelHeight / 2).toString());
    rect.setAttribute("width", labelWidth.toString());
    rect.setAttribute("height", labelHeight.toString());
    rect.setAttribute("rx", cornerRadius.toString());
    rect.setAttribute("ry", cornerRadius.toString());
    rect.setAttribute("fill", bgColor);
    rect.style.filter = "drop-shadow(0 1px 2px rgba(0,0,0,0.2))";
    this.labelElement.appendChild(rect);
    let iconOffset = 0;
    if (hasIcon && this.label.icon) {
      iconOffset = -textWidth / 2 - 2;
      const iconText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      iconText.setAttribute("x", iconOffset.toString());
      iconText.setAttribute("y", "4");
      iconText.setAttribute("font-family", "SAP-icons");
      iconText.setAttribute("font-size", "12");
      iconText.setAttribute("fill", "white");
      iconText.setAttribute("text-anchor", "middle");
      iconText.textContent = this.label.icon;
      this.labelElement.appendChild(iconText);
    }
    const textEl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const textX = hasIcon ? iconOffset + iconWidth + 2 : 0;
    textEl.setAttribute("x", textX.toString());
    textEl.setAttribute("y", "4");
    textEl.setAttribute("font-family", '"72", "72full", Arial, Helvetica, sans-serif');
    textEl.setAttribute("font-size", fontSize.toString());
    textEl.setAttribute("font-weight", "600");
    textEl.setAttribute("fill", "white");
    textEl.setAttribute("text-anchor", hasIcon ? "start" : "middle");
    textEl.textContent = text;
    this.labelElement.appendChild(textEl);
    this.labelElement.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this.labelClickCallback) {
        this.labelClickCallback(this);
      }
    });
    this.labelElement.addEventListener("mouseenter", () => {
      rect.style.filter = "drop-shadow(0 2px 4px rgba(0,0,0,0.3))";
      rect.setAttribute("opacity", "0.9");
    });
    this.labelElement.addEventListener("mouseleave", () => {
      rect.style.filter = "drop-shadow(0 1px 2px rgba(0,0,0,0.2))";
      rect.setAttribute("opacity", "1");
    });
    this.element.appendChild(this.labelElement);
    this.positionLabel();
  }
  positionLabel() {
    if (!this.labelElement || !this.sourceNode || !this.targetNode)
      return;
    const sourceX = this.sourceNode.x + this.sourceNode.getWidth();
    const sourceY = this.sourceNode.y + this.sourceNode.getHeight() / 2;
    const targetX = this.targetNode.x;
    const targetY = this.targetNode.y + this.targetNode.getHeight() / 2;
    const midX = sourceX + (targetX - sourceX) / 2;
    const midY = (sourceY + targetY) / 2;
    this.labelElement.setAttribute("transform", `translate(${midX}, ${midY})`);
  }
  removeLabel() {
    if (this.labelElement && this.labelElement.parentNode) {
      this.labelElement.parentNode.removeChild(this.labelElement);
      this.labelElement = null;
    }
  }
  setHighlighted(highlighted) {
    if (highlighted) {
      this.element.classList.add("highlighted");
      this.element.classList.remove("dimmed");
      this.path.setAttribute("stroke", PROCESS_FLOW_COLORS.connection.highlighted);
      this.path.setAttribute("stroke-width", "3");
      this.arrow.setAttribute("fill", PROCESS_FLOW_COLORS.connection.highlighted);
      this.element.style.filter = "drop-shadow(0 0 4px rgba(10, 110, 209, 0.5))";
    } else {
      this.element.classList.remove("highlighted");
      this.path.setAttribute("stroke", this.getStateColor());
      this.path.setAttribute("stroke-width", PROCESS_FLOW_LAYOUT.connection.strokeWidth.toString());
      this.arrow.setAttribute("fill", this.getStateColor());
      this.element.style.filter = "";
    }
  }
  setDimmed(dimmed) {
    if (dimmed) {
      this.element.classList.add("dimmed");
      this.element.classList.remove("highlighted");
      this.path.setAttribute("stroke", PROCESS_FLOW_COLORS.connection.dimmed);
      this.arrow.setAttribute("fill", PROCESS_FLOW_COLORS.connection.dimmed);
      this.element.style.opacity = "0.3";
    } else {
      this.element.classList.remove("dimmed");
      this.path.setAttribute("stroke", this.getStateColor());
      this.arrow.setAttribute("fill", this.getStateColor());
      this.element.style.opacity = "1";
    }
  }
  destroy() {
    this.removeLabel();
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

// ProcessFlow/ProcessFlow.ts
class ProcessFlow {
  container;
  svg;
  contentGroup;
  lanesGroup;
  connectionsGroup;
  nodesGroup;
  nodes = new Map;
  lanes = new Map;
  connections = new Map;
  config;
  lanePositions = new Map;
  columnPositions = [];
  selectedNodeId = null;
  hoveredNodeId = null;
  eventListeners = new Map;
  currentZoomLevel = "Two" /* Two */;
  isAutoZoom = true;
  resizeObserver = null;
  leftOverflowIndicator = null;
  rightOverflowIndicator = null;
  constructor(container, config) {
    if (typeof container === "string") {
      const el = document.querySelector(container);
      if (!el)
        throw new Error(`Container ${container} not found`);
      this.container = el;
    } else {
      this.container = container;
    }
    this.config = { ...DEFAULT_PROCESS_FLOW_CONFIG, ...config };
    this.init();
  }
  init() {
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("class", "process-flow-svg");
    this.svg.style.width = "100%";
    this.svg.style.height = "100%";
    this.svg.style.fontFamily = '"72", "72full", Arial, Helvetica, sans-serif';
    this.contentGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    this.contentGroup.setAttribute("class", "process-flow-content");
    this.svg.appendChild(this.contentGroup);
    this.lanesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    this.lanesGroup.setAttribute("class", "process-flow-lanes");
    this.contentGroup.appendChild(this.lanesGroup);
    this.connectionsGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    this.connectionsGroup.setAttribute("class", "process-flow-connections");
    this.contentGroup.appendChild(this.connectionsGroup);
    this.nodesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    this.nodesGroup.setAttribute("class", "process-flow-nodes");
    this.contentGroup.appendChild(this.nodesGroup);
    this.container.appendChild(this.svg);
    this.createOverflowIndicators();
    this.setupInteractions();
    this.setupResizeObserver();
    if (this.config.zoomLevel) {
      this.isAutoZoom = false;
      this.setZoomLevel(this.config.zoomLevel);
    } else {
      this.isAutoZoom = true;
      this.updateAutoZoomLevel();
    }
  }
  createOverflowIndicators() {
    this.leftOverflowIndicator = document.createElement("div");
    this.leftOverflowIndicator.className = "process-flow-overflow-indicator process-flow-overflow-left";
    this.leftOverflowIndicator.style.cssText = `
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(10, 110, 209, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 0 4px 4px 0;
            font-family: "72", "72full", Arial, Helvetica, sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: none;
            z-index: 100;
            box-shadow: 2px 0 8px rgba(0,0,0,0.15);
            transition: background 0.2s ease;
        `;
    this.leftOverflowIndicator.addEventListener("mouseenter", () => {
      this.leftOverflowIndicator.style.background = "rgba(10, 110, 209, 1)";
    });
    this.leftOverflowIndicator.addEventListener("mouseleave", () => {
      this.leftOverflowIndicator.style.background = "rgba(10, 110, 209, 0.9)";
    });
    this.container.appendChild(this.leftOverflowIndicator);
    this.rightOverflowIndicator = document.createElement("div");
    this.rightOverflowIndicator.className = "process-flow-overflow-indicator process-flow-overflow-right";
    this.rightOverflowIndicator.style.cssText = `
            position: absolute;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(10, 110, 209, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 4px 0 0 4px;
            font-family: "72", "72full", Arial, Helvetica, sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: none;
            z-index: 100;
            box-shadow: -2px 0 8px rgba(0,0,0,0.15);
            transition: background 0.2s ease;
        `;
    this.rightOverflowIndicator.addEventListener("mouseenter", () => {
      this.rightOverflowIndicator.style.background = "rgba(10, 110, 209, 1)";
    });
    this.rightOverflowIndicator.addEventListener("mouseleave", () => {
      this.rightOverflowIndicator.style.background = "rgba(10, 110, 209, 0.9)";
    });
    this.container.appendChild(this.rightOverflowIndicator);
    if (getComputedStyle(this.container).position === "static") {
      this.container.style.position = "relative";
    }
  }
  setupResizeObserver() {
    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target === this.container) {
          if (this.isAutoZoom) {
            this.updateAutoZoomLevel();
          }
          this.updateOverflowIndicators();
        }
      }
    });
    this.resizeObserver.observe(this.container);
  }
  detectZoomLevel() {
    const width = this.container.clientWidth;
    if (width >= 1024)
      return "Two" /* Two */;
    if (width >= 600)
      return "Three" /* Three */;
    return "Four" /* Four */;
  }
  updateAutoZoomLevel() {
    const detectedLevel = this.detectZoomLevel();
    if (detectedLevel !== this.currentZoomLevel) {
      this.applyZoomLevel(detectedLevel);
    }
  }
  setupInteractions() {
    this.svg.addEventListener("click", (e) => {
      const target = e.target;
      const nodeEl = target.closest(".process-flow-node");
      if (nodeEl) {
        const nodeId = nodeEl.getAttribute("data-node-id");
        if (nodeId) {
          this.handleNodeClick(nodeId, e);
        }
      }
    });
    this.svg.addEventListener("mouseover", (e) => {
      const target = e.target;
      const nodeEl = target.closest(".process-flow-node");
      if (nodeEl) {
        const nodeId = nodeEl.getAttribute("data-node-id");
        if (nodeId) {
          this.handleNodeHover(nodeId);
        }
      }
    });
    this.svg.addEventListener("mouseout", (e) => {
      const target = e.target;
      const nodeEl = target.closest(".process-flow-node");
      if (nodeEl) {
        this.handleNodeLeave();
      }
    });
    if (this.config.wheelZoomable) {
      this.svg.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.handleWheel(e);
      });
    }
  }
  setLanes(lanesConfig) {
    this.lanes.clear();
    this.lanePositions.clear();
    lanesConfig.sort((a, b) => a.position - b.position);
    for (const config of lanesConfig) {
      const lane = new ProcessFlowLane(config);
      this.lanes.set(config.id, lane);
      this.lanesGroup.appendChild(lane.element);
    }
    this.updateLayout();
  }
  setNodes(nodesConfig) {
    this.nodes.clear();
    for (const config of nodesConfig) {
      const node = new ProcessFlowNode(config);
      this.nodes.set(config.id, node);
      this.nodesGroup.appendChild(node.element);
    }
    this.updateLayout();
  }
  setConnections(connectionsConfig) {
    this.connections.clear();
    for (const config of connectionsConfig) {
      const connection = new ProcessFlowConnection(config);
      const sourceNode = this.nodes.get(config.from);
      const targetNode = this.nodes.get(config.to);
      if (sourceNode && targetNode) {
        connection.setNodes(sourceNode, targetNode);
        this.connections.set(connection.id, connection);
        this.connectionsGroup.appendChild(connection.element);
      }
    }
  }
  loadData(data) {
    this.setLanes(data.lanes);
    this.setNodes(data.nodes);
    this.setConnections(data.connections);
  }
  updateLayout() {
    this.calculateLanePositions();
    this.calculateColumnPositions();
    this.positionNodes();
    this.updateConnections();
    this.updateSVGSize();
  }
  calculateLanePositions() {
    const lanesArray = Array.from(this.lanes.values());
    lanesArray.sort((a, b) => a.position - b.position);
    let y = PROCESS_FLOW_LAYOUT.spacing.topMargin;
    for (const lane of lanesArray) {
      this.lanePositions.set(lane.id, y);
      lane.setPosition(y);
      y += PROCESS_FLOW_LAYOUT.node.height + PROCESS_FLOW_LAYOUT.spacing.vertical;
    }
  }
  calculateColumnPositions() {
    const columns = new Map;
    for (const node of this.nodes.values()) {
      const pos = node.position;
      if (!columns.has(pos)) {
        columns.set(pos, []);
      }
      columns.get(pos).push(node.toJSON());
    }
    this.columnPositions = [];
    let x = PROCESS_FLOW_LAYOUT.spacing.laneHeader + PROCESS_FLOW_LAYOUT.spacing.leftMargin;
    const maxColumn = Math.max(...Array.from(columns.keys()));
    for (let i = 0;i <= maxColumn; i++) {
      this.columnPositions.push(x);
      x += PROCESS_FLOW_LAYOUT.node.width + PROCESS_FLOW_LAYOUT.spacing.horizontal;
    }
  }
  positionNodes() {
    for (const node of this.nodes.values()) {
      const laneY = this.lanePositions.get(node.lane);
      const columnX = this.columnPositions[node.position];
      if (laneY !== undefined && columnX !== undefined) {
        node.setPosition(columnX, laneY);
      }
    }
  }
  updateConnections() {
    for (const connection of this.connections.values()) {
      connection.updatePath();
    }
  }
  updateSVGSize() {
    const maxColumn = Math.max(...Array.from(this.nodes.values()).map((n) => n.position));
    const width = this.columnPositions[maxColumn] + PROCESS_FLOW_LAYOUT.node.width + PROCESS_FLOW_LAYOUT.spacing.rightMargin;
    const maxLane = Array.from(this.lanes.values()).length;
    const height = maxLane * (PROCESS_FLOW_LAYOUT.node.height + PROCESS_FLOW_LAYOUT.spacing.vertical) + PROCESS_FLOW_LAYOUT.spacing.topMargin + PROCESS_FLOW_LAYOUT.spacing.bottomMargin;
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  }
  handleNodeClick(nodeId, event) {
    const node = this.nodes.get(nodeId);
    if (!node)
      return;
    if (this.selectedNodeId === nodeId) {
      this.selectedNodeId = null;
      this.clearSelection();
    } else {
      this.selectedNodeId = nodeId;
      this.highlightNode(nodeId);
    }
    this.emit("nodeClick", { type: "click", node: node.toJSON(), originalEvent: event });
  }
  handleNodeHover(nodeId) {
    if (this.hoveredNodeId === nodeId)
      return;
    this.hoveredNodeId = nodeId;
    const node = this.nodes.get(nodeId);
    if (node) {
      node.setDisplayState("Highlighted" /* Highlighted */);
    }
  }
  handleNodeLeave() {
    if (this.hoveredNodeId) {
      const node = this.nodes.get(this.hoveredNodeId);
      if (node && this.selectedNodeId !== this.hoveredNodeId) {
        node.setDisplayState("Regular" /* Regular */);
      }
      this.hoveredNodeId = null;
    }
  }
  handleWheel(event) {
    const delta = event.deltaY > 0 ? -1 : 1;
    this.adjustZoom(delta);
  }
  highlightNode(nodeId) {
    const selectedNode = this.nodes.get(nodeId);
    if (!selectedNode)
      return;
    const connectedNodeIds = this.getConnectedNodes(nodeId);
    for (const [id, node] of this.nodes) {
      if (id === nodeId) {
        node.setDisplayState("Selected" /* Selected */);
      } else if (connectedNodeIds.has(id)) {
        node.setDisplayState("Highlighted" /* Highlighted */);
      } else {
        node.setDisplayState("Dimmed" /* Dimmed */);
      }
    }
  }
  clearSelection() {
    for (const node of this.nodes.values()) {
      node.setDisplayState("Regular" /* Regular */);
    }
  }
  getConnectedNodes(nodeId) {
    const connected = new Set;
    for (const connection of this.connections.values()) {
      if (connection.from === nodeId) {
        connected.add(connection.to);
      }
      if (connection.to === nodeId) {
        connected.add(connection.from);
      }
    }
    return connected;
  }
  highlightPath(nodeIds) {
    const pathNodeSet = new Set(nodeIds);
    this.dimNonPathElements(nodeIds);
    for (const nodeId of nodeIds) {
      const node = this.nodes.get(nodeId);
      if (node) {
        node.setHighlighted(true);
      }
    }
    for (const connection of this.connections.values()) {
      if (pathNodeSet.has(connection.from) && pathNodeSet.has(connection.to)) {
        connection.setHighlighted(true);
      }
    }
  }
  clearHighlight() {
    for (const node of this.nodes.values()) {
      node.setHighlighted(false);
      node.setDimmed(false);
    }
    for (const connection of this.connections.values()) {
      connection.setHighlighted(false);
      connection.setDimmed(false);
    }
  }
  dimNonPathElements(pathNodeIds) {
    const pathNodeSet = new Set(pathNodeIds);
    for (const [nodeId, node] of this.nodes) {
      if (!pathNodeSet.has(nodeId)) {
        node.setDimmed(true);
      } else {
        node.setDimmed(false);
      }
    }
    for (const connection of this.connections.values()) {
      const isPathConnection = pathNodeSet.has(connection.from) && pathNodeSet.has(connection.to);
      if (!isPathConnection) {
        connection.setDimmed(true);
      } else {
        connection.setDimmed(false);
      }
    }
  }
  setZoomLevel(level) {
    this.isAutoZoom = false;
    this.applyZoomLevel(level);
  }
  getZoomLevel() {
    return this.currentZoomLevel;
  }
  enableAutoZoom() {
    this.isAutoZoom = true;
    this.updateAutoZoomLevel();
  }
  applyZoomLevel(level) {
    this.currentZoomLevel = level;
    this.config.zoomLevel = level;
    const zoomConfig = ZOOM_LEVEL_CONFIG[level];
    for (const node of this.nodes.values()) {
      node.setZoomLevel(level);
    }
    this.contentGroup.setAttribute("transform", `scale(${zoomConfig.scale})`);
    this.contentGroup.setAttribute("data-zoom", level);
    this.updateLayout();
    this.updateOverflowIndicators();
    this.emit("zoomChange", { level, config: zoomConfig });
  }
  adjustZoom(delta) {
    const levels = [
      "Four" /* Four */,
      "Three" /* Three */,
      "Two" /* Two */,
      "One" /* One */
    ];
    const currentIndex = levels.indexOf(this.currentZoomLevel);
    const newIndex = Math.max(0, Math.min(levels.length - 1, currentIndex + delta));
    if (levels[newIndex] !== this.currentZoomLevel) {
      this.isAutoZoom = false;
      this.applyZoomLevel(levels[newIndex]);
    }
  }
  updateOverflowIndicators() {
    if (!this.leftOverflowIndicator || !this.rightOverflowIndicator)
      return;
    const containerRect = this.container.getBoundingClientRect();
    const zoomConfig = ZOOM_LEVEL_CONFIG[this.currentZoomLevel];
    const scale = zoomConfig.scale;
    const visibleLeft = 0;
    const visibleRight = containerRect.width / scale;
    let hiddenLeft = 0;
    let hiddenRight = 0;
    for (const node of this.nodes.values()) {
      const nodeRight = node.x + node.getWidth();
      const nodeLeft = node.x;
      if (nodeRight < visibleLeft) {
        hiddenLeft++;
      } else if (nodeLeft > visibleRight) {
        hiddenRight++;
      }
    }
    if (hiddenLeft > 0) {
      this.leftOverflowIndicator.textContent = `< ${hiddenLeft}`;
      this.leftOverflowIndicator.style.display = "block";
    } else {
      this.leftOverflowIndicator.style.display = "none";
    }
    if (hiddenRight > 0) {
      this.rightOverflowIndicator.textContent = `${hiddenRight} >`;
      this.rightOverflowIndicator.style.display = "block";
    } else {
      this.rightOverflowIndicator.style.display = "none";
    }
  }
  on(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(callback);
  }
  off(event, callback) {
    if (!this.eventListeners.has(event))
      return;
    if (callback) {
      const callbacks = this.eventListeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    } else {
      this.eventListeners.delete(event);
    }
  }
  emit(event, data) {
    if (!this.eventListeners.has(event))
      return;
    for (const callback of this.eventListeners.get(event)) {
      callback(data);
    }
  }
  getNode(nodeId) {
    return this.nodes.get(nodeId);
  }
  selectNode(nodeId) {
    if (nodeId === null) {
      this.selectedNodeId = null;
      this.clearSelection();
    } else {
      this.selectedNodeId = nodeId;
      this.highlightNode(nodeId);
    }
  }
  exportData() {
    return {
      lanes: Array.from(this.lanes.values()).map((l) => ({ id: l.id, label: l.label, position: l.position })),
      nodes: Array.from(this.nodes.values()).map((n) => n.toJSON()),
      connections: Array.from(this.connections.values()).map((c) => ({ from: c.from, to: c.to, state: c.state, type: c.type }))
    };
  }
  destroy() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.leftOverflowIndicator && this.leftOverflowIndicator.parentNode) {
      this.leftOverflowIndicator.parentNode.removeChild(this.leftOverflowIndicator);
    }
    if (this.rightOverflowIndicator && this.rightOverflowIndicator.parentNode) {
      this.rightOverflowIndicator.parentNode.removeChild(this.rightOverflowIndicator);
    }
    for (const node of this.nodes.values()) {
      node.destroy();
    }
    for (const lane of this.lanes.values()) {
      lane.destroy();
    }
    for (const connection of this.connections.values()) {
      connection.destroy();
    }
    this.nodes.clear();
    this.lanes.clear();
    this.connections.clear();
    if (this.svg && this.svg.parentNode) {
      this.svg.parentNode.removeChild(this.svg);
    }
  }
}
export {
  ProcessFlow
};

//# debugId=D3480976FDB32E6264756E2164756E21
