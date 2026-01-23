// NetworkGraph/types.ts
var SAP_COLORS = {
  brand: "#0070f2",
  success: "#30914c",
  warning: "#df6e0c",
  error: "#cc1919",
  neutral: "#6c757d",
  background: "#f7f7f7",
  backgroundAlt: "#ffffff",
  border: "#d9d9d9",
  text: "#32363a",
  textMuted: "#6c757d"
};
var DEFAULT_RENDER_CONFIG = {
  nodeRadius: 40,
  nodeStrokeWidth: 2,
  edgeStrokeWidth: 2,
  arrowSize: 8,
  fontSize: 14,
  iconSize: 24,
  showLabels: true,
  showMetrics: true,
  enableAnimations: true,
  theme: "light"
};
var DEFAULT_FORCE_CONFIG = {
  repulsion: 1000,
  attraction: 0.01,
  gravity: 0.1,
  damping: 0.9,
  maxVelocity: 10
};
var DEFAULT_VIEWPORT = {
  x: 0,
  y: 0,
  scale: 1,
  width: 800,
  height: 600
};

// NetworkGraph/GraphNode.ts
class GraphNode {
  id;
  name;
  description;
  type;
  status;
  model;
  metrics;
  group;
  shape;
  position;
  velocity;
  force;
  mass;
  radius;
  fixed;
  rectWidth = 120;
  rectHeight = 80;
  element;
  circle = null;
  rect = null;
  icon;
  label;
  statusIndicator;
  halo;
  expandButton = null;
  isSelected = false;
  isHovered = false;
  isDragging = false;
  expandState = "collapsed";
  hasChildren = false;
  onExpandClickCallback = null;
  constructor(config) {
    this.id = config.id;
    this.name = config.name;
    this.description = config.description || "";
    this.type = config.type;
    this.status = config.status;
    this.model = config.model || "N/A";
    this.metrics = config.metrics || {
      totalRequests: 0,
      avgLatency: 0,
      successRate: 0
    };
    this.group = config.group || null;
    this.shape = config.shape || "circle";
    this.position = config.position || this.randomPosition();
    this.velocity = { x: 0, y: 0 };
    this.force = { x: 0, y: 0 };
    this.mass = 1;
    this.radius = DEFAULT_RENDER_CONFIG.nodeRadius;
    this.fixed = false;
    this.element = this.createElement();
  }
  createElement() {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "graph-node");
    g.setAttribute("data-node-id", this.id);
    g.setAttribute("data-node-type", this.type);
    g.setAttribute("data-node-shape", this.shape);
    this.halo = this.createHalo();
    g.appendChild(this.halo);
    if (this.shape === "rectangle") {
      this.rect = this.createRect();
      g.appendChild(this.rect);
    } else {
      this.circle = this.createCircle();
      g.appendChild(this.circle);
    }
    this.statusIndicator = this.createStatusIndicator();
    g.appendChild(this.statusIndicator);
    this.icon = this.createIcon();
    g.appendChild(this.icon);
    this.label = this.createLabel();
    g.appendChild(this.label);
    this.expandButton = this.renderExpandButton();
    g.appendChild(this.expandButton);
    this.updatePosition();
    return g;
  }
  createHalo() {
    if (this.shape === "rectangle") {
      const halo = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      halo.setAttribute("class", "node-halo");
      halo.setAttribute("x", (-(this.rectWidth / 2) - 8).toString());
      halo.setAttribute("y", (-(this.rectHeight / 2) - 8).toString());
      halo.setAttribute("width", (this.rectWidth + 16).toString());
      halo.setAttribute("height", (this.rectHeight + 16).toString());
      halo.setAttribute("rx", "12");
      halo.setAttribute("ry", "12");
      halo.setAttribute("fill", "none");
      halo.setAttribute("stroke", SAP_COLORS.brand);
      halo.setAttribute("stroke-width", "3");
      halo.setAttribute("opacity", "0");
      return halo;
    } else {
      const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      halo.setAttribute("class", "node-halo");
      halo.setAttribute("r", (this.radius + 8).toString());
      halo.setAttribute("fill", "none");
      halo.setAttribute("stroke", SAP_COLORS.brand);
      halo.setAttribute("stroke-width", "3");
      halo.setAttribute("opacity", "0");
      return halo;
    }
  }
  createRect() {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("class", "node-rect graph-node-rect");
    rect.setAttribute("x", (-this.rectWidth / 2).toString());
    rect.setAttribute("y", (-this.rectHeight / 2).toString());
    rect.setAttribute("width", this.rectWidth.toString());
    rect.setAttribute("height", this.rectHeight.toString());
    rect.setAttribute("rx", "8");
    rect.setAttribute("ry", "8");
    rect.setAttribute("fill", this.getStatusColor());
    rect.setAttribute("stroke", "#ffffff");
    rect.setAttribute("stroke-width", "2");
    rect.setAttribute("filter", "url(#nodeRectShadow)");
    rect.style.cursor = "pointer";
    rect.style.transition = "all 0.3s ease";
    return rect;
  }
  createCircle() {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "node-circle");
    circle.setAttribute("r", this.radius.toString());
    circle.setAttribute("fill", this.getStatusColor());
    circle.setAttribute("stroke", "#ffffff");
    circle.setAttribute("stroke-width", "2");
    circle.setAttribute("filter", "url(#nodeShadow)");
    circle.style.cursor = "pointer";
    circle.style.transition = "all 0.3s ease";
    return circle;
  }
  createStatusIndicator() {
    const indicator = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    indicator.setAttribute("class", "status-indicator");
    indicator.setAttribute("r", "8");
    indicator.setAttribute("fill", this.getStatusColor());
    indicator.setAttribute("stroke", "#ffffff");
    indicator.setAttribute("stroke-width", "2");
    return indicator;
  }
  createIcon() {
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "text");
    icon.setAttribute("class", "node-icon");
    icon.setAttribute("text-anchor", "middle");
    icon.setAttribute("dominant-baseline", "central");
    icon.setAttribute("font-size", "24");
    icon.setAttribute("fill", "#ffffff");
    icon.setAttribute("pointer-events", "none");
    icon.textContent = this.getIconText();
    return icon;
  }
  createLabel() {
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "node-label");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "12");
    label.setAttribute("font-weight", "bold");
    label.setAttribute("fill", SAP_COLORS.text);
    label.setAttribute("pointer-events", "none");
    label.textContent = this.name;
    return label;
  }
  renderExpandButton() {
    const buttonGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    buttonGroup.setAttribute("class", "node-expand-button");
    const angle = Math.PI / 4;
    const offsetX = this.radius * Math.cos(angle);
    const offsetY = this.radius * Math.sin(angle);
    buttonGroup.setAttribute("transform", `translate(${offsetX}, ${offsetY})`);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "expand-button-bg");
    circle.setAttribute("r", "10");
    circle.setAttribute("fill", SAP_COLORS.brand);
    circle.setAttribute("stroke", "#ffffff");
    circle.setAttribute("stroke-width", "2");
    circle.style.cursor = "pointer";
    buttonGroup.appendChild(circle);
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "text");
    icon.setAttribute("class", "expand-button-icon");
    icon.setAttribute("text-anchor", "middle");
    icon.setAttribute("dominant-baseline", "central");
    icon.setAttribute("font-size", "10");
    icon.setAttribute("fill", "#ffffff");
    icon.setAttribute("pointer-events", "none");
    icon.textContent = this.getExpandIcon();
    buttonGroup.appendChild(icon);
    buttonGroup.style.display = this.hasChildren ? "block" : "none";
    buttonGroup.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this.onExpandClickCallback) {
        this.onExpandClickCallback(this.id, this.expandState);
      }
    });
    return buttonGroup;
  }
  getExpandIcon() {
    switch (this.expandState) {
      case "expanded":
        return "▼";
      case "partial":
        return "◐";
      case "collapsed":
      default:
        return "▶";
    }
  }
  setExpandState(state) {
    this.expandState = state;
    if (this.expandButton) {
      const icon = this.expandButton.querySelector(".expand-button-icon");
      if (icon) {
        icon.textContent = this.getExpandIcon();
      }
      const bg = this.expandButton.querySelector(".expand-button-bg");
      if (bg) {
        switch (state) {
          case "expanded":
            bg.setAttribute("fill", SAP_COLORS.success);
            break;
          case "partial":
            bg.setAttribute("fill", SAP_COLORS.warning);
            break;
          case "collapsed":
          default:
            bg.setAttribute("fill", SAP_COLORS.brand);
            break;
        }
      }
      if (DEFAULT_RENDER_CONFIG.enableAnimations) {
        this.expandButton.animate([
          { transform: `translate(${this.radius * Math.cos(Math.PI / 4)}, ${this.radius * Math.sin(Math.PI / 4)}) scale(1.3)` },
          { transform: `translate(${this.radius * Math.cos(Math.PI / 4)}, ${this.radius * Math.sin(Math.PI / 4)}) scale(1.0)` }
        ], {
          duration: 200,
          easing: "ease-out"
        });
      }
    }
  }
  setHasChildren(hasChildren) {
    this.hasChildren = hasChildren;
    if (this.expandButton) {
      this.expandButton.style.display = hasChildren ? "block" : "none";
    }
  }
  onExpandClick(callback) {
    this.onExpandClickCallback = callback;
  }
  updatePosition() {
    if (!this.element)
      return;
    this.element.setAttribute("transform", `translate(${this.position.x}, ${this.position.y})`);
    if (this.shape === "rectangle") {
      const offsetX = this.rectWidth / 2 - 8;
      const offsetY = -this.rectHeight / 2 + 8;
      this.statusIndicator.setAttribute("cx", offsetX.toString());
      this.statusIndicator.setAttribute("cy", offsetY.toString());
    } else {
      const angle = -Math.PI / 4;
      const offsetX = this.radius * Math.cos(angle);
      const offsetY = this.radius * Math.sin(angle);
      this.statusIndicator.setAttribute("cx", offsetX.toString());
      this.statusIndicator.setAttribute("cy", offsetY.toString());
    }
  }
  applyForce(fx, fy) {
    if (this.fixed)
      return;
    this.force.x += fx;
    this.force.y += fy;
  }
  updatePhysics(dt = 1) {
    if (this.fixed)
      return;
    const ax = this.force.x / this.mass;
    const ay = this.force.y / this.mass;
    this.velocity.x += ax * dt;
    this.velocity.y += ay * dt;
    this.velocity.x *= 0.9;
    this.velocity.y *= 0.9;
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    if (speed > 10) {
      this.velocity.x = this.velocity.x / speed * 10;
      this.velocity.y = this.velocity.y / speed * 10;
    }
    this.position.x += this.velocity.x * dt;
    this.position.y += this.velocity.y * dt;
    this.force.x = 0;
    this.force.y = 0;
    this.updatePosition();
  }
  setStatus(status) {
    this.status = status;
    const color = this.getStatusColor();
    const shapeElement = this.getShapeElement();
    if (shapeElement) {
      shapeElement.setAttribute("fill", color);
    }
    this.statusIndicator.setAttribute("fill", color);
    if (DEFAULT_RENDER_CONFIG.enableAnimations) {
      this.pulse();
    }
  }
  setSelected(selected) {
    this.isSelected = selected;
    const shapeElement = this.getShapeElement();
    if (selected) {
      this.halo.setAttribute("opacity", "1");
      if (shapeElement) {
        shapeElement.setAttribute("stroke-width", "4");
      }
      this.element.style.filter = "drop-shadow(0 4px 8px rgba(0,0,0,0.3))";
    } else {
      this.halo.setAttribute("opacity", "0");
      if (shapeElement) {
        shapeElement.setAttribute("stroke-width", "2");
      }
      this.element.style.filter = "";
    }
  }
  setHovered(hovered) {
    this.isHovered = hovered;
    const shapeElement = this.getShapeElement();
    if (hovered) {
      this.halo.setAttribute("opacity", "0.5");
      if (shapeElement) {
        shapeElement.style.transform = "scale(1.1)";
      }
    } else if (!this.isSelected) {
      this.halo.setAttribute("opacity", "0");
      if (shapeElement) {
        shapeElement.style.transform = "scale(1.0)";
      }
    }
  }
  setDragging(dragging) {
    this.isDragging = dragging;
    this.fixed = dragging;
    const shapeElement = this.getShapeElement();
    if (dragging) {
      this.element.style.cursor = "grabbing";
      if (shapeElement) {
        shapeElement.style.opacity = "0.8";
      }
    } else {
      this.element.style.cursor = "pointer";
      if (shapeElement) {
        shapeElement.style.opacity = "1.0";
      }
    }
  }
  getShapeElement() {
    return this.shape === "rectangle" ? this.rect : this.circle;
  }
  pulse() {
    const shapeElement = this.getShapeElement();
    if (!shapeElement)
      return;
    shapeElement.animate([
      { transform: "scale(1.0)" },
      { transform: "scale(1.2)", offset: 0.5 },
      { transform: "scale(1.0)" }
    ], {
      duration: 600,
      easing: "ease-out"
    });
  }
  highlight() {
    this.element.style.filter = "drop-shadow(0 0 10px " + this.getStatusColor() + ")";
    setTimeout(() => {
      this.element.style.filter = "";
    }, 1000);
  }
  startPulseLoop() {
    if (this.status === "Running" /* Running */) {
      const animate = () => {
        if (this.status === "Running" /* Running */) {
          this.pulse();
          setTimeout(animate, 1500);
        }
      };
      animate();
    }
  }
  getStatusColor() {
    switch (this.status) {
      case "Success" /* Success */:
        return SAP_COLORS.success;
      case "Warning" /* Warning */:
        return SAP_COLORS.warning;
      case "Error" /* Error */:
        return SAP_COLORS.error;
      case "Running" /* Running */:
        return SAP_COLORS.brand;
      case "None" /* None */:
      default:
        return SAP_COLORS.neutral;
    }
  }
  getIconText() {
    const iconMap = {
      code_intelligence: "\uD83D\uDCDD",
      vector_search: "\uD83D\uDD0D",
      graph_database: "\uD83D\uDD78️",
      verification: "✓",
      workflow: "⚙️",
      lineage: "\uD83D\uDCCA",
      orchestrator: "\uD83C\uDFAF",
      router: "\uD83D\uDD00",
      translation: "\uD83C\uDF10",
      rag: "\uD83D\uDCDA"
    };
    return iconMap[this.type] || "⚡";
  }
  randomPosition() {
    const angle = Math.random() * 2 * Math.PI;
    const distance = Math.random() * 200 + 100;
    return {
      x: Math.cos(angle) * distance,
      y: Math.sin(angle) * distance
    };
  }
  distanceTo(other) {
    const dx = this.position.x - other.position.x;
    const dy = this.position.y - other.position.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  angleTo(other) {
    const dx = other.position.x - this.position.x;
    const dy = other.position.y - this.position.y;
    return Math.atan2(dy, dx);
  }
  overlaps(other) {
    const minDistance = this.radius + other.radius;
    return this.distanceTo(other) < minDistance;
  }
  containsPoint(point) {
    const dx = point.x - this.position.x;
    const dy = point.y - this.position.y;
    if (this.shape === "rectangle") {
      return Math.abs(dx) <= this.rectWidth / 2 && Math.abs(dy) <= this.rectHeight / 2;
    } else {
      const distance = Math.sqrt(dx * dx + dy * dy);
      return distance <= this.radius;
    }
  }
  getWidth() {
    return this.shape === "rectangle" ? this.rectWidth : this.radius * 2;
  }
  getHeight() {
    return this.shape === "rectangle" ? this.rectHeight : this.radius * 2;
  }
  toJSON() {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      type: this.type,
      status: this.status,
      model: this.model,
      metrics: this.metrics,
      group: this.group || undefined,
      position: { ...this.position },
      shape: this.shape
    };
  }
  destroy() {
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

// NetworkGraph/GraphEdge.ts
class GraphEdge {
  id;
  from;
  to;
  label;
  status;
  animated;
  element;
  path;
  arrowHead;
  labelText;
  flowCircle;
  sourceNode = null;
  targetNode = null;
  flowAnimation = null;
  constructor(config) {
    this.id = config.id;
    this.from = config.from;
    this.to = config.to;
    this.label = config.label || "";
    this.status = config.status || "Inactive" /* Inactive */;
    this.animated = config.animated || false;
    this.element = this.createElement();
  }
  createElement() {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "graph-edge");
    g.setAttribute("data-edge-id", this.id);
    g.setAttribute("data-from", this.from);
    g.setAttribute("data-to", this.to);
    this.path = this.createPath();
    g.appendChild(this.path);
    this.arrowHead = this.createArrowHead();
    g.appendChild(this.arrowHead);
    this.flowCircle = this.createFlowCircle();
    g.appendChild(this.flowCircle);
    if (this.label) {
      this.labelText = this.createLabel();
      g.appendChild(this.labelText);
    }
    return g;
  }
  createPath() {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "edge-path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", this.getStatusColor());
    path.setAttribute("stroke-width", DEFAULT_RENDER_CONFIG.edgeStrokeWidth.toString());
    path.setAttribute("stroke-linecap", "round");
    path.style.transition = "stroke 0.3s ease";
    if (this.status === "Inactive" /* Inactive */) {
      path.setAttribute("stroke-dasharray", "5,5");
    }
    return path;
  }
  createArrowHead() {
    const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    arrow.setAttribute("class", "edge-arrow");
    arrow.setAttribute("fill", this.getStatusColor());
    arrow.style.transition = "fill 0.3s ease";
    return arrow;
  }
  createFlowCircle() {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "edge-flow");
    circle.setAttribute("r", "4");
    circle.setAttribute("fill", SAP_COLORS.brand);
    circle.setAttribute("opacity", "0");
    return circle;
  }
  createLabel() {
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "edge-label");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "10");
    label.setAttribute("fill", SAP_COLORS.textMuted);
    label.setAttribute("pointer-events", "none");
    label.textContent = this.label;
    return label;
  }
  setNodes(source, target) {
    this.sourceNode = source;
    this.targetNode = target;
    this.updatePath();
  }
  updatePath() {
    if (!this.sourceNode || !this.targetNode)
      return;
    const start = this.sourceNode.position;
    const end = this.targetNode.position;
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    const startOffset = {
      x: start.x + this.sourceNode.radius * Math.cos(angle),
      y: start.y + this.sourceNode.radius * Math.sin(angle)
    };
    const endOffset = {
      x: end.x - this.targetNode.radius * Math.cos(angle),
      y: end.y - this.targetNode.radius * Math.sin(angle)
    };
    const pathData = this.createBezierPath(startOffset, endOffset);
    this.path.setAttribute("d", pathData);
    this.positionArrowHead(endOffset, angle);
    if (this.labelText) {
      const midX = (startOffset.x + endOffset.x) / 2;
      const midY = (startOffset.y + endOffset.y) / 2;
      this.labelText.setAttribute("x", midX.toString());
      this.labelText.setAttribute("y", (midY - 10).toString());
    }
  }
  createBezierPath(start, end) {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const curvature = 0.2;
    const offset = distance * curvature;
    const angle = Math.atan2(dy, dx);
    const perpAngle = angle + Math.PI / 2;
    const cp1x = start.x + dx * 0.33 + offset * Math.cos(perpAngle);
    const cp1y = start.y + dy * 0.33 + offset * Math.sin(perpAngle);
    const cp2x = start.x + dx * 0.67 + offset * Math.cos(perpAngle);
    const cp2y = start.y + dy * 0.67 + offset * Math.sin(perpAngle);
    if (distance < 100) {
      return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
    }
    return `M ${start.x} ${start.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${end.x} ${end.y}`;
  }
  positionArrowHead(point, angle) {
    const arrowSize = DEFAULT_RENDER_CONFIG.arrowSize;
    const tip = point;
    const base1 = {
      x: tip.x - arrowSize * Math.cos(angle - Math.PI / 6),
      y: tip.y - arrowSize * Math.sin(angle - Math.PI / 6)
    };
    const base2 = {
      x: tip.x - arrowSize * Math.cos(angle + Math.PI / 6),
      y: tip.y - arrowSize * Math.sin(angle + Math.PI / 6)
    };
    const points = `${tip.x},${tip.y} ${base1.x},${base1.y} ${base2.x},${base2.y}`;
    this.arrowHead.setAttribute("points", points);
  }
  setStatus(status) {
    this.status = status;
    const color = this.getStatusColor();
    this.path.setAttribute("stroke", color);
    this.arrowHead.setAttribute("fill", color);
    if (status === "Inactive" /* Inactive */) {
      this.path.setAttribute("stroke-dasharray", "5,5");
    } else {
      this.path.removeAttribute("stroke-dasharray");
    }
    if (status === "Flowing" /* Flowing */ || status === "Active" /* Active */) {
      this.startFlowAnimation();
    } else {
      this.stopFlowAnimation();
    }
  }
  setHighlighted(highlighted) {
    if (highlighted) {
      this.path.setAttribute("stroke-width", "4");
      this.element.style.filter = "drop-shadow(0 2px 4px rgba(0,0,0,0.3))";
    } else {
      this.path.setAttribute("stroke-width", DEFAULT_RENDER_CONFIG.edgeStrokeWidth.toString());
      this.element.style.filter = "";
    }
  }
  startFlowAnimation() {
    if (this.flowAnimation)
      return;
    this.flowCircle.setAttribute("opacity", "1");
    this.flowAnimation = this.flowCircle.animate([
      { offsetDistance: "0%" },
      { offsetDistance: "100%" }
    ], {
      duration: 2000,
      iterations: Infinity,
      easing: "linear"
    });
    this.flowCircle.style.offsetPath = `path("${this.path.getAttribute("d")}")`;
  }
  stopFlowAnimation() {
    if (this.flowAnimation) {
      this.flowAnimation.cancel();
      this.flowAnimation = null;
    }
    this.flowCircle.setAttribute("opacity", "0");
  }
  pulse() {
    this.element.animate([
      { opacity: 1 },
      { opacity: 0.5, offset: 0.5 },
      { opacity: 1 }
    ], {
      duration: 800,
      easing: "ease-in-out"
    });
  }
  flash() {
    this.path.animate([
      { stroke: this.getStatusColor() },
      { stroke: SAP_COLORS.brand },
      { stroke: this.getStatusColor() }
    ], {
      duration: 400,
      easing: "ease-out"
    });
  }
  getStatusColor() {
    switch (this.status) {
      case "Active" /* Active */:
        return SAP_COLORS.brand;
      case "Flowing" /* Flowing */:
        return SAP_COLORS.success;
      case "Error" /* Error */:
        return SAP_COLORS.error;
      case "Inactive" /* Inactive */:
      default:
        return SAP_COLORS.neutral;
    }
  }
  getLength() {
    if (!this.sourceNode || !this.targetNode)
      return 0;
    const dx = this.targetNode.position.x - this.sourceNode.position.x;
    const dy = this.targetNode.position.y - this.sourceNode.position.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  getMidpoint() {
    if (!this.sourceNode || !this.targetNode) {
      return { x: 0, y: 0 };
    }
    return {
      x: (this.sourceNode.position.x + this.targetNode.position.x) / 2,
      y: (this.sourceNode.position.y + this.targetNode.position.y) / 2
    };
  }
  containsPoint(point, threshold = 5) {
    if (!this.sourceNode || !this.targetNode)
      return false;
    const start = this.sourceNode.position;
    const end = this.targetNode.position;
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared === 0) {
      const distX2 = point.x - start.x;
      const distY2 = point.y - start.y;
      return Math.sqrt(distX2 * distX2 + distY2 * distY2) <= threshold;
    }
    const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
    const projX = start.x + t * dx;
    const projY = start.y + t * dy;
    const distX = point.x - projX;
    const distY = point.y - projY;
    const distance = Math.sqrt(distX * distX + distY * distY);
    return distance <= threshold;
  }
  toJSON() {
    return {
      id: this.id,
      from: this.from,
      to: this.to,
      label: this.label,
      status: this.status,
      animated: this.animated
    };
  }
  destroy() {
    this.stopFlowAnimation();
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

// NetworkGraph/LayoutEngine.ts
class LayoutEngine {
  nodes = [];
  edges = [];
  config;
  alpha = 1;
  alphaDecay = 0.95;
  animationFrame = null;
  constructor(config) {
    this.config = {
      type: "force-directed" /* ForceDirected */,
      animate: true,
      duration: 1000,
      forces: DEFAULT_FORCE_CONFIG,
      padding: 50,
      ...config
    };
  }
  setNodes(nodes) {
    this.nodes = nodes;
  }
  setEdges(edges) {
    this.edges = edges;
  }
  setConfig(config) {
    this.config = { ...this.config, ...config };
  }
  start() {
    this.alpha = 1;
    switch (this.config.type) {
      case "force-directed" /* ForceDirected */:
        this.startForceDirectedLayout();
        break;
      case "hierarchical" /* Hierarchical */:
        this.applyHierarchicalLayout();
        break;
      case "circular" /* Circular */:
        this.applyCircularLayout();
        break;
      case "grid" /* Grid */:
        this.applyGridLayout();
        break;
      case "manual" /* Manual */:
        break;
    }
  }
  stop() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    this.alpha = 0;
  }
  tick() {
    if (this.alpha <= 0.01) {
      this.stop();
      return;
    }
    if (this.config.type === "force-directed" /* ForceDirected */) {
      this.tickForceDirected();
    }
    this.alpha *= this.alphaDecay;
  }
  startForceDirectedLayout() {
    if (!this.config.animate) {
      while (this.alpha > 0.01) {
        this.tickForceDirected();
        this.alpha *= this.alphaDecay;
      }
      return;
    }
    const animate = () => {
      this.tickForceDirected();
      this.alpha *= this.alphaDecay;
      if (this.alpha > 0.01) {
        this.animationFrame = requestAnimationFrame(animate);
      }
    };
    animate();
  }
  tickForceDirected() {
    const forces = this.config.forces || DEFAULT_FORCE_CONFIG;
    for (const node of this.nodes) {
      node.force.x = 0;
      node.force.y = 0;
    }
    for (let i = 0;i < this.nodes.length; i++) {
      for (let j = i + 1;j < this.nodes.length; j++) {
        this.applyRepulsionForce(this.nodes[i], this.nodes[j], forces.repulsion);
      }
    }
    for (const edge of this.edges) {
      const source = this.nodes.find((n) => n.id === edge.from);
      const target = this.nodes.find((n) => n.id === edge.to);
      if (source && target) {
        this.applyAttractionForce(source, target, forces.attraction);
      }
    }
    const centerX = 0;
    const centerY = 0;
    for (const node of this.nodes) {
      this.applyGravityForce(node, centerX, centerY, forces.gravity);
    }
    for (const node of this.nodes) {
      node.updatePhysics(this.alpha);
    }
    for (const edge of this.edges) {
      edge.updatePath();
    }
  }
  applyRepulsionForce(node1, node2, strength) {
    const dx = node2.position.x - node1.position.x;
    const dy = node2.position.y - node1.position.y;
    const distanceSquared = dx * dx + dy * dy;
    if (distanceSquared === 0)
      return;
    const distance = Math.sqrt(distanceSquared);
    const force = strength / distanceSquared;
    const fx = dx / distance * force;
    const fy = dy / distance * force;
    node1.applyForce(-fx, -fy);
    node2.applyForce(fx, fy);
  }
  applyAttractionForce(source, target, strength) {
    const dx = target.position.x - source.position.x;
    const dy = target.position.y - source.position.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance === 0)
      return;
    const force = distance * strength;
    const fx = dx / distance * force;
    const fy = dy / distance * force;
    source.applyForce(fx, fy);
    target.applyForce(-fx, -fy);
  }
  applyGravityForce(node, cx, cy, strength) {
    const dx = cx - node.position.x;
    const dy = cy - node.position.y;
    const fx = dx * strength;
    const fy = dy * strength;
    node.applyForce(fx, fy);
  }
  applyHierarchicalLayout() {
    if (this.nodes.length === 0)
      return;
    const layers = this.assignLayers();
    const layerSpacing = 150;
    const nodeSpacing = 100;
    let y = -((layers.length - 1) * layerSpacing) / 2;
    for (const layer of layers) {
      let x = -((layer.length - 1) * nodeSpacing) / 2;
      for (const node of layer) {
        node.position.x = x;
        node.position.y = y;
        node.fixed = false;
        x += nodeSpacing;
      }
      y += layerSpacing;
    }
    for (const edge of this.edges) {
      edge.updatePath();
    }
    if (this.config.animate) {
      this.animateToPositions(this.config.duration);
    }
  }
  assignLayers() {
    const inDegree = new Map;
    const layers = [];
    for (const node of this.nodes) {
      inDegree.set(node.id, 0);
    }
    for (const edge of this.edges) {
      const degree = inDegree.get(edge.to) || 0;
      inDegree.set(edge.to, degree + 1);
    }
    const queue = this.nodes.filter((n) => inDegree.get(n.id) === 0);
    const assigned = new Set;
    while (queue.length > 0) {
      const layer = [...queue];
      layers.push(layer);
      const nextQueue = [];
      for (const node of layer) {
        assigned.add(node.id);
        for (const edge of this.edges) {
          if (edge.from === node.id) {
            const target = this.nodes.find((n) => n.id === edge.to);
            if (target && !assigned.has(target.id) && !nextQueue.includes(target)) {
              nextQueue.push(target);
            }
          }
        }
      }
      queue.length = 0;
      queue.push(...nextQueue);
    }
    const unassigned = this.nodes.filter((n) => !assigned.has(n.id));
    if (unassigned.length > 0) {
      layers.push(unassigned);
    }
    return layers;
  }
  applyCircularLayout() {
    if (this.nodes.length === 0)
      return;
    const radius = Math.max(200, this.nodes.length * 20);
    const angleStep = 2 * Math.PI / this.nodes.length;
    this.nodes.forEach((node, i) => {
      const angle = i * angleStep;
      node.position.x = radius * Math.cos(angle);
      node.position.y = radius * Math.sin(angle);
      node.fixed = false;
    });
    for (const edge of this.edges) {
      edge.updatePath();
    }
    if (this.config.animate) {
      this.animateToPositions(this.config.duration);
    }
  }
  applyGridLayout() {
    if (this.nodes.length === 0)
      return;
    const cols = Math.ceil(Math.sqrt(this.nodes.length));
    const spacing = 150;
    this.nodes.forEach((node, i) => {
      const row = Math.floor(i / cols);
      const col = i % cols;
      node.position.x = (col - cols / 2) * spacing;
      node.position.y = (row - cols / 2) * spacing;
      node.fixed = false;
    });
    for (const edge of this.edges) {
      edge.updatePath();
    }
    if (this.config.animate) {
      this.animateToPositions(this.config.duration);
    }
  }
  animateToPositions(duration) {
    const startPositions = this.nodes.map((n) => ({ ...n.position }));
    const startTime = Date.now();
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = this.easeInOutCubic(progress);
      this.nodes.forEach((node, i) => {
        const start = startPositions[i];
        node.position.x = start.x + (node.position.x - start.x) * eased;
        node.position.y = start.y + (node.position.y - start.y) * eased;
        node.updatePosition();
      });
      for (const edge of this.edges) {
        edge.updatePath();
      }
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    animate();
  }
  easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }
  centerGraph() {
    if (this.nodes.length === 0)
      return;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const node of this.nodes) {
      minX = Math.min(minX, node.position.x);
      maxX = Math.max(maxX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxY = Math.max(maxY, node.position.y);
    }
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    for (const node of this.nodes) {
      node.position.x -= centerX;
      node.position.y -= centerY;
      node.updatePosition();
    }
    for (const edge of this.edges) {
      edge.updatePath();
    }
  }
  fitToViewport(width, height) {
    if (this.nodes.length === 0)
      return 1;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const node of this.nodes) {
      minX = Math.min(minX, node.position.x - node.radius);
      maxX = Math.max(maxX, node.position.x + node.radius);
      minY = Math.min(minY, node.position.y - node.radius);
      maxY = Math.max(maxY, node.position.y + node.radius);
    }
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;
    const padding = this.config.padding;
    const scaleX = (width - 2 * padding) / graphWidth;
    const scaleY = (height - 2 * padding) / graphHeight;
    return Math.min(scaleX, scaleY, 2);
  }
}

// NetworkGraph/InteractionHandler.ts
class InteractionHandler {
  svg;
  viewport;
  nodes = new Map;
  edges = new Map;
  isDragging = false;
  isPanning = false;
  draggedNode = null;
  dragStart = { x: 0, y: 0 };
  panStart = { x: 0, y: 0 };
  lastMousePos = { x: 0, y: 0 };
  onNodeClick = null;
  onNodeDrag = null;
  onViewportChange = null;
  constructor(svg, viewport) {
    this.svg = svg;
    this.viewport = viewport;
    this.setupEventListeners();
  }
  setupEventListeners() {
    this.svg.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.svg.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.svg.addEventListener("mouseup", this.onMouseUp.bind(this));
    this.svg.addEventListener("wheel", this.onWheel.bind(this), { passive: false });
    this.svg.addEventListener("mouseleave", this.onMouseLeave.bind(this));
    this.svg.addEventListener("touchstart", this.onTouchStart.bind(this), { passive: false });
    this.svg.addEventListener("touchmove", this.onTouchMove.bind(this), { passive: false });
    this.svg.addEventListener("touchend", this.onTouchEnd.bind(this));
    this.svg.addEventListener("contextmenu", (e) => e.preventDefault());
  }
  setNodes(nodes) {
    this.nodes = nodes;
  }
  setEdges(edges) {
    this.edges = edges;
  }
  setViewport(viewport) {
    this.viewport = viewport;
  }
  on(event, callback) {
    switch (event) {
      case "nodeClick":
        this.onNodeClick = callback;
        break;
      case "nodeDrag":
        this.onNodeDrag = callback;
        break;
      case "viewportChange":
        this.onViewportChange = callback;
        break;
    }
  }
  onMouseDown(event) {
    const worldPos = this.screenToWorld({ x: event.clientX, y: event.clientY });
    const clickedNode = this.getNodeAtPosition(worldPos);
    if (clickedNode) {
      this.isDragging = true;
      this.draggedNode = clickedNode;
      this.dragStart = { ...worldPos };
      clickedNode.setDragging(true);
      if (this.onNodeClick) {
        this.onNodeClick({
          type: "click",
          node: clickedNode,
          position: worldPos,
          originalEvent: event
        });
      }
    } else {
      this.isPanning = true;
      this.panStart = {
        x: event.clientX - this.viewport.x,
        y: event.clientY - this.viewport.y
      };
    }
    this.lastMousePos = { x: event.clientX, y: event.clientY };
    event.preventDefault();
  }
  onMouseMove(event) {
    const worldPos = this.screenToWorld({ x: event.clientX, y: event.clientY });
    if (this.isDragging && this.draggedNode) {
      this.draggedNode.position.x = worldPos.x;
      this.draggedNode.position.y = worldPos.y;
      this.draggedNode.updatePosition();
      this.updateEdgesForNode(this.draggedNode);
      if (this.onNodeDrag) {
        this.onNodeDrag({
          type: "drag",
          node: this.draggedNode,
          position: worldPos,
          originalEvent: event
        });
      }
    } else if (this.isPanning) {
      this.viewport.x = event.clientX - this.panStart.x;
      this.viewport.y = event.clientY - this.panStart.y;
      this.updateSVGTransform();
      if (this.onViewportChange) {
        this.onViewportChange({
          type: "pan",
          viewport: this.viewport,
          originalEvent: event
        });
      }
    } else {
      const hoveredNode = this.getNodeAtPosition(worldPos);
      for (const node of this.nodes.values()) {
        node.setHovered(node === hoveredNode);
      }
      this.svg.style.cursor = hoveredNode ? "pointer" : this.isPanning ? "grabbing" : "grab";
    }
    this.lastMousePos = { x: event.clientX, y: event.clientY };
  }
  onMouseUp(event) {
    if (this.draggedNode) {
      this.draggedNode.setDragging(false);
      this.draggedNode.fixed = false;
      this.draggedNode = null;
    }
    this.isDragging = false;
    this.isPanning = false;
    this.svg.style.cursor = "grab";
  }
  onMouseLeave(event) {
    this.onMouseUp(event);
    for (const node of this.nodes.values()) {
      node.setHovered(false);
    }
  }
  onWheel(event) {
    event.preventDefault();
    const mousePos = { x: event.clientX, y: event.clientY };
    const worldBefore = this.screenToWorld(mousePos);
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    this.viewport.scale = Math.max(0.1, Math.min(5, this.viewport.scale * zoomFactor));
    const worldAfter = this.screenToWorld(mousePos);
    this.viewport.x += (worldAfter.x - worldBefore.x) * this.viewport.scale;
    this.viewport.y += (worldAfter.y - worldBefore.y) * this.viewport.scale;
    this.updateSVGTransform();
    if (this.onViewportChange) {
      this.onViewportChange({
        type: "zoom",
        viewport: this.viewport,
        originalEvent: event
      });
    }
  }
  onTouchStart(event) {
    event.preventDefault();
    if (event.touches.length === 1) {
      const touch = event.touches[0];
      const mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.onMouseDown(mouseEvent);
    } else if (event.touches.length === 2) {
      this.isPanning = true;
      const center = this.getTouchCenter(event.touches);
      this.panStart = {
        x: center.x - this.viewport.x,
        y: center.y - this.viewport.y
      };
    }
  }
  onTouchMove(event) {
    event.preventDefault();
    if (event.touches.length === 1 && (this.isDragging || this.isPanning)) {
      const touch = event.touches[0];
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.onMouseMove(mouseEvent);
    } else if (event.touches.length === 2) {
      const center = this.getTouchCenter(event.touches);
      if (this.isPanning) {
        this.viewport.x = center.x - this.panStart.x;
        this.viewport.y = center.y - this.panStart.y;
        this.updateSVGTransform();
      }
    }
  }
  onTouchEnd(event) {
    if (event.touches.length === 0) {
      this.onMouseUp(new MouseEvent("mouseup"));
    }
  }
  getTouchCenter(touches) {
    let x = 0, y = 0;
    for (let i = 0;i < touches.length; i++) {
      x += touches[i].clientX;
      y += touches[i].clientY;
    }
    return {
      x: x / touches.length,
      y: y / touches.length
    };
  }
  screenToWorld(screenPos) {
    const rect = this.svg.getBoundingClientRect();
    return {
      x: (screenPos.x - rect.left - this.viewport.x - rect.width / 2) / this.viewport.scale,
      y: (screenPos.y - rect.top - this.viewport.y - rect.height / 2) / this.viewport.scale
    };
  }
  worldToScreen(worldPos) {
    const rect = this.svg.getBoundingClientRect();
    return {
      x: worldPos.x * this.viewport.scale + this.viewport.x + rect.width / 2,
      y: worldPos.y * this.viewport.scale + this.viewport.y + rect.height / 2
    };
  }
  getNodeAtPosition(worldPos) {
    const nodesArray = Array.from(this.nodes.values());
    for (let i = nodesArray.length - 1;i >= 0; i--) {
      if (nodesArray[i].containsPoint(worldPos)) {
        return nodesArray[i];
      }
    }
    return null;
  }
  updateEdgesForNode(node) {
    for (const edge of this.edges.values()) {
      if (edge.from === node.id || edge.to === node.id) {
        edge.updatePath();
      }
    }
  }
  updateSVGTransform() {
    const container = this.svg.querySelector(".graph-container");
    if (container) {
      container.setAttribute("transform", `translate(${this.viewport.x}, ${this.viewport.y}) scale(${this.viewport.scale})`);
    }
  }
  zoomIn() {
    this.viewport.scale = Math.min(5, this.viewport.scale * 1.2);
    this.updateSVGTransform();
    if (this.onViewportChange) {
      this.onViewportChange({
        type: "zoom",
        viewport: this.viewport
      });
    }
  }
  zoomOut() {
    this.viewport.scale = Math.max(0.1, this.viewport.scale / 1.2);
    this.updateSVGTransform();
    if (this.onViewportChange) {
      this.onViewportChange({
        type: "zoom",
        viewport: this.viewport
      });
    }
  }
  resetZoom() {
    this.viewport.scale = 1;
    this.viewport.x = 0;
    this.viewport.y = 0;
    this.updateSVGTransform();
    if (this.onViewportChange) {
      this.onViewportChange({
        type: "zoom",
        viewport: this.viewport
      });
    }
  }
  fitToView() {
    if (this.nodes.size === 0)
      return;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const node of this.nodes.values()) {
      minX = Math.min(minX, node.position.x - node.radius);
      maxX = Math.max(maxX, node.position.x + node.radius);
      minY = Math.min(minY, node.position.y - node.radius);
      maxY = Math.max(maxY, node.position.y + node.radius);
    }
    const rect = this.svg.getBoundingClientRect();
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;
    const padding = 100;
    const scaleX = (rect.width - 2 * padding) / graphWidth;
    const scaleY = (rect.height - 2 * padding) / graphHeight;
    this.viewport.scale = Math.min(scaleX, scaleY, 2);
    this.viewport.x = -(minX + maxX) / 2 * this.viewport.scale;
    this.viewport.y = -(minY + maxY) / 2 * this.viewport.scale;
    this.updateSVGTransform();
    if (this.onViewportChange) {
      this.onViewportChange({
        type: "zoom",
        viewport: this.viewport
      });
    }
  }
  destroy() {
    this.svg.removeEventListener("mousedown", this.onMouseDown.bind(this));
    this.svg.removeEventListener("mousemove", this.onMouseMove.bind(this));
    this.svg.removeEventListener("mouseup", this.onMouseUp.bind(this));
    this.svg.removeEventListener("wheel", this.onWheel.bind(this));
    this.svg.removeEventListener("mouseleave", this.onMouseLeave.bind(this));
    this.svg.removeEventListener("touchstart", this.onTouchStart.bind(this));
    this.svg.removeEventListener("touchmove", this.onTouchMove.bind(this));
    this.svg.removeEventListener("touchend", this.onTouchEnd.bind(this));
  }
}

// NetworkGraph/Minimap.ts
class Minimap {
  container;
  canvas;
  ctx;
  width = 200;
  height = 150;
  visible = true;
  nodes = [];
  edges = [];
  viewport;
  isDragging = false;
  onViewportMove = null;
  constructor(container, viewport) {
    this.container = container;
    this.viewport = viewport;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.canvas.className = "minimap-canvas";
    this.canvas.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            border: 2px solid ${SAP_COLORS.border};
            border-radius: 4px;
            background: ${SAP_COLORS.backgroundAlt};
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            cursor: pointer;
        `;
    const ctx = this.canvas.getContext("2d");
    if (!ctx)
      throw new Error("Canvas 2D context not available");
    this.ctx = ctx;
    this.container.appendChild(this.canvas);
    this.setupEventListeners();
  }
  setupEventListeners() {
    this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.canvas.addEventListener("mouseup", this.onMouseUp.bind(this));
    this.canvas.addEventListener("mouseleave", this.onMouseUp.bind(this));
  }
  onMouseDown(event) {
    this.isDragging = true;
    this.updateViewportFromClick(event);
  }
  onMouseMove(event) {
    if (this.isDragging) {
      this.updateViewportFromClick(event);
    }
  }
  onMouseUp(event) {
    this.isDragging = false;
  }
  updateViewportFromClick(event) {
    const rect = this.canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    const bounds = this.calculateGraphBounds();
    const worldX = bounds.minX + clickX / this.width * (bounds.maxX - bounds.minX);
    const worldY = bounds.minY + clickY / this.height * (bounds.maxY - bounds.minY);
    this.viewport.x = -worldX * this.viewport.scale;
    this.viewport.y = -worldY * this.viewport.scale;
    if (this.onViewportMove) {
      this.onViewportMove(this.viewport);
    }
    this.render();
  }
  setNodes(nodes) {
    this.nodes = nodes;
  }
  setEdges(edges) {
    this.edges = edges;
  }
  setViewport(viewport) {
    this.viewport = viewport;
  }
  render() {
    if (!this.visible)
      return;
    this.ctx.clearRect(0, 0, this.width, this.height);
    if (this.nodes.length === 0)
      return;
    const bounds = this.calculateGraphBounds();
    this.ctx.strokeStyle = SAP_COLORS.neutral;
    this.ctx.lineWidth = 1;
    for (const edge of this.edges) {
      const source = this.nodes.find((n) => n.id === edge.from);
      const target = this.nodes.find((n) => n.id === edge.to);
      if (source && target) {
        const startX = this.worldToMinimapX(source.position.x, bounds);
        const startY = this.worldToMinimapY(source.position.y, bounds);
        const endX = this.worldToMinimapX(target.position.x, bounds);
        const endY = this.worldToMinimapY(target.position.y, bounds);
        this.ctx.beginPath();
        this.ctx.moveTo(startX, startY);
        this.ctx.lineTo(endX, endY);
        this.ctx.stroke();
      }
    }
    for (const node of this.nodes) {
      const x = this.worldToMinimapX(node.position.x, bounds);
      const y = this.worldToMinimapY(node.position.y, bounds);
      const radius = 3;
      this.ctx.fillStyle = this.getStatusColor(node);
      this.ctx.beginPath();
      this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
      this.ctx.fill();
    }
    this.drawViewportRect(bounds);
  }
  drawViewportRect(bounds) {
    const viewWidth = this.viewport.width / this.viewport.scale;
    const viewHeight = this.viewport.height / this.viewport.scale;
    const centerX = -this.viewport.x / this.viewport.scale;
    const centerY = -this.viewport.y / this.viewport.scale;
    const x1 = this.worldToMinimapX(centerX - viewWidth / 2, bounds);
    const y1 = this.worldToMinimapY(centerY - viewHeight / 2, bounds);
    const x2 = this.worldToMinimapX(centerX + viewWidth / 2, bounds);
    const y2 = this.worldToMinimapY(centerY + viewHeight / 2, bounds);
    this.ctx.strokeStyle = SAP_COLORS.brand;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  calculateGraphBounds() {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const node of this.nodes) {
      minX = Math.min(minX, node.position.x);
      maxX = Math.max(maxX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxY = Math.max(maxY, node.position.y);
    }
    const padding = 50;
    return {
      minX: minX - padding,
      maxX: maxX + padding,
      minY: minY - padding,
      maxY: maxY + padding
    };
  }
  worldToMinimapX(worldX, bounds) {
    const rangeX = bounds.maxX - bounds.minX;
    return (worldX - bounds.minX) / rangeX * this.width;
  }
  worldToMinimapY(worldY, bounds) {
    const rangeY = bounds.maxY - bounds.minY;
    return (worldY - bounds.minY) / rangeY * this.height;
  }
  getStatusColor(node) {
    return SAP_COLORS.brand;
  }
  show() {
    this.visible = true;
    this.canvas.style.display = "block";
    this.render();
  }
  hide() {
    this.visible = false;
    this.canvas.style.display = "none";
  }
  toggle() {
    if (this.visible) {
      this.hide();
    } else {
      this.show();
    }
  }
  onViewportChange(callback) {
    this.onViewportMove = callback;
  }
  destroy() {
    if (this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
  }
}

// NetworkGraph/SearchFilter.ts
class SearchFilter {
  nodes = [];
  edges = [];
  filteredNodeIds = new Set;
  highlightedNodeIds = new Set;
  onFilterChange = null;
  onHighlightChange = null;
  setNodes(nodes) {
    this.nodes = nodes;
  }
  setEdges(edges) {
    this.edges = edges;
  }
  search(query) {
    const results = new Set;
    if (!query || query.trim() === "") {
      this.highlightedNodeIds.clear();
      if (this.onHighlightChange) {
        this.onHighlightChange(this.highlightedNodeIds);
      }
      return results;
    }
    const lowerQuery = query.toLowerCase();
    for (const node of this.nodes) {
      if (node.name.toLowerCase().includes(lowerQuery) || node.description.toLowerCase().includes(lowerQuery) || node.type.toLowerCase().includes(lowerQuery) || node.model.toLowerCase().includes(lowerQuery)) {
        results.add(node.id);
      }
    }
    this.highlightedNodeIds = results;
    if (this.onHighlightChange) {
      this.onHighlightChange(this.highlightedNodeIds);
    }
    return results;
  }
  searchByType(type) {
    const results = new Set;
    for (const node of this.nodes) {
      if (node.type === type) {
        results.add(node.id);
      }
    }
    return results;
  }
  searchByStatus(status) {
    const results = new Set;
    for (const node of this.nodes) {
      if (node.status === status) {
        results.add(node.id);
      }
    }
    return results;
  }
  applyFilter(criteria) {
    const filtered = new Set;
    for (const node of this.nodes) {
      if (this.matchesCriteria(node, criteria)) {
        filtered.add(node.id);
      }
    }
    this.filteredNodeIds = filtered;
    if (this.onFilterChange) {
      this.onFilterChange(this.filteredNodeIds);
    }
    return filtered;
  }
  matchesCriteria(node, criteria) {
    if (criteria.searchText) {
      const query = criteria.searchText.toLowerCase();
      const matches = node.name.toLowerCase().includes(query) || node.description.toLowerCase().includes(query) || node.type.toLowerCase().includes(query) || node.model.toLowerCase().includes(query);
      if (!matches)
        return false;
    }
    if (criteria.nodeTypes && criteria.nodeTypes.length > 0) {
      if (!criteria.nodeTypes.includes(node.type)) {
        return false;
      }
    }
    if (criteria.statuses && criteria.statuses.length > 0) {
      if (!criteria.statuses.includes(node.status)) {
        return false;
      }
    }
    if (criteria.minLatency !== undefined) {
      if (node.metrics.avgLatency < criteria.minLatency) {
        return false;
      }
    }
    if (criteria.maxLatency !== undefined) {
      if (node.metrics.avgLatency > criteria.maxLatency) {
        return false;
      }
    }
    if (criteria.minSuccessRate !== undefined) {
      if (node.metrics.successRate < criteria.minSuccessRate) {
        return false;
      }
    }
    return true;
  }
  clearFilter() {
    this.filteredNodeIds.clear();
    if (this.onFilterChange) {
      this.onFilterChange(this.filteredNodeIds);
    }
  }
  findPath(fromId, toId) {
    const queue = [fromId];
    const visited = new Set([fromId]);
    const parent = new Map;
    while (queue.length > 0) {
      const current = queue.shift();
      if (current === toId) {
        const path = [];
        let node = toId;
        while (node !== undefined) {
          path.unshift(node);
          node = parent.get(node);
        }
        return path;
      }
      for (const edge of this.edges) {
        if (edge.from === current && !visited.has(edge.to)) {
          visited.add(edge.to);
          parent.set(edge.to, current);
          queue.push(edge.to);
        }
      }
    }
    return [];
  }
  highlightPath(path) {
    this.highlightedNodeIds = new Set(path);
    if (this.onHighlightChange) {
      this.onHighlightChange(this.highlightedNodeIds);
    }
  }
  getNeighbors(nodeId, depth = 1) {
    const neighbors = new Set;
    const queue = [{ id: nodeId, depth: 0 }];
    const visited = new Set([nodeId]);
    while (queue.length > 0) {
      const current = queue.shift();
      if (current.depth >= depth)
        continue;
      for (const edge of this.edges) {
        if (edge.from === current.id && !visited.has(edge.to)) {
          neighbors.add(edge.to);
          visited.add(edge.to);
          queue.push({ id: edge.to, depth: current.depth + 1 });
        }
        if (edge.to === current.id && !visited.has(edge.from)) {
          neighbors.add(edge.from);
          visited.add(edge.from);
          queue.push({ id: edge.from, depth: current.depth + 1 });
        }
      }
    }
    return neighbors;
  }
  focusOnNode(nodeId, includeNeighbors = true) {
    const focused = new Set([nodeId]);
    if (includeNeighbors) {
      const neighbors = this.getNeighbors(nodeId, 1);
      for (const neighbor of neighbors) {
        focused.add(neighbor);
      }
    }
    this.filteredNodeIds = focused;
    if (this.onFilterChange) {
      this.onFilterChange(this.filteredNodeIds);
    }
    return focused;
  }
  getNodesByType() {
    const counts = new Map;
    for (const node of this.nodes) {
      counts.set(node.type, (counts.get(node.type) || 0) + 1);
    }
    return counts;
  }
  getNodesByStatus() {
    const counts = new Map;
    for (const node of this.nodes) {
      counts.set(node.status, (counts.get(node.status) || 0) + 1);
    }
    return counts;
  }
  getAverageMetrics() {
    if (this.nodes.length === 0) {
      return { avgLatency: 0, avgSuccessRate: 0, totalRequests: 0 };
    }
    let totalLatency = 0;
    let totalSuccessRate = 0;
    let totalRequests = 0;
    for (const node of this.nodes) {
      totalLatency += node.metrics.avgLatency;
      totalSuccessRate += node.metrics.successRate;
      totalRequests += node.metrics.totalRequests;
    }
    return {
      avgLatency: totalLatency / this.nodes.length,
      avgSuccessRate: totalSuccessRate / this.nodes.length,
      totalRequests
    };
  }
  onFilter(callback) {
    this.onFilterChange = callback;
  }
  onHighlight(callback) {
    this.onHighlightChange = callback;
  }
  getFilteredNodes() {
    return new Set(this.filteredNodeIds);
  }
  getHighlightedNodes() {
    return new Set(this.highlightedNodeIds);
  }
}

// NetworkGraph/HistoryManager.ts
class HistoryManager {
  undoStack = [];
  redoStack = [];
  maxHistorySize = 50;
  onHistoryChange = null;
  recordNodeAdd(node, addFn, removeFn) {
    const command = {
      type: "node_add",
      execute: addFn,
      undo: removeFn,
      data: { node }
    };
    this.executeAndRecord(command);
  }
  recordNodeRemove(node, removeFn, addFn) {
    const command = {
      type: "node_remove",
      execute: removeFn,
      undo: addFn,
      data: { node }
    };
    this.executeAndRecord(command);
  }
  recordNodeMove(nodeId, oldPos, newPos, moveFn) {
    const command = {
      type: "node_move",
      execute: () => moveFn(newPos),
      undo: () => moveFn(oldPos),
      data: { nodeId, oldPos, newPos }
    };
    this.executeAndRecord(command);
  }
  recordEdgeAdd(edge, addFn, removeFn) {
    const command = {
      type: "edge_add",
      execute: addFn,
      undo: removeFn,
      data: { edge }
    };
    this.executeAndRecord(command);
  }
  recordEdgeRemove(edge, removeFn, addFn) {
    const command = {
      type: "edge_remove",
      execute: removeFn,
      undo: addFn,
      data: { edge }
    };
    this.executeAndRecord(command);
  }
  recordStatusChange(nodeId, oldStatus, newStatus, changeFn) {
    const command = {
      type: "status_change",
      execute: () => changeFn(newStatus),
      undo: () => changeFn(oldStatus),
      data: { nodeId, oldStatus, newStatus }
    };
    this.executeAndRecord(command);
  }
  recordBatchOperation(operations, reverseOperations) {
    const command = {
      type: "batch",
      execute: () => operations.forEach((op) => op()),
      undo: () => reverseOperations.reverse().forEach((op) => op()),
      data: { count: operations.length }
    };
    this.executeAndRecord(command);
  }
  executeAndRecord(command) {
    command.execute();
    this.undoStack.push(command);
    this.redoStack = [];
    if (this.undoStack.length > this.maxHistorySize) {
      this.undoStack.shift();
    }
    this.notifyChange();
  }
  undo() {
    if (this.undoStack.length === 0)
      return false;
    const command = this.undoStack.pop();
    command.undo();
    this.redoStack.push(command);
    this.notifyChange();
    return true;
  }
  redo() {
    if (this.redoStack.length === 0)
      return false;
    const command = this.redoStack.pop();
    command.execute();
    this.undoStack.push(command);
    this.notifyChange();
    return true;
  }
  canUndo() {
    return this.undoStack.length > 0;
  }
  canRedo() {
    return this.redoStack.length > 0;
  }
  clear() {
    this.undoStack = [];
    this.redoStack = [];
    this.notifyChange();
  }
  getUndoStack() {
    return [...this.undoStack];
  }
  getRedoStack() {
    return [...this.redoStack];
  }
  getUndoStackSize() {
    return this.undoStack.length;
  }
  getRedoStackSize() {
    return this.redoStack.length;
  }
  goToState(index) {
    while (this.undoStack.length > index) {
      if (!this.undo())
        break;
    }
    while (this.undoStack.length < index) {
      if (!this.redo())
        break;
    }
  }
  getStateCount() {
    return this.undoStack.length + this.redoStack.length + 1;
  }
  onChange(callback) {
    this.onHistoryChange = callback;
  }
  notifyChange() {
    if (this.onHistoryChange) {
      this.onHistoryChange();
    }
  }
}

// NetworkGraph/MultiSelectHandler.ts
class MultiSelectHandler {
  svg;
  selectedNodes = new Set;
  lassoActive = false;
  lassoPath = null;
  lassoPoints = [];
  rubberBandActive = false;
  rubberBandRect = null;
  rubberBandStart = { x: 0, y: 0 };
  selectionMode = "single";
  onSelectionChange = null;
  constructor(svg) {
    this.svg = svg;
    this.createSelectionElements();
  }
  createSelectionElements() {
    this.lassoPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    this.lassoPath.setAttribute("class", "lasso-path");
    this.lassoPath.setAttribute("fill", "rgba(0, 112, 242, 0.1)");
    this.lassoPath.setAttribute("stroke", "#0070f2");
    this.lassoPath.setAttribute("stroke-width", "2");
    this.lassoPath.setAttribute("stroke-dasharray", "5,5");
    this.lassoPath.setAttribute("visibility", "hidden");
    this.svg.appendChild(this.lassoPath);
    this.rubberBandRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    this.rubberBandRect.setAttribute("class", "rubber-band");
    this.rubberBandRect.setAttribute("fill", "rgba(0, 112, 242, 0.1)");
    this.rubberBandRect.setAttribute("stroke", "#0070f2");
    this.rubberBandRect.setAttribute("stroke-width", "2");
    this.rubberBandRect.setAttribute("stroke-dasharray", "5,5");
    this.rubberBandRect.setAttribute("visibility", "hidden");
    this.svg.appendChild(this.rubberBandRect);
  }
  setMode(mode) {
    this.selectionMode = mode;
    switch (mode) {
      case "lasso":
        this.svg.style.cursor = "crosshair";
        break;
      case "rubberband":
        this.svg.style.cursor = "crosshair";
        break;
      default:
        this.svg.style.cursor = "grab";
    }
  }
  getMode() {
    return this.selectionMode;
  }
  startLasso(point) {
    this.lassoActive = true;
    this.lassoPoints = [point];
    if (this.lassoPath) {
      this.lassoPath.setAttribute("visibility", "visible");
    }
  }
  updateLasso(point) {
    if (!this.lassoActive || !this.lassoPath)
      return;
    this.lassoPoints.push(point);
    if (this.lassoPoints.length > 1) {
      let pathData = `M ${this.lassoPoints[0].x} ${this.lassoPoints[0].y}`;
      for (let i = 1;i < this.lassoPoints.length; i++) {
        pathData += ` L ${this.lassoPoints[i].x} ${this.lassoPoints[i].y}`;
      }
      this.lassoPath.setAttribute("d", pathData);
    }
  }
  endLasso(nodes) {
    if (!this.lassoActive)
      return;
    this.lassoActive = false;
    if (this.lassoPath && this.lassoPoints.length > 2) {
      let pathData = this.lassoPath.getAttribute("d") || "";
      pathData += " Z";
      this.lassoPath.setAttribute("d", pathData);
      const selected = this.getNodesInLasso(nodes);
      this.setSelection(selected);
    }
    if (this.lassoPath) {
      this.lassoPath.setAttribute("visibility", "hidden");
    }
    this.lassoPoints = [];
  }
  getNodesInLasso(nodes) {
    const selected = new Set;
    for (const node of nodes) {
      if (this.isPointInPolygon(node.position, this.lassoPoints)) {
        selected.add(node.id);
      }
    }
    return selected;
  }
  isPointInPolygon(point, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1;i < polygon.length; j = i++) {
      const xi = polygon[i].x, yi = polygon[i].y;
      const xj = polygon[j].x, yj = polygon[j].y;
      const intersect = yi > point.y !== yj > point.y && point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi;
      if (intersect)
        inside = !inside;
    }
    return inside;
  }
  startRubberBand(point) {
    this.rubberBandActive = true;
    this.rubberBandStart = point;
    if (this.rubberBandRect) {
      this.rubberBandRect.setAttribute("visibility", "visible");
      this.rubberBandRect.setAttribute("x", point.x.toString());
      this.rubberBandRect.setAttribute("y", point.y.toString());
      this.rubberBandRect.setAttribute("width", "0");
      this.rubberBandRect.setAttribute("height", "0");
    }
  }
  updateRubberBand(point) {
    if (!this.rubberBandActive || !this.rubberBandRect)
      return;
    const x = Math.min(this.rubberBandStart.x, point.x);
    const y = Math.min(this.rubberBandStart.y, point.y);
    const width = Math.abs(point.x - this.rubberBandStart.x);
    const height = Math.abs(point.y - this.rubberBandStart.y);
    this.rubberBandRect.setAttribute("x", x.toString());
    this.rubberBandRect.setAttribute("y", y.toString());
    this.rubberBandRect.setAttribute("width", width.toString());
    this.rubberBandRect.setAttribute("height", height.toString());
  }
  endRubberBand(nodes, currentPoint) {
    if (!this.rubberBandActive)
      return;
    this.rubberBandActive = false;
    const x1 = Math.min(this.rubberBandStart.x, currentPoint.x);
    const y1 = Math.min(this.rubberBandStart.y, currentPoint.y);
    const x2 = Math.max(this.rubberBandStart.x, currentPoint.x);
    const y2 = Math.max(this.rubberBandStart.y, currentPoint.y);
    const selected = new Set;
    for (const node of nodes) {
      if (node.position.x >= x1 && node.position.x <= x2 && node.position.y >= y1 && node.position.y <= y2) {
        selected.add(node.id);
      }
    }
    this.setSelection(selected);
    if (this.rubberBandRect) {
      this.rubberBandRect.setAttribute("visibility", "hidden");
    }
  }
  setSelection(nodeIds) {
    this.selectedNodes = new Set(nodeIds);
    if (this.onSelectionChange) {
      this.onSelectionChange(this.selectedNodes);
    }
  }
  addToSelection(nodeId) {
    this.selectedNodes.add(nodeId);
    if (this.onSelectionChange) {
      this.onSelectionChange(this.selectedNodes);
    }
  }
  removeFromSelection(nodeId) {
    this.selectedNodes.delete(nodeId);
    if (this.onSelectionChange) {
      this.onSelectionChange(this.selectedNodes);
    }
  }
  toggleSelection(nodeId) {
    if (this.selectedNodes.has(nodeId)) {
      this.removeFromSelection(nodeId);
    } else {
      this.addToSelection(nodeId);
    }
  }
  clearSelection() {
    this.selectedNodes.clear();
    if (this.onSelectionChange) {
      this.onSelectionChange(this.selectedNodes);
    }
  }
  getSelection() {
    return new Set(this.selectedNodes);
  }
  selectAll(nodes) {
    this.selectedNodes = new Set(nodes.map((n) => n.id));
    if (this.onSelectionChange) {
      this.onSelectionChange(this.selectedNodes);
    }
  }
  invertSelection(nodes) {
    const allIds = new Set(nodes.map((n) => n.id));
    const newSelection = new Set;
    for (const id of allIds) {
      if (!this.selectedNodes.has(id)) {
        newSelection.add(id);
      }
    }
    this.setSelection(newSelection);
  }
  onSelection(callback) {
    this.onSelectionChange = callback;
  }
  destroy() {
    if (this.lassoPath && this.lassoPath.parentNode) {
      this.lassoPath.parentNode.removeChild(this.lassoPath);
    }
    if (this.rubberBandRect && this.rubberBandRect.parentNode) {
      this.rubberBandRect.parentNode.removeChild(this.rubberBandRect);
    }
    this.selectedNodes.clear();
  }
}

// NetworkGraph/NetworkGraph.ts
class NetworkGraph {
  container;
  svg;
  defsElement;
  graphContainer;
  state;
  layoutEngine;
  interactionHandler;
  minimap = null;
  searchFilter;
  historyManager;
  multiSelectHandler;
  ws = null;
  eventHandlers = new Map;
  constructor(container) {
    if (typeof container === "string") {
      const element = document.querySelector(container);
      if (!element)
        throw new Error(`Container not found: ${container}`);
      this.container = element;
    } else {
      this.container = container;
    }
    this.state = {
      nodes: new Map,
      edges: new Map,
      groups: new Map,
      selectedNodeId: null,
      hoveredNodeId: null,
      viewport: { ...DEFAULT_VIEWPORT },
      layout: "force-directed" /* ForceDirected */
    };
    this.svg = this.createSVG();
    this.defsElement = this.createDefs();
    this.graphContainer = this.createGraphContainer();
    this.svg.appendChild(this.defsElement);
    this.svg.appendChild(this.graphContainer);
    this.container.appendChild(this.svg);
    this.layoutEngine = new LayoutEngine;
    this.interactionHandler = new InteractionHandler(this.svg, this.state.viewport);
    this.setupEventHandlers();
    this.searchFilter = new SearchFilter;
    this.historyManager = new HistoryManager;
    this.multiSelectHandler = new MultiSelectHandler(this.svg);
    this.setupSubComponentCallbacks();
    this.updateViewportSize();
    window.addEventListener("resize", () => this.updateViewportSize());
  }
  createSVG() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "network-graph");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.style.background = SAP_COLORS.background;
    svg.style.cursor = "grab";
    return svg;
  }
  createDefs() {
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const filter = document.createElementNS("http://www.w3.org/2000/svg", "filter");
    filter.setAttribute("id", "nodeShadow");
    filter.setAttribute("x", "-50%");
    filter.setAttribute("y", "-50%");
    filter.setAttribute("width", "200%");
    filter.setAttribute("height", "200%");
    const feGaussianBlur = document.createElementNS("http://www.w3.org/2000/svg", "feGaussianBlur");
    feGaussianBlur.setAttribute("in", "SourceAlpha");
    feGaussianBlur.setAttribute("stdDeviation", "3");
    const feOffset = document.createElementNS("http://www.w3.org/2000/svg", "feOffset");
    feOffset.setAttribute("dx", "0");
    feOffset.setAttribute("dy", "2");
    feOffset.setAttribute("result", "offsetblur");
    const feComponentTransfer = document.createElementNS("http://www.w3.org/2000/svg", "feComponentTransfer");
    const feFuncA = document.createElementNS("http://www.w3.org/2000/svg", "feFuncA");
    feFuncA.setAttribute("type", "linear");
    feFuncA.setAttribute("slope", "0.3");
    feComponentTransfer.appendChild(feFuncA);
    const feMerge = document.createElementNS("http://www.w3.org/2000/svg", "feMerge");
    const feMergeNode1 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
    const feMergeNode2 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
    feMergeNode2.setAttribute("in", "SourceGraphic");
    feMerge.appendChild(feMergeNode1);
    feMerge.appendChild(feMergeNode2);
    filter.appendChild(feGaussianBlur);
    filter.appendChild(feOffset);
    filter.appendChild(feComponentTransfer);
    filter.appendChild(feMerge);
    defs.appendChild(filter);
    return defs;
  }
  createGraphContainer() {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "graph-container");
    return g;
  }
  setupEventHandlers() {
    this.interactionHandler.on("nodeClick", (event) => {
      this.selectNode(event.node.id);
      this.emit("nodeClick", event);
    });
    this.interactionHandler.on("nodeDrag", (event) => {
      this.emit("nodeDrag", event);
    });
    this.interactionHandler.on("viewportChange", (event) => {
      if (this.minimap) {
        this.minimap.setViewport(event.viewport);
        this.minimap.render();
      }
      this.emit("viewportChange", event);
    });
  }
  setupSubComponentCallbacks() {
    this.historyManager.onChange(() => {
      this.emit("historyChange", {
        canUndo: this.historyManager.canUndo(),
        canRedo: this.historyManager.canRedo()
      });
    });
    this.multiSelectHandler.onSelection((selectedIds) => {
      this.emit("multiSelectionChange", { selectedIds });
    });
    this.searchFilter.onFilter((filteredIds) => {
      this.emit("filterChange", { filteredIds });
    });
    this.searchFilter.onHighlight((highlightedIds) => {
      this.emit("highlightChange", { highlightedIds });
    });
  }
  addNode(config) {
    const node = new GraphNode(config);
    this.state.nodes.set(node.id, node);
    this.graphContainer.appendChild(node.element);
    this.interactionHandler.setNodes(this.state.nodes);
    this.updateSubComponentsData();
    this.emit("nodeAdded", { node });
  }
  addEdge(config) {
    const edge = new GraphEdge(config);
    this.state.edges.set(edge.id, edge);
    const firstNode = this.graphContainer.querySelector(".graph-node");
    if (firstNode) {
      this.graphContainer.insertBefore(edge.element, firstNode);
    } else {
      this.graphContainer.appendChild(edge.element);
    }
    const sourceNode = this.state.nodes.get(edge.from);
    const targetNode = this.state.nodes.get(edge.to);
    if (sourceNode && targetNode) {
      edge.setNodes(sourceNode, targetNode);
    }
    this.interactionHandler.setEdges(this.state.edges);
    this.updateSubComponentsData();
    this.emit("edgeAdded", { edge });
  }
  removeNode(nodeId) {
    const node = this.state.nodes.get(nodeId);
    if (!node)
      return;
    for (const [edgeId, edge] of this.state.edges) {
      if (edge.from === nodeId || edge.to === nodeId) {
        this.removeEdge(edgeId);
      }
    }
    node.destroy();
    this.state.nodes.delete(nodeId);
    this.updateSubComponentsData();
    this.emit("nodeRemoved", { nodeId });
  }
  removeEdge(edgeId) {
    const edge = this.state.edges.get(edgeId);
    if (!edge)
      return;
    edge.destroy();
    this.state.edges.delete(edgeId);
    this.updateSubComponentsData();
    this.emit("edgeRemoved", { edgeId });
  }
  clear() {
    for (const node of this.state.nodes.values()) {
      node.destroy();
    }
    this.state.nodes.clear();
    for (const edge of this.state.edges.values()) {
      edge.destroy();
    }
    this.state.edges.clear();
    this.updateSubComponentsData();
    this.emit("cleared", {});
  }
  updateSubComponentsData() {
    const nodesArray = Array.from(this.state.nodes.values());
    const edgesArray = Array.from(this.state.edges.values());
    this.searchFilter.setNodes(nodesArray);
    this.searchFilter.setEdges(edgesArray);
    if (this.minimap) {
      this.minimap.setNodes(nodesArray);
      this.minimap.setEdges(edgesArray);
      this.minimap.render();
    }
  }
  setLayout(layoutType) {
    this.state.layout = layoutType;
    this.layoutEngine.setConfig({ type: layoutType });
    this.applyLayout();
  }
  applyLayout() {
    this.layoutEngine.setNodes(Array.from(this.state.nodes.values()));
    this.layoutEngine.setEdges(Array.from(this.state.edges.values()));
    this.layoutEngine.start();
    this.emit("layoutApplied", { layout: this.state.layout });
  }
  centerGraph() {
    this.layoutEngine.setNodes(Array.from(this.state.nodes.values()));
    this.layoutEngine.setEdges(Array.from(this.state.edges.values()));
    this.layoutEngine.centerGraph();
  }
  fitToView() {
    this.interactionHandler.fitToView();
  }
  updateNodeStatus(nodeId, status) {
    const node = this.state.nodes.get(nodeId);
    if (node) {
      node.setStatus(status);
      this.emit("nodeStatusChanged", { nodeId, status });
    }
  }
  selectNode(nodeId) {
    if (this.state.selectedNodeId) {
      const prevNode = this.state.nodes.get(this.state.selectedNodeId);
      if (prevNode)
        prevNode.setSelected(false);
    }
    this.state.selectedNodeId = nodeId;
    if (nodeId) {
      const node = this.state.nodes.get(nodeId);
      if (node)
        node.setSelected(true);
    }
    this.emit("selectionChanged", { nodeId });
  }
  getNode(nodeId) {
    return this.state.nodes.get(nodeId);
  }
  getSelectedNode() {
    return this.state.selectedNodeId ? this.state.nodes.get(this.state.selectedNodeId) || null : null;
  }
  updateEdgeStatus(edgeId, status) {
    const edge = this.state.edges.get(edgeId);
    if (edge) {
      edge.setStatus(status);
      this.emit("edgeStatusChanged", { edgeId, status });
    }
  }
  getEdge(edgeId) {
    return this.state.edges.get(edgeId);
  }
  enableMinimap() {
    if (this.minimap)
      return;
    this.minimap = new Minimap(this.container, this.state.viewport);
    this.minimap.setNodes(Array.from(this.state.nodes.values()));
    this.minimap.setEdges(Array.from(this.state.edges.values()));
    this.minimap.onViewportChange((viewport) => {
      this.interactionHandler.setViewport(viewport);
      this.emit("viewportChange", { viewport });
    });
    this.minimap.show();
    this.minimap.render();
  }
  disableMinimap() {
    if (!this.minimap)
      return;
    this.minimap.destroy();
    this.minimap = null;
  }
  search(query) {
    return this.searchFilter.search(query);
  }
  filter(criteria) {
    this.searchFilter.applyFilter(criteria);
  }
  clearFilter() {
    this.searchFilter.clearFilter();
  }
  undo() {
    this.historyManager.undo();
  }
  redo() {
    this.historyManager.redo();
  }
  canUndo() {
    return this.historyManager.canUndo();
  }
  canRedo() {
    return this.historyManager.canRedo();
  }
  getMultiSelectHandler() {
    return this.multiSelectHandler;
  }
  getHistoryManager() {
    return this.historyManager;
  }
  getSearchFilter() {
    return this.searchFilter;
  }
  zoomIn() {
    this.interactionHandler.zoomIn();
  }
  zoomOut() {
    this.interactionHandler.zoomOut();
  }
  resetZoom() {
    this.interactionHandler.resetZoom();
  }
  getViewport() {
    return { ...this.state.viewport };
  }
  async loadFromAPI(apiUrl) {
    try {
      const response = await fetch(apiUrl);
      if (!response.ok)
        throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      this.loadData(data);
      this.emit("dataLoaded", { source: apiUrl });
    } catch (error) {
      console.error("Failed to load from API:", error);
      this.emit("error", { error, source: "loadFromAPI" });
    }
  }
  loadData(data) {
    this.clear();
    if (data.agents) {
      for (const agentData of data.agents) {
        this.addNode({
          id: agentData.id,
          name: agentData.name,
          description: agentData.description,
          type: agentData.type,
          status: this.mapStatus(agentData.status),
          model: agentData.model_id,
          metrics: {
            totalRequests: agentData.total_requests || 0,
            avgLatency: agentData.avg_latency || 0,
            successRate: agentData.success_rate || 0
          }
        });
      }
    }
    if (data.agents) {
      for (const agentData of data.agents) {
        if (agentData.next_agents) {
          for (const targetId of agentData.next_agents) {
            const edgeId = `${agentData.id}-${targetId}`;
            this.addEdge({
              id: edgeId,
              from: agentData.id,
              to: targetId,
              status: "Active" /* Active */
            });
          }
        }
      }
    }
    setTimeout(() => {
      this.applyLayout();
      setTimeout(() => this.fitToView(), 1000);
    }, 100);
  }
  mapStatus(status) {
    const statusMap = {
      healthy: "Success" /* Success */,
      warning: "Warning" /* Warning */,
      error: "Error" /* Error */,
      running: "Running" /* Running */,
      busy: "Warning" /* Warning */
    };
    return statusMap[status?.toLowerCase()] || "None" /* None */;
  }
  connectWebSocket(url) {
    if (this.ws) {
      this.ws.close();
    }
    this.ws = new WebSocket(url);
    this.ws.onopen = () => {
      console.log("✅ NetworkGraph WebSocket connected");
      this.emit("wsConnected", {});
    };
    this.ws.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        this.handleWSUpdate(update);
      } catch (error) {
        console.error("Failed to parse WS message:", error);
      }
    };
    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.emit("wsError", { error });
    };
    this.ws.onclose = () => {
      console.log("WebSocket closed");
      this.emit("wsDisconnected", {});
    };
  }
  handleWSUpdate(update) {
    if (update.type === "agent_status") {
      this.updateNodeStatus(update.agent_id, this.mapStatus(update.status));
    } else if (update.type === "workflow_step") {
      const edgeId = `${update.from}-${update.to}`;
      const edge = this.state.edges.get(edgeId);
      if (edge) {
        edge.flash();
      }
    }
    this.emit("wsUpdate", update);
  }
  disconnectWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
  on(event, callback) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(callback);
  }
  off(event, callback) {
    if (!callback) {
      this.eventHandlers.delete(event);
    } else {
      const handlers = this.eventHandlers.get(event);
      if (handlers) {
        const index = handlers.indexOf(callback);
        if (index !== -1)
          handlers.splice(index, 1);
      }
    }
  }
  emit(event, data) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      for (const handler of handlers) {
        handler(data);
      }
    }
  }
  exportData() {
    return {
      nodes: Array.from(this.state.nodes.values()).map((n) => n.toJSON()),
      edges: Array.from(this.state.edges.values()).map((e) => e.toJSON()),
      viewport: this.state.viewport,
      layout: this.state.layout
    };
  }
  exportImage() {
    const svgData = new XMLSerializer().serializeToString(this.svg);
    return "data:image/svg+xml;base64," + btoa(svgData);
  }
  updateViewportSize() {
    const rect = this.container.getBoundingClientRect();
    this.state.viewport.width = rect.width;
    this.state.viewport.height = rect.height;
    this.interactionHandler.setViewport(this.state.viewport);
  }
  getStats() {
    return {
      nodeCount: this.state.nodes.size,
      edgeCount: this.state.edges.size,
      viewport: this.state.viewport,
      layout: this.state.layout
    };
  }
  destroy() {
    this.disconnectWebSocket();
    this.clear();
    this.interactionHandler.destroy();
    this.layoutEngine.stop();
    if (this.minimap) {
      this.minimap.destroy();
      this.minimap = null;
    }
    this.multiSelectHandler.destroy();
    if (this.svg.parentNode) {
      this.svg.parentNode.removeChild(this.svg);
    }
    this.eventHandlers.clear();
  }
}
if (typeof window !== "undefined") {
  window.NetworkGraph = NetworkGraph;
}
export {
  NetworkGraph
};

//# debugId=591FF0134B72AA8264756E2164756E21
