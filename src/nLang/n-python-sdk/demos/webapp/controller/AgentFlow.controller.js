 sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast"
], function (Controller, MessageToast) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.AgentFlow", {

        onInit: function () {
            // Initialize state
            this._isPlaying = false;
            this._simulationSpeed = 1;
            this._executionCount = 0;
            this._tokens = [];

            // Load components after view is rendered with delay
            this.getView().addEventDelegate({
                onAfterRendering: function () {
                    setTimeout(function () {
                        this._initializeComponents();
                    }.bind(this), 300);
                }.bind(this)
            });
        },

        _initializeComponents: function () {
            // Initialize NetworkGraph
            this._initNetworkGraph();

            // Initialize ProcessFlow
            this._initProcessFlow();

            // Load sample Petri Net
            this._loadSamplePetriNet();
        },

        _initNetworkGraph: function () {
            try {
                // Load NetworkGraph component dynamically
                const container = document.getElementById('networkGraphContainer');
                if (!container) {
                    console.error("NetworkGraph container not found");
                    return;
                }

                // Check if NetworkGraph is available
                if (typeof NetworkGraph !== 'undefined') {
                    this._networkGraph = new NetworkGraph(container);
                    console.log("NetworkGraph initialized");
                } else {
                    // Fallback: Create simple SVG visualization
                    this._createFallbackNetworkGraph(container);
                }
            } catch (error) {
                console.error("Error initializing NetworkGraph:", error);
                this._createFallbackNetworkGraph(document.getElementById('networkGraphContainer'));
            }
        },

        _createFallbackNetworkGraph: function (container) {
            // Create simple SVG-based Petri Net visualization
            container.innerHTML = `
                <svg width="100%" height="100%" viewBox="0 0 800 500">
                    <!-- Places (circles) -->
                    <circle cx="150" cy="100" r="40" fill="#0a6ed1" stroke="#000" stroke-width="2"/>
                    <text x="150" y="105" text-anchor="middle" fill="white" font-size="14">Ready</text>
                    <text x="150" y="125" text-anchor="middle" fill="white" font-size="12">●●●</text>
                    
                    <circle cx="400" cy="100" r="40" fill="#0a6ed1" stroke="#000" stroke-width="2"/>
                    <text x="400" y="105" text-anchor="middle" fill="white" font-size="14">Running</text>
                    
                    <circle cx="650" cy="100" r="40" fill="#0a6ed1" stroke="#000" stroke-width="2"/>
                    <text x="650" y="105" text-anchor="middle" fill="white" font-size="14">Complete</text>
                    
                    <circle cx="150" cy="300" r="40" fill="#0a6ed1" stroke="#000" stroke-width="2"/>
                    <text x="150" y="305" text-anchor="middle" fill="white" font-size="14">Queue</text>
                    
                    <circle cx="400" cy="300" r="40" fill="#0a6ed1" stroke="#000" stroke-width="2"/>
                    <text x="400" y="305" text-anchor="middle" fill="white" font-size="14">Process</text>
                    
                    <!-- Transitions (rectangles) -->
                    <rect x="250" y="80" width="60" height="40" fill="#bb0000" stroke="#000" stroke-width="2" rx="5"/>
                    <text x="280" y="105" text-anchor="middle" fill="white" font-size="12">Execute</text>
                    
                    <rect x="520" y="80" width="60" height="40" fill="#bb0000" stroke="#000" stroke-width="2" rx="5"/>
                    <text x="550" y="105" text-anchor="middle" fill="white" font-size="12">Finish</text>
                    
                    <rect x="250" y="280" width="60" height="40" fill="#bb0000" stroke="#000" stroke-width="2" rx="5"/>
                    <text x="280" y="305" text-anchor="middle" fill="white" font-size="12">Start</text>
                    
                    <!-- Arcs (arrows) -->
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
                        </marker>
                    </defs>
                    
                    <line x1="190" y1="100" x2="250" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <line x1="310" y1="100" x2="360" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <line x1="440" y1="100" x2="520" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <line x1="580" y1="100" x2="610" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    
                    <line x1="190" y1="300" x2="250" y2="300" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <line x1="310" y1="300" x2="360" y2="300" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    
                    <!-- Feedback arcs -->
                    <path d="M 650 140 Q 650 250, 440 300" stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
                    <path d="M 150 260 Q 100 200, 150 140" stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
                    
                    <!-- Legend -->
                    <text x="20" y="450" font-size="12" fill="#666">● Places (States)</text>
                    <text x="200" y="450" font-size="12" fill="#666">▪ Transitions (Actions)</text>
                    <text x="400" y="450" font-size="12" fill="#666">→ Token Flow</text>
                </svg>
            `;
        },

        _initProcessFlow: function () {
            const container = document.getElementById('processFlowContainer');
            if (!container) {
                console.error("ProcessFlow container not found");
                return;
            }

            // Create simple timeline visualization
            container.innerHTML = `
                <div id="processTimeline" style="padding: 20px;">
                    <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #0a6ed1; background: #f5f5f5;">
                        <strong>Step 1:</strong> Initialize (0.0s)<br/>
                        <span style="color: #666;">3 tokens placed in Ready state</span>
                    </div>
                    <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #999; background: #fafafa;">
                        <strong>Step 2:</strong> Ready → Execute<br/>
                        <span style="color: #666;">Waiting for simulation...</span>
                    </div>
                    <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #999; background: #fafafa;">
                        <strong>Step 3:</strong> Execute → Running<br/>
                        <span style="color: #666;">Waiting for simulation...</span>
                    </div>
                    <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #999; background: #fafafa;">
                        <strong>Step 4:</strong> Running → Complete<br/>
                        <span style="color: #666;">Waiting for simulation...</span>
                    </div>
                </div>
            `;
        },

        _loadSamplePetriNet: function () {
            // Sample Petri Net data
            const petriNetData = {
                nodes: [
                    { id: 'p1', type: 'place', label: 'Ready', tokens: 3, x: 150, y: 100 },
                    { id: 't1', type: 'transition', label: 'Execute', x: 280, y: 100 },
                    { id: 'p2', type: 'place', label: 'Running', tokens: 0, x: 400, y: 100 },
                    { id: 't2', type: 'transition', label: 'Finish', x: 550, y: 100 },
                    { id: 'p3', type: 'place', label: 'Complete', tokens: 0, x: 650, y: 100 },
                    { id: 'p4', type: 'place', label: 'Queue', tokens: 0, x: 150, y: 300 },
                    { id: 't3', type: 'transition', label: 'Start', x: 280, y: 300 },
                    { id: 'p5', type: 'place', label: 'Process', tokens: 0, x: 400, y: 300 }
                ],
                edges: [
                    { from: 'p1', to: 't1' },
                    { from: 't1', to: 'p2' },
                    { from: 'p2', to: 't2' },
                    { from: 't2', to: 'p3' },
                    { from: 'p4', to: 't3' },
                    { from: 't3', to: 'p5' },
                    { from: 'p3', to: 'p5' },
                    { from: 'p5', to: 'p1' }
                ]
            };

            // Update statistics
            this._updateStatistics(petriNetData);

            // If NetworkGraph is available, load the data
            if (this._networkGraph && typeof this._networkGraph.loadData === 'function') {
                this._networkGraph.loadData(petriNetData);
                this._networkGraph.setLayout('hierarchical');
            }
        },

        _updateStatistics: function (data) {
            const places = data.nodes.filter(n => n.type === 'place').length;
            const transitions = data.nodes.filter(n => n.type === 'transition').length;
            const tokens = data.nodes.filter(n => n.type === 'place').reduce((sum, n) => sum + (n.tokens || 0), 0);

            this.byId("placesCount").setText(places.toString());
            this.byId("transitionsCount").setText(transitions.toString());
            this.byId("tokensCount").setText(tokens.toString());
            this.byId("executionsCount").setText(this._executionCount.toString());
        },

        onViewModeChange: function (oEvent) {
            const selectedKey = oEvent.getParameter("key");
            const networkPane = this.byId("networkPane");
            const processPane = this.byId("processPane");

            switch (selectedKey) {
                case "network":
                    networkPane.setVisible(true);
                    processPane.setVisible(false);
                    break;
                case "process":
                    networkPane.setVisible(false);
                    processPane.setVisible(true);
                    break;
                case "split":
                    networkPane.setVisible(true);
                    processPane.setVisible(true);
                    break;
            }
        },

        onPlaySimulation: function () {
            this._isPlaying = true;
            this.byId("playButton").setEnabled(false);
            this.byId("pauseButton").setEnabled(true);
            
            MessageToast.show("Simulation started");
            this._runSimulation();
        },

        onPauseSimulation: function () {
            this._isPlaying = false;
            this.byId("playButton").setEnabled(true);
            this.byId("pauseButton").setEnabled(false);
            
            if (this._simulationTimer) {
                clearTimeout(this._simulationTimer);
            }
            
            MessageToast.show("Simulation paused");
        },

        onResetSimulation: function () {
            this._isPlaying = false;
            this._executionCount = 0;
            
            this.byId("playButton").setEnabled(true);
            this.byId("pauseButton").setEnabled(false);
            
            if (this._simulationTimer) {
                clearTimeout(this._simulationTimer);
            }
            
            this._loadSamplePetriNet();
            this._initProcessFlow();
            
            MessageToast.show("Simulation reset");
        },

        onSpeedChange: function (oEvent) {
            this._simulationSpeed = parseFloat(oEvent.getParameter("value"));
        },

        onExport: function () {
            MessageToast.show("Exporting Petri Net data...");
            
            const data = {
                type: "PetriNet",
                timestamp: new Date().toISOString(),
                executions: this._executionCount,
                speed: this._simulationSpeed
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'petri-net-export.json';
            a.click();
            URL.revokeObjectURL(url);
        },

        _runSimulation: function () {
            if (!this._isPlaying) return;

            this._executionCount++;
            this.byId("executionsCount").setText(this._executionCount.toString());

            // Add execution step to timeline
            const timeline = document.getElementById('processTimeline');
            if (timeline) {
                const step = document.createElement('div');
                step.style.cssText = 'margin-bottom: 15px; padding: 10px; border-left: 3px solid #2b7d2b; background: #f0f8f0;';
                step.innerHTML = `<strong>Execution ${this._executionCount}:</strong> Token moved (${new Date().toLocaleTimeString()})<br/><span style="color: #666;">Transition fired successfully</span>`;
                timeline.insertBefore(step, timeline.firstChild);
            }

            // Schedule next execution
            const delay = 2000 / this._simulationSpeed;
            this._simulationTimer = setTimeout(() => {
                this._runSimulation();
            }, delay);
        }

    });
});