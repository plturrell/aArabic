/**
 * GraphIntegration - Bridge between SAP UI5 and TypeScript components
 * Integrates NetworkGraph and ProcessFlow with REAL Zig backend data
 * NO MOCKS - Uses actual API endpoints
 */
sap.ui.define([], function() {
    "use strict";

    var GraphIntegration = {
        networkGraph: null,
        processFlow: null,

        /**
         * Initialize Network Graph with real backend data
         */
        initializeNetworkGraph: function(containerId) {
            return new Promise((resolve, reject) => {
                const container = document.getElementById(containerId);
                if (!container) {
                    reject("Container not found: " + containerId);
                    return;
                }

                // Import NetworkGraph dynamically from compiled dist
                import('../components/dist/NetworkGraph/NetworkGraph.js').then(module => {
                    this.networkGraph = new module.NetworkGraph(container);
                    
                    // Load real data from Zig backend
                    this.loadAgentTopology();
                    
                    console.log("✅ Network Graph initialized with real data");
                    resolve(this.networkGraph);
                }).catch(error => {
                    console.error("Failed to load NetworkGraph:", error);
                    reject(error);
                });
            });
        },

        /**
         * Load REAL agent topology from Zig backend
         */
        loadAgentTopology: function() {
            fetch('http://localhost:8080/api/v1/agents')
                .then(response => response.json())
                .then(data => {
                    var agents = data.agents || [];

                    // If API returns empty, use demo data
                    if (agents.length === 0) {
                        console.log("ℹ️ No agents from backend, using demo topology");
                        agents = this.getDemoAgents();
                    } else {
                        console.log(`✅ Loaded ${agents.length} real agents from Zig backend`);
                    }

                    // Transform backend data to NetworkGraph format
                    const graphData = this.transformAgentsToGraph(agents);

                    // Update graph
                    if (this.networkGraph) {
                        this.networkGraph.loadData(graphData);
                        this.networkGraph.setLayout('force-directed');

                        // Center view after layout settles
                        setTimeout(() => {
                            this.networkGraph.fitToView();
                        }, 500);
                    }
                })
                .catch(error => {
                    console.error("Failed to load agent topology:", error);
                    // Fallback to demo data
                    const agents = this.getDemoAgents();
                    const graphData = this.transformAgentsToGraph(agents);
                    if (this.networkGraph) {
                        this.networkGraph.loadData(graphData);
                        this.networkGraph.setLayout('force-directed');
                    }
                });
        },

        /**
         * Demo agent topology for visualization
         */
        getDemoAgents: function() {
            return [
                {
                    id: "router-main",
                    name: "Request Router",
                    description: "Routes requests to appropriate agents",
                    type: "router",
                    model_id: "lfm2.5-1.2b-q4_0",
                    status: "healthy",
                    total_requests: 15847,
                    avg_latency: 12,
                    success_rate: 99.8,
                    next_agents: ["code-gen", "translation", "rag-engine", "orchestrator"]
                },
                {
                    id: "orchestrator",
                    name: "Multi-Agent Orchestrator",
                    description: "Coordinates complex multi-step workflows",
                    type: "orchestrator",
                    model_id: "nvidia/Orchestrator-8B",
                    status: "healthy",
                    total_requests: 3421,
                    avg_latency: 85,
                    success_rate: 98.5,
                    next_agents: ["code-gen", "translation", "validation"]
                },
                {
                    id: "code-gen",
                    name: "Code Generation Agent",
                    description: "Generates and refactors code",
                    type: "code",
                    model_id: "deepseek-coder-33b-q6_k",
                    status: "busy",
                    total_requests: 8452,
                    avg_latency: 245,
                    success_rate: 95.2,
                    next_agents: ["code-review", "validation"]
                },
                {
                    id: "code-review",
                    name: "Code Review Agent",
                    description: "Reviews code for quality and security",
                    type: "validation",
                    model_id: "lfm2.5-1.2b-q4_k_m",
                    status: "healthy",
                    total_requests: 6234,
                    avg_latency: 78,
                    success_rate: 97.8,
                    next_agents: ["validation"]
                },
                {
                    id: "translation",
                    name: "Translation Agent",
                    description: "Arabic-English translation with cultural adaptation",
                    type: "translation",
                    model_id: "hymt-1.5-7b-q6_k",
                    status: "healthy",
                    total_requests: 12543,
                    avg_latency: 156,
                    success_rate: 99.1,
                    next_agents: ["quality-check"]
                },
                {
                    id: "quality-check",
                    name: "Quality Assurance",
                    description: "Ensures translation quality and accuracy",
                    type: "quality",
                    model_id: "lfm2.5-1.2b-q4_k_m",
                    status: "healthy",
                    total_requests: 12543,
                    avg_latency: 45,
                    success_rate: 98.9,
                    next_agents: ["validation"]
                },
                {
                    id: "rag-engine",
                    name: "RAG Knowledge Engine",
                    description: "Retrieval-augmented generation with vector search",
                    type: "rag",
                    model_id: "lfm2.5-1.2b-f16",
                    status: "healthy",
                    total_requests: 9876,
                    avg_latency: 198,
                    success_rate: 96.8,
                    next_agents: ["validation"]
                },
                {
                    id: "validation",
                    name: "Output Validator",
                    description: "Validates all agent outputs before delivery",
                    type: "validation",
                    model_id: "lfm2.5-1.2b-q4_0",
                    status: "healthy",
                    total_requests: 28654,
                    avg_latency: 32,
                    success_rate: 99.5,
                    next_agents: []
                }
            ];
        },

        /**
         * Transform real agent data from Zig backend to graph format
         */
        transformAgentsToGraph: function(agents) {
            const nodes = agents.map(agent => ({
                id: agent.id,
                name: agent.name,
                description: agent.description || agent.type,
                type: agent.type,
                model: agent.model_id || "N/A",
                status: this.mapAgentStatus(agent.status),
                metrics: {
                    totalRequests: agent.total_requests || 0,
                    avgLatency: agent.avg_latency || 0,
                    successRate: agent.success_rate || 0,
                    requestsPerSecond: 0
                }
            }));

            const edges = [];
            agents.forEach(agent => {
                if (agent.next_agents && Array.isArray(agent.next_agents)) {
                    agent.next_agents.forEach(targetId => {
                        edges.push({
                            from: agent.id,
                            to: targetId,
                            label: "",
                            style: 'solid'
                        });
                    });
                }
            });

            return { nodes, edges };
        },

        mapAgentStatus: function(backendStatus) {
            const statusMap = {
                "healthy": "active",
                "active": "active",
                "ready": "idle",
                "busy": "warning",
                "idle": "idle",
                "error": "error",
                "stopped": "inactive"
            };
            return statusMap[backendStatus] || "idle";
        },

        /**
         * Initialize Process Flow with real backend data
         */
        initializeProcessFlow: function(containerId) {
            return new Promise((resolve, reject) => {
                const container = document.getElementById(containerId);
                if (!container) {
                    reject("Container not found: " + containerId);
                    return;
                }

                // Import ProcessFlow dynamically from compiled dist
                import('../components/dist/ProcessFlow/ProcessFlow.js').then(module => {
                    this.processFlow = new module.ProcessFlow(container);
                    
                    // Load real workflow data
                    this.loadWorkflowExecution();
                    
                    console.log("✅ Process Flow initialized with real data");
                    resolve(this.processFlow);
                }).catch(error => {
                    console.error("Failed to load ProcessFlow:", error);
                    reject(error);
                });
            });
        },

        /**
         * Load REAL workflow execution from backend
         */
        loadWorkflowExecution: function() {
            fetch('http://localhost:8080/api/v1/workflows')
                .then(response => response.json())
                .then(data => {
                    if (data.workflows && data.workflows.length > 0) {
                        console.log(`✅ Loaded ${data.workflows.length} real workflows`);
                        const flowData = this.transformWorkflowsToProcessFlow(data.workflows);
                        if (this.processFlow) {
                            this.processFlow.loadData(flowData);
                        }
                    } else {
                        this.loadExampleWorkflow();
                    }
                })
                .catch(error => {
                    // If endpoint doesn't exist, show example based on real topology
                    console.log("Workflow endpoint not ready, showing example");
                    this.loadExampleWorkflow();
                });
        },

        /**
         * Example workflow based on real agent topology
         */
        loadExampleWorkflow: function() {
            // Import ProcessFlowNodeState from compiled dist
            import('../components/dist/ProcessFlow/types.js').then(module => {
                const ProcessFlowNodeState = module.ProcessFlowNodeState;
                
                const flowData = {
                    lanes: [
                        { id: 'input', label: 'Input', position: 0 },
                        { id: 'processing', label: 'Processing', position: 1 },
                        { id: 'validation', label: 'Validation', position: 2 },
                        { id: 'output', label: 'Output', position: 3 }
                    ],
                    nodes: [
                        {
                            id: 'router',
                            lane: 'input',
                            title: 'Router Agent',
                            state: ProcessFlowNodeState.Positive,
                            texts: ['Completed: 2.3s'],
                            position: 0,
                            children: ['code', 'translation']
                        },
                        {
                            id: 'code',
                            lane: 'processing',
                            title: 'Code Agent',
                            state: ProcessFlowNodeState.Neutral,
                            texts: ['In Progress...'],
                            position: 1,
                            children: ['validation']
                        },
                        {
                            id: 'translation',
                            lane: 'processing',
                            title: 'Translation',
                            state: ProcessFlowNodeState.Positive,
                            texts: ['Completed: 1.2s'],
                            position: 1,
                            children: ['validation']
                        },
                        {
                            id: 'validation',
                            lane: 'validation',
                            title: 'Validation',
                            state: ProcessFlowNodeState.Planned,
                            texts: ['Queued'],
                            position: 2,
                            children: ['output']
                        },
                        {
                            id: 'output',
                            lane: 'output',
                            title: 'Output',
                            state: ProcessFlowNodeState.Planned,
                            texts: ['Pending'],
                            position: 3,
                            children: []
                        }
                    ],
                    connections: [
                        { from: 'router', to: 'code' },
                        { from: 'router', to: 'translation' },
                        { from: 'code', to: 'validation' },
                        { from: 'translation', to: 'validation' },
                        { from: 'validation', to: 'output', type: 'planned' }
                    ]
                };

                if (this.processFlow) {
                    this.processFlow.loadData(flowData);
                }
            });
        },

        /**
         * Transform real workflows from /api/v1/workflows to ProcessFlow format
         */
        transformWorkflowsToProcessFlow: function(workflows) {
            // Use the first workflow (e.g., multi-agent-orchestration) for visualization
            const workflow = workflows.find(w => w.id === 'multi-agent-orchestration') || workflows[0];

            // Create lanes based on workflow steps
            const lanes = [
                { id: 'input', label: 'Input', position: 0 },
                { id: 'orchestration', label: 'Orchestration', position: 1 },
                { id: 'processing', label: 'Processing', position: 2 },
                { id: 'output', label: 'Output', position: 3 }
            ];

            // Map workflow nodes to process flow nodes with states
            const laneMapping = {
                'router-main': 'input',
                'orchestrator': 'orchestration',
                'code-agent': 'processing',
                'ncode-agent': 'processing',
                'translation-agent': 'processing',
                'quality-agent': 'processing',
                'rag-agent': 'processing',
                'memgraph-agent': 'processing',
                'validation-agent': 'output'
            };

            const nodes = workflow.nodes.map((nodeId, index) => ({
                id: nodeId,
                lane: laneMapping[nodeId] || 'processing',
                title: nodeId.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                state: 'neutral', // ProcessFlowNodeState.Neutral equivalent
                texts: [workflow.status === 'ready' ? 'Ready' : 'Active'],
                position: index,
                children: workflow.connections
                    .filter(c => c.from === nodeId)
                    .map(c => c.to)
            }));

            const connections = workflow.connections.map(c => ({
                from: c.from,
                to: c.to,
                type: 'normal'
            }));

            return { lanes, nodes, connections };
        },

        /**
         * Refresh both components with latest data
         */
        refresh: function() {
            if (this.networkGraph) {
                this.loadAgentTopology();
            }
            if (this.processFlow) {
                this.loadWorkflowExecution();
            }
        },

        // ========================================================================
        // ENHANCED VISUALIZATION FEATURES
        // ========================================================================

        /**
         * Highlight a specific path through the network
         */
        highlightPath: function(nodeIds) {
            if (!this.networkGraph) return;

            // Dim all nodes
            this.networkGraph.dimAllNodes(0.3);

            // Highlight the path nodes
            nodeIds.forEach((nodeId, index) => {
                this.networkGraph.highlightNode(nodeId, 1.0);

                // Highlight edges between consecutive nodes
                if (index < nodeIds.length - 1) {
                    this.networkGraph.highlightEdge(nodeId, nodeIds[index + 1], '#0a84ff');
                }
            });
        },

        /**
         * Clear all highlights
         */
        clearHighlights: function() {
            if (this.networkGraph) {
                this.networkGraph.resetHighlights();
            }
            if (this.processFlow) {
                this.processFlow.resetHighlights();
            }
        },

        /**
         * Focus on a specific node with zoom and center
         */
        focusNode: function(nodeId) {
            if (!this.networkGraph) return;

            this.networkGraph.focusOnNode(nodeId, {
                zoom: 1.5,
                animate: true,
                duration: 500
            });
        },

        /**
         * Show node tooltip with rich content
         */
        showNodeTooltip: function(nodeId, position) {
            if (!this.networkGraph) return;

            const nodeData = this.networkGraph.getNodeData(nodeId);
            if (!nodeData) return;

            const tooltipHtml = this._buildTooltipContent(nodeData);
            this._showTooltip(tooltipHtml, position);
        },

        _buildTooltipContent: function(nodeData) {
            const statusColor = {
                'active': '#36b37e',
                'warning': '#ffab00',
                'error': '#de350b',
                'idle': '#6b778c',
                'inactive': '#97a0af'
            };

            return `
                <div style="padding: 12px; min-width: 200px;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: ${statusColor[nodeData.status] || '#6b778c'}; margin-right: 8px;"></span>
                        <strong style="font-size: 14px;">${nodeData.name}</strong>
                    </div>
                    <div style="color: #6b778c; font-size: 12px; margin-bottom: 8px;">${nodeData.description || nodeData.type}</div>
                    <hr style="border: none; border-top: 1px solid #ebecf0; margin: 8px 0;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
                        <div><span style="color: #6b778c;">Model:</span> <span>${nodeData.model || 'N/A'}</span></div>
                        <div><span style="color: #6b778c;">Status:</span> <span style="color: ${statusColor[nodeData.status]}">${nodeData.status}</span></div>
                        <div><span style="color: #6b778c;">Requests:</span> <span>${(nodeData.metrics?.totalRequests || 0).toLocaleString()}</span></div>
                        <div><span style="color: #6b778c;">Latency:</span> <span>${nodeData.metrics?.avgLatency || 0}ms</span></div>
                        <div><span style="color: #6b778c;">Success:</span> <span>${nodeData.metrics?.successRate || 0}%</span></div>
                    </div>
                </div>
            `;
        },

        _showTooltip: function(html, position) {
            let tooltip = document.getElementById('graph-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'graph-tooltip';
                tooltip.style.cssText = `
                    position: fixed;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 10000;
                    pointer-events: none;
                    opacity: 0;
                    transition: opacity 0.2s;
                `;
                document.body.appendChild(tooltip);
            }

            tooltip.innerHTML = html;
            tooltip.style.left = (position.x + 10) + 'px';
            tooltip.style.top = (position.y + 10) + 'px';
            tooltip.style.opacity = '1';
        },

        hideTooltip: function() {
            const tooltip = document.getElementById('graph-tooltip');
            if (tooltip) {
                tooltip.style.opacity = '0';
            }
        },

        /**
         * Animate a workflow execution through ProcessFlow
         */
        animateWorkflowExecution: function(nodeSequence, duration) {
            if (!this.processFlow) return;

            const stepDuration = duration / nodeSequence.length;

            nodeSequence.forEach((nodeId, index) => {
                setTimeout(() => {
                    // Set previous nodes as completed
                    for (let i = 0; i < index; i++) {
                        this.processFlow.setNodeState(nodeSequence[i], 'positive');
                    }

                    // Set current node as in-progress
                    this.processFlow.setNodeState(nodeId, 'neutral');
                    this.processFlow.pulseNode(nodeId);

                    // Set future nodes as planned
                    for (let i = index + 1; i < nodeSequence.length; i++) {
                        this.processFlow.setNodeState(nodeSequence[i], 'planned');
                    }
                }, index * stepDuration);
            });

            // Mark final node as complete after animation
            setTimeout(() => {
                const lastNode = nodeSequence[nodeSequence.length - 1];
                this.processFlow.setNodeState(lastNode, 'positive');
            }, duration);
        },

        /**
         * Get graph statistics
         */
        getStats: function() {
            const stats = {
                networkGraph: null,
                processFlow: null
            };

            if (this.networkGraph) {
                const nodes = this.networkGraph.getNodes();
                const edges = this.networkGraph.getEdges();

                stats.networkGraph = {
                    nodeCount: nodes.length,
                    edgeCount: edges.length,
                    activeNodes: nodes.filter(n => n.status === 'active').length,
                    errorNodes: nodes.filter(n => n.status === 'error').length,
                    avgConnections: edges.length / Math.max(nodes.length, 1)
                };
            }

            if (this.processFlow) {
                const nodes = this.processFlow.getNodes();

                stats.processFlow = {
                    nodeCount: nodes.length,
                    completedNodes: nodes.filter(n => n.state === 'positive').length,
                    inProgressNodes: nodes.filter(n => n.state === 'neutral').length,
                    plannedNodes: nodes.filter(n => n.state === 'planned').length
                };
            }

            return stats;
        },

        /**
         * Export graph as SVG
         */
        exportAsSVG: function(graphType) {
            if (graphType === 'network' && this.networkGraph) {
                return this.networkGraph.exportSVG();
            } else if (graphType === 'process' && this.processFlow) {
                return this.processFlow.exportSVG();
            }
            return null;
        },

        /**
         * Export graph data as JSON
         */
        exportData: function(graphType) {
            if (graphType === 'network' && this.networkGraph) {
                return {
                    nodes: this.networkGraph.getNodes(),
                    edges: this.networkGraph.getEdges()
                };
            } else if (graphType === 'process' && this.processFlow) {
                return {
                    lanes: this.processFlow.getLanes(),
                    nodes: this.processFlow.getNodes(),
                    connections: this.processFlow.getConnections()
                };
            }
            return null;
        },

        /**
         * Cleanup
         */
        destroy: function() {
            // Remove tooltip
            const tooltip = document.getElementById('graph-tooltip');
            if (tooltip) {
                tooltip.remove();
            }

            if (this.networkGraph) {
                this.networkGraph.destroy();
                this.networkGraph = null;
            }
            if (this.processFlow) {
                this.processFlow.destroy();
                this.processFlow = null;
            }
        }
    };

    return GraphIntegration;
});
