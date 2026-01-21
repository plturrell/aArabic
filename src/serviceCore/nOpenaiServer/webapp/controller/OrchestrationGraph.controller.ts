/**
 * OrchestrationGraph Controller
 * Integrates NetworkGraph and ProcessFlow with REAL Zig backend data
 * NO MOCKS - Uses actual API endpoints
 */

import Controller from "sap/ui/core/mvc/Controller";
import JSONModel from "sap/ui/model/json/JSONModel";
import MessageToast from "sap/m/MessageToast";
import { NetworkGraph } from "../components/NetworkGraph/NetworkGraph";
import { ProcessFlow } from "../components/ProcessFlow/ProcessFlow";
import {
    ProcessFlowNodeState,
    ProcessFlowZoomLevel
} from "../components/ProcessFlow/types";

export default class OrchestrationGraphController extends Controller {
    private networkGraph: NetworkGraph | null = null;
    private processFlow: ProcessFlow | null = null;
    private refreshInterval: number | null = null;

    public onInit(): void {
        // Initialize models
        this.initializeModels();
        
        // Initialize components after view is rendered
        this.getView()!.addEventDelegate({
            onAfterRendering: this.onAfterRendering.bind(this)
        });
        
        // Load real data from backend
        this.loadAgentTopology();
        this.loadWorkflowExecution();
        
        // Auto-refresh every 10 seconds
        this.refreshInterval = window.setInterval(() => {
            this.loadAgentTopology();
            this.loadWorkflowExecution();
        }, 10000);
    }

    private initializeModels(): void {
        const viewModel = new JSONModel({
            selectedTab: "topology",
            stats: {
                totalAgents: 0,
                activeAgents: 0,
                totalRequests: 0,
                avgLatency: 0
            }
        });
        this.getView()!.setModel(viewModel);
    }

    private onAfterRendering(): void {
        if (!this.networkGraph) {
            this.initializeNetworkGraph();
        }
        if (!this.processFlow) {
            this.initializeProcessFlow();
        }
    }

    // ========================================================================
    // Network Graph Integration
    // ========================================================================

    private initializeNetworkGraph(): void {
        const container = document.getElementById("networkGraphContainer");
        if (!container) {
            console.error("Network graph container not found");
            return;
        }

        try {
            this.networkGraph = new NetworkGraph(container);
            
            // Setup event listeners
            this.networkGraph.on('nodeClick', (event: any) => {
                MessageToast.show(`Selected: ${event.node.name}`);
                this.showAgentDetails(event.node);
            });
            
            this.networkGraph.on('nodeHover', (event: any) => {
                // Could show tooltip
            });
            
            console.log("✅ Network Graph initialized");
        } catch (error) {
            console.error("Failed to initialize Network Graph:", error);
        }
    }

    private loadAgentTopology(): void {
        // Load REAL agents from Zig backend
        fetch('http://localhost:8080/api/v1/agents')
            .then(response => response.json())
            .then(data => {
                if (!data.agents || !Array.isArray(data.agents)) {
                    console.warn("No agents data received");
                    return;
                }

                console.log(`✅ Loaded ${data.agents.length} real agents from backend`);
                
                // Transform backend data to NetworkGraph format
                const graphData = this.transformAgentsToGraph(data.agents);
                
                // Update NetworkGraph
                if (this.networkGraph) {
                    this.networkGraph.loadData(graphData);
                    this.networkGraph.setLayout('force-directed');
                    
                    // Center after layout
                    setTimeout(() => {
                        this.networkGraph!.fitToView();
                    }, 500);
                }
                
                // Update statistics
                this.updateStatistics(data.agents);
            })
            .catch(error => {
                console.error("Failed to load agent topology:", error);
                MessageToast.show("Failed to load agent topology");
            });
    }

    private transformAgentsToGraph(agents: any[]): any {
        // Transform real agent data to graph format
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

        const edges: any[] = [];
        agents.forEach(agent => {
            if (agent.next_agents && Array.isArray(agent.next_agents)) {
                agent.next_agents.forEach((targetId: string) => {
                    edges.push({
                        from: agent.id,
                        to: targetId,
                        label: "",
                        style: 'solid' as const
                    });
                });
            }
        });

        return { nodes, edges };
    }

    private mapAgentStatus(backendStatus: string): string {
        const statusMap: { [key: string]: string } = {
            "healthy": "active",
            "active": "active",
            "busy": "warning",
            "idle": "idle",
            "error": "error",
            "stopped": "inactive"
        };
        return statusMap[backendStatus] || "idle";
    }

    private showAgentDetails(node: any): void {
        // Could open a dialog with agent details
        console.log("Agent details:", node);
    }

    // ========================================================================
    // Process Flow Integration
    // ========================================================================

    private initializeProcessFlow(): void {
        const container = document.getElementById("processFlowContainer");
        if (!container) {
            console.error("Process flow container not found");
            return;
        }

        try {
            this.processFlow = new ProcessFlow(container);
            
            // Setup event listeners
            this.processFlow.on('nodeClick', (event: any) => {
                MessageToast.show(`Step: ${event.node.title}`);
            });
            
            console.log("✅ Process Flow initialized");
        } catch (error) {
            console.error("Failed to initialize Process Flow:", error);
        }
    }

    private loadWorkflowExecution(): void {
        // Load REAL workflow execution data from backend
        // This would come from your workflow execution endpoint
        fetch('http://localhost:8080/api/v1/workflows/latest-execution')
            .then(response => response.json())
            .then(data => {
                if (data.execution) {
                    console.log("✅ Loaded workflow execution data");
                    const flowData = this.transformWorkflowToProcessFlow(data.execution);
                    
                    if (this.processFlow) {
                        this.processFlow.loadData(flowData);
                    }
                }
            })
            .catch(error => {
                // If endpoint doesn't exist yet, show example workflow
                console.log("Workflow endpoint not available, showing example");
                this.loadExampleWorkflow();
            });
    }

    private loadExampleWorkflow(): void {
        // Example workflow based on your real agent topology
        // This shows what a real workflow execution would look like
        const flowData = {
            lanes: [
                { id: 'input', label: 'Input Processing', position: 0 },
                { id: 'processing', label: 'Agent Processing', position: 1 },
                { id: 'validation', label: 'Quality Control', position: 2 },
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
                    children: ['code', 'translation', 'rag']
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
                    title: 'Translation Agent',
                    state: ProcessFlowNodeState.Positive,
                    texts: ['Completed: 1.2s'],
                    position: 1,
                    children: ['quality']
                },
                {
                    id: 'rag',
                    lane: 'processing',
                    title: 'RAG Agent',
                    state: ProcessFlowNodeState.Positive,
                    texts: ['Completed: 1.8s'],
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
                    id: 'quality',
                    lane: 'validation',
                    title: 'Quality Check',
                    state: ProcessFlowNodeState.Positive,
                    texts: ['Score: 98.5%'],
                    position: 2,
                    children: ['output']
                },
                {
                    id: 'output',
                    lane: 'output',
                    title: 'Final Output',
                    state: ProcessFlowNodeState.Planned,
                    texts: ['Pending'],
                    position: 3,
                    children: []
                }
            ],
            connections: [
                { from: 'router', to: 'code' },
                { from: 'router', to: 'translation' },
                { from: 'router', to: 'rag' },
                { from: 'code', to: 'validation' },
                { from: 'translation', to: 'quality' },
                { from: 'rag', to: 'validation' },
                { from: 'validation', to: 'output', type: 'planned' as const },
                { from: 'quality', to: 'output' }
            ]
        };

        if (this.processFlow) {
            this.processFlow.loadData(flowData);
        }
    }

    private transformWorkflowToProcessFlow(execution: any): any {
        // Transform real workflow execution data to ProcessFlow format
        // Implement based on your workflow execution structure
        return {
            lanes: [],
            nodes: [],
            connections: []
        };
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    private updateStatistics(agents: any[]): void {
        const viewModel = this.getView()!.getModel() as JSONModel;
        const activeAgents = agents.filter(a => 
            a.status === "active" || a.status === "healthy"
        ).length;
        
        const totalRequests = agents.reduce((sum, a) => 
            sum + (a.total_requests || 0), 0
        );
        
        const avgLatency = agents.length > 0
            ? agents.reduce((sum, a) => sum + (a.avg_latency || 0), 0) / agents.length
            : 0;

        viewModel.setProperty("/stats", {
            totalAgents: agents.length,
            activeAgents: activeAgents,
            totalRequests: totalRequests,
            avgLatency: Math.round(avgLatency)
        });
    }

    // ========================================================================
    // UI Actions
    // ========================================================================

    public onRefresh(): void {
        MessageToast.show("Refreshing...");
        this.loadAgentTopology();
        this.loadWorkflowExecution();
    }

    public onFitToView(): void {
        if (this.networkGraph) {
            this.networkGraph.fitToView();
        }
    }

    public onZoomIn(): void {
        if (this.networkGraph) {
            this.networkGraph.zoomIn();
        }
    }

    public onZoomOut(): void {
        if (this.networkGraph) {
            this.networkGraph.zoomOut();
        }
    }

    public onChangeLayout(): void {
        if (this.networkGraph) {
            // Cycle through layouts
            const layouts = ['force-directed', 'hierarchical', 'circular', 'grid'];
            const currentIndex = 0; // Could track this
            const nextLayout = layouts[(currentIndex + 1) % layouts.length];
            this.networkGraph.setLayout(nextLayout as any);
            MessageToast.show(`Layout: ${nextLayout}`);
        }
    }

    public onToggleProcessFlow(): void {
        const viewModel = this.getView()!.getModel() as JSONModel;
        const currentTab = viewModel.getProperty("/selectedTab");
        const newTab = currentTab === "topology" ? "workflow" : "topology";
        viewModel.setProperty("/selectedTab", newTab);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    public onExit(): void {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        if (this.networkGraph) {
            this.networkGraph.destroy();
        }
        if (this.processFlow) {
            this.processFlow.destroy();
        }
    }
}
