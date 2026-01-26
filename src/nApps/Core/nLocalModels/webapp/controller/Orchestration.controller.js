sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "llm/server/dashboard/utils/ApiService",
    "llm/server/dashboard/utils/GraphIntegration",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/m/Dialog",
    "sap/m/Input",
    "sap/m/Select",
    "sap/m/TextArea",
    "sap/m/MultiComboBox",
    "sap/m/VBox",
    "sap/m/Label",
    "sap/m/Button",
    "sap/ui/core/Item"
], function (Controller, JSONModel, ApiService, GraphIntegration, MessageBox, MessageToast, Dialog, Input, Select, TextArea, MultiComboBox, VBox, Label, Button, Item) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.Orchestration", {

        onInit: function () {
            // Initialize models
            this._initializeModels();
            
            // Initialize modern graph components (NetworkGraph + ProcessFlow)
            this.getView().addEventDelegate({
                onAfterRendering: function() {
                    if (!this._graphsInitialized) {
                        this._initializeGraphComponents();
                        this._graphsInitialized = true;
                    }
                }.bind(this)
            });
            
            // Load agent topology
            this._loadAgentTopology();
            
            // Subscribe to real-time updates
            ApiService.onMetricsUpdate(this._onAgentStatusUpdate.bind(this));
            
            // Auto-refresh every 10 seconds
            this._refreshInterval = setInterval(() => {
                this._loadAgentTopology();
                GraphIntegration.refresh();
            }, 10000);
        },
        
        _initializeGraphComponents: function() {
            // Initialize Network Graph with real backend data
            GraphIntegration.initializeNetworkGraph("networkGraphContainer")
                .then(() => {
                    console.log("✅ Network Graph ready");
                })
                .catch(error => {
                    console.error("Network Graph initialization failed:", error);
                });
            
            // Initialize Process Flow with real backend data
            GraphIntegration.initializeProcessFlow("processFlowContainer")
                .then(() => {
                    console.log("✅ Process Flow ready");
                })
                .catch(error => {
                    console.error("Process Flow initialization failed:", error);
                });
        },
        
        _initializeModels: function() {
            // Graph model for Network Graph
            var oGraphModel = new JSONModel({
                nodes: [],
                lines: [],
                groups: []
            });
            this.getView().setModel(oGraphModel, "graph");
            
            // View model for UI state
            var oViewModel = new JSONModel({
                selectedAgent: {
                    visible: false
                },
                workflows: [],
                workflowInputData: "",
                workflowResults: {
                    visible: false
                },
                stats: {
                    totalAgents: 0,
                    activeAgents: 0,
                    totalWorkflows: 0,
                    totalRequests: 0,
                    requestsPerSecond: 0,
                    avgNetworkLatency: 0
                }
            });
            this.getView().setModel(oViewModel);
        },
        
        _loadAgentTopology: function() {
            // Load REAL agents from dashboard_api_server
            // GET http://localhost:8080/api/v1/agents
            fetch('http://localhost:8080/api/v1/agents')
                .then(response => response.json())
                .then(response => {
                    console.log("✅ Loaded real agent topology:", response.agents.length, "agents");
                    this._buildNetworkGraph(response.agents || []);
                    this._updateStatistics(response.agents || []);
                    this._loadWorkflows();
                })
                .catch(error => {
                    console.error("Failed to load agent topology:", error);
                    sap.m.MessageBox.error("Agent topology unavailable. Please check the orchestration API.");
                    this._buildNetworkGraph([]);
                    this._updateStatistics([]);
                });
        },
        
        _buildNetworkGraph: function(agents) {
            var oGraphModel = this.getView().getModel("graph");
            
            // Build nodes
            var nodes = agents.map((agent, idx) => ({
                id: agent.id,
                name: agent.name,
                description: agent.description || agent.type,
                icon: this._getAgentIcon(agent.type),
                status: this._mapAgentStatus(agent.status),
                group: agent.type,
                type: agent.type,
                model: agent.model_id || "N/A",
                totalRequests: agent.total_requests || 0,
                avgLatency: agent.avg_latency || 0,
                successRate: agent.success_rate || 0,
                // Position calculation (can be improved with layout algorithm)
                x: (idx % 3) * 300 + 100,
                y: Math.floor(idx / 3) * 200 + 100
            }));
            
            // Build connections (edges)
            var lines = [];
            agents.forEach(agent => {
                if (agent.next_agents && Array.isArray(agent.next_agents)) {
                    agent.next_agents.forEach(targetId => {
                        lines.push({
                            from: agent.id,
                            to: targetId,
                            label: "forwards to",
                            status: "Success",
                            dataType: agent.data_type || "JSON",
                            throughput: agent.throughput || 0
                        });
                    });
                }
            });
            
            // Build groups (optional - for organizing agents)
            var groupMap = {};
            agents.forEach(agent => {
                if (!groupMap[agent.type]) {
                    groupMap[agent.type] = {
                        key: agent.type,
                        title: agent.type.charAt(0).toUpperCase() + agent.type.slice(1) + " Agents",
                        description: `All ${agent.type} agents in the network`,
                        icon: this._getAgentIcon(agent.type),
                        collapsed: false
                    };
                }
            });
            
            oGraphModel.setProperty("/nodes", nodes);
            oGraphModel.setProperty("/lines", lines);
            oGraphModel.setProperty("/groups", Object.values(groupMap));
            
            console.log(`Network graph updated: ${nodes.length} nodes, ${lines.length} edges`);
        },
        
        _getAgentIcon: function(type) {
            const iconMap = {
                "orchestrator": "sap-icon://decision",
                "router": "sap-icon://overview-chart",
                "code": "sap-icon://syntax",
                "translation": "sap-icon://globe",
                "rag": "sap-icon://database",
                "llm": "sap-icon://message-assistant",
                "validation": "sap-icon://quality-issue",
                "quality": "sap-icon://complete"
            };
            return iconMap[type] || "sap-icon://puzzle";
        },
        
        _mapAgentStatus: function(status) {
            // Map agent status to SAP UI5 status
            const statusMap = {
                "healthy": "Success",
                "active": "Success",
                "busy": "Warning",
                "idle": "None",
                "error": "Error",
                "stopped": "None"
            };
            return statusMap[status] || "None";
        },
        
        _updateStatistics: function(agents) {
            var oViewModel = this.getView().getModel();
            var activeAgents = agents.filter(a => a.status === "active" || a.status === "healthy").length;
            var totalRequests = agents.reduce((sum, a) => sum + (a.total_requests || 0), 0);
            var avgLatency = agents.length > 0
                ? agents.reduce((sum, a) => sum + (a.avg_latency || 0), 0) / agents.length
                : 0;

            oViewModel.setProperty("/stats", {
                totalAgents: agents.length,
                activeAgents: activeAgents,
                totalWorkflows: oViewModel.getProperty("/workflows").length,
                totalRequests: totalRequests,
                requestsPerSecond: (totalRequests / 60).toFixed(1), // Rough estimate
                avgNetworkLatency: avgLatency.toFixed(1)
            });
        },
        
        _loadWorkflows: function() {
            // Load available workflows
            var oViewModel = this.getView().getModel();
            oViewModel.setProperty("/workflows", [
                { id: "code-gen", name: "Code Generation Pipeline" },
                { id: "translation", name: "Translation Workflow" },
                { id: "rag-qa", name: "RAG Q&A System" },
                { id: "multi-agent", name: "Multi-Agent Collaboration" }
            ]);
        },
        
        _onAgentStatusUpdate: function(message) {
            if (message.type === "agent_status") {
                // Update specific agent status in real-time
                var oGraphModel = this.getView().getModel("graph");
                var nodes = oGraphModel.getProperty("/nodes");
                
                var agentIndex = nodes.findIndex(n => n.id === message.agent_id);
                if (agentIndex >= 0) {
                    nodes[agentIndex].status = this._mapAgentStatus(message.status);
                    nodes[agentIndex].totalRequests = message.total_requests || nodes[agentIndex].totalRequests;
                    nodes[agentIndex].avgLatency = message.avg_latency || nodes[agentIndex].avgLatency;
                    
                    oGraphModel.setProperty("/nodes", nodes);
                }
            }
        },
        
        onNodePress: function(oEvent) {
            var oNode = oEvent.getParameter("node");
            var oViewModel = this.getView().getModel();
            var oGraphModel = this.getView().getModel("graph");
            
            // Find the full node data
            var nodes = oGraphModel.getProperty("/nodes");
            var nodeData = nodes.find(n => n.id === oNode.getKey());
            
            if (nodeData) {
                // Get connections
                var lines = oGraphModel.getProperty("/lines");
                var connections = lines
                    .filter(l => l.from === nodeData.id)
                    .map(l => {
                        var targetNode = nodes.find(n => n.id === l.to);
                        return {
                            name: targetNode ? targetNode.name : l.to,
                            icon: targetNode ? targetNode.icon : "sap-icon://puzzle"
                        };
                    });
                
                // Update selected agent details
                oViewModel.setProperty("/selectedAgent", {
                    visible: true,
                    id: nodeData.id,
                    name: nodeData.name,
                    description: nodeData.description,
                    icon: nodeData.icon,
                    statusText: nodeData.status,
                    state: nodeData.status,
                    statusColor: this._getStatusColor(nodeData.status),
                    model: nodeData.model,
                    type: nodeData.type,
                    totalRequests: nodeData.totalRequests,
                    avgLatency: nodeData.avgLatency,
                    successRate: nodeData.successRate,
                    connections: connections
                });
                
                MessageToast.show(`Selected agent: ${nodeData.name}`);
            }
        },
        
        _getStatusColor: function(status) {
            const colorMap = {
                "Success": "#00A600",
                "Warning": "#FF9500",
                "Error": "#DC143C",
                "None": "#6c757d"
            };
            return colorMap[status] || "#6c757d";
        },
        
        onLinePress: function(oEvent) {
            var oLine = oEvent.getParameter("line");
            MessageToast.show(`Connection: ${oLine.getFrom()} → ${oLine.getTo()}`);
        },
        
        onRefreshGraph: function() {
            MessageToast.show("Refreshing agent topology...");
            this._loadAgentTopology();
        },
        
        onAutoLayout: function() {
            var oGraph = this.byId("agentGraph");
            if (oGraph && oGraph.layoutGraph) {
                oGraph.layoutGraph();
                MessageToast.show("Graph layout updated");
            }
        },
        
        onAddAgent: function() {
            var that = this;

            // Reuse existing dialog if available
            if (this._oAddAgentDialog) {
                this._updateConnectedAgentsComboBox();
                this._oAddAgentDialog.open();
                return;
            }

            // Create form controls
            var oNameInput = new Input({
                id: "addAgentNameInput",
                placeholder: "Enter agent name",
                width: "100%"
            });

            var oTypeSelect = new Select({
                id: "addAgentTypeSelect",
                width: "100%",
                items: [
                    new Item({ key: "router", text: "Router" }),
                    new Item({ key: "orchestrator", text: "Orchestrator" }),
                    new Item({ key: "code", text: "Code" }),
                    new Item({ key: "translation", text: "Translation" }),
                    new Item({ key: "rag", text: "RAG" }),
                    new Item({ key: "validation", text: "Validation" }),
                    new Item({ key: "quality", text: "Quality" }),
                    new Item({ key: "tool", text: "Tool" })
                ]
            });

            var oDescriptionTextArea = new TextArea({
                id: "addAgentDescriptionTextArea",
                placeholder: "Enter agent description",
                width: "100%",
                rows: 3
            });

            var oModelIdInput = new Input({
                id: "addAgentModelIdInput",
                placeholder: "e.g., gpt-4, claude-3-opus",
                width: "100%"
            });

            var oConnectedAgentsComboBox = new MultiComboBox({
                id: "addAgentConnectedAgentsComboBox",
                placeholder: "Select connected agents",
                width: "100%"
            });

            // Create the dialog
            this._oAddAgentDialog = new Dialog({
                title: "Add New Agent",
                contentWidth: "450px",
                content: [
                    new VBox({
                        items: [
                            new Label({ text: "Agent Name", required: true }),
                            oNameInput,
                            new Label({ text: "Agent Type", labelFor: "addAgentTypeSelect" }).addStyleClass("sapUiSmallMarginTop"),
                            oTypeSelect,
                            new Label({ text: "Description" }).addStyleClass("sapUiSmallMarginTop"),
                            oDescriptionTextArea,
                            new Label({ text: "Model ID" }).addStyleClass("sapUiSmallMarginTop"),
                            oModelIdInput,
                            new Label({ text: "Connected Agents" }).addStyleClass("sapUiSmallMarginTop"),
                            oConnectedAgentsComboBox
                        ]
                    }).addStyleClass("sapUiSmallMargin")
                ],
                beginButton: new Button({
                    text: "Create",
                    type: "Emphasized",
                    press: function() {
                        var sName = oNameInput.getValue().trim();
                        var sType = oTypeSelect.getSelectedKey();
                        var sDescription = oDescriptionTextArea.getValue().trim();
                        var sModelId = oModelIdInput.getValue().trim();
                        var aConnectedAgents = oConnectedAgentsComboBox.getSelectedKeys();

                        // Validate required fields
                        if (!sName) {
                            MessageBox.error("Agent Name is required");
                            oNameInput.setValueState("Error");
                            return;
                        }
                        oNameInput.setValueState("None");

                        var oAgentConfig = {
                            name: sName,
                            type: sType,
                            description: sDescription,
                            model_id: sModelId,
                            connected_agents: aConnectedAgents
                        };

                        ApiService.createAgent(oAgentConfig)
                            .then(function() {
                                MessageToast.show("Agent '" + sName + "' created successfully");
                                that._oAddAgentDialog.close();
                                that._loadAgentTopology();
                            })
                            .catch(function(error) {
                                MessageBox.error("Failed to create agent: " + (error.message || error));
                            });
                    }
                }),
                endButton: new Button({
                    text: "Cancel",
                    press: function() {
                        that._oAddAgentDialog.close();
                    }
                }),
                afterClose: function() {
                    // Reset form fields
                    oNameInput.setValue("");
                    oNameInput.setValueState("None");
                    oTypeSelect.setSelectedKey("router");
                    oDescriptionTextArea.setValue("");
                    oModelIdInput.setValue("");
                    oConnectedAgentsComboBox.setSelectedKeys([]);
                }
            });

            this.getView().addDependent(this._oAddAgentDialog);
            this._updateConnectedAgentsComboBox();
            this._oAddAgentDialog.open();
        },

        _updateConnectedAgentsComboBox: function() {
            var oComboBox = sap.ui.getCore().byId("addAgentConnectedAgentsComboBox");
            if (!oComboBox) {
                return;
            }

            oComboBox.removeAllItems();

            var oGraphModel = this.getView().getModel("graph");
            var aNodes = oGraphModel ? oGraphModel.getProperty("/nodes") || [] : [];

            aNodes.forEach(function(node) {
                oComboBox.addItem(new Item({
                    key: node.id,
                    text: node.name + " (" + node.type + ")"
                }));
            });
        },
        
        onCreateWorkflow: function() {
            var that = this;

            // Reuse existing dialog if available
            if (this._oCreateWorkflowDialog) {
                this._oCreateWorkflowDialog.open();
                return;
            }

            // Get existing agents from graph model for the MultiComboBox
            var oGraphModel = this.getView().getModel("graph");
            var aNodes = oGraphModel.getProperty("/nodes") || [];

            // Create form controls
            var oNameInput = new sap.m.Input({
                id: "workflowNameInput",
                placeholder: "Enter workflow name",
                required: true,
                width: "100%"
            });

            var oDescriptionArea = new sap.m.TextArea({
                id: "workflowDescriptionInput",
                placeholder: "Enter workflow description (optional)",
                rows: 3,
                width: "100%"
            });

            // Create MultiComboBox with available agents
            var oAgentCombo = new sap.m.MultiComboBox({
                id: "workflowAgentSelector",
                placeholder: "Select agents for workflow",
                width: "100%",
                selectionChange: function(oEvent) {
                    that._updateSelectedAgentsList(oAgentCombo, oSelectedAgentsList);
                }
            });

            // Populate agents in MultiComboBox
            aNodes.forEach(function(node) {
                oAgentCombo.addItem(new sap.ui.core.Item({
                    key: node.id,
                    text: node.name + " (" + node.description + ")"
                }));
            });

            // List to show selected agents order
            var oSelectedAgentsList = new sap.m.List({
                id: "selectedAgentsList",
                headerText: "Workflow Sequence (drag to reorder)",
                noDataText: "Select at least 2 agents above",
                mode: "None"
            });

            // Create dialog
            this._oCreateWorkflowDialog = new sap.m.Dialog({
                title: "Create New Workflow",
                contentWidth: "500px",
                content: [
                    new sap.m.VBox({
                        class: "sapUiSmallMargin",
                        items: [
                            new sap.m.Label({ text: "Workflow Name", required: true }),
                            oNameInput,
                            new sap.m.Label({ text: "Description", class: "sapUiSmallMarginTop" }),
                            oDescriptionArea,
                            new sap.m.Label({ text: "Select Agents", required: true, class: "sapUiSmallMarginTop" }),
                            oAgentCombo,
                            oSelectedAgentsList
                        ]
                    })
                ],
                beginButton: new sap.m.Button({
                    text: "Create",
                    type: "Emphasized",
                    press: function() {
                        that._handleCreateWorkflow(oNameInput, oDescriptionArea, oAgentCombo);
                    }
                }),
                endButton: new sap.m.Button({
                    text: "Cancel",
                    press: function() {
                        that._oCreateWorkflowDialog.close();
                    }
                }),
                afterClose: function() {
                    // Reset form fields
                    oNameInput.setValue("");
                    oDescriptionArea.setValue("");
                    oAgentCombo.setSelectedKeys([]);
                    oSelectedAgentsList.removeAllItems();
                }
            });

            this._oCreateWorkflowDialog.open();
        },

        _updateSelectedAgentsList: function(oAgentCombo, oList) {
            var aSelectedKeys = oAgentCombo.getSelectedKeys();
            var aItems = oAgentCombo.getItems();

            oList.removeAllItems();

            aSelectedKeys.forEach(function(sKey, iIndex) {
                var oItem = aItems.find(function(item) { return item.getKey() === sKey; });
                if (oItem) {
                    oList.addItem(new sap.m.StandardListItem({
                        title: (iIndex + 1) + ". " + oItem.getText(),
                        icon: "sap-icon://process",
                        info: iIndex < aSelectedKeys.length - 1 ? "→" : "End"
                    }));
                }
            });
        },

        _handleCreateWorkflow: function(oNameInput, oDescriptionArea, oAgentCombo) {
            var that = this;
            var sName = oNameInput.getValue().trim();
            var sDescription = oDescriptionArea.getValue().trim();
            var aSelectedKeys = oAgentCombo.getSelectedKeys();

            // Validate required fields
            if (!sName) {
                MessageBox.error("Please enter a workflow name");
                oNameInput.setValueState("Error");
                return;
            }
            oNameInput.setValueState("None");

            if (aSelectedKeys.length < 2) {
                MessageBox.error("Please select at least 2 agents for the workflow");
                return;
            }

            // Build workflow config
            var sWorkflowId = "workflow-" + Date.now();

            // Build nodes array from selected agents
            var aNodes = aSelectedKeys.map(function(sAgentId, iIndex) {
                return {
                    id: sAgentId,
                    position: iIndex
                };
            });

            // Build sequential connections between agents
            var aConnections = [];
            for (var i = 0; i < aSelectedKeys.length - 1; i++) {
                aConnections.push({
                    from: aSelectedKeys[i],
                    to: aSelectedKeys[i + 1]
                });
            }

            var oWorkflowConfig = {
                id: sWorkflowId,
                name: sName,
                description: sDescription,
                nodes: aNodes,
                connections: aConnections,
                createdAt: new Date().toISOString()
            };

            // Call API to create workflow
            ApiService.createWorkflow(oWorkflowConfig)
                .then(function(response) {
                    MessageToast.show("Workflow '" + sName + "' created successfully");
                    that._oCreateWorkflowDialog.close();

                    // Refresh workflows list
                    that._loadWorkflows();

                    // Update statistics
                    var oViewModel = that.getView().getModel();
                    var iCurrentWorkflows = oViewModel.getProperty("/stats/totalWorkflows") || 0;
                    oViewModel.setProperty("/stats/totalWorkflows", iCurrentWorkflows + 1);
                })
                .catch(function(error) {
                    MessageBox.error("Failed to create workflow: " + (error.message || "Unknown error"));
                });
        },
        
        onExecuteWorkflow: function() {
            var oViewModel = this.getView().getModel();
            var sWorkflowId = this.byId("workflowSelector").getSelectedKey();
            var sInput = oViewModel.getProperty("/workflowInputData");
            
            if (!sWorkflowId) {
                MessageBox.error("Please select a workflow");
                return;
            }
            
            var startTime = Date.now();
            
            ApiService.executeWorkflow(sWorkflowId, { input: sInput })
                .then(response => {
                    var duration = Date.now() - startTime;
                    
                    oViewModel.setProperty("/workflowResults", {
                        visible: true,
                        status: "Completed",
                        state: "Success",
                        duration: duration,
                        output: JSON.stringify(response.output, null, 2)
                    });
                    
                    MessageToast.show(`Workflow completed in ${duration}ms`);
                })
                .catch(error => {
                    oViewModel.setProperty("/workflowResults", {
                        visible: true,
                        status: "Failed",
                        state: "Error",
                        duration: Date.now() - startTime,
                        output: error.message
                    });
                    
                    MessageBox.error("Workflow execution failed: " + error.message);
                });
        },
        
        onConfigureAgent: function() {
            var oViewModel = this.getView().getModel();
            var sAgentId = oViewModel.getProperty("/selectedAgent/id");
            MessageToast.show(`Configure agent: ${sAgentId}`);
        },
        
        onViewAgentLogs: function() {
            var oViewModel = this.getView().getModel();
            var sAgentId = oViewModel.getProperty("/selectedAgent/id");
            MessageToast.show(`View logs for: ${sAgentId}`);
        },
        
        onStopAgent: function() {
            var oViewModel = this.getView().getModel();
            var sAgentId = oViewModel.getProperty("/selectedAgent/id");
            var sAgentName = oViewModel.getProperty("/selectedAgent/name");
            
            MessageBox.confirm(
                `Are you sure you want to stop agent "${sAgentName}"?`,
                {
                    title: "Stop Agent",
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            MessageToast.show(`Stopping agent: ${sAgentId}`);
                            // TODO: Call API to stop agent
                        }
                    }
                }
            );
        },
        
        onExportTopology: function() {
            var oGraphModel = this.getView().getModel("graph");
            var topology = {
                nodes: oGraphModel.getProperty("/nodes"),
                lines: oGraphModel.getProperty("/lines"),
                groups: oGraphModel.getProperty("/groups")
            };
            
            ApiService.exportData(topology, "agent_topology", "json");
            MessageToast.show("Topology exported");
        },
        
        onImportTopology: function() {
            MessageBox.information("Import topology from JSON file (coming soon)");
        },
        
        onRefreshGraphs: function() {
            MessageToast.show("Refreshing visualizations...");
            GraphIntegration.refresh();
        },
        
        onExit: function() {
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
            }
            GraphIntegration.destroy();
        }
    });
});
