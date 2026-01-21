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
    "sap/m/VBox",
    "sap/m/Label",
    "sap/m/Button",
    "sap/ui/core/Item"
], function (Controller, JSONModel, ApiService, GraphIntegration, MessageBox, MessageToast, Dialog, Input, Select, VBox, Label, Button, Item) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.ModelRouter", {

        // Mock model data for fallback
        _mockModels: [
            { id: "gpt-5", name: "GPT-5", capabilities: ["coding", "reasoning", "creative", "tool_use"], quality: 95, provider: "OpenAI" },
            { id: "claude-3-opus", name: "Claude 3 Opus", capabilities: ["reasoning", "creative", "long_context"], quality: 92, provider: "Anthropic" },
            { id: "deepseek-coder-33b", name: "DeepSeek Coder 33B", capabilities: ["coding"], quality: 88, provider: "DeepSeek" },
            { id: "Qwen2.5-Math-72B", name: "Qwen 2.5 Math 72B", capabilities: ["math", "reasoning"], quality: 90, provider: "Alibaba" },
            { id: "lfm2.5-1.2b-q4_0", name: "LFM 2.5 1.2B Q4", capabilities: ["fast_inference"], quality: 75, provider: "Local" }
        ],

        onInit: function () {
            // Initialize models
            this._initializeModels();
            
            // Initialize graph components after rendering
            this.getView().addEventDelegate({
                onAfterRendering: function() {
                    if (!this._graphsInitialized) {
                        this._initializeGraphComponents();
                        this._graphsInitialized = true;
                    }
                }.bind(this)
            });
            
            // Load data
            this._loadModels();
            this._loadAssignments();
        },

        _initializeModels: function() {
            var oViewModel = new JSONModel({
                assignments: [],
                models: [],
                stats: {
                    totalAgents: 0,
                    assignedAgents: 0,
                    totalModels: 0,
                    avgMatchScore: 0
                },
                routingConfig: {
                    autoAssignEnabled: true,
                    preferQuality: true,
                    fallbackModel: "lfm2.5-1.2b-q4_0",
                    strategy: "balanced" // Day 25: greedy, optimal, or balanced
                },
                metricsConfig: {
                    autoRefresh: true,
                    refreshInterval: 5000
                },
                liveMetrics: {
                    totalDecisions: 0,
                    successRate: 0,
                    avgLatency: 0,
                    fallbacksUsed: 0,
                    recentDecisions: []
                }
            });
            this.getView().setModel(oViewModel);

            // Start live metrics polling
            this._startMetricsPolling();
            
            // Graph model for NetworkGraph
            var oGraphModel = new JSONModel({
                nodes: [],
                lines: [],
                groups: []
            });
            this.getView().setModel(oGraphModel, "graph");
        },

        _initializeGraphComponents: function() {
            // Initialize Network Graph for model-agent relationships
            GraphIntegration.initializeNetworkGraph("modelRouterNetworkGraphContainer")
                .then(() => {
                    console.log("✅ Model Router Network Graph ready");
                    this._buildNetworkGraphData();
                })
                .catch(error => {
                    console.error("Model Router Network Graph initialization failed:", error);
                });
            
            // Initialize Process Flow for routing visualization
            GraphIntegration.initializeProcessFlow("modelRouterProcessFlowContainer")
                .then(() => {
                    console.log("✅ Model Router Process Flow ready");
                    this._buildProcessFlowData();
                })
                .catch(error => {
                    console.error("Model Router Process Flow initialization failed:", error);
                });
        },

        _loadModels: function() {
            var that = this;
            var oViewModel = this.getView().getModel();
            
            fetch('http://localhost:8080/api/v1/models')
                .then(response => response.json())
                .then(data => {
                    var models = data.models || [];
                    if (models.length === 0) {
                        console.log("ℹ️ No models from backend, using mock data");
                        models = that._mockModels;
                    } else {
                        console.log(`✅ Loaded ${models.length} models from backend`);
                    }
                    oViewModel.setProperty("/models", models);
                    that._updateStatistics();
                })
                .catch(error => {
                    console.error("Failed to load models:", error);
                    oViewModel.setProperty("/models", that._mockModels);
                    that._updateStatistics();
                });
        },

        _loadAssignments: function() {
            var that = this;
            var oViewModel = this.getView().getModel();
            
            // Day 25: Use Day 24 API endpoint with pagination
            fetch('http://localhost:8080/api/v1/model-router/assignments?page=1&page_size=100&status=ACTIVE')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('API request failed: ' + response.status);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        // Convert API format to UI format
                        var assignments = data.assignments.map(function(a) {
                            return {
                                assignmentId: a.assignment_id,
                                agentId: a.agent_id,
                                agentName: a.agent_name,
                                agentType: "inference",
                                modelId: a.model_id,
                                modelName: a.model_name,
                                matchScore: Math.round(a.match_score),
                                status: a.status.toLowerCase(),
                                assignmentMethod: a.assignment_method,
                                totalRequests: a.total_requests || 0,
                                successfulRequests: a.successful_requests || 0,
                                avgLatencyMs: a.avg_latency_ms
                            };
                        });
                        
                        console.log(`✅ Loaded ${assignments.length} assignments from Day 24 API`);
                        oViewModel.setProperty("/assignments", assignments);
                        that._updateStatistics();
                        that._buildNetworkGraphData();
                    } else {
                        throw new Error('API returned success=false');
                    }
                })
                .catch(error => {
                    console.error("Failed to load assignments from Day 24 API, loading agents instead:", error);
                    that._loadAgentsAndBuildAssignments();
                });
        },

        _loadAgentsAndBuildAssignments: function() {
            var that = this;
            var oViewModel = this.getView().getModel();

            fetch('http://localhost:8080/api/v1/agents')
                .then(response => response.json())
                .then(data => {
                    var agents = data.agents || [];
                    var assignments = agents.map(function(agent) {
                        return {
                            agentId: agent.id,
                            agentName: agent.name,
                            agentType: agent.type,
                            modelId: agent.model_id || null,
                            matchScore: 0,
                            status: agent.model_id ? "assigned" : "unassigned"
                        };
                    });
                    oViewModel.setProperty("/assignments", assignments);
                    that._updateStatistics();
                    that._buildNetworkGraphData();
                })
                .catch(error => {
                    console.error("Failed to load agents:", error);
                    oViewModel.setProperty("/assignments", []);
                });
        },

        _calculateMatchScore: function(agent, model) {
            // Score 0-100 based on agent type vs model capabilities
            var score = 0;
            var agentType = agent.agentType || agent.type || "";
            var capabilities = model.capabilities || [];

            // Capability matching rules
            var typeCapabilityMap = {
                "code": ["coding", "tool_use"],
                "coding": ["coding", "tool_use"],
                "translation": ["creative", "long_context"],
                "rag": ["reasoning", "long_context"],
                "orchestrator": ["reasoning", "tool_use"],
                "router": ["fast_inference", "reasoning"],
                "validation": ["reasoning"],
                "quality": ["reasoning", "creative"],
                "math": ["math", "reasoning"]
            };

            var requiredCapabilities = typeCapabilityMap[agentType] || ["reasoning"];
            var matchedCapabilities = 0;

            requiredCapabilities.forEach(function(cap) {
                if (capabilities.indexOf(cap) !== -1) {
                    matchedCapabilities++;
                }
            });

            // Base score from capability matching (0-60)
            if (requiredCapabilities.length > 0) {
                score = Math.round((matchedCapabilities / requiredCapabilities.length) * 60);
            }

            // Quality bonus (0-40)
            var qualityBonus = Math.round((model.quality || 0) * 0.4);
            score += qualityBonus;

            return Math.min(100, Math.max(0, score));
        },

        onAutoAssignAll: function() {
            var that = this;
            var oViewModel = this.getView().getModel();
            
            // Get strategy from routing config (default to 'balanced')
            var strategy = oViewModel.getProperty("/routingConfig/strategy") || "balanced";

            // Call Day 24 API endpoint
            fetch('http://localhost:8080/api/v1/model-router/auto-assign-all', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ strategy: strategy })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('API request failed: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    // Update assignments with API response
                    var apiAssignments = data.assignments.map(function(a) {
                        return {
                            agentId: a.agent_id,
                            agentName: a.agent_name,
                            agentType: "inference", // Default type
                            modelId: a.model_id,
                            modelName: a.model_name,
                            matchScore: Math.round(a.match_score),
                            status: "assigned",
                            assignmentMethod: a.assignment_method,
                            capabilityOverlap: a.capability_overlap || [],
                            missingRequired: a.missing_required || []
                        };
                    });
                    
                    oViewModel.setProperty("/assignments", apiAssignments);
                    that._updateStatistics();
                    that._buildNetworkGraphData();
                    
                    MessageToast.show(
                        `Auto-assigned ${data.total_assignments} agents | ` +
                        `Avg Score: ${Math.round(data.avg_match_score)} | ` +
                        `Strategy: ${data.strategy}`
                    );
                } else {
                    MessageBox.error("Auto-assignment failed: " + (data.message || "Unknown error"));
                }
            })
            .catch(error => {
                console.error("Auto-assign API call failed, falling back to local algorithm:", error);
                // Fallback to local algorithm
                that._autoAssignAllLocal();
            });
        },
        
        _autoAssignAllLocal: function() {
            // Original local algorithm as fallback
            var that = this;
            var oViewModel = this.getView().getModel();
            var assignments = oViewModel.getProperty("/assignments") || [];
            var models = oViewModel.getProperty("/models") || [];

            if (models.length === 0) {
                MessageBox.warning("No models available for assignment");
                return;
            }

            var updatedCount = 0;

            assignments.forEach(function(assignment) {
                var bestModel = null;
                var bestScore = -1;

                models.forEach(function(model) {
                    var score = that._calculateMatchScore(assignment, model);
                    if (score > bestScore) {
                        bestScore = score;
                        bestModel = model;
                    }
                });

                if (bestModel) {
                    assignment.modelId = bestModel.id;
                    assignment.matchScore = bestScore;
                    assignment.status = "assigned";
                    updatedCount++;
                }
            });

            oViewModel.setProperty("/assignments", assignments);
            this._updateStatistics();
            this._buildNetworkGraphData();

            MessageToast.show(`Auto-assigned ${updatedCount} agents (local fallback)`);
        },

        onAutoAssignAgent: function(oEvent) {
            var that = this;
            var oContext = oEvent.getSource().getBindingContext();
            var oViewModel = this.getView().getModel();
            var models = oViewModel.getProperty("/models") || [];

            if (!oContext) {
                MessageBox.error("No agent selected");
                return;
            }

            var sPath = oContext.getPath();
            var assignment = oViewModel.getProperty(sPath);

            if (models.length === 0) {
                MessageBox.warning("No models available for assignment");
                return;
            }

            var bestModel = null;
            var bestScore = -1;

            models.forEach(function(model) {
                var score = that._calculateMatchScore(assignment, model);
                if (score > bestScore) {
                    bestScore = score;
                    bestModel = model;
                }
            });

            if (bestModel) {
                oViewModel.setProperty(sPath + "/modelId", bestModel.id);
                oViewModel.setProperty(sPath + "/matchScore", bestScore);
                oViewModel.setProperty(sPath + "/status", "assigned");

                this._updateStatistics();
                this._buildNetworkGraphData();

                MessageToast.show(`Assigned ${assignment.agentName} to ${bestModel.name} (score: ${bestScore})`);
            }
        },

        onConfigureAssignment: function(oEvent) {
            var that = this;
            var oContext = oEvent.getSource().getBindingContext();
            var oViewModel = this.getView().getModel();

            if (!oContext) {
                MessageBox.error("No agent selected");
                return;
            }

            var sPath = oContext.getPath();
            var assignment = oViewModel.getProperty(sPath);
            var models = oViewModel.getProperty("/models") || [];

            // Create model select
            var oModelSelect = new Select({
                id: "configureModelSelect",
                width: "100%",
                selectedKey: assignment.modelId || ""
            });

            // Add empty option
            oModelSelect.addItem(new Item({ key: "", text: "(Unassigned)" }));

            // Add model options
            models.forEach(function(model) {
                var score = that._calculateMatchScore(assignment, model);
                oModelSelect.addItem(new Item({
                    key: model.id,
                    text: model.name + " (Score: " + score + ")"
                }));
            });

            var oDialog = new Dialog({
                title: "Configure Model Assignment",
                contentWidth: "400px",
                content: [
                    new VBox({
                        items: [
                            new Label({ text: "Agent: " + assignment.agentName }),
                            new Label({ text: "Type: " + assignment.agentType }).addStyleClass("sapUiSmallMarginBottom"),
                            new Label({ text: "Select Model", required: true }),
                            oModelSelect
                        ]
                    }).addStyleClass("sapUiSmallMargin")
                ],
                beginButton: new Button({
                    text: "Save",
                    type: "Emphasized",
                    press: function() {
                        var selectedKey = oModelSelect.getSelectedKey();
                        var selectedModel = models.find(function(m) { return m.id === selectedKey; });

                        if (selectedKey && selectedModel) {
                            var score = that._calculateMatchScore(assignment, selectedModel);
                            oViewModel.setProperty(sPath + "/modelId", selectedKey);
                            oViewModel.setProperty(sPath + "/matchScore", score);
                            oViewModel.setProperty(sPath + "/status", "assigned");
                        } else {
                            oViewModel.setProperty(sPath + "/modelId", null);
                            oViewModel.setProperty(sPath + "/matchScore", 0);
                            oViewModel.setProperty(sPath + "/status", "unassigned");
                        }

                        that._updateStatistics();
                        that._buildNetworkGraphData();
                        oDialog.close();
                        MessageToast.show("Assignment updated");
                    }
                }),
                endButton: new Button({
                    text: "Cancel",
                    press: function() {
                        oDialog.close();
                    }
                }),
                afterClose: function() {
                    oDialog.destroy();
                }
            });

            this.getView().addDependent(oDialog);
            oDialog.open();
        },

        onAddModel: function() {
            var that = this;
            var oViewModel = this.getView().getModel();

            // Create form inputs
            var oIdInput = new Input({ id: "addModelIdInput", placeholder: "e.g., gpt-5-turbo", width: "100%" });
            var oNameInput = new Input({ id: "addModelNameInput", placeholder: "e.g., GPT-5 Turbo", width: "100%" });
            var oProviderInput = new Input({ id: "addModelProviderInput", placeholder: "e.g., OpenAI", width: "100%" });
            var oCapabilitiesInput = new Input({ id: "addModelCapabilitiesInput", placeholder: "coding, reasoning, creative (comma-separated)", width: "100%" });
            var oQualityInput = new Input({ id: "addModelQualityInput", placeholder: "0-100", type: "Number", width: "100%" });

            var oDialog = new Dialog({
                title: "Add New Model",
                contentWidth: "450px",
                content: [
                    new VBox({
                        items: [
                            new Label({ text: "Model ID", required: true }),
                            oIdInput,
                            new Label({ text: "Model Name", required: true }).addStyleClass("sapUiSmallMarginTop"),
                            oNameInput,
                            new Label({ text: "Provider" }).addStyleClass("sapUiSmallMarginTop"),
                            oProviderInput,
                            new Label({ text: "Capabilities" }).addStyleClass("sapUiSmallMarginTop"),
                            oCapabilitiesInput,
                            new Label({ text: "Quality Score (0-100)" }).addStyleClass("sapUiSmallMarginTop"),
                            oQualityInput
                        ]
                    }).addStyleClass("sapUiSmallMargin")
                ],
                beginButton: new Button({
                    text: "Add",
                    type: "Emphasized",
                    press: function() {
                        var sId = oIdInput.getValue().trim();
                        var sName = oNameInput.getValue().trim();

                        if (!sId || !sName) {
                            MessageBox.error("Model ID and Name are required");
                            return;
                        }

                        var capabilitiesStr = oCapabilitiesInput.getValue().trim();
                        var capabilities = capabilitiesStr ? capabilitiesStr.split(",").map(function(c) { return c.trim(); }) : [];
                        var quality = parseInt(oQualityInput.getValue(), 10) || 75;
                        quality = Math.min(100, Math.max(0, quality));

                        var newModel = {
                            id: sId,
                            name: sName,
                            provider: oProviderInput.getValue().trim() || "Custom",
                            capabilities: capabilities,
                            quality: quality
                        };

                        var models = oViewModel.getProperty("/models") || [];
                        models.push(newModel);
                        oViewModel.setProperty("/models", models);

                        that._updateStatistics();
                        oDialog.close();
                        MessageToast.show("Model '" + sName + "' added successfully");
                    }
                }),
                endButton: new Button({
                    text: "Cancel",
                    press: function() {
                        oDialog.close();
                    }
                }),
                afterClose: function() {
                    oDialog.destroy();
                }
            });

            this.getView().addDependent(oDialog);
            oDialog.open();
        },

        onRefresh: function() {
            MessageToast.show("Refreshing model router data...");
            this._loadModels();
            this._loadAssignments();
            GraphIntegration.refresh();
        },

        onExportConfig: function() {
            var oViewModel = this.getView().getModel();
            var config = {
                assignments: oViewModel.getProperty("/assignments"),
                models: oViewModel.getProperty("/models"),
                routingConfig: oViewModel.getProperty("/routingConfig"),
                exportedAt: new Date().toISOString()
            };

            ApiService.exportData(config, "model_router_config", "json");
            MessageToast.show("Configuration exported");
        },

        _updateStatistics: function() {
            var oViewModel = this.getView().getModel();
            var assignments = oViewModel.getProperty("/assignments") || [];
            var models = oViewModel.getProperty("/models") || [];

            var assignedAgents = assignments.filter(function(a) { return a.status === "assigned"; }).length;
            var totalMatchScore = assignments.reduce(function(sum, a) { return sum + (a.matchScore || 0); }, 0);
            var avgMatchScore = assignments.length > 0 ? Math.round(totalMatchScore / assignments.length) : 0;

            oViewModel.setProperty("/stats", {
                totalAgents: assignments.length,
                assignedAgents: assignedAgents,
                totalModels: models.length,
                avgMatchScore: avgMatchScore
            });
        },

        _buildNetworkGraphData: function() {
            var oGraphModel = this.getView().getModel("graph");
            var oViewModel = this.getView().getModel();
            var assignments = oViewModel.getProperty("/assignments") || [];
            var models = oViewModel.getProperty("/models") || [];

            var nodes = [];
            var lines = [];
            var groups = [
                { key: "agents", title: "Agents", description: "AI Agents", icon: "sap-icon://puzzle" },
                { key: "models", title: "Models", description: "LLM Models", icon: "sap-icon://machine" }
            ];

            // Add agent nodes
            assignments.forEach(function(assignment, idx) {
                nodes.push({
                    id: "agent-" + assignment.agentId,
                    name: assignment.agentName,
                    description: assignment.agentType,
                    icon: "sap-icon://puzzle",
                    status: assignment.status === "assigned" ? "Success" : "Warning",
                    group: "agents",
                    x: 100,
                    y: idx * 80 + 50
                });
            });

            // Add model nodes
            models.forEach(function(model, idx) {
                nodes.push({
                    id: "model-" + model.id,
                    name: model.name,
                    description: model.provider + " | Quality: " + model.quality,
                    icon: "sap-icon://machine",
                    status: "Success",
                    group: "models",
                    x: 400,
                    y: idx * 80 + 50
                });
            });

            // Add edges for assignments
            assignments.forEach(function(assignment) {
                if (assignment.modelId) {
                    lines.push({
                        from: "agent-" + assignment.agentId,
                        to: "model-" + assignment.modelId,
                        label: "Score: " + (assignment.matchScore || 0),
                        status: assignment.matchScore >= 70 ? "Success" : (assignment.matchScore >= 40 ? "Warning" : "Error")
                    });
                }
            });

            oGraphModel.setProperty("/nodes", nodes);
            oGraphModel.setProperty("/lines", lines);
            oGraphModel.setProperty("/groups", groups);

            console.log(`Model Router graph updated: ${nodes.length} nodes, ${lines.length} edges`);
        },

        _buildProcessFlowData: function() {
            var oViewModel = this.getView().getModel();
            var assignments = oViewModel.getProperty("/assignments") || [];

            // Build lanes for task routing visualization
            var lanes = [
                { id: "request", label: "Request", position: 0 },
                { id: "routing", label: "Model Routing", position: 1 },
                { id: "inference", label: "Inference", position: 2 },
                { id: "response", label: "Response", position: 3 }
            ];

            // Build nodes for process flow
            var flowNodes = [
                {
                    id: "incoming-request",
                    lane: "request",
                    title: "Incoming Request",
                    state: "Positive",
                    texts: ["Agent task received"],
                    position: 0,
                    children: ["router-node"]
                },
                {
                    id: "router-node",
                    lane: "routing",
                    title: "Model Router",
                    state: "Neutral",
                    texts: ["Selecting optimal model"],
                    position: 1,
                    children: assignments.slice(0, 3).map(function(a) { return "model-" + (a.modelId || "unassigned"); })
                }
            ];

            // Add model nodes
            var modelsAdded = {};
            assignments.slice(0, 3).forEach(function(assignment) {
                var modelId = assignment.modelId || "unassigned";
                if (!modelsAdded[modelId]) {
                    flowNodes.push({
                        id: "model-" + modelId,
                        lane: "inference",
                        title: modelId === "unassigned" ? "Unassigned" : modelId,
                        state: modelId === "unassigned" ? "Negative" : "Positive",
                        texts: [assignment.agentName],
                        position: 2,
                        children: ["response-node"]
                    });
                    modelsAdded[modelId] = true;
                }
            });

            flowNodes.push({
                id: "response-node",
                lane: "response",
                title: "Response",
                state: "Planned",
                texts: ["Return to agent"],
                position: 3,
                children: []
            });

            // Store for potential use by ProcessFlow component
            oViewModel.setProperty("/processFlowData", { lanes: lanes, nodes: flowNodes });
        },

        // ========================================================================
        // LIVE METRICS
        // ========================================================================

        _startMetricsPolling: function() {
            var that = this;
            var oViewModel = this.getView().getModel();

            this._metricsInterval = setInterval(function() {
                if (oViewModel.getProperty("/metricsConfig/autoRefresh")) {
                    that._fetchLiveMetrics();
                }
            }, oViewModel.getProperty("/metricsConfig/refreshInterval") || 5000);
        },

        _fetchLiveMetrics: function() {
            var that = this;
            var oViewModel = this.getView().getModel();

            // Day 25: Fetch from Day 24 stats API endpoint
            fetch('http://localhost:8080/api/v1/model-router/stats')
                .then(response => response.json())
                .then(data => {
                    oViewModel.setProperty("/liveMetrics", {
                        totalDecisions: data.total_decisions || 0,
                        successRate: data.success_rate || 0,
                        avgLatency: data.avg_latency_ms || 0,
                        fallbacksUsed: data.fallbacks_used || 0,
                        recentDecisions: (data.recent_decisions || []).map(function(d) {
                            return {
                                taskType: d.task_type,
                                selectedModel: d.selected_model,
                                score: d.score,
                                latency: d.latency_ms,
                                success: d.success,
                                timestamp: new Date(d.timestamp).toLocaleTimeString()
                            };
                        })
                    });
                })
                .catch(function() {
                    // Simulate metrics for demo
                    that._simulateLiveMetrics();
                });
        },

        _simulateLiveMetrics: function() {
            var oViewModel = this.getView().getModel();
            var currentMetrics = oViewModel.getProperty("/liveMetrics");

            // Simulate incremental updates
            var newDecision = {
                taskType: ['coding', 'reasoning', 'creative', 'math'][Math.floor(Math.random() * 4)],
                selectedModel: this._mockModels[Math.floor(Math.random() * this._mockModels.length)].name,
                score: Math.floor(Math.random() * 30) + 70,
                latency: Math.floor(Math.random() * 500) + 100,
                success: Math.random() > 0.1,
                timestamp: new Date().toLocaleTimeString()
            };

            var recentDecisions = currentMetrics.recentDecisions || [];
            recentDecisions.unshift(newDecision);
            if (recentDecisions.length > 10) {
                recentDecisions = recentDecisions.slice(0, 10);
            }

            var totalDecisions = (currentMetrics.totalDecisions || 0) + 1;
            var successCount = recentDecisions.filter(function(d) { return d.success; }).length;
            var successRate = Math.round((successCount / recentDecisions.length) * 100);
            var avgLatency = Math.round(recentDecisions.reduce(function(sum, d) { return sum + d.latency; }, 0) / recentDecisions.length);

            oViewModel.setProperty("/liveMetrics", {
                totalDecisions: totalDecisions,
                successRate: successRate,
                avgLatency: avgLatency,
                fallbacksUsed: Math.floor(totalDecisions * 0.05),
                recentDecisions: recentDecisions
            });
        },

        onAutoRefreshToggle: function(oEvent) {
            var bState = oEvent.getParameter("state");
            var oViewModel = this.getView().getModel();
            oViewModel.setProperty("/metricsConfig/autoRefresh", bState);

            if (bState) {
                MessageToast.show("Auto-refresh enabled");
            } else {
                MessageToast.show("Auto-refresh disabled");
            }
        },

        onRefreshMetrics: function() {
            this._fetchLiveMetrics();
            MessageToast.show("Metrics refreshed");
        },

        onRefreshNetwork: function() {
            this._buildNetworkGraphData();
            GraphIntegration.refresh();
            MessageToast.show("Network graph refreshed");
        },

        onRefreshTaskFlow: function() {
            this._buildProcessFlowData();
            GraphIntegration.refresh();
            MessageToast.show("Task flow refreshed");
        },

        onModelPress: function(oEvent) {
            var oTile = oEvent.getSource();
            var oContext = oTile.getBindingContext();
            if (oContext) {
                var model = oContext.getObject();
                MessageToast.show("Model: " + model.name + " | Quality: " + model.quality + "%");
            }
        },

        onExit: function() {
            if (this._metricsInterval) {
                clearInterval(this._metricsInterval);
            }
            GraphIntegration.destroy();
        }
    });
});
