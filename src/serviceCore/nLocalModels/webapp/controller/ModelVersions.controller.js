sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function (Controller, JSONModel, Fragment, MessageBox, MessageToast) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.ModelVersions", {

        onInit: function () {
            // Initialize versions model for TreeTable hierarchical data
            var oVersionsModel = new JSONModel({
                modelHierarchy: []
            });
            this.getView().setModel(oVersionsModel, "versions");

            // Initialize filters model
            var oFiltersModel = new JSONModel({
                selectedStatus: "ALL",
                statusOptions: [
                    { key: "ALL", text: "All Statuses" },
                    { key: "PRODUCTION", text: "Production" },
                    { key: "STAGING", text: "Staging" },
                    { key: "CANARY", text: "Canary" },
                    { key: "ARCHIVED", text: "Archived" }
                ]
            });
            this.getView().setModel(oFiltersModel, "filters");

            // Initialize model comparison model for T-Account fragment
            var oModelComparisonModel = new JSONModel({
                versionA: {
                    modelName: "",
                    version: "",
                    status: "",
                    createdDate: "",
                    promotedBy: "",
                    trainingExperimentId: "",
                    trainingMetrics: {
                        finalLoss: 0,
                        accuracy: 0,
                        trainingTime: "",
                        epochsCompleted: 0
                    },
                    inferenceMetrics: {
                        latencyP50: 0,
                        latencyP95: 0,
                        throughput: 0,
                        errorRate: 0
                    },
                    abTesting: {
                        trafficPercent: 0,
                        totalRequests: 0,
                        successRate: 0
                    }
                },
                versionB: {
                    modelName: "",
                    version: "",
                    status: "",
                    createdDate: "",
                    promotedBy: "",
                    trainingExperimentId: "",
                    trainingMetrics: {
                        finalLoss: 0,
                        accuracy: 0,
                        trainingTime: "",
                        epochsCompleted: 0
                    },
                    inferenceMetrics: {
                        latencyP50: 0,
                        latencyP95: 0,
                        throughput: 0,
                        errorRate: 0
                    },
                    abTesting: {
                        trafficPercent: 0,
                        totalRequests: 0,
                        successRate: 0
                    }
                },
                deltas: {
                    finalLoss: { display: "", winner: "none" },
                    latencyP50: { display: "", winner: "none" },
                    latencyP95: { display: "", winner: "none" },
                    accuracy: { display: "", winner: "none" },
                    throughput: { display: "", winner: "none" },
                    errorRate: { display: "", winner: "none" },
                    successRate: { display: "", winner: "none" },
                    overall: { winner: "none", recommendation: "" },
                    summary: {
                        metricsWonByA: 0,
                        metricsWonByB: 0,
                        metricsTied: 0
                    }
                }
            });
            this.getView().setModel(oModelComparisonModel, "modelComparison");

            // Load versions from API
            this._loadVersions();
        },

        // ==================== Event Handlers ====================

        onPromote: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("versions");
            var oVersion = oContext.getObject();
            var sCurrentStatus = oVersion.status;
            var sNextStatus = this._getNextStatus(sCurrentStatus);

            if (!sNextStatus) {
                MessageToast.show("Cannot promote version with status: " + sCurrentStatus);
                return;
            }

            var that = this;
            MessageBox.confirm(
                "Are you sure you want to promote version '" + oVersion.name + "' from " + sCurrentStatus + " to " + sNextStatus + "?",
                {
                    title: "Confirm Promotion",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            that._promoteVersion(oVersion.id, sNextStatus)
                                .then(function () {
                                    oContext.getModel().setProperty(oContext.getPath() + "/status", sNextStatus);
                                    MessageToast.show("Version promoted to " + sNextStatus);
                                })
                                .catch(function (error) {
                                    MessageBox.error("Failed to promote version: " + error.message);
                                });
                        }
                    }
                }
            );
        },

        onRollback: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("versions");
            var oVersion = oContext.getObject();
            var that = this;

            // Create dialog for rollback reason
            var oDialog = new sap.m.Dialog({
                title: "Rollback Version",
                type: "Message",
                content: [
                    new sap.m.Label({ text: "Please provide a reason for rollback:", labelFor: "rollbackReason" }),
                    new sap.m.TextArea("rollbackReason", { width: "100%", placeholder: "Enter rollback reason..." })
                ],
                beginButton: new sap.m.Button({
                    text: "Rollback",
                    type: "Emphasized",
                    press: function () {
                        var sReason = sap.ui.getCore().byId("rollbackReason").getValue();
                        if (!sReason) {
                            MessageToast.show("Please provide a reason");
                            return;
                        }
                        that._rollbackVersion(oVersion.id, sReason)
                            .then(function (oResult) {
                                oContext.getModel().setProperty(oContext.getPath() + "/status", "ARCHIVED");
                                oContext.getModel().setProperty(oContext.getPath() + "/ROLLBACK_FROM_ID", oResult.rollbackFromId);
                                MessageToast.show("Version rolled back successfully");
                                oDialog.close();
                            })
                            .catch(function (error) {
                                MessageBox.error("Failed to rollback version: " + error.message);
                            });
                    }
                }),
                endButton: new sap.m.Button({
                    text: "Cancel",
                    press: function () { oDialog.close(); }
                }),
                afterClose: function () { oDialog.destroy(); }
            });
            oDialog.open();
        },

        onArchive: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("versions");
            var oVersion = oContext.getObject();

            MessageBox.confirm(
                "Are you sure you want to archive version '" + oVersion.name + "'? This will remove it from active deployments.",
                {
                    title: "Confirm Archive",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            oContext.getModel().setProperty(oContext.getPath() + "/status", "ARCHIVED");
                            oContext.getModel().setProperty(oContext.getPath() + "/traffic", 0);
                            MessageToast.show("Version archived successfully");
                        }
                    }
                }
            );
        },

        onCreateVersion: function (oEvent) {
            var that = this;
            var oDialog = new sap.m.Dialog({
                title: "Create New Version",
                contentWidth: "400px",
                content: [
                    new sap.m.VBox({
                        items: [
                            new sap.m.Label({ text: "Version Name:", labelFor: "versionName" }),
                            new sap.m.Input("versionName", { placeholder: "e.g., v3.2.2" }),
                            new sap.m.Label({ text: "Base Model:", labelFor: "baseModel" }),
                            new sap.m.Select("baseModel", {
                                items: [
                                    new sap.ui.core.Item({ key: "llama", text: "Llama 3.2" }),
                                    new sap.ui.core.Item({ key: "qwen", text: "Qwen 2.5" }),
                                    new sap.ui.core.Item({ key: "phi", text: "Phi-2" })
                                ]
                            }),
                            new sap.m.Label({ text: "Initial Status:", labelFor: "initialStatus" }),
                            new sap.m.Select("initialStatus", {
                                items: [
                                    new sap.ui.core.Item({ key: "CANARY", text: "Canary" }),
                                    new sap.ui.core.Item({ key: "STAGING", text: "Staging" })
                                ]
                            })
                        ]
                    }).addStyleClass("sapUiSmallMargin")
                ],
                beginButton: new sap.m.Button({
                    text: "Create",
                    type: "Emphasized",
                    press: function () {
                        var sName = sap.ui.getCore().byId("versionName").getValue();
                        var sStatus = sap.ui.getCore().byId("initialStatus").getSelectedKey();
                        if (!sName) {
                            MessageToast.show("Please enter a version name");
                            return;
                        }
                        MessageToast.show("Version '" + sName + "' created with status " + sStatus);
                        oDialog.close();
                        that._loadVersions();
                    }
                }),
                endButton: new sap.m.Button({
                    text: "Cancel",
                    press: function () { oDialog.close(); }
                }),
                afterClose: function () { oDialog.destroy(); }
            });
            oDialog.open();
        },

        onFilterChange: function (oEvent) {
            var sSelectedStatus = oEvent.getParameter("selectedItem").getKey();
            this.getView().getModel("filters").setProperty("/selectedStatus", sSelectedStatus);

            var oTreeTable = this.byId("versionsTreeTable");
            if (!oTreeTable) {
                return;
            }

            var oBinding = oTreeTable.getBinding("rows");
            if (oBinding) {
                if (sSelectedStatus === "ALL") {
                    oBinding.filter([]);
                } else {
                    oBinding.filter([
                        new sap.ui.model.Filter("status", sap.ui.model.FilterOperator.EQ, sSelectedStatus)
                    ]);
                }
            }
        },

        onRefresh: function () {
            MessageToast.show("Refreshing version data...");
            this._loadVersions();
        },

        onCompareVersions: function () {
            var oTreeTable = this.byId("versionsTreeTable");
            if (!oTreeTable) {
                MessageToast.show("TreeTable not found");
                return;
            }

            var aSelectedIndices = oTreeTable.getSelectedIndices();
            if (aSelectedIndices.length !== 2) {
                MessageBox.warning("Please select exactly 2 versions to compare");
                return;
            }

            var aVersions = [];
            aSelectedIndices.forEach(function (iIndex) {
                var oContext = oTreeTable.getContextByIndex(iIndex);
                if (oContext) {
                    aVersions.push(oContext.getObject());
                }
            });

            if (aVersions.length === 2) {
                // Load versions into comparison model
                this._loadVersionIntoSlot(aVersions[0], "A");
                this._loadVersionIntoSlot(aVersions[1], "B");
                this._calculateModelDeltas();

                // Open T-Account comparison dialog
                this._openModelComparisonDialog();
            }
        },

        _openModelComparisonDialog: function () {
            var that = this;
            var oView = this.getView();

            if (!this._oModelComparisonDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "llm.server.dashboard.view.fragments.TAccountModelComparison",
                    controller: this
                }).then(function (oFragment) {
                    that._oModelComparisonDialog = new sap.m.Dialog({
                        title: "Model Version Comparison",
                        contentWidth: "90%",
                        contentHeight: "85%",
                        resizable: true,
                        draggable: true,
                        content: [oFragment],
                        endButton: new sap.m.Button({
                            text: "Close",
                            press: function () {
                                that._oModelComparisonDialog.close();
                            }
                        })
                    });
                    oView.addDependent(that._oModelComparisonDialog);
                    that._oModelComparisonDialog.open();
                });
            } else {
                this._oModelComparisonDialog.open();
            }
        },

        _loadVersionIntoSlot: function (oVersion, sSlot) {
            var oModel = this.getView().getModel("modelComparison");
            var sBasePath = "/version" + sSlot;

            // Map version data to comparison model structure
            oModel.setProperty(sBasePath + "/modelName", oVersion.name || "");
            oModel.setProperty(sBasePath + "/version", oVersion.id || "");
            oModel.setProperty(sBasePath + "/status", oVersion.status || "");
            oModel.setProperty(sBasePath + "/createdDate", oVersion.createdAt || "");
            oModel.setProperty(sBasePath + "/promotedBy", oVersion.promotedBy || "system");
            oModel.setProperty(sBasePath + "/trainingExperimentId", oVersion.experimentId || (oVersion.id ? "EXP-" + oVersion.id : ""));

            // Training metrics
            oModel.setProperty(sBasePath + "/trainingMetrics/finalLoss", oVersion.finalLoss || 0);
            oModel.setProperty(sBasePath + "/trainingMetrics/accuracy", oVersion.accuracy ? (oVersion.accuracy * 100).toFixed(1) : 0);
            oModel.setProperty(sBasePath + "/trainingMetrics/trainingTime", oVersion.trainingTime || "");
            oModel.setProperty(sBasePath + "/trainingMetrics/epochsCompleted", oVersion.epochs || 0);

            // Inference metrics
            oModel.setProperty(sBasePath + "/inferenceMetrics/latencyP50", oVersion.latencyP50 || 0);
            oModel.setProperty(sBasePath + "/inferenceMetrics/latencyP95", oVersion.latencyP95 || 0);
            oModel.setProperty(sBasePath + "/inferenceMetrics/throughput", oVersion.throughput || 0);
            oModel.setProperty(sBasePath + "/inferenceMetrics/errorRate", oVersion.errorRate || 0);

            // A/B testing metrics
            oModel.setProperty(sBasePath + "/abTesting/trafficPercent", oVersion.traffic || 0);
            oModel.setProperty(sBasePath + "/abTesting/totalRequests", oVersion.totalRequests || 0);
            oModel.setProperty(sBasePath + "/abTesting/successRate", oVersion.successRate || 0);
        },

        _calculateModelDeltas: function () {
            var oModel = this.getView().getModel("modelComparison");
            var oVersionA = oModel.getProperty("/versionA");
            var oVersionB = oModel.getProperty("/versionB");

            var nAWins = 0;
            var nBWins = 0;
            var nTies = 0;

            // Calculate final loss delta (lower is better)
            var fLossA = parseFloat(oVersionA.trainingMetrics.finalLoss) || 0;
            var fLossB = parseFloat(oVersionB.trainingMetrics.finalLoss) || 0;
            var fLossDelta = Math.abs(fLossA - fLossB).toFixed(3);
            var sLossWinner = fLossA < fLossB ? "A" : (fLossB < fLossA ? "B" : "none");
            if (sLossWinner === "A") { nAWins++; } else if (sLossWinner === "B") { nBWins++; } else { nTies++; }
            oModel.setProperty("/deltas/finalLoss/display", fLossDelta + " diff");
            oModel.setProperty("/deltas/finalLoss/winner", sLossWinner);

            // Calculate latency P50 delta (lower is better)
            var nLatP50A = parseFloat(oVersionA.inferenceMetrics.latencyP50) || 0;
            var nLatP50B = parseFloat(oVersionB.inferenceMetrics.latencyP50) || 0;
            var nLatP50Delta = Math.abs(nLatP50A - nLatP50B);
            var sLatP50Winner = nLatP50A < nLatP50B ? "A" : (nLatP50B < nLatP50A ? "B" : "none");
            if (sLatP50Winner === "A") { nAWins++; } else if (sLatP50Winner === "B") { nBWins++; } else { nTies++; }
            oModel.setProperty("/deltas/latencyP50/display", nLatP50Delta + "ms faster");
            oModel.setProperty("/deltas/latencyP50/winner", sLatP50Winner);

            // Calculate latency P95 delta (lower is better)
            var nLatP95A = parseFloat(oVersionA.inferenceMetrics.latencyP95) || 0;
            var nLatP95B = parseFloat(oVersionB.inferenceMetrics.latencyP95) || 0;
            var nLatP95Delta = Math.abs(nLatP95A - nLatP95B);
            var sLatP95Winner = nLatP95A < nLatP95B ? "A" : (nLatP95B < nLatP95A ? "B" : "none");
            if (sLatP95Winner === "A") { nAWins++; } else if (sLatP95Winner === "B") { nBWins++; } else { nTies++; }
            oModel.setProperty("/deltas/latencyP95/display", nLatP95Delta + "ms faster");
            oModel.setProperty("/deltas/latencyP95/winner", sLatP95Winner);

            // Calculate accuracy delta (higher is better)
            var fAccA = parseFloat(oVersionA.trainingMetrics.accuracy) || 0;
            var fAccB = parseFloat(oVersionB.trainingMetrics.accuracy) || 0;
            var fAccDelta = Math.abs(fAccA - fAccB).toFixed(1);
            var sAccWinner = fAccA > fAccB ? "A" : (fAccB > fAccA ? "B" : "none");
            if (sAccWinner === "A") { nAWins++; } else if (sAccWinner === "B") { nBWins++; } else { nTies++; }
            oModel.setProperty("/deltas/accuracy/display", fAccDelta + "% higher");
            oModel.setProperty("/deltas/accuracy/winner", sAccWinner);

            // Calculate throughput delta (higher is better)
            var nThroughputA = parseFloat(oVersionA.inferenceMetrics.throughput) || 0;
            var nThroughputB = parseFloat(oVersionB.inferenceMetrics.throughput) || 0;
            var nThroughputDelta = Math.abs(nThroughputA - nThroughputB);
            var sThroughputWinner = nThroughputA > nThroughputB ? "A" : (nThroughputB > nThroughputA ? "B" : "none");
            if (sThroughputWinner === "A") { nAWins++; } else if (sThroughputWinner === "B") { nBWins++; } else { nTies++; }
            oModel.setProperty("/deltas/throughput/display", nThroughputDelta + " TPS higher");
            oModel.setProperty("/deltas/throughput/winner", sThroughputWinner);

            // Calculate error rate delta (lower is better)
            var fErrA = parseFloat(oVersionA.inferenceMetrics.errorRate) || 0;
            var fErrB = parseFloat(oVersionB.inferenceMetrics.errorRate) || 0;
            var fErrDelta = Math.abs(fErrA - fErrB).toFixed(2);
            var sErrWinner = fErrA < fErrB ? "A" : (fErrB < fErrA ? "B" : "none");
            if (sErrWinner === "A") { nAWins++; } else if (sErrWinner === "B") { nBWins++; } else { nTies++; }
            oModel.setProperty("/deltas/errorRate/display", fErrDelta + "% lower");
            oModel.setProperty("/deltas/errorRate/winner", sErrWinner);

            // Calculate success rate delta (higher is better)
            var fSuccessA = parseFloat(oVersionA.abTesting.successRate) || 0;
            var fSuccessB = parseFloat(oVersionB.abTesting.successRate) || 0;
            var fSuccessDelta = Math.abs(fSuccessA - fSuccessB).toFixed(1);
            var sSuccessWinner = fSuccessA > fSuccessB ? "A" : (fSuccessB > fSuccessA ? "B" : "none");
            if (sSuccessWinner === "A") { nAWins++; } else if (sSuccessWinner === "B") { nBWins++; } else { nTies++; }
            oModel.setProperty("/deltas/successRate/display", fSuccessDelta + "% higher");
            oModel.setProperty("/deltas/successRate/winner", sSuccessWinner);

            // Set summary
            oModel.setProperty("/deltas/summary/metricsWonByA", nAWins);
            oModel.setProperty("/deltas/summary/metricsWonByB", nBWins);
            oModel.setProperty("/deltas/summary/metricsTied", nTies);

            // Determine overall winner
            var sOverallWinner = nAWins > nBWins ? "A" : (nBWins > nAWins ? "B" : "none");
            oModel.setProperty("/deltas/overall/winner", sOverallWinner);

            // Set recommendation
            var sRecommendation = "";
            if (sOverallWinner === "A") {
                sRecommendation = "Version A (" + oVersionA.modelName + ") wins " + nAWins + " of " + (nAWins + nBWins + nTies) + " metrics. Consider promoting to production.";
            } else if (sOverallWinner === "B") {
                sRecommendation = "Version B (" + oVersionB.modelName + ") wins " + nBWins + " of " + (nAWins + nBWins + nTies) + " metrics. Consider promoting to production.";
            } else {
                sRecommendation = "No clear winner. Consider running an A/B test to gather more data.";
            }
            oModel.setProperty("/deltas/overall/recommendation", sRecommendation);
        },

        onCloseModelComparison: function () {
            if (this._oModelComparisonDialog) {
                this._oModelComparisonDialog.close();
            }
        },

        onPromoteWinnerVersion: function () {
            var oModel = this.getView().getModel("modelComparison");
            var sWinner = oModel.getProperty("/deltas/overall/winner");

            if (!sWinner || sWinner === "none") {
                MessageToast.show("No clear winner to promote");
                return;
            }

            var oWinnerVersion = oModel.getProperty("/version" + sWinner);
            var that = this;

            MessageBox.confirm(
                "Are you sure you want to promote " + oWinnerVersion.modelName + " (" + oWinnerVersion.version + ") to PRODUCTION?",
                {
                    title: "Confirm Promotion",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            that._promoteVersion(oWinnerVersion.version, "PRODUCTION")
                                .then(function () {
                                    MessageToast.show("Version promoted to PRODUCTION successfully");
                                    that.onCloseModelComparison();
                                    that._loadVersions();
                                })
                                .catch(function (error) {
                                    MessageBox.error("Failed to promote version: " + error.message);
                                });
                        }
                    }
                }
            );
        },

        onSetupABTest: function () {
            var oModel = this.getView().getModel("modelComparison");
            var oVersionA = oModel.getProperty("/versionA");
            var oVersionB = oModel.getProperty("/versionB");
            var that = this;

            var oDialog = new sap.m.Dialog({
                title: "Set Up A/B Test",
                contentWidth: "400px",
                content: [
                    new sap.m.VBox({
                        items: [
                            new sap.m.Label({ text: "Version A: " + oVersionA.modelName }),
                            new sap.m.Label({ text: "Version B: " + oVersionB.modelName, class: "sapUiSmallMarginBottom" }),
                            new sap.m.Label({ text: "Traffic Split (% to Version A):", labelFor: "trafficSplit" }),
                            new sap.m.Slider("trafficSplit", {
                                min: 0,
                                max: 100,
                                value: 50,
                                enableTickmarks: true,
                                showAdvancedTooltip: true
                            }),
                            new sap.m.Label({ text: "Test Duration (hours):", labelFor: "testDuration" }),
                            new sap.m.Input("testDuration", { type: "Number", value: "24" }),
                            new sap.m.Label({ text: "Success Metric:", labelFor: "successMetric" }),
                            new sap.m.Select("successMetric", {
                                items: [
                                    new sap.ui.core.Item({ key: "accuracy", text: "Accuracy" }),
                                    new sap.ui.core.Item({ key: "latency", text: "Latency P50" }),
                                    new sap.ui.core.Item({ key: "errorRate", text: "Error Rate" }),
                                    new sap.ui.core.Item({ key: "successRate", text: "Success Rate" })
                                ]
                            })
                        ]
                    }).addStyleClass("sapUiSmallMargin")
                ],
                beginButton: new sap.m.Button({
                    text: "Start A/B Test",
                    type: "Emphasized",
                    press: function () {
                        var nTrafficSplit = sap.ui.getCore().byId("trafficSplit").getValue();
                        var nDuration = sap.ui.getCore().byId("testDuration").getValue();
                        var sMetric = sap.ui.getCore().byId("successMetric").getSelectedKey();
                        MessageToast.show("A/B Test started: " + nTrafficSplit + "% to A, " + (100 - nTrafficSplit) + "% to B for " + nDuration + "h");
                        oDialog.close();
                        that.onCloseModelComparison();
                    }
                }),
                endButton: new sap.m.Button({
                    text: "Cancel",
                    press: function () { oDialog.close(); }
                }),
                afterClose: function () { oDialog.destroy(); }
            });
            oDialog.open();
        },

        onRollbackVersion: function () {
            var oModel = this.getView().getModel("modelComparison");
            var oVersionA = oModel.getProperty("/versionA");
            var oVersionB = oModel.getProperty("/versionB");
            var that = this;

            var oDialog = new sap.m.Dialog({
                title: "Rollback Version",
                contentWidth: "400px",
                content: [
                    new sap.m.VBox({
                        items: [
                            new sap.m.Label({ text: "Select version to rollback:", labelFor: "rollbackSelect" }),
                            new sap.m.Select("rollbackSelect", {
                                width: "100%",
                                items: [
                                    new sap.ui.core.Item({ key: "A", text: oVersionA.modelName + " (" + oVersionA.version + ")" }),
                                    new sap.ui.core.Item({ key: "B", text: oVersionB.modelName + " (" + oVersionB.version + ")" })
                                ]
                            }),
                            new sap.m.Label({ text: "Rollback Reason:", labelFor: "rollbackReasonInput" }),
                            new sap.m.TextArea("rollbackReasonInput", { width: "100%", rows: 3 })
                        ]
                    }).addStyleClass("sapUiSmallMargin")
                ],
                beginButton: new sap.m.Button({
                    text: "Rollback",
                    type: "Reject",
                    press: function () {
                        var sSelected = sap.ui.getCore().byId("rollbackSelect").getSelectedKey();
                        var sReason = sap.ui.getCore().byId("rollbackReasonInput").getValue();
                        var oSelectedVersion = sSelected === "A" ? oVersionA : oVersionB;

                        if (!sReason) {
                            MessageToast.show("Please provide a rollback reason");
                            return;
                        }

                        that._rollbackVersion(oSelectedVersion.version, sReason)
                            .then(function () {
                                MessageToast.show("Version rolled back successfully");
                                oDialog.close();
                                that.onCloseModelComparison();
                                that._loadVersions();
                            })
                            .catch(function (error) {
                                MessageBox.error("Failed to rollback: " + error.message);
                            });
                    }
                }),
                endButton: new sap.m.Button({
                    text: "Cancel",
                    press: function () { oDialog.close(); }
                }),
                afterClose: function () { oDialog.destroy(); }
            });
            oDialog.open();
        },

        onExportComparisonReport: function () {
            var oModel = this.getView().getModel("modelComparison");
            var oData = oModel.getData();

            var oReport = {
                exportDate: new Date().toISOString(),
                versionA: oData.versionA,
                versionB: oData.versionB,
                comparison: {
                    deltas: oData.deltas,
                    winner: oData.deltas.overall.winner,
                    recommendation: oData.deltas.overall.recommendation
                }
            };

            // Create downloadable JSON
            var sJson = JSON.stringify(oReport, null, 2);
            var oBlob = new Blob([sJson], { type: "application/json" });
            var sUrl = URL.createObjectURL(oBlob);

            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = "model_comparison_" + new Date().toISOString().split("T")[0] + ".json";
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);

            MessageToast.show("Comparison report exported successfully");
        },

        onNavigateToExperiment: function (oEvent) {
            var sExperimentId = oEvent.getSource().getText();
            MessageToast.show("Navigating to experiment: " + sExperimentId);
            // Navigation logic would go here
        },

        onViewAuditLog: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("versions");
            var oVersion = oContext.getObject();

            var oDialog = new sap.m.Dialog({
                title: "Audit Log - " + oVersion.name,
                contentWidth: "600px",
                content: [
                    new sap.m.List({
                        items: [
                            new sap.m.StandardListItem({ title: "Created", description: oVersion.createdAt + " by " + (oVersion.promotedBy || "system"), icon: "sap-icon://create" }),
                            new sap.m.StandardListItem({ title: "Status: " + oVersion.status, description: "Current deployment status", icon: "sap-icon://status-positive" }),
                            new sap.m.StandardListItem({ title: "Traffic: " + oVersion.traffic + "%", description: "Current traffic allocation", icon: "sap-icon://pie-chart" })
                        ]
                    })
                ],
                endButton: new sap.m.Button({
                    text: "Close",
                    press: function () { oDialog.close(); }
                }),
                afterClose: function () { oDialog.destroy(); }
            });
            oDialog.open();
        },

        onTrafficChange: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("versions");
            var oVersion = oContext.getObject();
            var nNewTraffic = oEvent.getParameter("value");
            var that = this;

            if (nNewTraffic < 0 || nNewTraffic > 100) {
                MessageToast.show("Traffic must be between 0 and 100");
                return;
            }

            this._updateTraffic(oVersion.id, nNewTraffic)
                .then(function () {
                    oContext.getModel().setProperty(oContext.getPath() + "/traffic", nNewTraffic);
                    MessageToast.show("Traffic updated to " + nNewTraffic + "%");
                })
                .catch(function (error) {
                    MessageBox.error("Failed to update traffic: " + error.message);
                });
        },

        // ==================== Helper Methods ====================

        _getNextStatus: function (sCurrentStatus) {
            var aStatusOrder = ["CANARY", "STAGING", "PRODUCTION"];
            var iCurrentIndex = aStatusOrder.indexOf(sCurrentStatus);
            if (iCurrentIndex >= 0 && iCurrentIndex < aStatusOrder.length - 1) {
                return aStatusOrder[iCurrentIndex + 1];
            }
            return null;
        },

        // ==================== API Integration Methods (Placeholders) ====================

        _loadVersions: function () {
            var that = this;
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent ? oComponent.getApiBaseUrl() : "";
            var oVersionsModel = this.getView().getModel("versions");

            // Load live models and map to a simple hierarchy (no mock fallback)
            return fetch(sApiBaseUrl + "/v1/models")
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to fetch models");
                    return response.json();
                })
                .then(function (data) {
                    var aModels = data.data || data.models || [];
                    var aTree = aModels.map(function (m) {
                        var createdAt = m.created ? new Date(m.created * 1000).toISOString() : "";
                        var status = (m.status || m.state || "PRODUCTION").toUpperCase();

                        return {
                            name: m.display_name || m.name || m.id,
                            id: m.id,
                            level: 1,
                            status: status,
                            accuracy: m.accuracy || 0,
                            latencyP50: m.latency_p50 || m.p50 || 0,
                            latencyP95: m.latency_p95 || m.p95 || 0,
                            trafficPercent: m.traffic_percent || 100,
                            createdAt: createdAt,
                            promotedBy: m.owned_by || "system",
                            children: [
                                {
                                    name: m.version || "current",
                                    id: m.id,
                                    level: 3,
                                    status: status,
                                    accuracy: m.accuracy || 0,
                                    latencyP50: m.latency_p50 || m.p50 || 0,
                                    latencyP95: m.latency_p95 || m.p95 || 0,
                                    trafficPercent: m.traffic_percent || 100,
                                    createdAt: createdAt,
                                    promotedBy: m.owned_by || "system",
                                    children: []
                                }
                            ]
                        };
                    });

                    oVersionsModel.setProperty("/modelHierarchy", aTree);
                    MessageToast.show("Versions loaded from API");
                })
                .catch(function (error) {
                    console.error("Failed to load versions:", error.message);
                    oVersionsModel.setProperty("/modelHierarchy", []);
                    MessageBox.error("Version catalog unavailable. Please verify the API.");
                });
        },

        _promoteVersion: function (sId, sTargetStatus) {
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent ? oComponent.getApiBaseUrl() : "";

            // POST /v1/models/versions/{id}/promote
            return fetch(sApiBaseUrl + "/v1/models/versions/" + sId + "/promote", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ targetStatus: sTargetStatus })
            }).then(function (response) {
                if (!response.ok) throw new Error("Promotion failed");
                return response.json();
            });
        },

        _rollbackVersion: function (sId, sReason) {
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent ? oComponent.getApiBaseUrl() : "";

            // POST /v1/models/versions/{id}/rollback
            return fetch(sApiBaseUrl + "/v1/models/versions/" + sId + "/rollback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ reason: sReason })
            }).then(function (response) {
                if (!response.ok) throw new Error("Rollback failed");
                return response.json();
            });
        },

        _updateTraffic: function (sId, nPercentage) {
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent ? oComponent.getApiBaseUrl() : "";

            // PUT /v1/models/deployments/{id}/traffic
            return fetch(sApiBaseUrl + "/v1/models/deployments/" + sId + "/traffic", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ percentage: nPercentage })
            }).then(function (response) {
                if (!response.ok) throw new Error("Traffic update failed");
                return response.json();
            });
        }
    });
});
