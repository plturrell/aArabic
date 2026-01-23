sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function (Controller, JSONModel, MessageBox, MessageToast) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.TrainingDashboard", {

        onInit: function () {
            // Initialize training model
            var oTrainingModel = new JSONModel({
                algorithms: [],
                jobs: [],
                experiments: [],
                ktoMetrics: {
                    winRate: 0,
                    klDivergence: 0,
                    desirableCount: 0,
                    undesirableCount: 0,
                    referenceEma: 0,
                    policyLoss: 0
                },
                newTraining: {
                    algorithm: "",
                    model: "",
                    dataset: "",
                    name: ""
                },
                models: [],
                datasets: []
            });
            this.getView().setModel(oTrainingModel, "training");
            this._oTrainingModel = oTrainingModel;

            // Chart instances (initialized after view renders)
            this._charts = {
                winRate: null,
                trainingLoss: null,
                sankey: null
            };

            // Load initial data
            this._loadAlgorithms();
            this._loadJobs();
            this._loadExperiments();
            this._loadKTOMetrics();
        },

        onAfterRendering: function () {
            // Initialize custom charts after DOM is ready
            var that = this;
            setTimeout(function () {
                that._initializeCharts();
            }, 500);
        },

        _initializeCharts: function () {
            var that = this;

            // Dynamically import the Charts module
            var chartScriptPath = sap.ui.require.toUrl("llm/server/dashboard") + "/components-dist/Charts/Charts.js";

            import(chartScriptPath).then(function (ChartsModule) {
                that._ChartsModule = ChartsModule;
                that._createWinRateChart();
                that._createTrainingLossChart();
                that._createSankeyChart();
            }).catch(function (error) {
                console.warn("Charts module not loaded, using fallback:", error);
                // Fallback: Charts will display data via ObjectNumber controls
            });
        },

        _createWinRateChart: function () {
            var container = document.getElementById("winRateChart");
            if (!container || !this._ChartsModule) return;

            var winRate = this._oTrainingModel.getProperty("/ktoMetrics/winRate") || 0;

            this._charts.winRate = new this._ChartsModule.RadialChart(container, {
                value: winRate,
                maxValue: 100,
                minValue: 0,
                label: "Win Rate",
                unit: "%",
                width: 150,
                height: 150,
                arcWidth: 15,
                thresholds: [
                    { value: 45, color: "#bb0000" },
                    { value: 60, color: "#e9730c" },
                    { value: 100, color: "#107e3e" }
                ]
            });
        },

        _createTrainingLossChart: function () {
            var container = document.getElementById("trainingLossChart");
            if (!container || !this._ChartsModule) return;

            this._charts.trainingLoss = new this._ChartsModule.LineChart(container, {
                series: this._oTrainingModel.getProperty("/trainingLossSeries") || [],
                xAxisLabel: "Training Steps",
                yAxisLabel: "Loss",
                showGrid: true,
                showLegend: true,
                showTooltip: true,
                width: container.offsetWidth || 800,
                height: 300,
                margin: { top: 20, right: 30, bottom: 50, left: 60 }
            });
        },

        _createSankeyChart: function () {
            var container = document.getElementById("sankeyChart");
            if (!container || !this._ChartsModule) return;

            var sankey = this._oTrainingModel.getProperty("/sankey") || { nodes: [], links: [] };

            this._charts.sankey = new this._ChartsModule.SankeyDiagram(container, {
                nodes: sankey.nodes || [],
                links: sankey.links || [],
                width: container.offsetWidth || 800,
                height: 350,
                nodeWidth: 24,
                nodePadding: 15,
                showLabels: true,
                showValues: true,
                margin: { top: 20, right: 150, bottom: 20, left: 150 }
            });
        },

        _loadKTOMetrics: function () {
            var that = this;
            var oModel = this._oTrainingModel;
            fetch(this._getApiBaseUrl() + "/v1/training/kto")
                .then(function (response) { return response.ok ? response.json() : Promise.reject(response.statusText); })
                .then(function (data) {
                    oModel.setProperty("/ktoMetrics", data.metrics || {});
                    oModel.setProperty("/trainingLossSeries", data.training_loss || []);
                    oModel.setProperty("/sankey", data.sankey || { nodes: [], links: [] });
                })
                .catch(function (error) {
                    console.error("Failed to load KTO metrics:", error);
                    oModel.setProperty("/ktoMetrics", {});
                    oModel.setProperty("/trainingLossSeries", []);
                    oModel.setProperty("/sankey", { nodes: [], links: [] });
                    sap.m.MessageToast.show("Training metrics unavailable");
                });
        },

        // ==================== API Base URL ====================

        _getApiBaseUrl: function () {
            var oComponent = this.getOwnerComponent();
            return oComponent ? oComponent.getApiBaseUrl() : "http://localhost:11434";
        },

        // ==================== API Loading Methods ====================

        _loadAlgorithms: function () {
            var that = this;
            fetch(this._getApiBaseUrl() + "/v1/training/algorithms")
                .then(function (response) { return response.json(); })
                .then(function (data) {
                    that._oTrainingModel.setProperty("/algorithms", data.algorithms || []);
                })
                .catch(function (error) {
                    console.error("Error loading algorithms:", error);
                });
        },

        _loadJobs: function () {
            var that = this;
            fetch(this._getApiBaseUrl() + "/v1/training/jobs")
                .then(function (response) { return response.json(); })
                .then(function (data) {
                    that._oTrainingModel.setProperty("/jobs", data.jobs || []);
                })
                .catch(function (error) {
                    console.error("Error loading jobs:", error);
                });
        },

        _loadExperiments: function () {
            var that = this;
            fetch(this._getApiBaseUrl() + "/v1/training/experiments")
                .then(function (response) {
                    if (!response.ok) {
                        // Fallback to HANA endpoint
                        return fetch(that._getApiBaseUrl() + "/api/v1/hana/training-experiments");
                    }
                    return response;
                })
                .then(function (response) { return response.json(); })
                .then(function (data) {
                    that._oTrainingModel.setProperty("/experiments", data.experiments || []);
                })
                .catch(function (error) {
                    console.error("Error loading experiments:", error);
                });
        },

        _loadDatasets: function () {
            var that = this;
            fetch(this._getApiBaseUrl() + "/v1/training/datasets")
                .then(function (response) { return response.json(); })
                .then(function (data) {
                    that._oTrainingModel.setProperty("/datasets", data.datasets || []);
                })
                .catch(function (error) {
                    console.error("Error loading datasets:", error);
                });
        },

        _loadKTOMetrics: function (experimentId) {
            var that = this;
            fetch(this._getApiBaseUrl() + "/v1/training/experiments/" + experimentId + "/metrics")
                .then(function (response) { return response.json(); })
                .then(function (data) {
                    that._updateKTOMetrics(data.metrics || {});
                })
                .catch(function (error) {
                    console.error("Error loading KTO metrics:", error);
                });
        },

        // ==================== Event Handlers ====================

        onRefreshAll: function () {
            MessageToast.show("Refreshing training data...");
            this._loadAlgorithms();
            this._loadJobs();
            this._loadExperiments();
            this._loadDatasets();
        },

        onRefresh: function () {
            this.onRefreshAll();
        },

        onBreadcrumbHome: function () {
            // Navigate back to main page
            var oNavContainer = this.getView().getParent();
            if (oNavContainer && oNavContainer.to) {
                var oMainPage = oNavContainer.getParent().byId("mainPageContent");
                if (oMainPage) {
                    oNavContainer.to(oMainPage);
                }
            }
        },

        onStartTraining: function () {
            var that = this;
            var oNewTraining = this._oTrainingModel.getProperty("/newTraining");

            if (!oNewTraining.algorithm || !oNewTraining.model || !oNewTraining.dataset) {
                MessageBox.error("Please fill in all required fields.");
                return;
            }

            fetch(this._getApiBaseUrl() + "/v1/training/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    algorithm: oNewTraining.algorithm,
                    model_id: oNewTraining.model,
                    dataset_id: oNewTraining.dataset,
                    name: oNewTraining.name
                })
            })
            .then(function (response) {
                if (!response.ok) throw new Error("Failed to start training");
                return response.json();
            })
            .then(function (data) {
                MessageBox.success("Training job started: " + data.job_id);
                that._loadJobs();
            })
            .catch(function (error) {
                MessageBox.error("Failed to start training: " + error.message);
            });
        },

        onStopJob: function (oEvent) {
            var that = this;
            var oSource = oEvent.getSource();
            var oContext = oSource.getBindingContext("training");
            var sJobId = oContext.getProperty("id");

            MessageBox.confirm("Stop training job " + sJobId + "?", {
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        fetch(that._getApiBaseUrl() + "/v1/training/jobs/" + sJobId + "/stop", {
                            method: "POST"
                        })
                        .then(function (response) {
                            if (!response.ok) throw new Error("Failed to stop job");
                            MessageToast.show("Job stopped");
                            that._loadJobs();
                        })
                        .catch(function (error) {
                            MessageBox.error("Failed to stop job: " + error.message);
                        });
                    }
                }
            });
        },

        onPauseJob: function (oEvent) {
            var that = this;
            var oSource = oEvent.getSource();
            var oContext = oSource.getBindingContext("training");
            var sJobId = oContext.getProperty("id");

            fetch(this._getApiBaseUrl() + "/v1/training/jobs/" + sJobId + "/pause", {
                method: "POST"
            })
            .then(function (response) {
                if (!response.ok) throw new Error("Failed to pause job");
                MessageToast.show("Job paused");
                that._loadJobs();
            })
            .catch(function (error) {
                MessageBox.error("Failed to pause job: " + error.message);
            });
        },

        onViewMetrics: function (oEvent) {
            var oSource = oEvent.getSource();
            var oContext = oSource.getBindingContext("training");
            var sExperimentId = oContext.getProperty("id");

            // Load KTO metrics for the experiment
            this._loadKTOMetrics(sExperimentId);

            // Navigate to metrics view or open dialog
            MessageToast.show("Loading metrics for experiment: " + sExperimentId);
        },

        onCompareExperiments: function () {
            // Open T-Account comparison dialog
            MessageToast.show("Opening experiment comparison...");
            // TODO: Implement comparison dialog
        },

        onAlgorithmSelect: function (oEvent) {
            var oSource = oEvent.getSource();
            var sSelectedKey = oSource.getSelectedKey();
            this._oTrainingModel.setProperty("/newTraining/algorithm", sSelectedKey);

            // Update algorithm description if available
            var aAlgorithms = this._oTrainingModel.getProperty("/algorithms");
            var oSelectedAlgorithm = aAlgorithms.find(function (alg) {
                return alg.id === sSelectedKey;
            });

            if (oSelectedAlgorithm) {
                MessageToast.show("Selected: " + oSelectedAlgorithm.name);
            }
        },

        // ==================== KTO-specific Methods ====================

        _updateKTOMetrics: function (metrics) {
            var oKTOMetrics = {
                winRate: this._calculateWinRate(metrics.desirableCount, metrics.undesirableCount),
                klDivergence: metrics.klDivergence || 0,
                desirableCount: metrics.desirableCount || 0,
                undesirableCount: metrics.undesirableCount || 0,
                referenceEma: metrics.referenceEma || 0,
                policyLoss: metrics.policyLoss || 0
            };
            this._oTrainingModel.setProperty("/ktoMetrics", oKTOMetrics);
        },

        _calculateWinRate: function (desirable, undesirable) {
            var total = (desirable || 0) + (undesirable || 0);
            if (total === 0) return 0;
            return Math.round((desirable / total) * 100);
        },

        _getKLDivergenceState: function (value) {
            if (value < 15) return "Success";
            if (value < 20) return "Warning";
            return "Error";
        },

        // ==================== Helper Methods ====================

        _formatDuration: function (startTime, endTime) {
            if (!startTime) return "N/A";

            var start = new Date(startTime);
            var end = endTime ? new Date(endTime) : new Date();
            var diffMs = end - start;

            var hours = Math.floor(diffMs / (1000 * 60 * 60));
            var minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

            if (hours > 0) {
                return hours + "h " + minutes + "m";
            }
            return minutes + "m";
        },

        _getStatusState: function (status) {
            switch (status) {
                case "RUNNING":
                    return "Information";
                case "COMPLETED":
                    return "Success";
                case "FAILED":
                    return "Error";
                case "PAUSED":
                    return "Warning";
                case "PENDING":
                    return "None";
                default:
                    return "None";
            }
        }

    });
});
