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
            this._loadMockKTOMetrics();  // Load mock KTO data for demo
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
            var chartScriptPath = sap.ui.require.toUrl("llm/server/dashboard") + "/../components/dist/Charts/Charts.js";

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

            // Mock training loss data
            var mockData = this._generateMockTrainingData();

            this._charts.trainingLoss = new this._ChartsModule.LineChart(container, {
                series: mockData,
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

            // Training data pipeline flow
            var nodes = [
                { id: "raw", name: "Raw Dataset" },
                { id: "filtered", name: "Filtered" },
                { id: "tokenized", name: "Tokenized" },
                { id: "train", name: "Training Set" },
                { id: "val", name: "Validation Set" },
                { id: "kto", name: "KTO Training" },
                { id: "sft", name: "SFT Training" },
                { id: "eval", name: "Evaluation" }
            ];

            var links = [
                { source: "raw", target: "filtered", value: 100000 },
                { source: "filtered", target: "tokenized", value: 85000 },
                { source: "tokenized", target: "train", value: 68000 },
                { source: "tokenized", target: "val", value: 17000 },
                { source: "train", target: "kto", value: 40000 },
                { source: "train", target: "sft", value: 28000 },
                { source: "val", target: "eval", value: 17000 }
            ];

            this._charts.sankey = new this._ChartsModule.SankeyDiagram(container, {
                nodes: nodes,
                links: links,
                width: container.offsetWidth || 800,
                height: 350,
                nodeWidth: 24,
                nodePadding: 15,
                showLabels: true,
                showValues: true,
                margin: { top: 20, right: 150, bottom: 20, left: 150 }
            });
        },

        _generateMockTrainingData: function () {
            var policyLoss = [];
            var valueLoss = [];
            var klDivergence = [];

            for (var i = 0; i < 100; i++) {
                var step = i * 10;
                // Simulated decaying loss with noise
                policyLoss.push({ x: step, y: 2.5 * Math.exp(-i/30) + 0.1 + Math.random() * 0.1 });
                valueLoss.push({ x: step, y: 1.8 * Math.exp(-i/40) + 0.05 + Math.random() * 0.08 });
                klDivergence.push({ x: step, y: 5 + 10 * (1 - Math.exp(-i/50)) + Math.random() * 2 });
            }

            return [
                { name: "Policy Loss", data: policyLoss, color: "#0a6ed1" },
                { name: "Value Loss", data: valueLoss, color: "#107e3e" },
                { name: "KL Divergence", data: klDivergence, color: "#e9730c", dashed: true }
            ];
        },

        _loadMockKTOMetrics: function () {
            // Mock KTO metrics for demo
            this._oTrainingModel.setProperty("/ktoMetrics", {
                winRate: 67.5,
                klDivergence: 12.4,
                desirableCount: 15420,
                undesirableCount: 8230,
                ratio: "1.87",
                refEmaAlpha: 0.95,
                policyLoss: 0.342,
                policyLossTrend: "down",
                valueLoss: 0.128
            });

            // Update chart if it exists
            if (this._charts.winRate) {
                this._charts.winRate.setValue(67.5);
            }
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