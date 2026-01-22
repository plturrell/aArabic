sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "llm/server/dashboard/utils/ApiService",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/BusyIndicator"
], function (Controller, JSONModel, ApiService, MessageToast, MessageBox, BusyIndicator) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.MHCTuning", {
        onInit: function () {
            // Initialize MHC configuration model
            var oMHCModel = new JSONModel({
                enabled: false,
                selectedModel: "",
                availableModels: [],
                config: {
                    sinkhorn_iterations: 10,
                    manifold_epsilon: 1e-6,
                    stability_threshold: 1e-4,
                    manifold_beta: 10.0,
                    early_stopping: true,
                    log_stability_metrics: false
                },
                geometricConfig: {
                    manifold_type: "euclidean",
                    hyperbolic_curvature: -1.0,
                    use_poincare: true,
                    spherical_radius: 1.0
                },
                jobs: [],
                selectedJob: null,
                isTraining: false,
                trainingProgress: 0,
                trainingStatus: "",
                trainingMetrics: null,
                trainingResources: null,
                currentJobId: null,
                currentExperimentId: null,
                wizardStep: 1,
                trainingMode: 0,  // 0 = inference-only, 1 = fine-tune
                selectedDataset: "",
                selectedAlgorithm: "sft",
                algorithmDescription: "",
                selectedDatasetDescription: "",
                selectedDatasetUrl: "",
                selectedAlgorithmDescription: "",
                availableDatasets: [],
                availableAlgorithms: [],
                downloadStatus: "",
                downloadId: null,
                downloadProgress: 0
            });
            this.getView().setModel(oMHCModel, "mhc");
            this._oMHCModel = oMHCModel;

            // Load initial data
            this._loadModels();
            this._loadMHCConfig();
            this._loadMHCJobs();
            this._loadTrainingOptions();
        },

        onBreadcrumbHome: function () {
            var oComponent = this.getOwnerComponent();
            if (oComponent && oComponent.navigateTo) {
                oComponent.navigateTo("main");
            }
        },

        _loadModels: function () {
            var that = this;
            ApiService.getModels().then(function (oData) {
                var aModels = oData.data || oData.models || [];
                that.getView().getModel("mhc").setProperty("/availableModels", aModels);
                if (aModels.length > 0) {
                    that.getView().getModel("mhc").setProperty("/selectedModel", aModels[0].id);
                }
            }).catch(function (oError) {
                console.error("Failed to load models:", oError);
            });
        },

        _loadMHCConfig: function () {
            var that = this;
            var sModelId = this.getView().getModel("mhc").getProperty("/selectedModel");

            ApiService.getMHCConfig(sModelId).then(function (oData) {
                if (oData.enabled !== undefined) {
                    that.getView().getModel("mhc").setProperty("/enabled", oData.enabled);
                }
                if (oData.config) {
                    that.getView().getModel("mhc").setProperty("/config", oData.config);
                }
            }).catch(function (oError) {
                console.error("Failed to load MHC config:", oError);
            });
        },

        _loadMHCJobs: function () {
            var that = this;
            ApiService.getMHCJobs().then(function (oData) {
                var aJobs = oData.jobs || [];
                that.getView().getModel("mhc").setProperty("/jobs", aJobs);
            }).catch(function (oError) {
                console.error("Failed to load MHC jobs:", oError);
            });
        },

        onModelChange: function (oEvent) {
            this._loadMHCConfig();
        },

        onToggleMHC: function (oEvent) {
            var bEnabled = oEvent.getParameter("state");
            this.getView().getModel("mhc").setProperty("/enabled", bEnabled);
            MessageToast.show("mHC constraints " + (bEnabled ? "enabled" : "disabled"));
        },

        onSaveConfig: function () {
            var that = this;
            var oModel = this.getView().getModel("mhc");
            var sModelId = oModel.getProperty("/selectedModel");
            var oConfig = oModel.getProperty("/config");

            ApiService.updateMHCConfig(sModelId, oConfig).then(function () {
                MessageToast.show("mHC configuration saved successfully");
            }).catch(function (oError) {
                MessageBox.error("Failed to save configuration: " + oError.message);
            });
        },

        onStartTraining: function () {
            var that = this;
            var oModel = this.getView().getModel("mhc");
            var sModelId = oModel.getProperty("/selectedModel");
            var oConfig = oModel.getProperty("/config");
            var oGeometricConfig = oModel.getProperty("/geometricConfig");
            var sDatasetId = oModel.getProperty("/selectedDataset");
            var sAlgorithm = oModel.getProperty("/selectedAlgorithm");
            var iTrainingMode = oModel.getProperty("/trainingMode");

            var oTrainingRequest = {
                model_id: sModelId,
                dataset_id: sDatasetId,
                algorithm: sAlgorithm,
                training_mode: iTrainingMode,
                mhc_config: oConfig,
                geometric_config: oGeometricConfig
            };

            // Show busy indicator during submission
            BusyIndicator.show(0);
            oModel.setProperty("/isTraining", true);
            oModel.setProperty("/trainingProgress", 0);
            oModel.setProperty("/trainingStatus", "starting");

            fetch(this._getApiBaseUrl() + "/v1/training/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(oTrainingRequest)
            })
            .then(function(response) { return response.json(); })
            .then(function (oData) {
                BusyIndicator.hide();
                // Store the returned job_id and experiment_id
                oModel.setProperty("/currentJobId", oData.job_id);
                oModel.setProperty("/currentExperimentId", oData.experiment_id);
                oModel.setProperty("/trainingStatus", "running");

                MessageBox.success("Training job started!\n\nJob ID: " + oData.job_id +
                    "\nExperiment ID: " + oData.experiment_id +
                    "\n\nView metrics at: /api/v1/training/jobs/" + oData.job_id + "/metrics");

                that._loadMHCJobs();
                // Start polling for job status
                that._pollJobStatus(oData.job_id);
            }).catch(function (oError) {
                BusyIndicator.hide();
                oModel.setProperty("/isTraining", false);
                oModel.setProperty("/trainingStatus", "error");
                MessageBox.error("Failed to start training: " + oError.message);
            });
        },

        _pollTrainingProgress: function (sJobId) {
            var that = this;
            var oModel = this.getView().getModel("mhc");

            var fnPoll = function () {
                ApiService.getMHCJob(sJobId).then(function (oData) {
                    var nProgress = oData.progress || 0;
                    oModel.setProperty("/trainingProgress", nProgress);

                    if (oData.status === "completed") {
                        oModel.setProperty("/isTraining", false);
                        MessageToast.show("Training completed successfully!");
                        that._loadMHCJobs();
                    } else if (oData.status === "failed") {
                        oModel.setProperty("/isTraining", false);
                        MessageBox.error("Training failed: " + (oData.error || "Unknown error"));
                    } else if (oModel.getProperty("/isTraining")) {
                        setTimeout(fnPoll, 2000);
                    }
                }).catch(function () {
                    oModel.setProperty("/isTraining", false);
                });
            };

            fnPoll();
        },

        onDownloadDataset: function() {
            var oModel = this.getView().getModel("mhc");
            var sDatasetId = oModel.getProperty("/selectedDataset");

            if (!sDatasetId || sDatasetId === "none") {
                MessageBox.warning("Please select a dataset first");
                return;
            }

            oModel.setProperty("/downloadStatus", "downloading");

            fetch(this._getApiBaseUrl() + "/v1/training/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dataset_id: sDatasetId })
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                oModel.setProperty("/downloadId", data.download_id);
                oModel.setProperty("/downloadStatus", "started");
                MessageToast.show("Dataset download started: " + sDatasetId);
                // Start polling for download status
                this._pollDownloadStatus(data.download_id);
            }.bind(this))
            .catch(function(error) {
                oModel.setProperty("/downloadStatus", "error");
                MessageBox.error("Download failed: " + error.message);
            });
        },

        _pollDownloadStatus: function(downloadId) {
            var oModel = this.getView().getModel("mhc");
            var self = this;

            var poll = function() {
                fetch(self._getApiBaseUrl() + "/v1/training/download/" + downloadId)
                    .then(function(response) { return response.json(); })
                    .then(function(data) {
                        oModel.setProperty("/downloadProgress", data.progress);
                        if (data.status === "completed") {
                            oModel.setProperty("/downloadStatus", "completed");
                            MessageToast.show("Dataset ready for training!");
                        } else if (data.status === "failed") {
                            oModel.setProperty("/downloadStatus", "error");
                        } else {
                            setTimeout(poll, 2000);
                        }
                    });
            };
            poll();
        },

        _pollJobStatus: function(jobId) {
            var oModel = this.getView().getModel("mhc");
            var self = this;

            var poll = function() {
                fetch(self._getApiBaseUrl() + "/v1/training/jobs/" + jobId)
                    .then(function(response) { return response.json(); })
                    .then(function(data) {
                        oModel.setProperty("/trainingProgress", data.progress);
                        oModel.setProperty("/trainingMetrics", data.metrics);
                        oModel.setProperty("/trainingResources", data.resources);

                        if (data.status === "COMPLETED") {
                            oModel.setProperty("/trainingStatus", "completed");
                            oModel.setProperty("/isTraining", false);
                            MessageBox.success("Training completed! Model ready for deployment.");
                        } else if (data.status === "FAILED") {
                            oModel.setProperty("/trainingStatus", "failed");
                            oModel.setProperty("/isTraining", false);
                            MessageBox.error("Training failed. Check logs for details.");
                        } else {
                            setTimeout(poll, 5000);
                        }
                    });
            };
            poll();
        },

        onCancelTraining: function () {
            this.getView().getModel("mhc").setProperty("/isTraining", false);
            MessageToast.show("Training cancelled");
        },

        onRefreshJobs: function () {
            this._loadMHCJobs();
            MessageToast.show("Jobs refreshed");
        },

        onJobSelect: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext("mhc");
            var oJob = oContext.getObject();
            this.getView().getModel("mhc").setProperty("/selectedJob", oJob);
        },

        // Wizard handlers
        onWizardComplete: function () {
            MessageToast.show("Wizard completed - Ready to start training");
        },

        onModelSelectionStepActivate: function () {
            this.getView().getModel("mhc").setProperty("/wizardStep", 1);
        },

        onTrainingDataStepActivate: function () {
            this._oMHCModel.setProperty("/wizardStep", 2);
        },

        onMHCConfigStepActivate: function () {
            this._oMHCModel.setProperty("/wizardStep", 3);
        },

        onGeometricConfigStepActivate: function () {
            this._oMHCModel.setProperty("/wizardStep", 4);
        },

        onReviewStepActivate: function () {
            this._oMHCModel.setProperty("/wizardStep", 5);
        },

        onTrainingModeChange: function (oEvent) {
            var iSelectedIndex = oEvent.getParameter("selectedIndex");
            this._oMHCModel.setProperty("/trainingMode", iSelectedIndex);
        },

        onDatasetChange: function (oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var aDatasets = this._oMHCModel.getProperty("/availableDatasets") || [];
            var oDataset = aDatasets.find(function (d) { return d.id === sSelectedKey; });
            if (oDataset) {
                this._oMHCModel.setProperty("/selectedDataset", sSelectedKey);
                this._oMHCModel.setProperty("/selectedDatasetDescription", oDataset.description || "");
                this._oMHCModel.setProperty("/selectedDatasetUrl", oDataset.url || "");
            }
        },

        onAlgorithmChange: function (oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var aAlgorithms = this._oMHCModel.getProperty("/availableAlgorithms") || [];
            var oAlgorithm = aAlgorithms.find(function (a) { return a.id === sSelectedKey; });
            if (oAlgorithm) {
                this._oMHCModel.setProperty("/selectedAlgorithm", sSelectedKey);
                this._oMHCModel.setProperty("/selectedAlgorithmDescription", oAlgorithm.description || "");
            }
        },

        _loadTrainingOptions: function () {
            var that = this;
            Promise.all([
                fetch(this._getApiBaseUrl() + "/v1/training/datasets").then(function (r) { return r.json(); }),
                fetch(this._getApiBaseUrl() + "/v1/training/algorithms").then(function (r) { return r.json(); })
            ]).then(function (results) {
                var aDatasets = results[0].datasets || [];
                var aAlgorithms = results[1].algorithms || [];
                that._oMHCModel.setProperty("/availableDatasets", aDatasets);
                that._oMHCModel.setProperty("/availableAlgorithms", aAlgorithms);
                // Set default descriptions
                if (aAlgorithms.length > 0) {
                    that._oMHCModel.setProperty("/selectedAlgorithmDescription", aAlgorithms[0].description || "");
                }
            }).catch(function (err) {
                console.error("Error loading training options:", err);
            });
        },

        _getApiBaseUrl: function () {
            var oComponent = this.getOwnerComponent();
            return oComponent ? oComponent.getApiBaseUrl() : "http://localhost:8080/api";
        }
    });
});
