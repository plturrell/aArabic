sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function (Controller, JSONModel, MessageBox, MessageToast) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.ABTesting", {

        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("abTesting").attachPatternMatched(this._onObjectMatched, this);

            // Initialize A/B testing model
            this._oABTestModel = new JSONModel(this._getDefaultData());
            this.getView().setModel(this._oABTestModel, "abtest");

            // Load available models
            this._loadAvailableModels();
            
            // Load comparison history
            this._loadComparisonHistory();

            // Initialize example prompts
            this._initExamples();
        },

        _getDefaultData: function () {
            return {
                availableModels: [],
                modelA: { id: "", display_name: "", quantization: "" },
                modelB: { id: "", display_name: "", quantization: "" },
                testPrompt: "",
                isLoading: false,
                responseA: { text: "", latency_ms: 0, tokens_per_second: 0, cost_estimate: 0 },
                responseB: { text: "", latency_ms: 0, tokens_per_second: 0, cost_estimate: 0 },
                ratingA: "",
                ratingB: "",
                metricsComparison: [],
                comparisonHistory: [],
                aggregateStats: {
                    totalComparisons: 0,
                    modelAWins: 0,
                    modelBWins: 0,
                    ties: 0,
                    modelAWinPercent: 0,
                    modelBWinPercent: 0,
                    tiePercent: 0
                }
            };
        },

        _initExamples: function () {
            this._examples = {
                "Simple Math": "What is 2+2? Explain your answer step by step.",
                "Code Generation": "Write a Python function to calculate the fibonacci sequence using dynamic programming.",
                "Arabic Text": "ما هي عاصمة السعودية؟ وما هي أهم المعالم السياحية فيها؟",
                "Reasoning": "Explain the concept of recursion with a practical example. How does it differ from iteration?"
            };
        },

        _onObjectMatched: function () {
            this._loadComparisonHistory();
            this._updateAggregateStats();
        },

        _loadAvailableModels: function () {
            var that = this;
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent.getApiBaseUrl();

            fetch(sApiBaseUrl + "/v1/models")
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to load models");
                    return response.json();
                })
                .then(function (data) {
                    var aModels = data.data || data.models || [];
                    that._oABTestModel.setProperty("/availableModels", aModels);
                })
                .catch(function (error) {
                    console.error("Error loading models:", error);
                    that._oABTestModel.setProperty("/availableModels", []);
                    MessageBox.error("Unable to load models. Please verify the API.");
                });
        },

        onModelAChange: function (oEvent) {
            var sKey = oEvent.getParameter("selectedItem").getKey();
            var aModels = this._oABTestModel.getProperty("/availableModels");
            var oModel = aModels.find(function (m) { return m.id === sKey; });
            if (oModel) {
                this._oABTestModel.setProperty("/modelA", oModel);
            }
        },

        onModelBChange: function (oEvent) {
            var sKey = oEvent.getParameter("selectedItem").getKey();
            var aModels = this._oABTestModel.getProperty("/availableModels");
            var oModel = aModels.find(function (m) { return m.id === sKey; });
            if (oModel) {
                this._oABTestModel.setProperty("/modelB", oModel);
            }
        },

        onExamplePrompt: function (oEvent) {
            var sButtonText = oEvent.getSource().getText();
            var sExample = this._examples[sButtonText];
            if (sExample) {
                this._oABTestModel.setProperty("/testPrompt", sExample);
            }
        },

        onRunABTest: function () {
            var oData = this._oABTestModel.getData();
            
            if (!oData.modelA.id || !oData.modelB.id) {
                MessageBox.error("Please select both Model A and Model B");
                return;
            }
            
            if (!oData.testPrompt) {
                MessageBox.error("Please enter a test prompt");
                return;
            }

            this._oABTestModel.setProperty("/isLoading", true);
            this._oABTestModel.setProperty("/responseA", { text: "", latency_ms: 0, tokens_per_second: 0, cost_estimate: 0 });
            this._oABTestModel.setProperty("/responseB", { text: "", latency_ms: 0, tokens_per_second: 0, cost_estimate: 0 });
            this._oABTestModel.setProperty("/ratingA", "");
            this._oABTestModel.setProperty("/ratingB", "");

            var that = this;
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent.getApiBaseUrl();

            // Run both requests in parallel
            Promise.all([
                this._runModelTest(sApiBaseUrl, oData.modelA.id, oData.testPrompt),
                this._runModelTest(sApiBaseUrl, oData.modelB.id, oData.testPrompt)
            ])
            .then(function (aResults) {
                that._oABTestModel.setProperty("/responseA", aResults[0]);
                that._oABTestModel.setProperty("/responseB", aResults[1]);
                that._updateMetricsComparison(aResults[0], aResults[1]);
                MessageToast.show("A/B Test completed successfully");
            })
            .catch(function (error) {
                console.error("A/B Test error:", error);
                MessageBox.error("A/B Test failed: " + error.message);
            })
            .finally(function () {
                that._oABTestModel.setProperty("/isLoading", false);
            });
        },

        _runModelTest: function (sApiBaseUrl, sModelId, sPrompt) {
            var nStartTime = Date.now();

            return fetch(sApiBaseUrl + "/v1/chat/completions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: sModelId,
                    messages: [{ role: "user", content: sPrompt }],
                    max_tokens: 512,
                    temperature: 0.7,
                    stream: false
                })
            })
            .then(function (response) {
                if (!response.ok) throw new Error("API request failed for " + sModelId);
                return response.json();
            })
            .then(function (data) {
                var nEndTime = Date.now();
                var nLatency = nEndTime - nStartTime;
                var oChoice = data.choices && data.choices[0];
                var nTokens = data.usage?.completion_tokens || 0;
                var nTps = nTokens > 0 && nLatency > 0 ? Math.round(nTokens / (nLatency / 1000)) : 0;

                return {
                    text: oChoice ? oChoice.message.content : "No response generated",
                    latency_ms: nLatency,
                    tokens_per_second: nTps,
                    tokens_generated: nTokens,
                    cost_estimate: (nTokens / 1000 * 0.001 + nLatency / 1000 * 0.0001).toFixed(6)
                };
            })
            .catch(function (error) {
                return Promise.reject(error);
            });
        },

        _updateMetricsComparison: function (oResponseA, oResponseB) {
            var aMetrics = [
                {
                    metric: "Latency",
                    valueA: oResponseA.latency_ms,
                    valueB: oResponseB.latency_ms,
                    unit: "ms",
                    winner: oResponseA.latency_ms < oResponseB.latency_ms ? "A" :
                            oResponseB.latency_ms < oResponseA.latency_ms ? "B" : "tie"
                },
                {
                    metric: "Tokens/sec",
                    valueA: oResponseA.tokens_per_second,
                    valueB: oResponseB.tokens_per_second,
                    unit: "tok/s",
                    winner: oResponseA.tokens_per_second > oResponseB.tokens_per_second ? "A" :
                            oResponseB.tokens_per_second > oResponseA.tokens_per_second ? "B" : "tie"
                },
                {
                    metric: "Tokens Generated",
                    valueA: oResponseA.tokens_generated || 0,
                    valueB: oResponseB.tokens_generated || 0,
                    unit: "tokens",
                    winner: "tie"
                },
                {
                    metric: "Cost Estimate",
                    valueA: parseFloat(oResponseA.cost_estimate) * 1000,
                    valueB: parseFloat(oResponseB.cost_estimate) * 1000,
                    unit: "m$",
                    winner: parseFloat(oResponseA.cost_estimate) < parseFloat(oResponseB.cost_estimate) ? "A" :
                            parseFloat(oResponseB.cost_estimate) < parseFloat(oResponseA.cost_estimate) ? "B" : "tie"
                }
            ];

            this._oABTestModel.setProperty("/metricsComparison", aMetrics);
        },

        onRateModelA: function (oEvent) {
            var sCurrentRating = this._oABTestModel.getProperty("/ratingA");
            var sButtonText = oEvent.getSource().getText().toLowerCase();

            // Toggle rating if clicking the same button
            if (sCurrentRating === sButtonText) {
                this._oABTestModel.setProperty("/ratingA", "");
            } else {
                this._oABTestModel.setProperty("/ratingA", sButtonText);
            }
        },

        onRateModelB: function (oEvent) {
            var sCurrentRating = this._oABTestModel.getProperty("/ratingB");
            var sButtonText = oEvent.getSource().getText().toLowerCase();

            if (sCurrentRating === sButtonText) {
                this._oABTestModel.setProperty("/ratingB", "");
            } else {
                this._oABTestModel.setProperty("/ratingB", sButtonText);
            }
        },

        onSaveComparison: function () {
            var oData = this._oABTestModel.getData();

            // Determine winner based on ratings and metrics
            var sWinner = this._determineWinner(oData);

            var oComparison = {
                id: this._generateUUID(),
                timestamp: new Date().toISOString(),
                prompt: oData.testPrompt,
                modelA: {
                    id: oData.modelA.id,
                    display_name: oData.modelA.display_name,
                    latency_ms: oData.responseA.latency_ms,
                    tps: oData.responseA.tokens_per_second,
                    rating: oData.ratingA
                },
                modelB: {
                    id: oData.modelB.id,
                    display_name: oData.modelB.display_name,
                    latency_ms: oData.responseB.latency_ms,
                    tps: oData.responseB.tokens_per_second,
                    rating: oData.ratingB
                },
                winner: sWinner
            };

            // Add to history
            var aHistory = this._oABTestModel.getProperty("/comparisonHistory");
            aHistory.unshift(oComparison);
            this._oABTestModel.setProperty("/comparisonHistory", aHistory);

            // Update aggregate stats
            this._updateAggregateStats();

            // Try to save to API
            this._saveComparisonToAPI(oComparison);

            MessageToast.show("Comparison saved successfully");
        },

        _determineWinner: function (oData) {
            var nScoreA = 0;
            var nScoreB = 0;

            // Rating scores (most important)
            if (oData.ratingA === "good") nScoreA += 3;
            if (oData.ratingA === "poor") nScoreA -= 3;
            if (oData.ratingB === "good") nScoreB += 3;
            if (oData.ratingB === "poor") nScoreB -= 3;

            // Latency scores
            if (oData.responseA.latency_ms < oData.responseB.latency_ms) nScoreA += 1;
            else if (oData.responseB.latency_ms < oData.responseA.latency_ms) nScoreB += 1;

            // TPS scores
            if (oData.responseA.tokens_per_second > oData.responseB.tokens_per_second) nScoreA += 1;
            else if (oData.responseB.tokens_per_second > oData.responseA.tokens_per_second) nScoreB += 1;

            if (nScoreA > nScoreB) return "A";
            if (nScoreB > nScoreA) return "B";
            return "tie";
        },

        _updateAggregateStats: function () {
            var aHistory = this._oABTestModel.getProperty("/comparisonHistory");
            var nTotal = aHistory.length;
            var nAWins = 0, nBWins = 0, nTies = 0;

            aHistory.forEach(function (comp) {
                if (comp.winner === "A") nAWins++;
                else if (comp.winner === "B") nBWins++;
                else nTies++;
            });

            this._oABTestModel.setProperty("/aggregateStats", {
                totalComparisons: nTotal,
                modelAWins: nAWins,
                modelBWins: nBWins,
                ties: nTies,
                modelAWinPercent: nTotal > 0 ? Math.round(nAWins / nTotal * 100) : 0,
                modelBWinPercent: nTotal > 0 ? Math.round(nBWins / nTotal * 100) : 0,
                tiePercent: nTotal > 0 ? Math.round(nTies / nTotal * 100) : 0
            });
        },

        _saveComparisonToAPI: function (oComparison) {
            fetch("/api/v1/ab-testing/comparisons", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(oComparison)
            })
            .then(function (response) {
                if (!response.ok) throw new Error("Failed to save comparison");
                console.log("Comparison saved to API");
            })
            .catch(function (error) {
                console.warn("Could not save to API (saved locally):", error);
            });
        },

        _loadComparisonHistory: function () {
            var that = this;

            fetch("/api/v1/ab-testing/comparisons?limit=50")
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to load history");
                    return response.json();
                })
                .then(function (data) {
                    that._oABTestModel.setProperty("/comparisonHistory", data.comparisons || []);
                    that._updateAggregateStats();
                })
                .catch(function (error) {
                    console.warn("Could not load history from API:", error);
                    that._oABTestModel.setProperty("/comparisonHistory", []);
                    that._updateAggregateStats();
                    MessageBox.error("A/B comparison history is unavailable.");
                });
        },

        onSearchHistory: function (oEvent) {
            var sQuery = oEvent.getParameter("query");
            var oTable = this.byId("comparisonHistoryTable");
            var aFilters = [];

            if (sQuery) {
                aFilters.push(new sap.ui.model.Filter("prompt", sap.ui.model.FilterOperator.Contains, sQuery));
            }

            oTable.getBinding("items").filter(aFilters);
        },

        onRefreshHistory: function () {
            this._loadComparisonHistory();
            MessageToast.show("History refreshed");
        },

        onExportHistory: function () {
            var aHistory = this._oABTestModel.getProperty("/comparisonHistory");

            var sCsv = "Timestamp,Prompt,Model A,Model A Latency,Model B,Model B Latency,Winner\n";
            aHistory.forEach(function (entry) {
                sCsv += [
                    entry.timestamp,
                    '"' + (entry.prompt || "").replace(/"/g, '""') + '"',
                    entry.modelA.display_name,
                    entry.modelA.latency_ms,
                    entry.modelB.display_name,
                    entry.modelB.latency_ms,
                    entry.winner
                ].join(",") + "\n";
            });

            var oBlob = new Blob([sCsv], { type: "text/csv" });
            var sUrl = URL.createObjectURL(oBlob);
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = "ab-testing-history-" + new Date().toISOString().split('T')[0] + ".csv";
            oLink.click();
            URL.revokeObjectURL(sUrl);

            MessageToast.show("History exported to CSV");
        },

        onHistorySelect: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext("abtest");
            var oEntry = oContext.getObject();

            // Load the prompt from history
            this._oABTestModel.setProperty("/testPrompt", oEntry.prompt);
            MessageToast.show("Loaded prompt from history");
        },

        onViewComparisonDetails: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("abtest");
            var oEntry = oContext.getObject();

            MessageBox.information(
                "Comparison Details\n\n" +
                "Prompt: " + oEntry.prompt.substring(0, 100) + "...\n\n" +
                "Model A: " + oEntry.modelA.display_name + "\n" +
                "  - Latency: " + oEntry.modelA.latency_ms + " ms\n" +
                "  - TPS: " + oEntry.modelA.tps + "\n\n" +
                "Model B: " + oEntry.modelB.display_name + "\n" +
                "  - Latency: " + oEntry.modelB.latency_ms + " ms\n" +
                "  - TPS: " + oEntry.modelB.tps + "\n\n" +
                "Winner: " + (oEntry.winner === "A" ? "Model A" : oEntry.winner === "B" ? "Model B" : "Tie"),
                { title: "Comparison Details" }
            );
        },

        onResetTest: function () {
            this._oABTestModel.setProperty("/testPrompt", "");
            this._oABTestModel.setProperty("/responseA", { text: "", latency_ms: 0, tokens_per_second: 0, cost_estimate: 0 });
            this._oABTestModel.setProperty("/responseB", { text: "", latency_ms: 0, tokens_per_second: 0, cost_estimate: 0 });
            this._oABTestModel.setProperty("/ratingA", "");
            this._oABTestModel.setProperty("/ratingB", "");
            this._oABTestModel.setProperty("/metricsComparison", []);
            MessageToast.show("Test reset");
        },

        formatTimestamp: function (sTimestamp) {
            if (!sTimestamp) return "";
            var oDate = new Date(sTimestamp);
            return oDate.toLocaleString();
        },

        _generateUUID: function () {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
                var r = Math.random() * 16 | 0;
                var v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        },

        onBreadcrumbHome: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("main");
        },

        onNavBack: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("main");
        }
    });
});
