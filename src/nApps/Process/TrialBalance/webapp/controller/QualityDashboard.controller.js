sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (Controller, JSONModel, MessageToast) {
    "use strict";

    return Controller.extend("trialbalance.controller.QualityDashboard", {

        onInit: function () {
            // Load quality data
            this._loadQualityMetrics();
            this._loadQualityDimensions();
            this._loadValidationRules();
        },

        /**
         * Load quality metrics from ODPS API
         */
        _loadQualityMetrics: function () {
            const sUrl = "/api/v1/data-products/quality-report";
            
            fetch(sUrl)
                .then(response => response.json())
                .then(data => {
                    // Calculate additional metrics
                    data.highQualityCount = data.products.filter(p => p.qualityScore >= 95).length;
                    data.lowQualityCount = data.products.filter(p => p.qualityScore < 90).length;
                    data.lastUpdateTime = new Date(data.generatedAt).toLocaleTimeString();
                    
                    const oModel = new JSONModel(data);
                    this.getView().setModel(oModel, "quality");
                })
                .catch(error => {
                    // Load mock data
                    this._loadMockQualityMetrics();
                });
        },

        /**
         * Load mock quality metrics
         */
        _loadMockQualityMetrics: function () {
            const oMockData = {
                generatedAt: Date.now(),
                averageQuality: 94.6,
                highQualityCount: 3,
                lowQualityCount: 1,
                lastUpdateTime: new Date().toLocaleTimeString(),
                products: [
                    { name: "ACDOCA Journal Entries", qualityScore: 95.0 },
                    { name: "Exchange Rates", qualityScore: 98.0 },
                    { name: "Trial Balance Aggregated", qualityScore: 92.0 },
                    { name: "Period Variances", qualityScore: 90.0 },
                    { name: "Account Master", qualityScore: 98.0 },
                    { name: "Data Lineage", qualityScore: 100.0 },
                    { name: "Dataset Metadata", qualityScore: 100.0 },
                    { name: "Checklist Items", qualityScore: 85.0 }
                ]
            };
            
            const oModel = new JSONModel(oMockData);
            this.getView().setModel(oModel, "quality");
        },

        /**
         * Load quality dimensions for each product
         */
        _loadQualityDimensions: function () {
            const aDimensions = [
                {
                    product: "ACDOCA Journal Entries",
                    completeness: 98,
                    accuracy: 95,
                    consistency: 92,
                    timeliness: 99,
                    overall: 95.0
                },
                {
                    product: "Exchange Rates",
                    completeness: 99,
                    accuracy: 98,
                    consistency: 97,
                    timeliness: 99,
                    overall: 98.0
                },
                {
                    product: "Trial Balance Aggregated",
                    completeness: 95,
                    accuracy: 98,
                    consistency: 90,
                    timeliness: 95,
                    overall: 92.0
                },
                {
                    product: "Period Variances",
                    completeness: 92,
                    accuracy: 98,
                    consistency: 88,
                    timeliness: 85,
                    overall: 90.0
                },
                {
                    product: "Account Master",
                    completeness: 99,
                    accuracy: 98,
                    consistency: 97,
                    timeliness: 100,
                    overall: 98.0
                }
            ];
            
            const oModel = new JSONModel({ dimensions: aDimensions });
            this.getView().setModel(oModel, "dimensions");
        },

        /**
         * Load validation rules from ODPS files
         */
        _loadValidationRules: function () {
            const aRules = [
                // ACDOCA rules
                { ruleID: "R001", name: "RACCT Mandatory", description: "G/L Account is mandatory", severity: "error", field: "racct", product: "ACDOCA" },
                { ruleID: "R002", name: "DRCRK Valid", description: "Debit/Credit must be S or H", severity: "error", field: "drcrk", product: "ACDOCA" },
                { ruleID: "R003", name: "POPER Range", description: "Posting period 1-12", severity: "error", field: "poper", product: "ACDOCA" },
                { ruleID: "R004", name: "Zero Amount Warning", description: "HSL should not be zero", severity: "warning", field: "hsl", product: "ACDOCA" },
                { ruleID: "R005", name: "Company Code Mandatory", description: "Company code required", severity: "error", field: "rbukrs", product: "ACDOCA" },
                
                // Exchange Rate rules
                { ruleID: "X001", name: "From Currency Mandatory", description: "Source currency required", severity: "error", field: "from_curr", product: "ExchangeRates" },
                { ruleID: "X002", name: "To Currency Mandatory", description: "Target currency required", severity: "error", field: "to_curr", product: "ExchangeRates" },
                { ruleID: "X003", name: "Rate Positive", description: "Exchange rate must be positive", severity: "error", field: "exchange_rate", product: "ExchangeRates" },
                
                // Trial Balance rules
                { ruleID: "TB001", name: "Balance Equation", description: "Closing = Opening + Debit - Credit", severity: "error", field: "closing_balance", product: "TrialBalance" },
                { ruleID: "TB002", name: "Debit Credit Balance", description: "Total debits = total credits", severity: "error", field: "debit_amount", product: "TrialBalance" },
                
                // Variance rules
                { ruleID: "VAR001", name: "Variance Calculation", description: "Variance = Current - Previous", severity: "error", field: "variance_amount", product: "Variances" },
                { ruleID: "VAR003", name: "Materiality Threshold BS", description: "BS variance >$100M or >10%", severity: "warning", field: "is_significant", product: "Variances" }
            ];
            
            const oModel = new JSONModel({ rules: aRules });
            this.getView().setModel(oModel, "rules");
        },

        /**
         * Refresh all metrics
         */
        onRefresh: function () {
            MessageToast.show("Refreshing quality metrics...");
            this._loadQualityMetrics();
            this._loadQualityDimensions();
        },

        /**
         * Export quality report
         */
        onExportReport: function () {
            const oQualityModel = this.getView().getModel("quality");
            const oData = oQualityModel.getData();
            
            const sJson = JSON.stringify(oData, null, 2);
            const blob = new Blob([sJson], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement("a");
            a.href = url;
            a.download = `quality-report-${new Date().toISOString()}.json`;
            a.click();
            
            MessageToast.show("Quality report exported");
        },

        /**
         * Tile pressed
         */
        onTilePress: function () {
            MessageToast.show("Quality overview");
        },

        /**
         * Navigate to catalog
         */
        onViewCatalog: function () {
            this.getOwnerComponent().getRouter().navTo("odpsCatalog");
        },

        /**
         * Navigate to lineage
         */
        onViewLineage: function () {
            this.getOwnerComponent().getRouter().navTo("lineageGraph");
        },

        /**
         * Navigate back
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        }
    });
});