/**
 * ============================================================================
 * Data Quality Dashboard Controller
 * ODPS data quality metrics and validation rule status
 * ============================================================================
 *
 * [CODE:file=QualityDashboard.controller.js]
 * [CODE:module=controller]
 * [CODE:language=javascript]
 *
 * [ODPS:product=trial-balance-aggregated]
 * [ODPS:product=variances]
 * [ODPS:product=exchange-rates]
 * [ODPS:rules=TB001,TB002,TB003,TB004,TB005,TB006,VAR001,VAR002,VAR003,VAR004,VAR005,VAR006,VAR007,VAR008,FX001,FX002,FX003,FX004,FX005,FX006,FX007]
 *
 * [VIEW:binding=QualityDashboard.view.xml]
 *
 * [API:consumes=/api/v1/quality]
 * [API:consumes=/api/v1/quality/rules]
 * [API:consumes=/api/v1/quality/trends]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:calls=CODE:odps_quality_service.zig]
 * [RELATION:calls=CODE:data_quality.zig]
 *
 * This controller displays data quality metrics, validation rule status,
 * and quality trends for the trial balance data.
 */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "trialbalance/service/ApiService"
], function (Controller, JSONModel, MessageToast, ApiService) {
    "use strict";

    // ODPS Rule IDs from backend
    var ODPSRuleID = ApiService.ODPSRuleID;

    return Controller.extend("trialbalance.controller.QualityDashboard", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Initialize API service
            this._oApiService = new ApiService();

            // Create view model
            var oViewModel = new JSONModel({
                busy: true,
                metrics: {
                    overallScore: 0,
                    completenessScore: 0,
                    accuracyScore: 0,
                    consistencyScore: 0,
                    timelinessScore: 0
                },
                targets: {
                    completeness: 95,
                    accuracy: 98,
                    consistency: 90,
                    timeliness: 95
                },
                ruleCategories: [
                    {
                        name: "Trial Balance Rules (TB001-TB006)",
                        rules: [
                            { id: "TB001", name: "Balance Equation", description: "Closing = Opening + Debits - Credits", passed: false },
                            { id: "TB002", name: "Debit Credit Balance", description: "Total debits = Total credits", passed: false },
                            { id: "TB003", name: "IFRS Classification", description: "All accounts have IFRS category", passed: false },
                            { id: "TB004", name: "Period Data Accuracy", description: "Period dates match expected", passed: false },
                            { id: "TB005", name: "GCOA Mapping", description: "All accounts mapped to GCOA", passed: false },
                            { id: "TB006", name: "Global Mapping Currency", description: "Mapping version current", passed: false }
                        ]
                    },
                    {
                        name: "Variance Rules (VAR001-VAR008)",
                        rules: [
                            { id: "VAR001", name: "Variance Calculation", description: "Variance = Current - Previous", passed: false },
                            { id: "VAR002", name: "Variance Percent", description: "Percentage calculated correctly", passed: false },
                            { id: "VAR003", name: "Materiality BS", description: "$100M AND 10% threshold", passed: false },
                            { id: "VAR004", name: "Materiality P&L", description: "$3M AND 10% threshold", passed: false },
                            { id: "VAR005", name: "Commentary Required", description: "Material variances have commentary", passed: false },
                            { id: "VAR006", name: "Commentary Coverage", description: "90% coverage requirement", passed: false },
                            { id: "VAR007", name: "Exception Flagging", description: "Exceptions properly flagged", passed: false },
                            { id: "VAR008", name: "Driver Identification", description: "Major drivers identified", passed: false }
                        ]
                    },
                    {
                        name: "Exchange Rate Rules (FX001-FX007)",
                        rules: [
                            { id: "FX001", name: "From Currency", description: "Source currency mandatory", passed: false },
                            { id: "FX002", name: "To Currency", description: "Target currency mandatory", passed: false },
                            { id: "FX003", name: "Rate Positive", description: "Exchange rate > 0", passed: false },
                            { id: "FX004", name: "Ratio Positive", description: "Currency ratios > 0", passed: false },
                            { id: "FX005", name: "Rate Verification", description: "Rates match Group rates", passed: false },
                            { id: "FX006", name: "Period Rate", description: "Period-appropriate rates", passed: false },
                            { id: "FX007", name: "Group Source", description: "Approved rate sources", passed: false }
                        ]
                    }
                ],
                trends: [],
                summary: {
                    totalRules: 21,
                    passedRules: 0,
                    failedRules: 0,
                    passRate: 0
                },
                lastUpdated: null
            });
            this.getView().setModel(oViewModel, "view");

            // Create quality model for view bindings
            var oQualityModel = new JSONModel({
                averageQuality: 92,
                highQualityCount: 4,
                lowQualityCount: 1,
                lastUpdateTime: "Today",
                products: [
                    { name: "Trial Balance", qualityScore: 95 },
                    { name: "Variances", qualityScore: 92 },
                    { name: "Exchange Rates", qualityScore: 98 },
                    { name: "Journal Entries", qualityScore: 88 },
                    { name: "Account Master", qualityScore: 94 }
                ]
            });
            this.getView().setModel(oQualityModel, "quality");

            // Create dimensions model for view bindings
            var oDimensionsModel = new JSONModel({
                dimensions: [
                    { product: "Trial Balance", completeness: 98, accuracy: 96, consistency: 94, timeliness: 92, overall: 95 },
                    { product: "Variances", completeness: 95, accuracy: 93, consistency: 90, timeliness: 88, overall: 92 },
                    { product: "Exchange Rates", completeness: 100, accuracy: 99, consistency: 98, timeliness: 95, overall: 98 },
                    { product: "Journal Entries", completeness: 92, accuracy: 88, consistency: 85, timeliness: 87, overall: 88 },
                    { product: "Account Master", completeness: 96, accuracy: 95, consistency: 93, timeliness: 92, overall: 94 }
                ]
            });
            this.getView().setModel(oDimensionsModel, "dimensions");

            // Create rules model for view bindings
            var oRulesModel = new JSONModel({
                rules: [
                    { ruleID: "TB001", name: "Balance Equation", description: "Closing = Opening + Debits - Credits", severity: "error", field: "closing_balance", product: "Trial Balance" },
                    { ruleID: "TB002", name: "Debit Credit Balance", description: "Total debits = Total credits", severity: "error", field: "debit_amount", product: "Trial Balance" },
                    { ruleID: "VAR001", name: "Variance Calculation", description: "Variance = Current - Previous", severity: "error", field: "variance", product: "Variances" },
                    { ruleID: "FX001", name: "From Currency", description: "Source currency mandatory", severity: "error", field: "from_currency", product: "Exchange Rates" }
                ]
            });
            this.getView().setModel(oRulesModel, "rules");

            // Load data when route is matched
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("qualityDashboard").attachPatternMatched(this._onRouteMatched, this);
        },

        /**
         * Route matched handler
         * @private
         */
        _onRouteMatched: function () {
            this._loadQualityData();
        },

        /**
         * Load quality data from API
         * @private
         */
        _loadQualityData: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);
            
            Promise.all([
                this._oApiService.getQualityMetrics(),
                this._oApiService.getRuleValidationStatus(),
                this._oApiService.getQualityTrends({ days: 30 })
            ]).then(function (aResults) {
                var oMetrics = aResults[0] || {};
                var aRuleStatus = aResults[1] || [];
                var aTrends = aResults[2] || [];
                
                // Update metrics
                oViewModel.setProperty("/metrics", {
                    overallScore: oMetrics.overall_score || 0,
                    completenessScore: oMetrics.completeness_score || 0,
                    accuracyScore: oMetrics.accuracy_score || 0,
                    consistencyScore: oMetrics.consistency_score || 0,
                    timelinessScore: oMetrics.timeliness_score || 0
                });
                
                // Update rule status
                var aCategories = oViewModel.getProperty("/ruleCategories");
                var iPassed = 0;
                var iFailed = 0;
                
                aCategories.forEach(function (oCategory) {
                    oCategory.rules.forEach(function (oRule) {
                        var oStatus = aRuleStatus.find(function (s) {
                            return s.rule_id === oRule.id;
                        });
                        if (oStatus) {
                            oRule.passed = oStatus.passed;
                            if (oStatus.passed) iPassed++;
                            else iFailed++;
                        }
                    });
                });
                
                oViewModel.setProperty("/ruleCategories", aCategories);
                oViewModel.setProperty("/summary", {
                    totalRules: 21,
                    passedRules: iPassed,
                    failedRules: iFailed,
                    passRate: iPassed / 21 * 100
                });
                
                // Update trends
                oViewModel.setProperty("/trends", aTrends.map(function (t) {
                    return {
                        date: new Date(t.date),
                        score: t.overall_score
                    };
                }));
                
                oViewModel.setProperty("/lastUpdated", new Date());
                oViewModel.setProperty("/busy", false);
            }).catch(function (oError) {
                MessageToast.show("Failed to load quality data: " + oError.message);
                oViewModel.setProperty("/busy", false);
            });
        },

        /**
         * Format score as percentage
         */
        formatScore: function (fValue) {
            if (fValue === null || fValue === undefined) return "0%";
            return fValue.toFixed(1) + "%";
        },

        /**
         * Get score state for radial micro chart
         */
        getScoreState: function (fValue, fTarget) {
            if (fValue >= fTarget) return "Good";
            if (fValue >= fTarget * 0.9) return "Critical";
            return "Error";
        },

        /**
         * Get rule status icon
         */
        getRuleIcon: function (bPassed) {
            return bPassed ? "sap-icon://accept" : "sap-icon://error";
        },

        /**
         * Get rule status state
         */
        getRuleState: function (bPassed) {
            return bPassed ? "Success" : "Error";
        },

        /**
         * Refresh data
         */
        onRefresh: function () {
            this._loadQualityData();
            MessageToast.show("Data refreshed");
        },

        /**
         * Export quality report
         */
        onExportReport: function () {
            MessageToast.show("Exporting quality report...");
        },

        /**
         * Handle tile press
         */
        onTilePress: function () {
            MessageToast.show("Quality tile pressed");
        },

        /**
         * Navigate to data catalog
         */
        onViewCatalog: function () {
            MessageToast.show("Navigating to Data Catalog...");
            this.getOwnerComponent().getRouter().navTo("odpsCatalog");
        },

        /**
         * Navigate to data lineage
         */
        onViewLineage: function () {
            MessageToast.show("Navigating to Data Lineage...");
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