/**
 * ============================================================================
 * Trial Balance Overview Controller
 * Executive summary and KPI dashboard
 * ============================================================================
 *
 * [CODE:file=Overview.controller.js]
 * [CODE:module=controller]
 * [CODE:language=javascript]
 *
 * [ODPS:product=trial-balance-aggregated]
 * [ODPS:rules=TB001,TB002,TB003,TB004,TB005,TB006]
 *
 * [VIEW:binding=Overview.view.xml]
 *
 * [API:consumes=/api/v1/trial-balance]
 * [API:consumes=/api/v1/quality]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:displays=DATA:trial-balance-aggregated]
 *
 * This controller provides the executive overview displaying trial balance
 * status, validation results, and key performance indicators.
 */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "trialbalance/service/ApiService"
], function (Controller, JSONModel, MessageToast, ApiService) {
    "use strict";

    return Controller.extend("trialbalance.controller.Overview", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Initialize API service
            this._oApiService = new ApiService();
            
            // Create view model
            var oViewModel = new JSONModel({
                busy: true,
                trialBalance: {
                    totalDebits: 0,
                    totalCredits: 0,
                    balanceDifference: 0,
                    isBalanced: false,
                    accountCount: 0
                },
                validationResults: {
                    tb001_passed: false,
                    tb002_passed: false,
                    tb003_passed: false,
                    tb004_passed: false,
                    tb005_passed: false,
                    tb006_passed: false
                },
                dataQuality: {
                    overallScore: 0,
                    completenessScore: 0,
                    accuracyScore: 0,
                    consistencyScore: 0,
                    timelinessScore: 0
                },
                lastUpdated: null
            });
            this.getView().setModel(oViewModel, "view");
            
            // Load data when route is matched
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("overview").attachPatternMatched(this._onRouteMatched, this);
        },

        /**
         * Route matched handler
         * @private
         */
        _onRouteMatched: function () {
            this._loadTrialBalanceData();
        },

        /**
         * Load trial balance data from API
         * @private
         */
        _loadTrialBalanceData: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);
            
            // Default parameters
            var oParams = {
                company_code: "1000",
                fiscal_year: "2025",
                period: "001"
            };
            
            // Load trial balance
            this._oApiService.getTrialBalance(oParams)
                .then(function (oData) {
                    oViewModel.setProperty("/trialBalance", {
                        totalDebits: oData.total_debits || 0,
                        totalCredits: oData.total_credits || 0,
                        balanceDifference: oData.balance_difference || 0,
                        isBalanced: oData.is_balanced || false,
                        accountCount: oData.account_count || 0
                    });
                    
                    // Update validation results
                    oViewModel.setProperty("/validationResults", {
                        tb001_passed: oData.tb001_passed || false,
                        tb002_passed: oData.tb002_passed || false,
                        tb003_passed: oData.tb003_passed || false,
                        tb004_passed: oData.tb004_passed || false,
                        tb005_passed: oData.tb005_passed || false,
                        tb006_passed: oData.tb006_passed || false
                    });
                })
                .catch(function (oError) {
                    MessageToast.show("Failed to load trial balance: " + oError.message);
                });
            
            // Load quality metrics
            this._oApiService.getQualityMetrics()
                .then(function (oData) {
                    oViewModel.setProperty("/dataQuality", {
                        overallScore: oData.overall_score || 0,
                        completenessScore: oData.completeness_score || 0,
                        accuracyScore: oData.accuracy_score || 0,
                        consistencyScore: oData.consistency_score || 0,
                        timelinessScore: oData.timeliness_score || 0
                    });
                })
                .catch(function (oError) {
                    // Non-critical - don't show error toast
                });
            
            oViewModel.setProperty("/lastUpdated", new Date());
            oViewModel.setProperty("/busy", false);
        },

        /**
         * Format currency amount
         * @param {number} fValue - Amount to format
         * @returns {string} Formatted currency string
         */
        formatCurrency: function (fValue) {
            if (fValue === null || fValue === undefined) return "";
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 2
            }).format(fValue);
        },

        /**
         * Format percentage
         * @param {number} fValue - Value to format (0-100)
         * @returns {string} Formatted percentage string
         */
        formatPercent: function (fValue) {
            if (fValue === null || fValue === undefined) return "0%";
            return fValue.toFixed(1) + "%";
        },

        /**
         * Get validation status state
         * @param {boolean} bPassed - Whether validation passed
         * @returns {string} State string for UI
         */
        getValidationState: function (bPassed) {
            return bPassed ? "Success" : "Error";
        },

        /**
         * Get validation status icon
         * @param {boolean} bPassed - Whether validation passed
         * @returns {string} Icon URI
         */
        getValidationIcon: function (bPassed) {
            return bPassed ? "sap-icon://accept" : "sap-icon://error";
        },

        /**
         * Refresh data
         */
        onRefresh: function () {
            this._loadTrialBalanceData();
            MessageToast.show("Data refreshed");
        },

        /**
         * Navigate back to home
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        }

    });
});