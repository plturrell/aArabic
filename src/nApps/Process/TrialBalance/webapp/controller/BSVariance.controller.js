/**
 * ============================================================================
 * Balance Sheet Variance Controller
 * Material variance analysis with DOI threshold validation
 * ============================================================================
 *
 * [CODE:file=BSVariance.controller.js]
 * [CODE:module=controller]
 * [CODE:language=javascript]
 *
 * [ODPS:product=variances]
 * [ODPS:rules=VAR001,VAR002,VAR003,VAR004,VAR005,VAR006,VAR007,VAR008]
 *
 * [DOI:controls=MKR-CHK-001,MKR-CHK-002]
 * [DOI:thresholds=REQ-THRESH-001,REQ-THRESH-002,REQ-THRESH-003,REQ-THRESH-004]
 *
 * [VIEW:binding=BSVariance.view.xml]
 *
 * [API:consumes=/api/v1/variances]
 * [API:consumes=/api/v1/variances/material]
 * [API:consumes=/api/v1/variances/coverage]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:displays=DATA:variances]
 * [RELATION:calls=CODE:balance_engine.zig#VarianceAnalysis]
 *
 * This controller handles variance analysis display, commentary management,
 * and materiality threshold application per DOI requirements.
 */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "trialbalance/service/ApiService"
], function (Controller, JSONModel, MessageToast, MessageBox, ApiService) {
    "use strict";

    // DOI Thresholds - Must match backend balance_engine.zig
    var DOIThresholds = ApiService.DOIThresholds;
    var DriverCategory = ApiService.DriverCategory;

    return Controller.extend("trialbalance.controller.BSVariance", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Initialize API service
            this._oApiService = new ApiService();
            
            // Create view model
            var oViewModel = new JSONModel({
                busy: true,
                variances: [],
                materialVariances: [],
                summary: {
                    totalVariances: 0,
                    materialCount: 0,
                    withCommentary: 0,
                    exceptionsCount: 0,
                    coveragePercent: 0,
                    meetsCoverage: false
                },
                thresholds: {
                    balanceSheetAmount: DOIThresholds.BALANCE_SHEET_AMOUNT,
                    profitLossAmount: DOIThresholds.PROFIT_LOSS_AMOUNT,
                    variancePercent: DOIThresholds.VARIANCE_PERCENTAGE * 100,
                    commentaryCoverage: DOIThresholds.COMMENTARY_COVERAGE * 100
                },
                driverCategories: Object.keys(DriverCategory).map(function(key) {
                    return { key: key, text: DriverCategory[key].replace(/_/g, " ") };
                }),
                selectedVariance: null,
                lastUpdated: null
            });
            this.getView().setModel(oViewModel, "view");
            
            // Load data when route is matched
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("bsVariance").attachPatternMatched(this._onRouteMatched, this);
        },

        /**
         * Route matched handler
         * @private
         */
        _onRouteMatched: function () {
            this._loadVarianceData();
        },

        /**
         * Load variance data from API
         * @private
         */
        _loadVarianceData: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);
            
            // Load all variances
            Promise.all([
                this._oApiService.getVariances({}),
                this._oApiService.getMaterialVariances({}),
                this._oApiService.getCommentaryCoverage()
            ]).then(function (aResults) {
                var aVariances = aResults[0] || [];
                var aMaterialVariances = aResults[1] || [];
                var oCoverage = aResults[2] || {};
                
                // Transform data for display
                var aTransformed = aVariances.map(function (v) {
                    return {
                        id: v.account_id,
                        accountId: v.account_id,
                        accountType: v.account_type || "Unknown",
                        currentBalance: v.current_balance || 0,
                        previousBalance: v.previous_balance || 0,
                        varianceAmount: v.variance_absolute || 0,
                        variancePercent: v.variance_percentage || 0,
                        isMaterial: v.exceeds_threshold || false,
                        hasCommentary: v.has_commentary || false,
                        commentary: v.commentary || "",
                        majorDriver: v.major_driver || "",
                        driverCategory: v.driver_category || null,
                        isException: v.is_exception || false,
                        thresholdAmount: v.threshold_amount || 0,
                        // UI state
                        state: that._getVarianceState(v)
                    };
                });
                
                oViewModel.setProperty("/variances", aTransformed);
                oViewModel.setProperty("/materialVariances", aTransformed.filter(function(v) {
                    return v.isMaterial;
                }));
                
                // Update summary
                var iWithCommentary = aTransformed.filter(function(v) {
                    return v.isMaterial && v.hasCommentary;
                }).length;
                var iExceptions = aTransformed.filter(function(v) {
                    return v.isException;
                }).length;
                var iMaterial = aTransformed.filter(function(v) {
                    return v.isMaterial;
                }).length;
                
                oViewModel.setProperty("/summary", {
                    totalVariances: aTransformed.length,
                    materialCount: iMaterial,
                    withCommentary: iWithCommentary,
                    exceptionsCount: iExceptions,
                    coveragePercent: oCoverage.coverage || (iMaterial > 0 ? (iWithCommentary / iMaterial * 100) : 100),
                    meetsCoverage: oCoverage.meets_coverage || (iMaterial > 0 ? (iWithCommentary / iMaterial >= 0.9) : true)
                });
                
                oViewModel.setProperty("/lastUpdated", new Date());
                oViewModel.setProperty("/busy", false);
            }).catch(function (oError) {
                MessageToast.show("Failed to load variance data: " + oError.message);
                oViewModel.setProperty("/busy", false);
            });
        },

        /**
         * Get variance state for UI
         * @private
         */
        _getVarianceState: function (oVariance) {
            if (oVariance.is_exception) return "Error";
            if (oVariance.exceeds_threshold && oVariance.has_commentary) return "Warning";
            if (oVariance.exceeds_threshold) return "Error";
            return "None";
        },

        /**
         * Format currency amount
         */
        formatCurrency: function (fValue) {
            if (fValue === null || fValue === undefined) return "";
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(fValue);
        },

        /**
         * Format variance percentage
         */
        formatVariancePercent: function (fValue) {
            if (fValue === null || fValue === undefined) return "0%";
            var sSign = fValue >= 0 ? "+" : "";
            return sSign + fValue.toFixed(1) + "%";
        },

        /**
         * Get icon for variance row
         */
        getVarianceIcon: function (bIsMaterial, bHasCommentary, bIsException) {
            if (bIsException) return "sap-icon://alert";
            if (bIsMaterial && bHasCommentary) return "sap-icon://message-success";
            if (bIsMaterial) return "sap-icon://message-warning";
            return "";
        },

        /**
         * Get highlight state for variance row
         */
        getRowHighlight: function (bIsMaterial, bHasCommentary, bIsException) {
            if (bIsException) return "Error";
            if (bIsMaterial && !bHasCommentary) return "Warning";
            if (bIsMaterial) return "Success";
            return "None";
        },

        /**
         * Handle variance selection
         */
        onVarianceSelect: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext("view");
            var oVariance = oContext.getObject();
            
            this.getView().getModel("view").setProperty("/selectedVariance", oVariance);
        },

        /**
         * Open commentary dialog
         */
        onAddCommentary: function (oEvent) {
            var oSource = oEvent.getSource();
            var oContext = oSource.getBindingContext("view");
            var oVariance = oContext.getObject();
            
            this.getView().getModel("view").setProperty("/selectedVariance", oVariance);
            
            // Open dialog (would need to create fragment)
            MessageBox.information("Commentary dialog for account: " + oVariance.accountId);
        },

        /**
         * Save commentary
         */
        onSaveCommentary: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            var oVariance = oViewModel.getProperty("/selectedVariance");
            
            if (!oVariance) return;
            
            this._oApiService.updateVarianceCommentary(oVariance.id, {
                commentary: oVariance.commentary,
                major_driver: oVariance.majorDriver,
                driver_category: oVariance.driverCategory
            }).then(function () {
                MessageToast.show("Commentary saved");
                that._loadVarianceData();
            }).catch(function (oError) {
                MessageBox.error("Failed to save commentary: " + oError.message);
            });
        },

        /**
         * Filter material variances only
         */
        onFilterMaterial: function () {
            var oTable = this.byId("varianceTable");
            var oBinding = oTable.getBinding("items");
            oBinding.filter([
                new sap.ui.model.Filter("isMaterial", sap.ui.model.FilterOperator.EQ, true)
            ]);
        },

        /**
         * Filter exceptions only
         */
        onFilterExceptions: function () {
            var oTable = this.byId("varianceTable");
            var oBinding = oTable.getBinding("items");
            oBinding.filter([
                new sap.ui.model.Filter("isException", sap.ui.model.FilterOperator.EQ, true)
            ]);
        },

        /**
         * Clear all filters
         */
        onClearFilters: function () {
            var oTable = this.byId("varianceTable");
            var oBinding = oTable.getBinding("items");
            oBinding.filter([]);
        },

        /**
         * Refresh data
         */
        onRefresh: function () {
            this._loadVarianceData();
            MessageToast.show("Data refreshed");
        },

        /**
         * Navigate back
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        }

    });
});