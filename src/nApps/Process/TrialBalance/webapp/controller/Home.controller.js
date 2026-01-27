/**
 * ============================================================================
 * Trial Balance Home Controller
 * Main navigation hub for the Trial Balance application
 * ============================================================================
 *
 * [CODE:file=Home.controller.js]
 * [CODE:module=controller]
 * [CODE:language=javascript]
 *
 * [VIEW:binding=Home.view.xml]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:navigates_to=Overview.view.xml]
 * [RELATION:navigates_to=YTDAnalysis.view.xml]
 * [RELATION:navigates_to=RawData.view.xml]
 * [RELATION:navigates_to=BSVariance.view.xml]
 * [RELATION:navigates_to=Checklist.view.xml]
 * [RELATION:navigates_to=Metadata.view.xml]
 * [RELATION:navigates_to=ODPSCatalog.view.xml]
 * [RELATION:navigates_to=QualityDashboard.view.xml]
 * [RELATION:navigates_to=LineageGraph.view.xml]
 *
 * This controller provides navigation and summary display for the Trial Balance
 * application dashboard.
 */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "trialbalance/service/ApiService"
], function (Controller, MessageToast, JSONModel, ApiService) {
    "use strict";

    return Controller.extend("trialbalance.controller.Home", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Initialize API service
            this._oApiService = new ApiService();
            
            // Create view model for dashboard
            var oViewModel = new JSONModel({
                dashboardSummary: {
                    trialBalanceStatus: "Loading...",
                    qualityScore: 0,
                    varianceCount: 0,
                    pendingItems: 0,
                    lastUpdated: null
                },
                notifications: []
            });
            this.getView().setModel(oViewModel, "view");
            
            // Load dashboard summary
            this._loadDashboardSummary();
        },

        /**
         * Load dashboard summary data from API
         * @private
         */
        _loadDashboardSummary: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            
            // Load quality metrics
            this._oApiService.getQualityMetrics()
                .then(function (oData) {
                    oViewModel.setProperty("/dashboardSummary/qualityScore", oData.overallScore || 0);
                    oViewModel.setProperty("/dashboardSummary/trialBalanceStatus", 
                        oData.overallScore >= 90 ? "Healthy" : "Attention Needed");
                })
                .catch(function (oError) {
                    oViewModel.setProperty("/dashboardSummary/trialBalanceStatus", "Offline");
                });
            
            // Load variance summary
            this._oApiService.getMaterialVariances({})
                .then(function (aData) {
                    oViewModel.setProperty("/dashboardSummary/varianceCount", aData.length || 0);
                })
                .catch(function () {
                    oViewModel.setProperty("/dashboardSummary/varianceCount", 0);
                });
            
            // Load checklist status
            this._oApiService.getChecklistItems()
                .then(function (aData) {
                    var iPending = aData.filter(function (item) {
                        return item.status !== "completed";
                    }).length;
                    oViewModel.setProperty("/dashboardSummary/pendingItems", iPending);
                })
                .catch(function () {
                    oViewModel.setProperty("/dashboardSummary/pendingItems", 0);
                });
            
            oViewModel.setProperty("/dashboardSummary/lastUpdated", new Date());
        },

        /**
         * Navigate to Overview page
         */
        onNavigateToOverview: function () {
            this.getOwnerComponent().getRouter().navTo("overview");
        },

        /**
         * Navigate to YTD Analysis page
         */
        onNavigateToYTD: function () {
            this.getOwnerComponent().getRouter().navTo("ytdAnalysis");
        },

        /**
         * Navigate to Raw Data page
         */
        onNavigateToRawData: function () {
            this.getOwnerComponent().getRouter().navTo("rawData");
        },

        /**
         * Navigate to Balance Sheet Variance page
         */
        onNavigateToBSVariance: function () {
            this.getOwnerComponent().getRouter().navTo("bsVariance");
        },

        /**
         * Navigate to Checklist (Maker/Checker) page
         */
        onNavigateToChecklist: function () {
            this.getOwnerComponent().getRouter().navTo("checklist");
        },

        /**
         * Navigate to Metadata page
         */
        onNavigateToMetadata: function () {
            this.getOwnerComponent().getRouter().navTo("metadata");
        },

        /**
         * Navigate to ODPS Catalog page
         */
        onODPSCatalog: function () {
            this.getOwnerComponent().getRouter().navTo("odpsCatalog");
        },

        /**
         * Navigate to Quality Dashboard page
         */
        onQualityDashboard: function () {
            this.getOwnerComponent().getRouter().navTo("qualityDashboard");
        },

        /**
         * Navigate to Lineage Graph page
         */
        onLineageGraph: function () {
            this.getOwnerComponent().getRouter().navTo("lineageGraph");
        },

        /**
         * Handle search event
         * @param {sap.ui.base.Event} oEvent - Search event
         */
        onSearch: function (oEvent) {
            var sQuery = oEvent.getParameter("query");
            if (sQuery) {
                MessageToast.show("Searching for: " + sQuery);
                // TODO: Implement global search
            }
        },

        /**
         * Handle notifications press
         */
        onNotificationsPress: function () {
            var oViewModel = this.getView().getModel("view");
            var iPending = oViewModel.getProperty("/dashboardSummary/pendingItems");
            MessageToast.show(iPending + " pending checklist items");
        },

        /**
         * Handle avatar press (user profile)
         */
        onAvatarPress: function () {
            MessageToast.show("User Profile");
        },

        /**
         * Refresh dashboard data
         */
        onRefresh: function () {
            this._loadDashboardSummary();
            MessageToast.show("Dashboard refreshed");
        }

    });
});