sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast"
], function (Controller, MessageToast) {
    "use strict";

    return Controller.extend("trialbalance.controller.Overview", {

        onMenuPress: function () {
            const oActionSheet = this.byId("navigationMenu");
            oActionSheet.openBy(this.byId("overviewPage"));
        },

        onMenuItemPress: function (oEvent) {
            const sText = oEvent.getSource().getText();
            const oRouter = this.getOwnerComponent().getRouter();
            
            // Map button text to route names
            const routeMap = {
                "Home": "home",
                "Overview": "overview",
                "ODPS Catalog": "odpsCatalog",
                "Quality Dashboard": "qualityDashboard",
                "Data Lineage": "lineageGraph",
                "YTD Analysis": "ytdAnalysis",
                "Raw Data": "rawData",
                "BS Variance": "bsVariance",
                "Checklist": "checklist",
                "Metadata": "metadata"
            };
            
            const sRoute = routeMap[sText];
            if (sRoute) {
                oRouter.navTo(sRoute);
            }
        },

        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        },

        onKPIPress: function (oEvent) {
            var sHeader = oEvent.getSource().getHeader();
            MessageToast.show("KPI: " + sHeader);
        },

        onCalculate: function () {
            MessageToast.show("Calculating Trial Balance...");
            // TODO: Connect to backend API
        },

        onExport: function () {
            MessageToast.show("Exporting to Excel...");
        },

        onRefresh: function () {
            MessageToast.show("Refreshing data...");
        },

        onSearch: function (oEvent) {
            var sQuery = oEvent.getParameter("query");
            if (sQuery) {
                MessageToast.show("Searching for: " + sQuery);
            }
        },

        onNotificationsPress: function () {
            MessageToast.show("3 new notifications");
        },

        onAvatarPress: function () {
            MessageToast.show("User Profile");
        }

    });
});