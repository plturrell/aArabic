sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast"
], function (Controller, MessageToast) {
    "use strict";

    return Controller.extend("trialbalance.controller.Home", {

        onNavigateToOverview: function () {
            this.getOwnerComponent().getRouter().navTo("overview");
        },

        onNavigateToYTD: function () {
            this.getOwnerComponent().getRouter().navTo("ytdAnalysis");
        },

        onNavigateToRawData: function () {
            this.getOwnerComponent().getRouter().navTo("rawData");
        },

        onNavigateToBSVariance: function () {
            this.getOwnerComponent().getRouter().navTo("bsVariance");
        },

        onNavigateToChecklist: function () {
            this.getOwnerComponent().getRouter().navTo("checklist");
        },

        onNavigateToMetadata: function () {
            this.getOwnerComponent().getRouter().navTo("metadata");
        },

        onODPSCatalog: function () {
            this.getOwnerComponent().getRouter().navTo("odpsCatalog");
        },

        onQualityDashboard: function () {
            this.getOwnerComponent().getRouter().navTo("qualityDashboard");
        },

        onLineageGraph: function () {
            this.getOwnerComponent().getRouter().navTo("lineageGraph");
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