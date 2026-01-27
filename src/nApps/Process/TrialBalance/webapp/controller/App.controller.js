sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (Controller, JSONModel, MessageToast) {
    "use strict";

    return Controller.extend("trialbalance.controller.App", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Create view model for app-level state
            var oViewModel = new JSONModel({
                sideExpanded: true
            });
            this.getView().setModel(oViewModel, "appView");

            // Get router for navigation
            this._oRouter = this.getOwnerComponent().getRouter();
        },

        /**
         * Toggle side navigation expanded/collapsed state
         */
        onSideNavButtonPress: function () {
            var oViewModel = this.getView().getModel("appView");
            var bExpanded = oViewModel.getProperty("/sideExpanded");
            oViewModel.setProperty("/sideExpanded", !bExpanded);
        },

        /**
         * Handle navigation item selection from sidebar
         * @param {sap.ui.base.Event} oEvent - Item select event
         */
        onItemSelect: function (oEvent) {
            var oItem = oEvent.getParameter("item");
            var sKey = oItem.getKey();

            // Skip parent navigation items that are just containers
            if (sKey === "analysis" || sKey === "settings") {
                return;
            }

            // Navigate to the selected route
            if (sKey && this._oRouter) {
                this._oRouter.navTo(sKey);
            }
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
         * Handle notifications button press
         */
        onNotificationsPress: function () {
            MessageToast.show("3 pending notifications");
        },

        /**
         * Handle help button press
         */
        onHelpPress: function () {
            MessageToast.show("Help documentation coming soon");
        },

        /**
         * Handle avatar/profile press
         */
        onAvatarPress: function () {
            MessageToast.show("User Profile");
        }
    });
});