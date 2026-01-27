sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast"
], function (Controller, MessageToast) {
    "use strict";

    return Controller.extend("trialbalance.controller.RawData", {

        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
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