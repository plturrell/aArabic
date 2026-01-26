sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History"
], function (Controller, History) {
    "use strict";

    return Controller.extend("trial.balance.controller.Home", {
        onInit: function () {
            // Initialize
        },

        onNavigateToTrialBalance: function () {
            this.getOwnerComponent().getRouter().navTo("trialBalance");
        },

        onNavigateToReconciliation: function () {
            this.getOwnerComponent().getRouter().navTo("reconciliation");
        },

        onNavigateToApproval: function () {
            this.getOwnerComponent().getRouter().navTo("approval");
        },

        onNavigateToAnalytics: function () {
            this.getOwnerComponent().getRouter().navTo("analytics");
        },

        onUserMenuPress: function (oEvent) {
            // TODO: Implement user menu
            var oButton = oEvent.getSource();
            
            // Show user menu popover
            if (!this._oUserMenu) {
                this._oUserMenu = sap.ui.xmlfragment(
                    "trial.balance.view.fragments.UserMenu",
                    this
                );
                this.getView().addDependent(this._oUserMenu);
            }
            
            this._oUserMenu.openBy(oButton);
        }
    });
});