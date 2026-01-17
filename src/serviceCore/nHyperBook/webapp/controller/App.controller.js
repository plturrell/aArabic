sap.ui.define([
    "sap/ui/core/mvc/Controller"
], function (Controller) {
    "use strict";

    return Controller.extend("hypershimmy.controller.App", {
        
        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            // Get the FlexibleColumnLayout control
            this.oFlexibleColumnLayout = this.byId("layout");
        },

        /**
         * Handler for layout state change
         * @param {sap.ui.base.Event} oEvent the state change event
         */
        onStateChanged: function (oEvent) {
            var bIsNavigationArrow = oEvent.getParameter("isNavigationArrow");
            var sLayout = oEvent.getParameter("layout");

            // Update app state model with new layout
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            oAppStateModel.setProperty("/currentLayout", sLayout);

            // If navigation arrow was clicked, handle navigation
            if (bIsNavigationArrow) {
                this._handleNavigationArrow(sLayout);
            }
        },

        /**
         * Handles navigation when layout arrows are clicked
         * @param {string} sLayout the new layout
         * @private
         */
        _handleNavigationArrow: function (sLayout) {
            var oRouter = this.getOwnerComponent().getRouter();
            
            // Navigate back when collapsing columns
            if (sLayout === "OneColumn") {
                oRouter.navTo("main");
            }
        }
    });
});
