sap.ui.define([
    "sap/ui/core/mvc/Controller"
], function (Controller) {
    "use strict";

    return Controller.extend("trial.balance.controller.App", {
        onInit: function () {
            // Apply content density class
            this.getView().addStyleClass(this.getOwnerComponent().getContentDensityClass());
        }
    });
});