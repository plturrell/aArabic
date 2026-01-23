sap.ui.define([
    "sap/ui/core/mvc/Controller"
], function (Controller) {
    "use strict";

    return Controller.extend("serviceCore.nWorkflow.controller.NotFound", {
        onNavToHome: function () {
            this.getOwnerComponent().getRouter().navTo("dashboard", {}, true);
        }
    });
});

