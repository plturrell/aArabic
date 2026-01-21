sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel"
], function (Controller, JSONModel) {
    "use strict";

    return Controller.extend("nCode.webapp.controller.App", {
        onInit: function () {
            this.getOwnerComponent().setModel(new JSONModel({ layout: "OneColumn" }), "appView");
        },
        onStateChanged: function (oEvent) {
            var bIsNavigationArrow = oEvent.getParameter("isNavigationArrow"),
                sLayout = oEvent.getParameter("layout");
            this.getOwnerComponent().getModel("appView").setProperty("/layout", sLayout);
        }
    });
});