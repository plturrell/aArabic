sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/UIComponent"
], function (Controller, UIComponent) {
    "use strict";

    return Controller.extend("nExtract.webapp.controller.BaseController", {
        getRouter: function () {
            return UIComponent.getRouterFor(this);
        },
        getModel: function (sName) {
            return this.getView().getModel(sName);
        },
        setModel: function (oModel, sName) {
            return this.getView().setModel(oModel, sName);
        },
        onPressHome: function () {
            window.location.href = "http://localhost:8091";
        }
    });
});
