sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/UIComponent",
    "sap/m/library"
], function (Controller, UIComponent, mobileLibrary) {
    "use strict";

    return Controller.extend("nCode.webapp.controller.BaseController", {
        getRouter: function () {
            return UIComponent.getRouterFor(this);
        },

        getModel: function (sName) {
            return this.getView().getModel(sName);
        },

        setModel: function (oModel, sName) {
            return this.getView().setModel(oModel, sName);
        },

        getResourceBundle: function () {
            return this.getOwnerComponent().getModel("i18n").getResourceBundle();
        },
        
        onStateChanged: function (oEvent) {
            var bIsNavigationArrow = oEvent.getParameter("isNavigationArrow"),
                sLayout = oEvent.getParameter("layout");
            this.updateUIElements();
            this.getOwnerComponent().getModel("appView").setProperty("/layout", sLayout);
        },

        onPressHome: function () {
            // Redirect to Launchpad (assuming port 8091 or relative path)
            window.location.href = "http://localhost:8091"; 
        },

        updateUIElements: function () {
            var oModel = this.getOwnerComponent().getModel("appView");
            var sLayout = oModel.getProperty("/layout");
            // Hook for subclasses
        }
    });
});
