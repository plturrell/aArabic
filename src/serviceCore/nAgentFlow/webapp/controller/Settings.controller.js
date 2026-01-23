sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/m/MessageToast"
], function (Controller, History, MessageToast) {
    "use strict";

    return Controller.extend("serviceCore.nWorkflow.controller.Settings", {
        onInit: function () {
            // Load settings from component model
            var oComponent = this.getOwnerComponent();
            var oModel = oComponent.getModel();
            if (oModel && !oModel.getProperty("/settings")) {
                oModel.setProperty("/settings", {
                    theme: "sap_horizon",
                    autoSave: true,
                    gridSize: 20
                });
            }
        },

        onSavePress: function () {
            var oBundle = this.getView().getModel("i18n").getResourceBundle();
            MessageToast.show(oBundle.getText("settingsSaved"));
        },

        onNavBack: function () {
            var oHistory = History.getInstance();
            var sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getOwnerComponent().getRouter().navTo("dashboard", {}, true);
            }
        },

        onThemeChange: function (oEvent) {
            var sTheme = oEvent.getParameter("selectedItem").getKey();
            sap.ui.getCore().applyTheme(sTheme);
        }
    });
});

