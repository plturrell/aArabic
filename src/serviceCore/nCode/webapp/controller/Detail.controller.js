sap.ui.define([
    "nCode/webapp/controller/BaseController",
    "sap/ui/model/json/JSONModel",
    "nCode/webapp/service/ServiceWrapper"
], function (BaseController, JSONModel, ServiceWrapper) {
    "use strict";

    return BaseController.extend("nCode.webapp.controller.Detail", {
        onInit: function () {
            var oRouter = this.getRouter();
            oRouter.getRoute("detail").attachPatternMatched(this._onObjectMatched, this);
            
            this.setModel(new JSONModel({
                code: "",
                currentFile: {},
                busy: false
            }), "editor");
        },

        _onObjectMatched: function (oEvent) {
            var sFileId = oEvent.getParameter("arguments").fileId;
            this.getOwnerComponent().getModel("appView").setProperty("/layout", "TwoColumnsMidExpanded");
            this._loadFile(sFileId);
        },

        _loadFile: function(sFileId) {
            var oModel = this.getModel("editor");
            oModel.setProperty("/busy", true);

            ServiceWrapper.getFile(sFileId)
                .then(oFile => {
                    oModel.setProperty("/currentFile", oFile);
                    oModel.setProperty("/code", oFile.content);
                    
                    // Try to fetch symbols if connected
                    if (oFile.path) {
                        ServiceWrapper.getSymbols(oFile.path).then(symbols => {
                            // Can populate a symbol model here if we had a view for it
                            console.log("Symbols loaded:", symbols);
                        });
                    }
                })
                .catch(err => {
                    console.error(err);
                    oModel.setProperty("/code", "// Error loading file");
                })
                .finally(() => {
                    oModel.setProperty("/busy", false);
                });
        },

        onFullScreen: function () {
             var oModel = this.getOwnerComponent().getModel("appView");
             var sLayout = oModel.getProperty("/layout");
             oModel.setProperty("/layout", sLayout === "MidColumnFullScreen" ? "TwoColumnsMidExpanded" : "MidColumnFullScreen");
        },

        onCloseDetail: function () {
            this.getOwnerComponent().getModel("appView").setProperty("/layout", "OneColumn");
            this.getRouter().navTo("master");
        }
    });
});