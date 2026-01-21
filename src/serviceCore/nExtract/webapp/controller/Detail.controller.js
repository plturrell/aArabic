sap.ui.define([
    "nExtract/webapp/controller/BaseController",
    "sap/ui/model/json/JSONModel",
    "nExtract/webapp/service/ServiceWrapper"
], function (BaseController, JSONModel, ServiceWrapper) {
    "use strict";

    return BaseController.extend("nExtract.webapp.controller.Detail", {
        onInit: function () {
            var oRouter = this.getRouter();
            oRouter.getRoute("detail").attachPatternMatched(this._onObjectMatched, this);
            
            this.setModel(new JSONModel({ name: "", status: "Processed", busy: false }), "doc");
            this.setModel(new JSONModel({ data: [] }), "results");
        },

        _onObjectMatched: function (oEvent) {
            this.getOwnerComponent().getModel("appView").setProperty("/layout", "TwoColumnsMidExpanded");
            var sId = oEvent.getParameter("arguments").docId;
            this._loadDocument(sId);
        },

        _loadDocument: function(sId) {
            var oDocModel = this.getModel("doc");
            var oResModel = this.getModel("results");
            
            oDocModel.setProperty("/busy", true);
            
            ServiceWrapper.getDocument(sId)
                .then(data => {
                    oDocModel.setProperty("/name", data.meta.name);
                    oDocModel.setProperty("/status", data.meta.status);
                    oDocModel.setProperty("/id", data.meta.id);
                    oResModel.setProperty("/data", data.results);
                })
                .finally(() => oDocModel.setProperty("/busy", false));
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