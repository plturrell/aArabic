sap.ui.define([
    "nAudioLab/webapp/controller/BaseController",
    "sap/ui/model/json/JSONModel",
    "nAudioLab/webapp/service/ServiceWrapper"
], function (BaseController, JSONModel, ServiceWrapper) {
    "use strict";

    return BaseController.extend("nAudioLab.webapp.controller.Master", {
        onInit: function () {
            this.setModel(new JSONModel({ projects: [], busy: true }), "audio");
            this._loadProjects();
        },

        _loadProjects: function() {
            var oModel = this.getModel("audio");
            oModel.setProperty("/busy", true);
            ServiceWrapper.getProjects()
                .then(data => oModel.setProperty("/projects", data))
                .finally(() => oModel.setProperty("/busy", false));
        },

        onSelectionChange: function (oEvent) {
             var oItem = oEvent.getParameter("listItem") || oEvent.getSource();
            var sId = oItem.getBindingContext("audio").getProperty("id");
            this.getRouter().navTo("detail", { projectId: sId });
        }
    });
});