sap.ui.define([
    "nCode/webapp/controller/BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "nCode/webapp/model/formatter",
    "nCode/webapp/service/ServiceWrapper"
], function (BaseController, JSONModel, Filter, FilterOperator, formatter, ServiceWrapper) {
    "use strict";

    return BaseController.extend("nCode.webapp.controller.Master", {
        formatter: formatter,

        onInit: function () {
            this.setModel(new JSONModel({ items: [], busy: true }), "files");
            this._loadData();
        },

        _loadData: function() {
            var oModel = this.getModel("files");
            oModel.setProperty("/busy", true);
            
            ServiceWrapper.getFiles()
                .then(data => {
                    oModel.setProperty("/items", data);
                })
                .finally(() => {
                    oModel.setProperty("/busy", false);
                });
        },

        onSearch: function (oEvent) {
            var aFilters = [];
            var sQuery = oEvent.getSource().getValue();
            if (sQuery && sQuery.length > 0) {
                var filter = new Filter("name", FilterOperator.Contains, sQuery);
                aFilters.push(filter);
            }
            var oList = this.byId("fileList");
            var oBinding = oList.getBinding("items");
            oBinding.filter(aFilters, "Application");
        },

        onSelectionChange: function (oEvent) {
            var oItem = oEvent.getParameter("listItem") || oEvent.getSource();
            var sId = oItem.getBindingContext("files").getProperty("id");
            this.getRouter().navTo("detail", { fileId: sId });
        }
    });
});
