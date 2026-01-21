sap.ui.define([
    "nExtract/webapp/controller/BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "nExtract/webapp/service/ServiceWrapper"
], function (BaseController, JSONModel, Filter, FilterOperator, ServiceWrapper) {
    "use strict";

    return BaseController.extend("nExtract.webapp.controller.Master", {
        onInit: function () {
            this.setModel(new JSONModel({ items: [], busy: true }), "history");
            this._loadHistory();
        },

        _loadHistory: function() {
            var oModel = this.getModel("history");
            oModel.setProperty("/busy", true);
            ServiceWrapper.getHistory()
                .then(data => oModel.setProperty("/items", data))
                .finally(() => oModel.setProperty("/busy", false));
        },

        onSearch: function (oEvent) {
            var sQuery = oEvent.getSource().getValue();
            var aFilters = [];
            if (sQuery && sQuery.length > 0) {
                aFilters.push(new Filter("name", FilterOperator.Contains, sQuery));
            }
            var oList = this.byId("historyList");
            oList.getBinding("items").filter(aFilters);
        },

        onSelectionChange: function (oEvent) {
            var oItem = oEvent.getParameter("listItem") || oEvent.getSource();
            var sId = oItem.getBindingContext("history").getProperty("id");
            this.getRouter().navTo("detail", { docId: sId });
        }
    });
});
