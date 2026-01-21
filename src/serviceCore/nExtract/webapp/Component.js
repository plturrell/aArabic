sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "nExtract/webapp/model/models"
], function (UIComponent, Device, models) {
    "use strict";

    return UIComponent.extend("nExtract.webapp.Component", {
        metadata: {
            manifest: "json"
        },

        init: function () {
            UIComponent.prototype.init.apply(this, arguments);
            this.getRouter().initialize();
            this.setModel(models.createDeviceModel(), "device");
        }
    });
});
