sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/model/json/JSONModel"
], function (UIComponent, JSONModel) {
    "use strict";

    return UIComponent.extend("galaxy.sim.Component", {
        metadata: {
            manifest: "json",
            config: {
                // Disable component preload in development mode
                async: true
            }
        },

        init: function () {
            // Call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // Set the device model
            this.setModel(new JSONModel({
                isPhone: sap.ui.Device.system.phone,
                isTablet: sap.ui.Device.system.tablet,
                isDesktop: sap.ui.Device.system.desktop
            }), "device");

            // Create the views based on the url/hash
            this.getRouter().initialize();
        }
    });
});
